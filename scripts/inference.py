import os
import sys
import numpy as np
import torch
import trimesh
import torchvision
import torchvision.transforms as transforms
import pytorch3d
import nvdiffrast.torch as dr
from easydict import EasyDict
import yaml
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from torch.utils.data import DataLoader
from types import SimpleNamespace
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import tinycudann as tcnn
import commentjson as json
import pickle 

from scripts.utils import *
from scripts.models.model import Model
from scripts.utils.general_utils import load_mesh_path, calculate_face_centers, calculate_face_normals
from scripts.renderer.renderer import GTInitializer, AlphaRenderer
from scripts.losses.loss import Loss
from scripts.dataloader.Snap_data import MonocularDataset
from scripts.dataloader.Dress4D_data import MonocularDataset4DDress
from scripts.utils import *
from scripts.models import model
from scripts.models.template import Template
from scripts.models.core.NeuralJacobianFields import SourceMesh
from scripts.utils import smpl
from scripts.utils.general_utils import *
from scripts.utils.smpl_utils import *
from scripts.utils.cloth import Cloth
from scripts.utils.eval_utils import *
from scripts.models.remesh.remesh import *
from scripts.models.template import ClothModel
from scripts.models.temporal_deform import ClothDeform
from scripts.models.model_utils import get_linear_noise_func
from scripts.texture.texture import *
from scripts.models.core.siren.mlp import Siren
from scripts.utils.helper import *
from scripts.models.model_utils import get_expon_lr_func
from scripts.losses.rendering_loss import ssim
from scripts.models.model_utils import get_linear_interpolation_func
# from scripts.geometry import geometry_training_loop
# from scripts.appearnce import appearance_training_loop
# from scripts.inference import time_eval

with open("scripts/config_hash.json") as f:
	hash_cfg = json.load(f)

def Inference(cfg,  texture = False, device = 'cuda', mesh_inter = None):

    output_path = os.path.join(cfg.output_path, cfg.Exp_name)
    remesh_dir = os.path.join(output_path, 'remeshed_template')
    # body = smpl.SMPL(cfg.smpl_path).to(device)
    with open(cfg.template_smpl_pkl, 'rb') as f:
        body_data = pickle.load(f, encoding='latin1')
    templpate_smpl_shape = torch.tensor(body_data['shape']).to(device)
    body = smpl.SMPL(cfg.smpl_path).to(device)
    body.update_shape(shape = templpate_smpl_shape)
    jacobian_source = SourceMesh.SourceMesh(0, os.path.join(remesh_dir, f'source_mesh.obj'), {}, 1, ttype=torch.float)
    # jacobian_source = SourceMesh.SourceMesh(0, cfg.mesh, {}, 1, ttype=torch.float)

    jacobian_source.load()
    jacobian_source.to(device)
    sources_vertices,source_faces = load_mesh_path(os.path.join(remesh_dir, f'source_mesh.obj'), scale = -2, device = 'cuda')
    # sources_vertices,source_faces = load_mesh_path(cfg.mesh, scale = -2, device = 'cuda')
    tri_mesh_source = trimesh.Trimesh(vertices=sources_vertices.detach().cpu().numpy(), faces=source_faces.clone().cpu().numpy())
    tri_mesh_source.export(os.path.join(output_path,f'mesh_source.obj')) 
    source_faces = source_faces.to(torch.int64)

    if cfg.skinning_func == 'rbf':
        garment_skinning = compute_rbf_skinning_weight(sources_vertices,body)
    elif cfg.skinning_func == 'k-near':
        garment_skinning = compute_k_nearest_skinning_weights(sources_vertices,body)
    
    face_normals = calculate_face_normals(sources_vertices, source_faces)
    face_centers = calculate_face_centers(sources_vertices, source_faces)
    cloth_template = Template(
        sources_vertices = sources_vertices,
        source_faces = source_faces,
        garment_skinning = garment_skinning,
        face_normals = face_normals,
        face_centers = face_centers,
        cloth_template = None
    )

    cloth_deform = ClothDeform(cfg,cloth_template)
    cloth_deform.load_weights(os.path.join(output_path,'model.pth'))
    cloth_deform.set_eval()
    gt_jacobians = torch.load(os.path.join(output_path,'jacobians.pt'))
    os.makedirs(os.path.join(output_path, 'Saved_Meshes'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'Meshes'), exist_ok=True)
    if cfg.model_type == 'Dress4D':
        cam_data = MonocularDataset4DDress(cfg)
    else:
        cam_data = MonocularDataset(cfg)

    dataloader = DataLoader(dataset = cam_data, batch_size = cfg.batch_size, shuffle=False)
    n_verts_cannonical = jacobian_source.vertices_from_jacobians(gt_jacobians.detach()).squeeze()
    n_verts_cannonical = n_verts_cannonical + torch.mean(sources_vertices.detach(), axis=0, keepdims=True) 

    tri_mesh_template = trimesh.Trimesh(vertices=n_verts_cannonical.detach().cpu().numpy(), faces = source_faces.cpu().numpy())
    tri_mesh_template.export(os.path.join(output_path, 'Saved_Meshes',f'output_template.obj'))

    glctx = dr.RasterizeCudaContext()  
    gt_manager_source = GTInitializer()

    if texture :

        uv_tex_static = torch.load(os.path.join(output_path, 'texture_mlp_static.pt'))
        uvs = torch.load(os.path.join(output_path, 'uvs.pt'))
        indices = torch.load(os.path.join(output_path, 'indices.pt'))
        max_mip_level = cfg.max_mip_level

        if cfg.use_dynamic_texture:
            encoding = tcnn.Encoding(3, hash_cfg["encoding"])
            network = tcnn.Network(encoding.n_output_dims, 3, hash_cfg["network"])
            texture_mlp_dynamic = torch.nn.Sequential(encoding, network)
            texture_mlp_dynamic = tcnn.NetworkWithInputEncoding(
                n_input_dims=4,
                n_output_dims=3,
                encoding_config=hash_cfg["encoding"],
                network_config=hash_cfg["network"],
            ).to(device)
            texture_mlp_dynamic.load_state_dict(torch.load(os.path.join(output_path, 'texture_mlp_dynamic.pt')))
            y_coords = torch.arange(cfg.texture_map[0])
            x_coords = torch.arange(cfg.texture_map[1])            
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            img_pixel_indices = torch.stack([grid_x, grid_y], dim=-1).to(device).to(torch.float32)  # (H, W, 2)
            img_pixel_indices = img_pixel_indices / 1024.0

    front_images = []
    back_images = []
    texture_images = []
    texture_images_left = []
    texture_images_right = []

    save_image_dir = os.path.join(output_path,'save_img_mask')
    os.makedirs(save_image_dir, exist_ok=True)
    save_image_dir = os.path.join(output_path,'save_img_texture')
    os.makedirs(save_image_dir, exist_ok=True)
    save_image_dir = os.path.join(output_path,'save_img_texture_left')
    os.makedirs(save_image_dir, exist_ok=True)
    save_image_dir = os.path.join(output_path,'save_img_texture_right')
    os.makedirs(save_image_dir, exist_ok=True)
    
    with torch.no_grad():
        for _,sample in enumerate(dataloader):
            time = sample['time']
            idx = sample['idx']
            pose = sample['reduced_pose']
            if cfg.model_type == 'Dress4D':
                n_verts_cannonical_before = garment_skinning_function(n_verts_cannonical.unsqueeze(0).detach(), sample['pose'], sample['betas'], body, garment_skinning, sample['translation'])
            else:
                n_verts_cannonical_before = garment_skinning_function(n_verts_cannonical.unsqueeze(0).detach(), sample['pose'], sample['betas'], body, garment_skinning)
            
            n_verts_cannonical_before = n_verts_cannonical_before.squeeze(0)
            if cfg.model_type == 'Dress4D':
                vertices_post_skinning_cannonical = garment_skinning_function(n_verts_cannonical.unsqueeze(0), sample['pose'], \
                                                                    sample['betas'], body, garment_skinning, sample['translation'])
            else:
                vertices_post_skinning_cannonical = garment_skinning_function(n_verts_cannonical.unsqueeze(0), sample['pose'], \
                                                                    sample['betas'],body, garment_skinning)
            vertices_post_skinning_cannonical = vertices_post_skinning_cannonical.squeeze(0)
            cannonical_face_normals = calculate_face_normals(vertices_post_skinning_cannonical.detach(), source_faces)
            cannonical_face_centers = calculate_face_centers(vertices_post_skinning_cannonical.detach(), source_faces)  
            # face_normals = calculate_face_normals(n_verts_cannonical, source_faces)
            # face_centers = calculate_face_centers(n_verts_cannonical, source_faces)
            time_extended = time[None, ...].repeat(face_centers.shape[0], 1)    
            pose_extended = pose.repeat(face_centers.shape[0], 1)
            input = SimpleNamespace(
                        n_verts = n_verts_cannonical,
                        n_faces = source_faces,
                        face_centers = cannonical_face_centers,
                        face_normals = cannonical_face_normals,
                        time = time,
                        time_extended = time_extended,
                        pose_extended = pose_extended
                    )
            residual_jacobians = cloth_deform.forward(input)
            residual_jacobians = residual_jacobians.view(residual_jacobians.shape[0],3,3)
            iter_jacobians = gt_jacobians + residual_jacobians

            n_vert = jacobian_source.vertices_from_jacobians(iter_jacobians).squeeze()
            n_vert = n_vert + torch.mean(sources_vertices.detach(), axis=0, keepdims=True) 

            if idx[0] == 98.0:
                tri_mesh_cnannonical = trimesh.Trimesh(vertices=n_vert.detach().cpu().numpy(), faces = source_faces.cpu().numpy())
                tri_mesh_cnannonical.export(os.path.join(output_path, 'cannonical', f'cannonical_{idx[0].detach().cpu().numpy()}.obj')) 
            if cfg.model_type == 'Dress4D':
                new_vertices = garment_skinning_function(n_vert.unsqueeze(0), sample['pose'], sample['betas'], body, garment_skinning, sample['translation'])
            else:
                new_vertices = garment_skinning_function(n_vert.unsqueeze(0), sample['pose'], sample['betas'], body, garment_skinning)
            new_vertices = new_vertices.squeeze(0)   
            renderer_front = AlphaRenderer(sample['mv'].to('cuda'), sample['proj'].to('cuda'), [cfg.image_size, cfg.image_size])
            _, render_info,_ = gt_manager_source.render(new_vertices, source_faces, renderer_front)
            render_front = gt_manager_source.diffuse_images()
            mask_front = gt_manager_source.shillouette_images()
            # breakpoint()
            front_images.append(render_front)
            save_any_image(render_front, os.path.join(output_path, 'save_img',f'final_front_{idx[0].detach().cpu().numpy()}.png'))
            save_any_image(mask_front, os.path.join(output_path, 'save_img_mask',f'final_front_mask_{idx[0].detach().cpu().numpy()}.png'))
            

            if texture :
                if cfg.use_dynamic_texture:
                    pose = sample['reduced_pose_eight']
                    time_extended = time[None, None, ...].repeat(img_pixel_indices.shape[0], img_pixel_indices.shape[1], 1) 
                    pose_extended = pose[None, ...].repeat(img_pixel_indices.shape[0], img_pixel_indices.shape[1], 1)
                    tex_coords = torch.cat((img_pixel_indices, pose_extended), dim  = 2).view(-1,4)
                    uv_tex_dynamic = texture_mlp_dynamic(tex_coords.view(-1,4))
                    uv_tex_dynamic = uv_tex_dynamic.view(img_pixel_indices.shape[0], img_pixel_indices.shape[1], 3)

                    # tex_coords = torch.cat((img_pixel_indices, time_extended_texture), dim  = 2).view(-1,3)
                    # # tex_coords = torch.cat((img_pixel_indices, time_extended), dim  = 2).view(-1,3)
                    # uv_tex_dynamic = texture_mlp_dynamic(tex_coords.view(-1,3))
                    # uv_tex_dynamic = uv_tex_dynamic.view(img_pixel_indices.shape[0], img_pixel_indices.shape[1], 3)
            
                else:
                    uv_tex_total = uv_tex_static
                # breakpoint()
                uv_tex_total = uv_tex_static + uv_tex_dynamic
                # uv_tex_total = torch.clamp(uv_tex_total, 0, 1)
                r_mvp = torch.matmul(sample['proj'], sample['mv'])
                textured_image = render_texture(glctx, r_mvp, new_vertices, source_faces.to(torch.int32), uvs, indices, uv_tex_total , 1080, True, max_mip_level) 
                texture_images.append(textured_image)
                textured_image_masked = torch.mul(textured_image, mask_front)
                save_any_image(textured_image, os.path.join(output_path, 'save_img_texture',f'final_front_texture_{idx[0].detach().cpu().numpy()}.png'))

                r_mvp_left = torch.matmul(sample['proj'], sample['mv_left'])
                textured_image = render_texture(glctx, r_mvp_left, new_vertices, source_faces.to(torch.int32), uvs, indices, uv_tex_total , 1080, True, max_mip_level) 
                texture_images_left.append(textured_image)
                # textured_image_masked = torch.mul(textured_image, mask_front)
                save_any_image(textured_image, os.path.join(output_path, 'save_img_texture_left',f'final_left_texture_{idx[0].detach().cpu().numpy()}.png'))

                r_mvp_right = torch.matmul(sample['proj'], sample['mv_right'])
                textured_image = render_texture(glctx, r_mvp_right, new_vertices, source_faces.to(torch.int32), uvs, indices, uv_tex_total , 1080, True, max_mip_level) 
                texture_images_right.append(textured_image)
                # textured_image_masked = torch.mul(textured_image, mask_front)
                save_any_image(textured_image, os.path.join(output_path, 'save_img_texture_right',f'final_right_texture_{idx[0].detach().cpu().numpy()}.png'))

            sample['mv'][:,2,2] = -1*sample['mv'][:,2,2]
            renderer_back = AlphaRenderer(sample['mv'].to('cuda'), sample['proj'].to('cuda'), [cfg.image_size, cfg.image_size])
            _, render_info,_ = gt_manager_source.render(new_vertices, source_faces, renderer_back)
            render_back = gt_manager_source.diffuse_images()
            back_images.append(render_back)
            tri_mesh = trimesh.Trimesh(vertices=new_vertices.detach().cpu().numpy(), faces = source_faces.cpu().numpy())
            tri_mesh.export(os.path.join(output_path, 'Meshes', f'output_target_{idx[0].detach().cpu().numpy()}.obj'))
            tri_mesh_cnannonical = trimesh.Trimesh(vertices=n_verts_cannonical_before.detach().cpu().numpy(), faces = source_faces.cpu().numpy())
            tri_mesh_cnannonical.export(os.path.join(output_path, 'Saved_Meshes', f'output_before_{idx[0].detach().cpu().numpy()}.obj')) 

    save_video(front_images, output_path, 'front')
    save_video(back_images, output_path, 'back')
    if texture :
        save_video(texture_images, output_path, 'texture')
        save_video(texture_images_left, output_path, 'texture_left')
        save_video(texture_images_right, output_path, 'texture_right')

