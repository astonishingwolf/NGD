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
import matplotlib.pyplot as plt
import pickle

from scripts.utils import *
from scripts.models.model import Model
from scripts.utils.general_utils import load_mesh_path, calculate_face_centers, calculate_face_normals
from scripts.renderer.renderer import GTInitializer, AlphaRenderer
from scripts.losses.loss import Loss
from scripts.dataloader.Snap_data import MonocularDataset
from scripts.dataloader.Dress4D_data import *
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
from scripts.models.model_utils import get_embedder
from scripts.models.texture_model import TextureModel
from scripts.models.temporal_texture import ClothTexture

with open("scripts/config_hash.json") as f:
	hash_cfg = json.load(f)

def appearance_training_loop(cfg, device = 'cuda'):

    output_path = os.path.join(cfg.output_path, cfg.Exp_name)
    remesh_dir = os.path.join(output_path, 'remeshed_template')
    body = smpl.SMPL(cfg.smpl_path).to(device)
    smooth_term = get_linear_noise_func(lr_init=0.01, lr_final=1e-15, lr_delay_mult=0.01, max_steps=100000)
    if cfg.custom_template:
        with open(cfg.template_smpl_pkl, 'rb') as f:
            body_data = pickle.load(f, encoding='latin1')
        templpate_smpl_shape = torch.tensor(body_data['shape']).to(device)
        body.update_shape(shape = templpate_smpl_shape)
    # jacobian_source = SourceMesh.SourceMesh(0, os.path.join(remesh_dir, f'source_mesh.obj'), {}, 1, ttype=torch.float)
    # jacobian_source = SourceMesh.SourceMesh(0, os.path.join(remesh_dir, f'source_mesh.obj'), {}, 1, ttype=torch.float)
    if cfg.remeshing:
        jacobian_source = SourceMesh.SourceMesh(0, os.path.join(remesh_dir, f'source_mesh.obj'), {}, 1, ttype=torch.float)
    else:
        jacobian_source = SourceMesh.SourceMesh(0, cfg.mesh, {}, 1, ttype=torch.float)

    jacobian_source.load()
    jacobian_source.to(device)
    if cfg.remeshing:
        sources_vertices,source_faces = load_mesh_path(os.path.join(remesh_dir, f'source_mesh.obj'), scale = -2, device = 'cuda')
    else:
        sources_vertices,source_faces = load_mesh_path(cfg.mesh, scale = -2, device = 'cuda')
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
    
    cam_data = (MonocularTextureDataset4DDress(cfg) if cfg.model_type == 'Dress4D' else MonocularDataset(cfg))
        
    dataloader = DataLoader(dataset = cam_data, batch_size = cfg.batch_size, shuffle=True)
    if cfg.remeshing:
        delta =  torch.from_numpy(np.load(os.path.join(remesh_dir, f'delta_transform.npy'))).to(device)  
        n_verts_cannonical = jacobian_source.vertices_from_jacobians(gt_jacobians.detach()).squeeze()
        n_verts_cannonical = n_verts_cannonical + torch.mean(sources_vertices.detach(), axis=0, keepdims=True) - delta
    else:
        n_verts_cannonical = jacobian_source.vertices_from_jacobians(gt_jacobians.detach()).squeeze()
        n_verts_cannonical = n_verts_cannonical + torch.mean(sources_vertices.detach(), axis=0, keepdims=True) 
    total_frames = dataloader.__len__()
    indices, uvs, vmapping = xatlas_uvmap(n_verts_cannonical, source_faces)
    indices = indices.to(torch.int32)
    
    cloth_texture = ClothTexture(cfg)
    cloth_texture.training_init()

    y_coords = torch.arange(cfg.texture_map[0])
    x_coords = torch.arange(cfg.texture_map[1])            
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    img_pixel_indices = torch.stack([grid_x, grid_y], dim=-1).to(device).to(torch.float32)  # (H, W, 2)

    uv_tex_static = torch.full([cfg.texture_map[0], cfg.texture_map[1], 3], 0.2, device='cuda', requires_grad=True)
    max_mip_level = cfg.max_mip_level
    l = [
            {'params': [uv_tex_static],
            'lr': cfg.texture_lr_init * cfg.spatial_lr_scale,
            "name": "texture"}
        ]
    optimizer_static = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    scheduler_static = get_expon_lr_func(lr_init=cfg.texture_lr_init * cfg.spatial_lr_scale,
                                                    lr_final=cfg.texture_lr_final,
                                                    lr_delay_mult=cfg.texture_lr_delay_mult,
                                                    max_steps=cfg.texture_lr_max_steps)

    glctx = dr.RasterizeCudaContext()
    if cfg.save_instance:
        save_image_dir_fr = os.path.join(output_path, f'texture_image_epoch_{cfg.save_index}')
    os.makedirs(save_image_dir_fr, exist_ok=True)
    img_pixel_indices = img_pixel_indices / 1024.0
    warm_up = get_linear_interpolation_func(0, 1, max_steps = 100)
    
    for e in range(cfg.tex_epochs):

        loss_each_epoch = 0.0
        
        for it,sample in enumerate(dataloader):
            
            cloth_texture.optimizer.zero_grad()
            time = sample['time']
            pose = sample['reduced_pose']
            idx = sample['idx']
            if cfg.model_type == 'Dress4D':
                vertices_post_skinning_cannonical = garment_skinning_function(n_verts_cannonical.unsqueeze(0), sample['pose'], \
                                                                    sample['betas'], body, garment_skinning, sample['translation'])
            else:
                vertices_post_skinning_cannonical = garment_skinning_function(n_verts_cannonical.unsqueeze(0), sample['pose'], \
                                                                    sample['betas'],body, garment_skinning)

            vertices_post_skinning_cannonical = vertices_post_skinning_cannonical.squeeze(0)
            cannonical_face_normals = calculate_face_normals(vertices_post_skinning_cannonical.detach(), source_faces)
            cannonical_face_centers = calculate_face_centers(vertices_post_skinning_cannonical.detach(), source_faces)  
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
            if cfg.remeshing:
                n_vert = n_vert + torch.mean(sources_vertices.detach(), axis=0, keepdims=True)  - delta
            else :
                n_vert = n_vert + torch.mean(sources_vertices.detach(), axis=0, keepdims=True)

            # n_vert = n_vert + torch.mean(sources_vertices.detach(), axis=0, keepdims=True) - delta           
            if cfg.model_type == 'Dress4D':
                new_vertices = garment_skinning_function(n_vert.unsqueeze(0), sample['pose'], sample['betas'], body, garment_skinning, sample['translation'])
            else:
                new_vertices = garment_skinning_function(n_vert.unsqueeze(0), sample['pose'], sample['betas'], body, garment_skinning)
            
            new_vertices = new_vertices.squeeze(0)   
            pose_noise = 0.01 * torch.randn(2, device='cuda') * cam_data.__len__() * smooth_term((e+1) * cam_data.__len__() + it) 
            if cfg.use_dynamic_texture and e >=  cfg.texture_warm_ups:
                pose = sample['reduced_pose_eight'] 
                # breakpoint()
                pose_extended = pose[None, ...].repeat(img_pixel_indices.shape[0], img_pixel_indices.shape[1], 1) + pose_noise
                input = SimpleNamespace(
                img_pixel_indices=img_pixel_indices,
                pose_extended=pose_extended,
                )
                uv_tex_dynamic = cloth_texture.forward(input)
                uv_tex_dynamic = uv_tex_dynamic.view(img_pixel_indices.shape[0], img_pixel_indices.shape[1], 3)
            else:
                uv_tex_dynamic = torch.zeros_like(uv_tex_static)

            uv_tex_total = uv_tex_static + uv_tex_dynamic
            uv_tex_total = torch.clamp(uv_tex_total, 0, 1)
            # if idx[0] == 0:
            #     breakpoint()
            r_mvp = torch.matmul(sample['proj'], sample['mv'])
            textured_image = render_texture(glctx, r_mvp, new_vertices, source_faces.to(torch.int32), uvs, indices, uv_tex_total , 1080, False, max_mip_level)

            if cfg.save_instance and sample['idx'][0] == cfg.save_index:
                save_image(textured_image.permute(0,3,1,2), os.path.join(save_image_dir_fr, f'texture_{e}.png'))

            textured_image_masked = torch.mul(textured_image.permute(0,3,1,2), sample['target_shil'])
            # breakpoint()
            target_seg_image =  torch.mul(sample['tex_image'], sample['target_shil'])
            loss = (textured_image_masked - target_seg_image.to(textured_image_masked.dtype)) ** 2 / (textured_image_masked.detach() ** 2 + 0.01) 
            loss = loss.mean()
            optimizer_static.zero_grad()
            loss.backward()
            optimizer_static.step()
            new_lr = scheduler_static((e)*total_frames + it)
        
            for param_group in optimizer_static.param_groups:
                if param_group["name"] == "texture":
                    param_group['lr'] =new_lr

            cloth_texture.step(e, it, total_frames)          
            loss_each_epoch += loss.item()

        save_any_image(uv_tex_total,os.path.join(output_path, 'logs', f"save_tex_{e}.png"))
        print(f"Epoch {e} loss {loss_each_epoch}")

    # breakpoint()   
    torch.save(uv_tex_static, os.path.join(output_path, 'texture_mlp_static.pt'))
    torch.save(uvs, os.path.join(output_path, 'uvs.pt'))
    torch.save(indices, os.path.join(output_path, 'indices.pt'))
    torch.save(vmapping, os.path.join(output_path, 'vmapping.pt'))
    cloth_texture.save_weights(output_path)