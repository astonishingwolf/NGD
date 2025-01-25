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


def geometry_training_loop(cfg,device):

    output_path = os.path.join(cfg.output_path, cfg.Exp_name)
    cloth_optim = ClothModel(cfg)
    cloth_deform = ClothDeform(cfg, cloth_optim.cloth_template)
    cloth_optim.training_init()
    cloth_deform.training_init()
    smooth_term = get_linear_noise_func(lr_init=0.01, lr_final=1e-15, lr_delay_mult=0.01, max_steps=120000)

    glctx = dr.RasterizeCudaContext()  
    gt_manager_source = GTInitializer()
        
    if cfg.model_type == 'Dress4D':
        cam_data = MonocularDataset4DDress(cfg)
    else:
        cam_data = MonocularDataset(cfg)

    dataloader = DataLoader(dataset = cam_data, batch_size = cfg.batch_size, shuffle=True)
    total_frames = cam_data.__len__()
    epochs = cfg.num_epochs
    total_loss = Loss(cfg, cloth_optim.cloth_template)

    log_dir = os.path.join(output_path,'logs')
    os.makedirs(os.path.join(output_path,'logs'), exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    cannonical_dir = os.path.join(output_path,'cannonical')
    os.makedirs(os.path.join(output_path,'cannonical'), exist_ok=True)
    deform_model_weights_dir = os.path.join(output_path,'model_weights')
    os.makedirs(os.path.join(output_path,'model_weights'), exist_ok=True)
    remesh_dir = os.path.join(output_path, 'remeshed_template')
    os.makedirs(remesh_dir, exist_ok=True)
    mesh_source = trimesh.Trimesh(vertices=cloth_optim.template_vertices_orig.detach().cpu().numpy(), faces=cloth_optim.template_faces_orig.clone().cpu().numpy())
    mesh_source.export(os.path.join(remesh_dir, f'original_mesh.obj'))
    save_image_dir = os.path.join(output_path,'save_img')
    os.makedirs(save_image_dir, exist_ok=True)
    save_image_dir = os.path.join(output_path,'Meshes_UV')
    os.makedirs(save_image_dir, exist_ok=True)
    save_image_dir = os.path.join(output_path,'Maps')
    os.makedirs(save_image_dir, exist_ok=True)

    if cfg.save_instance:
        save_image_dir_fr = os.path.join(output_path,f'save_img_epoch_{cfg.save_index}')
        os.makedirs(save_image_dir_fr, exist_ok=True)

    for e in range(epochs) :

        loss_each_epoch = 0.0
        loss_rendering = 0.0
        loss_regularization = 0.0
        loss_depth = 0.0
        
        for it,sample in enumerate(dataloader):

            cloth_deform.optimizer.zero_grad()
            
            ast_noise = torch.randn(1, 1, device='cuda').expand(cloth_optim.template_face_centers.shape[0], -1) * cam_data.__len__() * smooth_term((e+1) * cam_data.__len__() + it)            
            time = torch.zeros_like(sample['time']).to(device)
            # time = torchsample['time']
            pose = sample['reduced_pose']
            if cfg.model_type == 'Dress4D':
                time_extended = time[None, ...].repeat(cloth_optim.template_face_centers.shape[0], 1)
            elif cfg.pose_noise:
                time_extended = time[None, ...].repeat(cloth_optim.template_face_centers.shape[0], 1) + ast_noise
            else:
                time_extended = time[None, ...].repeat(cloth_optim.template_face_centers.shape[0], 1)
            # breakpoint()
            if cfg.model_type == 'Dress4D':
                pose_extended = pose.repeat(cloth_optim.template_face_centers.shape[0], 1)
            elif cfg.pose_noise:
                pose_extended = pose.repeat(cloth_optim.template_face_centers.shape[0], 1) + ast_noise
            else:
                pose_extended = pose.repeat(cloth_optim.template_face_centers.shape[0], 1)

            if e >= cfg.warm_ups:
                # breakpoint()
                cannonical_verts, cannonical_faces = cloth_optim.get_mesh_attr_from_jacobians(cloth_optim.cannonical_jacobians.detach())
                
                if cfg.model_type == 'Dress4D':
                    vertices_post_skinning_cannonical = garment_skinning_function(cannonical_verts.unsqueeze(0), sample['pose'], \
                                                                        sample['betas'], cloth_optim.body, cloth_optim.template_garment_skinning, sample['translation'])
                else:
                    vertices_post_skinning_cannonical = garment_skinning_function(cannonical_verts.unsqueeze(0), sample['pose'], \
                                                                        sample['betas'], cloth_optim.body, cloth_optim.template_garment_skinning)

                vertices_post_skinning_cannonical = vertices_post_skinning_cannonical.squeeze(0)
                cannonical_face_normals = calculate_face_normals(vertices_post_skinning_cannonical.detach(), cannonical_faces)
                cannonical_face_centers = calculate_face_centers(vertices_post_skinning_cannonical.detach(), cannonical_faces)
                
                input = SimpleNamespace(
                    n_verts = cannonical_verts,
                    n_faces = cannonical_faces,
                    face_centers = cannonical_face_centers,
                    face_normals = cannonical_face_normals,
                    time = time,
                    time_extended = time_extended,
                    pose_extended = pose_extended
                )
                residual_jacobians = cloth_deform.forward(input)
                residual_jacobians = residual_jacobians.view(residual_jacobians.shape[0],3,3)
            else : 
                residual_jacobians  = torch.zeros_like(cloth_optim.cannonical_jacobians)

            combined_jacobians = cloth_optim.cannonical_jacobians + residual_jacobians
            vertices_pre_skinning,_ = cloth_optim.get_mesh_attr_from_jacobians(combined_jacobians)

            if cfg.model_type == 'Dress4D':
                vertices_post_skinning = garment_skinning_function(vertices_pre_skinning.unsqueeze(0), sample['pose'], \
                                                                    sample['betas'], cloth_optim.body, cloth_optim.template_garment_skinning, sample['translation'])
            else:
                vertices_post_skinning = garment_skinning_function(vertices_pre_skinning.unsqueeze(0), sample['pose'], \
                                                                    sample['betas'], cloth_optim.body, cloth_optim.template_garment_skinning)
            vertices_post_skinning = vertices_post_skinning.squeeze(0)

            if cfg.vertex_noise:
                vertex_noise = cfg.noise_level * torch.randn(vertices_post_skinning.shape[0], 3, device='cuda')* cam_data.__len__() * smooth_term((e+1) * cam_data.__len__() + it)      
                vertices_post_skinning = vertices_post_skinning + vertex_noise
                # breakpoint()

            deformed_mesh_p3d = Meshes(verts = [vertices_post_skinning], faces = [cloth_optim.cannonical_faces])               
            renderer = AlphaRenderer(sample['mv'].to('cuda'), sample['proj'].to('cuda'), [cfg.image_size, cfg.image_size])  
            _, render_info, rast_out = gt_manager_source.render(vertices_post_skinning, cloth_optim.cannonical_faces, renderer)
            face_mask = render_info['masks']

            train_render = gt_manager_source.diffuse_images()
            train_render.requires_grad_(True)
            train_render.retain_grad()

            train_shil = gt_manager_source.shillouette_images()
            train_norm = gt_manager_source.normal_images()
            train_depth = gt_manager_source.depth_images()
            cloth_optim.jacobian_optimizer_cannonical.zero_grad()
            train_target_render,train_target_render_shil, target_complete_shil = sample['target_diffuse'].to('cuda'),sample['target_shil'].to('cuda'), sample['target_complete_shil'].to('cuda')
            train_target_depth = sample['target_depth'].to('cuda')
            train_target_normal = (sample['target_norm_map'].to('cuda') + 1)/2
            if cfg.save_instance and sample['idx'][0] == cfg.save_index:
                save_image(train_render.permute(0,3,1,2), os.path.join(save_image_dir_fr, f'output_{e}.png'))

            pred = SimpleNamespace(
                pred_verts=vertices_post_skinning,
                pred_faces=cloth_optim.cannonical_faces,
                deformed_mesh_p3d=deformed_mesh_p3d,
                iter_jacobians = combined_jacobians,
                residual_jacobians = residual_jacobians,
                train_render=train_render,
                train_shil = train_shil,
                train_norm = train_norm,
                train_depth = train_depth,
                face_masks= face_mask
            )

            target = SimpleNamespace(
                train_target_render = train_target_render,
                train_target_render_shil = train_target_render_shil,
                train_target_complete = target_complete_shil,
                train_target_depth = train_target_depth,
                train_target_normal = train_target_normal,
                hands_shil = sample['hands_shil'].to('cuda'),
            )

            loss, loss_dict =  total_loss.forward(pred,target)  
            loss.backward()

            if cfg.gradient_clipping :
                torch.nn.utils.clip_grad_norm_([cloth_optim.cannonical_jacobians], max_norm=1.0)

            if e >= cfg.warm_ups_remesh and e % cfg.remesh_freq == 0 and cfg.remeshing:
                cloth_optim.save_img_grad(train_render.grad, train_shil, rast_out)

            cloth_optim.step(e, it, total_frames)
            cloth_deform.step(e, it, total_frames)

            if cfg.logging:
                global_step = e * len(dataloader) + it
                writer.add_scalar('Loss/Global', loss, global_step)
                writer.add_scalar('Loss/Regularization', loss_dict['loss_render'], global_step)
                writer.add_scalar('Loss/Rendering', loss_dict['loss_regularization'], global_step)
                writer.add_scalar('Loss/Regularization/loss_edge', loss_dict['loss_edge'], global_step)
                writer.add_scalar('Loss/Regularization/loss_normal', loss_dict['loss_normal'], global_step)
                writer.add_scalar('Loss/Regularization/loss_laplacian', loss_dict['loss_laplacian'], global_step)
                writer.add_scalar('Loss/Regularization/loss_jacobian', loss_dict['loss_jacobian'], global_step)
                writer.add_images('render_images', train_render.permute(0,3,1,2), global_step)
                writer.add_images('target_images', train_target_render, global_step)
                writer.add_scalar('Training/Epoch', e, global_step)
                writer.add_scalar('Training/Iteration', it, global_step)

            # breakpoint()
            loss_each_epoch += loss.item()
            loss_rendering += loss_dict['loss_render'].item()
            loss_regularization += loss_dict['loss_regularization'].item()
            loss_depth += loss_dict['loss_depth'].item()

            # loss_rendering += 0
            # loss_regularization += 0

        if e % 100 == 0:
            cloth_optim.save_cannonical_mesh(cannonical_dir, e)
            cloth_deform.save_weights(output_path,e)

        if e >= cfg.warm_ups_remesh and e % cfg.remesh_freq == 0 and cfg.remeshing and e <= cfg.remesh_stop:
            cloth_optim.remesh_triangles(e, max_percentile = cfg.remesh_percentile)
    
        print(f"Epoch {e} loss {loss_each_epoch} loss rendering {loss_rendering} loss regular {loss_regularization} loss depth {loss_depth}")

    writer.close()
    cloth_optim.save_cannonical_mesh(output_path)
    cloth_optim.save_jacobians(output_path)
    cloth_deform.save_weights(output_path)
    
    return