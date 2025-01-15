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
from torch.utils.tensorboard import SummaryWriter

from scripts.utils import *
from scripts.models.model import Model
from scripts.utils.general_utils import load_mesh_path, calculate_face_centers, calculate_face_normals
from scripts.renderer.renderer import GTInitializer, AlphaRenderer
from scripts.losses.loss import Loss
from scripts.dataloader.Snap_data import MonocularDataset
from scripts.utils import *
from scripts.models import model
from scripts.models.template import Template
from scripts.models.core.NeuralJacobianFields import SourceMesh
from scripts.utils import smpl
from scripts.utils.general_utils import *
from scripts.utils.smpl_utils import *
from scripts.utils.cloth import Cloth
from scripts.models.remesh.remesh import *

def training_loop(cfg,device):

    output_path = os.path.join(cfg.output_path, cfg.Exp_name)
    body = smpl.SMPL(cfg.smpl_path).to(device)
    load_mesh = get_mesh(cfg.mesh, output_path, cfg.retriangulate, cfg.bsdf)

    jacobians_intialized = SourceMesh.SourceMesh(0, cfg.mesh, {}, 1, ttype=torch.float)
    jacobians_intialized.load()
    jacobians_intialized.to(device)
    jacobians_remeshed = jacobians_intialized

    template_vertices_orig, template_faces_orig = load_mesh_path(cfg.mesh, scale = -2, device = 'cuda')    
    # template_vertices_orig, template_faces_orig = template_vertices_orig.clone(), template_faces_orig.clone()

    if cfg.skinning_func == 'rbf':
        garment_skinning_orig = compute_rbf_skinning_weight(template_vertices_orig,body)
    elif cfg.skinning_func == 'k-near':
        garment_skinning_orig = compute_k_nearest_skinning_weights(template_vertices_orig,body)

    # garment_skinning_orig = compute_rbf_skinning_weight(template_vertices_orig, body)

    template_faces_orig = template_faces_orig.to(torch.int64)
    
    with torch.no_grad():
        cannonical_jacobians = jacobians_intialized.jacobians_from_vertices(template_vertices_orig.unsqueeze(0))
    
    template_vertices_remeshed = template_vertices_orig.clone()
    template_faces_remeshed = template_faces_orig.clone()
   
    template_face_normals_orig = calculate_face_normals(template_vertices_orig, template_faces_orig)
    template_face_centers_orig = calculate_face_centers(template_vertices_orig, template_faces_orig)

    template_cloth = Cloth(template_vertices_orig,template_faces_orig)

    cloth_template = Template(
        sources_vertices = template_vertices_orig,
        source_faces = template_faces_orig,
        garment_skinning = garment_skinning_orig,
        face_normals = template_face_normals_orig,
        face_centers = template_face_centers_orig,
        cloth_template = template_cloth
    )

    template_garment_skinning = garment_skinning_orig
    template_face_normal = template_face_normals_orig
    template_face_centers = template_face_centers_orig
    cannonical_faces = template_faces_orig

    mlp = Model(cfg,cloth_template)
    mlp = mlp.to(device)

    jacobian_optimizer_residual = torch.optim.Adam(mlp.parameters(), lr = 1e-3)
    jacobian_scheduler_residual = torch.optim.lr_scheduler.LambdaLR(jacobian_optimizer_residual,lr_lambda=lambda x: max(0.01, 10**(-x*0.05)))
    
    jacobian_optimizer_cannonical = torch.optim.Adam([cannonical_jacobians], lr = 1e-3)
    # jacobian_scheduler_gt = torch.optim.lr_scheduler.LambdaLR(jacobian_optimizer,lr_lambda=lambda x: max(0.01, 10**(-x*0.05)))
    cannonical_jacobians.requires_grad_(True)   
        
    glctx = dr.RasterizeCudaContext()  
    gt_manager_source = GTInitializer()
        
    cam_data = MonocularDataset(cfg)
    dataloader = DataLoader(dataset = cam_data, batch_size = cfg.batch_size, shuffle=True)
    
    epochs = cfg.num_epochs
    total_loss = Loss(cfg, cloth_template)

    log_dir = os.makedirs(os.path.join(output_path,'logs'), exist_ok=True)
    writer = SummaryWriter(log_dir)
    cannonical_verts = template_vertices_orig
    
    for e in range(epochs) :
        for it,sample in enumerate(dataloader):

            # if e >= cfg.warm_ups:
                
            #     n_verts_cannonical = jacobian_source.vertices_from_jacobians(gt_jacobians.detach()).squeeze()
            #     n_verts_cannonical = n_verts_cannonical + torch.mean(sources_vertices.detach(), axis=0, keepdims=True)
            #     cloth = Cloth(n_verts_cannonical,source_faces)
  
            jacobian_optimizer_residual.zero_grad()
            time = sample['time']
            time_extended = time[None, ...].repeat(template_face_centers.shape[0], 1)
            
            if e >= cfg.warm_ups:

                cannonical_verts = jacobians_remeshed.vertices_from_jacobians(cannonical_jacobians.detach()).squeeze()
                cannonical_verts = cannonical_verts + torch.mean(template_vertices_orig.detach(), axis=0).repeat(cannonical_verts.shape[0],1)
                cannonical_faces = template_faces_remeshed
                # breakpoint()
                cannonical_face_normals = calculate_face_normals(cannonical_verts.detach(), cannonical_faces)
                cannonical_face_centers = calculate_face_centers(cannonical_verts.detach(), cannonical_faces)

                input = SimpleNamespace(
                    n_verts = cannonical_verts,
                    n_faces = cannonical_faces,
                    face_centers = cannonical_face_centers,
                    face_normals = cannonical_face_normals,
                    time = time,
                    time_extended = time_extended
                )

                residual_jacobians = mlp(input)
                residual_jacobians = residual_jacobians.view(residual_jacobians.shape[0],3,3)
                # residual_jacobians.retain_grad(True)
                cannonical_mesh = Meshes(verts = [cannonical_verts.detach()], faces = [cannonical_faces])
                
            else : 

                residual_jacobians  = torch.zeros_like(cannonical_jacobians)
            # residual_jacobians.retain_grad()
            residual_jacobians.requires_grad_(True)  # Note the underscore at the end
            residual_jacobians.retain_grad()
            combined_jacobians = cannonical_jacobians + residual_jacobians

            vertices_derived_before_mean = jacobians_remeshed.vertices_from_jacobians(combined_jacobians).squeeze()
            vertices_pre_skinning = vertices_derived_before_mean + torch.mean(template_vertices_orig.detach(), axis=0).repeat(cannonical_verts.shape[0],1)

            vertices_post_skinning = garment_skinning_function(vertices_pre_skinning.unsqueeze(0), sample['pose'], sample['betas'], body, template_garment_skinning)
            vertices_post_skinning = vertices_post_skinning.squeeze(0)
            # vertices_post_skinning.retain_grad()
            tmp_path = '/hdd_data/nakul/soham/people_snapshot_preet/monocloth_preet/tmp'
            tri_mesh = trimesh.Trimesh(vertices=vertices_post_skinning.detach().cpu().numpy(), faces = template_faces_remeshed.clone().cpu().numpy())
            tri_mesh.export(os.path.join(tmp_path,f'output_debug.obj')) 
            deformed_mesh_p3d = Meshes(verts = [vertices_post_skinning], faces = [cannonical_faces])          
        
            renderer = AlphaRenderer(sample['mv'].to('cuda'), sample['proj'].to('cuda'), [cfg.image_size, cfg.image_size])  
            _, render_info = gt_manager_source.render(vertices_post_skinning, cannonical_faces, renderer)
            
            face_mask = render_info['masks']
            train_render = gt_manager_source.diffuse_images()
            train_shil = gt_manager_source.shillouette_images()
            jacobian_optimizer_cannonical.zero_grad()

            train_target_render = sample['target_diffuse'].to('cuda')
            train_target_render_shil = sample['target_shil'].to('cuda')
            
            # breakpoint()

            pred = SimpleNamespace(
                pred_verts=vertices_post_skinning,
                pred_faces=cannonical_faces,
                deformed_mesh_p3d=deformed_mesh_p3d,
                iter_jacobians = combined_jacobians,
                residual_jacobians = residual_jacobians,
                train_render=train_render,
                train_shil = train_shil,
                face_masks= face_mask
            )

            target = SimpleNamespace(
                train_target_render=train_target_render,
                train_target_render_shil=train_target_render_shil
            )

            loss, loss_dict =  total_loss.forward(pred,target)            

            loss.backward()

            # if e > 300:
            #     breakpoint()
            #Gradient Clipping
            # clip grads;
            # with th.no_grad():
            #     p_pos_grad = p_pos.grad if p_pos.grad is not None else th.zeros_like(p_pos)
            #     p_w_grad = p_w.grad if p_w.grad is not None else th.zeros_like(p_w)
            #     p_r_grad = p_r.grad if p_r.grad is not None else th.zeros_like(p_r)

            #     p_pos_grad_norm = th.norm(p_pos_grad, dim=-1) + 1e-6
            #     p_w_grad_norm = th.abs(p_w_grad) + 1e-6
            #     p_r_grad_norm = th.abs(p_r_grad) + 1e-6

            #     max_grad_norm = self.update_bound / self.lr

            #     p_pos_idx = p_pos_grad_norm > max_grad_norm
            #     p_w_idx = p_w_grad_norm > max_grad_norm
            #     p_r_idx = p_r_grad_norm > max_grad_norm

            #     p_pos_grad[p_pos_idx] = (p_pos_grad[p_pos_idx] / p_pos_grad_norm[p_pos_idx].unsqueeze(-1)) * max_grad_norm
            #     p_w_grad[p_w_idx] = (p_w_grad[p_w_idx] / p_w_grad_norm[p_w_idx]) * max_grad_norm
            #     p_r_grad[p_r_idx] = (p_r_grad[p_r_idx] / p_r_grad_norm[p_r_idx]) * max_grad_norm
                
            #     # fix for nan grads;
            #     p_pos_grad_nan_idx = th.any(th.isnan(p_pos_grad), dim=-1)
            #     p_w_grad_nan_idx = th.isnan(p_w_grad)
            #     p_r_grad_nan_idx = th.isnan(p_r_grad)
                
            #     p_pos_grad[p_pos_grad_nan_idx] = 0.0
            #     p_w_grad[p_w_grad_nan_idx] = 0.0
            #     p_r_grad[p_r_grad_nan_idx] = 0.0

            #     if p_pos.grad is not None:
            #         p_pos.grad.data = p_pos_grad
            #     if p_w.grad is not None:
            #         p_w.grad.data = p_w_grad
            #     if p_r.grad is not None:
            #         p_r.grad.data = p_r_grad
                
            #     p_pos_nan_grad_ratio = th.count_nonzero(p_pos_grad_nan_idx) / p_pos_grad_nan_idx.shape[0]
            #     p_w_nan_grad_ratio = th.count_nonzero(p_w_grad_nan_idx) / p_w_grad_nan_idx.shape[0]
            #     p_r_nan_grad_ratio = th.count_nonzero(p_r_grad_nan_idx) / p_r_grad_nan_idx.shape[0]


            # jacobian_scheduler_gt.step()
            if e < cfg.warm_ups:
                jacobian_optimizer_cannonical.step()
            # jacobian_optimizer_cannonical.step()
            jacobian_optimizer_residual.step()
            jacobian_scheduler_residual.step()
            # breakpoint()
            # Calculate the global step count
            global_step = e * len(dataloader) + it

            if cfg.logging:
                ## Logging Losses
                writer.add_scalar('Loss/Global', loss, global_step)
                writer.add_scalar('Loss/Regularization', loss_dict['loss_render'], global_step)
                writer.add_scalar('Loss/Rendering', loss_dict['loss_regularization'], global_step)
                writer.add_scalar('Loss/Regularization/loss_edge', loss_dict['loss_edge'], global_step)
                writer.add_scalar('Loss/Regularization/loss_normal', loss_dict['loss_normal'], global_step)
                writer.add_scalar('Loss/Regularization/loss_laplacian', loss_dict['loss_laplacian'], global_step)
                writer.add_scalar('Loss/Regularization/loss_jacobian', loss_dict['loss_jacobian'], global_step)
                
                ## Logging Images
                writer.add_images('render_images', train_render.permute(0,3,1,2), global_step)
                writer.add_images('target_images', train_target_render, global_step)

                # Log the current epoch and iteration
                writer.add_scalar('Training/Epoch', e, global_step)
                writer.add_scalar('Training/Iteration', it, global_step)

                # # Logging in the mesh 
                # colors = torch.full_like(vertices, 255)  # 255 for white in RGB
                # writer.add_mesh('white_3d_object', 
                #                 vertices=vertices.unsqueeze(0),
                #                 colors=colors.unsqueeze(0),
                #                 faces=faces.unsqueeze(0))

        # jacobian_optimizer_cannonical.step()
        # with torch.no_grad():
        # breakpoint()

        if e == cfg.warm_ups - 1:
            tri_mesh = trimesh.Trimesh(vertices=vertices_post_skinning.detach().cpu().numpy(), faces = template_faces_remeshed.clone().cpu().numpy())
            tri_mesh.export(os.path.join(output_path,f'output_warmup.obj')) 

        if e >= cfg.warm_ups and e % cfg.remesh_freq == 0:

            print(f"Starting to Remesh {e}")
            remesh_dir = os.path.join(output_path, 'remeshed_template')
            if not os.path.exists(remesh_dir):
                os.makedirs(remesh_dir)
            
            # breakpoint()
            
            template_vertices_remeshed_tmp, template_faces_remeshed_tmp = adaptive_remesh(cannonical_verts, template_faces_remeshed,\
                vertices_orig = template_vertices_remeshed, faces_orig = template_faces_remeshed)

            # sources_vertices , source_faces = remeshed_template.clone(), remeshed_template_faces.clone()
            template_vertices_remeshed_tmp, template_faces_remeshed_tmp = merge_close_vertices(template_vertices_remeshed_tmp, template_faces_remeshed_tmp)
            # # breakpoint()
            
            tri_mesh = trimesh.Trimesh(vertices = template_vertices_remeshed_tmp.detach().cpu().numpy(), \
                faces = template_faces_remeshed_tmp.detach().cpu().numpy())

            tri_mesh.export(os.path.join(remesh_dir, f'remeshed_{e}.obj'))
            tri_mesh.export(os.path.join(remesh_dir, f'remeshed_last.obj'))   
            
            # tri_mesh = trimesh.Trimesh(vertices = template_vertices_remeshed_tmp.detach().cpu().numpy(), \
            #     faces = template_faces_remeshed_tmp.detach().cpu().numpy())
            # tri_mesh.export(os.path.join(remesh_dir, f'remeshed_before_{e}.obj'))
            

            jacobians_remeshed = SourceMesh.SourceMesh(0, os.path.join(remesh_dir, f'remeshed_{e}.obj'), {}, 1, ttype=torch.float)
            jacobians_remeshed.load(make_new = True)
            jacobians_remeshed.to(device)

            # # with torch.no_grad():
            interpolated_jacobians = jacobian_interpolation(cannonical_jacobians.detach(), template_vertices_remeshed_tmp, template_vertices_remeshed, \
                template_faces_remeshed_tmp, template_faces_remeshed)
            
            # breakpoint()
            cannonical_jacobians = interpolated_jacobians.clone().detach()
            cannonical_jacobians.requires_grad_(True)  
            jacobian_optimizer_cannonical = torch.optim.Adam([cannonical_jacobians], lr = 1e-3)
                       
            
            cannonical_verts = jacobians_remeshed.vertices_from_jacobians(cannonical_jacobians.detach()).squeeze()
            cannonical_verts = cannonical_verts + torch.mean(template_vertices_remeshed.detach(), axis=0).unsqueeze(0).repeat(cannonical_verts.shape[0],1)
            
            # breakpoint()
            template_garment_skinning = garment_skinning_interpolation(template_vertices_remeshed_tmp, template_garment_skinning, template_vertices_remeshed)
            # breakpoint() 
            # source_faces = remeshed_template_faces.to(torch.int64)

            template_face_normals = calculate_face_normals(template_vertices_remeshed_tmp, template_faces_remeshed_tmp)
            template_face_centers = calculate_face_centers(template_vertices_remeshed_tmp, template_faces_remeshed_tmp)

            cloth_template.update(
                sources_vertices = template_vertices_remeshed_tmp,
                source_faces = template_faces_remeshed_tmp,
                garment_skinning = template_garment_skinning,
                face_normals = template_face_normals,
                face_centers = template_face_centers,
                cloth_template = template_cloth
            )

            template_vertices_remeshed = template_vertices_remeshed_tmp
            template_faces_remeshed = template_faces_remeshed_tmp
            cannonical_faces = template_faces_remeshed
        
        if e >= cfg.warm_ups and e % 50 == 0:
            tri_mesh = trimesh.Trimesh(vertices = cannonical_verts.detach().cpu().numpy(), faces= cannonical_faces.detach().cpu().numpy())

            debug_dir = os.path.join(output_path,'debug')
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)

            tri_mesh.export(os.path.join(output_path,'debug',f'cannonical_{e}.obj')) 
        
        print(f"Epoch {e} loss {loss.item()}")
    
    writer.close()
    
    tri_mesh = trimesh.Trimesh(vertices = cannonical_verts.detach().cpu().numpy(), faces= template_faces_remeshed.detach().cpu().numpy())
    tri_mesh.export(os.path.join(output_path,f'cannonical_final.obj')) 
    
    tri_mesh = trimesh.Trimesh(vertices = vertices_post_skinning.detach().cpu().numpy(), faces = template_faces_remeshed.clone().cpu().numpy())
    tri_mesh.export(os.path.join(output_path,f'output_final.obj')) 
    
    torch.save(mlp.state_dict(), os.path.join(output_path,'model.pth'))
    torch.save(cannonical_jacobians, os.path.join(output_path,'jacobians.pt'))
    
    return

def time_eval(cfg, device = 'cuda', mesh_inter = None):

    output_path = os.path.join(cfg.output_path, cfg.Exp_name)
    body = smpl.SMPL(cfg.smpl_path).to(device)
    # load_mesh = get_mesh(cfg.mesh, output_path, cfg.retriangulate, cfg.bsdf)
    # jacobian_source = SourceMesh.SourceMesh(0, cfg.mesh, {}, 1, ttype=torch.float)

    # body = smpl.SMPL(cfg.smpl_path).to(device)
    remesh_dir = os.path.join(output_path, 'remeshed_template')
    load_mesh = get_mesh(os.path.join(remesh_dir, f'remeshed_last.obj'), output_path, cfg.retriangulate, cfg.bsdf)
    jacobian_source = SourceMesh.SourceMesh(0, os.path.join(remesh_dir, f'remeshed_last.obj'), {}, 1, ttype=torch.float)
    jacobian_source.load()
    jacobian_source.to(device)
    
    

    sources_vertices,source_faces = load_mesh_path(os.path.join(remesh_dir, f'remeshed_last.obj'), scale = -2, device = 'cuda')
    sources_vertices_orig,_ = load_mesh_path(cfg.mesh, scale = -2, device = 'cuda')
    tri_mesh_source = trimesh.Trimesh(vertices=sources_vertices.detach().cpu().numpy(), faces=source_faces.clone().cpu().numpy())
    tri_mesh_source.export(os.path.join(output_path,f'mesh_source.obj')) 
    if cfg.skinning_func == 'rbf':
        garment_skinning = compute_rbf_skinning_weight(sources_vertices,body)
    elif cfg.skinning_func == 'k-near':
        garment_skinning = compute_k_nearest_skinning_weights(sources_vertices,body)
    # garment_skinning = compute_rbf_skinning_weight(sources_vertices,body)
    source_faces = source_faces.to(torch.int64)
    # breakpoint()
    with torch.no_grad():
        gt_jacobians = jacobian_source.jacobians_from_vertices(sources_vertices.unsqueeze(0))
    
    # template_cloth = Cloth(sources_vertices,source_faces)

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

    mlp = Model(cfg,cloth_template)
    mlp.load_state_dict(torch.load(os.path.join(output_path,'model.pth')))
    mlp = mlp.to(device)
    mlp.eval()

    gt_jacobians = torch.load(os.path.join(output_path,'jacobians.pt'))
      
    os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)
    rot_ang = 0.0
    
    cam_data = MonocularDataset(cfg)
    dataloader = DataLoader(dataset = cam_data, batch_size = cfg.batch_size, shuffle=False)
        
    gt_manager_source = GTInitializer()
    epochs = cfg.num_epochs
    
    n_verts_cannonical = jacobian_source.vertices_from_jacobians(gt_jacobians.detach()).squeeze()
    n_verts_cannonical = n_verts_cannonical + torch.mean(sources_vertices_orig.detach(), axis=0, keepdims=True)
    eface_normals = calculate_face_normals(n_verts_cannonical, source_faces)
    face_centers = calculate_face_centers(n_verts_cannonical, source_faces)

    tri_mesh_template = trimesh.Trimesh(vertices=n_verts_cannonical.detach().cpu().numpy(), faces=load_mesh.t_pos_idx.clone().cpu().numpy())
    tri_mesh_template.export(os.path.join(output_path,f'output_template.obj'))
    i = 0
    
    for i,sample in enumerate(dataloader):

        time = sample['time']
        
        n_verts_cannonical_before = garment_skinning_function(n_verts_cannonical.unsqueeze(0).detach(), sample['pose'], sample['betas'], body, garment_skinning)
        n_verts_cannonical_before = n_verts_cannonical_before.squeeze(0)
        
        face_normals = calculate_face_normals(n_verts_cannonical, source_faces)
        face_centers = calculate_face_centers(n_verts_cannonical, source_faces)

        time_extended = time[None, ...].repeat(face_centers.shape[0], 1)
        input = SimpleNamespace(
                    n_verts = n_verts_cannonical,
                    n_faces = source_faces,
                    face_centers = face_centers,
                    face_normals = face_normals,
                    time = time,
                    time_extended = time_extended
                )
        
        residual_jacobians = mlp(input)
        residual_jacobians = residual_jacobians.view(residual_jacobians.shape[0],3,3)
        
        iter_jacobians = gt_jacobians + residual_jacobians
        
        n_vert = jacobian_source.vertices_from_jacobians(iter_jacobians).squeeze()
        n_vert = n_vert + torch.mean(sources_vertices_orig.detach(), axis=0, keepdims=True)
        new_vertices = garment_skinning_function(n_vert.unsqueeze(0), sample['pose'], sample['betas'], body, garment_skinning)
        new_vertices = new_vertices.squeeze(0)   
        
        tri_mesh = trimesh.Trimesh(vertices=new_vertices.detach().cpu().numpy(), faces=load_mesh.t_pos_idx.clone().cpu().numpy())
        tri_mesh.export(os.path.join(output_path,f'output_target_{i}.obj'))
        tri_mesh_cnannonical = trimesh.Trimesh(vertices=n_verts_cannonical_before.detach().cpu().numpy(), faces=load_mesh.t_pos_idx.clone().cpu().numpy())
        tri_mesh_cnannonical.export(os.path.join(output_path,f'output_before_{i}.obj')) 
        i = i + 1
            
def loop(cfg):
    
    output_path = cfg['output_path']
    os.makedirs(output_path, exist_ok=True)
    
    with open(os.path.join(output_path,'config.yml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
        
    cfg = EasyDict(cfg['timeloop'])
    print(f"Running Module 1")
    
    print(f'Output directory {output_path} created')

    device = cfg['device']
    torch.cuda.set_device(device)
    
    training_loop(cfg, device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    print(f"Running Module 2")
    
    # time_exp(cfg, cfg.mesh_inter)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    print(f"Evaluating")
    time_eval(cfg, device)

