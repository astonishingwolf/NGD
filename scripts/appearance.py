import os
import pickle
import numpy as np
import torch
import nvdiffrast.torch as dr
from torch.utils.data import DataLoader
from types import SimpleNamespace
from torchvision.utils import save_image

from scripts.utils.general_utils import load_mesh_path, calculate_face_centers, calculate_face_normals
from scripts.dataloader.Snap_data import MonocularDataset
from scripts.dataloader.Dress4D_data import MonocularTextureDataset4DDress
from scripts.models.template import Template
from scripts.models.core.NeuralJacobianFields import SourceMesh
from scripts.utils import smpl
from scripts.utils.smpl_utils import garment_skinning_function, compute_rbf_skinning_weight, compute_k_nearest_skinning_weights
from scripts.models.temporal_deform import ClothDeform
from scripts.texture.texture import render_texture
from scripts.models.temporal_texture import ClothTexture
from scripts.models.model_utils import get_linear_noise_func, get_expon_lr_func, get_linear_interpolation_func
from scripts.utils.eval_utils import save_any_image
from scripts.texture.texture import xatlas_uvmap
from scripts.utils.config_utils import load_hash_config

hash_cfg = load_hash_config()

def _get_mesh_path(cfg, remesh_dir):
    """Get mesh path based on remeshing config."""
    return os.path.join(remesh_dir, 'source_mesh.obj') if cfg.remeshing else cfg.mesh

def _load_jacobian_source(mesh_path, device):
    """Load jacobian source mesh."""
    jacobian_source = SourceMesh.SourceMesh(0, mesh_path, {}, 1, ttype=torch.float)
    jacobian_source.load()
    jacobian_source.to(device)
    return jacobian_source

def _compute_canonical_vertices(jacobian_source, gt_jacobians, sources_vertices, delta=None):
    """Compute canonical vertices from jacobians."""
    n_verts = jacobian_source.vertices_from_jacobians(gt_jacobians.detach()).squeeze()
    center = torch.mean(sources_vertices.detach(), axis=0, keepdims=True)
    return n_verts + center - (delta if delta is not None else 0)

def _apply_garment_skinning(verts, sample, body, garment_skinning, is_dress4d=False):
    """Apply garment skinning."""
    kwargs = {'translation': sample['translation']} if is_dress4d else {}
    return garment_skinning_function(
        verts.unsqueeze(0), sample['pose'], sample['betas'], body, garment_skinning, **kwargs
    ).squeeze(0)

def appearance_training_loop(cfg, device='cuda'):
    output_path = os.path.join(cfg.output_path, cfg.Exp_name)
    remesh_dir = os.path.join(output_path, 'remeshed_template')
    is_dress4d = cfg.model_type == 'Dress4D'
    
    # Initialize body
    body = smpl.SMPL(cfg.smpl_path).to(device)
    if cfg.custom_template:
        with open(cfg.template_smpl_pkl, 'rb') as f:
            body_data = pickle.load(f, encoding='latin1')
        template_smpl_shape = torch.tensor(body_data['shape']).to(device)
        body.update_shape(shape=template_smpl_shape)
    
    # Load mesh and jacobian source
    mesh_path = _get_mesh_path(cfg, remesh_dir)
    jacobian_source = _load_jacobian_source(mesh_path, device)
    sources_vertices, source_faces = load_mesh_path(mesh_path, scale=-2, device='cuda')
    source_faces = source_faces.to(torch.int64)
    
    # Compute skinning weights
    if cfg.skinning_func == 'rbf':
        garment_skinning = compute_rbf_skinning_weight(sources_vertices, body)
    elif cfg.skinning_func == 'k-near':
        garment_skinning = compute_k_nearest_skinning_weights(sources_vertices, body)
    
    # Create template
    face_normals = calculate_face_normals(sources_vertices, source_faces)
    face_centers = calculate_face_centers(sources_vertices, source_faces)
    cloth_template = Template(
        sources_vertices=sources_vertices, source_faces=source_faces,
        garment_skinning=garment_skinning, face_normals=face_normals,
        face_centers=face_centers, cloth_template=None
    )
    
    # Load deformation model
    cloth_deform = ClothDeform(cfg, cloth_template)
    cloth_deform.load_weights(os.path.join(output_path, 'model.pth'))
    cloth_deform.set_eval()
    gt_jacobians = torch.load(os.path.join(output_path, 'jacobians.pt'))
    
    # Setup data loader
    cam_data = MonocularTextureDataset4DDress(cfg) if is_dress4d else MonocularDataset(cfg)
    dataloader = DataLoader(dataset=cam_data, batch_size=cfg.batch_size, shuffle=True)
    
    # Compute canonical vertices
    delta = None
    if cfg.remeshing:
        delta = torch.from_numpy(np.load(os.path.join(remesh_dir, 'delta_transform.npy'))).to(device)
    n_verts_cannonical = _compute_canonical_vertices(jacobian_source, gt_jacobians, sources_vertices, delta)
    
    # Setup UV mapping
    total_frames = len(dataloader)
    indices, uvs, vmapping = xatlas_uvmap(n_verts_cannonical, source_faces)
    indices = indices.to(torch.int32)
    
    # Initialize texture model
    cloth_texture = ClothTexture(cfg)
    cloth_texture.training_init()
    
    # Setup texture grid
    y_coords = torch.arange(cfg.texture_map[0])
    x_coords = torch.arange(cfg.texture_map[1])
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    img_pixel_indices = torch.stack([grid_x, grid_y], dim=-1).to(device).to(torch.float32) / 1024.0
    
    # Setup static texture optimizer
    uv_tex_static = torch.full([cfg.texture_map[0], cfg.texture_map[1], 3], 0.2, device='cuda', requires_grad=True)
    max_mip_level = cfg.max_mip_level
    optimizer_static = torch.optim.Adam(
        [{'params': [uv_tex_static], 'lr': cfg.texture_lr_init * cfg.spatial_lr_scale, "name": "texture"}],
        lr=0.0, eps=1e-15
    )
    scheduler_static = get_expon_lr_func(
        lr_init=cfg.texture_lr_init * cfg.spatial_lr_scale,
        lr_final=cfg.texture_lr_final,
        lr_delay_mult=cfg.texture_lr_delay_mult,
        max_steps=cfg.texture_lr_max_steps
    )
    
    # Setup rendering
    glctx = dr.RasterizeCudaContext()
    if cfg.save_instance:
        save_image_dir_fr = os.path.join(output_path, f'texture_image_epoch_{cfg.save_index}')
        os.makedirs(save_image_dir_fr, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'logs'), exist_ok=True)
    smooth_term = get_linear_noise_func(lr_init=0.01, lr_final=1e-15, lr_delay_mult=0.01, max_steps=100000)
    
    for e in range(cfg.tex_epochs):
        loss_each_epoch = 0.0
        
        for it, sample in enumerate(dataloader):
            cloth_texture.optimizer.zero_grad()
            time, pose = sample['time'], sample['reduced_pose']
            
            # Compute canonical vertices
            vertices_post_skinning_cannonical = _apply_garment_skinning(
                n_verts_cannonical, sample, body, garment_skinning, is_dress4d
            )
            cannonical_face_normals = calculate_face_normals(vertices_post_skinning_cannonical.detach(), source_faces)
            cannonical_face_centers = calculate_face_centers(vertices_post_skinning_cannonical.detach(), source_faces)
            
            # Forward through deformation network
            time_extended = time[None, ...].repeat(face_centers.shape[0], 1)
            pose_extended = pose.repeat(face_centers.shape[0], 1)
            input = SimpleNamespace(
                n_verts=n_verts_cannonical, n_faces=source_faces,
                face_centers=cannonical_face_centers, face_normals=cannonical_face_normals,
                time=time, time_extended=time_extended, pose_extended=pose_extended
            )
            residual_jacobians = cloth_deform.forward(input).view(-1, 3, 3)
            iter_jacobians = gt_jacobians + residual_jacobians
            n_vert = _compute_canonical_vertices(jacobian_source, iter_jacobians, sources_vertices, delta)
            
            # Apply final skinning
            new_vertices = _apply_garment_skinning(n_vert, sample, body, garment_skinning, is_dress4d)
            
            # Compute dynamic texture
            if cfg.use_dynamic_texture and e >= cfg.texture_warm_ups:
                pose_noise = 0.01 * torch.randn(2, device='cuda') * total_frames * smooth_term((e + 1) * total_frames + it)
                pose_eight = sample['reduced_pose_eight']
                pose_tex = pose_eight[None, ...].repeat(*img_pixel_indices.shape[:2], 1) + pose_noise
                uv_tex_dynamic = cloth_texture.forward(SimpleNamespace(
                    img_pixel_indices=img_pixel_indices, pose_extended=pose_tex
                )).view(*img_pixel_indices.shape[:2], 3)
            else:
                uv_tex_dynamic = torch.zeros_like(uv_tex_static)
            
            uv_tex_total = torch.clamp(uv_tex_static + uv_tex_dynamic, 0, 1)
            
            # Render textured image
            r_mvp = torch.matmul(sample['proj'], sample['mv'])
            textured_image = render_texture(glctx, r_mvp, new_vertices, source_faces.to(torch.int32),
                                            uvs, indices, uv_tex_total, 1080, False, max_mip_level)
            
            if cfg.save_instance and sample['idx'][0] == cfg.save_index:
                save_image(textured_image.permute(0, 3, 1, 2), os.path.join(save_image_dir_fr, f'texture_{e}.png'))
            
            # Compute loss
            textured_image_masked = torch.mul(textured_image.permute(0, 3, 1, 2), sample['target_shil'])
            target_seg_image = torch.mul(sample['tex_image'], sample['target_shil'])
            loss = ((textured_image_masked - target_seg_image.to(textured_image_masked.dtype)) ** 2 /
                   (textured_image_masked.detach() ** 2 + 0.01)).mean()
            
            # Optimize static texture
            optimizer_static.zero_grad()
            loss.backward()
            optimizer_static.step()
            new_lr = scheduler_static(e * total_frames + it)
            for param_group in optimizer_static.param_groups:
                if param_group["name"] == "texture":
                    param_group['lr'] = new_lr

            cloth_texture.step(e, it, total_frames)
            loss_each_epoch += loss.item()

        save_any_image(uv_tex_total, os.path.join(output_path, 'logs', f"save_tex_{e}.png"))
        print(f"Epoch {e} loss {loss_each_epoch:.4f}")

    torch.save(uv_tex_static, os.path.join(output_path, 'texture_mlp_static.pt'))
    torch.save(uvs, os.path.join(output_path, 'uvs.pt'))
    torch.save(indices, os.path.join(output_path, 'indices.pt'))
    torch.save(vmapping, os.path.join(output_path, 'vmapping.pt'))
    cloth_texture.save_weights(output_path)