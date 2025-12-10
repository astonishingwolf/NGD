import os
import pickle
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from torch.utils.data import DataLoader
from types import SimpleNamespace

from scripts.utils.general_utils import load_mesh_path, calculate_face_centers, calculate_face_normals
from scripts.renderer.renderer import GTInitializer, AlphaRenderer
from scripts.dataloader.Snap_data import MonocularDataset
from scripts.dataloader.Dress4D_data import MonocularDataset4DDress
from scripts.models.template import Template
from scripts.models.core.NeuralJacobianFields import SourceMesh
from scripts.utils import smpl
from scripts.utils.smpl_utils import garment_skinning_function, compute_rbf_skinning_weight, compute_k_nearest_skinning_weights
from scripts.models.temporal_deform import ClothDeform
from scripts.texture.texture import render_texture
from scripts.models.temporal_texture import ClothTexture
from scripts.utils.eval_utils import save_any_image, save_video
from scripts.utils.config_utils import load_hash_config

hash_cfg = load_hash_config()

def _get_mesh_path(cfg, remesh_dir):
    """Get mesh path based on remeshing config."""
    if cfg.remeshing:
        return os.path.join(remesh_dir, 'source_mesh.obj')
    return cfg.mesh

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

def _apply_garment_skinning(verts, sample, body, garment_skinning, is_dress4d=False, zero_pose=False):
    """Apply garment skinning."""
    pose = sample['pose'].clone() if zero_pose else sample['pose']
    if zero_pose:
        pose[..., :3] = 0.0
    kwargs = {'translation': sample['translation']} if is_dress4d else {}
    return garment_skinning_function(
        verts.unsqueeze(0), pose, sample['betas'], body, garment_skinning, **kwargs
    ).squeeze(0)

def _create_directories(output_path, dirs):
    """Create directories."""
    for dir_name in dirs:
        os.makedirs(os.path.join(output_path, dir_name), exist_ok=True)

def Inference(cfg, texture=False, device='cuda', mesh_inter=None):
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
    
    # Export source mesh
    tri_mesh_source = trimesh.Trimesh(
        vertices=sources_vertices.detach().cpu().numpy(),
        faces=source_faces.clone().cpu().numpy()
    )
    tri_mesh_source.export(os.path.join(output_path, 'mesh_source.obj'))
    
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
    
    # Create output directories
    _create_directories(output_path, ['Saved_Meshes', 'Meshes', 'save_img', 'cannonical'])
    
    # Setup data loader
    cam_data = MonocularDataset4DDress(cfg) if is_dress4d else MonocularDataset(cfg)
    dataloader = DataLoader(dataset=cam_data, batch_size=cfg.batch_size, shuffle=False)
    
    # Compute canonical vertices
    delta = None
    if cfg.remeshing:
        os.makedirs(remesh_dir, exist_ok=True)
        delta = torch.from_numpy(np.load(os.path.join(remesh_dir, 'delta_transform.npy'))).to(device)
    n_verts_cannonical = _compute_canonical_vertices(jacobian_source, gt_jacobians, sources_vertices, delta) 

    # Export template mesh
    tri_mesh_template = trimesh.Trimesh(
        vertices=n_verts_cannonical.detach().cpu().numpy(),
        faces=source_faces.cpu().numpy()
    )
    tri_mesh_template.export(os.path.join(output_path, 'Saved_Meshes', 'output_template.obj'))

    # Setup rendering
    glctx = dr.RasterizeCudaContext()
    gt_manager_source = GTInitializer()

    # Load texture if enabled
    img_pixel_indices = None
    cloth_texture = None
    if texture:
        uv_tex_static = torch.load(os.path.join(output_path, 'texture_mlp_static.pt'))
        uvs = torch.load(os.path.join(output_path, 'uvs.pt'))
        indices = torch.load(os.path.join(output_path, 'indices.pt'))
        max_mip_level = cfg.max_mip_level

        if cfg.use_dynamic_texture:
            cloth_texture = ClothTexture(cfg)
            cloth_texture.load_weights(os.path.join(output_path, 'texture_model_weights.pth'))
            y_coords = torch.arange(cfg.texture_map[0])
            x_coords = torch.arange(cfg.texture_map[1])
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            img_pixel_indices = torch.stack([grid_x, grid_y], dim=-1).to(device).to(torch.float32) / 1024.0

    # Create texture output directories
    texture_dirs = ['save_img_mask', 'save_img_mask_back', 'save_img_mask_left', 'save_img_mask_right',
                    'save_img_texture', 'save_img_texture_left', 'save_img_texture_right',
                    'save_img_back', 'texture_save']
    if texture:
        _create_directories(output_path, texture_dirs)

    front_images, back_images = [], []
    texture_images, texture_images_left, texture_images_right = [], [], []
    def _get_idx_val(sample):
        return int(sample['idx'][0].cpu().numpy())

    def _render_view(mv, proj, vertices, faces, image_size):
        """Render a view and return images."""
        renderer = AlphaRenderer(mv.to('cuda'), proj.to('cuda'), [image_size, image_size])
        _, render_info, _ = gt_manager_source.render(vertices, faces, renderer)
        return gt_manager_source.diffuse_images(), gt_manager_source.shillouette_images()

    with torch.no_grad():
        for _, sample in enumerate(dataloader):
            time, idx, pose = sample['time'], sample['idx'], sample['reduced_pose']
            idx_val = _get_idx_val(sample)
            
            # Get canonical vertices before deformation
            n_verts_cannonical_before = _apply_garment_skinning(
                n_verts_cannonical.detach(), sample, body, garment_skinning, is_dress4d
            )
            
            # Compute canonical vertices with zero pose for non-Dress4D
            vertices_post_skinning_cannonical = _apply_garment_skinning(
                n_verts_cannonical, sample, body, garment_skinning, is_dress4d, zero_pose=not is_dress4d
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
            
            # Save canonical mesh at specific frame
            if idx_val == 98:
                tri_mesh = trimesh.Trimesh(vertices=n_vert.detach().cpu().numpy(), faces=source_faces.cpu().numpy())
                tri_mesh.export(os.path.join(output_path, 'cannonical', f'cannonical_{idx_val}.obj'))
            
            # Apply final skinning
            new_vertices = _apply_garment_skinning(n_vert, sample, body, garment_skinning, is_dress4d)   

            # Render front view
            render_front, mask_front = _render_view(sample['mv'], sample['proj'], new_vertices, source_faces, cfg.image_size)
            front_images.append(render_front)
            save_any_image(render_front, os.path.join(output_path, 'save_img', f'final_front_{idx_val}.png'))
            save_any_image(mask_front, os.path.join(output_path, 'save_img_mask', f'final_front_mask_{idx_val}.png'))
            
            # Process texture if enabled
            if texture:
                # Get texture
                if cfg.use_dynamic_texture:
                    pose_eight = sample['reduced_pose_eight']
                    time_tex = time[None, None, ...].repeat(*img_pixel_indices.shape[:2], 1)
                    pose_tex = pose_eight[None, ...].repeat(*img_pixel_indices.shape[:2], 1)
                    uv_tex_dynamic = cloth_texture.forward(SimpleNamespace(
                        img_pixel_indices=img_pixel_indices, pose_extended=pose_tex
                    )).view(*img_pixel_indices.shape[:2], 3)
                    uv_tex_total = uv_tex_static + uv_tex_dynamic
                else:
                    uv_tex_total = uv_tex_static
                uv_tex_total = torch.clamp(uv_tex_total, 0, 1)
                
                save_any_image(uv_tex_total, os.path.join(output_path, 'texture_save', f'final_texture_{idx_val}.png'))
                
                # Render front textured view
                r_mvp = torch.matmul(sample['proj'], sample['mv'])
                textured_img = render_texture(glctx, r_mvp, new_vertices, source_faces.to(torch.int32),
                                              uvs, indices, uv_tex_total, 1080, False, max_mip_level)
                texture_images.append(textured_img)
                save_any_image(textured_img, os.path.join(output_path, 'save_img_texture', f'final_front_texture_{idx_val}.png'))
                
                # Render left textured view
                r_mvp_left = torch.matmul(sample['proj'], sample['mv_left'])
                textured_img_left = render_texture(glctx, r_mvp_left, new_vertices, source_faces.to(torch.int32),
                                                   uvs, indices, uv_tex_total, 1080, True, max_mip_level)
                texture_images_left.append(textured_img_left)
                save_any_image(textured_img_left, os.path.join(output_path, 'save_img_texture_left', f'final_left_texture_{idx_val}.png'))
                _, mask_left = _render_view(sample['mv_left'], sample['proj'], new_vertices, source_faces, cfg.image_size)
                save_any_image(mask_left, os.path.join(output_path, 'save_img_mask_left', f'final_left_mask_{idx_val}.png'))
                
                # Render right textured view
                r_mvp_right = torch.matmul(sample['proj'], sample['mv_right'])
                textured_img_right = render_texture(glctx, r_mvp_right, new_vertices, source_faces.to(torch.int32),
                                                    uvs, indices, uv_tex_total, 1080, True, max_mip_level)
                texture_images_right.append(textured_img_right)
                save_any_image(textured_img_right, os.path.join(output_path, 'save_img_texture_right', f'final_right_texture_{idx_val}.png'))
                _, mask_right = _render_view(sample['mv_right'], sample['proj'], new_vertices, source_faces, cfg.image_size)
                save_any_image(mask_right, os.path.join(output_path, 'save_img_mask_right', f'final_right_mask_{idx_val}.png'))

            # Render back view
            render_back, mask_back = _render_view(sample['mv_back'], sample['proj'], new_vertices, source_faces, cfg.image_size)
            back_images.append(render_back)
            save_any_image(mask_back, os.path.join(output_path, 'save_img_mask_back', f'final_back_mask_{idx_val}.png'))
            save_any_image(render_back, os.path.join(output_path, 'save_img_back', f'render_back_{idx_val}.png'))
            
            # Save meshes
            tri_mesh = trimesh.Trimesh(vertices=new_vertices.detach().cpu().numpy(), faces=source_faces.cpu().numpy())
            tri_mesh.export(os.path.join(output_path, 'Meshes', f'output_target_{idx_val}.obj'))
            tri_mesh_before = trimesh.Trimesh(vertices=n_verts_cannonical_before.detach().cpu().numpy(), faces=source_faces.cpu().numpy())
            tri_mesh_before.export(os.path.join(output_path, 'Saved_Meshes', f'output_before_{idx_val}.obj')) 

    save_video(front_images, output_path, 'front')
    save_video(back_images, output_path, 'back')
    if texture:
        save_video(texture_images, output_path, 'texture')
        save_video(texture_images_left, output_path, 'texture_left')
        save_video(texture_images_right, output_path, 'texture_right')

