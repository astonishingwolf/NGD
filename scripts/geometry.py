import os
import torch
import trimesh
import nvdiffrast.torch as dr
from torch.utils.data import DataLoader
from types import SimpleNamespace
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.structures import Meshes

from scripts.utils.general_utils import calculate_face_centers, calculate_face_normals, jacobian_interpolation
from scripts.renderer.renderer import GTInitializer, AlphaRenderer
from scripts.losses.loss import Loss
from scripts.dataloader.Snap_data import MonocularDataset 
from scripts.dataloader.Dress4D_data import MonocularDataset4DDress
from scripts.models.template import ClothModel
from scripts.models.temporal_deform import ClothDeform
from scripts.models.model_utils import get_linear_noise_func, get_linear_interpolation_func
from scripts.utils.smpl_utils import garment_skinning_function
from scripts.utils.eval_utils import save_mesh 

def _get_dataset(cfg):
    return MonocularDataset4DDress(cfg) if cfg.model_type == 'Dress4D' else MonocularDataset(cfg)

def _apply_garment_skinning(verts, sample, body, garment_skinning, is_dress4d=False, zero_pose=False):
    """Apply garment skinning with optional translation for Dress4D and zero pose for non-Dress4D."""
    pose = sample['pose'] if not zero_pose else sample['pose'].clone()
    if zero_pose:
        pose[..., :3] = 0.0
    kwargs = {'translation': sample['translation']} if is_dress4d else {}
    return garment_skinning_function(
        verts.unsqueeze(0), pose, sample['betas'], body, garment_skinning, **kwargs
    ).squeeze(0)

def _create_directories(output_path, dirs):
    """Create directories and return path dictionary."""
    paths = {}
    for dir_name in dirs:
        path = os.path.join(output_path, dir_name)
        os.makedirs(path, exist_ok=True)
        paths[dir_name] = path
    return paths

def geometry_training_loop(cfg, device):
    output_path = os.path.join(cfg.output_path, cfg.Exp_name)
    is_dress4d = cfg.model_type == 'Dress4D'
    
    # Initialize models
    cloth_optim = ClothModel(cfg)
    cloth_deform = ClothDeform(cfg, cloth_optim.cloth_template)
    cloth_optim.training_init()
    cloth_deform.training_init()
    
    # Setup data and utilities
    cam_data = _get_dataset(cfg)
    dataloader = DataLoader(dataset=cam_data, batch_size=cfg.batch_size, shuffle=True)
    total_frames = len(cam_data)
    smooth_term = get_linear_noise_func(lr_init=0.01, lr_final=1e-15, lr_delay_mult=0.01, max_steps=120000)
    
    glctx = dr.RasterizeCudaContext()
    gt_manager_source = GTInitializer()
    total_loss = Loss(cfg, cloth_optim.cloth_template)
    
    # Create directories
    dir_names = ['cannonical', 'model_weights', 'remeshed_template', 'save_img', 'Meshes_UV', 
                 'Maps', 'mesh_epochs', 'logs', f'save_img_epoch_{cfg.save_index}']
    dirs = _create_directories(output_path, dir_names)
    
    log_dir = dirs['logs']
    cannonical_dir = dirs['cannonical']
    remesh_dir = dirs['remeshed_template']
    save_image_dir_fr = dirs[f'save_img_epoch_{cfg.save_index}']
    save_mesh_dir_fr = dirs['mesh_epochs']
    
    writer = SummaryWriter(log_dir)
    warm_up = get_linear_interpolation_func(0, 1, max_steps=10)
    epochs = cfg.num_epochs
    
    # Export original mesh
    mesh_source = trimesh.Trimesh(
        vertices=cloth_optim.template_vertices_orig.detach().cpu().numpy(),
        faces=cloth_optim.template_faces_orig.clone().cpu().numpy()
    )
    mesh_source.export(os.path.join(remesh_dir, 'original_mesh.obj'))
    jacobian_dict = {}

    for e in range(epochs):

        loss_each_epoch = 0.0
        loss_rendering = 0.0
        loss_regularization = 0.0
        loss_depth = 0.0
        
        for it, sample in enumerate(dataloader):
            cloth_deform.optimizer.zero_grad()
            
            n_faces = cloth_optim.template_face_centers.shape[0]
            time = torch.zeros_like(sample['time']).to(device)
            pose = sample['reduced_pose']
            time_extended = time[None, ...].repeat(n_faces, 1)
            
            # Apply pose noise if enabled
            pose_extended = pose.repeat(n_faces, 1)
            if cfg.pose_noise and not is_dress4d:
                ast_noise = 0.1 * torch.randn(4, device='cuda').unsqueeze(0).expand(n_faces, -1)
                ast_noise *= total_frames * smooth_term((e + 1) * total_frames + it)
                pose_extended += ast_noise
                
            if e >= cfg.warm_ups:
                cannonical_verts, cannonical_faces = cloth_optim.get_mesh_attr_from_jacobians(
                    cloth_optim.cannonical_jacobians.detach(), with_align=True
                )
                vertices_post_skinning_cannonical = _apply_garment_skinning(
                    cannonical_verts, sample, cloth_optim.body,
                    cloth_optim.template_garment_skinning, is_dress4d, zero_pose=not is_dress4d
                ) 
                cannonical_face_normals = calculate_face_normals(vertices_post_skinning_cannonical.detach(), cannonical_faces)
                cannonical_face_centers = calculate_face_centers(vertices_post_skinning_cannonical.detach(), cannonical_faces)
                
                input = SimpleNamespace(
                    n_verts = cannonical_verts,
                    n_faces = cannonical_faces,
                    face_centers = cannonical_face_centers,
                    face_normals = cannonical_face_normals,
                    time = time,
                    time_extended = time_extended,
                    pose_extended = pose_extended,
                    epoch = e,
                    warm_ups = cfg.warm_ups,
                )

                residual_jacobians = cloth_deform.forward(input)
                residual_jacobians = residual_jacobians.view(residual_jacobians.shape[0],3,3)

                # Remeshing logic
                should_remesh = cfg.remeshing and e < cfg.remesh_stop
                if should_remesh and e >= cfg.warm_ups_remesh and e % cfg.remesh_freq == 0:
                    index = int(sample['idx'][0].cpu().numpy())
                    jacobian_dict[str(index)] = residual_jacobians.detach()
                    save_mesh(cannonical_verts, cannonical_faces, os.path.join(remesh_dir, f'remeshed_{e}.obj'))

                # Interpolation during remeshing transition
                in_interpolation = (should_remesh and e > cfg.warm_ups_remesh and 
                    ((e - 1) % cfg.remesh_freq - 10) < 0)
                if in_interpolation:
                    index = int(sample['idx'][0].cpu().numpy())
                    remesh_it = (e - cfg.warm_ups_remesh) % cfg.remesh_freq
                    residual_jacobians_interpolated = jacobian_interpolation(
                        jacobian_dict[str(index)].unsqueeze(0).detach(),
                        cloth_optim.template_vertices_remeshed,
                        cloth_optim.template_vertices_remeshed_prev,
                        cloth_optim.template_faces_remeshed,
                        cloth_optim.template_faces_remeshed_prev
                    )
                    alpha = warm_up(remesh_it)
                    save_mesh(cannonical_verts, cannonical_faces, os.path.join(remesh_dir, f'remeshed_can_{e}.obj'))
                    residual_jacobians_blend = alpha * residual_jacobians + (1 - alpha) * residual_jacobians_interpolated.detach()
                    combined_jacobians = cloth_optim.cannonical_jacobians + residual_jacobians_blend
                else:
                    combined_jacobians = cloth_optim.cannonical_jacobians + residual_jacobians
                
                # Save canonical mesh periodically (skip if already saved during interpolation)
                if e % 100 == 0 and not in_interpolation:
                    save_mesh(cannonical_verts, cannonical_faces, os.path.join(remesh_dir, f'remeshed_can_{e}.obj'))
            
            else:
                residual_jacobians = torch.zeros_like(cloth_optim.cannonical_jacobians)
                combined_jacobians = cloth_optim.cannonical_jacobians + residual_jacobians

            vertices_pre_skinning, _ = cloth_optim.get_mesh_attr_from_jacobians(combined_jacobians)
            vertices_post_skinning = _apply_garment_skinning(
                vertices_pre_skinning, sample, cloth_optim.body,
                cloth_optim.template_garment_skinning, is_dress4d
            )

            if cfg.vertex_noise:
                vertex_noise = cfg.noise_level * torch.randn(vertices_post_skinning.shape[0], 3, device='cuda')
                vertex_noise *= total_frames * smooth_term((e + 1) * total_frames + it)
                vertices_post_skinning += vertex_noise

            # Rendering
            deformed_mesh_p3d = Meshes(verts=[vertices_post_skinning], faces=[cloth_optim.cannonical_faces])
            renderer = AlphaRenderer(sample['mv'].to('cuda'), sample['proj'].to('cuda'), [cfg.image_size, cfg.image_size])
            _, render_info, rast_out = gt_manager_source.render(vertices_post_skinning, cloth_optim.cannonical_faces, renderer)
            
            train_render = gt_manager_source.diffuse_images()
            train_render.requires_grad_(True).retain_grad()
            train_shil = gt_manager_source.shillouette_images()
            train_norm = gt_manager_source.normal_images()
            train_depth = gt_manager_source.depth_images()
            
            cloth_optim.jacobian_optimizer_cannonical.zero_grad()
            train_target_render = sample['target_diffuse'].to('cuda')
            train_target_render_shil = sample['target_shil'].to('cuda')
            target_complete_shil = sample['target_complete_shil'].to('cuda')
            train_target_depth = sample['target_depth'].to('cuda')
            train_target_normal = (sample['target_norm_map'].to('cuda') + 1) / 2

            # Save outputs
            if cfg.save_instance and sample['idx'][0] == cfg.save_index:
                save_image(train_render.permute(0, 3, 1, 2), os.path.join(save_image_dir_fr, f'output_{e}.png'))
                if e % cfg.save_freq == 0 or e % cfg.remesh_freq in [1, 5, 10]:
                    save_mesh(vertices_post_skinning, cloth_optim.cannonical_faces, 
                             os.path.join(save_mesh_dir_fr, f'output_{e}.obj'))

            pred = SimpleNamespace(
                pred_verts=vertices_post_skinning, pred_faces=cloth_optim.cannonical_faces,
                deformed_mesh_p3d=deformed_mesh_p3d, iter_jacobians=combined_jacobians,
                residual_jacobians=residual_jacobians, train_render=train_render,
                train_shil=train_shil, train_norm=train_norm, train_depth=train_depth,
                face_masks=render_info['masks'], epoch=e, warm_ups=cfg.warm_ups
            )
            target = SimpleNamespace(
                train_target_render=train_target_render, train_target_render_shil=train_target_render_shil,
                train_target_complete=target_complete_shil, train_target_depth=train_target_depth,
                train_target_normal=train_target_normal, hands_shil=sample['hands_shil'].to('cuda')
            )

            loss, loss_dict = total_loss.forward(pred, target)
            loss.backward()

            if cfg.gradient_clipping:
                torch.nn.utils.clip_grad_norm_([cloth_optim.cannonical_jacobians], max_norm=1.0)

            if (should_remesh and e >= cfg.warm_ups_remesh and 
                e % cfg.remesh_freq == 0 and e <= cfg.remesh_stop):
                cloth_optim.save_img_grad(train_render.grad, train_target_render_shil, rast_out)

            cloth_optim.step(e, it, total_frames)
            cloth_deform.step(e, it, total_frames)

            if cfg.logging:
                global_step = e * len(dataloader) + it
                writer.add_scalar('Loss/Global', loss, global_step)
                writer.add_scalar('Loss/Regularization', loss_dict['loss_render'], global_step)
                writer.add_scalar('Loss/Rendering', loss_dict['loss_regularization'], global_step)
                for key in ['loss_edge', 'loss_normal', 'loss_laplacian', 'loss_jacobian']:
                    writer.add_scalar(f'Loss/Regularization/{key}', loss_dict[key], global_step)
                writer.add_images('render_images', train_render.permute(0, 3, 1, 2), global_step)
                writer.add_images('target_images', train_target_render, global_step)
                writer.add_scalar('Training/Epoch', e, global_step)
                writer.add_scalar('Training/Iteration', it, global_step)

            loss_each_epoch += loss.item()
            loss_rendering += loss_dict['loss_render'].item()
            loss_regularization += loss_dict['loss_regularization'].item()
            loss_depth += loss_dict['loss_depth'].item()

        if e % 100 == 0:
            cloth_optim.save_cannonical_mesh(cannonical_dir, e)
            cloth_deform.save_weights(output_path,e)

        if should_remesh and e >= cfg.warm_ups_remesh and e % cfg.remesh_freq == 0:
            cloth_optim.remesh_triangles(e, max_percentile=cfg.remesh_percentile)

        print(f"Epoch {e} loss {loss_each_epoch:.4f} rendering {loss_rendering:.4f} "
              f"regular {loss_regularization:.4f} depth {loss_depth:.4f}")

    writer.close()
    cloth_optim.save_cannonical_mesh(output_path)
    cloth_optim.save_jacobians(output_path)
    cloth_deform.save_weights(output_path)
