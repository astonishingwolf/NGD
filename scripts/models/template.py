import os
import sys
import numpy as np
import torch
import torch.nn as nn
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

from dataclasses import dataclass
from scripts.utils.cloth import Cloth
from scripts.utils import *
from scripts.models.model import Model
from scripts.utils.general_utils import load_mesh_path, calculate_face_centers, calculate_face_normals
from scripts.renderer.renderer import GTInitializer, AlphaRenderer
from scripts.losses.loss import Loss
from scripts.dataloader.Snap_data import MonocularDataset
from scripts.utils import *
from scripts.models import model
from scripts.models.core.NeuralJacobianFields import SourceMesh
from scripts.utils import smpl
from scripts.utils.general_utils import *
from scripts.utils.smpl_utils import *
from scripts.utils.cloth import Cloth
from scripts.models.remesh.remesh import loop_subdivision, triangle_subdivision,remesh
# from scripts.models.remesh.remesh_legacy import *
from scripts.models.model_utils import get_linear_interpolation_func,get_expon_lr_func, interpolate_vertices
from scripts.utils.helper import *

@dataclass
class Template:
    
    sources_vertices : torch.Tensor
    source_faces : torch.Tensor
    garment_skinning : torch.Tensor
    face_normals : torch.Tensor
    face_centers : torch.Tensor
    cloth_template : Cloth

    def update(self, 
               sources_vertices: torch.Tensor = None, 
               source_faces: torch.Tensor = None, 
               garment_skinning: torch.Tensor = None, 
               face_normals: torch.Tensor = None, 
               face_centers: torch.Tensor = None, 
               cloth_template: Cloth = None):
               
        if sources_vertices is not None:
            self.sources_vertices = sources_vertices
        if source_faces is not None:
            self.source_faces = source_faces
        if garment_skinning is not None:
            self.garment_skinning = garment_skinning
        if face_normals is not None:
            self.face_normals = face_normals
        if face_centers is not None:
            self.face_centers = face_centers
        if cloth_template is not None:
            self.cloth_template = cloth_template

class ClothModel:

    def __init__(self,cfg):

        self.device = cfg.device
        self.template_vertices_orig, self.template_faces_orig = load_mesh_path(cfg.mesh, scale = -2, device = self.device)    
        self.jacobians_intialized = SourceMesh.SourceMesh(0, cfg.mesh, {}, 1, ttype=torch.float)
        self.jacobians_intialized.load()
        self.jacobians_intialized.to(self.device)
        self.jacobians_remeshed = self.jacobians_intialized
        self.body = smpl.SMPL(cfg.smpl_path).to(self.device)
        self.spatial_lr_scale = 5

        if cfg.skinning_func == 'rbf':
            self.garment_skinning_orig = compute_rbf_skinning_weight(self.template_vertices_orig, self.body)
        elif cfg.skinning_func == 'k-near':
            self.garment_skinning_orig = compute_k_nearest_skinning_weights(self.template_vertices_orig, self.body)    

        self.template_faces_orig = self.template_faces_orig.to(torch.int64)

        with torch.no_grad():
            self.cannonical_jacobians = nn.Parameter(self.jacobians_intialized.jacobians_from_vertices(self.template_vertices_orig.unsqueeze(0)).requires_grad_(True))
            # self.cannonical_jacobians = self.jacobians_intialized.jacobians_from_vertices(self.template_vertices_orig.unsqueeze(0)).requires_grad_(True)

        self.template_vertices_remeshed = self.template_vertices_orig.clone()
        self.template_faces_remeshed = self.template_faces_orig.clone()
    
        self.template_face_normals_orig = calculate_face_normals(self.template_vertices_orig, self.template_faces_orig)
        self.template_face_centers_orig = calculate_face_centers(self.template_vertices_orig, self.template_faces_orig)

        self.template_cloth = Cloth(self.template_vertices_orig,self.template_faces_orig)

        self.cloth_template = Template(
            sources_vertices = self.template_vertices_orig,
            source_faces = self.template_faces_orig,
            garment_skinning = self.garment_skinning_orig,
            face_normals = self.template_face_normals_orig,
            face_centers = self.template_face_centers_orig,
            cloth_template = self.template_cloth
        )

        self.template_garment_skinning = self.garment_skinning_orig
        self.template_face_normal = self.template_face_normals_orig
        self.template_face_centers = self.template_face_centers_orig
        self.cannonical_faces = self.template_faces_orig
        self.cannonical_verts = self.template_vertices_orig

        self.jacobian_optimizer_cannonical = None
        self.jacobian_scheduler_cannonical = None

        ##Gradients Informations
        self.face_gradients_acc = torch.zeros((self.template_faces_orig.shape[0], 1), device = self.device)
        self.face_gradients_count = torch.zeros((self.template_faces_orig.shape[0], 1), device = self.device)
        
        ##Training Informations
        self.stop_train_after_warmup = cfg.stop_train_after_warmup
        self.use_grad_clipping = cfg.use_grad_clipping
        self.warm_ups = cfg.warm_ups
        self.position_lr_init = cfg.position_lr_init
        self.position_lr_final = cfg.position_lr_final
        self.position_lr_delay_mult = cfg.position_lr_delay_mult
        self.deform_lr_max_steps = cfg.deform_lr_max_steps
        self.position_lr_max_steps = cfg.position_lr_max_steps
        self.jacobian_optimizer_cannonical = None
        self.custom_training = cfg.custom_training_cannonical
        self.create_new_optimizer = cfg.create_new_optimizer
        self.interpolate_new_optimizer = cfg.interpolate_new_optimizer

        self.output_path = os.path.join(cfg.output_path, cfg.Exp_name)
        tri_mesh = trimesh.Trimesh(vertices = self.template_vertices_orig.detach().cpu().numpy(), \
            faces = self.template_faces_orig.detach().cpu().numpy())
        tri_mesh.export(os.path.join(self.output_path, f'source_mesh.obj'))
        self.delta = torch.zeros(3, device = self.device)
        self.remesh_step = 0
        self.max_verts = cfg.max_verts
        self.min_edge_len = get_linear_interpolation_func(cfg.start_edge_len, cfg.end_edge_len)
        self.remesh_percentile = get_linear_interpolation_func(cfg.start_percentile, cfg.end_percentile)

    @property
    def get_cannonical_jacobians(self):
        return self.cannonical_jacobians

    def training_init(self):

        # self.cannonical_jacobians.requires_grad_(True)
        if self.custom_training:
            self.lr = self.position_lr_init * self.spatial_lr_scale
            l = [
                {'params': [self.cannonical_jacobians], 'lr': self.position_lr_init * self.spatial_lr_scale, "name": "face_deform"},
            ]

            self.jacobian_optimizer_cannonical = torch.optim.Adam(l, lr=0.0, eps=1e-15)
            self.optimizer_scheduler_args = get_expon_lr_func(lr_init=self.position_lr_init * self.spatial_lr_scale,
                                                        lr_final=self.position_lr_final * self.spatial_lr_scale,
                                                        lr_delay_mult=self.position_lr_delay_mult,
                                                        max_steps=self.position_lr_max_steps)

        else:
            self.jacobian_optimizer_cannonical = torch.optim.Adam([self.cannonical_jacobians], lr = 1e-3)
            # self.jacobian_scheduler_gt = torch.optim.lr_scheduler.LambdaLR(jacobian_optimizer,lr_lambda=lambda x: max(0.01, 10**(-x*0.05)))
              
    def step(self, epochs, iterations, total_frames):
        
        # if epochs == 3:
        #     breakpoint()
        if self.custom_training:
            self.jacobian_optimizer_cannonical.step()
            self.update_learning_rate((epochs+1)*total_frames + iterations)
        else:
            if self.stop_train_after_warmup and epochs < self.warm_ups:
                self.jacobian_optimizer_cannonical.step()
            elif self.stop_train_after_warmup:
                pass 
            else :
                self.jacobian_optimizer_cannonical.step()

    def get_mesh_attr_from_jacobians(self, jacobians, detach = False):

        # self.cannonical_verts = self.jacobians_remeshed.vertices_from_jacobians(jacobians).squeeze()
        # self.cannonical_verts = self.cannonical_verts + torch.mean(self.template_vertices_orig.detach(), axis=0).repeat(self.cannonical_verts.shape[0],1)
        # self.cannonical_faces = self.template_faces_remeshed
        # return self.cannonical_verts, self.cannonical_faces
        
        cannonical_verts = self.jacobians_remeshed.vertices_from_jacobians(jacobians).squeeze()
        # breakpoint()
        # cannonical_verts = cannonical_verts + (torch.mean(self.template_vertices_orig.detach(), axis=0) + self.delta).repeat(cannonical_verts.shape[0],1)
        cannonical_verts = cannonical_verts + (torch.mean(self.template_vertices_remeshed, axis=0)).repeat(cannonical_verts.shape[0],1)
        cannonical_faces = self.template_faces_remeshed

        return cannonical_verts, cannonical_faces

    def remesh_legacy(self, epoch):

        print(f"Starting to Remesh {epoch}")
        self.remesh_dir = os.path.join(self.output_path, 'remeshed_template')
        os.makedirs(self.remesh_dir, exist_ok=True)
                
        self.template_vertices_remeshed_tmp, self.template_faces_remeshed_tmp = adaptive_remesh(self.cannonical_verts, self.template_faces_remeshed,\
            vertices_orig = self.template_vertices_remeshed, faces_orig = self.template_faces_remeshed)

        self.template_vertices_remeshed_tmp, self.template_faces_remeshed_tmp = \
            merge_close_vertices(self.template_vertices_remeshed_tmp, self.template_faces_remeshed_tmp)
        
        tri_mesh = trimesh.Trimesh(vertices = self.template_vertices_remeshed_tmp.detach().cpu().numpy(), \
            faces = self.template_faces_remeshed_tmp.detach().cpu().numpy(), process=False)
        tri_mesh.export(os.path.join(self.remesh_dir, f'remeshed_{epoch}.obj'))
        tri_mesh.export(os.path.join(self.remesh_dir, f'source_mesh.obj'))   

        self.jacobians_remeshed = SourceMesh.SourceMesh(0, os.path.join(self.remesh_dir, f'remeshed_{epoch}.obj'), {}, 1, ttype=torch.float)
        self.jacobians_remeshed.load(make_new = True)
        self.jacobians_remeshed.to(self.device)

        self.interpolated_jacobians = jacobian_interpolation(self.cannonical_jacobians.detach(), self.template_vertices_remeshed_tmp,\
            self.template_vertices_remeshed, self.template_faces_remeshed_tmp, self.template_faces_remeshed)
        
        
        self.cannonical_jacobians = self.interpolated_jacobians.clone().detach()
        self.cannonical_jacobians.requires_grad_(True)  
        # self.jacobian_optimizer_cannonical = torch.optim.Adam([self.cannonical_jacobians], lr = 1e-3)
                    
        if self.custom_training:
            l = [
                {'params': [self.cannonical_jacobians], 'lr': self.lr, "name": "face_deform"},
            ]
            self.jacobian_optimizer_cannonical = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        else:
            self.jacobian_optimizer_cannonical = torch.optim.Adam([self.cannonical_jacobians], lr = 1e-3)
            # self.jacobian_scheduler_gt = torch.optim.lr_scheduler.LambdaLR(jacobian_optimizer,lr_lambda=lambda x: max(0.01, 10**(-x*0.05)))

        self.cannonical_verts = self.jacobians_remeshed.vertices_from_jacobians(self.cannonical_jacobians.detach()).squeeze()
        self.cannonical_verts = self.cannonical_verts + torch.mean(self.template_vertices_orig.detach(), axis=0).unsqueeze(0).repeat(self.cannonical_verts.shape[0],1)
        
        self.template_garment_skinning = garment_skinning_interpolation(self.template_vertices_remeshed_tmp,\
                                                                        self.template_garment_skinning, self.template_vertices_remeshed)
        self.template_face_normals = calculate_face_normals(self.template_vertices_remeshed_tmp, \
                                                            self.template_faces_remeshed_tmp)
        self.template_face_centers = calculate_face_centers(self.template_vertices_remeshed_tmp, \
                                                            self.template_faces_remeshed_tmp)
        self.cloth_template.update(
            sources_vertices = self.template_vertices_remeshed_tmp,
            source_faces = self.template_faces_remeshed_tmp,
            garment_skinning = self.template_garment_skinning,
            face_normals = self.template_face_normals,
            face_centers = self.template_face_centers,
            cloth_template = self.template_cloth
        )

        self.template_vertices_remeshed = self.template_vertices_remeshed_tmp
        self.template_faces_remeshed = self.template_faces_remeshed_tmp
        self.cannonical_faces = self.template_faces_remeshed

    def save_jacobians(self, output_path, epoch = None):
        if epoch is not None:
            torch.save(self.cannonical_jacobians, os.path.join(output_path,f'jacobians_epoch_{epoch}.pt'))
        else:
            torch.save(self.cannonical_jacobians, os.path.join(output_path,'jacobians.pt'))
    
    def save_cannonical_mesh(self, output_path, epoch = None):
        
            V,F = self.get_mesh_attr_from_jacobians(self.cannonical_jacobians.detach())
            if epoch is not None:
                tri_mesh = trimesh.Trimesh(vertices = V.detach().cpu().numpy(), faces = F.detach().cpu().numpy())
                tri_mesh.export(os.path.join(output_path,f'cannonical_epoch_{epoch}.obj')) 
            else : 
                tri_mesh = trimesh.Trimesh(vertices = V.detach().cpu().numpy(), faces = F.detach().cpu().numpy())
                tri_mesh.export(os.path.join(output_path,f'cannonical_final.obj')) 

    def save_cannonical_mesh(self, output_path, epoch = None):
        
        torch.save(self.delta, os.path.join(output_path,'delta_pos.pt'))

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.jacobian_optimizer_cannonical.param_groups:
            if param_group["name"] == "face_deform":
                self.lr = self.optimizer_scheduler_args(iteration)
                param_group['lr'] = self.lr
                return self.lr

    def add_new_elements_optimizer(self, optimizer_tensor, mask, orig_vertices, changed_vertices, orig_faces, changed_faces):

        mask = mask.squeeze(-1).to(torch.bool)
        prune_faces_num = mask[mask == True].shape[0]
        optimizable_tensors = {}
        if self.create_new_optimizer:
            del self.jacobian_optimizer_cannonical
            self.cannonical_jacobians = nn.Parameter(optimizer_tensor.requires_grad_(True))
            l = [
                {'params': [self.cannonical_jacobians], 'lr': self.lr, "name": "face_deform"},
            ]
            self.jacobian_optimizer_cannonical = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        elif self.interpolate_new_optimizer:
            for group in self.jacobian_optimizer_cannonical.param_groups:
                assert len(group["params"]) == 1
                stored_state = self.jacobian_optimizer_cannonical.state.get(group['params'][0], None)
                if stored_state is not None:
                    
                    stored_state["exp_avg"] = optimizer_interpolation(stored_state["exp_avg"], orig_vertices, changed_vertices, \
                                            orig_faces, changed_faces, k = 1)
                    stored_state["exp_avg_sq"] = optimizer_interpolation(stored_state["exp_avg_sq"], orig_vertices, changed_vertices, \
                                            orig_faces, changed_faces, k = 1)

                    # stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"][:,~mask, ...], torch.zeros((a, diff, b, c), dtype = torch.float32, device = stored_state["exp_avg"].device)),
                    #                                     dim=1)
                    # stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"][:,~mask, ...], torch.zeros((a,diff,b,c), dtype = torch.float32, device = stored_state["exp_avg_sq"].device)),
                    #                                     dim=1)
                    del self.jacobian_optimizer_cannonical.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(optimizer_tensor.requires_grad_(True))
                    self.jacobian_optimizer_cannonical.state[group['params'][0]] = stored_state
                    self.cannonical_jacobians = group["params"][0]
        else:
            for group in self.jacobian_optimizer_cannonical.param_groups:
                # breakpoint()
                assert len(group["params"]) == 1
                stored_state = self.jacobian_optimizer_cannonical.state.get(group['params'][0], None)
                diff = optimizer_tensor.shape[1] - group['params'][0].shape[1] + prune_faces_num
                a,_,b,c = optimizer_tensor.shape[0], optimizer_tensor.shape[1], optimizer_tensor.shape[2], optimizer_tensor.shape[3]

                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"][:,~mask, ...], torch.zeros((a, diff, b, c), dtype = torch.float32, device = stored_state["exp_avg"].device)),
                                                        dim=1)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"][:,~mask, ...], torch.zeros((a,diff,b,c), dtype = torch.float32, device = stored_state["exp_avg_sq"].device)),
                                                        dim=1)
                    # breakpoint()
                    del self.jacobian_optimizer_cannonical.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(optimizer_tensor.requires_grad_(True))
                    self.jacobian_optimizer_cannonical.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    raise NotImplementedError('Not implemented')
                    # group["params"][0] = nn.Parameter(
                    #     torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    # optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def remesh(self, epoch, grads, max_percentile = 0.9):

        min_val = grads.min()
        max_val = grads.max()

        max_percentile = self.remesh_percentile(self.remesh_step)
        min_edge_length = self.min_edge_len(self.remesh_step)
        grads = (grads - min_val) / (max_val - min_val)
        threshold = torch.quantile(grads, max_percentile)
        face_mask = torch.zeros_like(grads)
        face_mask[grads > threshold] = 1.0

        face_mask = face_mask.to(torch.long)
        self.remesh_dir = os.path.join(self.output_path, 'remeshed_template')
        os.makedirs(self.remesh_dir, exist_ok=True)

        self.template_vertices_remeshed_tmp, self.template_faces_remeshed_tmp, face_mask = remesh(self.template_vertices_remeshed, self.template_faces_remeshed, face_mask, \
            flip = True, max_vertices = self.max_verts, threshold= min_edge_length)
        self.template_vertices_remeshed_tmp, self.template_faces_remeshed_tmp = \
            merge_close_vertices(self.template_vertices_remeshed_tmp, self.template_faces_remeshed_tmp)
        
        tri_mesh = trimesh.Trimesh(vertices = self.template_vertices_remeshed_tmp.detach().cpu().numpy(), faces = self.template_faces_remeshed_tmp.detach().cpu().numpy())
        tri_mesh.export(os.path.join(self.remesh_dir, f'remeshed_{epoch}.obj'))
        tri_mesh.export(os.path.join(self.remesh_dir, f'source_mesh.obj'))   
        
        self.jacobians_remeshed = SourceMesh.SourceMesh(0, os.path.join(self.remesh_dir, f'source_mesh.obj'), {}, 1, ttype=torch.float)
        self.jacobians_remeshed.load(make_new = True)
        self.jacobians_remeshed.to(self.device)

        self.delta = torch.mean(self.template_vertices_remeshed_tmp, axis=0)  - torch.mean(self.template_vertices_remeshed, axis=0) + self.delta
        new_jacobians = jacobian_interpolation(self.cannonical_jacobians.detach(), self.template_vertices_remeshed_tmp,\
            self.template_vertices_remeshed, self.template_faces_remeshed_tmp, self.template_faces_remeshed)
        
        self.template_garment_skinning = garment_skinning_interpolation(self.template_vertices_remeshed_tmp,\
                                                                        self.template_garment_skinning, self.template_vertices_remeshed)
        
        self.template_garment_skinning = compute_rbf_skinning_weight(self.template_vertices_remeshed_tmp, self.body)
        self.template_face_normals = calculate_face_normals(self.template_vertices_remeshed_tmp, \
                                                            self.template_faces_remeshed_tmp)
        self.template_face_centers = calculate_face_centers(self.template_vertices_remeshed_tmp, \
                                                            self.template_faces_remeshed_tmp)
        self.cloth_template.update(
            sources_vertices = self.template_vertices_remeshed_tmp,
            source_faces = self.template_faces_remeshed_tmp,
            garment_skinning = self.template_garment_skinning,
            face_normals = self.template_face_normals,
            face_centers = self.template_face_centers,
            cloth_template = self.template_cloth
        )

        self.add_new_elements_optimizer(new_jacobians, face_mask, self.template_vertices_remeshed_tmp,self.template_vertices_remeshed, self.template_faces_remeshed_tmp, self.template_faces_remeshed)
        self.template_vertices_remeshed = self.template_vertices_remeshed_tmp
        self.template_faces_remeshed = self.template_faces_remeshed_tmp
        self.cannonical_faces = self.template_faces_remeshed
        
        self.face_gradients_acc = torch.zeros((self.template_faces_remeshed.shape[0], 1), device = self.device)
        self.face_gradients_count = torch.zeros((self.template_faces_remeshed.shape[0], 1), device = self.device) 
        self.remesh_step += 1

    def remesh_triangles(self,epoch, max_percentile = 0.9):

        grads = self.face_gradients_acc / self.face_gradients_count
        grads[grads.isnan()] = 0.0
        self.remesh(epoch, grads, max_percentile)
        torch.cuda.empty_cache()

    @torch.no_grad
    def save_img_grad(self, img_grads, img_shil, rast_out):

        ## Check whether this normalisation works. Make them positive
        img_grads_norm = img_grads.norm(p = 2 , dim = -1, keepdim = False)
        # img_shil = img_shil.mean(dim = -1)
        # img_grads_norm = torch.mul(img_grads_norm,img_shil.squeeze(-1))
        min_val = img_grads_norm.min()
        max_val = img_grads_norm.max()
        
        normalized_img_tensor = (img_grads_norm - min_val) / (max_val - min_val)
        # breakpoint()
        # save_gradient_image(normalized)
        visible_faces = torch.unique(rast_out[...,3]) - 1
        visible_faces = visible_faces[visible_faces >= 0].to(torch.long)
        chunk_size = 500 
        
        for chunk_start in range(0, len(visible_faces), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(visible_faces))
            unique_chunk = visible_faces[chunk_start:chunk_end]

            masks = (rast_out[..., -1].unsqueeze(-1) - 1) == unique_chunk.unsqueeze(0)
            masked_values = torch.where(masks, normalized_img_tensor.unsqueeze(-1), torch.tensor(0.0, device=self.device))
            chunk_summed_values = torch.sum(masked_values,dim=(0,1,2))
            chunk_occurrence_counts = torch.sum(masks, dim=(0,1,2))

            average_grad = chunk_summed_values / chunk_occurrence_counts
            self.face_gradients_acc[unique_chunk] += average_grad.unsqueeze(-1)
            self.face_gradients_count[unique_chunk] += 1

        del masks, masked_values, chunk_summed_values, chunk_occurrence_counts
        
        # self.face_gradients_acc[visible_faces] += average_grad.unsqueeze(-1)
        # self.face_gradients_count[visible_faces] += 1

    

