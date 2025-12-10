import torch
import torch.nn as nn
import os
import tinycudann as tcnn

from scripts.models.core import diffusionnet as diffusion_net
from scripts.models.core.diffusionnet.geometry import get_operators
from scripts.models.core.siren.mlp import Siren
from scripts.models.model_utils import get_embedder
from scripts.utils.config_utils import load_hash_config

hash_cfg = load_hash_config()

class HashGrid_w_pose(nn.Module):
    def __init__(self, config, template, multires=10, device = 'cuda'):  
        super(HashGrid_w_pose, self).__init__()
        self.device = device
        self.template = template
        self.config = config
        self.t_multires = 10
        self.position_encoding = tcnn.Encoding(3, hash_cfg["encoding"])
        self.normal_encoding = tcnn.Encoding(3, hash_cfg["encoding"])
        self.pose_encoding = tcnn.Encoding(4, hash_cfg["encoding"])
        # self.pose_encoding, self.pose_encoding_ch = get_embedder(self.t_multires,4)
        self.network = tcnn.Network(self.position_encoding.n_output_dims + self.normal_encoding.n_output_dims + self.pose_encoding.n_output_dims, 9, hash_cfg["network"])
        # self.mlp = torch.nn.Sequential(self.position_encoding, self.normal_encoding, self.network)

    def forward(self, inputs) -> torch.Tensor:
        # breakpoint()
        pos_encoded = self.position_encoding(inputs.face_centers)
        # pos_encoded = self.position_encoding(torch.cat((inputs.face_centers, inputs.pose_extended), dim=1))
        normal_encoded = self.normal_encoding(inputs.face_normals)
        pose_encoded = self.pose_encoding(inputs.pose_extended)
        residual_jacobians = self.network(torch.cat((pos_encoded, normal_encoded,pose_encoded), dim=1))
        
        return residual_jacobians


class HashGrid(nn.Module):
    def __init__(self, config, template, device = 'cuda'):  
        super(HashGrid, self).__init__()
        self.device = device
        self.template = template
        self.config = config
        self.position_encoding = tcnn.Encoding(4, hash_cfg["encoding"])
        self.normal_encoding = tcnn.Encoding(3, hash_cfg["encoding"])
        self.network = tcnn.Network(self.position_encoding.n_output_dims + self.normal_encoding.n_output_dims, 9, hash_cfg["network"])
        # self.mlp = torch.nn.Sequential(self.position_encoding, self.normal_encoding, self.network)

    def forward(self, inputs) -> torch.Tensor:
        
        pos_encoded = self.position_encoding(torch.cat((inputs.face_centers, inputs.time_extended), dim=1))
        normal_encoding = self.normal_encoding(inputs.face_normals)
        residual_jacobians = self.network(torch.cat((pos_encoded, normal_encoding), dim=1))
        
        return residual_jacobians



class DiffusionModel(nn.Module):
    def __init__(self, config, template, device = 'cuda'):
        
        super(DiffusionModel, self).__init__()
        self.device = 'cuda'
        self.template = template
        self.config = config
        self.t_multires = 10
        self.time_input_ch = 21
        self.embed_time_fn, self.time_input_ch = get_embedder(self.t_multires, 1)
        # breakpoint()
        self.mlp = diffusion_net.layers.DiffusionNet(
                C_in= 3 + self.time_input_ch,
                C_out= 9,
                C_width= 128, # internal size of the diffusion net. 32 -- 512 is a reasonable range
                last_activation= None,
                outputs_at='faces')
        self.frames, self.mass, self.L, self.evals, self.evecs, self.gradX, self.gradY = \
                get_operators(template.sources_vertices.cpu(), template.source_faces.cpu(), \
                op_cache_dir='/home/nakul/soham/3d/Garment3DGen/cache')

    def forward(self, inputs) -> torch.Tensor:
        
        # self.embed_time_fn, self.time_input_ch = get_embedder(self.t_multires, 1)
        time_emb = self.embed_time_fn(inputs.time)
        time_extended = time_emb[None, ...].repeat(inputs.n_verts.shape[0], 1)

        residual_jacobians = self.mlp(torch.cat((inputs.n_verts, time_extended), dim=1), self.mass.to(self.device), \
            L = self.L.to(self.device), evals = self.evals.to(self.device), evecs = self.evecs.to(self.device), gradX = self.gradX.to(self.device), \
            gradY = self.gradY.to(self.device), faces = self.template.source_faces)
        
        return residual_jacobians



class GeneralSiren(nn.Module):
    def __init__(self, config, template):
        super(GeneralSiren, self).__init__()

        self.template = template
        self.config = config
        self.mlp = Siren(in_features = 10, out_features = 9, hidden_features = 256, 
                hidden_layers = self.config.hidden_layers, outermost_linear=True)

    def forward(self, inputs) -> torch.Tensor:
        
        residual_jacobians,_ = self.mlp(torch.cat((inputs.face_centers, inputs.face_normals,\
            inputs.pose_extended), dim=1))
        
        return residual_jacobians
    
class Model(nn.Module):

    def __init__(self, config,template):

        super(Model, self).__init__()
        self.config = config
        if config.model == 'diffusion':
            self.model = DiffusionModel(config, template)
        elif config.model == 'general':
            self.model = GeneralSiren(config, template)
        elif config.model == 'hashgrid':
            self.model = HashGrid(config, template)
        elif config.model == 'hashgrid_vd':
            self.model = HashGrid_w_pose(config, template)

    def forward(self, inputs)   -> torch.Tensor:

        residual_jacobians = self.model(inputs)
        return residual_jacobians