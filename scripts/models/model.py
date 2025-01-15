import torch
import torch.nn as nn
import os
import json
import tinycudann as tcnn
import commentjson as json

from scripts.models.core import diffusionnet as diffusion_net
from scripts.models.core.diffusionnet.geometry import get_operators
# from scripts.models.core.NeuralJacobianFields.MeshProcessor import WaveKernelSignature
from scripts.models.core.siren.mlp import Siren
from scripts.models.model_utils import get_embedder

with open("scripts/models/config/config_hash.json") as f:
	hash_cfg = json.load(f)

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


class SirenWKS(nn.Module):
    def __init__(self, config, template):
        super(SirenWKS, self).__init__()

        self.template = template
        self.config = config
        self.wave_function = WaveKernelSignature(template.sources_vertices.cpu().numpy(), template.source_faces.cpu().numpy())
        self.verts_wks = wave_function.compute()
        self.face_wks = wave_function.compute_on_triangles()
        self.face_wks_tensor = torch.from_numpy(face_wks).to(DEVICE).to(torch.float32)
        self.mlp = Siren(in_features = 32, out_features = 9, hidden_features = 256, 
                hidden_layers = 3, outermost_linear=True).to(DEVICE)

    def forward(self, inputs) -> torch.Tensor:
        
        residual_jacobians,_ = self.mlp(torch.cat((inputs.face_centers, inputs.face_normals,\
            self.face_wks_tensor, inputs.time_extended), dim=1))
        
        return self.layers(x)


class GeneralSiren(nn.Module):
    def __init__(self, config, template):
        super(GeneralSiren, self).__init__()

        self.template = template
        self.config = config
        self.mlp = Siren(in_features = 7, out_features = 9, hidden_features = 256, 
                hidden_layers = self.config.hidden_layers, outermost_linear=True)

    def forward(self, inputs) -> torch.Tensor:
        
        residual_jacobians,_ = self.mlp(torch.cat((inputs.face_centers, inputs.face_normals,\
            inputs.time_extended), dim=1))
        
        return residual_jacobians
    
class Model(nn.Module):

    def __init__(self, config,template):

        super(Model, self).__init__()
        self.config = config
        if config.model == 'sirenWKS':
            self.model = SirenWKS(config, template)
        elif config.model == 'diffusion':
            self.model = DiffusionModel(config, template)
        elif config.model == 'general':
            self.model = GeneralSiren(config, template)
        elif config.model == 'hashgrid':
            self.model = HashGrid(config, template)

    def forward(self, inputs) -> torch.Tensor:

        residual_jacobians = self.model(inputs)
        return residual_jacobians