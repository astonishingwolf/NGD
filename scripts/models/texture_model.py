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

class HashGrid_w_pose(nn.Module):
    def __init__(self, config, template, multires=10, device = 'cuda'):  
        super(HashGrid_w_pose, self).__init__()
        self.device = device
        self.template = template
        self.config = config
        self.t_multires = 10
        self.uv_encoding = tcnn.Encoding(4, hash_cfg["encoding"])
        # self.normal_encoding = tcnn.Encoding(3, hash_cfg["encoding"])
        # self.pose_encoding = tcnn.Encoding(4, hash_cfg["encoding"])
        self.network = tcnn.Network(self.position_encoding.n_output_dims + self.normal_encoding.n_output_dims + self.pose_encoding.n_output_dims, 9, hash_cfg["network"])

    def forward(self, inputs) -> torch.Tensor:

        pos_encoded = self.uv_encoding(torch.cat((inputs.img_pixel_indices, inputs.pose_extended), dim=1))
        uv_tex_dynamic = self.network(torch.cat((pos_encoded), dim=1))
        
        return uv_tex_dynamic


class TextureModel(nn.Module):

    def __init__(self, config,template):

        super(Model, self).__init__()
        self.config = config
        if config.model == 'hashgrid_base':
            self.model = HashGrid_w_pose(config, template)

    def forward(self, inputs)   -> torch.Tensor:

        texture = self.model(inputs)
        return texture