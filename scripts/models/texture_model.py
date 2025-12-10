import torch
import torch.nn as nn
import os
import tinycudann as tcnn

from scripts.models.model_utils import get_embedder
from scripts.utils.config_utils import load_texture_config

hash_cfg = load_texture_config()

class HashGrid_w_pose(nn.Module):
    def __init__(self, config, multires=10, device = 'cuda'):  
        super(HashGrid_w_pose, self).__init__()
        self.device = device
        self.config = config
        self.t_multires = 10
        self.uv_encoding = tcnn.Encoding(4, hash_cfg["encoding"])
        # self.normal_encoding = tcnn.Encoding(3, hash_cfg["encoding"])
        # self.pose_encoding = tcnn.Encoding(4, hash_cfg["encoding"])
        self.texture_network = tcnn.Network(self.uv_encoding.n_output_dims, 3, hash_cfg["network"])

    def forward(self, inputs) -> torch.Tensor:

        # breakpoint()
        input_coords = torch.cat((inputs.img_pixel_indices, inputs.pose_extended), dim=-1).view(-1, 4)
        pos_encoded = self.uv_encoding(input_coords)
        uv_tex_dynamic = self.texture_network(pos_encoded)
        # breakpoint()
        return uv_tex_dynamic

class HashGrid_w_Reg(nn.Module):
    def __init__(self, config, multires=10, device = 'cuda'):  
        super(HashGrid_w_Reg, self).__init__()
        self.device = device
        self.config = config
        self.t_multires = 10
        self.uv_encoding = tcnn.Encoding(4, hash_cfg["encoding"])
        self.embed_uv_fn, self.uv_ch = get_embedder(self.t_multires, 2)
        # self.normal_encoding = tcnn.Encoding(3, hash_cfg["encoding"])
        # self.pose_encoding = tcnn.Encoding(4, hash_cfg["encoding"])
        self.uv_mlp = tcnn.Network(self.uv_ch, 12, hash_cfg["tex_network"])
        self.texture_network = tcnn.Network(self.uv_encoding.n_output_dims + 12, 3, hash_cfg["network"])

    def forward(self, inputs) -> torch.Tensor:

        # breakpoint()
        uv_encoded = self.embed_uv_fn(inputs.img_pixel_indices.view(-1,2))
        uv_mlp_out = self.uv_mlp(uv_encoded)
        input_coords = torch.cat((inputs.img_pixel_indices, inputs.pose_extended), dim=-1).view(-1, 4)
        tex_encoded = self.uv_encoding(input_coords)
        uv_tex_dynamic = self.texture_network(torch.cat((tex_encoded, uv_mlp_out), dim=-1))
        
        return uv_tex_dynamic   


class TextureModel(nn.Module):

    def __init__(self, config,):

        super(TextureModel, self).__init__()
        self.config = config
        if config.texture_model == 'hashgrid_base':
            self.model = HashGrid_w_pose(config)
        elif config.texture_model == 'hashgrid_reg':
            self.model = HashGrid_w_Reg(config)

    def forward(self, inputs)   -> torch.Tensor:

        texture = self.model(inputs)
        return texture