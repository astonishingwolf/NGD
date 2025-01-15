# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import pathlib
import sys
import numpy as np
import torch

import xatlas
import nvdiffrast.torch as dr


def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return  posw @ t_mtx.transpose(-2,-1)

def render_texture(glctx, mtx, pos, pos_idx, uv, uv_idx, tex, resolution, enable_mip, max_mip_level):
    pos_clip = transform_pos(mtx, pos.squeeze(0))
    # breakpoint()
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])
    # breakpoint()
    if enable_mip:
        texc, texd = dr.interpolate(uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
        color = dr.texture(tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
    else:
        texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
        color = dr.texture(tex[None, ...], texc, filter_mode='linear')
    # breakpoint()
    color = color * torch.clamp(rast_out[..., -1:], 0, 1) # Mask out background.
    return color
    
@torch.no_grad()
def xatlas_uvmap(v_pos, faces):

    v_pos = v_pos.detach().cpu().numpy()
    faces = faces.detach().cpu().numpy()
    
    _, indices, uvs = xatlas.parametrize(v_pos, faces)
    indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
    # breakpoint()
    # vmapping = torch.tensor(vmapping, dtype = torch.float32, device='cuda')
    uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
    indices_tensor = torch.tensor(indices_int64, dtype=torch.int64, device='cuda') 

    return indices_tensor, uvs

def update_texture_learning_rate(self, optimizer,iteration):
    for param_group in optimizer:
        if param_group["name"] == "texture":
            lr = optimizer_scheduler_args(iteration)
            param_group['lr'] = lr