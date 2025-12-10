"""
Texture rendering and UV mapping utilities.
"""
import numpy as np
import torch
import nvdiffrast.torch as dr
import xatlas


def transform_pos(mtx: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Transform vertex positions using transformation matrix.
    
    Args:
        mtx: Transformation matrix [4, 4] or [B, 4, 4]
        pos: Vertex positions [V, 3]
        
    Returns:
        Transformed positions [V, 3] or [B, V, 3]
    """
    if isinstance(mtx, np.ndarray):
        mtx = torch.from_numpy(mtx).cuda()
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1], device=pos.device)], dim=1)
    return posw @ mtx.transpose(-2, -1)


def render_texture(glctx: dr.RasterizeCudaContext, mtx: torch.Tensor, pos: torch.Tensor,
                  pos_idx: torch.Tensor, uv: torch.Tensor, uv_idx: torch.Tensor,
                  tex: torch.Tensor, resolution: int, enable_mip: bool, max_mip_level: int) -> torch.Tensor:
    """Render textured mesh.
    
    Args:
        glctx: Rasterization context
        mtx: Model-view-projection matrix
        pos: Vertex positions [V, 3]
        pos_idx: Position face indices
        uv: UV coordinates [U, 2]
        uv_idx: UV face indices
        tex: Texture map [H, W, 3]
        resolution: Output resolution
        enable_mip: Whether to enable mipmapping
        max_mip_level: Maximum mip level
        
    Returns:
        Rendered image [1, H, W, 3]
    """
    pos_clip = transform_pos(mtx, pos.squeeze(0))
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])
    
    if enable_mip:
        texc, texd = dr.interpolate(uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
        color = dr.texture(tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear',
                          max_mip_level=max_mip_level)
    else:
        texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
        color = dr.texture(tex[None, ...], texc, filter_mode='linear')
    
    color = color * torch.clamp(rast_out[..., -1:], 0, 1)  # Mask background
    return color


@torch.no_grad()
def xatlas_uvmap(v_pos: torch.Tensor, faces: torch.Tensor):
    """Generate UV mapping using xatlas.
    
    Args:
        v_pos: Vertex positions [V, 3]
        faces: Face indices [F, 3]
        
    Returns:
        Tuple of (indices, uvs, vmapping)
    """
    v_pos_np = v_pos.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()
    
    vmapping, indices, uvs = xatlas.parametrize(v_pos_np, faces_np)
    
    indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
    uvs_tensor = torch.tensor(uvs, dtype=torch.float32, device=v_pos.device)
    indices_tensor = torch.tensor(indices_int64, dtype=torch.int64, device=v_pos.device)
    
    return indices_tensor, uvs_tensor, vmapping

