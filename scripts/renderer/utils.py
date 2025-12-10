"""
Utility functions for rendering operations.
"""
import torch
import nvdiffrast.torch as dr


def _translation(x: float, y: float, z: float, device: str) -> torch.Tensor:
    """Create a 4x4 translation matrix."""
    return torch.tensor([
        [1., 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ], device=device)


def _projection(r: float, device: str, l: float = None, t: float = None, b: float = None,
                n: float = 1.0, f: float = 50.0, flip_y: bool = True) -> torch.Tensor:
    """Create a 4x4 projection matrix."""
    if l is None:
        l = -r
    if t is None:
        t = r
    if b is None:
        b = -t
    
    p = torch.zeros([4, 4], device=device)
    p[0, 0] = 2 * n / (r - l)
    p[0, 2] = (r + l) / (r - l)
    p[1, 1] = 2 * n / (t - b) * (-1 if flip_y else 1)
    p[1, 2] = (t + b) / (t - b)
    p[2, 2] = -(f + n) / (f - n)
    p[2, 3] = -(2 * f * n) / (f - n)
    p[3, 2] = -1
    return p


def make_star_cameras(az_count: int, pol_count: int, distance: float = 10.0, r: float = None,
                     n: float = None, f: float = None, image_size: list = [512, 512],
                     device: str = 'cuda'):
    """Generate star-pattern camera configurations."""
    r = r if r is not None else 1 / distance
    n = n if n is not None else 1
    f = f if f is not None else 50
    
    A, P = az_count, pol_count
    C = A * P
    
    # Azimuth rotations
    phi = torch.arange(0, A, device=device) * (2 * torch.pi / A)
    phi_rot = torch.eye(3, device=device)[None, None].expand(A, 1, 3, 3).clone()
    phi_rot[:, 0, 2, 2] = phi.cos()
    phi_rot[:, 0, 2, 0] = -phi.sin()
    phi_rot[:, 0, 0, 2] = phi.sin()
    phi_rot[:, 0, 0, 0] = phi.cos()
    
    # Polar rotations
    theta = torch.arange(1, P + 1, device=device) * (torch.pi / (P + 1)) - torch.pi / 2
    theta_rot = torch.eye(3, device=device)[None, None].expand(1, P, 3, 3).clone()
    theta_rot[0, :, 1, 1] = theta.cos()
    theta_rot[0, :, 1, 2] = -theta.sin()
    theta_rot[0, :, 2, 1] = theta.sin()
    theta_rot[0, :, 2, 2] = theta.cos()
    
    # Combine rotations
    mv = torch.empty((C, 4, 4), device=device)
    mv[:] = torch.eye(4, device=device)
    mv[:, :3, :3] = (theta_rot @ phi_rot).reshape(C, 3, 3)
    mv = _translation(0, 0, -distance, device) @ mv
    
    return mv, _projection(r, device, n=n, f=f)


def _warmup(glctx: dr.RasterizeCudaContext):
    """Warmup rasterization context (Windows workaround for nvdiffrast issue #59)."""
    def tensor(*args, **kwargs):
        return torch.tensor(*args, device='cuda', **kwargs)
    pos = tensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]], dtype=torch.float32)
    tri = tensor([[0, 1, 2]], dtype=torch.int32)
    dr.rasterize(glctx, pos, tri, resolution=[512, 512])


class NormalsRenderer:
    """Base renderer for normal map visualization."""
    
    def __init__(self, mv: torch.Tensor, proj: torch.Tensor, image_size: tuple[int, int]):
        """Initialize base renderer.
        
        Args:
            mv: Model-view matrix [C, 4, 4]
            proj: Projection matrix [C, 4, 4]
            image_size: Output image dimensions (height, width)
        """
        self._mvp = proj @ mv
        self._image_size = image_size
        self._glctx = dr.RasterizeCudaContext()
        _warmup(self._glctx)
    
    def render(self, vertices: torch.Tensor, normals: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """Render normal map.
        
        Args:
            vertices: Vertex positions [V, 3]
            normals: Vertex normals [V, 3]
            faces: Face indices [F, 3]
            
        Returns:
            Rendered image [C, H, W, 4]
        """
        V = vertices.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat([vertices, torch.ones(V, 1, device=vertices.device)], dim=-1)
        vertices_clip = vert_hom @ self._mvp.transpose(-2, -1)
        rast_out, _ = dr.rasterize(
            self._glctx, vertices_clip, faces, resolution=self._image_size, grad_db=False
        )
        vert_col = (normals + 1) / 2
        col, _ = dr.interpolate(vert_col, rast_out, faces)
        alpha = torch.clamp(rast_out[..., -1:], max=1)
        col = torch.concat([col, alpha], dim=-1)
        col = dr.antialias(col, rast_out, vertices_clip, faces)
        return col


def calc_face_normals(vertices: torch.Tensor, faces: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """Calculate face normals from vertices and faces.
    
    Args:
        vertices: Vertex positions [V, 3]
        faces: Face indices [F, 3]
        normalize: Whether to normalize the normals
        
    Returns:
        Face normals [F, 3]
    """
    full_vertices = vertices[faces]
    v0, v1, v2 = full_vertices.unbind(dim=1)
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    if normalize:
        face_normals = torch.nn.functional.normalize(face_normals, eps=1e-6, dim=1)
    return face_normals


def calc_vertex_normals(vertices: torch.Tensor, faces: torch.Tensor,
                       face_normals: torch.Tensor = None) -> torch.Tensor:
    """Calculate vertex normals by averaging adjacent face normals.
    
    Args:
        vertices: Vertex positions [V, 3]
        faces: Face indices [F, 3]
        face_normals: Pre-computed face normals [F, 3] (optional)
        
    Returns:
        Vertex normals [V, 3]
    """
    if face_normals is None:
        face_normals = calc_face_normals(vertices, faces)
    
    F = faces.shape[0]
    vertex_normals = torch.zeros((vertices.shape[0], 3, 3), dtype=vertices.dtype, device=vertices.device)
    vertex_normals.scatter_add_(
        dim=0, index=faces[:, :, None].expand(F, 3, 3),
        src=face_normals[:, None, :].expand(F, 3, 3)
    )
    vertex_normals = vertex_normals.sum(dim=1)
    return torch.nn.functional.normalize(vertex_normals, eps=1e-6, dim=1)
