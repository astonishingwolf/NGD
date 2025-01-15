import torch
import nvdiffrast.torch as dr

def _translation(x, y, z, device):
    return torch.tensor([[1., 0, 0, x],
                    [0, 1, 0, y],
                    [0, 0, 1, z],
                    [0, 0, 0, 1]],device=device) #4,4

def _projection(r, device, l=None, t=None, b=None, n=1.0, f=50.0, flip_y=True):
    if l is None:
        l = -r
    if t is None:
        t = r
    if b is None:
        b = -t
    p = torch.zeros([4,4],device=device)
    p[0,0] = 2*n/(r-l)
    p[0,2] = (r+l)/(r-l)
    p[1,1] = 2*n/(t-b) * (-1 if flip_y else 1)
    p[1,2] = (t+b)/(t-b)
    p[2,2] = -(f+n)/(f-n)
    p[2,3] = -(2*f*n)/(f-n)
    p[3,2] = -1
    return p #4,4

def make_star_cameras(az_count,pol_count,distance:float=10.,r=None, n=None, f=None, image_size=[512,512],device='cuda'):
    if r is None:
        r = 1/distance
    if n is None:
        n = 1
    if f is None:
        f = 50
    A = az_count
    P = pol_count
    C = A * P

    phi = torch.arange(0,A) * (2*torch.pi/A)
    phi_rot = torch.eye(3,device=device)[None,None].expand(A,1,3,3).clone()
    phi_rot[:,0,2,2] = phi.cos()
    phi_rot[:,0,2,0] = -phi.sin()
    phi_rot[:,0,0,2] = phi.sin()
    phi_rot[:,0,0,0] = phi.cos()
    
    theta = torch.arange(1,P+1) * (torch.pi/(P+1)) - torch.pi/2
    theta_rot = torch.eye(3,device=device)[None,None].expand(1,P,3,3).clone()
    theta_rot[0,:,1,1] = theta.cos()
    theta_rot[0,:,1,2] = -theta.sin()
    theta_rot[0,:,2,1] = theta.sin()
    theta_rot[0,:,2,2] = theta.cos()

    mv = torch.empty((C,4,4), device=device)
    mv[:] = torch.eye(4, device=device)
    mv[:,:3,:3] = (theta_rot @ phi_rot).reshape(C,3,3)
    # mv[:,:3,:3] = (phi_rot @ theta_rot).reshape(C,3,3)
    mv = _translation(0, 0, -distance, device) @ mv

    return mv, _projection(r,device, n=n, f=f)

def _warmup(glctx):
    #windows workaround for https://github.com/NVlabs/nvdiffrast/issues/59
    def tensor(*args, **kwargs):
        return torch.tensor(*args, device='cuda', **kwargs)
    pos = tensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]], dtype=torch.float32)
    tri = tensor([[0, 1, 2]], dtype=torch.int32)
    dr.rasterize(glctx, pos, tri, resolution=[512, 512])

class NormalsRenderer:
    
    _glctx:dr.RasterizeGLContext = None
    
    def __init__(
            self,
            mv: torch.Tensor, #C,4,4
            proj: torch.Tensor, #C,4,4
            image_size: "tuple[int,int]",
            ):
        self._mvp = proj @ mv #C,4,4
        self._image_size = image_size
        self._glctx = dr.RasterizeCudaContext()
        _warmup(self._glctx)

    def render(self,
            vertices: torch.Tensor, #V,3 float
            normals: torch.Tensor, #V,3 float
            faces: torch.Tensor, #F,3 long
            ) ->torch.Tensor: #C,H,W,4

        V = vertices.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((vertices, torch.ones(V,1,device=vertices.device)),axis=-1) #V,3 -> V,4
        vertices_clip = vert_hom @ self._mvp.transpose(-2,-1) #C,V,4
        rast_out,_ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=self._image_size, grad_db=False) #C,H,W,4
        vert_col = (normals+1)/2 #V,3
        col,_ = dr.interpolate(vert_col, rast_out, faces) #C,H,W,3
        alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1
        col = torch.concat((col,alpha),dim=-1) #C,H,W,4
        col = dr.antialias(col, rast_out, vertices_clip, faces) #C,H,W,4
        return col #C,H,W,4

def calc_face_normals(
        vertices:torch.Tensor, #V,3 first vertex may be unreferenced
        faces:torch.Tensor, #F,3 long, first face may be all zero
        normalize:bool=False,
        )->torch.Tensor: #F,3
    """
         n
         |
         c0     corners ordered counterclockwise when
        / \     looking onto surface (in neg normal direction)
      c1---c2
    """
    full_vertices = vertices[faces] #F,C=3,3
    v0,v1,v2 = full_vertices.unbind(dim=1) #F,3
    face_normals = torch.cross(v1-v0,v2-v0, dim=1) #F,3
    if normalize:
        face_normals = torch.nn.functional.normalize(face_normals, eps=1e-6, dim=1) #TODO inplace?
    return face_normals #F,3

def calc_vertex_normals(
        vertices:torch.Tensor, #V,3 first vertex may be unreferenced
        faces:torch.Tensor, #F,3 long, first face may be all zero
        face_normals:torch.Tensor=None, #F,3, not normalized
        )->torch.Tensor: #F,3

    F = faces.shape[0]

    if face_normals is None:
        face_normals = calc_face_normals(vertices,faces)
    
    vertex_normals = torch.zeros((vertices.shape[0],3,3),dtype=vertices.dtype,device=vertices.device) #V,C=3,3
    vertex_normals.scatter_add_(dim=0,index=faces[:,:,None].expand(F,3,3),src=face_normals[:,None,:].expand(F,3,3))
    vertex_normals = vertex_normals.sum(dim=1) #V,3
    # 
    return torch.nn.functional.normalize(vertex_normals, eps=1e-6, dim=1)