import torch 
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import nvdiffrast.torch as dr
import imageio
from scripts.renderer.utils import *

LIGHT_DIR = [0., 0., 1.] 

class AlphaRenderer(NormalsRenderer):
    '''
    Renderer that renders
    * normal
    * depth
    * shillouette
    '''
    
    def __init__(
            self,
            mv: torch.Tensor, #C,4,4
            proj: torch.Tensor, #C,4,4
            image_size: "tuple[int,int]",
            ):
        super().__init__(mv,proj,image_size)
        self._mv = mv
        self._proj = proj
        self.eps = 1e-4

    def forward(self,
                verts: torch.Tensor,
                normals: torch.Tensor,
                faces: torch.Tensor):
        
        '''
        Single pass without transparency.
        '''
        V = verts.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((verts, torch.ones(V,1,device=verts.device)),axis=-1) #V,3 -> V,4
        verts_clip = vert_hom @ self._mvp.transpose(-2,-1) #C,V,4
        rast_out,_ = dr.rasterize(self._glctx, 
                                verts_clip, 
                                faces, 
                                resolution=self._image_size, 
                                grad_db=False) #C,H,W,4
        
        face_mask = torch.zeros_like(faces, dtype=torch.bool)    
        
        

        vert_normals_hom = torch.cat((normals, torch.zeros(V,1,device=verts.device)),axis=-1) #V,3 -> V,4
        vert_normals_view = vert_normals_hom @ self._mv.transpose(-2,-1) #C,V,4
        vert_normals_view = vert_normals_view[..., :3] #C,V,3
        # breakpoint    ()
        # in the view space, normals should be oriented toward viewer, 
        # so the z coordinates should be negative;
        # vert_normals_view[vert_normals_view[..., 2] > 0.] = \
        #     -vert_normals_view[vert_normals_view[..., 2] > 0.]
        vert_normals_view = vert_normals_view.contiguous()
        lightdir = torch.tensor(LIGHT_DIR, dtype=torch.float32, device=verts.device) #3
        lightdir = lightdir.view((1, 1, 1, 3)) #1,1,1,3


        pixel_normals_view, _ = dr.interpolate(vert_normals_view, rast_out, faces)  #C,H,W,3
        diffuse = torch.sum(lightdir * pixel_normals_view, -1, keepdim=True)           #C,H,W,1
        diffuse = torch.clamp(diffuse, min=0.0, max=1.0)
        front_mask = diffuse[..., 0] > 0
        diffuse = diffuse[..., [0, 0, 0]] #C,H,W,3
        # breakpoint()
        verts_clip_w = verts_clip[..., [3]]
        verts_clip_w[torch.logical_and(verts_clip_w >= 0.0, verts_clip_w < self.eps)] = self.eps
        verts_clip_w[torch.logical_and(verts_clip_w < 0.0, verts_clip_w > -self.eps)] = -self.eps

        pixel_normals_view = pixel_normals_view / torch.clamp(torch.norm(pixel_normals_view, p=2, dim=-1, keepdim=True), min=1e-5)
        pixel_normals_view[rast_out[..., -1] == 0] = -1
        pixel_normals_view = (pixel_normals_view + 1)/2

        verts_depth = (verts_clip[..., [2]] / verts_clip_w)     
        depth, _ = dr.interpolate(verts_depth, rast_out, faces) 

        unique_face_rast = torch.mul(rast_out[..., 3], front_mask.detach())
        unique_indices_visible = torch.unique(unique_face_rast.view(-1)).to(torch.int32) 
        unique_indices_visible = unique_indices_visible[unique_indices_visible > 0] - 1
        face_mask[unique_indices_visible] = True
        alpha = torch.clamp(rast_out[..., [-1]], max=1) #C,H,W,1   


        # depth = (-depth + 1.) * 0.5      # since depth in [-1, 1], normalize to [0, 1        
        # depth[rast_out[..., -1] == 0] = 1.0         # exclude background;
        # depth = 1 - depth                           # C,H,W,1
        # max_depth = depth.max()
        # min_depth = depth[depth > 0.0].min()  # exclude background;
        # depth = (depth - min_depth) / (max_depth - min_depth)

        # depth, _ = dr.interpolate(verts_depth, rast_out, faces) 
        depth = (depth + 1.) * 0.5      # since depth in [-1, 1], normalize to [0, 1]
        depth[rast_out[..., -1] == 0] = 1.0         # exclude background;
        depth = 1 - depth                           # C,H,W,1
        max_depth = depth.max()
        min_depth = depth[depth > 0.0].min()  # exclude background;

        # breakpoint()
        alpha = torch.mul(alpha, front_mask.unsqueeze(-1).detach())
        depth = torch.mul(depth, front_mask.unsqueeze(-1).detach())
        depth_info = {'raw': depth, 'masks' : face_mask}


        col = torch.concat((pixel_normals_view, diffuse, depth, alpha),dim=-1) #C,H,W,5
        col = dr.antialias(col, rast_out, verts_clip, faces) #C,H,W,5
        return col, depth_info,rast_out.clone()
    
    def forward_w_col(self,
                verts: torch.Tensor,
                verts_color : torch.Tensor,
                normals: torch.Tensor,
                faces: torch.Tensor):
        '''
        Single pass without transparency.
        '''
        V = verts.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((verts, torch.ones(V,1,device=verts.device)),axis=-1) #V,3 -> V,4
        verts_clip = vert_hom @ self._mvp.transpose(-2,-1) #C,V,4
        rast_out,_ = dr.rasterize(self._glctx, 
                                verts_clip, 
                                faces, 
                                resolution=self._image_size, 
                                grad_db=False) #C,H,W,4

        # view space normal;
        vert_normals_hom = torch.cat((normals, torch.zeros(V,1,device=verts.device)),axis=-1) #V,3 -> V,4
        vert_normals_view = vert_normals_hom @ self._mv.transpose(-2,-1) #C,V,4
        vert_normals_view = vert_normals_view[..., :3] #C,V,3
        
        # so the z coordinates should be negative;
        vert_normals_view[vert_normals_view[..., 2] > 0.] = \
            -vert_normals_view[vert_normals_view[..., 2] > 0.]
        vert_normals_view = vert_normals_view.contiguous()

        # view space lightdir;
        lightdir = torch.tensor(LIGHT_DIR, dtype=torch.float32, device=verts.device) #3
        lightdir = lightdir.view((1, 1, 1, 3)) #1,1,1,3

        # normal;
        pixel_normals_view, _ = dr.interpolate(vert_normals_view, rast_out, faces)  #C,H,W,3
        pixel_normals_view = pixel_normals_view / torch.clamp(torch.norm(pixel_normals_view, p=2, dim=-1, keepdim=True), min=1e-5)
        diffuse = torch.sum(lightdir * pixel_normals_view, -1, keepdim=True)           #C,H,W,1
        diffuse = torch.clamp(diffuse, min=0.0, max=1.0)
        diffuse = diffuse[..., [0, 0, 0]] #C,H,W,3
        verts_color = verts_color.to(torch.float32).contiguous()
        verts_col, _ = dr.interpolate(verts_color[None,...], rast_out, faces)
        
        # depth;
        verts_clip_w = verts_clip[..., [3]]
        verts_clip_w[torch.logical_and(verts_clip_w >= 0.0, verts_clip_w < self.eps)] = self.eps
        verts_clip_w[torch.logical_and(verts_clip_w < 0.0, verts_clip_w > -self.eps)] = -self.eps

        verts_depth = (verts_clip[..., [2]] / verts_clip_w)     #C,V,1
        depth, _ = dr.interpolate(verts_depth, rast_out, faces) #C,H,W,1    range: [-1, 1], -1 is near, 1 is far
        depth = (depth + 1.) * 0.5      

        depth[rast_out[..., -1] == 0] = 1.0         # exclude background;
        depth = 1 - depth                           # C,H,W,1
        max_depth = depth.max()
        min_depth = depth[depth > 0.0].min()  # exclude background;
        depth_info = {'raw': depth, 'max': max_depth, 'min': min_depth}

        # shillouette;
        alpha = torch.clamp(rast_out[..., [-1]], max=1) #C,H,W,1
        col = torch.concat((verts_col, diffuse, depth, alpha),dim=-1) #C,H,W,5
        col = dr.antialias(col, rast_out, verts_clip, faces) #C,H,W,5
        return col, depth_info
    
    
    def forward_with_texture(self,
                verts: torch.Tensor,
                normals: torch.Tensor,
                faces: torch.Tensor,
                textures: torch.Tensor,
                faces_texture: torch.Tensor,
                tex : torch.Tensor):
        '''
        Single pass without transparency.
        '''
        V = verts.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((verts, torch.ones(V,1,device=verts.device)),axis=-1) #V,3 -> V,4
        verts_clip = vert_hom @ self._mvp.transpose(-2,-1) #C,V,4
        rast_out,_ = dr.rasterize(self._glctx, 
                                verts_clip, 
                                faces, 
                                resolution=self._image_size, 
                                grad_db=False) #C,H,W,4
        tex = tex.contiguous()
        faces_texture = faces_texture.contiguous()
        texc, _ = dr.interpolate(textures[None, ...], rast_out, faces_texture)
        color = dr.texture(tex[None, ...], texc, filter_mode='linear')
        color = color * torch.clamp(rast_out[..., -1:], 0, 1)
        
        
        # view space normal;
        vert_normals_hom = torch.cat((normals, torch.zeros(V,1,device=verts.device)),axis=-1) #V,3 -> V,4
        vert_normals_view = vert_normals_hom @ self._mv.transpose(-2,-1) #C,V,4
        vert_normals_view = vert_normals_view[..., :3] #C,V,3
        # in the view space, normals should be oriented toward viewer, 
        # so the z coordinates should be negative;
        vert_normals_view[vert_normals_view[..., 2] > 0.] = \
            -vert_normals_view[vert_normals_view[..., 2] > 0.]
        vert_normals_view = vert_normals_view.contiguous()

        lightdir = torch.tensor(LIGHT_DIR, dtype=torch.float32, device=verts.device) #3
        lightdir = lightdir.view((1, 1, 1, 3)) #1,1,1,3

        # normal;
        pixel_normals_view, _ = dr.interpolate(vert_normals_view, rast_out, faces)  #C,H,W,3
        pixel_normals_view = pixel_normals_view / torch.clamp(torch.norm(pixel_normals_view, p=2, dim=-1, keepdim=True), min=1e-5)
        diffuse = torch.sum(lightdir * pixel_normals_view, -1, keepdim=True)           #C,H,W,1
        diffuse = torch.clamp(diffuse, min=0.0, max=1.0)
        diffuse = diffuse[..., [0, 0, 0]] #C,H,W,3

        # depth;
        verts_clip_w = verts_clip[..., [3]]
        verts_clip_w[torch.logical_and(verts_clip_w >= 0.0, verts_clip_w < self.eps)] = self.eps
        verts_clip_w[torch.logical_and(verts_clip_w < 0.0, verts_clip_w > -self.eps)] = -self.eps

        # verts_depth = (verts_clip[..., [2]] / verts_clip_w)     #C,V,1
        # depth, _ = dr.interpolate(verts_depth, rast_out, faces) #C,H,W,1    range: [-1, 1], -1 is near, 1 is far
        # depth = (depth + 1.) * 0.5      # since depth in [-1, 1], normalize to [0, 1]
        # depth[rast_out[..., -1] == 0] = 1.0         # exclude background;
        # depth = 1 - depth                           # C,H,W,1
        # max_depth = depth.max()
        # min_depth = depth[depth > 0.0].min()  # exclude background;


        depth, _ = dr.interpolate(verts_depth, rast_out, faces) 
        depth = (depth + 1.) * 0.5      # since depth in [-1, 1], normalize to [0, 1]
        depth[rast_out[..., -1] == 0] = 1.0         # exclude background;
        depth = 1 - depth                           # C,H,W,1
        max_depth = depth.max()
        min_depth = depth[depth > 0.0].min()  # exclude background;

        # depth_info = {'raw': depth, 'max': max_depth, 'min': min_depth}

        # shillouette;
        alpha = torch.clamp(rast_out[..., [-1]], max=1) #C,H,W,1
        
        col = torch.concat((color,diffuse, depth, alpha),dim=-1) #C,H,W,5
        col = dr.antialias(col, rast_out, verts_clip, faces) #C,H,W,5
        return col, None, vert_normals_view
    
class GTInitializer:

    def __init__(self, verts: torch.Tensor = None, faces: torch.Tensor = None, vert_texture : torch.Tensor = None, \
        face_texture : torch.Tensor = None, tex : torch.Tensor = None, with_texture: bool = False, \
            device: str = 'cuda'):
        
        # geometry info;
        self.gt_vertices = verts
        self.gt_faces = faces
        # self.gt_vertex_normals = calc_vertex_normals(verts, faces)
        self.gt_vertex_texture = vert_texture
        self.gt_face_texture = face_texture
        self.gt_texture = tex
        self.with_texture = with_texture
        self.gt_images = None
        self.gt_depth_info = None
    
    def update(self, modified_vertices, modified_faces):

        self.gt_vertices = modified_vertices
        self.gt_faces = modified_faces
        self.gt_vertex_normals = calc_vertex_normals(modified_vertices, modified_faces)

    def render(self, modified_verts, modified_faces, renderer: AlphaRenderer):
        
        mod_vertex_normals = calc_vertex_normals(modified_verts, modified_faces)
        target_images, target_depth_info, rast_out = \
            renderer.forward(modified_verts, mod_vertex_normals, modified_faces)

        self.gt_images = target_images
        self.gt_depth_info = target_depth_info
        # self.norm_images = target_normals

        return self.gt_images, self.gt_depth_info, rast_out

    def render_with_color(self, modified_verts, vert_color, modified_faces, renderer: AlphaRenderer):
        
        mod_vertex_normals = calc_vertex_normals(modified_verts, modified_faces)
        target_images, target_depth_info = \
            renderer.forward_w_col(modified_verts, vert_color, mod_vertex_normals, modified_faces)

        self.gt_images = target_images
        self.gt_depth_info = target_depth_info

        return self.gt_images
    
    def render_with_texture(self, modified_verts, modified_faces, renderer: AlphaRenderer):
        
        mod_vertex_normals = calc_vertex_normals(modified_verts, modified_faces)
        target_images, target_depth_info, target_normals = \
            renderer.forward_with_texture(modified_verts, mod_vertex_normals, \
                modified_faces ,self.gt_vertex_texture, self.gt_face_texture, self.gt_texture)

        self.gt_images = target_images
        self.norm_images = target_normals
        self.gt_depth_info = target_depth_info
        
        return self.gt_images

    @staticmethod
    def extract_PCA(features,num_dims = 16):
        
        features_mean = torch.mean(features, dim = 2, keepdim = True)
        features_centered = features - features_mean
        u,s,V = torch.svd(features_centered)
        pca_features = torch.matmul(features_centered, V[...,:num_dims])

        return pca_features
    def texture_images(self):
        if self.gt_images is None:
            raise ValueError("Ground truth image is None")
        return self.gt_images[...,:3]
    
    def diffuse_images(self):
        if self.gt_images is None:
            raise ValueError("Ground truth image is None")
        return self.gt_images[...,-5:-2]

    def depth_images(self):
        if self.gt_images is None:
            raise ValueError("Ground truth image is None")
        return self.gt_images[...,[-2,-2,-2]]

    def shillouette_images(self):
        if self.gt_images is None:
            raise ValueError("Ground truth image is None")
        return self.gt_images[...,[-1,-1,-1]]
    
    def normal_images(self):
        if self.gt_images is None:
            raise ValueError("Ground truth image is None")
        return self.gt_images[...,-8:-5]

    def dino_images(self):
        ## Check this 
        texture_images = self.gt_images[...,:3]
        images_batch = texture_images.permute(0,3,1,2)
        dino_images = extract_dino_features(images_batch)
        # print(dino_images.shape)
        # dino_images = feat.view(feat.shape[0]*feat.shape[1],feat.shape[2],feat.shape[3],feat.shape[4])
        dino_pca_images = self.extract_PCA(dino_images)
        return dino_pca_images