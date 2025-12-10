"""
Provides AlphaRenderer and GTInitializer classes for mesh rendering.
"""
import torch
import nvdiffrast.torch as dr
from scripts.renderer.utils import NormalsRenderer, calc_vertex_normals
from scripts.utils.general_utils import extract_dino_features

LIGHT_DIR = torch.tensor([0., 0., 1.0])
EPS = 1e-4


class AlphaRenderer(NormalsRenderer):
    """Renderer for normal maps, depth maps, and silhouettes."""
    
    def __init__(self, mv: torch.Tensor, proj: torch.Tensor, image_size: tuple[int, int]):
        """Initialize renderer with camera matrices.
        
        Args:
            mv: Model-view matrix [C, 4, 4]
            proj: Projection matrix [C, 4, 4]
            image_size: Output image dimensions (height, width)
        """
        super().__init__(mv, proj, image_size)
        self._mv = mv
        self._proj = proj
        self._eps = EPS
    
    def _rasterize_vertices(self, verts: torch.Tensor, faces: torch.Tensor):
        """Rasterize vertices to screen space."""
        V = verts.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat([verts, torch.ones(V, 1, device=verts.device)], dim=-1)
        verts_clip = vert_hom @ self._mvp.transpose(-2, -1)
        rast_out, _ = dr.rasterize(
            self._glctx, verts_clip, faces, resolution=self._image_size, grad_db=False
        )
        return verts_clip, rast_out
    
    def _compute_view_normals(self, normals: torch.Tensor, verts: torch.Tensor):
        """Transform normals to view space."""
        V = normals.shape[0]
        vert_normals_hom = torch.cat([normals, torch.zeros(V, 1, device=normals.device)], dim=-1)
        vert_normals_view = (vert_normals_hom @ self._mv.transpose(-2, -1))[..., :3]
        return vert_normals_view.contiguous()
    
    def _compute_diffuse(self, pixel_normals: torch.Tensor, rast_out: torch.Tensor, faces: torch.Tensor):
        """Compute diffuse shading."""
        lightdir = LIGHT_DIR.to(pixel_normals.device).view(1, 1, 1, 3)
        diffuse = torch.sum(lightdir * pixel_normals, dim=-1, keepdim=True)
        diffuse = torch.clamp(diffuse, min=0.0, max=1.0)
        front_mask = diffuse[..., 0] > 0
        return diffuse[..., [0, 0, 0]], front_mask
    
    def _compute_depth(self, verts_clip: torch.Tensor, rast_out: torch.Tensor, faces: torch.Tensor):
        """Compute depth map from clip space vertices."""
        verts_clip_w = verts_clip[..., [3]]
        verts_clip_w[torch.logical_and(verts_clip_w >= 0.0, verts_clip_w < self._eps)] = self._eps
        verts_clip_w[torch.logical_and(verts_clip_w < 0.0, verts_clip_w > -self._eps)] = -self._eps
        
        verts_depth = verts_clip[..., [2]] / verts_clip_w
        depth, _ = dr.interpolate(verts_depth, rast_out, faces)
        depth = (depth + 1.0) * 0.5  # Normalize from [-1, 1] to [0, 1]
        depth[rast_out[..., -1] == 0] = 1.0
        depth = 1 - depth
        return depth
    
    def _get_visible_faces(self, rast_out: torch.Tensor, faces: torch.Tensor, front_mask: torch.Tensor = None):
        """Extract visible face indices from rasterization output."""
        face_mask = torch.zeros_like(faces, dtype=torch.bool)
        unique_face_rast = rast_out[..., 3] if front_mask is None else rast_out[..., 3] * front_mask.detach()
        unique_indices_visible = torch.unique(unique_face_rast.view(-1)).to(torch.int32)
        unique_indices_visible = unique_indices_visible[unique_indices_visible > 0] - 1
        face_mask[unique_indices_visible] = True
        return face_mask
    
    def forward(self, verts: torch.Tensor, normals: torch.Tensor, faces: torch.Tensor):
        """Forward pass: render normal maps, depth, and silhouette.
        
        Returns:
            Tuple of (output_image, depth_info, rasterization_output)
        """
        verts_clip, rast_out = self._rasterize_vertices(verts, faces)
        vert_normals_view = self._compute_view_normals(normals, verts)
        
        # Interpolate normals and compute diffuse
        pixel_normals_view, _ = dr.interpolate(vert_normals_view, rast_out, faces)
        pixel_normals_view = pixel_normals_view / torch.clamp(
            torch.norm(pixel_normals_view, p=2, dim=-1, keepdim=True), min=1e-5
        )
        pixel_normals_view[rast_out[..., -1] == 0] = -1
        pixel_normals_view = (pixel_normals_view + 1) / 2
        
        diffuse, front_mask = self._compute_diffuse(pixel_normals_view, rast_out, faces)
        depth = self._compute_depth(verts_clip, rast_out, faces)
        depth = depth * front_mask.unsqueeze(-1).detach()
        
        face_mask = self._get_visible_faces(rast_out, faces, front_mask)
        alpha = torch.clamp(rast_out[..., [-1]], max=1) * front_mask.unsqueeze(-1).detach()
        
        depth_info = {'raw': depth, 'masks': face_mask}
        col = torch.concat([pixel_normals_view, diffuse, depth, alpha], dim=-1)
        col = dr.antialias(col, rast_out, verts_clip, faces)
        
        return col, depth_info, rast_out.clone()
    
    def forward_w_col(self, verts: torch.Tensor, verts_color: torch.Tensor, 
                     normals: torch.Tensor, faces: torch.Tensor):
        """Forward pass with vertex colors."""
        verts_clip, rast_out = self._rasterize_vertices(verts, faces)
        vert_normals_view = self._compute_view_normals(normals, verts)
        vert_normals_view[vert_normals_view[..., 2] > 0.] *= -1
        vert_normals_view = vert_normals_view.contiguous()
        
        pixel_normals_view, _ = dr.interpolate(vert_normals_view, rast_out, faces)
        pixel_normals_view = pixel_normals_view / torch.clamp(
            torch.norm(pixel_normals_view, p=2, dim=-1, keepdim=True), min=1e-5
        )
        diffuse, _ = self._compute_diffuse(pixel_normals_view, rast_out, faces)
        
        verts_col, _ = dr.interpolate(verts_color.to(torch.float32).contiguous()[None, ...], rast_out, faces)
        depth = self._compute_depth(verts_clip, rast_out, faces)
        depth_info = {'raw': depth, 'max': depth.max(), 'min': depth[depth > 0.0].min()}
        
        alpha = torch.clamp(rast_out[..., [-1]], max=1)
        col = torch.concat([verts_col, diffuse, depth, alpha], dim=-1)
        col = dr.antialias(col, rast_out, verts_clip, faces)
        
        return col, depth_info
    
    def forward_with_texture(self, verts: torch.Tensor, normals: torch.Tensor, faces: torch.Tensor,
                            textures: torch.Tensor, faces_texture: torch.Tensor, tex: torch.Tensor):
        """Forward pass with texture mapping."""
        verts_clip, rast_out = self._rasterize_vertices(verts, faces)
        
        # Sample texture
        texc, _ = dr.interpolate(textures[None, ...], rast_out, faces_texture.contiguous())
        color = dr.texture(tex[None, ...].contiguous(), texc, filter_mode='linear')
        color = color * torch.clamp(rast_out[..., -1:], 0, 1)
        
        # Compute normals and diffuse
        vert_normals_view = self._compute_view_normals(normals, verts)
        vert_normals_view[vert_normals_view[..., 2] > 0.] *= -1
        vert_normals_view = vert_normals_view.contiguous()
        
        pixel_normals_view, _ = dr.interpolate(vert_normals_view, rast_out, faces)
        pixel_normals_view = pixel_normals_view / torch.clamp(
            torch.norm(pixel_normals_view, p=2, dim=-1, keepdim=True), min=1e-5
        )
        diffuse, _ = self._compute_diffuse(pixel_normals_view, rast_out, faces)
        
        depth = self._compute_depth(verts_clip, rast_out, faces)
        alpha = torch.clamp(rast_out[..., [-1]], max=1)
        
        col = torch.concat([color, diffuse, depth, alpha], dim=-1)
        col = dr.antialias(col, rast_out, verts_clip, faces)
        
        return col, None, vert_normals_view


class GTInitializer:
    """Ground truth image initializer and renderer wrapper."""
    
    def __init__(self, verts: torch.Tensor = None, faces: torch.Tensor = None,
                 vert_texture: torch.Tensor = None, face_texture: torch.Tensor = None,
                 tex: torch.Tensor = None, with_texture: bool = False, device: str = 'cuda'):
        """Initialize with geometry and optional texture data."""
        self.gt_vertices = verts
        self.gt_faces = faces
        self.gt_vertex_texture = vert_texture
        self.gt_face_texture = face_texture
        self.gt_texture = tex
        self.with_texture = with_texture
        self.gt_images = None
        self.gt_depth_info = None
        self.norm_images = None
    
    def update(self, modified_vertices: torch.Tensor, modified_faces: torch.Tensor):
        """Update geometry."""
        self.gt_vertices = modified_vertices
        self.gt_faces = modified_faces
    
    def render(self, modified_verts: torch.Tensor, modified_faces: torch.Tensor, 
               renderer: AlphaRenderer):
        """Render mesh and store results."""
        mod_vertex_normals = calc_vertex_normals(modified_verts, modified_faces)
        target_images, target_depth_info, rast_out = renderer.forward(
            modified_verts, mod_vertex_normals, modified_faces
        )
        self.gt_images = target_images
        self.gt_depth_info = target_depth_info
        return self.gt_images, self.gt_depth_info, rast_out
    
    def render_with_color(self, modified_verts: torch.Tensor, vert_color: torch.Tensor,
                         modified_faces: torch.Tensor, renderer: AlphaRenderer):
        """Render mesh with vertex colors."""
        mod_vertex_normals = calc_vertex_normals(modified_verts, modified_faces)
        target_images, target_depth_info = renderer.forward_w_col(
            modified_verts, vert_color, mod_vertex_normals, modified_faces
        )
        self.gt_images = target_images
        self.gt_depth_info = target_depth_info
        return self.gt_images
    
    def render_with_texture(self, modified_verts: torch.Tensor, modified_faces: torch.Tensor,
                           renderer: AlphaRenderer):
        """Render mesh with texture mapping."""
        mod_vertex_normals = calc_vertex_normals(modified_verts, modified_faces)
        target_images, target_depth_info, target_normals = renderer.forward_with_texture(
            modified_verts, mod_vertex_normals, modified_faces,
            self.gt_vertex_texture, self.gt_face_texture, self.gt_texture
        )
        self.gt_images = target_images
        self.norm_images = target_normals
        self.gt_depth_info = target_depth_info
        return self.gt_images
    
    @staticmethod
    def extract_PCA(features: torch.Tensor, num_dims: int = 16):
        """Extract PCA features."""
        features_mean = torch.mean(features, dim=2, keepdim=True)
        features_centered = features - features_mean
        _, _, V = torch.svd(features_centered)
        return torch.matmul(features_centered, V[..., :num_dims])
    
    def _check_images(self):
        """Check if images are available."""
        if self.gt_images is None:
            raise ValueError("Ground truth image is None")
    
    def texture_images(self):
        """Extract texture images."""
        self._check_images()
        return self.gt_images[..., :3]
    
    def diffuse_images(self):
        """Extract diffuse images."""
        self._check_images()
        return self.gt_images[..., -5:-2]
    
    def depth_images(self):
        """Extract depth images."""
        self._check_images()
        return self.gt_images[..., [-2, -2, -2]]
    
    def shillouette_images(self):
        """Extract silhouette images."""
        self._check_images()
        return self.gt_images[..., [-1, -1, -1]]
    
    def normal_images(self):
        """Extract normal images."""
        if self.norm_images is not None:
            return self.norm_images
        self._check_images()
        return self.gt_images[..., -8:-5]
    
    def dino_images(self):
        """Extract DINO features and apply PCA."""
        texture_images = self.texture_images()
        images_batch = texture_images.permute(0, 3, 1, 2)
        dino_images = extract_dino_features(images_batch)
        return self.extract_PCA(dino_images)
