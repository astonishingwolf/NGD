
import sys
import numpy as np
import torch
import torch.nn as nn
from math import exp
from torch.autograd import Variable
import torch.nn.functional as F

sys.path.append("../nvdiffmodeling")

import packages.nvdiffmodeling.src.renderutils as ru

device = 'cuda'

def get_vp_map(v_pos, mtx_in, resolution):
    device = v_pos.device
    with torch.no_grad():
        vp_mtx = torch.tensor([
            [resolution / 2, 0., 0., (resolution - 1) / 2],
            [0., resolution / 2, 0., (resolution - 1) / 2],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.,]
        ], device=device)

        v_pos_clip = ru.xfm_points(v_pos[None, ...], mtx_in)
        v_pos_div = v_pos_clip / v_pos_clip[..., -1:]

        v_vp = (vp_mtx @ v_pos_div.transpose(1, 2)).transpose(1, 2)[..., :-1]

        # don't need manual z-buffer here since we're using the rast map to do occlusion
        if False:
            v_pix = v_vp[..., :-1].int().cpu().numpy()
            v_depth = v_vp[..., -1].cpu().numpy()

            # pix_v_map = -torch.ones(len(v_pix), resolution, resolution, dtype=int)
            pix_v_map = -np.ones((len(v_pix), resolution, resolution), dtype=int)
            # v_pix_map = resolution * torch.ones(len(v_pix), len(v_pos), 2, dtype=int)
            v_pix_map = resolution * np.ones_like(v_pix, dtype=int)
            # buffer = torch.ones_like(pix_v_map) / 0
            buffer = -np.ones_like(pix_v_map) / 0
            for i, vs in enumerate(v_pix):
                for j, (y, x) in enumerate(vs):
                    if x < 0 or x > resolution - 1 or y < 0 or y > resolution - 1:
                        continue
                    else:
                        if v_depth[i, j] > buffer[i, x, y]:
                            buffer[i, x, y] = v_depth[i, j]
                            if pix_v_map[i, x, y] != -1:
                                v_pix_map[i, pix_v_map[i, x, y]] = np.array([resolution, resolution])
                            pix_v_map[i, x, y] = j
                            v_pix_map[i, j] = np.array([x, y])
            v_pix_map = torch.tensor(v_pix_map, device=device)
        v_pix_map = v_vp[..., :-1].int().flip([-1])
        v_pix_map [(v_pix_map > resolution - 1) | (v_pix_map < 0)] = resolution
    return v_pix_map.long()
 
def total_triangle_area(vertices):
    
    num_triangles = vertices.shape[0] // 3
    triangle_vertices = vertices[:num_triangles * 3].reshape(num_triangles, 3, 3)
    cross_products = torch.cross(triangle_vertices[:, 1] - triangle_vertices[:, 0],
                                 triangle_vertices[:, 2] - triangle_vertices[:, 0])
    areas = 0.5 * torch.norm(cross_products, dim=1)
    total_area = torch.sum(areas)
    
    return total_area

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


# def cal_face_connections(faces : torch.Tensor):

# ## Given a face and vertices and the values defined on the faces of the mesh what we want is the 
# ## Laplacian smoothing for the sam 

# def mesh_laplacian_smoothing_jacobians(predictions):

#     face_connectivity = cal_face_connections(predictions.faces)
#     num_edges = face_connectivity.shape[0]
#     vertices  = predictions.vertices

#     neighbour_smooth =  torch.zeros_like(vertices)
#     torch.scatter_mean(src = vertices, index = face_connectivity, dim = 1, out = neighbour_smooth)
#     neighbour_smooth = neighbour_smooth - vertices
    
#     # neighbour_smooth = neighbour_smooth / num_edges
#     # neighbour_smooth = neighbour_smooth.unsqueeze(1).repeat(1, 3, 1)

#     return torch.addcmul(predictions.iter_jacobians, neighbour_smooth, 1, value = 1)


class DepthDistillationLoss(nn.Module):
    def __init__(self,weight, template, margin=0.1, use_ndc=False, reduction='mean'):
        super().__init__()

        self.weight = weight
        self.template = template
        self.margin = margin
        self.use_ndc = use_ndc
        self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction=reduction)
    
    def sample_pairs(self, depth_map, num_boxes=50, box_size=64):
        """
        Vectorized sampling of pairs from small bounding boxes in the depth map
        Args:
            depth_map (torch.Tensor): Input depth map [B, 1, H, W]
            num_boxes (int): Number of boxes to sample per batch
            box_size (int): Size of each bounding box
        Returns:
            Tuple of tensors containing indices for pairs
        """
        B, _, H, W = depth_map.shape
        device = depth_map.device
        
        x_boxes = torch.randint(0, W - box_size, (B, num_boxes), device=device)
        y_boxes = torch.randint(0, H - box_size, (B, num_boxes), device=device)
        
        x1_offset = torch.randint(0, box_size, (B, num_boxes), device=device)
        y1_offset = torch.randint(0, box_size, (B, num_boxes), device=device)
        x2_offset = torch.randint(0, box_size, (B, num_boxes), device=device)
        y2_offset = torch.randint(0, box_size, (B, num_boxes), device=device)
        x1 = x_boxes + x1_offset
        y1 = y_boxes + y1_offset
        x2 = x_boxes + x2_offset
        y2 = y_boxes + y2_offset
        batch_indices = torch.arange(B, device=device).view(-1, 1).expand(-1, num_boxes)
        point1_indices = torch.stack([batch_indices, y1, x1], dim=-1).view(-1, 3)
        point2_indices = torch.stack([batch_indices, y2, x2], dim=-1).view(-1, 3)
        
        return point1_indices, point2_indices

    def forward(self, pred_depth, target_depth, gt_depth=None):
        """
        Compute depth distillation loss using vectorized operations
        Args:
            pred_depth (torch.Tensor): Predicted depth from DPT [B, 1, H, W]
            rendered_depth (torch.Tensor): Rendered depth map [B, 1, H, W]
            gt_depth (torch.Tensor, optional): Ground truth depth for additional supervision
        Returns:
            torch.Tensor: Computed loss
        """
        predicted_depth = pred_depth.train_depth.mean(dim=3, keepdim=True).permute(0,3,1,2)
        target_estimated_depth = target_depth.train_target_depth.unsqueeze(0)

        # breakpoint()
        # if self.use_ndc:
        #     pred_depth = 1.0 / (pred_depth + 1e-6)
        #     rendered_depth = 1.0 / (rendered_depth + 1e-6)

        point1_indices, point2_indices = self.sample_pairs(target_estimated_depth)
        
        depth1_pred = target_estimated_depth[point1_indices[:, 0], 0, point1_indices[:, 1], point1_indices[:, 2]]
        depth2_pred = target_estimated_depth[point2_indices[:, 0], 0, point2_indices[:, 1], point2_indices[:, 2]]
        depth1_rendered = predicted_depth[point1_indices[:, 0], 0, point1_indices[:, 1], point1_indices[:, 2]]
        depth2_rendered = predicted_depth[point2_indices[:, 0], 0, point2_indices[:, 1], point2_indices[:, 2]]
        
        target = torch.sign(depth2_pred - depth1_pred)
        
        loss = self.weight * self.ranking_loss(depth1_rendered, depth2_rendered, target)

        loss_dict = {
            'loss_depth': loss
        }
        # breakpoint()
        return loss, loss_dict


class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, pred, target):
        error = pred - target
        abs_error = torch.abs(error)
        
        quadratic = 0.5 * error ** 2
        linear = self.delta * (abs_error - 0.5 * self.delta)

        loss = torch.where(abs_error < self.delta, quadratic, linear)
        return loss.mean()
