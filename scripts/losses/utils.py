
import sys
import numpy as np
import torch
import torch.nn as nn
from math import ceil
from math import exp
from torch.autograd import Variable
import torch.nn.functional as F

sys.path.append("../nvdiffmodeling")

import packages.nvdiffmodeling.src.mesh as mesh
import packages.nvdiffmodeling.src.texture as texture
import packages.nvdiffmodeling.src.renderutils as ru
from torchvision.models import vgg19_bn
from torchvision.models import VGG19_BN_Weights

device = 'cuda'
class Vgg19(torch.nn.Module):

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
    
        vgg_pretrained_features = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
    
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

cosine_sim = torch.nn.CosineSimilarity()
render_loss = torch.nn.L1Loss()
mean = torch.tensor([0.485, 0.456, 0.406], device='cuda')
std = torch.tensor([0.229, 0.224, 0.225], device='cuda')

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
            # cos_loss += self.weights[i] * cosine_sim(x_vgg[i], y_vgg[i].detach()).mean()
        return loss

vgg_loss = VGGLoss()
def get_vgg_loss(base_image, target_image):

    base_image = (torch.clamp(base_image, 0, 1) - mean[None, :, None, None]) / std[None, :, None, None]
    target_image = (torch.clamp(target_image, 0, 1) - mean[None, :, None, None]) / std[None, :, None, None]
    
    return vgg_loss(base_image, target_image)

def cosine_sum(features, targets):

    return -cosine_sim(features, targets).sum()

def cosine_avg(features, targets):

    return -cosine_sim(features, targets).mean()

def l1_avg(source, targets):

    return render_loss(source, targets).mean()
    
def _merge_attr_idx(a, b, a_idx, b_idx, scale_a=1.0, scale_b=1.0, add_a=0.0, add_b=0.0):
    
    if a is None and b is None:
        return None, None
    elif a is not None and b is None:
        return (a*scale_a)+add_a, a_idx
    elif a is None and b is not None:
        return (b*scale_b)+add_b, b_idx
    else:
        return torch.cat(((a*scale_a)+add_a, (b*scale_b)+add_b), dim=0), torch.cat((a_idx, b_idx + a.shape[0]), dim=0)

def create_scene(meshes, sz=1024):
    
    # Need to comment and fix code
    
    scene = mesh.Mesh()

    tot = len(meshes) if len(meshes) % 2 == 0 else len(meshes)+1

    nx = 2
    ny = ceil(tot / 2) if ceil(tot / 2) % 2 == 0 else ceil(tot / 2) + 1

    w = int(sz*ny)
    h = int(sz*nx)

    dev = meshes[0].v_pos.device

    kd_atlas = torch.ones ( (1, w, h, 4) ).to(dev)
    ks_atlas = torch.zeros( (1, w, h, 3) ).to(dev)
    kn_atlas = torch.ones ( (1, w, h, 3) ).to(dev)

    for i, m in enumerate(meshes):
        v_pos, t_pos_idx = _merge_attr_idx(scene.v_pos, m.v_pos, scene.t_pos_idx, m.t_pos_idx)
        v_nrm, t_nrm_idx = _merge_attr_idx(scene.v_nrm, m.v_nrm, scene.t_nrm_idx, m.t_nrm_idx)
        v_tng, t_tng_idx = _merge_attr_idx(scene.v_tng, m.v_tng, scene.t_tng_idx, m.t_tng_idx)

        pos_x = i % nx
        pos_y = int(i / ny)

        sc_x = 1./nx
        sc_y = 1./ny

        v_tex, t_tex_idx = _merge_attr_idx(
            scene.v_tex,
            m.v_tex,
            scene.t_tex_idx,
            m.t_tex_idx,
            scale_a=1.,
            scale_b=torch.tensor([sc_x, sc_y]).to(dev),
            add_a=0.,
            add_b=torch.tensor([sc_x*pos_x, sc_y*pos_y]).to(dev)
        )

        kd_atlas[:, pos_y*sz:(pos_y*sz)+sz, pos_x*sz:(pos_x*sz)+sz, :m.material['kd'].data.shape[-1]] = m.material['kd'].data
        ks_atlas[:, pos_y*sz:(pos_y*sz)+sz, pos_x*sz:(pos_x*sz)+sz, :m.material['ks'].data.shape[-1]] = m.material['ks'].data
        kn_atlas[:, pos_y*sz:(pos_y*sz)+sz, pos_x*sz:(pos_x*sz)+sz, :m.material['normal'].data.shape[-1]] = m.material['normal'].data

        scene = mesh.Mesh(
            v_pos=v_pos,
            t_pos_idx=t_pos_idx,
            v_nrm=v_nrm,
            t_nrm_idx=t_nrm_idx,
            v_tng=v_tng,
            t_tng_idx=t_tng_idx,
            v_tex=v_tex,
            t_tex_idx=t_tex_idx,
            base=scene 
        )

    scene = mesh.Mesh(
        material={
            'bsdf': 'diffuse',
            'kd': texture.Texture2D(
                kd_atlas
            ),
            'ks': texture.Texture2D(
                ks_atlas
            ),
            'normal': texture.Texture2D(
                kn_atlas
            ),
        },
        base=scene # gets uvs etc from here
    )

    return scene

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

def triangle_size_regularization(vertices_cannonical,vertices_deformed):
    return (total_triangle_area(vertices_cannonical) - total_triangle_area(vertices_deformed))**2

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

grid = None

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6, mode=None):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.mode = mode

    def forward(self, x, y, mask=None):
        N = x.size(1)
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps)
        if mask is not None:
            loss = loss * mask
        if self.mode == 'sum':
            loss = torch.sum(loss) / N
        else:
            loss = loss.mean()
        return loss


def gauss_kernel(size=5, channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def upsample(x):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).to(device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
    cc = cc.permute(0,1,3,2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2).to(device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
    x_up = cc.permute(0,1,3,2)
    return conv_gauss(x_up, 4*gauss_kernel(channels=x.shape[1]))

def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out

def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current-up
        pyr.append(diff)
        current = down
    return pyr

class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=3, channels=3):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels)
        
    def forward(self, input, target):
        pyr_input  = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))

def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy            

def grad_loss(img):
    img_dx, img_dy = gradient(img)
    dx, dy = gradient(img)
    loss_x = dx.abs()
    loss_y = dy.abs()
    
    return loss_x.mean() + loss_y.mean()

def grad_loss_2(img):
    img_dx, img_dy = gradient(img)
    dx, dy = gradient(img)
    loss_x = dx ** 2
    loss_y = dy ** 2
    
    return loss_x.mean() + loss_y.mean()

def smooth_grad_2nd(flo, image, alpha=10):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)
    
    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)
    
    loss_x = weights_x[:, :, :, 1:] * dx2.abs()
    loss_y = weights_y[:, :, 1:, :] * dy2.abs()
    
    return loss_x.mean() / 2. + loss_y.mean() / 2.

Grid = {}
class EPE(nn.Module):
    def __init__(self):
        super(EPE, self).__init__()

    def forward(self, flow, gt, loss_mask):
        loss_map = (flow - gt.detach()) ** 2
        loss_map = (loss_map.sum(1, True) + 1e-6) ** 0.5
        return (loss_map * loss_mask)

class Ternary(nn.Module):
    def __init__(self):
        super(Ternary, self).__init__()
        patch_size = 7
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float().to(device)

    def transform(self, img):
        patches = F.conv2d(img, self.w, padding=3, bias=None)
        transf = patches - img
        transf_norm = transf / torch.sqrt(0.81 + transf**2)
        return transf_norm

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, 0:1,:,:], rgb[:, 1:2,:,:], rgb[:, 2:3,:,:]        
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    
    def hamming(self, t1, t2):
        dist = (t1 - t2) ** 2
        dist_norm = torch.mean(dist / (0.1 + dist), 1, True)
        return dist_norm

    def valid_mask(self, t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask
    
    def forward(self, img0, img1):
        img0 = self.transform(self.rgb2gray(img0))
        img1 = self.transform(self.rgb2gray(img1))
        return self.hamming(img0, img1) * self.valid_mask(img0, 1)
    
class SOBEL(nn.Module):
    def __init__(self):
        super(SOBEL, self).__init__()
        self.kernelX = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1],
        ]).float()
        self.kernelY = self.kernelX.clone().T
        self.kernelX = self.kernelX.unsqueeze(0).unsqueeze(0).to(device)
        self.kernelY = self.kernelY.unsqueeze(0).unsqueeze(0).to(device)

    def forward(self, pred, gt):
        N, C, H, W = pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3]
        img_stack = torch.cat([pred.reshape(N*C, 1, H, W), gt.reshape(N*C, 1, H, W)], 0)
        sobel_stack_x = F.conv2d(img_stack, self.kernelX, padding=1)
        sobel_stack_y = F.conv2d(img_stack, self.kernelY, padding=1)
        pred_X, gt_X = sobel_stack_x[:N*C], sobel_stack_x[N*C:]
        pred_Y, gt_Y = sobel_stack_y[:N*C], sobel_stack_y[N*C:]
    
        L1X, L1Y = torch.abs(pred_X-gt_X), torch.abs(pred_Y-gt_Y)
        # breakpoint()
        loss = (L1X+L1Y)
        return loss

class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, rank=0):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        pretrained = True
        self.vgg_pretrained_features = models.vgg19(pretrained=pretrained).features
        self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X, Y, indices=None):
        X = self.normalize(X)
        Y = self.normalize(Y)
        indices = [2, 7, 12, 21, 30]
        weights = [1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 10/1.5]
        k = 0
        loss = 0
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            Y = self.vgg_pretrained_features[i](Y)
            if (i+1) in indices:
                loss += weights[k] * (X - Y.detach()).abs().mean() * 0.1
                k += 1
        return loss


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