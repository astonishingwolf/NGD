import torch
import torch.nn as nn
import torch.nn.functional as F 

from scripts.renderer.renderer import *
from scripts.losses.utils import *

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)   
        
class RenderingLoss(nn.Module):
    def __init__(self, weight, template, device = 'cuda'):
        super(RenderingLoss, self).__init__()

        self.weight = weight
        self.template = template
        self.device = device
        self.lambda_dssim = 0.35
        self.shil_loss = 0.4
        self.soft_shil =  False
        
    def forward(self, predictions, targets):
        
        train_render = predictions.train_render.permute(0,3,1,2)
        train_shil = predictions.train_shil.permute(0,3,1,2)
        # train_norm = predictions.train_norm

        without_target_cloth = torch.logical_and(torch.logical_not(targets.train_target_render_shil),targets.train_target_complete)
        with_cloth = without_target_cloth.to(torch.float32) + train_shil
        with_cloth = with_cloth.clamp(max = 1.0)
        train_shil_withouthand = train_shil.to(torch.float32)* (1 - targets.hands_shil.to(torch.float32))
        train_shil_withouthand = train_shil_withouthand.to(torch.float32)

        # shil_intersection = torch.mul(train_shil,targets.train_target_render_shil)
        # shil_loss = F.mse_loss(with_cloth, targets.train_target_complete)
        # shil_loss = F.mse_loss(train_shil, targets.train_target_render_shil)
        # l1_rendering_loss = l1_avg(train_norm, targets.target_norm_map)

        # Default
        train_render = torch.mul(train_render,targets.train_target_render_shil)
        shil_loss = F.mse_loss(train_shil_withouthand, targets.train_target_render_shil)
        # shil_loss = F.mse_loss(with_cloth, targets.train_target_complete)
        l1_rendering_loss = l1_avg(train_render, targets.train_target_render)
        rendering_loss =  (1.0 - self.lambda_dssim) * l1_rendering_loss + self.lambda_dssim * (1.0 - ssim(train_render, targets.train_target_render)) + shil_loss * self.shil_loss
        loss_shad = rendering_loss* self.weight
        
        # Normal_Supervison
        # train_render = predictions.train_norm.permute(0,3,1,2)
        # train_render = torch.mul(train_render,targets.train_target_render_shil)
        # target_render = targets.train_target_normal.permute(0,3,1,2)
        # target_render = torch.mul(target_render,targets.train_target_render_shil)
        # shil_loss = F.mse_loss(train_shil_withouthand, targets.train_target_render_shil)
        # l1_rendering_loss = l1_avg(train_render, target_render)
        # # rendering_loss =  (1.0 - self.lambda_dssim) * l1_rendering_loss + self.lambda_dssim * (1.0 - ssim(train_render, targets.train_target_render)) + shil_loss * self.shil_loss
        # rendering_loss = l1_rendering_loss + shil_loss * self.shil_loss
        # loss_shad = rendering_loss* self.weight

        # loss_depth = get_depth_loss(predictions.train_depth, targets.train_target_depth)
        # Hash Grid Specific
        # loss = (train_render - targets.train_target_render) ** 2 / (train_render.detach() ** 2 + 0.01) + \
        #        (train_shil_withouthand - targets.train_target_render_shil) ** 2 / (train_shil_withouthand.detach() ** 2 + 0.01) * self.shil_loss

        # loss_shad = loss.mean()
        # loss_shad = loss_shad * self.weight

        
        # rendering_loss =  l1_avg(train_render, targets.train_
        # target_render) + \
        #                     self.lamda_laplacian * (self.lap_loss(train_render, targets.train_target_render)) + shil_loss * self.shil_loss
        
        # rendering_loss =  l1_avg(train_render, targets.train_target_render) + shil_loss * self.shil_loss
        # rendering_loss =  l1_rendering_loss + shil_loss * self.shil_loss
        # loss_mass = F.mse_loss(shil_intersection, targets.train_target_render_shil)
        loss = loss_shad
        # breakpoint()
        
        loss_dict = {
            'loss_render': loss_shad
        }

        return loss, loss_dict