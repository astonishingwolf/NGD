"""
Rendering loss functions for image and silhouette matching.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.losses.utils import HuberLoss, create_window


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    """Compute Structural Similarity Index (SSIM).
    
    Args:
        img1: First image tensor [B, C, H, W]
        img2: Second image tensor [B, C, H, W]
        window_size: Size of sliding window
        size_average: Whether to average over spatial dimensions
        
    Returns:
        SSIM value(s)
    """
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor, window_size: int,
         channel: int, size_average: bool = True) -> torch.Tensor:
    """Internal SSIM computation."""
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class RenderingLoss(nn.Module):
    """Rendering loss combining L1, SSIM, and silhouette losses."""
    
    def __init__(self, weight: float, template, device: str = 'cuda'):
        """Initialize rendering loss.
        
        Args:
            weight: Loss weight
            template: Template/cloth object
            device: Device to run on
        """
        super(RenderingLoss, self).__init__()
        self.weight = weight
        self.template = template
        self.device = device
        self.lambda_dssim = 0.35
        self.shil_loss_weight = 0.4
        self.rendering_loss = HuberLoss(0.005)
    
    def forward(self, predictions, targets):
        """Compute rendering loss.
        
        Args:
            predictions: Predictions namespace
            targets: Targets namespace
            
        Returns:
            Tuple of (loss, loss_dict)
        """
        train_render = predictions.train_render.permute(0, 3, 1, 2)
        train_shil = predictions.train_shil.permute(0, 3, 1, 2)
        
        # Mask out hands from silhouette
        train_shil_withouthand = train_shil.to(torch.float32) * (1 - targets.hands_shil.to(torch.float32))
        
        # Mask render with silhouette
        train_render = torch.mul(train_render, targets.train_target_render_shil)
        
        # Compute losses
        shil_loss = F.mse_loss(train_shil_withouthand, targets.train_target_render_shil)
        l1_rendering_loss = self.rendering_loss(train_render, targets.train_target_render)
        
        # Combine losses
        rendering_loss = (
            (1.0 - self.lambda_dssim) * l1_rendering_loss +
            self.lambda_dssim * (1.0 - ssim(train_render, targets.train_target_render)) +
            shil_loss * self.shil_loss_weight
        )
        
        loss_shad = rendering_loss * self.weight
        
        loss_dict = {'loss_render': loss_shad}
        return loss_shad, loss_dict
