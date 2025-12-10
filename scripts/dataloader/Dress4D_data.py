"""
Monocular dataset for Dress4D dataset.
"""
import torch
from torch.utils.data import Dataset
from easydict import EasyDict
from scripts.dataloader.utils import (
    load_camera_and_smpl_dress4d, get_targets_diffuse, get_targets_npy, get_targets_normal, get_target_imgs
)

DEVICE = 'cuda'


def calculate_pca(pose_data: torch.Tensor, dim: int = 4) -> torch.Tensor:
    """Calculate PCA reduction of pose data.
    
    Args:
        pose_data: Pose data [N, D]
        dim: Target dimensionality
        
    Returns:
        Reduced pose data [N, dim]
    """
    if dim > pose_data.shape[1]:
        raise ValueError(f"Reduction dimension ({dim}) cannot be larger than input dimension ({pose_data.shape[1]})")
    
    mean_pose = torch.mean(pose_data, dim=0)
    centered_data = pose_data - mean_pose
    cov_matrix = torch.mm(centered_data.T, centered_data) / (pose_data.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    top_eigenvectors = eigenvectors[:, sorted_indices[:dim]]
    return torch.mm(centered_data, top_eigenvectors)


class MonocularDataset4DDress(Dataset):
    """Dataset for monocular reconstruction from Dress4D dataset."""
    
    def __init__(self, cfg: EasyDict):
        """Initialize dataset.
        
        Args:
            cfg: Configuration object with data paths and parameters
        """
        start_end = [cfg.start_frame, cfg.end_frame]
        self.mv, self.proj, self.pose, self.betas, self.translation = load_camera_and_smpl_dress4d(
            cfg, cfg.smpl_pkl, start_end
        )
        self.reduced_pose = calculate_pca(self.pose)
        self.reduced_pose_eight = calculate_pca(self.pose, dim=2)
        
        # Generate side views using circular shifts
        shift = 1
        self.mv_left = torch.cat([self.mv[shift:], self.mv[:shift]])
        self.mv_right = torch.cat([self.mv[-shift:], self.mv[:-shift]])
        shift = 10
        self.mv_back = torch.cat([self.mv[shift:], self.mv[:shift]])
        
        self.time_iterators = torch.linspace(0, 1, len(self.mv)).to(DEVICE)
        self.iterator_helper = torch.arange(start_end[0], start_end[1], cfg.skip_frames)
        
        # Load target images
        self.orig_image = get_targets_diffuse(cfg.target_images, cfg.image_size, start_end, cfg.skip_frames)
        self.target_diffuse = get_targets_diffuse(cfg.target_diffuse_maps, cfg.image_size, start_end, cfg.skip_frames)
        self.target_shil = get_targets_diffuse(cfg.target_shil_maps, cfg.image_size, start_end, cfg.skip_frames)
        self.target_complete_shil = get_targets_diffuse(
            cfg.target_complete_shil_maps, cfg.image_size, start_end, cfg.skip_frames
        )
        self.target_shil_seg = get_targets_diffuse(cfg.target_shil_seg_maps, cfg.image_size, start_end, cfg.skip_frames)
        self.target_hands_shil = get_targets_diffuse(cfg.target_hand_mask, cfg.image_size, start_end, cfg.skip_frames)
        self.target_depth = get_targets_npy(cfg.target_depth, cfg.image_size, start_end, cfg.skip_frames)
        self.target_norm_map = get_targets_normal(cfg.target_normal, cfg.image_size, start_end, cfg.skip_frames)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.target_diffuse)

    def __getitem__(self, idx: int) -> dict:
        """Get sample at index."""
        return {
            'mv': self.mv[idx],
            'mv_left': self.mv_left[idx],
            'mv_right': self.mv_right[idx],
            'mv_back': self.mv_back[idx],
            'proj': self.proj[idx],
            'time': self.time_iterators[idx],
            'target_diffuse': self.target_diffuse[idx],
            'target_shil': self.target_shil[idx],
            'target_shil_seg': self.target_shil_seg[idx],
            'target_complete_shil': self.target_complete_shil[idx],
            'hands_shil': self.target_hands_shil[idx],
            'target_depth': self.target_depth[idx],
            'target_norm_map': self.target_norm_map[idx],
            'tex_image': self.orig_image[idx],
            'reduced_pose': self.reduced_pose[idx],
            'reduced_pose_eight': self.reduced_pose_eight[idx],
            'pose': self.pose[idx],
            'betas': self.betas[idx],
            'translation': self.translation[idx],
            'idx': self.iterator_helper[idx]
        }


class MonocularTextureDataset4DDress(Dataset):
    """Dataset for texture training from Dress4D dataset."""
    
    def __init__(self, cfg: EasyDict):
        """Initialize texture dataset."""
        start_end = [cfg.start_frame, cfg.end_frame]
        self.mv, self.proj, self.pose, self.betas, self.translation = load_camera_and_smpl_dress4d(
            cfg, cfg.smpl_pkl, start_end
        )
        self.time_iterators = torch.linspace(0, 1, len(self.mv)).to(DEVICE)
        self.iterator_helper = torch.arange(start_end[0], start_end[1], cfg.skip_frames)
        
        # Load images
        self.target_image, self.background = get_target_imgs(
            cfg.target_images, cfg.image_size, start_end, cfg.skip_frames
        )
        self.target_diffuse = get_targets_diffuse(cfg.target_diffuse_maps, cfg.image_size, start_end, cfg.skip_frames)
        self.target_shil = get_targets_diffuse(cfg.target_shil_maps, cfg.image_size, start_end, cfg.skip_frames)
        self.target_complete_shil = get_targets_diffuse(
            cfg.target_complete_shil_maps, cfg.image_size, start_end, cfg.skip_frames
        )

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.target_diffuse)

    def __getitem__(self, idx: int) -> dict:
        """Get texture sample at index."""
        return {
            'mv': self.mv[idx],
            'proj': self.proj[idx],
            'time': self.time_iterators[idx],
            'target_diffuse': self.target_diffuse[idx],
            'target_shil': self.target_shil[idx],
            'target_complete_shil': self.target_complete_shil[idx],
            'tex_image': self.target_image[idx],
            'pose': self.pose[idx],
            'betas': self.betas[idx],
            'translation': self.translation[idx],
            'idx': self.iterator_helper[idx]
        }
