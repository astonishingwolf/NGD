"""
Monocular dataset for People Snapshot dataset.
"""
import math
import torch
from torch.utils.data import Dataset
from easydict import EasyDict
from scripts.dataloader.utils import (
    load_camera_and_smpl, get_targets_diffuse, get_targets_npy, get_targets_normal
)

DEVICE = 'cuda'


def y_rotation_matrix(degrees: float) -> torch.Tensor:
    """Create Y-axis rotation matrix.
    
    Args:
        degrees: Rotation angle in degrees
        
    Returns:
        4x4 rotation matrix
    """
    radians = math.radians(degrees)
    cos_r, sin_r = math.cos(radians), math.sin(radians)
    return torch.tensor([
        [cos_r, 0, sin_r, 0],
        [0, 1, 0, 0],
        [-sin_r, 0, cos_r, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)


# Pre-computed rotation matrices for side views
RY_LEFT = y_rotation_matrix(+18)   # Left rotation (CCW around Y)
RY_RIGHT = y_rotation_matrix(-18)  # Right rotation


def calculate_pca_people_snap(pose_data: torch.Tensor, dim: int = 4) -> torch.Tensor:
    """Calculate PCA reduction of pose data.
    
    Args:
        pose_data: Pose data [N, D]
        dim: Target dimensionality
        
    Returns:
        Reduced pose data [N, dim]
    """
    if dim > pose_data.shape[1]:
        raise ValueError(f"Reduction dimension ({dim}) cannot be larger than input dimension ({pose_data.shape[1]})")

    pose_data = pose_data.clone()
    pose_data[:, :3] = 0.0  # Zero out root rotation
    mean_pose = torch.mean(pose_data, dim=0)
    centered_data = pose_data - mean_pose
    cov_matrix = torch.mm(centered_data.T, centered_data) / (pose_data.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    top_eigenvectors = eigenvectors[:, sorted_indices[:dim]]
    return torch.mm(centered_data, top_eigenvectors)


class MonocularDataset(Dataset):
    """Dataset for monocular reconstruction from People Snapshot dataset."""
    
    def __init__(self, cfg: EasyDict):
        """Initialize dataset.
        
        Args:
            cfg: Configuration object with data paths and parameters
        """
        start_end = [cfg.start_frame, cfg.end_frame]
        self.mv, self.proj, self.pose, self.betas = load_camera_and_smpl(
            cfg, cfg.smpl_pkl, start_end
        )
        self.reduced_pose = calculate_pca_people_snap(self.pose.clone())
        self.reduced_pose_eight = calculate_pca_people_snap(self.pose.clone(), dim=2)
        
        # Generate side view matrices
        self.mv_left = torch.matmul(RY_LEFT.to(DEVICE), self.mv)
        self.mv_right = torch.matmul(RY_RIGHT.to(DEVICE), self.mv)
        
        self.time_iterators = torch.linspace(0, 1, len(self.mv)).to(DEVICE)
        self.iterator_helper = torch.arange(start_end[0], start_end[1], cfg.skip_frames)
        
        # Load target images
        self.orig_image = get_targets_diffuse(cfg.target_images, cfg.image_size, start_end, cfg.skip_frames)
        self.target_diffuse = get_targets_diffuse(cfg.target_diffuse_maps, cfg.image_size, start_end, cfg.skip_frames)
        self.target_shil = get_targets_diffuse(cfg.target_shil_maps, cfg.image_size, start_end, cfg.skip_frames)
        self.target_complete_shil = get_targets_diffuse(
            cfg.target_complete_shil_maps, cfg.image_size, start_end, cfg.skip_frames
        )
        self.target_hands_shil = get_targets_diffuse(cfg.target_hand_mask, cfg.image_size, start_end, cfg.skip_frames)
        self.target_depth = get_targets_npy(cfg.target_depth, cfg.image_size, start_end, cfg.skip_frames)
        self.target_norm_map = get_targets_normal(cfg.target_normal, cfg.image_size, start_end, cfg.skip_frames)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.target_diffuse)

    def __getitem__(self, idx: int) -> dict:
        """Get sample at index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing all sample data
        """
        mv_back = self.mv[idx].clone()
        mv_back[2, 2] = -1 * self.mv[idx][2, 2]  # Flip Z for back view
        
        return {
            'mv': self.mv[idx],
            'mv_back': mv_back,
            'mv_left': self.mv_left[idx],
            'mv_right': self.mv_right[idx],
            'proj': self.proj[idx],
            'time': self.time_iterators[idx],
            'target_diffuse': self.target_diffuse[idx],
            'target_shil': self.target_shil[idx],
            'target_complete_shil': self.target_complete_shil[idx],
            'hands_shil': self.target_hands_shil[idx],
            'target_depth': self.target_depth[idx],
            'target_norm_map': self.target_norm_map[idx],
            'tex_image': self.orig_image[idx],
            'reduced_pose': self.reduced_pose[idx],
            'reduced_pose_eight': self.reduced_pose_eight[idx],
            'pose': self.pose[idx],
            'betas': self.betas[idx],
            'idx': self.iterator_helper[idx]
        }


