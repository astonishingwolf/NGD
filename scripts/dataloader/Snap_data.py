import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from easydict import EasyDict
import pyrender
import pickle
import glob


from scripts.dataloader.utils import *
DEVICE = 'cuda'

def calculate_pca_people_snap(pose_data, dim=4):

    if dim > pose_data.shape[1]:
        raise ValueError(f"Reduction dimension ({dim}) cannot be larger than input dimension ({pose_data.shape[1]})")

    pose_data[:,:3] = pose_data[:,:3] * 0.0
    mean_pose = torch.mean(pose_data, dim=0)
    centered_data = pose_data - mean_pose
    cov_matrix = torch.mm(centered_data.T, centered_data) / (pose_data.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    top_eigenvectors = eigenvectors[:, :dim]
    reduced_pose = torch.mm(centered_data, top_eigenvectors)
    
    return reduced_pose

class MonocularDataset(Dataset):
    def __init__(self, cfg):

        """
        Parameters:
        ----------
        cfg : 
            Configuration object.
        """
        start_end = [cfg.start_frame, cfg.end_frame]
        self.mv, self.proj, self.pose, self.betas = load_camera_and_smpl(cfg,cfg.smpl_pkl,start_end)
        self.reduced_pose = calculate_pca_people_snap(self.pose.clone())
        self.reduced_pose_eight = calculate_pca_people_snap(self.pose.clone(), dim=2)
        shift = 1
        self.mv_left = torch.cat([self.mv[shift:], self.mv[:shift]])  
        self.mv_right = torch.cat([self.mv[-shift:], self.mv[:-shift]]) 
        # back_mv[:,2,2] = -1 * back_mv[:,2,2]

        self.time_iterators = torch.linspace(0, 1, len(self.mv)).to(DEVICE)
        self.orig_image = get_targets_diffuse(cfg.target_images, cfg.image_size, start_end, cfg.skip_frames)
        self.target_diffuse = get_targets_diffuse(cfg.target_diffuse_maps, cfg.image_size, start_end,cfg.skip_frames)
        self.target_shil = get_targets_diffuse(cfg.target_shil_maps, cfg.image_size,start_end,cfg.skip_frames)
        self.target_complete_shil = get_targets_diffuse(cfg.target_complete_shil_maps, cfg.image_size,start_end,cfg.skip_frames)
        self.target_hands_shil = get_targets_diffuse(cfg.target_hand_mask, cfg.image_size,start_end,cfg.skip_frames)
        # self.target_norm_map = get_targets_npy(cfg.target_normal, cfg.image_size,start_end,cfg.skip_frames)
        self.target_depth = get_targets_npy(cfg.target_depth, cfg.image_size,start_end,cfg.skip_frames)
        self.target_norm_map = get_targets_normal(cfg.target_normal, cfg.image_size,start_end,cfg.skip_frames)
        self.iterator_helper = torch.arange(start_end[0],start_end[1],cfg.skip_frames)

    def __len__(self) -> int:
        return len(self.target_diffuse)

    def __getitem__(self, idx: int):
        
        sample = {}
        sample['mv'] = self.mv[idx]
        sample['mv_back'] = self.mv[idx].clone()
        sample['mv_back'][2,2] = -1 * self.mv[idx][2,2]
        sample['mv_left'] = self.mv_left[idx]
        sample['mv_right'] = self.mv_right[idx]
        sample['proj'] = self.proj[idx]
        sample['time'] = self.time_iterators[idx]
        sample['target_diffuse'] = self.target_diffuse[idx]
        sample['target_shil'] = self.target_shil[idx]
        # sample['target_shil_seg'] = self.target_shil_seg[idx]
        sample['target_complete_shil'] = self.target_complete_shil[idx]
        sample['hands_shil'] = self.target_hands_shil[idx]
        sample['target_depth'] = self.target_depth[idx]
        sample['target_norm_map'] = self.target_norm_map[idx]
        sample['tex_image'] = self.orig_image[idx]
        sample['reduced_pose'] = self.reduced_pose[idx]
        sample['reduced_pose_eight'] =  self.reduced_pose_eight[idx]
        sample['pose'] = self.pose[idx]
        sample['betas'] = self.betas[idx]
        # sample['translation'] = self.translation[idx]
        sample['idx'] = self.iterator_helper[idx]
        
        return sample


class MonocularTextureDataset(Dataset):

    def __init__(self, cfg):
        """
        Parameters:
        ----------
        cfg : 
            Configuration object.
        """
        start_end = [cfg.start_frame, cfg.end_frame]
        self.mv, self.proj, self.pose, self.betas = load_camera_and_smpl(cfg,cfg.smpl_pkl,start_end)
        self.time_iterators = torch.linspace(0, 1, len(self.mv)).to(DEVICE)
        self.target_image, self.background = get_target_imgs(cfg.target_images, cfg.image_size, start_end,cfg.skip_frames)
        self.target_diffuse = get_targets_diffuse(cfg.target_diffuse_maps, cfg.image_size, start_end,cfg.skip_frames)
        self.target_shil = get_targets_diffuse(cfg.target_shil_maps, cfg.image_size,start_end,cfg.skip_frames)
        self.target_complete_shil = get_targets_diffuse(cfg.target_complete_shil_maps, cfg.image_size,start_end,cfg.skip_frames)
        # self.target_norm_map = get_targets_npy(cfg.target_normal, cfg.image_size,start_end,cfg.skip_frames)
        self.iterator_helper = torch.arange(start_end[0],start_end[1],cfg.skip_frames)

    def __len__(self) -> int:
        return len(self.target_diffuse)

    def __getitem__(self, idx: int):
        
        sample = {}
        sample['mv'] = self.mv[idx]
        sample['proj'] = self.proj[idx]
        sample['time'] = self.time_iterators[idx]
        sample['target_diffuse'] = self.target_diffuse[idx]
        sample['target_shil'] = self.target_shil[idx]
        sample['target_complete_shil'] = self.target_complete_shil[idx]
        ## Maybe some extra manipulation is needed
        sample['target_image'] = self.target_image[idx]
        sample['target_background'] = self.background[idx]
        # sample['target_norm_map'] = self.target_norm_map[idx]
        sample['pose'] = self.pose[idx]
        sample['betas'] = self.betas[idx]
        sample['idx'] = self.iterator_helper[idx]
        
        return sample

def main():
    
    cfg = {
        'ROOT': {
            'smpl_pkl': '/hdd_data/nakul/soham/people_snapshot/people_snapshot_public/male-1-sport/smpl_fitting/sample_video/vibe_output.pkl',
            'target_diffuse_maps' : '/hdd_data/nakul/soham/people_snapshot/people_snapshot_public/male-1-sport/images_diffuse',
            'image_size' : 1080,
            'num_views': 64,
            'device': 'cuda:0',
            'resolution': (1080, 1080),
            'num_chunks': 16,
            'start_frame' : 0,
            'end_frame' : 64,
        },
        'DEVICE' : 'cuda'
    }
    
    dataset = MonocularDataset(EasyDict(cfg['ROOT']))
    dataLoader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    for batch in dataLoader:
        print(f"Model View shape: {batch['mv'].shape}")
        print(f"Proj shape: {batch['proj'].shape}")
        print(f"Diffuse shape: {batch['target_diffuse'].shape}")
        print(f"Pose Shape : {batch['pose'].shape}")
        print(f"Betas Shape : {batch['betas'].shape}")
        print(f"Time shape: {batch['time'].shape}")
        

if __name__=='__main__':
    main()
