
import numpy as np
import pyrender
import os
import glob
from natsort import natsorted
import torch
import torchvision.transforms as transforms
import pickle
from PIL import Image
import cv2

class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scalget_targets_diffuse_erosion
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P

def get_targets_diffuse(normal_dir, image_size, start_end, skip, DEVICE = 'cuda'):
    
    normal_files = natsorted(glob.glob(os.path.join(normal_dir,'*.png')))
    # 
    diffuse_imgs = []
    
    start = start_end[0]
    end = start_end[1]
    
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size))
    ])
    normal_files = normal_files[start:end:skip]
    # 
    # breakpoint()
    for nor_file in normal_files:
        normal_img = Image.open(os.path.join(normal_dir,nor_file))
        normal_tensor = trans(normal_img)
        normal_tensor = normal_tensor.to(DEVICE)
        diffuse_imgs.append(normal_tensor)
        

    #         
    diffuse_img_stacked = torch.stack(diffuse_imgs)
    return diffuse_img_stacked

def get_targets_npy(npy_dir, image_size, start_end, skip, DEVICE='cuda'):
    npy_files = natsorted(glob.glob(os.path.join(npy_dir, '*.npy')))
    
    diffuse_data = []
    
    start = start_end[0]
    end = start_end[1]
    
    npy_files = npy_files[start:end:skip]
    # breakpoint()
    for npy_file in npy_files:
        data = np.load(npy_file)
        data_tensor = torch.from_numpy(data).float() 
        data_tensor = data_tensor.to(DEVICE)
        diffuse_data.append(data_tensor)
    
    diffuse_data_stacked = torch.stack(diffuse_data)
    return diffuse_data_stacked

def get_targets_normal(npy_dir, image_size, start_end, skip, DEVICE='cuda'):
    npy_files = natsorted(glob.glob(os.path.join(npy_dir, '*.npy')))
    
    diffuse_data = []
    
    start = start_end[0]
    end = start_end[1]
    
    npy_files = npy_files[start:end:skip]
    # breakpoint()
    for npy_file in npy_files:
        data = np.load(npy_file)
        data_tensor = torch.from_numpy(data).float() 
        data_tensor = data_tensor.to(DEVICE)
        # breakpoint()
        diffuse_data.append(data_tensor)
    
    diffuse_data_stacked = torch.stack(diffuse_data)
    return diffuse_data_stacked


def load_camera_and_smpl(cfg,smpl_pkl,start_end, DEVICE='cuda'):
    
    with open(smpl_pkl, 'rb') as f:
        smpl_data = pickle.load(f, encoding='latin1')
    
    mv_cat = []
    proj_cat = []
    pose_cat = []
    betas_cat = []
    
    for t in range(start_end[0],start_end[1],cfg.skip_frames):
        
        camera = WeakPerspectiveCamera(
                scale=[smpl_data['orig_cam'][t][0], smpl_data['orig_cam'][t][1]],
                translation=[smpl_data['orig_cam'][t][2], smpl_data['orig_cam'][t][3]],
                zfar=1000.
            )
        mv = np.eye(4)
        mv[2,2] = -1
        mv = torch.from_numpy(mv).float().to(DEVICE)
        betas = torch.from_numpy(smpl_data['betas'][t]).float().to(DEVICE)
        pose = torch.from_numpy(smpl_data['pose'][t]).float().to(DEVICE)
        proj = camera.get_projection_matrix(width=1080, height=1080)
        proj = torch.from_numpy(proj).float().to(DEVICE)
        
        mv_cat.append(mv)
        # pose = pose * (-1)
        proj_cat.append(proj)
        pose_cat.append(pose)
        betas_cat.append(betas)
        
        # pose[1] = -pose[1]
        # pose[2] = -pose[2]
        # breakpoint()
        # 
    mv_cat = torch.stack(mv_cat)
    proj_cat = torch.stack(proj_cat)
    pose_cat = torch.stack(pose_cat)
    betas_cat = torch.stack(betas_cat)
    
    return mv_cat, proj_cat, pose_cat, betas_cat

def load_camera_and_smpl_dress4d(cfg,smpl_pkl,start_end, DEVICE='cuda'):
    
    with open(smpl_pkl, 'rb') as f:
        smpl_data = pickle.load(f)
    
    mv_cat = []
    proj_cat = []
    pose_cat = []
    betas_cat = []
    global_orient_cat = []
    translation_cat = []


    for t in range(start_end[0],start_end[1],cfg.skip_frames):
        
        betas = torch.from_numpy(smpl_data[t]['betas']).float().to(DEVICE)
        pose = torch.from_numpy(smpl_data[t]['pose']).float().to(DEVICE)
        mv = torch.from_numpy(smpl_data[t]['mv']).float().to(DEVICE)
        proj = torch.from_numpy(smpl_data[t]['proj']).float().to(DEVICE)
        global_orient = torch.from_numpy(smpl_data[t]['global_orient']).float().to(DEVICE)
        translation = torch.from_numpy(smpl_data[t]['translation']).float().to(DEVICE)
        
        mv_cat.append(mv)
        proj_cat.append(proj)
        pose_cat.append(torch.cat((global_orient, pose),dim=0))
        betas_cat.append(betas)
        translation_cat.append(translation)

    mv_cat = torch.stack(mv_cat)
    proj_cat = torch.stack(proj_cat)
    pose_cat = torch.stack(pose_cat)
    betas_cat = torch.stack(betas_cat)
    translation_cat = torch.stack(translation_cat)

    return mv_cat, proj_cat, pose_cat, betas_cat, translation_cat

# get_targets_diffuse_erosion
def get_targets_diffuse_erosion(normal_dir, image_size, start_end, skip, DEVICE='cuda'):
    normal_files = natsorted(glob.glob(os.path.join(normal_dir, '*.png')))
    diffuse_imgs = []

    start, end = start_end
    normal_files = normal_files[start:end:skip]

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size))
    ])

    # Define erosion kernel
    kernel_size = 3  # Adjust this as needed
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for nor_file in normal_files:
        normal_img = cv2.imread(nor_file, cv2.IMREAD_GRAYSCALE)

        eroded_img = cv2.erode(normal_img, kernel, iterations=2)

        eroded_pil = Image.fromarray(eroded_img)

        normal_tensor = trans(eroded_pil).to(DEVICE)
        diffuse_imgs.append(normal_tensor)

    # Stack all tensors
    diffuse_img_stacked = torch.stack(diffuse_imgs)
    return diffuse_img_stacked