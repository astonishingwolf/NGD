import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
from natsort import natsorted
import glob
LIGHT_DIR = [0 , 0. , 1.]
def generate_diffuse_map(shil_path, normal_path, save_dir : str = None):
    
    lightdir = np.array(LIGHT_DIR).reshape(1,1,3)
    normal_map = np.load(normal_path)
    shil_map = np.load(shil_path)
    basename = os.path.basename(normal_path)
    frame_id = basename.split('.')[0]
    diffuse = (lightdir*normal_map)
    diffuse = np.sum(diffuse, axis=2)
    diffuse = np.clip(diffuse, a_min=0.0, a_max=1.0)
    diffuse = diffuse[:,:,np.newaxis]   
    diffuse = diffuse[..., [0,0,0]]
    shil_cloth = (shil_map > 0).astype(np.uint8)
    shil_cloth = shil_cloth[..., np.newaxis]
    diffuse = diffuse*shil_cloth
    diffuse = diffuse * 255.0
    diffuse = diffuse.astype(np.uint8)
    np.save(os.path.join(save_dir,f'{frame_id}'), diffuse)
    image = Image.fromarray(diffuse) 
    image.save(os.path.join(save_dir,f'{frame_id}.png'))
    
def generate_shil_maps(shil_path, save_dir : str = None):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    shil_map = np.load(shil_path)
    basename = os.path.basename(shil_path)
    frame_id = basename.split('.')[0]
    shil = (shil_map == 22).astype(np.uint8)
    shil = shil[..., np.newaxis]
    shil = shil * 255.0
    shil = shil.astype(np.uint8)
    np.save(os.path.join(save_dir,f'{frame_id}'), shil)
    shil = shil[..., [-1,-1,-1]]
    # 
    image = Image.fromarray(shil) 
    image.save(os.path.join(save_dir,f'{frame_id}.png'))
    
def main():
    normal_path = '/hdd_data/nakul/soham/Dataset/people_snapshot_public_proprecess/anran_tic/sapien_norm'
    shil_path = '/hdd_data/nakul/soham/Dataset/people_snapshot_public_proprecess/anran_tic/sapien_seg'
    save_dir = '/hdd_data/nakul/soham/Dataset/people_snapshot_public_proprecess/anran_tic/diffuse'
    # save_dir_shils = '/hdd_data/nakul/soham/people_snapshot/people_snapshot_public/male-1-sport/images_shils'
    
    seg_files = natsorted(glob.glob(os.path.join(shil_path, '*_seg.npy')))
    normal_files = natsorted(glob.glob(os.path.join(normal_path, '*.npy')))
    for i,normal_file in enumerate(normal_files):
        print(i)
        generate_diffuse_map(seg_files[i], normal_file, save_dir)
    
    # for i,shil_file in enumerate(seg_files):
    #     generate_shil_maps(shil_file, save_dir_shils)

if __name__=='__main__':
    main()
