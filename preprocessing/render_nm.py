import argparse
import os 
import numpy as np
from PIL import Image
from natsort import natsorted
def save_image(img, path):
        # img = img.cpu().numpy()
        img = img * 255.0
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        img.save(path)
        
LIGHT_DIR = [0 , 0. , -1.]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/home/nakul/soham/3d/fixnormalndepth/dataset/monocular/jumpingjacks/train/normal_npy")
    parser.add_argument("--input_dir_shil", type=str, default="/home/nakul/soham/3d/fixnormalndepth/dataset/monocular/jumpingjacks/train/shil")
    parser.add_argument("--output_path", type=str, default="/home/nakul/soham/3d/fixnormalndepth/dataset/monocular/jumpingjacks/train/normal_light")
    args = parser.parse_args()
    input_dir = args.input_path
    input_dir_shil = args.input_dir_shil
    output_dir = args.output_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    normal_maps = []
    names = []
    file_names_normals = os.listdir(input_dir)
    file_names = natsorted(file_names_normals)
    file_names_shil = os.listdir(input_dir_shil)
    file_names_shils = natsorted(file_names_shil)
    # 
    for i,file in enumerate(file_names):
        if file.endswith(".npy"):
            npy = np.load(os.path.join(input_dir, file))
            file_name = file.split(".")[0]
            names.append(file_name)
            # npy = npy * 255
            # npy = npy.astype(np.uint8)
            # shil = Image.open(os.path.join(input_dir_shil, file_names_shils[i])).convert('RGB')
            # npy = npy / np.linalg.norm(npy, axis = -1, keepdims = True)
            npy[:,:, -1] = - npy[:,:, -1]
            # npy = npy*shil
            # print(npy)
            # 
            normal_maps.append(npy)

    lightdir = np.array(LIGHT_DIR).reshape(1,1,3)    

    for i,nmap in enumerate(normal_maps):
        # print(nmap.max())
        # nmap 
        diffuse = lightdir*nmap
        diffuse = np.sum(diffuse, axis=2)
        diffuse = np.clip(diffuse, a_min=0.0, a_max=1.0)
        diffuse = diffuse[:,:,np.newaxis]   
        diffuse = diffuse[..., [0,0,0]]
        shil = np.array(Image.open(os.path.join(input_dir_shil, file_names_shils[i])).convert('RGB'))
        shil = shil / 255
        diffuse = diffuse*shil
        print(f"{diffuse[:,:,-1].max()} and {diffuse[:,:,-1].min()}")
        # 
        save_image(diffuse, os.path.join(output_dir, names[i].split(".")[0] +".png"))
        
        
    
            # np.save(os.path.join(args.output_path, file), npy)
    # with open(args.input_path, "rb") as f: