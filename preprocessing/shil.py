import numpy as np
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import argparse
from natsort import natsorted

def save_image(img, path):
        img = img.cpu().numpy()
        img = img * 255.0
        img = img.astype(np.uint8)
        # 
        img = Image.fromarray(img)
        img.save(path)
        
def extract_shil(input_dir, output_dir):
    trans = transforms.Compose([
            transforms.ToTensor()
        ])
    image_list = []
    image_filenames = natsorted(os.listdir(input_dir))

    for image in image_filenames:
        if image.split('.')[-1] == 'png':
            image_path = os.path.join(input_dir, image)
            image_rgb = Image.open(image_path).convert('RGB')
            image_rgb = trans(image_rgb)
            image_norm = torch.norm(image_rgb, p=2, dim=0, keepdim=True)
            image_mask = torch.where(image_norm > 1e-2, 1.0, 0.0).permute(1,2,0)[..., [-1,-1,-1]] 
            # print(image_mask.shape)
            # image_mask
            # image = image * image_mask
            # 
            
            save_image(image_mask, os.path.join(output_dir, image.split('.')[0] +".png"))
            # image.save(os.path.join(output_dir, image.split('.')[0] +".png"))
            # image_list.append(image)
            # 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/home/nakul/soham/3d/fixnormalndepth/dataset/monocular/jumpingjacks/train/raw")
    parser.add_argument("--output_path", type=str, default="/home/nakul/soham/3d/fixnormalndepth/dataset/monocular/jumpingjacks/train/shil")
    args = parser.parse_args()
    input_dir = args.input_path
    output_dir = args.output_path
    extract_shil(input_dir, output_dir)