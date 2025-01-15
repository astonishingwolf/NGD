import torch
import torchvision
from torchvision.utils import save_image
import os
import numpy
import trimesh

def save_any_image(image, path):
    if image.dim() == 4 and image.shape[3] == 3:
        save_image(image.permute(0, 3, 1, 2), path)
    elif image.dim() == 4 and image.shape[1] == 3:
        save_image(image, path)
    elif image.dim() == 3  and image.shape[2] == 3:
        save_image(image.unsqueeze(0).permute(0, 3, 1, 2), path)
    elif image.dim() == 3  and image.shape[0] == 3:
        save_image(image.unsqueeze(0), path)
    elif image.dim() == 3  and image.shape[0] == 1:
        save_image(image.unsqueeze(0).repeat(1,3,1,1), path)
    else:
        print(f"image shape {image.shape} not supported")

def save_mesh(verts, faces, path):
    if verts.dtype == torch.float32:
        mesh = trimesh.Trimesh(verts.detach().cpu().numpy(), faces.detach().cpu().numpy())
        mesh.export(path)