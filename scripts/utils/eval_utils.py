"""
Evaluation and visualization utilities.
"""
import os
import cv2
import numpy as np
import torch
import trimesh
from tqdm import tqdm


def save_video(camera_images: list, video_save_path: str, save_file_name: str = "diffuse",
               height: int = 1080, width: int = 1080, fps: int = 10):
    """Save list of images as MP4 video.
    
    Args:
        camera_images: List of image tensors [C, H, W, 3] or [H, W, 3]
        video_save_path: Directory to save video
        save_file_name: Output filename (without extension)
        height: Video height
        width: Video width
        fps: Frames per second
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(video_save_path, f'{save_file_name}.mp4')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for camera in tqdm(camera_images, desc="Saving video"):
        frame = camera.squeeze(0).cpu().numpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
        out.write(frame)
    
    out.release()
    cv2.destroyAllWindows()


def save_mesh(vertices: torch.Tensor, faces: torch.Tensor, filepath: str):
    """Save mesh to OBJ file.
    
    Args:
        vertices: Vertex positions [V, 3]
        faces: Face indices [F, 3]
        filepath: Output file path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    mesh = trimesh.Trimesh(
        vertices=vertices.detach().cpu().numpy(),
        faces=faces.detach().cpu().numpy()
    )
    mesh.export(filepath)


def save_any_image(image: torch.Tensor, filepath: str):
    """Save image tensor to file. Handles various tensor formats.
    
    Args:
        image: Image tensor in various formats:
            - [H, W, 3] - single image
            - [C, H, W, 3] - batch of images (last dim is RGB)
            - [C, 3, H, W] - batch of images (standard format)
            - [3, H, W] - single image (standard format)
            - [1, H, W] - single grayscale image
        filepath: Output file path
    """
    from torchvision.utils import save_image
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if image.dim() == 4 and image.shape[-1] == 3:
        save_image(image.permute(0, 3, 1, 2), filepath)
    elif image.dim() == 4 and image.shape[1] == 3:
        save_image(image, filepath)
    elif image.dim() == 3 and image.shape[-1] == 3:
        save_image(image.unsqueeze(0).permute(0, 3, 1, 2), filepath)
    elif image.dim() == 3 and image.shape[0] == 3:
        save_image(image.unsqueeze(0), filepath)
    elif image.dim() == 3 and image.shape[0] == 1:
        save_image(image.unsqueeze(0).repeat(1, 3, 1, 1), filepath)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
