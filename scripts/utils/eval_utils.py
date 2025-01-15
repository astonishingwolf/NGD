import torch
import cv2
import numpy as np
import os
import trimesh
from tqdm import tqdm

def save_video(camera_images, video_save_path : str, save_file_name :str = "diffuse", height = 1080, width = 1080 , FPS: int = 10):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter(os.path.join(video_save_path, save_file_name + '.mp4'), fourcc, FPS, (width, height))

    for camera in tqdm(camera_images, desc = "extracting videos"):
        frame = camera.squeeze(0).cpu().numpy()
        # breakpoint()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
        
        out.write(frame)  # Write frame to video
    out.release()
    cv2.destroyAllWindows()