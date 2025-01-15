import os
import yaml
import torch
from easydict import EasyDict

from scripts.geometry import geometry_training_loop
from scripts.appearance import appearance_training_loop
from scripts.inference import Inference

def loop(cfg):
    output_path = os.path.join(cfg['timeloop']['output_path'], cfg['timeloop']['Exp_name'])
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, 'config.yml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    cfg = EasyDict(cfg['timeloop'])
    device = cfg['device']
    torch.cuda.set_device(device)

    print("Running Geometry Training Loop")
    geometry_training_loop(cfg, device)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    if cfg.get('texture_recon', False):
        print("Running Texture Reconstruction")
        appearance_training_loop(cfg, device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    print("Evaluating Mesh")
    Inference(cfg, texture=cfg.get('texture_recon', False), device=device)

