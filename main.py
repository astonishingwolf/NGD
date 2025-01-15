import os
import yaml
import torch
import random
import argparse
import numpy as np
from scripts import mono_recon_new

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config file', type=str, default='./example_config.yml')
    parser.add_argument('--output_path', help='Output directory (will be created)', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--gpu', help='GPU index', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--seed', help='Random seed', type=int, default=argparse.SUPPRESS)

    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as f:
            try:
                cfg = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(e)
    
    for key in vars(args):
        cfg[key] = vars(args)[key]

    print(yaml.dump(cfg, default_flow_style=False))
    random.seed(cfg['seed'])
    os.environ['PYTHONHASHSEED'] = str(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    torch.backends.cudnn.deterministic = True

    mono_recon_new.loop(cfg)
    print('Done')

if __name__ == '__main__':
    main()

