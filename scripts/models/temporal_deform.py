import os
import sys
import numpy as np
import torch
import trimesh
import torchvision
import torchvision.transforms as transforms
import pytorch3d
import nvdiffrast.torch as dr
from easydict import EasyDict
import yaml
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from torch.utils.data import DataLoader
from types import SimpleNamespace
from torch.utils.tensorboard import SummaryWriter

from dataclasses import dataclass
from scripts.utils.cloth import Cloth
from scripts.utils import *
from scripts.models.model import Model
from scripts.utils.general_utils import load_mesh_path, calculate_face_centers, calculate_face_normals
from scripts.renderer.renderer import GTInitializer, AlphaRenderer
from scripts.losses.loss import Loss
from scripts.dataloader.Snap_data import MonocularDataset
from scripts.utils import *
from scripts.models import model
from scripts.models.template import Template
from scripts.models.core.NeuralJacobianFields import SourceMesh
from scripts.utils import smpl
from scripts.utils.general_utils import *
from scripts.utils.smpl_utils import *
from scripts.utils.cloth import Cloth
from scripts.models.remesh.remesh import *
from scripts.models.template import ClothModel
from scripts.models.model_utils import get_expon_lr_func

class ClothDeform:

    def __init__(self, cfg, cloth_template):
        self.mlp = Model(cfg,cloth_template)
        self.device = cfg.device
        self.mlp = self.mlp.to(self.device)
        self.jacobian_optimizer_residual = None
        self.jacobian_scheduler_residual = None
        self.spatial_lr_scale = 5
        self.position_lr_init = cfg.position_lr_init
        self.position_lr_final = cfg.position_lr_final
        self.position_lr_delay_mult = cfg.position_lr_delay_mult
        self.deform_lr_max_steps = cfg.deform_lr_max_steps
        self.optimizer = None

    def training_init(self):
        l = [
            {'params': list(self.mlp.parameters()),
             'lr': self.position_lr_init * self.spatial_lr_scale,
             "name": "deform"}
        ]

        # self.jacobian_optimizer_residual = torch.optim.Adam(self.mlp.parameters(), lr = 1e-3)
        # self.jacobian_scheduler_residual = torch.optim.lr_scheduler.LambdaLR(self.jacobian_optimizer_residual,lr_lambda=lambda x: max(0.01, 10**(-x*0.05)))
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.deform_scheduler_args = get_expon_lr_func(lr_init=self.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=self.position_lr_final,
                                                       lr_delay_mult=self.position_lr_delay_mult,
                                                       max_steps=self.deform_lr_max_steps)

    def forward(self, inputs):
        return self.mlp(inputs)

    def step(self, epochs, iterations, total_frames):
        self.optimizer.step()
        self.update_learning_rate((epochs+1)*total_frames + iterations)

    def save_weights(self, output_path, epoch = None):
        if epoch is not None:
            torch.save(self.mlp.state_dict(), os.path.join(output_path,'model_weights',f'model_epoch_{epoch}.pth'))
        else:
            torch.save(self.mlp.state_dict(), os.path.join(output_path,'model.pth'))

    def load_weights(self, model_path):
        self.mlp.load_state_dict(torch.load(model_path))
        self.mlp = self.mlp.to(self.device)
        
    def set_eval(self):
        self.mlp.eval()

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr