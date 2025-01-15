import torch 
import torch.nn as nn

from scripts.losses import *
from scripts.losses.rendering_loss import RenderingLoss
from scripts.losses.regularize import Regularization
from scripts.losses.physics import StVKLoss,BendingLoss
from scripts.losses.utils import DepthDistillationLoss

class Loss(nn.Module):
    def __init__(self, config, template):

        super(Loss, self).__init__()
        
        self.loss_functions = []
        self.config = config
        self.template = template
        self._initialize_losses(self.config)
        
    def _initialize_losses(self,cfg):
        
        available_losses = {
            'RenderingLoss': RenderingLoss,
            'Regularization': Regularization,
            'DepthDistillation': DepthDistillationLoss,
        }

        for loss_name, loss_params in self.config.losses.items():
            if loss_name in available_losses:
                
                loss_class = available_losses[loss_name]
                weight = loss_params
                self.loss_functions.append(loss_class(weight=weight, \
                    template = self.template))

    def forward(self, pred, target):

        total_loss = 0.0
        global_loss_dict = {}

        for loss_fn in self.loss_functions:

            part_loss, loss_dict = loss_fn(pred, target)
            total_loss += part_loss
            global_loss_dict.update(loss_dict)

        return total_loss, global_loss_dict
