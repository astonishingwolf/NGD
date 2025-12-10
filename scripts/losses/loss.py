"""
Loss function module for training.
"""
import torch.nn as nn
from scripts.losses.rendering_loss import RenderingLoss
from scripts.losses.regularize import Regularization
from scripts.losses.utils import DepthDistillationLoss


class Loss(nn.Module):
    """Composite loss function combining multiple loss terms."""
    
    def __init__(self, config, template):
        """Initialize loss module.
        
        Args:
            config: Configuration object with loss weights
            template: Template mesh/cloth object
        """
        super(Loss, self).__init__()
        self.loss_functions = []
        self.config = config
        self.template = template
        self._initialize_losses(config)
    
    def _initialize_losses(self, cfg):
        """Initialize loss functions from configuration.
        
        Args:
            cfg: Configuration object with losses dictionary
        """
        available_losses = {
            'RenderingLoss': RenderingLoss,
            'Regularization': Regularization,
            'DepthDistillation': DepthDistillationLoss,
        }

        for loss_name, loss_weight in cfg.losses.items():
            if loss_name in available_losses:
                loss_class = available_losses[loss_name]
                self.loss_functions.append(loss_class(weight=loss_weight, template=self.template))

    def forward(self, pred, target):
        """Compute total loss.
        
        Args:
            pred: Predictions namespace
            target: Targets namespace
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        total_loss = 0.0
        global_loss_dict = {}

        for loss_fn in self.loss_functions:
            part_loss, loss_dict = loss_fn(pred, target)
            total_loss += part_loss
            global_loss_dict.update(loss_dict)

        return total_loss, global_loss_dict
