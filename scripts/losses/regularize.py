import torch
import torch.nn as nn
import pytorch3d
import pytorch3d.loss
import pytorch3d.structures
import pytorch3d.transforms
from pytorch3d.loss import (
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

class Regularization(nn.Module):
    
    def __init__(self, weight, template, device = 'cuda'):
        super(Regularization, self).__init__()

        self.weight = weight
        self.template = template
        self.device = device
    
    def forward(self, predictions, targets):
        
        loss_edge = mesh_edge_loss(predictions.deformed_mesh_p3d)
        loss_normal = mesh_normal_consistency(predictions.deformed_mesh_p3d)
        loss_laplacian = mesh_laplacian_smoothing(predictions.deformed_mesh_p3d, method="uniform")
        # loss_laplacian_jacobians = me sh_laplacian_smoothing_jacobians(predictions)
        identity = torch.eye(3, 3, device = self.device)

        ## Previous Regularizer 
        ## Start here
        identity = torch.eye(3, 3, device = self.device)
        expanded_identity = identity.expand(predictions.iter_jacobians.shape)
        expanded_masks = ~predictions.face_masks.unsqueeze(-1).expand(-1, -1, 3)
        zeros = torch.zeros(3, 3, device = self.device)
        expanded_zeros = zeros.expand(predictions.residual_jacobians.shape)
        predictions.iter_jacobians = torch.where(
            expanded_masks,
            expanded_identity,
            predictions.iter_jacobians
        )
        expanded_masks_residual = predictions.face_masks.unsqueeze(-1).expand(-1, -1, 3)
        predictions.residual_jacobians = torch.where(
            expanded_masks_residual,
            expanded_zeros,
            predictions.residual_jacobians
        )
        U, S, Vh = torch.linalg.svd(predictions.iter_jacobians.clone().detach())
        rotation_transpose = torch.matmul(U, Vh)
        nearest_rotation = torch.transpose(rotation_transpose, -2, -1)
        # iter_loss = (((predictions.iter_jacobians) - nearest_rotation) ** 2).mean()
        iter_loss = (((predictions.iter_jacobians) - torch.eye(3, 3, device = self.device)) ** 2).mean()
        residual_loss = (((predictions.residual_jacobians) - torch.zeros(3, 3, device = self.device)) ** 2).mean()
        # r_loss = iter_loss + residual_loss
        ## End here


        ## New Regularizer
        ## Start here
        # iter_loss = (((predictions.iter_jacobians) - torch.eye(3, 3, device = self.device)) ** 2).mean()
        r_loss = iter_loss 
        ## Enf here
        
        loss = (loss_normal + loss_laplacian)*self.weight + r_loss * 0.25
        # breakpoint()
        loss_dict = {
            'loss_regularization' : loss,
            'loss_edge': loss_edge,
            'loss_normal': loss_normal,
            'loss_laplacian': loss_laplacian,
            'loss_jacobian': iter_loss,
            # 'loss_residual_jacobian': residual_loss,
        }
        
        return loss, loss_dict
