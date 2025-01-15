import torch.nn as nn

class FashionLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
    
    def forward(self, predictions, targets):
        # Example: Custom loss that combines MSE and L1 loss
        mse_loss = nn.MSELoss()(predictions, targets)  # Mean Squared Error
        l1_loss = nn.L1Loss()(predictions, targets)    # L1 Loss (Mean Absolute Error)
        
        # You can weight the two losses if needed
        loss = mse_loss + 0.5 * l1_loss  # Custom combination of losses
        
        return loss