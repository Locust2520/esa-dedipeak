from .dilate_loss import dilate_loss
from torch import nn

class DilateLoss(nn.Module):
    def __init__(self, alpha, gamma, device):
        super(DilateLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device

    def forward(self, outputs, targets):
        for c in range(outputs.shape[2]):
            loss, loss_shape, loss_temporal = dilate_loss(
                outputs[:,:,c:c+1],
                targets[:,:,c:c+1],
                self.alpha,
                self.gamma,
                self.device
            )
            if c == 0:
                loss_total = loss
            else:
                loss_total += loss
        return loss_total