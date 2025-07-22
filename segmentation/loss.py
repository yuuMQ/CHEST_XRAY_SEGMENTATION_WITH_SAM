import torch
import torch.nn as nn
import torch.nn.functional as F

# DICE LOSS
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = 1 - (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return dice

# BCE LOSS
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss() # This loss combines a Sigmoid layer and the BCELoss in one single class
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = DiceLoss()(inputs, targets)
        return bce_loss + dice_loss
