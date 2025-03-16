import torch.nn as nn
import segmentation_models_pytorch as smp

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.diceloss = smp.losses.DiceLoss(mode='binary')
        self.binloss = smp.losses.SoftBCEWithLogitsLoss(reduction='mean', smooth_factor=0.1)

    def forward(self, output, mask):
        dice = self.diceloss(output, mask)
        bce = self.binloss(output, mask)
        loss = dice * 0.7 + bce * 0.3
        return loss