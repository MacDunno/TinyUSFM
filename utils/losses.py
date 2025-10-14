import torch  
import torch.nn as nn  


class MaskDiceLoss(nn.Module):
    def __init__(self):
        super(MaskDiceLoss, self).__init__()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, pred, target, weight=None, sigmoid=False):
        if sigmoid:
            pred = torch.sigmoid(pred) # b 1 h w
        assert pred.size() == target.size(), 'predict {} & target {} shape do not match'.format(pred.size(), target.size())
        dice_loss = self._dice_loss(pred[:, 0], target[:, 0])
        return dice_loss

class Mask_DC_and_BCE_loss(nn.Module):
    def __init__(self, dice_weight=0.8):
        super(Mask_DC_and_BCE_loss, self).__init__()

        self.ce =  torch.nn.BCEWithLogitsLoss()
        self.dc = MaskDiceLoss()
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        loss_ce = self.ce(pred, target)
        loss_dice = self.dc(pred, target, sigmoid=True)
        loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice

        return loss