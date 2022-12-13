import torch 
import torch.nn as nn
import torch.nn.functional as F

"""
Dice Similarity Coefficient 
"""
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predicted, target):

        # get number of predictions
        batch = predicted.size()[0]
        # initialize loss
        batch_loss = 0

        for index in range(batch):
            coef = self._dice_coefficient(predicted[index], target[index])
            batch_loss += coef

        batch_loss = batch_loss / batch
        return 1 - batch_loss

    def _dice_coefficient(self, predicted, target):
        smooth = 1
        product = torch.mul(predicted, target)
        intersection = product.sum()
        coef = (2*intersection + smooth) / (predicted.sum() + target.sum() + smooth) 
        return coef

"""
Binary Cross Entropy Loss
"""
class BCEDiceLoss(nn.Module):
    def __init__(self, device):
        super(BCEDiceLoss, self).__init__()
        self.dice_loss = DiceLoss().to(device)
    
    def forward(self, predicted, target):
        return F.binary_cross_entropy(predicted, target) + self.dice_loss(predicted, target)

    