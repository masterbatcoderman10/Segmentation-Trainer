#PyTorch imports
import torch
from torch.nn import functional as f
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision import transforms
from torch import nn

class DiceLoss(nn.Module):
    def __init__(self, num_classes, log_loss=False):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.log_loss = log_loss

    def forward(self, prediction, target):
        prediction = torch.softmax(prediction, dim=1)
        prediction = prediction.view(prediction.size(0), self.num_classes, -1)
        target = target.view(target.size(0), self.num_classes, -1)
        intersection = (prediction * target).sum(dim=-1)
        union = prediction.sum(dim=-1) + target.sum(dim=-1)
        dice = 2 * intersection / union

        if self.log_loss:
            dice = torch.log(dice)

        return 1 - dice.mean()

#For binary segmentation
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = f.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

#For multi-class segmentation
class DiceCE(nn.Module):
    def __init__(self, num_classes, class_weights=None,dice_weight=1, ce_weight=1, log_dice=False):
        super(DiceCE, self).__init__()
        self.dice_loss = DiceLoss(num_classes, log_dice)
        self.ce_loss = CrossEntropy4D(class_weights=class_weights)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
    def forward(self, prediction, target):
        dice_loss = self.dice_loss(prediction, target)
        ce_loss = self.ce_loss(prediction, target)
        return self.dice_weight*dice_loss + self.ce_weight*ce_loss