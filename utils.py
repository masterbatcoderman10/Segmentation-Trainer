import torch
import torch.nn as nn


class Dice(nn.Module):
    def __init__(self, num_classes):
        super(Dice, self).__init__()
        self.num_classes = num_classes

    def forward(self, prediction, target):
        prediction = torch.softmax(prediction, dim=1)
        prediction = prediction.view(prediction.size(0), self.num_classes, -1)
        target = target.view(target.size(0), self.num_classes, -1)
        intersection = (prediction * target).sum(dim=-1)
        union = prediction.sum(dim=-1) + target.sum(dim=-1)
        dice = 2 * intersection / union

        return dice.mean()