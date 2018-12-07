import torch
import torch.nn as nn
from torchvision import models

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.VGG = models.vgg19(pretrained=True)

    def forward(self, x):
        x = self.VGG.features(x)
        x = x.view(x.size(0), -1)
        x = self.VGG.classifier[0](x)
        x = self.VGG.classifier[1](x)
        x = self.VGG.classifier[2](x)
        x = self.VGG.classifier[3](x)
        x = self.VGG.classifier[4](x)

        return x

class Mapping(nn.Module):
    def __init__(self, att_size):
        super(Mapping, self).__init__()
        self.mapping = torch.nn.Sequential(
            torch.nn.Linear(4096, 1024),
            torch.nn.ReLU(),
            #torch.nn.Dropout(0)
            torch.nn.Linear(1024, att_size),
            #torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.mapping(x)
        return x
