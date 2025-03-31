import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import math
from models.sge import SpatialGroupEnhance


class ResNet18WithSGE(nn.Module):
    def __init__(self, num_classes, groups, pretrained=True):
        super(ResNet18WithSGE, self).__init__()
        
        self.resnet = models.resnet18(pretrained=pretrained)
        self.layer0 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
        self.avgpool = self.resnet.avgpool
        in_features = self.resnet.fc.in_features  # Get the number of input features for the original FC layer
        self.fc = nn.Linear(in_features, num_classes)  # New FC layer
        
        self.module1 = SpatialGroupEnhance(groups[0])
        self.module2 = SpatialGroupEnhance(groups[1])
        self.module3 = SpatialGroupEnhance(groups[2])
        self.module4 = SpatialGroupEnhance(groups[3])
        nn.init.xavier_uniform_(self.fc.weight)


    def get_info(self):
        return 'ResNet18', 'SGE'
    

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.module1(x)
        x = self.layer2(x)
        x = self.module2(x)
        x = self.layer3(x)
        x = self.module3(x)
        x = self.layer4(x)
        x = self.module4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x