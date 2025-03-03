from models import ResNet18WithSGE


class ResNet18WithSGEFeatureExtractor(ResNet18WithSGE):
    def __init__(self, num_classes, groups, pretrained=True):
        super(ResNet18WithSGEFeatureExtractor, self).__init__(num_classes, groups, pretrained)

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
        x = torch.flatten(x, 1)  # Flatten before FC layer (Feature Vector)
        return x 