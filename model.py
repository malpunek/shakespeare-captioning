from torch import nn
from torchvision.models import resnet101


class FeatureExtractor(nn.Module):
    def __init__(self):
        self.resnet = resnet101(pretrained=True)
        self.resnet.fc = nn.Sequential()
        self.resnet.out_features = self.out_features = 2048

    def forward(self, x):
        return self.resnet(x)
