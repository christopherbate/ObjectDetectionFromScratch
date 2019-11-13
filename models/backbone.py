import torch
import numpy as np

class Backbone(torch.nn.Module):
    '''
    A sequence of residual blocks

    This is defined heuristically.

    Keep in mind - you need to downsample as you go along (either 
    by stride 2 conv layers or pooling)

    The sooner you downsample, the faster and smaller your network will be

    However, the sooner you downsample, the more spatial information you loose.
    '''
    def __init__(self, **kwargs):
        super(Backbone, self).__init__(**kwargs)

        self.first_conv = torch.nn.Conv2d(1,16, kernel_size=5)

        self.layers = torch.nn.ModuleList([
            ResBlock(1, 16, kernel_size=(3,3), stride=1,downsample=False)
        ])        

    def forward(self, x):

        out = self.first_conv(x)

        return x


class ResBlock(torch.nn.Module):
    '''
    ResBlock

    Implements a simple residual layer 

    Note the ordering of the batch norm w.r.t. the ReLU
    '''

    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            bias=False, groups=1, stride=stride, padding=(kernel_size[0]//2, kernel_size[1]//2))
        self.bn1 = torch.nn.BatchNorm2d(out_channels)

        self.conv2 = torch.nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
            bias=False, groups=1, stride=1, padding=(kernel_size[0]//2, kernel_size[1]//2))
        self.bn2 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        features = []
        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        features.append(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if(self.downsample):
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)

        features.append(out)

        return features



