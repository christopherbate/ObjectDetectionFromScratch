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

    def __init__(self, layer_depths=[16, 32, 64], **kwargs):
        super(Backbone, self).__init__(**kwargs)

        self.first_conv = torch.nn.Conv2d(1, layer_depths[0], kernel_size=3,
                                          stride=1, padding=1)

        self.res_blks = torch.nn.ModuleList([
            ResBlock(layer_depths[0], layer_depths[1], kernel_size=(3, 3), stride=1, downsample=torch.nn.Conv2d(
                layer_depths[0], layer_depths[1], kernel_size=1, padding=0, stride=1)),
            ResBlock(layer_depths[1], layer_depths[2], kernel_size=(3, 3), stride=1, downsample=torch.nn.Conv2d(
                layer_depths[1], layer_depths[2], kernel_size=1, padding=0, stride=1))
        ])

        self.activation = torch.nn.ReLU()
        self.bns = torch.nn.ModuleList([
            torch.nn.BatchNorm2d(layer_depths[0])
        ])

        torch.nn.init.kaiming_normal_(self.first_conv.weight,
                                      mode='fan_out', nonlinearity='relu')

    def forward(self, x):

        out = self.first_conv(x)
        out = self.bns[0](out)
        out = self.activation(out)

        feature_maps = [out]

        for res_blk in self.res_blks:
            out = res_blk(out)
            feature_maps.append(out)

        return out, feature_maps


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

        torch.nn.init.kaiming_normal_(
            self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.bn1 = torch.nn.BatchNorm2d(out_channels)

        self.conv2 = torch.nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
            bias=False, groups=1, stride=1, padding=(kernel_size[0]//2, kernel_size[1]//2))

        torch.nn.init.kaiming_normal_(
            self.conv2.weight, mode='fan_out', nonlinearity='relu')
        self.bn2 = torch.nn.BatchNorm2d(num_features=out_channels)

        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if(self.downsample):
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)

        return out
