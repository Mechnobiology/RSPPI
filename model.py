import numpy as np
import pandas as pd
import pickle
import torch
import math
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv2d, MaxPool2d, AvgPool2d, InstanceNorm2d, Linear, ELU, Tanh, ReLU


class SPPLayer(nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        # num:样本数量 c:通道数 h:高 w:宽
        num, c, h, w = x.size()
        for i in range(self.num_levels):
            level = i + 1

            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.floor(h / level), math.floor(w / level))
            pooling = (
                math.floor((kernel_size[0] * level - h + 1) / 2), math.floor((kernel_size[1] * level - w + 1) / 2))

            h_new = 2 * pooling[0] + h
            w_new = 2 * pooling[1] + w
            kernel_size = (math.ceil(h_new / level), math.ceil(w_new / level))
            stride = (math.floor(h_new / level), math.floor(w_new / level))

            zero_pad = torch.nn.ZeroPad2d((pooling[1], pooling[1], pooling[0], pooling[0]))
            x_new = zero_pad(x)

            # 选择池化方式
            if self.pool_type == 'max_pool':
                try:
                    tensor = F.max_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)
                except Exception as e:
                    print(str(e))
                    print(x.size())
                    print(level)
            else:
                tensor = F.avg_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)

            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten
sppnet = SPPLayer(num_levels=4)


class Residual(nn.Module):  
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.InstanceNorm2d(num_channels)
        self.bn2 = nn.InstanceNorm2d(num_channels)

    def forward(self, X):
        Y = F.elu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.elu(Y)

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

    

class RSPPI(nn.Module):
    def __init__(self):
        super(RSPPI, self).__init__()

        self.model1 = nn.Sequential(
            nn.Sequential(*resnet_block(3, 16, 2)),
            nn.Sequential(*resnet_block(16, 32, 2)),
            nn.Sequential(*resnet_block(32, 64, 2))
        )
        self.model2 = nn.Sequential(
            Linear(1920*2, 960),
            ReLU(),
            Linear(960, 480),
            ReLU(),
            Linear(480,1)


        )


    def forward(self, x1, x2):
        x1 = self.model1(x1)
        x2 = self.model1(x2)
        spp1 = sppnet(x1)
        spp2 = sppnet(x2)
        spp_dif = torch.abs(spp1 - spp2)
        spp_mul = torch.mul(spp1, spp2)
        spp = torch.cat([spp_dif, spp_mul], 1)
        fc = self.model2(spp)
        active = nn.Sigmoid()
        output = active(fc)

        return output


