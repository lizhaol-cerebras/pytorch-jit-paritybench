
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchvision, types, typing, uuid, warnings
import operator as op
from dataclasses import dataclass
import numpy as np
from torch import Tensor
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import math


import torch


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


import torchvision.transforms as transforms


import numpy as np


import torch.nn as nn


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Stem(nn.Module):

    def __init__(self, in_channels):
        super(Stem, self).__init__()
        self.conv2d_1a_3x3 = Conv2d(in_channels, 32, 3, stride=2, padding=0, bias=False)
        self.conv2d_2a_3x3 = Conv2d(32, 32, 3, stride=1, padding=0, bias=False)
        self.conv2d_2b_3x3 = Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.mixed_3a_branch_0 = nn.MaxPool2d(3, stride=2, padding=0)
        self.mixed_3a_branch_1 = Conv2d(64, 96, 3, stride=2, padding=0, bias=False)
        self.mixed_4a_branch_0 = nn.Sequential(Conv2d(160, 64, 1, stride=1, padding=0, bias=False), Conv2d(64, 96, 3, stride=1, padding=0, bias=False))
        self.mixed_4a_branch_1 = nn.Sequential(Conv2d(160, 64, 1, stride=1, padding=0, bias=False), Conv2d(64, 64, (1, 7), stride=1, padding=(0, 3), bias=False), Conv2d(64, 64, (7, 1), stride=1, padding=(3, 0), bias=False), Conv2d(64, 96, 3, stride=1, padding=0, bias=False))
        self.mixed_5a_branch_0 = Conv2d(192, 192, 3, stride=2, padding=0, bias=False)
        self.mixed_5a_branch_1 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        x = self.conv2d_1a_3x3(x)
        x = self.conv2d_2a_3x3(x)
        x = self.conv2d_2b_3x3(x)
        x0 = self.mixed_3a_branch_0(x)
        x1 = self.mixed_3a_branch_1(x)
        x = torch.cat((x0, x1), dim=1)
        x0 = self.mixed_4a_branch_0(x)
        x1 = self.mixed_4a_branch_1(x)
        x = torch.cat((x0, x1), dim=1)
        x0 = self.mixed_5a_branch_0(x)
        x1 = self.mixed_5a_branch_1(x)
        x = torch.cat((x0, x1), dim=1)
        return x


class Inception_ResNet_A(nn.Module):

    def __init__(self, in_channels, scale=1.0):
        super(Inception_ResNet_A, self).__init__()
        self.scale = scale
        self.branch_0 = Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False), Conv2d(32, 32, 3, stride=1, padding=1, bias=False))
        self.branch_2 = nn.Sequential(Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False), Conv2d(32, 48, 3, stride=1, padding=1, bias=False), Conv2d(48, 64, 3, stride=1, padding=1, bias=False))
        self.conv = nn.Conv2d(128, 320, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x_res = torch.cat((x0, x1, x2), dim=1)
        x_res = self.conv(x_res)
        return self.relu(x + self.scale * x_res)


class Inception_ResNet_B(nn.Module):

    def __init__(self, in_channels, scale=1.0):
        super(Inception_ResNet_B, self).__init__()
        self.scale = scale
        self.branch_0 = Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(Conv2d(in_channels, 128, 1, stride=1, padding=0, bias=False), Conv2d(128, 160, (1, 7), stride=1, padding=(0, 3), bias=False), Conv2d(160, 192, (7, 1), stride=1, padding=(3, 0), bias=False))
        self.conv = nn.Conv2d(384, 1088, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        return self.relu(x + self.scale * x_res)


class Reduciton_B(nn.Module):

    def __init__(self, in_channels):
        super(Reduciton_B, self).__init__()
        self.branch_0 = nn.Sequential(Conv2d(in_channels, 256, 1, stride=1, padding=0, bias=False), Conv2d(256, 384, 3, stride=2, padding=0, bias=False))
        self.branch_1 = nn.Sequential(Conv2d(in_channels, 256, 1, stride=1, padding=0, bias=False), Conv2d(256, 288, 3, stride=2, padding=0, bias=False))
        self.branch_2 = nn.Sequential(Conv2d(in_channels, 256, 1, stride=1, padding=0, bias=False), Conv2d(256, 288, 3, stride=1, padding=1, bias=False), Conv2d(288, 320, 3, stride=2, padding=0, bias=False))
        self.branch_3 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class Inception_ResNet_C(nn.Module):

    def __init__(self, in_channels, scale=1.0, activation=True):
        super(Inception_ResNet_C, self).__init__()
        self.scale = scale
        self.activation = activation
        self.branch_0 = Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False), Conv2d(192, 224, (1, 3), stride=1, padding=(0, 1), bias=False), Conv2d(224, 256, (3, 1), stride=1, padding=(1, 0), bias=False))
        self.conv = nn.Conv2d(448, 2080, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        if self.activation:
            return self.relu(x + self.scale * x_res)
        return x + self.scale * x_res


class Reduction_A(nn.Module):

    def __init__(self, in_channels, k, l, m, n):
        super(Reduction_A, self).__init__()
        self.branch_0 = Conv2d(in_channels, n, 3, stride=2, padding=0, bias=False)
        self.branch_1 = nn.Sequential(Conv2d(in_channels, k, 1, stride=1, padding=0, bias=False), Conv2d(k, l, 3, stride=1, padding=1, bias=False), Conv2d(l, m, 3, stride=2, padding=0, bias=False))
        self.branch_2 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        return torch.cat((x0, x1, x2), dim=1)


class Inception_ResNetv2(nn.Module):

    def __init__(self, in_channels=3, classes=1000, k=256, l=256, m=384, n=384):
        super(Inception_ResNetv2, self).__init__()
        blocks = []
        blocks.append(Stem(in_channels))
        for i in range(10):
            blocks.append(Inception_ResNet_A(320, 0.17))
        blocks.append(Reduction_A(320, k, l, m, n))
        for i in range(20):
            blocks.append(Inception_ResNet_B(1088, 0.1))
        blocks.append(Reduciton_B(1088))
        for i in range(9):
            blocks.append(Inception_ResNet_C(2080, 0.2))
        blocks.append(Inception_ResNet_C(2080, activation=False))
        self.features = nn.Sequential(*blocks)
        self.conv = Conv2d(2080, 1536, 1, stride=1, padding=0, bias=False)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1536, classes)

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.global_average_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class Inception_A(nn.Module):

    def __init__(self, in_channels):
        super(Inception_A, self).__init__()
        self.branch_0 = Conv2d(in_channels, 96, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(Conv2d(in_channels, 64, 1, stride=1, padding=0, bias=False), Conv2d(64, 96, 3, stride=1, padding=1, bias=False))
        self.branch_2 = nn.Sequential(Conv2d(in_channels, 64, 1, stride=1, padding=0, bias=False), Conv2d(64, 96, 3, stride=1, padding=1, bias=False), Conv2d(96, 96, 3, stride=1, padding=1, bias=False))
        self.brance_3 = nn.Sequential(nn.AvgPool2d(3, 1, padding=1, count_include_pad=False), Conv2d(384, 96, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.brance_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class Inception_B(nn.Module):

    def __init__(self, in_channels):
        super(Inception_B, self).__init__()
        self.branch_0 = Conv2d(in_channels, 384, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False), Conv2d(192, 224, (1, 7), stride=1, padding=(0, 3), bias=False), Conv2d(224, 256, (7, 1), stride=1, padding=(3, 0), bias=False))
        self.branch_2 = nn.Sequential(Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False), Conv2d(192, 192, (7, 1), stride=1, padding=(3, 0), bias=False), Conv2d(192, 224, (1, 7), stride=1, padding=(0, 3), bias=False), Conv2d(224, 224, (7, 1), stride=1, padding=(3, 0), bias=False), Conv2d(224, 256, (1, 7), stride=1, padding=(0, 3), bias=False))
        self.branch_3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), Conv2d(in_channels, 128, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class Reduction_B(nn.Module):

    def __init__(self, in_channels):
        super(Reduction_B, self).__init__()
        self.branch_0 = nn.Sequential(Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False), Conv2d(192, 192, 3, stride=2, padding=0, bias=False))
        self.branch_1 = nn.Sequential(Conv2d(in_channels, 256, 1, stride=1, padding=0, bias=False), Conv2d(256, 256, (1, 7), stride=1, padding=(0, 3), bias=False), Conv2d(256, 320, (7, 1), stride=1, padding=(3, 0), bias=False), Conv2d(320, 320, 3, stride=2, padding=0, bias=False))
        self.branch_2 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        return torch.cat((x0, x1, x2), dim=1)


class Inception_C(nn.Module):

    def __init__(self, in_channels):
        super(Inception_C, self).__init__()
        self.branch_0 = Conv2d(in_channels, 256, 1, stride=1, padding=0, bias=False)
        self.branch_1 = Conv2d(in_channels, 384, 1, stride=1, padding=0, bias=False)
        self.branch_1_1 = Conv2d(384, 256, (1, 3), stride=1, padding=(0, 1), bias=False)
        self.branch_1_2 = Conv2d(384, 256, (3, 1), stride=1, padding=(1, 0), bias=False)
        self.branch_2 = nn.Sequential(Conv2d(in_channels, 384, 1, stride=1, padding=0, bias=False), Conv2d(384, 448, (3, 1), stride=1, padding=(1, 0), bias=False), Conv2d(448, 512, (1, 3), stride=1, padding=(0, 1), bias=False))
        self.branch_2_1 = Conv2d(512, 256, (1, 3), stride=1, padding=(0, 1), bias=False)
        self.branch_2_2 = Conv2d(512, 256, (3, 1), stride=1, padding=(1, 0), bias=False)
        self.branch_3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), Conv2d(in_channels, 256, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x1_1 = self.branch_1_1(x1)
        x1_2 = self.branch_1_2(x1)
        x1 = torch.cat((x1_1, x1_2), 1)
        x2 = self.branch_2(x)
        x2_1 = self.branch_2_1(x2)
        x2_2 = self.branch_2_2(x2)
        x2 = torch.cat((x2_1, x2_2), dim=1)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class Inceptionv4(nn.Module):

    def __init__(self, in_channels=3, classes=1000, k=192, l=224, m=256, n=384):
        super(Inceptionv4, self).__init__()
        blocks = []
        blocks.append(Stem(in_channels))
        for i in range(4):
            blocks.append(Inception_A(384))
        blocks.append(Reduction_A(384, k, l, m, n))
        for i in range(7):
            blocks.append(Inception_B(1024))
        blocks.append(Reduction_B(1024))
        for i in range(3):
            blocks.append(Inception_C(1536))
        self.features = nn.Sequential(*blocks)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1536, classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_average_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Conv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Inception_B,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Inception_C,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Inceptionv4,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {})),
    (Reduciton_B,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Reduction_A,
     lambda: ([], {'in_channels': 4, 'k': 4, 'l': 4, 'm': 4, 'n': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Reduction_B,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Stem,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {})),
]

