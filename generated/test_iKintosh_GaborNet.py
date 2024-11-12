
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


from typing import Any


import torch


from torch.nn import Parameter


from torch.nn.modules import Conv2d


from torch.nn.modules import Module


import numpy as np


from torch.utils.data import Dataset


import time


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torch.utils.data import DataLoader


from torchvision import transforms


class GaborConv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'):
        super().__init__()
        self.is_calculated = False
        self.conv_layer = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.kernel_size = self.conv_layer.kernel_size
        self.delta = 0.001
        self.freq = Parameter(math.pi / 2 * math.sqrt(2) ** (-torch.randint(0, 5, (out_channels, in_channels))).type(torch.Tensor), requires_grad=True)
        self.theta = Parameter(math.pi / 8 * torch.randint(0, 8, (out_channels, in_channels)).type(torch.Tensor), requires_grad=True)
        self.sigma = Parameter(math.pi / self.freq, requires_grad=True)
        self.psi = Parameter(math.pi * torch.rand(out_channels, in_channels), requires_grad=True)
        self.x0 = Parameter(torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0], requires_grad=False)
        self.y0 = Parameter(torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0], requires_grad=False)
        self.y, self.x = torch.meshgrid([torch.linspace(-self.x0 + 1, self.x0 + 0, self.kernel_size[0]), torch.linspace(-self.y0 + 1, self.y0 + 0, self.kernel_size[1])])
        self.y = Parameter(self.y)
        self.x = Parameter(self.x)
        self.weight = Parameter(torch.empty(self.conv_layer.weight.shape, requires_grad=True), requires_grad=True)
        self.register_parameter('freq', self.freq)
        self.register_parameter('theta', self.theta)
        self.register_parameter('sigma', self.sigma)
        self.register_parameter('psi', self.psi)
        self.register_parameter('x_shape', self.x0)
        self.register_parameter('y_shape', self.y0)
        self.register_parameter('y_grid', self.y)
        self.register_parameter('x_grid', self.x)
        self.register_parameter('weight', self.weight)

    def forward(self, input_tensor):
        if self.training:
            self.calculate_weights()
            self.is_calculated = False
        if not self.training:
            if not self.is_calculated:
                self.calculate_weights()
                self.is_calculated = True
        return self.conv_layer(input_tensor)

    def calculate_weights(self):
        for i in range(self.conv_layer.out_channels):
            for j in range(self.conv_layer.in_channels):
                sigma = self.sigma[i, j].expand_as(self.y)
                freq = self.freq[i, j].expand_as(self.y)
                theta = self.theta[i, j].expand_as(self.y)
                psi = self.psi[i, j].expand_as(self.y)
                rotx = self.x * torch.cos(theta) + self.y * torch.sin(theta)
                roty = -self.x * torch.sin(theta) + self.y * torch.cos(theta)
                g = torch.exp(-0.5 * ((rotx ** 2 + roty ** 2) / (sigma + self.delta) ** 2))
                g = g * torch.cos(freq * rotx + psi)
                g = g / (2 * math.pi * sigma ** 2)
                self.conv_layer.weight.data[i, j] = g

    def _forward_unimplemented(self, *inputs: Any):
        """
        code checkers makes implement this method,
        looks like error in PyTorch
        """
        raise NotImplementedError


class GaborNN(nn.Module):

    def __init__(self):
        super(GaborNN, self).__init__()
        self.g1 = GaborConv2d(3, 32, kernel_size=(15, 15), stride=1)
        self.c1 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2)
        self.c2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2)
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.max_pool2d(F.leaky_relu(self.g1(x)), kernel_size=2)
        x = nn.Dropout2d()(x)
        x = F.max_pool2d(F.leaky_relu(self.c1(x)), kernel_size=2)
        x = F.max_pool2d(F.leaky_relu(self.c2(x)), kernel_size=2)
        x = nn.Dropout2d()(x)
        x = x.view(-1, 128 * 7 * 7)
        x = F.leaky_relu(self.fc1(x))
        x = nn.Dropout()(x)
        x = self.fc3(x)
        return x

    def _forward_unimplemented(self, *inputs: Any):
        """
        code checkers makes implement this method,
        looks like error in PyTorch
        """
        raise NotImplementedError


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (GaborConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GaborNN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 256, 256])], {})),
]

