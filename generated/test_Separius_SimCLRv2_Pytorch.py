
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


import torch


import numpy as np


import torch.nn as nn


import tensorflow as tf


import torch.nn.functional as F


from collections import Counter


import torchvision.transforms as transforms


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


BATCH_NORM_EPSILON = 1e-05


class BatchNormRelu(nn.Sequential):

    def __init__(self, num_channels, relu=True):
        super().__init__(nn.BatchNorm2d(num_channels, eps=BATCH_NORM_EPSILON), nn.ReLU() if relu else nn.Identity())


def conv(in_channels, out_channels, kernel_size=3, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=bias)


class SelectiveKernel(nn.Module):

    def __init__(self, in_channels, out_channels, stride, sk_ratio, min_dim=32):
        super().__init__()
        assert sk_ratio > 0.0
        self.main_conv = nn.Sequential(conv(in_channels, 2 * out_channels, stride=stride), BatchNormRelu(2 * out_channels))
        mid_dim = max(int(out_channels * sk_ratio), min_dim)
        self.mixing_conv = nn.Sequential(conv(out_channels, mid_dim, kernel_size=1), BatchNormRelu(mid_dim), conv(mid_dim, 2 * out_channels, kernel_size=1))

    def forward(self, x):
        x = self.main_conv(x)
        x = torch.stack(torch.chunk(x, 2, dim=1), dim=0)
        g = x.sum(dim=0).mean(dim=[2, 3], keepdim=True)
        m = self.mixing_conv(g)
        m = torch.stack(torch.chunk(m, 2, dim=1), dim=0)
        return (x * F.softmax(m, dim=0)).sum(dim=0)


class Projection(nn.Module):

    def __init__(self, in_channels, out_channels, stride, sk_ratio=0):
        super().__init__()
        if sk_ratio > 0:
            self.shortcut = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.AvgPool2d(kernel_size=2, stride=stride, padding=0), conv(in_channels, out_channels, kernel_size=1))
        else:
            self.shortcut = conv(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn = BatchNormRelu(out_channels, relu=False)

    def forward(self, x):
        return self.bn(self.shortcut(x))


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, sk_ratio=0, use_projection=False):
        super().__init__()
        if use_projection:
            self.projection = Projection(in_channels, out_channels * 4, stride, sk_ratio)
        else:
            self.projection = nn.Identity()
        ops = [conv(in_channels, out_channels, kernel_size=1), BatchNormRelu(out_channels)]
        if sk_ratio > 0:
            ops.append(SelectiveKernel(out_channels, out_channels, stride, sk_ratio))
        else:
            ops.append(conv(out_channels, out_channels, stride=stride))
            ops.append(BatchNormRelu(out_channels))
        ops.append(conv(out_channels, out_channels * 4, kernel_size=1))
        ops.append(BatchNormRelu(out_channels * 4, relu=False))
        self.net = nn.Sequential(*ops)

    def forward(self, x):
        shortcut = self.projection(x)
        return F.relu(shortcut + self.net(x))


class Blocks(nn.Module):

    def __init__(self, num_blocks, in_channels, out_channels, stride, sk_ratio=0):
        super().__init__()
        self.blocks = nn.ModuleList([BottleneckBlock(in_channels, out_channels, stride, sk_ratio, True)])
        self.channels_out = out_channels * BottleneckBlock.expansion
        for _ in range(num_blocks - 1):
            self.blocks.append(BottleneckBlock(self.channels_out, out_channels, 1, sk_ratio))

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class Stem(nn.Sequential):

    def __init__(self, sk_ratio, width_multiplier):
        ops = []
        channels = 64 * width_multiplier // 2
        if sk_ratio > 0:
            ops.append(conv(3, channels, stride=2))
            ops.append(BatchNormRelu(channels))
            ops.append(conv(channels, channels))
            ops.append(BatchNormRelu(channels))
            ops.append(conv(channels, channels * 2))
        else:
            ops.append(conv(3, channels * 2, kernel_size=7, stride=2))
        ops.append(BatchNormRelu(channels * 2))
        ops.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        super().__init__(*ops)


class ResNet(nn.Module):

    def __init__(self, layers, width_multiplier, sk_ratio):
        super().__init__()
        ops = [Stem(sk_ratio, width_multiplier)]
        channels_in = 64 * width_multiplier
        ops.append(Blocks(layers[0], channels_in, 64 * width_multiplier, 1, sk_ratio))
        channels_in = ops[-1].channels_out
        ops.append(Blocks(layers[1], channels_in, 128 * width_multiplier, 2, sk_ratio))
        channels_in = ops[-1].channels_out
        ops.append(Blocks(layers[2], channels_in, 256 * width_multiplier, 2, sk_ratio))
        channels_in = ops[-1].channels_out
        ops.append(Blocks(layers[3], channels_in, 512 * width_multiplier, 2, sk_ratio))
        channels_in = ops[-1].channels_out
        self.channels_out = channels_in
        self.net = nn.Sequential(*ops)
        self.fc = nn.Linear(channels_in, 1000)

    def forward(self, x, apply_fc=False):
        h = self.net(x).mean(dim=[2, 3])
        if apply_fc:
            h = self.fc(h)
        return h


class ContrastiveHead(nn.Module):

    def __init__(self, channels_in, out_dim=128, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i != num_layers - 1:
                dim, relu = channels_in, True
            else:
                dim, relu = out_dim, False
            self.layers.append(nn.Linear(channels_in, dim, bias=False))
            bn = nn.BatchNorm1d(dim, eps=BATCH_NORM_EPSILON, affine=True)
            if i == num_layers - 1:
                nn.init.zeros_(bn.bias)
            self.layers.append(bn)
            if relu:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for b in self.layers:
            x = b(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BatchNormRelu,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Blocks,
     lambda: ([], {'num_blocks': 4, 'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ContrastiveHead,
     lambda: ([], {'channels_in': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (Projection,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SelectiveKernel,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1, 'sk_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Stem,
     lambda: ([], {'sk_ratio': 4, 'width_multiplier': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

