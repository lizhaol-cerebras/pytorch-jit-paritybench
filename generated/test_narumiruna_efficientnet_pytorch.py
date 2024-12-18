
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


from torch.utils import data


from torchvision import datasets


from torchvision import transforms


import warnings


from torch.utils.data import DataLoader


from torch.utils.data import DistributedSampler


from torchvision.datasets import ImageNet


from torch import Tensor


import math


import torch


from torch import nn


from torch.hub import load_state_dict_from_url


from torch import optim


import torch.nn.functional as F


from torch import distributed


import numpy as np


import tensorflow as tf


class Swish(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes: 'int', out_planes: 'int', kernel_size: 'int', stride: 'int'=1, groups: 'int'=1) ->None:
        padding = self._get_padding(kernel_size, stride)
        super().__init__(nn.ZeroPad2d(padding), nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=0, groups=groups, bias=False), nn.BatchNorm2d(out_planes), Swish())

    def _get_padding(self, kernel_size, stride):
        p = max(kernel_size - stride, 0)
        return [p // 2, p - p // 2, p // 2, p - p // 2]


class SqueezeExcitation(nn.Module):

    def __init__(self, in_planes: 'int', reduced_dim: 'int') ->None:
        super().__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_planes, reduced_dim, 1), Swish(), nn.Conv2d(reduced_dim, in_planes, 1), nn.Sigmoid())

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return x * self.se(x)


class MBConvBlock(nn.Module):

    def __init__(self, in_planes: 'int', out_planes: 'int', expand_ratio: 'int', kernel_size: 'int', stride: 'int', reduction_ratio: 'int'=4, drop_connect_rate: 'int'=0.2) ->None:
        super().__init__()
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = in_planes == out_planes and stride == 1
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        hidden_dim = in_planes * expand_ratio
        reduced_dim = max(1, int(in_planes / reduction_ratio))
        layers = []
        if in_planes != hidden_dim:
            layers += [ConvBNReLU(in_planes, hidden_dim, 1)]
        layers += [ConvBNReLU(hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim), SqueezeExcitation(hidden_dim, reduced_dim), nn.Conv2d(hidden_dim, out_planes, 1, bias=False), nn.BatchNorm2d(out_planes)]
        self.conv = nn.Sequential(*layers)

    def _drop_connect(self, x: 'torch.Tensor') ->torch.Tensor:
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        if self.use_residual:
            return x + self._drop_connect(self.conv(x))
        else:
            return self.conv(x)


def _make_divisible(value, divisor=8):
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def _round_filters(filters, width_mult):
    if width_mult == 1.0:
        return filters
    return int(_make_divisible(filters * width_mult))


def _round_repeats(repeats, depth_mult):
    if depth_mult == 1.0:
        return repeats
    return int(math.ceil(depth_mult * repeats))


class EfficientNet(nn.Module):

    def __init__(self, width_mult: 'float'=1.0, depth_mult: 'float'=1.0, dropout_rate: 'float'=0.2, num_classes: 'int'=1000) ->None:
        super().__init__()
        self.num_classes = num_classes
        settings = [[1, 16, 1, 1, 3], [6, 24, 2, 2, 3], [6, 40, 2, 2, 5], [6, 80, 3, 2, 3], [6, 112, 3, 1, 5], [6, 192, 4, 2, 5], [6, 320, 1, 1, 3]]
        out_channels = _round_filters(32, width_mult)
        features = [ConvBNReLU(3, out_channels, 3, stride=2)]
        in_channels = out_channels
        for t, c, n, s, k in settings:
            out_channels = _round_filters(c, width_mult)
            repeats = _round_repeats(n, depth_mult)
            for i in range(repeats):
                stride = s if i == 0 else 1
                features += [MBConvBlock(in_channels, out_channels, expand_ratio=t, stride=stride, kernel_size=k)]
                in_channels = out_channels
        last_channels = _round_filters(1280, width_mult)
        features += [ConvBNReLU(in_channels, last_channels, 1)]
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(last_channels, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ConvBNReLU,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (EfficientNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (MBConvBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'expand_ratio': 4, 'kernel_size': 3, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SqueezeExcitation,
     lambda: ([], {'in_planes': 4, 'reduced_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

