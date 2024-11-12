
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


import torch.nn as nn


import torch.optim as optim


import torchvision


import torchvision.transforms as transforms


from torchvision.utils import save_image


import numpy as np


import torch.autograd as autograd


class Discriminator(nn.Module):
    """全连接判别器，用于1x28x28的MNIST数据"""

    def __init__(self):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Linear(in_features=28 * 28, out_features=512, bias=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Linear(in_features=512, out_features=256, bias=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Linear(in_features=256, out_features=1, bias=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        validity = self.model(x)
        return validity


class Generator(nn.Module):
    """全连接生成器，用于1x28x28的MNIST数据"""

    def __init__(self, z_dim):
        super(Generator, self).__init__()
        layers = []
        layers.append(nn.Linear(in_features=z_dim, out_features=128))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Linear(in_features=128, out_features=256))
        layers.append(nn.BatchNorm1d(256, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Linear(in_features=256, out_features=512))
        layers.append(nn.BatchNorm1d(512, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Linear(in_features=512, out_features=28 * 28))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        x = self.model(z)
        x = x.view(-1, 1, 28, 28)
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Generator,
     lambda: ([], {'z_dim': 4}),
     lambda: ([torch.rand([4, 4])], {})),
]

