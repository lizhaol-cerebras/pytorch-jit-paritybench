
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


import numpy as np


import torch


import torch.utils.data as data


import torchvision.transforms as transforms


import torchvision.utils as vutils


import torch.nn as nn


import itertools


import torch.optim as optim


def get_norm(name, nc):
    if name == 'batchnorm':
        return nn.BatchNorm2d(nc)
    if name == 'instancenorm':
        return nn.InstanceNorm2d(nc)
    raise ValueError('Unsupported normalization layer: {:s}'.format(name))


class ResBlk(nn.Module):

    def __init__(self, n_in, n_out):
        super(ResBlk, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(n_in, n_out, 3, 1, 1), get_norm('batchnorm', n_out), nn.ReLU(inplace=True), nn.Conv2d(n_out, n_out, 3, 1, 1), get_norm('batchnorm', n_out))

    def forward(self, x):
        return self.layers(x) + x


def get_nonlinear(name):
    if name == 'relu':
        return nn.ReLU(inplace=True)
    if name == 'lrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    if name == 'sigmoid':
        return nn.Sigmoid()
    if name == 'tanh':
        return nn.Tanh()
    raise ValueError('Unsupported activation layer: {:s}'.format(name))


class _Generator(nn.Module):

    def __init__(self, input_channels, output_channels, last_nonlinear):
        super(_Generator, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_channels, 32, 7, 1, 3), get_norm('instancenorm', 32), get_nonlinear('relu'), nn.Conv2d(32, 64, 4, 2, 1), get_norm('instancenorm', 64), get_nonlinear('relu'), nn.Conv2d(64, 128, 4, 2, 1), get_norm('instancenorm', 128), get_nonlinear('relu'), nn.Conv2d(128, 256, 4, 2, 1), get_norm('instancenorm', 256), get_nonlinear('relu'))
        self.resblk = nn.Sequential(ResBlk(256, 256), ResBlk(256, 256), ResBlk(256, 256), ResBlk(256, 256))
        self.deconv = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), get_norm('instancenorm', 128), get_nonlinear('relu'), nn.ConvTranspose2d(128, 64, 4, 2, 1), get_norm('instancenorm', 64), get_nonlinear('relu'), nn.ConvTranspose2d(64, 32, 4, 2, 1), get_norm('instancenorm', 32), get_nonlinear('relu'), nn.ConvTranspose2d(32, output_channels, 7, 1, 3), get_nonlinear(last_nonlinear))

    def forward(self, x, a=None):
        if a is not None:
            assert a.dim() == 2 and x.size(0) == a.size(0)
            a = a.type(x.dtype)
            a = a.unsqueeze(2).unsqueeze(3).repeat(1, 1, x.size(2), x.size(3))
            x = torch.cat((x, a), dim=1)
        h = self.conv(x)
        h = self.resblk(h)
        y = self.deconv(h)
        return y


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.AMN = _Generator(4, 3, 'tanh')
        self.SAN = _Generator(3, 1, 'sigmoid')

    def forward(self, x, a):
        y = self.AMN(x, a)
        m = self.SAN(x)
        y_ = y * m + x * (1 - m)
        return y_, m


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1), get_nonlinear('lrelu'), nn.Conv2d(32, 64, 4, 2, 1), get_nonlinear('lrelu'), nn.Conv2d(64, 128, 4, 2, 1), get_nonlinear('lrelu'), nn.Conv2d(128, 256, 4, 2, 1), get_nonlinear('lrelu'), nn.Conv2d(256, 512, 4, 2, 1), get_nonlinear('lrelu'), nn.Conv2d(512, 1024, 4, 2, 1), get_nonlinear('lrelu'))
        self.src = nn.Conv2d(1024, 1, 3, 1, 1)
        self.cls = nn.Sequential(nn.Conv2d(1024, 1, 2, 1, 0), get_nonlinear('sigmoid'))

    def forward(self, x):
        h = self.conv(x)
        return self.src(h), self.cls(h).squeeze().unsqueeze(1)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Discriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {})),
    (ResBlk,
     lambda: ([], {'n_in': 4, 'n_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

