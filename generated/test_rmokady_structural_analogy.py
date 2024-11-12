
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


import torch.utils.data


import math


import torchvision


import random


import torchvision.utils as vutils


import numpy as np


from scipy.ndimage import filters


from scipy.ndimage import measurements


from scipy.ndimage import interpolation


from math import pi


import torch


import torch.nn as nn


import torch.optim as optim


import torch.utils.data as Data


from torch.utils.data import DataLoader


class ConvBlock(nn.Sequential):

    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd)),
        self.add_module('norm', nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))


class WDiscriminator(nn.Module):

    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, i + 1))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class GeneratorConcatSkip2CleanAdd(nn.Module):

    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, i + 1))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Sequential(nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size), nn.Tanh())

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:y.shape[2] - ind, ind:y.shape[3] - ind]
        return x + y


class Generator(nn.Module):

    def __init__(self, opt):
        super(Generator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, ker_size=3, padd=1, stride=1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, i + 1))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), ker_size=3, padd=1, stride=1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Sequential(nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=3, stride=1, padding=1), nn.Tanh())

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:y.shape[2] - ind, ind:y.shape[3] - ind]
        return x + y


class Generator_no_res(nn.Module):

    def __init__(self, opt):
        super(Generator_no_res, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im, N, ker_size=3, padd=1, stride=1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc / pow(2, i + 1))
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), ker_size=3, padd=1, stride=1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Sequential(nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=3, stride=1, padding=1), nn.Tanh())

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ConvBlock,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'ker_size': 4, 'padd': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Generator,
     lambda: ([], {'opt': SimpleNamespace(nfc=4, nc_im=4, num_layer=1, min_nfc=4)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Generator_no_res,
     lambda: ([], {'opt': SimpleNamespace(nfc=4, nc_im=4, num_layer=1, min_nfc=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (WDiscriminator,
     lambda: ([], {'opt': SimpleNamespace(nfc=4, nc_im=4, ker_size=4, padd_size=4, num_layer=1, min_nfc=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

