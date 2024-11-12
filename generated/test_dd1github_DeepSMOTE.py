
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


import collections


import torch


import torch.nn as nn


from torch.utils.data import TensorDataset


import numpy as np


from sklearn.neighbors import NearestNeighbors


import time


class Encoder(nn.Module):

    def __init__(self, args):
        super(Encoder, self).__init__()
        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']
        self.conv = nn.Sequential(nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(self.dim_h * 2), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(self.dim_h * 4), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(self.dim_h * 8), nn.LeakyReLU(0.2, inplace=True))
        self.fc = nn.Linear(self.dim_h * 2 ** 3, self.n_z)

    def forward(self, x):
        x = self.conv(x)
        x = x.squeeze()
        x = self.fc(x)
        return x


class Decoder(nn.Module):

    def __init__(self, args):
        super(Decoder, self).__init__()
        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']
        self.fc = nn.Sequential(nn.Linear(self.n_z, self.dim_h * 8 * 7 * 7), nn.ReLU())
        self.deconv = nn.Sequential(nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4), nn.BatchNorm2d(self.dim_h * 4), nn.ReLU(True), nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4), nn.BatchNorm2d(self.dim_h * 2), nn.ReLU(True), nn.ConvTranspose2d(self.dim_h * 2, 1, 4, stride=2), nn.Tanh())

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.dim_h * 8, 7, 7)
        x = self.deconv(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Decoder,
     lambda: ([], {'args': SimpleNamespace(n_channel=4, dim_h=4, n_z=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

