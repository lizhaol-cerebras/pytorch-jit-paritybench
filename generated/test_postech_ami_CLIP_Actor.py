
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


import random


import copy


import torchvision


import torch.nn.functional as F


import warnings


from torchvision import transforms


import time


import torch.nn as nn


import torch.optim


from collections import *


from scipy import spatial


import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ProgressiveEncoding(nn.Module):

    def __init__(self, mapping_size, T, d=3, apply=True):
        super(ProgressiveEncoding, self).__init__()
        self._t = 0
        self.n = mapping_size
        self.T = T
        self.d = d
        self._tau = 2 * self.n / self.T
        self.indices = torch.tensor([i for i in range(self.n)], device=device)
        self.apply = apply

    def forward(self, x):
        alpha = ((self._t - self._tau * self.indices) / self._tau).clamp(0, 1).repeat(2)
        if not self.apply:
            alpha = torch.ones_like(alpha, device=device)
        alpha = torch.cat([torch.ones(self.d, device=device), alpha], dim=0)
        self._t += 1
        return x * alpha


class FourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10, exclude=0):
        super().__init__()
        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self.exclude = exclude
        B = torch.randn((num_input_channels, mapping_size)) * scale
        B_sort = sorted(B, key=lambda x: torch.norm(x, p=2))
        self._B = torch.stack(B_sort)

    def forward(self, x):
        batches, channels = x.shape
        assert channels == self._num_input_channels, 'Expected input to have {} channels (got {} channels)'.format(self._num_input_channels, channels)
        res = x @ self._B
        res = 2 * np.pi * res
        return torch.cat([x, torch.sin(res), torch.cos(res)], dim=1)


class NeuralStyleField(nn.Module):

    def __init__(self, sigma, depth, width, encoding, colordepth=2, normdepth=2, normratio=0.1, clamp=None, normclamp=None, niter=6000, input_dim=3, progressive_encoding=True, exclude=0):
        super(NeuralStyleField, self).__init__()
        self.pe = ProgressiveEncoding(mapping_size=width, T=niter, d=input_dim)
        self.clamp = clamp
        self.normclamp = normclamp
        self.normratio = normratio
        layers = []
        if encoding == 'gaussian':
            layers.append(FourierFeatureTransform(input_dim, width, sigma, exclude))
            if progressive_encoding:
                layers.append(self.pe)
            layers.append(nn.Linear(width * 2 + input_dim, width))
            layers.append(nn.ReLU())
        else:
            layers.append(nn.Linear(input_dim, width))
            layers.append(nn.ReLU())
        for i in range(depth):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        self.base = nn.Sequential(*layers)
        color_layers = []
        for _ in range(colordepth):
            color_layers.append(nn.Linear(width, width))
            color_layers.append(nn.ReLU())
        color_layers.append(nn.Linear(width, 3))
        self.mlp_rgb = nn.Sequential(*color_layers)
        normal_layers = []
        for _ in range(normdepth):
            normal_layers.append(nn.Linear(width, width))
            normal_layers.append(nn.ReLU())
        normal_layers.append(nn.Linear(width, 1))
        self.mlp_normal = nn.Sequential(*normal_layers)

    def reset_weights(self):
        self.mlp_rgb[-1].weight.data.zero_()
        self.mlp_rgb[-1].bias.data.zero_()
        self.mlp_normal[-1].weight.data.zero_()
        self.mlp_normal[-1].bias.data.zero_()

    def forward(self, x):
        for layer in self.base:
            x = layer(x)
        colors = self.mlp_rgb[0](x)
        for layer in self.mlp_rgb[1:]:
            colors = layer(colors)
        displ = self.mlp_normal[0](x)
        for layer in self.mlp_normal[1:]:
            displ = layer(displ)
        if self.clamp == 'tanh':
            colors = torch.tanh(colors) / 2
        elif self.clamp == 'clamp':
            colors = torch.clamp(colors, 0, 1)
        if self.normclamp == 'tanh':
            displ = torch.tanh(displ) * self.normratio
        elif self.normclamp == 'clamp':
            displ = torch.clamp(displ, -self.normratio, self.normratio)
        return colors, displ


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (FourierFeatureTransform,
     lambda: ([], {'num_input_channels': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (ProgressiveEncoding,
     lambda: ([], {'mapping_size': 4, 'T': 4}),
     lambda: ([torch.rand([4, 4, 4, 11])], {})),
]

