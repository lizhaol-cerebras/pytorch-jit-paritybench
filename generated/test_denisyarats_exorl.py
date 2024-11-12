
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


import torch.nn as nn


import torch.nn.functional as F


from collections import OrderedDict


import functools


from collections import defaultdict


import torchvision


from torch.utils.tensorboard import SummaryWriter


import random


import copy


from torch.utils.data import IterableDataset


import warnings


import re


import time


import math


from torch import distributions as pyd


from torch.distributions.utils import _standard_normal


class Actor(nn.Module):

    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.policy = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, action_dim))
        self.apply(utils.weight_init)

    def forward(self, obs, std):
        mu = self.policy(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):

    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.q1_net = nn.Sequential(nn.Linear(obs_dim + action_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
        self.q2_net = nn.Sequential(nn.Linear(obs_dim + action_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.q1_net(obs_action)
        q2 = self.q2_net(obs_action)
        return q1, q2


class RandomShiftsAug(nn.Module):

    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (RandomShiftsAug,
     lambda: ([], {'pad': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

