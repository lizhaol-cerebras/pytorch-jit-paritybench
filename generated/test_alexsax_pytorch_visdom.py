
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


import torch.optim as optim


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


class ShallowMLP(nn.Module):

    def __init__(self, shape, force_no_cuda=False):
        super(ShallowMLP, self).__init__()
        self.in_shape = shape[0]
        self.hidden_shape = shape[1]
        self.out_shape = shape[2]
        self.fc1 = nn.Linear(self.in_shape, self.hidden_shape)
        self.relu = F.relu
        self.fc2 = nn.Linear(self.hidden_shape, self.out_shape)
        self.use_cuda = torch.cuda.is_available() and not force_no_cuda
        if self.use_cuda:
            self = self

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        y = self.fc2(x)
        return y


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ShallowMLP,
     lambda: ([], {'shape': [4, 4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

