
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


import torchvision


from torch.utils.tensorboard import SummaryWriter


import numpy as np


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torchvision import transforms


import torch


from torch import nn


import torch.nn.functional as F


from torch.nn import Conv2d


from torch.nn import Linear


from torch.nn import L1Loss


from torch.nn import Sequential


from torch.nn import MaxPool2d


from torch.nn import Flatten


from torch.optim.lr_scheduler import StepLR


from torch.nn import ReLU


from torch.nn import Sigmoid


from torchvision.utils import make_grid


class Tudui(nn.Module):

    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(3, 32, 5, 1, 2), nn.MaxPool2d(2), nn.Conv2d(32, 32, 5, 1, 2), nn.MaxPool2d(2), nn.Conv2d(32, 64, 5, 1, 2), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(64 * 4 * 4, 64), nn.Linear(64, 10))

    def forward(self, x):
        x = self.model(x)
        return x

