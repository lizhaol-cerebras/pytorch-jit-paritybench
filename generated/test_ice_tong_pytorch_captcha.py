
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


from torch.utils.data import Dataset


import torch.nn as nn


from torchvision.transforms import Compose


from torchvision.transforms import ToTensor


import matplotlib.pyplot as plot


from torch.autograd import Variable


from torch.utils.data import DataLoader


import time


class CNN(nn.Module):

    def __init__(self, num_class=36, num_char=4):
        super(CNN, self).__init__()
        self.num_class = num_class
        self.num_char = num_char
        self.conv = nn.Sequential(nn.Conv2d(3, 16, 3, padding=(1, 1)), nn.MaxPool2d(2, 2), nn.BatchNorm2d(16), nn.ReLU(), nn.Conv2d(16, 64, 3, padding=(1, 1)), nn.MaxPool2d(2, 2), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 512, 3, padding=(1, 1)), nn.MaxPool2d(2, 2), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, 3, padding=(1, 1)), nn.MaxPool2d(2, 2), nn.BatchNorm2d(512), nn.ReLU())
        self.fc = nn.Linear(512 * 11 * 6, self.num_class * self.num_char)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512 * 11 * 6)
        x = self.fc(x)
        return x

