
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


import torchvision.transforms as transforms


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import numpy as np


import torch.nn as nn


import torch


import copy


import matplotlib


from torch.autograd import Variable


import torch.nn.functional as F


import torchvision.datasets as datasets


import matplotlib.pyplot as plt


import torch.optim as optim


from torchvision import datasets


from torchvision import transforms


class coarseNet(nn.Module):

    def __init__(self, init_weights=True):
        super(coarseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(12288, 4096)
        self.fc2 = nn.Linear(4096, 4070)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d()
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(-1, 1, 55, 74)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class fineNet(nn.Module):

    def __init__(self, init_weights=True):
        super(fineNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 63, kernel_size=9, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2)
        if init_weights:
            self._initialize_weights()

    def forward(self, x, y):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.cat((x, y), 1)
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

