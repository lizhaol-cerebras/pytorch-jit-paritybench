
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


import torch.nn.functional as F


import numpy as np


class PCN1(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, dilation=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2, stride=1)
        self.rotate = nn.Conv2d(128, 2, kernel_size=1, stride=1)
        self.cls_prob = nn.Conv2d(128, 2, kernel_size=1, stride=1)
        self.bbox = nn.Conv2d(128, 3, kernel_size=1, stride=1)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        cls_prob = F.softmax(self.cls_prob(x), dim=1)
        rotate = F.softmax(self.rotate(x), dim=1)
        bbox = self.bbox(x)
        return cls_prob, rotate, bbox


class PCN2(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(40, 70, kernel_size=2, stride=1)
        self.fc = nn.Linear(70 * 3 * 3, 140)
        self.rotate = nn.Linear(140, 3)
        self.cls_prob = nn.Linear(140, 2)
        self.bbox = nn.Linear(140, 3)
        self.mp = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = F.relu(self.mp(x), inplace=True)
        x = self.conv2(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = F.relu(self.mp(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc(x), inplace=True)
        cls_prob = F.softmax(self.cls_prob(x), dim=1)
        rotate = F.softmax(self.rotate(x), dim=1)
        bbox = self.bbox(x)
        return cls_prob, rotate, bbox


class PCN3(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(48, 96, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(96, 144, kernel_size=2, stride=1)
        self.fc = nn.Linear(144 * 3 * 3, 192)
        self.cls_prob = nn.Linear(192, 2)
        self.bbox = nn.Linear(192, 3)
        self.rotate = nn.Linear(192, 1)
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = F.relu(self.mp1(x), inplace=True)
        x = self.conv2(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = F.relu(self.mp1(x), inplace=True)
        x = self.conv3(x)
        x = F.relu(self.mp2(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc(x), inplace=True)
        cls_prob = F.softmax(self.cls_prob(x), dim=1)
        rotate = self.rotate(x)
        bbox = self.bbox(x)
        return cls_prob, rotate, bbox


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (PCN1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

