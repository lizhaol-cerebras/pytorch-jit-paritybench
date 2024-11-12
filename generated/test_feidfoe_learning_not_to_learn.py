
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


from torch.utils.data import DataLoader


import torchvision.transforms as transforms


import numpy as np


import torch.utils.data as data


import torchvision.datasets as datasets


from torch.backends import cudnn


import random


import torch.nn as nn


import math


import torch.utils.model_zoo as model_zoo


from torch import nn


from torch import optim


from torch.autograd import Variable


import time


class convnet(nn.Module):

    def __init__(self, num_classes=10):
        super(convnet, self).__init__()
        self.bn0 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        feat_out = x
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        feat_low = x
        feat_low = self.avgpool(feat_low)
        feat_low = feat_low.view(feat_low.size(0), -1)
        y_low = self.fc(feat_low)
        return feat_out, y_low


class Predictor(nn.Module):

    def __init__(self, input_ch=32, num_classes=8):
        super(Predictor, self).__init__()
        self.pred_conv1 = nn.Conv2d(input_ch, input_ch, kernel_size=3, stride=1, padding=1)
        self.pred_bn1 = nn.BatchNorm2d(input_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pred_conv2 = nn.Conv2d(input_ch, num_classes, kernel_size=3, stride=1, padding=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pred_conv1(x)
        x = self.pred_bn1(x)
        x = self.relu(x)
        x = self.pred_conv2(x)
        px = self.softmax(x)
        return x, px


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Predictor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 32, 64, 64])], {})),
]

