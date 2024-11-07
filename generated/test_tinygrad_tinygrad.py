
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


import math


import time


from typing import List


from typing import Dict


from typing import Optional


from typing import Union


from typing import Tuple


from typing import Callable


from torch.nn import functional as F


from torchvision import transforms as T


from torchvision.transforms import functional as Ft


import random


import torch


from torchvision.utils import make_grid


from torchvision.utils import save_image


from typing import cast


from functools import partial


from torch import nn


from torch import optim


from collections import namedtuple


from typing import Any


from itertools import chain


from collections import defaultdict


import functools


from scipy import signal


from scipy import ndimage


import torch.nn.functional as F


import torch.mps


import re


import torch.nn as nn


import scipy.ndimage


from torch.utils.data import Dataset


from torchvision import transforms


import copy


import warnings


from math import prod


from typing import get_args


from collections import OrderedDict


import itertools


import inspect


import string


from typing import ClassVar


from typing import Type


from typing import Sequence


from typing import DefaultDict


from typing import Literal


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 5)
        self.c2 = nn.Conv2d(32, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        self.m1 = nn.MaxPool2d(2)
        self.c3 = nn.Conv2d(32, 64, 3)
        self.c4 = nn.Conv2d(64, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.m2 = nn.MaxPool2d(2)
        self.lin = nn.Linear(576, 10)

    def forward(self, x):
        x = nn.functional.relu(self.c1(x))
        x = nn.functional.relu(self.c2(x), 0)
        x = self.m1(self.bn1(x))
        x = nn.functional.relu(self.c3(x), 0)
        x = nn.functional.relu(self.c4(x), 0)
        x = self.m2(self.bn2(x))
        return self.lin(torch.flatten(x, 1))


class Block(nn.Module):

    def __init__(self, in_dims, dims, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dims, dims, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(dims)
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(dims)
        self.downsample = []
        if stride != 1:
            self.downsample = [nn.Conv2d(in_dims, dims, kernel_size=1, stride=stride, bias=False), nn.BatchNorm(dims)]

    def __call__(self, x):
        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        for l in self.downsample:
            x = l(x)
        out += x
        out = nn.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, in_dims, dims, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_dims, dims, stride))
            in_dims = dims
        return layers

    def __call__(self, x):
        x = nn.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        for l in (self.layer1 + self.layer2 + self.layer3 + self.layer4):
            x = l(x)
        x = mx.mean(x, axis=[1, 2])
        x = self.fc(x)
        return x


def to_one_hot(array, layout, channel_axis):
    if len(array.shape) >= 5:
        array = torch.squeeze(array, dim=channel_axis)
    array = F.one_hot(array.long(), num_classes=3)
    if layout == 'NCDHW':
        array = array.permute(0, 4, 1, 2, 3).float()
    return array


class Dice:

    def __init__(self, to_onehot_y: 'bool'=True, to_onehot_x: 'bool'=False, use_softmax: 'bool'=True, use_argmax: 'bool'=False, include_background: 'bool'=False, layout: 'str'='NCDHW'):
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.to_onehot_x = to_onehot_x
        self.use_softmax = use_softmax
        self.use_argmax = use_argmax
        self.smooth_nr = 1e-06
        self.smooth_dr = 1e-06
        self.layout = layout

    def __call__(self, prediction, target):
        if self.layout == 'NCDHW':
            channel_axis = 1
            reduce_axis = list(range(2, len(prediction.shape)))
        else:
            channel_axis = -1
            reduce_axis = list(range(1, len(prediction.shape) - 1))
        num_pred_ch = prediction.shape[channel_axis]
        if self.use_softmax:
            prediction = torch.softmax(prediction, dim=channel_axis)
        elif self.use_argmax:
            prediction = torch.argmax(prediction, dim=channel_axis)
        if self.to_onehot_y:
            target = to_one_hot(target, self.layout, channel_axis)
        if self.to_onehot_x:
            prediction = to_one_hot(prediction, self.layout, channel_axis)
        if not self.include_background:
            assert num_pred_ch > 1, f'To exclude background the prediction needs more than one channel. Got {num_pred_ch}.'
            if self.layout == 'NCDHW':
                target = target[:, 1:]
                prediction = prediction[:, 1:]
            else:
                target = target[..., 1:]
                prediction = prediction[..., 1:]
        assert target.shape == prediction.shape, f'Target and prediction shape do not match. Target: ({target.shape}), prediction: ({prediction.shape}).'
        intersection = torch.sum(target * prediction, dim=reduce_axis)
        target_sum = torch.sum(target, dim=reduce_axis)
        prediction_sum = torch.sum(prediction, dim=reduce_axis)
        return (2.0 * intersection + self.smooth_nr) / (target_sum + prediction_sum + self.smooth_dr)


class DiceCELoss(nn.Module):

    def __init__(self, to_onehot_y, use_softmax, layout, include_background):
        super(DiceCELoss, self).__init__()
        self.dice = Dice(to_onehot_y=to_onehot_y, use_softmax=use_softmax, layout=layout, include_background=include_background)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim=1).long())
        dice = torch.mean(1.0 - self.dice(y_pred, y_true))
        return (dice + cross_entropy) / 2

