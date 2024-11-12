
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


import logging


import queue


import time


from typing import Any


import tensorflow as tf


from typing import Callable


from typing import Dict


from typing import List


from typing import Optional


from typing import Union


from itertools import chain


from itertools import repeat


import numpy as np


import torch


import torch.nn.functional as F


from torch import nn


from torch import optim


from torch.utils.data import DataLoader


from torchvision import datasets


from torchvision import transforms


from torch import utils


from torchvision.datasets import MNIST


from torchvision.transforms import ToTensor


import torch.nn as nn


import torchvision


import torchvision.transforms as transforms


from queue import Queue


from torch.utils.tensorboard.summary import histogram


from torch.utils.tensorboard.summary import histogram_raw


from torch.utils.tensorboard.summary import image


from torch.utils.tensorboard.summary import scalar


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.convlayer1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.convlayer2 = nn.Sequential(nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.fc1 = nn.Linear(64 * 6 * 6, 600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(600, 120)
        self.fc3 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = x.view(-1, 64 * 6 * 6)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class ConvNet(nn.Module):

    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (CNN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {})),
]

