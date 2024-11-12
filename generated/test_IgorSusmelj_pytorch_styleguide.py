
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


from torch.utils import data


from torchvision import datasets


from torchvision import transforms


from torchvision import models


import numpy as np


import time


class MyModel(nn.Module):

    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(4 * 4 * 128, num_classes))

    def forward(self, x):
        feat = self.features(x)
        out = self.fc(feat)
        return out

