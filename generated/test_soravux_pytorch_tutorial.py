
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


import torch.optim as optim


from torchvision import datasets


from torchvision import transforms


from matplotlib import pyplot as plt


import numpy as np


import random


import torch.utils.data as data


import torchvision.models as models


from torch.autograd import Variable


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(512, 2, kernel_size=1)
        self.avgpool = nn.AvgPool2d(13)

    def forward(self, x):
        x = F.dropout(x, training=self.training)
        x = self.conv(x)
        x = self.avgpool(x)
        x = F.log_softmax(x)
        x = x.squeeze(dim=3).squeeze(dim=2)
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Net,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64])], {})),
]

