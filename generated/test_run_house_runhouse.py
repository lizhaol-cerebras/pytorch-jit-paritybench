
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


from typing import Dict


from typing import List


import torch


import time


from torch.nn.parallel import DistributedDataParallel as DDP


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torch.utils.data import DataLoader


from torchvision import datasets


from torchvision import transforms


import logging


from typing import Optional


from typing import Union


import copy


import re


import functools


import inspect


from enum import Enum


from typing import Callable


from typing import Type


import numpy as np


class TorchExampleBasic(nn.Module):

    def __init__(self):
        super(TorchExampleBasic, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (TorchExampleBasic,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 784])], {})),
]

