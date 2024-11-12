
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


from torch.autograd import Variable


import time


import torch.utils.data.dataloader


import numpy as np


class Net(nn.Module):

    def __init__(self, hidden_size=7000, activation='leaky_relu'):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.activation = getattr(F, activation)
        if activation in ['relu', 'leaky_relu']:
            torch.nn.init.xavier_uniform(self.fc1.weight, gain=nn.init.calculate_gain(activation))
        else:
            torch.nn.init.xavier_uniform(self.fc1.weight, gain=1)
        self.fc2 = nn.Linear(hidden_size, 10, bias=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

    def forwardToHidden(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activation(x)
        return x

