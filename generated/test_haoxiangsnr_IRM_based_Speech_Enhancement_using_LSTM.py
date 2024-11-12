
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


import random


import numpy as np


from torch.utils.data import Dataset


import torch


import torch.nn as nn


from torch.utils.data import DataLoader


import time


import matplotlib.pyplot as plt


from torch.nn.utils.rnn import pad_sequence


import math


from torch.utils.tensorboard import SummaryWriter


class LSTMModel(nn.Module):

    def __init__(self):
        """Construct LSTM model.
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size=161, hidden_size=1024, num_layers=2, batch_first=True, dropout=0.4)
        self.linear = nn.Linear(in_features=1024, out_features=161)
        self.activation = nn.Sigmoid()

    def forward(self, ipt):
        o, h = self.lstm(ipt)
        o = self.linear(o)
        o = self.activation(o)
        return o


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (LSTMModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 161])], {})),
]

