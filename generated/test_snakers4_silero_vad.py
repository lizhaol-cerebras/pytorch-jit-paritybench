
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


import collections


import queue


import numpy as np


import torch


from typing import Callable


from typing import List


import warnings


import torch.nn as nn


from sklearn.metrics import roc_auc_score


from sklearn.metrics import accuracy_score


from torch.utils.data import Dataset


import pandas as pd


import random


class VADDecoderRNNJIT(nn.Module):

    def __init__(self):
        super(VADDecoderRNNJIT, self).__init__()
        self.rnn = nn.LSTMCell(128, 128)
        self.decoder = nn.Sequential(nn.Dropout(0.1), nn.ReLU(), nn.Conv1d(128, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x, state=torch.zeros(0)):
        x = x.squeeze(-1)
        if len(state):
            h, c = self.rnn(x, (state[0], state[1]))
        else:
            h, c = self.rnn(x)
        x = h.unsqueeze(-1).float()
        state = torch.stack([h, c])
        x = self.decoder(x)
        return x, state

