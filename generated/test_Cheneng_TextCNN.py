
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


from torch.utils import data


import torch


import torch.nn as nn


import torch.autograd as autograd


import torch.nn.functional as F


class BasicModule(nn.Module):

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def forward(self):
        pass


class TextCNN(BasicModule):

    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.config = config
        self.out_channel = config.out_channel
        self.conv3 = nn.Conv2d(1, 1, (3, config.word_embedding_dimension))
        self.conv4 = nn.Conv2d(1, 1, (4, config.word_embedding_dimension))
        self.conv5 = nn.Conv2d(1, 1, (5, config.word_embedding_dimension))
        self.Max3_pool = nn.MaxPool2d((self.config.sentence_max_size - 3 + 1, 1))
        self.Max4_pool = nn.MaxPool2d((self.config.sentence_max_size - 4 + 1, 1))
        self.Max5_pool = nn.MaxPool2d((self.config.sentence_max_size - 5 + 1, 1))
        self.linear1 = nn.Linear(3, config.label_num)

    def forward(self, x):
        batch = x.shape[0]
        x1 = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x3 = F.relu(self.conv5(x))
        x1 = self.Max3_pool(x1)
        x2 = self.Max4_pool(x2)
        x3 = self.Max5_pool(x3)
        x = torch.cat((x1, x2, x3), -1)
        x = x.view(batch, 1, -1)
        x = self.linear1(x)
        x = x.view(-1, self.config.label_num)
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicModule,
     lambda: ([], {}),
     lambda: ([], {})),
]

