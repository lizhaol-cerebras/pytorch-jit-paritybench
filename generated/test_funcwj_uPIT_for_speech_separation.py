
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


import torch as th


from torch.nn.utils.rnn import pack_sequence


from torch.nn.utils.rnn import pad_sequence


from torch.nn.utils.rnn import PackedSequence


from torch.nn.utils.rnn import pad_packed_sequence


import scipy.io as sio


import time


import torch.nn.functional as F


from itertools import permutations


from torch.optim.lr_scheduler import ReduceLROnPlateau


class PITNet(th.nn.Module):

    def __init__(self, num_bins, rnn='lstm', num_spks=2, num_layers=3, hidden_size=896, dropout=0.0, non_linear='relu', bidirectional=True):
        super(PITNet, self).__init__()
        if non_linear not in ['relu', 'sigmoid', 'tanh']:
            raise ValueError('Unsupported non-linear type:{}'.format(non_linear))
        self.num_spks = num_spks
        rnn = rnn.upper()
        if rnn not in ['RNN', 'LSTM', 'GRU']:
            raise ValueError('Unsupported rnn type: {}'.format(rnn))
        self.rnn = getattr(th.nn, rnn)(num_bins, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.drops = th.nn.Dropout(p=dropout)
        self.linear = th.nn.ModuleList([th.nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_bins) for _ in range(self.num_spks)])
        self.non_linear = {'relu': th.nn.functional.relu, 'sigmoid': th.nn.functional.sigmoid, 'tanh': th.nn.functional.tanh}[non_linear]
        self.num_bins = num_bins

    def forward(self, x, train=True):
        is_packed = isinstance(x, PackedSequence)
        if not is_packed and x.dim() != 3:
            x = th.unsqueeze(x, 0)
        x, _ = self.rnn(x)
        if is_packed:
            x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.drops(x)
        m = []
        for linear in self.linear:
            y = linear(x)
            y = self.non_linear(y)
            if not train:
                y = y.view(-1, self.num_bins)
            m.append(y)
        return m

    def disturb(self, std):
        for p in self.parameters():
            noise = th.zeros_like(p).normal_(0, std)
            p.data.add_(noise)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (PITNet,
     lambda: ([], {'num_bins': 4}),
     lambda: ([torch.rand([4, 4])], {})),
]

