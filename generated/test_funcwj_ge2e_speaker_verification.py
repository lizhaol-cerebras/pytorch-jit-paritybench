
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


import torch as th


import numpy as np


import random


import math


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.utils.rnn import PackedSequence


from torch.nn.utils.rnn import pad_packed_sequence


import time


import logging


from collections import defaultdict


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.nn.utils import clip_grad_norm_


class TorchRNN(nn.Module):

    def __init__(self, feature_dim, rnn='lstm', num_layers=2, hidden_size=512, dropout=0.0, bidirectional=False):
        super(TorchRNN, self).__init__()
        RNN = rnn.upper()
        supported_rnn = {'LSTM': nn.LSTM, 'RNN': nn.RNN, 'GRU': nn.GRU}
        if RNN not in supported_rnn:
            raise RuntimeError('unknown RNN type: {}'.format(RNN))
        self.rnn = supported_rnn[RNN](feature_dim, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.output_dim = hidden_size if not bidirectional else hidden_size * 2

    def forward(self, x, squeeze=False, total_length=None):
        """
        Accept tensor([N]xTxF) or PackedSequence Object
        """
        is_packed = isinstance(x, PackedSequence)
        if not is_packed:
            if x.dim() not in [2, 3]:
                raise RuntimeError('RNN expect input dim as 2 or 3, got {:d}'.format(x.dim()))
            if x.dim() != 3:
                x = th.unsqueeze(x, 0)
        x, _ = self.rnn(x)
        if is_packed:
            x, _ = pad_packed_sequence(x, batch_first=True, total_length=total_length)
        if squeeze:
            x = th.squeeze(x)
        return x


class Nnet(nn.Module):

    def __init__(self, feature_dim=40, embedding_dim=256, lstm_conf=None):
        super(Nnet, self).__init__()
        self.encoder = TorchRNN(feature_dim, **lstm_conf)
        self.linear = nn.Linear(self.encoder.output_dim, embedding_dim)

    def forward(self, x):
        x = self.encoder(x)
        if x.dim() == 3:
            x = self.linear(x[:, -1, :])
        else:
            x = self.linear(x[-1, :])
        return x / th.norm(x, dim=-1, keepdim=True)


class GE2ELoss(nn.Module):

    def __init__(self):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(th.tensor(10.0))
        self.b = nn.Parameter(th.tensor(-5.0))

    def forward(self, e, N, M):
        """
        e: N x M x D, after L2 norm
        N: number of spks
        M: number of utts
        """
        c = th.mean(e, dim=1)
        s = th.sum(e, dim=1)
        e = e.view(N * M, -1)
        sim = th.mm(e, th.transpose(c, 0, 1))
        for j in range(N):
            for i in range(M):
                cj = (s[j] - e[j * M + i]) / (M - 1)
                sim[j * M + i][j] = th.dot(cj, e[j * M + i])
        sim = self.w * sim + self.b
        ref = th.zeros(N * M, dtype=th.int64, device=e.device)
        for r, s in enumerate(range(0, N * M, M)):
            ref[s:s + M] = r
        loss = F.cross_entropy(sim, ref)
        return loss


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (TorchRNN,
     lambda: ([], {'feature_dim': 4}),
     lambda: ([torch.rand([4, 4])], {})),
]

