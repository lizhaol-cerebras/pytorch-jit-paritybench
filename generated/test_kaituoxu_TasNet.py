
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


import numpy as np


import torch


import torch.utils.data as data


from itertools import permutations


import time


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


class Decoder(nn.Module):

    def __init__(self, N, L):
        super(Decoder, self).__init__()
        self.N, self.L = N, L
        self.basis_signals = nn.Linear(N, L, bias=False)

    def forward(self, mixture_w, est_mask, norm_coef):
        """
        Args:
            mixture_w: [B, K, N]
            est_mask: [B, K, nspk, N]
            norm_coef: [B, K, 1]
        Returns:
            est_source: [B, nspk, K, L]
        """
        source_w = torch.unsqueeze(mixture_w, 2) * est_mask
        est_source = self.basis_signals(source_w)
        norm_coef = torch.unsqueeze(norm_coef, 2)
        est_source = est_source * norm_coef
        est_source = est_source.permute((0, 2, 1, 3)).contiguous()
        return est_source


EPS = 1e-08


class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D gated conv layer.
    """

    def __init__(self, L, N):
        super(Encoder, self).__init__()
        self.L = L
        self.N = N
        self.conv1d_U = nn.Conv1d(L, N, kernel_size=1, stride=1, bias=False)
        self.conv1d_V = nn.Conv1d(L, N, kernel_size=1, stride=1, bias=False)

    def forward(self, mixture):
        """
        Args:
            mixture: [B, K, L]
        Returns:
            mixture_w: [B, K, N]
            norm_coef: [B, K, 1]
        """
        B, K, L = mixture.size()
        norm_coef = torch.norm(mixture, p=2, dim=2, keepdim=True)
        norm_mixture = mixture / (norm_coef + EPS)
        norm_mixture = torch.unsqueeze(norm_mixture.view(-1, L), 2)
        conv = F.relu(self.conv1d_U(norm_mixture))
        gate = torch.sigmoid(self.conv1d_V(norm_mixture))
        mixture_w = conv * gate
        mixture_w = mixture_w.view(B, K, self.N)
        return mixture_w, norm_coef


class Separator(nn.Module):
    """Estimation of source masks
    TODO: 1. normlization described in paper
          2. LSTM with skip connection
    """

    def __init__(self, N, hidden_size, num_layers, bidirectional=True, nspk=2):
        super(Separator, self).__init__()
        self.N = N
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.nspk = nspk
        self.layer_norm = nn.LayerNorm(N)
        self.rnn = nn.LSTM(N, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        fc_in_dim = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_in_dim, nspk * N)

    def forward(self, mixture_w, mixture_lengths):
        """
        Args:
            mixture_w: [B, K, N], padded
        Returns:
            est_mask: [B, K, nspk, N]
        """
        B, K, N = mixture_w.size()
        norm_mixture_w = self.layer_norm(mixture_w)
        total_length = norm_mixture_w.size(1)
        packed_input = pack_padded_sequence(norm_mixture_w, mixture_lengths, batch_first=True)
        packed_output, hidden = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=total_length)
        score = self.fc(output)
        score = score.view(B, K, self.nspk, N)
        est_mask = F.softmax(score, dim=2)
        return est_mask


class TasNet(nn.Module):

    def __init__(self, L, N, hidden_size, num_layers, bidirectional=True, nspk=2):
        super(TasNet, self).__init__()
        self.L, self.N = L, N
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.bidirectional = bidirectional
        self.nspk = nspk
        self.encoder = Encoder(L, N)
        self.separator = Separator(N, hidden_size, num_layers, bidirectional=bidirectional, nspk=nspk)
        self.decoder = Decoder(N, L)

    def forward(self, mixture, mixture_lengths):
        """
        Args:
            mixture: [B, K, L]
            mixture_lengths: [B]
        Returns:
            est_source: [B, nspk, K, L]
        """
        mixture_w, norm_coef = self.encoder(mixture)
        est_mask = self.separator(mixture_w, mixture_lengths)
        est_source = self.decoder(mixture_w, est_mask, norm_coef)
        return est_source

    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(package['L'], package['N'], package['hidden_size'], package['num_layers'], bidirectional=package['bidirectional'], nspk=package['nspk'])
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {'L': model.L, 'N': model.N, 'hidden_size': model.hidden_size, 'num_layers': model.num_layers, 'bidirectional': model.bidirectional, 'nspk': model.nspk, 'state_dict': model.state_dict(), 'optim_dict': optimizer.state_dict(), 'epoch': epoch}
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Decoder,
     lambda: ([], {'N': 4, 'L': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (Encoder,
     lambda: ([], {'L': 4, 'N': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (Separator,
     lambda: ([], {'N': 4, 'hidden_size': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4]), torch.ones([4], dtype=torch.int64)], {})),
    (TasNet,
     lambda: ([], {'L': 4, 'N': 4, 'hidden_size': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4]), torch.ones([4], dtype=torch.int64)], {})),
]

