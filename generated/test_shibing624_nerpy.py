
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


from torch.utils.data import Dataset


from collections import Counter


from typing import Optional


from typing import Union


from numbers import Real


import warnings


from torch.nn import CrossEntropyLoss


import collections


import math


import random


import numpy as np


import pandas as pd


from torch.utils.data import DataLoader


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


from torch.utils.data import TensorDataset


from torch.utils.tensorboard import SummaryWriter


class FeedForwardNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0):
        super(FeedForwardNetwork, self).__init__()
        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_proj = F.dropout(F.relu(self.linear1(x)), p=self.dropout_rate, training=self.training)
        x_proj = self.linear2(x_proj)
        return x_proj


class PoolerStartLogits(nn.Module):

    def __init__(self, hidden_size, num_classes):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, p_mask=None):
        x = self.dense(hidden_states)
        return x


class PoolerEndLogits(nn.Module):

    def __init__(self, hidden_size, num_classes):
        super(PoolerEndLogits, self).__init__()
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dense_1 = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, start_positions=None, p_mask=None):
        x = self.dense_0(torch.cat([hidden_states, start_positions], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x)
        return x


class FocalLoss(nn.Module):
    """Criterion that computes Focal loss.
    According to [1], the Focal loss is computed as follows:
    .. math::
        \\text{FL}(p_t) = -\\alpha_t (1 - p_t)^{\\gamma} \\, \\text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Arguments:
        alpha (float): Weighting factor :math:`\\alpha \\in [0, 1]` for one-vs-others mode (weight of negative class)
                        or :math:`\\alpha_i \\in \\R`
                        vector of weights for each class (analogous to weight argument for CrossEntropyLoss)
        gamma (float): Focusing parameter :math:`\\gamma >= 0`. When 0 is equal to CrossEntropyLoss
        reduction (Optional[str]): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’.
         ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
                in the output, uses geometric mean if alpha set to list of weights
         ‘sum’: the output will be summed. Default: ‘none’.
        ignore_index (Optional[int]): specifies indexes that are ignored during loss calculation
         (identical to PyTorch's CrossEntropyLoss 'ignore_index' parameter). Default: -100

    Shape:
        - Input: :math:`(N, C)` where C = number of classes.
        - Target: :math:`(N)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    Examples:
        >>> C = 5  # num_classes
        >>> N = 1 # num_examples
        >>> loss = FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
        >>> input = torch.randn(N, C, requires_grad=True)
        >>> target = torch.empty(N, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    References:
        [1] https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: 'Optional[Union[float, Iterable]]'=None, gamma: 'Real'=2.0, reduction: 'str'='mean', ignore_index: 'int'=-100) ->None:
        super(FocalLoss, self).__init__()
        if alpha is not None and not isinstance(alpha, float) and not isinstance(alpha, Iterable):
            raise ValueError(f'alpha value should be None, float value or list of real values. Got: {type(alpha)}')
        self.alpha: 'Optional[Union[float, torch.Tensor]]' = alpha if alpha is None or isinstance(alpha, float) else torch.FloatTensor(alpha)
        if isinstance(alpha, float) and not 0.0 <= alpha <= 1.0:
            warnings.warn('[Focal Loss] alpha value is to high must be between [0, 1]')
        self.gamma: 'Real' = gamma
        self.reduction: 'str' = reduction
        self.ignore_index: 'int' = ignore_index

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor') ->torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(input)))
        if input.shape[0] != target.shape[0]:
            raise ValueError(f'First dimension of inputs and targets should be same shape. Got: {input.shape} and {target.shape}')
        if len(input.shape) != 2 or len(target.shape) != 1:
            raise ValueError(f'input tensors should be of shape (N, C) and (N,). Got: {input.shape} and {target.shape}')
        if input.device != target.device:
            raise ValueError('input and target must be in the same device. Got: {}'.format(input.device, target.device))
        target = target.type(torch.long)
        input_mask = target != self.ignore_index
        target = target[input_mask]
        input = input[input_mask]
        pt = F.softmax(input, dim=1)
        logpt = F.log_softmax(input, dim=1)
        pt = pt.gather(1, target.unsqueeze(-1)).squeeze()
        logpt = logpt.gather(1, target.unsqueeze(-1)).squeeze()
        focal_loss = -1 * (1 - pt) ** self.gamma * logpt
        weights = torch.ones_like(focal_loss, dtype=focal_loss.dtype, device=focal_loss.device)
        if self.alpha is not None:
            if isinstance(self.alpha, float):
                alpha = torch.tensor(self.alpha, device=input.device)
                weights = torch.where(target > 0, 1 - alpha, alpha)
            elif torch.is_tensor(self.alpha):
                alpha = self.alpha
                weights = alpha.gather(0, target)
        tmp_loss = focal_loss * weights
        if self.reduction == 'none':
            loss = tmp_loss
        elif self.reduction == 'mean':
            loss = tmp_loss.sum() / weights.sum() if torch.is_tensor(self.alpha) else torch.mean(tmp_loss)
        elif self.reduction == 'sum':
            loss = tmp_loss.sum()
        else:
            raise NotImplementedError('Invalid reduction mode: {}'.format(self.reduction))
        return loss


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (FeedForwardNetwork,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PoolerStartLogits,
     lambda: ([], {'hidden_size': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

