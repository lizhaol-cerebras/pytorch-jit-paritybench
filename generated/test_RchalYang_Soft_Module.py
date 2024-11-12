
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


import time


import numpy as np


import random


import copy


import torch.nn.functional as F


import math


import torch.optim as optim


from torch import nn as nn


from collections import deque


import torch.multiprocessing as mp


from collections import OrderedDict


import torch.nn as nn


from torch.distributions import Normal


from torch.distributions import Distribution


from torch.multiprocessing import RawArray


def calc_next_shape(input_shape, conv_info):
    """
    take input shape per-layer conv-info as input
    """
    out_channels, kernel_size, stride, padding = conv_info
    c, h, w = input_shape
    h = int((h + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1)
    w = int((w + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1)
    return out_channels, h, w


class ZeroNet(nn.Module):

    def forward(self, x):
        return torch.zeros(1)


def null_activation(x):
    return x


class UniformPolicyContinuous(nn.Module):

    def __init__(self, action_shape):
        super().__init__()
        self.action_shape = action_shape

    def forward(self, x):
        return torch.Tensor(np.random.uniform(-1.0, 1.0, self.action_shape))

    def explore(self, x):
        return {'action': torch.Tensor(np.random.uniform(-1.0, 1.0, self.action_shape))}


LOG_SIG_MAX = 2


LOG_SIG_MIN = -20


class TanhNormal(Distribution):
    """
    Basically from RLKIT
    
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """

    def __init__(self, normal_mean, normal_std, epsilon=1e-06):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log((1 + value) / (1 - value)) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(1 - value * value + self.epsilon)

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        z = self.normal_mean + self.normal_std * Normal(torch.zeros(self.normal_mean.size()), torch.ones(self.normal_std.size())).sample()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def entropy(self):
        return self.normal.entropy()


class EmbeddingGuassianContPolicyBase:

    def eval_act(self, x, embedding_input):
        with torch.no_grad():
            mean, std, log_std = self.forward(x, embedding_input)
        return torch.tanh(mean.squeeze(0)).detach().cpu().numpy()

    def explore(self, x, embedding_input, return_log_probs=False, return_pre_tanh=False):
        mean, std, log_std = self.forward(x, embedding_input)
        dis = TanhNormal(mean, std)
        ent = dis.entropy().sum(-1, keepdim=True)
        dic = {'mean': mean, 'log_std': log_std, 'ent': ent}
        if return_log_probs:
            action, z = dis.rsample(return_pretanh_value=True)
            log_prob = dis.log_prob(action, pre_tanh_value=z)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            dic['pre_tanh'] = z.squeeze(0)
            dic['log_prob'] = log_prob
        else:
            if return_pre_tanh:
                action, z = dis.rsample(return_pretanh_value=True)
                dic['pre_tanh'] = z.squeeze(0)
            action = dis.rsample(return_pretanh_value=False)
        dic['action'] = action.squeeze(0)
        return dic

    def update(self, obs, embedding_input, actions):
        mean, std, log_std = self.forward(obs, embedding_input)
        dis = TanhNormal(mean, std)
        log_prob = dis.log_prob(actions).sum(-1, keepdim=True)
        ent = dis.entropy().sum(1, keepdim=True)
        out = {'mean': mean, 'log_std': log_std, 'log_prob': log_prob, 'ent': ent}
        return out


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (UniformPolicyContinuous,
     lambda: ([], {'action_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ZeroNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

