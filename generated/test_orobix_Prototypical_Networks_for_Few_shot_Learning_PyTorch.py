
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


import torch.utils.data as data


import numpy as np


import torch


import torch.nn as nn


from torch.nn import functional as F


from torch.nn.modules import Module


def conv_block(in_channels, out_channels):
    """
    returns a block conv-bn-relu-pool
    """
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(), nn.MaxPool2d(2))


class ProtoNet(nn.Module):
    """
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    """

    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(conv_block(x_dim, hid_dim), conv_block(hid_dim, hid_dim), conv_block(hid_dim, hid_dim), conv_block(hid_dim, z_dim))

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


def euclidean_dist(x, y):
    """
    Compute euclidean distance between two tensors
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(input, target, n_support):
    """
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    """
    target_cpu = target
    input_cpu = input

    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support
    support_idxs = list(map(supp_idxs, classes))
    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)
    query_samples = input[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()
    return loss_val, acc_val


class PrototypicalLoss(Module):
    """
    Loss class deriving from Module for the prototypical loss function defined below
    """

    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ProtoNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {})),
]

