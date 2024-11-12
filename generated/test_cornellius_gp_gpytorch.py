
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


import re


import warnings


from collections import OrderedDict


import torch


import torch.nn as nn


import torch.nn.functional as F


from copy import deepcopy


from functools import reduce


import matplotlib.pyplot as plt


import numpy as np


from torch.optim import Optimizer


from typing import Optional


from typing import Tuple


from typing import Union


from torch import Tensor


import math


from torch import sigmoid


from torch.nn import Module


import numbers


from torch.distributions import constraints


from torch.distributions.kl import register_kl


from torch.distributions import Distribution as TDistribution


from numbers import Number


from torch.distributions import MultivariateNormal as TMultivariateNormal


from torch.distributions.utils import _standard_normal


from torch.distributions.utils import lazy_property


from typing import Any


from torch.autograd import Function


from torch.distributions import Normal


from math import pi


from typing import Callable


from typing import List


from torch import nn


import copy


from abc import abstractmethod


from typing import Dict


from typing import Iterable


from torch.nn import ModuleList


from torch.nn.parallel import DataParallel


import logging


import functools


from functools import wraps


from torch.distributions import Bernoulli


from torch.distributions import Beta


from torch.distributions import Distribution


from torch.distributions import Laplace


from abc import ABC


from torch.distributions import Distribution as _Distribution


from torch.nn import Parameter


from torch.distributions import Categorical


from torch.distributions import StudentT


from torch.distributions import kl_divergence


import string


import inspect


import itertools


from torch.distributions import HalfCauchy


from torch.nn import Module as TModule


from torch.distributions import LKJCholesky


from typing import Mapping


from torch.distributions import TransformedDistribution


from torch.distributions.utils import broadcast_all


from torch.distributions import Gamma


from torch.distributions import HalfNormal


from torch.distributions import LogNormal


from torch.distributions import MultivariateNormal


from torch.distributions import Uniform


import random


from typing import Generator


from abc import abstractproperty


from torch import LongTensor


from torch.distributions.kl import kl_divergence


from torch.autograd.function import FunctionCtx


import abc


from torch.nn.functional import softplus


from torch import optim


from math import exp


from torch.utils.data import DataLoader


from torch.utils.data import TensorDataset


import time


from itertools import combinations


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class DenseNet(nn.Module):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5, num_init_features=24, bn_size=4, drop_rate=0, avgpool_size=8, num_classes=10):
        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = avgpool_size
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False))]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features.size(0), -1)
        out = self.classifier(out)
        return out


def inv_sigmoid(x):
    return torch.log(x) - torch.log(1 - x)


def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))


TRANSFORM_REGISTRY = {torch.exp: torch.log, torch.nn.functional.softplus: inv_softplus, torch.sigmoid: inv_sigmoid}


def _get_inv_param_transform(param_transform, inv_param_transform=None):
    reg_inv_tf = TRANSFORM_REGISTRY.get(param_transform, None)
    if reg_inv_tf is None:
        if inv_param_transform is None:
            raise RuntimeError('Must specify inv_param_transform for custom param_transforms')
        return inv_param_transform
    elif inv_param_transform is not None and reg_inv_tf != inv_param_transform:
        raise RuntimeError('TODO')
    return reg_inv_tf


def sq_dist(x1, x2, x1_eq_x2=False):
    """Equivalent to the square of `torch.cdist` with p=2."""
    adjustment = x1.mean(-2, keepdim=True)
    x1 = x1 - adjustment
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm)
    if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
        x2, x2_norm, x2_pad = x1, x1_norm, x1_pad
    else:
        x2 = x2 - adjustment
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        x2_pad = torch.ones_like(x2_norm)
    x1_ = torch.cat([-2.0 * x1, x1_norm, x1_pad], dim=-1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
    res = x1_.matmul(x2_.transpose(-2, -1))
    if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
        res.diagonal(dim1=-2, dim2=-1).fill_(0)
    return res.clamp_min_(0)


def dist(x1, x2, x1_eq_x2=False):
    """
    Equivalent to `torch.cdist` with p=2, but clamps the minimum element to 1e-15.
    """
    if not x1_eq_x2:
        res = torch.cdist(x1, x2)
        return res.clamp_min(1e-15)
    res = sq_dist(x1, x2, x1_eq_x2=x1_eq_x2)
    return res.clamp_min_(1e-30).sqrt_()


class Distance(torch.nn.Module):

    def __init__(self, postprocess: 'Optional[Callable]'=None):
        super().__init__()
        if postprocess is not None:
            warnings.warn('The `postprocess` argument is deprecated. See https://github.com/cornellius-gp/gpytorch/pull/2205 for details.', DeprecationWarning)
        self._postprocess = postprocess

    def _sq_dist(self, x1, x2, x1_eq_x2=False, postprocess=False):
        res = sq_dist(x1, x2, x1_eq_x2=x1_eq_x2)
        return self._postprocess(res) if postprocess else res

    def _dist(self, x1, x2, x1_eq_x2=False, postprocess=False):
        res = dist(x1, x2, x1_eq_x2=x1_eq_x2)
        return self._postprocess(res) if postprocess else res

