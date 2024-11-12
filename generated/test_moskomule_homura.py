
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


import torch.nn.functional as F


from torch.nn import functional as F


from torchvision.models import resnet50


import math


import warnings


from functools import partial


from torch.optim import lr_scheduler as _lr_scheduler


from torch import Tensor


from torch import nn


import copy


from typing import Iterator


import random


from torch.autograd import Function


from torch.distributions import RelaxedBernoulli


from torch.optim import Optimizer


from collections import defaultdict


from numbers import Number


from typing import Any


from typing import Callable


from torch import distributed


from abc import ABCMeta


from abc import abstractmethod


from functools import partial as Partial


from types import MethodType


from typing import Iterable


from typing import TypeVar


from torch.optim.lr_scheduler import _LRScheduler as Scheduler


from torch.utils.data import DataLoader


import numpy as np


from torch.utils.dlpack import from_dlpack


from torch.utils.dlpack import to_dlpack


import functools


import time


import types


from typing import Type


from functools import wraps


from torch.cuda import device_count


from collections.abc import Iterable


from torch.autograd import grad


import numpy


from torchvision import datasets


from torchvision import transforms


from torchvision import datasets as VD


from torchvision.io import read_image


from typing import Sequence


from torchvision.datasets.folder import IMG_EXTENSIONS


from torchvision.transforms.functional import to_tensor


import inspect


from typing import Protocol


from torch.utils.data import DistributedSampler


from torch.utils.data import RandomSampler


from torchvision import transforms as VT


from torchvision import models


from abc import ABC


from typing import Literal


from typing import Optional


from torchvision.transforms import functional as VF


from torchvision.transforms import transforms as VT


from torchvision.transforms import InterpolationMode


def kv_attention(query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor', mask: 'torch.Tensor'=None, additive_mask: 'torch.Tensor'=None, training: 'bool'=True, dropout_prob: 'float'=0, scaling: 'bool'=True) ->tuple[torch.Tensor, torch.Tensor]:
    """Attention using queries, keys and value

    :param query: `...JxM`
    :param key: `...KxM`
    :param value: `...KxM`
    :param mask: `...JxK`
    :param additive_mask:
    :param training:
    :param dropout_prob:
    :param scaling:
    :return: torch.Tensor whose shape of `...JxM`
    """
    if scaling:
        query /= query.size(-1) ** 0.5
    attn = torch.einsum('...jm,...km->...jk', query, key).softmax(dim=-1)
    if mask is not None:
        if mask.dim() < attn.dim():
            mask.unsqueeze_(0)
        attn = attn.masked_fill(mask == 0, 1e-09)
    if additive_mask is not None:
        attn += additive_mask
    if training and dropout_prob > 0:
        attn = F.dropout(attn, p=dropout_prob)
    return torch.einsum('...jk,...km->...jm', attn, value), attn


class KeyValAttention(nn.Module):
    """ Key-value attention.

    :param scaling:
    :param dropout_prob:
    """

    def __init__(self, scaling: 'bool'=False, dropout_prob: 'float'=0):
        super(KeyValAttention, self).__init__()
        self._scaling = scaling
        self._dropout = dropout_prob

    def forward(self, query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor', mask: 'torch.Tensor'=None, additive_mask: 'torch.Tensor'=None) ->tuple[torch.Tensor, torch.Tensor]:
        """ See `functional.attention.kv_attention` for details

        :param query:
        :param key:
        :param value:
        :param mask:
        :param additive_mask:
        :return:
        """
        return kv_attention(query, key, value, mask, additive_mask, self.training, self._dropout, self._scaling)


class AttentionPool2d(nn.Module):

    def __init__(self, embed_dim: 'int', num_heads: 'int'):
        super().__init__()
        self.k_proj = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.q_proj = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.v_proj = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.bias = nn.Parameter(torch.randn(3 * embed_dim))
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.initialize_weights()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = x.flatten(-2).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x, _ = F.multi_head_attention_forward(query=x, key=x, value=x, embed_dim_to_check=x.shape[-1], num_heads=self.num_heads, q_proj_weight=self.q_proj, k_proj_weight=self.k_proj, v_proj_weight=self.v_proj, in_proj_weight=None, in_proj_bias=self.bias, bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0, out_proj_weight=self.c_proj.weight, out_proj_bias=self.c_proj.bias, use_separate_proj_weight=True, training=self.training, need_weights=False)
        return x[0]

    def initialize_weights(self):
        std = self.c_proj.in_features ** -0.5
        nn.init.normal_(self.k_proj, std=std)
        nn.init.normal_(self.q_proj, std=std)
        nn.init.normal_(self.v_proj, std=std)
        nn.init.normal_(self.c_proj.weight, std=std)
        nn.init.zeros_(self.bias)


def gumbel_sigmoid(input: 'torch.Tensor', temp: 'float') ->torch.Tensor:
    """ gumbel sigmoid function
    """
    return RelaxedBernoulli(temp, probs=input.sigmoid()).rsample()


class GumbelSigmoid(nn.Module):
    """ This module outputs `gumbel_sigmoid` while training and `input.sigmoid() >= threshold` while evaluation
    """

    def __init__(self, temp: 'float'=0.1, threshold: 'float'=0.5):
        super(GumbelSigmoid, self).__init__()
        self.temp = temp
        self.threshold = threshold

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        if self.training:
            return gumbel_sigmoid(input, self.temp)
        else:
            return (input.sigmoid() >= self.threshold).float()


class _STE(Function):
    """ Straight Through Estimator
    """

    @staticmethod
    def forward(ctx, input: 'torch.Tensor') ->torch.Tensor:
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output: 'torch.Tensor') ->torch.Tensor:
        return F.hardtanh(grad_output)


def straight_through_estimator(input: 'torch.Tensor') ->torch.Tensor:
    """ straight through estimator

    >>> straight_through_estimator(torch.randn(3, 3))
    tensor([[0., 1., 0.],
            [0., 1., 1.],
            [0., 0., 1.]])
    """
    return _STE.apply(input)


class StraightThroughEstimator(nn.Module):

    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, input: 'torch.Tensor'):
        return straight_through_estimator(input)


def _saturated_sigmoid(input: 'torch.Tensor') ->torch.Tensor:
    return F.relu(1 - F.relu(1.1 - 1.2 * input.sigmoid()))


def semantic_hashing(input: 'torch.Tensor', is_training: 'bool') ->torch.Tensor:
    """ Semantic hashing

    >>> semantic_hashing(torch.randn(3, 3), True) # by 0.5
    tensor([[0.3515, 0.0918, 0.7717],
            [0.8246, 0.1620, 0.0689],
            [1.0000, 0.3575, 0.6598]])

    >>> semantic_hashing(torch.randn(3, 3), False)
    tensor([[0., 0., 1.],
            [0., 1., 1.],
            [0., 1., 1.]])
    """
    v1 = _saturated_sigmoid(input)
    v2 = (input < 0).float().detach()
    if not is_training or random.random() > 0.5:
        return v1 - v1.detach() + v2
    else:
        return v1


class SemanticHashing(nn.Module):

    def __init__(self):
        super(SemanticHashing, self).__init__()

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return semantic_hashing(input, self.training)


class EMA(nn.Module):
    """ Exponential moving average of a given model. ::

    model = EMA(original_model, 0.99999)

    :param original_model: Original model
    :param momentum: Momentum value for EMA
    :param copy_buffer: If true, copy float buffers instead of EMA
    """

    def __init__(self, original_model: 'nn.Module', momentum: 'float'=0.999, copy_buffer: 'bool'=False):
        super().__init__()
        if not 0 <= momentum <= 1:
            raise ValueError(f'Invalid momentum: {momentum}')
        self.momentum = momentum
        self.copy_buffer = copy_buffer
        self._original_model = original_model
        self._ema_model = copy.deepcopy(original_model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def __getattr__(self, item: 'str'):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self.original_model, item)

    @property
    def original_model(self) ->nn.Module:
        return self._original_model

    @property
    def ema_model(self) ->nn.Module:
        return self._ema_model

    def parameters(self, recurse: 'bool'=True) ->Iterator[nn.Parameter]:
        return self._original_model.parameters(recurse)

    def requires_grad_(self, requires_grad: 'bool'=True) ->nn.Module:
        self._original_model.requires_grad_(requires_grad)
        return self

    @torch.no_grad()
    def _update(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        o_p = [p.data for p in self._original_model.parameters() if isinstance(p, torch.Tensor)]
        e_p = [p.data for p in self._ema_model.parameters() if isinstance(p, torch.Tensor)]
        torch._foreach_mul_(e_p, self.momentum)
        torch._foreach_add_(e_p, o_p, alpha=1 - self.momentum)
        alpha = 0 if self.copy_buffer else self.momentum
        o_b = [b for b in self._original_model.buffers() if isinstance(b, torch.Tensor) and torch.is_floating_point(b)]
        if len(o_b) > 0:
            e_b = [b for b in self._ema_model.buffers() if isinstance(b, torch.Tensor) and torch.is_floating_point(b)]
            torch._foreach_mul_(e_b, alpha)
            torch._foreach_add_(e_b, o_b, alpha=1 - alpha)
        o_b = [b for b in self._original_model.buffers() if isinstance(b, torch.Tensor) and not torch.is_floating_point(b)]
        if len(o_b) > 0:
            e_b = [b for b in self._ema_model.buffers() if isinstance(b, torch.Tensor) and not torch.is_floating_point(b)]
            for o, e in zip(o_b, e_b):
                e.copy_(o)

    def forward(self, *args, **kwargs):
        if self.training:
            self._update()
            return self._original_model(*args, **kwargs)
        return self._ema_model(*args, **kwargs)

    def __repr__(self):
        s = f'EMA(beta={self.momentum},\n'
        s += f'  {self._original_model}\n'
        s += ')'
        return s


class _LossFunction(nn.Module):

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor') ->torch.Tensor:
        return self.impl(input, target)


def _reduction(input: 'torch.Tensor', reduction: 'str') ->torch.Tensor:
    if reduction == 'mean':
        return input.mean()
    elif reduction == 'sum':
        return input.sum()
    elif reduction == 'none' or reduction is None:
        return input
    else:
        raise NotImplementedError(f'Wrong reduction: {reduction}')


def cross_entropy_with_softlabels(input: 'torch.Tensor', target: 'torch.Tensor', dim: 'int'=1, reduction: 'str'='mean') ->torch.Tensor:
    """

    :param input:
    :param target:
    :param dim:
    :param reduction:
    :return:
    """
    if hasattr(torch.nn.CrossEntropyLoss, 'label_smoothing'):
        warnings.warn("Use PyTorch's F.cross_entropy", DeprecationWarning)
    if input.size() != target.size():
        raise RuntimeError(f'Input size ({input.size()}) and target size ({target.size()}) should be same!')
    return _reduction(-(input.log_softmax(dim=dim) * target).sum(dim=dim), reduction)


class SoftLabelCrossEntropy(_LossFunction):

    def __init__(self, dim: 'int'=1, reduction: 'str'='mean'):
        super().__init__()
        if hasattr(nn.CrossEntropyLoss, 'label_smoothing'):
            warnings.warn("Use PyTorch's nn.CrossEntropyLoss", DeprecationWarning)
        self.impl = partial(cross_entropy_with_softlabels, dim=dim, reduction=reduction)


def cross_entropy_with_smoothing(input: 'torch.Tensor', target: 'torch.Tensor', smoothing: 'float', dim: 'int'=1, reduction: 'str'='mean') ->torch.Tensor:
    """

    :param input:
    :param target:
    :param smoothing:
    :param dim:
    :param reduction:
    :return:
    """
    if hasattr(torch.nn.CrossEntropyLoss, 'label_smoothing'):
        warnings.warn("Use PyTorch's F.cross_entropy", DeprecationWarning)
    log_prob = input.log_softmax(dim=dim)
    nll_loss = -log_prob.gather(dim=dim, index=target.unsqueeze(dim=dim))
    nll_loss = nll_loss.squeeze(dim=dim)
    smooth_loss = -log_prob.mean(dim=dim)
    loss = (1 - smoothing) * nll_loss + smoothing * smooth_loss
    return _reduction(loss, reduction)


class SmoothedCrossEntropy(_LossFunction):

    def __init__(self, smoothing: 'float'=0.1, dim: 'int'=1, reduction: 'str'='mean'):
        super().__init__()
        if hasattr(nn.CrossEntropyLoss, 'label_smoothing'):
            warnings.warn("Use PyTorch's nn.CrossEntropyLoss", DeprecationWarning)
        self.impl = partial(cross_entropy_with_smoothing, smoothing=smoothing, dim=dim, reduction=reduction)


def conv1x1(in_planes: 'int', out_planes: 'int', stride=1, bias: 'bool'=False) ->nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)


class SELayer(nn.Module):

    def __init__(self, planes: 'int', reduction: 'int'):
        super().__init__()
        self.module = nn.Sequential(nn.AdaptiveAvgPool2d(1), conv1x1(planes, planes // reduction, bias=False), nn.ReLU(inplace=True), conv1x1(planes // reduction, planes, bias=False), nn.Sigmoid())

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return x * self.module(x)


def conv3x3(in_planes: 'int', out_planes: 'int', stride: 'int'=1, bias: 'bool'=False, groups: 'int'=1) ->nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: 'int', planes: 'int', stride: 'int', groups: 'int', width_per_group: 'int', norm: 'Type[nn.BatchNorm2d]', act: 'Callable[[torch.Tensor], torch.Tensor]'):
        super().__init__()
        planes = int(planes * (width_per_group / 16)) * groups
        self.conv1 = conv3x3(in_planes, planes, stride, bias=norm is None)
        self.conv2 = conv3x3(planes, planes, bias=norm is None)
        self.act = act
        self.norm1 = nn.Identity() if norm is None else norm(num_features=planes)
        self.norm2 = nn.Identity() if norm is None else norm(num_features=planes)
        self.downsample = nn.Identity()
        if in_planes != planes:
            self.downsample = nn.Sequential(conv1x1(in_planes, planes, stride=stride, bias=norm is None), nn.Identity() if norm is None else norm(num_features=planes))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += self.downsample(x)
        out = self.act(out)
        return out


class PreactBasicBlock(BasicBlock):

    def __init__(self, in_planes: 'int', planes: 'int', stride: 'int', groups: 'int', width_per_group: 'int', norm: 'Type[nn.BatchNorm2d]', act: 'Callable[[torch.Tensor], torch.Tensor]'):
        super().__init__(in_planes, planes, stride, groups, width_per_group, norm, act)
        self.norm1 = nn.Identity() if norm is None else norm(num_features=in_planes)
        if in_planes != planes:
            self.downsample = conv1x1(in_planes, planes, stride=stride, bias=norm is None)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        out = self.norm1(x)
        out = self.act(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.act(out)
        out = self.conv2(out)
        out += self.downsample(x)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes: 'int', planes: 'int', stride: 'int', groups: 'int', width_per_group: 'int', norm: 'Type[nn.BatchNorm2d]', act: 'Callable[[torch.Tensor], torch.Tensor]'):
        super().__init__()
        width = int(planes * (width_per_group / 64)) * groups
        self.conv1 = conv1x1(in_planes, width, bias=norm is None)
        self.conv2 = conv3x3(width, width, stride, groups=groups, bias=norm is None)
        self.conv3 = conv1x1(width, planes * self.expansion, bias=norm is None)
        self.act = act
        self.norm1 = nn.Identity() if norm is None else norm(width)
        self.norm2 = nn.Identity() if norm is None else norm(width)
        self.norm3 = nn.Identity() if norm is None else norm(planes * self.expansion)
        self.downsample = nn.Identity()
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(conv1x1(in_planes, planes * self.expansion, stride=stride, bias=norm is None), nn.Identity() if norm is None else norm(num_features=planes * self.expansion))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)
        out = self.conv3(out)
        out = self.norm3(out)
        out += self.downsample(x)
        return self.act(out)


class SEBasicBlock(BasicBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm2 = nn.Sequential(self.norm2, SELayer(self.conv2.out_channels, kwargs['reduction']))


class SEBottleneck(Bottleneck):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm3 = nn.Sequential(self.norm3, SELayer(self.conv3.out_channels, kwargs['reduction']))


def init_parameters(module: 'nn.Module'):
    """initialize parameters using kaiming normal"""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def initialization(module: 'nn.Module', use_zero_init: 'bool'):
    init_parameters(module)
    if use_zero_init:
        for m in module.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.norm2.weight, 0)


class ResNet(nn.Module):
    """ResNet for CIFAR data. For ImageNet classification, use `torchvision`'s.
    """

    def __init__(self, block: 'Type[BasicBlock | Bottleneck]', num_classes: 'int', layer_depth: 'int', width: 'int'=16, widen_factor: 'int'=1, in_channels: 'int'=3, groups: 'int'=1, width_per_group: 'int'=16, norm: 'Type[nn.BatchNorm2d]'=nn.BatchNorm2d, act: 'Callable[[torch.Tensor], torch.Tensor]'=nn.ReLU(), preact: 'bool'=False, final_pool: 'Callable[[torch.Tensor], torch.Tensor]'=nn.AdaptiveAvgPool2d(1), initializer: 'Callable[[nn.Module, None]]'=None):
        super(ResNet, self).__init__()
        self.inplane = width
        self.groups = groups
        self.norm = norm
        self.width_per_group = width_per_group
        self.preact = preact
        self.conv1 = conv3x3(in_channels, width, stride=1, bias=norm is None)
        self.norm1 = nn.Identity() if norm is None else norm(4 * width * block.expansion * widen_factor if self.preact else width)
        self.act = act
        self.layer1 = self._make_layer(block, width * widen_factor, layer_depth=layer_depth, stride=1)
        self.layer2 = self._make_layer(block, width * 2 * widen_factor, layer_depth=layer_depth, stride=2)
        self.layer3 = self._make_layer(block, width * 4 * widen_factor, layer_depth=layer_depth, stride=2)
        self.final_pool = final_pool
        self.fc = nn.Linear(4 * width * block.expansion * widen_factor, num_classes)
        if initializer is None:
            initialization(self, False)
        else:
            initializer(self)

    def _make_layer(self, block: 'Type[BasicBlock | Bottleneck]', planes: 'int', layer_depth: 'int', stride: 'int') ->nn.Sequential:
        layers = []
        for i in range(layer_depth):
            layers.append(block(self.inplane, planes, stride if i == 0 else 1, self.groups, self.width_per_group, self.norm, self.act))
            if i == 0:
                self.inplane = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if not self.preact:
            x = self.norm1(x)
            x = self.act(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.preact:
            x = self.norm1(x)
            x = self.act(x)
        x = self.final_pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class TVResNet(models.ResNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxpool = nn.Identity()


_padding = {'reflect': nn.ReflectionPad2d, 'zero': nn.ZeroPad2d}


class _DenseLayer(nn.Module):

    def __init__(self, in_channels, bn_size, growth_rate, dropout_rate, padding):
        super(_DenseLayer, self).__init__()
        assert padding in _padding.keys()
        self.dropout_rate = dropout_rate
        self.layers = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True), nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(bn_size * growth_rate), nn.ReLU(inplace=True), _padding[padding](1), nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, bias=False))

    def forward(self, input):
        x = self.layers(input)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return torch.cat([input, x], dim=1)


class _DenseBlock(nn.Module):

    def __init__(self, num_layers, in_channels, bn_size, growth_rate, dropout_rate, padding):
        super(_DenseBlock, self).__init__()
        layers = [_DenseLayer(in_channels + i * growth_rate, bn_size, growth_rate, dropout_rate, padding) for i in range(num_layers)]
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)


class _Transition(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(_Transition, self).__init__()
        self.layers = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True), nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False), nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, input):
        return self.layers(input)


class CIFARDenseNet(nn.Module):
    """
    DenseNet-BC (bottleneck and compactness) for CIFAR dataset. For ImageNet classification, use `torchvision`'s.

    :param num_classes: (int) number of output classes
    :param init_channels: (int) output channels which is performed on the input. 16 or 2 * growth_rate
    :param num_layers: (int) number of layers of each dense block
    :param growth_rate: (int) growth rate, which is referred as k in the paper
    :param dropout_rate: (float=0) dropout rate
    :param bn_size: (int=4) multiplicative factor in bottleneck
    :param reduction: (int=2) divisional factor in transition
    """

    def __init__(self, num_classes, init_channels, num_layers, growth_rate, dropout_rate=0, bn_size=4, reduction=2, padding='reflect'):
        super(CIFARDenseNet, self).__init__()
        num_channels = init_channels
        layers = [_padding[padding](1), nn.Conv2d(3, num_channels, kernel_size=3, bias=False)]
        for _ in range(2):
            layers.append(_DenseBlock(num_layers, in_channels=num_channels, bn_size=bn_size, growth_rate=growth_rate, dropout_rate=dropout_rate, padding=padding))
            num_channels = num_channels + num_layers * growth_rate
            layers.append(_Transition(num_channels, num_channels // reduction))
            num_channels = num_channels // reduction
        layers.append(_DenseBlock(num_layers, in_channels=num_channels, bn_size=bn_size, growth_rate=growth_rate, dropout_rate=dropout_rate, padding='reflect'))
        self.features = nn.Sequential(*layers)
        self.bn1 = nn.BatchNorm2d(num_channels + num_layers * growth_rate)
        self.linear = nn.Linear(num_channels + num_layers * growth_rate, num_classes)
        self.initialize()

    def forward(self, input):
        x = self.features(input)
        x = F.relu(self.bn1(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.linear(x)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


class Upsample(nn.Module):

    def __init__(self, scale_factor, mode):
        super(Upsample, self).__init__()
        assert mode in ('nearest', 'fc', 'bilinear', 'trilinear', 'area')
        self._scale_factor = scale_factor
        self._mode = mode

    def forward(self, input):
        return F.interpolate(input, scale_factor=self._scale_factor, mode=self._mode, align_corners=False)


class Block(nn.Module):

    def __init__(self, in_channel, out_channel):
        """
        >>> a = torch.randn(1, 1, 128, 128)
        >>> encoder = Block(1, 64)
        >>> encoder(a).size()
        torch.Size([1, 64, 128, 128])
        """
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True), nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))

    def forward(self, input):
        return self.block(input)


class UpsampleBlock(nn.Module):

    def __init__(self, in_channel, out_channel, upsample=True):
        """
        >>> a = torch.randn(1, 1, 128, 128)
        >>> encoder = Block(1, 64)
        >>> encoder(a).size()
        torch.Size([1, 64, 128, 128])
        """
        super().__init__()
        if upsample:
            self.upsample = nn.Sequential(Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1))
        else:
            self.upsample = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2)
        self.decoder = Block(in_channel, out_channel)

    def forward(self, input, bypass):
        x = self.upsample(input)
        _, _, i_h, i_w = x.shape
        _, _, b_h, b_w = bypass.shape
        pad = math.ceil((b_w - i_w) / 2), math.floor((b_w - i_w) / 2), math.ceil((b_h - i_h) / 2), math.floor((b_h - i_h) / 2)
        x = F.pad(x, pad)
        x = self.decoder(torch.cat([x, bypass], dim=1))
        return x


class DownsampleBlock(Block):

    def forward(self, input):
        input = F.max_pool2d(input, 2, 2)
        return self.block(input)


class UNet(nn.Module):

    def __init__(self, num_classes, input_channels, config=((64, 128, 256, 512, 1024), (1024, 512, 256, 128, 64))):
        """
        UNet, proposed in Ronneberger et al. (2015)
        :param num_classes: number of output classes
        :param input_channels: number of input channels
        """
        super(UNet, self).__init__()
        encoder_config, decoder_config = config
        encoder_config = list(encoder_config)
        decoder_config = list(decoder_config)
        encoder_config = list(zip([input_channels] + encoder_config[:-1], encoder_config))
        decoder_config = list(zip(decoder_config, decoder_config[1:]))
        self.encoders = nn.ModuleList([Block(*encoder_config[0])] + [DownsampleBlock(i, j) for i, j in encoder_config[1:]])
        self.decoders = nn.ModuleList([UpsampleBlock(i, j) for i, j in decoder_config])
        self.channel_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        init_parameters(self)

    def forward(self, input):
        down1 = self.encoders[0](input)
        down2 = self.encoders[1](down1)
        down3 = self.encoders[2](down2)
        down4 = self.encoders[3](down3)
        down5 = self.encoders[4](down4)
        up1 = self.decoders[0](down5, down4)
        up2 = self.decoders[1](up1, down3)
        up3 = self.decoders[2](up2, down2)
        up4 = self.decoders[3](up3, down1)
        return self.channel_conv(up4)


class CustomUNet(UNet):

    def forward(self, input):
        x = [input]
        for enc in self.encoders:
            x += [enc(x[-1])]
        x, *rest = reversed(x)
        for dec, _x in zip(self.decoders, rest):
            x = dec(x, _x)
        return self.channel_conv(x)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AttentionPool2d,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4, 'stride': 1, 'groups': 1, 'width_per_group': 4, 'norm': torch.nn.ReLU, 'act': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Block,
     lambda: ([], {'in_channel': 4, 'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CIFARDenseNet,
     lambda: ([], {'num_classes': 4, 'init_channels': 4, 'num_layers': 1, 'growth_rate': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (CustomUNet,
     lambda: ([], {'num_classes': 4, 'input_channels': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {})),
    (DownsampleBlock,
     lambda: ([], {'in_channel': 4, 'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GumbelSigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (KeyValAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (PreactBasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4, 'stride': 1, 'groups': 1, 'width_per_group': 4, 'norm': torch.nn.ReLU, 'act': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SELayer,
     lambda: ([], {'planes': 4, 'reduction': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SemanticHashing,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SoftLabelCrossEntropy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (StraightThroughEstimator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (UNet,
     lambda: ([], {'num_classes': 4, 'input_channels': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {})),
    (_Transition,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

