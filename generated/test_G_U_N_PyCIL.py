
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


from typing import Optional


from typing import Union


from typing import Callable


from abc import ABCMeta


from abc import abstractmethod


import math


import torch.nn as nn


import torch.nn.functional as F


from torch import nn


from torch.nn import functional as F


from typing import Dict


import torch.utils.model_zoo as model_zoo


import logging


import numpy as np


from typing import Any


from typing import Sized


from torch.utils.data import DataLoader


from torch.utils.data import Sampler


from torch import optim


import copy


from scipy.spatial.distance import cdist


from torch.utils.data import Dataset


from sklearn.svm import LinearSVC


from torchvision import datasets


from torchvision import transforms


from torch.serialization import load


from torch._C import device


import torch.optim as optim


import random


import collections


from enum import Enum


class Buffer(torch.nn.Module, metaclass=ABCMeta):

    def __init__(self) ->None:
        super().__init__()

    @abstractmethod
    def forward(self, X: 'torch.Tensor') ->torch.Tensor:
        raise NotImplementedError()


class RandomBuffer(torch.nn.Linear, Buffer):
    """
    Random buffer layer for the ACIL [1] and DS-AL [2].

    This implementation refers to the official implementation https://github.com/ZHUANGHP/Analytic-continual-learning.

    References:
    [1] Zhuang, Huiping, et al.
        "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection."
        Advances in Neural Information Processing Systems 35 (2022): 11602-11614.
    [2] Zhuang, Huiping, et al.
        "DS-AL: A Dual-Stream Analytic Learning for Exemplar-Free Class-Incremental Learning."
        Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 15. 2024.
    """

    def __init__(self, in_features: 'int', out_features: 'int', bias: 'bool'=False, device=None, dtype=torch.float, activation: 'Optional[activation_t]'=torch.relu_) ->None:
        super(torch.nn.Linear, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.activation: 'activation_t' = torch.nn.Identity() if activation is None else activation
        W = torch.empty((out_features, in_features), **factory_kwargs)
        b = torch.empty(out_features, **factory_kwargs) if bias else None
        self.register_buffer('weight', W)
        self.register_buffer('bias', b)
        self.reset_parameters()

    @torch.no_grad()
    def forward(self, X: 'torch.Tensor') ->torch.Tensor:
        X = X
        return self.activation(super().forward(X))


class DownsampleA(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class DownsampleB(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleB, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class DownsampleC(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleC, self).__init__()
        assert stride != 1 or nIn != nOut
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownsampleD(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleD, self).__init__()
        assert stride == 2
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=2, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ResNetBasicblock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, last=False):
        super(ResNetBasicblock, self).__init__()
        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)
        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.last = last

    def forward(self, x):
        residual = x
        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)
        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = residual + basicblock
        if not self.last:
            out = F.relu(out, inplace=True)
        return out


class CifarResNet(nn.Module):
    """
    ResNet optimized for the Cifar Dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(self, block, depth, channels=3):
        super(CifarResNet, self).__init__()
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6
        self.conv_1_3x3 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)
        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2, last_phase=True)
        self.avgpool = nn.AvgPool2d(8)
        self.out_dim = 64 * block.expansion
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, last_phase=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleB(self.inplanes, planes * block.expansion, stride)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if last_phase:
            for i in range(1, blocks - 1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, last=True))
        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x_1 = self.stage_1(x)
        x_2 = self.stage_2(x_1)
        x_3 = self.stage_3(x_2)
        pooled = self.avgpool(x_3)
        features = pooled.view(pooled.size(0), -1)
        return {'fmaps': [x_1, x_2, x_3], 'features': features}

    @property
    def last_conv(self):
        return self.stage_3[-1].conv_b


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class conv_block(nn.Module):

    def __init__(self, in_planes, planes, mode, stride=1):
        super(conv_block, self).__init__()
        self.conv = conv3x3(in_planes, planes, stride)
        self.mode = mode
        if mode == 'parallel_adapters':
            self.adapter = conv1x1(in_planes, planes, stride)

    def re_init_conv(self):
        nn.init.kaiming_normal_(self.adapter.weight, mode='fan_out', nonlinearity='relu')
        return

    def forward(self, x):
        y = self.conv(x)
        if self.mode == 'parallel_adapters':
            y = y + self.adapter(x)
        return y


class ConvNet2(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.out_dim = 64
        self.avgpool = nn.AvgPool2d(8)
        self.encoder = nn.Sequential(conv_block(x_dim, hid_dim), conv_block(hid_dim, z_dim))

    def forward(self, x):
        x = self.encoder(x)
        x = self.avgpool(x)
        features = x.view(x.shape[0], -1)
        return {'features': features}


class GeneralizedConvNet2(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(conv_block(x_dim, hid_dim))

    def forward(self, x):
        base_features = self.encoder(x)
        return base_features


class SpecializedConvNet2(nn.Module):

    def __init__(self, hid_dim=64, z_dim=64):
        super().__init__()
        self.feature_dim = 64
        self.avgpool = nn.AvgPool2d(8)
        self.AdaptiveBlock = conv_block(hid_dim, z_dim)

    def forward(self, x):
        base_features = self.AdaptiveBlock(x)
        pooled = self.avgpool(base_features)
        features = pooled.view(pooled.size(0), -1)
        return features


def first_block(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(out_channels), nn.ReLU(), nn.MaxPool2d(2))


class ConvNet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=128, z_dim=512):
        super().__init__()
        self.block1 = first_block(x_dim, hid_dim)
        self.block2 = conv_block(hid_dim, hid_dim)
        self.block3 = conv_block(hid_dim, hid_dim)
        self.block4 = conv_block(hid_dim, z_dim)
        self.avgpool = nn.AvgPool2d(7)
        self.out_dim = 512

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        features = x.view(x.shape[0], -1)
        return {'features': features}


class GeneralizedConvNet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=128, z_dim=512):
        super().__init__()
        self.block1 = first_block(x_dim, hid_dim)
        self.block2 = conv_block(hid_dim, hid_dim)
        self.block3 = conv_block(hid_dim, hid_dim)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class SpecializedConvNet(nn.Module):

    def __init__(self, hid_dim=128, z_dim=512):
        super().__init__()
        self.block4 = conv_block(hid_dim, z_dim)
        self.avgpool = nn.AvgPool2d(7)
        self.feature_dim = 512

    def forward(self, x):
        x = self.block4(x)
        x = self.avgpool(x)
        features = x.view(x.shape[0], -1)
        return features


class SimpleLinear(nn.Module):
    """
    Reference:
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    """

    def __init__(self, in_features, out_features, bias=True):
        super(SimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        nn.init.constant_(self.bias, 0)

    def forward(self, input):
        return {'logits': F.linear(input, self.weight, self.bias)}


def reduce_proxies(out, nb_proxy):
    if nb_proxy == 1:
        return out
    bs = out.shape[0]
    nb_classes = out.shape[1] / nb_proxy
    assert nb_classes.is_integer(), 'Shape error'
    nb_classes = int(nb_classes)
    simi_per_class = out.view(bs, nb_classes, nb_proxy)
    attentions = F.softmax(simi_per_class, dim=-1)
    return (attentions * simi_per_class).sum(-1)


class CosineLinear(nn.Module):

    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.to_reduce:
            out = reduce_proxies(out, self.nb_proxy)
        if self.sigma is not None:
            out = self.sigma * out
        return {'logits': out}


class SplitCosineLinear(nn.Module):

    def __init__(self, in_features, out_features1, out_features2, nb_proxy=1, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = (out_features1 + out_features2) * nb_proxy
        self.nb_proxy = nb_proxy
        self.fc1 = CosineLinear(in_features, out_features1, nb_proxy, False, False)
        self.fc2 = CosineLinear(in_features, out_features2, nb_proxy, False, False)
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out = torch.cat((out1['logits'], out2['logits']), dim=1)
        out = reduce_proxies(out, self.nb_proxy)
        if self.sigma is not None:
            out = self.sigma * out
        return {'old_scores': reduce_proxies(out1['logits'], self.nb_proxy), 'new_scores': reduce_proxies(out2['logits'], self.nb_proxy), 'logits': out}


class AnalyticLinear(torch.nn.Linear, metaclass=ABCMeta):
    """
    Abstract linear module for the analytic continual learning [1-3].

    This implementation refers to the official implementation https://github.com/ZHUANGHP/Analytic-continual-learning.

    References:
    [1] Zhuang, Huiping, et al.
        "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection."
        Advances in Neural Information Processing Systems 35 (2022): 11602-11614.
    [2] Zhuang, Huiping, et al.
        "GKEAL: Gaussian Kernel Embedded Analytic Learning for Few-Shot Class Incremental Task."
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
    [3] Zhuang, Huiping, et al.
        "DS-AL: A Dual-Stream Analytic Learning for Exemplar-Free Class-Incremental Learning."
        Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 15. 2024.
    """

    def __init__(self, in_features: 'int', gamma: 'float'=0.1, bias: 'bool'=False, device: 'Optional[Union[torch.device, str, int]]'=None, dtype=torch.double) ->None:
        super(torch.nn.Linear, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.gamma: 'float' = gamma
        self.bias: 'bool' = bias
        self.dtype = dtype
        if bias:
            in_features += 1
        weight = torch.zeros((in_features, 0), **factory_kwargs)
        self.register_buffer('weight', weight)

    @torch.no_grad()
    def forward(self, X: 'torch.Tensor') ->Dict[str, torch.Tensor]:
        X = X
        if self.bias:
            X = torch.cat((X, torch.ones(X.shape[0], 1)), dim=-1)
        return {'logits': X @ self.weight}

    @property
    def in_features(self) ->int:
        if self.bias:
            return self.weight.shape[0] - 1
        return self.weight.shape[0]

    @property
    def out_features(self) ->int:
        return self.weight.shape[1]

    def reset_parameters(self) ->None:
        self.weight = torch.zeros((self.weight.shape[0], 0))

    @abstractmethod
    def fit(self, X: 'torch.Tensor', Y: 'torch.Tensor') ->None:
        raise NotImplementedError()

    def after_task(self) ->None:
        assert torch.isfinite(self.weight).all(), 'Pay attention to the numerical stability! A possible solution is to increase the value of gamma. Setting self.dtype=torch.double also helps.'


class RecursiveLinear(AnalyticLinear):
    """
    Recursive analytic linear (ridge regression) modules for the analytic continual learning [1-3].

    This implementation refers to the official implementation https://github.com/ZHUANGHP/Analytic-continual-learning.

    References:
    [1] Zhuang, Huiping, et al.
        "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection."
        Advances in Neural Information Processing Systems 35 (2022): 11602-11614.
    [2] Zhuang, Huiping, et al.
        "GKEAL: Gaussian Kernel Embedded Analytic Learning for Few-Shot Class Incremental Task."
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
    [3] Zhuang, Huiping, et al.
        "DS-AL: A Dual-Stream Analytic Learning for Exemplar-Free Class-Incremental Learning."
        Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 15. 2024.
    """

    def __init__(self, in_features: 'int', gamma: 'float'=0.1, bias: 'bool'=False, device: 'Optional[Union[torch.device, str, int]]'=None, dtype=torch.double) ->None:
        super().__init__(in_features, gamma, bias, device, dtype)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.R: 'torch.Tensor'
        R = torch.eye(self.weight.shape[0], **factory_kwargs) / self.gamma
        self.register_buffer('R', R)

    def update_fc(self, nb_classes: 'int') ->None:
        increment_size = nb_classes - self.out_features
        assert increment_size >= 0, 'The number of classes should be increasing.'
        tail = torch.zeros((self.weight.shape[0], increment_size))
        self.weight = torch.cat((self.weight, tail), dim=1)

    @torch.no_grad()
    def fit(self, X: 'torch.Tensor', Y: 'torch.Tensor') ->None:
        """The core code of the ACIL [1].
        This implementation, which is different but equivalent to the equations shown in the paper,
        which supports mini-batch learning.
        """
        X, Y = X, Y
        if self.bias:
            X = torch.cat((X, torch.ones(X.shape[0], 1)), dim=-1)
        K = torch.inverse(torch.eye(X.shape[0]) + X @ self.R @ X.T)
        self.R -= self.R @ X.T @ K @ X @ self.R
        self.weight += self.R @ X.T @ (Y - X @ self.weight)


class GeneralizedResNet_cifar(nn.Module):

    def __init__(self, block, depth, channels=3):
        super(GeneralizedResNet_cifar, self).__init__()
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6
        self.conv_1_3x3 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)
        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.out_dim = 64 * block.expansion
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x_1 = self.stage_1(x)
        x_2 = self.stage_2(x_1)
        return x_2


class SpecializedResNet_cifar(nn.Module):

    def __init__(self, block, depth, inplanes=32, feature_dim=64):
        super(SpecializedResNet_cifar, self).__init__()
        self.inplanes = inplanes
        self.feature_dim = feature_dim
        layer_blocks = (depth - 2) // 6
        self.final_stage = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, base_feature_map):
        final_feature_map = self.final_stage(base_feature_map)
        pooled = self.avgpool(final_feature_map)
        features = pooled.view(pooled.size(0), -1)
        return features


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, last=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.last = last

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        if not self.last:
            out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, last=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.last = last

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        if not self.last:
            out = self.relu(out)
        return out


class GeneralizedResNet_imagenet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(GeneralizedResNet_imagenet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.out_dim = 512 * block.expansion
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        return x_3

    def forward(self, x):
        return self._forward_impl(x)


class SpecializedResNet_imagenet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(SpecializedResNet_imagenet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.feature_dim = 512 * block.expansion
        self.inplanes = 256 * block.expansion
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 512 * block.expansion
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x_4 = self.layer4(x)
        pooled = self.avgpool(x_4)
        features = torch.flatten(pooled, 1)
        return features


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, args=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        assert args is not None, 'you should pass args to resnet'
        if 'cifar' in args['dataset']:
            self.conv1 = nn.Sequential(nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(self.inplanes), nn.ReLU(inplace=True))
        elif 'imagenet' in args['dataset']:
            if args['init_cls'] == args['increment']:
                self.conv1 = nn.Sequential(nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(self.inplanes), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            else:
                self.conv1 = nn.Sequential(nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(self.inplanes), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], last_phase=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 512 * block.expansion
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, last_phase=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        if last_phase:
            for _ in range(1, blocks - 1):
                layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer, last=True))
        else:
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        pooled = self.avgpool(x_4)
        features = torch.flatten(pooled, 1)
        return {'fmaps': [x_1, x_2, x_3, x_4], 'features': features}

    def forward(self, x):
        return self._forward_impl(x)

    @property
    def last_conv(self):
        if hasattr(self.layer4[-1], 'conv3'):
            return self.layer4[-1].conv3
        else:
            return self.layer4[-1].conv2


class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class _Extract(torch.nn.Module):

    def __init__(self, name: 'str', *args, **kwargs) ->None:
        super().__init__(*args, **kwargs)
        self.name = name

    def forward(self, X: 'Dict[str, Any]') ->torch.Tensor:
        return X[self.name]


model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth', 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth', 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth', 'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth', 'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth', 'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth', 'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth'}


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    """ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet18_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet18_rep(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet32():
    """Constructs a ResNet-32 model for CIFAR-10."""
    model = CifarResNet(ResNetBasicblock, 32)
    return model


def resnet34(pretrained=False, progress=True, **kwargs):
    """ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet34_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet34'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    """ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def get_convnet(args, pretrained=False):
    name = args['convnet_type'].lower()
    if name == 'resnet32':
        return resnet32()
    elif name == 'resnet18':
        return resnet18(pretrained=pretrained, args=args)
    elif name == 'resnet34':
        return resnet34(pretrained=pretrained, args=args)
    elif name == 'resnet50':
        return resnet50(pretrained=pretrained, args=args)
    elif name == 'cosine_resnet18':
        return cosine_resnet18(pretrained=pretrained, args=args)
    elif name == 'cosine_resnet32':
        return cosine_resnet32()
    elif name == 'cosine_resnet34':
        return cosine_resnet34(pretrained=pretrained, args=args)
    elif name == 'cosine_resnet50':
        return cosine_resnet50(pretrained=pretrained, args=args)
    elif name == 'resnet18_rep':
        return resnet18_rep(pretrained=pretrained, args=args)
    elif name == 'resnet18_cbam':
        return resnet18_cbam(pretrained=pretrained, args=args)
    elif name == 'resnet34_cbam':
        return resnet34_cbam(pretrained=pretrained, args=args)
    elif name == 'resnet50_cbam':
        return resnet50_cbam(pretrained=pretrained, args=args)
    elif name == 'memo_resnet18':
        _basenet, _adaptive_net = get_memo_resnet18()
        return _basenet, _adaptive_net
    elif name == 'memo_resnet32':
        _basenet, _adaptive_net = get_memo_resnet32()
        return _basenet, _adaptive_net
    else:
        raise NotImplementedError('Unknown type {}'.format(name))


class BaseNet(nn.Module):

    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()
        self.convnet = get_convnet(args, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)['features']

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x['features'])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)
        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def load_checkpoint(self, args):
        if args['init_cls'] == 50:
            pkl_name = '{}_{}_{}_B{}_Inc{}'.format(args['dataset'], args['seed'], args['convnet_type'], 0, args['init_cls'])
            checkpoint_name = f'checkpoints/finetune_{pkl_name}_0.pkl'
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        self.convnet.load_state_dict(model_infos['convnet'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc


class IncrementalNet(BaseNet):

    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        if hasattr(self, 'gradcam') and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias
        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        None
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x['features'])
        out.update(x)
        if hasattr(self, 'gradcam') and self.gradcam:
            out['gradcam_gradients'] = self._gradcam_gradients
            out['gradcam_activations'] = self._gradcam_activations
        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None
        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(backward_hook)
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(forward_hook)


class IL2ANet(IncrementalNet):

    def update_fc(self, num_old, num_total, num_aux):
        fc = self.generate_fc(self.feature_dim, num_total + num_aux)
        if self.fc is not None:
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:num_old] = weight[:num_old]
            fc.bias.data[:num_old] = bias[:num_old]
        del self.fc
        self.fc = fc


class CosineIncrementalNet(BaseNet):

    def __init__(self, args, pretrained, nb_proxy=1):
        super().__init__(args, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num == 1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            fc = SplitCosineLinear(in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy)
        return fc


class BiasLayer_BIC(nn.Module):

    def __init__(self):
        super(BiasLayer_BIC, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, low_range, high_range):
        ret_x = x.clone()
        ret_x[:, low_range:high_range] = self.alpha * x[:, low_range:high_range] + self.beta
        return ret_x

    def get_params(self):
        return self.alpha.item(), self.beta.item()


class IncrementalNetWithBias(BaseNet):

    def __init__(self, args, pretrained, bias_correction=False):
        super().__init__(args, pretrained)
        self.bias_correction = bias_correction
        self.bias_layers = nn.ModuleList([])
        self.task_sizes = []

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x['features'])
        if self.bias_correction:
            logits = out['logits']
            for i, layer in enumerate(self.bias_layers):
                logits = layer(logits, sum(self.task_sizes[:i]), sum(self.task_sizes[:i + 1]))
            out['logits'] = logits
        out.update(x)
        return out

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias
        del self.fc
        self.fc = fc
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.bias_layers.append(BiasLayer_BIC())

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def get_bias_params(self):
        params = []
        for layer in self.bias_layers:
            params.append(layer.get_params())
        return params

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class DERNet(nn.Module):

    def __init__(self, args, pretrained):
        super(DERNet, self).__init__()
        self.convnet_type = args['convnet_type']
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []
        self.args = args

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)['features'] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)['features'] for convnet in self.convnets]
        features = torch.cat(features, 1)
        out = self.fc(features)
        aux_logits = self.aux_fc(features[:, -self.out_dim:])['logits']
        out.update({'aux_logits': aux_logits, 'features': features})
        return out
        """
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        """

    def update_fc(self, nb_classes):
        if len(self.convnets) == 0:
            self.convnets.append(get_convnet(self.args))
        else:
            self.convnets.append(get_convnet(self.args))
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())
        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, :self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias
        del self.fc
        self.fc = fc
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        None
        self.fc.weight.data[-increment:, :] *= gamma

    def load_checkpoint(self, args):
        checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.convnets) == 1
        self.convnets[0].load_state_dict(model_infos['convnet'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc


class SimpleCosineIncrementalNet(BaseNet):

    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def regenerate_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        del self.fc
        self.fc = fc
        return fc


class MultiBranchCosineIncrementalNet(BaseNet):

    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        None
        self.convnet = torch.nn.Identity()
        for param in self.convnet.parameters():
            param.requires_grad = False
        self.convnets = nn.ModuleList()
        self.args = args

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self._feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self._feature_dim)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        features = [convnet(x)['features'] for convnet in self.convnets]
        features = torch.cat(features, 1)
        out = self.fc(features)
        out.update({'features': features})
        return out

    def construct_dual_branch_network(self, trained_model, tuned_model, cls_num):
        self.convnets.append(trained_model.convnet)
        self.convnets.append(tuned_model.convnet)
        self._feature_dim = self.convnets[0].out_dim * len(self.convnets)
        self.fc = self.generate_fc(self._feature_dim, cls_num)


class FOSTERNet(nn.Module):

    def __init__(self, args, pretrained):
        super(FOSTERNet, self).__init__()
        self.convnet_type = args['convnet_type']
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.fe_fc = None
        self.task_sizes = []
        self.oldfc = None
        self.args = args

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)['features'] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)['features'] for convnet in self.convnets]
        features = torch.cat(features, 1)
        out = self.fc(features)
        fe_logits = self.fe_fc(features[:, -self.out_dim:])['logits']
        out.update({'fe_logits': fe_logits, 'features': features})
        if self.oldfc is not None:
            old_logits = self.oldfc(features[:, :-self.out_dim])['logits']
            out.update({'old_logits': old_logits})
        out.update({'eval_logits': out['logits']})
        return out

    def update_fc(self, nb_classes):
        self.convnets.append(get_convnet(self.args))
        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, :self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())
        self.oldfc = self.fc
        self.fc = fc
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.fe_fc = self.generate_fc(self.out_dim, nb_classes)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, old, increment, value):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew * value ** (old / increment)
        logging.info('align weights, gamma = {} '.format(gamma))
        self.fc.weight.data[-increment:, :] *= gamma

    def load_checkpoint(self, args):
        if args['init_cls'] == 50:
            pkl_name = '{}_{}_{}_B{}_Inc{}'.format(args['dataset'], args['seed'], args['convnet_type'], 0, args['init_cls'])
            checkpoint_name = f'checkpoints/finetune_{pkl_name}_0.pkl'
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.convnets) == 1
        self.convnets[0].load_state_dict(model_infos['convnet'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc


class BiasLayer(nn.Module):

    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, bias=True):
        ret_x = x.clone()
        ret_x = (self.alpha + 1) * x
        if bias:
            ret_x = ret_x + self.beta
        return ret_x

    def get_params(self):
        return self.alpha.item(), self.beta.item()


class BEEFISONet(nn.Module):

    def __init__(self, args, pretrained):
        super(BEEFISONet, self).__init__()
        self.convnet_type = args['convnet_type']
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.old_fc = None
        self.new_fc = None
        self.task_sizes = []
        self.forward_prototypes = None
        self.backward_prototypes = None
        self.args = args
        self.biases = nn.ModuleList()

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)['features'] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)['features'] for convnet in self.convnets]
        features = torch.cat(features, 1)
        if self.old_fc is None:
            fc = self.new_fc
            out = fc(features)
        else:
            """
            merge the weights
            """
            new_task_size = self.task_sizes[-1]
            fc_weight = torch.cat([self.old_fc.weight, torch.zeros((new_task_size, self.feature_dim - self.out_dim))], dim=0)
            new_fc_weight = self.new_fc.weight
            new_fc_bias = self.new_fc.bias
            for i in range(len(self.task_sizes) - 2, -1, -1):
                new_fc_weight = torch.cat([*[self.biases[i](self.backward_prototypes.weight[i].unsqueeze(0), bias=False) for _ in range(self.task_sizes[i])], new_fc_weight], dim=0)
                new_fc_bias = torch.cat([*[self.biases[i](self.backward_prototypes.bias[i].unsqueeze(0), bias=True) for _ in range(self.task_sizes[i])], new_fc_bias])
            fc_weight = torch.cat([fc_weight, new_fc_weight], dim=1)
            fc_bias = torch.cat([self.old_fc.bias, torch.zeros(new_task_size)])
            fc_bias = +new_fc_bias
            logits = features @ fc_weight.permute(1, 0) + fc_bias
            out = {'logits': logits}
            new_fc_weight = self.new_fc.weight
            new_fc_bias = self.new_fc.bias
            for i in range(len(self.task_sizes) - 2, -1, -1):
                new_fc_weight = torch.cat([self.backward_prototypes.weight[i].unsqueeze(0), new_fc_weight], dim=0)
                new_fc_bias = torch.cat([self.backward_prototypes.bias[i].unsqueeze(0), new_fc_bias])
            out['train_logits'] = features[:, -self.out_dim:] @ new_fc_weight.permute(1, 0) + new_fc_bias
        out.update({'eval_logits': out['logits'], 'energy_logits': self.forward_prototypes(features[:, -self.out_dim:])['logits']})
        return out

    def update_fc_before(self, nb_classes):
        new_task_size = nb_classes - sum(self.task_sizes)
        self.biases = nn.ModuleList([BiasLayer() for i in range(len(self.task_sizes))])
        self.convnets.append(get_convnet(self.args))
        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        if self.new_fc is not None:
            self.fe_fc = self.generate_fc(self.out_dim, nb_classes)
            self.backward_prototypes = self.generate_fc(self.out_dim, len(self.task_sizes))
            self.convnets[-1].load_state_dict(self.convnets[0].state_dict())
        self.forward_prototypes = self.generate_fc(self.out_dim, nb_classes)
        self.new_fc = self.generate_fc(self.out_dim, new_task_size)
        self.task_sizes.append(new_task_size)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def update_fc_after(self):
        if self.old_fc is not None:
            old_fc = self.generate_fc(self.feature_dim, sum(self.task_sizes))
            new_task_size = self.task_sizes[-1]
            old_fc.weight.data = torch.cat([self.old_fc.weight.data, torch.zeros((new_task_size, self.feature_dim - self.out_dim))], dim=0)
            new_fc_weight = self.new_fc.weight.data
            new_fc_bias = self.new_fc.bias.data
            for i in range(len(self.task_sizes) - 2, -1, -1):
                new_fc_weight = torch.cat([*[self.biases[i](self.backward_prototypes.weight.data[i].unsqueeze(0), bias=False) for _ in range(self.task_sizes[i])], new_fc_weight], dim=0)
                new_fc_bias = torch.cat([*[self.biases[i](self.backward_prototypes.bias.data[i].unsqueeze(0), bias=True) for _ in range(self.task_sizes[i])], new_fc_bias])
            old_fc.weight.data = torch.cat([old_fc.weight.data, new_fc_weight], dim=1)
            old_fc.bias.data = torch.cat([self.old_fc.bias.data, torch.zeros(new_task_size)])
            old_fc.bias.data += new_fc_bias
            self.old_fc = old_fc
        else:
            self.old_fc = self.new_fc

    def copy(self):
        return copy.deepcopy(self)

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, old, increment, value):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew * value ** (old / increment)
        logging.info('align weights, gamma = {} '.format(gamma))
        self.fc.weight.data[-increment:, :] *= gamma


class AdaptiveNet(nn.Module):

    def __init__(self, args, pretrained):
        super(AdaptiveNet, self).__init__()
        self.convnet_type = args['convnet_type']
        self.TaskAgnosticExtractor, _ = get_convnet(args, pretrained)
        self.TaskAgnosticExtractor.train()
        self.AdaptiveExtractors = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []
        self.args = args

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.AdaptiveExtractors)

    def extract_vector(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        out = self.fc(features)
        aux_logits = self.aux_fc(features[:, -self.out_dim:])['logits']
        out.update({'aux_logits': aux_logits, 'features': features})
        out.update({'base_features': base_feature_map})
        return out
        """
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        """

    def update_fc(self, nb_classes):
        _, _new_extractor = get_convnet(self.args)
        if len(self.AdaptiveExtractors) == 0:
            self.AdaptiveExtractors.append(_new_extractor)
        else:
            self.AdaptiveExtractors.append(_new_extractor)
            self.AdaptiveExtractors[-1].load_state_dict(self.AdaptiveExtractors[-2].state_dict())
        if self.out_dim is None:
            logging.info(self.AdaptiveExtractors[-1])
            self.out_dim = self.AdaptiveExtractors[-1].feature_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, :self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias
        del self.fc
        self.fc = fc
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        None
        self.fc.weight.data[-increment:, :] *= gamma

    def load_checkpoint(self, args):
        if args['init_cls'] == 50:
            pkl_name = '{}_{}_{}_B{}_Inc{}'.format(args['dataset'], args['seed'], args['convnet_type'], 0, args['init_cls'])
            checkpoint_name = f'checkpoints/finetune_{pkl_name}_0.pkl'
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        checkpoint_name = checkpoint_name.replace('memo_', '')
        model_infos = torch.load(checkpoint_name)
        model_dict = model_infos['convnet']
        assert len(self.AdaptiveExtractors) == 1
        base_state_dict = self.TaskAgnosticExtractor.state_dict()
        adap_state_dict = self.AdaptiveExtractors[0].state_dict()
        pretrained_base_dict = {k: v for k, v in model_dict.items() if k in base_state_dict}
        pretrained_adap_dict = {k: v for k, v in model_dict.items() if k in adap_state_dict}
        base_state_dict.update(pretrained_base_dict)
        adap_state_dict.update(pretrained_adap_dict)
        self.TaskAgnosticExtractor.load_state_dict(base_state_dict)
        self.AdaptiveExtractors[0].load_state_dict(adap_state_dict)
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc


class ACILNet(BaseNet):
    """
    Network structure of the ACIL [1].

    This implementation refers to the official implementation https://github.com/ZHUANGHP/Analytic-continual-learning.

    References:
    [1] Zhuang, Huiping, et al.
        "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection."
        Advances in Neural Information Processing Systems 35 (2022): 11602-11614.
    """

    def __init__(self, args: 'Dict[str, Any]', buffer_size: 'int'=8192, gamma: 'float'=0.1, pretrained: 'bool'=False, device=None, dtype=torch.double) ->None:
        super().__init__(args, pretrained)
        assert isinstance(self.convnet, torch.nn.Module), 'The backbone network `convnet` must be a `torch.nn.Module`.'
        self.convnet: 'torch.nn.Module' = self.convnet.to(device, non_blocking=True)
        self.args = args
        self.buffer_size: 'int' = buffer_size
        self.gamma: 'float' = gamma
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def forward(self, X: 'torch.Tensor') ->Dict[str, torch.Tensor]:
        X = self.convnet(X)['features']
        X = self.buffer(X)
        X = self.fc(X)['logits']
        return {'logits': X}

    def update_fc(self, nb_classes: 'int') ->None:
        self.fc.update_fc(nb_classes)

    def generate_fc(self, *_) ->None:
        self.fc = RecursiveLinear(self.buffer_size, self.gamma, bias=False, device=self.device, dtype=self.dtype)

    def generate_buffer(self) ->None:
        self.buffer = RandomBuffer(self.feature_dim, self.buffer_size, device=self.device, dtype=self.dtype)

    def after_task(self) ->None:
        self.fc.after_task()

    @torch.no_grad()
    def fit(self, X: 'torch.Tensor', y: 'torch.Tensor') ->None:
        X = self.convnet(X)['features']
        X = self.buffer(X)
        Y: 'torch.Tensor' = torch.nn.functional.one_hot(y, self.fc.out_features)
        self.fc.fit(X, Y)


class DSALNet(ACILNet):
    """
    Network structure of the DS-AL [1].

    This implementation refers to the official implementation https://github.com/ZHUANGHP/Analytic-continual-learning.

    References:
    [1] Zhuang, Huiping, et al.
        "DS-AL: A Dual-Stream Analytic Learning for Exemplar-Free Class-Incremental Learning."
        Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 15. 2024.
    """

    def __init__(self, args: 'Dict[str, Any]', buffer_size: 'int'=8192, gamma_main: 'float'=0.001, gamma_comp: 'float'=0.001, C: 'float'=1, activation_main: 'activation_t'=torch.relu, activation_comp: 'activation_t'=torch.tanh, pretrained: 'bool'=False, device=None, dtype=torch.double) ->None:
        self.C = C
        self.gamma_comp = gamma_comp
        self.activation_main = activation_main
        self.activation_comp = activation_comp
        super().__init__(args, buffer_size, gamma_main, pretrained, device, dtype)

    @torch.no_grad()
    def forward(self, X: 'torch.Tensor') ->Dict[str, torch.Tensor]:
        X = self.buffer(self.convnet(X)['features'])
        X_main = self.fc(self.activation_main(X))['logits']
        X_comp = self.fc_comp(self.activation_comp(X))['logits']
        return {'logits': X_main + self.C * X_comp}

    @torch.no_grad()
    def fit(self, X: 'torch.Tensor', y: 'torch.Tensor') ->None:
        num_classes = max(self.fc.out_features, int(y.max().item()) + 1)
        Y_main = torch.nn.functional.one_hot(y, num_classes=num_classes)
        X = self.buffer(self.convnet(X)['features'])
        X_main = self.activation_main(X)
        self.fc.fit(X_main, Y_main)
        self.fc.after_task()
        Y_comp = Y_main - self.fc(X_main)['logits']
        Y_comp[:, :-self.increment_size] = 0
        X_comp = self.activation_comp(X)
        self.fc_comp.fit(X_comp, Y_comp)

    @torch.no_grad()
    def after_task(self) ->None:
        self.fc.after_task()
        self.fc_comp.after_task()

    def generate_buffer(self) ->None:
        self.buffer = RandomBuffer(self.feature_dim, self.buffer_size, activation=None, device=self.device, dtype=self.dtype)

    def generate_fc(self, *_) ->None:
        self.fc = RecursiveLinear(self.buffer_size, self.gamma, bias=False, device=self.device, dtype=self.dtype)
        self.fc_comp = RecursiveLinear(self.buffer_size, self.gamma_comp, bias=False, device=self.device, dtype=self.dtype)

    def update_fc(self, nb_classes) ->None:
        self.increment_size = nb_classes - self.fc.out_features
        self.fc.update_fc(nb_classes)
        self.fc_comp.update_fc(nb_classes)


class PolicyNet(torch.nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound


class RMMPolicyNet(torch.nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(RMMPolicyNet, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, action_dim))
        self.fc2 = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, action_dim))

    def forward(self, x):
        a1 = torch.sigmoid(self.fc1(x))
        x = torch.cat([x, a1], dim=1)
        a2 = torch.tanh(self.fc2(x))
        return torch.cat([a1, a2], dim=1)


class QValueNet(torch.nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        return self.fc2(x)


class TwoLayerFC(torch.nn.Module):

    def __init__(self, num_in, num_out, hidden_dim, activation=F.relu, out_fn=lambda x: x):
        super().__init__()
        self.fc1 = nn.Linear(num_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_out)
        self.activation = activation
        self.out_fn = out_fn

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.out_fn(self.fc3(x))
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BiasLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CosineLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DownsampleA,
     lambda: ([], {'nIn': 4, 'nOut': 4, 'stride': 2}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DownsampleB,
     lambda: ([], {'nIn': 4, 'nOut': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DownsampleC,
     lambda: ([], {'nIn': 1, 'nOut': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 1, 64, 64])], {})),
    (DownsampleD,
     lambda: ([], {'nIn': 4, 'nOut': 4, 'stride': 2}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PolicyNet,
     lambda: ([], {'state_dim': 4, 'hidden_dim': 4, 'action_dim': 4, 'action_bound': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (QValueNet,
     lambda: ([], {'state_dim': 4, 'hidden_dim': 4, 'action_dim': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (RMMPolicyNet,
     lambda: ([], {'state_dim': 4, 'hidden_dim': 4, 'action_dim': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (RandomBuffer,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResNetBasicblock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SimpleLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SpatialAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SplitCosineLinear,
     lambda: ([], {'in_features': 4, 'out_features1': 4, 'out_features2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TwoLayerFC,
     lambda: ([], {'num_in': 4, 'num_out': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (_Extract,
     lambda: ([], {'name': 4}),
     lambda: ([torch.rand([5, 4, 4, 4])], {})),
    (conv_block,
     lambda: ([], {'in_planes': 4, 'planes': 4, 'mode': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

