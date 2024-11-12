
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


import numpy as np


from functools import partial


from torch import nn


import math


import time


import random


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim as optim


import torch.utils.data as data


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import torchvision.models as models


from torch.optim.lr_scheduler import _LRScheduler


import torch.nn.functional as F


from torch.nn import init


import matplotlib.pyplot as plt


import torch.nn.init as init


from torch.autograd import Variable


import torchvision


class MixBatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(MixBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.aux_bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.batch_type = 'clean'

    def forward(self, input):
        if self.batch_type == 'adv':
            input = self.aux_bn(input)
        elif self.batch_type == 'clean':
            input = super(MixBatchNorm2d, self).forward(input)
        else:
            assert self.batch_type == 'mix'
            batch_size = input.shape[0]
            input0 = super(MixBatchNorm2d, self).forward(input[:batch_size // 2])
            input1 = self.aux_bn(input[batch_size // 2:])
            input = torch.cat((input0, input1), 0)
        return input


def calc_mean_std(feat, eps=1e-05):
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adain(content_feat, style_feat):
    assert content_feat.size()[:2] == style_feat.size()[:2]
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class Net(nn.Module):

    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])
        self.enc_2 = nn.Sequential(*enc_layers[4:11])
        self.enc_3 = nn.Sequential(*enc_layers[11:18])
        self.enc_4 = nn.Sequential(*enc_layers[18:31])
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert input.size() == target.size()
        assert target.requires_grad is False
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert input.size() == target.size()
        assert target.requires_grad is False
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + self.mse_loss(input_std, target_std)

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = adain(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat
        g_t = self.decoder(t)
        g_t_feats = self.encode_with_intermediate(g_t)
        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s


class Bottleneck(nn.Module):
    """
    RexNeXt bottleneck type C
    """
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None, norm_layer=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
        """
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        D = int(math.floor(planes * (baseWidth / 64)))
        C = cardinality
        self.conv1 = nn.Conv2d(inplanes, D * C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(D * C)
        self.conv2 = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2 = norm_layer(D * C)
        self.conv3 = nn.Conv2d(D * C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNeXt(nn.Module):
    """
    ResNext optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, baseWidth, cardinality, layers, num_classes=1000, norm_layer=None):
        """ Constructor
        Args:
            baseWidth: baseWidth for ResNeXt.
            cardinality: number of convolution groups.
            layers: config of layers, e.g., [3, 4, 6, 3]
            num_classes: number of classes
        """
        super(ResNeXt, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        block = Bottleneck
        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.num_classes = num_classes
        self.inplanes = 64
        self.output_size = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct ResNext
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class _ModeNormalization(nn.Module):

    def __init__(self, dim, n_components, eps):
        super(_ModeNormalization, self).__init__()
        self.eps = eps
        self.dim = dim
        self.n_components = n_components
        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.phi = lambda x: x.mean(3).mean(2)


class ModeNorm(_ModeNormalization):
    """
    An implementation of mode normalization. Input samples x are allocated into individual modes (their number is controlled by n_components) by a gating network; samples belonging together are jointly normalized and then passed back to the network.
    args:
        dim:                int
        momentum:           float
        n_components:       int
        eps:                float
    """

    def __init__(self, dim, momentum, n_components, eps=1e-05):
        super(ModeNorm, self).__init__(dim, n_components, eps)
        self.momentum = momentum
        self.x_ra = torch.zeros(n_components, 1, dim, 1, 1)
        self.x2_ra = torch.zeros(n_components, 1, dim, 1, 1)
        self.W = torch.nn.Linear(dim, n_components)
        self.W.weight.data = torch.ones(n_components, dim) / n_components + 0.01 * torch.randn(n_components, dim)
        self.softmax = torch.nn.Softmax(dim=1)
        self.weighted_mean = lambda w, x, n: (w * x).mean(3, keepdim=True).mean(2, keepdim=True).sum(0, keepdim=True) / n

    def forward(self, x):
        g = self._g(x)
        n_k = torch.sum(g, dim=1).squeeze()
        if self.training:
            self._update_running_means(g.detach(), x.detach())
        x_split = torch.zeros(x.size()).cuda()
        for k in range(self.n_components):
            if self.training:
                mu_k = self.weighted_mean(g[k], x, n_k[k])
                var_k = self.weighted_mean(g[k], (x - mu_k) ** 2, n_k[k])
            else:
                mu_k, var_k = self._mu_var(k)
                mu_k = mu_k
                var_k = var_k
            x_split += g[k] * ((x - mu_k) / torch.sqrt(var_k + self.eps))
        x = self.alpha * x_split + self.beta
        return x

    def _g(self, x):
        """
        Image inputs are first flattened along their height and width dimensions by phi(x), then mode memberships are determined via a linear transformation, followed by a softmax activation. The gates are returned with size (k, n, c, 1, 1).
        args:
            x:          torch.Tensor
        returns:
            g:          torch.Tensor
        """
        g = self.softmax(self.W(self.phi(x))).transpose(0, 1)[:, :, None, None, None]
        return g

    def _mu_var(self, k):
        """
        At test time, this function is used to compute the k'th mean and variance from weighted running averages of x and x^2.
        args:
            k:              int
        returns:
            mu, var:        torch.Tensor, torch.Tensor
        """
        mu = self.x_ra[k]
        var = self.x2_ra[k] - self.x_ra[k] ** 2
        return mu, var

    def _update_running_means(self, g, x):
        """
        Updates weighted running averages. These are kept and used to compute estimators at test time.
        args:
            g:              torch.Tensor
            x:              torch.Tensor
        """
        n_k = torch.sum(g, dim=1).squeeze()
        for k in range(self.n_components):
            x_new = self.weighted_mean(g[k], x, n_k[k])
            x2_new = self.weighted_mean(g[k], x ** 2, n_k[k])
            self.x_ra = self.x_ra
            self.x2_ra = self.x2_ra
            self.x_ra[k] = self.momentum * x_new + (1 - self.momentum) * self.x_ra[k]
            self.x2_ra[k] = self.momentum * x2_new + (1 - self.momentum) * self.x2_ra[k]


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (MixBatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResNeXt,
     lambda: ([], {'baseWidth': 4, 'cardinality': 4, 'layers': [4, 4, 4, 4]}),
     lambda: ([torch.rand([4, 3, 256, 256])], {})),
]

