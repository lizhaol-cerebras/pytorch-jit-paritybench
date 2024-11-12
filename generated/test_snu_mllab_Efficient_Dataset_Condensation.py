
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


import torch.nn as nn


import torch.optim as optim


import torch.nn.functional as F


from torchvision import datasets


from torchvision import transforms


from math import ceil


from torch.optim import lr_scheduler


import copy


from torchvision.utils import save_image


from typing import Any


from typing import Callable


from typing import Optional


from typing import Tuple


import torchvision.transforms as transforms


import torchvision


import time


from time import time


import math


import torchvision.datasets as datasets


import warnings


import random


import matplotlib


import matplotlib.pyplot as plt


from torch.nn.utils import spectral_norm


from torch.utils.data import Dataset


from torch.utils.data import Subset


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torchvision.models as models


class ConvNet(nn.Module):

    def __init__(self, num_classes, net_norm='instance', net_depth=3, net_width=128, channel=3, net_act='relu', net_pooling='avgpooling', im_size=(32, 32)):
        super(ConvNet, self).__init__()
        if net_act == 'sigmoid':
            self.net_act = nn.Sigmoid()
        elif net_act == 'relu':
            self.net_act = nn.ReLU()
        elif net_act == 'leakyrelu':
            self.net_act = nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s' % net_act)
        if net_pooling == 'maxpooling':
            self.net_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            self.net_pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            self.net_pooling = None
        else:
            exit('unknown net_pooling: %s' % net_pooling)
        self.depth = net_depth
        self.net_norm = net_norm
        self.layers, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_pooling, im_size)
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x, return_features=False):
        for d in range(self.depth):
            x = self.layers['conv'][d](x)
            if len(self.layers['norm']) > 0:
                x = self.layers['norm'][d](x)
            x = self.layers['act'][d](x)
            if len(self.layers['pool']) > 0:
                x = self.layers['pool'][d](x)
        out = x.view(x.shape[0], -1)
        logit = self.classifier(out)
        if return_features:
            return logit, out
        else:
            return logit

    def get_feature(self, x, idx_from, idx_to=-1, return_prob=False, return_logit=False):
        if idx_to == -1:
            idx_to = idx_from
        features = []
        for d in range(self.depth):
            x = self.layers['conv'][d](x)
            if self.net_norm:
                x = self.layers['norm'][d](x)
            x = self.layers['act'][d](x)
            if self.net_pooling:
                x = self.layers['pool'][d](x)
            features.append(x)
            if idx_to < len(features):
                return features[idx_from:idx_to + 1]
        if return_prob:
            out = x.view(x.size(0), -1)
            logit = self.classifier(out)
            prob = torch.softmax(logit, dim=-1)
            return features, prob
        elif return_logit:
            out = x.view(x.size(0), -1)
            logit = self.classifier(out)
            return features, logit
        else:
            return features[idx_from:idx_to + 1]

    def _get_normlayer(self, net_norm, shape_feat):
        if net_norm == 'batch':
            norm = nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layer':
            norm = nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instance':
            norm = nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'group':
            norm = nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            norm = None
        else:
            norm = None
            exit('unknown net_norm: %s' % net_norm)
        return norm

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_pooling, im_size):
        layers = {'conv': [], 'norm': [], 'act': [], 'pool': []}
        in_channels = channel
        if im_size[0] == 28:
            im_size = 32, 32
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers['conv'] += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers['norm'] += [self._get_normlayer(net_norm, shape_feat)]
            layers['act'] += [self.net_act]
            in_channels = net_width
            if net_pooling != 'none':
                layers['pool'] += [self.net_pooling]
                shape_feat[1] //= 2
                shape_feat[2] //= 2
        layers['conv'] = nn.ModuleList(layers['conv'])
        layers['norm'] = nn.ModuleList(layers['norm'])
        layers['act'] = nn.ModuleList(layers['act'])
        layers['pool'] = nn.ModuleList(layers['pool'])
        layers = nn.ModuleDict(layers)
        return layers, shape_feat


def conv_stride1(in_planes, out_planes, kernel_size=3, norm_type='instance'):
    """3x3 convolution with padding"""
    if norm_type in ['sn', 'none']:
        bias = True
    else:
        bias = False
    layer = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=bias)
    if norm_type == 'sn':
        return spectral_norm(layer)
    else:
        return layer


class Null(nn.Module):

    def __init__(self):
        super(Null, self).__init__()

    def forward(self, x):
        return x


def normalization(inplanes, norm_type):
    if norm_type == 'batch':
        bn = nn.BatchNorm2d(inplanes)
    elif norm_type == 'instance':
        bn = nn.GroupNorm(inplanes, inplanes)
    elif norm_type in ['sn', 'none']:
        bn = Null()
    else:
        raise AssertionError(f'Check normalization type! {norm_type}')
    return bn


class IntroBlock(nn.Module):

    def __init__(self, size, planes, norm_type, nch=3):
        super(IntroBlock, self).__init__()
        self.size = size
        if size == 'large':
            self.conv1 = conv_stride1(nch, planes, kernel_size=7, norm_type=norm_type)
            self.bn1 = normalization(planes, norm_type)
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.AvgPool2d(kernel_size=4, stride=4)
        elif size == 'mid':
            self.conv1 = conv_stride1(nch, planes, kernel_size=3, norm_type=norm_type)
            self.bn1 = normalization(planes, norm_type)
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        elif size == 'small':
            self.conv1 = conv_stride1(nch, planes, kernel_size=3, norm_type=norm_type)
            self.bn1 = normalization(planes, norm_type)
            self.relu = nn.ReLU(inplace=True)
        else:
            raise AssertionError('Check network size type!')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.size != 'small':
            x = self.pool(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, norm_type='batch', stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_stride1(inplanes, planes, kernel_size=3, norm_type=norm_type)
        self.bn1 = normalization(planes, norm_type)
        self.conv2 = conv_stride1(planes, planes, kernel_size=3, norm_type=norm_type)
        self.bn2 = normalization(planes, norm_type)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.stride != 1:
            out = F.avg_pool2d(out, kernel_size=self.stride, stride=self.stride)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, norm_type='batch', stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = normalization(planes, norm_type)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = normalization(planes, norm_type)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = normalization(planes * Bottleneck.expansion, norm_type)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.stride != 1:
            out = F.avg_pool2d(out, kernel_size=self.stride, stride=self.stride)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, dataset, depth, num_classes, norm_type='batch', size=-1, nch=3):
        super(ResNet, self).__init__()
        self.dataset = dataset
        self.norm_type = norm_type
        if self.dataset.startswith('cifar') or 0 < size and size <= 64:
            self.net_size = 'small'
        elif 64 < size and size <= 128:
            self.net_size = 'mid'
        else:
            self.net_size = 'large'
        if self.dataset.startswith('cifar'):
            self.inplanes = 32
            n = int((depth - 2) / 6)
            block = BasicBlock
            self.layer0 = IntroBlock(self.net_size, self.inplanes, norm_type, nch=nch)
            self.layer1 = self._make_layer(block, 32, n, stride=1)
            self.layer2 = self._make_layer(block, 64, n, stride=2)
            self.layer3 = self._make_layer(block, 128, n, stride=2)
            self.layer4 = self._make_layer(block, 256, n, stride=2)
            self.avgpool = nn.AvgPool2d(4)
            self.fc = nn.Linear(256 * block.expansion, num_classes)
        else:
            blocks = {(10): BasicBlock, (18): BasicBlock, (34): BasicBlock, (50): Bottleneck, (101): Bottleneck, (152): Bottleneck, (200): Bottleneck}
            layers = {(10): [1, 1, 1, 1], (18): [2, 2, 2, 2], (34): [3, 4, 6, 3], (50): [3, 4, 6, 3], (101): [3, 4, 23, 3], (152): [3, 8, 36, 3], (200): [3, 24, 36, 3]}
            assert layers[depth], 'invalid detph for ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'
            self.inplanes = 64
            self.layer0 = IntroBlock(self.net_size, self.inplanes, norm_type, nch=nch)
            self.layer1 = self._make_layer(blocks[depth], 64, layers[depth][0])
            self.layer2 = self._make_layer(blocks[depth], 128, layers[depth][1], stride=2)
            self.layer3 = self._make_layer(blocks[depth], 256, layers[depth][2], stride=2)
            self.layer4 = self._make_layer(blocks[depth], 512, layers[depth][3], stride=2)
            self.avgpool = nn.AvgPool2d(7)
            self.fc = nn.Linear(512 * blocks[depth].expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), normalization(planes * block.expansion, self.norm_type))
        layers = []
        layers.append(block(self.inplanes, planes, norm_type=self.norm_type, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_type=self.norm_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, x.shape[-1])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_feature(self, x, idx_from, idx_to=-1):
        if idx_to == -1:
            idx_to = idx_from
        features = []
        x = self.layer0(x)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from:idx_to + 1]
        x = self.layer1(x)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from:idx_to + 1]
        x = self.layer2(x)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from:idx_to + 1]
        x = self.layer3(x)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from:idx_to + 1]
        x = self.layer4(x)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from:idx_to + 1]
        x = F.avg_pool2d(x, x.shape[-1])
        x = x.view(x.size(0), -1)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from:idx_to + 1]
        x = self.fc(x)
        features.append(x)
        return features[idx_from:idx_to + 1]


class Transition(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes
        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes
        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes
        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate
        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNetAP(nn.Module):

    def __init__(self, dataset, depth, num_classes, width=1.0, norm_type='batch', size=-1, nch=3):
        super(ResNetAP, self).__init__()
        self.dataset = dataset
        self.norm_type = norm_type
        self.nch = nch
        if self.dataset.startswith('cifar') or 0 < size and size <= 64:
            self.net_size = 'small'
        elif 64 < size and size <= 128:
            self.net_size = 'mid'
        else:
            self.net_size = 'large'
        if self.dataset.startswith('cifar'):
            self.inplanes = 32
            n = int((depth - 2) / 6)
            block = BasicBlock
            self.layer0 = IntroBlock(self.net_size, self.inplanes, norm_type, nch=nch)
            self.layer1 = self._make_layer(block, 32, n, stride=1)
            self.layer2 = self._make_layer(block, 64, n, stride=2)
            self.layer3 = self._make_layer(block, 128, n, stride=2)
            self.layer4 = self._make_layer(block, 256, n, stride=2)
            self.avgpool = nn.AvgPool2d(4)
            self.fc = nn.Linear(256 * block.expansion, num_classes)
        else:
            blocks = {(10): BasicBlock, (18): BasicBlock, (34): BasicBlock, (50): Bottleneck, (101): Bottleneck, (152): Bottleneck, (200): Bottleneck}
            layers = {(10): [1, 1, 1, 1], (18): [2, 2, 2, 2], (34): [3, 4, 6, 3], (50): [3, 4, 6, 3], (101): [3, 4, 23, 3], (152): [3, 8, 36, 3], (200): [3, 24, 36, 3]}
            assert layers[depth], 'invalid detph for ResNet'
            self.inplanes = int(64 * width)
            self.layer0 = IntroBlock(self.net_size, self.inplanes, norm_type, nch=nch)
            nc = self.inplanes
            self.layer1 = self._make_layer(blocks[depth], nc, layers[depth][0])
            self.layer2 = self._make_layer(blocks[depth], nc * 2, layers[depth][1], stride=2)
            self.layer3 = self._make_layer(blocks[depth], nc * 4, layers[depth][2], stride=2)
            self.layer4 = self._make_layer(blocks[depth], nc * 8, layers[depth][3], stride=2)
            self.avgpool = nn.AvgPool2d(7)
            self.fc = nn.Linear(self.inplanes, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv_stride1(self.inplanes, planes * block.expansion, kernel_size=1, norm_type=self.norm_type), nn.AvgPool2d(kernel_size=stride, stride=stride), normalization(planes * block.expansion, self.norm_type))
        layers = []
        layers.append(block(self.inplanes, planes, norm_type=self.norm_type, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_type=self.norm_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, x.shape[-1])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_feature(self, x, idx_from, idx_to=-1):
        if idx_to == -1:
            idx_to = idx_from
        features = []
        x = self.layer0(x)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from:idx_to + 1]
        x = self.layer1(x)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from:idx_to + 1]
        x = self.layer2(x)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from:idx_to + 1]
        x = self.layer3(x)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from:idx_to + 1]
        x = self.layer4(x)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from:idx_to + 1]
        x = F.avg_pool2d(x, x.shape[-1])
        x = x.view(x.size(0), -1)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from:idx_to + 1]
        x = self.fc(x)
        features.append(x)
        return features[idx_from:idx_to + 1]


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Null,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Transition,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

