
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


import time


import torch


import torch.optim as optim


import torch.backends.cudnn as cudnn


from torch.utils.tensorboard import SummaryWriter


import torchvision


import warnings


import torch.utils.data.distributed


import torchvision.models as models


import numpy


import torchvision.datasets as datasets


import torch.distributed as dist


import torch.multiprocessing as mp


import torch.nn as nn


import numpy as np


import torch.nn.functional as F


from torch.autograd import Function


import math


import torchvision.models


from collections import OrderedDict


from torch.autograd import Variable


import torchvision.transforms as transforms


import torch.nn.init as init


import torch.utils.data


class DummyModule(nn.Module):

    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x):
        return x


class Quantizer(Function):

    @staticmethod
    def forward(ctx, input, nbit):
        scale = 2 ** nbit - 1
        return torch.round(input * scale) / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def quantize(input, nbit):
    return Quantizer.apply(input, nbit)


def dorefa_a(input, nbit_a):
    return quantize(torch.clamp(0.1 * input, 0, 1), nbit_a)


class ScaleSigner(Function):
    """take a real value x, output sign(x)*E(|x|)"""

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input) * torch.mean(torch.abs(input))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def scale_sign(input):
    return ScaleSigner.apply(input)


def dorefa_w(w, nbit_w):
    if nbit_w == 1:
        w = scale_sign(w)
    else:
        w = torch.tanh(w)
        w = w / (2 * torch.max(torch.abs(w))) + 0.5
        w = 2 * quantize(w, nbit_w) - 1
    return w


class QuanConv(nn.Conv2d):
    """docstring for QuanConv"""

    def __init__(self, in_channels, out_channels, kernel_size, quan_name_w='dorefa', quan_name_a='dorefa', nbit_w=1, nbit_a=1, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(QuanConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        name_w_dict = {'dorefa': dorefa_w}
        name_a_dict = {'dorefa': dorefa_a}
        self.quan_w = name_w_dict[quan_name_w]
        self.quan_a = name_a_dict[quan_name_a]

    def forward(self, input):
        if self.nbit_w < 32:
            w = self.quan_w(self.weight, self.nbit_w)
        else:
            w = self.weight
        if self.nbit_a < 32:
            x = self.quan_a(input, self.nbit_a)
        else:
            x = input
        output = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output


class Linear_Q(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, quan_name_w='dorefa', quan_name_a='dorefa', nbit_w=1, nbit_a=1):
        super(Linear_Q, self).__init__(in_features, out_features, bias)
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        name_w_dict = {'dorefa': dorefa_w}
        name_a_dict = {'dorefa': dorefa_a}
        self.quan_w = name_w_dict[quan_name_w]
        self.quan_a = name_a_dict[quan_name_a]

    def forward(self, input):
        if self.nbit_w < 32:
            w = self.quan_w(self.weight, self.nbit_w)
        else:
            w = self.weight
        if self.nbit_a < 32:
            x = self.quan_a(input, self.nbit_a)
        else:
            x = input
        output = F.linear(x, w, self.bias)
        return output


def conv3x3(in_planes, out_planes, wbit, abit, stride=1):
    """3x3 convolution with padding"""
    return Conv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, nbit_w=wbit, nbit_a=abit)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, wbit, abit, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, wbit, abit, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, wbit, abit)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def conv1x1(in_planes, out_planes, wbit, abit, stride=1):
    """1x1 convolution"""
    return Conv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, nbit_w=wbit, nbit_a=abit)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, wbit, abit, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, wbit, abit)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, wbit, abit, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * 4, wbit, abit)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def linear(in_featrues, out_features, wbit, abit):
    return Linear_Q(in_featrues, out_features, nbit_w=wbit, nbit_a=abit)


class ResNet(nn.Module):

    def __init__(self, block, layers, wbit, abit, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], wbit=wbit, abit=abit)
        self.layer2 = self._make_layer(block, 128, layers[1], wbit=wbit, abit=abit, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], wbit=wbit, abit=abit, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], wbit=wbit, abit=abit, stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = linear(512 * block.expansion, num_classes, wbit=8, abit=abit)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, wbit, abit, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, wbit=wbit, abit=abit, stride=stride), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, wbit, abit, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, wbit, abit))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (DummyModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Linear_Q,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (QuanConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

