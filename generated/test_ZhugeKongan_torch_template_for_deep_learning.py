
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


import random


import numpy as np


from torch.utils.data.dataset import Dataset


import torch.nn.functional as F


from torch import nn


import torch.nn as nn


from functools import reduce


import matplotlib.pyplot as plt


from torch.autograd import Variable


import torch.utils.model_zoo as model_zoo


import torchvision


import torchvision.transforms as transforms


from torch.utils.data import TensorDataset


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


import math


from torchvision import transforms


import time


from torch.hub import load_state_dict_from_url


from torch.nn import functional as F


import re


import collections


from functools import partial


from torch.utils import model_zoo


from collections import OrderedDict


from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork


from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


from torchvision.ops import misc as misc_nn_ops


from torchvision.ops import MultiScaleRoIAlign


from torchvision.ops import boxes as box_ops


from torchvision.ops import roi_align


import warnings


from collections import namedtuple


import torch.nn.init as init


from sklearn.metrics import f1_score


from sklearn.metrics import precision_score


from sklearn.metrics import recall_score


from sklearn.metrics import fbeta_score


from sklearn.metrics import classification_report


from sklearn.metrics import hamming_loss


from sklearn.metrics import accuracy_score


from sklearn.metrics import coverage_error


from sklearn.metrics import label_ranking_loss


from sklearn.metrics import label_ranking_average_precision_score


from torch.nn import init


from torch import sqrt


from math import sqrt


from torch import flatten


from torch.nn.modules.activation import ReLU


from torch.nn.modules.batchnorm import BatchNorm2d


from torch import einsum


from torch.nn.parameter import Parameter


from torch.functional import norm


import itertools


from torchvision.utils import save_image


from torchvision import datasets


from torchvision.models import resnet18


import scipy


import torch.utils.data as data


from torchvision.utils import make_grid


import torch.autograd as autograd


from torchvision.models import vgg19


from torchvision.models.resnet import resnet18


from torchvision.models.resnet import resnet34


from torchvision.models.resnet import resnet50


from torchvision.models.resnet import resnet101


from torchvision.models.resnet import resnet152


from torch import nn as nnl


import torchvision.models as models


import types


import torch.cuda.amp as amp


import tensorflow as tf


import functools


import pandas as pd


from sklearn.model_selection import train_test_split


import torch.optim as optim


from torch import sigmoid


from torch import softmax


from torch.utils.tensorboard import SummaryWriter


from math import log10


from matplotlib import pyplot as plt


from torch.optim.lr_scheduler import _LRScheduler


def cross_entropy(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))


class CutMixCrossEntropyLoss(Module):

    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average

    def forward(self, input, target):
        if len(target.size()) == 1:
            target = torch.nn.functional.one_hot(target, num_classes=input.size(-1))
            target = target.float()
        return cross_entropy(input, target, self.size_average)


class DropBlock2D(nn.Module):
    """Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size
        self.training = True

    def forward(self, x):
        assert x.dim() == 4, 'Expected input with 4 dimensions (bsize, channels, height, width)'
        if not self.training or self.drop_prob == 0.0:
            return x
        else:
            gamma = self._compute_gamma(x)
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
            mask = mask
            block_mask = self._compute_block_mask(mask)
            out = x * block_mask[:, None, :, :]
            out = out * block_mask.numel() / block_mask.sum()
            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :], kernel_size=(self.block_size, self.block_size), stride=(1, 1), padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / self.block_size ** 2


class DropBlock3D(DropBlock2D):
    """Randomly zeroes 3D spatial blocks of the input tensor.
    An extension to the concept described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, D, H, W)`
        - Output: `(N, C, D, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock3D, self).__init__(drop_prob, block_size)

    def forward(self, x):
        assert x.dim() == 5, 'Expected input with 5 dimensions (bsize, channels, depth, height, width)'
        if not self.training or self.drop_prob == 0.0:
            return x
        else:
            gamma = self._compute_gamma(x)
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
            mask = mask
            block_mask = self._compute_block_mask(mask)
            out = x * block_mask[:, None, :, :, :]
            out = out * block_mask.numel() / block_mask.sum()
            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool3d(input=mask[:, None, :, :, :], kernel_size=(self.block_size, self.block_size, self.block_size), stride=(1, 1, 1), padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / self.block_size ** 3


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
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


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

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
        out = self.se_module(out) + residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
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
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=True), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        self.conv1_input = x.clone()
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


class LinearScheduler(nn.Module):

    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=int(nr_steps))

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]
        self.i += 1


class LabelSmoothingCrossEntropy(nn.Module):

    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)


class ShakeDropFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, training=True, p_drop=0.5, alpha_range=[-1, 1]):
        if training:
            gate = torch.FloatTensor([0]).bernoulli_(1 - p_drop)
            ctx.save_for_backward(gate)
            if gate.item() == 0:
                alpha = torch.FloatTensor(x.size(0)).uniform_(*alpha_range)
                alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x)
                alpha = Variable(torch.zeros(1))
                return alpha * x
            else:
                return x
        else:
            return (1 - p_drop) * x

    @staticmethod
    def backward(ctx, grad_output):
        gate = ctx.saved_tensors[0]
        if gate.item() == 0:
            beta = torch.FloatTensor(grad_output.size(0)).uniform_(0, 1)
            beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
            beta = Variable(beta)
            return beta * grad_output, None, None, None
        else:
            return grad_output, None, None, None


class ShakeDrop(nn.Module):

    def __init__(self, p_drop=0.5, alpha_range=[-1, 1]):
        super(ShakeDrop, self).__init__()
        self.p_drop = p_drop
        self.alpha_range = alpha_range

    def forward(self, x):
        return ShakeDropFunction.apply(x, self.training, self.p_drop, self.alpha_range)


class StoDepth_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, prob, multFlag, inplanes, planes, stride=1, downsample=None):
        super(StoDepth_BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.prob = prob
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))
        self.multFlag = multFlag

    def forward(self, x):
        identity = x.clone()
        if self.training:
            if torch.equal(self.m.sample(), torch.ones(1)):
                self.conv1.weight.requires_grad = True
                self.conv2.weight.requires_grad = True
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)
                if self.downsample is not None:
                    identity = self.downsample(x)
                out += identity
            else:
                self.conv1.weight.requires_grad = False
                self.conv2.weight.requires_grad = False
                if self.downsample is not None:
                    identity = self.downsample(x)
                out = identity
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            if self.multFlag:
                out = self.prob * out + identity
            else:
                out = out + identity
        out = self.relu(out)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class StoDepth_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, prob, multFlag, inplanes, planes, stride=1, downsample=None):
        super(StoDepth_Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.prob = prob
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))
        self.multFlag = multFlag

    def forward(self, x):
        identity = x.clone()
        if self.training:
            if torch.equal(self.m.sample(), torch.ones(1)):
                self.conv1.weight.requires_grad = True
                self.conv2.weight.requires_grad = True
                self.conv3.weight.requires_grad = True
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
            else:
                self.conv1.weight.requires_grad = False
                self.conv2.weight.requires_grad = False
                self.conv3.weight.requires_grad = False
                if self.downsample is not None:
                    identity = self.downsample(x)
                out = identity
        else:
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
            if self.multFlag:
                out = self.prob * out + identity
            else:
                out = out + identity
        out = self.relu(out)
        return out


class ResNet_StoDepth_lineardecay(nn.Module):

    def __init__(self, block, prob_0_L, multFlag, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet_StoDepth_lineardecay, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.multFlag = multFlag
        self.prob_now = prob_0_L[0]
        self.prob_delta = prob_0_L[0] - prob_0_L[1]
        self.prob_step = self.prob_delta / (sum(layers) - 1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, StoDepth_Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, StoDepth_BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.prob_now, self.multFlag, self.inplanes, planes, stride, downsample))
        self.prob_now = self.prob_now - self.prob_step
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.prob_now, self.multFlag, self.inplanes, planes))
            self.prob_now = self.prob_now - self.prob_step
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SwishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):

    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


def get_width_and_height_from_size(x):
    """Obtain height and width from x.

    Args:
        x (int, tuple or list): Data size.

    Returns:
        size: A tuple or list (H,W).
    """
    if isinstance(x, int):
        return x, x
    if isinstance(x, list) or isinstance(x, tuple):
        return x
    else:
        raise TypeError()


def calculate_output_image_size(input_image_size, stride):
    """Calculates the output image size when using Conv2dSamePadding with a stride.
       Necessary for static padding. Thanks to mannatsingh for pointing this out.

    Args:
        input_image_size (int, tuple or list): Size of input image.
        stride (int, tuple or list): Conv2d operation's stride.

    Returns:
        output_image_size: A list [H,W].
    """
    if input_image_size is None:
        return None
    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]


def drop_connect(inputs, p, training):
    """Drop connect.

    Args:
        input (tensor: BCWH): Input of this structure.
        p (float: 0.0~1.0): Probability of drop connection.
        training (bool): The running mode.

    Returns:
        output: Output after drop connection.
    """
    assert 0 <= p <= 1, 'p must be in range of [0,1]'
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


class Conv2dDynamicSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow, for a dynamic image size.
       The padding is operated in forward function by calculating dynamically.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dStaticSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w - pad_w // 2, pad_w - pad_w // 2, pad_h - pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


def get_same_padding_conv2d(image_size=None):
    """Chooses static padding if you have specified an image size, and dynamic padding otherwise.
       Static padding is necessary for ONNX exporting of models.

    Args:
        image_size (int or tuple): Size of the image.

    Returns:
        Conv2dDynamicSamePadding or Conv2dStaticSamePadding.
    """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.

    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].

    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = self._block_args.se_ratio is not None and 0 < self._block_args.se_ratio <= 1
        self.id_skip = block_args.id_skip
        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(in_channels=oup, out_channels=oup, groups=oup, kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x
        x = self._project_conv(x)
        x = self._bn2(x)
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


VALID_MODELS = 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7', 'efficientnet-b8', 'efficientnet-l2'


def efficientnet_params(model_name):
    """Map EfficientNet model name to parameter coefficients.

    Args:
        model_name (str): Model name to be queried.

    Returns:
        params_dict[model_name]: A (width,depth,res,dropout) tuple.
    """
    params_dict = {'efficientnet-b0': (1.0, 1.0, 224, 0.2), 'efficientnet-b1': (1.0, 1.1, 240, 0.2), 'efficientnet-b2': (1.1, 1.2, 260, 0.3), 'efficientnet-b3': (1.2, 1.4, 300, 0.3), 'efficientnet-b4': (1.4, 1.8, 380, 0.4), 'efficientnet-b5': (1.6, 2.2, 456, 0.4), 'efficientnet-b6': (1.8, 2.6, 528, 0.5), 'efficientnet-b7': (2.0, 3.1, 600, 0.5), 'efficientnet-b8': (2.2, 3.6, 672, 0.5), 'efficientnet-l2': (4.3, 5.3, 800, 0.5)}
    return params_dict[model_name]


BlockArgs = collections.namedtuple('BlockArgs', ['num_repeat', 'kernel_size', 'stride', 'expand_ratio', 'input_filters', 'output_filters', 'se_ratio', 'id_skip'])


class BlockDecoder(object):
    """Block Decoder for readability,
       straight from the official TensorFlow repository.
    """

    @staticmethod
    def _decode_block_string(block_string):
        """Get a block through a string notation of arguments.

        Args:
            block_string (str): A string notation of arguments.
                                Examples: 'r1_k3_s11_e1_i32_o16_se0.25_noskip'.

        Returns:
            BlockArgs: The namedtuple defined at the top of this file.
        """
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split('(\\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value
        assert 's' in options and len(options['s']) == 1 or len(options['s']) == 2 and options['s'][0] == options['s'][1]
        return BlockArgs(num_repeat=int(options['r']), kernel_size=int(options['k']), stride=[int(options['s'][0])], expand_ratio=int(options['e']), input_filters=int(options['i']), output_filters=int(options['o']), se_ratio=float(options['se']) if 'se' in options else None, id_skip='noskip' not in block_string)

    @staticmethod
    def _encode_block_string(block):
        """Encode a block to a string.

        Args:
            block (namedtuple): A BlockArgs type argument.

        Returns:
            block_string: A String form of BlockArgs.
        """
        args = ['r%d' % block.num_repeat, 'k%d' % block.kernel_size, 's%d%d' % (block.strides[0], block.strides[1]), 'e%s' % block.expand_ratio, 'i%d' % block.input_filters, 'o%d' % block.output_filters]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """Decode a list of string notations to specify blocks inside the network.

        Args:
            string_list (list[str]): A list of strings, each string is a notation of block.

        Returns:
            blocks_args: A list of BlockArgs namedtuples of block args.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """Encode a list of BlockArgs to a list of strings.

        Args:
            blocks_args (list[namedtuples]): A list of BlockArgs namedtuples of block args.

        Returns:
            block_strings: A list of strings, each string is a notation of block.
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


GlobalParams = collections.namedtuple('GlobalParams', ['width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate', 'num_classes', 'batch_norm_momentum', 'batch_norm_epsilon', 'drop_connect_rate', 'depth_divisor', 'min_depth', 'include_top'])


def efficientnet(width_coefficient=None, depth_coefficient=None, image_size=None, dropout_rate=0.2, drop_connect_rate=0.2, num_classes=1000, include_top=True):
    """Create BlockArgs and GlobalParams for efficientnet model.

    Args:
        width_coefficient (float)
        depth_coefficient (float)
        image_size (int)
        dropout_rate (float)
        drop_connect_rate (float)
        num_classes (int)

        Meaning as the name suggests.

    Returns:
        blocks_args, global_params.
    """
    blocks_args = ['r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25', 'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25', 'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25', 'r1_k3_s11_e6_i192_o320_se0.25']
    blocks_args = BlockDecoder.decode(blocks_args)
    global_params = GlobalParams(width_coefficient=width_coefficient, depth_coefficient=depth_coefficient, image_size=image_size, dropout_rate=dropout_rate, num_classes=num_classes, batch_norm_momentum=0.99, batch_norm_epsilon=0.001, drop_connect_rate=drop_connect_rate, depth_divisor=8, min_depth=None, include_top=include_top)
    return blocks_args, global_params


def get_model_params(model_name, override_params):
    """Get the block args and global params for a given model name.

    Args:
        model_name (str): Model's name.
        override_params (dict): A dict to modify global_params.

    Returns:
        blocks_args, global_params
    """
    if model_name.startswith('efficientnet'):
        w, d, s, p = efficientnet_params(model_name)
        blocks_args, global_params = efficientnet(width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)
    else:
        raise NotImplementedError('model name is not pre-defined: {}'.format(model_name))
    if override_params:
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


url_map = {'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth', 'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth', 'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth', 'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth', 'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth', 'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth', 'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth', 'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth'}


url_map_advprop = {'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b0-b64d5a18.pth', 'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b1-0f3ce85a.pth', 'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b2-6e9d97e5.pth', 'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b3-cdd7c0f4.pth', 'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b4-44fb3a87.pth', 'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b5-86493f6b.pth', 'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b6-ac80338e.pth', 'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b7-4652b6dd.pth', 'efficientnet-b8': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b8-22a8fe65.pth'}


def load_pretrained_weights(model, model_name, weights_path=None, load_fc=True, advprop=False):
    """Loads pretrained weights from weights path or download using url.

    Args:
        model (Module): The whole model of efficientnet.
        model_name (str): Model name of efficientnet.
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.
        advprop (bool): Whether to load pretrained weights
                        trained with advprop (valid when weights_path is None).
    """
    if isinstance(weights_path, str):
        state_dict = torch.load(weights_path)
    else:
        url_map_ = url_map_advprop if advprop else url_map
        state_dict = model_zoo.load_url(url_map_[model_name])
    if load_fc:
        ret = model.load_state_dict(state_dict, strict=False)
        assert not ret.missing_keys, 'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
    else:
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        ret = model.load_state_dict(state_dict, strict=False)
        assert set(ret.missing_keys) == set(['_fc.weight', '_fc.bias']), 'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
    assert not ret.unexpected_keys, 'Missing keys when loading pretrained weights: {}'.format(ret.unexpected_keys)
    None


def round_filters(filters, global_params):
    """Calculate and round number of filters based on width multiplier.
       Use width_coefficient, depth_divisor and min_depth of global_params.

    Args:
        filters (int): Filters number to be calculated.
        global_params (namedtuple): Global params of the model.

    Returns:
        new_filters: New filters number after calculating.
    """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Calculate module's repeat number of a block based on depth multiplier.
       Use depth_coefficient of global_params.

    Args:
        repeats (int): num_repeat to be calculated.
        global_params (namedtuple): Global params of the model.

    Returns:
        new repeat: New repeat number after calculating.
    """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class EfficientNet(nn.Module):
    """EfficientNet model.
       Most easily loaded with the .from_name or .from_pretrained methods.

    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.

    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)

    Example:
        
        
        import torch
        >>> from efficientnet.model import EfficientNet
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> model = EfficientNet.from_pretrained('efficientnet-b0')
        >>> model.eval()
        >>> outputs = model(inputs)
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        in_channels = 3
        out_channels = round_filters(32, self._global_params)
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:
            block_args = block_args._replace(input_filters=round_filters(block_args.input_filters, self._global_params), output_filters=round_filters(block_args.output_filters, self._global_params), num_repeat=round_repeats(block_args.num_repeat, self._global_params))
            self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
        in_channels = block_args.output_filters
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.

        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_endpoints(self, inputs):
        """Use convolution layer to extract features
        from reduction levels i in [1, 2, 3, 4, 5].

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Dictionary of last intermediate features
            with reduction levels i in [1, 2, 3, 4, 5].
            Example:
                >>> import torch
                >>> from efficientnet.model import EfficientNet
                >>> inputs = torch.rand(1, 3, 224, 224)
                >>> model = EfficientNet.from_pretrained('efficientnet-b0')
                >>> endpoints = model.extract_endpoints(inputs)
                >>> print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
                >>> print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
                >>> print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
                >>> print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
                >>> print(endpoints['reduction_5'].shape)  # torch.Size([1, 1280, 7, 7])
        """
        endpoints = dict()
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
        return endpoints

    def extract_features(self, inputs):
        """use convolution layer to extract feature .

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        x = self._swish(self._bn1(self._conv_head(x)))
        return x

    def forward(self, inputs):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of this model after processing.
        """
        x = self.extract_features(inputs)
        x = self._avg_pooling(x)
        if self._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, in_channels=3, **override_params):
        """create an efficientnet model according to name.

        Args:
            model_name (str): Name for efficientnet.
            in_channels (int): Input data's channel number.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'num_classes', 'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            An efficientnet model.
        """
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        model = cls(blocks_args, global_params)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def from_pretrained(cls, model_name, weights_path=None, advprop=False, in_channels=3, num_classes=1000, **override_params):
        """create an efficientnet model according to name.

        Args:
            model_name (str): Name for efficientnet.
            weights_path (None or str):
                str: path to pretrained weights file on the local disk.
                None: use pretrained weights downloaded from the Internet.
            advprop (bool):
                Whether to load pretrained weights
                trained with advprop (valid when weights_path is None).
            in_channels (int): Input data's channel number.
            num_classes (int):
                Number of categories for classification.
                It controls the output size for final linear layer.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            A pretrained efficientnet model.
        """
        model = cls.from_name(model_name, num_classes=num_classes, **override_params)
        load_pretrained_weights(model, model_name, weights_path=weights_path, load_fc=num_classes == 1000, advprop=advprop)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        """Get the input image size for a given efficientnet model.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            Input image size (resolution).
        """
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """Validates model name.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            bool: Is a valid name or not.
        """
        if model_name not in VALID_MODELS:
            raise ValueError('model_name should be one of: ' + ', '.join(VALID_MODELS))

    def _change_in_channels(self, in_channels):
        """Adjust model's first convolution layer to in_channels, if in_channels not equals 3.

        Args:
            in_channels (int): Input data's channel number.
        """
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError('return_layers are not present in model')
        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.feature_extraction = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=0), nn.Conv2d(in_channels=96, out_channels=192, kernel_size=5, stride=1, padding=2, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=0), nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        self.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features=256 * 6 * 6, out_features=4096), nn.Dropout(p=0.5), nn.Linear(in_features=4096, out_features=4096), nn.Linear(in_features=4096, out_features=num_classes))

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class _DenseLayer(nn.Module):

    def __init__(self, inplace, growth_rate, bn_size, drop_rate=0):
        super(_DenseLayer, self).__init__()
        self.drop_rate = drop_rate
        self.dense_layer = nn.Sequential(nn.BatchNorm2d(inplace), nn.ReLU(inplace=True), nn.Conv2d(in_channels=inplace, out_channels=bn_size * growth_rate, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(bn_size * growth_rate), nn.ReLU(inplace=True), nn.Conv2d(in_channels=bn_size * growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        y = self.dense_layer(x)
        if self.drop_rate > 0:
            y = self.dropout(y)
        return torch.cat([x, y], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class BN_Conv2d(nn.Module):
    """
    BN_CONV, default activation is ReLU
    """

    def __init__(self, in_channels: 'object', out_channels: 'object', kernel_size: 'object', stride: 'object', padding: 'object', dilation=1, groups=1, bias=False, activation=nn.ReLU(inplace=True)) ->object:
        super(BN_Conv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias), nn.BatchNorm2d(out_channels)]
        if activation is not None:
            layers.append(activation)
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class DenseBlock(nn.Module):

    def __init__(self, input_channels, num_layers, growth_rate):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.k0 = input_channels
        self.k = growth_rate
        self.layers = self.__make_layers()

    def __make_layers(self):
        layer_list = []
        for i in range(self.num_layers):
            layer_list.append(nn.Sequential(BN_Conv2d(self.k0 + i * self.k, 4 * self.k, 1, 1, 0), BN_Conv2d(4 * self.k, self.k, 3, 1, 1)))
        return layer_list

    def forward(self, x):
        feature = self.layers[0](x)
        out = torch.cat((x, feature), 1)
        for i in range(1, len(self.layers)):
            feature = self.layers[i](out)
            out = torch.cat((feature, out), 1)
        return out


class CSP_DenseBlock(nn.Module):

    def __init__(self, in_channels, num_layers, k, part_ratio=0.5):
        super(CSP_DenseBlock, self).__init__()
        self.part1_chnls = int(in_channels * part_ratio)
        self.part2_chnls = in_channels - self.part1_chnls
        self.dense = DenseBlock(self.part2_chnls, num_layers, k)

    def forward(self, x):
        part1 = x[:, :self.part1_chnls, :, :]
        part2 = x[:, self.part1_chnls:, :, :]
        part2 = self.dense(part2)
        out = torch.cat((part1, part2), 1)
        return out


class DenseNet(nn.Module):

    def __init__(self, layers: 'object', k, theta, num_classes, part_ratio=0) ->object:
        super(DenseNet, self).__init__()
        self.layers = layers
        self.k = k
        self.theta = theta
        self.Block = DenseBlock if part_ratio == 0 else CSP_DenseBlock
        self.conv = BN_Conv2d(3, 2 * k, 7, 2, 3)
        self.blocks, patches = self.__make_blocks(2 * k)
        self.fc = nn.Linear(patches, num_classes)

    def __make_transition(self, in_chls):
        out_chls = int(self.theta * in_chls)
        return nn.Sequential(BN_Conv2d(in_chls, out_chls, 1, 1, 0), nn.AvgPool2d(2)), out_chls

    def __make_blocks(self, k0):
        """
        make block-transition structures
        :param k0:
        :return:
        """
        layers_list = []
        patches = 0
        for i in range(len(self.layers)):
            layers_list.append(self.Block(k0, self.layers[i], self.k))
            patches = k0 + self.layers[i] * self.k
            if i != len(self.layers) - 1:
                transition, k0 = self.__make_transition(patches)
                layers_list.append(transition)
        return nn.Sequential(*layers_list), patches

    def forward(self, x):
        out = self.conv(x)
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.blocks(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class BackboneWithFPN(nn.Sequential):
    """
    Adds a FPN on top of a model.

    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.

    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.

    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(self, backbone, return_layers, in_channels_list, out_channels):
        body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=out_channels, extra_blocks=LastLevelMaxPool())
        super(BackboneWithFPN, self).__init__(OrderedDict([('body', body), ('fpn', fpn)]))
        self.out_channels = out_channels


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError('In training mode, targets should be passed')
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        if self.training:
            return losses
        return detections


class KeypointRCNNHeads(nn.Sequential):

    def __init__(self, in_channels, layers):
        d = []
        next_feature = in_channels
        for l in layers:
            d.append(misc_nn_ops.Conv2d(next_feature, l, 3, stride=1, padding=1))
            d.append(nn.ReLU(inplace=True))
            next_feature = l
        super(KeypointRCNNHeads, self).__init__(*d)
        for m in self.children():
            if isinstance(m, misc_nn_ops.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


class KeypointRCNNPredictor(nn.Module):

    def __init__(self, in_channels, num_keypoints):
        super(KeypointRCNNPredictor, self).__init__()
        input_features = in_channels
        deconv_kernel = 4
        self.kps_score_lowres = misc_nn_ops.ConvTranspose2d(input_features, num_keypoints, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1)
        nn.init.kaiming_normal_(self.kps_score_lowres.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.kps_score_lowres.bias, 0)
        self.up_scale = 2
        self.out_channels = num_keypoints

    def forward(self, x):
        x = self.kps_score_lowres(x)
        x = misc_nn_ops.interpolate(x, scale_factor=self.up_scale, mode='bilinear', align_corners=False)
        return x


class MaskRCNNHeads(nn.Sequential):

    def __init__(self, in_channels, layers, dilation):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d['mask_fcn{}'.format(layer_idx)] = misc_nn_ops.Conv2d(next_feature, layer_features, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
            d['relu{}'.format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features
        super(MaskRCNNHeads, self).__init__(d)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')


class MaskRCNNPredictor(nn.Sequential):

    def __init__(self, in_channels, dim_reduced, num_classes):
        super(MaskRCNNPredictor, self).__init__(OrderedDict([('conv5_mask', misc_nn_ops.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)), ('relu', nn.ReLU(inplace=True)), ('mask_fcn_logits', misc_nn_ops.Conv2d(dim_reduced, num_classes, 1, 1, 0))]))
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    """
    Computes the loss for Faster R-CNN.

    Arguments:
        class_logits (Tensor)
        box_regression (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels)
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)
    box_loss = F.smooth_l1_loss(box_regression[sampled_pos_inds_subset, labels_pos], regression_targets[sampled_pos_inds_subset], reduction='sum')
    box_loss = box_loss / labels.numel()
    return classification_loss, box_loss


def heatmaps_to_keypoints(maps, rois):
    """Extract predicted keypoint locations from heatmaps. Output has shape
    (#rois, 4, #keypoints) with the 4 rows corresponding to (x, y, logit, prob)
    for each keypoint.
    """
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    widths = rois[:, 2] - rois[:, 0]
    heights = rois[:, 3] - rois[:, 1]
    widths = widths.clamp(min=1)
    heights = heights.clamp(min=1)
    widths_ceil = widths.ceil()
    heights_ceil = heights.ceil()
    num_keypoints = maps.shape[1]
    xy_preds = torch.zeros((len(rois), 3, num_keypoints), dtype=torch.float32, device=maps.device)
    end_scores = torch.zeros((len(rois), num_keypoints), dtype=torch.float32, device=maps.device)
    for i in range(len(rois)):
        roi_map_width = int(widths_ceil[i].item())
        roi_map_height = int(heights_ceil[i].item())
        width_correction = widths[i] / roi_map_width
        height_correction = heights[i] / roi_map_height
        roi_map = torch.nn.functional.interpolate(maps[i][None], size=(roi_map_height, roi_map_width), mode='bicubic', align_corners=False)[0]
        w = roi_map.shape[2]
        pos = roi_map.reshape(num_keypoints, -1).argmax(dim=1)
        x_int = pos % w
        y_int = (pos - x_int) // w
        x = (x_int.float() + 0.5) * width_correction
        y = (y_int.float() + 0.5) * height_correction
        xy_preds[i, 0, :] = x + offset_x[i]
        xy_preds[i, 1, :] = y + offset_y[i]
        xy_preds[i, 2, :] = 1
        end_scores[i, :] = roi_map[torch.arange(num_keypoints), y_int, x_int]
    return xy_preds.permute(0, 2, 1), end_scores


def keypointrcnn_inference(x, boxes):
    kp_probs = []
    kp_scores = []
    boxes_per_image = [len(box) for box in boxes]
    x2 = x.split(boxes_per_image, dim=0)
    for xx, bb in zip(x2, boxes):
        kp_prob, scores = heatmaps_to_keypoints(xx, bb)
        kp_probs.append(kp_prob)
        kp_scores.append(scores)
    return kp_probs, kp_scores


def keypoints_to_heatmap(keypoints, rois, heatmap_size):
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])
    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]
    x = keypoints[..., 0]
    y = keypoints[..., 1]
    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]
    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()
    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1
    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()
    lin_ind = y * heatmap_size + x
    heatmaps = lin_ind * valid
    return heatmaps, valid


def keypointrcnn_loss(keypoint_logits, proposals, gt_keypoints, keypoint_matched_idxs):
    N, K, H, W = keypoint_logits.shape
    assert H == W
    discretization_size = H
    heatmaps = []
    valid = []
    for proposals_per_image, gt_kp_in_image, midx in zip(proposals, gt_keypoints, keypoint_matched_idxs):
        kp = gt_kp_in_image[midx]
        heatmaps_per_image, valid_per_image = keypoints_to_heatmap(kp, proposals_per_image, discretization_size)
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))
    keypoint_targets = torch.cat(heatmaps, dim=0)
    valid = torch.cat(valid, dim=0)
    valid = torch.nonzero(valid).squeeze(1)
    if keypoint_targets.numel() == 0 or len(valid) == 0:
        return keypoint_logits.sum() * 0
    keypoint_logits = keypoint_logits.view(N * K, H * W)
    keypoint_loss = F.cross_entropy(keypoint_logits[valid], keypoint_targets[valid])
    return keypoint_loss


def maskrcnn_inference(x, labels):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    Arguments:
        x (Tensor): the mask logits
        boxes (list[BoxList]): bounding boxes that are used as
            reference, one for ech image

    Returns:
        results (list[BoxList]): one BoxList for each image, containing
            the extra field mask
    """
    mask_prob = x.sigmoid()
    num_masks = x.shape[0]
    boxes_per_image = [len(l) for l in labels]
    labels = torch.cat(labels)
    index = torch.arange(num_masks, device=labels.device)
    mask_prob = mask_prob[index, labels][:, None]
    mask_prob = mask_prob.split(boxes_per_image, dim=0)
    return mask_prob


def project_masks_on_boxes(gt_masks, boxes, matched_idxs, M):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    """
    matched_idxs = matched_idxs
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    gt_masks = gt_masks[:, None]
    return roi_align(gt_masks, rois, (M, M), 1)[:, 0]


def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
    """
    Arguments:
        proposals (list[BoxList])
        mask_logits (Tensor)
        targets (list[BoxList])

    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """
    discretization_size = mask_logits.shape[-1]
    labels = [l[idxs] for l, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [project_masks_on_boxes(m, p, i, discretization_size) for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)]
    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0
    mask_loss = F.binary_cross_entropy_with_logits(mask_logits[torch.arange(labels.shape[0], device=labels.device), labels], mask_targets)
    return mask_loss


class RoIHeads(torch.nn.Module):

    def __init__(self, box_roi_pool, box_head, box_predictor, fg_iou_thresh, bg_iou_thresh, batch_size_per_image, positive_fraction, bbox_reg_weights, score_thresh, nms_thresh, detections_per_img, mask_roi_pool=None, mask_head=None, mask_predictor=None, keypoint_roi_pool=None, keypoint_head=None, keypoint_predictor=None):
        super(RoIHeads, self).__init__()
        self.box_similarity = box_ops.box_iou
        self.proposal_matcher = det_utils.Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)
        if bbox_reg_weights is None:
            bbox_reg_weights = 10.0, 10.0, 5.0, 5.0
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)
        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor
        self.keypoint_roi_pool = keypoint_roi_pool
        self.keypoint_head = keypoint_head
        self.keypoint_predictor = keypoint_predictor

    @property
    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_head is None:
            return False
        if self.mask_predictor is None:
            return False
        return True

    @property
    def has_keypoint(self):
        if self.keypoint_roi_pool is None:
            return False
        if self.keypoint_head is None:
            return False
        if self.keypoint_predictor is None:
            return False
        return True

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            match_quality_matrix = self.box_similarity(gt_boxes_in_image, proposals_in_image)
            matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)
            clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
            labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
            labels_in_image = labels_in_image
            bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_in_image[bg_inds] = 0
            ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_in_image[ignore_inds] = -1
            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        proposals = [torch.cat((proposal, gt_box)) for proposal, gt_box in zip(proposals, gt_boxes)]
        return proposals

    def check_targets(self, targets):
        assert targets is not None
        assert all('boxes' in t for t in targets)
        assert all('labels' in t for t in targets)
        if self.has_mask:
            assert all('masks' in t for t in targets)

    def select_training_samples(self, proposals, targets):
        self.check_targets(targets)
        gt_boxes = [t['boxes'] for t in targets]
        gt_labels = [t['labels'] for t in targets]
        proposals = self.add_gt_proposals(proposals, gt_boxes)
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
            matched_gt_boxes.append(gt_boxes[img_id][matched_idxs[img_id]])
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)
        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes, pred_scores, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
        return all_boxes, all_scores, all_labels

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)
        result, losses = [], {}
        if self.training:
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(dict(boxes=boxes[i], labels=labels[i], scores=scores[i]))
        if self.has_mask:
            mask_proposals = [p['boxes'] for p in result]
            if self.training:
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
            mask_features = self.mask_head(mask_features)
            mask_logits = self.mask_predictor(mask_features)
            loss_mask = {}
            if self.training:
                gt_masks = [t['masks'] for t in targets]
                gt_labels = [t['labels'] for t in targets]
                loss_mask = maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = dict(loss_mask=loss_mask)
            else:
                labels = [r['labels'] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r['masks'] = mask_prob
            losses.update(loss_mask)
        if self.has_keypoint:
            keypoint_proposals = [p['boxes'] for p in result]
            if self.training:
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)
            loss_keypoint = {}
            if self.training:
                gt_keypoints = [t['keypoints'] for t in targets]
                loss_keypoint = keypointrcnn_loss(keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs)
                loss_keypoint = dict(loss_keypoint=loss_keypoint)
            else:
                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r['keypoints'] = keypoint_prob
                    r['keypoints_scores'] = kps
            losses.update(loss_keypoint)
        return result, losses


class AnchorGenerator(nn.Module):
    """
    Module that generates anchors for a set of feature maps and
    image sizes.

    The module support computing anchors at multiple sizes and aspect ratios
    per feature map.

    sizes and aspect_ratios should have the same number of elements, and it should
    correspond to the number of feature maps.

    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
    and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
    per spatial location for feature map i.

    Arguments:
        sizes (Tuple[Tuple[int]]):
        aspect_ratios (Tuple[Tuple[float]]):
    """

    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        super(AnchorGenerator, self).__init__()
        if not isinstance(sizes[0], (list, tuple)):
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)
        assert len(sizes) == len(aspect_ratios)
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    @staticmethod
    def generate_anchors(scales, aspect_ratios, device='cpu'):
        scales = torch.as_tensor(scales, dtype=torch.float32, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=torch.float32, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()

    def set_cell_anchors(self, device):
        if self.cell_anchors is not None:
            return self.cell_anchors
        cell_anchors = [self.generate_anchors(sizes, aspect_ratios, device) for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)]
        self.cell_anchors = cell_anchors

    def num_anchors_per_location(self):
        return [(len(s) * len(a)) for s, a in zip(self.sizes, self.aspect_ratios)]

    def grid_anchors(self, grid_sizes, strides):
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, strides, self.cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device
            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))
        return anchors

    def cached_grid_anchors(self, grid_sizes, strides):
        key = tuple(grid_sizes) + tuple(strides)
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors

    def forward(self, image_list, feature_maps):
        grid_sizes = tuple([feature_map.shape[-2:] for feature_map in feature_maps])
        image_size = image_list.tensors.shape[-2:]
        strides = tuple((image_size[0] / g[0], image_size[1] / g[1]) for g in grid_sizes)
        self.set_cell_anchors(feature_maps[0].device)
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)
        anchors = []
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        return anchors


class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Arguments:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """

    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)
        for l in self.children():
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    box_cls_flattened = []
    box_regression_flattened = []
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)
    box_cls = torch.cat(box_cls_flattened, dim=1).reshape(-1, C)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


class RegionProposalNetwork(torch.nn.Module):
    """
    Implements Region Proposal Network (RPN).

    Arguments:
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): module that computes the objectness and regression deltas
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        pre_nms_top_n (Dict[int]): number of proposals to keep before applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        post_nms_top_n (Dict[int]): number of proposals to keep after applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals

    """

    def __init__(self, anchor_generator, head, fg_iou_thresh, bg_iou_thresh, batch_size_per_image, positive_fraction, pre_nms_top_n, post_nms_top_n, nms_thresh):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.box_similarity = box_ops.box_iou
        self.proposal_matcher = det_utils.Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True)
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 0

    @property
    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    @property
    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    def assign_targets_to_anchors(self, anchors, targets):
        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image['boxes']
            match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]
            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image
            bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0
            inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1
            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):
            num_anchors = ob.shape[1]
            pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        num_images = proposals.shape[0]
        device = proposals.device
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)
        levels = [torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)]
        levels = torch.cat(levels, 0)
        levels = levels.reshape(1, -1).expand_as(objectness)
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)
        batch_idx = torch.arange(num_images, device=device)[:, None]
        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]
        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, img_shape in zip(proposals, objectness, levels, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)
            keep = keep[:self.post_nms_top_n]
            boxes, scores = boxes[keep], scores[keep]
            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        """
        Arguments:
            objectness (Tensor)
            pred_bbox_deltas (Tensor)
            labels (List[Tensor])
            regression_targets (List[Tensor])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        objectness = objectness.flatten()
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        box_loss = F.l1_loss(pred_bbox_deltas[sampled_pos_inds], regression_targets[sampled_pos_inds], reduction='sum') / sampled_inds.numel()
        objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds])
        return objectness_loss, box_loss

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (List[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[Tensor]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)
        num_images = len(anchors)
        num_anchors_per_level = [o[0].numel() for o in objectness]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)
        losses = {}
        if self.training:
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(objectness, pred_bbox_deltas, labels, regression_targets)
            losses = {'loss_objectness': loss_objectness, 'loss_rpn_box_reg': loss_rpn_box_reg}
        return boxes, losses


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors
        return ImageList(cast_tensor, self.image_sizes)


def expand_boxes(boxes, scale):
    w_half = (boxes[:, 2] - boxes[:, 0]) * 0.5
    h_half = (boxes[:, 3] - boxes[:, 1]) * 0.5
    x_c = (boxes[:, 2] + boxes[:, 0]) * 0.5
    y_c = (boxes[:, 3] + boxes[:, 1]) * 0.5
    w_half *= scale
    h_half *= scale
    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


def expand_masks(mask, padding):
    M = mask.shape[-1]
    scale = float(M + 2 * padding) / M
    padded_mask = torch.nn.functional.pad(mask, (padding,) * 4)
    return padded_mask, scale


def paste_mask_in_image(mask, box, im_h, im_w):
    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)
    mask = mask.expand((1, 1, -1, -1))
    mask = misc_nn_ops.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
    mask = mask[0][0]
    im_mask = torch.zeros((im_h, im_w), dtype=mask.dtype, device=mask.device)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)
    im_mask[y_0:y_1, x_0:x_1] = mask[y_0 - box[1]:y_1 - box[1], x_0 - box[0]:x_1 - box[0]]
    return im_mask


def paste_masks_in_image(masks, boxes, img_shape, padding=1):
    masks, scale = expand_masks(masks, padding=padding)
    boxes = expand_boxes(boxes, scale).tolist()
    im_h, im_w = img_shape
    res = [paste_mask_in_image(m[0], b, im_h, im_w) for m, b in zip(masks, boxes)]
    if len(res) > 0:
        res = torch.stack(res, dim=0)[:, None]
    else:
        res = masks.new_empty((0, 1, im_h, im_w))
    return res


def resize_boxes(boxes, original_size, new_size):
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, original_size))
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


def resize_keypoints(keypoints, original_size, new_size):
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, original_size))
    ratio_h, ratio_w = ratios
    resized_data = keypoints.clone()
    resized_data[..., 0] *= ratio_w
    resized_data[..., 1] *= ratio_h
    return resized_data


class GeneralizedRCNNTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNN
    model.

    The transformations it perform are:
        - input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    """

    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = min_size,
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std

    def forward(self, images, targets=None):
        for i in range(len(images)):
            image = images[i]
            target = targets[i] if targets is not None else targets
            if image.dim() != 3:
                raise ValueError('images is expected to be a list of 3d tensors of shape [C, H, W], got {}'.format(image.shape))
            image = self.normalize(image)
            image, target = self.resize(image, target)
            images[i] = image
            if targets is not None:
                targets[i] = target
        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)
        image_list = ImageList(images, image_sizes)
        return image_list, targets

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        h, w = image.shape[-2:]
        min_size = float(min(image.shape[-2:]))
        max_size = float(max(image.shape[-2:]))
        if self.training:
            size = random.choice(self.min_size)
        else:
            size = self.min_size[-1]
        scale_factor = size / min_size
        if max_size * scale_factor > self.max_size:
            scale_factor = self.max_size / max_size
        image = torch.nn.functional.interpolate(image[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]
        if target is None:
            return image, target
        bbox = target['boxes']
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target['boxes'] = bbox
        if 'masks' in target:
            mask = target['masks']
            mask = misc_nn_ops.interpolate(mask[None].float(), scale_factor=scale_factor)[0].byte()
            target['masks'] = mask
        if 'keypoints' in target:
            keypoints = target['keypoints']
            keypoints = resize_keypoints(keypoints, (h, w), image.shape[-2:])
            target['keypoints'] = keypoints
        return image, target

    def batch_images(self, images, size_divisible=32):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
        stride = size_divisible
        max_size = list(max_size)
        max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
        max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
        max_size = tuple(max_size)
        batch_shape = (len(images),) + max_size
        batched_imgs = images[0].new(*batch_shape).zero_()
        for img, pad_img in zip(images, batched_imgs):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
        return batched_imgs

    def postprocess(self, result, image_shapes, original_image_sizes):
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred['boxes']
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]['boxes'] = boxes
            if 'masks' in pred:
                masks = pred['masks']
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]['masks'] = masks
            if 'keypoints' in pred:
                keypoints = pred['keypoints']
                keypoints = resize_keypoints(keypoints, im_s, o_im_s)
                result[i]['keypoints'] = keypoints
        return result


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_relu=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU() if output_relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x


class Inception(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(BasicConv2d(in_channels, ch3x3red, kernel_size=1), BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1))
        self.branch3 = nn.Sequential(BasicConv2d(in_channels, ch5x5red, kernel_size=1), BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1))
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True), BasicConv2d(in_channels, pool_proj, kernel_size=1))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


def ConvBNReLU(in_channels, out_channels, kernel_size, stride, padding=1):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


class InceptionAux(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(InceptionAux, self).__init__()
        self.auxiliary_avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.auxiliary_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=128, kernel_size=1)
        self.auxiliary_conv2 = nn.Conv2d(in_channels=128, out_channels=768, kernel_size=5, stride=1)
        self.auxiliary_dropout = nn.Dropout(p=0.7)
        self.auxiliary_linear1 = nn.Linear(in_features=768, out_features=out_channels)

    def forward(self, x):
        x = self.auxiliary_conv1(self.auxiliary_avgpool(x))
        x = self.auxiliary_conv2(x)
        x = x.view(x.size(0), -1)
        out = self.auxiliary_linear1(self.auxiliary_dropout(x))
        return out


_GoogLeNetOuputs = namedtuple('GoogLeNetOuputs', ['logits', 'aux_logits2', 'aux_logits1'])


class GoogLeNet(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False, init_weights=True):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        if self.training and self.aux_logits:
            return _GoogLeNetOuputs(x, aux2, aux1)
        return x


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)
        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)
        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)
        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


_InceptionOuputs = namedtuple('InceptionOuputs', ['logits', 'aux_logits'])


class Inception3(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False):
        super(Inception3, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.fc = nn.Linear(2048, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = F.dropout(x, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.training and self.aux_logits:
            return _InceptionOuputs(x, aux)
        return x


def Conv1x1BN(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1), nn.BatchNorm2d(out_channels))


def Conv1x1BNReLU(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


def Conv3x3BNReLU(in_channels, out_channels, stride, padding=1):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expansion_factor=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        mid_channels = in_channels * expansion_factor
        self.bottleneck = nn.Sequential(Conv1x1BNReLU(in_channels, mid_channels), Conv3x3BNReLU(mid_channels, mid_channels, stride, groups=mid_channels), Conv1x1BN(mid_channels, out_channels))
        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        out = self.bottleneck(x)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2(nn.Module):

    def __init__(self, num_classes=1000):
        super(MobileNetV2, self).__init__()
        self.first_conv = Conv3x3BNReLU(3, 32, 2, groups=1)
        self.layer1 = self.make_layer(in_channels=32, out_channels=16, stride=1, block_num=1)
        self.layer2 = self.make_layer(in_channels=16, out_channels=24, stride=2, block_num=2)
        self.layer3 = self.make_layer(in_channels=24, out_channels=32, stride=2, block_num=3)
        self.layer4 = self.make_layer(in_channels=32, out_channels=64, stride=2, block_num=4)
        self.layer5 = self.make_layer(in_channels=64, out_channels=96, stride=1, block_num=3)
        self.layer6 = self.make_layer(in_channels=96, out_channels=160, stride=2, block_num=3)
        self.layer7 = self.make_layer(in_channels=160, out_channels=320, stride=1, block_num=1)
        self.last_conv = Conv1x1BNReLU(320, 1280)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_features=1280, out_features=num_classes)

    def make_layer(self, in_channels, out_channels, stride, block_num):
        layers = []
        layers.append(InvertedResidual(in_channels, out_channels, stride))
        for i in range(1, block_num):
            layers.append(InvertedResidual(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.last_conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        out = self.linear(x)
        return out


class _SimpleSegmentationModel(nn.Module):

    def __init__(self, backbone, classifier, aux_classifier=None):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        result = OrderedDict()
        x = features['out']
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result['out'] = x
        if self.aux_classifier is not None:
            x = features['aux']
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result['aux'] = x
        return result


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass


class ASPP(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.depthwise1 = ConvBNReLU(in_channels, out_channels, 3, 1, 6, dilation=6)
        self.depthwise2 = ConvBNReLU(in_channels, out_channels, 3, 1, 12, dilation=12)
        self.depthwise3 = ConvBNReLU(in_channels, out_channels, 3, 1, 18, dilation=18)
        self.pointconv = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        x1 = self.depthwise1(x)
        x2 = self.depthwise2(x)
        x3 = self.depthwise3(x)
        x4 = self.pointconv(x)
        return torch.cat([x1, x2, x3, x4], dim=1)


class DeepLabHead(nn.Sequential):

    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(ASPP(in_channels, [12, 24, 36]), nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, num_classes, 1))


class ASPPConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, dilation):
        modules = [nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU()]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class FCN(_SimpleSegmentationModel):
    """
    Implements a Fully-Convolutional Network for semantic segmentation.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass


class FCNHead(nn.Sequential):

    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False), nn.BatchNorm2d(inter_channels), nn.ReLU(), nn.Dropout(0.1), nn.Conv2d(inter_channels, channels, 1)]
        super(FCNHead, self).__init__(*layers)


class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)


class ShuffleNetUnits(nn.Module):

    def __init__(self, in_channels, out_channels, stride, groups):
        super(ShuffleNetUnits, self).__init__()
        self.stride = stride
        out_channels = out_channels - in_channels if self.stride > 1 else out_channels
        mid_channels = out_channels // 4
        self.bottleneck = nn.Sequential(Conv1x1BNReLU(in_channels, mid_channels, groups), ChannelShuffle(groups), Conv3x3BNReLU(mid_channels, mid_channels, stride, groups), Conv1x1BN(mid_channels, out_channels, groups))
        if self.stride > 1:
            self.shortcut = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = self.bottleneck(x)
        out = torch.cat([self.shortcut(x), out], dim=1) if self.stride > 1 else out + x
        return self.relu(out)


class ShuffleNetV2(nn.Module):

    def __init__(self, planes, layers, groups, num_classes=1000):
        super(ShuffleNetV2, self).__init__()
        self.groups = groups
        self.stage1 = nn.Sequential(Conv3x3BNReLU(in_channels=3, out_channels=24, stride=2, groups=1), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.stage2 = self._make_layer(24, planes[0], layers[0], True)
        self.stage3 = self._make_layer(planes[0], planes[1], layers[1], False)
        self.stage4 = self._make_layer(planes[1], planes[2], layers[2], False)
        self.global_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_features=planes[2] * 7 * 7, out_features=num_classes)
        self.init_params()

    def _make_layer(self, in_channels, out_channels, block_num, is_stage2):
        layers = []
        layers.append(ShuffleNetUnits(in_channels=in_channels, out_channels=out_channels, stride=2, groups=1 if is_stage2 else self.groups))
        for idx in range(1, block_num):
            layers.append(ShuffleNetUnits(in_channels=out_channels, out_channels=out_channels, stride=1, groups=self.groups))
        return nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        out = self.linear(x)
        return out


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1)


class FireModule(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(FireModule, self).__init__()
        mid_channels = out_channels // 4
        self.squeeze = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1)
        self.squeeze_relu = nn.ReLU6(inplace=True)
        self.expand3x3 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.expand3x3_relu = nn.ReLU6(inplace=True)
        self.expand1x1 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.expand1x1_relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.squeeze_relu(self.squeeze(x))
        y = self.expand3x3_relu(self.expand3x3(x))
        z = self.expand1x1_relu(self.expand1x1(x))
        out = torch.cat([y, z], dim=1)
        return out


class SqueezeNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(SqueezeNet, self).__init__()
        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(96), nn.ReLU6(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), FireModule(in_channels=96, out_channels=64), FireModule(in_channels=128, out_channels=64), FireModule(in_channels=128, out_channels=128), nn.MaxPool2d(kernel_size=3, stride=2), FireModule(in_channels=256, out_channels=128), FireModule(in_channels=256, out_channels=192), FireModule(in_channels=384, out_channels=192), FireModule(in_channels=384, out_channels=256), nn.MaxPool2d(kernel_size=3, stride=2), FireModule(in_channels=512, out_channels=256), nn.Dropout(p=0.5), nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1, stride=1), nn.ReLU(inplace=True), nn.AvgPool2d(kernel_size=13, stride=1))

    def forward(self, x):
        out = self.bottleneck(x)
        return out.view(out.size(1), -1)


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, num_classes))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Precision_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):
        sample_prec = precision_score(true_labels, predict_labels, average='samples')
        micro_prec = precision_score(true_labels, predict_labels, average='micro')
        macro_prec = precision_score(true_labels, predict_labels, average='macro')
        return macro_prec, micro_prec, sample_prec


class Recall_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):
        sample_rec = recall_score(true_labels, predict_labels, average='samples')
        micro_rec = recall_score(true_labels, predict_labels, average='micro')
        macro_rec = recall_score(true_labels, predict_labels, average='macro')
        return macro_rec, micro_rec, sample_rec


class F1_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):
        macro_f1 = f1_score(true_labels, predict_labels, average='macro')
        micro_f1 = f1_score(true_labels, predict_labels, average='micro')
        sample_f1 = f1_score(true_labels, predict_labels, average='samples')
        return macro_f1, micro_f1, sample_f1


class F2_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):
        macro_f2 = fbeta_score(true_labels, predict_labels, beta=2, average='macro')
        micro_f2 = fbeta_score(true_labels, predict_labels, beta=2, average='micro')
        sample_f2 = fbeta_score(true_labels, predict_labels, beta=2, average='samples')
        return macro_f2, micro_f2, sample_f2


class Hamming_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):
        return hamming_loss(true_labels, predict_labels)


class Subset_accuracy(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):
        return accuracy_score(true_labels, predict_labels)


class One_error(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_probs, true_labels):
        row_inds = np.arange(predict_probs.shape[0])
        col_inds = np.argmax(predict_probs, axis=1)
        return np.mean((true_labels[tuple(row_inds), tuple(col_inds)] == 0).astype(int))


class Coverage_error(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_probs, true_labels):
        return coverage_error(true_labels, predict_probs)


class Ranking_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_probs, true_labels):
        return label_ranking_loss(true_labels, predict_probs)


class LabelAvgPrec_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_probs, true_labels):
        return label_ranking_average_precision_score(true_labels, predict_probs)


class calssification_report(nn.Module):

    def __init__(self, target_names):
        super().__init__()
        self.target_names = target_names

    def forward(self, predict_labels, true_labels):
        report = classification_report(true_labels, predict_labels, target_names=self.target_names, output_dict=True)
        return report


class Accuracy_score(nn.Module):

    def __init__(self):
        super().__init__()

    def Mean_average_precision(self, preds, targets):
        map = []
        for i in range(len(targets[0])):
            pred, target = preds[:, i], targets[:, i]
            z = np.argsort(-pred)
            target = target[z]
            prec = []
            rec = []
            for i in range(len(target)):
                p = float(np.sum(target[0:i + 1])) / (i + 1)
                r = float(np.sum(target[0:i + 1])) / np.sum(target)
                prec.append(p)
                rec.append(r)
            prec, rec = np.array(prec), np.array(rec)
            n = int(np.sum(target))
            ap = 0
            if n != 0:
                for i in range(n):
                    t = float(i) / n
                    z = np.where(rec > t)
                    p = np.max(prec[z])
                    ap += p / n
                map.append(ap)
        map = np.array(map)
        return map.mean()

    def forward(self, predict_labels, true_labels):
        LRAP = label_ranking_average_precision_score(true_labels, predict_labels)
        map = self.Mean_average_precision(predict_labels, true_labels)
        predict_labels = np.round(predict_labels)
        TP = np.logical_and(predict_labels == 1, true_labels == 1).astype(int)
        FP = np.logical_and(predict_labels == 1, true_labels == 0).astype(int)
        TN = np.logical_and(predict_labels == 0, true_labels == 0).astype(int)
        FN = np.logical_and(predict_labels == 0, true_labels == 1).astype(int)
        union = np.logical_or(predict_labels == 1, true_labels == 1).astype(int)
        TP_sample = TP.sum(axis=1)
        union_sample = union.sum(axis=1)
        sample_Acc = TP_sample / union_sample
        assert np.isfinite(sample_Acc).all(), 'Nan found in sample accuracy'
        TP_cls = TP.sum(axis=0)
        FP_cls = FP.sum(axis=0)
        TN_cls = TN.sum(axis=0)
        FN_cls = FN.sum(axis=0)
        assert (TP_cls + FP_cls + TN_cls + FN_cls == predict_labels.shape[0]).all(), 'wrong'
        prec = TP_cls / (TP_cls + FP_cls)
        prec = prec[np.where(TP_cls + FP_cls != 0)].mean()
        rec = TP_cls / (TP_cls + FN_cls)
        rec = rec[np.where(TP_cls + FN_cls != 0)].mean()
        f1 = 2 * prec * rec / (prec + rec)
        macro_Acc = (TP_cls + TN_cls) / (TP_cls + FP_cls + TN_cls + FN_cls)
        return LRAP, map, sample_Acc.mean(), macro_Acc, prec, rec, f1


class DoubleAttention(nn.Module):

    def __init__(self, in_channels, c_m, c_n, reconstruct=True):
        super().__init__()
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        self.c_m = c_m
        self.c_n = c_n
        self.convA = nn.Conv2d(in_channels, c_m, 1)
        self.convB = nn.Conv2d(in_channels, c_n, 1)
        self.convV = nn.Conv2d(in_channels, c_n, 1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        assert c == self.in_channels
        A = self.convA(x)
        B = self.convB(x)
        V = self.convV(x)
        tmpA = A.view(b, self.c_m, -1)
        attention_maps = F.softmax(B.view(b, self.c_n, -1))
        attention_vectors = F.softmax(V.view(b, self.c_n, -1))
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))
        tmpZ = global_descriptors.matmul(attention_vectors)
        tmpZ = tmpZ.view(b, self.c_m, h, w)
        if self.reconstruct:
            tmpZ = self.conv_reconstruct(tmpZ)
        return tmpZ


class AFT_FULL(nn.Module):

    def __init__(self, d_model, n=49, simple=False):
        super(AFT_FULL, self).__init__()
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        if simple:
            self.position_biases = torch.zeros((n, n))
        else:
            self.position_biases = nn.Parameter(torch.ones((n, n)))
        self.d_model = d_model
        self.n = n
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        bs, n, dim = input.shape
        q = self.fc_q(input)
        k = self.fc_k(input).view(1, bs, n, dim)
        v = self.fc_v(input).view(1, bs, n, dim)
        numerator = torch.sum(torch.exp(k + self.position_biases.view(n, 1, -1, 1)) * v, dim=2)
        denominator = torch.sum(torch.exp(k + self.position_biases.view(n, 1, -1, 1)), dim=2)
        out = numerator / denominator
        out = self.sigmoid(q) * out.permute(1, 0, 2)
        return out


class SpatialPyramidPooling(nn.Module):

    def __init__(self, output_sizes=[1, 3, 6, 8]):
        super(SpatialPyramidPooling, self).__init__()
        self.pool_layers = nn.ModuleList()
        for output_size in output_sizes:
            self.pool_layers.append(nn.AdaptiveMaxPool2d(output_size=output_size))

    def forward(self, x):
        outputs = []
        for pool_layer in self.pool_layers:
            outputs.append(pool_layer(x).flatten())
        out = torch.cat(outputs, dim=0)
        return out


class APNB(nn.Module):

    def __init__(self, channel):
        super(APNB, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        x_phi = self.conv_phi(x).view(b, c, -1)
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out


class AFNB(nn.Module):

    def __init__(self, channel):
        super(AFNB, self).__init__()
        self.inter_channel = channel // 2
        self.output_sizes = [1, 3, 6, 8]
        self.sample_dim = np.sum([(size * size) for size in self.output_sizes])
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_theta_spp = SpatialPyramidPooling(self.output_sizes)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g_spp = SpatialPyramidPooling(self.output_sizes)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        x_phi = self.conv_phi(x).view(b, c, -1)
        xxx = self.conv_theta_spp(self.conv_theta(x))
        None
        x_theta = self.conv_theta_spp(self.conv_theta(x)).view(b, self.sample_dim, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g_spp(self.conv_g(x)).view(b, self.sample_dim, -1).permute(0, 2, 1).contiguous()
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.shape[0], -1)


class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.global_avgpool = nn.AdaptiveAvgPool1d(5)
        self.shared_MLP = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(), nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.conv1 = nn.Conv1d(7, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_all = self.shared_MLP(self.avg_pool(x))
        max_all = self.shared_MLP(self.max_pool(x))
        B, C, H, W = x.size()
        sortx = x.view(B, C, -1)
        sortx = torch.sort(sortx, dim=-1)[0].clone()
        bavg_out = self.global_avgpool(sortx)
        avg_out = torch.cat([avg_all, max_all], dim=1).view(B, 2, -1)
        for i in range(5):
            m = bavg_out[:, :, i].unsqueeze(-1).unsqueeze(-1)
            avg_m = self.shared_MLP(m).view(B, 1, -1)
            avg_out = torch.cat([avg_out, avg_m], dim=1)
        out = self.conv1(avg_out).view(B, -1, 1)
        out = out.unsqueeze(-1)
        out = self.sigmoid(out)
        return out


class Conv2dBnRelu(nn.Module):

    def __init__(self, in_ch, out_ch, kelnel_size=3, stride=1, padding=0, bias=False):
        super(Conv2dBnRelu, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kelnel_size, stride=stride, padding=padding, bias=bias), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class SpatialAttention(nn.Module):

    def __init__(self, in_planes):
        super(SpatialAttention, self).__init__()
        self.pyramid1 = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2), Conv2dBnRelu(in_planes, 1, kelnel_size=7, stride=1, padding=3))
        self.pyramid2 = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2), Conv2dBnRelu(1, 1, kelnel_size=5, stride=1, padding=2))
        self.pyramid3 = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2), Conv2dBnRelu(1, 1, kelnel_size=3, stride=1, padding=1))
        self.conv1 = Conv2dBnRelu(1, 1, kelnel_size=7, stride=1, padding=3)
        self.conv2 = Conv2dBnRelu(1, 1, kelnel_size=5, stride=1, padding=2)
        self.conv3 = Conv2dBnRelu(1, 1, kelnel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x1 = self.pyramid1(x)
        x2 = self.pyramid2(x1)
        x3 = self.pyramid3(x2)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x3 = nn.Upsample(size=(h // 4, w // 4), mode='bilinear', align_corners=True)(x3)
        x = x3 + x2
        x = nn.Upsample(size=(h // 2, w // 2), mode='bilinear', align_corners=True)(x)
        x = x + x1
        x = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(x)
        return self.sigmoid(x)


class BAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, dia_val=2):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(channel=channel, reduction=reduction, dia_val=dia_val)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        sa_out = self.sa(x)
        ca_out = self.ca(x)
        weight = self.sigmoid(sa_out + ca_out)
        out = (1 + weight) * x
        return out


class ChannelAttentionModule(nn.Module):

    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(nn.Conv2d(channel, channel // ratio, 1, bias=False), nn.ReLU(), nn.Conv2d(channel // ratio, channel, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):

    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):

    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class ResBlock_CBAM(nn.Module):

    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(ResBlock_CBAM, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(places), nn.ReLU(inplace=True), nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(places), nn.ReLU(inplace=True), nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(places * self.expansion))
        self.cbam = CBAM(channel=places * self.expansion)
        if self.downsampling:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(places * self.expansion))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        out = self.cbam(out)
        if self.downsampling:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1000000000.0)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class CoAtNet(nn.Module):

    def __init__(self, in_ch, image_size, out_chs=[64, 96, 192, 384, 768]):
        super().__init__()
        self.out_chs = out_chs
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)
        self.s0 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1))
        self.mlp0 = nn.Sequential(nn.Conv2d(in_ch, out_chs[0], kernel_size=1), nn.ReLU(), nn.Conv2d(out_chs[0], out_chs[0], kernel_size=1))
        self.s1 = MBConvBlock(ksize=3, input_filters=out_chs[0], output_filters=out_chs[0], image_size=image_size // 2)
        self.mlp1 = nn.Sequential(nn.Conv2d(out_chs[0], out_chs[1], kernel_size=1), nn.ReLU(), nn.Conv2d(out_chs[1], out_chs[1], kernel_size=1))
        self.s2 = MBConvBlock(ksize=3, input_filters=out_chs[1], output_filters=out_chs[1], image_size=image_size // 4)
        self.mlp2 = nn.Sequential(nn.Conv2d(out_chs[1], out_chs[2], kernel_size=1), nn.ReLU(), nn.Conv2d(out_chs[2], out_chs[2], kernel_size=1))
        self.s3 = ScaledDotProductAttention(out_chs[2], out_chs[2] // 8, out_chs[2] // 8, 8)
        self.mlp3 = nn.Sequential(nn.Linear(out_chs[2], out_chs[3]), nn.ReLU(), nn.Linear(out_chs[3], out_chs[3]))
        self.s4 = ScaledDotProductAttention(out_chs[3], out_chs[3] // 8, out_chs[3] // 8, 8)
        self.mlp4 = nn.Sequential(nn.Linear(out_chs[3], out_chs[4]), nn.ReLU(), nn.Linear(out_chs[4], out_chs[4]))

    def forward(self, x):
        B, C, H, W = x.shape
        y = self.mlp0(self.s0(x))
        y = self.maxpool2d(y)
        y = self.mlp1(self.s1(y))
        y = self.maxpool2d(y)
        y = self.mlp2(self.s2(y))
        y = self.maxpool2d(y)
        y = y.reshape(B, self.out_chs[2], -1).permute(0, 2, 1)
        y = self.mlp3(self.s3(y, y, y))
        y = self.maxpool1d(y.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.mlp4(self.s4(y, y, y))
        y = self.maxpool1d(y.permute(0, 2, 1))
        N = y.shape[-1]
        y = y.reshape(B, self.out_chs[4], int(sqrt(N)), int(sqrt(N)))
        return y


class CoTAttention(nn.Module):

    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.key_embed = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=4, bias=False), nn.BatchNorm2d(dim), nn.ReLU())
        self.value_embed = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim))
        factor = 4
        self.attention_embed = nn.Sequential(nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False), nn.BatchNorm2d(2 * dim // factor), nn.ReLU(), nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1))

    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)
        v = self.value_embed(x).view(bs, c, -1)
        y = torch.cat([k1, x], dim=1)
        att = self.attention_embed(y)
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)
        k2 = F.softmax(att, dim=-1) * v
        k2 = k2.view(bs, c, h, w)
        return k1 + k2


class h_sigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):

    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):

    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out


class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x
        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)
        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x
        return out


class _DAHead(nn.Module):

    def __init__(self, in_channels, nclass, aux=True, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DAHead, self).__init__()
        self.aux = aux
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False), norm_layer(inter_channels, **{} if norm_kwargs is None else norm_kwargs), nn.ReLU(True))
        self.conv_c1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False), norm_layer(inter_channels, **{} if norm_kwargs is None else norm_kwargs), nn.ReLU(True))
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False), norm_layer(inter_channels, **{} if norm_kwargs is None else norm_kwargs), nn.ReLU(True))
        self.conv_c2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False), norm_layer(inter_channels, **{} if norm_kwargs is None else norm_kwargs), nn.ReLU(True))
        self.out = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(inter_channels, nclass, 1))
        if aux:
            self.conv_p3 = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(inter_channels, nclass, 1))
            self.conv_c3 = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(inter_channels, nclass, 1))

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)
        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)
        feat_fusion = feat_p + feat_c
        outputs = []
        fusion_out = self.out(feat_fusion)
        outputs.append(fusion_out)
        if self.aux:
            p_out = self.conv_p3(feat_p)
            c_out = self.conv_c3(feat_c)
            outputs.append(p_out)
            outputs.append(c_out)
        return tuple(outputs)


class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.gap(x)
        y = y.squeeze(-1).permute(0, 2, 1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1).unsqueeze(-1)
        return x * y.expand_as(x)


class EMSA(nn.Module):

    def __init__(self, d_model, d_k, d_v, h, dropout=0.1, H=7, W=7, ratio=3, apply_transform=True):
        super(EMSA, self).__init__()
        self.H = H
        self.W = W
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.ratio = ratio
        if self.ratio > 1:
            self.sr = nn.Sequential()
            self.sr_conv = nn.Conv2d(d_model, d_model, kernel_size=ratio + 1, stride=ratio, padding=ratio // 2, groups=d_model)
            self.sr_ln = nn.LayerNorm(d_model)
        self.apply_transform = apply_transform and h > 1
        if self.apply_transform:
            self.transform = nn.Sequential()
            self.transform.add_module('conv', nn.Conv2d(h, h, kernel_size=1, stride=1))
            self.transform.add_module('softmax', nn.Softmax(-1))
            self.transform.add_module('in', nn.InstanceNorm2d(h))
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        b_s, nq, c = queries.shape
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        if self.ratio > 1:
            x = queries.permute(0, 2, 1).view(b_s, c, self.H, self.W)
            x = self.sr_conv(x)
            x = x.contiguous().view(b_s, c, -1).permute(0, 2, 1)
            x = self.sr_ln(x)
            k = self.fc_k(x).view(b_s, -1, self.h, self.d_k).permute(0, 2, 3, 1)
            v = self.fc_v(x).view(b_s, -1, self.h, self.d_v).permute(0, 2, 1, 3)
        else:
            k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
            v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)
        if self.apply_transform:
            att = torch.matmul(q, k) / np.sqrt(self.d_k)
            att = self.transform(att)
        else:
            att = torch.matmul(q, k) / np.sqrt(self.d_k)
            att = torch.softmax(att, -1)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = self.dropout(att)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)
        return out


class ExternalAttention(nn.Module):

    def __init__(self, d_model, S=64):
        super().__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn = self.mk(queries)
        attn = self.softmax(attn)
        attn = attn / torch.sum(attn, dim=2, keepdim=True)
        out = self.mv(attn)
        return out


class GlobalContextBlock(nn.Module):

    def __init__(self, inplanes, ratio, pooling_type='att', fusion_types=('channel_add',)):
        super(GlobalContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([(f in valid_fusion_types) for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(nn.Conv2d(self.inplanes, self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(nn.Conv2d(self.inplanes, self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            input_x = input_x.view(batch, channel, height * width)
            input_x = input_x.unsqueeze(1)
            context_mask = self.conv_mask(x)
            context_mask = context_mask.view(batch, 1, height * width)
            context_mask = self.softmax(context_mask)
            context_mask = context_mask.unsqueeze(-1)
            context = torch.matmul(input_x, context_mask)
            context = context.view(batch, channel, 1, 1)
        else:
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def to(x):
    return {'device': x.device, 'dtype': x.dtype}


def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2
    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim=2)
    flat_x = rearrange(x, 'b l c -> b (l c)')
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x


def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2
    logits = einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')
    logits = rel_to_abs(logits)
    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim=2, k=r)
    return logits


class RelPosEmb(nn.Module):

    def __init__(self, block_size, rel_size, dim_head):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5
        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size
        q = rearrange(q, 'b (x y) c -> b x y c', x=block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')
        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        return rel_logits_w + rel_logits_h


class HaloAttention(nn.Module):

    def __init__(self, *, dim, block_size, halo_size, dim_head=64, heads=8):
        super().__init__()
        assert halo_size > 0, 'halo size must be greater than 0'
        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.block_size = block_size
        self.halo_size = halo_size
        inner_dim = dim_head * heads
        self.rel_pos_emb = RelPosEmb(block_size=block_size, rel_size=block_size + halo_size * 2, dim_head=dim_head)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        b, c, h, w, block, halo, heads, device = *x.shape, self.block_size, self.halo_size, self.heads, x.device
        assert h % block == 0 and w % block == 0, 'fmap dimensions must be divisible by the block size'
        assert c == self.dim, f'channels for input ({c}) does not equal to the correct dimension ({self.dim})'
        q_inp = rearrange(x, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=block, p2=block)
        kv_inp = F.unfold(x, kernel_size=block + halo * 2, stride=block, padding=halo)
        kv_inp = rearrange(kv_inp, 'b (c j) i -> (b i) j c', c=c)
        q = self.to_q(q_inp)
        k, v = self.to_kv(kv_inp).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=heads), (q, k, v))
        q *= self.scale
        sim = einsum('b i d, b j d -> b i j', q, k)
        sim += self.rel_pos_emb(q)
        mask = torch.ones(1, 1, h, w, device=device)
        mask = F.unfold(mask, kernel_size=block + halo * 2, stride=block, padding=halo)
        mask = repeat(mask, '() j i -> (b i h) () j', b=b, h=heads)
        mask = mask.bool()
        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(mask, max_neg_value)
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=heads)
        out = self.to_out(out)
        out = rearrange(out, '(b h w) (p1 p2) c -> b c (h p1) (w p2)', b=b, h=h // block, w=w // block, p1=block, p2=block)
        return out


class Depth_Pointwise_Conv1d(nn.Module):

    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        if k == 1:
            self.depth_conv = nn.Identity()
        else:
            self.depth_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, groups=1)

    def forward(self, x):
        out = self.pointwise_conv(self.depth_conv(x))
        return out


class MUSEAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, h, dropout=0.1):
        super(MUSEAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = Depth_Pointwise_Conv1d(h * d_v, d_model, 1)
        self.conv3 = Depth_Pointwise_Conv1d(h * d_v, d_model, 3)
        self.conv5 = Depth_Pointwise_Conv1d(h * d_v, d_model, 5)
        self.dy_paras = nn.Parameter(torch.ones(3))
        self.softmax = nn.Softmax(-1)
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)
        v2 = v.permute(0, 1, 3, 2).contiguous().view(b_s, -1, nk)
        self.dy_paras = nn.Parameter(self.softmax(self.dy_paras))
        out2 = self.dy_paras[0] * self.conv1(v2) + self.dy_paras[1] * self.conv3(v2) + self.dy_paras[2] * self.conv5(v2)
        out2 = out2.permute(0, 2, 1)
        out = out + out2
        return out


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.ln(x), **kwargs)


class FeedForward(nn.Module):

    def __init__(self, dim, mlp_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, mlp_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(mlp_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, heads, head_dim, dropout):
        super().__init__()
        inner_dim = heads * head_dim
        project_out = not (heads == 1 and head_dim == dim)
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Decoder(nn.Module):

    def __init__(self, num_classes=2):
        super(Decoder, self).__init__()
        self.aspp = ASPP(320, 128)
        self.pconv1 = Conv1x1BN(128 * 4, 512)
        self.pconv2 = Conv1x1BN(512 + 32, 128)
        self.pconv3 = Conv1x1BN(128, num_classes)

    def forward(self, x, y):
        x = self.pconv1(self.aspp(x))
        x = F.interpolate(x, y.shape[2:], align_corners=True, mode='bilinear')
        x = torch.cat([x, y], dim=1)
        out = self.pconv3(self.pconv2(x))
        return out


class LWbottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(LWbottleneck, self).__init__()
        self.stride = stride
        self.pyramid_list = nn.ModuleList()
        self.pyramid_list.append(ConvBNReLU(in_channels, in_channels, kernel_size=[5, 1], stride=stride, padding=[2, 0]))
        self.pyramid_list.append(ConvBNReLU(in_channels, in_channels, kernel_size=[1, 5], stride=stride, padding=[0, 2]))
        self.pyramid_list.append(ConvBNReLU(in_channels, in_channels, kernel_size=[3, 1], stride=stride, padding=[1, 0]))
        self.pyramid_list.append(ConvBNReLU(in_channels, in_channels, kernel_size=[1, 3], stride=stride, padding=[0, 1]))
        self.pyramid_list.append(ConvBNReLU(in_channels, in_channels, kernel_size=[2, 1], stride=stride, padding=[1, 0]))
        self.pyramid_list.append(ConvBNReLU(in_channels, in_channels, kernel_size=[1, 2], stride=stride, padding=[0, 1]))
        self.pyramid_list.append(ConvBNReLU(in_channels, in_channels, kernel_size=2, stride=stride, padding=1))
        self.pyramid_list.append(ConvBNReLU(in_channels, in_channels, kernel_size=3, stride=stride, padding=1))
        self.shrink = Conv1x1BN(in_channels * 8, out_channels)

    def forward(self, x):
        b, c, w, h = x.shape
        if self.stride > 1:
            w, h = w // self.stride, h // self.stride
        outputs = []
        for pyconv in self.pyramid_list:
            pyconv_x = pyconv(x)
            if x.shape[2:] != pyconv_x.shape[2:]:
                pyconv_x = pyconv_x[:, :, :w, :h]
            outputs.append(pyconv_x)
        out = torch.cat(outputs, 1)
        return self.shrink(out)


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.stage1 = nn.Sequential(ConvBNReLU(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1), Conv1x1BN(in_channels=32, out_channels=16))
        self.stage2 = nn.Sequential(LWbottleneck(in_channels=16, out_channels=24, stride=2), LWbottleneck(in_channels=24, out_channels=24, stride=1))
        self.stage3 = nn.Sequential(LWbottleneck(in_channels=24, out_channels=32, stride=2), LWbottleneck(in_channels=32, out_channels=32, stride=1))
        self.stage4 = nn.Sequential(LWbottleneck(in_channels=32, out_channels=32, stride=2))
        self.stage5 = nn.Sequential(LWbottleneck(in_channels=32, out_channels=64, stride=2), LWbottleneck(in_channels=64, out_channels=64, stride=1), LWbottleneck(in_channels=64, out_channels=64, stride=1), LWbottleneck(in_channels=64, out_channels=64, stride=1))
        self.conv1 = Conv1x1BN(in_channels=64, out_channels=320)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0)
        out1 = x = self.stage3(x)
        x = self.stage4(x)
        x = F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0)
        x = self.stage5(x)
        out2 = self.conv1(x)
        return out1, out2


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    """ For masking out the subsequent info. """
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx, d_word_vec=512, d_model=512, d_inner=2048, n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200, trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True):
        super().__init__()
        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx
        self.encoder = Encoder(n_src_vocab=n_src_vocab, n_position=n_position, d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner, n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, pad_idx=src_pad_idx, dropout=dropout)
        self.decoder = Decoder(n_trg_vocab=n_trg_vocab, n_position=n_position, d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner, n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, pad_idx=trg_pad_idx, dropout=dropout)
        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        assert d_model == d_word_vec, 'To facilitate the residual connections,          the dimensions of all module outputs shall be the same.'
        self.x_logit_scale = 1.0
        if trg_emb_prj_weight_sharing:
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight
            self.x_logit_scale = d_model ** -0.5
        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight

    def forward(self, src_seq, trg_seq):
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)
        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output) * self.x_logit_scale
        return seq_logit.view(-1, seq_logit.size(2))


class MobileViTAttention(nn.Module):

    def __init__(self, in_channel=3, dim=512, kernel_size=3, patch_size=7):
        super().__init__()
        self.ph, self.pw = patch_size, patch_size
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(in_channel, dim, kernel_size=1)
        self.trans = Transformer(dim=dim, depth=3, heads=8, head_dim=64, mlp_dim=1024)
        self.conv3 = nn.Conv2d(dim, in_channel, kernel_size=1)
        self.conv4 = nn.Conv2d(2 * in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        y = x.clone()
        y = self.conv2(self.conv1(x))
        _, _, h, w = y.shape
        y = rearrange(y, 'bs dim (nh ph) (nw pw) -> bs (ph pw) (nh nw) dim', ph=self.ph, pw=self.pw)
        y = self.trans(y)
        y = rearrange(y, 'bs (ph pw) (nh nw) dim -> bs dim (nh ph) (nw pw)', ph=self.ph, pw=self.pw, nh=h // self.ph, nw=w // self.pw)
        y = self.conv3(y)
        y = torch.cat([x, y], 1)
        y = self.conv4(y)
        return y


class NonLocalBlock(nn.Module):

    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        x_phi = self.conv_phi(x).view(b, c, -1)
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out


class OutlookAttention(nn.Module):

    def __init__(self, dim, num_heads=1, kernel_size=3, padding=1, stride=1, qkv_bias=False, attn_drop=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = self.head_dim ** -0.5
        self.v_pj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn = nn.Linear(dim, kernel_size ** 4 * num_heads)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(attn_drop)
        self.unflod = nn.Unfold(kernel_size, padding, stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, x):
        B, H, W, C = x.shape
        v = self.v_pj(x).permute(0, 3, 1, 2)
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        v = self.unflod(v).reshape(B, self.num_heads, self.head_dim, self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)
        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        attn = self.attn(attn).reshape(B, h * w, self.num_heads, self.kernel_size * self.kernel_size, self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)
        attn = self.scale * attn
        attn = attn.softmax(-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
        out = F.fold(out, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        out = self.proj(out.permute(0, 2, 3, 1))
        out = self.proj_drop(out)
        return out


class PSA(nn.Module):

    def __init__(self, channel=512, reduction=4, S=4):
        super().__init__()
        self.S = S
        self.convs = []
        for i in range(S):
            self.convs.append(nn.Conv2d(channel // S, channel // S, kernel_size=2 * (i + 1) + 1, padding=i + 1))
        self.se_blocks = []
        for i in range(S):
            self.se_blocks.append(nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channel // S, channel // (S * reduction), kernel_size=1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(channel // (S * reduction), channel // S, kernel_size=1, bias=False), nn.Sigmoid()))
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.size()
        SPC_out = x.view(b, self.S, c // self.S, h, w)
        for idx, conv in enumerate(self.convs):
            SPC_out[:, idx, :, :, :] = conv(SPC_out[:, idx, :, :, :])
        se_out = []
        for idx, se in enumerate(self.se_blocks):
            se_out.append(se(SPC_out[:, idx, :, :, :]))
        SE_out = torch.stack(se_out, dim=1)
        SE_out = SE_out.expand_as(SPC_out)
        softmax_out = self.softmax(SE_out)
        PSA_out = SPC_out * softmax_out
        PSA_out = PSA_out.view(b, -1, h, w)
        return PSA_out


class ParNetAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.sse = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channel, channel, kernel_size=1), nn.Sigmoid())
        self.conv1x1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=1), nn.BatchNorm2d(channel))
        self.conv3x3 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel))
        self.silu = nn.SiLU()

    def forward(self, x):
        b, c, _, _ = x.size()
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.sse(x) * x
        y = self.silu(x1 + x2 + x3)
        return y


class ParallelPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        self.softmax_channel = nn.Softmax(1)
        self.softmax_spatial = nn.Softmax(-1)
        self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()
        self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.size()
        channel_wv = self.ch_wv(x)
        channel_wq = self.ch_wq(x)
        channel_wv = channel_wv.reshape(b, c // 2, -1)
        channel_wq = channel_wq.reshape(b, -1, 1)
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)
        channel_weight = self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))).permute(0, 2, 1).reshape(b, c, 1, 1)
        channel_out = channel_weight * x
        spatial_wv = self.sp_wv(x)
        spatial_wq = self.sp_wq(x)
        spatial_wq = self.agp(spatial_wq)
        spatial_wv = spatial_wv.reshape(b, c // 2, -1)
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)
        spatial_wq = self.softmax_spatial(spatial_wq)
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))
        spatial_out = spatial_weight * x
        out = spatial_out + channel_out
        return out


class SequentialPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        self.softmax_channel = nn.Softmax(1)
        self.softmax_spatial = nn.Softmax(-1)
        self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()
        self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.size()
        channel_wv = self.ch_wv(x)
        channel_wq = self.ch_wq(x)
        channel_wv = channel_wv.reshape(b, c // 2, -1)
        channel_wq = channel_wq.reshape(b, -1, 1)
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)
        channel_weight = self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))).permute(0, 2, 1).reshape(b, c, 1, 1)
        channel_out = channel_weight * x
        spatial_wv = self.sp_wv(channel_out)
        spatial_wq = self.sp_wq(channel_out)
        spatial_wq = self.agp(spatial_wq)
        spatial_wv = spatial_wv.reshape(b, c // 2, -1)
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)
        spatial_wq = self.softmax_spatial(spatial_wq)
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))
        spatial_out = spatial_weight * channel_out
        return spatial_out


class ResidualAttention(nn.Module):

    def __init__(self, channel=512, num_class=1000, la=0.2):
        super().__init__()
        self.la = la
        self.fc = nn.Conv2d(in_channels=channel, out_channels=num_class, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        y_raw = self.fc(x).flatten(2)
        y_avg = torch.mean(y_raw, dim=2)
        y_max = torch.max(y_raw, dim=2)[0]
        score = y_avg + self.la * y_max
        return score


class SplitAttention(nn.Module):

    def __init__(self, channel=512, k=3):
        super().__init__()
        self.channel = channel
        self.k = k
        self.mlp1 = nn.Linear(channel, channel, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, x_all):
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)
        a = torch.sum(torch.sum(x_all, 1), 1)
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))
        hat_a = hat_a.reshape(b, self.k, c)
        bar_a = self.softmax(hat_a)
        attention = bar_a.unsqueeze(-2)
        out = attention * x_all
        out = torch.sum(out, 1).reshape(b, h, w, c)
        return out


def spatial_shift1(x):
    b, w, h, c = x.size()
    x[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
    x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
    x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
    x[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
    return x


def spatial_shift2(x):
    b, w, h, c = x.size()
    x[:, :, 1:, :c // 4] = x[:, :, :h - 1, :c // 4]
    x[:, :, :h - 1, c // 4:c // 2] = x[:, :, 1:, c // 4:c // 2]
    x[:, 1:, :, c // 2:c * 3 // 4] = x[:, :w - 1, :, c // 2:c * 3 // 4]
    x[:, :w - 1, :, 3 * c // 4:] = x[:, 1:, :, 3 * c // 4:]
    return x


class S2Attention(nn.Module):

    def __init__(self, channels=512):
        super().__init__()
        self.mlp1 = nn.Linear(channels, channels * 3)
        self.mlp2 = nn.Linear(channels, channels)
        self.split_attention = SplitAttention()

    def forward(self, x):
        b, c, w, h = x.size()
        x = x.permute(0, 2, 3, 1)
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:, :, :, :c])
        x2 = spatial_shift2(x[:, :, :, c:c * 2])
        x3 = x[:, :, :, c * 2:]
        x_all = torch.stack([x1, x2, x3], 1)
        a = self.split_attention(x_all)
        x = self.mlp2(a)
        x = x.permute(0, 3, 1, 2)
        return x


class SEAttention(nn.Module):

    def __init__(self, channel, ratio=16):
        super(SEAttention, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(nn.Linear(in_features=channel, out_features=channel // ratio), nn.ReLU(inplace=True), nn.Linear(in_features=channel // ratio, out_features=channel), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)


class cSE_Module(nn.Module):

    def __init__(self, channel, ratio=16):
        super(cSE_Module, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(nn.Linear(in_features=channel, out_features=channel // ratio), nn.ReLU(inplace=True), nn.Linear(in_features=channel // ratio, out_features=channel), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)


class sSE_Module(nn.Module):

    def __init__(self, channel):
        super(sSE_Module, self).__init__()
        self.spatial_excitation = nn.Sequential(nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())

    def forward(self, x):
        z = self.spatial_excitation(x)
        return x * z.expand_as(x)


class scSE_Module(nn.Module):

    def __init__(self, channel, ratio=16):
        super(scSE_Module, self).__init__()
        self.cSE = cSE_Module(channel, ratio)
        self.sSE = sSE_Module(channel)

    def forward(self, x):
        return self.cSE(x) + self.sSE(x)


class SpatialGroupEnhance(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.sig = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b * self.groups, -1, h, w)
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-05
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x


class SKAttention(nn.Module):

    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(nn.Sequential(OrderedDict([('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)), ('bn', nn.BatchNorm2d(channel)), ('relu', nn.ReLU())])))
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)
        U = sum(conv_outs)
        S = U.mean(-1).mean(-1)
        Z = self.fc(S)
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))
        attention_weughts = torch.stack(weights, 0)
        attention_weughts = self.softmax(attention_weughts)
        V = (attention_weughts * feats).sum(0)
        return V


class ShuffleAttention(nn.Module):

    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b * self.G, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)
        x_channel = self.avg_pool(x_0)
        x_channel = self.cweight * x_channel + self.cbias
        x_channel = x_0 * self.sigmoid(x_channel)
        x_spatial = self.gn(x_1)
        x_spatial = self.sweight * x_spatial + self.sbias
        x_spatial = x_1 * self.sigmoid(x_spatial)
        out = torch.cat([x_channel, x_spatial], dim=1)
        out = out.contiguous().view(b, -1, h, w)
        out = self.channel_shuffle(out, 2)
        return out


class SimplifiedScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention
    """

    def __init__(self, d_model, h, dropout=0.1):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(SimplifiedScaledDotProductAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h
        self.fc_o = nn.Linear(h * self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        """
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        """
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = queries.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = keys.view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = values.view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)
        return out


class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):

    def __init__(self):
        super(SpatialGate, self).__init__()
        self.channel_pool = ChannelPool()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3), nn.BatchNorm2d(1))
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(self.channel_pool(x))
        return out * self.sigmod(out)


class TripletAttention(nn.Module):

    def __init__(self, spatial=True):
        super(TripletAttention, self).__init__()
        self.spatial = spatial
        self.height_gate = SpatialGate()
        self.width_gate = SpatialGate()
        if self.spatial:
            self.spatial_gate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.height_gate(x_perm1)
        x_out1 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.width_gate(x_perm2)
        x_out2 = x_out2.permute(0, 3, 2, 1).contiguous()
        if self.spatial:
            x_out3 = self.spatial_gate(x)
            return 1 / 3 * (x_out1 + x_out2 + x_out3)
        else:
            return 1 / 2 * (x_out1 + x_out2)


def XNorm(x, gamma):
    norm_tensor = torch.norm(x, 2, -1, True)
    return x * gamma / norm_tensor


class UFOAttention(nn.Module):
    """
    Scaled dot-product attention
    """

    def __init__(self, d_model, d_k, d_v, h, dropout=0.1):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(UFOAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.randn((1, h, 1, 1)))
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)
        kv = torch.matmul(k, v)
        kv_norm = XNorm(kv, self.gamma)
        q_norm = XNorm(q, self.gamma)
        out = torch.matmul(q_norm, kv_norm).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)
        return out


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, dim=256, n_blk=3, activ='relu'):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, dim), nn.ReLU(inplace=True)]
        for _ in range(n_blk - 2):
            layers += [nn.Linear(dim, dim), nn.ReLU(inplace=True)]
        layers += [nn.Linear(dim, output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class WeightedPermuteMLP(nn.Module):

    def __init__(self, dim, seg_dim=8, qkv_bias=False, proj_drop=0.0):
        super().__init__()
        self.seg_dim = seg_dim
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp_h = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp_w = nn.Linear(dim, dim, bias=qkv_bias)
        self.reweighting = MLP(dim, dim // 4, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        c_embed = self.mlp_c(x)
        S = C // self.seg_dim
        h_embed = x.reshape(B, H, W, self.seg_dim, S).permute(0, 3, 2, 1, 4).reshape(B, self.seg_dim, W, H * S)
        h_embed = self.mlp_h(h_embed).reshape(B, self.seg_dim, W, H, S).permute(0, 3, 2, 1, 4).reshape(B, H, W, C)
        w_embed = x.reshape(B, H, W, self.seg_dim, S).permute(0, 3, 1, 2, 4).reshape(B, self.seg_dim, H, W * S)
        w_embed = self.mlp_w(w_embed).reshape(B, self.seg_dim, H, W, S).permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        weight = (c_embed + h_embed + w_embed).permute(0, 3, 1, 2).flatten(2).mean(2)
        weight = self.reweighting(weight).reshape(B, C, 3).permute(2, 0, 1).softmax(0).unsqueeze(2).unsqueeze(2)
        x = c_embed * weight[0] + w_embed * weight[1] + h_embed * weight[2]
        x = self.proj_drop(self.proj(x))
        return x


class residual_attention(nn.Module):

    def __init__(self, block, in_channels, out_channels, size1, size2, size3):
        super(residual_attention, self).__init__()
        self.first_residual_blocks = block(in_channels, out_channels)
        self.trunk_branches = nn.Sequential(block(in_channels, out_channels), block(in_channels, out_channels))
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = block(in_channels, out_channels)
        self.skip1_connection_residual_block = block(in_channels, out_channels)
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax2_blocks = block(in_channels, out_channels)
        self.skip2_connection_residual_block = block(in_channels, out_channels)
        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax3_blocks = nn.Sequential(block(in_channels, out_channels), block(in_channels, out_channels))
        self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)
        self.softmax4_blocks = block(in_channels, out_channels)
        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)
        self.softmax5_blocks = block(in_channels, out_channels)
        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)
        self.softmax6_blocks = nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False), nn.Sigmoid())
        self.w1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(1.0)
        self.w2.data.fill_(1.0)
        self.last_blocks = block(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
        out_mpool3 = self.mpool3(out_softmax2)
        out_softmax3 = self.softmax3_blocks(out_mpool3)
        out_interp3 = self.interpolation3(out_softmax3)
        out = out_interp3 + out_skip2_connection
        out_softmax4 = self.softmax4_blocks(out)
        out_interp2 = self.interpolation2(out_softmax4)
        out = out_interp2 + out_skip1_connection
        out_softmax5 = self.softmax5_blocks(out)
        out_interp1 = self.interpolation1(out_softmax5)
        out_softmax6 = self.softmax6_blocks(out_interp1)
        out = out_softmax6 * out_trunk
        out = self.w1 * out + self.w2 * out_trunk
        out_last = self.last_blocks(out)
        return out_last


class _TransitionLayer(nn.Module):

    def __init__(self, inplace, plance):
        super(_TransitionLayer, self).__init__()
        self.transition_layer = nn.Sequential(nn.BatchNorm2d(inplace), nn.ReLU(inplace=True), nn.Conv2d(in_channels=inplace, out_channels=plance, kernel_size=1, stride=1, padding=0, bias=False), nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.transition_layer(x)


class InceptionV1Module(nn.Module):

    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV1Module, self).__init__()
        self.branch1_conv = ConvBNReLU(in_channels=in_channels, out_channels=out_channels1, kernel_size=1)
        self.branch2_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1)
        self.branch2_conv2 = ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=3)
        self.branch3_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels3reduce, kernel_size=1)
        self.branch3_conv2 = ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=5)
        self.branch4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1)

    def forward(self, x):
        out1 = self.branch1_conv(x)
        out2 = self.branch2_conv2(self.branch2_conv1(x))
        out3 = self.branch3_conv2(self.branch3_conv1(x))
        out4 = self.branch4_conv1(self.branch4_pool(x))
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionV1(nn.Module):

    def __init__(self, num_classes=1000, stage='train'):
        super(InceptionV1, self).__init__()
        self.stage = stage
        self.block1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.MaxPool2d(kernel_size=3, stride=2, padding=1), nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1), nn.BatchNorm2d(64))
        self.block2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(192), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block3 = nn.Sequential(InceptionV1Module(in_channels=192, out_channels1=64, out_channels2reduce=96, out_channels2=128, out_channels3reduce=16, out_channels3=32, out_channels4=32), InceptionV1Module(in_channels=256, out_channels1=128, out_channels2reduce=128, out_channels2=192, out_channels3reduce=32, out_channels3=96, out_channels4=64), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block4_1 = InceptionV1Module(in_channels=480, out_channels1=192, out_channels2reduce=96, out_channels2=208, out_channels3reduce=16, out_channels3=48, out_channels4=64)
        if self.stage == 'train':
            self.aux_logits1 = InceptionAux(in_channels=512, out_channels=num_classes)
        self.block4_2 = nn.Sequential(InceptionV1Module(in_channels=512, out_channels1=160, out_channels2reduce=112, out_channels2=224, out_channels3reduce=24, out_channels3=64, out_channels4=64), InceptionV1Module(in_channels=512, out_channels1=128, out_channels2reduce=128, out_channels2=256, out_channels3reduce=24, out_channels3=64, out_channels4=64), InceptionV1Module(in_channels=512, out_channels1=112, out_channels2reduce=144, out_channels2=288, out_channels3reduce=32, out_channels3=64, out_channels4=64))
        if self.stage == 'train':
            self.aux_logits2 = InceptionAux(in_channels=528, out_channels=num_classes)
        self.block4_3 = nn.Sequential(InceptionV1Module(in_channels=528, out_channels1=256, out_channels2reduce=160, out_channels2=320, out_channels3reduce=32, out_channels3=128, out_channels4=128), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block5 = nn.Sequential(InceptionV1Module(in_channels=832, out_channels1=256, out_channels2reduce=160, out_channels2=320, out_channels3reduce=32, out_channels3=128, out_channels4=128), InceptionV1Module(in_channels=832, out_channels1=384, out_channels2reduce=192, out_channels2=384, out_channels3reduce=48, out_channels3=128, out_channels4=128))
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        aux1 = x = self.block4_1(x)
        aux2 = x = self.block4_2(x)
        x = self.block4_3(x)
        out = self.block5(x)
        out = self.avgpool(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.stage == 'train':
            aux1 = self.aux_logits1(aux1)
            aux2 = self.aux_logits2(aux2)
            return aux1, aux2, out
        else:
            return out


class InceptionV2ModuleA(nn.Module):

    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2ModuleA, self).__init__()
        self.branch1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels1, kernel_size=1)
        self.branch2 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1), ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=3, padding=1))
        self.branch3 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels3reduce, kernel_size=1), ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=3, padding=1), ConvBNReLU(in_channels=out_channels3, out_channels=out_channels3, kernel_size=3, padding=1))
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1), ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1))

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


def ConvBNReLUFactorization(in_channels, out_channels, kernel_sizes, paddings):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_sizes, stride=1, padding=paddings), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


class InceptionV2ModuleB(nn.Module):

    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2ModuleB, self).__init__()
        self.branch1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels1, kernel_size=1)
        self.branch2 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1), ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2reduce, kernel_sizes=[1, 3], paddings=[0, 1]), ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=[3, 1], paddings=[1, 0]))
        self.branch3 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels3reduce, kernel_size=1), ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce, kernel_sizes=[1, 3], paddings=[0, 1]), ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce, kernel_sizes=[3, 1], paddings=[1, 0]), ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce, kernel_sizes=[1, 3], paddings=[0, 1]), ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_sizes=[3, 1], paddings=[1, 0]))
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1), ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1))

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionV2ModuleC(nn.Module):

    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2ModuleC, self).__init__()
        self.branch1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels1, kernel_size=1)
        self.branch2_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1)
        self.branch2_conv2a = ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=[1, 3], paddings=[0, 1])
        self.branch2_conv2b = ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=[3, 1], paddings=[1, 0])
        self.branch3_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels3reduce, kernel_size=1)
        self.branch3_conv2 = ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=3, stride=1, padding=1)
        self.branch3_conv3a = ConvBNReLUFactorization(in_channels=out_channels3, out_channels=out_channels3, kernel_sizes=[3, 1], paddings=[1, 0])
        self.branch3_conv3b = ConvBNReLUFactorization(in_channels=out_channels3, out_channels=out_channels3, kernel_sizes=[1, 3], paddings=[0, 1])
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1), ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1))

    def forward(self, x):
        out1 = self.branch1(x)
        x2 = self.branch2_conv1(x)
        out2 = torch.cat([self.branch2_conv2a(x2), self.branch2_conv2b(x2)], dim=1)
        x3 = self.branch3_conv2(self.branch3_conv1(x))
        out3 = torch.cat([self.branch3_conv3a(x3), self.branch3_conv3b(x3)], dim=1)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionV3ModuleD(nn.Module):

    def __init__(self, in_channels, out_channels1reduce, out_channels1, out_channels2reduce, out_channels2):
        super(InceptionV3ModuleD, self).__init__()
        self.branch1 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels1reduce, kernel_size=1), ConvBNReLU(in_channels=out_channels1reduce, out_channels=out_channels1, kernel_size=3, stride=2))
        self.branch2 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1), ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=3, stride=1, padding=1), ConvBNReLU(in_channels=out_channels2, out_channels=out_channels2, kernel_size=3, stride=2))
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out


class InceptionV2(nn.Module):

    def __init__(self, num_classes=1000, stage='train'):
        super(InceptionV2, self).__init__()
        self.stage = stage
        self.block1 = nn.Sequential(ConvBNReLU(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block2 = nn.Sequential(ConvBNReLU(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block3 = nn.Sequential(InceptionV2ModuleA(in_channels=192, out_channels1=64, out_channels2reduce=64, out_channels2=64, out_channels3reduce=64, out_channels3=96, out_channels4=32), InceptionV2ModuleA(in_channels=256, out_channels1=64, out_channels2reduce=64, out_channels2=96, out_channels3reduce=64, out_channels3=96, out_channels4=64), InceptionV3ModuleD(in_channels=320, out_channels1reduce=128, out_channels1=160, out_channels2reduce=64, out_channels2=96))
        self.block4 = nn.Sequential(InceptionV2ModuleB(in_channels=576, out_channels1=224, out_channels2reduce=64, out_channels2=96, out_channels3reduce=96, out_channels3=128, out_channels4=128), InceptionV2ModuleB(in_channels=576, out_channels1=192, out_channels2reduce=96, out_channels2=128, out_channels3reduce=96, out_channels3=128, out_channels4=128), InceptionV2ModuleB(in_channels=576, out_channels1=160, out_channels2reduce=128, out_channels2=160, out_channels3reduce=128, out_channels3=128, out_channels4=128), InceptionV2ModuleB(in_channels=576, out_channels1=96, out_channels2reduce=128, out_channels2=192, out_channels3reduce=160, out_channels3=160, out_channels4=128), InceptionV3ModuleD(in_channels=576, out_channels1reduce=128, out_channels1=192, out_channels2reduce=192, out_channels2=256))
        self.block5 = nn.Sequential(InceptionV2ModuleC(in_channels=1024, out_channels1=352, out_channels2reduce=192, out_channels2=160, out_channels3reduce=160, out_channels3=112, out_channels4=128), InceptionV2ModuleC(in_channels=1024, out_channels1=352, out_channels2reduce=192, out_channels2=160, out_channels3reduce=192, out_channels3=112, out_channels4=128))
        self.max_pool = nn.MaxPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out


class InceptionV3ModuleA(nn.Module):

    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV3ModuleA, self).__init__()
        self.branch1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels1, kernel_size=1)
        self.branch2 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1), ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=5, padding=2))
        self.branch3 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels3reduce, kernel_size=1), ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=3, padding=1), ConvBNReLU(in_channels=out_channels3, out_channels=out_channels3, kernel_size=3, padding=1))
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1), ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1))

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionV3ModuleB(nn.Module):

    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV3ModuleB, self).__init__()
        self.branch1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels1, kernel_size=1)
        self.branch2 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1), ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2reduce, kernel_sizes=[1, 7], paddings=[0, 3]), ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=[7, 1], paddings=[3, 0]))
        self.branch3 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels3reduce, kernel_size=1), ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce, kernel_sizes=[1, 7], paddings=[0, 3]), ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce, kernel_sizes=[7, 1], paddings=[3, 0]), ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce, kernel_sizes=[1, 7], paddings=[0, 3]), ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_sizes=[7, 1], paddings=[3, 0]))
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1), ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1))

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionV3ModuleC(nn.Module):

    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV3ModuleC, self).__init__()
        self.branch1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels1, kernel_size=1)
        self.branch2_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1)
        self.branch2_conv2a = ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=[1, 3], paddings=[0, 1])
        self.branch2_conv2b = ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=[3, 1], paddings=[1, 0])
        self.branch3_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels3reduce, kernel_size=1)
        self.branch3_conv2 = ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=3, stride=1, padding=1)
        self.branch3_conv3a = ConvBNReLUFactorization(in_channels=out_channels3, out_channels=out_channels3, kernel_sizes=[3, 1], paddings=[1, 0])
        self.branch3_conv3b = ConvBNReLUFactorization(in_channels=out_channels3, out_channels=out_channels3, kernel_sizes=[1, 3], paddings=[0, 1])
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1), ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1))

    def forward(self, x):
        out1 = self.branch1(x)
        x2 = self.branch2_conv1(x)
        out2 = torch.cat([self.branch2_conv2a(x2), self.branch2_conv2b(x2)], dim=1)
        x3 = self.branch3_conv2(self.branch3_conv1(x))
        out3 = torch.cat([self.branch3_conv3a(x3), self.branch3_conv3b(x3)], dim=1)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionV3ModuleE(nn.Module):

    def __init__(self, in_channels, out_channels1reduce, out_channels1, out_channels2reduce, out_channels2):
        super(InceptionV3ModuleE, self).__init__()
        self.branch1 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels1reduce, kernel_size=1), ConvBNReLU(in_channels=out_channels1reduce, out_channels=out_channels1, kernel_size=3, stride=2))
        self.branch2 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1), ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2reduce, kernel_sizes=[1, 7], paddings=[0, 3]), ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2reduce, kernel_sizes=[7, 1], paddings=[3, 0]), ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=3, stride=2))
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out


class InceptionV3(nn.Module):

    def __init__(self, num_classes=1000, stage='train'):
        super(InceptionV3, self).__init__()
        self.stage = stage
        self.block1 = nn.Sequential(ConvBNReLU(in_channels=3, out_channels=32, kernel_size=3, stride=2), ConvBNReLU(in_channels=32, out_channels=32, kernel_size=3, stride=1), ConvBNReLU(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = nn.Sequential(ConvBNReLU(in_channels=64, out_channels=80, kernel_size=3, stride=1), ConvBNReLU(in_channels=80, out_channels=192, kernel_size=3, stride=1, padding=1), nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(InceptionV3ModuleA(in_channels=192, out_channels1=64, out_channels2reduce=48, out_channels2=64, out_channels3reduce=64, out_channels3=96, out_channels4=32), InceptionV3ModuleA(in_channels=256, out_channels1=64, out_channels2reduce=48, out_channels2=64, out_channels3reduce=64, out_channels3=96, out_channels4=64), InceptionV3ModuleA(in_channels=288, out_channels1=64, out_channels2reduce=48, out_channels2=64, out_channels3reduce=64, out_channels3=96, out_channels4=64))
        self.block4 = nn.Sequential(InceptionV3ModuleD(in_channels=288, out_channels1reduce=384, out_channels1=384, out_channels2reduce=64, out_channels2=96), InceptionV3ModuleB(in_channels=768, out_channels1=192, out_channels2reduce=128, out_channels2=192, out_channels3reduce=128, out_channels3=192, out_channels4=192), InceptionV3ModuleB(in_channels=768, out_channels1=192, out_channels2reduce=160, out_channels2=192, out_channels3reduce=160, out_channels3=192, out_channels4=192), InceptionV3ModuleB(in_channels=768, out_channels1=192, out_channels2reduce=160, out_channels2=192, out_channels3reduce=160, out_channels3=192, out_channels4=192), InceptionV3ModuleB(in_channels=768, out_channels1=192, out_channels2reduce=192, out_channels2=192, out_channels3reduce=192, out_channels3=192, out_channels4=192))
        if self.stage == 'train':
            self.aux_logits = InceptionAux(in_channels=768, out_channels=num_classes)
        self.block5 = nn.Sequential(InceptionV3ModuleE(in_channels=768, out_channels1reduce=192, out_channels1=320, out_channels2reduce=192, out_channels2=192), InceptionV3ModuleC(in_channels=1280, out_channels1=320, out_channels2reduce=384, out_channels2=384, out_channels3reduce=448, out_channels3=384, out_channels4=192), InceptionV3ModuleC(in_channels=2048, out_channels1=320, out_channels2reduce=384, out_channels2=384, out_channels3reduce=448, out_channels3=384, out_channels4=192))
        self.max_pool = nn.MaxPool2d(kernel_size=8, stride=1)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        aux = x = self.block4(x)
        x = self.block5(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        if self.stage == 'train':
            aux = self.aux_logits(aux)
            return aux, out
        else:
            return out


class Inception_A(nn.Module):

    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(384, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(384, 96, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Inception_B(nn.Module):

    def __init__(self):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1), BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(224, 256, kernel_size=(7, 1), stride=1, padding=(3, 0)))
        self.branch2 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1), BasicConv2d(192, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(224, 224, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(224, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(1024, 128, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Inception_C(nn.Module):

    def __init__(self):
        super(Inception_C, self).__init__()
        self.branch0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)
        self.branch1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(384, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch1_1b = BasicConv2d(384, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(384, 448, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch2_2 = BasicConv2d(448, 512, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3a = BasicConv2d(512, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3b = BasicConv2d(512, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(1536, 256, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)
        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_3a(nn.Module):

    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_4a(nn.Module):

    def __init__(self):
        super(Mixed_4a, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1))
        self.branch1 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1, stride=1), BasicConv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(64, 96, kernel_size=(3, 3), stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_5a(nn.Module):

    def __init__(self):
        super(Mixed_5a, self).__init__()
        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class Reduction_A(nn.Module):

    def __init__(self):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv2d(384, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(BasicConv2d(384, 192, kernel_size=1, stride=1), BasicConv2d(192, 224, kernel_size=3, stride=1, padding=1), BasicConv2d(224, 256, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Reduction_B(nn.Module):

    def __init__(self):
        super(Reduction_B, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1), BasicConv2d(192, 192, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(1024, 256, kernel_size=1, stride=1), BasicConv2d(256, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(256, 320, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(320, 320, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class InceptionV4(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionV4, self).__init__()
        self.input_space = None
        self.input_size = 299, 299, 3
        self.mean = None
        self.std = None
        self.features = nn.Sequential(BasicConv2d(3, 32, kernel_size=3, stride=2), BasicConv2d(32, 32, kernel_size=3, stride=1), BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1), Mixed_3a(), Mixed_4a(), Mixed_5a(), Inception_A(), Inception_A(), Inception_A(), Inception_A(), Reduction_A(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Reduction_B(), Inception_C(), Inception_C(), Inception_C())
        self.last_linear = nn.Linear(1536, num_classes)

    def logits(self, features):
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class _NonLocalBlockND(nn.Module):

    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()
        assert dimension in [1, 2, 3]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=2)
            bn = nn.BatchNorm1d
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        if bn_layer:
            self.W = nn.Sequential(conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = max_pool_layer

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock2D(_NonLocalBlockND):

    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels, inter_channels=inter_channels, dimension=2, sub_sample=sub_sample, bn_layer=bn_layer)


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.nl_1 = NONLocalBlock2D(in_channels=32)
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.nl_2 = NONLocalBlock2D(in_channels=64)
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.fc = nn.Sequential(nn.Linear(in_features=128 * 3 * 3, out_features=256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(in_features=256, out_features=10))

    def forward(self, x):
        batch_size = x.size(0)
        feature_1 = self.conv_1(x)
        nl_feature_1 = self.nl_1(feature_1)
        feature_2 = self.conv_2(nl_feature_1)
        nl_feature_2 = self.nl_2(feature_2)
        output = self.conv_3(nl_feature_2).view(batch_size, -1)
        output = self.fc(output)
        return output

    def forward_with_nl_map(self, x):
        batch_size = x.size(0)
        feature_1 = self.conv_1(x)
        nl_feature_1, nl_map_1 = self.nl_1(feature_1, return_nl_map=True)
        feature_2 = self.conv_2(nl_feature_1)
        nl_feature_2, nl_map_2 = self.nl_2(feature_2, return_nl_map=True)
        output = self.conv_3(nl_feature_2).view(batch_size, -1)
        output = self.fc(output)
        return output, [nl_map_1, nl_map_2]


class FPAModule1(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(FPAModule1, self).__init__()
        self.branch1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), Conv2dBnRelu(in_ch, out_ch, kelnel_size=1, stride=1, padding=0))
        self.mid = nn.Sequential(Conv2dBnRelu(in_ch, out_ch, kelnel_size=1, stride=1, padding=0))
        self.pyramid1 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), Conv2dBnRelu(in_ch, 1, kelnel_size=7, stride=1, padding=3))
        self.pyramid2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), Conv2dBnRelu(1, 1, kelnel_size=5, stride=1, padding=2))
        self.pyramid3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), Conv2dBnRelu(1, 1, kelnel_size=3, stride=1, padding=1))
        self.conv1 = Conv2dBnRelu(1, 1, kelnel_size=7, stride=1, padding=3)
        self.conv2 = Conv2dBnRelu(1, 1, kelnel_size=5, stride=1, padding=2)
        self.conv3 = Conv2dBnRelu(1, 1, kelnel_size=3, stride=1, padding=1)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        b1 = self.branch1(x)
        b1 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(b1)
        mid = self.mid(x)
        x1 = self.pyramid1(x)
        x2 = self.pyramid2(x1)
        x3 = self.pyramid3(x2)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x3 = nn.Upsample(size=(h // 4, w // 4), mode='bilinear', align_corners=True)(x3)
        x = x3 + x2
        x = nn.Upsample(size=(h // 2, w // 2), mode='bilinear', align_corners=True)(x)
        x = x + x1
        x = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(x)
        x = torch.mul(x, mid)
        x += b1
        return x


class FPAModule2(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(FPAModule2, self).__init__()
        self.branch1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), Conv2dBnRelu(in_ch, out_ch, kelnel_size=1, stride=1, padding=0))
        self.mid = nn.Sequential(Conv2dBnRelu(in_ch, out_ch, kelnel_size=1, stride=1, padding=0))
        self.pyramid1 = nn.Sequential(Conv2dBnRelu(in_ch, 1, kelnel_size=7, stride=2, padding=3))
        self.pyramid2 = nn.Sequential(Conv2dBnRelu(1, 1, kelnel_size=5, stride=2, padding=2))
        self.pyramid3 = nn.Sequential(Conv2dBnRelu(1, 1, kelnel_size=3, stride=2, padding=1))
        self.conv1 = Conv2dBnRelu(1, 1, kelnel_size=7, stride=1, padding=3)
        self.conv2 = Conv2dBnRelu(1, 1, kelnel_size=5, stride=1, padding=2)
        self.conv3 = Conv2dBnRelu(1, 1, kelnel_size=3, stride=1, padding=1)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        b1 = self.branch1(x)
        b1 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(b1)
        mid = self.mid(x)
        x1 = self.pyramid1(x)
        x2 = self.pyramid2(x1)
        x3 = self.pyramid3(x2)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x3 = nn.Upsample(size=(h // 4, w // 4), mode='bilinear', align_corners=True)(x3)
        x = x3 + x2
        x = nn.Upsample(size=(h // 2, w // 2), mode='bilinear', align_corners=True)(x)
        x = x + x1
        x = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(x)
        x = torch.mul(x, mid)
        x += b1
        return x


class GAUModule(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(GAUModule, self).__init__()
        self.low_conv = Conv2dBnRelu(in_ch, out_ch, kelnel_size=3, stride=1, padding=1)
        self.high_conv = nn.Sequential(nn.AdaptiveAvgPool2d(1), BN_Conv2d(out_ch, out_ch, kelnel_size=1, stride=1, padding=0), nn.Sigmoid())

    def forward(self, lowx, highx):
        h, w = lowx.size(2), lowx.size(3)
        highx = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(highx)
        lowx = self.low_conv(lowx)
        h_up = self.high_conv(highx)
        z = lowx * h_up
        return z + highx


class PAN(nn.Module):

    def __init__(self, block, num_blocks, num_clssess=43):
        super(PAN, self).__init__()
        self.in_planes = 64
        self.conv1 = Conv2dBnRelu(in_ch=3, out_ch=64, kelnel_size=7, stride=2, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fpa = FPAModule2(512 * block.expansion, num_clssess)
        self.gau1 = GAUModule(512 * block.expansion // 2, num_clssess)
        self.gau2 = GAUModule(512 * block.expansion // 4, num_clssess)
        self.gau3 = GAUModule(512 * block.expansion // 8, num_clssess)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride=stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.maxpool(self.conv1(x))
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.fpa(x4)
        x3 = self.gau1(x3, x5)
        x2 = self.gau2(x2, x3)
        x1 = self.gau3(x1, x2)
        out = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(x1)
        return out


class SE(nn.Module):
    """
    input: (bt_size, C, H, W)
    output:
        alpha:(C,1)
        att_output:(bt_size, C, H, W)
    """

    def __init__(self, inplanes):
        super(SE, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        if inplanes % 16 == 0:
            self.conv1 = nn.Conv2d(inplanes, int(inplanes / 16), kernel_size=1, stride=1)
            self.conv2 = nn.Conv2d(int(inplanes / 16), inplanes, kernel_size=1, stride=1)
        else:
            self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1)
            self.conv2 = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return x * out


class ResNeXt_Block(nn.Module):
    """
    ResNeXt block with group convolutions
    """

    def __init__(self, in_chnls, cardinality, group_depth, stride, is_se=False):
        super(ResNeXt_Block, self).__init__()
        self.is_se = is_se
        self.group_chnls = cardinality * group_depth
        self.conv1 = BN_Conv2d(in_chnls, self.group_chnls, 1, stride=1, padding=0)
        self.conv2 = BN_Conv2d(self.group_chnls, self.group_chnls, 3, stride=stride, padding=1, groups=cardinality)
        self.conv3 = nn.Conv2d(self.group_chnls, self.group_chnls * 2, 1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(self.group_chnls * 2)
        if self.is_se:
            self.se = SE(self.group_chnls * 2, 16)
        self.short_cut = nn.Sequential(nn.Conv2d(in_chnls, self.group_chnls * 2, 1, stride, 0, bias=False), nn.BatchNorm2d(self.group_chnls * 2))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(self.conv3(out))
        if self.is_se:
            coefficient = self.se(out)
            out *= coefficient
        out += self.short_cut(x)
        return F.relu(out)


class ResNeXt(nn.Module):
    """
    ResNeXt builder
    """

    def __init__(self, layers: 'object', cardinality, group_depth, num_classes, is_se=False) ->object:
        super(ResNeXt, self).__init__()
        self.is_se = is_se
        self.cardinality = cardinality
        self.channels = 64
        self.conv1 = BN_Conv2d(3, self.channels, 3, stride=1, padding=1)
        d1 = group_depth
        self.conv2 = self.___make_layers(d1, layers[0], stride=1)
        d2 = d1 * 2
        self.conv3 = self.___make_layers(d2, layers[1], stride=2)
        d3 = d2 * 2
        self.conv4 = self.___make_layers(d3, layers[2], stride=2)
        d4 = d3 * 2
        self.conv5 = self.___make_layers(d4, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.channels, num_classes)

    def ___make_layers(self, d, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResNeXt_Block(self.channels, self.cardinality, d, stride, self.is_se))
            self.channels = self.cardinality * d * 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class VGGNet(nn.Module):

    def __init__(self, block_nums, num_classes=1000):
        super(VGGNet, self).__init__()
        self.stage1 = self._make_layers(in_channels=3, out_channels=64, block_num=block_nums[0])
        self.stage2 = self._make_layers(in_channels=64, out_channels=128, block_num=block_nums[1])
        self.stage3 = self._make_layers(in_channels=128, out_channels=256, block_num=block_nums[2])
        self.stage4 = self._make_layers(in_channels=256, out_channels=512, block_num=block_nums[3])
        self.stage5 = self._make_layers(in_channels=512, out_channels=512, block_num=block_nums[4])
        self.classifier = nn.Sequential(nn.Linear(in_features=512 * 7 * 7, out_features=4096), nn.Dropout(p=0.2), nn.Linear(in_features=4096, out_features=4096), nn.Dropout(p=0.2), nn.Linear(in_features=4096, out_features=num_classes))
        self._init_params()

    def _make_layers(self, in_channels, out_channels, block_num):
        layers = []
        layers.append(Conv3x3BNReLU(in_channels, out_channels))
        for i in range(1, block_num):
            layers.append(Conv3x3BNReLU(out_channels, out_channels))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False))
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class MFPA(nn.Module):

    def __init__(self, planes):
        super(MFPA, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention(planes)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class SKConv(nn.Module):

    def __init__(self, features, M=2, G=32, r=16, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i, groups=G, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=False)))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(d), nn.ReLU(inplace=False))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Conv2d(d, features, kernel_size=1, stride=1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        feats = [conv(x) for conv in self.convs]
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])
        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(feats * attention_vectors, dim=1)
        return feats_V


class SF(nn.Module):

    def __init__(self, channels, M=2):
        super(SF, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(M):
            self.convs.append(nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3 + 2 * i, stride=1, padding=1 + i, dilation=1, groups=32, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=False)))
        self.eca = ECA(channels)

    def forward(self, input):
        batch_size = input.size(0)
        output = [conv(input) for conv in self.convs]
        U = reduce(lambda x, y: x + y, output)
        return self.eca(U)


class STN(nn.Module):
    """Constructs a ECA module.
    Args:
        inplanes: Number of channels of the input feature map

    """

    def __init__(self, in_planes, places):
        super(STN, self).__init__()
        self.localization = nn.Sequential(nn.Conv2d(in_planes, places, kernel_size=7), nn.MaxPool2d(2, stride=2), nn.ReLU(True), nn.Conv2d(places, places, kernel_size=5), nn.MaxPool2d(2, stride=1), nn.ReLU(True))
        self.fc_loc = nn.Sequential(nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2))
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(x.size()[0], -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x


class softmax_layer(nn.Module):
    """Constructs a ECA module.
        Args:
            input: [B,K,F]
           output: [B,F]
        """

    def __init__(self, dim=512):
        super(softmax_layer, self).__init__()
        self.dim = dim
        self.w_omega = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.u_omega = nn.Parameter(torch.Tensor(self.dim, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, x):
        u = torch.tanh(torch.matmul(x, self.w_omega))
        att = torch.matmul(u, self.u_omega)
        att_score = F.softmax(att, dim=1)
        scored_x = x * att_score
        outs = torch.sum(scored_x, dim=1)
        return outs


class BN_Conv2d_Leaky(nn.Module):
    """
    BN_CONV_LeakyRELU
    """

    def __init__(self, in_channels: 'object', out_channels: 'object', kernel_size: 'object', stride: 'object', padding: 'object', dilation=1, groups=1, bias=False) ->object:
        super(BN_Conv2d_Leaky, self).__init__()
        self.seq = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return F.leaky_relu(self.seq(x))


class Mish(nn.Module):

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class BN_Conv_Mish(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=False):
        super(BN_Conv_Mish, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.bn(self.conv(x))
        return Mish()(out)


class DPN_Block(nn.Module):
    """
    Dual Path block
    """

    def __init__(self, in_chnls, add_chnl, cat_chnl, cardinality, d, stride):
        super(DPN_Block, self).__init__()
        self.add = add_chnl
        self.cat = cat_chnl
        self.chnl = cardinality * d
        self.conv1 = BN_Conv2d(in_chnls, self.chnl, 1, 1, 0)
        self.conv2 = BN_Conv2d(self.chnl, self.chnl, 3, stride, 1, groups=cardinality)
        self.conv3 = nn.Conv2d(self.chnl, add_chnl + cat_chnl, 1, 1, 0)
        self.bn = nn.BatchNorm2d(add_chnl + cat_chnl)
        self.shortcut = nn.Sequential()
        if add_chnl != in_chnls:
            self.shortcut = nn.Sequential(nn.Conv2d(in_chnls, add_chnl, 1, stride, 0), nn.BatchNorm2d(add_chnl))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(self.conv3(out))
        add = out[:, :self.add, :, :] + self.shortcut(x)
        out = torch.cat((add, out[:, self.add:, :, :]), dim=1)
        return F.relu(out)


class Stem_v4_Res2(nn.Module):
    """
    stem block for Inception-v4 and Inception-RestNet-v2
    """

    def __init__(self):
        super(Stem_v4_Res2, self).__init__()
        self.step1 = nn.Sequential(BN_Conv2d(3, 32, 3, 2, 0, bias=False), BN_Conv2d(32, 32, 3, 1, 0, bias=False), BN_Conv2d(32, 64, 3, 1, 1, bias=False))
        self.step2_pool = nn.MaxPool2d(3, 2, 0)
        self.step2_conv = BN_Conv2d(64, 96, 3, 2, 0, bias=False)
        self.step3_1 = nn.Sequential(BN_Conv2d(160, 64, 1, 1, 0, bias=False), BN_Conv2d(64, 96, 3, 1, 0, bias=False))
        self.step3_2 = nn.Sequential(BN_Conv2d(160, 64, 1, 1, 0, bias=False), BN_Conv2d(64, 64, (7, 1), (1, 1), (3, 0), bias=False), BN_Conv2d(64, 64, (1, 7), (1, 1), (0, 3), bias=False), BN_Conv2d(64, 96, 3, 1, 0, bias=False))
        self.step4_pool = nn.MaxPool2d(3, 2, 0)
        self.step4_conv = BN_Conv2d(192, 192, 3, 2, 0, bias=False)

    def forward(self, x):
        out = self.step1(x)
        tmp1 = self.step2_pool(out)
        tmp2 = self.step2_conv(out)
        out = torch.cat((tmp1, tmp2), 1)
        tmp1 = self.step3_1(out)
        tmp2 = self.step3_2(out)
        out = torch.cat((tmp1, tmp2), 1)
        tmp1 = self.step4_pool(out)
        tmp2 = self.step4_conv(out)
        None
        None
        out = torch.cat((tmp1, tmp2), 1)
        return out


class Stem_Res1(nn.Module):
    """
    stem block for Inception-ResNet-v1
    """

    def __init__(self):
        super(Stem_Res1, self).__init__()
        self.stem = nn.Sequential(BN_Conv2d(3, 32, 3, 2, 0, bias=False), BN_Conv2d(32, 32, 3, 1, 0, bias=False), BN_Conv2d(32, 64, 3, 1, 1, bias=False), nn.MaxPool2d(3, 2, 0), BN_Conv2d(64, 80, 1, 1, 0, bias=False), BN_Conv2d(80, 192, 3, 1, 0, bias=False), BN_Conv2d(192, 256, 3, 2, 0, bias=False))

    def forward(self, x):
        return self.stem(x)


class Reduction_B_v4(nn.Module):
    """
    Reduction-B block for Inception-v4 net
    """

    def __init__(self, in_channels, b2_n1, b2_n3, b3_n1, b3_n1x7, b3_n7x1, b3_n3):
        super(Reduction_B_v4, self).__init__()
        self.branch2 = nn.Sequential(BN_Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False), BN_Conv2d(b2_n1, b2_n3, 3, 2, 0, bias=False))
        self.branch3 = nn.Sequential(BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False), BN_Conv2d(b3_n1, b3_n1x7, (1, 7), (1, 1), (0, 3), bias=False), BN_Conv2d(b3_n1x7, b3_n7x1, (7, 1), (1, 1), (3, 0), bias=False), BN_Conv2d(b3_n7x1, b3_n3, 3, 2, 0, bias=False))

    def forward(self, x):
        out1 = F.max_pool2d(x, 3, 2, 0)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        return torch.cat((out1, out2, out3), 1)


class Reduction_B_Res(nn.Module):
    """
    Reduction-B block for Inception-ResNet-v1     and Inception-ResNet-v1  net
    """

    def __init__(self, in_channels, b2_n1, b2_n3, b3_n1, b3_n3, b4_n1, b4_n3_1, b4_n3_2):
        super(Reduction_B_Res, self).__init__()
        self.branch2 = nn.Sequential(BN_Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False), BN_Conv2d(b2_n1, b2_n3, 3, 2, 0, bias=False))
        self.branch3 = nn.Sequential(BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False), BN_Conv2d(b3_n1, b3_n3, 3, 2, 0, bias=False))
        self.branch4 = nn.Sequential(BN_Conv2d(in_channels, b4_n1, 1, 1, 0, bias=False), BN_Conv2d(b4_n1, b4_n3_1, 3, 1, 1, bias=False), BN_Conv2d(b4_n3_1, b4_n3_2, 3, 2, 0, bias=False))

    def forward(self, x):
        out1 = F.max_pool2d(x, 3, 2, 0)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat((out1, out2, out3, out4), 1)


class Inception_A_res(nn.Module):
    """
    Inception-A block for Inception-ResNet-v1    and Inception-ResNet-v2 net
    """

    def __init__(self, in_channels, b1, b2_n1, b2_n3, b3_n1, b3_n3_1, b3_n3_2, n1_linear):
        super(Inception_A_res, self).__init__()
        self.branch1 = BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False)
        self.branch2 = nn.Sequential(BN_Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False), BN_Conv2d(b2_n1, b2_n3, 3, 1, 1, bias=False))
        self.branch3 = nn.Sequential(BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False), BN_Conv2d(b3_n1, b3_n3_1, 3, 1, 1, bias=False), BN_Conv2d(b3_n3_1, b3_n3_2, 3, 1, 1, bias=False))
        self.conv_linear = nn.Conv2d(b1 + b2_n3 + b3_n3_2, n1_linear, 1, 1, 0, bias=True)
        self.short_cut = nn.Sequential()
        if in_channels != n1_linear:
            self.short_cut = nn.Sequential(nn.Conv2d(in_channels, n1_linear, 1, 1, 0, bias=False), nn.BatchNorm2d(n1_linear))

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat((out1, out2, out3), 1)
        out = self.conv_linear(out)
        out += self.short_cut(x)
        return F.relu(out)


class Inception_B_res(nn.Module):
    """
    Inception-A block for Inception-ResNet-v1    and Inception-ResNet-v2 net
    """

    def __init__(self, in_channels, b1, b2_n1, b2_n1x7, b2_n7x1, n1_linear):
        super(Inception_B_res, self).__init__()
        self.branch1 = BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False)
        self.branch2 = nn.Sequential(BN_Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False), BN_Conv2d(b2_n1, b2_n1x7, (1, 7), (1, 1), (0, 3), bias=False), BN_Conv2d(b2_n1x7, b2_n7x1, (7, 1), (1, 1), (3, 0), bias=False))
        self.conv_linear = nn.Conv2d(b1 + b2_n7x1, n1_linear, 1, 1, 0, bias=False)
        self.short_cut = nn.Sequential()
        if in_channels != n1_linear:
            self.short_cut = nn.Sequential(nn.Conv2d(in_channels, n1_linear, 1, 1, 0, bias=False), nn.BatchNorm2d(n1_linear))

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = torch.cat((out1, out2), 1)
        out = self.conv_linear(out)
        out += self.short_cut(x)
        return F.relu(out)


class Inception_C_res(nn.Module):
    """
    Inception-C block for Inception-ResNet-v1    and Inception-ResNet-v2 net
    """

    def __init__(self, in_channels, b1, b2_n1, b2_n1x3, b2_n3x1, n1_linear):
        super(Inception_C_res, self).__init__()
        self.branch1 = BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False)
        self.branch2 = nn.Sequential(BN_Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False), BN_Conv2d(b2_n1, b2_n1x3, (1, 3), (1, 1), (0, 1), bias=False), BN_Conv2d(b2_n1x3, b2_n3x1, (3, 1), (1, 1), (1, 0), bias=False))
        self.conv_linear = nn.Conv2d(b1 + b2_n3x1, n1_linear, 1, 1, 0, bias=False)
        self.short_cut = nn.Sequential()
        if in_channels != n1_linear:
            self.short_cut = nn.Sequential(nn.Conv2d(in_channels, n1_linear, 1, 1, 0, bias=False), nn.BatchNorm2d(n1_linear))

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = torch.cat((out1, out2), 1)
        out = self.conv_linear(out)
        out += self.short_cut(x)
        return F.relu(out)


class NONLocalBlock1D(_NonLocalBlockND):

    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels, inter_channels=inter_channels, dimension=1, sub_sample=sub_sample, bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):

    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels, inter_channels=inter_channels, dimension=3, sub_sample=sub_sample, bn_layer=bn_layer)


class BottleNeck(nn.Module):
    """
    BottleNeck block for RestNet-50, ResNet-101, ResNet-152
    """
    message = 'bottleneck'

    def __init__(self, in_channels, out_channels, strides, is_se=False):
        super(BottleNeck, self).__init__()
        self.is_se = is_se
        self.conv1 = BN_Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = BN_Conv2d(out_channels, out_channels, 3, stride=strides, padding=1, bias=False)
        self.conv3 = BN_Conv2d(out_channels, out_channels * 4, 1, stride=1, padding=0, bias=False, activation=None)
        if self.is_se:
            self.se = SE(out_channels * 4, 16)
        self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels * 4, 1, stride=strides, padding=0, bias=False), nn.BatchNorm2d(out_channels * 4))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.is_se:
            coefficient = self.se(out)
            out = out * coefficient
        out = out + self.shortcut(x)
        return F.relu(out)


class Dark_block(nn.Module):
    """block for darknet"""

    def __init__(self, channels, is_se=False, inner_channels=None):
        super(Dark_block, self).__init__()
        self.is_se = is_se
        if inner_channels is None:
            inner_channels = channels // 2
        self.conv1 = BN_Conv2d_Leaky(channels, inner_channels, 1, 1, 0)
        self.conv2 = nn.Conv2d(inner_channels, channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(channels)
        if self.is_se:
            self.se = SE(channels, 16)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(out)
        if self.is_se:
            coefficient = self.se(out)
            out *= coefficient
        out += x
        return F.leaky_relu(out)


def shuffle_chnls(x, groups=2):
    """Channel Shuffle"""
    bs, chnls, h, w = x.data.size()
    if chnls % groups:
        return x
    chnls_per_group = chnls // groups
    x = x.view(bs, groups, chnls_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bs, -1, h, w)
    return x


class BasicUnit(nn.Module):
    """Basic Unit of ShuffleNet-v2"""

    def __init__(self, in_chnls, out_chnls, is_se=False, is_residual=False, c_ratio=0.5, groups=2):
        super(BasicUnit, self).__init__()
        self.is_se, self.is_res = is_se, is_residual
        self.l_chnls = int(in_chnls * c_ratio)
        self.r_chnls = in_chnls - self.l_chnls
        self.ro_chnls = out_chnls - self.l_chnls
        self.groups = groups
        self.conv1 = BN_Conv2d(self.r_chnls, self.ro_chnls, 1, 1, 0)
        self.dwconv2 = BN_Conv2d(self.ro_chnls, self.ro_chnls, 3, 1, 1, groups=self.ro_chnls, activation=None)
        act = None if self.is_res else nn.ReLU(inplace=True)
        self.conv3 = BN_Conv2d(self.ro_chnls, self.ro_chnls, 1, 1, 0, activation=act)
        if self.is_se:
            self.se = SE(self.ro_chnls, 16)
        if self.is_res:
            self.shortcut = nn.Sequential()
            if self.r_chnls != self.ro_chnls:
                self.shortcut = BN_Conv2d(self.r_chnls, self.ro_chnls, 1, 1, 0, activation=None)

    def forward(self, x):
        x_l = x[:, :self.l_chnls, :, :]
        x_r = x[:, self.l_chnls:, :, :]
        out_r = self.conv1(x_r)
        out_r = self.dwconv2(out_r)
        out_r = self.conv3(out_r)
        if self.is_se:
            coefficient = self.se(out_r)
            out_r *= coefficient
        if self.is_res:
            out_r += self.shortcut(x_r)
        out = torch.cat((x_l, out_r), 1)
        return shuffle_chnls(out, self.groups)


class DSampling(nn.Module):
    """Spatial down sampling of SuffleNet-v2"""

    def __init__(self, in_chnls, groups=2):
        super(DSampling, self).__init__()
        self.groups = groups
        self.dwconv_l1 = BN_Conv2d(in_chnls, in_chnls, 3, 2, 1, groups=in_chnls, activation=None)
        self.conv_l2 = BN_Conv2d(in_chnls, in_chnls, 1, 1, 0)
        self.conv_r1 = BN_Conv2d(in_chnls, in_chnls, 1, 1, 0)
        self.dwconv_r2 = BN_Conv2d(in_chnls, in_chnls, 3, 2, 1, groups=in_chnls, activation=None)
        self.conv_r3 = BN_Conv2d(in_chnls, in_chnls, 1, 1, 0)

    def forward(self, x):
        out_l = self.dwconv_l1(x)
        out_l = self.conv_l2(out_l)
        out_r = self.conv_r1(x)
        out_r = self.dwconv_r2(out_r)
        out_r = self.conv_r3(out_r)
        out = torch.cat((out_l, out_r), 1)
        out = shuffle_chnls(out, self.groups)
        return shuffle_chnls(out, self.groups)


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        mid_channels = out_channels // 2
        self.bottleneck = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1), ConvBNReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1), ConvBNReLU(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1))
        self.shortcut = ConvBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.bottleneck(x)
        return out + self.shortcut(x)


class CSPFirst(nn.Module):
    """
    First CSP Stage
    """

    def __init__(self, in_chnnls, out_chnls):
        super(CSPFirst, self).__init__()
        self.dsample = BN_Conv_Mish(in_chnnls, out_chnls, 3, 2, 1)
        self.trans_0 = BN_Conv_Mish(out_chnls, out_chnls, 1, 1, 0)
        self.trans_1 = BN_Conv_Mish(out_chnls, out_chnls, 1, 1, 0)
        self.block = ResidualBlock(out_chnls, out_chnls // 2)
        self.trans_cat = BN_Conv_Mish(2 * out_chnls, out_chnls, 1, 1, 0)

    def forward(self, x):
        x = self.dsample(x)
        out_0 = self.trans_0(x)
        out_1 = self.trans_1(x)
        out_1 = self.block(out_1)
        out = torch.cat((out_0, out_1), 1)
        out = self.trans_cat(out)
        return out


class CSPStem(nn.Module):
    """
    CSP structures including downsampling
    """

    def __init__(self, in_chnls, out_chnls, num_block):
        super(CSPStem, self).__init__()
        self.dsample = BN_Conv_Mish(in_chnls, out_chnls, 3, 2, 1)
        self.trans_0 = BN_Conv_Mish(out_chnls, out_chnls // 2, 1, 1, 0)
        self.trans_1 = BN_Conv_Mish(out_chnls, out_chnls // 2, 1, 1, 0)
        self.blocks = nn.Sequential(*[ResidualBlock(out_chnls // 2) for _ in range(num_block)])
        self.trans_cat = BN_Conv_Mish(out_chnls, out_chnls, 1, 1, 0)

    def forward(self, x):
        x = self.dsample(x)
        out_0 = self.trans_0(x)
        out_1 = self.trans_1(x)
        out_1 = self.blocks(out_1)
        out = torch.cat((out_0, out_1), 1)
        out = self.trans_cat(out)
        return out


class CSP_DarkNet(nn.Module):
    """
    CSP-DarkNet
    """

    def __init__(self, num_blocks: 'object', num_classes=1000) ->object:
        super(CSP_DarkNet, self).__init__()
        chnls = [64, 128, 256, 512, 1024]
        self.conv0 = BN_Conv_Mish(3, 32, 3, 1, 1)
        self.neck = CSPFirst(32, chnls[0])
        self.body = nn.Sequential(*[CSPStem(chnls[i], chnls[i + 1], num_blocks[i]) for i in range(4)])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(chnls[4], num_classes)

    def forward(self, x):
        out = self.conv0(x)
        out = self.neck(out)
        out = self.body(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Stem(nn.Module):

    def __init__(self):
        super(Stem, self).__init__()
        self.conv1 = nn.Sequential(BasicConv2d(3, 32, kernel_size=3, stride=2), BasicConv2d(32, 32, kernel_size=3), BasicConv2d(32, 64, kernel_size=3, padding=1))
        self.conv1_pool_branch = nn.MaxPool2d(3, stride=2)
        self.conv1_branch = BasicConv2d(64, 96, kernel_size=3, stride=2)
        self.conv2_short = nn.Sequential(BasicConv2d(160, 64, kernel_size=1), BasicConv2d(64, 96, kernel_size=3))
        self.conv2_long = nn.Sequential(BasicConv2d(160, 64, kernel_size=1), BasicConv2d(64, 64, kernel_size=(7, 1), padding=(3, 0)), BasicConv2d(64, 64, kernel_size=(1, 7), padding=(0, 3)), BasicConv2d(64, 96, kernel_size=3))
        self.conv2_pool_branch = nn.MaxPool2d(3, stride=2)
        self.conv2_branch = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x0 = self.conv1_pool_branch(x)
        x1 = self.conv1_branch(x)
        x = torch.cat((x0, x1), 1)
        x0 = self.conv2_short(x)
        x1 = self.conv2_long(x)
        x = torch.cat((x0, x1), 1)
        x0 = self.conv2_pool_branch(x)
        x1 = self.conv2_branch(x)
        out = torch.cat((x0, x1), 1)
        return out


class CSP_ResNeXt(nn.Module):

    def __init__(self, num_blocks, cadinality, group_width, num_classes):
        super(CSP_ResNeXt, self).__init__()
        self.conv0 = BN_Conv2d_Leaky(3, 64, 7, 2, 3)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.conv1 = BN_Conv2d_Leaky(64, 128, 1, 1, 0)
        self.stem0 = Stem(cadinality * group_width * 2, num_blocks[0], cadinality, group_width, stride=1)
        self.stem1 = Stem(cadinality * group_width * 4, num_blocks[1], cadinality, group_width * 2)
        self.stem2 = Stem(cadinality * group_width * 8, num_blocks[2], cadinality, group_width * 4)
        self.stem3 = Stem(cadinality * group_width * 16, num_blocks[3], cadinality, group_width * 8)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(cadinality * group_width * 16, num_classes)

    def forward(self, x):
        out = self.conv0(x)
        out = self.pool1(out)
        out = self.conv1(out)
        out = self.stem0(out)
        out = self.stem1(out)
        out = self.stem2(out)
        out = self.stem3(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class DarkNet(nn.Module):

    def __init__(self, layers: 'object', num_classes, is_se=False) ->object:
        super(DarkNet, self).__init__()
        self.is_se = is_se
        filters = [64, 128, 256, 512, 1024]
        self.conv1 = BN_Conv2d(3, 32, 3, 1, 1)
        self.redu1 = BN_Conv2d(32, 64, 3, 2, 1)
        self.conv2 = self.__make_layers(filters[0], layers[0])
        self.redu2 = BN_Conv2d(filters[0], filters[1], 3, 2, 1)
        self.conv3 = self.__make_layers(filters[1], layers[1])
        self.redu3 = BN_Conv2d(filters[1], filters[2], 3, 2, 1)
        self.conv4 = self.__make_layers(filters[2], layers[2])
        self.redu4 = BN_Conv2d(filters[2], filters[3], 3, 2, 1)
        self.conv5 = self.__make_layers(filters[3], layers[3])
        self.redu5 = BN_Conv2d(filters[3], filters[4], 3, 2, 1)
        self.conv6 = self.__make_layers(filters[4], layers[4])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters[4], num_classes)

    def __make_layers(self, num_filter, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(Dark_block(num_filter, self.is_se))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.redu1(out)
        out = self.conv2(out)
        out = self.redu2(out)
        out = self.conv3(out)
        out = self.redu3(out)
        out = self.conv4(out)
        out = self.redu4(out)
        out = self.conv5(out)
        out = self.redu5(out)
        out = self.conv6(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CatBnAct(nn.Module):

    def __init__(self, in_chs, activation_fn=nn.ReLU(inplace=True)):
        super(CatBnAct, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn

    def forward(self, x):
        x = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        return self.act(self.bn(x))


class BnActConv2d(nn.Module):

    def __init__(self, in_chs, out_chs, kernel_size, stride, padding=0, groups=1, activation_fn=nn.ReLU(inplace=True)):
        super(BnActConv2d, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups, bias=False)

    def forward(self, x):
        return self.conv(self.act(self.bn(x)))


class DualPathBlock(nn.Module):

    def __init__(self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups, block_type='normal', b=False):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c
        self.inc = inc
        self.b = b
        if block_type is 'proj':
            self.key_stride = 1
            self.has_proj = True
        elif block_type is 'down':
            self.key_stride = 2
            self.has_proj = True
        else:
            assert block_type is 'normal'
            self.key_stride = 1
            self.has_proj = False
        if self.has_proj:
            if self.key_stride == 2:
                self.c1x1_w_s2 = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=2)
            else:
                self.c1x1_w_s1 = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=1)
        self.c1x1_a = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)
        self.c3x3_b = BnActConv2d(in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3, stride=self.key_stride, padding=1, groups=groups)
        if b:
            self.c1x1_c = CatBnAct(in_chs=num_3x3_b)
            self.c1x1_c1 = nn.Conv2d(num_3x3_b, num_1x1_c, kernel_size=1, bias=False)
            self.c1x1_c2 = nn.Conv2d(num_3x3_b, inc, kernel_size=1, bias=False)
        else:
            self.c1x1_c = BnActConv2d(in_chs=num_3x3_b, out_chs=num_1x1_c + inc, kernel_size=1, stride=1)

    def forward(self, x):
        x_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        if self.has_proj:
            if self.key_stride == 2:
                x_s = self.c1x1_w_s2(x_in)
            else:
                x_s = self.c1x1_w_s1(x_in)
            x_s1 = x_s[:, :self.num_1x1_c, :, :]
            x_s2 = x_s[:, self.num_1x1_c:, :, :]
        else:
            x_s1 = x[0]
            x_s2 = x[1]
        x_in = self.c1x1_a(x_in)
        x_in = self.c3x3_b(x_in)
        if self.b:
            x_in = self.c1x1_c(x_in)
            out1 = self.c1x1_c1(x_in)
            out2 = self.c1x1_c2(x_in)
        else:
            x_in = self.c1x1_c(x_in)
            out1 = x_in[:, :self.num_1x1_c, :, :]
            out2 = x_in[:, self.num_1x1_c:, :, :]
        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)
        return resid, dense


class InputBlock(nn.Module):

    def __init__(self, num_init_features, kernel_size=7, padding=3, activation_fn=nn.ReLU(inplace=True)):
        super(InputBlock, self).__init__()
        self.conv = nn.Conv2d(3, num_init_features, kernel_size=kernel_size, stride=2, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(num_init_features, eps=0.001)
        self.act = activation_fn
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


def adaptive_avgmax_pool2d(x, pool_type='avg', padding=0, count_include_pad=False):
    """Selectable global pooling function with dynamic input kernel size
    """
    if pool_type == 'avgmaxc':
        x = torch.cat([F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=padding, count_include_pad=count_include_pad), F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=padding)], dim=1)
    elif pool_type == 'avgmax':
        x_avg = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=padding, count_include_pad=count_include_pad)
        x_max = F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=padding)
        x = 0.5 * (x_avg + x_max)
    elif pool_type == 'max':
        x = F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=padding)
    else:
        if pool_type != 'avg':
            None
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=padding, count_include_pad=count_include_pad)
    return x


class DPN(nn.Module):

    def __init__(self, small=False, num_init_features=64, k_r=96, groups=32, b=False, k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128), num_classes=1000, test_time_pool=False):
        super(DPN, self).__init__()
        self.test_time_pool = test_time_pool
        self.b = b
        bw_factor = 1 if small else 4
        blocks = OrderedDict()
        if small:
            blocks['conv1_1'] = InputBlock(num_init_features, kernel_size=3, padding=1)
        else:
            blocks['conv1_1'] = InputBlock(num_init_features, kernel_size=7, padding=3)
        bw = 64 * bw_factor
        inc = inc_sec[0]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv2_1'] = DualPathBlock(num_init_features, r, r, bw, inc, groups, 'proj', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0] + 1):
            blocks['conv2_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        bw = 128 * bw_factor
        inc = inc_sec[1]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv3_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1] + 1):
            blocks['conv3_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        bw = 256 * bw_factor
        inc = inc_sec[2]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv4_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2] + 1):
            blocks['conv4_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        bw = 512 * bw_factor
        inc = inc_sec[3]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv5_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3] + 1):
            blocks['conv5_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        blocks['conv5_bn_ac'] = CatBnAct(in_chs)
        self.features = nn.Sequential(blocks)
        self.last_linear = nn.Conv2d(in_chs, num_classes, kernel_size=1, bias=True)

    def logits(self, features):
        if not self.training and self.test_time_pool:
            x = F.avg_pool2d(features, kernel_size=7, stride=1)
            out = self.last_linear(x)
            out = adaptive_avgmax_pool2d(out, pool_type='avgmax')
        else:
            x = adaptive_avgmax_pool2d(features, pool_type='avg')
            out = self.last_linear(x)
        return out.view(out.size(0), -1)

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class MaxPool2dDynamicSamePadding(nn.MaxPool2d):
    """2D MaxPooling like TensorFlow's 'SAME' mode, with a dynamic image size.
       The padding is operated in forward function by calculating dynamically.
    """

    def __init__(self, kernel_size, stride, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.stride = [self.stride] * 2 if isinstance(self.stride, int) else self.stride
        self.kernel_size = [self.kernel_size] * 2 if isinstance(self.kernel_size, int) else self.kernel_size
        self.dilation = [self.dilation] * 2 if isinstance(self.dilation, int) else self.dilation

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, self.return_indices)


class MaxPool2dStaticSamePadding(nn.MaxPool2d):
    """2D MaxPooling like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """

    def __init__(self, kernel_size, stride, image_size=None, **kwargs):
        super().__init__(kernel_size, stride, **kwargs)
        self.stride = [self.stride] * 2 if isinstance(self.stride, int) else self.stride
        self.kernel_size = [self.kernel_size] * 2 if isinstance(self.kernel_size, int) else self.kernel_size
        self.dilation = [self.dilation] * 2 if isinstance(self.dilation, int) else self.dilation
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, self.return_indices)
        return x


class Inception_builder(nn.Module):
    """
    types of Inception block
    """

    def __init__(self, block_type, in_channels, b1_reduce, b1, b2_reduce, b2, b3, b4):
        super(Inception_builder, self).__init__()
        self.block_type = block_type
        self.branch1_type1 = nn.Sequential(BN_Conv2d(in_channels, b1_reduce, 1, stride=1, padding=0, bias=False), BN_Conv2d(b1_reduce, b1, 5, stride=1, padding=2, bias=False))
        self.branch1_type2 = nn.Sequential(BN_Conv2d(in_channels, b1_reduce, 1, stride=1, padding=0, bias=False), BN_Conv2d(b1_reduce, b1, 3, stride=1, padding=1, bias=False), BN_Conv2d(b1, b1, 3, stride=1, padding=1, bias=False))
        self.branch2 = nn.Sequential(BN_Conv2d(in_channels, b2_reduce, 1, stride=1, padding=0, bias=False), BN_Conv2d(b2_reduce, b2, 3, stride=1, padding=1, bias=False))
        self.branch3 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), BN_Conv2d(in_channels, b3, 1, stride=1, padding=0, bias=False))
        self.branch4 = BN_Conv2d(in_channels, b4, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        if self.block_type == 'type1':
            out1 = self.branch1_type1(x)
            out2 = self.branch2(x)
        elif self.block_type == 'type2':
            out1 = self.branch1_type2(x)
            out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat((out1, out2, out3, out4), 1)
        return out


class GoogleNet(nn.Module):
    """
    Inception-v1, Inception-v2
    """

    def __init__(self, str_version, num_classes):
        super(GoogleNet, self).__init__()
        self.block_type = 'type1' if str_version == 'v1' else 'type2'
        self.version = str_version
        self.conv1 = BN_Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.conv2 = BN_Conv2d(64, 192, 3, stride=1, padding=1, bias=False)
        self.inception3_a = Inception_builder(self.block_type, 192, 16, 32, 96, 128, 32, 64)
        self.inception3_b = Inception_builder(self.block_type, 256, 32, 96, 128, 192, 64, 128)
        self.inception4_a = Inception_builder(self.block_type, 480, 16, 48, 96, 208, 64, 192)
        self.inception4_b = Inception_builder(self.block_type, 512, 24, 64, 112, 224, 64, 160)
        self.inception4_c = Inception_builder(self.block_type, 512, 24, 64, 128, 256, 64, 128)
        self.inception4_d = Inception_builder(self.block_type, 512, 32, 64, 144, 288, 64, 112)
        self.inception4_e = Inception_builder(self.block_type, 528, 32, 128, 160, 320, 128, 256)
        self.inception5_a = Inception_builder(self.block_type, 832, 32, 128, 160, 320, 128, 256)
        self.inception5_b = Inception_builder(self.block_type, 832, 48, 128, 192, 384, 128, 384)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.conv2(out)
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.inception3_a(out)
        out = self.inception3_b(out)
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.inception4_a(out)
        out = self.inception4_b(out)
        out = self.inception4_c(out)
        out = self.inception4_d(out)
        out = self.inception4_e(out)
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.inception5_a(out)
        out = self.inception5_b(out)
        out = F.avg_pool2d(out, 7)
        out = F.dropout(out, 0.4, training=self.training)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class SeqConv(nn.Module):

    def __init__(self, in_chnls, out_chnls, kernel_size, activation=nn.ReLU(inplace=True)):
        super(SeqConv, self).__init__()
        self.DWConv = BN_Conv2d(in_chnls, in_chnls, kernel_size, stride=1, padding=kernel_size // 2, groups=in_chnls, activation=activation)
        self.trans = BN_Conv2d(in_chnls, out_chnls, 1, 1, 0, activation=None)

    def forward(self, x):
        out = self.DWConv(x)
        return self.trans(out)


class MBConv(nn.Module):
    """Mobile inverted bottleneck conv"""

    def __init__(self, in_chnls, out_chnls, kernel_size, expansion, stride, is_se=False, activation=nn.ReLU(inplace=True)):
        super(MBConv, self).__init__()
        self.is_se = is_se
        self.is_shortcut = stride == 1 and in_chnls == out_chnls
        self.trans1 = BN_Conv2d(in_chnls, in_chnls * expansion, 1, 1, 0, activation=activation)
        self.DWConv = BN_Conv2d(in_chnls * expansion, in_chnls * expansion, kernel_size, stride=stride, padding=kernel_size // 2, groups=in_chnls * expansion, activation=activation)
        if self.is_se:
            self.se = SE(in_chnls * expansion, 4)
        self.trans2 = BN_Conv2d(in_chnls * expansion, out_chnls, 1, 1, 0, activation=None)

    def forward(self, x):
        out = self.trans1(x)
        out = self.DWConv(out)
        if self.is_se:
            coeff = self.se(out)
            out *= coeff
        out = self.trans2(out)
        if self.is_shortcut:
            out += x
        return out


class MnasNet_A1(nn.Module):
    """MnasNet-A1"""
    _defaults = {'blocks': [2, 3, 4, 2, 3, 1], 'chnls': [24, 40, 80, 112, 160, 320], 'expans': [6, 3, 6, 6, 6, 6], 'k_sizes': [3, 5, 3, 3, 5, 3], 'strides': [2, 2, 2, 1, 2, 1], 'is_se': [False, True, False, True, True, False], 'dropout_ratio': 0.2}

    def __init__(self, num_classes=1000, input_size=224):
        super(MnasNet_A1, self).__init__()
        self.__dict__.update(self._defaults)
        self.body = self.__make_body()
        self.trans = BN_Conv2d(self.chnls[-1], 1280, 1, 1, 0, activation=nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(self.dropout_ratio), nn.Linear(1280, num_classes))

    def __make_block(self, id):
        in_chnls = 16 if id == 0 else self.chnls[id - 1]
        strides = [self.strides[id]] + [1] * (self.blocks[id] - 1)
        layers = []
        for i in range(self.blocks[id]):
            layers.append(MBConv(in_chnls, self.chnls[id], self.k_sizes[id], self.expans[id], strides[i], self.is_se[id]))
            in_chnls = self.chnls[id]
        return nn.Sequential(*layers)

    def __make_body(self):
        blocks = [BN_Conv2d(3, 32, 3, 2, 1, activation=None), SeqConv(32, 16, 3)]
        for index in range(len(self.blocks)):
            blocks.append(self.__make_block(index))
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.body(x)
        out = self.trans(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        None
        out = self.fc(out)
        return out


class PositionalEncoding(nn.Module):

    def __init__(self, embed_size, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = np.zeros((1, max_len, embed_size))
        X = np.arange(0, max_len).reshape(-1, 1) / np.power(10000, np.arange(0, embed_size, 2) / embed_size)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)
        self.P = torch.FloatTensor(self.P)

    def forward(self, X):
        if X.is_cuda and not self.P.is_cuda:
            self.P = self.P
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X)


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-06)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        q, attn = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-06)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    """ Compose with three layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn


class Conv2dCReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2dCReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.bn(self.conv(x))
        out = torch.cat([x, -x], dim=1)
        return self.relu(out)


class InceptionModules(nn.Module):

    def __init__(self):
        super(InceptionModules, self).__init__()
        self.branch1_conv1 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1)
        self.branch1_conv1_bn = nn.BatchNorm2d(32)
        self.branch2_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch2_conv1 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1)
        self.branch2_conv1_bn = nn.BatchNorm2d(32)
        self.branch3_conv1 = nn.Conv2d(in_channels=128, out_channels=24, kernel_size=1, stride=1)
        self.branch3_conv1_bn = nn.BatchNorm2d(24)
        self.branch3_conv2 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch3_conv2_bn = nn.BatchNorm2d(32)
        self.branch4_conv1 = nn.Conv2d(in_channels=128, out_channels=24, kernel_size=1, stride=1)
        self.branch4_conv1_bn = nn.BatchNorm2d(24)
        self.branch4_conv2 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch4_conv2_bn = nn.BatchNorm2d(32)
        self.branch4_conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch4_conv3_bn = nn.BatchNorm2d(32)

    def forward(self, x):
        x1 = self.branch1_conv1_bn(self.branch1_conv1(x))
        x2 = self.branch2_conv1_bn(self.branch2_conv1(self.branch2_pool(x)))
        x3 = self.branch3_conv2_bn(self.branch3_conv2(self.branch3_conv1_bn(self.branch3_conv1(x))))
        x4 = self.branch4_conv3_bn(self.branch4_conv3(self.branch4_conv2_bn(self.branch4_conv2(self.branch4_conv1_bn(self.branch4_conv1(x))))))
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return out


class FaceBoxes(nn.Module):

    def __init__(self, num_classes, phase):
        super(FaceBoxes, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.RapidlyDigestedConvolutionalLayers = nn.Sequential(Conv2dCReLU(in_channels=3, out_channels=24, kernel_size=7, stride=4, padding=3), nn.MaxPool2d(kernel_size=3, stride=2, padding=1), Conv2dCReLU(in_channels=48, out_channels=64, kernel_size=5, stride=2, padding=2), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.MultipleScaleConvolutionalLayers = nn.Sequential(InceptionModules(), InceptionModules(), InceptionModules())
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1)
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.loc_layer1 = nn.Conv2d(in_channels=128, out_channels=21 * 4, kernel_size=3, stride=1, padding=1)
        self.conf_layer1 = nn.Conv2d(in_channels=128, out_channels=21 * num_classes, kernel_size=3, stride=1, padding=1)
        self.loc_layer2 = nn.Conv2d(in_channels=256, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conf_layer2 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=3, stride=1, padding=1)
        self.loc_layer3 = nn.Conv2d(in_channels=256, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conf_layer3 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=3, stride=1, padding=1)
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
        elif self.phase == 'train':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.xavier_normal_(m.weight.data)
                        nn.init.constant_(m.bias, 0)
                    else:
                        nn.init.xavier_normal_(m.weight.data)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.RapidlyDigestedConvolutionalLayers(x)
        out1 = self.MultipleScaleConvolutionalLayers(x)
        out2 = self.conv3_2(self.conv3_1(out1))
        out3 = self.conv4_2(self.conv4_1(out2))
        loc1 = self.loc_layer1(out1)
        conf1 = self.conf_layer1(out1)
        loc2 = self.loc_layer2(out2)
        conf2 = self.conf_layer2(out2)
        loc3 = self.loc_layer3(out3)
        conf3 = self.conf_layer3(out3)
        locs = torch.cat([loc1.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), loc2.permute(0, 2, 3, 1).contiguous().view(loc2.size(0), -1), loc3.permute(0, 2, 3, 1).contiguous().view(loc3.size(0), -1)], dim=1)
        confs = torch.cat([conf1.permute(0, 2, 3, 1).contiguous().view(conf1.size(0), -1), conf2.permute(0, 2, 3, 1).contiguous().view(conf2.size(0), -1), conf3.permute(0, 2, 3, 1).contiguous().view(conf3.size(0), -1)], dim=1)
        if self.phase == 'test':
            out = locs.view(locs.size(0), -1, 4), self.softmax(confs.view(-1, self.num_classes))
        else:
            out = locs.view(locs.size(0), -1, 4), confs.view(-1, self.num_classes)
        return out


def Conv1x1ReLU(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1), nn.ReLU6(inplace=True))


class LossBranch(nn.Module):

    def __init__(self, in_channels, mid_channels=64):
        super(LossBranch, self).__init__()
        self.conv1 = Conv1x1ReLU(in_channels, mid_channels)
        self.conv2_score = Conv1x1ReLU(mid_channels, mid_channels)
        self.classify = nn.Conv2d(in_channels=mid_channels, out_channels=2, kernel_size=1, stride=1)
        self.conv2_bbox = Conv1x1ReLU(mid_channels, mid_channels)
        self.regress = nn.Conv2d(in_channels=mid_channels, out_channels=4, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        cls = self.classify(self.conv2_score(x))
        reg = self.regress(self.conv2_bbox(x))
        return cls, reg


def Conv3x3ReLU(in_channels, out_channels, stride, padding=1):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=padding), nn.ReLU6(inplace=True))


class LFFDBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(LFFDBlock, self).__init__()
        mid_channels = out_channels
        self.downsampling = True if stride == 2 else False
        if self.downsampling:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=stride, padding=0)
        self.branch1_relu1 = nn.ReLU6(inplace=True)
        self.branch1_conv1 = Conv3x3ReLU(in_channels=mid_channels, out_channels=mid_channels, stride=1, padding=1)
        self.branch1_conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        if self.downsampling:
            x = self.conv(x)
        out = self.branch1_conv2(self.branch1_conv1(self.branch1_relu1(x)))
        return self.relu(out + x)


class LFFD(nn.Module):

    def __init__(self, classes_num=2):
        super(LFFD, self).__init__()
        self.tiny_part1 = nn.Sequential(Conv3x3ReLU(in_channels=3, out_channels=64, stride=2, padding=0), LFFDBlock(in_channels=64, out_channels=64, stride=2), LFFDBlock(in_channels=64, out_channels=64, stride=1), LFFDBlock(in_channels=64, out_channels=64, stride=1))
        self.tiny_part2 = LFFDBlock(in_channels=64, out_channels=64, stride=1)
        self.small_part1 = LFFDBlock(in_channels=64, out_channels=64, stride=2)
        self.small_part2 = LFFDBlock(in_channels=64, out_channels=64, stride=1)
        self.medium_part = nn.Sequential(LFFDBlock(in_channels=64, out_channels=128, stride=2), LFFDBlock(in_channels=128, out_channels=128, stride=1))
        self.large_part1 = LFFDBlock(in_channels=128, out_channels=128, stride=2)
        self.large_part2 = LFFDBlock(in_channels=128, out_channels=128, stride=1)
        self.large_part3 = LFFDBlock(in_channels=128, out_channels=128, stride=1)
        self.loss_branch1 = LossBranch(in_channels=64)
        self.loss_branch2 = LossBranch(in_channels=64)
        self.loss_branch3 = LossBranch(in_channels=64)
        self.loss_branch4 = LossBranch(in_channels=64)
        self.loss_branch5 = LossBranch(in_channels=128)
        self.loss_branch6 = LossBranch(in_channels=128)
        self.loss_branch7 = LossBranch(in_channels=128)
        self.loss_branch8 = LossBranch(in_channels=128)

    def forward(self, x):
        branch1 = self.tiny_part1(x)
        branch2 = self.tiny_part2(branch1)
        branch3 = self.small_part1(branch2)
        branch4 = self.small_part2(branch3)
        branch5 = self.medium_part(branch4)
        branch6 = self.large_part1(branch5)
        branch7 = self.large_part2(branch6)
        branch8 = self.large_part3(branch7)
        cls1, loc1 = self.loss_branch1(branch1)
        cls2, loc2 = self.loss_branch2(branch2)
        cls3, loc3 = self.loss_branch3(branch3)
        cls4, loc4 = self.loss_branch4(branch4)
        cls5, loc5 = self.loss_branch5(branch5)
        cls6, loc6 = self.loss_branch6(branch6)
        cls7, loc7 = self.loss_branch7(branch7)
        cls8, loc8 = self.loss_branch8(branch8)
        cls = torch.cat([cls1.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), cls2.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), cls3.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), cls4.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), cls5.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), cls6.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), cls7.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), cls8.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1)], dim=1)
        loc = torch.cat([loc1.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), loc2.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), loc3.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), loc4.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), loc5.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), loc6.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), loc7.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), loc8.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1)], dim=1)
        out = cls, loc
        return out


class HardSwish(nn.Module):

    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6(inplace)

    def forward(self, x):
        return x * self.relu6(x + 3) / 6


class SqueezeAndExcite(nn.Module):

    def __init__(self, in_channels, out_channels, se_kernel_size, divide=4):
        super(SqueezeAndExcite, self).__init__()
        mid_channels = in_channels // divide
        self.pool = nn.AvgPool2d(kernel_size=se_kernel_size, stride=1)
        self.SEblock = nn.Sequential(nn.Linear(in_features=in_channels, out_features=mid_channels), nn.ReLU6(inplace=True), nn.Linear(in_features=mid_channels, out_features=out_channels), HardSwish(inplace=True))

    def forward(self, x):
        b, c, h, w = x.size()
        out = self.pool(x)
        out = out.view(b, -1)
        out = self.SEblock(out)
        out = out.view(b, c, 1, 1)
        return out * x


def VarGConv(in_channels, out_channels, kernel_size, stride, S):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, groups=in_channels // S, bias=False), nn.BatchNorm2d(out_channels), nn.PReLU())


def VarGPointConv(in_channels, out_channels, stride, S, isRelu):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, padding=0, groups=in_channels // S, bias=False), nn.BatchNorm2d(out_channels), nn.PReLU() if isRelu else nn.Sequential())


class VarGBlock_S1(nn.Module):

    def __init__(self, in_plances, kernel_size, stride=1, S=8):
        super(VarGBlock_S1, self).__init__()
        plances = 2 * in_plances
        self.varGConv1 = VarGConv(in_plances, plances, kernel_size, stride, S)
        self.varGPointConv1 = VarGPointConv(plances, in_plances, stride, S, isRelu=True)
        self.varGConv2 = VarGConv(in_plances, plances, kernel_size, stride, S)
        self.varGPointConv2 = VarGPointConv(plances, in_plances, stride, S, isRelu=False)
        self.se = SqueezeAndExcite(in_plances, in_plances)
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = x
        x = self.varGPointConv1(self.varGConv1(x))
        x = self.varGPointConv2(self.varGConv2(x))
        x = self.se(x)
        out += x
        return self.prelu(out)


class VarGBlock_S2(nn.Module):

    def __init__(self, in_plances, kernel_size, stride=2, S=8):
        super(VarGBlock_S2, self).__init__()
        plances = 2 * in_plances
        self.varGConvBlock_branch1 = nn.Sequential(VarGConv(in_plances, plances, kernel_size, stride, S), VarGPointConv(plances, plances, 1, S, isRelu=True))
        self.varGConvBlock_branch2 = nn.Sequential(VarGConv(in_plances, plances, kernel_size, stride, S), VarGPointConv(plances, plances, 1, S, isRelu=True))
        self.varGConvBlock_3 = nn.Sequential(VarGConv(plances, plances * 2, kernel_size, 1, S), VarGPointConv(plances * 2, plances, 1, S, isRelu=False))
        self.shortcut = nn.Sequential(VarGConv(in_plances, plances, kernel_size, stride, S), VarGPointConv(plances, plances, 1, S, isRelu=False))
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.shortcut(x)
        x1 = x2 = x
        x1 = self.varGConvBlock_branch1(x1)
        x2 = self.varGConvBlock_branch2(x2)
        x_new = x1 + x2
        x_new = self.varGConvBlock_3(x_new)
        out += x_new
        return self.prelu(out)


class HeadBlock(nn.Module):

    def __init__(self, in_plances, kernel_size, S=8):
        super(HeadBlock, self).__init__()
        self.varGConvBlock = nn.Sequential(VarGConv(in_plances, in_plances, kernel_size, 2, S), VarGPointConv(in_plances, in_plances, 1, S, isRelu=True), VarGConv(in_plances, in_plances, kernel_size, 1, S), VarGPointConv(in_plances, in_plances, 1, S, isRelu=False))
        self.shortcut = nn.Sequential(VarGConv(in_plances, in_plances, kernel_size, 2, S), VarGPointConv(in_plances, in_plances, 1, S, isRelu=False))

    def forward(self, x):
        out = self.shortcut(x)
        x = self.varGConvBlock(x)
        out += x
        return out


class TailEmbedding(nn.Module):

    def __init__(self, in_plances, plances=512, S=8):
        super(TailEmbedding, self).__init__()
        self.embedding = nn.Sequential(Conv1x1BNReLU(in_plances, 1024), nn.Conv2d(1024, 1024, 7, 1, padding=0, groups=1024 // S, bias=False), nn.Conv2d(1024, 512, 1, 1, padding=0, groups=512, bias=False))
        self.fc = nn.Linear(in_features=512, out_features=plances)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


class VarGFaceNet(nn.Module):

    def __init__(self, num_classes=512):
        super(VarGFaceNet, self).__init__()
        S = 8
        self.conv1 = Conv3x3BNReLU(3, 40, 1)
        self.head = HeadBlock(40, 3)
        self.stage2 = nn.Sequential(VarGBlock_S2(40, 3, 2), VarGBlock_S1(80, 3, 1), VarGBlock_S1(80, 3, 1))
        self.stage3 = nn.Sequential(VarGBlock_S2(80, 3, 2), VarGBlock_S1(160, 3, 1), VarGBlock_S1(160, 3, 1), VarGBlock_S1(160, 3, 1), VarGBlock_S1(160, 3, 1), VarGBlock_S1(160, 3, 1), VarGBlock_S1(160, 3, 1))
        self.stage4 = nn.Sequential(VarGBlock_S2(160, 3, 2), VarGBlock_S1(320, 3, 1), VarGBlock_S1(320, 3, 1), VarGBlock_S1(320, 3, 1))
        self.tail = TailEmbedding(320, num_classes)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.head(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        out = self.tail(x)
        return out


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(nn.Linear(int(np.prod(img_shape)), 512), nn.LeakyReLU(0.2, inplace=True), nn.Linear(512, 256), nn.LeakyReLU(0.2, inplace=True), nn.Linear(256, 1))

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(*block(opt.latent_dim, 128, normalize=False), *block(128, 256), *block(256, 512), *block(512, 1024), nn.Linear(1024, int(np.prod(img_shape))), nn.Tanh())

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class UNetDown(nn.Module):

    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):

    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False), nn.InstanceNorm2d(out_size), nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class MultiDiscriminator(nn.Module):

    def __init__(self, in_channels=3):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module('disc_%d' % i, nn.Sequential(*discriminator_block(in_channels, 64, normalize=False), *discriminator_block(64, 128), *discriminator_block(128, 256), *discriminator_block(256, 512), nn.Conv2d(512, 1, 3, padding=1)))
        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs


class Reshape(nn.Module):
    """
    Class for performing a reshape as a layer in a sequential model.
    """

    def __init__(self, shape=[]):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

    def extra_repr(self):
        return 'shape={}'.format(self.shape)


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


class Generator_CNN(nn.Module):
    """
    CNN to model the generator of a ClusterGAN
    Input is a vector from representation space of dimension z_dim
    output is a vector from image space of dimension X_dim
    """

    def __init__(self, latent_dim, n_c, x_shape, verbose=False):
        super(Generator_CNN, self).__init__()
        self.name = 'generator'
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.x_shape = x_shape
        self.ishape = 128, 7, 7
        self.iels = int(np.prod(self.ishape))
        self.verbose = verbose
        self.model = nn.Sequential(torch.nn.Linear(self.latent_dim + self.n_c, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2, inplace=True), torch.nn.Linear(1024, self.iels), nn.BatchNorm1d(self.iels), nn.LeakyReLU(0.2, inplace=True), Reshape(self.ishape), nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True), nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=True), nn.Sigmoid())
        initialize_weights(self)
        if self.verbose:
            None
            None

    def forward(self, zn, zc):
        z = torch.cat((zn, zc), 1)
        x_gen = self.model(z)
        x_gen = x_gen.view(x_gen.size(0), *self.x_shape)
        return x_gen


class Encoder_CNN(nn.Module):
    """
    CNN to model the encoder of a ClusterGAN
    Input is vector X from image space if dimension X_dim
    Output is vector z from representation space of dimension z_dim
    """

    def __init__(self, latent_dim, n_c, verbose=False):
        super(Encoder_CNN, self).__init__()
        self.name = 'encoder'
        self.channels = 1
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.cshape = 128, 5, 5
        self.iels = int(np.prod(self.cshape))
        self.lshape = self.iels,
        self.verbose = verbose
        self.model = nn.Sequential(nn.Conv2d(self.channels, 64, 4, stride=2, bias=True), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(64, 128, 4, stride=2, bias=True), nn.LeakyReLU(0.2, inplace=True), Reshape(self.lshape), torch.nn.Linear(self.iels, 1024), nn.LeakyReLU(0.2, inplace=True), torch.nn.Linear(1024, latent_dim + n_c))
        initialize_weights(self)
        if self.verbose:
            None
            None

    def forward(self, in_feat):
        z_img = self.model(in_feat)
        z = z_img.view(z_img.shape[0], -1)
        zn = z[:, 0:self.latent_dim]
        zc_logits = z[:, self.latent_dim:]
        zc = softmax(zc_logits)
        return zn, zc, zc_logits


class Discriminator_CNN(nn.Module):
    """
    CNN to model the discriminator of a ClusterGAN
    Input is tuple (X,z) of an image vector and its corresponding
    representation z vector. For example, if X comes from the dataset, corresponding
    z is Encoder(X), and if z is sampled from representation space, X is Generator(z)
    Output is a 1-dimensional value
    """

    def __init__(self, wass_metric=False, verbose=False):
        super(Discriminator_CNN, self).__init__()
        self.name = 'discriminator'
        self.channels = 1
        self.cshape = 128, 5, 5
        self.iels = int(np.prod(self.cshape))
        self.lshape = self.iels,
        self.wass = wass_metric
        self.verbose = verbose
        self.model = nn.Sequential(nn.Conv2d(self.channels, 64, 4, stride=2, bias=True), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(64, 128, 4, stride=2, bias=True), nn.LeakyReLU(0.2, inplace=True), Reshape(self.lshape), torch.nn.Linear(self.iels, 1024), nn.LeakyReLU(0.2, inplace=True), torch.nn.Linear(1024, 1))
        if not self.wass:
            self.model = nn.Sequential(self.model, torch.nn.Sigmoid())
        initialize_weights(self)
        if self.verbose:
            None
            None

    def forward(self, img):
        validity = self.model(img)
        return validity


class CoupledGenerators(nn.Module):

    def __init__(self):
        super(CoupledGenerators, self).__init__()
        self.init_size = opt.img_size // 4
        self.fc = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))
        self.shared_conv = nn.Sequential(nn.BatchNorm2d(128), nn.Upsample(scale_factor=2), nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128, 0.8), nn.LeakyReLU(0.2, inplace=True), nn.Upsample(scale_factor=2))
        self.G1 = nn.Sequential(nn.Conv2d(128, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(64, opt.channels, 3, stride=1, padding=1), nn.Tanh())
        self.G2 = nn.Sequential(nn.Conv2d(128, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(64, opt.channels, 3, stride=1, padding=1), nn.Tanh())

    def forward(self, noise):
        out = self.fc(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img_emb = self.shared_conv(out)
        img1 = self.G1(img_emb)
        img2 = self.G2(img_emb)
        return img1, img2


class CoupledDiscriminators(nn.Module):

    def __init__(self):
        super(CoupledDiscriminators, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            block.extend([nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)])
            return block
        self.shared_conv = nn.Sequential(*discriminator_block(opt.channels, 16, bn=False), *discriminator_block(16, 32), *discriminator_block(32, 64), *discriminator_block(64, 128))
        ds_size = opt.img_size // 2 ** 4
        self.D1 = nn.Linear(128 * ds_size ** 2, 1)
        self.D2 = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, img1, img2):
        out = self.shared_conv(img1)
        out = out.view(out.shape[0], -1)
        validity1 = self.D1(out)
        out = self.shared_conv(img2)
        out = out.view(out.shape[0], -1)
        validity2 = self.D2(out)
        return validity1, validity2


class GeneratorResNet(nn.Module):

    def __init__(self, img_shape=(3, 128, 128), res_blocks=9, c_dim=5):
        super(GeneratorResNet, self).__init__()
        channels, img_size, _ = img_shape
        model = [nn.Conv2d(channels + c_dim, 64, 7, stride=1, padding=3, bias=False), nn.InstanceNorm2d(64, affine=True, track_running_stats=True), nn.ReLU(inplace=True)]
        curr_dim = 64
        for _ in range(2):
            model += [nn.Conv2d(curr_dim, curr_dim * 2, 4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True), nn.ReLU(inplace=True)]
            curr_dim *= 2
        for _ in range(res_blocks):
            model += [ResidualBlock(curr_dim)]
        for _ in range(2):
            model += [nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True), nn.ReLU(inplace=True)]
            curr_dim = curr_dim // 2
        model += [nn.Conv2d(curr_dim, channels, 7, stride=1, padding=3), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat((x, c), 1)
        return self.model(x)


class GeneratorUNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)
        self.final = nn.Sequential(nn.Upsample(scale_factor=2), nn.ZeroPad2d((1, 0, 1, 0)), nn.Conv2d(128, out_channels, 4, padding=1), nn.Tanh())

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)


class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.feature_extractor(img)


class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)
        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):

    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters))

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x


class GeneratorRRDB(nn.Module):

    def __init__(self, channels, filters=64, num_res_blocks=16, num_upsample=2):
        super(GeneratorRRDB, self).__init__()
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(), nn.PixelShuffle(upscale_factor=2)]
        self.upsampling = nn.Sequential(*upsample_layers)
        self.conv3 = nn.Sequential(nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(), nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


class ContentEncoder(nn.Module):

    def __init__(self, in_channels=3, dim=64, n_residual=3, n_downsample=2):
        super(ContentEncoder, self).__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(in_channels, dim, 7), nn.InstanceNorm2d(dim), nn.ReLU(inplace=True)]
        for _ in range(n_downsample):
            layers += [nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1), nn.InstanceNorm2d(dim * 2), nn.ReLU(inplace=True)]
            dim *= 2
        for _ in range(n_residual):
            layers += [ResidualBlock(dim, norm='in')]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class StyleEncoder(nn.Module):

    def __init__(self, in_channels=3, dim=64, n_downsample=2, style_dim=8):
        super(StyleEncoder, self).__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(in_channels, dim, 7), nn.ReLU(inplace=True)]
        for _ in range(2):
            layers += [nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1), nn.ReLU(inplace=True)]
            dim *= 2
        for _ in range(n_downsample - 2):
            layers += [nn.Conv2d(dim, dim, 4, stride=2, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class AdaptiveInstanceNorm2d(nn.Module):
    """Reference: https://github.com/NVlabs/MUNIT/blob/master/networks.py"""

    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, 'Please assign weight and bias before calling AdaIN!'
        b, c, h, w = x.size()
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, h, w)
        out = F.batch_norm(x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps)
        return out.view(b, c, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-05, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


def Conv3x3BN(in_channels, out_channels, stride, dilation=1):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False), nn.BatchNorm2d(out_channels))


def DSConv(in_channels, out_channels, stride):
    return nn.Sequential(Conv3x3BN(in_channels=in_channels, out_channels=in_channels, stride=stride, groups=in_channels), Conv1x1BNReLU(in_channels=in_channels, out_channels=out_channels))


class Classifier(nn.Module):

    def __init__(self, num_classes=19):
        super(Classifier, self).__init__()
        self.dsConv = nn.Sequential(DSConv(in_channels=128, out_channels=128, stride=1), DSConv(in_channels=128, out_channels=128, stride=1))
        self.conv = Conv3x3BNReLU(in_channels=128, out_channels=num_classes, stride=1)

    def forward(self, x):
        x = self.dsConv(x)
        out = self.conv(x)
        return out


class HourglassModule(nn.Module):

    def __init__(self, nChannels=256, nModules=2, numReductions=4):
        super(HourglassModule, self).__init__()
        self.nChannels = nChannels
        self.nModules = nModules
        self.numReductions = numReductions
        self.residual_block = self._make_residual_layer(self.nModules, self.nChannels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.after_pool_block = self._make_residual_layer(self.nModules, self.nChannels)
        if numReductions > 1:
            self.hourglass_module = HourglassModule(self.nChannels, self.numReductions - 1, self.nModules)
        else:
            self.num1res_block = self._make_residual_layer(self.nModules, self.nChannels)
        self.lowres_block = self._make_residual_layer(self.nModules, self.nChannels)
        self.upsample = nn.Upsample(scale_factor=2)

    def _make_residual_layer(self, nModules, nChannels):
        _residual_blocks = []
        for _ in range(nModules):
            _residual_blocks.append(ResidualBlock(in_channels=nChannels, out_channels=nChannels))
        return nn.Sequential(*_residual_blocks)

    def forward(self, x):
        out1 = self.residual_block(x)
        out2 = self.max_pool(x)
        out2 = self.after_pool_block(out2)
        if self.numReductions > 1:
            out2 = self.hourglass_module(out2)
        else:
            out2 = self.num1res_block(out2)
        out2 = self.lowres_block(out2)
        out2 = self.upsample(out2)
        return out1 + out2


class Hourglass(nn.Module):

    def __init__(self, nJoints):
        super(Hourglass, self).__init__()
        self.first_conv = ConvBNReLU(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.residual_block1 = ResidualBlock(in_channels=64, out_channels=128)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.residual_block2 = ResidualBlock(in_channels=128, out_channels=128)
        self.residual_block3 = ResidualBlock(in_channels=128, out_channels=256)
        self.hourglass_module1 = HourglassModule(nChannels=256, nModules=2, numReductions=4)
        self.hourglass_module2 = HourglassModule(nChannels=256, nModules=2, numReductions=4)
        self.after_hourglass_conv1 = ConvBNReLU(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.proj_conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)
        self.out_conv1 = nn.Conv2d(in_channels=256, out_channels=nJoints, kernel_size=1, stride=1)
        self.remap_conv1 = nn.Conv2d(in_channels=nJoints, out_channels=256, kernel_size=1, stride=1)
        self.after_hourglass_conv2 = ConvBNReLU(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.proj_conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)
        self.out_conv2 = nn.Conv2d(in_channels=256, out_channels=nJoints, kernel_size=1, stride=1)
        self.remap_conv2 = nn.Conv2d(in_channels=nJoints, out_channels=256, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.max_pool(self.residual_block1(self.first_conv(x)))
        x = self.residual_block3(self.residual_block2(x))
        x = self.hourglass_module1(x)
        residual1 = x = self.after_hourglass_conv1(x)
        out1 = self.out_conv1(x)
        residual2 = x = residual1 + self.remap_conv1(out1) + self.proj_conv1(x)
        x = self.hourglass_module2(x)
        x = self.after_hourglass_conv2(x)
        out2 = self.out_conv2(x)
        x = residual2 + self.remap_conv2(out2) + self.proj_conv2(x)
        return out1, out2


class ContextBlock(nn.Module):

    def __init__(self, inplanes, ratio, pooling_type='att', fusion_types=('channel_add',)):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([(f in valid_fusion_types) for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(nn.Conv2d(self.inplanes, self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(nn.Conv2d(self.inplanes, self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            input_x = input_x.view(batch, channel, height * width)
            input_x = input_x.unsqueeze(1)
            context_mask = self.conv_mask(x)
            context_mask = context_mask.view(batch, 1, height * width)
            context_mask = self.softmax(context_mask)
            context_mask = context_mask.unsqueeze(-1)
            context = torch.matmul(input_x, context_mask)
            context = context.view(batch, channel, 1, 1)
        else:
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out


class LBwithGCBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(LBwithGCBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1, stride=1, padding=0)
        self.conv1_bn = nn.BatchNorm2d(planes)
        self.conv1_bn_relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, padding=1)
        self.conv2_bn = nn.BatchNorm2d(planes)
        self.conv2_bn_relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=planes * self.expansion, kernel_size=1, stride=1, padding=0)
        self.conv3_bn = nn.BatchNorm2d(planes * self.expansion)
        self.gcb = ContextBlock(planes * self.expansion, ratio=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1_bn_relu(self.conv1_bn(self.conv1(x)))
        out = self.conv2_bn_relu(self.conv2_bn(self.conv2(out)))
        out = self.conv3_bn(self.conv3(out))
        out = self.gcb(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


def computeGCD(a, b):
    while a != b:
        if a > b:
            a = a - b
        else:
            b = b - a
    return b


def GroupDeconv(inplanes, planes, kernel_size, stride, padding, output_padding):
    groups = computeGCD(inplanes, planes)
    return nn.Sequential(nn.ConvTranspose2d(in_channels=inplanes, out_channels=2 * planes, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups), nn.Conv2d(2 * planes, planes, kernel_size=1, stride=1, padding=0))


class LPN(nn.Module):

    def __init__(self, nJoints):
        super(LPN, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(LBwithGCBlock, 64, 3)
        self.layer2 = self._make_layer(LBwithGCBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(LBwithGCBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(LBwithGCBlock, 512, 3, stride=1)
        self.deconv_layers = self._make_deconv_group_layer()
        self.final_layer = nn.Conv2d(in_channels=self.inplanes, out_channels=nJoints, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_deconv_group_layer(self):
        layers = []
        planes = 256
        for i in range(2):
            planes = planes // 2
            layers.append(GroupDeconv(inplanes=self.inplanes, planes=planes, kernel_size=4, stride=2, padding=1, output_padding=0))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
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
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x


class ResBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class SimpleBaseline(nn.Module):

    def __init__(self, nJoints):
        super(SimpleBaseline, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(ResBlock, 64, 3)
        self.layer2 = self._make_layer(ResBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(ResBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(ResBlock, 512, 3, stride=2)
        self.deconv_layers = self._make_deconv_layer()
        self.final_layer = nn.Conv2d(in_channels=256, out_channels=nJoints, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_deconv_layer(self):
        layers = []
        for i in range(3):
            layers.append(nn.ConvTranspose2d(in_channels=self.inplanes, out_channels=256, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False))
            layers.append(nn.BatchNorm2d(256))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = 256
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
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x


def conf_centernessLayer(in_channels, out_channels):
    return nn.Sequential(Conv3x3ReLU(in_channels=in_channels, out_channels=in_channels), Conv3x3ReLU(in_channels=in_channels, out_channels=in_channels), Conv3x3ReLU(in_channels=in_channels, out_channels=in_channels), Conv3x3ReLU(in_channels=in_channels, out_channels=in_channels), nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))


def locLayer(in_channels, out_channels):
    return nn.Sequential(Conv3x3ReLU(in_channels=in_channels, out_channels=in_channels), Conv3x3ReLU(in_channels=in_channels, out_channels=in_channels), Conv3x3ReLU(in_channels=in_channels, out_channels=in_channels), Conv3x3ReLU(in_channels=in_channels, out_channels=in_channels), nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))


class PolarMask(nn.Module):

    def __init__(self, num_classes=21):
        super(PolarMask, self).__init__()
        self.num_classes = num_classes
        resnet = torchvision.models.resnet50()
        layers = list(resnet.children())
        self.layer1 = nn.Sequential(*layers[:5])
        self.layer2 = nn.Sequential(*layers[5])
        self.layer3 = nn.Sequential(*layers[6])
        self.layer4 = nn.Sequential(*layers[7])
        self.lateral5 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)
        self.lateral4 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.lateral3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.upsample4 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.downsample6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.downsample5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.loc_layer3 = locLayer(in_channels=256, out_channels=36)
        self.conf_centerness_layer3 = conf_centernessLayer(in_channels=256, out_channels=self.num_classes)
        self.loc_layer4 = locLayer(in_channels=256, out_channels=36)
        self.conf_centerness_layer4 = conf_centernessLayer(in_channels=256, out_channels=self.num_classes)
        self.loc_layer5 = locLayer(in_channels=256, out_channels=36)
        self.conf_centerness_layer5 = conf_centernessLayer(in_channels=256, out_channels=self.num_classes)
        self.loc_layer6 = locLayer(in_channels=256, out_channels=36)
        self.conf_centerness_layer6 = conf_centernessLayer(in_channels=256, out_channels=self.num_classes)
        self.loc_layer7 = locLayer(in_channels=256, out_channels=36)
        self.conf_centerness_layer7 = conf_centernessLayer(in_channels=256, out_channels=self.num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        c3 = x = self.layer2(x)
        c4 = x = self.layer3(x)
        c5 = x = self.layer4(x)
        p5 = self.lateral5(c5)
        p4 = self.upsample4(p5) + self.lateral4(c4)
        p3 = self.upsample3(p4) + self.lateral3(c3)
        p6 = self.downsample5(p5)
        p7 = self.downsample6(p6)
        loc3 = self.loc_layer3(p3)
        conf_centerness3 = self.conf_centerness_layer3(p3)
        conf3, centerness3 = conf_centerness3.split([self.num_classes, 1], dim=1)
        loc4 = self.loc_layer4(p4)
        conf_centerness4 = self.conf_centerness_layer4(p4)
        conf4, centerness4 = conf_centerness4.split([self.num_classes, 1], dim=1)
        loc5 = self.loc_layer5(p5)
        conf_centerness5 = self.conf_centerness_layer5(p5)
        conf5, centerness5 = conf_centerness5.split([self.num_classes, 1], dim=1)
        loc6 = self.loc_layer6(p6)
        conf_centerness6 = self.conf_centerness_layer6(p6)
        conf6, centerness6 = conf_centerness6.split([self.num_classes, 1], dim=1)
        loc7 = self.loc_layer7(p7)
        conf_centerness7 = self.conf_centerness_layer7(p7)
        conf7, centerness7 = conf_centerness7.split([self.num_classes, 1], dim=1)
        locs = torch.cat([loc3.permute(0, 2, 3, 1).contiguous().view(loc3.size(0), -1), loc4.permute(0, 2, 3, 1).contiguous().view(loc4.size(0), -1), loc5.permute(0, 2, 3, 1).contiguous().view(loc5.size(0), -1), loc6.permute(0, 2, 3, 1).contiguous().view(loc6.size(0), -1), loc7.permute(0, 2, 3, 1).contiguous().view(loc7.size(0), -1)], dim=1)
        confs = torch.cat([conf3.permute(0, 2, 3, 1).contiguous().view(conf3.size(0), -1), conf4.permute(0, 2, 3, 1).contiguous().view(conf4.size(0), -1), conf5.permute(0, 2, 3, 1).contiguous().view(conf5.size(0), -1), conf6.permute(0, 2, 3, 1).contiguous().view(conf6.size(0), -1), conf7.permute(0, 2, 3, 1).contiguous().view(conf7.size(0), -1)], dim=1)
        centernesses = torch.cat([centerness3.permute(0, 2, 3, 1).contiguous().view(centerness3.size(0), -1), centerness4.permute(0, 2, 3, 1).contiguous().view(centerness4.size(0), -1), centerness5.permute(0, 2, 3, 1).contiguous().view(centerness5.size(0), -1), centerness6.permute(0, 2, 3, 1).contiguous().view(centerness6.size(0), -1), centerness7.permute(0, 2, 3, 1).contiguous().view(centerness7.size(0), -1)], dim=1)
        out = locs, confs, centernesses
        return out


def DW_Conv3x3BNReLU(in_channels, out_channels, stride, groups=1):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


class GhostModule(nn.Module):

    def __init__(self, in_channels, out_channels, s=2, kernel_size=1, stride=1, use_relu=True):
        super(GhostModule, self).__init__()
        intrinsic_channels = out_channels // s
        ghost_channels = intrinsic_channels * (s - 1)
        self.primary_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=intrinsic_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False), nn.BatchNorm2d(intrinsic_channels), nn.ReLU(inplace=True) if use_relu else nn.Sequential())
        self.cheap_op = DW_Conv3x3BNReLU(in_channels=intrinsic_channels, out_channels=ghost_channels, stride=stride, groups=intrinsic_channels)

    def forward(self, x):
        y = self.primary_conv(x)
        z = self.cheap_op(y)
        out = torch.cat([y, z], dim=1)
        return out


class GhostBottleneck(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride, use_se, se_kernel_size=1):
        super(GhostBottleneck, self).__init__()
        self.stride = stride
        self.bottleneck = nn.Sequential(GhostModule(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, use_relu=True), DW_Conv3x3BNReLU(in_channels=mid_channels, out_channels=mid_channels, stride=stride, groups=mid_channels) if self.stride > 1 else nn.Sequential(), SqueezeAndExcite(mid_channels, mid_channels, se_kernel_size) if use_se else nn.Sequential(), GhostModule(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, use_relu=False))
        if self.stride > 1:
            self.shortcut = DW_Conv3x3BNReLU(in_channels=in_channels, out_channels=out_channels, stride=stride)
        else:
            self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.bottleneck(x)
        residual = self.shortcut(x)
        out += residual
        return out


class GhostNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(GhostNet, self).__init__()
        self.first_conv = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU6(inplace=True))
        self.features = nn.Sequential(GhostBottleneck(in_channels=16, mid_channels=16, out_channels=16, kernel_size=3, stride=1, use_se=False), GhostBottleneck(in_channels=16, mid_channels=64, out_channels=24, kernel_size=3, stride=2, use_se=False), GhostBottleneck(in_channels=24, mid_channels=72, out_channels=24, kernel_size=3, stride=1, use_se=False), GhostBottleneck(in_channels=24, mid_channels=72, out_channels=40, kernel_size=5, stride=2, use_se=True, se_kernel_size=28), GhostBottleneck(in_channels=40, mid_channels=120, out_channels=40, kernel_size=5, stride=1, use_se=True, se_kernel_size=28), GhostBottleneck(in_channels=40, mid_channels=120, out_channels=40, kernel_size=5, stride=1, use_se=True, se_kernel_size=28), GhostBottleneck(in_channels=40, mid_channels=240, out_channels=80, kernel_size=3, stride=1, use_se=False), GhostBottleneck(in_channels=80, mid_channels=200, out_channels=80, kernel_size=3, stride=1, use_se=False), GhostBottleneck(in_channels=80, mid_channels=184, out_channels=80, kernel_size=3, stride=2, use_se=False), GhostBottleneck(in_channels=80, mid_channels=184, out_channels=80, kernel_size=3, stride=1, use_se=False), GhostBottleneck(in_channels=80, mid_channels=480, out_channels=112, kernel_size=3, stride=1, use_se=True, se_kernel_size=14), GhostBottleneck(in_channels=112, mid_channels=672, out_channels=112, kernel_size=3, stride=1, use_se=True, se_kernel_size=14), GhostBottleneck(in_channels=112, mid_channels=672, out_channels=160, kernel_size=5, stride=2, use_se=True, se_kernel_size=7), GhostBottleneck(in_channels=160, mid_channels=960, out_channels=160, kernel_size=5, stride=1, use_se=True, se_kernel_size=7), GhostBottleneck(in_channels=160, mid_channels=960, out_channels=160, kernel_size=5, stride=1, use_se=True, se_kernel_size=7))
        self.last_stage = nn.Sequential(nn.Conv2d(in_channels=160, out_channels=960, kernel_size=1, stride=1), nn.BatchNorm2d(960), nn.ReLU6(inplace=True), nn.AvgPool2d(kernel_size=7, stride=1), nn.Conv2d(in_channels=960, out_channels=1280, kernel_size=1, stride=1), nn.ReLU6(inplace=True))
        self.classifier = nn.Linear(in_features=1280, out_features=num_classes)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.features(x)
        x = self.last_stage(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


class MDConv(nn.Module):

    def __init__(self, nchannels, kernel_sizes, stride):
        super(MDConv, self).__init__()
        self.nchannels = nchannels
        self.groups = len(kernel_sizes)
        self.split_channels = [(nchannels // self.groups) for _ in range(self.groups)]
        self.split_channels[0] += nchannels - sum(self.split_channels)
        self.layers = []
        for i in range(self.groups):
            self.layers.append(nn.Conv2d(in_channels=self.split_channels[i], out_channels=self.split_channels[i], kernel_size=kernel_sizes[i], stride=stride, padding=int(kernel_sizes[i] // 2), groups=self.split_channels[i]))

    def forward(self, x):
        split_x = torch.split(x, self.split_channels, dim=1)
        outputs = [layer(sp_x) for layer, sp_x in zip(self.layers, split_x)]
        return torch.cat(outputs, dim=1)


def Conv1x1BNActivation(in_channels, out_channels, activate):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True) if activate == 'relu' else HardSwish())


class MDConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes, stride, expand_ratio, activate='relu', se_ratio=1):
        super(MDConvBlock, self).__init__()
        self.stride = stride
        self.se_ratio = se_ratio
        mid_channels = in_channels * expand_ratio
        self.expand_conv = Conv1x1BNActivation(in_channels, mid_channels, activate)
        self.md_conv = nn.Sequential(MDConv(mid_channels, kernel_sizes, stride), nn.BatchNorm2d(mid_channels), nn.ReLU6(inplace=True) if activate == 'relu' else HardSwish(inplace=True))
        if self.se_ratio > 0:
            self.squeeze_excite = SqueezeAndExcite(mid_channels, in_channels)
        self.projection_conv = Conv1x1BN(mid_channels, out_channels)

    def forward(self, x):
        x = self.expand_conv(x)
        x = self.md_conv(x)
        if self.se_ratio > 0:
            x = self.squeeze_excite(x)
        out = self.projection_conv(x)
        return out


class MixNet(nn.Module):
    mixnet_s = [(16, 16, [3], 1, 1, 'ReLU', 0.0), (16, 24, [3], 2, 6, 'ReLU', 0.0), (24, 24, [3], 1, 3, 'ReLU', 0.0), (24, 40, [3, 5, 7], 2, 6, 'Swish', 0.5), (40, 40, [3, 5], 1, 6, 'Swish', 0.5), (40, 40, [3, 5], 1, 6, 'Swish', 0.5), (40, 40, [3, 5], 1, 6, 'Swish', 0.5), (40, 80, [3, 5, 7], 2, 6, 'Swish', 0.25), (80, 80, [3, 5], 1, 6, 'Swish', 0.25), (80, 80, [3, 5], 1, 6, 'Swish', 0.25), (80, 120, [3, 5, 7], 1, 6, 'Swish', 0.5), (120, 120, [3, 5, 7, 9], 1, 3, 'Swish', 0.5), (120, 120, [3, 5, 7, 9], 1, 3, 'Swish', 0.5), (120, 200, [3, 5, 7, 9, 11], 2, 6, 'Swish', 0.5), (200, 200, [3, 5, 7, 9], 1, 6, 'Swish', 0.5), (200, 200, [3, 5, 7, 9], 1, 6, 'Swish', 0.5)]
    mixnet_m = [(24, 24, [3], 1, 1, 'ReLU', 0.0), (24, 32, [3, 5, 7], 2, 6, 'ReLU', 0.0), (32, 32, [3], 1, 3, 'ReLU', 0.0), (32, 40, [3, 5, 7, 9], 2, 6, 'Swish', 0.5), (40, 40, [3, 5], 1, 6, 'Swish', 0.5), (40, 40, [3, 5], 1, 6, 'Swish', 0.5), (40, 40, [3, 5], 1, 6, 'Swish', 0.5), (40, 80, [3, 5, 7], 2, 6, 'Swish', 0.25), (80, 80, [3, 5, 7, 9], 1, 6, 'Swish', 0.25), (80, 80, [3, 5, 7, 9], 1, 6, 'Swish', 0.25), (80, 80, [3, 5, 7, 9], 1, 6, 'Swish', 0.25), (80, 120, [3], 1, 6, 'Swish', 0.5), (120, 120, [3, 5, 7, 9], 1, 3, 'Swish', 0.5), (120, 120, [3, 5, 7, 9], 1, 3, 'Swish', 0.5), (120, 120, [3, 5, 7, 9], 1, 3, 'Swish', 0.5), (120, 200, [3, 5, 7, 9], 2, 6, 'Swish', 0.5), (200, 200, [3, 5, 7, 9], 1, 6, 'Swish', 0.5), (200, 200, [3, 5, 7, 9], 1, 6, 'Swish', 0.5), (200, 200, [3, 5, 7, 9], 1, 6, 'Swish', 0.5)]

    def __init__(self, type='mixnet_s'):
        super(MixNet, self).__init__()
        if type == 'mixnet_s':
            config = self.mixnet_s
            stem_channels = 16
        elif type == 'mixnet_m':
            config = self.mixnet_m
            stem_channels = 24
        self.stem = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=stem_channels, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(stem_channels), HardSwish(inplace=True))
        layers = []
        for in_channels, out_channels, kernel_sizes, stride, expand_ratio, activate, se_ratio in config:
            layers.append(MDConvBlock(in_channels, out_channels, kernel_sizes=kernel_sizes, stride=stride, expand_ratio=expand_ratio, activate=activate, se_ratio=se_ratio))
        self.bottleneck = nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        out = self.bottleneck(x)
        return out


def BottleneckV1(in_channels, out_channels, stride):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels), nn.BatchNorm2d(in_channels), nn.ReLU6(inplace=True), nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


class MobileNetV1(nn.Module):

    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()
        self.first_conv = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU6(inplace=True))
        self.bottleneck = nn.Sequential(BottleneckV1(32, 64, stride=1), BottleneckV1(64, 128, stride=2), BottleneckV1(128, 128, stride=1), BottleneckV1(128, 256, stride=2), BottleneckV1(256, 256, stride=1), BottleneckV1(256, 512, stride=2), BottleneckV1(512, 512, stride=1), BottleneckV1(512, 512, stride=1), BottleneckV1(512, 512, stride=1), BottleneckV1(512, 512, stride=1), BottleneckV1(512, 512, stride=1), BottleneckV1(512, 1024, stride=2), BottleneckV1(1024, 1024, stride=1))
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.linear = nn.Linear(in_features=1024, out_features=num_classes)
        self.dropout = nn.Dropout(p=0.2)
        self.softmax = nn.Softmax(dim=1)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.bottleneck(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
        out = self.softmax(x)
        return out


def ConvBNActivation(in_channels, out_channels, kernel_size, stride, activate):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, groups=in_channels), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True) if activate == 'relu' else HardSwish())


class SEInvertedBottleneck(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride, activate, use_se, se_kernel_size=1):
        super(SEInvertedBottleneck, self).__init__()
        self.stride = stride
        self.use_se = use_se
        self.conv = Conv1x1BNActivation(in_channels, mid_channels, activate)
        self.depth_conv = ConvBNActivation(mid_channels, mid_channels, kernel_size, stride, activate)
        if self.use_se:
            self.SEblock = SqueezeAndExcite(mid_channels, mid_channels, se_kernel_size)
        self.point_conv = Conv1x1BNActivation(mid_channels, out_channels, activate)
        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        out = self.depth_conv(self.conv(x))
        if self.use_se:
            out = self.SEblock(out)
        out = self.point_conv(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV3(nn.Module):

    def __init__(self, num_classes=1000, type='large'):
        super(MobileNetV3, self).__init__()
        self.type = type
        self.first_conv = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(16), HardSwish(inplace=True))
        if type == 'large':
            self.large_bottleneck = nn.Sequential(SEInvertedBottleneck(in_channels=16, mid_channels=16, out_channels=16, kernel_size=3, stride=1, activate='relu', use_se=False), SEInvertedBottleneck(in_channels=16, mid_channels=64, out_channels=24, kernel_size=3, stride=2, activate='relu', use_se=False), SEInvertedBottleneck(in_channels=24, mid_channels=72, out_channels=24, kernel_size=3, stride=1, activate='relu', use_se=False), SEInvertedBottleneck(in_channels=24, mid_channels=72, out_channels=40, kernel_size=5, stride=2, activate='relu', use_se=True, se_kernel_size=28), SEInvertedBottleneck(in_channels=40, mid_channels=120, out_channels=40, kernel_size=5, stride=1, activate='relu', use_se=True, se_kernel_size=28), SEInvertedBottleneck(in_channels=40, mid_channels=120, out_channels=40, kernel_size=5, stride=1, activate='relu', use_se=True, se_kernel_size=28), SEInvertedBottleneck(in_channels=40, mid_channels=240, out_channels=80, kernel_size=3, stride=1, activate='hswish', use_se=False), SEInvertedBottleneck(in_channels=80, mid_channels=200, out_channels=80, kernel_size=3, stride=1, activate='hswish', use_se=False), SEInvertedBottleneck(in_channels=80, mid_channels=184, out_channels=80, kernel_size=3, stride=2, activate='hswish', use_se=False), SEInvertedBottleneck(in_channels=80, mid_channels=184, out_channels=80, kernel_size=3, stride=1, activate='hswish', use_se=False), SEInvertedBottleneck(in_channels=80, mid_channels=480, out_channels=112, kernel_size=3, stride=1, activate='hswish', use_se=True, se_kernel_size=14), SEInvertedBottleneck(in_channels=112, mid_channels=672, out_channels=112, kernel_size=3, stride=1, activate='hswish', use_se=True, se_kernel_size=14), SEInvertedBottleneck(in_channels=112, mid_channels=672, out_channels=160, kernel_size=5, stride=2, activate='hswish', use_se=True, se_kernel_size=7), SEInvertedBottleneck(in_channels=160, mid_channels=960, out_channels=160, kernel_size=5, stride=1, activate='hswish', use_se=True, se_kernel_size=7), SEInvertedBottleneck(in_channels=160, mid_channels=960, out_channels=160, kernel_size=5, stride=1, activate='hswish', use_se=True, se_kernel_size=7))
            self.large_last_stage = nn.Sequential(nn.Conv2d(in_channels=160, out_channels=960, kernel_size=1, stride=1), nn.BatchNorm2d(960), HardSwish(inplace=True), nn.AvgPool2d(kernel_size=7, stride=1), nn.Conv2d(in_channels=960, out_channels=1280, kernel_size=1, stride=1), HardSwish(inplace=True))
        else:
            self.small_bottleneck = nn.Sequential(SEInvertedBottleneck(in_channels=16, mid_channels=16, out_channels=16, kernel_size=3, stride=2, activate='relu', use_se=True, se_kernel_size=56), SEInvertedBottleneck(in_channels=16, mid_channels=72, out_channels=24, kernel_size=3, stride=2, activate='relu', use_se=False), SEInvertedBottleneck(in_channels=24, mid_channels=88, out_channels=24, kernel_size=3, stride=1, activate='relu', use_se=False), SEInvertedBottleneck(in_channels=24, mid_channels=96, out_channels=40, kernel_size=5, stride=2, activate='hswish', use_se=True, se_kernel_size=14), SEInvertedBottleneck(in_channels=40, mid_channels=240, out_channels=40, kernel_size=5, stride=1, activate='hswish', use_se=True, se_kernel_size=14), SEInvertedBottleneck(in_channels=40, mid_channels=240, out_channels=40, kernel_size=5, stride=1, activate='hswish', use_se=True, se_kernel_size=14), SEInvertedBottleneck(in_channels=40, mid_channels=120, out_channels=48, kernel_size=5, stride=1, activate='hswish', use_se=True, se_kernel_size=14), SEInvertedBottleneck(in_channels=48, mid_channels=144, out_channels=48, kernel_size=5, stride=1, activate='hswish', use_se=True, se_kernel_size=14), SEInvertedBottleneck(in_channels=48, mid_channels=288, out_channels=96, kernel_size=5, stride=2, activate='hswish', use_se=True, se_kernel_size=7), SEInvertedBottleneck(in_channels=96, mid_channels=576, out_channels=96, kernel_size=5, stride=1, activate='hswish', use_se=True, se_kernel_size=7), SEInvertedBottleneck(in_channels=96, mid_channels=576, out_channels=96, kernel_size=5, stride=1, activate='hswish', use_se=True, se_kernel_size=7))
            self.small_last_stage = nn.Sequential(nn.Conv2d(in_channels=96, out_channels=576, kernel_size=1, stride=1), nn.BatchNorm2d(576), HardSwish(inplace=True), nn.AvgPool2d(kernel_size=7, stride=1), nn.Conv2d(in_channels=576, out_channels=1280, kernel_size=1, stride=1), HardSwish(inplace=True))
        self.classifier = nn.Linear(in_features=1280, out_features=num_classes)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        if self.type == 'large':
            x = self.large_bottleneck(x)
            x = self.large_last_stage(x)
        else:
            x = self.small_bottleneck(x)
            x = self.small_last_stage(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


class SandglassBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expansion_factor=6):
        super(SandglassBlock, self).__init__()
        self.stride = stride
        mid_channels = in_channels // expansion_factor
        self.identity = stride == 1 and in_channels == out_channels
        self.bottleneck = nn.Sequential(Conv3x3BNReLU(in_channels, in_channels, 1, groups=in_channels), Conv1x1BN(in_channels, mid_channels), Conv1x1BNReLU(mid_channels, out_channels), Conv3x3BN(out_channels, out_channels, stride, groups=out_channels))

    def forward(self, x):
        out = self.bottleneck(x)
        if self.identity:
            return out + x
        else:
            return out


class MobileNetXt(nn.Module):

    def __init__(self, num_classes=1000):
        super(MobileNetXt, self).__init__()
        self.first_conv = Conv3x3BNReLU(3, 32, 2, groups=1)
        self.layer1 = self.make_layer(in_channels=32, out_channels=96, stride=2, expansion_factor=2, block_num=1)
        self.layer2 = self.make_layer(in_channels=96, out_channels=144, stride=1, expansion_factor=6, block_num=1)
        self.layer3 = self.make_layer(in_channels=144, out_channels=192, stride=2, expansion_factor=6, block_num=3)
        self.layer4 = self.make_layer(in_channels=192, out_channels=288, stride=2, expansion_factor=6, block_num=3)
        self.layer5 = self.make_layer(in_channels=288, out_channels=384, stride=1, expansion_factor=6, block_num=4)
        self.layer6 = self.make_layer(in_channels=384, out_channels=576, stride=2, expansion_factor=6, block_num=4)
        self.layer7 = self.make_layer(in_channels=576, out_channels=960, stride=1, expansion_factor=6, block_num=2)
        self.layer8 = self.make_layer(in_channels=960, out_channels=1280, stride=1, expansion_factor=6, block_num=1)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_features=1280, out_features=num_classes)

    def make_layer(self, in_channels, out_channels, stride, expansion_factor, block_num):
        layers = []
        layers.append(SandglassBlock(in_channels, out_channels, stride, expansion_factor))
        for i in range(1, block_num):
            layers.append(SandglassBlock(out_channels, out_channels, 1, expansion_factor))
        return nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        out = self.linear(x)
        return out


class ShuffleNet(nn.Module):

    def __init__(self, planes, layers, groups, num_classes=1000):
        super(ShuffleNet, self).__init__()
        self.stage1 = nn.Sequential(Conv3x3BNReLU(in_channels=3, out_channels=24, stride=2, groups=1), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.stage2 = self._make_layer(24, planes[0], groups, layers[0], True)
        self.stage3 = self._make_layer(planes[0], planes[1], groups, layers[1], False)
        self.stage4 = self._make_layer(planes[1], planes[2], groups, layers[2], False)
        self.global_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_features=planes[2] * 7 * 7, out_features=num_classes)
        self.init_params()

    def _make_layer(self, in_channels, out_channels, groups, block_num, is_stage2):
        layers = []
        layers.append(ShuffleNetUnits(in_channels=in_channels, out_channels=out_channels, stride=2, groups=1 if is_stage2 else groups))
        for idx in range(1, block_num):
            layers.append(ShuffleNetUnits(in_channels=out_channels, out_channels=out_channels, stride=1, groups=groups))
        return nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        out = self.linear(x)
        return out


class HalfSplit(nn.Module):

    def __init__(self, dim=1):
        super(HalfSplit, self).__init__()
        self.dim = dim

    def forward(self, input):
        splits = torch.chunk(input, 2, dim=self.dim)
        return splits[0], splits[1]


def ConvBN(in_channels, out_channels, kernel_size, stride, padding=1):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding), nn.BatchNorm2d(out_channels))


def SeparableConvolution(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels), nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))


def ReluSeparableConvolution(in_channels, out_channels):
    return nn.Sequential(nn.ReLU6(inplace=True), SeparableConvolution(in_channels, out_channels))


class EntryBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, first_relu=True):
        super(EntryBottleneck, self).__init__()
        mid_channels = out_channels
        self.shortcut = ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2)
        self.bottleneck = nn.Sequential(ReluSeparableConvolution(in_channels=in_channels, out_channels=mid_channels) if first_relu else SeparableConvolution(in_channels=in_channels, out_channels=mid_channels), ReluSeparableConvolution(in_channels=mid_channels, out_channels=out_channels), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        out = self.shortcut(x)
        x = self.bottleneck(x)
        return out + x


class MiddleBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(MiddleBottleneck, self).__init__()
        mid_channels = out_channels
        self.bottleneck = nn.Sequential(ReluSeparableConvolution(in_channels=in_channels, out_channels=mid_channels), ReluSeparableConvolution(in_channels=mid_channels, out_channels=mid_channels), ReluSeparableConvolution(in_channels=mid_channels, out_channels=out_channels))

    def forward(self, x):
        out = self.bottleneck(x)
        return out + x


class ExitBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ExitBottleneck, self).__init__()
        mid_channels = in_channels
        self.shortcut = ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2)
        self.bottleneck = nn.Sequential(ReluSeparableConvolution(in_channels=in_channels, out_channels=mid_channels), ReluSeparableConvolution(in_channels=mid_channels, out_channels=out_channels), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        out = self.shortcut(x)
        x = self.bottleneck(x)
        return out + x


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):

    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None
        rep = []
        filters = in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters
        for i in range(reps - 1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))
        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.fc = nn.Linear(2048, num_classes)

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = nn.ReLU(inplace=True)(features)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def Conv1x1BnRelu(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


def downSampling1(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


def downSampling2(in_channels, out_channels):
    return nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1), downSampling1(in_channels=in_channels, out_channels=out_channels))


def upSampling1(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True), nn.Upsample(scale_factor=2, mode='nearest'))


def upSampling2(in_channels, out_channels):
    return nn.Sequential(upSampling1(in_channels, out_channels), nn.Upsample(scale_factor=2, mode='nearest'))


class ASFF(nn.Module):

    def __init__(self, level, channel1, channel2, channel3, out_channel):
        super(ASFF, self).__init__()
        self.level = level
        funsed_channel = 8
        if self.level == 1:
            self.level2_1 = downSampling1(channel2, channel1)
            self.level3_1 = downSampling2(channel3, channel1)
            self.weight1 = Conv1x1BnRelu(channel1, funsed_channel)
            self.weight2 = Conv1x1BnRelu(channel1, funsed_channel)
            self.weight3 = Conv1x1BnRelu(channel1, funsed_channel)
            self.expand_conv = Conv1x1BnRelu(channel1, out_channel)
        if self.level == 2:
            self.level1_2 = upSampling1(channel1, channel2)
            self.level3_2 = downSampling1(channel3, channel2)
            self.weight1 = Conv1x1BnRelu(channel2, funsed_channel)
            self.weight2 = Conv1x1BnRelu(channel2, funsed_channel)
            self.weight3 = Conv1x1BnRelu(channel2, funsed_channel)
            self.expand_conv = Conv1x1BnRelu(channel2, out_channel)
        if self.level == 3:
            self.level1_3 = upSampling2(channel1, channel3)
            self.level2_3 = upSampling1(channel2, channel3)
            self.weight1 = Conv1x1BnRelu(channel3, funsed_channel)
            self.weight2 = Conv1x1BnRelu(channel3, funsed_channel)
            self.weight3 = Conv1x1BnRelu(channel3, funsed_channel)
            self.expand_conv = Conv1x1BnRelu(channel3, out_channel)
        self.weight_level = nn.Conv2d(funsed_channel * 3, 3, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y, z):
        if self.level == 1:
            level_x = x
            level_y = self.level2_1(y)
            level_z = self.level3_1(z)
        if self.level == 2:
            level_x = self.level1_2(x)
            level_y = y
            level_z = self.level3_2(z)
        if self.level == 3:
            level_x = self.level1_3(x)
            level_y = self.level2_3(y)
            level_z = z
        weight1 = self.weight1(level_x)
        weight2 = self.weight2(level_y)
        weight3 = self.weight3(level_z)
        level_weight = torch.cat((weight1, weight2, weight3), 1)
        weight_level = self.weight_level(level_weight)
        weight_level = self.softmax(weight_level)
        fused_level = level_x * weight_level[:, 0, :, :] + level_y * weight_level[:, 1, :, :] + level_z * weight_level[:, 2, :, :]
        out = self.expand_conv(fused_level)
        return out


class HourglassNetwork(nn.Module):

    def __init__(self):
        super(HourglassNetwork, self).__init__()

    def forward(self, x):
        return out


class PredictionModule(nn.Module):

    def __init__(self):
        super(PredictionModule, self).__init__()

    def forward(self, x):
        return out


class CornerNet(nn.Module):

    def __init__(self):
        super(CornerNet, self).__init__()

    def forward(self, x):
        return out


class FCOS(nn.Module):

    def __init__(self, num_classes=21):
        super(FCOS, self).__init__()
        self.num_classes = num_classes
        resnet = torchvision.models.resnet50()
        layers = list(resnet.children())
        self.layer1 = nn.Sequential(*layers[:5])
        self.layer2 = nn.Sequential(*layers[5])
        self.layer3 = nn.Sequential(*layers[6])
        self.layer4 = nn.Sequential(*layers[7])
        self.lateral5 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)
        self.lateral4 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.lateral3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.upsample4 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.downsample6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.downsample5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.loc_layer3 = locLayer(in_channels=256, out_channels=4)
        self.conf_centerness_layer3 = conf_centernessLayer(in_channels=256, out_channels=self.num_classes + 1)
        self.loc_layer4 = locLayer(in_channels=256, out_channels=4)
        self.conf_centerness_layer4 = conf_centernessLayer(in_channels=256, out_channels=self.num_classes + 1)
        self.loc_layer5 = locLayer(in_channels=256, out_channels=4)
        self.conf_centerness_layer5 = conf_centernessLayer(in_channels=256, out_channels=self.num_classes + 1)
        self.loc_layer6 = locLayer(in_channels=256, out_channels=4)
        self.conf_centerness_layer6 = conf_centernessLayer(in_channels=256, out_channels=self.num_classes + 1)
        self.loc_layer7 = locLayer(in_channels=256, out_channels=4)
        self.conf_centerness_layer7 = conf_centernessLayer(in_channels=256, out_channels=self.num_classes + 1)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        c3 = x = self.layer2(x)
        c4 = x = self.layer3(x)
        c5 = x = self.layer4(x)
        p5 = self.lateral5(c5)
        p4 = self.upsample4(p5) + self.lateral4(c4)
        p3 = self.upsample3(p4) + self.lateral3(c3)
        p6 = self.downsample5(p5)
        p7 = self.downsample6(p6)
        loc3 = self.loc_layer3(p3)
        conf_centerness3 = self.conf_centerness_layer3(p3)
        conf3, centerness3 = conf_centerness3.split([self.num_classes, 1], dim=1)
        loc4 = self.loc_layer4(p4)
        conf_centerness4 = self.conf_centerness_layer4(p4)
        conf4, centerness4 = conf_centerness4.split([self.num_classes, 1], dim=1)
        loc5 = self.loc_layer5(p5)
        conf_centerness5 = self.conf_centerness_layer5(p5)
        conf5, centerness5 = conf_centerness5.split([self.num_classes, 1], dim=1)
        loc6 = self.loc_layer6(p6)
        conf_centerness6 = self.conf_centerness_layer6(p6)
        conf6, centerness6 = conf_centerness6.split([self.num_classes, 1], dim=1)
        loc7 = self.loc_layer7(p7)
        conf_centerness7 = self.conf_centerness_layer7(p7)
        conf7, centerness7 = conf_centerness7.split([self.num_classes, 1], dim=1)
        locs = torch.cat([loc3.permute(0, 2, 3, 1).contiguous().view(loc3.size(0), -1), loc4.permute(0, 2, 3, 1).contiguous().view(loc4.size(0), -1), loc5.permute(0, 2, 3, 1).contiguous().view(loc5.size(0), -1), loc6.permute(0, 2, 3, 1).contiguous().view(loc6.size(0), -1), loc7.permute(0, 2, 3, 1).contiguous().view(loc7.size(0), -1)], dim=1)
        confs = torch.cat([conf3.permute(0, 2, 3, 1).contiguous().view(conf3.size(0), -1), conf4.permute(0, 2, 3, 1).contiguous().view(conf4.size(0), -1), conf5.permute(0, 2, 3, 1).contiguous().view(conf5.size(0), -1), conf6.permute(0, 2, 3, 1).contiguous().view(conf6.size(0), -1), conf7.permute(0, 2, 3, 1).contiguous().view(conf7.size(0), -1)], dim=1)
        centernesses = torch.cat([centerness3.permute(0, 2, 3, 1).contiguous().view(centerness3.size(0), -1), centerness4.permute(0, 2, 3, 1).contiguous().view(centerness4.size(0), -1), centerness5.permute(0, 2, 3, 1).contiguous().view(centerness5.size(0), -1), centerness6.permute(0, 2, 3, 1).contiguous().view(centerness6.size(0), -1), centerness7.permute(0, 2, 3, 1).contiguous().view(centerness7.size(0), -1)], dim=1)
        out = locs, confs, centernesses
        return out


class FPN(nn.Module):

    def __init__(self):
        super(FPN, self).__init__()
        resnet = torchvision.models.resnet50()
        layers = list(resnet.children())
        self.layer1 = nn.Sequential(*layers[:5])
        self.layer2 = nn.Sequential(*layers[5])
        self.layer3 = nn.Sequential(*layers[6])
        self.layer4 = nn.Sequential(*layers[7])
        self.lateral5 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)
        self.lateral4 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.lateral3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.lateral2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.upsample2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.upsample4 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.smooth2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.smooth4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        c2 = x = self.layer1(x)
        c3 = x = self.layer2(x)
        c4 = x = self.layer3(x)
        c5 = x = self.layer4(x)
        p5 = self.lateral5(c5)
        p4 = self.upsample4(p5) + self.lateral4(c4)
        p3 = self.upsample3(p4) + self.lateral3(c3)
        p2 = self.upsample2(p3) + self.lateral2(c2)
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth4(p2)
        return p2, p3, p4, p5


class FisheyeMODNet(nn.Module):

    def __init__(self, groups=1, num_classes=2):
        super(FisheyeMODNet, self).__init__()
        layers = [4, 8, 4]
        self.stage1a = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1), nn.MaxPool2d(kernel_size=2, stride=2))
        self.stage2a = self._make_layer(24, 120, groups, layers[0])
        self.stage1b = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1), nn.MaxPool2d(kernel_size=2, stride=2))
        self.stage2b = self._make_layer(24, 120, groups, layers[0])
        self.stage3 = self._make_layer(240, 480, groups, layers[1])
        self.stage4 = self._make_layer(480, 960, groups, layers[2])
        self.adapt_conv3 = nn.Conv2d(960, num_classes, kernel_size=1)
        self.adapt_conv2 = nn.Conv2d(480, num_classes, kernel_size=1)
        self.adapt_conv1 = nn.Conv2d(240, num_classes, kernel_size=1)
        self.up_sampling3 = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=4, stride=2, padding=1)
        self.up_sampling2 = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=4, stride=2, padding=1)
        self.up_sampling1 = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=16, stride=8, padding=4)
        self.softmax = nn.Softmax(dim=1)
        self.init_params()

    def _make_layer(self, in_channels, out_channels, groups, block_num):
        layers = []
        layers.append(ShuffleNetUnits(in_channels=in_channels, out_channels=out_channels, stride=2, groups=groups))
        for idx in range(1, block_num):
            layers.append(ShuffleNetUnits(in_channels=out_channels, out_channels=out_channels, stride=1, groups=groups))
        return nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        x = self.stage2a(self.stage1a(x))
        y = self.stage2b(self.stage1b(y))
        feature1 = torch.cat([x, y], dim=1)
        feature2 = self.stage3(feature1)
        feature3 = self.stage4(feature2)
        out3 = self.up_sampling3(self.adapt_conv3(feature3))
        out2 = self.up_sampling2(self.adapt_conv2(feature2) + out3)
        out1 = self.up_sampling1(self.adapt_conv1(feature1) + out2)
        out = self.softmax(out1)
        return out


def confLayer(in_channels, out_channels):
    return nn.Sequential(Conv3x3ReLU(in_channels=in_channels, out_channels=in_channels), Conv3x3ReLU(in_channels=in_channels, out_channels=in_channels), Conv3x3ReLU(in_channels=in_channels, out_channels=in_channels), Conv3x3ReLU(in_channels=in_channels, out_channels=in_channels), nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))


class FoveaBox(nn.Module):

    def __init__(self, num_classes=80):
        super(FoveaBox, self).__init__()
        self.num_classes = num_classes
        resnet = torchvision.models.resnet50()
        layers = list(resnet.children())
        self.layer1 = nn.Sequential(*layers[:5])
        self.layer2 = nn.Sequential(*layers[5])
        self.layer3 = nn.Sequential(*layers[6])
        self.layer4 = nn.Sequential(*layers[7])
        self.lateral5 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)
        self.lateral4 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.lateral3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.upsample4 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.downsample6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.downsample6_relu = nn.ReLU6(inplace=True)
        self.downsample5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.loc_layer3 = locLayer(in_channels=256, out_channels=4)
        self.conf_layer3 = confLayer(in_channels=256, out_channels=self.num_classes)
        self.loc_layer4 = locLayer(in_channels=256, out_channels=4)
        self.conf_layer4 = confLayer(in_channels=256, out_channels=self.num_classes)
        self.loc_layer5 = locLayer(in_channels=256, out_channels=4)
        self.conf_layer5 = confLayer(in_channels=256, out_channels=self.num_classes)
        self.loc_layer6 = locLayer(in_channels=256, out_channels=4)
        self.conf_layer6 = confLayer(in_channels=256, out_channels=self.num_classes)
        self.loc_layer7 = locLayer(in_channels=256, out_channels=4)
        self.conf_layer7 = confLayer(in_channels=256, out_channels=self.num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        c3 = x = self.layer2(x)
        c4 = x = self.layer3(x)
        c5 = x = self.layer4(x)
        p5 = self.lateral5(c5)
        p4 = self.upsample4(p5) + self.lateral4(c4)
        p3 = self.upsample3(p4) + self.lateral3(c3)
        p6 = self.downsample5(p5)
        p7 = self.downsample6_relu(self.downsample6(p6))
        loc3 = self.loc_layer3(p3)
        conf3 = self.conf_layer3(p3)
        loc4 = self.loc_layer4(p4)
        conf4 = self.conf_layer4(p4)
        loc5 = self.loc_layer5(p5)
        conf5 = self.conf_layer5(p5)
        loc6 = self.loc_layer6(p6)
        conf6 = self.conf_layer6(p6)
        loc7 = self.loc_layer7(p7)
        conf7 = self.conf_layer7(p7)
        locs = torch.cat([loc3.permute(0, 2, 3, 1).contiguous().view(loc3.size(0), -1), loc4.permute(0, 2, 3, 1).contiguous().view(loc4.size(0), -1), loc5.permute(0, 2, 3, 1).contiguous().view(loc5.size(0), -1), loc6.permute(0, 2, 3, 1).contiguous().view(loc6.size(0), -1), loc7.permute(0, 2, 3, 1).contiguous().view(loc7.size(0), -1)], dim=1)
        confs = torch.cat([conf3.permute(0, 2, 3, 1).contiguous().view(conf3.size(0), -1), conf4.permute(0, 2, 3, 1).contiguous().view(conf4.size(0), -1), conf5.permute(0, 2, 3, 1).contiguous().view(conf5.size(0), -1), conf6.permute(0, 2, 3, 1).contiguous().view(conf6.size(0), -1), conf7.permute(0, 2, 3, 1).contiguous().view(conf7.size(0), -1)], dim=1)
        out = locs, confs
        return out


class RetinaNet(nn.Module):

    def __init__(self, num_classes=80, num_anchores=9):
        super(RetinaNet, self).__init__()
        self.num_classes = num_classes
        resnet = torchvision.models.resnet50()
        layers = list(resnet.children())
        self.layer1 = nn.Sequential(*layers[:5])
        self.layer2 = nn.Sequential(*layers[5])
        self.layer3 = nn.Sequential(*layers[6])
        self.layer4 = nn.Sequential(*layers[7])
        self.lateral5 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)
        self.lateral4 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.lateral3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.upsample4 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.downsample6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.downsample6_relu = nn.ReLU6(inplace=True)
        self.downsample5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.loc_layer3 = locLayer(in_channels=256, out_channels=4 * num_anchores)
        self.conf_layer3 = confLayer(in_channels=256, out_channels=self.num_classes * num_anchores)
        self.loc_layer4 = locLayer(in_channels=256, out_channels=4 * num_anchores)
        self.conf_layer4 = confLayer(in_channels=256, out_channels=self.num_classes * num_anchores)
        self.loc_layer5 = locLayer(in_channels=256, out_channels=4 * num_anchores)
        self.conf_layer5 = confLayer(in_channels=256, out_channels=self.num_classes * num_anchores)
        self.loc_layer6 = locLayer(in_channels=256, out_channels=4 * num_anchores)
        self.conf_layer6 = confLayer(in_channels=256, out_channels=self.num_classes * num_anchores)
        self.loc_layer7 = locLayer(in_channels=256, out_channels=4 * num_anchores)
        self.conf_layer7 = confLayer(in_channels=256, out_channels=self.num_classes * num_anchores)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        c3 = x = self.layer2(x)
        c4 = x = self.layer3(x)
        c5 = x = self.layer4(x)
        p5 = self.lateral5(c5)
        p4 = self.upsample4(p5) + self.lateral4(c4)
        p3 = self.upsample3(p4) + self.lateral3(c3)
        p6 = self.downsample5(p5)
        p7 = self.downsample6_relu(self.downsample6(p6))
        loc3 = self.loc_layer3(p3)
        conf3 = self.conf_layer3(p3)
        loc4 = self.loc_layer4(p4)
        conf4 = self.conf_layer4(p4)
        loc5 = self.loc_layer5(p5)
        conf5 = self.conf_layer5(p5)
        loc6 = self.loc_layer6(p6)
        conf6 = self.conf_layer6(p6)
        loc7 = self.loc_layer7(p7)
        conf7 = self.conf_layer7(p7)
        locs = torch.cat([loc3.permute(0, 2, 3, 1).contiguous().view(loc3.size(0), -1), loc4.permute(0, 2, 3, 1).contiguous().view(loc4.size(0), -1), loc5.permute(0, 2, 3, 1).contiguous().view(loc5.size(0), -1), loc6.permute(0, 2, 3, 1).contiguous().view(loc6.size(0), -1), loc7.permute(0, 2, 3, 1).contiguous().view(loc7.size(0), -1)], dim=1)
        confs = torch.cat([conf3.permute(0, 2, 3, 1).contiguous().view(conf3.size(0), -1), conf4.permute(0, 2, 3, 1).contiguous().view(conf4.size(0), -1), conf5.permute(0, 2, 3, 1).contiguous().view(conf5.size(0), -1), conf6.permute(0, 2, 3, 1).contiguous().view(conf6.size(0), -1), conf7.permute(0, 2, 3, 1).contiguous().view(conf7.size(0), -1)], dim=1)
        out = locs, confs
        return out


def ConvTransBNReLU(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


class SSD(nn.Module):

    def __init__(self, phase='train', num_classes=21):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.detector1 = nn.Sequential(Conv3x3BNReLU(in_channels=3, out_channels=64, stride=1), Conv3x3BNReLU(in_channels=64, out_channels=64, stride=1), nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), Conv3x3BNReLU(in_channels=64, out_channels=128, stride=1), Conv3x3BNReLU(in_channels=128, out_channels=128, stride=1), nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), Conv3x3BNReLU(in_channels=128, out_channels=256, stride=1), Conv3x3BNReLU(in_channels=256, out_channels=256, stride=1), Conv3x3BNReLU(in_channels=256, out_channels=256, stride=1), nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), Conv3x3BNReLU(in_channels=256, out_channels=512, stride=1), Conv3x3BNReLU(in_channels=512, out_channels=512, stride=1), Conv3x3BNReLU(in_channels=512, out_channels=512, stride=1))
        self.detector2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), Conv3x3BNReLU(in_channels=512, out_channels=512, stride=1), Conv3x3BNReLU(in_channels=512, out_channels=512, stride=1), Conv3x3BNReLU(in_channels=512, out_channels=512, stride=1), nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), ConvTransBNReLU(in_channels=512, out_channels=1024, kernel_size=3, stride=2), Conv1x1BNReLU(in_channels=1024, out_channels=1024))
        self.detector3 = nn.Sequential(Conv1x1BNReLU(in_channels=1024, out_channels=256), Conv3x3BNReLU(in_channels=256, out_channels=512, stride=2))
        self.detector4 = nn.Sequential(Conv1x1BNReLU(in_channels=512, out_channels=128), Conv3x3BNReLU(in_channels=128, out_channels=256, stride=2))
        self.detector5 = nn.Sequential(Conv1x1BNReLU(in_channels=256, out_channels=128), Conv3x3ReLU(in_channels=128, out_channels=256, stride=1, padding=0))
        self.detector6 = nn.Sequential(Conv1x1BNReLU(in_channels=256, out_channels=128), Conv3x3ReLU(in_channels=128, out_channels=256, stride=1, padding=0))
        self.loc_layer1 = nn.Conv2d(in_channels=512, out_channels=4 * 4, kernel_size=3, stride=1, padding=1)
        self.conf_layer1 = nn.Conv2d(in_channels=512, out_channels=4 * num_classes, kernel_size=3, stride=1, padding=1)
        self.loc_layer2 = nn.Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, stride=1, padding=1)
        self.conf_layer2 = nn.Conv2d(in_channels=1024, out_channels=6 * num_classes, kernel_size=3, stride=1, padding=1)
        self.loc_layer3 = nn.Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, stride=1, padding=1)
        self.conf_layer3 = nn.Conv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, stride=1, padding=1)
        self.loc_layer4 = nn.Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, stride=1, padding=1)
        self.conf_layer4 = nn.Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, stride=1, padding=1)
        self.loc_layer5 = nn.Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, stride=1, padding=1)
        self.conf_layer5 = nn.Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, stride=1, padding=1)
        self.loc_layer6 = nn.Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, stride=1, padding=1)
        self.conf_layer6 = nn.Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, stride=1, padding=1)
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
        elif self.phase == 'train':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.xavier_normal_(m.weight.data)
                        nn.init.constant_(m.bias, 0)
                    else:
                        nn.init.xavier_normal_(m.weight.data)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feature_map1 = self.detector1(x)
        feature_map2 = self.detector2(feature_map1)
        feature_map3 = self.detector3(feature_map2)
        feature_map4 = self.detector4(feature_map3)
        feature_map5 = self.detector5(feature_map4)
        out = feature_map6 = self.detector6(feature_map5)
        loc1 = self.loc_layer1(feature_map1)
        conf1 = self.conf_layer1(feature_map1)
        loc2 = self.loc_layer2(feature_map2)
        conf2 = self.conf_layer2(feature_map2)
        loc3 = self.loc_layer3(feature_map3)
        conf3 = self.conf_layer3(feature_map3)
        loc4 = self.loc_layer4(feature_map4)
        conf4 = self.conf_layer4(feature_map4)
        loc5 = self.loc_layer5(feature_map5)
        conf5 = self.conf_layer5(feature_map5)
        loc6 = self.loc_layer6(feature_map6)
        conf6 = self.conf_layer6(feature_map6)
        locs = torch.cat([loc1.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), loc2.permute(0, 2, 3, 1).contiguous().view(loc2.size(0), -1), loc3.permute(0, 2, 3, 1).contiguous().view(loc3.size(0), -1), loc4.permute(0, 2, 3, 1).contiguous().view(loc4.size(0), -1), loc5.permute(0, 2, 3, 1).contiguous().view(loc5.size(0), -1), loc6.permute(0, 2, 3, 1).contiguous().view(loc6.size(0), -1)], dim=1)
        confs = torch.cat([conf1.permute(0, 2, 3, 1).contiguous().view(conf1.size(0), -1), conf2.permute(0, 2, 3, 1).contiguous().view(conf2.size(0), -1), conf3.permute(0, 2, 3, 1).contiguous().view(conf3.size(0), -1), conf4.permute(0, 2, 3, 1).contiguous().view(conf4.size(0), -1), conf5.permute(0, 2, 3, 1).contiguous().view(conf5.size(0), -1), conf6.permute(0, 2, 3, 1).contiguous().view(conf6.size(0), -1)], dim=1)
        if self.phase == 'test':
            out = locs.view(locs.size(0), -1, 4), self.softmax(confs.view(confs.size(0), -1, self.num_classes))
        else:
            out = locs.view(locs.size(0), -1, 4), confs.view(confs.size(0), -1, self.num_classes)
        return out


class OSA_module(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, block_nums=5):
        super(OSA_module, self).__init__()
        self._layers = nn.ModuleList()
        self._layers.append(Conv3x3BNReLU(in_channels=in_channels, out_channels=mid_channels, stride=1))
        for idx in range(block_nums - 1):
            self._layers.append(Conv3x3BNReLU(in_channels=mid_channels, out_channels=mid_channels, stride=1))
        self.conv1x1 = Conv1x1BNReLU(in_channels + mid_channels * block_nums, out_channels)

    def forward(self, x):
        outputs = []
        outputs.append(x)
        for _layer in self._layers:
            x = _layer(x)
            outputs.append(x)
        out = torch.cat(outputs, dim=1)
        out = self.conv1x1(out)
        return out


class eSE_Module(nn.Module):

    def __init__(self, channel, ratio=16):
        super(eSE_Module, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=1, padding=0), nn.ReLU(inplace=True), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x)
        z = self.excitation(y)
        return x * z.expand_as(x)


class OSAv2_module(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, block_nums=5):
        super(OSAv2_module, self).__init__()
        self._layers = nn.ModuleList()
        self._layers.append(Conv3x3BNReLU(in_channels=in_channels, out_channels=mid_channels, stride=1))
        for idx in range(block_nums - 1):
            self._layers.append(Conv3x3BNReLU(in_channels=mid_channels, out_channels=mid_channels, stride=1))
        self.conv1x1 = Conv1x1BNReLU(in_channels + mid_channels * block_nums, out_channels)
        self.ese = eSE_Module(out_channels)
        self.pass_conv1x1 = Conv1x1BNReLU(in_channels, out_channels)

    def forward(self, x):
        residual = x
        outputs = []
        outputs.append(x)
        for _layer in self._layers:
            x = _layer(x)
            outputs.append(x)
        out = self.ese(self.conv1x1(torch.cat(outputs, dim=1)))
        return out + self.pass_conv1x1(residual)


class VoVNet(nn.Module):

    def __init__(self, planes, layers, num_classes=2):
        super(VoVNet, self).__init__()
        self.groups = 1
        self.stage1 = nn.Sequential(Conv3x3BNReLU(in_channels=3, out_channels=64, stride=2, groups=self.groups), Conv3x3BNReLU(in_channels=64, out_channels=64, stride=1, groups=self.groups), Conv3x3BNReLU(in_channels=64, out_channels=128, stride=1, groups=self.groups))
        self.stage2 = self._make_layer(planes[0][0], planes[0][1], planes[0][2], layers[0])
        self.stage3 = self._make_layer(planes[1][0], planes[1][1], planes[1][2], layers[1])
        self.stage4 = self._make_layer(planes[2][0], planes[2][1], planes[2][2], layers[2])
        self.stage5 = self._make_layer(planes[3][0], planes[3][1], planes[3][2], layers[3])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_features=planes[3][2], out_features=num_classes)

    def _make_layer(self, in_channels, mid_channels, out_channels, block_num):
        layers = []
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        for idx in range(block_num):
            layers.append(OSAv2_module(in_channels=in_channels, mid_channels=mid_channels, out_channels=out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        out = self.linear(x)
        return out


class SAG_Mask(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SAG_Mask, self).__init__()
        mid_channels = in_channels
        self.fisrt_convs = nn.Sequential(Conv3x3BNReLU(in_channels=in_channels, out_channels=mid_channels, stride=1), Conv3x3BNReLU(in_channels=mid_channels, out_channels=mid_channels, stride=1), Conv3x3BNReLU(in_channels=mid_channels, out_channels=mid_channels, stride=1), Conv3x3BNReLU(in_channels=mid_channels, out_channels=mid_channels, stride=1))
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv3x3 = Conv3x3BNReLU(in_channels=mid_channels * 2, out_channels=mid_channels, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.deconv = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=2, stride=2)
        self.conv1x1 = Conv1x1BN(mid_channels, out_channels)

    def forward(self, x):
        residual = x = self.fisrt_convs(x)
        aggregate = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        sag = self.sigmoid(self.conv3x3(aggregate))
        sag_x = residual + sag * x
        out = self.conv1x1(self.deconv(sag_x))
        return out


class YOLO(nn.Module):

    def __init__(self):
        super(YOLO, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), nn.MaxPool2d(kernel_size=2, stride=2), Conv3x3BNReLU(in_channels=64, out_channels=192), nn.MaxPool2d(kernel_size=2, stride=2), Conv1x1BNReLU(in_channels=192, out_channels=128), Conv3x3BNReLU(in_channels=128, out_channels=256), Conv1x1BNReLU(in_channels=256, out_channels=256), Conv3x3BNReLU(in_channels=256, out_channels=512), nn.MaxPool2d(kernel_size=2, stride=2), Conv1x1BNReLU(in_channels=512, out_channels=256), Conv3x3BNReLU(in_channels=256, out_channels=512), Conv1x1BNReLU(in_channels=512, out_channels=256), Conv3x3BNReLU(in_channels=256, out_channels=512), Conv1x1BNReLU(in_channels=512, out_channels=256), Conv3x3BNReLU(in_channels=256, out_channels=512), Conv1x1BNReLU(in_channels=512, out_channels=256), Conv3x3BNReLU(in_channels=256, out_channels=512), Conv1x1BNReLU(in_channels=512, out_channels=512), Conv3x3BNReLU(in_channels=512, out_channels=1024), nn.MaxPool2d(kernel_size=2, stride=2), Conv1x1BNReLU(in_channels=1024, out_channels=512), Conv3x3BNReLU(in_channels=512, out_channels=1024), Conv1x1BNReLU(in_channels=1024, out_channels=512), Conv3x3BNReLU(in_channels=512, out_channels=1024), Conv3x3BNReLU(in_channels=1024, out_channels=1024), Conv3x3BNReLU(in_channels=1024, out_channels=1024, stride=2), Conv3x3BNReLU(in_channels=1024, out_channels=1024), Conv3x3BNReLU(in_channels=1024, out_channels=1024))
        self.classifier = nn.Sequential(nn.Linear(1024 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 1470))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


def DWConv3x3BNReLU(in_channels, out_channels, stride):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


class EP(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expansion_factor=6):
        super(EP, self).__init__()
        self.stride = stride
        mid_channels = in_channels * expansion_factor
        self.bottleneck = nn.Sequential(Conv1x1BNReLU(in_channels, mid_channels), DWConv3x3BNReLU(mid_channels, mid_channels, stride), Conv1x1BN(mid_channels, out_channels))
        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        out = self.bottleneck(x)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class PEP(nn.Module):

    def __init__(self, in_channels, proj_channels, out_channels, stride, expansion_factor=6):
        super(PEP, self).__init__()
        self.stride = stride
        mid_channels = proj_channels * expansion_factor
        self.bottleneck = nn.Sequential(Conv1x1BNReLU(in_channels, proj_channels), Conv1x1BNReLU(proj_channels, mid_channels), DWConv3x3BNReLU(mid_channels, mid_channels, stride), Conv1x1BN(mid_channels, out_channels))
        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        out = self.bottleneck(x)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class FCA(nn.Module):

    def __init__(self, channel, ratio=8):
        super(FCA, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(nn.Linear(in_features=channel, out_features=channel // ratio), nn.ReLU(inplace=True), nn.Linear(in_features=channel // ratio, out_features=channel), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)


class YOLO_Nano(nn.Module):

    def __init__(self, out_channel=75):
        super(YOLO_Nano, self).__init__()
        self.stage1 = nn.Sequential(Conv3x3BNReLU(in_channels=3, out_channels=12, stride=1), Conv3x3BNReLU(in_channels=12, out_channels=24, stride=2), PEP(in_channels=24, proj_channels=7, out_channels=24, stride=1), EP(in_channels=24, out_channels=70, stride=2), PEP(in_channels=70, proj_channels=25, out_channels=70, stride=1), PEP(in_channels=70, proj_channels=24, out_channels=70, stride=1), EP(in_channels=70, out_channels=150, stride=2), PEP(in_channels=150, proj_channels=56, out_channels=150, stride=1), Conv1x1BNReLU(in_channels=150, out_channels=150), FCA(channel=150, ratio=8), PEP(in_channels=150, proj_channels=73, out_channels=150, stride=1), PEP(in_channels=150, proj_channels=71, out_channels=150, stride=1), PEP(in_channels=150, proj_channels=75, out_channels=150, stride=1))
        self.stage2 = nn.Sequential(EP(in_channels=150, out_channels=325, stride=2))
        self.stage3 = nn.Sequential(PEP(in_channels=325, proj_channels=132, out_channels=325, stride=1), PEP(in_channels=325, proj_channels=124, out_channels=325, stride=1), PEP(in_channels=325, proj_channels=141, out_channels=325, stride=1), PEP(in_channels=325, proj_channels=140, out_channels=325, stride=1), PEP(in_channels=325, proj_channels=137, out_channels=325, stride=1), PEP(in_channels=325, proj_channels=135, out_channels=325, stride=1), PEP(in_channels=325, proj_channels=133, out_channels=325, stride=1), PEP(in_channels=325, proj_channels=140, out_channels=325, stride=1))
        self.stage4 = nn.Sequential(EP(in_channels=325, out_channels=545, stride=2), PEP(in_channels=545, proj_channels=276, out_channels=545, stride=1), Conv1x1BNReLU(in_channels=545, out_channels=230), EP(in_channels=230, out_channels=489, stride=1), PEP(in_channels=489, proj_channels=213, out_channels=469, stride=1), Conv1x1BNReLU(in_channels=469, out_channels=189))
        self.stage5 = nn.Sequential(Conv1x1BNReLU(in_channels=189, out_channels=105), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.stage6 = nn.Sequential(PEP(in_channels=105 + 325, proj_channels=113, out_channels=325, stride=1), PEP(in_channels=325, proj_channels=99, out_channels=207, stride=1), Conv1x1BNReLU(in_channels=207, out_channels=98))
        self.stage7 = nn.Sequential(Conv1x1BNReLU(in_channels=98, out_channels=47), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.out_stage1 = nn.Sequential(PEP(in_channels=150 + 47, proj_channels=58, out_channels=122, stride=1), PEP(in_channels=122, proj_channels=52, out_channels=87, stride=1), PEP(in_channels=87, proj_channels=47, out_channels=93, stride=1), Conv1x1BNReLU(in_channels=93, out_channels=out_channel))
        self.out_stage2 = nn.Sequential(EP(in_channels=98, out_channels=183, stride=1), Conv1x1BNReLU(in_channels=183, out_channels=out_channel))
        self.out_stage3 = nn.Sequential(EP(in_channels=189, out_channels=462, stride=1), Conv1x1BNReLU(in_channels=462, out_channels=out_channel))

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        x6 = self.stage6(torch.cat([x3, x5], dim=1))
        x7 = self.stage7(x6)
        out1 = self.out_stage1(torch.cat([x1, x7], dim=1))
        out2 = self.out_stage2(x6)
        out3 = self.out_stage3(x4)
        return out1, out2, out3


class Residual(nn.Module):

    def __init__(self, nchannels):
        super(Residual, self).__init__()
        mid_channels = nchannels // 2
        self.conv1x1 = Conv1x1BNReLU(in_channels=nchannels, out_channels=mid_channels)
        self.conv3x3 = Conv3x3BNReLU(in_channels=mid_channels, out_channels=nchannels)

    def forward(self, x):
        out = self.conv3x3(self.conv1x1(x))
        return out + x


class Darknet19(nn.Module):

    def __init__(self, num_classes=1000):
        super(Darknet19, self).__init__()
        self.first_conv = Conv3x3BNReLU(in_channels=3, out_channels=32)
        self.block1 = self._make_layers(in_channels=32, out_channels=64, block_num=1)
        self.block2 = self._make_layers(in_channels=64, out_channels=128, block_num=2)
        self.block3 = self._make_layers(in_channels=128, out_channels=256, block_num=8)
        self.block4 = self._make_layers(in_channels=256, out_channels=512, block_num=8)
        self.block5 = self._make_layers(in_channels=512, out_channels=1024, block_num=4)
        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.linear = nn.Linear(in_features=1024, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _make_layers(self, in_channels, out_channels, block_num):
        _layers = []
        _layers.append(Conv3x3BNReLU(in_channels=in_channels, out_channels=out_channels, stride=2))
        for _ in range(block_num):
            _layers.append(Residual(nchannels=out_channels))
        return nn.Sequential(*_layers)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        out = self.softmax(x)
        return out


class BatchNorm(nn.Module):

    def forward(self, x):
        return 2 * x - 1


class DynamicReLU_A(nn.Module):

    def __init__(self, channels, K=2, ratio=6):
        super(DynamicReLU_A, self).__init__()
        mid_channels = 2 * K
        self.K = K
        self.lambdas = torch.Tensor([1.0] * K + [0.5] * K).float()
        self.init_v = torch.Tensor([1.0] + [0.0] * (2 * K - 1)).float()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.dynamic = nn.Sequential(nn.Linear(in_features=channels, out_features=channels // ratio), nn.ReLU(inplace=True), nn.Linear(in_features=channels // ratio, out_features=mid_channels), nn.Sigmoid(), BatchNorm())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        z = self.dynamic(y)
        relu_coefs = z.view(-1, 2 * self.K) * self.lambdas + self.init_v
        x_perm = x.transpose(0, -1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :self.K] + relu_coefs[:, self.K:]
        output = torch.max(output, dim=-1)[0].transpose(0, -1)
        return output


class DynamicReLU_B(nn.Module):

    def __init__(self, channels, K=2, ratio=6):
        super(DynamicReLU_B, self).__init__()
        mid_channels = 2 * K * channels
        self.K = K
        self.channels = channels
        self.lambdas = torch.Tensor([1.0] * K + [0.5] * K).float()
        self.init_v = torch.Tensor([1.0] + [0.0] * (2 * K - 1)).float()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.dynamic = nn.Sequential(nn.Linear(in_features=channels, out_features=channels // ratio), nn.ReLU(inplace=True), nn.Linear(in_features=channels // ratio, out_features=mid_channels), nn.Sigmoid(), BatchNorm())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        z = self.dynamic(y)
        relu_coefs = z.view(-1, self.channels, 2 * self.K) * self.lambdas + self.init_v
        x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :, :self.K] + relu_coefs[:, :, self.K:]
        output = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)
        return output


class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3Plus(nn.Module):

    def __init__(self, num_classes=None):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = ResNet('resnet50', None)
        self.aspp = ASPP(inplanes=self.backbone.final_out_channels)
        self.decoder = Decoder(self.num_classes, self.backbone.low_level_inplanes)

    def forward(self, imgs, labels=None, mode='infer', **kwargs):
        x, low_level_feat = self.backbone(imgs)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        outputs = F.interpolate(x, size=imgs.size()[2:], mode='bilinear', align_corners=True)
        return outputs


class InitialBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(InitialBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.PReLU()

    def forward(self, x):
        return self.relu(self.bn(torch.cat([self.conv(x), self.pool(x)], dim=1)))


def AsymmetricConv(channels, stride, is_relu=False):
    return nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=[5, 1], stride=stride, padding=[2, 0], bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True) if is_relu else nn.PReLU(), nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=[1, 5], stride=stride, padding=[0, 2], bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True) if is_relu else nn.PReLU())


class RegularBottleneck(nn.Module):

    def __init__(self, in_places, places, stride=1, expansion=4, dilation=1, is_relu=False, asymmetric=False, p=0.01):
        super(RegularBottleneck, self).__init__()
        mid_channels = in_places // expansion
        self.bottleneck = nn.Sequential(Conv1x1BNReLU(in_places, mid_channels, False), AsymmetricConv(mid_channels, 1, is_relu) if asymmetric else Conv3x3BNReLU(mid_channels, mid_channels, 1, dilation, is_relu), Conv1x1BNReLU(mid_channels, places, is_relu), nn.Dropout2d(p=p))
        self.relu = nn.ReLU(inplace=True) if is_relu else nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        out += residual
        out = self.relu(out)
        return out


def Conv2x2BNReLU(in_channels, out_channels, is_relu=True):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True) if is_relu else nn.PReLU())


class DownBottleneck(nn.Module):

    def __init__(self, in_places, places, stride=2, expansion=4, is_relu=False, p=0.01):
        super(DownBottleneck, self).__init__()
        mid_channels = in_places // expansion
        self.bottleneck = nn.Sequential(Conv2x2BNReLU(in_places, mid_channels, is_relu), Conv3x3BNReLU(mid_channels, mid_channels, 1, 1, is_relu), Conv1x1BNReLU(mid_channels, places, is_relu), nn.Dropout2d(p=p))
        self.downsample = nn.MaxPool2d(3, stride=stride, padding=1, return_indices=True)
        self.relu = nn.ReLU(inplace=True) if is_relu else nn.PReLU()

    def forward(self, x):
        out = self.bottleneck(x)
        residual, indices = self.downsample(x)
        n, ch, h, w = out.size()
        ch_res = residual.size()[1]
        padding = torch.zeros(n, ch - ch_res, h, w)
        residual = torch.cat((residual, padding), 1)
        out += residual
        out = self.relu(out)
        return out, indices


def TransposeConv3x3BNReLU(in_channels, out_channels, stride=2, is_relu=True):
    return nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True) if is_relu else nn.PReLU())


class UpBottleneck(nn.Module):

    def __init__(self, in_places, places, stride=2, expansion=4, is_relu=True, p=0.01):
        super(UpBottleneck, self).__init__()
        mid_channels = in_places // expansion
        self.bottleneck = nn.Sequential(Conv1x1BNReLU(in_places, mid_channels, is_relu), TransposeConv3x3BNReLU(mid_channels, mid_channels, stride, is_relu), Conv1x1BNReLU(mid_channels, places, is_relu), nn.Dropout2d(p=p))
        self.upsample_conv = Conv1x1BN(in_places, places)
        self.upsample_unpool = nn.MaxUnpool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True) if is_relu else nn.PReLU()

    def forward(self, x, indices):
        out = self.bottleneck(x)
        residual = self.upsample_conv(x)
        residual = self.upsample_unpool(residual, indices)
        out += residual
        out = self.relu(out)
        return out


class ENet(nn.Module):

    def __init__(self, num_classes):
        super(ENet, self).__init__()
        self.initialBlock = InitialBlock(3, 16)
        self.stage1_1 = DownBottleneck(16, 64, 2)
        self.stage1_2 = nn.Sequential(RegularBottleneck(64, 64, 1), RegularBottleneck(64, 64, 1), RegularBottleneck(64, 64, 1), RegularBottleneck(64, 64, 1))
        self.stage2_1 = DownBottleneck(64, 128, 2)
        self.stage2_2 = nn.Sequential(RegularBottleneck(128, 128, 1), RegularBottleneck(128, 128, 1, dilation=2), RegularBottleneck(128, 128, 1, asymmetric=True), RegularBottleneck(128, 128, 1, dilation=4), RegularBottleneck(128, 128, 1), RegularBottleneck(128, 128, 1, dilation=8), RegularBottleneck(128, 128, 1, asymmetric=True), RegularBottleneck(128, 128, 1, dilation=16))
        self.stage3 = nn.Sequential(RegularBottleneck(128, 128, 1), RegularBottleneck(128, 128, 1, dilation=2), RegularBottleneck(128, 128, 1, asymmetric=True), RegularBottleneck(128, 128, 1, dilation=4), RegularBottleneck(128, 128, 1), RegularBottleneck(128, 128, 1, dilation=8), RegularBottleneck(128, 128, 1, asymmetric=True), RegularBottleneck(128, 128, 1, dilation=16))
        self.stage4_1 = UpBottleneck(128, 64, 2, is_relu=True)
        self.stage4_2 = nn.Sequential(RegularBottleneck(64, 64, 1, is_relu=True), RegularBottleneck(64, 64, 1, is_relu=True))
        self.stage5_1 = UpBottleneck(64, 16, 2, is_relu=True)
        self.stage5_2 = RegularBottleneck(16, 16, 1, is_relu=True)
        self.final_conv = nn.ConvTranspose2d(in_channels=16, out_channels=num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x):
        x = self.initialBlock(x)
        x, indices1 = self.stage1_1(x)
        x = self.stage1_2(x)
        x, indices2 = self.stage2_1(x)
        x = self.stage2_2(x)
        x = self.stage3(x)
        x = self.stage4_1(x, indices2)
        x = self.stage4_2(x)
        x = self.stage5_1(x, indices1)
        x = self.stage5_2(x)
        out = self.final_conv(x)
        return out


class FCN8s(nn.Module):

    def __init__(self, num_classes):
        super(FCN8s, self).__init__()
        vgg = torchvision.models.vgg16()
        features = list(vgg.features.children())
        self.padd = nn.ZeroPad2d([100, 100, 100, 100])
        self.pool3 = nn.Sequential(*features[:17])
        self.pool4 = nn.Sequential(*features[17:24])
        self.pool5 = nn.Sequential(*features[24:])
        self.pool3_conv1x1 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.pool4_conv1x1 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.output5 = nn.Sequential(nn.Conv2d(512, 4096, kernel_size=7), nn.ReLU(inplace=True), nn.Dropout(), nn.Conv2d(4096, 4096, kernel_size=1), nn.ReLU(inplace=True), nn.Dropout(), nn.Conv2d(4096, num_classes, kernel_size=1))
        self.up_pool3_out = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8)
        self.up_pool4_out = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2)
        self.up_pool5_out = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2)

    def forward(self, x):
        _, _, w, h = x.size()
        x = self.padd(x)
        pool3 = self.pool3(x)
        pool4 = self.pool4(pool3)
        pool5 = self.pool5(pool4)
        output5 = self.up_pool5_out(self.output5(pool5))
        pool4_out = self.pool4_conv1x1(0.01 * pool4)
        output4 = self.up_pool4_out(pool4_out[:, :, 5:5 + output5.size()[2], 5:5 + output5.size()[3]] + output5)
        pool3_out = self.pool3_conv1x1(0.0001 * pool3)
        output3 = self.up_pool3_out(pool3_out[:, :, 9:9 + output4.size()[2], 9:9 + output4.size()[3]] + output4)
        out = self.up_pool3_out(output3)
        out = out[:, :, 31:31 + h, 31:31 + w].contiguous()
        return out


class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels):
        super(PyramidPooling, self).__init__()
        mid_channels = in_channels // 4
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1)
        self.out = Conv3x3BNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


class LearningToDownsample(nn.Module):

    def __init__(self):
        super(LearningToDownsample, self).__init__()
        self.conv = Conv3x3BNReLU(in_channels=3, out_channels=32, stride=2)
        self.dsConv1 = DSConv(in_channels=32, out_channels=48, stride=2)
        self.dsConv2 = DSConv(in_channels=48, out_channels=64, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsConv1(x)
        out = self.dsConv2(x)
        return out


class GlobalFeatureExtractor(nn.Module):

    def __init__(self):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(inplanes=64, planes=64, blocks_num=3, stride=2)
        self.bottleneck2 = self._make_layer(inplanes=64, planes=96, blocks_num=3, stride=2)
        self.bottleneck3 = self._make_layer(inplanes=96, planes=128, blocks_num=3, stride=1)
        self.ppm = PyramidPooling(in_channels=128, out_channels=128)

    def _make_layer(self, inplanes, planes, blocks_num, stride=1):
        layers = []
        layers.append(InvertedResidual(inplanes, planes, stride))
        for i in range(1, blocks_num):
            layers.append(InvertedResidual(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        out = self.ppm(x)
        return out


class FeatureFusionModule(nn.Module):

    def __init__(self, num_classes=20):
        super(FeatureFusionModule, self).__init__()
        self.dsConv1 = nn.Sequential(DSConv(in_channels=128, out_channels=128, stride=1), Conv3x3BN(in_channels=128, out_channels=128, stride=1))
        self.dsConv2 = DSConv(in_channels=64, out_channels=128, stride=1)

    def forward(self, x, y):
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x = self.dsConv1(x)
        return x + self.dsConv2(y)


class FastSCNN(nn.Module):

    def __init__(self, num_classes):
        super(FastSCNN, self).__init__()
        self.learning_to_downsample = LearningToDownsample()
        self.global_feature_extractor = GlobalFeatureExtractor()
        self.feature_fusion = FeatureFusionModule()
        self.classifier = Classifier(num_classes=num_classes)

    def forward(self, x):
        y = self.learning_to_downsample(x)
        x = self.global_feature_extractor(y)
        x = self.feature_fusion(x, y)
        out = self.classifier(x)
        return out


class CascadeFeatureFusion(nn.Module):

    def __init__(self, low_channels, high_channels, out_channels, num_classes):
        super(CascadeFeatureFusion, self).__init__()
        self.conv_low = Conv3x3BNReLU(low_channels, out_channels, 1, dilation=2)
        self.conv_high = Conv3x3BNReLU(high_channels, out_channels, 1, dilation=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_low_cls = nn.Conv2d(out_channels, num_classes, 1, bias=False)

    def forward(self, x_low, x_high):
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        out = self.relu(x_low + x_high)
        x_low_cls = self.conv_low_cls(x_low)
        return out, x_low_cls


class Backbone(nn.Module):

    def __init__(self, pyramids=[1, 2, 3, 6]):
        super(Backbone, self).__init__()
        self.pretrained = torchvision.models.resnet50(pretrained=True)

    def forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        return c1, c2, c3, c4


class PyramidPoolingModule(nn.Module):

    def __init__(self, pyramids=[1, 2, 3, 6]):
        super(PyramidPoolingModule, self).__init__()
        self.pyramids = pyramids

    def forward(self, x):
        feat = x
        height, width = x.shape[2:]
        for bin_size in self.pyramids:
            feat_x = F.adaptive_avg_pool2d(x, output_size=bin_size)
            feat_x = F.interpolate(feat_x, size=(height, width), mode='bilinear', align_corners=True)
            feat = feat + feat_x
        return feat


class ICNet(nn.Module):

    def __init__(self, num_classes):
        super(ICNet, self).__init__()
        self.conv_sub1 = nn.Sequential(Conv3x3BNReLU(3, 32, 2), Conv3x3BNReLU(32, 32, 2), Conv3x3BNReLU(32, 64, 2))
        self.backbone = Backbone()
        self.ppm = PyramidPoolingModule()
        self.cff_12 = CascadeFeatureFusion(128, 64, 128, num_classes)
        self.cff_24 = CascadeFeatureFusion(2048, 512, 128, num_classes)
        self.conv_cls = nn.Conv2d(128, num_classes, 1, bias=False)

    def forward(self, x):
        x_sub1 = self.conv_sub1(x)
        x_sub2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        _, x_sub2, _, _ = self.backbone(x_sub2)
        x_sub4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')
        _, _, _, x_sub4 = self.backbone(x_sub4)
        x_sub4 = self.ppm(x_sub4)
        outs = list()
        x_cff_24, x_24_cls = self.cff_24(x_sub4, x_sub2)
        outs.append(x_24_cls)
        x_cff_12, x_12_cls = self.cff_12(x_cff_24, x_sub1)
        outs.append(x_12_cls)
        up_x2 = F.interpolate(x_cff_12, scale_factor=2, mode='bilinear')
        up_x2 = self.conv_cls(up_x2)
        outs.append(up_x2)
        up_x8 = F.interpolate(up_x2, scale_factor=4, mode='bilinear')
        outs.append(up_x8)
        outs.reverse()
        return outs


def ConvReLU(in_channels, out_channels, kernel_size, stride, padding, dilation=[1, 1], groups=1):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False), nn.ReLU6(inplace=True))


class SS_nbt(nn.Module):

    def __init__(self, channels, dilation=1, groups=4):
        super(SS_nbt, self).__init__()
        mid_channels = channels // 2
        self.half_split = HalfSplit(dim=1)
        self.first_bottleneck = nn.Sequential(ConvReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=[3, 1], stride=1, padding=[1, 0]), ConvBNReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=[1, 3], stride=1, padding=[0, 1]), ConvReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=[3, 1], stride=1, dilation=[dilation, 1], padding=[dilation, 0]), ConvBNReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=[1, 3], stride=1, dilation=[1, dilation], padding=[0, dilation]))
        self.second_bottleneck = nn.Sequential(ConvReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=[1, 3], stride=1, padding=[0, 1]), ConvBNReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=[3, 1], stride=1, padding=[1, 0]), ConvReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=[1, 3], stride=1, dilation=[1, dilation], padding=[0, dilation]), ConvBNReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=[3, 1], stride=1, dilation=[dilation, 1], padding=[dilation, 0]))
        self.channelShuffle = ChannelShuffle(groups)

    def forward(self, x):
        x1, x2 = self.half_split(x)
        x1 = self.first_bottleneck(x1)
        x2 = self.second_bottleneck(x2)
        out = torch.cat([x1, x2], dim=1)
        return self.channelShuffle(out + x)


class DownSampling(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownSampling, self).__init__()
        mid_channels = out_channels - in_channels
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.maxpool(x)
        output = torch.cat([x1, x2], 1)
        return self.relu(self.bn(output))


class APN(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(APN, self).__init__()
        self.conv1_1 = ConvBNReLU(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = Conv1x1BNReLU(in_channels=in_channels, out_channels=out_channels)
        self.conv2_1 = ConvBNReLU(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=2, padding=2)
        self.conv2_2 = Conv1x1BNReLU(in_channels=in_channels, out_channels=out_channels)
        self.conv3 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=in_channels, kernel_size=7, stride=2, padding=3), Conv1x1BNReLU(in_channels=in_channels, out_channels=out_channels))
        self.conv1 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1), Conv1x1BNReLU(in_channels=in_channels, out_channels=out_channels))
        self.branch2 = Conv1x1BNReLU(in_channels=in_channels, out_channels=out_channels)
        self.branch3 = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=1), nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        _, _, h, w = x.shape
        x1 = self.conv1_1(x)
        x2 = self.conv2_1(x1)
        x3 = self.conv3(x2)
        x3 = F.interpolate(x3, size=(h // 4, w // 4), mode='bilinear', align_corners=True)
        x2 = self.conv2_2(x2) + x3
        x2 = F.interpolate(x2, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        x1 = self.conv1_2(x1) + x2
        out1 = F.interpolate(x1, size=(h, w), mode='bilinear', align_corners=True)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out3 = F.interpolate(out3, size=(h, w), mode='bilinear', align_corners=True)
        return out1 * out2 + out3


class LEDnet(nn.Module):

    def __init__(self, num_classes=20):
        super(LEDnet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(in_channels=128, num_classes=num_classes)

    def forward(self, x):
        x = self.encoder(x)
        out = self.decoder(x)
        return out


class LW_Network(nn.Module):

    def __init__(self, num_classes=2):
        super(LW_Network, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_classes)

    def forward(self, x):
        x1, x2 = self.encoder(x)
        out = self.decoder(x2, x1)
        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(Conv3x3BNReLU(in_channels, out_channels, stride=1), Conv3x3BNReLU(out_channels, out_channels, stride=1))

    def forward(self, x):
        return self.double_conv(x)


class TripleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 3"""

    def __init__(self, in_channels, out_channels, reverse=False):
        super().__init__()
        if reverse:
            self.triple_conv = nn.Sequential(Conv3x3BNReLU(in_channels, in_channels, stride=1), Conv3x3BNReLU(in_channels, in_channels, stride=1), Conv3x3BNReLU(in_channels, out_channels, stride=1))
        else:
            self.triple_conv = nn.Sequential(Conv3x3BNReLU(in_channels, out_channels, stride=1), Conv3x3BNReLU(out_channels, out_channels, stride=1), Conv3x3BNReLU(out_channels, out_channels, stride=1))

    def forward(self, x):
        return self.triple_conv(x)


class SegNet(nn.Module):
    """
        SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
        https://arxiv.org/pdf/1511.00561.pdf
    """

    def __init__(self, classes=19):
        super(SegNet, self).__init__()
        self.conv_down1 = DoubleConv(3, 64)
        self.conv_down2 = DoubleConv(64, 128)
        self.conv_down3 = TripleConv(128, 256)
        self.conv_down4 = TripleConv(256, 512)
        self.conv_down5 = TripleConv(512, 512)
        self.conv_up5 = TripleConv(512, 512, reverse=True)
        self.conv_up4 = TripleConv(512, 256, reverse=True)
        self.conv_up3 = TripleConv(256, 128, reverse=True)
        self.conv_up2 = DoubleConv(128, 64, reverse=True)
        self.conv_up1 = Conv3x3BNReLU(64, 64, stride=1)
        self.outconv = nn.Conv2d(64, classes, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv_down1(x)
        x1_size = x1.size()
        x1p, id1 = F.max_pool2d(x1, kernel_size=2, stride=2, return_indices=True)
        x2 = self.conv_down2(x1p)
        x2_size = x2.size()
        x2p, id2 = F.max_pool2d(x2, kernel_size=2, stride=2, return_indices=True)
        x3 = self.conv_down3(x2p)
        x3_size = x3.size()
        x3p, id3 = F.max_pool2d(x3, kernel_size=2, stride=2, return_indices=True)
        x4 = self.conv_down4(x3p)
        x4_size = x4.size()
        x4p, id4 = F.max_pool2d(x4, kernel_size=2, stride=2, return_indices=True)
        x5 = self.conv_down5(x4p)
        x5_size = x5.size()
        x5p, id5 = F.max_pool2d(x5, kernel_size=2, stride=2, return_indices=True)
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2, output_size=x5_size)
        x5d = self.conv_up5(x5d)
        x4d = F.max_unpool2d(x5d, id4, kernel_size=2, stride=2, output_size=x4_size)
        x4d = self.conv_up4(x4d)
        x3d = F.max_unpool2d(x4d, id3, kernel_size=2, stride=2, output_size=x3_size)
        x3d = self.conv_up3(x3d)
        x2d = F.max_unpool2d(x3d, id2, kernel_size=2, stride=2, output_size=x2_size)
        x2d = self.conv_up2(x2d)
        x1d = F.max_unpool2d(x2d, id1, kernel_size=2, stride=2, output_size=x1_size)
        x1d = self.conv_up1(x1d)
        out = self.outconv(x1d)
        return out


class DownConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=stride)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.pool(self.double_conv(x))


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.reduce = Conv1x1BNReLU(in_channels, in_channels // 2)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(self.reduce(x1))
        _, channel1, height1, width1 = x1.size()
        _, channel2, height2, width2 = x2.size()
        diffY = height2 - height1
        diffX = width2 - width1
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, num_classes):
        super(UNet, self).__init__()
        bilinear = True
        self.conv = DoubleConv(3, 64)
        self.down1 = DownConv(64, 128)
        self.down2 = DownConv(128, 256)
        self.down3 = DownConv(256, 512)
        self.down4 = DownConv(512, 1024)
        self.up1 = UpConv(1024, 512, bilinear)
        self.up2 = UpConv(512, 256, bilinear)
        self.up3 = UpConv(256, 128, bilinear)
        self.up4 = UpConv(128, 64, bilinear)
        self.outconv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        xx = self.up1(x5, x4)
        xx = self.up2(xx, x3)
        xx = self.up3(xx, x2)
        xx = self.up4(xx, x1)
        outputs = self.outconv(xx)
        return outputs


class BNInception(nn.Module):

    def __init__(self, num_classes=1000):
        super(BNInception, self).__init__()
        inplace = True
        self.conv1_7x7_s2 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.conv1_7x7_s2_bn = nn.BatchNorm2d(64, affine=True)
        self.conv1_relu_7x7 = nn.ReLU(inplace)
        self.pool1_3x3_s2 = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.conv2_3x3_reduce = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.conv2_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.conv2_relu_3x3_reduce = nn.ReLU(inplace)
        self.conv2_3x3 = nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_3x3_bn = nn.BatchNorm2d(192, affine=True)
        self.conv2_relu_3x3 = nn.ReLU(inplace)
        self.pool2_3x3_s2 = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.inception_3a_1x1 = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_1x1_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3a_relu_1x1 = nn.ReLU(inplace)
        self.inception_3a_3x3_reduce = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_3x3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_3x3_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3a_relu_3x3 = nn.ReLU(inplace)
        self.inception_3a_double_3x3_reduce = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_double_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_1_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3a_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_2_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3a_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_3a_pool_proj = nn.Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_pool_proj_bn = nn.BatchNorm2d(32, affine=True)
        self.inception_3a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_3b_1x1 = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_1x1_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3b_relu_1x1 = nn.ReLU(inplace)
        self.inception_3b_3x3_reduce = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_3x3 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_3x3_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3b_relu_3x3 = nn.ReLU(inplace)
        self.inception_3b_double_3x3_reduce = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_double_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_1_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3b_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_2_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3b_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_3b_pool_proj = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_pool_proj_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3b_relu_pool_proj = nn.ReLU(inplace)
        self.inception_3c_3x3_reduce = nn.Conv2d(320, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3c_3x3_reduce_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_3c_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3c_3x3 = nn.Conv2d(128, 160, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_3c_3x3_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_3c_relu_3x3 = nn.ReLU(inplace)
        self.inception_3c_double_3x3_reduce = nn.Conv2d(320, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3c_double_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_3c_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3c_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3c_double_3x3_1_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3c_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3c_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_3c_double_3x3_2_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_3c_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3c_pool = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.inception_4a_1x1 = nn.Conv2d(576, 224, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_1x1_bn = nn.BatchNorm2d(224, affine=True)
        self.inception_4a_relu_1x1 = nn.ReLU(inplace)
        self.inception_4a_3x3_reduce = nn.Conv2d(576, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_3x3_reduce_bn = nn.BatchNorm2d(64, affine=True)
        self.inception_4a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_3x3 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_3x3_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_4a_relu_3x3 = nn.ReLU(inplace)
        self.inception_4a_double_3x3_reduce = nn.Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_double_3x3_reduce_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_4a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_double_3x3_1 = nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_double_3x3_1_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4a_double_3x3_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_double_3x3_2_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4a_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4a_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4b_1x1 = nn.Conv2d(576, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_1x1_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4b_relu_1x1 = nn.ReLU(inplace)
        self.inception_4b_3x3_reduce = nn.Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_3x3_reduce_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_4b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_3x3 = nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_3x3_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4b_relu_3x3 = nn.ReLU(inplace)
        self.inception_4b_double_3x3_reduce = nn.Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_double_3x3_reduce_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_4b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_double_3x3_1 = nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_double_3x3_1_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4b_double_3x3_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_double_3x3_2_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4b_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4b_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4b_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4c_1x1 = nn.Conv2d(576, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_1x1_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_4c_relu_1x1 = nn.ReLU(inplace)
        self.inception_4c_3x3_reduce = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_3x3_reduce_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4c_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_3x3 = nn.Conv2d(128, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_3x3_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_4c_relu_3x3 = nn.ReLU(inplace)
        self.inception_4c_double_3x3_reduce = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_double_3x3_reduce_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4c_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_double_3x3_1 = nn.Conv2d(128, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_double_3x3_1_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_4c_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4c_double_3x3_2 = nn.Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_double_3x3_2_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_4c_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4c_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4c_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4c_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4d_1x1 = nn.Conv2d(608, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_1x1_bn = nn.BatchNorm2d(96, affine=True)
        self.inception_4d_relu_1x1 = nn.ReLU(inplace)
        self.inception_4d_3x3_reduce = nn.Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_3x3_reduce_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4d_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_3x3 = nn.Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_3x3_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4d_relu_3x3 = nn.ReLU(inplace)
        self.inception_4d_double_3x3_reduce = nn.Conv2d(608, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_double_3x3_reduce_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_4d_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_double_3x3_1 = nn.Conv2d(160, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_double_3x3_1_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4d_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4d_double_3x3_2 = nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_double_3x3_2_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4d_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4d_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4d_pool_proj = nn.Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4d_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4e_3x3_reduce = nn.Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4e_3x3_reduce_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_4e_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_3x3 = nn.Conv2d(128, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_4e_3x3_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4e_relu_3x3 = nn.ReLU(inplace)
        self.inception_4e_double_3x3_reduce = nn.Conv2d(608, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4e_double_3x3_reduce_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_4e_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_double_3x3_1 = nn.Conv2d(192, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4e_double_3x3_1_bn = nn.BatchNorm2d(256, affine=True)
        self.inception_4e_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4e_double_3x3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_4e_double_3x3_2_bn = nn.BatchNorm2d(256, affine=True)
        self.inception_4e_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4e_pool = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.inception_5a_1x1 = nn.Conv2d(1056, 352, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_1x1_bn = nn.BatchNorm2d(352, affine=True)
        self.inception_5a_relu_1x1 = nn.ReLU(inplace)
        self.inception_5a_3x3_reduce = nn.Conv2d(1056, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_3x3_reduce_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_5a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_3x3 = nn.Conv2d(192, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_3x3_bn = nn.BatchNorm2d(320, affine=True)
        self.inception_5a_relu_3x3 = nn.ReLU(inplace)
        self.inception_5a_double_3x3_reduce = nn.Conv2d(1056, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_double_3x3_reduce_bn = nn.BatchNorm2d(160, affine=True)
        self.inception_5a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_double_3x3_1 = nn.Conv2d(160, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_double_3x3_1_bn = nn.BatchNorm2d(224, affine=True)
        self.inception_5a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_5a_double_3x3_2 = nn.Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_double_3x3_2_bn = nn.BatchNorm2d(224, affine=True)
        self.inception_5a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_5a_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_5a_pool_proj = nn.Conv2d(1056, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_5a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_5b_1x1 = nn.Conv2d(1024, 352, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_1x1_bn = nn.BatchNorm2d(352, affine=True)
        self.inception_5b_relu_1x1 = nn.ReLU(inplace)
        self.inception_5b_3x3_reduce = nn.Conv2d(1024, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_3x3_reduce_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_5b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_3x3 = nn.Conv2d(192, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_3x3_bn = nn.BatchNorm2d(320, affine=True)
        self.inception_5b_relu_3x3 = nn.ReLU(inplace)
        self.inception_5b_double_3x3_reduce = nn.Conv2d(1024, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_double_3x3_reduce_bn = nn.BatchNorm2d(192, affine=True)
        self.inception_5b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_double_3x3_1 = nn.Conv2d(192, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_double_3x3_1_bn = nn.BatchNorm2d(224, affine=True)
        self.inception_5b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_5b_double_3x3_2 = nn.Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_double_3x3_2_bn = nn.BatchNorm2d(224, affine=True)
        self.inception_5b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_5b_pool = nn.MaxPool2d((3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), ceil_mode=True)
        self.inception_5b_pool_proj = nn.Conv2d(1024, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_pool_proj_bn = nn.BatchNorm2d(128, affine=True)
        self.inception_5b_relu_pool_proj = nn.ReLU(inplace)
        self.last_linear = nn.Linear(1024, num_classes)

    def features(self, input):
        conv1_7x7_s2_out = self.conv1_7x7_s2(input)
        conv1_7x7_s2_bn_out = self.conv1_7x7_s2_bn(conv1_7x7_s2_out)
        conv1_relu_7x7_out = self.conv1_relu_7x7(conv1_7x7_s2_bn_out)
        pool1_3x3_s2_out = self.pool1_3x3_s2(conv1_relu_7x7_out)
        conv2_3x3_reduce_out = self.conv2_3x3_reduce(pool1_3x3_s2_out)
        conv2_3x3_reduce_bn_out = self.conv2_3x3_reduce_bn(conv2_3x3_reduce_out)
        conv2_relu_3x3_reduce_out = self.conv2_relu_3x3_reduce(conv2_3x3_reduce_bn_out)
        conv2_3x3_out = self.conv2_3x3(conv2_relu_3x3_reduce_out)
        conv2_3x3_bn_out = self.conv2_3x3_bn(conv2_3x3_out)
        conv2_relu_3x3_out = self.conv2_relu_3x3(conv2_3x3_bn_out)
        pool2_3x3_s2_out = self.pool2_3x3_s2(conv2_relu_3x3_out)
        inception_3a_1x1_out = self.inception_3a_1x1(pool2_3x3_s2_out)
        inception_3a_1x1_bn_out = self.inception_3a_1x1_bn(inception_3a_1x1_out)
        inception_3a_relu_1x1_out = self.inception_3a_relu_1x1(inception_3a_1x1_bn_out)
        inception_3a_3x3_reduce_out = self.inception_3a_3x3_reduce(pool2_3x3_s2_out)
        inception_3a_3x3_reduce_bn_out = self.inception_3a_3x3_reduce_bn(inception_3a_3x3_reduce_out)
        inception_3a_relu_3x3_reduce_out = self.inception_3a_relu_3x3_reduce(inception_3a_3x3_reduce_bn_out)
        inception_3a_3x3_out = self.inception_3a_3x3(inception_3a_relu_3x3_reduce_out)
        inception_3a_3x3_bn_out = self.inception_3a_3x3_bn(inception_3a_3x3_out)
        inception_3a_relu_3x3_out = self.inception_3a_relu_3x3(inception_3a_3x3_bn_out)
        inception_3a_double_3x3_reduce_out = self.inception_3a_double_3x3_reduce(pool2_3x3_s2_out)
        inception_3a_double_3x3_reduce_bn_out = self.inception_3a_double_3x3_reduce_bn(inception_3a_double_3x3_reduce_out)
        inception_3a_relu_double_3x3_reduce_out = self.inception_3a_relu_double_3x3_reduce(inception_3a_double_3x3_reduce_bn_out)
        inception_3a_double_3x3_1_out = self.inception_3a_double_3x3_1(inception_3a_relu_double_3x3_reduce_out)
        inception_3a_double_3x3_1_bn_out = self.inception_3a_double_3x3_1_bn(inception_3a_double_3x3_1_out)
        inception_3a_relu_double_3x3_1_out = self.inception_3a_relu_double_3x3_1(inception_3a_double_3x3_1_bn_out)
        inception_3a_double_3x3_2_out = self.inception_3a_double_3x3_2(inception_3a_relu_double_3x3_1_out)
        inception_3a_double_3x3_2_bn_out = self.inception_3a_double_3x3_2_bn(inception_3a_double_3x3_2_out)
        inception_3a_relu_double_3x3_2_out = self.inception_3a_relu_double_3x3_2(inception_3a_double_3x3_2_bn_out)
        inception_3a_pool_out = self.inception_3a_pool(pool2_3x3_s2_out)
        inception_3a_pool_proj_out = self.inception_3a_pool_proj(inception_3a_pool_out)
        inception_3a_pool_proj_bn_out = self.inception_3a_pool_proj_bn(inception_3a_pool_proj_out)
        inception_3a_relu_pool_proj_out = self.inception_3a_relu_pool_proj(inception_3a_pool_proj_bn_out)
        inception_3a_output_out = torch.cat([inception_3a_relu_1x1_out, inception_3a_relu_3x3_out, inception_3a_relu_double_3x3_2_out, inception_3a_relu_pool_proj_out], 1)
        inception_3b_1x1_out = self.inception_3b_1x1(inception_3a_output_out)
        inception_3b_1x1_bn_out = self.inception_3b_1x1_bn(inception_3b_1x1_out)
        inception_3b_relu_1x1_out = self.inception_3b_relu_1x1(inception_3b_1x1_bn_out)
        inception_3b_3x3_reduce_out = self.inception_3b_3x3_reduce(inception_3a_output_out)
        inception_3b_3x3_reduce_bn_out = self.inception_3b_3x3_reduce_bn(inception_3b_3x3_reduce_out)
        inception_3b_relu_3x3_reduce_out = self.inception_3b_relu_3x3_reduce(inception_3b_3x3_reduce_bn_out)
        inception_3b_3x3_out = self.inception_3b_3x3(inception_3b_relu_3x3_reduce_out)
        inception_3b_3x3_bn_out = self.inception_3b_3x3_bn(inception_3b_3x3_out)
        inception_3b_relu_3x3_out = self.inception_3b_relu_3x3(inception_3b_3x3_bn_out)
        inception_3b_double_3x3_reduce_out = self.inception_3b_double_3x3_reduce(inception_3a_output_out)
        inception_3b_double_3x3_reduce_bn_out = self.inception_3b_double_3x3_reduce_bn(inception_3b_double_3x3_reduce_out)
        inception_3b_relu_double_3x3_reduce_out = self.inception_3b_relu_double_3x3_reduce(inception_3b_double_3x3_reduce_bn_out)
        inception_3b_double_3x3_1_out = self.inception_3b_double_3x3_1(inception_3b_relu_double_3x3_reduce_out)
        inception_3b_double_3x3_1_bn_out = self.inception_3b_double_3x3_1_bn(inception_3b_double_3x3_1_out)
        inception_3b_relu_double_3x3_1_out = self.inception_3b_relu_double_3x3_1(inception_3b_double_3x3_1_bn_out)
        inception_3b_double_3x3_2_out = self.inception_3b_double_3x3_2(inception_3b_relu_double_3x3_1_out)
        inception_3b_double_3x3_2_bn_out = self.inception_3b_double_3x3_2_bn(inception_3b_double_3x3_2_out)
        inception_3b_relu_double_3x3_2_out = self.inception_3b_relu_double_3x3_2(inception_3b_double_3x3_2_bn_out)
        inception_3b_pool_out = self.inception_3b_pool(inception_3a_output_out)
        inception_3b_pool_proj_out = self.inception_3b_pool_proj(inception_3b_pool_out)
        inception_3b_pool_proj_bn_out = self.inception_3b_pool_proj_bn(inception_3b_pool_proj_out)
        inception_3b_relu_pool_proj_out = self.inception_3b_relu_pool_proj(inception_3b_pool_proj_bn_out)
        inception_3b_output_out = torch.cat([inception_3b_relu_1x1_out, inception_3b_relu_3x3_out, inception_3b_relu_double_3x3_2_out, inception_3b_relu_pool_proj_out], 1)
        inception_3c_3x3_reduce_out = self.inception_3c_3x3_reduce(inception_3b_output_out)
        inception_3c_3x3_reduce_bn_out = self.inception_3c_3x3_reduce_bn(inception_3c_3x3_reduce_out)
        inception_3c_relu_3x3_reduce_out = self.inception_3c_relu_3x3_reduce(inception_3c_3x3_reduce_bn_out)
        inception_3c_3x3_out = self.inception_3c_3x3(inception_3c_relu_3x3_reduce_out)
        inception_3c_3x3_bn_out = self.inception_3c_3x3_bn(inception_3c_3x3_out)
        inception_3c_relu_3x3_out = self.inception_3c_relu_3x3(inception_3c_3x3_bn_out)
        inception_3c_double_3x3_reduce_out = self.inception_3c_double_3x3_reduce(inception_3b_output_out)
        inception_3c_double_3x3_reduce_bn_out = self.inception_3c_double_3x3_reduce_bn(inception_3c_double_3x3_reduce_out)
        inception_3c_relu_double_3x3_reduce_out = self.inception_3c_relu_double_3x3_reduce(inception_3c_double_3x3_reduce_bn_out)
        inception_3c_double_3x3_1_out = self.inception_3c_double_3x3_1(inception_3c_relu_double_3x3_reduce_out)
        inception_3c_double_3x3_1_bn_out = self.inception_3c_double_3x3_1_bn(inception_3c_double_3x3_1_out)
        inception_3c_relu_double_3x3_1_out = self.inception_3c_relu_double_3x3_1(inception_3c_double_3x3_1_bn_out)
        inception_3c_double_3x3_2_out = self.inception_3c_double_3x3_2(inception_3c_relu_double_3x3_1_out)
        inception_3c_double_3x3_2_bn_out = self.inception_3c_double_3x3_2_bn(inception_3c_double_3x3_2_out)
        inception_3c_relu_double_3x3_2_out = self.inception_3c_relu_double_3x3_2(inception_3c_double_3x3_2_bn_out)
        inception_3c_pool_out = self.inception_3c_pool(inception_3b_output_out)
        inception_3c_output_out = torch.cat([inception_3c_relu_3x3_out, inception_3c_relu_double_3x3_2_out, inception_3c_pool_out], 1)
        inception_4a_1x1_out = self.inception_4a_1x1(inception_3c_output_out)
        inception_4a_1x1_bn_out = self.inception_4a_1x1_bn(inception_4a_1x1_out)
        inception_4a_relu_1x1_out = self.inception_4a_relu_1x1(inception_4a_1x1_bn_out)
        inception_4a_3x3_reduce_out = self.inception_4a_3x3_reduce(inception_3c_output_out)
        inception_4a_3x3_reduce_bn_out = self.inception_4a_3x3_reduce_bn(inception_4a_3x3_reduce_out)
        inception_4a_relu_3x3_reduce_out = self.inception_4a_relu_3x3_reduce(inception_4a_3x3_reduce_bn_out)
        inception_4a_3x3_out = self.inception_4a_3x3(inception_4a_relu_3x3_reduce_out)
        inception_4a_3x3_bn_out = self.inception_4a_3x3_bn(inception_4a_3x3_out)
        inception_4a_relu_3x3_out = self.inception_4a_relu_3x3(inception_4a_3x3_bn_out)
        inception_4a_double_3x3_reduce_out = self.inception_4a_double_3x3_reduce(inception_3c_output_out)
        inception_4a_double_3x3_reduce_bn_out = self.inception_4a_double_3x3_reduce_bn(inception_4a_double_3x3_reduce_out)
        inception_4a_relu_double_3x3_reduce_out = self.inception_4a_relu_double_3x3_reduce(inception_4a_double_3x3_reduce_bn_out)
        inception_4a_double_3x3_1_out = self.inception_4a_double_3x3_1(inception_4a_relu_double_3x3_reduce_out)
        inception_4a_double_3x3_1_bn_out = self.inception_4a_double_3x3_1_bn(inception_4a_double_3x3_1_out)
        inception_4a_relu_double_3x3_1_out = self.inception_4a_relu_double_3x3_1(inception_4a_double_3x3_1_bn_out)
        inception_4a_double_3x3_2_out = self.inception_4a_double_3x3_2(inception_4a_relu_double_3x3_1_out)
        inception_4a_double_3x3_2_bn_out = self.inception_4a_double_3x3_2_bn(inception_4a_double_3x3_2_out)
        inception_4a_relu_double_3x3_2_out = self.inception_4a_relu_double_3x3_2(inception_4a_double_3x3_2_bn_out)
        inception_4a_pool_out = self.inception_4a_pool(inception_3c_output_out)
        inception_4a_pool_proj_out = self.inception_4a_pool_proj(inception_4a_pool_out)
        inception_4a_pool_proj_bn_out = self.inception_4a_pool_proj_bn(inception_4a_pool_proj_out)
        inception_4a_relu_pool_proj_out = self.inception_4a_relu_pool_proj(inception_4a_pool_proj_bn_out)
        inception_4a_output_out = torch.cat([inception_4a_relu_1x1_out, inception_4a_relu_3x3_out, inception_4a_relu_double_3x3_2_out, inception_4a_relu_pool_proj_out], 1)
        inception_4b_1x1_out = self.inception_4b_1x1(inception_4a_output_out)
        inception_4b_1x1_bn_out = self.inception_4b_1x1_bn(inception_4b_1x1_out)
        inception_4b_relu_1x1_out = self.inception_4b_relu_1x1(inception_4b_1x1_bn_out)
        inception_4b_3x3_reduce_out = self.inception_4b_3x3_reduce(inception_4a_output_out)
        inception_4b_3x3_reduce_bn_out = self.inception_4b_3x3_reduce_bn(inception_4b_3x3_reduce_out)
        inception_4b_relu_3x3_reduce_out = self.inception_4b_relu_3x3_reduce(inception_4b_3x3_reduce_bn_out)
        inception_4b_3x3_out = self.inception_4b_3x3(inception_4b_relu_3x3_reduce_out)
        inception_4b_3x3_bn_out = self.inception_4b_3x3_bn(inception_4b_3x3_out)
        inception_4b_relu_3x3_out = self.inception_4b_relu_3x3(inception_4b_3x3_bn_out)
        inception_4b_double_3x3_reduce_out = self.inception_4b_double_3x3_reduce(inception_4a_output_out)
        inception_4b_double_3x3_reduce_bn_out = self.inception_4b_double_3x3_reduce_bn(inception_4b_double_3x3_reduce_out)
        inception_4b_relu_double_3x3_reduce_out = self.inception_4b_relu_double_3x3_reduce(inception_4b_double_3x3_reduce_bn_out)
        inception_4b_double_3x3_1_out = self.inception_4b_double_3x3_1(inception_4b_relu_double_3x3_reduce_out)
        inception_4b_double_3x3_1_bn_out = self.inception_4b_double_3x3_1_bn(inception_4b_double_3x3_1_out)
        inception_4b_relu_double_3x3_1_out = self.inception_4b_relu_double_3x3_1(inception_4b_double_3x3_1_bn_out)
        inception_4b_double_3x3_2_out = self.inception_4b_double_3x3_2(inception_4b_relu_double_3x3_1_out)
        inception_4b_double_3x3_2_bn_out = self.inception_4b_double_3x3_2_bn(inception_4b_double_3x3_2_out)
        inception_4b_relu_double_3x3_2_out = self.inception_4b_relu_double_3x3_2(inception_4b_double_3x3_2_bn_out)
        inception_4b_pool_out = self.inception_4b_pool(inception_4a_output_out)
        inception_4b_pool_proj_out = self.inception_4b_pool_proj(inception_4b_pool_out)
        inception_4b_pool_proj_bn_out = self.inception_4b_pool_proj_bn(inception_4b_pool_proj_out)
        inception_4b_relu_pool_proj_out = self.inception_4b_relu_pool_proj(inception_4b_pool_proj_bn_out)
        inception_4b_output_out = torch.cat([inception_4b_relu_1x1_out, inception_4b_relu_3x3_out, inception_4b_relu_double_3x3_2_out, inception_4b_relu_pool_proj_out], 1)
        inception_4c_1x1_out = self.inception_4c_1x1(inception_4b_output_out)
        inception_4c_1x1_bn_out = self.inception_4c_1x1_bn(inception_4c_1x1_out)
        inception_4c_relu_1x1_out = self.inception_4c_relu_1x1(inception_4c_1x1_bn_out)
        inception_4c_3x3_reduce_out = self.inception_4c_3x3_reduce(inception_4b_output_out)
        inception_4c_3x3_reduce_bn_out = self.inception_4c_3x3_reduce_bn(inception_4c_3x3_reduce_out)
        inception_4c_relu_3x3_reduce_out = self.inception_4c_relu_3x3_reduce(inception_4c_3x3_reduce_bn_out)
        inception_4c_3x3_out = self.inception_4c_3x3(inception_4c_relu_3x3_reduce_out)
        inception_4c_3x3_bn_out = self.inception_4c_3x3_bn(inception_4c_3x3_out)
        inception_4c_relu_3x3_out = self.inception_4c_relu_3x3(inception_4c_3x3_bn_out)
        inception_4c_double_3x3_reduce_out = self.inception_4c_double_3x3_reduce(inception_4b_output_out)
        inception_4c_double_3x3_reduce_bn_out = self.inception_4c_double_3x3_reduce_bn(inception_4c_double_3x3_reduce_out)
        inception_4c_relu_double_3x3_reduce_out = self.inception_4c_relu_double_3x3_reduce(inception_4c_double_3x3_reduce_bn_out)
        inception_4c_double_3x3_1_out = self.inception_4c_double_3x3_1(inception_4c_relu_double_3x3_reduce_out)
        inception_4c_double_3x3_1_bn_out = self.inception_4c_double_3x3_1_bn(inception_4c_double_3x3_1_out)
        inception_4c_relu_double_3x3_1_out = self.inception_4c_relu_double_3x3_1(inception_4c_double_3x3_1_bn_out)
        inception_4c_double_3x3_2_out = self.inception_4c_double_3x3_2(inception_4c_relu_double_3x3_1_out)
        inception_4c_double_3x3_2_bn_out = self.inception_4c_double_3x3_2_bn(inception_4c_double_3x3_2_out)
        inception_4c_relu_double_3x3_2_out = self.inception_4c_relu_double_3x3_2(inception_4c_double_3x3_2_bn_out)
        inception_4c_pool_out = self.inception_4c_pool(inception_4b_output_out)
        inception_4c_pool_proj_out = self.inception_4c_pool_proj(inception_4c_pool_out)
        inception_4c_pool_proj_bn_out = self.inception_4c_pool_proj_bn(inception_4c_pool_proj_out)
        inception_4c_relu_pool_proj_out = self.inception_4c_relu_pool_proj(inception_4c_pool_proj_bn_out)
        inception_4c_output_out = torch.cat([inception_4c_relu_1x1_out, inception_4c_relu_3x3_out, inception_4c_relu_double_3x3_2_out, inception_4c_relu_pool_proj_out], 1)
        inception_4d_1x1_out = self.inception_4d_1x1(inception_4c_output_out)
        inception_4d_1x1_bn_out = self.inception_4d_1x1_bn(inception_4d_1x1_out)
        inception_4d_relu_1x1_out = self.inception_4d_relu_1x1(inception_4d_1x1_bn_out)
        inception_4d_3x3_reduce_out = self.inception_4d_3x3_reduce(inception_4c_output_out)
        inception_4d_3x3_reduce_bn_out = self.inception_4d_3x3_reduce_bn(inception_4d_3x3_reduce_out)
        inception_4d_relu_3x3_reduce_out = self.inception_4d_relu_3x3_reduce(inception_4d_3x3_reduce_bn_out)
        inception_4d_3x3_out = self.inception_4d_3x3(inception_4d_relu_3x3_reduce_out)
        inception_4d_3x3_bn_out = self.inception_4d_3x3_bn(inception_4d_3x3_out)
        inception_4d_relu_3x3_out = self.inception_4d_relu_3x3(inception_4d_3x3_bn_out)
        inception_4d_double_3x3_reduce_out = self.inception_4d_double_3x3_reduce(inception_4c_output_out)
        inception_4d_double_3x3_reduce_bn_out = self.inception_4d_double_3x3_reduce_bn(inception_4d_double_3x3_reduce_out)
        inception_4d_relu_double_3x3_reduce_out = self.inception_4d_relu_double_3x3_reduce(inception_4d_double_3x3_reduce_bn_out)
        inception_4d_double_3x3_1_out = self.inception_4d_double_3x3_1(inception_4d_relu_double_3x3_reduce_out)
        inception_4d_double_3x3_1_bn_out = self.inception_4d_double_3x3_1_bn(inception_4d_double_3x3_1_out)
        inception_4d_relu_double_3x3_1_out = self.inception_4d_relu_double_3x3_1(inception_4d_double_3x3_1_bn_out)
        inception_4d_double_3x3_2_out = self.inception_4d_double_3x3_2(inception_4d_relu_double_3x3_1_out)
        inception_4d_double_3x3_2_bn_out = self.inception_4d_double_3x3_2_bn(inception_4d_double_3x3_2_out)
        inception_4d_relu_double_3x3_2_out = self.inception_4d_relu_double_3x3_2(inception_4d_double_3x3_2_bn_out)
        inception_4d_pool_out = self.inception_4d_pool(inception_4c_output_out)
        inception_4d_pool_proj_out = self.inception_4d_pool_proj(inception_4d_pool_out)
        inception_4d_pool_proj_bn_out = self.inception_4d_pool_proj_bn(inception_4d_pool_proj_out)
        inception_4d_relu_pool_proj_out = self.inception_4d_relu_pool_proj(inception_4d_pool_proj_bn_out)
        inception_4d_output_out = torch.cat([inception_4d_relu_1x1_out, inception_4d_relu_3x3_out, inception_4d_relu_double_3x3_2_out, inception_4d_relu_pool_proj_out], 1)
        inception_4e_3x3_reduce_out = self.inception_4e_3x3_reduce(inception_4d_output_out)
        inception_4e_3x3_reduce_bn_out = self.inception_4e_3x3_reduce_bn(inception_4e_3x3_reduce_out)
        inception_4e_relu_3x3_reduce_out = self.inception_4e_relu_3x3_reduce(inception_4e_3x3_reduce_bn_out)
        inception_4e_3x3_out = self.inception_4e_3x3(inception_4e_relu_3x3_reduce_out)
        inception_4e_3x3_bn_out = self.inception_4e_3x3_bn(inception_4e_3x3_out)
        inception_4e_relu_3x3_out = self.inception_4e_relu_3x3(inception_4e_3x3_bn_out)
        inception_4e_double_3x3_reduce_out = self.inception_4e_double_3x3_reduce(inception_4d_output_out)
        inception_4e_double_3x3_reduce_bn_out = self.inception_4e_double_3x3_reduce_bn(inception_4e_double_3x3_reduce_out)
        inception_4e_relu_double_3x3_reduce_out = self.inception_4e_relu_double_3x3_reduce(inception_4e_double_3x3_reduce_bn_out)
        inception_4e_double_3x3_1_out = self.inception_4e_double_3x3_1(inception_4e_relu_double_3x3_reduce_out)
        inception_4e_double_3x3_1_bn_out = self.inception_4e_double_3x3_1_bn(inception_4e_double_3x3_1_out)
        inception_4e_relu_double_3x3_1_out = self.inception_4e_relu_double_3x3_1(inception_4e_double_3x3_1_bn_out)
        inception_4e_double_3x3_2_out = self.inception_4e_double_3x3_2(inception_4e_relu_double_3x3_1_out)
        inception_4e_double_3x3_2_bn_out = self.inception_4e_double_3x3_2_bn(inception_4e_double_3x3_2_out)
        inception_4e_relu_double_3x3_2_out = self.inception_4e_relu_double_3x3_2(inception_4e_double_3x3_2_bn_out)
        inception_4e_pool_out = self.inception_4e_pool(inception_4d_output_out)
        inception_4e_output_out = torch.cat([inception_4e_relu_3x3_out, inception_4e_relu_double_3x3_2_out, inception_4e_pool_out], 1)
        inception_5a_1x1_out = self.inception_5a_1x1(inception_4e_output_out)
        inception_5a_1x1_bn_out = self.inception_5a_1x1_bn(inception_5a_1x1_out)
        inception_5a_relu_1x1_out = self.inception_5a_relu_1x1(inception_5a_1x1_bn_out)
        inception_5a_3x3_reduce_out = self.inception_5a_3x3_reduce(inception_4e_output_out)
        inception_5a_3x3_reduce_bn_out = self.inception_5a_3x3_reduce_bn(inception_5a_3x3_reduce_out)
        inception_5a_relu_3x3_reduce_out = self.inception_5a_relu_3x3_reduce(inception_5a_3x3_reduce_bn_out)
        inception_5a_3x3_out = self.inception_5a_3x3(inception_5a_relu_3x3_reduce_out)
        inception_5a_3x3_bn_out = self.inception_5a_3x3_bn(inception_5a_3x3_out)
        inception_5a_relu_3x3_out = self.inception_5a_relu_3x3(inception_5a_3x3_bn_out)
        inception_5a_double_3x3_reduce_out = self.inception_5a_double_3x3_reduce(inception_4e_output_out)
        inception_5a_double_3x3_reduce_bn_out = self.inception_5a_double_3x3_reduce_bn(inception_5a_double_3x3_reduce_out)
        inception_5a_relu_double_3x3_reduce_out = self.inception_5a_relu_double_3x3_reduce(inception_5a_double_3x3_reduce_bn_out)
        inception_5a_double_3x3_1_out = self.inception_5a_double_3x3_1(inception_5a_relu_double_3x3_reduce_out)
        inception_5a_double_3x3_1_bn_out = self.inception_5a_double_3x3_1_bn(inception_5a_double_3x3_1_out)
        inception_5a_relu_double_3x3_1_out = self.inception_5a_relu_double_3x3_1(inception_5a_double_3x3_1_bn_out)
        inception_5a_double_3x3_2_out = self.inception_5a_double_3x3_2(inception_5a_relu_double_3x3_1_out)
        inception_5a_double_3x3_2_bn_out = self.inception_5a_double_3x3_2_bn(inception_5a_double_3x3_2_out)
        inception_5a_relu_double_3x3_2_out = self.inception_5a_relu_double_3x3_2(inception_5a_double_3x3_2_bn_out)
        inception_5a_pool_out = self.inception_5a_pool(inception_4e_output_out)
        inception_5a_pool_proj_out = self.inception_5a_pool_proj(inception_5a_pool_out)
        inception_5a_pool_proj_bn_out = self.inception_5a_pool_proj_bn(inception_5a_pool_proj_out)
        inception_5a_relu_pool_proj_out = self.inception_5a_relu_pool_proj(inception_5a_pool_proj_bn_out)
        inception_5a_output_out = torch.cat([inception_5a_relu_1x1_out, inception_5a_relu_3x3_out, inception_5a_relu_double_3x3_2_out, inception_5a_relu_pool_proj_out], 1)
        inception_5b_1x1_out = self.inception_5b_1x1(inception_5a_output_out)
        inception_5b_1x1_bn_out = self.inception_5b_1x1_bn(inception_5b_1x1_out)
        inception_5b_relu_1x1_out = self.inception_5b_relu_1x1(inception_5b_1x1_bn_out)
        inception_5b_3x3_reduce_out = self.inception_5b_3x3_reduce(inception_5a_output_out)
        inception_5b_3x3_reduce_bn_out = self.inception_5b_3x3_reduce_bn(inception_5b_3x3_reduce_out)
        inception_5b_relu_3x3_reduce_out = self.inception_5b_relu_3x3_reduce(inception_5b_3x3_reduce_bn_out)
        inception_5b_3x3_out = self.inception_5b_3x3(inception_5b_relu_3x3_reduce_out)
        inception_5b_3x3_bn_out = self.inception_5b_3x3_bn(inception_5b_3x3_out)
        inception_5b_relu_3x3_out = self.inception_5b_relu_3x3(inception_5b_3x3_bn_out)
        inception_5b_double_3x3_reduce_out = self.inception_5b_double_3x3_reduce(inception_5a_output_out)
        inception_5b_double_3x3_reduce_bn_out = self.inception_5b_double_3x3_reduce_bn(inception_5b_double_3x3_reduce_out)
        inception_5b_relu_double_3x3_reduce_out = self.inception_5b_relu_double_3x3_reduce(inception_5b_double_3x3_reduce_bn_out)
        inception_5b_double_3x3_1_out = self.inception_5b_double_3x3_1(inception_5b_relu_double_3x3_reduce_out)
        inception_5b_double_3x3_1_bn_out = self.inception_5b_double_3x3_1_bn(inception_5b_double_3x3_1_out)
        inception_5b_relu_double_3x3_1_out = self.inception_5b_relu_double_3x3_1(inception_5b_double_3x3_1_bn_out)
        inception_5b_double_3x3_2_out = self.inception_5b_double_3x3_2(inception_5b_relu_double_3x3_1_out)
        inception_5b_double_3x3_2_bn_out = self.inception_5b_double_3x3_2_bn(inception_5b_double_3x3_2_out)
        inception_5b_relu_double_3x3_2_out = self.inception_5b_relu_double_3x3_2(inception_5b_double_3x3_2_bn_out)
        inception_5b_pool_out = self.inception_5b_pool(inception_5a_output_out)
        inception_5b_pool_proj_out = self.inception_5b_pool_proj(inception_5b_pool_out)
        inception_5b_pool_proj_bn_out = self.inception_5b_pool_proj_bn(inception_5b_pool_proj_out)
        inception_5b_relu_pool_proj_out = self.inception_5b_relu_pool_proj(inception_5b_pool_proj_bn_out)
        inception_5b_output_out = torch.cat([inception_5b_relu_1x1_out, inception_5b_relu_3x3_out, inception_5b_relu_double_3x3_2_out, inception_5b_relu_pool_proj_out], 1)
        return inception_5b_output_out

    def logits(self, features):
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def pooling_factor(pool_type='avg'):
    return 2 if pool_type == 'avgmaxc' else 1


class AdaptiveAvgMaxPool2d(torch.nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """

    def __init__(self, output_size=1, pool_type='avg'):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size
        self.pool_type = pool_type
        if pool_type == 'avgmaxc' or pool_type == 'avgmax':
            self.pool = nn.ModuleList([nn.AdaptiveAvgPool2d(output_size), nn.AdaptiveMaxPool2d(output_size)])
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            if pool_type != 'avg':
                None
            self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        if self.pool_type == 'avgmaxc':
            x = torch.cat([p(x) for p in self.pool], dim=1)
        elif self.pool_type == 'avgmax':
            x = 0.5 * torch.sum(torch.stack([p(x) for p in self.pool]), 0).squeeze(dim=0)
        else:
            x = self.pool(x)
        return x

    def factor(self):
        return pooling_factor(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + 'output_size=' + str(self.output_size) + ', pool_type=' + self.pool_type + ')'


class FBResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        self.input_space = None
        self.input_size = 299, 299, 3
        self.mean = None
        self.std = None
        super(FBResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=True), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def features(self, input):
        x = self.conv1(input)
        self.conv1_input = x.clone()
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, features):
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()
        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(192, 48, kernel_size=1, stride=1), BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2))
        self.branch2 = nn.Sequential(BasicConv2d(192, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(192, 64, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1, stride=1), BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1, stride=1), BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1), BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1))
        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()
        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(BasicConv2d(320, 256, kernel_size=1, stride=1), BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1088, 128, kernel_size=1, stride=1), BasicConv2d(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)))
        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1, stride=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1, stride=1), BasicConv2d(256, 288, kernel_size=3, stride=2))
        self.branch2 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1, stride=1), BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1), BasicConv2d(288, 320, kernel_size=3, stride=2))
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()
        self.scale = scale
        self.noReLU = noReLU
        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(2080, 192, kernel_size=1, stride=1), BasicConv2d(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1)), BasicConv2d(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0)))
        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class InceptionResNetV2(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionResNetV2, self).__init__()
        self.input_space = None
        self.input_size = 299, 299, 3
        self.mean = None
        self.std = None
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17))
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1))
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.Sequential(Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2))
        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        self.avgpool_1a = nn.AvgPool2d(8, count_include_pad=False)
        self.last_linear = nn.Linear(1536, num_classes)

    def features(self, input):
        x = self.conv2d_1a(input)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        return x

    def logits(self, features):
        x = self.avgpool_1a(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class MaxPoolPad(nn.Module):

    def __init__(self):
        super(MaxPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:].contiguous()
        return x


class AvgPoolPad(nn.Module):

    def __init__(self, stride=2, padding=1):
        super(AvgPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.AvgPool2d(3, stride=stride, padding=padding, count_include_pad=False)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:].contiguous()
        return x


class BranchSeparables(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, stem_cell=False, zero_pad=False):
        super(BranchSeparables, self).__init__()
        padding = kernel_size // 2
        middle_channels = out_channels if stem_cell else in_channels
        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0)) if zero_pad else None
        self.relu_1 = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, middle_channels, kernel_size, dw_stride=stride, dw_padding=padding)
        self.bn_sep_1 = nn.BatchNorm2d(middle_channels, eps=0.001)
        self.relu_2 = nn.ReLU()
        self.separable_2 = SeparableConv2d(middle_channels, out_channels, kernel_size, dw_stride=1, dw_padding=padding)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.relu_1(x)
        if self.zero_pad:
            x = self.zero_pad(x)
        x = self.separable_1(x)
        if self.zero_pad:
            x = x[:, :, 1:, 1:].contiguous()
        x = self.bn_sep_1(x)
        x = self.relu_2(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class BranchSeparablesStem(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(BranchSeparablesStem, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(out_channels, out_channels, kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class BranchSeparablesReduction(BranchSeparables):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, z_padding=1, bias=False):
        BranchSeparables.__init__(self, in_channels, out_channels, kernel_size, stride, padding, bias)
        self.padding = nn.ZeroPad2d((z_padding, 0, z_padding, 0))

    def forward(self, x):
        x = self.relu(x)
        x = self.padding(x)
        x = self.separable_1(x)
        x = x[:, :, 1:, 1:].contiguous()
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class CellBase(nn.Module):

    def cell_forward(self, x_left, x_right):
        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_left = self.comb_iter_3_left(x_comb_iter_2)
        x_comb_iter_3_right = self.comb_iter_3_right(x_right)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_left)
        if self.comb_iter_4_right:
            x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        else:
            x_comb_iter_4_right = x_right
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class MaxPool(nn.Module):

    def __init__(self, kernel_size, stride=1, padding=1, zero_pad=False):
        super(MaxPool, self).__init__()
        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0)) if zero_pad else None
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        if self.zero_pad:
            x = self.zero_pad(x)
        x = self.pool(x)
        if self.zero_pad:
            x = x[:, :, 1:, 1:]
        return x


class ReluConvBn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ReluConvBn, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class CellStem0(CellBase):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(CellStem0, self).__init__()
        self.conv_1x1 = ReluConvBn(in_channels_right, out_channels_right, kernel_size=1)
        self.comb_iter_0_left = BranchSeparables(in_channels_left, out_channels_left, kernel_size=5, stride=2, stem_cell=True)
        self.comb_iter_0_right = nn.Sequential(OrderedDict([('max_pool', MaxPool(3, stride=2)), ('conv', nn.Conv2d(in_channels_left, out_channels_left, kernel_size=1, bias=False)), ('bn', nn.BatchNorm2d(out_channels_left, eps=0.001))]))
        self.comb_iter_1_left = BranchSeparables(out_channels_right, out_channels_right, kernel_size=7, stride=2)
        self.comb_iter_1_right = MaxPool(3, stride=2)
        self.comb_iter_2_left = BranchSeparables(out_channels_right, out_channels_right, kernel_size=5, stride=2)
        self.comb_iter_2_right = BranchSeparables(out_channels_right, out_channels_right, kernel_size=3, stride=2)
        self.comb_iter_3_left = BranchSeparables(out_channels_right, out_channels_right, kernel_size=3)
        self.comb_iter_3_right = MaxPool(3, stride=2)
        self.comb_iter_4_left = BranchSeparables(in_channels_right, out_channels_right, kernel_size=3, stride=2, stem_cell=True)
        self.comb_iter_4_right = ReluConvBn(out_channels_right, out_channels_right, kernel_size=1, stride=2)

    def forward(self, x_left):
        x_right = self.conv_1x1(x_left)
        x_out = self.cell_forward(x_left, x_right)
        return x_out


class CellStem1(nn.Module):

    def __init__(self, stem_filters, num_filters):
        super(CellStem1, self).__init__()
        self.num_filters = num_filters
        self.stem_filters = stem_filters
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(2 * self.num_filters, self.num_filters, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(self.num_filters, eps=0.001, momentum=0.1, affine=True))
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(self.stem_filters, self.num_filters // 2, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(self.stem_filters, self.num_filters // 2, 1, stride=1, bias=False))
        self.final_path_bn = nn.BatchNorm2d(self.num_filters, eps=0.001, momentum=0.1, affine=True)
        self.comb_iter_0_left = BranchSeparables(self.num_filters, self.num_filters, 5, 2, 2, name='specific', bias=False)
        self.comb_iter_0_right = BranchSeparables(self.num_filters, self.num_filters, 7, 2, 3, name='specific', bias=False)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparables(self.num_filters, self.num_filters, 7, 2, 3, name='specific', bias=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparables(self.num_filters, self.num_filters, 5, 2, 2, name='specific', bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(self.num_filters, self.num_filters, 3, 1, 1, name='specific', bias=False)
        self.comb_iter_4_right = MaxPoolPad()

    def forward(self, x_conv0, x_stem_0):
        x_left = self.conv_1x1(x_stem_0)
        x_relu = self.relu(x_conv0)
        x_path1 = self.path_1(x_relu)
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        x_right = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_right)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_left)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_left)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class FirstCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(FirstCell, self).__init__()
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.final_path_bn = nn.BatchNorm2d(out_channels_left * 2, eps=0.001, momentum=0.1, affine=True)
        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_1_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_relu = self.relu(x_prev)
        x_path1 = self.path_1(x_relu)
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        x_left = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left
        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NormalCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(NormalCell, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_left, out_channels_left, 3, 1, 1, bias=False)
        self.comb_iter_1_left = BranchSeparables(out_channels_left, out_channels_left, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_left, out_channels_left, 3, 1, 1, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left
        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class ReductionCell0(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(ReductionCell0, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparablesReduction(out_channels_right, out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparablesReduction(out_channels_right, out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparablesReduction(out_channels_right, out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparablesReduction(out_channels_right, out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_4_left = BranchSeparablesReduction(out_channels_right, out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_4_right = MaxPoolPad()

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class ReductionCell1(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(ReductionCell1, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 2, 2, name='specific', bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right, out_channels_right, 7, 2, 3, name='specific', bias=False)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparables(out_channels_right, out_channels_right, 7, 2, 3, name='specific', bias=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparables(out_channels_right, out_channels_right, 5, 2, 2, name='specific', bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, name='specific', bias=False)
        self.comb_iter_4_right = MaxPoolPad()

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NASNetALarge(nn.Module):
    """NASNetALarge (6 @ 4032) """

    def __init__(self, num_classes=1001, stem_filters=96, penultimate_filters=4032, filters_multiplier=2):
        super(NASNetALarge, self).__init__()
        self.num_classes = num_classes
        self.stem_filters = stem_filters
        self.penultimate_filters = penultimate_filters
        self.filters_multiplier = filters_multiplier
        filters = self.penultimate_filters // 24
        self.conv0 = nn.Sequential()
        self.conv0.add_module('conv', nn.Conv2d(in_channels=3, out_channels=self.stem_filters, kernel_size=3, padding=0, stride=2, bias=False))
        self.conv0.add_module('bn', nn.BatchNorm2d(self.stem_filters, eps=0.001, momentum=0.1, affine=True))
        self.cell_stem_0 = CellStem0(self.stem_filters, num_filters=filters // filters_multiplier ** 2)
        self.cell_stem_1 = CellStem1(self.stem_filters, num_filters=filters // filters_multiplier)
        self.cell_0 = FirstCell(in_channels_left=filters, out_channels_left=filters // 2, in_channels_right=2 * filters, out_channels_right=filters)
        self.cell_1 = NormalCell(in_channels_left=2 * filters, out_channels_left=filters, in_channels_right=6 * filters, out_channels_right=filters)
        self.cell_2 = NormalCell(in_channels_left=6 * filters, out_channels_left=filters, in_channels_right=6 * filters, out_channels_right=filters)
        self.cell_3 = NormalCell(in_channels_left=6 * filters, out_channels_left=filters, in_channels_right=6 * filters, out_channels_right=filters)
        self.cell_4 = NormalCell(in_channels_left=6 * filters, out_channels_left=filters, in_channels_right=6 * filters, out_channels_right=filters)
        self.cell_5 = NormalCell(in_channels_left=6 * filters, out_channels_left=filters, in_channels_right=6 * filters, out_channels_right=filters)
        self.reduction_cell_0 = ReductionCell0(in_channels_left=6 * filters, out_channels_left=2 * filters, in_channels_right=6 * filters, out_channels_right=2 * filters)
        self.cell_6 = FirstCell(in_channels_left=6 * filters, out_channels_left=filters, in_channels_right=8 * filters, out_channels_right=2 * filters)
        self.cell_7 = NormalCell(in_channels_left=8 * filters, out_channels_left=2 * filters, in_channels_right=12 * filters, out_channels_right=2 * filters)
        self.cell_8 = NormalCell(in_channels_left=12 * filters, out_channels_left=2 * filters, in_channels_right=12 * filters, out_channels_right=2 * filters)
        self.cell_9 = NormalCell(in_channels_left=12 * filters, out_channels_left=2 * filters, in_channels_right=12 * filters, out_channels_right=2 * filters)
        self.cell_10 = NormalCell(in_channels_left=12 * filters, out_channels_left=2 * filters, in_channels_right=12 * filters, out_channels_right=2 * filters)
        self.cell_11 = NormalCell(in_channels_left=12 * filters, out_channels_left=2 * filters, in_channels_right=12 * filters, out_channels_right=2 * filters)
        self.reduction_cell_1 = ReductionCell1(in_channels_left=12 * filters, out_channels_left=4 * filters, in_channels_right=12 * filters, out_channels_right=4 * filters)
        self.cell_12 = FirstCell(in_channels_left=12 * filters, out_channels_left=2 * filters, in_channels_right=16 * filters, out_channels_right=4 * filters)
        self.cell_13 = NormalCell(in_channels_left=16 * filters, out_channels_left=4 * filters, in_channels_right=24 * filters, out_channels_right=4 * filters)
        self.cell_14 = NormalCell(in_channels_left=24 * filters, out_channels_left=4 * filters, in_channels_right=24 * filters, out_channels_right=4 * filters)
        self.cell_15 = NormalCell(in_channels_left=24 * filters, out_channels_left=4 * filters, in_channels_right=24 * filters, out_channels_right=4 * filters)
        self.cell_16 = NormalCell(in_channels_left=24 * filters, out_channels_left=4 * filters, in_channels_right=24 * filters, out_channels_right=4 * filters)
        self.cell_17 = NormalCell(in_channels_left=24 * filters, out_channels_left=4 * filters, in_channels_right=24 * filters, out_channels_right=4 * filters)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(11, stride=1, padding=0)
        self.dropout = nn.Dropout()
        self.last_linear = nn.Linear(24 * filters, self.num_classes)

    def features(self, input):
        x_conv0 = self.conv0(input)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.cell_3(x_cell_2, x_cell_1)
        x_cell_4 = self.cell_4(x_cell_3, x_cell_2)
        x_cell_5 = self.cell_5(x_cell_4, x_cell_3)
        x_reduction_cell_0 = self.reduction_cell_0(x_cell_5, x_cell_4)
        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_4)
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.cell_9(x_cell_8, x_cell_7)
        x_cell_10 = self.cell_10(x_cell_9, x_cell_8)
        x_cell_11 = self.cell_11(x_cell_10, x_cell_9)
        x_reduction_cell_1 = self.reduction_cell_1(x_cell_11, x_cell_10)
        x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_10)
        x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.cell_15(x_cell_14, x_cell_13)
        x_cell_16 = self.cell_16(x_cell_15, x_cell_14)
        x_cell_17 = self.cell_17(x_cell_16, x_cell_15)
        return x_cell_17

    def logits(self, features):
        x = self.relu(features)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class NASNetAMobile(nn.Module):
    """NASNetAMobile (4 @ 1056) """

    def __init__(self, num_classes=1000, stem_filters=32, penultimate_filters=1056, filters_multiplier=2):
        super(NASNetAMobile, self).__init__()
        self.num_classes = num_classes
        self.stem_filters = stem_filters
        self.penultimate_filters = penultimate_filters
        self.filters_multiplier = filters_multiplier
        filters = self.penultimate_filters // 24
        self.conv0 = nn.Sequential()
        self.conv0.add_module('conv', nn.Conv2d(in_channels=3, out_channels=self.stem_filters, kernel_size=3, padding=0, stride=2, bias=False))
        self.conv0.add_module('bn', nn.BatchNorm2d(self.stem_filters, eps=0.001, momentum=0.1, affine=True))
        self.cell_stem_0 = CellStem0(self.stem_filters, num_filters=filters // filters_multiplier ** 2)
        self.cell_stem_1 = CellStem1(self.stem_filters, num_filters=filters // filters_multiplier)
        self.cell_0 = FirstCell(in_channels_left=filters, out_channels_left=filters // 2, in_channels_right=2 * filters, out_channels_right=filters)
        self.cell_1 = NormalCell(in_channels_left=2 * filters, out_channels_left=filters, in_channels_right=6 * filters, out_channels_right=filters)
        self.cell_2 = NormalCell(in_channels_left=6 * filters, out_channels_left=filters, in_channels_right=6 * filters, out_channels_right=filters)
        self.cell_3 = NormalCell(in_channels_left=6 * filters, out_channels_left=filters, in_channels_right=6 * filters, out_channels_right=filters)
        self.reduction_cell_0 = ReductionCell0(in_channels_left=6 * filters, out_channels_left=2 * filters, in_channels_right=6 * filters, out_channels_right=2 * filters)
        self.cell_6 = FirstCell(in_channels_left=6 * filters, out_channels_left=filters, in_channels_right=8 * filters, out_channels_right=2 * filters)
        self.cell_7 = NormalCell(in_channels_left=8 * filters, out_channels_left=2 * filters, in_channels_right=12 * filters, out_channels_right=2 * filters)
        self.cell_8 = NormalCell(in_channels_left=12 * filters, out_channels_left=2 * filters, in_channels_right=12 * filters, out_channels_right=2 * filters)
        self.cell_9 = NormalCell(in_channels_left=12 * filters, out_channels_left=2 * filters, in_channels_right=12 * filters, out_channels_right=2 * filters)
        self.reduction_cell_1 = ReductionCell1(in_channels_left=12 * filters, out_channels_left=4 * filters, in_channels_right=12 * filters, out_channels_right=4 * filters)
        self.cell_12 = FirstCell(in_channels_left=12 * filters, out_channels_left=2 * filters, in_channels_right=16 * filters, out_channels_right=4 * filters)
        self.cell_13 = NormalCell(in_channels_left=16 * filters, out_channels_left=4 * filters, in_channels_right=24 * filters, out_channels_right=4 * filters)
        self.cell_14 = NormalCell(in_channels_left=24 * filters, out_channels_left=4 * filters, in_channels_right=24 * filters, out_channels_right=4 * filters)
        self.cell_15 = NormalCell(in_channels_left=24 * filters, out_channels_left=4 * filters, in_channels_right=24 * filters, out_channels_right=4 * filters)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(7, stride=1, padding=0)
        self.dropout = nn.Dropout()
        self.last_linear = nn.Linear(24 * filters, self.num_classes)

    def features(self, input):
        x_conv0 = self.conv0(input)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.cell_3(x_cell_2, x_cell_1)
        x_reduction_cell_0 = self.reduction_cell_0(x_cell_3, x_cell_2)
        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_3)
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.cell_9(x_cell_8, x_cell_7)
        x_reduction_cell_1 = self.reduction_cell_1(x_cell_9, x_cell_8)
        x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_9)
        x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.cell_15(x_cell_14, x_cell_13)
        return x_cell_15

    def logits(self, features):
        x = self.relu(features)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class FactorizedReduction(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FactorizedReduction, self).__init__()
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential(OrderedDict([('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False)), ('conv', nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, bias=False))]))
        self.path_2 = nn.Sequential(OrderedDict([('pad', nn.ZeroPad2d((0, 1, 0, 1))), ('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False)), ('conv', nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, bias=False))]))
        self.final_path_bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.relu(x)
        x_path1 = self.path_1(x)
        x_path2 = self.path_2.pad(x)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        out = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        return out


class Cell(CellBase):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right, is_reduction=False, zero_pad=False, match_prev_layer_dimensions=False):
        super(Cell, self).__init__()
        stride = 2 if is_reduction else 1
        self.match_prev_layer_dimensions = match_prev_layer_dimensions
        if match_prev_layer_dimensions:
            self.conv_prev_1x1 = FactorizedReduction(in_channels_left, out_channels_left)
        else:
            self.conv_prev_1x1 = ReluConvBn(in_channels_left, out_channels_left, kernel_size=1)
        self.conv_1x1 = ReluConvBn(in_channels_right, out_channels_right, kernel_size=1)
        self.comb_iter_0_left = BranchSeparables(out_channels_left, out_channels_left, kernel_size=5, stride=stride, zero_pad=zero_pad)
        self.comb_iter_0_right = MaxPool(3, stride=stride, zero_pad=zero_pad)
        self.comb_iter_1_left = BranchSeparables(out_channels_right, out_channels_right, kernel_size=7, stride=stride, zero_pad=zero_pad)
        self.comb_iter_1_right = MaxPool(3, stride=stride, zero_pad=zero_pad)
        self.comb_iter_2_left = BranchSeparables(out_channels_right, out_channels_right, kernel_size=5, stride=stride, zero_pad=zero_pad)
        self.comb_iter_2_right = BranchSeparables(out_channels_right, out_channels_right, kernel_size=3, stride=stride, zero_pad=zero_pad)
        self.comb_iter_3_left = BranchSeparables(out_channels_right, out_channels_right, kernel_size=3)
        self.comb_iter_3_right = MaxPool(3, stride=stride, zero_pad=zero_pad)
        self.comb_iter_4_left = BranchSeparables(out_channels_left, out_channels_left, kernel_size=3, stride=stride, zero_pad=zero_pad)
        if is_reduction:
            self.comb_iter_4_right = ReluConvBn(out_channels_right, out_channels_right, kernel_size=1, stride=stride)
        else:
            self.comb_iter_4_right = None

    def forward(self, x_left, x_right):
        x_left = self.conv_prev_1x1(x_left)
        x_right = self.conv_1x1(x_right)
        x_out = self.cell_forward(x_left, x_right)
        return x_out


class PNASNet5Large(nn.Module):

    def __init__(self, num_classes=1001):
        super(PNASNet5Large, self).__init__()
        self.num_classes = num_classes
        self.conv_0 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(3, 96, kernel_size=3, stride=2, bias=False)), ('bn', nn.BatchNorm2d(96, eps=0.001))]))
        self.cell_stem_0 = CellStem0(in_channels_left=96, out_channels_left=54, in_channels_right=96, out_channels_right=54)
        self.cell_stem_1 = Cell(in_channels_left=96, out_channels_left=108, in_channels_right=270, out_channels_right=108, match_prev_layer_dimensions=True, is_reduction=True)
        self.cell_0 = Cell(in_channels_left=270, out_channels_left=216, in_channels_right=540, out_channels_right=216, match_prev_layer_dimensions=True)
        self.cell_1 = Cell(in_channels_left=540, out_channels_left=216, in_channels_right=1080, out_channels_right=216)
        self.cell_2 = Cell(in_channels_left=1080, out_channels_left=216, in_channels_right=1080, out_channels_right=216)
        self.cell_3 = Cell(in_channels_left=1080, out_channels_left=216, in_channels_right=1080, out_channels_right=216)
        self.cell_4 = Cell(in_channels_left=1080, out_channels_left=432, in_channels_right=1080, out_channels_right=432, is_reduction=True, zero_pad=True)
        self.cell_5 = Cell(in_channels_left=1080, out_channels_left=432, in_channels_right=2160, out_channels_right=432, match_prev_layer_dimensions=True)
        self.cell_6 = Cell(in_channels_left=2160, out_channels_left=432, in_channels_right=2160, out_channels_right=432)
        self.cell_7 = Cell(in_channels_left=2160, out_channels_left=432, in_channels_right=2160, out_channels_right=432)
        self.cell_8 = Cell(in_channels_left=2160, out_channels_left=864, in_channels_right=2160, out_channels_right=864, is_reduction=True)
        self.cell_9 = Cell(in_channels_left=2160, out_channels_left=864, in_channels_right=4320, out_channels_right=864, match_prev_layer_dimensions=True)
        self.cell_10 = Cell(in_channels_left=4320, out_channels_left=864, in_channels_right=4320, out_channels_right=864)
        self.cell_11 = Cell(in_channels_left=4320, out_channels_left=864, in_channels_right=4320, out_channels_right=864)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(11, stride=1, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.last_linear = nn.Linear(4320, num_classes)

    def features(self, x):
        x_conv_0 = self.conv_0(x)
        x_stem_0 = self.cell_stem_0(x_conv_0)
        x_stem_1 = self.cell_stem_1(x_conv_0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_0, x_stem_1)
        x_cell_1 = self.cell_1(x_stem_1, x_cell_0)
        x_cell_2 = self.cell_2(x_cell_0, x_cell_1)
        x_cell_3 = self.cell_3(x_cell_1, x_cell_2)
        x_cell_4 = self.cell_4(x_cell_2, x_cell_3)
        x_cell_5 = self.cell_5(x_cell_3, x_cell_4)
        x_cell_6 = self.cell_6(x_cell_4, x_cell_5)
        x_cell_7 = self.cell_7(x_cell_5, x_cell_6)
        x_cell_8 = self.cell_8(x_cell_6, x_cell_7)
        x_cell_9 = self.cell_9(x_cell_7, x_cell_8)
        x_cell_10 = self.cell_10(x_cell_8, x_cell_9)
        x_cell_11 = self.cell_11(x_cell_9, x_cell_10)
        return x_cell_11

    def logits(self, features):
        x = self.relu(features)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class PolyConv2d(nn.Module):
    """A block that is used inside poly-N (poly-2, poly-3, and so on) modules.
    The Convolution layer is shared between all Inception blocks inside
    a poly-N module. BatchNorm layers are not shared between Inception blocks
    and therefore the number of BatchNorm layers is equal to the number of
    Inception blocks inside a poly-N module.
    """

    def __init__(self, in_planes, out_planes, kernel_size, num_blocks, stride=1, padding=0):
        super(PolyConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn_blocks = nn.ModuleList([nn.BatchNorm2d(out_planes) for _ in range(num_blocks)])
        self.relu = nn.ReLU()

    def forward(self, x, block_index):
        x = self.conv(x)
        bn = self.bn_blocks[block_index]
        x = bn(x)
        x = self.relu(x)
        return x


class BlockA(nn.Module):
    """Inception-ResNet-A block."""

    def __init__(self):
        super(BlockA, self).__init__()
        self.path0 = nn.Sequential(BasicConv2d(384, 32, kernel_size=1), BasicConv2d(32, 48, kernel_size=3, padding=1), BasicConv2d(48, 64, kernel_size=3, padding=1))
        self.path1 = nn.Sequential(BasicConv2d(384, 32, kernel_size=1), BasicConv2d(32, 32, kernel_size=3, padding=1))
        self.path2 = BasicConv2d(384, 32, kernel_size=1)
        self.conv2d = BasicConv2d(128, 384, kernel_size=1, output_relu=False)

    def forward(self, x):
        x0 = self.path0(x)
        x1 = self.path1(x)
        x2 = self.path2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        return out


class BlockB(nn.Module):
    """Inception-ResNet-B block."""

    def __init__(self):
        super(BlockB, self).__init__()
        self.path0 = nn.Sequential(BasicConv2d(1152, 128, kernel_size=1), BasicConv2d(128, 160, kernel_size=(1, 7), padding=(0, 3)), BasicConv2d(160, 192, kernel_size=(7, 1), padding=(3, 0)))
        self.path1 = BasicConv2d(1152, 192, kernel_size=1)
        self.conv2d = BasicConv2d(384, 1152, kernel_size=1, output_relu=False)

    def forward(self, x):
        x0 = self.path0(x)
        x1 = self.path1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        return out


class BlockC(nn.Module):
    """Inception-ResNet-C block."""

    def __init__(self):
        super(BlockC, self).__init__()
        self.path0 = nn.Sequential(BasicConv2d(2048, 192, kernel_size=1), BasicConv2d(192, 224, kernel_size=(1, 3), padding=(0, 1)), BasicConv2d(224, 256, kernel_size=(3, 1), padding=(1, 0)))
        self.path1 = BasicConv2d(2048, 192, kernel_size=1)
        self.conv2d = BasicConv2d(448, 2048, kernel_size=1, output_relu=False)

    def forward(self, x):
        x0 = self.path0(x)
        x1 = self.path1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        return out


class ReductionA(nn.Module):
    """A dimensionality reduction block that is placed after stage-a
    Inception-ResNet blocks.
    """

    def __init__(self):
        super(ReductionA, self).__init__()
        self.path0 = nn.Sequential(BasicConv2d(384, 256, kernel_size=1), BasicConv2d(256, 256, kernel_size=3, padding=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.path1 = BasicConv2d(384, 384, kernel_size=3, stride=2)
        self.path2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.path0(x)
        x1 = self.path1(x)
        x2 = self.path2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class ReductionB(nn.Module):
    """A dimensionality reduction block that is placed after stage-b
    Inception-ResNet blocks.
    """

    def __init__(self):
        super(ReductionB, self).__init__()
        self.path0 = nn.Sequential(BasicConv2d(1152, 256, kernel_size=1), BasicConv2d(256, 256, kernel_size=3, padding=1), BasicConv2d(256, 256, kernel_size=3, stride=2))
        self.path1 = nn.Sequential(BasicConv2d(1152, 256, kernel_size=1), BasicConv2d(256, 256, kernel_size=3, stride=2))
        self.path2 = nn.Sequential(BasicConv2d(1152, 256, kernel_size=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.path3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.path0(x)
        x1 = self.path1(x)
        x2 = self.path2(x)
        x3 = self.path3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionResNetBPoly(nn.Module):
    """Base class for constructing poly-N Inception-ResNet-B modules.
    When `num_blocks` is equal to 1, a module will have only a first-order path
    and will be equal to a standard Inception-ResNet-B block.
    When `num_blocks` is equal to 2, a module will have first-order and
    second-order paths and will be called Inception-ResNet-B poly-2 module.
    Increasing value of the `num_blocks` parameter will produce a higher order
    Inception-ResNet-B poly-N modules.
    """

    def __init__(self, scale, num_blocks):
        super(InceptionResNetBPoly, self).__init__()
        assert num_blocks >= 1, 'num_blocks should be greater or equal to 1'
        self.scale = scale
        self.num_blocks = num_blocks
        self.path0_1x1 = PolyConv2d(1152, 128, kernel_size=1, num_blocks=self.num_blocks)
        self.path0_1x7 = PolyConv2d(128, 160, kernel_size=(1, 7), num_blocks=self.num_blocks, padding=(0, 3))
        self.path0_7x1 = PolyConv2d(160, 192, kernel_size=(7, 1), num_blocks=self.num_blocks, padding=(3, 0))
        self.path1 = PolyConv2d(1152, 192, kernel_size=1, num_blocks=self.num_blocks)
        self.conv2d_blocks = nn.ModuleList([BasicConv2d(384, 1152, kernel_size=1, output_relu=False) for _ in range(self.num_blocks)])
        self.relu = nn.ReLU()

    def forward_block(self, x, block_index):
        x0 = self.path0_1x1(x, block_index)
        x0 = self.path0_1x7(x0, block_index)
        x0 = self.path0_7x1(x0, block_index)
        x1 = self.path1(x, block_index)
        out = torch.cat((x0, x1), 1)
        conv2d_block = self.conv2d_blocks[block_index]
        out = conv2d_block(out)
        return out

    def forward(self, x):
        out = x
        for block_index in range(self.num_blocks):
            x = self.forward_block(x, block_index)
            out = out + x * self.scale
            x = self.relu(x)
        out = self.relu(out)
        return out


class InceptionResNetCPoly(nn.Module):
    """Base class for constructing poly-N Inception-ResNet-C modules.
    When `num_blocks` is equal to 1, a module will have only a first-order path
    and will be equal to a standard Inception-ResNet-C block.
    When `num_blocks` is equal to 2, a module will have first-order and
    second-order paths and will be called Inception-ResNet-C poly-2 module.
    Increasing value of the `num_blocks` parameter will produce a higher order
    Inception-ResNet-C poly-N modules.
    """

    def __init__(self, scale, num_blocks):
        super(InceptionResNetCPoly, self).__init__()
        assert num_blocks >= 1, 'num_blocks should be greater or equal to 1'
        self.scale = scale
        self.num_blocks = num_blocks
        self.path0_1x1 = PolyConv2d(2048, 192, kernel_size=1, num_blocks=self.num_blocks)
        self.path0_1x3 = PolyConv2d(192, 224, kernel_size=(1, 3), num_blocks=self.num_blocks, padding=(0, 1))
        self.path0_3x1 = PolyConv2d(224, 256, kernel_size=(3, 1), num_blocks=self.num_blocks, padding=(1, 0))
        self.path1 = PolyConv2d(2048, 192, kernel_size=1, num_blocks=self.num_blocks)
        self.conv2d_blocks = nn.ModuleList([BasicConv2d(448, 2048, kernel_size=1, output_relu=False) for _ in range(self.num_blocks)])
        self.relu = nn.ReLU()

    def forward_block(self, x, block_index):
        x0 = self.path0_1x1(x, block_index)
        x0 = self.path0_1x3(x0, block_index)
        x0 = self.path0_3x1(x0, block_index)
        x1 = self.path1(x, block_index)
        out = torch.cat((x0, x1), 1)
        conv2d_block = self.conv2d_blocks[block_index]
        out = conv2d_block(out)
        return out

    def forward(self, x):
        out = x
        for block_index in range(self.num_blocks):
            x = self.forward_block(x, block_index)
            out = out + x * self.scale
            x = self.relu(x)
        out = self.relu(out)
        return out


class MultiWay(nn.Module):
    """Base class for constructing N-way modules (2-way, 3-way, and so on)."""

    def __init__(self, scale, block_cls, num_blocks):
        super(MultiWay, self).__init__()
        assert num_blocks >= 1, 'num_blocks should be greater or equal to 1'
        self.scale = scale
        self.blocks = nn.ModuleList([block_cls() for _ in range(num_blocks)])
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = out + block(x) * self.scale
        out = self.relu(out)
        return out


class InceptionResNetA2Way(MultiWay):

    def __init__(self, scale):
        super(InceptionResNetA2Way, self).__init__(scale, block_cls=BlockA, num_blocks=2)


class InceptionResNetB2Way(MultiWay):

    def __init__(self, scale):
        super(InceptionResNetB2Way, self).__init__(scale, block_cls=BlockB, num_blocks=2)


class InceptionResNetC2Way(MultiWay):

    def __init__(self, scale):
        super(InceptionResNetC2Way, self).__init__(scale, block_cls=BlockC, num_blocks=2)


class InceptionResNetBPoly3(InceptionResNetBPoly):

    def __init__(self, scale):
        super(InceptionResNetBPoly3, self).__init__(scale, num_blocks=3)


class InceptionResNetCPoly3(InceptionResNetCPoly):

    def __init__(self, scale):
        super(InceptionResNetCPoly3, self).__init__(scale, num_blocks=3)


class PolyNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(PolyNet, self).__init__()
        self.stem = Stem()
        self.stage_a = nn.Sequential(InceptionResNetA2Way(scale=1), InceptionResNetA2Way(scale=0.992308), InceptionResNetA2Way(scale=0.984615), InceptionResNetA2Way(scale=0.976923), InceptionResNetA2Way(scale=0.969231), InceptionResNetA2Way(scale=0.961538), InceptionResNetA2Way(scale=0.953846), InceptionResNetA2Way(scale=0.946154), InceptionResNetA2Way(scale=0.938462), InceptionResNetA2Way(scale=0.930769))
        self.reduction_a = ReductionA()
        self.stage_b = nn.Sequential(InceptionResNetBPoly3(scale=0.923077), InceptionResNetB2Way(scale=0.915385), InceptionResNetBPoly3(scale=0.907692), InceptionResNetB2Way(scale=0.9), InceptionResNetBPoly3(scale=0.892308), InceptionResNetB2Way(scale=0.884615), InceptionResNetBPoly3(scale=0.876923), InceptionResNetB2Way(scale=0.869231), InceptionResNetBPoly3(scale=0.861538), InceptionResNetB2Way(scale=0.853846), InceptionResNetBPoly3(scale=0.846154), InceptionResNetB2Way(scale=0.838462), InceptionResNetBPoly3(scale=0.830769), InceptionResNetB2Way(scale=0.823077), InceptionResNetBPoly3(scale=0.815385), InceptionResNetB2Way(scale=0.807692), InceptionResNetBPoly3(scale=0.8), InceptionResNetB2Way(scale=0.792308), InceptionResNetBPoly3(scale=0.784615), InceptionResNetB2Way(scale=0.776923))
        self.reduction_b = ReductionB()
        self.stage_c = nn.Sequential(InceptionResNetCPoly3(scale=0.769231), InceptionResNetC2Way(scale=0.761538), InceptionResNetCPoly3(scale=0.753846), InceptionResNetC2Way(scale=0.746154), InceptionResNetCPoly3(scale=0.738462), InceptionResNetC2Way(scale=0.730769), InceptionResNetCPoly3(scale=0.723077), InceptionResNetC2Way(scale=0.715385), InceptionResNetCPoly3(scale=0.707692), InceptionResNetC2Way(scale=0.7))
        self.avg_pool = nn.AvgPool2d(9, stride=1)
        self.dropout = nn.Dropout(0.2)
        self.last_linear = nn.Linear(2048, num_classes)

    def features(self, x):
        x = self.stem(x)
        x = self.stage_a(x)
        x = self.reduction_a(x)
        x = self.stage_b(x)
        x = self.reduction_b(x)
        x = self.stage_c(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


class LambdaBase(nn.Sequential):

    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):

    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


def identity(x):
    return x


class LambdaMap(LambdaBase):

    def __init__(self, *args):
        super(LambdaMap, self).__init__(*args)
        self.lambda_func = identity

    def forward(self, input):
        return list(map(self.lambda_func, self.forward_prepare(input)))


def add(x, y):
    return x + y


class LambdaReduce(LambdaBase):

    def __init__(self, *args):
        super(LambdaReduce, self).__init__(*args)
        self.lambda_func = add

    def forward(self, input):
        return reduce(self.lambda_func, self.forward_prepare(input))


class ResNeXt101_32x4d(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNeXt101_32x4d, self).__init__()
        self.num_classes = num_classes
        self.features = resnext101_32x4d_features
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        self.last_linear = nn.Linear(2048, num_classes)

    def logits(self, input):
        x = self.avg_pool(input)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class ResNeXt101_64x4d(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNeXt101_64x4d, self).__init__()
        self.num_classes = num_classes
        self.features = resnext101_64x4d_features
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        self.last_linear = nn.Linear(2048, num_classes)

    def logits(self, input):
        x = self.avg_pool(input)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2, inplanes=128, input_3x3=True, downsample_kernel_size=3, downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)), ('bn1', nn.BatchNorm2d(64)), ('relu1', nn.ReLU(inplace=True)), ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)), ('bn2', nn.BatchNorm2d(64)), ('relu2', nn.ReLU(inplace=True)), ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False)), ('bn3', nn.BatchNorm2d(inplanes)), ('relu3', nn.ReLU(inplace=True))]
        else:
            layer0_modules = [('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)), ('bn1', nn.BatchNorm2d(inplanes)), ('relu1', nn.ReLU(inplace=True))]
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(block, planes=64, blocks=layers[0], groups=groups, reduction=reduction, downsample_kernel_size=1, downsample_padding=0)
        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1, downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=downsample_kernel_size, stride=stride, padding=downsample_padding, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


class SpatialCrossMapLRN(nn.Module):

    def __init__(self, local_size=1, alpha=1.0, beta=0.75, k=1, ACROSS_CHANNELS=True):
        super(SpatialCrossMapLRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1), stride=1, padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size, stride=1, padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        x = x.div(div)
        return x


class VGGM(nn.Module):

    def __init__(self, num_classes=1000):
        super(VGGM, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(nn.Conv2d(3, 96, (7, 7), (2, 2)), nn.ReLU(), SpatialCrossMapLRN(5, 0.0005, 0.75, 2), nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True), nn.Conv2d(96, 256, (5, 5), (2, 2), (1, 1)), nn.ReLU(), SpatialCrossMapLRN(5, 0.0005, 0.75, 2), nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True), nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)), nn.ReLU(), nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True))
        self.classif = nn.Sequential(nn.Linear(18432, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classif(x)
        return x


class WideResNet(nn.Module):

    def __init__(self, pooling):
        super(WideResNet, self).__init__()
        self.pooling = pooling
        self.params = params

    def forward(self, x):
        x = f(x, self.params, self.pooling)
        return x


@torch.no_grad()
def convert_to_one_hot(x, minleng, ignore_idx=-1):
    """
    encode input x into one hot
    inputs:
        x: tensor of shape (N, ...) with type long
        minleng: minimum length of one hot code, this should be larger than max value in x
        ignore_idx: the index in x that should be ignored, default is 255

    return:
        tensor of shape (N, minleng, ...) with type float
    """
    device = x.device
    size = list(x.size())
    size.insert(1, minleng)
    assert x[x != ignore_idx].max() < minleng, 'minleng should larger than max value in x'
    if ignore_idx < 0:
        out = torch.zeros(size, device=device).scatter_(1, x.unsqueeze(1), 1)
    else:
        x = x.clone().detach()
        ignore = x == ignore_idx
        x[ignore] = 0
        out = torch.zeros(size, device=device).scatter_(1, x.unsqueeze(1), 1)
        ignore = ignore.nonzero(as_tuple=False)
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        out[[a, torch.arange(minleng), *b]] = 0
    return out


class AffinityLoss(nn.Module):

    def __init__(self, kernel_size=3, ignore_index=-100):
        super(AffinityLoss, self).__init__()
        self.kernel_size = kernel_size
        self.ignore_index = ignore_index
        self.unfold = nn.Unfold(kernel_size=kernel_size)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, logits, labels):
        """
        usage similar to nn.CrossEntropyLoss:
            >>> criteria = AffinityLoss(kernel_size=3, ignore_index=255)
            >>> logits = torch.randn(8, 19, 384, 384) # nchw
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw
            >>> loss = criteria(logits, lbs)
        """
        n, c, h, w = logits.size()
        context_size = self.kernel_size * self.kernel_size
        lb_one_hot = convert_to_one_hot(labels, c, self.ignore_index).detach()
        logits_unfold = self.unfold(logits).view(n, c, context_size, -1)
        lbs_unfold = self.unfold(lb_one_hot).view(n, c, context_size, -1)
        aff_map = torch.einsum('ncal,ncbl->nabl', logits_unfold, logits_unfold)
        lb_map = torch.einsum('ncal,ncbl->nabl', lbs_unfold, lbs_unfold)
        loss = self.bce(aff_map, lb_map)
        return loss


class AffinityFieldLoss(nn.Module):
    """
        loss proposed in the paper: https://arxiv.org/abs/1803.10335
        used for sigmentation tasks
    """

    def __init__(self, kl_margin, lambda_edge=1.0, lambda_not_edge=1.0, ignore_lb=255):
        super(AffinityFieldLoss, self).__init__()
        self.kl_margin = kl_margin
        self.ignore_lb = ignore_lb
        self.lambda_edge = lambda_edge
        self.lambda_not_edge = lambda_not_edge
        self.kldiv = nn.KLDivLoss(reduction='none')

    def forward(self, logits, labels):
        ignore_mask = labels.cpu() == self.ignore_lb
        n_valid = ignore_mask.numel() - ignore_mask.sum().item()
        indices = [((1, None, None, None), (None, -1, None, None)), ((None, -1, None, None), (1, None, None, None)), ((None, None, 1, None), (None, None, None, -1)), ((None, None, None, -1), (None, None, 1, None)), ((1, None, 1, None), (None, -1, None, -1)), ((1, None, None, -1), (None, -1, 1, None)), ((None, -1, 1, None), (1, None, None, -1)), ((None, -1, None, -1), (1, None, 1, None))]
        losses = []
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log_softmax(logits, dim=1)
        for idx_c, idx_e in indices:
            lbcenter = labels[:, idx_c[0]:idx_c[1], idx_c[2]:idx_c[3]].detach()
            lbedge = labels[:, idx_e[0]:idx_e[1], idx_e[2]:idx_e[3]].detach()
            igncenter = ignore_mask[:, idx_c[0]:idx_c[1], idx_c[2]:idx_c[3]].detach()
            ignedge = ignore_mask[:, idx_e[0]:idx_e[1], idx_e[2]:idx_e[3]].detach()
            lgp_center = probs[:, :, idx_c[0]:idx_c[1], idx_c[2]:idx_c[3]]
            lgp_edge = probs[:, :, idx_e[0]:idx_e[1], idx_e[2]:idx_e[3]]
            prob_edge = probs[:, :, idx_e[0]:idx_e[1], idx_e[2]:idx_e[3]]
            kldiv = (prob_edge * (lgp_edge - lgp_center)).sum(dim=1)
            kldiv[ignedge | igncenter] = 0
            loss = torch.where(lbcenter == lbedge, self.lambda_edge * kldiv, self.lambda_not_edge * F.relu(self.kl_margin - kldiv, inplace=True)).sum() / n_valid
            losses.append(loss)
        return sum(losses) / 8


class AMSoftmax(nn.Module):

    def __init__(self, in_feats, n_classes=10, m=0.3, s=15):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, lb):
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-09)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-09)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        delt_costh = torch.zeros_like(costh).scatter_(1, lb.unsqueeze(1), self.m)
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, lb)
        return loss


class CoordConv2d(nn.Conv2d):

    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(CoordConv2d, self).__init__(in_chan + 2, out_chan, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        batchsize, H, W = x.size(0), x.size(2), x.size(3)
        h_range = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
        w_range = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
        h_chan, w_chan = torch.meshgrid(h_range, w_range)
        h_chan = h_chan.expand([batchsize, 1, -1, -1])
        w_chan = w_chan.expand([batchsize, 1, -1, -1])
        feat = torch.cat([h_chan, w_chan, x], dim=1)
        return F.conv2d(feat, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class DY_Conv2d(nn.Conv2d):

    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, act=nn.ReLU(inplace=True), K=4, temperature=30, temp_anneal_steps=3000):
        super(DY_Conv2d, self).__init__(in_chan, out_chan * K, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        assert in_chan // 4 > 0
        self.K = K
        self.act = act
        self.se_conv1 = nn.Conv2d(in_chan, in_chan // 4, 1, 1, 0, bias=True)
        self.se_conv2 = nn.Conv2d(in_chan // 4, K, 1, 1, 0, bias=True)
        self.temperature = temperature
        self.temp_anneal_steps = temp_anneal_steps
        self.temp_interval = (temperature - 1) / temp_anneal_steps

    def get_atten(self, x):
        bs, _, h, w = x.size()
        atten = torch.mean(x, dim=(2, 3), keepdim=True)
        atten = self.se_conv1(atten)
        atten = self.act(atten)
        atten = self.se_conv2(atten)
        if self.training and self.temp_anneal_steps > 0:
            atten = atten / self.temperature
            self.temperature -= self.temp_interval
            self.temp_anneal_steps -= 1
        atten = atten.softmax(dim=1).view(bs, -1)
        return atten

    def forward(self, x):
        bs, _, h, w = x.size()
        atten = self.get_atten(x)
        out_chan, in_chan, k1, k2 = self.weight.size()
        W = self.weight.view(1, self.K, -1, in_chan, k1, k2)
        W = (W * atten.view(bs, self.K, 1, 1, 1, 1)).sum(dim=1)
        W = W.view(-1, in_chan, k1, k2)
        b = self.bias
        if not b is None:
            b = b.view(1, self.K, -1)
            b = (b * atten.view(bs, self.K, 1)).sum(dim=1).view(-1)
        x = x.view(1, -1, h, w)
        out = F.conv2d(x, W, b, self.stride, self.padding, self.dilation, self.groups * bs)
        out = out.view(bs, -1, out.size(2), out.size(3))
        return out


class GeneralizedSoftDiceLoss(nn.Module):

    def __init__(self, p=1, smooth=1, reduction='mean', weight=None, ignore_lb=255):
        super(GeneralizedSoftDiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth
        self.reduction = reduction
        self.weight = None if weight is None else torch.tensor(weight)
        self.ignore_lb = ignore_lb

    def forward(self, logits, label):
        """
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        """
        logits = logits.float()
        ignore = label.data.cpu() == self.ignore_lb
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = torch.zeros_like(logits).scatter_(1, label.unsqueeze(1), 1)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        lb_one_hot[[a, torch.arange(lb_one_hot.size(1)).long(), *b]] = 0
        lb_one_hot = lb_one_hot.detach()
        probs = torch.sigmoid(logits)
        numer = torch.sum(probs * lb_one_hot, dim=(2, 3))
        denom = torch.sum(probs.pow(self.p) + lb_one_hot.pow(self.p), dim=(2, 3))
        if not self.weight is None:
            numer = numer * self.weight.view(1, -1)
            denom = denom * self.weight.view(1, -1)
        numer = torch.sum(numer, dim=1)
        denom = torch.sum(denom, dim=1)
        loss = 1 - (2 * numer + self.smooth) / (denom + self.smooth)
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss


class BatchSoftDiceLoss(nn.Module):

    def __init__(self, p=1, smooth=1, weight=None, ignore_lb=255):
        super(BatchSoftDiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth
        self.weight = None if weight is None else torch.tensor(weight)
        self.ignore_lb = ignore_lb

    def forward(self, logits, label):
        """
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        """
        logits = logits.float()
        ignore = label.data.cpu() == self.ignore_lb
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = torch.zeros_like(logits).scatter_(1, label.unsqueeze(1), 1)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        lb_one_hot[[a, torch.arange(lb_one_hot.size(1)).long(), *b]] = 0
        lb_one_hot = lb_one_hot.detach()
        probs = torch.sigmoid(logits)
        numer = torch.sum(probs * lb_one_hot, dim=(2, 3))
        denom = torch.sum(probs.pow(self.p) + lb_one_hot.pow(self.p), dim=(2, 3))
        if not self.weight is None:
            numer = numer * self.weight.view(1, -1)
            denom = denom * self.weight.view(1, -1)
        numer = torch.sum(numer)
        denom = torch.sum(denom)
        loss = 1 - (2 * numer + self.smooth) / (denom + self.smooth)
        return loss


class Dual_Focal_loss(nn.Module):
    """
    This loss is proposed in this paper: https://arxiv.org/abs/1909.11932
    It does not work in my projects, hope it will work well in your projects.
    Hope you can correct me if there are any mistakes in the implementation.
    """

    def __init__(self, ignore_lb=255, eps=1e-05, reduction='mean'):
        super(Dual_Focal_loss, self).__init__()
        self.ignore_lb = ignore_lb
        self.eps = eps
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, logits, label):
        ignore = label.data.cpu() == self.ignore_lb
        n_valid = (ignore == 0).sum()
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1).detach()
        pred = torch.softmax(logits, dim=1)
        loss = -torch.log(self.eps + 1.0 - self.mse(pred, lb_one_hot)).sum(dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            loss = loss
        return loss


class FocalLossV1(nn.Module):

    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        """
        logits and label have same shape, and label data type is long
        args:
            logits: tensor of shape (N, ...)
            label: tensor of shape(N, ...)
        Usage is like this:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)# nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nchw, int64_t
            >>> loss = criteria(logits, lbs)
        """
        logits = logits.float()
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha
        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = alpha * torch.pow(1 - pt, self.gamma) * ce_loss
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class FocalSigmoidLossFuncV2(torch.autograd.Function):
    """
    compute backward directly for better numeric stability
    """

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, label, alpha, gamma):
        logits = logits.float()
        coeff = torch.empty_like(logits).fill_(1 - alpha)
        coeff[label == 1] = alpha
        probs = torch.sigmoid(logits)
        log_probs = torch.where(logits >= 0, F.softplus(logits, -1, 50), logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0, -logits + F.softplus(logits, -1, 50), -F.softplus(logits, 1, 50))
        probs_gamma = probs ** gamma
        probs_1_gamma = (1.0 - probs) ** gamma
        ctx.vars = coeff, probs, log_probs, log_1_probs, probs_gamma, probs_1_gamma, label, gamma
        term1 = probs_1_gamma * log_probs
        term2 = probs_gamma * log_1_probs
        loss = torch.where(label == 1, term1, term2).mul_(coeff).neg_()
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        """
        compute gradient of focal loss
        """
        coeff, probs, log_probs, log_1_probs, probs_gamma, probs_1_gamma, label, gamma = ctx.vars
        term1 = (1.0 - probs - gamma * probs * log_probs).mul_(probs_1_gamma).neg_()
        term2 = (probs - gamma * (1.0 - probs) * log_1_probs).mul_(probs_gamma)
        grads = torch.where(label == 1, term1, term2).mul_(coeff).mul_(grad_output)
        return grads, None, None, None


class FocalLossV2(nn.Module):
    """
    This use better formula to compute the gradient, which has better numeric stability
    Usage is like this:
        >>> criteria = FocalLossV2()
        >>> logits = torch.randn(8, 19, 384, 384)# nchw, float/half
        >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nchw, int64_t
        >>> loss = criteria(logits, lbs)
    """

    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLossV2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, label):
        loss = FocalSigmoidLossFuncV2.apply(logits, label, self.alpha, self.gamma)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class FocalSigmoidLossFuncV3(torch.autograd.Function):
    """
    use cpp/cuda to accelerate and shrink memory usage
    """

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, labels, alpha, gamma):
        logits = logits.float()
        loss = focal_cpp.focalloss_forward(logits, labels, gamma, alpha)
        ctx.variables = logits, labels, alpha, gamma
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        """
        compute gradient of focal loss
        """
        logits, labels, alpha, gamma = ctx.variables
        grads = focal_cpp.focalloss_backward(grad_output, logits, labels, gamma, alpha)
        return grads, None, None, None


class FocalLossV3(nn.Module):
    """
    This use better formula to compute the gradient, which has better numeric stability
    Usage is like this:
        >>> criteria = FocalLossV3()
        >>> logits = torch.randn(8, 19, 384, 384)# nchw, float/half
        >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nchw, int64_t
        >>> loss = criteria(logits, lbs)
    """

    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLossV3, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, label):
        loss = FocalSigmoidLossFuncV3.apply(logits, label, self.alpha, self.gamma)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class FReLU(nn.Module):

    def __init__(self, in_chan):
        super(FReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, in_chan, 3, 1, 1, groups=in_chan)
        self.bn = nn.BatchNorm2d(in_chan)
        nn.init.xavier_normal_(self.conv.weight, gain=1.0)

    def forward(self, x):
        branch = self.bn(self.conv(x))
        out = torch.max(x, branch)
        return out


class HSwishV1(nn.Module):

    def __init__(self):
        super(HSwishV1, self).__init__()

    def forward(self, feat):
        return feat * F.relu6(feat + 3) / 6


class HSwishFunctionV2(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, feat):
        act = F.relu6(feat + 3).mul_(feat).div_(6)
        ctx.variables = feat
        return act

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        feat = ctx.variables
        grad = F.relu6(feat + 3).div_(6)
        grad.add_(torch.where(torch.eq(-3 < feat, feat < 3), torch.ones_like(feat).div_(6), torch.zeros_like(feat)).mul_(feat))
        grad *= grad_output
        return grad


class HSwishV2(nn.Module):

    def __init__(self):
        super(HSwishV2, self).__init__()

    def forward(self, feat):
        return HSwishFunctionV2.apply(feat)


class HSwishFunctionV3(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, feat):
        ctx.feat = feat
        return swish_cpp.hswish_forward(feat)

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        feat = ctx.feat
        return swish_cpp.hswish_backward(grad_output, feat)


class HSwishV3(nn.Module):

    def __init__(self):
        super(HSwishV3, self).__init__()

    def forward(self, feat):
        return HSwishFunctionV3.apply(feat)


class LabelSmoothSoftmaxCEV1(nn.Module):
    """
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    """

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        """
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        """
        logits = logits.float()
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1.0 - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()
        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class LSRCrossEntropyFunctionV2(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, label, lb_smooth, lb_ignore):
        num_classes = logits.size(1)
        lb_pos, lb_neg = 1.0 - lb_smooth, lb_smooth / num_classes
        label = label.clone().detach()
        ignore = label.eq(lb_ignore)
        n_valid = ignore.eq(0).sum()
        label[ignore] = 0
        lb_one_hot = torch.empty_like(logits).fill_(lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()
        ignore = ignore.nonzero(as_tuple=False)
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        mask = [a, torch.arange(logits.size(1)), *b]
        lb_one_hot[mask] = 0
        coeff = (num_classes - 1) * lb_neg + lb_pos
        ctx.variables = coeff, mask, logits, lb_one_hot
        loss = torch.log_softmax(logits, dim=1).neg_().mul_(lb_one_hot).sum(dim=1)
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        coeff, mask, logits, lb_one_hot = ctx.variables
        scores = torch.softmax(logits, dim=1).mul_(coeff)
        grad = scores.sub_(lb_one_hot).mul_(grad_output.unsqueeze(1))
        grad[mask] = 0
        return grad, None, None, None


class LabelSmoothSoftmaxCEV2(nn.Module):

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV2, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index

    def forward(self, logits, labels):
        """
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV2()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        """
        losses = LSRCrossEntropyFunctionV2.apply(logits, labels, self.lb_smooth, self.lb_ignore)
        if self.reduction == 'sum':
            losses = losses.sum()
        elif self.reduction == 'mean':
            n_valid = (labels != self.lb_ignore).sum()
            losses = losses.sum() / n_valid
        return losses


class LSRCrossEntropyFunctionV3(torch.autograd.Function):
    """
    use cpp/cuda to accelerate and shrink memory usage
    """

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, labels, lb_smooth, lb_ignore):
        losses = lsr_cpp.lsr_forward(logits, labels, lb_ignore, lb_smooth)
        ctx.variables = logits, labels, lb_ignore, lb_smooth
        return losses

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        logits, labels, lb_ignore, lb_smooth = ctx.variables
        grad = lsr_cpp.lsr_backward(logits, labels, lb_ignore, lb_smooth)
        grad.mul_(grad_output.unsqueeze(1))
        return grad, None, None, None


class LabelSmoothSoftmaxCEV3(nn.Module):

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV3, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index

    def forward(self, logits, labels):
        """
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV3()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        """
        losses = LSRCrossEntropyFunctionV3.apply(logits, labels, self.lb_smooth, self.lb_ignore)
        if self.reduction == 'sum':
            losses = losses.sum()
        elif self.reduction == 'mean':
            n_valid = (labels != self.lb_ignore).sum()
            losses = losses.sum() / n_valid
        return losses


class LargeMarginSoftmaxV1(nn.Module):

    def __init__(self, lam=0.3, reduction='mean', ignore_index=255):
        super(LargeMarginSoftmaxV1, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lam = lam
        self.ce_crit = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    def forward(self, logits, label):
        """
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LargeMarginSoftmaxV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        """
        logits = logits.float()
        logits.retain_grad()
        logits.register_hook(lambda grad: grad)
        with torch.no_grad():
            num_classes = logits.size(1)
            coeff = 1.0 / (num_classes - 1.0)
            lb = label.clone().detach()
            mask = label == self.ignore_index
            lb[mask] = 0
            idx = torch.zeros_like(logits).scatter_(1, lb.unsqueeze(1), 1.0)
        lgts = logits - idx * 1000000.0
        q = lgts.softmax(dim=1)
        q = q * (1.0 - idx)
        log_q = lgts.log_softmax(dim=1)
        log_q = log_q * (1.0 - idx)
        mg_loss = (q - coeff) * log_q * (self.lam / 2)
        mg_loss = mg_loss * (1.0 - idx)
        mg_loss = mg_loss.sum(dim=1)
        ce_loss = self.ce_crit(logits, label)
        loss = ce_loss + mg_loss
        loss = loss[mask == 0]
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class LargeMarginSoftmaxFuncV2(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, labels, lam=0.3):
        num_classes = logits.size(1)
        coeff = 1.0 / (num_classes - 1.0)
        idx = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1.0)
        lgts = logits.clone()
        lgts[idx.bool()] = -1000000.0
        q = lgts.softmax(dim=1)
        log_q = lgts.log_softmax(dim=1)
        losses = q.sub_(coeff).mul_(log_q).mul_(lam / 2.0)
        losses[idx.bool()] = 0
        losses = losses.sum(dim=1).add_(F.cross_entropy(logits, labels, reduction='none'))
        ctx.variables = logits, labels, idx, coeff, lam
        return losses

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        """
        compute gradient
        """
        logits, labels, idx, coeff, lam = ctx.variables
        num_classes = logits.size(1)
        p = logits.softmax(dim=1)
        lgts = logits.clone()
        lgts[idx.bool()] = -1000000.0
        q = lgts.softmax(dim=1)
        qx = q * lgts
        qx[idx.bool()] = 0
        grad = qx + q - q * qx.sum(dim=1).unsqueeze(1) - coeff
        grad = grad * lam / 2.0
        grad[idx.bool()] = -1
        grad = grad + p
        grad.mul_(grad_output.unsqueeze(1))
        return grad, None, None


class LargeMarginSoftmaxV2(nn.Module):

    def __init__(self, lam=0.3, reduction='mean', ignore_index=255):
        super(LargeMarginSoftmaxV2, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lam = lam

    def forward(self, logits, labels):
        """
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LargeMarginSoftmaxV2()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        """
        logits = logits.float()
        mask = labels == self.ignore_index
        lb = labels.clone().detach()
        lb[mask] = 0
        loss = LargeMarginSoftmaxFuncV2.apply(logits, lb, self.lam)
        loss = loss[mask == 0]
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class LargeMarginSoftmaxFuncV3(torch.autograd.Function):
    """
    use cpp/cuda to accelerate and shrink memory usage
    """

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, labels, lam=0.3, ignore_index=255):
        losses = large_margin_cpp.l_margin_forward(logits, labels, lam, ignore_index)
        ctx.variables = logits, labels, lam, ignore_index
        return losses

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        """
        compute gradient
        """
        logits, labels, lam, ignore_index = ctx.variables
        grads = large_margin_cpp.l_margin_backward(logits, labels, lam, ignore_index)
        grads.mul_(grad_output.unsqueeze(1))
        return grads, None, None, None


class LargeMarginSoftmaxV3(nn.Module):

    def __init__(self, lam=0.3, reduction='mean', ignore_index=255):
        super(LargeMarginSoftmaxV3, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lam = lam

    def forward(self, logits, labels):
        """
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LargeMarginSoftmaxV3()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        """
        logits = logits.float()
        losses = LargeMarginSoftmaxFuncV3.apply(logits, labels, self.lam, self.ignore_index)
        if self.reduction == 'mean':
            n_valid = (labels != self.ignore_index).sum()
            losses = losses.sum() / n_valid
        elif self.reduction == 'sum':
            losses = losses.sum()
        return losses


class LovaszSoftmaxV1(nn.Module):
    """
    This is the autograd version, used in the multi-category classification case
    """

    def __init__(self, reduction='mean', ignore_index=-100):
        super(LovaszSoftmaxV1, self).__init__()
        self.reduction = reduction
        self.lb_ignore = ignore_index

    def forward(self, logits, label):
        """
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LovaszSoftmaxV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        """
        n, c, h, w = logits.size()
        logits = logits.transpose(0, 1).reshape(c, -1).float()
        label = label.view(-1)
        idx = label.ne(self.lb_ignore).nonzero(as_tuple=False).squeeze()
        probs = logits.softmax(dim=0)[:, idx]
        label = label[idx]
        lb_one_hot = torch.zeros_like(probs).scatter_(0, label.unsqueeze(0), 1).detach()
        errs = (lb_one_hot - probs).abs()
        errs_sort, errs_order = torch.sort(errs, dim=1, descending=True)
        n_samples = errs.size(1)
        with torch.no_grad():
            lb_one_hot_sort = torch.cat([lb_one_hot[i, ord].unsqueeze(0) for i, ord in enumerate(errs_order)], dim=0)
            n_pos = lb_one_hot_sort.sum(dim=1, keepdim=True)
            inter = n_pos - lb_one_hot_sort.cumsum(dim=1)
            union = n_pos + (1.0 - lb_one_hot_sort).cumsum(dim=1)
            jacc = 1.0 - inter / union
            if n_samples > 1:
                jacc[:, 1:] = jacc[:, 1:] - jacc[:, :-1]
        losses = torch.einsum('ab,ab->a', errs_sort, jacc)
        if self.reduction == 'sum':
            losses = losses.sum()
        elif self.reduction == 'mean':
            losses = losses.mean()
        return losses, errs


class LovaszSoftmaxFunctionV3(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, labels, ignore_index):
        losses, jacc = lovasz_softmax_cpp.lovasz_softmax_forward(logits, labels, ignore_index)
        ctx.vars = logits, labels, jacc, ignore_index
        return losses

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        logits, labels, jacc, ignore_index = ctx.vars
        grad = lovasz_softmax_cpp.lovasz_softmax_backward(grad_output, logits, labels, jacc, ignore_index)
        return grad, None, None


class LovaszSoftmaxV3(nn.Module):
    """
    """

    def __init__(self, reduction='mean', ignore_index=-100):
        super(LovaszSoftmaxV3, self).__init__()
        self.reduction = reduction
        self.lb_ignore = ignore_index

    def forward(self, logits, label):
        """
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LovaszSoftmaxV3()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        """
        losses = LovaszSoftmaxFunctionV3.apply(logits, label, self.lb_ignore)
        if self.reduction == 'sum':
            losses = losses.sum()
        elif self.reduction == 'mean':
            losses = losses.mean()
        return losses


class MishV1(nn.Module):

    def __init__(self):
        super(MishV1, self).__init__()

    def forward(self, feat):
        return feat * torch.tanh(F.softplus(feat))


class MishFunctionV2(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, feat):
        tanhX = torch.tanh(F.softplus(feat))
        out = feat * tanhX
        grad = tanhX + feat * (1 - torch.pow(tanhX, 2)) * torch.sigmoid(feat)
        ctx.grad = grad
        return out

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        grad = ctx.grad
        grad *= grad_output
        return grad


class MishV2(nn.Module):

    def __init__(self):
        super(MishV2, self).__init__()

    def forward(self, feat):
        return MishFunctionV2.apply(feat)


class MishFunctionV3(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, feat):
        ctx.feat = feat
        return mish_cpp.mish_forward(feat)

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        feat = ctx.feat
        return mish_cpp.mish_backward(grad_output, feat)


class MishV3(nn.Module):

    def __init__(self):
        super(MishV3, self).__init__()

    def forward(self, feat):
        return MishFunctionV3.apply(feat)


class OhemCELoss(nn.Module):

    def __init__(self, score_thresh, n_min=None, ignore_index=255):
        super(OhemCELoss, self).__init__()
        self.score_thresh = score_thresh
        self.ignore_lb = ignore_index
        self.n_min = n_min
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')

    def forward(self, logits, labels):
        n_min = labels.numel() // 16 if self.n_min is None else self.n_min
        labels = ohem_cpp.score_ohem_label(logits.float(), labels, self.ignore_lb, self.score_thresh, n_min).detach()
        loss = self.criteria(logits, labels)
        return loss


class OhemLargeMarginLoss(nn.Module):

    def __init__(self, score_thresh, n_min=None, ignore_index=255):
        super(OhemLargeMarginLoss, self).__init__()
        self.score_thresh = score_thresh
        self.ignore_lb = ignore_index
        self.n_min = n_min
        self.criteria = LargeMarginSoftmaxV3(ignore_index=ignore_index, reduction='mean')

    def forward(self, logits, labels):
        n_min = labels.numel() // 16 if self.n_min is None else self.n_min
        labels = ohem_cpp.score_ohem_label(logits.float(), labels, self.ignore_lb, self.score_thresh, n_min).detach()
        loss = self.criteria(logits, labels)
        return loss


def convert_to_one_hot_cu(x, minleng, smooth=0.0, ignore_idx=-1):
    """
    cuda version of encoding x into one hot, the difference from above is that, this support label smooth.
    inputs:
        x: tensor of shape (N, ...) with type long
        minleng: minimum length of one hot code, this should be larger than max value in x
        smooth: sets positive to **1. - smooth**, while sets negative to **smooth / minleng**
        ignore_idx: the index in x that should be ignored, default is 255

    return:
        tensor of shape (N, minleng, ...) with type float32
    """
    return one_hot_cpp.label_one_hot(x, ignore_idx, smooth, minleng)


class OnehotEncoder(nn.Module):

    def __init__(self, n_classes, lb_smooth=0.0, ignore_idx=-1):
        super(OnehotEncoder, self).__init__()
        self.n_classes = n_classes
        self.lb_smooth = lb_smooth
        self.ignore_idx = ignore_idx

    @torch.no_grad()
    def forward(self, label):
        return convert_to_one_hot_cu(label, self.n_classes, self.lb_smooth, self.ignore_idx).detach()


def pc_softmax_func(logits, lb_proportion):
    assert logits.size(1) == len(lb_proportion)
    shape = [1, -1] + [(1) for _ in range(len(logits.size()) - 2)]
    W = torch.tensor(lb_proportion).view(*shape).detach()
    logits = logits - logits.max(dim=1, keepdim=True)[0]
    exp = torch.exp(logits)
    pc_softmax = exp.div_((W * exp).sum(dim=1, keepdim=True))
    return pc_softmax


class PCSoftmax(nn.Module):

    def __init__(self, lb_proportion):
        super(PCSoftmax, self).__init__()
        self.weight = lb_proportion

    def forward(self, logits):
        return pc_softmax_func(logits, self.weight)


class PCSoftmaxCrossEntropyV1(nn.Module):

    def __init__(self, lb_proportion, ignore_index=255, reduction='mean'):
        super(PCSoftmaxCrossEntropyV1, self).__init__()
        self.weight = torch.tensor(lb_proportion).detach()
        self.nll = nn.NLLLoss(reduction=reduction, ignore_index=ignore_index)

    def forward(self, logits, label):
        shape = [1, -1] + [(1) for _ in range(len(logits.size()) - 2)]
        W = self.weight.view(*shape).detach()
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        wexp_sum = torch.exp(logits).mul(W).sum(dim=1, keepdim=True)
        log_wsoftmax = logits - torch.log(wexp_sum)
        loss = self.nll(log_wsoftmax, label)
        return loss


class PCSoftmaxCrossEntropyFunction(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, logits, label, lb_proportion, reduction, ignore_index):
        label = label.clone().detach()
        ignore = label == ignore_index
        n_valid = (ignore == 0).sum()
        label[ignore] = 0
        lb_one_hot = torch.zeros_like(logits).scatter_(1, label.unsqueeze(1), 1).detach()
        shape = [1, -1] + [(1) for _ in range(len(logits.size()) - 2)]
        W = torch.tensor(lb_proportion).view(*shape).detach()
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        exp_wsum = torch.exp(logits).mul_(W).sum(dim=1, keepdim=True)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        mask = [a, torch.arange(lb_one_hot.size(1)), *b]
        lb_one_hot[mask] = 0
        ctx.mask = mask
        ctx.W = W
        ctx.lb_one_hot = lb_one_hot
        ctx.logits = logits
        ctx.exp_wsum = exp_wsum
        ctx.reduction = reduction
        ctx.n_valid = n_valid
        log_wsoftmax = logits - torch.log(exp_wsum)
        loss = -log_wsoftmax.mul_(lb_one_hot).sum(dim=1)
        if reduction == 'mean':
            loss = loss.sum().div_(n_valid)
        if reduction == 'sum':
            loss = loss.sum()
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        mask = ctx.mask
        W = ctx.W
        lb_one_hot = ctx.lb_one_hot
        logits = ctx.logits
        exp_wsum = ctx.exp_wsum
        reduction = ctx.reduction
        n_valid = ctx.n_valid
        wlabel = torch.sum(W * lb_one_hot, dim=1, keepdim=True)
        wscores = torch.exp(logits).div_(exp_wsum).mul_(wlabel)
        wscores[mask] = 0
        grad = wscores.sub_(lb_one_hot)
        if reduction == 'none':
            grad.mul_(grad_output.unsqueeze(1))
        elif reduction == 'sum':
            grad.mul_(grad_output)
        elif reduction == 'mean':
            grad.div_(n_valid).mul_(grad_output)
        return grad, None, None, None, None, None


class PCSoftmaxCrossEntropyV2(nn.Module):

    def __init__(self, lb_proportion, reduction='mean', ignore_index=-100):
        super(PCSoftmaxCrossEntropyV2, self).__init__()
        self.lb_proportion = lb_proportion
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, label):
        return PCSoftmaxCrossEntropyFunction.apply(logits, label, self.lb_proportion, self.reduction, self.ignore_index)


class SoftDiceLossV1(nn.Module):
    """
    soft-dice loss, useful in binary segmentation
    """

    def __init__(self, p=1, smooth=1):
        super(SoftDiceLossV1, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        """
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        """
        probs = torch.sigmoid(logits)
        numer = (probs * labels).sum()
        denor = (probs.pow(self.p) + labels.pow(self.p)).sum()
        loss = 1.0 - (2 * numer + self.smooth) / (denor + self.smooth)
        return loss


class SoftDiceLossV2Func(torch.autograd.Function):
    """
    compute backward directly for better numeric stability
    """

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, labels, p, smooth):
        """
        inputs:
            logits: (N, L)
            labels: (N, L)
        outpus:
            loss: (N,)
        """
        probs = torch.sigmoid(logits)
        numer = 2 * (probs * labels).sum(dim=1) + smooth
        denor = (probs.pow(p) + labels.pow(p)).sum(dim=1) + smooth
        loss = 1.0 - numer / denor
        ctx.vars = probs, labels, numer, denor, p, smooth
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        """
        compute gradient of soft-dice loss
        """
        probs, labels, numer, denor, p, smooth = ctx.vars
        numer, denor = numer.view(-1, 1), denor.view(-1, 1)
        term1 = (1.0 - probs).mul_(2).mul_(labels).mul_(probs).div_(denor)
        term2 = probs.pow(p).mul_(1.0 - probs).mul_(numer).mul_(p).div_(denor.pow_(2))
        grads = term2.sub_(term1).mul_(grad_output)
        return grads, None, None, None


class SoftDiceLossV2(nn.Module):
    """
    soft-dice loss, useful in binary segmentation
    """

    def __init__(self, p=1, smooth=1):
        super(SoftDiceLossV2, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        """
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        """
        logits = logits.view(1, -1)
        labels = labels.view(1, -1)
        loss = SoftDiceLossV2Func.apply(logits, labels, self.p, self.smooth)
        return loss


class SoftDiceLossV3Func(torch.autograd.Function):
    """
    compute backward directly for better numeric stability
    """

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, labels, p, smooth):
        """
        inputs:
            logits: (N, L)
            labels: (N, L)
        outpus:
            loss: (N,)
        """
        assert logits.size() == labels.size() and logits.dim() == 2
        loss = soft_dice_cpp.soft_dice_forward(logits, labels, p, smooth)
        ctx.vars = logits, labels, p, smooth
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        """
        compute gradient of soft-dice loss
        """
        logits, labels, p, smooth = ctx.vars
        grads = soft_dice_cpp.soft_dice_backward(grad_output, logits, labels, p, smooth)
        return grads, None, None, None


class SoftDiceLossV3(nn.Module):
    """
    soft-dice loss, useful in binary segmentation
    """

    def __init__(self, p=1, smooth=1.0):
        super(SoftDiceLossV3, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        """
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        """
        logits = logits.view(1, -1)
        labels = labels.view(1, -1)
        loss = SoftDiceLossV3Func.apply(logits, labels, self.p, self.smooth)
        return loss


class SwishV1(nn.Module):

    def __init__(self):
        super(SwishV1, self).__init__()

    def forward(self, feat):
        return feat * torch.sigmoid(feat)


class SwishFunction(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, feat):
        sig = torch.sigmoid(feat)
        out = feat * torch.sigmoid(feat)
        grad = sig * (1 + feat * (1 - sig))
        ctx.grad = grad
        return out

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        grad = ctx.grad
        grad *= grad_output
        return grad


class SwishV2(nn.Module):

    def __init__(self):
        super(SwishV2, self).__init__()

    def forward(self, feat):
        return SwishFunction.apply(feat)


class SwishFunctionV3(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, feat):
        ctx.feat = feat
        return swish_cpp.swish_forward(feat)

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        feat = ctx.feat
        return swish_cpp.swish_backward(grad_output, feat)


class SwishV3(nn.Module):

    def __init__(self):
        super(SwishV3, self).__init__()

    def forward(self, feat):
        return SwishFunctionV3.apply(feat)


def taylor_softmax_v1(x, dim=1, n=4, use_log=False):
    assert n % 2 == 0 and n > 0
    fn = torch.ones_like(x)
    denor = 1.0
    for i in range(1, n + 1):
        denor *= i
        fn = fn + x.pow(i) / denor
    out = fn / fn.sum(dim=dim, keepdims=True)
    if use_log:
        out = out.log()
    return out


class TaylorSoftmaxV1(nn.Module):
    """
    This is the autograd version
    """

    def __init__(self, dim=1, n=2):
        super(TaylorSoftmaxV1, self).__init__()
        self.dim = dim
        self.n = n

    def forward(self, x):
        """
        usage similar to nn.Softmax:
            >>> mod = TaylorSoftmaxV1(dim=1, n=4)
            >>> inten = torch.randn(1, 32, 64, 64)
            >>> out = mod(inten)
        """
        return taylor_softmax_v1(x, self.dim, self.n, use_log=False)


class TaylorSoftmaxFunc(torch.autograd.Function):
    """
    use cpp/cuda to accelerate and shrink memory usage
    """

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, feat, dim=1, n=2, use_log=False):
        ctx.vars = feat, dim, n, use_log
        return taylor_softmax_cpp.taylor_softmax_forward(feat, dim, n, use_log)

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        """
        compute gradient of focal loss
        """
        feat, dim, n, use_log = ctx.vars
        return taylor_softmax_cpp.taylor_softmax_backward(grad_output, feat, dim, n, use_log), None, None, None


def taylor_softmax_v3(inten, dim=1, n=4, use_log=False):
    assert n % 2 == 0 and n > 0
    return TaylorSoftmaxFunc.apply(inten, dim, n, use_log)


class TaylorSoftmaxV3(nn.Module):
    """
    This is the autograd version
    """

    def __init__(self, dim=1, n=2):
        super(TaylorSoftmaxV3, self).__init__()
        assert n % 2 == 0 and n > 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        """
        usage similar to nn.Softmax:
            >>> mod = TaylorSoftmaxV3(dim=1, n=4)
            >>> inten = torch.randn(1, 32, 64, 64)
            >>> out = mod(inten)
        """
        return taylor_softmax_v3(x, self.dim, self.n, use_log=False)


class LogTaylorSoftmaxV1(nn.Module):
    """
    This is the autograd version
    """

    def __init__(self, dim=1, n=2):
        super(LogTaylorSoftmaxV1, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        """
        usage similar to nn.Softmax:
            >>> mod = LogTaylorSoftmaxV1(dim=1, n=4)
            >>> inten = torch.randn(1, 32, 64, 64)
            >>> out = mod(inten)
        """
        return taylor_softmax_v1(x, self.dim, self.n, use_log=True)


class LogTaylorSoftmaxV3(nn.Module):
    """
    This is the autograd version
    """

    def __init__(self, dim=1, n=2):
        super(LogTaylorSoftmaxV3, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        """
        usage similar to nn.Softmax:
            >>> mod = LogTaylorSoftmaxV3(dim=1, n=4)
            >>> inten = torch.randn(1, 32, 64, 64)
            >>> out = mod(inten)
        """
        return taylor_softmax_v3(x, self.dim, self.n, use_log=True)


class TaylorCrossEntropyLossV1(nn.Module):
    """
    This is the autograd version
    """

    def __init__(self, n=2, ignore_index=-1, reduction='mean'):
        super(TaylorCrossEntropyLossV1, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = LogTaylorSoftmaxV1(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        """
        usage similar to nn.CrossEntropyLoss:
            >>> crit = TaylorCrossEntropyLossV1(n=4)
            >>> inten = torch.randn(1, 10, 64, 64)
            >>> label = torch.randint(0, 10, (1, 64, 64))
            >>> out = crit(inten, label)
        """
        log_probs = self.taylor_softmax(logits)
        loss = F.nll_loss(log_probs, labels, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss


class TaylorCrossEntropyLossV3(nn.Module):
    """
    This is the autograd version
    """

    def __init__(self, n=2, ignore_index=-1, reduction='mean'):
        super(TaylorCrossEntropyLossV3, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = LogTaylorSoftmaxV3(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        """
        usage similar to nn.CrossEntropyLoss:
            >>> crit = TaylorCrossEntropyLossV3(n=4)
            >>> inten = torch.randn(1, 10, 64, 64)
            >>> label = torch.randint(0, 10, (1, 64, 64))
            >>> out = crit(inten, label)
        """
        log_probs = self.taylor_softmax(logits)
        loss = F.nll_loss(log_probs, labels, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss


class Model(nn.Module):

    def __init__(self, n_classes):
        super(Model, self).__init__()
        net = torchvision.models.resnet18(pretrained=False)
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.maxpool = net.maxpool
        self.relu = net.relu
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.fc = nn.Conv2d(512, n_classes, 3, 1, 1)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.bn1(feat)
        feat = self.relu(feat)
        feat = self.maxpool(feat)
        feat = self.layer1(feat)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        feat = self.layer4(feat)
        feat = self.fc(feat)
        out = torch.mean(feat, dim=(2, 3))
        return out


class TripletLoss(nn.Module):
    """
    Compute normal triplet loss or soft margin triplet loss given triplets
    """

    def __init__(self, margin=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda:
                y = y
            ap_dist = torch.norm(anchor - pos, 2, dim=1).view(-1)
            an_dist = torch.norm(anchor - neg, 2, dim=1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)
        return loss


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AMSoftmax,
     lambda: ([], {'in_feats': 4}),
     lambda: ([torch.rand([4, 4]), torch.ones([4], dtype=torch.int64)], {})),
    (APN,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (APNB,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ASPPConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ASPPPooling,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AdaptiveAvgMaxPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AffinityLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.ones([4, 4, 4], dtype=torch.int64)], {})),
    (AvgPoolPad,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BNInception,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (BN_Conv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BN_Conv2d_Leaky,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BN_Conv_Mish,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Backbone,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicConv2d,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicUnit,
     lambda: ([], {'in_chnls': 4, 'out_chnls': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BatchNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Block,
     lambda: ([], {'in_filters': 4, 'out_filters': 4, 'reps': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Block17,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1088, 64, 64])], {})),
    (Block35,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 320, 64, 64])], {})),
    (Block8,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2080, 64, 64])], {})),
    (BlockA,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 384, 64, 64])], {})),
    (BlockB,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1152, 64, 64])], {})),
    (BlockC,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2048, 64, 64])], {})),
    (BnActConv2d,
     lambda: ([], {'in_chs': 4, 'out_chs': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BottleNeck,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'strides': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BranchSeparablesStem,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CSP_DenseBlock,
     lambda: ([], {'in_channels': 4, 'num_layers': 1, 'k': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CatBnAct,
     lambda: ([], {'in_chs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ChannelPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ChannelShuffle,
     lambda: ([], {'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CoTAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64])], {})),
    (ContextBlock,
     lambda: ([], {'inplanes': 4, 'ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv2dBnRelu,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv2dCReLU,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv2dDynamicSamePadding,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CoordAtt,
     lambda: ([], {'inp': 4, 'oup': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CoordConv2d,
     lambda: ([], {'in_chan': 4, 'out_chan': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CutMixCrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (DPN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (DPN_Block,
     lambda: ([], {'in_chnls': 4, 'add_chnl': 4, 'cat_chnl': 4, 'cardinality': 4, 'd': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DSampling,
     lambda: ([], {'in_chnls': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DY_Conv2d,
     lambda: ([], {'in_chan': 4, 'out_chan': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DarkNet,
     lambda: ([], {'layers': [4, 4, 4, 4, 4], 'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (Dark_block,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DecoderLayer,
     lambda: ([], {'d_model': 4, 'd_inner': 4, 'n_head': 4, 'd_k': 4, 'd_v': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (DenseBlock,
     lambda: ([], {'input_channels': 4, 'num_layers': 1, 'growth_rate': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DenseResidualBlock,
     lambda: ([], {'filters': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Depth_Pointwise_Conv1d,
     lambda: ([], {'in_ch': 4, 'out_ch': 4, 'k': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (DoubleAttention,
     lambda: ([], {'in_channels': 4, 'c_m': 4, 'c_n': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DoubleConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DownConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DropBlock2D,
     lambda: ([], {'drop_prob': 4, 'block_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DropBlock3D,
     lambda: ([], {'drop_prob': 4, 'block_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (DynamicReLU_A,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DynamicReLU_B,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ECA,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ECAAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (EP,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (EncoderLayer,
     lambda: ([], {'d_model': 4, 'd_inner': 4, 'n_head': 4, 'd_k': 4, 'd_v': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (ExternalAttention,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FCA,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FCN8s,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (FCNHead,
     lambda: ([], {'in_channels': 4, 'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FPAModule1,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {})),
    (FPAModule2,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FPN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (FReLU,
     lambda: ([], {'in_chan': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FaceBoxes,
     lambda: ([], {'num_classes': 4, 'phase': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (FactorizedReduction,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FastRCNNPredictor,
     lambda: ([], {'in_channels': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (FeatureExtractor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (FeedForward,
     lambda: ([], {'dim': 4, 'mlp_dim': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Fire,
     lambda: ([], {'inplanes': 4, 'squeeze_planes': 4, 'expand1x1_planes': 4, 'expand3x3_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FireModule,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FocalLossV1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (FocalLossV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (GeneratorRRDB,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GhostModule,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GlobalContextBlock,
     lambda: ([], {'inplanes': 4, 'ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GoogleNet,
     lambda: ([], {'str_version': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 256, 256])], {})),
    (HSwishV1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HSwishV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HalfSplit,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HardSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Inception,
     lambda: ([], {'in_channels': 4, 'ch1x1': 4, 'ch3x3red': 4, 'ch3x3': 4, 'ch5x5red': 4, 'ch5x5': 4, 'pool_proj': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InceptionA,
     lambda: ([], {'in_channels': 4, 'pool_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InceptionB,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InceptionC,
     lambda: ([], {'in_channels': 4, 'channels_7x7': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InceptionD,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InceptionE,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InceptionModules,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 128, 64, 64])], {})),
    (InceptionResNetA2Way,
     lambda: ([], {'scale': 1.0}),
     lambda: ([torch.rand([4, 384, 64, 64])], {})),
    (InceptionResNetB2Way,
     lambda: ([], {'scale': 1.0}),
     lambda: ([torch.rand([4, 1152, 64, 64])], {})),
    (InceptionResNetBPoly,
     lambda: ([], {'scale': 1.0, 'num_blocks': 4}),
     lambda: ([torch.rand([4, 1152, 64, 64])], {})),
    (InceptionResNetBPoly3,
     lambda: ([], {'scale': 1.0}),
     lambda: ([torch.rand([4, 1152, 64, 64])], {})),
    (InceptionResNetC2Way,
     lambda: ([], {'scale': 1.0}),
     lambda: ([torch.rand([4, 2048, 64, 64])], {})),
    (InceptionResNetCPoly,
     lambda: ([], {'scale': 1.0, 'num_blocks': 4}),
     lambda: ([torch.rand([4, 2048, 64, 64])], {})),
    (InceptionResNetCPoly3,
     lambda: ([], {'scale': 1.0}),
     lambda: ([torch.rand([4, 2048, 64, 64])], {})),
    (InceptionResNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 512, 512])], {})),
    (InceptionV4,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {})),
    (Inception_A,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 384, 64, 64])], {})),
    (Inception_A_res,
     lambda: ([], {'in_channels': 4, 'b1': 4, 'b2_n1': 4, 'b2_n3': 4, 'b3_n1': 4, 'b3_n3_1': 4, 'b3_n3_2': 4, 'n1_linear': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Inception_B,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1024, 64, 64])], {})),
    (Inception_B_res,
     lambda: ([], {'in_channels': 4, 'b1': 4, 'b2_n1': 4, 'b2_n1x7': 4, 'b2_n7x1': 4, 'n1_linear': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Inception_C,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1536, 64, 64])], {})),
    (Inception_C_res,
     lambda: ([], {'in_channels': 4, 'b1': 4, 'b2_n1': 4, 'b2_n1x3': 4, 'b2_n3x1': 4, 'n1_linear': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InputBlock,
     lambda: ([], {'num_init_features': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (LBwithGCBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LFFD,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (LFFDBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LPN,
     lambda: ([], {'nJoints': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (LWbottleneck,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LabelSmoothSoftmaxCEV1,
     lambda: ([], {}),
     lambda: ([torch.ones([4, 4], dtype=torch.int64), torch.ones([4], dtype=torch.int64)], {})),
    (LambdaBase,
     lambda: ([], {'fn': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LargeMarginSoftmaxV2,
     lambda: ([], {}),
     lambda: ([torch.ones([4, 4], dtype=torch.int64), torch.ones([4], dtype=torch.int64)], {})),
    (LayerNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LinearScheduler,
     lambda: ([], {'dropblock': torch.nn.ReLU(), 'start_value': 4, 'stop_value': 4, 'nr_steps': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LogTaylorSoftmaxV1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LossBranch,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LovaszSoftmaxV1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)], {})),
    (MDConv,
     lambda: ([], {'nchannels': 4, 'kernel_sizes': [4, 4], 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (MUSEAttention,
     lambda: ([], {'d_model': 4, 'd_k': 4, 'd_v': 4, 'h': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (MaxPool,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MaxPool2dDynamicSamePadding,
     lambda: ([], {'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MaxPoolPad,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MemoryEfficientSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MiddleBottleneck,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Mish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MishV1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MishV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Mixed_3a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {})),
    (Mixed_4a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 160, 64, 64])], {})),
    (Mixed_5a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 192, 64, 64])], {})),
    (Mixed_5b,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 192, 64, 64])], {})),
    (Mixed_6a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 320, 64, 64])], {})),
    (Mixed_7a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1088, 64, 64])], {})),
    (Model,
     lambda: ([], {'n_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (MultiHeadAttention,
     lambda: ([], {'n_head': 4, 'd_model': 4, 'd_k': 4, 'd_v': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (MultiWay,
     lambda: ([], {'scale': 1.0, 'block_cls': torch.nn.ReLU, 'num_blocks': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NONLocalBlock1D,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (NONLocalBlock2D,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NonLocalBlock,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (OSA_module,
     lambda: ([], {'in_channels': 4, 'mid_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (OSAv2_module,
     lambda: ([], {'in_channels': 4, 'mid_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (OutlookAttention,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PCSoftmaxCrossEntropyV2,
     lambda: ([], {'lb_proportion': 4}),
     lambda: ([torch.ones([4, 4], dtype=torch.int64), torch.ones([4], dtype=torch.int64)], {})),
    (PEP,
     lambda: ([], {'in_channels': 4, 'proj_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PSA,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 4, 4])], {})),
    (ParNetAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64])], {})),
    (ParallelPolarizedSelfAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64])], {})),
    (PolyConv2d,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4, 'num_blocks': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), 0], {})),
    (PositionalEncoding,
     lambda: ([], {'embed_size': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PositionwiseFeedForward,
     lambda: ([], {'d_in': 4, 'd_hid': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PreNorm,
     lambda: ([], {'dim': 4, 'fn': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PyramidPooling,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PyramidPoolingModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RPNHead,
     lambda: ([], {'in_channels': 4, 'num_anchors': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ReductionA,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 384, 64, 64])], {})),
    (ReductionB,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1152, 64, 64])], {})),
    (Reduction_A,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 384, 64, 64])], {})),
    (Reduction_B,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1024, 64, 64])], {})),
    (Reduction_B_Res,
     lambda: ([], {'in_channels': 4, 'b2_n1': 4, 'b2_n3': 4, 'b3_n1': 4, 'b3_n3': 4, 'b4_n1': 4, 'b4_n3_1': 4, 'b4_n3_2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Reduction_B_v4,
     lambda: ([], {'in_channels': 4, 'b2_n1': 4, 'b2_n3': 4, 'b3_n1': 4, 'b3_n1x7': 4, 'b3_n7x1': 4, 'b3_n3': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ReluConvBn,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResNeXt,
     lambda: ([], {'layers': [4, 4, 4, 4], 'cardinality': 4, 'group_depth': 1, 'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (ResNeXt_Block,
     lambda: ([], {'in_chnls': 4, 'cardinality': 4, 'group_depth': 1, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Reshape,
     lambda: ([], {}),
     lambda: ([torch.rand([4])], {})),
    (ResidualAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64])], {})),
    (ResidualInResidualDenseBlock,
     lambda: ([], {'filters': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SAG_Mask,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SE,
     lambda: ([], {'inplanes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SEAttention,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SEModule,
     lambda: ([], {'channels': 4, 'reduction': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SF,
     lambda: ([], {'channels': 32}),
     lambda: ([torch.rand([4, 32, 64, 64])], {})),
    (SKAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64])], {})),
    (SKConv,
     lambda: ([], {'features': 32}),
     lambda: ([torch.rand([4, 32, 64, 64])], {})),
    (SSD,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 288, 288])], {})),
    (ScaledDotProductAttention,
     lambda: ([], {'temperature': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SeparableConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SeqConv,
     lambda: ([], {'in_chnls': 4, 'out_chnls': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SequentialPolarizedSelfAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64])], {})),
    (ShakeDrop,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SimpleBaseline,
     lambda: ([], {'nJoints': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (SimplifiedScaledDotProductAttention,
     lambda: ([], {'d_model': 4, 'h': 4}),
     lambda: ([torch.rand([4, 4, 4, 1]), torch.rand([4, 4, 4, 1]), torch.rand([4, 4, 4, 1])], {})),
    (SoftDiceLossV1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SoftDiceLossV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SpatialAttention,
     lambda: ([], {'in_planes': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {})),
    (SpatialAttentionModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SpatialCrossMapLRN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SpatialGate,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SpatialGroupEnhance,
     lambda: ([], {'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SpatialPyramidPooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SqueezeAndExcite,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'se_kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SqueezeNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 256, 256])], {})),
    (Stem,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (Stem_Res1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (Stem_v4_Res2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (StyleEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {})),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SwishV1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SwishV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TaylorSoftmaxV1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TripleConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TripletAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TwoMLPHead,
     lambda: ([], {'in_channels': 4, 'representation_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (UFOAttention,
     lambda: ([], {'d_model': 4, 'd_k': 4, 'd_v': 4, 'h': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (UNet,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (UNetDown,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (YOLO_Nano,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (_ASPPModule,
     lambda: ([], {'inplanes': 4, 'planes': 4, 'kernel_size': 4, 'padding': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (_ChannelAttentionModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (_DenseBlock,
     lambda: ([], {'num_layers': 1, 'num_input_features': 4, 'bn_size': 4, 'growth_rate': 4, 'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (_DenseLayer,
     lambda: ([], {'inplace': 4, 'growth_rate': 4, 'bn_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (_Transition,
     lambda: ([], {'num_input_features': 4, 'num_output_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (_TransitionLayer,
     lambda: ([], {'inplace': 4, 'plance': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (cSE_Module,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (eSE_Module,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (h_sigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (h_swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (sSE_Module,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (scSE_Module,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

