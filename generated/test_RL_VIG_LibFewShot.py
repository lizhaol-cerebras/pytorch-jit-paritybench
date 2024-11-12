
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


import itertools


from collections import Iterable


import torch


import numpy as np


import random


from torch.utils import data


from torch.utils.data import DataLoader


from torch.utils.data.distributed import DistributedSampler


from torchvision import transforms


from queue import Queue


from torch.utils.data import Dataset


from torch.utils.data import Sampler


from abc import abstractmethod


from torch import nn


import torch.nn as nn


import torch.nn.functional as F


from torch.distributions import Bernoulli


import math


from torch.nn.parameter import Parameter


from torch.nn.modules.module import Module


from torch.nn.modules.utils import _pair


from torch.autograd import Variable


from torch.nn.utils.weight_norm import WeightNorm


from torch import einsum


from functools import partial


import warnings


from torch.nn.utils import weight_norm


import copy


from sklearn import metrics


from sklearn.linear_model import LogisticRegression


from itertools import combinations


from torch.nn import Parameter


from torch.optim.lr_scheduler import _LRScheduler


from torch.nn import functional as F


from re import T


from torch.nn import init


from torch import digamma


from logging import getLogger


from time import time


import torch.distributed as dist


import logging


from collections import OrderedDict


import pandas as pd


import scipy as sp


import scipy.stats


import torch.multiprocessing


from torch.utils import tensorboard


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError


class Conv64F(nn.Module):
    """
    Four convolutional blocks network, each of which consists of a Covolutional layer,
    a Batch Normalizaiton layer, a ReLU layer and a Maxpooling layer.
    Used in the original ProtoNet: https://github.com/jakesnell/prototypical-networks.git.

    Input:  3 * 84 *84
    Output: 64 * 5 * 5
    """

    def __init__(self, is_flatten=False, is_feature=False, leaky_relu=False, negative_slope=0.2, last_pool=True, maxpool_last2=True, use_running_statistics=True):
        super(Conv64F, self).__init__()
        self.is_flatten = is_flatten
        self.is_feature = is_feature
        self.last_pool = last_pool
        self.maxpool_last2 = maxpool_last2
        if leaky_relu:
            activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        else:
            activation = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, track_running_stats=use_running_statistics), activation, nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, track_running_stats=use_running_statistics), activation, nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, track_running_stats=use_running_statistics), activation)
        self.layer3_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, track_running_stats=use_running_statistics), activation)
        self.layer4_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        if self.maxpool_last2:
            out3 = self.layer3_maxpool(out3)
        out4 = self.layer4(out3)
        if self.last_pool:
            out4 = self.layer4_pool(out4)
        if self.is_flatten:
            out4 = out4.view(out4.size(0), -1)
        if self.is_feature:
            return out1, out2, out3, out4
        return out4


class Conv32F(nn.Module):
    """
    Four convolutional blocks network, each of which consists of a Covolutional layer,
    a Batch Normalizaiton layer, a ReLU layer and a Maxpooling layer.
    Used in the original ProtoNet: https://github.com/jakesnell/prototypical-networks.git.

    Input:  3 * 84 *84
    Output: 32 * 5 * 5
    """

    def __init__(self, is_flatten=False, is_feature=False, leaky_relu=False, negative_slope=0.2, last_pool=True):
        super(Conv32F, self).__init__()
        self.is_flatten = is_flatten
        self.is_feature = is_feature
        self.last_pool = last_pool
        if leaky_relu:
            activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        else:
            activation = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), activation, nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), activation, nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), activation, nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), activation)
        self.layer4_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        if self.last_pool:
            out4 = self.layer4_pool(out4)
        if self.is_flatten:
            out4 = out4.view(out4.size(0), -1)
        if self.is_feature:
            return out1, out2, out3, out4
        return out4


def R2D2_conv_block(in_channels, out_channels, retain_activation=True, keep_prob=1.0, pool_stride=2):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.MaxPool2d(2, stride=pool_stride))
    if retain_activation:
        block.add_module('LeakyReLU', nn.LeakyReLU(0.1))
    if keep_prob < 1.0:
        block.add_module('Dropout', nn.Dropout(p=1 - keep_prob, inplace=False))
    return block


class R2D2Embedding(nn.Module):
    """
    https://github.com/kjunelee/MetaOptNet/blob/master/models/R2D2_embedding.py
    """

    def __init__(self, x_dim=3, h1_dim=96, h2_dim=192, h3_dim=384, z_dim=512, retain_last_activation=False):
        super(R2D2Embedding, self).__init__()
        self.block1 = R2D2_conv_block(x_dim, h1_dim)
        self.block2 = R2D2_conv_block(h1_dim, h2_dim)
        self.block3 = R2D2_conv_block(h2_dim, h3_dim, keep_prob=0.9)
        self.block4 = R2D2_conv_block(h3_dim, z_dim, retain_activation=retain_last_activation, keep_prob=0.9, pool_stride=1)

    def forward(self, x):
        b1 = self.block1(x)
        b2 = self.block2(b1)
        b3 = self.block3(b2)
        b4 = self.block4(b3)
        return torch.cat((b3.view(b3.size(0), -1), b4.view(b4.size(0), -1)), 1)


class Conv64F_MCL(nn.Module):
    """docstring for ClassName"""

    def __init__(self):
        super(Conv64F_MCL, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True), nn.MaxPool2d(kernel_size=2, stride=2))
        self.out_channels = 64
        for l in self.modules():
            if isinstance(l, nn.Conv2d):
                torch.nn.init.xavier_uniform_(l.weight)
                if l.bias is not None:
                    torch.nn.init.constant_(l.bias, 0)
            elif isinstance(l, nn.Linear):
                torch.nn.init.normal_(l.weight, 0, 0.01)
                if l.bias is not None:
                    torch.nn.init.constant(l.bias, 0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = not self.equalInOut and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class DropBlock(nn.Module):

    def __init__(self, block_size):
        super(DropBlock, self).__init__()
        self.block_size = block_size

    def forward(self, x, gamma):
        if self.training:
            batch_size, channels, height, width = x.shape
            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample((batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1)))
            if torch.cuda.is_available():
                mask = mask
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()
            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)
        batch_size, channels, height, width = mask.shape
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]
        offsets = torch.stack([torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1), torch.arange(self.block_size).repeat(self.block_size)]).t()
        offsets = torch.cat((torch.zeros(self.block_size ** 2, 2).long(), offsets.long()), 1)
        if torch.cuda.is_available():
            offsets = offsets
        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()
            block_idxs = non_zero_idxs + offsets
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.0
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
        block_mask = 1 - padded_mask
        return block_mask


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlockWithoutResidual(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1, use_pool=True):
        super(BasicBlockWithoutResidual, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.use_pool = use_pool

    def forward(self, x):
        self.num_batches_tracked += 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.use_pool:
            out = self.maxpool(out)
        if self.drop_rate > 0:
            if self.drop_block:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * self.num_batches_tracked, 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        return out


def BDCovpool(x, t):
    batchSize, dim, h, w = x.data.shape
    M = h * w
    x = x.reshape(batchSize, dim, M)
    I = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(x.dtype)
    I_M = torch.ones(batchSize, dim, dim, device=x.device).type(x.dtype)
    x_pow2 = x.bmm(x.transpose(1, 2))
    dcov = I_M.bmm(x_pow2 * I) + (x_pow2 * I).bmm(I_M) - 2 * x_pow2
    dcov = torch.clamp(dcov, min=0.0)
    dcov = torch.exp(t) * dcov
    dcov = torch.sqrt(dcov + 1e-05)
    t = dcov - 1.0 / dim * dcov.bmm(I_M) - 1.0 / dim * I_M.bmm(dcov) + 1.0 / (dim * dim) * I_M.bmm(dcov).bmm(I_M)
    return t


def Triuvec(x):
    batchSize, dim, dim = x.shape
    r = x.reshape(batchSize, dim * dim)
    I = torch.ones(dim, dim).triu().reshape(dim * dim)
    index = I.nonzero(as_tuple=False)
    y = torch.zeros(batchSize, int(dim * (dim + 1) / 2), device=x.device).type(x.dtype)
    y = r[:, index].squeeze()
    return y


class BdcPool(nn.Module):
    """ https://github.com/Fei-Long121/DeepBDC/blob/main/methods/bdc_module.py """

    def __init__(self, is_vec=True, input_dim=640, dimension_reduction=None, activate='relu'):
        super(BdcPool, self).__init__()
        self.is_vec = is_vec
        self.dr = dimension_reduction
        self.activate = activate
        self.input_dim = input_dim[0]
        if self.dr is not None and self.dr != self.input_dim:
            if activate == 'relu':
                self.act = nn.ReLU(inplace=True)
            elif activate == 'leaky_relu':
                self.act = nn.LeakyReLU(0.1)
            else:
                self.act = nn.ReLU(inplace=True)
            self.conv_dr_block = nn.Sequential(nn.Conv2d(self.input_dim, self.dr, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(self.dr), self.act)
        output_dim = self.dr if self.dr else self.input_dim
        if self.is_vec:
            self.output_dim = int(output_dim * (output_dim + 1) / 2)
        else:
            self.output_dim = int(output_dim * output_dim)
        self.temperature = nn.Parameter(torch.log(1.0 / (2 * input_dim[1] * input_dim[2]) * torch.ones(1, 1)), requires_grad=True)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.dr is not None and self.dr != self.input_dim:
            x = self.conv_dr_block(x)
        x = BDCovpool(x, self.temperature)
        if self.is_vec:
            x = Triuvec(x)
        else:
            x = x.reshape(x.shape[0], -1)
        return x


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def init_layer(L):
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class ResNet(nn.Module):
    maml = False

    def __init__(self, block, list_of_num_layers, list_of_out_dims, flatten=False, reduce_dim=640):
        super(ResNet, self).__init__()
        assert len(list_of_num_layers) == 4, 'Can have only four stages'
        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        bn1 = nn.BatchNorm2d(64)
        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        init_layer(conv1)
        init_layer(bn1)
        trunk = [conv1, bn1, relu, pool1]
        indim = 64
        for i in range(4):
            for j in range(list_of_num_layers[i]):
                half_res = i >= 1 and j == 0 and i != 3
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]
        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
        self.feat_dim = [512, 14, 14]
        self.trunk = nn.Sequential(*trunk)
        self.bdc_pool = BdcPool(is_vec=True, input_dim=self.feat_dim, dimension_reduction=reduce_dim)

    def forward(self, x):
        out = self.trunk(x)
        out = self.bdc_pool(out)
        return out


class ResNet_r2d2(ResNet):

    def __init__(self, block=BasicBlock, drop_rate=0.0, dropblock_size=3):
        super(ResNet_r2d2, self).__init__(block, drop_rate, dropblock_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        n, c, h, w = x.shape
        x = F.adaptive_avg_pool2d(x, 1).view(n, c)
        return x


class _ConvNdMtl(nn.Module):
    """The class for meta-transfer convolution"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, bias, MTL):
        super(_ConvNdMtl, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.MTL = MTL
        if transposed:
            self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels // groups, *kernel_size))
            self.mtl_weight = nn.Parameter(torch.ones(in_channels, out_channels // groups, 1, 1))
        else:
            self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
            self.mtl_weight = nn.Parameter(torch.ones(out_channels, in_channels // groups, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            self.mtl_bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('mtl_bias', None)
        if MTL:
            self.weight.requires_grad = False
            if bias:
                self.bias.requires_grad = False
        else:
            self.mtl_weight.requires_grad = False
            if bias:
                self.mtl_bias.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.mtl_weight.data.uniform_(1, 1)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.mtl_bias.data.uniform_(0, 0)

    def extra_repr(self):
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.MTL is not None:
            s += ', MTL={MTL}'
        return s.format(**self.__dict__)


class Conv2dMtl(_ConvNdMtl):
    """The class for meta-transfer convolution"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, MTL=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.MTL = MTL
        super(Conv2dMtl, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, MTL)

    def forward(self, inp):
        if self.MTL:
            new_mtl_weight = self.mtl_weight.expand(self.weight.shape)
            new_weight = self.weight.mul(new_mtl_weight)
            if self.bias is not None:
                new_bias = self.bias + self.mtl_bias
            else:
                new_bias = None
        else:
            new_weight = self.weight
            new_bias = self.bias
        return F.conv2d(inp, new_weight, new_bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3MTL(in_planes, out_planes, stride=1, MTL=False):
    """3x3 convolution with padding"""
    return Conv2dMtl(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, MTL=MTL)


class BasicBlockMTL(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, MTL=False):
        super(BasicBlockMTL, self).__init__()
        self.conv1 = conv3x3MTL(inplanes, planes, stride, MTL=MTL)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3MTL(planes, planes, MTL=MTL)
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


class ResNetMTLOfficial(nn.Module):

    def __init__(self, MTL=False):
        super(ResNetMTLOfficial, self).__init__()
        self.Conv2d = Conv2dMtl
        block = BasicBlockMTL
        self.inplanes = iChannels = 80
        self.conv1 = self.Conv2d(3, iChannels, kernel_size=3, stride=1, padding=1, MTL=MTL)
        self.bn1 = nn.BatchNorm2d(iChannels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 160, 4, stride=2, MTL=MTL)
        self.layer2 = self._make_layer(block, 320, 4, stride=2, MTL=MTL)
        self.layer3 = self._make_layer(block, 640, 4, stride=2, MTL=MTL)
        self.avgpool = nn.AvgPool2d(10, stride=1)
        for m in self.modules():
            if isinstance(m, self.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, MTL=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(self.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, MTL=MTL), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, MTL=MTL))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, MTL=MTL))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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
        out = self.relu(out)
        return out


class SimpleBlock(nn.Module):
    maml = False

    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(outdim)
        self.C2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]
        self.half_res = half_res
        if indim != outdim:
            self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
            self.BNshortcut = nn.BatchNorm2d(outdim)
            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'
        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out


class BottleneckBlock(nn.Module):
    maml = False

    def __init__(self, indim, outdim, half_res):
        super(BottleneckBlock, self).__init__()
        bottleneckdim = int(outdim / 4)
        self.indim = indim
        self.outdim = outdim
        self.C1 = nn.Conv2d(indim, bottleneckdim, kernel_size=1, bias=False)
        self.BN1 = nn.BatchNorm2d(bottleneckdim)
        self.C2 = nn.Conv2d(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1, padding=1)
        self.BN2 = nn.BatchNorm2d(bottleneckdim)
        self.C3 = nn.Conv2d(bottleneckdim, outdim, kernel_size=1, bias=False)
        self.BN3 = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU()
        self.parametrized_layers = [self.C1, self.BN1, self.C2, self.BN2, self.C3, self.BN3]
        self.half_res = half_res
        if indim != outdim:
            self.shortcut = nn.Conv2d(indim, outdim, 1, stride=2 if half_res else 1, bias=False)
            self.parametrized_layers.append(self.shortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'
        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        short_out = x if self.shortcut_type == 'identity' else self.shortcut(x)
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.C3(out)
        out = self.BN3(out)
        out = out + short_out
        out = self.relu(out)
        return out


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BasicBlockVariant(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1, use_se=False):
        super(BasicBlockVariant, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.use_se = use_se
        if self.use_se:
            self.se = SELayer(planes, 4)

    def forward(self, x):
        self.num_batches_tracked += 1
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.use_se:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * self.num_batches_tracked, 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        return out


class resnet(nn.Module):

    def __init__(self, block, n_blocks, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5, num_classes=-1, use_se=False, reduce_dim=640):
        super(resnet, self).__init__()
        self.inplanes = 3
        self.use_se = use_se
        self.layer1 = self._make_layer(block, n_blocks[0], 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, n_blocks[1], 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, n_blocks[2], 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, n_blocks[3], 640, stride=1, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        self.feat_dim = [640, 10, 10]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.bdc_pool = BdcPool(is_vec=True, input_dim=self.feat_dim, dimension_reduction=reduce_dim)
        self.num_classes = num_classes
        if self.num_classes > 0:
            self.classifier = nn.Linear(640, self.num_classes)

    def _make_layer(self, block, n_block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        if n_block == 1:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size, self.use_se)
        else:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, self.use_se)
        layers.append(layer)
        self.inplanes = planes * block.expansion
        for i in range(1, n_block):
            if i == n_block - 1:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, drop_block=drop_block, block_size=block_size, use_se=self.use_se)
            else:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, use_se=self.use_se)
            layers.append(layer)
        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bdc_pool(x)
        return x


class CyclicShift(nn.Module):

    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)
    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')
    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')
    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):

    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted
        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement, upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement, upper_lower=False, left_right=True), requires_grad=False)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)
        _, n_h, n_w, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size
        q, k, v = map(lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d', h=h, w_h=self.window_size, w_w=self.window_size), qkv)
        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale
        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding
        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask
        attn = dots.softmax(dim=-1)
        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)', h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)
        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):

    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim, heads=heads, head_dim=head_dim, shifted=shifted, window_size=window_size, relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class PatchMerging(nn.Module):

    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class StageModule(nn.Module):

    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size, relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'
        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension, downscaling_factor=downscaling_factor)
        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4, shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding), SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4, shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding)]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)


class SwinTransformer(nn.Module):

    def __init__(self, *, hidden_dim, layers, heads, channels=3, output_dim=512, head_dim=32, window_size=7, downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True, pool=True):
        super().__init__()
        self.pool = pool
        self.stage1 = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=layers[0], downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim, window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1], downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim, window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2], downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim, window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage4 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3], downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim, window_size=window_size, relative_pos_embedding=relative_pos_embedding)

    def forward(self, img):
        x = self.stage1(img)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        if self.pool:
            x = x.mean(dim=[2, 3])
        else:
            x = x.reshape(x.size(0), -1)
        return x


class Linear_fw(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast)
        else:
            out = super(Linear_fw, self).forward(x)
        return out


class Conv2d_fw(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if self.bias is not None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        elif self.weight.fast is not None and self.bias.fast is not None:
            out = F.conv2d(x, self.weight.fast, self.bias.fast, stride=self.stride, padding=self.padding)
        else:
            out = super(Conv2d_fw, self).forward(x)
        return out


class BatchNorm2d_fw(nn.BatchNorm2d):

    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1])
        running_var = torch.ones(x.data.size()[1])
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training=True, momentum=1)
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training=True, momentum=1)
        return out


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)), PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class ViT(nn.Module):

    def __init__(self, *, image_size=84, patch_size=28, dim=1024, depth=6, heads=16, mlp_dim=2048, pool='mean', channels=3, dim_head=64, dropout=0.0, emb_dropout=0.0):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = image_height // patch_height * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width), nn.Linear(patch_dim, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n + 1]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        return x


def drop_path(x, drop_prob: 'float'=0.0, training: 'bool'=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        if self.gamma_1 is None:
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = img_size // patch_size * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        return self.proj(x)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
    with torch.no_grad():
        L = norm_cdf((a - mean) / std)
        U = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * L - 1, 2 * U - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class VisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.5, norm_layer=partial(nn.LayerNorm, eps=1e-06), return_all_tokens=True, init_values=0, use_mean_pooling=False, masked_im_modeling=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.return_all_tokens = return_all_tokens
        self.patch_embed = PatchEmbed(img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values) for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        self.masked_im_modeling = masked_im_modeling
        if masked_im_modeling:
            self.masked_embed = nn.Parameter(torch.zeros(1, embed_dim))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2), scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)), mode='bicubic')
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x, mask=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if mask is not None:
            x = self.mask_model(x, mask)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

    def forward(self, x, return_all_tokens=None, mask=None):
        if self.masked_im_modeling:
            assert mask is not None
            x = self.prepare_tokens(x, mask=mask)
        else:
            x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.fc_norm is not None:
            x[:, 0] = self.fc_norm(x[:, 1:, :].mean(1))
        return_all_tokens = self.return_all_tokens if return_all_tokens is None else return_all_tokens
        if return_all_tokens:
            return x
        return x[:, 0]

    def get_num_layers(self):
        return len(self.blocks)

    def mask_model(self, x, mask):
        x.permute(0, 2, 3, 1)[mask, :] = self.masked_embed
        return x


class NetworkBlock(nn.Module):

    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):

    def __init__(self, depth, widen_factor=1, dropRate=0.0, is_flatten=True, avg_pool=True):
        super(WideResNet, self).__init__()
        self.is_flatten = is_flatten
        self.avg_pool = avg_pool
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.nChannels = nChannels[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        if self.avg_pool:
            out = F.adaptive_max_pool2d(out, 1)
        if self.is_flatten:
            out = out.reshape(out.size(0), -1)
        return out


class DistLinear(nn.Module):
    """
    Coming from "A Closer Look at Few-shot Classification. ICLR 2019."
    https://github.com/wyharveychen/CloserLookFewShot.git
    """

    def __init__(self, in_channel, out_channel):
        super(DistLinear, self).__init__()
        self.fc = nn.Linear(in_channel, out_channel, bias=False)
        self.class_wise_learnable_norm = True
        if self.class_wise_learnable_norm:
            weight_norm(self.fc, 'weight', dim=0)
        if out_channel <= 200:
            self.scale_factor = 2
        else:
            self.scale_factor = 10

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-05)
        if not self.class_wise_learnable_norm:
            fc_norm = torch.norm(self.fc.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.fc.weight.data)
            self.fc.weight.data = self.fc.weight.data.div(fc_norm + 1e-05)
        cos_dist = self.fc(x_normalized)
        score = self.scale_factor * cos_dist
        return score


class DistillLayer(nn.Module):

    def __init__(self, emb_func, cls_classifier, is_distill, emb_func_path=None, cls_classifier_path=None):
        super(DistillLayer, self).__init__()
        self.emb_func = self._load_state_dict(emb_func, emb_func_path, is_distill)
        self.cls_classifier = self._load_state_dict(cls_classifier, cls_classifier_path, is_distill)

    def _load_state_dict(self, model, state_dict_path, is_distill):
        new_model = None
        if is_distill and state_dict_path is not None:
            model_state_dict = torch.load(state_dict_path, map_location='cpu')
            model.load_state_dict(model_state_dict)
            new_model = copy.deepcopy(model)
        return new_model

    @torch.no_grad()
    def forward(self, x):
        output = None
        if self.emb_func is not None and self.cls_classifier is not None:
            output = self.emb_func(x)
            output = self.cls_classifier(output)
        return output


class FRNLayer(nn.Module):

    def __init__(self, num_cat=None, num_channel=640):
        super().__init__()
        self.resolution = 25
        self.d = num_channel
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.r = nn.Parameter(torch.zeros(2), requires_grad=True)

    def forward(self, support, query, way_num, shot_num, query_num):
        n2g_l2_dist = self.get_neg_l2_dist(support, query, way_num, shot_num, query_num)
        logits = n2g_l2_dist * self.scale
        log_prediction = F.log_softmax(logits, dim=2)
        return log_prediction

    def get_recon_dist(self, query, support, alpha, beta, Woodbury=True):
        reg = support.size(2) / support.size(3)
        lam = reg * alpha.exp() + 1e-06
        rho = beta.exp()
        st = support.permute(0, 1, 3, 2)
        if Woodbury:
            sts = st.matmul(support)
            m_inv = (sts + torch.eye(sts.size(-1)).unsqueeze(0).unsqueeze(0).mul(lam)).inverse()
            hat = m_inv.matmul(sts)
        else:
            sst = support.matmul(st)
            m_inv = (sst + torch.eye(sst.size(-1)).unsqueeze(0).unsqueeze(0).mul(lam)).inverse()
            hat = st.matmul(m_inv).matmul(support)
        Q_bar = query.unsqueeze(1).matmul(hat).mul(rho)
        dist = (Q_bar - query.unsqueeze(1)).pow(2).sum(3).permute(0, 2, 1)
        return dist

    def get_neg_l2_dist(self, support, query, way, shot, query_shot):
        resolution = self.resolution
        d = self.d
        alpha = self.r[0]
        beta = self.r[1]
        recon_dist = self.get_recon_dist(query=query, support=support, alpha=alpha, beta=beta)
        neg_l2_dist = recon_dist.neg().view(-1, way * query_shot, resolution, way).mean(2)
        return neg_l2_dist


def topk_(matrix, K, axis):
    """
    the function to calc topk acc of ndarrary.

    TODO

    """
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[topk_index_sort, row_index]
        topk_index_sort = topk_index[0:K, :][topk_index_sort, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:, 0:K][column_index, topk_index_sort]
    return topk_data_sort, topk_index_sort


def accuracy(output, target, topk=1):
    """
    Calc the acc of tpok.

    output and target have the same dtype and the same shape.

    Args:
        output (torch.Tensor or np.ndarray): The output.
        target (torch.Tensor or np.ndarray): The target.
        topk (int or list or tuple): topk . Defaults to 1.

    Returns:
        float: acc.
    """
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = {'Tensor': torch.topk, 'ndarray': lambda output, maxk, axis: (None, torch.from_numpy(topk_(output, maxk, axis)[1]))}[output.__class__.__name__](output, topk, 1)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:topk].view(-1).float().sum(0, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(correct_k, op=dist.ReduceOp.SUM)
            batch_size *= dist.get_world_size()
        res = correct_k.mul_(100.0 / batch_size).item()
        return res


class ProtoLayer(nn.Module):

    def __init__(self):
        super(ProtoLayer, self).__init__()

    def forward(self, query_feat, support_feat, way_num, shot_num, query_num, mode='euclidean'):
        t, wq, c = query_feat.size()
        _, ws, _ = support_feat.size()
        query_feat = query_feat.reshape(t, way_num * query_num, c)
        support_feat = support_feat.reshape(t, way_num, shot_num, c)
        proto_feat = torch.mean(support_feat, dim=2)
        return {'euclidean': lambda x, y: -torch.sum(torch.pow(x.unsqueeze(2) - y.unsqueeze(1), 2), dim=3), 'cos_sim': lambda x, y: torch.matmul(F.normalize(x, p=2, dim=-1), torch.transpose(F.normalize(y, p=2, dim=-1), -1, -2))}[mode](query_feat, proto_feat)


class MTLBaseLearner(nn.Module):
    """The class for inner loop."""

    def __init__(self, ways, z_dim):
        super().__init__()
        self.ways = ways
        self.z_dim = z_dim
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([self.ways, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(self.ways))
        self.vars.append(self.fc1_b)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        fc1_b = the_vars[1]
        net = F.linear(input_x, fc1_w, fc1_b)
        return net

    def parameters(self):
        return self.vars


class NegLayer(nn.Module):

    def __init__(self, in_features, out_features, margin=0.4, scale_factor=30.0):
        super(NegLayer, self).__init__()
        self.margin = margin
        self.scale_factor = scale_factor
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feature, label=None):
        cosine = F.linear(F.normalize(feature), F.normalize(self.weight))
        if label is None:
            return cosine * self.scale_factor
        phi = cosine - self.margin
        output = torch.where(self.one_hot(label, cosine.shape[1]).byte(), phi, cosine)
        output *= self.scale_factor
        return output

    def one_hot(self, y, num_class):
        return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)


class SepConv4d(nn.Module):
    """approximates 3 x 3 x 3 x 3 kernels via two subsequent 3 x 3 x 1 x 1 and 1 x 1 x 3 x 3"""

    def __init__(self, in_planes, out_planes, stride=(1, 1, 1), ksize=3, do_padding=True, bias=False):
        super(SepConv4d, self).__init__()
        self.isproj = False
        padding1 = (0, ksize // 2, ksize // 2) if do_padding else (0, 0, 0)
        padding2 = (ksize // 2, ksize // 2, 0) if do_padding else (0, 0, 0)
        if in_planes != out_planes:
            self.isproj = True
            self.proj = nn.Sequential(nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, bias=bias, padding=0), nn.BatchNorm2d(out_planes))
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels=in_planes, out_channels=in_planes, kernel_size=(1, ksize, ksize), stride=stride, bias=bias, padding=padding1), nn.BatchNorm3d(in_planes))
        self.conv2 = nn.Sequential(nn.Conv3d(in_channels=in_planes, out_channels=in_planes, kernel_size=(ksize, ksize, 1), stride=stride, bias=bias, padding=padding2), nn.BatchNorm3d(in_planes))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, u, v, h, w = x.shape
        x = self.conv2(x.view(b, c, u, v, -1))
        b, c, u, v, _ = x.shape
        x = self.relu(x)
        x = self.conv1(x.view(b, c, -1, h, w))
        b, c, _, h, w = x.shape
        if self.isproj:
            x = self.proj(x.view(b, c, -1, w))
        x = x.view(b, -1, u, v, h, w)
        return x


class CCA(nn.Module):

    def __init__(self, kernel_sizes=[3, 3], planes=[16, 1]):
        super(CCA, self).__init__()
        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers):
            ch_in = 1 if i == 0 else planes[i - 1]
            ch_out = planes[i]
            k_size = kernel_sizes[i]
            nn_modules.append(SepConv4d(in_planes=ch_in, out_planes=ch_out, ksize=k_size, do_padding=True))
            if i != num_layers - 1:
                nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)

    def forward(self, x):
        x = self.conv(x) + self.conv(x.permute(0, 1, 4, 5, 2, 3)).permute(0, 1, 4, 5, 2, 3)
        return x


class SCR(nn.Module):

    def __init__(self, planes=[640, 64, 64, 64, 640], stride=(1, 1, 1), ksize=3, do_padding=False, bias=False):
        super(SCR, self).__init__()
        self.ksize = (ksize,) * 4 if isinstance(ksize, int) else ksize
        padding1 = (0, self.ksize[2] // 2, self.ksize[3] // 2) if do_padding else (0, 0, 0)
        self.conv1x1_in = nn.Sequential(nn.Conv2d(planes[0], planes[1], kernel_size=1, bias=False, padding=0), nn.BatchNorm2d(planes[1]), nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv3d(planes[1], planes[2], (1, self.ksize[2], self.ksize[3]), stride=stride, bias=bias, padding=padding1), nn.BatchNorm3d(planes[2]), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv3d(planes[2], planes[3], (1, self.ksize[2], self.ksize[3]), stride=stride, bias=bias, padding=padding1), nn.BatchNorm3d(planes[3]), nn.ReLU(inplace=True))
        self.conv1x1_out = nn.Sequential(nn.Conv2d(planes[3], planes[4], kernel_size=1, bias=False, padding=0), nn.BatchNorm2d(planes[4]))

    def forward(self, x):
        b, c, h, w, u, v = x.shape
        x = x.view(b, c, h * w, u * v)
        x = self.conv1x1_in(x)
        c = x.shape[1]
        x = x.view(b, c, h * w, u, v)
        x = self.conv1(x)
        x = self.conv2(x)
        c = x.shape[1]
        x = x.view(b, c, h, w)
        x = self.conv1x1_out(x)
        return x


class SelfCorrelationComputation(nn.Module):

    def __init__(self, kernel_size=(5, 5), padding=2):
        super(SelfCorrelationComputation, self).__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.relu(x)
        x = F.normalize(x, dim=1, p=2)
        identity = x
        x = self.unfold(x)
        x = x.view(b, c, self.kernel_size[0], self.kernel_size[1], h, w)
        x = x * identity.unsqueeze(2).unsqueeze(2)
        x = x.permute(0, 1, 4, 5, 2, 3).contiguous()
        return x


class SCRLayer(nn.Module):

    def __init__(self, planes=[640, 64, 64, 64, 640]):
        super(SCRLayer, self).__init__()
        kernel_size = 5, 5
        padding = 2
        stride = 1, 1, 1
        self.model = nn.Sequential(SelfCorrelationComputation(kernel_size=kernel_size, padding=padding), SCR(planes=planes, stride=stride))

    def forward(self, x):
        return self.model(x)


class CCALayer(nn.Module):

    def __init__(self, feat_dim, way_num, shot_num, query_num, temperature, temperature_attn):
        super(CCALayer, self).__init__()
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num
        self.temperature = temperature
        self.temperature_attn = temperature_attn
        self.cca_module = CCA(kernel_sizes=[3, 3], planes=[16, 1])
        self.cca_1x1 = nn.Sequential(nn.Conv2d(feat_dim, 64, kernel_size=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())

    def gaussian_normalize(self, x, dim, eps=1e-05):
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.var(x, dim=dim, keepdim=True)
        x = torch.div(x - x_mean, torch.sqrt(x_var + eps))
        return x

    def get_4d_correlation_map(self, spt, qry):
        """
        The value H and W both for support and query is the same, but their subscripts are symbolic.
        :param spt: way * C * H_s * W_s
        :param qry: num_qry * C * H_q * W_q
        :return: 4d correlation tensor: num_qry * way * H_s * W_s * H_q * W_q
        :rtype:
        """
        way = spt.shape[0]
        num_qry = qry.shape[0]
        spt = self.cca_1x1(spt)
        qry = self.cca_1x1(qry)
        spt = F.normalize(spt, p=2, dim=1, eps=1e-08)
        qry = F.normalize(qry, p=2, dim=1, eps=1e-08)
        spt = spt.unsqueeze(0).repeat(num_qry, 1, 1, 1, 1)
        qry = qry.unsqueeze(1).repeat(1, way, 1, 1, 1)
        similarity_map = torch.einsum('qncij,qnckl->qnijkl', spt, qry)
        return similarity_map

    def normalize_feature(self, x):
        return x - x.mean(1).unsqueeze(1)

    def forward(self, spt, qry):
        spt = spt.squeeze(0)
        spt = self.normalize_feature(spt)
        qry = self.normalize_feature(qry)
        corr4d = self.get_4d_correlation_map(spt, qry)
        num_qry, way, H_s, W_s, H_q, W_q = corr4d.size()
        corr4d = self.cca_module(corr4d.view(-1, 1, H_s, W_s, H_q, W_q))
        corr4d_s = corr4d.view(num_qry, way, H_s * W_s, H_q, W_q)
        corr4d_q = corr4d.view(num_qry, way, H_s, W_s, H_q * W_q)
        corr4d_s = self.gaussian_normalize(corr4d_s, dim=2)
        corr4d_q = self.gaussian_normalize(corr4d_q, dim=4)
        corr4d_s = F.softmax(corr4d_s / self.temperature_attn, dim=2)
        corr4d_s = corr4d_s.view(num_qry, way, H_s, W_s, H_q, W_q)
        corr4d_q = F.softmax(corr4d_q / self.temperature_attn, dim=4)
        corr4d_q = corr4d_q.view(num_qry, way, H_s, W_s, H_q, W_q)
        attn_s = corr4d_s.sum(dim=[4, 5])
        attn_q = corr4d_q.sum(dim=[2, 3])
        spt_attended = attn_s.unsqueeze(2) * spt.unsqueeze(0)
        qry_attended = attn_q.unsqueeze(2) * qry.unsqueeze(1)
        if self.shot_num > 1:
            spt_attended = spt_attended.view(num_qry, self.way_num, self.shot_num, *spt_attended.shape[2:])
            qry_attended = qry_attended.view(num_qry, self.way_num, self.shot_num, *qry_attended.shape[2:])
            spt_attended = spt_attended.mean(dim=2)
            qry_attended = qry_attended.mean(dim=2)
        spt_attended = spt_attended.mean(dim=[-1, -2])
        qry_attended = qry_attended.mean(dim=[-1, -2])
        qry_pooled = qry.mean(dim=[-1, -2])
        similarity_matrix = F.cosine_similarity(spt_attended, qry_attended, dim=-1)
        return similarity_matrix / self.temperature, qry_pooled


class DistillKLLoss(nn.Module):

    def __init__(self, T):
        super(DistillKLLoss, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        if y_t is None:
            return 0.0
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * self.T ** 2 / y_s.size(0)
        return loss


class distLinear(nn.Module):

    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        self.class_wise_learnable_norm = True
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0)
        if outdim <= 200:
            self.scale_factor = 2
        else:
            self.scale_factor = 10

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-05)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 1e-05)
        cos_dist = self.L(x_normalized)
        scores = self.scale_factor * cos_dist
        return scores


class L2DistLoss(nn.Module):

    def __init__(self):
        super(L2DistLoss, self).__init__()

    def forward(self, feat1, feat2):
        loss = torch.mean(torch.sqrt(torch.sum((feat1 - feat2) ** 2, dim=1)))
        if torch.isnan(loss).any():
            loss = 0.0
        return loss


class LabelSmoothCELoss(nn.Module):

    def __init__(self, smoothing):
        super(LabelSmoothCELoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, output, target):
        log_prob = F.log_softmax(output, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class ANILLayer(nn.Module):

    def __init__(self, feat_dim, hid_dim, way_num):
        super(ANILLayer, self).__init__()
        self.layers = nn.Sequential(nn.Linear(feat_dim, way_num))

    def forward(self, x):
        return self.layers(x)


class BOILLayer(nn.Module):

    def __init__(self, feat_dim=64, way_num=5) ->None:
        super(BOILLayer, self).__init__()
        self.layers = nn.Sequential(nn.Linear(feat_dim, way_num))

    def forward(self, x):
        return self.layers(x)


def cal_log_prob(x, mean, var):
    eps = 1e-20
    log_unnormalized = -0.5 * ((x - mean) / (var + eps)) ** 2
    log_normalization = torch.log(var + eps) + 0.5 * torch.log(2 * torch.tensor(math.pi))
    return log_unnormalized - log_normalization


def cal_kl_div(latent, mean, var):
    return torch.mean(cal_log_prob(latent, mean, var) - cal_log_prob(latent, torch.zeros(mean.size()), torch.ones(var.size())))


def sample(weight, size):
    mean, var = weight[:, :, :size], weight[:, :, size:]
    z = torch.normal(0.0, 1.0, mean.size())
    return mean + var * z


class Encoder(nn.Module):

    def __init__(self, way_num, shot_num, feat_dim, hid_dim, drop_prob=0.0):
        super(Encoder, self).__init__()
        self.way_num = way_num
        self.shot_num = shot_num
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.encoder_func = nn.Linear(feat_dim, hid_dim)
        self.relation_net = nn.Sequential(nn.Linear(2 * hid_dim, 2 * hid_dim, bias=False), nn.ReLU(), nn.Linear(2 * hid_dim, 2 * hid_dim, bias=False), nn.ReLU(), nn.Linear(2 * hid_dim, 2 * hid_dim, bias=False), nn.ReLU())
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.drop_out(x)
        out = self.encoder_func(x)
        episode_size = out.size(0)
        out = out.contiguous().reshape(episode_size, self.way_num, self.shot_num, -1)
        t1 = torch.repeat_interleave(out, self.shot_num, dim=2)
        t1 = torch.repeat_interleave(t1, self.way_num, dim=1)
        t2 = out.repeat((1, self.way_num, self.shot_num, 1))
        x = torch.cat((t1, t2), dim=-1)
        x = self.relation_net(x)
        x = x.reshape(episode_size, self.way_num, self.way_num * self.shot_num * self.shot_num, -1)
        x = torch.mean(x, dim=2)
        latent = sample(x, self.hid_dim)
        mean, var = x[:, :, :self.hid_dim], x[:, :, self.hid_dim:]
        kl_div = cal_kl_div(latent, mean, var)
        return latent, kl_div


class Decoder(nn.Module):

    def __init__(self, feat_dim, hid_dim):
        super(Decoder, self).__init__()
        self.decoder_func = nn.Linear(hid_dim, 2 * feat_dim)

    def forward(self, x):
        return self.decoder_func(x)


class MAMLLayer(nn.Module):

    def __init__(self, feat_dim=64, way_num=5) ->None:
        super(MAMLLayer, self).__init__()
        self.layers = nn.Sequential(nn.Linear(feat_dim, way_num))

    def forward(self, x):
        return self.layers(x)


class IFSLUtils(nn.Module):

    def __init__(self, embed_func, feat_dim, ifsl_param, device):
        super(IFSLUtils, self).__init__()
        self.embed_func = embed_func
        self.feat_dim = feat_dim
        self.device = device
        for key, value in ifsl_param.items():
            setattr(self, key, value)
        self.linear = nn.Linear(feat_dim, self.class_num)
        self.linear = self._load_state_dict(self.linear, self.cls_path)
        self.softmax = nn.Softmax(dim=1)
        self.features = torch.from_numpy(self.get_pretrain_features()).float()
        if self.normalize_d:
            self.features = self.normalize(self.features)
        self.mean_features = self.features.mean(dim=0)

    def classify(self, x, is_feature=False):
        if is_feature is True:
            return self.softmax(self.linear(x))
        return self.softmax(self.linear(self(x)))

    def _load_state_dict(self, model, state_dict_path):
        if state_dict_path is not None:
            model_state_dict = torch.load(state_dict_path, map_location='cpu')
            model.load_state_dict(model_state_dict)
        return model

    def get_pretrain_features(self):
        if self.feature_path is not None:
            return np.load(self.feature_path)
        None
        return np.zeros((self.class_num, self.feat_dim))

    def normalize(self, x, dim=1):
        x_norm = torch.norm(x, p=2, dim=dim).unsqueeze(dim).expand_as(x).detach()
        x_normalized = x.div(x_norm + 1e-05)
        return x_normalized

    def fuse_proba(self, p1, p2):
        sigmoid = torch.nn.Sigmoid()
        if self.logit_fusion == 'linear_sum':
            return p1 + p2
        elif self.logit_fusion == 'product':
            return torch.log(sigmoid(p1) * sigmoid(p2))
        elif self.logit_fusion == 'sum':
            return torch.log(sigmoid(p1 + p2))
        elif self.logit_fusion == 'harmonic':
            p = sigmoid(p1) * sigmoid(p2)
            return torch.log(p / (1 + p))

    def fuse_features(self, x1, x2):
        if self.fusion == 'concat':
            return torch.cat((x1, x2), dim=2)
        elif self.fusion == '+':
            return x1 + x2
        elif self.fusion == '-':
            return x1 - x2

    def get_feat_dim(self):
        split_feat_dim = int(self.feat_dim / self.n_splits)
        if self.d_feature == 'pd':
            return split_feat_dim + self.num_classes
        elif self.fusion == 'concat':
            return split_feat_dim * 2
        else:
            return split_feat_dim

    def fusing(self, support, query):
        support = self.embed_func(support)
        query = self.embed_func(query)
        split_support, support_d = self.get_feature(support)
        split_query, query_d = self.get_feature(query)
        fused_support = self.fuse_features(split_support, support_d)
        fused_query = self.fuse_features(split_query, query_d)
        if self.x_zero:
            c_split_query = torch.zeros_like(split_query)
        else:
            c_split_query = split_support.mean(dim=1).unsqueeze(1).expand(split_query.shape)
        c_fused_query = self.fuse_features(c_split_query, query_d)
        if self.single is True:
            return fused_support, fused_query, c_fused_query
        else:
            return split_support, support_d, split_query, query_d

    def get_split_features(self, x, preprocess=False, center=None, preprocess_method='l2n'):
        split_dim = int(self.feat_dim / self.n_splits)
        split_features = torch.zeros(self.n_splits, x.shape[0], split_dim)
        for i in range(self.n_splits):
            start_idx = split_dim * i
            end_idx = split_dim * i + split_dim
            split_features[i] = x[:, start_idx:end_idx]
            if preprocess:
                if preprocess_method != 'dl2n':
                    split_features[i] = self.nn_preprocess(split_features[i], center[:, start_idx:end_idx], preprocessing=preprocess_method)
                else:
                    if self.normalize_before_center:
                        split_features[i] = self.normalize(split_features[i])
                    centered_data = split_features[i] - center[i]
                    split_features[i] = self.normalize(centered_data)
        return split_features

    def nn_preprocess(self, data, center=None, preprocessing='l2n'):
        if preprocessing == 'none':
            return data
        elif preprocessing == 'l2n':
            return self.normalize(data)
        elif preprocessing == 'cl2n':
            if self.normalize_before_center:
                data = self.normalize(data)
            centered_data = data - center
            return self.normalize(centered_data)

    def calc_pd(self, x):
        with torch.no_grad():
            proba = self.classify(x, True)
        return proba

    def get_d_feature(self, x):
        feat_dim = int(self.feat_dim / self.n_splits)
        if self.d_feature == 'ed':
            d_feat_dim = int(self.feat_dim / self.n_splits)
        else:
            d_feat_dim = self.num_classes
        d_feature = torch.zeros(self.n_splits, x.shape[0], d_feat_dim)
        for i in range(self.n_splits):
            start = i * feat_dim
            stop = start + feat_dim
            pd = self.calc_pd(x)
            if self.d_feature == 'pd':
                d_feature[i] = pd
            else:
                d_feature[i] = torch.mm(pd, self.features)[:, start:stop]
        return d_feature

    def get_feature(self, x):
        x_d = self.get_d_feature(x)
        if self.normalize_ed:
            x_d = self.normalize(x_d, dim=2)
        x_size = x.shape[0]
        pmean_x = self.mean_features.expand((x_size, self.feat_dim))
        x = self.nn_preprocess(x, pmean_x, preprocessing=self.preprocess_before_split)
        split_x = self.get_split_features(x, preprocess=True, center=pmean_x, preprocess_method=self.preprocess_after_split)
        return split_x, x_d

    def one_hot(self, y, num_class):
        return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)


class FullyContextualEmbedding(nn.Module):

    def __init__(self, feat_dim):
        super(FullyContextualEmbedding, self).__init__()
        self.lstmcell = nn.LSTMCell(feat_dim * 2, feat_dim)
        self.softmax = nn.Softmax(dim=1)
        self.c_0 = Variable(torch.zeros(1, feat_dim))
        self.feat_dim = feat_dim

    def forward(self, f, G):
        h = f
        c = self.c_0.expand_as(f)
        G_T = G.transpose(0, 1)
        K = G.size(0)
        for k in range(K):
            logit_a = h.mm(G_T)
            a = self.softmax(logit_a)
            r = a.mm(G)
            x = torch.cat((f, r), 1)
            h, c = self.lstmcell(x, (h, c))
            h = h + f
        return h

    def cuda(self):
        super(FullyContextualEmbedding, self)
        self.c_0 = self.c_0
        self.lstmcell = self.lstmcell
        return self


class MatchingNetLayer(nn.Module):

    def __init__(self, feat_dim):
        super(MatchingNetLayer, self).__init__()
        self.feat_dim = feat_dim
        self.FCE = FullyContextualEmbedding(self.feat_dim)
        self.G_encoder = nn.LSTM(self.feat_dim, self.feat_dim, 1, batch_first=True, bidirectional=True)

    def forward(self, support, query):
        G_encoder = self.G_encoder
        FCE = self.FCE
        out_G = G_encoder(support.unsqueeze(0))[0]
        out_G = out_G.squeeze(0)
        G = support + out_G[:, :support.size(1)] + out_G[:, support.size(1):]
        F = FCE(query, G)
        return G, F

    def cuda(self):
        super(MatchingNetLayer, self)
        self.FCE = self.FCE
        self.G_encoder = self.G_encoder
        return self


class StepLossAdapter(nn.Module):

    def __init__(self, input_dim, num_loss_net_layers, args):
        super(StepLossAdapter, self).__init__()
        self.args = args
        output_dim = num_loss_net_layers * 2 * 2
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.activation = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(input_dim, output_dim)
        self.multiplier_bias = nn.Parameter(torch.zeros(output_dim // 2))
        self.offset_bias = nn.Parameter(torch.zeros(output_dim // 2))

    def forward(self, task_state, num_step, loss_params):
        out = self.linear1(task_state)
        out = F.relu_(out)
        out = self.linear2(out)
        generated_multiplier, generated_offset = torch.chunk(out, chunks=2, dim=-1)
        i = 0
        updated_loss_weights = dict()
        for key, val in loss_params.items():
            if 'step{}'.format(num_step) in key:
                updated_loss_weights[key] = (1 + self.multiplier_bias[i] * generated_multiplier[i]) * val + self.offset_bias[i] * generated_offset[i]
                i += 1
        return updated_loss_weights


class LossAdapter(nn.Module):

    def __init__(self, input_dim, num_loss_net_layers, args):
        super(LossAdapter, self).__init__()
        self.args = args
        self.num_steps = args['test_iter']
        self.loss_adapter = nn.ModuleList()
        for _ in range(self.num_steps):
            self.loss_adapter.append(StepLossAdapter(input_dim, num_loss_net_layers, args))

    def forward(self, task_state, num_step, loss_params):
        return self.loss_adapter[num_step](task_state, num_step, loss_params)


def extract_top_level_dict(current_dict):
    output_dict = {}
    for key, value in current_dict.items():
        name = key.replace('layer_dict.', '').replace('block_dict.', '').replace('module-', '')
        parts = name.split('.', 1)
        top_level = parts[0]
        sub_level = parts[1] if len(parts) > 1 else ''
        if top_level not in output_dict:
            output_dict[top_level] = value if sub_level == '' else {sub_level: value}
        elif isinstance(output_dict[top_level], dict):
            output_dict[top_level][sub_level] = value
        else:
            output_dict[top_level] = {sub_level: value}
    return output_dict


class MetaLinearLayer(nn.Module):

    def __init__(self, input_shape, num_filters, use_bias):
        super(MetaLinearLayer, self).__init__()
        b, c = input_shape
        self.use_bias = use_bias
        self.weights = nn.Parameter(torch.ones(num_filters, c))
        nn.init.xavier_uniform_(self.weights)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x, params=None):
        weight = self.weights
        bias = self.bias if self.use_bias else None
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            weight = params['weights']
            if self.use_bias:
                bias = params['bias']
        out = F.linear(input=x, weight=weight, bias=bias)
        return out


class MetaStepLossNetwork(nn.Module):

    def __init__(self, input_dim, args):
        super(MetaStepLossNetwork, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.input_shape = 1, input_dim
        self.build_network()

    def build_network(self):
        x = torch.zeros(self.input_shape)
        out = x
        self.linear1 = MetaLinearLayer(input_shape=self.input_shape, num_filters=self.input_dim, use_bias=True)
        self.linear2 = MetaLinearLayer(input_shape=(1, self.input_dim), num_filters=1, use_bias=True)
        out = self.linear1(out)
        out = F.relu_(out)
        out = self.linear2(out)

    def forward(self, x, params=None):
        linear1_params = None
        linear2_params = None
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            linear1_params = params['linear1']
            linear2_params = params['linear2']
        out = x
        out = self.linear1(out, linear1_params)
        out = F.relu_(out)
        out = self.linear2(out, linear2_params)
        return out

    def restore_backup_stats(self):
        for i in range(self.num_stages):
            self.layer_dict['conv{}'.format(i)].restore_backup_stats()


class MetaLossNetwork(nn.Module):

    def __init__(self, input_dim, args):
        super(MetaLossNetwork, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.input_shape = 1, input_dim
        self.num_steps = args['test_iter']
        self.build_network()

    def build_network(self):
        x = torch.zeros(self.input_shape)
        self.layer_dict = nn.ModuleDict()
        for i in range(self.num_steps):
            self.layer_dict['step{}'.format(i)] = MetaStepLossNetwork(self.input_dim, args=self.args)
            out = self.layer_dict['step{}'.format(i)](x)

    def forward(self, x, num_step, params=None):
        param_dict = dict()
        if params is not None:
            params = {key: value for key, value in params.items()}
            param_dict = extract_top_level_dict(current_dict=params)
        for name, _ in self.layer_dict.named_parameters():
            path_bits = name.split('.')
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None
        out = x
        out = self.layer_dict['step{}'.format(num_step)](out, param_dict['step{}'.format(num_step)])
        return out

    def restore_backup_stats(self):
        for i in range(self.num_stages):
            self.layer_dict['conv{}'.format(i)].restore_backup_stats()


def convert_maml_module(module):
    """Convert a normal model to MAML model.

    Replace nn.Linear with Linear_fw, nn.Conv2d with Conv2d_fw.

    Args:
        module: The module (model component) to be converted.

    Returns: A MAML model.

    """
    module_output = module
    if isinstance(module, torch.nn.modules.Linear):
        module_output = Linear_fw(module.in_features, module.out_features, False if module.bias is None else True)
    elif isinstance(module, torch.nn.modules.Conv2d):
        module_output = Conv2d_fw(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, False if module.bias is None else True)
    elif isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
        module_output = BatchNorm2d_fw(module.num_features)
    for name, child in module.named_children():
        module_output.add_module(name, convert_maml_module(child))
    del module
    return module_output


def convert_mtl_module(module, MTL=False):
    """Convert a normal model to MTL model.

    Replace nn.Conv2d with Conv2dMtl.

    Args:
        module: The module (model component) to be converted.

    Returns: A MTL model.

    """
    module_output = module
    if isinstance(module, torch.nn.modules.Conv2d):
        module_output = Conv2dMtl(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, False if module.bias is None else True, MTL)
    for name, child in module.named_children():
        module_output.add_module(name, convert_mtl_module(child, MTL))
    del module
    return module_output


def binv(b_mat):
    """
    Computes an inverse of each matrix in the batch.
    Pytorch 0.4.1 does not support batched matrix inverse.
    Hence, we are solving AX=I.

    Parameters:
      b_mat:  a (n_batch, n, n) Tensor.
    Returns: a (n_batch, n, n) Tensor.
    """
    id_matrix = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    b_inv = torch.linalg.solve(b_mat, id_matrix)
    return b_inv


def computeGramMatrix(A, B):
    """
    Constructs a linear kernel matrix between A and B.
    We assume that each row in A and B represents a d-dimensional feature vector.

    Parameters:
      A:  a (n_batch, n, d) Tensor.
      B:  a (n_batch, m, d) Tensor.
    Returns: a (n_batch, n, m) Tensor.
    """
    assert A.dim() == 3, 'A must be a 3-D Tensor.'
    assert B.dim() == 3, 'B must be a 3-D Tensor.'
    assert A.size(0) == B.size(0) and A.size(2) == B.size(2), 'A and B must have the same batch size and feature dimension.'
    return torch.bmm(A, B.transpose(1, 2))


def one_hot(indices, depth, use_cuda=True):
    if use_cuda:
        encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    else:
        encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)
    return encoded_indicies


class R2D2Layer(nn.Module):

    def __init__(self):
        super(R2D2Layer, self).__init__()
        self.register_parameter('alpha', nn.Parameter(torch.tensor([1.0])))
        self.register_parameter('beta', nn.Parameter(torch.tensor([0.0])))
        self.register_parameter('gamma', nn.Parameter(torch.tensor([50.0])))

    def forward(self, way_num, shot_num, query, support, support_target):
        tasks_per_batch = query.size(0)
        n_support = support.size(1)
        support_target = support_target.squeeze()
        assert query.dim() == 3, 'query must be a 3-D Tensor.'
        assert support.dim() == 3, 'support must be a 3-D Tensor.'
        assert query.size(0) == support.size(0) and query.size(2) == support.size(2), 'query and support must have the same batch size and feature dimension.'
        assert n_support == way_num * shot_num, 'n_support must be equal to way_num * shot_num.'
        support_labels_one_hot = one_hot(support_target.view(tasks_per_batch * n_support), way_num)
        support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, way_num)
        id_matrix = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support)
        ridge_sol = computeGramMatrix(support, support) + self.gamma * id_matrix
        ridge_sol = binv(ridge_sol)
        ridge_sol = torch.bmm(support.transpose(1, 2), ridge_sol)
        ridge_sol = torch.bmm(ridge_sol, support_labels_one_hot)
        logit = torch.bmm(query, ridge_sol)
        logit = self.alpha * logit + self.beta
        return logit, ridge_sol


def _l2norm(x, dim=1, keepdim=True):
    return x / (1e-16 + torch.norm(x, 2, dim, keepdim))


def l2distance(x, y):
    assert x.shape[:-2] == y.shape[:-2]
    prefix_shape = x.shape[:-2]
    c, M_x = x.shape[-2:]
    M_y = y.shape[-1]
    x = x.view(-1, c, M_x)
    y = y.view(-1, c, M_y)
    x_t = x.transpose(1, 2)
    x_t2 = x_t.pow(2.0).sum(-1, keepdim=True)
    y2 = y.pow(2.0).sum(1, keepdim=True)
    ret = x_t2 + y2 - 2.0 * x_t @ y
    ret = ret.view(prefix_shape + (M_x, M_y))
    return ret


class Similarity(nn.Module):

    def __init__(self, metric='cosine'):
        super().__init__()
        self.metric = metric

    def forward(self, support_xf, query_xf):
        if query_xf.dim() == 5:
            b, q, c, h, w = query_xf.shape
            query_xf = query_xf.view(b, q, c, h * w)
        else:
            b, q = query_xf.shape[:2]
        s = support_xf.shape[1]
        support_xf = support_xf.unsqueeze(1).expand(-1, q, -1, -1, -1)
        query_xf = query_xf.unsqueeze(2).expand(-1, -1, s, -1, -1)
        M_q = query_xf.shape[-1]
        M_s = support_xf.shape[-1]
        if self.metric == 'cosine':
            support_xf = _l2norm(support_xf, dim=-2)
            query_xf = _l2norm(query_xf, dim=-2)
            query_xf = torch.transpose(query_xf, 3, 4)
            return query_xf @ support_xf
        elif self.metric == 'innerproduct':
            query_xf = torch.transpose(query_xf, 3, 4)
            return query_xf @ support_xf
        elif self.metric == 'euclidean':
            return 1 - l2distance(query_xf, support_xf)
        elif self.metric == 'neg_ed':
            query_xf = query_xf.contiguous().view(-1, c, M_q).transpose(-2, -1).contiguous()
            support_xf = support_xf.contiguous().view(-1, c, M_s).transpose(-2, -1).contiguous()
            dist = torch.cdist(query_xf, support_xf)
            return -dist.view(b, q, s, M_q, M_s) / 2.0
        else:
            raise NotImplementedError


class MCLMask(nn.Module):

    def __init__(self, katz_factor, gamma, gamma2):
        super().__init__()
        self.inner_simi = Similarity(metric='cosine')
        self.gamma = gamma
        self.gamma2 = gamma2
        self.katz_factor = katz_factor

    def forward(self, support_xf, query_xf, n_way, k_shot):
        self.n_way = n_way
        self.k_shot = k_shot
        b, s, c, h, w = support_xf.shape
        q = query_xf.shape[1]
        support_xf = support_xf.view(b, self.n_way, self.k_shot, c, h, w).mean(2)
        support_xf = support_xf.view(b, self.n_way, c, h * w)
        S = self.inner_simi(support_xf, query_xf)
        M_q = S.shape[-2]
        M_s = S.shape[2] * S.shape[-1]
        S = S.permute(0, 1, 3, 2, 4).contiguous().view(b * q, M_q, M_s)
        N_examples = b * q
        St = S.transpose(-2, -1)
        device = S.device
        T_sq = torch.exp(self.gamma * (S - S.max(-1, keepdim=True)[0]))
        T_sq = T_sq / T_sq.sum(-1, keepdim=True)
        T_qs = torch.exp(self.gamma2 * (St - St.max(-1, keepdim=True)[0]))
        T_qs = T_qs / T_qs.sum(-1, keepdim=True)
        T = torch.cat([torch.cat([torch.zeros((N_examples, M_s, M_s), device=device), T_sq.transpose(-2, -1)], dim=-1), torch.cat([T_qs.transpose(-2, -1), torch.zeros((N_examples, M_q, M_q), device=device)], dim=-1)], dim=-2)
        katz = (torch.inverse(torch.eye(M_s + M_q, device=device)[None].repeat(N_examples, 1, 1) - self.katz_factor * T) - torch.eye(M_s + M_q, device=S.device)[None].repeat(N_examples, 1, 1)) @ torch.ones((N_examples, M_s + M_q, 1), device=device)
        katz_query = katz.squeeze(-1)[:, M_s:] / katz.squeeze(-1)[:, M_s:].sum(-1, keepdim=True)
        katz_query = katz_query.view(b, q, h, w).unsqueeze(2)
        return katz_query


class Predictor(nn.Module):

    def __init__(self, feat_dim, hid_dim, out_dim):
        super(Predictor, self).__init__()
        self.layers = nn.Sequential(nn.Linear(feat_dim, hid_dim), nn.ELU(), nn.Linear(hid_dim, hid_dim), nn.ELU(), nn.Linear(hid_dim, out_dim))

    def forward(self, x):
        out = self.layers(x)
        return out


class VERSALayer(nn.Module):

    def __init__(self, sample_num):
        super(VERSALayer, self).__init__()
        self.sample_num = sample_num
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def forward(self, way_num, query_feat, query_target, weight_mean, weight_logvar, bias_mean, bias_logvar):
        query_target = query_target.contiguous().reshape(-1)
        episode_size = query_feat.size(0)
        logits_mean_query = torch.matmul(query_feat, weight_mean) + bias_mean
        logits_log_var_query = torch.log(torch.matmul(query_feat ** 2, torch.exp(weight_logvar)) + torch.exp(bias_logvar))
        logits_sample_query = self.sample_normal(logits_mean_query, logits_log_var_query, self.sample_num).contiguous().reshape(-1, way_num)
        query_label_tiled = query_target.repeat(self.sample_num)
        loss = -self.loss_func(logits_sample_query, query_label_tiled)
        loss = loss.contiguous().reshape(episode_size, self.sample_num, -1).permute([1, 0, 2]).contiguous().reshape(self.sample_num, -1)
        task_score = torch.logsumexp(loss, dim=0) - torch.log(torch.as_tensor(self.sample_num, dtype=torch.float))
        logits_sample_query = logits_sample_query.contiguous().reshape(self.sample_num, -1, way_num)
        averaged_prediction = torch.logsumexp(logits_sample_query, dim=0) - torch.log(torch.as_tensor(self.sample_num, dtype=torch.float))
        return averaged_prediction, task_score

    def sample_normal(self, mu, log_variance, num_samples):
        shape = torch.cat([torch.as_tensor([num_samples]), torch.as_tensor(mu.size())])
        eps = torch.randn(shape.cpu().numpy().tolist())
        return mu + eps * torch.sqrt(torch.exp(log_variance))


class ADMLayer(nn.Module):

    def __init__(self, way_num, shot_num, query_num, n_k, device):
        super(ADMLayer, self).__init__()
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num
        self.n_k = n_k
        self.device = device
        self.normLayer = nn.BatchNorm1d(self.way_num * 2, affine=True)
        self.fcLayer = nn.Conv1d(1, 1, kernel_size=2, stride=1, dilation=5, bias=False)

    def _cal_cov_matrix_batch(self, feat):
        e, _, n_local, c = feat.size()
        feature_mean = torch.mean(feat, 2, True)
        feat = feat - feature_mean
        cov_matrix = torch.matmul(feat.permute(0, 1, 3, 2), feat)
        cov_matrix = torch.div(cov_matrix, n_local - 1)
        cov_matrix = cov_matrix + 0.01 * torch.eye(c)
        return feature_mean, cov_matrix

    def _cal_cov_batch(self, feat):
        e, b, c, h, w = feat.size()
        feat = feat.reshape(e, b, c, -1).permute(0, 1, 3, 2)
        feat_mean = torch.mean(feat, 2, True)
        feat = feat - feat_mean
        cov_matrix = torch.matmul(feat.permute(0, 1, 3, 2), feat)
        cov_matrix = torch.div(cov_matrix, h * w - 1)
        cov_matrix = cov_matrix + 0.01 * torch.eye(c)
        return feat_mean, cov_matrix

    def _calc_kl_dist_batch(self, mean1, cov1, mean2, cov2):
        """

        :param mean1: e * 75 * 1 * 64
        :param cov1: e * 75 * 64 * 64
        :param mean2: e * 5 * 1 * 64
        :param cov2: e * 5 * 64 * 64
        :return:
        """
        cov2_inverse = torch.inverse(cov2)
        mean_diff = -(mean1 - mean2.squeeze(2).unsqueeze(1))
        matrix_prod = torch.matmul(cov1.unsqueeze(2), cov2_inverse.unsqueeze(1))
        trace_dist = torch.diagonal(matrix_prod, offset=0, dim1=-2, dim2=-1)
        trace_dist = torch.sum(trace_dist, dim=-1)
        maha_prod = torch.matmul(mean_diff.unsqueeze(3), cov2_inverse.unsqueeze(1))
        maha_prod = torch.matmul(maha_prod, mean_diff.unsqueeze(4))
        maha_prod = maha_prod.squeeze(4)
        maha_prod = maha_prod.squeeze(3)
        matrix_det = torch.slogdet(cov2).logabsdet.unsqueeze(1) - torch.slogdet(cov1).logabsdet.unsqueeze(2)
        kl_dist = trace_dist + maha_prod + matrix_det - mean1.size(3)
        return kl_dist / 2.0

    def _cal_adm_sim(self, query_feat, support_feat):
        """

        :param query_feat: e * 75 * 64 * 21 * 21
        :param support_feat: e * 25 * 64 * 21 * 21
        :return:
        """
        e, b, c, h, w = query_feat.size()
        e, s, _, _, _ = support_feat.size()
        query_mean, query_cov = self._cal_cov_batch(query_feat)
        query_feat = query_feat.reshape(e, b, c, -1).permute(0, 1, 3, 2).contiguous()
        support_feat = support_feat.reshape(e, s, c, -1).permute(0, 1, 3, 2).contiguous()
        support_set = support_feat.reshape(e, self.way_num, self.shot_num * h * w, c)
        s_mean, s_cov = self._cal_cov_matrix_batch(support_set)
        kl_dis = -self._calc_kl_dist_batch(query_mean, query_cov, s_mean, s_cov)
        query_norm = F.normalize(query_feat, p=2, dim=3)
        support_norm = F.normalize(support_feat, p=2, dim=3)
        support_norm = support_norm.reshape(e, self.way_num, self.shot_num * h * w, c)
        inner_prod_matrix = torch.matmul(query_norm.unsqueeze(2), support_norm.permute(0, 1, 3, 2).unsqueeze(1))
        topk_value, topk_index = torch.topk(inner_prod_matrix, self.n_k, 4)
        inner_sim = torch.sum(torch.sum(topk_value, 4), 3)
        adm_sim_soft = torch.cat((kl_dis, inner_sim), 2)
        adm_sim_soft = torch.cat([self.normLayer(each_task).unsqueeze(1) for each_task in adm_sim_soft])
        adm_sim_soft = self.fcLayer(adm_sim_soft).squeeze(1).reshape([e, b, -1])
        return adm_sim_soft

    def forward(self, query_feat, support_feat):
        return self._cal_adm_sim(query_feat, support_feat)


class KLLayer(nn.Module):

    def __init__(self, way_num, shot_num, query_num, n_k, device, CMS=False):
        super(KLLayer, self).__init__()
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num
        self.n_k = n_k
        self.device = device
        self.CMS = CMS

    def _cal_cov_matrix_batch(self, feat):
        e, _, n_local, c = feat.size()
        feature_mean = torch.mean(feat, 2, True)
        feat = feat - feature_mean
        cov_matrix = torch.matmul(feat.permute(0, 1, 3, 2), feat)
        cov_matrix = torch.div(cov_matrix, n_local - 1)
        cov_matrix = cov_matrix + 0.01 * torch.eye(c)
        return feature_mean, cov_matrix

    def _cal_cov_batch(self, feat):
        e, b, c, h, w = feat.size()
        feat = feat.reshape(e, b, c, -1).permute(0, 1, 3, 2)
        feat_mean = torch.mean(feat, 2, True)
        feat = feat - feat_mean
        cov_matrix = torch.matmul(feat.permute(0, 1, 3, 2), feat)
        cov_matrix = torch.div(cov_matrix, h * w - 1)
        cov_matrix = cov_matrix + 0.01 * torch.eye(c)
        return feat_mean, cov_matrix

    def _calc_kl_dist_batch(self, mean1, cov1, mean2, cov2):
        """

        :param mean1: e * 75 * 1 * 64
        :param cov1: e * 75 * 64 * 64
        :param mean2: e * 5 * 1 * 64
        :param cov2: e * 5 * 64 * 64
        :return:
        """
        cov2_inverse = torch.inverse(cov2)
        mean_diff = -(mean1 - mean2.squeeze(2).unsqueeze(1))
        matrix_prod = torch.matmul(cov1.unsqueeze(2), cov2_inverse.unsqueeze(1))
        trace_dist = torch.diagonal(matrix_prod, offset=0, dim1=-2, dim2=-1)
        trace_dist = torch.sum(trace_dist, dim=-1)
        maha_prod = torch.matmul(mean_diff.unsqueeze(3), cov2_inverse.unsqueeze(1))
        maha_prod = torch.matmul(maha_prod, mean_diff.unsqueeze(4))
        maha_prod = maha_prod.squeeze(4)
        maha_prod = maha_prod.squeeze(3)
        matrix_det = torch.logdet(cov2).unsqueeze(1) - torch.logdet(cov1).unsqueeze(2)
        kl_dist = trace_dist + maha_prod + matrix_det - mean1.size(3)
        return kl_dist / 2.0

    def _cal_support_remaining(self, S):
        e, w, d, c = S.shape
        episode_indices = torch.tensor([j for i in range(S.size(1)) for j in range(S.size(1)) if i != j])
        S_new = torch.index_select(S, 1, episode_indices)
        S_new = S_new.reshape([e, w, -1, c])
        return S_new

    def _cal_adm_sim(self, query_feat, support_feat):
        """

        :param query_feat: e * 75 * 64 * 21 * 21
        :param support_feat: e * 25 * 64 * 21 * 21
        :return:
        """
        e, b, c, h, w = query_feat.size()
        e, s, _, _, _ = support_feat.size()
        query_mean, query_cov = self._cal_cov_batch(query_feat)
        query_feat = query_feat.reshape(e, b, c, -1).permute(0, 1, 3, 2).contiguous()
        support_feat = support_feat.reshape(e, s, c, -1).permute(0, 1, 3, 2).contiguous()
        support_set = support_feat.reshape(e, self.way_num, self.shot_num * h * w, c)
        s_mean, s_cov = self._cal_cov_matrix_batch(support_set)
        kl_dis = -self._calc_kl_dist_batch(query_mean, query_cov, s_mean, s_cov)
        if self.CMS:
            support_set_remain = self._cal_support_remaining(support_set)
            s_remain_mean, s_remain_cov = self._cal_cov_matrix_batch(support_set_remain)
            kl_dis2 = self._calc_kl_dist_batch(query_mean, query_cov, s_remain_mean, s_remain_cov)
            kl_dis = kl_dis + kl_dis2
        return kl_dis

    def forward(self, query_feat, support_feat):
        return self._cal_adm_sim(query_feat, support_feat)


class AEAModule(nn.Module):

    def __init__(self, feat_dim, scale_value, from_value, value_interval):
        super(AEAModule, self).__init__()
        self.feat_dim = feat_dim
        self.scale_value = scale_value
        self.from_value = from_value
        self.value_interval = value_interval
        self.f_psi = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim // 16), nn.LeakyReLU(0.2, inplace=True), nn.Linear(self.feat_dim // 16, 1), nn.Sigmoid())

    def forward(self, x, f_x):
        t, wq, hw, c = x.size()
        clamp_value = self.f_psi(x.reshape(t * wq * hw, c)) * self.value_interval + self.from_value
        clamp_value = clamp_value.reshape(t, wq, hw, 1)
        clamp_fx = torch.sigmoid(self.scale_value * (f_x - clamp_value))
        attention_mask = F.normalize(clamp_fx, p=1, dim=-1)
        return attention_mask


class ATL_Layer(nn.Module):

    def __init__(self, feat_dim, scale_value, atten_scale_value, from_value, value_interval):
        super(ATL_Layer, self).__init__()
        self.feat_dim = feat_dim
        self.scale_value = scale_value
        self.atten_scale_value = atten_scale_value
        self.from_value = from_value
        self.value_interval = value_interval
        self.W = nn.Sequential(nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(self.feat_dim), nn.LeakyReLU(0.2, inplace=True))
        self.attenLayer = AEAModule(self.feat_dim, self.atten_scale_value, self.from_value, self.value_interval)

    def forward(self, way_num, shot_num, query_feat, support_feat):
        t, wq, c, h, w = query_feat.size()
        _, ws, _, _, _ = support_feat.size()
        w_query = self.W(query_feat.reshape(t * wq, c, h, w)).reshape(t, wq, c, h * w).permute(0, 1, 3, 2).contiguous()
        w_support = self.W(support_feat.reshape(t * ws, c, h, w)).reshape(t, ws, c, h * w).permute(0, 2, 1, 3).contiguous().reshape(t, 1, c, ws * h * w)
        w_query = F.normalize(w_query, dim=3)
        w_support = F.normalize(w_support, dim=2)
        f_x = torch.matmul(w_query, w_support)
        atten_score = self.attenLayer(w_query, f_x)
        query_feat = query_feat.reshape(t, wq, c, h * w).permute(0, 1, 3, 2).contiguous()
        support_feat = support_feat.reshape(t, ws, c, h * w).permute(0, 2, 1, 3).contiguous().reshape(t, 1, c, ws * h * w)
        query_feat = F.normalize(query_feat, dim=3)
        support_feat = F.normalize(support_feat, dim=2)
        match_score = torch.matmul(query_feat, support_feat)
        atten_match_score = torch.mul(atten_score, match_score).reshape(t, wq, h * w, way_num, shot_num, h * w).permute(0, 1, 3, 4, 2, 5)
        score = torch.sum(atten_match_score, dim=5)
        score = torch.mean(score, dim=[3, 4]) * self.scale_value
        return score


class CrossEntropyLoss(nn.Module):

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(inputs.size(0), inputs.size(1)).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.unsqueeze(-1)
        targets = targets
        loss = (-targets * log_probs).mean(0).sum()
        return loss / inputs.size(2)


class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization.
    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
    - in_c (int): number of input channels.
    - out_c (int): number of output channels.
    - k (int or tuple): kernel size.
    - s (int or tuple): stride.
    - p (int or tuple): padding.
    """

    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class CAM(nn.Module):
    """
    Support & Query share one attention
    """

    def __init__(self, mid_channels):
        super(CAM, self).__init__()
        self.conv1 = ConvBlock(mid_channels * mid_channels, mid_channels, 1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels * mid_channels, 1, stride=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

    def get_attention(self, a):
        input_a = a
        a = a.mean(3)
        a = a.transpose(1, 3)
        a = F.relu(self.conv1(a))
        a = self.conv2(a)
        a = a.transpose(1, 3)
        a = a.unsqueeze(3)
        a = torch.mean(input_a * a, -1)
        a = F.softmax(a / 0.025, dim=-1) + 1
        return a

    def forward(self, f1, f2):
        b, n1, c, h, w = f1.size()
        n2 = f2.size(1)
        f1 = f1.reshape(b, n1, c, -1)
        f2 = f2.reshape(b, n2, c, -1)
        f1_norm = F.normalize(f1, p=2, dim=2, eps=1e-12)
        f2_norm = F.normalize(f2, p=2, dim=2, eps=1e-12)
        f1_norm = f1_norm.transpose(2, 3).unsqueeze(2)
        f2_norm = f2_norm.unsqueeze(1)
        a1 = torch.matmul(f1_norm, f2_norm)
        a2 = a1.transpose(3, 4)
        a1 = self.get_attention(a1)
        a2 = self.get_attention(a2)
        f1 = f1.unsqueeze(2) * a1.unsqueeze(3)
        f1 = f1.reshape(b, n1, n2, c, h, w)
        f2 = f2.unsqueeze(1) * a2.unsqueeze(3)
        f2 = f2.reshape(b, n1, n2, c, h, w)
        return f1.transpose(1, 2), f2.transpose(1, 2)


class CAMLayer(nn.Module):

    def __init__(self, scale_cls, iter_num_prob=35.0 / 75, num_classes=64, nFeat=512, HW=5):
        super(CAMLayer, self).__init__()
        self.scale_cls = scale_cls
        self.cam = CAM(HW)
        self.iter_num_prob = iter_num_prob
        self.nFeat = nFeat
        self.classifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1)

    def val(self, support_feat, query_feat):
        query_feat = query_feat.mean(4)
        query_feat = query_feat.mean(4)
        support_feat = F.normalize(support_feat, p=2, dim=support_feat.dim() - 1, eps=1e-12)
        query_feat = F.normalize(query_feat, p=2, dim=query_feat.dim() - 1, eps=1e-12)
        scores = self.scale_cls * torch.sum(query_feat * support_feat, dim=-1)
        return scores

    def forward(self, support_feat, query_feat, support_targets, query_targets):
        """
        support_feat: [4, 5, 512, 6, 6]
        query_feat: [4, 75, 512, 6, 6]
        support_targets: [4, 5, 5] one-hot
        query_targets: [4, 75, 5] one-hot
        """
        original_feat_shape = support_feat.size()
        batch_size = support_feat.size(0)
        n_support = support_feat.size(1)
        n_query = query_feat.size(1)
        way_num = support_targets.size(-1)
        support_feat = support_feat.reshape(batch_size, n_support, -1)
        labels_train_transposed = support_targets.transpose(1, 2)
        prototypes = torch.bmm(labels_train_transposed, support_feat)
        prototypes = prototypes.div(labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes))
        prototypes = prototypes.reshape(batch_size, -1, *original_feat_shape[2:])
        prototypes, query_feat = self.cam(prototypes, query_feat)
        prototypes = prototypes.mean(4)
        prototypes = prototypes.mean(4)
        if not self.training:
            return self.val(prototypes, query_feat)
        proto_norm = F.normalize(prototypes, p=2, dim=3, eps=1e-12)
        query_norm = F.normalize(query_feat, p=2, dim=3, eps=1e-12)
        proto_norm = proto_norm.unsqueeze(4)
        proto_norm = proto_norm.unsqueeze(5)
        cls_scores = self.scale_cls * torch.sum(query_norm * proto_norm, dim=3)
        cls_scores = cls_scores.reshape(batch_size * n_query, *cls_scores.size()[2:])
        query_feat = query_feat.reshape(batch_size, n_query, way_num, -1)
        query_feat = query_feat.transpose(2, 3)
        query_targets = query_targets.unsqueeze(3)
        query_feat = torch.matmul(query_feat, query_targets)
        query_feat = query_feat.reshape(batch_size * n_query, -1, *original_feat_shape[-2:])
        query_targets = self.classifier(query_feat)
        return query_targets, cls_scores

    def helper(self, support_feat, query_feat, support_targets):
        """
        support_targets_transposed: one-hot
        """
        b, n, c, h, w = support_feat.size()
        support_targets_transposed = support_targets.transpose(1, 2)
        support_feat = torch.bmm(support_targets_transposed, support_feat.reshape(b, n, -1))
        support_feat = support_feat.div(support_targets_transposed.sum(dim=2, keepdim=True).expand_as(support_feat))
        support_feat = support_feat.reshape(b, -1, c, h, w)
        support_feat, query_feat = self.cam(support_feat, query_feat)
        support_feat = support_feat.mean(-1).mean(-1)
        query_feat = query_feat.mean(-1).mean(-1)
        query_feat = F.normalize(query_feat, p=2, dim=query_feat.dim() - 1, eps=1e-12)
        support_feat = F.normalize(support_feat, p=2, dim=support_feat.dim() - 1, eps=1e-12)
        scores = self.scale_cls * torch.sum(query_feat * support_feat, dim=-1)
        return scores


class ConvMLayer(nn.Module):

    def __init__(self, way_num, shot_num, query_num, n_local):
        super(ConvMLayer, self).__init__()
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num
        self.conv1dLayer = nn.Sequential(nn.LeakyReLU(0.2, inplace=True), nn.Dropout(), nn.Conv1d(in_channels=1, out_channels=1, kernel_size=n_local, stride=n_local))

    def _calc_support_cov(self, support_feat):
        t, ws, c, h, w = support_feat.size()
        support_feat = support_feat.view(t, ws, c, h * w).permute(0, 1, 3, 2).contiguous()
        support_feat = support_feat.view(t, self.way_num, self.shot_num * h * w, c)
        support_feat = support_feat - torch.mean(support_feat, dim=2, keepdim=True)
        cov_mat = torch.matmul(support_feat.permute(0, 1, 3, 2), support_feat)
        cov_mat = torch.div(cov_mat, h * w - 1)
        return cov_mat

    def _calc_similarity(self, query_feat, support_cov_mat):
        t, wq, c, h, w = query_feat.size()
        query_feat = query_feat.view(t, wq, c, h * w).permute(0, 1, 3, 2).contiguous()
        query_feat = query_feat - torch.mean(query_feat, dim=2, keepdim=True)
        query_feat = query_feat.unsqueeze(2)
        support_cov_mat = support_cov_mat.unsqueeze(1)
        prod_mat = torch.matmul(query_feat, support_cov_mat)
        prod_mat = torch.matmul(prod_mat, torch.transpose(query_feat, 3, 4)).contiguous().view(t * self.way_num * wq, h * w, h * w)
        cov_sim = torch.diagonal(prod_mat, dim1=1, dim2=2).contiguous()
        cov_sim = cov_sim.view(t * wq, 1, self.way_num * h * w)
        return cov_sim

    def forward(self, query_feat, support_feat):
        t, wq, c, h, w = query_feat.size()
        support_cov_mat = self._calc_support_cov(support_feat)
        cov_sim = self._calc_similarity(query_feat, support_cov_mat)
        score = self.conv1dLayer(cov_sim).view(t, wq, self.way_num)
        return score


class SmoothCELoss(nn.Module):

    def __init__(self, eps=0.1, way=5):
        super(SmoothCELoss, self).__init__()
        self.eps = eps

    def forward(self, results, label, way=5):
        one_hot = torch.zeros_like(results).scatter(1, label.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / (way - 1)
        log_prb = F.log_softmax(results, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.mean()
        return loss


class CPEALayer(nn.Module):

    def __init__(self, in_dim=384):
        super(CPEALayer, self).__init__()
        self.fc1 = Mlp(in_features=in_dim, hidden_features=int(in_dim / 4), out_features=in_dim)
        self.fc_norm1 = nn.LayerNorm(in_dim)
        self.fc2 = Mlp(in_features=196 ** 2, hidden_features=256, out_features=1)

    def forward(self, feat_query, feat_shot, shot):
        _, n, c = feat_query.size()
        feat_query = self.fc1(torch.mean(feat_query, dim=1, keepdim=True)) + feat_query
        feat_shot = self.fc1(torch.mean(feat_shot, dim=1, keepdim=True)) + feat_shot
        feat_query = self.fc_norm1(feat_query)
        feat_shot = self.fc_norm1(feat_shot)
        query_class = feat_query[:, 0, :].unsqueeze(1)
        query_image = feat_query[:, 1:, :]
        support_class = feat_shot[:, 0, :].unsqueeze(1)
        support_image = feat_shot[:, 1:, :]
        feat_query = query_image + 2.0 * query_class
        feat_shot = support_image + 2.0 * support_class
        feat_query = F.normalize(feat_query, p=2, dim=2)
        feat_query = feat_query - torch.mean(feat_query, dim=2, keepdim=True)
        feat_shot = feat_shot.contiguous().reshape(shot, -1, n - 1, c)
        feat_shot = feat_shot.mean(dim=0)
        feat_shot = F.normalize(feat_shot, p=2, dim=2)
        feat_shot = feat_shot - torch.mean(feat_shot, dim=2, keepdim=True)
        results = []
        for idx in range(feat_query.size(0)):
            tmp_query = feat_query[idx]
            tmp_query = tmp_query.unsqueeze(0)
            out = torch.matmul(feat_shot, tmp_query.transpose(1, 2))
            out = out.flatten(1)
            out = self.fc2(out.pow(2))
            out = out.transpose(0, 1)
            results.append(out)
        return results


class DN4Layer(nn.Module):

    def __init__(self, n_k):
        super(DN4Layer, self).__init__()
        self.n_k = n_k

    def forward(self, query_feat, support_feat, way_num, shot_num, query_num):
        t, wq, c, h, w = query_feat.size()
        _, ws, _, _, _ = support_feat.size()
        query_feat = query_feat.view(t, way_num * query_num, c, h * w).permute(0, 1, 3, 2)
        query_feat = F.normalize(query_feat, p=2, dim=-1).unsqueeze(2)
        support_feat = support_feat.view(t, way_num, shot_num, c, h * w).permute(0, 1, 3, 2, 4).contiguous().view(t, way_num, c, shot_num * h * w)
        support_feat = F.normalize(support_feat, p=2, dim=2).unsqueeze(1)
        relation = torch.matmul(query_feat, support_feat)
        topk_value, _ = torch.topk(relation, self.n_k, dim=-1)
        score = torch.sum(topk_value, dim=[3, 4])
        return score


class DSNLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, query_feat, support_feat, way_num, shot_num, normalize=True, discriminative=False):
        e, ws, d = support_feat.size()
        support_feat = support_feat.reshape(e, way_num, shot_num, -1)
        query_feat = query_feat.unsqueeze(1)
        try:
            UU, _, _ = torch.linalg.svd(support_feat.permute(0, 1, 3, 2).double())
        except AttributeError:
            UU, _, _ = torch.svd(support_feat.permute(0, 1, 3, 2).double())
        UU = UU.float()
        subspace = UU[:, :, :, :shot_num - 1].permute(0, 1, 3, 2)
        projection = subspace.permute(0, 1, 3, 2).matmul(subspace.matmul(query_feat.permute(0, 1, 3, 2))).permute(0, 1, 3, 2)
        dist = torch.sum((query_feat - projection) ** 2, dim=-1).permute(0, 2, 1)
        logits = -dist
        if normalize:
            logits /= d
        disc_loss = 0
        if discriminative:
            subspace_metric = torch.norm(torch.matmul(subspace.unsqueeze(1), subspace.unsqueeze(2).transpose(-2, -1)), p='fro', dim=[-2, -1])
            mask = torch.eye(way_num).bool()
            subspace_metric = subspace_metric[:, ~mask]
            disc_loss = torch.sum(subspace_metric ** 2)
        return logits, disc_loss


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        residual = q
        q = self.w_qs(q).reshape(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).reshape(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).reshape(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().reshape(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().reshape(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().reshape(-1, len_v, d_v)
        output, attn, log_attn = self.attention(q, k, v)
        output = output.reshape(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().reshape(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output


class MCLLayer(nn.Module):

    def __init__(self, n_k, katz_factor, gamma, gamma2):
        super(MCLLayer, self).__init__()
        self.n_k = n_k
        self.gamma = gamma
        self.gamma2 = gamma2
        self.katz_factor = katz_factor
        self.inner_simi = Similarity(metric='cosine')
        self.criterion = nn.NLLLoss()

    def averaging_based_similarities(self, support_xf, support_y, query_xf, query_y):
        b, s, c, h, w = support_xf.shape
        q = query_xf.shape[1]
        support_xf = support_xf.view(b, self.n_way, self.k_shot, c, h, w).mean(2)
        support_xf = support_xf.view(b, self.n_way, c, h * w)
        S = self.inner_simi(support_xf, query_xf)
        M_q = S.shape[-2]
        M_s = S.shape[2] * S.shape[-1]
        S = S.permute(0, 1, 3, 2, 4).contiguous().view(b * q, M_q, M_s)
        return S

    def bipartite_katz_forward(self, support_xf, support_y, query_xf, query_y, similarity_f):
        katz_factor = self.katz_factor
        S = similarity_f(support_xf, support_y, query_xf, query_y)
        N_examples, M_q, M_s = S.shape
        St = S.transpose(-2, -1)
        device = S.device
        T_sq = torch.exp(self.gamma * (S - S.max(-1, keepdim=True)[0]))
        T_sq = T_sq / T_sq.sum(-1, keepdim=True)
        T_qs = torch.exp(self.gamma2 * (St - St.max(-1, keepdim=True)[0]))
        T_qs = T_qs / T_qs.sum(-1, keepdim=True)
        T = torch.cat([torch.cat([torch.zeros((N_examples, M_s, M_s), device=device), T_sq.transpose(-2, -1)], dim=-1), torch.cat([T_qs.transpose(-2, -1), torch.zeros((N_examples, M_q, M_q), device=device)], dim=-1)], dim=-2)
        katz = (torch.inverse(torch.eye(M_s + M_q, device=device)[None].repeat(N_examples, 1, 1) - katz_factor * T) - torch.eye(M_s + M_q, device=S.device)[None].repeat(N_examples, 1, 1)) @ torch.ones((N_examples, M_s + M_q, 1), device=device)
        partial_katz = katz.squeeze(-1)[:, :M_s] / katz.squeeze(-1)[:, :M_s].sum(-1, keepdim=True)
        predicts = partial_katz.view(N_examples, self.n_way, -1).sum(-1)
        return predicts

    def forward(self, support_xf, support_y, query_xf, query_y, n_way, k_shot):
        self.n_way = n_way
        self.k_shot = k_shot
        return self.bipartite_katz_forward(support_xf, support_y, query_xf, query_y, self.averaging_based_similarities)


class ProtoLayer_temperature(nn.Module):

    def __init__(self):
        super(ProtoLayer_temperature, self).__init__()

    def forward(self, query_feat, support_feat, way_num, shot_num, query_num, mode='cos_sim'):
        t, wq, c = query_feat.size()
        _, ws, _ = support_feat.size()
        query_feat = query_feat.reshape(t, way_num * query_num, c)
        support_feat = support_feat.reshape(t, way_num, shot_num, c)
        proto_feat = torch.mean(support_feat, dim=2)
        return {'euclidean': lambda x, y: -torch.sum(torch.pow(x.unsqueeze(2) - y.unsqueeze(1), 2), dim=3), 'cos_sim': lambda x, y: torch.matmul(F.normalize(x, p=2, dim=-1), torch.transpose(F.normalize(y, p=2, dim=-1), -1, -2))}[mode](query_feat, proto_feat)


class RelationLayer(nn.Module):

    def __init__(self, feat_dim=64, feat_height=3, feat_width=3):
        super(RelationLayer, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=3, padding=0), nn.BatchNorm2d(feat_dim, momentum=1, affine=True), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=0), nn.BatchNorm2d(feat_dim, momentum=1, affine=True), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.fc = nn.Sequential(nn.Linear(feat_dim * feat_height * feat_width, 8), nn.ReLU(inplace=True), nn.Linear(8, 1))

    def forward(self, x):
        out = self.layers(x)
        out = out.reshape(x.size(0), -1)
        out = self.fc(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AEAModule,
     lambda: ([], {'feat_dim': 4, 'scale_value': 1.0, 'from_value': 4, 'value_interval': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ANILLayer,
     lambda: ([], {'feat_dim': 4, 'hid_dim': 4, 'way_num': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicBlockMTL,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicBlockVariant,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicBlockWithoutResidual,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BatchNorm2d_fw,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Block,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (BottleneckBlock,
     lambda: ([], {'indim': 4, 'outdim': 4, 'half_res': 4}),
     lambda: ([torch.rand([4, 4, 2, 2])], {})),
    (CCA,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 4, 4, 4, 4])], {})),
    (Conv2dMtl,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv2d_fw,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv32F,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (Conv64F,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (Conv64F_MCL,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (ConvBlock,
     lambda: ([], {'in_c': 4, 'out_c': 4, 'k': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CyclicShift,
     lambda: ([], {'displacement': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Decoder,
     lambda: ([], {'feat_dim': 4, 'hid_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DistLinear,
     lambda: ([], {'in_channel': 4, 'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DistillKLLoss,
     lambda: ([], {'T': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (DistillLayer,
     lambda: ([], {'emb_func': 4, 'cls_classifier': 4, 'is_distill': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DropBlock,
     lambda: ([], {'block_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Encoder,
     lambda: ([], {'way_num': 4, 'shot_num': 4, 'feat_dim': 4, 'hid_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FeedForward,
     lambda: ([], {'dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FullyContextualEmbedding,
     lambda: ([], {'feat_dim': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (L2DistLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Linear_fw,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MTLBaseLearner,
     lambda: ([], {'ways': 4, 'z_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MatchingNetLayer,
     lambda: ([], {'feat_dim': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (MetaLinearLayer,
     lambda: ([], {'input_shape': [4, 4], 'num_filters': 4, 'use_bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MetaStepLossNetwork,
     lambda: ([], {'input_dim': 4, 'args': SimpleNamespace()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MultiHeadAttention,
     lambda: ([], {'n_head': 4, 'd_model': 4, 'd_k': 4, 'd_v': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (NegLayer,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NetworkBlock,
     lambda: ([], {'nb_layers': 1, 'in_planes': 4, 'out_planes': 4, 'block': torch.nn.ReLU, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PatchEmbed,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (PatchMerging,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'downscaling_factor': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PreNorm,
     lambda: ([], {'dim': 4, 'fn': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Predictor,
     lambda: ([], {'feat_dim': 4, 'hid_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (R2D2Embedding,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (ResNetMTLOfficial,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {})),
    (Residual,
     lambda: ([], {'fn': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SCRLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 640, 4, 4])], {})),
    (SELayer,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ScaledDotProductAttention,
     lambda: ([], {'temperature': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (SelfCorrelationComputation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SepConv4d,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4, 4])], {})),
    (Similarity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SimpleBlock,
     lambda: ([], {'indim': 4, 'outdim': 4, 'half_res': 4}),
     lambda: ([torch.rand([4, 4, 2, 2])], {})),
    (VisionTransformer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (distLinear,
     lambda: ([], {'indim': 4, 'outdim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

