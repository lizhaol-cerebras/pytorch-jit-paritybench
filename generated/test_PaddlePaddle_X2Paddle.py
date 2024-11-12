
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


import time


import logging


import torch


import copy


from torch import nn


from itertools import product


from inspect import isfunction


import torch.nn as nn


import torch.nn.functional as F


from torchvision.models import AlexNet


import torchvision


from torch.autograd import Variable


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torchvision import transforms


from torchvision import utils


import torch.optim as optim


import torchvision.transforms as standard_transforms


import math


import matplotlib.pyplot as plt


from torchvision import models


import torch.utils.model_zoo as model_zoo


from math import exp


from collections import OrderedDict


from torchvision.models.segmentation import deeplabv3_resnet50


from torch.nn import functional as F


from torchvision.models.segmentation import fcn_resnet50


from torchvision.models import Inception3


from torchvision.models import MNASNet


from torch.nn import Linear


from torch.nn import Conv2d


from torch.nn import BatchNorm1d


from torch.nn import BatchNorm2d


from torch.nn import PReLU


from torch.nn import ReLU


from torch.nn import Sigmoid


from torch.nn import AdaptiveAvgPool2d


from torch.nn import Sequential


from torch.nn import Module


from torchvision.models import MobileNetV2


import torch.utils.data


from matplotlib import cm


from scipy.interpolate import interp1d


from sklearn.metrics import roc_curve


from torch.nn.utils import clip_grad_norm_


from scipy.optimize import brentq


from typing import Union


from typing import List


from torch import optim


from torch.utils.tensorboard import SummaryWriter


from time import perf_counter as timer


import re


from scipy.io.wavfile import write


import random


from scipy.io.wavfile import read


from torch.nn import Conv1d


from torch.nn import ConvTranspose1d


from torch.nn import AvgPool1d


from torch.nn.utils import weight_norm


from torch.nn.utils import remove_weight_norm


from torch.nn.utils import spectral_norm


import warnings


import itertools


from torch.utils.data import DistributedSampler


import torch.multiprocessing as mp


from torch.distributed import init_process_group


from torch.nn.parallel import DistributedDataParallel


import matplotlib


import matplotlib.pylab as plt


from torchvision.models import resnet18


from torchvision.ops import RoIAlign


from typing import Dict


from abc import abstractmethod


from abc import ABC


import sklearn


import sklearn.svm


from scipy import linalg


from torch.nn.functional import adaptive_avg_pool2d


from torchvision.transforms import CenterCrop


from torchvision.transforms import Compose


from torchvision.transforms import Resize


from torchvision.transforms import ToTensor


from scipy.ndimage import zoom


from collections import namedtuple


from torchvision import models as tv


from enum import Enum


from torch.utils.data import IterableDataset


from torch.utils.data import ConcatDataset


from typing import Tuple


from typing import Optional


import torchvision.models as models


import abc


from torch.fft import rfftn


from torch.fft import irfftn


import collections


from functools import partial


import functools


from collections import defaultdict


import pandas as pd


import numbers


from torchvision.models import shufflenet_v2_x1_0


from torchvision.models import squeezenet1_0


import torch.distributed as dist


from torchvision import datasets


import torch.utils.data as data


import torch.backends.cudnn as cudnn


import torch.utils.checkpoint as checkpoint


from torch import optim as optim


from torchvision.models import vgg16


from typing import Callable


from torch import Tensor


from math import sqrt


import inspect


class BaseNet(nn.Module):
    """
    define Net
    """

    def __init__(self, config):
        super(BaseNet, self).__init__()
        self.config = copy.copy(config)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class ResBlock(nn.Module):

    def __init__(self, base_channels):
        super(ResBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect')
        self.conv2 = nn.Conv2d(in_channels=base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect')
        self.norm1 = nn.InstanceNorm2d(base_channels, affine=True)
        self.norm2 = nn.InstanceNorm2d(base_channels, affine=True)

    def forward(self, x):
        res = self.norm1(x)
        res = self.relu(res)
        res = self.conv1(res)
        res = self.norm2(res)
        res = self.relu(res)
        res = self.conv2(res)
        res1 = res + x
        return res1


class Net(nn.Module):

    def __init__(self, base_channels, depth_num):
        super(Net, self).__init__()
        self.input = nn.Conv2d(in_channels=3, out_channels=base_channels, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect')
        self.conv1 = nn.Conv2d(in_channels=base_channels, out_channels=2 * base_channels, kernel_size=3, stride=2, padding=1, bias=False, padding_mode='reflect')
        self.residual_layer = self.make_layer(ResBlock, depth_num, base_channels * 2)
        self.conv2 = nn.ConvTranspose2d(in_channels=2 * base_channels, out_channels=base_channels, kernel_size=2, stride=2, padding=0, bias=False)
        self.output = nn.Conv2d(in_channels=base_channels, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect')
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2.0 / n))

    def make_layer(self, block, num_of_layer, channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        res = self.relu(self.input(x))
        res = self.conv1(res)
        res = self.residual_layer(res)
        res = self.relu(res)
        res = self.conv2(res)
        res = self.output(res)
        res = res + x
        return res


class BasicTemporalModel(nn.Module):

    def __init__(self, in_channels=80, num_features=512, out_channels=4, time_window=2, num_blocks=2):
        super().__init__()
        self.in_channels = in_channels
        self.num_features = num_features
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.time_window = time_window
        self.conv1 = nn.Sequential(nn.ReplicationPad1d(1), nn.Conv1d(self.in_channels, self.num_features, kernel_size=3, bias=False), nn.BatchNorm1d(self.num_features), nn.ReLU(inplace=True), nn.Dropout(p=0.25))
        self._make_blocks()
        self.pad = nn.ReplicationPad1d(1)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.25)
        self.reduce = nn.Conv1d(self.num_features, self.num_features, kernel_size=2 * self.time_window + 1)
        self.out = nn.Linear(self.num_features, self.out_channels)

    def _make_blocks(self):
        layers_conv = []
        layers_bn = []
        for i in range(self.num_blocks):
            layers_conv.append(nn.Conv1d(self.num_features, self.num_features, kernel_size=3, bias=False))
            layers_bn.append(nn.BatchNorm1d(self.num_features))
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def forward(self, x):
        """
        Args:
        x - (B x T x J x C)
        """
        B, T, _, _ = x.shape
        x = x.view(B, T, -1).permute((0, 2, 1))
        x = self.conv1(x)
        for i in range(self.num_blocks):
            pre = x
            x = self.pad(x)
            x = self.layers_conv[i](x)
            x = self.layers_bn[i](x)
            x = self.drop(self.relu(x))
            x = pre + x
        x = self.relu(self.reduce(x))
        x = x.view(B, -1)
        x = self.out(x)
        if not self.training:
            x = torch.sigmoid(x)
        return x


class RefUnet(nn.Module):

    def __init__(self, in_ch, inc_ch):
        super(RefUnet, self).__init__()
        self.conv0 = nn.Conv2d(in_ch, inc_ch, 3, padding=1)
        self.conv1 = nn.Conv2d(inc_ch, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv_d4 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)
        self.conv_d3 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)
        self.conv_d2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)
        self.conv_d1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)
        self.conv_d0 = nn.Conv2d(64, 1, 3, padding=1)
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        hx = x
        hx = self.conv0(hx)
        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)
        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)
        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)
        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)
        hx5 = self.relu5(self.bn5(self.conv5(hx)))
        hx = self.upscore2(hx5)
        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx, hx4), 1))))
        hx = self.upscore2(d4)
        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx, hx3), 1))))
        hx = self.upscore2(d3)
        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx, hx2), 1))))
        hx = self.upscore2(d2)
        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx, hx1), 1))))
        residual = self.conv_d0(d1)
        return x + residual


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    kernel_size = np.asarray((3, 3))
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size
    full_padding = (upsampled_kernel_size - 1) // 2
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=full_padding, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
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


class BASNet(nn.Module):

    def __init__(self, n_channels, n_classes):
        super(BASNet, self).__init__()
        resnet = models.resnet34(pretrained=False)
        self.inconv = nn.Conv2d(n_channels, 64, 3, padding=1)
        self.inbn = nn.BatchNorm2d(64)
        self.inrelu = nn.ReLU(inplace=True)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.resb5_1 = BasicBlock(512, 512)
        self.resb5_2 = BasicBlock(512, 512)
        self.resb5_3 = BasicBlock(512, 512)
        self.pool5 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.resb6_1 = BasicBlock(512, 512)
        self.resb6_2 = BasicBlock(512, 512)
        self.resb6_3 = BasicBlock(512, 512)
        self.convbg_1 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_1 = nn.BatchNorm2d(512)
        self.relubg_1 = nn.ReLU(inplace=True)
        self.convbg_m = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_m = nn.BatchNorm2d(512)
        self.relubg_m = nn.ReLU(inplace=True)
        self.convbg_2 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_2 = nn.BatchNorm2d(512)
        self.relubg_2 = nn.ReLU(inplace=True)
        self.conv6d_1 = nn.Conv2d(1024, 512, 3, padding=1)
        self.bn6d_1 = nn.BatchNorm2d(512)
        self.relu6d_1 = nn.ReLU(inplace=True)
        self.conv6d_m = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bn6d_m = nn.BatchNorm2d(512)
        self.relu6d_m = nn.ReLU(inplace=True)
        self.conv6d_2 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bn6d_2 = nn.BatchNorm2d(512)
        self.relu6d_2 = nn.ReLU(inplace=True)
        self.conv5d_1 = nn.Conv2d(1024, 512, 3, padding=1)
        self.bn5d_1 = nn.BatchNorm2d(512)
        self.relu5d_1 = nn.ReLU(inplace=True)
        self.conv5d_m = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5d_m = nn.BatchNorm2d(512)
        self.relu5d_m = nn.ReLU(inplace=True)
        self.conv5d_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5d_2 = nn.BatchNorm2d(512)
        self.relu5d_2 = nn.ReLU(inplace=True)
        self.conv4d_1 = nn.Conv2d(1024, 512, 3, padding=1)
        self.bn4d_1 = nn.BatchNorm2d(512)
        self.relu4d_1 = nn.ReLU(inplace=True)
        self.conv4d_m = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4d_m = nn.BatchNorm2d(512)
        self.relu4d_m = nn.ReLU(inplace=True)
        self.conv4d_2 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn4d_2 = nn.BatchNorm2d(256)
        self.relu4d_2 = nn.ReLU(inplace=True)
        self.conv3d_1 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn3d_1 = nn.BatchNorm2d(256)
        self.relu3d_1 = nn.ReLU(inplace=True)
        self.conv3d_m = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3d_m = nn.BatchNorm2d(256)
        self.relu3d_m = nn.ReLU(inplace=True)
        self.conv3d_2 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn3d_2 = nn.BatchNorm2d(128)
        self.relu3d_2 = nn.ReLU(inplace=True)
        self.conv2d_1 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn2d_1 = nn.BatchNorm2d(128)
        self.relu2d_1 = nn.ReLU(inplace=True)
        self.conv2d_m = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2d_m = nn.BatchNorm2d(128)
        self.relu2d_m = nn.ReLU(inplace=True)
        self.conv2d_2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn2d_2 = nn.BatchNorm2d(64)
        self.relu2d_2 = nn.ReLU(inplace=True)
        self.conv1d_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn1d_1 = nn.BatchNorm2d(64)
        self.relu1d_1 = nn.ReLU(inplace=True)
        self.conv1d_m = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1d_m = nn.BatchNorm2d(64)
        self.relu1d_m = nn.ReLU(inplace=True)
        self.conv1d_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1d_2 = nn.BatchNorm2d(64)
        self.relu1d_2 = nn.ReLU(inplace=True)
        self.upscore6 = nn.Upsample(scale_factor=32, mode='bilinear')
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.outconvb = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv6 = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv5 = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv4 = nn.Conv2d(256, 1, 3, padding=1)
        self.outconv3 = nn.Conv2d(128, 1, 3, padding=1)
        self.outconv2 = nn.Conv2d(64, 1, 3, padding=1)
        self.outconv1 = nn.Conv2d(64, 1, 3, padding=1)
        self.refunet = RefUnet(1, 64)

    def forward(self, x):
        hx = x
        hx = self.inconv(hx)
        hx = self.inbn(hx)
        hx = self.inrelu(hx)
        h1 = self.encoder1(hx)
        h2 = self.encoder2(h1)
        h3 = self.encoder3(h2)
        h4 = self.encoder4(h3)
        hx = self.pool4(h4)
        hx = self.resb5_1(hx)
        hx = self.resb5_2(hx)
        h5 = self.resb5_3(hx)
        hx = self.pool5(h5)
        hx = self.resb6_1(hx)
        hx = self.resb6_2(hx)
        h6 = self.resb6_3(hx)
        hx = self.relubg_1(self.bnbg_1(self.convbg_1(h6)))
        hx = self.relubg_m(self.bnbg_m(self.convbg_m(hx)))
        hbg = self.relubg_2(self.bnbg_2(self.convbg_2(hx)))
        hx = self.relu6d_1(self.bn6d_1(self.conv6d_1(torch.cat((hbg, h6), 1))))
        hx = self.relu6d_m(self.bn6d_m(self.conv6d_m(hx)))
        hd6 = self.relu6d_2(self.bn6d_2(self.conv6d_2(hx)))
        hx = self.upscore2(hd6)
        hx = self.relu5d_1(self.bn5d_1(self.conv5d_1(torch.cat((hx, h5), 1))))
        hx = self.relu5d_m(self.bn5d_m(self.conv5d_m(hx)))
        hd5 = self.relu5d_2(self.bn5d_2(self.conv5d_2(hx)))
        hx = self.upscore2(hd5)
        hx = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((hx, h4), 1))))
        hx = self.relu4d_m(self.bn4d_m(self.conv4d_m(hx)))
        hd4 = self.relu4d_2(self.bn4d_2(self.conv4d_2(hx)))
        hx = self.upscore2(hd4)
        hx = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((hx, h3), 1))))
        hx = self.relu3d_m(self.bn3d_m(self.conv3d_m(hx)))
        hd3 = self.relu3d_2(self.bn3d_2(self.conv3d_2(hx)))
        hx = self.upscore2(hd3)
        hx = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((hx, h2), 1))))
        hx = self.relu2d_m(self.bn2d_m(self.conv2d_m(hx)))
        hd2 = self.relu2d_2(self.bn2d_2(self.conv2d_2(hx)))
        hx = self.upscore2(hd2)
        hx = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((hx, h1), 1))))
        hx = self.relu1d_m(self.bn1d_m(self.conv1d_m(hx)))
        hd1 = self.relu1d_2(self.bn1d_2(self.conv1d_2(hx)))
        db = self.outconvb(hbg)
        db = self.upscore6(db)
        d6 = self.outconv6(hd6)
        d6 = self.upscore6(d6)
        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5)
        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4)
        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3)
        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2)
        d1 = self.outconv1(hd1)
        dout = self.refunet(d1)
        return F.sigmoid(dout), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6), F.sigmoid(db)


class BasicBlockDe(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockDe, self).__init__()
        self.convRes = conv3x3(inplanes, planes, stride)
        self.bnRes = nn.BatchNorm2d(planes)
        self.reluRes = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = self.convRes(x)
        residual = self.bnRes(residual)
        residual = self.reluRes(residual)
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
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
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


def _iou(pred, target, size_average=True):
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0, b):
        Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
        Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
        IoU1 = Iand1 / Ior1
        IoU = IoU + (1 - IoU1)
    return IoU / b


class IOU(torch.nn.Module):

    def __init__(self, size_average=True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        return _iou(pred, target, self.size_average)


def add_tensor_function(func):
    setattr(paddle.Tensor, func.__name__, func)


class SSIM(torch.nn.Module):
    """SSIM. Modified from:
    https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
    """

    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.register_buffer('window', self._create_window(window_size, self.channel))

    def forward(self, img1, img2):
        assert len(img1.shape) == 4
        channel = img1.size()[1]
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window.expand(channel, 1, window_size, window_size).contiguous()

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        if size_average:
            return ssim_map.mean()
        return ssim_map.mean(1).mean(1).mean(1)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        return


def _logssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_map = (ssim_map - torch.min(ssim_map)) / (torch.max(ssim_map) - torch.min(ssim_map))
    ssim_map = -torch.log(ssim_map + 1e-08)
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class LOGSSIM(torch.nn.Module):

    def __init__(self, window_size=11, size_average=True):
        super(LOGSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _logssim(img1, img2, window, self.window_size, channel, self.size_average)


class HSigmoid(nn.Module):

    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class HSwish(nn.Module):

    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):

    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(in_size // reduction), nn.ReLU(inplace=True), nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(in_size), nn.Sigmoid())

    def forward(self, x):
        return x * self.se(self.pool(x))


class Block(nn.Module):

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule
        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(out_size))

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class CBNModule(nn.Module):

    def __init__(self, inchannel, outchannel=24, kernel_size=3, stride=1, padding=0, bias=False):
        super(CBNModule, self).__init__()
        self.conv = nn.Conv2d(inchannel, outchannel, kernel_size, stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(outchannel)
        self.act = HSwish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class CBAModule(nn.Module):

    def __init__(self, in_channels, out_channels=24, kernel_size=3, stride=1, padding=0, bias=False):
        super(CBAModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class UpModule(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, bias=False, mode='UCBA'):
        super(UpModule, self).__init__()
        self.mode = mode
        if self.mode == 'UCBA':
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            self.conv = CBAModule(in_channels, out_channels, 3, padding=1, bias=bias)
        elif self.mode == 'DeconvBN':
            self.dconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, bias=bias)
            self.bn = nn.BatchNorm2d(out_channels)
        elif self.mode == 'DeCBA':
            self.dconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, bias=bias)
            self.conv = CBAModule(out_channels, out_channels, 3, padding=1, bias=bias)
        else:
            raise RuntimeError(f'Unsupport mode: {mode}')

    def forward(self, x):
        if self.mode == 'UCBA':
            return self.conv(self.up(x))
        elif self.mode == 'DeconvBN':
            return F.relu(self.bn(self.dconv(x)))
        elif self.mode == 'DeCBA':
            return self.conv(self.dconv(x))


class ContextModule(nn.Module):

    def __init__(self, in_channels):
        super(ContextModule, self).__init__()
        block_wide = in_channels // 4
        self.inconv = CBAModule(in_channels, block_wide, 3, 1, padding=1)
        self.upconv = CBAModule(block_wide, block_wide, 3, 1, padding=1)
        self.downconv = CBAModule(block_wide, block_wide, 3, 1, padding=1)
        self.downconv2 = CBAModule(block_wide, block_wide, 3, 1, padding=1)

    def forward(self, x):
        x = self.inconv(x)
        up = self.upconv(x)
        down = self.downconv(x)
        down = self.downconv2(down)
        return torch.cat([up, down], dim=1)


class DetectModule(nn.Module):

    def __init__(self, in_channels):
        super(DetectModule, self).__init__()
        self.upconv = CBAModule(in_channels, in_channels // 2, 3, 1, padding=1)
        self.context = ContextModule(in_channels)

    def forward(self, x):
        up = self.upconv(x)
        down = self.context(x)
        return torch.cat([up, down], dim=1)


class HeadModule(nn.Module):

    def __init__(self, in_channels, out_channels, has_ext=False):
        super(HeadModule, self).__init__()
        self.head = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.has_ext = has_ext
        if has_ext:
            self.ext = CBAModule(in_channels, in_channels, kernel_size=3, padding=1, bias=False)

    def init_normal(self, std, bias):
        nn.init.normal_(self.head.weight, std=std)
        nn.init.constant_(self.head.bias, bias)

    def forward(self, x):
        if self.has_ext:
            x = self.ext(x)
        return self.head(x)


_MODEL_URL_DOMAIN = 'http://zifuture.com:1000/fs/public_models'


_MODEL_URL_SMALL = 'mbv3small-09ace125.pth'


class Mbv3SmallFast(nn.Module):

    def __init__(self):
        super(Mbv3SmallFast, self).__init__()
        self.keep = [0, 2, 7]
        self.uplayer_shape = [16, 24, 48]
        self.output_channels = 96
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = nn.ReLU(inplace=True)
        self.bneck = nn.Sequential(Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 2), Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2), Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1), Block(5, 24, 96, 40, nn.ReLU(inplace=True), SeModule(40), 2), Block(5, 40, 240, 40, nn.ReLU(inplace=True), SeModule(40), 1), Block(5, 40, 240, 40, nn.ReLU(inplace=True), SeModule(40), 1), Block(5, 40, 120, 48, nn.ReLU(inplace=True), SeModule(48), 1), Block(5, 48, 144, 48, nn.ReLU(inplace=True), SeModule(48), 1), Block(5, 48, 288, 96, nn.ReLU(inplace=True), SeModule(96), 2))

    def load_pretrain(self):
        checkpoint = model_zoo.load_url(f'{_MODEL_URL_DOMAIN}/{_MODEL_URL_SMALL}')
        self.load_state_dict(checkpoint, strict=False)

    def forward(self, x):
        x = self.hs1(self.bn1(self.conv1(x)))
        outs = []
        for index, item in enumerate(self.bneck):
            x = item(x)
            if index in self.keep:
                outs.append(x)
        outs.append(x)
        return outs


class DBFace(nn.Module):

    def __init__(self, has_landmark=True, wide=64, has_ext=True, upmode='UCBA'):
        super(DBFace, self).__init__()
        self.has_landmark = has_landmark
        self.bb = Mbv3SmallFast()
        c0, c1, c2 = self.bb.uplayer_shape
        self.conv3 = CBAModule(self.bb.output_channels, wide, kernel_size=1, stride=1, padding=0, bias=False)
        self.connect0 = CBAModule(c0, wide, kernel_size=1)
        self.connect1 = CBAModule(c1, wide, kernel_size=1)
        self.connect2 = CBAModule(c2, wide, kernel_size=1)
        self.up0 = UpModule(wide, wide, kernel_size=2, stride=2, mode=upmode)
        self.up1 = UpModule(wide, wide, kernel_size=2, stride=2, mode=upmode)
        self.up2 = UpModule(wide, wide, kernel_size=2, stride=2, mode=upmode)
        self.detect = DetectModule(wide)
        self.center = HeadModule(wide, 1, has_ext=has_ext)
        self.box = HeadModule(wide, 4, has_ext=has_ext)
        if self.has_landmark:
            self.landmark = HeadModule(wide, 10, has_ext=has_ext)

    def init_weights(self):
        prob = 0.01
        d = -np.log((1 - prob) / prob)
        self.bb.load_pretrain()
        self.center.init_normal(0.001, d)
        self.box.init_normal(0.001, 0)
        if self.has_landmark:
            self.landmark.init_normal(0.001, 0)

    def load(self, file):
        checkpoint = torch.load(file, map_location='cpu')
        self.load_state_dict(checkpoint)

    def forward(self, x):
        s4, s8, s16, s32 = self.bb(x)
        s32 = self.conv3(s32)
        s16 = self.up0(s32) + self.connect2(s16)
        s8 = self.up1(s16) + self.connect1(s8)
        s4 = self.up2(s8) + self.connect0(s4)
        x = self.detect(s4)
        center = self.center(x)
        box = self.box(x)
        center = center.sigmoid()
        box = torch.exp(box)
        if self.has_landmark:
            landmark = self.landmark(x)
            return center, box, landmark
        return center, box


class MeanShift(nn.Conv2d):

    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.404), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class Upsampler(nn.Sequential):

    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if scale & scale - 1 == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)


class EDSR(nn.Module):

    def __init__(self, n_resblocks=32, n_feats=256, scale=[2], rgb_range=255, n_colors=3, res_scale=0.1, conv=default_conv):
        super(EDSR, self).__init__()
        n_resblocks = n_resblocks
        n_feats = n_feats
        kernel_size = 3
        scale = scale[0]
        act = nn.ReLU(True)
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)
        m_head = [conv(n_colors, n_feats, kernel_size)]
        m_body = [ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale) for _ in range(n_resblocks)]
        m_body.append(conv(n_feats, n_feats, kernel_size))
        m_tail = [Upsampler(conv, scale, n_feats, act=False), conv(n_feats, n_colors, kernel_size)]
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))


class GruModel(nn.Module):
    """
    tsn
    """

    def __init__(self):
        super(GruModel, self).__init__()
        self.query_model = nn.GRU(768, 256, batch_first=True, bidirectional=True)
        self.query_fc = nn.Linear(512, 256)

    def forward(self, query_input):
        """
        :param input:
        :param title_input:
        :return:
        """
        o1, h_query = self.query_model(query_input)
        new_query = []
        h_query_concat = torch.cat([h_query[0], h_query[1]], 1)
        new_query.append(h_query_concat[0])
        output_query = self.query_fc(torch.stack(new_query))
        output_query = F.normalize(output_query, p=2, dim=1)
        return output_query


class L2Norm(Module):

    def forward(self, input):
        return F.normalize(input)


class Flatten(Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class Conv_block(Module):

    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Linear_block(Module):

    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(Module):

    def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = Conv_block(c1_in, out_c=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Module):

    def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for i in range(num_block):
            c1_tuple = c1[i]
            c2_tuple = c2[i]
            c3_tuple = c3[i]
            modules.append(Depth_Wise(c1_tuple, c2_tuple, c3_tuple, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class SEModule(Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.bn1 = BatchNorm2d(channels // reduction)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.bn2 = BatchNorm2d(channels)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.sigmoid(x)
        return module_input * x


class Depth_Wise_SE(Module):

    def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, se_reduct=8):
        super(Depth_Wise_SE, self).__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = Conv_block(c1_in, out_c=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
        self.se_module = SEModule(c3_out, se_reduct)

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            x = self.se_module(x)
            output = short_cut + x
        else:
            output = x
        return output


class ResidualSE(Module):

    def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1), se_reduct=4):
        super(ResidualSE, self).__init__()
        modules = []
        for i in range(num_block):
            c1_tuple = c1[i]
            c2_tuple = c2[i]
            c3_tuple = c3[i]
            if i == num_block - 1:
                modules.append(Depth_Wise_SE(c1_tuple, c2_tuple, c3_tuple, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups, se_reduct=se_reduct))
            else:
                modules.append(Depth_Wise(c1_tuple, c2_tuple, c3_tuple, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class MiniFASNet(Module):

    def __init__(self, keep, embedding_size, conv6_kernel=(7, 7), drop_p=0.0, num_classes=3, img_channel=3):
        super(MiniFASNet, self).__init__()
        self.embedding_size = embedding_size
        self.conv1 = Conv_block(img_channel, keep[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(keep[0], keep[1], kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=keep[1])
        c1 = [(keep[1], keep[2])]
        c2 = [(keep[2], keep[3])]
        c3 = [(keep[3], keep[4])]
        self.conv_23 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[3])
        c1 = [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])]
        c2 = [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])]
        c3 = [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])]
        self.conv_3 = Residual(c1, c2, c3, num_block=4, groups=keep[4], kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        c1 = [(keep[16], keep[17])]
        c2 = [(keep[17], keep[18])]
        c3 = [(keep[18], keep[19])]
        self.conv_34 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[19])
        c1 = [(keep[19], keep[20]), (keep[22], keep[23]), (keep[25], keep[26]), (keep[28], keep[29]), (keep[31], keep[32]), (keep[34], keep[35])]
        c2 = [(keep[20], keep[21]), (keep[23], keep[24]), (keep[26], keep[27]), (keep[29], keep[30]), (keep[32], keep[33]), (keep[35], keep[36])]
        c3 = [(keep[21], keep[22]), (keep[24], keep[25]), (keep[27], keep[28]), (keep[30], keep[31]), (keep[33], keep[34]), (keep[36], keep[37])]
        self.conv_4 = Residual(c1, c2, c3, num_block=6, groups=keep[19], kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        c1 = [(keep[37], keep[38])]
        c2 = [(keep[38], keep[39])]
        c3 = [(keep[39], keep[40])]
        self.conv_45 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[40])
        c1 = [(keep[40], keep[41]), (keep[43], keep[44])]
        c2 = [(keep[41], keep[42]), (keep[44], keep[45])]
        c3 = [(keep[42], keep[43]), (keep[45], keep[46])]
        self.conv_5 = Residual(c1, c2, c3, num_block=2, groups=keep[40], kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(keep[46], keep[47], kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(keep[47], keep[48], groups=keep[48], kernel=conv6_kernel, stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
        self.drop = torch.nn.Dropout(p=drop_p)
        self.prob = Linear(embedding_size, num_classes, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        if self.embedding_size != 512:
            out = self.linear(out)
        out = self.bn(out)
        out = self.drop(out)
        out = self.prob(out)
        return out


class MiniFASNetSE(MiniFASNet):

    def __init__(self, keep, embedding_size, conv6_kernel=(7, 7), drop_p=0.75, num_classes=4, img_channel=3):
        super(MiniFASNetSE, self).__init__(keep=keep, embedding_size=embedding_size, conv6_kernel=conv6_kernel, drop_p=drop_p, num_classes=num_classes, img_channel=img_channel)
        c1 = [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])]
        c2 = [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])]
        c3 = [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])]
        self.conv_3 = ResidualSE(c1, c2, c3, num_block=4, groups=keep[4], kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        c1 = [(keep[19], keep[20]), (keep[22], keep[23]), (keep[25], keep[26]), (keep[28], keep[29]), (keep[31], keep[32]), (keep[34], keep[35])]
        c2 = [(keep[20], keep[21]), (keep[23], keep[24]), (keep[26], keep[27]), (keep[29], keep[30]), (keep[32], keep[33]), (keep[35], keep[36])]
        c3 = [(keep[21], keep[22]), (keep[24], keep[25]), (keep[27], keep[28]), (keep[30], keep[31]), (keep[33], keep[34]), (keep[36], keep[37])]
        self.conv_4 = ResidualSE(c1, c2, c3, num_block=6, groups=keep[19], kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        c1 = [(keep[40], keep[41]), (keep[43], keep[44])]
        c2 = [(keep[41], keep[42]), (keep[44], keep[45])]
        c3 = [(keep[42], keep[43]), (keep[45], keep[46])]
        self.conv_5 = ResidualSE(c1, c2, c3, num_block=2, groups=keep[40], kernel=(3, 3), stride=(1, 1), padding=(1, 1))


class MobileV2_Residual(nn.Module):

    def __init__(self, inp, oup, stride, expanse_ratio, dilation=1):
        super(MobileV2_Residual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(inp * expanse_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        pad = dilation
        if expanse_ratio == 1:
            self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class hourglass2D(nn.Module):

    def __init__(self, in_channels):
        super(hourglass2D, self).__init__()
        self.expanse_ratio = 2
        self.conv1 = MobileV2_Residual(in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)
        self.conv2 = MobileV2_Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)
        self.conv3 = MobileV2_Residual(in_channels * 2, in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)
        self.conv4 = MobileV2_Residual(in_channels * 4, in_channels * 4, stride=1, expanse_ratio=self.expanse_ratio)
        self.conv5 = nn.Sequential(nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm2d(in_channels * 2))
        self.conv6 = nn.Sequential(nn.ConvTranspose2d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm2d(in_channels))
        self.redir1 = MobileV2_Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
        self.redir2 = MobileV2_Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return conv6


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False), nn.BatchNorm2d(out_channels))


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)


def convbn_dws(inp, oup, kernel_size, stride, pad, dilation, second_relu=True):
    if second_relu:
        return nn.Sequential(nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, groups=inp, bias=False), nn.BatchNorm2d(inp), nn.ReLU6(inplace=True), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=False))
    else:
        return nn.Sequential(nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, groups=inp, bias=False), nn.BatchNorm2d(inp), nn.ReLU6(inplace=True), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))


class MobileV1_Residual(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(MobileV1_Residual, self).__init__()
        self.stride = stride
        self.downsample = downsample
        self.conv1 = convbn_dws(inplanes, planes, 3, stride, pad, dilation)
        self.conv2 = convbn_dws(planes, planes, 3, 1, pad, dilation, second_relu=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class feature_extraction(nn.Module):

    def __init__(self, add_relus=False):
        super(feature_extraction, self).__init__()
        self.expanse_ratio = 3
        self.inplanes = 32
        if add_relus:
            self.firstconv = nn.Sequential(MobileV2_Residual(3, 32, 2, self.expanse_ratio), nn.ReLU(inplace=True), MobileV2_Residual(32, 32, 1, self.expanse_ratio), nn.ReLU(inplace=True), MobileV2_Residual(32, 32, 1, self.expanse_ratio), nn.ReLU(inplace=True))
        else:
            self.firstconv = nn.Sequential(MobileV2_Residual(3, 32, 2, self.expanse_ratio), MobileV2_Residual(32, 32, 1, self.expanse_ratio), MobileV2_Residual(32, 32, 1, self.expanse_ratio))
        self.layer1 = self._make_layer(MobileV1_Residual, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(MobileV1_Residual, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(MobileV1_Residual, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(MobileV1_Residual, 128, 3, 1, 1, 2)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes))
        layers = [block(self.inplanes, planes, stride, downsample, pad, dilation)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        feature_volume = torch.cat((l2, l3, l4), dim=1)
        return feature_volume


def interweave_tensors(refimg_fea, targetimg_fea):
    B, C, H, W = refimg_fea.shape
    interwoven_features = torch.cat((refimg_fea, targetimg_fea), 1)
    interwoven_features = torch.reshape(interwoven_features, (B, 2, C, H, W)).permute(0, 2, 1, 3, 4)
    interwoven_features = interwoven_features.reshape([B, -1, H, W])
    return interwoven_features


class MSNet2D(nn.Module):

    def __init__(self, maxdisp):
        super(MSNet2D, self).__init__()
        self.maxdisp = maxdisp
        self.num_groups = 1
        self.volume_size = 48
        self.hg_size = 48
        self.dres_expanse_ratio = 3
        self.feature_extraction = feature_extraction(add_relus=True)
        self.preconv11 = nn.Sequential(convbn(320, 256, 1, 1, 0, 1), nn.ReLU(inplace=True), convbn(256, 128, 1, 1, 0, 1), nn.ReLU(inplace=True), convbn(128, 64, 1, 1, 0, 1), nn.ReLU(inplace=True), nn.Conv2d(64, 32, 1, 1, 0, 1))
        self.conv3d = nn.Sequential(nn.Conv3d(1, 16, kernel_size=(8, 3, 3), stride=[8, 1, 1], padding=[0, 1, 1]), nn.BatchNorm3d(16), nn.ReLU(), nn.Conv3d(16, 32, kernel_size=(4, 3, 3), stride=[4, 1, 1], padding=[0, 1, 1]), nn.BatchNorm3d(32), nn.ReLU(), nn.Conv3d(32, 16, kernel_size=(2, 3, 3), stride=[2, 1, 1], padding=[0, 1, 1]), nn.BatchNorm3d(16), nn.ReLU())
        self.volume11 = nn.Sequential(convbn(16, 1, 1, 1, 0, 1), nn.ReLU(inplace=True))
        self.dres0 = nn.Sequential(MobileV2_Residual(self.volume_size, self.hg_size, 1, self.dres_expanse_ratio), nn.ReLU(inplace=True), MobileV2_Residual(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio), nn.ReLU(inplace=True))
        self.dres1 = nn.Sequential(MobileV2_Residual(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio), nn.ReLU(inplace=True), MobileV2_Residual(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio))
        self.encoder_decoder1 = hourglass2D(self.hg_size)
        self.encoder_decoder2 = hourglass2D(self.hg_size)
        self.encoder_decoder3 = hourglass2D(self.hg_size)
        self.classif0 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1, bias=False, dilation=1))
        self.classif1 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1, bias=False, dilation=1))
        self.classif2 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1, bias=False, dilation=1))
        self.classif3 = nn.Sequential(convbn(self.hg_size, self.hg_size, 3, 1, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(self.hg_size, self.hg_size, kernel_size=3, padding=1, stride=1, bias=False, dilation=1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, L, R):
        features_L = self.feature_extraction(L)
        features_R = self.feature_extraction(R)
        featL = self.preconv11(features_L)
        featR = self.preconv11(features_R)
        B, C, H, W = featL.shape
        volume_list = list()
        for i in range(self.volume_size):
            if i > 0:
                x = interweave_tensors(featL[:, :, :, i:], featR[:, :, :, :-i])
            else:
                x = interweave_tensors(featL, featR)
            x = torch.unsqueeze(x, 1)
            x = self.conv3d(x)
            x = torch.squeeze(x, 2)
            x = self.volume11(x)
            x = torch.nn.functional.pad(x, (i, 0))
            volume_list.append(x)
        volume = torch.cat(volume_list, 1)
        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0
        out1 = self.encoder_decoder1(cost0)
        out2 = self.encoder_decoder2(out1)
        out3 = self.encoder_decoder3(out2)
        if self.training:
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)
            cost3 = self.classif3(out3)
            cost0 = torch.unsqueeze(cost0, 1)
            cost0 = F.interpolate(cost0, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, self.maxdisp)
            cost1 = torch.unsqueeze(cost1, 1)
            cost1 = F.interpolate(cost1, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, self.maxdisp)
            cost2 = torch.unsqueeze(cost2, 1)
            cost2 = F.interpolate(cost2, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)
            cost3 = torch.unsqueeze(cost3, 1)
            cost3 = F.interpolate(cost3, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)
            return [pred0, pred1, pred2, pred3]
        else:
            cost3 = self.classif3(out3)
            cost3 = torch.unsqueeze(cost3, 1)
            cost3 = F.interpolate(cost3, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)
            return [pred3]


class MobileV2_Residual_3D(nn.Module):

    def __init__(self, inp, oup, stride, expanse_ratio):
        super(MobileV2_Residual_3D, self).__init__()
        self.stride = stride
        hidden_dim = round(inp * expanse_ratio)
        self.use_res_connect = self.stride == (1, 1, 1) and inp == oup
        if expanse_ratio == 1:
            self.conv = nn.Sequential(nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm3d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm3d(oup))
        else:
            self.conv = nn.Sequential(nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm3d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm3d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm3d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class hourglass3D(nn.Module):

    def __init__(self, in_channels):
        super(hourglass3D, self).__init__()
        self.expanse_ratio = 2
        self.conv1 = MobileV2_Residual_3D(in_channels, in_channels * 2, 2, self.expanse_ratio)
        self.conv2 = MobileV2_Residual_3D(in_channels * 2, in_channels * 2, 1, self.expanse_ratio)
        self.conv3 = MobileV2_Residual_3D(in_channels * 2, in_channels * 4, 2, self.expanse_ratio)
        self.conv4 = MobileV2_Residual_3D(in_channels * 4, in_channels * 4, 1, self.expanse_ratio)
        self.conv5 = nn.Sequential(nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm3d(in_channels * 2))
        self.conv6 = nn.Sequential(nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm3d(in_channels))
        self.redir1 = MobileV2_Residual_3D(in_channels, in_channels, 1, self.expanse_ratio)
        self.redir2 = MobileV2_Residual_3D(in_channels * 2, in_channels * 2, 1, self.expanse_ratio)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return conv6


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = torch.zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i], num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    return volume


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad, bias=False), nn.BatchNorm3d(out_channels))


class MSNet3D(nn.Module):

    def __init__(self, maxdisp):
        super(MSNet3D, self).__init__()
        self.maxdisp = maxdisp
        self.hourglass_size = 32
        self.dres_expanse_ratio = 3
        self.num_groups = 40
        self.feature_extraction = feature_extraction()
        self.dres0 = nn.Sequential(MobileV2_Residual_3D(self.num_groups, self.hourglass_size, 1, self.dres_expanse_ratio), MobileV2_Residual_3D(self.hourglass_size, self.hourglass_size, 1, self.dres_expanse_ratio))
        self.dres1 = nn.Sequential(MobileV2_Residual_3D(self.hourglass_size, self.hourglass_size, 1, self.dres_expanse_ratio), MobileV2_Residual_3D(self.hourglass_size, self.hourglass_size, 1, self.dres_expanse_ratio))
        self.encoder_decoder1 = hourglass3D(self.hourglass_size)
        self.encoder_decoder2 = hourglass3D(self.hourglass_size)
        self.encoder_decoder3 = hourglass3D(self.hourglass_size)
        self.classif0 = nn.Sequential(convbn_3d(self.hourglass_size, self.hourglass_size, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(self.hourglass_size, 1, kernel_size=3, padding=1, stride=1, bias=False, dilation=1))
        self.classif1 = nn.Sequential(convbn_3d(self.hourglass_size, self.hourglass_size, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(self.hourglass_size, 1, kernel_size=3, padding=1, stride=1, bias=False, dilation=1))
        self.classif2 = nn.Sequential(convbn_3d(self.hourglass_size, self.hourglass_size, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(self.hourglass_size, 1, kernel_size=3, padding=1, stride=1, bias=False, dilation=1))
        self.classif3 = nn.Sequential(convbn_3d(self.hourglass_size, self.hourglass_size, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(self.hourglass_size, 1, kernel_size=3, padding=1, stride=1, bias=False, dilation=1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, L, R):
        features_left = self.feature_extraction(L)
        features_right = self.feature_extraction(R)
        volume = build_gwc_volume(features_left, features_right, self.maxdisp // 4, self.num_groups)
        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0
        out1 = self.encoder_decoder1(cost0)
        out2 = self.encoder_decoder2(out1)
        out3 = self.encoder_decoder3(out2)
        if self.training:
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)
            cost3 = self.classif3(out3)
            cost0 = F.interpolate(cost0, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, self.maxdisp)
            cost1 = F.interpolate(cost1, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, self.maxdisp)
            cost2 = F.interpolate(cost2, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)
            cost3 = F.interpolate(cost3, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)
            return [pred0, pred1, pred2, pred3]
        else:
            cost3 = self.classif3(out3)
            cost3 = F.interpolate(cost3, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)
            return [pred3]


mel_n_channels = 40


model_embedding_size = 256


model_hidden_size = 256


model_num_layers = 3


class SpeakerEncoder(nn.Module):

    def __init__(self, device, loss_device):
        super().__init__()
        self.loss_device = loss_device
        self.lstm = nn.LSTM(input_size=mel_n_channels, hidden_size=model_hidden_size, num_layers=model_num_layers, batch_first=True)
        self.linear = nn.Linear(in_features=model_hidden_size, out_features=model_embedding_size)
        self.relu = torch.nn.ReLU()
        self.similarity_weight = nn.Parameter(torch.tensor([10.0]))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.0]))
        self.loss_fn = nn.CrossEntropyLoss()

    def do_gradient_ops(self):
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01
        clip_grad_norm_(self.parameters(), 3, norm_type=2)

    def forward(self, utterances, hidden_init=None):
        """
        Computes the embeddings of a batch of utterance spectrograms.

        :param utterances: batch of mel-scale filterbanks of same duration as a tensor of shape
        (batch_size, n_frames, n_channels)
        :param hidden_init: initial hidden state of the LSTM as a tensor of shape (num_layers,
        batch_size, hidden_size). Will default to a tensor of zeros if None.
        :return: the embeddings as a tensor of shape (batch_size, embedding_size)
        """
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-05)
        return embeds

    def similarity_matrix(self, embeds):
        """
        Computes the similarity matrix according the section 2.1 of GE2E.

        :param embeds: the embeddings as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, embedding_size)
        :return: the similarity matrix as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, speakers_per_batch)
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / (torch.norm(centroids_incl, dim=2, keepdim=True) + 1e-05)
        centroids_excl = torch.sum(embeds, dim=1, keepdim=True) - embeds
        centroids_excl /= utterances_per_speaker - 1
        centroids_excl = centroids_excl.clone() / (torch.norm(centroids_excl, dim=2, keepdim=True) + 1e-05)
        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker, speakers_per_batch)
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        return sim_matrix

    def loss(self, embeds):
        """
        Computes the softmax loss according the section 2.1 of GE2E.

        :param embeds: the embeddings as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, embedding_size)
        :return: the loss and the EER for this batch of embeddings.
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, speakers_per_batch))
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long()
        loss = self.loss_fn(sim_matrix, target)
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()
            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())
            eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        return loss, eer


class HighwayNetwork(nn.Module):

    def __init__(self, size):
        super().__init__()
        self.W1 = nn.Linear(size, size)
        self.W2 = nn.Linear(size, size)
        self.W1.bias.data.fill_(0.0)

    def forward(self, x):
        x1 = self.W1(x)
        x2 = self.W2(x)
        g = torch.sigmoid(x2)
        y = g * F.relu(x1) + (1.0 - g) * x
        return y


class BatchNormConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, relu=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride=1, padding=kernel // 2, bias=False)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x) if self.relu is True else x
        return self.bnorm(x)


class CBHG(nn.Module):

    def __init__(self, K, in_channels, channels, proj_channels, num_highways):
        super().__init__()
        self._to_flatten = []
        self.bank_kernels = [i for i in range(1, K + 1)]
        self.conv1d_bank = nn.ModuleList()
        for k in self.bank_kernels:
            conv = BatchNormConv(in_channels, channels, k)
            self.conv1d_bank.append(conv)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        self.conv_project1 = BatchNormConv(len(self.bank_kernels) * channels, proj_channels[0], 3)
        self.conv_project2 = BatchNormConv(proj_channels[0], proj_channels[1], 3, relu=False)
        if proj_channels[-1] != channels:
            self.highway_mismatch = True
            self.pre_highway = nn.Linear(proj_channels[-1], channels, bias=False)
        else:
            self.highway_mismatch = False
        self.highways = nn.ModuleList()
        for i in range(num_highways):
            hn = HighwayNetwork(channels)
            self.highways.append(hn)
        self.rnn = nn.GRU(channels, channels // 2, batch_first=True, bidirectional=True)
        self._to_flatten.append(self.rnn)
        self._flatten_parameters()
        self.counts = 0

    def forward(self, x):
        self._flatten_parameters()
        residual = x
        seq_len = x.size(-1)
        conv_bank = []
        for conv in self.conv1d_bank:
            c = conv(x)
            conv_bank.append(c[:, :, :seq_len])
        conv_bank = torch.cat(conv_bank, dim=1)
        x = self.maxpool(conv_bank)[:, :, :seq_len]
        x = self.conv_project1(x)
        x = self.conv_project2(x)
        x = x + residual
        x = x.transpose(1, 2)
        if self.highway_mismatch is True:
            x = self.pre_highway(x)
        for h in self.highways:
            x = h(x)
        x, _ = self.rnn(x)
        return x

    def _flatten_parameters(self):
        """Calls `flatten_parameters` on all the rnns used by the WaveRNN. Used
        to improve efficiency and avoid PyTorch yelling at us."""
        [m.flatten_parameters() for m in self._to_flatten]


class PreNet(nn.Module):

    def __init__(self, in_dims, fc1_dims=256, fc2_dims=128, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.p = dropout
        self.counts = 0

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        self.counts += 1
        x = F.dropout(x, self.p, training=True)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=True)
        return x


class Encoder(nn.Module):

    def __init__(self, embed_dims, num_chars, encoder_dims, K, num_highways, dropout):
        super().__init__()
        prenet_dims = encoder_dims, encoder_dims
        cbhg_channels = encoder_dims
        self.embedding = nn.Embedding(num_chars, embed_dims)
        self.pre_net = PreNet(embed_dims, fc1_dims=prenet_dims[0], fc2_dims=prenet_dims[1], dropout=dropout)
        self.cbhg = CBHG(K=K, in_channels=cbhg_channels, channels=cbhg_channels, proj_channels=[cbhg_channels, cbhg_channels], num_highways=num_highways)

    def forward(self, x, speaker_embedding=None):
        x = self.embedding(x)
        x = self.pre_net(x)
        x.transpose_(1, 2)
        x = self.cbhg(x)
        if speaker_embedding is not None:
            x = self.add_speaker_embedding(x, speaker_embedding)
        return x

    def add_speaker_embedding(self, x, speaker_embedding):
        batch_size = x.size()[0]
        num_chars = x.size()[1]
        if speaker_embedding.dim() == 1:
            idx = 0
        else:
            idx = 1
        speaker_embedding_size = speaker_embedding.size()[idx]
        e = speaker_embedding.repeat_interleave(num_chars, dim=idx)
        e = e.reshape(batch_size, speaker_embedding_size, num_chars)
        e = e.transpose(1, 2)
        x = torch.cat((x, e), 2)
        return x


class Attention(nn.Module):

    def __init__(self, attn_dims):
        super().__init__()
        self.W = nn.Linear(attn_dims, attn_dims, bias=False)
        self.v = nn.Linear(attn_dims, 1, bias=False)

    def forward(self, encoder_seq_proj, query, t):
        query_proj = self.W(query).unsqueeze(1)
        u = self.v(torch.tanh(encoder_seq_proj + query_proj))
        scores = F.softmax(u, dim=1)
        return scores.transpose(1, 2)


class LSA(nn.Module):

    def __init__(self, attn_dim, kernel_size=31, filters=32):
        super().__init__()
        self.conv = nn.Conv1d(1, filters, padding=(kernel_size - 1) // 2, kernel_size=kernel_size, bias=True)
        self.L = nn.Linear(filters, attn_dim, bias=False)
        self.W = nn.Linear(attn_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.cumulative = None
        self.attention = None

    def init_attention(self, encoder_seq_proj):
        device = next(self.parameters()).device
        b, t, c = encoder_seq_proj.size()
        self.cumulative = torch.zeros(b, t, device=device)
        self.attention = torch.zeros(b, t, device=device)

    def forward(self, encoder_seq_proj, query, t, chars):
        if t == 0:
            self.init_attention(encoder_seq_proj)
        processed_query = self.W(query).unsqueeze(1)
        location = self.cumulative.unsqueeze(1)
        processed_loc = self.L(self.conv(location).transpose(1, 2))
        u = self.v(torch.tanh(processed_query + encoder_seq_proj + processed_loc))
        u = u.squeeze(-1)
        u = u * (chars != 0).float()
        scores = F.softmax(u, dim=1)
        self.attention = scores
        self.cumulative = self.cumulative + self.attention
        return scores.unsqueeze(-1).transpose(1, 2)


class Decoder(object):

    def _optimize_graph(self, graph):
        torch._C._jit_pass_constant_propagation(graph)
        torch._C._jit_pass_dce(graph)
        torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_peephole(graph)
        torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_dce(graph)
        torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_canonicalize(graph)
        torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_constant_propagation(graph)
        return graph


def sum(input, dim=None, keepdim=False, *, out=None):
    if dim is None:
        warnings.warn('The output of paddle.sum is not scalar!')
        return paddle.sum(input)
    else:
        return paddle.sum(input, axis=dim, keepdim=keepdim)


class Tacotron(nn.Module):

    def __init__(self, embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, fft_bins, postnet_dims, encoder_K, lstm_dims, postnet_K, num_highways, dropout, stop_threshold, speaker_embedding_size):
        super().__init__()
        self.n_mels = n_mels
        self.lstm_dims = lstm_dims
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims
        self.speaker_embedding_size = speaker_embedding_size
        self.encoder = Encoder(embed_dims, num_chars, encoder_dims, encoder_K, num_highways, dropout)
        self.encoder_proj = nn.Linear(encoder_dims + speaker_embedding_size, decoder_dims, bias=False)
        self.decoder = Decoder(n_mels, encoder_dims, decoder_dims, lstm_dims, dropout, speaker_embedding_size)
        self.postnet = CBHG(postnet_K, n_mels, postnet_dims, [postnet_dims, fft_bins], num_highways)
        self.post_proj = nn.Linear(postnet_dims, fft_bins, bias=False)
        self.init_model()
        self.num_params()
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
        self.register_buffer('stop_threshold', torch.tensor(stop_threshold, dtype=torch.float32))

    @property
    def r(self):
        return self.decoder.r.item()

    @r.setter
    def r(self, value):
        self.decoder.r = self.decoder.r.new_tensor(value, requires_grad=False)

    def forward(self, x, m, speaker_embedding):
        device = next(self.parameters()).device
        self.step += 1
        batch_size, _, steps = m.size()
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        hidden_states = attn_hidden, rnn1_hidden, rnn2_hidden
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = rnn1_cell, rnn2_cell
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)
        context_vec = torch.zeros(batch_size, self.encoder_dims + self.speaker_embedding_size, device=device)
        encoder_seq = self.encoder(x, speaker_embedding)
        encoder_seq_proj = self.encoder_proj(encoder_seq)
        mel_outputs, attn_scores, stop_outputs = [], [], []
        for t in range(0, steps, self.r):
            prenet_in = m[:, :, t - 1] if t > 0 else go_frame
            mel_frames, scores, hidden_states, cell_states, context_vec, stop_tokens = self.decoder(encoder_seq, encoder_seq_proj, prenet_in, hidden_states, cell_states, context_vec, t, x)
            mel_outputs.append(mel_frames)
            attn_scores.append(scores)
            stop_outputs.extend([stop_tokens] * self.r)
        mel_outputs = torch.cat(mel_outputs, dim=2)
        None
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)
        attn_scores = torch.cat(attn_scores, 1)
        stop_outputs = torch.cat(stop_outputs, 1)
        return mel_outputs, linear, attn_scores, stop_outputs

    def generate(self, x, speaker_embedding=None, steps=2000):
        self.eval()
        device = next(self.parameters()).device
        batch_size, _ = x.size()
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        hidden_states = attn_hidden, rnn1_hidden, rnn2_hidden
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = rnn1_cell, rnn2_cell
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)
        context_vec = torch.zeros(batch_size, self.encoder_dims + self.speaker_embedding_size, device=device)
        encoder_seq = self.encoder(x, speaker_embedding)
        encoder_seq_proj = self.encoder_proj(encoder_seq)
        mel_outputs, attn_scores, stop_outputs = [], [], []
        for t in range(0, steps, self.r):
            prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
            mel_frames, scores, hidden_states, cell_states, context_vec, stop_tokens = self.decoder(encoder_seq, encoder_seq_proj, prenet_in, hidden_states, cell_states, context_vec, t, x)
            mel_outputs.append(mel_frames)
            attn_scores.append(scores)
            stop_outputs.extend([stop_tokens] * self.r)
            if (stop_tokens > 0.5).all() and t > 10:
                break
        mel_outputs = torch.cat(mel_outputs, dim=2)
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)
        attn_scores = torch.cat(attn_scores, 1)
        stop_outputs = torch.cat(stop_outputs, 1)
        self.train()
        return mel_outputs, linear, attn_scores

    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_step(self):
        return self.step.data.item()

    def reset_step(self):
        self.step = self.step.data.new_tensor(1)

    def log(self, path, msg):
        with open(path, 'a') as f:
            None

    def load(self, path, optimizer=None):
        device = next(self.parameters()).device
        checkpoint = torch.load(str(path), map_location=device)
        self.load_state_dict(checkpoint['model_state'])
        if 'optimizer_state' in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])

    def save(self, path, optimizer=None):
        if optimizer is not None:
            torch.save({'model_state': self.state_dict(), 'optimizer_state': optimizer.state_dict()}, str(path))
        else:
            torch.save({'model_state': self.state_dict()}, str(path))

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
        if print_out:
            None
        return parameters


LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean, std)


class ResBlock1(torch.nn.Module):

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2])))])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1])))])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):

    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(ConvTranspose1d(h.upsample_initial_channel // 2 ** i, h.upsample_initial_channel // 2 ** (i + 1), k, u, padding=u // 2 + u % 2, output_padding=u % 2)))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // 2 ** (i + 1)
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        None
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(torch.nn.Module):

    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)))])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - t % self.period
            x = F.pad(x, (0, n_pad), 'reflect')
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):

    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(2), DiscriminatorP(3), DiscriminatorP(5), DiscriminatorP(7), DiscriminatorP(11)])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):

    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([norm_f(Conv1d(1, 128, 15, 1, padding=7)), norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)), norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)), norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)), norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)), norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)), norm_f(Conv1d(1024, 1024, 5, 1, padding=2))])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):

    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([DiscriminatorS(use_spectral_norm=True), DiscriminatorS(), DiscriminatorS()])
        self.meanpools = nn.ModuleList([AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class MelResNet(nn.Module):

    def __init__(self, res_blocks, in_dims, compute_dims, res_out_dims, pad):
        super().__init__()
        k_size = pad * 2 + 1
        self.conv_in = nn.Conv1d(in_dims, compute_dims, kernel_size=k_size, bias=False)
        self.batch_norm = nn.BatchNorm1d(compute_dims)
        self.layers = nn.ModuleList()
        for i in range(res_blocks):
            self.layers.append(ResBlock(compute_dims))
        self.conv_out = nn.Conv1d(compute_dims, res_out_dims, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers:
            x = f(x)
        x = self.conv_out(x)
        return x


class Stretch2d(nn.Module):

    def __init__(self, x_scale, y_scale):
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.unsqueeze(-1).unsqueeze(3)
        x = x.repeat(1, 1, 1, self.y_scale, 1, self.x_scale)
        return x.view(b, c, h * self.y_scale, w * self.x_scale)


class UpsampleNetwork(nn.Module):

    def __init__(self, feat_dims, upsample_scales, compute_dims, res_blocks, res_out_dims, pad):
        super().__init__()
        total_scale = np.cumproduct(upsample_scales)[-1]
        self.indent = pad * total_scale
        self.resnet = MelResNet(res_blocks, feat_dims, compute_dims, res_out_dims, pad)
        self.resnet_stretch = Stretch2d(total_scale, 1)
        self.up_layers = nn.ModuleList()
        for scale in upsample_scales:
            k_size = 1, scale * 2 + 1
            padding = 0, scale
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
            conv.weight.data.fill_(1.0 / k_size[1])
            self.up_layers.append(stretch)
            self.up_layers.append(conv)

    def forward(self, m):
        aux = self.resnet(m).unsqueeze(1)
        aux = self.resnet_stretch(aux)
        aux = aux.squeeze(1)
        m = m.unsqueeze(1)
        for f in self.up_layers:
            m = f(m)
        m = m.squeeze(1)[:, :, self.indent:-self.indent]
        return m.transpose(1, 2), aux.transpose(1, 2)


def de_emphasis(x):
    return lfilter([1], [1, -hp.preemphasis], x)


def label_2_float(x, bits):
    return 2 * x / (2 ** bits - 1.0) - 1.0


def decode_mu_law(y, mu, from_labels=True):
    if from_labels:
        y = label_2_float(y, math.log2(mu))
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


def progbar(i, n, size=16):
    done = i * size // n
    bar = ''
    for i in range(size):
        bar += '█' if i <= done else '░'
    return bar


def to_one_hot(tensor, n, fill_with=1.0):
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda:
        one_hot = one_hot
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot


def sample_from_discretized_mix_logistic(y, log_scale_min=None):
    """
    Sample from discretized mixture of logistic distributions
    Args:
        y (Tensor): B x C x T
        log_scale_min (float): Log scale minimum value
    Returns:
        Tensor: sample in range of [-1, 1].
    """
    if log_scale_min is None:
        log_scale_min = float(np.log(1e-14))
    assert y.size(1) % 3 == 0
    nr_mix = y.size(1) // 3
    y = y.transpose(1, 2)
    logit_probs = y[:, :, :nr_mix]
    temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-05, 1.0 - 1e-05)
    temp = logit_probs.data - torch.log(-torch.log(temp))
    _, argmax = temp.max(dim=-1)
    one_hot = to_one_hot(argmax, nr_mix)
    means = torch.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, dim=-1)
    log_scales = torch.clamp(torch.sum(y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, dim=-1), min=log_scale_min)
    u = means.data.new(means.size()).uniform_(1e-05, 1.0 - 1e-05)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1.0 - u))
    x = torch.clamp(torch.clamp(x, min=-1.0), max=1.0)
    return x


def stream(message):
    try:
        sys.stdout.write('\r{%s}' % message)
    except:
        message = ''.join(i for i in message if ord(i) < 128)
        sys.stdout.write('\r{%s}' % message)


class WaveRNN(nn.Module):

    def __init__(self, rnn_dims, fc_dims, bits, pad, upsample_factors, feat_dims, compute_dims, res_out_dims, res_blocks, hop_length, sample_rate, mode='RAW'):
        super().__init__()
        self.mode = mode
        self.pad = pad
        if self.mode == 'RAW':
            self.n_classes = 2 ** bits
        elif self.mode == 'MOL':
            self.n_classes = 30
        else:
            RuntimeError('Unknown model mode value - ', self.mode)
        self.rnn_dims = rnn_dims
        self.aux_dims = res_out_dims // 4
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.upsample = UpsampleNetwork(feat_dims, upsample_factors, compute_dims, res_blocks, res_out_dims, pad)
        self.I = nn.Linear(feat_dims + self.aux_dims + 1, rnn_dims)
        self.rnn1 = nn.GRU(rnn_dims, rnn_dims, batch_first=True)
        self.rnn2 = nn.GRU(rnn_dims + self.aux_dims, rnn_dims, batch_first=True)
        self.fc1 = nn.Linear(rnn_dims + self.aux_dims, fc_dims)
        self.fc2 = nn.Linear(fc_dims + self.aux_dims, fc_dims)
        self.fc3 = nn.Linear(fc_dims, self.n_classes)
        self.step = nn.Parameter(torch.zeros(1).long(), requires_grad=False)
        self.num_params()

    def forward(self, x, mels):
        self.step += 1
        bsize = x.size(0)
        if torch.cuda.is_available():
            h1 = torch.zeros(1, bsize, self.rnn_dims)
            h2 = torch.zeros(1, bsize, self.rnn_dims)
        else:
            h1 = torch.zeros(1, bsize, self.rnn_dims).cpu()
            h2 = torch.zeros(1, bsize, self.rnn_dims).cpu()
        mels, aux = self.upsample(mels)
        aux_idx = [(self.aux_dims * i) for i in range(5)]
        a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
        a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
        a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
        a4 = aux[:, :, aux_idx[3]:aux_idx[4]]
        x = torch.cat([x.unsqueeze(-1), mels, a1], dim=2)
        x = self.I(x)
        res = x
        x, _ = self.rnn1(x, h1)
        x = x + res
        res = x
        x = torch.cat([x, a2], dim=2)
        x, _ = self.rnn2(x, h2)
        x = x + res
        x = torch.cat([x, a3], dim=2)
        x = F.relu(self.fc1(x))
        x = torch.cat([x, a4], dim=2)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def generate(self, mels, batched, target, overlap, mu_law, progress_callback=None):
        mu_law = mu_law if self.mode == 'RAW' else False
        progress_callback = progress_callback or self.gen_display
        self.eval()
        output = []
        start = time.time()
        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)
        with torch.no_grad():
            if torch.cuda.is_available():
                mels = mels
            else:
                mels = mels.cpu()
            wave_len = (mels.size(-1) - 1) * self.hop_length
            mels = self.pad_tensor(mels.transpose(1, 2), pad=self.pad, side='both')
            mels, aux = self.upsample(mels.transpose(1, 2))
            if batched:
                mels = self.fold_with_overlap(mels, target, overlap)
                aux = self.fold_with_overlap(aux, target, overlap)
            b_size, seq_len, _ = mels.size()
            if torch.cuda.is_available():
                h1 = torch.zeros(b_size, self.rnn_dims)
                h2 = torch.zeros(b_size, self.rnn_dims)
                x = torch.zeros(b_size, 1)
            else:
                h1 = torch.zeros(b_size, self.rnn_dims).cpu()
                h2 = torch.zeros(b_size, self.rnn_dims).cpu()
                x = torch.zeros(b_size, 1).cpu()
            d = self.aux_dims
            aux_split = [aux[:, :, d * i:d * (i + 1)] for i in range(4)]
            for i in range(seq_len):
                m_t = mels[:, i, :]
                a1_t, a2_t, a3_t, a4_t = (a[:, i, :] for a in aux_split)
                x = torch.cat([x, m_t, a1_t], dim=1)
                x = self.I(x)
                h1 = rnn1(x, h1)
                x = x + h1
                inp = torch.cat([x, a2_t], dim=1)
                h2 = rnn2(inp, h2)
                x = x + h2
                x = torch.cat([x, a3_t], dim=1)
                x = F.relu(self.fc1(x))
                x = torch.cat([x, a4_t], dim=1)
                x = F.relu(self.fc2(x))
                logits = self.fc3(x)
                if self.mode == 'MOL':
                    sample = sample_from_discretized_mix_logistic(logits.unsqueeze(0).transpose(1, 2))
                    output.append(sample.view(-1))
                    if torch.cuda.is_available():
                        x = sample.transpose(0, 1)
                    else:
                        x = sample.transpose(0, 1)
                elif self.mode == 'RAW':
                    posterior = F.softmax(logits, dim=1)
                    distrib = torch.distributions.Categorical(posterior)
                    sample = 2 * distrib.sample().float() / (self.n_classes - 1.0) - 1.0
                    output.append(sample)
                    x = sample.unsqueeze(-1)
                else:
                    raise RuntimeError('Unknown model mode value - ', self.mode)
                if i % 100 == 0:
                    gen_rate = (i + 1) / (time.time() - start) * b_size / 1000
                    progress_callback(i, seq_len, b_size, gen_rate)
        output = torch.stack(output).transpose(0, 1)
        output = output.cpu().numpy()
        output = output.astype(np.float64)
        if batched:
            output = self.xfade_and_unfold(output, target, overlap)
        else:
            output = output[0]
        if mu_law:
            output = decode_mu_law(output, self.n_classes, False)
        if hp.apply_preemphasis:
            output = de_emphasis(output)
        fade_out = np.linspace(1, 0, 20 * self.hop_length)
        output = output[:wave_len]
        output[-20 * self.hop_length:] *= fade_out
        self.train()
        return output

    def gen_display(self, i, seq_len, b_size, gen_rate):
        pbar = progbar(i, seq_len)
        msg = f'| {pbar} {i * b_size}/{seq_len * b_size} | Batch Size: {b_size} | Gen Rate: {gen_rate:.1f}kHz | '
        stream(msg)

    def get_gru_cell(self, gru):
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell

    def pad_tensor(self, x, pad, side='both'):
        b, t, c = x.size()
        total = t + 2 * pad if side == 'both' else t + pad
        if torch.cuda.is_available():
            padded = torch.zeros(b, total, c)
        else:
            padded = torch.zeros(b, total, c).cpu()
        if side == 'before' or side == 'both':
            padded[:, pad:pad + t, :] = x
        elif side == 'after':
            padded[:, :t, :] = x
        return padded

    def fold_with_overlap(self, x, target, overlap):
        """ Fold the tensor with overlap for quick batched inference.
            Overlap will be used for crossfading in xfade_and_unfold()

        Args:
            x (tensor)    : Upsampled conditioning features.
                            shape=(1, timesteps, features)
            target (int)  : Target timesteps for each index of batch
            overlap (int) : Timesteps for both xfade and rnn warmup

        Return:
            (tensor) : shape=(num_folds, target + 2 * overlap, features)

        Details:
            x = [[h1, h2, ... hn]]

            Where each h is a vector of conditioning features

            Eg: target=2, overlap=1 with x.size(1)=10

            folded = [[h1, h2, h3, h4],
                      [h4, h5, h6, h7],
                      [h7, h8, h9, h10]]
        """
        _, total_len, features = x.size()
        num_folds = (total_len - overlap) // (target + overlap)
        extended_len = num_folds * (overlap + target) + overlap
        remaining = total_len - extended_len
        if remaining != 0:
            num_folds += 1
            padding = target + 2 * overlap - remaining
            x = self.pad_tensor(x, padding, side='after')
        if torch.cuda.is_available():
            folded = torch.zeros(num_folds, target + 2 * overlap, features)
        else:
            folded = torch.zeros(num_folds, target + 2 * overlap, features).cpu()
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            folded[i] = x[:, start:end, :]
        return folded

    def xfade_and_unfold(self, y, target, overlap):
        """ Applies a crossfade and unfolds into a 1d array.

        Args:
            y (ndarry)    : Batched sequences of audio samples
                            shape=(num_folds, target + 2 * overlap)
                            dtype=np.float64
            overlap (int) : Timesteps for both xfade and rnn warmup

        Return:
            (ndarry) : audio samples in a 1d array
                       shape=(total_len)
                       dtype=np.float64

        Details:
            y = [[seq1],
                 [seq2],
                 [seq3]]

            Apply a gain envelope at both ends of the sequences

            y = [[seq1_in, seq1_target, seq1_out],
                 [seq2_in, seq2_target, seq2_out],
                 [seq3_in, seq3_target, seq3_out]]

            Stagger and add up the groups of samples:

            [seq1_in, seq1_target, (seq1_out + seq2_in), seq2_target, ...]

        """
        num_folds, length = y.shape
        target = length - 2 * overlap
        total_len = num_folds * (target + overlap) + overlap
        silence_len = overlap // 2
        fade_len = overlap - silence_len
        silence = np.zeros(silence_len, dtype=np.float64)
        t = np.linspace(-1, 1, fade_len, dtype=np.float64)
        fade_in = np.sqrt(0.5 * (1 + t))
        fade_out = np.sqrt(0.5 * (1 - t))
        fade_in = np.concatenate([silence, fade_in])
        fade_out = np.concatenate([fade_out, silence])
        y[:, :overlap] *= fade_in
        y[:, -overlap:] *= fade_out
        unfolded = np.zeros(total_len, dtype=np.float64)
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            unfolded[start:end] += y[i]
        return unfolded

    def get_step(self):
        return self.step.data.item()

    def checkpoint(self, model_dir, optimizer):
        k_steps = self.get_step() // 1000
        self.save(model_dir.joinpath('checkpoint_%dk_steps.pt' % k_steps), optimizer)

    def log(self, path, msg):
        with open(path, 'a') as f:
            None

    def load(self, path, optimizer):
        checkpoint = torch.load(path)
        if 'optimizer_state' in checkpoint:
            self.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        else:
            self.load_state_dict(checkpoint)

    def save(self, path, optimizer):
        torch.save({'model_state': self.state_dict(), 'optimizer_state': optimizer.state_dict()}, path)

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
        if print_out:
            None


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, fully_conv=False, remove_avg_pool_layer=False, output_stride=32):
        self.output_stride = output_stride
        self.current_stride = 4
        self.current_dilation = 1
        self.remove_avg_pool_layer = remove_avg_pool_layer
        self.inplanes = 64
        self.fully_conv = fully_conv
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        if self.fully_conv:
            self.avgpool = nn.AvgPool2d(7, padding=3, stride=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.current_stride == self.output_stride:
                self.current_dilation = self.current_dilation * stride
                stride = 1
            else:
                self.current_stride = self.current_stride * stride
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=self.current_dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=self.current_dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x2s = self.relu(x)
        x = self.maxpool(x2s)
        x4s = self.layer1(x)
        x8s = self.layer2(x4s)
        x16s = self.layer3(x8s)
        x32s = self.layer4(x16s)
        x = x32s
        if not self.remove_avg_pool_layer:
            x = self.avgpool(x)
        if not self.fully_conv:
            x = x.view(x.size(0), -1)
        xfc = self.fc(x)
        return x2s, x4s, x8s, x16s, x32s, xfc


def abs(input, *, out=None):
    return paddle.abs(input)


def max(input, dim_other=None, keepdim=False, *, out=None):
    if dim_other is None:
        return paddle.max(input)
    elif isinstance(dim_other, paddle.Tensor):
        return paddle.maximum(input, dim_other)
    else:
        return paddle.max(input, axis=dim_other, keepdim=keepdim)


def min(input, dim_other=None, keepdim=False, *, out=None):
    if dim_other is None:
        return paddle.min(input)
    elif isinstance(dim_other, paddle.Tensor):
        return paddle.minimum(input, dim_other)
    else:
        return paddle.min(input, axis=dim_other, keepdim=keepdim)


class Resnet18(nn.Module):

    def __init__(self, ver_dim, seg_dim, normal_dim, fcdim=256, s8dim=128, s4dim=64, s2dim=32, raw_dim=32):
        super(Resnet18, self).__init__()
        resnet18_8s = resnet18(fully_conv=True, pretrained=False, output_stride=8, remove_avg_pool_layer=True)
        self.ver_dim = ver_dim
        self.seg_dim = seg_dim
        self.norm_dim = normal_dim
        resnet18_8s.fc = nn.Sequential(nn.Conv2d(resnet18_8s.inplanes, fcdim, 3, 1, 1, bias=False), nn.BatchNorm2d(fcdim), nn.ReLU(True))
        self.resnet18_8s = resnet18_8s
        self.conv8s = nn.Sequential(nn.Conv2d(128 + fcdim, s8dim, 3, 1, 1, bias=False), nn.BatchNorm2d(s8dim), nn.LeakyReLU(0.1, True))
        self.up8sto4s = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv4s = nn.Sequential(nn.Conv2d(64 + s8dim, s4dim, 3, 1, 1, bias=False), nn.BatchNorm2d(s4dim), nn.LeakyReLU(0.1, True))
        self.conv2s = nn.Sequential(nn.Conv2d(64 + s4dim, s2dim, 3, 1, 1, bias=False), nn.BatchNorm2d(s2dim), nn.LeakyReLU(0.1, True))
        self.up4sto2s = nn.UpsamplingBilinear2d(scale_factor=2)
        self.convraw = nn.Sequential(nn.Conv2d(3 + 2 + s2dim, raw_dim, 3, 1, 1, bias=False), nn.BatchNorm2d(raw_dim), nn.LeakyReLU(0.1, True))
        self.convraw2 = nn.Sequential(nn.Conv2d(3 + s2dim, raw_dim, 3, 1, 1, bias=False), nn.BatchNorm2d(raw_dim), nn.LeakyReLU(0.1, True))
        self.convNormal = nn.Sequential(nn.Conv2d(raw_dim, raw_dim, 3, 1, 1, bias=False), nn.BatchNorm2d(raw_dim), nn.LeakyReLU(0.1, True), nn.Conv2d(raw_dim, normal_dim + seg_dim, 1, 1))
        self.ActivateNormal = nn.Sequential(nn.BatchNorm2d(3), nn.LeakyReLU(0.1, True))
        self.ActivateMask = nn.Sequential(nn.BatchNorm2d(2), nn.LeakyReLU(0.1, True))
        self.convNormal2 = nn.Sequential(nn.Conv2d(3 + 2, raw_dim, 3, 1, 1, bias=False), nn.BatchNorm2d(raw_dim), nn.LeakyReLU(0.1, True), nn.Conv2d(raw_dim, raw_dim, 3, 1, 1, bias=False), nn.BatchNorm2d(raw_dim), nn.LeakyReLU(0.1, True))
        self.convVote = nn.Sequential(nn.Conv2d(raw_dim * 2, raw_dim * 2, 3, 1, 1, bias=False), nn.BatchNorm2d(raw_dim * 2), nn.LeakyReLU(0.1, True), nn.Conv2d(raw_dim * 2, raw_dim, 3, 1, 1, bias=False), nn.BatchNorm2d(raw_dim), nn.LeakyReLU(0.1, True), nn.Conv2d(raw_dim, seg_dim + ver_dim, 1, 1))
        self.up2storaw = nn.UpsamplingBilinear2d(scale_factor=2)
        for name, parameter in self.named_parameters():
            if 'resnet18_8s.conv1' in name or 'resnet18_8s.bn1' in name or 'resnet18_8s.layer1' in name:
                parameter.requires_grad = False

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def compute_roi(self, hmax, hmin, wmax, wmin, h, w):
        min_y = hmin
        min_x = wmin
        max_y = hmax
        max_x = wmax
        center_x = int(math.floor((min_x + max_x) / 2))
        center_y = int(math.floor((min_y + max_y) / 2))
        half_roi_len = max(abs(center_x - min_x), abs(center_x - max_x), abs(center_y - min_y), abs(center_y - max_y))
        roi_len = int(min(h, w, half_roi_len * 2 + 1))
        half_roi_len = int(math.ceil((roi_len - 1) / 2))
        roi_box = torch.Tensor([max(0, center_x - half_roi_len), max(0, center_y - half_roi_len), min(w - 1, center_x + half_roi_len), min(h - 1, center_y + half_roi_len)])
        roi = torch.zeros(4, dtype=torch.int32)
        if roi_box[0] == 0:
            roi[0] = roi_box[0]
        elif roi_box[2] == w - 1:
            roi[0] = w - roi_len
        else:
            roi[0] = roi_box[0]
        if roi_box[1] == 0:
            roi[1] = roi_box[1]
        elif roi_box[3] == h - 1:
            roi[1] = h - roi_len
        else:
            roi[1] = roi_box[1]
        roi[2] = roi_len
        roi[3] = roi_len
        return roi.int()

    def forward(self, x, rois=None, feature_alignment=False):
        x_inp = x[:, 0:3]
        x_mesh = x[:, 3:]
        x2s, x4s, x8s, x16s, x32s, xfc = self.resnet18_8s(x_inp)
        fm = self.conv8s(torch.cat([xfc, x8s], 1))
        fm = self.up8sto4s(fm)
        if fm.shape[2] == 136:
            fm = nn.functional.interpolate(fm, (135, 180), mode='bilinear', align_corners=False)
        fm = self.conv4s(torch.cat([fm, x4s], 1))
        fm = self.up4sto2s(fm)
        fm = self.conv2s(torch.cat([fm, x2s], 1))
        fm = self.up2storaw(fm)
        x1 = self.convraw(torch.cat([fm, x_inp, x_mesh], 1))
        norm_mask = self.convNormal(x1)
        norm_activate = self.ActivateNormal(norm_mask[:, self.norm_dim - 3:self.norm_dim, :, :])
        mask_activate = self.ActivateMask(norm_mask[:, self.norm_dim:, :, :])
        norm_mask_to_cat = self.convNormal2(torch.cat([norm_activate, mask_activate], 1))
        x2 = self.convraw2(torch.cat([fm, x_inp], 1))
        vote_mask = self.convVote(torch.cat([norm_mask_to_cat, x2], 1))
        coarse_seg_pred = norm_mask[:, self.norm_dim:, :, :]
        norm_pred = norm_mask[:, :self.norm_dim, :, :]
        fine_seg_pred = vote_mask[:, self.ver_dim:, :, :]
        ver_pred = vote_mask[:, :self.ver_dim, :, :]
        ret = {'coarse_seg': coarse_seg_pred, 'norm': norm_pred, 'seg': fine_seg_pred, 'vertex': ver_pred}
        return ret


LOGGER = logging.getLogger(__name__)


class InpaintingEvaluatorOnline(nn.Module):

    def __init__(self, scores, bins=10, image_key='image', inpainted_key='inpainted', integral_func=None, integral_title=None, clamp_image_range=None):
        """
        :param scores: dict {score_name: EvaluatorScore object}
        :param bins: number of groups, partition is generated by np.linspace(0., 1., bins + 1)
        :param device: device to use
        """
        super().__init__()
        LOGGER.info(f'{type(self)} init called')
        self.scores = nn.ModuleDict(scores)
        self.image_key = image_key
        self.inpainted_key = inpainted_key
        self.bins_num = bins
        self.bin_edges = np.linspace(0, 1, self.bins_num + 1)
        num_digits = max(0, math.ceil(math.log10(self.bins_num)) - 1)
        self.interval_names = []
        for idx_bin in range(self.bins_num):
            start_percent, end_percent = round(100 * self.bin_edges[idx_bin], num_digits), round(100 * self.bin_edges[idx_bin + 1], num_digits)
            start_percent = '{:.{n}f}'.format(start_percent, n=num_digits)
            end_percent = '{:.{n}f}'.format(end_percent, n=num_digits)
            self.interval_names.append('{0}-{1}%'.format(start_percent, end_percent))
        self.groups = []
        self.integral_func = integral_func
        self.integral_title = integral_title
        self.clamp_image_range = clamp_image_range
        LOGGER.info(f'{type(self)} init done')

    def _get_bins(self, mask_batch):
        batch_size = mask_batch.shape[0]
        area = mask_batch.view(batch_size, -1).mean(dim=-1).detach().cpu().numpy()
        bin_indices = np.clip(np.searchsorted(self.bin_edges, area) - 1, 0, self.bins_num - 1)
        return bin_indices

    def forward(self, batch: 'Dict[str, torch.Tensor]'):
        """
        Calculate and accumulate metrics for batch. To finalize evaluation and obtain final metrics, call evaluation_end
        :param batch: batch dict with mandatory fields mask, image, inpainted (can be overriden by self.inpainted_key)
        """
        result = {}
        with torch.no_grad():
            image_batch, mask_batch, inpainted_batch = batch[self.image_key], batch['mask'], batch[self.inpainted_key]
            if self.clamp_image_range is not None:
                image_batch = torch.clamp(image_batch, min=self.clamp_image_range[0], max=self.clamp_image_range[1])
            self.groups.extend(self._get_bins(mask_batch))
            for score_name, score in self.scores.items():
                result[score_name] = score(inpainted_batch, image_batch, mask_batch)
        return result

    def process_batch(self, batch: 'Dict[str, torch.Tensor]'):
        return self(batch)

    def evaluation_end(self, states=None):
        """:return: dict with (score_name, group_type) as keys, where group_type can be either 'overall' or
            name of the particular group arranged by area of mask (e.g. '10-20%')
            and score statistics for the group as values.
        """
        LOGGER.info(f'{type(self)}: evaluation_end called')
        self.groups = np.array(self.groups)
        results = {}
        for score_name, score in self.scores.items():
            LOGGER.info(f'Getting value of {score_name}')
            cur_states = [s[score_name] for s in states] if states is not None else None
            total_results, group_results = score.get_value(groups=self.groups, states=cur_states)
            LOGGER.info(f'Getting value of {score_name} done')
            results[score_name, 'total'] = total_results
            for group_index, group_values in group_results.items():
                group_name = self.interval_names[group_index]
                results[score_name, group_name] = group_values
        if self.integral_func is not None:
            results[self.integral_title, 'total'] = dict(mean=self.integral_func(results))
        LOGGER.info(f'{type(self)}: reset scores')
        self.groups = []
        for sc in self.scores.values():
            sc.reset()
        LOGGER.info(f'{type(self)}: reset scores done')
        LOGGER.info(f'{type(self)}: evaluation_end done')
        return results


class EvaluatorScore(nn.Module):

    @abstractmethod
    def forward(self, pred_batch, target_batch, mask):
        pass

    @abstractmethod
    def get_value(self, groups=None, states=None):
        pass

    @abstractmethod
    def reset(self):
        pass


def get_groupings(groups):
    """
    :param groups: group numbers for respective elements
    :return: dict of kind {group_idx: indices of the corresponding group elements}
    """
    label_groups, count_groups = np.unique(groups, return_counts=True)
    indices = np.argsort(groups)
    grouping = dict()
    cur_start = 0
    for label, count in zip(label_groups, count_groups):
        cur_end = cur_start + count
        cur_indices = indices[cur_start:cur_end]
        grouping[label] = cur_indices
        cur_start = cur_end
    return grouping


class PairwiseScore(EvaluatorScore, ABC):

    def __init__(self):
        super().__init__()
        self.individual_values = None

    def get_value(self, groups=None, states=None):
        """
        :param groups:
        :return:
            total_results: dict of kind {'mean': score mean, 'std': score std}
            group_results: None, if groups is None;
                else dict {group_idx: {'mean': score mean among group, 'std': score std among group}}
        """
        individual_values = torch.cat(states, dim=-1).reshape(-1).cpu().numpy() if states is not None else self.individual_values
        total_results = {'mean': individual_values.mean(), 'std': individual_values.std()}
        if groups is None:
            return total_results, None
        group_results = dict()
        grouping = get_groupings(groups)
        for label, index in grouping.items():
            group_scores = individual_values[index]
            group_results[label] = {'mean': group_scores.mean(), 'std': group_scores.std()}
        return total_results, group_results

    def reset(self):
        self.individual_values = []


class SSIMScore(PairwiseScore):

    def __init__(self, window_size=11):
        super().__init__()
        self.score = SSIM(window_size=window_size, size_average=False).eval()
        self.reset()

    def forward(self, pred_batch, target_batch, mask=None):
        batch_values = self.score(pred_batch, target_batch)
        self.individual_values = np.hstack([self.individual_values, batch_values.detach().cpu().numpy()])
        return batch_values


class VGG19(torch.nn.Module):

    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()
        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()
        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()
        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()
        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()
        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])
        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])
        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])
        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])
        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])
        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])
        for x in range(14, 16):
            self.relu3_2.add_module(str(x), features[x])
        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])
        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])
        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])
        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])
        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])
        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])
        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])
        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])
        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)
        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)
        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)
        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)
        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)
        out = {'relu1_1': relu1_1, 'relu1_2': relu1_2, 'relu2_1': relu2_1, 'relu2_2': relu2_2, 'relu3_1': relu3_1, 'relu3_2': relu3_2, 'relu3_3': relu3_3, 'relu3_4': relu3_4, 'relu4_1': relu4_1, 'relu4_2': relu4_2, 'relu4_3': relu4_3, 'relu4_4': relu4_4, 'relu5_1': relu5_1, 'relu5_2': relu5_2, 'relu5_3': relu5_3, 'relu5_4': relu5_4}
        return out


class PerceptualLoss(nn.Module):
    """
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])
        return content_loss


class LPIPSScore(PairwiseScore):

    def __init__(self, model='net-lin', net='vgg', model_path=None, use_gpu=True):
        super().__init__()
        self.score = PerceptualLoss(model=model, net=net, model_path=model_path, use_gpu=use_gpu, spatial=False).eval()
        self.reset()

    def forward(self, pred_batch, target_batch, mask=None):
        batch_values = self.score(pred_batch, target_batch).flatten()
        self.individual_values = np.hstack([self.individual_values, batch_values.detach().cpu().numpy()])
        return batch_values


class FIDInceptionA(models.inception.InceptionA):
    """InceptionA block patched for FID computation"""

    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(models.inception.InceptionC):
    """InceptionC block patched for FID computation"""

    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

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
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


def load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False, file_name=None):
    return paddle.load(url)


def fid_inception_v3():
    """Build pretrained Inception model for FID computation

    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.

    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    LOGGER.info('fid_inception_v3 called')
    inception = models.inception_v3(num_classes=1008, aux_logits=False, pretrained=False)
    LOGGER.info('models.inception_v3 done')
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)
    LOGGER.info('fid_inception_v3 patching done')
    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    LOGGER.info('fid_inception_v3 weights downloaded')
    inception.load_state_dict(state_dict)
    LOGGER.info('fid_inception_v3 weights loaded into model')
    return inception


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {(64): 0, (192): 1, (768): 2, (2048): 3}

    def __init__(self, output_blocks=[DEFAULT_BLOCK_INDEX], resize_input=True, normalize_input=True, requires_grad=False, use_fid_inception=True):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
        """
        super(InceptionV3, self).__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        assert self.last_needed_block <= 3, 'Last possible output block index is 3'
        self.blocks = nn.ModuleList()
        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = models.inception_v3(pretrained=True)
        block0 = [inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
        self.blocks.append(nn.Sequential(*block0))
        if self.last_needed_block >= 1:
            block1 = [inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
            self.blocks.append(nn.Sequential(*block1))
        if self.last_needed_block >= 2:
            block2 = [inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d, inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e]
            self.blocks.append(nn.Sequential(*block2))
        if self.last_needed_block >= 3:
            block3 = [inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c, nn.AdaptiveAvgPool2d(output_size=(1, 1))]
            self.blocks.append(nn.Sequential(*block3))
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        if self.normalize_input:
            x = 2 * x - 1
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)
            if idx == self.last_needed_block:
                break
        return outp


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-06):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = 'fid calculation produces singular product; adding %s to diagonal of cov estimates' % eps
        None
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=0.01):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


class FIDScore(EvaluatorScore):

    def __init__(self, dims=2048, eps=1e-06):
        LOGGER.info('FIDscore init called')
        super().__init__()
        if getattr(FIDScore, '_MODEL', None) is None:
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            FIDScore._MODEL = InceptionV3([block_idx]).eval()
        self.model = FIDScore._MODEL
        self.eps = eps
        self.reset()
        LOGGER.info('FIDscore init done')

    def forward(self, pred_batch, target_batch, mask=None):
        activations_pred = self._get_activations(pred_batch)
        activations_target = self._get_activations(target_batch)
        self.activations_pred.append(activations_pred.detach().cpu())
        self.activations_target.append(activations_target.detach().cpu())
        return activations_pred, activations_target

    def get_value(self, groups=None, states=None):
        LOGGER.info('FIDscore get_value called')
        activations_pred, activations_target = zip(*states) if states is not None else (self.activations_pred, self.activations_target)
        activations_pred = torch.cat(activations_pred).cpu().numpy()
        activations_target = torch.cat(activations_target).cpu().numpy()
        total_distance = calculate_frechet_distance(activations_pred, activations_target, eps=self.eps)
        total_results = dict(mean=total_distance)
        if groups is None:
            group_results = None
        else:
            group_results = dict()
            grouping = get_groupings(groups)
            for label, index in grouping.items():
                if len(index) > 1:
                    group_distance = calculate_frechet_distance(activations_pred[index], activations_target[index], eps=self.eps)
                    group_results[label] = dict(mean=group_distance)
                else:
                    group_results[label] = dict(mean=float('nan'))
        self.reset()
        LOGGER.info('FIDscore get_value done')
        return total_results, group_results

    def reset(self):
        self.activations_pred = []
        self.activations_target = []

    def _get_activations(self, batch):
        activations = self.model(batch)[0]
        if activations.shape[2] != 1 or activations.shape[3] != 1:
            assert False, 'We should not have got here, because Inception always scales inputs to 299x299'
        activations = activations.squeeze(-1).squeeze(-1)
        return activations


class SegmentationAwareScore(EvaluatorScore):

    def __init__(self, weights_path):
        super().__init__()
        self.segm_network = SegmentationModule(weights_path=weights_path, use_default_normalization=True).eval()
        self.target_class_freq_by_image_total = []
        self.target_class_freq_by_image_mask = []
        self.pred_class_freq_by_image_mask = []

    def forward(self, pred_batch, target_batch, mask):
        pred_segm_flat = self.segm_network.predict(pred_batch)[0].view(pred_batch.shape[0], -1).long().detach().cpu().numpy()
        target_segm_flat = self.segm_network.predict(target_batch)[0].view(pred_batch.shape[0], -1).long().detach().cpu().numpy()
        mask_flat = (mask.view(mask.shape[0], -1) > 0.5).detach().cpu().numpy()
        batch_target_class_freq_total = []
        batch_target_class_freq_mask = []
        batch_pred_class_freq_mask = []
        for cur_pred_segm, cur_target_segm, cur_mask in zip(pred_segm_flat, target_segm_flat, mask_flat):
            cur_target_class_freq_total = np.bincount(cur_target_segm, minlength=NUM_CLASS)[None, ...]
            cur_target_class_freq_mask = np.bincount(cur_target_segm[cur_mask], minlength=NUM_CLASS)[None, ...]
            cur_pred_class_freq_mask = np.bincount(cur_pred_segm[cur_mask], minlength=NUM_CLASS)[None, ...]
            self.target_class_freq_by_image_total.append(cur_target_class_freq_total)
            self.target_class_freq_by_image_mask.append(cur_target_class_freq_mask)
            self.pred_class_freq_by_image_mask.append(cur_pred_class_freq_mask)
            batch_target_class_freq_total.append(cur_target_class_freq_total)
            batch_target_class_freq_mask.append(cur_target_class_freq_mask)
            batch_pred_class_freq_mask.append(cur_pred_class_freq_mask)
        batch_target_class_freq_total = np.concatenate(batch_target_class_freq_total, axis=0)
        batch_target_class_freq_mask = np.concatenate(batch_target_class_freq_mask, axis=0)
        batch_pred_class_freq_mask = np.concatenate(batch_pred_class_freq_mask, axis=0)
        return batch_target_class_freq_total, batch_target_class_freq_mask, batch_pred_class_freq_mask

    def reset(self):
        super().reset()
        self.target_class_freq_by_image_total = []
        self.target_class_freq_by_image_mask = []
        self.pred_class_freq_by_image_mask = []


def distribute_values_to_classes(target_class_freq_by_image_mask, values, idx2name):
    assert target_class_freq_by_image_mask.ndim == 2 and target_class_freq_by_image_mask.shape[0] == values.shape[0]
    total_class_freq = target_class_freq_by_image_mask.sum(0)
    distr_values = (target_class_freq_by_image_mask * values[..., None]).sum(0)
    result = distr_values / (total_class_freq + 0.001)
    return {idx2name[i]: val for i, val in enumerate(result) if total_class_freq[i] > 0}


def get_segmentation_idx2name():
    return {(i - 1): name for i, name in segm_options['classes'].set_index('Idx', drop=True)['Name'].to_dict().items()}


class SegmentationAwarePairwiseScore(SegmentationAwareScore):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.individual_values = []
        self.segm_idx2name = get_segmentation_idx2name()

    def forward(self, pred_batch, target_batch, mask):
        cur_class_stats = super().forward(pred_batch, target_batch, mask)
        score_values = self.calc_score(pred_batch, target_batch, mask)
        self.individual_values.append(score_values)
        return cur_class_stats + (score_values,)

    @abstractmethod
    def calc_score(self, pred_batch, target_batch, mask):
        raise NotImplementedError()

    def get_value(self, groups=None, states=None):
        """
        :param groups:
        :return:
            total_results: dict of kind {'mean': score mean, 'std': score std}
            group_results: None, if groups is None;
                else dict {group_idx: {'mean': score mean among group, 'std': score std among group}}
        """
        if states is not None:
            target_class_freq_by_image_total, target_class_freq_by_image_mask, pred_class_freq_by_image_mask, individual_values = states
        else:
            target_class_freq_by_image_total = self.target_class_freq_by_image_total
            target_class_freq_by_image_mask = self.target_class_freq_by_image_mask
            pred_class_freq_by_image_mask = self.pred_class_freq_by_image_mask
            individual_values = self.individual_values
        target_class_freq_by_image_total = np.concatenate(target_class_freq_by_image_total, axis=0)
        target_class_freq_by_image_mask = np.concatenate(target_class_freq_by_image_mask, axis=0)
        pred_class_freq_by_image_mask = np.concatenate(pred_class_freq_by_image_mask, axis=0)
        individual_values = np.concatenate(individual_values, axis=0)
        total_results = {'mean': individual_values.mean(), 'std': individual_values.std(), **distribute_values_to_classes(target_class_freq_by_image_mask, individual_values, self.segm_idx2name)}
        if groups is None:
            return total_results, None
        group_results = dict()
        grouping = get_groupings(groups)
        for label, index in grouping.items():
            group_class_freq = target_class_freq_by_image_mask[index]
            group_scores = individual_values[index]
            group_results[label] = {'mean': group_scores.mean(), 'std': group_scores.std(), **distribute_values_to_classes(group_class_freq, group_scores, self.segm_idx2name)}
        return total_results, group_results

    def reset(self):
        super().reset()
        self.individual_values = []


class SegmentationClassStats(SegmentationAwarePairwiseScore):

    def calc_score(self, pred_batch, target_batch, mask):
        return 0

    def get_value(self, groups=None, states=None):
        """
        :param groups:
        :return:
            total_results: dict of kind {'mean': score mean, 'std': score std}
            group_results: None, if groups is None;
                else dict {group_idx: {'mean': score mean among group, 'std': score std among group}}
        """
        if states is not None:
            target_class_freq_by_image_total, target_class_freq_by_image_mask, pred_class_freq_by_image_mask, _ = states
        else:
            target_class_freq_by_image_total = self.target_class_freq_by_image_total
            target_class_freq_by_image_mask = self.target_class_freq_by_image_mask
            pred_class_freq_by_image_mask = self.pred_class_freq_by_image_mask
        target_class_freq_by_image_total = np.concatenate(target_class_freq_by_image_total, axis=0)
        target_class_freq_by_image_mask = np.concatenate(target_class_freq_by_image_mask, axis=0)
        pred_class_freq_by_image_mask = np.concatenate(pred_class_freq_by_image_mask, axis=0)
        target_class_freq_by_image_total_marginal = target_class_freq_by_image_total.sum(0).astype('float32')
        target_class_freq_by_image_total_marginal /= target_class_freq_by_image_total_marginal.sum()
        target_class_freq_by_image_mask_marginal = target_class_freq_by_image_mask.sum(0).astype('float32')
        target_class_freq_by_image_mask_marginal /= target_class_freq_by_image_mask_marginal.sum()
        pred_class_freq_diff = (pred_class_freq_by_image_mask - target_class_freq_by_image_mask).sum(0) / (target_class_freq_by_image_mask.sum(0) + 0.001)
        total_results = dict()
        total_results.update({f'total_freq/{self.segm_idx2name[i]}': v for i, v in enumerate(target_class_freq_by_image_total_marginal) if v > 0})
        total_results.update({f'mask_freq/{self.segm_idx2name[i]}': v for i, v in enumerate(target_class_freq_by_image_mask_marginal) if v > 0})
        total_results.update({f'mask_freq_diff/{self.segm_idx2name[i]}': v for i, v in enumerate(pred_class_freq_diff) if target_class_freq_by_image_total_marginal[i] > 0})
        if groups is None:
            return total_results, None
        group_results = dict()
        grouping = get_groupings(groups)
        for label, index in grouping.items():
            group_target_class_freq_by_image_total = target_class_freq_by_image_total[index]
            group_target_class_freq_by_image_mask = target_class_freq_by_image_mask[index]
            group_pred_class_freq_by_image_mask = pred_class_freq_by_image_mask[index]
            group_target_class_freq_by_image_total_marginal = group_target_class_freq_by_image_total.sum(0).astype('float32')
            group_target_class_freq_by_image_total_marginal /= group_target_class_freq_by_image_total_marginal.sum()
            group_target_class_freq_by_image_mask_marginal = group_target_class_freq_by_image_mask.sum(0).astype('float32')
            group_target_class_freq_by_image_mask_marginal /= group_target_class_freq_by_image_mask_marginal.sum()
            group_pred_class_freq_diff = (group_pred_class_freq_by_image_mask - group_target_class_freq_by_image_mask).sum(0) / (group_target_class_freq_by_image_mask.sum(0) + 0.001)
            cur_group_results = dict()
            cur_group_results.update({f'total_freq/{self.segm_idx2name[i]}': v for i, v in enumerate(group_target_class_freq_by_image_total_marginal) if v > 0})
            cur_group_results.update({f'mask_freq/{self.segm_idx2name[i]}': v for i, v in enumerate(group_target_class_freq_by_image_mask_marginal) if v > 0})
            cur_group_results.update({f'mask_freq_diff/{self.segm_idx2name[i]}': v for i, v in enumerate(group_pred_class_freq_diff) if group_target_class_freq_by_image_total_marginal[i] > 0})
            group_results[label] = cur_group_results
        return total_results, group_results


class SegmentationAwareSSIM(SegmentationAwarePairwiseScore):

    def __init__(self, *args, window_size=11, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_impl = SSIM(window_size=window_size, size_average=False).eval()

    def calc_score(self, pred_batch, target_batch, mask):
        return self.score_impl(pred_batch, target_batch).detach().cpu().numpy()


class SegmentationAwareLPIPS(SegmentationAwarePairwiseScore):

    def __init__(self, *args, model='net-lin', net='vgg', model_path=None, use_gpu=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_impl = PerceptualLoss(model=model, net=net, model_path=model_path, use_gpu=use_gpu, spatial=False).eval()

    def calc_score(self, pred_batch, target_batch, mask):
        return self.score_impl(pred_batch, target_batch).flatten().detach().cpu().numpy()


def calculade_fid_no_img(img_i, activations_pred, activations_target, eps=1e-06):
    activations_pred = activations_pred.copy()
    activations_pred[img_i] = activations_target[img_i]
    return calculate_frechet_distance(activations_pred, activations_target, eps=eps)


class SegmentationAwareFID(SegmentationAwarePairwiseScore):

    def __init__(self, *args, dims=2048, eps=1e-06, n_jobs=-1, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(FIDScore, '_MODEL', None) is None:
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            FIDScore._MODEL = InceptionV3([block_idx]).eval()
        self.model = FIDScore._MODEL
        self.eps = eps
        self.n_jobs = n_jobs

    def calc_score(self, pred_batch, target_batch, mask):
        activations_pred = self._get_activations(pred_batch)
        activations_target = self._get_activations(target_batch)
        return activations_pred, activations_target

    def get_value(self, groups=None, states=None):
        """
        :param groups:
        :return:
            total_results: dict of kind {'mean': score mean, 'std': score std}
            group_results: None, if groups is None;
                else dict {group_idx: {'mean': score mean among group, 'std': score std among group}}
        """
        if states is not None:
            target_class_freq_by_image_total, target_class_freq_by_image_mask, pred_class_freq_by_image_mask, activation_pairs = states
        else:
            target_class_freq_by_image_total = self.target_class_freq_by_image_total
            target_class_freq_by_image_mask = self.target_class_freq_by_image_mask
            pred_class_freq_by_image_mask = self.pred_class_freq_by_image_mask
            activation_pairs = self.individual_values
        target_class_freq_by_image_total = np.concatenate(target_class_freq_by_image_total, axis=0)
        target_class_freq_by_image_mask = np.concatenate(target_class_freq_by_image_mask, axis=0)
        pred_class_freq_by_image_mask = np.concatenate(pred_class_freq_by_image_mask, axis=0)
        activations_pred, activations_target = zip(*activation_pairs)
        activations_pred = np.concatenate(activations_pred, axis=0)
        activations_target = np.concatenate(activations_target, axis=0)
        total_results = {'mean': calculate_frechet_distance(activations_pred, activations_target, eps=self.eps), 'std': 0, **self.distribute_fid_to_classes(target_class_freq_by_image_mask, activations_pred, activations_target)}
        if groups is None:
            return total_results, None
        group_results = dict()
        grouping = get_groupings(groups)
        for label, index in grouping.items():
            if len(index) > 1:
                group_activations_pred = activations_pred[index]
                group_activations_target = activations_target[index]
                group_class_freq = target_class_freq_by_image_mask[index]
                group_results[label] = {'mean': calculate_frechet_distance(group_activations_pred, group_activations_target, eps=self.eps), 'std': 0, **self.distribute_fid_to_classes(group_class_freq, group_activations_pred, group_activations_target)}
            else:
                group_results[label] = dict(mean=float('nan'), std=0)
        return total_results, group_results

    def distribute_fid_to_classes(self, class_freq, activations_pred, activations_target):
        real_fid = calculate_frechet_distance(activations_pred, activations_target, eps=self.eps)
        fid_no_images = Parallel(n_jobs=self.n_jobs)(delayed(calculade_fid_no_img)(img_i, activations_pred, activations_target, eps=self.eps) for img_i in range(activations_pred.shape[0]))
        errors = real_fid - fid_no_images
        return distribute_values_to_classes(class_freq, errors, self.segm_idx2name)

    def _get_activations(self, batch):
        activations = self.model(batch)[0]
        if activations.shape[2] != 1 or activations.shape[3] != 1:
            activations = F.adaptive_avg_pool2d(activations, output_size=(1, 1))
        activations = activations.squeeze(-1).squeeze(-1).detach().cpu().numpy()
        return activations


class BaseModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def name(self):
        return 'BaseModel'

    def initialize(self, use_gpu=True):
        self.use_gpu = use_gpu

    def forward(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    def save_network(self, network, path, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(path, save_filename)
        torch.save(network.state_dict(), save_path)

    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        None
        network.load_state_dict(torch.load(save_path, map_location='cpu'))

    def update_learning_rate():
        pass

    def get_image_paths(self):
        return self.image_paths

    def save_done(self, flag=False):
        np.save(os.path.join(self.save_dir, 'done_flag'), flag)
        np.savetxt(os.path.join(self.save_dir, 'done_flag'), [flag], fmt='%i')


class Dist2LogitLayer(nn.Module):
    """ takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) """

    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()
        layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True)]
        layers += [nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True)]
        layers += [nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True)]
        if use_sigmoid:
            layers += [nn.Sigmoid()]
        self.model = nn.Sequential(*layers)

    def forward(self, d0, d1, eps=0.1):
        return self.model(torch.cat((d0, d1, d0 - d1, d0 / (d1 + eps), d1 / (d0 + eps)), dim=1))


class BCERankingLoss(nn.Module):

    def __init__(self, chn_mid=32):
        super(BCERankingLoss, self).__init__()
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        self.loss = torch.nn.BCELoss()

    def forward(self, d0, d1, judge):
        per = (judge + 1.0) / 2.0
        self.logit = self.net(d0, d1)
        return self.loss(self.logit, per)


class FakeNet(nn.Module):

    def __init__(self, use_gpu=True, colorspace='Lab'):
        super(FakeNet, self).__init__()
        self.use_gpu = use_gpu
        self.colorspace = colorspace


def dssim(p0, p1, range=255.0):
    return (1 - compare_ssim(p0, p1, data_range=range, multichannel=True)) / 2.0


def tensor2im(image_tensor, imtype=np.uint8, cent=1.0, factor=255.0 / 2.0):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + cent) * factor
    return image_numpy.astype(imtype)


def tensor2np(tensor_obj):
    return tensor_obj[0].cpu().float().numpy().transpose((1, 2, 0))


def np2tensor(np_obj):
    return torch.Tensor(np_obj[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


def tensor2tensorlab(image_tensor, to_norm=True, mc_only=False):
    img = tensor2im(image_tensor)
    img_lab = color.rgb2lab(img)
    if mc_only:
        img_lab[:, :, 0] = img_lab[:, :, 0] - 50
    if to_norm and not mc_only:
        img_lab[:, :, 0] = img_lab[:, :, 0] - 50
        img_lab = img_lab / 100.0
    return np2tensor(img_lab)


class DSSIM(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert in0.size()[0] == 1
        if self.colorspace == 'RGB':
            value = dssim(1.0 * tensor2im(in0.data), 1.0 * tensor2im(in1.data), range=255.0).astype('float')
        elif self.colorspace == 'Lab':
            value = dssim(tensor2np(tensor2tensorlab(in0.data, to_norm=False)), tensor2np(tensor2tensorlab(in1.data, to_norm=False)), range=100.0).astype('float')
        ret_var = Variable(torch.Tensor((value,)))
        return ret_var


def l2(p0, p1, range=255.0):
    return 0.5 * np.mean((p0 / range - p1 / range) ** 2)


class L2(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert in0.size()[0] == 1
        if self.colorspace == 'RGB':
            N, C, X, Y = in0.size()
            value = torch.mean(torch.mean(torch.mean((in0 - in1) ** 2, dim=1).view(N, 1, X, Y), dim=2).view(N, 1, 1, Y), dim=3).view(N)
            return value
        elif self.colorspace == 'Lab':
            value = l2(tensor2np(tensor2tensorlab(in0.data, to_norm=False)), tensor2np(tensor2tensorlab(in1.data, to_norm=False)), range=100.0).astype('float')
            ret_var = Variable(torch.Tensor((value,)))
            return ret_var


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)


class ScalingLayer(nn.Module):

    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-0.03, -0.088, -0.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.458, 0.448, 0.45])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class alexnet(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True):
        super(alexnet, self).__init__()
        alexnet_pretrained_features = tv.alexnet(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple('AlexnetOutputs', ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)
        return out


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


class squeezenet(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True):
        super(squeezenet, self).__init__()
        pretrained_features = tv.squeezenet1_1(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        self.N_slices = 7
        for x in range(2):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), pretrained_features[x])
        for x in range(10, 11):
            self.slice5.add_module(str(x), pretrained_features[x])
        for x in range(11, 12):
            self.slice6.add_module(str(x), pretrained_features[x])
        for x in range(12, 13):
            self.slice7.add_module(str(x), pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        vgg_outputs = namedtuple('SqueezeOutputs', ['relu1', 'relu2', 'relu3', 'relu4', 'relu5', 'relu6', 'relu7'])
        out = vgg_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5, h_relu6, h_relu7)
        return out


def upsample(in_tens, out_H=64):
    in_H = in_tens.shape[2]
    scale_factor = 1.0 * out_H / in_H
    return nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)(in_tens)


class PNetLin(nn.Module):

    def __init__(self, pnet_type='vgg', pnet_rand=False, pnet_tune=False, use_dropout=True, spatial=False, version='0.1', lpips=True):
        super(PNetLin, self).__init__()
        self.pnet_type = pnet_type
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips
        self.version = version
        self.scaling_layer = ScalingLayer()
        if self.pnet_type in ['vgg', 'vgg16']:
            net_type = vgg16
            self.chns = [64, 128, 256, 512, 512]
        elif self.pnet_type == 'alex':
            net_type = alexnet
            self.chns = [64, 192, 384, 256, 256]
        elif self.pnet_type == 'squeeze':
            net_type = squeezenet
            self.chns = [64, 128, 256, 384, 384, 512, 512]
        self.L = len(self.chns)
        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)
        if lpips:
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            if self.pnet_type == 'squeeze':
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins += [self.lin5, self.lin6]

    def forward(self, in0, in1, retPerLayer=False):
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version == '0.1' else (in0, in1)
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
        if self.lpips:
            if self.spatial:
                res = [upsample(self.lins[kk].model(diffs[kk]), out_H=in0.shape[2]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk].model(diffs[kk]), keepdim=True) for kk in range(self.L)]
        elif self.spatial:
            res = [upsample(diffs[kk].sum(dim=1, keepdim=True), out_H=in0.shape[2]) for kk in range(self.L)]
        else:
            res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]
        val = res[0]
        for l in range(1, self.L):
            val += res[l]
        if retPerLayer:
            return val, res
        else:
            return val


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    None
    None


class DistModel(BaseModel):

    def name(self):
        return self.model_name

    def initialize(self, model='net-lin', net='alex', colorspace='Lab', pnet_rand=False, pnet_tune=False, model_path=None, use_gpu=True, printNet=False, spatial=False, is_train=False, lr=0.0001, beta1=0.5, version='0.1'):
        """
        INPUTS
            model - ['net-lin'] for linearly calibrated network
                    ['net'] for off-the-shelf network
                    ['L2'] for L2 distance in Lab colorspace
                    ['SSIM'] for ssim in RGB colorspace
            net - ['squeeze','alex','vgg']
            model_path - if None, will look in weights/[NET_NAME].pth
            colorspace - ['Lab','RGB'] colorspace to use for L2 and SSIM
            use_gpu - bool - whether or not to use a GPU
            printNet - bool - whether or not to print network architecture out
            spatial - bool - whether to output an array containing varying distances across spatial dimensions
            spatial_shape - if given, output spatial shape. if None then spatial shape is determined automatically via spatial_factor (see below).
            spatial_factor - if given, specifies upsampling factor relative to the largest spatial extent of a convolutional layer. if None then resized to size of input images.
            spatial_order - spline order of filter for upsampling in spatial mode, by default 1 (bilinear).
            is_train - bool - [True] for training mode
            lr - float - initial learning rate
            beta1 - float - initial momentum term for adam
            version - 0.1 for latest, 0.0 was original (with a bug)
        """
        BaseModel.initialize(self, use_gpu=use_gpu)
        self.model = model
        self.net = net
        self.is_train = is_train
        self.spatial = spatial
        self.model_name = '%s [%s]' % (model, net)
        if self.model == 'net-lin':
            self.net = PNetLin(pnet_rand=pnet_rand, pnet_tune=pnet_tune, pnet_type=net, use_dropout=True, spatial=spatial, version=version, lpips=True)
            kw = dict(map_location='cpu')
            if model_path is None:
                model_path = '/ssd3/lixin41/lama-main/models/lpips_models/vgg.pth'
            if not is_train:
                self.net.load_state_dict(torch.load(model_path, **kw), strict=False)
        elif self.model == 'net':
            self.net = PNetLin(pnet_rand=pnet_rand, pnet_type=net, lpips=False)
        elif self.model in ['L2', 'l2']:
            self.net = L2(use_gpu=use_gpu, colorspace=colorspace)
            self.model_name = 'L2'
        elif self.model in ['DSSIM', 'dssim', 'SSIM', 'ssim']:
            self.net = DSSIM(use_gpu=use_gpu, colorspace=colorspace)
            self.model_name = 'SSIM'
        else:
            raise ValueError('Model [%s] not recognized.' % self.model)
        self.trainable_parameters = list(self.net.parameters())
        if self.is_train:
            self.rankLoss = BCERankingLoss()
            self.trainable_parameters += list(self.rankLoss.net.parameters())
            self.lr = lr
            self.old_lr = lr
            self.optimizer_net = torch.optim.Adam(self.trainable_parameters, lr=lr, betas=(beta1, 0.999))
        else:
            self.net.eval()
        if printNet:
            None
            print_network(self.net)
            None

    def forward(self, in0, in1, retPerLayer=False):
        """ Function computes the distance between image patches in0 and in1
        INPUTS
            in0, in1 - torch.Tensor object of shape Nx3xXxY - image patch scaled to [-1,1]
        OUTPUT
            computed distances between in0 and in1
        """
        return self.net(in0, in1, retPerLayer=retPerLayer)

    def optimize_parameters(self):
        self.forward_train()
        self.optimizer_net.zero_grad()
        self.backward_train()
        self.optimizer_net.step()
        self.clamp_weights()

    def clamp_weights(self):
        for module in self.net.modules():
            if hasattr(module, 'weight') and module.kernel_size == (1, 1):
                module.weight.data = torch.clamp(module.weight.data, min=0)

    def set_input(self, data):
        self.input_ref = data['ref']
        self.input_p0 = data['p0']
        self.input_p1 = data['p1']
        self.input_judge = data['judge']

    def forward_train(self):
        assert False, "We shoud've not get here when using LPIPS as a metric"
        self.d0 = self(self.var_ref, self.var_p0)
        self.d1 = self(self.var_ref, self.var_p1)
        self.acc_r = self.compute_accuracy(self.d0, self.d1, self.input_judge)
        self.var_judge = Variable(1.0 * self.input_judge).view(self.d0.size())
        self.loss_total = self.rankLoss(self.d0, self.d1, self.var_judge * 2.0 - 1.0)
        return self.loss_total

    def backward_train(self):
        torch.mean(self.loss_total).backward()

    def compute_accuracy(self, d0, d1, judge):
        """ d0, d1 are Variables, judge is a Tensor """
        d1_lt_d0 = (d1 < d0).cpu().data.numpy().flatten()
        judge_per = judge.cpu().numpy().flatten()
        return d1_lt_d0 * judge_per + (1 - d1_lt_d0) * (1 - judge_per)

    def get_current_errors(self):
        retDict = OrderedDict([('loss_total', self.loss_total.data.cpu().numpy()), ('acc_r', self.acc_r)])
        for key in retDict.keys():
            retDict[key] = np.mean(retDict[key])
        return retDict

    def get_current_visuals(self):
        zoom_factor = 256 / self.var_ref.data.size()[2]
        ref_img = tensor2im(self.var_ref.data)
        p0_img = tensor2im(self.var_p0.data)
        p1_img = tensor2im(self.var_p1.data)
        ref_img_vis = zoom(ref_img, [zoom_factor, zoom_factor, 1], order=0)
        p0_img_vis = zoom(p0_img, [zoom_factor, zoom_factor, 1], order=0)
        p1_img_vis = zoom(p1_img, [zoom_factor, zoom_factor, 1], order=0)
        return OrderedDict([('ref', ref_img_vis), ('p0', p0_img_vis), ('p1', p1_img_vis)])

    def save(self, path, label):
        if self.use_gpu:
            self.save_network(self.net.module, path, '', label)
        else:
            self.save_network(self.net, path, '', label)
        self.save_network(self.rankLoss.net, path, 'rank', label)

    def update_learning_rate(self, nepoch_decay):
        lrd = self.lr / nepoch_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_net.param_groups:
            param_group['lr'] = lr
        None
        self.old_lr = lr


class vgg16(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class resnet(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True, num=18):
        super(resnet, self).__init__()
        if num == 18:
            self.net = tv.resnet18(pretrained=pretrained)
        elif num == 34:
            self.net = tv.resnet34(pretrained=pretrained)
        elif num == 50:
            self.net = tv.resnet50(pretrained=pretrained)
        elif num == 101:
            self.net = tv.resnet101(pretrained=pretrained)
        elif num == 152:
            self.net = tv.resnet152(pretrained=pretrained)
        self.N_slices = 5
        self.conv1 = self.net.conv1
        self.bn1 = self.net.bn1
        self.relu = self.net.relu
        self.maxpool = self.net.maxpool
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4

    def forward(self, X):
        h = self.conv1(X)
        h = self.bn1(h)
        h = self.relu(h)
        h_relu1 = h
        h = self.maxpool(h)
        h = self.layer1(h)
        h_conv2 = h
        h = self.layer2(h)
        h_conv3 = h
        h = self.layer3(h)
        h_conv4 = h
        h = self.layer4(h)
        h_conv5 = h
        outputs = namedtuple('Outputs', ['relu1', 'conv2', 'conv3', 'conv4', 'conv5'])
        out = outputs(h_relu1, h_conv2, h_conv3, h_conv4, h_conv5)
        return out


def get_gauss_kernel(kernel_size, width_factor=1):
    coords = torch.stack(torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size)), dim=0).float()
    diff = torch.exp(-((coords - kernel_size // 2) ** 2).sum(0) / kernel_size / width_factor)
    diff /= diff.sum()
    return diff


class BlurMask(nn.Module):

    def __init__(self, kernel_size=5, width_factor=1):
        super().__init__()
        self.filter = nn.Conv2d(1, 1, kernel_size, padding=kernel_size // 2, padding_mode='replicate', bias=False)
        self.filter.weight.data.copy_(get_gauss_kernel(kernel_size, width_factor=width_factor))

    def forward(self, real_img, pred_img, mask):
        with torch.no_grad():
            result = self.filter(mask) * mask
            return result


class EmulatedEDTMask(nn.Module):

    def __init__(self, dilate_kernel_size=5, blur_kernel_size=5, width_factor=1):
        super().__init__()
        self.dilate_filter = nn.Conv2d(1, 1, dilate_kernel_size, padding=dilate_kernel_size // 2, padding_mode='replicate', bias=False)
        self.dilate_filter.weight.data.copy_(torch.ones(1, 1, dilate_kernel_size, dilate_kernel_size, dtype=torch.float))
        self.blur_filter = nn.Conv2d(1, 1, blur_kernel_size, padding=blur_kernel_size // 2, padding_mode='replicate', bias=False)
        self.blur_filter.weight.data.copy_(get_gauss_kernel(blur_kernel_size, width_factor=width_factor))

    def forward(self, real_img, pred_img, mask):
        with torch.no_grad():
            known_mask = 1 - mask
            dilated_known_mask = (self.dilate_filter(known_mask) > 1).float()
            result = self.blur_filter(1 - dilated_known_mask) * mask
            return result


IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]


IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]


class PropagatePerceptualSim(nn.Module):

    def __init__(self, level=2, max_iters=10, temperature=500, erode_mask_size=3):
        super().__init__()
        vgg = torchvision.models.vgg19(pretrained=True).features
        vgg_avg_pooling = []
        for weights in vgg.parameters():
            weights.requires_grad = False
        cur_level_i = 0
        for module in vgg.modules():
            if module.__class__.__name__ == 'Sequential':
                continue
            elif module.__class__.__name__ == 'MaxPool2d':
                vgg_avg_pooling.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                vgg_avg_pooling.append(module)
                if module.__class__.__name__ == 'ReLU':
                    cur_level_i += 1
                if cur_level_i == level:
                    break
        self.features = nn.Sequential(*vgg_avg_pooling)
        self.max_iters = max_iters
        self.temperature = temperature
        self.do_erode = erode_mask_size > 0
        if self.do_erode:
            self.erode_mask = nn.Conv2d(1, 1, erode_mask_size, padding=erode_mask_size // 2, bias=False)
            self.erode_mask.weight.data.fill_(1)

    def forward(self, real_img, pred_img, mask):
        with torch.no_grad():
            real_img = (real_img - IMAGENET_MEAN) / IMAGENET_STD
            real_feats = self.features(real_img)
            vertical_sim = torch.exp(-(real_feats[:, :, 1:] - real_feats[:, :, :-1]).pow(2).sum(1, keepdim=True) / self.temperature)
            horizontal_sim = torch.exp(-(real_feats[:, :, :, 1:] - real_feats[:, :, :, :-1]).pow(2).sum(1, keepdim=True) / self.temperature)
            mask_scaled = F.interpolate(mask, size=real_feats.shape[-2:], mode='bilinear', align_corners=False)
            if self.do_erode:
                mask_scaled = (self.erode_mask(mask_scaled) > 1).float()
            cur_knowness = 1 - mask_scaled
            for iter_i in range(self.max_iters):
                new_top_knowness = F.pad(cur_knowness[:, :, :-1] * vertical_sim, (0, 0, 1, 0), mode='replicate')
                new_bottom_knowness = F.pad(cur_knowness[:, :, 1:] * vertical_sim, (0, 0, 0, 1), mode='replicate')
                new_left_knowness = F.pad(cur_knowness[:, :, :, :-1] * horizontal_sim, (1, 0, 0, 0), mode='replicate')
                new_right_knowness = F.pad(cur_knowness[:, :, :, 1:] * horizontal_sim, (0, 1, 0, 0), mode='replicate')
                new_knowness = torch.stack([new_top_knowness, new_bottom_knowness, new_left_knowness, new_right_knowness], dim=0).max(0).values
                cur_knowness = torch.max(cur_knowness, new_knowness)
            cur_knowness = F.interpolate(cur_knowness, size=mask.shape[-2:], mode='bilinear')
            result = torch.min(mask, 1 - cur_knowness)
            return result


class ResNetPL(nn.Module):

    def __init__(self, weight=1, weights_path=None, arch_encoder='resnet50dilated', segmentation=True):
        super().__init__()
        self.impl = ModelBuilder.get_encoder(weights_path=weights_path, arch_encoder=arch_encoder, arch_decoder='ppm_deepsup', fc_dim=2048, segmentation=segmentation)
        self.impl.eval()
        for w in self.impl.parameters():
            w.requires_grad_(False)
        self.weight = weight

    def forward(self, pred, target):
        pred = (pred - IMAGENET_MEAN) / IMAGENET_STD
        target = (target - IMAGENET_MEAN) / IMAGENET_STD
        pred_feats = self.impl(pred, return_feature_maps=True)
        target_feats = self.impl(target, return_feature_maps=True)
        result = torch.stack([F.mse_loss(cur_pred, cur_target) for cur_pred, cur_target in zip(pred_feats, target_feats)]).sum() * self.weight
        return result


class CrossEntropy2d(nn.Module):

    def __init__(self, reduction='mean', ignore_label=255, weights=None, *args, **kwargs):
        """
        weight (Tensor, optional): a manual rescaling weight given to each class.
            If given, has to be a Tensor of size "nclasses"
        """
        super(CrossEntropy2d, self).__init__()
        self.reduction = reduction
        self.ignore_label = ignore_label
        self.weights = weights
        if self.weights is not None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.weights = torch.FloatTensor(constant_weights[weights])

    def forward(self, predict, target):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, 1, h, w)
        """
        target = target.long()
        assert not target.requires_grad
        assert predict.dim() == 4, '{0}'.format(predict.size())
        assert target.dim() == 4, '{0}'.format(target.size())
        assert predict.size(0) == target.size(0), '{0} vs {1} '.format(predict.size(0), target.size(0))
        assert target.size(1) == 1, '{0}'.format(target.size(1))
        assert predict.size(2) == target.size(2), '{0} vs {1} '.format(predict.size(2), target.size(2))
        assert predict.size(3) == target.size(3), '{0} vs {1} '.format(predict.size(3), target.size(3))
        target = target.squeeze(1)
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=self.weights, reduction=self.reduction)
        return loss


class BaseDiscriminator(nn.Module):

    @abc.abstractmethod
    def forward(self, x: 'torch.Tensor') ->Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Predict scores and get intermediate activations. Useful for feature matching loss
        :return tuple (scores, list of intermediate activations)
        """
        raise NotImplemented()


class SimpleMultiStepGenerator(nn.Module):

    def __init__(self, steps: 'List[nn.Module]'):
        super().__init__()
        self.steps = nn.ModuleList(steps)

    def forward(self, x):
        cur_in = x
        outs = []
        for step in self.steps:
            cur_out = step(cur_in)
            outs.append(cur_out)
            cur_in = torch.cat((cur_in, cur_out), dim=1)
        return torch.cat(outs[::-1], dim=1)


class DepthWiseSeperableConv(nn.Module):

    def __init__(self, in_dim, out_dim, *args, **kwargs):
        super().__init__()
        if 'groups' in kwargs:
            del kwargs['groups']
        self.depthwise = nn.Conv2d(in_dim, in_dim, *args, groups=in_dim, **kwargs)
        self.pointwise = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class FFCSE_block(nn.Module):

    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r, kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x
        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))
        x_l = 0 if self.conv_a2l is None else id_l * self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * self.sigmoid(self.conv_a2g(x))
        return x_l, x_g


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        res = x * y.expand_as(x)
        return res


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear', spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0), out_channels=out_channels * 2, kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)
        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]
        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)
        r_size = x.size()
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = rfftn(x, s=None, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1) + ffted.size()[3:])
        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)
        if self.use_se:
            ffted = self.se(ffted)
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))
        ffted = ffted.view((batch, -1, 2) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)
        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()
        self.stride = stride
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False), nn.BatchNorm2d(out_channels // 2), nn.ReLU(inplace=True))
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)
        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0
        output = self.conv2(x + output + xs)
        return output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride=1, padding=0, dilation=1, groups=1, bias=False, enable_lfu=True, padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()
        assert stride == 1 or stride == 2, 'Stride should be 1 or 2.'
        self.stride = stride
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg
        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)
        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0
        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)
            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1
        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)
        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride=1, padding=0, dilation=1, groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity, padding_type='reflect', enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride, padding, dilation, groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)
        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class LearnableSpatialTransformWrapper(nn.Module):

    def __init__(self, impl, pad_coef=0.5, angle_init_range=80, train_angle=True):
        super().__init__()
        self.impl = impl
        self.angle = torch.rand(1) * angle_init_range
        if train_angle:
            self.angle = nn.Parameter(self.angle, requires_grad=True)
        self.pad_coef = pad_coef

    def forward(self, x):
        if torch.is_tensor(x):
            return self.inverse_transform(self.impl(self.transform(x)), x)
        elif isinstance(x, tuple):
            x_trans = tuple(self.transform(elem) for elem in x)
            y_trans = self.impl(x_trans)
            return tuple(self.inverse_transform(elem, orig_x) for elem, orig_x in zip(y_trans, x))
        else:
            raise ValueError(f'Unexpected input type {type(x)}')

    def transform(self, x):
        height, width = x.shape[2:]
        pad_h, pad_w = int(height * self.pad_coef), int(width * self.pad_coef)
        x_padded = F.pad(x, [pad_w, pad_w, pad_h, pad_h], mode='reflect')
        x_padded_rotated = rotate(x_padded, angle=self.angle)
        return x_padded_rotated

    def inverse_transform(self, y_padded_rotated, orig_x):
        height, width = orig_x.shape[2:]
        pad_h, pad_w = int(height * self.pad_coef), int(width * self.pad_coef)
        y_padded = rotate(y_padded_rotated, angle=-self.angle)
        y_height, y_width = y_padded.shape[2:]
        y = y_padded[:, :, pad_h:y_height - pad_h, pad_w:y_width - pad_w]
        return y


class FFCResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1, spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation, norm_layer=norm_layer, activation_layer=activation_layer, padding_type=padding_type, **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation, norm_layer=norm_layer, activation_layer=activation_layer, padding_type=padding_type, **conv_kwargs)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out


class ConcatTupleLayer(nn.Module):

    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)


def get_activation(kind='tanh'):
    if kind == 'tanh':
        return nn.Tanh()
    if kind == 'sigmoid':
        return nn.Sigmoid()
    if kind is False:
        return nn.Identity()
    raise ValueError(f'Unknown activation kind {kind}')


class FFCResNetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type='reflect', activation_layer=nn.ReLU, up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True), init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={}, spatial_transform_layers=None, spatial_transform_kwargs={}, add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert n_blocks >= 0
        super().__init__()
        model = [nn.ReflectionPad2d(3), FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer, activation_layer=activation_layer, **init_conv_kwargs)]
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult), min(max_features, ngf * mult * 2), kernel_size=3, stride=2, padding=1, norm_layer=norm_layer, activation_layer=activation_layer, **cur_conv_kwargs)]
        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer, norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]
        model += [ConcatTupleLayer()]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult), min(max_features, int(ngf * mult / 2)), kernel_size=3, stride=2, padding=1, output_padding=1), up_norm_layer(min(max_features, int(ngf * mult / 2))), up_activation]
        if out_ffc:
            model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer, norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class FFCNLayerDiscriminator(BaseDiscriminator):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, max_features=512, init_conv_kwargs={}, conv_kwargs={}):
        super().__init__()
        self.n_layers = n_layers

        def _act_ctor(inplace=True):
            return nn.LeakyReLU(negative_slope=0.2, inplace=inplace)
        kw = 3
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[FFC_BN_ACT(input_nc, ndf, kernel_size=kw, padding=padw, norm_layer=norm_layer, activation_layer=_act_ctor, **init_conv_kwargs)]]
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, max_features)
            cur_model = [FFC_BN_ACT(nf_prev, nf, kernel_size=kw, stride=2, padding=padw, norm_layer=norm_layer, activation_layer=_act_ctor, **conv_kwargs)]
            sequence.append(cur_model)
        nf_prev = nf
        nf = min(nf * 2, 512)
        cur_model = [FFC_BN_ACT(nf_prev, nf, kernel_size=kw, stride=1, padding=padw, norm_layer=norm_layer, activation_layer=lambda *args, **kwargs: nn.LeakyReLU(*args, negative_slope=0.2, **kwargs), **conv_kwargs), ConcatTupleLayer()]
        sequence.append(cur_model)
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        for n in range(len(sequence)):
            setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, x):
        act = self.get_all_activations(x)
        feats = []
        for out in act[:-1]:
            if isinstance(out, tuple):
                if torch.is_tensor(out[1]):
                    out = torch.cat(out, dim=1)
                else:
                    out = out[0]
            feats.append(out)
        return act[-1], feats


class MultidilatedConv(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size, dilation_num=3, comb_mode='sum', equal_dim=True, shared_weights=False, padding=1, min_dilation=1, shuffle_in_channels=False, use_depthwise=False, **kwargs):
        super().__init__()
        convs = []
        self.equal_dim = equal_dim
        assert comb_mode in ('cat_out', 'sum', 'cat_in', 'cat_both'), comb_mode
        if comb_mode in ('cat_out', 'cat_both'):
            self.cat_out = True
            if equal_dim:
                assert out_dim % dilation_num == 0
                out_dims = [out_dim // dilation_num] * dilation_num
                self.index = sum([[(i + j * out_dims[0]) for j in range(dilation_num)] for i in range(out_dims[0])], [])
            else:
                out_dims = [(out_dim // 2 ** (i + 1)) for i in range(dilation_num - 1)]
                out_dims.append(out_dim - sum(out_dims))
                index = []
                starts = [0] + out_dims[:-1]
                lengths = [(out_dims[i] // out_dims[-1]) for i in range(dilation_num)]
                for i in range(out_dims[-1]):
                    for j in range(dilation_num):
                        index += list(range(starts[j], starts[j] + lengths[j]))
                        starts[j] += lengths[j]
                self.index = index
                assert len(index) == out_dim
            self.out_dims = out_dims
        else:
            self.cat_out = False
            self.out_dims = [out_dim] * dilation_num
        if comb_mode in ('cat_in', 'cat_both'):
            if equal_dim:
                assert in_dim % dilation_num == 0
                in_dims = [in_dim // dilation_num] * dilation_num
            else:
                in_dims = [(in_dim // 2 ** (i + 1)) for i in range(dilation_num - 1)]
                in_dims.append(in_dim - sum(in_dims))
            self.in_dims = in_dims
            self.cat_in = True
        else:
            self.cat_in = False
            self.in_dims = [in_dim] * dilation_num
        conv_type = DepthWiseSeperableConv if use_depthwise else nn.Conv2d
        dilation = min_dilation
        for i in range(dilation_num):
            if isinstance(padding, int):
                cur_padding = padding * dilation
            else:
                cur_padding = padding[i]
            convs.append(conv_type(self.in_dims[i], self.out_dims[i], kernel_size, padding=cur_padding, dilation=dilation, **kwargs))
            if i > 0 and shared_weights:
                convs[-1].weight = convs[0].weight
                convs[-1].bias = convs[0].bias
            dilation *= 2
        self.convs = nn.ModuleList(convs)
        self.shuffle_in_channels = shuffle_in_channels
        if self.shuffle_in_channels:
            in_channels_permute = list(range(in_dim))
            random.shuffle(in_channels_permute)
            self.register_buffer('in_channels_permute', torch.tensor(in_channels_permute))

    def forward(self, x):
        if self.shuffle_in_channels:
            x = x[:, self.in_channels_permute]
        outs = []
        if self.cat_in:
            if self.equal_dim:
                x = x.chunk(len(self.convs), dim=1)
            else:
                new_x = []
                start = 0
                for dim in self.in_dims:
                    new_x.append(x[:, start:start + dim])
                    start += dim
                x = new_x
        for i, conv in enumerate(self.convs):
            if self.cat_in:
                input = x[i]
            else:
                input = x
            outs.append(conv(input))
        if self.cat_out:
            out = torch.cat(outs, dim=1)[:, self.index]
        else:
            out = sum(outs)
        return out


def get_conv_block_ctor(kind='default'):
    if not isinstance(kind, str):
        return kind
    if kind == 'default':
        return nn.Conv2d
    if kind == 'depthwise':
        return DepthWiseSeperableConv
    if kind == 'multidilated':
        return MultidilatedConv
    raise ValueError(f'Unknown convolutional block kind {kind}')


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False, conv_kind='default', dilation=1, in_dim=None, groups=1, second_dilation=None):
        super(ResnetBlock, self).__init__()
        self.in_dim = in_dim
        self.dim = dim
        if second_dilation is None:
            second_dilation = dilation
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout, conv_kind=conv_kind, dilation=dilation, in_dim=in_dim, groups=groups, second_dilation=second_dilation)
        if self.in_dim is not None:
            self.input_conv = nn.Conv2d(in_dim, dim, 1)
        self.out_channnels = dim

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout, conv_kind='default', dilation=1, in_dim=None, groups=1, second_dilation=1):
        conv_layer = get_conv_block_ctor(conv_kind)
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(dilation)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(dilation)]
        elif padding_type == 'zero':
            p = dilation
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        if in_dim is None:
            in_dim = dim
        conv_block += [conv_layer(in_dim, dim, kernel_size=3, padding=p, dilation=dilation), norm_layer(dim), activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(second_dilation)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(second_dilation)]
        elif padding_type == 'zero':
            p = second_dilation
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [conv_layer(dim, dim, kernel_size=3, padding=p, dilation=second_dilation, groups=groups), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        x_before = x
        if self.in_dim is not None:
            x = self.input_conv(x)
        out = x + self.conv_block(x_before)
        return out


class ResNetHead(nn.Module):

    def __init__(self, input_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type='reflect', conv_kind='default', activation=nn.ReLU(True)):
        assert n_blocks >= 0
        super(ResNetHead, self).__init__()
        conv_layer = get_conv_block_ctor(conv_kind)
        model = [nn.ReflectionPad2d(3), conv_layer(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [conv_layer(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer, conv_kind=conv_kind)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResNetTail(nn.Module):

    def __init__(self, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type='reflect', conv_kind='default', activation=nn.ReLU(True), up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True), add_out_act=False, out_extra_layers_n=0, add_in_proj=None):
        assert n_blocks >= 0
        super(ResNetTail, self).__init__()
        mult = 2 ** n_downsampling
        model = []
        if add_in_proj is not None:
            model.append(nn.Conv2d(add_in_proj, ngf * mult, kernel_size=1))
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer, conv_kind=conv_kind)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1), up_norm_layer(int(ngf * mult / 2)), up_activation]
        self.model = nn.Sequential(*model)
        out_layers = []
        for _ in range(out_extra_layers_n):
            out_layers += [nn.Conv2d(ngf, ngf, kernel_size=1, padding=0), up_norm_layer(ngf), up_activation]
        out_layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            out_layers.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.out_proj = nn.Sequential(*out_layers)

    def forward(self, input, return_last_act=False):
        features = self.model(input)
        out = self.out_proj(features)
        if return_last_act:
            return out, features
        else:
            return out


class MultiscaleResNet(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=2, n_blocks_head=2, n_blocks_tail=6, n_scales=3, norm_layer=nn.BatchNorm2d, padding_type='reflect', conv_kind='default', activation=nn.ReLU(True), up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True), add_out_act=False, out_extra_layers_n=0, out_cumulative=False, return_only_hr=False):
        super().__init__()
        self.heads = nn.ModuleList([ResNetHead(input_nc, ngf=ngf, n_downsampling=n_downsampling, n_blocks=n_blocks_head, norm_layer=norm_layer, padding_type=padding_type, conv_kind=conv_kind, activation=activation) for i in range(n_scales)])
        tail_in_feats = ngf * 2 ** n_downsampling + ngf
        self.tails = nn.ModuleList([ResNetTail(output_nc, ngf=ngf, n_downsampling=n_downsampling, n_blocks=n_blocks_tail, norm_layer=norm_layer, padding_type=padding_type, conv_kind=conv_kind, activation=activation, up_norm_layer=up_norm_layer, up_activation=up_activation, add_out_act=add_out_act, out_extra_layers_n=out_extra_layers_n, add_in_proj=None if i == n_scales - 1 else tail_in_feats) for i in range(n_scales)])
        self.out_cumulative = out_cumulative
        self.return_only_hr = return_only_hr

    @property
    def num_scales(self):
        return len(self.heads)

    def forward(self, ms_inputs: 'List[torch.Tensor]', smallest_scales_num: 'Optional[int]'=None) ->Union[torch.Tensor, List[torch.Tensor]]:
        """
        :param ms_inputs: List of inputs of different resolutions from HR to LR
        :param smallest_scales_num: int or None, number of smallest scales to take at input
        :return: Depending on return_only_hr:
            True: Only the most HR output
            False: List of outputs of different resolutions from HR to LR
        """
        if smallest_scales_num is None:
            assert len(self.heads) == len(ms_inputs), (len(self.heads), len(ms_inputs), smallest_scales_num)
            smallest_scales_num = len(self.heads)
        else:
            assert smallest_scales_num == len(ms_inputs) <= len(self.heads), (len(self.heads), len(ms_inputs), smallest_scales_num)
        cur_heads = self.heads[-smallest_scales_num:]
        ms_features = [cur_head(cur_inp) for cur_head, cur_inp in zip(cur_heads, ms_inputs)]
        all_outputs = []
        prev_tail_features = None
        for i in range(len(ms_features)):
            scale_i = -i - 1
            cur_tail_input = ms_features[-i - 1]
            if prev_tail_features is not None:
                if prev_tail_features.shape != cur_tail_input.shape:
                    prev_tail_features = F.interpolate(prev_tail_features, size=cur_tail_input.shape[2:], mode='bilinear', align_corners=False)
                cur_tail_input = torch.cat((cur_tail_input, prev_tail_features), dim=1)
            cur_out, cur_tail_feats = self.tails[scale_i](cur_tail_input, return_last_act=True)
            prev_tail_features = cur_tail_feats
            all_outputs.append(cur_out)
        if self.out_cumulative:
            all_outputs_cum = [all_outputs[0]]
            for i in range(1, len(ms_features)):
                cur_out = all_outputs[i]
                cur_out_cum = cur_out + F.interpolate(all_outputs_cum[-1], size=cur_out.shape[2:], mode='bilinear', align_corners=False)
                all_outputs_cum.append(cur_out_cum)
            all_outputs = all_outputs_cum
        if self.return_only_hr:
            return all_outputs[-1]
        else:
            return all_outputs[::-1]


class MultiscaleDiscriminatorSimple(nn.Module):

    def __init__(self, ms_impl):
        super().__init__()
        self.ms_impl = nn.ModuleList(ms_impl)

    @property
    def num_scales(self):
        return len(self.ms_impl)

    def forward(self, ms_inputs: 'List[torch.Tensor]', smallest_scales_num: 'Optional[int]'=None) ->List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        :param ms_inputs: List of inputs of different resolutions from HR to LR
        :param smallest_scales_num: int or None, number of smallest scales to take at input
        :return: List of pairs (prediction, features) for different resolutions from HR to LR
        """
        if smallest_scales_num is None:
            assert len(self.ms_impl) == len(ms_inputs), (len(self.ms_impl), len(ms_inputs), smallest_scales_num)
            smallest_scales_num = len(self.heads)
        else:
            assert smallest_scales_num == len(ms_inputs) <= len(self.ms_impl), (len(self.ms_impl), len(ms_inputs), smallest_scales_num)
        return [cur_discr(cur_input) for cur_discr, cur_input in zip(self.ms_impl[-smallest_scales_num:], ms_inputs)]


class DiscriminatorMultiToSingleOutputStackedMixin:

    def __init__(self, *args, return_feats_only_levels=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_feats_only_levels = return_feats_only_levels

    def forward(self, x):
        out_feat_tuples = super().forward(x)
        outs = [out for out, _ in out_feat_tuples]
        scaled_outs = [outs[0]] + [F.interpolate(cur_out, size=outs[0].shape[-2:], mode='bilinear', align_corners=False) for cur_out in outs[1:]]
        out = torch.cat(scaled_outs, dim=1)
        if self.return_feats_only_levels is not None:
            feat_lists = [out_feat_tuples[i][1] for i in self.return_feats_only_levels]
        else:
            feat_lists = [flist for _, flist in out_feat_tuples]
        feats = [f for flist in feat_lists for f in flist]
        return out, feats


class SingleToMultiScaleInputMixin:

    def forward(self, x: 'torch.Tensor') ->List:
        orig_height, orig_width = x.shape[2:]
        factors = [(2 ** i) for i in range(self.num_scales)]
        ms_inputs = [F.interpolate(x, size=(orig_height // f, orig_width // f), mode='bilinear', align_corners=False) for f in factors]
        return super().forward(ms_inputs)


class MultiscaleDiscrSingleInput(SingleToMultiScaleInputMixin, DiscriminatorMultiToSingleOutputStackedMixin, MultiscaleDiscriminatorSimple):
    pass


class GeneratorMultiToSingleOutputMixin:

    def forward(self, x):
        return super().forward(x)[0]


class MultiscaleResNetSingle(GeneratorMultiToSingleOutputMixin, SingleToMultiScaleInputMixin, MultiscaleResNet):
    pass


class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ResnetBlock5x5(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False, conv_kind='default', dilation=1, in_dim=None, groups=1, second_dilation=None):
        super(ResnetBlock5x5, self).__init__()
        self.in_dim = in_dim
        self.dim = dim
        if second_dilation is None:
            second_dilation = dilation
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout, conv_kind=conv_kind, dilation=dilation, in_dim=in_dim, groups=groups, second_dilation=second_dilation)
        if self.in_dim is not None:
            self.input_conv = nn.Conv2d(in_dim, dim, 1)
        self.out_channnels = dim

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout, conv_kind='default', dilation=1, in_dim=None, groups=1, second_dilation=1):
        conv_layer = get_conv_block_ctor(conv_kind)
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(dilation * 2)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(dilation * 2)]
        elif padding_type == 'zero':
            p = dilation * 2
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        if in_dim is None:
            in_dim = dim
        conv_block += [conv_layer(in_dim, dim, kernel_size=5, padding=p, dilation=dilation), norm_layer(dim), activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(second_dilation * 2)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(second_dilation * 2)]
        elif padding_type == 'zero':
            p = second_dilation * 2
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [conv_layer(dim, dim, kernel_size=5, padding=p, dilation=second_dilation, groups=groups), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        x_before = x
        if self.in_dim is not None:
            x = self.input_conv(x)
        out = x + self.conv_block(x_before)
        return out


class MultidilatedResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, conv_layer, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, conv_layer, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, conv_layer, norm_layer, activation, use_dropout, dilation=1):
        conv_block = []
        conv_block += [conv_layer(dim, dim, kernel_size=3, padding_mode=padding_type), norm_layer(dim), activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [conv_layer(dim, dim, kernel_size=3, padding_mode=padding_type), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


def deconv_factory(kind, ngf, mult, norm_layer, activation, max_features):
    if kind == 'convtranspose':
        return [nn.ConvTranspose2d(min(max_features, ngf * mult), min(max_features, int(ngf * mult / 2)), kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(min(max_features, int(ngf * mult / 2))), activation]
    elif kind == 'bilinear':
        return [nn.Upsample(scale_factor=2, mode='bilinear'), DepthWiseSeperableConv(min(max_features, ngf * mult), min(max_features, int(ngf * mult / 2)), kernel_size=3, stride=1, padding=1), norm_layer(min(max_features, int(ngf * mult / 2))), activation]
    else:
        raise Exception(f'Invalid deconv kind: {kind}')


def get_norm_layer(kind='bn'):
    if not isinstance(kind, str):
        return kind
    if kind == 'bn':
        return nn.BatchNorm2d
    if kind == 'in':
        return nn.InstanceNorm2d
    raise ValueError(f'Unknown norm block kind {kind}')


class MultiDilatedGlobalGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=3, norm_layer=nn.BatchNorm2d, padding_type='reflect', conv_kind='default', deconv_kind='convtranspose', activation=nn.ReLU(True), up_norm_layer=nn.BatchNorm2d, affine=None, up_activation=nn.ReLU(True), add_out_act=True, max_features=1024, multidilation_kwargs={}, ffc_positions=None, ffc_kwargs={}):
        assert n_blocks >= 0
        super().__init__()
        conv_layer = get_conv_block_ctor(conv_kind)
        resnet_conv_layer = functools.partial(get_conv_block_ctor('multidilated'), **multidilation_kwargs)
        norm_layer = get_norm_layer(norm_layer)
        if affine is not None:
            norm_layer = partial(norm_layer, affine=affine)
        up_norm_layer = get_norm_layer(up_norm_layer)
        if affine is not None:
            up_norm_layer = partial(up_norm_layer, affine=affine)
        model = [nn.ReflectionPad2d(3), conv_layer(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        identity = Identity()
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [conv_layer(min(max_features, ngf * mult), min(max_features, ngf * mult * 2), kernel_size=3, stride=2, padding=1), norm_layer(min(max_features, ngf * mult * 2)), activation]
        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)
        for i in range(n_blocks):
            if ffc_positions is not None and i in ffc_positions:
                model += [FFCResnetBlock(feats_num_bottleneck, padding_type, norm_layer, activation_layer=nn.ReLU, inline=True, **ffc_kwargs)]
            model += [MultidilatedResnetBlock(feats_num_bottleneck, padding_type=padding_type, conv_layer=resnet_conv_layer, activation=activation, norm_layer=norm_layer)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += deconv_factory(deconv_kind, ngf, mult, up_norm_layer, up_activation, max_features)
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class DotDict(defaultdict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = defaultdict.get
    __setattr__ = defaultdict.__setitem__
    __delattr__ = defaultdict.__delitem__


class ConfigGlobalGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=3, norm_layer=nn.BatchNorm2d, padding_type='reflect', conv_kind='default', deconv_kind='convtranspose', activation=nn.ReLU(True), up_norm_layer=nn.BatchNorm2d, affine=None, up_activation=nn.ReLU(True), add_out_act=True, max_features=1024, manual_block_spec=[], resnet_block_kind='multidilatedresnetblock', resnet_conv_kind='multidilated', resnet_dilation=1, multidilation_kwargs={}):
        assert n_blocks >= 0
        super().__init__()
        conv_layer = get_conv_block_ctor(conv_kind)
        resnet_conv_layer = functools.partial(get_conv_block_ctor(resnet_conv_kind), **multidilation_kwargs)
        norm_layer = get_norm_layer(norm_layer)
        if affine is not None:
            norm_layer = partial(norm_layer, affine=affine)
        up_norm_layer = get_norm_layer(up_norm_layer)
        if affine is not None:
            up_norm_layer = partial(up_norm_layer, affine=affine)
        model = [nn.ReflectionPad2d(3), conv_layer(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        identity = Identity()
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [conv_layer(min(max_features, ngf * mult), min(max_features, ngf * mult * 2), kernel_size=3, stride=2, padding=1), norm_layer(min(max_features, ngf * mult * 2)), activation]
        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)
        if len(manual_block_spec) == 0:
            manual_block_spec = [DotDict(lambda : None, {'n_blocks': n_blocks, 'use_default': True})]
        for block_spec in manual_block_spec:

            def make_and_add_blocks(model, block_spec):
                block_spec = DotDict(lambda : None, block_spec)
                if not block_spec.use_default:
                    resnet_conv_layer = functools.partial(get_conv_block_ctor(block_spec.resnet_conv_kind), **block_spec.multidilation_kwargs)
                    resnet_conv_kind = block_spec.resnet_conv_kind
                    resnet_block_kind = block_spec.resnet_block_kind
                    if block_spec.resnet_dilation is not None:
                        resnet_dilation = block_spec.resnet_dilation
                for i in range(block_spec.n_blocks):
                    if resnet_block_kind == 'multidilatedresnetblock':
                        model += [MultidilatedResnetBlock(feats_num_bottleneck, padding_type=padding_type, conv_layer=resnet_conv_layer, activation=activation, norm_layer=norm_layer)]
                    if resnet_block_kind == 'resnetblock':
                        model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer, conv_kind=resnet_conv_kind)]
                    if resnet_block_kind == 'resnetblock5x5':
                        model += [ResnetBlock5x5(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer, conv_kind=resnet_conv_kind)]
                    if resnet_block_kind == 'resnetblockdwdil':
                        model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer, conv_kind=resnet_conv_kind, dilation=resnet_dilation, second_dilation=resnet_dilation)]
            make_and_add_blocks(model, block_spec)
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += deconv_factory(deconv_kind, ngf, mult, up_norm_layer, up_activation, max_features)
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


def make_dil_blocks(dilated_blocks_n, dilation_block_kind, dilated_block_kwargs):
    blocks = []
    for i in range(dilated_blocks_n):
        if dilation_block_kind == 'simple':
            blocks.append(ResnetBlock(**dilated_block_kwargs, dilation=2 ** (i + 1)))
        elif dilation_block_kind == 'multi':
            blocks.append(MultidilatedResnetBlock(**dilated_block_kwargs))
        else:
            raise ValueError(f'dilation_block_kind could not be "{dilation_block_kind}"')
    return blocks


class GlobalGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type='reflect', conv_kind='default', activation=nn.ReLU(True), up_norm_layer=nn.BatchNorm2d, affine=None, up_activation=nn.ReLU(True), dilated_blocks_n=0, dilated_blocks_n_start=0, dilated_blocks_n_middle=0, add_out_act=True, max_features=1024, is_resblock_depthwise=False, ffc_positions=None, ffc_kwargs={}, dilation=1, second_dilation=None, dilation_block_kind='simple', multidilation_kwargs={}):
        assert n_blocks >= 0
        super().__init__()
        conv_layer = get_conv_block_ctor(conv_kind)
        norm_layer = get_norm_layer(norm_layer)
        if affine is not None:
            norm_layer = partial(norm_layer, affine=affine)
        up_norm_layer = get_norm_layer(up_norm_layer)
        if affine is not None:
            up_norm_layer = partial(up_norm_layer, affine=affine)
        if ffc_positions is not None:
            ffc_positions = collections.Counter(ffc_positions)
        model = [nn.ReflectionPad2d(3), conv_layer(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        identity = Identity()
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [conv_layer(min(max_features, ngf * mult), min(max_features, ngf * mult * 2), kernel_size=3, stride=2, padding=1), norm_layer(min(max_features, ngf * mult * 2)), activation]
        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)
        dilated_block_kwargs = dict(dim=feats_num_bottleneck, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        if dilation_block_kind == 'simple':
            dilated_block_kwargs['conv_kind'] = conv_kind
        elif dilation_block_kind == 'multi':
            dilated_block_kwargs['conv_layer'] = functools.partial(get_conv_block_ctor('multidilated'), **multidilation_kwargs)
        if dilated_blocks_n_start is not None and dilated_blocks_n_start > 0:
            model += make_dil_blocks(dilated_blocks_n_start, dilation_block_kind, dilated_block_kwargs)
        for i in range(n_blocks):
            if i == n_blocks // 2 and dilated_blocks_n_middle is not None and dilated_blocks_n_middle > 0:
                model += make_dil_blocks(dilated_blocks_n_middle, dilation_block_kind, dilated_block_kwargs)
            if ffc_positions is not None and i in ffc_positions:
                for _ in range(ffc_positions[i]):
                    model += [FFCResnetBlock(feats_num_bottleneck, padding_type, norm_layer, activation_layer=nn.ReLU, inline=True, **ffc_kwargs)]
            if is_resblock_depthwise:
                resblock_groups = feats_num_bottleneck
            else:
                resblock_groups = 1
            model += [ResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation=activation, norm_layer=norm_layer, conv_kind=conv_kind, groups=resblock_groups, dilation=dilation, second_dilation=second_dilation)]
        if dilated_blocks_n is not None and dilated_blocks_n > 0:
            model += make_dil_blocks(dilated_blocks_n, dilation_block_kind, dilated_block_kwargs)
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult), min(max_features, int(ngf * mult / 2)), kernel_size=3, stride=2, padding=1, output_padding=1), up_norm_layer(min(max_features, int(ngf * mult / 2))), up_activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class GlobalGeneratorGated(GlobalGenerator):

    def __init__(self, *args, **kwargs):
        real_kwargs = dict(conv_kind='gated_bn_relu', activation=nn.Identity(), norm_layer=nn.Identity)
        real_kwargs.update(kwargs)
        super().__init__(*args, **real_kwargs)


class GlobalGeneratorFromSuperChannels(nn.Module):

    def __init__(self, input_nc, output_nc, n_downsampling, n_blocks, super_channels, norm_layer='bn', padding_type='reflect', add_out_act=True):
        super().__init__()
        self.n_downsampling = n_downsampling
        norm_layer = get_norm_layer(norm_layer)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        channels = self.convert_super_channels(super_channels)
        self.channels = channels
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, channels[0], kernel_size=7, padding=0, bias=use_bias), norm_layer(channels[0]), nn.ReLU(True)]
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(channels[0 + i], channels[1 + i], kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(channels[1 + i]), nn.ReLU(True)]
        mult = 2 ** n_downsampling
        n_blocks1 = n_blocks // 3
        n_blocks2 = n_blocks1
        n_blocks3 = n_blocks - n_blocks1 - n_blocks2
        for i in range(n_blocks1):
            c = n_downsampling
            dim = channels[c]
            model += [ResnetBlock(dim, padding_type=padding_type, norm_layer=norm_layer)]
        for i in range(n_blocks2):
            c = n_downsampling + 1
            dim = channels[c]
            kwargs = {}
            if i == 0:
                kwargs = {'in_dim': channels[c - 1]}
            model += [ResnetBlock(dim, padding_type=padding_type, norm_layer=norm_layer, **kwargs)]
        for i in range(n_blocks3):
            c = n_downsampling + 2
            dim = channels[c]
            kwargs = {}
            if i == 0:
                kwargs = {'in_dim': channels[c - 1]}
            model += [ResnetBlock(dim, padding_type=padding_type, norm_layer=norm_layer, **kwargs)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(channels[n_downsampling + 3 + i], channels[n_downsampling + 3 + i + 1], kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias), norm_layer(channels[n_downsampling + 3 + i + 1]), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(channels[2 * n_downsampling + 3], output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

    def convert_super_channels(self, super_channels):
        n_downsampling = self.n_downsampling
        result = []
        cnt = 0
        if n_downsampling == 2:
            N1 = 10
        elif n_downsampling == 3:
            N1 = 13
        else:
            raise NotImplementedError
        for i in range(0, N1):
            if i in [1, 4, 7, 10]:
                channel = super_channels[cnt] * 2 ** cnt
                config = {'channel': channel}
                result.append(channel)
                logging.info(f'Downsample channels {result[-1]}')
                cnt += 1
        for i in range(3):
            for counter, j in enumerate(range(N1 + i * 3, N1 + 3 + i * 3)):
                if len(super_channels) == 6:
                    channel = super_channels[3] * 4
                else:
                    channel = super_channels[i + 3] * 4
                config = {'channel': channel}
                if counter == 0:
                    result.append(channel)
                    logging.info(f'Bottleneck channels {result[-1]}')
        cnt = 2
        for i in range(N1 + 9, N1 + 21):
            if i in [22, 25, 28]:
                cnt -= 1
                if len(super_channels) == 6:
                    channel = super_channels[5 - cnt] * 2 ** cnt
                else:
                    channel = super_channels[7 - cnt] * 2 ** cnt
                result.append(int(channel))
                logging.info(f'Upsample channels {result[-1]}')
        return result

    def forward(self, input):
        return self.model(input)


class NLayerDiscriminator(BaseDiscriminator):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.n_layers = n_layers
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            cur_model = []
            cur_model += [nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw), norm_layer(nf), nn.LeakyReLU(0.2, True)]
            sequence.append(cur_model)
        nf_prev = nf
        nf = min(nf * 2, 512)
        cur_model = []
        cur_model += [nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw), norm_layer(nf), nn.LeakyReLU(0.2, True)]
        sequence.append(cur_model)
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        for n in range(len(sequence)):
            setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, x):
        act = self.get_all_activations(x)
        return act[-1], act[:-1]


class MultidilatedNLayerDiscriminator(BaseDiscriminator):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, multidilation_kwargs={}):
        super().__init__()
        self.n_layers = n_layers
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            cur_model = []
            cur_model += [MultidilatedConv(nf_prev, nf, kernel_size=kw, stride=2, padding=[2, 3], **multidilation_kwargs), norm_layer(nf), nn.LeakyReLU(0.2, True)]
            sequence.append(cur_model)
        nf_prev = nf
        nf = min(nf * 2, 512)
        cur_model = []
        cur_model += [nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw), norm_layer(nf), nn.LeakyReLU(0.2, True)]
        sequence.append(cur_model)
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        for n in range(len(sequence)):
            setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, x):
        act = self.get_all_activations(x)
        return act[-1], act[:-1]


class NLayerDiscriminatorAsGen(NLayerDiscriminator):

    def forward(self, x):
        return super().forward(x)[0]


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
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


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
            w_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}'

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    """ Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        assert H % 2 == 0 and W % 2 == 0, f'x size ({H}*{W}) are not even.'
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) ->str:
        return f'input_resolution={self.input_resolution}, dim={self.dim}'

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += H // 2 * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer(nn.Module):
    """ Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), input_resolution=(patches_resolution[0] // 2 ** i_layer, patches_resolution[1] // 2 ** i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=PatchMerging if i_layer < self.num_layers - 1 else None, use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // 2 ** self.num_layers
        flops += self.num_features * self.num_classes
        return flops


class ConvBNActivation(nn.Sequential):

    def __init__(self, in_planes: 'int', out_planes: 'int', kernel_size: 'int'=3, stride: 'int'=1, groups: 'int'=1, norm_layer: 'Optional[Callable[..., nn.Module]]'=None, activation_layer: 'Optional[Callable[..., nn.Module]]'=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False), norm_layer(out_planes), activation_layer(inplace=True))


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class SqueezeExcitation(nn.Module):

    def __init__(self, input_c: 'int', squeeze_factor: 'int'=4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: 'Tensor') ->Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x


class InvertedResidual(nn.Module):

    def __init__(self, cnf: 'InvertedResidualConfig', norm_layer: 'Callable[..., nn.Module]'):
        super(InvertedResidual, self).__init__()
        if cnf.stride not in [1, 2]:
            raise ValueError('illegal stride value.')
        self.use_res_connect = cnf.stride == 1 and cnf.input_c == cnf.out_c
        layers: 'List[nn.Module]' = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU
        if cnf.expanded_c != cnf.input_c:
            layers.append(ConvBNActivation(cnf.input_c, cnf.expanded_c, kernel_size=1, norm_layer=norm_layer, activation_layer=activation_layer))
        layers.append(ConvBNActivation(cnf.expanded_c, cnf.expanded_c, kernel_size=cnf.kernel, stride=cnf.stride, groups=cnf.expanded_c, norm_layer=norm_layer, activation_layer=activation_layer))
        if cnf.use_se:
            layers.append(SqueezeExcitation(cnf.expanded_c))
        layers.append(ConvBNActivation(cnf.expanded_c, cnf.out_c, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Identity))
        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

    def forward(self, x: 'Tensor') ->Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result


class InvertedResidualConfig:

    def __init__(self, input_c: 'int', kernel: 'int', expanded_c: 'int', out_c: 'int', use_se: 'bool', activation: 'str', stride: 'int', width_multi: 'float'):
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        self.use_hs = activation == 'HS'
        self.stride = stride

    @staticmethod
    def adjust_channels(channels: 'int', width_multi: 'float'):
        return _make_divisible(channels * width_multi, 8)


class MobileNetV3(nn.Module):

    def __init__(self, inverted_residual_setting: 'List[InvertedResidualConfig]', last_channel: 'int', num_classes: 'int'=1000, block: 'Optional[Callable[..., nn.Module]]'=None, norm_layer: 'Optional[Callable[..., nn.Module]]'=None):
        super(MobileNetV3, self).__init__()
        if not inverted_residual_setting:
            raise ValueError('The inverted_residual_setting should not be empty.')
        elif not (isinstance(inverted_residual_setting, List) and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError('The inverted_residual_setting should be List[InvertedResidualConfig]')
        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        layers: 'List[nn.Module]' = []
        firstconv_output_c = inverted_residual_setting[0].input_c
        layers.append(ConvBNActivation(3, firstconv_output_c, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.Hardswish))
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))
        lastconv_input_c = inverted_residual_setting[-1].out_c
        lastconv_output_c = 6 * lastconv_input_c
        layers.append(ConvBNActivation(lastconv_input_c, lastconv_output_c, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Hardswish))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(lastconv_output_c, last_channel), nn.Hardswish(inplace=True), nn.Dropout(p=0.2, inplace=True), nn.Linear(last_channel, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: 'Tensor') ->Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: 'Tensor') ->Tensor:
        return self._forward_impl(x)


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class ScriptDecoder(Decoder):
    """ 当script_path非None，直接load ScriptModule;
        当model_path非None，load PyTorchModule后使用script方式转换为ScriptModule。

        Args:
            script_path (str): ScriptModule保存路径。
            model_path (str): PyTorchModule保存路径。
    """

    def __init__(self, module, input_examples=None):
        self.script = torch.jit.script(module)
        self.graph = self._optimize_graph(self.script.inlined_graph)
        self.input_examples = input_examples


class TraceDecoder(Decoder):
    """ PyTorchModule后使用trace方式转换为ScriptModule。

        Args:
            model_path (str): PyTorchModule保存路径。
            input_files (list): 输入网络的numpy，每个numpy保存成.npy文件,
                                文件路径存储在input_files中。
    """

    def __init__(self, module, input_examples):
        try:
            self.script = torch.jit.trace(module, input_examples)
        except RuntimeError as e:
            if 'strict' in str(e):
                self.script = torch.jit.trace(module, input_examples, strict=False)
            else:
                None
                exit(0)
        self.graph = self._optimize_graph(self.script.inlined_graph)
        self.input_examples = input_examples


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Attention,
     lambda: ([], {'attn_dims': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (BaseModel,
     lambda: ([], {}),
     lambda: ([], {})),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicBlockDe,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BatchNormConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (BlurMask,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 1, 4, 4])], {})),
    (CBAModule,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CBNModule,
     lambda: ([], {'inchannel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ContextModule,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvBNActivation,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv_block,
     lambda: ([], {'in_c': 4, 'out_c': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DBFace,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (DepthWiseSeperableConv,
     lambda: ([], {'in_dim': 4, 'out_dim': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Depth_Wise,
     lambda: ([], {'c1': [4, 4], 'c2': [4, 4], 'c3': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Depth_Wise_SE,
     lambda: ([], {'c1': [4, 4], 'c2': [4, 4], 'c3': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DetectModule,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DiscriminatorP,
     lambda: ([], {'period': 4}),
     lambda: ([torch.rand([4, 1, 4])], {})),
    (DiscriminatorS,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64])], {})),
    (Dist2LogitLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 1, 4, 4])], {})),
    (EmulatedEDTMask,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 1, 1]), torch.rand([4, 1, 1, 1]), torch.rand([4, 1, 4, 4])], {})),
    (EvaluatorScore,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (FIDInceptionA,
     lambda: ([], {'in_channels': 4, 'pool_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FIDInceptionC,
     lambda: ([], {'in_channels': 4, 'channels_7x7': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FIDInceptionE_1,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FIDInceptionE_2,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FourierUnit,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GruModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 768, 768])], {})),
    (HSigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HeadModule,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HighwayNetwork,
     lambda: ([], {'size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (IOU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (L2Norm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LOGSSIM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (LSA,
     lambda: ([], {'attn_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4]), 0, torch.rand([4, 4, 4, 4])], {})),
    (Linear_block,
     lambda: ([], {'in_c': 4, 'out_c': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MSNet3D,
     lambda: ([], {'maxdisp': 4}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {})),
    (Mbv3SmallFast,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (MeanShift,
     lambda: ([], {'rgb_range': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MobileV2_Residual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expanse_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MobileV2_Residual_3D,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expanse_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (MultiScaleDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64]), torch.rand([4, 1, 64])], {})),
    (MultiscaleResNetSingle,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {})),
    (Net,
     lambda: ([], {'base_channels': 4, 'depth_num': 1}),
     lambda: ([torch.rand([4, 3, 4, 4])], {})),
    (PNetLin,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 512, 512]), torch.rand([4, 3, 512, 512])], {})),
    (PairwiseScore,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (PreNet,
     lambda: ([], {'in_dims': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResBlock,
     lambda: ([], {'base_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResBlock1,
     lambda: ([], {'h': 4, 'channels': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (ResBlock2,
     lambda: ([], {'h': 4, 'channels': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (ResNetHead,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {})),
    (ResNetTail,
     lambda: ([], {'output_nc': 4}),
     lambda: ([torch.rand([4, 512, 4, 4])], {})),
    (SELayer,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SEModule,
     lambda: ([], {'channels': 4, 'reduction': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SSIM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SSIMScore,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ScalingLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {})),
    (SeModule,
     lambda: ([], {'in_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SeparableConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SimpleMultiStepGenerator,
     lambda: ([], {'steps': [torch.nn.ReLU()]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SpeakerEncoder,
     lambda: ([], {'device': 0, 'loss_device': 0}),
     lambda: ([torch.rand([4, 40, 40])], {})),
    (UpModule,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (VGG19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (alexnet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (feature_extraction,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (hourglass2D,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (hourglass3D,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (resnet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (squeezenet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (vgg16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

