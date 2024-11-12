
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


from collections import namedtuple


import torch


from torchvision import models


from torch.utils.data import DataLoader


import time


from torch.optim import Adam


from torch.utils.tensorboard import SummaryWriter


import numpy as np


from torch.hub import download_url_to_file


import re


import matplotlib.pyplot as plt


from torchvision import transforms


from torchvision import datasets


from torch.utils.data import Dataset


from torch.utils.data import Sampler


class Vgg16(torch.nn.Module):
    """Only those layers are exposed which have already proven to work nicely."""

    def __init__(self, requires_grad=False, show_progress=False):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True, progress=show_progress).eval()
        vgg_pretrained_features = vgg16.features
        self.layer_names = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.slice1(x)
        relu1_2 = x
        x = self.slice2(x)
        relu2_2 = x
        x = self.slice3(x)
        relu3_3 = x
        x = self.slice4(x)
        relu4_3 = x
        vgg_outputs = namedtuple('VggOutputs', self.layer_names)
        out = vgg_outputs(relu1_2, relu2_2, relu3_3, relu4_3)
        return out


class ConvLayer(torch.nn.Module):
    """
        A small wrapper around nn.Conv2d, so as to make the code cleaner and allow for experimentation with padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, padding_mode='reflect')

    def forward(self, x):
        return self.conv2d(x)


class ResidualBlock(torch.nn.Module):
    """
        Originally introduced in (Microsoft Research Asia, He et al.): https://arxiv.org/abs/1512.03385
        Modified architecture according to suggestions in this blog: http://torch.ch/blog/2016/02/04/resnets.html

        The only difference from the original is: There is no ReLU layer after the addition of identity and residual
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        kernel_size = 3
        stride_size = 1
        self.conv1 = ConvLayer(channels, channels, kernel_size=kernel_size, stride=stride_size)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=kernel_size, stride=stride_size)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out + residual


class UpsampleConvLayer(torch.nn.Module):
    """
        Nearest-neighbor up-sampling followed by a convolution
        Appears to give better results than learned up-sampling aka transposed conv (avoids the checkerboard artifact)

        Initially proposed on distill pub: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.upsampling_factor = stride
        self.conv2d = ConvLayer(in_channels, out_channels, kernel_size, stride=1)

    def forward(self, x):
        if self.upsampling_factor > 1:
            x = torch.nn.functional.interpolate(x, scale_factor=self.upsampling_factor, mode='nearest')
        return self.conv2d(x)


class TransformerNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        num_of_channels = [3, 32, 64, 128]
        kernel_sizes = [9, 3, 3]
        stride_sizes = [1, 2, 2]
        self.conv1 = ConvLayer(num_of_channels[0], num_of_channels[1], kernel_size=kernel_sizes[0], stride=stride_sizes[0])
        self.in1 = torch.nn.InstanceNorm2d(num_of_channels[1], affine=True)
        self.conv2 = ConvLayer(num_of_channels[1], num_of_channels[2], kernel_size=kernel_sizes[1], stride=stride_sizes[1])
        self.in2 = torch.nn.InstanceNorm2d(num_of_channels[2], affine=True)
        self.conv3 = ConvLayer(num_of_channels[2], num_of_channels[3], kernel_size=kernel_sizes[2], stride=stride_sizes[2])
        self.in3 = torch.nn.InstanceNorm2d(num_of_channels[3], affine=True)
        res_block_num_of_filters = 128
        self.res1 = ResidualBlock(res_block_num_of_filters)
        self.res2 = ResidualBlock(res_block_num_of_filters)
        self.res3 = ResidualBlock(res_block_num_of_filters)
        self.res4 = ResidualBlock(res_block_num_of_filters)
        self.res5 = ResidualBlock(res_block_num_of_filters)
        num_of_channels.reverse()
        kernel_sizes.reverse()
        stride_sizes.reverse()
        self.up1 = UpsampleConvLayer(num_of_channels[0], num_of_channels[1], kernel_size=kernel_sizes[0], stride=stride_sizes[0])
        self.in4 = torch.nn.InstanceNorm2d(num_of_channels[1], affine=True)
        self.up2 = UpsampleConvLayer(num_of_channels[1], num_of_channels[2], kernel_size=kernel_sizes[1], stride=stride_sizes[1])
        self.in5 = torch.nn.InstanceNorm2d(num_of_channels[2], affine=True)
        self.up3 = ConvLayer(num_of_channels[2], num_of_channels[3], kernel_size=kernel_sizes[2], stride=stride_sizes[2])

    def forward(self, x):
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.up1(y)))
        y = self.relu(self.in5(self.up2(y)))
        return self.up3(y)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ConvLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualBlock,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TransformerNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (UpsampleConvLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Vgg16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

