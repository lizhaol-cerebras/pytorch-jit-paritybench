
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


import random


import uuid


import numpy as np


import pandas as pd


import torch


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torchvision import transforms


import time


import torch.nn as nn


from torch.utils.data import random_split


import torch.optim as optim


import math


from collections import OrderedDict


from torch.nn.modules.utils import _triple


import itertools


from matplotlib import pyplot as plt


from sklearn.neighbors import KNeighborsClassifier


import sklearn.metrics as metrics


from sklearn.metrics.pairwise import cosine_distances


from sklearn.metrics.pairwise import euclidean_distances


class AlexNet(nn.Module):
    """AlexNet with BN and pool5 to be AdaptiveAvgPool2d(1)"""

    def __init__(self, with_classifier=False, return_conv=False, num_classes=1000):
        super(AlexNet, self).__init__()
        self.with_classifier = with_classifier
        self.return_conv = return_conv
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(192)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.AdaptiveAvgPool2d(1)
        if self.return_conv:
            self.feature_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        if self.with_classifier:
            self.linear = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        if self.return_conv:
            x = self.feature_pool(x)
            return x.view(x.shape[0], -1)
        x = self.pool5(x)
        x = x.view(-1, 256)
        if self.with_classifier:
            x = self.linear(x)
        return x


class C3D(nn.Module):
    """C3D with BN and pool5 to be AdaptiveAvgPool3d(1)."""

    def __init__(self, with_classifier=False, return_conv=False, num_classes=101):
        super(C3D, self).__init__()
        self.with_classifier = with_classifier
        self.num_classes = num_classes
        self.return_conv = return_conv
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3a = nn.BatchNorm3d(256)
        self.relu3a = nn.ReLU()
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3b = nn.BatchNorm3d(256)
        self.relu3b = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4a = nn.BatchNorm3d(512)
        self.relu4a = nn.ReLU()
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4b = nn.BatchNorm3d(512)
        self.relu4b = nn.ReLU()
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5a = nn.BatchNorm3d(512)
        self.relu5a = nn.ReLU()
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5b = nn.BatchNorm3d(512)
        self.relu5b = nn.ReLU()
        if self.return_conv:
            self.feature_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.pool5 = nn.AdaptiveAvgPool3d(1)
        if self.with_classifier:
            self.linear = nn.Linear(512, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3a(x)
        x = self.bn3a(x)
        x = self.relu3a(x)
        x = self.conv3b(x)
        x = self.bn3b(x)
        x = self.relu3b(x)
        x = self.pool3(x)
        x = self.conv4a(x)
        x = self.bn4a(x)
        x = self.relu4a(x)
        x = self.conv4b(x)
        x = self.bn4b(x)
        x = self.relu4b(x)
        x = self.pool4(x)
        x = self.conv5a(x)
        x = self.bn5a(x)
        x = self.relu5a(x)
        x = self.conv5b(x)
        x = self.bn5b(x)
        x = self.relu5b(x)
        if self.return_conv:
            x = self.feature_pool(x)
            return x.view(x.shape[0], -1)
        x = self.pool5(x)
        x = x.view(-1, 512)
        if self.with_classifier:
            x = self.linear(x)
        return x


class OPN(nn.Module):
    """Frame Order Prediction Network"""

    def __init__(self, base_network, feature_size, tuple_len):
        """
        Args:
            feature_size (int): 256
        """
        super(OPN, self).__init__()
        self.base_network = base_network
        self.tuple_len = tuple_len
        self.feature_size = feature_size
        self.class_num = math.factorial(tuple_len)
        self.fc7 = nn.Linear(self.feature_size * 2, 512)
        pair_num = int(tuple_len * (tuple_len - 1) / 2)
        self.fc8 = nn.Linear(512 * pair_num, self.class_num)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, tuple):
        f = []
        for i in range(self.tuple_len):
            frame = tuple[:, i, :, :, :]
            f.append(self.base_network(frame))
        pf = []
        for i in range(self.tuple_len):
            for j in range(i + 1, self.tuple_len):
                pf.append(torch.cat([f[i], f[j]], dim=1))
        pf = [self.fc7(i) for i in pf]
        pf = [self.relu(i) for i in pf]
        h = torch.cat(pf, dim=1)
        h = self.dropout(h)
        h = self.fc8(h)
        return h


class OPN_RNN(nn.Module):
    """Frame Order Prediction Network with RNN"""

    def __init__(self, base_network, feature_size, tuple_len, hidden_size, rnn_type='LSTM'):
        """
        Args:
            feature_size (int): 256
        """
        super(OPN_RNN, self).__init__()
        self.base_network = base_network
        self.tuple_len = tuple_len
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.class_num = math.factorial(tuple_len)
        if self.rnn_type == 'LSTM':
            self.lstm = nn.LSTM(self.feature_size, self.hidden_size, num_layers=2, bidirectional=True)
        elif self.rnn_type == 'GRU':
            self.gru = nn.GRU(self.feature_size, self.hidden_size, num_layers=2, bidirectional=True)
        elif self.rnn_type == 'RNN':
            self.rnn = nn.RNN(self.feature_size, self.hidden_size, num_layers=2, bidirectional=True, nonlinearity='relu')
        self.fc = nn.Linear(self.hidden_size * 2, self.class_num)

    def forward(self, tuple):
        f = []
        for i in range(self.tuple_len):
            frame = tuple[:, i, :, :, :]
            f.append(self.base_network(frame))
        inputs = torch.stack(f)
        if self.rnn_type == 'LSTM':
            outputs, (hn, cn) = self.lstm(inputs)
        elif self.rnn_type == 'GRU':
            outputs, hn = self.gru(inputs)
        elif self.rnn_type == 'RNN':
            outputs, hn = self.rnn(inputs)
        h = self.fc(outputs[-1])
        return h


class SpatioTemporalConv(nn.Module):
    """Applies a factored 3D convolution over an input signal composed of several input
    planes with distinct spatial and time axes, by performing a 2D convolution over the
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(SpatioTemporalConv, self).__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        self.temporal_spatial_conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        x = self.temporal_spatial_conv(x)
        return x


class SpatioTemporalResBlock(nn.Module):
    """Single block for the ResNet network. Uses SpatioTemporalConv in
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)
        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the block.
            kernel_size (int or tuple): Size of the convolving kernels.
            downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()
        self.downsample = downsample
        padding = kernel_size // 2
        if self.downsample:
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))
        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))
        return self.outrelu(x + res)


class SpatioTemporalResLayer(nn.Module):
    """Forms a single layer of the ResNet network, with a number of repeating
    blocks of same output size stacked on top of each other
        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the layer.
            kernel_size (int or tuple): Size of the convolving kernels.
            layer_size (int): Number of blocks to be stacked to form the layer
            block_type (Module, optional): Type of block that is to be used to form the layer. Default: SpatioTemporalResBlock.
            downsample (bool, optional): If ``True``, the first block in layer will implement downsampling. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=SpatioTemporalResBlock, downsample=False):
        super(SpatioTemporalResLayer, self).__init__()
        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)
        return x


class R2Plus1DNet(nn.Module):
    """Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in
    each layer set by layer_sizes, and by performing a global average pool at the end producing a
    512-dimensional vector for each element in the batch.
        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
    """

    def __init__(self, layer_sizes, block_type=SpatioTemporalResBlock, with_classifier=False, return_conv=False, num_classes=101):
        super(R2Plus1DNet, self).__init__()
        self.with_classifier = with_classifier
        self.return_conv = return_conv
        self.num_classes = num_classes
        self.conv1 = SpatioTemporalConv(3, 64, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU()
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[0], block_type=block_type)
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True)
        self.conv4 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True)
        self.conv5 = SpatioTemporalResLayer(256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True)
        if self.return_conv:
            self.feature_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.pool = nn.AdaptiveAvgPool3d(1)
        if self.with_classifier:
            self.linear = nn.Linear(512, self.num_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        if self.return_conv:
            x = self.feature_pool(x)
            return x.view(x.shape[0], -1)
        x = self.pool(x)
        x = x.view(-1, 512)
        if self.with_classifier:
            x = self.linear(x)
        return x


class R3DNet(nn.Module):
    """Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in
    each layer set by layer_sizes, and by performing a global average pool at the end producing a
    512-dimensional vector for each element in the batch.
        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
    """

    def __init__(self, layer_sizes, block_type=SpatioTemporalResBlock, with_classifier=False, return_conv=False, num_classes=101):
        super(R3DNet, self).__init__()
        self.with_classifier = with_classifier
        self.return_conv = return_conv
        self.num_classes = num_classes
        self.conv1 = SpatioTemporalConv(3, 64, [3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3])
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU()
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[0], block_type=block_type)
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True)
        self.conv4 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True)
        self.conv5 = SpatioTemporalResLayer(256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True)
        if self.return_conv:
            self.feature_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.pool = nn.AdaptiveAvgPool3d(1)
        if self.with_classifier:
            self.linear = nn.Linear(512, self.num_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        if self.return_conv:
            x = self.feature_pool(x)
            return x.view(x.shape[0], -1)
        x = self.pool(x)
        x = x.view(-1, 512)
        if self.with_classifier:
            x = self.linear(x)
        return x


class VCOPN(nn.Module):
    """Video clip order prediction with PFE (Pairwire Feature Extraction), the same as OPN."""

    def __init__(self, base_network, feature_size, tuple_len):
        """
        Args:
            feature_size (int): 512
        """
        super(VCOPN, self).__init__()
        self.base_network = base_network
        self.feature_size = feature_size
        self.tuple_len = tuple_len
        self.class_num = math.factorial(tuple_len)
        self.fc7 = nn.Linear(self.feature_size * 2, 512)
        pair_num = int(tuple_len * (tuple_len - 1) / 2)
        self.fc8 = nn.Linear(512 * pair_num, self.class_num)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, tuple):
        f = []
        for i in range(self.tuple_len):
            clip = tuple[:, i, :, :, :, :]
            f.append(self.base_network(clip))
        pf = []
        for i in range(self.tuple_len):
            for j in range(i + 1, self.tuple_len):
                pf.append(torch.cat([f[i], f[j]], dim=1))
        pf = [self.fc7(i) for i in pf]
        pf = [self.relu(i) for i in pf]
        h = torch.cat(pf, dim=1)
        h = self.dropout(h)
        h = self.fc8(h)
        return h


class VCOPN_RNN(nn.Module):
    """Video clip order prediction with RNN."""

    def __init__(self, base_network, feature_size, tuple_len, hidden_size, rnn_type='LSTM'):
        """
        Args:
            feature_size (int): 1024
        """
        super(VCOPN_RNN, self).__init__()
        self.base_network = base_network
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.tuple_len = tuple_len
        self.class_num = math.factorial(tuple_len)
        self.rnn_type = rnn_type
        if self.rnn_type == 'LSTM':
            self.lstm = nn.LSTM(self.feature_size, self.hidden_size)
        elif self.rnn_type == 'GRU':
            self.gru = nn.GRU(self.feature_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.class_num)

    def forward(self, tuple):
        f = []
        for i in range(self.tuple_len):
            clip = tuple[:, i, :, :, :, :]
            f.append(self.base_network(clip))
        inputs = torch.stack(f)
        if self.rnn_type == 'LSTM':
            outputs, (hn, cn) = self.lstm(inputs)
        elif self.rnn_type == 'GRU':
            outputs, hn = self.gru(inputs)
        h = self.fc(hn.squeeze(dim=0))
        return h


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AlexNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (C3D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64, 64])], {})),
    (R2Plus1DNet,
     lambda: ([], {'layer_sizes': [4, 4, 4, 4]}),
     lambda: ([torch.rand([4, 3, 64, 64, 64])], {})),
    (R3DNet,
     lambda: ([], {'layer_sizes': [4, 4, 4, 4]}),
     lambda: ([torch.rand([4, 3, 64, 64, 64])], {})),
    (SpatioTemporalConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

