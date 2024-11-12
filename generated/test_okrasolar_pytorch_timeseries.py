
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


from torch import nn


from typing import cast


from typing import Union


from typing import List


import torch.nn.functional as F


import numpy as np


from sklearn.metrics import roc_auc_score


from sklearn.metrics import accuracy_score


from torch.utils.data import DataLoader


from typing import Any


from typing import Dict


from typing import Tuple


from typing import Optional


from sklearn.preprocessing import OneHotEncoder


from sklearn.model_selection import train_test_split


from torch.utils.data import TensorDataset


from torch.nn import functional as F


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (l_out - 1) * stride - l_in + dilation * (kernel - 1) + 1
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])
    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride, padding=padding // 2, dilation=dilation, groups=groups)


class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """

    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride, self.dilation, self.groups)


class ConvBlock(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'int', stride: 'int') ->None:
        super().__init__()
        self.layers = nn.Sequential(Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride), nn.BatchNorm1d(num_features=out_channels), nn.ReLU())

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self.layers(x)


class FCNBaseline(nn.Module):
    """A PyTorch implementation of the FCN Baseline
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, in_channels: 'int', num_pred_classes: 'int'=1) ->None:
        super().__init__()
        self.input_args = {'in_channels': in_channels, 'num_pred_classes': num_pred_classes}
        self.layers = nn.Sequential(*[ConvBlock(in_channels, 128, 8, 1), ConvBlock(128, 256, 5, 1), ConvBlock(256, 128, 3, 1)])
        self.final = nn.Linear(128, num_pred_classes)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.layers(x)
        return self.final(x.mean(dim=-1))


class InceptionBlock(nn.Module):
    """An inception block consists of an (optional) bottleneck, followed
    by 3 conv1d layers. Optionally residual
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', residual: 'bool', stride: 'int'=1, bottleneck_channels: 'int'=32, kernel_size: 'int'=41) ->None:
        assert kernel_size > 3, 'Kernel size must be strictly greater than 3'
        super().__init__()
        self.use_bottleneck = bottleneck_channels > 0
        if self.use_bottleneck:
            self.bottleneck = Conv1dSamePadding(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        kernel_size_s = [(kernel_size // 2 ** i) for i in range(3)]
        start_channels = bottleneck_channels if self.use_bottleneck else in_channels
        channels = [start_channels] + [out_channels] * 3
        self.conv_layers = nn.Sequential(*[Conv1dSamePadding(in_channels=channels[i], out_channels=channels[i + 1], kernel_size=kernel_size_s[i], stride=stride, bias=False) for i in range(len(kernel_size_s))])
        self.batchnorm = nn.BatchNorm1d(num_features=channels[-1])
        self.relu = nn.ReLU()
        self.use_residual = residual
        if residual:
            self.residual = nn.Sequential(*[Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm1d(out_channels), nn.ReLU()])

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        org_x = x
        if self.use_bottleneck:
            x = self.bottleneck(x)
        x = self.conv_layers(x)
        if self.use_residual:
            x = x + self.residual(org_x)
        return x


class InceptionModel(nn.Module):
    """A PyTorch implementation of the InceptionTime model.
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    num_blocks:
        The number of inception blocks to use. One inception block consists
        of 3 convolutional layers, (optionally) a bottleneck and (optionally) a residual
        connector
    in_channels:
        The number of input channels (i.e. input.shape[-1])
    out_channels:
        The number of "hidden channels" to use. Can be a list (for each block) or an
        int, in which case the same value will be applied to each block
    bottleneck_channels:
        The number of channels to use for the bottleneck. Can be list or int. If 0, no
        bottleneck is applied
    kernel_sizes:
        The size of the kernels to use for each inception block. Within each block, each
        of the 3 convolutional layers will have kernel size
        `[kernel_size // (2 ** i) for i in range(3)]`
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, num_blocks: 'int', in_channels: 'int', out_channels: 'Union[List[int], int]', bottleneck_channels: 'Union[List[int], int]', kernel_sizes: 'Union[List[int], int]', use_residuals: 'Union[List[bool], bool, str]'='default', num_pred_classes: 'int'=1) ->None:
        super().__init__()
        self.input_args = {'num_blocks': num_blocks, 'in_channels': in_channels, 'out_channels': out_channels, 'bottleneck_channels': bottleneck_channels, 'kernel_sizes': kernel_sizes, 'use_residuals': use_residuals, 'num_pred_classes': num_pred_classes}
        channels = [in_channels] + cast(List[int], self._expand_to_blocks(out_channels, num_blocks))
        bottleneck_channels = cast(List[int], self._expand_to_blocks(bottleneck_channels, num_blocks))
        kernel_sizes = cast(List[int], self._expand_to_blocks(kernel_sizes, num_blocks))
        if use_residuals == 'default':
            use_residuals = [(True if i % 3 == 2 else False) for i in range(num_blocks)]
        use_residuals = cast(List[bool], self._expand_to_blocks(cast(Union[bool, List[bool]], use_residuals), num_blocks))
        self.blocks = nn.Sequential(*[InceptionBlock(in_channels=channels[i], out_channels=channels[i + 1], residual=use_residuals[i], bottleneck_channels=bottleneck_channels[i], kernel_size=kernel_sizes[i]) for i in range(num_blocks)])
        self.linear = nn.Linear(in_features=channels[-1], out_features=num_pred_classes)

    @staticmethod
    def _expand_to_blocks(value: 'Union[int, bool, List[int], List[bool]]', num_blocks: 'int') ->Union[List[int], List[bool]]:
        if isinstance(value, list):
            assert len(value) == num_blocks, f'Length of inputs lists must be the same as num blocks, expected length {num_blocks}, got {len(value)}'
        else:
            value = [value] * num_blocks
        return value

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.blocks(x).mean(dim=-1)
        return self.linear(x)


class LinearBlock(nn.Module):

    def __init__(self, input_size: 'int', output_size: 'int', dropout: 'float') ->None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU(), nn.Dropout(p=dropout))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self.layers(x)


class LinearBaseline(nn.Module):
    """A PyTorch implementation of the Linear Baseline
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, num_inputs: 'int', num_pred_classes: 'int'=1) ->None:
        super().__init__()
        self.input_args = {'num_inputs': num_inputs, 'num_pred_classes': num_pred_classes}
        self.layers = nn.Sequential(nn.Dropout(0.1), LinearBlock(num_inputs, 500, 0.2), LinearBlock(500, 500, 0.2), LinearBlock(500, num_pred_classes, 0.3))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self.layers(x.view(x.shape[0], -1))


class ResNetBlock(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int') ->None:
        super().__init__()
        channels = [in_channels, out_channels, out_channels, out_channels]
        kernel_sizes = [8, 5, 3]
        self.layers = nn.Sequential(*[ConvBlock(in_channels=channels[i], out_channels=channels[i + 1], kernel_size=kernel_sizes[i], stride=1) for i in range(len(kernel_sizes))])
        self.match_channels = False
        if in_channels != out_channels:
            self.match_channels = True
            self.residual = nn.Sequential(*[Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1), nn.BatchNorm1d(num_features=out_channels)])

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        if self.match_channels:
            return self.layers(x) + self.residual(x)
        return self.layers(x)


class ResNetBaseline(nn.Module):
    """A PyTorch implementation of the ResNet Baseline
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    mid_channels:
        The 3 residual blocks will have as output channels:
        [mid_channels, mid_channels * 2, mid_channels * 2]
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, in_channels: 'int', mid_channels: 'int'=64, num_pred_classes: 'int'=1) ->None:
        super().__init__()
        self.input_args = {'in_channels': in_channels, 'num_pred_classes': num_pred_classes}
        self.layers = nn.Sequential(*[ResNetBlock(in_channels=in_channels, out_channels=mid_channels), ResNetBlock(in_channels=mid_channels, out_channels=mid_channels * 2), ResNetBlock(in_channels=mid_channels * 2, out_channels=mid_channels * 2)])
        self.final = nn.Linear(mid_channels * 2, num_pred_classes)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.layers(x)
        return self.final(x.mean(dim=-1))


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Conv1dSamePadding,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (ConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (FCNBaseline,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (InceptionBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'residual': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (InceptionModel,
     lambda: ([], {'num_blocks': 4, 'in_channels': 4, 'out_channels': 4, 'bottleneck_channels': 4, 'kernel_sizes': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (LinearBaseline,
     lambda: ([], {'num_inputs': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (LinearBlock,
     lambda: ([], {'input_size': 4, 'output_size': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResNetBaseline,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (ResNetBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
]

