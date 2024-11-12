
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


import copy as cp


import numpy as np


import warnings


import random


from collections import deque


import time


import copy


import logging


import queue


from abc import ABCMeta


from abc import abstractmethod


from typing import List


from typing import Optional


from typing import Tuple


from typing import Union


import torch.nn as nn


from torch.optim import SGD


from torch.optim.sgd import SGD


from torch.optim import AdamW


from torch.utils.data import ConcatDataset


from typing import Callable


from typing import Dict


from typing import Sequence


import scipy


from scipy.stats import mode


from torch.nn.modules.utils import _pair


from numbers import Number


import functools


import math


from functools import reduce


from torch.utils.data import DataLoader


from collections import OrderedDict


from itertools import product


from typing import Any


import torch.nn.functional as F


import torch.utils.checkpoint as cp


from torch.utils import checkpoint as cp


from torch.nn.modules.utils import _ntuple


from torch.nn.modules.utils import _triple


from functools import lru_cache


import torch.utils.checkpoint as checkpoint


from copy import deepcopy


from torch import nn


from torch import Tensor


import itertools


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import torchvision


import torch.utils.checkpoint


from torch import device


from torch import dtype


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


from torch.nn import LayerNorm


from torch.nn import Linear


from torch.nn import MultiheadAttention


from scipy import interpolate


from torch.distributed.nn import all_gather as all_gather_with_grad


import inspect


import torch.distributed as dist


from functools import partial


from torch.distributions.beta import Beta


from torchvision.transforms import Normalize


from numpy.testing import assert_array_equal


from numpy.testing import assert_array_almost_equal


from numpy.testing import assert_almost_equal


from torch.autograd import Variable


from abc import abstractproperty


import torch.multiprocessing as mp


import re


import matplotlib.pyplot as plt


class C3D(nn.Module):
    """C3D backbone.

    Args:
        pretrained (str | None): Name of pretrained model.
        style (str): ``pytorch`` or ``caffe``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: 'pytorch'.
        conv_cfg (dict | None): Config dict for convolution layer.
            If set to None, it uses ``dict(type='Conv3d')`` to construct
            layers. Default: None.
        norm_cfg (dict | None): Config for norm layers. required keys are
            ``type``, Default: None.
        act_cfg (dict | None): Config dict for activation layer. If set to
            None, it uses ``dict(type='ReLU')`` to construct layers.
            Default: None.
        out_dim (int): The dimension of last layer feature (after flatten).
            Depends on the input shape. Default: 8192.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation of fc layers. Default: 0.01.
    """

    def __init__(self, pretrained=None, style='pytorch', conv_cfg=None, norm_cfg=None, act_cfg=None, out_dim=8192, dropout_ratio=0.5, init_std=0.005):
        super().__init__()
        if conv_cfg is None:
            conv_cfg = dict(type='Conv3d')
        if act_cfg is None:
            act_cfg = dict(type='ReLU')
        self.pretrained = pretrained
        self.style = style
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        c3d_conv_param = dict(kernel_size=(3, 3, 3), padding=(1, 1, 1), conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv1a = ConvModule(3, 64, **c3d_conv_param)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2a = ConvModule(64, 128, **c3d_conv_param)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv3a = ConvModule(128, 256, **c3d_conv_param)
        self.conv3b = ConvModule(256, 256, **c3d_conv_param)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv4a = ConvModule(256, 512, **c3d_conv_param)
        self.conv4b = ConvModule(512, 512, **c3d_conv_param)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv5a = ConvModule(512, 512, **c3d_conv_param)
        self.conv5b = ConvModule(512, 512, **c3d_conv_param)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        self.fc6 = nn.Linear(out_dim, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_ratio)

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=self.init_std)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
                the size of x is (num_batches, 3, 16, 112, 112).

        Returns:
            torch.Tensor: The feature of the input
            samples extracted by the backbone.
        """
        x = self.conv1a(x)
        x = self.pool1(x)
        x = self.conv2a(x)
        x = self.pool2(x)
        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pool3(x)
        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.pool4(x)
        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.pool5(x)
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        return x


class InvertedResidual(nn.Module):
    """InvertedResidual block for MobileNetV2.

    Args:
        in_channels (int): The input channels of the InvertedResidual block.
        out_channels (int): The output channels of the InvertedResidual block.
        stride (int): Stride of the middle (first) 3x3 convolution.
        expand_ratio (int): adjusts number of channels of the hidden layer
            in InvertedResidual by this amount.
        conv_cfg (dict): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='ReLU6').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
    Returns:
        Tensor: The output tensor
    """

    def __init__(self, in_channels, out_channels, stride, expand_ratio, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU6'), with_cp=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2], f'stride must in [1, 2]. But received {stride}.'
        self.with_cp = with_cp
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))
        layers = []
        if expand_ratio != 1:
            layers.append(ConvModule(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        layers.extend([ConvModule(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg), ConvModule(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The output of the module.
        """

        def _inner_forward(x):
            if self.use_res_connect:
                return x + self.conv(x)
            return self.conv(x)
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        return out


class BasicBlock(nn.Module):
    """Basic block for ResNet.

    Args:
        inplanes (int): Number of channels for the input in first conv2d layer.
        planes (int): Number of channels produced by some norm/conv2d layers.
        stride (int): Stride in the conv layer. Defaults to 1.
        dilation (int): Spacing between kernel elements. Defaults to 1.
        downsample (nn.Module, optional): Downsample layer. Defaults to None.
        style (str): ``pytorch`` or ``caffe``. If set to ``pytorch``, the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Defaults to ``pytorch``.
        conv_cfg (Union[dict, ConfigDict]): Config for norm layers.
            Defaults to ``dict(type='Conv')``.
        norm_cfg (Union[dict, ConfigDict]): Config for norm layers. required
            keys are ``type`` and ``requires_grad``.
            Defaults to ``dict(type='BN2d', requires_grad=True)``.
        act_cfg (Union[dict, ConfigDict]): Config for activate layers.
            Defaults to ``dict(type='ReLU', inplace=True)``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
    """
    expansion = 1

    def __init__(self, inplanes: 'int', planes: 'int', stride: 'int'=1, dilation: 'int'=1, downsample: 'Optional[nn.Module]'=None, style: 'str'='pytorch', conv_cfg: 'ConfigType'=dict(type='Conv'), norm_cfg: 'ConfigType'=dict(type='BN', requires_grad=True), act_cfg: 'ConfigType'=dict(type='ReLU', inplace=True), with_cp: 'bool'=False) ->None:
        super().__init__()
        assert style in ['pytorch', 'caffe']
        self.conv1 = ConvModule(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(planes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.style = style
        self.stride = stride
        self.dilation = dilation
        self.norm_cfg = norm_cfg
        assert not with_cp

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet.

    Args:
        inplanes (int):
            Number of channels for the input feature in first conv layer.
        planes (int):
            Number of channels produced by some norm layes and conv layers.
        stride (int): Spatial stride in the conv layer. Defaults to 1.
        dilation (int): Spacing between kernel elements. Defaults to 1.
        downsample (nn.Module, optional): Downsample layer. Defaults to None.
        style (str): ``pytorch`` or ``caffe``. If set to ``pytorch``, the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Defaults to ``pytorch``.
        conv_cfg (Union[dict, ConfigDict]): Config for norm layers.
            Defaults to ``dict(type='Conv')``.
        norm_cfg (Union[dict, ConfigDict]): Config for norm layers. required
            keys are ``type`` and ``requires_grad``.
            Defaults to ``dict(type='BN2d', requires_grad=True)``.
        act_cfg (Union[dict, ConfigDict]): Config for activate layers.
            Defaults to ``dict(type='ReLU', inplace=True)``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
    """
    expansion = 4

    def __init__(self, inplanes: 'int', planes: 'int', stride: 'int'=1, dilation: 'int'=1, downsample: 'Optional[nn.Module]'=None, style: 'str'='pytorch', conv_cfg: 'ConfigType'=dict(type='Conv'), norm_cfg: 'ConfigType'=dict(type='BN', requires_grad=True), act_cfg: 'ConfigType'=dict(type='ReLU', inplace=True), with_cp: 'bool'=False) ->None:
        super().__init__()
        assert style in ['pytorch', 'caffe']
        self.inplanes = inplanes
        self.planes = planes
        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1
        self.conv1 = ConvModule(inplanes, planes, kernel_size=1, stride=self.conv1_stride, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(planes, planes, kernel_size=3, stride=self.conv2_stride, padding=dilation, dilation=dilation, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv3 = ConvModule(planes, planes * self.expansion, kernel_size=1, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            identity = x
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out = out + identity
            return out
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.relu(out)
        return out


class Bottleneck2dAudio(nn.Module):
    """Bottleneck2D block for ResNet2D.

    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        stride (int): Stride in the conv layer. Defaults to 2.
        dilation (int): Spacing between kernel elements. Defaults to 1.
        downsample (nn.Module, optional): Downsample layer. Defaults to None.
        factorize (bool): Whether to factorize kernel. Defaults to True.
        norm_cfg (dict): Config for norm layers. required keys are ``type`` and
            ``requires_grad``. Defaults to None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the trgaining speed. Defaults to False.
    """
    expansion = 4

    def __init__(self, inplanes: 'int', planes: 'int', stride: 'int'=2, dilation: 'int'=1, downsample: 'Optional[nn.Module]'=None, factorize: 'bool'=True, norm_cfg: 'ConfigType'=None, with_cp: 'bool'=False) ->None:
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.factorize = factorize
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.conv1_stride = 1
        self.conv2_stride = stride
        conv1_kernel_size = 1, 1
        conv1_padding = 0
        conv2_kernel_size = 3, 3
        conv2_padding = dilation, dilation
        self.conv1 = ConvModule(inplanes, planes, kernel_size=conv1_kernel_size, padding=conv1_padding, dilation=dilation, norm_cfg=self.norm_cfg, bias=False)
        self.conv2 = ConvModule(planes, planes, kernel_size=conv2_kernel_size, stride=stride, padding=conv2_padding, dilation=dilation, bias=False, conv_cfg=dict(type='ConvAudio') if factorize else dict(type='Conv'), norm_cfg=None, act_cfg=None)
        self.conv3 = ConvModule(2 * planes if factorize else planes, planes * self.expansion, kernel_size=1, bias=False, norm_cfg=self.norm_cfg, act_cfg=None)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return out
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.relu(out)
        return out


class ResNetAudio(nn.Module):
    """ResNet 2d audio backbone. Reference:

        <https://arxiv.org/abs/2001.08740>`_.

    Args:
        depth (int): Depth of resnet, from ``{50, 101, 152}``.
        pretrained (str, optional): Name of pretrained model. Defaults to None.
        in_channels (int): Channel num of input features. Defaults to 1.
        base_channels (int): Channel num of stem output features.
            Defaults to 32.
        num_stages (int): Resnet stages. Defaults to 4.
        strides (Sequence[int]): Strides of residual blocks of each stage.
            Defaults to ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Defaults to ``(1, 1, 1, 1)``.
        conv1_kernel (int): Kernel size of the first conv layer. Defaults to 9.
        conv1_stride (Union[int, Tuple[int]]): Stride of the first conv layer.
            Defaults to 1.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters. Defaults to -1.
        factorize (Sequence[int]): factorize Dims of each block for audio.
            Defaults to ``(1, 1, 0, 0)``.
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var). Defaults to False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        conv_cfg (Union[dict, ConfigDict]): Config for norm layers.
            Defaults to ``dict(type='Conv')``.
        norm_cfg (Union[dict, ConfigDict]): Config for norm layers. required
            keys are ``type`` and ``requires_grad``.
            Defaults to ``dict(type='BN2d', requires_grad=True)``.
        act_cfg (Union[dict, ConfigDict]): Config for activate layers.
            Defaults to ``dict(type='ReLU', inplace=True)``.
        zero_init_residual (bool): Whether to use zero initialization
            for residual block. Defaults to True.
    """
    arch_settings = {(50): (Bottleneck2dAudio, (3, 4, 6, 3)), (101): (Bottleneck2dAudio, (3, 4, 23, 3)), (152): (Bottleneck2dAudio, (3, 8, 36, 3))}

    def __init__(self, depth: 'int', pretrained: 'str'=None, in_channels: 'int'=1, num_stages: 'int'=4, base_channels: 'int'=32, strides: 'Sequence[int]'=(1, 2, 2, 2), dilations: 'Sequence[int]'=(1, 1, 1, 1), conv1_kernel: 'int'=9, conv1_stride: 'int'=1, frozen_stages: 'int'=-1, factorize: 'Sequence[int]'=(1, 1, 0, 0), norm_eval: 'bool'=False, with_cp: 'bool'=False, conv_cfg: 'ConfigType'=dict(type='Conv'), norm_cfg: 'ConfigType'=dict(type='BN2d', requires_grad=True), act_cfg: 'ConfigType'=dict(type='ReLU', inplace=True), zero_init_residual: 'bool'=True) ->None:
        super().__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.dilations = dilations
        self.conv1_kernel = conv1_kernel
        self.conv1_stride = conv1_stride
        self.frozen_stages = frozen_stages
        self.stage_factorization = _ntuple(num_stages)(factorize)
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = self.base_channels
        self._make_stem_layer()
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = self.base_channels * 2 ** i
            res_layer = self.make_res_layer(self.block, self.inplanes, planes, num_blocks, stride=stride, dilation=dilation, factorize=self.stage_factorization[i], norm_cfg=self.norm_cfg, with_cp=with_cp)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
        self.feat_dim = self.block.expansion * self.base_channels * 2 ** (len(self.stage_blocks) - 1)

    @staticmethod
    def make_res_layer(block: 'nn.Module', inplanes: 'int', planes: 'int', blocks: 'int', stride: 'int'=1, dilation: 'int'=1, factorize: 'int'=1, norm_cfg: 'Optional[ConfigType]'=None, with_cp: 'bool'=False) ->nn.Module:
        """Build residual layer for ResNetAudio.

        Args:
            block (nn.Module): Residual module to be built.
            inplanes (int): Number of channels for the input feature
                in each block.
            planes (int): Number of channels for the output feature
                in each block.
            blocks (int): Number of residual blocks.
            stride (int): Strides of residual blocks of each stage.
                Defaults to  1.
            dilation (int): Spacing between kernel elements. Defaults to 1.
            factorize (Uninon[int, Sequence[int]]): Determine whether to
                factorize for each block. Defaults to 1.
            norm_cfg (Union[dict, ConfigDict], optional): Config for norm
                layers. Defaults to None.
            with_cp (bool): Use checkpoint or not. Using checkpoint will save
                some memory while slowing down the training speed.
                Defaults to False.

        Returns:
            nn.Module: A residual layer for the given config.
        """
        factorize = factorize if not isinstance(factorize, int) else (factorize,) * blocks
        assert len(factorize) == blocks
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = ConvModule(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, norm_cfg=norm_cfg, act_cfg=None)
        layers = []
        layers.append(block(inplanes, planes, stride, dilation, downsample, factorize=factorize[0] == 1, norm_cfg=norm_cfg, with_cp=with_cp))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, 1, dilation, factorize=factorize[i] == 1, norm_cfg=norm_cfg, with_cp=with_cp))
        return nn.Sequential(*layers)

    def _make_stem_layer(self) ->None:
        """Construct the stem layers consists of a ``conv+norm+act`` module and
        a pooling layer."""
        self.conv1 = ConvModule(self.in_channels, self.base_channels, kernel_size=self.conv1_kernel, stride=self.conv1_stride, bias=False, conv_cfg=dict(type='ConvAudio', op='sum'), norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def _freeze_stages(self) ->None:
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.conv1.bn.eval()
            for m in [self.conv1.conv, self.conv1.bn]:
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self) ->None:
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck2dAudio):
                        constant_init(m.conv3.bn, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The feature of the input samples extracted
                by the backbone.
        """
        x = self.conv1(x)
        for layer_name in self.res_layers:
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
        return x

    def train(self, mode: 'bool'=True) ->None:
        """Set the optimization status when training."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


class CombineNet(nn.Module):
    """Combine Net.

    It combines Temporal interlace module with some part of ResNet layer.

    Args:
        net1 (nn.module): Temporal interlace module.
        net2 (nn.module): Some part of ResNet layer.
    """

    def __init__(self, net1, net2):
        super().__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        x = self.net1(x)
        x = self.net2(x)
        return x


class WeightNet(nn.Module):
    """WeightNet in Temporal interlace module.

    The WeightNet consists of two parts: one convolution layer
    and a sigmoid function. Following the convolution layer, the sigmoid
    function and rescale module can scale our output to the range (0, 2).
    Here we set the initial bias of the convolution layer to 0, and the
    final initial output will be 1.0.

    Args:
        in_channels (int): Channel num of input features.
        groups (int): Number of groups for fc layer outputs.
    """

    def __init__(self, in_channels, groups):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.groups = groups
        self.conv = nn.Conv1d(in_channels, groups, 3, padding=1)
        self.init_weights()

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        self.conv.bias.data[...] = 0

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        n, _, t = x.shape
        x = self.conv(x)
        x = x.view(n, self.groups, t)
        x = x.permute(0, 2, 1)
        x = 2 * self.sigmoid(x)
        return x


class OffsetNet(nn.Module):
    """OffsetNet in Temporal interlace module.

    The OffsetNet consists of one convolution layer and two fc layers
    with a relu activation following with a sigmoid function. Following
    the convolution layer, two fc layers and relu are applied to the output.
    Then, apply the sigmoid function with a multiply factor and a minus 0.5
    to transform the output to (-4, 4).

    Args:
        in_channels (int): Channel num of input features.
        groups (int): Number of groups for fc layer outputs.
        num_segments (int): Number of frame segments.
    """

    def __init__(self, in_channels, groups, num_segments):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        kernel_size = 3
        padding = 1
        self.conv = nn.Conv1d(in_channels, 1, kernel_size, padding=padding)
        self.fc1 = nn.Linear(num_segments, num_segments)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_segments, groups)
        self.init_weights()

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        self.fc2.bias.data[...] = 0.5108

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        n, _, t = x.shape
        x = self.conv(x)
        x = x.view(n, t)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(n, 1, -1)
        x = 4 * (self.sigmoid(x) - 0.5)
        return x


class TemporalInterlace(nn.Module):
    """Temporal interlace module.

    This module is proposed in `Temporal Interlacing Network
    <https://arxiv.org/abs/2001.06499>`_

    Args:
        in_channels (int): Channel num of input features.
        num_segments (int): Number of frame segments. Default: 3.
        shift_div (int): Number of division parts for shift. Default: 1.
    """

    def __init__(self, in_channels, num_segments=3, shift_div=1):
        super().__init__()
        self.num_segments = num_segments
        self.shift_div = shift_div
        self.in_channels = in_channels
        self.deform_groups = 2
        self.offset_net = OffsetNet(in_channels // shift_div, self.deform_groups, num_segments)
        self.weight_net = WeightNet(in_channels // shift_div, self.deform_groups)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        n, c, h, w = x.size()
        num_batches = n // self.num_segments
        num_folds = c // self.shift_div
        x_out = torch.zeros((n, c, h, w), device=x.device)
        x_descriptor = x[:, :num_folds, :, :].view(num_batches, self.num_segments, num_folds, h, w)
        x_pooled = torch.mean(x_descriptor, 3)
        x_pooled = torch.mean(x_pooled, 3)
        x_pooled = x_pooled.permute(0, 2, 1).contiguous()
        x_offset = self.offset_net(x_pooled).view(num_batches, -1)
        x_weight = self.weight_net(x_pooled)
        x_offset = torch.cat([x_offset, -x_offset], 1)
        x_shift = linear_sampler(x_descriptor, x_offset)
        x_weight = x_weight[:, :, :, None]
        x_weight = x_weight.repeat(1, 1, 2, num_folds // 2 // 2)
        x_weight = x_weight.view(x_weight.size(0), x_weight.size(1), -1)
        x_weight = x_weight[:, :, :, None, None]
        x_shift = x_shift * x_weight
        x_shift = x_shift.contiguous().view(n, num_folds, h, w)
        x_out[:, :num_folds, :] = x_shift
        x_out[:, num_folds:, :] = x[:, num_folds:, :]
        return x_out


class NL3DWrapper(nn.Module):
    """3D Non-local wrapper for ResNet50.

    Wrap ResNet layers with 3D NonLocal modules.

    Args:
        block (nn.Module): Residual blocks to be built.
        num_segments (int): Number of frame segments.
        non_local_cfg (dict): Config for non-local layers. Default: ``dict()``.
    """

    def __init__(self, block, num_segments, non_local_cfg=dict()):
        super(NL3DWrapper, self).__init__()
        self.block = block
        self.non_local_cfg = non_local_cfg
        self.non_local_block = NonLocal3d(self.block.conv3.norm.num_features, **self.non_local_cfg)
        self.num_segments = num_segments

    def forward(self, x):
        """Defines the computation performed at every call."""
        x = self.block(x)
        n, c, h, w = x.size()
        x = x.view(n // self.num_segments, self.num_segments, c, h, w).transpose(1, 2).contiguous()
        x = self.non_local_block(x)
        x = x.transpose(1, 2).contiguous().view(n, c, h, w)
        return x


class TemporalShift(nn.Module):
    """Temporal shift module.

    This module is proposed in
    `TSM: Temporal Shift Module for Efficient Video Understanding
    <https://arxiv.org/abs/1811.08383>`_

    Args:
        net (nn.module): Module to make temporal shift.
        num_segments (int): Number of frame segments. Default: 3.
        shift_div (int): Number of divisions for shift. Default: 8.
    """

    def __init__(self, net, num_segments=3, shift_div=8):
        super().__init__()
        self.net = net
        self.num_segments = num_segments
        self.shift_div = shift_div

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        x = self.shift(x, self.num_segments, shift_div=self.shift_div)
        return self.net(x)

    @staticmethod
    def shift(x, num_segments, shift_div=3):
        """Perform temporal shift operation on the feature.

        Args:
            x (torch.Tensor): The input feature to be shifted.
            num_segments (int): Number of frame segments.
            shift_div (int): Number of divisions for shift. Default: 3.

        Returns:
            torch.Tensor: The shifted feature.
        """
        n, c, h, w = x.size()
        x = x.view(-1, num_segments, c, h * w)
        fold = c // shift_div
        left_split = x[:, :, :fold, :]
        mid_split = x[:, :, fold:2 * fold, :]
        right_split = x[:, :, 2 * fold:, :]
        zeros = left_split - left_split
        blank = zeros[:, :1, :, :]
        left_split = left_split[:, 1:, :, :]
        left_split = torch.cat((left_split, blank), 1)
        zeros = mid_split - mid_split
        blank = zeros[:, :1, :, :]
        mid_split = mid_split[:, :-1, :, :]
        mid_split = torch.cat((blank, mid_split), 1)
        out = torch.cat((left_split, mid_split, right_split), 2)
        return out.view(n, c, h, w)


class TAM(nn.Module):
    """Temporal Adaptive Module(TAM) for TANet.

    This module is proposed in `TAM: TEMPORAL ADAPTIVE MODULE FOR VIDEO
    RECOGNITION <https://arxiv.org/pdf/2005.06803>`_

    Args:
        in_channels (int): Channel num of input features.
        num_segments (int): Number of frame segments.
        alpha (int): ``alpha`` in the paper and is the ratio of the
            intermediate channel number to the initial channel number in the
            global branch. Defaults to 2.
        adaptive_kernel_size (int): ``K`` in the paper and is the size of the
            adaptive kernel size in the global branch. Defaults to 3.
        beta (int): ``beta`` in the paper and is set to control the model
            complexity in the local branch. Defaults to 4.
        conv1d_kernel_size (int): Size of the convolution kernel of Conv1d in
            the local branch. Defaults to 3.
        adaptive_convolution_stride (int): The first dimension of strides in
            the adaptive convolution of ``Temporal Adaptive Aggregation``.
            Defaults to 1.
        adaptive_convolution_padding (int): The first dimension of paddings in
            the adaptive convolution of ``Temporal Adaptive Aggregation``.
            Defaults to 1.
        init_std (float): Std value for initiation of `nn.Linear`. Defaults to
            0.001.
    """

    def __init__(self, in_channels: 'int', num_segments: 'int', alpha: 'int'=2, adaptive_kernel_size: 'int'=3, beta: 'int'=4, conv1d_kernel_size: 'int'=3, adaptive_convolution_stride: 'int'=1, adaptive_convolution_padding: 'int'=1, init_std: 'float'=0.001) ->None:
        super().__init__()
        assert beta > 0 and alpha > 0
        self.in_channels = in_channels
        self.num_segments = num_segments
        self.alpha = alpha
        self.adaptive_kernel_size = adaptive_kernel_size
        self.beta = beta
        self.conv1d_kernel_size = conv1d_kernel_size
        self.adaptive_convolution_stride = adaptive_convolution_stride
        self.adaptive_convolution_padding = adaptive_convolution_padding
        self.init_std = init_std
        self.G = nn.Sequential(nn.Linear(num_segments, num_segments * alpha, bias=False), nn.BatchNorm1d(num_segments * alpha), nn.ReLU(inplace=True), nn.Linear(num_segments * alpha, adaptive_kernel_size, bias=False), nn.Softmax(-1))
        self.L = nn.Sequential(nn.Conv1d(in_channels, in_channels // beta, conv1d_kernel_size, stride=1, padding=conv1d_kernel_size // 2, bias=False), nn.BatchNorm1d(in_channels // beta), nn.ReLU(inplace=True), nn.Conv1d(in_channels // beta, in_channels, 1, bias=False), nn.Sigmoid())

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        n, c, h, w = x.size()
        num_segments = self.num_segments
        num_batches = n // num_segments
        assert c == self.in_channels
        x = x.view(num_batches, num_segments, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        theta_out = F.adaptive_avg_pool2d(x.view(-1, num_segments, h, w), (1, 1))
        conv_kernel = self.G(theta_out.view(-1, num_segments)).view(num_batches * c, 1, -1, 1)
        local_activation = self.L(theta_out.view(-1, c, num_segments)).view(num_batches, c, num_segments, 1, 1)
        new_x = x * local_activation
        y = F.conv2d(new_x.view(1, num_batches * c, num_segments, h * w), conv_kernel, bias=None, stride=(self.adaptive_convolution_stride, 1), padding=(self.adaptive_convolution_padding, 0), groups=num_batches * c)
        y = y.view(num_batches, c, num_segments, h, w)
        y = y.permute(0, 2, 1, 3, 4).contiguous().view(n, c, h, w)
        return y


class TABlock(nn.Module):
    """Temporal Adaptive Block (TA-Block) for TANet.

    This block is proposed in `TAM: TEMPORAL ADAPTIVE MODULE FOR VIDEO
    RECOGNITION <https://arxiv.org/pdf/2005.06803>`_

    The temporal adaptive module (TAM) is embedded into ResNet-Block
    after the first Conv2D, which turns the vanilla ResNet-Block
    into TA-Block.

    Args:
        block (nn.Module): Residual blocks to be substituted.
        num_segments (int): Number of frame segments.
        tam_cfg (dict): Config for temporal adaptive module (TAM).
    """

    def __init__(self, block: 'nn.Module', num_segments: 'int', tam_cfg: 'dict') ->None:
        super().__init__()
        self.tam_cfg = deepcopy(tam_cfg)
        self.block = block
        self.num_segments = num_segments
        self.tam = TAM(in_channels=block.conv1.out_channels, num_segments=num_segments, **self.tam_cfg)
        if not isinstance(self.block, Bottleneck):
            raise NotImplementedError('TA-Blocks have not been fully implemented except the pattern based on Bottleneck block.')

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Defines the computation performed at every call."""
        assert isinstance(self.block, Bottleneck)

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            identity = x
            out = self.block.conv1(x)
            out = self.tam(out)
            out = self.block.conv2(out)
            out = self.block.conv3(out)
            if self.block.downsample is not None:
                identity = self.block.downsample(x)
            out = out + identity
            return out
        if self.block.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.block.relu(out)
        return out


class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = img_size[1] // patch_size[1] * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]), stride=(self.tubelet_size, patch_size[0], patch_size[1]))

    def forward(self, x):
        B, C, T, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class TimeSformer(nn.Module):
    """TimeSformer. A PyTorch impl of `Is Space-Time Attention All You Need for
    Video Understanding? <https://arxiv.org/abs/2102.05095>`_

    Args:
        num_frames (int): Number of frames in the video.
        img_size (int | tuple): Size of input image.
        patch_size (int): Size of one patch.
        pretrained (str | None): Name of pretrained model. Default: None.
        embed_dims (int): Dimensions of embedding. Defaults to 768.
        num_heads (int): Number of parallel attention heads in
            TransformerCoder. Defaults to 12.
        num_transformer_layers (int): Number of transformer layers. Defaults to
            12.
        in_channels (int): Channel num of input features. Defaults to 3.
        dropout_ratio (float): Probability of dropout layer. Defaults to 0..
        transformer_layers (list[obj:`mmcv.ConfigDict`] |
            obj:`mmcv.ConfigDict` | None): Config of transformerlayer in
            TransformerCoder. If it is obj:`mmcv.ConfigDict`, it would be
            repeated `num_transformer_layers` times to a
            list[obj:`mmcv.ConfigDict`]. Defaults to None.
        attention_type (str): Type of attentions in TransformerCoder. Choices
            are 'divided_space_time', 'space_only' and 'joint_space_time'.
            Defaults to 'divided_space_time'.
        norm_cfg (dict): Config for norm layers. Defaults to
            `dict(type='LN', eps=1e-6)`.
    """
    supported_attention_types = ['divided_space_time', 'space_only', 'joint_space_time']

    def __init__(self, num_frames, img_size, patch_size, pretrained=None, embed_dims=768, num_heads=12, num_transformer_layers=12, in_channels=3, dropout_ratio=0.0, transformer_layers=None, attention_type='divided_space_time', norm_cfg=dict(type='LN', eps=1e-06), **kwargs):
        super().__init__(**kwargs)
        assert attention_type in self.supported_attention_types, f'Unsupported Attention Type {attention_type}!'
        assert transformer_layers is None or isinstance(transformer_layers, (dict, list))
        self.num_frames = num_frames
        self.pretrained = pretrained
        self.embed_dims = embed_dims
        self.num_transformer_layers = num_transformer_layers
        self.attention_type = attention_type
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dims=embed_dims)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dims))
        self.drop_after_pos = nn.Dropout(p=dropout_ratio)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dims))
            self.drop_after_time = nn.Dropout(p=dropout_ratio)
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        if transformer_layers is None:
            dpr = np.linspace(0, 0.1, num_transformer_layers)
            if self.attention_type == 'divided_space_time':
                _transformerlayers_cfg = [dict(type='BaseTransformerLayer', attn_cfgs=[dict(type='DividedTemporalAttentionWithNorm', embed_dims=embed_dims, num_heads=num_heads, num_frames=num_frames, dropout_layer=dict(type='DropPath', drop_prob=dpr[i]), norm_cfg=dict(type='LN', eps=1e-06)), dict(type='DividedSpatialAttentionWithNorm', embed_dims=embed_dims, num_heads=num_heads, num_frames=num_frames, dropout_layer=dict(type='DropPath', drop_prob=dpr[i]), norm_cfg=dict(type='LN', eps=1e-06))], ffn_cfgs=dict(type='FFNWithNorm', embed_dims=embed_dims, feedforward_channels=embed_dims * 4, num_fcs=2, act_cfg=dict(type='GELU'), dropout_layer=dict(type='DropPath', drop_prob=dpr[i]), norm_cfg=dict(type='LN', eps=1e-06)), operation_order=('self_attn', 'self_attn', 'ffn')) for i in range(num_transformer_layers)]
            else:
                _transformerlayers_cfg = [dict(type='BaseTransformerLayer', attn_cfgs=[dict(type='MultiheadAttention', embed_dims=embed_dims, num_heads=num_heads, batch_first=True, dropout_layer=dict(type='DropPath', drop_prob=dpr[i]))], ffn_cfgs=dict(type='FFN', embed_dims=embed_dims, feedforward_channels=embed_dims * 4, num_fcs=2, act_cfg=dict(type='GELU'), dropout_layer=dict(type='DropPath', drop_prob=dpr[i])), operation_order=('norm', 'self_attn', 'norm', 'ffn'), norm_cfg=dict(type='LN', eps=1e-06), batch_first=True) for i in range(num_transformer_layers)]
            transformer_layers = ConfigDict(dict(type='TransformerLayerSequence', transformerlayers=_transformerlayers_cfg, num_layers=num_transformer_layers))
        self.transformer_layers = build_transformer_layer_sequence(transformer_layers)

    def init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')
            state_dict = _load_checkpoint(self.pretrained, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            if self.attention_type == 'divided_space_time':
                old_state_dict_keys = list(state_dict.keys())
                for old_key in old_state_dict_keys:
                    if 'norms' in old_key:
                        new_key = old_key.replace('norms.0', 'attentions.0.norm')
                        new_key = new_key.replace('norms.1', 'ffns.0.norm')
                        state_dict[new_key] = state_dict.pop(old_key)
                old_state_dict_keys = list(state_dict.keys())
                for old_key in old_state_dict_keys:
                    if 'attentions.0' in old_key:
                        new_key = old_key.replace('attentions.0', 'attentions.1')
                        state_dict[new_key] = state_dict[old_key].clone()
            load_state_dict(self, state_dict, strict=False, logger=logger)

    def forward(self, x):
        """Defines the computation performed at every call."""
        batches = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.drop_after_pos(x)
        if self.attention_type != 'space_only':
            cls_tokens = x[:batches, 0, :].unsqueeze(1)
            x = rearrange(x[:, 1:, :], '(b t) p m -> (b p) t m', b=batches)
            x = x + self.time_embed
            x = self.drop_after_time(x)
            x = rearrange(x, '(b p) t m -> b (p t) m', b=batches)
            x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer_layers(x, None, None)
        if self.attention_type == 'space_only':
            x = x.view(-1, self.num_frames, *x.size()[-2:])
            x = torch.mean(x, 1)
        x = self.norm(x)
        return x[:, 0]


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.bottleneck = self._round_width(channels, reduction)
        self.fc1 = nn.Conv3d(channels, self.bottleneck, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv3d(self.bottleneck, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _round_width(width, multiplier, min_width=8, divisor=8):
        """Round width of filters based on width multiplier."""
        width *= multiplier
        min_width = min_width or divisor
        width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
        if width_out < 0.9 * width:
            width_out += divisor
        return int(width_out)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The output of the module.
        """
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class BlockX3D(nn.Module):
    """BlockX3D 3d building block for X3D.

    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        outplanes (int): Number of channels produced by final the conv3d layer.
        spatial_stride (int): Spatial stride in the conv3d layer. Default: 1.
        downsample (nn.Module | None): Downsample layer. Default: None.
        se_ratio (float | None): The reduction ratio of squeeze and excitation
            unit. If set as None, it means not using SE unit. Default: None.
        use_swish (bool): Whether to use swish as the activation function
            before and after the 3x3x3 conv. Default: True.
        conv_cfg (dict): Config dict for convolution layer.
            Default: ``dict(type='Conv3d')``.
        norm_cfg (dict): Config for norm layers. required keys are ``type``,
            Default: ``dict(type='BN3d')``.
        act_cfg (dict): Config dict for activation layer.
            Default: ``dict(type='ReLU')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self, inplanes, planes, outplanes, spatial_stride=1, downsample=None, se_ratio=None, use_swish=True, conv_cfg=dict(type='Conv3d'), norm_cfg=dict(type='BN3d'), act_cfg=dict(type='ReLU'), with_cp=False):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.outplanes = outplanes
        self.spatial_stride = spatial_stride
        self.downsample = downsample
        self.se_ratio = se_ratio
        self.use_swish = use_swish
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.act_cfg_swish = dict(type='Swish')
        self.with_cp = with_cp
        self.conv1 = ConvModule(in_channels=inplanes, out_channels=planes, kernel_size=1, stride=1, padding=0, bias=False, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv2 = ConvModule(in_channels=planes, out_channels=planes, kernel_size=3, stride=(1, self.spatial_stride, self.spatial_stride), padding=1, groups=planes, bias=False, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=None)
        self.swish = Swish()
        self.conv3 = ConvModule(in_channels=planes, out_channels=outplanes, kernel_size=1, stride=1, padding=0, bias=False, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=None)
        if self.se_ratio is not None:
            self.se_module = SEModule(planes, self.se_ratio)
        self.relu = build_activation_layer(self.act_cfg)

    def forward(self, x):
        """Defines the computation performed at every call."""

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            identity = x
            out = self.conv1(x)
            out = self.conv2(out)
            if self.se_ratio is not None:
                out = self.se_module(out)
            out = self.swish(out)
            out = self.conv3(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out = out + identity
            return out
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.relu(out)
        return out


class X3D(nn.Module):
    """X3D backbone. https://arxiv.org/pdf/2004.04730.pdf.

    Args:
        gamma_w (float): Global channel width expansion factor. Default: 1.
        gamma_b (float): Bottleneck channel width expansion factor. Default: 1.
        gamma_d (float): Network depth expansion factor. Default: 1.
        pretrained (str | None): Name of pretrained model. Default: None.
        in_channels (int): Channel num of input features. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        spatial_strides (Sequence[int]):
            Spatial strides of residual blocks of each stage.
            Default: ``(1, 2, 2, 2)``.
        frozen_stages (int): Stages to be frozen (all param fixed). If set to
            -1, it means not freezing any parameters. Default: -1.
        se_style (str): The style of inserting SE modules into BlockX3D, 'half'
            denotes insert into half of the blocks, while 'all' denotes insert
            into all blocks. Default: 'half'.
        se_ratio (float | None): The reduction ratio of squeeze and excitation
            unit. If set as None, it means not using SE unit. Default: 1 / 16.
        use_swish (bool): Whether to use swish as the activation function
            before and after the 3x3x3 conv. Default: True.
        conv_cfg (dict): Config for conv layers. required keys are ``type``
            Default: ``dict(type='Conv3d')``.
        norm_cfg (dict): Config for norm layers. required keys are ``type`` and
            ``requires_grad``.
            Default: ``dict(type='BN3d', requires_grad=True)``.
        act_cfg (dict): Config dict for activation layer.
            Default: ``dict(type='ReLU', inplace=True)``.
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var). Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool):
            Whether to use zero initialization for residual block,
            Default: True.
        kwargs (dict, optional): Key arguments for "make_res_layer".
    """

    def __init__(self, gamma_w=1.0, gamma_b=1.0, gamma_d=1.0, pretrained=None, in_channels=3, num_stages=4, spatial_strides=(2, 2, 2, 2), frozen_stages=-1, se_style='half', se_ratio=1 / 16, use_swish=True, conv_cfg=dict(type='Conv3d'), norm_cfg=dict(type='BN3d', requires_grad=True), act_cfg=dict(type='ReLU', inplace=True), norm_eval=False, with_cp=False, zero_init_residual=True, **kwargs):
        super().__init__()
        self.gamma_w = gamma_w
        self.gamma_b = gamma_b
        self.gamma_d = gamma_d
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.base_channels = 24
        self.stage_blocks = [1, 2, 5, 3]
        self.base_channels = self._round_width(self.base_channels, self.gamma_w)
        self.stage_blocks = [self._round_repeats(x, self.gamma_d) for x in self.stage_blocks]
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.spatial_strides = spatial_strides
        assert len(spatial_strides) == num_stages
        self.frozen_stages = frozen_stages
        self.se_style = se_style
        assert self.se_style in ['all', 'half']
        self.se_ratio = se_ratio
        assert self.se_ratio is None or self.se_ratio > 0
        self.use_swish = use_swish
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual
        self.block = BlockX3D
        self.stage_blocks = self.stage_blocks[:num_stages]
        self.layer_inplanes = self.base_channels
        self._make_stem_layer()
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            inplanes = self.base_channels * 2 ** i
            planes = int(inplanes * self.gamma_b)
            res_layer = self.make_res_layer(self.block, self.layer_inplanes, inplanes, planes, num_blocks, spatial_stride=spatial_stride, se_style=self.se_style, se_ratio=self.se_ratio, use_swish=self.use_swish, norm_cfg=self.norm_cfg, conv_cfg=self.conv_cfg, act_cfg=self.act_cfg, with_cp=with_cp, **kwargs)
            self.layer_inplanes = inplanes
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
        self.feat_dim = self.base_channels * 2 ** (len(self.stage_blocks) - 1)
        self.conv5 = ConvModule(self.feat_dim, int(self.feat_dim * self.gamma_b), kernel_size=1, stride=1, padding=0, bias=False, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.feat_dim = int(self.feat_dim * self.gamma_b)

    @staticmethod
    def _round_width(width, multiplier, min_depth=8, divisor=8):
        """Round width of filters based on width multiplier."""
        if not multiplier:
            return width
        width *= multiplier
        min_depth = min_depth or divisor
        new_filters = max(min_depth, int(width + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * width:
            new_filters += divisor
        return int(new_filters)

    @staticmethod
    def _round_repeats(repeats, multiplier):
        """Round number of layers based on depth multiplier."""
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

    def make_res_layer(self, block, layer_inplanes, inplanes, planes, blocks, spatial_stride=1, se_style='half', se_ratio=None, use_swish=True, norm_cfg=None, act_cfg=None, conv_cfg=None, with_cp=False, **kwargs):
        """Build residual layer for ResNet3D.

        Args:
            block (nn.Module): Residual module to be built.
            layer_inplanes (int): Number of channels for the input feature
                of the res layer.
            inplanes (int): Number of channels for the input feature in each
                block, which equals to base_channels * gamma_w.
            planes (int): Number of channels for the output feature in each
                block, which equals to base_channel * gamma_w * gamma_b.
            blocks (int): Number of residual blocks.
            spatial_stride (int): Spatial strides in residual and conv layers.
                Default: 1.
            se_style (str): The style of inserting SE modules into BlockX3D,
                'half' denotes insert into half of the blocks, while 'all'
                denotes insert into all blocks. Default: 'half'.
            se_ratio (float | None): The reduction ratio of squeeze and
                excitation unit. If set as None, it means not using SE unit.
                Default: None.
            use_swish (bool): Whether to use swish as the activation function
                before and after the 3x3x3 conv. Default: True.
            conv_cfg (dict | None): Config for norm layers. Default: None.
            norm_cfg (dict | None): Config for norm layers. Default: None.
            act_cfg (dict | None): Config for activate layers. Default: None.
            with_cp (bool | None): Use checkpoint or not. Using checkpoint
                will save some memory while slowing down the training speed.
                Default: False.

        Returns:
            nn.Module: A residual layer for the given config.
        """
        downsample = None
        if spatial_stride != 1 or layer_inplanes != inplanes:
            downsample = ConvModule(layer_inplanes, inplanes, kernel_size=1, stride=(1, spatial_stride, spatial_stride), padding=0, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
        use_se = [False] * blocks
        if self.se_style == 'all':
            use_se = [True] * blocks
        elif self.se_style == 'half':
            use_se = [(i % 2 == 0) for i in range(blocks)]
        else:
            raise NotImplementedError
        layers = []
        layers.append(block(layer_inplanes, planes, inplanes, spatial_stride=spatial_stride, downsample=downsample, se_ratio=se_ratio if use_se[0] else None, use_swish=use_swish, norm_cfg=norm_cfg, conv_cfg=conv_cfg, act_cfg=act_cfg, with_cp=with_cp, **kwargs))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, inplanes, spatial_stride=1, se_ratio=se_ratio if use_se[i] else None, use_swish=use_swish, norm_cfg=norm_cfg, conv_cfg=conv_cfg, act_cfg=act_cfg, with_cp=with_cp, **kwargs))
        return nn.Sequential(*layers)

    def _make_stem_layer(self):
        """Construct the stem layers consists of a conv+norm+act module and a
        pooling layer."""
        self.conv1_s = ConvModule(self.in_channels, self.base_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False, conv_cfg=self.conv_cfg, norm_cfg=None, act_cfg=None)
        self.conv1_t = ConvModule(self.base_channels, self.base_channels, kernel_size=(5, 1, 1), stride=(1, 1, 1), padding=(2, 0, 0), groups=self.base_channels, bias=False, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.conv1_s.eval()
            self.conv1_t.eval()
            for param in self.conv1_s.parameters():
                param.requires_grad = False
            for param in self.conv1_t.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, BlockX3D):
                        constant_init(m.conv3.bn, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The feature of the input
            samples extracted by the backbone.
        """
        x = self.conv1_s(x)
        x = self.conv1_t(x)
        for layer_name in self.res_layers:
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
        x = self.conv5(x)
        return x

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


class Conv2plus1d(nn.Module):
    """(2+1)d Conv module for R(2+1)d backbone.

    https://arxiv.org/pdf/1711.11248.pdf.

    Args:
        in_channels (int): Same as ``nn.Conv3d``.
        out_channels (int): Same as ``nn.Conv3d``.
        kernel_size (Union[int, Tuple[int]]): Same as ``nn.Conv3d``.
        stride (Union[int, Tuple[int]]): Same as ``nn.Conv3d``. Defaults to 1.
        padding (Union[int, Tuple[int]]): Same as ``nn.Conv3d``. Defaults to 0.
        dilation (Union[int, Tuple[int]]): Same as ``nn.Conv3d``.
            Defaults to 1.
        groups (int): Same as ``nn.Conv3d``. Defaults to 1.
        bias (Union[bool, str]): If specified as `auto`, it will be decided by
            the norm_cfg. Bias will be set as True if norm_cfg is None,
            otherwise False.
        norm_cfg (Union[dict, ConfigDict]): Config for norm layers.
            Defaults to ``dict(type='BN3d')``.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'Union[int, Tuple[int]]', stride: 'Union[int, Tuple[int]]'=1, padding: 'Union[int, Tuple[int]]'=0, dilation: 'Union[int, Tuple[int]]'=1, groups: 'int'=1, bias: 'Union[bool, str]'=True, norm_cfg: 'ConfigType'=dict(type='BN3d')) ->None:
        super().__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        assert len(kernel_size) == len(stride) == len(padding) == 3
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.norm_cfg = norm_cfg
        self.output_padding = 0, 0, 0
        self.transposed = False
        mid_channels = 3 * (in_channels * out_channels * kernel_size[1] * kernel_size[2])
        mid_channels /= in_channels * kernel_size[1] * kernel_size[2] + 3 * out_channels
        mid_channels = int(mid_channels)
        self.conv_s = nn.Conv3d(in_channels, mid_channels, kernel_size=(1, kernel_size[1], kernel_size[2]), stride=(1, stride[1], stride[2]), padding=(0, padding[1], padding[2]), bias=bias)
        _, self.bn_s = build_norm_layer(self.norm_cfg, mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv_t = nn.Conv3d(mid_channels, out_channels, kernel_size=(kernel_size[0], 1, 1), stride=(stride[0], 1, 1), padding=(padding[0], 0, 0), bias=bias)
        self.init_weights()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        x = self.conv_s(x)
        x = self.bn_s(x)
        x = self.relu(x)
        x = self.conv_t(x)
        return x

    def init_weights(self) ->None:
        """Initiate the parameters from scratch."""
        kaiming_init(self.conv_s)
        kaiming_init(self.conv_t)
        constant_init(self.bn_s, 1, bias=0)


class ConvAudio(nn.Module):
    """Conv2d module for AudioResNet backbone.

        <https://arxiv.org/abs/2001.08740>`_.

    Args:
        in_channels (int): Same as ``nn.Conv2d``.
        out_channels (int): Same as ``nn.Conv2d``.
        kernel_size (Union[int, Tuple[int]]): Same as ``nn.Conv2d``.
        op (str): Operation to merge the output of freq
            and time feature map. Choices are ``sum`` and ``concat``.
            Defaults to ``concat``.
        stride (Union[int, Tuple[int]]): Same as ``nn.Conv2d``. Defaults to 1.
        padding (Union[int, Tuple[int]]): Same as ``nn.Conv2d``. Defaults to 0.
        dilation (Union[int, Tuple[int]]): Same as ``nn.Conv2d``.
            Defaults to 1.
        groups (int): Same as ``nn.Conv2d``. Defaults to 1.
        bias (Union[bool, str]): If specified as ``auto``, it will be decided
            by the ``norm_cfg``. Bias will be set as True if ``norm_cfg``
            is None, otherwise False. Defaults to False.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'Union[int, Tuple[int]]', op: 'str'='concat', stride: 'Union[int, Tuple[int]]'=1, padding: 'Union[int, Tuple[int]]'=0, dilation: 'Union[int, Tuple[int]]'=1, groups: 'int'=1, bias: 'Union[bool, str]'=False) ->None:
        super().__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert op in ['concat', 'sum']
        self.op = op
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.output_padding = 0, 0
        self.transposed = False
        self.conv_1 = ConvModule(in_channels, out_channels, kernel_size=(kernel_size[0], 1), stride=stride, padding=(kernel_size[0] // 2, 0), bias=bias, conv_cfg=dict(type='Conv'), norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'))
        self.conv_2 = ConvModule(in_channels, out_channels, kernel_size=(1, kernel_size[1]), stride=stride, padding=(0, kernel_size[1] // 2), bias=bias, conv_cfg=dict(type='Conv'), norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'))
        self.init_weights()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        x_1 = self.conv_1(x)
        x_2 = self.conv_2(x)
        if self.op == 'concat':
            out = torch.cat([x_1, x_2], 1)
        else:
            out = x_1 + x_2
        return out

    def init_weights(self) ->None:
        """Initiate the parameters from scratch."""
        kaiming_init(self.conv_1.conv)
        kaiming_init(self.conv_2.conv)
        constant_init(self.conv_1.bn, 1, bias=0)
        constant_init(self.conv_2.bn, 1, bias=0)


class SubBatchNorm3D(nn.Module):
    """Sub BatchNorm3d splits the batch dimension into N splits, and run BN on
    each of them separately (so that the stats are computed on each subset of
    examples (1/N of batch) independently). During evaluation, it aggregates
    the stats from all splits into one BN.

    Args:
        num_features (int): Dimensions of BatchNorm.
    """

    def __init__(self, num_features, **cfg):
        super(SubBatchNorm3D, self).__init__()
        self.num_features = num_features
        self.cfg_ = deepcopy(cfg)
        self.num_splits = self.cfg_.pop('num_splits', 1)
        self.num_features_split = self.num_features * self.num_splits
        self.cfg_['affine'] = False
        self.bn = nn.BatchNorm3d(num_features, **self.cfg_)
        self.split_bn = nn.BatchNorm3d(self.num_features_split, **self.cfg_)
        self.init_weights(cfg)

    def init_weights(self, cfg):
        """Initialize weights."""
        if cfg.get('affine', True):
            self.weight = torch.nn.Parameter(torch.ones(self.num_features))
            self.bias = torch.nn.Parameter(torch.zeros(self.num_features))
            self.affine = True
        else:
            self.affine = False

    def _get_aggregated_mean_std(self, means, stds, n):
        """Calculate aggregated mean and std."""
        mean = means.view(n, -1).sum(0) / n
        std = stds.view(n, -1).sum(0) / n + ((means.view(n, -1) - mean) ** 2).view(n, -1).sum(0) / n
        return mean.detach(), std.detach()

    def aggregate_stats(self):
        """Synchronize running_mean, and running_var to self.bn.

        Call this before eval, then call model.eval(); When eval, forward
        function will call self.bn instead of self.split_bn, During this time
        the running_mean, and running_var of self.bn has been obtained from
        self.split_bn.
        """
        if self.split_bn.track_running_stats:
            aggre_func = self._get_aggregated_mean_std
            self.bn.running_mean.data, self.bn.running_var.data = aggre_func(self.split_bn.running_mean, self.split_bn.running_var, self.num_splits)
        self.bn.num_batches_tracked = self.split_bn.num_batches_tracked.detach()

    def forward(self, x):
        """Defines the computation performed at every call."""
        if self.training:
            n, c, t, h, w = x.shape
            assert n % self.num_splits == 0
            x = x.view(n // self.num_splits, c * self.num_splits, t, h, w)
            x = self.split_bn(x)
            x = x.view(n, c, t, h, w)
        else:
            x = self.bn(x)
        if self.affine:
            x = x * self.weight.view(-1, 1, 1, 1)
            x = x + self.bias.view(-1, 1, 1, 1)
        return x


class AvgConsensus(nn.Module):
    """Average consensus module.

    Args:
        dim (int): Decide which dim consensus function to apply.
            Defaults to 1.
    """

    def __init__(self, dim: 'int'=1) ->None:
        super().__init__()
        self.dim = dim

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Defines the computation performed at every call."""
        return x.mean(dim=self.dim, keepdim=True)


class RelationModule(nn.Module):
    """Relation Module of TRN.

    Args:
        hidden_dim (int): The dimension of hidden layer of MLP in relation
            module.
        num_segments (int): Number of frame segments.
        num_classes (int): Number of classes to be classified.
    """

    def __init__(self, hidden_dim, num_segments, num_classes):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_segments = num_segments
        self.num_classes = num_classes
        bottleneck_dim = 512
        self.classifier = nn.Sequential(nn.ReLU(), nn.Linear(self.num_segments * self.hidden_dim, bottleneck_dim), nn.ReLU(), nn.Linear(bottleneck_dim, self.num_classes))

    def init_weights(self):
        """Use the default kaiming_uniform for all nn.linear layers."""
        pass

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.
        Returns:
            Tensor: The classification scores for input samples.
        """
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class RelationModuleMultiScale(nn.Module):
    """Relation Module with Multi Scale of TRN.

    Args:
        hidden_dim (int): The dimension of hidden layer of MLP in relation
            module.
        num_segments (int): Number of frame segments.
        num_classes (int): Number of classes to be classified.
    """

    def __init__(self, hidden_dim, num_segments, num_classes):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_segments = num_segments
        self.num_classes = num_classes
        self.scales = range(num_segments, 1, -1)
        self.relations_scales = []
        self.subsample_scales = []
        max_subsample = 3
        for scale in self.scales:
            relations_scale = list(itertools.combinations(range(self.num_segments), scale))
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(max_subsample, len(relations_scale)))
        assert len(self.relations_scales[0]) == 1
        bottleneck_dim = 256
        self.fc_fusion_scales = nn.ModuleList()
        for scale in self.scales:
            fc_fusion = nn.Sequential(nn.ReLU(), nn.Linear(scale * self.hidden_dim, bottleneck_dim), nn.ReLU(), nn.Linear(bottleneck_dim, self.num_classes))
            self.fc_fusion_scales.append(fc_fusion)

    def init_weights(self):
        """Use the default kaiming_uniform for all nn.linear layers."""
        pass

    def forward(self, x):
        act_all = x[:, self.relations_scales[0][0], :]
        act_all = act_all.view(act_all.size(0), self.scales[0] * self.hidden_dim)
        act_all = self.fc_fusion_scales[0](act_all)
        for scaleID in range(1, len(self.scales)):
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
            for idx in idx_relations_randomsample:
                act_relation = x[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.hidden_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_all += act_relation
        return act_all


def conv_block(in_channels: 'int', out_channels: 'int', kernel_size: 'int'=3, stride: 'int'=1) ->nn.Module:
    module = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False), nn.BatchNorm1d(out_channels), nn.ReLU())
    return module


class FPN(nn.Module):

    def __init__(self, in_channels_list: 'List', out_channels: 'int') ->None:
        super(FPN, self).__init__()
        inner_blocks = []
        layer_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = conv_block(in_channels, out_channels, 1, 1)
            layer_block = conv_block(out_channels, out_channels, 3, 1)
            inner_blocks.append(inner_block)
            layer_blocks.append(layer_block)
        self.inner_blocks = nn.ModuleList(inner_blocks)
        self.layer_blocks = nn.ModuleList(layer_blocks)

    def forward(self, x: 'Tensor') ->Tuple[Tensor]:
        last_inner = self.inner_blocks[-1](x[-1])
        results = [self.layer_blocks[-1](last_inner)]
        for feature, inner_block, layer_block in zip(x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]):
            if not inner_block:
                continue
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode='nearest')
            inner_lateral = inner_block(feature)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, layer_block(last_inner))
        return tuple(results)


class Backbone(nn.Module):

    def __init__(self, channels_list: 'List[tuple]') ->None:
        super(Backbone, self).__init__()
        self.num_layers = len(channels_list)
        layers = []
        for idx, channels_config in enumerate(channels_list):
            layer = conv_block(*channels_config)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, x: 'Tensor', query_fts: 'Tensor', position_fts: 'Tensor') ->Tuple[Tensor]:
        results = []
        for idx in range(self.num_layers):
            query_ft = query_fts[idx].unsqueeze(1).permute(0, 2, 1)
            position_ft = position_fts[idx]
            x = query_ft * x
            if idx == 0:
                x = torch.cat([x, position_ft], dim=1)
            x = self.layers[idx](x)
            results.append(x)
        return tuple(results)


class Scale(nn.Module):

    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return x * self.scale


class FCOSHead(torch.nn.Module):

    def __init__(self, in_channels: 'int', fcos_num_class: 'int', fcos_conv_layers: 'int', fcos_prior_prob: 'float', is_second_stage: 'bool') ->None:
        super(FCOSHead, self).__init__()
        num_classes = fcos_num_class - 1
        cls_tower = []
        bbox_tower = []
        for i in range(fcos_conv_layers):
            cls_tower.append(nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            cls_tower.append(nn.BatchNorm1d(in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            bbox_tower.append(nn.BatchNorm1d(in_channels))
            bbox_tower.append(nn.ReLU())
        self.cls_tower = nn.Sequential(*cls_tower)
        self.bbox_tower = nn.Sequential(*bbox_tower)
        self.cls_logits = nn.Conv1d(in_channels, num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv1d(in_channels, 2, kernel_size=3, stride=1, padding=1)
        self.mix_fc = nn.Sequential(nn.Conv1d(2 * in_channels, in_channels, kernel_size=1, stride=1), nn.BatchNorm1d(in_channels), nn.ReLU())
        self.iou_scores = nn.Sequential(nn.Conv1d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1), nn.BatchNorm1d(in_channels // 2), nn.ReLU(), nn.Conv1d(in_channels // 2, 1, kernel_size=1, stride=1))
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                torch.nn.init.normal_(module.weight, std=0.01)
                torch.nn.init.constant_(module.bias, 0)
        bias_value = -math.log((1 - fcos_prior_prob) / fcos_prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(3)])
        self.is_second_stage = is_second_stage

    def forward(self, x):
        logits = []
        bbox_reg = []
        iou_scores = []
        for idx, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)
            logits.append(self.cls_logits(cls_tower))
            bbox_reg_ = torch.exp(self.scales[idx](self.bbox_pred(box_tower)))
            if self.is_second_stage:
                bbox_reg_ = bbox_reg_.detach()
            bbox_reg.append(bbox_reg_)
            mix_feature = torch.cat([cls_tower, box_tower], dim=1)
            if self.is_second_stage:
                mix_feature = mix_feature.detach()
            mix_feature = self.mix_fc(mix_feature)
            iou_scores.append(self.iou_scores(mix_feature))
        return logits, bbox_reg, iou_scores


INF = 100000000


def IOULoss():

    def loss_fn(pred, target):
        pred_left = pred[:, 0]
        pred_right = pred[:, 1]
        target_left = target[:, 0]
        target_right = target[:, 1]
        intersect = torch.min(pred_right, target_right) + torch.min(pred_left, target_left)
        target_area = target_left + target_right
        pred_area = pred_left + pred_right
        union = target_area + pred_area - intersect
        losses = -torch.log((intersect + 1e-08) / (union + 1e-08))
        return losses.mean()
    return loss_fn


def SigmoidFocalLoss(alpha, gamma):

    def loss_fn(inputs, targets):
        loss = torchvision.ops.sigmoid_focal_loss(inputs=inputs, targets=targets, alpha=alpha, gamma=gamma, reduction='sum')
        return loss
    return loss_fn


def segment_tiou(box_a, box_b):
    inter_max_xy = torch.min(box_a[:, :, -1], box_b[:, :, -1])
    inter_min_xy = torch.max(box_a[:, :, 0], box_b[:, :, 0])
    inter = torch.clamp(inter_max_xy - inter_min_xy, min=0)
    union_max_xy = torch.max(box_a[:, :, -1], box_b[:, :, -1])
    union_min_xy = torch.min(box_a[:, :, 0], box_b[:, :, 0])
    union = torch.clamp(union_max_xy - union_min_xy, min=0)
    iou = inter / (union + 1e-06)
    return iou


class FCOSLossComputation(object):
    """This class computes the FCOS losses."""

    def __init__(self, focal_alpha, focal_gamma):
        self.cls_loss_fn = SigmoidFocalLoss(focal_alpha, focal_gamma)
        self.box_reg_loss_fn = IOULoss()
        self.centerness_loss_fn = nn.BCEWithLogitsLoss()
        self.iou_loss_fn = nn.SmoothL1Loss()

    def prepare_targets(self, points, targets):
        object_sizes_of_interest = [[-1, 6], [5.6, 11], [11, INF]]
        expanded_object_sizes_of_interest = []
        for idx, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = points_per_level.new_tensor(object_sizes_of_interest[idx])
            expanded_object_sizes_of_interest.append(object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1))
        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets = self.compute_targets_for_locations(points_all_level, targets, expanded_object_sizes_of_interest)
        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)
        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0))
            reg_targets_level_first.append(torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets], dim=0))
        return labels_level_first, reg_targets_level_first

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        ts = locations
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im * 32
            left = ts[:, None] - bboxes[None, 0]
            right = bboxes[None, 1] - ts[:, None]
            reg_targets_per_im = torch.cat([left, right], dim=1)
            is_in_boxes = reg_targets_per_im.min(dim=1)[0] > 0
            max_reg_targets_per_im = reg_targets_per_im.max(dim=1)[0]
            is_cared_in_the_level = (max_reg_targets_per_im >= object_sizes_of_interest[:, 0]) & (max_reg_targets_per_im <= object_sizes_of_interest[:, 1])
            locations_to_gt_area = bboxes[1] - bboxes[0]
            locations_to_gt_area = locations_to_gt_area.repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF
            _ = locations_to_gt_area.min(dim=1)
            locations_to_min_area, locations_to_gt_inds = _
            labels_per_im = reg_targets_per_im.new_ones(len(reg_targets_per_im))
            labels_per_im[locations_to_min_area == INF] = 0
            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
        return labels, reg_targets

    def __call__(self, locations, box_cls, box_regression, targets, iou_scores, is_first_stage=True):
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        labels, reg_targets = self.prepare_targets(locations, targets)
        box_cls_flatten = []
        box_regression_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        for idx in range(len(labels)):
            box_cls_flatten.append(box_cls[idx].permute(0, 2, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[idx].permute(0, 2, 1).reshape(-1, 2))
            labels_flatten.append(labels[idx].reshape(-1))
            reg_targets_flatten.append(reg_targets[idx].reshape(-1, 2))
        if not is_first_stage:
            merged_box_regression = torch.cat(box_regression, dim=-1).transpose(2, 1)
            merged_locations = torch.cat(locations, dim=0)
            full_locations = merged_locations[None, :].expand(merged_box_regression.size(0), -1).contiguous()
            pred_start = full_locations - merged_box_regression[:, :, 0]
            pred_end = full_locations + merged_box_regression[:, :, 1]
            predictions = torch.cat([pred_start.unsqueeze(-1), pred_end.unsqueeze(-1)], dim=-1) / 32
            predictions.clamp_(min=0, max=1)
            gt_box = targets[:, None, :]
            iou_target = segment_tiou(predictions, gt_box)
            iou_pred = torch.cat(iou_scores, dim=-1).squeeze().sigmoid()
            iou_pos_ind = iou_target > 0.9
            pos_iou_target = iou_target[iou_pos_ind]
            pos_iou_pred = iou_pred[iou_pos_ind]
            if iou_pos_ind.sum().item() == 0:
                iou_loss = torch.tensor([0.0])
            else:
                iou_loss = self.iou_loss_fn(pos_iou_pred, pos_iou_target)
        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)
        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        cls_loss = self.cls_loss_fn(box_cls_flatten, labels_flatten.unsqueeze(1)) / (pos_inds.numel() + N)
        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        if pos_inds.numel() > 0:
            reg_loss = self.box_reg_loss_fn(box_regression_flatten, reg_targets_flatten)
        else:
            reg_loss = box_regression_flatten.sum()
        if not is_first_stage:
            return cls_loss, reg_loss, iou_loss
        return cls_loss, reg_loss, torch.tensor([0.0])


def make_fcos_loss_evaluator(focal_alpha, focal_gamma):
    loss_evaluator = FCOSLossComputation(focal_alpha, focal_gamma)
    return loss_evaluator


class FCOSPostProcessor(torch.nn.Module):
    """Performs post-processing on the outputs of the RetinaNet boxes.

    This is only used in the testing.
    """

    def __init__(self, pre_nms_thresh, pre_nms_top_n, nms_thresh, fpn_post_nms_top_n, min_size, num_classes, is_first_stage):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(FCOSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.innerness_threshold = 0.15
        self.downsample_scale = 32
        self.is_first_stage = is_first_stage

    def forward_for_single_feature_map(self, locations, box_cls, box_regression, level, iou_scores):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, T = box_cls.shape
        box_cls = box_cls.permute(0, 2, 1).contiguous().sigmoid()
        iou_scores = iou_scores.permute(0, 2, 1).contiguous().sigmoid()
        box_regression = box_regression.permute(0, 2, 1)
        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)
        if not self.is_first_stage:
            box_cls = box_cls * iou_scores
        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]
            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1
            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]
            per_pre_nms_top_n = pre_nms_top_n[i]
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
            detections = torch.stack([per_locations - per_box_regression[:, 0], per_locations + per_box_regression[:, 1]], dim=1) / self.downsample_scale
            detections[:, 0].clamp_(min=0, max=1)
            detections[:, 1].clamp_(min=0, max=1)
            p_start, p_end = detections.unbind(dim=1)
            duration = p_end - p_start
            keep = (duration >= self.min_size).nonzero().squeeze(1)
            detections = detections[keep]
            temp_dict = {}
            temp_dict['detections'] = detections
            temp_dict['labels'] = per_class
            temp_dict['scores'] = torch.sqrt(per_box_cls)
            temp_dict['level'] = [level]
            temp_dict['locations'] = per_locations / 32
            results.append(temp_dict)
        return results

    def forward(self, locations, box_cls, box_regression, iou_scores):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        for i, (l, o, b, iou_s) in enumerate(zip(locations, box_cls, box_regression, iou_scores)):
            sampled_boxes.append(self.forward_for_single_feature_map(l, o, b, i, iou_s))
        boxlists = list(zip(*sampled_boxes))
        boxlists = self.select_over_all_levels(boxlists)
        return boxlists

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            dicts = boxlists[i]
            per_vid_scores = []
            per_vid_detections = []
            per_vid_labels = []
            per_vid_level = []
            per_vid_locations = []
            for per_scale_dict in dicts:
                if len(per_scale_dict['detections']) != 0:
                    per_vid_detections.append(per_scale_dict['detections'])
                if len(per_scale_dict['scores']) != 0:
                    per_vid_scores.append(per_scale_dict['scores'])
                if len(per_scale_dict['level']) != 0:
                    per_vid_level.append(per_scale_dict['level'] * len(per_scale_dict['detections']))
                if len(per_scale_dict['locations']) != 0:
                    per_vid_locations.append(per_scale_dict['locations'])
            if len(per_vid_detections) == 0:
                per_vid_detections = torch.Tensor([0, 1]).unsqueeze(0)
                per_vid_scores = torch.Tensor([1])
                per_vid_level = [[-1]]
                per_vid_locations = torch.Tensor([0.5])
            else:
                per_vid_detections = torch.cat(per_vid_detections, dim=0)
                per_vid_scores = torch.cat(per_vid_scores, dim=0)
                per_vid_level = per_vid_level
                per_vid_locations = torch.cat(per_vid_locations, dim=0)
            temp_dict = {}
            temp_dict['detections'] = per_vid_detections
            temp_dict['labels'] = per_vid_labels
            temp_dict['scores'] = per_vid_scores
            temp_dict['level'] = per_vid_level
            temp_dict['locations'] = per_vid_locations
            results.append(temp_dict)
        return results


def make_fcos_postprocessor(fcos_num_class, fcos_inference_thr, fcos_pre_nms_top_n, fcos_nms_thr, test_detections_per_img, is_first_stage):
    box_selector = FCOSPostProcessor(pre_nms_thresh=fcos_inference_thr, pre_nms_top_n=fcos_pre_nms_top_n, nms_thresh=fcos_nms_thr, fpn_post_nms_top_n=test_detections_per_img, min_size=0, num_classes=fcos_num_class, is_first_stage=is_first_stage)
    return box_selector


class FCOSModule(torch.nn.Module):

    def __init__(self, in_channels: 'int', fcos_num_class: 'int', fcos_conv_layers: 'int', fcos_prior_prob: 'float', fcos_inference_thr: 'float', fcos_pre_nms_top_n: 'int', fcos_nms_thr: 'float', test_detections_per_img: 'int', fpn_stride: 'int', focal_alpha: 'float', focal_gamma: 'float', is_first_stage: 'bool', is_second_stage: 'bool') ->None:
        super(FCOSModule, self).__init__()
        head = FCOSHead(in_channels=in_channels, fcos_num_class=fcos_num_class, fcos_conv_layers=fcos_conv_layers, fcos_prior_prob=fcos_prior_prob, is_second_stage=is_second_stage)
        self.is_first_stage = is_first_stage
        self.is_second_stage = is_second_stage
        box_selector_test = make_fcos_postprocessor(fcos_num_class, fcos_inference_thr, fcos_pre_nms_top_n, fcos_nms_thr, test_detections_per_img, is_first_stage)
        loss_evaluator = make_fcos_loss_evaluator(focal_alpha, focal_gamma)
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = fpn_stride

    def forward(self, features, targets=None):
        box_cls, box_regression, iou_scores = self.head(features)
        locations = self.compute_locations(features)
        if self.training:
            return self._forward_train(locations, box_cls, box_regression, targets, iou_scores)
        else:
            return self._forward_test(locations, box_cls, box_regression, targets, iou_scores)

    def _forward_train(self, locations, box_cls, box_regression, targets, iou_scores):
        loss_box_cls, loss_box_reg, loss_iou = self.loss_evaluator(locations, box_cls, box_regression, targets, iou_scores, self.is_first_stage)
        if self.is_second_stage:
            loss_box_cls = loss_box_cls.detach()
            loss_box_reg = loss_box_reg.detach()
        if self.is_first_stage:
            loss_iou = loss_iou.detach()
        losses = {'loss_cls': loss_box_cls, 'loss_reg': loss_box_reg, 'loss_iou': loss_iou}
        return None, losses

    def _forward_test(self, locations, box_cls, box_regression, targets, iou_scores):
        boxes = self.box_selector_test(locations, box_cls, box_regression, iou_scores)
        losses = None
        return boxes, losses

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            t = feature.size(-1)
            locations_per_level = self.compute_locations_per_level(t, self.fpn_strides[level], feature.device)
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, t, stride, device):
        shifts_t = torch.arange(0, t * stride, step=stride, dtype=torch.float32, device=device)
        shifts_t = shifts_t.reshape(-1)
        locations = shifts_t + stride / 2
        return locations


def apply_mask1d(attention: 'Tensor', image_locs: 'Tensor') ->Tensor:
    batch_size, num_loc = attention.size()
    tmp1 = torch.arange(num_loc, dtype=attention.dtype, device=attention.device)
    tmp1 = tmp1.expand(batch_size, num_loc)
    tmp2 = image_locs.unsqueeze(dim=1).expand(batch_size, num_loc)
    mask = tmp1 >= tmp2
    attention = attention.masked_fill(mask, -1e+30)
    return attention


class QueryEncoder(nn.Module):

    def __init__(self, vocab_size: 'int', hidden_dim: 'int'=512, embed_dim: 'int'=300, num_layers: 'int'=1, bidirection: 'bool'=True) ->None:
        super(QueryEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_embeddings=vocab_size + 1, embedding_dim=embed_dim, padding_idx=0)
        self.biLSTM = nn.LSTM(input_size=embed_dim, hidden_size=self.hidden_dim, num_layers=num_layers, dropout=0.0, batch_first=True, bidirectional=bidirection)
        self.W3 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.W2 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim * 2) for _ in range(3)])
        self.W1 = nn.Linear(hidden_dim * 2, 1)

    def extract_textual(self, q_encoding: 'Tensor', lstm_outputs: 'Tensor', q_length: 'Tensor', t: 'int'):
        q_cmd = self.W3(q_encoding).relu()
        q_cmd = self.W2[t](q_cmd)
        q_cmd = q_cmd[:, None, :] * lstm_outputs
        raw_att = self.W1(q_cmd).squeeze(-1)
        raw_att = apply_mask1d(raw_att, q_length)
        att = raw_att.softmax(dim=-1)
        cmd = torch.bmm(att[:, None, :], lstm_outputs).squeeze(1)
        return cmd

    def forward(self, query_tokens: 'Tensor', query_length: 'Tensor') ->List[Tensor]:
        self.biLSTM.flatten_parameters()
        query_embedding = self.embedding(query_tokens)
        query_embedding = pack_padded_sequence(query_embedding, query_length.cpu(), batch_first=True)
        output, _ = self.biLSTM(query_embedding)
        output, _ = pad_packed_sequence(output, batch_first=True)
        q_vector_list = []
        for i, length in enumerate(query_length):
            h1 = output[i][0]
            hs = output[i][length - 1]
            q_vector = torch.cat((h1, hs), dim=-1)
            q_vector_list.append(q_vector)
        q_vector = torch.stack(q_vector_list)
        outputs = []
        for cmd_t in range(3):
            query_feat = self.extract_textual(q_vector, output, query_length, cmd_t)
            outputs.append(query_feat)
        return outputs


class BaseWeightedLoss(nn.Module, metaclass=ABCMeta):
    """Base class for loss.

    All subclass should overwrite the ``_forward()`` method which returns the
    normal loss without loss weights.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    @abstractmethod
    def _forward(self, *args, **kwargs):
        """Forward function."""
        pass

    def forward(self, *args, **kwargs):
        """Defines the computation performed at every call.

        Args:
            *args: The positional arguments for the corresponding
                loss.
            **kwargs: The keyword arguments for the corresponding
                loss.

        Returns:
            torch.Tensor: The calculated loss.
        """
        ret = self._forward(*args, **kwargs)
        if isinstance(ret, dict):
            for k in ret:
                if 'loss' in k:
                    ret[k] *= self.loss_weight
        else:
            ret *= self.loss_weight
        return ret


def binary_logistic_regression_loss(reg_score, label, threshold=0.5, ratio_range=(1.05, 21), eps=1e-05):
    """Binary Logistic Regression Loss."""
    label = label.view(-1)
    reg_score = reg_score.contiguous().view(-1)
    pmask = (label > threshold).float()
    num_positive = max(torch.sum(pmask), 1)
    num_entries = len(label)
    ratio = num_entries / num_positive
    ratio = min(max(ratio, ratio_range[0]), ratio_range[1])
    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    loss = coef_1 * pmask * torch.log(reg_score + eps) + coef_0 * (1.0 - pmask) * torch.log(1.0 - reg_score + eps)
    loss = -torch.mean(loss)
    return loss


class BinaryLogisticRegressionLoss(nn.Module):
    """Binary Logistic Regression Loss.

    It will calculate binary logistic regression loss given reg_score and
    label.
    """

    def forward(self, reg_score, label, threshold=0.5, ratio_range=(1.05, 21), eps=1e-05):
        """Calculate Binary Logistic Regression Loss.

        Args:
                reg_score (torch.Tensor): Predicted score by model.
                label (torch.Tensor): Groundtruth labels.
                threshold (float): Threshold for positive instances.
                    Default: 0.5.
                ratio_range (tuple): Lower bound and upper bound for ratio.
                    Default: (1.05, 21)
                eps (float): Epsilon for small value. Default: 1e-5.

        Returns:
                torch.Tensor: Returned binary logistic loss.
        """
        return binary_logistic_regression_loss(reg_score, label, threshold, ratio_range, eps)


class BMNLoss(nn.Module):
    """BMN Loss.

    From paper https://arxiv.org/abs/1907.09702,
    code https://github.com/JJBOY/BMN-Boundary-Matching-Network.
    It will calculate loss for BMN Model. This loss is a weighted sum of

        1) temporal evaluation loss based on confidence score of start and
        end positions.
        2) proposal evaluation regression loss based on confidence scores of
        candidate proposals.
        3) proposal evaluation classification loss based on classification
        results of candidate proposals.
    """

    @staticmethod
    def tem_loss(pred_start, pred_end, gt_start, gt_end):
        """Calculate Temporal Evaluation Module Loss.

        This function calculate the binary_logistic_regression_loss for start
        and end respectively and returns the sum of their losses.

        Args:
            pred_start (torch.Tensor): Predicted start score by BMN model.
            pred_end (torch.Tensor): Predicted end score by BMN model.
            gt_start (torch.Tensor): Groundtruth confidence score for start.
            gt_end (torch.Tensor): Groundtruth confidence score for end.

        Returns:
            torch.Tensor: Returned binary logistic loss.
        """
        loss_start = binary_logistic_regression_loss(pred_start, gt_start)
        loss_end = binary_logistic_regression_loss(pred_end, gt_end)
        loss = loss_start + loss_end
        return loss

    @staticmethod
    def pem_reg_loss(pred_score, gt_iou_map, mask, high_temporal_iou_threshold=0.7, low_temporal_iou_threshold=0.3):
        """Calculate Proposal Evaluation Module Regression Loss.

        Args:
            pred_score (torch.Tensor): Predicted temporal_iou score by BMN.
            gt_iou_map (torch.Tensor): Groundtruth temporal_iou score.
            mask (torch.Tensor): Boundary-Matching mask.
            high_temporal_iou_threshold (float): Higher threshold of
                temporal_iou. Default: 0.7.
            low_temporal_iou_threshold (float): Higher threshold of
                temporal_iou. Default: 0.3.

        Returns:
            torch.Tensor: Proposal evaluation regression loss.
        """
        u_hmask = (gt_iou_map > high_temporal_iou_threshold).float()
        u_mmask = ((gt_iou_map <= high_temporal_iou_threshold) & (gt_iou_map > low_temporal_iou_threshold)).float()
        u_lmask = ((gt_iou_map <= low_temporal_iou_threshold) & (gt_iou_map > 0.0)).float()
        u_lmask = u_lmask * mask
        num_h = torch.sum(u_hmask)
        num_m = torch.sum(u_mmask)
        num_l = torch.sum(u_lmask)
        r_m = num_h / num_m
        u_smmask = torch.rand_like(gt_iou_map)
        u_smmask = u_mmask * u_smmask
        u_smmask = (u_smmask > 1.0 - r_m).float()
        r_l = num_h / num_l
        u_slmask = torch.rand_like(gt_iou_map)
        u_slmask = u_lmask * u_slmask
        u_slmask = (u_slmask > 1.0 - r_l).float()
        weights = u_hmask + u_smmask + u_slmask
        loss = F.mse_loss(pred_score * weights, gt_iou_map * weights)
        loss = 0.5 * torch.sum(loss * torch.ones_like(weights)) / torch.sum(weights)
        return loss

    @staticmethod
    def pem_cls_loss(pred_score, gt_iou_map, mask, threshold=0.9, ratio_range=(1.05, 21), eps=1e-05):
        """Calculate Proposal Evaluation Module Classification Loss.

        Args:
            pred_score (torch.Tensor): Predicted temporal_iou score by BMN.
            gt_iou_map (torch.Tensor): Groundtruth temporal_iou score.
            mask (torch.Tensor): Boundary-Matching mask.
            threshold (float): Threshold of temporal_iou for positive
                instances. Default: 0.9.
            ratio_range (tuple): Lower bound and upper bound for ratio.
                Default: (1.05, 21)
            eps (float): Epsilon for small value. Default: 1e-5

        Returns:
            torch.Tensor: Proposal evaluation classification loss.
        """
        pmask = (gt_iou_map > threshold).float()
        nmask = (gt_iou_map <= threshold).float()
        nmask = nmask * mask
        num_positive = max(torch.sum(pmask), 1)
        num_entries = num_positive + torch.sum(nmask)
        ratio = num_entries / num_positive
        ratio = torch.clamp(ratio, ratio_range[0], ratio_range[1])
        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio
        loss_pos = coef_1 * torch.log(pred_score + eps) * pmask
        loss_neg = coef_0 * torch.log(1.0 - pred_score + eps) * nmask
        loss = -1 * torch.sum(loss_pos + loss_neg) / num_entries
        return loss

    def forward(self, pred_bm, pred_start, pred_end, gt_iou_map, gt_start, gt_end, bm_mask, weight_tem=1.0, weight_pem_reg=10.0, weight_pem_cls=1.0):
        """Calculate Boundary Matching Network Loss.

        Args:
            pred_bm (torch.Tensor): Predicted confidence score for boundary
                matching map.
            pred_start (torch.Tensor): Predicted confidence score for start.
            pred_end (torch.Tensor): Predicted confidence score for end.
            gt_iou_map (torch.Tensor): Groundtruth score for boundary matching
                map.
            gt_start (torch.Tensor): Groundtruth temporal_iou score for start.
            gt_end (torch.Tensor): Groundtruth temporal_iou score for end.
            bm_mask (torch.Tensor): Boundary-Matching mask.
            weight_tem (float): Weight for tem loss. Default: 1.0.
            weight_pem_reg (float): Weight for pem regression loss.
                Default: 10.0.
            weight_pem_cls (float): Weight for pem classification loss.
                Default: 1.0.

        Returns:
            tuple([torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
                (loss, tem_loss, pem_reg_loss, pem_cls_loss). Loss is the bmn
                loss, tem_loss is the temporal evaluation loss, pem_reg_loss is
                the proposal evaluation regression loss, pem_cls_loss is the
                proposal evaluation classification loss.
        """
        pred_bm_reg = pred_bm[:, 0].contiguous()
        pred_bm_cls = pred_bm[:, 1].contiguous()
        gt_iou_map = gt_iou_map * bm_mask
        pem_reg_loss = self.pem_reg_loss(pred_bm_reg, gt_iou_map, bm_mask)
        pem_cls_loss = self.pem_cls_loss(pred_bm_cls, gt_iou_map, bm_mask)
        tem_loss = self.tem_loss(pred_start, pred_end, gt_start, gt_end)
        loss = weight_tem * tem_loss + weight_pem_reg * pem_reg_loss + weight_pem_cls * pem_cls_loss
        return loss, tem_loss, pem_reg_loss, pem_cls_loss


class CrossEntropyLoss(BaseWeightedLoss):
    """Cross Entropy Loss.

    Support two kinds of labels and their corresponding loss type. It's worth
    mentioning that loss type will be detected by the shape of ``cls_score``
    and ``label``.
    1) Hard label: This label is an integer array and all of the elements are
        in the range [0, num_classes - 1]. This label's shape should be
        ``cls_score``'s shape with the `num_classes` dimension removed.
    2) Soft label(probability distribution over classes): This label is a
        probability distribution and all of the elements are in the range
        [0, 1]. This label's shape must be the same as ``cls_score``. For now,
        only 2-dim soft label is supported.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Defaults to 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Defaults to None.
    """

    def __init__(self, loss_weight: 'float'=1.0, class_weight: 'Optional[List[float]]'=None) ->None:
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)

    def _forward(self, cls_score: 'torch.Tensor', label: 'torch.Tensor', **kwargs) ->torch.Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.

        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        if cls_score.size() == label.size():
            assert cls_score.dim() == 2, 'Only support 2-dim soft label'
            assert len(kwargs) == 0, f'For now, no extra args are supported for soft label, but get {kwargs}'
            lsm = F.log_softmax(cls_score, 1)
            if self.class_weight is not None:
                self.class_weight = self.class_weight
                lsm = lsm * self.class_weight.unsqueeze(0)
            loss_cls = -(label * lsm).sum(1)
            if self.class_weight is not None:
                loss_cls = loss_cls.sum() / torch.sum(self.class_weight.unsqueeze(0) * label)
            else:
                loss_cls = loss_cls.mean()
        else:
            if self.class_weight is not None:
                assert 'weight' not in kwargs, "The key 'weight' already exists."
                kwargs['weight'] = self.class_weight
            loss_cls = F.cross_entropy(cls_score, label, **kwargs)
        return loss_cls


class BCELossWithLogits(BaseWeightedLoss):
    """Binary Cross Entropy Loss with logits.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Defaults to 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Defaults to None.
    """

    def __init__(self, loss_weight: 'float'=1.0, class_weight: 'Optional[List[float]]'=None) ->None:
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)

    def _forward(self, cls_score: 'torch.Tensor', label: 'torch.Tensor', **kwargs) ->torch.Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                bce loss with logits.

        Returns:
            torch.Tensor: The returned bce loss with logits.
        """
        if self.class_weight is not None:
            assert 'weight' not in kwargs, "The key 'weight' already exists."
            kwargs['weight'] = self.class_weight
        loss_cls = F.binary_cross_entropy_with_logits(cls_score, label, **kwargs)
        return loss_cls


class CBFocalLoss(BaseWeightedLoss):
    """Class Balanced Focal Loss. Adapted from https://github.com/abhinanda-
    punnakkal/BABEL/. This loss is used in the skeleton-based action
    recognition baseline for BABEL.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Defaults to 1.0.
        samples_per_cls (list[int]): The number of samples per class.
            Defaults to [].
        beta (float): Hyperparameter that controls the per class loss weight.
            Defaults to 0.9999.
        gamma (float): Hyperparameter of the focal loss. Defaults to 2.0.
    """

    def __init__(self, loss_weight: 'float'=1.0, samples_per_cls: 'List[int]'=[], beta: 'float'=0.9999, gamma: 'float'=2.0) ->None:
        super().__init__(loss_weight=loss_weight)
        self.samples_per_cls = samples_per_cls
        self.beta = beta
        self.gamma = gamma
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(weights)
        self.weights = weights
        self.num_classes = len(weights)

    def _forward(self, cls_score: 'torch.Tensor', label: 'torch.Tensor', **kwargs) ->torch.Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                bce loss with logits.

        Returns:
            torch.Tensor: The returned bce loss with logits.
        """
        weights = torch.tensor(self.weights).float()
        label_one_hot = F.one_hot(label, self.num_classes).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(label_one_hot.shape[0], 1) * label_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.num_classes)
        BCELoss = F.binary_cross_entropy_with_logits(input=cls_score, target=label_one_hot, reduction='none')
        modulator = 1.0
        if self.gamma:
            modulator = torch.exp(-self.gamma * label_one_hot * cls_score - self.gamma * torch.log(1 + torch.exp(-1.0 * cls_score)))
        loss = modulator * BCELoss
        weighted_loss = weights * loss
        focal_loss = torch.sum(weighted_loss)
        focal_loss /= torch.sum(label_one_hot)
        return focal_loss


class HVULoss(BaseWeightedLoss):
    """Calculate the BCELoss for HVU.

    Args:
        categories (tuple[str]): Names of tag categories, tags are organized in
            this order. Default: ['action', 'attribute', 'concept', 'event',
            'object', 'scene'].
        category_nums (tuple[int]): Number of tags for each category. Default:
            (739, 117, 291, 69, 1678, 248).
        category_loss_weights (tuple[float]): Loss weights of categories, it
            applies only if `loss_type == 'individual'`. The loss weights will
            be normalized so that the sum equals to 1, so that you can give any
            positive number as loss weight. Default: (1, 1, 1, 1, 1, 1).
        loss_type (str): The loss type we calculate, we can either calculate
            the BCELoss for all tags, or calculate the BCELoss for tags in each
            category. Choices are 'individual' or 'all'. Default: 'all'.
        with_mask (bool): Since some tag categories are missing for some video
            clips. If `with_mask == True`, we will not calculate loss for these
            missing categories. Otherwise, these missing categories are treated
            as negative samples.
        reduction (str): Reduction way. Choices are 'mean' or 'sum'. Default:
            'mean'.
        loss_weight (float): The loss weight. Default: 1.0.
    """

    def __init__(self, categories=('action', 'attribute', 'concept', 'event', 'object', 'scene'), category_nums=(739, 117, 291, 69, 1678, 248), category_loss_weights=(1, 1, 1, 1, 1, 1), loss_type='all', with_mask=False, reduction='mean', loss_weight=1.0):
        super().__init__(loss_weight)
        self.categories = categories
        self.category_nums = category_nums
        self.category_loss_weights = category_loss_weights
        assert len(self.category_nums) == len(self.category_loss_weights)
        for category_loss_weight in self.category_loss_weights:
            assert category_loss_weight >= 0
        self.loss_type = loss_type
        self.with_mask = with_mask
        self.reduction = reduction
        self.category_startidx = [0]
        for i in range(len(self.category_nums) - 1):
            self.category_startidx.append(self.category_startidx[-1] + self.category_nums[i])
        assert self.loss_type in ['individual', 'all']
        assert self.reduction in ['mean', 'sum']

    def _forward(self, cls_score, label, mask, category_mask):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            mask (torch.Tensor): The mask of tags. 0 indicates that the
                category of this tag is missing in the label of the video.
            category_mask (torch.Tensor): The category mask. For each sample,
                it's a tensor with length `len(self.categories)`, denotes that
                if the category is labeled for this video.

        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        if self.loss_type == 'all':
            loss_cls = F.binary_cross_entropy_with_logits(cls_score, label, reduction='none')
            if self.with_mask:
                w_loss_cls = mask * loss_cls
                w_loss_cls = torch.sum(w_loss_cls, dim=1)
                if self.reduction == 'mean':
                    w_loss_cls = w_loss_cls / torch.sum(mask, dim=1)
                w_loss_cls = torch.mean(w_loss_cls)
                return dict(loss_cls=w_loss_cls)
            if self.reduction == 'sum':
                loss_cls = torch.sum(loss_cls, dim=-1)
            return dict(loss_cls=torch.mean(loss_cls))
        if self.loss_type == 'individual':
            losses = {}
            loss_weights = {}
            for name, num, start_idx in zip(self.categories, self.category_nums, self.category_startidx):
                category_score = cls_score[:, start_idx:start_idx + num]
                category_label = label[:, start_idx:start_idx + num]
                category_loss = F.binary_cross_entropy_with_logits(category_score, category_label, reduction='none')
                if self.reduction == 'mean':
                    category_loss = torch.mean(category_loss, dim=1)
                elif self.reduction == 'sum':
                    category_loss = torch.sum(category_loss, dim=1)
                idx = self.categories.index(name)
                if self.with_mask:
                    category_mask_i = category_mask[:, idx].reshape(-1)
                    if torch.sum(category_mask_i) < 0.5:
                        losses[f'{name}_LOSS'] = torch.tensor(0.0, device=get_device())
                        loss_weights[f'{name}_LOSS'] = 0.0
                        continue
                    category_loss = torch.sum(category_loss * category_mask_i)
                    category_loss = category_loss / torch.sum(category_mask_i)
                else:
                    category_loss = torch.mean(category_loss)
                losses[f'{name}_LOSS'] = category_loss
                loss_weights[f'{name}_LOSS'] = self.category_loss_weights[idx]
            loss_weight_sum = sum(loss_weights.values())
            loss_weights = {k: (v / loss_weight_sum) for k, v in loss_weights.items()}
            loss_cls = sum([(losses[k] * loss_weights[k]) for k in losses])
            losses['loss_cls'] = loss_cls
            losses.update({(k + '_weight'): torch.tensor(v) for k, v in loss_weights.items()})
            return losses
        else:
            raise ValueError(f"loss_type should be 'all' or 'individual', but got {self.loss_type}")


class NLLLoss(BaseWeightedLoss):
    """NLL Loss.

    It will calculate NLL loss given cls_score and label.
    """

    def _forward(self, cls_score, label, **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate nll loss.

        Returns:
            torch.Tensor: The returned nll loss.
        """
        loss_cls = F.nll_loss(cls_score, label, **kwargs)
        return loss_cls


class OHEMHingeLoss(torch.autograd.Function):
    """This class is the core implementation for the completeness loss in
    paper.

    It compute class-wise hinge loss and performs online hard example mining
    (OHEM).
    """

    @staticmethod
    def forward(ctx, pred, labels, is_positive, ohem_ratio, group_size):
        """Calculate OHEM hinge loss.

        Args:
            pred (torch.Tensor): Predicted completeness score.
            labels (torch.Tensor): Groundtruth class label.
            is_positive (int): Set to 1 when proposals are positive and
                set to -1 when proposals are incomplete.
            ohem_ratio (float): Ratio of hard examples.
            group_size (int): Number of proposals sampled per video.

        Returns:
            torch.Tensor: Returned class-wise hinge loss.
        """
        num_samples = pred.size(0)
        if num_samples != len(labels):
            raise ValueError(f'Number of samples should be equal to that of labels, but got {num_samples} samples and {len(labels)} labels.')
        losses = torch.zeros(num_samples, device=pred.device)
        slopes = torch.zeros(num_samples, device=pred.device)
        for i in range(num_samples):
            losses[i] = max(0, 1 - is_positive * pred[i, labels[i] - 1])
            slopes[i] = -is_positive if losses[i] != 0 else 0
        losses = losses.view(-1, group_size).contiguous()
        sorted_losses, indices = torch.sort(losses, dim=1, descending=True)
        keep_length = int(group_size * ohem_ratio)
        loss = torch.zeros(1, device=pred.device)
        for i in range(losses.size(0)):
            loss += sorted_losses[i, :keep_length].sum()
        ctx.loss_index = indices[:, :keep_length]
        ctx.labels = labels
        ctx.slopes = slopes
        ctx.shape = pred.size()
        ctx.group_size = group_size
        ctx.num_groups = losses.size(0)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """Defines a formula for differentiating the operation with backward
        mode automatic differentiation."""
        labels = ctx.labels
        slopes = ctx.slopes
        grad_in = torch.zeros(ctx.shape, device=ctx.slopes.device)
        for group in range(ctx.num_groups):
            for idx in ctx.loss_index[group]:
                loc = idx + group * ctx.group_size
                grad_in[loc, labels[loc] - 1] = slopes[loc] * grad_output.data[0]
        return torch.autograd.Variable(grad_in), None, None, None, None


class SSNLoss(nn.Module):

    @staticmethod
    def activity_loss(activity_score, labels, activity_indexer):
        """Activity Loss.

        It will calculate activity loss given activity_score and label.

        Args：
            activity_score (torch.Tensor): Predicted activity score.
            labels (torch.Tensor): Groundtruth class label.
            activity_indexer (torch.Tensor): Index slices of proposals.

        Returns:
            torch.Tensor: Returned cross entropy loss.
        """
        pred = activity_score[activity_indexer, :]
        gt = labels[activity_indexer]
        return F.cross_entropy(pred, gt)

    @staticmethod
    def completeness_loss(completeness_score, labels, completeness_indexer, positive_per_video, incomplete_per_video, ohem_ratio=0.17):
        """Completeness Loss.

        It will calculate completeness loss given completeness_score and label.

        Args：
            completeness_score (torch.Tensor): Predicted completeness score.
            labels (torch.Tensor): Groundtruth class label.
            completeness_indexer (torch.Tensor): Index slices of positive and
                incomplete proposals.
            positive_per_video (int): Number of positive proposals sampled
                per video.
            incomplete_per_video (int): Number of incomplete proposals sampled
                pre video.
            ohem_ratio (float): Ratio of online hard example mining.
                Default: 0.17.

        Returns:
            torch.Tensor: Returned class-wise completeness loss.
        """
        pred = completeness_score[completeness_indexer, :]
        gt = labels[completeness_indexer]
        pred_dim = pred.size(1)
        pred = pred.view(-1, positive_per_video + incomplete_per_video, pred_dim)
        gt = gt.view(-1, positive_per_video + incomplete_per_video)
        positive_pred = pred[:, :positive_per_video, :].contiguous().view(-1, pred_dim)
        incomplete_pred = pred[:, positive_per_video:, :].contiguous().view(-1, pred_dim)
        positive_loss = OHEMHingeLoss.apply(positive_pred, gt[:, :positive_per_video].contiguous().view(-1), 1, 1.0, positive_per_video)
        incomplete_loss = OHEMHingeLoss.apply(incomplete_pred, gt[:, positive_per_video:].contiguous().view(-1), -1, ohem_ratio, incomplete_per_video)
        num_positives = positive_pred.size(0)
        num_incompletes = int(incomplete_pred.size(0) * ohem_ratio)
        return (positive_loss + incomplete_loss) / float(num_positives + num_incompletes)

    @staticmethod
    def classwise_regression_loss(bbox_pred, labels, bbox_targets, regression_indexer):
        """Classwise Regression Loss.

        It will calculate classwise_regression loss given
        class_reg_pred and targets.

        Args：
            bbox_pred (torch.Tensor): Predicted interval center and span
                of positive proposals.
            labels (torch.Tensor): Groundtruth class label.
            bbox_targets (torch.Tensor): Groundtruth center and span
                of positive proposals.
            regression_indexer (torch.Tensor): Index slices of
                positive proposals.

        Returns:
            torch.Tensor: Returned class-wise regression loss.
        """
        pred = bbox_pred[regression_indexer, :, :]
        gt = labels[regression_indexer]
        reg_target = bbox_targets[regression_indexer, :]
        class_idx = gt.data - 1
        classwise_pred = pred[:, class_idx, :]
        classwise_reg_pred = torch.cat((torch.diag(classwise_pred[:, :, 0]).view(-1, 1), torch.diag(classwise_pred[:, :, 1]).view(-1, 1)), dim=1)
        loss = F.smooth_l1_loss(classwise_reg_pred.view(-1), reg_target.view(-1)) * 2
        return loss

    def forward(self, activity_score, completeness_score, bbox_pred, proposal_type, labels, bbox_targets, train_cfg):
        """Calculate Boundary Matching Network Loss.

        Args:
            activity_score (torch.Tensor): Predicted activity score.
            completeness_score (torch.Tensor): Predicted completeness score.
            bbox_pred (torch.Tensor): Predicted interval center and span
                of positive proposals.
            proposal_type (torch.Tensor): Type index slices of proposals.
            labels (torch.Tensor): Groundtruth class label.
            bbox_targets (torch.Tensor): Groundtruth center and span
                of positive proposals.
            train_cfg (dict): Config for training.

        Returns:
            dict([torch.Tensor, torch.Tensor, torch.Tensor]):
                (loss_activity, loss_completeness, loss_reg).
                Loss_activity is the activity loss, loss_completeness is
                the class-wise completeness loss,
                loss_reg is the class-wise regression loss.
        """
        self.sampler = train_cfg.ssn.sampler
        self.loss_weight = train_cfg.ssn.loss_weight
        losses = dict()
        proposal_type = proposal_type.view(-1)
        labels = labels.view(-1)
        activity_indexer = ((proposal_type == 0) + (proposal_type == 2)).nonzero().squeeze(1)
        completeness_indexer = ((proposal_type == 0) + (proposal_type == 1)).nonzero().squeeze(1)
        total_ratio = self.sampler.positive_ratio + self.sampler.background_ratio + self.sampler.incomplete_ratio
        positive_per_video = int(self.sampler.num_per_video * (self.sampler.positive_ratio / total_ratio))
        background_per_video = int(self.sampler.num_per_video * (self.sampler.background_ratio / total_ratio))
        incomplete_per_video = self.sampler.num_per_video - positive_per_video - background_per_video
        losses['loss_activity'] = self.activity_loss(activity_score, labels, activity_indexer)
        losses['loss_completeness'] = self.completeness_loss(completeness_score, labels, completeness_indexer, positive_per_video, incomplete_per_video, ohem_ratio=positive_per_video / incomplete_per_video)
        losses['loss_completeness'] *= self.loss_weight.comp_loss_weight
        if bbox_pred is not None:
            regression_indexer = (proposal_type == 0).nonzero().squeeze(1)
            bbox_targets = bbox_targets.view(-1, 2)
            losses['loss_reg'] = self.classwise_regression_loss(bbox_pred, labels, bbox_targets, regression_indexer)
            losses['loss_reg'] *= self.loss_weight.reg_loss_weight
        return losses


class TemporalAttentionBeit(nn.Module):
    """temporal attention using BeitAttention."""

    def __init__(self, config: 'BeitConfig'):
        """TODO: to be defined."""
        super().__init__()
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = BeitAttention(config, window_size=None)
        self.scale = nn.Parameter(config.temporal_model_init_value * torch.ones(config.hidden_size), requires_grad=True)
        self.drop_path = BeitDropPath(config.drop_path_rate)

    def forward(self, hidden_states: 'torch.Tensor'):
        """forward function.

        Args:
            hidden_states (torch.Tensor): The input. Shape: [b,t,l,c]

        Returns: TODO
        """
        b = hidden_states.shape[0]
        output = einops.rearrange(hidden_states, 'b t l c -> (b l) t c')
        output = self.layernorm_before(output)
        output = self.attention(output)
        output = einops.rearrange(output[0], '(b l) t c -> b t l c', b=b)
        return hidden_states + self.drop_path(output[0]) * self.scale


class BeitPooler3D(nn.Module):

    def __init__(self, config: 'BeitConfig') ->None:
        super().__init__()
        self.num_prompts = config.add_k_prompts
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) if config.use_mean_pooling else None

    def forward(self, hidden_states: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): Shape: [B,T,L,C]
        """
        if self.layernorm is not None:
            if self.num_prompts > 0:
                patch_tokens = hidden_states[:, :, 1:-self.num_prompts, :]
            else:
                patch_tokens = hidden_states[:, :, 1:, :]
            pooled_output = self.layernorm(patch_tokens.mean(2))
        else:
            pooled_output = hidden_states[:, :, 0]
        return pooled_output


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type
    embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        self.config = config

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:seq_length + past_key_values_length]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):

    def __init__(self, config, is_cross_attention):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, 'embedding_size'):
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            self.key = nn.Linear(config.encoder_width, self.all_head_size)
            self.value = nn.Linear(config.encoder_width, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.save_attention = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        past_key_value = key_layer, value_layer
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding
            if self.position_embedding_type == 'relative_key':
                relative_position_scores = torch.einsum('bhld,lrd->bhlr', query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == 'relative_key_query':
                relative_position_scores_query = torch.einsum('bhld,lrd->bhlr', query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum('bhrd,lrd->bhlr', key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        if is_cross_attention and self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)
        attention_probs_dropped = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask
        context_layer = torch.matmul(attention_probs_dropped, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs, attention_scores) if output_attentions else (context_layer,)
        outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.self = BertSelfAttention(config, is_cross_attention)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads)
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self, config, layer_num):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.has_cross_attention = layer_num >= config.fusion_layer
        if self.has_cross_attention:
            self.crossattention = BertAttention(config, is_cross_attention=True)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions, past_key_value=self_attn_past_key_value)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]
        if self.has_cross_attention:
            assert encoder_hidden_states is not None, 'encoder_hidden_states must be given for cross-attention layers'
            if type(encoder_hidden_states) == list:
                cross_attention_outputs = self.crossattention(attention_output, attention_mask, head_mask, encoder_hidden_states[(self.layer_num - self.config.fusion_layer) % len(encoder_hidden_states)], encoder_attention_mask[(self.layer_num - self.config.fusion_layer) % len(encoder_hidden_states)], output_attentions=output_attentions)
                attention_output = cross_attention_outputs[0]
                outputs = outputs + cross_attention_outputs[1:-1]
            else:
                cross_attention_outputs = self.crossattention(attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions=output_attentions)
                attention_output = cross_attention_outputs[0]
                outputs = outputs + cross_attention_outputs[1:-1]
        layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
        outputs = (layer_output,) + outputs
        outputs = outputs + (present_key_value,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config, i) for i in range(config.num_hidden_layers)])
        logger = MMLogger.get_current_instance()
        logger.info(f'build bert with cross_module: {config.cross_module}')

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, use_cache=None, output_attentions=False, output_hidden_states=False, return_dict=True, mode='multi_modal', normalize_attention=True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        if mode == 'text' or mode == 'temporal':
            start_layer = 0
            output_layer = self.config.fusion_layer
        elif mode == 'fusion':
            start_layer = self.config.fusion_layer
            output_layer = self.config.num_hidden_layers
        elif mode == 'multi_modal':
            start_layer = 0
            output_layer = self.config.num_hidden_layers
        for i in range(start_layer, output_layer):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if getattr(self.config, 'gradient_checkpointing', False) and self.training:
                if use_cache:
                    logger = MMLogger.get_current_instance()
                    logger.warn('`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting `use_cache=False`...')
                    use_cache = False

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)
                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(layer_module), hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, use_reentrant=False)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += layer_outputs[-1],
            if output_attentions:
                offset = int(normalize_attention)
                all_self_attentions = all_self_attentions + (layer_outputs[2 - offset],)
                if hasattr(layer_module, 'crossattention'):
                    all_cross_attentions = all_cross_attentions + (layer_outputs[4 - offset],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=next_decoder_cache, hidden_states=all_hidden_states, attentions=all_self_attentions, cross_attentions=all_cross_attentions)


class BertPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class STAdapter(nn.Module):
    """ST Adapter."""

    def __init__(self, kernel_size=(3, 3, 3), input_dim=768, hidden_dim=384, img_size=224, patch_size=16, drop_prob=0.1):
        super(STAdapter, self).__init__()
        self.kernel_size = kernel_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.h = self.w = img_size // patch_size
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.act = nn.ReLU()
        self.conv = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding='same', groups=hidden_dim)
        self.droppath = DropPath(drop_prob=drop_prob)
        self.scale = nn.parameter.Parameter(torch.zeros([]))

    def forward(self, x: 'torch.Tensor'):
        """forward.

        Args:
            x (torch.Tensor): input features.
            Shape: [bs, nframes, l, c]. l = 1 + h*w

        Returns: features after adapter. The same shape as input.
        """
        if x.shape[1] == 1:
            return x
        shortcut = x
        x = self.linear1(x)
        cls = x[:, :, :1, :]
        tokens = x[:, :, 1:, :]
        tokens = einops.rearrange(tokens, 'b t (h w) c -> b c t h w', h=self.h).contiguous()
        tokens = self.conv(tokens)
        tokens = einops.rearrange(tokens, 'b c t h w -> b t (h w) c')
        x = torch.cat([cls, tokens], dim=2)
        x = self.act(x)
        x = self.linear2(x)
        return shortcut + self.scale * self.droppath(x)


class TemporalAttention(nn.Module):
    """perform temporal self-attention."""

    def __init__(self, input_dim=768, droppath_rate=0.1):
        """

        Kwargs:
            input_dim (int): The input feature dimension.


        """
        super().__init__()
        self._input_dim = input_dim
        self.temporal_attn = MultiheadAttention(input_dim, num_heads=input_dim // 64)
        self.norm = LayerNorm(input_dim, eps=1e-12)
        self.linear = Linear(input_dim, input_dim)
        self.droppath = DropPath(droppath_rate)
        self.scale = nn.parameter.Parameter(torch.zeros([]))

    def forward(self, x: 'torch.Tensor'):
        """forward.

        Args:
            x (torch.Tensor): input features.
            Shape: [bs, nframes, l, c]. l = 1 + h*w

        Returns: features after adapter. The same shape as input.
        """
        if x.shape[1] == 1:
            return x
        shortcut = x
        x = einops.rearrange(x, 'b t l c -> t (b l) c')
        x = self.norm(x)
        x = self.temporal_attn(x, x, x)[0]
        x = einops.rearrange(x, 't (b l) c -> b t l c', b=shortcut.shape[0])
        return shortcut + self.scale * self.droppath(x)


class WindowTemporalAttention(nn.Module):
    """perform windowed temporal self-attention."""

    def __init__(self, input_dim=768, droppath_rate=0.1, window_size=(2, 2)):
        """

        Kwargs:
            input_dim (int): The input feature dimension.


        """
        super().__init__()
        self._input_dim = input_dim
        self.temporal_attn = MultiheadAttention(input_dim, num_heads=input_dim // 64)
        self.norm = LayerNorm(input_dim, eps=1e-12)
        self.droppath = DropPath(droppath_rate)
        self.scale = nn.parameter.Parameter(torch.zeros([]))
        self.wh, self.ww = window_size

    def forward(self, x: 'torch.Tensor'):
        """forward.

        Args:
            x (torch.Tensor): input features.
            Shape: [bs, nframes, l, c]. l = 1 + h*w

        Returns: features after adapter. The same shape as input.
        """
        if x.shape[1] == 1:
            return x
        shortcut = x
        h = w = int(math.sqrt(x.shape[2] - 1))
        cls_token = x[:, :, :1, :]
        x = einops.rearrange(x[:, :, 1:, :], 'b t (nh wh nw ww) c -> (t wh ww) (b nh nw) c', nh=h // self.wh, wh=self.wh, nw=w // self.ww, ww=self.ww)
        x = self.norm(x)
        x = self.temporal_attn(x, x, x)[0]
        x = einops.rearrange(x, '(t wh ww) (b nh nw) c -> b t (nh wh nw ww) c', wh=self.wh, ww=self.ww, nh=h // self.wh, nw=w // self.ww)
        x = torch.concat([cls_token, x], dim=2)
        return shortcut + self.scale * self.droppath(x)


class X_CLIP(nn.Module):
    """perform windowed temporal self-attention."""

    def __init__(self, input_dim=768, droppath_rate=0.1, num_prompts=1):
        """

        Kwargs:
            input_dim (int): The input feature dimension.


        """
        super().__init__()
        d_model = input_dim
        self.message_fc = nn.Linear(d_model, d_model)
        self.message_ln = LayerNorm(d_model, eps=1e-12)
        self.message_attn = nn.MultiheadAttention(d_model, d_model // 64)
        self.num_prompts = num_prompts
        self.droppath = DropPath(droppath_rate)

    def forward(self, x: 'torch.Tensor'):
        """forward.

        Args:
            x (torch.Tensor): input features.
            Shape: [bs, nframes, l, c]. l = 1 + h*w

        Returns: features after adapter. The same shape as input.
        """
        if x.shape[1] == 1:
            return x
        msg_token = self.message_ln(self.message_fc(x[:, :, 0, :]))
        msg_token = rearrange(msg_token, 'b t c -> t b c')
        msg_token = msg_token + self.droppath(self.message_attn(msg_token, msg_token, msg_token)[0])
        msg_token = rearrange(msg_token, 't b c -> b t c')
        x = torch.cat([x[:, :, :-1, :], msg_token.unsqueeze(2)], dim=2)
        return x


class DownSample(nn.Module):
    """DownSample modules.

    It uses convolution and maxpooling to downsample the input feature,
    and specifies downsample position to determine `pool-conv` or `conv-pool`.

    Args:
        in_channels (int): Channel number of input features.
        out_channels (int): Channel number of output feature.
        kernel_size (int or Tuple[int]): Same as :class:`ConvModule`.
            Defaults to ``(3, 1, 1)``.
        stride (int or Tuple[int]): Same as :class:`ConvModule`.
            Defaults to ``(1, 1, 1)``.
        padding (int or Tuple[int]): Same as :class:`ConvModule`.
            Defaults to ``(1, 0, 0)``.
        groups (int): Same as :class:`ConvModule`. Defaults to 1.
        bias (bool or str): Same as :class:`ConvModule`. Defaults to False.
        conv_cfg (dict or ConfigDict): Same as :class:`ConvModule`.
            Defaults to ``dict(type='Conv3d')``.
        norm_cfg (dict or ConfigDict, optional): Same as :class:`ConvModule`.
            Defaults to None.
        act_cfg (dict or ConfigDict, optional): Same as :class:`ConvModule`.
            Defaults to None.
        downsample_position (str): Type of downsample position. Options are
            ``before`` and ``after``. Defaults to ``after``.
        downsample_scale (int or Tuple[int]): downsample scale for maxpooling.
            It will be used for kernel size and stride of maxpooling.
            Defaults to ``(1, 2, 2)``.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'Union[int, Tuple[int]]'=(3, 1, 1), stride: 'Union[int, Tuple[int]]'=(1, 1, 1), padding: 'Union[int, Tuple[int]]'=(1, 0, 0), groups: 'int'=1, bias: 'Union[bool, str]'=False, conv_cfg: 'ConfigType'=dict(type='Conv3d'), norm_cfg: 'OptConfigType'=None, act_cfg: 'OptConfigType'=None, downsample_position: 'str'='after', downsample_scale: 'Union[int, Tuple[int]]'=(1, 2, 2)) ->None:
        super().__init__()
        self.conv = ConvModule(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        assert downsample_position in ['before', 'after']
        self.downsample_position = downsample_position
        self.pool = nn.MaxPool3d(downsample_scale, downsample_scale, (0, 0, 0), ceil_mode=True)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Defines the computation performed at every call."""
        if self.downsample_position == 'before':
            x = self.pool(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
            x = self.pool(x)
        return x


class LevelFusion(nn.Module):
    """Level Fusion module.

    This module is used to aggregate the hierarchical features dynamic in
    visual tempos and consistent in spatial semantics. The top/bottom features
    for top-down/bottom-up flow would be combined to achieve two additional
    options, namely 'Cascade Flow' or 'Parallel Flow'. While applying a
    bottom-up flow after a top-down flow will lead to the cascade flow,
    applying them simultaneously will result in the parallel flow.

    Args:
        in_channels (Tuple[int]): Channel numbers of input features tuple.
        mid_channels (Tuple[int]): Channel numbers of middle features tuple.
        out_channels (int): Channel numbers of output features.
        downsample_scales (Tuple[int | Tuple[int]]): downsample scales for
            each :class:`DownSample` module.
            Defaults to ``((1, 1, 1), (1, 1, 1))``.
    """

    def __init__(self, in_channels: 'Tuple[int]', mid_channels: 'Tuple[int]', out_channels: 'int', downsample_scales: 'Tuple[int, Tuple[int]]'=((1, 1, 1), (1, 1, 1))) ->None:
        super().__init__()
        num_stages = len(in_channels)
        self.downsamples = nn.ModuleList()
        for i in range(num_stages):
            downsample = DownSample(in_channels[i], mid_channels[i], kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False, padding=(0, 0, 0), groups=32, norm_cfg=dict(type='BN3d', requires_grad=True), act_cfg=dict(type='ReLU', inplace=True), downsample_position='before', downsample_scale=downsample_scales[i])
            self.downsamples.append(downsample)
        self.fusion_conv = ConvModule(sum(mid_channels), out_channels, 1, stride=1, padding=0, bias=False, conv_cfg=dict(type='Conv3d'), norm_cfg=dict(type='BN3d', requires_grad=True), act_cfg=dict(type='ReLU', inplace=True))

    def forward(self, x: 'Tuple[torch.Tensor]') ->torch.Tensor:
        """Defines the computation performed at every call."""
        out = [self.downsamples[i](feature) for i, feature in enumerate(x)]
        out = torch.cat(out, 1)
        out = self.fusion_conv(out)
        return out


class SpatialModulation(nn.Module):
    """Spatial Semantic Modulation.

    This module is used to align spatial semantics of features in the
    multi-depth pyramid. For each but the top-level feature, a stack
    of convolutions with level-specific stride are applied to it, matching
    its spatial shape and receptive field with the top one.

    Args:
        in_channels (Tuple[int]): Channel numbers of input features tuple.
        out_channels (int): Channel numbers of output features tuple.
    """

    def __init__(self, in_channels: 'Tuple[int]', out_channels: 'int') ->None:
        super().__init__()
        self.spatial_modulation = nn.ModuleList()
        for channel in in_channels:
            downsample_scale = out_channels // channel
            downsample_factor = int(np.log2(downsample_scale))
            op = nn.ModuleList()
            if downsample_factor < 1:
                op = nn.Identity()
            else:
                for factor in range(downsample_factor):
                    in_factor = 2 ** factor
                    out_factor = 2 ** (factor + 1)
                    op.append(ConvModule(channel * in_factor, channel * out_factor, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False, conv_cfg=dict(type='Conv3d'), norm_cfg=dict(type='BN3d', requires_grad=True), act_cfg=dict(type='ReLU', inplace=True)))
            self.spatial_modulation.append(op)

    def forward(self, x: 'Tuple[torch.Tensor]') ->list:
        """Defines the computation performed at every call."""
        out = []
        for i, _ in enumerate(x):
            if isinstance(self.spatial_modulation[i], nn.ModuleList):
                out_ = x[i]
                for op in self.spatial_modulation[i]:
                    out_ = op(out_)
                out.append(out_)
            else:
                out.append(self.spatial_modulation[i](x[i]))
        return out


class AuxHead(nn.Module):
    """Auxiliary Head.

    This auxiliary head is appended to receive stronger supervision,
    leading to enhanced semantics.

    Args:
        in_channels (int): Channel number of input features.
        out_channels (int): Channel number of output features.
        loss_weight (float): weight of loss for the auxiliary head.
            Defaults to 0.5.
        loss_cls (dict or ConfigDict): Config for building loss.
            Defaults to ``dict(type='CrossEntropyLoss')``.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', loss_weight: 'float'=0.5, loss_cls: 'ConfigType'=dict(type='CrossEntropyLoss')) ->None:
        super().__init__()
        self.conv = ConvModule(in_channels, in_channels * 2, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False, conv_cfg=dict(type='Conv3d'), norm_cfg=dict(type='BN3d', requires_grad=True))
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.loss_weight = loss_weight
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_channels * 2, out_channels)
        self.loss_cls = MODELS.build(loss_cls)

    def init_weights(self) ->None:
        """Initiate the parameters from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.01)
            if isinstance(m, nn.Conv3d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1)

    def loss(self, x: 'torch.Tensor', data_samples: 'Optional[SampleList]') ->dict:
        """Calculate auxiliary loss."""
        x = self(x)
        labels = [x.gt_label for x in data_samples]
        labels = torch.stack(labels)
        labels = labels.squeeze()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        losses = dict()
        losses['loss_aux'] = self.loss_weight * self.loss_cls(x, labels)
        return losses

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Auxiliary head forward function."""
        x = self.conv(x)
        x = self.avg_pool(x).squeeze(-1).squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class TemporalModulation(nn.Module):
    """Temporal Rate Modulation.

    The module is used to equip TPN with a similar flexibility for temporal
    tempo modulation as in the input-level frame pyramid.

    Args:
        in_channels (int): Channel number of input features.
        out_channels (int): Channel number of output features.
        downsample_scale (int): Downsample scale for maxpooling. Defaults to 8.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', downsample_scale: 'int'=8) ->None:
        super().__init__()
        self.conv = ConvModule(in_channels, out_channels, (3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False, groups=32, conv_cfg=dict(type='Conv3d'), act_cfg=None)
        self.pool = nn.MaxPool3d((downsample_scale, 1, 1), (downsample_scale, 1, 1), (0, 0, 0), ceil_mode=True)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Defines the computation performed at every call."""
        x = self.conv(x)
        x = self.pool(x)
        return x


class TPN(nn.Module):
    """TPN neck.

    This module is proposed in `Temporal Pyramid Network for Action Recognition
    <https://arxiv.org/pdf/2004.03548.pdf>`_

    Args:
        in_channels (Tuple[int]): Channel numbers of input features tuple.
        out_channels (int): Channel number of output feature.
        spatial_modulation_cfg (dict or ConfigDict, optional): Config for
            spatial modulation layers. Required keys are ``in_channels`` and
            ``out_channels``. Defaults to None.
        temporal_modulation_cfg (dict or ConfigDict, optional): Config for
            temporal modulation layers. Defaults to None.
        upsample_cfg (dict or ConfigDict, optional): Config for upsample
            layers. The keys are same as that in :class:``nn.Upsample``.
            Defaults to None.
        downsample_cfg (dict or ConfigDict, optional): Config for downsample
            layers. Defaults to None.
        level_fusion_cfg (dict or ConfigDict, optional): Config for level
            fusion layers.
            Required keys are ``in_channels``, ``mid_channels``,
            ``out_channels``. Defaults to None.
        aux_head_cfg (dict or ConfigDict, optional): Config for aux head
            layers. Required keys are ``out_channels``. Defaults to None.
        flow_type (str): Flow type to combine the features. Options are
            ``cascade`` and ``parallel``. Defaults to ``cascade``.
    """

    def __init__(self, in_channels: 'Tuple[int]', out_channels: 'int', spatial_modulation_cfg: 'OptConfigType'=None, temporal_modulation_cfg: 'OptConfigType'=None, upsample_cfg: 'OptConfigType'=None, downsample_cfg: 'OptConfigType'=None, level_fusion_cfg: 'OptConfigType'=None, aux_head_cfg: 'OptConfigType'=None, flow_type: 'str'='cascade') ->None:
        super().__init__()
        assert isinstance(in_channels, tuple)
        assert isinstance(out_channels, int)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_tpn_stages = len(in_channels)
        assert spatial_modulation_cfg is None or isinstance(spatial_modulation_cfg, dict)
        assert temporal_modulation_cfg is None or isinstance(temporal_modulation_cfg, dict)
        assert upsample_cfg is None or isinstance(upsample_cfg, dict)
        assert downsample_cfg is None or isinstance(downsample_cfg, dict)
        assert aux_head_cfg is None or isinstance(aux_head_cfg, dict)
        assert level_fusion_cfg is None or isinstance(level_fusion_cfg, dict)
        if flow_type not in ['cascade', 'parallel']:
            raise ValueError(f"flow type in TPN should be 'cascade' or 'parallel', but got {flow_type} instead.")
        self.flow_type = flow_type
        self.temporal_modulation_ops = nn.ModuleList()
        self.upsample_ops = nn.ModuleList()
        self.downsample_ops = nn.ModuleList()
        self.level_fusion_1 = LevelFusion(**level_fusion_cfg)
        self.spatial_modulation = SpatialModulation(**spatial_modulation_cfg)
        for i in range(self.num_tpn_stages):
            if temporal_modulation_cfg is not None:
                downsample_scale = temporal_modulation_cfg['downsample_scales'][i]
                temporal_modulation = TemporalModulation(in_channels[-1], out_channels, downsample_scale)
                self.temporal_modulation_ops.append(temporal_modulation)
            if i < self.num_tpn_stages - 1:
                if upsample_cfg is not None:
                    upsample = nn.Upsample(**upsample_cfg)
                    self.upsample_ops.append(upsample)
                if downsample_cfg is not None:
                    downsample = DownSample(out_channels, out_channels, **downsample_cfg)
                    self.downsample_ops.append(downsample)
        out_dims = level_fusion_cfg['out_channels']
        self.level_fusion_2 = LevelFusion(**level_fusion_cfg)
        self.pyramid_fusion = ConvModule(out_dims * 2, 2048, 1, stride=1, padding=0, bias=False, conv_cfg=dict(type='Conv3d'), norm_cfg=dict(type='BN3d', requires_grad=True))
        if aux_head_cfg is not None:
            self.aux_head = AuxHead(self.in_channels[-2], **aux_head_cfg)
        else:
            self.aux_head = None

    def init_weights(self) ->None:
        """Default init_weights for conv(msra) and norm in ConvModule."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1)
        if self.aux_head is not None:
            self.aux_head.init_weights()

    def forward(self, x: 'Tuple[torch.Tensor]', data_samples: 'Optional[SampleList]'=None) ->tuple:
        """Defines the computation performed at every call."""
        loss_aux = dict()
        if self.aux_head is not None and data_samples is not None:
            loss_aux = self.aux_head.loss(x[-2], data_samples)
        spatial_modulation_outs = self.spatial_modulation(x)
        temporal_modulation_outs = []
        for i, temporal_modulation in enumerate(self.temporal_modulation_ops):
            temporal_modulation_outs.append(temporal_modulation(spatial_modulation_outs[i]))
        outs = [out.clone() for out in temporal_modulation_outs]
        if len(self.upsample_ops) != 0:
            for i in range(self.num_tpn_stages - 1, 0, -1):
                outs[i - 1] = outs[i - 1] + self.upsample_ops[i - 1](outs[i])
        top_down_outs = self.level_fusion_1(outs)
        if self.flow_type == 'parallel':
            outs = [out.clone() for out in temporal_modulation_outs]
        if len(self.downsample_ops) != 0:
            for i in range(self.num_tpn_stages - 1):
                outs[i + 1] = outs[i + 1] + self.downsample_ops[i](outs[i])
        botton_up_outs = self.level_fusion_2(outs)
        outs = self.pyramid_fusion(torch.cat([top_down_outs, botton_up_outs], 1))
        return outs, loss_aux


def bbox_target(pos_bboxes_list: 'List[torch.Tensor]', neg_bboxes_list: 'List[torch.Tensor]', gt_labels: 'List[torch.Tensor]', cfg: 'Union[dict, mmengine.ConfigDict]') ->tuple:
    """Generate classification targets for bboxes.

    Args:
        pos_bboxes_list (List[torch.Tensor]): Positive bboxes list.
        neg_bboxes_list (List[torch.Tensor]): Negative bboxes list.
        gt_labels (List[torch.Tensor]): Groundtruth classification label list.
        cfg (dict | mmengine.ConfigDict): RCNN config.

    Returns:
        tuple: Label and label_weight for bboxes.
    """
    labels, label_weights = [], []
    pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
    assert len(pos_bboxes_list) == len(neg_bboxes_list) == len(gt_labels)
    length = len(pos_bboxes_list)
    for i in range(length):
        pos_bboxes = pos_bboxes_list[i]
        neg_bboxes = neg_bboxes_list[i]
        gt_label = gt_labels[i]
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg
        label = F.pad(gt_label, (0, 0, 0, num_neg))
        label_weight = pos_bboxes.new_zeros(num_samples)
        label_weight[:num_pos] = pos_weight
        label_weight[-num_neg:] = 1.0
        labels.append(label)
        label_weights.append(label_weight)
    labels = torch.cat(labels, 0)
    label_weights = torch.cat(label_weights, 0)
    return labels, label_weights


class ACRNHead(nn.Module):
    """ACRN Head: Tile + 1x1 convolution + 3x3 convolution.

    This module is proposed in
    `Actor-Centric Relation Network
    <https://arxiv.org/abs/1807.10982>`_

    Args:
        in_channels (int): The input channel.
        out_channels (int): The output channel.
        stride (int): The spatial stride.
        num_convs (int): The number of 3x3 convolutions in ACRNHead.
        conv_cfg (dict): Config for norm layers. Default: dict(type='Conv').
        norm_cfg (dict):
            Config for norm layers. required keys are `type` and
            `requires_grad`. Default: dict(type='BN2d', requires_grad=True).
        act_cfg (dict): Config for activate layers.
            Default: dict(type='ReLU', inplace=True).
        kwargs (dict): Other new arguments, to be compatible with MMDet update.
    """

    def __init__(self, in_channels, out_channels, stride=1, num_convs=1, conv_cfg=dict(type='Conv3d'), norm_cfg=dict(type='BN3d', requires_grad=True), act_cfg=dict(type='ReLU', inplace=True), **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.num_convs = num_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.conv1 = ConvModule(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        assert num_convs >= 1
        self.conv2 = ConvModule(out_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        convs = []
        for _ in range(num_convs - 1):
            conv = ConvModule(out_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            convs.append(conv)
        self.convs = nn.ModuleList(convs)

    def init_weights(self, **kwargs):
        """Weight Initialization for ACRNHead."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)

    def forward(self, x, feat, rois, **kwargs):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The extracted RoI feature.
            feat (torch.Tensor): The context feature.
            rois (torch.Tensor): The regions of interest.

        Returns:
            torch.Tensor: The RoI features that have interacted with context
                feature.
        """
        x = self.max_pool(x)
        h, w = feat.shape[-2:]
        x_tile = x.repeat(1, 1, 1, h, w)
        roi_inds = rois[:, 0].type(torch.long)
        roi_gfeat = feat[roi_inds]
        new_feat = torch.cat([x_tile, roi_gfeat], dim=1)
        new_feat = self.conv1(new_feat)
        new_feat = self.conv2(new_feat)
        for conv in self.convs:
            new_feat = conv(new_feat)
        return new_feat


class NonLocalLayer(nn.Module):
    """Non-local layer used in `FBONonLocal` is a variation of the vanilla non-
    local block.

    Args:
        st_feat_channels (int): Channels of short-term features.
        lt_feat_channels (int): Channels of long-term features.
        latent_channels (int): Channels of latent features.
        use_scale (bool): Whether to scale pairwise_weight by
            `1/sqrt(latent_channels)`. Default: True.
        pre_activate (bool): Whether to use the activation function before
            upsampling. Default: False.
        conv_cfg (Dict | None): The config dict for convolution layers. If
            not specified, it will use `nn.Conv2d` for convolution layers.
            Default: None.
        norm_cfg (Dict | None): he config dict for normalization layers.
            Default: None.
        dropout_ratio (float, optional): Probability of dropout layer.
            Default: 0.2.
        zero_init_out_conv (bool): Whether to use zero initialization for
            out_conv. Default: False.
    """

    def __init__(self, st_feat_channels, lt_feat_channels, latent_channels, num_st_feat, num_lt_feat, use_scale=True, pre_activate=True, pre_activate_with_ln=True, conv_cfg=None, norm_cfg=None, dropout_ratio=0.2, zero_init_out_conv=False):
        super().__init__()
        if conv_cfg is None:
            conv_cfg = dict(type='Conv3d')
        self.st_feat_channels = st_feat_channels
        self.lt_feat_channels = lt_feat_channels
        self.latent_channels = latent_channels
        self.num_st_feat = num_st_feat
        self.num_lt_feat = num_lt_feat
        self.use_scale = use_scale
        self.pre_activate = pre_activate
        self.pre_activate_with_ln = pre_activate_with_ln
        self.dropout_ratio = dropout_ratio
        self.zero_init_out_conv = zero_init_out_conv
        self.st_feat_conv = ConvModule(self.st_feat_channels, self.latent_channels, kernel_size=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
        self.lt_feat_conv = ConvModule(self.lt_feat_channels, self.latent_channels, kernel_size=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
        self.global_conv = ConvModule(self.lt_feat_channels, self.latent_channels, kernel_size=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
        if pre_activate:
            self.ln = nn.LayerNorm([latent_channels, num_st_feat, 1, 1])
        else:
            self.ln = nn.LayerNorm([st_feat_channels, num_st_feat, 1, 1])
        self.relu = nn.ReLU()
        self.out_conv = ConvModule(self.latent_channels, self.st_feat_channels, kernel_size=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout(self.dropout_ratio)

    def init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {pretrained}')
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
            if self.zero_init_out_conv:
                constant_init(self.out_conv, 0, bias=0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, st_feat, lt_feat):
        """Defines the computation performed at every call."""
        n, c = st_feat.size(0), self.latent_channels
        num_st_feat, num_lt_feat = self.num_st_feat, self.num_lt_feat
        theta = self.st_feat_conv(st_feat)
        theta = theta.view(n, c, num_st_feat)
        phi = self.lt_feat_conv(lt_feat)
        phi = phi.view(n, c, num_lt_feat)
        g = self.global_conv(lt_feat)
        g = g.view(n, c, num_lt_feat)
        theta_phi = torch.matmul(theta.permute(0, 2, 1), phi)
        if self.use_scale:
            theta_phi /= c ** 0.5
        p = theta_phi.softmax(dim=-1)
        out = torch.matmul(g, p.permute(0, 2, 1)).view(n, c, num_st_feat, 1, 1)
        if self.pre_activate:
            if self.pre_activate_with_ln:
                out = self.ln(out)
            out = self.relu(out)
        out = self.out_conv(out)
        if not self.pre_activate:
            out = self.ln(out)
        if self.dropout_ratio > 0:
            out = self.dropout(out)
        return out


class FBONonLocal(nn.Module):
    """Non local feature bank operator.

    Args:
        st_feat_channels (int): Channels of short-term features.
        lt_feat_channels (int): Channels of long-term features.
        latent_channels (int): Channels of latent features.
        num_st_feat (int): Number of short-term roi features.
        num_lt_feat (int): Number of long-term roi features.
        num_non_local_layers (int): Number of non-local layers, which is
            at least 1. Default: 2.
        st_feat_dropout_ratio (float): Probability of dropout layer for
            short-term features. Default: 0.2.
        lt_feat_dropout_ratio (float): Probability of dropout layer for
            long-term features. Default: 0.2.
        pre_activate (bool): Whether to use the activation function before
            upsampling in non local layers. Default: True.
        zero_init_out_conv (bool): Whether to use zero initialization for
            out_conv in NonLocalLayer. Default: False.
    """

    def __init__(self, st_feat_channels, lt_feat_channels, latent_channels, num_st_feat, num_lt_feat, num_non_local_layers=2, st_feat_dropout_ratio=0.2, lt_feat_dropout_ratio=0.2, pre_activate=True, zero_init_out_conv=False, **kwargs):
        super().__init__()
        assert num_non_local_layers >= 1, 'At least one non_local_layer is needed.'
        self.st_feat_channels = st_feat_channels
        self.lt_feat_channels = lt_feat_channels
        self.latent_channels = latent_channels
        self.num_st_feat = num_st_feat
        self.num_lt_feat = num_lt_feat
        self.num_non_local_layers = num_non_local_layers
        self.st_feat_dropout_ratio = st_feat_dropout_ratio
        self.lt_feat_dropout_ratio = lt_feat_dropout_ratio
        self.pre_activate = pre_activate
        self.zero_init_out_conv = zero_init_out_conv
        self.st_feat_conv = nn.Conv3d(st_feat_channels, latent_channels, kernel_size=1)
        self.lt_feat_conv = nn.Conv3d(lt_feat_channels, latent_channels, kernel_size=1)
        if self.st_feat_dropout_ratio > 0:
            self.st_feat_dropout = nn.Dropout(self.st_feat_dropout_ratio)
        if self.lt_feat_dropout_ratio > 0:
            self.lt_feat_dropout = nn.Dropout(self.lt_feat_dropout_ratio)
        if not self.pre_activate:
            self.relu = nn.ReLU()
        self.non_local_layers = []
        for idx in range(self.num_non_local_layers):
            layer_name = f'non_local_layer_{idx + 1}'
            self.add_module(layer_name, NonLocalLayer(latent_channels, latent_channels, latent_channels, num_st_feat, num_lt_feat, pre_activate=self.pre_activate, zero_init_out_conv=self.zero_init_out_conv))
            self.non_local_layers.append(layer_name)

    def init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            kaiming_init(self.st_feat_conv)
            kaiming_init(self.lt_feat_conv)
            for layer_name in self.non_local_layers:
                non_local_layer = getattr(self, layer_name)
                non_local_layer.init_weights(pretrained=pretrained)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, st_feat, lt_feat):
        """Defines the computation performed at every call."""
        st_feat = self.st_feat_conv(st_feat)
        if self.st_feat_dropout_ratio > 0:
            st_feat = self.st_feat_dropout(st_feat)
        lt_feat = self.lt_feat_conv(lt_feat)
        if self.lt_feat_dropout_ratio > 0:
            lt_feat = self.lt_feat_dropout(lt_feat)
        for layer_name in self.non_local_layers:
            identity = st_feat
            non_local_layer = getattr(self, layer_name)
            nl_out = non_local_layer(st_feat, lt_feat)
            nl_out = identity + nl_out
            if not self.pre_activate:
                nl_out = self.relu(nl_out)
            st_feat = nl_out
        return nl_out


class FBOAvg(nn.Module):
    """Avg pool feature bank operator."""

    def __init__(self, **kwargs):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, None, None))

    def init_weights(self, pretrained=None):
        pass

    def forward(self, st_feat, lt_feat):
        out = self.avg_pool(lt_feat)
        return out


class FBOMax(nn.Module):
    """Max pool feature bank operator."""

    def __init__(self, **kwargs):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool3d((1, None, None))

    def init_weights(self, pretrained=None):
        """FBOMax has no parameters to be initialized."""
        pass

    def forward(self, st_feat, lt_feat):
        """Defines the computation performed at every call."""
        out = self.max_pool(lt_feat)
        return out


class LFB:
    """Long-Term Feature Bank (LFB). LFB is proposed in `Long-Term Feature
    Banks for Detailed Video Understanding <https://arxiv.org/abs/1812.05038>`_
    The ROI features of videos are stored in the feature bank. The feature bank
    was generated by inferring with a lfb infer config. Formally, LFB is a Dict
    whose keys are video IDs and its values are also Dicts whose keys are
    timestamps in seconds. Example of LFB:

    .. code-block:: Python
        {
            '0f39OWEqJ24': {
                901: tensor([[ 1.2760,  1.1965,  ...,  0.0061, -0.0639],
                    [-0.6320,  0.3794,  ..., -1.2768,  0.5684],
                    [ 0.2535,  1.0049,  ...,  0.4906,  1.2555],
                    [-0.5838,  0.8549,  ..., -2.1736,  0.4162]]),
                ...
                1705: tensor([[-1.0169, -1.1293,  ...,  0.6793, -2.0540],
                    [ 1.2436, -0.4555,  ...,  0.2281, -0.8219],
                    [ 0.2815, -0.0547,  ..., -0.4199,  0.5157]]),
                ...
            },
            'xmqSaQPzL1E': {
                ...
            },
            ...
        }
    Args:
        lfb_prefix_path (str): The storage path of lfb.
        max_num_sampled_feat (int): The max number of sampled features.
            Default: 5.
        window_size (int): Window size of sampling long term feature.
            Default: 60.
        lfb_channels (int): Number of the channels of the features stored
            in LFB. Default: 2048.
        dataset_modes (tuple[str] | str): Load LFB of datasets with different
            modes, such as training, validation, testing datasets. If you don't
            do cross validation during training, just load the training dataset
            i.e. setting `dataset_modes = ('train')`.
            Default: ('train', 'val').
        device (str): Where to load lfb. Choices are 'gpu', 'cpu' and 'lmdb'.
            A 1.65GB half-precision ava lfb (including training and validation)
            occupies about 2GB GPU memory. Default: 'gpu'.
        lmdb_map_size (int): Map size of lmdb. Default: 4e9.
        construct_lmdb (bool): Whether to construct lmdb. If you have
            constructed lmdb of lfb, you can set to False to skip the
            construction. Default: True.
    """

    def __init__(self, lfb_prefix_path, max_num_sampled_feat=5, window_size=60, lfb_channels=2048, dataset_modes=('train', 'val'), device='gpu', lmdb_map_size=4000000000.0, construct_lmdb=True):
        if not osp.exists(lfb_prefix_path):
            raise ValueError(f'lfb prefix path {lfb_prefix_path} does not exist!')
        self.lfb_prefix_path = lfb_prefix_path
        self.max_num_sampled_feat = max_num_sampled_feat
        self.window_size = window_size
        self.lfb_channels = lfb_channels
        if not isinstance(dataset_modes, tuple):
            assert isinstance(dataset_modes, str)
            dataset_modes = dataset_modes,
        self.dataset_modes = dataset_modes
        self.device = device
        rank, world_size = get_dist_info()
        if self.device == 'gpu':
            if 'LOCAL_RANK' in os.environ:
                local_rank = int(os.environ['LOCAL_RANK'])
            else:
                gpus_per_node = torch.cuda.device_count()
                local_rank = rank % gpus_per_node
            self.load_lfb(f'cuda:{local_rank}')
        elif self.device == 'cpu':
            if world_size > 1:
                warnings.warn("If distributed training is used with multi-GPUs, lfb will be loaded multiple times on RAM. In this case, 'lmdb' is recommended.", UserWarning)
            self.load_lfb('cpu')
        elif self.device == 'lmdb':
            assert lmdb_imported, 'Please install `lmdb` to load lfb on lmdb!'
            self.lmdb_map_size = lmdb_map_size
            self.construct_lmdb = construct_lmdb
            self.lfb_lmdb_path = osp.normpath(osp.join(self.lfb_prefix_path, 'lmdb'))
            if rank == 0 and self.construct_lmdb:
                None
                self.load_lfb_on_lmdb()
            if world_size > 1:
                dist.barrier()
            self.lmdb_env = lmdb.open(self.lfb_lmdb_path, readonly=True)
        else:
            raise ValueError("Device must be 'gpu', 'cpu' or 'lmdb', ", f'but get {self.device}.')

    def load_lfb(self, map_location):
        self.lfb = {}
        for dataset_mode in self.dataset_modes:
            lfb_path = osp.normpath(osp.join(self.lfb_prefix_path, f'lfb_{dataset_mode}.pkl'))
            None
            self.lfb.update(torch.load(lfb_path, map_location=map_location))
        for video_id in self.lfb:
            video_features = self.lfb[video_id]
            for sec in video_features:
                if isinstance(video_features[sec], (list, tuple)):
                    video_features[sec] = torch.stack(video_features[sec])
            self.lfb[video_id] = video_features
        None

    def load_lfb_on_lmdb(self):
        lfb = {}
        for dataset_mode in self.dataset_modes:
            lfb_path = osp.normpath(osp.join(self.lfb_prefix_path, f'lfb_{dataset_mode}.pkl'))
            lfb.update(torch.load(lfb_path, map_location='cpu'))
        lmdb_env = lmdb.open(self.lfb_lmdb_path, map_size=self.lmdb_map_size)
        for key, value in lfb.items():
            txn = lmdb_env.begin(write=True)
            buff = io.BytesIO()
            torch.save(value, buff)
            buff.seek(0)
            txn.put(key.encode(), buff.read())
            txn.commit()
            buff.close()
        None

    def sample_long_term_features(self, video_id, timestamp):
        if self.device == 'lmdb':
            with self.lmdb_env.begin(write=False) as txn:
                buf = txn.get(video_id.encode())
                video_features = torch.load(io.BytesIO(buf))
        else:
            video_features = self.lfb[video_id]
        window_size, K = self.window_size, self.max_num_sampled_feat
        start = timestamp - window_size // 2
        lt_feats = torch.zeros(window_size, K, self.lfb_channels)
        for idx, sec in enumerate(range(start, start + window_size)):
            if sec in video_features:
                feat = video_features[sec]
                num_feat = feat.shape[0]
                random_lfb_indices = torch.randperm(num_feat)[:K]
                lt_feats[idx, :num_feat] = feat[random_lfb_indices]
        return lt_feats.reshape(-1, self.lfb_channels)

    def __getitem__(self, img_key):
        """Sample long term features like `lfb['0f39OWEqJ24,0902']` where `lfb`
        is a instance of class LFB."""
        video_id, timestamp = img_key.split(',')
        return self.sample_long_term_features(video_id, int(timestamp))

    def __len__(self):
        """The number of videos whose ROI features are stored in LFB."""
        return len(self.lfb)


class FBOHead(nn.Module):
    """Feature Bank Operator Head.

    Add feature bank operator for the spatiotemporal detection model to fuse
    short-term features and long-term features.
    Args:
        lfb_cfg (Dict): The config dict for LFB which is used to sample
            long-term features.
        fbo_cfg (Dict): The config dict for feature bank operator (FBO). The
            type of fbo is also in the config dict and supported fbo type is
            `fbo_dict`.
        temporal_pool_type (str): The temporal pool type. Choices are 'avg' or
            'max'. Default: 'avg'.
        spatial_pool_type (str): The spatial pool type. Choices are 'avg' or
            'max'. Default: 'max'.
    """
    fbo_dict = {'non_local': FBONonLocal, 'avg': FBOAvg, 'max': FBOMax}

    def __init__(self, lfb_cfg, fbo_cfg, temporal_pool_type='avg', spatial_pool_type='max'):
        super().__init__()
        fbo_type = fbo_cfg.pop('type', 'non_local')
        assert fbo_type in FBOHead.fbo_dict
        assert temporal_pool_type in ['max', 'avg']
        assert spatial_pool_type in ['max', 'avg']
        self.lfb_cfg = copy.deepcopy(lfb_cfg)
        self.fbo_cfg = copy.deepcopy(fbo_cfg)
        self.lfb = LFB(**self.lfb_cfg)
        self.fbo = self.fbo_dict[fbo_type](**self.fbo_cfg)
        if temporal_pool_type == 'avg':
            self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        else:
            self.temporal_pool = nn.AdaptiveMaxPool3d((1, None, None))
        if spatial_pool_type == 'avg':
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        else:
            self.spatial_pool = nn.AdaptiveMaxPool3d((None, 1, 1))

    def init_weights(self, pretrained=None):
        """Initialize the weights in the module.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        self.fbo.init_weights(pretrained=pretrained)

    def sample_lfb(self, rois, img_metas):
        """Sample long-term features for each ROI feature."""
        inds = rois[:, 0].type(torch.int64)
        lt_feat_list = []
        for ind in inds:
            lt_feat_list.append(self.lfb[img_metas[ind]['img_key']])
        lt_feat = torch.stack(lt_feat_list, dim=0)
        lt_feat = lt_feat.permute(0, 2, 1).contiguous()
        return lt_feat.unsqueeze(-1).unsqueeze(-1)

    def forward(self, x, rois, img_metas, **kwargs):
        """Defines the computation performed at every call."""
        st_feat = self.temporal_pool(x)
        st_feat = self.spatial_pool(st_feat)
        identity = st_feat
        lt_feat = self.sample_lfb(rois, img_metas)
        fbo_feat = self.fbo(st_feat, lt_feat)
        out = torch.cat([identity, fbo_feat], dim=1)
        return out


class LFBInferHead(nn.Module):
    """Long-Term Feature Bank Infer Head.

    This head is used to derive and save the LFB without affecting the input.
    Args:
        lfb_prefix_path (str): The prefix path to store the lfb.
        dataset_mode (str, optional): Which dataset to be inferred. Choices are
            'train', 'val' or 'test'. Default: 'train'.
        use_half_precision (bool, optional): Whether to store the
            half-precision roi features. Default: True.
        temporal_pool_type (str): The temporal pool type. Choices are 'avg' or
            'max'. Default: 'avg'.
        spatial_pool_type (str): The spatial pool type. Choices are 'avg' or
            'max'. Default: 'max'.
    """

    def __init__(self, lfb_prefix_path, dataset_mode='train', use_half_precision=True, temporal_pool_type='avg', spatial_pool_type='max'):
        super().__init__()
        rank, _ = mmengine.dist.get_dist_info()
        if rank == 0:
            if not osp.exists(lfb_prefix_path):
                None
                mmengine.mkdir_or_exist(lfb_prefix_path)
            None
        assert temporal_pool_type in ['max', 'avg']
        assert spatial_pool_type in ['max', 'avg']
        self.lfb_prefix_path = lfb_prefix_path
        self.dataset_mode = dataset_mode
        self.use_half_precision = use_half_precision
        if temporal_pool_type == 'avg':
            self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        else:
            self.temporal_pool = nn.AdaptiveMaxPool3d((1, None, None))
        if spatial_pool_type == 'avg':
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        else:
            self.spatial_pool = nn.AdaptiveMaxPool3d((None, 1, 1))
        self.all_features = []
        self.all_metadata = []

    def init_weights(self, pretrained=None):
        """LFBInferHead has no parameters to be initialized."""
        pass

    def forward(self, x, rois, img_metas, **kwargs):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The extracted RoI feature.
            rois (torch.Tensor): The regions of interest.
            img_metas (List[dict]): The meta information of the data.

        Returns:
            torch.Tensor: The RoI features that have interacted with context
        """
        features = self.temporal_pool(x)
        features = self.spatial_pool(features)
        if self.use_half_precision:
            features = features.half()
        inds = rois[:, 0].type(torch.int64)
        for ind in inds:
            self.all_metadata.append(img_metas[ind]['img_key'])
        self.all_features += list(features)
        return x

    def __del__(self):
        assert len(self.all_features) == len(self.all_metadata), 'features and metadata are not equal in length!'
        rank, world_size = mmengine.dist.get_dist_info()
        if world_size > 1:
            dist.barrier()
        _lfb = {}
        for feature, metadata in zip(self.all_features, self.all_metadata):
            video_id, timestamp = metadata.split(',')
            timestamp = int(timestamp)
            if video_id not in _lfb:
                _lfb[video_id] = {}
            if timestamp not in _lfb[video_id]:
                _lfb[video_id][timestamp] = []
            _lfb[video_id][timestamp].append(torch.squeeze(feature))
        _lfb_file_path = osp.normpath(osp.join(self.lfb_prefix_path, f'_lfb_{self.dataset_mode}_{rank}.pkl'))
        torch.save(_lfb, _lfb_file_path)
        None
        if world_size > 1:
            dist.barrier()
        if rank > 0:
            return
        None
        lfb = {}
        for rank_id in range(world_size):
            _lfb_file_path = osp.normpath(osp.join(self.lfb_prefix_path, f'_lfb_{self.dataset_mode}_{rank_id}.pkl'))
            _lfb = torch.load(_lfb_file_path)
            for video_id in _lfb:
                if video_id not in lfb:
                    lfb[video_id] = _lfb[video_id]
                else:
                    lfb[video_id].update(_lfb[video_id])
            osp.os.remove(_lfb_file_path)
        lfb_file_path = osp.normpath(osp.join(self.lfb_prefix_path, f'lfb_{self.dataset_mode}.pkl'))
        torch.save(lfb, lfb_file_path)
        None


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Defines the computation performed at every call."""
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Perform quick gelu."""
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """"ResidualAttentionBlock.

    Args:
        d_model (int): The dimension of the model.
        n_head (int): The number of heads.
        attn_mask (torch.Tensor, optional): The attention mask.
            Defaults to None.
    """

    def __init__(self, d_model: 'int', n_head: 'int', attn_mask: 'Optional[torch.Tensor]'=None) ->None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)), ('gelu', QuickGELU()), ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: 'torch.Tensor') ->torch.Tensor:
        """Perform attention."""
        self.attn_mask = self.attn_mask if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Defines the computation performed at every call."""
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """"ResidualAttentionBlock.

    Args:
        width (int): The width of transformer.
        heads (int): The number of heads of transformer.
        layers (int): The number of layers of transformer.
        attn_mask (torch.Tensor, optional): The attention mask.
            Defaults to None.
    """

    def __init__(self, width: 'int', layers: 'int', heads: 'int', attn_mask: 'Optional[torch.Tensor]'=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Defines the computation performed at every call."""
        return self.resblocks(x)


class AdaptivePadding(nn.Module):
    """Applies padding adaptively to the input.

    This module can make input get fully covered by filter
    you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad
    zero around input. The "corner"  mode would pad zero
    to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel. Default: 1.
        stride (int | tuple): Stride of the filter. Default: 1.
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1.
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".

    Example:
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):
        super().__init__()
        assert padding in ('same', 'corner')
        kernel_size = to_3tuple(kernel_size)
        stride = to_3tuple(stride)
        dilation = to_3tuple(dilation)
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        """Calculate the padding size of input.

        Args:
            input_shape (:obj:`torch.Size`): arrange as (H, W).

        Returns:
            Tuple[int]: The padding size along the
            original H and W directions
        """
        input_t, input_h, input_w = input_shape
        kernel_d, kernel_h, kernel_w = self.kernel_size
        stride_d, stride_h, stride_w = self.stride
        output_d = math.ceil(input_t / stride_d)
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_d = max((output_d - 1) * stride_d + (kernel_d - 1) * self.dilation[0] + 1 - input_t, 0)
        pad_h = max((output_h - 1) * stride_h + (kernel_h - 1) * self.dilation[1] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + (kernel_w - 1) * self.dilation[2] + 1 - input_w, 0)
        return pad_d, pad_h, pad_w

    def forward(self, x):
        """Add padding to `x`

        Args:
            x (Tensor): Input tensor has shape (B, C, H, W).

        Returns:
            Tensor: The tensor with adaptive padding
        """
        pad_d, pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h, 0, pad_d])
            elif self.padding == 'same':
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_d // 2, pad_d - pad_d // 2])
        return x


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
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


def get_sinusoid_encoding_table(n_position, d_hid, cur_frame=-1, pre_n_position=1568):
    """Sinusoid position encoding table."""

    def get_position_angle_vec(position):
        return [(position / np.power(10000, 2 * (hid_j // 2) / d_hid)) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)
    None
    None
    if n_position // cur_frame * 8 != pre_n_position and cur_frame != -1:
        T = 8
        P = 14
        C = d_hid
        new_P = int((n_position // cur_frame) ** 0.5)
        None
        None
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
        sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=(new_P, new_P), mode='bicubic', align_corners=False)
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
        sinusoid_table = sinusoid_table.flatten(1, 3)
    if cur_frame != -1 and cur_frame != 8:
        None
        None
        T = 8
        new_T = cur_frame
        P = int((n_position // cur_frame) ** 0.5)
        C = d_hid
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)
        sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
        sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3)
        sinusoid_table = sinusoid_table.flatten(1, 3)
    if n_position == pre_n_position:
        return sinusoid_table
    else:
        None
        return nn.Parameter(sinusoid_table, requires_grad=True)


class UMTViT(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-06), init_values=0.0, use_learnable_pos_emb=False, all_frames=16, tubelet_size=1, use_checkpoint=False, checkpoint_num=0, use_mean_pooling=True):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        None
        None
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            if patch_size == 14:
                pre_n_position = 2048
            else:
                pre_n_position = 1568
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim, all_frames // tubelet_size, pre_n_position=pre_n_position)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values) for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, _, _ = x.size()
        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).clone().detach()
        x = self.pos_drop(x)
        for idx, blk in enumerate(self.blocks):
            if self.use_checkpoint and idx < self.checkpoint_num:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.norm(x)
        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))
        else:
            return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        return x


class SubModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 2, kernel_size=1, groups=2)
        self.gn = nn.GroupNorm(2, 2)
        self.fc = nn.Linear(2, 2)
        self.param1 = nn.Parameter(torch.ones(1))


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(torch.ones(1))
        self.conv1 = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(4, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)
        self.sub = SubModel()
        self.fc = nn.Linear(2, 1)


class AvgPool2d(nn.Module):

    def forward(self, x):
        return x.mean(dim=(-1, -2), keepdims=True)


class MaxPool2d(nn.Module):

    def forward(self, x):
        x = x.max(dim=-1, keepdim=True)[0]
        x = x.max(dim=-2, keepdim=True)[0]
        return x


class AvgPool3d(nn.Module):

    def forward(self, x):
        return x.mean(dim=(-1, -2, -3), keepdims=True)


class MaxPool3d(nn.Module):

    def forward(self, x):
        x = x.max(dim=-1, keepdim=True)[0]
        x = x.max(dim=-2, keepdim=True)[0]
        x = x.max(dim=-3, keepdim=True)[0]
        return x


class GCNNet(nn.Module):

    def __init__(self, base_model):
        super(GCNNet, self).__init__()
        self.backbone = base_model.backbone
        self.head = base_model.cls_head
        if hasattr(self.head, 'pool'):
            pool = self.head.pool
            if isinstance(pool, nn.AdaptiveAvgPool3d):
                assert pool.output_size == 1
                self.head.pool = AvgPool3d()
            elif isinstance(pool, nn.AdaptiveMaxPool3d):
                assert pool.output_size == 1
                self.head.pool = MaxPool3d()

    def forward(self, input_tensor):
        feat = self.backbone(input_tensor)
        cls_score = self.head(feat)
        return cls_score


class SpatialMaxPool3d(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.max(dim=-1, keepdim=True)[0]
        return x.max(dim=-2, keepdim=True)[0]


class SpatialAvgPool(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(dim=(-1, -2), keepdims=True)


class TemporalMaxPool3d(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.max(dim=-3, keepdim=True)[0]


class TemporalAvgPool3d(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(dim=-3, keepdim=True)


class GlobalPool2d(nn.Module):

    def __init__(self, pool_size, output_size, later_max=True):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size)
        self.max = later_max
        self.output_size = output_size

    def forward(self, x):
        x = self.pool(x)
        if self.max:
            x = x.max(dim=-1, keepdim=True)[0]
            x = x.max(dim=-2, keepdim=True)[0]
        else:
            x = x.mean(dim=(-1, -2), keepdims=True)
        x = x.expand(-1, -1, self.output_size, self.output_size)
        return x


class STDet(nn.Module):

    def __init__(self, base_model, input_tensor):
        super(STDet, self).__init__()
        self.backbone = base_model.backbone
        self.bbox_roi_extractor = base_model.roi_head.bbox_roi_extractor
        self.bbox_head = base_model.roi_head.bbox_head
        output_size = self.bbox_roi_extractor.global_pool.output_size
        pool_size = min(input_tensor.shape[-2:]) // 16 // output_size
        if isinstance(self.bbox_head.temporal_pool, nn.AdaptiveAvgPool3d):
            self.bbox_head.temporal_pool = TemporalAvgPool3d()
        else:
            self.bbox_head.temporal_pool = TemporalMaxPool3d()
        if isinstance(self.bbox_head.spatial_pool, nn.AdaptiveAvgPool3d):
            self.bbox_head.spatial_pool = SpatialAvgPool()
            self.bbox_roi_extractor.global_pool = GlobalPool2d(pool_size, output_size, later_max=False)
        else:
            self.bbox_head.spatial_pool = SpatialMaxPool3d()
            self.bbox_roi_extractor.global_pool = GlobalPool2d(pool_size, output_size, later_max=True)

    def forward(self, input_tensor, rois):
        feat = self.backbone(input_tensor)
        bbox_feats, _ = self.bbox_roi_extractor(feat, rois)
        cls_score = self.bbox_head(bbox_feats)
        return cls_score


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AvgConsensus,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AvgPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AvgPool3d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BCELossWithLogits,
     lambda: ([], {}),
     lambda: ([], {'cls_score': torch.rand([4, 4]), 'label': torch.rand([4, 4])})),
    (BMNLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (BeitPooler3D,
     lambda: ([], {'config': SimpleNamespace(add_k_prompts=4, use_mean_pooling=4, hidden_size=4, layer_norm_eps=1)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BertAttention,
     lambda: ([], {'config': SimpleNamespace(hidden_size=4, num_attention_heads=4, attention_probs_dropout_prob=0.5, position_embedding_type=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (BertIntermediate,
     lambda: ([], {'config': SimpleNamespace(hidden_size=4, intermediate_size=4, hidden_act=torch.nn.ReLU())}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BertOnlyNSPHead,
     lambda: ([], {'config': SimpleNamespace(hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BertOutput,
     lambda: ([], {'config': SimpleNamespace(intermediate_size=4, hidden_size=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (BertPooler,
     lambda: ([], {'config': SimpleNamespace(hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BertSelfAttention,
     lambda: ([], {'config': SimpleNamespace(hidden_size=4, num_attention_heads=4, encoder_width=4, attention_probs_dropout_prob=0.5, position_embedding_type=4), 'is_cross_attention': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (BertSelfOutput,
     lambda: ([], {'config': SimpleNamespace(hidden_size=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (BinaryLogisticRegressionLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (CombineNet,
     lambda: ([], {'net1': torch.nn.ReLU(), 'net2': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([], {'cls_score': torch.rand([4, 4]), 'label': torch.rand([4, 4])})),
    (FBOAvg,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (FBOMax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (GlobalPool2d,
     lambda: ([], {'pool_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HVULoss,
     lambda: ([], {}),
     lambda: ([], {'cls_score': torch.rand([4, 4]), 'label': torch.rand([4, 4]), 'mask': 4, 'category_mask': 4})),
    (LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MaxPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MaxPool3d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (OffsetNet,
     lambda: ([], {'in_channels': 4, 'groups': 1, 'num_segments': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (QuickGELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RelationModule,
     lambda: ([], {'hidden_dim': 4, 'num_segments': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (RelationModuleMultiScale,
     lambda: ([], {'hidden_dim': 4, 'num_segments': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (ResidualAttentionBlock,
     lambda: ([], {'d_model': 4, 'n_head': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (SEModule,
     lambda: ([], {'channels': 4, 'reduction': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Scale,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SpatialAvgPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SpatialMaxPool3d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SubBatchNorm3D,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (TAM,
     lambda: ([], {'in_channels': 4, 'num_segments': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TemporalAvgPool3d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TemporalMaxPool3d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Transformer,
     lambda: ([], {'width': 4, 'layers': 1, 'heads': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (WeightNet,
     lambda: ([], {'in_channels': 4, 'groups': 1}),
     lambda: ([torch.rand([4, 4, 4])], {})),
]

