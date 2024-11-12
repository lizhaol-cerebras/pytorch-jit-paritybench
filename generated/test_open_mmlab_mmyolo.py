
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


from collections import OrderedDict


import torch


import numpy as np


from abc import ABCMeta


from copy import deepcopy


from typing import List


from typing import Optional


from typing import Sequence


from typing import Tuple


from typing import Type


from typing import TypeVar


from typing import Union


from torch import Tensor


import collections


import copy


from abc import abstractmethod


from numpy import random


import math


from functools import partial


from typing import Callable


from typing import Dict


import torch.nn as nn


from torch.nn.modules.batchnorm import _BatchNorm


import random


import torch.nn.functional as F


import warnings


from collections import defaultdict


from typing import Any


import torchvision


from enum import Enum


from collections import namedtuple


from torch.utils.cpp_extension import BuildExtension


from torch import nn


from torch.utils.data import Dataset


from torch.nn.modules import GroupNorm


import itertools


from scipy.optimize import differential_evolution


import re


import matplotlib.pyplot as plt


from itertools import repeat


import logging


class RepVGGBlock(nn.Module):
    """RepVGGBlock is a basic rep-style block, including training and deploy
    status This code is based on
    https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple): Stride of the convolution. Default: 1
        padding (int, tuple): Padding added to all four sides of
            the input. Default: 1
        dilation (int or tuple): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        padding_mode (string, optional): Default: 'zeros'
        use_se (bool): Whether to use se. Default: False
        use_alpha (bool): Whether to use `alpha` parameter at 1x1 conv.
            In PPYOLOE+ model backbone, `use_alpha` will be set to True.
            Default: False.
        use_bn_first (bool): Whether to use bn layer before conv.
            In YOLOv6 and YOLOv7, this will be set to True.
            In PPYOLOE, this will be set to False.
            Default: True.
        deploy (bool): Whether in deploy mode. Default: False
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'Union[int, Tuple[int]]'=3, stride: 'Union[int, Tuple[int]]'=1, padding: 'Union[int, Tuple[int]]'=1, dilation: 'Union[int, Tuple[int]]'=1, groups: 'Optional[int]'=1, padding_mode: 'Optional[str]'='zeros', norm_cfg: 'ConfigType'=dict(type='BN', momentum=0.03, eps=0.001), act_cfg: 'ConfigType'=dict(type='ReLU', inplace=True), use_se: 'bool'=False, use_alpha: 'bool'=False, use_bn_first=True, deploy: 'bool'=False):
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert kernel_size == 3
        assert padding == 1
        padding_11 = padding - kernel_size // 2
        self.nonlinearity = MODELS.build(act_cfg)
        if use_se:
            raise NotImplementedError('se block not supported yet')
        else:
            self.se = nn.Identity()
        if use_alpha:
            alpha = torch.ones([1], dtype=torch.float32, requires_grad=True)
            self.alpha = nn.Parameter(alpha, requires_grad=True)
        else:
            self.alpha = None
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            if use_bn_first and out_channels == in_channels and stride == 1:
                self.rbr_identity = build_norm_layer(norm_cfg, num_features=in_channels)[1]
            else:
                self.rbr_identity = None
            self.rbr_dense = ConvModule(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False, norm_cfg=norm_cfg, act_cfg=None)
            self.rbr_1x1 = ConvModule(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups, bias=False, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, inputs: 'Tensor') ->Tensor:
        """Forward process.
        Args:
            inputs (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        if self.alpha:
            return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.alpha * self.rbr_1x1(inputs) + id_out))
        else:
            return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        """Derives the equivalent kernel and bias in a differentiable way.

        Returns:
            tuple: Equivalent kernel and bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        if self.alpha:
            return kernel3x3 + self.alpha * self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + self.alpha * bias1x1 + biasid
        else:
            return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pad 1x1 tensor to 3x3.
        Args:
            kernel1x1 (Tensor): The input 1x1 kernel need to be padded.

        Returns:
            Tensor: 3x3 kernel after padded.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: 'nn.Module') ->Tuple[np.ndarray, Tensor]:
        """Derives the equivalent kernel and bias of a specific branch layer.

        Args:
            branch (nn.Module): The layer that needs to be equivalently
                transformed, which can be nn.Sequential or nn.Batchnorm2d

        Returns:
            tuple: Equivalent kernel and bias
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, ConvModule):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, (nn.SyncBatchNorm, nn.BatchNorm2d))
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        """Switch to deploy mode."""
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels, kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride, padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class BottleRep(nn.Module):
    """Bottle Rep Block.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        block_cfg (dict): Config dict for the block used to build each
            layer. Defaults to dict(type='RepVGGBlock').
        adaptive_weight (bool): Add adaptive_weight when forward calculate.
            Defaults False.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', block_cfg: 'ConfigType'=dict(type='RepVGGBlock'), adaptive_weight: 'bool'=False):
        super().__init__()
        conv1_cfg = block_cfg.copy()
        conv2_cfg = block_cfg.copy()
        conv1_cfg.update(dict(in_channels=in_channels, out_channels=out_channels))
        conv2_cfg.update(dict(in_channels=out_channels, out_channels=out_channels))
        self.conv1 = MODELS.build(conv1_cfg)
        self.conv2 = MODELS.build(conv2_cfg)
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if adaptive_weight:
            self.alpha = nn.Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x: 'Tensor') ->Tensor:
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs


class RepStageBlock(nn.Module):
    """RepStageBlock is a stage block with rep-style basic block.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        num_blocks (int, tuple[int]): Number of blocks.  Defaults to 1.
        bottle_block (nn.Module): Basic unit of RepStage.
            Defaults to RepVGGBlock.
        block_cfg (ConfigType): Config of RepStage.
            Defaults to 'RepVGGBlock'.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', num_blocks: 'int'=1, bottle_block: 'nn.Module'=RepVGGBlock, block_cfg: 'ConfigType'=dict(type='RepVGGBlock')):
        super().__init__()
        block_cfg = block_cfg.copy()
        block_cfg.update(dict(in_channels=in_channels, out_channels=out_channels))
        self.conv1 = MODELS.build(block_cfg)
        block_cfg.update(dict(in_channels=out_channels, out_channels=out_channels))
        self.block = None
        if num_blocks > 1:
            self.block = nn.Sequential(*(MODELS.build(block_cfg) for _ in range(num_blocks - 1)))
        if bottle_block == BottleRep:
            self.conv1 = BottleRep(in_channels, out_channels, block_cfg=block_cfg, adaptive_weight=True)
            num_blocks = num_blocks // 2
            self.block = None
            if num_blocks > 1:
                self.block = nn.Sequential(*(BottleRep(out_channels, out_channels, block_cfg=block_cfg, adaptive_weight=True) for _ in range(num_blocks - 1)))

    def forward(self, x: 'Tensor') ->Tensor:
        """Forward process.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


class BepC3StageBlock(nn.Module):
    """Beer-mug RepC3 Block.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        num_blocks (int): Number of blocks. Defaults to 1
        hidden_ratio (float): Hidden channel expansion.
            Default: 0.5
        concat_all_layer (bool): Concat all layer when forward calculate.
            Default: True
        block_cfg (dict): Config dict for the block used to build each
            layer. Defaults to dict(type='RepVGGBlock').
        norm_cfg (ConfigType): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (ConfigType): Config dict for activation layer.
            Defaults to dict(type='ReLU', inplace=True).
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', num_blocks: 'int'=1, hidden_ratio: 'float'=0.5, concat_all_layer: 'bool'=True, block_cfg: 'ConfigType'=dict(type='RepVGGBlock'), norm_cfg: 'ConfigType'=dict(type='BN', momentum=0.03, eps=0.001), act_cfg: 'ConfigType'=dict(type='ReLU', inplace=True)):
        super().__init__()
        hidden_channels = int(out_channels * hidden_ratio)
        self.conv1 = ConvModule(in_channels, hidden_channels, kernel_size=1, stride=1, groups=1, bias=False, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(in_channels, hidden_channels, kernel_size=1, stride=1, groups=1, bias=False, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv3 = ConvModule(2 * hidden_channels, out_channels, kernel_size=1, stride=1, groups=1, bias=False, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.block = RepStageBlock(in_channels=hidden_channels, out_channels=hidden_channels, num_blocks=num_blocks, block_cfg=block_cfg, bottle_block=BottleRep)
        self.concat_all_layer = concat_all_layer
        if not concat_all_layer:
            self.conv3 = ConvModule(hidden_channels, out_channels, kernel_size=1, stride=1, groups=1, bias=False, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        if self.concat_all_layer is True:
            return self.conv3(torch.cat((self.block(self.conv1(x)), self.conv2(x)), dim=1))
        else:
            return self.conv3(self.block(self.conv1(x)))


class ConvWrapper(nn.Module):
    """Wrapper for normal Conv with SiLU activation.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple): Stride of the convolution. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): Conv bias. Default: True.
        norm_cfg (ConfigType): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (ConfigType): Config dict for activation layer.
            Defaults to dict(type='ReLU', inplace=True).
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'int'=3, stride: 'int'=1, groups: 'int'=1, bias: 'bool'=True, norm_cfg: 'ConfigType'=None, act_cfg: 'ConfigType'=dict(type='SiLU')):
        super().__init__()
        self.block = ConvModule(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, groups=groups, bias=bias, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x: 'Tensor') ->Tensor:
        return self.block(x)


class EffectiveSELayer(nn.Module):
    """Effective Squeeze-Excitation.

    From `CenterMask : Real-Time Anchor-Free Instance Segmentation`
    arxiv (https://arxiv.org/abs/1911.06667)
    This code referenced to
    https://github.com/youngwanLEE/CenterMask/blob/72147e8aae673fcaf4103ee90a6a6b73863e7fa1/maskrcnn_benchmark/modeling/backbone/vovnet.py#L108-L121  # noqa

    Args:
        channels (int): The input and output channels of this Module.
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='HSigmoid').
    """

    def __init__(self, channels: 'int', act_cfg: 'ConfigType'=dict(type='HSigmoid')):
        super().__init__()
        assert isinstance(act_cfg, dict)
        self.fc = ConvModule(channels, channels, 1, act_cfg=None)
        act_cfg_ = act_cfg.copy()
        self.activate = MODELS.build(act_cfg_)

    def forward(self, x: 'Tensor') ->Tensor:
        """Forward process
         Args:
             x (Tensor): The input tensor.
         """
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.activate(x_se)


class PPYOLOESELayer(nn.Module):
    """Squeeze-and-Excitation Attention Module for PPYOLOE.
        There are some differences between the current implementation and
        SELayer in mmdet:
            1. For fast speed and avoiding double inference in ppyoloe,
               use `F.adaptive_avg_pool2d` before PPYOLOESELayer.
            2. Special ways to init weights.
            3. Different convolution order.

    Args:
        feat_channels (int): The input (and output) channels of the SE layer.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.1, eps=1e-5).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
    """

    def __init__(self, feat_channels: 'int', norm_cfg: 'ConfigType'=dict(type='BN', momentum=0.1, eps=1e-05), act_cfg: 'ConfigType'=dict(type='SiLU', inplace=True)):
        super().__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
        self.sig = nn.Sigmoid()
        self.conv = ConvModule(feat_channels, feat_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self._init_weights()

    def _init_weights(self):
        """Init weights."""
        nn.init.normal_(self.fc.weight, mean=0, std=0.001)

    def forward(self, feat: 'Tensor', avg_feat: 'Tensor') ->Tensor:
        """Forward process
         Args:
             feat (Tensor): The input tensor.
             avg_feat (Tensor): Average pooling feature tensor.
         """
        weight = self.sig(self.fc(avg_feat))
        return self.conv(feat * weight)


class ImplicitA(nn.Module):
    """Implicit add layer in YOLOv7.

    Args:
        in_channels (int): The input channels of this Module.
        mean (float): Mean value of implicit module. Defaults to 0.
        std (float): Std value of implicit module. Defaults to 0.02
    """

    def __init__(self, in_channels: 'int', mean: 'float'=0.0, std: 'float'=0.02):
        super().__init__()
        self.implicit = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        nn.init.normal_(self.implicit, mean=mean, std=std)

    def forward(self, x):
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        return self.implicit + x


class ImplicitM(nn.Module):
    """Implicit multiplier layer in YOLOv7.

    Args:
        in_channels (int): The input channels of this Module.
        mean (float): Mean value of implicit module. Defaults to 1.
        std (float): Std value of implicit module. Defaults to 0.02.
    """

    def __init__(self, in_channels: 'int', mean: 'float'=1.0, std: 'float'=0.02):
        super().__init__()
        self.implicit = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        nn.init.normal_(self.implicit, mean=mean, std=std)

    def forward(self, x):
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        return self.implicit * x


class PPYOLOEBasicBlock(nn.Module):
    """PPYOLOE Backbone BasicBlock.

    Args:
         in_channels (int): The input channels of this Module.
         out_channels (int): The output channels of this Module.
         norm_cfg (dict): Config dict for normalization layer.
             Defaults to dict(type='BN', momentum=0.1, eps=1e-5).
         act_cfg (dict): Config dict for activation layer.
             Defaults to dict(type='SiLU', inplace=True).
         shortcut (bool): Whether to add inputs and outputs together
         at the end of this layer. Defaults to True.
         use_alpha (bool): Whether to use `alpha` parameter at 1x1 conv.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', norm_cfg: 'ConfigType'=dict(type='BN', momentum=0.1, eps=1e-05), act_cfg: 'ConfigType'=dict(type='SiLU', inplace=True), shortcut: 'bool'=True, use_alpha: 'bool'=False):
        super().__init__()
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.conv1 = ConvModule(in_channels, out_channels, 3, stride=1, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = RepVGGBlock(out_channels, out_channels, use_alpha=use_alpha, act_cfg=act_cfg, norm_cfg=norm_cfg, use_bn_first=False)
        self.shortcut = shortcut

    def forward(self, x: 'Tensor') ->Tensor:
        """Forward process.
        Args:
            inputs (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y


class CSPResLayer(nn.Module):
    """PPYOLOE Backbone Stage.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        num_block (int): Number of blocks in this stage.
        block_cfg (dict): Config dict for block. Default config is
            suitable for PPYOLOE+ backbone. And in PPYOLOE neck,
            block_cfg is set to dict(type='PPYOLOEBasicBlock',
            shortcut=False, use_alpha=False). Defaults to
            dict(type='PPYOLOEBasicBlock', shortcut=True, use_alpha=True).
        stride (int): Stride of the convolution. In backbone, the stride
            must be set to 2. In neck, the stride must be set to 1.
            Defaults to 1.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.1, eps=1e-5).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        attention_cfg (dict, optional): Config dict for `EffectiveSELayer`.
            Defaults to dict(type='EffectiveSELayer',
            act_cfg=dict(type='HSigmoid')).
        use_spp (bool): Whether to use `SPPFBottleneck` layer.
            Defaults to False.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', num_block: 'int', block_cfg: 'ConfigType'=dict(type='PPYOLOEBasicBlock', shortcut=True, use_alpha=True), stride: 'int'=1, norm_cfg: 'ConfigType'=dict(type='BN', momentum=0.1, eps=1e-05), act_cfg: 'ConfigType'=dict(type='SiLU', inplace=True), attention_cfg: 'OptMultiConfig'=dict(type='EffectiveSELayer', act_cfg=dict(type='HSigmoid')), use_spp: 'bool'=False):
        super().__init__()
        self.num_block = num_block
        self.block_cfg = block_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.use_spp = use_spp
        assert attention_cfg is None or isinstance(attention_cfg, dict)
        if stride == 2:
            conv1_in_channels = conv2_in_channels = conv3_in_channels = (in_channels + out_channels) // 2
            blocks_channels = conv1_in_channels // 2
            self.conv_down = ConvModule(in_channels, conv1_in_channels, 3, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            conv1_in_channels = conv2_in_channels = in_channels
            conv3_in_channels = out_channels
            blocks_channels = out_channels // 2
            self.conv_down = None
        self.conv1 = ConvModule(conv1_in_channels, blocks_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(conv2_in_channels, blocks_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.blocks = self.build_blocks_layer(blocks_channels)
        self.conv3 = ConvModule(conv3_in_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        if attention_cfg:
            attention_cfg = attention_cfg.copy()
            attention_cfg['channels'] = blocks_channels * 2
            self.attn = MODELS.build(attention_cfg)
        else:
            self.attn = None

    def build_blocks_layer(self, blocks_channels: 'int') ->nn.Module:
        """Build blocks layer.

        Args:
            blocks_channels: The channels of this Module.
        """
        blocks = nn.Sequential()
        block_cfg = self.block_cfg.copy()
        block_cfg.update(dict(in_channels=blocks_channels, out_channels=blocks_channels))
        block_cfg.setdefault('norm_cfg', self.norm_cfg)
        block_cfg.setdefault('act_cfg', self.act_cfg)
        for i in range(self.num_block):
            blocks.add_module(str(i), MODELS.build(block_cfg))
            if i == (self.num_block - 1) // 2 and self.use_spp:
                blocks.add_module('spp', SPPFBottleneck(blocks_channels, blocks_channels, kernel_sizes=[5, 9, 13], use_conv_first=False, conv_cfg=None, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
        return blocks

    def forward(self, x: 'Tensor') ->Tensor:
        """Forward process
         Args:
             x (Tensor): The input tensor.
         """
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.cat([y1, y2], axis=1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        return y


class BiFusion(nn.Module):
    """BiFusion Block in YOLOv6.

    BiFusion fuses current-, high- and low-level features.
    Compared with concatenation in PAN, it fuses an extra low-level feature.

    Args:
        in_channels0 (int): The channels of current-level feature.
        in_channels1 (int): The input channels of lower-level feature.
        out_channels (int): The out channels of the BiFusion module.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self, in_channels0: 'int', in_channels1: 'int', out_channels: 'int', norm_cfg: 'ConfigType'=dict(type='BN', momentum=0.03, eps=0.001), act_cfg: 'ConfigType'=dict(type='ReLU', inplace=True)):
        super().__init__()
        self.conv1 = ConvModule(in_channels0, out_channels, kernel_size=1, stride=1, padding=0, bias=False, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(in_channels1, out_channels, kernel_size=1, stride=1, padding=0, bias=False, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv3 = ConvModule(out_channels * 3, out_channels, kernel_size=1, stride=1, padding=0, bias=False, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, bias=True)
        self.downsample = ConvModule(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x: 'List[torch.Tensor]') ->Tensor:
        """Forward process
        Args:
            x (List[torch.Tensor]): The tensor list of length 3.
                x[0]: The high-level feature.
                x[1]: The current-level feature.
                x[2]: The low-level feature.
        """
        x0 = self.upsample(x[0])
        x1 = self.conv1(x[1])
        x2 = self.downsample(self.conv2(x[2]))
        return self.conv3(torch.cat((x0, x1, x2), dim=1))


def bbox_overlaps(pred: 'torch.Tensor', target: 'torch.Tensor', iou_mode: 'str'='ciou', bbox_format: 'str'='xywh', siou_theta: 'float'=4.0, eps: 'float'=1e-07) ->torch.Tensor:
    """Calculate overlap between two set of bboxes.
    `Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.

    In the CIoU implementation of YOLOv5 and MMDetection, there is a slight
    difference in the way the alpha parameter is computed.

    mmdet version:
        alpha = (ious > 0.5).float() * v / (1 - ious + v)
    YOLOv5 version:
        alpha = v / (v - ious + (1 + eps)

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2)
            or (x, y, w, h),shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        iou_mode (str): Options are ('iou', 'ciou', 'giou', 'siou').
            Defaults to "ciou".
        bbox_format (str): Options are "xywh" and "xyxy".
            Defaults to "xywh".
        siou_theta (float): siou_theta for SIoU when calculate shape cost.
            Defaults to 4.0.
        eps (float): Eps to avoid log(0).

    Returns:
        Tensor: shape (n, ).
    """
    assert iou_mode in ('iou', 'ciou', 'giou', 'siou')
    assert bbox_format in ('xyxy', 'xywh')
    if bbox_format == 'xywh':
        pred = HorizontalBoxes.cxcywh_to_xyxy(pred)
        target = HorizontalBoxes.cxcywh_to_xyxy(target)
    bbox1_x1, bbox1_y1 = pred[..., 0], pred[..., 1]
    bbox1_x2, bbox1_y2 = pred[..., 2], pred[..., 3]
    bbox2_x1, bbox2_y1 = target[..., 0], target[..., 1]
    bbox2_x2, bbox2_y2 = target[..., 2], target[..., 3]
    overlap = (torch.min(bbox1_x2, bbox2_x2) - torch.max(bbox1_x1, bbox2_x1)).clamp(0) * (torch.min(bbox1_y2, bbox2_y2) - torch.max(bbox1_y1, bbox2_y1)).clamp(0)
    w1, h1 = bbox1_x2 - bbox1_x1, bbox1_y2 - bbox1_y1
    w2, h2 = bbox2_x2 - bbox2_x1, bbox2_y2 - bbox2_y1
    union = w1 * h1 + w2 * h2 - overlap + eps
    h1 = bbox1_y2 - bbox1_y1 + eps
    h2 = bbox2_y2 - bbox2_y1 + eps
    ious = overlap / union
    enclose_x1y1 = torch.min(pred[..., :2], target[..., :2])
    enclose_x2y2 = torch.max(pred[..., 2:], target[..., 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
    enclose_w = enclose_wh[..., 0]
    enclose_h = enclose_wh[..., 1]
    if iou_mode == 'ciou':
        enclose_area = enclose_w ** 2 + enclose_h ** 2 + eps
        rho2_left_item = (bbox2_x1 + bbox2_x2 - (bbox1_x1 + bbox1_x2)) ** 2 / 4
        rho2_right_item = (bbox2_y1 + bbox2_y2 - (bbox1_y1 + bbox1_y2)) ** 2 / 4
        rho2 = rho2_left_item + rho2_right_item
        wh_ratio = 4 / math.pi ** 2 * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
        with torch.no_grad():
            alpha = wh_ratio / (wh_ratio - ious + (1 + eps))
        ious = ious - (rho2 / enclose_area + alpha * wh_ratio)
    elif iou_mode == 'giou':
        convex_area = enclose_w * enclose_h + eps
        ious = ious - (convex_area - union) / convex_area
    elif iou_mode == 'siou':
        sigma_cw = (bbox2_x1 + bbox2_x2) / 2 - (bbox1_x1 + bbox1_x2) / 2 + eps
        sigma_ch = (bbox2_y1 + bbox2_y2) / 2 - (bbox1_y1 + bbox1_y2) / 2 + eps
        sigma = torch.pow(sigma_cw ** 2 + sigma_ch ** 2, 0.5)
        sin_alpha = torch.abs(sigma_ch) / sigma
        sin_beta = torch.abs(sigma_cw) / sigma
        sin_alpha = torch.where(sin_alpha <= math.sin(math.pi / 4), sin_alpha, sin_beta)
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
        rho_x = (sigma_cw / enclose_w) ** 2
        rho_y = (sigma_ch / enclose_h) ** 2
        gamma = 2 - angle_cost
        distance_cost = 1 - torch.exp(-1 * gamma * rho_x) + (1 - torch.exp(-1 * gamma * rho_y))
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), siou_theta) + torch.pow(1 - torch.exp(-1 * omiga_h), siou_theta)
        ious = ious - (distance_cost + shape_cost) * 0.5
    return ious.clamp(min=-1.0, max=1.0)


class IoULoss(nn.Module):
    """IoULoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    Args:
        iou_mode (str): Options are "ciou".
            Defaults to "ciou".
        bbox_format (str): Options are "xywh" and "xyxy".
            Defaults to "xywh".
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        return_iou (bool): If True, return loss and iou.
    """

    def __init__(self, iou_mode: 'str'='ciou', bbox_format: 'str'='xywh', eps: 'float'=1e-07, reduction: 'str'='mean', loss_weight: 'float'=1.0, return_iou: 'bool'=True):
        super().__init__()
        assert bbox_format in ('xywh', 'xyxy')
        assert iou_mode in ('ciou', 'siou', 'giou')
        self.iou_mode = iou_mode
        self.bbox_format = bbox_format
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.return_iou = return_iou

    def forward(self, pred: 'torch.Tensor', target: 'torch.Tensor', weight: 'Optional[torch.Tensor]'=None, avg_factor: 'Optional[float]'=None, reduction_override: 'Optional[Union[str, bool]]'=None) ->Tuple[Union[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2)
                or (x, y, w, h),shape (n, 4).
            target (Tensor): Corresponding gt bboxes, shape (n, 4).
            weight (Tensor, optional): Element-wise weights.
            avg_factor (float, optional): Average factor when computing the
                mean of losses.
            reduction_override (str, bool, optional): Same as built-in losses
                of PyTorch. Defaults to None.
        Returns:
            loss or tuple(loss, iou):
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and weight.dim() > 1:
            weight = weight.mean(-1)
        iou = bbox_overlaps(pred, target, iou_mode=self.iou_mode, bbox_format=self.bbox_format, eps=self.eps)
        loss = self.loss_weight * weight_reduce_loss(1.0 - iou, weight, reduction, avg_factor)
        if self.return_iou:
            return loss, iou
        else:
            return loss


class OksLoss(nn.Module):
    """A PyTorch implementation of the Object Keypoint Similarity (OKS) loss as
    described in the paper "YOLO-Pose: Enhancing YOLO for Multi Person Pose
    Estimation Using Object Keypoint Similarity Loss" by Debapriya et al.

    (2022).
    The OKS loss is used for keypoint-based object recognition and consists
    of a measure of the similarity between predicted and ground truth
    keypoint locations, adjusted by the size of the object in the image.
    The loss function takes as input the predicted keypoint locations, the
    ground truth keypoint locations, a mask indicating which keypoints are
    valid, and bounding boxes for the objects.
    Args:
        metainfo (Optional[str]): Path to a JSON file containing information
            about the dataset's annotations.
        loss_weight (float): Weight for the loss.
    """

    def __init__(self, metainfo: 'Optional[str]'=None, loss_weight: 'float'=1.0):
        super().__init__()
        if metainfo is not None:
            if parse_pose_metainfo is None:
                raise ImportError('Please run "mim install -r requirements/mmpose.txt" to install mmpose first for OksLossn.')
            metainfo = parse_pose_metainfo(dict(from_file=metainfo))
            sigmas = metainfo.get('sigmas', None)
            if sigmas is not None:
                self.register_buffer('sigmas', torch.as_tensor(sigmas))
        self.loss_weight = loss_weight

    def forward(self, output: 'Tensor', target: 'Tensor', target_weights: 'Tensor', bboxes: 'Optional[Tensor]'=None) ->Tensor:
        oks = self.compute_oks(output, target, target_weights, bboxes)
        loss = 1 - oks
        return loss * self.loss_weight

    def compute_oks(self, output: 'Tensor', target: 'Tensor', target_weights: 'Tensor', bboxes: 'Optional[Tensor]'=None) ->Tensor:
        """Calculates the OKS loss.

        Args:
            output (Tensor): Predicted keypoints in shape N x k x 2, where N
                is batch size, k is the number of keypoints, and 2 are the
                xy coordinates.
            target (Tensor): Ground truth keypoints in the same shape as
                output.
            target_weights (Tensor): Mask of valid keypoints in shape N x k,
                with 1 for valid and 0 for invalid.
            bboxes (Optional[Tensor]): Bounding boxes in shape N x 4,
                where 4 are the xyxy coordinates.
        Returns:
            Tensor: The calculated OKS loss.
        """
        dist = torch.norm(output - target, dim=-1)
        if hasattr(self, 'sigmas'):
            sigmas = self.sigmas.reshape(*((1,) * (dist.ndim - 1)), -1)
            dist = dist / sigmas
        if bboxes is not None:
            area = torch.norm(bboxes[..., 2:] - bboxes[..., :2], dim=-1)
            dist = dist / area.clip(min=1e-08).unsqueeze(-1)
        return (torch.exp(-dist.pow(2) / 2) * target_weights).sum(dim=-1) / target_weights.sum(dim=-1).clip(min=1e-08)


def bbox_center_distance(bboxes: 'Tensor', priors: 'Tensor') ->Tuple[Tensor, Tensor]:
    """Compute the center distance between bboxes and priors.

    Args:
        bboxes (Tensor): Shape (n, 4) for bbox, "xyxy" format.
        priors (Tensor): Shape (num_priors, 4) for priors, "xyxy" format.

    Returns:
        distances (Tensor): Center distances between bboxes and priors,
            shape (num_priors, n).
        priors_points (Tensor): Priors cx cy points,
            shape (num_priors, 2).
    """
    bbox_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    bbox_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
    bbox_points = torch.stack((bbox_cx, bbox_cy), dim=1)
    priors_cx = (priors[:, 0] + priors[:, 2]) / 2.0
    priors_cy = (priors[:, 1] + priors[:, 3]) / 2.0
    priors_points = torch.stack((priors_cx, priors_cy), dim=1)
    distances = (bbox_points[:, None, :] - priors_points[None, :, :]).pow(2).sum(-1).sqrt()
    return distances, priors_points


def select_candidates_in_gts(priors_points: 'Tensor', gt_bboxes: 'Tensor', eps: 'float'=1e-09) ->Tensor:
    """Select the positive priors' center in gt.

    Args:
        priors_points (Tensor): Model priors points,
            shape(num_priors, 2)
        gt_bboxes (Tensor): Ground true bboxes,
            shape(batch_size, num_gt, 4)
        eps (float): Default to 1e-9.
    Return:
        (Tensor): shape(batch_size, num_gt, num_priors)
    """
    batch_size, num_gt, _ = gt_bboxes.size()
    gt_bboxes = gt_bboxes.reshape([-1, 4])
    priors_number = priors_points.size(0)
    priors_points = priors_points.unsqueeze(0).repeat(batch_size * num_gt, 1, 1)
    gt_bboxes_lt = gt_bboxes[:, 0:2].unsqueeze(1).repeat(1, priors_number, 1)
    gt_bboxes_rb = gt_bboxes[:, 2:4].unsqueeze(1).repeat(1, priors_number, 1)
    bbox_deltas = torch.cat([priors_points - gt_bboxes_lt, gt_bboxes_rb - priors_points], dim=-1)
    bbox_deltas = bbox_deltas.reshape([batch_size, num_gt, priors_number, -1])
    return bbox_deltas.min(axis=-1)[0] > eps


def select_highest_overlaps(pos_mask: 'Tensor', overlaps: 'Tensor', num_gt: 'int') ->Tuple[Tensor, Tensor, Tensor]:
    """If an anchor box is assigned to multiple gts, the one with the highest
    iou will be selected.

    Args:
        pos_mask (Tensor): The assigned positive sample mask,
            shape(batch_size, num_gt, num_priors)
        overlaps (Tensor): IoU between all bbox and ground truth,
            shape(batch_size, num_gt, num_priors)
        num_gt (int): Number of ground truth.
    Return:
        gt_idx_pre_prior (Tensor): Target ground truth index,
            shape(batch_size, num_priors)
        fg_mask_pre_prior (Tensor): Force matching ground truth,
            shape(batch_size, num_priors)
        pos_mask (Tensor): The assigned positive sample mask,
            shape(batch_size, num_gt, num_priors)
    """
    fg_mask_pre_prior = pos_mask.sum(axis=-2)
    if fg_mask_pre_prior.max() > 1:
        mask_multi_gts = (fg_mask_pre_prior.unsqueeze(1) > 1).repeat([1, num_gt, 1])
        index = overlaps.argmax(axis=1)
        is_max_overlaps = F.one_hot(index, num_gt)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1)
        pos_mask = torch.where(mask_multi_gts, is_max_overlaps, pos_mask)
        fg_mask_pre_prior = pos_mask.sum(axis=-2)
    gt_idx_pre_prior = pos_mask.argmax(axis=-2)
    return gt_idx_pre_prior, fg_mask_pre_prior, pos_mask


def yolov6_iou_calculator(bbox1: 'Tensor', bbox2: 'Tensor', eps: 'float'=1e-09) ->Tensor:
    """Calculate iou for batch.

    Args:
        bbox1 (Tensor): shape(batch size, num_gt, 4)
        bbox2 (Tensor): shape(batch size, num_priors, 4)
        eps (float): Default to 1e-9.
    Return:
        (Tensor): IoU, shape(size, num_gt, num_priors)
    """
    bbox1 = bbox1.unsqueeze(2)
    bbox2 = bbox2.unsqueeze(1)
    bbox1_x1y1, bbox1_x2y2 = bbox1[:, :, :, 0:2], bbox1[:, :, :, 2:4]
    bbox2_x1y1, bbox2_x2y2 = bbox2[:, :, :, 0:2], bbox2[:, :, :, 2:4]
    overlap = (torch.minimum(bbox1_x2y2, bbox2_x2y2) - torch.maximum(bbox1_x1y1, bbox2_x1y1)).clip(0).prod(-1)
    bbox1_area = (bbox1_x2y2 - bbox1_x1y1).clip(0).prod(-1)
    bbox2_area = (bbox2_x2y2 - bbox2_x1y1).clip(0).prod(-1)
    union = bbox1_area + bbox2_area - overlap + eps
    return overlap / union


class BatchATSSAssigner(nn.Module):
    """Assign a batch of corresponding gt bboxes or background to each prior.

    This code is based on
    https://github.com/meituan/YOLOv6/blob/main/yolov6/assigners/atss_assigner.py

    Each proposal will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        num_classes (int): number of class
        iou_calculator (:obj:`ConfigDict` or dict): Config dict for iou
            calculator. Defaults to ``dict(type='BboxOverlaps2D')``
        topk (int): number of priors selected in each level
    """

    def __init__(self, num_classes: 'int', iou_calculator: 'ConfigType'=dict(type='mmdet.BboxOverlaps2D'), topk: 'int'=9):
        super().__init__()
        self.num_classes = num_classes
        self.iou_calculator = TASK_UTILS.build(iou_calculator)
        self.topk = topk

    @torch.no_grad()
    def forward(self, pred_bboxes: 'Tensor', priors: 'Tensor', num_level_priors: 'List', gt_labels: 'Tensor', gt_bboxes: 'Tensor', pad_bbox_flag: 'Tensor') ->dict:
        """Assign gt to priors.

        The assignment is done in following steps

        1. compute iou between all prior (prior of all pyramid levels) and gt
        2. compute center distance between all prior and gt
        3. on each pyramid level, for each gt, select k prior whose center
           are closest to the gt center, so we total select k*l prior as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt

        Args:
            pred_bboxes (Tensor): Predicted bounding boxes,
                shape(batch_size, num_priors, 4)
            priors (Tensor): Model priors with stride, shape(num_priors, 4)
            num_level_priors (List): Number of bboxes in each level, len(3)
            gt_labels (Tensor): Ground truth label,
                shape(batch_size, num_gt, 1)
            gt_bboxes (Tensor): Ground truth bbox,
                shape(batch_size, num_gt, 4)
            pad_bbox_flag (Tensor): Ground truth bbox mask,
                1 means bbox, 0 means no bbox,
                shape(batch_size, num_gt, 1)
        Returns:
            assigned_result (dict): Assigned result
                'assigned_labels' (Tensor): shape(batch_size, num_gt)
                'assigned_bboxes' (Tensor): shape(batch_size, num_gt, 4)
                'assigned_scores' (Tensor):
                    shape(batch_size, num_gt, number_classes)
                'fg_mask_pre_prior' (Tensor): shape(bs, num_gt)
        """
        cell_half_size = priors[:, 2:] * 2.5
        priors_gen = torch.zeros_like(priors)
        priors_gen[:, :2] = priors[:, :2] - cell_half_size
        priors_gen[:, 2:] = priors[:, :2] + cell_half_size
        priors = priors_gen
        batch_size = gt_bboxes.size(0)
        num_gt, num_priors = gt_bboxes.size(1), priors.size(0)
        assigned_result = {'assigned_labels': gt_bboxes.new_full([batch_size, num_priors], self.num_classes), 'assigned_bboxes': gt_bboxes.new_full([batch_size, num_priors, 4], 0), 'assigned_scores': gt_bboxes.new_full([batch_size, num_priors, self.num_classes], 0), 'fg_mask_pre_prior': gt_bboxes.new_full([batch_size, num_priors], 0)}
        if num_gt == 0:
            return assigned_result
        overlaps = self.iou_calculator(gt_bboxes.reshape([-1, 4]), priors)
        overlaps = overlaps.reshape([batch_size, -1, num_priors])
        distances, priors_points = bbox_center_distance(gt_bboxes.reshape([-1, 4]), priors)
        distances = distances.reshape([batch_size, -1, num_priors])
        is_in_candidate, candidate_idxs = self.select_topk_candidates(distances, num_level_priors, pad_bbox_flag)
        overlaps_thr_per_gt, iou_candidates = self.threshold_calculator(is_in_candidate, candidate_idxs, overlaps, num_priors, batch_size, num_gt)
        is_pos = torch.where(iou_candidates > overlaps_thr_per_gt.repeat([1, 1, num_priors]), is_in_candidate, torch.zeros_like(is_in_candidate))
        is_in_gts = select_candidates_in_gts(priors_points, gt_bboxes)
        pos_mask = is_pos * is_in_gts * pad_bbox_flag
        gt_idx_pre_prior, fg_mask_pre_prior, pos_mask = select_highest_overlaps(pos_mask, overlaps, num_gt)
        assigned_labels, assigned_bboxes, assigned_scores = self.get_targets(gt_labels, gt_bboxes, gt_idx_pre_prior, fg_mask_pre_prior, num_priors, batch_size, num_gt)
        if pred_bboxes is not None:
            ious = yolov6_iou_calculator(gt_bboxes, pred_bboxes) * pos_mask
            ious = ious.max(axis=-2)[0].unsqueeze(-1)
            assigned_scores *= ious
        assigned_result['assigned_labels'] = assigned_labels.long()
        assigned_result['assigned_bboxes'] = assigned_bboxes
        assigned_result['assigned_scores'] = assigned_scores
        assigned_result['fg_mask_pre_prior'] = fg_mask_pre_prior.bool()
        return assigned_result

    def select_topk_candidates(self, distances: 'Tensor', num_level_priors: 'List[int]', pad_bbox_flag: 'Tensor') ->Tuple[Tensor, Tensor]:
        """Selecting candidates based on the center distance.

        Args:
            distances (Tensor): Distance between all bbox and gt,
                shape(batch_size, num_gt, num_priors)
            num_level_priors (List[int]): Number of bboxes in each level,
                len(3)
            pad_bbox_flag (Tensor): Ground truth bbox mask,
                shape(batch_size, num_gt, 1)

        Return:
            is_in_candidate_list (Tensor): Flag show that each level have
                topk candidates or not,  shape(batch_size, num_gt, num_priors)
            candidate_idxs (Tensor): Candidates index,
                shape(batch_size, num_gt, num_gt)
        """
        is_in_candidate_list = []
        candidate_idxs = []
        start_idx = 0
        distances_dtype = distances.dtype
        distances = torch.split(distances, num_level_priors, dim=-1)
        pad_bbox_flag = pad_bbox_flag.repeat(1, 1, self.topk).bool()
        for distances_per_level, priors_per_level in zip(distances, num_level_priors):
            end_index = start_idx + priors_per_level
            selected_k = min(self.topk, priors_per_level)
            _, topk_idxs_per_level = distances_per_level.topk(selected_k, dim=-1, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            topk_idxs_per_level = torch.where(pad_bbox_flag, topk_idxs_per_level, torch.zeros_like(topk_idxs_per_level))
            is_in_candidate = F.one_hot(topk_idxs_per_level, priors_per_level).sum(dim=-2)
            is_in_candidate = torch.where(is_in_candidate > 1, torch.zeros_like(is_in_candidate), is_in_candidate)
            is_in_candidate_list.append(is_in_candidate)
            start_idx = end_index
        is_in_candidate_list = torch.cat(is_in_candidate_list, dim=-1)
        candidate_idxs = torch.cat(candidate_idxs, dim=-1)
        return is_in_candidate_list, candidate_idxs

    @staticmethod
    def threshold_calculator(is_in_candidate: 'List', candidate_idxs: 'Tensor', overlaps: 'Tensor', num_priors: 'int', batch_size: 'int', num_gt: 'int') ->Tuple[Tensor, Tensor]:
        """Get corresponding iou for the these candidates, and compute the mean
        and std, set mean + std as the iou threshold.

        Args:
            is_in_candidate (Tensor): Flag show that each level have
                topk candidates or not, shape(batch_size, num_gt, num_priors).
            candidate_idxs (Tensor): Candidates index,
                shape(batch_size, num_gt, num_gt)
            overlaps (Tensor): Overlaps area,
                shape(batch_size, num_gt, num_priors).
            num_priors (int): Number of priors.
            batch_size (int): Batch size.
            num_gt (int): Number of ground truth.

        Return:
            overlaps_thr_per_gt (Tensor): Overlap threshold of
                per ground truth, shape(batch_size, num_gt, 1).
            candidate_overlaps (Tensor): Candidate overlaps,
                shape(batch_size, num_gt, num_priors).
        """
        batch_size_num_gt = batch_size * num_gt
        candidate_overlaps = torch.where(is_in_candidate > 0, overlaps, torch.zeros_like(overlaps))
        candidate_idxs = candidate_idxs.reshape([batch_size_num_gt, -1])
        assist_indexes = num_priors * torch.arange(batch_size_num_gt, device=candidate_idxs.device)
        assist_indexes = assist_indexes[:, None]
        flatten_indexes = candidate_idxs + assist_indexes
        candidate_overlaps_reshape = candidate_overlaps.reshape(-1)[flatten_indexes]
        candidate_overlaps_reshape = candidate_overlaps_reshape.reshape([batch_size, num_gt, -1])
        overlaps_mean_per_gt = candidate_overlaps_reshape.mean(axis=-1, keepdim=True)
        overlaps_std_per_gt = candidate_overlaps_reshape.std(axis=-1, keepdim=True)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt
        return overlaps_thr_per_gt, candidate_overlaps

    def get_targets(self, gt_labels: 'Tensor', gt_bboxes: 'Tensor', assigned_gt_inds: 'Tensor', fg_mask_pre_prior: 'Tensor', num_priors: 'int', batch_size: 'int', num_gt: 'int') ->Tuple[Tensor, Tensor, Tensor]:
        """Get target info.

        Args:
            gt_labels (Tensor): Ground true labels,
                shape(batch_size, num_gt, 1)
            gt_bboxes (Tensor): Ground true bboxes,
                shape(batch_size, num_gt, 4)
            assigned_gt_inds (Tensor): Assigned ground truth indexes,
                shape(batch_size, num_priors)
            fg_mask_pre_prior (Tensor): Force ground truth matching mask,
                shape(batch_size, num_priors)
            num_priors (int): Number of priors.
            batch_size (int): Batch size.
            num_gt (int): Number of ground truth.

        Return:
            assigned_labels (Tensor): Assigned labels,
                shape(batch_size, num_priors)
            assigned_bboxes (Tensor): Assigned bboxes,
                shape(batch_size, num_priors)
            assigned_scores (Tensor): Assigned scores,
                shape(batch_size, num_priors)
        """
        batch_index = torch.arange(batch_size, dtype=gt_labels.dtype, device=gt_labels.device)
        batch_index = batch_index[..., None]
        assigned_gt_inds = (assigned_gt_inds + batch_index * num_gt).long()
        assigned_labels = gt_labels.flatten()[assigned_gt_inds.flatten()]
        assigned_labels = assigned_labels.reshape([batch_size, num_priors])
        assigned_labels = torch.where(fg_mask_pre_prior > 0, assigned_labels, torch.full_like(assigned_labels, self.num_classes))
        assigned_bboxes = gt_bboxes.reshape([-1, 4])[assigned_gt_inds.flatten()]
        assigned_bboxes = assigned_bboxes.reshape([batch_size, num_priors, 4])
        assigned_scores = F.one_hot(assigned_labels.long(), self.num_classes + 1).float()
        assigned_scores = assigned_scores[:, :, :self.num_classes]
        return assigned_labels, assigned_bboxes, assigned_scores


EPS = 1e-07


INF = 100000.0


def find_inside_points(boxes: 'Tensor', points: 'Tensor', box_dim: 'int'=4, eps: 'float'=0.01) ->Tensor:
    """Find inside box points in batches. Boxes dimension must be 3.

    Args:
        boxes (Tensor): Boxes tensor. Must be batch input.
            Has shape of (batch_size, n_boxes, box_dim).
        points (Tensor): Points coordinates. Has shape of (n_points, 2).
        box_dim (int): The dimension of box. 4 means horizontal box and
            5 means rotated box. Defaults to 4.
        eps (float): Make sure the points are inside not on the boundary.
            Only use in rotated boxes. Defaults to 0.01.

    Returns:
        Tensor: A BoolTensor indicating whether a point is inside
        boxes. The index has shape of (n_points, batch_size, n_boxes).
    """
    if box_dim == 4:
        lt_ = points[:, None, None] - boxes[..., :2]
        rb_ = boxes[..., 2:] - points[:, None, None]
        deltas = torch.cat([lt_, rb_], dim=-1)
        is_in_gts = deltas.min(dim=-1).values > 0
    elif box_dim == 5:
        points = points[:, None, None]
        ctrs, wh, t = torch.split(boxes, [2, 2, 1], dim=-1)
        cos_value, sin_value = torch.cos(t), torch.sin(t)
        matrix = torch.cat([cos_value, sin_value, -sin_value, cos_value], dim=-1).reshape(*boxes.shape[:-1], 2, 2)
        offset = points - ctrs
        offset = torch.matmul(matrix, offset[..., None])
        offset = offset.squeeze(-1)
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        w, h = wh[..., 0], wh[..., 1]
        is_in_gts = (offset_x <= w / 2 - eps) & (offset_x >= -w / 2 + eps) & (offset_y <= h / 2 - eps) & (offset_y >= -h / 2 + eps)
    else:
        raise NotImplementedError(f'Unsupport box_dim:{box_dim}')
    return is_in_gts


def get_box_center(boxes: 'Tensor', box_dim: 'int'=4) ->Tensor:
    """Return a tensor representing the centers of boxes.

    Args:
        boxes (Tensor): Boxes tensor. Has shape of (b, n, box_dim)
        box_dim (int): The dimension of box. 4 means horizontal box and
            5 means rotated box. Defaults to 4.

    Returns:
        Tensor: Centers have shape of (b, n, 2)
    """
    if box_dim == 4:
        return (boxes[..., :2] + boxes[..., 2:]) / 2.0
    elif box_dim == 5:
        return boxes[..., :2]
    else:
        raise NotImplementedError(f'Unsupported box_dim:{box_dim}')


class BatchDynamicSoftLabelAssigner(nn.Module):
    """Computes matching between predictions and ground truth with dynamic soft
    label assignment.

    Args:
        num_classes (int): number of class
        soft_center_radius (float): Radius of the soft center prior.
            Defaults to 3.0.
        topk (int): Select top-k predictions to calculate dynamic k
            best matches for each gt. Defaults to 13.
        iou_weight (float): The scale factor of iou cost. Defaults to 3.0.
        iou_calculator (ConfigType): Config of overlaps Calculator.
            Defaults to dict(type='BboxOverlaps2D').
        batch_iou (bool): Use batch input when calculate IoU.
            If set to False use loop instead. Defaults to True.
    """

    def __init__(self, num_classes, soft_center_radius: 'float'=3.0, topk: 'int'=13, iou_weight: 'float'=3.0, iou_calculator: 'ConfigType'=dict(type='mmdet.BboxOverlaps2D'), batch_iou: 'bool'=True) ->None:
        super().__init__()
        self.num_classes = num_classes
        self.soft_center_radius = soft_center_radius
        self.topk = topk
        self.iou_weight = iou_weight
        self.iou_calculator = TASK_UTILS.build(iou_calculator)
        self.batch_iou = batch_iou

    @torch.no_grad()
    def forward(self, pred_bboxes: 'Tensor', pred_scores: 'Tensor', priors: 'Tensor', gt_labels: 'Tensor', gt_bboxes: 'Tensor', pad_bbox_flag: 'Tensor') ->dict:
        num_gt = gt_bboxes.size(1)
        decoded_bboxes = pred_bboxes
        batch_size, num_bboxes, box_dim = decoded_bboxes.size()
        if num_gt == 0 or num_bboxes == 0:
            return {'assigned_labels': gt_labels.new_full(pred_scores[..., 0].shape, self.num_classes, dtype=torch.long), 'assigned_labels_weights': gt_bboxes.new_full(pred_scores[..., 0].shape, 1), 'assigned_bboxes': gt_bboxes.new_full(pred_bboxes.shape, 0), 'assign_metrics': gt_bboxes.new_full(pred_scores[..., 0].shape, 0)}
        prior_center = priors[:, :2]
        if isinstance(gt_bboxes, BaseBoxes):
            raise NotImplementedError(f'type of {type(gt_bboxes)} are not implemented !')
        else:
            is_in_gts = find_inside_points(gt_bboxes, prior_center, box_dim)
        is_in_gts = is_in_gts * pad_bbox_flag[..., 0][None]
        is_in_gts = is_in_gts.permute(1, 0, 2)
        valid_mask = is_in_gts.sum(dim=-1) > 0
        gt_center = get_box_center(gt_bboxes, box_dim)
        strides = priors[..., 2]
        distance = (priors[None].unsqueeze(2)[..., :2] - gt_center[:, None, :, :]).pow(2).sum(-1).sqrt() / strides[None, :, None]
        distance = distance * valid_mask.unsqueeze(-1)
        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)
        if self.batch_iou:
            pairwise_ious = self.iou_calculator(decoded_bboxes, gt_bboxes)
        else:
            ious = []
            for box, gt in zip(decoded_bboxes, gt_bboxes):
                iou = self.iou_calculator(box, gt)
                ious.append(iou)
            pairwise_ious = torch.stack(ious, dim=0)
        iou_cost = -torch.log(pairwise_ious + EPS) * self.iou_weight
        pairwise_pred_scores = pred_scores.permute(0, 2, 1)
        idx = torch.zeros([2, batch_size, num_gt], dtype=torch.long)
        idx[0] = torch.arange(end=batch_size).view(-1, 1).repeat(1, num_gt)
        idx[1] = gt_labels.long().squeeze(-1)
        pairwise_pred_scores = pairwise_pred_scores[idx[0], idx[1]].permute(0, 2, 1)
        scale_factor = pairwise_ious - pairwise_pred_scores.sigmoid()
        pairwise_cls_cost = F.binary_cross_entropy_with_logits(pairwise_pred_scores, pairwise_ious, reduction='none') * scale_factor.abs().pow(2.0)
        cost_matrix = pairwise_cls_cost + iou_cost + soft_center_prior
        max_pad_value = torch.ones_like(cost_matrix) * INF
        cost_matrix = torch.where(valid_mask[..., None].repeat(1, 1, num_gt), cost_matrix, max_pad_value)
        matched_pred_ious, matched_gt_inds, fg_mask_inboxes = self.dynamic_k_matching(cost_matrix, pairwise_ious, pad_bbox_flag)
        del pairwise_ious, cost_matrix
        batch_index = (fg_mask_inboxes > 0).nonzero(as_tuple=True)[0]
        assigned_labels = gt_labels.new_full(pred_scores[..., 0].shape, self.num_classes)
        assigned_labels[fg_mask_inboxes] = gt_labels[batch_index, matched_gt_inds].squeeze(-1)
        assigned_labels = assigned_labels.long()
        assigned_labels_weights = gt_bboxes.new_full(pred_scores[..., 0].shape, 1)
        assigned_bboxes = gt_bboxes.new_full(pred_bboxes.shape, 0)
        assigned_bboxes[fg_mask_inboxes] = gt_bboxes[batch_index, matched_gt_inds]
        assign_metrics = gt_bboxes.new_full(pred_scores[..., 0].shape, 0)
        assign_metrics[fg_mask_inboxes] = matched_pred_ious
        return dict(assigned_labels=assigned_labels, assigned_labels_weights=assigned_labels_weights, assigned_bboxes=assigned_bboxes, assign_metrics=assign_metrics)

    def dynamic_k_matching(self, cost_matrix: 'Tensor', pairwise_ious: 'Tensor', pad_bbox_flag: 'int') ->Tuple[Tensor, Tensor, Tensor]:
        """Use IoU and matching cost to calculate the dynamic top-k positive
        targets.

        Args:
            cost_matrix (Tensor): Cost matrix.
            pairwise_ious (Tensor): Pairwise iou matrix.
            num_gt (int): Number of gt.
            valid_mask (Tensor): Mask for valid bboxes.
        Returns:
            tuple: matched ious and gt indexes.
        """
        matching_matrix = torch.zeros_like(cost_matrix, dtype=torch.uint8)
        candidate_topk = min(self.topk, pairwise_ious.size(1))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        num_gts = pad_bbox_flag.sum((1, 2)).int()
        _, sorted_indices = torch.sort(cost_matrix, dim=1)
        for b in range(pad_bbox_flag.shape[0]):
            for gt_idx in range(num_gts[b]):
                topk_ids = sorted_indices[b, :dynamic_ks[b, gt_idx], gt_idx]
                matching_matrix[b, :, gt_idx][topk_ids] = 1
        del topk_ious, dynamic_ks
        prior_match_gt_mask = matching_matrix.sum(2) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(cost_matrix[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1
        fg_mask_inboxes = matching_matrix.sum(2) > 0
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(2)[fg_mask_inboxes]
        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        return matched_pred_ious, matched_gt_inds, fg_mask_inboxes


class BatchTaskAlignedAssigner(nn.Module):
    """This code referenced to
    https://github.com/meituan/YOLOv6/blob/main/yolov6/
    assigners/tal_assigner.py.
    Batch Task aligned assigner base on the paper:
    `TOOD: Task-aligned One-stage Object Detection.
    <https://arxiv.org/abs/2108.07755>`_.
    Assign a corresponding gt bboxes or background to a batch of
    predicted bboxes. Each bbox will be assigned with `0` or a
    positive integer indicating the ground truth index.
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        num_classes (int): number of class
        topk (int): number of bbox selected in each level
        alpha (float): Hyper-parameters related to alignment_metrics.
            Defaults to 1.0
        beta (float): Hyper-parameters related to alignment_metrics.
            Defaults to 6.
        eps (float): Eps to avoid log(0). Default set to 1e-9
        use_ciou (bool): Whether to use ciou while calculating iou.
            Defaults to False.
    """

    def __init__(self, num_classes: 'int', topk: 'int'=13, alpha: 'float'=1.0, beta: 'float'=6.0, eps: 'float'=1e-07, use_ciou: 'bool'=False):
        super().__init__()
        self.num_classes = num_classes
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.use_ciou = use_ciou

    @torch.no_grad()
    def forward(self, pred_bboxes: 'Tensor', pred_scores: 'Tensor', priors: 'Tensor', gt_labels: 'Tensor', gt_bboxes: 'Tensor', pad_bbox_flag: 'Tensor') ->dict:
        """Assign gt to bboxes.

        The assignment is done in following steps
        1. compute alignment metric between all bbox (bbox of all pyramid
           levels) and gt
        2. select top-k bbox as candidates for each gt
        3. limit the positive sample's center in gt (because the anchor-free
           detector only can predict positive distance)
        Args:
            pred_bboxes (Tensor): Predict bboxes,
                shape(batch_size, num_priors, 4)
            pred_scores (Tensor): Scores of predict bboxes,
                shape(batch_size, num_priors, num_classes)
            priors (Tensor): Model priors,  shape (num_priors, 4)
            gt_labels (Tensor): Ground true labels,
                shape(batch_size, num_gt, 1)
            gt_bboxes (Tensor): Ground true bboxes,
                shape(batch_size, num_gt, 4)
            pad_bbox_flag (Tensor): Ground truth bbox mask,
                1 means bbox, 0 means no bbox,
                shape(batch_size, num_gt, 1)
        Returns:
            assigned_result (dict) Assigned result:
                assigned_labels (Tensor): Assigned labels,
                    shape(batch_size, num_priors)
                assigned_bboxes (Tensor): Assigned boxes,
                    shape(batch_size, num_priors, 4)
                assigned_scores (Tensor): Assigned scores,
                    shape(batch_size, num_priors, num_classes)
                fg_mask_pre_prior (Tensor): Force ground truth matching mask,
                    shape(batch_size, num_priors)
        """
        priors = priors[:, :2]
        batch_size = pred_scores.size(0)
        num_gt = gt_bboxes.size(1)
        assigned_result = {'assigned_labels': gt_bboxes.new_full(pred_scores[..., 0].shape, self.num_classes), 'assigned_bboxes': gt_bboxes.new_full(pred_bboxes.shape, 0), 'assigned_scores': gt_bboxes.new_full(pred_scores.shape, 0), 'fg_mask_pre_prior': gt_bboxes.new_full(pred_scores[..., 0].shape, 0)}
        if num_gt == 0:
            return assigned_result
        pos_mask, alignment_metrics, overlaps = self.get_pos_mask(pred_bboxes, pred_scores, priors, gt_labels, gt_bboxes, pad_bbox_flag, batch_size, num_gt)
        assigned_gt_idxs, fg_mask_pre_prior, pos_mask = select_highest_overlaps(pos_mask, overlaps, num_gt)
        assigned_labels, assigned_bboxes, assigned_scores = self.get_targets(gt_labels, gt_bboxes, assigned_gt_idxs, fg_mask_pre_prior, batch_size, num_gt)
        alignment_metrics *= pos_mask
        pos_align_metrics = alignment_metrics.max(axis=-1, keepdim=True)[0]
        pos_overlaps = (overlaps * pos_mask).max(axis=-1, keepdim=True)[0]
        norm_align_metric = (alignment_metrics * pos_overlaps / (pos_align_metrics + self.eps)).max(-2)[0].unsqueeze(-1)
        assigned_scores = assigned_scores * norm_align_metric
        assigned_result['assigned_labels'] = assigned_labels
        assigned_result['assigned_bboxes'] = assigned_bboxes
        assigned_result['assigned_scores'] = assigned_scores
        assigned_result['fg_mask_pre_prior'] = fg_mask_pre_prior.bool()
        return assigned_result

    def get_pos_mask(self, pred_bboxes: 'Tensor', pred_scores: 'Tensor', priors: 'Tensor', gt_labels: 'Tensor', gt_bboxes: 'Tensor', pad_bbox_flag: 'Tensor', batch_size: 'int', num_gt: 'int') ->Tuple[Tensor, Tensor, Tensor]:
        """Get possible mask.

        Args:
            pred_bboxes (Tensor): Predict bboxes,
                shape(batch_size, num_priors, 4)
            pred_scores (Tensor): Scores of predict bbox,
                shape(batch_size, num_priors, num_classes)
            priors (Tensor): Model priors, shape (num_priors, 2)
            gt_labels (Tensor): Ground true labels,
                shape(batch_size, num_gt, 1)
            gt_bboxes (Tensor): Ground true bboxes,
                shape(batch_size, num_gt, 4)
            pad_bbox_flag (Tensor): Ground truth bbox mask,
                1 means bbox, 0 means no bbox,
                shape(batch_size, num_gt, 1)
            batch_size (int): Batch size.
            num_gt (int): Number of ground truth.
        Returns:
            pos_mask (Tensor): Possible mask,
                shape(batch_size, num_gt, num_priors)
            alignment_metrics (Tensor): Alignment metrics,
                shape(batch_size, num_gt, num_priors)
            overlaps (Tensor): Overlaps of gt_bboxes and pred_bboxes,
                shape(batch_size, num_gt, num_priors)
        """
        alignment_metrics, overlaps = self.get_box_metrics(pred_bboxes, pred_scores, gt_labels, gt_bboxes, batch_size, num_gt)
        is_in_gts = select_candidates_in_gts(priors, gt_bboxes)
        topk_metric = self.select_topk_candidates(alignment_metrics * is_in_gts, topk_mask=pad_bbox_flag.repeat([1, 1, self.topk]).bool())
        pos_mask = topk_metric * is_in_gts * pad_bbox_flag
        return pos_mask, alignment_metrics, overlaps

    def get_box_metrics(self, pred_bboxes: 'Tensor', pred_scores: 'Tensor', gt_labels: 'Tensor', gt_bboxes: 'Tensor', batch_size: 'int', num_gt: 'int') ->Tuple[Tensor, Tensor]:
        """Compute alignment metric between all bbox and gt.

        Args:
            pred_bboxes (Tensor): Predict bboxes,
                shape(batch_size, num_priors, 4)
            pred_scores (Tensor): Scores of predict bbox,
                shape(batch_size, num_priors, num_classes)
            gt_labels (Tensor): Ground true labels,
                shape(batch_size, num_gt, 1)
            gt_bboxes (Tensor): Ground true bboxes,
                shape(batch_size, num_gt, 4)
            batch_size (int): Batch size.
            num_gt (int): Number of ground truth.
        Returns:
            alignment_metrics (Tensor): Align metric,
                shape(batch_size, num_gt, num_priors)
            overlaps (Tensor): Overlaps, shape(batch_size, num_gt, num_priors)
        """
        pred_scores = pred_scores.permute(0, 2, 1)
        gt_labels = gt_labels
        idx = torch.zeros([2, batch_size, num_gt], dtype=torch.long)
        idx[0] = torch.arange(end=batch_size).view(-1, 1).repeat(1, num_gt)
        idx[1] = gt_labels.squeeze(-1)
        bbox_scores = pred_scores[idx[0], idx[1]]
        if self.use_ciou:
            overlaps = bbox_overlaps(pred_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(2), iou_mode='ciou', bbox_format='xyxy').clamp(0)
        else:
            overlaps = yolov6_iou_calculator(gt_bboxes, pred_bboxes)
        alignment_metrics = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return alignment_metrics, overlaps

    def select_topk_candidates(self, alignment_gt_metrics: 'Tensor', using_largest_topk: 'bool'=True, topk_mask: 'Optional[Tensor]'=None) ->Tensor:
        """Compute alignment metric between all bbox and gt.

        Args:
            alignment_gt_metrics (Tensor): Alignment metric of gt candidates,
                shape(batch_size, num_gt, num_priors)
            using_largest_topk (bool): Controls whether to using largest or
                smallest elements.
            topk_mask (Tensor): Topk mask,
                shape(batch_size, num_gt, self.topk)
        Returns:
            Tensor: Topk candidates mask,
                shape(batch_size, num_gt, num_priors)
        """
        num_priors = alignment_gt_metrics.shape[-1]
        topk_metrics, topk_idxs = torch.topk(alignment_gt_metrics, self.topk, axis=-1, largest=using_largest_topk)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(axis=-1, keepdim=True) > self.eps).tile([1, 1, self.topk])
        topk_idxs = torch.where(topk_mask, topk_idxs, torch.zeros_like(topk_idxs))
        is_in_topk = F.one_hot(topk_idxs, num_priors).sum(axis=-2)
        is_in_topk = torch.where(is_in_topk > 1, torch.zeros_like(is_in_topk), is_in_topk)
        return is_in_topk

    def get_targets(self, gt_labels: 'Tensor', gt_bboxes: 'Tensor', assigned_gt_idxs: 'Tensor', fg_mask_pre_prior: 'Tensor', batch_size: 'int', num_gt: 'int') ->Tuple[Tensor, Tensor, Tensor]:
        """Get assigner info.

        Args:
            gt_labels (Tensor): Ground true labels,
                shape(batch_size, num_gt, 1)
            gt_bboxes (Tensor): Ground true bboxes,
                shape(batch_size, num_gt, 4)
            assigned_gt_idxs (Tensor): Assigned ground truth indexes,
                shape(batch_size, num_priors)
            fg_mask_pre_prior (Tensor): Force ground truth matching mask,
                shape(batch_size, num_priors)
            batch_size (int): Batch size.
            num_gt (int): Number of ground truth.
        Returns:
            assigned_labels (Tensor): Assigned labels,
                shape(batch_size, num_priors)
            assigned_bboxes (Tensor): Assigned bboxes,
                shape(batch_size, num_priors)
            assigned_scores (Tensor): Assigned scores,
                shape(batch_size, num_priors)
        """
        batch_ind = torch.arange(end=batch_size, dtype=torch.int64, device=gt_labels.device)[..., None]
        assigned_gt_idxs = assigned_gt_idxs + batch_ind * num_gt
        assigned_labels = gt_labels.long().flatten()[assigned_gt_idxs]
        assigned_bboxes = gt_bboxes.reshape([-1, 4])[assigned_gt_idxs]
        assigned_labels[assigned_labels < 0] = 0
        assigned_scores = F.one_hot(assigned_labels, self.num_classes)
        force_gt_scores_mask = fg_mask_pre_prior[:, :, None].repeat(1, 1, self.num_classes)
        assigned_scores = torch.where(force_gt_scores_mask > 0, assigned_scores, torch.full_like(assigned_scores, 0))
        return assigned_labels, assigned_bboxes, assigned_scores


def _cat_multi_level_tensor_in_place(*multi_level_tensor, place_hold_var):
    """concat multi-level tensor in place."""
    for level_tensor in multi_level_tensor:
        for i, var in enumerate(level_tensor):
            if len(var) > 0:
                level_tensor[i] = torch.cat(var, dim=0)
            else:
                level_tensor[i] = place_hold_var


class BatchYOLOv7Assigner(nn.Module):
    """Batch YOLOv7 Assigner.

    It consists of two assigning steps:

        1. YOLOv5 cross-grid sample assigning
        2. SimOTA assigning

    This code referenced to
    https://github.com/WongKinYiu/yolov7/blob/main/utils/loss.py.

    Args:
        num_classes (int): Number of classes.
        num_base_priors (int): Number of base priors.
        featmap_strides (Sequence[int]): Feature map strides.
        prior_match_thr (float): Threshold to match priors.
            Defaults to 4.0.
        candidate_topk (int): Number of topk candidates to
            assign. Defaults to 10.
        iou_weight (float): IOU weight. Defaults to 3.0.
        cls_weight (float): Class weight. Defaults to 1.0.
    """

    def __init__(self, num_classes: 'int', num_base_priors: 'int', featmap_strides: 'Sequence[int]', prior_match_thr: 'float'=4.0, candidate_topk: 'int'=10, iou_weight: 'float'=3.0, cls_weight: 'float'=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_base_priors = num_base_priors
        self.featmap_strides = featmap_strides
        self.prior_match_thr = prior_match_thr
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight

    @torch.no_grad()
    def forward(self, pred_results, batch_targets_normed, batch_input_shape, priors_base_sizes, grid_offset, near_neighbor_thr=0.5) ->dict:
        """Forward function."""
        if batch_targets_normed.shape[1] == 0:
            num_levels = len(pred_results)
            return dict(mlvl_positive_infos=[pred_results[0].new_empty((0, 4))] * num_levels, mlvl_priors=[] * num_levels, mlvl_targets_normed=[] * num_levels)
        mlvl_positive_infos, mlvl_priors = self.yolov5_assigner(pred_results, batch_targets_normed, priors_base_sizes, grid_offset, near_neighbor_thr=near_neighbor_thr)
        mlvl_positive_infos, mlvl_priors, mlvl_targets_normed = self.simota_assigner(pred_results, batch_targets_normed, mlvl_positive_infos, mlvl_priors, batch_input_shape)
        place_hold_var = batch_targets_normed.new_empty((0, 4))
        _cat_multi_level_tensor_in_place(mlvl_positive_infos, mlvl_priors, mlvl_targets_normed, place_hold_var=place_hold_var)
        return dict(mlvl_positive_infos=mlvl_positive_infos, mlvl_priors=mlvl_priors, mlvl_targets_normed=mlvl_targets_normed)

    def yolov5_assigner(self, pred_results, batch_targets_normed, priors_base_sizes, grid_offset, near_neighbor_thr=0.5):
        """YOLOv5 cross-grid sample assigner."""
        num_batch_gts = batch_targets_normed.shape[1]
        assert num_batch_gts > 0
        mlvl_positive_infos, mlvl_priors = [], []
        scaled_factor = torch.ones(7, device=pred_results[0].device)
        for i in range(len(pred_results)):
            priors_base_sizes_i = priors_base_sizes[i]
            scaled_factor[2:6] = torch.tensor(pred_results[i].shape)[[3, 2, 3, 2]]
            batch_targets_scaled = batch_targets_normed * scaled_factor
            wh_ratio = batch_targets_scaled[..., 4:6] / priors_base_sizes_i[:, None]
            match_inds = torch.max(wh_ratio, 1.0 / wh_ratio).max(2)[0] < self.prior_match_thr
            batch_targets_scaled = batch_targets_scaled[match_inds]
            if batch_targets_scaled.shape[0] == 0:
                mlvl_positive_infos.append(batch_targets_scaled.new_empty((0, 4)))
                mlvl_priors.append([])
                continue
            batch_targets_cxcy = batch_targets_scaled[:, 2:4]
            grid_xy = scaled_factor[[2, 3]] - batch_targets_cxcy
            left, up = ((batch_targets_cxcy % 1 < near_neighbor_thr) & (batch_targets_cxcy > 1)).T
            right, bottom = ((grid_xy % 1 < near_neighbor_thr) & (grid_xy > 1)).T
            offset_inds = torch.stack((torch.ones_like(left), left, up, right, bottom))
            batch_targets_scaled = batch_targets_scaled.repeat((5, 1, 1))[offset_inds]
            retained_offsets = grid_offset.repeat(1, offset_inds.shape[1], 1)[offset_inds]
            mlvl_positive_info = batch_targets_scaled[:, [0, 6, 2, 3]]
            retained_offsets = retained_offsets * near_neighbor_thr
            mlvl_positive_info[:, 2:] = mlvl_positive_info[:, 2:] - retained_offsets
            mlvl_positive_info[:, 2].clamp_(0, scaled_factor[2] - 1)
            mlvl_positive_info[:, 3].clamp_(0, scaled_factor[3] - 1)
            mlvl_positive_info = mlvl_positive_info.long()
            priors_inds = mlvl_positive_info[:, 1]
            mlvl_positive_infos.append(mlvl_positive_info)
            mlvl_priors.append(priors_base_sizes_i[priors_inds])
        return mlvl_positive_infos, mlvl_priors

    def simota_assigner(self, pred_results, batch_targets_normed, mlvl_positive_infos, mlvl_priors, batch_input_shape):
        """SimOTA assigner."""
        num_batch_gts = batch_targets_normed.shape[1]
        assert num_batch_gts > 0
        num_levels = len(mlvl_positive_infos)
        mlvl_positive_infos_matched = [[] for _ in range(num_levels)]
        mlvl_priors_matched = [[] for _ in range(num_levels)]
        mlvl_targets_normed_matched = [[] for _ in range(num_levels)]
        for batch_idx in range(pred_results[0].shape[0]):
            targets_normed = batch_targets_normed[0]
            targets_normed = targets_normed[targets_normed[:, 0] == batch_idx]
            num_gts = targets_normed.shape[0]
            if num_gts == 0:
                continue
            _mlvl_decoderd_bboxes = []
            _mlvl_obj_cls = []
            _mlvl_priors = []
            _mlvl_positive_infos = []
            _from_which_layer = []
            for i, head_pred in enumerate(pred_results):
                _mlvl_positive_info = mlvl_positive_infos[i]
                if _mlvl_positive_info.shape[0] == 0:
                    continue
                idx = _mlvl_positive_info[:, 0] == batch_idx
                _mlvl_positive_info = _mlvl_positive_info[idx]
                _mlvl_positive_infos.append(_mlvl_positive_info)
                priors = mlvl_priors[i][idx]
                _mlvl_priors.append(priors)
                _from_which_layer.append(_mlvl_positive_info.new_full(size=(_mlvl_positive_info.shape[0],), fill_value=i))
                level_batch_idx, prior_ind, grid_x, grid_y = _mlvl_positive_info.T
                pred_positive = head_pred[level_batch_idx, prior_ind, grid_y, grid_x]
                _mlvl_obj_cls.append(pred_positive[:, 4:])
                grid = torch.stack([grid_x, grid_y], dim=1)
                pred_positive_cxcy = (pred_positive[:, :2].sigmoid() * 2.0 - 0.5 + grid) * self.featmap_strides[i]
                pred_positive_wh = (pred_positive[:, 2:4].sigmoid() * 2) ** 2 * priors * self.featmap_strides[i]
                pred_positive_xywh = torch.cat([pred_positive_cxcy, pred_positive_wh], dim=-1)
                _mlvl_decoderd_bboxes.append(pred_positive_xywh)
            if len(_mlvl_decoderd_bboxes) == 0:
                continue
            _mlvl_decoderd_bboxes = torch.cat(_mlvl_decoderd_bboxes, dim=0)
            num_pred_positive = _mlvl_decoderd_bboxes.shape[0]
            if num_pred_positive == 0:
                continue
            batch_input_shape_wh = pred_results[0].new_tensor(batch_input_shape[::-1]).repeat((1, 2))
            targets_scaled_bbox = targets_normed[:, 2:6] * batch_input_shape_wh
            targets_scaled_bbox = bbox_cxcywh_to_xyxy(targets_scaled_bbox)
            _mlvl_decoderd_bboxes = bbox_cxcywh_to_xyxy(_mlvl_decoderd_bboxes)
            pair_wise_iou = bbox_overlaps(targets_scaled_bbox, _mlvl_decoderd_bboxes)
            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-08)
            _mlvl_obj_cls = torch.cat(_mlvl_obj_cls, dim=0).float().sigmoid()
            _mlvl_positive_infos = torch.cat(_mlvl_positive_infos, dim=0)
            _from_which_layer = torch.cat(_from_which_layer, dim=0)
            _mlvl_priors = torch.cat(_mlvl_priors, dim=0)
            gt_cls_per_image = F.one_hot(targets_normed[:, 1], self.num_classes).float().unsqueeze(1).repeat(1, num_pred_positive, 1)
            cls_preds_ = _mlvl_obj_cls[:, 1:].unsqueeze(0).repeat(num_gts, 1, 1) * _mlvl_obj_cls[:, 0:1].unsqueeze(0).repeat(num_gts, 1, 1)
            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(torch.log(y / (1 - y)), gt_cls_per_image, reduction='none').sum(-1)
            del cls_preds_
            cost = self.cls_weight * pair_wise_cls_loss + self.iou_weight * pair_wise_iou_loss
            matching_matrix = torch.zeros_like(cost)
            top_k, _ = torch.topk(pair_wise_iou, min(self.candidate_topk, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)
            for gt_idx in range(num_gts):
                _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
                matching_matrix[gt_idx][pos_idx] = 1.0
            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
            targets_normed = targets_normed[matched_gt_inds]
            _mlvl_positive_infos = _mlvl_positive_infos[fg_mask_inboxes]
            _from_which_layer = _from_which_layer[fg_mask_inboxes]
            _mlvl_priors = _mlvl_priors[fg_mask_inboxes]
            for i in range(num_levels):
                layer_idx = _from_which_layer == i
                mlvl_positive_infos_matched[i].append(_mlvl_positive_infos[layer_idx])
                mlvl_priors_matched[i].append(_mlvl_priors[layer_idx])
                mlvl_targets_normed_matched[i].append(targets_normed[layer_idx])
        results = mlvl_positive_infos_matched, mlvl_priors_matched, mlvl_targets_normed_matched
        return results


def init_detector(config: 'Union[str, Path, Config]', checkpoint: 'Optional[str]'=None, palette: 'str'='coco', device: 'str'='cuda:0', cfg_options: 'Optional[dict]'=None) ->nn.Module:
    """Initialize a detector from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        palette (str): Color palette used for visualization. If palette
            is stored in checkpoint, use checkpoint's palette first, otherwise
            use externally passed palette. Currently, supports 'coco', 'voc',
            'citys' and 'random'. Defaults to coco.
        device (str): The device where the anchors will be put on.
            Defaults to cuda:0.
        cfg_options (dict, optional): Options to override some settings in
            the used config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError(f'config must be a filename or Config object, but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None
    init_default_scope(config.get('default_scope', 'mmyolo'))
    model = MODELS.build(config.model)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        checkpoint_meta = checkpoint.get('meta', {})
        if 'dataset_meta' in checkpoint_meta:
            model.dataset_meta = {k.lower(): v for k, v in checkpoint_meta['dataset_meta'].items()}
        elif 'CLASSES' in checkpoint_meta:
            classes = checkpoint_meta['CLASSES']
            model.dataset_meta = {'classes': classes, 'palette': palette}
        else:
            warnings.simplefilter('once')
            warnings.warn("dataset_meta or class names are not saved in the checkpoint's meta data, use COCO classes by default.")
            model.dataset_meta = {'classes': get_classes('coco'), 'palette': palette}
    model.cfg = config
    model
    model.eval()
    return model


class BoxAMDetectorWrapper(nn.Module):
    """Wrap the mmdet model class to facilitate handling of non-tensor
    situations during inference."""

    def __init__(self, cfg: 'ConfigType', checkpoint: 'str', score_thr: 'float', device: 'str'='cuda:0'):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.score_thr = score_thr
        self.checkpoint = checkpoint
        self.detector = init_detector(self.cfg, self.checkpoint, device=device)
        pipeline_cfg = copy.deepcopy(self.cfg.test_dataloader.dataset.pipeline)
        pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
        new_test_pipeline = []
        for pipeline in pipeline_cfg:
            if not pipeline['type'].endswith('LoadAnnotations'):
                new_test_pipeline.append(pipeline)
        self.test_pipeline = Compose(new_test_pipeline)
        self.is_need_loss = False
        self.input_data = None
        self.image = None

    def need_loss(self, is_need_loss: 'bool'):
        """Grad-based methods require loss."""
        self.is_need_loss = is_need_loss

    def set_input_data(self, image: 'np.ndarray', pred_instances: 'Optional[InstanceData]'=None):
        """Set the input data to be used in the next step."""
        self.image = image
        if self.is_need_loss:
            assert pred_instances is not None
            pred_instances = pred_instances.numpy()
            data = dict(img=self.image, img_id=0, gt_bboxes=pred_instances.bboxes, gt_bboxes_labels=pred_instances.labels)
            data = self.test_pipeline(data)
        else:
            data = dict(img=self.image, img_id=0)
            data = self.test_pipeline(data)
            data['inputs'] = [data['inputs']]
            data['data_samples'] = [data['data_samples']]
        self.input_data = data

    def __call__(self, *args, **kwargs):
        assert self.input_data is not None
        if self.is_need_loss:
            if hasattr(self.detector.bbox_head, 'head_module'):
                self.detector.bbox_head.head_module.training = True
            else:
                self.detector.bbox_head.training = True
            if hasattr(self.detector.bbox_head, 'featmap_sizes'):
                self.detector.bbox_head.featmap_sizes = None
            data_ = {}
            data_['inputs'] = [self.input_data['inputs']]
            data_['data_samples'] = [self.input_data['data_samples']]
            data = self.detector.data_preprocessor(data_, training=False)
            loss = self.detector._run_forward(data, mode='loss')
            if hasattr(self.detector.bbox_head, 'featmap_sizes'):
                self.detector.bbox_head.featmap_sizes = None
            return [loss]
        else:
            if hasattr(self.detector.bbox_head, 'head_module'):
                self.detector.bbox_head.head_module.training = False
            else:
                self.detector.bbox_head.training = False
            with torch.no_grad():
                results = self.detector.test_step(self.input_data)
                return results


class DeployC2f(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: 'Tensor') ->Tensor:
        x_main = self.main_conv(x)
        x_main = [x_main, x_main[:, self.mid_channels:, ...]]
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        x_main.pop(1)
        return self.final_conv(torch.cat(x_main, 1))


class DeployFocus(nn.Module):

    def __init__(self, orin_Focus: 'nn.Module'):
        super().__init__()
        self.__dict__.update(orin_Focus.__dict__)

    def forward(self, x: 'Tensor') ->Tensor:
        batch_size, channel, height, width = x.shape
        x = x.reshape(batch_size, channel, -1, 2, width)
        x = x.reshape(batch_size, channel, x.shape[2], 2, -1, 2)
        half_h = x.shape[2]
        half_w = x.shape[4]
        x = x.permute(0, 5, 3, 1, 2, 4)
        x = x.reshape(batch_size, channel * 4, half_h, half_w)
        return self.conv(x)


class NcnnFocus(nn.Module):

    def __init__(self, orin_Focus: 'nn.Module'):
        super().__init__()
        self.__dict__.update(orin_Focus.__dict__)

    def forward(self, x: 'Tensor') ->Tensor:
        batch_size, c, h, w = x.shape
        assert h % 2 == 0 and w % 2 == 0, f'focus for yolox needs even feature            height and width, got {h, w}.'
        x = x.reshape(batch_size, c * h, 1, w)
        _b, _c, _h, _w = x.shape
        g = _c // 2
        x = x.view(_b, g, 2, _h, _w)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(_b, -1, _h, _w)
        x = x.reshape(_b, c * h * w, 1, 1)
        _b, _c, _h, _w = x.shape
        g = _c // 2
        x = x.view(_b, g, 2, _h, _w)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(_b, -1, _h, _w)
        x = x.reshape(_b, c * 4, h // 2, w // 2)
        return self.conv(x)


class GConvFocus(nn.Module):

    def __init__(self, orin_Focus: 'nn.Module'):
        super().__init__()
        device = next(orin_Focus.parameters()).device
        self.weight1 = torch.tensor([[1.0, 0], [0, 0]]).expand(3, 1, 2, 2)
        self.weight2 = torch.tensor([[0, 0], [1.0, 0]]).expand(3, 1, 2, 2)
        self.weight3 = torch.tensor([[0, 1.0], [0, 0]]).expand(3, 1, 2, 2)
        self.weight4 = torch.tensor([[0, 0], [0, 1.0]]).expand(3, 1, 2, 2)
        self.__dict__.update(orin_Focus.__dict__)

    def forward(self, x: 'Tensor') ->Tensor:
        conv1 = F.conv2d(x, self.weight1, stride=2, groups=3)
        conv2 = F.conv2d(x, self.weight2, stride=2, groups=3)
        conv3 = F.conv2d(x, self.weight3, stride=2, groups=3)
        conv4 = F.conv2d(x, self.weight4, stride=2, groups=3)
        return self.conv(torch.cat([conv1, conv2, conv3, conv4], dim=1))


class TRTWrapper(torch.nn.Module):
    dtype_mapping = {}

    def __init__(self, weight: 'Union[str, Path]', device: 'Optional[torch.device]'):
        super().__init__()
        weight = Path(weight) if isinstance(weight, str) else weight
        assert weight.exists() and weight.suffix in ('.engine', '.plan')
        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device(f'cuda:{device}')
        self.weight = weight
        self.device = device
        self.stream = torch.Stream(device=device)
        self.__update_mapping()
        self.__init_engine()
        self.__init_bindings()

    def __update_mapping(self):
        self.dtype_mapping.update({trt.bool: torch.bool, trt.int8: torch.int8, trt.int32: torch.int32, trt.float16: torch.float16, trt.float32: torch.float32})

    def __init_engine(self):
        logger = trt.Logger(trt.Logger.ERROR)
        self.log = partial(logger.log, trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, namespace='')
        self.logger = logger
        with trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(self.weight.read_bytes())
        context = model.create_execution_context()
        names = [model.get_binding_name(i) for i in range(model.num_bindings)]
        num_inputs, num_outputs = 0, 0
        for i in range(model.num_bindings):
            if model.binding_is_input(i):
                num_inputs += 1
            else:
                num_outputs += 1
        self.is_dynamic = -1 in model.get_binding_shape(0)
        self.model = model
        self.context = context
        self.input_names = names[:num_inputs]
        self.output_names = names[num_inputs:]
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_bindings = num_inputs + num_outputs
        self.bindings: 'List[int]' = [0] * self.num_bindings

    def __init_bindings(self):
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape'))
        inputs_info = []
        outputs_info = []
        for i, name in enumerate(self.input_names):
            assert self.model.get_binding_name(i) == name
            dtype = self.dtype_mapping[self.model.get_binding_dtype(i)]
            shape = tuple(self.model.get_binding_shape(i))
            inputs_info.append(Binding(name, dtype, shape))
        for i, name in enumerate(self.output_names):
            i += self.num_inputs
            assert self.model.get_binding_name(i) == name
            dtype = self.dtype_mapping[self.model.get_binding_dtype(i)]
            shape = tuple(self.model.get_binding_shape(i))
            outputs_info.append(Binding(name, dtype, shape))
        self.inputs_info = inputs_info
        self.outputs_info = outputs_info
        if not self.is_dynamic:
            self.output_tensor = [torch.empty(o.shape, dtype=o.dtype, device=self.device) for o in outputs_info]

    def forward(self, *inputs):
        assert len(inputs) == self.num_inputs
        contiguous_inputs: 'List[torch.Tensor]' = [i.contiguous() for i in inputs]
        for i in range(self.num_inputs):
            self.bindings[i] = contiguous_inputs[i].data_ptr()
            if self.is_dynamic:
                self.context.set_binding_shape(i, tuple(contiguous_inputs[i].shape))
        outputs: 'List[torch.Tensor]' = []
        for i in range(self.num_outputs):
            j = i + self.num_inputs
            if self.is_dynamic:
                shape = tuple(self.context.get_binding_shape(j))
                output = torch.empty(size=shape, dtype=self.output_dtypes[i], device=self.device)
            else:
                output = self.output_tensor[i]
            outputs.append(output)
            self.bindings[j] = output.data_ptr()
        self.context.execute_async_v2(self.bindings, self.stream.cuda_stream)
        self.stream.synchronize()
        return tuple(outputs)


class ORTWrapper(torch.nn.Module):

    def __init__(self, weight: 'Union[str, Path]', device: 'Optional[torch.device]'):
        super().__init__()
        weight = Path(weight) if isinstance(weight, str) else weight
        assert weight.exists() and weight.suffix == '.onnx'
        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device(f'cuda:{device}')
        self.weight = weight
        self.device = device
        self.__init_session()
        self.__init_bindings()

    def __init_session(self):
        providers = ['CPUExecutionProvider']
        if 'cuda' in self.device.type:
            providers.insert(0, 'CUDAExecutionProvider')
        session = onnxruntime.InferenceSession(str(self.weight), providers=providers)
        self.session = session

    def __init_bindings(self):
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape'))
        inputs_info = []
        outputs_info = []
        self.is_dynamic = False
        for i, tensor in enumerate(self.session.get_inputs()):
            if any(not isinstance(i, int) for i in tensor.shape):
                self.is_dynamic = True
            inputs_info.append(Binding(tensor.name, tensor.type, tuple(tensor.shape)))
        for i, tensor in enumerate(self.session.get_outputs()):
            outputs_info.append(Binding(tensor.name, tensor.type, tuple(tensor.shape)))
        self.inputs_info = inputs_info
        self.outputs_info = outputs_info
        self.num_inputs = len(inputs_info)

    def forward(self, *inputs):
        assert len(inputs) == self.num_inputs
        contiguous_inputs: 'List[np.ndarray]' = [i.contiguous().cpu().numpy() for i in inputs]
        if not self.is_dynamic:
            for i in range(self.num_inputs):
                assert contiguous_inputs[i].shape == self.inputs_info[i].shape
        outputs = self.session.run([o.name for o in self.outputs_info], {j.name: contiguous_inputs[i] for i, j in enumerate(self.inputs_info)})
        return tuple(torch.from_numpy(o) for o in outputs)


class MMYOLOBackend(Enum):
    AX620A = 'ax620a'
    COREML = 'coreml'
    HORIZONX3 = 'horizonx3'
    NCNN = 'ncnn'
    ONNXRUNTIME = 'onnxruntime'
    OPENVINO = 'openvino'
    PPLNN = 'pplnn'
    RKNN = 'rknn'
    TENSORRT8 = 'tensorrt8'
    TENSORRT7 = 'tensorrt7'
    TORCHSCRIPT = 'torchscript'
    TVM = 'tvm'


def get_prior_xy_info(index: 'int', num_base_priors: 'int', featmap_sizes: 'int') ->Tuple[int, int, int]:
    """Get prior index and xy index in feature map by flatten index."""
    _, featmap_w = featmap_sizes
    priors = index % num_base_priors
    xy_index = index // num_base_priors
    grid_y = xy_index // featmap_w
    grid_x = xy_index % featmap_w
    return priors, grid_x, grid_y


def gt_instances_preprocess(batch_gt_instances: 'Union[Tensor, Sequence]', batch_size: 'int') ->Tensor:
    """Split batch_gt_instances with batch size.

    From [all_gt_bboxes, box_dim+2] to [batch_size, number_gt, box_dim+1].
    For horizontal box, box_dim=4, for rotated box, box_dim=5

    If some shape of single batch smaller than
    gt bbox len, then using zeros to fill.

    Args:
        batch_gt_instances (Sequence[Tensor]): Ground truth
            instances for whole batch, shape [all_gt_bboxes, box_dim+2]
        batch_size (int): Batch size.

    Returns:
        Tensor: batch gt instances data, shape
                [batch_size, number_gt, box_dim+1]
    """
    if isinstance(batch_gt_instances, Sequence):
        max_gt_bbox_len = max([len(gt_instances) for gt_instances in batch_gt_instances])
        batch_instance_list = []
        for index, gt_instance in enumerate(batch_gt_instances):
            bboxes = gt_instance.bboxes
            labels = gt_instance.labels
            box_dim = get_box_tensor(bboxes).size(-1)
            batch_instance_list.append(torch.cat((labels[:, None], bboxes), dim=-1))
            if bboxes.shape[0] >= max_gt_bbox_len:
                continue
            fill_tensor = bboxes.new_full([max_gt_bbox_len - bboxes.shape[0], box_dim + 1], 0)
            batch_instance_list[index] = torch.cat((batch_instance_list[index], fill_tensor), dim=0)
        return torch.stack(batch_instance_list)
    else:
        assert isinstance(batch_gt_instances, Tensor)
        box_dim = batch_gt_instances.size(-1) - 2
        if len(batch_gt_instances) > 0:
            gt_images_indexes = batch_gt_instances[:, 0]
            max_gt_bbox_len = gt_images_indexes.unique(return_counts=True)[1].max()
            batch_instance = torch.zeros((batch_size, max_gt_bbox_len, box_dim + 1), dtype=batch_gt_instances.dtype, device=batch_gt_instances.device)
            for i in range(batch_size):
                match_indexes = gt_images_indexes == i
                gt_num = match_indexes.sum()
                if gt_num:
                    batch_instance[i, :gt_num] = batch_gt_instances[match_indexes, 1:]
        else:
            batch_instance = torch.zeros((batch_size, 0, box_dim + 1), dtype=batch_gt_instances.dtype, device=batch_gt_instances.device)
        return batch_instance


class TRTbatchedNMSop(torch.autograd.Function):
    """TensorRT NMS operation."""

    @staticmethod
    def forward(ctx, boxes: 'Tensor', scores: 'Tensor', plugin_version: 'str'='1', shareLocation: 'int'=1, backgroundLabelId: 'int'=-1, numClasses: 'int'=80, topK: 'int'=1000, keepTopK: 'int'=100, scoreThreshold: 'float'=0.25, iouThreshold: 'float'=0.45, isNormalized: 'int'=0, clipBoxes: 'int'=0, scoreBits: 'int'=16, caffeSemantics: 'int'=1):
        batch_size, _, numClasses = scores.shape
        num_det = torch.randint(0, keepTopK, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, keepTopK, 4)
        det_scores = torch.randn(batch_size, keepTopK)
        det_classes = torch.randint(0, numClasses, (batch_size, keepTopK)).float()
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(g, boxes: 'Tensor', scores: 'Tensor', plugin_version: 'str'='1', shareLocation: 'int'=1, backgroundLabelId: 'int'=-1, numClasses: 'int'=80, topK: 'int'=1000, keepTopK: 'int'=100, scoreThreshold: 'float'=0.25, iouThreshold: 'float'=0.45, isNormalized: 'int'=0, clipBoxes: 'int'=0, scoreBits: 'int'=16, caffeSemantics: 'int'=1):
        out = g.op('TRT::BatchedNMSDynamic_TRT', boxes, scores, shareLocation_i=shareLocation, plugin_version_s=plugin_version, backgroundLabelId_i=backgroundLabelId, numClasses_i=numClasses, topK_i=topK, keepTopK_i=keepTopK, scoreThreshold_f=scoreThreshold, iouThreshold_f=iouThreshold, isNormalized_i=isNormalized, clipBoxes_i=clipBoxes, scoreBits_i=scoreBits, caffeSemantics_i=caffeSemantics, outputs=4)
        num_det, det_boxes, det_scores, det_classes = out
        return num_det, det_boxes, det_scores, det_classes


_XYWH2XYXY = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [-0.5, 0.0, 0.5, 0.0], [0.0, -0.5, 0.0, 0.5]], dtype=torch.float32)


def _batched_nms(boxes: 'Tensor', scores: 'Tensor', max_output_boxes_per_class: 'int'=1000, iou_threshold: 'float'=0.5, score_threshold: 'float'=0.05, pre_top_k: 'int'=-1, keep_top_k: 'int'=100, box_coding: 'int'=0):
    """Wrapper for `efficient_nms` with TensorRT.
    Args:
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
        scores (Tensor): The detection scores of shape
            [N, num_boxes, num_classes].
        max_output_boxes_per_class (int): Maximum number of output
            boxes per class of nms. Defaults to 1000.
        iou_threshold (float): IOU threshold of nms. Defaults to 0.5.
        score_threshold (float): score threshold of nms.
            Defaults to 0.05.
        pre_top_k (int): Number of top K boxes to keep before nms.
            Defaults to -1.
        keep_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.
        box_coding (int): Bounding boxes format for nms.
            Defaults to 0 means [x1, y1 ,x2, y2].
            Set to 1 means [x, y, w, h].
    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]:
        (num_det, det_boxes, det_scores, det_classes),
        `num_det` of shape [N, 1]
        `det_boxes` of shape [N, num_det, 4]
        `det_scores` of shape [N, num_det]
        `det_classes` of shape [N, num_det]
    """
    if box_coding == 1:
        boxes = boxes @ _XYWH2XYXY
    boxes = boxes if boxes.dim() == 4 else boxes.unsqueeze(2)
    _, _, numClasses = scores.shape
    num_det, det_boxes, det_scores, det_classes = TRTbatchedNMSop.apply(boxes, scores, '1', 1, -1, int(numClasses), min(pre_top_k, 4096), keep_top_k, score_threshold, iou_threshold, 0, 0, 16, 1)
    det_classes = det_classes.int()
    return num_det, det_boxes, det_scores, det_classes


def batched_nms(*args, **kwargs):
    """Wrapper function for `_batched_nms`."""
    return _batched_nms(*args, **kwargs)


class TRTEfficientNMSop(torch.autograd.Function):

    @staticmethod
    def forward(ctx, boxes: 'Tensor', scores: 'Tensor', background_class: 'int'=-1, box_coding: 'int'=0, iou_threshold: 'float'=0.45, max_output_boxes: 'int'=100, plugin_version: 'str'='1', score_activation: 'int'=0, score_threshold: 'float'=0.25):
        batch_size, _, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(g, boxes: 'Tensor', scores: 'Tensor', background_class: 'int'=-1, box_coding: 'int'=0, iou_threshold: 'float'=0.45, max_output_boxes: 'int'=100, plugin_version: 'str'='1', score_activation: 'int'=0, score_threshold: 'float'=0.25):
        out = g.op('TRT::EfficientNMS_TRT', boxes, scores, background_class_i=background_class, box_coding_i=box_coding, iou_threshold_f=iou_threshold, max_output_boxes_i=max_output_boxes, plugin_version_s=plugin_version, score_activation_i=score_activation, score_threshold_f=score_threshold, outputs=4)
        num_det, det_boxes, det_scores, det_classes = out
        return num_det, det_boxes, det_scores, det_classes


def _efficient_nms(boxes: 'Tensor', scores: 'Tensor', max_output_boxes_per_class: 'int'=1000, iou_threshold: 'float'=0.5, score_threshold: 'float'=0.05, pre_top_k: 'int'=-1, keep_top_k: 'int'=100, box_coding: 'int'=0):
    """Wrapper for `efficient_nms` with TensorRT.
    Args:
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
        scores (Tensor): The detection scores of shape
            [N, num_boxes, num_classes].
        max_output_boxes_per_class (int): Maximum number of output
            boxes per class of nms. Defaults to 1000.
        iou_threshold (float): IOU threshold of nms. Defaults to 0.5.
        score_threshold (float): score threshold of nms.
            Defaults to 0.05.
        pre_top_k (int): Number of top K boxes to keep before nms.
            Defaults to -1.
        keep_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.
        box_coding (int): Bounding boxes format for nms.
            Defaults to 0 means [x1, y1 ,x2, y2].
            Set to 1 means [x, y, w, h].
    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]:
        (num_det, det_boxes, det_scores, det_classes),
        `num_det` of shape [N, 1]
        `det_boxes` of shape [N, num_det, 4]
        `det_scores` of shape [N, num_det]
        `det_classes` of shape [N, num_det]
    """
    num_det, det_boxes, det_scores, det_classes = TRTEfficientNMSop.apply(boxes, scores, -1, box_coding, iou_threshold, keep_top_k, '1', 0, score_threshold)
    return num_det, det_boxes, det_scores, det_classes


def efficient_nms(*args, **kwargs):
    """Wrapper function for `_efficient_nms`."""
    return _efficient_nms(*args, **kwargs)


class ONNXNMSop(torch.autograd.Function):

    @staticmethod
    def forward(ctx, boxes: 'Tensor', scores: 'Tensor', max_output_boxes_per_class: 'Tensor'=torch.tensor([100]), iou_threshold: 'Tensor'=torch.tensor([0.5]), score_threshold: 'Tensor'=torch.tensor([0.05])) ->Tensor:
        device = boxes.device
        batch = scores.shape[0]
        num_det = 20
        batches = torch.randint(0, batch, (num_det,)).sort()[0]
        idxs = torch.arange(100, 100 + num_det)
        zeros = torch.zeros((num_det,), dtype=torch.int64)
        selected_indices = torch.cat([batches[None], zeros[None], idxs[None]], 0).T.contiguous()
        selected_indices = selected_indices
        return selected_indices

    @staticmethod
    def symbolic(g, boxes: 'Tensor', scores: 'Tensor', max_output_boxes_per_class: 'Tensor'=torch.tensor([100]), iou_threshold: 'Tensor'=torch.tensor([0.5]), score_threshold: 'Tensor'=torch.tensor([0.05])):
        return g.op('NonMaxSuppression', boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, outputs=1)


def select_nms_index(scores: 'Tensor', boxes: 'Tensor', nms_index: 'Tensor', batch_size: 'int', keep_top_k: 'int'=-1):
    batch_inds, cls_inds = nms_index[:, 0], nms_index[:, 1]
    box_inds = nms_index[:, 2]
    scores = scores[batch_inds, cls_inds, box_inds].unsqueeze(1)
    boxes = boxes[batch_inds, box_inds, ...]
    dets = torch.cat([boxes, scores], dim=1)
    batched_dets = dets.unsqueeze(0).repeat(batch_size, 1, 1)
    batch_template = torch.arange(0, batch_size, dtype=batch_inds.dtype, device=batch_inds.device)
    batched_dets = batched_dets.where((batch_inds == batch_template.unsqueeze(1)).unsqueeze(-1), batched_dets.new_zeros(1))
    batched_labels = cls_inds.unsqueeze(0).repeat(batch_size, 1)
    batched_labels = batched_labels.where(batch_inds == batch_template.unsqueeze(1), batched_labels.new_ones(1) * -1)
    N = batched_dets.shape[0]
    batched_dets = torch.cat((batched_dets, batched_dets.new_zeros((N, 1, 5))), 1)
    batched_labels = torch.cat((batched_labels, -batched_labels.new_ones((N, 1))), 1)
    _, topk_inds = batched_dets[:, :, -1].sort(dim=1, descending=True)
    topk_batch_inds = torch.arange(batch_size, dtype=topk_inds.dtype, device=topk_inds.device).view(-1, 1)
    batched_dets = batched_dets[topk_batch_inds, topk_inds, ...]
    batched_labels = batched_labels[topk_batch_inds, topk_inds, ...]
    batched_dets, batched_scores = batched_dets.split([4, 1], 2)
    batched_scores = batched_scores.squeeze(-1)
    num_dets = (batched_scores > 0).sum(1, keepdim=True)
    return num_dets, batched_dets, batched_scores, batched_labels


def onnx_nms(boxes: 'torch.Tensor', scores: 'torch.Tensor', max_output_boxes_per_class: 'int'=100, iou_threshold: 'float'=0.5, score_threshold: 'float'=0.05, pre_top_k: 'int'=-1, keep_top_k: 'int'=100, box_coding: 'int'=0):
    max_output_boxes_per_class = torch.tensor([max_output_boxes_per_class])
    iou_threshold = torch.tensor([iou_threshold])
    score_threshold = torch.tensor([score_threshold])
    batch_size, _, _ = scores.shape
    if box_coding == 1:
        boxes = boxes @ _XYWH2XYXY
    scores = scores.transpose(1, 2).contiguous()
    selected_indices = ONNXNMSop.apply(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
    num_dets, batched_dets, batched_scores, batched_labels = select_nms_index(scores, boxes, selected_indices, batch_size, keep_top_k=keep_top_k)
    return num_dets, batched_dets, batched_scores, batched_labels


def rtmdet_bbox_decoder(priors: 'Tensor', bbox_preds: 'Tensor', stride: 'Optional[Tensor]') ->Tensor:
    stride = stride[None, :, None]
    bbox_preds *= stride
    tl_x = priors[..., 0] - bbox_preds[..., 0]
    tl_y = priors[..., 1] - bbox_preds[..., 1]
    br_x = priors[..., 0] + bbox_preds[..., 2]
    br_y = priors[..., 1] + bbox_preds[..., 3]
    decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
    return decoded_bboxes


def yolov5_bbox_decoder(priors: 'Tensor', bbox_preds: 'Tensor', stride: 'Tensor') ->Tensor:
    bbox_preds = bbox_preds.sigmoid()
    x_center = (priors[..., 0] + priors[..., 2]) * 0.5
    y_center = (priors[..., 1] + priors[..., 3]) * 0.5
    w = priors[..., 2] - priors[..., 0]
    h = priors[..., 3] - priors[..., 1]
    x_center_pred = (bbox_preds[..., 0] - 0.5) * 2 * stride + x_center
    y_center_pred = (bbox_preds[..., 1] - 0.5) * 2 * stride + y_center
    w_pred = (bbox_preds[..., 2] * 2) ** 2 * w
    h_pred = (bbox_preds[..., 3] * 2) ** 2 * h
    decoded_bboxes = torch.stack([x_center_pred, y_center_pred, w_pred, h_pred], dim=-1)
    return decoded_bboxes


def yolox_bbox_decoder(priors: 'Tensor', bbox_preds: 'Tensor', stride: 'Optional[Tensor]') ->Tensor:
    stride = stride[None, :, None]
    xys = bbox_preds[..., :2] * stride + priors
    whs = bbox_preds[..., 2:].exp() * stride
    decoded_bboxes = torch.cat([xys, whs], -1)
    return decoded_bboxes


class DeployModel(nn.Module):
    transpose = False

    def __init__(self, baseModel: 'nn.Module', backend: 'MMYOLOBackend', postprocess_cfg: 'Optional[ConfigDict]'=None):
        super().__init__()
        self.baseModel = baseModel
        self.baseHead = baseModel.bbox_head
        self.backend = backend
        if postprocess_cfg is None:
            self.with_postprocess = False
        else:
            self.with_postprocess = True
            self.__init_sub_attributes()
            self.detector_type = type(self.baseHead)
            self.pre_top_k = postprocess_cfg.get('pre_top_k', 1000)
            self.keep_top_k = postprocess_cfg.get('keep_top_k', 100)
            self.iou_threshold = postprocess_cfg.get('iou_threshold', 0.65)
            self.score_threshold = postprocess_cfg.get('score_threshold', 0.25)
        self.__switch_deploy()

    def __init_sub_attributes(self):
        self.bbox_decoder = self.baseHead.bbox_coder.decode
        self.prior_generate = self.baseHead.prior_generator.grid_priors
        self.num_base_priors = self.baseHead.num_base_priors
        self.featmap_strides = self.baseHead.featmap_strides
        self.num_classes = self.baseHead.num_classes

    def __switch_deploy(self):
        headType = type(self.baseHead)
        if not self.with_postprocess:
            if headType in (YOLOv5Head, YOLOv7Head):
                self.baseHead.head_module.forward_single = self.forward_single
            elif headType in (PPYOLOEHead, YOLOv8Head):
                self.baseHead.head_module.reg_max = 0
        if self.backend in (MMYOLOBackend.HORIZONX3, MMYOLOBackend.NCNN, MMYOLOBackend.TORCHSCRIPT):
            self.transpose = True
        for layer in self.baseModel.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()
            elif isinstance(layer, ChannelAttention):
                layer.global_avgpool.forward = self.forward_gvp
            elif isinstance(layer, Focus):
                if self.backend in (MMYOLOBackend.ONNXRUNTIME, MMYOLOBackend.OPENVINO, MMYOLOBackend.TENSORRT8, MMYOLOBackend.TENSORRT7):
                    self.baseModel.backbone.stem = DeployFocus(layer)
                elif self.backend == MMYOLOBackend.NCNN:
                    self.baseModel.backbone.stem = NcnnFocus(layer)
                else:
                    self.baseModel.backbone.stem = GConvFocus(layer)

    def pred_by_feat(self, cls_scores: 'List[Tensor]', bbox_preds: 'List[Tensor]', objectnesses: 'Optional[List[Tensor]]'=None, **kwargs):
        assert len(cls_scores) == len(bbox_preds)
        dtype = cls_scores[0].dtype
        device = cls_scores[0].device
        nms_func = self.select_nms()
        if self.detector_type in (YOLOv5Head, YOLOv7Head):
            bbox_decoder = yolov5_bbox_decoder
        elif self.detector_type is RTMDetHead:
            bbox_decoder = rtmdet_bbox_decoder
        elif self.detector_type is YOLOXHead:
            bbox_decoder = yolox_bbox_decoder
        else:
            bbox_decoder = self.bbox_decoder
        num_imgs = cls_scores[0].shape[0]
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generate(featmap_sizes, dtype=dtype, device=device)
        flatten_priors = torch.cat(mlvl_priors)
        mlvl_strides = [flatten_priors.new_full((featmap_size[0] * featmap_size[1] * self.num_base_priors,), stride) for featmap_size, stride in zip(featmap_sizes, self.featmap_strides)]
        flatten_stride = torch.cat(mlvl_strides)
        flatten_cls_scores = [cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes) for cls_score in cls_scores]
        cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for bbox_pred in bbox_preds]
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        if objectnesses is not None:
            flatten_objectness = [objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1) for objectness in objectnesses]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
            cls_scores = cls_scores * flatten_objectness.unsqueeze(-1)
        scores = cls_scores
        bboxes = bbox_decoder(flatten_priors[None], flatten_bbox_preds, flatten_stride)
        return nms_func(bboxes, scores, self.keep_top_k, self.iou_threshold, self.score_threshold, self.pre_top_k, self.keep_top_k)

    def select_nms(self):
        if self.backend in (MMYOLOBackend.ONNXRUNTIME, MMYOLOBackend.OPENVINO):
            nms_func = onnx_nms
        elif self.backend == MMYOLOBackend.TENSORRT8:
            nms_func = efficient_nms
        elif self.backend == MMYOLOBackend.TENSORRT7:
            nms_func = batched_nms
        else:
            raise NotImplementedError
        if type(self.baseHead) in (YOLOv5Head, YOLOv7Head, YOLOXHead):
            nms_func = partial(nms_func, box_coding=1)
        return nms_func

    def forward(self, inputs: 'Tensor'):
        neck_outputs = self.baseModel(inputs)
        if self.with_postprocess:
            return self.pred_by_feat(*neck_outputs)
        else:
            outputs = []
            if self.transpose:
                for feats in zip(*neck_outputs):
                    if self.backend in (MMYOLOBackend.NCNN, MMYOLOBackend.TORCHSCRIPT):
                        outputs.append(torch.cat([feat.permute(0, 2, 3, 1) for feat in feats], -1))
                    else:
                        outputs.append(torch.cat(feats, 1).permute(0, 2, 3, 1))
            else:
                for feats in zip(*neck_outputs):
                    outputs.append(torch.cat(feats, 1))
            return tuple(outputs)

    @staticmethod
    def forward_single(x: 'Tensor', convs: 'nn.Module') ->Tuple[Tensor]:
        if isinstance(convs, nn.Sequential) and any(type(m) in (ImplicitA, ImplicitM) for m in convs):
            a, c, m = convs
            aw = a.implicit.clone()
            mw = m.implicit.clone()
            c = deepcopy(c)
            nw, cw, _, _ = c.weight.shape
            na, ca, _, _ = aw.shape
            nm, cm, _, _ = mw.shape
            c.bias = nn.Parameter(c.bias + (c.weight.reshape(nw, cw) @ aw.reshape(ca, na)).squeeze(1))
            c.bias = nn.Parameter(c.bias * mw.reshape(cm))
            c.weight = nn.Parameter(c.weight * mw.transpose(0, 1))
            convs = c
        feat = convs(x)
        return feat,

    @staticmethod
    def forward_gvp(x: 'Tensor') ->Tensor:
        return torch.mean(x, [2, 3], keepdim=True)


class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, inputs, data_samples, mode='tensor'):
        labels = torch.stack(data_samples)
        inputs = torch.stack(inputs)
        outputs = self.linear(inputs)
        if mode == 'tensor':
            return outputs
        elif mode == 'loss':
            loss = (labels - outputs).sum()
            outputs = dict(loss=loss)
            return outputs
        else:
            return outputs


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(torch.ones(1))
        self.conv1 = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(4, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ImplicitA,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ImplicitM,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (OksLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
]

