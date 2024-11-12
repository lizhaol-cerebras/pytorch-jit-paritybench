
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


import warnings


from scipy.optimize import linear_sum_assignment


import re


import time


import torch.distributed as dist


import functools


import copy


from abc import ABCMeta


from abc import abstractmethod


from collections import OrderedDict


from collections import defaultdict


from torch.utils.data import Dataset


import random


from functools import partial


from torch.utils.data import DataLoader


from collections.abc import Sequence


from torch.nn.modules.utils import _pair


import math


from torch.utils.data import DistributedSampler as _DistributedSampler


import torch.nn as nn


from torch import nn


from torch.nn.modules.utils import _ntuple


from torch.nn.modules.utils import _triple


import copy as cp


import torch.nn.functional as F


def download_file(url, filename=None):
    if filename is None:
        filename = url.split('/')[-1]
    response = requests.get(url)
    open(filename, 'wb').write(response.content)


def cache_checkpoint(filename, cache_dir='.cache'):
    if filename.startswith('http://') or filename.startswith('https://'):
        url = filename.split('//')[1]
        basename = filename.split('/')[-1]
        filehash = hashlib.md5(url.encode('utf8')).hexdigest()[-8:]
        os.makedirs(cache_dir, exist_ok=True)
        local_pth = osp.join(cache_dir, basename.replace('.pth', f'_{filehash}.pth'))
        if not osp.exists(local_pth):
            download_file(filename, local_pth)
        filename = local_pth
    return filename


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Use ``get_logger`` method in mmcv to get the root logger.
    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If ``log_file`` is specified, a FileHandler
    will also be added. The name of the root logger is the top-level package
    name, e.g., "pyskl".
    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.
    Returns:
        :obj:`logging.Logger`: The root logger.
    """
    return get_logger(__name__.split('.')[0], log_file, log_level)


class C3D(nn.Module):
    """C3D backbone, without flatten and mlp.

    Args:
        pretrained (str | None): Name of pretrained model.
    """

    def __init__(self, in_channels=3, base_channels=64, num_stages=4, temporal_downsample=True, pretrained=None):
        super().__init__()
        conv_cfg = dict(type='Conv3d')
        norm_cfg = dict(type='BN3d')
        act_cfg = dict(type='ReLU')
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.base_channels = base_channels
        assert num_stages in [3, 4]
        self.num_stages = num_stages
        self.temporal_downsample = temporal_downsample
        pool_kernel, pool_stride = 2, 2
        if not self.temporal_downsample:
            pool_kernel, pool_stride = (1, 2, 2), (1, 2, 2)
        c3d_conv_param = dict(kernel_size=3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1a = ConvModule(self.in_channels, self.base_channels, **c3d_conv_param)
        self.pool1 = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2a = ConvModule(self.base_channels, self.base_channels * 2, **c3d_conv_param)
        self.pool2 = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_stride)
        self.conv3a = ConvModule(self.base_channels * 2, self.base_channels * 4, **c3d_conv_param)
        self.conv3b = ConvModule(self.base_channels * 4, self.base_channels * 4, **c3d_conv_param)
        self.pool3 = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_stride)
        self.conv4a = ConvModule(self.base_channels * 4, self.base_channels * 8, **c3d_conv_param)
        self.conv4b = ConvModule(self.base_channels * 8, self.base_channels * 8, **c3d_conv_param)
        if self.num_stages == 4:
            self.pool4 = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_stride)
            self.conv5a = ConvModule(self.base_channels * 8, self.base_channels * 8, **c3d_conv_param)
            self.conv5b = ConvModule(self.base_channels * 8, self.base_channels * 8, **c3d_conv_param)

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data. The size of x is (num_batches, 3, 16, 112, 112).

        Returns:
            torch.Tensor: The feature of the input samples extracted by the backbone.
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
        if self.num_stages == 3:
            return x
        x = self.pool4(x)
        x = self.conv5a(x)
        x = self.conv5b(x)
        return x


class PoTion(nn.Module):

    def __init__(self, in_channels, channels=[128, 256, 512], num_layers=[2, 2, 2], lw_dropout=0, conv_cfg=dict(type='Conv'), norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_layers = num_layers
        self.lw_dropout = lw_dropout
        assert len(self.channels) == len(self.num_layers)
        layer_names = []
        inplanes = in_channels
        for i, (ch, num_layer) in enumerate(zip(channels, num_layers)):
            layer_name = f'layer{i + 1}'
            layer_names.append(layer_name)
            layer = []
            for j in range(num_layer):
                stride = 2 if j == 0 else 1
                conv = ConvModule(inplanes, ch, kernel_size=3, stride=stride, padding=1, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
                layer.append(conv)
                if self.lw_dropout > 0:
                    layer.append(nn.Dropout(self.lw_dropout))
                inplanes = ch
            layer = nn.Sequential(*layer)
            setattr(self, layer_name, layer)
        self.layer_names = layer_names

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The feature of the input samples extracted
            by the backbone.
        """
        for layer_name in self.layer_names:
            layer = getattr(self, layer_name)
            x = layer(x)
        return x

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)


class BasicBlock(nn.Module):
    """Basic block for ResNet.

    Args:
        inplanes (int): Number of channels for the input in first conv2d layer.
        planes (int): Number of channels produced by some norm/conv2d layers.
        stride (int): Stride in the conv layer. Default: 1.
        downsample (nn.Module | None): Downsample layer. Default: None.
        conv_cfg (dict): Config for norm layers. Default: dict(type='Conv').
        norm_cfg (dict): Config for norm layers. required keys are 'type' and 'requires_grad'.
            Default: dict(type='BN2d', requires_grad=True).
        act_cfg (dict): Config for activate layers. Default: dict(type='ReLU', inplace=True).
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, conv_cfg=dict(type='Conv'), norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        self.conv1 = ConvModule(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.norm_cfg = norm_cfg

    def forward(self, x):
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
        inplanes (int): Number of channels for the input feature in first conv layer.
        planes (int): Number of channels produced by some norm layes and conv layers.
        stride (int): Spatial stride in the conv layer. Default: 1.
        downsample (nn.Module | None): Downsample layer. Default: None.
        conv_cfg (dict): Config for norm layers. Default: dict(type='Conv').
        norm_cfg (dict): Config for norm layers. required keys are 'type' and 'requires_grad'.
            Default: dict(type='BN2d', requires_grad=True).
        act_cfg (dict): Config for activate layers. Default: dict(type='ReLU', inplace=True).
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, conv_cfg=dict(type='Conv'), norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1_stride = 1
        self.conv2_stride = stride
        self.conv1 = ConvModule(inplanes, planes, kernel_size=1, stride=self.conv1_stride, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(planes, planes, kernel_size=3, stride=self.conv2_stride, padding=1, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv3 = ConvModule(planes, planes * self.expansion, kernel_size=1, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.norm_cfg = norm_cfg

    def forward(self, x):
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
        out = _inner_forward(x)
        out = self.relu(out)
        return out


def make_res_layer(block, inplanes, planes, blocks, stride=1, conv_cfg=None, norm_cfg=None, act_cfg=None):
    """Build residual layer for ResNet.

    Args:
        block: (nn.Module): Residual module to be built.
        inplanes (int): Number of channels for the input feature in each block.
        planes (int): Number of channels for the output feature in each block.
        blocks (int): Number of residual blocks.
        stride (int): Stride in the conv layer. Default: 1.
        conv_cfg (dict | None): Config for norm layers. Default: None.
        norm_cfg (dict | None): Config for norm layers. Default: None.
        act_cfg (dict | None): Config for activate layers. Default: None.

    Returns:
        nn.Module: A residual layer for the given config.
    """
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = ConvModule(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
    layers = []
    layers.append(block(inplanes, planes, stride, downsample, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}. Default: 50.
        pretrained (str | None): Name of pretrained model. Default: None.
        in_channels (int): Channel num of input features. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        out_indices (Sequence[int]): Indices of output feature. Default: (3, ).
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict): Config for norm layers. Default: dict(type='Conv').
        norm_cfg (dict): Config for norm layers. required keys are 'type' and 'requires_grad'.
            Default: dict(type='BN2d', requires_grad=True).
        act_cfg (dict): Config for activate layers. Default: dict(type='ReLU', inplace=True).
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze running stats (mean and var).
            Default: False.
    """
    arch_settings = {(18): (BasicBlock, (2, 2, 2, 2)), (34): (BasicBlock, (3, 4, 6, 3)), (50): (Bottleneck, (3, 4, 6, 3)), (101): (Bottleneck, (3, 4, 23, 3)), (152): (Bottleneck, (3, 8, 36, 3))}

    def __init__(self, depth=50, pretrained=None, torchvision_pretrain=True, in_channels=3, num_stages=4, out_indices=(3,), strides=(1, 2, 2, 2), frozen_stages=-1, conv_cfg=dict(type='Conv'), norm_cfg=dict(type='BN2d', requires_grad=True), act_cfg=dict(type='ReLU', inplace=True), norm_eval=False):
        super().__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.torchvision_pretrain = torchvision_pretrain
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.strides = strides
        assert len(strides) == num_stages
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64
        self._make_stem_layer()
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            planes = 64 * 2 ** i
            res_layer = make_res_layer(self.block, self.inplanes, planes, num_blocks, stride=stride, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
        self.feat_dim = self.block.expansion * 64 * 2 ** (len(self.stage_blocks) - 1)

    def _make_stem_layer(self):
        """Construct the stem layers consists of a conv+norm+act module and a
        pooling layer."""
        self.conv1 = ConvModule(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    @staticmethod
    def _load_conv_params(conv, state_dict_tv, module_name_tv, loaded_param_names):
        """Load the conv parameters of resnet from torchvision.

        Args:
            conv (nn.Module): The destination conv module.
            state_dict_tv (OrderedDict): The state dict of pretrained torchvision model.
            module_name_tv (str): The name of corresponding conv module in the torchvision model.
            loaded_param_names (list[str]): List of parameters that have been loaded.
        """
        weight_tv_name = module_name_tv + '.weight'
        if conv.weight.data.shape == state_dict_tv[weight_tv_name].shape:
            conv.weight.data.copy_(state_dict_tv[weight_tv_name])
            loaded_param_names.append(weight_tv_name)
        if getattr(conv, 'bias') is not None:
            bias_tv_name = module_name_tv + '.bias'
            if conv.bias.data.shape == state_dict_tv[bias_tv_name].shape:
                conv.bias.data.copy_(state_dict_tv[bias_tv_name])
                loaded_param_names.append(bias_tv_name)

    @staticmethod
    def _load_bn_params(bn, state_dict_tv, module_name_tv, loaded_param_names):
        """Load the bn parameters of resnet from torchvision.

        Args:
            bn (nn.Module): The destination bn module.
            state_dict_tv (OrderedDict): The state dict of pretrained torchvision model.
            module_name_tv (str): The name of corresponding bn module in the torchvision model.
            loaded_param_names (list[str]): List of parameters that have been loaded.
        """
        for param_name, param in bn.named_parameters():
            param_tv_name = f'{module_name_tv}.{param_name}'
            param_tv = state_dict_tv[param_tv_name]
            if param.data.shape == param_tv.shape:
                param.data.copy_(param_tv)
                loaded_param_names.append(param_tv_name)
        for param_name, param in bn.named_buffers():
            param_tv_name = f'{module_name_tv}.{param_name}'
            if param_tv_name in state_dict_tv:
                param_tv = state_dict_tv[param_tv_name]
                if param.data.shape == param_tv.shape:
                    param.data.copy_(param_tv)
                    loaded_param_names.append(param_tv_name)

    def _load_torchvision_checkpoint(self, logger=None):
        """Initiate the parameters from torchvision pretrained checkpoint."""
        state_dict_torchvision = _load_checkpoint(self.pretrained)
        if 'state_dict' in state_dict_torchvision:
            state_dict_torchvision = state_dict_torchvision['state_dict']
        loaded_param_names = []
        for name, module in self.named_modules():
            if isinstance(module, ConvModule):
                if 'downsample' in name:
                    original_conv_name = name + '.0'
                    original_bn_name = name + '.1'
                else:
                    original_conv_name = name
                    original_bn_name = name.replace('conv', 'bn')
                self._load_conv_params(module.conv, state_dict_torchvision, original_conv_name, loaded_param_names)
                self._load_bn_params(module.bn, state_dict_torchvision, original_bn_name, loaded_param_names)
        remaining_names = set(state_dict_torchvision.keys()) - set(loaded_param_names)
        if remaining_names:
            logger.info(f'These parameters in pretrained checkpoint are not loaded: {remaining_names}')

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            if self.torchvision_pretrain:
                self._load_torchvision_checkpoint(logger)
            else:
                self.pretrained = cache_checkpoint(self.pretrained)
                load_checkpoint(self, self.pretrained, strict=False, logger=logger)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The feature of the input samples extracted
            by the backbone.
        """
        x = self.conv1(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        return tuple(outs)

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before self.frozen_stages."""
        if self.frozen_stages >= 0:
            self.conv1.bn.eval()
            for m in self.conv1.modules():
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


class BasicBlock3d(nn.Module):
    """BasicBlock 3d block for ResNet3D.

    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        stride (tuple): Stride is a two element tuple (temporal, spatial). Default: (1, 1).
        downsample (nn.Module | None): Downsample layer. Default: None.
        inflate (bool): Whether to inflate kernel. Default: True.
        conv_cfg (dict): Config dict for convolution layer. Default: 'dict(type='Conv3d')'.
        norm_cfg (dict): Config for norm layers. required keys are 'type'. Default: 'dict(type='BN3d')'.
        act_cfg (dict): Config dict for activation layer. Default: 'dict(type='ReLU')'.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None, inflate=True, inflate_style='3x3x3', conv_cfg=dict(type='Conv3d'), norm_cfg=dict(type='BN3d'), act_cfg=dict(type='ReLU')):
        super().__init__()
        assert inflate_style == '3x3x3'
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.inflate = inflate
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.conv1 = ConvModule(inplanes, planes, 3 if self.inflate else (1, 3, 3), stride=(self.stride[0], self.stride[1], self.stride[1]), padding=1 if self.inflate else (0, 1, 1), bias=False, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv2 = ConvModule(planes, planes * self.expansion, 3 if self.inflate else (1, 3, 3), stride=1, padding=1 if self.inflate else (0, 1, 1), bias=False, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=None)
        self.downsample = downsample
        self.relu = build_activation_layer(self.act_cfg)

    def forward(self, x):
        """Defines the computation performed at every call."""

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            identity = x
            out = self.conv1(x)
            out = self.conv2(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out = out + identity
            return out
        out = _inner_forward(x)
        out = self.relu(out)
        return out


class Bottleneck3d(nn.Module):
    """Bottleneck 3d block for ResNet3D.

    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        stride (tuple): Stride is a two element tuple (temporal, spatial). Default: (1, 1).
        downsample (nn.Module | None): Downsample layer. Default: None.
        inflate (bool): Whether to inflate kernel. Default: True.
        inflate_style (str): '3x1x1' or '3x3x3'. which determines the kernel sizes and padding strides
            for conv1 and conv2 in each block. Default: '3x1x1'.
        conv_cfg (dict): Config dict for convolution layer. Default: 'dict(type='Conv3d')'.
        norm_cfg (dict): Config for norm layers. required keys are 'type'. Default: 'dict(type='BN3d')'.
        act_cfg (dict): Config dict for activation layer. Default: 'dict(type='ReLU')'.
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None, inflate=True, inflate_style='3x1x1', conv_cfg=dict(type='Conv3d'), norm_cfg=dict(type='BN3d'), act_cfg=dict(type='ReLU')):
        super().__init__()
        assert inflate_style in ['3x1x1', '3x3x3']
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.inflate = inflate
        self.inflate_style = inflate_style
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        mode = 'no_inflate' if not self.inflate else self.inflate_style
        conv1_kernel_size = {'no_inflate': 1, '3x1x1': (3, 1, 1), '3x3x3': 1}
        conv1_padding = {'no_inflate': 0, '3x1x1': (1, 0, 0), '3x3x3': 0}
        conv2_kernel_size = {'no_inflate': (1, 3, 3), '3x1x1': (1, 3, 3), '3x3x3': 3}
        conv2_padding = {'no_inflate': (0, 1, 1), '3x1x1': (0, 1, 1), '3x3x3': 1}
        self.conv1 = ConvModule(inplanes, planes, conv1_kernel_size[mode], stride=1, padding=conv1_padding[mode], bias=False, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv2 = ConvModule(planes, planes, conv2_kernel_size[mode], stride=(self.stride[0], self.stride[1], self.stride[1]), padding=conv2_padding[mode], bias=False, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv3 = ConvModule(planes, planes * self.expansion, 1, bias=False, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=None)
        self.downsample = downsample
        self.relu = build_activation_layer(self.act_cfg)

    def forward(self, x):
        """Defines the computation performed at every call."""

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
        out = _inner_forward(x)
        out = self.relu(out)
        return out


class ResNet3d(nn.Module):
    """ResNet 3d backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}. Default: 50.
        pretrained (str | None): Name of pretrained model.
        stage_blocks (tuple | None): Set number of stages for each res layer. Default: None.
        pretrained2d (bool): Whether to load pretrained 2D model. Default: True.
        in_channels (int): Channel num of input features. Default: 3.
        base_channels (int): Channel num of stem output features. Default: 64.
        out_indices (tuple[int]): Indices of output feature. Default: (3, ).
        num_stages (int): Resnet stages. Default: 4.
        spatial_strides (tuple[int]): Spatial strides of residual blocks of each stage. Default: (1, 2, 2, 2).
        temporal_strides (tuple[int]): Temporal strides of residual blocks of each stage. Default: (1, 1, 1, 1).
        conv1_kernel (tuple[int]): Kernel size of the first conv layer. Default: (3, 7, 7).
        conv1_stride (tuple[int]): Stride of the first conv layer (temporal, spatial). Default: (1, 2).
        pool1_stride (tuple[int]): Stride of the first pooling layer (temporal, spatial). Default: (1, 2).
        advanced (bool): Flag indicating if an advanced design for downsample is adopted. Default: False.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means not freezing any parameters. Default: -1.
        inflate (tuple[int]): Inflate Dims of each block. Default: (1, 1, 1, 1).
        inflate_style (str): '3x1x1' or '3x3x3'. which determines the kernel sizes and padding strides
            for conv1 and conv2 in each block. Default: '3x1x1'.
        conv_cfg (dict): Config for conv layers. required keys are 'type'. Default: 'dict(type='Conv3d')'.
        norm_cfg (dict): Config for norm layers. required keys are 'type' and 'requires_grad'.
            Default: 'dict(type='BN3d', requires_grad=True)'.
        act_cfg (dict): Config dict for activation layer. Default: 'dict(type='ReLU', inplace=True)'.
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze running stats (mean and var).
            Default: False.
        zero_init_residual (bool): Whether to use zero initialization for residual block. Default: True.
    """
    arch_settings = {(18): (BasicBlock3d, (2, 2, 2, 2)), (34): (BasicBlock3d, (3, 4, 6, 3)), (50): (Bottleneck3d, (3, 4, 6, 3)), (101): (Bottleneck3d, (3, 4, 23, 3)), (152): (Bottleneck3d, (3, 8, 36, 3))}

    def __init__(self, depth=50, pretrained=None, stage_blocks=None, pretrained2d=True, in_channels=3, num_stages=4, base_channels=64, out_indices=(3,), spatial_strides=(1, 2, 2, 2), temporal_strides=(1, 1, 1, 1), conv1_kernel=(3, 7, 7), conv1_stride=(1, 2), pool1_stride=(1, 2), advanced=False, frozen_stages=-1, inflate=(1, 1, 1, 1), inflate_style='3x1x1', conv_cfg=dict(type='Conv3d'), norm_cfg=dict(type='BN3d', requires_grad=True), act_cfg=dict(type='ReLU', inplace=True), norm_eval=False, zero_init_residual=True):
        super().__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.stage_blocks = stage_blocks
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        assert len(spatial_strides) == len(temporal_strides) == num_stages
        if self.stage_blocks is not None:
            assert len(self.stage_blocks) == num_stages
        self.conv1_kernel = conv1_kernel
        self.conv1_stride = conv1_stride
        self.pool1_stride = pool1_stride
        self.advanced = advanced
        self.frozen_stages = frozen_stages
        self.stage_inflations = _ntuple(num_stages)(inflate)
        self.inflate_style = inflate_style
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        if self.stage_blocks is None:
            self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = self.base_channels
        self._make_stem_layer()
        self.res_layers = []
        lateral_inplanes = getattr(self, 'lateral_inplanes', [0, 0, 0, 0])
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            temporal_stride = temporal_strides[i]
            planes = self.base_channels * 2 ** i
            res_layer = self.make_res_layer(self.block, self.inplanes + lateral_inplanes[i], planes, num_blocks, stride=(temporal_stride, spatial_stride), norm_cfg=self.norm_cfg, conv_cfg=self.conv_cfg, act_cfg=self.act_cfg, advanced=self.advanced, inflate=self.stage_inflations[i], inflate_style=self.inflate_style)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
        self.feat_dim = self.block.expansion * self.base_channels * 2 ** (len(self.stage_blocks) - 1)

    @staticmethod
    def make_res_layer(block, inplanes, planes, blocks, stride=(1, 1), inflate=1, inflate_style='3x1x1', advanced=False, norm_cfg=None, act_cfg=None, conv_cfg=None):
        """Build residual layer for ResNet3D.

        Args:
            block (nn.Module): Residual module to be built.
            inplanes (int): Number of channels for the input feature in each block.
            planes (int): Number of channels for the output feature in each block.
            blocks (int): Number of residual blocks.
            stride (tuple[int]): Stride (temporal, spatial) in residual and conv layers. Default: (1, 1).
            inflate (int | tuple[int]): Determine whether to inflate for each block. Default: 1.
            inflate_style (str): '3x1x1' or '3x3x3'. which determines the kernel sizes and padding strides
                for conv1 and conv2 in each block. Default: '3x1x1'.
            conv_cfg (dict | None): Config for norm layers. Default: None.
            norm_cfg (dict | None): Config for norm layers. Default: None.
            act_cfg (dict | None): Config for activate layers. Default: None.

        Returns:
            nn.Module: A residual layer for the given config.
        """
        inflate = inflate if not isinstance(inflate, int) else (inflate,) * blocks
        assert len(inflate) == blocks
        downsample = None
        if stride[1] != 1 or inplanes != planes * block.expansion:
            if advanced:
                conv = ConvModule(inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
                pool = nn.AvgPool3d(kernel_size=(stride[0], stride[1], stride[1]), stride=(stride[0], stride[1], stride[1]), ceil_mode=True)
                downsample = nn.Sequential(conv, pool)
            else:
                downsample = ConvModule(inplanes, planes * block.expansion, kernel_size=1, stride=(stride[0], stride[1], stride[1]), bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
        layers = []
        layers.append(block(inplanes, planes, stride=stride, downsample=downsample, inflate=inflate[0] == 1, inflate_style=inflate_style, norm_cfg=norm_cfg, conv_cfg=conv_cfg, act_cfg=act_cfg))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=(1, 1), inflate=inflate[i] == 1, inflate_style=inflate_style, norm_cfg=norm_cfg, conv_cfg=conv_cfg, act_cfg=act_cfg))
        return nn.Sequential(*layers)

    @staticmethod
    def _inflate_conv_params(conv3d, state_dict_2d, module_name_2d, inflated_param_names):
        """Inflate a conv module from 2d to 3d.

        Args:
            conv3d (nn.Module): The destination conv3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
            module_name_2d (str): The name of corresponding conv module in the 2d model.
            inflated_param_names (list[str]): List of parameters that have been inflated.
        """
        weight_2d_name = module_name_2d + '.weight'
        conv2d_weight = state_dict_2d[weight_2d_name]
        kernel_t = conv3d.weight.data.shape[2]
        new_weight = conv2d_weight.data.unsqueeze(2).expand_as(conv3d.weight) / kernel_t
        conv3d.weight.data.copy_(new_weight)
        inflated_param_names.append(weight_2d_name)
        if getattr(conv3d, 'bias') is not None:
            bias_2d_name = module_name_2d + '.bias'
            conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
            inflated_param_names.append(bias_2d_name)

    @staticmethod
    def _inflate_bn_params(bn3d, state_dict_2d, module_name_2d, inflated_param_names):
        """Inflate a norm module from 2d to 3d.

        Args:
            bn3d (nn.Module): The destination bn3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
            module_name_2d (str): The name of corresponding bn module in the 2d model.
            inflated_param_names (list[str]): List of parameters that have been inflated.
        """
        for param_name, param in bn3d.named_parameters():
            param_2d_name = f'{module_name_2d}.{param_name}'
            param_2d = state_dict_2d[param_2d_name]
            if param.data.shape != param_2d.shape:
                warnings.warn(f'The parameter of {module_name_2d} is not loaded due to incompatible shapes. ')
                return
            param.data.copy_(param_2d)
            inflated_param_names.append(param_2d_name)
        for param_name, param in bn3d.named_buffers():
            param_2d_name = f'{module_name_2d}.{param_name}'
            if param_2d_name in state_dict_2d:
                param_2d = state_dict_2d[param_2d_name]
                param.data.copy_(param_2d)
                inflated_param_names.append(param_2d_name)

    @staticmethod
    def _inflate_weights(self, logger):
        """Inflate the resnet2d parameters to resnet3d.

        The differences between resnet3d and resnet2d mainly lie in an extra
        axis of conv kernel. To utilize the pretrained parameters in 2d model,
        the weight of conv2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (logging.Logger): The logger used to print
                debugging information.
        """
        state_dict_r2d = _load_checkpoint(self.pretrained)
        if 'state_dict' in state_dict_r2d:
            state_dict_r2d = state_dict_r2d['state_dict']
        inflated_param_names = []
        for name, module in self.named_modules():
            if isinstance(module, ConvModule):
                if 'downsample' in name:
                    original_conv_name = name + '.0'
                    original_bn_name = name + '.1'
                else:
                    original_conv_name = name
                    original_bn_name = name.replace('conv', 'bn')
                if original_conv_name + '.weight' not in state_dict_r2d:
                    logger.warning(f'Module not exist in the state_dict_r2d: {original_conv_name}')
                else:
                    shape_2d = state_dict_r2d[original_conv_name + '.weight'].shape
                    shape_3d = module.conv.weight.data.shape
                    if shape_2d != shape_3d[:2] + shape_3d[3:]:
                        logger.warning(f'Weight shape mismatch for: {original_conv_name}: 3d weight shape: {shape_3d}; 2d weight shape: {shape_2d}.')
                    else:
                        self._inflate_conv_params(module.conv, state_dict_r2d, original_conv_name, inflated_param_names)
                if original_bn_name + '.weight' not in state_dict_r2d:
                    logger.warning(f'Module not exist in the state_dict_r2d: {original_bn_name}')
                else:
                    self._inflate_bn_params(module.bn, state_dict_r2d, original_bn_name, inflated_param_names)
        remaining_names = set(state_dict_r2d.keys()) - set(inflated_param_names)
        if remaining_names:
            logger.info(f'These parameters in the 2d checkpoint are not loaded: {remaining_names}')

    def inflate_weights(self, logger):
        self._inflate_weights(self, logger)

    def _make_stem_layer(self):
        """Construct the stem layers consists of a conv+norm+act module and a
        pooling layer."""
        self.conv1 = ConvModule(self.in_channels, self.base_channels, kernel_size=self.conv1_kernel, stride=(self.conv1_stride[0], self.conv1_stride[1], self.conv1_stride[1]), padding=tuple([((k - 1) // 2) for k in _triple(self.conv1_kernel)]), bias=False, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(self.pool1_stride[0], self.pool1_stride[1], self.pool1_stride[1]), padding=(0, 1, 1))

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        'self.frozen_stages'."""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    @staticmethod
    def _init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch.

        Args:
            pretrained (str | None): The path of the pretrained weight. Will override the original 'pretrained' if set.
                The arg is added to be compatible with mmdet. Default: None.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck3d):
                    constant_init(m.conv3.bn, 0)
                elif isinstance(m, BasicBlock3d):
                    constant_init(m.conv2.bn, 0)
        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')
            if self.pretrained2d:
                self.inflate_weights(logger)
            else:
                self.pretrained = cache_checkpoint(self.pretrained)
                load_checkpoint(self, self.pretrained, strict=False, logger=logger)

    def init_weights(self, pretrained=None):
        self._init_weights(self, pretrained)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The feature of the input
            samples extracted by the backbone.
        """
        x = self.conv1(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        return tuple(outs)

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


class DeConvModule(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), padding=0, bias=False, with_bn=True, with_relu=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.with_bn = with_bn
        self.with_relu = with_relu
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        assert len(x.shape) == 5
        N, C, T, H, W = x.shape
        out_shape = N, self.out_channels, self.stride[0] * T, self.stride[1] * H, self.stride[2] * W
        x = self.conv(x, output_size=out_shape)
        if self.with_bn:
            x = self.bn(x)
        if self.with_relu:
            x = self.relu(x)
        return x


class ResNet3dPathway(ResNet3d):
    """A pathway of Slowfast based on ResNet3d.

    Args:
        lateral (bool): Determines whether to enable the lateral connection from another pathway. Default: False.
        speed_ratio (int): Speed ratio indicating the ratio between time dimension of the fast and slow pathway,
            corresponding to the 'alpha' in the paper. Default: 8.
        channel_ratio (int): Reduce the channel number of fast pathway by 'channel_ratio',
            corresponding to 'beta' in the paper. Default: 8.
        fusion_kernel (int): The kernel size of lateral fusion. Default: 7.
        **kwargs (keyword arguments): Keywords arguments for ResNet3d.
    """

    def __init__(self, lateral=False, lateral_inv=False, speed_ratio=8, channel_ratio=8, fusion_kernel=7, lateral_infl=2, lateral_activate=[1, 1, 1, 1], **kwargs):
        self.lateral = lateral
        self.lateral_inv = lateral_inv
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio
        self.fusion_kernel = fusion_kernel
        self.lateral_infl = lateral_infl
        self.lateral_activate = lateral_activate
        self.calculate_lateral_inplanes(kwargs)
        super().__init__(**kwargs)
        self.inplanes = self.base_channels
        if self.lateral and self.lateral_activate[0] == 1:
            if self.lateral_inv:
                self.conv1_lateral = DeConvModule(self.inplanes * self.channel_ratio, self.inplanes * self.channel_ratio // self.lateral_infl, kernel_size=(fusion_kernel, 1, 1), stride=(self.speed_ratio, 1, 1), padding=((fusion_kernel - 1) // 2, 0, 0), with_bn=True, with_relu=True)
            else:
                self.conv1_lateral = ConvModule(self.inplanes // self.channel_ratio, self.inplanes * lateral_infl // self.channel_ratio, kernel_size=(fusion_kernel, 1, 1), stride=(self.speed_ratio, 1, 1), padding=((fusion_kernel - 1) // 2, 0, 0), bias=False, conv_cfg=self.conv_cfg, norm_cfg=None, act_cfg=None)
        self.lateral_connections = []
        for i in range(len(self.stage_blocks)):
            planes = self.base_channels * 2 ** i
            self.inplanes = planes * self.block.expansion
            if lateral and i != self.num_stages - 1 and self.lateral_activate[i + 1]:
                lateral_name = f'layer{i + 1}_lateral'
                if self.lateral_inv:
                    conv_module = DeConvModule(self.inplanes * self.channel_ratio, self.inplanes * self.channel_ratio // self.lateral_infl, kernel_size=(fusion_kernel, 1, 1), stride=(self.speed_ratio, 1, 1), padding=((fusion_kernel - 1) // 2, 0, 0), bias=False, with_bn=True, with_relu=True)
                else:
                    conv_module = ConvModule(self.inplanes // self.channel_ratio, self.inplanes * lateral_infl // self.channel_ratio, kernel_size=(fusion_kernel, 1, 1), stride=(self.speed_ratio, 1, 1), padding=((fusion_kernel - 1) // 2, 0, 0), bias=False, conv_cfg=self.conv_cfg, norm_cfg=None, act_cfg=None)
                setattr(self, lateral_name, conv_module)
                self.lateral_connections.append(lateral_name)

    def calculate_lateral_inplanes(self, kwargs):
        depth = kwargs.get('depth', 50)
        expansion = 1 if depth < 50 else 4
        base_channels = kwargs.get('base_channels', 64)
        lateral_inplanes = []
        for i in range(kwargs.get('num_stages', 4)):
            if expansion % 2 == 0:
                planes = base_channels * 2 ** i * (expansion // 2) ** (i > 0)
            else:
                planes = base_channels * 2 ** i // 2 ** (i > 0)
            if self.lateral and self.lateral_activate[i]:
                if self.lateral_inv:
                    lateral_inplane = planes * self.channel_ratio // self.lateral_infl
                else:
                    lateral_inplane = planes * self.lateral_infl // self.channel_ratio
            else:
                lateral_inplane = 0
            lateral_inplanes.append(lateral_inplane)
        self.lateral_inplanes = lateral_inplanes

    def inflate_weights(self, logger):
        """Inflate the resnet2d parameters to resnet3d pathway.

        The differences between resnet3d and resnet2d mainly lie in an extra
        axis of conv kernel. To utilize the pretrained parameters in 2d model,
        the weight of conv2d models should be inflated to fit in the shapes of
        the 3d counterpart. For pathway the 'lateral_connection' part should
        not be inflated from 2d weights.

        Args:
            logger (logging.Logger): The logger used to print debugging information.
        """
        state_dict_r2d = _load_checkpoint(self.pretrained)
        if 'state_dict' in state_dict_r2d:
            state_dict_r2d = state_dict_r2d['state_dict']
        inflated_param_names = []
        for name, module in self.named_modules():
            if 'lateral' in name:
                continue
            if isinstance(module, ConvModule):
                if 'downsample' in name:
                    original_conv_name = name + '.0'
                    original_bn_name = name + '.1'
                else:
                    original_conv_name = name
                    original_bn_name = name.replace('conv', 'bn')
                if original_conv_name + '.weight' not in state_dict_r2d:
                    logger.warning(f'Module not exist in the state_dict_r2d: {original_conv_name}')
                else:
                    self._inflate_conv_params(module.conv, state_dict_r2d, original_conv_name, inflated_param_names)
                if original_bn_name + '.weight' not in state_dict_r2d:
                    logger.warning(f'Module not exist in the state_dict_r2d: {original_bn_name}')
                else:
                    self._inflate_bn_params(module.bn, state_dict_r2d, original_bn_name, inflated_param_names)
        remaining_names = set(state_dict_r2d.keys()) - set(inflated_param_names)
        if remaining_names:
            logger.info(f'These parameters in the 2d checkpoint are not loaded: {remaining_names}')

    def _inflate_conv_params(self, conv3d, state_dict_2d, module_name_2d, inflated_param_names):
        """Inflate a conv module from 2d to 3d.

        The differences of conv modules betweene 2d and 3d in Pathway
        mainly lie in the inplanes due to lateral connections. To fit the
        shapes of the lateral connection counterpart, it will expand
        parameters by concatting conv2d parameters and extra zero paddings.

        Args:
            conv3d (nn.Module): The destination conv3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
            module_name_2d (str): The name of corresponding conv module in the 2d model.
            inflated_param_names (list[str]): List of parameters that have been inflated.
        """
        weight_2d_name = module_name_2d + '.weight'
        conv2d_weight = state_dict_2d[weight_2d_name]
        old_shape = conv2d_weight.shape
        new_shape = conv3d.weight.data.shape
        kernel_t = new_shape[2]
        if new_shape[1] != old_shape[1]:
            if new_shape[1] < old_shape[1]:
                warnings.warn(f'The parameter of {module_name_2d} is notloaded due to incompatible shapes. ')
                return
            new_channels = new_shape[1] - old_shape[1]
            pad_shape = old_shape
            pad_shape = pad_shape[:1] + (new_channels,) + pad_shape[2:]
            conv2d_weight = torch.cat((conv2d_weight, torch.zeros(pad_shape).type_as(conv2d_weight)), dim=1)
        new_weight = conv2d_weight.data.unsqueeze(2).expand_as(conv3d.weight) / kernel_t
        conv3d.weight.data.copy_(new_weight)
        inflated_param_names.append(weight_2d_name)
        if getattr(conv3d, 'bias') is not None:
            bias_2d_name = module_name_2d + '.bias'
            conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
            inflated_param_names.append(bias_2d_name)

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before'self.frozen_stages'. """
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
            if i != len(self.res_layers) and self.lateral:
                lateral_name = self.lateral_connections[i - 1]
                conv_lateral = getattr(self, lateral_name)
                conv_lateral.eval()
                for param in conv_lateral.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if pretrained:
            self.pretrained = pretrained
        super().init_weights()
        for module_name in self.lateral_connections:
            layer = getattr(self, module_name)
            for m in layer.modules():
                if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                    kaiming_init(m)


class ResNet3dSlowFast(nn.Module):
    """Slowfast backbone.

    This module is proposed in `SlowFast Networks for Video Recognition
    <https://arxiv.org/abs/1812.03982>`_

    Args:
        pretrained (str): The file path to a pretrained model.
        resample_rate (int): A large temporal stride 'resample_rate' on input frames. The actual resample rate is
            calculated by multipling the 'interval' in 'SampleFrames' in the pipeline with 'resample_rate', equivalent
            to the :math:`\\tau` in the paper, i.e. it processes only one out of 'resample_rate * interval' frames.
            Default: 8.
        speed_ratio (int): Speed ratio indicating the ratio between time dimension of the fast and slow pathway,
            corresponding to the :math:`\\alpha` in the paper. Default: 8.
        channel_ratio (int): Reduce the channel number of fast pathway by 'channel_ratio', corresponding to
            :math:`\\beta` in the paper. Default: 8.
        slow_pathway (dict): Configuration of slow branch.
            Default: dict(lateral=True, depth=50, conv1_kernel=(1, 7, 7), inflate=(0, 0, 1, 1))
        fast_pathway (dict): Configuration of fast branch.
            Default: dict(lateral=False, depth=50, base_channels=8, conv1_kernel=(5, 7, 7))
    """

    def __init__(self, pretrained=None, resample_rate=8, speed_ratio=8, channel_ratio=8, slow_pathway=dict(depth=50, lateral=True, conv1_kernel=(1, 7, 7), inflate=(0, 0, 1, 1)), fast_pathway=dict(depth=50, lateral=False, base_channels=8, conv1_kernel=(5, 7, 7))):
        super().__init__()
        self.pretrained = pretrained
        self.resample_rate = resample_rate
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio
        if slow_pathway['lateral']:
            slow_pathway['speed_ratio'] = speed_ratio
            slow_pathway['channel_ratio'] = channel_ratio
        self.slow_path = ResNet3dPathway(**slow_pathway)
        self.fast_path = ResNet3dPathway(**fast_pathway)

    def init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            msg = f'load model from: {self.pretrained}'
            print_log(msg, logger=logger)
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=True, logger=logger)
        elif self.pretrained is None:
            self.fast_path.init_weights()
            self.slow_path.init_weights()
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            tuple[torch.Tensor]: The feature of the input samples extracted by the backbone.
        """
        x_slow = nn.functional.interpolate(x, mode='nearest', scale_factor=(1.0 / self.resample_rate, 1.0, 1.0))
        x_slow = self.slow_path.conv1(x_slow)
        x_slow = self.slow_path.maxpool(x_slow)
        x_fast = nn.functional.interpolate(x, mode='nearest', scale_factor=(1.0 / (self.resample_rate // self.speed_ratio), 1.0, 1.0))
        x_fast = self.fast_path.conv1(x_fast)
        x_fast = self.fast_path.maxpool(x_fast)
        if self.slow_path.lateral:
            x_fast_lateral = self.slow_path.conv1_lateral(x_fast)
            x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)
        for i, layer_name in enumerate(self.slow_path.res_layers):
            res_layer = getattr(self.slow_path, layer_name)
            x_slow = res_layer(x_slow)
            res_layer_fast = getattr(self.fast_path, layer_name)
            x_fast = res_layer_fast(x_fast)
            if i != len(self.slow_path.res_layers) - 1 and self.slow_path.lateral:
                lateral_name = self.slow_path.lateral_connections[i]
                conv_lateral = getattr(self.slow_path, lateral_name)
                x_fast_lateral = conv_lateral(x_fast)
                x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)
        out = x_slow, x_fast
        return out


class RGBPoseConv3D(nn.Module):
    """Slowfast backbone.

    Args:
        pretrained (str): The file path to a pretrained model.
        speed_ratio (int): Speed ratio indicating the ratio between time
            dimension of the fast and slow pathway, corresponding to the
            :math:`\\alpha` in the paper. Default: 4.
        channel_ratio (int): Reduce the channel number of fast pathway
            by ``channel_ratio``, corresponding to :math:`\\beta` in the paper.
            Default: 4.
    """

    def __init__(self, pretrained=None, speed_ratio=4, channel_ratio=4, rgb_detach=False, pose_detach=False, rgb_drop_path=0, pose_drop_path=0, rgb_pathway=dict(num_stages=4, lateral=True, lateral_infl=1, lateral_activate=(0, 0, 1, 1), base_channels=64, conv1_kernel=(1, 7, 7), inflate=(0, 0, 1, 1)), pose_pathway=dict(num_stages=3, stage_blocks=(4, 6, 3), lateral=True, lateral_inv=True, lateral_infl=16, lateral_activate=(0, 1, 1), in_channels=17, base_channels=32, out_indices=(2,), conv1_kernel=(1, 7, 7), conv1_stride=(1, 1), pool1_stride=(1, 1), inflate=(0, 1, 1), spatial_strides=(2, 2, 2), temporal_strides=(1, 1, 1))):
        super().__init__()
        self.pretrained = pretrained
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio
        if rgb_pathway['lateral']:
            rgb_pathway['speed_ratio'] = speed_ratio
            rgb_pathway['channel_ratio'] = channel_ratio
        if pose_pathway['lateral']:
            pose_pathway['speed_ratio'] = speed_ratio
            pose_pathway['channel_ratio'] = channel_ratio
        self.rgb_path = ResNet3dPathway(**rgb_pathway)
        self.pose_path = ResNet3dPathway(**pose_pathway)
        self.rgb_detach = rgb_detach
        self.pose_detach = pose_detach
        assert 0 <= rgb_drop_path <= 1
        assert 0 <= pose_drop_path <= 1
        self.rgb_drop_path = rgb_drop_path
        self.pose_drop_path = pose_drop_path

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            msg = f'load model from: {self.pretrained}'
            print_log(msg, logger=logger)
            load_checkpoint(self, self.pretrained, strict=True, logger=logger)
        elif self.pretrained is None:
            self.rgb_path.init_weights()
            self.pose_path.init_weights()
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, imgs, heatmap_imgs):
        """Defines the computation performed at every call.

        Args:
            imgs (torch.Tensor): The input data.
            heatmap_imgs (torch.Tensor): The input data.

        Returns:
            tuple[torch.Tensor]: The feature of the input
            samples extracted by the backbone.
        """
        if self.training:
            rgb_drop_path = torch.rand(1) < self.rgb_drop_path
            pose_drop_path = torch.rand(1) < self.pose_drop_path
        else:
            rgb_drop_path, pose_drop_path = False, False
        x_rgb = self.rgb_path.conv1(imgs)
        x_rgb = self.rgb_path.maxpool(x_rgb)
        x_pose = self.pose_path.conv1(heatmap_imgs)
        x_pose = self.pose_path.maxpool(x_pose)
        x_rgb = self.rgb_path.layer1(x_rgb)
        x_rgb = self.rgb_path.layer2(x_rgb)
        x_pose = self.pose_path.layer1(x_pose)
        if hasattr(self.rgb_path, 'layer2_lateral'):
            feat = x_pose.detach() if self.rgb_detach else x_pose
            x_pose_lateral = self.rgb_path.layer2_lateral(feat)
            if rgb_drop_path:
                x_pose_lateral = x_pose_lateral.new_zeros(x_pose_lateral.shape)
        if hasattr(self.pose_path, 'layer1_lateral'):
            feat = x_rgb.detach() if self.pose_detach else x_rgb
            x_rgb_lateral = self.pose_path.layer1_lateral(feat)
            if pose_drop_path:
                x_rgb_lateral = x_rgb_lateral.new_zeros(x_rgb_lateral.shape)
        if hasattr(self.rgb_path, 'layer2_lateral'):
            x_rgb = torch.cat((x_rgb, x_pose_lateral), dim=1)
        if hasattr(self.pose_path, 'layer1_lateral'):
            x_pose = torch.cat((x_pose, x_rgb_lateral), dim=1)
        x_rgb = self.rgb_path.layer3(x_rgb)
        x_pose = self.pose_path.layer2(x_pose)
        if hasattr(self.rgb_path, 'layer3_lateral'):
            feat = x_pose.detach() if self.rgb_detach else x_pose
            x_pose_lateral = self.rgb_path.layer3_lateral(feat)
            if rgb_drop_path:
                x_pose_lateral = x_pose_lateral.new_zeros(x_pose_lateral.shape)
        if hasattr(self.pose_path, 'layer2_lateral'):
            feat = x_rgb.detach() if self.pose_detach else x_rgb
            x_rgb_lateral = self.pose_path.layer2_lateral(feat)
            if pose_drop_path:
                x_rgb_lateral = x_rgb_lateral.new_zeros(x_rgb_lateral.shape)
        if hasattr(self.rgb_path, 'layer3_lateral'):
            x_rgb = torch.cat((x_rgb, x_pose_lateral), dim=1)
        if hasattr(self.pose_path, 'layer2_lateral'):
            x_pose = torch.cat((x_pose, x_rgb_lateral), dim=1)
        x_rgb = self.rgb_path.layer4(x_rgb)
        x_pose = self.pose_path.layer3(x_pose)
        return x_rgb, x_pose

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        self.training = True

    def eval(self):
        super().eval()
        self.training = False


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
        width *= multiplier
        min_width = min_width or divisor
        width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
        if width_out < 0.9 * width:
            width_out += divisor
        return int(width_out)

    def forward(self, x):
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
    """

    def __init__(self, inplanes, planes, outplanes, spatial_stride=1, downsample=None, se_ratio=None, use_swish=True, conv_cfg=dict(type='Conv3d'), norm_cfg=dict(type='BN3d'), act_cfg=dict(type='ReLU')):
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
        self.conv1 = ConvModule(in_channels=inplanes, out_channels=planes, kernel_size=1, stride=1, padding=0, bias=False, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv2 = ConvModule(in_channels=planes, out_channels=planes, kernel_size=3, stride=(1, self.spatial_stride, self.spatial_stride), padding=1, groups=planes, bias=False, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=None)
        self.swish = Swish() if self.use_swish else nn.Identity()
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
        zero_init_residual (bool):
            Whether to use zero initialization for residual block,
            Default: True.
        kwargs (dict, optional): Key arguments for "make_res_layer".
    """

    def __init__(self, gamma_w=1.0, gamma_b=2.25, gamma_d=2.2, pretrained=None, in_channels=3, base_channels=24, num_stages=4, stage_blocks=(1, 2, 5, 3), spatial_strides=(2, 2, 2, 2), frozen_stages=-1, se_style='half', se_ratio=1 / 16, use_swish=True, conv_cfg=dict(type='Conv3d'), norm_cfg=dict(type='BN3d', requires_grad=True), act_cfg=dict(type='ReLU', inplace=True), norm_eval=False, zero_init_residual=True, **kwargs):
        super().__init__()
        self.gamma_w = gamma_w
        self.gamma_b = gamma_b
        self.gamma_d = gamma_d
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.stage_blocks = stage_blocks
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
            res_layer = self.make_res_layer(self.block, self.layer_inplanes, inplanes, planes, num_blocks, spatial_stride=spatial_stride, se_style=self.se_style, se_ratio=self.se_ratio, use_swish=self.use_swish, norm_cfg=self.norm_cfg, conv_cfg=self.conv_cfg, act_cfg=self.act_cfg, **kwargs)
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

    def make_res_layer(self, block, layer_inplanes, inplanes, planes, blocks, spatial_stride=1, se_style='half', se_ratio=None, use_swish=True, norm_cfg=None, act_cfg=None, conv_cfg=None, **kwargs):
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
        layers.append(block(layer_inplanes, planes, inplanes, spatial_stride=spatial_stride, downsample=downsample, se_ratio=se_ratio if use_se[0] else None, use_swish=use_swish, norm_cfg=norm_cfg, conv_cfg=conv_cfg, act_cfg=act_cfg, **kwargs))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, inplanes, spatial_stride=1, se_ratio=se_ratio if use_se[i] else None, use_swish=use_swish, norm_cfg=norm_cfg, conv_cfg=conv_cfg, act_cfg=act_cfg, **kwargs))
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
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, BlockX3D):
                    constant_init(m.conv3.bn, 0)
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

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


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


class unit_tcn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1, norm='BN', dropout=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1), dilation=(dilation, 1))
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1] if norm is not None else nn.Identity()
        self.drop = nn.Dropout(dropout, inplace=True)
        self.stride = stride

    def forward(self, x):
        return self.drop(self.bn(self.conv(x)))

    def init_weights(self):
        conv_init(self.conv)
        bn_init(self.bn, 1)


class mstcn(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.0, ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'], stride=1):
        super().__init__()
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.ReLU()
        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            assert isinstance(mid_channels, float) and mid_channels > 0
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels
        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels
        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == 'max':
                branches.append(nn.Sequential(nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act, nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0))))
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branch = nn.Sequential(nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act, unit_tcn(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1], norm=None))
            branches.append(branch)
        self.branches = nn.ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels
        self.transform = nn.Sequential(nn.BatchNorm2d(tin_channels), self.act, nn.Conv2d(tin_channels, out_channels, kernel_size=1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def inner_forward(self, x):
        N, C, T, V = x.shape
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)
        feat = torch.cat(branch_outs, dim=1)
        feat = self.transform(feat)
        return feat

    def forward(self, x):
        out = self.inner_forward(x)
        out = self.bn(out)
        return self.drop(out)

    def init_weights(self):
        pass


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2.0 / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


class unit_aagcn(nn.Module):

    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, attention=True):
        super(unit_aagcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.adaptive = adaptive
        self.attention = attention
        num_joints = A.shape[-1]
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))
        if self.adaptive:
            self.A = nn.Parameter(A)
            self.alpha = nn.Parameter(torch.zeros(1))
            self.conv_a = nn.ModuleList()
            self.conv_b = nn.ModuleList()
            for i in range(self.num_subset):
                self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
                self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
        else:
            self.register_buffer('A', A)
        if self.attention:
            self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
            ker_joint = num_joints if num_joints % 2 else num_joints - 1
            pad = (ker_joint - 1) // 2
            self.conv_sa = nn.Conv1d(out_channels, 1, ker_joint, padding=pad)
            rr = 2
            self.fc1c = nn.Linear(out_channels, out_channels // rr)
            self.fc2c = nn.Linear(out_channels // rr, out_channels)
        self.down = lambda x: x
        if in_channels != out_channels:
            self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        self.bn = nn.BatchNorm2d(out_channels)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-06)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)
        if self.attention:
            nn.init.constant_(self.conv_ta.weight, 0)
            nn.init.constant_(self.conv_ta.bias, 0)
            nn.init.xavier_normal_(self.conv_sa.weight)
            nn.init.constant_(self.conv_sa.bias, 0)
            nn.init.kaiming_normal_(self.fc1c.weight)
            nn.init.constant_(self.fc1c.bias, 0)
            nn.init.constant_(self.fc2c.weight, 0)
            nn.init.constant_(self.fc2c.bias, 0)

    def forward(self, x):
        N, C, T, V = x.size()
        y = None
        if self.adaptive:
            for i in range(self.num_subset):
                A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
                A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
                A1 = self.tan(torch.matmul(A1, A2) / A1.size(-1))
                A1 = self.A[i] + A1 * self.alpha
                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z
        else:
            for i in range(self.num_subset):
                A1 = self.A[i]
                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z
        y = self.relu(self.bn(y) + self.down(x))
        if self.attention:
            se = y.mean(-2)
            se1 = self.sigmoid(self.conv_sa(se))
            y = y * se1.unsqueeze(-2) + y
            se = y.mean(-1)
            se1 = self.sigmoid(self.conv_ta(se))
            y = y * se1.unsqueeze(-1) + y
            se = y.mean(-1).mean(-1)
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
        return y


class AAGCNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, **kwargs):
        super().__init__()
        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[:4] not in ['gcn_', 'tcn_']}
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'
        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn']
        gcn_type = gcn_kwargs.pop('type', 'unit_aagcn')
        assert gcn_type in ['unit_aagcn']
        self.gcn = unit_aagcn(in_channels, out_channels, A, **gcn_kwargs)
        if tcn_type == 'unit_tcn':
            self.tcn = unit_tcn(out_channels, out_channels, 9, stride=stride, **tcn_kwargs)
        elif tcn_type == 'mstcn':
            self.tcn = mstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def init_weights(self):
        self.tcn.init_weights()
        self.gcn.init_weights()

    def forward(self, x):
        return self.relu(self.tcn(self.gcn(x)) + self.residual(x))


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.eye(num_node)
    for i, j in edge:
        A[i, j] = 1
        A[j, i] = 1
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A, dim=0):
    Dl = np.sum(A, dim)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** -1
    AD = np.dot(A, Dn)
    return AD


class Graph:
    """The Graph to model the skeletons.

    Args:
        layout (str): must be one of the following candidates: 'openpose', 'nturgb+d', 'coco'. Default: 'coco'.
        mode (str): must be one of the following candidates: 'stgcn_spatial', 'spatial'. Default: 'spatial'.
        max_hop (int): the maximal distance between two connected nodes.
            Default: 1
    """

    def __init__(self, layout='coco', mode='spatial', max_hop=1, nx_node=1, num_filter=3, init_std=0.02, init_off=0.04):
        self.max_hop = max_hop
        self.layout = layout
        self.mode = mode
        self.num_filter = num_filter
        self.init_std = init_std
        self.init_off = init_off
        self.nx_node = nx_node
        assert nx_node == 1 or mode == 'random', "nx_node can be > 1 only if mode is 'random'"
        assert layout in ['openpose', 'nturgb+d', 'coco', 'handmp']
        self.get_layout(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.inward, max_hop)
        assert hasattr(self, mode), f'Do Not Exist This Mode: {mode}'
        self.A = getattr(self, mode)()

    def __str__(self):
        return self.A

    def get_layout(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self.inward = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.center = 1
        elif layout == 'nturgb+d':
            self.num_node = 25
            neighbor_base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 8), (23, 8), (24, 12), (25, 12)]
            self.inward = [(i - 1, j - 1) for i, j in neighbor_base]
            self.center = 21 - 1
        elif layout == 'coco':
            self.num_node = 17
            self.inward = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6), (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0), (1, 0), (3, 1), (2, 0), (4, 2)]
            self.center = 0
        elif layout == 'handmp':
            self.num_node = 21
            self.inward = [(1, 0), (2, 1), (3, 2), (4, 3), (5, 0), (6, 5), (7, 6), (8, 7), (9, 0), (10, 9), (11, 10), (12, 11), (13, 0), (14, 13), (15, 14), (16, 15), (17, 0), (18, 17), (19, 18), (20, 19)]
            self.center = 0
        else:
            raise ValueError(f'Do Not Exist This Layout: {layout}')
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.outward = [(j, i) for i, j in self.inward]
        self.neighbor = self.inward + self.outward

    def stgcn_spatial(self):
        adj = np.zeros((self.num_node, self.num_node))
        adj[self.hop_dis <= self.max_hop] = 1
        normalize_adj = normalize_digraph(adj)
        hop_dis = self.hop_dis
        center = self.center
        A = []
        for hop in range(self.max_hop + 1):
            a_close = np.zeros((self.num_node, self.num_node))
            a_further = np.zeros((self.num_node, self.num_node))
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if hop_dis[j, i] == hop:
                        if hop_dis[j, center] >= hop_dis[i, center]:
                            a_close[j, i] = normalize_adj[j, i]
                        else:
                            a_further[j, i] = normalize_adj[j, i]
            A.append(a_close)
            if hop > 0:
                A.append(a_further)
        return np.stack(A)

    def spatial(self):
        Iden = edge2mat(self.self_link, self.num_node)
        In = normalize_digraph(edge2mat(self.inward, self.num_node))
        Out = normalize_digraph(edge2mat(self.outward, self.num_node))
        A = np.stack((Iden, In, Out))
        return A

    def binary_adj(self):
        A = edge2mat(self.inward + self.outward, self.num_node)
        return A[None]

    def random(self):
        num_node = self.num_node * self.nx_node
        return np.random.randn(self.num_filter, num_node, num_node) * self.init_std + self.init_off


class AAGCN(nn.Module):

    def __init__(self, graph_cfg, in_channels=3, base_channels=64, data_bn_type='MVC', num_person=2, num_stages=10, inflate_stages=[5, 8], down_stages=[5, 8], pretrained=None, **kwargs):
        super().__init__()
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        self.kwargs = kwargs
        assert data_bn_type in ['MVC', 'VC', None]
        self.data_bn_type = data_bn_type
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_person = num_person
        self.num_stages = num_stages
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages
        if self.data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif self.data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()
        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)
        modules = []
        if self.in_channels != self.base_channels:
            modules = [AAGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]
        for i in range(2, num_stages + 1):
            in_channels = base_channels
            out_channels = base_channels * (1 + (i in inflate_stages))
            stride = 1 + (i in down_stages)
            modules.append(AAGCNBlock(base_channels, out_channels, A.clone(), stride=stride, **lw_kwargs[i - 1]))
            base_channels = out_channels
        if self.in_channels == self.base_channels:
            self.num_stages -= 1
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained

    def init_weights(self):
        bn_init(self.data_bn, 1)
        for module in self.gcn:
            module.init_weights()
        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for i in range(self.num_stages):
            x = self.gcn[i](x)
        x = x.reshape((N, M) + x.shape[1:])
        return x


class MSTCN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilations=[1, 2, 3, 4], residual=True, act_cfg=dict(type='ReLU'), tcn_dropout=0):
        super().__init__()
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        branch_channels_rem = out_channels - branch_channels * (self.num_branches - 1)
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        self.branches = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0), nn.BatchNorm2d(branch_channels), build_activation_layer(act_cfg), unit_tcn(branch_channels, branch_channels, kernel_size=ks, stride=stride, dilation=dilation)) for ks, dilation in zip(kernel_size, dilations)])
        self.branches.append(nn.Sequential(nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0), nn.BatchNorm2d(branch_channels), build_activation_layer(act_cfg), nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)), nn.BatchNorm2d(branch_channels)))
        self.branches.append(nn.Sequential(nn.Conv2d(in_channels, branch_channels_rem, kernel_size=1, padding=0, stride=(stride, 1)), nn.BatchNorm2d(branch_channels_rem)))
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)
        self.act = build_activation_layer(act_cfg)
        self.drop = nn.Dropout(tcn_dropout)

    def forward(self, x):
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)
        out = torch.cat(branch_outs, dim=1)
        out += res
        out = self.act(out)
        out = self.drop(out)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, _BatchNorm):
                bn_init(m, 1)


class CTRGC(nn.Module):

    def __init__(self, in_channels, out_channels, rel_reduction=8):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels <= 16:
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        self.init_weights()

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A[None, None] if A is not None else 0)
        x1 = torch.einsum('ncuv,nctu->nctv', x1, x3)
        return x1

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)


class unit_ctrgcn(nn.Module):

    def __init__(self, in_channels, out_channels, A):
        super(unit_ctrgcn, self).__init__()
        inter_channels = out_channels // 4
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))
        if in_channels != out_channels:
            self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.down = lambda x: x
        self.A = nn.Parameter(A.clone())
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = None
        for i in range(self.num_subset):
            z = self.convs[i](x, self.A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-06)


class CTRGCNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, kernel_size=5, dilations=[1, 2], tcn_dropout=0):
        super(CTRGCNBlock, self).__init__()
        self.gcn1 = unit_ctrgcn(in_channels, out_channels, A)
        self.tcn1 = MSTCN(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations, residual=False, tcn_dropout=tcn_dropout)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y

    def init_weights(self):
        self.tcn1.init_weights()
        self.gcn1.init_weights()


class CTRGCN(nn.Module):

    def __init__(self, graph_cfg, in_channels=3, base_channels=64, num_stages=10, inflate_stages=[5, 8], down_stages=[5, 8], pretrained=None, num_person=2, **kwargs):
        super(CTRGCN, self).__init__()
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        self.num_person = num_person
        self.base_channels = base_channels
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'tcn_dropout'}
        modules = [CTRGCNBlock(in_channels, base_channels, A.clone(), residual=False, **kwargs0)]
        for i in range(2, num_stages + 1):
            in_channels = base_channels
            out_channels = base_channels * (1 + (i in inflate_stages))
            stride = 1 + (i in down_stages)
            modules.append(CTRGCNBlock(base_channels, out_channels, A.clone(), stride=stride, **kwargs))
            base_channels = out_channels
        self.net = nn.ModuleList(modules)

    def init_weights(self):
        for module in self.net:
            module.init_weights()

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.data_bn(x.view(N, M * V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for gcn in self.net:
            x = gcn(x)
        x = x.reshape((N, M) + x.shape[1:])
        return x


class dggcn(nn.Module):

    def __init__(self, in_channels, out_channels, A, ratio=0.25, ctr='T', ada='T', subset_wise=False, ada_act='softmax', ctr_act='tanh', norm='BN', act='ReLU'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        num_subsets = A.size(0)
        self.num_subsets = num_subsets
        self.ctr = ctr
        self.ada = ada
        self.ada_act = ada_act
        self.ctr_act = ctr_act
        assert ada_act in ['tanh', 'relu', 'sigmoid', 'softmax']
        assert ctr_act in ['tanh', 'relu', 'sigmoid', 'softmax']
        self.subset_wise = subset_wise
        assert self.ctr in [None, 'NA', 'T']
        assert self.ada in [None, 'NA', 'T']
        if ratio is None:
            ratio = 1 / self.num_subsets
        self.ratio = ratio
        mid_channels = int(ratio * out_channels)
        self.mid_channels = mid_channels
        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        self.act = build_activation_layer(self.act_cfg)
        self.A = nn.Parameter(A.clone())
        self.pre = nn.Sequential(nn.Conv2d(in_channels, mid_channels * num_subsets, 1), build_norm_layer(self.norm_cfg, mid_channels * num_subsets)[1], self.act)
        self.post = nn.Conv2d(mid_channels * num_subsets, out_channels, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-2)
        self.alpha = nn.Parameter(torch.zeros(self.num_subsets))
        self.beta = nn.Parameter(torch.zeros(self.num_subsets))
        if self.ada or self.ctr:
            self.conv1 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)
            self.conv2 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)
        if in_channels != out_channels:
            self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), build_norm_layer(self.norm_cfg, out_channels)[1])
        else:
            self.down = lambda x: x
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape
        res = self.down(x)
        A = self.A
        A = A[None, :, None, None]
        pre_x = self.pre(x).reshape(n, self.num_subsets, self.mid_channels, t, v)
        x1, x2 = None, None
        if self.ctr is not None or self.ada is not None:
            tmp_x = x
            if not (self.ctr == 'NA' or self.ada == 'NA'):
                tmp_x = tmp_x.mean(dim=-2, keepdim=True)
            x1 = self.conv1(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)
            x2 = self.conv2(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)
        if self.ctr is not None:
            diff = x1.unsqueeze(-1) - x2.unsqueeze(-2)
            ada_graph = getattr(self, self.ctr_act)(diff)
            if self.subset_wise:
                ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.alpha)
            else:
                ada_graph = ada_graph * self.alpha[0]
            A = ada_graph + A
        if self.ada is not None:
            ada_graph = torch.einsum('nkctv,nkctw->nktvw', x1, x2)[:, :, None]
            ada_graph = getattr(self, self.ada_act)(ada_graph)
            if self.subset_wise:
                ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.beta)
            else:
                ada_graph = ada_graph * self.beta[0]
            A = ada_graph + A
        if self.ctr is not None or self.ada is not None:
            assert len(A.shape) == 6
            if A.shape[2] == 1 and A.shape[3] == 1:
                A = A.squeeze(2).squeeze(2)
                x = torch.einsum('nkctv,nkvw->nkctw', pre_x, A).contiguous()
            elif A.shape[2] == 1:
                A = A.squeeze(2)
                x = torch.einsum('nkctv,nktvw->nkctw', pre_x, A).contiguous()
            elif A.shape[3] == 1:
                A = A.squeeze(3)
                x = torch.einsum('nkctv,nkcvw->nkctw', pre_x, A).contiguous()
            else:
                x = torch.einsum('nkctv,nkctvw->nkctw', pre_x, A).contiguous()
        else:
            A = A.squeeze()
            assert len(A.shape) in [2, 3] and A.shape[-2] == A.shape[-1]
            if len(A.shape) == 2:
                A = A[None]
            x = torch.einsum('nkctv,kvw->nkctw', pre_x, A).contiguous()
        x = x.reshape(n, -1, t, v)
        x = self.post(x)
        return self.act(self.bn(x) + res)


class dgmstcn(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None, num_joints=25, dropout=0.0, ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'], stride=1):
        super().__init__()
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.ReLU()
        self.num_joints = num_joints
        self.add_coeff = nn.Parameter(torch.zeros(self.num_joints))
        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            assert isinstance(mid_channels, float) and mid_channels > 0
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels
        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels
        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == 'max':
                branches.append(nn.Sequential(nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act, nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0))))
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branch = nn.Sequential(nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act, unit_tcn(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1], norm=None))
            branches.append(branch)
        self.branches = nn.ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels
        self.transform = nn.Sequential(nn.BatchNorm2d(tin_channels), self.act, nn.Conv2d(tin_channels, out_channels, kernel_size=1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def inner_forward(self, x):
        N, C, T, V = x.shape
        x = torch.cat([x, x.mean(-1, keepdim=True)], -1)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)
        out = torch.cat(branch_outs, dim=1)
        local_feat = out[..., :V]
        global_feat = out[..., V]
        global_feat = torch.einsum('nct,v->nctv', global_feat, self.add_coeff[:V])
        feat = local_feat + global_feat
        feat = self.transform(feat)
        return feat

    def forward(self, x):
        out = self.inner_forward(x)
        out = self.bn(out)
        return self.drop(out)


class DGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, **kwargs):
        super().__init__()
        common_args = ['act', 'norm', 'g1x1']
        for arg in common_args:
            if arg in kwargs:
                value = kwargs.pop(arg)
                kwargs['tcn_' + arg] = value
                kwargs['gcn_' + arg] = value
        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[1:4] != 'cn_'}
        assert len(kwargs) == 0
        self.gcn = dggcn(in_channels, out_channels, A, **gcn_kwargs)
        self.tcn = dgmstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x = self.tcn(self.gcn(x, A)) + res
        return self.relu(x)


EPS = 0.0001


class DGSTGCN(nn.Module):

    def __init__(self, graph_cfg, in_channels=3, base_channels=64, ch_ratio=2, num_stages=10, inflate_stages=[5, 8], down_stages=[5, 8], data_bn_type='VC', num_person=2, pretrained=None, **kwargs):
        super().__init__()
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type
        self.kwargs = kwargs
        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()
        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)
        lw_kwargs[0].pop('g1x1', None)
        lw_kwargs[0].pop('gcn_g1x1', None)
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages
        modules = []
        if self.in_channels != self.base_channels:
            modules = [DGBlock(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]
        inflate_times = 0
        down_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(DGBlock(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i - 1]))
            down_times += i in down_stages
        if self.in_channels == self.base_channels:
            num_stages -= 1
        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained

    def init_weights(self):
        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for i in range(self.num_stages):
            x = self.gcn[i](x)
        x = x.reshape((N, M) + x.shape[1:])
        return x


class MLP(nn.Module):

    def __init__(self, in_channels, out_channels, act_cfg=dict(type='ReLU'), dropout=0):
        super().__init__()
        channels = [in_channels] + out_channels
        self.layers = nn.ModuleList()
        for i in range(1, len(channels)):
            if dropout > 0.001:
                self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.Conv2d(channels[i - 1], channels[i], kernel_size=1))
            self.layers.append(nn.BatchNorm2d(channels[i]))
            if act_cfg:
                self.layers.append(build_activation_layer(act_cfg))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def k_adjacency(A, k, with_self=False, self_factor=1):
    if isinstance(A, torch.Tensor):
        A = A.data.cpu().numpy()
    assert isinstance(A, np.ndarray)
    Iden = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return Iden
    Ak = np.minimum(np.linalg.matrix_power(A + Iden, k), 1) - np.minimum(np.linalg.matrix_power(A + Iden, k - 1), 1)
    if with_self:
        Ak += self_factor * Iden
    return Ak


class MSGCN(nn.Module):

    def __init__(self, num_scales, in_channels, out_channels, A, dropout=0, act_cfg=dict(type='ReLU')):
        super().__init__()
        self.num_scales = num_scales
        A_powers = [k_adjacency(A, k, with_self=True) for k in range(num_scales)]
        A_powers = np.stack([normalize_digraph(g) for g in A_powers])
        self.register_buffer('A', torch.Tensor(A_powers))
        self.PA = nn.Parameter(self.A.clone())
        nn.init.uniform_(self.PA, -1e-06, 1e-06)
        self.mlp = MLP(in_channels * num_scales, [out_channels], dropout=dropout, act_cfg=act_cfg)

    def forward(self, x):
        N, C, T, V = x.shape
        A = self.A
        A = A + self.PA
        support = torch.einsum('kvu,nctv->nkctu', A, x)
        support = support.reshape(N, self.num_scales * C, T, V)
        out = self.mlp(support)
        return out


class ST_MSGCN(nn.Module):

    def __init__(self, in_channels, out_channels, A, num_scales, window_size, residual=False, dropout=0, act_cfg=dict(type='ReLU')):
        super().__init__()
        self.num_scales = num_scales
        self.window_size = window_size
        A = self.build_st_graph(A, window_size)
        A_scales = [k_adjacency(A, k, with_self=True) for k in range(num_scales)]
        A_scales = np.stack([normalize_digraph(g) for g in A_scales])
        self.register_buffer('A', torch.Tensor(A_scales))
        self.V = len(A)
        self.PA = nn.Parameter(self.A.clone())
        nn.init.uniform_(self.PA, -1e-06, 1e-06)
        self.mlp = MLP(in_channels * num_scales, [out_channels], dropout=dropout, act_cfg=act_cfg)
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = MLP(in_channels, [out_channels], act_cfg=None)
        self.act = build_activation_layer(act_cfg)

    def build_st_graph(self, A, window_size):
        if not isinstance(A, np.ndarray):
            A = A.data.cpu().numpy()
        assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
        V = len(A)
        A_with_I = A + np.eye(V, dtype=A.dtype)
        A_large = np.tile(A_with_I, (window_size, window_size)).copy()
        return A_large

    def forward(self, x):
        N, C, T, V = x.shape
        A = self.A + self.PA
        res = self.residual(x)
        agg = torch.einsum('kvu,nctv->nkctu', A, x)
        agg = agg.reshape(N, self.num_scales * C, T, V)
        out = self.mlp(agg)
        out += res
        return self.act(out)


class UnfoldTemporalWindows(nn.Module):

    def __init__(self, window_size, window_stride, window_dilation=1):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_dilation = window_dilation
        self.padding = (window_size + (window_size - 1) * (window_dilation - 1) - 1) // 2
        self.unfold = nn.Unfold(kernel_size=(self.window_size, 1), dilation=(self.window_dilation, 1), stride=(self.window_stride, 1), padding=(self.padding, 0))

    def forward(self, x):
        N, C, T, V = x.shape
        x = self.unfold(x)
        x = x.reshape(N, C, self.window_size, -1, V).permute(0, 1, 3, 2, 4).contiguous()
        x = x.reshape(N, C, -1, self.window_size * V)
        return x


class MSG3DBlock(nn.Module):

    def __init__(self, in_channels, out_channels, A, num_scales, window_size, window_stride, window_dilation, embed_factor=1, activation='relu'):
        super().__init__()
        self.window_size = window_size
        self.out_channels = out_channels
        self.embed_channels_in = self.embed_channels_out = out_channels // embed_factor
        if embed_factor == 1:
            self.in1x1 = nn.Identity()
            self.embed_channels_in = self.embed_channels_out = in_channels
            if in_channels == 3:
                self.embed_channels_out = out_channels
        else:
            self.in1x1 = MLP(in_channels, [self.embed_channels_in])
        self.gcn3d = nn.Sequential(UnfoldTemporalWindows(window_size, window_stride, window_dilation), ST_MSGCN(in_channels=self.embed_channels_in, out_channels=self.embed_channels_out, A=A, num_scales=num_scales, window_size=window_size))
        self.out_conv = nn.Conv3d(self.embed_channels_out, out_channels, kernel_size=(1, self.window_size, 1))
        self.out_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        N, _, T, V = x.shape
        x = self.in1x1(x)
        x = self.gcn3d(x)
        x = x.reshape(N, self.embed_channels_out, -1, self.window_size, V)
        x = self.out_conv(x).squeeze(dim=3)
        x = self.out_bn(x)
        return x


class MW_MSG3DBlock(nn.Module):

    def __init__(self, in_channels, out_channels, A, num_scales, window_sizes=[3, 5], window_stride=1, window_dilations=[1, 1]):
        super().__init__()
        self.gcn3d = nn.ModuleList([MSG3DBlock(in_channels, out_channels, A, num_scales, window_size, window_stride, window_dilation) for window_size, window_dilation in zip(window_sizes, window_dilations)])

    def forward(self, x):
        out_sum = 0
        for gcn3d in self.gcn3d:
            out_sum += gcn3d(x)
        return out_sum


class MSG3D(nn.Module):

    def __init__(self, graph_cfg, in_channels=3, base_channels=96, num_gcn_scales=13, num_g3d_scales=6, num_person=2, tcn_dropout=0):
        super().__init__()
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A[0], dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        self.num_point = A.shape[-1]
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.data_bn = nn.BatchNorm1d(self.num_point * in_channels * num_person)
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.gcn3d1 = MW_MSG3DBlock(3, c1, A, num_g3d_scales, window_stride=1)
        self.sgcn1 = nn.Sequential(MSGCN(num_gcn_scales, 3, c1, A), MSTCN(c1, c1), MSTCN(c1, c1))
        self.sgcn1[-1].act = nn.Identity()
        self.tcn1 = MSTCN(c1, c1, tcn_dropout=tcn_dropout)
        self.gcn3d2 = MW_MSG3DBlock(c1, c2, A, num_g3d_scales, window_stride=2)
        self.sgcn2 = nn.Sequential(MSGCN(num_gcn_scales, c1, c1, A), MSTCN(c1, c2, stride=2), MSTCN(c2, c2))
        self.sgcn2[-1].act = nn.Identity()
        self.tcn2 = MSTCN(c2, c2, tcn_dropout=tcn_dropout)
        self.gcn3d3 = MW_MSG3DBlock(c2, c3, A, num_g3d_scales, window_stride=2)
        self.sgcn3 = nn.Sequential(MSGCN(num_gcn_scales, c2, c2, A), MSTCN(c2, c3, stride=2), MSTCN(c3, c3))
        self.sgcn3[-1].act = nn.Identity()
        self.tcn3 = MSTCN(c3, c3, tcn_dropout=tcn_dropout)

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous().reshape(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.reshape(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()
        x = F.relu(self.sgcn1(x) + self.gcn3d1(x), inplace=True)
        x = self.tcn1(x)
        x = F.relu(self.sgcn2(x) + self.gcn3d2(x), inplace=True)
        x = self.tcn2(x)
        x = F.relu(self.sgcn3(x) + self.gcn3d3(x), inplace=True)
        x = self.tcn3(x)
        return x.reshape((N, M) + x.shape[1:])

    def init_weights(self):
        pass


class unit_sgn(nn.Module):

    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, A):
        x1 = x.permute(0, 2, 3, 1).contiguous()
        x1 = A.matmul(x1).permute(0, 3, 1, 2).contiguous()
        return self.relu(self.bn(self.conv(x1) + self.residual(x)))


class SGN(nn.Module):

    def __init__(self, in_channels=3, base_channels=64, num_joints=25, T=30, bias=True):
        super(SGN, self).__init__()
        self.T = T
        self.num_joints = num_joints
        self.base_channel = base_channels
        self.joint_bn = nn.BatchNorm1d(in_channels * num_joints)
        self.motion_bn = nn.BatchNorm1d(in_channels * num_joints)
        self.t_embed = self.embed_mlp(self.T, base_channels * 4, base_channels, bias=bias)
        self.s_embed = self.embed_mlp(self.num_joints, base_channels, base_channels, bias=bias)
        self.joint_embed = self.embed_mlp(in_channels, base_channels, base_channels, bias=bias)
        self.motion_embed = self.embed_mlp(in_channels, base_channels, base_channels, bias=bias)
        self.compute_A1 = ConvModule(base_channels * 2, base_channels * 4, kernel_size=1, bias=bias)
        self.compute_A2 = ConvModule(base_channels * 2, base_channels * 4, kernel_size=1, bias=bias)
        self.tcn = nn.Sequential(nn.AdaptiveMaxPool2d((20, 1)), ConvModule(base_channels * 4, base_channels * 4, kernel_size=(3, 1), padding=(1, 0), bias=bias, norm_cfg=dict(type='BN2d')), nn.Dropout(0.2), ConvModule(base_channels * 4, base_channels * 8, kernel_size=1, bias=bias, norm_cfg=dict(type='BN2d')))
        self.gcn1 = unit_sgn(base_channels * 2, base_channels * 2, bias=bias)
        self.gcn2 = unit_sgn(base_channels * 2, base_channels * 4, bias=bias)
        self.gcn3 = unit_sgn(base_channels * 4, base_channels * 4, bias=bias)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
        nn.init.constant_(self.gcn1.conv.weight, 0)
        nn.init.constant_(self.gcn2.conv.weight, 0)
        nn.init.constant_(self.gcn3.conv.weight, 0)

    def embed_mlp(self, in_channels, out_channels, mid_channels=64, bias=False):
        return nn.Sequential(ConvModule(in_channels, mid_channels, kernel_size=1, bias=bias), ConvModule(mid_channels, out_channels, kernel_size=1, bias=bias))

    def compute_A(self, x):
        A1 = self.compute_A1(x).permute(0, 2, 3, 1).contiguous()
        A2 = self.compute_A2(x).permute(0, 2, 1, 3).contiguous()
        A = A1.matmul(A2)
        return nn.Softmax(dim=-1)(A)

    def forward(self, joint):
        N, M, T, V, C = joint.shape
        joint = joint.reshape(N * M, T, V, C)
        joint = joint.permute(0, 3, 2, 1).contiguous()
        motion = torch.diff(joint, dim=3, append=torch.zeros(N * M, C, V, 1))
        joint = self.joint_bn(joint.view(N * M, C * V, T))
        motion = self.motion_bn(motion.view(N * M, C * V, T))
        joint = joint.view(N * M, C, V, T).permute(0, 1, 3, 2).contiguous()
        motion = motion.view(N * M, C, V, T).permute(0, 1, 3, 2).contiguous()
        joint_embed = self.joint_embed(joint)
        motion_embed = self.motion_embed(motion)
        t_code = torch.eye(T)
        t_code = t_code[None, :, None].repeat(N * M, 1, V, 1)
        s_code = torch.eye(V)
        s_code = s_code[None, ..., None].repeat(N * M, 1, 1, T)
        t_embed = self.t_embed(t_code).permute(0, 1, 3, 2).contiguous()
        s_embed = self.s_embed(s_code).permute(0, 1, 3, 2).contiguous()
        x = torch.cat([joint_embed + motion_embed, s_embed], 1)
        A = self.compute_A(x)
        for gcn in [self.gcn1, self.gcn2, self.gcn3]:
            x = gcn(x, A)
        x = x + t_embed
        x = self.tcn(x)
        return x.reshape((N, M) + x.shape[1:])


class unit_gcn(nn.Module):

    def __init__(self, in_channels, out_channels, A, adaptive='importance', conv_pos='pre', with_res=False, norm='BN', act='ReLU'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subsets = A.size(0)
        assert adaptive in [None, 'init', 'offset', 'importance']
        self.adaptive = adaptive
        assert conv_pos in ['pre', 'post']
        self.conv_pos = conv_pos
        self.with_res = with_res
        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]
        self.act = build_activation_layer(self.act_cfg)
        if self.adaptive == 'init':
            self.A = nn.Parameter(A.clone())
        else:
            self.register_buffer('A', A)
        if self.adaptive in ['offset', 'importance']:
            self.PA = nn.Parameter(A.clone())
            if self.adaptive == 'offset':
                nn.init.uniform_(self.PA, -1e-06, 1e-06)
            elif self.adaptive == 'importance':
                nn.init.constant_(self.PA, 1)
        if self.conv_pos == 'pre':
            self.conv = nn.Conv2d(in_channels, out_channels * A.size(0), 1)
        elif self.conv_pos == 'post':
            self.conv = nn.Conv2d(A.size(0) * in_channels, out_channels, 1)
        if self.with_res:
            if in_channels != out_channels:
                self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), build_norm_layer(self.norm_cfg, out_channels)[1])
            else:
                self.down = lambda x: x

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape
        res = self.down(x) if self.with_res else 0
        A_switch = {None: self.A, 'init': self.A}
        if hasattr(self, 'PA'):
            A_switch.update({'offset': self.A + self.PA, 'importance': self.A * self.PA})
        A = A_switch[self.adaptive]
        if self.conv_pos == 'pre':
            x = self.conv(x)
            x = x.view(n, self.num_subsets, -1, t, v)
            x = torch.einsum('nkctv,kvw->nctw', (x, A)).contiguous()
        elif self.conv_pos == 'post':
            x = torch.einsum('nctv,kvw->nkctw', (x, A)).contiguous()
            x = x.view(n, -1, t, v)
            x = self.conv(x)
        return self.act(self.bn(x) + res)

    def init_weights(self):
        pass


class STGCNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, **kwargs):
        super().__init__()
        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[:4] not in ['gcn_', 'tcn_']}
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'
        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn']
        gcn_type = gcn_kwargs.pop('type', 'unit_gcn')
        assert gcn_type in ['unit_gcn']
        self.gcn = unit_gcn(in_channels, out_channels, A, **gcn_kwargs)
        if tcn_type == 'unit_tcn':
            self.tcn = unit_tcn(out_channels, out_channels, 9, stride=stride, **tcn_kwargs)
        elif tcn_type == 'mstcn':
            self.tcn = mstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x = self.tcn(self.gcn(x, A)) + res
        return self.relu(x)


class STGCN(nn.Module):

    def __init__(self, graph_cfg, in_channels=3, base_channels=64, data_bn_type='VC', ch_ratio=2, num_person=2, num_stages=10, inflate_stages=[5, 8], down_stages=[5, 8], pretrained=None, **kwargs):
        super().__init__()
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type
        self.kwargs = kwargs
        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()
        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages
        modules = []
        if self.in_channels != self.base_channels:
            modules = [STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]
        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(STGCNBlock(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i - 1]))
        if self.in_channels == self.base_channels:
            num_stages -= 1
        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained

    def init_weights(self):
        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for i in range(self.num_stages):
            x = self.gcn[i](x)
        x = x.reshape((N, M) + x.shape[1:])
        return x


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def top_k_accuracy(scores, labels, topk=(1,)):
    """Calculate top k accuracy score.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

    Returns:
        list[float]: Top k accuracy score for each k.
    """
    res = []
    labels = np.array(labels)[:, np.newaxis]
    for k in topk:
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res.append(topk_acc_score)
    return res


class BaseHead(nn.Module, metaclass=ABCMeta):
    """Base class for head.

    All Head should subclass it.
    All subclass should overwrite:
    - Methods:``init_weights``, initializing weights in some modules.
    - Methods:``forward``, supporting to forward both for training and testing.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss', loss_weight=1.0).
        multi_class (bool): Determines whether it is a multi-class
            recognition task. Default: False.
        label_smooth_eps (float): Epsilon used in label smooth.
            Reference: arxiv.org/abs/1906.02629. Default: 0.
    """

    def __init__(self, num_classes, in_channels, loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0), multi_class=False, label_smooth_eps=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.loss_cls = build_loss(loss_cls)
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps

    @abstractmethod
    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""

    @abstractmethod
    def forward(self, x):
        """Defines the computation performed at every call."""

    def loss(self, cls_score, label, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``label``.

        Args:
            cls_score (torch.Tensor): The output of the model.
            label (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'top1_acc', 'top5_acc'(optional).
        """
        losses = dict()
        if label.shape == torch.Size([]):
            label = label.unsqueeze(0)
        elif label.dim() == 1 and label.size()[0] == self.num_classes and cls_score.size()[0] == 1:
            label = label.unsqueeze(0)
        if not self.multi_class and cls_score.size() != label.size():
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(), label.detach().cpu().numpy(), (1, 5))
            losses['top1_acc'] = torch.tensor(top_k_acc[0], device=cls_score.device)
            losses['top5_acc'] = torch.tensor(top_k_acc[1], device=cls_score.device)
        elif self.multi_class and self.label_smooth_eps != 0:
            label = (1 - self.label_smooth_eps) * label + self.label_smooth_eps / self.num_classes
        loss_cls = self.loss_cls(cls_score, label, **kwargs)
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls
        return losses


class RGBPoseHead(BaseHead):
    """The classification head for Slowfast.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (tuple[int]): Number of channels in input feature.
        loss_cls (dict): Config for building loss. Default: dict(type='CrossEntropyLoss').
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initializ the head.
    """

    def __init__(self, num_classes, in_channels, loss_cls=dict(type='CrossEntropyLoss'), loss_components=['rgb', 'pose'], loss_weights=1.0, dropout=0.5, init_std=0.01, **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        if isinstance(dropout, float):
            dropout = {'rgb': dropout, 'pose': dropout}
        assert isinstance(dropout, dict)
        self.dropout = dropout
        self.init_std = init_std
        self.in_channels = in_channels
        self.loss_components = loss_components
        if isinstance(loss_weights, float):
            loss_weights = [loss_weights] * len(loss_components)
        assert len(loss_weights) == len(loss_components)
        self.loss_weights = loss_weights
        self.dropout_rgb = nn.Dropout(p=self.dropout['rgb'])
        self.dropout_pose = nn.Dropout(p=self.dropout['pose'])
        self.fc_rgb = nn.Linear(in_channels[0], num_classes)
        self.fc_pose = nn.Linear(in_channels[1], num_classes)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_rgb, std=self.init_std)
        normal_init(self.fc_pose, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        x_rgb, x_pose = self.avg_pool(x[0]), self.avg_pool(x[1])
        x_rgb = x_rgb.view(x_rgb.size(0), -1)
        x_pose = x_pose.view(x_pose.size(0), -1)
        x_rgb = self.dropout_rgb(x_rgb)
        x_pose = self.dropout_pose(x_pose)
        cls_scores = {}
        cls_scores['rgb'] = self.fc_rgb(x_rgb)
        cls_scores['pose'] = self.fc_pose(x_pose)
        return cls_scores


class SimpleHead(BaseHead):
    """ A simple classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss. Default: dict(type='CrossEntropyLoss')
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self, num_classes, in_channels, loss_cls=dict(type='CrossEntropyLoss'), dropout=0.5, init_std=0.01, mode='3D', **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.dropout_ratio = dropout
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        assert mode in ['3D', 'GCN', '2D']
        self.mode = mode
        self.in_c = in_channels
        self.fc_cls = nn.Linear(self.in_c, num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        if isinstance(x, list):
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)
        if len(x.shape) != 2:
            if self.mode == '2D':
                assert len(x.shape) == 5
                N, S, C, H, W = x.shape
                pool = nn.AdaptiveAvgPool2d(1)
                x = x.reshape(N * S, C, H, W)
                x = pool(x)
                x = x.reshape(N, S, C)
                x = x.mean(dim=1)
            if self.mode == '3D':
                pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(x, tuple) or isinstance(x, list):
                    x = torch.cat(x, dim=1)
                x = pool(x)
                x = x.view(x.shape[:2])
            if self.mode == 'GCN':
                pool = nn.AdaptiveAvgPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V)
                x = pool(x)
                x = x.reshape(N, M, C)
                x = x.mean(dim=1)
        assert x.shape[1] == self.in_c
        if self.dropout is not None:
            x = self.dropout(x)
        cls_score = self.fc_cls(x)
        return cls_score


class I3DHead(SimpleHead):

    def __init__(self, num_classes, in_channels, loss_cls=dict(type='CrossEntropyLoss'), dropout=0.5, init_std=0.01, **kwargs):
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, dropout=dropout, init_std=init_std, mode='3D', **kwargs)


class SlowFastHead(I3DHead):
    pass


class GCNHead(SimpleHead):

    def __init__(self, num_classes, in_channels, loss_cls=dict(type='CrossEntropyLoss'), dropout=0.0, init_std=0.01, **kwargs):
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, dropout=dropout, init_std=init_std, mode='GCN', **kwargs)


class TSNHead(BaseHead):

    def __init__(self, num_classes, in_channels, loss_cls=dict(type='CrossEntropyLoss'), dropout=0.5, init_std=0.01, **kwargs):
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, dropout=dropout, init_std=init_std, mode='2D', **kwargs)


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


class CrossEntropyLoss(BaseWeightedLoss):
    """Cross Entropy Loss.

    Support two kinds of labels and their corresponding loss type. It's worth
    mentioning that loss type will be detected by the shape of ``cls_score``
    and ``label``.
    1) Hard label: This label is an integer array and all of the elements are
        in the range [0, num_classes - 1]. This label's shape should be
        ``cls_score``'s shape with the `num_classes` dimension removed.
    2) Soft label(probablity distribution over classes): This label is a
        probability distribution and all of the elements are in the range
        [0, 1]. This label's shape must be the same as ``cls_score``. For now,
        only 2-dim soft label is supported.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Default: None.
    """

    def __init__(self, loss_weight=1.0, class_weight=None):
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)

    def _forward(self, cls_score, label, **kwargs):
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
            Default: 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Default: None.
    """

    def __init__(self, loss_weight=1.0, class_weight=None):
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)

    def _forward(self, cls_score, label, **kwargs):
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


class BaseRecognizer(nn.Module, metaclass=ABCMeta):
    """Base class for recognizers.

    All recognizers should subclass it.
    All subclass should overwrite:

    - Methods:``forward_train``, supporting to forward when training.
    - Methods:``forward_test``, supporting to forward when testing.

    Args:
        backbone (dict): Backbone modules to extract feature.
        cls_head (dict | None): Classification head to process feature. Default: None.
        train_cfg (dict): Config for training. Default: {}.
        test_cfg (dict): Config for testing. Default: {}.
    """

    def __init__(self, backbone, cls_head=None, train_cfg=dict(), test_cfg=dict()):
        super().__init__()
        self.backbone = builder.build_backbone(backbone)
        self.cls_head = builder.build_head(cls_head) if cls_head else None
        if train_cfg is None:
            train_cfg = dict()
        if test_cfg is None:
            test_cfg = dict()
        assert isinstance(train_cfg, dict)
        assert isinstance(test_cfg, dict)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.max_testing_views = test_cfg.get('max_testing_views', None)
        self.init_weights()

    @property
    def with_cls_head(self):
        """bool: whether the recognizer has a cls_head"""
        return hasattr(self, 'cls_head') and self.cls_head is not None

    def init_weights(self):
        """Initialize the model network weights."""
        self.backbone.init_weights()
        if self.with_cls_head:
            self.cls_head.init_weights()

    def extract_feat(self, imgs):
        """Extract features through a backbone.

        Args:
            imgs (torch.Tensor): The input images.

        Returns:
            torch.tensor: The extracted features.
        """
        x = self.backbone(imgs)
        return x

    def average_clip(self, cls_score):
        """Averaging class score over multiple clips.

        Using different averaging types ('score' or 'prob' or None, which defined in test_cfg) to computed the final
        averaged class score. Only called in test mode. By default, we use 'prob' mode.

        Args:
            cls_score (torch.Tensor): Class score to be averaged.

        Returns:
            torch.Tensor: Averaged class score.
        """
        assert len(cls_score.shape) == 3
        average_clips = self.test_cfg.get('average_clips', 'prob')
        if average_clips not in ['score', 'prob', None]:
            raise ValueError(f'{average_clips} is not supported. Supported: ["score", "prob", None]')
        if average_clips is None:
            return cls_score
        if average_clips == 'prob':
            return F.softmax(cls_score, dim=2).mean(dim=1)
        elif average_clips == 'score':
            return cls_score.mean(dim=1)

    @abstractmethod
    def forward_train(self, imgs, label, **kwargs):
        """Defines the computation performed at every call when training."""

    @abstractmethod
    def forward_test(self, imgs, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""

    def _parse_losses(self, losses):
        """Parse the ra w outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')
        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()
        return loss, log_vars

    def forward(self, imgs, label=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(imgs, label, **kwargs)
        return self.forward_test(imgs, **kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data_batch, return_loss=True)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(next(iter(data_batch.values()))))
        return outputs


class Recognizer2D(BaseRecognizer):
    """2D recognizer model framework."""

    def forward_train(self, imgs, label, **kwargs):
        """Defines the computation performed at every call when training."""
        assert self.with_cls_head
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches
        losses = dict()
        x = self.extract_feat(imgs)
        x = x.reshape((batches, num_segs) + x.shape[1:])
        cls_score = self.cls_head(x)
        gt_label = label.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_label, **kwargs)
        losses.update(loss_cls)
        return losses

    def forward_test(self, imgs, **kwargs):
        """Defines the computation performed at every call when evaluation and testing."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        x = self.extract_feat(imgs)
        assert 'num_segs' in self.test_cfg
        num_segs = self.test_cfg['num_segs']
        assert x.shape[0] % (batches * num_segs) == 0
        num_crops = x.shape[0] // (batches * num_segs)
        if self.test_cfg.get('feat_ext', False):
            avg_pool = nn.AdaptiveAvgPool2d(1)
            x = avg_pool(x)
            x = x.reshape((batches, num_crops, num_segs, -1))
            x = x.mean(axis=1).mean(axis=1)
            return x.cpu().numpy()
        x = x.reshape((batches * num_crops, num_segs) + x.shape[1:])
        cls_score = self.cls_head(x)
        cls_score = cls_score.reshape(batches, num_crops, cls_score.shape[-1])
        cls_score = self.average_clip(cls_score)
        return cls_score.cpu().numpy()


class Recognizer3D(BaseRecognizer):
    """3D recognizer model framework."""

    def forward_train(self, imgs, label, **kwargs):
        """Defines the computation performed at every call when training."""
        assert self.with_cls_head
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        losses = dict()
        x = self.extract_feat(imgs)
        cls_score = self.cls_head(x)
        gt_label = label.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_label, **kwargs)
        losses.update(loss_cls)
        return losses

    def forward_test(self, imgs, **kwargs):
        """Defines the computation performed at every call when evaluation, testing."""
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        if self.max_testing_views is not None:
            total_views = imgs.shape[0]
            assert num_segs == total_views, 'max_testing_views is only compatible with batch_size == 1'
            view_ptr = 0
            feats = []
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                x = self.extract_feat(batch_imgs)
                feats.append(x)
                view_ptr += self.max_testing_views
            if isinstance(feats[0], tuple):
                len_tuple = len(feats[0])
                feat = [torch.cat([x[i] for x in feats]) for i in range(len_tuple)]
                feat = tuple(feat)
            else:
                feat = torch.cat(feats)
        else:
            feat = self.extract_feat(imgs)
        if self.test_cfg.get('feat_ext', False):
            feat_dim = len(feat[0].size()) if isinstance(feat, tuple) else len(feat.size())
            assert feat_dim in [5, 2], 'Got feature of unknown architecture, only 3D-CNN-like ([N, in_channels, T, H, W]), and transformer-like ([N, in_channels]) features are supported.'
            if feat_dim == 5:
                avg_pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(feat, tuple):
                    feat = [avg_pool(x) for x in feat]
                    feat = torch.cat(feat, axis=1)
                else:
                    feat = avg_pool(feat)
                feat = feat.reshape((batches, num_segs, -1))
                feat = feat.mean(axis=1)
            return feat.cpu().numpy()
        assert self.with_cls_head
        cls_score = self.cls_head(feat)
        cls_score = cls_score.reshape(batches, num_segs, cls_score.shape[-1])
        cls_score = self.average_clip(cls_score)
        return cls_score.cpu().numpy()


class RecognizerGCN(BaseRecognizer):
    """GCN-based recognizer for skeleton-based action recognition. """

    def forward_train(self, keypoint, label, **kwargs):
        """Defines the computation performed at every call when training."""
        assert self.with_cls_head
        assert keypoint.shape[1] == 1
        keypoint = keypoint[:, 0]
        losses = dict()
        x = self.extract_feat(keypoint)
        cls_score = self.cls_head(x)
        gt_label = label.squeeze(-1)
        loss = self.cls_head.loss(cls_score, gt_label)
        losses.update(loss)
        return losses

    def forward_test(self, keypoint, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        assert self.with_cls_head or self.feat_ext
        bs, nc = keypoint.shape[:2]
        keypoint = keypoint.reshape((bs * nc,) + keypoint.shape[2:])
        x = self.extract_feat(keypoint)
        feat_ext = self.test_cfg.get('feat_ext', False)
        pool_opt = self.test_cfg.get('pool_opt', 'all')
        score_ext = self.test_cfg.get('score_ext', False)
        if feat_ext or score_ext:
            assert bs == 1
            assert isinstance(pool_opt, str)
            dim_idx = dict(n=0, m=1, t=3, v=4)
            if pool_opt == 'all':
                pool_opt == 'nmtv'
            if pool_opt != 'none':
                for digit in pool_opt:
                    assert digit in dim_idx
            if isinstance(x, tuple) or isinstance(x, list):
                x = torch.cat(x, dim=2)
            assert len(x.shape) == 5, 'The shape is N, M, C, T, V'
            if pool_opt != 'none':
                for d in pool_opt:
                    x = x.mean(dim_idx[d], keepdim=True)
            if score_ext:
                w = self.cls_head.fc_cls.weight
                b = self.cls_head.fc_cls.bias
                x = torch.einsum('nmctv,oc->nmotv', x, w)
                if b is not None:
                    x = x + b[..., None, None]
                x = x[None]
            return x.data.cpu().numpy().astype(np.float16)
        cls_score = self.cls_head(x)
        cls_score = cls_score.reshape(bs, nc, cls_score.shape[-1])
        if 'average_clips' not in self.test_cfg:
            self.test_cfg['average_clips'] = 'prob'
        cls_score = self.average_clip(cls_score)
        if isinstance(cls_score, tuple) or isinstance(cls_score, list):
            cls_score = [x.data.cpu().numpy() for x in cls_score]
            return [[x[i] for x in cls_score] for i in range(bs)]
        return cls_score.data.cpu().numpy()

    def forward(self, keypoint, label=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(keypoint, label, **kwargs)
        return self.forward_test(keypoint, **kwargs)

    def extract_feat(self, keypoint):
        """Extract features through a backbone.

        Args:
            keypoint (torch.Tensor): The input keypoints.

        Returns:
            torch.tensor: The extracted features.
        """
        return self.backbone(keypoint)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BCELossWithLogits,
     lambda: ([], {}),
     lambda: ([], {'cls_score': torch.rand([4, 4]), 'label': torch.rand([4, 4])})),
    (CTRGC,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([], {'cls_score': torch.rand([4, 4]), 'label': torch.rand([4, 4])})),
    (SEModule,
     lambda: ([], {'channels': 4, 'reduction': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (UnfoldTemporalWindows,
     lambda: ([], {'window_size': 4, 'window_stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (unit_aagcn,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'A': torch.rand([4, 4])}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (unit_ctrgcn,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'A': torch.rand([4, 4])}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (unit_sgn,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
]

