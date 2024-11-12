
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


import copy


import logging


import warnings


from abc import ABCMeta


from abc import abstractmethod


from torch.optim import Optimizer


import time


from collections import OrderedDict


import torchvision


from torch.utils import model_zoo


from torch.nn import functional as F


import numpy as np


from random import choice


import random


from collections import defaultdict


from itertools import chain


from torch.nn.utils import clip_grad


from copy import deepcopy


from enum import IntEnum


from enum import unique


import math


import torch.nn.functional as F


from torch.distributions import Normal


import matplotlib.pyplot as plt


from matplotlib import pyplot as plt


from torch.utils.data import Dataset


from typing import Any


from typing import Dict


from numpy import random


import torch.nn as nn


import torch.utils.checkpoint as cp


from torch.nn.modules.batchnorm import _BatchNorm


from torch import nn as nn


import torch.utils.checkpoint as checkpoint


from torch import nn


from torch.nn.parameter import Parameter


from torch.nn import Linear


from torch.nn.init import xavier_uniform_


from torch.nn.init import constant_


from torchvision.utils import save_image


from torchvision.models.resnet import resnet18


from functools import partial


from torch.nn.functional import l1_loss


from torch.nn.functional import mse_loss


from torch.nn.functional import smooth_l1_loss


from torch.autograd import Function


from typing import List


from typing import Tuple


from torch import distributed as dist


from torch.autograd.function import Function


from torch.nn import init


from torch.nn.modules.utils import _pair


import torch.distributed as dist


from math import inf


from torch.utils.data import DataLoader


from torch._utils import _flatten_dense_tensors


from torch._utils import _take_tensors


from torch._utils import _unflatten_dense_tensors


from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


from collections.abc import Sequence


from torch.utils.data import DistributedSampler as _DistributedSampler


from torch.utils.data import Sampler


from logging import warning


from math import ceil


from math import log


from inspect import signature


import functools


from torch.utils.checkpoint import checkpoint


from math import sqrt


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.nn.modules import AvgPool2d


from torch.nn.modules import GroupNorm


import re


from torch.nn import BatchNorm1d


from torch.nn import ReLU


from torch.autograd import gradcheck


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, style='pytorch', with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), dcn=None, plugins=None):
        super(BasicBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.conv1 = build_conv_layer(conv_cfg, inplanes, planes, 3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.norm2(out)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, style='pytorch', with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), dcn=None, plugins=None):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None
        if self.with_plugins:
            self.after_conv1_plugins = [plugin['cfg'] for plugin in plugins if plugin['position'] == 'after_conv1']
            self.after_conv2_plugins = [plugin['cfg'] for plugin in plugins if plugin['position'] == 'after_conv2']
            self.after_conv3_plugins = [plugin['cfg'] for plugin in plugins if plugin['position'] == 'after_conv3']
        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(norm_cfg, planes * self.expansion, postfix=3)
        self.conv1 = build_conv_layer(conv_cfg, inplanes, planes, kernel_size=1, stride=self.conv1_stride, bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(conv_cfg, planes, planes, kernel_size=3, stride=self.conv2_stride, padding=dilation, dilation=dilation, bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(dcn, planes, planes, kernel_size=3, stride=self.conv2_stride, padding=dilation, dilation=dilation, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(conv_cfg, planes, planes * self.expansion, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(plugin, in_channels=in_channels, postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)
            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)
            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)
            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)
            out = self.conv3(out)
            out = self.norm3(out)
            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)
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


BN_MOMENTUM = 0.1


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out


class Root(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)
        return x


class Tree(nn.Module):

    def __init__(self, levels, block, in_channels, out_channels, stride=1, level_root=False, root_dim=0, root_kernel_size=1, dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels, stride, root_dim=0, root_kernel_size=root_kernel_size, dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels, root_dim=root_dim + out_channels, root_kernel_size=root_kernel_size, dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM))

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


class DLA(nn.Module):

    def __init__(self, levels, channels, num_classes=1000, block=BasicBlock, residual_root=False, linear_root=False, opt=None):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False), nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2, level_root=False, root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2, level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2, level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2, level_root=True, root_residual=residual_root)
        if opt.pre_img:
            self.pre_img_layer = nn.Sequential(nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False), nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
        if opt.pre_hm:
            self.pre_hm_layer = nn.Sequential(nn.Conv2d(1, channels[0], kernel_size=7, stride=1, padding=3, bias=False), nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM), nn.ReLU(inplace=True))

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(nn.MaxPool2d(stride, stride=stride), nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride if i == 0 else 1, padding=dilation, bias=False, dilation=dilation), nn.BatchNorm2d(planes, momentum=BN_MOMENTUM), nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x, pre_img=None, pre_hm=None):
        y = []
        x = self.base_layer(x)
        if pre_img is not None:
            x = x + self.pre_img_layer(pre_img)
        if pre_hm is not None:
            x = x + self.pre_hm_layer(pre_hm)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(self.channels[-1], num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights, strict=False)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Conv(nn.Module):

    def __init__(self, chi, cho):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(chi, cho, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(cho, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class GlobalConv(nn.Module):

    def __init__(self, chi, cho, k=7, d=1):
        super(GlobalConv, self).__init__()
        gcl = nn.Sequential(nn.Conv2d(chi, cho, kernel_size=(k, 1), stride=1, bias=False, dilation=d, padding=(d * (k // 2), 0)), nn.Conv2d(cho, cho, kernel_size=(1, k), stride=1, bias=False, dilation=d, padding=(0, d * (k // 2))))
        gcr = nn.Sequential(nn.Conv2d(chi, cho, kernel_size=(1, k), stride=1, bias=False, dilation=d, padding=(0, d * (k // 2))), nn.Conv2d(cho, cho, kernel_size=(k, 1), stride=1, bias=False, dilation=d, padding=(d * (k // 2), 0)))
        fill_fc_weights(gcl)
        fill_fc_weights(gcr)
        self.gcl = gcl
        self.gcr = gcr
        self.act = nn.Sequential(nn.BatchNorm2d(cho, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.gcl(x) + self.gcr(x)
        x = self.act(x)
        return x


class DeformConv(nn.Module):

    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(nn.BatchNorm2d(cho, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
        self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f, node_type=(DeformConv, DeformConv)):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = node_type[0](c, o)
            node = node_type[1](o, o)
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, padding=f // 2, output_padding=0, groups=o, bias=False)
            fill_up_weights(up)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):

    def __init__(self, startp, channels, scales, in_channels=None, node_type=DeformConv):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i), IDAUp(channels[j], in_channels[j:], scales[j:] // scales[j], node_type=node_type))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class Interpolate(nn.Module):

    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x


class BaseModel(nn.Module):

    def __init__(self, heads, head_convs, num_stacks, last_channel, opt=None):
        super(BaseModel, self).__init__()
        if opt is not None and opt.head_kernel != 3:
            None
            head_kernel = opt.head_kernel
        else:
            head_kernel = 3
        self.num_stacks = num_stacks
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            head_conv = head_convs[head]
            if len(head_conv) > 0:
                out = nn.Conv2d(head_conv[-1], classes, kernel_size=1, stride=1, padding=0, bias=True)
                conv = nn.Conv2d(last_channel, head_conv[0], kernel_size=head_kernel, padding=head_kernel // 2, bias=True)
                convs = [conv]
                for k in range(1, len(head_conv)):
                    convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k], kernel_size=1, bias=True))
                if len(convs) == 1:
                    fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
                elif len(convs) == 2:
                    fc = nn.Sequential(convs[0], nn.ReLU(inplace=True), convs[1], nn.ReLU(inplace=True), out)
                elif len(convs) == 3:
                    fc = nn.Sequential(convs[0], nn.ReLU(inplace=True), convs[1], nn.ReLU(inplace=True), convs[2], nn.ReLU(inplace=True), out)
                elif len(convs) == 4:
                    fc = nn.Sequential(convs[0], nn.ReLU(inplace=True), convs[1], nn.ReLU(inplace=True), convs[2], nn.ReLU(inplace=True), convs[3], nn.ReLU(inplace=True), out)
                if 'hm' in head:
                    fc[-1].bias.data.fill_(opt.prior_bias)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(last_channel, classes, kernel_size=1, stride=1, padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(opt.prior_bias)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def img2feats(self, x):
        raise NotImplementedError

    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
        raise NotImplementedError

    def forward(self, x, pre_img=None, pre_hm=None):
        if pre_hm is not None or pre_img is not None:
            feats = self.imgpre2feats(x, pre_img, pre_hm)
        else:
            feats = self.img2feats(x)
        return feats


DLA_NODE = {'dcn': (DeformConv, DeformConv), 'gcn': (Conv, GlobalConv), 'conv': (Conv, Conv)}


class Opt:
    head_kernel = 3
    levels = [1, 1, 1, 2, 2, 1]
    channels = [16, 32, 64, 128, 256, 512]
    pre_img = False
    pre_hm = False
    dla_node = 'dcn'
    model_output_list = False


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name='mmdet', log_file=log_file, log_level=log_level)
    return logger


DEFAULT_CACHE_DIR = '~/.cache'


ENV_MMCV_HOME = 'MMCV_HOME'


ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'


def _get_mmcv_home():
    mmcv_home = os.path.expanduser(os.getenv(ENV_MMCV_HOME, os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'mmcv')))
    mkdir_or_exist(mmcv_home)
    return mmcv_home


def _process_mmcls_checkpoint(checkpoint):
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            new_state_dict[k[9:]] = v
    new_checkpoint = dict(state_dict=new_state_dict)
    return new_checkpoint


def get_deprecated_model_names():
    deprecate_json_path = osp.join(mmcv.__path__[0], 'model_zoo/deprecated.json')
    deprecate_urls = load_file(deprecate_json_path)
    assert isinstance(deprecate_urls, dict)
    return deprecate_urls


def get_external_models():
    mmcv_home = _get_mmcv_home()
    default_json_path = osp.join(mmcv.__path__[0], 'model_zoo/open_mmlab.json')
    default_urls = load_file(default_json_path)
    assert isinstance(default_urls, dict)
    external_json_path = osp.join(mmcv_home, 'open_mmlab.json')
    if osp.exists(external_json_path):
        external_urls = load_file(external_json_path)
        assert isinstance(external_urls, dict)
        default_urls.update(external_urls)
    return default_urls


def get_mmcls_models():
    mmcls_json_path = osp.join(mmcv.__path__[0], 'model_zoo/mmcls.json')
    mmcls_urls = load_file(mmcls_json_path)
    return mmcls_urls


def get_torchvision_models():
    model_urls = dict()
    for _, name, ispkg in pkgutil.walk_packages(torchvision.models.__path__):
        if ispkg:
            continue
        _zoo = import_module(f'torchvision.models.{name}')
        if hasattr(_zoo, 'model_urls'):
            _urls = getattr(_zoo, 'model_urls')
            model_urls.update(_urls)
    return model_urls


def load_fileclient_dist(filename, backend, map_location):
    """In distributed setting, this function only download checkpoint at local
    rank 0."""
    rank, world_size = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    allowed_backends = ['ceph']
    if backend not in allowed_backends:
        raise ValueError(f'Load from Backend {backend} is not supported.')
    if rank == 0:
        fileclient = FileClient(backend=backend)
        buffer = io.BytesIO(fileclient.get(filename))
        checkpoint = torch.load(buffer, map_location=map_location)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            fileclient = FileClient(backend=backend)
            buffer = io.BytesIO(fileclient.get(filename))
            checkpoint = torch.load(buffer, map_location=map_location)
    return checkpoint


def load_url_dist(url, model_dir=None):
    None
    """In distributed setting, this function only download checkpoint at local
    rank 0."""
    rank, world_size = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    if rank == 0:
        checkpoint = model_zoo.load_url(url, model_dir=model_dir)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = model_zoo.load_url(url, model_dir=model_dir)
    return checkpoint


def _load_checkpoint(filename, map_location=None):
    """Load checkpoint from somewhere (modelzoo, file, url).

    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str | None): Same as :func:`torch.load`. Default: None.

    Returns:
        dict | OrderedDict: The loaded checkpoint. It can be either an
            OrderedDict storing model weights or a dict containing other
            information, which depends on the checkpoint.
    """
    if filename.startswith('modelzoo://'):
        warnings.warn('The URL scheme of "modelzoo://" is deprecated, please use "torchvision://" instead')
        model_urls = get_torchvision_models()
        model_name = filename[11:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith('torchvision://'):
        model_urls = get_torchvision_models()
        model_name = filename[14:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith('open-mmlab://'):
        model_urls = get_external_models()
        model_name = filename[13:]
        deprecated_urls = get_deprecated_model_names()
        if model_name in deprecated_urls:
            warnings.warn(f'open-mmlab://{model_name} is deprecated in favor of open-mmlab://{deprecated_urls[model_name]}')
            model_name = deprecated_urls[model_name]
        model_url = model_urls[model_name]
        if model_url.startswith(('http://', 'https://')):
            checkpoint = load_url_dist(model_url)
        else:
            filename = osp.join(_get_mmcv_home(), model_url)
            if not osp.isfile(filename):
                raise IOError(f'{filename} is not a checkpoint file')
            checkpoint = torch.load(filename, map_location=map_location)
    elif filename.startswith('mmcls://'):
        model_urls = get_mmcls_models()
        model_name = filename[8:]
        checkpoint = load_url_dist(model_urls[model_name])
        checkpoint = _process_mmcls_checkpoint(checkpoint)
    elif filename.startswith(('http://', 'https://')):
        checkpoint = load_url_dist(filename)
    elif filename.startswith('pavi://'):
        model_path = filename[7:]
        checkpoint = load_pavimodel_dist(model_path, map_location=map_location)
    elif filename.startswith('s3://'):
        checkpoint = load_fileclient_dist(filename, backend='ceph', map_location=map_location)
    else:
        if not osp.isfile(filename):
            raise IOError(f'{filename} is not a checkpoint file')
        checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        if is_module_wrapper(module):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True, all_missing_keys, unexpected_keys, err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')
    load(module)
    load = None
    missing_keys = [key for key in all_missing_keys if 'num_batches_tracked' not in key]
    if unexpected_keys:
        err_msg.append(f"unexpected key in source state_dict: {', '.join(unexpected_keys)}\n")
    if missing_keys:
        err_msg.append(f"missing keys in source state_dict: {', '.join(missing_keys)}\n")
    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            None


def load_checkpoint(model, filename, map_location='cpu', strict=False, logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    if state_dict.get('absolute_pos_embed') is not None:
        absolute_pos_embed = state_dict['absolute_pos_embed']
        N1, L, C1 = absolute_pos_embed.size()
        N2, C2, H, W = model.absolute_pos_embed.size()
        if N1 != N2 or C1 != C2 or L != H * W:
            logger.warning('Error in loading absolute_pos_embed, pass')
        else:
            state_dict['absolute_pos_embed'] = absolute_pos_embed.view(N2, H, W, C2).permute(0, 3, 1, 2)
    relative_position_bias_table_keys = [k for k in model.state_dict().keys() if 'relative_position_bias_table' in k]
    for table_key in relative_position_bias_table_keys:
        table_pretrained = state_dict[table_key]
        table_current = model.state_dict()[table_key]
        L1, nH1 = table_pretrained.size()
        L2, nH2 = table_current.size()
        if nH1 != nH2:
            logger.warning(f'Error in loading {table_key}, pass')
        elif L1 != L2:
            S1 = int(L1 ** 0.5)
            S2 = int(L2 ** 0.5)
            table_pretrained_resized = F.interpolate(table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2), mode='bicubic')
            state_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(1, 0)
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


class DLASeg(BaseModel):

    def __init__(self, num_layers, heads, head_convs):
        opt = Opt()
        super(DLASeg, self).__init__(heads, head_convs, 1, 64 if num_layers == 34 else 128, opt=opt)
        down_ratio = 4
        self.opt = opt
        self.node_type = DLA_NODE[opt.dla_node]
        None
        self.first_level = int(np.log2(down_ratio))
        self.last_level = 5
        self.base = globals()['dla{}'.format(num_layers)](pretrained=False, opt=opt)
        channels = self.base.channels
        scales = [(2 ** i) for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales, node_type=self.node_type)
        out_channel = channels[self.first_level]
        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], [(2 ** i) for i in range(self.last_level - self.first_level)], node_type=self.node_type)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        else:
            pass

    def img2feats(self, x):
        x = self.base(x)
        x = self.dla_up(x)
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))
        return [y[-1]]

    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
        x = self.base(x, pre_img, pre_hm)
        x = self.dla_up(x)
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))
        return [y[-1]]


class BasePointNet(nn.Module, metaclass=ABCMeta):
    """Base class for PointNet."""

    def __init__(self):
        super(BasePointNet, self).__init__()
        self.fp16_enabled = False

    def init_weights(self, pretrained=None):
        """Initialize the weights of PointNet backbone."""
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    @staticmethod
    def _split_point_feats(points):
        """Split coordinates and features of input points.

        Args:
            points (torch.Tensor): Point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            torch.Tensor: Coordinates of input points.
            torch.Tensor: Features of input points.
        """
        xyz = points[..., 0:3].contiguous()
        if points.size(-1) > 3:
            features = points[..., 3:].transpose(1, 2).contiguous()
        else:
            features = None
        return xyz, features


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    """Build backbone."""
    return build(cfg, BACKBONES)


def build_sa_module(cfg, *args, **kwargs):
    """Build PointNet2 set abstraction (SA) module.

    Args:
        cfg (None or dict): The SA module config, which should contain:
            - type (str): Module type.
            - module args: Args needed to instantiate an SA module.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding module.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding SA module .

    Returns:
        nn.Module: Created SA module.
    """
    if cfg is None:
        cfg_ = dict(type='PointSAModule')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()
    module_type = cfg_.pop('type')
    if module_type not in SA_MODULES:
        raise KeyError(f'Unrecognized module type {module_type}')
    else:
        sa_module = SA_MODULES.get(module_type)
    module = sa_module(*args, **kwargs, **cfg_)
    return module


class ThreeInterpolate(Function):

    @staticmethod
    def forward(ctx, features: 'torch.Tensor', indices: 'torch.Tensor', weight: 'torch.Tensor') ->torch.Tensor:
        """Performs weighted linear interpolation on 3 features.

        Args:
            features (Tensor): (B, C, M) Features descriptors to be
                interpolated from
            indices (Tensor): (B, n, 3) index three nearest neighbors
                of the target features in features
            weight (Tensor): (B, n, 3) weights of interpolation

        Returns:
            Tensor: (B, C, N) tensor of the interpolated features
        """
        assert features.is_contiguous()
        assert indices.is_contiguous()
        assert weight.is_contiguous()
        B, c, m = features.size()
        n = indices.size(1)
        ctx.three_interpolate_for_backward = indices, weight, m
        output = torch.FloatTensor(B, c, n)
        interpolate_ext.three_interpolate_wrapper(B, c, m, n, features, indices, weight, output)
        return output

    @staticmethod
    def backward(ctx, grad_out: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward of three interpolate.

        Args:
            grad_out (Tensor): (B, C, N) tensor with gradients of outputs

        Returns:
            Tensor: (B, C, M) tensor with gradients of features
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        B, c, n = grad_out.size()
        grad_features = torch.FloatTensor(B, c, m).zero_()
        grad_out_data = grad_out.data.contiguous()
        interpolate_ext.three_interpolate_grad_wrapper(B, c, n, m, grad_out_data, idx, weight, grad_features.data)
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


class ThreeNN(Function):

    @staticmethod
    def forward(ctx, target: 'torch.Tensor', source: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        """Find the top-3 nearest neighbors of the target set from the source
        set.

        Args:
            target (Tensor): shape (B, N, 3), points set that needs to
                find the nearest neighbors.
            source (Tensor): shape (B, M, 3), points set that is used
                to find the nearest neighbors of points in target set.

        Returns:
            Tensor: shape (B, N, 3), L2 distance of each point in target
                set to their corresponding nearest neighbors.
        """
        assert target.is_contiguous()
        assert source.is_contiguous()
        B, N, _ = target.size()
        m = source.size(1)
        dist2 = torch.FloatTensor(B, N, 3)
        idx = torch.IntTensor(B, N, 3)
        interpolate_ext.three_nn_wrapper(B, N, m, target, source, dist2, idx)
        ctx.mark_non_differentiable(idx)
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class SECOND(nn.Module):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(self, in_channels=128, out_channels=[128, 128, 256], layer_nums=[3, 5, 5], layer_strides=[2, 2, 2], norm_cfg=dict(type='BN', eps=0.001, momentum=0.01), conv_cfg=dict(type='Conv2d', bias=False)):
        super(SECOND, self).__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)
        in_filters = [in_channels, *out_channels[:-1]]
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [build_conv_layer(conv_cfg, in_filters[i], out_channels[i], 3, stride=layer_strides[i], padding=1), build_norm_layer(norm_cfg, out_channels[i])[1], nn.ReLU(inplace=True)]
            for j in range(layer_num):
                block.append(build_conv_layer(conv_cfg, out_channels[i], out_channels[i], 3, padding=1))
                block.append(build_norm_layer(norm_cfg, out_channels[i])[1])
                block.append(nn.ReLU(inplace=True))
            block = nn.Sequential(*block)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

    def init_weights(self, pretrained=None):
        """Initialize weights of the 2D backbone."""
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        return tuple(outs)


class Mlp(nn.Module):
    """ Multilayer perceptron."""

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
        """ Forward function.

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

    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, 'input feature has wrong size'
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        x = x.view(B, H, W, C)
        pad_input = H % 2 == 1 or W % 2 == 1
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
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
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.

    Args:
        val (torch.Tensor): The value to be converted.
        offset (float, optional): Offset to set the value range.             Defaults to 0.5.
        period ([type], optional): Period of the value. Defaults to np.pi.

    Returns:
        torch.Tensor: Value in the range of             [-offset * period, (1-offset) * period]
    """
    return val - torch.floor(val / period + offset) * period


def get_direction_target(anchors, reg_targets, dir_offset=0, num_bins=2, one_hot=True):
    """Encode direction to 0 ~ num_bins-1.

    Args:
        anchors (torch.Tensor): Concatenated multi-level anchor.
        reg_targets (torch.Tensor): Bbox regression targets.
        dir_offset (int): Direction offset.
        num_bins (int): Number of bins to divide 2*PI.
        one_hot (bool): Whether to encode as one hot.

    Returns:
        torch.Tensor: Encoded direction targets.
    """
    rot_gt = reg_targets[..., 6] + anchors[..., 6]
    offset_rot = limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
    dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
    dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
    if one_hot:
        dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype, device=dir_cls_targets.device)
        dir_targets.scatter_(dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
        dir_cls_targets = dir_targets
    return dir_cls_targets


def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        level_targets.append(target[:, start:end])
        start = end
    return level_targets


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains             a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


class AnchorTrainMixin(object):
    """Mixin class for target assigning of dense heads."""

    def anchor_target_3d(self, anchor_list, gt_bboxes_list, input_metas, gt_bboxes_ignore_list=None, gt_labels_list=None, label_channels=1, num_classes=1, sampling=True):
        """Compute regression and classification targets for anchors.

        Args:
            anchor_list (list[list]): Multi level anchors of each image.
            gt_bboxes_list (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                bboxes of each image.
            input_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (None | list): Ignore list of gt bboxes.
            gt_labels_list (list[torch.Tensor]): Gt labels of batches.
            label_channels (int): The channel of labels.
            num_classes (int): The number of classes.
            sampling (bool): Whether to sample anchors.

        Returns:
            tuple (list, list, list, list, list, list, int, int):
                Anchor targets, including labels, label weights,
                bbox targets, bbox weights, direction targets,
                direction weights, number of postive anchors and
                number of negative anchors.
        """
        num_imgs = len(input_metas)
        assert len(anchor_list) == num_imgs
        if isinstance(anchor_list[0][0], list):
            num_level_anchors = [sum([anchor.size(0) for anchor in anchors]) for anchors in anchor_list[0]]
            for i in range(num_imgs):
                anchor_list[i] = anchor_list[i][0]
        else:
            num_level_anchors = [anchors.view(-1, self.box_code_size).size(0) for anchors in anchor_list[0]]
            for i in range(num_imgs):
                anchor_list[i] = torch.cat(anchor_list[i])
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, all_dir_targets, all_dir_weights, pos_inds_list, neg_inds_list = multi_apply(self.anchor_target_3d_single, anchor_list, gt_bboxes_list, gt_bboxes_ignore_list, gt_labels_list, input_metas, label_channels=label_channels, num_classes=num_classes, sampling=sampling)
        if any([(labels is None) for labels in all_labels]):
            return None
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
        dir_targets_list = images_to_levels(all_dir_targets, num_level_anchors)
        dir_weights_list = images_to_levels(all_dir_weights, num_level_anchors)
        return labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, dir_targets_list, dir_weights_list, num_total_pos, num_total_neg

    def anchor_target_3d_single(self, anchors, gt_bboxes, gt_bboxes_ignore, gt_labels, input_meta, label_channels=1, num_classes=1, sampling=True):
        """Compute targets of anchors in single batch.

        Args:
            anchors (torch.Tensor): Concatenated multi-level anchor.
            gt_bboxes (:obj:`BaseInstance3DBoxes`): Gt bboxes.
            gt_bboxes_ignore (torch.Tensor): Ignored gt bboxes.
            gt_labels (torch.Tensor): Gt class labels.
            input_meta (dict): Meta info of each image.
            label_channels (int): The channel of labels.
            num_classes (int): The number of classes.
            sampling (bool): Whether to sample anchors.

        Returns:
            tuple[torch.Tensor]: Anchor targets.
        """
        if isinstance(self.bbox_assigner, list) and not isinstance(anchors, list):
            feat_size = anchors.size(0) * anchors.size(1) * anchors.size(2)
            rot_angles = anchors.size(-2)
            assert len(self.bbox_assigner) == anchors.size(-3)
            total_labels, total_label_weights, total_bbox_targets, total_bbox_weights, total_dir_targets, total_dir_weights, total_pos_inds, total_neg_inds = [], [], [], [], [], [], [], []
            current_anchor_num = 0
            for i, assigner in enumerate(self.bbox_assigner):
                current_anchors = anchors[..., i, :, :].reshape(-1, self.box_code_size)
                current_anchor_num += current_anchors.size(0)
                if self.assign_per_class:
                    gt_per_cls = gt_labels == i
                    anchor_targets = self.anchor_target_single_assigner(assigner, current_anchors, gt_bboxes[gt_per_cls, :], gt_bboxes_ignore, gt_labels[gt_per_cls], input_meta, num_classes, sampling)
                else:
                    anchor_targets = self.anchor_target_single_assigner(assigner, current_anchors, gt_bboxes, gt_bboxes_ignore, gt_labels, input_meta, num_classes, sampling)
                labels, label_weights, bbox_targets, bbox_weights, dir_targets, dir_weights, pos_inds, neg_inds = anchor_targets
                total_labels.append(labels.reshape(feat_size, 1, rot_angles))
                total_label_weights.append(label_weights.reshape(feat_size, 1, rot_angles))
                total_bbox_targets.append(bbox_targets.reshape(feat_size, 1, rot_angles, anchors.size(-1)))
                total_bbox_weights.append(bbox_weights.reshape(feat_size, 1, rot_angles, anchors.size(-1)))
                total_dir_targets.append(dir_targets.reshape(feat_size, 1, rot_angles))
                total_dir_weights.append(dir_weights.reshape(feat_size, 1, rot_angles))
                total_pos_inds.append(pos_inds)
                total_neg_inds.append(neg_inds)
            total_labels = torch.cat(total_labels, dim=-2).reshape(-1)
            total_label_weights = torch.cat(total_label_weights, dim=-2).reshape(-1)
            total_bbox_targets = torch.cat(total_bbox_targets, dim=-3).reshape(-1, anchors.size(-1))
            total_bbox_weights = torch.cat(total_bbox_weights, dim=-3).reshape(-1, anchors.size(-1))
            total_dir_targets = torch.cat(total_dir_targets, dim=-2).reshape(-1)
            total_dir_weights = torch.cat(total_dir_weights, dim=-2).reshape(-1)
            total_pos_inds = torch.cat(total_pos_inds, dim=0).reshape(-1)
            total_neg_inds = torch.cat(total_neg_inds, dim=0).reshape(-1)
            return total_labels, total_label_weights, total_bbox_targets, total_bbox_weights, total_dir_targets, total_dir_weights, total_pos_inds, total_neg_inds
        elif isinstance(self.bbox_assigner, list) and isinstance(anchors, list):
            assert len(self.bbox_assigner) == len(anchors), 'The number of bbox assigners and anchors should be the same.'
            total_labels, total_label_weights, total_bbox_targets, total_bbox_weights, total_dir_targets, total_dir_weights, total_pos_inds, total_neg_inds = [], [], [], [], [], [], [], []
            current_anchor_num = 0
            for i, assigner in enumerate(self.bbox_assigner):
                current_anchors = anchors[i]
                current_anchor_num += current_anchors.size(0)
                if self.assign_per_class:
                    gt_per_cls = gt_labels == i
                    anchor_targets = self.anchor_target_single_assigner(assigner, current_anchors, gt_bboxes[gt_per_cls, :], gt_bboxes_ignore, gt_labels[gt_per_cls], input_meta, num_classes, sampling)
                else:
                    anchor_targets = self.anchor_target_single_assigner(assigner, current_anchors, gt_bboxes, gt_bboxes_ignore, gt_labels, input_meta, num_classes, sampling)
                labels, label_weights, bbox_targets, bbox_weights, dir_targets, dir_weights, pos_inds, neg_inds = anchor_targets
                total_labels.append(labels)
                total_label_weights.append(label_weights)
                total_bbox_targets.append(bbox_targets.reshape(-1, anchors[i].size(-1)))
                total_bbox_weights.append(bbox_weights.reshape(-1, anchors[i].size(-1)))
                total_dir_targets.append(dir_targets)
                total_dir_weights.append(dir_weights)
                total_pos_inds.append(pos_inds)
                total_neg_inds.append(neg_inds)
            total_labels = torch.cat(total_labels, dim=0)
            total_label_weights = torch.cat(total_label_weights, dim=0)
            total_bbox_targets = torch.cat(total_bbox_targets, dim=0)
            total_bbox_weights = torch.cat(total_bbox_weights, dim=0)
            total_dir_targets = torch.cat(total_dir_targets, dim=0)
            total_dir_weights = torch.cat(total_dir_weights, dim=0)
            total_pos_inds = torch.cat(total_pos_inds, dim=0)
            total_neg_inds = torch.cat(total_neg_inds, dim=0)
            return total_labels, total_label_weights, total_bbox_targets, total_bbox_weights, total_dir_targets, total_dir_weights, total_pos_inds, total_neg_inds
        else:
            return self.anchor_target_single_assigner(self.bbox_assigner, anchors, gt_bboxes, gt_bboxes_ignore, gt_labels, input_meta, num_classes, sampling)

    def anchor_target_single_assigner(self, bbox_assigner, anchors, gt_bboxes, gt_bboxes_ignore, gt_labels, input_meta, num_classes=1, sampling=True):
        """Assign anchors and encode positive anchors.

        Args:
            bbox_assigner (BaseAssigner): assign positive and negative boxes.
            anchors (torch.Tensor): Concatenated multi-level anchor.
            gt_bboxes (:obj:`BaseInstance3DBoxes`): Gt bboxes.
            gt_bboxes_ignore (torch.Tensor): Ignored gt bboxes.
            gt_labels (torch.Tensor): Gt class labels.
            input_meta (dict): Meta info of each image.
            num_classes (int): The number of classes.
            sampling (bool): Whether to sample anchors.

        Returns:
            tuple[torch.Tensor]: Anchor targets.
        """
        anchors = anchors.reshape(-1, anchors.size(-1))
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        dir_targets = anchors.new_zeros(anchors.shape[0], dtype=torch.long)
        dir_weights = anchors.new_zeros(anchors.shape[0], dtype=torch.float)
        labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        if len(gt_bboxes) > 0:
            if not isinstance(gt_bboxes, torch.Tensor):
                gt_bboxes = gt_bboxes.tensor
            assign_result = bbox_assigner.assign(anchors, gt_bboxes, gt_bboxes_ignore, gt_labels)
            sampling_result = self.bbox_sampler.sample(assign_result, anchors, gt_bboxes)
            pos_inds = sampling_result.pos_inds
            neg_inds = sampling_result.neg_inds
        else:
            pos_inds = torch.nonzero(anchors.new_zeros((anchors.shape[0],), dtype=torch.bool) > 0, as_tuple=False).squeeze(-1).unique()
            neg_inds = torch.nonzero(anchors.new_zeros((anchors.shape[0],), dtype=torch.bool) == 0, as_tuple=False).squeeze(-1).unique()
        if gt_labels is not None:
            labels += num_classes
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            pos_dir_targets = get_direction_target(sampling_result.pos_bboxes, pos_bbox_targets, self.dir_offset, one_hot=False)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            dir_targets[pos_inds] = pos_dir_targets
            dir_weights[pos_inds] = 1.0
            if gt_labels is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        return labels, label_weights, bbox_targets, bbox_weights, dir_targets, dir_weights, pos_inds, neg_inds

