
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


import torchvision.transforms as transforms


import numpy as np


import torch.nn as nn


import warnings


import random


import torchvision


import time


import copy


from sklearn.utils.class_weight import compute_class_weight


from copy import deepcopy


import torch.nn.functional as F


from functools import partial


from typing import Any


from typing import Dict


from typing import Optional


from typing import Tuple


from typing import Union


import re


import torch.utils.checkpoint as cp


from collections import OrderedDict


from torchvision._internally_replaced_utils import load_state_dict_from_url


from torch import Tensor


from typing import List


import math


from typing import Callable


from typing import Sequence


from torch import nn


from torchvision.ops import StochasticDepth


from torchvision.ops.misc import Conv2dNormActivation


from torchvision.ops.misc import SqueezeExcitation


from torchvision.transforms._presets import ImageClassification


from torchvision.transforms._presets import InterpolationMode


from torchvision.utils import _log_api_usage_once


from torchvision.models._api import WeightsEnum


from torchvision.models._api import Weights


from torchvision.models._meta import _IMAGENET_CATEGORIES


from torchvision.models._utils import handle_legacy_interface


from torchvision.models._utils import _ovewrite_named_param


from torchvision.models._utils import _make_divisible


from torch.hub import load_state_dict_from_url


from torchvision.ops.misc import ConvNormActivation


from torchvision.ops.misc import SqueezeExcitation as SElayer


import torch.utils.checkpoint as checkpoint


from torch.nn import Conv2d


from torch.nn import Module


from torch.nn import ReLU


from torch.nn.modules.utils import _pair


from typing import Type


from typing import cast


import matplotlib.pyplot as plt


from sklearn import utils


import itertools


import pandas as pd


from math import cos


from math import pi


from sklearn.metrics import classification_report


from sklearn.metrics import confusion_matrix


from sklearn.metrics import cohen_kappa_score


from sklearn.metrics import precision_score


from sklearn.metrics import recall_score


from sklearn.metrics import f1_score


from sklearn.metrics import accuracy_score


from sklearn.manifold import TSNE


from collections import namedtuple


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-06, data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError
        self.normalized_shape = normalized_shape,

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == 'channels_first':
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    """ ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-06):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-06)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    """ ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.0, layer_scale_init_value=1e-06, head_init_scale=1.0):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4), LayerNorm(dims[0], eps=1e-06, data_format='channels_first'))
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-06, data_format='channels_first'), nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]
        self.norm = nn.LayerNorm(dims[-1], eps=1e-06)
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x, need_fea=False):
        if need_fea:
            features = []
            for i in range(4):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
                features.append(x)
            return features, self.norm(features[-1].mean([-2, -1]))
        else:
            for i in range(4):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
            return self.norm(x.mean([-2, -1]))

    def forward(self, x, need_fea=False):
        if need_fea:
            features, features_fc = self.forward_features(x, need_fea=need_fea)
            x = self.head(features_fc)
            return features, features_fc, x
        else:
            x = self.forward_features(x)
            x = self.head(x)
            return x

    def cam_layer(self):
        return self.stages[-1]


def fuse_conv_bn(conv, bn):
    fusedconv = nn.Conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, groups=conv.groups, bias=True).requires_grad_(False)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv


class BottleneckBlock(nn.Module):
    """ ResNe(X)t Bottleneck Block
    """

    def __init__(self, in_chs, out_chs, dilation=1, bottle_ratio=0.25, groups=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_last=False, attn_layer=None, drop_block=None, drop_path=0.0):
        super(BottleneckBlock, self).__init__()
        mid_chs = int(round(out_chs * bottle_ratio))
        ckwargs = dict(act_layer=act_layer, norm_layer=norm_layer)
        attn_last = attn_layer is not None and attn_last
        attn_first = attn_layer is not None and not attn_last
        self.conv1 = ConvNormAct(in_chs, mid_chs, kernel_size=1, **ckwargs)
        self.conv2 = ConvNormAct(mid_chs, mid_chs, kernel_size=3, dilation=dilation, groups=groups, drop_layer=drop_block, **ckwargs)
        self.attn2 = attn_layer(mid_chs, act_layer=act_layer) if attn_first else nn.Identity()
        self.conv3 = ConvNormAct(mid_chs, out_chs, kernel_size=1, apply_act=False, **ckwargs)
        self.attn3 = attn_layer(out_chs, act_layer=act_layer) if attn_last else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        self.act3 = create_act_layer(act_layer)

    def zero_init_last(self):
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attn2(x)
        x = self.conv3(x)
        x = self.attn3(x)
        x = self.drop_path(x) + shortcut
        x = self.act3(x)
        return x

    def switch_to_deploy(self):
        self.conv1 = nn.Sequential(fuse_conv_bn(self.conv1.conv, self.conv1.bn), self.conv1.bn.act)
        self.conv2 = nn.Sequential(fuse_conv_bn(self.conv2.conv, self.conv2.bn), self.conv2.bn.act)
        self.conv3 = nn.Sequential(fuse_conv_bn(self.conv3.conv, self.conv3.bn))


class DarkBlock(nn.Module):
    """ DarkNet Block
    """

    def __init__(self, in_chs, out_chs, dilation=1, bottle_ratio=0.5, groups=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_layer=None, drop_block=None, drop_path=0.0):
        super(DarkBlock, self).__init__()
        mid_chs = int(round(out_chs * bottle_ratio))
        ckwargs = dict(act_layer=act_layer, norm_layer=norm_layer)
        self.conv1 = ConvNormAct(in_chs, mid_chs, kernel_size=1, **ckwargs)
        self.attn = attn_layer(mid_chs, act_layer=act_layer) if attn_layer is not None else nn.Identity()
        self.conv2 = ConvNormAct(mid_chs, out_chs, kernel_size=3, dilation=dilation, groups=groups, drop_layer=drop_block, **ckwargs)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def zero_init_last(self):
        nn.init.zeros_(self.conv2.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.attn(x)
        x = self.conv2(x)
        x = self.drop_path(x) + shortcut
        return x

    def switch_to_deploy(self):
        self.conv1 = nn.Sequential(fuse_conv_bn(self.conv1.conv, self.conv1.bn), self.conv1.bn.act)
        self.conv2 = nn.Sequential(fuse_conv_bn(self.conv2.conv, self.conv2.bn), self.conv2.bn.act)


class EdgeBlock(nn.Module):
    """ EdgeResidual / Fused-MBConv / MobileNetV1-like 3x3 + 1x1 block (w/ activated output)
    """

    def __init__(self, in_chs, out_chs, dilation=1, bottle_ratio=0.5, groups=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_layer=None, drop_block=None, drop_path=0.0):
        super(EdgeBlock, self).__init__()
        mid_chs = int(round(out_chs * bottle_ratio))
        ckwargs = dict(act_layer=act_layer, norm_layer=norm_layer)
        self.conv1 = ConvNormAct(in_chs, mid_chs, kernel_size=3, dilation=dilation, groups=groups, drop_layer=drop_block, **ckwargs)
        self.attn = attn_layer(mid_chs, act_layer=act_layer) if attn_layer is not None else nn.Identity()
        self.conv2 = ConvNormAct(mid_chs, out_chs, kernel_size=1, **ckwargs)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def zero_init_last(self):
        nn.init.zeros_(self.conv2.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.attn(x)
        x = self.conv2(x)
        x = self.drop_path(x) + shortcut
        return x

    def switch_to_deploy(self):
        self.conv1 = nn.Sequential(fuse_conv_bn(self.conv1.conv, self.conv1.bn), self.conv1.bn.act)
        self.conv2 = nn.Sequential(fuse_conv_bn(self.conv2.conv, self.conv2.bn), self.conv2.bn.act)


class CrossStage(nn.Module):
    """Cross Stage."""

    def __init__(self, in_chs, out_chs, stride, dilation, depth, block_ratio=1.0, bottle_ratio=1.0, expand_ratio=1.0, groups=1, first_dilation=None, avg_down=False, down_growth=False, cross_linear=False, block_dpr=None, block_fn=BottleneckBlock, **block_kwargs):
        super(CrossStage, self).__init__()
        first_dilation = first_dilation or dilation
        down_chs = out_chs if down_growth else in_chs
        self.expand_chs = exp_chs = int(round(out_chs * expand_ratio))
        block_out_chs = int(round(out_chs * block_ratio))
        conv_kwargs = dict(act_layer=block_kwargs.get('act_layer'), norm_layer=block_kwargs.get('norm_layer'))
        aa_layer = block_kwargs.pop('aa_layer', None)
        if stride != 1 or first_dilation != dilation:
            if avg_down:
                self.conv_down = nn.Sequential(nn.AvgPool2d(2) if stride == 2 else nn.Identity(), ConvNormActAa(in_chs, out_chs, kernel_size=1, stride=1, groups=groups, **conv_kwargs))
            else:
                self.conv_down = ConvNormActAa(in_chs, down_chs, kernel_size=3, stride=stride, dilation=first_dilation, groups=groups, aa_layer=aa_layer, **conv_kwargs)
            prev_chs = down_chs
        else:
            self.conv_down = nn.Identity()
            prev_chs = in_chs
        self.conv_exp = ConvNormAct(prev_chs, exp_chs, kernel_size=1, apply_act=not cross_linear, **conv_kwargs)
        prev_chs = exp_chs // 2
        self.blocks = nn.Sequential()
        for i in range(depth):
            self.blocks.add_module(str(i), block_fn(in_chs=prev_chs, out_chs=block_out_chs, dilation=dilation, bottle_ratio=bottle_ratio, groups=groups, drop_path=block_dpr[i] if block_dpr is not None else 0.0, **block_kwargs))
            prev_chs = block_out_chs
        self.conv_transition_b = ConvNormAct(prev_chs, exp_chs // 2, kernel_size=1, **conv_kwargs)
        self.conv_transition = ConvNormAct(exp_chs, out_chs, kernel_size=1, **conv_kwargs)

    def forward(self, x):
        x = self.conv_down(x)
        x = self.conv_exp(x)
        xs, xb = x.split(self.expand_chs // 2, dim=1)
        xb = self.blocks(xb)
        xb = self.conv_transition_b(xb).contiguous()
        out = self.conv_transition(torch.cat([xs, xb], dim=1))
        return out

    def switch_to_deploy(self):
        if type(self.conv_down) is nn.Sequential:
            self.conv_down = nn.Sequential(self.conv_down[0], fuse_conv_bn(self.conv_down[1].conv, self.conv_down[1].bn), self.conv_down[1].bn.act, self.conv_down[1].aa)
        elif type(self.conv_down) is nn.Identity:
            pass
        else:
            self.conv_down = nn.Sequential(fuse_conv_bn(self.conv_down.conv, self.conv_down.bn), self.conv_down.bn.act, self.conv_down.aa)
        self.conv_exp = nn.Sequential(fuse_conv_bn(self.conv_exp.conv, self.conv_exp.bn), self.conv_exp.bn.act)
        self.conv_transition_b = nn.Sequential(fuse_conv_bn(self.conv_transition_b.conv, self.conv_transition_b.bn), self.conv_transition_b.bn.act)
        self.conv_transition = nn.Sequential(fuse_conv_bn(self.conv_transition.conv, self.conv_transition.bn), self.conv_transition.bn.act)


class CrossStage3(nn.Module):
    """Cross Stage 3.
    Similar to CrossStage, but with only one transition conv for the output.
    """

    def __init__(self, in_chs, out_chs, stride, dilation, depth, block_ratio=1.0, bottle_ratio=1.0, expand_ratio=1.0, groups=1, first_dilation=None, avg_down=False, down_growth=False, cross_linear=False, block_dpr=None, block_fn=BottleneckBlock, **block_kwargs):
        super(CrossStage3, self).__init__()
        first_dilation = first_dilation or dilation
        down_chs = out_chs if down_growth else in_chs
        self.expand_chs = exp_chs = int(round(out_chs * expand_ratio))
        block_out_chs = int(round(out_chs * block_ratio))
        conv_kwargs = dict(act_layer=block_kwargs.get('act_layer'), norm_layer=block_kwargs.get('norm_layer'))
        aa_layer = block_kwargs.pop('aa_layer', None)
        if stride != 1 or first_dilation != dilation:
            if avg_down:
                self.conv_down = nn.Sequential(nn.AvgPool2d(2) if stride == 2 else nn.Identity(), ConvNormActAa(in_chs, out_chs, kernel_size=1, stride=1, groups=groups, **conv_kwargs))
            else:
                self.conv_down = ConvNormActAa(in_chs, down_chs, kernel_size=3, stride=stride, dilation=first_dilation, groups=groups, aa_layer=aa_layer, **conv_kwargs)
            prev_chs = down_chs
        else:
            self.conv_down = None
            prev_chs = in_chs
        self.conv_exp = ConvNormAct(prev_chs, exp_chs, kernel_size=1, apply_act=not cross_linear, **conv_kwargs)
        prev_chs = exp_chs // 2
        self.blocks = nn.Sequential()
        for i in range(depth):
            self.blocks.add_module(str(i), block_fn(in_chs=prev_chs, out_chs=block_out_chs, dilation=dilation, bottle_ratio=bottle_ratio, groups=groups, drop_path=block_dpr[i] if block_dpr is not None else 0.0, **block_kwargs))
            prev_chs = block_out_chs
        self.conv_transition = ConvNormAct(exp_chs, out_chs, kernel_size=1, **conv_kwargs)

    def forward(self, x):
        x = self.conv_down(x)
        x = self.conv_exp(x)
        x1, x2 = x.split(self.expand_chs // 2, dim=1)
        x1 = self.blocks(x1)
        out = self.conv_transition(torch.cat([x1, x2], dim=1))
        return out

    def switch_to_deploy(self):
        if self.conv_down is not None:
            if type(self.conv_down) is nn.Sequential:
                self.conv_down = nn.Sequential(self.conv_down[0], fuse_conv_bn(self.conv_down[1].conv, self.conv_down[1].bn), self.conv_down[1].bn.act, self.conv_down[1].aa)
            else:
                self.conv_down = nn.Sequential(fuse_conv_bn(self.conv_down.conv, self.conv_down.bn), self.conv_down.bn.act, self.conv_down.aa)
        self.conv_exp = nn.Sequential(fuse_conv_bn(self.conv_exp.conv, self.conv_exp.bn), self.conv_exp.bn.act)
        self.conv_transition = nn.Sequential(fuse_conv_bn(self.conv_transition.conv, self.conv_transition.bn), self.conv_transition.bn.act)


class DarkStage(nn.Module):
    """DarkNet stage."""

    def __init__(self, in_chs, out_chs, stride, dilation, depth, block_ratio=1.0, bottle_ratio=1.0, groups=1, first_dilation=None, avg_down=False, block_fn=BottleneckBlock, block_dpr=None, **block_kwargs):
        super(DarkStage, self).__init__()
        first_dilation = first_dilation or dilation
        conv_kwargs = dict(act_layer=block_kwargs.get('act_layer'), norm_layer=block_kwargs.get('norm_layer'))
        aa_layer = block_kwargs.pop('aa_layer', None)
        if avg_down:
            self.conv_down = nn.Sequential(nn.AvgPool2d(2) if stride == 2 else nn.Identity(), ConvNormActAa(in_chs, out_chs, kernel_size=1, stride=1, groups=groups, **conv_kwargs))
        else:
            self.conv_down = ConvNormActAa(in_chs, out_chs, kernel_size=3, stride=stride, dilation=first_dilation, groups=groups, aa_layer=aa_layer, **conv_kwargs)
        prev_chs = out_chs
        block_out_chs = int(round(out_chs * block_ratio))
        self.blocks = nn.Sequential()
        for i in range(depth):
            self.blocks.add_module(str(i), block_fn(in_chs=prev_chs, out_chs=block_out_chs, dilation=dilation, bottle_ratio=bottle_ratio, groups=groups, drop_path=block_dpr[i] if block_dpr is not None else 0.0, **block_kwargs))
            prev_chs = block_out_chs

    def forward(self, x):
        x = self.conv_down(x)
        x = self.blocks(x)
        return x

    def switch_to_deploy(self):
        if type(self.conv_down) is nn.Sequential:
            self.conv_down = nn.Sequential(self.conv_down[0], fuse_conv_bn(self.conv_down[1].conv, self.conv_down[1].bn), self.conv_down[1].bn.act, self.conv_down[1].aa)
        else:
            self.conv_down = nn.Sequential(fuse_conv_bn(self.conv_down.conv, self.conv_down.bn), self.conv_down.bn.act, self.conv_down.aa)


def _init_weights(module: 'nn.Module', name: 'str', head_bias: 'float'=0.0, flax=False):
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif flax:
            lecun_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                if 'mlp' in name:
                    nn.init.normal_(module.bias, std=1e-06)
                else:
                    nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.RNN, nn.GRU, nn.LSTM)):
        stdv = 1.0 / math.sqrt(module.hidden_size)
        for weight in module.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def _get_attn_fn(stage_args):
    attn_layer = stage_args.pop('attn_layer')
    attn_kwargs = stage_args.pop('attn_kwargs', None) or {}
    if attn_layer is not None:
        attn_layer = get_attn(attn_layer)
        if attn_kwargs:
            attn_layer = partial(attn_layer, **attn_kwargs)
    return attn_layer, stage_args


def _get_block_fn(stage_args):
    block_type = stage_args.pop('block_type')
    assert block_type in ('dark', 'edge', 'bottle')
    if block_type == 'dark':
        return DarkBlock, stage_args
    elif block_type == 'edge':
        return EdgeBlock, stage_args
    else:
        return BottleneckBlock, stage_args


def _get_stage_fn(stage_args):
    stage_type = stage_args.pop('stage_type')
    assert stage_type in ('dark', 'csp', 'cs3')
    if stage_type == 'dark':
        stage_args.pop('expand_ratio', None)
        stage_args.pop('cross_linear', None)
        stage_args.pop('down_growth', None)
        stage_fn = DarkStage
    elif stage_type == 'csp':
        stage_fn = CrossStage
    else:
        stage_fn = CrossStage3
    return stage_fn, stage_args


def create_csp_stages(cfg: 'CspModelCfg', drop_path_rate: 'float', output_stride: 'int', stem_feat: 'Dict[str, Any]'):
    cfg_dict = asdict(cfg.stages)
    num_stages = len(cfg.stages.depth)
    cfg_dict['block_dpr'] = [None] * num_stages if not drop_path_rate else [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(cfg.stages.depth)).split(cfg.stages.depth)]
    stage_args = [dict(zip(cfg_dict.keys(), values)) for values in zip(*cfg_dict.values())]
    block_kwargs = dict(act_layer=cfg.act_layer, norm_layer=cfg.norm_layer)
    dilation = 1
    net_stride = stem_feat['reduction']
    prev_chs = stem_feat['num_chs']
    prev_feat = stem_feat
    feature_info = []
    stages = []
    for stage_idx, stage_args in enumerate(stage_args):
        stage_fn, stage_args = _get_stage_fn(stage_args)
        block_fn, stage_args = _get_block_fn(stage_args)
        attn_fn, stage_args = _get_attn_fn(stage_args)
        stride = stage_args.pop('stride')
        if stride != 1 and prev_feat:
            feature_info.append(prev_feat)
        if net_stride >= output_stride and stride > 1:
            dilation *= stride
            stride = 1
        net_stride *= stride
        first_dilation = 1 if dilation in (1, 2) else 2
        stages += [stage_fn(prev_chs, **stage_args, stride=stride, first_dilation=first_dilation, dilation=dilation, block_fn=block_fn, aa_layer=cfg.aa_layer, attn_layer=attn_fn, **block_kwargs)]
        prev_chs = stage_args['out_chs']
        prev_feat = dict(num_chs=prev_chs, reduction=net_stride, module=f'stages.{stage_idx}')
    feature_info.append(prev_feat)
    return nn.Sequential(*stages), feature_info


def create_csp_stem(in_chans=3, out_chs=32, kernel_size=3, stride=2, pool='', padding='', act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None):
    stem = nn.Sequential()
    feature_info = []
    if not isinstance(out_chs, (tuple, list)):
        out_chs = [out_chs]
    stem_depth = len(out_chs)
    assert stem_depth
    assert stride in (1, 2, 4)
    prev_feat = None
    prev_chs = in_chans
    last_idx = stem_depth - 1
    stem_stride = 1
    for i, chs in enumerate(out_chs):
        conv_name = f'conv{i + 1}'
        conv_stride = 2 if i == 0 and stride > 1 or i == last_idx and stride > 2 and not pool else 1
        if conv_stride > 1 and prev_feat is not None:
            feature_info.append(prev_feat)
        stem.add_module(conv_name, ConvNormAct(prev_chs, chs, kernel_size, stride=conv_stride, padding=padding if i == 0 else '', act_layer=act_layer, norm_layer=norm_layer))
        stem_stride *= conv_stride
        prev_chs = chs
        prev_feat = dict(num_chs=prev_chs, reduction=stem_stride, module='.'.join(['stem', conv_name]))
    if pool:
        assert stride > 2
        if prev_feat is not None:
            feature_info.append(prev_feat)
        if aa_layer is not None:
            stem.add_module('pool', nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
            stem.add_module('aa', aa_layer(channels=prev_chs, stride=2))
            pool_name = 'aa'
        else:
            stem.add_module('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            pool_name = 'pool'
        stem_stride *= 2
        prev_feat = dict(num_chs=prev_chs, reduction=stem_stride, module='.'.join(['stem', pool_name]))
    feature_info.append(prev_feat)
    return stem, feature_info


class CspNet(nn.Module):
    """Cross Stage Partial base model.
    Paper: `CSPNet: A New Backbone that can Enhance Learning Capability of CNN` - https://arxiv.org/abs/1911.11929
    Ref Impl: https://github.com/WongKinYiu/CrossStagePartialNetworks
    NOTE: There are differences in the way I handle the 1x1 'expansion' conv in this impl vs the
    darknet impl. I did it this way for simplicity and less special cases.
    """

    def __init__(self, cfg: 'CspModelCfg', in_chans=3, num_classes=1000, output_stride=32, global_pool='avg', drop_rate=0.0, drop_path_rate=0.0, zero_init_last=True):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        assert output_stride in (8, 16, 32)
        layer_args = dict(act_layer=cfg.act_layer, norm_layer=cfg.norm_layer, aa_layer=cfg.aa_layer)
        self.feature_info = []
        self.stem, stem_feat_info = create_csp_stem(in_chans, **asdict(cfg.stem), **layer_args)
        self.feature_info.extend(stem_feat_info[:-1])
        self.stages, stage_feat_info = create_csp_stages(cfg, drop_path_rate=drop_path_rate, output_stride=output_stride, stem_feat=stem_feat_info[-1])
        prev_chs = stage_feat_info[-1]['num_chs']
        self.feature_info.extend(stage_feat_info)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_features = prev_chs
        self.head = nn.Linear(in_features=self.num_features, out_features=num_classes)
        named_apply(partial(_init_weights, zero_init_last=zero_init_last), self)

    def forward_features(self, x, need_fea=False):
        x = self.stem(x)
        if need_fea:
            features = []
            for layer in self.stages:
                x = layer(x)
                features.append(x)
            x = self.avgpool(x)
            return features, torch.flatten(x, start_dim=1, end_dim=3)
        else:
            x = self.stages(x)
            x = self.avgpool(x)
            return torch.flatten(x, start_dim=1, end_dim=3)

    def forward_head(self, x):
        return self.head(x)

    def forward(self, x, need_fea=False):
        if need_fea:
            features, features_fc = self.forward_features(x, need_fea=need_fea)
            x = self.forward_head(features_fc)
            return features, features_fc, x
        else:
            x = self.forward_features(x)
            x = self.forward_head(x)
            return x

    def cam_layer(self):
        return self.stages[-1]


class _DenseLayer(nn.Module):

    def __init__(self, num_input_features: 'int', growth_rate: 'int', bn_size: 'int', drop_rate: 'float', memory_efficient: 'bool'=False) ->None:
        super(_DenseLayer, self).__init__()
        self.norm1: 'nn.BatchNorm2d'
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.relu1: 'nn.ReLU'
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.conv1: 'nn.Conv2d'
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))
        self.norm2: 'nn.BatchNorm2d'
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.relu2: 'nn.ReLU'
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.conv2: 'nn.Conv2d'
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: 'List[Tensor]') ->Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def any_requires_grad(self, input: 'List[Tensor]') ->bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused
    def call_checkpoint_bottleneck(self, input: 'List[Tensor]') ->Tensor:

        def closure(*inputs):
            return self.bn_function(inputs)
        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method
    def forward(self, input: 'List[Tensor]') ->Tensor:
        pass

    @torch.jit._overload_method
    def forward(self, input: 'Tensor') ->Tensor:
        pass

    def forward(self, input: 'Tensor') ->Tensor:
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input
        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception('Memory Efficient not supported in JIT')
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers: 'int', num_input_features: 'int', bn_size: 'int', growth_rate: 'int', drop_rate: 'float', memory_efficient: 'bool'=False) ->None:
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate, memory_efficient=memory_efficient)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features: 'Tensor') ->Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features: 'int', num_output_features: 'int') ->None:
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(self, growth_rate: 'int'=32, block_config: 'Tuple[int, int, int, int]'=(6, 12, 24, 16), num_init_features: 'int'=64, bn_size: 'int'=4, drop_rate: 'float'=0, num_classes: 'int'=1000, memory_efficient: 'bool'=False) ->None:
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)), ('norm0', nn.BatchNorm2d(num_init_features)), ('relu0', nn.ReLU(inplace=True)), ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: 'Tensor', need_fea=False) ->Tensor:
        if need_fea:
            features, features_fc = self.forward_features(x, need_fea=True)
            out = self.classifier(features_fc)
            return features, features_fc, out
        else:
            features = self.forward_features(x)
            out = self.classifier(features)
            return out

    def forward_features(self, x, need_fea=False):
        if need_fea:
            input_size = x.size(2)
            scale = [4, 8, 16, 32]
            features = [None, None, None, None]
            for idx, layer in enumerate(self.features):
                x = layer(x)
                if input_size // x.size(2) in scale:
                    features[scale.index(input_size // x.size(2))] = x
            else:
                features[-1] = F.relu(features[-1], inplace=True)
            out = F.adaptive_avg_pool2d(features[-1], (1, 1))
            out = torch.flatten(out, 1)
            return features, out
        else:
            features = self.features(x)
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            return out

    def cam_layer(self):
        return self.features[-1]

    def switch_to_deploy(self):
        self.features.conv0 = fuse_conv_bn(self.features.conv0, self.features.norm0)
        del self.features.norm0


class DualPathBlock(nn.Module):

    def __init__(self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups, block_type='normal', b=False):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c
        self.inc = inc
        self.b = b
        if block_type == 'proj':
            self.key_stride = 1
            self.has_proj = True
        elif block_type == 'down':
            self.key_stride = 2
            self.has_proj = True
        else:
            assert block_type == 'normal'
            self.key_stride = 1
            self.has_proj = False
        self.c1x1_w_s1 = None
        self.c1x1_w_s2 = None
        if self.has_proj:
            if self.key_stride == 2:
                self.c1x1_w_s2 = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=2)
            else:
                self.c1x1_w_s1 = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=1)
        self.c1x1_a = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)
        self.c3x3_b = BnActConv2d(in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3, stride=self.key_stride, groups=groups)
        if b:
            self.c1x1_c = CatBnAct(in_chs=num_3x3_b)
            self.c1x1_c1 = create_conv2d(num_3x3_b, num_1x1_c, kernel_size=1)
            self.c1x1_c2 = create_conv2d(num_3x3_b, inc, kernel_size=1)
        else:
            self.c1x1_c = BnActConv2d(in_chs=num_3x3_b, out_chs=num_1x1_c + inc, kernel_size=1, stride=1)
            self.c1x1_c1 = None
            self.c1x1_c2 = None

    @torch.jit._overload_method
    def forward(self, x):
        pass

    @torch.jit._overload_method
    def forward(self, x):
        pass

    def forward(self, x) ->Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, tuple):
            x_in = torch.cat(x, dim=1)
        else:
            x_in = x
        if self.c1x1_w_s1 is None and self.c1x1_w_s2 is None:
            x_s1 = x[0]
            x_s2 = x[1]
        else:
            if self.c1x1_w_s1 is not None:
                x_s = self.c1x1_w_s1(x_in)
            else:
                x_s = self.c1x1_w_s2(x_in)
            x_s1 = x_s[:, :self.num_1x1_c, :, :]
            x_s2 = x_s[:, self.num_1x1_c:, :, :]
        x_in = self.c1x1_a(x_in)
        x_in = self.c3x3_b(x_in)
        x_in = self.c1x1_c(x_in)
        if self.c1x1_c1 is not None:
            out1 = self.c1x1_c1(x_in)
            out2 = self.c1x1_c2(x_in)
        else:
            out1 = x_in[:, :self.num_1x1_c, :, :]
            out2 = x_in[:, self.num_1x1_c:, :, :]
        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)
        return resid, dense


class DPN(nn.Module):

    def __init__(self, small=False, num_init_features=64, k_r=96, groups=32, global_pool='avg', b=False, k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128), output_stride=32, num_classes=1000, in_chans=3, drop_rate=0.0, fc_act_layer=nn.ELU):
        super(DPN, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.b = b
        assert output_stride == 32
        norm_layer = partial(BatchNormAct2d, eps=0.001)
        fc_norm_layer = partial(BatchNormAct2d, eps=0.001, act_layer=fc_act_layer, inplace=False)
        bw_factor = 1 if small else 4
        blocks = OrderedDict()
        blocks['conv1_1'] = ConvNormAct(in_chans, num_init_features, kernel_size=3 if small else 7, stride=2, norm_layer=norm_layer)
        blocks['conv1_pool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.feature_info = [dict(num_chs=num_init_features, reduction=2, module='features.conv1_1')]
        bw = 64 * bw_factor
        inc = inc_sec[0]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv2_1'] = DualPathBlock(num_init_features, r, r, bw, inc, groups, 'proj', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0] + 1):
            blocks['conv2_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        self.feature_info += [dict(num_chs=in_chs, reduction=4, module=f'features.conv2_{k_sec[0]}')]
        bw = 128 * bw_factor
        inc = inc_sec[1]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv3_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1] + 1):
            blocks['conv3_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        self.feature_info += [dict(num_chs=in_chs, reduction=8, module=f'features.conv3_{k_sec[1]}')]
        bw = 256 * bw_factor
        inc = inc_sec[2]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv4_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2] + 1):
            blocks['conv4_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        self.feature_info += [dict(num_chs=in_chs, reduction=16, module=f'features.conv4_{k_sec[2]}')]
        bw = 512 * bw_factor
        inc = inc_sec[3]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv5_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3] + 1):
            blocks['conv5_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        self.feature_info += [dict(num_chs=in_chs, reduction=32, module=f'features.conv5_{k_sec[3]}')]
        blocks['conv5_bn_ac'] = CatBnAct(in_chs, norm_layer=fc_norm_layer)
        self.num_features = in_chs
        self.features = nn.Sequential(blocks)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_features=self.num_features, out_features=num_classes)

    def forward_features(self, x, need_fea=False):
        if need_fea:
            input_size = x.size(2)
            scale = [4, 8, 16, 32]
            features = [None, None, None, None]
            for idx, layer in enumerate(self.features):
                x = layer(x)
                temp_x = torch.cat(x, dim=1) if type(x) is tuple else x
                if input_size // temp_x.size(2) in scale:
                    features[scale.index(input_size // temp_x.size(2))] = temp_x
            out = self.global_pool(x)
            return features, out.flatten(1)
        else:
            x = self.features(x)
            x = self.global_pool(x)
            return x.flatten(1)

    def forward_head(self, x):
        return self.classifier(x)

    def forward(self, x, need_fea=False):
        if need_fea:
            features, features_fc = self.forward_features(x, need_fea=need_fea)
            x = self.forward_head(features_fc)
            return features, features_fc, x
        else:
            x = self.forward_features(x)
            x = self.forward_head(x)
            return x

    def cam_layer(self):
        return self.features[-1]


class MBConv(nn.Module):

    def __init__(self, cnf: 'MBConvConfig', stochastic_depth_prob: 'float', norm_layer: 'Callable[..., nn.Module]', se_layer: 'Callable[..., nn.Module]'=SqueezeExcitation) ->None:
        super().__init__()
        if not 1 <= cnf.stride <= 2:
            raise ValueError('illegal stride value')
        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        layers: 'List[nn.Module]' = []
        activation_layer = nn.SiLU
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(Conv2dNormActivation(cnf.input_channels, expanded_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=activation_layer))
        layers.append(Conv2dNormActivation(expanded_channels, expanded_channels, kernel_size=cnf.kernel, stride=cnf.stride, groups=expanded_channels, norm_layer=norm_layer, activation_layer=activation_layer))
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))
        layers.append(Conv2dNormActivation(expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None))
        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, 'row')
        self.out_channels = cnf.out_channels

    def forward(self, input: 'Tensor') ->Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result

    def switch_to_deploy(self):
        new_block = []
        for layer in self.block:
            if type(layer) is Conv2dNormActivation:
                new_block.append(fuse_conv_bn(layer[0], layer[1]))
                if len(layer) > 2:
                    new_block.append(layer[2])
            else:
                new_block.append(layer)
        self.block = nn.Sequential(*new_block)


class FusedMBConv(nn.Module):

    def __init__(self, cnf: 'FusedMBConvConfig', stochastic_depth_prob: 'float', norm_layer: 'Callable[..., nn.Module]') ->None:
        super().__init__()
        if not 1 <= cnf.stride <= 2:
            raise ValueError('illegal stride value')
        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        layers: 'List[nn.Module]' = []
        activation_layer = nn.SiLU
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(Conv2dNormActivation(cnf.input_channels, expanded_channels, kernel_size=cnf.kernel, stride=cnf.stride, norm_layer=norm_layer, activation_layer=activation_layer))
            layers.append(Conv2dNormActivation(expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None))
        else:
            layers.append(Conv2dNormActivation(cnf.input_channels, cnf.out_channels, kernel_size=cnf.kernel, stride=cnf.stride, norm_layer=norm_layer, activation_layer=activation_layer))
        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, 'row')
        self.out_channels = cnf.out_channels

    def forward(self, input: 'Tensor') ->Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result

    def switch_to_deploy(self):
        new_block = []
        for layer in self.block:
            if type(layer) is Conv2dNormActivation:
                new_block.append(fuse_conv_bn(layer[0], layer[1]))
                if len(layer) > 2:
                    new_block.append(layer[2])
            else:
                new_block.append(layer)
        self.block = nn.Sequential(*new_block)


class EfficientNet(nn.Module):

    def __init__(self, inverted_residual_setting: 'Sequence[Union[MBConvConfig, FusedMBConvConfig]]', dropout: 'float', stochastic_depth_prob: 'float'=0.2, num_classes: 'int'=1000, norm_layer: 'Optional[Callable[..., nn.Module]]'=None, last_channel: 'Optional[int]'=None, **kwargs: Any) ->None:
        """
        EfficientNet V1 and V2 main class

        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
        """
        super().__init__()
        _log_api_usage_once(self)
        if not inverted_residual_setting:
            raise ValueError('The inverted_residual_setting should not be empty')
        elif not (isinstance(inverted_residual_setting, Sequence) and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])):
            raise TypeError('The inverted_residual_setting should be List[MBConvConfig]')
        if 'block' in kwargs:
            warnings.warn("The parameter 'block' is deprecated since 0.13 and will be removed 0.15. Please pass this information on 'MBConvConfig.block' instead.")
            if kwargs['block'] is not None:
                for s in inverted_residual_setting:
                    if isinstance(s, MBConvConfig):
                        s.block = kwargs['block']
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        layers: 'List[nn.Module]' = []
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(Conv2dNormActivation(3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU))
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: 'List[nn.Module]' = []
            for _ in range(cnf.num_layers):
                block_cnf = copy.copy(cnf)
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        layers.append(Conv2dNormActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.SiLU))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True), nn.Linear(lastconv_output_channels, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: 'Tensor', need_fea=False) ->Tensor:
        if need_fea:
            features, features_fc = self.forward_features(x, need_fea)
            x = self.classifier(features_fc)
            return features, features_fc, x
        else:
            x = self.forward_features(x)
            x = self.classifier(x)
            return x

    def forward(self, x: 'Tensor', need_fea=False) ->Tensor:
        return self._forward_impl(x, need_fea)

    def forward_features(self, x, need_fea=False):
        if need_fea:
            input_size = x.size(2)
            scale = [4, 8, 16, 32]
            features = [None, None, None, None]
            for idx, layer in enumerate(self.features):
                x = layer(x)
                if input_size // x.size(2) in scale:
                    features[scale.index(input_size // x.size(2))] = x
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return features, x
        else:
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return x

    def cam_layer(self):
        return self.features[-5:]

    def switch_to_deploy(self):
        new_block = []
        for layer in self.features:
            if type(layer) is Conv2dNormActivation:
                new_block.append(fuse_conv_bn(layer[0], layer[1]))
                if len(layer) > 2:
                    new_block.append(layer[2])
            else:
                new_block.append(layer)
        self.features = nn.Sequential(*new_block)


class SELayer(nn.Module):

    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel))

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y


class GhostModule(nn.Module):

    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False), nn.BatchNorm2d(init_channels), nn.ReLU(inplace=True) if relu else nn.Sequential())
        self.cheap_operation = nn.Sequential(nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False), nn.BatchNorm2d(new_channels), nn.ReLU(inplace=True) if relu else nn.Sequential())

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

    def switch_to_deploy(self):
        self.primary_conv = nn.Sequential(fuse_conv_bn(self.primary_conv[0], self.primary_conv[1]), self.primary_conv[2])
        self.cheap_operation = nn.Sequential(fuse_conv_bn(self.cheap_operation[0], self.cheap_operation[1]), self.cheap_operation[2])


def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, groups=inp, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True) if relu else nn.Sequential())


class GhostBottleneck(nn.Module):

    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]
        self.conv = nn.Sequential(GhostModule(inp, hidden_dim, kernel_size=1, relu=True), depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride == 2 else nn.Sequential(), SELayer(hidden_dim) if use_se else nn.Sequential(), GhostModule(hidden_dim, oup, kernel_size=1, relu=False))
        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(depthwise_conv(inp, inp, 3, stride, relu=True), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

    def switch_to_deploy(self):
        if len(self.conv[1]) > 0:
            self.conv = nn.Sequential(self.conv[0], fuse_conv_bn(self.conv[1][0], self.conv[1][1]), self.conv[1][2], self.conv[2], self.conv[3])
        else:
            self.conv = nn.Sequential(self.conv[0], self.conv[2], self.conv[3])
        if len(self.shortcut) != 0:
            self.shortcut = nn.Sequential(fuse_conv_bn(self.shortcut[0][0], self.shortcut[0][1]), self.shortcut[0][2], fuse_conv_bn(self.shortcut[1], self.shortcut[2]))


class GhostNet(nn.Module):

    def __init__(self, cfgs, num_classes=1000, width_mult=1.0):
        super(GhostNet, self).__init__()
        self.cfgs = cfgs
        output_channel = _make_divisible(16 * width_mult, 4)
        layers = [nn.Sequential(nn.Conv2d(3, output_channel, 3, 2, 1, bias=False), nn.BatchNorm2d(output_channel), nn.ReLU(inplace=True))]
        input_channel = output_channel
        block = GhostBottleneck
        for k, exp_size, c, use_se, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4)
            hidden_channel = _make_divisible(exp_size * width_mult, 4)
            layers.append(block(input_channel, hidden_channel, output_channel, k, s, use_se))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        output_channel = _make_divisible(exp_size * width_mult, 4)
        self.squeeze = nn.Sequential(nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False), nn.BatchNorm2d(output_channel), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
        input_channel = output_channel
        output_channel = 1280
        self.classifier = nn.Sequential(nn.Linear(input_channel, output_channel, bias=False), nn.BatchNorm1d(output_channel), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(output_channel, num_classes))
        self._initialize_weights()

    def forward(self, x, need_fea=False):
        if need_fea:
            features, features_fc = self.forward_features(x, need_fea)
            x = self.classifier(features_fc)
            return features, features_fc, x
        else:
            x = self.forward_features(x)
            x = self.classifier(x)
            return x

    def forward_features(self, x, need_fea=False):
        if need_fea:
            input_size = x.size(2)
            scale = [4, 8, 16, 32]
            features = [None, None, None, None]
            for idx, layer in enumerate(self.features):
                x = layer(x)
                if input_size // x.size(2) in scale:
                    features[scale.index(input_size // x.size(2))] = x
            x = self.squeeze(x)
            return features, x.view(x.size(0), -1)
        else:
            x = self.features(x)
            x = self.squeeze(x)
            return x.view(x.size(0), -1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def cam_layer(self):
        return self.features[-1]

    def switch_to_deploy(self):
        self.features[0] = nn.Sequential(fuse_conv_bn(self.features[0][0], self.features[0][1]), self.features[0][2])
        self.squeeze = nn.Sequential(fuse_conv_bn(self.squeeze[0], self.squeeze[1]), self.squeeze[2], self.squeeze[3])


class _InvertedResidual(nn.Module):

    def __init__(self, in_ch: 'int', out_ch: 'int', kernel_size: 'int', stride: 'int', expansion_factor: 'int', bn_momentum: 'float'=0.1) ->None:
        super(_InvertedResidual, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        mid_ch = in_ch * expansion_factor
        self.apply_residual = in_ch == out_ch and stride == 1
        self.layers = nn.Sequential(nn.Conv2d(in_ch, mid_ch, 1, bias=False), nn.BatchNorm2d(mid_ch, momentum=bn_momentum), nn.ReLU(inplace=True), nn.Conv2d(mid_ch, mid_ch, kernel_size, padding=kernel_size // 2, stride=stride, groups=mid_ch, bias=False), nn.BatchNorm2d(mid_ch, momentum=bn_momentum), nn.ReLU(inplace=True), nn.Conv2d(mid_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch, momentum=bn_momentum))

    def forward(self, input: 'Tensor') ->Tensor:
        if self.apply_residual:
            return self.layers(input) + input
        else:
            return self.layers(input)

    def switch_to_deploy(self):
        self.layers = nn.Sequential(fuse_conv_bn(self.layers[0], self.layers[1]), self.layers[2], fuse_conv_bn(self.layers[3], self.layers[4]), self.layers[5], fuse_conv_bn(self.layers[6], self.layers[7]))


_BN_MOMENTUM = 1 - 0.9997


def _round_to_multiple_of(val: 'float', divisor: 'int', round_up_bias: 'float'=0.9) ->int:
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(alpha: 'float') ->List[int]:
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


def _stack(in_ch: 'int', out_ch: 'int', kernel_size: 'int', stride: 'int', exp_factor: 'int', repeats: 'int', bn_momentum: 'float') ->nn.Sequential:
    """ Creates a stack of inverted residuals. """
    assert repeats >= 1
    first = _InvertedResidual(in_ch, out_ch, kernel_size, stride, exp_factor, bn_momentum=bn_momentum)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(_InvertedResidual(out_ch, out_ch, kernel_size, 1, exp_factor, bn_momentum=bn_momentum))
    return nn.Sequential(first, *remaining)


class MNASNet(torch.nn.Module):
    """ MNASNet, as described in https://arxiv.org/pdf/1807.11626.pdf. This
    implements the B1 variant of the model.
    >>> model = MNASNet(1.0, num_classes=1000)
    >>> x = torch.rand(1, 3, 224, 224)
    >>> y = model(x)
    >>> y.dim()
    2
    >>> y.nelement()
    1000
    """
    _version = 2

    def __init__(self, alpha: 'float', num_classes: 'int'=1000, dropout: 'float'=0.2) ->None:
        super(MNASNet, self).__init__()
        assert alpha > 0.0
        self.alpha = alpha
        self.num_classes = num_classes
        depths = _get_depths(alpha)
        layers = [nn.Conv2d(3, depths[0], 3, padding=1, stride=2, bias=False), nn.BatchNorm2d(depths[0], momentum=_BN_MOMENTUM), nn.ReLU(inplace=True), nn.Conv2d(depths[0], depths[0], 3, padding=1, stride=1, groups=depths[0], bias=False), nn.BatchNorm2d(depths[0], momentum=_BN_MOMENTUM), nn.ReLU(inplace=True), nn.Conv2d(depths[0], depths[1], 1, padding=0, stride=1, bias=False), nn.BatchNorm2d(depths[1], momentum=_BN_MOMENTUM), _stack(depths[1], depths[2], 3, 2, 3, 3, _BN_MOMENTUM), _stack(depths[2], depths[3], 5, 2, 3, 3, _BN_MOMENTUM), _stack(depths[3], depths[4], 5, 2, 6, 3, _BN_MOMENTUM), _stack(depths[4], depths[5], 3, 1, 6, 2, _BN_MOMENTUM), _stack(depths[5], depths[6], 5, 2, 6, 4, _BN_MOMENTUM), _stack(depths[6], depths[7], 3, 1, 6, 1, _BN_MOMENTUM), nn.Conv2d(depths[7], 1280, 1, padding=0, stride=1, bias=False), nn.BatchNorm2d(1280, momentum=_BN_MOMENTUM), nn.ReLU(inplace=True)]
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True), nn.Linear(1280, num_classes))
        self._initialize_weights()

    def switch_to_deploy(self):
        self.layers = nn.Sequential(fuse_conv_bn(self.layers[0], self.layers[1]), self.layers[2], fuse_conv_bn(self.layers[3], self.layers[4]), self.layers[5], fuse_conv_bn(self.layers[6], self.layers[7]), self.layers[8:14], fuse_conv_bn(self.layers[14], self.layers[15]), self.layers[16])

    def forward(self, x: 'Tensor', need_fea=False) ->Tensor:
        if need_fea:
            features, features_fc = self.forward_features(x, need_fea)
            x = self.classifier(features_fc)
            return features, features_fc, x
        else:
            x = self.forward_features(x)
            x = self.classifier(x)
            return x

    def forward_features(self, x, need_fea=False):
        if need_fea:
            input_size = x.size(2)
            scale = [4, 8, 16, 32]
            features = [None, None, None, None]
            for idx, layer in enumerate(self.layers):
                x = layer(x)
                if input_size // x.size(2) in scale:
                    features[scale.index(input_size // x.size(2))] = x
            return features, x.mean([2, 3])
        else:
            x = self.layers(x)
            x = x.mean([2, 3])
            return x

    def _initialize_weights(self) ->None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='sigmoid')
                nn.init.zeros_(m.bias)

    def cam_layer(self):
        return self.layers[-1]

    def _load_from_state_dict(self, state_dict: 'Dict', prefix: 'str', local_metadata: 'Dict', strict: 'bool', missing_keys: 'List[str]', unexpected_keys: 'List[str]', error_msgs: 'List[str]') ->None:
        version = local_metadata.get('version', None)
        assert version in [1, 2]
        if version == 1 and not self.alpha == 1.0:
            depths = _get_depths(self.alpha)
            v1_stem = [nn.Conv2d(3, 32, 3, padding=1, stride=2, bias=False), nn.BatchNorm2d(32, momentum=_BN_MOMENTUM), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1, stride=1, groups=32, bias=False), nn.BatchNorm2d(32, momentum=_BN_MOMENTUM), nn.ReLU(inplace=True), nn.Conv2d(32, 16, 1, padding=0, stride=1, bias=False), nn.BatchNorm2d(16, momentum=_BN_MOMENTUM), _stack(16, depths[2], 3, 2, 3, 3, _BN_MOMENTUM)]
            for idx, layer in enumerate(v1_stem):
                self.layers[idx] = layer
            self._version = 1
            warnings.warn('A new version of MNASNet model has been implemented. Your checkpoint was saved using the previous version. This checkpoint will load and work as before, but you may want to upgrade by training a newer model or transfer learning from an updated ImageNet checkpoint.', UserWarning)
        super(MNASNet, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


def channel_shuffle(x: 'Tensor', groups: 'int') ->Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class InvertedResidual(nn.Module):

    def __init__(self, inp: 'int', oup: 'int', stride: 'int') ->None:
        super(InvertedResidual, self).__init__()
        if not 1 <= stride <= 3:
            raise ValueError('illegal stride value')
        self.stride = stride
        branch_features = oup // 2
        assert self.stride != 1 or inp == branch_features << 1
        if self.stride > 1:
            self.branch1 = nn.Sequential(self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1), nn.BatchNorm2d(inp), nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(branch_features), nn.ReLU(inplace=True))
        else:
            self.branch1 = nn.Sequential()
        self.branch2 = nn.Sequential(nn.Conv2d(inp if self.stride > 1 else branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(branch_features), nn.ReLU(inplace=True), self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1), nn.BatchNorm2d(branch_features), nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(branch_features), nn.ReLU(inplace=True))

    @staticmethod
    def depthwise_conv(i: 'int', o: 'int', kernel_size: 'int', stride: 'int'=1, padding: 'int'=0, bias: 'bool'=False) ->nn.Conv2d:
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x: 'Tensor') ->Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out

    def switch_to_deploy(self):
        if len(self.branch1) > 0:
            self.branch1 = nn.Sequential(fuse_conv_bn(self.branch1[0], self.branch1[1]), fuse_conv_bn(self.branch1[2], self.branch1[3]), self.branch1[4])
        self.branch2 = nn.Sequential(fuse_conv_bn(self.branch2[0], self.branch2[1]), self.branch2[2], fuse_conv_bn(self.branch2[3], self.branch2[4]), fuse_conv_bn(self.branch2[5], self.branch2[6]), self.branch2[7])


class MobileNetV2(nn.Module):

    def __init__(self, num_classes: 'int'=1000, width_mult: 'float'=1.0, inverted_residual_setting: 'Optional[List[List[int]]]'=None, round_nearest: 'int'=8, block: 'Optional[Callable[..., nn.Module]]'=None, norm_layer: 'Optional[Callable[..., nn.Module]]'=None) ->None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()
        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        input_channel = 32
        last_channel = 1280
        if inverted_residual_setting is None:
            inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError('inverted_residual_setting should be non-empty or a 4-element list, got {}'.format(inverted_residual_setting))
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: 'List[nn.Module]' = [ConvNormActivation(3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        features.append(ConvNormActivation(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.last_channel, num_classes))
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

    def _forward_impl(self, x: 'Tensor', need_fea=False) ->Tensor:
        if need_fea:
            features, features_fc = self.forward_features(x, need_fea)
            x = self.classifier(features_fc)
            return features, features_fc, x
        else:
            x = self.forward_features(x)
            x = self.classifier(x)
            return x

    def forward(self, x: 'Tensor', need_fea=False) ->Tensor:
        return self._forward_impl(x, need_fea)

    def forward_features(self, x, need_fea=False):
        if need_fea:
            input_size = x.size(2)
            scale = [4, 8, 16, 32]
            features = [None, None, None, None]
            for idx, layer in enumerate(self.features):
                x = layer(x)
                if input_size // x.size(2) in scale:
                    features[scale.index(input_size // x.size(2))] = x
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            return features, x
        else:
            x = self.features(x)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            return x

    def cam_layer(self):
        return self.features[-1]

    def switch_to_deploy(self):
        self.features[0] = nn.Sequential(fuse_conv_bn(self.features[0][0], self.features[0][1]), self.features[0][2])
        self.features[-1] = nn.Sequential(fuse_conv_bn(self.features[-1][0], self.features[-1][1]), self.features[-1][2])


class InvertedResidualConfig:

    def __init__(self, input_channels: 'int', kernel: 'int', expanded_channels: 'int', out_channels: 'int', use_se: 'bool', activation: 'str', stride: 'int', dilation: 'int', width_mult: 'float'):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == 'HS'
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: 'int', width_mult: 'float'):
        return _make_divisible(channels * width_mult, 8)


class MobileNetV3(nn.Module):

    def __init__(self, inverted_residual_setting: 'List[InvertedResidualConfig]', last_channel: 'int', num_classes: 'int'=1000, block: 'Optional[Callable[..., nn.Module]]'=None, norm_layer: 'Optional[Callable[..., nn.Module]]'=None, **kwargs: Any) ->None:
        """
        MobileNet V3 main class

        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()
        if not inverted_residual_setting:
            raise ValueError('The inverted_residual_setting should not be empty')
        elif not (isinstance(inverted_residual_setting, Sequence) and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError('The inverted_residual_setting should be List[InvertedResidualConfig]')
        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        layers: 'List[nn.Module]' = []
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(Conv2dNormActivation(3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.Hardswish))
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(Conv2dNormActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Hardswish))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(lastconv_output_channels, last_channel), nn.Hardswish(inplace=True), nn.Dropout(p=0.2, inplace=True), nn.Linear(last_channel, num_classes))
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

    def switch_to_deploy(self):
        new_layers = []
        for i in range(len(self.features)):
            if type(self.features[i]) is Conv2dNormActivation:
                new_layers.append(fuse_conv_bn(self.features[i][0], self.features[i][1]))
                if len(self.features[i]) == 3:
                    new_layers.append(self.features[i][2])
            else:
                new_layers.append(self.features[i])
        self.features = nn.Sequential(*new_layers)

    def _forward_impl(self, x: 'Tensor', need_fea=False) ->Tensor:
        if need_fea:
            features, features_fc = self.forward_features(x, need_fea)
            x = self.classifier(features_fc)
            return features, features_fc, x
        else:
            x = self.forward_features(x)
            x = self.classifier(x)
            return x

    def forward(self, x: 'Tensor', need_fea=False) ->Tensor:
        return self._forward_impl(x, need_fea)

    def forward_features(self, x, need_fea=False):
        if need_fea:
            input_size = x.size(2)
            scale = [4, 8, 16, 32]
            features = [None, None, None, None]
            for idx, layer in enumerate(self.features):
                x = layer(x)
                if input_size // x.size(2) in scale:
                    features[scale.index(input_size // x.size(2))] = x
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return features, x
        else:
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return x

    def cam_layer(self):
        return self.features[-1]


def hard_sigmoid(x, inplace: 'bool'=False):
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(x + 3.0) / 6.0


class SqueezeExcite(nn.Module):

    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None, act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):

    def __init__(self, in_chs, out_chs, kernel_size, stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, 'bn1'):
            x = self.bn1(x)
        x = self.act1(x)
        return x

    def switch_to_deploy(self):
        self.conv = fuse_conv_bn(self.conv, self.bn1)
        del self.bn1


class RepGhostModule(nn.Module):

    def __init__(self, inp, oup, kernel_size=1, dw_size=3, stride=1, relu=True, deploy=False, reparam_bn=True, reparam_identity=False):
        super(RepGhostModule, self).__init__()
        init_channels = oup
        new_channels = oup
        self.deploy = deploy
        self.primary_conv = nn.Sequential(nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False), nn.BatchNorm2d(init_channels), nn.ReLU(inplace=True) if relu else nn.Sequential())
        fusion_conv = []
        fusion_bn = []
        if not deploy and reparam_bn:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.BatchNorm2d(init_channels))
        if not deploy and reparam_identity:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.Identity())
        self.fusion_conv = nn.Sequential(*fusion_conv)
        self.fusion_bn = nn.Sequential(*fusion_bn)
        self.cheap_operation = nn.Sequential(nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=deploy), nn.BatchNorm2d(new_channels) if not deploy else nn.Sequential())
        if deploy:
            self.cheap_operation = self.cheap_operation[0]
        if relu:
            self.relu = nn.ReLU(inplace=False)
        else:
            self.relu = nn.Sequential()

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            x2 = x2 + bn(conv(x1))
        return self.relu(x2)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.cheap_operation[0], self.cheap_operation[1])
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            kernel, bias = self._fuse_bn_tensor(conv, bn, kernel3x3.shape[0], kernel3x3.device)
            kernel3x3 += self._pad_1x1_to_3x3_tensor(kernel)
            bias3x3 += bias
        return kernel3x3, bias3x3

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    @staticmethod
    def _fuse_bn_tensor(conv, bn, in_channels=None, device=None):
        in_channels = in_channels if in_channels else bn.running_mean.shape[0]
        device = device if device else bn.weight.device
        if isinstance(conv, nn.Conv2d):
            kernel = conv.weight
            assert conv.bias is None
        else:
            assert isinstance(conv, nn.Identity)
            kernel_value = np.zeros((in_channels, 1, 1, 1), dtype=np.float32)
            for i in range(in_channels):
                kernel_value[i, 0, 0, 0] = 1
            kernel = torch.from_numpy(kernel_value)
        if isinstance(bn, nn.BatchNorm2d):
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std
        assert isinstance(bn, nn.Identity)
        return kernel, torch.zeros(in_channels)

    def switch_to_deploy(self):
        if len(self.fusion_conv) == 0 and len(self.fusion_bn) == 0:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.cheap_operation = nn.Conv2d(in_channels=self.cheap_operation[0].in_channels, out_channels=self.cheap_operation[0].out_channels, kernel_size=self.cheap_operation[0].kernel_size, padding=self.cheap_operation[0].padding, dilation=self.cheap_operation[0].dilation, groups=self.cheap_operation[0].groups, bias=True)
        self.cheap_operation.weight.data = kernel
        self.cheap_operation.bias.data = bias
        self.__delattr__('fusion_conv')
        self.__delattr__('fusion_bn')
        self.fusion_conv = []
        self.fusion_bn = []
        self.deploy = True
        self.primary_conv = nn.Sequential(fuse_conv_bn(self.primary_conv[0], self.primary_conv[1]), self.primary_conv[2])


class RepGhostBottleneck(nn.Module):
    """RepGhost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3, stride=1, se_ratio=0.0, shortcut=True, reparam=True, reparam_bn=True, reparam_identity=False, deploy=False):
        super(RepGhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride
        self.enable_shortcut = shortcut
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.ghost1 = RepGhostModule(in_chs, mid_chs, relu=True, reparam_bn=reparam and reparam_bn, reparam_identity=reparam and reparam_identity, deploy=deploy)
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride, padding=(dw_kernel_size - 1) // 2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None
        self.ghost2 = RepGhostModule(mid_chs, out_chs, relu=False, reparam_bn=reparam and reparam_bn, reparam_identity=reparam and reparam_identity, deploy=deploy)
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride, padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False), nn.BatchNorm2d(in_chs), nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False), nn.BatchNorm2d(out_chs))

    def forward(self, x):
        residual = x
        x1 = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x1)
            x = self.bn_dw(x)
        else:
            x = x1
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        if not self.enable_shortcut and self.in_chs == self.out_chs and self.stride == 1:
            return x
        return x + self.shortcut(residual)

    def switch_to_deploy(self):
        if len(self.shortcut) != 0:
            self.shortcut = nn.Sequential(fuse_conv_bn(self.shortcut[0], self.shortcut[1]), fuse_conv_bn(self.shortcut[2], self.shortcut[3]))


def repghost_model_convert(model: 'torch.nn.Module', save_path=None, do_copy=True):
    """
    taken from from https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


class RepGhostNet(nn.Module):

    def __init__(self, cfgs, num_classes=1000, width=1.0, dropout=0.2, shortcut=True, reparam=True, reparam_bn=True, reparam_identity=False, deploy=False):
        super(RepGhostNet, self).__init__()
        self.cfgs = cfgs
        self.dropout = dropout
        self.num_classes = num_classes
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel
        stages = []
        block = RepGhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s, se_ratio=se_ratio, shortcut=shortcut, reparam=reparam, reparam_bn=reparam_bn, reparam_identity=reparam_identity, deploy=deploy))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))
        output_channel = _make_divisible(exp_size * width * 2, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel
        self.blocks = nn.Sequential(*stages)
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x, need_fea=False):
        if need_fea:
            features, features_fc = self.forward_features(x, need_fea)
            x = self.classifier(features_fc)
            return features, features_fc, x
        else:
            x = self.forward_features(x)
            x = self.classifier(x)
            return x

    def forward_features(self, x, need_fea=False):
        input_size = x.size(2)
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        if need_fea:
            scale = [4, 8, 16, 32]
            features = [None, None, None, None]
            for idx, layer in enumerate(self.blocks):
                x = layer(x)
                if input_size // x.size(2) in scale:
                    features[scale.index(input_size // x.size(2))] = x
            x = self.global_pool(x)
            x = self.conv_head(x)
            x = self.act2(x)
            return features, x.view(x.size(0), -1)
        else:
            x = self.blocks(x)
            x = self.global_pool(x)
            x = self.conv_head(x)
            x = self.act2(x)
            return x.view(x.size(0), -1)

    def convert_to_deploy(self):
        repghost_model_convert(self, do_copy=False)


class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        assert kernel_size == 3
        assert padding == 1
        padding_11 = padding - kernel_size // 2
        self.nonlinearity = nn.ReLU()
        if use_se:
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / (self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt()).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / (self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt()).reshape(-1, 1, 1, 1).detach()
        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()
        return l2_loss_eq_kernel + l2_loss_circle

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
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
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels, kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride, padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class RepVGG(nn.Module):

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False, use_se=False, use_checkpoint=False):
        super(RepVGG, self).__init__()
        assert len(width_multiplier) == 4
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        assert 0 not in self.override_groups_map
        self.use_se = use_se
        self.use_checkpoint = use_checkpoint
        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, deploy=self.deploy, use_se=self.use_se)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, groups=cur_groups, deploy=self.deploy, use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.ModuleList(blocks)

    def forward(self, x, need_fea=False):
        if need_fea:
            features, features_fc = self.forward_features(x, need_fea)
            out = self.linear(features_fc)
            return features, features_fc, out
        else:
            out = self.forward_features(x)
            out = self.linear(out)
            return out

    def forward_features(self, x, need_fea=False):
        out = self.stage0(x)
        if need_fea:
            features = []
            for stage in (self.stage1, self.stage2, self.stage3, self.stage4):
                for block in stage:
                    if self.use_checkpoint:
                        out = checkpoint.checkpoint(block, out)
                    else:
                        out = block(out)
                features.append(out)
            out = self.gap(out)
            out = out.view(out.size(0), -1)
            return features, out
        else:
            for stage in (self.stage1, self.stage2, self.stage3, self.stage4):
                for block in stage:
                    if self.use_checkpoint:
                        out = checkpoint.checkpoint(block, out)
                    else:
                        out = block(out)
            out = self.gap(out)
            out = out.view(out.size(0), -1)
            return out

    def cam_layer(self):
        return self.stage4


class rSoftMax(nn.Module):

    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class DropBlock2D(object):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """

    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=True, radix=2, reduction_factor=4, rectify=False, rectify_avg=False, norm_layer=None, dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            self.conv = RFConv2d(in_channels, channels * radix, kernel_size, stride, padding, dilation, groups=groups * radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, channels * radix, kernel_size, stride, padding, dilation, groups=groups * radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels * radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels * radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn and hasattr(self, 'bn0'):
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)
        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            if torch.__version__ < '1.5':
                splited = torch.split(x, int(rchannel // self.radix), dim=1)
            else:
                splited = torch.split(x, rchannel // self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        if self.radix > 1:
            if torch.__version__ < '1.5':
                attens = torch.split(atten, int(rchannel // self.radix), dim=1)
            else:
                attens = torch.split(atten, rchannel // self.radix, dim=1)
            out = sum([(att * split) for att, split in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()

    def switch_to_deploy(self):
        if self.use_bn:
            try:
                self.conv = fuse_conv_bn(self.conv, self.bn0)
                del self.bn0
            except:
                pass


def conv1x1(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=1, padding=0):
    """1x1 convolution"""
    return [('{}_{}/conv'.format(module_name, postfix), nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)), ('{}_{}/norm'.format(module_name, postfix), nn.BatchNorm2d(out_channels)), ('{}_{}/relu'.format(module_name, postfix), nn.ReLU(inplace=True))]


def conv3x3(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [('{}_{}/conv'.format(module_name, postfix), nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)), ('{}_{}/norm'.format(module_name, postfix), nn.BatchNorm2d(out_channels)), ('{}_{}/relu'.format(module_name, postfix), nn.ReLU(inplace=True))]


class Bottleneck(nn.Module):
    expansion: 'int' = 4

    def __init__(self, inplanes: 'int', planes: 'int', stride: 'int'=1, downsample: 'Optional[nn.Module]'=None, groups: 'int'=1, base_width: 'int'=64, dilation: 'int'=1, norm_layer: 'Optional[Callable[..., nn.Module]]'=None) ->None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: 'Tensor') ->Tensor:
        identity = x
        out = self.conv1(x)
        if hasattr(self, 'bn1'):
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if hasattr(self, 'bn2'):
            out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        if hasattr(self, 'bn3'):
            out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

    def switch_to_deploy(self):
        self.conv1 = fuse_conv_bn(self.conv1, self.bn1)
        del self.bn1
        self.conv2 = fuse_conv_bn(self.conv2, self.bn2)
        del self.bn2
        self.conv3 = fuse_conv_bn(self.conv3, self.bn3)
        del self.bn3


class BasicBlock(nn.Module):
    expansion: 'int' = 1

    def __init__(self, inplanes: 'int', planes: 'int', stride: 'int'=1, downsample: 'Optional[nn.Module]'=None, groups: 'int'=1, base_width: 'int'=64, dilation: 'int'=1, norm_layer: 'Optional[Callable[..., nn.Module]]'=None) ->None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: 'Tensor') ->Tensor:
        identity = x
        out = self.conv1(x)
        if hasattr(self, 'bn1'):
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if hasattr(self, 'bn2'):
            out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

    def switch_to_deploy(self):
        self.conv1 = fuse_conv_bn(self.conv1, self.bn1)
        del self.bn1
        self.conv2 = fuse_conv_bn(self.conv2, self.bn2)
        del self.bn2


class ResNet(nn.Module):

    def __init__(self, block: 'Type[Union[BasicBlock, Bottleneck]]', layers: 'List[int]', num_classes: 'int'=1000, zero_init_residual: 'bool'=False, groups: 'int'=1, width_per_group: 'int'=64, replace_stride_with_dilation: 'Optional[List[bool]]'=None, norm_layer: 'Optional[Callable[..., nn.Module]]'=None) ->None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def switch_to_deploy(self):
        self.conv1 = fuse_conv_bn(self.conv1, self.bn1)
        del self.bn1

    def _make_layer(self, block: 'Type[Union[BasicBlock, Bottleneck]]', planes: 'int', blocks: 'int', stride: 'int'=1, dilate: 'bool'=False) ->nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x: 'Tensor', need_fea=False) ->Tensor:
        if need_fea:
            features, features_fc = self.forward_features(x, need_fea)
            x = self.fc(features_fc)
            return features, features_fc, x
        else:
            x = self.forward_features(x)
            x = self.fc(x)
            return x

    def forward(self, x: 'Tensor', need_fea=False) ->Tensor:
        return self._forward_impl(x, need_fea)

    def forward_features(self, x, need_fea=False):
        x = self.conv1(x)
        if hasattr(self, 'bn1'):
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if need_fea:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            x = self.avgpool(x4)
            x = torch.flatten(x, 1)
            return [x1, x2, x3, x4], x
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return x

    def cam_layer(self):
        return self.layer4


class RNNIdentity(nn.Module):

    def __init__(self, *args, **kwargs):
        super(RNNIdentity, self).__init__()

    def forward(self, x: 'Tensor') ->Tuple[Tensor, None]:
        return x, None


class RNNBase(nn.Module):

    def __init__(self, input_size, hidden_size=None, num_layers: 'int'=1, bias: 'bool'=True, bidirectional: 'bool'=True):
        super().__init__()
        self.rnn = RNNIdentity()

    def forward(self, x):
        B, H, W, C = x.shape
        x, _ = self.rnn(x.view(B, -1, C))
        return x.view(B, H, W, -1)


class RNN(RNNBase):

    def __init__(self, input_size, hidden_size=None, num_layers: 'int'=1, bias: 'bool'=True, bidirectional: 'bool'=True, nonlinearity='tanh'):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional, nonlinearity=nonlinearity)


class GRU(RNNBase):

    def __init__(self, input_size, hidden_size=None, num_layers: 'int'=1, bias: 'bool'=True, bidirectional: 'bool'=True):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional)
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional)


class LSTM(RNNBase):

    def __init__(self, input_size, hidden_size=None, num_layers: 'int'=1, bias: 'bool'=True, bidirectional: 'bool'=True):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional)


class RNN2DBase(nn.Module):

    def __init__(self, input_size: 'int', hidden_size: 'int', num_layers: 'int'=1, bias: 'bool'=True, bidirectional: 'bool'=True, union='cat', with_fc=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 2 * hidden_size if bidirectional else hidden_size
        self.union = union
        self.with_vertical = True
        self.with_horizontal = True
        self.with_fc = with_fc
        if with_fc:
            if union == 'cat':
                self.fc = nn.Linear(2 * self.output_size, input_size)
            elif union == 'add':
                self.fc = nn.Linear(self.output_size, input_size)
            elif union == 'vertical':
                self.fc = nn.Linear(self.output_size, input_size)
                self.with_horizontal = False
            elif union == 'horizontal':
                self.fc = nn.Linear(self.output_size, input_size)
                self.with_vertical = False
            else:
                raise ValueError('Unrecognized union: ' + union)
        elif union == 'cat':
            pass
            if 2 * self.output_size != input_size:
                raise ValueError(f'The output channel {2 * self.output_size} is different from the input channel {input_size}.')
        elif union == 'add':
            pass
            if self.output_size != input_size:
                raise ValueError(f'The output channel {self.output_size} is different from the input channel {input_size}.')
        elif union == 'vertical':
            if self.output_size != input_size:
                raise ValueError(f'The output channel {self.output_size} is different from the input channel {input_size}.')
            self.with_horizontal = False
        elif union == 'horizontal':
            if self.output_size != input_size:
                raise ValueError(f'The output channel {self.output_size} is different from the input channel {input_size}.')
            self.with_vertical = False
        else:
            raise ValueError('Unrecognized union: ' + union)
        self.rnn_v = RNNIdentity()
        self.rnn_h = RNNIdentity()

    def forward(self, x):
        B, H, W, C = x.shape
        if self.with_vertical:
            v = x.permute(0, 2, 1, 3)
            v = v.reshape(-1, H, C)
            v, _ = self.rnn_v(v)
            v = v.reshape(B, W, H, -1)
            v = v.permute(0, 2, 1, 3)
        if self.with_horizontal:
            h = x.reshape(-1, W, C)
            h, _ = self.rnn_h(h)
            h = h.reshape(B, H, W, -1)
        if self.with_vertical and self.with_horizontal:
            if self.union == 'cat':
                x = torch.cat([v, h], dim=-1)
            else:
                x = v + h
        elif self.with_vertical:
            x = v
        elif self.with_horizontal:
            x = h
        if self.with_fc:
            x = self.fc(x)
        return x


class RNN2D(RNN2DBase):

    def __init__(self, input_size: 'int', hidden_size: 'int', num_layers: 'int'=1, bias: 'bool'=True, bidirectional: 'bool'=True, union='cat', with_fc=True, nonlinearity='tanh'):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional, union, with_fc)
        if self.with_vertical:
            self.rnn_v = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional, nonlinearity=nonlinearity)
        if self.with_horizontal:
            self.rnn_h = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional, nonlinearity=nonlinearity)


class LSTM2D(RNN2DBase):

    def __init__(self, input_size: 'int', hidden_size: 'int', num_layers: 'int'=1, bias: 'bool'=True, bidirectional: 'bool'=True, union='cat', with_fc=True):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional, union, with_fc)
        if self.with_vertical:
            self.rnn_v = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional)
        if self.with_horizontal:
            self.rnn_h = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional)


class GRU2D(RNN2DBase):

    def __init__(self, input_size: 'int', hidden_size: 'int', num_layers: 'int'=1, bias: 'bool'=True, bidirectional: 'bool'=True, union='cat', with_fc=True):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional, union, with_fc)
        if self.with_vertical:
            self.rnn_v = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional)
        if self.with_horizontal:
            self.rnn_h = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional)


class Shuffle(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        if self.training:
            B, H, W, C = x.shape
            r = torch.randperm(H * W)
            x = x.reshape(B, -1, C)
            x = x[:, r, :].reshape(B, H, W, -1)
        return x


class Downsample2D(nn.Module):

    def __init__(self, input_dim, output_dim, patch_size):
        super().__init__()
        self.down = nn.Conv2d(input_dim, output_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.down(x)
        x = x.permute(0, 2, 3, 1)
        return x


def get_stage(index, layers, patch_sizes, embed_dims, hidden_sizes, mlp_ratios, block_layer, rnn_layer, mlp_layer, norm_layer, act_layer, num_layers, bidirectional, union, with_fc, drop=0.0, drop_path_rate=0.0, **kwargs):
    assert len(layers) == len(patch_sizes) == len(embed_dims) == len(hidden_sizes) == len(mlp_ratios)
    blocks = []
    for block_idx in range(layers[index]):
        drop_path = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(block_layer(embed_dims[index], hidden_sizes[index], mlp_ratio=mlp_ratios[index], rnn_layer=rnn_layer, mlp_layer=mlp_layer, norm_layer=norm_layer, act_layer=act_layer, num_layers=num_layers, bidirectional=bidirectional, union=union, with_fc=with_fc, drop=drop, drop_path=drop_path))
    if index < len(embed_dims) - 1:
        blocks.append(Downsample2D(embed_dims[index], embed_dims[index + 1], patch_sizes[index + 1]))
    blocks = nn.Sequential(*blocks)
    return blocks


class ShuffleNetV2(nn.Module):

    def __init__(self, stages_repeats: 'List[int]', stages_out_channels: 'List[int]', num_classes: 'int'=1000, inverted_residual: 'Callable[..., nn.Module]'=InvertedResidual) ->None:
        super(ShuffleNetV2, self).__init__()
        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False), nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True))
        input_channels = output_channels
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage2: 'nn.Sequential'
        self.stage3: 'nn.Sequential'
        self.stage4: 'nn.Sequential'
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True))
        self.fc = nn.Linear(output_channels, num_classes)

    def switch_to_deploy(self):
        self.conv1 = nn.Sequential(fuse_conv_bn(self.conv1[0], self.conv1[1]), self.conv1[2])
        self.conv5 = nn.Sequential(fuse_conv_bn(self.conv5[0], self.conv5[1]), self.conv5[2])

    def _forward_impl(self, x: 'Tensor', need_fea=False) ->Tensor:
        if need_fea:
            features, features_fc = self.forward_features(x, need_fea)
            return features, features_fc, self.fc(features_fc)
        else:
            x = self.forward_features(x)
            x = self.fc(x)
            return x

    def forward(self, x: 'Tensor', need_fea=False) ->Tensor:
        return self._forward_impl(x, need_fea)

    def forward_features(self, x, need_fea=False):
        x = self.conv1(x)
        x = self.maxpool(x)
        if need_fea:
            x2 = self.stage2(x)
            x3 = self.stage3(x2)
            x4 = self.stage4(x3)
            x4 = self.conv5(x4)
            return [x, x2, x3, x4], x4.mean([2, 3])
        else:
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.conv5(x)
            x = x.mean([2, 3])
            return x

    def cam_layer(self):
        return self.stage4


class VGG(nn.Module):

    def __init__(self, features: 'nn.Module', num_classes: 'int'=1000, init_weights: 'bool'=True) ->None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, num_classes))
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) ->None:
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

    def forward(self, x: 'torch.Tensor', need_fea=False) ->torch.Tensor:
        if need_fea:
            features, features_fc = self.forward_features(x, need_fea)
            return features, features_fc, self.classifier(features_fc)
        else:
            x = self.forward_features(x)
            x = self.classifier(x)
            return x

    def forward_features(self, x, need_fea=False):
        if need_fea:
            input_size = x.size(2)
            scale = [4, 8, 16, 32]
            features = [None, None, None, None]
            for idx, layer in enumerate(self.features):
                x = layer(x)
                if input_size // x.size(2) in scale:
                    features[scale.index(input_size // x.size(2))] = x
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return features, x
        else:
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return x

    def cam_layer(self):
        return self.features[-1]

    def switch_to_deploy(self):
        new_features = []
        for i in range(len(self.features)):
            if type(self.features[i]) is nn.BatchNorm2d:
                new_features[-1] = fuse_conv_bn(new_features[-1], self.features[i])
            else:
                new_features.append(self.features[i])
        self.features = nn.Sequential(*new_features)


class _OSA_module(nn.Module):

    def __init__(self, in_ch, stage_ch, concat_ch, layer_per_block, module_name, identity=False):
        super(_OSA_module, self).__init__()
        self.identity = identity
        self.layers = nn.ModuleList()
        in_channel = in_ch
        for i in range(layer_per_block):
            self.layers.append(nn.Sequential(OrderedDict(conv3x3(in_channel, stage_ch, module_name, i))))
            in_channel = stage_ch
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(OrderedDict(conv1x1(in_channel, concat_ch, module_name, 'concat')))

    def forward(self, x):
        identity_feat = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        xt = self.concat(x)
        if self.identity:
            xt = xt + identity_feat
        return xt

    def switch_to_deploy(self):
        new_features = []
        for i in range(len(self.layers)):
            if type(self.layers[i]) is nn.Sequential:
                new_features.append(nn.Sequential(fuse_conv_bn(self.layers[i][0], self.layers[i][1]), self.layers[i][2]))
            elif type(self.layers[i]) is nn.BatchNorm2d:
                new_features[-1] = fuse_conv_bn(new_features[-1], self.layers[i])
                None
            else:
                new_features.append(self.layers[i])
        self.layers = nn.Sequential(*new_features)
        self.concat = nn.Sequential(fuse_conv_bn(self.concat[0], self.concat[1]), self.concat[2])


class _OSA_stage(nn.Sequential):

    def __init__(self, in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num):
        super(_OSA_stage, self).__init__()
        if not stage_num == 2:
            self.add_module('Pooling', nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        module_name = f'OSA{stage_num}_1'
        self.add_module(module_name, _OSA_module(in_ch, stage_ch, concat_ch, layer_per_block, module_name))
        for i in range(block_per_stage - 1):
            module_name = f'OSA{stage_num}_{i + 2}'
            self.add_module(module_name, _OSA_module(concat_ch, stage_ch, concat_ch, layer_per_block, module_name, identity=True))


class VoVNet(nn.Module):

    def __init__(self, config_stage_ch, config_concat_ch, block_per_stage, layer_per_block, num_classes=1000):
        super(VoVNet, self).__init__()
        stem = conv3x3(3, 64, 'stem', '1', 2)
        stem += conv3x3(64, 64, 'stem', '2', 1)
        stem += conv3x3(64, 128, 'stem', '3', 2)
        self.add_module('stem', nn.Sequential(OrderedDict(stem)))
        stem_out_ch = [128]
        in_ch_list = stem_out_ch + config_concat_ch[:-1]
        self.stage_names = []
        for i in range(4):
            name = 'stage%d' % (i + 2)
            self.stage_names.append(name)
            self.add_module(name, _OSA_stage(in_ch_list[i], config_stage_ch[i], config_concat_ch[i], block_per_stage[i], layer_per_block, i + 2))
        self.classifier = nn.Linear(config_concat_ch[-1], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def switch_to_deploy(self):
        self.stem = nn.Sequential(fuse_conv_bn(self.stem[0], self.stem[1]), self.stem[2], fuse_conv_bn(self.stem[3], self.stem[4]), self.stem[5], fuse_conv_bn(self.stem[6], self.stem[7]), self.stem[8])

    def forward(self, x, need_fea=False):
        if need_fea:
            features, features_fc = self.forward_features(x, need_fea)
            return features, features_fc, self.classifier(features_fc)
        else:
            x = self.forward_features(x)
            x = self.classifier(x)
            return x

    def forward_features(self, x, need_fea=False):
        if need_fea:
            features = []
            x = self.stem(x)
            for name in self.stage_names:
                x = getattr(self, name)(x)
                features.append(x)
            x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
            return features, x
        else:
            x = self.stem(x)
            for name in self.stage_names:
                x = getattr(self, name)(x)
            x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
            return x

    def cam_layer(self):
        return getattr(self, self.stage_names[-1])


class SoftTarget(nn.Module):

    def __init__(self, T=4):
        super(SoftTarget, self).__init__()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.T = T

    def forward(self, student_pred, teacher_pred):
        student_pred_logsoftmax = torch.log_softmax(student_pred / self.T, dim=1)
        teacher_pred_softmax = torch.softmax(teacher_pred / self.T, dim=1)
        kd_loss = self.kl_loss(student_pred_logsoftmax, teacher_pred_softmax) * self.T * self.T
        return kd_loss

    def __str__(self):
        return 'SoftTarget'


class MGD(nn.Module):
    """PyTorch version of `Masked Generative Distillation`
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        name (str): the loss name of the layer
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00007
        lambda_mgd (float, optional): masked ratio. Defaults to 0.15
    """

    def __init__(self, student_channels, teacher_channels, alpha_mgd=7e-05, lambda_mgd=0.15):
        super(MGD, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None
        self.generation = nn.Sequential(nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1))

    def forward(self, preds_S, preds_T):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]
        if self.align is not None:
            preds_S = self.align(preds_S)
        loss = self.get_dis_loss(preds_S, preds_T) * self.alpha_mgd
        return loss

    def get_dis_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape
        device = preds_S.device
        mat = torch.rand((N, C, 1, 1))
        mat = torch.where(mat < self.lambda_mgd, 0, 1)
        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation(masked_fea)
        dis_loss = loss_mse(new_fea, preds_T) / N
        return dis_loss

    def __str__(self):
        return 'MGD'


class SP(nn.Module):
    """
    Similarity-Preserving Knowledge Distillation
    https://arxiv.org/pdf/1907.09682.pdf
    """

    def __init__(self):
        super(SP, self).__init__()

    def matmul_and_normalize(self, z):
        z = torch.flatten(z, 1)
        return F.normalize(torch.matmul(z, torch.t(z)), 1)

    def forward(self, fm_s, fm_t):
        g_t = self.matmul_and_normalize(fm_t)
        g_s = self.matmul_and_normalize(fm_s)
        sp_loss = torch.norm(g_t - g_s) ** 2
        sp_loss = sp_loss.sum()
        return sp_loss / fm_s.size(0) ** 2

    def __str__(self):
        return 'SP'


class AT(nn.Module):
    """
	Paying More Attention to Attention: Improving the Performance of Convolutional
	Neural Netkworks wia Attention Transfer
	https://arxiv.org/pdf/1612.03928.pdf
	"""

    def __init__(self):
        super(AT, self).__init__()

    def forward(self, fm_s, fm_t):
        fm_s_att = self.attention_map(fm_s)
        fm_t_att = self.attention_map(fm_t)
        return (fm_s_att - fm_t_att).pow(2).mean()

    def attention_map(self, x):
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

    def __str__(self):
        return 'AT'


class PolyLoss(torch.nn.Module):
    """
    PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
    <https://arxiv.org/abs/2204.12511>
    """

    def __init__(self, label_smoothing: 'float'=0.0, weight: 'torch.Tensor'=None, epsilon=2.0):
        super().__init__()
        self.epsilon = epsilon
        self.label_smoothing = label_smoothing
        self.weight = weight

    def forward(self, outputs, targets):
        ce = F.cross_entropy(outputs, targets, label_smoothing=self.label_smoothing, weight=self.weight)
        pt = F.one_hot(targets, outputs.size()[1]) * F.softmax(outputs, 1)
        return (ce + self.epsilon * (1.0 - pt.sum(dim=1))).mean()


class CrossEntropyLoss(nn.Module):

    def __init__(self, label_smoothing: 'float'=0.0, weight: 'torch.Tensor'=None):
        super(CrossEntropyLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor') ->torch.Tensor:
        return self.cross_entropy(input, target)


class FocalLoss(nn.Module):

    def __init__(self, label_smoothing: 'float'=0.0, weight: 'torch.Tensor'=None, gamma: 'float'=2.0):
        super(FocalLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.weight = weight
        self.gamma = gamma

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor') ->torch.Tensor:
        target_onehot = F.one_hot(target, num_classes=input.size(1))
        target_onehot_labelsmoothing = torch.clamp(target_onehot.float(), min=self.label_smoothing / (input.size(1) - 1), max=1.0 - self.label_smoothing)
        input_softmax = F.softmax(input, dim=1) + 1e-07
        input_logsoftmax = torch.log(input_softmax)
        ce = -1 * input_logsoftmax * target_onehot_labelsmoothing
        fl = torch.pow(1 - input_softmax, self.gamma) * ce
        fl = fl.sum(1) * self.weight[target.long()]
        return fl.mean()


class RDropLoss(nn.Module):

    def __init__(self, loss, a=0.3):
        super(RDropLoss, self).__init__()
        self.loss = loss
        self.a = a

    def forward(self, input, target: 'torch.Tensor') ->torch.Tensor:
        if type(input) is list:
            input1, input2 = input
            main_loss = (self.loss(input1, target) + self.loss(input2, target)) * 0.5
            kl_loss = self.compute_kl_loss(input1, input2)
            return main_loss + self.a * kl_loss
        else:
            return self.loss(input, target)

    def compute_kl_loss(self, p, q, pad_mask=None):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.0)
            q_loss.masked_fill_(pad_mask, 0.0)
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()
        loss = (p_loss + q_loss) / 2
        return loss


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AT,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Block,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvBnAct,
     lambda: ([], {'in_chs': 4, 'out_chs': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (DenseNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (Downsample2D,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'patch_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GRU2D,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GhostBottleneck,
     lambda: ([], {'inp': 4, 'hidden_dim': 4, 'oup': 4, 'kernel_size': 4, 'stride': 1, 'use_se': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GhostModule,
     lambda: ([], {'inp': 4, 'oup': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LSTM2D,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MGD,
     lambda: ([], {'student_channels': 4, 'teacher_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MNASNet,
     lambda: ([], {'alpha': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (RDropLoss,
     lambda: ([], {'loss': MSELoss()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (RNN2D,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RNNBase,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RNNIdentity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RepGhostBottleneck,
     lambda: ([], {'in_chs': 4, 'mid_chs': 4, 'out_chs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RepGhostModule,
     lambda: ([], {'inp': 4, 'oup': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SEBlock,
     lambda: ([], {'input_channels': 4, 'internal_neurons': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SELayer,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Shuffle,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ShuffleNetV2,
     lambda: ([], {'stages_repeats': [4, 4, 4], 'stages_out_channels': [4, 4, 4, 4, 4]}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (SoftTarget,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SplAtConv2d,
     lambda: ([], {'in_channels': 4, 'channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SqueezeExcite,
     lambda: ([], {'in_chs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (_DenseBlock,
     lambda: ([], {'num_layers': 1, 'num_input_features': 4, 'bn_size': 4, 'growth_rate': 4, 'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (_DenseLayer,
     lambda: ([], {'num_input_features': 4, 'growth_rate': 4, 'bn_size': 4, 'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (_InvertedResidual,
     lambda: ([], {'in_ch': 4, 'out_ch': 4, 'kernel_size': 3, 'stride': 1, 'expansion_factor': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (_OSA_stage,
     lambda: ([], {'in_ch': 4, 'stage_ch': 4, 'concat_ch': 4, 'block_per_stage': 4, 'layer_per_block': 1, 'stage_num': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (_Transition,
     lambda: ([], {'num_input_features': 4, 'num_output_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (rSoftMax,
     lambda: ([], {'radix': 4, 'cardinality': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

