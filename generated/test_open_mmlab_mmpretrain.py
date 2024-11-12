
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


import logging


from time import perf_counter


import numpy as np


import torch


import re


import time


from copy import deepcopy


from typing import OrderedDict


import math


import matplotlib.pyplot as plt


from scipy import stats


import copy


from functools import partial


import warnings


from abc import abstractmethod


from math import ceil


from typing import Callable


from typing import Iterable


from typing import List


from typing import Optional


from typing import Tuple


from typing import Union


from collections import defaultdict


from itertools import chain


from typing import Dict


import torch.nn as nn


from torch.nn.modules.activation import Hardswish


from torch.optim import SGD


from torch.optim import AdamW


from torch.optim.adamw import AdamW


from torch.optim import RMSprop


from torch.nn.modules.batchnorm import BatchNorm2d


from typing import Iterator


from typing import Sized


from torch.utils.data import Sampler


from collections.abc import Sequence


import torchvision.transforms.functional as F


import inspect


import numbers


import string


from enum import EnumMeta


from numbers import Number


from typing import Sequence


import torchvision


from torchvision import transforms


from torchvision.transforms.transforms import InterpolationMode


import itertools


from torch.functional import Tensor


from torch.nn import GroupNorm


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn.modules.instancenorm import _InstanceNorm


from torch.utils.data import DataLoader


from torch import Tensor


from torch.optim.optimizer import Optimizer


from torch.optim import Optimizer


from torch import nn


from torch.nn import LayerNorm


from itertools import product


import torch.nn.functional as F


import torchvision.ops.boxes as boxes


import torch.utils.checkpoint as cp


from torch.jit.annotations import List


import torch.utils.checkpoint as checkpoint


from torch.utils.checkpoint import checkpoint


from torch.autograd import Function as Function


from torch.nn import functional as F


from abc import ABCMeta


from collections import OrderedDict


import torch.distributed as dist


import functools


from collections import ChainMap


from torch import distributed as torch_dist


from torch import device


import torch.utils.checkpoint


from torch.nn import CrossEntropyLoss


import random


from torch import einsum


from torch.nn.parameter import Parameter


from typing import Any


from functools import reduce


from math import cos


from math import pi


import collections.abc


from itertools import repeat


from torch.autograd import Function


from torch.autograd.function import once_differentiable


from torch.cuda.amp import custom_bwd


from torch.cuda.amp import custom_fwd


from torch.nn.init import constant_


from torch.nn.init import xavier_uniform_


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.autograd import gradcheck


from torch.utils.data import Dataset


import sklearn.metrics


from torch.nn.modules import GroupNorm


from types import MethodType


from torch.nn.modules import AvgPool2d


from tensorflow.python.training import py_checkpoint_reader


from collections import namedtuple


class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(dim, norm_layer, in_format='channels_last', out_format='channels_last', eps=1e-06):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


class DenseBlock(nn.Module):
    """DenseNet Blocks."""

    def __init__(self, num_layers, in_channels, bn_size, growth_rate, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), drop_rate=0.0, memory_efficient=False):
        super(DenseBlock, self).__init__()
        self.block = nn.ModuleList([DenseLayer(in_channels + i * growth_rate, growth_rate=growth_rate, bn_size=bn_size, norm_cfg=norm_cfg, act_cfg=act_cfg, drop_rate=drop_rate, memory_efficient=memory_efficient) for i in range(num_layers)])

    def forward(self, init_features):
        features = [init_features]
        for layer in self.block:
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseTransition(nn.Sequential):
    """DenseNet Transition Layers."""

    def __init__(self, in_channels, out_channels, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')):
        super(DenseTransition, self).__init__()
        self.add_module('norm', build_norm_layer(norm_cfg, in_channels)[1])
        self.add_module('act', build_activation_layer(act_cfg))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class Flat(nn.Module):
    """Flat the input from (B, C, H, W) to (B, H*W, C)."""

    def __init__(self):
        super().__init__()

    def forward(self, x: 'torch.Tensor'):
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    """Mlp implemented by with 1*1 convolutions.

    Input: Tensor with shape [B, C, H, W].
    Output: Tensor with shape [B, C, H, W].
    Args:
        in_features (int): Dimension of input features.
        hidden_features (int): Dimension of hidden features.
        out_features (int): Dimension of output features.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.0.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_cfg=dict(type='GELU'), drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def get_rel_pos(q_size: 'int', k_size: 'int', rel_pos: 'torch.Tensor') ->torch.Tensor:
    """Get relative positional embeddings according to the relative positions
    of query and key sizes.

    Borrowed from https://github.com/facebookresearch/segment-anything/

    Args:
        q_size (int): Size of query q.
        k_size (int): Size of key k.
        rel_pos (torch.Tensor): Relative position embeddings (L, C).

    Returns:
        torch.Tensor: Extracted positional embeddings according to relative
        positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1), size=max_rel_dist, mode='linear')
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = q_coords - k_coords + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn: 'torch.Tensor', q: 'torch.Tensor', rel_pos_h: 'torch.Tensor', rel_pos_w: 'torch.Tensor', q_size: 'Tuple[int, int]', k_size: 'Tuple[int, int]') ->torch.Tensor:
    """Borrowed from https://github.com/facebookresearch/segment-anything/

    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

    Args:
        attn (torch.Tensor): Attention map.
        q (torch.Tensor): Query q in the attention layer with shape
            (B, q_h * q_w, C).
        rel_pos_h (torch.Tensor): Relative position embeddings (Lh, C) for
            height axis.
        rel_pos_w (torch.Tensor): Relative position embeddings (Lw, C) for
            width axis.
        q_size (tuple): Spatial sequence size of query q with (q_h, q_w).
        k_size (tuple): Spatial sequence size of key k with (k_h, k_w).

    Returns:
        torch.Tensor: Attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum('bhwc,hkc->bhwk', r_q, Rh)
    rel_w = torch.einsum('bhwc,wkc->bhwk', r_q, Rw)
    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(B, q_h * q_w, k_h * k_w)
    return attn


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings.

    Borrowed from https://github.com/facebookresearch/segment-anything/

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        use_rel_pos (bool):Whether to use relative position embedding.
            Defaults to False.
        input_size (int, optional): Input resolution for calculating the
            relative positional parameter size. Defaults to None.
    """

    def __init__(self, embed_dims: 'int', num_heads: 'int'=8, qkv_bias: 'bool'=True, use_rel_pos: 'bool'=False, input_size: 'Optional[Tuple[int, int]]'=None) ->None:
        super().__init__()
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = head_embed_dims ** -0.5
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, 'Input size must be provided if using relative position embed.'
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_embed_dims))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_embed_dims))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn = q * self.scale @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        return x


class BlockWithRPE(nn.Module):
    """HiViT block.

    Args:
        input_size (int): Input size.
        dim (int): Number of input dims.
        num_heads (int): Number of attention heads.
        mlp_ratio (int): Ratio of MLP hidden dim to embedding dim.
        qkv_bias (bool): Enable bias for qkv projections if True.
        qk_scale (float): The number of divider after q@k. Default to None.
        drop (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path (float): Stochastic depth rate. Defaults to 0.
        rpe (bool): If True, add relative position embedding to
            the patch embedding.
        layer_scale_init_value (float): Layer-scale init values. Defaults to 0.
        act_layer: MLP activation layer.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
    """

    def __init__(self, input_size, dim, num_heads=0.0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, rpe=True, layer_scale_init_value=0.0, act_layer=nn.GELU, norm_cfg=dict(type='LN')):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        with_attn = num_heads > 0.0
        self.norm1 = build_norm_layer(norm_cfg, dim) if with_attn else None
        self.attn = Attention(input_size, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, rpe=rpe) if with_attn else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if layer_scale_init_value > 0:
            self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if with_attn else None
            self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rpe_index=None, mask=None):
        if self.attn is not None:
            if self.gamma_1 is not None:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rpe_index, mask))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x), rpe_index, mask))
        if self.gamma_2 is not None:
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def _ntuple(n):
    """A `to_tuple` function generator.

    It returns a function, this function will repeat the input to a tuple of
    length ``n`` if the input is not an Iterable object, otherwise, return the
    input directly.

    Args:
        n (int): The number of the target length.
    """

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


class PatchMerge(nn.Module):
    """PatchMerge for HiViT.

    Args:
        dim (int): Number of input channels.
        norm_cfg (dict): Config dict for normalization layer.
    """

    def __init__(self, dim, norm_cfg):
        super().__init__()
        self.norm = build_norm_layer(norm_cfg, dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)

    def forward(self, x, *args, **kwargs):
        is_main_stage = len(x.shape) == 3
        if is_main_stage:
            B, N, C = x.shape
            S = int(math.sqrt(N))
            x = x.reshape(B, S // 2, 2, S // 2, 2, C).permute(0, 1, 3, 2, 4, 5).reshape(B, -1, 2, 2, C)
        x0 = x[..., 0::2, 0::2, :]
        x1 = x[..., 1::2, 0::2, :]
        x2 = x[..., 0::2, 1::2, :]
        x3 = x[..., 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        if is_main_stage:
            x = x[:, :, 0, 0, :]
        return x


class HorNetLayerNorm(nn.Module):
    """An implementation of LayerNorm of HorNet.

    The differences between HorNetLayerNorm & torch LayerNorm:
        1. Supports two data formats channels_last or channels_first.
    Args:
        normalized_shape (int or list or torch.Size): input shape from an
            expected input of size.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        data_format (str): The ordering of the dimensions in the inputs.
            channels_last corresponds to inputs with shape (batch_size, height,
            width, channels) while channels_first corresponds to inputs with
            shape (batch_size, channels, height, width).
            Defaults to 'channels_last'.
    """

    def __init__(self, normalized_shape, eps=1e-06, data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise ValueError('data_format must be channels_last or channels_first')
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


class GlobalLocalFilter(nn.Module):
    """A GlobalLocalFilter of HorNet.

    Args:
        dim (int): Number of input channels.
        h (int): Height of complex_weight.
            Defaults to 14.
        w (int): Width of complex_weight.
            Defaults to 8.
    """

    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
        self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        self.pre_norm = HorNetLayerNorm(dim, eps=1e-06, data_format='channels_first')
        self.post_norm = HorNetLayerNorm(dim, eps=1e-06, data_format='channels_first')

    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)
        x2 = x2
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')
        weight = self.complex_weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3, 0, 1, 2), size=x2.shape[2:4], mode='bilinear', align_corners=True).permute(1, 2, 3, 0)
        weight = torch.view_as_complex(weight.contiguous())
        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')
        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x


def get_dwconv(dim, kernel_size, bias=True):
    """build a pepth-wise convolution."""
    return nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=bias, groups=dim)


class gnConv(nn.Module):
    """A gnConv of HorNet.

    Args:
        dim (int): Number of input channels.
        order (int): Order of gnConv.
            Defaults to 5.
        dw_cfg (dict): The Config for dw conv.
            Defaults to ``dict(type='DW', kernel_size=7)``.
        scale (float): Scaling parameter of gflayer outputs.
            Defaults to 1.0.
    """

    def __init__(self, dim, order=5, dw_cfg=dict(type='DW', kernel_size=7), scale=1.0):
        super().__init__()
        self.order = order
        self.dims = [(dim // 2 ** i) for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)
        cfg = copy.deepcopy(dw_cfg)
        dw_type = cfg.pop('type')
        assert dw_type in ['DW', 'GF'], 'dw_type should be `DW` or `GF`'
        if dw_type == 'DW':
            self.dwconv = get_dwconv(sum(self.dims), **cfg)
        elif dw_type == 'GF':
            self.dwconv = GlobalLocalFilter(sum(self.dims), **cfg)
        self.proj_out = nn.Conv2d(dim, dim, 1)
        self.projs = nn.ModuleList([nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)])
        self.scale = scale

    def forward(self, x):
        x = self.proj_in(x)
        y, x = torch.split(x, (self.dims[0], sum(self.dims)), dim=1)
        x = self.dwconv(x) * self.scale
        dw_list = torch.split(x, self.dims, dim=1)
        x = y * dw_list[0]
        for i in range(self.order - 1):
            x = self.projs[i](x) * dw_list[i + 1]
        x = self.proj_out(x)
        return x


class LayerScale(nn.Module):
    """LayerScale layer.

    Args:
        dim (int): Dimension of input features.
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 1e-5.
        inplace (bool): inplace: can optionally do the
            operation in-place. Defaults to False.
        data_format (str): The input data format, could be 'channels_last'
             or 'channels_first', representing (B, C, H, W) and
             (B, N, C) format data respectively. Defaults to 'channels_last'.
    """

    def __init__(self, dim: 'int', layer_scale_init_value: 'Union[float, torch.Tensor]'=1e-05, inplace: 'bool'=False, data_format: 'str'='channels_last'):
        super().__init__()
        assert data_format in ('channels_last', 'channels_first'), "'data_format' could only be channels_last or channels_first."
        self.inplace = inplace
        self.data_format = data_format
        self.weight = nn.Parameter(torch.ones(dim) * layer_scale_init_value)

    def forward(self, x):
        if self.data_format == 'channels_first':
            if self.inplace:
                return x.mul_(self.weight.view(-1, 1, 1))
            else:
                return x * self.weight.view(-1, 1, 1)
        return x.mul_(self.weight) if self.inplace else x * self.weight


class HorNetBlock(nn.Module):
    """A block of HorNet.

    Args:
        dim (int): Number of input channels.
        order (int): Order of gnConv.
            Defaults to 5.
        dw_cfg (dict): The Config for dw conv.
            Defaults to ``dict(type='DW', kernel_size=7)``.
        scale (float): Scaling parameter of gflayer outputs.
            Defaults to 1.0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        use_layer_scale (bool): Whether to use use_layer_scale in HorNet
             block. Defaults to True.
    """

    def __init__(self, dim, order=5, dw_cfg=dict(type='DW', kernel_size=7), scale=1.0, drop_path_rate=0.0, use_layer_scale=True):
        super().__init__()
        self.out_channels = dim
        self.norm1 = HorNetLayerNorm(dim, eps=1e-06, data_format='channels_first')
        self.gnconv = gnConv(dim, order, dw_cfg, scale)
        self.norm2 = HorNetLayerNorm(dim, eps=1e-06)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        if use_layer_scale:
            self.gamma1 = LayerScale(dim, data_format='channels_first')
            self.gamma2 = LayerScale(dim)
        else:
            self.gamma1, self.gamma2 = nn.Identity(), nn.Identity()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.gamma1(self.gnconv(self.norm1(x))))
        input = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma2(x)
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


class AttentionSubsample(nn.Sequential):

    def __init__(self, in_dim, out_dim, key_dim, num_heads=8, attn_ratio=2, act_cfg=dict(type='HSwish'), stride=2, resolution=14):
        super(AttentionSubsample, self).__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * self.num_heads
        self.attn_ratio = attn_ratio
        self.sub_resolution = (resolution - 1) // stride + 1
        h = self.dh + nh_kd
        self.kv = LinearBatchNorm(in_dim, h)
        self.q = nn.Sequential(Subsample(stride, resolution), LinearBatchNorm(in_dim, nh_kd))
        self.proj = nn.Sequential(build_activation_layer(act_cfg), LinearBatchNorm(self.dh, out_dim))
        self.stride = stride
        self.resolution = resolution
        points = list(itertools.product(range(resolution), range(resolution)))
        sub_points = list(itertools.product(range(self.sub_resolution), range(self.sub_resolution)))
        N = len(points)
        N_sub = len(sub_points)
        attention_offsets = {}
        idxs = []
        for p1 in sub_points:
            for p2 in points:
                size = 1
                offset = abs(p1[0] * stride - p2[0] + (size - 1) / 2), abs(p1[1] * stride - p2[1] + (size - 1) / 2)
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(N_sub, N))

    @torch.no_grad()
    def train(self, mode=True):
        super(AttentionSubsample, self).train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, N, C = x.shape
        k, v = self.kv(x).view(B, N, self.num_heads, -1).split([self.key_dim, self.d], dim=3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        q = self.q(x).view(B, self.sub_resolution ** 2, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
        attn = q @ k.transpose(-2, -1) * self.scale + (self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dh)
        x = self.proj(x)
        return x


class FFN(nn.Module):
    """"out_features = out_features or in_features

        hidden_features = hidden_features or in_features"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_cfg=dict(type='GELU'), drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Sequential(nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0), build_norm_layer(dict(type='BN'), hidden_features))
        self.act = build_activation_layer(act_cfg)
        self.fc2 = Sequential(nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0), build_norm_layer(dict(type='BN'), out_features))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


def window_partition(x: 'torch.Tensor', window_size: 'int') ->Tuple[torch.Tensor, Tuple[int, int]]:
    """Partition into non-overlapping windows with padding if needed.

    Borrowed from https://github.com/facebookresearch/segment-anything/

    Args:
        x (torch.Tensor): Input tokens with [B, H, W, C].
        window_size (int): Window size.

    Returns:
        Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]

        - ``windows``: Windows after partition with
        [B * num_windows, window_size, window_size, C].
        - ``(Hp, Wp)``: Padded height and width before partition
    """
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows: 'torch.Tensor', window_size: 'int', pad_hw: 'Tuple[int, int]', hw: 'Tuple[int, int]') ->torch.Tensor:
    """Window unpartition into original sequences and removing padding.

    Borrowed from https://github.com/facebookresearch/segment-anything/

    Args:
        x (torch.Tensor): Input tokens with
            [B * num_windows, window_size, window_size, C].
        window_size (int): Window size.
        pad_hw (tuple): Padded height and width (Hp, Wp).
        hw (tuple): Original height and width (H, W) before padding.

    Returns:
        torch.Tensor: Unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class MobileVitBlock(nn.Module):
    """MobileViT block.

    According to the paper, the MobileViT block has a local representation.
    a transformer-as-convolution layer which consists of a global
    representation with unfolding and folding, and a final fusion layer.

    Args:
        in_channels (int): Number of input image channels.
        transformer_dim (int): Number of transformer channels.
        ffn_dim (int): Number of ffn channels in transformer block.
        out_channels (int): Number of channels in output.
        conv_ksize (int): Conv kernel size in local representation
            and fusion. Defaults to 3.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Defaults to dict(type='BN').
        act_cfg (dict, optional): Config dict for activation layer.
            Defaults to dict(type='Swish').
        num_transformer_blocks (int): Number of transformer blocks in
            a MobileViT block. Defaults to 2.
        patch_size (int): Patch size for unfolding and folding.
             Defaults to 2.
        num_heads (int): Number of heads in global representation.
             Defaults to 4.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        no_fusion (bool): Whether to remove the fusion layer.
            Defaults to False.
        transformer_norm_cfg (dict, optional): Config dict for normalization
            layer in transformer. Defaults to dict(type='LN').
    """

    def __init__(self, in_channels: 'int', transformer_dim: 'int', ffn_dim: 'int', out_channels: 'int', conv_ksize: 'int'=3, conv_cfg: 'Optional[dict]'=None, norm_cfg: 'Optional[dict]'=dict(type='BN'), act_cfg: 'Optional[dict]'=dict(type='Swish'), num_transformer_blocks: 'int'=2, patch_size: 'int'=2, num_heads: 'int'=4, drop_rate: 'float'=0.0, attn_drop_rate: 'float'=0.0, drop_path_rate: 'float'=0.0, no_fusion: 'bool'=False, transformer_norm_cfg: 'Callable'=dict(type='LN')):
        super(MobileVitBlock, self).__init__()
        self.local_rep = nn.Sequential(ConvModule(in_channels=in_channels, out_channels=in_channels, kernel_size=conv_ksize, padding=int((conv_ksize - 1) / 2), conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg), ConvModule(in_channels=in_channels, out_channels=transformer_dim, kernel_size=1, bias=False, conv_cfg=conv_cfg, norm_cfg=None, act_cfg=None))
        global_rep = [TransformerEncoderLayer(embed_dims=transformer_dim, num_heads=num_heads, feedforward_channels=ffn_dim, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, qkv_bias=True, act_cfg=dict(type='Swish'), norm_cfg=transformer_norm_cfg) for _ in range(num_transformer_blocks)]
        global_rep.append(build_norm_layer(transformer_norm_cfg, transformer_dim)[1])
        self.global_rep = nn.Sequential(*global_rep)
        self.conv_proj = ConvModule(in_channels=transformer_dim, out_channels=out_channels, kernel_size=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        if no_fusion:
            self.conv_fusion = None
        else:
            self.conv_fusion = ConvModule(in_channels=in_channels + out_channels, out_channels=out_channels, kernel_size=conv_ksize, padding=int((conv_ksize - 1) / 2), conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.patch_size = patch_size, patch_size
        self.patch_area = self.patch_size[0] * self.patch_size[1]

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        shortcut = x
        x = self.local_rep(x)
        patch_h, patch_w = self.patch_size
        B, C, H, W = x.shape
        new_h, new_w = math.ceil(H / patch_h) * patch_h, math.ceil(W / patch_w) * patch_w
        num_patch_h, num_patch_w = new_h // patch_h, new_w // patch_w
        num_patches = num_patch_h * num_patch_w
        interpolate = False
        if new_h != H or new_w != W:
            x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            interpolate = True
        x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
        x = self.global_rep(x)
        x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        if interpolate:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        x = self.conv_proj(x)
        if self.conv_fusion is not None:
            x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        return x


class Pooling(nn.Module):
    """Pooling module.

    Args:
        pool_size (int): Pooling size. Defaults to 3.
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class ConvFFN(nn.Module):
    """ConvFFN implemented by using point-wise convs."""

    def __init__(self, in_channels, hidden_channels=None, out_channels=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='GELU')):
        super().__init__()
        out_features = out_channels or in_channels
        hidden_features = hidden_channels or in_channels
        self.ffn_fc1 = ConvModule(in_channels=in_channels, out_channels=hidden_features, kernel_size=1, stride=1, padding=0, norm_cfg=norm_cfg, act_cfg=None)
        self.ffn_fc2 = ConvModule(in_channels=hidden_features, out_channels=out_features, kernel_size=1, stride=1, padding=0, norm_cfg=norm_cfg, act_cfg=None)
        self.act = build_activation_layer(act_cfg)

    def forward(self, x):
        x = self.ffn_fc1(x)
        x = self.act(x)
        x = self.ffn_fc2(x)
        return x


class RSoftmax(nn.Module):
    """Radix Softmax module in ``SplitAttentionConv2d``.

    Args:
        radix (int): Radix of input.
        groups (int): Groups of input.
    """

    def __init__(self, radix, groups):
        super().__init__()
        self.radix = radix
        self.groups = groups

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.groups, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttentionConv2d(nn.Module):
    """Split-Attention Conv2d.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int | tuple[int]): Same as nn.Conv2d.
        stride (int | tuple[int]): Same as nn.Conv2d.
        padding (int | tuple[int]): Same as nn.Conv2d.
        dilation (int | tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        radix (int): Radix of SpltAtConv2d. Default: 2
        reduction_factor (int): Reduction factor of SplitAttentionConv2d.
            Default: 4.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
    """

    def __init__(self, in_channels, channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, radix=2, reduction_factor=4, conv_cfg=None, norm_cfg=dict(type='BN')):
        super(SplitAttentionConv2d, self).__init__()
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.groups = groups
        self.channels = channels
        self.conv = build_conv_layer(conv_cfg, in_channels, channels * radix, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups * radix, bias=False)
        self.norm0_name, norm0 = build_norm_layer(norm_cfg, channels * radix, postfix=0)
        self.add_module(self.norm0_name, norm0)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = build_conv_layer(None, channels, inter_channels, 1, groups=self.groups)
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, inter_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.fc2 = build_conv_layer(None, inter_channels, channels * radix, 1, groups=self.groups)
        self.rsoftmax = RSoftmax(radix, groups)

    @property
    def norm0(self):
        return getattr(self, self.norm0_name)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm0(x)
        x = self.relu(x)
        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splits = x.view(batch, self.radix, -1, *x.shape[2:])
            gap = splits.sum(dim=1)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.norm1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        if self.radix > 1:
            attens = atten.view(batch, self.radix, -1, *atten.shape[2:])
            out = torch.sum(attens * splits, dim=1)
        else:
            out = atten * x
        return out.contiguous()


eps = 1e-05


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([('-1', nn.AvgPool2d(stride)), ('0', nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)), ('1', nn.BatchNorm2d(planes * self.expansion))]))

    def forward(self, x: 'torch.Tensor'):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def get_expansion(block, expansion=None):
    """Get the expansion of a residual block.

    The block expansion will be obtained by the following order:

    1. If ``expansion`` is given, just return it.
    2. If ``block`` has the attribute ``expansion``, then return
       ``block.expansion``.
    3. Return the default value according the the block type:
       1 for ``BasicBlock`` and 4 for ``Bottleneck``.

    Args:
        block (class): The block class.
        expansion (int | None): The given expansion ratio.

    Returns:
        int: The expansion of the block.
    """
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, BasicBlock):
            expansion = 1
        elif issubclass(block, Bottleneck):
            expansion = 4
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError('expansion must be an integer or None')
    return expansion


class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): Residual block used to build ResLayer.
        num_blocks (int): Number of blocks.
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int, optional): The expansion for BasicBlock/Bottleneck.
            If not specified, it will firstly be obtained via
            ``block.expansion``. If the block has no attribute "expansion",
            the following default values will be used: 1 for BasicBlock and
            4 for Bottleneck. Default: None.
        stride (int): stride of the first block. Default: 1.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict, optional): dictionary to construct and config conv
            layer. Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        drop_path_rate (float or list): stochastic depth rate.
            Default: 0.
    """

    def __init__(self, block, num_blocks, in_channels, out_channels, expansion=None, stride=1, avg_down=False, conv_cfg=None, norm_cfg=dict(type='BN'), drop_path_rate=0.0, **kwargs):
        self.block = block
        self.expansion = get_expansion(block, expansion)
        if isinstance(drop_path_rate, float):
            drop_path_rate = [drop_path_rate] * num_blocks
        assert len(drop_path_rate) == num_blocks, 'Please check the length of drop_path_rate'
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False))
            downsample.extend([build_conv_layer(conv_cfg, in_channels, out_channels, kernel_size=1, stride=conv_stride, bias=False), build_norm_layer(norm_cfg, out_channels)[1]])
            downsample = nn.Sequential(*downsample)
        layers = []
        layers.append(block(in_channels=in_channels, out_channels=out_channels, expansion=self.expansion, stride=stride, downsample=downsample, conv_cfg=conv_cfg, norm_cfg=norm_cfg, drop_path_rate=drop_path_rate[0], **kwargs))
        in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(block(in_channels=in_channels, out_channels=out_channels, expansion=self.expansion, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, drop_path_rate=drop_path_rate[i], **kwargs))
        super(ResLayer, self).__init__(*layers)


class TwoStreamFusion(nn.Module):
    """A general constructor for neural modules fusing two equal sized tensors
    in forward.

    Args:
        mode (str): The mode of fusion. Options are 'add', 'max', 'min',
            'avg', 'concat'.
    """

    def __init__(self, mode: 'str'):
        super().__init__()
        self.mode = mode
        if mode == 'add':
            self.fuse_fn = lambda x: torch.stack(x).sum(dim=0)
        elif mode == 'max':
            self.fuse_fn = lambda x: torch.stack(x).max(dim=0).values
        elif mode == 'min':
            self.fuse_fn = lambda x: torch.stack(x).min(dim=0).values
        elif mode == 'avg':
            self.fuse_fn = lambda x: torch.stack(x).mean(dim=0)
        elif mode == 'concat':
            self.fuse_fn = lambda x: torch.cat(x, dim=-1)
        else:
            raise NotImplementedError

    def forward(self, x):
        x = torch.chunk(x, 2, dim=2)
        return self.fuse_fn(x)


class Affine(nn.Module):
    """Affine Transformation module.

    Args:
        in_features (int): Input dimension.
    """

    def __init__(self, in_features):
        super().__init__()
        self.affine = nn.Conv2d(in_features, in_features, kernel_size=1, stride=1, padding=0, groups=in_features, bias=True)

    def forward(self, x):
        return self.affine(x) - x


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block for TinyViT. Adapted from
    https://github.com/microsoft/Cream/blob/main/TinyViT/models/tiny_vit.py.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        expand_ratio (int): The expand ratio of the hidden channels.
        drop_rate (float): The drop rate of the block.
        act_cfg (dict): The activation config of the module.
            Default: dict(type='GELU').
    """

    def __init__(self, in_channels, out_channels, expand_ratio, drop_path, act_cfg=dict(type='GELU')):
        super().__init__()
        self.in_channels = in_channels
        hidden_channels = int(in_channels * expand_ratio)
        self.conv1 = ConvBN2d(in_channels, hidden_channels, kernel_size=1)
        self.act = build_activation_layer(act_cfg)
        self.conv2 = ConvBN2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, groups=hidden_channels)
        self.conv3 = ConvBN2d(hidden_channels, out_channels, kernel_size=1, bn_weight_init=0.0)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.drop_path(x)
        x += shortcut
        x = self.act(x)
        return x


class DenseDilated(nn.Module):
    """Find dilated neighbor from neighbor list.

    edge_index: (2, batch_size, num_points, k)
    """

    def __init__(self, k=9, dilation=1, use_stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.use_stochastic = use_stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.use_stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index


def xy_pairwise_distance(x, y):
    """Compute pairwise distance of a point cloud.

    Args:
        x: tensor (batch_size, num_points, num_dims)
        y: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        xy_inner = -2 * torch.matmul(x, y.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True)
        return x_square + xy_inner + y_square.transpose(2, 1)


def xy_dense_knn_matrix(x, y, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.

    Args:
        x: (batch_size, num_dims, num_points, 1)
        y: (batch_size, num_dims, num_points, 1)
        k: int
        relative_pos:Whether to use relative_pos
    Returns:
        nearest neighbors:
        (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = xy_pairwise_distance(x.detach(), y.detach())
        if relative_pos is not None:
            dist += relative_pos
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


class DenseDilatedKnnGraph(nn.Module):
    """Find the neighbors' indices based on dilated knn."""

    def __init__(self, k=9, dilation=1, use_stochastic=False, epsilon=0.0):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.use_stochastic = use_stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, use_stochastic, epsilon)

    def forward(self, x, y=None, relative_pos=None):
        if y is not None:
            x = F.normalize(x, p=2.0, dim=1)
            y = F.normalize(y, p=2.0, dim=1)
            edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation, relative_pos)
        else:
            x = F.normalize(x, p=2.0, dim=1)
            y = x.clone()
            edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation, relative_pos)
        return self._dilated(edge_index)


def batched_index_select(x, idx):
    """fetches neighbors features from a given neighbor idx

    Args:
        x (Tensor): input feature Tensor
                :math:
                `\\mathbf{X} \\in \\mathbb{R}^{B \\times C \\times N \\times 1}`.
        idx (Tensor): edge_idx
                :math:`\\mathbf{X} \\in \\mathbb{R}^{B \\times N \\times l}`.
    Returns:
        Tensor: output neighbors features
            :math:`\\mathbf{X} \\in \\mathbb{R}^{B \\times C \\times N \\times k}`.
    """
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_reduced
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)
    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices_reduced, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature


class MRConv2d(nn.Module):
    """Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751)
    for dense data type."""

    def __init__(self, in_channels, out_channels, act_cfg, norm_cfg=None, graph_conv_bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act_cfg, norm_cfg, graph_conv_bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        return self.nn(x)


class EdgeConv2d(nn.Module):
    """Edge convolution layer (with activation, batch normalization) for dense
    data type."""

    def __init__(self, in_channels, out_channels, act_cfg, norm_cfg=None, graph_conv_bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act_cfg, norm_cfg, graph_conv_bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GraphSAGE(nn.Module):
    """GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216)
    for dense data type."""

    def __init__(self, in_channels, out_channels, act_cfg, norm_cfg=None, graph_conv_bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act_cfg, norm_cfg, graph_conv_bias)
        self.nn2 = BasicConv([in_channels * 2, out_channels], act_cfg, norm_cfg, graph_conv_bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv2d(nn.Module):
    """GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for
    dense data type."""

    def __init__(self, in_channels, out_channels, act_cfg, norm_cfg=None, graph_conv_bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act_cfg, norm_cfg, graph_conv_bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)


class GraphConv2d(nn.Module):
    """Static graph convolution layer."""

    def __init__(self, in_channels, out_channels, graph_conv_type, act_cfg, norm_cfg=None, graph_conv_bias=True):
        super(GraphConv2d, self).__init__()
        if graph_conv_type == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act_cfg, norm_cfg, graph_conv_bias)
        elif graph_conv_type == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels, act_cfg, norm_cfg, graph_conv_bias)
        elif graph_conv_type == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act_cfg, norm_cfg, graph_conv_bias)
        elif graph_conv_type == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels, act_cfg, norm_cfg, graph_conv_bias)
        else:
            raise NotImplementedError('graph_conv_type:{} is not supported'.format(graph_conv_type))

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)


class DyGraphConv2d(GraphConv2d):
    """Dynamic graph convolution layer."""

    def __init__(self, in_channels, out_channels, k=9, dilation=1, graph_conv_type='mr', act_cfg=dict(type='GELU'), norm_cfg=None, graph_conv_bias=True, use_stochastic=False, epsilon=0.2, r=1):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, graph_conv_type, act_cfg, norm_cfg, graph_conv_bias)
        self.k = k
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(k, dilation, use_stochastic, epsilon)

    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        x = super(DyGraphConv2d, self).forward(x, edge_index, y)
        return x.reshape(B, -1, H, W).contiguous()


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_relative_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, grid_size*grid_size]
    """
    pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)
    relative_pos = 2 * np.matmul(pos_embed, pos_embed.transpose()) / pos_embed.shape[1]
    return relative_pos


class Grapher(nn.Module):
    """Grapher module with graph convolution and fc layers."""

    def __init__(self, in_channels, k=9, dilation=1, graph_conv_type='mr', act_cfg=dict(type='GELU'), norm_cfg=None, graph_conv_bias=True, use_stochastic=False, epsilon=0.2, r=1, n=196, drop_path=0.0, relative_pos=False):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0), build_norm_layer(dict(type='BN'), in_channels))
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, k, dilation, graph_conv_type, act_cfg, norm_cfg, graph_conv_bias, use_stochastic, epsilon, r)
        self.fc2 = Sequential(nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0), build_norm_layer(dict(type='BN'), in_channels))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels, int(n ** 0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(relative_pos_tensor, size=(n, n // (r * r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode='bicubic').squeeze(0)

    def forward(self, x):
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        shortcut = x
        x = self.fc1(x)
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class Pooler(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class NormProduct(nn.Linear):
    """An enhanced linear layer with k clustering centers to calculate product
    between normalized input and linear weight.

    Args:
        in_features (int): size of each input sample.
        out_features (int): size of each output sample
        k (int): The number of clustering centers. Defaults to 1.
        bias (bool): Whether there is bias. If set to ``False``, the
            layer will not learn an additive bias. Defaults to ``True``.
        feature_norm (bool): Whether to normalize the input feature.
            Defaults to ``True``.
        weight_norm (bool):Whether to normalize the weight.
            Defaults to ``True``.
    """

    def __init__(self, in_features: 'int', out_features: 'int', k=1, bias: 'bool'=False, feature_norm: 'bool'=True, weight_norm: 'bool'=True):
        super().__init__(in_features, out_features * k, bias=bias)
        self.weight_norm = weight_norm
        self.feature_norm = feature_norm
        self.out_features = out_features
        self.k = k

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        if self.feature_norm:
            input = F.normalize(input)
        if self.weight_norm:
            weight = F.normalize(self.weight)
        else:
            weight = self.weight
        cosine_all = F.linear(input, weight, self.bias)
        if self.k == 1:
            return cosine_all
        else:
            cosine_all = cosine_all.view(-1, self.out_features, self.k)
            cosine, _ = torch.max(cosine_all, dim=2)
            return cosine


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    elif reduction == 'mean':
        loss = loss.sum() / avg_factor
    elif reduction != 'none':
        raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def asymmetric_loss(pred, target, weight=None, gamma_pos=1.0, gamma_neg=4.0, clip=0.05, reduction='mean', avg_factor=None, use_sigmoid=True, eps=1e-08):
    """asymmetric loss.

    Please refer to the `paper <https://arxiv.org/abs/2009.14119>`__ for
    details.

    Args:
        pred (torch.Tensor): The prediction with shape (N, \\*).
        target (torch.Tensor): The ground truth label of the prediction with
            shape (N, \\*).
        weight (torch.Tensor, optional): Sample-wise loss weight with shape
            (N, ). Defaults to None.
        gamma_pos (float): positive focusing parameter. Defaults to 0.0.
        gamma_neg (float): Negative focusing parameter. We usually set
            gamma_neg > gamma_pos. Defaults to 4.0.
        clip (float, optional): Probability margin. Defaults to 0.05.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". If reduction is 'none' , loss
            is same shape as pred and label. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        use_sigmoid (bool): Whether the prediction uses sigmoid instead
            of softmax. Defaults to True.
        eps (float): The minimum value of the argument of logarithm. Defaults
            to 1e-8.

    Returns:
        torch.Tensor: Loss.
    """
    assert pred.shape == target.shape, 'pred and target should be in the same shape.'
    if use_sigmoid:
        pred_sigmoid = pred.sigmoid()
    else:
        pred_sigmoid = nn.functional.softmax(pred, dim=-1)
    target = target.type_as(pred)
    if clip and clip > 0:
        pt = (1 - pred_sigmoid + clip).clamp(max=1) * (1 - target) + pred_sigmoid * target
    else:
        pt = (1 - pred_sigmoid) * (1 - target) + pred_sigmoid * target
    asymmetric_weight = (1 - pt).pow(gamma_pos * target + gamma_neg * (1 - target))
    loss = -torch.log(pt.clamp(min=eps)) * asymmetric_weight
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def convert_to_one_hot(targets: 'torch.Tensor', classes) ->torch.Tensor:
    """This function converts target class indices to one-hot vectors, given
    the number of classes.

    Args:
        targets (Tensor): The ground truth label of the prediction
                with shape (N, 1)
        classes (int): the number of classes.

    Returns:
        Tensor: Processed loss values.
    """
    assert torch.max(targets).item() < classes, 'Class Index must be less than number of classes'
    one_hot_targets = F.one_hot(targets.long().squeeze(-1), num_classes=classes)
    return one_hot_targets


class AsymmetricLoss(nn.Module):
    """asymmetric loss.

    Args:
        gamma_pos (float): positive focusing parameter.
            Defaults to 0.0.
        gamma_neg (float): Negative focusing parameter. We
            usually set gamma_neg > gamma_pos. Defaults to 4.0.
        clip (float, optional): Probability margin. Defaults to 0.05.
        reduction (str): The method used to reduce the loss into
            a scalar.
        loss_weight (float): Weight of loss. Defaults to 1.0.
        use_sigmoid (bool): Whether the prediction uses sigmoid instead
            of softmax. Defaults to True.
        eps (float): The minimum value of the argument of logarithm. Defaults
            to 1e-8.
    """

    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, reduction='mean', loss_weight=1.0, use_sigmoid=True, eps=1e-08):
        super(AsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.use_sigmoid = use_sigmoid
        self.eps = eps

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """asymmetric loss.

        Args:
            pred (torch.Tensor): The prediction with shape (N, \\*).
            target (torch.Tensor): The ground truth label of the prediction
                with shape (N, \\*), N or (N,1).
            weight (torch.Tensor, optional): Sample-wise loss weight with shape
                (N, \\*). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss into a scalar. Options are "none", "mean" and "sum".
                Defaults to None.

        Returns:
            torch.Tensor: Loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if target.dim() == 1 or target.dim() == 2 and target.shape[1] == 1:
            target = convert_to_one_hot(target.view(-1, 1), pred.shape[-1])
        loss_cls = self.loss_weight * asymmetric_loss(pred, target, weight, gamma_pos=self.gamma_pos, gamma_neg=self.gamma_neg, clip=self.clip, reduction=reduction, avg_factor=avg_factor, use_sigmoid=self.use_sigmoid, eps=self.eps)
        return loss_cls


def binary_cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None, class_weight=None, pos_weight=None):
    """Calculate the binary CrossEntropy loss with logits.

    Args:
        pred (torch.Tensor): The prediction with shape (N, \\*).
        label (torch.Tensor): The gt label with shape (N, \\*).
        weight (torch.Tensor, optional): Element-wise weight of loss with shape
            (N, ). Defaults to None.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". If reduction is 'none' , loss
            is same shape as pred and label. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.
        pos_weight (torch.Tensor, optional): The positive weight for each
            class with shape (C), C is the number of classes. Default None.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert pred.dim() == label.dim()
    if class_weight is not None:
        N = pred.size()[0]
        class_weight = class_weight.repeat(N, 1)
    loss = F.binary_cross_entropy_with_logits(pred, label.float(), weight=class_weight, pos_weight=pos_weight, reduction='none')
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None, class_weight=None):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.

    Returns:
        torch.Tensor: The calculated loss
    """
    loss = F.cross_entropy(pred, label, weight=class_weight, reduction='none')
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


def soft_cross_entropy(pred, label, weight=None, reduction='mean', class_weight=None, avg_factor=None):
    """Calculate the Soft CrossEntropy loss. The label can be float.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction with shape (N, C).
            When using "mixup", the label can be float.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.

    Returns:
        torch.Tensor: The calculated loss
    """
    loss = -label * F.log_softmax(pred, dim=-1)
    if class_weight is not None:
        loss *= class_weight
    loss = loss.sum(dim=-1)
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


class CrossEntropyLoss(nn.Module):
    """Cross entropy loss.

    Args:
        use_sigmoid (bool): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_soft (bool): Whether to use the soft version of CrossEntropyLoss.
            Defaults to False.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
        class_weight (List[float], optional): The weight for each class with
            shape (C), C is the number of classes. Default None.
        pos_weight (List[float], optional): The positive weight for each
            class with shape (C), C is the number of classes. Only enabled in
            BCE loss when ``use_sigmoid`` is True. Default None.
    """

    def __init__(self, use_sigmoid=False, use_soft=False, reduction='mean', loss_weight=1.0, class_weight=None, pos_weight=None):
        super(CrossEntropyLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.use_soft = use_soft
        assert not (self.use_soft and self.use_sigmoid), 'use_sigmoid and use_soft could not be set simultaneously'
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.pos_weight = pos_weight
        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_soft:
            self.cls_criterion = soft_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        if self.pos_weight is not None and self.use_sigmoid:
            pos_weight = cls_score.new_tensor(self.pos_weight)
            kwargs.update({'pos_weight': pos_weight})
        else:
            pos_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(cls_score, label, weight, class_weight=class_weight, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss_cls


def sigmoid_focal_loss(pred, target, weight=None, gamma=2.0, alpha=0.25, reduction='mean', avg_factor=None):
    """Sigmoid focal loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, \\*).
        target (torch.Tensor): The ground truth label of the prediction with
            shape (N, \\*).
        weight (torch.Tensor, optional): Sample-wise loss weight with shape
            (N, ). Defaults to None.
        gamma (float): The gamma for calculating the modulating factor.
            Defaults to 2.0.
        alpha (float): A balanced form for Focal Loss. Defaults to 0.25.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". If reduction is 'none' ,
            loss is same shape as pred and label. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: Loss.
    """
    assert pred.shape == target.shape, 'pred and target should be in the same shape.'
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class FocalLoss(nn.Module):
    """Focal loss.

    Args:
        gamma (float): Focusing parameter in focal loss.
            Defaults to 2.0.
        alpha (float): The parameter in balanced form of focal
            loss. Defaults to 0.25.
        reduction (str): The method used to reduce the loss into
            a scalar. Options are "none" and "mean". Defaults to 'mean'.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """Sigmoid focal loss.

        Args:
            pred (torch.Tensor): The prediction with shape (N, \\*).
            target (torch.Tensor): The ground truth label of the prediction
                with shape (N, \\*), N or (N,1).
            weight (torch.Tensor, optional): Sample-wise loss weight with shape
                (N, \\*). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss into a scalar. Options are "none", "mean" and "sum".
                Defaults to None.

        Returns:
            torch.Tensor: Loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if target.dim() == 1 or target.dim() == 2 and target.shape[1] == 1:
            target = convert_to_one_hot(target.view(-1, 1), pred.shape[-1])
        loss_cls = self.loss_weight * sigmoid_focal_loss(pred, target, weight, gamma=self.gamma, alpha=self.alpha, reduction=reduction, avg_factor=avg_factor)
        return loss_cls


class LabelSmoothLoss(nn.Module):
    """Initializer for the label smoothed cross entropy loss.

    Refers to `Rethinking the Inception Architecture for Computer Vision
    <https://arxiv.org/abs/1512.00567>`_

    This decreases gap between output scores and encourages generalization.
    Labels provided to forward can be one-hot like vectors (NxC) or class
    indices (Nx1).
    And this accepts linear combination of one-hot like labels from mixup or
    cutmix except multi-label task.

    Args:
        label_smooth_val (float): The degree of label smoothing.
        num_classes (int, optional): Number of classes. Defaults to None.
        mode (str): Refers to notes, Options are 'original', 'classy_vision',
            'multi_label'. Defaults to 'original'.
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid of
            softmax. Defaults to None, which means to use sigmoid in
            "multi_label" mode and not use in other modes.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.

    Notes:
        - if the mode is **"original"**, this will use the same label smooth
          method as the original paper as:

          .. math::
              (1-\\epsilon)\\delta_{k, y} + \\frac{\\epsilon}{K}

          where :math:`\\epsilon` is the ``label_smooth_val``, :math:`K` is the
          ``num_classes`` and :math:`\\delta_{k, y}` is Dirac delta, which
          equals 1 for :math:`k=y` and 0 otherwise.

        - if the mode is **"classy_vision"**, this will use the same label
          smooth method as the facebookresearch/ClassyVision repo as:

          .. math::
              \\frac{\\delta_{k, y} + \\epsilon/K}{1+\\epsilon}

        - if the mode is **"multi_label"**, this will accept labels from
          multi-label task and smoothing them as:

          .. math::
              (1-2\\epsilon)\\delta_{k, y} + \\epsilon
    """

    def __init__(self, label_smooth_val, num_classes=None, use_sigmoid=None, mode='original', reduction='mean', loss_weight=1.0, class_weight=None, pos_weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        assert isinstance(label_smooth_val, float) and 0 <= label_smooth_val < 1, f'LabelSmoothLoss accepts a float label_smooth_val over [0, 1), but gets {label_smooth_val}'
        self.label_smooth_val = label_smooth_val
        accept_reduction = {'none', 'mean', 'sum'}
        assert reduction in accept_reduction, f'LabelSmoothLoss supports reduction {accept_reduction}, but gets {mode}.'
        self.reduction = reduction
        accept_mode = {'original', 'classy_vision', 'multi_label'}
        assert mode in accept_mode, f'LabelSmoothLoss supports mode {accept_mode}, but gets {mode}.'
        self.mode = mode
        self._eps = label_smooth_val
        if mode == 'classy_vision':
            self._eps = label_smooth_val / (1 + label_smooth_val)
        if mode == 'multi_label':
            if not use_sigmoid:
                MMLogger.get_current_instance().warning('For multi-label tasks, please set `use_sigmoid=True` to use binary cross entropy.')
            self.smooth_label = self.multilabel_smooth_label
            use_sigmoid = True if use_sigmoid is None else use_sigmoid
        else:
            self.smooth_label = self.original_smooth_label
            use_sigmoid = False if use_sigmoid is None else use_sigmoid
        self.ce = CrossEntropyLoss(use_sigmoid=use_sigmoid, use_soft=not use_sigmoid, reduction=reduction, class_weight=class_weight, pos_weight=pos_weight)

    def generate_one_hot_like_label(self, label):
        """This function takes one-hot or index label vectors and computes one-
        hot like label vectors (float)"""
        if label.dim() == 1 or label.dim() == 2 and label.shape[1] == 1:
            label = convert_to_one_hot(label.view(-1, 1), self.num_classes)
        return label.float()

    def original_smooth_label(self, one_hot_like_label):
        assert self.num_classes > 0
        smooth_label = one_hot_like_label * (1 - self._eps)
        smooth_label += self._eps / self.num_classes
        return smooth_label

    def multilabel_smooth_label(self, one_hot_like_label):
        assert self.num_classes > 0
        smooth_label = torch.full_like(one_hot_like_label, self._eps)
        smooth_label.masked_fill_(one_hot_like_label > 0, 1 - self._eps)
        return smooth_label

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        """Label smooth loss.

        Args:
            pred (torch.Tensor): The prediction with shape (N, \\*).
            label (torch.Tensor): The ground truth label of the prediction
                with shape (N, \\*).
            weight (torch.Tensor, optional): Sample-wise loss weight with shape
                (N, \\*). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss into a scalar. Options are "none", "mean" and "sum".
                Defaults to None.

        Returns:
            torch.Tensor: Loss.
        """
        if self.num_classes is not None:
            assert self.num_classes == cls_score.shape[1], f'num_classes should equal to cls_score.shape[1], but got num_classes: {self.num_classes} and cls_score.shape[1]: {cls_score.shape[1]}'
        else:
            self.num_classes = cls_score.shape[1]
        one_hot_like_label = self.generate_one_hot_like_label(label=label)
        assert one_hot_like_label.shape == cls_score.shape, f'LabelSmoothLoss requires output and target to be same shape, but got output.shape: {cls_score.shape} and target.shape: {one_hot_like_label.shape}'
        smoothed_label = self.smooth_label(one_hot_like_label)
        return self.loss_weight * self.ce.forward(cls_score, smoothed_label, weight=weight, avg_factor=avg_factor, reduction_override=reduction_override, **kwargs)


def seesaw_ce_loss(cls_score, labels, weight, cum_samples, num_classes, p, q, eps, reduction='mean', avg_factor=None):
    """Calculate the Seesaw CrossEntropy loss.

    Args:
        cls_score (torch.Tensor): The prediction with shape (N, C),
             C is the number of classes.
        labels (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor): Sample-wise loss weight.
        cum_samples (torch.Tensor): Cumulative samples for each category.
        num_classes (int): The number of classes.
        p (float): The ``p`` in the mitigation factor.
        q (float): The ``q`` in the compenstation factor.
        eps (float): The minimal value of divisor to smooth
             the computation of compensation factor
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert cls_score.size(-1) == num_classes
    assert len(cum_samples) == num_classes
    onehot_labels = F.one_hot(labels, num_classes)
    seesaw_weights = cls_score.new_ones(onehot_labels.size())
    if p > 0:
        sample_ratio_matrix = cum_samples[None, :].clamp(min=1) / cum_samples[:, None].clamp(min=1)
        index = (sample_ratio_matrix < 1.0).float()
        sample_weights = sample_ratio_matrix.pow(p) * index + (1 - index)
        mitigation_factor = sample_weights[labels.long(), :]
        seesaw_weights = seesaw_weights * mitigation_factor
    if q > 0:
        scores = F.softmax(cls_score.detach(), dim=1)
        self_scores = scores[torch.arange(0, len(scores)).long(), labels.long()]
        score_matrix = scores / self_scores[:, None].clamp(min=eps)
        index = (score_matrix > 1.0).float()
        compensation_factor = score_matrix.pow(q) * index + (1 - index)
        seesaw_weights = seesaw_weights * compensation_factor
    cls_score = cls_score + seesaw_weights.log() * (1 - onehot_labels)
    loss = F.cross_entropy(cls_score, labels, weight=None, reduction='none')
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


class SeesawLoss(nn.Module):
    """Implementation of seesaw loss.

    Refers to `Seesaw Loss for Long-Tailed Instance Segmentation (CVPR 2021)
    <https://arxiv.org/abs/2008.10032>`_

    Args:
        use_sigmoid (bool): Whether the prediction uses sigmoid of softmax.
             Only False is supported. Defaults to False.
        p (float): The ``p`` in the mitigation factor.
             Defaults to 0.8.
        q (float): The ``q`` in the compenstation factor.
             Defaults to 2.0.
        num_classes (int): The number of classes.
             Defaults to 1000 for the ImageNet dataset.
        eps (float): The minimal value of divisor to smooth
             the computation of compensation factor, default to 1e-2.
        reduction (str): The method that reduces the loss to a scalar.
             Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, use_sigmoid=False, p=0.8, q=2.0, num_classes=1000, eps=0.01, reduction='mean', loss_weight=1.0):
        super(SeesawLoss, self).__init__()
        assert not use_sigmoid, '`use_sigmoid` is not supported'
        self.use_sigmoid = False
        self.p = p
        self.q = q
        self.num_classes = num_classes
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.cls_criterion = seesaw_ce_loss
        self.register_buffer('cum_samples', torch.zeros(self.num_classes, dtype=torch.float))

    def forward(self, cls_score, labels, weight=None, avg_factor=None, reduction_override=None):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C).
            labels (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                 the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                 Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum'), f'The `reduction_override` should be one of (None, "none", "mean", "sum"), but get "{reduction_override}".'
        assert cls_score.size(0) == labels.view(-1).size(0), f'Expected `labels` shape [{cls_score.size(0)}], but got {list(labels.size())}'
        reduction = reduction_override if reduction_override else self.reduction
        assert cls_score.size(-1) == self.num_classes, f'The channel number of output ({cls_score.size(-1)}) does not match the `num_classes` of seesaw loss ({self.num_classes}).'
        unique_labels = labels.unique()
        for u_l in unique_labels:
            inds_ = labels == u_l.item()
            self.cum_samples[u_l] += inds_.sum()
        if weight is not None:
            weight = weight.float()
        else:
            weight = labels.new_ones(labels.size(), dtype=torch.float)
        loss_cls = self.loss_weight * self.cls_criterion(cls_score, labels, weight, self.cum_samples, self.num_classes, self.p, self.q, self.eps, reduction, avg_factor)
        return loss_cls


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        self.config = config

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:seq_length + past_key_values_length]
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


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


class BertSelfAttention(nn.Module):

    def __init__(self, config, is_cross_attention):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, 'embedding_size'):
            raise ValueError("""The hidden size (%d) is not a multiple of
                the number of attention heads (%d)""" % (config.hidden_size, config.num_attention_heads))
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
        if key_layer.shape[0] > query_layer.shape[0]:
            key_layer = key_layer[:query_layer.shape[0], :, :, :]
            attention_mask = attention_mask[:query_layer.shape[0], :, :]
            value_layer = value_layer[:query_layer.shape[0], :, :, :]
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
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
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


def gelu(x):
    """Original Implementation of the gelu activation function in Google Bert
    repo when initially created.

    For information: OpenAI GPT's gelu is slightly different (and gives
    slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """Implementation of the gelu activation function currently in Google Bert
    repo (identical to OpenAI GPT) https://arxiv.org/abs/1606.08415."""
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {'gelu': gelu, 'relu': torch.nn.functional.relu, 'swish': swish, 'gelu_new': gelu_new}


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
        self.layer_num = layer_num
        if self.config.add_cross_attention:
            self.crossattention = BertAttention(config, is_cross_attention=self.config.add_cross_attention)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False, mode=None):
        if mode == 'tagging':
            assert encoder_hidden_states is not None, """encoder_hidden_states must be given
                for cross-attention layers"""
            cross_attention_outputs = self.crossattention(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions=output_attentions)
            attention_output = cross_attention_outputs[0]
            outputs = cross_attention_outputs[1:-1]
            present_key_value = cross_attention_outputs[-1]
        else:
            self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
            self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions, past_key_value=self_attn_past_key_value)
            attention_output = self_attention_outputs[0]
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
            if mode == 'multimodal':
                assert encoder_hidden_states is not None, """encoder_hidden_states must be
                    given for cross-attention layers"""
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
        self.gradient_checkpointing = False

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, use_cache=None, output_attentions=False, output_hidden_states=False, return_dict=True, mode='multimodal'):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        next_decoder_cache = () if use_cache else None
        for i in range(self.config.num_hidden_layers):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warn("""`use_cache=True` is incompatible with
                        gradient checkpointing. Setting `use_cache=False`...""")
                    use_cache = False

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)
                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(layer_module), hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, mode=mode)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions, mode=mode)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += layer_outputs[-1],
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=next_decoder_cache, hidden_states=all_hidden_states, attentions=all_self_attentions, cross_attentions=all_cross_attentions)


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


class BaseEncoder(nn.Module):
    """Base class for primitive encoders, such as ViT, TimeSformer, etc."""

    def __init__(self):
        super().__init__()

    def forward_features(self, samples, **kwargs):
        raise NotImplementedError

    @property
    def device(self):
        return list(self.parameters())[0].device


class Linear(torch.nn.Linear):
    """Wrapper for linear function."""


class OPTLearnedPositionalEmbedding(nn.Embedding):
    """This module learns positional embeddings up to a fixed maximum size."""

    def __init__(self, num_embeddings: 'int', embedding_dim: 'int'):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask: 'torch.LongTensor', past_key_values_length: 'int'=0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1
        positions = positions[:, past_key_values_length:]
        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(self, embed_dim: 'int', num_heads: 'int', dropout: 'float'=0.0, is_decoder: 'bool'=False, bias: 'bool'=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).')
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: 'torch.Tensor', seq_len: 'int', bsz: 'int'):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: 'torch.Tensor', key_value_states: 'Optional[torch.Tensor]'=None, past_key_value: 'Optional[Tuple[torch.Tensor]]'=None, attention_mask: 'Optional[torch.Tensor]'=None, layer_head_mask: 'Optional[torch.Tensor]'=None, output_attentions: 'bool'=False) ->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel."""
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        if self.is_decoder:
            past_key_value = key_states, value_states
        proj_shape = bsz * self.num_heads, -1, self.head_dim
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(f'Attention weights should be of size {bsz * self.num_heads, tgt_len, src_len}, but is {attn_weights.size()}')
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f'Attention mask should be of size {bsz, 1, tgt_len, src_len}, but is {attention_mask.size()}')
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(f'Head mask for a single layer should be of size {self.num_heads,}, but is {layer_head_mask.size()}')
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {bsz, self.num_heads, tgt_len, self.head_dim}, but is {attn_output.size()}')
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped, past_key_value


class OPTDecoderLayer(nn.Module):

    def __init__(self, config: 'OPTConfig'):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(embed_dim=self.embed_dim, num_heads=config.num_attention_heads, dropout=config.attention_dropout, is_decoder=True)
        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]'=None, layer_head_mask: 'Optional[torch.Tensor]'=None, output_attentions: 'Optional[bool]'=False, use_cache: 'Optional[bool]'=False, past_key_value: 'Optional[Tuple[torch.Tensor]]'=None) ->Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(hidden_states=hidden_states, past_key_value=past_key_value, attention_mask=attention_mask, layer_head_mask=layer_head_mask, output_attentions=output_attentions)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = (residual + hidden_states).view(hidden_states_shape)
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        outputs = hidden_states,
        if output_attentions:
            outputs += self_attn_weights,
        if use_cache:
            outputs += present_key_value,
        return outputs


class AttentionPool2d(nn.Module):

    def __init__(self, spacial_dim: 'int', embed_dim: 'int', num_heads: 'int', output_dim: 'int'=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(query=x, key=x, value=x, embed_dim_to_check=x.shape[-1], num_heads=self.num_heads, q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight, in_proj_weight=None, in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]), bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0, out_proj_weight=self.c_proj.weight, out_proj_bias=self.c_proj.bias, use_separate_proj_weight=True, training=self.training, need_weights=False)
        return x[0]


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Forward function."""
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """A faster version of GELU."""

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Forward function."""
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Residual Attention Block (RAB).

    This module implements the same function as the MultiheadAttention,
    but with a different interface, which is mainly used
    in CLIP.

    Args:
        d_model (int): The feature dimension.
        n_head (int): The number of attention heads.
        attn_mask (torch.Tensor, optional): The attention mask.
            Defaults to None.
    """

    def __init__(self, d_model: 'int', n_head: 'int', attn_mask: 'Optional[torch.Tensor]'=None, return_attention: 'bool'=False) ->None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)), ('gelu', QuickGELU()), ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.return_attention = return_attention

    def attention(self, x: 'torch.Tensor') ->torch.Tensor:
        """Attention function."""
        self.attn_mask = self.attn_mask if self.attn_mask is not None else None
        if self.return_attention:
            return self.attn(x, x, x, need_weights=self.return_attention, attn_mask=self.attn_mask)
        else:
            return self.attn(x, x, x, need_weights=self.return_attention, attn_mask=self.attn_mask)[0]

    def forward(self, x: 'torch.Tensor') ->Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward function."""
        if self.return_attention:
            x_, attention = self.attention(self.ln_1(x))
            x = x + x_
            x = x + self.mlp(self.ln_2(x))
            return x, attention
        else:
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x


class CLIPTransformer(nn.Module):
    """Transformer.

    Both visual and text branches use this transformer.

    Args:
        width (int): The feature dimension.
        layers (int): The number of layers.
        heads (int): The number of attention heads.
        attn_mask (torch.Tensor, optional): The attention mask.
    """

    def __init__(self, width: 'int', layers: 'int', heads: 'int', attn_mask: 'Optional[torch.Tensor]'=None) ->None:
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList()
        for _ in range(layers - 1):
            self.resblocks.append(ResidualAttentionBlock(width, heads, attn_mask))
        self.resblocks.append(ResidualAttentionBlock(width, heads, attn_mask, return_attention=True))

    def forward(self, x: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward function."""
        z = []
        for idx, blk in enumerate(self.resblocks):
            if idx < self.layers - 1:
                x = blk(x)
                z.append(x.permute(1, 0, 2))
            else:
                x, attention = blk(x)
                z.append(x.permute(1, 0, 2))
        return x, attention, z


class PerceiverAttention(nn.Module):
    """Perceiver attetion layers.

    Args:
        dim (int): Input dimensions.
        dim_head (int): Number of dimension heads. Defaults to 64.
        heads (int): Number of heads. Defaults to 8.
    """

    def __init__(self, *, dim: int, dim_head: int=64, heads: int=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: 'torch.Tensor', latents: 'torch.Tensor'):
        """Forward function.

        Args:
            x (torch.Tensor): image features of shape (b, T, n1, D).
            latent (torch.Tensor): latent features of shape (b, T, n2, D).
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)
        h = self.heads
        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q = rearrange(q, 'b t n (h d) -> b h t n d', h=h)
        k = rearrange(k, 'b t n (h d) -> b h t n d', h=h)
        v = rearrange(v, 'b t n (h d) -> b h t n d', h=h)
        q = q * self.scale
        sim = einsum('... i d, ... j d  -> ... i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h=h)
        return self.to_out(out)


def FeedForward(dim, mult: 'int'=4):
    """Feedforward layers.

    Args:
        mult (int): Layer expansion muliplier. Defaults to 4.
    """
    inner_dim = int(dim * mult)
    return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, inner_dim, bias=False), nn.GELU(), nn.Linear(inner_dim, dim, bias=False))


class PerceiverResampler(nn.Module):
    """Perceiver resampler layers.

    Args:
        dim (int): Input dimensions.
        depth (int): Depth of resampler. Defaults to 6.
        dim_head (int): Number of dimension heads. Defaults to 64.
        heads (int): Number of heads. Defaults to 8.
        num_latents (int): Number of latents. Defaults to 64.
        max_num_media (int, optional): Max number of media.
            Defaults to None.
        max_num_frames (int, optional): Max number of frames.
            Defaults to None.
        ff_mult (int): Feed forward multiplier. Defaults to 4.
    """

    def __init__(self, *, dim: int, depth: int=6, dim_head: int=64, heads: int=8, num_latents: int=64, max_num_media: Optional[int]=None, max_num_frames: Optional[int]=None, ff_mult: int=4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.frame_embs = nn.Parameter(torch.randn(max_num_frames, dim)) if max_num_frames is not None else None
        self.media_time_embs = nn.Parameter(torch.randn(max_num_media, 1, dim)) if max_num_media is not None else None
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads), FeedForward(dim=dim, mult=ff_mult)]))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: 'torch.Tensor'):
        """Forward function for perceiver sampler.

        Args:
            x (torch.Tensor): image features of shape (b, T, F, v, D)

        Returns:
            torch.Tensor: shape (b, T, n, D) where n is self.num_latents
        """
        b, T, F, v = x.shape[:4]
        if self.frame_embs is not None:
            frame_embs = repeat(self.frame_embs[:F], 'F d -> b T F v d', b=b, T=T, v=v)
            x = x + frame_embs
        x = rearrange(x, 'b T F v d -> b T (F v) d')
        if self.media_time_embs is not None:
            x = x + self.media_time_embs[:T]
        latents = repeat(self.latents, 'n d -> b T n d', b=b, T=T)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm(latents)


class MaskedCrossAttention(nn.Module):
    """Masked cross attention layers.

    Args:
        dim (int): Input text feature dimensions.
        dim_visual (int): Input visual feature dimensions.
        dim_head (int): Number of dimension heads. Defaults to 64.
        heads (int): Number of heads. Defaults to 8.
        only_attend_immediate_media (bool): Whether attend immediate media.
            Defaults to True.
    """

    def __init__(self, *, dim: int, dim_visual: int, dim_head: int=64, heads: int=8, only_attend_immediate_media: bool=True):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.only_attend_immediate_media = only_attend_immediate_media

    def forward(self, x: 'torch.Tensor', media: 'torch.Tensor', media_locations: 'Optional[torch.Tensor]'=None, attend_previous: 'bool'=True):
        """Forward function for perceiver sampler.

        Args:
            x (torch.Tensor): text features of shape (B, T_txt, D_txt).
            media (torch.Tensor): image features of shape
                (B, T_img, n, D_img) where n is the dim of the latents.
            media_locations (torch.Tensor, optional): boolean mask identifying
                the media tokens in x of shape (B, T_txt). Defaults to None.
            attend_previous (bool): If false, ignores immediately preceding
                image and starts attending when following image.
                Defaults to True.
        """
        _, T_img, n = media.shape[:3]
        h = self.heads
        x = self.norm(x)
        q = self.to_q(x)
        media = rearrange(media, 'b t n d -> b (t n) d')
        k, v = self.to_kv(media).chunk(2, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)
        q = q * self.scale
        sim = einsum('... i d, ... j d -> ... i j', q, k)
        if media_locations is not None:
            text_time = media_locations.cumsum(dim=-1)
            media_time = torch.arange(T_img, device=x.device) + 1
            if not attend_previous:
                text_time[~media_locations] += 1
                text_time[text_time > repeat(torch.count_nonzero(media_locations, dim=1), 'b -> b i', i=text_time.shape[1])] = 0
            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge
            text_to_media_mask = mask_op(rearrange(text_time, 'b i -> b 1 i 1'), repeat(media_time, 'j -> 1 1 1 (j n)', n=n))
            sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        if media_locations is not None and self.only_attend_immediate_media:
            text_without_media_mask = text_time == 0
            text_without_media_mask = rearrange(text_without_media_mask, 'b i -> b 1 i 1')
            attn = attn.masked_fill(text_without_media_mask, 0.0)
        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class GatedCrossAttentionBlock(nn.Module):
    """Gated cross attention layers.

    Args:
        dim (int): Input text feature dimensions.
        dim_visual (int): Input visual feature dimensions.
        dim_head (int): Number of dimension heads. Defaults to 64.
        heads (int): Number of heads. Defaults to 8.
        ff_mult (int): Feed forward multiplier. Defaults to 4.
        only_attend_immediate_media (bool): Whether attend immediate media.
            Defaults to True.
    """

    def __init__(self, *, dim: int, dim_visual: int, dim_head: int=64, heads: int=8, ff_mult: int=4, only_attend_immediate_media: bool=True):
        super().__init__()
        self.attn = MaskedCrossAttention(dim=dim, dim_visual=dim_visual, dim_head=dim_head, heads=heads, only_attend_immediate_media=only_attend_immediate_media)
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))
        self.ff = FeedForward(dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x: 'torch.Tensor', media: 'torch.Tensor', media_locations: 'Optional[torch.Tensor]'=None, attend_previous: 'bool'=True):
        """Forward function for perceiver sampler.

        Args:
            x (torch.Tensor): text features of shape (B, T_txt, D_txt).
            media (torch.Tensor): image features of shape
                (B, T_img, n, D_img) where n is the dim of the latents.
            media_locations (torch.Tensor, optional): boolean mask identifying
                the media tokens in x of shape (B, T_txt). Defaults to None.
            attend_previous (bool): If false, ignores immediately preceding
                image and starts attending when following image.
                Defaults to True.
        """
        x = self.attn(x, media, media_locations=media_locations, attend_previous=attend_previous) * self.attn_gate.tanh() + x
        x = self.ff(x) * self.ff_gate.tanh() + x
        return x


class FlamingoLayer(nn.Module):
    """Faminogo layers.

    Args:
        gated_cross_attn_layer (nn.Module): Gated cross attention layer.
        decoder_layer (nn.Module): Decoder layer.
    """

    def __init__(self, gated_cross_attn_layer: 'nn.Module', decoder_layer: 'nn.Module'):
        super().__init__()
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer
        self.vis_x = None
        self.media_locations = None

    def is_conditioned(self) ->bool:
        """Check whether the layer is conditioned."""
        return self.vis_x is not None

    def condition_vis_x(self, vis_x):
        """Set condition vision features."""
        self.vis_x = vis_x

    def condition_media_locations(self, media_locations):
        """Set condition media locations."""
        self.media_locations = media_locations

    def condition_attend_previous(self, attend_previous):
        """Set attend previous."""
        self.attend_previous = attend_previous

    def forward(self, lang_x: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]'=None, **decoder_layer_kwargs):
        """Forward function.

        Args:
            lang_x (torch.Tensor): language inputs.
            attention_mask (torch.Tensor, optional): text attention mask.
                Defaults to None.
            **decoder_layer_kwargs: Other decoder layer keyword arguments.
        """
        if self.gated_cross_attn_layer is None:
            return self.decoder_layer(lang_x, attention_mask=attention_mask, **decoder_layer_kwargs)
        if self.vis_x is None:
            raise ValueError('vis_x must be conditioned before forward pass')
        if self.media_locations is None:
            raise ValueError('media_locations must be conditioned before forward pass')
        lang_x = self.gated_cross_attn_layer(lang_x, self.vis_x, media_locations=self.media_locations, attend_previous=self.attend_previous)
        lang_x = self.decoder_layer(lang_x, attention_mask=attention_mask, **decoder_layer_kwargs)
        return lang_x


def scaled_dot_product_attention_pyimpl(query, key, value, attn_mask=None, dropout_p=0.0, scale=None, is_causal=False):
    scale = scale or query.size(-1) ** 0.5
    if is_causal and attn_mask is not None:
        attn_mask = torch.ones(query.size(-2), key.size(-2), dtype=torch.bool).tril(diagonal=0)
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf'))
    attn_weight = query @ key.transpose(-2, -1) / scale
    if attn_mask is not None:
        attn_weight += attn_mask
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, True)
    return attn_weight @ value


class OFAEncoderLayer(nn.Module):
    """OFAEncoder layer block."""

    def __init__(self, embedding_dim, num_heads, dropout_rate=0.0, drop_path_rate=0.0, attn_drop=0.0, act_drop=0.0, scale_factor=2.0, mlp_ratio=4.0, scale_heads=True, normformer=True, pre_norm=True, act_cfg=dict(type='GELU')):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pre_norm = pre_norm
        self.attn = MultiheadAttention(embedding_dim=embedding_dim, num_heads=num_heads, attn_drop=attn_drop, scale_factor=scale_factor, scale_heads=scale_heads)
        mid_channels = int(embedding_dim * mlp_ratio)
        self.fc1 = nn.Linear(embedding_dim, mid_channels)
        self.fc2 = nn.Linear(mid_channels, embedding_dim)
        self.act = MODELS.build(act_cfg)
        self.act_drop = nn.Dropout(act_drop) if act_drop > 0.0 else nn.Identity()
        self.attn_ln = nn.LayerNorm(embedding_dim)
        self.ffn_ln = nn.LayerNorm(embedding_dim)
        self.normformer = normformer
        if self.normformer:
            self.attn_mid_ln = nn.LayerNorm(embedding_dim)
            self.ffn_mid_ln = nn.LayerNorm(mid_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x, attention_mask=None, attn_bias=None, output_attentions=False):
        """Forward the encoder layer.

        Args:
            x (torch.tensor): The input to the layer of shape ``(B, L, C)``.
            attention_mask (torch.Tensor, optional): The attention mask of size
                ``(B, 1, L, L)``, where padding elements are indicated by very
                large negative values. Defaults to None.
            attn_bias (torch.tensor, optional): The bias for positional
                information. Defaults to None.
            output_attentions (bool): Whether to return the attentions tensors
                of the attention layer.

        Returns:
            List[torch.tensor]: The first element is the encoded output of
            shape ``(B, L, C)``. And the second element is the output
            attentions if ``output_attentions=True``.
        """
        residual = x
        if self.pre_norm:
            x = self.attn_ln(x)
        x, attn_weights, _ = self.attn(query=x, attn_mask=attention_mask, attn_bias=attn_bias, output_attentions=output_attentions)
        if self.normformer:
            x = self.attn_mid_ln(x)
        x = self.dropout(x)
        x = residual + self.drop_path(x)
        if not self.pre_norm:
            x = self.attn_ln(x)
        residual = x
        if self.pre_norm:
            x = self.ffn_ln(x)
        x = self.act(self.fc1(x))
        x = self.act_drop(x)
        if self.normformer:
            x = self.ffn_mid_ln(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + self.drop_path(x)
        if not self.pre_norm:
            x = self.ffn_ln(x)
        if output_attentions:
            return [x, attn_weights]
        else:
            return [x]


class OFADecoderLayer(nn.Module):
    """OFADecoder layer block."""

    def __init__(self, embedding_dim, num_heads, dropout_rate=0.0, drop_path_rate=0.0, attn_drop=0.0, act_drop=0.0, scale_factor=2.0, mlp_ratio=4.0, encoder_embed_dim=None, scale_heads=True, normformer=True, pre_norm=True, act_cfg=dict(type='GELU')):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pre_norm = pre_norm
        self.self_attn = MultiheadAttention(embedding_dim=embedding_dim, num_heads=num_heads, attn_drop=attn_drop, scale_factor=scale_factor, scale_heads=scale_heads)
        self.cross_attn = MultiheadAttention(embedding_dim=embedding_dim, kdim=encoder_embed_dim, vdim=encoder_embed_dim, num_heads=num_heads, attn_drop=attn_drop, scale_factor=scale_factor, scale_heads=scale_heads)
        mid_channels = int(embedding_dim * mlp_ratio)
        self.fc1 = nn.Linear(embedding_dim, mid_channels)
        self.fc2 = nn.Linear(mid_channels, embedding_dim)
        self.act = MODELS.build(act_cfg)
        self.act_drop = nn.Dropout(act_drop) if act_drop > 0.0 else nn.Identity()
        self.self_attn_ln = nn.LayerNorm(embedding_dim)
        self.cross_attn_ln = nn.LayerNorm(embedding_dim)
        self.ffn_ln = nn.LayerNorm(embedding_dim)
        self.normformer = normformer
        if self.normformer:
            self.self_attn_mid_ln = nn.LayerNorm(embedding_dim)
            self.cross_attn_mid_ln = nn.LayerNorm(embedding_dim)
            self.ffn_mid_ln = nn.LayerNorm(mid_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x, attention_mask=None, encoder_hidden_states: 'Optional[torch.Tensor]'=None, encoder_attention_mask: 'Optional[torch.Tensor]'=None, past_key_value: 'Optional[List[torch.Tensor]]'=None, output_attentions: 'bool'=False, use_cache: 'bool'=False, self_attn_bias: 'Optional[torch.Tensor]'=None, cross_attn_bias: 'Optional[torch.Tensor]'=None):
        """Forward the decoder layer.

        Args:
            x (torch.tensor): The input to the layer of shape ``(B, L, C)``.
            attention_mask (torch.Tensor, optional): The attention mask of size
                ``(B, 1, L, L)``, where padding elements are indicated by very
                large negative values. Defaults to None.
            encoder_hidden_states (torch.Tensor, optional): The cross attention
                input to the layer of size ``(B, L, C)``. Defaults to None.
            encoder_attention_mask (torch.Tensor, optional): The cross
                attention mask where padding elements are indicated by very
                large negative values. Defaults to None.
            past_key_value (Tuple[torch.tensor], optional): The cached past key
                and value projection states. Defaults to none.
            output_attentions (bool): whether to return the attentions tensors
                of all attention layers. Defaults to False.
            use_cache (bool, optional): Whether to use cache.
                Defaults to False.
            self_attn_bias (torch.Tensor, optional): The self attention bias
                for positional information. Defaults to None.
            cross_attn_bias (torch.Tensor, optional): The cross attention bias
                for positional information. Defaults to None.

        Returns:
            List[torch.tensor]: The first element is the encoded output of
            shape ``(B, L, C)``. The following two elements can be the output
            self-attentions and cross-attentions if ``output_attentions=True``.
            The following one element can be the cached past key and value
            projection states.
        """
        residual = x
        if past_key_value is not None:
            self_past_key_value = past_key_value[:2]
            cross_past_key_value = past_key_value[2:]
        else:
            self_past_key_value, cross_past_key_value = None, None
        if self.pre_norm:
            x = self.self_attn_ln(x)
        x, self_attn_weights, present_key_value = self.self_attn(query=x, past_key_value=self_past_key_value, attn_mask=attention_mask, output_attentions=output_attentions, attn_bias=self_attn_bias)
        if self.normformer:
            x = self.self_attn_mid_ln(x)
        x = self.dropout(x)
        x = residual + self.drop_path(x)
        if not self.pre_norm:
            x = self.self_attn_ln(x)
        if encoder_hidden_states is not None:
            residual = x
            if self.pre_norm:
                x = self.cross_attn_ln(x)
            x, cross_attn_weights, cross_key_value = self.cross_attn.forward(query=x, key_value=encoder_hidden_states, attn_mask=encoder_attention_mask, past_key_value=cross_past_key_value, output_attentions=output_attentions, attn_bias=cross_attn_bias)
            if self.normformer:
                x = self.cross_attn_mid_ln(x)
            x = self.dropout(x)
            x = residual + self.drop_path(x)
            if not self.pre_norm:
                x = self.cross_attn_ln(x)
            present_key_value = present_key_value + cross_key_value
        residual = x
        if self.pre_norm:
            x = self.ffn_ln(x)
        x = self.act(self.fc1(x))
        x = self.act_drop(x)
        if self.normformer:
            x = self.ffn_mid_ln(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + self.drop_path(x)
        if not self.pre_norm:
            x = self.ffn_ln(x)
        outputs = [x]
        if output_attentions:
            outputs.extend([self_attn_weights, cross_attn_weights])
        if use_cache:
            outputs.append(present_key_value)
        return outputs


class BertEmbeddings_nopos(nn.Module):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        """self.LayerNorm is not snake-cased to stick with
        TensorFlow model variable name and be able to load"""
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class GlobalAveragePooling(nn.Module):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """

    def __init__(self, dim=2):
        super(GlobalAveragePooling, self).__init__()
        assert dim in [1, 2, 3], f'GlobalAveragePooling dim only support {1, 2, 3}, get {dim} instead.'
        if dim == 1:
            self.gap = nn.AdaptiveAvgPool1d(1)
        elif dim == 2:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple([out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.gap(inputs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs


def gem(x: 'Tensor', p: 'Parameter', eps: 'float'=1e-06, clamp=True) ->Tensor:
    if clamp:
        x = x.clamp(min=eps)
    return F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeneralizedMeanPooling(nn.Module):
    """Generalized Mean Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        p (float): Parameter value. Defaults to 3.
        eps (float): epsilon. Defaults to 1e-6.
        clamp (bool): Use clamp before pooling. Defaults to True
        p_trainable (bool): Toggle whether Parameter p is trainable or not.
            Defaults to True.
    """

    def __init__(self, p=3.0, eps=1e-06, clamp=True, p_trainable=True):
        assert p >= 1, "'p' must be a value greater than 1"
        super(GeneralizedMeanPooling, self).__init__()
        self.p = Parameter(torch.ones(1) * p, requires_grad=p_trainable)
        self.eps = eps
        self.clamp = clamp
        self.p_trainable = p_trainable

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([gem(x, p=self.p, eps=self.eps, clamp=self.clamp) for x in inputs])
            outs = tuple([out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = gem(inputs, p=self.p, eps=self.eps, clamp=self.clamp)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs


class PatchSplit(nn.Module):
    """The up-sample module used in neck (transformer pyramid network)

    Args:
        dim (int): the input dimension (channel number).
        fpn_dim (int): the fpn dimension (channel number).
        norm_cfg (dict): Config dict for normalization layer.
                Defaults to ``dict(type='LN')``.
    """

    def __init__(self, dim, fpn_dim, norm_cfg):
        super().__init__()
        _, self.norm = build_norm_layer(norm_cfg, dim)
        self.reduction = nn.Linear(dim, fpn_dim * 4, bias=False)
        self.fpn_dim = fpn_dim

    def forward(self, x):
        B, N, H, W, C = x.shape
        x = self.norm(x)
        x = self.reduction(x)
        x = x.reshape(B, N, H, W, 2, 2, self.fpn_dim).permute(0, 1, 2, 4, 3, 5, 6).reshape(B, N, 2 * H, 2 * W, self.fpn_dim)
        return x


class LoRALinear(nn.Module):
    """Implements LoRA in a linear layer.

    Args:
        original_layer (nn.Linear): The linear layer to be finetuned.
        alpha (int): The scale factor of LoRA. Defaults to 1.
        rank (int): The rank of LoRA. Defaults to 0.
        drop_rate (float): The drop out rate for LoRA. Defaults to 0.

    Note:
        The forward process of LoRA linear layer is:

        .. math::
            `y = W_0 x + BAx * (\\alpha / r)`

        Where :math:`x` is the input, :math:`y` is the output,
        :math:`W_0` is the parameter of the original layer,
        :math:`A` and :math:`B` are the low-rank decomposition matrixs,
        :math: `\\alpha` is the scale factor and :math: `r` is the rank.
    """

    def __init__(self, original_layer: 'nn.Linear', alpha: 'int'=1, rank: 'int'=0, drop_rate: 'float'=0.0):
        super(LoRALinear, self).__init__()
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        self.lora_dropout = nn.Dropout(drop_rate)
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        self.original_layer = original_layer

    def forward(self, x: 'torch.Tensor'):
        out = self.original_layer(x)
        lora_x = self.lora_dropout(x)
        lora_out = self.lora_up(self.lora_down(lora_x)) * self.scaling
        return out + lora_out


class Conv2d(nn.Module):
    """Rewrite Conv2d module according to DALL-E code."""

    def __init__(self, n_in: 'int', n_out: 'int', kw: 'int', use_float16: 'bool'=True, device: 'torch.device'=torch.device('cpu'), requires_grad: 'bool'=False) ->None:
        super().__init__()
        w = torch.empty((n_out, n_in, kw, kw), dtype=torch.float32, device=device, requires_grad=requires_grad)
        w.normal_(std=1 / math.sqrt(n_in * kw ** 2))
        b = torch.zeros((n_out,), dtype=torch.float32, device=device, requires_grad=requires_grad)
        self.kw = kw
        self.w, self.b = nn.Parameter(w), nn.Parameter(b)
        self.use_float16 = use_float16

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        if self.use_float16 and 'cuda' in self.w.device.type:
            if x.dtype != torch.float16:
                x = x.half()
            w, b = self.w.half(), self.b.half()
        else:
            if x.dtype != torch.float32:
                x = x.float()
            w, b = self.w, self.b
        return F.conv2d(x, w, b, padding=(self.kw - 1) // 2)


class EncoderBlock(nn.Module):
    """Rewrite EncoderBlock module according to DALL-E code."""

    def __init__(self, n_in: 'int', n_out: 'int', n_layers: 'int', device: 'torch.device'=None, requires_grad: 'bool'=False) ->None:
        super().__init__()
        self.n_hid = n_out // 4
        self.post_gain = 1 / n_layers ** 2
        make_conv = partial(Conv2d, device=device, requires_grad=requires_grad)
        self.id_path = make_conv(n_in, n_out, 1) if n_in != n_out else nn.Identity()
        self.res_path = nn.Sequential(OrderedDict([('relu_1', nn.ReLU()), ('conv_1', make_conv(n_in, self.n_hid, 3)), ('relu_2', nn.ReLU()), ('conv_2', make_conv(self.n_hid, self.n_hid, 3)), ('relu_3', nn.ReLU()), ('conv_3', make_conv(self.n_hid, self.n_hid, 3)), ('relu_4', nn.ReLU()), ('conv_4', make_conv(self.n_hid, n_out, 1))]))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self.id_path(x) + self.post_gain * self.res_path(x)


class Transformer(nn.Module):
    """Transformer.

    Both visual and text branches use this transformer.

    Args:
        width (int): The feature dimension.
        layers (int): The number of layers.
        heads (int): The number of attention heads.
        attn_mask (torch.Tensor, optional): The attention mask.
    """

    def __init__(self, width: 'int', layers: 'int', heads: 'int', attn_mask: 'Optional[torch.Tensor]'=None) ->None:
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList()
        for _ in range(layers - 1):
            self.resblocks.append(ResidualAttentionBlock(width, heads, attn_mask))
        self.resblocks.append(ResidualAttentionBlock(width, heads, attn_mask, return_attention=True))

    def forward(self, x: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward function."""
        z = []
        for idx, blk in enumerate(self.resblocks):
            if idx < self.layers - 1:
                x = blk(x)
                z.append(x.permute(1, 0, 2))
            else:
                x, attention = blk(x)
                z.append(x.permute(1, 0, 2))
        return x, attention, z


class VisionTransformer(nn.Module):
    """Vision Transformer for CLIP.

    Args:
        input_resolution (int): The image size.
        patch_size (int): The patch size.
        width (int): The feature dimension.
        layers (int): The number of layers.
        heads (int): The number of attention heads.
        out_dim (int): The output dimension.
        fineturn (bool): Whether to fineturn the model.
        average_target (bool): Whether to average the target.
    """

    def __init__(self, input_resolution: 'int', patch_size: 'int', width: 'int', layers: 'int', heads: 'int', output_dim: 'int', finetune=False, average_targets: 'int'=1) ->None:
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.finetune = finetune
        if finetune is False:
            self.ln_post = LayerNorm(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.average_targets = average_targets

    def forward(self, x: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        """Forward function."""
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x, attention, z = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        if self.proj is not None:
            x = x @ self.proj
        return x, attention


class CLIP(nn.Module):
    """CLIP.

    Args:
        embed_dim (int): The embedding dimension.
        image_resolution (int): The image size.
        vision_layers (int): The number of layers in the vision transformer.
        vision_width (int): The feature dimension in the vision transformer.
        vision_patch_size (int): The patch size in the vision transformer.
        context_length (int): The context length.
        vocab_size (int): The vocabulary size.
        transformer_width (int): The feature dimension in the text transformer.
        transformer_heads (int): The number of attention heads in the
            text transformer.
        transformer_layers (int): The number of layers in the text transformer.
        fineturn (bool): Whether to fineturn the model.
        average_target (bool): Whether to average the target.
    """

    def __init__(self, embed_dim: 'int', image_resolution: 'int', vision_layers: 'Union[Tuple[int, int, int, int], int]', vision_width: 'int', vision_patch_size: 'int', context_length: 'int', vocab_size: 'int', transformer_width: 'int', transformer_heads: 'int', transformer_layers: 'int', finetune: 'bool'=False, average_targets: 'int'=1) ->None:
        super().__init__()
        self.context_length = context_length
        vision_heads = vision_width // 64
        self.visual = VisionTransformer(input_resolution=image_resolution, patch_size=vision_patch_size, width=vision_width, layers=vision_layers, heads=vision_heads, output_dim=embed_dim, finetune=finetune, average_targets=average_targets)
        self.transformer = Transformer(width=transformer_width, layers=transformer_layers, heads=transformer_heads, attn_mask=self.build_attention_mask())
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self) ->None:
        """Initialize the parameters.

        The pretrained weight will override the initialized parameters by this
        function.
        """
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        proj_std = self.transformer.width ** -0.5 * (2 * self.transformer.layers) ** -0.5
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self) ->torch.Tensor:
        """Build the attention mask."""
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)
        return mask

    @property
    def dtype(self) ->torch.dtype:
        """Get the dtype."""
        return self.visual.conv1.weight.dtype

    def encode_image(self, image: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        """Encode the image.

        Get the feature and attention mask from the last layer of the visual
        branch of CLIP.

        Args:
            image (torch.Tensor): The image tensor with shape NCHW.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The feature and attention mask.
        """
        return self.visual(image.type(self.dtype))


def build_clip_model(state_dict: 'dict', finetune: 'bool'=False, average_targets: 'int'=1) ->nn.Module:
    """Build the CLIP model.

    Args:
        state_dict (dict): The pretrained state dict.
        finetune (bool): Whether to fineturn the model.
        average_targets (bool): Whether to average the target.

    Returns:
        nn.Module: The CLIP model.
    """
    vit = 'visual.proj' in state_dict
    if vit:
        vision_width = state_dict['visual.conv1.weight'].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith('visual.') and k.endswith('.attn.in_proj_weight')])
        vision_patch_size = state_dict['visual.conv1.weight'].shape[-1]
        grid_size = round((state_dict['visual.positional_embedding'].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    embed_dim = state_dict['text_projection'].shape[1]
    context_length = state_dict['positional_embedding'].shape[0]
    vocab_size = state_dict['token_embedding.weight'].shape[0]
    transformer_width = state_dict['ln_final.weight'].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split('.')[2] for k in state_dict if k.startswith('transformer.resblocks')))
    model = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size, context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, finetune, average_targets)
    for key in ['input_resolution', 'context_length', 'vocab_size']:
        if key in state_dict:
            del state_dict[key]
    msg = model.load_state_dict(state_dict, strict=False)
    MMLogger.get_current_instance().info(f'Load CLIP model: {msg}')
    return model.eval()


class CLIPGenerator(nn.Module):
    """Get the features and attention from the last layer of CLIP.

    This module is used to generate target features in masked image modeling.

    Args:
        tokenizer_path (str): The path of the checkpoint of CLIP.
    """

    def __init__(self, tokenizer_path: 'str') ->None:
        super().__init__()
        self.tokenizer_path = tokenizer_path
        self.tokenizer = build_clip_model(_load_checkpoint(self.tokenizer_path), False)

    @torch.no_grad()
    def forward(self, x: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        """Get the features and attention from the last layer of CLIP.

        Args:
            x (torch.Tensor): The input image, which is of shape (N, 3, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The features and attention from
            the last layer of CLIP, which are of shape (N, L, C) and (N, L, L),
            respectively.
        """
        assert self.tokenizer is not None, 'Please check whether the `self.tokenizer` is initialized correctly.'
        clip_features = self.tokenizer.encode_image(x)
        return clip_features


class GRN(nn.Module):
    """Global Response Normalization Module.

    Come from `ConvNeXt V2: Co-designing and Scaling ConvNets with Masked
    Autoencoders <http://arxiv.org/abs/2301.00808>`_

    Args:
        in_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-6.
    """

    def __init__(self, in_channels, eps=1e-06):
        super().__init__()
        self.in_channels = in_channels
        self.gamma = nn.Parameter(torch.zeros(in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))
        self.eps = eps

    def forward(self, x: 'torch.Tensor', data_format='channel_first'):
        """Forward method.

        Args:
            x (torch.Tensor): The input tensor.
            data_format (str): The format of the input tensor. If
                ``"channel_first"``, the shape of the input tensor should be
                (B, C, H, W). If ``"channel_last"``, the shape of the input
                tensor should be (B, H, W, C). Defaults to "channel_first".
        """
        if data_format == 'channel_last':
            gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
            nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
            x = self.gamma * (x * nx) + self.beta + x
        elif data_format == 'channel_first':
            gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
            x = self.gamma.view(1, -1, 1, 1) * (x * nx) + self.beta.view(1, -1, 1, 1) + x
        return x


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images.

    Args:
        num_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_channels: 'int', **kwargs) ->None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x, data_format='channel_first'):
        """Forward method.

        Args:
            x (torch.Tensor): The input tensor.
            data_format (str): The format of the input tensor. If
                ``"channel_first"``, the shape of the input tensor should be
                (B, C, H, W). If ``"channel_last"``, the shape of the input
                tensor should be (B, H, W, C). Defaults to "channel_first".
        """
        assert x.dim() == 4, f'LayerNorm2d only supports inputs with shape (N, C, H, W), but got tensor with shape {x.shape}'
        if data_format == 'channel_last':
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif data_format == 'channel_first':
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 3, 1, 2).contiguous()
        return x


class SparseHelper:
    """The helper to compute sparse operation with pytorch, such as sparse
    convlolution, sparse batch norm, etc."""
    _cur_active: 'torch.Tensor' = None

    @staticmethod
    def _get_active_map_or_index(H: 'int', returning_active_map: 'bool'=True) ->torch.Tensor:
        """Get current active map with (B, 1, f, f) shape or index format."""
        downsample_raito = H // SparseHelper._cur_active.shape[-1]
        active_ex = SparseHelper._cur_active.repeat_interleave(downsample_raito, 2).repeat_interleave(downsample_raito, 3)
        return active_ex if returning_active_map else active_ex.squeeze(1).nonzero(as_tuple=True)

    @staticmethod
    def sp_conv_forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Sparse convolution forward function."""
        x = super(type(self), self).forward(x)
        x *= SparseHelper._get_active_map_or_index(H=x.shape[2], returning_active_map=True)
        return x

    @staticmethod
    def sp_bn_forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Sparse batch norm forward function."""
        active_index = SparseHelper._get_active_map_or_index(H=x.shape[2], returning_active_map=False)
        x_permuted = x.permute(0, 2, 3, 1)
        x_flattened = x_permuted[active_index]
        x_flattened = super(type(self), self).forward(x_flattened)
        output = torch.zeros_like(x_permuted, dtype=x_flattened.dtype)
        output[active_index] = x_flattened
        output = output.permute(0, 3, 1, 2)
        return output


class SparseConv2d(nn.Conv2d):
    """hack: override the forward function.
    See `sp_conv_forward` above for more details
    """
    forward = SparseHelper.sp_conv_forward


class SparseMaxPooling(nn.MaxPool2d):
    """hack: override the forward function.
    See `sp_conv_forward` above for more details
    """
    forward = SparseHelper.sp_conv_forward


class SparseAvgPooling(nn.AvgPool2d):
    """hack: override the forward function.
    See `sp_conv_forward` above for more details
    """
    forward = SparseHelper.sp_conv_forward


class SparseBatchNorm2d(nn.BatchNorm1d):
    """hack: override the forward function.
    See `sp_bn_forward` above for more details
    """
    forward = SparseHelper.sp_bn_forward


class SparseSyncBatchNorm2d(nn.SyncBatchNorm):
    """hack: override the forward function.
    See `sp_bn_forward` above for more details
    """
    forward = SparseHelper.sp_bn_forward


class SparseLayerNorm2D(nn.LayerNorm):
    """Implementation of sparse LayerNorm on channels for 2d images."""

    def forward(self, x: 'torch.Tensor', data_format='channel_first') ->torch.Tensor:
        """Sparse layer norm forward function with 2D data.

        Args:
            x (torch.Tensor): The input tensor.
            data_format (str): The format of the input tensor. If
                ``"channel_first"``, the shape of the input tensor should be
                (B, C, H, W). If ``"channel_last"``, the shape of the input
                tensor should be (B, H, W, C). Defaults to "channel_first".
        """
        assert x.dim() == 4, f'LayerNorm2d only supports inputs with shape (N, C, H, W), but got tensor with shape {x.shape}'
        if data_format == 'channel_last':
            index = SparseHelper._get_active_map_or_index(H=x.shape[1], returning_active_map=False)
            x_flattened = x[index]
            x_flattened = super().forward(x_flattened)
            x = torch.zeros_like(x, dtype=x_flattened.dtype)
            x[index] = x_flattened
        elif data_format == 'channel_first':
            index = SparseHelper._get_active_map_or_index(H=x.shape[2], returning_active_map=False)
            x_permuted = x.permute(0, 2, 3, 1)
            x_flattened = x_permuted[index]
            x_flattened = super().forward(x_flattened)
            x = torch.zeros_like(x_permuted, dtype=x_flattened.dtype)
            x[index] = x_flattened
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            raise NotImplementedError
        return x


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN layer.

    Modified from https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/swiglu_ffn.py
    """

    def __init__(self, embed_dims: 'int', feedforward_channels: 'Optional[int]'=None, out_dims: 'Optional[int]'=None, layer_scale_init_value: 'float'=0.0, bias: 'bool'=True, dropout_layer: 'Optional[dict]'=None, norm_cfg: 'Optional[dict]'=None, add_identity: 'bool'=True) ->None:
        super().__init__()
        self.embed_dims = embed_dims
        self.out_dims = out_dims or embed_dims
        hidden_dims = feedforward_channels or embed_dims
        self.w12 = nn.Linear(self.embed_dims, 2 * hidden_dims, bias=bias)
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, hidden_dims)
        else:
            self.norm = nn.Identity()
        self.w3 = nn.Linear(hidden_dims, self.out_dims, bias=bias)
        if layer_scale_init_value > 0:
            self.gamma2 = LayerScale(dim=embed_dims, layer_scale_init_value=layer_scale_init_value)
        else:
            self.gamma2 = nn.Identity()
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

    def forward(self, x: 'torch.Tensor', identity: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        hidden = self.norm(hidden)
        out = self.w3(hidden)
        out = self.gamma2(out)
        out = self.dropout_layer(out)
        if self.out_dims != self.embed_dims or not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out


class SwiGLUFFNFused(SwiGLUFFN):
    """SwiGLU FFN layer with fusing.

    Modified from https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/swiglu_ffn.py
    """

    def __init__(self, embed_dims: 'int', feedforward_channels: 'Optional[int]'=None, out_dims: 'Optional[int]'=None, layer_scale_init_value: 'float'=0.0, bias: 'bool'=True) ->None:
        out_dims = out_dims or embed_dims
        feedforward_channels = feedforward_channels or embed_dims
        feedforward_channels = (int(feedforward_channels * 2 / 3) + 7) // 8 * 8
        super().__init__(embed_dims=embed_dims, feedforward_channels=feedforward_channels, out_dims=out_dims, layer_scale_init_value=layer_scale_init_value, bias=bias)


def sample_vectors(samples: 'torch.Tensor', num: 'int') ->torch.Tensor:
    """Sample vectors according to the given number."""
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)
    return samples[indices]


def kmeans(samples: 'torch.Tensor', num_clusters: 'int', num_iters: 'int'=10, use_cosine_sim: 'bool'=False) ->Tuple[torch.Tensor, torch.Tensor]:
    """Run k-means algorithm."""
    dim, dtype, _ = samples.shape[-1], samples.dtype, samples.device
    means = sample_vectors(samples, num_clusters)
    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') - rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim=-1)
        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)
        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]
        if use_cosine_sim:
            new_means = F.normalize(new_means, p=2, dim=-1)
        means = torch.where(zero_mask[..., None], means, new_means)
    return means, bins


class EmbeddingEMA(nn.Module):
    """The codebook of embedding vectors.

    Args:
        num_tokens (int): Number of embedding vectors in the codebook.
        codebook_dim (int) : The dimension of embedding vectors in the
            codebook.
        kmeans_init (bool): Whether to use k-means to initialize the
            VectorQuantizer. Defaults to True.
        codebook_init_path (str): The initialization checkpoint for codebook.
            Defaults to None.
    """

    def __init__(self, num_tokens: 'int', codebook_dim: 'int', kmeans_init: 'bool'=True, codebook_init_path: 'Optional[str]'=None):
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_dim = codebook_dim
        if codebook_init_path is None:
            if not kmeans_init:
                weight = torch.randn(num_tokens, codebook_dim)
                weight = F.normalize(weight, p=2, dim=-1)
            else:
                weight = torch.zeros(num_tokens, codebook_dim)
            self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        else:
            None
            codebook_ckpt_weight = torch.load(codebook_init_path, map_location='cpu')
            weight = codebook_ckpt_weight.clone()
            self.register_buffer('initted', torch.Tensor([True]))
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.update = True

    @torch.jit.ignore
    def init_embed_(self, data: 'torch.Tensor') ->None:
        """Initialize embedding vectors of codebook."""
        if self.initted:
            return
        None
        embed, _ = kmeans(data, self.num_tokens, 10, use_cosine_sim=True)
        self.weight.data.copy_(embed)
        self.initted.data.copy_(torch.Tensor([True]))

    def forward(self, embed_id: 'torch.Tensor') ->torch.Tensor:
        """Get embedding vectors."""
        return F.embedding(embed_id, self.weight)


def ema_inplace(moving_avg: 'torch.Tensor', new: 'torch.Tensor', decay: 'torch.Tensor') ->None:
    """Update moving average."""
    moving_avg.data.mul_(decay).add_(new, alpha=1 - decay)


def norm_ema_inplace(moving_avg: 'torch.Tensor', new: 'torch.Tensor', decay: 'torch.Tensor') ->None:
    """Update moving average with norm data."""
    moving_avg.data.mul_(decay).add_(new, alpha=1 - decay)
    moving_avg.data.copy_(F.normalize(moving_avg.data, p=2, dim=-1))


class NormEMAVectorQuantizer(nn.Module):
    """Normed EMA vector quantizer module.

    Args:
        num_embed (int): Number of embedding vectors in the codebook. Defaults
            to 8192.
        embed_dims (int) : The dimension of embedding vectors in the codebook.
            Defaults to 32.
        beta (float): The mutiplier for VectorQuantizer embedding loss.
            Defaults to 1.
        decay (float): The decay parameter of EMA. Defaults to 0.99.
        statistic_code_usage (bool): Whether to use cluster_size to record
            statistic. Defaults to True.
        kmeans_init (bool): Whether to use k-means to initialize the
            VectorQuantizer. Defaults to True.
        codebook_init_path (str): The initialization checkpoint for codebook.
            Defaults to None.
    """

    def __init__(self, num_embed: 'int', embed_dims: 'int', beta: 'float', decay: 'float'=0.99, statistic_code_usage: 'bool'=True, kmeans_init: 'bool'=True, codebook_init_path: 'Optional[str]'=None) ->None:
        super().__init__()
        self.codebook_dim = embed_dims
        self.num_tokens = num_embed
        self.beta = beta
        self.decay = decay
        self.embedding = EmbeddingEMA(num_tokens=self.num_tokens, codebook_dim=self.codebook_dim, kmeans_init=kmeans_init, codebook_init_path=codebook_init_path)
        self.statistic_code_usage = statistic_code_usage
        if statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(num_embed))

    def reset_cluster_size(self, device):
        if self.statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(self.num_tokens))
            self.cluster_size = self.cluster_size

    def forward(self, z):
        """Forward function."""
        z = rearrange(z, 'b c h w -> b h w c')
        z = F.normalize(z, p=2, dim=-1)
        z_flattened = z.reshape(-1, self.codebook_dim)
        self.embedding.init_embed_(z_flattened)
        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + self.embedding.weight.pow(2).sum(dim=1) - 2 * torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight)
        encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(encoding_indices).view(z.shape)
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)
        if not self.training:
            with torch.no_grad():
                cluster_size = encodings.sum(0)
                all_reduce(cluster_size)
                ema_inplace(self.cluster_size, cluster_size, self.decay)
        if self.training and self.embedding.update:
            bins = encodings.sum(0)
            all_reduce(bins)
            ema_inplace(self.cluster_size, bins, self.decay)
            zero_mask = bins == 0
            bins = bins.masked_fill(zero_mask, 1.0)
            embed_sum = z_flattened.t() @ encodings
            all_reduce(embed_sum)
            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
            embed_normalized = F.normalize(embed_normalized, p=2, dim=-1)
            embed_normalized = torch.where(zero_mask[..., None], self.embedding.weight, embed_normalized)
            norm_ema_inplace(self.embedding.weight, embed_normalized, self.decay)
        loss = self.beta * F.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'b h w c -> b c h w')
        return z_q, loss, encoding_indices


class AttentiveBlock(nn.Module):
    """Attentive Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: False.
        qk_scale (float, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop (float, optional): Dropout rate. Default: 0.0.
        attn_drop (float, optional): Attention dropout rate. Default: 0.0.
        drop_path (float, optional): Stochastic depth rate. Default: 0.0.
        norm_cfg (dict, optional): Normalization layer.
            Default: dict(type='LN')
        out_dim (int, optional): Dimension of output. Default: None.
    """

    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_cfg=dict(type='LN'), out_dim=None):
        super().__init__()
        norm_layer = norm_cfg['type']
        self.norm1_q = build_norm_layer(dim, norm_layer, eps=1e-06)
        self.norm1_k = build_norm_layer(dim, norm_layer, eps=1e-06)
        self.norm1_v = build_norm_layer(dim, norm_layer, eps=1e-06)
        self.cross_dcn = CrossMultiheadAttention(embed_dims=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        if out_dim and out_dim != dim:
            self.cross_dcn.proj = nn.Linear(dim, out_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x_q, x_kv, pos_q, pos_k):
        x_q = self.norm1_q(x_q + pos_q)
        x_k = self.norm1_k(x_kv + pos_k)
        x_v = self.norm1_v(x_kv)
        x = self.cross_dcn(x_q, k=x_k, v=x_v)
        return x


class AttentionPoolingBlock(AttentiveBlock):

    def forward(self, x):
        x_q = x.mean(1, keepdim=True)
        x_kv = x
        pos_q, pos_k = 0, 0
        x = super().forward(x_q, x_kv, pos_q, pos_k)
        x = x.squeeze(1)
        return x


class DownsampleLayer(nn.Module):
    """Downsample layer of InternImage.

    Args:
        channels (int): number of input channels
        norm_layer (str): normalization layer
    """

    def __init__(self, channels, norm_layer='LN'):
        super().__init__()
        self.conv = nn.Conv2d(channels, 2 * channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = build_norm_layer(2 * channels, norm_layer, 'channels_first', 'channels_last')

    def forward(self, x):
        x = self.conv(x.permute(0, 3, 1, 2))
        x = self.norm(x)
        return x


class InternImageLayer(nn.Module):
    """Basic layer of InternImage.

    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_cfg (dict): activation layer
        norm_cfg (dict): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(self, core_op, channels, groups, mlp_ratio=4.0, drop=0.0, drop_path=0.0, act_cfg=dict(type='GELU'), norm_cfg=dict(type='LN'), post_norm=False, layer_scale=None, offset_scale=1.0, with_cp=False, dw_kernel_size=None, res_post_norm=False, center_feature_scale=False, remove_center=False):
        super().__init__()
        self.channels = channels
        self.groups = groups
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp
        self.norm1 = build_norm_layer(channels, 'LN')
        self.post_norm = post_norm
        self.dcn = core_op(channels=channels, kernel_size=3, stride=1, pad=1, dilation=1, group=groups, offset_scale=offset_scale, act_layer=act_cfg['type'], norm_layer=norm_cfg['type'], dw_kernel_size=dw_kernel_size, center_feature_scale=center_feature_scale, remove_center=remove_center)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = build_norm_layer(channels, 'LN')
        self.mlp = FFN(embed_dims=channels, feedforward_channels=int(channels * mlp_ratio), act_cfg=act_cfg, ffn_drop=drop, add_identity=False)
        self.layer_scale = layer_scale is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(channels), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(channels), requires_grad=True)
        self.res_post_norm = res_post_norm
        if res_post_norm:
            self.res_post_norm1 = build_norm_layer(channels, 'LN')
            self.res_post_norm2 = build_norm_layer(channels, 'LN')

    def forward(self, x):

        def _inner_forward(x):
            if not self.layer_scale:
                if self.post_norm:
                    x = x + self.drop_path(self.norm1(self.dcn(x)))
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
                elif self.res_post_norm:
                    x = x + self.drop_path(self.res_post_norm1(self.dcn(self.norm1(x))))
                    x = x + self.drop_path(self.res_post_norm2(self.mlp(self.norm2(x))))
                else:
                    x = x + self.drop_path(self.dcn(self.norm1(x)))
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x
            if self.post_norm:
                x = x + self.drop_path(self.gamma1 * self.norm1(self.dcn(x)))
                x = x + self.drop_path(self.gamma2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma1 * self.dcn(self.norm1(x)))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class InternImageBlock(nn.Module):
    """Block of InternImage.

    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        depths (list): Depth of each block.
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_cfg (dict): activation layer
        norm_cfg (dict): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(self, core_op, channels, depth, groups, downsample=True, mlp_ratio=4.0, drop=0.0, drop_path=0.0, act_cfg=dict(type='GELU'), norm_cfg=dict(type='LN'), post_norm=False, offset_scale=1.0, layer_scale=None, with_cp=False, dw_kernel_size=None, post_norm_block_ids=None, res_post_norm=False, center_feature_scale=False, remove_center=False):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.post_norm = post_norm
        self.center_feature_scale = center_feature_scale
        self.blocks = nn.ModuleList([InternImageLayer(core_op=core_op, channels=channels, groups=groups, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, act_cfg=act_cfg, norm_cfg=norm_cfg, post_norm=post_norm, layer_scale=layer_scale, offset_scale=offset_scale, with_cp=with_cp, dw_kernel_size=dw_kernel_size, res_post_norm=res_post_norm, center_feature_scale=center_feature_scale, remove_center=remove_center) for i in range(depth)])
        if not self.post_norm or center_feature_scale:
            self.norm = build_norm_layer(channels, 'LN')
        self.post_norm_block_ids = post_norm_block_ids
        if post_norm_block_ids is not None:
            self.post_norms = nn.ModuleList([build_norm_layer(channels, 'LN', eps=1e-06) for _ in post_norm_block_ids])
        self.downsample = DownsampleLayer(channels=channels, norm_layer=norm_cfg['type']) if downsample else None

    def forward(self, x, return_wo_downsample=False):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.post_norm_block_ids is not None and i in self.post_norm_block_ids:
                index = self.post_norm_block_ids.index(i)
                x = self.post_norms[index](x)
        if not self.post_norm or self.center_feature_scale:
            x = self.norm(x)
        if return_wo_downsample:
            x_ = x
        if self.downsample is not None:
            x = self.downsample(x)
        if return_wo_downsample:
            return x, x_
        return x


class CenterFeatureScaleModule(nn.Module):

    def forward(self, query, center_feature_scale_proj_weight, center_feature_scale_proj_bias):
        center_feature_scale = F.linear(query, weight=center_feature_scale_proj_weight, bias=center_feature_scale_proj_bias).sigmoid()
        return center_feature_scale


def _is_power_of_2(n):
    if not isinstance(n, int) or n < 0:
        raise ValueError('invalid input for _is_power_of_2: {} (type: {})'.format(n, type(n)))
    return n & n - 1 == 0 and n != 0


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()
    raise NotImplementedError(f'build_act_layer does not support {act_layer}')


def _generate_dilation_grids(spatial_shapes, kernel_h, kernel_w, dilation_h, dilation_w, group, device):
    _, H_, W_, _ = spatial_shapes
    points_list = []
    x, y = torch.meshgrid(torch.linspace(-(dilation_w * (kernel_w - 1) // 2), -(dilation_w * (kernel_w - 1) // 2) + (kernel_w - 1) * dilation_w, kernel_w, dtype=torch.float32, device=device), torch.linspace(-(dilation_h * (kernel_h - 1) // 2), -(dilation_h * (kernel_h - 1) // 2) + (kernel_h - 1) * dilation_h, kernel_h, dtype=torch.float32, device=device))
    points_list.extend([x / W_, y / H_])
    grid = torch.stack(points_list, -1).reshape(-1, 1, 2).repeat(1, group, 1).permute(1, 0, 2)
    grid = grid.reshape(1, 1, 1, group * kernel_h * kernel_w, 2)
    return grid


def _get_reference_points(spatial_shapes, device, kernel_h, kernel_w, dilation_h, dilation_w, pad_h=0, pad_w=0, stride_h=1, stride_w=1):
    _, H_, W_, _ = spatial_shapes
    H_out = (H_ - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    W_out = (W_ - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    ref_y, ref_x = torch.meshgrid(torch.linspace(dilation_h * (kernel_h - 1) // 2 + 0.5, dilation_h * (kernel_h - 1) // 2 + 0.5 + (H_out - 1) * stride_h, H_out, dtype=torch.float32, device=device), torch.linspace(dilation_w * (kernel_w - 1) // 2 + 0.5, dilation_w * (kernel_w - 1) // 2 + 0.5 + (W_out - 1) * stride_w, W_out, dtype=torch.float32, device=device))
    ref_y = ref_y.reshape(-1)[None] / H_
    ref_x = ref_x.reshape(-1)[None] / W_
    ref = torch.stack((ref_x, ref_y), -1).reshape(1, H_out, W_out, 1, 2)
    return ref


def remove_center_sampling_locations(sampling_locations, kernel_w, kernel_h):
    idx = list(range(sampling_locations.shape[-2]))
    C = (kernel_w * kernel_h - 1) // 2
    idx = [i for i in idx if i != C and (i - C) % (C * 2 + 1) != 0]
    sampling_locations = sampling_locations[:, :, :, idx, :]
    return sampling_locations


def dcnv3_core_pytorch(input, offset, mask, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group, group_channels, offset_scale, remove_center):
    if remove_center and (kernel_h % 2 == 0 or kernel_w % 2 == 0 or kernel_w != kernel_h):
        raise ValueError('remove_center is only compatible with square odd kernel size.')
    input = F.pad(input, [0, 0, pad_h, pad_h, pad_w, pad_w])
    N_, H_in, W_in, _ = input.shape
    _, H_out, W_out, _ = offset.shape
    ref = _get_reference_points(input.shape, input.device, kernel_h, kernel_w, dilation_h, dilation_w, pad_h, pad_w, stride_h, stride_w)
    grid = _generate_dilation_grids(input.shape, kernel_h, kernel_w, dilation_h, dilation_w, group, input.device)
    spatial_norm = torch.tensor([W_in, H_in]).reshape(1, 1, 1, 2).repeat(1, 1, 1, group * (kernel_h * kernel_w - remove_center))
    sampling_locations = (ref + grid * offset_scale).repeat(N_, 1, 1, 1, 1)
    if remove_center:
        sampling_locations = remove_center_sampling_locations(sampling_locations, kernel_w=kernel_w, kernel_h=kernel_h)
    sampling_locations = sampling_locations.flatten(3, 4)
    sampling_locations = sampling_locations + offset * offset_scale / spatial_norm
    P_ = kernel_h * kernel_w - remove_center
    sampling_grids = 2 * sampling_locations - 1
    input_ = input.view(N_, H_in * W_in, group * group_channels).transpose(1, 2).reshape(N_ * group, group_channels, H_in, W_in)
    sampling_grid_ = sampling_grids.view(N_, H_out * W_out, group, P_, 2).transpose(1, 2).flatten(0, 1)
    sampling_input_ = F.grid_sample(input_, sampling_grid_, mode='bilinear', padding_mode='zeros', align_corners=False)
    mask = mask.view(N_, H_out * W_out, group, P_).transpose(1, 2).reshape(N_ * group, 1, H_out * W_out, P_)
    output = (sampling_input_ * mask).sum(-1).view(N_, group * group_channels, H_out * W_out)
    return output.transpose(1, 2).reshape(N_, H_out, W_out, -1).contiguous()


class DCNv3_pytorch(nn.Module):

    def __init__(self, channels=64, kernel_size=3, dw_kernel_size=None, stride=1, pad=1, dilation=1, group=4, offset_scale=1.0, act_layer='GELU', norm_layer='LN', center_feature_scale=False, remove_center=False):
        """DCNv3 Module.

        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        if not _is_power_of_2(_d_per_group):
            warnings.warn("You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 which is more efficient in our CUDA implementation.")
        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)
        self.dw_conv = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=dw_kernel_size, stride=1, padding=(dw_kernel_size - 1) // 2, groups=channels), build_norm_layer(channels, norm_layer, 'channels_first', 'channels_last'), build_act_layer(act_layer))
        self.offset = nn.Linear(channels, group * (kernel_size * kernel_size - remove_center) * 2)
        self.mask = nn.Linear(channels, group * (kernel_size * kernel_size - remove_center))
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self._reset_parameters()
        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.0)
        constant_(self.offset.bias.data, 0.0)
        constant_(self.mask.weight.data, 0.0)
        constant_(self.mask.bias.data, 0.0)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(self, input):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        N, H, W, _ = input.shape
        x = self.input_proj(input)
        x_proj = x
        x1 = input.permute(0, 3, 1, 2)
        x1 = self.dw_conv(x1)
        offset = self.offset(x1)
        mask = self.mask(x1).reshape(N, H, W, self.group, -1)
        mask = F.softmax(mask, -1).reshape(N, H, W, -1)
        x = dcnv3_core_pytorch(x, offset, mask, self.kernel_size, self.kernel_size, self.stride, self.stride, self.pad, self.pad, self.dilation, self.dilation, self.group, self.group_channels, self.offset_scale, self.remove_center)
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            center_feature_scale = center_feature_scale[..., None].repeat(1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        x = self.output_proj(x)
        return x

