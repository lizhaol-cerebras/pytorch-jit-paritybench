
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


import math


from typing import Iterable


from typing import Optional


import torch


import torch.nn.functional as F


import numpy as np


from torch.nn import functional as F


import logging


from functools import partial


from collections import OrderedDict


import torch.nn as nn


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import torch.multiprocessing


import torch.utils.data as data


import re


from torchvision import transforms


import random


import torchvision


from torchvision.transforms import functional as torchvision_F


import torch.utils.data


from torchvision.ops import roi_align


from scipy import ndimage


import torch.nn.init as init


import time


import torch.backends.cudnn as cudnn


import torchvision.utils


from torch.nn.parallel import DistributedDataParallel as NativeDDP


import torch.distributed as dist


from torch.utils.data import Sampler


import torchvision.transforms.functional as F


import warnings


from typing import Tuple


from typing import Dict


from typing import Union


from typing import Any


from typing import Callable


import torch.utils.checkpoint as cp


from torch.jit.annotations import List


from typing import List


from copy import deepcopy


from collections import defaultdict


from torch.hub import load_state_dict_from_url


from torch.hub import download_url_to_file


from torch.hub import urlparse


from torch.hub import HASH_REGEX


from torch import nn as nn


import torch.nn.parallel


import types


import functools


from torch import nn


from math import ceil


import copy


from typing import cast


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


from torch.optim import Optimizer


from torch import optim as optim


from torch import distributed as dist


from collections import deque


from torchvision import utils as vutils


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, base_criterion: 'torch.nn.Module', teacher_model: 'torch.nn.Module', distillation_type: 'str', alpha: 'float', tau: 'float'):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss
        if outputs_kd is None:
            raise ValueError('When knowledge distillation is enabled, the model is expected to return a Tuple[Tensor, Tensor] with the output of the class_token and the dist_token')
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)
        if self.distillation_type == 'soft':
            T = self.tau
            distillation_loss = F.kl_div(F.log_softmax(outputs_kd / T, dim=1), F.log_softmax(teacher_outputs / T, dim=1), reduction='sum', log_target=True) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))
        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


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


def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


class MultiResoPatchEmbed(nn.Module):
    """ 
    add patch number(num_patches_list) corresponding to multi input resolution
    """

    def __init__(self, img_size_list=[112, 224], patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size_list = [to_2tuple(k) for k in img_size_list]
        patch_size = to_2tuple(patch_size)
        num_patches_list = [(img_size[1] // patch_size[1] * (img_size[0] // patch_size[0])) for img_size in img_size_list]
        self.patch_size = patch_size
        self.num_patches_list = num_patches_list
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def get_same_padding(x: 'int', k: 'int', s: 'int', d: 'int'):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


def pad_same(x, k: 'List[int]', s: 'List[int]', d: 'List[int]'=(1, 1), value: 'float'=0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x


def conv2d_same(x, weight: 'torch.Tensor', bias: 'Optional[torch.Tensor]'=None, stride: 'Tuple[int, int]'=(1, 1), padding: 'Tuple[int, int]'=(0, 0), dilation: 'Tuple[int, int]'=(1, 1), groups: 'int'=1):
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


def get_condconv_initializer(initializer, num_experts, expert_shape):

    def condconv_initializer(weight):
        """CondConv initializer function."""
        num_params = np.prod(expert_shape)
        if len(weight.shape) != 2 or weight.shape[0] != num_experts or weight.shape[1] != num_params:
            raise ValueError('CondConv variables must have shape [num_experts, num_params]')
        for i in range(num_experts):
            initializer(weight[i].view(expert_shape))
    return condconv_initializer


def get_padding(kernel_size, stride, dilation=1):
    padding = (stride - 1 + dilation * (kernel_size - 1)) // 2
    return padding


def is_static_pad(kernel_size: 'int', stride: 'int'=1, dilation: 'int'=1, **_):
    return stride == 1 and dilation * (kernel_size - 1) % 2 == 0


def get_padding_value(padding, kernel_size, **kwargs) ->Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        padding = padding.lower()
        if padding == 'same':
            if is_static_pad(kernel_size, **kwargs):
                padding = get_padding(kernel_size, **kwargs)
            else:
                padding = 0
                dynamic = True
        elif padding == 'valid':
            padding = 0
        else:
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic


class CondConv2d(nn.Module):
    """ Conditionally Parameterized Convolution
    Inspired by: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/condconv/condconv_layers.py

    Grouped convolution hackery for parallel execution of the per-sample kernel filters inspired by this discussion:
    https://github.com/pytorch/pytorch/issues/17983
    """
    __constants__ = ['in_channels', 'out_channels', 'dynamic_padding']

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='', dilation=1, groups=1, bias=False, num_experts=4):
        super(CondConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        padding_val, is_padding_dynamic = get_padding_value(padding, kernel_size, stride=stride, dilation=dilation)
        self.dynamic_padding = is_padding_dynamic
        self.padding = to_2tuple(padding_val)
        self.dilation = to_2tuple(dilation)
        self.groups = groups
        self.num_experts = num_experts
        self.weight_shape = (self.out_channels, self.in_channels // self.groups) + self.kernel_size
        weight_num_param = 1
        for wd in self.weight_shape:
            weight_num_param *= wd
        self.weight = torch.nn.Parameter(torch.Tensor(self.num_experts, weight_num_param))
        if bias:
            self.bias_shape = self.out_channels,
            self.bias = torch.nn.Parameter(torch.Tensor(self.num_experts, self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init_weight = get_condconv_initializer(partial(nn.init.kaiming_uniform_, a=math.sqrt(5)), self.num_experts, self.weight_shape)
        init_weight(self.weight)
        if self.bias is not None:
            fan_in = np.prod(self.weight_shape[1:])
            bound = 1 / math.sqrt(fan_in)
            init_bias = get_condconv_initializer(partial(nn.init.uniform_, a=-bound, b=bound), self.num_experts, self.bias_shape)
            init_bias(self.bias)

    def forward(self, x, routing_weights):
        B, C, H, W = x.shape
        weight = torch.matmul(routing_weights, self.weight)
        new_weight_shape = (B * self.out_channels, self.in_channels // self.groups) + self.kernel_size
        weight = weight.view(new_weight_shape)
        bias = None
        if self.bias is not None:
            bias = torch.matmul(routing_weights, self.bias)
            bias = bias.view(B * self.out_channels)
        x = x.view(1, B * C, H, W)
        if self.dynamic_padding:
            out = conv2d_same(x, weight, bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups * B)
        else:
            out = F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups * B)
        out = out.permute([1, 0, 2, 3]).view(B, self.out_channels, out.shape[-2], out.shape[-1])
        return out


def _split_channels(num_chan, num_groups):
    split = [(num_chan // num_groups) for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)


class MixedConv2d(nn.ModuleDict):
    """ Mixed Grouped Convolution

    Based on MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='', dilation=1, depthwise=False, **kwargs):
        super(MixedConv2d, self).__init__()
        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)
        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            conv_groups = in_ch if depthwise else 1
            self.add_module(str(idx), create_conv2d_pad(in_ch, out_ch, k, stride=stride, padding=padding, dilation=dilation, groups=conv_groups, **kwargs))
        self.splits = in_splits

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        x = torch.cat(x_out, 1)
        return x


def create_conv2d(in_channels, out_channels, kernel_size, **kwargs):
    """ Select a 2d convolution implementation based on arguments
    Creates and returns one of torch.nn.Conv2d, Conv2dSame, MixedConv2d, or CondConv2d.

    Used extensively by EfficientNet, MobileNetv3 and related networks.
    """
    if isinstance(kernel_size, list):
        assert 'num_experts' not in kwargs
        assert 'groups' not in kwargs
        m = MixedConv2d(in_channels, out_channels, kernel_size, **kwargs)
    else:
        depthwise = kwargs.pop('depthwise', False)
        groups = in_channels if depthwise else kwargs.pop('groups', 1)
        if 'num_experts' in kwargs and kwargs['num_experts'] > 0:
            m = CondConv2d(in_channels, out_channels, kernel_size, groups=groups, **kwargs)
        else:
            m = create_conv2d_pad(in_channels, out_channels, kernel_size, groups=groups, **kwargs)
    return m


class SeparableConv2d(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, padding='', act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv_dw = create_conv2d(inplanes, inplanes, kernel_size, stride=stride, padding=padding, dilation=dilation, depthwise=True)
        self.bn_dw = norm_layer(inplanes)
        if act_layer is not None:
            self.act_dw = act_layer(inplace=True)
        else:
            self.act_dw = None
        self.conv_pw = create_conv2d(inplanes, planes, kernel_size=1)
        self.bn_pw = norm_layer(planes)
        if act_layer is not None:
            self.act_pw = act_layer(inplace=True)
        else:
            self.act_pw = None

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.bn_dw(x)
        if self.act_dw is not None:
            x = self.act_dw(x)
        x = self.conv_pw(x)
        x = self.bn_pw(x)
        if self.act_pw is not None:
            x = self.act_pw(x)
        return x


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()
        if out_channels != in_channels or strides != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None
        rep = []
        for i in range(reps):
            if grow_first:
                inc = in_channels if i == 0 else out_channels
                outc = out_channels
            else:
                inc = in_channels
                outc = in_channels if i < reps - 1 else out_channels
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(inc, outc, 3, stride=1, padding=1))
            rep.append(nn.BatchNorm2d(outc))
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B * N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B * N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError


def get_index(idx, image_size=224, patch_size1=32, patch_size2=16):
    """
    get index of fine stage corresponding to coarse stage 
    """
    H1 = int(image_size / patch_size1)
    H2 = int(image_size / patch_size2)
    y = idx % H1
    idx1 = 4 * idx - 2 * y
    idx2 = idx1 + 1
    idx3 = idx1 + H2
    idx4 = idx3 + 1
    idx_finnal = torch.cat((idx1, idx2, idx3, idx4), dim=1)
    return idx_finnal


def _no_grad_trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \\text{mean} \\leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class CFVisionTransformer(nn.Module):
    """ 
    Vision Transformer with support for patch or hybrid CNN input stage
    
    """

    def __init__(self, img_size_list=[112, 224], patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.informative_selection = False
        self.alpha = 0.5
        self.beta = 0.99
        self.target_index = [3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.patch_h = img_size_list[1] // patch_size
        self.patch_w = img_size_list[1] // patch_size
        self.img_size_list = img_size_list
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = MultiResoPatchEmbed(img_size_list=img_size_list, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches_list = self.patch_embed.num_patches_list
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_list = [nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) for num_patches in num_patches_list]
        self.pos_embed_list = nn.ParameterList(self.pos_embed_list)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.reuse_block = nn.Sequential(norm_layer(embed_dim), Mlp(in_features=embed_dim, hidden_features=mlp_ratio * embed_dim, out_features=embed_dim, act_layer=nn.GELU, drop=drop_rate))
        for pos_embed in self.pos_embed_list:
            trunc_normal_(pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
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
        return {'pos_embed_list', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def get_vis_data(self):
        """
            For Visualization
        """
        return self.important_index

    def forward(self, xx):
        results = []
        global_attention = 0
        x = xx[0]
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed_list[0]
        x = self.pos_drop(x)
        embedding_x1 = x
        for index, blk in enumerate(self.blocks):
            x, atten = blk(x)
            if index in self.target_index:
                global_attention = self.beta * global_attention + (1 - self.beta) * atten
        x = self.norm(x)
        self.first_stage_output = x
        results.append(self.head(x[:, 0]))
        x = xx[1]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        feature_temp = self.first_stage_output[:, 1:, :]
        feature_temp = self.reuse_block(feature_temp)
        B, new_HW, C = feature_temp.shape
        feature_temp = feature_temp.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        feature_temp = torch.nn.functional.interpolate(feature_temp, (self.patch_h, self.patch_w), mode='nearest')
        feature_temp = feature_temp.view(B, C, x.size(1) - 1).transpose(1, 2)
        feature_temp = torch.cat((torch.zeros(B, 1, self.embed_dim), feature_temp), dim=1)
        x = x + feature_temp
        embedding_x2 = x + self.pos_embed_list[1]
        if self.informative_selection:
            cls_attn = global_attention.mean(dim=1)[:, 0, 1:]
            import_token_num = math.ceil(self.alpha * self.patch_embed.num_patches_list[0])
            policy_index = torch.argsort(cls_attn, dim=1, descending=True)
            unimportan_index = policy_index[:, import_token_num:]
            important_index = policy_index[:, :import_token_num]
            unimportan_tokens = batch_index_select(embedding_x1, unimportan_index + 1)
            important_index = get_index(important_index, image_size=self.img_size_list[1])
            self.important_index = important_index
            cls_index = torch.zeros((B, 1)).long()
            important_index = torch.cat((cls_index, important_index + 1), dim=1)
            important_tokens = batch_index_select(embedding_x2, important_index)
            x = torch.cat((important_tokens, unimportan_tokens), dim=1)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x, _ = blk(x)
        x = self.norm(x)
        results.append(self.head(x[:, 0]))
        return results

    def forward_early_exit(self, xx, threshold):
        alpha = 0.99
        global_attention = 0
        target_index = [3, 4, 5, 6, 7, 8, 9, 10, 11]
        x = xx[0]
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed_list[0]
        x = self.pos_drop(x)
        embedding_x1 = x
        for index, blk in enumerate(self.blocks):
            x, atten = blk(x)
            if index in target_index:
                global_attention = alpha * global_attention + (1 - alpha) * atten
        x = self.norm(x)
        self.first_stage_output = x
        coarse_result = self.head(x[:, 0])
        logits_temp = F.softmax(coarse_result, 1)
        max_preds, _ = logits_temp.max(dim=1, keepdim=False)
        no_exit = max_preds < threshold
        x = xx[1][no_exit]
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        feature_temp = self.first_stage_output[:, 1:, :][no_exit]
        feature_temp = self.reuse_block(feature_temp)
        B, new_HW, C = feature_temp.shape
        feature_temp = feature_temp.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        feature_temp = torch.nn.functional.interpolate(feature_temp, (14, 14), mode='nearest')
        feature_temp = feature_temp.view(B, C, x.size(1) - 1).transpose(1, 2)
        feature_temp = torch.cat((torch.zeros(B, 1, self.embed_dim), feature_temp), dim=1)
        x = x + feature_temp
        embedding_x2 = x + self.pos_embed_list[1]
        cls_attn = global_attention[no_exit].mean(dim=1)[:, 0, 1:]
        import_token_num = math.ceil(self.alpha * self.patch_embed.num_patches_list[0])
        policy_index = torch.argsort(cls_attn, dim=1, descending=True)
        unimportan_index = policy_index[:, import_token_num:]
        important_index = policy_index[:, :import_token_num]
        unimportan_tokens = batch_index_select(embedding_x1[no_exit], unimportan_index + 1)
        important_index = get_index(important_index)
        cls_index = torch.zeros((B, 1)).long()
        important_index = torch.cat((cls_index, important_index + 1), dim=1)
        important_tokens = batch_index_select(embedding_x2, important_index)
        x = torch.cat((important_tokens, unimportan_tokens), dim=1)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x, _ = blk(x)
        x = self.norm(x)
        fine_result = self.head(x[:, 0])
        coarse_result[no_exit] = fine_result
        return coarse_result


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class TokenLabelSoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(TokenLabelSoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        N_rep = x.shape[0]
        N = target.shape[0]
        if not N == N_rep:
            target = target.repeat(N_rep // N, 1)
        if len(target.shape) == 3 and target.shape[-1] == 2:
            ground_truth = target[:, :, 0]
            target = target[:, :, 1]
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class TokenLabelCrossEntropy(nn.Module):
    """
    Token labeling loss.
    """

    def __init__(self, dense_weight=1.0, cls_weight=1.0, mixup_active=True, classes=1000, ground_truth=False):
        """
        Constructor Token labeling loss.
        """
        super(TokenLabelCrossEntropy, self).__init__()
        self.CE = SoftTargetCrossEntropy()
        self.dense_weight = dense_weight
        self.mixup_active = mixup_active
        self.classes = classes
        self.cls_weight = cls_weight
        self.ground_truth = ground_truth
        assert dense_weight + cls_weight > 0

    def forward(self, x, target):
        output, aux_output, bb = x
        bbx1, bby1, bbx2, bby2 = bb
        B, N, C = aux_output.shape
        if len(target.shape) == 2:
            target_cls = target
            target_aux = target.repeat(1, N).reshape(B * N, C)
        else:
            target_cls = target[:, :, 1]
            if self.ground_truth:
                ground_truth = target[:, :, 0]
                ratio = (0.9 - 0.4 * (ground_truth.max(-1)[1] == target_cls.max(-1)[1])).unsqueeze(-1)
                target_cls = target_cls * ratio + ground_truth * (1 - ratio)
            target_aux = target[:, :, 2:]
            target_aux = target_aux.transpose(1, 2).reshape(-1, C)
        lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / N
        if lam < 1:
            target_cls = lam * target_cls + (1 - lam) * target_cls.flip(0)
        aux_output = aux_output.reshape(-1, C)
        loss_cls = self.CE(output, target_cls)
        loss_aux = self.CE(aux_output, target_aux)
        return self.cls_weight * loss_cls + self.dense_weight * loss_aux


class GroupLinear(nn.Module):
    """
    Group Linear operator 
    """

    def __init__(self, in_planes, out_channels, groups=1, bias=True):
        super(GroupLinear, self).__init__()
        assert in_planes % groups == 0
        assert out_channels % groups == 0
        self.in_dim = in_planes
        self.out_dim = out_channels
        self.groups = groups
        self.bias = bias
        self.group_in_dim = int(self.in_dim / self.groups)
        self.group_out_dim = int(self.out_dim / self.groups)
        self.group_weight = nn.Parameter(torch.zeros(self.groups, self.group_in_dim, self.group_out_dim))
        self.group_bias = nn.Parameter(torch.zeros(self.out_dim))

    def forward(self, x):
        t, b, d = x.size()
        x = x.view(t, b, self.groups, int(d / self.groups))
        out = torch.einsum('tbgd,gdf->tbgf', (x, self.group_weight)).reshape(t, b, self.out_dim) + self.group_bias
        return out

    def extra_repr(self):
        s = '{in_dim}, {out_dim}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class GroupNorm(nn.Module):

    def __init__(self, num_groups, embed_dim, eps=1e-05, affine=True):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, embed_dim, eps, affine)

    def forward(self, x):
        B, T, C = x.shape
        x = x.view(B * T, C)
        x = self.gn(x)
        x = x.view(B, T, C)
        return x


class MultiResoPatchEmbed4_2(nn.Module):
    """ 
    Image to Patch Embedding with 4 layer convolution
    """

    def __init__(self, img_size_list=[112, 224], patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        new_patch_size = to_2tuple(patch_size // 2)
        img_size_list = [to_2tuple(k) for k in img_size_list]
        patch_size = to_2tuple(patch_size)
        num_patches_list = [(img_size[1] // patch_size[1] * (img_size[0] // patch_size[0])) for img_size in img_size_list]
        self.img_size_list = img_size_list
        self.patch_size = patch_size
        self.num_patches_list = num_patches_list
        self.embed_dim = embed_dim
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.proj = nn.Conv2d(64, embed_dim, kernel_size=new_patch_size, stride=new_patch_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.proj(x)
        return x

    def flops(self):
        img_size = self.img_size[0]
        block_flops = dict(conv1=img_size / 2 * img_size / 2 * 3 * 64 * 7 * 7, conv2=img_size / 2 * img_size / 2 * 64 * 64 * 3 * 3, conv3=img_size / 2 * img_size / 2 * 64 * 64 * 3 * 3, proj=img_size / 2 * img_size / 2 * 64 * self.embed_dim)
        return sum(block_flops.values())


def get_block(block_type, **kargs):
    if block_type == 'mha':
        return MHABlock(**kargs)
    elif block_type == 'ffn':
        return FFNBlock(**kargs)
    elif block_type == 'tr':
        return Block(**kargs)


def get_dpr(drop_path_rate, depth, drop_path_decay='linear'):
    if drop_path_decay == 'linear':
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
    elif drop_path_decay == 'fix':
        dpr = [drop_path_rate] * depth
    else:
        assert len(drop_path_rate) == depth
        dpr = drop_path_rate
    return dpr


def rand_bbox(img_shape, lam, margin=0.0, count=None):
    """ Standard CutMix bounding-box
    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.

    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh


class CF_LV_ViT(nn.Module):
    """ Vision Transformer with tricks
    Arguements:
        p_emb: different conv based position embedding (default: 4 layer conv)
        skip_lam: residual scalar for skip connection (default: 1.0)
        order: which order of layers will be used (default: None, will override depth if given)
        mix_token: use mix token augmentation for batch of tokens (default: False)
        return_dense: whether to return feature of all tokens with an additional aux_head (default: False)
    """

    def __init__(self, img_size_list=[112, 224], patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, drop_path_decay='linear', hybrid_backbone=None, norm_layer=nn.LayerNorm, p_emb='4_2', head_dim=None, skip_lam=1.0, order=None, mix_token=False, return_dense=False):
        super().__init__()
        self.informative_selection = False
        self.alpha = 0.5
        self.beta_2 = 0.99
        self.target_index = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.patch_h = img_size_list[1] // patch_size
        self.patch_w = img_size_list[1] // patch_size
        self.img_size_list = img_size_list
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.output_dim = embed_dim if num_classes == 0 else num_classes
        patch_embed_fn = MultiResoPatchEmbed4_2
        self.patch_embed = patch_embed_fn(img_size_list=img_size_list, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches_list = self.patch_embed.num_patches_list
        self.patches_h_list = [(img_size // patch_size) for img_size in img_size_list]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_list = [nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) for num_patches in self.num_patches_list]
        self.pos_embed_list = nn.ParameterList(self.pos_embed_list)
        self.pos_drop = nn.Dropout(p=drop_rate)
        if order is None:
            dpr = get_dpr(drop_path_rate, depth, drop_path_decay)
            self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, skip_lam=skip_lam) for i in range(depth)])
        else:
            dpr = get_dpr(drop_path_rate, len(order), drop_path_decay)
            self.blocks = nn.ModuleList([get_block(order[i], dim=embed_dim, num_heads=num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, skip_lam=skip_lam) for i in range(len(order))])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.return_dense = return_dense
        self.mix_token = mix_token
        if return_dense:
            self.aux_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if mix_token:
            self.beta = 1.0
            assert return_dense, 'always return all features when mixtoken is enabled'
        self.reuse_block = nn.Sequential(norm_layer(embed_dim), Mlp(in_features=embed_dim, hidden_features=int(mlp_ratio * embed_dim), out_features=embed_dim, act_layer=nn.GELU, drop=drop_rate))
        for pos_embed in self.pos_embed_list:
            trunc_normal_(pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, GroupLinear):
            trunc_normal_(m.group_weight, std=0.02)
            if isinstance(m, GroupLinear) and m.group_bias is not None:
                nn.init.constant_(m.group_bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed_list', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward(self, xx):
        results = []
        self.first_stage_output = None
        x = self.forward_embeddings(xx[0])
        if self.mix_token and self.training:
            lam = np.random.beta(self.beta, self.beta)
            bbx1_0, bby1_0, bbx2_0, bby2_0 = rand_bbox(x.size(), lam)
            bbx1_1, bby1_1, bbx2_1, bby2_1 = bbx1_0 * 2, bby1_0 * 2, bbx2_0 * 2, bby2_0 * 2
            temp_x = x.clone()
            temp_x[:, :, bbx1_0:bbx2_0, bby1_0:bby2_0] = x.flip(0)[:, :, bbx1_0:bbx2_0, bby1_0:bby2_0]
            x = temp_x
        else:
            bbx1_1, bby1_1, bbx2_1, bby2_1 = 0, 0, 0, 0
            bbx1_0, bby1_0, bbx2_0, bby2_0 = 0, 0, 0, 0
        x = x.flatten(2).transpose(1, 2)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed_list[0]
        embedding_coarse = x
        x = self.pos_drop(x)
        global_attention = 0
        for i, blk in enumerate(self.blocks):
            x, atten = blk(x)
            if i in self.target_index:
                global_attention = self.beta_2 * global_attention + (1 - self.beta_2) * atten
        x = self.norm(x)
        self.first_stage_output = x
        x_cls = self.head(x[:, 0])
        if self.return_dense:
            x_aux = self.aux_head(x[:, 1:])
            if not self.training:
                results.append((x_cls + 0.25 * x_aux.max(1)[0]) / 1.25)
            else:
                if self.mix_token and self.training:
                    x_aux = x_aux.reshape(x_aux.shape[0], self.patches_h_list[0], self.patches_h_list[0], x_aux.shape[-1])
                    temp_x = x_aux.clone()
                    temp_x[:, bbx1_0:bbx2_0, bby1_0:bby2_0, :] = x_aux.flip(0)[:, bbx1_0:bbx2_0, bby1_0:bby2_0, :]
                    x_aux = temp_x
                    x_aux = x_aux.reshape(x_aux.shape[0], self.num_patches_list[0], x_aux.shape[-1])
                results.append((x_cls, x_aux, (bbx1_0, bby1_0, bbx2_0, bby2_0)))
        else:
            results.append(x_cls)
        x = self.forward_embeddings(xx[1])
        if self.mix_token and self.training:
            temp_x = x.clone()
            temp_x[:, :, bbx1_1:bbx2_1, bby1_1:bby2_1] = x.flip(0)[:, :, bbx1_1:bbx2_1, bby1_1:bby2_1]
            x = temp_x
        x = x.flatten(2).transpose(1, 2)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        feature_temp = self.first_stage_output[:, 1:, :]
        feature_temp = self.reuse_block(feature_temp)
        B, new_HW, C = feature_temp.shape
        feature_temp = feature_temp.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        feature_temp = torch.nn.functional.interpolate(feature_temp, (self.patch_h, self.patch_w), mode='nearest').view(B, C, x.size(1) - 1).transpose(1, 2)
        feature_temp = torch.cat((torch.zeros(B, 1, self.embed_dim), feature_temp), dim=1)
        x = x + feature_temp
        x = x + self.pos_embed_list[1]
        embedding_fine = x
        x = self.pos_drop(x)
        if not self.informative_selection:
            for blk in self.blocks:
                x, _ = blk(x)
            x = self.norm(x)
            x_cls = self.head(x[:, 0])
            if self.return_dense:
                x_aux = self.aux_head(x[:, 1:])
                if not self.training:
                    results.append(x_cls + 0.25 * x_aux.max(1)[0])
                else:
                    if self.mix_token and self.training:
                        x_aux = x_aux.reshape(x_aux.shape[0], self.patches_h_list[1], self.patches_h_list[1], x_aux.shape[-1])
                        temp_x = x_aux.clone()
                        temp_x[:, bbx1_1:bbx2_1, bby1_1:bby2_1, :] = x_aux.flip(0)[:, bbx1_1:bbx2_1, bby1_1:bby2_1, :]
                        x_aux = temp_x
                        x_aux = x_aux.reshape(x_aux.shape[0], self.num_patches_list[1], x_aux.shape[-1])
                    results.append((x_cls, x_aux, (bbx1_1, bby1_1, bbx2_1, bby2_1)))
            else:
                results.append(x_cls)
        else:
            cls_attn = global_attention.mean(dim=1)[:, 0, 1:]
            import_token_num = math.ceil(self.alpha * self.patch_embed.num_patches_list[0])
            policy_index = torch.argsort(cls_attn, dim=1, descending=True)
            patch_unimportan_index = policy_index[:, import_token_num:]
            important_index = policy_index[:, :import_token_num]
            unimportan_tokens = batch_index_select(embedding_coarse, patch_unimportan_index + 1)
            patch_important_index = get_index(important_index, image_size=self.img_size_list[1])
            cls_index = torch.zeros((B, 1)).long()
            important_index = torch.cat((cls_index, patch_important_index + 1), dim=1)
            important_tokens = batch_index_select(embedding_fine, important_index)
            x = torch.cat((important_tokens, unimportan_tokens), dim=1)
            for blk in self.blocks:
                x, _ = blk(x)
            x = self.norm(x)
            x_cls = self.head(x[:, 0])
            if self.return_dense:
                x_aux = self.aux_head(x[:, 1:])
                if not self.training:
                    results.append(x_cls + 0.25 * x_aux.max(1)[0])
                else:
                    if self.mix_token and self.training:
                        x_temp = torch.zeros(B, self.num_patches_list[1], self.num_classes, dtype=x_aux.dtype)
                        idx = patch_important_index
                        B, N, C = x_temp.size()
                        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
                        idx = idx + offset
                        x_temp.view(B * N, C)[idx.reshape(-1)] = x_aux[:, :import_token_num * 4].reshape(-1, C)
                        idx = patch_unimportan_index
                        idx = get_index(idx, image_size=self.img_size_list[1])
                        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
                        idx = idx + offset
                        x_temp.view(B * N, C)[idx.reshape(-1)] = x_aux[:, import_token_num * 4:].repeat(1, 4, 1).reshape(-1, C)
                        x_aux = x_temp.reshape(B, self.patches_h_list[1], self.patches_h_list[1], C)
                        temp_x = x_aux.clone()
                        temp_x[:, bbx1_1:bbx2_1, bby1_1:bby2_1, :] = x_aux.flip(0)[:, bbx1_1:bbx2_1, bby1_1:bby2_1, :]
                        x_aux = temp_x
                        x_aux = x_aux.reshape(x_aux.shape[0], self.num_patches_list[1], x_aux.shape[-1])
                    results.append((x_cls, x_aux, (bbx1_1, bby1_1, bbx2_1, bby2_1)))
            else:
                results.append(x_cls)
        return results

    def forward_early_exit(self, xx, threshold):
        self.first_stage_output = None
        x = self.forward_embeddings(xx[0])
        x = x.flatten(2).transpose(1, 2)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed_list[0]
        embedding_coarse = x
        x = self.pos_drop(x)
        global_attention = 0
        for i, blk in enumerate(self.blocks):
            x, atten = blk(x)
            if i in self.target_index:
                global_attention = self.beta_2 * global_attention + (1 - self.beta_2) * atten
        x = self.norm(x)
        self.first_stage_output = x
        x_cls = self.head(x[:, 0])
        x_aux = self.aux_head(x[:, 1:])
        coarse_result = (x_cls + 0.25 * x_aux.max(1)[0]) / 1.25
        logits_temp = F.softmax(coarse_result, 1)
        max_preds, _ = logits_temp.max(dim=1, keepdim=False)
        no_exit = max_preds < threshold
        x = xx[1][no_exit]
        x = self.forward_embeddings(x)
        x = x.flatten(2).transpose(1, 2)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        feature_temp = self.first_stage_output[:, 1:, :][no_exit]
        feature_temp = self.reuse_block(feature_temp)
        B, new_HW, C = feature_temp.shape
        feature_temp = feature_temp.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        feature_temp = torch.nn.functional.interpolate(feature_temp, (self.patch_h, self.patch_w), mode='nearest').view(B, C, x.size(1) - 1).transpose(1, 2)
        feature_temp = torch.cat((torch.zeros(B, 1, self.embed_dim), feature_temp), dim=1)
        x = x + feature_temp
        x = x + self.pos_embed_list[1]
        embedding_fine = x
        x = self.pos_drop(x)
        cls_attn = global_attention[no_exit].mean(dim=1)[:, 0, 1:]
        import_token_num = math.ceil(self.alpha * self.patch_embed.num_patches_list[0])
        policy_index = torch.argsort(cls_attn, dim=1, descending=True)
        patch_unimportan_index = policy_index[:, import_token_num:]
        important_index = policy_index[:, :import_token_num]
        unimportan_tokens = batch_index_select(embedding_coarse[no_exit], patch_unimportan_index + 1)
        patch_important_index = get_index(important_index, image_size=self.img_size_list[1])
        cls_index = torch.zeros((B, 1)).long()
        important_index = torch.cat((cls_index, patch_important_index + 1), dim=1)
        important_tokens = batch_index_select(embedding_fine, important_index)
        x = torch.cat((important_tokens, unimportan_tokens), dim=1)
        for blk in self.blocks:
            x, _ = blk(x)
        x = self.norm(x)
        x_cls = self.head(x[:, 0])
        x_aux = self.aux_head(x[:, 1:])
        fine_result = x_cls + 0.25 * x_aux.max(1)[0]
        coarse_result[no_exit] = fine_result
        return coarse_result


class AsymmetricLossMultiLabel(nn.Module):

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-08, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossMultiLabel, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w
        return -loss.sum()


class AsymmetricLossSingleLabel(nn.Module):

    def __init__(self, gamma_pos=1, gamma_neg=4, eps: 'float'=0.1, reduction='mean'):
        super(AsymmetricLossSingleLabel, self).__init__()
        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target, reduction=None):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (1-hot vector)
        """
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg, self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w
        if self.eps > 0:
            self.targets_classes.mul_(1 - self.eps).add_(self.eps / num_classes)
        loss = -self.targets_classes.mul(log_preds)
        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class JsdCrossEntropy(nn.Module):
    """ Jensen-Shannon Divergence + Cross-Entropy Loss

    Based on impl here: https://github.com/google-research/augmix/blob/master/imagenet.py
    From paper: 'AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty -
    https://arxiv.org/abs/1912.02781

    Hacked together by / Copyright 2020 Ross Wightman
    """

    def __init__(self, num_splits=3, alpha=12, smoothing=0.1):
        super().__init__()
        self.num_splits = num_splits
        self.alpha = alpha
        if smoothing is not None and smoothing > 0:
            self.cross_entropy_loss = LabelSmoothingCrossEntropy(smoothing)
        else:
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def __call__(self, output, target):
        split_size = output.shape[0] // self.num_splits
        assert split_size * self.num_splits == output.shape[0]
        logits_split = torch.split(output, split_size)
        loss = self.cross_entropy_loss(logits_split[0], target[:split_size])
        probs = [F.softmax(logits, dim=1) for logits in logits_split]
        logp_mixture = torch.clamp(torch.stack(probs).mean(axis=0), 1e-07, 1).log()
        loss += self.alpha * sum([F.kl_div(logp_mixture, p_split, reduction='batchmean') for p_split in probs]) / len(probs)
        return loss


def avg_pool2d_same(x, kernel_size: 'List[int]', stride: 'List[int]', padding: 'List[int]'=(0, 0), ceil_mode: 'bool'=False, count_include_pad: 'bool'=True):
    x = pad_same(x, kernel_size, stride)
    return F.avg_pool2d(x, kernel_size, stride, (0, 0), ceil_mode, count_include_pad)


class AvgPool2dSame(nn.AvgPool2d):
    """ Tensorflow like 'SAME' wrapper for 2D average pooling
    """

    def __init__(self, kernel_size: 'int', stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        super(AvgPool2dSame, self).__init__(kernel_size, stride, (0, 0), ceil_mode, count_include_pad)

    def forward(self, x):
        return avg_pool2d_same(x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)


class DownsampleAvg(nn.Module):

    def __init__(self, in_chs, out_chs, stride=1, dilation=1, first_dilation=None, preact=True, conv_layer=None, norm_layer=None):
        """ AvgPool Downsampling as in 'D' ResNet variants. This is not in RegNet space but I might experiment."""
        super(DownsampleAvg, self).__init__()
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            self.pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = nn.Identity()
        self.conv = conv_layer(in_chs, out_chs, 1, stride=1)
        self.norm = nn.Identity() if preact else norm_layer(out_chs, apply_act=False)

    def forward(self, x):
        return self.norm(self.conv(self.pool(x)))


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class InplaceAbn(nn.Module):
    """Activated Batch Normalization

    This gathers a BatchNorm and an activation function in a single module

    Parameters
    ----------
    num_features : int
        Number of feature channels in the input and output.
    eps : float
        Small constant to prevent numerical issues.
    momentum : float
        Momentum factor applied to compute running statistics.
    affine : bool
        If `True` apply learned scale and shift transformation after normalization.
    act_layer : str or nn.Module type
        Name or type of the activation functions, one of: `leaky_relu`, `elu`
    act_param : float
        Negative slope for the `leaky_relu` activation.
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, apply_act=True, act_layer='leaky_relu', act_param=0.01, drop_block=None):
        super(InplaceAbn, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if apply_act:
            if isinstance(act_layer, str):
                assert act_layer in ('leaky_relu', 'elu', 'identity', '')
                self.act_name = act_layer if act_layer else 'identity'
            elif act_layer == nn.ELU:
                self.act_name = 'elu'
            elif act_layer == nn.LeakyReLU:
                self.act_name = 'leaky_relu'
            elif act_layer == nn.Identity:
                self.act_name = 'identity'
            else:
                assert False, f'Invalid act layer {act_layer.__name__} for IABN'
        else:
            self.act_name = 'identity'
        self.act_param = act_param
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.running_mean, 0)
        nn.init.constant_(self.running_var, 1)
        if self.affine:
            nn.init.constant_(self.weight, 1)
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        output = inplace_abn(x, self.weight, self.bias, self.running_mean, self.running_var, self.training, self.momentum, self.eps, self.act_name, self.act_param)
        if isinstance(output, tuple):
            output = output[0]
        return output


def conv2d_iabn(ni, nf, stride, kernel_size=3, groups=1, act_layer='leaky_relu', act_param=0.01):
    return nn.Sequential(nn.Conv2d(ni, nf, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=groups, bias=False), InplaceAbn(nf, act_layer=act_layer, act_param=act_param))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, aa_layer=None):
        super(BasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = conv2d_iabn(inplanes, planes, stride=1, act_param=0.001)
        elif aa_layer is None:
            self.conv1 = conv2d_iabn(inplanes, planes, stride=2, act_param=0.001)
        else:
            self.conv1 = nn.Sequential(conv2d_iabn(inplanes, planes, stride=1, act_param=0.001), aa_layer(channels=planes, filt_size=3, stride=2))
        self.conv2 = conv2d_iabn(planes, planes, stride=1, act_layer='identity')
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        reduction_chs = max(planes * self.expansion // 4, 64)
        self.se = SEModule(planes * self.expansion, reduction_channels=reduction_chs) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)
        out += residual
        out = self.relu(out)
        return out


def swish(x, inplace: 'bool'=False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


class Swish(nn.Module):

    def __init__(self, inplace: 'bool'=False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)


class GELU(nn.Module):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg)
    """

    def __init__(self, inplace: 'bool'=False):
        super(GELU, self).__init__()

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return F.gelu(input)


def hard_mish(x, inplace: 'bool'=False):
    """ Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    if inplace:
        return x.mul_(0.5 * (x + 2).clamp(min=0, max=2))
    else:
        return 0.5 * x * (x + 2).clamp(min=0, max=2)


class HardMish(nn.Module):

    def __init__(self, inplace: 'bool'=False):
        super(HardMish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_mish(x, self.inplace)


def hard_sigmoid(x, inplace: 'bool'=False):
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(x + 3.0) / 6.0


class HardSigmoid(nn.Module):

    def __init__(self, inplace: 'bool'=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_sigmoid(x, self.inplace)


def hard_swish(x, inplace: 'bool'=False):
    inner = F.relu6(x + 3.0).div_(6.0)
    return x.mul_(inner) if inplace else x.mul(inner)


class HardSwish(nn.Module):

    def __init__(self, inplace: 'bool'=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, self.inplace)


def mish(x, inplace: 'bool'=False):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    NOTE: I don't have a working inplace variant
    """
    return x.mul(F.softplus(x).tanh())


class Mish(nn.Module):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """

    def __init__(self, inplace: 'bool'=False):
        super(Mish, self).__init__()

    def forward(self, x):
        return mish(x)


class PReLU(nn.PReLU):
    """Applies PReLU (w/ dummy inplace arg)
    """

    def __init__(self, num_parameters: 'int'=1, init: 'float'=0.25, inplace: 'bool'=False) ->None:
        super(PReLU, self).__init__(num_parameters=num_parameters, init=init)

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return F.prelu(input, self.weight)


class Sigmoid(nn.Module):

    def __init__(self, inplace: 'bool'=False):
        super(Sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.sigmoid_() if self.inplace else x.sigmoid()


class Tanh(nn.Module):

    def __init__(self, inplace: 'bool'=False):
        super(Tanh, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.tanh_() if self.inplace else x.tanh()


_has_silu = 'silu' in dir(torch.nn.functional)


_ACT_LAYER_DEFAULT = dict(silu=nn.SiLU if _has_silu else Swish, swish=nn.SiLU if _has_silu else Swish, mish=Mish, relu=nn.ReLU, relu6=nn.ReLU6, leaky_relu=nn.LeakyReLU, elu=nn.ELU, prelu=PReLU, celu=nn.CELU, selu=nn.SELU, gelu=GELU, sigmoid=Sigmoid, tanh=Tanh, hard_sigmoid=HardSigmoid, hard_swish=HardSwish, hard_mish=HardMish)


@torch.jit.script
def hard_mish_jit(x, inplace: 'bool'=False):
    """ Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    return 0.5 * x * (x + 2).clamp(min=0, max=2)


class HardMishJit(nn.Module):

    def __init__(self, inplace: 'bool'=False):
        super(HardMishJit, self).__init__()

    def forward(self, x):
        return hard_mish_jit(x)


@torch.jit.script
def hard_sigmoid_jit(x, inplace: 'bool'=False):
    return (x + 3).clamp(min=0, max=6).div(6.0)


class HardSigmoidJit(nn.Module):

    def __init__(self, inplace: 'bool'=False):
        super(HardSigmoidJit, self).__init__()

    def forward(self, x):
        return hard_sigmoid_jit(x)


@torch.jit.script
def hard_swish_jit(x, inplace: 'bool'=False):
    return x * (x + 3).clamp(min=0, max=6).div(6.0)


class HardSwishJit(nn.Module):

    def __init__(self, inplace: 'bool'=False):
        super(HardSwishJit, self).__init__()

    def forward(self, x):
        return hard_swish_jit(x)


class MishJit(nn.Module):

    def __init__(self, inplace: 'bool'=False):
        super(MishJit, self).__init__()

    def forward(self, x):
        return mish_jit(x)


@torch.jit.script
def swish_jit(x, inplace: 'bool'=False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.mul(x.sigmoid())


class SwishJit(nn.Module):

    def __init__(self, inplace: 'bool'=False):
        super(SwishJit, self).__init__()

    def forward(self, x):
        return swish_jit(x)


_ACT_LAYER_JIT = dict(silu=nn.SiLU if _has_silu else SwishJit, swish=nn.SiLU if _has_silu else SwishJit, mish=MishJit, hard_sigmoid=HardSigmoidJit, hard_swish=HardSwishJit, hard_mish=HardMishJit)


@torch.jit.script
def hard_mish_jit_bwd(x, grad_output):
    m = torch.ones_like(x) * (x >= -2.0)
    m = torch.where((x >= -2.0) & (x <= 0.0), x + 1.0, m)
    return grad_output * m


@torch.jit.script
def hard_mish_jit_fwd(x):
    return 0.5 * x * (x + 2).clamp(min=0, max=2)


class HardMishJitAutoFn(torch.autograd.Function):
    """ A memory efficient, jit scripted variant of Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return hard_mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return hard_mish_jit_bwd(x, grad_output)


class HardMishMe(nn.Module):

    def __init__(self, inplace: 'bool'=False):
        super(HardMishMe, self).__init__()

    def forward(self, x):
        return HardMishJitAutoFn.apply(x)


@torch.jit.script
def hard_sigmoid_jit_bwd(x, grad_output):
    m = torch.ones_like(x) * ((x >= -3.0) & (x <= 3.0)) / 6.0
    return grad_output * m


@torch.jit.script
def hard_sigmoid_jit_fwd(x, inplace: 'bool'=False):
    return (x + 3).clamp(min=0, max=6).div(6.0)


class HardSigmoidJitAutoFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return hard_sigmoid_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return hard_sigmoid_jit_bwd(x, grad_output)


class HardSigmoidMe(nn.Module):

    def __init__(self, inplace: 'bool'=False):
        super(HardSigmoidMe, self).__init__()

    def forward(self, x):
        return HardSigmoidJitAutoFn.apply(x)


@torch.jit.script
def hard_swish_jit_bwd(x, grad_output):
    m = torch.ones_like(x) * (x >= 3.0)
    m = torch.where((x >= -3.0) & (x <= 3.0), x / 3.0 + 0.5, m)
    return grad_output * m


@torch.jit.script
def hard_swish_jit_fwd(x):
    return x * (x + 3).clamp(min=0, max=6).div(6.0)


class HardSwishJitAutoFn(torch.autograd.Function):
    """A memory efficient, jit-scripted HardSwish activation"""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return hard_swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return hard_swish_jit_bwd(x, grad_output)

    @staticmethod
    def symbolic(g, self):
        input = g.op('Add', self, g.op('Constant', value_t=torch.tensor(3, dtype=torch.float)))
        hardtanh_ = g.op('Clip', input, g.op('Constant', value_t=torch.tensor(0, dtype=torch.float)), g.op('Constant', value_t=torch.tensor(6, dtype=torch.float)))
        hardtanh_ = g.op('Div', hardtanh_, g.op('Constant', value_t=torch.tensor(6, dtype=torch.float)))
        return g.op('Mul', self, hardtanh_)


class HardSwishMe(nn.Module):

    def __init__(self, inplace: 'bool'=False):
        super(HardSwishMe, self).__init__()

    def forward(self, x):
        return HardSwishJitAutoFn.apply(x)


class MishJitAutoFn(torch.autograd.Function):
    """ Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    A memory efficient, jit scripted variant of Mish
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return mish_jit_bwd(x, grad_output)


class MishMe(nn.Module):

    def __init__(self, inplace: 'bool'=False):
        super(MishMe, self).__init__()

    def forward(self, x):
        return MishJitAutoFn.apply(x)


@torch.jit.script
def swish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid)))


@torch.jit.script
def swish_jit_fwd(x):
    return x.mul(torch.sigmoid(x))


class SwishJitAutoFn(torch.autograd.Function):
    """ torch.jit.script optimised Swish w/ memory-efficient checkpoint
    Inspired by conversation btw Jeremy Howard & Adam Pazske
    https://twitter.com/jeremyphoward/status/1188251041835315200
    """

    @staticmethod
    def symbolic(g, x):
        return g.op('Mul', x, g.op('Sigmoid', x))

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return swish_jit_bwd(x, grad_output)


class SwishMe(nn.Module):

    def __init__(self, inplace: 'bool'=False):
        super(SwishMe, self).__init__()

    def forward(self, x):
        return SwishJitAutoFn.apply(x)


_ACT_LAYER_ME = dict(silu=nn.SiLU if _has_silu else SwishMe, swish=nn.SiLU if _has_silu else SwishMe, mish=MishMe, hard_sigmoid=HardSigmoidMe, hard_swish=HardSwishMe, hard_mish=HardMishMe)


_EXPORTABLE = False


def is_exportable():
    return _EXPORTABLE


_NO_JIT = False


def is_no_jit():
    return _NO_JIT


_SCRIPTABLE = False


def is_scriptable():
    return _SCRIPTABLE


def get_act_layer(name='relu'):
    """ Activation Layer Factory
    Fetching activation layers by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if not name:
        return None
    if not (is_no_jit() or is_exportable() or is_scriptable()):
        if name in _ACT_LAYER_ME:
            return _ACT_LAYER_ME[name]
    if is_exportable() and name in ('silu', 'swish'):
        return Swish
    if not (is_no_jit() or is_exportable()):
        if name in _ACT_LAYER_JIT:
            return _ACT_LAYER_JIT[name]
    return _ACT_LAYER_DEFAULT[name]


class BatchNormAct2d(nn.BatchNorm2d):
    """BatchNorm + Activation

    This module performs BatchNorm + Activation in a manner that will remain backwards
    compatible with weights trained with separate bn, act. This is why we inherit from BN
    instead of composing it as a .bn member.
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, apply_act=True, act_layer=nn.ReLU, inplace=True, drop_block=None):
        super(BatchNormAct2d, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        if isinstance(act_layer, str):
            act_layer = get_act_layer(act_layer)
        if act_layer is not None and apply_act:
            act_args = dict(inplace=True) if inplace else {}
            self.act = act_layer(**act_args)
        else:
            self.act = nn.Identity()

    def _forward_jit(self, x):
        """ A cut & paste of the contents of the PyTorch BatchNorm2d forward function
        """
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        x = F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, self.training or not self.track_running_stats, exponential_average_factor, self.eps)
        return x

    @torch.jit.ignore
    def _forward_python(self, x):
        return super(BatchNormAct2d, self).forward(x)

    def forward(self, x):
        if torch.jit.is_scripting():
            x = self._forward_jit(x)
        else:
            x = self._forward_python(x)
        x = self.act(x)
        return x


class GroupNormAct(nn.GroupNorm):

    def __init__(self, num_channels, num_groups, eps=1e-05, affine=True, apply_act=True, act_layer=nn.ReLU, inplace=True, drop_block=None):
        super(GroupNormAct, self).__init__(num_groups, num_channels, eps=eps, affine=affine)
        if isinstance(act_layer, str):
            act_layer = get_act_layer(act_layer)
        if act_layer is not None and apply_act:
            act_args = dict(inplace=True) if inplace else {}
            self.act = act_layer(**act_args)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x = F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        x = self.act(x)
        return x


_NORM_ACT_REQUIRES_ARG = {BatchNormAct2d, GroupNormAct, InplaceAbn}


class EvoNormBatch2d(nn.Module):

    def __init__(self, num_features, apply_act=True, momentum=0.1, eps=1e-05, drop_block=None):
        super(EvoNormBatch2d, self).__init__()
        self.apply_act = apply_act
        self.momentum = momentum
        self.eps = eps
        param_shape = 1, num_features, 1, 1
        self.weight = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(param_shape), requires_grad=True)
        if apply_act:
            self.v = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.apply_act:
            nn.init.ones_(self.v)

    def forward(self, x):
        assert x.dim() == 4, 'expected 4D input'
        x_type = x.dtype
        if self.training:
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            n = x.numel() / x.shape[1]
            self.running_var.copy_(var.detach() * self.momentum * (n / (n - 1)) + self.running_var * (1 - self.momentum))
        else:
            var = self.running_var
        if self.apply_act:
            v = self.v
            d = x * v + (x.var(dim=(2, 3), unbiased=False, keepdim=True) + self.eps).sqrt()
            d = d.max((var + self.eps).sqrt())
            x = x / d
        return x * self.weight + self.bias


class EvoNormSample2d(nn.Module):

    def __init__(self, num_features, apply_act=True, groups=8, eps=1e-05, drop_block=None):
        super(EvoNormSample2d, self).__init__()
        self.apply_act = apply_act
        self.groups = groups
        self.eps = eps
        param_shape = 1, num_features, 1, 1
        self.weight = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(param_shape), requires_grad=True)
        if apply_act:
            self.v = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.apply_act:
            nn.init.ones_(self.v)

    def forward(self, x):
        assert x.dim() == 4, 'expected 4D input'
        B, C, H, W = x.shape
        assert C % self.groups == 0
        if self.apply_act:
            n = x * (x * self.v).sigmoid()
            x = x.reshape(B, self.groups, -1)
            x = n.reshape(B, self.groups, -1) / (x.var(dim=-1, unbiased=False, keepdim=True) + self.eps).sqrt()
            x = x.reshape(B, C, H, W)
        return x * self.weight + self.bias


_NORM_ACT_TYPES = {BatchNormAct2d, GroupNormAct, EvoNormBatch2d, EvoNormSample2d, InplaceAbn}


def get_norm_act_layer(layer_class):
    layer_class = layer_class.replace('_', '').lower()
    if layer_class.startswith('batchnorm'):
        layer = BatchNormAct2d
    elif layer_class.startswith('groupnorm'):
        layer = GroupNormAct
    elif layer_class == 'evonormbatch':
        layer = EvoNormBatch2d
    elif layer_class == 'evonormsample':
        layer = EvoNormSample2d
    elif layer_class == 'iabn' or layer_class == 'inplaceabn':
        layer = InplaceAbn
    else:
        assert False, 'Invalid norm_act layer (%s)' % layer_class
    return layer


def convert_norm_act(norm_layer, act_layer):
    assert isinstance(norm_layer, (type, str, types.FunctionType, functools.partial))
    assert act_layer is None or isinstance(act_layer, (type, str, types.FunctionType, functools.partial))
    norm_act_kwargs = {}
    if isinstance(norm_layer, functools.partial):
        norm_act_kwargs.update(norm_layer.keywords)
        norm_layer = norm_layer.func
    if isinstance(norm_layer, str):
        norm_act_layer = get_norm_act_layer(norm_layer)
    elif norm_layer in _NORM_ACT_TYPES:
        norm_act_layer = norm_layer
    elif isinstance(norm_layer, types.FunctionType):
        norm_act_layer = norm_layer
    else:
        type_name = norm_layer.__name__.lower()
        if type_name.startswith('batchnorm'):
            norm_act_layer = BatchNormAct2d
        elif type_name.startswith('groupnorm'):
            norm_act_layer = GroupNormAct
        else:
            assert False, f'No equivalent norm_act layer for {type_name}'
    if norm_act_layer in _NORM_ACT_REQUIRES_ARG:
        norm_act_kwargs.setdefault('act_layer', act_layer)
    if norm_act_kwargs:
        norm_act_layer = functools.partial(norm_act_layer, **norm_act_kwargs)
    return norm_act_layer


class ConvBnAct(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding='', dilation=1, groups=1, bias=False, apply_act=True, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, aa_layer=None, drop_block=None):
        super(ConvBnAct, self).__init__()
        use_aa = aa_layer is not None
        self.conv = create_conv2d(in_channels, out_channels, kernel_size, stride=1 if use_aa else stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        norm_act_layer = convert_norm_act(norm_layer, act_layer)
        self.bn = norm_act_layer(out_channels, apply_act=apply_act, drop_block=drop_block)
        self.aa = aa_layer(channels=out_channels) if stride == 2 and use_aa else None

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.aa is not None:
            x = self.aa(x)
        return x


def drop_path(x, drop_prob: 'float'=0.0, training: 'bool'=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _ex_tuple(cfg: 'dict', *names):
    return tuple([cfg.get(n, None) for n in names])


def _na_args(cfg: 'dict'):
    return dict(norm_layer=cfg.get('norm_layer', nn.BatchNorm2d), act_layer=cfg.get('act_layer', nn.ReLU))


def create_downsample(type, **kwargs):
    if type == 'avg':
        return DownsampleAvg(**kwargs)
    else:
        return ConvBnAct(kwargs.pop('in_chs'), kwargs.pop('out_chs'), kernel_size=1, **kwargs)


def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def num_groups(group_size, channels):
    if not group_size:
        return 1
    else:
        assert channels % group_size == 0
        return channels // group_size


class BottleneckBlock(nn.Module):
    """ ResNet-like Bottleneck Block - 1x1 - kxk - 1x1
    """

    def __init__(self, in_chs, out_chs, kernel_size=3, stride=1, dilation=(1, 1), bottle_ratio=1.0, group_size=None, downsample='avg', linear_out=False, layer_cfg=None, drop_block=None, drop_path_rate=0.0):
        super(BottleneckBlock, self).__init__()
        layer_cfg = layer_cfg or {}
        act_layer, attn_layer = _ex_tuple(layer_cfg, 'act_layer', 'attn_layer')
        layer_args = _na_args(layer_cfg)
        mid_chs = make_divisible(out_chs * bottle_ratio)
        groups = num_groups(group_size, mid_chs)
        if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
            self.shortcut = create_downsample(downsample, in_chs=in_chs, out_chs=out_chs, stride=stride, dilation=dilation[0], apply_act=False, **layer_args)
        else:
            self.shortcut = nn.Identity()
        self.conv1_1x1 = ConvBnAct(in_chs, mid_chs, 1, **layer_args)
        self.conv2_kxk = ConvBnAct(mid_chs, mid_chs, kernel_size, stride=stride, dilation=dilation[0], groups=groups, drop_block=drop_block, **layer_args)
        self.attn = nn.Identity() if attn_layer is None else attn_layer(mid_chs)
        self.conv3_1x1 = ConvBnAct(mid_chs, out_chs, 1, apply_act=False, **layer_args)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.act = nn.Identity() if linear_out else act_layer(inplace=True)

    def init_weights(self, zero_init_last_bn=False):
        if zero_init_last_bn:
            nn.init.zeros_(self.conv3_1x1.bn.weight)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1_1x1(x)
        x = self.conv2_kxk(x)
        x = self.attn(x)
        x = self.conv3_1x1(x)
        x = self.drop_path(x)
        x = self.act(x + shortcut)
        return x


class ChannelAttn(nn.Module):
    """ Original CBAM channel attention module, currently avg + max pool variant only.
    """

    def __init__(self, channels, reduction=16, act_layer=nn.ReLU):
        super(ChannelAttn, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)

    def forward(self, x):
        x_avg = x.mean((2, 3), keepdim=True)
        x_max = F.adaptive_max_pool2d(x, 1)
        x_avg = self.fc2(self.act(self.fc1(x_avg)))
        x_max = self.fc2(self.act(self.fc1(x_max)))
        x_attn = x_avg + x_max
        return x * x_attn.sigmoid()


class SpatialAttn(nn.Module):
    """ Original CBAM spatial attention module
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttn, self).__init__()
        self.conv = ConvBnAct(2, 1, kernel_size, act_layer=None)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        x_attn = torch.cat([x_avg, x_max], dim=1)
        x_attn = self.conv(x_attn)
        return x * x_attn.sigmoid()


class CbamModule(nn.Module):

    def __init__(self, channels, spatial_kernel_size=7):
        super(CbamModule, self).__init__()
        self.channel = ChannelAttn(channels)
        self.spatial = SpatialAttn(spatial_kernel_size)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


class CecaModule(nn.Module):
    """Constructs a circular ECA module.

    ECA module where the conv uses circular padding rather than zero padding.
    Unlike the spatial dimension, the channels do not have inherent ordering nor
    locality. Although this module in essence, applies such an assumption, it is unnecessary
    to limit the channels on either "edge" from being circularly adapted to each other.
    This will fundamentally increase connectivity and possibly increase performance metrics
    (accuracy, robustness), without significantly impacting resource metrics
    (parameter size, throughput,latency, etc)

    Args:
        channels: Number of channels of the input feature map for use in adaptive kernel sizes
            for actual calculations according to channel.
            gamma, beta: when channel is given parameters of mapping function
            refer to original paper https://arxiv.org/pdf/1910.03151.pdf
            (default=None. if channel size not given, use k_size given for kernel size.)
        kernel_size: Adaptive selection of kernel size (default=3)
    """

    def __init__(self, channels=None, kernel_size=3, gamma=2, beta=1):
        super(CecaModule, self).__init__()
        assert kernel_size % 2 == 1
        if channels is not None:
            t = int(abs(math.log(channels, 2) + beta) / gamma)
            kernel_size = max(t if t % 2 else t + 1, 3)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=0, bias=False)
        self.padding = (kernel_size - 1) // 2

    def forward(self, x):
        y = x.mean((2, 3)).view(x.shape[0], 1, -1)
        y = F.pad(y, (self.padding, self.padding), mode='circular')
        y = self.conv(y)
        y = y.view(x.shape[0], -1, 1, 1).sigmoid()
        return x * y.expand_as(x)


class EcaModule(nn.Module):
    """Constructs an ECA module.

    Args:
        channels: Number of channels of the input feature map for use in adaptive kernel sizes
            for actual calculations according to channel.
            gamma, beta: when channel is given parameters of mapping function
            refer to original paper https://arxiv.org/pdf/1910.03151.pdf
            (default=None. if channel size not given, use k_size given for kernel size.)
        kernel_size: Adaptive selection of kernel size (default=3)
    """

    def __init__(self, channels=None, kernel_size=3, gamma=2, beta=1):
        super(EcaModule, self).__init__()
        assert kernel_size % 2 == 1
        if channels is not None:
            t = int(abs(math.log(channels, 2) + beta) / gamma)
            kernel_size = max(t if t % 2 else t + 1, 3)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        y = x.mean((2, 3)).view(x.shape[0], 1, -1)
        y = self.conv(y)
        y = y.view(x.shape[0], -1, 1, 1).sigmoid()
        return x * y.expand_as(x)


def create_act_layer(name, inplace=False, **kwargs):
    act_layer = get_act_layer(name)
    if act_layer is not None:
        return act_layer(inplace=inplace, **kwargs)
    else:
        return None


class EffectiveSEModule(nn.Module):
    """ 'Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(self, channels, gate_layer='hard_sigmoid'):
        super(EffectiveSEModule, self).__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.gate = create_act_layer(gate_layer, inplace=True)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.gate(x_se)


class LightChannelAttn(ChannelAttn):
    """An experimental 'lightweight' that sums avg + max pool first
    """

    def __init__(self, channels, reduction=16):
        super(LightChannelAttn, self).__init__(channels, reduction)

    def forward(self, x):
        x_pool = 0.5 * x.mean((2, 3), keepdim=True) + 0.5 * F.adaptive_max_pool2d(x, 1)
        x_attn = self.fc2(self.act(self.fc1(x_pool)))
        return x * x_attn.sigmoid()


class LightSpatialAttn(nn.Module):
    """An experimental 'lightweight' variant that sums avg_pool and max_pool results.
    """

    def __init__(self, kernel_size=7):
        super(LightSpatialAttn, self).__init__()
        self.conv = ConvBnAct(1, 1, kernel_size, act_layer=None)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        x_attn = 0.5 * x_avg + 0.5 * x_max
        x_attn = self.conv(x_attn)
        return x * x_attn.sigmoid()


class LightCbamModule(nn.Module):

    def __init__(self, channels, spatial_kernel_size=7):
        super(LightCbamModule, self).__init__()
        self.channel = LightChannelAttn(channels)
        self.spatial = LightSpatialAttn(spatial_kernel_size)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


def get_attn(attn_type):
    if isinstance(attn_type, torch.nn.Module):
        return attn_type
    module_cls = None
    if attn_type is not None:
        if isinstance(attn_type, str):
            attn_type = attn_type.lower()
            if attn_type == 'se':
                module_cls = SEModule
            elif attn_type == 'ese':
                module_cls = EffectiveSEModule
            elif attn_type == 'eca':
                module_cls = EcaModule
            elif attn_type == 'ceca':
                module_cls = CecaModule
            elif attn_type == 'cbam':
                module_cls = CbamModule
            elif attn_type == 'lcbam':
                module_cls = LightCbamModule
            else:
                assert False, 'Invalid attn module (%s)' % attn_type
        elif isinstance(attn_type, bool):
            if attn_type:
                module_cls = SEModule
        else:
            module_cls = attn_type
    return module_cls


def create_attn(attn_type, channels, **kwargs):
    module_cls = get_attn(attn_type)
    if module_cls is not None:
        return module_cls(channels, **kwargs)
    return None


class DarkBlock(nn.Module):
    """ DarkNet Block
    """

    def __init__(self, in_chs, out_chs, dilation=1, bottle_ratio=0.5, groups=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(DarkBlock, self).__init__()
        mid_chs = int(round(out_chs * bottle_ratio))
        ckwargs = dict(act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer, drop_block=drop_block)
        self.conv1 = ConvBnAct(in_chs, mid_chs, kernel_size=1, **ckwargs)
        self.conv2 = ConvBnAct(mid_chs, out_chs, kernel_size=3, dilation=dilation, groups=groups, **ckwargs)
        self.attn = create_attn(attn_layer, channels=out_chs)
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv2.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.attn is not None:
            x = self.attn(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        x = x + shortcut
        return x


class EdgeBlock(nn.Module):
    """ EdgeResidual-like (3x3 + 1x1) block

    A two layer block like DarkBlock, but with the order of the 3x3 and 1x1 convs reversed.
    Very similar to the EfficientNet Edge-Residual block but this block it ends with activations, is
    intended to be used with either expansion or bottleneck contraction, and can use DW/group/non-grouped convs.

    FIXME is there a more common 3x3 + 1x1 conv block to name this after?
    """

    def __init__(self, in_chs, out_chs, kernel_size=3, stride=1, dilation=(1, 1), bottle_ratio=1.0, group_size=None, downsample='avg', linear_out=False, layer_cfg=None, drop_block=None, drop_path_rate=0.0):
        super(EdgeBlock, self).__init__()
        layer_cfg = layer_cfg or {}
        act_layer, attn_layer = _ex_tuple(layer_cfg, 'act_layer', 'attn_layer')
        layer_args = _na_args(layer_cfg)
        mid_chs = make_divisible(out_chs * bottle_ratio)
        groups = num_groups(group_size, mid_chs)
        if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
            self.shortcut = create_downsample(downsample, in_chs=in_chs, out_chs=out_chs, stride=stride, dilation=dilation[0], apply_act=False, **layer_args)
        else:
            self.shortcut = nn.Identity()
        self.conv1_kxk = ConvBnAct(in_chs, mid_chs, kernel_size, stride=stride, dilation=dilation[0], groups=groups, drop_block=drop_block, **layer_args)
        self.attn = nn.Identity() if attn_layer is None else attn_layer(out_chs)
        self.conv2_1x1 = ConvBnAct(mid_chs, out_chs, 1, apply_act=False, **layer_args)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.act = nn.Identity() if linear_out else act_layer(inplace=True)

    def init_weights(self, zero_init_last_bn=False):
        if zero_init_last_bn:
            nn.init.zeros_(self.conv2_1x1.bn.weight)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1_kxk(x)
        x = self.attn(x)
        x = self.conv2_1x1(x)
        x = self.drop_path(x)
        x = self.act(x + shortcut)
        return x


class RepVggBlock(nn.Module):
    """ RepVGG Block.

    Adapted from impl at https://github.com/DingXiaoH/RepVGG

    This version does not currently support the deploy optimization. It is currently fixed in 'train' mode.
    """

    def __init__(self, in_chs, out_chs, kernel_size=3, stride=1, dilation=(1, 1), bottle_ratio=1.0, group_size=None, downsample='', layer_cfg=None, drop_block=None, drop_path_rate=0.0):
        super(RepVggBlock, self).__init__()
        layer_cfg = layer_cfg or {}
        act_layer, norm_layer, attn_layer = _ex_tuple(layer_cfg, 'act_layer', 'norm_layer', 'attn_layer')
        norm_layer = convert_norm_act(norm_layer=norm_layer, act_layer=act_layer)
        layer_args = _na_args(layer_cfg)
        groups = num_groups(group_size, in_chs)
        use_ident = in_chs == out_chs and stride == 1 and dilation[0] == dilation[1]
        self.identity = norm_layer(out_chs, apply_act=False) if use_ident else None
        self.conv_kxk = ConvBnAct(in_chs, out_chs, kernel_size, stride=stride, dilation=dilation[0], groups=groups, drop_block=drop_block, apply_act=False, **layer_args)
        self.conv_1x1 = ConvBnAct(in_chs, out_chs, 1, stride=stride, groups=groups, apply_act=False, **layer_args)
        self.attn = nn.Identity() if attn_layer is None else attn_layer(out_chs)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 and use_ident else nn.Identity()
        self.act = act_layer(inplace=True)

    def init_weights(self, zero_init_last_bn=False):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0.1, 0.1)
                nn.init.normal_(m.bias, 0, 0.1)

    def forward(self, x):
        if self.identity is None:
            x = self.conv_1x1(x) + self.conv_kxk(x)
        else:
            identity = self.identity(x)
            x = self.conv_1x1(x) + self.conv_kxk(x)
            x = self.drop_path(x)
            x = x + identity
        x = self.attn(x)
        x = self.act(x)
        return x


class Linear(nn.Linear):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Wraps torch.nn.Linear to support AMP + torchscript usage by manually casting
    weight & bias to input.dtype to work around an issue w/ torch.addmm in this use case.
    """

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        if torch.jit.is_scripting():
            bias = self.bias if self.bias is not None else None
            return F.linear(input, self.weight, bias=bias)
        else:
            return F.linear(input, self.weight, self.bias)


def _create_fc(num_features, num_classes, pool_type='avg', use_conv=False):
    if num_classes <= 0:
        fc = nn.Identity()
    elif use_conv:
        fc = nn.Conv2d(num_features, num_classes, 1, bias=True)
    else:
        fc = Linear(num_features, num_classes, bias=True)
    return fc


def adaptive_avgmax_pool2d(x, output_size=1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return 0.5 * (x_avg + x_max)


class AdaptiveAvgMaxPool2d(nn.Module):

    def __init__(self, output_size=1):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_avgmax_pool2d(x, self.output_size)


def adaptive_catavgmax_pool2d(x, output_size=1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return torch.cat((x_avg, x_max), 1)


class AdaptiveCatAvgMaxPool2d(nn.Module):

    def __init__(self, output_size=1):
        super(AdaptiveCatAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_catavgmax_pool2d(x, self.output_size)


class FastAdaptiveAvgPool2d(nn.Module):

    def __init__(self, flatten=False):
        super(FastAdaptiveAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        return x.mean((2, 3)) if self.flatten else x.mean((2, 3), keepdim=True)


def adaptive_pool_feat_mult(pool_type='avg'):
    if pool_type == 'catavgmax':
        return 2
    else:
        return 1


class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """

    def __init__(self, output_size=1, pool_type='fast', flatten=False):
        super(SelectAdaptivePool2d, self).__init__()
        self.pool_type = pool_type or ''
        self.flatten = flatten
        if pool_type == '':
            self.pool = nn.Identity()
        elif pool_type == 'fast':
            assert output_size == 1
            self.pool = FastAdaptiveAvgPool2d(self.flatten)
            self.flatten = False
        elif pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(output_size)
        elif pool_type == 'avgmax':
            self.pool = AdaptiveAvgMaxPool2d(output_size)
        elif pool_type == 'catavgmax':
            self.pool = AdaptiveCatAvgMaxPool2d(output_size)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            assert False, 'Invalid pool type: %s' % pool_type

    def is_identity(self):
        return self.pool_type == ''

    def forward(self, x):
        x = self.pool(x)
        if self.flatten:
            x = x.flatten(1)
        return x

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + 'pool_type=' + self.pool_type + ', flatten=' + str(self.flatten) + ')'


def _create_pool(num_features, num_classes, pool_type='avg', use_conv=False):
    flatten_in_pool = not use_conv
    if not pool_type:
        assert num_classes == 0 or use_conv, 'Pooling can only be disabled if classifier is also removed or conv classifier is used'
        flatten_in_pool = False
    global_pool = SelectAdaptivePool2d(pool_type=pool_type, flatten=flatten_in_pool)
    num_pooled_features = num_features * global_pool.feat_mult()
    return global_pool, num_pooled_features


class ClassifierHead(nn.Module):
    """Classifier head w/ configurable global pooling and dropout."""

    def __init__(self, in_chs, num_classes, pool_type='avg', drop_rate=0.0, use_conv=False):
        super(ClassifierHead, self).__init__()
        self.drop_rate = drop_rate
        self.global_pool, num_pooled_features = _create_pool(in_chs, num_classes, pool_type, use_conv=use_conv)
        self.fc = _create_fc(num_pooled_features, num_classes, use_conv=use_conv)
        self.flatten_after_fc = use_conv and pool_type

    def forward(self, x):
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x


_block_registry = dict(basic=BasicBlock, bottle=BottleneckBlock, dark=DarkBlock, edge=EdgeBlock, rep=RepVggBlock)


def create_block(block: 'Union[str, nn.Module]', **kwargs):
    if isinstance(block, (nn.Module, partial)):
        return block(**kwargs)
    assert block in _block_registry, f'Unknown block type ({block}'
    return _block_registry[block](**kwargs)


def max_pool2d_same(x, kernel_size: 'List[int]', stride: 'List[int]', padding: 'List[int]'=(0, 0), dilation: 'List[int]'=(1, 1), ceil_mode: 'bool'=False):
    x = pad_same(x, kernel_size, stride, value=-float('inf'))
    return F.max_pool2d(x, kernel_size, stride, (0, 0), dilation, ceil_mode)


class MaxPool2dSame(nn.MaxPool2d):
    """ Tensorflow like 'SAME' wrapper for 2D max pooling
    """

    def __init__(self, kernel_size: 'int', stride=None, padding=0, dilation=1, ceil_mode=False, count_include_pad=True):
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        super(MaxPool2dSame, self).__init__(kernel_size, stride, (0, 0), dilation, ceil_mode, count_include_pad)

    def forward(self, x):
        return max_pool2d_same(x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode)


def create_pool2d(pool_type, kernel_size, stride=None, **kwargs):
    stride = stride or kernel_size
    padding = kwargs.pop('padding', '')
    padding, is_dynamic = get_padding_value(padding, kernel_size, stride=stride, **kwargs)
    if is_dynamic:
        if pool_type == 'avg':
            return AvgPool2dSame(kernel_size, stride=stride, **kwargs)
        elif pool_type == 'max':
            return MaxPool2dSame(kernel_size, stride=stride, **kwargs)
        else:
            assert False, f'Unsupported pool type {pool_type}'
    elif pool_type == 'avg':
        return nn.AvgPool2d(kernel_size, stride=stride, padding=padding, **kwargs)
    elif pool_type == 'max':
        return nn.MaxPool2d(kernel_size, stride=stride, padding=padding, **kwargs)
    else:
        assert False, f'Unsupported pool type {pool_type}'


def create_stem(in_chs, out_chs, stem_type='', preact=True, conv_layer=None, norm_layer=None):
    stem = OrderedDict()
    assert stem_type in ('', 'fixed', 'same', 'deep', 'deep_fixed', 'deep_same')
    if 'deep' in stem_type:
        mid_chs = out_chs // 2
        stem['conv1'] = conv_layer(in_chs, mid_chs, kernel_size=3, stride=2)
        stem['conv2'] = conv_layer(mid_chs, mid_chs, kernel_size=3, stride=1)
        stem['conv3'] = conv_layer(mid_chs, out_chs, kernel_size=3, stride=1)
    else:
        stem['conv'] = conv_layer(in_chs, out_chs, kernel_size=7, stride=2)
    if not preact:
        stem['norm'] = norm_layer(out_chs)
    if 'fixed' in stem_type:
        stem['pad'] = nn.ConstantPad2d(1, 0.0)
        stem['pool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
    elif 'same' in stem_type:
        stem['pool'] = create_pool2d('max', kernel_size=3, stride=2, padding='same')
    else:
        stem['pool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    return nn.Sequential(stem)


class ByobNet(nn.Module):
    """ 'Bring-your-own-blocks' Net

    A flexible network backbone that allows building model stem + blocks via
    dataclass cfg definition w/ factory functions for module instantiation.

    Current assumption is that both stem and blocks are in conv-bn-act order (w/ block ending in act).
    """

    def __init__(self, cfg: 'ByobCfg', num_classes=1000, in_chans=3, global_pool='avg', output_stride=32, zero_init_last_bn=True, drop_rate=0.0, drop_path_rate=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        norm_layer = cfg.norm_layer
        act_layer = get_act_layer(cfg.act_layer)
        attn_layer = partial(get_attn(cfg.attn_layer), **cfg.attn_kwargs) if cfg.attn_layer else None
        layer_cfg = dict(norm_layer=norm_layer, act_layer=act_layer, attn_layer=attn_layer)
        stem_chs = int(round((cfg.stem_chs or cfg.blocks[0].c) * cfg.width_factor))
        self.stem = create_stem(in_chans, stem_chs, cfg.stem_type, layer_cfg=layer_cfg)
        self.feature_info = []
        depths = [bc.d for bc in cfg.blocks]
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        prev_name = 'stem'
        prev_chs = stem_chs
        net_stride = 2
        dilation = 1
        stages = []
        for stage_idx, block_cfg in enumerate(cfg.blocks):
            stride = block_cfg.s
            if stride != 1:
                self.feature_info.append(dict(num_chs=prev_chs, reduction=net_stride, module=prev_name))
            if net_stride >= output_stride and stride > 1:
                dilation *= stride
                stride = 1
            net_stride *= stride
            first_dilation = 1 if dilation in (1, 2) else 2
            blocks = []
            for block_idx in range(block_cfg.d):
                out_chs = make_divisible(block_cfg.c * cfg.width_factor)
                group_size = block_cfg.gs
                if isinstance(group_size, Callable):
                    group_size = group_size(out_chs, block_idx)
                block_kwargs = dict(in_chs=prev_chs, out_chs=out_chs, stride=stride if block_idx == 0 else 1, dilation=(first_dilation, dilation), group_size=group_size, bottle_ratio=block_cfg.br, downsample=cfg.downsample, drop_path_rate=dpr[stage_idx][block_idx], layer_cfg=layer_cfg)
                blocks += [create_block(block_cfg.type, **block_kwargs)]
                first_dilation = dilation
                prev_chs = out_chs
            stages += [nn.Sequential(*blocks)]
            prev_name = f'stages.{stage_idx}'
        self.stages = nn.Sequential(*stages)
        if cfg.num_features:
            self.num_features = int(round(cfg.width_factor * cfg.num_features))
            self.final_conv = ConvBnAct(prev_chs, self.num_features, 1, **_na_args(layer_cfg))
        else:
            self.num_features = prev_chs
            self.final_conv = nn.Identity()
        self.feature_info += [dict(num_chs=self.num_features, reduction=net_stride, module='final_conv')]
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.modules():
            if hasattr(m, 'init_weights'):
                m.init_weights(zero_init_last_bn=zero_init_last_bn)

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.final_conv(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class ResBottleneck(nn.Module):
    """ ResNe(X)t Bottleneck Block
    """

    def __init__(self, in_chs, out_chs, dilation=1, bottle_ratio=0.25, groups=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_last=False, attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(ResBottleneck, self).__init__()
        mid_chs = int(round(out_chs * bottle_ratio))
        ckwargs = dict(act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer, drop_block=drop_block)
        self.conv1 = ConvBnAct(in_chs, mid_chs, kernel_size=1, **ckwargs)
        self.conv2 = ConvBnAct(mid_chs, mid_chs, kernel_size=3, dilation=dilation, groups=groups, **ckwargs)
        self.attn2 = create_attn(attn_layer, channels=mid_chs) if not attn_last else None
        self.conv3 = ConvBnAct(mid_chs, out_chs, kernel_size=1, apply_act=False, **ckwargs)
        self.attn3 = create_attn(attn_layer, channels=out_chs) if attn_last else None
        self.drop_path = drop_path
        self.act3 = act_layer(inplace=True)

    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.attn2 is not None:
            x = self.attn2(x)
        x = self.conv3(x)
        if self.attn3 is not None:
            x = self.attn3(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        x = x + shortcut
        x = self.act3(x)
        return x


class CrossStage(nn.Module):
    """Cross Stage."""

    def __init__(self, in_chs, out_chs, stride, dilation, depth, block_ratio=1.0, bottle_ratio=1.0, exp_ratio=1.0, groups=1, first_dilation=None, down_growth=False, cross_linear=False, block_dpr=None, block_fn=ResBottleneck, **block_kwargs):
        super(CrossStage, self).__init__()
        first_dilation = first_dilation or dilation
        down_chs = out_chs if down_growth else in_chs
        exp_chs = int(round(out_chs * exp_ratio))
        block_out_chs = int(round(out_chs * block_ratio))
        conv_kwargs = dict(act_layer=block_kwargs.get('act_layer'), norm_layer=block_kwargs.get('norm_layer'))
        if stride != 1 or first_dilation != dilation:
            self.conv_down = ConvBnAct(in_chs, down_chs, kernel_size=3, stride=stride, dilation=first_dilation, groups=groups, aa_layer=block_kwargs.get('aa_layer', None), **conv_kwargs)
            prev_chs = down_chs
        else:
            self.conv_down = None
            prev_chs = in_chs
        self.conv_exp = ConvBnAct(prev_chs, exp_chs, kernel_size=1, apply_act=not cross_linear, **conv_kwargs)
        prev_chs = exp_chs // 2
        self.blocks = nn.Sequential()
        for i in range(depth):
            drop_path = DropPath(block_dpr[i]) if block_dpr and block_dpr[i] else None
            self.blocks.add_module(str(i), block_fn(prev_chs, block_out_chs, dilation, bottle_ratio, groups, drop_path=drop_path, **block_kwargs))
            prev_chs = block_out_chs
        self.conv_transition_b = ConvBnAct(prev_chs, exp_chs // 2, kernel_size=1, **conv_kwargs)
        self.conv_transition = ConvBnAct(exp_chs, out_chs, kernel_size=1, **conv_kwargs)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        x = self.conv_exp(x)
        split = x.shape[1] // 2
        xs, xb = x[:, :split], x[:, split:]
        xb = self.blocks(xb)
        xb = self.conv_transition_b(xb).contiguous()
        out = self.conv_transition(torch.cat([xs, xb], dim=1))
        return out


class DarkStage(nn.Module):
    """DarkNet stage."""

    def __init__(self, in_chs, out_chs, stride, dilation, depth, block_ratio=1.0, bottle_ratio=1.0, groups=1, first_dilation=None, block_fn=ResBottleneck, block_dpr=None, **block_kwargs):
        super(DarkStage, self).__init__()
        first_dilation = first_dilation or dilation
        self.conv_down = ConvBnAct(in_chs, out_chs, kernel_size=3, stride=stride, dilation=first_dilation, groups=groups, act_layer=block_kwargs.get('act_layer'), norm_layer=block_kwargs.get('norm_layer'), aa_layer=block_kwargs.get('aa_layer', None))
        prev_chs = out_chs
        block_out_chs = int(round(out_chs * block_ratio))
        self.blocks = nn.Sequential()
        for i in range(depth):
            drop_path = DropPath(block_dpr[i]) if block_dpr and block_dpr[i] else None
            self.blocks.add_module(str(i), block_fn(prev_chs, block_out_chs, dilation, bottle_ratio, groups, drop_path=drop_path, **block_kwargs))
            prev_chs = block_out_chs

    def forward(self, x):
        x = self.conv_down(x)
        x = self.blocks(x)
        return x


def _cfg_to_stage_args(cfg, curr_stride=2, output_stride=32, drop_path_rate=0.0):
    num_stages = len(cfg['depth'])
    if 'groups' not in cfg:
        cfg['groups'] = (1,) * num_stages
    if 'down_growth' in cfg and not isinstance(cfg['down_growth'], (list, tuple)):
        cfg['down_growth'] = (cfg['down_growth'],) * num_stages
    if 'cross_linear' in cfg and not isinstance(cfg['cross_linear'], (list, tuple)):
        cfg['cross_linear'] = (cfg['cross_linear'],) * num_stages
    cfg['block_dpr'] = [None] * num_stages if not drop_path_rate else [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(cfg['depth'])).split(cfg['depth'])]
    stage_strides = []
    stage_dilations = []
    stage_first_dilations = []
    dilation = 1
    for cfg_stride in cfg['stride']:
        stage_first_dilations.append(dilation)
        if curr_stride >= output_stride:
            dilation *= cfg_stride
            stride = 1
        else:
            stride = cfg_stride
            curr_stride *= stride
        stage_strides.append(stride)
        stage_dilations.append(dilation)
    cfg['stride'] = stage_strides
    cfg['dilation'] = stage_dilations
    cfg['first_dilation'] = stage_first_dilations
    stage_args = [dict(zip(cfg.keys(), values)) for values in zip(*cfg.values())]
    return stage_args


class CspNet(nn.Module):
    """Cross Stage Partial base model.

    Paper: `CSPNet: A New Backbone that can Enhance Learning Capability of CNN` - https://arxiv.org/abs/1911.11929
    Ref Impl: https://github.com/WongKinYiu/CrossStagePartialNetworks

    NOTE: There are differences in the way I handle the 1x1 'expansion' conv in this impl vs the
    darknet impl. I did it this way for simplicity and less special cases.
    """

    def __init__(self, cfg, in_chans=3, num_classes=1000, output_stride=32, global_pool='avg', drop_rate=0.0, act_layer=nn.LeakyReLU, norm_layer=nn.BatchNorm2d, aa_layer=None, drop_path_rate=0.0, zero_init_last_bn=True, stage_fn=CrossStage, block_fn=ResBottleneck):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        assert output_stride in (8, 16, 32)
        layer_args = dict(act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer)
        self.stem, stem_feat_info = create_stem(in_chans, **cfg['stem'], **layer_args)
        self.feature_info = [stem_feat_info]
        prev_chs = stem_feat_info['num_chs']
        curr_stride = stem_feat_info['reduction']
        if cfg['stem']['pool']:
            curr_stride *= 2
        per_stage_args = _cfg_to_stage_args(cfg['stage'], curr_stride=curr_stride, output_stride=output_stride, drop_path_rate=drop_path_rate)
        self.stages = nn.Sequential()
        for i, sa in enumerate(per_stage_args):
            self.stages.add_module(str(i), stage_fn(prev_chs, **sa, **layer_args, block_fn=block_fn))
            prev_chs = sa['out_chs']
            curr_stride *= sa['stride']
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{i}')]
        self.num_features = prev_chs
        self.head = ClassifierHead(in_chs=prev_chs, num_classes=num_classes, pool_type=global_pool, drop_rate=drop_rate)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, 'zero_init_last_bn'):
                    m.zero_init_last_bn()

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class DenseLayer(nn.Module):

    def __init__(self, num_input_features, growth_rate, bn_size, norm_layer=BatchNormAct2d, drop_rate=0.0, memory_efficient=False):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', norm_layer(num_input_features)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', norm_layer(bn_size * growth_rate)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bottleneck_fn(self, xs):
        concated_features = torch.cat(xs, 1)
        bottleneck_output = self.conv1(self.norm1(concated_features))
        return bottleneck_output

    def any_requires_grad(self, x):
        for tensor in x:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused
    def call_checkpoint_bottleneck(self, x):

        def closure(*xs):
            return self.bottleneck_fn(xs)
        return cp.checkpoint(closure, *x)

    @torch.jit._overload_method
    def forward(self, x):
        pass

    @torch.jit._overload_method
    def forward(self, x):
        pass

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            prev_features = [x]
        else:
            prev_features = x
        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception('Memory Efficient not supported in JIT')
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bottleneck_fn(prev_features)
        new_features = self.conv2(self.norm2(bottleneck_output))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, norm_layer=nn.ReLU, drop_rate=0.0, memory_efficient=False):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate=growth_rate, bn_size=bn_size, norm_layer=norm_layer, drop_rate=drop_rate, memory_efficient=memory_efficient)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseTransition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features, norm_layer=nn.BatchNorm2d, aa_layer=None):
        super(DenseTransition, self).__init__()
        self.add_module('norm', norm_layer(num_input_features))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        if aa_layer is not None:
            self.add_module('pool', aa_layer(num_output_features, stride=2))
        else:
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


def create_classifier(num_features, num_classes, pool_type='avg', use_conv=False):
    global_pool, num_pooled_features = _create_pool(num_features, num_classes, pool_type, use_conv=use_conv)
    fc = _create_fc(num_pooled_features, num_classes, use_conv=use_conv)
    return global_pool, fc


class DenseNet(nn.Module):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), bn_size=4, stem_type='', num_classes=1000, in_chans=3, global_pool='avg', norm_layer=BatchNormAct2d, aa_layer=None, drop_rate=0, memory_efficient=False, aa_stem_only=True):
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        super(DenseNet, self).__init__()
        deep_stem = 'deep' in stem_type
        num_init_features = growth_rate * 2
        if aa_layer is None:
            stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            stem_pool = nn.Sequential(*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1), aa_layer(channels=num_init_features, stride=2)])
        if deep_stem:
            stem_chs_1 = stem_chs_2 = growth_rate
            if 'tiered' in stem_type:
                stem_chs_1 = 3 * (growth_rate // 4)
                stem_chs_2 = num_init_features if 'narrow' in stem_type else 6 * (growth_rate // 4)
            self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(in_chans, stem_chs_1, 3, stride=2, padding=1, bias=False)), ('norm0', norm_layer(stem_chs_1)), ('conv1', nn.Conv2d(stem_chs_1, stem_chs_2, 3, stride=1, padding=1, bias=False)), ('norm1', norm_layer(stem_chs_2)), ('conv2', nn.Conv2d(stem_chs_2, num_init_features, 3, stride=1, padding=1, bias=False)), ('norm2', norm_layer(num_init_features)), ('pool0', stem_pool)]))
        else:
            self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(in_chans, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)), ('norm0', norm_layer(num_init_features)), ('pool0', stem_pool)]))
        self.feature_info = [dict(num_chs=num_init_features, reduction=2, module=f'features.norm{2 if deep_stem else 0}')]
        current_stride = 4
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, norm_layer=norm_layer, drop_rate=drop_rate, memory_efficient=memory_efficient)
            module_name = f'denseblock{i + 1}'
            self.features.add_module(module_name, block)
            num_features = num_features + num_layers * growth_rate
            transition_aa_layer = None if aa_stem_only else aa_layer
            if i != len(block_config) - 1:
                self.feature_info += [dict(num_chs=num_features, reduction=current_stride, module='features.' + module_name)]
                current_stride *= 2
                trans = DenseTransition(num_input_features=num_features, num_output_features=num_features // 2, norm_layer=norm_layer, aa_layer=transition_aa_layer)
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2
        self.features.add_module('norm5', norm_layer(num_features))
        self.feature_info += [dict(num_chs=num_features, reduction=current_stride, module='features.norm5')]
        self.num_features = num_features
        self.global_pool, self.classifier = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.classifier = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        return self.features(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


class DlaBasic(nn.Module):
    """DLA Basic"""

    def __init__(self, inplanes, planes, stride=1, dilation=1, **_):
        super(DlaBasic, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class DlaBottleneck(nn.Module):
    """DLA/DLA-X Bottleneck"""
    expansion = 2

    def __init__(self, inplanes, outplanes, stride=1, dilation=1, cardinality=1, base_width=64):
        super(DlaBottleneck, self).__init__()
        self.stride = stride
        mid_planes = int(math.floor(outplanes * (base_width / 64)) * cardinality)
        mid_planes = mid_planes // self.expansion
        self.conv1 = nn.Conv2d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

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


class DlaBottle2neck(nn.Module):
    """ Res2Net/Res2NeXT DLA Bottleneck
    Adapted from https://github.com/gasvn/Res2Net/blob/master/dla.py
    """
    expansion = 2

    def __init__(self, inplanes, outplanes, stride=1, dilation=1, scale=4, cardinality=8, base_width=4):
        super(DlaBottle2neck, self).__init__()
        self.is_first = stride > 1
        self.scale = scale
        mid_planes = int(math.floor(outplanes * (base_width / 64)) * cardinality)
        mid_planes = mid_planes // self.expansion
        self.width = mid_planes
        self.conv1 = nn.Conv2d(inplanes, mid_planes * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes * scale)
        num_scale_convs = max(1, scale - 1)
        convs = []
        bns = []
        for _ in range(num_scale_convs):
            convs.append(nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=cardinality, bias=False))
            bns.append(nn.BatchNorm2d(mid_planes))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        if self.is_first:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        self.conv3 = nn.Conv2d(mid_planes * scale, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        spo = []
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            sp = spx[i] if i == 0 or self.is_first else sp + spx[i]
            sp = conv(sp)
            sp = bn(sp)
            sp = self.relu(sp)
            spo.append(sp)
        if self.scale > 1:
            spo.append(self.pool(spx[-1]) if self.is_first else spx[-1])
        out = torch.cat(spo, 1)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out


class DlaRoot(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(DlaRoot, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels)
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


class DlaTree(nn.Module):

    def __init__(self, levels, block, in_channels, out_channels, stride=1, dilation=1, cardinality=1, base_width=64, level_root=False, root_dim=0, root_kernel_size=1, root_residual=False):
        super(DlaTree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        self.downsample = nn.MaxPool2d(stride, stride=stride) if stride > 1 else nn.Identity()
        self.project = nn.Identity()
        cargs = dict(dilation=dilation, cardinality=cardinality, base_width=base_width)
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, **cargs)
            self.tree2 = block(out_channels, out_channels, 1, **cargs)
            if in_channels != out_channels:
                self.project = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(out_channels))
        else:
            cargs.update(dict(root_kernel_size=root_kernel_size, root_residual=root_residual))
            self.tree1 = DlaTree(levels - 1, block, in_channels, out_channels, stride, root_dim=0, **cargs)
            self.tree2 = DlaTree(levels - 1, block, out_channels, out_channels, root_dim=root_dim + out_channels, **cargs)
        if levels == 1:
            self.root = DlaRoot(root_dim, out_channels, root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.levels = levels

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x)
        residual = self.project(bottom)
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


class DLA(nn.Module):

    def __init__(self, levels, channels, output_stride=32, num_classes=1000, in_chans=3, cardinality=1, base_width=64, block=DlaBottle2neck, residual_root=False, drop_rate=0.0, global_pool='avg'):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.cardinality = cardinality
        self.base_width = base_width
        self.drop_rate = drop_rate
        assert output_stride == 32
        self.base_layer = nn.Sequential(nn.Conv2d(in_chans, channels[0], kernel_size=7, stride=1, padding=3, bias=False), nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
        cargs = dict(cardinality=cardinality, base_width=base_width, root_residual=residual_root)
        self.level2 = DlaTree(levels[2], block, channels[1], channels[2], 2, level_root=False, **cargs)
        self.level3 = DlaTree(levels[3], block, channels[2], channels[3], 2, level_root=True, **cargs)
        self.level4 = DlaTree(levels[4], block, channels[3], channels[4], 2, level_root=True, **cargs)
        self.level5 = DlaTree(levels[5], block, channels[4], channels[5], 2, level_root=True, **cargs)
        self.feature_info = [dict(num_chs=channels[0], reduction=1, module='level0'), dict(num_chs=channels[1], reduction=2, module='level1'), dict(num_chs=channels[2], reduction=4, module='level2'), dict(num_chs=channels[3], reduction=8, module='level3'), dict(num_chs=channels[4], reduction=16, module='level4'), dict(num_chs=channels[5], reduction=32, module='level5')]
        self.num_features = channels[-1]
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool, use_conv=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride if i == 0 else 1, padding=dilation, bias=False, dilation=dilation), nn.BatchNorm2d(planes), nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool, use_conv=True)

    def forward_features(self, x):
        x = self.base_layer(x)
        x = self.level0(x)
        x = self.level1(x)
        x = self.level2(x)
        x = self.level3(x)
        x = self.level4(x)
        x = self.level5(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc(x)
        if not self.global_pool.is_identity():
            x = x.flatten(1)
        return x


class CatBnAct(nn.Module):

    def __init__(self, in_chs, norm_layer=BatchNormAct2d):
        super(CatBnAct, self).__init__()
        self.bn = norm_layer(in_chs, eps=0.001)

    @torch.jit._overload_method
    def forward(self, x):
        pass

    @torch.jit._overload_method
    def forward(self, x):
        pass

    def forward(self, x):
        if isinstance(x, tuple):
            x = torch.cat(x, dim=1)
        return self.bn(x)


class BnActConv2d(nn.Module):

    def __init__(self, in_chs, out_chs, kernel_size, stride, groups=1, norm_layer=BatchNormAct2d):
        super(BnActConv2d, self).__init__()
        self.bn = norm_layer(in_chs, eps=0.001)
        self.conv = create_conv2d(in_chs, out_chs, kernel_size, stride=stride, groups=groups)

    def forward(self, x):
        return self.conv(self.bn(x))


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

    def __init__(self, small=False, num_init_features=64, k_r=96, groups=32, b=False, k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128), output_stride=32, num_classes=1000, in_chans=3, drop_rate=0.0, global_pool='avg', fc_act=nn.ELU):
        super(DPN, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.b = b
        assert output_stride == 32
        norm_layer = partial(BatchNormAct2d, eps=0.001)
        fc_norm_layer = partial(BatchNormAct2d, eps=0.001, act_layer=fc_act, inplace=False)
        bw_factor = 1 if small else 4
        blocks = OrderedDict()
        blocks['conv1_1'] = ConvBnAct(in_chans, num_init_features, kernel_size=3 if small else 7, stride=2, norm_layer=norm_layer)
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
        self.global_pool, self.classifier = create_classifier(self.num_features, self.num_classes, pool_type=global_pool, use_conv=True)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.classifier = create_classifier(self.num_features, self.num_classes, pool_type=global_pool, use_conv=True)

    def forward_features(self, x):
        return self.features(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        if not self.global_pool.is_identity():
            x = x.flatten(1)
        return x


def sigmoid(x, inplace: 'bool'=False):
    return x.sigmoid_() if inplace else x.sigmoid()


class SqueezeExcite(nn.Module):

    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None, act_layer=nn.ReLU, gate_fn=sigmoid, divisor=1, **_):
        super(SqueezeExcite, self).__init__()
        reduced_chs = make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)
        self.gate_fn = gate_fn

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate_fn(x_se)


_SE_ARGS_DEFAULT = dict(gate_fn=sigmoid, act_layer=None, reduce_mid=False, divisor=1)


def resolve_se_args(kwargs, in_chs, act_layer=None):
    se_kwargs = kwargs.copy() if kwargs is not None else {}
    for k, v in _SE_ARGS_DEFAULT.items():
        se_kwargs.setdefault(k, v)
    if not se_kwargs.pop('reduce_mid'):
        se_kwargs['reduced_base_chs'] = in_chs
    if se_kwargs['act_layer'] is None:
        assert act_layer is not None
        se_kwargs['act_layer'] = act_layer
    return se_kwargs


class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE and CondConv routing"""

    def __init__(self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, pad_type='', act_layer=nn.ReLU, noskip=False, exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1, se_ratio=0.0, se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None, conv_kwargs=None, drop_path_rate=0.0):
        super(InvertedResidual, self).__init__()
        norm_kwargs = norm_kwargs or {}
        conv_kwargs = conv_kwargs or {}
        mid_chs = make_divisible(in_chs * exp_ratio)
        has_se = se_ratio is not None and se_ratio > 0.0
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_path_rate = drop_path_rate
        self.conv_pw = create_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn1 = norm_layer(mid_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        self.conv_dw = create_conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride, dilation=dilation, padding=pad_type, depthwise=True, **conv_kwargs)
        self.bn2 = norm_layer(mid_chs, **norm_kwargs)
        self.act2 = act_layer(inplace=True)
        if has_se:
            se_kwargs = resolve_se_args(se_kwargs, in_chs, act_layer)
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio, **se_kwargs)
        else:
            self.se = None
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn3 = norm_layer(out_chs, **norm_kwargs)

    def feature_info(self, location):
        if location == 'expansion':
            info = dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else:
            info = dict(module='', hook_type='', num_chs=self.conv_pwl.out_channels)
        return info

    def forward(self, x):
        residual = x
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)
        if self.se is not None:
            x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)
        if self.has_residual:
            if self.drop_path_rate > 0.0:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += residual
        return x


class CondConvResidual(InvertedResidual):
    """ Inverted residual block w/ CondConv routing"""

    def __init__(self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, pad_type='', act_layer=nn.ReLU, noskip=False, exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1, se_ratio=0.0, se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None, num_experts=0, drop_path_rate=0.0):
        self.num_experts = num_experts
        conv_kwargs = dict(num_experts=self.num_experts)
        super(CondConvResidual, self).__init__(in_chs, out_chs, dw_kernel_size=dw_kernel_size, stride=stride, dilation=dilation, pad_type=pad_type, act_layer=act_layer, noskip=noskip, exp_ratio=exp_ratio, exp_kernel_size=exp_kernel_size, pw_kernel_size=pw_kernel_size, se_ratio=se_ratio, se_kwargs=se_kwargs, norm_layer=norm_layer, norm_kwargs=norm_kwargs, conv_kwargs=conv_kwargs, drop_path_rate=drop_path_rate)
        self.routing_fn = nn.Linear(in_chs, self.num_experts)

    def forward(self, x):
        residual = x
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)
        routing_weights = torch.sigmoid(self.routing_fn(pooled_inputs))
        x = self.conv_pw(x, routing_weights)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv_dw(x, routing_weights)
        x = self.bn2(x)
        x = self.act2(x)
        if self.se is not None:
            x = self.se(x)
        x = self.conv_pwl(x, routing_weights)
        x = self.bn3(x)
        if self.has_residual:
            if self.drop_path_rate > 0.0:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += residual
        return x


class DepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """

    def __init__(self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, pad_type='', act_layer=nn.ReLU, noskip=False, pw_kernel_size=1, pw_act=False, se_ratio=0.0, se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None, drop_path_rate=0.0):
        super(DepthwiseSeparableConv, self).__init__()
        norm_kwargs = norm_kwargs or {}
        has_se = se_ratio is not None and se_ratio > 0.0
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act
        self.drop_path_rate = drop_path_rate
        self.conv_dw = create_conv2d(in_chs, in_chs, dw_kernel_size, stride=stride, dilation=dilation, padding=pad_type, depthwise=True)
        self.bn1 = norm_layer(in_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        if has_se:
            se_kwargs = resolve_se_args(se_kwargs, in_chs, act_layer)
            self.se = SqueezeExcite(in_chs, se_ratio=se_ratio, **se_kwargs)
        else:
            self.se = None
        self.conv_pw = create_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_layer(out_chs, **norm_kwargs)
        self.act2 = act_layer(inplace=True) if self.has_pw_act else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':
            info = dict(module='conv_pw', hook_type='forward_pre', num_chs=self.conv_pw.in_channels)
        else:
            info = dict(module='', hook_type='', num_chs=self.conv_pw.out_channels)
        return info

    def forward(self, x):
        residual = x
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.se is not None:
            x = self.se(x)
        x = self.conv_pw(x)
        x = self.bn2(x)
        x = self.act2(x)
        if self.has_residual:
            if self.drop_path_rate > 0.0:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += residual
        return x


class EdgeResidual(nn.Module):
    """ Residual block with expansion convolution followed by pointwise-linear w/ stride"""

    def __init__(self, in_chs, out_chs, exp_kernel_size=3, exp_ratio=1.0, fake_in_chs=0, stride=1, dilation=1, pad_type='', act_layer=nn.ReLU, noskip=False, pw_kernel_size=1, se_ratio=0.0, se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None, drop_path_rate=0.0):
        super(EdgeResidual, self).__init__()
        norm_kwargs = norm_kwargs or {}
        if fake_in_chs > 0:
            mid_chs = make_divisible(fake_in_chs * exp_ratio)
        else:
            mid_chs = make_divisible(in_chs * exp_ratio)
        has_se = se_ratio is not None and se_ratio > 0.0
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_path_rate = drop_path_rate
        self.conv_exp = create_conv2d(in_chs, mid_chs, exp_kernel_size, stride=stride, dilation=dilation, padding=pad_type)
        self.bn1 = norm_layer(mid_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        if has_se:
            se_kwargs = resolve_se_args(se_kwargs, in_chs, act_layer)
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio, **se_kwargs)
        else:
            self.se = None
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_layer(out_chs, **norm_kwargs)

    def feature_info(self, location):
        if location == 'expansion':
            info = dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else:
            info = dict(module='', hook_type='', num_chs=self.conv_pwl.out_channels)
        return info

    def forward(self, x):
        residual = x
        x = self.conv_exp(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.se is not None:
            x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn2(x)
        if self.has_residual:
            if self.drop_path_rate > 0.0:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += residual
        return x


_logger = logging.getLogger(__name__)


def _log_info_if(msg, condition):
    if condition:
        _logger.info(msg)


def round_channels(channels, multiplier=1.0, divisor=8, channel_min=None):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return channels
    channels *= multiplier
    return make_divisible(channels, divisor, channel_min)


class EfficientNetBuilder:
    """ Build Trunk Blocks

    This ended up being somewhat of a cross between
    https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_models.py
    and
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/fbnet_builder.py

    """

    def __init__(self, channel_multiplier=1.0, channel_divisor=8, channel_min=None, output_stride=32, pad_type='', act_layer=None, se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None, drop_path_rate=0.0, feature_location='', verbose=False):
        self.channel_multiplier = channel_multiplier
        self.channel_divisor = channel_divisor
        self.channel_min = channel_min
        self.output_stride = output_stride
        self.pad_type = pad_type
        self.act_layer = act_layer
        self.se_kwargs = se_kwargs
        self.norm_layer = norm_layer
        self.norm_kwargs = norm_kwargs
        self.drop_path_rate = drop_path_rate
        if feature_location == 'depthwise':
            _logger.warning("feature_location=='depthwise' is deprecated, using 'expansion'")
            feature_location = 'expansion'
        self.feature_location = feature_location
        assert feature_location in ('bottleneck', 'expansion', '')
        self.verbose = verbose
        self.in_chs = None
        self.features = []

    def _round_channels(self, chs):
        return round_channels(chs, self.channel_multiplier, self.channel_divisor, self.channel_min)

    def _make_block(self, ba, block_idx, block_count):
        drop_path_rate = self.drop_path_rate * block_idx / block_count
        bt = ba.pop('block_type')
        ba['in_chs'] = self.in_chs
        ba['out_chs'] = self._round_channels(ba['out_chs'])
        if 'fake_in_chs' in ba and ba['fake_in_chs']:
            ba['fake_in_chs'] = self._round_channels(ba['fake_in_chs'])
        ba['norm_layer'] = self.norm_layer
        ba['norm_kwargs'] = self.norm_kwargs
        ba['pad_type'] = self.pad_type
        ba['act_layer'] = ba['act_layer'] if ba['act_layer'] is not None else self.act_layer
        assert ba['act_layer'] is not None
        if bt == 'ir':
            ba['drop_path_rate'] = drop_path_rate
            ba['se_kwargs'] = self.se_kwargs
            _log_info_if('  InvertedResidual {}, Args: {}'.format(block_idx, str(ba)), self.verbose)
            if ba.get('num_experts', 0) > 0:
                block = CondConvResidual(**ba)
            else:
                block = InvertedResidual(**ba)
        elif bt == 'ds' or bt == 'dsa':
            ba['drop_path_rate'] = drop_path_rate
            ba['se_kwargs'] = self.se_kwargs
            _log_info_if('  DepthwiseSeparable {}, Args: {}'.format(block_idx, str(ba)), self.verbose)
            block = DepthwiseSeparableConv(**ba)
        elif bt == 'er':
            ba['drop_path_rate'] = drop_path_rate
            ba['se_kwargs'] = self.se_kwargs
            _log_info_if('  EdgeResidual {}, Args: {}'.format(block_idx, str(ba)), self.verbose)
            block = EdgeResidual(**ba)
        elif bt == 'cn':
            _log_info_if('  ConvBnAct {}, Args: {}'.format(block_idx, str(ba)), self.verbose)
            block = ConvBnAct(**ba)
        else:
            assert False, 'Uknkown block type (%s) while building model.' % bt
        self.in_chs = ba['out_chs']
        return block

    def __call__(self, in_chs, model_block_args):
        """ Build the blocks
        Args:
            in_chs: Number of input-channels passed to first block
            model_block_args: A list of lists, outer list defines stages, inner
                list contains strings defining block configuration(s)
        Return:
             List of block stacks (each stack wrapped in nn.Sequential)
        """
        _log_info_if('Building model trunk with %d stages...' % len(model_block_args), self.verbose)
        self.in_chs = in_chs
        total_block_count = sum([len(x) for x in model_block_args])
        total_block_idx = 0
        current_stride = 2
        current_dilation = 1
        stages = []
        if model_block_args[0][0]['stride'] > 1:
            feature_info = dict(module='act1', num_chs=in_chs, stage=0, reduction=current_stride, hook_type='forward' if self.feature_location != 'bottleneck' else '')
            self.features.append(feature_info)
        for stack_idx, stack_args in enumerate(model_block_args):
            last_stack = stack_idx + 1 == len(model_block_args)
            _log_info_if('Stack: {}'.format(stack_idx), self.verbose)
            assert isinstance(stack_args, list)
            blocks = []
            for block_idx, block_args in enumerate(stack_args):
                last_block = block_idx + 1 == len(stack_args)
                _log_info_if(' Block: {}'.format(block_idx), self.verbose)
                assert block_args['stride'] in (1, 2)
                if block_idx >= 1:
                    block_args['stride'] = 1
                extract_features = False
                if last_block:
                    next_stack_idx = stack_idx + 1
                    extract_features = next_stack_idx >= len(model_block_args) or model_block_args[next_stack_idx][0]['stride'] > 1
                next_dilation = current_dilation
                if block_args['stride'] > 1:
                    next_output_stride = current_stride * block_args['stride']
                    if next_output_stride > self.output_stride:
                        next_dilation = current_dilation * block_args['stride']
                        block_args['stride'] = 1
                        _log_info_if('  Converting stride to dilation to maintain output_stride=={}'.format(self.output_stride), self.verbose)
                    else:
                        current_stride = next_output_stride
                block_args['dilation'] = current_dilation
                if next_dilation != current_dilation:
                    current_dilation = next_dilation
                block = self._make_block(block_args, total_block_idx, total_block_count)
                blocks.append(block)
                if extract_features:
                    feature_info = dict(stage=stack_idx + 1, reduction=current_stride, **block.feature_info(self.feature_location))
                    module_name = f'blocks.{stack_idx}.{block_idx}'
                    leaf_name = feature_info.get('module', '')
                    feature_info['module'] = '.'.join([module_name, leaf_name]) if leaf_name else module_name
                    self.features.append(feature_info)
                total_block_idx += 1
            stages.append(nn.Sequential(*blocks))
        return stages


_DEBUG = False


def _init_weight_goog(m, n='', fix_group_fanout=True):
    """ Weight initialization as per Tensorflow official implementations.

    Args:
        m (nn.Module): module to init
        n (str): module name
        fix_group_fanout (bool): enable correct (matching Tensorflow TPU impl) fanout calculation w/ group convs

    Handles layers in EfficientNet, EfficientNet-CondConv, MixNet, MnasNet, MobileNetV3, etc:
    * https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    * https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    """
    if isinstance(m, CondConv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        init_weight_fn = get_condconv_initializer(lambda w: w.data.normal_(0, math.sqrt(2.0 / fan_out)), m.num_experts, m.weight_shape)
        init_weight_fn(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()


def efficientnet_init_weights(model: 'nn.Module', init_fn=None):
    init_fn = init_fn or _init_weight_goog
    for n, m in model.named_modules():
        init_fn(m, n)


class EfficientNet(nn.Module):
    """ (Generic) EfficientNet

    A flexible and performant PyTorch implementation of efficient network architectures, including:
      * EfficientNet B0-B8, L2
      * EfficientNet-EdgeTPU
      * EfficientNet-CondConv
      * MixNet S, M, L, XL
      * MnasNet A1, B1, and small
      * FBNet C
      * Single-Path NAS Pixel1

    """

    def __init__(self, block_args, num_classes=1000, num_features=1280, in_chans=3, stem_size=32, channel_multiplier=1.0, channel_divisor=8, channel_min=None, output_stride=32, pad_type='', fix_stem=False, act_layer=nn.ReLU, drop_rate=0.0, drop_path_rate=0.0, se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None, global_pool='avg'):
        super(EfficientNet, self).__init__()
        norm_kwargs = norm_kwargs or {}
        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate
        if not fix_stem:
            stem_size = round_channels(stem_size, channel_multiplier, channel_divisor, channel_min)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        builder = EfficientNetBuilder(channel_multiplier, channel_divisor, channel_min, output_stride, pad_type, act_layer, se_kwargs, norm_layer, norm_kwargs, drop_path_rate, verbose=_DEBUG)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = builder.features
        head_chs = builder.in_chs
        self.conv_head = create_conv2d(head_chs, self.num_features, 1, padding=pad_type)
        self.bn2 = norm_layer(self.num_features, **norm_kwargs)
        self.act2 = act_layer(inplace=True)
        self.global_pool, self.classifier = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
        efficientnet_init_weights(self)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1, self.act1]
        layers.extend(self.blocks)
        layers.extend([self.conv_head, self.bn2, self.act2, self.global_pool])
        layers.extend([nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.classifier = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)


class FeatureHooks:
    """ Feature Hook Helper

    This module helps with the setup and extraction of hooks for extracting features from
    internal nodes in a model by node name. This works quite well in eager Python but needs
    redesign for torcscript.
    """

    def __init__(self, hooks, named_modules, out_map=None, default_hook_type='forward'):
        modules = {k: v for k, v in named_modules}
        for i, h in enumerate(hooks):
            hook_name = h['module']
            m = modules[hook_name]
            hook_id = out_map[i] if out_map else hook_name
            hook_fn = partial(self._collect_output_hook, hook_id)
            hook_type = h['hook_type'] if 'hook_type' in h else default_hook_type
            if hook_type == 'forward_pre':
                m.register_forward_pre_hook(hook_fn)
            elif hook_type == 'forward':
                m.register_forward_hook(hook_fn)
            else:
                assert False, 'Unsupported hook type'
        self._feature_outputs = defaultdict(OrderedDict)

    def _collect_output_hook(self, hook_id, *args):
        x = args[-1]
        if isinstance(x, tuple):
            x = x[0]
        self._feature_outputs[x.device][hook_id] = x

    def get_output(self, device) ->Dict[str, torch.tensor]:
        output = self._feature_outputs[device]
        self._feature_outputs[device] = OrderedDict()
        return output


class FeatureInfo:

    def __init__(self, feature_info: 'List[Dict]', out_indices: 'Tuple[int]'):
        prev_reduction = 1
        for fi in feature_info:
            assert 'num_chs' in fi and fi['num_chs'] > 0
            assert 'reduction' in fi and fi['reduction'] >= prev_reduction
            prev_reduction = fi['reduction']
            assert 'module' in fi
        self.out_indices = out_indices
        self.info = feature_info

    def from_other(self, out_indices: 'Tuple[int]'):
        return FeatureInfo(deepcopy(self.info), out_indices)

    def get(self, key, idx=None):
        """ Get value by key at specified index (indices)
        if idx == None, returns value for key at each output index
        if idx is an integer, return value for that feature module index (ignoring output indices)
        if idx is a list/tupple, return value for each module index (ignoring output indices)
        """
        if idx is None:
            return [self.info[i][key] for i in self.out_indices]
        if isinstance(idx, (tuple, list)):
            return [self.info[i][key] for i in idx]
        else:
            return self.info[idx][key]

    def get_dicts(self, keys=None, idx=None):
        """ return info dicts for specified keys (or all if None) at specified indices (or out_indices if None)
        """
        if idx is None:
            if keys is None:
                return [self.info[i] for i in self.out_indices]
            else:
                return [{k: self.info[i][k] for k in keys} for i in self.out_indices]
        if isinstance(idx, (tuple, list)):
            return [(self.info[i] if keys is None else {k: self.info[i][k] for k in keys}) for i in idx]
        else:
            return self.info[idx] if keys is None else {k: self.info[idx][k] for k in keys}

    def channels(self, idx=None):
        """ feature channels accessor
        """
        return self.get('num_chs', idx)

    def reduction(self, idx=None):
        """ feature reduction (output stride) accessor
        """
        return self.get('reduction', idx)

    def module_name(self, idx=None):
        """ feature module name accessor
        """
        return self.get('module', idx)

    def __getitem__(self, item):
        return self.info[item]

    def __len__(self):
        return len(self.info)


class EfficientNetFeatures(nn.Module):
    """ EfficientNet Feature Extractor

    A work-in-progress feature extraction module for EfficientNet, to use as a backbone for segmentation
    and object detection models.
    """

    def __init__(self, block_args, out_indices=(0, 1, 2, 3, 4), feature_location='bottleneck', in_chans=3, stem_size=32, channel_multiplier=1.0, channel_divisor=8, channel_min=None, output_stride=32, pad_type='', fix_stem=False, act_layer=nn.ReLU, drop_rate=0.0, drop_path_rate=0.0, se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(EfficientNetFeatures, self).__init__()
        norm_kwargs = norm_kwargs or {}
        self.drop_rate = drop_rate
        if not fix_stem:
            stem_size = round_channels(stem_size, channel_multiplier, channel_divisor, channel_min)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        builder = EfficientNetBuilder(channel_multiplier, channel_divisor, channel_min, output_stride, pad_type, act_layer, se_kwargs, norm_layer, norm_kwargs, drop_path_rate, feature_location=feature_location, verbose=_DEBUG)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = FeatureInfo(builder.features, out_indices)
        self._stage_out_idx = {v['stage']: i for i, v in enumerate(self.feature_info) if i in out_indices}
        efficientnet_init_weights(self)
        self.feature_hooks = None
        if feature_location != 'bottleneck':
            hooks = self.feature_info.get_dicts(keys=('module', 'hook_type'))
            self.feature_hooks = FeatureHooks(hooks, self.named_modules())

    def forward(self, x) ->List[torch.Tensor]:
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.feature_hooks is None:
            features = []
            if 0 in self._stage_out_idx:
                features.append(x)
            for i, b in enumerate(self.blocks):
                x = b(x)
                if i + 1 in self._stage_out_idx:
                    features.append(x)
            return features
        else:
            self.blocks(x)
            out = self.feature_hooks.get_output(x.device)
            return list(out.values())


class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert C % g == 0, 'Incompatible group size {} for input channel {}'.format(g, C)
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)


def _get_feature_info(net, out_indices):
    feature_info = getattr(net, 'feature_info')
    if isinstance(feature_info, FeatureInfo):
        return feature_info.from_other(out_indices)
    elif isinstance(feature_info, (list, tuple)):
        return FeatureInfo(net.feature_info, out_indices)
    else:
        assert False, 'Provided feature_info is not valid'


def _get_return_layers(feature_info, out_map):
    module_names = feature_info.module_name()
    return_layers = {}
    for i, name in enumerate(module_names):
        return_layers[name] = out_map[i] if out_map is not None else feature_info.out_indices[i]
    return return_layers


def _module_list(module, flatten_sequential=False):
    ml = []
    for name, module in module.named_children():
        if flatten_sequential and isinstance(module, nn.Sequential):
            for child_name, child_module in module.named_children():
                combined = [name, child_name]
                ml.append(('_'.join(combined), '.'.join(combined), child_module))
        else:
            ml.append((name, name, module))
    return ml


class FeatureDictNet(nn.ModuleDict):
    """ Feature extractor with OrderedDict return

    Wrap a model and extract features as specified by the out indices, the network is
    partially re-built from contained modules.

    There is a strong assumption that the modules have been registered into the model in the same
    order as they are used. There should be no reuse of the same nn.Module more than once, including
    trivial modules like `self.relu = nn.ReLU`.

    Only submodules that are directly assigned to the model class (`model.feature1`) or at most
    one Sequential container deep (`model.features.1`, with flatten_sequent=True) can be captured.
    All Sequential containers that are directly assigned to the original model will have their
    modules assigned to this module with the name `model.features.1` being changed to `model.features_1`

    Arguments:
        model (nn.Module): model from which we will extract the features
        out_indices (tuple[int]): model output indices to extract features for
        out_map (sequence): list or tuple specifying desired return id for each out index,
            otherwise str(index) is used
        feature_concat (bool): whether to concatenate intermediate features that are lists or tuples
            vs select element [0]
        flatten_sequential (bool): whether to flatten sequential modules assigned to model
    """

    def __init__(self, model, out_indices=(0, 1, 2, 3, 4), out_map=None, feature_concat=False, flatten_sequential=False):
        super(FeatureDictNet, self).__init__()
        self.feature_info = _get_feature_info(model, out_indices)
        self.concat = feature_concat
        self.return_layers = {}
        return_layers = _get_return_layers(self.feature_info, out_map)
        modules = _module_list(model, flatten_sequential=flatten_sequential)
        remaining = set(return_layers.keys())
        layers = OrderedDict()
        for new_name, old_name, module in modules:
            layers[new_name] = module
            if old_name in remaining:
                self.return_layers[new_name] = str(return_layers[old_name])
                remaining.remove(old_name)
            if not remaining:
                break
        assert not remaining and len(self.return_layers) == len(return_layers), f'Return layers ({remaining}) are not present in model'
        self.update(layers)

    def _collect(self, x) ->Dict[str, torch.Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_id = self.return_layers[name]
                if isinstance(x, (tuple, list)):
                    out[out_id] = torch.cat(x, 1) if self.concat else x[0]
                else:
                    out[out_id] = x
        return out

    def forward(self, x) ->Dict[str, torch.Tensor]:
        return self._collect(x)


class FeatureListNet(FeatureDictNet):
    """ Feature extractor with list return

    See docstring for FeatureDictNet above, this class exists only to appease Torchscript typing constraints.
    In eager Python we could have returned List[Tensor] vs Dict[id, Tensor] based on a member bool.
    """

    def __init__(self, model, out_indices=(0, 1, 2, 3, 4), out_map=None, feature_concat=False, flatten_sequential=False):
        super(FeatureListNet, self).__init__(model, out_indices=out_indices, out_map=out_map, feature_concat=feature_concat, flatten_sequential=flatten_sequential)

    def forward(self, x) ->List[torch.Tensor]:
        return list(self._collect(x).values())


class FeatureHookNet(nn.ModuleDict):
    """ FeatureHookNet

    Wrap a model and extract features specified by the out indices using forward/forward-pre hooks.

    If `no_rewrite` is True, features are extracted via hooks without modifying the underlying
    network in any way.

    If `no_rewrite` is False, the model will be re-written as in the
    FeatureList/FeatureDict case by folding first to second (Sequential only) level modules into this one.

    FIXME this does not currently work with Torchscript, see FeatureHooks class
    """

    def __init__(self, model, out_indices=(0, 1, 2, 3, 4), out_map=None, out_as_dict=False, no_rewrite=False, feature_concat=False, flatten_sequential=False, default_hook_type='forward'):
        super(FeatureHookNet, self).__init__()
        assert not torch.jit.is_scripting()
        self.feature_info = _get_feature_info(model, out_indices)
        self.out_as_dict = out_as_dict
        layers = OrderedDict()
        hooks = []
        if no_rewrite:
            assert not flatten_sequential
            if hasattr(model, 'reset_classifier'):
                model.reset_classifier(0)
            layers['body'] = model
            hooks.extend(self.feature_info.get_dicts())
        else:
            modules = _module_list(model, flatten_sequential=flatten_sequential)
            remaining = {f['module']: (f['hook_type'] if 'hook_type' in f else default_hook_type) for f in self.feature_info.get_dicts()}
            for new_name, old_name, module in modules:
                layers[new_name] = module
                for fn, fm in module.named_modules(prefix=old_name):
                    if fn in remaining:
                        hooks.append(dict(module=fn, hook_type=remaining[fn]))
                        del remaining[fn]
                if not remaining:
                    break
            assert not remaining, f'Return layers ({remaining}) are not present in model'
        self.update(layers)
        self.hooks = FeatureHooks(hooks, model.named_modules(), out_map=out_map)

    def forward(self, x):
        for name, module in self.items():
            x = module(x)
        out = self.hooks.get_output(x.device)
        return out if self.out_as_dict else list(out.values())


class Xception65(nn.Module):
    """Modified Aligned Xception.

    NOTE: only the 65 layer version is included here, the 71 layer variant
    was not correct and had no pretrained weights
    """

    def __init__(self, num_classes=1000, in_chans=3, output_stride=32, norm_layer=nn.BatchNorm2d, drop_rate=0.0, global_pool='avg'):
        super(Xception65, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        if output_stride == 32:
            entry_block3_stride = 2
            exit_block20_stride = 2
            middle_dilation = 1
            exit_dilation = 1, 1
        elif output_stride == 16:
            entry_block3_stride = 2
            exit_block20_stride = 1
            middle_dilation = 1
            exit_dilation = 1, 2
        elif output_stride == 8:
            entry_block3_stride = 1
            exit_block20_stride = 1
            middle_dilation = 2
            exit_dilation = 2, 4
        else:
            raise NotImplementedError
        self.conv1 = nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(num_features=32)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(num_features=64)
        self.act2 = nn.ReLU(inplace=True)
        self.block1 = Block(64, 128, stride=2, start_with_relu=False, norm_layer=norm_layer)
        self.block1_act = nn.ReLU(inplace=True)
        self.block2 = Block(128, 256, stride=2, start_with_relu=False, norm_layer=norm_layer)
        self.block3 = Block(256, 728, stride=entry_block3_stride, norm_layer=norm_layer)
        self.mid = nn.Sequential(OrderedDict([('block%d' % i, Block(728, 728, stride=1, dilation=middle_dilation, norm_layer=norm_layer)) for i in range(4, 20)]))
        self.block20 = Block(728, (728, 1024, 1024), stride=exit_block20_stride, dilation=exit_dilation[0], norm_layer=norm_layer)
        self.block20_act = nn.ReLU(inplace=True)
        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=exit_dilation[1], norm_layer=norm_layer)
        self.bn3 = norm_layer(num_features=1536)
        self.act3 = nn.ReLU(inplace=True)
        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=exit_dilation[1], norm_layer=norm_layer)
        self.bn4 = norm_layer(num_features=1536)
        self.act4 = nn.ReLU(inplace=True)
        self.num_features = 2048
        self.conv5 = SeparableConv2d(1536, self.num_features, 3, stride=1, dilation=exit_dilation[1], norm_layer=norm_layer)
        self.bn5 = norm_layer(num_features=self.num_features)
        self.act5 = nn.ReLU(inplace=True)
        self.feature_info = [dict(num_chs=64, reduction=2, module='act2'), dict(num_chs=128, reduction=4, module='block1_act'), dict(num_chs=256, reduction=8, module='block3.rep.act1'), dict(num_chs=728, reduction=16, module='block20.rep.act1'), dict(num_chs=2048, reduction=32, module='act5')]
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.block1(x)
        x = self.block1_act(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.mid(x)
        x = self.block20(x)
        x = self.block20_act(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act5(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate:
            F.dropout(x, self.drop_rate, training=self.training)
        x = self.fc(x)
        return x


_BN_MOMENTUM = 0.1


class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.fuse_act = nn.ReLU(False)

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        error_msg = ''
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
        elif num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(num_channels))
        elif num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(num_branches, len(num_inchannels))
        if error_msg:
            _logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=_BN_MOMENTUM))
        layers = [block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample)]
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return nn.Identity()
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False), nn.BatchNorm2d(num_inchannels[i], momentum=_BN_MOMENTUM), nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(nn.Identity())
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), nn.BatchNorm2d(num_outchannels_conv3x3, momentum=_BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), nn.BatchNorm2d(num_outchannels_conv3x3, momentum=_BN_MOMENTUM), nn.ReLU(False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x: 'List[torch.Tensor]'):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i, branch in enumerate(self.branches):
            x[i] = branch(x[i])
        x_fuse = []
        for i, fuse_outer in enumerate(self.fuse_layers):
            y = x[0] if i == 0 else fuse_outer[0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + fuse_outer[j](x[j])
            x_fuse.append(self.fuse_act(y))
        return x_fuse


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, act_layer='leaky_relu', aa_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv2d_iabn(inplanes, planes, kernel_size=1, stride=1, act_layer=act_layer, act_param=0.001)
        if stride == 1:
            self.conv2 = conv2d_iabn(planes, planes, kernel_size=3, stride=1, act_layer=act_layer, act_param=0.001)
        elif aa_layer is None:
            self.conv2 = conv2d_iabn(planes, planes, kernel_size=3, stride=2, act_layer=act_layer, act_param=0.001)
        else:
            self.conv2 = nn.Sequential(conv2d_iabn(planes, planes, kernel_size=3, stride=1, act_layer=act_layer, act_param=0.001), aa_layer(channels=planes, filt_size=3, stride=2))
        reduction_chs = max(planes * self.expansion // 8, 64)
        self.se = SEModule(planes, reduction_channels=reduction_chs) if use_se else None
        self.conv3 = conv2d_iabn(planes, planes * self.expansion, kernel_size=1, stride=1, act_layer='identity')
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)
        out = self.conv3(out)
        out = out + residual
        out = self.relu(out)
        return out


blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}


class HighResolutionNet(nn.Module):

    def __init__(self, cfg, in_chans=3, num_classes=1000, global_pool='avg', drop_rate=0.0, head='classification'):
        super(HighResolutionNet, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        stem_width = cfg['STEM_WIDTH']
        self.conv1 = nn.Conv2d(in_chans, stem_width, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stem_width, momentum=_BN_MOMENTUM)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(stem_width, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=_BN_MOMENTUM)
        self.act2 = nn.ReLU(inplace=True)
        self.stage1_cfg = cfg['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels
        self.stage2_cfg = cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)
        self.stage3_cfg = cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)
        self.stage4_cfg = cfg['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)
        self.head = head
        self.head_channels = None
        if head == 'classification':
            self.num_features = 2048
            self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(pre_stage_channels)
            self.global_pool, self.classifier = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
        elif head == 'incre':
            self.num_features = 2048
            self.incre_modules, _, _ = self._make_head(pre_stage_channels, True)
        else:
            self.incre_modules = None
            self.num_features = 256
        curr_stride = 2
        self.feature_info = [dict(num_chs=64, reduction=curr_stride, module='stem')]
        for i, c in enumerate(self.head_channels if self.head_channels else num_channels):
            curr_stride *= 2
            c = c * 4 if self.head_channels else c
            self.feature_info += [dict(num_chs=c, reduction=curr_stride, module=f'stage{i + 1}')]
        self.init_weights()

    def _make_head(self, pre_stage_channels, incre_only=False):
        head_block = Bottleneck
        self.head_channels = [32, 64, 128, 256]
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_modules.append(self._make_layer(head_block, channels, self.head_channels[i], 1, stride=1))
        incre_modules = nn.ModuleList(incre_modules)
        if incre_only:
            return incre_modules, None, None
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = self.head_channels[i] * head_block.expansion
            out_channels = self.head_channels[i + 1] * head_block.expansion
            downsamp_module = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(out_channels, momentum=_BN_MOMENTUM), nn.ReLU(inplace=True))
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)
        final_layer = nn.Sequential(nn.Conv2d(in_channels=self.head_channels[3] * head_block.expansion, out_channels=self.num_features, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(self.num_features, momentum=_BN_MOMENTUM), nn.ReLU(inplace=True))
        return incre_modules, downsamp_modules, final_layer

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False), nn.BatchNorm2d(num_channels_cur_layer[i], momentum=_BN_MOMENTUM), nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(nn.Identity())
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False), nn.BatchNorm2d(outchannels, momentum=_BN_MOMENTUM), nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, momentum=_BN_MOMENTUM))
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']
        modules = []
        for i in range(num_modules):
            reset_multi_scale_output = multi_scale_output or i < num_modules - 1
            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.classifier = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def stages(self, x) ->List[torch.Tensor]:
        x = self.layer1(x)
        xl = [t(x) for i, t in enumerate(self.transition1)]
        yl = self.stage2(xl)
        xl = [(t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i]) for i, t in enumerate(self.transition2)]
        yl = self.stage3(xl)
        xl = [(t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i]) for i, t in enumerate(self.transition3)]
        yl = self.stage4(xl)
        return yl

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        yl = self.stages(x)
        y = self.incre_modules[0](yl[0])
        for i, down in enumerate(self.downsamp_modules):
            y = self.incre_modules[i + 1](yl[i + 1]) + down(y)
        y = self.final_layer(y)
        return y

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        return x


class HighResolutionNetFeatures(HighResolutionNet):
    """HighResolutionNet feature extraction

    The design of HRNet makes it easy to grab feature maps, this class provides a simple wrapper to do so.
    It would be more complicated to use the FeatureNet helpers.

    The `feature_location=incre` allows grabbing increased channel count features using part of the
    classification head. If `feature_location=''` the default HRNet features are returned. First stem
    conv is used for stride 2 features.
    """

    def __init__(self, cfg, in_chans=3, num_classes=1000, global_pool='avg', drop_rate=0.0, feature_location='incre', out_indices=(0, 1, 2, 3, 4)):
        assert feature_location in ('incre', '')
        super(HighResolutionNetFeatures, self).__init__(cfg, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool, drop_rate=drop_rate, head=feature_location)
        self.feature_info = FeatureInfo(self.feature_info, out_indices)
        self._out_idx = {i for i in out_indices}

    def forward_features(self, x):
        assert False, 'Not supported'

    def forward(self, x) ->List[torch.tensor]:
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        if 0 in self._out_idx:
            out.append(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.stages(x)
        if self.incre_modules is not None:
            x = [incre(f) for f, incre in zip(x, self.incre_modules)]
        for i, f in enumerate(x):
            if i + 1 in self._out_idx:
                out.append(f)
        return out


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()
        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(192, 48, kernel_size=1, stride=1), BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2))
        self.branch2 = nn.Sequential(BasicConv2d(192, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(192, 64, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1, stride=1), BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1, stride=1), BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1), BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1))
        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()
        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(BasicConv2d(320, 256, kernel_size=1, stride=1), BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1088, 128, kernel_size=1, stride=1), BasicConv2d(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)))
        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1, stride=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1, stride=1), BasicConv2d(256, 288, kernel_size=3, stride=2))
        self.branch2 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1, stride=1), BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1), BasicConv2d(288, 320, kernel_size=3, stride=2))
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, no_relu=False):
        super(Block8, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(2080, 192, kernel_size=1, stride=1), BasicConv2d(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1)), BasicConv2d(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0)))
        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        self.relu = None if no_relu else nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if self.relu is not None:
            out = self.relu(out)
        return out


class InceptionResnetV2(nn.Module):

    def __init__(self, num_classes=1000, in_chans=3, drop_rate=0.0, output_stride=32, global_pool='avg'):
        super(InceptionResnetV2, self).__init__()
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.num_features = 1536
        assert output_stride == 32
        self.conv2d_1a = BasicConv2d(in_chans, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.feature_info = [dict(num_chs=64, reduction=2, module='conv2d_2b')]
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.feature_info += [dict(num_chs=192, reduction=4, module='conv2d_4a')]
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17))
        self.feature_info += [dict(num_chs=320, reduction=8, module='repeat')]
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1))
        self.feature_info += [dict(num_chs=1088, reduction=16, module='repeat_1')]
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.Sequential(Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2))
        self.block8 = Block8(no_relu=True)
        self.conv2d_7b = BasicConv2d(2080, self.num_features, kernel_size=1, stride=1)
        self.feature_info += [dict(num_chs=self.num_features, reduction=32, module='conv2d_7b')]
        self.global_pool, self.classif = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def get_classifier(self):
        return self.classif

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.classif = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classif(x)
        return x


class InceptionA(nn.Module):

    def __init__(self):
        super(InceptionA, self).__init__()
        self.branch0 = BasicConv2d(384, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(384, 96, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionB(nn.Module):

    def __init__(self):
        super(InceptionB, self).__init__()
        self.branch0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1), BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(224, 256, kernel_size=(7, 1), stride=1, padding=(3, 0)))
        self.branch2 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1), BasicConv2d(192, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(224, 224, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(224, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(1024, 128, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionC(nn.Module):

    def __init__(self):
        super(InceptionC, self).__init__()
        self.branch0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)
        self.branch1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(384, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch1_1b = BasicConv2d(384, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(384, 448, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch2_2 = BasicConv2d(448, 512, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3a = BasicConv2d(512, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3b = BasicConv2d(512, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(1536, 256, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)
        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionD(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)
        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)
        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        x = self.conv0(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class InceptionV3(nn.Module):
    """Inception-V3 with no AuxLogits
    FIXME two class defs are redundant, but less screwing around with torchsript fussyness and inconsistent returns
    """

    def __init__(self, num_classes=1000, in_chans=3, drop_rate=0.0, global_pool='avg', aux_logits=False):
        super(InceptionV3, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.aux_logits = aux_logits
        self.Conv2d_1a_3x3 = BasicConv2d(in_chans, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        else:
            self.AuxLogits = None
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.feature_info = [dict(num_chs=64, reduction=2, module='Conv2d_2b_3x3'), dict(num_chs=192, reduction=4, module='Conv2d_4a_3x3'), dict(num_chs=288, reduction=8, module='Mixed_5d'), dict(num_chs=768, reduction=16, module='Mixed_6e'), dict(num_chs=2048, reduction=32, module='Mixed_7c')]
        self.num_features = 2048
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                trunc_normal_(m.weight, std=stddev)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_preaux(self, x):
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.Pool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.Pool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        return x

    def forward_postaux(self, x):
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        return x

    def forward_features(self, x):
        x = self.forward_preaux(x)
        x = self.forward_postaux(x)
        return x

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc(x)
        return x


class InceptionV3Aux(InceptionV3):
    """InceptionV3 with AuxLogits
    """

    def __init__(self, num_classes=1000, in_chans=3, drop_rate=0.0, global_pool='avg', aux_logits=True):
        super(InceptionV3Aux, self).__init__(num_classes, in_chans, drop_rate, global_pool, aux_logits)

    def forward_features(self, x):
        x = self.forward_preaux(x)
        aux = self.AuxLogits(x) if self.training else None
        x = self.forward_postaux(x)
        return x, aux

    def forward(self, x):
        x, aux = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc(x)
        return x, aux


class Mixed3a(nn.Module):

    def __init__(self):
        super(Mixed3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed4a(nn.Module):

    def __init__(self):
        super(Mixed4a, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1))
        self.branch1 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1, stride=1), BasicConv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(64, 96, kernel_size=(3, 3), stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed5a(nn.Module):

    def __init__(self):
        super(Mixed5a, self).__init__()
        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class ReductionA(nn.Module):

    def __init__(self):
        super(ReductionA, self).__init__()
        self.branch0 = BasicConv2d(384, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(BasicConv2d(384, 192, kernel_size=1, stride=1), BasicConv2d(192, 224, kernel_size=3, stride=1, padding=1), BasicConv2d(224, 256, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class ReductionB(nn.Module):

    def __init__(self):
        super(ReductionB, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1), BasicConv2d(192, 192, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(1024, 256, kernel_size=1, stride=1), BasicConv2d(256, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(256, 320, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(320, 320, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class InceptionV4(nn.Module):

    def __init__(self, num_classes=1000, in_chans=3, output_stride=32, drop_rate=0.0, global_pool='avg'):
        super(InceptionV4, self).__init__()
        assert output_stride == 32
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.num_features = 1536
        self.features = nn.Sequential(BasicConv2d(in_chans, 32, kernel_size=3, stride=2), BasicConv2d(32, 32, kernel_size=3, stride=1), BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1), Mixed3a(), Mixed4a(), Mixed5a(), InceptionA(), InceptionA(), InceptionA(), InceptionA(), ReductionA(), InceptionB(), InceptionB(), InceptionB(), InceptionB(), InceptionB(), InceptionB(), InceptionB(), ReductionB(), InceptionC(), InceptionC(), InceptionC())
        self.feature_info = [dict(num_chs=64, reduction=2, module='features.2'), dict(num_chs=160, reduction=4, module='features.3'), dict(num_chs=384, reduction=8, module='features.9'), dict(num_chs=1024, reduction=16, module='features.17'), dict(num_chs=1536, reduction=32, module='features.21')]
        self.global_pool, self.last_linear = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def get_classifier(self):
        return self.last_linear

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.last_linear = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        return self.features(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.last_linear(x)
        return x


class Downsample(nn.Module):

    def __init__(self, channels=None, filt_size=3, stride=2):
        super(Downsample, self).__init__()
        self.channels = channels
        self.filt_size = filt_size
        self.stride = stride
        assert self.filt_size == 3
        filt = torch.tensor([1.0, 2.0, 1.0])
        filt = filt[:, None] * filt[None, :]
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input):
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        return F.conv2d(input_pad, self.filt, stride=self.stride, padding=0, groups=input.shape[1])


class AntiAliasDownsampleLayer(nn.Module):

    def __init__(self, channels: 'int'=0, filt_size: 'int'=3, stride: 'int'=2, no_jit: 'bool'=False):
        super(AntiAliasDownsampleLayer, self).__init__()
        if no_jit:
            self.op = Downsample(channels, filt_size, stride)
        else:
            self.op = DownsampleJIT(channels, filt_size, stride)

    def forward(self, x):
        return self.op(x)


class BlurPool2d(nn.Module):
    """Creates a module that computes blurs and downsample a given feature map.
    See :cite:`zhang2019shiftinvar` for more details.
    Corresponds to the Downsample class, which does blurring and subsampling

    Args:
        channels = Number of input channels
        filt_size (int): binomial filter size for blurring. currently supports 3 (default) and 5.
        stride (int): downsampling filter stride

    Returns:
        torch.Tensor: the transformed tensor.
    """
    filt: 'Dict[str, torch.Tensor]'

    def __init__(self, channels, filt_size=3, stride=2) ->None:
        super(BlurPool2d, self).__init__()
        assert filt_size > 1
        self.channels = channels
        self.filt_size = filt_size
        self.stride = stride
        pad_size = [get_padding(filt_size, stride, dilation=1)] * 4
        self.padding = nn.ReflectionPad2d(pad_size)
        self._coeffs = torch.tensor((np.poly1d((0.5, 0.5)) ** (self.filt_size - 1)).coeffs)
        self.filt = {}

    def _create_filter(self, like: 'torch.Tensor'):
        blur_filter = self._coeffs[:, None] * self._coeffs[None, :]
        return blur_filter[None, None, :, :].repeat(self.channels, 1, 1, 1)

    def _apply(self, fn):
        self.filt = {}
        super(BlurPool2d, self)._apply(fn)

    def forward(self, input_tensor: 'torch.Tensor') ->torch.Tensor:
        C = input_tensor.shape[1]
        blur_filt = self.filt.get(str(input_tensor.device), self._create_filter(input_tensor))
        return F.conv2d(self.padding(input_tensor), blur_filt, stride=self.stride, groups=C)


def drop_block_2d(x, drop_prob: 'float'=0.1, block_size: 'int'=7, gamma_scale: 'float'=1.0, with_noise: 'bool'=False, inplace: 'bool'=False, batchwise: 'bool'=False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. This layer has been tested on a few training
    runs with success, but needs further validation and possibly optimization for lower runtime impact.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / ((W - block_size + 1) * (H - block_size + 1))
    w_i, h_i = torch.meshgrid(torch.arange(W), torch.arange(H))
    valid_block = (w_i >= clipped_block_size // 2) & (w_i < W - (clipped_block_size - 1) // 2) & ((h_i >= clipped_block_size // 2) & (h_i < H - (clipped_block_size - 1) // 2))
    valid_block = torch.reshape(valid_block, (1, 1, H, W))
    if batchwise:
        uniform_noise = torch.rand((1, C, H, W), dtype=x.dtype, device=x.device)
    else:
        uniform_noise = torch.rand_like(x)
    block_mask = 2 - gamma - valid_block + uniform_noise >= 1
    block_mask = -F.max_pool2d(-block_mask, kernel_size=clipped_block_size, stride=1, padding=clipped_block_size // 2)
    if with_noise:
        normal_noise = torch.randn((1, C, H, W), dtype=x.dtype, device=x.device) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(block_mask).add_(normal_noise * (1 - block_mask))
        else:
            x = x * block_mask + normal_noise * (1 - block_mask)
    else:
        normalize_scale = block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-07)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


def drop_block_fast_2d(x: 'torch.Tensor', drop_prob: 'float'=0.1, block_size: 'int'=7, gamma_scale: 'float'=1.0, with_noise: 'bool'=False, inplace: 'bool'=False, batchwise: 'bool'=False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. Simplied from above without concern for valid
    block mask at edges.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / ((W - block_size + 1) * (H - block_size + 1))
    if batchwise:
        block_mask = torch.rand((1, C, H, W), dtype=x.dtype, device=x.device) < gamma
    else:
        block_mask = torch.rand_like(x) < gamma
    block_mask = F.max_pool2d(block_mask, kernel_size=clipped_block_size, stride=1, padding=clipped_block_size // 2)
    if with_noise:
        normal_noise = torch.randn((1, C, H, W), dtype=x.dtype, device=x.device) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(1.0 - block_mask).add_(normal_noise * block_mask)
        else:
            x = x * (1.0 - block_mask) + normal_noise * block_mask
    else:
        block_mask = 1 - block_mask
        normalize_scale = block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-07)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


class DropBlock2d(nn.Module):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf
    """

    def __init__(self, drop_prob=0.1, block_size=7, gamma_scale=1.0, with_noise=False, inplace=False, batchwise=False, fast=True):
        super(DropBlock2d, self).__init__()
        self.drop_prob = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size = block_size
        self.with_noise = with_noise
        self.inplace = inplace
        self.batchwise = batchwise
        self.fast = fast

    def forward(self, x):
        if not self.training or not self.drop_prob:
            return x
        if self.fast:
            return drop_block_fast_2d(x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace, self.batchwise)
        else:
            return drop_block_2d(x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace, self.batchwise)


to_4tuple = _ntuple(4)


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        self.padding = to_4tuple(padding)
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - ih % self.stride[0], 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - iw % self.stride[1], 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = pl, pr, pt, pb
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


class SelectiveKernelAttn(nn.Module):

    def __init__(self, channels, num_paths=2, attn_channels=32, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        """ Selective Kernel Attention Module

        Selective Kernel attention mechanism factored out into its own module.

        """
        super(SelectiveKernelAttn, self).__init__()
        self.num_paths = num_paths
        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.bn = norm_layer(attn_channels)
        self.act = act_layer(inplace=True)
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1, bias=False)

    def forward(self, x):
        assert x.shape[1] == self.num_paths
        x = x.sum(1).mean((2, 3), keepdim=True)
        x = self.fc_reduce(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.fc_select(x)
        B, C, H, W = x.shape
        x = x.view(B, self.num_paths, C // self.num_paths, H, W)
        x = torch.softmax(x, dim=1)
        return x


def _kernel_valid(k):
    if isinstance(k, (list, tuple)):
        for ki in k:
            return _kernel_valid(ki)
    assert k >= 3 and k % 2


class SelectiveKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=None, stride=1, dilation=1, groups=1, attn_reduction=16, min_attn_channels=32, keep_3x3=True, split_input=False, drop_block=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None):
        """ Selective Kernel Convolution Module

        As described in Selective Kernel Networks (https://arxiv.org/abs/1903.06586) with some modifications.

        Largest change is the input split, which divides the input channels across each convolution path, this can
        be viewed as a grouping of sorts, but the output channel counts expand to the module level value. This keeps
        the parameter count from ballooning when the convolutions themselves don't have groups, but still provides
        a noteworthy increase in performance over similar param count models without this attention layer. -Ross W

        Args:
            in_channels (int):  module input (feature) channel count
            out_channels (int):  module output (feature) channel count
            kernel_size (int, list): kernel size for each convolution branch
            stride (int): stride for convolutions
            dilation (int): dilation for module as a whole, impacts dilation of each branch
            groups (int): number of groups for each branch
            attn_reduction (int, float): reduction factor for attention features
            min_attn_channels (int): minimum attention feature channels
            keep_3x3 (bool): keep all branch convolution kernels as 3x3, changing larger kernels for dilations
            split_input (bool): split input channels evenly across each convolution branch, keeps param count lower,
                can be viewed as grouping by path, output expands to module out_channels count
            drop_block (nn.Module): drop block module
            act_layer (nn.Module): activation layer to use
            norm_layer (nn.Module): batchnorm/norm layer to use
        """
        super(SelectiveKernelConv, self).__init__()
        kernel_size = kernel_size or [3, 5]
        _kernel_valid(kernel_size)
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * 2
        if keep_3x3:
            dilation = [(dilation * (k - 1) // 2) for k in kernel_size]
            kernel_size = [3] * len(kernel_size)
        else:
            dilation = [dilation] * len(kernel_size)
        self.num_paths = len(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.split_input = split_input
        if self.split_input:
            assert in_channels % self.num_paths == 0
            in_channels = in_channels // self.num_paths
        groups = min(out_channels, groups)
        conv_kwargs = dict(stride=stride, groups=groups, drop_block=drop_block, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer)
        self.paths = nn.ModuleList([ConvBnAct(in_channels, out_channels, kernel_size=k, dilation=d, **conv_kwargs) for k, d in zip(kernel_size, dilation)])
        attn_channels = max(int(out_channels / attn_reduction), min_attn_channels)
        self.attn = SelectiveKernelAttn(out_channels, self.num_paths, attn_channels)
        self.drop_block = drop_block

    def forward(self, x):
        if self.split_input:
            x_split = torch.split(x, self.in_channels // self.num_paths, 1)
            x_paths = [op(x_split[i]) for i, op in enumerate(self.paths)]
        else:
            x_paths = [op(x) for op in self.paths]
        x = torch.stack(x_paths, dim=1)
        x_attn = self.attn(x)
        x = x * x_attn
        x = torch.sum(x, dim=1)
        return x


class SeparableConvBnAct(nn.Module):
    """ Separable Conv w/ trailing Norm and Activation
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding='', bias=False, channel_multiplier=1.0, pw_kernel_size=1, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, apply_act=True, drop_block=None):
        super(SeparableConvBnAct, self).__init__()
        self.conv_dw = create_conv2d(in_channels, int(in_channels * channel_multiplier), kernel_size, stride=stride, dilation=dilation, padding=padding, depthwise=True)
        self.conv_pw = create_conv2d(int(in_channels * channel_multiplier), out_channels, pw_kernel_size, padding=padding, bias=bias)
        norm_act_layer = convert_norm_act(norm_layer, act_layer)
        self.bn = norm_act_layer(out_channels, apply_act=apply_act, drop_block=drop_block)

    @property
    def in_channels(self):
        return self.conv_dw.in_channels

    @property
    def out_channels(self):
        return self.conv_pw.out_channels

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class SpaceToDepth(nn.Module):

    def __init__(self, block_size=4):
        super().__init__()
        assert block_size == 4
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(N, C * self.bs ** 2, H // self.bs, W // self.bs)
        return x


class SpaceToDepthModule(nn.Module):

    def __init__(self, no_jit=False):
        super().__init__()
        if not no_jit:
            self.op = SpaceToDepthJit()
        else:
            self.op = SpaceToDepth()

    def forward(self, x):
        return self.op(x)


class DepthToSpace(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // self.bs ** 2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(N, C // self.bs ** 2, H * self.bs, W * self.bs)
        return x


class RadixSoftmax(nn.Module):

    def __init__(self, radix, cardinality):
        super(RadixSoftmax, self).__init__()
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


class SplitAttnConv2d(nn.Module):
    """Split-Attention Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, radix=2, reduction_factor=4, act_layer=nn.ReLU, norm_layer=None, drop_block=None, **kwargs):
        super(SplitAttnConv2d, self).__init__()
        self.radix = radix
        self.drop_block = drop_block
        mid_chs = out_channels * radix
        attn_chs = max(in_channels * radix // reduction_factor, 32)
        self.conv = nn.Conv2d(in_channels, mid_chs, kernel_size, stride, padding, dilation, groups=groups * radix, bias=bias, **kwargs)
        self.bn0 = norm_layer(mid_chs) if norm_layer is not None else None
        self.act0 = act_layer(inplace=True)
        self.fc1 = nn.Conv2d(out_channels, attn_chs, 1, groups=groups)
        self.bn1 = norm_layer(attn_chs) if norm_layer is not None else None
        self.act1 = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(attn_chs, mid_chs, 1, groups=groups)
        self.rsoftmax = RadixSoftmax(radix, groups)

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.fc1.out_channels

    def forward(self, x):
        x = self.conv(x)
        if self.bn0 is not None:
            x = self.bn0(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act0(x)
        B, RC, H, W = x.shape
        if self.radix > 1:
            x = x.reshape((B, self.radix, RC // self.radix, H, W))
            x_gap = x.sum(dim=1)
        else:
            x_gap = x
        x_gap = F.adaptive_avg_pool2d(x_gap, 1)
        x_gap = self.fc1(x_gap)
        if self.bn1 is not None:
            x_gap = self.bn1(x_gap)
        x_gap = self.act1(x_gap)
        x_attn = self.fc2(x_gap)
        x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        if self.radix > 1:
            out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        else:
            out = x * x_attn
        return out.contiguous()


class SplitBatchNorm2d(torch.nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, num_splits=2):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        assert num_splits > 1, 'Should have at least one aux BN layer (num_splits at least 2)'
        self.num_splits = num_splits
        self.aux_bn = nn.ModuleList([nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_splits - 1)])

    def forward(self, input: 'torch.Tensor'):
        if self.training:
            split_size = input.shape[0] // self.num_splits
            assert input.shape[0] == split_size * self.num_splits, 'batch size must be evenly divisible by num_splits'
            split_input = input.split(split_size)
            x = [super().forward(split_input[0])]
            for i, a in enumerate(self.aux_bn):
                x.append(a(split_input[i + 1]))
            return torch.cat(x, dim=0)
        else:
            return super().forward(input)


class StdConv2d(nn.Conv2d):
    """Conv2d with Weight Standardization. Used for BiT ResNet-V2 models.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """

    def __init__(self, in_channel, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1, bias=False, eps=1e-05):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(in_channel, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.eps = eps

    def get_weight(self):
        std, mean = torch.std_mean(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
        weight = (self.weight - mean) / (std + self.eps)
        return weight

    def forward(self, x):
        x = F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class StdConv2dSame(nn.Conv2d):
    """Conv2d with Weight Standardization. TF compatible SAME padding. Used for ViT Hybrid model.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """

    def __init__(self, in_channel, out_channels, kernel_size, stride=1, padding='SAME', dilation=1, groups=1, bias=False, eps=1e-05):
        padding, is_dynamic = get_padding_value(padding, kernel_size, stride=stride, dilation=dilation)
        super().__init__(in_channel, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.same_pad = is_dynamic
        self.eps = eps

    def get_weight(self):
        std, mean = torch.std_mean(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
        weight = (self.weight - mean) / (std + self.eps)
        return weight

    def forward(self, x):
        if self.same_pad:
            x = pad_same(x, self.kernel_size, self.stride, self.dilation)
        x = F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class ScaledStdConv2d(nn.Conv2d):
    """Conv2d layer with Scaled Weight Standardization.

    Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets` -
        https://arxiv.org/abs/2101.08692

    NOTE: the operations used in this impl differ slightly from the DeepMind Haiku impl. The impact is minor.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1, bias=True, gamma=1.0, eps=1e-05, use_layernorm=False):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.eps = eps ** 2 if use_layernorm else eps
        self.use_layernorm = use_layernorm

    def get_weight(self):
        if self.use_layernorm:
            weight = self.scale * F.layer_norm(self.weight, self.weight.shape[1:], eps=self.eps)
        else:
            std, mean = torch.std_mean(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
            weight = self.scale * (self.weight - mean) / (std + self.eps)
        return self.gain * weight

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)


class ScaledStdConv2dSame(nn.Conv2d):
    """Conv2d layer with Scaled Weight Standardization and Tensorflow-like SAME padding support

    Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets` -
        https://arxiv.org/abs/2101.08692

    NOTE: the operations used in this impl differ slightly from the DeepMind Haiku impl. The impact is minor.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='SAME', dilation=1, groups=1, bias=True, gamma=1.0, eps=1e-05, use_layernorm=False):
        padding, is_dynamic = get_padding_value(padding, kernel_size, stride=stride, dilation=dilation)
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.same_pad = is_dynamic
        self.eps = eps ** 2 if use_layernorm else eps
        self.use_layernorm = use_layernorm

    def get_weight(self):
        if self.use_layernorm:
            weight = self.scale * F.layer_norm(self.weight, self.weight.shape[1:], eps=self.eps)
        else:
            std, mean = torch.std_mean(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
            weight = self.scale * (self.weight - mean) / (std + self.eps)
        return self.gain * weight

    def forward(self, x):
        if self.same_pad:
            x = pad_same(x, self.kernel_size, self.stride, self.dilation)
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)


class TestTimePoolHead(nn.Module):

    def __init__(self, base, original_pool=7):
        super(TestTimePoolHead, self).__init__()
        self.base = base
        self.original_pool = original_pool
        base_fc = self.base.get_classifier()
        if isinstance(base_fc, nn.Conv2d):
            self.fc = base_fc
        else:
            self.fc = nn.Conv2d(self.base.num_features, self.base.num_classes, kernel_size=1, bias=True)
            self.fc.weight.data.copy_(base_fc.weight.data.view(self.fc.weight.size()))
            self.fc.bias.data.copy_(base_fc.bias.data.view(self.fc.bias.size()))
        self.base.reset_classifier(0)

    def forward(self, x):
        x = self.base.forward_features(x)
        x = F.avg_pool2d(x, kernel_size=self.original_pool, stride=1)
        x = self.fc(x)
        x = adaptive_avgmax_pool2d(x, 1)
        return x.view(x.size(0), -1)


class MobileNetV3(nn.Module):
    """ MobiletNet-V3

    Based on my EfficientNet implementation and building blocks, this model utilizes the MobileNet-v3 specific
    'efficient head', where global pooling is done before the head convolution without a final batch-norm
    layer before the classifier.

    Paper: https://arxiv.org/abs/1905.02244
    """

    def __init__(self, block_args, num_classes=1000, in_chans=3, stem_size=16, num_features=1280, head_bias=True, channel_multiplier=1.0, pad_type='', act_layer=nn.ReLU, drop_rate=0.0, drop_path_rate=0.0, se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None, global_pool='avg'):
        super(MobileNetV3, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate
        stem_size = round_channels(stem_size, channel_multiplier)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        builder = EfficientNetBuilder(channel_multiplier, 8, None, 32, pad_type, act_layer, se_kwargs, norm_layer, norm_kwargs, drop_path_rate, verbose=_DEBUG)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = builder.features
        head_chs = builder.in_chs
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        num_pooled_chs = head_chs * self.global_pool.feat_mult()
        self.conv_head = create_conv2d(num_pooled_chs, self.num_features, 1, padding=pad_type, bias=head_bias)
        self.act2 = act_layer(inplace=True)
        self.classifier = Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        efficientnet_init_weights(self)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1, self.act1]
        layers.extend(self.blocks)
        layers.extend([self.global_pool, self.conv_head, self.act2])
        layers.extend([nn.Flatten(), nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.classifier = Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if not self.global_pool.is_identity():
            x = x.flatten(1)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)


class MobileNetV3Features(nn.Module):
    """ MobileNetV3 Feature Extractor

    A work-in-progress feature extraction module for MobileNet-V3 to use as a backbone for segmentation
    and object detection models.
    """

    def __init__(self, block_args, out_indices=(0, 1, 2, 3, 4), feature_location='bottleneck', in_chans=3, stem_size=16, channel_multiplier=1.0, output_stride=32, pad_type='', act_layer=nn.ReLU, drop_rate=0.0, drop_path_rate=0.0, se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(MobileNetV3Features, self).__init__()
        norm_kwargs = norm_kwargs or {}
        self.drop_rate = drop_rate
        stem_size = round_channels(stem_size, channel_multiplier)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        builder = EfficientNetBuilder(channel_multiplier, 8, None, output_stride, pad_type, act_layer, se_kwargs, norm_layer, norm_kwargs, drop_path_rate, feature_location=feature_location, verbose=_DEBUG)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = FeatureInfo(builder.features, out_indices)
        self._stage_out_idx = {v['stage']: i for i, v in enumerate(self.feature_info) if i in out_indices}
        efficientnet_init_weights(self)
        self.feature_hooks = None
        if feature_location != 'bottleneck':
            hooks = self.feature_info.get_dicts(keys=('module', 'hook_type'))
            self.feature_hooks = FeatureHooks(hooks, self.named_modules())

    def forward(self, x) ->List[torch.Tensor]:
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.feature_hooks is None:
            features = []
            if 0 in self._stage_out_idx:
                features.append(x)
            for i, b in enumerate(self.blocks):
                x = b(x)
                if i + 1 in self._stage_out_idx:
                    features.append(x)
            return features
        else:
            self.blocks(x)
            out = self.feature_hooks.get_output(x.device)
            return list(out.values())


class ActConvBn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=''):
        super(ActConvBn, self).__init__()
        self.act = nn.ReLU()
        self.conv = create_conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.act(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class BranchSeparables(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, stem_cell=False, padding=''):
        super(BranchSeparables, self).__init__()
        middle_channels = out_channels if stem_cell else in_channels
        self.act_1 = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, middle_channels, kernel_size, stride=stride, padding=padding)
        self.bn_sep_1 = nn.BatchNorm2d(middle_channels, eps=0.001)
        self.act_2 = nn.ReLU()
        self.separable_2 = SeparableConv2d(middle_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.act_1(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.act_2(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class CellBase(nn.Module):

    def cell_forward(self, x_left, x_right):
        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_left = self.comb_iter_3_left(x_comb_iter_2)
        x_comb_iter_3_right = self.comb_iter_3_right(x_right)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_left)
        if self.comb_iter_4_right is not None:
            x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        else:
            x_comb_iter_4_right = x_right
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class CellStem0(CellBase):

    def __init__(self, in_chs_left, out_chs_left, in_chs_right, out_chs_right, pad_type=''):
        super(CellStem0, self).__init__()
        self.conv_1x1 = ActConvBn(in_chs_right, out_chs_right, kernel_size=1, padding=pad_type)
        self.comb_iter_0_left = BranchSeparables(in_chs_left, out_chs_left, kernel_size=5, stride=2, stem_cell=True, padding=pad_type)
        self.comb_iter_0_right = nn.Sequential(OrderedDict([('max_pool', create_pool2d('max', 3, stride=2, padding=pad_type)), ('conv', create_conv2d(in_chs_left, out_chs_left, kernel_size=1, padding=pad_type)), ('bn', nn.BatchNorm2d(out_chs_left, eps=0.001))]))
        self.comb_iter_1_left = BranchSeparables(out_chs_right, out_chs_right, kernel_size=7, stride=2, padding=pad_type)
        self.comb_iter_1_right = create_pool2d('max', 3, stride=2, padding=pad_type)
        self.comb_iter_2_left = BranchSeparables(out_chs_right, out_chs_right, kernel_size=5, stride=2, padding=pad_type)
        self.comb_iter_2_right = BranchSeparables(out_chs_right, out_chs_right, kernel_size=3, stride=2, padding=pad_type)
        self.comb_iter_3_left = BranchSeparables(out_chs_right, out_chs_right, kernel_size=3, padding=pad_type)
        self.comb_iter_3_right = create_pool2d('max', 3, stride=2, padding=pad_type)
        self.comb_iter_4_left = BranchSeparables(in_chs_right, out_chs_right, kernel_size=3, stride=2, stem_cell=True, padding=pad_type)
        self.comb_iter_4_right = ActConvBn(out_chs_right, out_chs_right, kernel_size=1, stride=2, padding=pad_type)

    def forward(self, x_left):
        x_right = self.conv_1x1(x_left)
        x_out = self.cell_forward(x_left, x_right)
        return x_out


class CellStem1(nn.Module):

    def __init__(self, stem_size, num_channels, pad_type=''):
        super(CellStem1, self).__init__()
        self.num_channels = num_channels
        self.stem_size = stem_size
        self.conv_1x1 = ActConvBn(2 * self.num_channels, self.num_channels, 1, stride=1)
        self.act = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(self.stem_size, self.num_channels // 2, 1, stride=1, bias=False))
        self.path_2 = nn.Sequential()
        self.path_2.add_module('pad', nn.ZeroPad2d((-1, 1, -1, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(self.stem_size, self.num_channels // 2, 1, stride=1, bias=False))
        self.final_path_bn = nn.BatchNorm2d(self.num_channels, eps=0.001, momentum=0.1)
        self.comb_iter_0_left = BranchSeparables(self.num_channels, self.num_channels, 5, 2, pad_type)
        self.comb_iter_0_right = BranchSeparables(self.num_channels, self.num_channels, 7, 2, pad_type)
        self.comb_iter_1_left = create_pool2d('max', 3, 2, padding=pad_type)
        self.comb_iter_1_right = BranchSeparables(self.num_channels, self.num_channels, 7, 2, pad_type)
        self.comb_iter_2_left = create_pool2d('avg', 3, 2, count_include_pad=False, padding=pad_type)
        self.comb_iter_2_right = BranchSeparables(self.num_channels, self.num_channels, 5, 2, pad_type)
        self.comb_iter_3_right = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)
        self.comb_iter_4_left = BranchSeparables(self.num_channels, self.num_channels, 3, 1, pad_type)
        self.comb_iter_4_right = create_pool2d('max', 3, 2, padding=pad_type)

    def forward(self, x_conv0, x_stem_0):
        x_left = self.conv_1x1(x_stem_0)
        x_relu = self.act(x_conv0)
        x_path1 = self.path_1(x_relu)
        x_path2 = self.path_2(x_relu)
        x_right = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_right)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_left)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_left)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class FirstCell(nn.Module):

    def __init__(self, in_chs_left, out_chs_left, in_chs_right, out_chs_right, pad_type=''):
        super(FirstCell, self).__init__()
        self.conv_1x1 = ActConvBn(in_chs_right, out_chs_right, 1, stride=1)
        self.act = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(in_chs_left, out_chs_left, 1, stride=1, bias=False))
        self.path_2 = nn.Sequential()
        self.path_2.add_module('pad', nn.ZeroPad2d((-1, 1, -1, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(in_chs_left, out_chs_left, 1, stride=1, bias=False))
        self.final_path_bn = nn.BatchNorm2d(out_chs_left * 2, eps=0.001, momentum=0.1)
        self.comb_iter_0_left = BranchSeparables(out_chs_right, out_chs_right, 5, 1, pad_type)
        self.comb_iter_0_right = BranchSeparables(out_chs_right, out_chs_right, 3, 1, pad_type)
        self.comb_iter_1_left = BranchSeparables(out_chs_right, out_chs_right, 5, 1, pad_type)
        self.comb_iter_1_right = BranchSeparables(out_chs_right, out_chs_right, 3, 1, pad_type)
        self.comb_iter_2_left = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)
        self.comb_iter_3_left = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)
        self.comb_iter_3_right = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)
        self.comb_iter_4_left = BranchSeparables(out_chs_right, out_chs_right, 3, 1, pad_type)

    def forward(self, x, x_prev):
        x_relu = self.act(x_prev)
        x_path1 = self.path_1(x_relu)
        x_path2 = self.path_2(x_relu)
        x_left = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left
        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NormalCell(nn.Module):

    def __init__(self, in_chs_left, out_chs_left, in_chs_right, out_chs_right, pad_type=''):
        super(NormalCell, self).__init__()
        self.conv_prev_1x1 = ActConvBn(in_chs_left, out_chs_left, 1, stride=1, padding=pad_type)
        self.conv_1x1 = ActConvBn(in_chs_right, out_chs_right, 1, stride=1, padding=pad_type)
        self.comb_iter_0_left = BranchSeparables(out_chs_right, out_chs_right, 5, 1, pad_type)
        self.comb_iter_0_right = BranchSeparables(out_chs_left, out_chs_left, 3, 1, pad_type)
        self.comb_iter_1_left = BranchSeparables(out_chs_left, out_chs_left, 5, 1, pad_type)
        self.comb_iter_1_right = BranchSeparables(out_chs_left, out_chs_left, 3, 1, pad_type)
        self.comb_iter_2_left = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)
        self.comb_iter_3_left = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)
        self.comb_iter_3_right = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)
        self.comb_iter_4_left = BranchSeparables(out_chs_right, out_chs_right, 3, 1, pad_type)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left
        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class ReductionCell0(nn.Module):

    def __init__(self, in_chs_left, out_chs_left, in_chs_right, out_chs_right, pad_type=''):
        super(ReductionCell0, self).__init__()
        self.conv_prev_1x1 = ActConvBn(in_chs_left, out_chs_left, 1, stride=1, padding=pad_type)
        self.conv_1x1 = ActConvBn(in_chs_right, out_chs_right, 1, stride=1, padding=pad_type)
        self.comb_iter_0_left = BranchSeparables(out_chs_right, out_chs_right, 5, 2, pad_type)
        self.comb_iter_0_right = BranchSeparables(out_chs_right, out_chs_right, 7, 2, pad_type)
        self.comb_iter_1_left = create_pool2d('max', 3, 2, padding=pad_type)
        self.comb_iter_1_right = BranchSeparables(out_chs_right, out_chs_right, 7, 2, pad_type)
        self.comb_iter_2_left = create_pool2d('avg', 3, 2, count_include_pad=False, padding=pad_type)
        self.comb_iter_2_right = BranchSeparables(out_chs_right, out_chs_right, 5, 2, pad_type)
        self.comb_iter_3_right = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)
        self.comb_iter_4_left = BranchSeparables(out_chs_right, out_chs_right, 3, 1, pad_type)
        self.comb_iter_4_right = create_pool2d('max', 3, 2, padding=pad_type)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class ReductionCell1(nn.Module):

    def __init__(self, in_chs_left, out_chs_left, in_chs_right, out_chs_right, pad_type=''):
        super(ReductionCell1, self).__init__()
        self.conv_prev_1x1 = ActConvBn(in_chs_left, out_chs_left, 1, stride=1, padding=pad_type)
        self.conv_1x1 = ActConvBn(in_chs_right, out_chs_right, 1, stride=1, padding=pad_type)
        self.comb_iter_0_left = BranchSeparables(out_chs_right, out_chs_right, 5, 2, pad_type)
        self.comb_iter_0_right = BranchSeparables(out_chs_right, out_chs_right, 7, 2, pad_type)
        self.comb_iter_1_left = create_pool2d('max', 3, 2, padding=pad_type)
        self.comb_iter_1_right = BranchSeparables(out_chs_right, out_chs_right, 7, 2, pad_type)
        self.comb_iter_2_left = create_pool2d('avg', 3, 2, count_include_pad=False, padding=pad_type)
        self.comb_iter_2_right = BranchSeparables(out_chs_right, out_chs_right, 5, 2, pad_type)
        self.comb_iter_3_right = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)
        self.comb_iter_4_left = BranchSeparables(out_chs_right, out_chs_right, 3, 1, pad_type)
        self.comb_iter_4_right = create_pool2d('max', 3, 2, padding=pad_type)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NASNetALarge(nn.Module):
    """NASNetALarge (6 @ 4032) """

    def __init__(self, num_classes=1000, in_chans=3, stem_size=96, channel_multiplier=2, num_features=4032, output_stride=32, drop_rate=0.0, global_pool='avg', pad_type='same'):
        super(NASNetALarge, self).__init__()
        self.num_classes = num_classes
        self.stem_size = stem_size
        self.num_features = num_features
        self.channel_multiplier = channel_multiplier
        self.drop_rate = drop_rate
        assert output_stride == 32
        channels = self.num_features // 24
        self.conv0 = ConvBnAct(in_channels=in_chans, out_channels=self.stem_size, kernel_size=3, padding=0, stride=2, norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.1), apply_act=False)
        self.cell_stem_0 = CellStem0(self.stem_size, num_channels=channels // channel_multiplier ** 2, pad_type=pad_type)
        self.cell_stem_1 = CellStem1(self.stem_size, num_channels=channels // channel_multiplier, pad_type=pad_type)
        self.cell_0 = FirstCell(in_chs_left=channels, out_chs_left=channels // 2, in_chs_right=2 * channels, out_chs_right=channels, pad_type=pad_type)
        self.cell_1 = NormalCell(in_chs_left=2 * channels, out_chs_left=channels, in_chs_right=6 * channels, out_chs_right=channels, pad_type=pad_type)
        self.cell_2 = NormalCell(in_chs_left=6 * channels, out_chs_left=channels, in_chs_right=6 * channels, out_chs_right=channels, pad_type=pad_type)
        self.cell_3 = NormalCell(in_chs_left=6 * channels, out_chs_left=channels, in_chs_right=6 * channels, out_chs_right=channels, pad_type=pad_type)
        self.cell_4 = NormalCell(in_chs_left=6 * channels, out_chs_left=channels, in_chs_right=6 * channels, out_chs_right=channels, pad_type=pad_type)
        self.cell_5 = NormalCell(in_chs_left=6 * channels, out_chs_left=channels, in_chs_right=6 * channels, out_chs_right=channels, pad_type=pad_type)
        self.reduction_cell_0 = ReductionCell0(in_chs_left=6 * channels, out_chs_left=2 * channels, in_chs_right=6 * channels, out_chs_right=2 * channels, pad_type=pad_type)
        self.cell_6 = FirstCell(in_chs_left=6 * channels, out_chs_left=channels, in_chs_right=8 * channels, out_chs_right=2 * channels, pad_type=pad_type)
        self.cell_7 = NormalCell(in_chs_left=8 * channels, out_chs_left=2 * channels, in_chs_right=12 * channels, out_chs_right=2 * channels, pad_type=pad_type)
        self.cell_8 = NormalCell(in_chs_left=12 * channels, out_chs_left=2 * channels, in_chs_right=12 * channels, out_chs_right=2 * channels, pad_type=pad_type)
        self.cell_9 = NormalCell(in_chs_left=12 * channels, out_chs_left=2 * channels, in_chs_right=12 * channels, out_chs_right=2 * channels, pad_type=pad_type)
        self.cell_10 = NormalCell(in_chs_left=12 * channels, out_chs_left=2 * channels, in_chs_right=12 * channels, out_chs_right=2 * channels, pad_type=pad_type)
        self.cell_11 = NormalCell(in_chs_left=12 * channels, out_chs_left=2 * channels, in_chs_right=12 * channels, out_chs_right=2 * channels, pad_type=pad_type)
        self.reduction_cell_1 = ReductionCell1(in_chs_left=12 * channels, out_chs_left=4 * channels, in_chs_right=12 * channels, out_chs_right=4 * channels, pad_type=pad_type)
        self.cell_12 = FirstCell(in_chs_left=12 * channels, out_chs_left=2 * channels, in_chs_right=16 * channels, out_chs_right=4 * channels, pad_type=pad_type)
        self.cell_13 = NormalCell(in_chs_left=16 * channels, out_chs_left=4 * channels, in_chs_right=24 * channels, out_chs_right=4 * channels, pad_type=pad_type)
        self.cell_14 = NormalCell(in_chs_left=24 * channels, out_chs_left=4 * channels, in_chs_right=24 * channels, out_chs_right=4 * channels, pad_type=pad_type)
        self.cell_15 = NormalCell(in_chs_left=24 * channels, out_chs_left=4 * channels, in_chs_right=24 * channels, out_chs_right=4 * channels, pad_type=pad_type)
        self.cell_16 = NormalCell(in_chs_left=24 * channels, out_chs_left=4 * channels, in_chs_right=24 * channels, out_chs_right=4 * channels, pad_type=pad_type)
        self.cell_17 = NormalCell(in_chs_left=24 * channels, out_chs_left=4 * channels, in_chs_right=24 * channels, out_chs_right=4 * channels, pad_type=pad_type)
        self.act = nn.ReLU(inplace=True)
        self.feature_info = [dict(num_chs=96, reduction=2, module='conv0'), dict(num_chs=168, reduction=4, module='cell_stem_1.conv_1x1.act'), dict(num_chs=1008, reduction=8, module='reduction_cell_0.conv_1x1.act'), dict(num_chs=2016, reduction=16, module='reduction_cell_1.conv_1x1.act'), dict(num_chs=4032, reduction=32, module='act')]
        self.global_pool, self.last_linear = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def get_classifier(self):
        return self.last_linear

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.last_linear = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x_conv0 = self.conv0(x)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.cell_3(x_cell_2, x_cell_1)
        x_cell_4 = self.cell_4(x_cell_3, x_cell_2)
        x_cell_5 = self.cell_5(x_cell_4, x_cell_3)
        x_reduction_cell_0 = self.reduction_cell_0(x_cell_5, x_cell_4)
        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_4)
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.cell_9(x_cell_8, x_cell_7)
        x_cell_10 = self.cell_10(x_cell_9, x_cell_8)
        x_cell_11 = self.cell_11(x_cell_10, x_cell_9)
        x_reduction_cell_1 = self.reduction_cell_1(x_cell_11, x_cell_10)
        x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_10)
        x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.cell_15(x_cell_14, x_cell_13)
        x_cell_16 = self.cell_16(x_cell_15, x_cell_14)
        x_cell_17 = self.cell_17(x_cell_16, x_cell_15)
        x = self.act(x_cell_17)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate > 0:
            x = F.dropout(x, self.drop_rate, training=self.training)
        x = self.last_linear(x)
        return x


def gelu(x: 'torch.Tensor', inplace: 'bool'=False) ->torch.Tensor:
    return F.gelu(x)


def tanh(x, inplace: 'bool'=False):
    return x.tanh_() if inplace else x.tanh()


def hard_mish_me(x, inplace: 'bool'=False):
    return HardMishJitAutoFn.apply(x)


def hard_sigmoid_me(x, inplace: 'bool'=False):
    return HardSigmoidJitAutoFn.apply(x)


def hard_swish_me(x, inplace=False):
    return HardSwishJitAutoFn.apply(x)


def mish_me(x, inplace=False):
    return MishJitAutoFn.apply(x)


def swish_me(x, inplace=False):
    return SwishJitAutoFn.apply(x)


def get_act_fn(name='relu'):
    """ Activation Function Factory
    Fetching activation fns by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if not name:
        return None
    if not (is_no_jit() or is_exportable() or is_scriptable()):
        if name in _ACT_FN_ME:
            return _ACT_FN_ME[name]
    if is_exportable() and name in ('silu', 'swish'):
        return swish
    if not (is_no_jit() or is_exportable()):
        if name in _ACT_FN_JIT:
            return _ACT_FN_JIT[name]
    return _ACT_FN_DEFAULT[name]


class GammaAct(nn.Module):

    def __init__(self, act_type='relu', gamma: 'float'=1.0, inplace=False):
        super().__init__()
        self.act_fn = get_act_fn(act_type)
        self.gamma = gamma
        self.inplace = inplace

    def forward(self, x):
        return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)


class NormFreeBlock(nn.Module):
    """Normalization-Free pre-activation block.
    """

    def __init__(self, in_chs, out_chs=None, stride=1, dilation=1, first_dilation=None, alpha=1.0, beta=1.0, bottle_ratio=0.25, group_size=None, ch_div=1, reg=True, extra_conv=False, skipinit=False, attn_layer=None, attn_gain=2.0, act_layer=None, conv_layer=None, drop_path_rate=0.0):
        super().__init__()
        first_dilation = first_dilation or dilation
        out_chs = out_chs or in_chs
        mid_chs = make_divisible(in_chs * bottle_ratio if reg else out_chs * bottle_ratio, ch_div)
        groups = 1 if not group_size else mid_chs // group_size
        if group_size and group_size % ch_div == 0:
            mid_chs = group_size * groups
        self.alpha = alpha
        self.beta = beta
        self.attn_gain = attn_gain
        if in_chs != out_chs or stride != 1 or dilation != first_dilation:
            self.downsample = DownsampleAvg(in_chs, out_chs, stride=stride, dilation=dilation, first_dilation=first_dilation, conv_layer=conv_layer)
        else:
            self.downsample = None
        self.act1 = act_layer()
        self.conv1 = conv_layer(in_chs, mid_chs, 1)
        self.act2 = act_layer(inplace=True)
        self.conv2 = conv_layer(mid_chs, mid_chs, 3, stride=stride, dilation=first_dilation, groups=groups)
        if extra_conv:
            self.act2b = act_layer(inplace=True)
            self.conv2b = conv_layer(mid_chs, mid_chs, 3, stride=1, dilation=dilation, groups=groups)
        else:
            self.act2b = None
            self.conv2b = None
        if reg and attn_layer is not None:
            self.attn = attn_layer(mid_chs)
        else:
            self.attn = None
        self.act3 = act_layer()
        self.conv3 = conv_layer(mid_chs, out_chs, 1)
        if not reg and attn_layer is not None:
            self.attn_last = attn_layer(out_chs)
        else:
            self.attn_last = None
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.skipinit_gain = nn.Parameter(torch.tensor(0.0)) if skipinit else None

    def forward(self, x):
        out = self.act1(x) * self.beta
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(out)
        out = self.conv1(out)
        out = self.conv2(self.act2(out))
        if self.conv2b is not None:
            out = self.conv2b(self.act2b(out))
        if self.attn is not None:
            out = self.attn_gain * self.attn(out)
        out = self.conv3(self.act3(out))
        if self.attn_last is not None:
            out = self.attn_gain * self.attn_last(out)
        out = self.drop_path(out)
        if self.skipinit_gain is not None:
            out.mul_(self.skipinit_gain)
        out = out * self.alpha + shortcut
        return out


_nonlin_gamma = dict(identity=1.0, celu=1.270926833152771, elu=1.2716004848480225, gelu=1.7015043497085571, leaky_relu=1.70590341091156, log_sigmoid=1.9193484783172607, log_softmax=1.0002083778381348, relu=1.7139588594436646, relu6=1.7131484746932983, selu=1.0008515119552612, sigmoid=4.803835391998291, silu=1.7881293296813965, softsign=2.338853120803833, softplus=1.9203323125839233, tanh=1.5939117670059204)


def act_with_gamma(act_type, gamma: 'float'=1.0):

    def _create(inplace=False):
        return GammaAct(act_type, gamma=gamma, inplace=inplace)
    return _create


class NormFreeNet(nn.Module):
    """ Normalization-Free Network

    As described in :
    `Characterizing signal propagation to close the performance gap in unnormalized ResNets`
        - https://arxiv.org/abs/2101.08692
    and
    `High-Performance Large-Scale Image Recognition Without Normalization` - https://arxiv.org/abs/2102.06171

    This model aims to cover both the NFRegNet-Bx models as detailed in the paper's code snippets and
    the (preact) ResNet models described earlier in the paper.

    There are a few differences:
        * channels are rounded to be divisible by 8 by default (keep tensor core kernels happy),
            this changes channel dim and param counts slightly from the paper models
        * activation correcting gamma constants are moved into the ScaledStdConv as it has less performance
            impact in PyTorch when done with the weight scaling there. This likely wasn't a concern in the JAX impl.
        * a config option `gamma_in_act` can be enabled to not apply gamma in StdConv as described above, but
            apply it in each activation. This is slightly slower, numerically different, but matches official impl.
        * skipinit is disabled by default, it seems to have a rather drastic impact on GPU memory use and throughput
            for what it is/does. Approx 8-10% throughput loss.
    """

    def __init__(self, cfg: 'NfCfg', num_classes=1000, in_chans=3, global_pool='avg', output_stride=32, drop_rate=0.0, drop_path_rate=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        assert cfg.act_layer in _nonlin_gamma, f'Please add non-linearity constants for activation ({cfg.act_layer}).'
        conv_layer = ScaledStdConv2dSame if cfg.same_padding else ScaledStdConv2d
        if cfg.gamma_in_act:
            act_layer = act_with_gamma(cfg.act_layer, gamma=_nonlin_gamma[cfg.act_layer])
            conv_layer = partial(conv_layer, eps=0.0001)
        else:
            act_layer = get_act_layer(cfg.act_layer)
            conv_layer = partial(conv_layer, gamma=_nonlin_gamma[cfg.act_layer])
        attn_layer = partial(get_attn(cfg.attn_layer), **cfg.attn_kwargs) if cfg.attn_layer else None
        stem_chs = make_divisible((cfg.stem_chs or cfg.channels[0]) * cfg.width_factor, cfg.ch_div)
        self.stem, stem_stride, stem_feat = create_stem(in_chans, stem_chs, cfg.stem_type, conv_layer=conv_layer, act_layer=act_layer)
        self.feature_info = [stem_feat]
        drop_path_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(cfg.depths)).split(cfg.depths)]
        prev_chs = stem_chs
        net_stride = stem_stride
        dilation = 1
        expected_var = 1.0
        stages = []
        for stage_idx, stage_depth in enumerate(cfg.depths):
            stride = 1 if stage_idx == 0 and stem_stride > 2 else 2
            if net_stride >= output_stride and stride > 1:
                dilation *= stride
                stride = 1
            net_stride *= stride
            first_dilation = 1 if dilation in (1, 2) else 2
            blocks = []
            for block_idx in range(cfg.depths[stage_idx]):
                first_block = block_idx == 0 and stage_idx == 0
                out_chs = make_divisible(cfg.channels[stage_idx] * cfg.width_factor, cfg.ch_div)
                blocks += [NormFreeBlock(in_chs=prev_chs, out_chs=out_chs, alpha=cfg.alpha, beta=1.0 / expected_var ** 0.5, stride=stride if block_idx == 0 else 1, dilation=dilation, first_dilation=first_dilation, group_size=cfg.group_size, bottle_ratio=1.0 if cfg.reg and first_block else cfg.bottle_ratio, ch_div=cfg.ch_div, reg=cfg.reg, extra_conv=cfg.extra_conv, skipinit=cfg.skipinit, attn_layer=attn_layer, attn_gain=cfg.attn_gain, act_layer=act_layer, conv_layer=conv_layer, drop_path_rate=drop_path_rates[stage_idx][block_idx])]
                if block_idx == 0:
                    expected_var = 1.0
                expected_var += cfg.alpha ** 2
                first_dilation = dilation
                prev_chs = out_chs
            self.feature_info += [dict(num_chs=prev_chs, reduction=net_stride, module=f'stages.{stage_idx}')]
            stages += [nn.Sequential(*blocks)]
        self.stages = nn.Sequential(*stages)
        if cfg.num_features:
            self.num_features = make_divisible(cfg.width_factor * cfg.num_features, cfg.ch_div)
            self.final_conv = conv_layer(prev_chs, self.num_features, 1)
            self.feature_info[-1] = dict(num_chs=self.num_features, reduction=net_stride, module=f'final_conv')
        else:
            self.num_features = prev_chs
            self.final_conv = nn.Identity()
        self.final_act = act_layer(inplace=cfg.num_features > 0)
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)
        for n, m in self.named_modules():
            if 'fc' in n and isinstance(m, nn.Linear):
                if cfg.zero_init_fc:
                    nn.init.zeros_(m.weight)
                else:
                    nn.init.normal_(m.weight, 0.0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.final_conv(x)
        x = self.final_act(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class FactorizedReduction(nn.Module):

    def __init__(self, in_channels, out_channels, padding=''):
        super(FactorizedReduction, self).__init__()
        self.act = nn.ReLU()
        self.path_1 = nn.Sequential(OrderedDict([('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False)), ('conv', create_conv2d(in_channels, out_channels // 2, kernel_size=1, padding=padding))]))
        self.path_2 = nn.Sequential(OrderedDict([('pad', nn.ZeroPad2d((-1, 1, -1, 1))), ('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False)), ('conv', create_conv2d(in_channels, out_channels // 2, kernel_size=1, padding=padding))]))
        self.final_path_bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.act(x)
        x_path1 = self.path_1(x)
        x_path2 = self.path_2(x)
        out = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        return out


class Cell(CellBase):

    def __init__(self, in_chs_left, out_chs_left, in_chs_right, out_chs_right, pad_type='', is_reduction=False, match_prev_layer_dims=False):
        super(Cell, self).__init__()
        stride = 2 if is_reduction else 1
        self.match_prev_layer_dimensions = match_prev_layer_dims
        if match_prev_layer_dims:
            self.conv_prev_1x1 = FactorizedReduction(in_chs_left, out_chs_left, padding=pad_type)
        else:
            self.conv_prev_1x1 = ActConvBn(in_chs_left, out_chs_left, kernel_size=1, padding=pad_type)
        self.conv_1x1 = ActConvBn(in_chs_right, out_chs_right, kernel_size=1, padding=pad_type)
        self.comb_iter_0_left = BranchSeparables(out_chs_left, out_chs_left, kernel_size=5, stride=stride, padding=pad_type)
        self.comb_iter_0_right = create_pool2d('max', 3, stride=stride, padding=pad_type)
        self.comb_iter_1_left = BranchSeparables(out_chs_right, out_chs_right, kernel_size=7, stride=stride, padding=pad_type)
        self.comb_iter_1_right = create_pool2d('max', 3, stride=stride, padding=pad_type)
        self.comb_iter_2_left = BranchSeparables(out_chs_right, out_chs_right, kernel_size=5, stride=stride, padding=pad_type)
        self.comb_iter_2_right = BranchSeparables(out_chs_right, out_chs_right, kernel_size=3, stride=stride, padding=pad_type)
        self.comb_iter_3_left = BranchSeparables(out_chs_right, out_chs_right, kernel_size=3)
        self.comb_iter_3_right = create_pool2d('max', 3, stride=stride, padding=pad_type)
        self.comb_iter_4_left = BranchSeparables(out_chs_left, out_chs_left, kernel_size=3, stride=stride, padding=pad_type)
        if is_reduction:
            self.comb_iter_4_right = ActConvBn(out_chs_right, out_chs_right, kernel_size=1, stride=stride, padding=pad_type)
        else:
            self.comb_iter_4_right = None

    def forward(self, x_left, x_right):
        x_left = self.conv_prev_1x1(x_left)
        x_right = self.conv_1x1(x_right)
        x_out = self.cell_forward(x_left, x_right)
        return x_out


class PNASNet5Large(nn.Module):

    def __init__(self, num_classes=1000, in_chans=3, output_stride=32, drop_rate=0.0, global_pool='avg', pad_type=''):
        super(PNASNet5Large, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.num_features = 4320
        assert output_stride == 32
        self.conv_0 = ConvBnAct(in_chans, 96, kernel_size=3, stride=2, padding=0, norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.1), apply_act=False)
        self.cell_stem_0 = CellStem0(in_chs_left=96, out_chs_left=54, in_chs_right=96, out_chs_right=54, pad_type=pad_type)
        self.cell_stem_1 = Cell(in_chs_left=96, out_chs_left=108, in_chs_right=270, out_chs_right=108, pad_type=pad_type, match_prev_layer_dims=True, is_reduction=True)
        self.cell_0 = Cell(in_chs_left=270, out_chs_left=216, in_chs_right=540, out_chs_right=216, pad_type=pad_type, match_prev_layer_dims=True)
        self.cell_1 = Cell(in_chs_left=540, out_chs_left=216, in_chs_right=1080, out_chs_right=216, pad_type=pad_type)
        self.cell_2 = Cell(in_chs_left=1080, out_chs_left=216, in_chs_right=1080, out_chs_right=216, pad_type=pad_type)
        self.cell_3 = Cell(in_chs_left=1080, out_chs_left=216, in_chs_right=1080, out_chs_right=216, pad_type=pad_type)
        self.cell_4 = Cell(in_chs_left=1080, out_chs_left=432, in_chs_right=1080, out_chs_right=432, pad_type=pad_type, is_reduction=True)
        self.cell_5 = Cell(in_chs_left=1080, out_chs_left=432, in_chs_right=2160, out_chs_right=432, pad_type=pad_type, match_prev_layer_dims=True)
        self.cell_6 = Cell(in_chs_left=2160, out_chs_left=432, in_chs_right=2160, out_chs_right=432, pad_type=pad_type)
        self.cell_7 = Cell(in_chs_left=2160, out_chs_left=432, in_chs_right=2160, out_chs_right=432, pad_type=pad_type)
        self.cell_8 = Cell(in_chs_left=2160, out_chs_left=864, in_chs_right=2160, out_chs_right=864, pad_type=pad_type, is_reduction=True)
        self.cell_9 = Cell(in_chs_left=2160, out_chs_left=864, in_chs_right=4320, out_chs_right=864, pad_type=pad_type, match_prev_layer_dims=True)
        self.cell_10 = Cell(in_chs_left=4320, out_chs_left=864, in_chs_right=4320, out_chs_right=864, pad_type=pad_type)
        self.cell_11 = Cell(in_chs_left=4320, out_chs_left=864, in_chs_right=4320, out_chs_right=864, pad_type=pad_type)
        self.act = nn.ReLU()
        self.feature_info = [dict(num_chs=96, reduction=2, module='conv_0'), dict(num_chs=270, reduction=4, module='cell_stem_1.conv_1x1.act'), dict(num_chs=1080, reduction=8, module='cell_4.conv_1x1.act'), dict(num_chs=2160, reduction=16, module='cell_8.conv_1x1.act'), dict(num_chs=4320, reduction=32, module='act')]
        self.global_pool, self.last_linear = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def get_classifier(self):
        return self.last_linear

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.last_linear = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x_conv_0 = self.conv_0(x)
        x_stem_0 = self.cell_stem_0(x_conv_0)
        x_stem_1 = self.cell_stem_1(x_conv_0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_0, x_stem_1)
        x_cell_1 = self.cell_1(x_stem_1, x_cell_0)
        x_cell_2 = self.cell_2(x_cell_0, x_cell_1)
        x_cell_3 = self.cell_3(x_cell_1, x_cell_2)
        x_cell_4 = self.cell_4(x_cell_2, x_cell_3)
        x_cell_5 = self.cell_5(x_cell_3, x_cell_4)
        x_cell_6 = self.cell_6(x_cell_4, x_cell_5)
        x_cell_7 = self.cell_7(x_cell_5, x_cell_6)
        x_cell_8 = self.cell_8(x_cell_6, x_cell_7)
        x_cell_9 = self.cell_9(x_cell_7, x_cell_8)
        x_cell_10 = self.cell_10(x_cell_8, x_cell_9)
        x_cell_11 = self.cell_11(x_cell_9, x_cell_10)
        x = self.act(x_cell_11)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate > 0:
            x = F.dropout(x, self.drop_rate, training=self.training)
        x = self.last_linear(x)
        return x


def downsample_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = first_dilation or dilation if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)
    return nn.Sequential(*[nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False), norm_layer(out_channels)])


class RegStage(nn.Module):
    """Stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, in_chs, out_chs, stride, dilation, depth, bottle_ratio, group_width, block_fn=Bottleneck, se_ratio=0.0, drop_path_rates=None, drop_block=None):
        super(RegStage, self).__init__()
        block_kwargs = {}
        first_dilation = 1 if dilation in (1, 2) else 2
        for i in range(depth):
            block_stride = stride if i == 0 else 1
            block_in_chs = in_chs if i == 0 else out_chs
            block_dilation = first_dilation if i == 0 else dilation
            if drop_path_rates is not None and drop_path_rates[i] > 0.0:
                drop_path = DropPath(drop_path_rates[i])
            else:
                drop_path = None
            if block_in_chs != out_chs or block_stride != 1:
                proj_block = downsample_conv(block_in_chs, out_chs, 1, block_stride, block_dilation)
            else:
                proj_block = None
            name = 'b{}'.format(i + 1)
            self.add_module(name, block_fn(block_in_chs, out_chs, block_stride, block_dilation, bottle_ratio, group_width, se_ratio, downsample=proj_block, drop_block=drop_block, drop_path=drop_path, **block_kwargs))

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_widths_groups_comp(widths, bottle_ratios, groups):
    """Adjusts the compatibility of widths and groups."""
    bottleneck_widths = [int(w * b) for w, b in zip(widths, bottle_ratios)]
    groups = [min(g, w_bot) for g, w_bot in zip(groups, bottleneck_widths)]
    bottleneck_widths = [quantize_float(w_bot, g) for w_bot, g in zip(bottleneck_widths, groups)]
    widths = [int(w_bot / b) for w_bot, b in zip(bottleneck_widths, bottle_ratios)]
    return widths, groups


def generate_regnet(width_slope, width_initial, width_mult, depth, q=8):
    """Generates per block widths from RegNet parameters."""
    assert width_slope >= 0 and width_initial > 0 and width_mult > 1 and width_initial % q == 0
    widths_cont = np.arange(depth) * width_slope + width_initial
    width_exps = np.round(np.log(widths_cont / width_initial) / np.log(width_mult))
    widths = width_initial * np.power(width_mult, width_exps)
    widths = np.round(np.divide(widths, q)) * q
    num_stages, max_stage = len(np.unique(widths)), width_exps.max() + 1
    widths, widths_cont = widths.astype(int).tolist(), widths_cont.tolist()
    return widths, num_stages, max_stage, widths_cont


class RegNet(nn.Module):
    """RegNet model.

    Paper: https://arxiv.org/abs/2003.13678
    Original Impl: https://github.com/facebookresearch/pycls/blob/master/pycls/models/regnet.py
    """

    def __init__(self, cfg, in_chans=3, num_classes=1000, output_stride=32, global_pool='avg', drop_rate=0.0, drop_path_rate=0.0, zero_init_last_bn=True):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        assert output_stride in (8, 16, 32)
        stem_width = cfg['stem_width']
        self.stem = ConvBnAct(in_chans, stem_width, 3, stride=2)
        self.feature_info = [dict(num_chs=stem_width, reduction=2, module='stem')]
        prev_width = stem_width
        curr_stride = 2
        stage_params = self._get_stage_params(cfg, output_stride=output_stride, drop_path_rate=drop_path_rate)
        se_ratio = cfg['se_ratio']
        for i, stage_args in enumerate(stage_params):
            stage_name = 's{}'.format(i + 1)
            self.add_module(stage_name, RegStage(prev_width, **stage_args, se_ratio=se_ratio))
            prev_width = stage_args['out_chs']
            curr_stride *= stage_args['stride']
            self.feature_info += [dict(num_chs=prev_width, reduction=curr_stride, module=stage_name)]
        self.num_features = prev_width
        self.head = ClassifierHead(in_chs=prev_width, num_classes=num_classes, pool_type=global_pool, drop_rate=drop_rate)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, 'zero_init_last_bn'):
                    m.zero_init_last_bn()

    def _get_stage_params(self, cfg, default_stride=2, output_stride=32, drop_path_rate=0.0):
        w_a, w_0, w_m, d = cfg['wa'], cfg['w0'], cfg['wm'], cfg['depth']
        widths, num_stages, _, _ = generate_regnet(w_a, w_0, w_m, d)
        stage_widths, stage_depths = np.unique(widths, return_counts=True)
        stage_groups = [cfg['group_w'] for _ in range(num_stages)]
        stage_bottle_ratios = [cfg['bottle_ratio'] for _ in range(num_stages)]
        stage_strides = []
        stage_dilations = []
        net_stride = 2
        dilation = 1
        for _ in range(num_stages):
            if net_stride >= output_stride:
                dilation *= default_stride
                stride = 1
            else:
                stride = default_stride
                net_stride *= stride
            stage_strides.append(stride)
            stage_dilations.append(dilation)
        stage_dpr = np.split(np.linspace(0, drop_path_rate, d), np.cumsum(stage_depths[:-1]))
        stage_widths, stage_groups = adjust_widths_groups_comp(stage_widths, stage_bottle_ratios, stage_groups)
        param_names = ['out_chs', 'stride', 'dilation', 'depth', 'bottle_ratio', 'group_width', 'drop_path_rates']
        stage_params = [dict(zip(param_names, params)) for params in zip(stage_widths, stage_strides, stage_dilations, stage_depths, stage_bottle_ratios, stage_groups, stage_dpr)]
        return stage_params

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        for block in list(self.children())[:-1]:
            x = block(x)
        return x

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class Bottle2neck(nn.Module):
    """ Res2Net/Res2NeXT Bottleneck
    Adapted from https://github.com/gasvn/Res2Net/blob/master/res2net.py
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=26, scale=4, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=None, attn_layer=None, **_):
        super(Bottle2neck, self).__init__()
        self.scale = scale
        self.is_first = stride > 1 or downsample is not None
        self.num_scales = max(1, scale - 1)
        width = int(math.floor(planes * (base_width / 64.0))) * cardinality
        self.width = width
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width * scale)
        convs = []
        bns = []
        for i in range(self.num_scales):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False))
            bns.append(norm_layer(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        if self.is_first:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        else:
            self.pool = None
        self.conv3 = nn.Conv2d(width * scale, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.se = attn_layer(outplanes) if attn_layer is not None else None
        self.relu = act_layer(inplace=True)
        self.downsample = downsample

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        spo = []
        sp = spx[0]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if i == 0 or self.is_first:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = conv(sp)
            sp = bn(sp)
            sp = self.relu(sp)
            spo.append(sp)
        if self.scale > 1:
            if self.pool is not None:
                spo.append(self.pool(spx[-1]))
            else:
                spo.append(spx[-1])
        out = torch.cat(spo, 1)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.se is not None:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNestBottleneck(nn.Module):
    """ResNet Bottleneck
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, radix=1, cardinality=1, base_width=64, avd=False, avd_first=False, is_first=False, reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(ResNestBottleneck, self).__init__()
        assert reduce_first == 1
        assert attn_layer is None
        assert aa_layer is None
        assert drop_path is None
        group_width = int(planes * (base_width / 64.0)) * cardinality
        first_dilation = first_dilation or dilation
        if avd and (stride > 1 or is_first):
            avd_stride = stride
            stride = 1
        else:
            avd_stride = 0
        self.radix = radix
        self.drop_block = drop_block
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.act1 = act_layer(inplace=True)
        self.avd_first = nn.AvgPool2d(3, avd_stride, padding=1) if avd_stride > 0 and avd_first else None
        if self.radix >= 1:
            self.conv2 = SplitAttnConv2d(group_width, group_width, kernel_size=3, stride=stride, padding=first_dilation, dilation=first_dilation, groups=cardinality, radix=radix, norm_layer=norm_layer, drop_block=drop_block)
            self.bn2 = None
            self.act2 = None
        else:
            self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)
            self.act2 = act_layer(inplace=True)
        self.avd_last = nn.AvgPool2d(3, avd_stride, padding=1) if avd_stride > 0 and not avd_first else None
        self.conv3 = nn.Conv2d(group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.act3 = act_layer(inplace=True)
        self.downsample = downsample

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        if self.drop_block is not None:
            out = self.drop_block(out)
        out = self.act1(out)
        if self.avd_first is not None:
            out = self.avd_first(out)
        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)
            if self.drop_block is not None:
                out = self.drop_block(out)
            out = self.act2(out)
        if self.avd_last is not None:
            out = self.avd_last(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.drop_block is not None:
            out = self.drop_block(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.act3(out)
        return out


def downsample_avg(in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
    return nn.Sequential(*[pool, nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False), norm_layer(out_channels)])


def drop_blocks(drop_block_rate=0.0):
    return [None, None, DropBlock2d(drop_block_rate, 5, 0.25) if drop_block_rate else None, DropBlock2d(drop_block_rate, 3, 1.0) if drop_block_rate else None]


def make_blocks(block_fn, channels, block_repeats, inplanes, reduce_first=1, output_stride=32, down_kernel_size=1, avg_down=False, drop_block_rate=0.0, drop_path_rate=0.0, **kwargs):
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f'layer{stage_idx + 1}'
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride
        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(in_channels=inplanes, out_channels=planes * block_fn.expansion, kernel_size=down_kernel_size, stride=stride, dilation=dilation, first_dilation=prev_dilation, norm_layer=kwargs.get('norm_layer'))
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)
        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)
            blocks.append(block_fn(inplanes, planes, stride, downsample, first_dilation=prev_dilation, drop_path=DropPath(block_dpr) if block_dpr > 0.0 else None, **block_kwargs))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1
        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))
    return stages, feature_info


class ResNet(nn.Module):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net

    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    This ResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon ResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.

    ResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample

    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled

    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled

    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockGl, BottleneckGl.
    layers : list of int
        Numbers of layers in each block
    num_classes : int, default 1000
        Number of classification classes.
    in_chans : int, default 3
        Number of input (color) channels.
    cardinality : int, default 1
        Number of convolution groups for 3x3 conv in Bottleneck.
    base_width : int, default 64
        Factor determining bottleneck channels. `planes * base_width / 64 * cardinality`
    stem_width : int, default 64
        Number of channels in stem convolutions
    stem_type : str, default ''
        The type of stem:
          * '', default - a single 7x7 conv with a width of stem_width
          * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
          * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
    block_reduce_first: int, default 1
        Reduction factor for first convolution output width of residual blocks,
        1 for all archs except senets, where 2
    down_kernel_size: int, default 1
        Kernel size of residual block downsampling path, 1x1 for most archs, 3x3 for senets
    avg_down : bool, default False
        Whether to use average pooling for projection skip connection between stages/downsample.
    output_stride : int, default 32
        Set the output stride of the network, 32, 16, or 8. Typically used in segmentation.
    act_layer : nn.Module, activation layer
    norm_layer : nn.Module, normalization layer
    aa_layer : nn.Module, anti-aliasing layer
    drop_rate : float, default 0.
        Dropout probability before classifier, for training
    global_pool : str, default 'avg'
        Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'
    """

    def __init__(self, block, layers, num_classes=1000, in_chans=3, cardinality=1, base_width=64, stem_width=64, stem_type='', output_stride=32, block_reduce_first=1, down_kernel_size=1, avg_down=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None, drop_rate=0.0, drop_path_rate=0.0, drop_block_rate=0.0, global_pool='avg', zero_init_last_bn=True, block_args=None):
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        super(ResNet, self).__init__()
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = stem_width, stem_width
            if 'tiered' in stem_type:
                stem_chs = 3 * (stem_width // 4), stem_width
            self.conv1 = nn.Sequential(*[nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False), norm_layer(stem_chs[0]), act_layer(inplace=True), nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False), norm_layer(stem_chs[1]), act_layer(inplace=True), nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False)])
        else:
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]
        if aa_layer is not None:
            self.maxpool = nn.Sequential(*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1), aa_layer(channels=inplanes, stride=2)])
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        channels = [64, 128, 256, 512]
        stage_modules, stage_feature_info = make_blocks(block, channels, layers, inplanes, cardinality=cardinality, base_width=base_width, output_stride=output_stride, reduce_first=block_reduce_first, avg_down=avg_down, down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer, drop_block_rate=drop_block_rate, drop_path_rate=drop_path_rate, **block_args)
        for stage in stage_modules:
            self.add_module(*stage)
        self.feature_info.extend(stage_feature_info)
        self.num_features = 512 * block.expansion
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, 'zero_init_last_bn'):
                    m.zero_init_last_bn()

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x


def make_div(v, divisor=8):
    min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.

    Follows the implementation of "Identity Mappings in Deep Residual Networks":
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

    Except it puts the stride on 3x3 conv when available.
    """

    def __init__(self, in_chs, out_chs=None, bottle_ratio=0.25, stride=1, dilation=1, first_dilation=None, groups=1, act_layer=None, conv_layer=None, norm_layer=None, proj_layer=None, drop_path_rate=0.0):
        super().__init__()
        first_dilation = first_dilation or dilation
        conv_layer = conv_layer or StdConv2d
        norm_layer = norm_layer or partial(GroupNormAct, num_groups=32)
        out_chs = out_chs or in_chs
        mid_chs = make_div(out_chs * bottle_ratio)
        if proj_layer is not None:
            self.downsample = proj_layer(in_chs, out_chs, stride=stride, dilation=dilation, first_dilation=first_dilation, preact=True, conv_layer=conv_layer, norm_layer=norm_layer)
        else:
            self.downsample = None
        self.norm1 = norm_layer(in_chs)
        self.conv1 = conv_layer(in_chs, mid_chs, 1)
        self.norm2 = norm_layer(mid_chs)
        self.conv2 = conv_layer(mid_chs, mid_chs, 3, stride=stride, dilation=first_dilation, groups=groups)
        self.norm3 = norm_layer(mid_chs)
        self.conv3 = conv_layer(mid_chs, out_chs, 1)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        x_preact = self.norm1(x)
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x_preact)
        x = self.conv1(x_preact)
        x = self.conv2(self.norm2(x))
        x = self.conv3(self.norm3(x))
        x = self.drop_path(x)
        return x + shortcut


class DownsampleConv(nn.Module):

    def __init__(self, in_chs, out_chs, stride=1, dilation=1, first_dilation=None, preact=True, conv_layer=None, norm_layer=None):
        super(DownsampleConv, self).__init__()
        self.conv = conv_layer(in_chs, out_chs, 1, stride=stride)
        self.norm = nn.Identity() if preact else norm_layer(out_chs, apply_act=False)

    def forward(self, x):
        return self.norm(self.conv(x))


class ResNetStage(nn.Module):
    """ResNet Stage."""

    def __init__(self, in_chs, out_chs, stride, dilation, depth, bottle_ratio=0.25, groups=1, avg_down=False, block_dpr=None, block_fn=PreActBottleneck, act_layer=None, conv_layer=None, norm_layer=None, **block_kwargs):
        super(ResNetStage, self).__init__()
        first_dilation = 1 if dilation in (1, 2) else 2
        layer_kwargs = dict(act_layer=act_layer, conv_layer=conv_layer, norm_layer=norm_layer)
        proj_layer = DownsampleAvg if avg_down else DownsampleConv
        prev_chs = in_chs
        self.blocks = nn.Sequential()
        for block_idx in range(depth):
            drop_path_rate = block_dpr[block_idx] if block_dpr else 0.0
            stride = stride if block_idx == 0 else 1
            self.blocks.add_module(str(block_idx), block_fn(prev_chs, out_chs, stride=stride, dilation=dilation, bottle_ratio=bottle_ratio, groups=groups, first_dilation=first_dilation, proj_layer=proj_layer, drop_path_rate=drop_path_rate, **layer_kwargs, **block_kwargs))
            prev_chs = out_chs
            first_dilation = dilation
            proj_layer = None

    def forward(self, x):
        x = self.blocks(x)
        return x


def tf2th(conv_weights):
    """Possibly convert HWIO to OIHW."""
    if conv_weights.ndim == 4:
        conv_weights = conv_weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(conv_weights)


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode.
    """

    def __init__(self, layers, channels=(256, 512, 1024, 2048), num_classes=1000, in_chans=3, global_pool='avg', output_stride=32, width_factor=1, stem_chs=64, stem_type='', avg_down=False, preact=True, act_layer=nn.ReLU, conv_layer=StdConv2d, norm_layer=partial(GroupNormAct, num_groups=32), drop_rate=0.0, drop_path_rate=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        wf = width_factor
        self.feature_info = []
        stem_chs = make_div(stem_chs * wf)
        self.stem = create_stem(in_chans, stem_chs, stem_type, preact, conv_layer=conv_layer, norm_layer=norm_layer)
        stem_feat = ('stem.conv3' if 'deep' in stem_type else 'stem.conv') if preact else 'stem.norm'
        self.feature_info.append(dict(num_chs=stem_chs, reduction=2, module=stem_feat))
        prev_chs = stem_chs
        curr_stride = 4
        dilation = 1
        block_dprs = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(layers)).split(layers)]
        block_fn = PreActBottleneck if preact else Bottleneck
        self.stages = nn.Sequential()
        for stage_idx, (d, c, bdpr) in enumerate(zip(layers, channels, block_dprs)):
            out_chs = make_div(c * wf)
            stride = 1 if stage_idx == 0 else 2
            if curr_stride >= output_stride:
                dilation *= stride
                stride = 1
            stage = ResNetStage(prev_chs, out_chs, stride=stride, dilation=dilation, depth=d, avg_down=avg_down, act_layer=act_layer, conv_layer=conv_layer, norm_layer=norm_layer, block_dpr=bdpr, block_fn=block_fn)
            prev_chs = out_chs
            curr_stride *= stride
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{stage_idx}')]
            self.stages.add_module(str(stage_idx), stage)
        self.num_features = prev_chs
        self.norm = norm_layer(self.num_features) if preact else nn.Identity()
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate, use_conv=True)
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear) or '.fc' in n and isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate, use_conv=True)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        if not self.head.global_pool.is_identity():
            x = x.flatten(1)
        return x

    def load_pretrained(self, checkpoint_path, prefix='resnet/'):
        import numpy as np
        weights = np.load(checkpoint_path)
        with torch.no_grad():
            stem_conv_w = tf2th(weights[f'{prefix}root_block/standardized_conv2d/kernel'])
            if self.stem.conv.weight.shape[1] == 1:
                self.stem.conv.weight.copy_(stem_conv_w.sum(dim=1, keepdim=True))
            else:
                self.stem.conv.weight.copy_(stem_conv_w)
            self.norm.weight.copy_(tf2th(weights[f'{prefix}group_norm/gamma']))
            self.norm.bias.copy_(tf2th(weights[f'{prefix}group_norm/beta']))
            self.head.fc.weight.copy_(tf2th(weights[f'{prefix}head/conv2d/kernel']))
            self.head.fc.bias.copy_(tf2th(weights[f'{prefix}head/conv2d/bias']))
            for i, (sname, stage) in enumerate(self.stages.named_children()):
                for j, (bname, block) in enumerate(stage.blocks.named_children()):
                    convname = 'standardized_conv2d'
                    block_prefix = f'{prefix}block{i + 1}/unit{j + 1:02d}/'
                    block.conv1.weight.copy_(tf2th(weights[f'{block_prefix}a/{convname}/kernel']))
                    block.conv2.weight.copy_(tf2th(weights[f'{block_prefix}b/{convname}/kernel']))
                    block.conv3.weight.copy_(tf2th(weights[f'{block_prefix}c/{convname}/kernel']))
                    block.norm1.weight.copy_(tf2th(weights[f'{block_prefix}a/group_norm/gamma']))
                    block.norm2.weight.copy_(tf2th(weights[f'{block_prefix}b/group_norm/gamma']))
                    block.norm3.weight.copy_(tf2th(weights[f'{block_prefix}c/group_norm/gamma']))
                    block.norm1.bias.copy_(tf2th(weights[f'{block_prefix}a/group_norm/beta']))
                    block.norm2.bias.copy_(tf2th(weights[f'{block_prefix}b/group_norm/beta']))
                    block.norm3.bias.copy_(tf2th(weights[f'{block_prefix}c/group_norm/beta']))
                    if block.downsample is not None:
                        w = weights[f'{block_prefix}a/proj/{convname}/kernel']
                        block.downsample.conv.weight.copy_(tf2th(w))


class SEWithNorm(nn.Module):

    def __init__(self, channels, se_ratio=1 / 12.0, act_layer=nn.ReLU, divisor=1, reduction_channels=None, gate_layer='sigmoid'):
        super(SEWithNorm, self).__init__()
        reduction_channels = reduction_channels or make_divisible(int(channels * se_ratio), divisor=divisor)
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(reduction_channels)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, bias=True)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.bn(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


class LinearBottleneck(nn.Module):

    def __init__(self, in_chs, out_chs, stride, exp_ratio=1.0, se_ratio=0.0, ch_div=1, act_layer='swish', dw_act_layer='relu6', drop_path=None):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_chs <= out_chs
        self.in_channels = in_chs
        self.out_channels = out_chs
        if exp_ratio != 1.0:
            dw_chs = make_divisible(round(in_chs * exp_ratio), divisor=ch_div)
            self.conv_exp = ConvBnAct(in_chs, dw_chs, act_layer=act_layer)
        else:
            dw_chs = in_chs
            self.conv_exp = None
        self.conv_dw = ConvBnAct(dw_chs, dw_chs, 3, stride=stride, groups=dw_chs, apply_act=False)
        self.se = SEWithNorm(dw_chs, se_ratio=se_ratio, divisor=ch_div) if se_ratio > 0.0 else None
        self.act_dw = create_act_layer(dw_act_layer)
        self.conv_pwl = ConvBnAct(dw_chs, out_chs, 1, apply_act=False)
        self.drop_path = drop_path

    def feat_channels(self, exp=False):
        return self.conv_dw.out_channels if exp else self.out_channels

    def forward(self, x):
        shortcut = x
        if self.conv_exp is not None:
            x = self.conv_exp(x)
        x = self.conv_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.act_dw(x)
        x = self.conv_pwl(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.use_shortcut:
            x[:, 0:self.in_channels] += shortcut
        return x


def _block_cfg(width_mult=1.0, depth_mult=1.0, initial_chs=16, final_chs=180, se_ratio=0.0, ch_div=1):
    layers = [1, 2, 2, 3, 3, 5]
    strides = [1, 2, 2, 2, 1, 2]
    layers = [ceil(element * depth_mult) for element in layers]
    strides = sum([([element] + [1] * (layers[idx] - 1)) for idx, element in enumerate(strides)], [])
    exp_ratios = [1] * layers[0] + [6] * sum(layers[1:])
    depth = sum(layers[:]) * 3
    base_chs = initial_chs / width_mult if width_mult < 1.0 else initial_chs
    out_chs_list = []
    for i in range(depth // 3):
        out_chs_list.append(make_divisible(round(base_chs * width_mult), divisor=ch_div))
        base_chs += final_chs / (depth // 3 * 1.0)
    se_ratios = [0.0] * (layers[0] + layers[1]) + [se_ratio] * sum(layers[2:])
    return list(zip(out_chs_list, exp_ratios, strides, se_ratios))


def _build_blocks(block_cfg, prev_chs, width_mult, ch_div=1, act_layer='swish', dw_act_layer='relu6', drop_path_rate=0.0):
    feat_chs = [prev_chs]
    feature_info = []
    curr_stride = 2
    features = []
    num_blocks = len(block_cfg)
    for block_idx, (chs, exp_ratio, stride, se_ratio) in enumerate(block_cfg):
        if stride > 1:
            fname = 'stem' if block_idx == 0 else f'features.{block_idx - 1}'
            feature_info += [dict(num_chs=feat_chs[-1], reduction=curr_stride, module=fname)]
            curr_stride *= stride
        block_dpr = drop_path_rate * block_idx / (num_blocks - 1)
        drop_path = DropPath(block_dpr) if block_dpr > 0.0 else None
        features.append(LinearBottleneck(in_chs=prev_chs, out_chs=chs, exp_ratio=exp_ratio, stride=stride, se_ratio=se_ratio, ch_div=ch_div, act_layer=act_layer, dw_act_layer=dw_act_layer, drop_path=drop_path))
        prev_chs = chs
        feat_chs += [features[-1].feat_channels()]
    pen_chs = make_divisible(1280 * width_mult, divisor=ch_div)
    feature_info += [dict(num_chs=feat_chs[-1], reduction=curr_stride, module=f'features.{len(features) - 1}')]
    features.append(ConvBnAct(prev_chs, pen_chs, act_layer=act_layer))
    return features, feature_info


class ReXNetV1(nn.Module):

    def __init__(self, in_chans=3, num_classes=1000, global_pool='avg', output_stride=32, initial_chs=16, final_chs=180, width_mult=1.0, depth_mult=1.0, se_ratio=1 / 12.0, ch_div=1, act_layer='swish', dw_act_layer='relu6', drop_rate=0.2, drop_path_rate=0.0):
        super(ReXNetV1, self).__init__()
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        assert output_stride == 32
        stem_base_chs = 32 / width_mult if width_mult < 1.0 else 32
        stem_chs = make_divisible(round(stem_base_chs * width_mult), divisor=ch_div)
        self.stem = ConvBnAct(in_chans, stem_chs, 3, stride=2, act_layer=act_layer)
        block_cfg = _block_cfg(width_mult, depth_mult, initial_chs, final_chs, se_ratio, ch_div)
        features, self.feature_info = _build_blocks(block_cfg, stem_chs, width_mult, ch_div, act_layer, dw_act_layer, drop_path_rate)
        self.num_features = features[-1].out_channels
        self.features = nn.Sequential(*features)
        self.head = ClassifierHead(self.num_features, num_classes, global_pool, drop_rate)
        efficientnet_init_weights(self)

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.features(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class SequentialList(nn.Sequential):

    def __init__(self, *args):
        super(SequentialList, self).__init__(*args)

    @torch.jit._overload_method
    def forward(self, x):
        pass

    @torch.jit._overload_method
    def forward(self, x):
        pass

    def forward(self, x) ->List[torch.Tensor]:
        for module in self:
            x = module(x)
        return x


class SelectSeq(nn.Module):

    def __init__(self, mode='index', index=0):
        super(SelectSeq, self).__init__()
        self.mode = mode
        self.index = index

    @torch.jit._overload_method
    def forward(self, x):
        pass

    @torch.jit._overload_method
    def forward(self, x):
        pass

    def forward(self, x) ->torch.Tensor:
        if self.mode == 'index':
            return x[self.index]
        else:
            return torch.cat(x, dim=1)


def conv_bn(in_chs, out_chs, k=3, stride=1, padding=None, dilation=1):
    if padding is None:
        padding = (stride - 1 + dilation * (k - 1)) // 2
    return nn.Sequential(nn.Conv2d(in_chs, out_chs, k, stride, padding=padding, dilation=dilation, bias=False), nn.BatchNorm2d(out_chs), nn.ReLU(inplace=True))


class SelecSLSBlock(nn.Module):

    def __init__(self, in_chs, skip_chs, mid_chs, out_chs, is_first, stride, dilation=1):
        super(SelecSLSBlock, self).__init__()
        self.stride = stride
        self.is_first = is_first
        assert stride in [1, 2]
        self.conv1 = conv_bn(in_chs, mid_chs, 3, stride, dilation=dilation)
        self.conv2 = conv_bn(mid_chs, mid_chs, 1)
        self.conv3 = conv_bn(mid_chs, mid_chs // 2, 3)
        self.conv4 = conv_bn(mid_chs // 2, mid_chs, 1)
        self.conv5 = conv_bn(mid_chs, mid_chs // 2, 3)
        self.conv6 = conv_bn(2 * mid_chs + (0 if is_first else skip_chs), out_chs, 1)

    def forward(self, x: 'List[torch.Tensor]') ->List[torch.Tensor]:
        if not isinstance(x, list):
            x = [x]
        assert len(x) in [1, 2]
        d1 = self.conv1(x[0])
        d2 = self.conv3(self.conv2(d1))
        d3 = self.conv5(self.conv4(d2))
        if self.is_first:
            out = self.conv6(torch.cat([d1, d2, d3], 1))
            return [out, out]
        else:
            return [self.conv6(torch.cat([d1, d2, d3, x[1]], 1)), x[1]]


class SelecSLS(nn.Module):
    """SelecSLS42 / SelecSLS60 / SelecSLS84

    Parameters
    ----------
    cfg : network config dictionary specifying block type, feature, and head args
    num_classes : int, default 1000
        Number of classification classes.
    in_chans : int, default 3
        Number of input (color) channels.
    drop_rate : float, default 0.
        Dropout probability before classifier, for training
    global_pool : str, default 'avg'
        Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'
    """

    def __init__(self, cfg, num_classes=1000, in_chans=3, drop_rate=0.0, global_pool='avg'):
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        super(SelecSLS, self).__init__()
        self.stem = conv_bn(in_chans, 32, stride=2)
        self.features = SequentialList(*[cfg['block'](*block_args) for block_args in cfg['features']])
        self.from_seq = SelectSeq()
        self.head = nn.Sequential(*[conv_bn(*conv_args) for conv_args in cfg['head']])
        self.num_features = cfg['num_features']
        self.feature_info = cfg['feature_info']
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.head(self.from_seq(x))
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc(x)
        return x


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes, reduction=reduction)
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
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.se_module(out) + residual
        out = self.relu(out)
        return out


def _weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, drop_rate=0.2, in_chans=3, inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0, num_classes=1000, global_pool='avg'):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        if input_3x3:
            layer0_modules = [('conv1', nn.Conv2d(in_chans, 64, 3, stride=2, padding=1, bias=False)), ('bn1', nn.BatchNorm2d(64)), ('relu1', nn.ReLU(inplace=True)), ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)), ('bn2', nn.BatchNorm2d(64)), ('relu2', nn.ReLU(inplace=True)), ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False)), ('bn3', nn.BatchNorm2d(inplanes)), ('relu3', nn.ReLU(inplace=True))]
        else:
            layer0_modules = [('conv1', nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)), ('bn1', nn.BatchNorm2d(inplanes)), ('relu1', nn.ReLU(inplace=True))]
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.pool0 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='layer0')]
        self.layer1 = self._make_layer(block, planes=64, blocks=layers[0], groups=groups, reduction=reduction, downsample_kernel_size=1, downsample_padding=0)
        self.feature_info += [dict(num_chs=64 * block.expansion, reduction=4, module='layer1')]
        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.feature_info += [dict(num_chs=128 * block.expansion, reduction=8, module='layer2')]
        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.feature_info += [dict(num_chs=256 * block.expansion, reduction=16, module='layer3')]
        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.feature_info += [dict(num_chs=512 * block.expansion, reduction=32, module='layer4')]
        self.num_features = 512 * block.expansion
        self.global_pool, self.last_linear = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
        for m in self.modules():
            _weight_init(m)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1, downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=downsample_kernel_size, stride=stride, padding=downsample_padding, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, groups, reduction, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))
        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.last_linear

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.last_linear = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.layer0(x)
        x = self.pool0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.global_pool(x)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.logits(x)
        return x


class SelectiveKernelBasic(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64, sk_kwargs=None, reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(SelectiveKernelBasic, self).__init__()
        sk_kwargs = sk_kwargs or {}
        conv_kwargs = dict(drop_block=drop_block, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer)
        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock doest not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        self.conv1 = SelectiveKernelConv(inplanes, first_planes, stride=stride, dilation=first_dilation, **conv_kwargs, **sk_kwargs)
        conv_kwargs['act_layer'] = None
        self.conv2 = ConvBnAct(first_planes, outplanes, kernel_size=3, dilation=dilation, **conv_kwargs)
        self.se = create_attn(attn_layer, outplanes)
        self.act = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv2.bn.weight)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.se is not None:
            x = self.se(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act(x)
        return x


class SelectiveKernelBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64, sk_kwargs=None, reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(SelectiveKernelBottleneck, self).__init__()
        sk_kwargs = sk_kwargs or {}
        conv_kwargs = dict(drop_block=drop_block, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer)
        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        self.conv1 = ConvBnAct(inplanes, first_planes, kernel_size=1, **conv_kwargs)
        self.conv2 = SelectiveKernelConv(first_planes, width, stride=stride, dilation=first_dilation, groups=cardinality, **conv_kwargs, **sk_kwargs)
        conv_kwargs['act_layer'] = None
        self.conv3 = ConvBnAct(width, outplanes, kernel_size=1, **conv_kwargs)
        self.se = create_attn(attn_layer, outplanes)
        self.act = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.se is not None:
            x = self.se(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act(x)
        return x


class TResNet(nn.Module):

    def __init__(self, layers, in_chans=3, num_classes=1000, width_factor=1.0, no_aa_jit=False, global_pool='fast', drop_rate=0.0):
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        super(TResNet, self).__init__()
        space_to_depth = SpaceToDepthModule()
        aa_layer = partial(AntiAliasDownsampleLayer, no_jit=no_aa_jit)
        self.inplanes = int(64 * width_factor)
        self.planes = int(64 * width_factor)
        conv1 = conv2d_iabn(in_chans * 16, self.planes, stride=1, kernel_size=3)
        layer1 = self._make_layer(BasicBlock, self.planes, layers[0], stride=1, use_se=True, aa_layer=aa_layer)
        layer2 = self._make_layer(BasicBlock, self.planes * 2, layers[1], stride=2, use_se=True, aa_layer=aa_layer)
        layer3 = self._make_layer(Bottleneck, self.planes * 4, layers[2], stride=2, use_se=True, aa_layer=aa_layer)
        layer4 = self._make_layer(Bottleneck, self.planes * 8, layers[3], stride=2, use_se=False, aa_layer=aa_layer)
        self.body = nn.Sequential(OrderedDict([('SpaceToDepth', space_to_depth), ('conv1', conv1), ('layer1', layer1), ('layer2', layer2), ('layer3', layer3), ('layer4', layer4)]))
        self.feature_info = [dict(num_chs=self.planes, reduction=2, module=''), dict(num_chs=self.planes, reduction=4, module='body.layer1'), dict(num_chs=self.planes * 2, reduction=8, module='body.layer2'), dict(num_chs=self.planes * 4 * Bottleneck.expansion, reduction=16, module='body.layer3'), dict(num_chs=self.planes * 8 * Bottleneck.expansion, reduction=32, module='body.layer4')]
        self.num_features = self.planes * 8 * Bottleneck.expansion
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, InplaceAbn):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.conv2[1].weight = nn.Parameter(torch.zeros_like(m.conv2[1].weight))
            if isinstance(m, Bottleneck):
                m.conv3[1].weight = nn.Parameter(torch.zeros_like(m.conv3[1].weight))
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True, aa_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            if stride == 2:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False))
            layers += [conv2d_iabn(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, act_layer='identity')]
            downsample = nn.Sequential(*layers)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=use_se, aa_layer=aa_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=use_se, aa_layer=aa_layer))
        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='fast'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        return self.body(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class ConvMlp(nn.Module):

    def __init__(self, in_features=512, out_features=4096, kernel_size=7, mlp_ratio=1.0, drop_rate: 'float'=0.2, act_layer: 'nn.Module'=None, conv_layer: 'nn.Module'=None):
        super(ConvMlp, self).__init__()
        self.input_kernel_size = kernel_size
        mid_features = int(out_features * mlp_ratio)
        self.fc1 = conv_layer(in_features, mid_features, kernel_size, bias=True)
        self.act1 = act_layer(True)
        self.drop = nn.Dropout(drop_rate)
        self.fc2 = conv_layer(mid_features, out_features, 1, bias=True)
        self.act2 = act_layer(True)

    def forward(self, x):
        if x.shape[-2] < self.input_kernel_size or x.shape[-1] < self.input_kernel_size:
            output_size = max(self.input_kernel_size, x.shape[-2]), max(self.input_kernel_size, x.shape[-1])
            x = F.adaptive_avg_pool2d(x, output_size)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x


class VGG(nn.Module):

    def __init__(self, cfg: 'List[Any]', num_classes: 'int'=1000, in_chans: 'int'=3, output_stride: 'int'=32, mlp_ratio: 'float'=1.0, act_layer: 'nn.Module'=nn.ReLU, conv_layer: 'nn.Module'=nn.Conv2d, norm_layer: 'nn.Module'=None, global_pool: 'str'='avg', drop_rate: 'float'=0.0) ->None:
        super(VGG, self).__init__()
        assert output_stride == 32
        self.num_classes = num_classes
        self.num_features = 4096
        self.drop_rate = drop_rate
        self.feature_info = []
        prev_chs = in_chans
        net_stride = 1
        pool_layer = nn.MaxPool2d
        layers: 'List[nn.Module]' = []
        for v in cfg:
            last_idx = len(layers) - 1
            if v == 'M':
                self.feature_info.append(dict(num_chs=prev_chs, reduction=net_stride, module=f'features.{last_idx}'))
                layers += [pool_layer(kernel_size=2, stride=2)]
                net_stride *= 2
            else:
                v = cast(int, v)
                conv2d = conv_layer(prev_chs, v, kernel_size=3, padding=1)
                if norm_layer is not None:
                    layers += [conv2d, norm_layer(v), act_layer(inplace=True)]
                else:
                    layers += [conv2d, act_layer(inplace=True)]
                prev_chs = v
        self.features = nn.Sequential(*layers)
        self.feature_info.append(dict(num_chs=prev_chs, reduction=net_stride, module=f'features.{len(layers) - 1}'))
        self.pre_logits = ConvMlp(prev_chs, self.num_features, 7, mlp_ratio=mlp_ratio, drop_rate=drop_rate, act_layer=act_layer, conv_layer=conv_layer)
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)
        self._initialize_weights()

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.head = ClassifierHead(self.num_features, self.num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.features(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x

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


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = img_size[1] // patch_size[1] * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, representation_size=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, hybrid_backbone=None, norm_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-06)
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([('fc', nn.Linear(embed_dim, representation_size)), ('act', nn.Tanh())]))
        else:
            self.pre_logits = nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
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
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)[:, 0]
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class DistilledVisionTransformer(VisionTransformer):
    """ Vision Transformer with distillation token.

    Paper: `Training data-efficient image transformers & distillation through attention` -
        https://arxiv.org/abs/2012.12877

    This impl of distilled ViT is taken from https://github.com/facebookresearch/deit
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        trunc_normal_(self.dist_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            return (x + x_dist) / 2


class SequentialAppendList(nn.Sequential):

    def __init__(self, *args):
        super(SequentialAppendList, self).__init__(*args)

    def forward(self, x: 'torch.Tensor', concat_list: 'List[torch.Tensor]') ->torch.Tensor:
        for i, module in enumerate(self):
            if i == 0:
                concat_list.append(module(x))
            else:
                concat_list.append(module(concat_list[-1]))
        x = torch.cat(concat_list, dim=1)
        return x


class OsaBlock(nn.Module):

    def __init__(self, in_chs, mid_chs, out_chs, layer_per_block, residual=False, depthwise=False, attn='', norm_layer=BatchNormAct2d, act_layer=nn.ReLU, drop_path=None):
        super(OsaBlock, self).__init__()
        self.residual = residual
        self.depthwise = depthwise
        conv_kwargs = dict(norm_layer=norm_layer, act_layer=act_layer)
        next_in_chs = in_chs
        if self.depthwise and next_in_chs != mid_chs:
            assert not residual
            self.conv_reduction = ConvBnAct(next_in_chs, mid_chs, 1, **conv_kwargs)
        else:
            self.conv_reduction = None
        mid_convs = []
        for i in range(layer_per_block):
            if self.depthwise:
                conv = SeparableConvBnAct(mid_chs, mid_chs, **conv_kwargs)
            else:
                conv = ConvBnAct(next_in_chs, mid_chs, 3, **conv_kwargs)
            next_in_chs = mid_chs
            mid_convs.append(conv)
        self.conv_mid = SequentialAppendList(*mid_convs)
        next_in_chs = in_chs + layer_per_block * mid_chs
        self.conv_concat = ConvBnAct(next_in_chs, out_chs, **conv_kwargs)
        if attn:
            self.attn = create_attn(attn, out_chs)
        else:
            self.attn = None
        self.drop_path = drop_path

    def forward(self, x):
        output = [x]
        if self.conv_reduction is not None:
            x = self.conv_reduction(x)
        x = self.conv_mid(x, output)
        x = self.conv_concat(x)
        if self.attn is not None:
            x = self.attn(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.residual:
            x = x + output[0]
        return x


class OsaStage(nn.Module):

    def __init__(self, in_chs, mid_chs, out_chs, block_per_stage, layer_per_block, downsample=True, residual=True, depthwise=False, attn='ese', norm_layer=BatchNormAct2d, act_layer=nn.ReLU, drop_path_rates=None):
        super(OsaStage, self).__init__()
        if downsample:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        else:
            self.pool = None
        blocks = []
        for i in range(block_per_stage):
            last_block = i == block_per_stage - 1
            if drop_path_rates is not None and drop_path_rates[i] > 0.0:
                drop_path = DropPath(drop_path_rates[i])
            else:
                drop_path = None
            blocks += [OsaBlock(in_chs, mid_chs, out_chs, layer_per_block, residual=residual and i > 0, depthwise=depthwise, attn=attn if last_block else '', norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)]
            in_chs = out_chs
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        x = self.blocks(x)
        return x


class VovNet(nn.Module):

    def __init__(self, cfg, in_chans=3, num_classes=1000, global_pool='avg', drop_rate=0.0, stem_stride=4, output_stride=32, norm_layer=BatchNormAct2d, act_layer=nn.ReLU, drop_path_rate=0.0):
        """ VovNet (v2)
        """
        super(VovNet, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        assert stem_stride in (4, 2)
        assert output_stride == 32
        stem_chs = cfg['stem_chs']
        stage_conv_chs = cfg['stage_conv_chs']
        stage_out_chs = cfg['stage_out_chs']
        block_per_stage = cfg['block_per_stage']
        layer_per_block = cfg['layer_per_block']
        conv_kwargs = dict(norm_layer=norm_layer, act_layer=act_layer)
        last_stem_stride = stem_stride // 2
        conv_type = SeparableConvBnAct if cfg['depthwise'] else ConvBnAct
        self.stem = nn.Sequential(*[ConvBnAct(in_chans, stem_chs[0], 3, stride=2, **conv_kwargs), conv_type(stem_chs[0], stem_chs[1], 3, stride=1, **conv_kwargs), conv_type(stem_chs[1], stem_chs[2], 3, stride=last_stem_stride, **conv_kwargs)])
        self.feature_info = [dict(num_chs=stem_chs[1], reduction=2, module=f'stem.{1 if stem_stride == 4 else 2}')]
        current_stride = stem_stride
        stage_dpr = torch.split(torch.linspace(0, drop_path_rate, sum(block_per_stage)), block_per_stage)
        in_ch_list = stem_chs[-1:] + stage_out_chs[:-1]
        stage_args = dict(residual=cfg['residual'], depthwise=cfg['depthwise'], attn=cfg['attn'], **conv_kwargs)
        stages = []
        for i in range(4):
            downsample = stem_stride == 2 or i > 0
            stages += [OsaStage(in_ch_list[i], stage_conv_chs[i], stage_out_chs[i], block_per_stage[i], layer_per_block, downsample=downsample, drop_path_rates=stage_dpr[i], **stage_args)]
            self.num_features = stage_out_chs[i]
            current_stride *= 2 if downsample else 1
            self.feature_info += [dict(num_chs=self.num_features, reduction=current_stride, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        x = self.stem(x)
        return self.stages(x)

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x)


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000, in_chans=3, drop_rate=0.0, global_pool='avg'):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.drop_rate = drop_rate
        self.global_pool = global_pool
        self.num_classes = num_classes
        self.num_features = 2048
        self.conv1 = nn.Conv2d(in_chans, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU(inplace=True)
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False)
        self.block2 = Block(128, 256, 2, 2)
        self.block3 = Block(256, 728, 2, 2)
        self.block4 = Block(728, 728, 3, 1)
        self.block5 = Block(728, 728, 3, 1)
        self.block6 = Block(728, 728, 3, 1)
        self.block7 = Block(728, 728, 3, 1)
        self.block8 = Block(728, 728, 3, 1)
        self.block9 = Block(728, 728, 3, 1)
        self.block10 = Block(728, 728, 3, 1)
        self.block11 = Block(728, 728, 3, 1)
        self.block12 = Block(728, 1024, 2, 2, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.act3 = nn.ReLU(inplace=True)
        self.conv4 = SeparableConv2d(1536, self.num_features, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(self.num_features)
        self.act4 = nn.ReLU(inplace=True)
        self.feature_info = [dict(num_chs=64, reduction=2, module='act2'), dict(num_chs=128, reduction=4, module='block2.rep.0'), dict(num_chs=256, reduction=8, module='block3.rep.0'), dict(num_chs=728, reduction=16, module='block12.rep.0'), dict(num_chs=2048, reduction=32, module='act4')]
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate:
            F.dropout(x, self.drop_rate, training=self.training)
        x = self.fc(x)
        return x


to_3tuple = _ntuple(3)


class XceptionModule(nn.Module):

    def __init__(self, in_chs, out_chs, stride=1, dilation=1, pad_type='', start_with_relu=True, no_skip=False, act_layer=nn.ReLU, norm_layer=None):
        super(XceptionModule, self).__init__()
        out_chs = to_3tuple(out_chs)
        self.in_channels = in_chs
        self.out_channels = out_chs[-1]
        self.no_skip = no_skip
        if not no_skip and (self.out_channels != self.in_channels or stride != 1):
            self.shortcut = ConvBnAct(in_chs, self.out_channels, 1, stride=stride, norm_layer=norm_layer, act_layer=None)
        else:
            self.shortcut = None
        separable_act_layer = None if start_with_relu else act_layer
        self.stack = nn.Sequential()
        for i in range(3):
            if start_with_relu:
                self.stack.add_module(f'act{i + 1}', nn.ReLU(inplace=i > 0))
            self.stack.add_module(f'conv{i + 1}', SeparableConv2d(in_chs, out_chs[i], 3, stride=stride if i == 2 else 1, dilation=dilation, padding=pad_type, act_layer=separable_act_layer, norm_layer=norm_layer))
            in_chs = out_chs[i]

    def forward(self, x):
        skip = x
        x = self.stack(x)
        if self.shortcut is not None:
            skip = self.shortcut(skip)
        if not self.no_skip:
            x = x + skip
        return x


class XceptionAligned(nn.Module):
    """Modified Aligned Xception
    """

    def __init__(self, block_cfg, num_classes=1000, in_chans=3, output_stride=32, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, drop_rate=0.0, global_pool='avg'):
        super(XceptionAligned, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        assert output_stride in (8, 16, 32)
        layer_args = dict(act_layer=act_layer, norm_layer=norm_layer)
        self.stem = nn.Sequential(*[ConvBnAct(in_chans, 32, kernel_size=3, stride=2, **layer_args), ConvBnAct(32, 64, kernel_size=3, stride=1, **layer_args)])
        curr_dilation = 1
        curr_stride = 2
        self.feature_info = []
        self.blocks = nn.Sequential()
        for i, b in enumerate(block_cfg):
            b['dilation'] = curr_dilation
            if b['stride'] > 1:
                self.feature_info += [dict(num_chs=to_3tuple(b['out_chs'])[-2], reduction=curr_stride, module=f'blocks.{i}.stack.act3')]
                next_stride = curr_stride * b['stride']
                if next_stride > output_stride:
                    curr_dilation *= b['stride']
                    b['stride'] = 1
                else:
                    curr_stride = next_stride
            self.blocks.add_module(str(i), XceptionModule(**b, **layer_args))
            self.num_features = self.blocks[-1].out_channels
        self.feature_info += [dict(num_chs=self.num_features, reduction=curr_stride, module='blocks.' + str(len(self.blocks) - 1))]
        self.head = ClassifierHead(in_chs=self.num_features, num_classes=num_classes, pool_type=global_pool, drop_rate=drop_rate)

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class ModelEmaV2(nn.Module):
    """ Model Exponential Moving Average V2

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AsymmetricLossMultiLabel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (BasicConv2d,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BatchNormAct2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Block17,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1088, 64, 64])], {})),
    (Block35,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 320, 64, 64])], {})),
    (Block8,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2080, 64, 64])], {})),
    (CatBnAct,
     lambda: ([], {'in_chs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ChannelShuffle,
     lambda: ([], {'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DenseBlock,
     lambda: ([], {'num_layers': 1, 'num_input_features': 4, 'bn_size': 4, 'growth_rate': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DenseLayer,
     lambda: ([], {'num_input_features': 4, 'growth_rate': 4, 'bn_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DenseTransition,
     lambda: ([], {'num_input_features': 4, 'num_output_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DlaBasic,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DlaBottleneck,
     lambda: ([], {'inplanes': 4, 'outplanes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DropBlock2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (EcaModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (EffectiveSEModule,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (EvoNormBatch2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FastAdaptiveAvgPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GroupLinear,
     lambda: ([], {'in_planes': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (GroupNorm,
     lambda: ([], {'num_groups': 1, 'embed_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (HardMish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HardMishJit,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HardMishMe,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HardSigmoidJit,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HardSigmoidMe,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HardSwishJit,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HardSwishMe,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InceptionA,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 384, 64, 64])], {})),
    (InceptionB,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1024, 64, 64])], {})),
    (InceptionC,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1536, 64, 64])], {})),
    (Mixed3a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {})),
    (Mixed4a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 160, 64, 64])], {})),
    (Mixed5a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 192, 64, 64])], {})),
    (Mixed_5b,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 192, 64, 64])], {})),
    (Mixed_6a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 320, 64, 64])], {})),
    (Mixed_7a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1088, 64, 64])], {})),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ReductionA,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 384, 64, 64])], {})),
    (ReductionB,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1024, 64, 64])], {})),
    (SEModule,
     lambda: ([], {'channels': 4, 'reduction': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SEResNetBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4, 'groups': 1, 'reduction': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SEWithNorm,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SelecSLSBlock,
     lambda: ([], {'in_chs': 4, 'skip_chs': 4, 'mid_chs': 4, 'out_chs': 4, 'is_first': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SelectAdaptivePool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SelectSeq,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SequentialList,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Sigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SpaceToDepth,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SplitBatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SqueezeExcite,
     lambda: ([], {'in_chs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SwishJit,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SwishMe,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Tanh,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

