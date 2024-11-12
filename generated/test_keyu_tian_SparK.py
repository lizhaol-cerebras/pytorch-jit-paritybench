
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


from typing import List


from typing import Dict


from typing import Set


from typing import Optional


from typing import Callable


from typing import Any


import copy


import logging


import time


from collections import OrderedDict


from collections import defaultdict


from functools import partial


import numpy as np


import random


import torchvision


from torch.utils.data import DataLoader


from torch.utils.data.sampler import Sampler


from torchvision.transforms import AutoAugment as TorchAutoAugment


from torchvision.transforms import transforms


from torchvision.transforms import TrivialAugmentWide


import torch.distributed as tdist


from torch.utils.tensorboard import SummaryWriter


import math


import torch.nn as nn


import torch.nn.functional as F


from typing import Tuple


import torch.multiprocessing as tmp


from torch.nn.parallel import DistributedDataParallel


from torch.optim.optimizer import Optimizer


from torch.optim import Optimizer


from typing import Union


import torch.multiprocessing as mp


from torchvision.datasets.folder import DatasetFolder


from torchvision.datasets.folder import IMG_EXTENSIONS


from torch.utils.data import Dataset


import functools


from collections import deque


from typing import Iterator


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


def _get_active_ex_or_ii(H, W, returning_active_ex=True):
    h_repeat, w_repeat = H // _cur_active.shape[-2], W // _cur_active.shape[-1]
    active_ex = _cur_active.repeat_interleave(h_repeat, dim=2).repeat_interleave(w_repeat, dim=3)
    return active_ex if returning_active_ex else active_ex.squeeze(1).nonzero(as_tuple=True)


class SparseConvNeXtLayerNorm(nn.LayerNorm):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-06, data_format='channels_last', sparse=True):
        if data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError
        super().__init__(normalized_shape, eps, elementwise_affine=True)
        self.data_format = data_format
        self.sparse = sparse

    def forward(self, x):
        if x.ndim == 4:
            if self.data_format == 'channels_last':
                if self.sparse:
                    ii = _get_active_ex_or_ii(H=x.shape[1], W=x.shape[2], returning_active_ex=False)
                    nc = x[ii]
                    nc = super(SparseConvNeXtLayerNorm, self).forward(nc)
                    x = torch.zeros_like(x)
                    x[ii] = nc
                    return x
                else:
                    return super(SparseConvNeXtLayerNorm, self).forward(x)
            elif self.sparse:
                ii = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=False)
                bhwc = x.permute(0, 2, 3, 1)
                nc = bhwc[ii]
                nc = super(SparseConvNeXtLayerNorm, self).forward(nc)
                x = torch.zeros_like(bhwc)
                x[ii] = nc
                return x.permute(0, 3, 1, 2)
            else:
                u = x.mean(1, keepdim=True)
                s = (x - u).pow(2).mean(1, keepdim=True)
                x = (x - u) / torch.sqrt(s + self.eps)
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
                return x
        elif self.sparse:
            raise NotImplementedError
        else:
            return super(SparseConvNeXtLayerNorm, self).forward(x)

    def __repr__(self):
        return super(SparseConvNeXtLayerNorm, self).__repr__()[:-1] + f", ch={self.data_format.split('_')[-1]}, sp={self.sparse})"


class SparseConvNeXtBlock(nn.Module):
    """ ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-06, sparse=True, ks=7):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=ks, padding=ks // 2, groups=dim)
        self.norm = SparseConvNeXtLayerNorm(dim, eps=1e-06, sparse=sparse)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path: 'nn.Module' = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.sparse = sparse

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
        if self.sparse:
            x *= _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=True)
        x = input + self.drop_path(x)
        return x

    def __repr__(self):
        return super(SparseConvNeXtBlock, self).__repr__()[:-1] + f', sp={self.sparse})'


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

    def __init__(self, in_chans=3, num_classes=1000, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.0, layer_scale_init_value=1e-06, head_init_scale=1.0, global_pool='avg', sparse=True):
        super().__init__()
        self.dims: 'List[int]' = dims
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4), SparseConvNeXtLayerNorm(dims[0], eps=1e-06, data_format='channels_first', sparse=sparse))
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(SparseConvNeXtLayerNorm(dims[i], eps=1e-06, data_format='channels_first', sparse=sparse), nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)
        self.stages = nn.ModuleList()
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[SparseConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value, sparse=sparse) for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]
        self.depths = depths
        self.apply(self._init_weights)
        if num_classes > 0:
            self.norm = SparseConvNeXtLayerNorm(dims[-1], eps=1e-06, sparse=False)
            self.fc = nn.Linear(dims[-1], num_classes)
        else:
            self.norm = nn.Identity()
            self.fc = nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def get_downsample_ratio(self) ->int:
        return 32

    def get_feature_map_channels(self) ->List[int]:
        return self.dims

    def forward(self, x, hierarchical=False):
        if hierarchical:
            ls = []
            for i in range(4):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
                ls.append(x)
            return ls
        else:
            return self.fc(self.norm(x.mean([-2, -1])))

    def get_classifier(self):
        return self.fc

    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate}, layer_scale_init_value={self.layer_scale_init_value:g}'


class UNetBlock(nn.Module):

    def __init__(self, cin, cout, bn2d):
        """
        a UNet block with 2x up sampling
        """
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(cin, cin, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv = nn.Sequential(nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False), bn2d(cin), nn.ReLU6(inplace=True), nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False), bn2d(cout))

    def forward(self, x):
        x = self.up_sample(x)
        return self.conv(x)


def is_pow2n(x):
    return x > 0 and x & x - 1 == 0


class LightDecoder(nn.Module):

    def __init__(self, up_sample_ratio, width=768, sbn=True):
        super().__init__()
        self.width = width
        assert is_pow2n(up_sample_ratio)
        n = round(math.log2(up_sample_ratio))
        channels = [(self.width // 2 ** i) for i in range(n + 1)]
        bn2d = nn.SyncBatchNorm if sbn else nn.BatchNorm2d
        self.dec = nn.ModuleList([UNetBlock(cin, cout, bn2d) for cin, cout in zip(channels[:-1], channels[1:])])
        self.proj = nn.Conv2d(channels[-1], 3, kernel_size=1, stride=1, bias=True)
        self.initialize()

    def forward(self, to_dec: 'List[torch.Tensor]'):
        x = 0
        for i, d in enumerate(self.dec):
            if i < len(to_dec) and to_dec[i] is not None:
                x = x + to_dec[i]
            x = self.dec[i](x)
        return self.proj(x)

    def extra_repr(self) ->str:
        return f'width={self.width}'

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


def sp_conv_forward(self, x: 'torch.Tensor'):
    x = super(type(self), self).forward(x)
    x *= _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=True)
    return x


class SparseConv2d(nn.Conv2d):
    forward = sp_conv_forward


class SparseMaxPooling(nn.MaxPool2d):
    forward = sp_conv_forward


class SparseAvgPooling(nn.AvgPool2d):
    forward = sp_conv_forward


def sp_bn_forward(self, x: 'torch.Tensor'):
    ii = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=False)
    bhwc = x.permute(0, 2, 3, 1)
    nc = bhwc[ii]
    nc = super(type(self), self).forward(nc)
    bchw = torch.zeros_like(bhwc)
    bchw[ii] = nc
    bchw = bchw.permute(0, 3, 1, 2)
    return bchw


class SparseBatchNorm2d(nn.BatchNorm1d):
    forward = sp_bn_forward


class SparseSyncBatchNorm2d(nn.SyncBatchNorm):
    forward = sp_bn_forward


class SparseEncoder(nn.Module):

    def __init__(self, cnn, input_size, sbn=False, verbose=False):
        super(SparseEncoder, self).__init__()
        self.sp_cnn = SparseEncoder.dense_model_to_sparse(m=cnn, verbose=verbose, sbn=sbn)
        self.input_size, self.downsample_raito, self.enc_feat_map_chs = input_size, cnn.get_downsample_ratio(), cnn.get_feature_map_channels()

    @staticmethod
    def dense_model_to_sparse(m: 'nn.Module', verbose=False, sbn=False):
        oup = m
        if isinstance(m, nn.Conv2d):
            m: 'nn.Conv2d'
            bias = m.bias is not None
            oup = SparseConv2d(m.in_channels, m.out_channels, kernel_size=m.kernel_size, stride=m.stride, padding=m.padding, dilation=m.dilation, groups=m.groups, bias=bias, padding_mode=m.padding_mode)
            oup.weight.data.copy_(m.weight.data)
            if bias:
                oup.bias.data.copy_(m.bias.data)
        elif isinstance(m, nn.MaxPool2d):
            m: 'nn.MaxPool2d'
            oup = SparseMaxPooling(m.kernel_size, stride=m.stride, padding=m.padding, dilation=m.dilation, return_indices=m.return_indices, ceil_mode=m.ceil_mode)
        elif isinstance(m, nn.AvgPool2d):
            m: 'nn.AvgPool2d'
            oup = SparseAvgPooling(m.kernel_size, m.stride, m.padding, ceil_mode=m.ceil_mode, count_include_pad=m.count_include_pad, divisor_override=m.divisor_override)
        elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m: 'nn.BatchNorm2d'
            oup = (SparseSyncBatchNorm2d if sbn else SparseBatchNorm2d)(m.weight.shape[0], eps=m.eps, momentum=m.momentum, affine=m.affine, track_running_stats=m.track_running_stats)
            oup.weight.data.copy_(m.weight.data)
            oup.bias.data.copy_(m.bias.data)
            oup.running_mean.data.copy_(m.running_mean.data)
            oup.running_var.data.copy_(m.running_var.data)
            oup.num_batches_tracked.data.copy_(m.num_batches_tracked.data)
            if hasattr(m, 'qconfig'):
                oup.qconfig = m.qconfig
        elif isinstance(m, nn.LayerNorm) and not isinstance(m, SparseConvNeXtLayerNorm):
            m: 'nn.LayerNorm'
            oup = SparseConvNeXtLayerNorm(m.weight.shape[0], eps=m.eps)
            oup.weight.data.copy_(m.weight.data)
            oup.bias.data.copy_(m.bias.data)
        elif isinstance(m, (nn.Conv1d,)):
            raise NotImplementedError
        for name, child in m.named_children():
            oup.add_module(name, SparseEncoder.dense_model_to_sparse(child, verbose=verbose, sbn=sbn))
        del m
        return oup

    def forward(self, x):
        return self.sp_cnn(x, hierarchical=True)


class LocalDDP(torch.nn.Module):

    def __init__(self, module):
        super(LocalDDP, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class YourConvNet(nn.Module):
    '\n    This is a template for your custom ConvNet.\n    It is required to implement the following three functions: `get_downsample_ratio`, `get_feature_map_channels`, `forward`.\n    You can refer to the implementations in `pretrain\\models\resnet.py` for an example.\n    '

    def get_downsample_ratio(self) ->int:
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).
        
        :return: the TOTAL downsample ratio of the ConvNet.
        E.g., for a ResNet-50, this should return 32.
        """
        raise NotImplementedError

    def get_feature_map_channels(self) ->List[int]:
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).
        
        :return: a list of the number of channels of each feature map.
        E.g., for a ResNet-50, this should return [256, 512, 1024, 2048].
        """
        raise NotImplementedError

    def forward(self, inp_bchw: 'torch.Tensor', hierarchical=False):
        """
        The forward with `hierarchical=True` would ONLY be used in `SparseEncoder.forward` (see `pretrain/encoder.py`).
        
        :param inp_bchw: input image tensor, shape: (batch_size, channels, height, width).
        :param hierarchical: return the logits (not hierarchical), or the feature maps (hierarchical).
        :return:
            - hierarchical == False: return the logits of the classification task, shape: (batch_size, num_classes).
            - hierarchical == True: return a list of all feature maps, which should have the same length as the return value of `get_feature_map_channels`.
              E.g., for a ResNet-50, it should return a list [1st_feat_map, 2nd_feat_map, 3rd_feat_map, 4th_feat_map].
                    for an input size of 224, the shapes are [(B, 256, 56, 56), (B, 512, 28, 28), (B, 1024, 14, 14), (B, 2048, 7, 7)]
        """
        raise NotImplementedError


class SparK(nn.Module):

    def __init__(self, sparse_encoder: 'encoder.SparseEncoder', dense_decoder: 'LightDecoder', mask_ratio=0.6, densify_norm='bn', sbn=False):
        super().__init__()
        input_size, downsample_raito = sparse_encoder.input_size, sparse_encoder.downsample_raito
        self.downsample_raito = downsample_raito
        self.fmap_h, self.fmap_w = input_size // downsample_raito, input_size // downsample_raito
        self.mask_ratio = mask_ratio
        self.len_keep = round(self.fmap_h * self.fmap_w * (1 - mask_ratio))
        self.sparse_encoder = sparse_encoder
        self.dense_decoder = dense_decoder
        self.sbn = sbn
        self.hierarchy = len(sparse_encoder.enc_feat_map_chs)
        self.densify_norm_str = densify_norm.lower()
        self.densify_norms = nn.ModuleList()
        self.densify_projs = nn.ModuleList()
        self.mask_tokens = nn.ParameterList()
        e_widths, d_width = self.sparse_encoder.enc_feat_map_chs, self.dense_decoder.width
        e_widths: 'List[int]'
        for i in range(self.hierarchy):
            e_width = e_widths.pop()
            p = nn.Parameter(torch.zeros(1, e_width, 1, 1))
            trunc_normal_(p, mean=0, std=0.02, a=-0.02, b=0.02)
            self.mask_tokens.append(p)
            if self.densify_norm_str == 'bn':
                densify_norm = (encoder.SparseSyncBatchNorm2d if self.sbn else encoder.SparseBatchNorm2d)(e_width)
            elif self.densify_norm_str == 'ln':
                densify_norm = encoder.SparseConvNeXtLayerNorm(e_width, data_format='channels_first', sparse=True)
            else:
                densify_norm = nn.Identity()
            self.densify_norms.append(densify_norm)
            if i == 0 and e_width == d_width:
                densify_proj = nn.Identity()
                None
            else:
                kernel_size = 1 if i <= 0 else 3
                densify_proj = nn.Conv2d(e_width, d_width, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True)
                None
            self.densify_projs.append(densify_proj)
            d_width //= 2
        None
        self.register_buffer('imn_m', torch.empty(1, 3, 1, 1))
        self.register_buffer('imn_s', torch.empty(1, 3, 1, 1))
        self.register_buffer('norm_black', torch.zeros(1, 3, input_size, input_size))
        self.vis_active = self.vis_active_ex = self.vis_inp = self.vis_inp_mask = ...

    def mask(self, B: 'int', device, generator=None):
        h, w = self.fmap_h, self.fmap_w
        idx = torch.rand(B, h * w, generator=generator).argsort(dim=1)
        idx = idx[:, :self.len_keep]
        return torch.zeros(B, h * w, dtype=torch.bool, device=device).scatter_(dim=1, index=idx, value=True).view(B, 1, h, w)

    def forward(self, inp_bchw: 'torch.Tensor', active_b1ff=None, vis=False):
        if active_b1ff is None:
            active_b1ff: 'torch.BoolTensor' = self.mask(inp_bchw.shape[0], inp_bchw.device)
        encoder._cur_active = active_b1ff
        active_b1hw = active_b1ff.repeat_interleave(self.downsample_raito, 2).repeat_interleave(self.downsample_raito, 3)
        masked_bchw = inp_bchw * active_b1hw
        fea_bcffs: 'List[torch.Tensor]' = self.sparse_encoder(masked_bchw)
        fea_bcffs.reverse()
        cur_active = active_b1ff
        to_dec = []
        for i, bcff in enumerate(fea_bcffs):
            if bcff is not None:
                bcff = self.densify_norms[i](bcff)
                mask_tokens = self.mask_tokens[i].expand_as(bcff)
                bcff = torch.where(cur_active.expand_as(bcff), bcff, mask_tokens)
                bcff: 'torch.Tensor' = self.densify_projs[i](bcff)
            to_dec.append(bcff)
            cur_active = cur_active.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        rec_bchw = self.dense_decoder(to_dec)
        inp, rec = self.patchify(inp_bchw), self.patchify(rec_bchw)
        mean = inp.mean(dim=-1, keepdim=True)
        var = (inp.var(dim=-1, keepdim=True) + 1e-06) ** 0.5
        inp = (inp - mean) / var
        l2_loss = ((rec - inp) ** 2).mean(dim=2, keepdim=False)
        non_active = active_b1ff.logical_not().int().view(active_b1ff.shape[0], -1)
        recon_loss = l2_loss.mul_(non_active).sum() / (non_active.sum() + 1e-08)
        if vis:
            masked_bchw = inp_bchw * active_b1hw
            rec_bchw = self.unpatchify(rec * var + mean)
            rec_or_inp = torch.where(active_b1hw, inp_bchw, rec_bchw)
            return inp_bchw, masked_bchw, rec_or_inp
        else:
            return recon_loss

    def patchify(self, bchw):
        p = self.downsample_raito
        h, w = self.fmap_h, self.fmap_w
        B, C = bchw.shape[:2]
        bchw = bchw.reshape(shape=(B, C, h, p, w, p))
        bchw = torch.einsum('bchpwq->bhwpqc', bchw)
        bln = bchw.reshape(shape=(B, h * w, C * p ** 2))
        return bln

    def unpatchify(self, bln):
        p = self.downsample_raito
        h, w = self.fmap_h, self.fmap_w
        B, C = bln.shape[0], bln.shape[-1] // p ** 2
        bln = bln.reshape(shape=(B, h, w, p, p, C))
        bln = torch.einsum('bhwpqc->bchpwq', bln)
        bchw = bln.reshape(shape=(B, C, h * p, w * p))
        return bchw

    def __repr__(self):
        return f"\n[SparK.config]: {pformat(self.get_config(), indent=2, width=250)}\n[SparK.structure]: {super(SparK, self).__repr__().replace(SparK.__name__, '')}"

    def get_config(self):
        return {'mask_ratio': self.mask_ratio, 'densify_norm_str': self.densify_norm_str, 'sbn': self.sbn, 'hierarchy': self.hierarchy, 'sparse_encoder.input_size': self.sparse_encoder.input_size, 'dense_decoder.width': self.dense_decoder.width}

    def state_dict(self, destination=None, prefix='', keep_vars=False, with_config=False):
        state = super(SparK, self).state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if with_config:
            state['config'] = self.get_config()
        return state

    def load_state_dict(self, state_dict, strict=True):
        config: 'dict' = state_dict.pop('config', None)
        incompatible_keys = super(SparK, self).load_state_dict(state_dict, strict=strict)
        if config is not None:
            for k, v in self.get_config().items():
                ckpt_v = config.get(k, None)
                if ckpt_v != v:
                    err = f'[SparseMIM.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={ckpt_v})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        None
        return incompatible_keys


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Block,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LocalDDP,
     lambda: ([], {'module': torch.nn.ReLU()}),
     lambda: ([], {'input': torch.rand([4, 4])})),
]

