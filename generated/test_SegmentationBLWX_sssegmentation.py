
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


import numpy as np


import matplotlib.pyplot as plt


import re


import copy


import warnings


import torch.nn.functional as F


import collections


import scipy.io as sio


from collections import OrderedDict


import numbers


import collections.abc


import torch.nn as nn


import torch.utils.model_zoo as model_zoo


from scipy import interpolate


import math


from functools import partial


import torch.utils.checkpoint as checkpoint


import itertools


import torch.distributed as dist


import torch.optim as optim


from torch.nn.utils import clip_grad


import random


from scipy.optimize import linear_sum_assignment


import torchvision


from sklearn.cluster import _kmeans


from sklearn.preprocessing import StandardScaler


from sklearn.metrics.pairwise import cosine_similarity


from abc import ABCMeta


from torch import nn


from torch.nn import functional as F


from torchvision.ops.boxes import batched_nms


from torchvision.ops.boxes import box_area


from copy import deepcopy


from torchvision.transforms.functional import resize


from torchvision.transforms.functional import to_pil_image


from itertools import product


from torchvision.transforms import Normalize


from torchvision.transforms import Resize


from torchvision.transforms import ToTensor


from typing import Optional


from collections import defaultdict


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """computes standard normal cumulative distribution function"""

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
    with torch.no_grad():
        lower_bound = norm_cdf((a - mean) / std)
        upper_bound = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower_bound - 1, 2 * upper_bound - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def truncnormal(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class BEiTAttention(nn.Module):

    def __init__(self, embed_dims, num_heads, window_size, bias='qv_bias', qk_scale=None, attn_drop_rate=0.0, proj_drop_rate=0.0):
        super(BEiTAttention, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.bias = bias
        self.scale = qk_scale or head_embed_dims ** -0.5
        qkv_bias = bias
        if bias == 'qv_bias':
            self.q_bias = nn.Parameter(torch.zeros(self.embed_dims))
            self.v_bias = nn.Parameter(torch.zeros(self.embed_dims))
            qkv_bias = False
        self.window_size = window_size
        self.initrelposembedding()
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        truncnormal(self.relative_position_bias_table, std=0.02)
    """initrelposembedding"""

    def initrelposembedding(self):
        Wh, Ww = self.window_size
        self.num_relative_distance = (2 * Wh - 1) * (2 * Ww - 1) + 3
        self.relative_position_bias_table = nn.Parameter(torch.zeros(self.num_relative_distance, self.num_heads))
        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += Wh - 1
        relative_coords[:, :, 1] += Ww - 1
        relative_coords[:, :, 0] *= 2 * Ww - 1
        relative_position_index = torch.zeros(size=(Wh * Ww + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1
        self.register_buffer('relative_position_index', relative_position_index)
    """forward"""

    def forward(self, x):
        B, N, C = x.shape
        if self.bias == 'qv_bias':
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            qkv_bias = torch.cat((self.q_bias, k_bias, self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        else:
            qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.relative_position_bias_table is not None:
            Wh = self.window_size[0]
            Ww = self.window_size[1]
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(Wh * Ww + 1, Wh * Ww + 1, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


AUTO_ASSERT_STRUCTURE_TYPES = {'jx_vit_large_p16_384': {'patch_size': 16, 'embed_dims': 1024, 'num_layers': 24, 'num_heads': 16, 'mlp_ratio': 4, 'qkv_bias': True, 'drop_rate': 0.1, 'attn_drop_rate': 0.0, 'drop_path_rate': 0.0, 'with_cls_token': True, 'output_cls_token': False, 'patch_norm': False, 'final_norm': False, 'num_fcs': 2}}


class BaseModuleBuilder:
    REGISTERED_MODULES = collections.OrderedDict()

    def __init__(self, requires_register_modules=None, requires_renew_modules=None):
        if requires_register_modules is not None and isinstance(requires_register_modules, (dict, collections.OrderedDict)):
            for name, module in requires_register_modules.items():
                self.register(name, module)
        if requires_renew_modules is not None and isinstance(requires_renew_modules, (dict, collections.OrderedDict)):
            for name, module in requires_renew_modules.items():
                self.renew(name, module)
        self.validate()
    """build"""

    def build(self, module_cfg):
        module_cfg = copy.deepcopy(module_cfg)
        module_type = module_cfg.pop('type')
        module = self.REGISTERED_MODULES[module_type](**module_cfg)
        return module
    """register"""

    def register(self, name, module):
        assert callable(module)
        assert name not in self.REGISTERED_MODULES
        self.REGISTERED_MODULES[name] = module
    """renew"""

    def renew(self, name, module):
        assert callable(module)
        assert name in self.REGISTERED_MODULES
        self.REGISTERED_MODULES[name] = module
    """validate"""

    def validate(self):
        for _, module in self.REGISTERED_MODULES.items():
            assert callable(module)
    """delete"""

    def delete(self, name):
        assert name in self.REGISTERED_MODULES
        del self.REGISTERED_MODULES[name]
    """pop"""

    def pop(self, name):
        assert name in self.REGISTERED_MODULES
        module = self.REGISTERED_MODULES.pop(name)
        return module
    """get"""

    def get(self, name):
        assert name in self.REGISTERED_MODULES
        module = self.REGISTERED_MODULES.get(name)
        return module
    """items"""

    def items(self):
        return self.REGISTERED_MODULES.items()
    """clear"""

    def clear(self):
        return self.REGISTERED_MODULES.clear()
    """values"""

    def values(self):
        return self.REGISTERED_MODULES.values()
    """keys"""

    def keys(self):
        return self.REGISTERED_MODULES.keys()
    """copy"""

    def copy(self):
        return self.REGISTERED_MODULES.copy()
    """update"""

    def update(self, requires_update_modules):
        return self.REGISTERED_MODULES.update(requires_update_modules)


class DropPath(nn.Module):

    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    """forward"""

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class DropoutBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {'DropPath': DropPath, 'Dropout': nn.Dropout, 'Dropout2d': nn.Dropout2d, 'Dropout3d': nn.Dropout3d}
    """build"""

    def build(self, dropout_cfg):
        if dropout_cfg is None:
            return nn.Identity()
        return super().build(dropout_cfg)


BuildDropout = DropoutBuilder().build


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-06, data_format='channels_last'):
        super(LayerNorm, self).__init__()
        assert data_format in ['channels_last', 'channels_first']
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = normalized_shape,
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
    """forward"""

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):

    def __init__(self, dim, eps=1e-06):
        super(GRN, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
    """forward"""

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * Nx) + self.beta + x


class LayerNorm2d(nn.Module):

    def __init__(self, num_channels, eps=1e-06):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    """forward"""

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class NormalizationBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {'LayerNorm': nn.LayerNorm, 'LayerNorm2d': LayerNorm2d, 'GroupNorm': nn.GroupNorm, 'LocalResponseNorm': nn.LocalResponseNorm, 'BatchNorm1d': nn.BatchNorm1d, 'BatchNorm2d': nn.BatchNorm2d, 'BatchNorm3d': nn.BatchNorm3d, 'SyncBatchNorm': nn.SyncBatchNorm, 'InstanceNorm1d': nn.InstanceNorm1d, 'InstanceNorm2d': nn.InstanceNorm2d, 'InstanceNorm3d': nn.InstanceNorm3d, 'GRN': GRN}
    for norm_type in ['LazyBatchNorm1d', 'LazyBatchNorm2d', 'LazyBatchNorm3d', 'LazyInstanceNorm1d', 'LazyInstanceNorm2d', 'LazyInstanceNorm3d']:
        if hasattr(nn, norm_type):
            REGISTERED_MODULES[norm_type] = getattr(nn, norm_type)
    """build"""

    def build(self, placeholder, norm_cfg):
        if norm_cfg is None:
            return nn.Identity()
        norm_cfg = copy.deepcopy(norm_cfg)
        norm_type = norm_cfg.pop('type')
        if norm_type in ['GroupNorm']:
            normalization = self.REGISTERED_MODULES[norm_type](num_channels=placeholder, **norm_cfg)
        else:
            normalization = self.REGISTERED_MODULES[norm_type](placeholder, **norm_cfg)
        return normalization
    """isnorm"""

    @staticmethod
    def isnorm(module, norm_list=None):
        if norm_list is None:
            norm_list = nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d, nn.SyncBatchNorm
        return isinstance(module, norm_list)


BuildNormalization = NormalizationBuilder(requires_register_modules={'LayerNormConvNeXtV2': LayerNorm}).build


DEFAULT_MODEL_URLS = {'jx_vit_large_p16_384': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth'}


class PatchEmbed(nn.Module):

    def __init__(self, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0), in_chans=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
    """forward"""

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


class BEiT(nn.Module):

    def __init__(self, structure_type, img_size=(640, 640), patch_size=16, in_channels=3, embed_dims=768, num_layers=12, num_heads=12, mlp_ratio=4, out_indices=(3, 5, 7, 11), qv_bias=True, attn_drop_rate=0.0, drop_path_rate=0.1, norm_cfg={'type': 'LayerNorm', 'eps': 1e-06}, act_cfg={'type': 'GELU'}, patch_norm=False, final_norm=False, num_fcs=2, init_values=0.1, pretrained=True, pretrained_model_path=''):
        super(BEiT, self).__init__()
        img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.structure_type = structure_type
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.out_indices = out_indices
        self.qv_bias = qv_bias
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.patch_norm = patch_norm
        self.final_norm = final_norm
        self.num_fcs = num_fcs
        self.init_values = init_values
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        self.window_size = img_size[0] // patch_size, img_size[1] // patch_size
        self.patch_shape = self.window_size
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and getattr(self, key) == value
        if isinstance(out_indices, int):
            if out_indices == -1:
                out_indices = num_layers - 1
            self.out_indices = [out_indices]
        elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
            self.out_indices = out_indices
        else:
            raise TypeError('out_indices must be type of int, list or tuple')
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.buildpatchembedding()
        self.buildlayers()
        if final_norm:
            self.norm1 = BuildNormalization(placeholder=embed_dims, norm_cfg=norm_cfg)
        if pretrained:
            self.loadpretrainedweights(structure_type, pretrained_model_path)
    """buildpatchembedding"""

    def buildpatchembedding(self):
        self.patch_embed = PatchEmbed(in_channels=self.in_channels, embed_dims=self.embed_dims, kernel_size=self.patch_size, stride=self.patch_size, padding=0, norm_cfg=self.norm_cfg if self.patch_norm else None)
    """buildlayers"""

    def buildlayers(self):
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.num_layers)]
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(BEiTTransformerEncoderLayer(embed_dims=self.embed_dims, num_heads=self.num_heads, feedforward_channels=self.mlp_ratio * self.embed_dims, attn_drop_rate=self.attn_drop_rate, drop_path_rate=dpr[i], num_fcs=self.num_fcs, bias='qv_bias' if self.qv_bias else False, act_cfg=self.act_cfg, norm_cfg=self.norm_cfg, window_size=self.window_size, init_values=self.init_values))
    """geometricsequenceinterpolation"""

    def geometricsequenceinterpolation(self, src_size, dst_size, sequence, num):

        def geometricprogression(a, r, n):
            return a * (1.0 - r ** n) / (1.0 - r)
        left, right = 1.01, 1.5
        while right - left > 1e-06:
            q = (left + right) / 2.0
            gp = geometricprogression(1, q, src_size // 2)
            if gp > dst_size // 2:
                right = q
            else:
                left = q
        dis, cur = [], 1
        for i in range(src_size // 2):
            dis.append(cur)
            cur += q ** (i + 1)
        r_ids = [(-d) for d in reversed(dis)]
        x = r_ids + [0] + dis
        y = r_ids + [0] + dis
        t = dst_size // 2.0
        dx = np.arange(-t, t + 0.1, 1.0)
        dy = np.arange(-t, t + 0.1, 1.0)
        new_sequence = []
        for i in range(num):
            z = sequence[:, i].view(src_size, src_size).float().numpy()
            f = interpolate.interp2d(x, y, z, kind='cubic')
            new_sequence.append(torch.Tensor(f(dx, dy)).contiguous().view(-1, 1))
        new_sequence = torch.cat(new_sequence, dim=-1)
        return new_sequence
    """resizerelposembed"""

    def resizerelposembed(self, checkpoint):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        all_keys = list(state_dict.keys())
        for key in all_keys:
            if 'relative_position_index' in key:
                state_dict.pop(key)
            if 'relative_position_bias_table' in key:
                rel_pos_bias = state_dict[key]
                src_num_pos, num_attn_heads = rel_pos_bias.size()
                dst_num_pos, _ = self.state_dict()[key].size()
                dst_patch_shape = self.patch_shape
                if dst_patch_shape[0] != dst_patch_shape[1]:
                    raise NotImplementedError()
                num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
                src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
                dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
                if src_size != dst_size:
                    extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                    rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]
                    new_rel_pos_bias = self.geometricsequenceinterpolation(src_size, dst_size, rel_pos_bias, num_attn_heads)
                    new_rel_pos_bias = torch.cat((new_rel_pos_bias, extra_tokens), dim=0)
                    state_dict[key] = new_rel_pos_bias
        return state_dict
    """loadpretrainedweights"""

    def loadpretrainedweights(self, structure_type='beit_base_patch16_224_pt22k_ft22k', pretrained_model_path=''):
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        else:
            checkpoint = model_zoo.load_url(DEFAULT_MODEL_URLS[structure_type], map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        state_dict = self.beitconvert(state_dict)
        state_dict = self.resizerelposembed(state_dict)
        self.load_state_dict(state_dict, strict=False)
    """beitconvert"""

    @staticmethod
    def beitconvert(ckpt):
        from collections import OrderedDict
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            if k.startswith('blocks'):
                new_key = k.replace('blocks', 'layers')
                if 'norm' in new_key:
                    new_key = new_key.replace('norm', 'ln')
                elif 'mlp.fc1' in new_key:
                    new_key = new_key.replace('mlp.fc1', 'ffn.layers.0.0')
                elif 'mlp.fc2' in new_key:
                    new_key = new_key.replace('mlp.fc2', 'ffn.layers.1')
                new_ckpt[new_key] = v
            elif k.startswith('patch_embed'):
                new_key = k.replace('patch_embed.proj', 'patch_embed.projection')
                new_ckpt[new_key] = v
            else:
                new_key = k
                new_ckpt[new_key] = v
        return new_ckpt
    """forward"""

    def forward(self, inputs):
        B = inputs.shape[0]
        x, hw_shape = self.patch_embed(inputs)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1:
                if self.final_norm:
                    x = self.norm1(x)
            if i in self.out_indices:
                out = x[:, 1:]
                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1], C).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return tuple(outs)


class HardSigmoid(nn.Module):

    def __init__(self, bias=1.0, divisor=2.0, min_value=0.0, max_value=1.0):
        super(HardSigmoid, self).__init__()
        assert divisor != 0, 'divisor is not allowed to be equal to zero'
        self.bias = bias
        self.divisor = divisor
        self.min_value = min_value
        self.max_value = max_value
    """forward"""

    def forward(self, x):
        x = (x + self.bias) / self.divisor
        return x.clamp_(self.min_value, self.max_value)


class HardSwish(nn.Module):

    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.act = nn.ReLU6(inplace)
    """forward"""

    def forward(self, x):
        return x * self.act(x + 3) / 6


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
    """forward"""

    def forward(self, x):
        return x * torch.sigmoid(x)


class ActivationBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {'ReLU': nn.ReLU, 'GELU': nn.GELU, 'ReLU6': nn.ReLU6, 'PReLU': nn.PReLU, 'Sigmoid': nn.Sigmoid, 'HardSwish': HardSwish, 'LeakyReLU': nn.LeakyReLU, 'HardSigmoid': HardSigmoid, 'Swish': Swish}
    for act_type in ['ELU', 'Hardshrink', 'Hardtanh', 'LogSigmoid', 'RReLU', 'SELU', 'CELU', 'SiLU', 'GLU', 'Mish', 'Softplus', 'Softshrink', 'Softsign', 'Tanh', 'Tanhshrink', 'Threshold']:
        if hasattr(nn, act_type):
            REGISTERED_MODULES[act_type] = getattr(nn, act_type)
    """build"""

    def build(self, act_cfg):
        if act_cfg is None:
            return nn.Identity()
        return super().build(act_cfg)


BuildActivation = ActivationBuilder().build


class SpatialPath(nn.Module):

    def __init__(self, in_channels=3, num_channels_list=(64, 64, 64, 128), norm_cfg=None, act_cfg=None):
        super(SpatialPath, self).__init__()
        assert len(num_channels_list) == 4
        self.layers = []
        for idx in range(len(num_channels_list)):
            layer_name = f'layer{idx + 1}'
            self.layers.append(layer_name)
            if idx == 0:
                conv = nn.Sequential(nn.Conv2d(in_channels, num_channels_list[idx], kernel_size=7, stride=2, padding=3, bias=False), BuildNormalization(placeholder=num_channels_list[idx], norm_cfg=norm_cfg), BuildActivation(act_cfg))
            elif idx == len(num_channels_list) - 1:
                conv = nn.Sequential(nn.Conv2d(num_channels_list[idx - 1], num_channels_list[idx], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=num_channels_list[idx], norm_cfg=norm_cfg), BuildActivation(act_cfg))
            else:
                conv = nn.Sequential(nn.Conv2d(num_channels_list[idx - 1], num_channels_list[idx], kernel_size=3, stride=2, padding=1, bias=False), BuildNormalization(placeholder=num_channels_list[idx], norm_cfg=norm_cfg), BuildActivation(act_cfg))
            self.add_module(layer_name, conv)
    """forward"""

    def forward(self, x):
        for idx, layer_name in enumerate(self.layers):
            layer_stage = getattr(self, layer_name)
            x = layer_stage(x)
        return x


class AttentionRefinementModule(nn.Module):

    def __init__(self, in_channels, out_channels, norm_cfg=None, act_cfg=None):
        super(AttentionRefinementModule, self).__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.atten_conv_layer = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), nn.Sigmoid())
    """forward"""

    def forward(self, x):
        x = self.conv_layer(x)
        x_atten = self.atten_conv_layer(x)
        x_out = x * x_atten
        return x_out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_cfg=None, act_cfg=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = BuildNormalization(placeholder=planes, norm_cfg=norm_cfg)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BuildNormalization(placeholder=planes, norm_cfg=norm_cfg)
        self.relu = BuildActivation(act_cfg)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
    """forward"""

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_cfg=None, act_cfg=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = BuildNormalization(placeholder=planes, norm_cfg=norm_cfg)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = BuildNormalization(placeholder=planes, norm_cfg=norm_cfg)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = BuildNormalization(placeholder=planes * self.expansion, norm_cfg=norm_cfg)
        self.relu = BuildActivation(act_cfg)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
    """forward"""

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def loadpretrainedweights(structure_type, pretrained_model_path='', default_model_urls={}, map_to_cpu=True, possible_model_keys=['model', 'state_dict']):
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path, map_location='cpu') if map_to_cpu else torch.load(pretrained_model_path)
    else:
        checkpoint = model_zoo.load_url(default_model_urls[structure_type], map_location='cpu') if map_to_cpu else model_zoo.load_url(default_model_urls[structure_type])
    state_dict = checkpoint
    for key in possible_model_keys:
        if key in checkpoint:
            state_dict = checkpoint[key]
            break
    return state_dict


class ResNet(nn.Module):
    arch_settings = {(18): (BasicBlock, (2, 2, 2, 2)), (34): (BasicBlock, (3, 4, 6, 3)), (50): (Bottleneck, (3, 4, 6, 3)), (101): (Bottleneck, (3, 4, 23, 3)), (152): (Bottleneck, (3, 8, 36, 3))}

    def __init__(self, structure_type, in_channels=3, base_channels=64, stem_channels=64, depth=101, outstride=8, contract_dilation=True, use_conv3x3_stem=True, out_indices=(0, 1, 2, 3), use_avg_for_downsample=False, norm_cfg={'type': 'SyncBatchNorm'}, act_cfg={'type': 'ReLU', 'inplace': True}, pretrained=True, pretrained_model_path=''):
        super(ResNet, self).__init__()
        self.structure_type = structure_type
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.stem_channels = stem_channels
        self.depth = depth
        self.outstride = outstride
        self.contract_dilation = contract_dilation
        self.use_conv3x3_stem = use_conv3x3_stem
        self.out_indices = out_indices
        self.use_avg_for_downsample = use_avg_for_downsample
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and getattr(self, key) == value
        self.inplanes = stem_channels
        assert depth in self.arch_settings, 'unsupport depth %s' % depth
        block, num_blocks_list = self.arch_settings[depth]
        outstride_to_strides_and_dilations = {(8): ((1, 2, 1, 1), (1, 1, 2, 4)), (16): ((1, 2, 2, 1), (1, 1, 1, 2)), (32): ((1, 2, 2, 2), (1, 1, 1, 1))}
        assert outstride in outstride_to_strides_and_dilations, 'unsupport outstride %s' % outstride
        stride_list, dilation_list = outstride_to_strides_and_dilations[outstride]
        if use_conv3x3_stem:
            self.stem = nn.Sequential(nn.Conv2d(in_channels, stem_channels // 2, kernel_size=3, stride=2, padding=1, bias=False), BuildNormalization(placeholder=stem_channels // 2, norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Conv2d(stem_channels // 2, stem_channels // 2, kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=stem_channels // 2, norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Conv2d(stem_channels // 2, stem_channels, kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=stem_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
        else:
            self.conv1 = nn.Conv2d(in_channels, stem_channels, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = BuildNormalization(placeholder=stem_channels, norm_cfg=norm_cfg)
            self.relu = BuildActivation(act_cfg)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.makelayer(block=block, inplanes=stem_channels, planes=base_channels, num_blocks=num_blocks_list[0], stride=stride_list[0], dilation=dilation_list[0], contract_dilation=contract_dilation, use_avg_for_downsample=use_avg_for_downsample, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.layer2 = self.makelayer(block=block, inplanes=base_channels * 4 if depth >= 50 else base_channels, planes=base_channels * 2, num_blocks=num_blocks_list[1], stride=stride_list[1], dilation=dilation_list[1], contract_dilation=contract_dilation, use_avg_for_downsample=use_avg_for_downsample, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.layer3 = self.makelayer(block=block, inplanes=base_channels * 8 if depth >= 50 else base_channels * 2, planes=base_channels * 4, num_blocks=num_blocks_list[2], stride=stride_list[2], dilation=dilation_list[2], contract_dilation=contract_dilation, use_avg_for_downsample=use_avg_for_downsample, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.layer4 = self.makelayer(block=block, inplanes=base_channels * 16 if depth >= 50 else base_channels * 4, planes=base_channels * 8, num_blocks=num_blocks_list[3], stride=stride_list[3], dilation=dilation_list[3], contract_dilation=contract_dilation, use_avg_for_downsample=use_avg_for_downsample, norm_cfg=norm_cfg, act_cfg=act_cfg)
        if pretrained:
            state_dict = loadpretrainedweights(structure_type=structure_type, pretrained_model_path=pretrained_model_path, default_model_urls=DEFAULT_MODEL_URLS)
            self.load_state_dict(state_dict, strict=False)
    """makelayer"""

    def makelayer(self, block, inplanes, planes, num_blocks, stride=1, dilation=1, contract_dilation=True, use_avg_for_downsample=False, norm_cfg=None, act_cfg=None):
        downsample = None
        dilations = [dilation] * num_blocks
        if contract_dilation and dilation > 1:
            dilations[0] = dilation // 2
        if stride != 1 or inplanes != planes * block.expansion:
            if use_avg_for_downsample:
                downsample = nn.Sequential(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False), nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=planes * block.expansion, norm_cfg=norm_cfg))
            else:
                downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, padding=0, bias=False), BuildNormalization(placeholder=planes * block.expansion, norm_cfg=norm_cfg))
        layers = []
        layers.append(block(inplanes, planes, stride=stride, dilation=dilations[0], downsample=downsample, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(planes * block.expansion, planes, stride=1, dilation=dilations[i], norm_cfg=norm_cfg, act_cfg=act_cfg))
        return nn.Sequential(*layers)
    """forward"""

    def forward(self, x):
        if self.use_conv3x3_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        outs = []
        for i, feats in enumerate([x1, x2, x3, x4]):
            if i in self.out_indices:
                outs.append(feats)
        return tuple(outs)


class ContextPath(nn.Module):

    def __init__(self, backbone_cfg, context_channels_list=(128, 256, 512), norm_cfg=None, act_cfg=None):
        super(ContextPath, self).__init__()
        assert len(context_channels_list) == 3
        if 'norm_cfg' not in backbone_cfg:
            backbone_cfg['norm_cfg'] = norm_cfg
        self.backbone_net = self.buildbackbone(backbone_cfg)
        self.arm16 = AttentionRefinementModule(context_channels_list[1], context_channels_list[0], norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.arm32 = AttentionRefinementModule(context_channels_list[2], context_channels_list[0], norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv_head32 = nn.Sequential(nn.Conv2d(context_channels_list[0], context_channels_list[0], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=context_channels_list[0], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.conv_head16 = nn.Sequential(nn.Conv2d(context_channels_list[0], context_channels_list[0], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=context_channels_list[0], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.gap_conv = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(context_channels_list[2], context_channels_list[0], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=context_channels_list[0], norm_cfg=norm_cfg), BuildActivation(act_cfg))
    """forward"""

    def forward(self, x):
        x_4, x_8, x_16, x_32 = self.backbone_net(x)
        x_gap = self.gap_conv(x_32)
        x_32_arm = self.arm32(x_32)
        x_32_sum = x_32_arm + x_gap
        x_32_up = F.interpolate(input=x_32_sum, size=x_16.shape[2:], mode='nearest')
        x_32_up = self.conv_head32(x_32_up)
        x_16_arm = self.arm16(x_16)
        x_16_sum = x_16_arm + x_32_up
        x_16_up = F.interpolate(input=x_16_sum, size=x_8.shape[2:], mode='nearest')
        x_16_up = self.conv_head16(x_16_up)
        return x_16_up, x_32_up
    """buildbackbone"""

    def buildbackbone(self, cfg):
        supported_backbones = {'ResNet': ResNet}
        backbone_type = cfg.pop('type')
        assert backbone_type, f'unsupport backbone type {backbone_type}'
        return supported_backbones[backbone_type](**cfg)


class FeatureFusionModule(nn.Module):

    def __init__(self, higher_in_channels, lower_in_channels, out_channels, norm_cfg=None, dwconv_act_cfg=None, conv_act_cfg=None, align_corners=False):
        super(FeatureFusionModule, self).__init__()
        self.norm_cfg = norm_cfg
        self.dwconv_act_cfg = dwconv_act_cfg
        self.conv_act_cfg = conv_act_cfg
        self.align_corners = align_corners
        self.dwconv = nn.Sequential(nn.Conv2d(lower_in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(dwconv_act_cfg))
        self.conv_lower_res = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg))
        self.conv_higher_res = nn.Sequential(nn.Conv2d(higher_in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg))
        self.act = BuildActivation(conv_act_cfg)
    """forward"""

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(lower_res_feature, size=higher_res_feature.size()[2:], mode='bilinear', align_corners=self.align_corners)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)
        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.act(out)


class BiSeNetV1(nn.Module):

    def __init__(self, structure_type, backbone_cfg=None, in_channels=3, spatial_channels_list=(64, 64, 64, 128), context_channels_list=(128, 256, 512), out_indices=(0, 1, 2), out_channels=256, norm_cfg={'type': 'SyncBatchNorm'}, act_cfg={'type': 'ReLU', 'inplace': True}, pretrained=False, pretrained_model_path=''):
        super(BiSeNetV1, self).__init__()
        assert len(spatial_channels_list) == 4 and len(context_channels_list) == 3
        self.structure_type = structure_type
        self.backbone_cfg = backbone_cfg
        self.in_channels = in_channels
        self.spatial_channels_list = spatial_channels_list
        self.context_channels_list = context_channels_list
        self.out_indices = out_indices
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and getattr(self, key) == value
        self.context_path = ContextPath(backbone_cfg, context_channels_list, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.spatial_path = SpatialPath(in_channels, spatial_channels_list, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.ffm = FeatureFusionModule(context_channels_list[1], out_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        if pretrained:
            state_dict = loadpretrainedweights(structure_type=structure_type, pretrained_model_path=pretrained_model_path, default_model_urls=DEFAULT_MODEL_URLS)
            self.load_state_dict(state_dict, strict=False)
    """forward"""

    def forward(self, x):
        x_context8, x_context16 = self.context_path(x)
        x_spatial = self.spatial_path(x)
        x_fuse = self.ffm(x_spatial, x_context8)
        outs = [x_context8, x_context16, x_fuse]
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)


class DetailBranch(nn.Module):

    def __init__(self, detail_channels=(64, 64, 128), in_channels=3, norm_cfg=None, act_cfg=None):
        super(DetailBranch, self).__init__()
        detail_branch = []
        for i in range(len(detail_channels)):
            if i == 0:
                detail_branch.append(nn.Sequential(nn.Conv2d(in_channels, detail_channels[i], kernel_size=3, stride=2, padding=1, bias=False), BuildNormalization(placeholder=detail_channels[i], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Conv2d(detail_channels[i], detail_channels[i], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=detail_channels[i], norm_cfg=norm_cfg), BuildActivation(act_cfg)))
            else:
                detail_branch.append(nn.Sequential(nn.Conv2d(detail_channels[i - 1], detail_channels[i], kernel_size=3, stride=2, padding=1, bias=False), BuildNormalization(placeholder=detail_channels[i], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Conv2d(detail_channels[i], detail_channels[i], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=detail_channels[i], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Conv2d(detail_channels[i], detail_channels[i], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=detail_channels[i], norm_cfg=norm_cfg), BuildActivation(act_cfg)))
        self.detail_branch = nn.ModuleList(detail_branch)
    """forward"""

    def forward(self, x):
        for stage in self.detail_branch:
            x = stage(x)
        return x


class StemBlock(nn.Module):

    def __init__(self, in_channels=3, out_channels=16, norm_cfg=None, act_cfg=None):
        super(StemBlock, self).__init__()
        self.conv_first = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.convs = nn.Sequential(nn.Conv2d(out_channels, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=out_channels // 2, norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse_last = nn.Sequential(nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
    """forward"""

    def forward(self, x):
        x = self.conv_first(x)
        x_left = self.convs(x)
        x_right = self.pool(x)
        x = self.fuse_last(torch.cat([x_left, x_right], dim=1))
        return x


class DepthwiseSeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, norm_cfg=None, act_cfg=None, dw_norm_cfg=None, dw_act_cfg=None, pw_norm_cfg=None, pw_act_cfg=None):
        super(DepthwiseSeparableConv2d, self).__init__()
        if dw_norm_cfg is None:
            dw_norm_cfg = norm_cfg
        if dw_act_cfg is None:
            dw_act_cfg = act_cfg
        if pw_norm_cfg is None:
            pw_norm_cfg = norm_cfg
        if pw_act_cfg is None:
            pw_act_cfg = act_cfg
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        if dw_norm_cfg is not None:
            self.depthwise_bn = BuildNormalization(placeholder=in_channels, norm_cfg=dw_norm_cfg)
        if dw_act_cfg is not None:
            self.depthwise_activate = BuildActivation(dw_act_cfg)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=bias)
        if pw_norm_cfg is not None:
            self.pointwise_bn = BuildNormalization(placeholder=out_channels, norm_cfg=pw_norm_cfg)
        if pw_act_cfg is not None:
            self.pointwise_activate = BuildActivation(pw_act_cfg)
    """forward"""

    def forward(self, x):
        x = self.depthwise_conv(x)
        if hasattr(self, 'depthwise_bn'):
            x = self.depthwise_bn(x)
        if hasattr(self, 'depthwise_activate'):
            x = self.depthwise_activate(x)
        x = self.pointwise_conv(x)
        if hasattr(self, 'pointwise_bn'):
            x = self.pointwise_bn(x)
        if hasattr(self, 'pointwise_activate'):
            x = self.pointwise_activate(x)
        return x


class GELayer(nn.Module):

    def __init__(self, in_channels, out_channels, exp_ratio=6, stride=1, norm_cfg=None, act_cfg=None):
        super(GELayer, self).__init__()
        mid_channel = in_channels * exp_ratio
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=in_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
        if stride == 1:
            self.dwconv = nn.Sequential(nn.Conv2d(in_channels, mid_channel, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False), BuildNormalization(placeholder=mid_channel, norm_cfg=norm_cfg), BuildActivation(act_cfg))
            self.shortcut = None
        else:
            self.dwconv = nn.Sequential(nn.Conv2d(in_channels, mid_channel, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False), BuildNormalization(placeholder=mid_channel, norm_cfg=norm_cfg), nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1, groups=mid_channel, bias=False), BuildNormalization(placeholder=mid_channel, norm_cfg=norm_cfg), BuildActivation(act_cfg))
            self.shortcut = nn.Sequential(DepthwiseSeparableConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, dw_norm_cfg=norm_cfg, dw_act_cfg=None, pw_norm_cfg=norm_cfg, pw_act_cfg=None))
        self.conv2 = nn.Sequential(nn.Conv2d(mid_channel, out_channels, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg))
        self.act = BuildActivation(act_cfg)
    """forward"""

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.conv2(x)
        if self.shortcut is not None:
            shortcut = self.shortcut(identity)
            x = x + shortcut
        else:
            x = x + identity
        x = self.act(x)
        return x


class CEBlock(nn.Module):

    def __init__(self, in_channels=3, out_channels=16, norm_cfg=None, act_cfg=None):
        super(CEBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), BuildNormalization(placeholder=in_channels, norm_cfg=norm_cfg))
        self.conv_gap = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.conv_last = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
    """forward"""

    def forward(self, x):
        identity = x
        x = self.gap(x)
        x = self.conv_gap(x)
        x = identity + x
        x = self.conv_last(x)
        return x


class SemanticBranch(nn.Module):

    def __init__(self, semantic_channels=(16, 32, 64, 128), in_channels=3, exp_ratio=6, norm_cfg=None, act_cfg=None):
        super(SemanticBranch, self).__init__()
        self.in_channels = in_channels
        self.semantic_channels = semantic_channels
        self.semantic_stages = []
        for i in range(len(semantic_channels)):
            stage_name = f'stage{i + 1}'
            self.semantic_stages.append(stage_name)
            if i == 0:
                self.add_module(stage_name, StemBlock(in_channels, semantic_channels[i], norm_cfg=norm_cfg, act_cfg=act_cfg))
            elif i == len(semantic_channels) - 1:
                self.add_module(stage_name, nn.Sequential(GELayer(semantic_channels[i - 1], semantic_channels[i], exp_ratio, 2, norm_cfg=norm_cfg, act_cfg=act_cfg), GELayer(semantic_channels[i], semantic_channels[i], exp_ratio, 1, norm_cfg=norm_cfg, act_cfg=act_cfg), GELayer(semantic_channels[i], semantic_channels[i], exp_ratio, 1, norm_cfg=norm_cfg, act_cfg=act_cfg), GELayer(semantic_channels[i], semantic_channels[i], exp_ratio, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)))
            else:
                self.add_module(stage_name, nn.Sequential(GELayer(semantic_channels[i - 1], semantic_channels[i], exp_ratio, 2, norm_cfg=norm_cfg, act_cfg=act_cfg), GELayer(semantic_channels[i], semantic_channels[i], exp_ratio, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)))
        self.add_module(f'stage{len(semantic_channels)}_CEBlock', CEBlock(semantic_channels[-1], semantic_channels[-1], norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.semantic_stages.append(f'stage{len(semantic_channels)}_CEBlock')
    """forward"""

    def forward(self, x):
        semantic_outs = []
        for stage_name in self.semantic_stages:
            semantic_stage = getattr(self, stage_name)
            x = semantic_stage(x)
            semantic_outs.append(x)
        return semantic_outs


class BGALayer(nn.Module):

    def __init__(self, out_channels=128, align_corners=False, norm_cfg=None, act_cfg=None):
        super(BGALayer, self).__init__()
        self.out_channels = out_channels
        self.align_corners = align_corners
        self.detail_dwconv = nn.Sequential(DepthwiseSeparableConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dw_norm_cfg=norm_cfg, dw_act_cfg=None, pw_norm_cfg=None, pw_act_cfg=None))
        self.detail_down = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
        self.semantic_conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg))
        self.semantic_dwconv = nn.Sequential(DepthwiseSeparableConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dw_norm_cfg=norm_cfg, dw_act_cfg=None, pw_norm_cfg=None, pw_act_cfg=None))
        self.conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
    """forward"""

    def forward(self, x_d, x_s):
        detail_dwconv = self.detail_dwconv(x_d)
        detail_down = self.detail_down(x_d)
        semantic_conv = self.semantic_conv(x_s)
        semantic_dwconv = self.semantic_dwconv(x_s)
        semantic_conv = F.interpolate(semantic_conv, size=detail_dwconv.shape[2:], mode='bilinear', align_corners=self.align_corners)
        fuse_1 = detail_dwconv * torch.sigmoid(semantic_conv)
        fuse_2 = detail_down * torch.sigmoid(semantic_dwconv)
        fuse_2 = F.interpolate(fuse_2, size=fuse_1.shape[2:], mode='bilinear', align_corners=self.align_corners)
        output = self.conv(fuse_1 + fuse_2)
        return output


class BiSeNetV2(nn.Module):

    def __init__(self, structure_type, in_channels=3, detail_channels=(64, 64, 128), semantic_channels=(16, 32, 64, 128), semantic_expansion_ratio=6, bga_channels=128, out_indices=(0, 1, 2, 3, 4), align_corners=False, norm_cfg={'type': 'SyncBatchNorm'}, act_cfg={'type': 'ReLU', 'inplace': True}, pretrained=False, pretrained_model_path=''):
        super(BiSeNetV2, self).__init__()
        self.structure_type = structure_type
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.detail_channels = detail_channels
        self.semantic_channels = semantic_channels
        self.semantic_expansion_ratio = semantic_expansion_ratio
        self.bga_channels = bga_channels
        self.align_corners = align_corners
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and getattr(self, key) == value
        self.detail = DetailBranch(self.detail_channels, self.in_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.semantic = SemanticBranch(self.semantic_channels, self.in_channels, self.semantic_expansion_ratio, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.bga = BGALayer(self.bga_channels, self.align_corners, norm_cfg=norm_cfg, act_cfg=act_cfg)
        if pretrained:
            state_dict = loadpretrainedweights(structure_type=structure_type, pretrained_model_path=pretrained_model_path, default_model_urls=DEFAULT_MODEL_URLS)
            self.load_state_dict(state_dict, strict=False)
    """forward"""

    def forward(self, x):
        x_detail = self.detail(x)
        x_semantic_lst = self.semantic(x)
        x_head = self.bga(x_detail, x_semantic_lst[-1])
        outs = x_semantic_lst[:-1] + [x_head]
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)


class AdptivePaddingConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, norm_cfg=None, act_cfg=None):
        super(AdptivePaddingConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)
        if norm_cfg is not None:
            self.norm = BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg)
        if act_cfg is not None:
            self.activation = BuildActivation(act_cfg)
    """forward"""

    def forward(self, x):
        img_h, img_w = x.size()[-2:]
        kernel_h, kernel_w = self.weight.size()[-2:]
        stride_h, stride_w = self.stride
        output_h = math.ceil(img_h / stride_h)
        output_w = math.ceil(img_w / stride_w)
        pad_h = max((output_h - 1) * self.stride[0] + (kernel_h - 1) * self.dilation[0] + 1 - img_h, 0)
        pad_w = max((output_w - 1) * self.stride[1] + (kernel_w - 1) * self.dilation[1] + 1 - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if hasattr(self, 'norm'):
            output = self.norm(output)
        if hasattr(self, 'activation'):
            output = self.activation(output)
        return output


class Attention2d(nn.Module):

    def __init__(self, in_channels, out_channels, temperature):
        super(Attention2d, self).__init__()
        assert temperature % 3 == 1
        self.temperature = temperature
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.convs = nn.Sequential(nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1, padding=0, bias=False), nn.ReLU(inplace=True), nn.Conv2d(in_channels // 4, out_channels, kernel_size=1, stride=1, padding=0))
    """update"""

    def update(self):
        if self.temperature != 1:
            self.temperature -= 3
    """forward"""

    def forward(self, x):
        x = self.avgpool(x)
        x = self.convs(x).view(x.size(0), -1)
        return F.softmax(x / self.temperature, 1)


class DynamicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4, temperature=34, norm_cfg=None, act_cfg=None):
        super(DynamicConv2d, self).__init__()
        assert in_channels % groups == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = None
        self.K = K
        self.temperature = temperature
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.attention = Attention2d(in_channels, K, temperature)
        self.weight = nn.Parameter(torch.randn(K, out_channels, in_channels // groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.randn(K, out_channels))
        if norm_cfg is not None:
            self.norm = BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg)
        if act_cfg is not None:
            self.activation = BuildActivation(act_cfg)
    """update"""

    def update(self):
        self.attention.update()
    """forward"""

    def forward(self, x):
        batch_size, in_channels, h, w = x.size()
        softmax_attention = self.attention(x)
        x = x.view(1, -1, h, w)
        weight = self.weight.view(self.K, -1)
        aggregate_weight = torch.mm(softmax_attention, weight)
        aggregate_weight = aggregate_weight.view(-1, in_channels, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(input=x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups * batch_size)
        else:
            output = F.conv2d(input=x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
        if hasattr(self, 'norm'):
            output = self.norm(output)
        if hasattr(self, 'activation'):
            output = self.activation(output)
        return output


class InvertedResidual(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expand_ratio, dilation=1, norm_cfg=None, act_cfg=None):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2], 'stride must in [1, 2], but received %s' % stride
        self.use_res_connect = stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))
        layers = []
        if expand_ratio != 1:
            layer = nn.Sequential()
            layer.add_module('conv', nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False))
            if norm_cfg is not None:
                layer.add_module('bn', BuildNormalization(placeholder=hidden_dim, norm_cfg=norm_cfg))
            if act_cfg is not None:
                layer.add_module('activation', BuildActivation(act_cfg))
            layers.append(layer)
        layer = nn.Sequential()
        layer.add_module('conv', nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=hidden_dim, bias=False))
        if norm_cfg is not None:
            layer.add_module('bn', BuildNormalization(placeholder=hidden_dim, norm_cfg=norm_cfg))
        if act_cfg is not None:
            layer.add_module('activation', BuildActivation(act_cfg))
        layers.extend([layer])
        layer = nn.Sequential()
        layer.add_module('conv', nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        if norm_cfg is not None:
            layer.add_module('bn', BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg))
        layers.extend([layer])
        self.conv = nn.Sequential(*layers)
    """forward"""

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def makedivisible(value, divisor, min_value=None, min_ratio=0.9):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value


class SqueezeExcitationConv2d(nn.Module):

    def __init__(self, channels, ratio=16, act_cfgs=None, makedivisible_args={'divisor': 8}):
        super(SqueezeExcitationConv2d, self).__init__()
        assert act_cfgs is not None, 'argument act_cfgs should be given'
        assert len(act_cfgs) == 2, 'length of act_cfgs should be equal to 2'
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        act_cfg = act_cfgs[0]
        self.conv1 = nn.Sequential()
        self.conv1.add_module('conv', nn.Conv2d(channels, makedivisible(channels // ratio, **makedivisible_args), kernel_size=1, stride=1, padding=0))
        self.conv1.add_module('activation', BuildActivation(act_cfg))
        act_cfg = act_cfgs[1]
        self.conv2 = nn.Sequential()
        self.conv2.add_module('conv', nn.Conv2d(makedivisible(channels // ratio, **makedivisible_args), channels, kernel_size=1, stride=1, padding=0))
        self.conv2.add_module('activation', BuildActivation(act_cfg))
    """forward"""

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out


class InvertedResidualV3(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels, kernel_size=3, stride=1, se_cfg=None, with_expand_conv=True, norm_cfg=None, act_cfg=None):
        super(InvertedResidualV3, self).__init__()
        assert stride in [1, 2], 'stride must in [1, 2], but received %s' % stride
        self.with_res_shortcut = stride == 1 and in_channels == out_channels
        self.with_expand_conv = with_expand_conv
        if not self.with_expand_conv:
            assert mid_channels == in_channels
        if self.with_expand_conv:
            self.expand_conv = nn.Sequential()
            self.expand_conv.add_module('conv', nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False))
            if norm_cfg is not None:
                self.expand_conv.add_module('bn', BuildNormalization(placeholder=mid_channels, norm_cfg=norm_cfg))
            if act_cfg is not None:
                self.expand_conv.add_module('activation', BuildActivation(act_cfg))
        self.depthwise_conv = nn.Sequential()
        if stride == 2:
            self.depthwise_conv.add_module('conv', AdptivePaddingConv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=mid_channels, bias=False))
            if norm_cfg is not None:
                self.depthwise_conv.add_module('bn', BuildNormalization(placeholder=mid_channels, norm_cfg=norm_cfg))
            if act_cfg is not None:
                self.depthwise_conv.add_module('activation', BuildActivation(act_cfg))
        else:
            self.depthwise_conv.add_module('conv', nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=mid_channels, bias=False))
            if norm_cfg is not None:
                self.depthwise_conv.add_module('bn', BuildNormalization(placeholder=mid_channels, norm_cfg=norm_cfg))
            if act_cfg is not None:
                self.depthwise_conv.add_module('activation', BuildActivation(act_cfg))
        if se_cfg is not None:
            self.se = SqueezeExcitationConv2d(**se_cfg)
        self.linear_conv = nn.Sequential()
        self.linear_conv.add_module('conv', nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        if norm_cfg is not None:
            self.linear_conv.add_module('bn', BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg))
    """forward"""

    def forward(self, x):
        out = x
        if self.with_expand_conv:
            out = self.expand_conv(out)
        out = self.depthwise_conv(out)
        if hasattr(self, 'se'):
            out = self.se(out)
        out = self.linear_conv(out)
        if self.with_res_shortcut:
            return x + out
        return out


class L2Norm(nn.Module):

    def __init__(self, channels, scale=10, eps=1e-10):
        super(L2Norm, self).__init__()
        self.channels, self.eps = channels, eps
        self.weight = nn.Parameter(torch.Tensor(channels))
        nn.init.constant_(self.weight, scale)
    """forward"""

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class Scale(nn.Module):

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))
    """forward"""

    def forward(self, x):
        return x * self.scale


class AdaptivePadding(nn.Module):

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):
        super(AdaptivePadding, self).__init__()
        assert padding in ('same', 'corner')
        self.padding = padding
        self.kernel_size = self.totuple(kernel_size)
        self.stride = self.totuple(stride)
        self.dilation = self.totuple(dilation)
    """getpadshape"""

    def getpadshape(self, input_shape):
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w
    """forward"""

    def forward(self, x):
        pad_h, pad_w = self.getpadshape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return x
    """totuple"""

    @staticmethod
    def totuple(x):
        if isinstance(x, int):
            return x, x
        assert isinstance(x, collections.abc.Sequence) and len(x) == 2
        for n in x:
            assert isinstance(n, int)
        return tuple(x)


class Conv2dBN(nn.Sequential):

    def __init__(self, in_chans, out_chans, kernel_size=1, stride=1, padding=0, dilation=1, groups=1):
        super(Conv2dBN, self).__init__()
        self.add_module('c', nn.Conv2d(in_chans, out_chans, kernel_size, stride, padding, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(out_chans))


class PatchMerging(nn.Module):

    def __init__(self, input_resolution, dim, out_dim, act_cfg={'type': 'GELU'}):
        super(PatchMerging, self).__init__()
        self.dim = dim
        self.act = BuildActivation(act_cfg=act_cfg)
        self.out_dim = out_dim
        self.input_resolution = PatchEmbed.totuple(input_resolution)
        self.conv1 = Conv2dBN(dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.conv2 = Conv2dBN(out_dim, out_dim, kernel_size=3, stride=1 if out_dim == 320 or out_dim == 448 or out_dim == 576 else 2, padding=1, groups=out_dim)
        self.conv3 = Conv2dBN(out_dim, out_dim, kernel_size=1, stride=1, padding=0)
    """forward"""

    def forward(self, x):
        if x.ndim == 3:
            x = x.view(x.shape[0], self.input_resolution[0], self.input_resolution[1], -1).permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class FFN(nn.Module):

    def __init__(self, embed_dims=256, feedforward_channels=1024, num_fcs=2, act_cfg=None, ffn_drop=0.0, dropout_cfg=None, add_identity=True):
        super(FFN, self).__init__()
        assert num_fcs >= 2, f'num_fcs should be no less than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = BuildActivation(act_cfg)
        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(nn.Sequential(nn.Linear(in_channels, feedforward_channels), self.activate, nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        if dropout_cfg:
            self.dropout_layer = BuildDropout(dropout_cfg)
        else:
            self.dropout_layer = torch.nn.Identity()
        self.add_identity = add_identity
    """forward"""

    def forward(self, x, identity=None):
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dims, num_heads, attn_drop=0.0, proj_drop=0.0, dropout_cfg=None, batch_first=False, **kwargs):
        super(MultiheadAttention, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop, **kwargs)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = BuildDropout(dropout_cfg) if dropout_cfg else nn.Identity()
    """forward"""

    def forward(self, query, key=None, value=None, identity=None, query_pos=None, key_pos=None, attn_mask=None, key_padding_mask=None):
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        out = self.attn(query=query, key=key, value=value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]
        if self.batch_first:
            out = out.transpose(0, 1)
        return identity + self.dropout_layer(self.proj_drop(out))


class PositionEmbeddingSine(nn.Module):

    def __init__(self, num_pos_feats, temperature=10000, normalize=True, scale=None):
        super(PositionEmbeddingSine, self).__init__()
        assert num_pos_feats % 2 == 0, 'Expecting even model width'
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.cache = {}
    """encodexy"""

    def encodexy(self, x, y):
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        return pos_x, pos_y
    """encodeboxes"""

    @torch.no_grad()
    def encodeboxes(self, x, y, w, h):
        pos_x, pos_y = self.encodexy(x, y)
        pos = torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)
        return pos
    """Backwards compatibility"""
    encode = encodeboxes
    """encodepoints"""

    @torch.no_grad()
    def encodepoints(self, x, y, labels):
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self.encodexy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
        pos = torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)
        return pos
    """forward"""

    @torch.no_grad()
    def forward(self, x):
        cache_key = x.shape[-2], x.shape[-1]
        if cache_key in self.cache:
            return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)
        y_embed = torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device).view(1, -1, 1).repeat(x.shape[0], 1, x.shape[-1])
        x_embed = torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device).view(1, 1, -1).repeat(x.shape[0], x.shape[-2], 1)
        if self.normalize:
            eps = 1e-06
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.cache[cache_key] = pos[0]
        return pos


class GlobalContextExtractor(nn.Module):

    def __init__(self, channels, reduction=16):
        super(GlobalContextExtractor, self).__init__()
        assert reduction >= 1 and channels >= reduction
        self.channels = channels
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, channels // reduction), nn.ReLU(inplace=True), nn.Linear(channels // reduction, channels), nn.Sigmoid())
    """forward"""

    def forward(self, x):
        batch_size, num_channels = x.size()[:2]
        y = self.avg_pool(x).view(batch_size, num_channels)
        y = self.fc(y).view(batch_size, num_channels, 1, 1)
        return x * y


class ContextGuidedBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dilation=2, reduction=16, skip_connect=True, downsample=False, norm_cfg=None, act_cfg=None):
        super(ContextGuidedBlock, self).__init__()
        self.downsample = downsample
        self.skip_connect = skip_connect and not downsample
        channels = out_channels if downsample else out_channels // 2
        if 'type' in act_cfg and act_cfg['type'] == 'PReLU':
            act_cfg['num_parameters'] = channels
        kernel_size = 3 if downsample else 1
        stride = 2 if downsample else 1
        padding = (kernel_size - 1) // 2
        self.conv1x1 = nn.Sequential(nn.Conv2d(in_channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False), BuildNormalization(placeholder=channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.f_loc = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False)
        self.f_sur = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, groups=channels, bias=False)
        self.bn = BuildNormalization(placeholder=channels * 2, norm_cfg=norm_cfg)
        self.activate = nn.PReLU(2 * channels)
        if downsample:
            self.bottleneck = nn.Conv2d(2 * channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.f_glo = GlobalContextExtractor(out_channels, reduction)
    """forward"""

    def forward(self, x):
        out = self.conv1x1(x)
        loc = self.f_loc(out)
        sur = self.f_sur(out)
        joi_feat = torch.cat([loc, sur], 1)
        joi_feat = self.bn(joi_feat)
        joi_feat = self.activate(joi_feat)
        if self.downsample:
            joi_feat = self.bottleneck(joi_feat)
        out = self.f_glo(joi_feat)
        if self.skip_connect:
            return x + out
        return out


class InputInjection(nn.Module):

    def __init__(self, num_downsamplings):
        super(InputInjection, self).__init__()
        self.pools = nn.ModuleList()
        for _ in range(num_downsamplings):
            self.pools.append(nn.AvgPool2d(3, stride=2, padding=1))
    """forward"""

    def forward(self, x):
        for pool in self.pools:
            x = pool(x)
        return x


class CGNet(nn.Module):

    def __init__(self, structure_type, in_channels=3, num_channels=(32, 64, 128), num_blocks=(3, 21), dilations=(2, 4), reductions=(8, 16), norm_cfg={'type': 'SyncBatchNorm'}, act_cfg={'type': 'PReLU'}, pretrained=False, pretrained_model_path=''):
        super(CGNet, self).__init__()
        self.structure_type = structure_type
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.dilations = dilations
        self.reductions = reductions
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        if 'type' in self.act_cfg and self.act_cfg['type'] == 'PReLU':
            self.act_cfg['num_parameters'] = num_channels[0]
        assert isinstance(num_channels, tuple) and len(num_channels) == 3
        assert isinstance(num_blocks, tuple) and len(num_blocks) == 2
        assert isinstance(dilations, tuple) and len(dilations) == 2
        assert isinstance(reductions, tuple) and len(reductions) == 2
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and getattr(self, key) == value
        cur_channels = in_channels
        self.stem = nn.ModuleList()
        for i in range(3):
            self.stem.append(nn.Sequential(nn.Conv2d(cur_channels, num_channels[0], kernel_size=3, stride=2 if i == 0 else 1, padding=1, bias=False), BuildNormalization(placeholder=num_channels[0], norm_cfg=norm_cfg), BuildActivation(act_cfg)))
            cur_channels = num_channels[0]
        self.inject_2x = InputInjection(1)
        self.inject_4x = InputInjection(2)
        cur_channels += in_channels
        self.norm_prelu_0 = nn.Sequential(BuildNormalization(placeholder=cur_channels, norm_cfg=norm_cfg), nn.PReLU(cur_channels))
        self.level1 = nn.ModuleList()
        for i in range(num_blocks[0]):
            self.level1.append(ContextGuidedBlock(in_channels=cur_channels if i == 0 else num_channels[1], out_channels=num_channels[1], dilation=dilations[0], reduction=reductions[0], skip_connect=True, downsample=i == 0, norm_cfg=norm_cfg, act_cfg=act_cfg))
        cur_channels = 2 * num_channels[1] + in_channels
        self.norm_prelu_1 = nn.Sequential(BuildNormalization(placeholder=cur_channels, norm_cfg=norm_cfg), nn.PReLU(cur_channels))
        self.level2 = nn.ModuleList()
        for i in range(num_blocks[1]):
            self.level2.append(ContextGuidedBlock(in_channels=cur_channels if i == 0 else num_channels[2], out_channels=num_channels[2], dilation=dilations[1], reduction=reductions[1], skip_connect=True, downsample=i == 0, norm_cfg=norm_cfg, act_cfg=act_cfg))
        cur_channels = 2 * num_channels[2]
        self.norm_prelu_2 = nn.Sequential(BuildNormalization(placeholder=cur_channels, norm_cfg=norm_cfg), nn.PReLU(cur_channels))
        if pretrained:
            state_dict = loadpretrainedweights(structure_type=structure_type, pretrained_model_path=pretrained_model_path, default_model_urls=DEFAULT_MODEL_URLS)
            self.load_state_dict(state_dict, strict=False)
    """forward"""

    def forward(self, x):
        output = []
        inp_2x = self.inject_2x(x)
        inp_4x = self.inject_4x(x)
        for layer in self.stem:
            x = layer(x)
        x = self.norm_prelu_0(torch.cat([x, inp_2x], 1))
        output.append(x)
        for i, layer in enumerate(self.level1):
            x = layer(x)
            if i == 0:
                down1 = x
        x = self.norm_prelu_1(torch.cat([x, down1, inp_4x], 1))
        output.append(x)
        for i, layer in enumerate(self.level2):
            x = layer(x)
            if i == 0:
                down2 = x
        x = self.norm_prelu_2(torch.cat([down2, x], 1))
        output.append(x)
        return output


class ConvNeXtBlock(nn.Module):

    def __init__(self, in_channels, norm_cfg=None, act_cfg=None, mlp_ratio=4.0, linear_pw_conv=True, drop_path_rate=0.0, layer_scale_init_value=1e-06):
        super(ConvNeXtBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.linear_pw_conv = linear_pw_conv
        self.norm = BuildNormalization(placeholder=in_channels, norm_cfg=norm_cfg)
        mid_channels = int(mlp_ratio * in_channels)
        if self.linear_pw_conv:
            pw_conv = nn.Linear
        else:
            pw_conv = partial(nn.Conv2d, kernel_size=1)
        self.pointwise_conv1 = pw_conv(in_channels, mid_channels)
        self.act = BuildActivation(act_cfg)
        self.pointwise_conv2 = pw_conv(mid_channels, in_channels)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(in_channels), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
    """forward"""

    def forward(self, x):
        shortcut = x
        x = self.depthwise_conv(x)
        x = self.norm(x)
        if self.linear_pw_conv:
            x = x.permute(0, 2, 3, 1)
        x = self.pointwise_conv1(x)
        x = self.act(x)
        x = self.pointwise_conv2(x)
        if self.linear_pw_conv:
            x = x.permute(0, 3, 1, 2)
        if self.gamma is not None:
            x = x.mul(self.gamma.view(1, -1, 1, 1))
        x = shortcut + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    arch_settings = {'tiny': {'depths': [3, 3, 9, 3], 'channels': [96, 192, 384, 768]}, 'small': {'depths': [3, 3, 27, 3], 'channels': [96, 192, 384, 768]}, 'base': {'depths': [3, 3, 27, 3], 'channels': [128, 256, 512, 1024]}, 'large': {'depths': [3, 3, 27, 3], 'channels': [192, 384, 768, 1536]}, 'xlarge': {'depths': [3, 3, 27, 3], 'channels': [256, 512, 1024, 2048]}}

    def __init__(self, structure_type, arch='tiny', in_channels=3, stem_patch_size=4, norm_cfg={'type': 'LayerNorm2d', 'eps': 1e-06}, act_cfg={'type': 'GELU'}, linear_pw_conv=True, drop_path_rate=0.0, layer_scale_init_value=1e-06, out_indices=(0, 1, 2, 3), gap_before_final_norm=True, pretrained=True, pretrained_model_path=''):
        super(ConvNeXt, self).__init__()
        self.structure_type = structure_type
        self.arch = arch
        self.in_channels = in_channels
        self.stem_patch_size = stem_patch_size
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.linear_pw_conv = linear_pw_conv
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        self.gap_before_final_norm = gap_before_final_norm
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        arch = self.arch_settings[arch]
        self.depths = arch['depths']
        self.channels = arch['channels']
        self.num_stages = len(self.depths)
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'invalid out_indices {index}'
        self.out_indices = out_indices
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and getattr(self, key) == value
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        self.downsample_layers = nn.ModuleList()
        norm_layer = BuildNormalization(placeholder=self.channels[0], norm_cfg=norm_cfg)
        stem = nn.Sequential(nn.Conv2d(in_channels, self.channels[0], kernel_size=stem_patch_size, stride=stem_patch_size), norm_layer)
        self.downsample_layers.append(stem)
        block_idx = 0
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]
            if i >= 1:
                downsample_layer = nn.Sequential(LayerNorm2d(self.channels[i - 1]), nn.Conv2d(self.channels[i - 1], channels, kernel_size=2, stride=2))
                self.downsample_layers.append(downsample_layer)
            stage = nn.Sequential(*[ConvNeXtBlock(in_channels=channels, drop_path_rate=dpr[block_idx + j], norm_cfg=norm_cfg, act_cfg=act_cfg, linear_pw_conv=linear_pw_conv, layer_scale_init_value=layer_scale_init_value) for j in range(depth)])
            block_idx += depth
            self.stages.append(stage)
            if i in self.out_indices:
                norm_layer = BuildNormalization(placeholder=channels, norm_cfg=norm_cfg)
                self.add_module(f'norm{i}', norm_layer)
        if pretrained:
            self.loadpretrainedweights(structure_type, pretrained_model_path)
    """forward"""

    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    outs.append(norm_layer(x).contiguous())
        return tuple(outs)
    """loadpretrainedweights"""

    def loadpretrainedweights(self, structure_type, pretrained_model_path=''):
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        else:
            checkpoint = model_zoo.load_url(DEFAULT_MODEL_URLS[structure_type], map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        state_dict_convert = {}
        for key, value in state_dict.items():
            state_dict_convert[key.replace('backbone.', '')] = value
        self.load_state_dict(state_dict_convert, strict=False)


class ConvNeXtV2Block(nn.Module):

    def __init__(self, dim, drop_path=0.0, norm_cfg=None, act_cfg=None):
        super(ConvNeXtV2Block, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = BuildNormalization(placeholder=dim, norm_cfg=norm_cfg)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = BuildActivation(act_cfg)
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    """forward"""

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        x = identity + self.drop_path(x)
        return x


class ConvNeXtV2(nn.Module):
    arch_settings = {'atto': {'depths': [2, 2, 6, 2], 'dims': [40, 80, 160, 320]}, 'femto': {'depths': [2, 2, 6, 2], 'dims': [48, 96, 192, 384]}, 'pico': {'depths': [2, 2, 6, 2], 'dims': [64, 128, 256, 512]}, 'nano': {'depths': [2, 2, 8, 2], 'dims': [80, 160, 320, 640]}, 'tiny': {'depths': [3, 3, 9, 3], 'dims': [96, 192, 384, 768]}, 'base': {'depths': [3, 3, 27, 3], 'dims': [128, 256, 512, 1024]}, 'large': {'depths': [3, 3, 27, 3], 'dims': [192, 384, 768, 1536]}, 'huge': {'depths': [3, 3, 27, 3], 'dims': [352, 704, 1408, 2816]}}

    def __init__(self, structure_type, in_channels=3, arch='tiny', drop_path_rate=0.0, out_indices=(0, 1, 2, 3), norm_cfg={'type': 'LayerNormConvNeXtV2', 'eps': 1e-06}, act_cfg={'type': 'GELU'}, pretrained=True, pretrained_model_path=''):
        super(ConvNeXtV2, self).__init__()
        assert arch in self.arch_settings
        arch = self.arch_settings[arch]
        self.structure_type = structure_type
        self.in_channels = in_channels
        self.depths = arch['depths']
        self.dims = arch['dims']
        self.drop_path_rate = drop_path_rate
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'invalid out_indices {index}'
        self.out_indices = out_indices
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and getattr(self, key) == value
        self.downsample_layers = nn.ModuleList()
        norm_cfg['data_format'] = 'channels_first'
        stem = nn.Sequential(nn.Conv2d(in_channels, self.dims[0], kernel_size=4, stride=4), BuildNormalization(placeholder=self.dims[0], norm_cfg=norm_cfg))
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(BuildNormalization(placeholder=self.dims[i], norm_cfg=norm_cfg), nn.Conv2d(self.dims[i], self.dims[i + 1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            norm_cfg['data_format'] = 'channels_last'
            stage = nn.Sequential(*[ConvNeXtV2Block(dim=self.dims[i], drop_path=dp_rates[cur + j], norm_cfg=norm_cfg, act_cfg=act_cfg) for j in range(self.depths[i])])
            self.stages.append(stage)
            cur += self.depths[i]
            if i in self.out_indices:
                norm_cfg['data_format'] = 'channels_first'
                norm_layer = BuildNormalization(placeholder=self.dims[i], norm_cfg=norm_cfg)
                self.add_module(f'norm{i}', norm_layer)
        if pretrained:
            self.loadpretrainedweights(structure_type, pretrained_model_path)
    """forward"""

    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                outs.append(norm_layer(x).contiguous())
        return tuple(outs)
    """loadpretrainedweights"""

    def loadpretrainedweights(self, structure_type, pretrained_model_path=''):
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        else:
            checkpoint = model_zoo.load_url(DEFAULT_MODEL_URLS[structure_type], map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        state_dict_convert = {}
        for key, value in state_dict.items():
            state_dict_convert[key.replace('backbone.', '')] = value
            if 'grn.gamma' in key:
                state_dict_convert[key] = value.reshape(1, 1, 1, -1)
            if 'grn.beta' in key:
                state_dict_convert[key] = value.reshape(1, 1, 1, -1)
        self.load_state_dict(state_dict_convert, strict=False)


class SqueezeExcite(nn.Module):

    def __init__(self, channels, rd_ratio=1.0 / 16, rd_channels=None, rd_divisor=8, add_maxpool=False, bias=True, act_cfg={'type': 'ReLU', 'inplace': True}, norm_cfg=None, gate_act_cfg={'type': 'Sigmoid'}):
        super(SqueezeExcite, self).__init__()
        self.add_maxpool = add_maxpool
        if not rd_channels:
            rd_channels = makedivisible(channels * rd_ratio, rd_divisor, min_ratio=0.0)
        self.fc1 = nn.Conv2d(channels, rd_channels, kernel_size=1, bias=bias)
        self.bn = BuildNormalization(placeholder=rd_channels, norm_cfg=norm_cfg) if norm_cfg else nn.Identity()
        self.act = BuildActivation(act_cfg)
        self.fc2 = nn.Conv2d(rd_channels, channels, kernel_size=1, bias=bias)
        self.gate = BuildActivation(gate_act_cfg)
    """forward"""

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


class Residual(nn.Module):

    def __init__(self, m, drop=0.0):
        super(Residual, self).__init__()
        self.m = m
        self.drop = drop
    """forward"""

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1, device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)
    """fuse"""

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2dBN):
            m = self.m.fuse()
            assert m.groups == m.in_channels
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = F.pad(identity, [1, 1, 1, 1])
            m.weight += identity
            return m
        elif isinstance(self.m, nn.Conv2d):
            m = self.m
            assert m.groups != m.in_channels
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = F.pad(identity, [1, 1, 1, 1])
            m.weight += identity
            return m
        else:
            return self


class RepVGGDW(nn.Module):

    def __init__(self, ed):
        super(RepVGGDW, self).__init__()
        self.conv = Conv2dBN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2dBN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
    """forward"""

    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x
    """fuse"""

    @torch.no_grad()
    def fuse(self):
        conv, conv1 = self.conv.fuse(), self.conv1.fuse()
        conv_w, conv_b = conv.weight, conv.bias
        conv1_w, conv1_b = conv1.weight, conv1.bias
        conv1_w = F.pad(conv1_w, [1, 1, 1, 1])
        identity = F.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1, 1, 1, 1])
        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b
        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv


class RepViTBlock(nn.Module):

    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs, skip_downsample=False):
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]
        assert hidden_dim == 2 * inp
        self.identity = stride == 1 and inp == oup
        if stride == 2:
            if skip_downsample:
                stride = 1
            self.token_mixer = nn.Sequential(Conv2dBN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp), SqueezeExcite(inp, 0.25) if use_se else nn.Identity(), Conv2dBN(inp, oup, kernel_size=1, stride=1, padding=0))
            self.channel_mixer = Residual(nn.Sequential(Conv2dBN(oup, 2 * oup, 1, 1, 0), nn.GELU() if use_hs else nn.GELU(), Conv2dBN(2 * oup, oup, 1, 1, 0, bn_weight_init=0)))
        else:
            assert self.identity
            self.token_mixer = nn.Sequential(RepVGGDW(inp), SqueezeExcite(inp, 0.25) if use_se else nn.Identity())
            self.channel_mixer = Residual(nn.Sequential(Conv2dBN(inp, hidden_dim, 1, 1, 0), nn.GELU() if use_hs else nn.GELU(), Conv2dBN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0)))
    """forward"""

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


class EdgeSAMRepViT(nn.Module):
    arch_settings = {'m1': [[3, 2, 48, 1, 0, 1], [3, 2, 48, 0, 0, 1], [3, 2, 48, 0, 0, 1], [3, 2, 96, 0, 0, 2], [3, 2, 96, 1, 0, 1], [3, 2, 96, 0, 0, 1], [3, 2, 96, 0, 0, 1], [3, 2, 192, 0, 1, 2], [3, 2, 192, 1, 1, 1], [3, 2, 192, 0, 1, 1], [3, 2, 192, 1, 1, 1], [3, 2, 192, 0, 1, 1], [3, 2, 192, 1, 1, 1], [3, 2, 192, 0, 1, 1], [3, 2, 192, 1, 1, 1], [3, 2, 192, 0, 1, 1], [3, 2, 192, 1, 1, 1], [3, 2, 192, 0, 1, 1], [3, 2, 192, 1, 1, 1], [3, 2, 192, 0, 1, 1], [3, 2, 192, 1, 1, 1], [3, 2, 192, 0, 1, 1], [3, 2, 192, 0, 1, 1], [3, 2, 384, 0, 1, 2], [3, 2, 384, 1, 1, 1], [3, 2, 384, 0, 1, 1]], 'm2': [[3, 2, 64, 1, 0, 1], [3, 2, 64, 0, 0, 1], [3, 2, 64, 0, 0, 1], [3, 2, 128, 0, 0, 2], [3, 2, 128, 1, 0, 1], [3, 2, 128, 0, 0, 1], [3, 2, 128, 0, 0, 1], [3, 2, 256, 0, 1, 2], [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 512, 0, 1, 2], [3, 2, 512, 1, 1, 1], [3, 2, 512, 0, 1, 1]], 'm3': [[3, 2, 64, 1, 0, 1], [3, 2, 64, 0, 0, 1], [3, 2, 64, 1, 0, 1], [3, 2, 64, 0, 0, 1], [3, 2, 64, 0, 0, 1], [3, 2, 128, 0, 0, 2], [3, 2, 128, 1, 0, 1], [3, 2, 128, 0, 0, 1], [3, 2, 128, 1, 0, 1], [3, 2, 128, 0, 0, 1], [3, 2, 128, 0, 0, 1], [3, 2, 256, 0, 1, 2], [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 256, 1, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 256, 0, 1, 1], [3, 2, 512, 0, 1, 2], [3, 2, 512, 1, 1, 1], [3, 2, 512, 0, 1, 1]]}

    def __init__(self, structure_type, arch, img_size=1024, upsample_mode='bicubic', pretrained=False, pretrained_model_path=''):
        super(EdgeSAMRepViT, self).__init__()
        self.arch = arch
        self.cfgs = self.arch_settings[arch]
        self.img_size = img_size
        self.structure_type = structure_type
        self.upsample_mode = upsample_mode
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and getattr(self, key) == value
        input_channel = self.cfgs[0][2]
        patch_embed = nn.Sequential(Conv2dBN(3, input_channel // 2, 3, 2, 1), nn.GELU(), Conv2dBN(input_channel // 2, input_channel, 3, 2, 1))
        layers = [patch_embed]
        block = RepViTBlock
        self.stage_idx = []
        prev_c = input_channel
        for idx, (k, t, c, use_se, use_hs, s) in enumerate(self.cfgs):
            output_channel = makedivisible(c, 8)
            exp_size = makedivisible(input_channel * t, 8)
            skip_downsample = False
            if c != prev_c:
                self.stage_idx.append(idx - 1)
                prev_c = c
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs, skip_downsample))
            input_channel = output_channel
        self.stage_idx.append(idx)
        self.features = nn.ModuleList(layers)
        stage2_channels = makedivisible(self.cfgs[self.stage_idx[2]][2], 8)
        stage3_channels = makedivisible(self.cfgs[self.stage_idx[3]][2], 8)
        self.fuse_stage2 = nn.Conv2d(stage2_channels, 256, kernel_size=1, bias=False)
        self.fuse_stage3 = nn.Sequential()
        self.fuse_stage3.add_module('op_list', nn.Sequential(nn.Conv2d(stage3_channels, 256, kernel_size=1, bias=False)))
        self.neck = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False), LayerNorm2d(256), nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False), LayerNorm2d(256))
        if pretrained:
            state_dict = loadpretrainedweights(structure_type=structure_type, pretrained_model_path=pretrained_model_path, default_model_urls=DEFAULT_MODEL_URLS)
            self.load_state_dict(state_dict, strict=False)
    """forward"""

    def forward(self, x):
        counter = 0
        output_dict = dict()
        x = self.features[0](x)
        output_dict['stem'] = x
        for idx, f in enumerate(self.features[1:]):
            x = f(x)
            if idx in self.stage_idx:
                output_dict[f'stage{counter}'] = x
                counter += 1
        x = self.fuse_stage2(output_dict['stage2']) + F.interpolate(self.fuse_stage3(output_dict['stage3']), scale_factor=2, mode=self.upsample_mode, align_corners=False)
        x = self.neck(x)
        return x


class DownsamplerBlock(nn.Module):

    def __init__(self, in_channels, out_channels, norm_cfg=None, act_cfg=None):
        super(DownsamplerBlock, self).__init__()
        self.norm_cfg, self.act_cfg = norm_cfg, act_cfg
        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg)
        self.act = BuildActivation(act_cfg)
    """forward"""

    def forward(self, x):
        conv_out, pool_out = self.conv(x), self.pool(x)
        pool_out = F.interpolate(pool_out, size=conv_out.size()[2:], mode='bilinear', align_corners=False)
        output = torch.cat([conv_out, pool_out], dim=1)
        output = self.bn(output)
        output = self.act(output)
        return output


class NonBottleneck1d(nn.Module):

    def __init__(self, channels, drop_rate=0, dilation=1, num_conv_layer=2, norm_cfg=None, act_cfg=None):
        super(NonBottleneck1d, self).__init__()
        self.norm_cfg, self.act_cfg = norm_cfg, act_cfg
        self.act = BuildActivation(act_cfg)
        self.convs_layers = nn.ModuleList()
        for conv_layer in range(num_conv_layer):
            first_conv_padding = (1, 0) if conv_layer == 0 else (dilation, 0)
            first_conv_dilation = 1 if conv_layer == 0 else (dilation, 1)
            second_conv_padding = (0, 1) if conv_layer == 0 else (0, dilation)
            second_conv_dilation = 1 if conv_layer == 0 else (1, dilation)
            self.convs_layers.append(nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=1, padding=first_conv_padding, bias=True, dilation=first_conv_dilation))
            self.convs_layers.append(self.act)
            self.convs_layers.append(nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=1, padding=second_conv_padding, bias=True, dilation=second_conv_dilation))
            self.convs_layers.append(BuildNormalization(placeholder=channels, norm_cfg=norm_cfg))
            if conv_layer == 0:
                self.convs_layers.append(self.act)
            else:
                self.convs_layers.append(nn.Dropout(p=drop_rate))
    """forward"""

    def forward(self, x):
        output = x
        for conv in self.convs_layers:
            output = conv(output)
        output = self.act(output + x)
        return output


class UpsamplerBlock(nn.Module):

    def __init__(self, in_channels, out_channels, norm_cfg=None, act_cfg=None):
        super(UpsamplerBlock, self).__init__()
        self.norm_cfg, self.act_cfg = norm_cfg, act_cfg
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg)
        self.act = BuildActivation(act_cfg)
    """forward"""

    def forward(self, x):
        output = self.conv(x)
        output = self.bn(output)
        output = self.act(output)
        return output


class ERFNet(nn.Module):

    def __init__(self, structure_type, in_channels=3, enc_downsample_channels=(16, 64, 128), enc_stage_non_bottlenecks=(5, 8), enc_non_bottleneck_dilations=(2, 4, 8, 16), enc_non_bottleneck_channels=(64, 128), dec_upsample_channels=(64, 16), dec_stages_non_bottleneck=(2, 2), dec_non_bottleneck_channels=(64, 16), dropout_ratio=0.1, norm_cfg={'type': 'SyncBatchNorm'}, act_cfg={'type': 'PReLU'}, pretrained=False, pretrained_model_path=''):
        super(ERFNet, self).__init__()
        self.structure_type = structure_type
        self.in_channels = in_channels
        self.enc_downsample_channels = enc_downsample_channels
        self.enc_stage_non_bottlenecks = enc_stage_non_bottlenecks
        self.enc_non_bottleneck_dilations = enc_non_bottleneck_dilations
        self.enc_non_bottleneck_channels = enc_non_bottleneck_channels
        self.dec_upsample_channels = dec_upsample_channels
        self.dec_stages_non_bottleneck = dec_stages_non_bottleneck
        self.dec_non_bottleneck_channels = dec_non_bottleneck_channels
        self.dropout_ratio = dropout_ratio
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        assert len(enc_downsample_channels) == len(dec_upsample_channels) + 1
        assert len(enc_downsample_channels) == len(enc_stage_non_bottlenecks) + 1
        assert len(enc_downsample_channels) == len(enc_non_bottleneck_channels) + 1
        assert enc_stage_non_bottlenecks[-1] % len(enc_non_bottleneck_dilations) == 0
        assert len(dec_upsample_channels) == len(dec_stages_non_bottleneck)
        assert len(dec_stages_non_bottleneck) == len(dec_non_bottleneck_channels)
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and getattr(self, key) == value
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.encoder.append(DownsamplerBlock(in_channels, enc_downsample_channels[0], norm_cfg=norm_cfg, act_cfg=act_cfg))
        for i in range(len(enc_downsample_channels) - 1):
            self.encoder.append(DownsamplerBlock(enc_downsample_channels[i], enc_downsample_channels[i + 1], norm_cfg=norm_cfg, act_cfg=act_cfg))
            if i == len(enc_downsample_channels) - 2:
                iteration_times = int(enc_stage_non_bottlenecks[-1] / len(enc_non_bottleneck_dilations))
                for j in range(iteration_times):
                    for k in range(len(enc_non_bottleneck_dilations)):
                        self.encoder.append(NonBottleneck1d(enc_downsample_channels[-1], dropout_ratio, enc_non_bottleneck_dilations[k], norm_cfg=norm_cfg, act_cfg=act_cfg))
            else:
                for j in range(enc_stage_non_bottlenecks[i]):
                    self.encoder.append(NonBottleneck1d(enc_downsample_channels[i + 1], dropout_ratio, norm_cfg=norm_cfg, act_cfg=act_cfg))
        for i in range(len(dec_upsample_channels)):
            if i == 0:
                self.decoder.append(UpsamplerBlock(enc_downsample_channels[-1], dec_non_bottleneck_channels[i], norm_cfg=norm_cfg, act_cfg=act_cfg))
            else:
                self.decoder.append(UpsamplerBlock(dec_non_bottleneck_channels[i - 1], dec_non_bottleneck_channels[i], norm_cfg=norm_cfg, act_cfg=act_cfg))
            for j in range(dec_stages_non_bottleneck[i]):
                self.decoder.append(NonBottleneck1d(dec_non_bottleneck_channels[i], norm_cfg=norm_cfg, act_cfg=act_cfg))
        if pretrained:
            state_dict = loadpretrainedweights(structure_type=structure_type, pretrained_model_path=pretrained_model_path, default_model_urls=DEFAULT_MODEL_URLS)
            self.load_state_dict(state_dict, strict=False)
    """forward"""

    def forward(self, x):
        for enc in self.encoder:
            x = enc(x)
        for dec in self.decoder:
            x = dec(x)
        return [x]


class PoolingPyramidModule(nn.ModuleList):

    def __init__(self, pool_scales, in_channels, out_channels, norm_cfg, act_cfg, align_corners):
        super(PoolingPyramidModule, self).__init__()
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        for pool_scale in pool_scales:
            self.append(nn.Sequential(nn.AdaptiveAvgPool2d(pool_scale), nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg)))
    """forward"""

    def forward(self, x):
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(input=ppm_out, size=x.shape[2:], mode='bilinear', align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class LearningToDownsample(nn.Module):

    def __init__(self, in_channels, dw_channels, out_channels, norm_cfg=None, act_cfg=None, dw_act_cfg=None):
        super(LearningToDownsample, self).__init__()
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.dw_act_cfg = dw_act_cfg
        dw_channels1, dw_channels2 = dw_channels
        self.conv = nn.Sequential(nn.Conv2d(in_channels, dw_channels1, kernel_size=3, stride=2, padding=1, bias=False), BuildNormalization(placeholder=dw_channels1, norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.dsconv1 = DepthwiseSeparableConv2d(in_channels=dw_channels1, out_channels=dw_channels2, kernel_size=3, stride=2, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, dw_act_cfg=self.dw_act_cfg)
        self.dsconv2 = DepthwiseSeparableConv2d(in_channels=dw_channels2, out_channels=out_channels, kernel_size=3, stride=2, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, dw_act_cfg=self.dw_act_cfg)
    """forward"""

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):

    def __init__(self, in_channels=64, block_channels=(64, 96, 128), out_channels=128, expand_ratio=6, num_blocks=(3, 3, 3), strides=(2, 2, 1), pool_scales=(1, 2, 3, 6), norm_cfg=None, act_cfg=None, align_corners=False):
        super(GlobalFeatureExtractor, self).__init__()
        assert len(block_channels) == len(num_blocks) == 3
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.bottleneck1 = self.makelayer(in_channels, block_channels[0], num_blocks[0], strides[0], expand_ratio)
        self.bottleneck2 = self.makelayer(block_channels[0], block_channels[1], num_blocks[1], strides[1], expand_ratio)
        self.bottleneck3 = self.makelayer(block_channels[1], block_channels[2], num_blocks[2], strides[2], expand_ratio)
        self.ppm = PoolingPyramidModule(pool_scales, block_channels[2], block_channels[2] // 4, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, align_corners=align_corners)
        self.out = nn.Sequential(nn.Conv2d(block_channels[2] * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
    """makelayer"""

    def makelayer(self, in_channels, out_channels, blocks, stride=1, expand_ratio=6):
        layers = [InvertedResidual(in_channels, out_channels, stride, expand_ratio, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)]
        for _ in range(1, blocks):
            layers.append(InvertedResidual(out_channels, out_channels, 1, expand_ratio, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
        return nn.Sequential(*layers)
    """forward"""

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = torch.cat([x, *self.ppm(x)], dim=1)
        x = self.out(x)
        return x


class FastSCNN(nn.Module):

    def __init__(self, structure_type, in_channels=3, downsample_dw_channels=(32, 48), global_in_channels=64, global_block_channels=(64, 96, 128), global_block_strides=(2, 2, 1), global_out_channels=128, higher_in_channels=64, lower_in_channels=128, fusion_out_channels=128, out_indices=(0, 1, 2), norm_cfg={'type': 'SyncBatchNorm'}, act_cfg={'type': 'ReLU', 'inplace': True}, align_corners=False, dw_act_cfg={'type': 'ReLU', 'inplace': True}, pretrained=False, pretrained_model_path=''):
        super(FastSCNN, self).__init__()
        self.structure_type = structure_type
        self.in_channels = in_channels
        self.downsample_dw_channels = downsample_dw_channels
        self.downsample_dw_channels1 = downsample_dw_channels[0]
        self.downsample_dw_channels2 = downsample_dw_channels[1]
        self.global_in_channels = global_in_channels
        self.global_block_channels = global_block_channels
        self.global_block_strides = global_block_strides
        self.global_out_channels = global_out_channels
        self.higher_in_channels = higher_in_channels
        self.lower_in_channels = lower_in_channels
        self.fusion_out_channels = fusion_out_channels
        self.out_indices = out_indices
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.dw_act_cfg = dw_act_cfg
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        assert global_in_channels == higher_in_channels, 'Global Input Channels must be the same with Higher Input Channels'
        assert global_out_channels == lower_in_channels, 'Global Output Channels must be the same with Lower Input Channels'
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and getattr(self, key) == value
        self.learning_to_downsample = LearningToDownsample(in_channels=in_channels, dw_channels=downsample_dw_channels, out_channels=global_in_channels, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, dw_act_cfg=self.dw_act_cfg)
        self.global_feature_extractor = GlobalFeatureExtractor(in_channels=global_in_channels, block_channels=global_block_channels, out_channels=global_out_channels, strides=self.global_block_strides, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, align_corners=self.align_corners)
        self.feature_fusion = FeatureFusionModule(higher_in_channels=higher_in_channels, lower_in_channels=lower_in_channels, out_channels=fusion_out_channels, norm_cfg=self.norm_cfg, dwconv_act_cfg=self.act_cfg, conv_act_cfg=self.act_cfg, align_corners=self.align_corners)
        if pretrained:
            state_dict = loadpretrainedweights(structure_type=structure_type, pretrained_model_path=pretrained_model_path, default_model_urls=DEFAULT_MODEL_URLS)
            self.load_state_dict(state_dict, strict=False)
    """forward"""

    def forward(self, x):
        higher_res_features = self.learning_to_downsample(x)
        lower_res_features = self.global_feature_extractor(higher_res_features)
        fusion_output = self.feature_fusion(higher_res_features, lower_res_features)
        outs = [higher_res_features, lower_res_features, fusion_output]
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)


class PositionEmbeddingRandom(nn.Module):

    def __init__(self, num_pos_feats=64, scale=None):
        super(PositionEmbeddingRandom, self).__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer('positional_encoding_gaussian_matrix', scale * torch.randn((2, num_pos_feats)))
    """peencoding"""

    def peencoding(self, coords):
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
    """forward"""

    def forward(self, size):
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self.peencoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)
    """forwardwithcoords"""

    def forwardwithcoords(self, coords_input, image_size):
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self.peencoding(coords)


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, sigmoid_output=False):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output
    """forward"""

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


def dopool(x, pool, norm=None):
    if pool is None:
        return x
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)
    return x


class MultiScaleAttention(nn.Module):

    def __init__(self, dim, dim_out, num_heads, q_pool=None):
        super(MultiScaleAttention, self).__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim ** -0.5
        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)
    """forward"""

    def forward(self, x):
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        q, k, v = torch.unbind(qkv, 2)
        if self.q_pool:
            q = dopool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]
            q = q.reshape(B, H * W, self.num_heads, -1)
        x = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)
        x = self.proj(x)
        return x


def windowpartition(x, window_size):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def windowunpartition(windows, window_size, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class MultiScaleBlock(nn.Module):

    def __init__(self, dim, dim_out, num_heads, mlp_ratio=4.0, drop_path=0.0, norm_cfg={'type': 'LayerNorm', 'eps': 1e-06}, q_stride=None, act_cfg={'type': 'GELU'}, window_size=0):
        super(MultiScaleBlock, self).__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = BuildNormalization(dim, norm_cfg)
        self.window_size = window_size
        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(kernel_size=q_stride, stride=q_stride, ceil_mode=False)
        self.attn = MultiScaleAttention(dim, dim_out, num_heads=num_heads, q_pool=self.pool)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = BuildNormalization(dim_out, norm_cfg)
        self.mlp = MLP(dim_out, int(dim_out * mlp_ratio), dim_out, num_layers=2, act_cfg=act_cfg)
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)
    """forward"""

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        if self.dim != self.dim_out:
            shortcut = dopool(self.proj(x), self.pool)
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = windowpartition(x, window_size)
        x = self.attn(x)
        if self.q_stride:
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]
            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = H + pad_h, W + pad_w
        if self.window_size > 0:
            x = windowunpartition(x, window_size, pad_hw, (H, W))
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Hiera(nn.Module):

    def __init__(self, embed_dim=96, num_heads=1, drop_path_rate=0.0, q_pool=3, q_stride=(2, 2), stages=(2, 3, 16, 3), dim_mul=2.0, head_mul=2.0, window_pos_embed_bkg_spatial_size=(14, 14), window_spec=(8, 4, 14, 7), global_att_blocks=(12, 16, 20), return_interm_layers=True):
        super(Hiera, self).__init__()
        assert len(stages) == len(window_spec)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop_path_rate = drop_path_rate
        self.q_pool = q_pool
        self.q_stride = q_stride
        self.stages = stages
        self.dim_mul = dim_mul
        self.head_mul = head_mul
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.window_spec = window_spec
        self.global_att_blocks = global_att_blocks
        self.return_interm_layers = return_interm_layers
        depth = sum(stages)
        self.stage_ends = [(sum(stages[:i]) - 1) for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [(x + 1) for x in self.stage_ends[:-1]][:q_pool]
        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size))
        self.pos_embed_window = nn.Parameter(torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0]))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        cur_stage = 1
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dim_out = embed_dim
            window_size = self.window_spec[cur_stage - 1]
            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size
            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1
            block = MultiScaleBlock(dim=embed_dim, dim_out=dim_out, num_heads=num_heads, drop_path=dpr[i], q_stride=self.q_stride if i in self.q_pool_blocks else None, window_size=window_size)
            embed_dim = dim_out
            self.blocks.append(block)
        self.channel_list = [self.blocks[i].dim_out for i in self.stage_ends[::-1]] if return_interm_layers else [self.blocks[-1].dim_out]
    """getposembed"""

    def getposembed(self, hw):
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode='bicubic')
        pos_embed = pos_embed + window_embed.tile([(x // y) for x, y in zip(pos_embed.shape, window_embed.shape)])
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed
    """forward"""

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.getposembed(x.shape[1:3])
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == self.stage_ends[-1] or i in self.stage_ends and self.return_interm_layers:
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)
        return outputs


class PEBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {'PositionEmbeddingSine': PositionEmbeddingSine, 'PositionEmbeddingRandom': PositionEmbeddingRandom}
    """build"""

    def build(self, pe_cfg):
        return super().build(pe_cfg)


BuildPE = PEBuilder().build


class FPNNeck(nn.Module):

    def __init__(self, position_encoding_cfg, d_model, backbone_channel_list, kernel_size=1, stride=1, padding=0, fpn_interp_model='bilinear', fuse_type='sum', fpn_top_down_levels=None):
        super(FPNNeck, self).__init__()
        assert fuse_type in ['sum', 'avg']
        self.fuse_type = fuse_type
        self.fpn_interp_model = fpn_interp_model
        self.position_encoding = BuildPE(position_encoding_cfg)
        self.backbone_channel_list = backbone_channel_list
        self.convs = nn.ModuleList()
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module('conv', nn.Conv2d(in_channels=dim, out_channels=d_model, kernel_size=kernel_size, stride=stride, padding=padding))
            self.convs.append(current)
        if fpn_top_down_levels is None:
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)
    """forward"""

    def forward(self, xs):
        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        prev_features = None
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode=self.fpn_interp_model, align_corners=None if self.fpn_interp_model == 'nearest' else False, antialias=False)
                prev_features = lateral_features + top_down_features
                if self.fuse_type == 'avg':
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out)
        return out, pos


class HieraWithFPN(nn.Module):

    def __init__(self, hiera_cfg, fpn_cfg, scalp=0):
        super(HieraWithFPN, self).__init__()
        self.trunk = Hiera(**hiera_cfg)
        self.neck = FPNNeck(**fpn_cfg)
        self.scalp = int(scalp)
        assert self.trunk.channel_list == self.neck.backbone_channel_list, f'Channel dims of trunk and neck do not match. Trunk: {self.trunk.channel_list}, neck: {self.neck.backbone_channel_list}'
    """forward"""

    def forward(self, sample):
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            features, pos = features[:-self.scalp], pos[:-self.scalp]
        src = features[-1]
        output = {'vision_features': src, 'vision_pos_enc': pos, 'backbone_fpn': features}
        return output


class HRModule(nn.Module):

    def __init__(self, num_branches, block, num_blocks, in_channels, num_channels, multiscale_output=True, norm_cfg=None, act_cfg=None):
        super(HRModule, self).__init__()
        self.checkbranches(num_branches, num_blocks, in_channels, num_channels)
        self.in_channels = in_channels
        self.num_branches = num_branches
        self.multiscale_output = multiscale_output
        self.branches = self.makebranches(num_branches, block, num_blocks, num_channels, norm_cfg, act_cfg)
        self.fuse_layers = self.makefuselayers(norm_cfg, act_cfg)
        self.relu = BuildActivation(act_cfg)
    """forward"""

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = 0
            for j in range(self.num_branches):
                if i == j:
                    y += x[j]
                elif j > i:
                    y = y + F.interpolate(self.fuse_layers[i][j](x[j]), size=x[i].shape[2:], mode='bilinear', align_corners=False)
                else:
                    y += self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse
    """checkbranches"""

    def checkbranches(self, num_branches, num_blocks, in_channels, num_channels):
        assert num_branches == len(num_blocks), 'num_branches should be equal to len(num_blocks)'
        assert num_branches == len(num_channels), 'num_branches should be equal to len(num_channels)'
        assert num_branches == len(in_channels), 'num_branches should be equal to len(in_channels)'
    """makebranches"""

    def makebranches(self, num_branches, block, num_blocks, num_channels, norm_cfg=None, act_cfg=None):
        branches = []
        for i in range(num_branches):
            branches.append(self.makebranch(i, block, num_blocks, num_channels, norm_cfg=norm_cfg, act_cfg=act_cfg))
        return nn.ModuleList(branches)
    """makebranch"""

    def makebranch(self, branch_index, block, num_blocks, num_channels, stride=1, norm_cfg=None, act_cfg=None):
        downsample = None
        if stride != 1 or self.in_channels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, padding=0, bias=False), BuildNormalization(placeholder=num_channels[branch_index] * block.expansion, norm_cfg=norm_cfg))
        layers = []
        layers.append(block(self.in_channels[branch_index], num_channels[branch_index], stride, downsample=downsample, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.in_channels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.in_channels[branch_index], num_channels[branch_index], norm_cfg=norm_cfg, act_cfg=act_cfg))
        return nn.Sequential(*layers)
    """makefuselayers"""

    def makefuselayers(self, norm_cfg=None, act_cfg=None):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(nn.Conv2d(in_channels[j], in_channels[i], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=in_channels[i], norm_cfg=norm_cfg), nn.Upsample(scale_factor=2 ** (j - i), mode='bilinear', align_corners=False)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(nn.Sequential(nn.Conv2d(in_channels[j], in_channels[i], kernel_size=3, stride=2, padding=1, bias=False), BuildNormalization(placeholder=in_channels[i], norm_cfg=norm_cfg)))
                        else:
                            conv_downsamples.append(nn.Sequential(nn.Conv2d(in_channels[j], in_channels[j], kernel_size=3, stride=2, padding=1, bias=False), BuildNormalization(placeholder=in_channels[j], norm_cfg=norm_cfg), BuildActivation(act_cfg)))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)


class HRNet(nn.Module):
    blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}
    arch_settings = {'hrnetv2_w18_small': {'stage1': {'num_modules': 1, 'num_branches': 1, 'block': 'BOTTLENECK', 'num_blocks': (2,), 'num_channels': (64,)}, 'stage2': {'num_modules': 1, 'num_branches': 2, 'block': 'BASIC', 'num_blocks': (2, 2), 'num_channels': (18, 36)}, 'stage3': {'num_modules': 3, 'num_branches': 3, 'block': 'BASIC', 'num_blocks': (2, 2, 2), 'num_channels': (18, 36, 72)}, 'stage4': {'num_modules': 2, 'num_branches': 4, 'block': 'BASIC', 'num_blocks': (2, 2, 2, 2), 'num_channels': (18, 36, 72, 144)}}, 'hrnetv2_w18': {'stage1': {'num_modules': 1, 'num_branches': 1, 'block': 'BOTTLENECK', 'num_blocks': (4,), 'num_channels': (64,)}, 'stage2': {'num_modules': 1, 'num_branches': 2, 'block': 'BASIC', 'num_blocks': (4, 4), 'num_channels': (18, 36)}, 'stage3': {'num_modules': 4, 'num_branches': 3, 'block': 'BASIC', 'num_blocks': (4, 4, 4), 'num_channels': (18, 36, 72)}, 'stage4': {'num_modules': 3, 'num_branches': 4, 'block': 'BASIC', 'num_blocks': (4, 4, 4, 4), 'num_channels': (18, 36, 72, 144)}}, 'hrnetv2_w32': {'stage1': {'num_modules': 1, 'num_branches': 1, 'block': 'BOTTLENECK', 'num_blocks': (4,), 'num_channels': (64,)}, 'stage2': {'num_modules': 1, 'num_branches': 2, 'block': 'BASIC', 'num_blocks': (4, 4), 'num_channels': (32, 64)}, 'stage3': {'num_modules': 4, 'num_branches': 3, 'block': 'BASIC', 'num_blocks': (4, 4, 4), 'num_channels': (32, 64, 128)}, 'stage4': {'num_modules': 3, 'num_branches': 4, 'block': 'BASIC', 'num_blocks': (4, 4, 4, 4), 'num_channels': (32, 64, 128, 256)}}, 'hrnetv2_w40': {'stage1': {'num_modules': 1, 'num_branches': 1, 'block': 'BOTTLENECK', 'num_blocks': (4,), 'num_channels': (64,)}, 'stage2': {'num_modules': 1, 'num_branches': 2, 'block': 'BASIC', 'num_blocks': (4, 4), 'num_channels': (40, 80)}, 'stage3': {'num_modules': 4, 'num_branches': 3, 'block': 'BASIC', 'num_blocks': (4, 4, 4), 'num_channels': (40, 80, 160)}, 'stage4': {'num_modules': 3, 'num_branches': 4, 'block': 'BASIC', 'num_blocks': (4, 4, 4, 4), 'num_channels': (40, 80, 160, 320)}}, 'hrnetv2_w48': {'stage1': {'num_modules': 1, 'num_branches': 1, 'block': 'BOTTLENECK', 'num_blocks': (4,), 'num_channels': (64,)}, 'stage2': {'num_modules': 1, 'num_branches': 2, 'block': 'BASIC', 'num_blocks': (4, 4), 'num_channels': (48, 96)}, 'stage3': {'num_modules': 4, 'num_branches': 3, 'block': 'BASIC', 'num_blocks': (4, 4, 4), 'num_channels': (48, 96, 192)}, 'stage4': {'num_modules': 3, 'num_branches': 4, 'block': 'BASIC', 'num_blocks': (4, 4, 4, 4), 'num_channels': (48, 96, 192, 384)}}}

    def __init__(self, structure_type, arch='hrnetv2_w18_small', in_channels=3, norm_cfg={'type': 'SyncBatchNorm'}, act_cfg={'type': 'ReLU', 'inplace': True}, pretrained=True, pretrained_model_path=''):
        super(HRNet, self).__init__()
        self.structure_type = structure_type
        self.arch = arch
        self.in_channels = in_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and getattr(self, key) == value
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = BuildNormalization(placeholder=64, norm_cfg=norm_cfg)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = BuildNormalization(placeholder=64, norm_cfg=norm_cfg)
        self.relu = BuildActivation(act_cfg)
        stages_cfg = self.arch_settings[arch]
        self.stage1_cfg = stages_cfg['stage1']
        num_channels = self.stage1_cfg['num_channels'][0]
        block_type = self.stage1_cfg['block']
        num_blocks = self.stage1_cfg['num_blocks'][0]
        block = self.blocks_dict[block_type]
        stage1_out_channels = num_channels * block.expansion
        self.layer1 = self.makelayer(block, 64, num_channels, num_blocks, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.stage2_cfg = stages_cfg['stage2']
        num_channels = self.stage2_cfg['num_channels']
        block_type = self.stage2_cfg['block']
        block = self.blocks_dict[block_type]
        num_channels = [(channel * block.expansion) for channel in num_channels]
        self.transition1 = self.maketransitionlayer([stage1_out_channels], num_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.stage2, pre_stage_channels = self.makestage(self.stage2_cfg, num_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.stage3_cfg = stages_cfg['stage3']
        num_channels = self.stage3_cfg['num_channels']
        block_type = self.stage3_cfg['block']
        block = self.blocks_dict[block_type]
        num_channels = [(channel * block.expansion) for channel in num_channels]
        self.transition2 = self.maketransitionlayer(pre_stage_channels, num_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.stage3, pre_stage_channels = self.makestage(self.stage3_cfg, num_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.stage4_cfg = stages_cfg['stage4']
        num_channels = self.stage4_cfg['num_channels']
        block_type = self.stage4_cfg['block']
        block = self.blocks_dict[block_type]
        num_channels = [(channel * block.expansion) for channel in num_channels]
        self.transition3 = self.maketransitionlayer(pre_stage_channels, num_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.stage4, pre_stage_channels = self.makestage(self.stage4_cfg, num_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        if pretrained:
            state_dict = loadpretrainedweights(structure_type=structure_type, pretrained_model_path=pretrained_model_path, default_model_urls=DEFAULT_MODEL_URLS)
            self.load_state_dict(state_dict, strict=False)
    """forward"""

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        h, w = max([y.shape[2] for y in y_list]), max([y.shape[3] for y in y_list])
        out = torch.cat([F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False) for y in y_list], dim=1)
        outs = [out]
        return tuple(outs)
    """makestage"""

    def makestage(self, layer_config, in_channels, multiscale_output=True, norm_cfg=None, act_cfg=None):
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = self.blocks_dict[layer_config['block']]
        hr_modules = []
        for i in range(num_modules):
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True
            hr_modules.append(HRModule(num_branches, block, num_blocks, in_channels, num_channels, reset_multiscale_output, norm_cfg, act_cfg))
        return nn.Sequential(*hr_modules), in_channels
    """makelayer"""

    def makelayer(self, block, inplanes, planes, num_blocks, stride=1, norm_cfg=None, act_cfg=None):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=planes * block.expansion, norm_cfg=norm_cfg))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample, norm_cfg=norm_cfg, act_cfg=act_cfg))
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(inplanes, planes, norm_cfg=norm_cfg, act_cfg=act_cfg))
        return nn.Sequential(*layers)
    """maketransitionlayer"""

    def maketransitionlayer(self, num_channels_pre_layer, num_channels_cur_layer, norm_cfg=None, act_cfg=None):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=num_channels_cur_layer[i], norm_cfg=norm_cfg), BuildActivation(act_cfg)))
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg)))
                transition_layers.append(nn.Sequential(*conv_downsamples))
        return nn.ModuleList(transition_layers)


class MAE(BEiT):

    def __init__(self, structure_type, img_size=(640, 640), patch_size=16, in_channels=3, embed_dims=768, num_layers=12, num_heads=12, mlp_ratio=4, out_indices=(3, 5, 7, 11), attn_drop_rate=0.0, drop_path_rate=0.1, norm_cfg={'type': 'LayerNorm', 'eps': 1e-06}, act_cfg={'type': 'GELU'}, patch_norm=False, final_norm=False, num_fcs=2, init_values=1.0, pretrained=True, pretrained_model_path=''):
        super(MAE, self).__init__(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dims=embed_dims, num_layers=num_layers, num_heads=num_heads, mlp_ratio=mlp_ratio, out_indices=out_indices, qv_bias=False, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_cfg=norm_cfg, act_cfg=act_cfg, patch_norm=patch_norm, final_norm=final_norm, num_fcs=num_fcs, init_values=init_values, pretrained=False, pretrained_model_path=pretrained_model_path, structure_type=structure_type)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dims))
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and getattr(self, key) == value
        if pretrained:
            self.loadpretrainedweights(structure_type, pretrained_model_path)
    """buildlayers"""

    def buildlayers(self):
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.num_layers)]
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(MAETransformerEncoderLayer(embed_dims=self.embed_dims, num_heads=self.num_heads, feedforward_channels=self.mlp_ratio * self.embed_dims, attn_drop_rate=self.attn_drop_rate, drop_path_rate=dpr[i], num_fcs=self.num_fcs, bias=True, act_cfg=self.act_cfg, norm_cfg=self.norm_cfg, window_size=self.patch_shape, init_values=self.init_values))
    """loadpretrainedweights"""

    def loadpretrainedweights(self, structure_type='mae_pretrain_vit_base', pretrained_model_path=''):
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        else:
            checkpoint = model_zoo.load_url(DEFAULT_MODEL_URLS[structure_type], map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        state_dict = self.beitconvert(state_dict)
        state_dict = self.resizerelposembed(state_dict)
        state_dict = self.resizeabsposembed(state_dict)
        self.load_state_dict(state_dict, strict=False)
    """resizeabsposembed"""

    def resizeabsposembed(self, state_dict):
        if 'pos_embed' in state_dict:
            pos_embed_checkpoint = state_dict['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_extra_tokens = self.pos_embed.shape[-2] - self.num_patches
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            new_size = int(self.num_patches ** 0.5)
            if orig_size != new_size:
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                state_dict['pos_embed'] = new_pos_embed
        return state_dict
    """forward"""

    def forward(self, inputs):
        B = inputs.shape[0]
        x, hw_shape = self.patch_embed(inputs)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        outs = []
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx == len(self.layers) - 1:
                if self.final_norm:
                    x = self.norm1(x)
            if idx in self.out_indices:
                out = x[:, 1:]
                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1], C).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return tuple(outs)


def nchwtonlc(x):
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()


def nlctonchw(x, hw_shape):
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, "The seq_len doesn't match H, W"
    return x.transpose(1, 2).reshape(B, C, H, W)


class MixFFN(nn.Module):

    def __init__(self, embed_dims, feedforward_channels, act_cfg=None, ffn_drop=0.0, dropout_cfg=None):
        super(MixFFN, self).__init__()
        self.act_cfg = act_cfg
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.layers = nn.Sequential(nn.Conv2d(embed_dims, feedforward_channels, kernel_size=1, stride=1, padding=0, bias=True), nn.Conv2d(feedforward_channels, feedforward_channels, kernel_size=3, stride=1, padding=1, bias=True, groups=feedforward_channels), BuildActivation(act_cfg), nn.Dropout(ffn_drop), nn.Conv2d(feedforward_channels, embed_dims, kernel_size=1, stride=1, padding=0, bias=True), nn.Dropout(ffn_drop))
        self.dropout_layer = BuildDropout(dropout_cfg) if dropout_cfg else torch.nn.Identity()
    """forward"""

    def forward(self, x, hw_shape, identity=None):
        out = nlctonchw(x, hw_shape)
        out = self.layers(out)
        out = nchwtonlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class EfficientMultiheadAttention(MultiheadAttention):

    def __init__(self, embed_dims, num_heads, attn_drop=0.0, proj_drop=0.0, dropout_cfg=None, batch_first=True, qkv_bias=False, norm_cfg=None, sr_ratio=1):
        super(EfficientMultiheadAttention, self).__init__(embed_dims, num_heads, attn_drop, proj_drop, dropout_cfg=dropout_cfg, batch_first=batch_first, bias=qkv_bias)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(embed_dims, embed_dims, kernel_size=sr_ratio, stride=sr_ratio, padding=0)
            self.norm = BuildNormalization(placeholder=embed_dims, norm_cfg=norm_cfg)
    """forward"""

    def forward(self, x, hw_shape, identity=None):
        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlctonchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchwtonlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x
        if identity is None:
            identity = x_q
        if self.batch_first:
            x_q = x_q.transpose(0, 1)
            x_kv = x_kv.transpose(0, 1)
        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]
        if self.batch_first:
            out = out.transpose(0, 1)
        return identity + self.dropout_layer(self.proj_drop(out))


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, norm_cfg=None, act_cfg=None, norm_before=False):
        super(TransformerEncoderLayer, self).__init__()
        self.norm_before = norm_before
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = BuildNormalization(placeholder=d_model, norm_cfg=norm_cfg)
        self.norm2 = BuildNormalization(placeholder=d_model, norm_cfg=norm_cfg)
        self.activation = BuildActivation(act_cfg)
    """withposembed"""

    def withposembed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos
    """normafterforward"""

    def normafterforward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = self.withposembed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    """normbeforeforward"""

    def normbeforeforward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src2 = self.norm1(src)
        q = k = self.withposembed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src
    """forward"""

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        if self.norm_before:
            return self.normbeforeforward(src, src_mask, src_key_padding_mask, pos)
        return self.normafterforward(src, src_mask, src_key_padding_mask, pos)


class MixVisionTransformer(nn.Module):

    def __init__(self, structure_type, in_channels=3, embed_dims=64, num_stages=4, num_layers=[3, 4, 6, 3], num_heads=[1, 2, 4, 8], patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2], sr_ratios=[8, 4, 2, 1], out_indices=(0, 1, 2, 3), mlp_ratio=4, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, act_cfg={'type': 'GELU'}, norm_cfg={'type': 'LayerNorm', 'eps': 1e-06}, use_checkpoint=False, pretrained=True, pretrained_model_path=''):
        super(MixVisionTransformer, self).__init__()
        self.structure_type = structure_type
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.out_indices = out_indices
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.use_checkpoint = use_checkpoint
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        assert num_stages == len(num_layers) == len(num_heads) == len(patch_sizes) == len(strides) == len(sr_ratios)
        assert max(out_indices) < num_stages
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and getattr(self, key) == value
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))]
        cur, self.layers = 0, nn.ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            patch_embed = PatchEmbed(in_channels=in_channels, embed_dims=embed_dims_i, kernel_size=patch_sizes[i], stride=strides[i], padding=patch_sizes[i] // 2, norm_cfg=norm_cfg)
            layer = nn.ModuleList([TransformerEncoderLayer(embed_dims=embed_dims_i, num_heads=num_heads[i], feedforward_channels=mlp_ratio * embed_dims_i, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=dpr[cur + idx], qkv_bias=qkv_bias, act_cfg=act_cfg, norm_cfg=norm_cfg, use_checkpoint=use_checkpoint, sr_ratio=sr_ratios[i]) for idx in range(num_layer)])
            in_channels = embed_dims_i
            norm = BuildNormalization(placeholder=embed_dims_i, norm_cfg=norm_cfg)
            self.layers.append(nn.ModuleList([patch_embed, layer, norm]))
            cur += num_layer
        if pretrained:
            self.loadpretrainedweights(structure_type, pretrained_model_path)
    """loadpretrainedweights"""

    def loadpretrainedweights(self, structure_type='', pretrained_model_path=''):
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        else:
            checkpoint = model_zoo.load_url(DEFAULT_MODEL_URLS[structure_type], map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        state_dict = self.mitconvert(state_dict)
        self.load_state_dict(state_dict, strict=False)
    """mitconvert"""

    @staticmethod
    def mitconvert(ckpt):
        from collections import OrderedDict
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            if k.startswith('head'):
                continue
            elif k.startswith('patch_embed'):
                stage_i = int(k.split('.')[0].replace('patch_embed', ''))
                new_k = k.replace(f'patch_embed{stage_i}', f'layers.{stage_i - 1}.0')
                new_v = v
                if 'proj.' in new_k:
                    new_k = new_k.replace('proj.', 'projection.')
            elif k.startswith('block'):
                stage_i = int(k.split('.')[0].replace('block', ''))
                new_k = k.replace(f'block{stage_i}', f'layers.{stage_i - 1}.1')
                new_v = v
                if 'attn.q.' in new_k:
                    sub_item_k = k.replace('q.', 'kv.')
                    new_k = new_k.replace('q.', 'attn.in_proj_')
                    new_v = torch.cat([v, ckpt[sub_item_k]], dim=0)
                elif 'attn.kv.' in new_k:
                    continue
                elif 'attn.proj.' in new_k:
                    new_k = new_k.replace('proj.', 'attn.out_proj.')
                elif 'attn.sr.' in new_k:
                    new_k = new_k.replace('sr.', 'sr.')
                elif 'mlp.' in new_k:
                    string = f'{new_k}-'
                    new_k = new_k.replace('mlp.', 'ffn.layers.')
                    if 'fc1.weight' in new_k or 'fc2.weight' in new_k:
                        new_v = v.reshape((*v.shape, 1, 1))
                    new_k = new_k.replace('fc1.', '0.')
                    new_k = new_k.replace('dwconv.dwconv.', '1.')
                    new_k = new_k.replace('fc2.', '4.')
                    string += f'{new_k} {v.shape}-{new_v.shape}'
            elif k.startswith('norm'):
                stage_i = int(k.split('.')[0].replace('norm', ''))
                new_k = k.replace(f'norm{stage_i}', f'layers.{stage_i - 1}.2')
                new_v = v
            else:
                new_k = k
                new_v = v
            new_ckpt[new_k] = new_v
        return new_ckpt
    """forward"""

    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlctonchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)
        return outs


class MobileNetV2(nn.Module):
    arch_settings = [[1, 16, 1], [6, 24, 2], [6, 32, 3], [6, 64, 4], [6, 96, 3], [6, 160, 3], [6, 320, 1]]

    def __init__(self, structure_type, in_channels=3, widen_factor=1, outstride=8, out_indices=(1, 2, 4, 6), norm_cfg={'type': 'SyncBatchNorm'}, act_cfg={'type': 'ReLU6', 'inplace': True}, pretrained=True, pretrained_model_path=''):
        super(MobileNetV2, self).__init__()
        self.out_indices = out_indices
        self.structure_type = structure_type
        self.in_channels = in_channels
        self.widen_factor = widen_factor
        self.outstride = outstride
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and getattr(self, key) == value
        outstride_to_strides_and_dilations = {(8): ((1, 2, 2, 1, 1, 1, 1), (1, 1, 1, 2, 2, 4, 4)), (16): ((1, 2, 2, 2, 1, 1, 1), (1, 1, 1, 1, 1, 2, 2)), (32): ((1, 2, 2, 2, 1, 2, 1), (1, 1, 1, 1, 1, 1, 1))}
        assert outstride in outstride_to_strides_and_dilations, 'unsupport outstride %s in MobileNetV2' % outstride
        stride_list, dilation_list = outstride_to_strides_and_dilations[outstride]
        self.in_channels = makedivisible(32 * widen_factor, 8)
        self.conv1 = nn.Sequential()
        self.conv1.add_module('conv', nn.Conv2d(in_channels, self.in_channels, kernel_size=3, stride=2, padding=1, bias=False))
        self.conv1.add_module('bn', BuildNormalization(placeholder=self.in_channels, norm_cfg=norm_cfg))
        self.conv1.add_module('activation', BuildActivation(act_cfg))
        self.layers = []
        for i, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks = layer_cfg
            stride = stride_list[i]
            dilation = dilation_list[i]
            out_channels = makedivisible(channel * widen_factor, 8)
            inverted_res_layer = self.makelayer(out_channels, num_blocks, stride, dilation, expand_ratio, norm_cfg, act_cfg)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)
        if pretrained:
            self.loadpretrainedweights(structure_type, pretrained_model_path)
    """loadpretrainedweights"""

    def loadpretrainedweights(self, structure_type='mobilenetv2', pretrained_model_path=''):
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        else:
            checkpoint = model_zoo.load_url(DEFAULT_MODEL_URLS[structure_type], map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith('backbone.'):
                value = state_dict.pop(key)
                key = '.'.join(key.split('.')[1:])
                state_dict[key] = value
        self.load_state_dict(state_dict, strict=False)
    """forward"""

    def forward(self, x):
        x = self.conv1(x)
        outs = []
        for idx, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if idx in self.out_indices:
                outs.append(x)
        return tuple(outs)
    """makelayer"""

    def makelayer(self, out_channels, num_blocks, stride, dilation, expand_ratio, norm_cfg=None, act_cfg=None):
        if act_cfg is None:
            act_cfg = {'type': 'ReLU6', 'inplace': True}
        layers = []
        for i in range(num_blocks):
            layers.append(InvertedResidual(self.in_channels, out_channels, stride=stride if i == 0 else 1, expand_ratio=expand_ratio, dilation=dilation if i == 0 else 1, norm_cfg=norm_cfg, act_cfg=act_cfg))
            self.in_channels = out_channels
        return nn.Sequential(*layers)


class MobileNetV3(nn.Module):
    arch_settings = {'small': [[3, 16, 16, True, {'type': 'ReLU'}, 2], [3, 72, 24, False, {'type': 'ReLU'}, 2], [3, 88, 24, False, {'type': 'ReLU'}, 1], [5, 96, 40, True, {'type': 'HardSwish'}, 2], [5, 240, 40, True, {'type': 'HardSwish'}, 1], [5, 240, 40, True, {'type': 'HardSwish'}, 1], [5, 120, 48, True, {'type': 'HardSwish'}, 1], [5, 144, 48, True, {'type': 'HardSwish'}, 1], [5, 288, 96, True, {'type': 'HardSwish'}, 2], [5, 576, 96, True, {'type': 'HardSwish'}, 1], [5, 576, 96, True, {'type': 'HardSwish'}, 1]], 'large': [[3, 16, 16, False, {'type': 'ReLU'}, 1], [3, 64, 24, False, {'type': 'ReLU'}, 2], [3, 72, 24, False, {'type': 'ReLU'}, 1], [5, 72, 40, True, {'type': 'ReLU'}, 2], [5, 120, 40, True, {'type': 'ReLU'}, 1], [5, 120, 40, True, {'type': 'ReLU'}, 1], [3, 240, 80, False, {'type': 'HardSwish'}, 2], [3, 200, 80, False, {'type': 'HardSwish'}, 1], [3, 184, 80, False, {'type': 'HardSwish'}, 1], [3, 184, 80, False, {'type': 'HardSwish'}, 1], [3, 480, 112, True, {'type': 'HardSwish'}, 1], [3, 672, 112, True, {'type': 'HardSwish'}, 1], [5, 672, 160, True, {'type': 'HardSwish'}, 2], [5, 960, 160, True, {'type': 'HardSwish'}, 1], [5, 960, 160, True, {'type': 'HardSwish'}, 1]]}

    def __init__(self, structure_type, in_channels=3, arch_type='large', outstride=8, out_indices=(1, 3, 16), reduction_factor=1, norm_cfg={'type': 'SyncBatchNorm'}, act_cfg={'type': 'HardSwish'}, pretrained=True, pretrained_model_path=''):
        super(MobileNetV3, self).__init__()
        self.structure_type = structure_type
        self.in_channels = in_channels
        self.arch_type = arch_type
        self.outstride = outstride
        self.out_indices = out_indices
        self.reduction_factor = reduction_factor
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        assert arch_type in self.arch_settings
        assert isinstance(reduction_factor, int) and reduction_factor > 0
        assert outstride in [8, 16, 32], 'unsupport outstride %s in MobileNetV3' % outstride
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and getattr(self, key) == value
        self.layers = self.makelayers(in_channels, arch_type, reduction_factor, outstride, norm_cfg, act_cfg)
        if pretrained:
            self.loadpretrainedweights(structure_type, pretrained_model_path)
    """loadpretrainedweights"""

    def loadpretrainedweights(self, structure_type='mobilenetv3_small', pretrained_model_path=''):
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        else:
            checkpoint = model_zoo.load_url(DEFAULT_MODEL_URLS[structure_type], map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith('backbone.'):
                value = state_dict.pop(key)
                key = '.'.join(key.split('.')[1:])
                state_dict[key] = value
        self.load_state_dict(state_dict, strict=False)
    """makelayers"""

    def makelayers(self, in_channels, arch_type, reduction_factor, outstride, norm_cfg=None, act_cfg=None):
        layers, act_cfg_default = [], act_cfg.copy()
        in_channels_first_layer, in_channels = in_channels, 16
        layer = nn.Sequential()
        layer.add_module('conv', AdptivePaddingConv2d(in_channels_first_layer, in_channels, kernel_size=3, stride=2, padding=1, bias=False))
        layer.add_module('bn', BuildNormalization(placeholder=in_channels, norm_cfg=norm_cfg))
        layer.add_module('activation', BuildActivation(act_cfg_default))
        self.add_module('layer0', layer)
        layers.append('layer0')
        layer_setting = self.arch_settings[arch_type]
        for i, params in enumerate(layer_setting):
            kernel_size, mid_channels, out_channels, with_se, act_cfg, stride = params
            if arch_type == 'large' and i >= 12 or arch_type == 'small' and i >= 8:
                mid_channels = mid_channels // reduction_factor
                out_channels = out_channels // reduction_factor
            se_cfg = None
            if with_se:
                se_cfg = {'channels': mid_channels, 'ratio': 4, 'act_cfgs': ({'type': 'ReLU'}, {'type': 'HardSigmoid', 'bias': 3.0, 'divisor': 6.0})}
            layer = InvertedResidualV3(in_channels=in_channels, out_channels=out_channels, mid_channels=mid_channels, kernel_size=kernel_size, stride=stride, se_cfg=se_cfg, with_expand_conv=in_channels != mid_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
            in_channels = out_channels
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, layer)
            layers.append(layer_name)
        out_channels = 576 if arch_type == 'small' else 960
        layer = nn.Sequential()
        layer.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, dilation={(8): 4, (16): 2, (32): 1}[outstride], padding=0, bias=False))
        layer.add_module('bn', BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg))
        layer.add_module('activation', BuildActivation(act_cfg_default))
        layer_name = 'layer{}'.format(len(layer_setting) + 1)
        self.add_module(layer_name, layer)
        layers.append(layer_name)
        if outstride == 32:
            return layers
        if arch_type == 'small':
            self.layer4.depthwise_conv[0].stride = 1, 1
            if outstride == 8:
                self.layer9.depthwise_conv[0].stride = 1, 1
            for i in range(4, len(layers)):
                layer = getattr(self, layers[i])
                if isinstance(layer, InvertedResidualV3):
                    modified_module = layer.depthwise_conv[0]
                else:
                    modified_module = layer[0]
                if i < 9 or outstride == 16:
                    modified_module.dilation = 2, 2
                    pad = 2
                else:
                    modified_module.dilation = 4, 4
                    pad = 4
                if not isinstance(modified_module, AdptivePaddingConv2d):
                    pad *= (modified_module.kernel_size[0] - 1) // 2
                    modified_module.padding = pad, pad
        else:
            self.layer7.depthwise_conv[0].stride = 1, 1
            if outstride == 8:
                self.layer13.depthwise_conv[0].stride = 1, 1
            for i in range(7, len(layers)):
                layer = getattr(self, layers[i])
                if isinstance(layer, InvertedResidualV3):
                    modified_module = layer.depthwise_conv[0]
                else:
                    modified_module = layer[0]
                if i < 13 or outstride == 16:
                    modified_module.dilation = 2, 2
                    pad = 2
                else:
                    modified_module.dilation = 4, 4
                    pad = 4
                if not isinstance(modified_module, AdptivePaddingConv2d):
                    pad *= (modified_module.kernel_size[0] - 1) // 2
                    modified_module.padding = pad, pad
        return layers
    """forward"""

    def forward(self, x):
        outs = []
        for idx, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if idx in self.out_indices:
                outs.append(x)
        return tuple(outs)


class MBConv(nn.Module):

    def __init__(self, in_chans, out_chans, expand_ratio, act_cfg={'type': 'GELU'}, drop_path=0.0):
        super(MBConv, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.conv1 = Conv2dBN(in_chans, self.hidden_chans, kernel_size=1, stride=1, padding=0)
        self.act1 = BuildActivation(act_cfg=act_cfg)
        self.conv2 = Conv2dBN(self.hidden_chans, self.hidden_chans, kernel_size=3, stride=1, padding=1, groups=self.hidden_chans)
        self.act2 = BuildActivation(act_cfg=act_cfg)
        self.conv3 = Conv2dBN(self.hidden_chans, out_chans, kernel_size=1, stride=1, padding=0)
        self.act3 = BuildActivation(act_cfg=act_cfg)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    """forward"""

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.drop_path(x)
        x += shortcut
        x = self.act3(x)
        return x


class ConvLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, act_cfg={'type': 'GELU'}, drop_path=0.0, downsample=None, use_checkpoint=False, out_dim=None, conv_expand_ratio=4.0):
        super(ConvLayer, self).__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.input_resolution = PatchEmbed.totuple(input_resolution)
        self.blocks = nn.ModuleList([MBConv(dim, dim, conv_expand_ratio, act_cfg, drop_path[i] if isinstance(drop_path, list) else drop_path) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, act_cfg=act_cfg)
        else:
            self.downsample = None
    """forward"""

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


def getsdpasettings():
    if torch.cuda.is_available():
        old_gpu = torch.cuda.get_device_properties(0).major < 7
        use_flash_attn = torch.cuda.get_device_properties(0).major >= 8
        if not use_flash_attn:
            warnings.warn('Flash Attention is disabled as it requires a GPU with Ampere (8.0) CUDA capability.', category=UserWarning, stacklevel=2)
        pytorch_version = tuple(int(v) for v in torch.__version__.split('.')[:2])
        if pytorch_version < (2, 2):
            warnings.warn(f'You are using PyTorch {torch.__version__} without Flash Attention v2 support. Consider upgrading to PyTorch 2.2+ for Flash Attention v2 (which could be faster).', category=UserWarning, stacklevel=2)
        math_kernel_on = pytorch_version < (2, 2) or not use_flash_attn
    else:
        old_gpu = True
        use_flash_attn = False
        math_kernel_on = True
    return old_gpu, use_flash_attn, math_kernel_on


OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = getsdpasettings()


class Attention(nn.Module):

    def __init__(self, embedding_dim, num_heads, downsample_rate=1, dropout=0.0, kv_in_dim=None):
        super(Attention, self).__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, 'num_heads must divide embedding_dim.'
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.dropout_p = dropout
    """separateheads"""

    def separateheads(self, x, num_heads):
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)
    """recombineheads"""

    def recombineheads(self, x):
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)
    """forward"""

    def forward(self, q, k, v):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = self.separateheads(q, self.num_heads)
        k = self.separateheads(k, self.num_heads)
        v = self.separateheads(v, self.num_heads)
        dropout_p = self.dropout_p if self.training else 0.0
        with torch.backends.cuda.sdp_kernel(enable_flash=USE_FLASH_ATTN, enable_math=OLD_GPU and dropout_p > 0.0 or MATH_KERNEL_ON, enable_mem_efficient=OLD_GPU):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        out = self.recombineheads(out)
        out = self.out_proj(out)
        return out


class TinyViTBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, mlp_ratio=4.0, drop=0.0, drop_path=0.0, local_conv_size=3, act_cfg={'type': 'GELU'}):
        super(TinyViTBlock, self).__init__()
        assert window_size > 0, 'window_size must be greater than 0'
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        self.input_resolution = PatchEmbed.totuple(input_resolution)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.attn = Attention(dim, dim // num_heads, num_heads, attn_ratio=1, resolution=PatchEmbed.totuple(window_size))
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_cfg=act_cfg, drop=drop)
        self.local_conv = Conv2dBN(dim, dim, kernel_size=local_conv_size, stride=1, padding=local_conv_size // 2, groups=dim)
    """forward"""

    def forward(self, x):
        x_identity = x
        batch_size, num_pixels, num_channels = x.shape
        assert x.shape[1] == self.input_resolution[0] * self.input_resolution[1], 'input feature has wrong size'
        if self.input_resolution[0] == self.window_size and self.input_resolution[1] == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(batch_size, self.input_resolution[0], self.input_resolution[1], num_channels)
            pad_b = (self.window_size - self.input_resolution[0] % self.window_size) % self.window_size
            pad_r = (self.window_size - self.input_resolution[1] % self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0
            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
            pH, pW = self.input_resolution[0] + pad_b, self.input_resolution[1] + pad_r
            nH, nW = pH // self.window_size, pW // self.window_size
            x = x.view(batch_size, nH, self.window_size, nW, self.window_size, num_channels).transpose(2, 3).reshape(batch_size * nH * nW, self.window_size * self.window_size, num_channels)
            x = self.attn(x)
            x = x.view(batch_size, nH, nW, self.window_size, self.window_size, num_channels).transpose(2, 3).reshape(batch_size, pH, pW, num_channels)
            if padding:
                x = x[:, :self.input_resolution[0], :self.input_resolution[1]].contiguous()
            x = x.view(batch_size, num_pixels, num_channels)
        x = x_identity + self.drop_path(x)
        x = x.transpose(1, 2).reshape(batch_size, num_channels, self.input_resolution[0], self.input_resolution[1])
        x = self.local_conv(x)
        x = x.view(batch_size, num_channels, num_pixels).transpose(1, 2)
        x = x + self.drop_path(self.mlp(x))
        return x


class BasicLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.0, drop=0.0, drop_path=0.0, downsample=None, use_checkpoint=False, local_conv_size=3, act_cfg={'type': 'GELU'}, out_dim=None):
        super(BasicLayer, self).__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.input_resolution = PatchEmbed.totuple(input_resolution)
        self.blocks = nn.ModuleList([TinyViTBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, local_conv_size=local_conv_size, act_cfg=act_cfg) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, act_cfg=act_cfg)
        else:
            self.downsample = None
    """forward"""

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class MobileSAMTinyViT(nn.Module):

    def __init__(self, structure_type, img_size=224, in_chans=3, embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_sizes=[7, 7, 14, 7], mlp_ratio=4.0, drop_rate=0.0, drop_path_rate=0.1, use_checkpoint=False, mbconv_expand_ratio=4.0, local_conv_size=3, act_cfg={'type': 'GELU'}, pretrained=False, pretrained_model_path=''):
        super(MobileSAMTinyViT, self).__init__()
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0], resolution=img_size, act_cfg=act_cfg)
        self.depths = depths
        self.img_size = img_size
        self.in_chans = in_chans
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.embed_dims = embed_dims
        self.num_layers = len(depths)
        self.window_sizes = window_sizes
        self.use_checkpoint = use_checkpoint
        self.drop_path_rate = drop_path_rate
        self.local_conv_size = local_conv_size
        self.mbconv_expand_ratio = mbconv_expand_ratio
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and getattr(self, key) == value
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            kwargs = dict(dim=embed_dims[layer_idx], depth=depths[layer_idx], drop_path=dpr[sum(depths[:layer_idx]):sum(depths[:layer_idx + 1])], downsample=PatchMerging if layer_idx < self.num_layers - 1 else None, input_resolution=(patches_resolution[0] // 2 ** (layer_idx - 1 if layer_idx == 3 else layer_idx), patches_resolution[1] // 2 ** (layer_idx - 1 if layer_idx == 3 else layer_idx)), use_checkpoint=use_checkpoint, out_dim=embed_dims[min(layer_idx + 1, len(embed_dims) - 1)], act_cfg=act_cfg)
            if layer_idx == 0:
                layer = ConvLayer(conv_expand_ratio=mbconv_expand_ratio, **kwargs)
            else:
                layer = BasicLayer(num_heads=num_heads[layer_idx], window_size=window_sizes[layer_idx], mlp_ratio=self.mlp_ratio, drop=drop_rate, local_conv_size=local_conv_size, **kwargs)
            self.layers.append(layer)
        self.neck = nn.Sequential(nn.Conv2d(embed_dims[-1], 256, kernel_size=1, stride=1, padding=0, bias=False), LayerNorm2d(256), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False), LayerNorm2d(256))
        if pretrained:
            state_dict = loadpretrainedweights(structure_type=structure_type, pretrained_model_path=pretrained_model_path, default_model_urls=DEFAULT_MODEL_URLS)
            self.load_state_dict(state_dict, strict=False)
    """forward"""

    def forward(self, x, return_interm_embeddings=False):
        x = self.patch_embed(x)
        interm_embeddings = []
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx == 1:
                interm_embeddings.append(x.view(x.shape[0], 64, 64, -1))
        x = x.view(x.shape[0], 64, 64, x.shape[-1])
        x = x.permute(0, 3, 1, 2)
        x = self.neck(x)
        if return_interm_embeddings:
            return x, interm_embeddings
        return x


class MobileVitBlock(nn.Module):

    def __init__(self, in_channels, transformer_dim, ffn_dim, out_channels, conv_ksize=3, norm_cfg=dict(type='SyncBatchNorm'), act_cfg=dict(type='Swish'), num_transformer_blocks=2, patch_size=2, num_heads=4, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, no_fusion=False, transformer_norm_cfg=dict(type='LayerNorm')):
        super(MobileVitBlock, self).__init__()
        self.local_rep = nn.Sequential(nn.Sequential(collections.OrderedDict([('conv', nn.Conv2d(in_channels, in_channels, kernel_size=conv_ksize, stride=1, padding=int((conv_ksize - 1) / 2), bias=False)), ('bn', BuildNormalization(placeholder=in_channels, norm_cfg=norm_cfg)), ('activate', BuildActivation(act_cfg))])), nn.Sequential(collections.OrderedDict([('conv', nn.Conv2d(in_channels, transformer_dim, kernel_size=1, stride=1, padding=0, bias=False))])))
        global_rep = [TransformerEncoderLayer(embed_dims=transformer_dim, num_heads=num_heads, feedforward_channels=ffn_dim, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, qkv_bias=True, act_cfg=dict(type='Swish'), norm_cfg=transformer_norm_cfg) for _ in range(num_transformer_blocks)]
        global_rep.append(BuildNormalization(placeholder=transformer_dim, norm_cfg=transformer_norm_cfg))
        self.global_rep = nn.Sequential(*global_rep)
        self.conv_proj = nn.Sequential(collections.OrderedDict([('conv', nn.Conv2d(transformer_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False)), ('bn', BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg))]))
        if no_fusion:
            self.conv_fusion = None
        else:
            self.conv_fusion = nn.Sequential(collections.OrderedDict([('conv', nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=conv_ksize, stride=1, padding=int((conv_ksize - 1) / 2), bias=False)), ('bn', BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg)), ('activate', BuildActivation(act_cfg))]))
        self.patch_size = patch_size, patch_size
        self.patch_area = self.patch_size[0] * self.patch_size[1]
    """forward"""

    def forward(self, x):
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


class MobileViT(nn.Module):
    arch_settings = {'small': [['mobilenetv2', 32, 1, 1, 4], ['mobilenetv2', 64, 2, 3, 4], ['mobilevit', 96, 2, 144, 288, 2, 4], ['mobilevit', 128, 2, 192, 384, 4, 4], ['mobilevit', 160, 2, 240, 480, 3, 4]], 'x_small': [['mobilenetv2', 32, 1, 1, 4], ['mobilenetv2', 48, 2, 3, 4], ['mobilevit', 64, 2, 96, 192, 2, 4], ['mobilevit', 80, 2, 120, 240, 4, 4], ['mobilevit', 96, 2, 144, 288, 3, 4]], 'xx_small': [['mobilenetv2', 16, 1, 1, 2], ['mobilenetv2', 24, 2, 3, 2], ['mobilevit', 48, 2, 64, 128, 2, 2], ['mobilevit', 64, 2, 80, 160, 4, 2], ['mobilevit', 80, 2, 96, 192, 3, 2]]}

    def __init__(self, structure_type, arch='small', in_channels=3, stem_channels=16, last_exp_factor=4, out_indices=(0, 1, 2, 3, 4), norm_cfg=dict(type='SyncBatchNorm'), act_cfg=dict(type='Swish'), pretrained=True, pretrained_model_path=''):
        super(MobileViT, self).__init__()
        arch = arch.lower()
        assert arch in self.arch_settings
        arch = self.arch_settings[arch]
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, collections.abc.Sequence)
        self.arch = arch
        self.num_stages = len(arch)
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_stages + index
                assert out_indices[i] >= 0, f'invalid out_indices {index}'
        self.out_indices = out_indices
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and getattr(self, key) == value
        _make_layer_func = {'mobilenetv2': self.makemobilenetv2layer, 'mobilevit': self.makemobilevitlayer}
        self.stem = nn.Sequential(collections.OrderedDict([('conv', nn.Conv2d(in_channels, stem_channels, kernel_size=3, stride=2, padding=1, bias=False)), ('bn', BuildNormalization(placeholder=stem_channels, norm_cfg=norm_cfg)), ('activate', BuildActivation(act_cfg))]))
        in_channels = stem_channels
        layers = []
        for i, layer_settings in enumerate(arch):
            layer_type, settings = layer_settings[0], layer_settings[1:]
            layer, out_channels = _make_layer_func[layer_type](in_channels, norm_cfg, act_cfg, *settings)
            layers.append(layer)
            in_channels = out_channels
        self.layers = nn.Sequential(*layers)
        self.conv_1x1_exp = nn.Sequential(collections.OrderedDict([('conv', nn.Conv2d(in_channels, last_exp_factor * in_channels, kernel_size=1, stride=1, padding=0, bias=False)), ('bn', BuildNormalization(placeholder=last_exp_factor * in_channels, norm_cfg=norm_cfg)), ('activate', BuildActivation(act_cfg))]))
        if pretrained and os.path.exists(pretrained_model_path):
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
            state_dict = self.convertstatedict(checkpoint)
            self.load_state_dict(state_dict, strict=True)
        elif pretrained:
            checkpoint = model_zoo.load_url(DEFAULT_MODEL_URLS[structure_type], map_location='cpu')
            state_dict = self.convertstatedict(checkpoint)
            self.load_state_dict(state_dict, strict=True)
    """convertstatedict"""

    @staticmethod
    def convertstatedict(checkpoint):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith('backbone.'):
                value = state_dict.pop(key)
                key = '.'.join(key.split('.')[1:])
                state_dict[key] = value
            if key.startswith('head'):
                state_dict.pop(key)
            if 'attn.qkv' in key:
                new_key = key.replace('attn.qkv.', 'attn.attn.in_proj_')
                assert new_key not in state_dict
                state_dict[new_key] = state_dict.pop(key)
            if 'attn.proj' in key:
                new_key = key.replace('attn.proj', 'attn.attn.out_proj')
                assert new_key not in state_dict
                state_dict[new_key] = state_dict.pop(key)
        return state_dict
    """makemobilevitlayer"""

    @staticmethod
    def makemobilevitlayer(in_channels, norm_cfg, act_cfg, out_channels, stride, transformer_dim, ffn_dim, num_transformer_blocks, expand_ratio=4):
        layer = []
        layer.append(InvertedResidual(in_channels=in_channels, out_channels=out_channels, stride=stride, expand_ratio=expand_ratio, act_cfg=act_cfg, norm_cfg=norm_cfg))
        layer.append(MobileVitBlock(in_channels=out_channels, transformer_dim=transformer_dim, ffn_dim=ffn_dim, out_channels=out_channels, num_transformer_blocks=num_transformer_blocks, act_cfg=act_cfg, norm_cfg=norm_cfg))
        return nn.Sequential(*layer), out_channels
    """makemobilenetv2layer"""

    @staticmethod
    def makemobilenetv2layer(in_channels, norm_cfg, act_cfg, out_channels, stride, num_blocks, expand_ratio=4):
        layer = []
        for i in range(num_blocks):
            stride = stride if i == 0 else 1
            layer.append(InvertedResidual(in_channels=in_channels, out_channels=out_channels, stride=stride, expand_ratio=expand_ratio, act_cfg=act_cfg, norm_cfg=norm_cfg))
            in_channels = out_channels
        return nn.Sequential(*layer), out_channels
    """forward"""

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1:
                x = self.conv_1x1_exp(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


class LinearSelfAttention(nn.Module):

    def __init__(self, embed_dim, attn_drop=0.0, proj_drop=0.0, bias=True):
        super(LinearSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.qkv_proj = nn.Conv2d(in_channels=embed_dim, out_channels=1 + 2 * embed_dim, bias=bias, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, bias=bias, kernel_size=1)
        self.out_drop = nn.Dropout(proj_drop)
    """forwardselfattn"""

    def forwardselfattn(self, x):
        qkv = self.qkv_proj(x)
        query, key, value = qkv.split([1, self.embed_dim, self.embed_dim], dim=1)
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_drop(context_scores)
        context_vector = (key * context_scores).sum(dim=-1, keepdim=True)
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        out = self.out_drop(out)
        return out
    """forwardcrossattn"""

    @torch.jit.ignore()
    def forwardcrossattn(self, x, x_prev=None):
        batch_size, in_dim, kv_patch_area, kv_num_patches = x.shape
        q_patch_area, q_num_patches = x.shape[-2:]
        assert kv_patch_area == q_patch_area, 'the number of pixels in a patch for query and key_value should be the same'
        qk = F.conv2d(x_prev, weight=self.qkv_proj.weight[:self.embed_dim + 1], bias=self.qkv_proj.bias[:self.embed_dim + 1])
        query, key = qk.split([1, self.embed_dim], dim=1)
        value = F.conv2d(x, weight=self.qkv_proj.weight[self.embed_dim + 1], bias=self.qkv_proj.bias[self.embed_dim + 1] if self.qkv_proj.bias is not None else None)
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_drop(context_scores)
        context_vector = (key * context_scores).sum(dim=-1, keepdim=True)
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        out = self.out_drop(out)
        return out
    """forward"""

    def forward(self, x, x_prev=None):
        if x_prev is None:
            return self.forwardselfattn(x)
        else:
            return self.forwardcrossattn(x, x_prev=x_prev)


class ConvMlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_cfg={'type': 'ReLU', 'inplace': True}, norm_cfg=None, bias=True, drop=0.0):
        super(ConvMlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = self.totuple(bias)
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = BuildNormalization(placeholder=hidden_features, norm_cfg=norm_cfg)
        self.act = BuildActivation(act_cfg)
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])
    """forward"""

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
    """totuple"""

    @staticmethod
    def totuple(x):
        if isinstance(x, (int, bool, float)):
            return x, x
        assert isinstance(x, collections.abc.Sequence) and len(x) == 2
        for n in x:
            assert isinstance(n, (int, bool, float))
        return tuple(x)


class LinearTransformerBlock(nn.Module):

    def __init__(self, embed_dim, mlp_ratio=2.0, drop=0.0, attn_drop=0.0, drop_path=0.0, act_cfg={'type': 'SiLU'}, norm_cfg={'type': 'GroupNorm', 'num_groups': 1}):
        super(LinearTransformerBlock, self).__init__()
        self.norm1 = BuildNormalization(placeholder=embed_dim, norm_cfg=norm_cfg)
        self.attn = LinearSelfAttention(embed_dim=embed_dim, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = BuildDropout({'type': 'DropPath', 'drop_prob': drop_path})
        self.norm2 = BuildNormalization(placeholder=embed_dim, norm_cfg=norm_cfg)
        self.mlp = ConvMlp(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), act_cfg=act_cfg, drop=drop)
        self.drop_path2 = BuildDropout({'type': 'DropPath', 'drop_prob': drop_path})
    """forward"""

    def forward(self, x, x_prev=None):
        if x_prev is None:
            x = x + self.drop_path1(self.attn(self.norm1(x)))
        else:
            res = x
            x = self.norm1(x)
            x = self.attn(x, x_prev)
            x = self.drop_path1(x) + res
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class MobileVitV2Block(nn.Module):

    def __init__(self, in_channels, out_channels=None, kernel_size=3, bottle_ratio=1.0, group_size=1, dilation=1, mlp_ratio=2.0, transformer_dim=None, transformer_depth=2, patch_size=2, attn_drop=0.0, drop=0.0, drop_path_rate=0.0, transformer_norm_cfg={'type': 'GroupNorm', 'num_groups': 1}, norm_cfg=dict(type='SyncBatchNorm'), act_cfg={'type': 'ReLU', 'inplace': True}):
        super(MobileVitV2Block, self).__init__()
        if not group_size:
            groups = 1
        else:
            groups = in_channels // group_size
        out_channels = out_channels or in_channels
        transformer_dim = transformer_dim or makedivisible(bottle_ratio * in_channels, 8)
        assert int((kernel_size - 1) / 2) == dilation
        self.conv_kxk = nn.Sequential(collections.OrderedDict([('conv', nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, groups=groups, padding=int((kernel_size - 1) / 2), bias=False, dilation=dilation)), ('bn', BuildNormalization(placeholder=in_channels, norm_cfg=norm_cfg)), ('activate', BuildActivation(act_cfg))]))
        self.conv_1x1 = nn.Conv2d(in_channels, transformer_dim, kernel_size=1, bias=False)
        self.transformer = nn.Sequential(*[LinearTransformerBlock(embed_dim=transformer_dim, mlp_ratio=mlp_ratio, attn_drop=attn_drop, drop=drop, drop_path=drop_path_rate, act_cfg=act_cfg, norm_cfg=transformer_norm_cfg) for _ in range(transformer_depth)])
        self.norm = BuildNormalization(placeholder=transformer_dim, norm_cfg=transformer_norm_cfg)
        self.conv_proj = nn.Sequential(collections.OrderedDict([('conv', nn.Conv2d(transformer_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False)), ('bn', BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg))]))
        self.patch_size = ConvMlp.totuple(patch_size)
        self.patch_area = self.patch_size[0] * self.patch_size[1]
    """forward"""

    def forward(self, x):
        B, C, H, W = x.shape
        patch_h, patch_w = self.patch_size
        new_h, new_w = math.ceil(H / patch_h) * patch_h, math.ceil(W / patch_w) * patch_w
        num_patch_h, num_patch_w = new_h // patch_h, new_w // patch_w
        num_patches = num_patch_h * num_patch_w
        if new_h != H or new_w != W:
            x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=True)
        x = self.conv_kxk(x)
        x = self.conv_1x1(x)
        C = x.shape[1]
        x = x.reshape(B, C, num_patch_h, patch_h, num_patch_w, patch_w).permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C, -1, num_patches)
        x = self.transformer(x)
        x = self.norm(x)
        x = x.reshape(B, C, patch_h, patch_w, num_patch_h, num_patch_w).permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        x = self.conv_proj(x)
        return x


class MobileViTV2(nn.Module):
    arch_settings = {'mobilevitv2_050': [[['mobilenetv2', 32, 1, 1, 2], ['mobilenetv2', 64, 2, 2, 2], ['mobilevitv2', 128, 2, 0.5, 2, 1, 2, 2], ['mobilevitv2', 192, 2, 0.5, 4, 1, 2, 2], ['mobilevitv2', 256, 2, 0.5, 3, 1, 2, 2]], 16], 'mobilevitv2_075': [[['mobilenetv2', 48, 1, 1, 2], ['mobilenetv2', 96, 2, 2, 2], ['mobilevitv2', 192, 2, 0.5, 2, 1, 2, 2], ['mobilevitv2', 288, 2, 0.5, 4, 1, 2, 2], ['mobilevitv2', 384, 2, 0.5, 3, 1, 2, 2]], 24], 'mobilevitv2_100': [[['mobilenetv2', 64, 1, 1, 2], ['mobilenetv2', 128, 2, 2, 2], ['mobilevitv2', 256, 2, 0.5, 2, 1, 2, 2], ['mobilevitv2', 384, 2, 0.5, 4, 1, 2, 2], ['mobilevitv2', 512, 2, 0.5, 3, 1, 2, 2]], 32], 'mobilevitv2_125': [[['mobilenetv2', 80, 1, 1, 2], ['mobilenetv2', 160, 2, 2, 2], ['mobilevitv2', 320, 2, 0.5, 2, 1, 2, 2], ['mobilevitv2', 480, 2, 0.5, 4, 1, 2, 2], ['mobilevitv2', 640, 2, 0.5, 3, 1, 2, 2]], 40], 'mobilevitv2_150': [[['mobilenetv2', 96, 1, 1, 2], ['mobilenetv2', 192, 2, 2, 2], ['mobilevitv2', 384, 2, 0.5, 2, 1, 2, 2], ['mobilevitv2', 576, 2, 0.5, 4, 1, 2, 2], ['mobilevitv2', 768, 2, 0.5, 3, 1, 2, 2]], 48], 'mobilevitv2_175': [[['mobilenetv2', 112, 1, 1, 2], ['mobilenetv2', 224, 2, 2, 2], ['mobilevitv2', 448, 2, 0.5, 2, 1, 2, 2], ['mobilevitv2', 672, 2, 0.5, 4, 1, 2, 2], ['mobilevitv2', 896, 2, 0.5, 3, 1, 2, 2]], 56], 'mobilevitv2_200': [[['mobilenetv2', 128, 1, 1, 2], ['mobilenetv2', 256, 2, 2, 2], ['mobilevitv2', 512, 2, 0.5, 2, 1, 2, 2], ['mobilevitv2', 768, 2, 0.5, 4, 1, 2, 2], ['mobilevitv2', 1024, 2, 0.5, 3, 1, 2, 2]], 64]}

    def __init__(self, structure_type, arch='mobilevitv2_050', in_channels=3, out_indices=(0, 1, 2, 3, 4), norm_cfg=dict(type='SyncBatchNorm'), act_cfg=dict(type='SiLU', inplace=True), pretrained=True, pretrained_model_path=''):
        super(MobileViTV2, self).__init__()
        arch = arch.lower()
        assert arch in self.arch_settings
        arch, stem_channels = self.arch_settings[arch]
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, collections.abc.Sequence)
        self.arch = arch
        self.num_stages = len(arch)
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_stages + index
                assert out_indices[i] >= 0, f'invalid out_indices {index}'
        self.out_indices = out_indices
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and getattr(self, key) == value
        self.stem = nn.Sequential(collections.OrderedDict([('conv', nn.Conv2d(in_channels, stem_channels, kernel_size=3, stride=2, padding=1, bias=False)), ('bn', BuildNormalization(placeholder=stem_channels, norm_cfg=norm_cfg)), ('activate', BuildActivation(act_cfg))]))
        _make_layer_func = {'mobilenetv2': self.makemobilenetv2layer, 'mobilevitv2': self.makemobilevitv2layer}
        in_channels = stem_channels
        layers = []
        for i, layer_settings in enumerate(arch):
            layer_type, settings = layer_settings[0], layer_settings[1:]
            layer, out_channels = _make_layer_func[layer_type](in_channels, norm_cfg, act_cfg, *settings)
            layers.append(layer)
            in_channels = out_channels
        self.stages = nn.Sequential(*layers)
        if pretrained:
            self.loadpretrainedweights(structure_type, pretrained_model_path)
    """loadpretrainedweights"""

    def loadpretrainedweights(self, structure_type='mobilevit-small', pretrained_model_path=''):
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        else:
            checkpoint = model_zoo.load_url(DEFAULT_MODEL_URLS[structure_type], map_location='cpu')
        state_dict = self.convertstatedict(checkpoint)
        self.load_state_dict(state_dict, strict=True)
    """convertstatedict"""

    @staticmethod
    def convertstatedict(checkpoint):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith('head'):
                state_dict.pop(key)
            if 'conv1_1x1' in key:
                new_key = key.replace('conv1_1x1', 'conv.0')
                assert new_key not in state_dict, new_key
                state_dict[new_key] = state_dict.pop(key)
            if 'conv2_kxk' in key:
                new_key = key.replace('conv2_kxk', 'conv.1')
                assert new_key not in state_dict, new_key
                state_dict[new_key] = state_dict.pop(key)
            if 'conv3_1x1' in key:
                new_key = key.replace('conv3_1x1', 'conv.2')
                assert new_key not in state_dict, new_key
                state_dict[new_key] = state_dict.pop(key)
        return state_dict
    """makemobilenetv2layer"""

    @staticmethod
    def makemobilenetv2layer(in_channels, norm_cfg, act_cfg, out_channels, stride, num_blocks, expand_ratio=2):
        layer = []
        for i in range(num_blocks):
            stride = stride if i == 0 else 1
            layer.append(InvertedResidual(in_channels=in_channels, out_channels=out_channels, stride=stride, expand_ratio=expand_ratio, act_cfg=act_cfg, norm_cfg=norm_cfg))
            in_channels = out_channels
        return nn.Sequential(*layer), out_channels
    """makemobilevitv2layer"""

    @staticmethod
    def makemobilevitv2layer(in_channels, norm_cfg, act_cfg, out_channels, stride, bottle_ratio, transformer_depth, num_transformer_blocks, mlp_ratio, expand_ratio=2):
        layer = []
        layer.append(InvertedResidual(in_channels=in_channels, out_channels=out_channels, stride=stride, expand_ratio=expand_ratio, act_cfg=act_cfg, norm_cfg=norm_cfg))
        for i in range(num_transformer_blocks):
            layer.append(MobileVitV2Block(in_channels=out_channels, out_channels=out_channels, transformer_depth=transformer_depth, bottle_ratio=bottle_ratio, act_cfg=act_cfg, norm_cfg=norm_cfg, mlp_ratio=mlp_ratio))
        return nn.Sequential(*layer), out_channels
    """forward"""

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, layer in enumerate(self.stages):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


class RSoftmax(nn.Module):

    def __init__(self, radix, groups):
        super(RSoftmax, self).__init__()
        self.radix = radix
        self.groups = groups
    """forward"""

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

    def __init__(self, in_channels, channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, radix=2, reduction_factor=4, norm_cfg=None, act_cfg=None):
        super(SplitAttentionConv2d, self).__init__()
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.conv = nn.Conv2d(in_channels, channels * radix, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups * radix, bias=False)
        self.bn0 = BuildNormalization(placeholder=channels * radix, norm_cfg=norm_cfg)
        self.relu = BuildActivation(act_cfg)
        self.fc1 = nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0, groups=groups)
        self.bn1 = BuildNormalization(placeholder=inter_channels, norm_cfg=norm_cfg)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, kernel_size=1, stride=1, padding=0, groups=groups)
        self.rsoftmax = RSoftmax(radix, groups)
    """forward"""

    def forward(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        x = self.relu(x)
        batch, rchannel = x.shape[:2]
        batch = x.size(0)
        if self.radix > 1:
            splits = x.view(batch, self.radix, -1, *x.shape[2:])
            gap = splits.sum(dim=1)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        if self.radix > 1:
            attens = atten.view(batch, self.radix, -1, *atten.shape[2:])
            out = torch.sum(attens * splits, dim=1)
        else:
            out = atten * x
        return out.contiguous()


class MLPBlock(nn.Module):

    def __init__(self, embedding_dim, mlp_dim, act_cfg={'type': 'GELU'}):
        super(MLPBlock, self).__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = BuildActivation(act_cfg=act_cfg)
    """forward"""

    def forward(self, x):
        return self.lin2(self.act(self.lin1(x)))


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, norm_cfg={'type': 'LayerNorm', 'eps': 1e-06}, act_cfg={'type': 'GELU'}, use_rel_pos=False, rel_pos_zero_init=True, window_size=0, input_size=None):
        super(Block, self).__init__()
        self.norm1 = BuildNormalization(placeholder=dim, norm_cfg=norm_cfg)
        self.attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, use_rel_pos=use_rel_pos, rel_pos_zero_init=rel_pos_zero_init, input_size=input_size if window_size == 0 else (window_size, window_size))
        self.norm2 = BuildNormalization(placeholder=dim, norm_cfg=norm_cfg)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act_cfg=act_cfg)
        self.window_size = window_size
    """forward"""

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = self.windowpartition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = self.windowunpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x
    """windowpartition"""

    def windowpartition(self, x, window_size):
        B, H, W, C = x.shape
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows, (Hp, Wp)
    """windowunpartition"""

    def windowunpartition(self, windows, window_size, pad_hw, hw):
        Hp, Wp = pad_hw
        H, W = hw
        B = windows.shape[0] // (Hp * Wp // window_size // window_size)
        x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
        if Hp > H or Wp > W:
            x = x[:, :H, :W, :].contiguous()
        return x


class SAMViT(nn.Module):

    def __init__(self, img_size=1024, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, out_chans=256, qkv_bias=True, norm_cfg={'type': 'LayerNorm', 'eps': 1e-06}, act_cfg={'type': 'GELU'}, use_abs_pos=True, use_rel_pos=False, rel_pos_zero_init=True, window_size=0, global_attn_indexes=()):
        super(SAMViT, self).__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed(kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), in_chans=in_chans, embed_dim=embed_dim)
        self.pos_embed = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim))
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_cfg=norm_cfg, act_cfg=act_cfg, use_rel_pos=use_rel_pos, rel_pos_zero_init=rel_pos_zero_init, window_size=window_size if i not in global_attn_indexes else 0, input_size=(img_size // patch_size, img_size // patch_size))
            self.blocks.append(block)
        self.neck = nn.Sequential(nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False), LayerNorm2d(out_chans), nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False), LayerNorm2d(out_chans))
    """forward"""

    def forward(self, x, return_interm_embeddings=False):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        interm_embeddings = []
        for blk in self.blocks:
            x = blk(x)
            if blk.window_size == 0:
                interm_embeddings.append(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        if return_interm_embeddings:
            return x, interm_embeddings
        return x


class WindowMSA(nn.Module):

    def __init__(self, embed_dims, num_heads, window_size, qkv_bias=True, qk_scale=None, attn_drop_rate=0.0, proj_drop_rate=0.0):
        super(WindowMSA, self).__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        Wh, Ww = self.window_size
        rel_index_coords = self.doublestepseq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.softmax = nn.Softmax(dim=-1)
    """forward"""

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    """doublestepseq"""

    @staticmethod
    def doublestepseq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(nn.Module):

    def __init__(self, embed_dims, num_heads, window_size, shift_size=0, qkv_bias=True, qk_scale=None, attn_drop_rate=0, proj_drop_rate=0, dropout_cfg=None):
        super(ShiftWindowMSA, self).__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size
        self.w_msa = WindowMSA(embed_dims=embed_dims, num_heads=num_heads, window_size=(window_size, window_size), qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_rate=attn_drop_rate, proj_drop_rate=proj_drop_rate)
        self.drop = BuildDropout(dropout_cfg)
    """forward"""

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]
        if self.shift_size > 0:
            shifted_query = torch.roll(query, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
            w_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = self.windowpartition(img_mask)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None
        query_windows = self.windowpartition(shifted_query)
        query_windows = query_windows.view(-1, self.window_size ** 2, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.windowreverse(attn_windows, H_pad, W_pad)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = self.drop(x)
        return x
    """windowreverse"""

    def windowreverse(self, windows, H, W):
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x
    """windowpartition"""

    def windowpartition(self, x):
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class SwinBlock(nn.Module):

    def __init__(self, embed_dims, num_heads, feedforward_channels, window_size=7, shift=False, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, act_cfg=None, norm_cfg=None, use_checkpoint=False):
        super(SwinBlock, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = BuildNormalization(placeholder=embed_dims, norm_cfg=norm_cfg)
        self.attn = ShiftWindowMSA(embed_dims=embed_dims, num_heads=num_heads, window_size=window_size, shift_size=window_size // 2 if shift else 0, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_rate=attn_drop_rate, proj_drop_rate=drop_rate, dropout_cfg={'type': 'DropPath', 'drop_prob': drop_path_rate})
        self.norm2 = BuildNormalization(placeholder=embed_dims, norm_cfg=norm_cfg)
        self.ffn = FFN(embed_dims=embed_dims, feedforward_channels=feedforward_channels, num_fcs=2, ffn_drop=drop_rate, dropout_cfg={'type': 'DropPath', 'drop_prob': drop_path_rate}, act_cfg=act_cfg, add_identity=True)
    """forward"""

    def forward(self, x, hw_shape):

        def _forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)
            x = x + identity
            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)
            return x
        if self.use_checkpoint and x.requires_grad:
            x = checkpoint.checkpoint(_forward, x)
        else:
            x = _forward(x)
        return x


class SwinBlockSequence(nn.Module):

    def __init__(self, embed_dims, num_heads, feedforward_channels, depth, window_size=7, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, downsample=None, act_cfg=None, norm_cfg=None, use_checkpoint=False):
        super(SwinBlockSequence, self).__init__()
        drop_path_rates = drop_path_rate if isinstance(drop_path_rate, list) else [copy.deepcopy(drop_path_rate) for _ in range(depth)]
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = SwinBlock(embed_dims=embed_dims, num_heads=num_heads, feedforward_channels=feedforward_channels, window_size=window_size, shift=False if i % 2 == 0 else True, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rates[i], act_cfg=act_cfg, norm_cfg=norm_cfg, use_checkpoint=use_checkpoint)
            self.blocks.append(block)
        self.downsample = downsample
    """forward"""

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)
        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


class SwinTransformer(nn.Module):

    def __init__(self, structure_type, pretrain_img_size=224, in_channels=3, embed_dims=96, patch_size=4, window_size=7, mlp_ratio=4, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), strides=(4, 2, 2, 2), out_indices=(0, 1, 2, 3), qkv_bias=True, qk_scale=None, patch_norm=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, use_abs_pos_embed=False, act_cfg={'type': 'GELU'}, norm_cfg={'type': 'LayerNorm'}, use_checkpoint=False, pretrained=True, pretrained_model_path=''):
        super(SwinTransformer, self).__init__()
        self.structure_type = structure_type
        self.pretrain_img_size = pretrain_img_size
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.depths = depths
        self.num_heads = num_heads
        self.strides = strides
        self.out_indices = out_indices
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.patch_norm = patch_norm
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.use_abs_pos_embed = use_abs_pos_embed
        self.use_checkpoint = use_checkpoint
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and getattr(self, key) == value
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = pretrain_img_size, pretrain_img_size
        self.pretrain_img_size = pretrain_img_size
        self.patch_embed = PatchEmbed(in_channels=in_channels, embed_dims=embed_dims, kernel_size=patch_size, stride=strides[0], padding='corner', norm_cfg=norm_cfg if patch_norm else None)
        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(torch.zeros((1, num_patches, embed_dims)))
        self.drop_after_pos = nn.Dropout(p=drop_rate)
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        self.stages = nn.ModuleList()
        in_channels = embed_dims
        num_layers = len(depths)
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(in_channels=in_channels, out_channels=2 * in_channels, stride=strides[i + 1], norm_cfg=norm_cfg if patch_norm else None)
            else:
                downsample = None
            stage = SwinBlockSequence(embed_dims=in_channels, num_heads=num_heads[i], feedforward_channels=int(mlp_ratio * in_channels), depth=depths[i], window_size=window_size, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])], downsample=downsample, act_cfg=act_cfg, norm_cfg=norm_cfg, use_checkpoint=use_checkpoint)
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels
        self.num_features = [int(embed_dims * 2 ** i) for i in range(num_layers)]
        for i in out_indices:
            layer = BuildNormalization(placeholder=self.num_features[i], norm_cfg=norm_cfg)
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)
        if pretrained:
            self.loadpretrainedweights(structure_type, pretrained_model_path)
    """loadpretrainedweights"""

    def loadpretrainedweights(self, structure_type='swin_tiny_patch4_window7_224', pretrained_model_path=''):
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        else:
            checkpoint = model_zoo.load_url(DEFAULT_MODEL_URLS[structure_type], map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        state_dict = self.swinconvert(state_dict)
        state_dict_new = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                state_dict_new[k[9:]] = v
            else:
                state_dict_new[k] = v
        state_dict = state_dict_new
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        if state_dict.get('absolute_pos_embed') is not None:
            absolute_pos_embed = state_dict['absolute_pos_embed']
            N1, L, C1 = absolute_pos_embed.size()
            N2, C2, H, W = self.absolute_pos_embed.size()
            if not (N1 != N2 or C1 != C2 or L != H * W):
                state_dict['absolute_pos_embed'] = absolute_pos_embed.view(N2, H, W, C2).permute(0, 3, 1, 2).contiguous()
        relative_position_bias_table_keys = [k for k in state_dict.keys() if 'relative_position_bias_table' in k]
        for table_key in relative_position_bias_table_keys:
            table_pretrained = state_dict[table_key]
            table_current = self.state_dict()[table_key]
            L1, nH1 = table_pretrained.size()
            L2, nH2 = table_current.size()
            if nH1 == nH2 and L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                table_pretrained_resized = F.interpolate(table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1), size=(S2, S2), mode='bicubic')
                state_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(1, 0).contiguous()
        self.load_state_dict(state_dict, strict=False)
    """swinconvert"""

    @staticmethod
    def swinconvert(ckpt):
        new_ckpt = OrderedDict()

        def correctunfoldreductionorder(x):
            out_channel, in_channel = x.shape
            x = x.reshape(out_channel, 4, in_channel // 4)
            x = x[:, [0, 2, 1, 3], :].transpose(1, 2).reshape(out_channel, in_channel)
            return x

        def correctunfoldnormorder(x):
            in_channel = x.shape[0]
            x = x.reshape(4, in_channel // 4)
            x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
            return x
        for k, v in ckpt.items():
            if k.startswith('head'):
                continue
            elif k.startswith('layers'):
                new_v = v
                if 'attn.' in k:
                    new_k = k.replace('attn.', 'attn.w_msa.')
                elif 'mlp.' in k:
                    if 'mlp.fc1.' in k:
                        new_k = k.replace('mlp.fc1.', 'ffn.layers.0.0.')
                    elif 'mlp.fc2.' in k:
                        new_k = k.replace('mlp.fc2.', 'ffn.layers.1.')
                    else:
                        new_k = k.replace('mlp.', 'ffn.')
                elif 'downsample' in k:
                    new_k = k
                    if 'reduction.' in k:
                        new_v = correctunfoldreductionorder(v)
                    elif 'norm.' in k:
                        new_v = correctunfoldnormorder(v)
                else:
                    new_k = k
                new_k = new_k.replace('layers', 'stages', 1)
            elif k.startswith('patch_embed'):
                new_v = v
                if 'proj' in k:
                    new_k = k.replace('proj', 'projection')
                else:
                    new_k = k
            else:
                new_v = v
                new_k = k
            new_ckpt[new_k] = new_v
        return new_ckpt
    """forward"""

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)
        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return outs


class TIMMBackbone(nn.Module):

    def __init__(self, structure_type, model_name, features_only=True, pretrained=True, pretrained_model_path='', in_channels=3, extra_args={}):
        super(TIMMBackbone, self).__init__()
        self.timm_model = timm.create_model(model_name=model_name, features_only=features_only, pretrained=pretrained, in_chans=in_channels, checkpoint_path=pretrained_model_path, **extra_args)
        self.timm_model.global_pool = None
        self.timm_model.fc = None
        self.timm_model.classifier = None
    """forward"""

    def forward(self, x):
        features = self.timm_model(x)
        return features


class GlobalSubsampledAttention(EfficientMultiheadAttention):

    def __init__(self, embed_dims, num_heads, attn_drop=0.0, proj_drop=0.0, dropout_cfg=None, batch_first=True, qkv_bias=True, norm_cfg=None, sr_ratio=1):
        super(GlobalSubsampledAttention, self).__init__(embed_dims, num_heads, attn_drop, proj_drop, dropout_cfg, batch_first, qkv_bias, norm_cfg, sr_ratio)


class GSAEncoderLayer(nn.Module):

    def __init__(self, embed_dims, num_heads, feedforward_channels, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, num_fcs=2, qkv_bias=True, act_cfg=None, norm_cfg=None, sr_ratio=1.0, dropout_cfg=None):
        super(GSAEncoderLayer, self).__init__()
        if dropout_cfg is None:
            dropout_cfg = {'type': 'DropPath', 'drop_prob': drop_path_rate}
        self.norm1 = BuildNormalization(placeholder=embed_dims, norm_cfg=norm_cfg)
        self.attn = GlobalSubsampledAttention(embed_dims=embed_dims, num_heads=num_heads, attn_drop=attn_drop_rate, proj_drop=drop_rate, dropout_cfg=dropout_cfg, qkv_bias=qkv_bias, norm_cfg=norm_cfg, sr_ratio=sr_ratio)
        self.norm2 = BuildNormalization(placeholder=embed_dims, norm_cfg=norm_cfg)
        self.ffn = FFN(embed_dims=embed_dims, feedforward_channels=feedforward_channels, num_fcs=num_fcs, ffn_drop=drop_rate, dropout_cfg=dropout_cfg, act_cfg=act_cfg, add_identity=False)
        self.drop_path = BuildDropout(dropout_cfg) if dropout_cfg and drop_path_rate > 0.0 else nn.Identity()
    """forward"""

    def forward(self, x, hw_shape):
        x = x + self.drop_path(self.attn(self.norm1(x), hw_shape, identity=0.0))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class LocallyGroupedSelfAttention(nn.Module):

    def __init__(self, embed_dims, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop_rate=0.0, proj_drop_rate=0.0, window_size=1):
        super(LocallyGroupedSelfAttention, self).__init__()
        assert embed_dims % num_heads == 0
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = embed_dims // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
    """forward"""

    def forward(self, x, hw_shape):
        b, n, c = x.shape
        h, w = hw_shape
        x = x.view(b, h, w, c)
        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        Hp, Wp = x.shape[1:-1]
        _h, _w = Hp // self.window_size, Wp // self.window_size
        mask = torch.zeros((1, Hp, Wp), device=x.device)
        mask[:, -pad_b:, :].fill_(1)
        mask[:, :, -pad_r:].fill_(1)
        x = x.reshape(b, _h, self.window_size, _w, self.window_size, c).transpose(2, 3)
        mask = mask.reshape(1, _h, self.window_size, _w, self.window_size).transpose(2, 3).reshape(1, _h * _w, self.window_size * self.window_size)
        attn_mask = mask.unsqueeze(2) - mask.unsqueeze(3)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-1000.0)).masked_fill(attn_mask == 0, float(0.0))
        qkv = self.qkv(x).reshape(b, _h * _w, self.window_size * self.window_size, 3, self.num_heads, c // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn + attn_mask.unsqueeze(2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(b, _h, _w, self.window_size, self.window_size, c)
        x = attn.transpose(2, 3).reshape(b, _h * self.window_size, _w * self.window_size, c)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()
        x = x.reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LSAEncoderLayer(nn.Module):

    def __init__(self, embed_dims, num_heads, feedforward_channels, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, num_fcs=2, qkv_bias=True, qk_scale=None, act_cfg=None, norm_cfg=None, window_size=1, dropout_cfg=None):
        super(LSAEncoderLayer, self).__init__()
        if dropout_cfg is None:
            dropout_cfg = {'type': 'DropPath', 'drop_prob': drop_path_rate}
        self.norm1 = BuildNormalization(placeholder=embed_dims, norm_cfg=norm_cfg)
        self.attn = LocallyGroupedSelfAttention(embed_dims, num_heads, qkv_bias, qk_scale, attn_drop_rate, drop_rate, window_size)
        self.norm2 = BuildNormalization(placeholder=embed_dims, norm_cfg=norm_cfg)
        self.ffn = FFN(embed_dims=embed_dims, feedforward_channels=feedforward_channels, num_fcs=num_fcs, ffn_drop=drop_rate, dropout_cfg=dropout_cfg, act_cfg=act_cfg, add_identity=False)
        self.drop_path = BuildDropout(dropout_cfg) if dropout_cfg and drop_path_rate > 0.0 else nn.Identity()
    """forward"""

    def forward(self, x, hw_shape):
        x = x + self.drop_path(self.attn(self.norm1(x), hw_shape))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class ConditionalPositionEncoding(nn.Module):

    def __init__(self, in_channels, embed_dims=768, stride=1):
        super(ConditionalPositionEncoding, self).__init__()
        self.stride = stride
        self.proj = nn.Conv2d(in_channels, embed_dims, kernel_size=3, stride=stride, padding=1, bias=True, groups=embed_dims)
    """forward"""

    def forward(self, x, hw_shape):
        b, n, c = x.shape
        h, w = hw_shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(b, c, h, w)
        if self.stride == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x


class PCPVT(nn.Module):

    def __init__(self, structure_type, in_channels=3, embed_dims=[64, 128, 320, 512], patch_sizes=[4, 2, 2, 2], strides=[4, 2, 2, 2], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], out_indices=(0, 1, 2, 3), qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], norm_after_stage=False, norm_cfg={'type': 'LayerNorm'}, act_cfg={'type': 'GELU'}, pretrained=True, pretrained_model_path=''):
        super(PCPVT, self).__init__()
        self.structure_type = structure_type
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.num_heads = num_heads
        self.mlp_ratios = mlp_ratios
        self.out_indices = out_indices
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.depths = depths
        self.sr_ratios = sr_ratios
        self.norm_after_stage = norm_after_stage
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and getattr(self, key) == value
        self.patch_embeds = nn.ModuleList()
        self.position_encoding_drops = nn.ModuleList()
        for i in range(len(depths)):
            self.patch_embeds.append(PatchEmbed(in_channels=in_channels if i == 0 else embed_dims[i - 1], embed_dims=embed_dims[i], kernel_size=patch_sizes[i], stride=strides[i], padding='corner', norm_cfg=norm_cfg))
            self.position_encoding_drops.append(nn.Dropout(p=drop_rate))
        self.position_encodings = nn.ModuleList([ConditionalPositionEncoding(embed_dim, embed_dim) for embed_dim in embed_dims])
        self.layers = nn.ModuleList()
        dpr, cur = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))], 0
        for k in range(len(depths)):
            _block = nn.ModuleList([GSAEncoderLayer(embed_dims=embed_dims[k], num_heads=num_heads[k], feedforward_channels=mlp_ratios[k] * embed_dims[k], attn_drop_rate=attn_drop_rate, drop_rate=drop_rate, drop_path_rate=dpr[cur + i], num_fcs=2, qkv_bias=qkv_bias, act_cfg=act_cfg, norm_cfg=norm_cfg, sr_ratio=sr_ratios[k]) for i in range(depths[k])])
            self.layers.append(_block)
            cur += depths[k]
        if self.norm_after_stage:
            self.norm_list = nn.ModuleList()
            for dim in embed_dims:
                self.norm_list.append(BuildNormalization(placeholder=dim, norm_cfg=norm_cfg))
        if pretrained:
            self.loadpretrainedweights(structure_type, pretrained_model_path)
    """forward"""

    def forward(self, x):
        outputs, b = list(), x.shape[0]
        for i in range(len(self.depths)):
            x, hw_shape = self.patch_embeds[i](x)
            h, w = hw_shape
            x = self.position_encoding_drops[i](x)
            for j, blk in enumerate(self.layers[i]):
                x = blk(x, hw_shape)
                if j == 0:
                    x = self.position_encodings[i](x, hw_shape)
            if self.norm_after_stage:
                x = self.norm_list[i](x)
            x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
            if i in self.out_indices:
                outputs.append(x)
        return tuple(outputs)
    """loadpretrainedweights"""

    def loadpretrainedweights(self, structure_type='pcpvt_small', pretrained_model_path=''):
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        else:
            checkpoint = model_zoo.load_url(DEFAULT_MODEL_URLS[structure_type], map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        state_dict = self.twinsconvert(structure_type, state_dict)
        self.load_state_dict(state_dict, strict=False)
    """twinsconvert"""

    @staticmethod
    def twinsconvert(structure_type, ckpt):
        new_ckpt = OrderedDict()
        for k, v in list(ckpt.items()):
            new_v = v
            if k.startswith('head'):
                continue
            elif k.startswith('patch_embeds'):
                if 'proj.' in k:
                    new_k = k.replace('proj.', 'projection.')
                else:
                    new_k = k
            elif k.startswith('blocks'):
                if 'attn.q.' in k:
                    new_k = k.replace('q.', 'attn.in_proj_')
                    new_v = torch.cat([v, ckpt[k.replace('attn.q.', 'attn.kv.')]], dim=0)
                elif 'mlp.fc1' in k:
                    new_k = k.replace('mlp.fc1', 'ffn.layers.0.0')
                elif 'mlp.fc2' in k:
                    new_k = k.replace('mlp.fc2', 'ffn.layers.1')
                elif structure_type.startswith('pcpvt'):
                    if 'attn.proj.' in k:
                        new_k = k.replace('proj.', 'attn.out_proj.')
                    else:
                        new_k = k
                elif 'attn.proj.' in k:
                    k_lst = k.split('.')
                    if int(k_lst[2]) % 2 == 1:
                        new_k = k.replace('proj.', 'attn.out_proj.')
                    else:
                        new_k = k
                else:
                    new_k = k
                new_k = new_k.replace('blocks.', 'layers.')
            elif k.startswith('pos_block'):
                new_k = k.replace('pos_block', 'position_encodings')
                if 'proj.0.' in new_k:
                    new_k = new_k.replace('proj.0.', 'proj.')
            else:
                new_k = k
            if 'attn.kv.' not in k:
                new_ckpt[new_k] = new_v
        return new_ckpt


class SVT(PCPVT):

    def __init__(self, structure_type, in_channels=3, embed_dims=[64, 128, 256], patch_sizes=[4, 2, 2, 2], strides=[4, 2, 2, 2], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], out_indices=(0, 1, 2, 3), qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.2, depths=[4, 4, 4], sr_ratios=[8, 4, 2, 1], windiow_sizes=[7, 7, 7], norm_after_stage=True, norm_cfg={'type': 'LayerNorm'}, act_cfg={'type': 'GELU'}, pretrained=True, pretrained_model_path=''):
        self.windiow_sizes = windiow_sizes
        super(SVT, self).__init__(in_channels=in_channels, embed_dims=embed_dims, patch_sizes=patch_sizes, strides=strides, num_heads=num_heads, mlp_ratios=mlp_ratios, out_indices=out_indices, qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, depths=depths, sr_ratios=sr_ratios, norm_after_stage=norm_after_stage, norm_cfg=norm_cfg, act_cfg=act_cfg, pretrained=False, pretrained_model_path='', structure_type=structure_type)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for k in range(len(depths)):
            for i in range(depths[k]):
                if i % 2 == 0:
                    self.layers[k][i] = LSAEncoderLayer(embed_dims=embed_dims[k], num_heads=num_heads[k], feedforward_channels=mlp_ratios[k] * embed_dims[k], drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=dpr[sum(depths[:k]) + i], num_fcs=2, qkv_bias=qkv_bias, window_size=windiow_sizes[k], norm_cfg=norm_cfg, act_cfg=act_cfg)
        if pretrained:
            self.loadpretrainedweights(structure_type, pretrained_model_path)


class BasicConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_convs=2, stride=1, dilation=1, norm_cfg=None, act_cfg=None):
        super(BasicConvBlock, self).__init__()
        convs = []
        for i in range(num_convs):
            in_c, out_c = in_channels if i == 0 else out_channels, out_channels
            s, d, p = stride if i == 0 else 1, 1 if i == 0 else dilation, 1 if i == 0 else dilation
            conv = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, stride=s, padding=p, dilation=d, bias=False), BuildNormalization(placeholder=out_c, norm_cfg=norm_cfg), BuildActivation(act_cfg))
            convs.append(conv)
        self.convs = nn.Sequential(*convs)
    """forward"""

    def forward(self, x):
        out = self.convs(x)
        return out


class DeconvModule(nn.Module):

    def __init__(self, in_channels, out_channels, norm_cfg=None, act_cfg=None, kernel_size=4, scale_factor=2):
        super(DeconvModule, self).__init__()
        assert kernel_size - scale_factor >= 0 and (kernel_size - scale_factor) % 2 == 0
        self.deconv_upsamping = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor, padding=(kernel_size - scale_factor) // 2), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
    """forward"""

    def forward(self, x):
        out = self.deconv_upsamping(x)
        return out


class InterpConv(nn.Module):

    def __init__(self, in_channels, out_channels, norm_cfg=None, act_cfg=None, conv_first=False, kernel_size=1, stride=1, padding=0, upsample_cfg=dict(scale_factor=2, mode='bilinear', align_corners=False)):
        super(InterpConv, self).__init__()
        conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
        upsample = nn.Upsample(**upsample_cfg)
        if conv_first:
            self.interp_upsample = nn.Sequential(conv, upsample)
        else:
            self.interp_upsample = nn.Sequential(upsample, conv)
    """forward"""

    def forward(self, x):
        out = self.interp_upsample(x)
        return out


class UpConvBlock(nn.Module):

    def __init__(self, conv_block, in_channels, skip_channels, out_channels, num_convs=2, stride=1, dilation=1, norm_cfg=None, act_cfg=None, upsample_type='InterpConv'):
        super(UpConvBlock, self).__init__()
        supported_upsamples = {'InterpConv': InterpConv, 'DeconvModule': DeconvModule}
        self.conv_block = conv_block(in_channels=2 * skip_channels, out_channels=out_channels, num_convs=num_convs, stride=stride, dilation=dilation, norm_cfg=norm_cfg, act_cfg=act_cfg)
        if upsample_type is not None:
            assert upsample_type in supported_upsamples, 'unsupport upsample_type %s' % upsample_type
            self.upsample = supported_upsamples[upsample_type](in_channels=in_channels, out_channels=skip_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.upsample = nn.Sequential(nn.Conv2d(in_channels, skip_channels, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=skip_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
    """forward"""

    def forward(self, skip, x):
        x = self.upsample(x)
        out = torch.cat([skip, x], dim=1)
        out = self.conv_block(out)
        return out


class UNet(nn.Module):

    def __init__(self, structure_type, in_channels=3, base_channels=64, num_stages=5, strides=(1, 1, 1, 1, 1), enc_num_convs=(2, 2, 2, 2, 2), dec_num_convs=(2, 2, 2, 2), downsamples=(True, True, True, True), enc_dilations=(1, 1, 1, 1, 1), dec_dilations=(1, 1, 1, 1), norm_cfg={'type': 'SyncBatchNorm'}, act_cfg={'type': 'ReLU', 'inplace': True}, upsample_type='InterpConv', pretrained=False, pretrained_model_path=''):
        super(UNet, self).__init__()
        self.structure_type = structure_type
        self.in_channels = in_channels
        self.enc_num_convs = enc_num_convs
        self.dec_num_convs = dec_num_convs
        self.num_stages = num_stages
        self.strides = strides
        self.downsamples = downsamples
        self.base_channels = base_channels
        self.enc_dilations = enc_dilations
        self.dec_dilations = dec_dilations
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.upsample_type = upsample_type
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        assert len(strides) == num_stages and len(enc_num_convs) == num_stages and len(dec_num_convs) == num_stages - 1 and len(downsamples) == num_stages - 1 and len(enc_dilations) == num_stages and len(dec_dilations) == num_stages - 1
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and getattr(self, key) == value
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(num_stages):
            enc_conv_block = []
            if i != 0:
                if strides[i] == 1 and downsamples[i - 1]:
                    enc_conv_block.append(nn.MaxPool2d(kernel_size=2))
                upsample = strides[i] != 1 or downsamples[i - 1]
                self.decoder.append(UpConvBlock(conv_block=BasicConvBlock, in_channels=base_channels * 2 ** i, skip_channels=base_channels * 2 ** (i - 1), out_channels=base_channels * 2 ** (i - 1), num_convs=dec_num_convs[i - 1], stride=1, dilation=dec_dilations[i - 1], norm_cfg=norm_cfg, act_cfg=act_cfg, upsample_type=upsample_type if upsample else None))
            enc_conv_block.append(BasicConvBlock(in_channels=in_channels, out_channels=base_channels * 2 ** i, num_convs=enc_num_convs[i], stride=strides[i], dilation=enc_dilations[i], norm_cfg=norm_cfg, act_cfg=act_cfg))
            self.encoder.append(nn.Sequential(*enc_conv_block))
            in_channels = base_channels * 2 ** i
        if pretrained:
            state_dict = loadpretrainedweights(structure_type=structure_type, pretrained_model_path=pretrained_model_path, default_model_urls=DEFAULT_MODEL_URLS)
            self.load_state_dict(state_dict, strict=False)
    """forward"""

    def forward(self, x):
        self.checkinputdivisible(x)
        enc_outs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outs.append(x)
        dec_outs = [x]
        for i in reversed(range(len(self.decoder))):
            x = self.decoder[i](enc_outs[i], x)
            dec_outs.append(x)
        return dec_outs
    """checkinputdivisible"""

    def checkinputdivisible(self, x):
        h, w = x.shape[-2:]
        whole_downsample_rate = 1
        for i in range(1, self.num_stages):
            if self.strides[i] == 2 or self.downsamples[i - 1]:
                whole_downsample_rate *= 2
        assert h % whole_downsample_rate == 0 and w % whole_downsample_rate == 0


class VisionTransformer(nn.Module):

    def __init__(self, structure_type, img_size=224, patch_size=16, in_channels=3, embed_dims=768, num_layers=12, num_heads=12, mlp_ratio=4, out_indices=(9, 14, 19, 23), qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, with_cls_token=True, output_cls_token=False, norm_cfg={'type': 'LayerNorm', 'eps': 1e-06}, act_cfg={'type': 'GELU'}, patch_norm=False, final_norm=False, interpolate_mode='bilinear', num_fcs=2, use_checkpoint=False, pretrained=True, pretrained_model_path=''):
        super(VisionTransformer, self).__init__()
        self.structure_type = structure_type
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.out_indices = out_indices
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.patch_norm = patch_norm
        self.num_fcs = num_fcs
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        self.interpolate_mode = interpolate_mode
        self.use_checkpoint = use_checkpoint
        self.final_norm = final_norm
        if output_cls_token:
            assert with_cls_token, 'with_cls_token must be True if set output_cls_token to True.'
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and getattr(self, key) == value
        if isinstance(img_size, int):
            img_size = img_size, img_size
        self.img_size = img_size
        self.patch_embed = PatchEmbed(in_channels=in_channels, embed_dims=embed_dims, kernel_size=patch_size, stride=patch_size, padding='corner', norm_cfg=norm_cfg if patch_norm else None)
        num_patches = img_size[0] // patch_size * (img_size[1] // patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dims))
        self.drop_after_pos = nn.Dropout(p=drop_rate)
        if isinstance(out_indices, int):
            if out_indices == -1:
                out_indices = num_layers - 1
            self.out_indices = [out_indices]
        elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
            self.out_indices = out_indices
        else:
            raise TypeError('out_indices must be type of int, list or tuple')
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(TransformerEncoderLayer(embed_dims=embed_dims, num_heads=num_heads, feedforward_channels=mlp_ratio * embed_dims, attn_drop_rate=attn_drop_rate, drop_rate=drop_rate, drop_path_rate=dpr[i], num_fcs=num_fcs, qkv_bias=qkv_bias, act_cfg=act_cfg, norm_cfg=norm_cfg, batch_first=True, use_checkpoint=use_checkpoint))
        if final_norm:
            self.ln1 = BuildNormalization(placeholder=embed_dims, norm_cfg=norm_cfg)
        if pretrained:
            self.loadpretrainedweights(structure_type, pretrained_model_path)
    """loadpretrainedweights"""

    def loadpretrainedweights(self, structure_type='jx_vit_large_p16_384', pretrained_model_path=''):
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        else:
            checkpoint = model_zoo.load_url(DEFAULT_MODEL_URLS[structure_type], map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        state_dict = self.vitconvert(state_dict)
        if 'pos_embed' in state_dict.keys():
            if self.pos_embed.shape != state_dict['pos_embed'].shape:
                h, w = self.img_size
                pos_size = int(math.sqrt(state_dict['pos_embed'].shape[1] - 1))
                state_dict['pos_embed'] = self.resizeposembed(state_dict['pos_embed'], (h // self.patch_size, w // self.patch_size), (pos_size, pos_size), self.interpolate_mode)
        self.load_state_dict(state_dict, strict=False)
    """vitconvert"""

    @staticmethod
    def vitconvert(ckpt):
        from collections import OrderedDict
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            if k.startswith('head'):
                continue
            if k.startswith('norm'):
                new_k = k.replace('norm.', 'ln1.')
            elif k.startswith('patch_embed'):
                if 'proj' in k:
                    new_k = k.replace('proj', 'projection')
                else:
                    new_k = k
            elif k.startswith('blocks'):
                if 'norm' in k:
                    new_k = k.replace('norm', 'ln')
                elif 'mlp.fc1' in k:
                    new_k = k.replace('mlp.fc1', 'ffn.layers.0.0')
                elif 'mlp.fc2' in k:
                    new_k = k.replace('mlp.fc2', 'ffn.layers.1')
                elif 'attn.qkv' in k:
                    new_k = k.replace('attn.qkv.', 'attn.attn.in_proj_')
                elif 'attn.proj' in k:
                    new_k = k.replace('attn.proj', 'attn.attn.out_proj')
                else:
                    new_k = k
                new_k = new_k.replace('blocks.', 'layers.')
            else:
                new_k = k
            new_ckpt[new_k] = v
        return new_ckpt
    """posembeding"""

    def posembeding(self, patched_img, hw_shape, pos_embed):
        assert patched_img.ndim == 3 and pos_embed.ndim == 3, 'the shapes of patched_img and pos_embed must be [B, L, C]'
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            if pos_len == self.img_size[0] // self.patch_size * (self.img_size[1] // self.patch_size) + 1:
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError('Unexpected shape of pos_embed, got {}.'.format(pos_embed.shape))
            pos_embed = self.resizeposembed(pos_embed, hw_shape, (pos_h, pos_w), self.interpolate_mode)
        return self.drop_after_pos(patched_img + pos_embed)
    """resizeposembed"""

    @staticmethod
    def resizeposembed(pos_embed, input_shpae, pos_shape, mode):
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0:1]
        pos_embed_weight = pos_embed[:, -1 * pos_h * pos_w:]
        pos_embed_weight = pos_embed_weight.reshape(1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = F.interpolate(pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed
    """forward"""

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        x, hw_shape = self.patch_embed(inputs)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.posembeding(x, hw_shape, self.pos_embed)
        if not self.with_cls_token:
            x = x[:, 1:]
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1:
                if self.final_norm:
                    x = self.ln1(x)
            if i in self.out_indices:
                if self.with_cls_token:
                    out = x[:, 1:]
                else:
                    out = x
                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1], C).permute(0, 3, 1, 2).contiguous()
                if self.output_cls_token:
                    out = [out, x[:, 0]]
                outs.append(out)
        return tuple(outs)


def bchw2nc(tensor: 'torch.Tensor'):
    tensor = tensor.transpose(0, 1)
    tensor = tensor.reshape(tensor.size(0), -1)
    tensor = tensor.transpose(0, 1).contiguous()
    return tensor


def calculateaccuracy(x_src, x_tgt, topk=1, thresh=None, ignore_index=None):
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk, return_single = (topk,), True
    else:
        return_single = False
    maxk = max(topk)
    if x_src.size(0) == 0:
        acc = [x_tgt.new_tensor(0.0) for _ in range(len(topk))]
        return acc[0] if return_single else acc
    assert x_src.ndim == x_tgt.ndim + 1
    assert x_src.size(0) == x_tgt.size(0)
    assert maxk <= x_src.size(1), f'maxk {maxk} exceeds x_src dimension {x_src.size(1)}'
    pred_value, pred_label = x_src.topk(maxk, dim=1)
    pred_label = pred_label.transpose(0, 1)
    correct = pred_label.eq(x_tgt.unsqueeze(0).expand_as(pred_label))
    if thresh is not None:
        correct = correct & (pred_value > thresh).t()
    if ignore_index is not None:
        correct = correct[:, x_tgt != ignore_index]
    res = []
    eps = torch.finfo(torch.float32).eps
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True) + eps
        if ignore_index is not None:
            total_num = x_tgt[x_tgt != ignore_index].numel() + eps
        else:
            total_num = x_tgt.numel() + eps
        res.append(correct_k.mul_(100.0 / total_num))
    return res[0] if return_single else res


class Accuracy(nn.Module):

    def __init__(self, topk=(1,), thresh=None, ignore_index=None):
        super(Accuracy, self).__init__()
        self.topk = topk
        self.thresh = thresh
        self.ignore_index = ignore_index
    """forward"""

    def forward(self, x_src, x_tgt):
        x_src = bchw2nc(x_src)
        x_tgt = x_tgt.view(-1)
        return calculateaccuracy(x_src, x_tgt, self.topk, self.thresh, self.ignore_index)


def reduceloss(loss, reduction='mean', avg_factor=None):
    assert reduction in ['mean', 'sum', 'none']
    if reduction == 'mean':
        if avg_factor is None:
            return torch.mean(loss)
        return torch.sum(loss) / avg_factor
    elif reduction == 'sum':
        return torch.sum(loss)
    else:
        return loss


def reducelosswithweight(loss, weight=None, reduction='mean', avg_factor=None):
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight.float()
    return reduceloss(loss=loss, reduction=reduction, avg_factor=avg_factor)


class CosineSimilarityLoss(nn.Module):

    def __init__(self, scale_factor=1.0, reduction='mean', lowest_loss_value=None):
        super(CosineSimilarityLoss, self).__init__()
        assert reduction in ['mean', 'sum', 'none']
        self.reduction = reduction
        self.scale_factor = scale_factor
        self.lowest_loss_value = lowest_loss_value
    """forward"""

    def forward(self, x_src, x_tgt, weight=None):
        assert x_src.shape == x_tgt.shape, 'invalid shape of x_src or x_tgt'
        loss = 1 - F.cosine_similarity(x_src, x_tgt, dim=1)
        loss = reducelosswithweight(loss, weight, self.reduction, None)
        loss = loss * self.scale_factor
        if self.lowest_loss_value is not None:
            loss = torch.abs(loss - self.lowest_loss_value) + self.lowest_loss_value
        return loss


def binarycrossentropy(x_src, x_tgt, weight=None, class_weight=None, reduction='mean', ignore_index=-100):
    original_shape, num_classes = x_src.shape, x_src.size(1)
    if x_src.shape == x_tgt.shape:
        x_src = bchw2nc(x_src)
        x_src = x_src[:, torch.arange(num_classes) != ignore_index]
        x_tgt = bchw2nc(x_tgt).float()
        x_tgt = x_tgt[:, torch.arange(num_classes) != ignore_index]
        ignore_index_mask = None
    else:
        x_src = bchw2nc(x_src)
        x_tgt = x_tgt.view(-1).contiguous().long()
        ignore_index_mask = x_tgt != ignore_index
        x_src = x_src[ignore_index_mask]
        x_tgt = x_tgt[ignore_index_mask]
        x_tgt = F.one_hot(x_tgt, num_classes=num_classes + 1).float()
        if num_classes == 1:
            x_tgt = x_tgt[:, 1].unsqueeze(-1)
        else:
            x_tgt = x_tgt[:, :num_classes]
    if weight is not None:
        weight = bchw2nc(weight).float()
        if ignore_index_mask is not None:
            weight = weight[ignore_index_mask]
    loss = F.binary_cross_entropy_with_logits(x_src, x_tgt, reduction='none')
    if class_weight is not None:
        class_weight = x_src.new_tensor(class_weight)
        if ignore_index_mask is None:
            class_weight = class_weight[torch.arange(num_classes) != ignore_index]
        class_weight = class_weight.reshape(1, -1)
        loss = loss * class_weight
        avg_factor = (class_weight * x_tgt).sum()
    else:
        avg_factor = None
    loss = reducelosswithweight(loss, weight, reduction, avg_factor)
    if reduction == 'none':
        loss = loss.transpose(0, 1)
        loss = loss.reshape(original_shape[1], original_shape[0], *original_shape[2:])
        loss = loss.transpose(0, 1).contiguous()
    return loss


def crossentropy(x_src, x_tgt, weight=None, class_weight=None, reduction='mean', ignore_index=-100, label_smoothing=None):
    original_shape, num_classes = x_src.shape, x_src.size(1)
    assert num_classes > 1, 'num classes of inputs should be larger than 1'
    if x_src.shape == x_tgt.shape:
        x_src = bchw2nc(x_src)
        x_src = x_src[:, torch.arange(num_classes) != ignore_index]
        x_tgt = bchw2nc(x_tgt).float()
        x_tgt = x_tgt[:, torch.arange(num_classes) != ignore_index]
        ignore_index = -100
        ignore_index_mask = None
    else:
        x_src = bchw2nc(x_src)
        x_tgt = x_tgt.view(-1).contiguous().long()
        ignore_index_mask = x_tgt != ignore_index
        x_src = x_src[ignore_index_mask]
        x_tgt = x_tgt[ignore_index_mask]
        x_tgt = F.one_hot(x_tgt, num_classes=num_classes + 1).float()
        x_tgt = x_tgt[:, :num_classes]
    if weight is not None:
        weight = weight.view(-1).contiguous()
        if ignore_index_mask is not None:
            weight = weight[ignore_index_mask]
    if label_smoothing is None:
        loss = F.cross_entropy(x_src, x_tgt.argmax(dim=1), reduction='none')
    else:
        loss = F.cross_entropy(x_src, x_tgt.argmax(dim=1), reduction='none', label_smoothing=label_smoothing)
    loss = loss.view(-1, 1).expand_as(x_tgt)
    loss = loss * x_tgt
    if class_weight is not None:
        class_weight = x_src.new_tensor(class_weight)
        if ignore_index_mask is None:
            class_weight = class_weight[torch.arange(num_classes) != ignore_index]
        class_weight = class_weight.reshape(1, -1)
        loss = loss * class_weight
        avg_factor = (class_weight * x_tgt).sum()
    else:
        avg_factor = None
    loss = reducelosswithweight(loss.sum(dim=1), weight, reduction, avg_factor)
    if reduction == 'none':
        loss = loss.transpose(0, 1)
        loss = loss.reshape(original_shape[1], original_shape[0], *original_shape[2:])
        loss = loss.transpose(0, 1).contiguous()
    return loss


class CrossEntropyLoss(nn.Module):

    def __init__(self, use_sigmoid=False, reduction='mean', class_weight=None, scale_factor=1.0, lowest_loss_value=None, label_smoothing=None, ignore_index=-100):
        super(CrossEntropyLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.class_weight = class_weight
        self.scale_factor = scale_factor
        self.lowest_loss_value = lowest_loss_value
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
    """forward"""

    def forward(self, x_src, x_tgt, weight=None):
        assert x_src.shape == x_tgt.shape or x_src.size(0) == x_tgt.size(0) and x_src.shape[2:] == x_tgt.shape[1:], 'invalid shape of x_src or x_tgt'
        if self.use_sigmoid:
            loss = binarycrossentropy(x_src, x_tgt, weight=weight, class_weight=self.class_weight, reduction=self.reduction, ignore_index=self.ignore_index)
        else:
            loss = crossentropy(x_src, x_tgt, weight=weight, class_weight=self.class_weight, reduction=self.reduction, ignore_index=self.ignore_index, label_smoothing=self.label_smoothing)
        loss = loss * self.scale_factor
        if self.lowest_loss_value is not None:
            loss = torch.abs(loss - self.lowest_loss_value) + self.lowest_loss_value
        return loss


def diceloss(x_src, x_tgt, weight=None, eps=0.001, reduction='mean', naive_dice=False, ignore_index=-100):
    num_classes = x_src.shape[1]
    if x_src.shape == x_tgt.shape:
        x_src = x_src[:, torch.arange(num_classes) != ignore_index, ...]
        x_tgt = x_tgt[:, torch.arange(num_classes) != ignore_index, ...]
        ignore_index_mask = None
    else:
        ignore_index_mask = (x_tgt != ignore_index).unsqueeze(1).float()
        x_tgt = torch.clamp(x_tgt, min=0, max=num_classes)
        if x_tgt.dim() == 1:
            x_tgt = F.one_hot(x_tgt, num_classes + 1).float()
            if num_classes == 1:
                x_tgt = x_tgt[..., 1].unsqueeze(-1)
            else:
                x_tgt = x_tgt[..., :num_classes]
        else:
            x_tgt = F.one_hot(x_tgt, num_classes + 1).float()
            if num_classes == 1:
                x_tgt = x_tgt[..., 1].unsqueeze(-1).permute(0, -1, *range(1, x_tgt.dim() - 1)).contiguous()
            else:
                x_tgt = x_tgt[..., :num_classes].permute(0, -1, *range(1, x_tgt.dim() - 1)).contiguous()
    if ignore_index_mask is not None:
        x_src = x_src * ignore_index_mask
        x_tgt = x_tgt * ignore_index_mask
    x_src = x_src.flatten(1)
    x_tgt = x_tgt.flatten(1)
    a = torch.sum(x_src * x_tgt, 1)
    if naive_dice:
        b = torch.sum(x_src, 1)
        c = torch.sum(x_tgt, 1)
        d = (2 * a + eps) / (b + c + eps)
    else:
        b = torch.sum(x_src * x_src, 1) + eps
        c = torch.sum(x_tgt * x_tgt, 1) + eps
        d = 2 * a / (b + c)
    loss = 1 - d
    loss = reducelosswithweight(loss, weight, reduction, None)
    return loss


class DiceLoss(nn.Module):

    def __init__(self, use_sigmoid=True, activate=True, reduction='mean', naive_dice=False, eps=0.001, scale_factor=1.0, ignore_index=-100, lowest_loss_value=None):
        super(DiceLoss, self).__init__()
        assert reduction in ['mean', 'sum', 'none']
        self.use_sigmoid = use_sigmoid
        self.activate = activate
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.eps = eps
        self.scale_factor = scale_factor
        self.ignore_index = ignore_index
        self.lowest_loss_value = lowest_loss_value
    """forward"""

    def forward(self, x_src, x_tgt, weight=None):
        assert x_src.shape == x_tgt.shape or x_src.size(0) == x_tgt.size(0) and x_src.shape[2:] == x_tgt.shape[1:], 'invalid shape of x_src or x_tgt'
        if self.activate:
            if self.use_sigmoid:
                x_src = x_src.sigmoid()
            elif x_src.shape[1] > 1:
                x_src = x_src.softmax(dim=1)
        loss = diceloss(x_src, x_tgt, weight=weight, eps=self.eps, reduction=self.reduction, naive_dice=self.naive_dice, ignore_index=self.ignore_index)
        loss = loss * self.scale_factor
        if self.lowest_loss_value is not None:
            loss = torch.abs(loss - self.lowest_loss_value) + self.lowest_loss_value
        return loss


def pysigmoidfocalloss(x_src, x_tgt, weight=None, gamma=2.0, alpha=0.25, class_weight=None, ignore_index=-100, reduction='mean'):
    original_shape, num_classes = x_src.shape, x_src.size(1)
    if x_src.shape == x_tgt.shape:
        x_src = bchw2nc(x_src)
        x_src = x_src[:, torch.arange(num_classes) != ignore_index]
        x_tgt = bchw2nc(x_tgt).float()
        x_tgt = x_tgt[:, torch.arange(num_classes) != ignore_index]
        ignore_index_mask = None
    else:
        x_src = bchw2nc(x_src)
        x_tgt = x_tgt.view(-1).contiguous().long()
        ignore_index_mask = x_tgt != ignore_index
        x_src = x_src[ignore_index_mask]
        x_tgt = x_tgt[ignore_index_mask]
        x_tgt = F.one_hot(x_tgt, num_classes=num_classes + 1).float()
        if num_classes == 1:
            x_tgt = x_tgt[:, 1].unsqueeze(-1)
        else:
            x_tgt = x_tgt[:, :num_classes]
    if weight is not None:
        weight = bchw2nc(weight).float()
        if ignore_index_mask is not None:
            weight = weight[ignore_index_mask]
    x_src_sigmoid, x_tgt = x_src.sigmoid(), x_tgt.type_as(x_src)
    one_minus_pt = (1 - x_src_sigmoid) * x_tgt + x_src_sigmoid * (1 - x_tgt)
    focal_weight = (alpha * x_tgt + (1 - alpha) * (1 - x_tgt)) * one_minus_pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(x_src, x_tgt, reduction='none') * focal_weight
    if class_weight is not None:
        class_weight = x_src.new_tensor(class_weight)
        if ignore_index_mask is None:
            class_weight = class_weight[torch.arange(num_classes) != ignore_index]
        class_weight = class_weight.reshape(1, -1)
        loss = loss * class_weight
        avg_factor = (class_weight * x_tgt).sum()
    else:
        avg_factor = None
    loss = reducelosswithweight(loss, weight, reduction, avg_factor)
    if reduction == 'none':
        loss = loss.transpose(0, 1)
        loss = loss.reshape(original_shape[1], original_shape[0], *original_shape[2:])
        loss = loss.transpose(0, 1).contiguous()
    return loss


def pysoftmaxfocalloss(x_src, x_tgt, weight=None, gamma=2.0, alpha=0.25, class_weight=None, ignore_index=-100, reduction='mean'):
    original_shape, num_classes = x_src.shape, x_src.size(1)
    assert num_classes > 1, 'num classes of inputs should be larger than 1'
    if x_src.shape == x_tgt.shape:
        x_src = bchw2nc(x_src)
        x_src = x_src[:, torch.arange(num_classes) != ignore_index]
        x_tgt = bchw2nc(x_tgt).float()
        x_tgt = x_tgt[:, torch.arange(num_classes) != ignore_index]
        ignore_index_mask = None
    else:
        x_src = bchw2nc(x_src)
        x_tgt = x_tgt.view(-1).contiguous().long()
        ignore_index_mask = x_tgt != ignore_index
        x_src = x_src[ignore_index_mask]
        x_tgt = x_tgt[ignore_index_mask]
        x_tgt = F.one_hot(x_tgt, num_classes=num_classes + 1).float()
        x_tgt = x_tgt[:, :num_classes]
    if weight is not None:
        weight = weight.view(-1).contiguous().float()
        if ignore_index_mask is not None:
            weight = weight[ignore_index_mask]
    loss = F.cross_entropy(x_src, x_tgt.argmax(dim=1), reduction='none')
    loss = loss.view(-1, 1).expand_as(x_tgt)
    probs = F.softmax(x_src, dim=1)
    pt = probs * x_tgt + (1 - probs) * (1 - x_tgt)
    loss = alpha * (1 - pt) ** gamma * loss * x_tgt
    if class_weight is not None:
        class_weight = x_src.new_tensor(class_weight)
        if ignore_index_mask is None:
            class_weight = class_weight[torch.arange(num_classes) != ignore_index]
        class_weight = class_weight.reshape(1, -1)
        loss = loss * class_weight
        avg_factor = (class_weight * x_tgt).sum()
    else:
        avg_factor = None
    loss = reducelosswithweight(loss.sum(dim=1), weight, reduction, avg_factor)
    if reduction == 'none':
        loss = loss.transpose(0, 1)
        loss = loss.reshape(original_shape[1], original_shape[0], *original_shape[2:])
        loss = loss.transpose(0, 1).contiguous()
    return loss


def sigmoidfocalloss(x_src, x_tgt, weight=None, gamma=2.0, alpha=0.25, class_weight=None, ignore_index=-100, reduction='mean'):
    if x_src.shape == x_tgt.shape or x_src.shape[1] == 1:
        return pysigmoidfocalloss(x_src, x_tgt, weight=weight, gamma=gamma, alpha=alpha, class_weight=class_weight, ignore_index=ignore_index, reduction=reduction)
    original_shape, num_classes = x_src.shape, x_src.size(1)
    x_src = bchw2nc(x_src)
    x_tgt = x_tgt.view(-1).contiguous().long()
    ignore_index_mask = x_tgt != ignore_index
    x_src = x_src[ignore_index_mask]
    x_tgt = x_tgt[ignore_index_mask]
    if weight is not None:
        weight = bchw2nc(weight).float()
        if ignore_index_mask is not None:
            weight = weight[ignore_index_mask]
    loss = _sigmoid_focal_loss(x_src, x_tgt, gamma, alpha, None, 'none')
    if class_weight is not None:
        class_weight = x_src.new_tensor(class_weight)
        if ignore_index_mask is None:
            class_weight = class_weight[torch.arange(num_classes) != ignore_index]
        class_weight = class_weight.reshape(1, -1)
        loss = loss * class_weight
        x_tgt = F.one_hot(x_tgt, num_classes=num_classes + 1).float()
        x_tgt = x_tgt[:, :num_classes]
        avg_factor = (class_weight * x_tgt).sum()
    else:
        avg_factor = None
    loss = reducelosswithweight(loss, weight, reduction, avg_factor)
    if reduction == 'none':
        loss = loss.transpose(0, 1)
        loss = loss.reshape(original_shape[1], original_shape[0], *original_shape[2:])
        loss = loss.transpose(0, 1).contiguous()
    return loss


def softmaxfocalloss(x_src, x_tgt, weight=None, gamma=2.0, alpha=0.25, class_weight=None, ignore_index=-100, reduction='mean'):
    if x_src.shape == x_tgt.shape:
        return pysoftmaxfocalloss(x_src, x_tgt, weight=weight, gamma=gamma, alpha=alpha, class_weight=class_weight, ignore_index=ignore_index, reduction=reduction)
    original_shape, num_classes = x_src.shape, x_src.size(1)
    assert num_classes > 1, 'num classes of inputs should be larger than 1'
    x_src = bchw2nc(x_src)
    x_tgt = x_tgt.view(-1).contiguous().long()
    ignore_index_mask = x_tgt != ignore_index
    x_src = x_src[ignore_index_mask]
    x_tgt = x_tgt[ignore_index_mask]
    if weight is not None:
        weight = weight.view(-1).contiguous().float()
        if ignore_index_mask is not None:
            weight = weight[ignore_index_mask]
    if class_weight is not None:
        class_weight = x_src.new_tensor(class_weight)
        if ignore_index_mask is None:
            class_weight = class_weight[torch.arange(num_classes) != ignore_index]
        avg_factor = class_weight[x_tgt].sum()
    else:
        avg_factor = None
    loss = _softmax_focal_loss(x_src, x_tgt, gamma, alpha, class_weight, 'none')
    loss = reducelosswithweight(loss, weight, reduction, avg_factor)
    if reduction == 'none':
        loss = loss.transpose(0, 1)
        loss = loss.reshape(original_shape[1], original_shape[0], *original_shape[2:])
        loss = loss.transpose(0, 1).contiguous()
    return loss


class FocalLoss(nn.Module):

    def __init__(self, use_sigmoid=True, scale_factor=1.0, gamma=2, alpha=0.25, class_weight=None, reduction='mean', ignore_index=-100, lowest_loss_value=None):
        super(FocalLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.gamma = gamma
        self.class_weight = class_weight
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.scale_factor = scale_factor
        self.lowest_loss_value = lowest_loss_value
    """forward"""

    def forward(self, x_src, x_tgt, weight=None):
        assert x_src.shape == x_tgt.shape or x_src.size(0) == x_tgt.size(0) and x_src.shape[2:] == x_tgt.shape[1:], 'invalid shape of x_src or x_tgt'
        if self.use_sigmoid:
            if torch.cuda.is_available() and x_src.is_cuda and _sigmoid_focal_loss is not None:
                loss = sigmoidfocalloss(x_src, x_tgt, weight=weight, gamma=self.gamma, alpha=self.alpha, class_weight=self.class_weight, ignore_index=self.ignore_index, reduction=self.reduction)
            else:
                loss = pysigmoidfocalloss(x_src, x_tgt, weight=weight, gamma=self.gamma, alpha=self.alpha, class_weight=self.class_weight, ignore_index=self.ignore_index, reduction=self.reduction)
        elif torch.cuda.is_available() and x_src.is_cuda and _softmax_focal_loss is not None:
            loss = softmaxfocalloss(x_src, x_tgt, weight=weight, gamma=self.gamma, alpha=self.alpha, class_weight=self.class_weight, ignore_index=self.ignore_index, reduction=self.reduction)
        else:
            loss = pysoftmaxfocalloss(x_src, x_tgt, weight=weight, gamma=self.gamma, alpha=self.alpha, class_weight=self.class_weight, ignore_index=self.ignore_index, reduction=self.reduction)
        loss = loss * self.scale_factor
        if self.lowest_loss_value is not None:
            loss = torch.abs(loss - self.lowest_loss_value) + self.lowest_loss_value
        return loss


class KLDivLoss(nn.Module):

    def __init__(self, scale_factor=1.0, temperature=1, reduction='mean', lowest_loss_value=None):
        super(KLDivLoss, self).__init__()
        assert reduction in ['batchmean', 'mean', 'sum', 'none']
        self.reduction = reduction
        self.temperature = temperature
        self.scale_factor = scale_factor
        self.lowest_loss_value = lowest_loss_value
    """forward"""

    def forward(self, x_src, x_tgt, weight=None):
        assert x_src.shape == x_tgt.shape, 'invalid shape of x_src or x_tgt'
        x_src = F.log_softmax(x_src / self.temperature, dim=1)
        x_tgt = F.softmax(x_tgt / self.temperature, dim=1)
        loss = F.kl_div(x_src, x_tgt, reduction='none', log_target=False)
        loss = loss * self.temperature ** 2
        if weight is not None:
            loss = loss * weight
        batch_size = x_src.shape[0]
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'batchmean':
            loss = loss.sum() / batch_size
        elif self.reduction == 'sum':
            loss = loss.sum()
        loss = loss * self.scale_factor
        if self.lowest_loss_value is not None:
            loss = torch.abs(loss - self.lowest_loss_value) + self.lowest_loss_value
        return loss


class L1Loss(nn.Module):

    def __init__(self, scale_factor=1.0, reduction='mean', lowest_loss_value=None):
        super(L1Loss, self).__init__()
        self.reduction = reduction
        self.scale_factor = scale_factor
        self.lowest_loss_value = lowest_loss_value
    """forward"""

    def forward(self, x_src, x_tgt, weight=None):
        assert x_src.shape == x_tgt.shape, 'invalid shape of x_src or x_tgt'
        loss = F.l1_loss(x_src, x_tgt, reduction='none')
        loss = reducelosswithweight(loss, weight, self.reduction, None)
        loss = loss * self.scale_factor
        if self.lowest_loss_value is not None:
            loss = torch.abs(loss - self.lowest_loss_value) + self.lowest_loss_value
        return loss


def lovaszgrad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _sigmoidlovaszloss(x_src, x_tgt, classes='present', class_weight=None, ignore_index=-100):
    num_classes = x_src.shape[1]
    if x_src.shape == x_tgt.shape:
        x_src = bchw2nc(x_src)
        x_tgt = bchw2nc(x_tgt).float()
        ignore_index_mask = None
    else:
        x_src = bchw2nc(x_src)
        x_tgt = x_tgt.view(-1).contiguous().long()
        ignore_index_mask = x_tgt != ignore_index
        x_src = x_src[ignore_index_mask]
        x_tgt = x_tgt[ignore_index_mask]
        x_tgt = F.one_hot(x_tgt, num_classes=num_classes + 1).float()
        if num_classes == 1:
            x_tgt = x_tgt[:, 1].unsqueeze(-1)
        else:
            x_tgt = x_tgt[:, :num_classes]
    if classes == 'present':
        classes = []
        for cls_id in list(range(x_src.shape[1])):
            if cls_id == ignore_index and ignore_index_mask is None:
                continue
            fg = x_tgt[:, cls_id]
            if fg.sum() == 0:
                continue
            classes.append(cls_id)
    elif classes == 'all':
        classes = torch.arange(x_src.shape[1])
    classes = list(set(classes))
    classes = [x for x in classes if x != ignore_index]
    x_src, x_tgt = x_src[:, classes], x_tgt[:, classes]
    if class_weight is not None:
        class_weight = x_src.new_tensor(class_weight)
        class_weight = class_weight[classes]
        class_weight = class_weight.reshape(1, -1)
    signs = 2.0 * x_tgt.float() - 1.0
    errors = 1.0 - x_src * signs
    if class_weight is not None:
        errors = errors * class_weight
    errors, x_tgt = errors.view(-1), x_tgt.view(-1)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = x_tgt[perm.data]
    grad = lovaszgrad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def sigmoidlovaszloss(x_src, x_tgt, weight=None, classes='present', per_image=False, class_weight=None, reduction='mean', ignore_index=-100):
    x_src = x_src.sigmoid()
    if per_image:
        loss = [_sigmoidlovaszloss(x_src[idx].unsqueeze(0), x_tgt[idx].unsqueeze(0), classes=classes, class_weight=class_weight, ignore_index=ignore_index) for idx in range(x_src.shape[0])]
        loss = torch.stack(loss).reshape(-1)
    else:
        loss = _sigmoidlovaszloss(x_src, x_tgt, classes=classes, class_weight=class_weight, ignore_index=ignore_index)
    loss = reducelosswithweight(loss, weight, reduction, None)
    return loss


def _softmaxlovaszloss(x_src, x_tgt, classes='present', class_weight=None, ignore_index=-100):
    num_classes = x_src.shape[1]
    if x_src.shape == x_tgt.shape:
        x_src = bchw2nc(x_src)
        x_tgt = bchw2nc(x_tgt).float()
        ignore_index_mask = None
    else:
        x_src = bchw2nc(x_src)
        x_tgt = x_tgt.view(-1).contiguous().long()
        ignore_index_mask = x_tgt != ignore_index
        x_src = x_src[ignore_index_mask]
        x_tgt = x_tgt[ignore_index_mask]
        x_tgt = F.one_hot(x_tgt, num_classes=num_classes + 1).float()
        if num_classes == 1:
            x_tgt = x_tgt[:, 1].unsqueeze(-1)
        else:
            x_tgt = x_tgt[:, :num_classes]
    loss, loss_weight = [], []
    classes_to_sum = list(range(num_classes)) if classes in ['all', 'present'] else list(set(classes))
    for cls_id in classes_to_sum:
        if cls_id == ignore_index and ignore_index_mask is None:
            continue
        fg = x_tgt[:, cls_id]
        if classes == 'present' and fg.sum() == 0:
            continue
        class_prob = x_src[:, cls_id]
        errors = (fg - class_prob).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm.data]
        loss.append(torch.dot(errors_sorted, lovaszgrad(fg_sorted)))
        if class_weight is not None:
            loss_weight.append(class_weight[cls_id])
    loss = torch.stack(loss).reshape(-1)
    if class_weight is not None:
        loss_weight = loss.new_tensor(loss_weight).reshape(-1)
        loss = (loss * loss_weight).sum() / loss_weight.sum()
    else:
        loss = loss.mean()
    return loss


def softmaxlovaszloss(x_src, x_tgt, weight=None, classes='present', per_image=False, class_weight=None, reduction='mean', ignore_index=-100):
    x_src = F.softmax(x_src, dim=1) if x_src.shape[1] > 1 else x_src
    if per_image:
        loss = [_softmaxlovaszloss(x_src[idx].unsqueeze(0), x_tgt[idx].unsqueeze(0), classes=classes, class_weight=class_weight, ignore_index=ignore_index) for idx in range(x_src.shape[0])]
        loss = torch.stack(loss).reshape(-1)
    else:
        loss = _softmaxlovaszloss(x_src, x_tgt, classes=classes, class_weight=class_weight, ignore_index=ignore_index)
    loss = reducelosswithweight(loss, weight, reduction, None)
    return loss


class LovaszLoss(nn.Module):

    def __init__(self, use_sigmoid=False, classes='present', per_image=False, reduction='mean', class_weight=None, scale_factor=1.0, ignore_index=-100, lowest_loss_value=None):
        super(LovaszLoss, self).__init__()
        assert classes in ('all', 'present') or isinstance(classes, (list, tuple)) and all(isinstance(elem, int) for elem in classes)
        self.use_sigmoid = use_sigmoid
        self.classes = classes
        self.per_image = per_image
        self.reduction = reduction
        self.class_weight = class_weight
        self.scale_factor = scale_factor
        self.ignore_index = ignore_index
        self.lowest_loss_value = lowest_loss_value
    """forward"""

    def forward(self, x_src, x_tgt, weight=None):
        assert x_src.shape == x_tgt.shape or x_src.size(0) == x_tgt.size(0) and x_src.shape[2:] == x_tgt.shape[1:], 'invalid shape of x_src or x_tgt'
        if self.use_sigmoid:
            loss = sigmoidlovaszloss(x_src, x_tgt, weight=weight, classes=self.classes, per_image=self.per_image, class_weight=self.class_weight, reduction=self.reduction, ignore_index=self.ignore_index)
        else:
            loss = softmaxlovaszloss(x_src, x_tgt, weight=weight, classes=self.classes, per_image=self.per_image, class_weight=self.class_weight, reduction=self.reduction, ignore_index=self.ignore_index)
        loss = loss * self.scale_factor
        if self.lowest_loss_value is not None:
            loss = torch.abs(loss - self.lowest_loss_value) + self.lowest_loss_value
        return loss


class MSELoss(nn.Module):

    def __init__(self, scale_factor=1.0, reduction='mean', lowest_loss_value=None):
        super(MSELoss, self).__init__()
        self.reduction = reduction
        self.scale_factor = scale_factor
        self.lowest_loss_value = lowest_loss_value
    """forward"""

    def forward(self, x_src, x_tgt, weight=None):
        assert x_src.shape == x_tgt.shape, 'invalid shape of x_src or x_tgt'
        loss = F.mse_loss(x_src, x_tgt, reduction='none')
        loss = reducelosswithweight(loss, weight, self.reduction, None)
        loss = loss * self.scale_factor
        if self.lowest_loss_value is not None:
            loss = torch.abs(loss - self.lowest_loss_value) + self.lowest_loss_value
        return loss


class PPMConcat(nn.Module):

    def __init__(self, pool_scales):
        super(PPMConcat, self).__init__()
        self.pool_nets = nn.ModuleList()
        for pool_scale in pool_scales:
            self.pool_nets.append(nn.AdaptiveAvgPool2d(pool_scale))
    """forward"""

    def forward(self, x):
        ppm_outs = []
        for pool_net in self.pool_nets:
            ppm_out = pool_net(x)
            ppm_outs.append(ppm_out.view(*x.shape[:2], -1))
        ppm_outs = torch.cat(ppm_outs, dim=2)
        return ppm_outs


class AFNBlock(nn.Module):

    def __init__(self, low_in_channels, high_in_channels, transform_channels, out_channels, query_scales, key_pool_scales, norm_cfg=None, act_cfg=None):
        super(AFNBlock, self).__init__()
        self.stages = nn.ModuleList()
        for query_scale in query_scales:
            key_psp = PPMConcat(key_pool_scales)
            if query_scale > 1:
                query_downsample = nn.MaxPool2d(kernel_size=query_scale)
            else:
                query_downsample = None
            self.stages.append(SelfAttentionBlock(key_in_channels=low_in_channels, query_in_channels=high_in_channels, transform_channels=transform_channels, out_channels=out_channels, share_key_query=False, query_downsample=query_downsample, key_downsample=key_psp, key_query_num_convs=1, value_out_num_convs=1, key_query_norm=True, value_out_norm=False, matmul_norm=True, with_out_project=True, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.bottleneck = nn.Sequential(nn.Conv2d(out_channels + high_in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg))
    """forward"""

    def forward(self, low_feats, high_feats):
        priors = [stage(high_feats, low_feats) for stage in self.stages]
        context = torch.stack(priors, dim=0).sum(dim=0)
        output = self.bottleneck(torch.cat([context, high_feats], 1))
        return output


class APNBlock(nn.Module):

    def __init__(self, in_channels, transform_channels, out_channels, query_scales, key_pool_scales, norm_cfg=None, act_cfg=None):
        super(APNBlock, self).__init__()
        self.stages = nn.ModuleList()
        for query_scale in query_scales:
            key_psp = PPMConcat(key_pool_scales)
            if query_scale > 1:
                query_downsample = nn.MaxPool2d(kernel_size=query_scale)
            else:
                query_downsample = None
            self.stages.append(SelfAttentionBlock(key_in_channels=in_channels, query_in_channels=in_channels, transform_channels=transform_channels, out_channels=out_channels, share_key_query=True, query_downsample=query_downsample, key_downsample=key_psp, key_query_num_convs=1, value_out_num_convs=1, key_query_norm=True, value_out_norm=False, matmul_norm=True, with_out_project=True, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.bottleneck = nn.Sequential(nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
    """forward"""

    def forward(self, feats):
        priors = [stage(feats, feats) for stage in self.stages]
        context = torch.stack(priors, dim=0).sum(dim=0)
        output = self.bottleneck(torch.cat([context, feats], 1))
        return output


class AdaptiveContextModule(nn.Module):

    def __init__(self, in_channels, out_channels, pool_scale, align_corners, norm_cfg=None, act_cfg=None):
        super(AdaptiveContextModule, self).__init__()
        self.pool_scale = pool_scale
        self.align_corners = align_corners
        self.pooled_redu_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.input_redu_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.global_info = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.gla = nn.Conv2d(out_channels, pool_scale ** 2, kernel_size=1, stride=1, padding=0)
        self.residual_conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.fusion_conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
    """forward"""

    def forward(self, x):
        batch_size = x.size(0)
        pooled_x = F.adaptive_avg_pool2d(x, self.pool_scale)
        x = self.input_redu_conv(x)
        pooled_x = self.pooled_redu_conv(pooled_x)
        pooled_x = pooled_x.view(batch_size, pooled_x.size(1), -1).permute(0, 2, 1).contiguous()
        affinity_matrix = x + F.interpolate(self.global_info(F.adaptive_avg_pool2d(x, 1)), size=x.shape[2:], align_corners=self.align_corners, mode='bilinear')
        affinity_matrix = self.gla(affinity_matrix).permute(0, 2, 3, 1).reshape(batch_size, -1, self.pool_scale ** 2)
        affinity_matrix = F.sigmoid(affinity_matrix)
        z_out = torch.matmul(affinity_matrix, pooled_x)
        z_out = z_out.permute(0, 2, 1).contiguous()
        z_out = z_out.view(batch_size, z_out.size(1), x.size(2), x.size(3))
        z_out = self.residual_conv(z_out)
        z_out = F.relu(z_out + x)
        z_out = self.fusion_conv(z_out)
        return z_out


class DefaultAuxiliaryDecoder(nn.Sequential):

    def __init__(self, **kwargs):
        auxiliary_cfg = kwargs
        num_convs, dec = auxiliary_cfg.get('num_convs', 1), []
        for idx in range(num_convs):
            if idx == 0:
                dec.append(nn.Conv2d(auxiliary_cfg['in_channels'], auxiliary_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False))
            else:
                dec.append(nn.Conv2d(auxiliary_cfg['out_channels'], auxiliary_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False))
            dec.append(BuildNormalization(placeholder=auxiliary_cfg['out_channels'], norm_cfg=auxiliary_cfg['norm_cfg']))
            dec.append(BuildActivation(auxiliary_cfg['act_cfg']))
            if 'upsample' in auxiliary_cfg:
                dec.append(nn.Upsample(**auxiliary_cfg['upsample']))
        dec.append(nn.Dropout2d(auxiliary_cfg['dropout']))
        if num_convs > 0:
            dec.append(nn.Conv2d(auxiliary_cfg['out_channels'], auxiliary_cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        else:
            dec.append(nn.Conv2d(auxiliary_cfg['in_channels'], auxiliary_cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        super(DefaultAuxiliaryDecoder, self).__init__(*dec)


class AuxiliaryDecoderBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {'DefaultAuxiliaryDecoder': DefaultAuxiliaryDecoder}
    """build"""

    def build(self, auxiliary_cfg, norm_cfg, act_cfg, num_classes):
        auxiliary_cfg = copy.deepcopy(auxiliary_cfg)
        if 'type' not in auxiliary_cfg:
            auxiliary_cfg['type'] = 'DefaultAuxiliaryDecoder'
        if 'norm_cfg' not in auxiliary_cfg:
            auxiliary_cfg['norm_cfg'] = norm_cfg
        if 'act_cfg' not in auxiliary_cfg:
            auxiliary_cfg['act_cfg'] = act_cfg
        if 'num_classes' not in auxiliary_cfg:
            auxiliary_cfg['num_classes'] = num_classes
        return super().build(auxiliary_cfg)


BuildAuxiliaryDecoder = AuxiliaryDecoderBuilder().build


class ResNeSt(ResNet):
    arch_settings = {(50): (Bottleneck, (3, 4, 6, 3)), (101): (Bottleneck, (3, 4, 23, 3)), (152): (Bottleneck, (3, 8, 36, 3)), (200): (Bottleneck, (3, 24, 36, 3))}

    def __init__(self, structure_type, groups=1, base_width=4, radix=2, reduction_factor=4, use_avg_after_block_conv2=True, in_channels=3, base_channels=64, stem_channels=128, depth=101, outstride=8, contract_dilation=True, use_conv3x3_stem=True, out_indices=(0, 1, 2, 3), use_avg_for_downsample=True, norm_cfg={'type': 'SyncBatchNorm'}, act_cfg={'type': 'ReLU', 'inplace': True}, pretrained=True, pretrained_model_path=''):
        self.extra_args_for_makelayer = {'radix': radix, 'groups': groups, 'base_width': base_width, 'reduction_factor': reduction_factor, 'base_channels': base_channels, 'use_avg_after_block_conv2': use_avg_after_block_conv2}
        super(ResNeSt, self).__init__(structure_type, in_channels, base_channels, stem_channels, depth, outstride, contract_dilation, use_conv3x3_stem, out_indices, use_avg_for_downsample, norm_cfg, act_cfg, False, '')
        self.structure_type = structure_type
        self.groups = groups
        self.base_width = base_width
        self.radix = radix
        self.reduction_factor = reduction_factor
        self.use_avg_after_block_conv2 = use_avg_after_block_conv2
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.stem_channels = stem_channels
        self.depth = depth
        self.outstride = outstride
        self.contract_dilation = contract_dilation
        self.use_conv3x3_stem = use_conv3x3_stem
        self.out_indices = out_indices
        self.use_avg_for_downsample = use_avg_for_downsample
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and getattr(self, key) == value
        if pretrained:
            state_dict = loadpretrainedweights(structure_type=structure_type, pretrained_model_path=pretrained_model_path, default_model_urls=DEFAULT_MODEL_URLS)
            self.load_state_dict(state_dict, strict=False)
    """makelayer"""

    def makelayer(self, block, inplanes, planes, num_blocks, stride=1, dilation=1, contract_dilation=True, use_avg_for_downsample=False, norm_cfg=None, act_cfg=None):
        downsample = None
        dilations = [dilation] * num_blocks
        if contract_dilation and dilation > 1:
            dilations[0] = dilation // 2
        if stride != 1 or inplanes != planes * block.expansion:
            if use_avg_for_downsample:
                downsample = nn.Sequential(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False), nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=planes * block.expansion, norm_cfg=norm_cfg))
            else:
                downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, padding=0, bias=False), BuildNormalization(placeholder=planes * block.expansion, norm_cfg=norm_cfg))
        layers = []
        layers.append(block(inplanes, planes, stride=stride, dilation=dilations[0], downsample=downsample, norm_cfg=norm_cfg, act_cfg=act_cfg, **self.extra_args_for_makelayer))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(planes * block.expansion, planes, stride=1, dilation=dilations[i], norm_cfg=norm_cfg, act_cfg=act_cfg, **self.extra_args_for_makelayer))
        return nn.Sequential(*layers)


class BackboneBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {'UNet': UNet, 'BEiT': BEiT, 'CGNet': CGNet, 'HRNet': HRNet, 'MobileViT': MobileViT, 'MobileViTV2': MobileViTV2, 'ERFNet': ERFNet, 'ResNet': ResNet, 'ResNeSt': ResNeSt, 'PCPVT': PCPVT, 'MobileSAMTinyViT': MobileSAMTinyViT, 'SVT': SVT, 'FastSCNN': FastSCNN, 'ConvNeXt': ConvNeXt, 'BiSeNetV1': BiSeNetV1, 'MAE': MAE, 'SAMViT': SAMViT, 'SwinTransformer': SwinTransformer, 'VisionTransformer': VisionTransformer, 'EdgeSAMRepViT': EdgeSAMRepViT, 'MixVisionTransformer': MixVisionTransformer, 'TIMMBackbone': TIMMBackbone, 'ConvNeXtV2': ConvNeXtV2, 'Hiera': Hiera, 'MobileNetV2': MobileNetV2, 'MobileNetV3': MobileNetV3, 'BiSeNetV2': BiSeNetV2, 'HieraWithFPN': HieraWithFPN}
    """build"""

    def build(self, backbone_cfg):
        backbone_cfg = copy.deepcopy(backbone_cfg)
        if 'selected_indices' in backbone_cfg:
            backbone_cfg.pop('selected_indices')
        return super().build(backbone_cfg)


BuildBackbone = BackboneBuilder().build


class BasePixelSampler(object):

    def __init__(self, sample_ratio=0.5):
        self.sample_ratio = sample_ratio
    """sample"""

    def sample(self, seg_logits, seg_targets, **kwargs):
        assert seg_logits.shape[-2:] == seg_targets.shape[-2:]
        n, c, h, w = seg_logits.shape
        num_pixels = h * w
        sampled_num_pixels = int(self.sample_ratio * num_pixels)
        indices = list(range(num_pixels))
        random.shuffle(indices)
        indices = indices[:sampled_num_pixels]
        seg_logits = seg_logits.permute(2, 3, 0, 1).contiguous().reshape(h * w, n, c)
        seg_logits = seg_logits[indices].permute(1, 2, 0).contiguous().reshape(n * c, sampled_num_pixels)
        seg_targets = seg_targets.permute(1, 2, 0).contiguous().reshape(h * w, n)
        seg_targets = seg_targets[indices].permute(1, 0).contiguous().reshape(n, sampled_num_pixels)
        return seg_logits, seg_targets


class OHEMPixelSampler(BasePixelSampler):

    def __init__(self, loss_func=None, thresh=None, min_kept_per_image=100000, ignore_index=255):
        super(OHEMPixelSampler, self).__init__()
        assert min_kept_per_image > 1
        assert loss_func is None or thresh is None
        self.loss_func = loss_func
        self.thresh = thresh
        self.min_kept_per_image = min_kept_per_image
        self.ignore_index = ignore_index
    """sample"""

    @torch.no_grad()
    def sample(self, seg_logits, seg_targets, **kwargs):
        assert seg_logits.shape[-2:] == seg_targets.shape[-2:]
        seg_targets = seg_targets.long()
        batch_kept = self.min_kept_per_image * seg_targets.size(0)
        valid_mask = seg_targets != self.ignore_index
        seg_weights = seg_logits.new_zeros(size=seg_targets.size())
        valid_seg_weights = seg_weights[valid_mask]
        if self.thresh is not None:
            seg_probs = F.softmax(seg_logits, dim=1)
            tmp_seg_targets = seg_targets.clone().unsqueeze(1)
            tmp_seg_targets[tmp_seg_targets == self.ignore_index] = 0
            seg_probs = seg_probs.gather(1, tmp_seg_targets).squeeze(1)
            sort_prob, sort_indices = seg_probs[valid_mask].sort()
            if sort_prob.numel() > 0:
                min_threshold = sort_prob[min(batch_kept, sort_prob.numel() - 1)]
            else:
                min_threshold = 0.0
            threshold = max(min_threshold, self.thresh)
            valid_seg_weights[seg_probs[valid_mask] < threshold] = 1.0
        else:
            losses = self.loss_func(seg_logits, seg_targets)
            _, sort_indices = losses[valid_mask].sort(descending=True)
            valid_seg_weights[sort_indices[:batch_kept]] = 1.0
        seg_weights[valid_mask] = valid_seg_weights
        with torch.enable_grad():
            point_logits = seg_logits.permute(0, 2, 3, 1).contiguous()[seg_weights > 0]
        point_targets = seg_targets[seg_weights > 0]
        return point_logits, point_targets


class UGSampler(BasePixelSampler):

    def __init__(self, num_points, oversample_ratio, importance_sample_ratio, ignore_index=255, num_classes=None, reformat_target=True):
        super(UGSampler, self).__init__()
        assert oversample_ratio >= 1
        assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.reformat_target = reformat_target
    """sample"""

    @torch.no_grad()
    def sample(self, seg_logits, seg_targets, **kwargs):
        if self.reformat_target:
            seg_logits_new, seg_targets_new = [], []
            for bs in range(seg_logits.shape[0]):
                seg_logits_new.append(seg_logits[bs])
                seg_targets_per_img = seg_targets[bs]
                masks = []
                for label in range(self.num_classes):
                    masks.append((seg_targets_per_img == label).unsqueeze(0))
                masks = torch.cat(masks, dim=0).type_as(seg_logits).long()
                seg_targets_new.append(masks)
            seg_logits = torch.cat(seg_logits_new, dim=0)
            seg_targets = torch.cat(seg_targets_new, dim=0)
        assert len(seg_logits.shape) == 3 and len(seg_targets.shape) == 3
        with torch.enable_grad():
            seg_logits, seg_targets = seg_logits[:, None], seg_targets[:, None]
        point_coords = self.getuncertainpointcoordswithrandomness(seg_logits, lambda logits: self.calculateuncertainty(logits), self.num_points, self.oversample_ratio, self.importance_sample_ratio)
        point_targets = self.pointsample(seg_targets, point_coords, align_corners=False).squeeze(1)
        with torch.enable_grad():
            point_logits = self.pointsample(seg_logits, point_coords, align_corners=False).squeeze(1)
        return point_logits, point_targets
    """calculateuncertainty"""

    @staticmethod
    def calculateuncertainty(logits):
        assert logits.shape[1] == 1
        gt_class_logits = logits.clone()
        return -torch.abs(gt_class_logits)
    """pointsample"""

    @staticmethod
    def pointsample(inputs, point_coords, **kwargs):
        add_dim = False
        if point_coords.dim() == 3:
            add_dim = True
            point_coords = point_coords.unsqueeze(2)
        output = F.grid_sample(inputs, 2.0 * point_coords - 1.0, **kwargs)
        if add_dim:
            output = output.squeeze(3)
        return output
    """getuncertainpointcoordswithrandomness"""

    @staticmethod
    def getuncertainpointcoordswithrandomness(coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio):
        assert oversample_ratio >= 1
        assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
        num_boxes = coarse_logits.shape[0]
        num_sampled = int(num_points * oversample_ratio)
        point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
        point_logits = UGSampler.pointsample(coarse_logits, point_coords, align_corners=False)
        point_uncertainties = uncertainty_func(point_logits)
        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points
        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
        idx += shift[:, None]
        point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)
        if num_random_points > 0:
            point_coords = torch.cat([point_coords, torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device)], dim=1)
        return point_coords


class PixelSamplerBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {'OHEMPixelSampler': OHEMPixelSampler, 'UGSampler': UGSampler}
    """build"""

    def build(self, pixelsampler_cfg):
        return super().build(pixelsampler_cfg)


BuildPixelSampler = PixelSamplerBuilder().build


class LossBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {'L1Loss': L1Loss, 'MSELoss': MSELoss, 'FocalLoss': FocalLoss, 'CosineSimilarityLoss': CosineSimilarityLoss, 'DiceLoss': DiceLoss, 'KLDivLoss': KLDivLoss, 'LovaszLoss': LovaszLoss, 'CrossEntropyLoss': CrossEntropyLoss}
    """build"""

    def build(self, loss_cfg):
        return super().build(loss_cfg)


BuildLoss = LossBuilder().build


def calculateloss(x_src, x_tgt, loss_cfg):
    assert isinstance(loss_cfg, (dict, list, tuple))
    if isinstance(loss_cfg, dict):
        loss = BuildLoss(loss_cfg)(x_src, x_tgt)
    else:
        loss = 0
        for l_cfg in loss_cfg:
            loss = loss + BuildLoss(l_cfg)(x_src, x_tgt)
    return loss


def calculatelosses(predictions, annotations, losses_cfg, preds_to_tgts_mapping=None, pixel_sampler=None):
    assert len(predictions) == len(losses_cfg), 'length of losses_cfg should be equal to the one of predictions'
    annotations_reorg = {}
    for pred_key in list(predictions.keys()):
        if preds_to_tgts_mapping is None:
            annotations_reorg[pred_key] = annotations['seg_targets']
        else:
            annotations_reorg[pred_key] = annotations[preds_to_tgts_mapping[pred_key]]
    annotations = annotations_reorg
    if pixel_sampler is not None:
        for pred_key in list(predictions.keys()):
            predictions[pred_key], annotations[pred_key] = pixel_sampler.sample(predictions[pred_key], annotations[pred_key])
    losses_log_dict = {}
    for loss_name, loss_cfg in losses_cfg.items():
        losses_log_dict[loss_name] = calculateloss(x_src=predictions[loss_name], x_tgt=annotations[loss_name], loss_cfg=loss_cfg)
    loss_total = 0
    for loss_key, loss_value in losses_log_dict.items():
        loss_value = loss_value.mean()
        loss_total = loss_total + loss_value
        loss_value = loss_value.data.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_value.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
        losses_log_dict[loss_key] = loss_value.item()
    losses_log_dict.update({'loss_total': sum(losses_log_dict.values())})
    return loss_total, losses_log_dict


class BaseSegmentor(nn.Module):

    def __init__(self, cfg, mode):
        super(BaseSegmentor, self).__init__()
        self.cfg = cfg
        self.mode = mode
        assert self.mode in ['TRAIN', 'TEST', 'TRAIN_DEVELOP']
        for key in ['align_corners', 'norm_cfg', 'act_cfg']:
            if key in cfg:
                setattr(self, key, cfg[key])
        self.setbackbone(cfg=cfg)
        self.setpixelsampler(cfg=cfg)
    """forward"""

    def forward(self, data_meta):
        raise NotImplementedError('not to be implemented')
    """customizepredsandlosses"""

    def customizepredsandlosses(self, seg_logits, annotations, backbone_outputs, losses_cfg, img_size, auto_calc_loss=True, preds_to_tgts_mapping=None):
        seg_logits = F.interpolate(seg_logits, size=img_size, mode='bilinear', align_corners=self.align_corners)
        predictions = {'loss_cls': seg_logits}
        if hasattr(self, 'auxiliary_decoder'):
            backbone_outputs = backbone_outputs[:-1]
            if isinstance(self.auxiliary_decoder, nn.ModuleList):
                assert len(backbone_outputs) >= len(self.auxiliary_decoder)
                backbone_outputs = backbone_outputs[-len(self.auxiliary_decoder):]
                for idx, (out, dec) in enumerate(zip(backbone_outputs, self.auxiliary_decoder)):
                    predictions[f'loss_aux{idx + 1}'] = F.interpolate(dec(out), size=img_size, mode='bilinear', align_corners=self.align_corners)
            else:
                predictions['loss_aux'] = F.interpolate(self.auxiliary_decoder(backbone_outputs[-1]), size=img_size, mode='bilinear', align_corners=self.align_corners)
        if not auto_calc_loss:
            return predictions
        return calculatelosses(predictions=predictions, annotations=annotations, losses_cfg=losses_cfg, preds_to_tgts_mapping=preds_to_tgts_mapping, pixel_sampler=self.pixel_sampler)
    """inference"""

    def inference(self, images, forward_args=None):
        inference_cfg = self.cfg['inference']
        assert inference_cfg['forward']['mode'] in ['whole', 'slide']
        use_probs_before_resize = inference_cfg['tta']['use_probs_before_resize']
        images = images
        if inference_cfg['forward']['mode'] == 'whole':
            if forward_args is None:
                seg_logits = self(SSSegInputStructure(images=images, mode=self.mode)).seg_logits
            else:
                seg_logits = self(SSSegInputStructure(images=images, mode=self.mode), **forward_args).seg_logits
            if use_probs_before_resize:
                seg_logits = F.softmax(seg_logits, dim=1)
        else:
            stride_h, stride_w = inference_cfg['forward']['stride']
            cropsize_h, cropsize_w = inference_cfg['forward']['cropsize']
            batch_size, _, image_h, image_w = images.size()
            num_grids_h = max(image_h - cropsize_h + stride_h - 1, 0) // stride_h + 1
            num_grids_w = max(image_w - cropsize_w + stride_w - 1, 0) // stride_w + 1
            seg_logits = images.new_zeros((batch_size, self.cfg['num_classes'], image_h, image_w))
            count_mat = images.new_zeros((batch_size, 1, image_h, image_w))
            for h_idx in range(num_grids_h):
                for w_idx in range(num_grids_w):
                    x1, y1 = w_idx * stride_w, h_idx * stride_h
                    x2, y2 = min(x1 + cropsize_w, image_w), min(y1 + cropsize_h, image_h)
                    x1, y1 = max(x2 - cropsize_w, 0), max(y2 - cropsize_h, 0)
                    crop_images = images[:, :, y1:y2, x1:x2]
                    if forward_args is None:
                        seg_logits_crop = self(SSSegInputStructure(images=crop_images, mode=self.mode)).seg_logits
                    else:
                        seg_logits_crop = self(SSSegInputStructure(images=crop_images, mode=self.mode), **forward_args).seg_logits
                    seg_logits_crop = F.interpolate(seg_logits_crop, size=crop_images.size()[2:], mode='bilinear', align_corners=self.align_corners)
                    if use_probs_before_resize:
                        seg_logits_crop = F.softmax(seg_logits_crop, dim=1)
                    seg_logits += F.pad(seg_logits_crop, (int(x1), int(seg_logits.shape[3] - x2), int(y1), int(seg_logits.shape[2] - y2)))
                    count_mat[:, :, y1:y2, x1:x2] += 1
            assert (count_mat == 0).sum() == 0
            seg_logits = seg_logits / count_mat
        return seg_logits
    """auginference"""

    def auginference(self, images, forward_args=None):
        inference_cfg = self.cfg['inference']
        infer_tta_cfg, seg_logits_list = inference_cfg['tta'], []
        for scale_factor in infer_tta_cfg['multiscale']:
            images_scale = F.interpolate(images, scale_factor=scale_factor, mode='bilinear', align_corners=self.align_corners)
            seg_logits = self.inference(images=images_scale, forward_args=forward_args).cpu()
            seg_logits_list.append(seg_logits)
            if infer_tta_cfg['flip']:
                images_scale_flip = torch.from_numpy(np.flip(images_scale.cpu().numpy(), axis=3).copy())
                seg_logits_flip = self.inference(images=images_scale_flip, forward_args=forward_args)
                fixed_seg_target_pairs = infer_tta_cfg.get('fixed_seg_target_pairs', None)
                if fixed_seg_target_pairs is None:
                    for data_pipeline in self.cfg['dataset']['train']['data_pipelines']:
                        if 'RandomFlip' in data_pipeline:
                            if isinstance(data_pipeline, dict):
                                fixed_seg_target_pairs = data_pipeline['RandomFlip'].get('fixed_seg_target_pairs', None)
                            else:
                                fixed_seg_target_pairs = data_pipeline[-1].get('fixed_seg_target_pairs', None)
                if fixed_seg_target_pairs is not None:
                    seg_logits_flip_clone = seg_logits_flip.data.clone()
                    for pair_a, pair_b in fixed_seg_target_pairs:
                        seg_logits_flip[:, pair_a, :, :] = seg_logits_flip_clone[:, pair_b, :, :]
                        seg_logits_flip[:, pair_b, :, :] = seg_logits_flip_clone[:, pair_a, :, :]
                seg_logits_flip = torch.from_numpy(np.flip(seg_logits_flip.cpu().numpy(), axis=3).copy()).type_as(seg_logits)
                seg_logits_list.append(seg_logits_flip)
        return seg_logits_list
    """transforminputs"""

    def transforminputs(self, x_list, selected_indices=None):
        if selected_indices is None:
            if self.cfg['backbone']['type'] in ['HRNet']:
                selected_indices = 0, 0, 0, 0
            else:
                selected_indices = 0, 1, 2, 3
        outs = []
        for idx in selected_indices:
            outs.append(x_list[idx])
        return outs
    """setauxiliarydecoder"""

    def setauxiliarydecoder(self, auxiliary_cfg):
        if auxiliary_cfg is None:
            return
        if isinstance(auxiliary_cfg, dict):
            auxiliary_cfg = [auxiliary_cfg]
        self.auxiliary_decoder = nn.ModuleList()
        for aux_cfg in auxiliary_cfg:
            self.auxiliary_decoder.append(BuildAuxiliaryDecoder(auxiliary_cfg=aux_cfg, norm_cfg=self.norm_cfg.copy(), act_cfg=self.act_cfg.copy(), num_classes=self.cfg['num_classes']))
        if len(self.auxiliary_decoder) == 1:
            self.auxiliary_decoder = self.auxiliary_decoder[0]
    """setbackbone"""

    def setbackbone(self, cfg):
        if 'backbone' not in cfg:
            return
        backbone_cfg = copy.deepcopy(cfg['backbone'])
        if 'norm_cfg' not in backbone_cfg:
            backbone_cfg.update({'norm_cfg': copy.deepcopy(self.norm_cfg)})
        self.backbone_net = BuildBackbone(backbone_cfg)
    """setpixelsampler"""

    def setpixelsampler(self, cfg):
        if 'pixelsampler' in cfg['head']:
            self.pixel_sampler = BuildPixelSampler(cfg['head']['pixelsampler'])
        else:
            self.pixel_sampler = None
    """freezenormalization"""

    def freezenormalization(self, norm_list=None):
        if norm_list is None:
            norm_list = nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm
        for module in self.modules():
            if NormalizationBuilder.isnorm(module, norm_list):
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False
    """train"""

    def train(self, mode=True):
        self.mode = 'TRAIN' if mode else 'TEST'
        return super().train(mode)
    """eval"""

    def eval(self):
        self.mode = 'TEST'
        return super().eval()


class EMASegmentor(nn.Module):

    def __init__(self, segmentor, momentum=0.0005, device='cpu'):
        super(EMASegmentor, self).__init__()
        self.device = device
        self.momentum = momentum
        if hasattr(segmentor, 'module'):
            segmentor = segmentor.module
        self.segmentor_ema = copy.deepcopy(segmentor)
        if device:
            self.segmentor_ema
        self.segmentor_ema.eval()
        for param in self.segmentor_ema.parameters():
            param.requires_grad = False
    """forward"""

    def forward(self, x, targets, **kwargs):
        return self.segmentor_ema(x, targets, **kwargs)
    """state"""

    def state(self):
        return self.segmentor_ema.state_dict()
    """setstate"""

    def setstate(self, state_dict, strict=True):
        return self.segmentor_ema.load_state_dict(state_dict, strict=strict)
    """update"""

    def update(self, segmentor):
        if self.device:
            self.segmentor_ema
        if hasattr(segmentor, 'module'):
            segmentor = segmentor.module
        with torch.no_grad():
            state_dict = segmentor.state_dict()
            for ema_k, ema_v in self.segmentor_ema.state_dict().items():
                cur_v = state_dict[ema_k].detach()
                if self.device:
                    cur_v = cur_v
                ema_v.copy_(ema_v * (1.0 - self.momentum) + self.momentum * cur_v)


class Feature2Pyramid(nn.Module):

    def __init__(self, embed_dim, rescales=[4, 2, 1, 0.5], norm_cfg=None):
        super(Feature2Pyramid, self).__init__()
        self.rescales = rescales
        self.upsample_4x = None
        for k in self.rescales:
            if k == 4:
                self.upsample_4x = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2), BuildNormalization(placeholder=embed_dim, norm_cfg=norm_cfg), nn.GELU(), nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
            elif k == 2:
                self.upsample_2x = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
            elif k == 1:
                self.identity = nn.Identity()
            elif k == 0.5:
                self.downsample_2x = nn.MaxPool2d(kernel_size=2, stride=2)
            elif k == 0.25:
                self.downsample_4x = nn.MaxPool2d(kernel_size=4, stride=4)
            else:
                raise KeyError(f'invalid {k} for feature2pyramid')
    """forward"""

    def forward(self, inputs):
        assert len(inputs) == len(self.rescales)
        outputs = []
        if self.upsample_4x is not None:
            ops = [self.upsample_4x, self.upsample_2x, self.identity, self.downsample_2x]
        else:
            ops = [self.upsample_2x, self.identity, self.downsample_2x, self.downsample_4x]
        for i in range(len(inputs)):
            outputs.append(ops[i](inputs[i]))
        return tuple(outputs)


class FPN(nn.Module):

    def __init__(self, in_channels_list, out_channels, upsample_cfg=dict(mode='nearest'), norm_cfg=None, act_cfg=None):
        super(FPN, self).__init__()
        self.in_channels_list = in_channels_list
        self.upsample_cfg = upsample_cfg
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        if 'inplace' in act_cfg:
            act_cfg['inplace'] = False
        for i in range(0, len(in_channels_list)):
            l_conv = nn.Sequential(nn.Conv2d(in_channels_list[i], out_channels, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
            fpn_conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
    """forward"""

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels_list)
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        return tuple(outs)


class CCNet(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(CCNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        self.conv_before_cca = nn.Sequential(nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.cca = CrissCrossAttention(head_cfg['feats_channels'])
        self.conv_after_cca = nn.Sequential(nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.decoder = nn.Sequential(nn.Conv2d(head_cfg['in_channels'] + head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.setauxiliarydecoder(cfg['auxiliary'])
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        feats = self.conv_before_cca(backbone_outputs[-1])
        for _ in range(self.cfg['head']['num_recurrence']):
            feats = self.cca(feats)
        feats = self.conv_after_cca(feats)
        feats = torch.cat([backbone_outputs[-1], feats], dim=1)
        seg_logits = self.decoder(feats)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            loss, losses_log_dict = self.customizepredsandlosses(seg_logits=seg_logits, annotations=data_meta.getannotations(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size)
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs


class EdgePerceivingModule(nn.Module):

    def __init__(self, in_channels_list=[256, 512, 1024], hidden_channels=256, out_channels=2, align_corners=False, norm_cfg=None, act_cfg=None):
        super(EdgePerceivingModule, self).__init__()
        self.align_corners = align_corners
        self.branches = nn.ModuleList()
        for in_channels in in_channels_list:
            self.branches.append(nn.Sequential(nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=hidden_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg)))
        self.edge_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.fuse_conv = nn.Conv2d(out_channels * len(in_channels_list), out_channels, kernel_size=1, stride=1, padding=0, bias=True)
    """forward"""

    def forward(self, x):
        assert len(x) == len(self.branches)
        h, w = x[0].size(2), x[0].size(3)
        edges_feats, edges = [], []
        for i in range(len(x)):
            edge_feats = self.branches[i](x[i])
            edge = self.edge_conv(edge_feats)
            if i > 0:
                edge_feats = F.interpolate(edge_feats, size=(h, w), mode='bilinear', align_corners=self.align_corners)
                edge = F.interpolate(edge, size=(h, w), mode='bilinear', align_corners=self.align_corners)
            edges_feats.append(edge_feats)
            edges.append(edge)
        edge_feats = torch.cat(edges_feats, dim=1)
        edge = torch.cat(edges, dim=1)
        edge = self.fuse_conv(edge)
        return edge, edge_feats


class PyramidPoolingModule(nn.Module):

    def __init__(self, in_channels, out_channels, pool_scales, align_corners=False, norm_cfg=None, act_cfg=None):
        super(PyramidPoolingModule, self).__init__()
        self.align_corners = align_corners
        self.branches = nn.ModuleList()
        for pool_scale in pool_scales:
            self.branches.append(nn.Sequential(nn.AdaptiveAvgPool2d(output_size=pool_scale), nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg)))
        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels + out_channels * len(pool_scales), out_channels, kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.in_channels = in_channels
        self.out_channels = out_channels
    """forward"""

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramid_lvls = [x]
        for branch in self.branches:
            out = branch(x)
            pyramid_lvls.append(F.interpolate(out, size=(h, w), mode='bilinear', align_corners=self.align_corners))
        output = torch.cat(pyramid_lvls, dim=1)
        output = self.bottleneck(output)
        return output


class CE2P(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(CE2P, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        ppm_cfg = {'in_channels': head_cfg['in_channels_list'][-1], 'out_channels': head_cfg['feats_channels'], 'pool_scales': head_cfg['pool_scales'], 'align_corners': align_corners, 'norm_cfg': copy.deepcopy(norm_cfg), 'act_cfg': copy.deepcopy(act_cfg)}
        self.ppm_net = PyramidPoolingModule(**ppm_cfg)
        epm_cfg = {'in_channels_list': head_cfg['in_channels_list'][:-1], 'hidden_channels': head_cfg['epm_hidden_channels'], 'out_channels': head_cfg['epm_out_channels'], 'align_corners': align_corners, 'norm_cfg': copy.deepcopy(norm_cfg), 'act_cfg': copy.deepcopy(act_cfg)}
        self.edge_net = EdgePerceivingModule(**epm_cfg)
        self.shortcut = nn.Sequential(nn.Conv2d(head_cfg['in_channels_list'][0], head_cfg['shortcut_feats_channels'], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=head_cfg['shortcut_feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.decoder_stage1 = nn.Sequential(nn.Conv2d(head_cfg['feats_channels'] + head_cfg['shortcut_feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Dropout2d(head_cfg['dropout_stage1']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.decoder_stage2 = nn.Sequential(nn.Conv2d(head_cfg['feats_channels'] + head_cfg['epm_hidden_channels'] * (len(head_cfg['in_channels_list']) - 1), head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Dropout2d(head_cfg['dropout_stage2']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        ppm_out = self.ppm_net(backbone_outputs[-1])
        ppm_out = F.interpolate(ppm_out, size=backbone_outputs[0].shape[2:], mode='bilinear', align_corners=self.align_corners)
        edge, edge_feats = self.edge_net(backbone_outputs[:-1])
        shortcut_out = self.shortcut(backbone_outputs[0])
        feats_stage1 = torch.cat([ppm_out, shortcut_out], dim=1)
        feats_stage1 = self.decoder_stage1[:-1](feats_stage1)
        feats_stage2 = torch.cat([feats_stage1, edge_feats], dim=1)
        preds_stage2 = self.decoder_stage2(feats_stage2)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            edge = F.interpolate(edge, size=img_size, mode='bilinear', align_corners=self.align_corners)
            preds_stage1 = self.decoder_stage1[-1](feats_stage1)
            preds_stage1 = F.interpolate(preds_stage1, size=img_size, mode='bilinear', align_corners=self.align_corners)
            preds_stage2 = F.interpolate(preds_stage2, size=img_size, mode='bilinear', align_corners=self.align_corners)
            edge_targets, losses_cfg = data_meta.getannotations()['edge_targets'], copy.deepcopy(self.cfg['losses'])
            num_neg_edge, num_pos_edge = torch.sum(edge_targets == 0, dtype=torch.float), torch.sum(edge_targets == 1, dtype=torch.float)
            weight_pos_edge, weight_neg_edge = num_neg_edge / (num_pos_edge + num_neg_edge), num_pos_edge / (num_pos_edge + num_neg_edge)
            cls_weight_edge = torch.Tensor([weight_neg_edge, weight_pos_edge]).type_as(edge_targets)
            for loss_name in list(losses_cfg.keys()):
                if 'edge' in loss_name:
                    if isinstance(losses_cfg[loss_name], list):
                        for loss_idx in range(len(losses_cfg[loss_name])):
                            losses_cfg[loss_name][loss_idx]['class_weight'] = cls_weight_edge
                    else:
                        assert isinstance(losses_cfg[loss_name], dict)
                        losses_cfg[loss_name]['class_weight'] = cls_weight_edge
            loss, losses_log_dict = calculatelosses(predictions={'loss_cls_stage1': preds_stage1, 'loss_cls_stage2': preds_stage2, 'loss_edge': edge}, annotations=data_meta.getannotations(), losses_cfg=losses_cfg, preds_to_tgts_mapping={'loss_cls_stage1': 'seg_targets', 'loss_cls_stage2': 'seg_targets', 'loss_edge': 'edge_targets'}, pixel_sampler=self.pixel_sampler)
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=preds_stage2)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=preds_stage2)
        return ssseg_outputs


class ChannelAttentionModule(nn.Module):

    def __init__(self):
        super(ChannelAttentionModule, self).__init__()
        self.gamma = Scale(scale=0)
    """forward"""

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        proj_query = x.view(batch_size, channels, -1)
        proj_key = x.view(batch_size, channels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy_new, dim=-1)
        proj_value = x.view(batch_size, channels, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, channels, height, width)
        out = self.gamma(out) + x
        return out


class DANet(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(DANet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        self.pam_in_conv = nn.Sequential(nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.pam_net = PositionAttentionModule(head_cfg['feats_channels'], head_cfg['transform_channels'])
        self.pam_out_conv = nn.Sequential(nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.decoder_pam = nn.Sequential(nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.cam_in_conv = nn.Sequential(nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.cam_net = ChannelAttentionModule()
        self.cam_out_conv = nn.Sequential(nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.decoder_cam = nn.Sequential(nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.decoder_pamcam = nn.Sequential(nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.setauxiliarydecoder(cfg['auxiliary'])
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        feats_pam = self.pam_in_conv(backbone_outputs[-1])
        feats_pam = self.pam_net(feats_pam)
        feats_pam = self.pam_out_conv(feats_pam)
        feats_cam = self.cam_in_conv(backbone_outputs[-1])
        feats_cam = self.cam_net(feats_cam)
        feats_cam = self.cam_out_conv(feats_cam)
        feats_sum = feats_pam + feats_cam
        preds_pamcam = self.decoder_pamcam(feats_sum)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            predictions = self.customizepredsandlosses(seg_logits=preds_pamcam, annotations=data_meta.getannotations(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size, auto_calc_loss=False)
            preds_pamcam = predictions.pop('loss_cls')
            preds_pam = self.decoder_pam(feats_pam)
            preds_pam = F.interpolate(preds_pam, size=img_size, mode='bilinear', align_corners=self.align_corners)
            preds_cam = self.decoder_cam(feats_cam)
            preds_cam = F.interpolate(preds_cam, size=img_size, mode='bilinear', align_corners=self.align_corners)
            predictions.update({'loss_cls_pam': preds_pam, 'loss_cls_cam': preds_cam, 'loss_cls_pamcam': preds_pamcam})
            loss, losses_log_dict = calculatelosses(predictions=predictions, annotations=data_meta.getannotations(), losses_cfg=self.cfg['losses'], pixel_sampler=self.pixel_sampler)
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=preds_pamcam)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=preds_pamcam)
        return ssseg_outputs


class ASPP(nn.Module):

    def __init__(self, in_channels, out_channels, dilations, align_corners=False, norm_cfg=None, act_cfg=None):
        super(ASPP, self).__init__()
        self.align_corners = align_corners
        self.parallel_branches = nn.ModuleList()
        for idx, dilation in enumerate(dilations):
            if dilation == 1:
                branch = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=dilation, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
            else:
                branch = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
            self.parallel_branches.append(branch)
        self.global_branch = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.bottleneck = nn.Sequential(nn.Conv2d(out_channels * (len(dilations) + 1), out_channels, kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.in_channels = in_channels
        self.out_channels = out_channels
    """forward"""

    def forward(self, x):
        size = x.size()
        outputs = []
        for branch in self.parallel_branches:
            outputs.append(branch(x))
        global_features = self.global_branch(x)
        global_features = F.interpolate(global_features, size=(size[2], size[3]), mode='bilinear', align_corners=self.align_corners)
        outputs.append(global_features)
        features = torch.cat(outputs, dim=1)
        features = self.bottleneck(features)
        return features


class Deeplabv3(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(Deeplabv3, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        aspp_cfg = {'in_channels': head_cfg['in_channels'], 'out_channels': head_cfg['feats_channels'], 'dilations': head_cfg['dilations'], 'align_corners': align_corners, 'norm_cfg': copy.deepcopy(norm_cfg), 'act_cfg': copy.deepcopy(act_cfg)}
        self.aspp_net = ASPP(**aspp_cfg)
        self.decoder = nn.Sequential(nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.setauxiliarydecoder(cfg['auxiliary'])
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        aspp_out = self.aspp_net(backbone_outputs[-1])
        seg_logits = self.decoder(aspp_out)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            loss, losses_log_dict = self.customizepredsandlosses(seg_logits=seg_logits, annotations=data_meta.getannotations(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size)
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs


class DepthwiseSeparableASPP(nn.Module):

    def __init__(self, in_channels, out_channels, dilations, align_corners=False, norm_cfg=None, act_cfg=None):
        super(DepthwiseSeparableASPP, self).__init__()
        self.align_corners = align_corners
        self.parallel_branches = nn.ModuleList()
        for idx, dilation in enumerate(dilations):
            if dilation == 1:
                branch = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=dilation, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
            else:
                branch = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False, norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.parallel_branches.append(branch)
        self.global_branch = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.bottleneck = nn.Sequential(nn.Conv2d(out_channels * (len(dilations) + 1), out_channels, kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.in_channels = in_channels
        self.out_channels = out_channels
    """forward"""

    def forward(self, x):
        size = x.size()
        outputs = []
        for branch in self.parallel_branches:
            outputs.append(branch(x))
        global_features = self.global_branch(x)
        global_features = F.interpolate(global_features, size=(size[2], size[3]), mode='bilinear', align_corners=self.align_corners)
        outputs.append(global_features)
        features = torch.cat(outputs, dim=1)
        features = self.bottleneck(features)
        return features


class Deeplabv3Plus(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(Deeplabv3Plus, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        aspp_cfg = {'in_channels': head_cfg['in_channels'][1], 'out_channels': head_cfg['feats_channels'], 'dilations': head_cfg['dilations'], 'align_corners': align_corners, 'norm_cfg': copy.deepcopy(norm_cfg), 'act_cfg': copy.deepcopy(act_cfg)}
        self.aspp_net = DepthwiseSeparableASPP(**aspp_cfg)
        self.shortcut = nn.Sequential(nn.Conv2d(head_cfg['in_channels'][0], head_cfg['shortcut_channels'], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=head_cfg['shortcut_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.decoder = nn.Sequential(DepthwiseSeparableConv2d(head_cfg['feats_channels'] + head_cfg['shortcut_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False, act_cfg=act_cfg, norm_cfg=norm_cfg), DepthwiseSeparableConv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False, act_cfg=act_cfg, norm_cfg=norm_cfg), nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.setauxiliarydecoder(cfg['auxiliary'])
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        aspp_out = self.aspp_net(backbone_outputs[-1])
        aspp_out = F.interpolate(aspp_out, size=backbone_outputs[0].shape[2:], mode='bilinear', align_corners=self.align_corners)
        shortcut_out = self.shortcut(backbone_outputs[0])
        feats = torch.cat([aspp_out, shortcut_out], dim=1)
        seg_logits = self.decoder(feats)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            loss, losses_log_dict = self.customizepredsandlosses(seg_logits=seg_logits, annotations=data_meta.getannotations(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size)
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs


class DynamicConvolutionalModule(nn.Module):

    def __init__(self, filter_size, is_fusion, in_channels, out_channels, norm_cfg=None, act_cfg=None):
        super(DynamicConvolutionalModule, self).__init__()
        self.filter_size, self.is_fusion = filter_size, is_fusion
        self.filter_gen_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.input_redu_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.norm = BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg)
        self.activate = BuildActivation(act_cfg)
        if is_fusion:
            self.fusion_conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
    """forward"""

    def forward(self, x):
        generated_filter = self.filter_gen_conv(F.adaptive_avg_pool2d(x, self.filter_size))
        x = self.input_redu_conv(x)
        b, c, h, w = x.shape
        x = x.view(1, b * c, h, w)
        generated_filter = generated_filter.view(b * c, 1, self.filter_size, self.filter_size)
        pad = (self.filter_size - 1) // 2
        if (self.filter_size - 1) % 2 == 0:
            p2d = pad, pad, pad, pad
        else:
            p2d = pad + 1, pad, pad + 1, pad
        x = F.pad(input=x, pad=p2d, mode='constant', value=0)
        output = F.conv2d(input=x, weight=generated_filter, groups=b * c)
        output = output.view(b, c, h, w)
        output = self.norm(output)
        output = self.activate(output)
        if self.is_fusion:
            output = self.fusion_conv(output)
        return output


class DMNet(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(DMNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        self.dcm_modules = nn.ModuleList()
        for filter_size in head_cfg['filter_sizes']:
            self.dcm_modules.append(DynamicConvolutionalModule(filter_size=filter_size, is_fusion=head_cfg['is_fusion'], in_channels=head_cfg['in_channels'], out_channels=head_cfg['feats_channels'], norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.decoder = nn.Sequential(nn.Conv2d(head_cfg['feats_channels'] * len(head_cfg['filter_sizes']) + head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.setauxiliarydecoder(cfg['auxiliary'])
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        dcm_outs = [backbone_outputs[-1]]
        for dcm_module in self.dcm_modules:
            dcm_outs.append(dcm_module(backbone_outputs[-1]))
        feats = torch.cat(dcm_outs, dim=1)
        seg_logits = self.decoder(feats)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            loss, losses_log_dict = self.customizepredsandlosses(seg_logits=seg_logits, annotations=data_meta.getannotations(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size)
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs


class _NonLocalNd(nn.Module, metaclass=ABCMeta):

    def __init__(self, in_channels, reduction=2, use_scale=True, mode='embeddedgaussian', norm_cfg=None, act_cfg=None):
        super(_NonLocalNd, self).__init__()
        assert mode in ['gaussian', 'embeddedgaussian', 'dotproduct', 'concatenation']
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = max(in_channels // reduction, 1)
        self.mode = mode
        self.g = nn.Sequential(nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0))
        self.conv_out = nn.Sequential(nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=self.in_channels, norm_cfg=norm_cfg))
        if self.mode != 'gaussian':
            self.theta = nn.Sequential(nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0))
            self.phi = nn.Sequential(nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0))
        if self.mode == 'concatenation':
            self.concat_project = nn.Sequential(nn.Conv2d(self.inter_channels * 2, 1, kernel_size=1, stride=1, padding=0, bias=False), BuildActivation(act_cfg))
    """gaussian"""

    def gaussian(self, theta_x, phi_x):
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight
    """embeddedgaussian"""

    def embeddedgaussian(self, theta_x, phi_x):
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            pairwise_weight /= theta_x.shape[-1] ** 0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight
    """dotproduct"""

    def dotproduct(self, theta_x, phi_x):
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight
    """concatenation"""

    def concatenation(self, theta_x, phi_x):
        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)
        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        pairwise_weight = self.concat_project(concat_feature)
        n, _, h, w = pairwise_weight.size()
        pairwise_weight = pairwise_weight.view(n, h, w)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight
    """forward"""

    def forward(self, x):
        n = x.size(0)
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        if self.mode == 'gaussian':
            theta_x = x.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(x).view(n, self.in_channels, -1)
            else:
                phi_x = x.view(n, self.in_channels, -1)
        elif self.mode == 'concatenation':
            theta_x = self.theta(x).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(x).view(n, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, -1)
        pairwise_func = getattr(self, self.mode)
        pairwise_weight = pairwise_func(theta_x, phi_x)
        y = torch.matmul(pairwise_weight, g_x)
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels, *x.size()[2:])
        output = x + self.conv_out(y)
        return output


class NonLocal2d(_NonLocalNd):

    def __init__(self, in_channels, sub_sample=False, **kwargs):
        super(NonLocal2d, self).__init__(in_channels, **kwargs)
        self.sub_sample = sub_sample
        if sub_sample:
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            self.g = nn.Sequential(self.g, max_pool_layer)
            if self.mode != 'gaussian':
                self.phi = nn.Sequential(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer


class DisentangledNonLocal2d(NonLocal2d):

    def __init__(self, *arg, temperature, **kwargs):
        super(DisentangledNonLocal2d, self).__init__(*arg, **kwargs)
        self.temperature = temperature
        self.conv_mask = nn.Conv2d(self.in_channels, 1, kernel_size=1, stride=1, padding=0)
    """embedded gaussian with temperature"""

    def embeddedgaussian(self, theta_x, phi_x):
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            pairwise_weight /= theta_x.shape[-1] ** 0.5
        pairwise_weight /= self.temperature
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight
    """forward"""

    def forward(self, x):
        n = x.size(0)
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        if self.mode == 'gaussian':
            theta_x = x.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(x).view(n, self.in_channels, -1)
            else:
                phi_x = x.view(n, self.in_channels, -1)
        elif self.mode == 'concatenation':
            theta_x = self.theta(x).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(x).view(n, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, -1)
        theta_x = theta_x - theta_x.mean(dim=-2, keepdim=True)
        phi_x = phi_x - phi_x.mean(dim=-1, keepdim=True)
        pairwise_func = getattr(self, self.mode)
        pairwise_weight = pairwise_func(theta_x, phi_x)
        y = torch.matmul(pairwise_weight, g_x)
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels, *x.size()[2:])
        unary_mask = self.conv_mask(x)
        unary_mask = unary_mask.view(n, 1, -1)
        unary_mask = unary_mask.softmax(dim=-1)
        unary_x = torch.matmul(unary_mask, g_x)
        unary_x = unary_x.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels, 1, 1)
        output = x + self.conv_out(y + unary_x)
        return output


class DNLNet(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(DNLNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        self.conv_before_dnl = nn.Sequential(nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.dnl_block = DisentangledNonLocal2d(in_channels=head_cfg['feats_channels'], reduction=head_cfg['reduction'], use_scale=head_cfg['use_scale'], mode=head_cfg['mode'], temperature=head_cfg['temperature'], norm_cfg=copy.deepcopy(norm_cfg), act_cfg=copy.deepcopy(act_cfg))
        self.conv_after_dnl = nn.Sequential(nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.decoder = nn.Sequential(nn.Conv2d(head_cfg['feats_channels'] + head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.setauxiliarydecoder(cfg['auxiliary'])
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        feats = self.conv_before_dnl(backbone_outputs[-1])
        feats = self.dnl_block(feats)
        feats = self.conv_after_dnl(feats)
        feats = torch.cat([backbone_outputs[-1], feats], dim=1)
        seg_logits = self.decoder(feats)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            loss, losses_log_dict = self.customizepredsandlosses(seg_logits=seg_logits, annotations=data_meta.getannotations(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size)
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs


class EMAModule(nn.Module):

    def __init__(self, channels, num_bases, num_stages, momentum):
        super(EMAModule, self).__init__()
        assert num_stages >= 1, 'num_stages must be at least 1'
        self.num_bases = num_bases
        self.num_stages = num_stages
        self.momentum = momentum
        bases = torch.zeros(1, channels, self.num_bases)
        bases.normal_(0, math.sqrt(2.0 / self.num_bases))
        bases = F.normalize(bases, dim=1, p=2)
        self.register_buffer('bases', bases)
    """forward"""

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        feats = x.view(batch_size, channels, height * width)
        bases = self.bases.repeat(batch_size, 1, 1)
        with torch.no_grad():
            for _ in range(self.num_stages):
                attention = torch.einsum('bcn,bck->bnk', feats, bases)
                attention = F.softmax(attention, dim=2)
                attention_normed = F.normalize(attention, dim=1, p=1)
                bases = torch.einsum('bcn,bnk->bck', feats, attention_normed)
                bases = F.normalize(bases, dim=1, p=2)
        feats_recon = torch.einsum('bck,bnk->bcn', bases, attention)
        feats_recon = feats_recon.view(batch_size, channels, height, width)
        if self.training:
            bases = bases.mean(dim=0, keepdim=True)
            bases = self.reducemean(bases)
            bases = F.normalize(bases, dim=1, p=2)
            self.bases = (1 - self.momentum) * self.bases + self.momentum * bases
        return feats_recon
    """reducemean"""

    def reducemean(self, tensor):
        if not (dist.is_available() and dist.is_initialized()):
            return tensor
        tensor = tensor.clone()
        dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
        return tensor


class EMANet(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(EMANet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        self.ema_in_conv = nn.Sequential(nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.ema_mid_conv = nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0)
        for param in self.ema_mid_conv.parameters():
            param.requires_grad = False
        self.ema_module = EMAModule(channels=head_cfg['feats_channels'], num_bases=head_cfg['num_bases'], num_stages=head_cfg['num_stages'], momentum=head_cfg['momentum'])
        self.ema_out_conv = nn.Sequential(nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg))
        self.bottleneck = nn.Sequential(nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.decoder = nn.Sequential(nn.Conv2d(head_cfg['feats_channels'] + head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.setauxiliarydecoder(cfg['auxiliary'])
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        feats = self.ema_in_conv(backbone_outputs[-1])
        identity = feats
        feats = self.ema_mid_conv(feats)
        recon = self.ema_module(feats)
        recon = F.relu(recon, inplace=True)
        recon = self.ema_out_conv(recon)
        feats = F.relu(identity + recon, inplace=True)
        feats = self.bottleneck(feats)
        feats = torch.cat([backbone_outputs[-1], feats], dim=1)
        seg_logits = self.decoder(feats)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            loss, losses_log_dict = self.customizepredsandlosses(seg_logits=seg_logits, annotations=data_meta.getannotations(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size)
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs


class Encoding(nn.Module):

    def __init__(self, channels, num_codes):
        super(Encoding, self).__init__()
        self.channels, self.num_codes = channels, num_codes
        std = 1.0 / (num_codes * channels) ** 0.5
        self.codewords = nn.Parameter(torch.empty(num_codes, channels, dtype=torch.float).uniform_(-std, std), requires_grad=True)
        self.scale = nn.Parameter(torch.empty(num_codes, dtype=torch.float).uniform_(-1, 0), requires_grad=True)
    """scaledl2"""

    @staticmethod
    def scaledl2(x, codewords, scale):
        batch_size = x.size(0)
        num_codes, channels = codewords.size()
        reshaped_scale = scale.view((1, 1, num_codes))
        expanded_x = x.unsqueeze(2).expand((batch_size, x.size(1), num_codes, channels))
        reshaped_codewords = codewords.view((1, 1, num_codes, channels))
        scaledl2_norm = reshaped_scale * (expanded_x - reshaped_codewords).pow(2).sum(dim=3)
        return scaledl2_norm
    """aggregate"""

    @staticmethod
    def aggregate(assigment_weights, x, codewords):
        batch_size = x.size(0)
        num_codes, channels = codewords.size()
        reshaped_codewords = codewords.view((1, 1, num_codes, channels))
        expanded_x = x.unsqueeze(2).expand((batch_size, x.size(1), num_codes, channels))
        encoded_feat = (assigment_weights.unsqueeze(3) * (expanded_x - reshaped_codewords)).sum(dim=1)
        return encoded_feat
    """forward"""

    def forward(self, x):
        assert x.dim() == 4 and x.size(1) == self.channels
        batch_size = x.size(0)
        x = x.view(batch_size, self.channels, -1).transpose(1, 2).contiguous()
        assigment_weights = F.softmax(self.scaledl2(x, self.codewords, self.scale), dim=2)
        encoded_feat = self.aggregate(assigment_weights, x, self.codewords)
        return encoded_feat


class ContextEncoding(nn.Module):

    def __init__(self, in_channels, num_codes, norm_cfg=None, act_cfg=None):
        super(ContextEncoding, self).__init__()
        self.encoding_project = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=in_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
        encoding_norm_cfg = copy.deepcopy(norm_cfg)
        encoding_norm_cfg['type'] = encoding_norm_cfg['type'].replace('2d', '1d')
        self.encoding = nn.Sequential(Encoding(channels=in_channels, num_codes=num_codes), BuildNormalization(placeholder=num_codes, norm_cfg=encoding_norm_cfg), BuildActivation(act_cfg))
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels), nn.Sigmoid())
    """forward"""

    def forward(self, x):
        encoding_projection = self.encoding_project(x)
        encoding_feat = self.encoding(encoding_projection).mean(dim=1)
        batch_size, channels, _, _ = x.size()
        gamma = self.fc(encoding_feat)
        y = gamma.view(batch_size, channels, 1, 1)
        output = F.relu_(x + x * y)
        return encoding_feat, output


class ENCNet(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(ENCNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        self.bottleneck = nn.Sequential(nn.Conv2d(head_cfg['in_channels_list'][-1], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.enc_module = ContextEncoding(in_channels=head_cfg['feats_channels'], num_codes=head_cfg['num_codes'], norm_cfg=norm_cfg, act_cfg=act_cfg)
        extra_cfg = head_cfg['extra']
        if extra_cfg['add_lateral']:
            self.lateral_convs = nn.ModuleList()
            for in_channels in head_cfg['in_channels_list'][:-1]:
                self.lateral_convs.append(nn.Conv2d(in_channels, head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
            self.fusion = nn.Sequential(nn.Conv2d(len(head_cfg['in_channels_list']) * head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        if extra_cfg['use_se_loss']:
            self.se_layer = nn.Linear(head_cfg['feats_channels'], cfg['num_classes'])
        self.decoder = nn.Sequential(nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.setauxiliarydecoder(cfg['auxiliary'])
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        feats = self.bottleneck(backbone_outputs[-1])
        if hasattr(self, 'lateral_convs'):
            lateral_outs = [F.interpolate(lateral_conv(backbone_outputs[idx]), size=feats.shape[2:], mode='bilinear', align_corners=self.align_corners) for idx, lateral_conv in enumerate(self.lateral_convs)]
            feats = self.fusion(torch.cat([feats, *lateral_outs], dim=1))
        encode_feats, feats = self.enc_module(feats)
        if hasattr(self, 'se_layer'):
            seg_logits_se = self.se_layer(encode_feats)
        seg_logits = self.decoder(feats)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            predictions = self.customizepredsandlosses(seg_logits=seg_logits, annotations=data_meta.getannotations(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size, auto_calc_loss=False)
            preds_to_tgts_mapping, annotations = None, data_meta.getannotations()
            if hasattr(self, 'se_layer'):
                predictions.update({'loss_se': seg_logits_se})
                annotations['seg_targets_onehot'] = self.onehot(annotations['seg_targets'], self.cfg['num_classes'])
                preds_to_tgts_mapping = {'loss_aux': 'seg_targets', 'loss_se': 'seg_targets_onehot', 'loss_cls': 'seg_targets'}
            loss, losses_log_dict = calculatelosses(predictions=predictions, annotations=annotations, losses_cfg=self.cfg['losses'], preds_to_tgts_mapping=preds_to_tgts_mapping, pixel_sampler=self.pixel_sampler)
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs
    """onehot"""

    def onehot(self, labels, num_classes):
        batch_size = labels.size(0)
        labels_onehot = labels.new_zeros((batch_size, num_classes))
        for i in range(batch_size):
            hist = labels[i].float().histc(bins=num_classes, min=0, max=num_classes - 1)
            labels_onehot[i] = hist > 0
        return labels_onehot


class JPU(nn.Module):

    def __init__(self, in_channels_list=(512, 1024, 2048), mid_channels=512, start_level=0, end_level=-1, dilations=(1, 2, 4, 8), align_corners=False, norm_cfg=None, act_cfg=None):
        super(JPU, self).__init__()
        self.in_channels_list = in_channels_list
        self.mid_channels = mid_channels
        self.start_level = start_level
        self.num_ins = len(in_channels_list)
        if end_level == -1:
            self.backbone_end_level = self.num_ins
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels_list)
        self.dilations = dilations
        self.align_corners = align_corners
        self.conv_layers = nn.ModuleList()
        self.dilation_layers = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            conv_layer = nn.Sequential(nn.Conv2d(self.in_channels_list[i], self.mid_channels, kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=self.mid_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
            self.conv_layers.append(conv_layer)
        for idx in range(len(dilations)):
            dilation_layer = DepthwiseSeparableConv2d(in_channels=(self.backbone_end_level - self.start_level) * self.mid_channels, out_channels=self.mid_channels, kernel_size=3, stride=1, padding=dilations[idx], dilation=dilations[idx], dw_norm_cfg=norm_cfg, dw_act_cfg=None, pw_norm_cfg=norm_cfg, pw_act_cfg=act_cfg)
            self.dilation_layers.append(dilation_layer)
    """forward"""

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels_list)
        feats = [self.conv_layers[idx - self.start_level](inputs[idx]) for idx in range(self.start_level, self.backbone_end_level)]
        h, w = feats[0].shape[2:]
        for idx in range(1, len(feats)):
            feats[idx] = F.interpolate(feats[idx], size=(h, w), mode='bilinear', align_corners=self.align_corners)
        feat = torch.cat(feats, dim=1)
        concat_feat = torch.cat([self.dilation_layers[idx](feat) for idx in range(len(self.dilations))], dim=1)
        outs = []
        for i in range(self.start_level, self.backbone_end_level - 1):
            outs.append(inputs[i])
        outs.append(concat_feat)
        return tuple(outs)


class FCN(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(FCN, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        convs = []
        for idx in range(head_cfg.get('num_convs', 2)):
            if idx == 0:
                conv = nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False)
            else:
                conv = nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False)
            norm = BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)
            act = BuildActivation(act_cfg)
            convs += [conv, norm, act]
        convs.append(nn.Dropout2d(head_cfg['dropout']))
        if head_cfg.get('num_convs', 2) > 0:
            convs.append(nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        else:
            convs.append(nn.Conv2d(head_cfg['in_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.decoder = nn.Sequential(*convs)
        self.setauxiliarydecoder(cfg['auxiliary'])
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        seg_logits = self.decoder(backbone_outputs[-1])
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            loss, losses_log_dict = self.customizepredsandlosses(seg_logits=seg_logits, annotations=data_meta.getannotations(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size)
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs


class DepthwiseSeparableFCN(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(DepthwiseSeparableFCN, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        convs = []
        for idx in range(head_cfg.get('num_convs', 2)):
            if idx == 0:
                conv = DepthwiseSeparableConv2d(in_channels=head_cfg['in_channels'], out_channels=head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            else:
                conv = DepthwiseSeparableConv2d(in_channels=head_cfg['feats_channels'], out_channels=head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            convs.append(conv)
        convs.append(nn.Dropout2d(head_cfg['dropout']))
        if head_cfg.get('num_convs', 2) > 0:
            convs.append(nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        else:
            convs.append(nn.Conv2d(head_cfg['in_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.decoder = nn.Sequential(*convs)
        self.setauxiliarydecoder(cfg['auxiliary'])
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        seg_logits = self.decoder(backbone_outputs[-1])
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            loss, losses_log_dict = self.customizepredsandlosses(seg_logits=seg_logits, annotations=data_meta.getannotations(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size)
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs


class ContextBlock(nn.Module):

    def __init__(self, in_channels, ratio, pooling_type='att', fusion_types=('channel_add',), norm_cfg=None, act_cfg=None):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([(f in valid_fusion_types) for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.in_channels = in_channels
        self.ratio = ratio
        self.planes = int(in_channels * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(nn.Conv2d(self.in_channels, self.planes, kernel_size=1, stride=1, padding=0), BuildNormalization(placeholder=[self.planes, 1, 1], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Conv2d(self.planes, self.in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(nn.Conv2d(self.in_channels, self.planes, kernel_size=1, stride=1, padding=0), BuildNormalization(placeholder=[self.planes, 1, 1], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Conv2d(self.planes, self.in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.channel_mul_conv = None
    """spatialpool"""

    def spatialpool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            input_x = input_x.view(batch, channel, height * width)
            input_x = input_x.unsqueeze(1)
            context_mask = self.conv_mask(x)
            context_mask = context_mask.view(batch, 1, height * width)
            context_mask = self.softmax(context_mask)
            context_mask = context_mask.unsqueeze(-1)
            context = torch.matmul(input_x, context_mask)
            context = context.view(batch, channel, 1, 1)
        else:
            context = self.avg_pool(x)
        return context
    """forward"""

    def forward(self, x):
        context = self.spatialpool(x)
        out = x
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out


class GCNet(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(GCNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        self.conv_before_cb = nn.Sequential(nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.contextblock_net = ContextBlock(in_channels=head_cfg['feats_channels'], ratio=head_cfg['ratio'], pooling_type=head_cfg['pooling_type'], fusion_types=head_cfg['fusion_types'], norm_cfg=head_cfg.get('norm_cfg', copy.deepcopy(norm_cfg)), act_cfg=head_cfg.get('act_cfg', copy.deepcopy(act_cfg)))
        self.conv_after_cb = nn.Sequential(nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.decoder = nn.Sequential(nn.Conv2d(head_cfg['in_channels'] + head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.setauxiliarydecoder(cfg['auxiliary'])
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        feats = self.conv_before_cb(backbone_outputs[-1])
        feats = self.contextblock_net(feats)
        feats = self.conv_after_cb(feats)
        feats = torch.cat([backbone_outputs[-1], feats], dim=1)
        seg_logits = self.decoder(feats)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            loss, losses_log_dict = self.customizepredsandlosses(seg_logits=seg_logits, annotations=data_meta.getannotations(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size)
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs


class CascadeFeatureFusion(nn.Module):

    def __init__(self, low_channels, high_channels, out_channels, norm_cfg=None, act_cfg=None, align_corners=False):
        super(CascadeFeatureFusion, self).__init__()
        self.align_corners = align_corners
        self.conv_low = nn.Sequential(nn.Conv2d(low_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.conv_high = nn.Sequential(nn.Conv2d(high_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
    """forward"""

    def forward(self, x_low, x_high):
        x_low = F.interpolate(x_low, size=x_high.shape[2:], mode='bilinear', align_corners=self.align_corners)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        x = F.relu(x_low + x_high, inplace=True)
        return x, x_low


class ICNeck(nn.Module):

    def __init__(self, in_channels_list=(64, 256, 256), out_channels=128, norm_cfg=None, act_cfg=None, align_corners=False):
        super(ICNeck, self).__init__()
        assert len(in_channels_list) == 3, 'in_channels_list should be equal to 3'
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.cff_24 = CascadeFeatureFusion(low_channels=in_channels_list[2], high_channels=in_channels_list[1], out_channels=out_channels, norm_cfg=norm_cfg, act_cfg=act_cfg, align_corners=align_corners)
        self.cff_12 = CascadeFeatureFusion(low_channels=out_channels, high_channels=in_channels_list[0], out_channels=out_channels, norm_cfg=norm_cfg, act_cfg=act_cfg, align_corners=align_corners)
    """forward"""

    def forward(self, inputs):
        assert len(inputs) == 3, 'length of input feature maps must be 3'
        x_sub1, x_sub2, x_sub4 = inputs
        x_cff_24, x_24 = self.cff_24(x_sub4, x_sub2)
        x_cff_12, x_12 = self.cff_12(x_cff_24, x_sub1)
        return x_24, x_12, x_cff_12


class ICNetEncoder(nn.Module):

    def __init__(self, backbone_cfg=None, in_channels=3, layer_channels_list=(512, 2048), light_branch_middle_channels=32, psp_out_channels=512, out_channels_list=(64, 256, 256), pool_scales=(1, 2, 3, 6), norm_cfg=None, act_cfg=None, align_corners=False):
        super(ICNetEncoder, self).__init__()
        self.align_corners = align_corners
        assert backbone_cfg is not None and isinstance(backbone_cfg, dict)
        if 'norm_cfg' not in backbone_cfg:
            backbone_cfg.update({'norm_cfg': copy.deepcopy(norm_cfg)})
        self.backbone_net = BuildBackbone(backbone_cfg)
        self.backbone_net.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.ppm_net = PyramidPoolingModule(pool_scales=pool_scales, in_channels=layer_channels_list[1], out_channels=psp_out_channels, norm_cfg=norm_cfg, act_cfg=act_cfg, align_corners=align_corners)
        self.conv_sub1 = nn.Sequential(nn.Conv2d(in_channels, light_branch_middle_channels, kernel_size=3, stride=2, padding=1, bias=False), BuildNormalization(placeholder=light_branch_middle_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Conv2d(light_branch_middle_channels, light_branch_middle_channels, kernel_size=3, stride=2, padding=1, bias=False), BuildNormalization(placeholder=light_branch_middle_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Conv2d(light_branch_middle_channels, out_channels_list[0], kernel_size=3, stride=2, padding=1, bias=False), BuildNormalization(placeholder=out_channels_list[0], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.conv_sub2 = nn.Sequential(nn.Conv2d(layer_channels_list[0], out_channels_list[1], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=out_channels_list[1], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.conv_sub4 = nn.Sequential(nn.Conv2d(psp_out_channels, out_channels_list[2], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=out_channels_list[2], norm_cfg=norm_cfg), BuildActivation(act_cfg))
    """forward"""

    def forward(self, x):
        output = []
        output.append(self.conv_sub1(x))
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=self.align_corners)
        x = self.backbone_net.stem(x)
        x = self.backbone_net.maxpool(x)
        x = self.backbone_net.layer1(x)
        x = self.backbone_net.layer2(x)
        output.append(self.conv_sub2(x))
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=self.align_corners)
        x = self.backbone_net.layer3(x)
        x = self.backbone_net.layer4(x)
        ppm_out = self.ppm_net(x)
        output.append(self.conv_sub4(ppm_out))
        return output


class ICNet(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(ICNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        delattr(self, 'backbone_net')
        encoder_cfg = head_cfg['encoder']
        encoder_cfg.update({'backbone_cfg': cfg['backbone']})
        if 'act_cfg' not in encoder_cfg:
            encoder_cfg.update({'act_cfg': act_cfg})
        if 'norm_cfg' not in encoder_cfg:
            encoder_cfg.update({'norm_cfg': norm_cfg})
        if 'align_corners' not in encoder_cfg:
            encoder_cfg.update({'align_corners': align_corners})
        self.backbone_net = ICNetEncoder(**encoder_cfg)
        neck_cfg = {'in_channels_list': head_cfg['in_channels_list'], 'out_channels': head_cfg['feats_channels'], 'act_cfg': act_cfg.copy(), 'norm_cfg': norm_cfg.copy(), 'align_corners': align_corners}
        self.neck = ICNeck(**neck_cfg)
        self.decoder = nn.Sequential(nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.setauxiliarydecoder(cfg['auxiliary'])
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        backbone_outputs = self.neck(backbone_outputs)
        seg_logits = self.decoder(backbone_outputs[-1])
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            loss, losses_log_dict = self.customizepredsandlosses(seg_logits=seg_logits, annotations=data_meta.getannotations(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size)
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs


class IDRNet(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(IDRNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        self.bottleneck = nn.Sequential(nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        if 'coarse_context' in head_cfg:
            supported_coarse_contexts = {'aspp': ASPP, 'ppm': PyramidPoolingModule}
            coarse_context_cfg = {'in_channels': head_cfg['feats_channels'], 'out_channels': head_cfg['feats_channels'], 'align_corners': align_corners, 'norm_cfg': norm_cfg, 'act_cfg': act_cfg}
            coarse_context_cfg.update(head_cfg['coarse_context'])
            coarse_context_type = coarse_context_cfg.pop('type')
            if 'fpn' in head_cfg:
                coarse_context_cfg['out_channels'] = head_cfg['fpn']['feats_channels']
            self.coarse_context_module = supported_coarse_contexts[coarse_context_type](**coarse_context_cfg)
            if head_cfg['use_sa_on_coarsecontext_before']:
                self.coarsecontext_refiner_before = SelfAttentionBlock(key_in_channels=coarse_context_cfg['out_channels'], query_in_channels=coarse_context_cfg['out_channels'], transform_channels=head_cfg['refine_coarsecontext_channels'], out_channels=coarse_context_cfg['out_channels'], share_key_query=False, query_downsample=None, key_downsample=None, key_query_num_convs=2, value_out_num_convs=1, key_query_norm=True, value_out_norm=True, matmul_norm=True, with_out_project=True, norm_cfg=norm_cfg, act_cfg=act_cfg)
            elif head_cfg['use_sa_on_coarsecontext_after']:
                self.coarsecontext_refiner_after = SelfAttentionBlock(key_in_channels=coarse_context_cfg['out_channels'], query_in_channels=coarse_context_cfg['out_channels'], transform_channels=head_cfg['refine_coarsecontext_channels'], out_channels=coarse_context_cfg['out_channels'], share_key_query=False, query_downsample=None, key_downsample=None, key_query_num_convs=2, value_out_num_convs=1, key_query_norm=True, value_out_norm=True, matmul_norm=True, with_out_project=True, norm_cfg=norm_cfg, act_cfg=act_cfg)
        if 'fpn' in head_cfg:
            act_cfg_copy = copy.deepcopy(act_cfg)
            if 'inplace' in act_cfg_copy:
                act_cfg_copy['inplace'] = False
            self.lateral_convs = nn.ModuleList()
            for in_channels in head_cfg['fpn']['in_channels_list'][:-1]:
                self.lateral_convs.append(nn.Sequential(nn.Conv2d(in_channels, head_cfg['fpn']['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=head_cfg['fpn']['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg_copy)))
            self.fpn_convs = nn.ModuleList()
            for in_channels in ([head_cfg['fpn']['feats_channels']] * len(self.lateral_convs)):
                self.fpn_convs.append(nn.Sequential(nn.Conv2d(in_channels, head_cfg['fpn']['out_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['fpn']['out_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg_copy)))
        for name in ['class_relations_mean', 'class_relations_var']:
            value = nn.Parameter(torch.eye(cfg['num_classes']).float(), requires_grad=False)
            setattr(self, name, value)
        self.selected_classes_counter = nn.Parameter(torch.ones(cfg['num_classes']).float() * 1e-06, requires_grad=False)
        self.idcontext_refiner = SelfAttentionBlock(key_in_channels=head_cfg['feats_channels'] * 6, query_in_channels=head_cfg['feats_channels'] * 6, transform_channels=head_cfg['refine_idcontext_channels'], out_channels=head_cfg['feats_channels'], share_key_query=False, query_downsample=None, key_downsample=None, key_query_num_convs=2, value_out_num_convs=1, key_query_norm=True, value_out_norm=True, matmul_norm=True, with_out_project=True, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dl_cls_representations = nn.Parameter(torch.zeros(cfg['num_classes'], head_cfg['feats_channels']).float(), requires_grad=False)
        if hasattr(self, 'coarse_context_module') and 'fpn' in head_cfg and head_cfg['use_fpn_before']:
            decoder_stage1_in_channels = coarse_context_cfg['out_channels'] + head_cfg['fpn']['out_channels'] * 3
        else:
            decoder_stage1_in_channels = coarse_context_cfg['out_channels'] if 'coarse_context' in head_cfg else head_cfg['feats_channels']
        if head_cfg['force_stage1_use_oripr']:
            decoder_stage1_in_channels = head_cfg['feats_channels']
        if not hasattr(self, 'coarse_context_module'):
            decoder_stage2_in_channels = head_cfg['feats_channels'] * 2
        elif hasattr(self, 'coarse_context_module') and 'fpn' not in head_cfg:
            decoder_stage2_in_channels = head_cfg['feats_channels'] * 2 + coarse_context_cfg['out_channels']
        elif hasattr(self, 'coarse_context_module') and 'fpn' in head_cfg:
            decoder_stage2_in_channels = head_cfg['feats_channels'] * 2 + coarse_context_cfg['out_channels'] + head_cfg['fpn']['out_channels'] * 3
        for name, in_channels in [('decoder_stage1', decoder_stage1_in_channels), ('decoder_stage2', decoder_stage2_in_channels)]:
            value = nn.Sequential(nn.Conv2d(in_channels, head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
            setattr(self, name, value)
        self.setauxiliarydecoder(cfg['auxiliary'])
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        seed = random.randint(1, 1e+16)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        feats, coarse_context = self.bottleneck(backbone_outputs[-1]), None
        if hasattr(self, 'coarse_context_module'):
            coarse_context = self.coarse_context_module(feats)
            if hasattr(self, 'coarsecontext_refiner_before'):
                assert not hasattr(self, 'coarsecontext_refiner_after')
                coarse_context = self.coarsecontext_refiner_before(coarse_context, coarse_context)
        if hasattr(self, 'fpn_convs') and self.cfg['head']['use_fpn_before']:
            assert not self.cfg['head']['use_fpn_after']
            assert coarse_context is not None, 'upernet setting error'
            inputs = backbone_outputs[:-1]
            lateral_outputs = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
            lateral_outputs.append(coarse_context)
            for i in range(len(lateral_outputs) - 1, 0, -1):
                prev_shape = lateral_outputs[i - 1].shape[2:]
                lateral_outputs[i - 1] = lateral_outputs[i - 1] + F.interpolate(lateral_outputs[i], size=prev_shape, mode='bilinear', align_corners=self.align_corners)
            fpn_outputs = [self.fpn_convs[i](lateral_outputs[i]) for i in range(len(lateral_outputs) - 1)]
            fpn_outputs.append(lateral_outputs[-1])
            fpn_outputs = [F.interpolate(out, size=fpn_outputs[0].size()[2:], mode='bilinear', align_corners=self.align_corners) for out in fpn_outputs]
            coarse_context = torch.cat(fpn_outputs, dim=1)
        if self.cfg['head']['force_stage1_use_oripr']:
            preds_stage1 = self.decoder_stage1(feats)
        else:
            preds_stage1 = self.decoder_stage1(feats if coarse_context is None else coarse_context)
        if preds_stage1.shape[2:] != feats.shape[2:]:
            preds_stage1 = F.interpolate(preds_stage1, size=feats.shape[2:], mode='bilinear', align_corners=self.align_corners)
        if hasattr(self, 'coarse_context_module') and hasattr(self, 'coarsecontext_refiner_after'):
            assert not hasattr(self, 'coarsecontext_refiner_before')
            coarse_context = self.coarsecontext_refiner_after(coarse_context, coarse_context)
        feats_withdl = self.insertdlrepresentations(feats, preds_stage1)
        id_context_mean, valid_clsids_batch = self.obtainidcontext(feats_withdl, preds_stage1, self.class_relations_mean)
        id_context_var, _ = self.obtainidcontext(feats_withdl, preds_stage1, self.class_relations_var, None, False)
        id_context = self.idcontext_refiner(torch.cat([feats_withdl, id_context_mean, id_context_var], dim=1), torch.cat([feats_withdl, id_context_mean, id_context_var], dim=1))
        if hasattr(self, 'fpn_convs') and self.cfg['head']['use_fpn_after']:
            assert not self.cfg['head']['use_fpn_before']
            assert coarse_context is not None, 'upernet setting error'
            inputs = backbone_outputs[:-1]
            lateral_outputs = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
            lateral_outputs.append(coarse_context)
            for i in range(len(lateral_outputs) - 1, 0, -1):
                prev_shape = lateral_outputs[i - 1].shape[2:]
                lateral_outputs[i - 1] = lateral_outputs[i - 1] + F.interpolate(lateral_outputs[i], size=prev_shape, mode='bilinear', align_corners=self.align_corners)
            fpn_outputs = [self.fpn_convs[i](lateral_outputs[i]) for i in range(len(lateral_outputs) - 1)]
            fpn_outputs.append(lateral_outputs[-1])
            fpn_outputs = [F.interpolate(out, size=fpn_outputs[0].size()[2:], mode='bilinear', align_corners=self.align_corners) for out in fpn_outputs]
            coarse_context = torch.cat(fpn_outputs, dim=1)
        torch.manual_seed(seed)
        if coarse_context is not None and feats.shape[2:] != coarse_context.shape[2:]:
            preds_stage2 = self.decoder_stage2(torch.cat([feats, id_context] if coarse_context is None else [F.interpolate(feats, size=coarse_context.size()[2:], mode='bilinear', align_corners=self.align_corners), F.interpolate(id_context, size=coarse_context.size()[2:], mode='bilinear', align_corners=self.align_corners), coarse_context], dim=1))
        else:
            preds_stage2 = self.decoder_stage2(torch.cat([feats, id_context] if coarse_context is None else [feats, id_context, coarse_context], dim=1))
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            with torch.no_grad():
                intervention_clsids = []
                for batch_idx in range(feats.shape[0]):
                    valid_clsids = valid_clsids_batch[batch_idx]
                    choice_weights = []
                    for intervention_clsid in valid_clsids:
                        choice_weights.append(1.0 / float(self.selected_classes_counter.data[intervention_clsid].item()))
                    choice_weights = np.array(choice_weights) / sum(choice_weights)
                    intervention_clsid = random.choices(valid_clsids, weights=choice_weights, k=1)[0]
                    intervention_clsids.append(intervention_clsid)
                    self.selected_classes_counter.data[intervention_clsid] = self.selected_classes_counter.data[intervention_clsid] + 1.0
                momentum = self.cfg['head']['clsrelation_momentum']
                id_context_mean, _ = self.obtainidcontext(feats_withdl, preds_stage1, self.class_relations_mean, intervention_clsids)
                id_context_var, _ = self.obtainidcontext(feats_withdl, preds_stage1, self.class_relations_var, intervention_clsids, False)
                id_context = self.idcontext_refiner(torch.cat([feats_withdl, id_context_mean, id_context_var], dim=1), torch.cat([feats_withdl, id_context_mean, id_context_var], dim=1))
                torch.manual_seed(seed)
                if coarse_context is not None and feats.shape[2:] != coarse_context.shape[2:]:
                    preds_intervention_stage2 = self.decoder_stage2(torch.cat([feats, id_context] if coarse_context is None else [F.interpolate(feats, size=coarse_context.size()[2:], mode='bilinear', align_corners=self.align_corners), F.interpolate(id_context, size=coarse_context.size()[2:], mode='bilinear', align_corners=self.align_corners), coarse_context], dim=1))
                else:
                    preds_intervention_stage2 = self.decoder_stage2(torch.cat([feats, id_context] if coarse_context is None else [feats, id_context, coarse_context], dim=1))
                preds_intervention_stage2 = F.interpolate(preds_intervention_stage2, size=img_size, mode='bilinear', align_corners=self.align_corners)
                preds_intervention_stage2 = preds_intervention_stage2.permute(0, 2, 3, 1).contiguous()
                preds_anchor_stage2 = F.interpolate(preds_stage2, size=img_size, mode='bilinear', align_corners=self.align_corners)
                preds_anchor_stage2 = preds_anchor_stage2.permute(0, 2, 3, 1).contiguous()
                for batch_idx in range(feats.shape[0]):
                    gts_iter = data_meta.getannotations()['seg_targets'][batch_idx]
                    clsids = data_meta.getannotations()['seg_targets'][batch_idx].unique()
                    logits_intervention_stage2_iter, logits_anchor_stage2_iter = preds_intervention_stage2[batch_idx], preds_anchor_stage2[batch_idx]
                    for clsid in clsids:
                        clsid = int(clsid.item())
                        if clsid == self.cfg['head']['ignore_index']:
                            continue
                        gts_iter_cls = gts_iter[gts_iter == clsid].long()
                        loss_intervention_stage2 = F.cross_entropy(logits_intervention_stage2_iter[gts_iter == clsid], gts_iter_cls, reduction='none')
                        loss_anchor_stage2 = F.cross_entropy(logits_anchor_stage2_iter[gts_iter == clsid], gts_iter_cls, reduction='none')
                        relation_mean_stage2 = loss_intervention_stage2.mean() - loss_anchor_stage2.mean()
                        self.class_relations_mean.data[intervention_clsids[batch_idx], clsid] = relation_mean_stage2 * momentum + self.class_relations_mean.data[intervention_clsids[batch_idx], clsid] * (1 - momentum)
                        if loss_anchor_stage2.shape[0] > 1:
                            relation_var_stage2 = loss_intervention_stage2.var(unbiased=False) - loss_anchor_stage2.var(unbiased=False)
                            self.class_relations_var.data[intervention_clsids[batch_idx], clsid] = relation_var_stage2 * momentum + self.class_relations_var.data[intervention_clsids[batch_idx], clsid] * (1 - momentum)
                if dist.is_available() and dist.is_initialized():
                    syn_list = ['class_relations_mean', 'class_relations_var', 'selected_classes_counter']
                    for syn in syn_list:
                        attr = getattr(self, syn).data.clone()
                        dist.all_reduce(attr.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
                        setattr(self, syn, nn.Parameter(attr, requires_grad=False))
            momentum = self.cfg['head']['dlclsreps_momentum']
            self.updatedlclsreps(feats, data_meta.getannotations()['seg_targets'], momentum, img_size)
            predictions = self.customizepredsandlosses(seg_logits=preds_stage2, annotations=data_meta.getannotations(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size, auto_calc_loss=False)
            preds_stage2 = predictions.pop('loss_cls')
            preds_stage1 = F.interpolate(preds_stage1, size=img_size, mode='bilinear', align_corners=self.align_corners)
            predictions.update({'loss_cls_stage1': preds_stage1, 'loss_cls_stage2': preds_stage2})
            loss, losses_log_dict = calculatelosses(predictions=predictions, annotations=data_meta.getannotations(), losses_cfg=self.cfg['losses'], pixel_sampler=self.pixel_sampler)
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=preds_stage2)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=preds_stage2)
        return ssseg_outputs
    """insertdlrepresentations"""

    def insertdlrepresentations(self, feats, logits):
        dl_cls_representations = self.dl_cls_representations.data.type_as(feats).clone()
        feats = feats.permute(0, 2, 3, 1).contiguous()
        logits = logits.permute(0, 2, 3, 1).contiguous()
        logits_argmax = logits.argmax(-1)
        feats_withdl = torch.zeros(feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3] * 2).type_as(feats)
        for cls_id in range(self.cfg['num_classes']):
            mask = logits_argmax == cls_id
            if mask.sum() == 0:
                continue
            feats_withdl[mask] = torch.cat([feats[mask], dl_cls_representations[cls_id].unsqueeze(0).expand_as(feats[mask])], dim=1)
        feats_withdl = feats_withdl.permute(0, 3, 1, 2).contiguous()
        return feats_withdl
    """obtainidcontext"""

    def obtainidcontext(self, context, logits, class_relations, intervention_clsids=None, remove_negative_cls_relation=True):
        batch_size, num_channels, context_h, context_w = context.size()
        valid_clsids_batch, id_context_batch = [], torch.zeros_like(context)
        class_relations = class_relations.data.type_as(context).clone()
        for batch_idx in range(batch_size):
            cls_contexts, selected_class_relations = [], []
            context_iter, logits_iter = context[batch_idx], logits[batch_idx]
            context_iter, logits_iter = context_iter.reshape(num_channels, -1), logits_iter.reshape(self.cfg['num_classes'], -1)
            context_iter = context_iter.permute(1, 0).contiguous()
            logits_iter_argmax = logits_iter.argmax(0)
            valid_clsids = []
            for cls_id in range(self.cfg['num_classes']):
                if intervention_clsids is not None:
                    if cls_id == intervention_clsids[batch_idx]:
                        continue
                mask = logits_iter_argmax == cls_id
                if mask.sum() == 0:
                    continue
                context_iter_cls = context_iter[mask]
                logits_iter_cls = logits_iter[cls_id, :][mask]
                weight = F.softmax(logits_iter_cls, dim=0)
                context_iter_cls = context_iter_cls * weight.unsqueeze(-1)
                context_iter_cls = context_iter_cls.sum(0)
                valid_clsids.append(cls_id)
                cls_contexts.append(context_iter_cls)
                selected_class_relations.append(class_relations[:, cls_id].unsqueeze(1))
            if len(cls_contexts) != 0:
                valid_clsids_batch.append(valid_clsids)
                cls_contexts = torch.stack(cls_contexts)
                selected_class_relations = torch.cat(selected_class_relations, dim=1)
                if remove_negative_cls_relation:
                    selected_class_relations[selected_class_relations <= 0] = -1e+16
                selected_class_relations = F.softmax(selected_class_relations, dim=1)
                selected_class_relations_tmp = []
                for cls_id in valid_clsids:
                    selected_class_relations_tmp.append(selected_class_relations[cls_id, :])
                selected_class_relations = torch.stack(selected_class_relations_tmp)
                id_context_tmp = torch.matmul(selected_class_relations, cls_contexts)
                id_context = torch.zeros(context_h * context_w, num_channels).type_as(context)
                for idx, cls_id in enumerate(valid_clsids):
                    mask = logits_iter_argmax == cls_id
                    assert mask.sum() > 0, 'mask assert error, bug exists'
                    id_context[mask] = id_context_tmp[idx]
                id_context = id_context.permute(1, 0).contiguous()
                id_context = id_context.reshape(num_channels, context_h, context_w)
                id_context_batch[batch_idx] = id_context
        return id_context_batch, valid_clsids_batch
    """updatedlclsreps"""

    def updatedlclsreps(self, feats, gts, momentum, img_size):
        with torch.no_grad():
            feats = F.interpolate(feats, size=img_size, mode='bilinear', align_corners=self.align_corners)
            feats = feats.permute(0, 2, 3, 1).contiguous()
            unique_cls_ids = gts.unique()
            for cls_id in unique_cls_ids:
                cls_id = int(cls_id.item())
                if cls_id == self.cfg['head']['ignore_index']:
                    continue
                feats_cls = feats[gts == cls_id].mean(0)
                self.dl_cls_representations.data[cls_id, :] = feats_cls * momentum + self.dl_cls_representations[cls_id, :].clone() * (1 - momentum)
            if dist.is_available() and dist.is_initialized():
                dl_cls_representations = self.dl_cls_representations.data.clone()
                dist.all_reduce(dl_cls_representations.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
                self.dl_cls_representations = nn.Parameter(dl_cls_representations, requires_grad=False)


class ISANet(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(ISANet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        self.down_factor = head_cfg['down_factor']
        self.in_conv = nn.Sequential(nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.global_relation = SelfAttentionBlock(in_channels=head_cfg['feats_channels'], feats_channels=head_cfg['isa_channels'], norm_cfg=copy.deepcopy(norm_cfg), act_cfg=copy.deepcopy(act_cfg))
        self.local_relation = SelfAttentionBlock(in_channels=head_cfg['feats_channels'], feats_channels=head_cfg['isa_channels'], norm_cfg=copy.deepcopy(norm_cfg), act_cfg=copy.deepcopy(act_cfg))
        self.out_conv = nn.Sequential(nn.Conv2d(head_cfg['feats_channels'] * 2, head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.decoder = nn.Sequential(nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.setauxiliarydecoder(cfg['auxiliary'])
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        feats = self.in_conv(backbone_outputs[-1])
        residual = feats
        n, c, h, w = feats.size()
        loc_h, loc_w = self.down_factor
        glb_h, glb_w = math.ceil(h / loc_h), math.ceil(w / loc_w)
        pad_h, pad_w = glb_h * loc_h - h, glb_w * loc_w - w
        if pad_h > 0 or pad_w > 0:
            padding = pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2
            feats = F.pad(feats, padding)
        feats = feats.view(n, c, glb_h, loc_h, glb_w, loc_w)
        feats = feats.permute(0, 3, 5, 1, 2, 4)
        feats = feats.reshape(-1, c, glb_h, glb_w)
        feats = self.global_relation(feats)
        feats = feats.view(n, loc_h, loc_w, c, glb_h, glb_w)
        feats = feats.permute(0, 4, 5, 3, 1, 2)
        feats = feats.reshape(-1, c, loc_h, loc_w)
        feats = self.local_relation(feats)
        feats = feats.view(n, glb_h, glb_w, c, loc_h, loc_w)
        feats = feats.permute(0, 3, 1, 4, 2, 5)
        feats = feats.reshape(n, c, glb_h * loc_h, glb_w * loc_w)
        if pad_h > 0 or pad_w > 0:
            feats = feats[:, :, pad_h // 2:pad_h // 2 + h, pad_w // 2:pad_w // 2 + w]
        feats = self.out_conv(torch.cat([feats, residual], dim=1))
        seg_logits = self.decoder(feats)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            loss, losses_log_dict = self.customizepredsandlosses(seg_logits=seg_logits, annotations=data_meta.getannotations(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size)
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs


class ImageLevelContext(nn.Module):

    def __init__(self, feats_channels, transform_channels, concat_input=False, align_corners=False, norm_cfg=None, act_cfg=None):
        super(ImageLevelContext, self).__init__()
        self.align_corners = align_corners
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.correlate_net = SelfAttentionBlock(key_in_channels=feats_channels * 2, query_in_channels=feats_channels, transform_channels=transform_channels, out_channels=feats_channels, share_key_query=False, query_downsample=None, key_downsample=None, key_query_num_convs=2, value_out_num_convs=1, key_query_norm=True, value_out_norm=True, matmul_norm=True, with_out_project=True, norm_cfg=norm_cfg, act_cfg=act_cfg)
        if concat_input:
            self.bottleneck = nn.Sequential(nn.Conv2d(feats_channels * 2, feats_channels, kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=feats_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
    """forward"""

    def forward(self, x):
        x_global = self.global_avgpool(x)
        x_global = F.interpolate(x_global, size=x.size()[2:], mode='bilinear', align_corners=self.align_corners)
        feats_il = self.correlate_net(x, torch.cat([x_global, x], dim=1))
        if hasattr(self, 'bottleneck'):
            feats_il = self.bottleneck(torch.cat([x, feats_il], dim=1))
        return feats_il


class SemanticLevelContext(nn.Module):

    def __init__(self, feats_channels, transform_channels, concat_input=False, norm_cfg=None, act_cfg=None):
        super(SemanticLevelContext, self).__init__()
        self.correlate_net = SelfAttentionBlock(key_in_channels=feats_channels, query_in_channels=feats_channels, transform_channels=transform_channels, out_channels=feats_channels, share_key_query=False, query_downsample=None, key_downsample=None, key_query_num_convs=2, value_out_num_convs=1, key_query_norm=True, value_out_norm=True, matmul_norm=True, with_out_project=True, norm_cfg=norm_cfg, act_cfg=act_cfg)
        if concat_input:
            self.bottleneck = nn.Sequential(nn.Conv2d(feats_channels * 2, feats_channels, kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=feats_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
    """forward"""

    def forward(self, x, preds, feats_il):
        inputs = x
        batch_size, num_channels, h, w = x.size()
        num_classes = preds.size(1)
        feats_sl = torch.zeros(batch_size, h * w, num_channels).type_as(x)
        for batch_idx in range(batch_size):
            feats_iter, preds_iter = x[batch_idx], preds[batch_idx]
            feats_iter, preds_iter = feats_iter.reshape(num_channels, -1), preds_iter.reshape(num_classes, -1)
            feats_iter, preds_iter = feats_iter.permute(1, 0), preds_iter.permute(1, 0)
            argmax = preds_iter.argmax(1)
            for clsid in range(num_classes):
                mask = argmax == clsid
                if mask.sum() == 0:
                    continue
                feats_iter_cls = feats_iter[mask]
                preds_iter_cls = preds_iter[:, clsid][mask]
                weight = F.softmax(preds_iter_cls, dim=0)
                feats_iter_cls = feats_iter_cls * weight.unsqueeze(-1)
                feats_iter_cls = feats_iter_cls.sum(0)
                feats_sl[batch_idx][mask] = feats_iter_cls
        feats_sl = feats_sl.reshape(batch_size, h, w, num_channels)
        feats_sl = feats_sl.permute(0, 3, 1, 2).contiguous()
        feats_sl = self.correlate_net(inputs, feats_sl)
        if hasattr(self, 'bottleneck'):
            feats_sl = self.bottleneck(torch.cat([feats_il, feats_sl], dim=1))
        return feats_sl


class ISNet(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(ISNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        self.bottleneck = nn.Sequential(nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        ilc_cfg = {'feats_channels': head_cfg['feats_channels'], 'transform_channels': head_cfg['transform_channels'], 'concat_input': head_cfg['concat_input'], 'norm_cfg': copy.deepcopy(norm_cfg), 'act_cfg': copy.deepcopy(act_cfg), 'align_corners': align_corners}
        self.ilc_net = ImageLevelContext(**ilc_cfg)
        slc_cfg = {'feats_channels': head_cfg['feats_channels'], 'transform_channels': head_cfg['transform_channels'], 'concat_input': head_cfg['concat_input'], 'norm_cfg': copy.deepcopy(norm_cfg), 'act_cfg': copy.deepcopy(act_cfg)}
        self.slc_net = SemanticLevelContext(**slc_cfg)
        self.decoder_stage1 = nn.Sequential(nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        if head_cfg['shortcut']['is_on']:
            self.shortcut = nn.Sequential(nn.Conv2d(head_cfg['shortcut']['in_channels'], head_cfg['shortcut']['feats_channels'], kernel_size=1, stride=1, padding=0), BuildNormalization(placeholder=head_cfg['shortcut']['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
            self.decoder_stage2 = nn.Sequential(nn.Conv2d(head_cfg['feats_channels'] + head_cfg['shortcut']['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        else:
            self.decoder_stage2 = nn.Sequential(nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.setauxiliarydecoder(cfg['auxiliary'])
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        feats = self.bottleneck(backbone_outputs[-1])
        feats_il = self.ilc_net(feats)
        preds_stage1 = self.decoder_stage1(feats)
        preds = preds_stage1
        if preds_stage1.size()[2:] != feats.size()[2:]:
            preds = F.interpolate(preds_stage1, size=feats.size()[2:], mode='bilinear', align_corners=self.align_corners)
        feats_sl = self.slc_net(feats, preds, feats_il)
        if hasattr(self, 'shortcut'):
            shortcut_out = self.shortcut(backbone_outputs[0])
            feats_sl = F.interpolate(feats_sl, size=shortcut_out.shape[2:], mode='bilinear', align_corners=self.align_corners)
            feats_sl = torch.cat([feats_sl, shortcut_out], dim=1)
        preds_stage2 = self.decoder_stage2(feats_sl)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            predictions = self.customizepredsandlosses(seg_logits=preds_stage2, annotations=data_meta.getannotations(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size, auto_calc_loss=False)
            preds_stage2 = predictions.pop('loss_cls')
            preds_stage1 = F.interpolate(preds_stage1, size=img_size, mode='bilinear', align_corners=self.align_corners)
            predictions.update({'loss_cls_stage1': preds_stage1, 'loss_cls_stage2': preds_stage2})
            loss, losses_log_dict = calculatelosses(predictions=predictions, annotations=data_meta.getannotations(), losses_cfg=self.cfg['losses'], pixel_sampler=self.pixel_sampler)
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=preds_stage2)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=preds_stage2)
        return ssseg_outputs


class LRASPPNet(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(LRASPPNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        self.branch_convs, self.branch_ups = nn.Sequential(), nn.Sequential()
        for idx, branch_channels in enumerate(head_cfg['branch_channels_list']):
            self.branch_convs.add_module(f'conv{idx}', nn.Conv2d(head_cfg['in_channels_list'][idx], branch_channels, kernel_size=1, stride=1, padding=0, bias=False))
            self.branch_ups.add_module(f'conv{idx}', nn.Sequential(nn.Conv2d(head_cfg['feats_channels'] + branch_channels, head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg)))
        self.aspp_conv = nn.Sequential(nn.Conv2d(head_cfg['in_channels_list'][-1], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.image_pool = nn.Sequential(nn.AvgPool2d(kernel_size=49, stride=(16, 20)), nn.Conv2d(head_cfg['in_channels_list'][-1], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), nn.Sigmoid())
        self.bottleneck = nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False)
        self.decoder = nn.Sequential(nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        feats = self.aspp_conv(backbone_outputs[-1]) * F.interpolate(self.image_pool(backbone_outputs[-1]), size=backbone_outputs[-1].size()[2:], mode='bilinear', align_corners=self.align_corners)
        feats = self.bottleneck(feats)
        for idx in range(len(self.cfg['head']['branch_channels_list']) - 1, -1, -1):
            feats = F.interpolate(feats, size=backbone_outputs[idx].size()[2:], mode='bilinear', align_corners=self.align_corners)
            feats = torch.cat([feats, self.branch_convs[idx](backbone_outputs[idx])], dim=1)
            feats = self.branch_ups[idx](feats)
        seg_logits = self.decoder(feats)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            loss, losses_log_dict = self.customizepredsandlosses(seg_logits=seg_logits, annotations=data_meta.getannotations(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size)
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs


class HungarianMatcher(nn.Module):

    def __init__(self, cost_class=1.0, cost_mask=1.0, cost_dice=1.0):
        super(HungarianMatcher, self).__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
    """memoryefficientforward"""

    @torch.no_grad()
    def memoryefficientforward(self, outputs, targets):
        bs, num_queries = outputs['pred_logits'].shape[:2]
        indices = []
        for b in range(bs):
            out_prob = outputs['pred_logits'][b].softmax(-1)
            out_mask = outputs['pred_masks'][b]
            tgt_ids, tgt_mask = targets[b]['labels'], targets[b]['masks']
            cost_class = -out_prob[:, tgt_ids]
            tgt_mask = F.interpolate(tgt_mask[:, None], size=out_mask.shape[-2:], mode='nearest')
            out_mask, tgt_mask = out_mask.flatten(1), tgt_mask[:, 0].flatten(1)
            cost_mask = self.sigmoidfocalloss(out_mask, tgt_mask)
            cost_dice = self.diceloss(out_mask, tgt_mask)
            C = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    """forward"""

    @torch.no_grad()
    def forward(self, outputs, targets):
        return self.memoryefficientforward(outputs, targets)
    """diceloss"""

    def diceloss(self, inputs, targets):
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        numerator = 2 * torch.einsum('nc,mc->nm', inputs, targets)
        denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss
    """sigmoidfocalloss"""

    def sigmoidfocalloss(self, inputs, targets, alpha=0.25, gamma=2.0):
        hw = inputs.shape[1]
        prob = inputs.sigmoid()
        focal_pos = (1 - prob) ** gamma * F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction='none')
        focal_neg = prob ** gamma * F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction='none')
        if alpha >= 0:
            focal_pos = focal_pos * alpha
            focal_neg = focal_neg * (1 - alpha)
        loss = torch.einsum('nc,mc->nm', focal_pos, targets) + torch.einsum('nc,mc->nm', focal_neg, 1 - targets)
        return loss / hw


class MSDeformAttn(nn.Module):

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super(MSDeformAttn, self).__init__()
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads)
        self.im2col_step = 128
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.resetparameters()
    """resetparameters"""

    def resetparameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)
    """forward"""

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError('Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        try:
            output = MSDeformAttnFunction.apply(value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        except:
            output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output


def getclones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MSDeformAttnTransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super(MSDeformAttnTransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.layers = getclones(encoder_layer, num_layers)
    """getreferencepoints"""

    @staticmethod
    def getreferencepoints(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device), torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
    """forward"""

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.getreferencepoints(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


class MSDeformAttnTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, act_cfg={'type': 'ReLU', 'inplace': True}, n_levels=4, n_heads=8, n_points=4):
        super(MSDeformAttnTransformerEncoderLayer, self).__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = BuildActivation(act_cfg)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
    """withposembed"""

    @staticmethod
    def withposembed(tensor, pos):
        return tensor if pos is None else tensor + pos
    """forwardffn"""

    def forwardffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src
    """forward"""

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        src2 = self.self_attn(self.withposembed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.forwardffn(src)
        return src


class MSDeformAttnTransformerEncoderOnly(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=1024, dropout=0.1, act_cfg={'type': 'ReLU', 'inplace': True}, num_feature_levels=4, enc_n_points=4):
        super(MSDeformAttnTransformerEncoderOnly, self).__init__()
        self.nhead = nhead
        self.d_model = d_model
        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward, dropout, act_cfg, num_feature_levels, nhead, enc_n_points)
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.resetparameters()
    """resetparameters"""

    def resetparameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m.resetparameters()
        nn.init.normal_(self.level_embed)
    """getvalidratio"""

    def getvalidratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    """forward"""

    def forward(self, srcs, pos_embeds):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes = [], [], [], []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = h, w
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.getvalidratio(m) for m in masks], 1)
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        return memory, spatial_shapes, level_start_index


class MSDeformAttnPixelDecoder(nn.Module):

    def __init__(self, input_shape, transformer_dropout, transformer_nheads, transformer_dim_feedforward, transformer_enc_layers, conv_dim, mask_dim, norm_cfg, act_cfg, transformer_in_features, common_stride):
        super(MSDeformAttnPixelDecoder, self).__init__()
        transformer_input_shape = {k: v for k, v in input_shape.items() if k in transformer_in_features}
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        self.feature_strides = [v.stride for k, v in input_shape]
        self.feature_channels = [v.channels for k, v in input_shape]
        transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
        self.transformer_in_features = [k for k, v in transformer_input_shape]
        transformer_in_channels = [v.channels for k, v in transformer_input_shape]
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape]
        self.transformer_num_feature_levels = len(self.transformer_in_features)
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(nn.Conv2d(in_channels, conv_dim, kernel_size=1), nn.GroupNorm(32, conv_dim)))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([nn.Sequential(nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1), nn.GroupNorm(32, conv_dim))])
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        self.transformer = MSDeformAttnTransformerEncoderOnly(d_model=conv_dim, dropout=transformer_dropout, nhead=transformer_nheads, dim_feedforward=transformer_dim_feedforward, num_encoder_layers=transformer_enc_layers, num_feature_levels=self.transformer_num_feature_levels)
        self.pe_layer = PositionEmbeddingSine(conv_dim // 2, apply_normalize=True)
        self.mask_dim = mask_dim
        self.mask_features = nn.Conv2d(conv_dim, mask_dim, kernel_size=1, stride=1, padding=0)
        weight_init.c2_xavier_fill(self.mask_features)
        self.maskformer_num_feature_levels = 3
        self.common_stride = common_stride
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))
        lateral_convs, output_convs = nn.ModuleList(), nn.ModuleList()
        use_bias = norm_cfg is None
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            lateral_conv = nn.Sequential(nn.Conv2d(in_channels, conv_dim, kernel_size=1, stride=1, padding=0, bias=use_bias), BuildNormalization(placeholder=conv_dim, norm_cfg=norm_cfg))
            output_conv = nn.Sequential(nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=use_bias), BuildNormalization(placeholder=conv_dim, norm_cfg=norm_cfg), BuildActivation(act_cfg=act_cfg))
            weight_init.c2_xavier_fill(lateral_conv[0])
            weight_init.c2_xavier_fill(output_conv[0])
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
    """forwardfeatures"""

    def forwardfeatures(self, features):
        srcs, pos = [], []
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))
        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = y.shape[0]
        split_size_or_sections = [None] * self.transformer_num_feature_levels
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)
        out, multi_scale_features, num_cur_levels = [], [], 0
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x = features[f].float()
            lateral_conv, output_conv = self.lateral_convs[idx], self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode='bilinear', align_corners=False)
            y = output_conv(y)
            out.append(y)
        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1
        return self.mask_features(out[-1]), out[0], multi_scale_features


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0, act_cfg={'type': 'ReLU', 'inplace': True}, normalize_before=False):
        super(CrossAttentionLayer, self).__init__()
        self.normalize_before = normalize_before
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = BuildActivation(act_cfg)
        self.resetparameters()
    """resetparameters"""

    def resetparameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    """withposembed"""

    def withposembed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    """forwardpost"""

    def forwardpost(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        tgt2 = self.multihead_attn(query=self.withposembed(tgt, query_pos), key=self.withposembed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt
    """forwardpre"""

    def forwardpre(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.withposembed(tgt2, query_pos), key=self.withposembed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt
    """forward"""

    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        if self.normalize_before:
            return self.forwardpre(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        return self.forwardpost(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0, act_cfg={'type': 'ReLU', 'inplace': True}, normalize_before=False):
        super(FFNLayer, self).__init__()
        self.normalize_before = normalize_before
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = BuildActivation(act_cfg)
        self.resetparameters()
    """resetparameters"""

    def resetparameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    """withposembed"""

    def withposembed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    """forwardpost"""

    def forwardpost(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt
    """forwardpre"""

    def forwardpre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt
    """forward"""

    def forward(self, tgt):
        if self.normalize_before:
            return self.forwardpre(tgt)
        return self.forwardpost(tgt)


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0, act_cfg={'type': 'ReLU', 'inplace': True}, normalize_before=False):
        super(SelfAttentionLayer, self).__init__()
        self.normalize_before = normalize_before
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = BuildActivation(act_cfg=act_cfg)
        self.resetparameters()
    """resetparameters"""

    def resetparameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    """withposembed"""

    def withposembed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    """forwardpost"""

    def forwardpost(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        q = k = self.withposembed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt
    """forwardpre"""

    def forwardpre(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        tgt2 = self.norm(tgt)
        q = k = self.withposembed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt
    """forward"""

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        if self.normalize_before:
            return self.forwardpre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forwardpost(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class MultiScaleMaskedTransformerDecoder(nn.Module):

    def __init__(self, in_channels, num_classes, hidden_dim, num_queries, nheads, dim_feedforward, dec_layers, pre_norm, mask_dim, enforce_input_project, mask_classification=True):
        super(MultiScaleMaskedTransformerDecoder, self).__init__()
        assert mask_classification, 'only support mask classification model'
        self.mask_classification = mask_classification
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, apply_normalize=True)
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(SelfAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm))
            self.transformer_cross_attention_layers.append(CrossAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm))
            self.transformer_ffn_layers.append(FFNLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm))
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.num_queries = num_queries
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
    """forward"""

    def forward(self, x, mask_features, mask=None):
        assert len(x) == self.num_feature_levels
        src, pos, size_list = [], [], []
        del mask
        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)
        _, bs, _ = src[0].shape
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        predictions_class, predictions_mask = [], []
        outputs_class, outputs_mask, attn_mask = self.forwardpredictionheads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            output = self.transformer_cross_attention_layers[i](output, src[level_index], memory_mask=attn_mask, memory_key_padding_mask=None, pos=pos[level_index], query_pos=query_embed)
            output = self.transformer_self_attention_layers[i](output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed)
            output = self.transformer_ffn_layers[i](output)
            outputs_class, outputs_mask, attn_mask = self.forwardpredictionheads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
        assert len(predictions_class) == self.num_layers + 1
        out = {'pred_logits': predictions_class[-1], 'pred_masks': predictions_mask[-1], 'aux_outputs': self.setauxloss(predictions_class if self.mask_classification else None, predictions_mask)}
        return out
    """forwardpredictionheads"""

    def forwardpredictionheads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_features)
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode='bilinear', align_corners=False)
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()
        return outputs_class, outputs_mask, attn_mask
    """setauxloss"""

    @torch.jit.unused
    def setauxloss(self, outputs_class, outputs_seg_masks):
        if self.mask_classification:
            return [{'pred_logits': a, 'pred_masks': b} for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]
        else:
            return [{'pred_masks': b} for b in outputs_seg_masks[:-1]]


class NestedTensor(object):

    def __init__(self, tensors, mask):
        self.mask = mask
        self.tensors = tensors
    """to"""

    def to(self, device):
        cast_tensor = self.tensors
        mask, cast_mask = self.mask, None
        if mask is not None:
            cast_mask = mask
        return NestedTensor(cast_tensor, cast_mask)
    """decompose"""

    def decompose(self):
        return self.tensors, self.mask


class SetCriterion(nn.Module):

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super(SetCriterion, self).__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
    """losslabels"""

    def losslabels(self, outputs, targets, indices, num_masks):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self.getsrcpermutationidx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {'loss_ce': loss_ce}
    """lossmasks"""

    def lossmasks(self, outputs, targets, indices, num_masks):
        assert 'pred_masks' in outputs
        src_idx = self.getsrcpermutationidx(indices)
        tgt_idx = self.gettgtpermutationidx(indices)
        src_masks = outputs['pred_masks']
        src_masks = src_masks[src_idx]
        masks = [t['masks'] for t in targets]
        target_masks, valid = self.nestedtensorfromtensorlist(masks).decompose()
        target_masks = target_masks
        target_masks = target_masks[tgt_idx]
        src_masks = F.interpolate(src_masks[:, None], size=target_masks.shape[-2:], mode='bilinear', align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)
        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {'loss_mask': self.sigmoidfocalloss(src_masks, target_masks, num_masks), 'loss_dice': self.diceloss(src_masks, target_masks, num_masks)}
        return losses
    """getsrcpermutationidx"""

    def getsrcpermutationidx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx
    """gettgtpermutationidx"""

    def gettgtpermutationidx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for _, tgt in indices])
        return batch_idx, tgt_idx
    """getloss"""

    def getloss(self, loss_type, outputs, targets, indices, num_masks):
        loss_map = {'labels': self.losslabels, 'masks': self.lossmasks}
        return loss_map[loss_type](outputs, targets, indices, num_masks)
    """formattargets"""

    def formattargets(self, seg, ignore_index=255, background_idx=0):
        labels, masks = [], []
        for label in torch.unique(seg):
            if int(label) == ignore_index:
                continue
            labels.append(label)
            masks.append((seg == label).unsqueeze(0))
        if not masks:
            masks = torch.zeros(1, seg.shape[0], seg.shape[1])
        else:
            masks = torch.cat(masks, dim=0).type_as(seg).long()
        if not labels:
            labels = [background_idx]
        labels = torch.tensor(labels).type_as(seg).long()
        return masks, labels
    """forward"""

    def forward(self, outputs, targets):
        segs = targets['seg_targets']
        batch_size, targets_format = segs.shape[0], []
        for idx in range(batch_size):
            masks, labels = self.formattargets(segs[idx])
            target_format = {'masks': masks, 'labels': labels}
            targets_format.append(target_format)
        targets = targets_format
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_masks = sum(len(t['labels']) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_masks, op=dist.ReduceOp.SUM)
        num_masks = torch.clamp(num_masks / self.getworldsize(), min=1).item()
        losses = {}
        for loss in self.losses:
            losses.update(self.getloss(loss, outputs, targets, indices, num_masks))
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.getloss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {(k + f'_{i}'): v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses
    """nestedtensorfromtensorlist"""

    def nestedtensorfromtensorlist(self, tensor_list):

        def maxbyaxis(the_list):
            maxes = the_list[0]
            for sublist in the_list[1:]:
                for index, item in enumerate(sublist):
                    maxes[index] = max(maxes[index], item)
            return maxes
        assert tensor_list[0].ndim == 3
        if torchvision._is_tracing():
            return self.onnxnestedtensorfromtensorlist(tensor_list)
        max_size = maxbyaxis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype, device = tensor_list[0].dtype, tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
            m[:img.shape[1], :img.shape[2]] = False
        return NestedTensor(tensor, mask)
    """onnxnestedtensorfromtensorlist"""

    @torch.jit.unused
    def onnxnestedtensorfromtensorlist(self, tensor_list):
        max_size = []
        for i in range(tensor_list[0].dim()):
            max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32))
            max_size.append(max_size_i)
        max_size, padded_imgs, padded_masks = tuple(max_size), [], []
        for img in tensor_list:
            padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
            padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
            padded_imgs.append(padded_img)
            m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
            padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), 'constant', 1)
            padded_masks.append(padded_mask)
        tensor, mask = torch.stack(padded_imgs), torch.stack(padded_masks)
        return NestedTensor(tensor, mask=mask)
    """getworldsize"""

    def getworldsize(self):
        if not dist.is_available() or not dist.is_initialized():
            return 1
        return dist.get_world_size()
    """diceloss"""

    def diceloss(self, inputs, targets, num_masks):
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_masks
    """sigmoidfocalloss"""

    def sigmoidfocalloss(self, inputs, targets, num_masks, alpha=0.25, gamma=2):
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * (1 - p_t) ** gamma
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean(1).sum() / num_masks


class ShapeSpec:

    def __init__(self, stride, channels):
        self.stride = stride
        self.channels = channels


class Mask2Former(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(Mask2Former, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        iterator = zip(head_cfg['pixel_decoder']['input_shape']['strides'], head_cfg['pixel_decoder']['input_shape']['in_channels'])
        assert len(head_cfg['pixel_decoder']['input_shape']['strides']) == 4
        head_cfg['pixel_decoder']['input_shape'] = {f'res{idx + 2}': ShapeSpec(stride, channels) for idx, (stride, channels) in enumerate(iterator)}
        self.pixel_decoder = MSDeformAttnPixelDecoder(**head_cfg['pixel_decoder'])
        predictor_cfg = copy.deepcopy(head_cfg['predictor'])
        predictor_cfg['dec_layers'] = predictor_cfg['dec_layers'] - 1
        self.predictor = MultiScaleMaskedTransformerDecoder(num_classes=cfg['num_classes'], **predictor_cfg)
        matcher = HungarianMatcher(**head_cfg['matcher'])
        weight_dict = {'loss_ce': head_cfg['matcher']['cost_class'], 'loss_mask': head_cfg['matcher']['cost_mask'], 'loss_dice': head_cfg['matcher']['cost_dice']}
        if head_cfg['deep_supervision']:
            dec_layers, aux_weight_dict = head_cfg['predictor']['dec_layers'], {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({(k + f'_{i}'): v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        self.criterion = SetCriterion(cfg['num_classes'], matcher=matcher, weight_dict=weight_dict, **head_cfg['criterion'])
        self.setauxiliarydecoder(cfg['auxiliary'])
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        from torch.cuda.amp import autocast
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        assert len(backbone_outputs) == 4
        features = {'res2': backbone_outputs[0], 'res3': backbone_outputs[1], 'res4': backbone_outputs[2], 'res5': backbone_outputs[3]}
        with autocast(enabled=False):
            mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forwardfeatures(features)
        predictions = self.predictor(multi_scale_features, mask_features, None)
        ssseg_outputs = SSSegOutputStructure(mode=self.mode, auto_validate=False)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            losses_dict = self.criterion(predictions, data_meta.getannotations())
            for k in list(losses_dict.keys()):
                if k in self.criterion.weight_dict:
                    losses_dict[k] *= self.criterion.weight_dict[k]
                else:
                    losses_dict.pop(k)
            loss, losses_log_dict = 0, {}
            for loss_key, loss_value in losses_dict.items():
                loss_value = loss_value.mean()
                loss = loss + loss_value
                loss_value = loss_value.data.clone()
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(loss_value.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
                losses_log_dict[loss_key] = loss_value.item()
            losses_log_dict.update({'loss_total': sum(losses_log_dict.values())})
            ssseg_outputs.setvariable('loss', loss)
            ssseg_outputs.setvariable('losses_log_dict', losses_log_dict)
            if self.mode in ['TRAIN']:
                return ssseg_outputs
        mask_cls_results = predictions['pred_logits']
        mask_pred_results = predictions['pred_masks']
        mask_pred_results = F.interpolate(mask_pred_results, size=img_size, mode='bilinear', align_corners=self.align_corners)
        predictions = []
        for mask_cls, mask_pred in zip(mask_cls_results, mask_pred_results):
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum('qc,qhw->chw', mask_cls, mask_pred)
            predictions.append(semseg.unsqueeze(0))
        seg_logits = torch.cat(predictions, dim=0)
        ssseg_outputs.setvariable('seg_logits', seg_logits)
        return ssseg_outputs


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super(TransformerDecoder, self).__init__()
        self.norm = norm
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
    """forward"""

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        output, intermediate = tgt, []
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, norm_cfg=None, act_cfg=None, norm_before=False):
        super(TransformerDecoderLayer, self).__init__()
        self.norm_before = norm_before
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = BuildNormalization(placeholder=d_model, norm_cfg=norm_cfg)
        self.norm2 = BuildNormalization(placeholder=d_model, norm_cfg=norm_cfg)
        self.norm3 = BuildNormalization(placeholder=d_model, norm_cfg=norm_cfg)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = BuildActivation(act_cfg)
    """withposembed"""

    def withposembed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos
    """normafterforward"""

    def normafterforward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        q = k = self.withposembed(tgt, query_pos)
        tgt = tgt + self.dropout1(self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0])
        tgt = self.norm1(tgt)
        tgt = tgt + self.dropout2(self.multihead_attn(self.withposembed(tgt, query_pos), self.withposembed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0])
        tgt = self.norm2(tgt)
        tgt = tgt + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(tgt)))))
        tgt = self.norm3(tgt)
        return tgt
    """normbeforeforward"""

    def normbeforeforward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        tgt_norm = self.norm1(tgt)
        q = k = self.withposembed(tgt_norm, query_pos)
        tgt = tgt + self.dropout1(self.self_attn(q, k, value=tgt_norm, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0])
        tgt_norm = self.norm2(tgt)
        tgt = tgt + self.dropout2(self.multihead_attn(self.withposembed(tgt_norm, query_pos), self.withposembed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0])
        tgt_norm = self.norm3(tgt)
        tgt = tgt + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(tgt_norm)))))
        return tgt
    """forward"""

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        if self.norm_before:
            return self.normbeforeforward(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.normafterforward(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.norm = norm
        self.num_layers = num_layers
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
    """forward"""

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, norm_cfg=None, act_cfg=None, norm_before=False, return_intermediate_dec=False):
        super(Transformer, self).__init__()
        self.nhead = nhead
        self.d_model = d_model
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, norm_cfg, act_cfg, norm_before)
        encoder_norm = BuildNormalization(placeholder=d_model, norm_cfg=norm_cfg) if norm_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, norm_cfg, act_cfg, norm_before)
        decoder_norm = BuildNormalization(placeholder=d_model, norm_cfg=norm_cfg)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)
        self.resetparameters()
    """resetparameters"""

    def resetparameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    """forward"""

    def forward(self, src, mask, query_embed, pos_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        if mask is not None:
            mask = mask.flatten(1)
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class Predictor(nn.Module):

    def __init__(self, in_channels, mask_classification, num_classes, hidden_dim, num_queries, nheads, dropout, dim_feedforward, enc_layers, dec_layers, pre_norm, deep_supervision, mask_dim, enforce_input_project, norm_cfg=None, act_cfg=None):
        super(Predictor, self).__init__()
        self.num_queries = num_queries
        self.in_channels = in_channels
        self.aux_loss = deep_supervision
        self.mask_classification = mask_classification
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, apply_normalize=True)
        self.transformer = Transformer(d_model=hidden_dim, nhead=nheads, num_encoder_layers=enc_layers, num_decoder_layers=dec_layers, dim_feedforward=dim_feedforward, dropout=dropout, norm_before=pre_norm, return_intermediate_dec=deep_supervision, act_cfg=act_cfg, norm_cfg=norm_cfg)
        hidden_dim = self.transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if in_channels != hidden_dim or enforce_input_project:
            self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0)
            weight_init.c2_xavier_fill(self.input_proj)
        else:
            self.input_proj = nn.Sequential()
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
    """forward"""

    def forward(self, x, mask_features):
        hs, memory = self.transformer(self.input_proj(x), None, self.query_embed.weight, self.pe_layer(x))
        outputs = {}
        if self.mask_classification:
            outputs_class = self.class_embed(hs)
            outputs.update({'pred_logits': self.class_embed(hs)[-1]})
        if self.aux_loss:
            mask_embed = self.mask_embed(hs)
            outputs_seg_masks = torch.einsum('lbqc,bchw->lbqhw', mask_embed, mask_features)
            outputs['pred_masks'] = outputs_seg_masks[-1]
            outputs['aux_outputs'] = self.setauxloss(outputs_class if self.mask_classification else None, outputs_seg_masks)
        else:
            mask_embed = self.mask_embed(hs[-1])
            outputs_seg_masks = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_features)
            outputs['pred_masks'] = outputs_seg_masks
        return outputs
    """setauxloss"""

    @torch.jit.unused
    def setauxloss(self, outputs_class, outputs_seg_masks):
        if self.mask_classification:
            return [{'pred_logits': a, 'pred_masks': b} for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]
        else:
            return [{'pred_masks': b} for b in outputs_seg_masks[:-1]]


class MaskFormer(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(MaskFormer, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        ppm_cfg = {'in_channels': head_cfg['in_channels_list'][-1], 'out_channels': head_cfg['feats_channels'], 'pool_scales': head_cfg['pool_scales'], 'align_corners': align_corners, 'norm_cfg': copy.deepcopy(norm_cfg), 'act_cfg': copy.deepcopy(act_cfg)}
        self.ppm_net = PyramidPoolingModule(**ppm_cfg)
        act_cfg_copy = copy.deepcopy(act_cfg)
        if 'inplace' in act_cfg_copy:
            act_cfg_copy['inplace'] = False
        self.lateral_convs = nn.ModuleList()
        for in_channels in head_cfg['in_channels_list'][:-1]:
            self.lateral_convs.append(nn.Sequential(nn.Conv2d(in_channels, head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg_copy)))
        self.fpn_convs = nn.ModuleList()
        for in_channels in ([head_cfg['feats_channels']] * len(self.lateral_convs)):
            self.fpn_convs.append(nn.Sequential(nn.Conv2d(in_channels, head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg_copy)))
        self.decoder_mask = nn.Sequential(nn.Conv2d(head_cfg['feats_channels'], head_cfg['mask_feats_channels'], kernel_size=3, stride=1, padding=1))
        head_cfg['predictor']['num_classes'] = cfg['num_classes']
        head_cfg['predictor']['mask_dim'] = head_cfg['mask_feats_channels']
        head_cfg['predictor']['in_channels'] = head_cfg['in_channels_list'][-1]
        self.decoder_predictor = Predictor(**head_cfg['predictor'])
        matcher = HungarianMatcher(**head_cfg['matcher'])
        weight_dict = {'loss_ce': head_cfg['matcher']['cost_class'], 'loss_mask': head_cfg['matcher']['cost_mask'], 'loss_dice': head_cfg['matcher']['cost_dice']}
        if head_cfg['predictor']['deep_supervision']:
            dec_layers = head_cfg['predictor']['dec_layers']
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({(k + f'_{i}'): v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        self.criterion = SetCriterion(cfg['num_classes'], matcher=matcher, weight_dict=weight_dict, eos_coef=0.1, losses=['labels', 'masks'])
        self.setauxiliarydecoder(cfg['auxiliary'])
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        ppm_out = self.ppm_net(backbone_outputs[-1])
        inputs = backbone_outputs[:-1]
        lateral_outputs = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        lateral_outputs.append(ppm_out)
        p1, p2, p3, p4 = lateral_outputs
        fpn_out = F.interpolate(p4, size=p3.shape[2:], mode='bilinear', align_corners=self.align_corners) + p3
        fpn_out = self.fpn_convs[0](fpn_out)
        fpn_out = F.interpolate(fpn_out, size=p2.shape[2:], mode='bilinear', align_corners=self.align_corners) + p2
        fpn_out = self.fpn_convs[1](fpn_out)
        fpn_out = F.interpolate(fpn_out, size=p1.shape[2:], mode='bilinear', align_corners=self.align_corners) + p1
        fpn_out = self.fpn_convs[2](fpn_out)
        mask_features = self.decoder_mask(fpn_out)
        predictions = self.decoder_predictor(backbone_outputs[-1], mask_features)
        ssseg_outputs = SSSegOutputStructure(mode=self.mode, auto_validate=False)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            losses_dict = self.criterion(predictions, data_meta.getannotations())
            for k in list(losses_dict.keys()):
                if k in self.criterion.weight_dict:
                    losses_dict[k] *= self.criterion.weight_dict[k]
                else:
                    losses_dict.pop(k)
            loss, losses_log_dict = 0, {}
            for loss_key, loss_value in losses_dict.items():
                loss_value = loss_value.mean()
                loss = loss + loss_value
                loss_value = loss_value.data.clone()
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(loss_value.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
                losses_log_dict[loss_key] = loss_value.item()
            losses_log_dict.update({'loss_total': sum(losses_log_dict.values())})
            ssseg_outputs.setvariable('loss', loss)
            ssseg_outputs.setvariable('losses_log_dict', losses_log_dict)
            if self.mode in ['TRAIN']:
                return ssseg_outputs
        mask_cls_results = predictions['pred_logits']
        mask_pred_results = predictions['pred_masks']
        mask_pred_results = F.interpolate(mask_pred_results, size=img_size, mode='bilinear', align_corners=self.align_corners)
        predictions = []
        for mask_cls, mask_pred in zip(mask_cls_results, mask_pred_results):
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum('qc,qhw->chw', mask_cls, mask_pred)
            predictions.append(semseg.unsqueeze(0))
        seg_logits = torch.cat(predictions, dim=0)
        ssseg_outputs.setvariable('seg_logits', seg_logits)
        return ssseg_outputs


class FeaturesMemory(nn.Module):

    def __init__(self, num_classes, feats_channels, transform_channels, out_channels, use_context_within_image=True, num_feats_per_cls=1, use_hard_aggregate=False, norm_cfg=None, act_cfg=None):
        super(FeaturesMemory, self).__init__()
        assert num_feats_per_cls > 0, 'num_feats_per_cls should be larger than 0'
        self.num_classes = num_classes
        self.feats_channels = feats_channels
        self.transform_channels = transform_channels
        self.out_channels = out_channels
        self.num_feats_per_cls = num_feats_per_cls
        self.use_context_within_image = use_context_within_image
        self.use_hard_aggregate = use_hard_aggregate
        self.memory = nn.Parameter(torch.zeros(num_classes, num_feats_per_cls, feats_channels, dtype=torch.float), requires_grad=False)
        if self.num_feats_per_cls > 1:
            self.self_attentions = nn.ModuleList()
            for _ in range(self.num_feats_per_cls):
                self_attention = SelfAttentionBlock(key_in_channels=feats_channels, query_in_channels=feats_channels, transform_channels=transform_channels, out_channels=feats_channels, share_key_query=False, query_downsample=None, key_downsample=None, key_query_num_convs=2, value_out_num_convs=1, key_query_norm=True, value_out_norm=True, matmul_norm=True, with_out_project=True, norm_cfg=norm_cfg, act_cfg=act_cfg)
                self.self_attentions.append(self_attention)
            self.fuse_memory_conv = nn.Sequential(nn.Conv2d(feats_channels * self.num_feats_per_cls, feats_channels, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=feats_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
        else:
            self.self_attention = SelfAttentionBlock(key_in_channels=feats_channels, query_in_channels=feats_channels, transform_channels=transform_channels, out_channels=feats_channels, share_key_query=False, query_downsample=None, key_downsample=None, key_query_num_convs=2, value_out_num_convs=1, key_query_norm=True, value_out_norm=True, matmul_norm=True, with_out_project=True, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.bottleneck = nn.Sequential(nn.Conv2d(feats_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
        if use_context_within_image:
            self.self_attention_ms = SelfAttentionBlock(key_in_channels=feats_channels, query_in_channels=feats_channels, transform_channels=transform_channels, out_channels=feats_channels, share_key_query=False, query_downsample=None, key_downsample=None, key_query_num_convs=2, value_out_num_convs=1, key_query_norm=True, value_out_norm=True, matmul_norm=True, with_out_project=True, norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.bottleneck_ms = nn.Sequential(nn.Conv2d(feats_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
    """forward"""

    def forward(self, feats, preds=None, feats_ms=None):
        batch_size, num_channels, h, w = feats.size()
        weight_cls = preds.permute(0, 2, 3, 1).contiguous()
        weight_cls = weight_cls.reshape(-1, self.num_classes)
        weight_cls = F.softmax(weight_cls, dim=-1)
        if self.use_hard_aggregate:
            labels = weight_cls.argmax(-1).reshape(-1, 1)
            onehot = torch.zeros_like(weight_cls).scatter_(1, labels.long(), 1)
            weight_cls = onehot
        selected_memory_list = []
        for idx in range(self.num_feats_per_cls):
            memory = self.memory.data[:, idx, :]
            selected_memory = torch.matmul(weight_cls, memory)
            selected_memory_list.append(selected_memory.unsqueeze(1))
        if self.num_feats_per_cls > 1:
            relation_selected_memory_list = []
            for idx, selected_memory in enumerate(selected_memory_list):
                selected_memory = selected_memory.view(batch_size, h, w, num_channels)
                selected_memory = selected_memory.permute(0, 3, 1, 2).contiguous()
                relation_selected_memory_list.append(self.self_attentions[idx](feats, selected_memory))
            selected_memory = torch.cat(relation_selected_memory_list, dim=1)
            selected_memory = self.fuse_memory_conv(selected_memory)
        else:
            assert len(selected_memory_list) == 1
            selected_memory = selected_memory_list[0].squeeze(1)
            selected_memory = selected_memory.view(batch_size, h, w, num_channels)
            selected_memory = selected_memory.permute(0, 3, 1, 2).contiguous()
            selected_memory = self.self_attention(feats, selected_memory)
        memory_output = self.bottleneck(torch.cat([feats, selected_memory], dim=1))
        if self.use_context_within_image:
            feats_ms = self.self_attention_ms(feats, feats_ms)
            memory_output = self.bottleneck_ms(torch.cat([feats_ms, memory_output], dim=1))
        return self.memory.data, memory_output
    """update"""

    def update(self, features, segmentation, ignore_index=255, strategy='cosine_similarity', momentum_cfg=None, learning_rate=None):
        assert strategy in ['mean', 'cosine_similarity']
        batch_size, num_channels, h, w = features.size()
        momentum = momentum_cfg['base_momentum']
        if momentum_cfg['adjust_by_learning_rate']:
            momentum = momentum_cfg['base_momentum'] / momentum_cfg['base_lr'] * learning_rate
        segmentation = segmentation.long()
        features = features.permute(0, 2, 3, 1).contiguous()
        features = features.view(batch_size * h * w, num_channels)
        clsids = segmentation.unique()
        for clsid in clsids:
            if clsid == ignore_index:
                continue
            seg_cls = segmentation.view(-1)
            feats_cls = features[seg_cls == clsid]
            need_update = True
            for idx in range(self.num_feats_per_cls):
                if (self.memory[clsid][idx] == 0).sum() == self.feats_channels:
                    self.memory[clsid][idx].data.copy_(feats_cls.mean(0))
                    need_update = False
                    break
            if not need_update:
                continue
            if self.num_feats_per_cls == 1:
                if strategy == 'mean':
                    feats_cls = feats_cls.mean(0)
                elif strategy == 'cosine_similarity':
                    similarity = F.cosine_similarity(feats_cls, self.memory[clsid].data.expand_as(feats_cls))
                    weight = (1 - similarity) / (1 - similarity).sum()
                    feats_cls = (feats_cls * weight.unsqueeze(-1)).sum(0)
                feats_cls = (1 - momentum) * self.memory[clsid].data + momentum * feats_cls.unsqueeze(0)
                self.memory[clsid].data.copy_(feats_cls)
            else:
                assert strategy in ['cosine_similarity']
                relation = torch.matmul(F.normalize(feats_cls, p=2, dim=1), F.normalize(self.memory[clsid].data.permute(1, 0).contiguous(), p=2, dim=0))
                argmax = relation.argmax(dim=1)
                for idx in range(self.num_feats_per_cls):
                    mask = argmax == idx
                    feats_cls_iter = feats_cls[mask]
                    memory_cls_iter = self.memory[clsid].data[idx].unsqueeze(0).expand_as(feats_cls_iter)
                    similarity = F.cosine_similarity(feats_cls_iter, memory_cls_iter)
                    weight = (1 - similarity) / (1 - similarity).sum()
                    feats_cls_iter = (feats_cls_iter * weight.unsqueeze(-1)).sum(0)
                    self.memory[clsid].data[idx].copy_(self.memory[clsid].data[idx] * (1 - momentum) + feats_cls_iter * momentum)
        if dist.is_available() and dist.is_initialized():
            memory = self.memory.data.clone()
            dist.all_reduce(memory.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
            self.memory = nn.Parameter(memory, requires_grad=False)


class MCIBI(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(MCIBI, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        if 'norm_cfg' in head_cfg:
            self.norm_layers = nn.ModuleList()
            for in_channels in head_cfg['norm_cfg']['in_channels_list']:
                norm_cfg_copy = head_cfg['norm_cfg'].copy()
                norm_cfg_copy.pop('in_channels_list')
                norm_layer = BuildNormalization(placeholder=in_channels, norm_cfg=norm_cfg_copy)
                self.norm_layers.append(norm_layer)
        if head_cfg['downsample_backbone']['stride'] > 1:
            self.downsample_backbone = nn.Sequential(nn.Conv2d(head_cfg['in_channels'], head_cfg['in_channels'], **head_cfg['downsample_backbone']), BuildNormalization(placeholder=head_cfg['in_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        context_within_image_cfg = head_cfg['context_within_image']
        if context_within_image_cfg['is_on']:
            cwi_cfg = context_within_image_cfg['cfg']
            cwi_cfg.update({'in_channels': head_cfg['in_channels'], 'out_channels': head_cfg['feats_channels'], 'align_corners': align_corners, 'norm_cfg': copy.deepcopy(norm_cfg), 'act_cfg': copy.deepcopy(act_cfg)})
            supported_context_modules = {'aspp': ASPP, 'ppm': PyramidPoolingModule}
            self.context_within_image_module = supported_context_modules[context_within_image_cfg['type']](**cwi_cfg)
        self.bottleneck = nn.Sequential(nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.memory_module = FeaturesMemory(num_classes=cfg['num_classes'], feats_channels=head_cfg['feats_channels'], transform_channels=head_cfg['transform_channels'], num_feats_per_cls=head_cfg['num_feats_per_cls'], out_channels=head_cfg['out_channels'], use_context_within_image=context_within_image_cfg['is_on'], use_hard_aggregate=head_cfg['use_hard_aggregate'], norm_cfg=copy.deepcopy(norm_cfg), act_cfg=copy.deepcopy(act_cfg))
        self.decoder_stage1 = nn.Sequential(nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.decoder_stage2 = nn.Sequential(nn.Conv2d(head_cfg['out_channels'], head_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=head_cfg['out_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.setauxiliarydecoder(cfg['auxiliary'])
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta, **kwargs):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        if hasattr(self, 'norm_layers'):
            assert len(backbone_outputs) == len(self.norm_layers)
            for idx in range(len(backbone_outputs)):
                backbone_outputs[idx] = self.norm(backbone_outputs[idx], self.norm_layers[idx])
        if self.cfg['head']['downsample_backbone']['stride'] > 1:
            for idx in range(len(backbone_outputs)):
                backbone_outputs[idx] = self.downsample_backbone(backbone_outputs[idx])
        feats_ms = self.context_within_image_module(backbone_outputs[-1]) if hasattr(self, 'context_within_image_module') else None
        memory_input = self.bottleneck(backbone_outputs[-1])
        preds_stage1 = self.decoder_stage1(memory_input)
        stored_memory, memory_output = self.memory_module(memory_input, preds_stage1, feats_ms)
        preds_stage2 = self.decoder_stage2(memory_output)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            predictions = self.customizepredsandlosses(seg_logits=preds_stage2, annotations=data_meta.getannotations(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size, auto_calc_loss=False)
            preds_stage2 = predictions.pop('loss_cls')
            preds_stage1 = F.interpolate(preds_stage1, size=img_size, mode='bilinear', align_corners=self.align_corners)
            predictions.update({'loss_cls_stage1': preds_stage1, 'loss_cls_stage2': preds_stage2})
            with torch.no_grad():
                self.memory_module.update(features=F.interpolate(memory_input, size=img_size, mode='bilinear', align_corners=self.align_corners), segmentation=data_meta.getannotations()['seg_targets'], learning_rate=kwargs['learning_rate'], **self.cfg['head']['update_cfg'])
            loss, losses_log_dict = calculatelosses(predictions=predictions, annotations=data_meta.getannotations(), losses_cfg=self.cfg['losses'], pixel_sampler=self.pixel_sampler)
            if kwargs['epoch'] > 1 and self.cfg['head']['use_loss']:
                loss_memory, loss_memory_log = self.calculatememoryloss(stored_memory)
                loss += loss_memory
                losses_log_dict['loss_memory'] = loss_memory_log
                total = losses_log_dict.pop('total') + losses_log_dict['loss_memory']
                losses_log_dict['total'] = total
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=preds_stage2)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=preds_stage2)
        return ssseg_outputs
    """norm"""

    def norm(self, x, norm_layer):
        n, c, h, w = x.shape
        x = x.reshape(n, c, h * w).transpose(2, 1).contiguous()
        x = norm_layer(x)
        x = x.transpose(1, 2).reshape(n, c, h, w).contiguous()
        return x
    """calculatememoryloss"""

    def calculatememoryloss(self, stored_memory):
        num_classes, num_feats_per_cls, feats_channels = stored_memory.size()
        stored_memory = stored_memory.reshape(num_classes * num_feats_per_cls, feats_channels, 1, 1)
        preds_memory = self.decoder_stage2(stored_memory)
        target = torch.range(0, num_classes - 1).type_as(stored_memory).long()
        target = target.unsqueeze(1).repeat(1, num_feats_per_cls).view(-1)
        loss_memory = calculateloss(preds_memory.sequeeze(-1).sequeeze(-1), target, self.cfg['head']['loss_cfg'])
        loss_memory_log = loss_memory.data.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_memory_log.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
        return loss_memory, loss_memory_log


class FeaturesMemoryV2(nn.Module):

    def __init__(self, num_classes, feats_channels, transform_channels, out_channels, use_hard_aggregate=False, downsample_before_sa=False, norm_cfg=None, act_cfg=None, align_corners=False):
        super(FeaturesMemoryV2, self).__init__()
        self.align_corners = align_corners
        self.num_classes = num_classes
        self.feats_channels = feats_channels
        self.transform_channels = transform_channels
        self.out_channels = out_channels
        self.use_hard_aggregate = use_hard_aggregate
        if downsample_before_sa:
            self.downsample_before_sa = nn.Sequential(nn.Conv2d(feats_channels, feats_channels, kernel_size=3, stride=2, padding=1, bias=False), BuildNormalization(placeholder=feats_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.memory = nn.Parameter(torch.cat([torch.zeros(num_classes, 1, dtype=torch.float), torch.ones(num_classes, 1, dtype=torch.float)], dim=1), requires_grad=False)
        self.self_attention = SelfAttentionBlock(key_in_channels=feats_channels, query_in_channels=feats_channels, transform_channels=transform_channels, out_channels=feats_channels, share_key_query=False, query_downsample=None, key_downsample=None, key_query_num_convs=2, value_out_num_convs=1, key_query_norm=True, value_out_norm=True, matmul_norm=True, with_out_project=True, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.bottleneck = nn.Sequential(nn.Conv2d(feats_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
    """forward"""

    def forward(self, feats, preds=None):
        batch_size, num_channels, h, w = feats.size()
        weight_cls = preds.permute(0, 2, 3, 1).contiguous()
        weight_cls = weight_cls.reshape(-1, self.num_classes)
        weight_cls = F.softmax(weight_cls, dim=-1)
        if self.use_hard_aggregate:
            labels = weight_cls.argmax(-1).reshape(-1, 1)
            onehot = torch.zeros_like(weight_cls).scatter_(1, labels.long(), 1)
            weight_cls = onehot
        memory_means = self.memory.data[:, 0]
        memory_stds = self.memory.data[:, 1]
        memory = []
        for idx in range(self.num_classes):
            torch.manual_seed(idx)
            cls_memory = torch.normal(mean=torch.full((1, self.feats_channels), memory_means[idx]), std=torch.full((1, self.feats_channels), memory_stds[idx]))
            memory.append(cls_memory)
        memory = torch.cat(memory, dim=0).type_as(weight_cls)
        selected_memory = torch.matmul(weight_cls, memory)
        selected_memory = selected_memory.view(batch_size, h, w, num_channels)
        selected_memory = selected_memory.permute(0, 3, 1, 2).contiguous()
        if hasattr(self, 'downsample_before_sa'):
            feats_in, selected_memory_in = self.downsample_before_sa(feats), self.downsample_before_sa(selected_memory)
        else:
            feats_in, selected_memory_in = feats, selected_memory
        selected_memory = self.self_attention(feats_in, selected_memory_in)
        if hasattr(self, 'downsample_before_sa'):
            selected_memory = F.interpolate(selected_memory, size=feats.size()[2:], mode='bilinear', align_corners=self.align_corners)
        memory_output = self.bottleneck(torch.cat([feats, selected_memory], dim=1))
        return memory.data, memory_output
    """update"""

    def update(self, features, segmentation, ignore_index=255, momentum_cfg=None, learning_rate=None):
        batch_size, num_channels, h, w = features.size()
        momentum = momentum_cfg['base_momentum']
        if momentum_cfg['adjust_by_learning_rate']:
            momentum = momentum_cfg['base_momentum'] / momentum_cfg['base_lr'] * learning_rate
        segmentation = segmentation.long()
        features = features.permute(0, 2, 3, 1).contiguous()
        features = features.view(batch_size * h * w, num_channels)
        clsids = segmentation.unique()
        for clsid in clsids:
            if clsid == ignore_index:
                continue
            seg_cls = segmentation.view(-1)
            feats_cls = features[seg_cls == clsid]
            feats_cls = feats_cls.mean(0)
            mean, std = feats_cls.mean(), feats_cls.std()
            self.memory[clsid][0] = (1 - momentum) * self.memory[clsid][0].data + momentum * mean
            self.memory[clsid][1] = (1 - momentum) * self.memory[clsid][1].data + momentum * std
        memory = self.memory.data.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(memory.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
        self.memory = nn.Parameter(memory, requires_grad=False)


class MCIBIPlusPlus(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(MCIBIPlusPlus, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        context_within_image_cfg = head_cfg['context_within_image']
        if context_within_image_cfg['is_on']:
            cwi_cfg = context_within_image_cfg['cfg']
            cwi_cfg.update({'in_channels': head_cfg['in_channels'], 'out_channels': head_cfg['feats_channels'], 'align_corners': align_corners, 'norm_cfg': copy.deepcopy(norm_cfg), 'act_cfg': copy.deepcopy(act_cfg)})
            supported_context_modules = {'aspp': ASPP, 'ppm': PyramidPoolingModule}
            if context_within_image_cfg['type'] == 'aspp':
                cwi_cfg.pop('pool_scales')
            elif context_within_image_cfg['type'] == 'ppm':
                cwi_cfg.pop('dilations')
            self.context_within_image_module = supported_context_modules[context_within_image_cfg['type']](**cwi_cfg)
            if context_within_image_cfg.get('use_self_attention', True):
                self.self_attention = SelfAttentionBlock(key_in_channels=head_cfg['feats_channels'], query_in_channels=head_cfg['feats_channels'], transform_channels=head_cfg['feats_channels'] // 2, out_channels=head_cfg['feats_channels'], share_key_query=False, query_downsample=None, key_downsample=None, key_query_num_convs=2, value_out_num_convs=1, key_query_norm=True, value_out_norm=True, matmul_norm=True, with_out_project=True, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.bottleneck = nn.Sequential(nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.memory_module = FeaturesMemoryV2(num_classes=cfg['num_classes'], feats_channels=head_cfg['feats_channels'], transform_channels=head_cfg['transform_channels'], out_channels=head_cfg['out_channels'], use_hard_aggregate=head_cfg['use_hard_aggregate'], downsample_before_sa=head_cfg['downsample_before_sa'], norm_cfg=copy.deepcopy(norm_cfg), act_cfg=copy.deepcopy(act_cfg), align_corners=align_corners)
        if head_cfg.get('fpn', None) is not None:
            act_cfg_copy = copy.deepcopy(act_cfg)
            if 'inplace' in act_cfg_copy:
                act_cfg_copy['inplace'] = False
            self.lateral_convs = nn.ModuleList()
            for in_channels in head_cfg['fpn']['in_channels_list'][:-1]:
                self.lateral_convs.append(nn.Sequential(nn.Conv2d(in_channels, head_cfg['fpn']['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=head_cfg['fpn']['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg_copy)))
            self.fpn_convs = nn.ModuleList()
            for in_channels in ([head_cfg['fpn']['feats_channels']] * len(self.lateral_convs)):
                self.fpn_convs.append(nn.Sequential(nn.Conv2d(in_channels, head_cfg['fpn']['out_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['fpn']['out_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg_copy)))
        for key, value in head_cfg['decoder'].items():
            if key == 'cwi' and not context_within_image_cfg['is_on']:
                continue
            setattr(self, f'decoder_{key}', nn.Sequential())
            decoder = getattr(self, f'decoder_{key}')
            decoder.add_module('conv1', nn.Conv2d(value['in_channels'], value['out_channels'], kernel_size=value.get('kernel_size', 1), stride=1, padding=value.get('padding', 0), bias=False))
            decoder.add_module('bn1', BuildNormalization(placeholder=value['out_channels'], norm_cfg=norm_cfg))
            decoder.add_module('act1', BuildActivation(act_cfg))
            decoder.add_module('dropout', nn.Dropout2d(value['dropout']))
            decoder.add_module('conv2', nn.Conv2d(value['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.setauxiliarydecoder(cfg['auxiliary'])
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta, **kwargs):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        if hasattr(self, 'context_within_image_module'):
            feats_cwi = self.context_within_image_module(backbone_outputs[-1])
            if hasattr(self, 'decoder_cwi'):
                preds_cwi = self.decoder_cwi(feats_cwi)
        pixel_representations = self.bottleneck(backbone_outputs[-1])
        preds_pr = self.decoder_pr(pixel_representations)
        if self.cfg['head'].get('force_use_preds_pr', False):
            memory_gather_logits = preds_pr
        else:
            memory_gather_logits = preds_cwi if hasattr(self, 'context_within_image_module') and hasattr(self, 'decoder_cwi') else preds_pr
        memory_input = pixel_representations
        assert memory_input.shape[2:] == memory_gather_logits.shape[2:]
        if self.mode == 'TRAIN' and kwargs['epoch'] < self.cfg['head'].get('warmup_epoch', 0):
            with torch.no_grad():
                gt = data_meta.getannotations()['seg_targets']
                gt = F.interpolate(gt.unsqueeze(1), size=memory_gather_logits.shape[2:], mode='nearest')[:, 0, :, :]
                assert len(gt.shape) == 3, 'seg_targets format error'
                preds_gt = gt.new_zeros(memory_gather_logits.shape).type_as(memory_gather_logits)
                valid_mask = (gt >= 0) & (gt < self.cfg['num_classes'])
                idxs = torch.nonzero(valid_mask, as_tuple=True)
                if idxs[0].numel() > 0:
                    preds_gt[idxs[0], gt[valid_mask].long(), idxs[1], idxs[2]] = 1
            stored_memory, memory_output = self.memory_module(memory_input, preds_gt.detach())
        else:
            if 'memory_gather_logits' in kwargs:
                memory_gather_logits_aux = F.interpolate(kwargs['memory_gather_logits'], size=memory_gather_logits.shape[2:], mode='bilinear', align_corners=self.align_corners)
                weights = kwargs.get('memory_gather_logits_weights', [2, 1.5])
                memory_gather_logits = (memory_gather_logits * weights[0] + memory_gather_logits_aux * weights[1]) / (sum(weights) - 1)
            stored_memory, memory_output = self.memory_module(memory_input, memory_gather_logits)
        if hasattr(self, 'context_within_image_module'):
            if hasattr(self, 'self_attention'):
                memory_output = self.self_attention(feats_cwi, memory_output)
            if hasattr(self, 'fpn_convs'):
                inputs = backbone_outputs[:-1]
                lateral_outputs = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
                if self.cfg['head'].get('fuse_memory_cwi_before_fpn', True):
                    lateral_outputs.append(torch.cat([memory_output, feats_cwi], dim=1))
                else:
                    lateral_outputs.append(feats_cwi)
                for i in range(len(lateral_outputs) - 1, 0, -1):
                    prev_shape = lateral_outputs[i - 1].shape[2:]
                    lateral_outputs[i - 1] = lateral_outputs[i - 1] + F.interpolate(lateral_outputs[i], size=prev_shape, mode='bilinear', align_corners=self.align_corners)
                fpn_outputs = [self.fpn_convs[i](lateral_outputs[i]) for i in range(len(lateral_outputs) - 1)]
                fpn_outputs.append(lateral_outputs[-1])
                fpn_outputs = [F.interpolate(out, size=fpn_outputs[0].size()[2:], mode='bilinear', align_corners=self.align_corners) for out in fpn_outputs]
                if not self.cfg['head'].get('fuse_memory_cwi_before_fpn', True):
                    fpn_outputs.append(F.interpolate(memory_output, size=fpn_outputs[0].shape[2:], mode='bilinear', align_corners=self.align_corners))
                memory_output = torch.cat(fpn_outputs, dim=1)
            else:
                memory_output = torch.cat([memory_output, feats_cwi], dim=1)
        preds_cls = self.decoder_cls(memory_output)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            predictions = self.customizepredsandlosses(seg_logits=preds_cls, annotations=data_meta.getannotations(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size, auto_calc_loss=False)
            preds_cls = predictions.pop('loss_cls')
            preds_pr = F.interpolate(preds_pr, size=img_size, mode='bilinear', align_corners=self.align_corners)
            predictions.update({'loss_pr': preds_pr, 'loss_cls': preds_cls})
            if hasattr(self, 'context_within_image_module') and hasattr(self, 'decoder_cwi'):
                preds_cwi = F.interpolate(preds_cwi, size=img_size, mode='bilinear', align_corners=self.align_corners)
                predictions.update({'loss_cwi': preds_cwi})
            with torch.no_grad():
                self.memory_module.update(features=F.interpolate(pixel_representations, size=img_size, mode='bilinear', align_corners=self.align_corners), segmentation=data_meta.getannotations()['seg_targets'], learning_rate=kwargs['learning_rate'], **self.cfg['head']['update_cfg'])
            loss, losses_log_dict = calculatelosses(predictions=predictions, annotations=data_meta.getannotations(), losses_cfg=self.cfg['losses'], pixel_sampler=self.pixel_sampler)
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=preds_cls)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=preds_cls)
        return ssseg_outputs


class NonLocal1d(_NonLocalNd):

    def __init__(self, in_channels, sub_sample=False, **kwargs):
        super(NonLocal1d, self).__init__(in_channels, **kwargs)
        self.sub_sample = sub_sample
        if sub_sample:
            max_pool_layer = nn.MaxPool1d(kernel_size=2)
            self.g = nn.Sequential(self.g, max_pool_layer)
            if self.mode != 'gaussian':
                self.phi = nn.Sequential(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer


class NonLocal3d(_NonLocalNd):

    def __init__(self, in_channels, sub_sample=False, **kwargs):
        super(NonLocal3d, self).__init__(in_channels, **kwargs)
        self.sub_sample = sub_sample
        if sub_sample:
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            self.g = nn.Sequential(self.g, max_pool_layer)
            if self.mode != 'gaussian':
                self.phi = nn.Sequential(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer


class NonLocalNet(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(NonLocalNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        self.conv_before_nl = nn.Sequential(nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.nl_block = NonLocal2d(in_channels=head_cfg['feats_channels'], reduction=head_cfg['reduction'], use_scale=head_cfg['use_scale'], mode=head_cfg['mode'], norm_cfg=copy.deepcopy(norm_cfg), act_cfg=copy.deepcopy(act_cfg))
        self.conv_after_nl = nn.Sequential(nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.decoder = nn.Sequential(nn.Conv2d(head_cfg['in_channels'] + head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.setauxiliarydecoder(cfg['auxiliary'])
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        feats = self.conv_before_nl(backbone_outputs[-1])
        feats = self.nl_block(feats)
        feats = self.conv_after_nl(feats)
        feats = torch.cat([backbone_outputs[-1], feats], dim=1)
        seg_logits = self.decoder(feats)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            loss, losses_log_dict = self.customizepredsandlosses(seg_logits=seg_logits, annotations=data_meta.getannotations(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size)
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs


class SpatialGatherModule(nn.Module):

    def __init__(self, scale=1):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale
    """forward"""

    def forward(self, features, probs):
        batch_size, num_classes, h, w = probs.size()
        probs = probs.view(batch_size, num_classes, -1)
        features = features.view(batch_size, features.size(1), -1)
        features = features.permute(0, 2, 1)
        probs = F.softmax(self.scale * probs, dim=2)
        ocr_context = torch.matmul(probs, features)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(3)
        return ocr_context


class OCRNet(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(OCRNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        assert cfg['auxiliary'] is not None and isinstance(cfg['auxiliary'], dict), 'auxiliary must be given and only support dict type'
        self.setauxiliarydecoder(cfg['auxiliary'])
        self.bottleneck = nn.Sequential(nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        spatialgather_cfg = {'scale': head_cfg['scale']}
        self.spatial_gather_module = SpatialGatherModule(**spatialgather_cfg)
        self.object_context_block = ObjectContextBlock(in_channels=head_cfg['feats_channels'], transform_channels=head_cfg['transform_channels'], scale=head_cfg['scale'], align_corners=align_corners, norm_cfg=copy.deepcopy(norm_cfg), act_cfg=copy.deepcopy(act_cfg))
        self.decoder = nn.Sequential(nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        seg_logits_aux = self.auxiliary_decoder(backbone_outputs[-2])
        feats = self.bottleneck(backbone_outputs[-1])
        context = self.spatial_gather_module(feats, seg_logits_aux)
        feats = self.object_context_block(feats, context)
        seg_logits = self.decoder(feats)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            seg_logits = F.interpolate(seg_logits, size=img_size, mode='bilinear', align_corners=self.align_corners)
            seg_logits_aux = F.interpolate(seg_logits_aux, size=img_size, mode='bilinear', align_corners=self.align_corners)
            loss, losses_log_dict = calculatelosses(predictions={'loss_cls': seg_logits, 'loss_aux': seg_logits_aux}, annotations=data_meta.getannotations(), losses_cfg=self.cfg['losses'], pixel_sampler=self.pixel_sampler)
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs


class PointRend(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(PointRend, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        self.fpn_neck = FPN(in_channels_list=head_cfg['fpn_in_channels_list'], out_channels=head_cfg['feats_channels'], upsample_cfg=head_cfg['upsample_cfg'], norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.scale_heads, feature_stride_list = nn.ModuleList(), head_cfg['feature_stride_list']
        for i in range(len(feature_stride_list)):
            head_length = max(1, int(np.log2(feature_stride_list[i]) - np.log2(feature_stride_list[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(nn.Sequential(nn.Conv2d(head_cfg['feats_channels'] if k == 0 else head_cfg['scale_head_channels'], head_cfg['scale_head_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['scale_head_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg)))
                if feature_stride_list[i] != feature_stride_list[0]:
                    scale_head.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))
        self.num_fcs, self.coarse_pred_each_layer = head_cfg['num_fcs'], head_cfg['coarse_pred_each_layer']
        fc_in_channels = sum(head_cfg['pointrend_in_channels_list']) + cfg['num_classes']
        fc_channels = head_cfg['feats_channels']
        self.fcs = nn.ModuleList()
        for k in range(self.num_fcs):
            fc = nn.Sequential(nn.Conv1d(fc_in_channels, fc_channels, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=fc_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg))
            self.fcs.append(fc)
            fc_in_channels = fc_channels
            fc_in_channels += cfg['num_classes'] if self.coarse_pred_each_layer else 0
        self.decoder = nn.Sequential(nn.Dropout(head_cfg['dropout']), nn.Conv1d(fc_in_channels, cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        assert cfg['auxiliary'] is not None and isinstance(cfg['auxiliary'], dict), 'auxiliary must be given and only support dict type'
        self.setauxiliarydecoder(cfg['auxiliary'])
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        fpn_outs = self.fpn_neck(list(backbone_outputs))
        feats = self.scale_heads[0](fpn_outs[0])
        for i in range(1, len(self.cfg['head']['feature_stride_list'])):
            feats = feats + F.interpolate(self.scale_heads[i](fpn_outs[i]), size=feats.shape[2:], mode='bilinear', align_corners=self.align_corners)
        seg_logits_aux = self.auxiliary_decoder(feats)
        feats = fpn_outs[0]
        ssseg_outputs = SSSegOutputStructure(mode=self.mode, auto_validate=False)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            with torch.no_grad():
                points = self.getpointstrain(seg_logits_aux, self.calculateuncertainty, cfg=self.cfg['head']['train'])
            fine_grained_point_feats = self.getfinegrainedpointfeats([feats], points)
            coarse_point_feats = self.getcoarsepointfeats(seg_logits_aux, points)
            feats_concat = torch.cat([fine_grained_point_feats, coarse_point_feats], dim=1)
            for fc in self.fcs:
                feats_concat = fc(feats_concat)
                if self.coarse_pred_each_layer:
                    feats_concat = torch.cat([feats_concat, coarse_point_feats], dim=1)
            seg_logits = self.decoder(feats_concat)
            point_labels = PointSample(data_meta.getannotations()['seg_targets'].unsqueeze(1).float(), points, mode='nearest', align_corners=self.align_corners)
            point_labels = point_labels.squeeze(1).long()
            annotations = data_meta.getannotations()
            annotations['point_labels'] = point_labels
            seg_logits_aux = F.interpolate(seg_logits_aux, size=img_size, mode='bilinear', align_corners=self.align_corners)
            loss, losses_log_dict = calculatelosses(predictions={'loss_cls': seg_logits, 'loss_aux': seg_logits_aux}, annotations=annotations, losses_cfg=self.cfg['losses'], preds_to_tgts_mapping={'loss_cls': 'point_labels', 'loss_aux': 'seg_targets'}, pixel_sampler=self.pixel_sampler)
            ssseg_outputs.setvariable('loss', loss)
            ssseg_outputs.setvariable('losses_log_dict', losses_log_dict)
            if self.mode in ['TRAIN']:
                return ssseg_outputs
        refined_seg_logits = seg_logits_aux.clone()
        for _ in range(self.cfg['head']['test']['subdivision_steps']):
            refined_seg_logits = F.interpolate(input=refined_seg_logits, scale_factor=self.cfg['head']['test']['scale_factor'], mode='bilinear', align_corners=self.align_corners)
            batch_size, channels, height, width = refined_seg_logits.shape
            point_indices, points = self.getpointstest(refined_seg_logits, self.calculateuncertainty, cfg=self.cfg['head']['test'])
            fine_grained_point_feats = self.getfinegrainedpointfeats([feats], points)
            coarse_point_feats = self.getcoarsepointfeats(seg_logits_aux, points)
            feats_concat = torch.cat([fine_grained_point_feats, coarse_point_feats], dim=1)
            for fc in self.fcs:
                feats_concat = fc(feats_concat)
                if self.coarse_pred_each_layer:
                    feats_concat = torch.cat([feats_concat, coarse_point_feats], dim=1)
            seg_logits = self.decoder(feats_concat)
            point_indices = point_indices.unsqueeze(1).expand(-1, channels, -1)
            refined_seg_logits = refined_seg_logits.reshape(batch_size, channels, height * width)
            refined_seg_logits = refined_seg_logits.scatter_(2, point_indices, seg_logits)
            refined_seg_logits = refined_seg_logits.view(batch_size, channels, height, width)
        ssseg_outputs.setvariable('seg_logits', refined_seg_logits)
        return ssseg_outputs
    """getcoarsepointfeats"""

    def getcoarsepointfeats(self, seg_logits, points):
        coarse_feats = PointSample(seg_logits, points, align_corners=self.align_corners)
        return coarse_feats
    """getfinegrainedpointfeats"""

    def getfinegrainedpointfeats(self, x, points):
        fine_grained_feats_list = [PointSample(_, points, align_corners=self.align_corners) for _ in x]
        if len(fine_grained_feats_list) > 1:
            fine_grained_feats = torch.cat(fine_grained_feats_list, dim=1)
        else:
            fine_grained_feats = fine_grained_feats_list[0]
        return fine_grained_feats
    """calculateuncertainty"""

    @staticmethod
    def calculateuncertainty(seg_logits):
        top2_scores = torch.topk(seg_logits, k=2, dim=1)[0]
        return (top2_scores[:, 1] - top2_scores[:, 0]).unsqueeze(1)
    """getpointstrain"""

    def getpointstrain(self, seg_logits, uncertainty_func, cfg):
        num_points = cfg['num_points']
        oversample_ratio = cfg['oversample_ratio']
        importance_sample_ratio = cfg['importance_sample_ratio']
        assert oversample_ratio >= 1 and 0 <= importance_sample_ratio <= 1
        batch_size = seg_logits.shape[0]
        num_sampled = int(num_points * oversample_ratio)
        point_coords = torch.rand(batch_size, num_sampled, 2, device=seg_logits.device)
        point_logits = PointSample(seg_logits, point_coords)
        point_uncertainties = uncertainty_func(point_logits)
        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points
        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_sampled * torch.arange(batch_size, dtype=torch.long, device=seg_logits.device)
        idx = idx + shift[:, None]
        point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(batch_size, num_uncertain_points, 2)
        if num_random_points > 0:
            rand_point_coords = torch.rand(batch_size, num_random_points, 2, device=seg_logits.device)
            point_coords = torch.cat((point_coords, rand_point_coords), dim=1)
        return point_coords
    """getpointstest"""

    def getpointstest(self, seg_logits, uncertainty_func, cfg):
        num_points = cfg['subdivision_num_points']
        uncertainty_map = uncertainty_func(seg_logits)
        batch_size, _, height, width = uncertainty_map.shape
        h_step, w_step = 1.0 / height, 1.0 / width
        uncertainty_map = uncertainty_map.view(batch_size, height * width)
        num_points = min(height * width, num_points)
        point_indices = uncertainty_map.topk(num_points, dim=1)[1]
        point_coords = torch.zeros(batch_size, num_points, 2, dtype=torch.float, device=seg_logits.device)
        point_coords[:, :, 0] = w_step / 2.0 + (point_indices % width).float() * w_step
        point_coords[:, :, 1] = h_step / 2.0 + (point_indices // width).float() * h_step
        return point_indices, point_coords


class PSANet(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(PSANet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        assert head_cfg['type'] in ['collect', 'distribute', 'bi-direction']
        mask_h, mask_w = head_cfg['mask_size']
        if 'normalization_factor' not in self.cfg['head']:
            self.cfg['head']['normalization_factor'] = mask_h * mask_w
        self.reduce = nn.Sequential(nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.attention = nn.Sequential(nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Conv2d(head_cfg['feats_channels'], mask_h * mask_w, kernel_size=1, stride=1, padding=0, bias=False))
        if head_cfg['type'] == 'bi-direction':
            self.reduce_p = nn.Sequential(nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
            self.attention_p = nn.Sequential(nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Conv2d(head_cfg['feats_channels'], mask_h * mask_w, kernel_size=1, stride=1, padding=0, bias=False))
            if not head_cfg['compact']:
                self.psamask_collect = PSAMask('collect', head_cfg['mask_size'])
                self.psamask_distribute = PSAMask('distribute', head_cfg['mask_size'])
        elif not head_cfg['compact']:
            self.psamask = PSAMask(head_cfg['type'], head_cfg['mask_size'])
        self.proj = nn.Sequential(nn.Conv2d(head_cfg['feats_channels'] * (2 if head_cfg['type'] == 'bi-direction' else 1), head_cfg['in_channels'], kernel_size=1, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['in_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg))
        self.decoder = nn.Sequential(nn.Conv2d(head_cfg['in_channels'] * 2, head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.setauxiliarydecoder(cfg['auxiliary'])
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        identity = backbone_outputs[-1]
        shrink_factor, align_corners = self.cfg['head']['shrink_factor'], self.align_corners
        if self.cfg['head']['type'] in ['collect', 'distribute']:
            out = self.reduce(backbone_outputs[-1])
            n, c, h, w = out.size()
            if shrink_factor != 1:
                if h % shrink_factor and w % shrink_factor:
                    h = (h - 1) // shrink_factor + 1
                    w = (w - 1) // shrink_factor + 1
                    align_corners = True
                else:
                    h = h // shrink_factor
                    w = w // shrink_factor
                    align_corners = False
                out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=align_corners)
            y = self.attention(out)
            if self.cfg['head']['compact']:
                if self.cfg['head']['type'] == 'collect':
                    y = y.view(n, h * w, h * w).transpose(1, 2).view(n, h * w, h, w)
            else:
                y = self.psamask(y)
            if self.cfg['head']['psa_softmax']:
                y = F.softmax(y, dim=1)
            out = torch.bmm(out.view(n, c, h * w), y.view(n, h * w, h * w)).view(n, c, h, w) * (1.0 / self.cfg['head']['normalization_factor'])
        else:
            x_col = self.reduce(backbone_outputs[-1])
            x_dis = self.reduce_p(backbone_outputs[-1])
            n, c, h, w = x_col.size()
            if shrink_factor != 1:
                if h % shrink_factor and w % shrink_factor:
                    h = (h - 1) // shrink_factor + 1
                    w = (w - 1) // shrink_factor + 1
                    align_corners = True
                else:
                    h = h // shrink_factor
                    w = w // shrink_factor
                    align_corners = False
                x_col = F.interpolate(x_col, size=(h, w), mode='bilinear', align_corners=align_corners)
                x_dis = F.interpolate(x_dis, size=(h, w), mode='bilinear', align_corners=align_corners)
            y_col = self.attention(x_col)
            y_dis = self.attention_p(x_dis)
            if self.cfg['head']['compact']:
                y_dis = y_dis.view(n, h * w, h * w).transpose(1, 2).view(n, h * w, h, w)
            else:
                y_col = self.psamask_collect(y_col)
                y_dis = self.psamask_distribute(y_dis)
            if self.cfg['head']['psa_softmax']:
                y_col = F.softmax(y_col, dim=1)
                y_dis = F.softmax(y_dis, dim=1)
            x_col = torch.bmm(x_col.view(n, c, h * w), y_col.view(n, h * w, h * w)).view(n, c, h, w) * (1.0 / self.cfg['head']['normalization_factor'])
            x_dis = torch.bmm(x_dis.view(n, c, h * w), y_dis.view(n, h * w, h * w)).view(n, c, h, w) * (1.0 / self.cfg['head']['normalization_factor'])
            out = torch.cat([x_col, x_dis], 1)
        feats = self.proj(out)
        feats = F.interpolate(feats, size=identity.shape[2:], mode='bilinear', align_corners=align_corners)
        feats = torch.cat([identity, feats], dim=1)
        seg_logits = self.decoder(feats)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            loss, losses_log_dict = self.customizepredsandlosses(seg_logits=seg_logits, annotations=data_meta.getannotations(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size)
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs


class PSPNet(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(PSPNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        ppm_cfg = {'in_channels': head_cfg['in_channels'], 'out_channels': head_cfg['feats_channels'], 'pool_scales': head_cfg['pool_scales'], 'align_corners': align_corners, 'norm_cfg': copy.deepcopy(norm_cfg), 'act_cfg': copy.deepcopy(act_cfg)}
        self.ppm_net = PyramidPoolingModule(**ppm_cfg)
        self.decoder = nn.Sequential(nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.setauxiliarydecoder(cfg['auxiliary'])
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        ppm_out = self.ppm_net(backbone_outputs[-1])
        seg_logits = self.decoder(ppm_out)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            loss, losses_log_dict = self.customizepredsandlosses(seg_logits=seg_logits, annotations=data_meta.getannotations(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size)
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs


class MaskDecoder(nn.Module):

    def __init__(self, *, transformer_dim, transformer, num_multimask_outputs=3, act_cfg={'type': 'GELU'}, iou_head_depth=3, iou_head_hidden_dim=256, use_high_res_features=False, iou_prediction_use_sigmoid=False, dynamic_multimask_via_stability=False, dynamic_multimask_stability_delta=0.05, dynamic_multimask_stability_thresh=0.98, pred_obj_scores=False, pred_obj_scores_mlp=False, use_multimask_token_for_obj_ptr=False):
        super(MaskDecoder, self).__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.output_upscaling = nn.Sequential(nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2), LayerNorm2d(transformer_dim // 4), BuildActivation(act_cfg=act_cfg), nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2), BuildActivation(act_cfg=act_cfg))
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(transformer_dim, transformer_dim // 8, kernel_size=1, stride=1)
            self.conv_s1 = nn.Conv2d(transformer_dim, transformer_dim // 4, kernel_size=1, stride=1)
        self.output_hypernetworks_mlps = nn.ModuleList([MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) for _ in range(self.num_mask_tokens)])
        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth, sigmoid_output=iou_prediction_use_sigmoid)
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh
    """forward"""

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output, repeat_image, high_res_features=None):
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predictmasks(image_embeddings=image_embeddings, image_pe=image_pe, sparse_prompt_embeddings=sparse_prompt_embeddings, dense_prompt_embeddings=dense_prompt_embeddings, repeat_image=repeat_image, high_res_features=high_res_features)
        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self.dynamicmultimaskviastability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]
        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]
        else:
            sam_tokens_out = mask_tokens_out[:, 0:1]
        return masks, iou_pred, sam_tokens_out, object_score_logits
    """predictmasks"""

    def predictmasks(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, repeat_image, high_res_features=None):
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat([self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight], dim=0)
            s = 1
        else:
            output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert image_pe.size(0) == 1, 'image_pe should have size 1 in batch dim (from `getdensepe()`)'
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1:s + 1 + self.num_mask_tokens, :]
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)
        return masks, iou_pred, mask_tokens_out, object_score_logits
    """getstabilityscores"""

    def getstabilityscores(self, mask_logits):
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores
    """dynamicmultimaskviastability"""

    def dynamicmultimaskviastability(self, all_mask_logits, all_iou_scores):
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(multimask_iou_scores.size(0), device=all_iou_scores.device)
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self.getstabilityscores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh
        mask_logits_out = torch.where(is_stable[..., None, None].expand_as(singlemask_logits), singlemask_logits, best_multimask_logits)
        iou_scores_out = torch.where(is_stable.expand_as(singlemask_iou_scores), singlemask_iou_scores, best_multimask_iou_scores)
        return mask_logits_out, iou_scores_out


class PromptEncoder(nn.Module):

    def __init__(self, embed_dim, image_embedding_size, input_image_size, mask_in_chans, act_cfg={'type': 'GELU'}):
        super(PromptEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.num_point_embeddings = 4
        point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.mask_input_size = 4 * image_embedding_size[0], 4 * image_embedding_size[1]
        self.mask_downscaling = nn.Sequential(nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2), LayerNorm2d(mask_in_chans // 4), BuildActivation(act_cfg=act_cfg), nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2), LayerNorm2d(mask_in_chans), BuildActivation(act_cfg=act_cfg), nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1))
        self.no_mask_embed = nn.Embedding(1, embed_dim)
    """getdensepe"""

    def getdensepe(self):
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)
    """embedpoints"""

    def embedpoints(self, points, labels, pad):
        points = points + 0.5
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forwardwithcoords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        point_embedding[labels == 2] += self.point_embeddings[2].weight
        point_embedding[labels == 3] += self.point_embeddings[3].weight
        return point_embedding
    """embedboxes"""

    def embedboxes(self, boxes):
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forwardwithcoords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding
    """embedmasks"""

    def embedmasks(self, masks):
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding
    """getbatchsize"""

    def getbatchsize(self, points, boxes, masks):
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1
    """getdevice"""

    def getdevice(self):
        return self.point_embeddings[0].weight.device
    """forward"""

    def forward(self, points, boxes, masks):
        bs = self.getbatchsize(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self.getdevice())
        if points is not None:
            coords, labels = points
            point_embeddings = self.embedpoints(coords, labels, pad=boxes is None)
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self.embedboxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        if masks is not None:
            dense_embeddings = self.embedmasks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(bs, -1, self.image_embedding_size[0], self.image_embedding_size[1])
        return sparse_embeddings, dense_embeddings


class SAM(BaseSegmentor):
    mask_threshold = 0.0
    image_format = 'RGB'

    def __init__(self, cfg, mode):
        backbone = cfg.pop('backbone')
        super(SAM, self).__init__(cfg=cfg, mode=mode)
        cfg['backbone'] = backbone
        assert mode in ['TEST'], f'only support TEST mode for {self.__class__.__name__}'
        pixel_mean = cfg.get('pixel_mean', [123.675, 116.28, 103.53])
        pixel_std = cfg.get('pixel_std', [58.395, 57.12, 57.375])
        self.image_encoder = BuildBackbone(cfg['backbone'])
        self.prompt_encoder = PromptEncoder(**cfg['prompt'])
        self.mask_decoder = MaskDecoder(**cfg['head'])
        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(-1, 1, 1), False)
    """device"""

    @property
    def device(self):
        return self.pixel_mean.device
    """forward"""

    def forward(self, data_meta):
        raise NotImplementedError(f'train {self.__class__.__name__} not to be implemented')
    """inference"""

    @torch.no_grad()
    def inference(self, batched_input, multimask_output=False):
        input_images = torch.stack([self.preprocess(x['image']) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)
        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if 'point_coords' in image_record:
                points = image_record['point_coords'], image_record['point_labels']
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(points=points, boxes=image_record.get('boxes', None), masks=image_record.get('mask_inputs', None))
            low_res_masks, iou_predictions = self.mask_decoder(image_embeddings=curr_embedding.unsqueeze(0), image_pe=self.prompt_encoder.getdensepe(), sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=multimask_output)
            masks = self.postprocessmasks(low_res_masks, input_size=image_record['image'].shape[-2:], original_size=image_record['original_size'])
            masks = masks > self.mask_threshold
            outputs.append({'masks': masks, 'iou_predictions': iou_predictions, 'low_res_logits': low_res_masks})
        return outputs
    """postprocessmasks"""

    def postprocessmasks(self, masks, input_size, original_size):
        masks = F.interpolate(masks, (self.image_encoder.img_size, self.image_encoder.img_size), mode='bilinear', align_corners=False)
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(masks, original_size, mode='bilinear', align_corners=False)
        return masks
    """preprocess"""

    def preprocess(self, x):
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


class ResizeLongestSide:

    def __init__(self, target_length):
        self.target_length = target_length
    """applyimage"""

    def applyimage(self, image):
        target_size = self.getpreprocessshape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))
    """applycoords"""

    def applycoords(self, coords, original_size):
        old_h, old_w = original_size
        new_h, new_w = self.getpreprocessshape(original_size[0], original_size[1], self.target_length)
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords
    """applyboxes"""

    def applyboxes(self, boxes, original_size):
        boxes = self.applycoords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)
    """applyimagetorch"""

    def applyimagetorch(self, image):
        target_size = self.getpreprocessshape(image.shape[2], image.shape[3], self.target_length)
        return F.interpolate(image, target_size, mode='bilinear', align_corners=False, antialias=True)
    """applycoordstorch"""

    def applycoordstorch(self, coords, original_size):
        old_h, old_w = original_size
        new_h, new_w = self.getpreprocessshape(original_size[0], original_size[1], self.target_length)
        coords = deepcopy(coords)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords
    """applyboxestorch"""

    def applyboxestorch(self, boxes, original_size):
        boxes = self.applycoordstorch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)
    """getpreprocessshape"""

    @staticmethod
    def getpreprocessshape(oldh, oldw, long_side_length):
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return newh, neww


class SAMPredictor(nn.Module):

    def __init__(self, sam_cfg=None, use_default_sam_h=False, use_default_sam_l=False, use_default_sam_b=False, device='cuda', load_ckpt_strict=True):
        super(SAMPredictor, self).__init__()
        if sam_cfg is None:
            sam_cfg = {'backbone': {'depth': None, 'embed_dim': None, 'img_size': 1024, 'mlp_ratio': 4, 'norm_cfg': {'type': 'LayerNorm', 'eps': 1e-06}, 'num_heads': None, 'patch_size': 16, 'qkv_bias': True, 'use_rel_pos': True, 'global_attn_indexes': None, 'window_size': 14, 'out_chans': 256, 'type': 'SAMViT'}, 'prompt': {'embed_dim': 256, 'image_embedding_size': (1024 // 16, 1024 // 16), 'input_image_size': (1024, 1024), 'mask_in_chans': 16}, 'head': {'num_multimask_outputs': 3, 'transformer_cfg': {'depth': 2, 'embedding_dim': 256, 'mlp_dim': 2048, 'num_heads': 8}, 'transformer_dim': 256, 'iou_head_depth': 3, 'iou_head_hidden_dim': 256}}
            if use_default_sam_h:
                assert not use_default_sam_l and not use_default_sam_b
                sam_cfg['backbone']['depth'] = 32
                sam_cfg['backbone']['embed_dim'] = 1280
                sam_cfg['backbone']['num_heads'] = 16
                sam_cfg['backbone']['global_attn_indexes'] = [7, 15, 23, 31]
                sam_cfg['ckptpath'] = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
            if use_default_sam_l:
                assert not use_default_sam_h and not use_default_sam_b
                sam_cfg['backbone']['depth'] = 24
                sam_cfg['backbone']['embed_dim'] = 1024
                sam_cfg['backbone']['num_heads'] = 16
                sam_cfg['backbone']['global_attn_indexes'] = [5, 11, 17, 23]
                sam_cfg['ckptpath'] = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth'
            if use_default_sam_b:
                assert not use_default_sam_h and not use_default_sam_l
                sam_cfg['backbone']['depth'] = 12
                sam_cfg['backbone']['embed_dim'] = 768
                sam_cfg['backbone']['num_heads'] = 12
                sam_cfg['backbone']['global_attn_indexes'] = [2, 5, 8, 11]
                sam_cfg['ckptpath'] = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
        else:
            assert not use_default_sam_h and not use_default_sam_l and not use_default_sam_b
        self.model = self.buildsam(sam_cfg, device)
        if 'ckptpath' in sam_cfg and (os.path.exists(sam_cfg['ckptpath']) or sam_cfg['ckptpath'].startswith('https')):
            if os.path.exists(sam_cfg['ckptpath']):
                with open(sam_cfg['ckptpath'], 'rb') as fp:
                    state_dict = torch.load(fp)
            elif sam_cfg['ckptpath'].startswith('https'):
                state_dict = model_zoo.load_url(sam_cfg['ckptpath'])
            else:
                raise ValueError('ckptpath %s could not be loaded' % sam_cfg['ckptpath'])
            self.model.load_state_dict(state_dict, strict=load_ckpt_strict)
        self.transform = ResizeLongestSide(self.model.image_encoder.img_size)
        self.resetimage()
    """buildsam"""

    def buildsam(self, sam_cfg, device):
        sam_model = SAM(sam_cfg, mode='TEST')
        sam_model
        sam_model.eval()
        return sam_model
    """setimage"""

    def setimage(self, image, image_format='RGB'):
        assert image_format in ['RGB', 'BGR'], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]
        input_image = self.transform.applyimage(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        self.settorchimage(input_image_torch, image.shape[:2])
    """settorchimage"""

    @torch.no_grad()
    def settorchimage(self, transformed_image, original_image_size):
        assert len(transformed_image.shape) == 4 and transformed_image.shape[1] == 3 and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size, f'set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}.'
        self.resetimage()
        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        self.features = self.model.image_encoder(input_image)
        self.is_image_set = True
    """predict"""

    def predict(self, point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=True, return_logits=False):
        if not self.is_image_set:
            raise RuntimeError('an image must be set with .set_image(...) before mask prediction')
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert point_labels is not None, 'point_labels must be supplied if point_coords is supplied.'
            point_coords = self.transform.applycoords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.applyboxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]
        masks, iou_predictions, low_res_masks = self.predicttorch(coords_torch, labels_torch, box_torch, mask_input_torch, multimask_output, return_logits=return_logits)
        masks_np = masks[0].detach().cpu().numpy()
        iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
        low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np
    """predicttorch"""

    @torch.no_grad()
    def predicttorch(self, point_coords, point_labels, boxes=None, mask_input=None, multimask_output=True, return_logits=False):
        if not self.is_image_set:
            raise RuntimeError('an image must be set with .set_image(...) before mask prediction.')
        if point_coords is not None:
            points = point_coords, point_labels
        else:
            points = None
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=points, boxes=boxes, masks=mask_input)
        low_res_masks, iou_predictions = self.model.mask_decoder(image_embeddings=self.features, image_pe=self.model.prompt_encoder.getdensepe(), sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=multimask_output)
        masks = self.model.postprocessmasks(low_res_masks, self.input_size, self.original_size)
        if not return_logits:
            masks = masks > self.model.mask_threshold
        return masks, iou_predictions, low_res_masks
    """getimageembedding"""

    def getimageembedding(self):
        if not self.is_image_set:
            raise RuntimeError('an image must be set with .set_image(...) to generate an embedding.')
        assert self.features is not None, 'features must exist if an image has been set.'
        return self.features
    """device"""

    @property
    def device(self):
        return self.model.device
    """resetimage"""

    def resetimage(self):
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None


class MaskData:

    def __init__(self, **kwargs):
        for v in kwargs.values():
            assert isinstance(v, (list, np.ndarray, torch.Tensor)), 'MaskData only supports list, numpy arrays, and torch tensors.'
        self._stats = dict(**kwargs)
    """setitem"""

    def __setitem__(self, key, item):
        assert isinstance(item, (list, np.ndarray, torch.Tensor)), 'MaskData only supports list, numpy arrays, and torch tensors.'
        self._stats[key] = item
    """delitem"""

    def __delitem__(self, key):
        del self._stats[key]
    """getitem"""

    def __getitem__(self, key):
        return self._stats[key]
    """items"""

    def items(self):
        return self._stats.items()
    """filter"""

    def filter(self, keep):
        for k, v in self._stats.items():
            if v is None:
                self._stats[k] = None
            elif isinstance(v, torch.Tensor):
                self._stats[k] = v[torch.as_tensor(keep, device=v.device)]
            elif isinstance(v, np.ndarray):
                self._stats[k] = v[keep.detach().cpu().numpy()]
            elif isinstance(v, list) and keep.dtype == torch.bool:
                self._stats[k] = [a for i, a in enumerate(v) if keep[i]]
            elif isinstance(v, list):
                self._stats[k] = [v[i] for i in keep]
            else:
                raise TypeError(f'MaskData key {k} has an unsupported type {type(v)}.')
    """cat"""

    def cat(self, new_stats):
        for k, v in new_stats.items():
            if k not in self._stats or self._stats[k] is None:
                self._stats[k] = copy.deepcopy(v)
            elif isinstance(v, torch.Tensor):
                self._stats[k] = torch.cat([self._stats[k], v], dim=0)
            elif isinstance(v, np.ndarray):
                self._stats[k] = np.concatenate([self._stats[k], v], axis=0)
            elif isinstance(v, list):
                self._stats[k] = self._stats[k] + copy.deepcopy(v)
            else:
                raise TypeError(f'MaskData key {k} has an unsupported type {type(v)}.')
    """tonumpy"""

    def tonumpy(self):
        for k, v in self._stats.items():
            if isinstance(v, torch.Tensor):
                self._stats[k] = v.float().detach().cpu().numpy()


def areafromrle(rle):
    return sum(rle['counts'][1::2])


def batchedmasktobox(masks):
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)
    else:
        masks = masks.unsqueeze(0)
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * ~in_height
    top_edges, _ = torch.min(in_height_coords, dim=-1)
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * ~in_width
    left_edges, _ = torch.min(in_width_coords, dim=-1)
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)
    if len(shape) > 2:
        out = out.reshape(*shape[:-2], 4)
    else:
        out = out[0]
    return out


def batchiterator(batch_size, *args):
    assert len(args) > 0 and all(len(a) == len(args[0]) for a in args), 'Batched iteration must have inputs of all the same size.'
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size:(b + 1) * batch_size] for arg in args]


def boxxyxytoxywh(box_xyxy):
    box_xywh = copy.deepcopy(box_xyxy)
    box_xywh[2] = box_xywh[2] - box_xywh[0]
    box_xywh[3] = box_xywh[3] - box_xywh[1]
    return box_xywh


def buildpointgrid(n_per_side):
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points


def buildalllayerpointgrids(n_per_side, n_layers, scale_per_layer):
    points_by_layer = []
    for i in range(n_layers + 1):
        n_points = int(n_per_side / scale_per_layer ** i)
        points_by_layer.append(buildpointgrid(n_points))
    return points_by_layer


def calculatestabilityscore(masks, mask_threshold, threshold_offset):
    intersections = (masks > mask_threshold + threshold_offset).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    unions = (masks > mask_threshold - threshold_offset).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    return intersections / unions


def cocoencoderle(uncompressed_rle):
    h, w = uncompressed_rle['size']
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def generatecropboxes(im_size, n_layers, overlap_ratio):
    crop_boxes, layer_idxs = [], []
    im_h, im_w = im_size
    short_side = min(im_h, im_w)
    crop_boxes.append([0, 0, im_w, im_h])
    layer_idxs.append(0)

    def croplen(orig_len, n_crops, overlap):
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))
    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))
        crop_w = croplen(im_w, n_crops_per_side, overlap)
        crop_h = croplen(im_h, n_crops_per_side, overlap)
        crop_box_x0 = [int((crop_w - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_h - overlap) * i) for i in range(n_crops_per_side)]
        for x0, y0 in product(crop_box_x0, crop_box_y0):
            box = [x0, y0, min(x0 + crop_w, im_w), min(y0 + crop_h, im_h)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)
    return crop_boxes, layer_idxs


def uncropboxesxyxy(boxes, crop_box):
    x0, y0, _, _ = crop_box
    offset = torch.tensor([[x0, y0, x0, y0]], device=boxes.device)
    if len(boxes.shape) == 3:
        offset = offset.unsqueeze(1)
    return boxes + offset


def isboxnearcropedge(boxes, crop_box, orig_box, atol=20.0):
    crop_box_torch = torch.as_tensor(crop_box, dtype=torch.float, device=boxes.device)
    orig_box_torch = torch.as_tensor(orig_box, dtype=torch.float, device=boxes.device)
    boxes = uncropboxesxyxy(boxes, crop_box).float()
    near_crop_edge = torch.isclose(boxes, crop_box_torch[None, :], atol=atol, rtol=0)
    near_image_edge = torch.isclose(boxes, orig_box_torch[None, :], atol=atol, rtol=0)
    near_crop_edge = torch.logical_and(near_crop_edge, ~near_image_edge)
    return torch.any(near_crop_edge, dim=1)


def masktorlepytorch(tensor):
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat([torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device), cur_idxs + 1, torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device)])
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({'size': [h, w], 'counts': counts})
    return out


def removesmallregions(mask, area_thresh, mode):
    assert mode in ['holes', 'islands']
    correct_holes = mode == 'holes'
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]
    small_regions = [(i + 1) for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True


def rletomask(rle):
    h, w = rle['size']
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle['counts']:
        mask[idx:idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()


def uncropmasks(masks, crop_box, orig_h, orig_w):
    x0, y0, x1, y1 = crop_box
    if x0 == 0 and y0 == 0 and x1 == orig_w and y1 == orig_h:
        return masks
    pad_x, pad_y = orig_w - (x1 - x0), orig_h - (y1 - y0)
    pad = x0, pad_x - x0, y0, pad_y - y0
    return torch.nn.functional.pad(masks, pad, value=0)


def uncroppoints(points, crop_box):
    x0, y0, _, _ = crop_box
    offset = torch.tensor([[x0, y0]], device=points.device)
    if len(points.shape) == 3:
        offset = offset.unsqueeze(1)
    return points + offset


class SAMAutomaticMaskGenerator(nn.Module):

    def __init__(self, points_per_side=32, points_per_batch=64, pred_iou_thresh=0.88, stability_score_thresh=0.95, stability_score_offset=1.0, device='cuda', box_nms_thresh=0.7, crop_n_layers=0, crop_nms_thresh=0.7, crop_overlap_ratio=512 / 1500, crop_n_points_downscale_factor=1, point_grids=None, min_mask_region_area=0, output_mode='binary_mask', sam_cfg=None, use_default_sam_h=False, use_default_sam_l=False, use_default_sam_b=False, user_defined_sam_predictor=None, load_ckpt_strict=True):
        super(SAMAutomaticMaskGenerator, self).__init__()
        assert (points_per_side is None) != (point_grids is None), 'exactly one of points_per_side or point_grid must be provided.'
        assert output_mode in ['binary_mask', 'uncompressed_rle', 'coco_rle'], f'unknown output_mode {output_mode}.'
        if points_per_side is not None:
            self.point_grids = buildalllayerpointgrids(points_per_side, crop_n_layers, crop_n_points_downscale_factor)
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("can't have both points_per_side and point_grid be None")
        if user_defined_sam_predictor is not None:
            self.predictor = user_defined_sam_predictor
        else:
            self.predictor = SAMPredictor(sam_cfg, use_default_sam_h, use_default_sam_l, use_default_sam_b, device=device, load_ckpt_strict=load_ckpt_strict)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode
    """generate"""

    @torch.no_grad()
    def generate(self, image):
        mask_data = self.generatemasks(image)
        if self.min_mask_region_area > 0:
            mask_data = self.postprocesssmallregions(mask_data, self.min_mask_region_area, max(self.box_nms_thresh, self.crop_nms_thresh))
        if self.output_mode == 'coco_rle':
            mask_data['segmentations'] = [cocoencoderle(rle) for rle in mask_data['rles']]
        elif self.output_mode == 'binary_mask':
            mask_data['segmentations'] = [rletomask(rle) for rle in mask_data['rles']]
        else:
            mask_data['segmentations'] = mask_data['rles']
        curr_anns = []
        for idx in range(len(mask_data['segmentations'])):
            ann = {'segmentation': mask_data['segmentations'][idx], 'area': areafromrle(mask_data['rles'][idx]), 'bbox': boxxyxytoxywh(mask_data['boxes'][idx]).tolist(), 'predicted_iou': mask_data['iou_preds'][idx].item(), 'point_coords': [mask_data['points'][idx].tolist()], 'stability_score': mask_data['stability_score'][idx].item(), 'crop_box': boxxyxytoxywh(mask_data['crop_boxes'][idx]).tolist()}
            curr_anns.append(ann)
        return curr_anns
    """generatemasks"""

    def generatemasks(self, image):
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generatecropboxes(orig_size, self.crop_n_layers, self.crop_overlap_ratio)
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self.processcrop(image, crop_box, layer_idx, orig_size)
            data.cat(crop_data)
        if len(crop_boxes) > 1:
            scores = 1 / box_area(data['crop_boxes'])
            scores = scores
            keep_by_nms = batched_nms(data['boxes'].float(), scores, torch.zeros_like(data['boxes'][:, 0]), iou_threshold=self.crop_nms_thresh)
            data.filter(keep_by_nms)
        data.tonumpy()
        return data
    """processcrop"""

    def processcrop(self, image, crop_box, crop_layer_idx, orig_size):
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        self.predictor.setimage(cropped_im)
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale
        data = MaskData()
        for points, in batchiterator(self.points_per_batch, points_for_image):
            batch_data = self.processbatch(points, cropped_im_size, crop_box, orig_size)
            data.cat(batch_data)
            del batch_data
        self.predictor.resetimage()
        keep_by_nms = batched_nms(data['boxes'].float(), data['iou_preds'], torch.zeros_like(data['boxes'][:, 0]), iou_threshold=self.box_nms_thresh)
        data.filter(keep_by_nms)
        data['boxes'] = uncropboxesxyxy(data['boxes'], crop_box)
        data['points'] = uncroppoints(data['points'], crop_box)
        data['crop_boxes'] = torch.tensor([crop_box for _ in range(len(data['rles']))])
        return data
    """processbatch"""

    def processbatch(self, points, im_size, crop_box, orig_size):
        orig_h, orig_w = orig_size
        transformed_points = self.predictor.transform.applycoords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        masks, iou_preds, _ = self.predictor.predicttorch(in_points[:, None, :], in_labels[:, None], multimask_output=True, return_logits=True)
        data = MaskData(masks=masks.flatten(0, 1), iou_preds=iou_preds.flatten(0, 1), points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)))
        del masks
        if self.pred_iou_thresh > 0.0:
            keep_mask = data['iou_preds'] > self.pred_iou_thresh
            data.filter(keep_mask)
        data['stability_score'] = calculatestabilityscore(data['masks'], self.predictor.model.mask_threshold, self.stability_score_offset)
        if self.stability_score_thresh > 0.0:
            keep_mask = data['stability_score'] >= self.stability_score_thresh
            data.filter(keep_mask)
        data['masks'] = data['masks'] > self.predictor.model.mask_threshold
        data['boxes'] = batchedmasktobox(data['masks'])
        keep_mask = ~isboxnearcropedge(data['boxes'], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)
        data['masks'] = uncropmasks(data['masks'], crop_box, orig_h, orig_w)
        data['rles'] = masktorlepytorch(data['masks'])
        del data['masks']
        return data
    """postprocesssmallregions"""

    @staticmethod
    def postprocesssmallregions(mask_data, min_area, nms_thresh):
        if len(mask_data['rles']) == 0:
            return mask_data
        new_masks = []
        scores = []
        for rle in mask_data['rles']:
            mask = rletomask(rle)
            mask, changed = removesmallregions(mask, min_area, mode='holes')
            unchanged = not changed
            mask, changed = removesmallregions(mask, min_area, mode='islands')
            unchanged = unchanged and not changed
            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            scores.append(float(unchanged))
        masks = torch.cat(new_masks, dim=0)
        boxes = batchedmasktobox(masks)
        keep_by_nms = batched_nms(boxes.float(), torch.as_tensor(scores), torch.zeros_like(boxes[:, 0]), iou_threshold=nms_thresh)
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data['rles'][i_mask] = masktorlepytorch(mask_torch)[0]
                mask_data['boxes'][i_mask] = boxes[i_mask]
        mask_data.filter(keep_by_nms)
        return mask_data


class TwoWayAttentionBlock(nn.Module):

    def __init__(self, embedding_dim, num_heads, mlp_dim=2048, act_cfg={'type': 'ReLU'}, attention_downsample_rate=2, skip_first_layer_pe=False):
        super(TwoWayAttentionBlock, self).__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, mlp_dim, embedding_dim, num_layers=2, act_cfg=act_cfg)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.skip_first_layer_pe = skip_first_layer_pe
    """forward"""

    def forward(self, queries, keys, query_pe, key_pe):
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        return queries, keys


class TwoWayTransformer(nn.Module):

    def __init__(self, depth, embedding_dim, num_heads, mlp_dim, act_cfg={'type': 'ReLU'}, attention_downsample_rate=2):
        super(TwoWayTransformer, self).__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(TwoWayAttentionBlock(embedding_dim=embedding_dim, num_heads=num_heads, mlp_dim=mlp_dim, act_cfg=act_cfg, attention_downsample_rate=attention_downsample_rate, skip_first_layer_pe=i == 0))
        self.final_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)
    """forward"""

    def forward(self, image_embedding, image_pe, point_embedding):
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        queries = point_embedding
        keys = image_embedding
        for layer in self.layers:
            queries, keys = layer(queries=queries, keys=keys, query_pe=point_embedding, key_pe=image_pe)
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        return queries, keys


class MaskDecoderHQ(MaskDecoder):

    def __init__(self, transformer_dim, transformer_cfg, num_multimask_outputs=3, act_cfg={'type': 'GELU'}, iou_head_depth=3, iou_head_hidden_dim=256, vit_dim=1024):
        super(MaskDecoderHQ, self).__init__(transformer_dim=transformer_dim, transformer_cfg=transformer_cfg, num_multimask_outputs=num_multimask_outputs, act_cfg=act_cfg, iou_head_depth=iou_head_depth, iou_head_hidden_dim=iou_head_hidden_dim)
        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1
        self.compress_vit_feat = nn.Sequential(nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2), LayerNorm2d(transformer_dim), nn.GELU(), nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))
        self.embedding_encoder = nn.Sequential(nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2), LayerNorm2d(transformer_dim // 4), nn.GELU(), nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2))
        self.embedding_maskfeature = nn.Sequential(nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1), LayerNorm2d(transformer_dim // 4), nn.GELU(), nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1))
    """forward"""

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output, hq_token_only, interm_embeddings):
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2)
        hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)
        masks, iou_pred = self.predictmasks(image_embeddings=image_embeddings, image_pe=image_pe, sparse_prompt_embeddings=sparse_prompt_embeddings, dense_prompt_embeddings=dense_prompt_embeddings, hq_features=hq_features)
        if multimask_output:
            mask_slice = slice(1, self.num_mask_tokens - 1)
            iou_pred = iou_pred[:, mask_slice]
            iou_pred, max_iou_idx = torch.max(iou_pred, dim=1)
            iou_pred = iou_pred.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[torch.arange(masks_multi.size(0)), max_iou_idx].unsqueeze(1)
        else:
            mask_slice = slice(0, 1)
            iou_pred = iou_pred[:, mask_slice]
            masks_sam = masks[:, mask_slice]
        masks_hq = masks[:, slice(self.num_mask_tokens - 1, self.num_mask_tokens)]
        if hq_token_only:
            masks = masks_hq
        else:
            masks = masks_sam + masks_hq
        return masks, iou_pred
    """predictmasks"""

    def predictmasks(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, hq_features):
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:1 + self.num_mask_tokens, :]
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_hq = self.embedding_maskfeature(upscaled_embedding_sam) + hq_features.repeat(b, 1, 1, 1)
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            if i < self.num_mask_tokens - 1:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape
        masks_sam = (hyper_in[:, :self.num_mask_tokens - 1] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_sam_hq = (hyper_in[:, self.num_mask_tokens - 1:] @ upscaled_embedding_hq.view(b, c, h * w)).view(b, -1, h, w)
        masks = torch.cat([masks_sam, masks_sam_hq], dim=1)
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred


class SAMHQ(SAM):
    mask_threshold = 0.0
    image_format = 'RGB'

    def __init__(self, cfg, mode):
        vit_dim = cfg['head'].pop('vit_dim')
        super(SAMHQ, self).__init__(cfg=cfg, mode=mode)
        cfg['head']['vit_dim'] = vit_dim
        self.mask_decoder = MaskDecoderHQ(**cfg['head'])
    """inference"""

    @torch.no_grad()
    def inference(self, batched_input, multimask_output=False, hq_token_only=False):
        input_images = torch.stack([self.preprocess(x['image']) for x in batched_input], dim=0)
        image_embeddings, interm_embeddings = self.image_encoder(input_images, return_interm_embeddings=True)
        interm_embeddings = interm_embeddings[0]
        outputs = []
        for image_record, curr_embedding, curr_interm in zip(batched_input, image_embeddings, interm_embeddings):
            if 'point_coords' in image_record:
                points = image_record['point_coords'], image_record['point_labels']
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(points=points, boxes=image_record.get('boxes', None), masks=image_record.get('mask_inputs', None))
            low_res_masks, iou_predictions = self.mask_decoder(image_embeddings=curr_embedding.unsqueeze(0), image_pe=self.prompt_encoder.getdensepe(), sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=multimask_output, hq_token_only=hq_token_only, interm_embeddings=curr_interm.unsqueeze(0).unsqueeze(0))
            masks = self.postprocessmasks(low_res_masks, input_size=image_record['image'].shape[-2:], original_size=image_record['original_size'])
            masks = masks > self.mask_threshold
            outputs.append({'masks': masks, 'iou_predictions': iou_predictions, 'low_res_logits': low_res_masks})
        return outputs


class SAMHQPredictor(SAMPredictor):

    def __init__(self, sam_cfg=None, use_default_samhq_t_5m=False, use_default_samhq_b=False, use_default_samhq_l=False, use_default_samhq_h=False, device='cuda', load_ckpt_strict=True):
        if sam_cfg is None:
            sam_cfg = {'backbone': {'depth': None, 'embed_dim': None, 'img_size': 1024, 'mlp_ratio': 4, 'norm_cfg': {'type': 'LayerNorm', 'eps': 1e-06}, 'num_heads': None, 'patch_size': 16, 'qkv_bias': True, 'use_rel_pos': True, 'global_attn_indexes': None, 'window_size': 14, 'out_chans': 256, 'type': 'SAMViT'}, 'prompt': {'embed_dim': 256, 'image_embedding_size': (1024 // 16, 1024 // 16), 'input_image_size': (1024, 1024), 'mask_in_chans': 16}, 'head': {'num_multimask_outputs': 3, 'transformer_cfg': {'depth': 2, 'embedding_dim': 256, 'mlp_dim': 2048, 'num_heads': 8}, 'transformer_dim': 256, 'iou_head_depth': 3, 'iou_head_hidden_dim': 256, 'vit_dim': None}}
            if use_default_samhq_h:
                assert not use_default_samhq_b and not use_default_samhq_l and not use_default_samhq_t_5m
                sam_cfg['backbone']['depth'] = 32
                sam_cfg['backbone']['embed_dim'] = 1280
                sam_cfg['backbone']['num_heads'] = 16
                sam_cfg['backbone']['global_attn_indexes'] = [7, 15, 23, 31]
                sam_cfg['head']['vit_dim'] = 1280
                sam_cfg['ckptpath'] = 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth'
            if use_default_samhq_l:
                assert not use_default_samhq_b and not use_default_samhq_h and not use_default_samhq_t_5m
                sam_cfg['backbone']['depth'] = 24
                sam_cfg['backbone']['embed_dim'] = 1024
                sam_cfg['backbone']['num_heads'] = 16
                sam_cfg['backbone']['global_attn_indexes'] = [5, 11, 17, 23]
                sam_cfg['head']['vit_dim'] = 1024
                sam_cfg['ckptpath'] = 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth'
            if use_default_samhq_b:
                assert not use_default_samhq_l and not use_default_samhq_h and not use_default_samhq_t_5m
                sam_cfg['backbone']['depth'] = 12
                sam_cfg['backbone']['embed_dim'] = 768
                sam_cfg['backbone']['num_heads'] = 12
                sam_cfg['backbone']['global_attn_indexes'] = [2, 5, 8, 11]
                sam_cfg['head']['vit_dim'] = 768
                sam_cfg['ckptpath'] = 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth'
            if use_default_samhq_t_5m:
                assert not use_default_samhq_b and not use_default_samhq_l and not use_default_samhq_h
                sam_cfg['backbone'] = {'structure_type': 'tiny_vit_5m_22kto1k_distill', 'img_size': 1024, 'in_chans': 3, 'embed_dims': [64, 128, 160, 320], 'depths': [2, 2, 6, 2], 'num_heads': [2, 4, 5, 10], 'window_sizes': [7, 7, 14, 7], 'mlp_ratio': 4.0, 'drop_rate': 0.0, 'drop_path_rate': 0.0, 'use_checkpoint': False, 'mbconv_expand_ratio': 4.0, 'local_conv_size': 3, 'pretrained': False, 'pretrained_model_path': '', 'type': 'MobileSAMTinyViT'}
                sam_cfg['head']['vit_dim'] = 160
                sam_cfg['ckptpath'] = 'https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth'
                load_ckpt_strict = False
        else:
            assert not use_default_samhq_b and not use_default_samhq_l and not use_default_samhq_h and not use_default_samhq_t_5m
        super(SAMHQPredictor, self).__init__(use_default_sam_h=False, use_default_sam_l=False, use_default_sam_b=False, sam_cfg=sam_cfg, device=device, load_ckpt_strict=load_ckpt_strict)
        self.model.eval()
    """buildsam"""

    def buildsam(self, sam_cfg, device):
        sam_model = SAMHQ(sam_cfg, mode='TEST')
        sam_model
        sam_model.eval()
        return sam_model
    """settorchimage"""

    @torch.no_grad()
    def settorchimage(self, transformed_image, original_image_size):
        assert len(transformed_image.shape) == 4 and transformed_image.shape[1] == 3 and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size, f'set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}.'
        self.resetimage()
        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        self.features, self.interm_features = self.model.image_encoder(input_image, return_interm_embeddings=True)
        self.is_image_set = True
    """predict"""

    def predict(self, point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=True, return_logits=False, hq_token_only=False):
        if not self.is_image_set:
            raise RuntimeError('an image must be set with .setimage(...) before mask prediction')
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert point_labels is not None, 'point_labels must be supplied if point_coords is supplied.'
            point_coords = self.transform.applycoords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.applyboxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]
        masks, iou_predictions, low_res_masks = self.predicttorch(coords_torch, labels_torch, box_torch, mask_input_torch, multimask_output, return_logits=return_logits, hq_token_only=hq_token_only)
        masks_np = masks[0].detach().cpu().numpy()
        iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
        low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np
    """predicttorch"""

    @torch.no_grad()
    def predicttorch(self, point_coords, point_labels, boxes=None, mask_input=None, multimask_output=True, return_logits=False, hq_token_only=False):
        if not self.is_image_set:
            raise RuntimeError('an image must be set with .setimage(...) before mask prediction.')
        if point_coords is not None:
            points = point_coords, point_labels
        else:
            points = None
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=points, boxes=boxes, masks=mask_input)
        low_res_masks, iou_predictions = self.model.mask_decoder(image_embeddings=self.features, image_pe=self.model.prompt_encoder.getdensepe(), sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=multimask_output, hq_token_only=hq_token_only, interm_embeddings=self.interm_features)
        masks = self.model.postprocessmasks(low_res_masks, self.input_size, self.original_size)
        if not return_logits:
            masks = masks > self.model.mask_threshold
        return masks, iou_predictions, low_res_masks


class SAMHQAutomaticMaskGenerator(SAMAutomaticMaskGenerator):

    def __init__(self, points_per_side=32, points_per_batch=64, pred_iou_thresh=0.88, stability_score_thresh=0.95, stability_score_offset=1.0, device='cuda', box_nms_thresh=0.7, crop_n_layers=0, crop_nms_thresh=0.7, crop_overlap_ratio=512 / 1500, crop_n_points_downscale_factor=1, point_grids=None, min_mask_region_area=0, output_mode='binary_mask', sam_cfg=None, use_default_samhq_t_5m=False, use_default_samhq_b=False, use_default_samhq_l=False, use_default_samhq_h=False, load_ckpt_strict=True):
        user_defined_sam_predictor = SAMHQPredictor(sam_cfg=sam_cfg, use_default_samhq_t_5m=use_default_samhq_t_5m, use_default_samhq_b=use_default_samhq_b, use_default_samhq_l=use_default_samhq_l, use_default_samhq_h=use_default_samhq_h, device=device, load_ckpt_strict=load_ckpt_strict)
        super(SAMHQAutomaticMaskGenerator, self).__init__(points_per_side=points_per_side, points_per_batch=points_per_batch, pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh, stability_score_offset=stability_score_offset, device=device, box_nms_thresh=box_nms_thresh, crop_n_layers=crop_n_layers, crop_nms_thresh=crop_nms_thresh, crop_overlap_ratio=crop_overlap_ratio, crop_n_points_downscale_factor=crop_n_points_downscale_factor, point_grids=point_grids, min_mask_region_area=min_mask_region_area, output_mode=output_mode, sam_cfg=None, use_default_sam_h=False, use_default_sam_l=False, use_default_sam_b=False, user_defined_sam_predictor=user_defined_sam_predictor, load_ckpt_strict=load_ckpt_strict)
    """generate"""

    @torch.no_grad()
    def generate(self, image, hq_token_only=False):
        mask_data = self.generatemasks(image, hq_token_only)
        if self.min_mask_region_area > 0:
            mask_data = self.postprocesssmallregions(mask_data, self.min_mask_region_area, max(self.box_nms_thresh, self.crop_nms_thresh))
        if self.output_mode == 'coco_rle':
            mask_data['segmentations'] = [cocoencoderle(rle) for rle in mask_data['rles']]
        elif self.output_mode == 'binary_mask':
            mask_data['segmentations'] = [rletomask(rle) for rle in mask_data['rles']]
        else:
            mask_data['segmentations'] = mask_data['rles']
        curr_anns = []
        for idx in range(len(mask_data['segmentations'])):
            ann = {'segmentation': mask_data['segmentations'][idx], 'area': areafromrle(mask_data['rles'][idx]), 'bbox': boxxyxytoxywh(mask_data['boxes'][idx]).tolist(), 'predicted_iou': mask_data['iou_preds'][idx].item(), 'point_coords': [mask_data['points'][idx].tolist()], 'stability_score': mask_data['stability_score'][idx].item(), 'crop_box': boxxyxytoxywh(mask_data['crop_boxes'][idx]).tolist()}
            curr_anns.append(ann)
        return curr_anns
    """generatemasks"""

    def generatemasks(self, image, hq_token_only=False):
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generatecropboxes(orig_size, self.crop_n_layers, self.crop_overlap_ratio)
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self.processcrop(image, crop_box, layer_idx, orig_size, hq_token_only)
            data.cat(crop_data)
        if len(crop_boxes) > 1:
            scores = 1 / box_area(data['crop_boxes'])
            scores = scores
            keep_by_nms = batched_nms(data['boxes'].float(), scores, torch.zeros_like(data['boxes'][:, 0]), iou_threshold=self.crop_nms_thresh)
            data.filter(keep_by_nms)
        data.tonumpy()
        return data
    """processcrop"""

    def processcrop(self, image, crop_box, crop_layer_idx, orig_size, hq_token_only):
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        self.predictor.setimage(cropped_im)
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale
        data = MaskData()
        for points, in batchiterator(self.points_per_batch, points_for_image):
            batch_data = self.processbatch(points, cropped_im_size, crop_box, orig_size, hq_token_only)
            data.cat(batch_data)
            del batch_data
        self.predictor.resetimage()
        keep_by_nms = batched_nms(data['boxes'].float(), data['iou_preds'], torch.zeros_like(data['boxes'][:, 0]), iou_threshold=self.box_nms_thresh)
        data.filter(keep_by_nms)
        data['boxes'] = uncropboxesxyxy(data['boxes'], crop_box)
        data['points'] = uncroppoints(data['points'], crop_box)
        data['crop_boxes'] = torch.tensor([crop_box for _ in range(len(data['rles']))])
        return data
    """processbatch"""

    def processbatch(self, points, im_size, crop_box, orig_size, hq_token_only):
        orig_h, orig_w = orig_size
        transformed_points = self.predictor.transform.applycoords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        masks, iou_preds, _ = self.predictor.predicttorch(in_points[:, None, :], in_labels[:, None], multimask_output=True, return_logits=True, hq_token_only=hq_token_only)
        data = MaskData(masks=masks.flatten(0, 1), iou_preds=iou_preds.flatten(0, 1), points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)))
        del masks
        if self.pred_iou_thresh > 0.0:
            keep_mask = data['iou_preds'] > self.pred_iou_thresh
            data.filter(keep_mask)
        data['stability_score'] = calculatestabilityscore(data['masks'], self.predictor.model.mask_threshold, self.stability_score_offset)
        if self.stability_score_thresh > 0.0:
            keep_mask = data['stability_score'] >= self.stability_score_thresh
            data.filter(keep_mask)
        data['masks'] = data['masks'] > self.predictor.model.mask_threshold
        data['boxes'] = batchedmasktobox(data['masks'])
        keep_mask = ~isboxnearcropedge(data['boxes'], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)
        data['masks'] = uncropmasks(data['masks'], crop_box, orig_h, orig_w)
        data['rles'] = masktorlepytorch(data['masks'])
        del data['masks']
        return data


def reshapeforbroadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [(d if i >= ndim - 2 else 1) for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def applyrotaryenc(xq, xk, freqs_cis, repeat_freqs_k=False):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) if xk.shape[-2] != 0 else None
    freqs_cis = reshapeforbroadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        return xq_out.type_as(xq), xk
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def inittxy(end_x, end_y):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y


def computeaxialcis(dim, end_x, end_y, theta=10000.0):
    freqs_x = 1.0 / theta ** (torch.arange(0, dim, 4)[:dim // 4].float() / dim)
    freqs_y = 1.0 / theta ** (torch.arange(0, dim, 4)[:dim // 4].float() / dim)
    t_x, t_y = inittxy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


class RoPEAttention(Attention):

    def __init__(self, *args, rope_theta=10000.0, rope_k_repeat=False, feat_sizes=(32, 32), **kwargs):
        super(RoPEAttention, self).__init__(*args, **kwargs)
        self.compute_cis = partial(computeaxialcis, dim=self.internal_dim // self.num_heads, theta=rope_theta)
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis = freqs_cis
        self.rope_k_repeat = rope_k_repeat
    """forward"""

    def forward(self, q, k, v, num_k_exclude_rope=0):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = self.separateheads(q, self.num_heads)
        k = self.separateheads(k, self.num_heads)
        v = self.separateheads(v, self.num_heads)
        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat
        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = applyrotaryenc(q, k[:, :, :num_k_rope], freqs_cis=self.freqs_cis, repeat_freqs_k=self.rope_k_repeat)
        dropout_p = self.dropout_p if self.training else 0.0
        with torch.backends.cuda.sdp_kernel(enable_flash=USE_FLASH_ATTN, enable_math=OLD_GPU and dropout_p > 0.0 or MATH_KERNEL_ON, enable_mem_efficient=OLD_GPU):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        out = self.recombineheads(out)
        out = self.out_proj(out)
        return out


class AttentionBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {'Attention': Attention, 'RoPEAttention': RoPEAttention}
    """build"""

    def build(self, pe_cfg):
        return super().build(pe_cfg)


BuildAttention = AttentionBuilder().build


class MemoryAttentionLayer(nn.Module):

    def __init__(self, act_cfg, cross_attention_cfg, d_model, dim_feedforward, dropout, pos_enc_at_attn, pos_enc_at_cross_attn_keys, pos_enc_at_cross_attn_queries, self_attention_cfg):
        super(MemoryAttentionLayer, self).__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = BuildAttention(self_attention_cfg)
        self.cross_attn_image = BuildAttention(cross_attention_cfg)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = BuildActivation(act_cfg=act_cfg)
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys
    """forwardsa"""

    def forwardsa(self, tgt, query_pos):
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt
    """forwardca"""

    def forwardca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {'num_k_exclude_rope': num_k_exclude_rope}
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2, k=memory + pos if self.pos_enc_at_cross_attn_keys else memory, v=memory, **kwds)
        tgt = tgt + self.dropout2(tgt2)
        return tgt
    """forward"""

    def forward(self, tgt, memory, pos=None, query_pos=None, num_k_exclude_rope=0):
        tgt = self.forwardsa(tgt, query_pos)
        tgt = self.forwardca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class MemoryAttention(nn.Module):

    def __init__(self, d_model, pos_enc_at_input, layer_cfg, num_layers, batch_first=True):
        super(MemoryAttention, self).__init__()
        self.d_model = d_model
        layer = MemoryAttentionLayer(**layer_cfg)
        self.layers = getclones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first
    """forward"""

    def forward(self, curr, memory, curr_pos=None, memory_pos=None, num_obj_ptr_tokens=0):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = curr[0], curr_pos[0]
        assert curr.shape[1] == memory.shape[1], 'Batch size must be the same for curr and memory'
        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos
        if self.batch_first:
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)
        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {'num_k_exclude_rope': num_obj_ptr_tokens}
            output = layer(tgt=output, memory=memory, pos=memory_pos, query_pos=curr_pos, **kwds)
        normed_output = self.norm(output)
        if self.batch_first:
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
        return normed_output


class MaskDownSampler(nn.Module):

    def __init__(self, embed_dim=256, kernel_size=4, stride=4, padding=0, total_stride=16, act_cfg={'type': 'GELU'}):
        super(MaskDownSampler, self).__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        assert stride ** num_layers == total_stride
        self.encoder = []
        mask_in_chans, mask_out_chans = 1, 1
        for _ in range(num_layers):
            mask_out_chans = mask_in_chans * stride ** 2
            self.encoder.append(nn.Conv2d(mask_in_chans, mask_out_chans, kernel_size=kernel_size, stride=stride, padding=padding))
            self.encoder.append(LayerNorm2d(mask_out_chans))
            self.encoder.append(BuildActivation(act_cfg=act_cfg))
            mask_in_chans = mask_out_chans
        self.encoder.append(nn.Conv2d(mask_out_chans, embed_dim, kernel_size=1))
        self.encoder = nn.Sequential(*self.encoder)
    """forward"""

    def forward(self, x):
        return self.encoder(x)


class CXBlock(nn.Module):

    def __init__(self, dim, kernel_size=7, padding=3, drop_path=0.0, layer_scale_init_value=1e-06, use_dwconv=True):
        super(CXBlock, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim if use_dwconv else 1)
        self.norm = LayerNorm2d(dim, eps=1e-06)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    """forward"""

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


class Fuser(nn.Module):

    def __init__(self, layer_cfg, num_layers, dim=None, input_projection=False):
        super(Fuser, self).__init__()
        self.proj = nn.Identity()
        layer = CXBlock(**layer_cfg)
        self.layers = getclones(layer, num_layers)
        if input_projection:
            assert dim is not None
            self.proj = nn.Conv2d(dim, dim, kernel_size=1)
    """forward"""

    def forward(self, x):
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


class MemoryEncoder(nn.Module):

    def __init__(self, out_dim, mask_downsampler_cfg, fuser_cfg, position_encoding_cfg, in_dim=256):
        super(MemoryEncoder, self).__init__()
        self.mask_downsampler = MaskDownSampler(**mask_downsampler_cfg)
        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.fuser = Fuser(**fuser_cfg)
        self.position_encoding = BuildPE(position_encoding_cfg)
        self.out_proj = nn.Identity()
        if out_dim != in_dim:
            self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)
    """forward"""

    def forward(self, pix_feat, masks, skip_mask_sigmoid=False):
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)
        masks = self.mask_downsampler(masks)
        pix_feat = pix_feat
        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)
        pos = self.position_encoding(x)
        return {'vision_features': x, 'vision_pos_enc': [pos]}


NO_OBJ_SCORE = -1024.0


def get1dsinepe(pos_inds, dim, temperature=10000):
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)
    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed


def selectclosestcondframes(frame_idx, cond_frame_outputs, max_cond_frame_num):
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, 'we should allow using 2+ conditioning frames'
        selected_outputs = {}
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted((t for t in cond_frame_outputs if t not in selected_outputs), key=lambda x: abs(x - frame_idx))[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs}
    return selected_outputs, unselected_outputs


class SAMV2(BaseSegmentor):

    def __init__(self, cfg, mode):
        backbone = cfg.pop('backbone')
        super(SAMV2, self).__init__(cfg=cfg, mode=mode)
        cfg['backbone'] = backbone
        assert mode in ['TEST'], f'only support TEST mode for {self.__class__.__name__}'
        self.image_encoder = BuildBackbone(cfg['backbone'])
        self.use_high_res_features_in_sam = cfg['head']['use_high_res_features_in_sam']
        self.num_feature_levels = 3 if cfg['head']['use_high_res_features_in_sam'] else 1
        self.use_obj_ptrs_in_encoder = cfg['head']['use_obj_ptrs_in_encoder']
        self.max_obj_ptrs_in_encoder = cfg['head']['max_obj_ptrs_in_encoder']
        if cfg['head']['use_obj_ptrs_in_encoder']:
            self.mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.add_tpos_enc_to_obj_ptrs = cfg['head']['add_tpos_enc_to_obj_ptrs']
        if cfg['head']['proj_tpos_enc_in_obj_ptrs']:
            assert cfg['head']['add_tpos_enc_to_obj_ptrs']
        self.proj_tpos_enc_in_obj_ptrs = cfg['head']['proj_tpos_enc_in_obj_ptrs']
        self.only_obj_ptrs_in_the_past_for_eval = cfg['head']['only_obj_ptrs_in_the_past_for_eval']
        self.memory_attention = MemoryAttention(**cfg['head']['memory_attention_cfg'])
        self.hidden_dim = self.memory_attention.d_model
        self.memory_encoder = MemoryEncoder(**cfg['head']['memory_encoder_cfg'])
        self.mem_dim = self.hidden_dim
        if hasattr(self.memory_encoder, 'out_proj') and hasattr(self.memory_encoder.out_proj, 'weight'):
            self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]
        self.num_maskmem = cfg['head']['num_maskmem']
        self.maskmem_tpos_enc = nn.Parameter(torch.zeros(cfg['head']['num_maskmem'], 1, 1, self.mem_dim))
        nn.init.trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        nn.init.trunc_normal_(self.no_mem_embed, std=0.02)
        nn.init.trunc_normal_(self.no_mem_pos_enc, std=0.02)
        self.directly_add_no_mem_embed = cfg['head']['directly_add_no_mem_embed']
        self.sigmoid_scale_for_mem_enc = cfg['head']['sigmoid_scale_for_mem_enc']
        self.sigmoid_bias_for_mem_enc = cfg['head']['sigmoid_bias_for_mem_enc']
        self.binarize_mask_from_pts_for_mem_enc = cfg['head']['binarize_mask_from_pts_for_mem_enc']
        self.non_overlap_masks_for_mem_enc = cfg['head']['non_overlap_masks_for_mem_enc']
        self.memory_temporal_stride_for_eval = cfg['head']['memory_temporal_stride_for_eval']
        self.use_mask_input_as_output_without_sam = cfg['head']['use_mask_input_as_output_without_sam']
        self.multimask_output_in_sam = cfg['head']['multimask_output_in_sam']
        self.multimask_min_pt_num = cfg['head']['multimask_min_pt_num']
        self.multimask_max_pt_num = cfg['head']['multimask_max_pt_num']
        self.multimask_output_for_tracking = cfg['head']['multimask_output_for_tracking']
        self.use_multimask_token_for_obj_ptr = cfg['head']['use_multimask_token_for_obj_ptr']
        self.iou_prediction_use_sigmoid = cfg['head']['iou_prediction_use_sigmoid']
        self.image_size = cfg['head']['image_size']
        self.backbone_stride = cfg['head']['backbone_stride']
        self.sam_mask_decoder_extra_args = cfg['head']['sam_mask_decoder_extra_args']
        self.pred_obj_scores = cfg['head']['pred_obj_scores']
        self.pred_obj_scores_mlp = cfg['head']['pred_obj_scores_mlp']
        self.fixed_no_obj_ptr = cfg['head']['fixed_no_obj_ptr']
        self.soft_no_obj_ptr = cfg['head']['soft_no_obj_ptr']
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = nn.Parameter(torch.zeros(1, self.hidden_dim))
            nn.init.trunc_normal_(self.no_obj_ptr, std=0.02)
        self.use_mlp_for_obj_ptr_proj = cfg['head']['use_mlp_for_obj_ptr_proj']
        self.buildsamheads()
        self.add_all_frames_to_correct_as_cond = cfg['head']['add_all_frames_to_correct_as_cond']
        self.max_cond_frames_in_attn = cfg['head']['max_cond_frames_in_attn']
        if cfg['head']['compile_image_encoder']:
            None
            self.image_encoder.forward = torch.compile(self.image_encoder.forward, mode='max-autotune', fullgraph=True, dynamic=False)
    """device"""

    @property
    def device(self):
        return next(self.parameters()).device
    """forward"""

    def forward(self, data_meta):
        raise NotImplementedError(f'train {self.__class__.__name__} not to be implemented')
    """buildsamheads"""

    def buildsamheads(self):
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride
        self.sam_prompt_encoder = PromptEncoder(embed_dim=self.sam_prompt_embed_dim, image_embedding_size=(self.sam_image_embedding_size, self.sam_image_embedding_size), input_image_size=(self.image_size, self.image_size), mask_in_chans=16)
        self.sam_mask_decoder = MaskDecoder(num_multimask_outputs=3, transformer=TwoWayTransformer(depth=2, embedding_dim=self.sam_prompt_embed_dim, mlp_dim=2048, num_heads=8), transformer_dim=self.sam_prompt_embed_dim, iou_head_depth=3, iou_head_hidden_dim=256, use_high_res_features=self.use_high_res_features_in_sam, iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid, pred_obj_scores=self.pred_obj_scores, pred_obj_scores_mlp=self.pred_obj_scores_mlp, use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr, **self.sam_mask_decoder_extra_args or {})
        if self.use_obj_ptrs_in_encoder:
            self.obj_ptr_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)
        else:
            self.obj_ptr_proj = nn.Identity()
        if self.proj_tpos_enc_in_obj_ptrs:
            self.obj_ptr_tpos_proj = nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = nn.Identity()
    """forwardsamheads"""

    def forwardsamheads(self, backbone_features, point_inputs=None, mask_inputs=None, high_res_features=None, multimask_output=False):
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size
        if point_inputs is not None:
            sam_point_coords = point_inputs['point_coords']
            sam_point_labels = point_inputs['point_labels']
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)
        if mask_inputs is not None:
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(mask_inputs.float(), size=self.sam_prompt_encoder.mask_input_size, align_corners=False, mode='bilinear', antialias=True)
            else:
                sam_mask_prompt = mask_inputs
        else:
            sam_mask_prompt = None
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(points=(sam_point_coords, sam_point_labels), boxes=None, masks=sam_mask_prompt)
        low_res_multimasks, ious, sam_output_tokens, object_score_logits = self.sam_mask_decoder(image_embeddings=backbone_features, image_pe=self.sam_prompt_encoder.getdensepe(), sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=multimask_output, repeat_image=False, high_res_features=high_res_features)
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0
            low_res_multimasks = torch.where(is_obj_appearing[:, None, None], low_res_multimasks, NO_OBJ_SCORE)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(low_res_multimasks, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr
        return low_res_multimasks, high_res_multimasks, ious, low_res_masks, high_res_masks, obj_ptr, object_score_logits
    """usemaskasoutput"""

    def usemaskasoutput(self, backbone_features, high_res_features, mask_inputs):
        out_scale, out_bias = 20.0, -10.0
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(high_res_masks, size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4), align_corners=False, mode='bilinear', antialias=True)
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.use_obj_ptrs_in_encoder:
            obj_ptr = torch.zeros(mask_inputs.size(0), self.hidden_dim, device=mask_inputs.device)
        else:
            _, _, _, _, _, obj_ptr, _ = self.forwardsamheads(backbone_features=backbone_features, mask_inputs=self.mask_downsample(mask_inputs_float), high_res_features=high_res_features)
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr
        return low_res_masks, high_res_masks, ious, low_res_masks, high_res_masks, obj_ptr, object_score_logits
    """forwardimage"""

    def forwardimage(self, img_batch):
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            backbone_out['backbone_fpn'][0] = self.sam_mask_decoder.conv_s0(backbone_out['backbone_fpn'][0])
            backbone_out['backbone_fpn'][1] = self.sam_mask_decoder.conv_s1(backbone_out['backbone_fpn'][1])
        return backbone_out
    """preparebackbonefeatures"""

    def preparebackbonefeatures(self, backbone_out):
        backbone_out = backbone_out.copy()
        assert len(backbone_out['backbone_fpn']) == len(backbone_out['vision_pos_enc'])
        assert len(backbone_out['backbone_fpn']) >= self.num_feature_levels
        feature_maps = backbone_out['backbone_fpn'][-self.num_feature_levels:]
        vision_pos_embeds = backbone_out['vision_pos_enc'][-self.num_feature_levels:]
        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]
        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes
    """preparememoryconditionedfeatures"""

    def preparememoryconditionedfeatures(self, frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds, feat_sizes, output_dict, num_frames, track_in_reverse=False):
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device
        if self.num_maskmem == 0:
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat
        num_obj_ptr_tokens = 0
        if not is_init_cond_frame:
            to_cat_memory, to_cat_memory_pos_embed = [], []
            assert len(output_dict['cond_frame_outputs']) > 0
            cond_outputs = output_dict['cond_frame_outputs']
            selected_cond_outputs, unselected_cond_outputs = selectclosestcondframes(frame_idx, cond_outputs, self.max_cond_frames_in_attn)
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            r = self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos
                if t_rel == 1:
                    if not track_in_reverse:
                        prev_frame_idx = frame_idx - t_rel
                    else:
                        prev_frame_idx = frame_idx + t_rel
                elif not track_in_reverse:
                    prev_frame_idx = (frame_idx - 2) // r * r
                    prev_frame_idx = prev_frame_idx - (t_rel - 2) * r
                else:
                    prev_frame_idx = -(-(frame_idx + 2) // r) * r
                    prev_frame_idx = prev_frame_idx + (t_rel - 2) * r
                out = output_dict['non_cond_frame_outputs'].get(prev_frame_idx, None)
                if out is None:
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))
            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue
                feats = prev['maskmem_features']
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                maskmem_enc = prev['maskmem_pos_enc'][-1]
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                maskmem_enc = maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                to_cat_memory_pos_embed.append(maskmem_enc)
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {t: out for t, out in selected_cond_outputs.items() if (t >= frame_idx if track_in_reverse else t <= frame_idx)}
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [(abs(frame_idx - t), out['obj_ptr']) for t, out in ptr_cond_outputs.items()]
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or num_frames is not None and t >= num_frames:
                        break
                    out = output_dict['non_cond_frame_outputs'].get(t, unselected_cond_outputs.get(t, None))
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out['obj_ptr']))
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list, device=device)
                        obj_pos = get1dsinepe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        obj_ptrs = obj_ptrs.reshape(-1, B, C // self.mem_dim, self.mem_dim)
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            if self.directly_add_no_mem_embed:
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
        pix_feat_with_mem = self.memory_attention(curr=current_vision_feats, curr_pos=current_vision_pos_embeds, memory=memory, memory_pos=memory_pos_embed, num_obj_ptr_tokens=num_obj_ptr_tokens)
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem
    """encodenewmemory"""

    def encodenewmemory(self, current_vision_feats, feat_sizes, pred_masks_high_res, is_mask_from_pts):
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            pred_masks_high_res = self.applynonoverlappingconstraints(pred_masks_high_res)
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(pix_feat, mask_for_mem, skip_mask_sigmoid=True)
        maskmem_features = maskmem_out['vision_features']
        maskmem_pos_enc = maskmem_out['vision_pos_enc']
        return maskmem_features, maskmem_pos_enc
    """trackstep"""

    def trackstep(self, frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds, feat_sizes, point_inputs, mask_inputs, output_dict, num_frames, track_in_reverse=False, run_mem_encoder=True, prev_sam_mask_logits=None):
        current_out = {'point_inputs': point_inputs, 'mask_inputs': mask_inputs}
        if len(current_vision_feats) > 1:
            high_res_features = [x.permute(1, 2, 0).view(x.size(1), x.size(2), *s) for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])]
        else:
            high_res_features = None
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self.usemaskasoutput(pix_feat, high_res_features, mask_inputs)
        else:
            pix_feat_with_mem = self.preparememoryconditionedfeatures(frame_idx=frame_idx, is_init_cond_frame=is_init_cond_frame, current_vision_feats=current_vision_feats[-1:], current_vision_pos_embeds=current_vision_pos_embeds[-1:], feat_sizes=feat_sizes[-1:], output_dict=output_dict, num_frames=num_frames, track_in_reverse=track_in_reverse)
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self.usemultimask(is_init_cond_frame, point_inputs)
            sam_outputs = self.forwardsamheads(backbone_features=pix_feat_with_mem, point_inputs=point_inputs, mask_inputs=mask_inputs, high_res_features=high_res_features, multimask_output=multimask_output)
        _, _, _, low_res_masks, high_res_masks, obj_ptr, _ = sam_outputs
        current_out['pred_masks'] = low_res_masks
        current_out['pred_masks_high_res'] = high_res_masks
        current_out['obj_ptr'] = obj_ptr
        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self.encodenewmemory(current_vision_feats=current_vision_feats, feat_sizes=feat_sizes, pred_masks_high_res=high_res_masks_for_mem_enc, is_mask_from_pts=point_inputs is not None)
            current_out['maskmem_features'] = maskmem_features
            current_out['maskmem_pos_enc'] = maskmem_pos_enc
        else:
            current_out['maskmem_features'] = None
            current_out['maskmem_pos_enc'] = None
        return current_out
    """usemultimask"""

    def usemultimask(self, is_init_cond_frame, point_inputs):
        num_pts = 0 if point_inputs is None else point_inputs['point_labels'].size(1)
        multimask_output = self.multimask_output_in_sam and (is_init_cond_frame or self.multimask_output_for_tracking) and self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num
        return multimask_output
    """applynonoverlappingconstraints"""

    def applynonoverlappingconstraints(self, pred_masks):
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks
        device = pred_masks.device
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks


def getconnectedcomponents(mask):
    return _C.get_connected_componnets(mask.contiguous())


class SAMV2Transforms(nn.Module):

    def __init__(self, resolution, mask_threshold, max_hole_area=0.0, max_sprinkle_area=0.0):
        super(SAMV2Transforms, self).__init__()
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.to_tensor = ToTensor()
        self.transforms = torch.jit.script(nn.Sequential(Resize((self.resolution, self.resolution)), Normalize(self.mean, self.std)))
    """call"""

    def __call__(self, x):
        x = self.to_tensor(x)
        return self.transforms(x)
    """forwardbatch"""

    def forwardbatch(self, img_list):
        img_batch = [self.transforms(self.to_tensor(img)) for img in img_list]
        img_batch = torch.stack(img_batch, dim=0)
        return img_batch
    """transformcoords"""

    def transformcoords(self, coords, normalize=False, orig_hw=None):
        if normalize:
            assert orig_hw is not None
            h, w = orig_hw
            coords = coords.clone()
            coords[..., 0] = coords[..., 0] / w
            coords[..., 1] = coords[..., 1] / h
        coords = coords * self.resolution
        return coords
    """transformboxes"""

    def transformboxes(self, boxes, normalize=False, orig_hw=None):
        boxes = self.transformcoords(boxes.reshape(-1, 2, 2), normalize, orig_hw)
        return boxes
    """postprocessmasks"""

    def postprocessmasks(self, masks, orig_hw):
        masks = masks.float()
        if self.max_hole_area > 0:
            mask_flat = masks.flatten(0, 1).unsqueeze(1)
            labels, areas = getconnectedcomponents(mask_flat <= self.mask_threshold)
            is_hole = (labels > 0) & (areas <= self.max_hole_area)
            is_hole = is_hole.reshape_as(masks)
            masks = torch.where(is_hole, self.mask_threshold + 10.0, masks)
        if self.max_sprinkle_area > 0:
            labels, areas = getconnectedcomponents(mask_flat > self.mask_threshold)
            is_hole = (labels > 0) & (areas <= self.max_sprinkle_area)
            is_hole = is_hole.reshape_as(masks)
            masks = torch.where(is_hole, self.mask_threshold - 10.0, masks)
        masks = F.interpolate(masks, orig_hw, mode='bilinear', align_corners=False)
        return masks


class SAMV2ImagePredictor(nn.Module):

    def __init__(self, samv2_cfg=None, use_default_samv2_t=False, use_default_samv2_s=False, use_default_samv2_bplus=False, use_default_samv2_l=False, device='cuda', load_ckpt_strict=True, mask_threshold=0.0, max_hole_area=0.0, max_sprinkle_area=0.0, apply_postprocessing=True):
        super(SAMV2ImagePredictor, self).__init__()
        if samv2_cfg is None:
            samv2_cfg = {'backbone': {'type': 'HieraWithFPN', 'scalp': 1.0, 'hiera_cfg': {'embed_dim': 144, 'num_heads': 2, 'stages': [2, 6, 36, 4], 'global_att_blocks': [23, 33, 43], 'window_pos_embed_bkg_spatial_size': [7, 7], 'window_spec': [8, 4, 16, 8]}, 'fpn_cfg': {'d_model': 256, 'backbone_channel_list': [1152, 576, 288, 144], 'fpn_top_down_levels': [2, 3], 'fpn_interp_model': 'nearest', 'position_encoding_cfg': dict(num_pos_feats=256, normalize=True, scale=None, temperature=10000, type='PositionEmbeddingSine')}}, 'head': {'memory_attention_cfg': {'d_model': 256, 'pos_enc_at_input': True, 'num_layers': 4, 'layer_cfg': {'act_cfg': {'type': 'ReLU'}, 'dim_feedforward': 2048, 'dropout': 0.1, 'pos_enc_at_attn': False, 'd_model': 256, 'pos_enc_at_cross_attn_keys': True, 'pos_enc_at_cross_attn_queries': False, 'self_attention_cfg': dict(type='RoPEAttention', rope_theta=10000.0, feat_sizes=[32, 32], embedding_dim=256, num_heads=1, downsample_rate=1, dropout=0.1), 'cross_attention_cfg': dict(type='RoPEAttention', rope_theta=10000.0, feat_sizes=[32, 32], rope_k_repeat=True, embedding_dim=256, num_heads=1, downsample_rate=1, dropout=0.1, kv_in_dim=64)}}, 'memory_encoder_cfg': {'out_dim': 64, 'position_encoding_cfg': dict(num_pos_feats=64, normalize=True, scale=None, temperature=10000, type='PositionEmbeddingSine'), 'mask_downsampler_cfg': dict(kernel_size=3, stride=2, padding=1), 'fuser_cfg': dict(num_layers=2, layer_cfg=dict(dim=256, kernel_size=7, padding=3, layer_scale_init_value=1e-06, use_dwconv=True))}, 'num_maskmem': 7, 'image_size': 1024, 'backbone_stride': 16, 'sigmoid_scale_for_mem_enc': 20.0, 'sigmoid_bias_for_mem_enc': -10.0, 'binarize_mask_from_pts_for_mem_enc': False, 'use_mask_input_as_output_without_sam': True, 'max_cond_frames_in_attn': -1, 'directly_add_no_mem_embed': True, 'use_high_res_features_in_sam': True, 'multimask_output_in_sam': True, 'multimask_min_pt_num': 0, 'multimask_max_pt_num': 1, 'multimask_output_for_tracking': True, 'use_multimask_token_for_obj_ptr': True, 'iou_prediction_use_sigmoid': True, 'memory_temporal_stride_for_eval': 1, 'add_all_frames_to_correct_as_cond': False, 'non_overlap_masks_for_mem_enc': False, 'use_obj_ptrs_in_encoder': True, 'max_obj_ptrs_in_encoder': 16, 'add_tpos_enc_to_obj_ptrs': False, 'proj_tpos_enc_in_obj_ptrs': False, 'only_obj_ptrs_in_the_past_for_eval': True, 'pred_obj_scores': True, 'pred_obj_scores_mlp': True, 'fixed_no_obj_ptr': True, 'soft_no_obj_ptr': False, 'use_mlp_for_obj_ptr_proj': True, 'sam_mask_decoder_extra_args': None, 'compile_image_encoder': False}}
            if use_default_samv2_l:
                assert not use_default_samv2_t and not use_default_samv2_s and not use_default_samv2_bplus
                samv2_cfg['ckptpath'] = 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt'
            elif use_default_samv2_bplus:
                assert not use_default_samv2_t and not use_default_samv2_s and not use_default_samv2_l
                samv2_cfg['backbone']['hiera_cfg'] = dict(embed_dim=112, num_heads=2)
                samv2_cfg['backbone']['fpn_cfg'] = dict(position_encoding_cfg=dict(num_pos_feats=256, normalize=True, scale=None, temperature=10000, type='PositionEmbeddingSine'), d_model=256, backbone_channel_list=[896, 448, 224, 112], fpn_top_down_levels=[2, 3], fpn_interp_model='nearest')
                samv2_cfg['ckptpath'] = 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt'
            elif use_default_samv2_s:
                assert not use_default_samv2_t and not use_default_samv2_bplus and not use_default_samv2_l
                samv2_cfg['backbone']['hiera_cfg'] = dict(embed_dim=96, num_heads=1, stages=[1, 2, 11, 2], global_att_blocks=[7, 10, 13], window_pos_embed_bkg_spatial_size=[7, 7])
                samv2_cfg['backbone']['fpn_cfg'] = dict(position_encoding_cfg=dict(num_pos_feats=256, normalize=True, scale=None, temperature=10000, type='PositionEmbeddingSine'), d_model=256, backbone_channel_list=[768, 384, 192, 96], fpn_top_down_levels=[2, 3], fpn_interp_model='nearest')
                samv2_cfg['ckptpath'] = 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt'
            elif use_default_samv2_t:
                assert not use_default_samv2_s and not use_default_samv2_bplus and not use_default_samv2_l
                samv2_cfg['backbone']['hiera_cfg'] = dict(embed_dim=96, num_heads=1, stages=[1, 2, 7, 2], global_att_blocks=[5, 7, 9], window_pos_embed_bkg_spatial_size=[7, 7])
                samv2_cfg['backbone']['fpn_cfg'] = dict(position_encoding_cfg=dict(num_pos_feats=256, normalize=True, scale=None, temperature=10000, type='PositionEmbeddingSine'), d_model=256, backbone_channel_list=[768, 384, 192, 96], fpn_top_down_levels=[2, 3], fpn_interp_model='nearest')
                samv2_cfg['ckptpath'] = 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt'
        else:
            assert not use_default_samv2_t and not use_default_samv2_s and not use_default_samv2_bplus and not use_default_samv2_l
        self.model = self.buildsamv2(samv2_cfg=samv2_cfg, device=device, apply_postprocessing=apply_postprocessing)
        if 'ckptpath' in samv2_cfg and (os.path.exists(samv2_cfg['ckptpath']) or samv2_cfg['ckptpath'].startswith('https')):
            if os.path.exists(samv2_cfg['ckptpath']):
                with open(samv2_cfg['ckptpath'], 'rb') as fp:
                    state_dict = torch.load(fp, map_location='cpu')
            elif samv2_cfg['ckptpath'].startswith('https'):
                state_dict = model_zoo.load_url(samv2_cfg['ckptpath'], map_location='cpu')
            else:
                raise ValueError('ckptpath %s could not be loaded.' % samv2_cfg['ckptpath'])
            self.model.load_state_dict(state_dict['model'], strict=load_ckpt_strict)
        self._transforms = SAMV2Transforms(resolution=self.model.image_size, mask_threshold=mask_threshold, max_hole_area=max_hole_area, max_sprinkle_area=max_sprinkle_area)
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False
        self.mask_threshold = mask_threshold
        self._bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
    """buildsamv2"""

    def buildsamv2(self, samv2_cfg, device, apply_postprocessing=True):
        if apply_postprocessing:
            samv2_cfg['head']['sam_mask_decoder_extra_args'] = {'dynamic_multimask_via_stability': True, 'dynamic_multimask_stability_delta': 0.05, 'dynamic_multimask_stability_thresh': 0.98}
        samv2_model = SAMV2(cfg=samv2_cfg, mode='TEST')
        samv2_model
        samv2_model.eval()
        return samv2_model
    """setimage"""

    @torch.no_grad()
    def setimage(self, image):
        self.resetpredictor()
        if isinstance(image, np.ndarray):
            assert image.shape[-1] <= 3, 'For numpy array image, we assume (HxWxC) format'
            self._orig_hw = [image.shape[:2]]
        elif isinstance(image, Image):
            w, h = image.size
            self._orig_hw = [(h, w)]
        else:
            raise NotImplementedError('Image format not supported.')
        input_image = self._transforms(image)
        input_image = input_image[None, ...]
        assert len(input_image.shape) == 4 and input_image.shape[1] == 3 and input_image.shape[0] == 1, f'input_image must be of size 1x3xHxW, got {input_image.shape}'
        backbone_out = self.model.forwardimage(input_image)
        _, vision_feats, _, _ = self.model.preparebackbonefeatures(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        feats = [feat.permute(1, 2, 0).view(1, -1, *feat_size) for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])][::-1]
        self._features = {'image_embed': feats[-1], 'high_res_feats': feats[:-1]}
        self._is_image_set = True
    """setimagebatch"""

    @torch.no_grad()
    def setimagebatch(self, image_list):
        self.resetpredictor()
        assert isinstance(image_list, list)
        self._orig_hw = []
        for image in image_list:
            assert isinstance(image, np.ndarray) and image.shape[-1] <= 3, 'images are expected to be an np.ndarray in RGB format, and of shape HxWxC'
            self._orig_hw.append(image.shape[:2])
        img_batch = self._transforms.forwardbatch(image_list)
        img_batch = img_batch
        batch_size = img_batch.shape[0]
        assert len(img_batch.shape) == 4 and img_batch.shape[1] == 3, f'img_batch must be of size Bx3xHxW, got {img_batch.shape}'
        backbone_out = self.model.forwardimage(img_batch)
        _, vision_feats, _, _ = self.model.preparebackbonefeatures(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        feats = [feat.permute(1, 2, 0).view(batch_size, -1, *feat_size) for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])][::-1]
        self._features = {'image_embed': feats[-1], 'high_res_feats': feats[:-1]}
        self._is_image_set = True
        self._is_batch = True
    """predictbatch"""

    def predictbatch(self, point_coords_batch=None, point_labels_batch=None, box_batch=None, mask_input_batch=None, multimask_output=True, return_logits=False, normalize_coords=True):
        assert self._is_batch, 'this function should only be used when in batched mode'
        assert self._is_image_set, 'an image must be set with .setimagebatch(...) before mask prediction.'
        num_images = len(self._features['image_embed'])
        all_masks, all_ious, all_low_res_masks = [], [], []
        for img_idx in range(num_images):
            point_coords = point_coords_batch[img_idx] if point_coords_batch is not None else None
            point_labels = point_labels_batch[img_idx] if point_labels_batch is not None else None
            box = box_batch[img_idx] if box_batch is not None else None
            mask_input = mask_input_batch[img_idx] if mask_input_batch is not None else None
            mask_input, unnorm_coords, labels, unnorm_box = self.prepprompts(point_coords, point_labels, box, mask_input, normalize_coords, img_idx=img_idx)
            masks, iou_predictions, low_res_masks = self.purepredict(unnorm_coords, labels, unnorm_box, mask_input, multimask_output, return_logits=return_logits, img_idx=img_idx)
            masks_np = masks.squeeze(0).float().detach().cpu().numpy()
            iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
            low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
            all_masks.append(masks_np)
            all_ious.append(iou_predictions_np)
            all_low_res_masks.append(low_res_masks_np)
        return all_masks, all_ious, all_low_res_masks
    """predict"""

    def predict(self, point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=True, return_logits=False, normalize_coords=True):
        assert self._is_image_set, 'an image must be set with .setimage(...) before mask prediction.'
        mask_input, unnorm_coords, labels, unnorm_box = self.prepprompts(point_coords, point_labels, box, mask_input, normalize_coords)
        masks, iou_predictions, low_res_masks = self.purepredict(unnorm_coords, labels, unnorm_box, mask_input, multimask_output, return_logits=return_logits)
        masks_np = masks.squeeze(0).float().detach().cpu().numpy()
        iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
        low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np
    """prepprompts"""

    def prepprompts(self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1):
        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            assert point_labels is not None, 'point_labels must be supplied if point_coords is supplied.'
            point_coords = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            unnorm_coords = self._transforms.transformcoords(point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx])
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transformboxes(box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx])
        if mask_logits is not None:
            mask_input = torch.as_tensor(mask_logits, dtype=torch.float, device=self.device)
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box
    """purepredict"""

    @torch.no_grad()
    def purepredict(self, point_coords, point_labels, boxes=None, mask_input=None, multimask_output=True, return_logits=False, img_idx=-1):
        assert self._is_image_set, 'an image must be set with .setimage(...) before mask prediction.'
        if point_coords is not None:
            concat_points = point_coords, point_labels
        else:
            concat_points = None
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = concat_coords, concat_labels
            else:
                concat_points = box_coords, box_labels
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(points=concat_points, boxes=None, masks=mask_input)
        batched_mode = concat_points is not None and concat_points[0].shape[0] > 1
        high_res_features = [feat_level[img_idx].unsqueeze(0) for feat_level in self._features['high_res_feats']]
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(image_embeddings=self._features['image_embed'][img_idx].unsqueeze(0), image_pe=self.model.sam_prompt_encoder.getdensepe(), sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=multimask_output, repeat_image=batched_mode, high_res_features=high_res_features)
        masks = self._transforms.postprocessmasks(low_res_masks, self._orig_hw[img_idx])
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        if not return_logits:
            masks = masks > self.mask_threshold
        return masks, iou_predictions, low_res_masks
    """getimageembedding"""

    def getimageembedding(self):
        assert self._is_image_set, 'an image must be set with .setimage(...) to generate an embedding.'
        assert self._features is not None, 'features must exist if an image has been set.'
        return self._features['image_embed']
    """device"""

    @property
    def device(self):
        return self.model.device
    """resetpredictor"""

    def resetpredictor(self):
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False


def concatpoints(old_point_inputs, new_points, new_labels):
    if old_point_inputs is None:
        points, labels = new_points, new_labels
    else:
        points = torch.cat([old_point_inputs['point_coords'], new_points], dim=1)
        labels = torch.cat([old_point_inputs['point_labels'], new_labels], dim=1)
    return {'point_coords': points, 'point_labels': labels}


def fillholesinmaskscores(mask, max_area):
    assert max_area > 0, 'max_area must be positive'
    labels, areas = getconnectedcomponents(mask <= 0)
    is_hole = (labels > 0) & (areas <= max_area)
    mask = torch.where(is_hole, 0.1, mask)
    return mask


def loadimgastensor(img_path, image_size):
    img_pil = Image.open(img_path)
    img_np = np.array(img_pil.convert('RGB').resize((image_size, image_size)))
    if img_np.dtype == np.uint8:
        img_np = img_np / 255.0
    else:
        raise RuntimeError(f'Unknown image dtype: {img_np.dtype} on {img_path}')
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    video_width, video_height = img_pil.size
    return img, video_height, video_width


class AsyncVideoFrameLoader:

    def __init__(self, img_paths, image_size, offload_video_to_cpu, img_mean, img_std):
        self.img_paths = img_paths
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = img_mean
        self.img_std = img_std
        self.images = [None] * len(img_paths)
        self.exception = None
        self.video_height = None
        self.video_width = None
        self.__getitem__(0)

        def _load_frames():
            try:
                for n in tqdm(range(len(self.images)), desc='frame loading (JPEG)'):
                    self.__getitem__(n)
            except Exception as e:
                self.exception = e
        self.thread = Thread(target=_load_frames, daemon=True)
        self.thread.start()
    """getitem"""

    def __getitem__(self, index):
        if self.exception is not None:
            raise RuntimeError('Failure in frame loading thread') from self.exception
        img = self.images[index]
        if img is not None:
            return img
        img, video_height, video_width = loadimgastensor(self.img_paths[index], self.image_size)
        self.video_height = video_height
        self.video_width = video_width
        img -= self.img_mean
        img /= self.img_std
        if not self.offload_video_to_cpu:
            img = img
        self.images[index] = img
        return img
    """len"""

    def __len__(self):
        return len(self.images)


def loadvideoframes(video_path, image_size, offload_video_to_cpu, img_mean=(0.485, 0.456, 0.406), img_std=(0.229, 0.224, 0.225), async_loading_frames=False):
    if isinstance(video_path, str) and os.path.isdir(video_path):
        jpg_folder = video_path
    else:
        raise NotImplementedError('Only JPEG frames are supported at this moment')
    frame_names = [p for p in os.listdir(jpg_folder) if os.path.splitext(p)[-1] in ['.jpg', '.jpeg', '.JPG', '.JPEG']]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    num_frames = len(frame_names)
    if num_frames == 0:
        raise RuntimeError(f'no images found in {jpg_folder}')
    img_paths = [os.path.join(jpg_folder, frame_name) for frame_name in frame_names]
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
    if async_loading_frames:
        lazy_images = AsyncVideoFrameLoader(img_paths, image_size, offload_video_to_cpu, img_mean, img_std)
        return lazy_images, lazy_images.video_height, lazy_images.video_width
    images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)
    for n, img_path in enumerate(tqdm(img_paths, desc='frame loading (JPEG)')):
        images[n], video_height, video_width = loadimgastensor(img_path, image_size)
    if not offload_video_to_cpu:
        images = images
        img_mean = img_mean
        img_std = img_std
    images -= img_mean
    images /= img_std
    return images, video_height, video_width


class SAMV2VideoPredictor(SAMV2ImagePredictor):

    def __init__(self, fill_hole_area=0, non_overlap_masks=False, clear_non_cond_mem_around_input=False, clear_non_cond_mem_for_multi_obj=False, **kwargs):
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj
        super(SAMV2VideoPredictor, self).__init__(**kwargs)
    """initstate"""

    @torch.inference_mode()
    def initstate(self, video_path, offload_video_to_cpu=False, offload_state_to_cpu=False, async_loading_frames=False):
        images, video_height, video_width = loadvideoframes(video_path=video_path, image_size=self.model.image_size, offload_video_to_cpu=offload_video_to_cpu, async_loading_frames=async_loading_frames)
        inference_state = {}
        inference_state['images'] = images
        inference_state['num_frames'] = len(images)
        inference_state['offload_video_to_cpu'] = offload_video_to_cpu
        inference_state['offload_state_to_cpu'] = offload_state_to_cpu
        inference_state['video_height'] = video_height
        inference_state['video_width'] = video_width
        inference_state['device'] = self.device
        if offload_state_to_cpu:
            inference_state['storage_device'] = torch.device('cpu')
        else:
            inference_state['storage_device'] = torch.device('cuda')
        inference_state['point_inputs_per_obj'] = {}
        inference_state['mask_inputs_per_obj'] = {}
        inference_state['cached_features'] = {}
        inference_state['constants'] = {}
        inference_state['obj_id_to_idx'] = OrderedDict()
        inference_state['obj_idx_to_id'] = OrderedDict()
        inference_state['obj_ids'] = []
        inference_state['output_dict'] = {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {}}
        inference_state['output_dict_per_obj'] = {}
        inference_state['temp_output_dict_per_obj'] = {}
        inference_state['consolidated_frame_inds'] = {'cond_frame_outputs': set(), 'non_cond_frame_outputs': set()}
        inference_state['tracking_has_started'] = False
        inference_state['frames_already_tracked'] = {}
        self.getimagefeature(inference_state, frame_idx=0, batch_size=1)
        return inference_state
    """buildsamv2"""

    def buildsamv2(self, samv2_cfg, device, apply_postprocessing=True):
        if apply_postprocessing:
            samv2_cfg['head']['sam_mask_decoder_extra_args'] = {'dynamic_multimask_via_stability': True, 'dynamic_multimask_stability_delta': 0.05, 'dynamic_multimask_stability_thresh': 0.98}
            samv2_cfg['head']['binarize_mask_from_pts_for_mem_enc'] = True
            self.fill_hole_area = 8
        samv2_model = SAMV2(cfg=samv2_cfg, mode='TEST')
        samv2_model
        samv2_model.eval()
        return samv2_model
    """objidtoidx"""

    def objidtoidx(self, inference_state, obj_id):
        obj_idx = inference_state['obj_id_to_idx'].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx
        allow_new_object = not inference_state['tracking_has_started']
        if allow_new_object:
            obj_idx = len(inference_state['obj_id_to_idx'])
            inference_state['obj_id_to_idx'][obj_id] = obj_idx
            inference_state['obj_idx_to_id'][obj_idx] = obj_id
            inference_state['obj_ids'] = list(inference_state['obj_id_to_idx'])
            inference_state['point_inputs_per_obj'][obj_idx] = {}
            inference_state['mask_inputs_per_obj'][obj_idx] = {}
            inference_state['output_dict_per_obj'][obj_idx] = {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {}}
            inference_state['temp_output_dict_per_obj'][obj_idx] = {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {}}
            return obj_idx
        else:
            raise RuntimeError(f"Cannot add new object id {obj_id} after tracking starts. All existing object ids: {inference_state['obj_ids']}. Please call 'resetstate' to restart from scratch.")
    """objidxtoid"""

    def objidxtoid(self, inference_state, obj_idx):
        return inference_state['obj_idx_to_id'][obj_idx]
    """getobjnum"""

    def getobjnum(self, inference_state):
        return len(inference_state['obj_idx_to_id'])
    """addnewpoints"""

    @torch.inference_mode()
    def addnewpoints(self, inference_state, frame_idx, obj_id, points, labels, clear_old_points=True, normalize_coords=True):
        obj_idx = self.objidtoidx(inference_state, obj_id)
        point_inputs_per_frame = inference_state['point_inputs_per_obj'][obj_idx]
        mask_inputs_per_frame = inference_state['mask_inputs_per_obj'][obj_idx]
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)
        if points.dim() == 2:
            points = points.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        if normalize_coords:
            video_H = inference_state['video_height']
            video_W = inference_state['video_width']
            points = points / torch.tensor([video_W, video_H])
        points = points * self.model.image_size
        points = points
        labels = labels
        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        point_inputs = concatpoints(point_inputs, points, labels)
        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)
        is_init_cond_frame = frame_idx not in inference_state['frames_already_tracked']
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state['frames_already_tracked'][frame_idx]['reverse']
        obj_output_dict = inference_state['output_dict_per_obj'][obj_idx]
        obj_temp_output_dict = inference_state['temp_output_dict_per_obj'][obj_idx]
        is_cond = is_init_cond_frame or self.model.add_all_frames_to_correct_as_cond
        storage_key = 'cond_frame_outputs' if is_cond else 'non_cond_frame_outputs'
        prev_sam_mask_logits = None
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict['cond_frame_outputs'].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict['non_cond_frame_outputs'].get(frame_idx)
        if prev_out is not None and prev_out['pred_masks'] is not None:
            prev_sam_mask_logits = prev_out['pred_masks']
            prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)
        current_out, _ = self.runsingleframeinference(inference_state=inference_state, output_dict=obj_output_dict, frame_idx=frame_idx, batch_size=1, is_init_cond_frame=is_init_cond_frame, point_inputs=point_inputs, mask_inputs=None, reverse=reverse, run_mem_encoder=False, prev_sam_mask_logits=prev_sam_mask_logits)
        obj_temp_output_dict[storage_key][frame_idx] = current_out
        obj_ids = inference_state['obj_ids']
        consolidated_out = self.consolidatetempoutputacrossobj(inference_state, frame_idx, is_cond=is_cond, run_mem_encoder=False, consolidate_at_video_res=True)
        _, video_res_masks = self.getorigvideoresoutput(inference_state, consolidated_out['pred_masks_video_res'])
        return frame_idx, obj_ids, video_res_masks
    """addnewmask"""

    @torch.inference_mode()
    def addnewmask(self, inference_state, frame_idx, obj_id, mask):
        obj_idx = self.objidtoidx(inference_state, obj_id)
        point_inputs_per_frame = inference_state['point_inputs_per_obj'][obj_idx]
        mask_inputs_per_frame = inference_state['mask_inputs_per_obj'][obj_idx]
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        assert mask.dim() == 2
        mask_H, mask_W = mask.shape
        mask_inputs_orig = mask[None, None]
        mask_inputs_orig = mask_inputs_orig.float()
        if mask_H != self.model.image_size or mask_W != self.model.image_size:
            mask_inputs = F.interpolate(mask_inputs_orig, size=(self.model.image_size, self.model.image_size), align_corners=False, mode='bilinear', antialias=True)
            mask_inputs = (mask_inputs >= 0.5).float()
        else:
            mask_inputs = mask_inputs_orig
        mask_inputs_per_frame[frame_idx] = mask_inputs
        point_inputs_per_frame.pop(frame_idx, None)
        is_init_cond_frame = frame_idx not in inference_state['frames_already_tracked']
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state['frames_already_tracked'][frame_idx]['reverse']
        obj_output_dict = inference_state['output_dict_per_obj'][obj_idx]
        obj_temp_output_dict = inference_state['temp_output_dict_per_obj'][obj_idx]
        is_cond = is_init_cond_frame or self.model.add_all_frames_to_correct_as_cond
        storage_key = 'cond_frame_outputs' if is_cond else 'non_cond_frame_outputs'
        current_out, _ = self.runsingleframeinference(inference_state=inference_state, output_dict=obj_output_dict, frame_idx=frame_idx, batch_size=1, is_init_cond_frame=is_init_cond_frame, point_inputs=None, mask_inputs=mask_inputs, reverse=reverse, run_mem_encoder=False)
        obj_temp_output_dict[storage_key][frame_idx] = current_out
        obj_ids = inference_state['obj_ids']
        consolidated_out = self.consolidatetempoutputacrossobj(inference_state, frame_idx, is_cond=is_cond, run_mem_encoder=False, consolidate_at_video_res=True)
        _, video_res_masks = self.getorigvideoresoutput(inference_state, consolidated_out['pred_masks_video_res'])
        return frame_idx, obj_ids, video_res_masks
    """getorigvideoresoutput"""

    def getorigvideoresoutput(self, inference_state, any_res_masks):
        device = inference_state['device']
        video_H = inference_state['video_height']
        video_W = inference_state['video_width']
        any_res_masks = any_res_masks
        if any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks
        else:
            video_res_masks = F.interpolate(any_res_masks, size=(video_H, video_W), mode='bilinear', align_corners=False)
        if self.non_overlap_masks:
            video_res_masks = self.model.applynonoverlappingconstraints(video_res_masks)
        return any_res_masks, video_res_masks
    """consolidatetempoutputacrossobj"""

    def consolidatetempoutputacrossobj(self, inference_state, frame_idx, is_cond, run_mem_encoder, consolidate_at_video_res=False):
        batch_size = self.getobjnum(inference_state)
        storage_key = 'cond_frame_outputs' if is_cond else 'non_cond_frame_outputs'
        if consolidate_at_video_res:
            assert not run_mem_encoder, 'memory encoder cannot run at video resolution'
            consolidated_H = inference_state['video_height']
            consolidated_W = inference_state['video_width']
            consolidated_mask_key = 'pred_masks_video_res'
        else:
            consolidated_H = consolidated_W = self.model.image_size // 4
            consolidated_mask_key = 'pred_masks'
        consolidated_out = {'maskmem_features': None, 'maskmem_pos_enc': None, 'obj_ptr': torch.full(size=(batch_size, self.model.hidden_dim), fill_value=NO_OBJ_SCORE, dtype=torch.float32, device=inference_state['device']), consolidated_mask_key: torch.full(size=(batch_size, 1, consolidated_H, consolidated_W), fill_value=NO_OBJ_SCORE, dtype=torch.float32, device=inference_state['storage_device'])}
        empty_mask_ptr = None
        for obj_idx in range(batch_size):
            obj_temp_output_dict = inference_state['temp_output_dict_per_obj'][obj_idx]
            obj_output_dict = inference_state['output_dict_per_obj'][obj_idx]
            out = obj_temp_output_dict[storage_key].get(frame_idx, None)
            if out is None:
                out = obj_output_dict['cond_frame_outputs'].get(frame_idx, None)
            if out is None:
                out = obj_output_dict['non_cond_frame_outputs'].get(frame_idx, None)
            if out is None:
                if run_mem_encoder:
                    if empty_mask_ptr is None:
                        empty_mask_ptr = self.getemptymaskptr(inference_state, frame_idx)
                    consolidated_out['obj_ptr'][obj_idx:obj_idx + 1] = empty_mask_ptr
                continue
            obj_mask = out['pred_masks']
            consolidated_pred_masks = consolidated_out[consolidated_mask_key]
            if obj_mask.shape[-2:] == consolidated_pred_masks.shape[-2:]:
                consolidated_pred_masks[obj_idx:obj_idx + 1] = obj_mask
            else:
                resized_obj_mask = F.interpolate(obj_mask, size=consolidated_pred_masks.shape[-2:], mode='bilinear', align_corners=False)
                consolidated_pred_masks[obj_idx:obj_idx + 1] = resized_obj_mask
            consolidated_out['obj_ptr'][obj_idx:obj_idx + 1] = out['obj_ptr']
        if run_mem_encoder:
            device = inference_state['device']
            high_res_masks = F.interpolate(consolidated_out['pred_masks'], size=(self.model.image_size, self.model.image_size), mode='bilinear', align_corners=False)
            if self.model.non_overlap_masks_for_mem_enc:
                high_res_masks = self.model.applynonoverlappingconstraints(high_res_masks)
            maskmem_features, maskmem_pos_enc = self.runmemoryencoder(inference_state=inference_state, frame_idx=frame_idx, batch_size=batch_size, high_res_masks=high_res_masks, is_mask_from_pts=True)
            consolidated_out['maskmem_features'] = maskmem_features
            consolidated_out['maskmem_pos_enc'] = maskmem_pos_enc
        return consolidated_out
    """getemptymaskptr"""

    def getemptymaskptr(self, inference_state, frame_idx):
        batch_size = 1
        mask_inputs = torch.zeros((batch_size, 1, self.model.image_size, self.model.image_size), dtype=torch.float32, device=inference_state['device'])
        _, _, current_vision_feats, current_vision_pos_embeds, feat_sizes = self.getimagefeature(inference_state, frame_idx, batch_size)
        current_out = self.model.trackstep(frame_idx=frame_idx, is_init_cond_frame=True, current_vision_feats=current_vision_feats, current_vision_pos_embeds=current_vision_pos_embeds, feat_sizes=feat_sizes, point_inputs=None, mask_inputs=mask_inputs, output_dict={}, num_frames=inference_state['num_frames'], track_in_reverse=False, run_mem_encoder=False, prev_sam_mask_logits=None)
        return current_out['obj_ptr']
    """propagateinvideopreflight"""

    @torch.inference_mode()
    def propagateinvideopreflight(self, inference_state):
        inference_state['tracking_has_started'] = True
        batch_size = self.getobjnum(inference_state)
        temp_output_dict_per_obj = inference_state['temp_output_dict_per_obj']
        output_dict = inference_state['output_dict']
        consolidated_frame_inds = inference_state['consolidated_frame_inds']
        for is_cond in [False, True]:
            storage_key = 'cond_frame_outputs' if is_cond else 'non_cond_frame_outputs'
            temp_frame_inds = set()
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
            consolidated_frame_inds[storage_key].update(temp_frame_inds)
            for frame_idx in temp_frame_inds:
                consolidated_out = self.consolidatetempoutputacrossobj(inference_state, frame_idx, is_cond=is_cond, run_mem_encoder=True)
                output_dict[storage_key][frame_idx] = consolidated_out
                self.addoutputperobject(inference_state, frame_idx, consolidated_out, storage_key)
                clear_non_cond_mem = self.clear_non_cond_mem_around_input and (self.clear_non_cond_mem_for_multi_obj or batch_size <= 1)
                if clear_non_cond_mem:
                    self.clearnoncondmemaroundinput(inference_state, frame_idx)
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                obj_temp_output_dict[storage_key].clear()
        for frame_idx in output_dict['cond_frame_outputs']:
            output_dict['non_cond_frame_outputs'].pop(frame_idx, None)
        for obj_output_dict in inference_state['output_dict_per_obj'].values():
            for frame_idx in obj_output_dict['cond_frame_outputs']:
                obj_output_dict['non_cond_frame_outputs'].pop(frame_idx, None)
        for frame_idx in consolidated_frame_inds['cond_frame_outputs']:
            assert frame_idx in output_dict['cond_frame_outputs']
            consolidated_frame_inds['non_cond_frame_outputs'].discard(frame_idx)
        all_consolidated_frame_inds = consolidated_frame_inds['cond_frame_outputs'] | consolidated_frame_inds['non_cond_frame_outputs']
        input_frames_inds = set()
        for point_inputs_per_frame in inference_state['point_inputs_per_obj'].values():
            input_frames_inds.update(point_inputs_per_frame.keys())
        for mask_inputs_per_frame in inference_state['mask_inputs_per_obj'].values():
            input_frames_inds.update(mask_inputs_per_frame.keys())
        assert all_consolidated_frame_inds == input_frames_inds
    """propagateinvideo"""

    @torch.inference_mode()
    def propagateinvideo(self, inference_state, start_frame_idx=None, max_frame_num_to_track=None, reverse=False):
        self.propagateinvideopreflight(inference_state)
        output_dict = inference_state['output_dict']
        consolidated_frame_inds = inference_state['consolidated_frame_inds']
        obj_ids = inference_state['obj_ids']
        num_frames = inference_state['num_frames']
        batch_size = self.getobjnum(inference_state)
        if len(output_dict['cond_frame_outputs']) == 0:
            raise RuntimeError('No points are provided; please add points first.')
        clear_non_cond_mem = self.clear_non_cond_mem_around_input and (self.clear_non_cond_mem_for_multi_obj or batch_size <= 1)
        if start_frame_idx is None:
            start_frame_idx = min(output_dict['cond_frame_outputs'])
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []
        else:
            end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
            processing_order = range(start_frame_idx, end_frame_idx + 1)
        for frame_idx in tqdm(processing_order, desc='propagate in video'):
            if frame_idx in consolidated_frame_inds['cond_frame_outputs']:
                storage_key = 'cond_frame_outputs'
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out['pred_masks']
                if clear_non_cond_mem:
                    self.clearnoncondmemaroundinput(inference_state, frame_idx)
            elif frame_idx in consolidated_frame_inds['non_cond_frame_outputs']:
                storage_key = 'non_cond_frame_outputs'
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out['pred_masks']
            else:
                storage_key = 'non_cond_frame_outputs'
                current_out, pred_masks = self.runsingleframeinference(inference_state=inference_state, output_dict=output_dict, frame_idx=frame_idx, batch_size=batch_size, is_init_cond_frame=False, point_inputs=None, mask_inputs=None, reverse=reverse, run_mem_encoder=True)
                output_dict[storage_key][frame_idx] = current_out
            self.addoutputperobject(inference_state, frame_idx, current_out, storage_key)
            inference_state['frames_already_tracked'][frame_idx] = {'reverse': reverse}
            _, video_res_masks = self.getorigvideoresoutput(inference_state, pred_masks)
            yield frame_idx, obj_ids, video_res_masks
    """addoutputperobject"""

    def addoutputperobject(self, inference_state, frame_idx, current_out, storage_key):
        maskmem_features = current_out['maskmem_features']
        assert maskmem_features is None or isinstance(maskmem_features, torch.Tensor)
        maskmem_pos_enc = current_out['maskmem_pos_enc']
        assert maskmem_pos_enc is None or isinstance(maskmem_pos_enc, list)
        output_dict_per_obj = inference_state['output_dict_per_obj']
        for obj_idx, obj_output_dict in output_dict_per_obj.items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_out = {'maskmem_features': None, 'maskmem_pos_enc': None, 'pred_masks': current_out['pred_masks'][obj_slice], 'obj_ptr': current_out['obj_ptr'][obj_slice]}
            if maskmem_features is not None:
                obj_out['maskmem_features'] = maskmem_features[obj_slice]
            if maskmem_pos_enc is not None:
                obj_out['maskmem_pos_enc'] = [x[obj_slice] for x in maskmem_pos_enc]
            obj_output_dict[storage_key][frame_idx] = obj_out
    """resetstate"""

    @torch.inference_mode()
    def resetstate(self, inference_state):
        self.resettrackingresults(inference_state)
        inference_state['obj_id_to_idx'].clear()
        inference_state['obj_idx_to_id'].clear()
        inference_state['obj_ids'].clear()
        inference_state['point_inputs_per_obj'].clear()
        inference_state['mask_inputs_per_obj'].clear()
        inference_state['output_dict_per_obj'].clear()
        inference_state['temp_output_dict_per_obj'].clear()
    """resettrackingresults"""

    def resettrackingresults(self, inference_state):
        for v in inference_state['point_inputs_per_obj'].values():
            v.clear()
        for v in inference_state['mask_inputs_per_obj'].values():
            v.clear()
        for v in inference_state['output_dict_per_obj'].values():
            v['cond_frame_outputs'].clear()
            v['non_cond_frame_outputs'].clear()
        for v in inference_state['temp_output_dict_per_obj'].values():
            v['cond_frame_outputs'].clear()
            v['non_cond_frame_outputs'].clear()
        inference_state['output_dict']['cond_frame_outputs'].clear()
        inference_state['output_dict']['non_cond_frame_outputs'].clear()
        inference_state['consolidated_frame_inds']['cond_frame_outputs'].clear()
        inference_state['consolidated_frame_inds']['non_cond_frame_outputs'].clear()
        inference_state['tracking_has_started'] = False
        inference_state['frames_already_tracked'].clear()
    """getimagefeature"""

    def getimagefeature(self, inference_state, frame_idx, batch_size):
        image, backbone_out = inference_state['cached_features'].get(frame_idx, (None, None))
        if backbone_out is None:
            image = inference_state['images'][frame_idx].float().unsqueeze(0)
            backbone_out = self.model.forwardimage(image)
            inference_state['cached_features'] = {frame_idx: (image, backbone_out)}
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {'backbone_fpn': backbone_out['backbone_fpn'].copy(), 'vision_pos_enc': backbone_out['vision_pos_enc'].copy()}
        for i, feat in enumerate(expanded_backbone_out['backbone_fpn']):
            expanded_backbone_out['backbone_fpn'][i] = feat.expand(batch_size, -1, -1, -1)
        for i, pos in enumerate(expanded_backbone_out['vision_pos_enc']):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out['vision_pos_enc'][i] = pos
        features = self.model.preparebackbonefeatures(expanded_backbone_out)
        features = (expanded_image,) + features
        return features
    """runsingleframeinference"""

    def runsingleframeinference(self, inference_state, output_dict, frame_idx, batch_size, is_init_cond_frame, point_inputs, mask_inputs, reverse, run_mem_encoder, prev_sam_mask_logits=None):
        _, _, current_vision_feats, current_vision_pos_embeds, feat_sizes = self.getimagefeature(inference_state, frame_idx, batch_size)
        assert point_inputs is None or mask_inputs is None
        current_out = self.model.trackstep(frame_idx=frame_idx, is_init_cond_frame=is_init_cond_frame, current_vision_feats=current_vision_feats, current_vision_pos_embeds=current_vision_pos_embeds, feat_sizes=feat_sizes, point_inputs=point_inputs, mask_inputs=mask_inputs, output_dict=output_dict, num_frames=inference_state['num_frames'], track_in_reverse=reverse, run_mem_encoder=run_mem_encoder, prev_sam_mask_logits=prev_sam_mask_logits)
        storage_device = inference_state['storage_device']
        maskmem_features = current_out['maskmem_features']
        if maskmem_features is not None:
            maskmem_features = maskmem_features
            maskmem_features = maskmem_features
        pred_masks_gpu = current_out['pred_masks']
        if self.fill_hole_area > 0:
            pred_masks_gpu = fillholesinmaskscores(pred_masks_gpu, self.fill_hole_area)
        pred_masks = pred_masks_gpu
        maskmem_pos_enc = self.getmaskmemposenc(inference_state, current_out)
        obj_ptr = current_out['obj_ptr']
        compact_current_out = {'maskmem_features': maskmem_features, 'maskmem_pos_enc': maskmem_pos_enc, 'pred_masks': pred_masks, 'obj_ptr': obj_ptr}
        return compact_current_out, pred_masks_gpu
    """runmemoryencoder"""

    def runmemoryencoder(self, inference_state, frame_idx, batch_size, high_res_masks, is_mask_from_pts):
        _, _, current_vision_feats, _, feat_sizes = self.getimagefeature(inference_state, frame_idx, batch_size)
        maskmem_features, maskmem_pos_enc = self.model.encodenewmemory(current_vision_feats=current_vision_feats, feat_sizes=feat_sizes, pred_masks_high_res=high_res_masks, is_mask_from_pts=is_mask_from_pts)
        storage_device = inference_state['storage_device']
        maskmem_features = maskmem_features
        maskmem_features = maskmem_features
        maskmem_pos_enc = self.getmaskmemposenc(inference_state, {'maskmem_pos_enc': maskmem_pos_enc})
        return maskmem_features, maskmem_pos_enc
    """getmaskmemposenc"""

    def getmaskmemposenc(self, inference_state, current_out):
        model_constants = inference_state['constants']
        out_maskmem_pos_enc = current_out['maskmem_pos_enc']
        if out_maskmem_pos_enc is not None:
            if 'maskmem_pos_enc' not in model_constants:
                assert isinstance(out_maskmem_pos_enc, list)
                maskmem_pos_enc = [x[0:1].clone() for x in out_maskmem_pos_enc]
                model_constants['maskmem_pos_enc'] = maskmem_pos_enc
            else:
                maskmem_pos_enc = model_constants['maskmem_pos_enc']
            batch_size = out_maskmem_pos_enc[0].size(0)
            expanded_maskmem_pos_enc = [x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc]
        else:
            expanded_maskmem_pos_enc = None
        return expanded_maskmem_pos_enc
    """clearnoncondmemaroundinput"""

    def clearnoncondmemaroundinput(self, inference_state, frame_idx):
        r = self.model.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.model.num_maskmem
        frame_idx_end = frame_idx + r * self.model.num_maskmem
        output_dict = inference_state['output_dict']
        non_cond_frame_outputs = output_dict['non_cond_frame_outputs']
        for t in range(frame_idx_begin, frame_idx_end + 1):
            non_cond_frame_outputs.pop(t, None)
            for obj_output_dict in inference_state['output_dict_per_obj'].values():
                obj_output_dict['non_cond_frame_outputs'].pop(t, None)


class Segformer(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(Segformer, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        self.convs = nn.ModuleList()
        for in_channels in head_cfg['in_channels_list']:
            self.convs.append(nn.Sequential(nn.Conv2d(in_channels, head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg)))
        self.decoder = nn.Sequential(nn.Conv2d(head_cfg['feats_channels'] * len(self.convs), head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        outs = []
        for idx, feats in enumerate(list(backbone_outputs)):
            outs.append(F.interpolate(self.convs[idx](feats), size=backbone_outputs[0].shape[2:], mode='bilinear', align_corners=self.align_corners))
        feats = torch.cat(outs, dim=1)
        seg_logits = self.decoder(feats)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            loss, losses_log_dict = self.customizepredsandlosses(seg_logits=seg_logits, annotations=data_meta.getannotations(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size)
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs


class SemanticFPN(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(SemanticFPN, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        self.fpn_neck = FPN(in_channels_list=head_cfg['in_channels_list'], out_channels=head_cfg['feats_channels'], upsample_cfg=head_cfg['upsample_cfg'], norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.scale_heads, feature_stride_list = nn.ModuleList(), head_cfg['feature_stride_list']
        for i in range(len(feature_stride_list)):
            head_length = max(1, int(np.log2(feature_stride_list[i]) - np.log2(feature_stride_list[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(nn.Sequential(nn.Conv2d(head_cfg['feats_channels'] if k == 0 else head_cfg['scale_head_channels'], head_cfg['scale_head_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['scale_head_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg)))
                if feature_stride_list[i] != feature_stride_list[0]:
                    scale_head.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))
        self.decoder = nn.Sequential(nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['scale_head_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        fpn_outs = self.fpn_neck(list(backbone_outputs))
        feats = self.scale_heads[0](fpn_outs[0])
        for i in range(1, len(self.cfg['head']['feature_stride_list'])):
            feats = feats + F.interpolate(self.scale_heads[i](fpn_outs[i]), size=feats.shape[2:], mode='bilinear', align_corners=self.align_corners)
        seg_logits = self.decoder(feats)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            loss, losses_log_dict = self.customizepredsandlosses(seg_logits=seg_logits, annotations=data_meta.getannotations(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size)
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs


class MLAModule(nn.Module):

    def __init__(self, in_channels_list=[1024, 1024, 1024, 1024], out_channels=256, norm_cfg=None, act_cfg=None):
        super(MLAModule, self).__init__()
        self.channel_proj = nn.ModuleList()
        for i in range(len(in_channels_list)):
            self.channel_proj.append(nn.Sequential(nn.Conv2d(in_channels_list[i], out_channels, kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg)))
        self.feat_extract = nn.ModuleList()
        for i in range(len(in_channels_list)):
            self.feat_extract.append(nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg), BuildActivation(act_cfg)))
    """forward"""

    def forward(self, inputs):
        feat_list = []
        for x, conv in zip(inputs, self.channel_proj):
            feat_list.append(conv(x))
        feat_list = feat_list[::-1]
        mid_list = []
        for feat in feat_list:
            if len(mid_list) == 0:
                mid_list.append(feat)
            else:
                mid_list.append(mid_list[-1] + feat)
        out_list = []
        for mid, conv in zip(mid_list, self.feat_extract):
            out_list.append(conv(mid))
        return tuple(out_list)


class MLANeck(nn.Module):

    def __init__(self, in_channels_list, out_channels, norm_layers, norm_cfg=None, act_cfg=None):
        super(MLANeck, self).__init__()
        assert isinstance(in_channels_list, list) or isinstance(in_channels_list, tuple)
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.norm_layers = norm_layers
        self.mla = MLAModule(in_channels_list, out_channels, norm_cfg, act_cfg)
    """forward"""

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels_list)
        outs = []
        for i in range(len(inputs)):
            x = inputs[i]
            n, c, h, w = x.shape
            x = x.reshape(n, c, h * w).transpose(2, 1).contiguous()
            x = self.norm_layers[i](x)
            x = x.transpose(1, 2).reshape(n, c, h, w).contiguous()
            outs.append(x)
        outs = self.mla(outs)
        return tuple(outs)


class SETRUP(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(SETRUP, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        self.norm_layers = nn.ModuleList()
        for in_channels in head_cfg['in_channels_list']:
            norm_cfg_copy = head_cfg['norm_cfg'].copy()
            norm_layer = BuildNormalization(placeholder=in_channels, norm_cfg=norm_cfg_copy)
            self.norm_layers.append(norm_layer)
        self.decoder = self.builddecoder({'in_channels': head_cfg['in_channels_list'][-1], 'out_channels': head_cfg['feats_channels'], 'kernel_size': head_cfg['kernel_size'], 'scale_factor': head_cfg['scale_factor'], 'dropout': head_cfg['dropout'], 'num_convs': head_cfg['num_convs']})
        auxiliary_cfg_list = cfg['auxiliary']
        assert isinstance(auxiliary_cfg_list, (tuple, list))
        self.auxiliary_decoders = nn.ModuleList()
        for auxiliary_cfg in auxiliary_cfg_list:
            decoder = self.builddecoder(auxiliary_cfg)
            self.auxiliary_decoders.append(decoder)
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        assert len(backbone_outputs) == len(self.norm_layers)
        for idx in range(len(backbone_outputs)):
            backbone_outputs[idx] = self.norm(backbone_outputs[idx], self.norm_layers[idx])
        seg_logits = self.decoder(backbone_outputs[-1])
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            seg_logits = F.interpolate(seg_logits, size=img_size, mode='bilinear', align_corners=self.align_corners)
            predictions = {'loss_cls': seg_logits}
            backbone_outputs = backbone_outputs[:-1]
            for idx, (out, dec) in enumerate(zip(backbone_outputs, self.auxiliary_decoders)):
                seg_logits_aux = dec(out)
                seg_logits_aux = F.interpolate(seg_logits_aux, size=img_size, mode='bilinear', align_corners=self.align_corners)
                predictions[f'loss_aux{idx + 1}'] = seg_logits_aux
            loss, losses_log_dict = calculatelosses(predictions=predictions, annotations=data_meta.getannotations(), losses_cfg=self.cfg['losses'], pixel_sampler=self.pixel_sampler)
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs
    """norm"""

    def norm(self, x, norm_layer):
        n, c, h, w = x.shape
        x = x.reshape(n, c, h * w).transpose(2, 1).contiguous()
        x = norm_layer(x)
        x = x.transpose(1, 2).reshape(n, c, h, w).contiguous()
        return x
    """builddecoder"""

    def builddecoder(self, decoder_cfg):
        layers, norm_cfg, act_cfg, num_classes, align_corners, kernel_size = [], self.norm_cfg.copy(), self.act_cfg.copy(), self.cfg['num_classes'], self.align_corners, decoder_cfg['kernel_size']
        for idx in range(decoder_cfg['num_convs']):
            if idx == 0:
                layers.append(nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=kernel_size, stride=1, padding=int(kernel_size - 1) // 2, bias=False))
            else:
                layers.append(nn.Conv2d(decoder_cfg['out_channels'], decoder_cfg['out_channels'], kernel_size=kernel_size, stride=1, padding=int(kernel_size - 1) // 2, bias=False))
            layers.append(BuildNormalization(placeholder=decoder_cfg['out_channels'], norm_cfg=norm_cfg))
            layers.append(BuildActivation(act_cfg))
            layers.append(nn.Upsample(scale_factor=decoder_cfg['scale_factor'], mode='bilinear', align_corners=align_corners))
        layers.append(nn.Dropout2d(decoder_cfg['dropout']))
        layers.append(nn.Conv2d(decoder_cfg['out_channels'], num_classes, kernel_size=1, stride=1, padding=0))
        return nn.Sequential(*layers)


class SETRMLA(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(SETRMLA, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        norm_layers = nn.ModuleList()
        for in_channels in head_cfg['in_channels_list']:
            norm_cfg_copy = head_cfg['norm_cfg'].copy()
            norm_layer = BuildNormalization(placeholder=in_channels, norm_cfg=norm_cfg_copy)
            norm_layers.append(norm_layer)
        self.mla_neck = MLANeck(in_channels_list=head_cfg['in_channels_list'], out_channels=head_cfg['mla_feats_channels'], norm_layers=norm_layers, norm_cfg=norm_cfg, act_cfg=act_cfg)
        assert head_cfg['mla_up_channels'] * len(head_cfg['in_channels_list']) == head_cfg['feats_channels']
        self.up_convs = nn.ModuleList()
        for i in range(len(head_cfg['in_channels_list'])):
            self.up_convs.append(nn.Sequential(nn.Conv2d(head_cfg['mla_feats_channels'], head_cfg['mla_up_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['mla_up_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Conv2d(head_cfg['mla_up_channels'], head_cfg['mla_up_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['mla_up_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Upsample(scale_factor=head_cfg['scale_factor'], mode='bilinear', align_corners=align_corners)))
        self.decoder = nn.Sequential(nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        auxiliary_cfg_list = cfg['auxiliary']
        assert isinstance(auxiliary_cfg_list, (tuple, list))
        self.auxiliary_decoders = nn.ModuleList()
        for auxiliary_cfg in auxiliary_cfg_list:
            decoder = self.builddecoder(auxiliary_cfg)
            self.auxiliary_decoders.append(decoder)
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        feats_list = self.mla_neck(list(backbone_outputs))
        feats_outputs = []
        assert len(feats_list) == len(self.up_convs)
        for feats, up_conv in zip(feats_list, self.up_convs):
            feats_outputs.append(up_conv(feats))
        feats_outputs = torch.cat(feats_outputs, dim=1)
        seg_logits = self.decoder(feats_outputs)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            seg_logits = F.interpolate(seg_logits, size=img_size, mode='bilinear', align_corners=self.align_corners)
            predictions = {'loss_cls': seg_logits}
            feats_list = feats_list[-len(self.auxiliary_decoders):]
            for idx, (out, dec) in enumerate(zip(feats_list, self.auxiliary_decoders)):
                seg_logits_aux = dec(out)
                seg_logits_aux = F.interpolate(seg_logits_aux, size=img_size, mode='bilinear', align_corners=self.align_corners)
                predictions[f'loss_aux{idx + 1}'] = seg_logits_aux
            loss, losses_log_dict = calculatelosses(predictions=predictions, annotations=data_meta.getannotations(), losses_cfg=self.cfg['losses'])
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs
    """builddecoder"""

    def builddecoder(self, decoder_cfg):
        layers, norm_cfg, act_cfg, num_classes, align_corners, kernel_size = [], self.norm_cfg.copy(), self.act_cfg.copy(), self.cfg['num_classes'], self.align_corners, decoder_cfg['kernel_size']
        for idx in range(decoder_cfg['num_convs']):
            if idx == 0:
                layers.append(nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=kernel_size, stride=1, padding=int(kernel_size - 1) // 2, bias=False))
            else:
                layers.append(nn.Conv2d(decoder_cfg['out_channels'], decoder_cfg['out_channels'], kernel_size=kernel_size, stride=1, padding=int(kernel_size - 1) // 2, bias=False))
            layers.append(BuildNormalization(placeholder=decoder_cfg['out_channels'], norm_cfg=norm_cfg))
            layers.append(BuildActivation(act_cfg))
            layers.append(nn.Upsample(scale_factor=decoder_cfg['scale_factor'], mode='bilinear', align_corners=align_corners))
        layers.append(nn.Dropout2d(decoder_cfg['dropout']))
        layers.append(nn.Conv2d(decoder_cfg['out_channels'], num_classes, kernel_size=1, stride=1, padding=0))
        return nn.Sequential(*layers)


class UPerNet(BaseSegmentor):

    def __init__(self, cfg, mode):
        super(UPerNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        if 'feature2pyramid' in head_cfg:
            head_cfg['feature2pyramid']['norm_cfg'] = norm_cfg.copy()
            self.feats_to_pyramid_net = Feature2Pyramid(**head_cfg['feature2pyramid'])
        ppm_cfg = {'in_channels': head_cfg['in_channels_list'][-1], 'out_channels': head_cfg['feats_channels'], 'pool_scales': head_cfg['pool_scales'], 'align_corners': align_corners, 'norm_cfg': copy.deepcopy(norm_cfg), 'act_cfg': copy.deepcopy(act_cfg)}
        self.ppm_net = PyramidPoolingModule(**ppm_cfg)
        act_cfg_copy = copy.deepcopy(act_cfg)
        if 'inplace' in act_cfg_copy:
            act_cfg_copy['inplace'] = False
        self.lateral_convs = nn.ModuleList()
        for in_channels in head_cfg['in_channels_list'][:-1]:
            self.lateral_convs.append(nn.Sequential(nn.Conv2d(in_channels, head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg_copy)))
        self.fpn_convs = nn.ModuleList()
        for in_channels in ([head_cfg['feats_channels']] * len(self.lateral_convs)):
            self.fpn_convs.append(nn.Sequential(nn.Conv2d(in_channels, head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg_copy)))
        self.decoder = nn.Sequential(nn.Conv2d(head_cfg['feats_channels'] * len(head_cfg['in_channels_list']), head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False), BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg), BuildActivation(act_cfg), nn.Dropout2d(head_cfg['dropout']), nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.setauxiliarydecoder(cfg['auxiliary'])
        if cfg.get('is_freeze_norm', False):
            self.freezenormalization()
    """forward"""

    def forward(self, data_meta):
        img_size = data_meta.images.size(2), data_meta.images.size(3)
        backbone_outputs = self.transforminputs(self.backbone_net(data_meta.images), selected_indices=self.cfg['backbone'].get('selected_indices'))
        if hasattr(self, 'feats_to_pyramid_net'):
            backbone_outputs = self.feats_to_pyramid_net(backbone_outputs)
        ppm_out = self.ppm_net(backbone_outputs[-1])
        inputs = backbone_outputs[:-1]
        lateral_outputs = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        lateral_outputs.append(ppm_out)
        for i in range(len(lateral_outputs) - 1, 0, -1):
            prev_shape = lateral_outputs[i - 1].shape[2:]
            lateral_outputs[i - 1] = lateral_outputs[i - 1] + F.interpolate(lateral_outputs[i], size=prev_shape, mode='bilinear', align_corners=self.align_corners)
        fpn_outputs = [self.fpn_convs[i](lateral_outputs[i]) for i in range(len(lateral_outputs) - 1)]
        fpn_outputs.append(lateral_outputs[-1])
        fpn_outputs = [F.interpolate(out, size=fpn_outputs[0].size()[2:], mode='bilinear', align_corners=self.align_corners) for out in fpn_outputs]
        fpn_out = torch.cat(fpn_outputs, dim=1)
        seg_logits = self.decoder(fpn_out)
        if self.mode in ['TRAIN', 'TRAIN_DEVELOP']:
            loss, losses_log_dict = self.customizepredsandlosses(seg_logits=seg_logits, annotations=data_meta.getannotations(), backbone_outputs=backbone_outputs, losses_cfg=self.cfg['losses'], img_size=img_size)
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict) if self.mode == 'TRAIN' else SSSegOutputStructure(mode=self.mode, loss=loss, losses_log_dict=losses_log_dict, seg_logits=seg_logits)
        else:
            ssseg_outputs = SSSegOutputStructure(mode=self.mode, seg_logits=seg_logits)
        return ssseg_outputs


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ASPP,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'dilations': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AdaptivePadding,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AdptivePaddingConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Attention,
     lambda: ([], {'embedding_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (Attention2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'temperature': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AttentionRefinementModule,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BGALayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 128, 16, 16]), torch.rand([4, 128, 4, 4])], {})),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BasicConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (BiSeNetV2,
     lambda: ([], {'structure_type': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (CGNet,
     lambda: ([], {'structure_type': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (CXBlock,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CascadeFeatureFusion,
     lambda: ([], {'low_channels': 4, 'high_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ChannelAttentionModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ContextBlock,
     lambda: ([], {'in_channels': 4, 'ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv2dBN,
     lambda: ([], {'in_chans': 4, 'out_chans': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvMlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvNeXtBlock,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvNeXtV2Block,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CosineSimilarityLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (CrossAttentionLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (CrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (DeconvModule,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DepthwiseSeparableASPP,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'dilations': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DepthwiseSeparableConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DetailBranch,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (DiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (DisentangledNonLocal2d,
     lambda: ([], {'temperature': 4, 'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DynamicConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DynamicConvolutionalModule,
     lambda: ([], {'filter_size': 4, 'is_fusion': 4, 'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (EMAModule,
     lambda: ([], {'channels': 4, 'num_bases': 4, 'num_stages': 4, 'momentum': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ERFNet,
     lambda: ([], {'structure_type': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (EfficientMultiheadAttention,
     lambda: ([], {'embed_dims': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (Encoding,
     lambda: ([], {'channels': 4, 'num_codes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FFNLayer,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FastSCNN,
     lambda: ([], {'structure_type': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (Feature2Pyramid,
     lambda: ([], {'embed_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FeatureFusionModule,
     lambda: ([], {'higher_in_channels': 4, 'lower_in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (FocalLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (GELayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GRN,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GSAEncoderLayer,
     lambda: ([], {'embed_dims': 4, 'num_heads': 4, 'feedforward_channels': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (GlobalFeatureExtractor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {})),
    (GlobalSubsampledAttention,
     lambda: ([], {'embed_dims': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (HardSigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HardSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InputInjection,
     lambda: ([], {'num_downsamplings': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InterpConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InvertedResidual,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InvertedResidualV3,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'mid_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (KLDivLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (L1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (L2Norm,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm2d,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LearningToDownsample,
     lambda: ([], {'in_channels': 4, 'dw_channels': [4, 4], 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LinearSelfAttention,
     lambda: ([], {'embed_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LinearTransformerBlock,
     lambda: ([], {'embed_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LovaszLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MBConv,
     lambda: ([], {'in_chans': 4, 'out_chans': 4, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLAModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1024, 4, 4])], {})),
    (MLP,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLPBlock,
     lambda: ([], {'embedding_dim': 4, 'mlp_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MaskDownSampler,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {})),
    (MobileVitV2Block,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MultiScaleAttention,
     lambda: ([], {'dim': 4, 'dim_out': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MultiheadAttention,
     lambda: ([], {'embed_dims': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (NonBottleneck1d,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NonLocal1d,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NonLocal2d,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NonLocal3d,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PatchEmbed,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (PositionEmbeddingSine,
     lambda: ([], {'num_pos_feats': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RSoftmax,
     lambda: ([], {'radix': 4, 'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RepVGGDW,
     lambda: ([], {'ed': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Residual,
     lambda: ([], {'m': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Scale,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SelfAttentionLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (SemanticBranch,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (SpatialGatherModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SpatialPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (SplitAttentionConv2d,
     lambda: ([], {'in_channels': 4, 'channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SqueezeExcite,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (StemBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TransformerDecoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (TransformerEncoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (UpsamplerBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (_NonLocalNd,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

