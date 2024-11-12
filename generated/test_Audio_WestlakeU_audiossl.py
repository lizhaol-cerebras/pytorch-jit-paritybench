
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


import pandas as pd


import torch


import random


import numpy as np


import copy


import re


import logging


from collections import defaultdict


from itertools import chain


from torch.utils.data import Dataset


import warnings


from collections import OrderedDict


from torch.utils import data


import torch.nn.functional as F


from torch.utils.data import DistributedSampler


from random import shuffle as sf


from torch.utils.data import ConcatDataset


from torch.nn import Module


import torch.utils.data as data


from copy import deepcopy


from torch.utils.data import DataLoader


from torch import nn


import time


import torch.distributed as dist


from torch.utils.data import WeightedRandomSampler


from typing import Any


from typing import Iterable


from typing import Iterator


from typing import List


from typing import Optional


from typing import Sized


from typing import Tuple


from typing import Union


from torch.utils.data import Sampler


from torch.nn import functional as F


from torchvision import transforms


from torch import Tensor


from sklearn import metrics


from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


from functools import partial


import math


import torch.nn as nn


import torch.nn.functional


from torch.nn import LayerNorm


from typing import Dict


from torch.nn import Parameter


import torch.distributed as distributed


from random import randrange


from matplotlib import pyplot as plt


import scipy


from scipy import stats


from sklearn.metrics import auc


from torch.nn.modules.utils import _quadruple


import matplotlib.pyplot as plt


from matplotlib.patches import Polygon


import torchvision


from torchvision import transforms as pth_transforms


from torch.nn.functional import mse_loss


from enum import auto


from torch.utils.data import get_worker_info


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
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed_v2(nn.Module):

    def __init__(self, patch_height=16, patch_width=16, embed_dim=768):
        super().__init__()
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_maker = Rearrange('b c (h p1) (w p2) -> b (w h) (p1 p2 c)', p1=patch_height, p2=patch_width)
        self.patch_embed = nn.Linear(patch_height * patch_width, embed_dim)

    def forward(self, melspec, length=None):
        height = melspec.shape[2] - melspec.shape[2] % self.patch_height
        width = melspec.shape[3] - melspec.shape[3] % self.patch_width
        patch = self.patch_maker(melspec[:, :, :height, :width])
        patch_embed = self.patch_embed(patch)
        if length is not None:
            patch_length = height // self.patch_height * ((length - length % self.patch_width) // self.patch_width)
        else:
            patch_length = None
        return patch, patch_embed, patch_length


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

    def forward(self, x, mask):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * self.scale
        if mask is not None:
            attn += mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


def drop_path(x, drop_prob: 'float'=0.0, training: 'bool'=False):
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


def get_attention_mask(x, length):
    batch_size, max_len, _ = x.shape
    mask = torch.arange(max_len, device=length.device).expand(batch_size, max_len) >= length[:, None]
    mask = -10000.0 * mask[:, None, None, :]
    mask = mask.expand(batch_size, 1, max_len, max_len)
    return mask


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, length=None, return_attention=False):
        if length is not None:
            mask_att = get_attention_mask(x, length)
        else:
            mask_att = None
        y, attn = self.attn(self.norm1(x), mask_att)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        else:
            return x


def get_num_patches(height=64, width=1001, patch_height=16, patch_width=16):
    return height // patch_height * (width // patch_width)


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
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class FrameAST(nn.Module):
    """ Vision Transformer """

    def __init__(self, nprompt=0, spec_h=64, spec_w=1001, patch_w=16, patch_h=16, pos_type='cut', in_chans=1, num_classes=0, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.spec_w = spec_w
        self.spec_h = spec_h
        self.embed_dim = embed_dim
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.pos_type = pos_type
        self.patch_embed = PatchEmbed_v2(patch_h, patch_w, embed_dim)
        self.mask_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.nprompt = nprompt
        if self.nprompt > 0:
            self.prompt_embed = nn.Parameter(torch.zeros(1, self.nprompt, self.embed_dim))
            trunc_normal_(self.prompt_embed, std=0.02)
        num_patches = get_num_patches(spec_h, spec_w, patch_h, patch_w)
        self.num_patches = num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer) for i in range(depth)])
        self.norm_frame = norm_layer(embed_dim)
        None
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.mask_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(self, x, mask_index, length, mask=True):
        B, nc, h, w = x.shape
        mel_patches, x, patch_length = self.patch_embed(x, length)
        B, T, C = x.shape
        if mask_index is not None and mask:
            mask_index_expand = mask_index.unsqueeze(2).expand(B, T, self.embed_dim).float()
            x = (1 - mask_index_expand) * x + mask_index_expand * self.mask_embed.expand(B, T, C)
        if self.pos_type == 'cut':
            pos = self.pos_embed[:, 1:T + 1, :].expand(B, -1, -1)
            x = x + pos
        else:
            pos = self.interpolate_pos_encoding(x, h, w)
            x = x + pos[:, 1:]
        return self.pos_drop(x), pos, mel_patches, h, w, patch_length

    def forward(self, x, mask_index=None, mask_input=True, length=None):
        x, pos, mel_patches, h, w, patch_length = self.prepare_tokens(x, mask_index, length, mask_input)
        length_mask = torch.arange(mel_patches.shape[1]) < patch_length.unsqueeze(1)
        length_mask = length_mask
        mask_index = mask_index & length_mask
        if self.nprompt > 0:
            x = torch.cat([self.prompt_embed.expand(x.shape[0], -1, -1), x], dim=1)
        for i, blk in enumerate(self.blocks):
            x = blk(x, patch_length + self.nprompt)
        frame_repr = self.norm_frame(x)
        return frame_repr[:, self.nprompt:][mask_index]

    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == self.spec_w and h == self.spec_h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_width
        h0 = h // self.patch_embed.patch_height
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed.reshape(1, self.spec_h // self.patch_h, self.spec_w // self.patch_w, dim).permute(0, 3, 1, 2), scale_factor=(h0 / (self.spec_h // self.patch_h), w0 / (self.spec_w // self.patch_w)), mode='bicubic')
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def get_last_selfattention(self, x):
        x, _, _, _, _, _ = self.prepare_tokens(x, mask_index=None, length=None, mask=False)
        atts = []
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x, att = blk(x, return_attention=True)
                atts.append(att)
            else:
                x, att = blk(x, return_attention=True)
                atts.append(att)
                return atts

    def get_intermediate_layers(self, x, length, n=1, scene=True):
        x, _, _, _, _, patch_length = self.prepare_tokens(x, mask_index=None, length=length, mask=False)
        output = []
        if self.nprompt > 0:
            x = torch.cat([self.prompt_embed.expand(x.shape[0], -1, -1), x], dim=1)
        for i, blk in enumerate(self.blocks):
            x = blk(x, patch_length + self.nprompt)
            if len(self.blocks) - i <= n:
                norm_x = self.norm_frame(x)
                if scene:
                    length_mask = torch.arange(x.shape[1] - self.nprompt) < patch_length.unsqueeze(1)
                    avg = torch.sum(norm_x[:, self.nprompt:] * length_mask.unsqueeze(-1), dim=1) / (patch_length.unsqueeze(-1) + 1e-06)
                    negative = ~length_mask * -10000000000.0
                    output.append(avg)
                    if self.nprompt > 0:
                        output.append(torch.mean(norm_x[:, :self.nprompt], dim=1))
                else:
                    output.append(norm_x[:, self.nprompt:])
        return torch.cat(output, dim=-1)


def byol_loss_func(p: 'torch.Tensor', z: 'torch.Tensor', simplified: 'bool'=True):
    """
    Computes BYOL's loss given batch of predicted features p and projected momentum features z.
    Args:
        p (torch.Tensor): NxD Tensor containing predicted features from view 1
        z (torch.Tensor): NxD Tensor containing projected momentum features from view 2
        simplified (bool): faster computation, but with same result. Defaults to True.
    Returns:
        torch.Tensor: BYOL loss.
    """
    if simplified:
        return 2 - 2 * F.cosine_similarity(p, z, dim=-1).mean()
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2 - 2 * (p * z).sum(dim=1).mean()


def compute_var(y):
    y = y.view(-1, y.size(-1))
    zc = torch.tensor(y.size(0))
    zs = y.sum(dim=0)
    zss = (y ** 2).sum(dim=0)
    torch.distributed.all_reduce(zc)
    torch.distributed.all_reduce(zs)
    torch.distributed.all_reduce(zss)
    var = zss / (zc - 1) - zs ** 2 / (zc * (zc - 1))
    return torch.sqrt(var + 1e-06)


class ByolLoss(nn.Module):

    def __init__(self, ncrops):
        super().__init__()
        self.ncrops = ncrops

    def forward(self, student, teacher):
        std_cls_s = compute_var(F.normalize(student, dim=-1)).mean()
        std_cls_t = compute_var(F.normalize(teacher, dim=-1)).mean()
        student = student.chunk(self.ncrops)
        teacher = teacher.detach().chunk(2)
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher):
            for iv, v in enumerate(student):
                if iq == iv:
                    continue
                loss = byol_loss_func(q, v, simplified=False)
                n_loss_terms += 1
                total_loss += loss.mean()
        total_loss /= n_loss_terms
        return total_loss, std_cls_s, std_cls_t


def build_mlp(num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
    mlp = []
    for l in range(num_layers):
        dim1 = input_dim if l == 0 else mlp_dim
        dim2 = output_dim if l == num_layers - 1 else mlp_dim
        mlp.append(nn.Linear(dim1, dim2, bias=False))
        if l < num_layers - 1:
            mlp.append(nn.BatchNorm1d(dim2))
            mlp.append(nn.ReLU(inplace=True))
        elif last_bn:
            mlp.append(nn.BatchNorm1d(dim2, affine=False))
    return nn.Sequential(*mlp)


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, encoder, embed_dim, predictor=True):
        super(MultiCropWrapper, self).__init__()
        self.encoder = encoder
        self.projector = build_mlp(2, embed_dim, 4096, 256, last_bn=False)
        if predictor:
            self.predictor = build_mlp(2, 256, 4096, 256, last_bn=False)
        else:
            self.predictor = nn.Identity()

    def forward(self, x, length, avg=False):
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(torch.tensor([inp.shape[-1] for inp in x]), return_counts=True)[1], 0)
        start_idx, output = 0, torch.empty(0)
        for end_idx in idx_crops:
            _out = self.encoder(torch.cat(x[start_idx:end_idx]), length=torch.cat(length[start_idx:end_idx]), avg=avg)
            output = torch.cat((output, _out))
            start_idx = end_idx
        return self.predictor(self.projector(output))


class PrecomputedNorm(nn.Module):
    """Normalization using Pre-computed Mean/Std.
    Args:
        stats: Precomputed (mean, std).
        axis: Axis setting used to calculate mean/variance.
    """

    def __init__(self, stats):
        super().__init__()
        self.mean, self.std = stats

    def forward(self, X: 'torch.Tensor') ->torch.Tensor:
        return (X - self.mean) / self.std

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'
        return format_string


class PatchEmbed_org(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = img_size[1] // patch_size[1] * (img_size[0] // patch_size[0])
        self.patch_hw = img_size[1] // patch_size[1], img_size[0] // patch_size[0]
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed_new(nn.Module):
    """ Flexible Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        _, _, h, w = self.get_output_shape(img_size)
        self.patch_hw = h, w
        self.num_patches = h * w

    def get_output_shape(self, img_size):
        return self.proj(torch.randn(1, 1, img_size[0], img_size[1])).shape

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class PatchEmbed3D_new(nn.Module):
    """ Flexible Image to Patch Embedding
    """

    def __init__(self, video_size=(16, 224, 224), patch_size=(2, 16, 16), in_chans=3, embed_dim=768, stride=(2, 16, 16)):
        super().__init__()
        self.video_size = video_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        _, _, t, h, w = self.get_output_shape(video_size)
        self.patch_thw = t, h, w
        self.num_patches = t * h * w

    def get_output_shape(self, video_size):
        return self.proj(torch.randn(1, self.in_chans, video_size[0], video_size[1], video_size[2])).shape

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """
    if p <= 0:
        return module
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))
    is_conv = module.weight.ndim == 4
    if not is_conv:
        assert module.weight.size(1) % block_size == 0, 'Input features must be a multiple of block sizes'
    elif module.kernel_size == (1, 1):
        assert module.in_channels % block_size == 0, 'Input channels must be a multiple of block sizes'
    else:
        k = module.kernel_size[0] * module.kernel_size[1]
        assert k % block_size == 0, 'Kernel size must be a multiple of block size'

    def _forward_pre_hook(mod, input):
        if mod.training:
            if not is_conv:
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)
                mask = torch.zeros(in_features // block_size * out_features, device=weight.device)
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)
            else:
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(int(in_channels // block_size * out_channels), device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(weight.size(0), weight.size(1), device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
            mask = mask
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)
    module.register_forward_pre_hook(_forward_pre_hook)
    return module


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, self_attention=False, encoder_decoder_attention=False, q_noise=0.0, qn_block_size=8, has_relative_attention_bias=False, num_buckets=32, max_distance=128, gru_rel_pos=False, rescale_init=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)
        self.has_relative_attention_bias = has_relative_attention_bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
        self.head_dim = embed_dim // num_heads
        self.q_head_dim = self.head_dim
        self.k_head_dim = self.head_dim
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.scaling = self.head_dim ** -0.5
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert not self.self_attention or self.qkv_same_dim, 'Self-attention requires query, key and value to be of the same size'
        k_bias = True
        if rescale_init:
            k_bias = False
        k_embed_dim = embed_dim
        q_embed_dim = embed_dim
        self.k_proj = quant_noise(nn.Linear(self.kdim, k_embed_dim, bias=k_bias), q_noise, qn_block_size)
        self.v_proj = quant_noise(nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.q_proj = quant_noise(nn.Linear(embed_dim, q_embed_dim, bias=bias), q_noise, qn_block_size)
        self.out_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        self.add_zero_attn = add_zero_attn
        self.gru_rel_pos = gru_rel_pos
        if self.gru_rel_pos:
            self.grep_linear = nn.Linear(self.q_head_dim, 8)
            self.grep_a = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
        if self.has_relative_attention_bias:
            nn.init.xavier_normal_(self.relative_attention_bias.weight)

    def _relative_positions_bucket(self, relative_positions, bidirectional=True):
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        relative_buckets = 0
        if bidirectional:
            num_buckets = num_buckets // 2
            relative_buckets += (relative_positions > 0) * num_buckets
            relative_positions = torch.abs(relative_positions)
        else:
            relative_positions = -torch.min(relative_positions, torch.zeros_like(relative_positions))
        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact
        relative_postion_if_large = max_exact + torch.log(relative_positions.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        relative_postion_if_large = torch.min(relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1))
        relative_buckets += torch.where(is_small, relative_positions, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(relative_position, bidirectional=True)
        relative_position_bucket = relative_position_bucket
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1])
        return values

    def forward(self, query, key: 'Optional[Tensor]', value: 'Optional[Tensor]', key_padding_mask: 'Optional[Tensor]'=None, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]'=None, need_weights: 'bool'=True, static_kv: 'bool'=False, attn_mask: 'Optional[Tensor]'=None, before_softmax: 'bool'=False, need_head_weights: 'bool'=False, position_bias: 'Optional[Tensor]'=None) ->Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True
        is_tpu = query.device.type == 'xla'
        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]
        if self.has_relative_attention_bias and position_bias is None:
            position_bias = self.compute_bias(tgt_len, src_len)
            position_bias = position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1).view(bsz * self.num_heads, tgt_len, src_len)
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and 'prev_key' in saved_state:
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None
        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling
        alpha = 32
        q *= 1 / alpha
        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.q_head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.k_head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if saved_state is not None:
            if 'prev_key' in saved_state:
                _prev_key = saved_state['prev_key']
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if 'prev_value' in saved_state:
                _prev_value = saved_state['prev_value']
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: 'Optional[Tensor]' = None
            if 'prev_key_padding_mask' in saved_state:
                prev_key_padding_mask = saved_state['prev_key_padding_mask']
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(key_padding_mask=key_padding_mask, prev_key_padding_mask=prev_key_padding_mask, batch_size=bsz, src_len=k.size(1), static_kv=static_kv)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_key_padding_mask'] = key_padding_mask
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        assert k.size(1) == src_len
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = (attn_weights - attn_weights.max(dim=-1, keepdim=True)[0]) * alpha
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if before_softmax:
            return attn_weights, v, position_bias
        if position_bias is not None:
            attn_mask_rel_pos = position_bias
            if self.gru_rel_pos == 1:
                query_layer = q.view(bsz, self.num_heads, tgt_len, self.q_head_dim) * alpha / self.scaling
                _B, _H, _L, __ = query_layer.size()
                gate_a, gate_b = torch.sigmoid(self.grep_linear(query_layer).view(_B, _H, _L, 2, 4).sum(-1, keepdim=False)).chunk(2, dim=-1)
                gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                attn_mask_rel_pos = gate_a_1.view(bsz * self.num_heads, tgt_len, 1) * position_bias
            attn_mask_rel_pos = attn_mask_rel_pos.view(attn_weights.size())
            attn_weights = attn_weights + attn_mask_rel_pos
        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)
        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: 'Optional[Tensor]' = None
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                attn_weights = attn_weights.mean(dim=0)
        return attn, attn_weights, position_bias

    @staticmethod
    def _append_prev_key_padding_mask(key_padding_mask: 'Optional[Tensor]', prev_key_padding_mask: 'Optional[Tensor]', batch_size: 'int', src_len: 'int', static_kv: 'bool') ->Optional[Tensor]:
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), key_padding_mask.float()], dim=1)
        elif prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros((batch_size, src_len - prev_key_padding_mask.size(1)), device=prev_key_padding_mask.device)
                new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), filler.float()], dim=1)
            else:
                new_key_padding_mask = prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros((batch_size, src_len - key_padding_mask.size(1)), device=key_padding_mask.device)
                new_key_padding_mask = torch.cat([filler.float(), key_padding_mask.float()], dim=1)
            else:
                new_key_padding_mask = key_padding_mask.float()
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def _get_input_buffer(self, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]') ->Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, 'attn_state')
        if result is not None:
            return result
        else:
            empty_result: 'Dict[str, Optional[Tensor]]' = {}
            return empty_result

    def _set_input_buffer(self, incremental_state: 'Dict[str, Dict[str, Optional[Tensor]]]', buffer: 'Dict[str, Optional[Tensor]]'):
        return self.set_incremental_state(incremental_state, 'attn_state', buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: 'int', src_len: 'int', bsz: 'int'):
        return attn_weights


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(self, embedding_dim: 'float'=768, ffn_embedding_dim: 'float'=3072, num_attention_heads: 'float'=8, dropout: 'float'=0.1, attention_dropout: 'float'=0.1, activation_dropout: 'float'=0.1, activation_fn: 'str'='relu', layer_norm_first: 'bool'=False) ->None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(self.embedding_dim, num_attention_heads, dropout=attention_dropout, self_attention=True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.layer_norm_first = layer_norm_first
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(self, x: 'torch.Tensor', self_attn_mask: 'torch.Tensor'=None, self_attn_padding_mask: 'torch.Tensor'=None, need_weights: 'bool'=False, att_args=None):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x
        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(query=x, key=x, value=x, key_padding_mask=self_attn_padding_mask, attn_mask=self_attn_mask)
            x = self.dropout1(x)
            x = residual + x
            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(query=x, key=x, value=x, key_padding_mask=self_attn_padding_mask)
            x = self.dropout1(x)
            x = residual + x
            x = self.self_attn_layer_norm(x)
            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)
        return x, attn


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02))
    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class TransformerEncoder(nn.Module):

    def __init__(self, args, decoder=False):
        super().__init__()
        self.dropout = args.dropout
        layers = []
        if decoder:
            num_layers = args.decoder_layers
            self.embedding_dim = args.decoder_embed_dim
            self.layerdrop = args.decoder_layerdrop
        else:
            num_layers = args.encoder_layers
            self.embedding_dim = args.encoder_embed_dim
            self.layerdrop = args.encoder_layerdrop
        for _ in range(num_layers):
            layer = TransformerSentenceEncoderLayer(embedding_dim=self.embedding_dim, ffn_embedding_dim=args.encoder_ffn_embed_dim, num_attention_heads=args.encoder_attention_heads, dropout=self.dropout, attention_dropout=args.attention_dropout, activation_dropout=args.activation_dropout, activation_fn=args.activation_fn, layer_norm_first=args.layer_norm_first)
            if args.checkpoint_activations:
                layer = fsdp_wrap(layer)
                layer = checkpoint_wrapper(layer)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, layer=None):
        x, layer_results = self.extract_features(x, padding_mask, layer)
        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)
        return x, layer_results

    def extract_features(self, x, padding_mask=None, tgt_layer=None):
        if not self.layer_norm_first:
            x = self.layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)
        hidden_states = []
        r = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or dropout_probability > self.layerdrop:
                x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
                hidden_states.append(x.transpose(0, 1))
            if i == tgt_layer:
                r = x
                break
        if r is not None:
            x = r
        x = x.transpose(0, 1)
        return x, hidden_states

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.cfg.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


logger = logging.getLogger(__name__)


class BEATs(nn.Module):

    def __init__(self, cfg: 'BEATsConfig') ->None:
        super().__init__()
        logger.info(f'BEATs Config: {cfg.__dict__}')
        self.cfg = cfg
        self.embed = cfg.embed_dim
        self.post_extract_proj = nn.Linear(self.embed, cfg.encoder_embed_dim) if self.embed != cfg.encoder_embed_dim else None
        self.input_patch_size = cfg.input_patch_size
        self.patch_embedding = nn.Conv2d(1, self.embed, kernel_size=self.input_patch_size, stride=self.input_patch_size, bias=cfg.conv_bias)
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        assert not cfg.deep_norm or not cfg.layer_norm_first
        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)
        if cfg.finetuned_model:
            self.predictor_dropout = nn.Dropout(cfg.predictor_dropout)
            self.predictor = nn.Linear(cfg.encoder_embed_dim, cfg.predictor_class)
        else:
            self.predictor = None

    def forward_padding_mask(self, features: 'torch.Tensor', padding_mask: 'torch.Tensor') ->torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def preprocess(self, source: 'torch.Tensor', fbank_mean: 'float'=15.41663, fbank_std: 'float'=6.55582) ->torch.Tensor:
        fbanks = []
        for waveform in source:
            waveform = waveform.unsqueeze(0) * 2 ** 15
            fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)
        fbank = (fbank - fbank_mean) / (2 * fbank_std)
        return fbank

    def extract_features(self, fbank: 'torch.Tensor', padding_mask: 'Optional[torch.Tensor]'=None):
        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(fbank, padding_mask)
        fbank = fbank.unsqueeze(1)
        features = self.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        x = self.dropout_input(features)
        x, layer_results = self.encoder(x, padding_mask=padding_mask)
        if self.predictor is not None:
            x = self.predictor_dropout(x)
            logits = self.predictor(x)
            if padding_mask is not None and padding_mask.any():
                logits[padding_mask] = 0
                logits = logits.sum(dim=1)
                logits = logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(logits)
            else:
                logits = logits.mean(dim=1)
            lprobs = torch.sigmoid(logits)
            return lprobs, padding_mask
        else:
            return x, padding_mask


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)
    return samples[indices]


def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device
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
            new_means = l2norm(new_means)
        means = torch.where(zero_mask[..., None], means, new_means)
    return means, bins


class EmbeddingEMA(nn.Module):

    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-05, kmeans_init=True, codebook_init_path=''):
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.eps = eps
        if codebook_init_path == '':
            if not kmeans_init:
                weight = torch.randn(num_tokens, codebook_dim)
                weight = l2norm(weight)
            else:
                weight = torch.zeros(num_tokens, codebook_dim)
            self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        else:
            None
            codebook_ckpt_weight = torch.load(codebook_init_path, map_location='cpu')
            weight = codebook_ckpt_weight.clone()
            self.register_buffer('initted', torch.Tensor([True]))
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)
        self.update = True

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return
        None
        embed, cluster_size = kmeans(data, self.num_tokens, 10, use_cosine_sim=True)
        self.weight.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg):
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized)


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=1 - decay)


def norm_ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=1 - decay)
    moving_avg.data.copy_(l2norm(moving_avg.data))


class NormEMAVectorQuantizer(nn.Module):

    def __init__(self, n_embed, embedding_dim, beta, decay=0.99, eps=1e-05, statistic_code_usage=True, kmeans_init=False, codebook_init_path=''):
        super().__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.beta = beta
        self.decay = decay
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps, kmeans_init, codebook_init_path)
        self.statistic_code_usage = statistic_code_usage
        if statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(n_embed))
        if distributed.is_available() and distributed.is_initialized():
            None
            self.all_reduce_fn = distributed.all_reduce
        else:
            self.all_reduce_fn = nn.Identity()

    def reset_cluster_size(self, device):
        if self.statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(self.num_tokens))
            self.cluster_size = self.cluster_size

    def forward(self, z):
        z = l2norm(z)
        z_flattened = z.reshape(-1, self.codebook_dim)
        self.embedding.init_embed_(z_flattened)
        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + self.embedding.weight.pow(2).sum(dim=1) - 2 * torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight)
        encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(encoding_indices).view(z.shape)
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)
        if not self.training:
            with torch.no_grad():
                cluster_size = encodings.sum(0)
                self.all_reduce_fn(cluster_size)
                ema_inplace(self.cluster_size, cluster_size, self.decay)
        if self.training and self.embedding.update:
            bins = encodings.sum(0)
            self.all_reduce_fn(bins)
            ema_inplace(self.cluster_size, bins, self.decay)
            zero_mask = bins == 0
            bins = bins.masked_fill(zero_mask, 1.0)
            embed_sum = z_flattened.t() @ encodings
            self.all_reduce_fn(embed_sum)
            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
            embed_normalized = l2norm(embed_normalized)
            embed_normalized = torch.where(zero_mask[..., None], self.embedding.weight, embed_normalized)
            norm_ema_inplace(self.embedding.weight, embed_normalized, self.decay)
        loss = self.beta * F.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()
        return z_q, loss, encoding_indices


class Tokenizers(nn.Module):

    def __init__(self, cfg: 'TokenizersConfig') ->None:
        super().__init__()
        logger.info(f'Tokenizers Config: {cfg.__dict__}')
        self.cfg = cfg
        self.embed = cfg.embed_dim
        self.post_extract_proj = nn.Linear(self.embed, cfg.encoder_embed_dim) if self.embed != cfg.encoder_embed_dim else None
        self.input_patch_size = cfg.input_patch_size
        self.patch_embedding = nn.Conv2d(1, self.embed, kernel_size=self.input_patch_size, stride=self.input_patch_size, bias=cfg.conv_bias)
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        assert not cfg.deep_norm or not cfg.layer_norm_first
        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)
        self.quantize = NormEMAVectorQuantizer(n_embed=cfg.quant_n, embedding_dim=cfg.quant_dim, beta=1.0, kmeans_init=True, decay=0.99)
        self.quant_n = cfg.quant_n
        self.quantize_layer = nn.Sequential(nn.Linear(cfg.encoder_embed_dim, cfg.encoder_embed_dim), nn.Tanh(), nn.Linear(cfg.encoder_embed_dim, cfg.quant_dim))

    def forward_padding_mask(self, features: 'torch.Tensor', padding_mask: 'torch.Tensor') ->torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def preprocess(self, source: 'torch.Tensor', fbank_mean: 'float'=15.41663, fbank_std: 'float'=6.55582) ->torch.Tensor:
        fbanks = []
        for waveform in source:
            waveform = waveform.unsqueeze(0) * 2 ** 15
            fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)
        fbank = (fbank - fbank_mean) / (2 * fbank_std)
        return fbank

    def extract_labels(self, source: 'torch.Tensor', padding_mask: 'Optional[torch.Tensor]'=None, fbank_mean: 'float'=15.41663, fbank_std: 'float'=6.55582):
        fbank = self.preprocess(source, fbank_mean=fbank_mean, fbank_std=fbank_std)
        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(fbank, padding_mask)
        fbank = fbank.unsqueeze(1)
        features = self.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        x = self.dropout_input(features)
        x, layer_results = self.encoder(x, padding_mask=padding_mask)
        quantize_input = self.quantize_layer(x)
        quantize_feature, embed_loss, embed_ind = self.quantize(quantize_input)
        return embed_ind


class SamePad(nn.Module):

    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, :-self.remove]
        return x


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        return x * self.act(x)


class GLU_Linear(nn.Module):

    def __init__(self, input_dim, output_dim, glu_type='sigmoid', bias_in_glu=True):
        super(GLU_Linear, self).__init__()
        self.glu_type = glu_type
        self.output_dim = output_dim
        if glu_type == 'sigmoid':
            self.glu_act = torch.nn.Sigmoid()
        elif glu_type == 'swish':
            self.glu_act = Swish()
        elif glu_type == 'relu':
            self.glu_act = torch.nn.ReLU()
        elif glu_type == 'gelu':
            self.glu_act = torch.nn.GELU()
        if bias_in_glu:
            self.linear = nn.Linear(input_dim, output_dim * 2, True)
        else:
            self.linear = nn.Linear(input_dim, output_dim * 2, False)

    def forward(self, x):
        x = self.linear(x)
        if self.glu_type == 'bilinear':
            x = x[:, :, 0:self.output_dim] * x[:, :, self.output_dim:self.output_dim * 2]
        else:
            x = x[:, :, 0:self.output_dim] * self.glu_act(x[:, :, self.output_dim:self.output_dim * 2])
        return x


class AudioNTT2022Encoder(nn.Module):
    """General Audio Feature Encoder Network"""

    def __init__(self, n_mels, d=3072, base_d=64, mlp_hidden_d=2048, conv_layers=2, stack=True):
        super().__init__()
        convs = [nn.Conv2d(1, base_d, 3, stride=1, padding=1), nn.BatchNorm2d(base_d), nn.ReLU(), nn.MaxPool2d(2, stride=2)]
        for c in range(1, conv_layers):
            convs.extend([nn.Conv2d(base_d, base_d, 3, stride=1, padding=1), nn.BatchNorm2d(base_d), nn.ReLU(), nn.MaxPool2d(2, stride=2)])
        self.features = nn.Sequential(*convs)
        self.conv_d = base_d * (n_mels // 2 ** conv_layers)
        self.fc = nn.Sequential(nn.Linear(self.conv_d, mlp_hidden_d), nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(mlp_hidden_d, d - self.conv_d), nn.ReLU())
        self.stack = stack

    def forward(self, x):
        x = self.features(x)
        x = x.permute(0, 3, 2, 1)
        B, T, D, C = x.shape
        x = x.reshape((B, T, C * D))
        x_fc = self.fc(x)
        x = torch.hstack([x.transpose(1, 2), x_fc.transpose(1, 2)]).transpose(1, 2) if self.stack else x_fc
        return x


def mean_max_pooling(frame_embeddings):
    assert len(frame_embeddings.shape) == 3
    x1, _ = torch.max(frame_embeddings, dim=1)
    x2 = torch.mean(frame_embeddings, dim=1)
    x = x1 + x2
    return x


class AudioNTT2022(AudioNTT2022Encoder):

    def __init__(self, n_mels, d=3072, mlp_hidden_d=2048):
        super().__init__(n_mels=n_mels, d=d, mlp_hidden_d=mlp_hidden_d)
        self.embed_dim = d

    def forward(self, x):
        x = super().forward(x)
        x = mean_max_pooling(x)
        return x


def FrameAST_base(patch_h=64, patch_w=4, **kwargs):
    return FrameAST(patch_h=patch_h, patch_w=patch_w, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, norm_layer=partial(nn.LayerNorm, eps=1e-06), **kwargs)


def FrameAST_small(patch_h=64, patch_w=4, **kwargs):
    return FrameAST(patch_h=patch_h, patch_w=patch_w, embed_dim=384, depth=12, num_heads=6, qkv_bias=False, norm_layer=partial(nn.LayerNorm, eps=1e-06), **kwargs)


class FrameATST(nn.Module):

    def __init__(self, arch='small', symmetric=True, pos_type='cut', avg_blocks=0, patch_embed='Linear', **kwargs):
        super().__init__()
        if arch == 'small':
            encoder_fn = FrameAST_small
            embed_dim = 384
        elif arch == 'base':
            encoder_fn = FrameAST_base
            embed_dim = 768
        else:
            raise RuntimeError('arch {} is not implemented'.format(arch))
        self.symmetric = symmetric
        if avg_blocks == 0:
            self.student = MultiCropWrapper(encoder_fn(pos_type=pos_type, patch_embed=patch_embed, **kwargs), embed_dim, predictor=True)
            self.teacher = MultiCropWrapper(encoder_fn(pos_type=pos_type, patch_embed=patch_embed, **kwargs), embed_dim, predictor=False)
        else:
            self.student = MultiCropWrapper(encoder_fn(pos_type=pos_type, patch_embed=patch_embed, **kwargs), embed_dim, projector='linear', predictor=False)
            self.teacher = MultiCropWrapper(encoder_fn(pos_type=pos_type, patch_embed=patch_embed, avg_blocks=8, **kwargs), embed_dim, projector=None, predictor=False)
        for p in self.teacher.parameters():
            p.requires_grad = False
        if avg_blocks == 0:
            self.teacher.load_state_dict({k: v for k, v in self.student.state_dict().items() if 'predictor' not in k})
        else:
            self.teacher.load_state_dict({k: v for k, v in self.student.state_dict().items() if 'projector' not in k})
        self.loss_fn = ByolLoss(symmetric=symmetric)

    def forward(self, x, length, mask):
        if self.symmetric:
            tea = self.teacher(x, length, mask, False)
            stu = self.student(x, length, mask, True)
            return self.loss_fn(stu, tea)
        else:
            tea = self.teacher(x[:1], length[:1], mask[:1], False)
            stu = self.student(x[1:], length[1:], mask[1:], True)
            return self.loss_fn(stu, tea)

    def update_teacher(self, m):
        with torch.no_grad():
            for param_q, param_k in zip(self.student.encoder.parameters(), self.teacher.encoder.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            for param_q, param_k in zip(self.student.projector.parameters(), self.teacher.projector.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def _init_teacher(self):
        self.teacher.load_state_dict({k: v for k, v in self.student.state_dict().items() if 'predictor' not in k})


class ConvPosEmbed(nn.Module):

    def __init__(self, args, decoder=False):
        super().__init__()
        self.embedding_dim = args.encoder_embed_dim
        self.pos_conv = nn.Conv1d(self.embedding_dim, self.embedding_dim, kernel_size=args.conv_pos, padding=args.conv_pos // 2, groups=args.conv_pos_groups)
        dropout = 0
        std = math.sqrt(4 * (1.0 - dropout) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)
        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name='weight', dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

    def forward(self, x, padding_mask=None):
        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)
        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        return x_conv


class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, d_model: 'int', max_len: 'int'=480000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, padding_mask):
        """
        Args:
            x: Tensor, shape [bsz, seq_len, embedding_dim]
        """
        pe = self.pe[:, :x.shape[1]].repeat((padding_mask.shape[0], 1, 1))
        pe[padding_mask] = 0.0
        return pe


class Config:
    weight_file = ''
    feature_d = 768 * 5
    norm_type = all
    pooling_type = 'mean'
    model = ''
    input_size = [80, 208]
    patch_size = [16, 16]
    cls_token = False
    training_mask = 0.0
    flat_features = False
    sample_rate = 16000
    n_fft = 400
    window_size = 400
    hop_size = 160
    n_mels = 80
    f_min = 50
    f_max = 8000
    window = 'hanning'


def drop_non_model_weights(model, checkpoint, filename):
    model_keys = [n for n, p in model.named_parameters()]
    new_ckpt = {}
    for k in checkpoint:
        if k not in model_keys:
            continue
        new_ckpt[k] = checkpoint[k]
    n_org = len(checkpoint.keys())
    n_cur = len(new_ckpt.keys())
    None
    return new_ckpt


max = -10000000000.0


def parse_sizes_by_name(name):
    model_cls = name.split('-')[0]
    params = name.split('-')[1]
    input_str, patch_str = params.split('p')[:2]
    input_size = [int(a) for a in input_str.split('x')]
    patch_size = [int(a) for a in patch_str.split('x')]
    return input_size, patch_size, model_cls


def get_model(args, weight_file, encoder_only):
    folder_name = Path(weight_file).parent.name
    args.input_size, args.patch_size, args.model = parse_sizes_by_name(folder_name)
    if encoder_only:
        args.model = args.model + '_encoder_only'
    if Path(weight_file).name.endswith('random'):
        checkpoint = None
        dec_blocks_nums = [4 - 1]
        None
        logging.info(' **CAUTION: Random Weights**')
    else:
        checkpoint = torch.load(weight_file, map_location='cpu')
        checkpoint = checkpoint['model'] if 'model' in checkpoint else checkpoint
        dec_blocks_nums = [int(k.split('.')[1]) for k in checkpoint.keys() if k.startswith('decoder_blocks.')]
    args.decoder_depth = max(dec_blocks_nums) + 1
    logging.info(f'Creating model: {args.model}(input={args.input_size}, patch={args.patch_size}, decoder_depth={args.decoder_depth})')
    model = models_mae.__dict__[args.model](img_size=args.input_size, patch_size=args.patch_size, decoder_depth=args.decoder_depth)
    args.flat_features = True if args.training_mask > 0.0 else args.flat_features
    n_stack_feature = 1 if args.flat_features else args.input_size[0] // args.patch_size[0]
    d = model.pos_embed.shape[-1]
    args.feature_d = d * n_stack_feature
    if checkpoint:
        checkpoint = drop_non_model_weights(model, checkpoint, weight_file)
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def get_timestamps(cfg, batch_audio, x):
    audio_len = len(batch_audio[0])
    sec = audio_len / cfg.sample_rate
    x_len = len(x[0])
    step = sec / x_len * 1000
    ts = torch.tensor([(step * i) for i in range(x_len)]).unsqueeze(0)
    ts = ts.repeat(len(batch_audio), 1)
    return ts


def get_to_melspec(cfg):
    to_spec = nnAudio.features.MelSpectrogram(sr=cfg.sample_rate, n_fft=cfg.n_fft, win_length=cfg.window_size, hop_length=cfg.hop_size, n_mels=cfg.n_mels, fmin=cfg.f_min, fmax=cfg.f_max, center=True, power=2, verbose=False)
    logging.info(f'Runtime MelSpectrogram({cfg.sample_rate}, {cfg.n_fft}, {cfg.window_size}, {cfg.hop_size}, ' + f'{cfg.n_mels}, {cfg.f_min}, {cfg.f_max}):')
    logging.info(to_spec)
    return to_spec


class RuntimeM2D(nn.Module):

    def __init__(self, cfg=Config(), weight_file=None, training_mask=0.0, encoder_only=False):
        super().__init__()
        cfg.weight_file = weight_file or cfg.weight_file
        cfg.training_mask = training_mask if training_mask > 0.0 else cfg.training_mask
        self.cfg = cfg
        self.backbone = get_model(cfg, cfg.weight_file, encoder_only)
        if self.is_training_mask():
            self.backbone.set_random_structured_mask()
        logging.info(str(cfg))
        logging.info(f'Model input size: {cfg.input_size}')
        logging.info(f'Using weights: {cfg.weight_file}')
        logging.info(f'[CLS] token?: {cfg.cls_token}')
        logging.info(f'training_mask: {cfg.training_mask}')
        logging.info(f'flat_features: {cfg.flat_features}')
        self.to_spec = get_to_melspec(cfg)
        self.sample_rate = cfg.sample_rate

    def is_training_mask(self):
        return self.cfg.training_mask > 0.0

    def to_feature(self, batch_audio):
        x = self.to_spec(batch_audio)
        x = (x + torch.finfo().eps).log()
        return x

    def normalize_batch(self, x, return_stats=False):
        mu, sigma = x.mean(), x.std()
        x = (x - mu) / sigma
        if return_stats:
            return x, (mu, sigma)
        return x

    def to_normalized_spec(self, batch_audio, return_stats=False):
        x = self.to_feature(batch_audio)
        x = self.normalize_batch(x, return_stats=return_stats)
        return x

    def encode_lms(self, lms, return_layers=False):
        x = lms
        patch_fbins = self.backbone.grid_size()[0]
        unit_frames = self.cfg.input_size[1]
        embed_d = self.backbone.patch_embed.proj.out_channels
        cur_frames = x.shape[-1]
        pad_frames = unit_frames - cur_frames % unit_frames
        if pad_frames > 0:
            x = torch.nn.functional.pad(x, (0, pad_frames))
        embeddings = []
        if self.cfg.flat_features:
            mask_ratio = self.cfg.training_mask if self.training else 0.0
            for i in range(x.shape[-1] // unit_frames):
                emb, *_ = self.backbone.forward_encoder(x[..., i * unit_frames:(i + 1) * unit_frames], mask_ratio=mask_ratio, return_layers=return_layers)
                cls_token, emb = emb[..., :1, :], emb[..., 1:, :]
                if self.cfg.cls_token:
                    emb = torch.cat([cls_token, emb], axis=-1)
                embeddings.append(emb)
            x = torch.cat(embeddings, axis=-2)
        else:
            for i in range(x.shape[-1] // unit_frames):
                emb, *_ = self.backbone.forward_encoder(x[..., i * unit_frames:(i + 1) * unit_frames], mask_ratio=0.0, return_layers=return_layers)
                cls_token, emb = emb[..., :1, :], emb[..., 1:, :]
                if len(emb.shape) > 3:
                    emb = rearrange(emb, 'L b (f t) d -> L b t (f d)', f=patch_fbins, d=embed_d)
                else:
                    emb = rearrange(emb, 'b (f t) d -> b t (f d)', f=patch_fbins, d=embed_d)
                if self.cfg.cls_token:
                    emb = torch.cat([cls_token.repeat(*([1] * (len(emb.shape) - 2)), emb.shape[-2], 1), emb], axis=-1)
                embeddings.append(emb)
            x = torch.cat(embeddings, axis=-2)
            pad_emb_frames = int(embeddings[0].shape[-2] * pad_frames / unit_frames)
            if pad_emb_frames > 0:
                x = x[..., :-pad_emb_frames, :]
        return x if len(emb.shape) == 3 else [x_ for x_ in x]

    def encode(self, batch_audio):
        x = self.to_normalized_spec(batch_audio)
        return self.encode_lms(x)

    def get_scene_embeddings(self, audio):
        """
        audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length.
        Returns:
            embedding: A float32 Tensor with shape (n_sounds, model.scene_embedding_size).
        """
        x = self.encode(audio)
        x = torch.mean(x, dim=1)
        return x

    def get_timestamp_embeddings(self, audio):
        """
        audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length.
        Returns:
            embedding: A float32 Tensor with shape (n_sounds, n_timestamps, model.timestamp_embedding_size).
            timestamps: A float32 Tensor with shape (`n_sounds, n_timestamps). Centered timestamps in milliseconds corresponding to each embedding in the output.
        """
        x = self.encode(audio)
        ts = get_timestamps(self.cfg, audio, x)
        return x, ts

    def reconstruct(self, lms, mask_ratio, start_frame=0):
        """A helper function to get reconstruction results.
        Use `lms_to_wav` if you may also want to convert the reconstruction results to wavs.
        **Note** this does *not* process the entire LMS frames but rather crops them from the start_frame with the duration of the model's unit frame.
        """
        unit_frames = self.backbone.patch_embed.img_size[1]
        last_frame = start_frame + unit_frames
        lms_cropped = lms[..., start_frame:last_frame]
        with torch.no_grad():
            loss, recons, errormap, mask = self.backbone.forward_viz(lms_cropped, mask_ratio)
        return loss, lms_cropped, recons, errormap, mask

    def decode_to_lms(self, lms_all):
        """Decode the embeddings into LMS.
        Note: To be very strict, we cannot guarantee that the decoder can reconstruct visible patch embeddings to the original LMS space
        because the training does not calculate the loss on the reconstruction result of the visible patches. Since the loss is only calculated on the masked tokens,
        the decoder learns to predict the original input patches of the masked tokens using the visible patch tokens.
        """
        ids_restore = torch.tensor(list(range(lms_all.shape[-2] - 1))).repeat(lms_all.shape[0], 1)
        with torch.no_grad():
            preds = self.backbone.forward_decoder(lms_all, ids_restore)
        decoded = self.backbone.unpatchify(preds)
        return decoded

    def lms_to_wav(self, single_lms, norm_stats, sr=16000, n_fft=400, hop_length=160, win_length=400):
        """A helper function to revert an LMS into an audio waveform.
        CAUTION: Be sure to use the normalization statistics you used to normalize the LMS.
        """
        mu, sigma = norm_stats
        M = (single_lms * sigma + mu).exp().numpy()
        wav = librosa.feature.inverse.mel_to_audio(M, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        return wav


def expand_size(sz):
    if isinstance(sz, int):
        return [sz, sz]
    return sz


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
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
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    gH, gW = grid_size
    grid_h = np.arange(gH, dtype=np.float32)
    grid_w = np.arange(gW, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, gH, gW])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


min = 10000000000.0


def random_structured_mask(shape, mask_ratio, device):
    """Random structured masking for training in audio tasks."""
    B, F, T = shape
    NF = int(F * (mask_ratio + 1.0 / F) * np.random.rand())
    NF = min(F - 1, NF)
    mask_ratio = max(mask_ratio + 0.5 / T - NF / F, 0.0)
    NT = int(T * mask_ratio)
    mask = torch.zeros((B, F, T), dtype=torch.int, device=device)
    for b in range(B):
        mask[b, torch.randperm(F)[:NF]] = 1
    for b in range(B):
        mask[b, :, torch.randperm(T)[:NT]] = 1
    ids_shuffle = torch.argsort(mask.view(B, -1), descending=True)
    len_keep = (mask[0] == 0).sum()
    return ids_shuffle, len_keep


def random_unstructured_mask(shape, mask_ratio, device):
    B, F, T = shape
    L = F * T
    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(B, L, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)
    return ids_shuffle, len_keep


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=24, num_heads=16, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4.0, norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        self.in_chans = in_chans
        img_size, patch_size = expand_size(img_size), expand_size(patch_size)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.img_patch_dim(), bias=True)
        self.norm_pix_loss = norm_pix_loss
        None
        self.initialize_weights()
        self._random_mask_fn = random_unstructured_mask

    def set_random_structured_mask(self):
        None
        self._random_mask_fn = random_structured_mask

    def patch_size(self):
        return self.patch_embed.proj.kernel_size

    def grid_size(self):
        img_size = np.array(self.patch_embed.img_size)
        patch_size = np.array(self.patch_embed.patch_size)
        grid_size = img_size // patch_size
        return grid_size

    def img_patch_dim(self):
        patch_size = self.patch_size()
        return patch_size[0] * patch_size[1] * self.in_chans

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_size(), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.grid_size(), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, L, patch_size[0]*patch_size[0]*in_chans)
        """
        ph, pw = self.patch_size()
        h, w = self.grid_size()
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, ph, w, pw))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, self.img_patch_dim()))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size[0]*patch_size[0]*in_chans)
        imgs: (N, C, H, W)
        """
        ph, pw = self.patch_size()
        h, w = self.grid_size()
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, ph, pw, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * ph, w * pw))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        if isinstance(mask_ratio, (torch.Tensor, np.ndarray)):
            mask = mask_ratio.clone().detach()
            ids_shuffle = torch.argsort(mask.reshape(N, -1), dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            len_keep = (mask[0] == 0).sum() // 2
        elif isinstance(mask_ratio, (list, tuple)):
            ids_restore = mask_ratio[0]
            ids_shuffle = torch.argsort(ids_restore, dim=1)
            len_keep = mask_ratio[1]
        elif mask_ratio == 0:
            mask = torch.zeros([N, L], device=x.device)
            ids_restore = torch.tensor(list(range(L)))
            return x, mask, ids_restore
        else:
            HorF, WorT = self.grid_size()
            ids_shuffle, len_keep = self._random_mask_fn((N, HorF, WorT), mask_ratio, x.device)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, return_layers=False):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        layers = []
        for blk in self.blocks:
            x = blk(x)
            if return_layers:
                layers.append(x)
        x = self.norm(x)
        if return_layers:
            layers.pop()
            layers.append(x)
        if return_layers:
            return torch.stack(layers), mask, ids_restore
        return x, mask, ids_restore

    def drop_cls_token(self, latent):
        return latent[:, 1:, :]

    def get_cls_token(self, latent):
        return latent[:, :1, :]

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, C, H, W]
        pred: [N, L, ph*pw*C]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-06) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    def forward_viz(self, imgs, mask_ratio=0.75):
        loss, pred, mask = self.forward(imgs, mask_ratio)
        pred_org_on_mask = pred.clone()
        visible = mask == 0.0
        pred_org_on_mask[visible] = self.patchify(imgs)[visible]
        recons = self.unpatchify(pred_org_on_mask)
        errormap = ((recons - imgs) ** 2).sqrt()
        return loss, recons, errormap, mask.reshape(mask.shape[0], *self.grid_size())


def ema_model_weight(decay, old_model, new_model):

    def ema(decay, old, new):
        return old * decay + (1 - decay) * new
    for new_params, old_params in zip(new_model.parameters(), old_model.parameters()):
        old_weight, new_weight = old_params.data, new_params.data
        old_params.data = ema(decay, old_weight, new_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class M2DViT(MaskedAutoencoderViT):
    """ Masked Modeling Duo (M2D) implementation based on the MAE.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=24, num_heads=16, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4.0, norm_layer=nn.LayerNorm, norm_pix_loss=False, loss_type='norm_mse', target_layers=None, **kwargs):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads, decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer, norm_pix_loss=norm_pix_loss)
        self.loss_type = loss_type
        self.target_layers = target_layers
        None
        if len(kwargs.keys()) > 0:
            None
        self.target_blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(depth)])
        self.target_norm = norm_layer(embed_dim)
        set_requires_grad(self.target_blocks, False)
        set_requires_grad(self.target_norm, False)
        self.target_blocks.apply(self._init_weights)
        self.target_norm.apply(self._init_weights)
        self.decoder_pred = nn.Linear(decoder_embed_dim, embed_dim, bias=True)
        self.decoder_pred.apply(self._init_weights)

    def update_target_network(self, ema_decay):
        ema_model_weight(ema_decay, self.target_blocks, self.blocks)
        ema_model_weight(ema_decay, self.target_norm, self.norm)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        if isinstance(mask_ratio, (torch.Tensor, np.ndarray)):
            mask = mask_ratio.clone().detach()
            ids_shuffle = torch.argsort(mask.reshape(N, -1), dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            len_keep = (mask[0] == 0).sum() // 2
        elif isinstance(mask_ratio, (list, tuple)):
            ids_restore = mask_ratio[0]
            ids_shuffle = torch.argsort(ids_restore, dim=1)
            len_keep = mask_ratio[1]
        elif mask_ratio == 0:
            mask = torch.zeros([N, L], device=x.device)
            ids_restore = torch.tensor(list(range(L)))
            return x, None, mask, ids_restore
        else:
            HorF, WorT = self.grid_size()
            ids_shuffle, len_keep = self._random_mask_fn((N, HorF, WorT), mask_ratio, x.device)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        ids_keep = ids_shuffle[:, len_keep:]
        x_masked2 = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, x_masked2, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, return_layers=False, blocks=None, norm=None):
        blocks, norm = blocks or self.blocks, norm or self.norm
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, x_targ, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        layers = []
        for blk in blocks:
            x = blk(x)
            if return_layers:
                layers.append(x)
        x = norm(x)
        if return_layers:
            layers.pop()
            layers.append(x)
        if return_layers:
            return torch.stack(layers), x_targ, mask, ids_restore
        return x, x_targ, mask, ids_restore

    def forward_decoder(self, x, ids_restore, keep_cls=False, also_pred_asis=False):
        len_keep = x.shape[1] - 1
        x = self.decoder_embed(x)
        D = x.shape[-1]
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        y = self.drop_cls_token(x)
        y_pred_asis = y
        ids_shuffle = torch.argsort(ids_restore, dim=1)
        y = torch.gather(y, dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, y.shape[-1]))
        y = y[:, len_keep:]
        if keep_cls:
            y = torch.cat([x[:, :1, :], y], dim=1)
        if also_pred_asis:
            return y, y_pred_asis
        return y

    def forward_target_encoder(self, x_targ, drop_cls=True):
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x_targ.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x_targ), dim=1)
        xs = []
        for l, blk in enumerate(self.target_blocks):
            x = blk(x)
            if self.target_layers and l in self.target_layers:
                xs.append(x)
        if xs:
            x = torch.stack(xs).mean(0)
        x = self.target_norm(x)
        if drop_cls:
            x = self.drop_cls_token(x)
        return x

    def forward_loss(self, target, pred, norm_pix_loss, loss_type):
        """
        target: [N, targL, D]
        pred: [N, targL, D]
        """
        if norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-06) ** 0.5
        if loss_type == 'mse':
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)
        elif loss_type == 'norm_mse':
            target = torch.nn.functional.normalize(target, dim=-1, p=2)
            pred = torch.nn.functional.normalize(pred, dim=-1, p=2)
            loss = target * pred
            loss = 2 - 2 * loss.sum(dim=-1)
        else:
            assert loss_type in ['WE NEED A KNOWN LOSS FN']
        loss = loss.mean()
        return loss

    def forward(self, imgs, mask_ratio=0.7):
        latent, x_targ, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        with torch.no_grad():
            target = self.forward_target_encoder(x_targ)
        loss = self.forward_loss(target, pred, self.norm_pix_loss, self.loss_type)
        return loss, pred, (ids_restore, mask)

    def forward_viz(self, imgs, mask_ratio=0.7):
        loss, pred, (ids_restore, mask) = self.forward(imgs, mask_ratio)
        recons, errormap = None, None
        return recons, errormap, mask.reshape(mask.shape[0], *self.grid_size())


class M2D_D2ViT(M2DViT):
    """A data2vec-like M2D variant that feeds all patches to the target network."""

    def random_masking(self, x, mask_ratio):
        """Random masking that returns all patches as the x_target.

        Returns:
            x_masked: Maked patches
            x_target: Target patches = all patches for Data2Vec
            mask: Mask
            ids_restore: indexes for restoration of masked patches
        """
        x_masked, _, mask, ids_restore = super().random_masking(x, mask_ratio)
        return x_masked, x, mask, ids_restore

    def forward(self, imgs, mask_ratio=0.75):
        latent, x_targ, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        with torch.no_grad():
            target = self.forward_target_encoder(x_targ)
            len_keep = latent.shape[1] - 1
            ids_shuffle = torch.argsort(ids_restore, dim=1)
            ids_keep = ids_shuffle[:, len_keep:]
            target = torch.gather(target, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, target.shape[-1]))
        loss = self.forward_loss(target, pred, norm_pix_loss=self.norm_pix_loss, loss_type=self.loss_type)
        return loss, pred, ids_restore


class ASTModel(nn.Module):

    def __init__(self, label_dim=527, fshape=128, tshape=2, fstride=128, tstride=2, input_fdim=128, input_tdim=1024, model_size='base', pretrain_stage=True, load_pretrained_mdl_path=None):
        super(ASTModel, self).__init__()
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        if pretrain_stage == True:
            if load_pretrained_mdl_path != None:
                raise ValueError('Setting load_pretrained_mdl_path at pretraining stage is useless, pretraining is always from scratch, please change it to None.')
            if fstride != fshape or tstride != tshape:
                raise ValueError('fstride != fshape or tstride != tshape, they must be same at the pretraining stage, patch split overlapping is not supported.')
            if model_size == 'tiny':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=False)
                self.heads, self.depth = 3, 12
                self.cls_token_num = 2
            elif model_size == 'small':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=False)
                self.heads, self.depth = 6, 12
                self.cls_token_num = 2
            elif model_size == 'base':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=False)
                self.heads, self.depth = 12, 12
                self.cls_token_num = 2
            elif model_size == 'base_nokd':
                self.v = timm.create_model('vit_deit_base_patch16_384', pretrained=False)
                self.heads, self.depth = 12, 12
                self.cls_token_num = 1
            else:
                raise Exception('Model size must be one of tiny, small, base, base_nokd')
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.softmax = nn.Softmax(dim=-1)
            self.lsoftmax = nn.LogSoftmax(dim=-1)
            self.fshape, self.tshape = fshape, tshape
            self.fstride, self.tstride = fstride, tstride
            self.input_fdim, self.input_tdim = input_fdim, input_tdim
            self.p_input_fdim, self.p_input_tdim = nn.Parameter(torch.tensor(input_fdim), requires_grad=False), nn.Parameter(torch.tensor(input_tdim), requires_grad=False)
            self.cpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, 256))
            self.gpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, 256))
            self.unfold = torch.nn.Unfold(kernel_size=(fshape, tshape), stride=(fstride, tstride))
            self.mask_embed = nn.Parameter(torch.zeros([1, 1, self.original_embedding_dim]))
            self.mask_embed = torch.nn.init.xavier_normal_(self.mask_embed)
            self.p_f_dim, self.p_t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
            num_patches = self.p_f_dim * self.p_t_dim
            self.num_patches = num_patches
            self.v.patch_embed.num_patches = num_patches
            None
            None
            None
            None
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
            self.v.patch_embed.proj = new_proj
            new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + self.cls_token_num, self.original_embedding_dim))
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=0.02)
        elif pretrain_stage == False:
            device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
            if load_pretrained_mdl_path == None:
                raise ValueError('Please set load_pretrained_mdl_path to load a pretrained models.')
            sd = torch.load(load_pretrained_mdl_path, map_location=device)
            try:
                p_fshape, p_tshape = sd['module.v.patch_embed.proj.weight'].shape[2], sd['module.v.patch_embed.proj.weight'].shape[3]
                p_input_fdim, p_input_tdim = sd['module.p_input_fdim'].item(), sd['module.p_input_tdim'].item()
            except:
                raise ValueError('The model loaded is not from a torch.nn.Dataparallel object. Wrap it with torch.nn.Dataparallel and try again.')
            None
            audio_model = ASTModel(fstride=p_fshape, tstride=p_tshape, fshape=p_fshape, tshape=p_tshape, input_fdim=p_input_fdim, input_tdim=p_input_tdim, pretrain_stage=True, model_size=model_size)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.cls_token_num = audio_model.module.cls_token_num
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
            p_f_dim, p_t_dim = audio_model.module.p_f_dim, audio_model.module.p_t_dim
            num_patches = f_dim * t_dim
            p_num_patches = p_f_dim * p_t_dim
            self.v.patch_embed.num_patches = num_patches
            None
            None
            if fshape != p_fshape or tshape != p_tshape:
                raise ValueError('The patch shape of pretraining and fine-tuning is not consistant, pretraining: f={:d}, t={:d}, finetuning: f={:d}, t={:d}'.format(p_fshape, p_tshape, fshape, tshape))
            if fstride != p_fshape or tstride != p_tshape:
                new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
                self.v.patch_embed.proj = new_proj
            new_pos_embed = self.v.pos_embed[:, self.cls_token_num:, :].detach().reshape(1, p_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, p_f_dim, p_t_dim)
            if t_dim < p_t_dim:
                new_pos_embed = new_pos_embed[:, :, :, int(p_t_dim / 2) - int(t_dim / 2):int(p_t_dim / 2) - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(8, t_dim), mode='bilinear')
            if f_dim < p_f_dim:
                new_pos_embed = new_pos_embed[:, :, int(p_f_dim / 2) - int(f_dim / 2):int(p_f_dim / 2) - int(f_dim / 2) + t_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :self.cls_token_num, :].detach(), new_pos_embed], dim=1))

    def get_shape(self, fstride, tstride, input_fdim, input_tdim, fshape, tshape):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def gen_maskid_patch(self, sequence_len=512, mask_size=100, cluster=3):
        mask_id = []
        cur_clus = randrange(cluster) + 3
        while len(list(set(mask_id))) <= mask_size:
            start_id = randrange(sequence_len)
            cur_mask = []
            for i in range(0, cur_clus):
                for j in range(0, cur_clus):
                    mask_cand = start_id + self.p_t_dim * i + j
                    if mask_cand > 0 and mask_cand < sequence_len:
                        cur_mask.append(mask_cand)
            mask_id = mask_id + cur_mask
        mask_id = list(set(mask_id))[:mask_size]
        return torch.tensor(mask_id)

    def gen_maskid_frame(self, sequence_len=512, mask_size=100):
        mask_id = random.sample(range(0, sequence_len), mask_size)
        return torch.tensor(mask_id)

    def finetuningavgtok(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        None
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)
        x = torch.mean(x[:, self.cls_token_num:, :], dim=1)
        x = self.mlp_head(x)
        return x

    def finetuningcls(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)
        if self.cls_token_num == 2:
            x = (x[:, 0] + x[:, 1]) / 2
        else:
            x = x[:, 0]
        x = self.mlp_head(x)
        return x

    def mpc(self, x, mask_patch, cluster, show_mask=False):
        input = self.unfold(x).transpose(1, 2)
        B = x.shape[0]
        x = self.v.patch_embed(x)
        encode_samples = torch.empty((B, mask_patch, 256), device=x.device, requires_grad=False).float()
        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)
        for i in range(B):
            if cluster == True:
                mask_index[i] = self.gen_maskid_patch(self.num_patches, mask_patch)
            else:
                mask_index[i] = self.gen_maskid_frame(self.num_patches, mask_patch)
            encode_samples[i] = input[i, mask_index[i], :].clone().detach()
            mask_dense[i, mask_index[i], :] = 0
        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)
        x = x * mask_dense + (1 - mask_dense) * mask_tokens
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        pred = torch.empty((B, mask_patch, 256), device=x.device).float()
        for i in range(B):
            pred[i] = self.cpredlayer(x[i, mask_index[i] + self.cls_token_num, :])
        nce = torch.tensor(0.0)
        correct = torch.tensor(0.0)
        for i in np.arange(0, B):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            correct += torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, mask_patch, device=x.device)))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        acc = 1.0 * correct / (B * mask_patch)
        nce = nce / (-1.0 * B * mask_patch)
        if show_mask == False:
            return acc, nce
        else:
            if B > 1:
                raise Exception('Currently only support single spectrogram probing test.')
            self.mask_correct = torch.nn.Parameter(torch.arange(0, mask_patch), requires_grad=False)
            pred = input.clone()
            masked = input.clone()
            for i in range(B):
                result = [(float(t) * 99) for t in torch.eq(torch.argmax(self.softmax(total), dim=0), self.mask_correct)]
                pred[i, mask_index[i], :] = torch.tensor(result).reshape(mask_patch, 1).expand(mask_patch, 256)
                masked[i, mask_index[i], :] = 99.0
            fold = torch.nn.Fold(output_size=[self.input_fdim, self.input_tdim], kernel_size=(self.fshape, self.tshape), stride=(self.fstride, self.tstride))
            pred = fold(pred.transpose(1, 2))
            masked = fold(masked.transpose(1, 2))
            return pred, masked

    def mpg(self, input, mask_patch, cluster):
        B = input.shape[0]
        x = self.v.patch_embed(input)
        input = self.unfold(input).transpose(1, 2)
        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)
        for i in range(B):
            if cluster == True:
                mask_index[i] = self.gen_maskid_patch(self.num_patches, mask_patch)
            else:
                mask_index[i] = self.gen_maskid_frame(self.num_patches, mask_patch)
            mask_dense[i, mask_index[i], :] = 0
        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)
        x = x * mask_dense + (1 - mask_dense) * mask_tokens
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        pred = torch.empty((B, mask_patch, self.fshape * self.tshape), device=x.device).float()
        target = torch.empty((B, mask_patch, self.fshape * self.tshape), device=x.device).float()
        for i in range(B):
            pred[i] = self.gpredlayer(x[i, mask_index[i] + self.cls_token_num, :])
            target[i] = input[i, mask_index[i], :]
        mse = torch.mean((pred - target) ** 2)
        return mse

    def forward(self, x, task, cluster=True, mask_patch=400):
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        if task == 'ft_avgtok':
            return self.finetuningavgtok(x)
        elif task == 'ft_cls':
            return self.finetuningcls(x)
        elif task == 'pretrain_mpc':
            return self.mpc(x, mask_patch=mask_patch, cluster=cluster)
        elif task == 'pretrain_mpg':
            return self.mpg(x, mask_patch=mask_patch, cluster=cluster)
        elif task == 'visualize_mask':
            return self.mpc(x, mask_patch=mask_patch, cluster=cluster, show_mask=True)
        else:
            raise Exception('Task unrecognized.')


class SSASTModel(ASTModel):

    def __init__(self, label_dim=527, fshape=128, tshape=2, fstride=128, tstride=2, input_fdim=128, input_tdim=1024, model_size='base', pretrain_stage=True, load_pretrained_mdl_path=None):
        super(SSASTModel, self).__init__(label_dim, fshape, tshape, fstride, tstride, input_fdim, input_tdim, model_size, pretrain_stage, load_pretrained_mdl_path)
        self.feat_mean = nn.AvgPool2d((2, 1), padding=(1, 0))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)
        x = x[:, self.cls_token_num:, :]
        x = self.feat_mean(x)
        return x


class LinearHead(nn.Module):
    """Linear layer
    """

    def __init__(self, dim, num_labels=1000, use_norm=True, affine=False):
        super().__init__()
        self.num_labels = num_labels
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.BatchNorm1d(dim, affine=affine)
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        if self.use_norm:
            x = x.unsqueeze(2)
            x = self.norm(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)


class MedianPool2d(nn.Module):

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = kernel_size
        self.stride = stride
        self.padding = _quadruple(padding)
        self.same = same

    def _padding(self, x):
        if self.same:
            iw = x.shape[-1]
            if iw % self.stride == 0:
                pw = max(self.k - self.stride, 0)
            else:
                pw = max(self.k - iw % self.stride, 0)
            pl = pw // 2
            pr = pw - pl
            padding = pr, pl, 0, 0
        else:
            padding = self.padding
        return padding

    def median(self, x, dim, keepdim=False):
        """
        Find the median along a particular dimension.

        If the dimension length is even take the average of the central values.

        Use *keepdim=True* to preserve the dimension after reduction.
        """
        index = torch.argsort(x, dim)
        deref = [slice(None, None)] * len(x.shape)
        middle = x.shape[dim] // 2
        even = 1 - x.shape[dim] % 2
        deref[dim] = slice(middle - even, middle + 1 + even)
        values = x.gather(dim, index[deref])
        return values.mean(dim, keepdim=keepdim) if even else values if keepdim else values.squeeze(dim)

    def scripy_pad(self, x, padding):
        assert len(x.shape) == 4, 'wrong x shape'
        x = F.pad(x, (1, 1, 0, 0), mode='constant')
        x = F.pad(x, padding, mode='reflect')
        x = torch.cat([x[:, :, :, :padding[0]], x[:, :, :, padding[0] + 1:-padding[1] - 1], x[:, :, :, -padding[1]:]], dim=-1)
        return x

    def forward(self, x):
        with torch.no_grad():
            if len(x.shape) == 3:
                single = True
                x = x.unsqueeze(0)
            else:
                single = False
            x = self.scripy_pad(x, self._padding(x))
            x = x.unfold(3, self.k, self.stride)
            x = self.median(x, dim=-1)
            if single:
                x = x.squeeze(0)
        return x


class SEDMetrics(nn.Module):

    def __init__(self, intersection_thd=0.7):
        super(SEDMetrics, self).__init__()
        self.intersection_thd = intersection_thd
        self.reset_stats()

    def reset_stats(self):
        self.tps = 0
        self.fps = 0
        self.fns = 0
        self.tns = 0

    def compute_truth_table(self, strong_preds, ground_truth):
        with torch.no_grad():
            bsz, num_cls, T = strong_preds.shape
            preds = strong_preds.bool()
            labels = ground_truth.bool()
            idv_event_triu = torch.FloatTensor(T + 1, T).fill_(1).triu().T
            all_events = torch.logical_or(preds, labels).float()
            events_bdry = torch.cat([all_events, torch.FloatTensor(bsz, num_cls, 1).fill_(0)], dim=-1) - torch.cat([torch.FloatTensor(bsz, num_cls, 1).fill_(0), all_events], dim=-1)
            events_start = torch.argwhere(events_bdry == 1)
            events_end = torch.argwhere(events_bdry == -1)
            pred_full_events = strong_preds[events_start[:, 0], events_start[:, 1], :]
            label_full_events = ground_truth[events_start[:, 0], events_start[:, 1], :]
            idv_event_mask = (torch.index_select(idv_event_triu, dim=1, index=events_start[:, -1]) - torch.index_select(idv_event_triu, dim=1, index=events_end[:, -1])).T
            tp_compute = (pred_full_events * idv_event_mask).sum(-1) / ((label_full_events * idv_event_mask).sum(-1) + 1e-07)
            longer_preds = tp_compute >= self.intersection_thd
            shorter_preds = tp_compute < 1 / self.intersection_thd
            tp_full = torch.logical_and(longer_preds, shorter_preds)
            fp_full = torch.logical_xor(longer_preds, tp_full).float()
            fn_full = torch.logical_xor(shorter_preds, tp_full).float()
            tp_full = tp_full.float()
            return tp_full, fp_full, fn_full, events_start

    def compute_tn(self, strong_preds, neg_truths):
        with torch.no_grad():
            bsz, num_cls, T = strong_preds.shape
            idv_event_triu = torch.FloatTensor(T + 1, T).fill_(1).triu().T
            events_bdry = torch.cat([neg_truths, torch.FloatTensor(bsz, num_cls, 1).fill_(0)], dim=-1) - torch.cat([torch.FloatTensor(bsz, num_cls, 1).fill_(0), neg_truths], dim=-1)
            events_start = torch.argwhere(events_bdry == 1)
            events_end = torch.argwhere(events_bdry == -1)
            pred_full_events = strong_preds[events_start[:, 0], events_start[:, 1], :]
            idv_event_mask = (torch.index_select(idv_event_triu, dim=1, index=events_start[:, -1]) - torch.index_select(idv_event_triu, dim=1, index=events_end[:, -1])).T
            tn_compute = (pred_full_events * idv_event_mask).sum(-1) / idv_event_mask.sum(-1)
            tn_full = (tn_compute == 1).float()
            return tn_full, events_start

    def compute_avg_f1(self, strong_preds, ground_truths):
        bsz, _, _ = strong_preds.shape
        event_eye = torch.eye(bsz, device=strong_preds.device)
        tps, _, _, events_index = self.compute_truth_table(strong_preds, ground_truths)
        event_to_clip = torch.index_select(event_eye, dim=0, index=events_index[:, 0])
        tp_clip = tps.unsqueeze(0).matmul(event_to_clip)
        tp_fn_fp_clip = event_to_clip.sum(0)
        f_score = tp_clip / (1 / 2 * tp_clip + 1 / 2 * tp_fn_fp_clip)
        f_score = f_score.nan_to_num(0)
        return f_score.mean()

    def accm_macro_f1(self, strong_preds, ground_truths):
        _, num_cls, _ = strong_preds.shape
        cls_eye = torch.eye(num_cls, device=strong_preds.device)
        tp_full, fp_full, fn_full, events_index = self.compute_truth_table(strong_preds, ground_truths)
        cls_one_hot = torch.index_select(cls_eye, dim=0, index=events_index[:, 1])
        cls_tp = tp_full.unsqueeze(0).matmul(cls_one_hot)
        cls_fp = fp_full.unsqueeze(0).matmul(cls_one_hot)
        cls_fn = fn_full.unsqueeze(0).matmul(cls_one_hot)
        self.tps += cls_tp
        self.fps += cls_fp
        self.fns += cls_fn

    def compute_macro_f1(self):
        false_num = self.fps + self.fns
        if false_num is 0:
            false_num += torch.FloatTensor(1).fill_(1e-07)
        f_score = self.tps / (self.tps + 1 / 2 * false_num)
        f_score = f_score.nan_to_num(0)
        self.reset_stats()
        return f_score.mean()

    def accm_auc(self, strong_preds, pos_truths, neg_truths):
        num_thds, _, num_cls, _ = strong_preds.shape
        self.tps += torch.FloatTensor(num_thds, num_cls).fill_(0)
        self.fps += torch.FloatTensor(num_thds, num_cls).fill_(0)
        self.fns += torch.FloatTensor(num_thds, num_cls).fill_(0)
        self.tns += torch.FloatTensor(num_thds, num_cls).fill_(0)
        cls_eye = torch.eye(num_cls, device=strong_preds.device)
        for i, strong_preds_thd in enumerate(strong_preds):
            tp_full, fp_full, fn_full, events_index = self.compute_truth_table(strong_preds_thd, pos_truths)
            tn_full, neg_index = self.compute_tn(1 - strong_preds_thd, neg_truths)
            cls_one_hot = torch.index_select(cls_eye, dim=0, index=events_index[:, 1])
            neg_cls_one_hot = torch.index_select(cls_eye, dim=0, index=neg_index[:, 1])
            cls_tp = tp_full.unsqueeze(0).matmul(cls_one_hot).squeeze(0)
            cls_fp = fp_full.unsqueeze(0).matmul(cls_one_hot).squeeze(0)
            cls_fn = fn_full.unsqueeze(0).matmul(cls_one_hot).squeeze(0)
            cls_tn = tn_full.unsqueeze(0).matmul(neg_cls_one_hot).squeeze(0)
            self.tps[i] += cls_tp
            self.fps[i] += cls_fp
            self.fns[i] += cls_fn
            self.tns[i] += cls_tn

    def compute_auc(self):
        tpr = self.tps / (self.tps + self.fps)
        fpr = self.fps / (self.fps + self.tns)
        cls_auc = []
        for i in range(tpr.shape[1]):
            fpr_np = fpr[:, i].cpu().numpy()
            tpr_np = tpr[:, i].cpu().numpy()
            cls_auc.append(auc(fpr_np[-1::-1], tpr_np[-1::-1]))
        auc_score = sum(cls_auc) / len(cls_auc)
        self.cls_auc = []
        self.reset_stats()
        return auc_score

    def compute_d_prime(self, auc):
        standard_normal = stats.norm()
        d_prime = standard_normal.ppf(auc) * math.sqrt(2.0)
        return d_prime


def get_cls_avg(output_i, cur_len, use_cls):
    if use_cls:
        length_mask = torch.arange(output_i[0].shape[1] - 1) < cur_len.unsqueeze(1)
    else:
        length_mask = torch.arange(output_i[0].shape[1]) < cur_len.unsqueeze(1)
    if use_cls:
        cls = [x[:, 0] for x in output_i]
        avg = [(torch.sum(x[:, 1:] * length_mask.unsqueeze(-1), dim=1) / (cur_len.unsqueeze(1) + 1e-06)) for x in output_i]
    else:
        cls = [torch.zeros_like(x[:, 0]) for x in output_i]
        avg = [(torch.sum(x * length_mask.unsqueeze(-1), dim=1) / (cur_len.unsqueeze(1) + 1e-06)) for x in output_i]
    return cls, avg


def AST_base(patch_h=64, patch_w=4, **kwargs):
    return AST(patch_h=patch_h, patch_w=patch_w, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, norm_layer=partial(nn.LayerNorm, eps=1e-06), **kwargs)


class ClipModel(nn.Module):

    def __init__(self, num_labels: 'int'):
        super().__init__()
        self.encoder = AST_base()
        self.head = LinearHead(self.encoder.embed_dim * 2, num_labels, use_norm=True, affine=False)

    def forward(self, batch):
        (x, length), y = batch
        x = self.encoder.get_intermediate_layers_chunks(x, length, 1, 601, avgpool=True)
        x = self.head(x)
        return x, y


class Distill(nn.Module):

    def __init__(self, ncls=0, project=False, nclasses=527):
        super().__init__()
        self.teacher = ClipModel(nclasses)
        self.student = FrameAST_base(nprompt=ncls)
        self.project = project
        if project:
            self.projector = build_mlp(2, 768, 4096, 768, last_bn=False)
            self.project_linear = LinearHead(768, 527, use_norm=True, affine=False)
        self.linear = LinearHead(768, nclasses, use_norm=True, affine=False)

    def forward(self, batch):
        target, _ = self.teacher(batch)
        (mel, length), y = batch
        chunk_len = 1001
        total_len = mel.shape[-1]
        num_chunks = total_len // chunk_len + 1
        output = []
        chunk_mark = []
        None
        for i in range(num_chunks):
            cur_len = torch.clip(length - i * chunk_len, 0, chunk_len)
            if i == 0:
                chunk_mark_ = cur_len > 0
            else:
                chunk_mark_ = cur_len > chunk_len // 2
            start = i * chunk_len
            end = (i + 1) * chunk_len
            if end > total_len:
                end = total_len
            if end > start + 20:
                None
                mel_chunk = mel[:, :, :, start:end]
                output_chunk = self.student.get_intermediate_layers(mel_chunk, cur_len, n=1, scene=True)
                output.append(output_chunk)
                chunk_mark.append(chunk_mark_)
        chunk_mark = torch.stack(chunk_mark, dim=0).unsqueeze(-1)
        output = torch.stack(output, dim=0)
        pred = torch.sum(chunk_mark * output, dim=0) / torch.sum(chunk_mark, dim=0)
        return pred, target, y


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


class Byol(nn.Module):
    """
    Build a Byol model with: a query encoder, a key encoder, and a queue
    """

    def __init__(self, base_encoder, num_classes=256, K=65536, m=0.999, T=0.07):
        """
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(Byol, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.encoder_q = base_encoder()
        self.encoder_k = base_encoder()
        dim = self.encoder_q.embed_dim
        self.head_q = self._build_mlp(2, dim, 4096, num_classes, last_bn=False)
        self.head_k = self._build_mlp(2, dim, 4096, num_classes, last_bn=False)
        self.predictor_q = self._build_mlp(2, num_classes, 4096, num_classes, last_bn=False)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim
            mlp.append(nn.Linear(dim1, dim2, bias=False))
            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                mlp.append(nn.BatchNorm1d(dim2, affine=False))
        return nn.Sequential(*mlp)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this
        idx_shuffle = torch.randperm(batch_size_all)
        torch.distributed.broadcast(idx_shuffle, src=0)
        idx_unshuffle = torch.argsort(idx_shuffle)
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this]

    def forward(self, im_q, len_q, mask_q, im_k, len_k, mask_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        q, loss_mse_q = self.encoder_q(im_q, mask_q, len_q)
        q = self.head_q(q)
        q = self.predictor_q(q)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k, _ = self.encoder_k(im_k, mask_k, len_k)
            k = self.head_k(k)
            k = nn.functional.normalize(k, dim=1)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        loss_byol = byol_loss_func(q, k)
        return loss_byol, loss_mse


class Encoder(nn.Module):
    """ Vision Transformer """

    def __init__(self, spec_h=64, spec_w=1001, patch_w=16, patch_h=16, in_chans=1, num_classes=0, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, mask_ratio=0.5, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.spec_w = spec_w
        self.spec_h = spec_h
        self.embed_dim = embed_dim
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.patch_embed = PatchEmbed_v2(patch_h, patch_w, embed_dim)
        num_patches = get_num_patches(spec_h, spec_w, patch_h, patch_w)
        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.mask_ratio = mask_ratio
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
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

    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == self.spec_w and h == self.spec_h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_width
        h0 = h // self.patch_embed.patch_height
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed.reshape(1, self.spec_h // self.patch_h, self.spec_w // self.patch_w, dim).permute(0, 3, 1, 2), scale_factor=(h0 / (self.spec_h // self.patch_h), w0 / (self.spec_w // self.patch_w)), mode='bicubic')
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x, mask=True):
        B, nc, h, w = x.shape
        mel_patches, x = self.patch_embed(x)
        B, T, C = x.shape
        mask_index = None
        if mask:
            mask_index = random_mask.get_mask_v2(B, T, self.mask_ratio)
            mask_index_expand = mask_index.unsqueeze(2).expand(B, T, self.embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos = self.interpolate_pos_encoding(x, h, w)
        x = x + pos
        return self.pos_drop(x), pos, mel_patches, mask_index, h, w

    def forward(self, x):
        x, pos, mel_patches, mask_index, h, w = self.prepare_tokens(x)
        x_cls, x_seq = x[:, 0:1], x[:, 1:]
        B, T, C = x_seq.shape
        x = torch.cat((x_cls, x_seq[~mask_index].reshape(B, -1, C)), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mel_patches, mask_index, h, w

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x, _, _, _, _, _ = self.prepare_tokens(x, mask=False)
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


class Decoder(nn.Module):

    def __init__(self, patch_w=16, patch_h=16, embed_dim=384, depth=6, num_heads=6, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = patch_w * patch_h
        self.num_features = self.embed_dim = embed_dim
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.embed_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)
        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:]))
        else:
            x = self.head(self.norm(x))
        return x


def encoder_small(patch_h=16, patch_w=16, **kwargs):
    return Encoder(patch_h=patch_h, patch_w=patch_w, embed_dim=384, depth=12, num_heads=6, **kwargs)


class MaskedAutoEncoder(nn.Module):

    def __init__(self, patch_h=16, patch_w=16, **kwargs):
        super().__init__()
        self.encoder = encoder_small(patch_h=patch_h, patch_w=patch_w, **kwargs)
        self.decoder = Decoder(patch_h=patch_h, patch_w=patch_w)
        self.middle = nn.Linear(self.encoder.embed_dim, self.decoder.embed_dim)
        self.mask_embed = nn.Parameter(torch.zeros(1, 1, self.decoder.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.encoder.num_patches + 1, self.decoder.embed_dim))
        trunc_normal_(self.mask_embed, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)

    def interpolate_pos_encoding(self, npatch, h, w):
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == self.encoder.spec_w and h == self.encoder.spec_h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = self.decoder.embed_dim
        w0 = w // self.encoder.patch_embed.patch_width
        h0 = h // self.encoder.patch_embed.patch_height
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed.reshape(1, self.encoder.spec_h // self.encoder.patch_h, self.encoder.spec_w // self.encoder.patch_w, dim).permute(0, 3, 1, 2), scale_factor=(h0 / (self.encoder.spec_h // self.encoder.patch_h), w0 / (self.encoder.spec_w // self.encoder.patch_w)), mode='bicubic')
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x, run_decoder=True):
        enc_x, mel_patches, mask_index, h, w = self.encoder(x)
        if run_decoder:
            x = self.middle(enc_x)
            B, npatch, _ = mel_patches.shape
            pos = self.interpolate_pos_encoding(npatch, h, w)
            pos = pos.expand(B, -1, -1)
            C = pos.shape[-1]
            pos_cls = pos[:, 0:1]
            pos_seq = pos[:, 1:]
            x += torch.cat((pos_cls, pos_seq[~mask_index].reshape(B, -1, C)), dim=1)
            x_mask = pos_seq[mask_index].reshape(B, -1, C) + self.mask_embed
            num_mask = x_mask.shape[1]
            x = torch.cat([x, x_mask], dim=1)
            x = self.decoder(x, 0)
            mel_mask = mel_patches[mask_index]
            x_mask = x[:, -num_mask:].reshape(B * num_mask, -1)
            mse_loss = F.mse_loss(mel_mask, x_mask)
            return enc_x[:, 0, :], mse_loss
        else:
            return enc_x[:, 0, :], torch.zeros([])


def AST_small(patch_h=64, patch_w=4, **kwargs):
    return AST(patch_h=patch_h, patch_w=patch_w, embed_dim=384, depth=12, num_heads=6, qkv_bias=False, norm_layer=partial(nn.LayerNorm, eps=1e-06), **kwargs)


def build_expander(num_layers, input_dim, mlp_dim, output_dim):
    mlp = []
    for l in range(num_layers):
        dim1 = input_dim if l == 0 else mlp_dim
        dim2 = output_dim if l == num_layers - 1 else mlp_dim
        mlp.append(nn.Linear(dim1, dim2, bias=False))
        if l < num_layers - 1:
            mlp.append(nn.BatchNorm1d(dim2))
            mlp.append(nn.ReLU(inplace=True))
    return nn.Sequential(*mlp)


def variance_loss(z1: 'torch.Tensor') ->torch.Tensor:
    """Computes variance loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.
    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
    Returns:
        torch.Tensor: variance regularization loss.
    """
    eps = 0.0001
    std_z1 = compute_var(z1)
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(1 - std_z1))
    return std_loss, std_z1.mean()


class DUAL(nn.Module):

    def __init__(self, arch):
        super().__init__()
        if arch == 'small':
            self.patchnet = AST_small(use_cls=False, patch_h=16, patch_w=16)
            self.framenet = AST_small(use_cls=False, patch_h=64, patch_w=4)
        elif arch == 'base':
            self.patchnet = AST_base(use_cls=False, patch_h=16, patch_w=16)
            self.framenet = AST_base(use_cls=False, patch_h=64, patch_w=4)
        self.patch_expander = build_expander(3, self.patchnet.embed_dim, 8192, 64 * 4)
        self.frame_expander = build_expander(3, self.framenet.embed_dim, 8192, 64 * 4)

    def forward(self, x_frame, x_patch, mask_frame, mask_patch):
        patch_x, patch_mel = self.patchnet(x_patch, mask_patch)
        frame_x, frame_mel = self.framenet(x_frame, mask_frame)
        mask = mask_patch | mask_frame
        patch_x = patch_x[mask]
        frame_x = frame_x[mask]
        patch_mel = patch_mel[mask]
        frame_mel = frame_mel[mask]
        T, C = patch_x.shape
        if 1:
            patch_x_big = patch_x.reshape(T // 4, 4, C)
            patch_x_big = torch.mean(patch_x_big, dim=1)
            frame_x_big = frame_x.reshape(T // 4, 4, C)
            frame_x_big = torch.mean(frame_x_big, dim=1)
        patch_x = self.patch_expander(patch_x)
        frame_x = self.frame_expander(frame_x)
        loss_mel_patch = mse_loss(patch_x, patch_mel)
        loss_mel_frame = mse_loss(frame_x, frame_mel)
        loss_dual = mse_loss(frame_x_big, patch_x_big)
        loss_uniform_patch, std_patch = variance_loss(patch_x_big.reshape(-1, patch_x_big.shape[-1]))
        loss_uniform_frame, std_frame = variance_loss(frame_x_big.reshape(-1, patch_x_big.shape[-1]))
        return loss_mel_patch, loss_mel_frame, loss_dual, loss_uniform_patch, loss_uniform_frame, std_patch, std_frame


class RandomResizeCrop(nn.Module):
    """Random Resize Crop block.

    Args:
        virtual_crop_scale: Virtual crop area `(F ratio, T ratio)` in ratio to input size.
        freq_scale: Random frequency range `(min, max)`.
        time_scale: Random time frame range `(min, max)`.
    """

    def __init__(self, virtual_crop_scale=(1.0, 1.5), freq_scale=(0.6, 1.5), time_scale=(0.6, 1.5)):
        super().__init__()
        self.virtual_crop_scale = virtual_crop_scale
        self.freq_scale = freq_scale
        self.time_scale = time_scale
        self.interpolation = 'bicubic'
        assert time_scale[1] >= 1.0 and freq_scale[1] >= 1.0

    @staticmethod
    def get_params(virtual_crop_size, in_size, time_scale, freq_scale):
        canvas_h, canvas_w = virtual_crop_size
        src_h, src_w = in_size
        h = np.clip(int(np.random.uniform(*freq_scale) * src_h), 1, canvas_h)
        w = np.clip(int(np.random.uniform(*time_scale) * src_w), 1, canvas_w)
        i = random.randint(0, canvas_h - h) if canvas_h > h else 0
        j = random.randint(0, canvas_w - w) if canvas_w > w else 0
        return i, j, h, w

    def forward(self, lms):
        virtual_crop_size = [int(s * c) for s, c in zip(lms.shape[-2:], self.virtual_crop_scale)]
        virtual_crop_area = torch.zeros((lms.shape[0], virtual_crop_size[0], virtual_crop_size[1])).to(torch.float)
        _, lh, lw = virtual_crop_area.shape
        c, h, w = lms.shape
        x, y = (lw - w) // 2, (lh - h) // 2
        virtual_crop_area[:, y:y + h, x:x + w] = lms
        i, j, h, w = self.get_params(virtual_crop_area.shape[-2:], lms.shape[-2:], self.time_scale, self.freq_scale)
        crop = virtual_crop_area[:, i:i + h, j:j + w]
        lms = F.interpolate(crop.unsqueeze(0), size=lms.shape[-2:], mode=self.interpolation, align_corners=True).squeeze(0)
        return lms

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(virtual_crop_size={self.virtual_crop_scale}'
        format_string += ', time_scale={0}'.format(tuple(round(s, 4) for s in self.time_scale))
        format_string += ', freq_scale={0})'.format(tuple(round(r, 4) for r in self.freq_scale))
        return format_string


def energy_scale(xa, xb):
    xa = xa.exp()
    xb = xb.exp()
    return torch.sum(xa) / torch.sum(xb)


def log_mixup_exp(xa, xb, alpha):
    xa = xa.exp()
    xb = xb.exp()
    len_xa = xa.shape[2]
    len_xb = xb.shape[2]
    if len_xa < len_xb:
        start = np.random.randint(0, len_xb - len_xa)
        scale = energy_scale(xa, xb[:, :, start:start + len_xa])
        scale = 1.0
        xa = alpha * xa + (1.0 - alpha) * scale * xb[:, :, start:start + len_xa]
        return torch.log(xa + torch.finfo(xa.dtype).eps)
    elif len_xa > len_xb:
        start = np.random.randint(0, len_xa - len_xb)
        scale = energy_scale(xa[:, :, start:start + len_xb], xb)
        scale = 1.0
        xa[:, :, start:start + len_xb] = alpha * xa[:, :, start:start + len_xb] + (1.0 - alpha) * scale * xb
        return torch.log(xa + torch.finfo(xa.dtype).eps)
    else:
        scale = energy_scale(xa, xb)
        scale = 1.0
        x = alpha * xa + (1.0 - alpha) * scale * xb
        return torch.log(x + torch.finfo(x.dtype).eps)


class Mixup(nn.Module):
    """Mixup.

    Args:
        ratio: Alpha in the paper.
        n_memory: Size of memory bank FIFO.
        log_mixup_exp: Use log-mixup-exp to mix if this is True, or mix without notion of log-scale.
    """

    def __init__(self, ratio=0.4, n_memory=2000, log_mixup_exp=True):
        super().__init__()
        self.ratio = ratio
        self.n = n_memory
        self.log_mixup_exp = log_mixup_exp
        self.memory_bank = []

    def forward(self, x):
        alpha = self.ratio * np.random.random()
        if self.memory_bank:
            z = self.memory_bank[np.random.randint(len(self.memory_bank))]
            mixed = log_mixup_exp(x, z, 1.0 - alpha) if self.log_mixup_exp else alpha * z + (1.0 - alpha) * x
        else:
            mixed = x
        self.memory_bank = (self.memory_bank + [x])[-self.n:]
        return mixed

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(ratio={self.ratio},n={self.n}'
        format_string += f',log_mixup_exp={self.log_mixup_exp})'
        return format_string


class MAE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward():
        pass


class ATST(nn.Module):

    def __init__(self, arch='small', ncrops=2, **kwargs):
        super().__init__()
        if arch == 'small':
            encoder_fn = AST_small
            embed_dim = 384
        elif arch == 'base':
            encoder_fn = AST_base
            embed_dim = 768
        else:
            raise RuntimeError('arch {} is not implemented'.format(arch))
        self.student = MultiCropWrapper(encoder_fn(**kwargs), embed_dim, predictor=True)
        self.teacher = MultiCropWrapper(encoder_fn(**kwargs), embed_dim, predictor=False)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.load_state_dict({k: v for k, v in self.student.state_dict().items() if 'predictor' not in k})
        self.loss_fn = ByolLoss(ncrops)

    def forward(self, melspecs, lengths):
        teacher_output = self.teacher(melspecs[:2], lengths[:2])
        student_output = self.student(melspecs, lengths)
        loss = self.loss_fn(student_output, teacher_output)
        return loss

    def update_teacher(self, m):
        with torch.no_grad():
            for param_q, param_k in zip(self.student.encoder.parameters(), self.teacher.encoder.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            for param_q, param_k in zip(self.student.projector.parameters(), self.teacher.projector.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


class AttentionHead(nn.Module):

    def __init__(self, dim, att_dim, num_heads, num_labels):
        super().__init__()
        self.pre_linear = nn.Linear(dim, att_dim)
        self.att = Block(att_dim, num_heads)
        self.norm = nn.BatchNorm1d(att_dim, affine=False)
        self.linear = nn.Linear(att_dim, num_labels)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, att_dim))
        trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.pre_linear(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.att(x)[:, 0]
        x = x.unsqueeze(2)
        x = self.norm(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AttentionHead,
     lambda: ([], {'dim': 4, 'att_dim': 4, 'num_heads': 4, 'num_labels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (Block,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (ConvPosEmbed,
     lambda: ([], {'args': SimpleNamespace(encoder_embed_dim=4, conv_pos=4, conv_pos_groups=1)}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (Decoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 384]), 0], {})),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GLU_Linear,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 0, 4])], {})),
    (LinearHead,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (MedianPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Mixup,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MultiheadAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (NormEMAVectorQuantizer,
     lambda: ([], {'n_embed': 4, 'embedding_dim': 4, 'beta': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PatchEmbed3D_new,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64, 64])], {})),
    (PrecomputedNorm,
     lambda: ([], {'stats': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RandomResizeCrop,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (SamePad,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SinusoidalPositionalEncoding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)], {})),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

