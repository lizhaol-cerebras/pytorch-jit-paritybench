
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


from collections import defaultdict


from collections import deque


import torch


import collections


import collections.abc


import copy


import warnings


from collections import OrderedDict


from typing import Any


from typing import Callable


from typing import Dict


from typing import List


from typing import Optional


from typing import Union


import logging


from torch.utils.data import Dataset


from torch.utils.data.dataset import Dataset


import uuid


from typing import Iterable


import pandas as pd


from torchvision.datasets.video_utils import VideoClips


from torchvision.io import read_video


import numpy as np


import torch.nn.functional as F


import torchvision


from torch import nn


from torch import Tensor


from torchvision import transforms


from typing import Type


import random


import functools


import types


from torch.utils.data import ConcatDataset


import torchvision.datasets.folder as tv_helpers


import math


from torch.utils.data import DataLoader


import typing


from copy import deepcopy


from typing import Iterator


from torch.utils.data.dataloader import DataLoader


from torch.utils.data.dataloader import Sampler


from typing import Tuple


import torchvision.transforms as T


import torchvision.transforms.functional as F


from torchvision.transforms import Compose


from torchvision.transforms import Normalize


from torchvision.transforms import Resize


from torchvision.transforms import ToTensor


import re


from collections import Counter


from torchvision import transforms as img_transforms


from torch.utils.data.dataset import Subset


from functools import partial


from itertools import repeat


import torch.nn as nn


from torch.nn import functional as F


from torch.nn import CrossEntropyLoss


from torch.nn import SmoothL1Loss


from abc import ABC


from abc import abstractmethod


from typing import NamedTuple


from torchvision.models._utils import IntermediateLayerGetter


from torchvision.ops.misc import FrozenBatchNorm2d


from scipy.optimize import linear_sum_assignment


from collections import namedtuple


from collections.abc import MutableMapping


from torchvision.models.resnet import Bottleneck


from torchvision.models.resnet import conv1x1


from torchvision.models.resnet import conv3x3


from torch.nn.utils.weight_norm import weight_norm


from functools import lru_cache


from enum import Enum


from torch.nn.utils.rnn import pack_padded_sequence


from sklearn.metrics import average_precision_score


from sklearn.metrics import f1_score


from sklearn.metrics import precision_recall_curve


from sklearn.metrics import precision_recall_fscore_support


from sklearn.metrics import roc_auc_score


from torch.optim.lr_scheduler import LambdaLR


from torchvision.ops.boxes import box_area


from itertools import chain


from torch import distributed as dist


import matplotlib as mpl


import matplotlib.colors as mplc


import matplotlib.figure as mplfigure


from matplotlib.backends.backend_agg import FigureCanvasAgg


import time


from functools import wraps


import itertools


import torchvision.models as models


import torchvision.transforms as transforms


from torch.autograd import Variable


from abc import ABCMeta


from torch.nn.modules.batchnorm import BatchNorm2d


from torchvision.ops import RoIPool


from torchvision.ops.boxes import batched_nms


from torchvision.ops.boxes import nms


def box_cxcywh_to_xyxy(x: 'Tensor'):
    x_c, y_c, w, h = x.unbind(-1)
    b = [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]
    return torch.stack(b, dim=-1)


class PostProcess(nn.Module):

    @torch.no_grad()
    def forward(self, outputs: 'Dict[str, Tensor]', target_sizes: 'Tensor'):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)
        boxes = box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        if 'attr_logits' in outputs:
            assert len(outputs['attr_logits']) == len(results)
            attr_scores, attr_labels = outputs['attr_logits'].max(-1)
            for idx, r in enumerate(results):
                r['attr_scores'] = attr_scores[idx]
                r['attr_labels'] = attr_labels[idx]
        return results


def drop_block_2d(x, drop_prob: 'float'=0.1, block_size: 'int'=7, gamma_scale: 'float'=1.0, with_noise: 'bool'=False, inplace: 'bool'=False, batchwise: 'bool'=False):
    """DropBlock. See https://arxiv.org/pdf/1810.12890.pdf
    DropBlock with an experimental gaussian noise option. This layer has been tested on
    a few training runs with success, but needs further validation and possibly
    optimization for lower runtime impact.
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
    """DropBlock. See https://arxiv.org/pdf/1810.12890.pdf
    DropBlock with an experimental gaussian noise option. Simplied from above without
    concern for valid block mask at edges.
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
    """DropBlock. See https://arxiv.org/pdf/1810.12890.pdf"""

    def __init__(self, drop_prob=0.1, block_size=7, gamma_scale=1.0, with_noise=False, inplace=False, batchwise=False, fast=True):
        super().__init__()
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


def drop_path(x, drop_prob: 'float'=0.0, training: 'bool'=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual
    blocks). This is the same as the DropConnect impl I created for EfficientNet, etc
    networks, however, the original name is misleading as 'Drop Connect' is a
    different form of dropout in a separate paper... See discussion:
    https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    I've opted for changing the layer and argument names to 'drop path' rather than mix
    DropConnect as a layer name and use 'survival rate' as the argument.
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
    """Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

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


class Attention(nn.Module):
    """Attention Layer as used in Vision Transformer."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, register_hook=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def get_chunks(x, sizes):
    out = []
    begin = 0
    for s in sizes:
        y = x.narrow(1, begin, s)
        out.append(y)
        begin += s
    return out


def get_sizes_list(dim, chunks):
    split_size = (dim + chunks - 1) // chunks
    sizes_list = [split_size] * chunks
    sizes_list[-1] = sizes_list[-1] - (sum(sizes_list) - dim)
    assert sum(sizes_list) == dim
    if sizes_list[-1] < 0:
        n_miss = sizes_list[-2] - sizes_list[-1]
        sizes_list[-1] = sizes_list[-2]
        for j in range(n_miss):
            sizes_list[-j - 1] -= 1
        assert sum(sizes_list) == dim
        assert min(sizes_list) > 0
    return sizes_list


def log_class_usage(component_type, klass):
    """This function is used to log the usage of different MMF components."""
    identifier = 'MMF'
    if klass and hasattr(klass, '__name__'):
        identifier += f'.{component_type}.{klass.__name__}'
    torch._C._log_api_usage_once(identifier)


class Block(nn.Module):

    def __init__(self, input_dims, output_dim, mm_dim=1600, chunks=20, rank=15, shared=False, dropout_input=0.0, dropout_pre_lin=0.0, dropout_output=0.0, pos_norm='before_cat'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.chunks = chunks
        self.rank = rank
        self.shared = shared
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        assert pos_norm in ['before_cat', 'after_cat']
        self.pos_norm = pos_norm
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        if shared:
            self.linear1 = self.linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim)
        merge_linears0, merge_linears1 = [], []
        self.sizes_list = get_sizes_list(mm_dim, chunks)
        for size in self.sizes_list:
            ml0 = nn.Linear(size, size * rank)
            merge_linears0.append(ml0)
            if self.shared:
                ml1 = ml0
            else:
                ml1 = nn.Linear(size, size * rank)
            merge_linears1.append(ml1)
        self.merge_linears0 = nn.ModuleList(merge_linears0)
        self.merge_linears1 = nn.ModuleList(merge_linears1)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log_class_usage('Fusion', self.__class__)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        bsize = x1.size(0)
        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        x0_chunks = get_chunks(x0, self.sizes_list)
        x1_chunks = get_chunks(x1, self.sizes_list)
        zs = []
        for chunk_id, m0, m1 in zip(range(len(self.sizes_list)), self.merge_linears0, self.merge_linears1):
            x0_c = x0_chunks[chunk_id]
            x1_c = x1_chunks[chunk_id]
            m = m0(x0_c) * m1(x1_c)
            m = m.view(bsize, self.rank, -1)
            z = torch.sum(m, 1)
            if self.pos_norm == 'before_cat':
                z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
                z = F.normalize(z, p=2)
            zs.append(z)
        z = torch.cat(zs, 1)
        if self.pos_norm == 'after_cat':
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)
        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        z = self.linear_out(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


def _no_grad_trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """
    # type: (Tensor, float, float, float, float) -> Tensor

    Fills the input Tensor with values drawn from a truncated
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


class VisionTransformer(nn.Module):
    """Vision Transformer
    A PyTorch impl of :
        `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
         https://arxiv.org/abs/2010.11929
    """

    def __init__(self, config: 'omegaconf.DictConfig'):
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
            representation_size (Optional[int]): enable and set representation layer
                                                 (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.img_size = config.get('img_size', 224)
        self.patch_size = config.get('patch_size', 16)
        self.in_chans = config.get('in_chans', 3)
        self.num_classes = config.get('num_classes', 1000)
        self.embed_dim = config.get('embed_dim', 768)
        self.depth = config.get('depth', 12)
        self.num_heads = config.get('num_heads', 12)
        self.mlp_ratio = config.get('mlp_ratio', 4.0)
        self.qkv_bias = config.get('qkv_bias', True)
        self.qk_scale = config.get('qk_scale', None)
        self.representation_size = config.get('representation_size', None)
        self.drop_rate = config.get('drop_rate', 0.0)
        self.attn_drop_rate = config.get('attn_drop_rate', 0.0)
        self.drop_path_rate = config.get('drop_path_rate', 0.0)
        self.norm_layer = config.get('norm_layer', None)
        self.num_features = self.embed_dim
        norm_layer = self.norm_layer or partial(nn.LayerNorm, eps=1e-06)
        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=self.drop_rate)
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = nn.ModuleList([Block(dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale, drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer) for i in range(self.depth)])
        self.norm = norm_layer(self.embed_dim)
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

    def forward(self, images: 'torch.Tensor', register_blk=-1):
        B = images.shape[0]
        x = self.patch_embed(images)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x, register_blk == i)
        x = self.norm(x)
        return x


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: 'NestedTensor'):
        x = tensor_list.tensors
        mask = tensor_list.mask
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
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
        return pos


class _NewEmptyTensorOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class Conv2d(torch.nn.Conv2d):

    def __init__(self, *args, **kwargs):
        norm = kwargs.pop('norm', None)
        activation = kwargs.pop('activation', None)
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if x.numel() == 0 and self.training:
            assert not isinstance(self.norm, torch.nn.SyncBatchNorm)
        if x.numel() == 0:
            assert not isinstance(self.norm, torch.nn.GroupNorm)
            output_shape = [((i + 2 * p - (di * (k - 1) + 1)) // s + 1) for i, p, di, k, s in zip(x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride)]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            empty = _NewEmptyTensorOp.apply(x, output_shape)
            if self.training:
                _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + _dummy
            else:
                return empty
        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def get_norm(norm, out_channels):
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {'BN': BatchNorm2d, 'GN': lambda channels: nn.GroupNorm(32, channels), 'nnSyncBN': nn.SyncBatchNorm, '': lambda x: x}[norm]
    return norm(out_channels)


class BasicStem(nn.Module):

    def __init__(self, in_channels=3, out_channels=64, norm='BN', caffe_maxpool=False):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False, norm=get_norm(norm, out_channels))
        self.caffe_maxpool = caffe_maxpool

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        if self.caffe_maxpool:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0, ceil_mode=True)
        else:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

    @property
    def out_channels(self):
        return self.conv1.out_channels

    @property
    def stride(self):
        return 4


CONFIG_NAME = 'config.yaml'


def _decode_value(value):
    if not isinstance(value, str):
        return value
    if value == 'None':
        value = None
    try:
        value = literal_eval(value)
    except ValueError:
        pass
    except SyntaxError:
        pass
    return value


logger = logging.getLogger(__name__)


def _merge_with_dotlist(config: 'DictConfig', opts: 'List[str]', skip_missing: 'bool'=False, log_info: 'bool'=True):
    if opts is None:
        opts = []
    if len(opts) == 0:
        return config
    has_equal = opts[0].find('=') != -1
    if has_equal:
        opt_values = [opt.split('=', maxsplit=1) for opt in opts]
        if not all(len(opt) == 2 for opt in opt_values):
            for opt in opt_values:
                assert len(opt) == 2, f'{opt} has no value'
    else:
        assert len(opts) % 2 == 0, 'Number of opts should be multiple of 2'
        opt_values = zip(opts[0::2], opts[1::2])
    for opt, value in opt_values:
        if opt == 'dataset':
            opt = 'datasets'
        splits = opt.split('.')
        current = config
        for idx, field in enumerate(splits):
            array_index = -1
            if field.find('[') != -1 and field.find(']') != -1:
                stripped_field = field[:field.find('[')]
                array_index = int(field[field.find('[') + 1:field.find(']')])
            else:
                stripped_field = field
            if stripped_field not in current:
                if skip_missing is True:
                    break
                raise AttributeError('While updating configuration option {} is missing from configuration at field {}'.format(opt, stripped_field))
            if isinstance(current[stripped_field], collections.abc.Mapping):
                current = current[stripped_field]
            elif isinstance(current[stripped_field], collections.abc.Sequence) and array_index != -1:
                try:
                    current_value = current[stripped_field][array_index]
                except OCErrors.ConfigIndexError:
                    if skip_missing:
                        break
                    raise
                if not isinstance(current_value, (collections.abc.Mapping, collections.abc.Sequence)) or idx == len(splits) - 1:
                    if log_info:
                        logger.info(f'Overriding option {opt} to {value}')
                    current[stripped_field][array_index] = _decode_value(value)
                else:
                    current = current_value
            elif idx == len(splits) - 1:
                if log_info:
                    logger.info(f'Overriding option {opt} to {value}')
                current[stripped_field] = _decode_value(value)
            else:
                if skip_missing:
                    break
                raise AttributeError('While updating configuration', 'option {} is not present after field {}'.format(opt, stripped_field))
    return config


def get_default_config_path():
    directory = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(directory, '..', 'configs')
    fb_defaults = os.path.join(configs_dir, 'fb_defaults.yaml')
    if PathManager.exists(fb_defaults):
        return fb_defaults
    else:
        return os.path.join(configs_dir, 'defaults.yaml')

