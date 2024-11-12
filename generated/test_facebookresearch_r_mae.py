
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


import warnings


from torch.utils.data import DataLoader


from torch.utils.data import DistributedSampler


import copy


import torch


from torch.utils.data.dataset import Dataset


import math


import random


import numpy as np


from torchvision.datasets.vision import VisionDataset


import collections


import re


import torchvision


import itertools


from collections import defaultdict


from typing import TypeVar


from typing import Optional


from typing import Iterator


import torch.distributed as dist


from torch.utils.data.sampler import Sampler


from torch.utils.data import Dataset


from functools import partial


from torch.utils.data import ConcatDataset


import torchvision.transforms.functional as F


from torchvision.transforms.functional import _interpolation_modes_from_int


from torch import nn


import torch.nn as nn


import torch.nn.functional as F


import collections.abc


import torch.optim as optim


import logging


from itertools import chain


from math import inf


from typing import TYPE_CHECKING


from typing import Any


from typing import Callable


from typing import Dict


from typing import List


from typing import Type


from typing import Union


from collections import OrderedDict


import torch.profiler as profiler


from functools import wraps


import time


from torch.cuda.amp import GradScaler


import torchvision.transforms as transforms


from torchvision.utils import make_grid


from torchvision.ops.boxes import box_area


import functools


from torch import distributed as dist


import torch.backends.cudnn as cudnn


from torch.nn.init import _calculate_fan_in_and_fan_out


from typing import Generator


from typing import Tuple


from torch import Tensor


import torch.utils.checkpoint as torch_checkpoint


from torch.nn.modules.batchnorm import _BatchNorm


from torch.utils.tensorboard import SummaryWriter


from collections import deque


from typing import Set


from collections import abc


class BaseModel(nn.Module):
    """For integration with the trainer, datasets and other features,
    models needs to inherit this class, call `super`, write a build function,
    write a forward function taking a ``SampleList`` as input and returning a
    dict as output and finally, register it using ``@registry.register_model``

    Args:
        config (DictConfig): ``model_config`` configuration from global config.

    """

    def __init__(self, config, global_config):
        super().__init__()
        self.config = config
        self._global_config = global_config

    def _build(self):
        """Function to be implemented by the child class, in case they need to
        build their model separately than ``__init__``. All model related
        downloads should also happen here.
        """
        raise NotImplementedError('Build method not implemented in the child model class.')

    def build(self):
        self._build()
        self.inference(False)

    def inference(self, mode=True):
        if mode:
            super().train(False)
        self.inferencing = mode
        for module in self.modules():
            if hasattr(module, 'inferencing'):
                module.inferencing = mode
            else:
                setattr(module, 'inferencing', mode)

    def train(self, mode=True):
        if mode:
            self.inferencing = False
            for module in self.modules():
                if hasattr(module, 'inferencing'):
                    module.inferencing = False
                else:
                    setattr(module, 'inferencing', False)
        super().train(mode)


def drop_path(x, drop_prob: 'float'=0.0, training: 'bool'=False, scale_by_keep: 'bool'=True):
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
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if isinstance(drop, tuple):
            drop_probs = drop
        else:
            drop_probs = drop, drop
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        if isinstance(img_size, tuple):
            img_size = img_size
        else:
            img_size = img_size, img_size
        if isinstance(patch_size, tuple):
            patch_size = patch_size
        else:
            patch_size = patch_size, patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        bchw = x.shape
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, bchw


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, use_cls_token=True):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.use_cls_token = use_cls_token
        self.qkv_bias = qkv_bias
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = torch.matmul(q * self.scale, k.transpose(-2, -1))
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask.unsqueeze(1), -65504.0)
        if rel_pos_bias is not None:
            if not self.use_cls_token:
                rel_pos_bias = rel_pos_bias[:, 1:, 1:]
            attn = attn + rel_pos_bias
        attn_wo_softmax = attn
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        self.attention = attn
        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_wo_softmax


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, win_size=0, use_cls_token=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, use_cls_token=use_cls_token)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.win_size = win_size
        self.hw = None

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        ori_x = x
        x = self.norm1(x)
        x, attn = self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)
        x = ori_x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class CrossAttention(nn.Module):

    def __init__(self, qdim, kvdim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, use_cls_token=True):
        super().__init__()
        assert qdim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = qdim // num_heads
        self.scale = head_dim ** -0.5
        self.use_cls_token = use_cls_token
        self.qkv_bias = qkv_bias
        self.q = nn.Linear(qdim, qdim, bias=qkv_bias)
        self.kv = nn.Linear(kvdim, qdim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(qdim, qdim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, memory, rel_pos_bias=None, attn_mask=None, with_mask=False):
        B, N, C = query.shape
        L = memory.shape[1]
        q = self.q(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k, v = self.kv(memory).reshape(B, L, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        attn = torch.matmul(q * self.scale, k.transpose(-2, -1))
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask.unsqueeze(1), -65504.0)
        if rel_pos_bias is not None:
            if not self.use_cls_token:
                rel_pos_bias = rel_pos_bias[:, 1:, 1:]
            attn = attn + rel_pos_bias
        attn_wo_softmax = attn
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        self.attention = attn
        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_wo_softmax


class DecoderBlock(nn.Module):

    def __init__(self, dim, enc_dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_cls_token=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, use_cls_token=use_cls_token)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.cross_attn = CrossAttention(dim, enc_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, use_cls_token=use_cls_token)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, memory, rel_pos_bias=None, attn_mask=None):
        ori_x = x
        x = self.norm1(x)
        x, attn = self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)
        x = ori_x + self.drop_path(x)
        ori_x = x
        x = self.norm2(x)
        x, attn = self.cross_attn(x, memory, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)
        x = ori_x + self.drop_path(x)
        ori_x = x
        x = self.drop_path(self.mlp(self.norm3(x)))
        x = ori_x + self.drop_path(x)
        return x, attn


class DecoderBlockWithExpansion(nn.Module):

    def __init__(self, dim, enc_dim, num_heads, qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_cls_token=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, use_cls_token=use_cls_token)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.Identity()
        self.proj = nn.Linear(enc_dim, dim)
        self.norm3 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)

    def forward(self, x, memory, rel_pos_bias=None, attn_mask=None):
        ori_x = x
        x = self.norm1(x)
        x, attn = self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)
        x = ori_x + self.drop_path(x)
        ori_x = x
        x = self.norm2(x)
        x_mask = self.proj(memory)
        x_mask = ori_x.unsqueeze(2) + self.drop_path(x_mask).unsqueeze(1)
        x_mask = self.mlp(self.norm3(x_mask))
        return x_mask, attn


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
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
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


class MaskedAutoencoderViT(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=24, num_heads=16, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4.0, norm_layer=nn.LayerNorm, norm_pix_loss=False, use_mae_loss=True, **kwargs):
        super().__init__()
        if len(list(kwargs.keys())) > 0:
            warnings.warn(f'Arguments {list(kwargs.keys())} are unused in {self.__class__.__name__}')
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        if use_mae_loss:
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
            self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(decoder_depth)])
            self.decoder_norm = norm_layer(decoder_embed_dim)
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)
        self.norm_pix_loss = norm_pix_loss
        self.use_mae_loss = use_mae_loss
        self.patch_size = patch_size
        self.initialize_weights()

    def initialize_weights(self):
        None
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=0.02)
        if self.use_mae_loss:
            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5), cls_token=True)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
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
        x: (N, L, patch_size**2 *C)
        """
        c = imgs.shape[1]
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
        return x

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)[0]
        x = x + self.pos_embed[:, 1:, :]
        x, mask = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)[0]
        x = self.norm(x)
        return x, mask

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.expand(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], -1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)[0]
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x

    def _forward_mae_loss(self, pred, pred_target):
        l2_loss = (pred - pred_target) ** 2
        l2_loss = l2_loss.mean(dim=-1)
        return l2_loss

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-06) ** 0.5
        mae_loss = self._forward_mae_loss(pred, target)
        mae_loss = (mae_loss * mask).sum() / mask.sum()
        return mae_loss, mae_loss

    def forward_loss(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss, metric = self.forward_loss(imgs, pred, mask)
        return loss, pred, metric, mask


class RegionMaskedAutoencoderViT(MaskedAutoencoderViT):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=24, num_heads=16, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4.0, norm_layer=nn.LayerNorm, norm_pix_loss=False, num_region=0, use_mae_loss=True, mae_loss_weight=1.0, bg_loss_weight=1.0, region_loss_weight=1.0, region_mask_ratio=0.75, region_enc_dim=768, region_sample_type='random'):
        if use_mae_loss is False and num_region == 0:
            raise ValueError('There should be at least one loss in training. Found use_mae_loss=False and num_region=0!')
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads, decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer, norm_pix_loss=norm_pix_loss, use_mae_loss=use_mae_loss)
        self.num_region = num_region
        self.mae_loss_weight = mae_loss_weight
        self.bg_loss_weight = bg_loss_weight
        self.region_loss_weight = region_loss_weight
        self.region_mask_ratio = region_mask_ratio
        self.region_enc_dim = region_enc_dim
        assert region_sample_type in ('random', 'random_fg'), 'Only random|random_fg are allowed for region_sample_type'
        self.region_sample_type = region_sample_type

    def random_masking(self, x, mask_ratio, region, shuffle_ids):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        region: [N, num_region, L, region_enc_dim]
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        if self.num_region > 0:
            len_region_keep = int(L * (1 - self.region_mask_ratio))
            if self.region_sample_type == 'random':
                noise = torch.rand(N, L, device=x.device)
                region_shuffle = torch.argsort(noise, dim=1)
            elif self.region_sample_type == 'random_fg':
                region_shuffle = shuffle_ids
            region_restore = torch.argsort(region_shuffle, dim=1)
            region_keep = region_shuffle[:, :len_region_keep]
            region_keep = region_keep.unsqueeze(-1).expand(-1, -1, self.region_enc_dim)
            region_mask = torch.ones([N, L], device=x.device)
            region_mask[:, :len_region_keep] = 0
            region_mask = torch.gather(region_mask, dim=1, index=region_restore)
            if region is not None:
                if region_keep.dim() < region.dim():
                    region_keep = region_keep.unsqueeze(1).expand(-1, self.num_region, -1, -1)
                region_masked = torch.gather(region, dim=-2, index=region_keep)
            else:
                region_masked = None
        else:
            region_mask = None
            region_masked = None
            region_restore = None
        if self.num_region > 0:
            ids_shuffle = region_shuffle
            ids_restore = region_restore
        else:
            noise = torch.rand(N, L, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore, region_masked, region_mask, region_restore

    def forward_encoder(self, x, mask_ratio, shuffle_ids=None, region=None):
        x = self.patch_embed(x)[0]
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore, region, region_mask, region_restore = self.random_masking(x, mask_ratio, region, shuffle_ids)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)[0]
        x = self.norm(x)
        return x, mask, ids_restore, region, region_mask, region_restore

    def forward_region_decoder(self):
        raise NotImplementedError

    def _forward_region_loss(self, pred, pred_target):
        region_loss = F.binary_cross_entropy_with_logits(pred, pred_target, reduction='none')
        if self.bg_loss_weight != 1.0:
            weight_loss = pred_target.detach().clone()
            weight_loss[pred_target == 0] = self.bg_loss_weight
            region_loss = region_loss * weight_loss
        region_loss = region_loss.mean(dim=-1)
        return region_loss

    def forward_loss(self, imgs, pred, mask, pred_region, region_mask, target_region):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        if self.use_mae_loss:
            target = self.patchify(imgs)
            if self.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1e-06) ** 0.5
            mae_loss = self._forward_mae_loss(pred, target)
        else:
            mae_loss = torch.zeros_like(mask)
        if self.num_region > 0:
            region_loss = self._forward_region_loss(pred_region, target_region)
        else:
            region_loss = torch.zeros_like(mae_loss)
            region_mask = torch.ones_like(mask)
        mae_loss = (mae_loss * mask).sum() / mask.sum()
        region_loss = (region_loss * region_mask).sum() / region_mask.sum()
        loss = mae_loss * self.mae_loss_weight + region_loss * self.region_loss_weight
        return loss, mae_loss, region_loss

    def forward(self):
        raise NotImplementedError


class RegionQueryRMAE(RegionMaskedAutoencoderViT):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=24, num_heads=16, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4.0, norm_layer=nn.LayerNorm, norm_pix_loss=False, num_region=0, use_mae_loss=True, mae_loss_weight=1.0, bg_loss_weight=1.0, region_loss_weight=1.0, region_mask_ratio=0.75, region_enc_dim=768, region_enc_depth=1, region_enc_num_heads=8, region_dec_dim=128, region_dec_depth=1, region_dec_num_heads=8, region_sample_type='random', region_cross_layer=8):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads, decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth if use_mae_loss else region_cross_layer, decoder_num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer, norm_pix_loss=norm_pix_loss, num_region=num_region, use_mae_loss=use_mae_loss, mae_loss_weight=mae_loss_weight, bg_loss_weight=bg_loss_weight, region_loss_weight=region_loss_weight, region_mask_ratio=region_mask_ratio, region_enc_dim=region_enc_dim, region_sample_type=region_sample_type)
        self.use_mae_loss = use_mae_loss
        if use_mae_loss is False:
            self.decoder_norm = None
            self.decoder_pred = None
        if num_region > 0:
            num_patches = self.patch_embed.num_patches
            self.region_cross_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, region_dec_dim), requires_grad=False)
            self.region_cross_embed = nn.Linear(embed_dim, region_dec_dim, bias=True)
            self.region_dec_norm = norm_layer(region_dec_dim)
            self.region_cross_mask_token = nn.Parameter(torch.zeros(1, 1, region_dec_dim))
            self.region_cross_blocks = nn.ModuleList()
            for _ in range(region_cross_layer):
                self.region_cross_blocks.append(Block(region_dec_dim, region_dec_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer))
            cross_embed_dim = region_dec_dim
            self.region_patch_embed = PatchEmbed(img_size, patch_size, 1, region_enc_dim)
            self.region_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, region_enc_dim), requires_grad=False)
            self.region_enc_blocks = nn.ModuleList([Block(region_enc_dim, region_enc_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(region_enc_depth)])
            self.region_proj = nn.Sequential(norm_layer(region_enc_dim), nn.Linear(region_enc_dim, region_dec_dim))
            self.region_dec_blocks = nn.ModuleList()
            for _ in range(region_dec_depth - 1):
                self.region_dec_blocks.append(DecoderBlock(region_dec_dim, cross_embed_dim, region_dec_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer))
            self.region_dec_blocks.append(DecoderBlockWithExpansion(region_dec_dim, cross_embed_dim, region_dec_num_heads, qkv_bias=True, norm_layer=norm_layer))
            self.region_pred = nn.Sequential(nn.GELU(), nn.Linear(region_dec_dim, patch_size ** 2, bias=True))
        self.region_cross_layer = region_cross_layer
        self.initialize_rmae_weights()

    def initialize_rmae_weights(self):
        super().initialize_weights()
        if self.num_region > 0:
            region_pos_embed = get_2d_sincos_pos_embed(self.region_pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5), cls_token=True)
            self.region_pos_embed.data.copy_(torch.from_numpy(region_pos_embed).float().unsqueeze(0))
            w = self.region_patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            torch.nn.init.normal_(self.region_cross_mask_token, std=0.02)
            region_cross_pos_embed = get_2d_sincos_pos_embed(self.region_cross_pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5), cls_token=True)
            self.region_cross_pos_embed.data.copy_(torch.from_numpy(region_cross_pos_embed).float().unsqueeze(0))
        self.apply(self._init_weights)

    def forward_encoder(self, x, mask_ratio, shuffle_ids=None, region=None):
        if self.num_region > 0:
            region = region + self.region_pos_embed[None, :, 1:, :]
        return super().forward_encoder(x, mask_ratio, shuffle_ids=shuffle_ids, region=region)

    def forward_decoder(self, x, ids_restore):
        x_cross = None
        if self.num_region > 0:
            x_cross = self.region_cross_embed(x)
            cross_mask_tokens = self.region_cross_mask_token.expand(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], -1)
            x_cross_ = torch.cat([x_cross[:, 1:, :], cross_mask_tokens], dim=1)
            x_cross_ = torch.gather(x_cross_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_cross.shape[2]))
            x_cross = torch.cat([x_cross[:, :1, :], x_cross_], dim=1)
            x_cross = x_cross + self.region_cross_pos_embed
            for blk in self.region_cross_blocks:
                x_cross = blk(x_cross)[0]
            x_cross = self.region_dec_norm(x_cross)
        if self.use_mae_loss:
            x = self.decoder_embed(x)
            mask_tokens = self.mask_token.expand(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], -1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
            x = torch.cat([x[:, :1, :], x_], dim=1)
            x = x + self.decoder_pos_embed
            for i, blk in enumerate(self.decoder_blocks):
                x = blk(x)[0]
        if self.use_mae_loss:
            x = self.decoder_norm(x)
            x = self.decoder_pred(x)
            x = x[:, 1:, :]
        else:
            x = None
        return x, x_cross

    def forward_region_encoder(self, region):
        l = region.shape[1]
        for blk in self.region_enc_blocks:
            region = blk(region)[0]
        region = region.view(-1, self.num_region, l, self.region_enc_dim)
        region = region.mean(dim=2)
        return region

    def forward_region_decoder(self, region, memory):
        region = self.region_proj(region)
        for blk in self.region_dec_blocks:
            region = blk(region, memory)[0]
        region = self.region_pred(region)
        region = region[:, :, 1:, :]
        return region

    def _forward_region_loss(self, pred, pred_target):
        region_loss = super()._forward_region_loss(pred, pred_target)
        region_loss = region_loss.mean(dim=1)
        return region_loss

    def forward(self, imgs, mask_ratio=0.75, region=None, shuffle_ids=None):
        if region is not None:
            b, c, h, w = region.shape
            region = region.view(b * c, 1, h, w)
            target_region = self.patchify(region).view(b, c, -1, self.patch_size ** 2)
            region = self.region_patch_embed(region - 0.5)[0]
            region = region.view(b, c, -1, self.region_enc_dim)
        else:
            region = None
            target_region = None
        latent, mask, ids_restore, region_masked, region_mask, _ = self.forward_encoder(imgs, mask_ratio, shuffle_ids=shuffle_ids, region=region)
        if region is not None:
            region_latent = self.forward_region_encoder(region_masked.flatten(0, 1))
        else:
            region_latent = None
        pred, memory = self.forward_decoder(latent, ids_restore)
        if self.num_region > 0:
            pred_region = self.forward_region_decoder(region_latent, memory)
        else:
            pred_region = None
        loss, mae_loss, region_loss = self.forward_loss(imgs, pred, mask, pred_region, region_mask, target_region)
        return loss, (pred, pred_region), (mae_loss, region_loss), (mask, region_mask)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Block,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (DecoderBlock,
     lambda: ([], {'dim': 4, 'enc_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (DecoderBlockWithExpansion,
     lambda: ([], {'dim': 4, 'enc_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PatchEmbed,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

