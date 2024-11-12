
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


import random


from torch.nn.functional import interpolate


import torch as th


import torch.nn as nn


import re


from functools import partial


import numpy as np


from torch import Tensor


from torch.nn import functional as F


from torchvision.transforms import ToTensor


from torchvision.transforms import ToPILImage


from functools import lru_cache


from torchvision.transforms.functional import normalize


from typing import List


from typing import Tuple


from torch.hub import download_url_to_file


from torch.hub import get_dir


from time import time


import math


import torch.version


import torch.nn.functional as F


import time


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import pandas as pd


import warnings


from typing import Optional


from typing import Union


from torch.nn import CrossEntropyLoss


from torch import nn


from typing import Dict


from types import MethodType


from typing import Any


from collections.abc import Sequence


from abc import ABC


from abc import abstractmethod


import uuid


from torch.utils.data import Sampler


import logging


from typing import Sequence


import logging.handlers


from torch.optim.lr_scheduler import LambdaLR


from inspect import isfunction


from collections import namedtuple


from torchvision import models


import functools


from typing import Callable


from typing import Iterable


from scipy import integrate


from torch.utils.checkpoint import checkpoint


import torch.cuda


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f'unsupported dimensions: {dims}')


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class ZeroConv(nn.Module):

    def __init__(self, label_nc, norm_nc, mask=False):
        super().__init__()
        self.zero_conv = zero_module(conv_nd(2, label_nc, norm_nc, 1, 1, 0))
        self.mask = mask

    def forward(self, c, h, h_ori=None):
        if not self.mask:
            h = h + self.zero_conv(c)
        else:
            h = h + self.zero_conv(c) * torch.zeros_like(h)
        if h_ori is not None:
            h = th.cat([h_ori, h], dim=1)
        return h


class GroupNorm32(nn.GroupNorm):

    def forward(self, x):
        return super().forward(x)


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


class ZeroSFT(nn.Module):

    def __init__(self, label_nc, norm_nc, concat_channels=0, norm=True, mask=False):
        super().__init__()
        ks = 3
        pw = ks // 2
        self.norm = norm
        if self.norm:
            self.param_free_norm = normalization(norm_nc + concat_channels)
        else:
            self.param_free_norm = nn.Identity()
        nhidden = 128
        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.SiLU())
        self.zero_mul = zero_module(nn.Conv2d(nhidden, norm_nc + concat_channels, kernel_size=ks, padding=pw))
        self.zero_add = zero_module(nn.Conv2d(nhidden, norm_nc + concat_channels, kernel_size=ks, padding=pw))
        self.zero_conv = zero_module(conv_nd(2, label_nc, norm_nc, 1, 1, 0))
        self.pre_concat = bool(concat_channels != 0)
        self.mask = mask

    def forward(self, c, h, h_ori=None, control_scale=1):
        assert self.mask is False
        if h_ori is not None and self.pre_concat:
            h_raw = th.cat([h_ori, h], dim=1)
        else:
            h_raw = h
        if self.mask:
            h = h + self.zero_conv(c) * torch.zeros_like(h)
        else:
            h = h + self.zero_conv(c)
        if h_ori is not None and self.pre_concat:
            h = th.cat([h_ori, h], dim=1)
        actv = self.mlp_shared(c)
        gamma = self.zero_mul(actv)
        beta = self.zero_add(actv)
        if self.mask:
            gamma = gamma * torch.zeros_like(gamma)
            beta = beta * torch.zeros_like(beta)
        h = self.param_free_norm(h) * (gamma + 1) + beta
        if h_ori is not None and not self.pre_concat:
            h = th.cat([h_ori, h], dim=1)
        return h * control_scale + h_raw * (1 - control_scale)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class CrossAttention(nn.Module):

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, backend=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.backend = backend

    def forward(self, x, context=None, mask=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        h = self.heads
        if additional_tokens is not None:
            n_tokens_to_mask = additional_tokens.shape[1]
            x = torch.cat([additional_tokens, x], dim=1)
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        if n_times_crossframe_attn_in_self:
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            n_cp = x.shape[0] // n_times_crossframe_attn_in_self
            k = repeat(k[::n_times_crossframe_attn_in_self], 'b ... -> (b n) ...', n=n_cp)
            v = repeat(v[::n_times_crossframe_attn_in_self], 'b ... -> (b n) ...', n=n_cp)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        """
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        """
        with sdp_kernel(**BACKEND_MAP[self.backend]):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        del q, k, v
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        if additional_tokens is not None:
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, **kwargs):
        super().__init__()
        None
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: 'Optional[Any]' = None

    def forward(self, x, context=None, mask=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        if additional_tokens is not None:
            n_tokens_to_mask = additional_tokens.shape[1]
            x = torch.cat([additional_tokens, x], dim=1)
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        if n_times_crossframe_attn_in_self:
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            k = repeat(k[::n_times_crossframe_attn_in_self], 'b ... -> (b n) ...', n=n_times_crossframe_attn_in_self)
            v = repeat(v[::n_times_crossframe_attn_in_self], 'b ... -> (b n) ...', n=n_times_crossframe_attn_in_self)
        b, _, _ = q.shape
        q, k, v = map(lambda t: t.unsqueeze(3).reshape(b, t.shape[1], self.heads, self.dim_head).permute(0, 2, 1, 3).reshape(b * self.heads, t.shape[1], self.dim_head).contiguous(), (q, k, v))
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)
        if exists(mask):
            raise NotImplementedError
        out = out.unsqueeze(0).reshape(b, self.heads, out.shape[1], self.dim_head).permute(0, 2, 1, 3).reshape(b, out.shape[1], self.heads * self.dim_head)
        if additional_tokens is not None:
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


class ZeroCrossAttn(nn.Module):
    ATTENTION_MODES = {'softmax': CrossAttention, 'softmax-xformers': MemoryEfficientCrossAttention}

    def __init__(self, context_dim, query_dim, zero_out=True, mask=False):
        super().__init__()
        attn_mode = 'softmax-xformers' if XFORMERS_IS_AVAILBLE else 'softmax'
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.attn = attn_cls(query_dim=query_dim, context_dim=context_dim, heads=query_dim // 64, dim_head=64)
        self.norm1 = normalization(query_dim)
        self.norm2 = normalization(context_dim)
        self.mask = mask

    def forward(self, context, x, control_scale=1):
        assert self.mask is False
        x_in = x
        x = self.norm1(x)
        context = self.norm2(context)
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        context = rearrange(context, 'b c h w -> b (h w) c').contiguous()
        x = self.attn(x, context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if self.mask:
            x = x * torch.zeros_like(x)
        x = x_in + x * control_scale
        return x


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * num_spatial ** 2 * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum('bct,bcs->bts', (q * scale).view(bs * self.n_heads, ch, length), (k * scale).view(bs * self.n_heads, ch, length))
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum('bts,bcs->bct', weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/output heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum('bct,bcs->bts', q * scale, k * scale)
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum('bts,bcs->bct', weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, num_head_channels=-1, use_checkpoint=False, use_new_attention_order=False):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, f'q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}'
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            self.attention = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, **kwargs):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f'unsupported dimensions: {dims}')


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, third_down=False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2) if not third_down else (2, 2, 2)
        if use_conv:
            None
            None
            if dims == 3:
                None
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, third_up=False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.third_up = third_up
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        _dtype = x.dtype
        x = x
        assert x.shape[1] == self.channels
        if self.dims == 3:
            t_factor = 1 if not self.third_up else 2
            x = F.interpolate(x, (t_factor * x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode='nearest')
        else:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = x
        if self.use_conv:
            x = self.conv(x)
        return x


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=False, use_scale_shift_norm=False, dims=2, use_checkpoint=False, up=False, down=False, kernel_size=3, exchange_temb_dims=False, skip_t_emb=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims
        if isinstance(kernel_size, Iterable):
            padding = [(k // 2) for k in kernel_size]
        else:
            padding = kernel_size // 2
        self.in_layers = nn.Sequential(normalization(channels), nn.SiLU(), conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding))
        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        self.skip_t_emb = skip_t_emb
        self.emb_out_channels = 2 * self.out_channels if use_scale_shift_norm else self.out_channels
        if self.skip_t_emb:
            None
            assert not self.use_scale_shift_norm
            self.emb_layers = None
            self.exchange_temb_dims = False
        else:
            self.emb_layers = nn.Sequential(nn.SiLU(), linear(emb_channels, self.emb_out_channels))
        self.out_layers = nn.Sequential(normalization(self.out_channels), nn.SiLU(), nn.Dropout(p=dropout), zero_module(conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=padding)))
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        if self.skip_t_emb:
            emb_out = th.zeros_like(h)
        else:
            emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            if self.exchange_temb_dims:
                emb_out = rearrange(emb_out, 'b t c ... -> b c t ...')
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class GEGLU(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):

    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)
        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {'softmax': CrossAttention, 'softmax-xformers': MemoryEfficientCrossAttention}

    def __init__(self, dim, n_heads, d_head, dropout=0.0, context_dim=None, gated_ff=True, checkpoint=True, disable_self_attn=False, attn_mode='softmax', sdp_backend=None):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        if attn_mode != 'softmax' and not XFORMERS_IS_AVAILABLE:
            None
            attn_mode = 'softmax'
        elif attn_mode == 'softmax' and not SDP_IS_AVAILABLE:
            None
            if not XFORMERS_IS_AVAILABLE:
                assert False, "Please install xformers via e.g. 'pip install xformers==0.0.16'"
            else:
                None
                attn_mode = 'softmax-xformers'
        attn_cls = self.ATTENTION_MODES[attn_mode]
        if version.parse(torch.__version__) >= version.parse('2.0.0'):
            assert sdp_backend is None or isinstance(sdp_backend, SDPBackend)
        else:
            assert sdp_backend is None
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, context_dim=context_dim if self.disable_self_attn else None, backend=sdp_backend)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout, backend=sdp_backend)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        if self.checkpoint:
            None

    def forward(self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        kwargs = {'x': x}
        if context is not None:
            kwargs.update({'context': context})
        if additional_tokens is not None:
            kwargs.update({'additional_tokens': additional_tokens})
        if n_times_crossframe_attn_in_self:
            kwargs.update({'n_times_crossframe_attn_in_self': n_times_crossframe_attn_in_self})
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None, additional_tokens=additional_tokens, n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self if not self.disable_self_attn else 0) + x
        x = self.attn2(self.norm2(x), context=context, additional_tokens=additional_tokens) + x
        x = self.ff(self.norm3(x)) + x
        return x


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-06, affine=True)


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0.0, context_dim=None, disable_self_attn=False, use_linear=False, attn_type='softmax', use_checkpoint=True, sdp_backend=None):
        super().__init__()
        None
        if exists(context_dim) and not isinstance(context_dim, (list, ListConfig)):
            context_dim = [context_dim]
        if exists(context_dim) and isinstance(context_dim, list):
            if depth != len(context_dim):
                None
                assert all(map(lambda x: x == context_dim[0], context_dim)), 'need homogenous context_dim to match depth automatically'
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d], disable_self_attn=disable_self_attn, attn_mode=attn_type, checkpoint=use_checkpoint, sdp_backend=sdp_backend) for d in range(depth)])
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            if i > 0 and len(context) == 1:
                i = 0
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


class Timestep(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        return timestep_embedding(t, self.dim)


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None, skip_time_mix=False, time_context=None, num_video_frames=None, time_context_cat=None, use_crossframe_attention_in_spatial_layers=False):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


def convert_module_to_f16(x):
    pass


def convert_module_to_f32(x):
    pass


class GLVControl(nn.Module):

    def __init__(self, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True, dims=2, num_classes=None, use_checkpoint=False, use_fp16=False, num_heads=-1, num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False, use_spatial_transformer=False, transformer_depth=1, context_dim=None, n_embed=None, legacy=True, disable_self_attentions=None, num_attention_blocks=None, disable_middle_self_attn=False, use_linear_in_transformer=False, spatial_transformer_attn_type='softmax', adm_in_channels=None, use_fairscale_checkpoint=False, offload_to_cpu=False, transformer_depth_middle=None, input_upscale=1):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'
        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'
        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        elif isinstance(transformer_depth, ListConfig):
            transformer_depth = list(transformer_depth)
        transformer_depth_middle = default(transformer_depth_middle, transformer_depth[-1])
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError('provide num_res_blocks either as an int (globally constant) or as a list/tuple (per-level) with the same length as channel_mult')
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            None
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        if use_fp16:
            None
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        assert use_fairscale_checkpoint != use_checkpoint or not (use_checkpoint or use_fairscale_checkpoint)
        self.use_fairscale_checkpoint = False
        checkpoint_wrapper_fn = partial(checkpoint_wrapper, offload_to_cpu=offload_to_cpu) if self.use_fairscale_checkpoint else lambda x: x
        time_embed_dim = model_channels * 4
        self.time_embed = checkpoint_wrapper_fn(nn.Sequential(linear(model_channels, time_embed_dim), nn.SiLU(), linear(time_embed_dim, time_embed_dim)))
        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == 'continuous':
                None
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == 'timestep':
                self.label_emb = checkpoint_wrapper_fn(nn.Sequential(Timestep(model_channels), nn.Sequential(linear(model_channels, time_embed_dim), nn.SiLU(), linear(time_embed_dim, time_embed_dim))))
            elif self.num_classes == 'sequential':
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(nn.Sequential(linear(adm_in_channels, time_embed_dim), nn.SiLU(), linear(time_embed_dim, time_embed_dim)))
            else:
                raise ValueError()
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))])
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [checkpoint_wrapper_fn(ResBlock(ch, time_embed_dim, dropout, out_channels=mult * model_channels, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm))]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False
                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(checkpoint_wrapper_fn(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=dim_head, use_new_attention_order=use_new_attention_order)) if not use_spatial_transformer else checkpoint_wrapper_fn(SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth[level], context_dim=context_dim, disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer, attn_type=spatial_transformer_attn_type, use_checkpoint=use_checkpoint)))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(TimestepEmbedSequential(checkpoint_wrapper_fn(ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True)) if resblock_updown else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(checkpoint_wrapper_fn(ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)), checkpoint_wrapper_fn(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=dim_head, use_new_attention_order=use_new_attention_order)) if not use_spatial_transformer else checkpoint_wrapper_fn(SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth_middle, context_dim=context_dim, disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer, attn_type=spatial_transformer_attn_type, use_checkpoint=use_checkpoint)), checkpoint_wrapper_fn(ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)))
        self.input_upscale = input_upscale
        self.input_hint_block = TimestepEmbedSequential(zero_module(conv_nd(dims, in_channels, model_channels, 3, padding=1)))

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x, timesteps, xt, context=None, y=None, **kwargs):
        xt, context, y = xt, context, y
        if self.input_upscale != 1:
            x = nn.functional.interpolate(x, scale_factor=self.input_upscale, mode='bilinear', antialias=True)
        assert (y is not None) == (self.num_classes is not None), 'must specify y if and only if the model is class-conditional'
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        if self.num_classes is not None:
            assert y.shape[0] == xt.shape[0]
            emb = emb + self.label_emb(y)
        guided_hint = self.input_hint_block(x, emb, context)
        h = xt
        for module in self.input_blocks:
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        hs.append(h)
        return hs


def _cast_if_autocast_enabled(tensor):
    if torch.is_autocast_enabled():
        if tensor.device.type == 'cuda':
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == 'cpu':
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor
    return tensor


class LPLayerNorm(torch.nn.LayerNorm):

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None):
        super().__init__(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device=device, dtype=dtype)

    def forward(self, x):
        module_device = x.device
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
        downcast_bias = _cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
        with torch.autocast(enabled=False, device_type=module_device.type):
            return torch.nn.functional.layer_norm(downcast_x, self.normalized_shape, downcast_weight, downcast_bias, self.eps)


def _reset_is_causal(num_query_tokens: 'int', num_key_tokens: 'int', original_is_causal: 'bool'):
    if original_is_causal and num_query_tokens != num_key_tokens:
        if num_query_tokens != 1:
            raise NotImplementedError('MPT does not support query and key with different number of tokens, unless number of query tokens is 1.')
        else:
            return False
    return original_is_causal


def check_valid_inputs(*tensors, valid_dtypes=[torch.float16, torch.bfloat16]):
    for tensor in tensors:
        if tensor.dtype not in valid_dtypes:
            raise TypeError(f'tensor.dtype={tensor.dtype!r} must be in valid_dtypes={valid_dtypes!r}.')
        if not tensor.is_cuda:
            raise TypeError(f'Inputs must be cuda tensors (tensor.is_cuda={tensor.is_cuda!r}).')


def scaled_multihead_dot_product_attention(query, key, value, n_heads, past_key_value=None, softmax_scale=None, attn_bias=None, key_padding_mask=None, is_causal=False, dropout_p=0.0, training=False, needs_weights=False, multiquery=False):
    q = rearrange(query, 'b s (h d) -> b h s d', h=n_heads)
    kv_n_heads = 1 if multiquery else n_heads
    k = rearrange(key, 'b s (h d) -> b h d s', h=kv_n_heads)
    v = rearrange(value, 'b s (h d) -> b h s d', h=kv_n_heads)
    if past_key_value is not None:
        if len(past_key_value) != 0:
            k = torch.cat([past_key_value[0], k], dim=3)
            v = torch.cat([past_key_value[1], v], dim=2)
        past_key_value = k, v
    b, _, s_q, d = q.shape
    s_k = k.size(-1)
    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(d)
    attn_weight = q.matmul(k) * softmax_scale
    if attn_bias is not None:
        _s_q = max(0, attn_bias.size(2) - s_q)
        _s_k = max(0, attn_bias.size(3) - s_k)
        attn_bias = attn_bias[:, :, _s_q:, _s_k:]
        if attn_bias.size(-1) != 1 and attn_bias.size(-1) != s_k or attn_bias.size(-2) != 1 and attn_bias.size(-2) != s_q:
            raise RuntimeError(f'attn_bias (shape: {attn_bias.shape}) is expected to broadcast to shape: {attn_weight.shape}.')
        attn_weight = attn_weight + attn_bias
    min_val = torch.finfo(q.dtype).min
    if key_padding_mask is not None:
        if attn_bias is not None:
            warnings.warn('Propogating key_padding_mask to the attention module ' + 'and applying it within the attention module can cause ' + 'unneccessary computation/memory usage. Consider integrating ' + 'into attn_bias once and passing that to each attention ' + 'module instead.')
        attn_weight = attn_weight.masked_fill(~key_padding_mask.view((b, 1, 1, s_k)), min_val)
    if is_causal and not q.size(2) == 1:
        s = max(s_q, s_k)
        causal_mask = attn_weight.new_ones(s, s, dtype=torch.float16)
        causal_mask = causal_mask.tril()
        causal_mask = causal_mask
        causal_mask = ~causal_mask
        causal_mask = causal_mask[-s_q:, -s_k:]
        attn_weight = attn_weight.masked_fill(causal_mask.view(1, 1, s_q, s_k), min_val)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if dropout_p:
        attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p, training=training, inplace=True)
    out = attn_weight.matmul(v)
    out = rearrange(out, 'b h s d -> b s (h d)')
    if needs_weights:
        return out, attn_weight, past_key_value
    return out, None, past_key_value


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def _flash_attn_backward(do, q, k, v, o, lse, dq, dk, dv, bias=None, causal=False, softmax_scale=None):
    if do.stride(-1) != 1:
        do = do.contiguous()
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    assert d <= 128
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    assert lse.shape == (batch, nheads, seqlen_q_rounded)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1
    assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == 1
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    dq_accum = torch.empty_like(q, dtype=torch.float32)
    delta = torch.empty_like(lse)
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    grid = lambda META: (triton.cdiv(seqlen_q, META['BLOCK_M']), batch * nheads)
    _bwd_preprocess_do_o_dot[grid](o, do, delta, o.stride(0), o.stride(2), o.stride(1), do.stride(0), do.stride(2), do.stride(1), nheads, seqlen_q, seqlen_q_rounded, d, BLOCK_M=128, BLOCK_HEADDIM=BLOCK_HEADDIM)
    has_bias = bias is not None
    bias_type = 'none'
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.dim() == 4
        assert bias.stride(-1) == 1
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = 'vector'
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = 'matrix'
        else:
            raise RuntimeError('Last 2 dimensions of bias must be (1, seqlen_k) or (seqlen_q, seqlen_k)')
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)
    grid = lambda META: (triton.cdiv(seqlen_k, META['BLOCK_N']) if META['SEQUENCE_PARALLEL'] else 1, batch * nheads)
    _bwd_kernel[grid](q, k, v, bias, do, dq_accum, dk, dv, lse, delta, softmax_scale, q.stride(0), q.stride(2), q.stride(1), k.stride(0), k.stride(2), k.stride(1), v.stride(0), v.stride(2), v.stride(1), *bias_strides, do.stride(0), do.stride(2), do.stride(1), dq_accum.stride(0), dq_accum.stride(2), dq_accum.stride(1), dk.stride(0), dk.stride(2), dk.stride(1), dv.stride(0), dv.stride(2), dv.stride(1), nheads, seqlen_q, seqlen_k, seqlen_q_rounded, d, seqlen_q // 32, seqlen_k // 32, bias_type, causal, BLOCK_HEADDIM)
    dq.copy_(dq_accum)


def _flash_attn_forward(q, k, v, bias=None, causal=False, softmax_scale=None):
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, 'FlashAttention only support head dimensions up to 128'
    assert q.dtype == k.dtype == v.dtype, 'All tensors must have the same type'
    assert q.dtype in [torch.float16, torch.bfloat16], 'Only support fp16 and bf16'
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    has_bias = bias is not None
    bias_type = 'none'
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.dim() == 4
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = 'vector'
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = 'matrix'
        else:
            raise RuntimeError('Last 2 dimensions of bias must be (1, seqlen_k) or (seqlen_q, seqlen_k)')
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 128
    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen_q, META['BLOCK_M']), batch * nheads)
    _fwd_kernel[grid](q, k, v, bias, o, lse, tmp, softmax_scale, q.stride(0), q.stride(2), q.stride(1), k.stride(0), k.stride(2), k.stride(1), v.stride(0), v.stride(2), v.stride(1), *bias_strides, o.stride(0), o.stride(2), o.stride(1), nheads, seqlen_q, seqlen_k, seqlen_q_rounded, d, seqlen_q // 32, seqlen_k // 32, bias_type, causal, BLOCK_HEADDIM, BLOCK_M=BLOCK, BLOCK_N=BLOCK, num_warps=num_warps, num_stages=1)
    return o, lse, softmax_scale


class FlashAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, bias=None, causal=False, softmax_scale=None):
        """
            q: (batch_size, seqlen_q, nheads, headdim)
            k, v: (batch_size, seqlen_k, nheads, headdim)
            bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
                For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
                ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
        """
        q, k, v = [(x if x.stride(-1) == 1 else x.contiguous()) for x in [q, k, v]]
        o, lse, ctx.softmax_scale = _flash_attn_forward(q, k, v, bias=bias, causal=causal, softmax_scale=softmax_scale)
        ctx.save_for_backward(q, k, v, o, lse, bias)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, bias = ctx.saved_tensors
        assert not ctx.needs_input_grad[3], 'FlashAttention does not support bias gradient yet'
        with torch.inference_mode():
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            _flash_attn_backward(do, q, k, v, o, lse, dq, dk, dv, bias=bias, causal=ctx.causal, softmax_scale=ctx.softmax_scale)
        return dq, dk, dv, None, None, None


flash_attn_func = FlashAttnFunc.apply


class MultiheadAttention(nn.Module):
    """Multi-head self attention.

    Using torch or triton attention implementation enables user to also use
    additive bias.
    """

    def __init__(self, d_model: 'int', n_heads: 'int', attn_impl: 'str'='triton', clip_qkv: 'Optional[float]'=None, qk_ln: 'bool'=False, softmax_scale: 'Optional[float]'=None, attn_pdrop: 'float'=0.0, low_precision_layernorm: 'bool'=False, verbose: 'int'=0, device: 'Optional[str]'=None):
        super().__init__()
        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.qk_ln = qk_ln
        self.d_model = d_model
        self.n_heads = n_heads
        self.softmax_scale = softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model / self.n_heads)
        self.attn_dropout_p = attn_pdrop
        self.Wqkv = nn.Linear(self.d_model, 3 * self.d_model, device=device)
        fuse_splits = d_model, 2 * d_model
        self.Wqkv._fused = 0, fuse_splits
        if self.qk_ln:
            layernorm_class = LPLayerNorm if low_precision_layernorm else nn.LayerNorm
            self.q_ln = layernorm_class(self.d_model, device=device)
            self.k_ln = layernorm_class(self.d_model, device=device)
        if self.attn_impl == 'flash':
            self.attn_fn = flash_attn_fn
        elif self.attn_impl == 'triton':
            self.attn_fn = triton_flash_attn_fn
            if verbose:
                warnings.warn('While `attn_impl: triton` can be faster than `attn_impl: flash` ' + 'it uses more memory. When training larger models this can trigger ' + 'alloc retries which hurts performance. If encountered, we recommend ' + 'using `attn_impl: flash` if your model does not use `alibi` or `prefix_lm`.')
        elif self.attn_impl == 'torch':
            self.attn_fn = scaled_multihead_dot_product_attention
            if torch.cuda.is_available() and verbose:
                warnings.warn('Using `attn_impl: torch`. If your model does not use `alibi` or ' + '`prefix_lm` we recommend using `attn_impl: flash` otherwise ' + 'we recommend using `attn_impl: triton`.')
        else:
            raise ValueError(f'attn_impl={attn_impl!r} is an invalid setting.')
        self.out_proj = nn.Linear(self.d_model, self.d_model, device=device)
        self.out_proj._is_residual = True

    def forward(self, x, past_key_value=None, attn_bias=None, attention_mask=None, is_causal=True, needs_weights=False):
        qkv = self.Wqkv(x)
        if self.clip_qkv:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        query, key, value = qkv.chunk(3, dim=2)
        key_padding_mask = attention_mask
        if self.qk_ln:
            dtype = query.dtype
            query = self.q_ln(query)
            key = self.k_ln(key)
        context, attn_weights, past_key_value = self.attn_fn(query, key, value, self.n_heads, past_key_value=past_key_value, softmax_scale=self.softmax_scale, attn_bias=attn_bias, key_padding_mask=key_padding_mask, is_causal=is_causal, dropout_p=self.attn_dropout_p, training=self.training, needs_weights=needs_weights)
        return self.out_proj(context), attn_weights, past_key_value


class MultiQueryAttention(nn.Module):
    """Multi-Query self attention.

    Using torch or triton attention implementation enables user to also use
    additive bias.
    """

    def __init__(self, d_model: 'int', n_heads: 'int', attn_impl: 'str'='triton', clip_qkv: 'Optional[float]'=None, qk_ln: 'bool'=False, softmax_scale: 'Optional[float]'=None, attn_pdrop: 'float'=0.0, low_precision_layernorm: 'bool'=False, verbose: 'int'=0, device: 'Optional[str]'=None):
        super().__init__()
        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.qk_ln = qk_ln
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.softmax_scale = softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.head_dim)
        self.attn_dropout_p = attn_pdrop
        self.Wqkv = nn.Linear(d_model, d_model + 2 * self.head_dim, device=device)
        fuse_splits = d_model, d_model + self.head_dim
        self.Wqkv._fused = 0, fuse_splits
        if self.qk_ln:
            layernorm_class = LPLayerNorm if low_precision_layernorm else nn.LayerNorm
            self.q_ln = layernorm_class(d_model, device=device)
            self.k_ln = layernorm_class(self.head_dim, device=device)
        if self.attn_impl == 'flash':
            self.attn_fn = flash_attn_fn
        elif self.attn_impl == 'triton':
            self.attn_fn = triton_flash_attn_fn
            if verbose:
                warnings.warn('While `attn_impl: triton` can be faster than `attn_impl: flash` ' + 'it uses more memory. When training larger models this can trigger ' + 'alloc retries which hurts performance. If encountered, we recommend ' + 'using `attn_impl: flash` if your model does not use `alibi` or `prefix_lm`.')
        elif self.attn_impl == 'torch':
            self.attn_fn = scaled_multihead_dot_product_attention
            if torch.cuda.is_available() and verbose:
                warnings.warn('Using `attn_impl: torch`. If your model does not use `alibi` or ' + '`prefix_lm` we recommend using `attn_impl: flash` otherwise ' + 'we recommend using `attn_impl: triton`.')
        else:
            raise ValueError(f'attn_impl={attn_impl!r} is an invalid setting.')
        self.out_proj = nn.Linear(self.d_model, self.d_model, device=device)
        self.out_proj._is_residual = True

    def forward(self, x, past_key_value=None, attn_bias=None, attention_mask=None, is_causal=True, needs_weights=False):
        qkv = self.Wqkv(x)
        if self.clip_qkv:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        query, key, value = qkv.split([self.d_model, self.head_dim, self.head_dim], dim=2)
        key_padding_mask = attention_mask
        if self.qk_ln:
            dtype = query.dtype
            query = self.q_ln(query)
            key = self.k_ln(key)
        context, attn_weights, past_key_value = self.attn_fn(query, key, value, self.n_heads, past_key_value=past_key_value, softmax_scale=self.softmax_scale, attn_bias=attn_bias, key_padding_mask=key_padding_mask, is_causal=is_causal, dropout_p=self.attn_dropout_p, training=self.training, needs_weights=needs_weights, multiquery=True)
        return self.out_proj(context), attn_weights, past_key_value


class MPTMLP(nn.Module):

    def __init__(self, d_model: 'int', expansion_ratio: 'int', device: 'Optional[str]'=None):
        super().__init__()
        self.up_proj = nn.Linear(d_model, expansion_ratio * d_model, device=device)
        self.act = nn.GELU(approximate='none')
        self.down_proj = nn.Linear(expansion_ratio * d_model, d_model, device=device)
        self.down_proj._is_residual = True

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))


ATTN_CLASS_REGISTRY = {'multihead_attention': MultiheadAttention, 'multiquery_attention': MultiQueryAttention}


def rms_norm(x, weight=None, eps=1e-05):
    output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        return output * weight
    return output


class RMSNorm(torch.nn.Module):

    def __init__(self, normalized_shape, eps=1e-05, weight=True, dtype=None, device=None):
        super().__init__()
        self.eps = eps
        if weight:
            self.weight = torch.nn.Parameter(torch.ones(normalized_shape, dtype=dtype, device=device))
        else:
            self.register_parameter('weight', None)

    def forward(self, x):
        return rms_norm(x.float(), self.weight, self.eps)


class LPRMSNorm(RMSNorm):

    def __init__(self, normalized_shape, eps=1e-05, weight=True, dtype=None, device=None):
        super().__init__(normalized_shape=normalized_shape, eps=eps, weight=weight, dtype=dtype, device=device)

    def forward(self, x):
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
        with torch.autocast(enabled=False, device_type=x.device.type):
            return rms_norm(downcast_x, downcast_weight, self.eps)


NORM_CLASS_REGISTRY = {'layernorm': torch.nn.LayerNorm, 'low_precision_layernorm': LPLayerNorm, 'rmsnorm': RMSNorm, 'low_precision_rmsnorm': LPRMSNorm}


class MPTBlock(nn.Module):

    def __init__(self, d_model: 'int', n_heads: 'int', expansion_ratio: 'int', attn_config: 'Dict'={'attn_type': 'multihead_attention', 'attn_pdrop': 0.0, 'attn_impl': 'triton', 'qk_ln': False, 'clip_qkv': None, 'softmax_scale': None, 'prefix_lm': False, 'attn_uses_sequence_id': False, 'alibi': False, 'alibi_bias_max': 8}, resid_pdrop: 'float'=0.0, norm_type: 'str'='low_precision_layernorm', verbose: 'int'=0, device: 'Optional[str]'=None, **kwargs):
        del kwargs
        super().__init__()
        norm_class = NORM_CLASS_REGISTRY[norm_type.lower()]
        attn_class = ATTN_CLASS_REGISTRY[attn_config['attn_type']]
        self.norm_1 = norm_class(d_model, device=device)
        self.attn = attn_class(attn_impl=attn_config['attn_impl'], clip_qkv=attn_config['clip_qkv'], qk_ln=attn_config['qk_ln'], softmax_scale=attn_config['softmax_scale'], attn_pdrop=attn_config['attn_pdrop'], d_model=d_model, n_heads=n_heads, verbose=verbose, device=device)
        self.norm_2 = norm_class(d_model, device=device)
        self.ffn = MPTMLP(d_model=d_model, expansion_ratio=expansion_ratio, device=device)
        self.resid_attn_dropout = nn.Dropout(resid_pdrop)
        self.resid_ffn_dropout = nn.Dropout(resid_pdrop)

    def forward(self, x: 'torch.Tensor', past_key_value: 'Optional[Tuple[torch.Tensor]]'=None, attn_bias: 'Optional[torch.Tensor]'=None, attention_mask: 'Optional[torch.ByteTensor]'=None, is_causal: 'bool'=True) ->Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        a = self.norm_1(x)
        b, attn_weights, past_key_value = self.attn(a, past_key_value=past_key_value, attn_bias=attn_bias, attention_mask=attention_mask, is_causal=is_causal)
        x = x + self.resid_attn_dropout(b)
        m = self.norm_2(x)
        n = self.ffn(m)
        x = x + self.resid_ffn_dropout(n)
        return x, attn_weights, past_key_value


class SharedEmbedding(nn.Embedding):

    def forward(self, input: 'Tensor', unembed: 'bool'=False) ->Tensor:
        if unembed:
            return F.linear(input, self.weight)
        return super().forward(input)


LLAVA_CLIP_PATH = '/opt/data/private/AIGC_pretrain/LLaVA1.5/clip-vit-large-patch14-336'


class CLIPVisionTower(nn.Module):

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        None
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name if LLAVA_CLIP_PATH is None else LLAVA_CLIP_PATH)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name if LLAVA_CLIP_PATH is None else LLAVA_CLIP_PATH)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name if LLAVA_CLIP_PATH is None else LLAVA_CLIP_PATH)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images, output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class IdentityMap(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {'mm_projector_type': 'identity'}


class SimpleResBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)
        self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class LinearAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)
        w_ = w_ * int(c) ** -0.5
        w_ = torch.nn.functional.softmax(w_, dim=2)
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)
        return x + h_


class BasicTransformerSingleLayerBlock(nn.Module):
    ATTENTION_MODES = {'softmax': CrossAttention, 'softmax-xformers': MemoryEfficientCrossAttention}

    def __init__(self, dim, n_heads, d_head, dropout=0.0, context_dim=None, gated_ff=True, checkpoint=True, attn_mode='softmax'):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, context_dim=context_dim)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context) + x
        x = self.ff(self.norm2(x)) + x
        return x


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)


class ScalingLayer(nn.Module):

    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-0.03, -0.088, -0.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.458, 0.448, 0.45])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


CKPT_MAP = {'vgg_lpips': 'vgg.pth'}


MD5_MAP = {'vgg_lpips': 'd507d7349b931f0638a25a48a722f98a'}


URL_MAP = {'vgg_lpips': 'https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1'}


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get('content-length', 0))
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            with open(local_path, 'wb') as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, 'rb') as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or check and not md5_hash(path) == MD5_MAP[name]:
        None
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


class vgg16(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class LPIPS(nn.Module):

    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name='vgg_lpips'):
        ckpt = get_ckpt_path(name, 'sgm/modules/autoencoding/lpips/loss')
        self.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')), strict=False)
        None

    @classmethod
    def from_pretrained(cls, name='vgg_lpips'):
        if name != 'vgg_lpips':
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name)
        model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')), strict=False)
        return model

    def forward(self, input, target):
        in0_input, in1_input = self.scaling_layer(input), self.scaling_layer(target)
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit('.', 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not 'target' in config:
        if config == '__is_first_stage__':
            return None
        elif config == '__is_unconditional__':
            return None
        raise KeyError('Expected key `target` to instantiate.')
    return get_obj_from_str(config['target'])(**config.get('params', dict()))


class LatentLPIPS(nn.Module):

    def __init__(self, decoder_config, perceptual_weight=1.0, latent_weight=1.0, scale_input_to_tgt_size=False, scale_tgt_to_input_size=False, perceptual_weight_on_inputs=0.0):
        super().__init__()
        self.scale_input_to_tgt_size = scale_input_to_tgt_size
        self.scale_tgt_to_input_size = scale_tgt_to_input_size
        self.init_decoder(decoder_config)
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.latent_weight = latent_weight
        self.perceptual_weight_on_inputs = perceptual_weight_on_inputs

    def init_decoder(self, config):
        self.decoder = instantiate_from_config(config)
        if hasattr(self.decoder, 'encoder'):
            del self.decoder.encoder

    def forward(self, latent_inputs, latent_predictions, image_inputs, split='train'):
        log = dict()
        loss = (latent_inputs - latent_predictions) ** 2
        log[f'{split}/latent_l2_loss'] = loss.mean().detach()
        image_reconstructions = None
        if self.perceptual_weight > 0.0:
            image_reconstructions = self.decoder.decode(latent_predictions)
            image_targets = self.decoder.decode(latent_inputs)
            perceptual_loss = self.perceptual_loss(image_targets.contiguous(), image_reconstructions.contiguous())
            loss = self.latent_weight * loss.mean() + self.perceptual_weight * perceptual_loss.mean()
            log[f'{split}/perceptual_loss'] = perceptual_loss.mean().detach()
        if self.perceptual_weight_on_inputs > 0.0:
            image_reconstructions = default(image_reconstructions, self.decoder.decode(latent_predictions))
            if self.scale_input_to_tgt_size:
                image_inputs = torch.nn.functional.interpolate(image_inputs, image_reconstructions.shape[2:], mode='bicubic', antialias=True)
            elif self.scale_tgt_to_input_size:
                image_reconstructions = torch.nn.functional.interpolate(image_reconstructions, image_inputs.shape[2:], mode='bicubic', antialias=True)
            perceptual_loss2 = self.perceptual_loss(image_inputs.contiguous(), image_reconstructions.contiguous())
            loss = loss + self.perceptual_weight_on_inputs * perceptual_loss2.mean()
            log[f'{split}/perceptual_loss_on_inputs'] = perceptual_loss2.mean().detach()
        return loss, log


class ActNorm(nn.Module):

    def __init__(self, num_features, logdet=False, affine=True, allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = flatten.mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
            std = flatten.std(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-06))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False
        _, _, height, width = input.shape
        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)
        h = self.scale * (input + self.loc)
        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0])
            return h, logdet
        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError('Initializing ActNorm in reverse direction is disabled by default. Use allow_reverse_init=True to enable.')
            else:
                self.initialize(output)
                self.initialized.fill_(1)
        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False
        h = output / self.scale - self.loc
        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (torch.mean(torch.nn.functional.softplus(-logits_real)) + torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GeneralLPIPSWithDiscriminator(nn.Module):

    def __init__(self, disc_start: 'int', logvar_init: 'float'=0.0, pixelloss_weight=1.0, disc_num_layers: 'int'=3, disc_in_channels: 'int'=3, disc_factor: 'float'=1.0, disc_weight: 'float'=1.0, perceptual_weight: 'float'=1.0, disc_loss: 'str'='hinge', scale_input_to_tgt_size: 'bool'=False, dims: 'int'=2, learn_logvar: 'bool'=False, regularization_weights: 'Union[None, dict]'=None):
        super().__init__()
        self.dims = dims
        if self.dims > 2:
            None
        self.scale_input_to_tgt_size = scale_input_to_tgt_size
        assert disc_loss in ['hinge', 'vanilla']
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.learn_logvar = learn_logvar
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=False).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == 'hinge' else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.regularization_weights = default(regularization_weights, {})

    def get_trainable_parameters(self) ->Any:
        return self.discriminator.parameters()

    def get_trainable_autoencoder_parameters(self) ->Any:
        if self.learn_logvar:
            yield self.logvar
        yield from ()

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 0.0001)
        d_weight = torch.clamp(d_weight, 0.0, 10000.0).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, regularization_log, inputs, reconstructions, optimizer_idx, global_step, last_layer=None, split='train', weights=None):
        if self.scale_input_to_tgt_size:
            inputs = torch.nn.functional.interpolate(inputs, reconstructions.shape[2:], mode='bicubic', antialias=True)
        if self.dims > 2:
            inputs, reconstructions = map(lambda x: rearrange(x, 'b c t h w -> (b t) c h w'), (inputs, reconstructions))
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        if optimizer_idx == 0:
            logits_fake = self.discriminator(reconstructions.contiguous())
            g_loss = -torch.mean(logits_fake)
            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + d_weight * disc_factor * g_loss
            log = dict()
            for k in regularization_log:
                if k in self.regularization_weights:
                    loss = loss + self.regularization_weights[k] * regularization_log[k]
                log[f'{split}/{k}'] = regularization_log[k].detach().mean()
            log.update({'{}/total_loss'.format(split): loss.clone().detach().mean(), '{}/logvar'.format(split): self.logvar.detach(), '{}/nll_loss'.format(split): nll_loss.detach().mean(), '{}/rec_loss'.format(split): rec_loss.detach().mean(), '{}/d_weight'.format(split): d_weight.detach(), '{}/disc_factor'.format(split): torch.tensor(disc_factor), '{}/g_loss'.format(split): g_loss.detach().mean()})
            return loss, log
        if optimizer_idx == 1:
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            log = {'{}/disc_loss'.format(split): d_loss.clone().detach().mean(), '{}/logits_real'.format(split): logits_real.detach().mean(), '{}/logits_fake'.format(split): logits_fake.detach().mean()}
            return d_loss, log


class AbstractRegularizer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, z: 'torch.Tensor') ->Tuple[torch.Tensor, dict]:
        raise NotImplementedError()

    @abstractmethod
    def get_trainable_parameters(self) ->Any:
        raise NotImplementedError()


class DiagonalGaussianDistribution(object):

    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        elif other is None:
            return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
        else:
            return 0.5 * torch.sum(torch.pow(self.mean - other.mean, 2) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar, dim=[1, 2, 3])

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean


class DiagonalGaussianRegularizer(AbstractRegularizer):

    def __init__(self, sample: 'bool'=True):
        super().__init__()
        self.sample = sample

    def get_trainable_parameters(self) ->Any:
        yield from ()

    def forward(self, z: 'torch.Tensor') ->Tuple[torch.Tensor, dict]:
        log = dict()
        posterior = DiagonalGaussianDistribution(z)
        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode()
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        log['kl_loss'] = kl_loss
        return z, log


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


class Denoiser(nn.Module):

    def __init__(self, weighting_config, scaling_config):
        super().__init__()
        self.weighting = instantiate_from_config(weighting_config)
        self.scaling = instantiate_from_config(scaling_config)

    def possibly_quantize_sigma(self, sigma):
        return sigma

    def possibly_quantize_c_noise(self, c_noise):
        return c_noise

    def w(self, sigma):
        return self.weighting(sigma)

    def __call__(self, network, input, sigma, cond):
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        return network(input * c_in, c_noise, cond) * c_out + input * c_skip


class DiscreteDenoiser(Denoiser):

    def __init__(self, weighting_config, scaling_config, num_idx, discretization_config, do_append_zero=False, quantize_c_noise=True, flip=True):
        super().__init__(weighting_config, scaling_config)
        sigmas = instantiate_from_config(discretization_config)(num_idx, do_append_zero=do_append_zero, flip=flip)
        self.register_buffer('sigmas', sigmas)
        self.quantize_c_noise = quantize_c_noise

    def sigma_to_idx(self, sigma):
        dists = sigma - self.sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def possibly_quantize_sigma(self, sigma):
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise):
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise


class DiscreteDenoiserWithControl(DiscreteDenoiser):

    def __call__(self, network, input, sigma, cond, control_scale):
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        return network(input * c_in, c_noise, cond, control_scale) * c_out + input * c_skip


class StandardDiffusionLoss(nn.Module):

    def __init__(self, sigma_sampler_config, type='l2', offset_noise_level=0.0, batch2model_keys: 'Optional[Union[str, List[str], ListConfig]]'=None):
        super().__init__()
        assert type in ['l2', 'l1', 'lpips']
        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.type = type
        self.offset_noise_level = offset_noise_level
        if type == 'lpips':
            self.lpips = LPIPS().eval()
        if not batch2model_keys:
            batch2model_keys = []
        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]
        self.batch2model_keys = set(batch2model_keys)

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}
        sigmas = self.sigma_sampler(input.shape[0])
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            noise = noise + self.offset_noise_level * append_dims(torch.randn(input.shape[0], device=input.device), input.ndim)
        noised_input = input + noise * append_dims(sigmas, input.ndim)
        model_output = denoiser(network, noised_input, sigmas, cond, **additional_model_inputs)
        w = append_dims(denoiser.w(sigmas), input.ndim)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if self.type == 'l2':
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.type == 'l1':
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.type == 'lpips':
            loss = self.lpips(model_output, target).reshape(-1)
            return loss


def nonlinearity(x):
    return x * torch.sigmoid(x)


class ResnetBlock(nn.Module):

    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""

    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def attention(self, h_: 'torch.Tensor') ->torch.Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q, k, v = map(lambda x: rearrange(x, 'b c h w -> b 1 (h w) c').contiguous(), (q, k, v))
        h_ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return rearrange(h_, 'b 1 (h w) c -> b c h w', h=h, w=w, c=c, b=b)

    def forward(self, x, **kwargs):
        h_ = x
        h_ = self.attention(h_)
        h_ = self.proj_out(h_)
        return x + h_


class MemoryEfficientAttnBlock(nn.Module):
    """
    Uses xformers efficient implementation,
    see https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    Note: this is a single-head self-attention operation
    """

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.attention_op: 'Optional[Any]' = None

    def attention(self, h_: 'torch.Tensor') ->torch.Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        B, C, H, W = q.shape
        q, k, v = map(lambda x: rearrange(x, 'b c h w -> b (h w) c'), (q, k, v))
        q, k, v = map(lambda t: t.unsqueeze(3).reshape(B, t.shape[1], 1, C).permute(0, 2, 1, 3).reshape(B * 1, t.shape[1], C).contiguous(), (q, k, v))
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)
        out = out.unsqueeze(0).reshape(B, 1, out.shape[1], C).permute(0, 2, 1, 3).reshape(B, out.shape[1], C)
        return rearrange(out, 'b (h w) c -> b c h w', b=B, h=H, w=W, c=C)

    def forward(self, x, **kwargs):
        h_ = x
        h_ = self.attention(h_)
        h_ = self.proj_out(h_)
        return x + h_


class MemoryEfficientCrossAttentionWrapper(MemoryEfficientCrossAttention):

    def forward(self, x, context=None, mask=None, **unused_kwargs):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        out = super().forward(x, context=context, mask=mask)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w, c=c)
        return x + out


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def make_attn(in_channels, attn_type='vanilla', attn_kwargs=None):
    assert attn_type in ['vanilla', 'vanilla-xformers', 'memory-efficient-cross-attn', 'linear', 'none'], f'attn_type {attn_type} unknown'
    if version.parse(torch.__version__) < version.parse('2.0.0') and attn_type != 'none':
        assert XFORMERS_IS_AVAILABLE, f"We do not support vanilla attention in {torch.__version__} anymore, as it is too expensive. Please install xformers via e.g. 'pip install xformers==0.0.16'"
        attn_type = 'vanilla-xformers'
    None
    if attn_type == 'vanilla':
        assert attn_kwargs is None
        return AttnBlock(in_channels)
    elif attn_type == 'vanilla-xformers':
        None
        return MemoryEfficientAttnBlock(in_channels)
    elif type == 'memory-efficient-cross-attn':
        attn_kwargs['query_dim'] = in_channels
        return MemoryEfficientCrossAttentionWrapper(**attn_kwargs)
    elif attn_type == 'none':
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


class Model(nn.Module):

    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels, resolution, use_timestep=True, use_linear_attn=False, attn_type='vanilla'):
        super().__init__()
        if use_linear_attn:
            attn_type = 'linear'
        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.use_timestep = use_timestep
        if self.use_timestep:
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([torch.nn.Linear(self.ch, self.temb_ch), torch.nn.Linear(self.temb_ch, self.temb_ch)])
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t=None, context=None):
        if context is not None:
            x = torch.cat((x, context), dim=1)
        if self.use_timestep:
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self):
        return self.conv_out.weight


class Encoder(nn.Module):

    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels, resolution, z_channels, double_z=True, use_linear_attn=False, attn_type='vanilla', **ignore_kwargs):
        super().__init__()
        if use_linear_attn:
            attn_type = 'linear'
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        temb = None
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):

    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels, resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False, attn_type='vanilla', **ignorekwargs):
        super().__init__()
        if use_linear_attn:
            attn_type = 'linear'
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = 1, z_channels, curr_res, curr_res
        None
        make_attn_cls = self._make_attn()
        make_resblock_cls = self._make_resblock()
        make_conv_cls = self._make_conv()
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = make_resblock_cls(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = make_attn_cls(block_in, attn_type=attn_type)
        self.mid.block_2 = make_resblock_cls(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(make_resblock_cls(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn_cls(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)
        self.norm_out = Normalize(block_in)
        self.conv_out = make_conv_cls(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def _make_attn(self) ->Callable:
        return make_attn

    def _make_resblock(self) ->Callable:
        return ResnetBlock

    def _make_conv(self) ->Callable:
        return torch.nn.Conv2d

    def get_last_layer(self, **kwargs):
        return self.conv_out.weight

    def forward(self, z, **kwargs):
        self.last_z_shape = z.shape
        temb = None
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb, **kwargs)
        h = self.mid.attn_1(h, **kwargs)
        h = self.mid.block_2(h, temb, **kwargs)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, **kwargs)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, **kwargs)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h, **kwargs)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(self, spacial_dim: 'int', embed_dim: 'int', num_heads_channels: 'int', output_dim: 'int'=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)
        x = x + self.positional_embedding[None, :, :]
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TransposedUpsample(nn.Module):
    """Learned 2x upsampling without padding"""

    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.up = nn.ConvTranspose2d(self.channels, self.out_channels, kernel_size=ks, stride=2)

    def forward(self, x):
        return self.up(x)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(self, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True, dims=2, num_classes=None, use_checkpoint=False, use_fp16=False, num_heads=-1, num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False, use_spatial_transformer=False, transformer_depth=1, context_dim=None, n_embed=None, legacy=True, disable_self_attentions=None, num_attention_blocks=None, disable_middle_self_attn=False, use_linear_in_transformer=False, spatial_transformer_attn_type='softmax', adm_in_channels=None, use_fairscale_checkpoint=False, offload_to_cpu=False, transformer_depth_middle=None):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'
        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'
        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        elif isinstance(transformer_depth, ListConfig):
            transformer_depth = list(transformer_depth)
        transformer_depth_middle = default(transformer_depth_middle, transformer_depth[-1])
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError('provide num_res_blocks either as an int (globally constant) or as a list/tuple (per-level) with the same length as channel_mult')
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            None
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        if use_fp16:
            None
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        assert use_fairscale_checkpoint != use_checkpoint or not (use_checkpoint or use_fairscale_checkpoint)
        self.use_fairscale_checkpoint = False
        checkpoint_wrapper_fn = partial(checkpoint_wrapper, offload_to_cpu=offload_to_cpu) if self.use_fairscale_checkpoint else lambda x: x
        time_embed_dim = model_channels * 4
        self.time_embed = checkpoint_wrapper_fn(nn.Sequential(linear(model_channels, time_embed_dim), nn.SiLU(), linear(time_embed_dim, time_embed_dim)))
        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == 'continuous':
                None
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == 'timestep':
                self.label_emb = checkpoint_wrapper_fn(nn.Sequential(Timestep(model_channels), nn.Sequential(linear(model_channels, time_embed_dim), nn.SiLU(), linear(time_embed_dim, time_embed_dim))))
            elif self.num_classes == 'sequential':
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(nn.Sequential(linear(adm_in_channels, time_embed_dim), nn.SiLU(), linear(time_embed_dim, time_embed_dim)))
            else:
                raise ValueError()
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))])
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [checkpoint_wrapper_fn(ResBlock(ch, time_embed_dim, dropout, out_channels=mult * model_channels, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm))]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False
                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(checkpoint_wrapper_fn(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=dim_head, use_new_attention_order=use_new_attention_order)) if not use_spatial_transformer else checkpoint_wrapper_fn(SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth[level], context_dim=context_dim, disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer, attn_type=spatial_transformer_attn_type, use_checkpoint=use_checkpoint)))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(TimestepEmbedSequential(checkpoint_wrapper_fn(ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True)) if resblock_updown else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(checkpoint_wrapper_fn(ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)), checkpoint_wrapper_fn(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=dim_head, use_new_attention_order=use_new_attention_order)) if not use_spatial_transformer else checkpoint_wrapper_fn(SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth_middle, context_dim=context_dim, disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer, attn_type=spatial_transformer_attn_type, use_checkpoint=use_checkpoint)), checkpoint_wrapper_fn(ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)))
        self._feature_size += ch
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [checkpoint_wrapper_fn(ResBlock(ch + ich, time_embed_dim, dropout, out_channels=model_channels * mult, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm))]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False
                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(checkpoint_wrapper_fn(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads_upsample, num_head_channels=dim_head, use_new_attention_order=use_new_attention_order)) if not use_spatial_transformer else checkpoint_wrapper_fn(SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth[level], context_dim=context_dim, disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer, attn_type=spatial_transformer_attn_type, use_checkpoint=use_checkpoint)))
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(checkpoint_wrapper_fn(ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, up=True)) if resblock_updown else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
        self.out = checkpoint_wrapper_fn(nn.Sequential(normalization(ch), nn.SiLU(), zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1))))
        if self.predict_codebook_ids:
            self.id_predictor = checkpoint_wrapper_fn(nn.Sequential(normalization(ch), conv_nd(dims, model_channels, n_embed, 1)))

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (self.num_classes is not None), 'must specify y if and only if the model is class-conditional'
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)
        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            assert False, 'not supported anymore. what the f*** are you doing?'
        else:
            return self.out(h)


class NoTimeUNetModel(UNetModel):

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        timesteps = th.zeros_like(timesteps)
        return super().forward(x, timesteps, context, y, **kwargs)


class EncoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.
    For usage, see UNet.
    """

    def __init__(self, image_size, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True, dims=2, use_checkpoint=False, use_fp16=False, num_heads=1, num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False, pool='adaptive', *args, **kwargs):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(linear(model_channels, time_embed_dim), nn.SiLU(), linear(time_embed_dim, time_embed_dim))
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))])
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, dropout, out_channels=mult * model_channels, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(TimestepEmbedSequential(ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True) if resblock_updown else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        self.middle_block = TimestepEmbedSequential(ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm), AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order), ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm))
        self._feature_size += ch
        self.pool = pool
        if pool == 'adaptive':
            self.out = nn.Sequential(normalization(ch), nn.SiLU(), nn.AdaptiveAvgPool2d((1, 1)), zero_module(conv_nd(dims, ch, out_channels, 1)), nn.Flatten())
        elif pool == 'attention':
            assert num_head_channels != -1
            self.out = nn.Sequential(normalization(ch), nn.SiLU(), AttentionPool2d(image_size // ds, ch, num_head_channels, out_channels))
        elif pool == 'spatial':
            self.out = nn.Sequential(nn.Linear(self._feature_size, 2048), nn.ReLU(), nn.Linear(2048, self.out_channels))
        elif pool == 'spatial_v2':
            self.out = nn.Sequential(nn.Linear(self._feature_size, 2048), normalization(2048), nn.SiLU(), nn.Linear(2048, self.out_channels))
        else:
            raise NotImplementedError(f'Unexpected {pool} pooling')

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        results = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith('spatial'):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)
        if self.pool.startswith('spatial'):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)


class SiLU(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


class IdentityWrapper(nn.Module):

    def __init__(self, diffusion_model, compile_model: 'bool'=False):
        super().__init__()
        compile = torch.compile if version.parse(torch.__version__) >= version.parse('2.0.0') and compile_model else lambda x: x
        self.diffusion_model = compile(diffusion_model)

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class OpenAIWrapper(IdentityWrapper):

    def forward(self, x: 'torch.Tensor', t: 'torch.Tensor', c: 'dict', **kwargs) ->torch.Tensor:
        x = torch.cat((x, c.get('concat', torch.Tensor([]).type_as(x))), dim=1)
        return self.diffusion_model(x, timesteps=t, context=c.get('crossattn', None), y=c.get('vector', None), **kwargs)


class OpenAIHalfWrapper(IdentityWrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diffusion_model = self.diffusion_model.half()

    def forward(self, x: 'torch.Tensor', t: 'torch.Tensor', c: 'dict', **kwargs) ->torch.Tensor:
        x = torch.cat((x, c.get('concat', torch.Tensor([]).type_as(x))), dim=1)
        _context = c.get('crossattn', None)
        _y = c.get('vector', None)
        if _context is not None:
            _context = _context.half()
        if _y is not None:
            _y = _y.half()
        x = x.half()
        t = t.half()
        out = self.diffusion_model(x, timesteps=t, context=_context, y=_y, **kwargs)
        return out.float()


class ControlWrapper(nn.Module):

    def __init__(self, diffusion_model, compile_model: 'bool'=False, dtype=torch.float32):
        super().__init__()
        self.compile = torch.compile if version.parse(torch.__version__) >= version.parse('2.0.0') and compile_model else lambda x: x
        self.diffusion_model = self.compile(diffusion_model)
        self.control_model = None
        self.dtype = dtype

    def load_control_model(self, control_model):
        self.control_model = self.compile(control_model)

    def forward(self, x: 'torch.Tensor', t: 'torch.Tensor', c: 'dict', control_scale=1, **kwargs) ->torch.Tensor:
        with torch.autocast('cuda', dtype=self.dtype):
            control = self.control_model(x=c.get('control', None), timesteps=t, xt=x, control_vector=c.get('control_vector', None), mask_x=c.get('mask_x', None), context=c.get('crossattn', None), y=c.get('vector', None))
            out = self.diffusion_model(x, timesteps=t, context=c.get('crossattn', None), y=c.get('vector', None), control=control, control_scale=control_scale, **kwargs)
        return out.float()


class LitEma(nn.Module):

    def __init__(self, model, decay=0.9999, use_num_upates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.m_name2s_name = {}
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int) if use_num_upates else torch.tensor(-1, dtype=torch.int))
        for name, p in model.named_parameters():
            if p.requires_grad:
                s_name = name.replace('.', '')
                self.m_name2s_name.update({name: s_name})
                self.register_buffer(s_name, p.clone().detach().data)
        self.collected_params = []

    def reset_num_updates(self):
        del self.num_updates
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int))

    def forward(self, model):
        decay = self.decay
        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())
            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)


class AbstractEmbModel(nn.Module):

    def __init__(self):
        super().__init__()
        self._is_trainable = None
        self._ucg_rate = None
        self._input_key = None

    @property
    def is_trainable(self) ->bool:
        return self._is_trainable

    @property
    def ucg_rate(self) ->Union[float, torch.Tensor]:
        return self._ucg_rate

    @property
    def input_key(self) ->str:
        return self._input_key

    @is_trainable.setter
    def is_trainable(self, value: 'bool'):
        self._is_trainable = value

    @ucg_rate.setter
    def ucg_rate(self, value: 'Union[float, torch.Tensor]'):
        self._ucg_rate = value

    @input_key.setter
    def input_key(self, value: 'str'):
        self._input_key = value

    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable

    @ucg_rate.deleter
    def ucg_rate(self):
        del self._ucg_rate

    @input_key.deleter
    def input_key(self):
        del self._input_key


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def expand_dims_like(x, y):
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    return x


class GeneralConditioner(nn.Module):
    OUTPUT_DIM2KEYS = {(2): 'vector', (3): 'crossattn', (4): 'concat', (5): 'concat'}
    KEY2CATDIM = {'vector': 1, 'crossattn': 2, 'concat': 1, 'control_vector': 1}

    def __init__(self, emb_models: 'Union[List, ListConfig]'):
        super().__init__()
        embedders = []
        for n, embconfig in enumerate(emb_models):
            embedder = instantiate_from_config(embconfig)
            assert isinstance(embedder, AbstractEmbModel), f'embedder model {embedder.__class__.__name__} has to inherit from AbstractEmbModel'
            embedder.is_trainable = embconfig.get('is_trainable', False)
            embedder.ucg_rate = embconfig.get('ucg_rate', 0.0)
            if not embedder.is_trainable:
                embedder.train = disabled_train
                for param in embedder.parameters():
                    param.requires_grad = False
                embedder.eval()
            None
            if 'input_key' in embconfig:
                embedder.input_key = embconfig['input_key']
            elif 'input_keys' in embconfig:
                embedder.input_keys = embconfig['input_keys']
            else:
                raise KeyError(f"need either 'input_key' or 'input_keys' for embedder {embedder.__class__.__name__}")
            embedder.legacy_ucg_val = embconfig.get('legacy_ucg_value', None)
            if embedder.legacy_ucg_val is not None:
                embedder.ucg_prng = np.random.RandomState()
            embedders.append(embedder)
        self.embedders = nn.ModuleList(embedders)

    def possibly_get_ucg_val(self, embedder: 'AbstractEmbModel', batch: 'Dict') ->Dict:
        assert embedder.legacy_ucg_val is not None
        p = embedder.ucg_rate
        val = embedder.legacy_ucg_val
        for i in range(len(batch[embedder.input_key])):
            if embedder.ucg_prng.choice(2, p=[1 - p, p]):
                batch[embedder.input_key][i] = val
        return batch

    def forward(self, batch: 'Dict', force_zero_embeddings: 'Optional[List]'=None) ->Dict:
        output = dict()
        if force_zero_embeddings is None:
            force_zero_embeddings = []
        for embedder in self.embedders:
            embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
            with embedding_context():
                if hasattr(embedder, 'input_key') and embedder.input_key is not None:
                    if embedder.legacy_ucg_val is not None:
                        batch = self.possibly_get_ucg_val(embedder, batch)
                    emb_out = embedder(batch[embedder.input_key])
                elif hasattr(embedder, 'input_keys'):
                    emb_out = embedder(*[batch[k] for k in embedder.input_keys])
            assert isinstance(emb_out, (torch.Tensor, list, tuple)), f'encoder outputs must be tensors or a sequence, but got {type(emb_out)}'
            if not isinstance(emb_out, (list, tuple)):
                emb_out = [emb_out]
            for emb in emb_out:
                out_key = self.OUTPUT_DIM2KEYS[emb.dim()]
                if embedder.ucg_rate > 0.0 and embedder.legacy_ucg_val is None:
                    emb = expand_dims_like(torch.bernoulli((1.0 - embedder.ucg_rate) * torch.ones(emb.shape[0], device=emb.device)), emb) * emb
                if hasattr(embedder, 'input_key') and embedder.input_key in force_zero_embeddings:
                    emb = torch.zeros_like(emb)
                if out_key in output:
                    output[out_key] = torch.cat((output[out_key], emb), self.KEY2CATDIM[out_key])
                else:
                    output[out_key] = emb
        return output

    def get_unconditional_conditioning(self, batch_c, batch_uc=None, force_uc_zero_embeddings=None):
        if force_uc_zero_embeddings is None:
            force_uc_zero_embeddings = []
        ucg_rates = list()
        for embedder in self.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0
        c = self(batch_c)
        uc = self(batch_c if batch_uc is None else batch_uc, force_uc_zero_embeddings)
        for embedder, rate in zip(self.embedders, ucg_rates):
            embedder.ucg_rate = rate
        return c, uc


class GeneralConditionerWithControl(GeneralConditioner):

    def forward(self, batch: 'Dict', force_zero_embeddings: 'Optional[List]'=None) ->Dict:
        output = dict()
        if force_zero_embeddings is None:
            force_zero_embeddings = []
        for embedder in self.embedders:
            embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
            with embedding_context():
                if hasattr(embedder, 'input_key') and embedder.input_key is not None:
                    if embedder.legacy_ucg_val is not None:
                        batch = self.possibly_get_ucg_val(embedder, batch)
                    emb_out = embedder(batch[embedder.input_key])
                elif hasattr(embedder, 'input_keys'):
                    emb_out = embedder(*[batch[k] for k in embedder.input_keys])
            assert isinstance(emb_out, (torch.Tensor, list, tuple)), f'encoder outputs must be tensors or a sequence, but got {type(emb_out)}'
            if not isinstance(emb_out, (list, tuple)):
                emb_out = [emb_out]
            for emb in emb_out:
                if 'control_vector' in embedder.input_key:
                    out_key = 'control_vector'
                else:
                    out_key = self.OUTPUT_DIM2KEYS[emb.dim()]
                if embedder.ucg_rate > 0.0 and embedder.legacy_ucg_val is None:
                    emb = expand_dims_like(torch.bernoulli((1.0 - embedder.ucg_rate) * torch.ones(emb.shape[0], device=emb.device)), emb) * emb
                if hasattr(embedder, 'input_key') and embedder.input_key in force_zero_embeddings:
                    emb = torch.zeros_like(emb)
                if out_key in output:
                    output[out_key] = torch.cat((output[out_key], emb), self.KEY2CATDIM[out_key])
                else:
                    output[out_key] = emb
        output['control'] = batch['control']
        return output


class PreparedConditioner(nn.Module):

    def __init__(self, cond_pth, un_cond_pth=None):
        super().__init__()
        conditions = torch.load(cond_pth)
        for k, v in conditions.items():
            self.register_buffer(k, v)
        self.un_cond_pth = un_cond_pth
        if un_cond_pth is not None:
            un_conditions = torch.load(un_cond_pth)
            for k, v in un_conditions.items():
                self.register_buffer(k + '_uc', v)

    @torch.no_grad()
    def forward(self, batch: 'Dict', return_uc=False) ->Dict:
        output = dict()
        for k, v in self.state_dict().items():
            if not return_uc:
                if k.endswith('_uc'):
                    continue
                else:
                    output[k] = v.detach().clone().repeat(batch['control'].shape[0], *[(1) for _ in range(v.ndim - 1)])
            elif k.endswith('_uc'):
                output[k[:-3]] = v.detach().clone().repeat(batch['control'].shape[0], *[(1) for _ in range(v.ndim - 1)])
            else:
                continue
        output['control'] = batch['control']
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                assert torch.isnan(v).any() is not None
        return output

    def get_unconditional_conditioning(self, batch_c, batch_uc=None, force_uc_zero_embeddings=None):
        c = self(batch_c)
        if self.un_cond_pth is not None:
            uc = self(batch_c, return_uc=True)
        else:
            uc = None
        return c, uc


class InceptionV3(nn.Module):
    """Wrapper around the https://github.com/mseitzer/pytorch-fid inception
    port with an additional squeeze at the end"""

    def __init__(self, normalize_input=False, **kwargs):
        super().__init__()
        kwargs['resize_input'] = True
        self.model = inception.InceptionV3(normalize_input=normalize_input, **kwargs)

    def forward(self, inp):
        outp = self.model(inp)
        if len(outp) == 1:
            return outp[0].squeeze()
        return outp


class IdentityEncoder(AbstractEmbModel):

    def encode(self, x):
        return x

    def forward(self, x):
        return x


class ClassEmbedder(AbstractEmbModel):

    def __init__(self, embed_dim, n_classes=1000, add_sequence_dim=False):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.add_sequence_dim = add_sequence_dim

    def forward(self, c):
        c = self.embedding(c)
        if self.add_sequence_dim:
            c = c[:, None, :]
        return c

    def get_unconditional_conditioning(self, bs, device='cuda'):
        uc_class = self.n_classes - 1
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc.long()}
        return uc


class ClassEmbedderForMultiCond(ClassEmbedder):

    def forward(self, batch, key=None, disable_dropout=False):
        out = batch
        key = default(key, self.key)
        islist = isinstance(batch[key], list)
        if islist:
            batch[key] = batch[key][0]
        c_out = super().forward(batch, key, disable_dropout)
        out[key] = [c_out] if islist else c_out
        return out


class FrozenT5Embedder(AbstractEmbModel):
    """Uses the T5 transformer encoder for text"""

    def __init__(self, version='google/t5-v1_1-xxl', device='cuda', max_length=77, freeze=True):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True, return_overflowing_tokens=False, padding='max_length', return_tensors='pt')
        tokens = batch_encoding['input_ids']
        with torch.autocast('cuda', enabled=False):
            outputs = self.transformer(input_ids=tokens)
        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenByT5Embedder(AbstractEmbModel):
    """
    Uses the ByT5 transformer encoder for text. Is character-aware.
    """

    def __init__(self, version='google/byt5-base', device='cuda', max_length=77, freeze=True):
        super().__init__()
        self.tokenizer = ByT5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True, return_overflowing_tokens=False, padding='max_length', return_tensors='pt')
        tokens = batch_encoding['input_ids']
        with torch.autocast('cuda', enabled=False):
            outputs = self.transformer(input_ids=tokens)
        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


SDXL_CLIP1_PATH = '/opt/data/private/AIGC_pretrain/clip-vit-large-patch14'


def autocast(f, enabled=True):

    def do_autocast(*args, **kwargs):
        with torch.amp.autocast(enabled=enabled, dtype=torch.get_autocast_gpu_dtype(), cache_enabled=torch.is_autocast_cache_enabled()):
            return f(*args, **kwargs)
    return do_autocast


class FrozenCLIPEmbedder(AbstractEmbModel):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = ['last', 'pooled', 'hidden']

    def __init__(self, version='openai/clip-vit-large-patch14', device='cuda', max_length=77, freeze=True, layer='last', layer_idx=None, always_return_pooled=False):
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version if SDXL_CLIP1_PATH is None else SDXL_CLIP1_PATH)
        self.transformer = CLIPTextModel.from_pretrained(version if SDXL_CLIP1_PATH is None else SDXL_CLIP1_PATH)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        self.return_pooled = always_return_pooled
        if layer == 'hidden':
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    @autocast
    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True, return_overflowing_tokens=False, padding='max_length', return_tensors='pt')
        tokens = batch_encoding['input_ids']
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer == 'hidden')
        if self.layer == 'last':
            z = outputs.last_hidden_state
        elif self.layer == 'pooled':
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        if self.return_pooled:
            return z, outputs.pooler_output
        return z

    def encode(self, text):
        return self(text)


SDXL_CLIP2_CKPT_PTH = '/opt/data/private/AIGC_pretrain/CLIP-ViT-bigG-14-laion2B-39B-b160k/open_clip_pytorch_model.bin'


class FrozenOpenCLIPEmbedder2(AbstractEmbModel):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = ['pooled', 'last', 'penultimate']

    def __init__(self, arch='ViT-H-14', version='laion2b_s32b_b79k', device='cuda', max_length=77, freeze=True, layer='last', always_return_pooled=False, legacy=True):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version if SDXL_CLIP2_CKPT_PTH is None else SDXL_CLIP2_CKPT_PTH)
        del model.visual
        self.model = model
        self.device = device
        self.max_length = max_length
        self.return_pooled = always_return_pooled
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == 'last':
            self.layer_idx = 0
        elif self.layer == 'penultimate':
            self.layer_idx = 1
        else:
            raise NotImplementedError()
        self.legacy = legacy

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    @autocast
    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens)
        if not self.return_pooled and self.legacy:
            return z
        if self.return_pooled:
            assert not self.legacy
            return z[self.layer], z['pooled']
        return z[self.layer]

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        if self.legacy:
            x = x[self.layer]
            x = self.model.ln_final(x)
            return x
        else:
            o = x['last']
            o = self.model.ln_final(o)
            pooled = self.pool(o, text)
            x['pooled'] = pooled
            return x

    def pool(self, x, text):
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection
        return x

    def text_transformer_forward(self, x: 'torch.Tensor', attn_mask=None):
        outputs = {}
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - 1:
                outputs['penultimate'] = x.permute(1, 0, 2)
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        outputs['last'] = x.permute(1, 0, 2)
        return outputs

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder(AbstractEmbModel):
    LAYERS = ['last', 'penultimate']

    def __init__(self, arch='ViT-H-14', version='laion2b_s32b_b79k', device='cuda', max_length=77, freeze=True, layer='last'):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == 'last':
            self.layer_idx = 0
        elif self.layer == 'penultimate':
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens)
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: 'torch.Tensor', attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPImageEmbedder(AbstractEmbModel):
    """
    Uses the OpenCLIP vision transformer encoder for images
    """

    def __init__(self, arch='ViT-H-14', version='laion2b_s32b_b79k', device='cuda', max_length=77, freeze=True, antialias=True, ucg_rate=0.0, unsqueeze_dim=False, repeat_to_max_len=False, num_image_crops=0, output_tokens=False):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.transformer
        self.model = model
        self.max_crops = num_image_crops
        self.pad_to_max_len = self.max_crops > 0
        self.repeat_to_max_len = repeat_to_max_len and not self.pad_to_max_len
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.antialias = antialias
        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)
        self.ucg_rate = ucg_rate
        self.unsqueeze_dim = unsqueeze_dim
        self.stored_batch = None
        self.model.visual.output_tokens = output_tokens
        self.output_tokens = output_tokens

    def preprocess(self, x):
        x = kornia.geometry.resize(x, (224, 224), interpolation='bicubic', align_corners=True, antialias=self.antialias)
        x = (x + 1.0) / 2.0
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    @autocast
    def forward(self, image, no_dropout=False):
        z = self.encode_with_vision_transformer(image)
        tokens = None
        if self.output_tokens:
            z, tokens = z[0], z[1]
        z = z
        if self.ucg_rate > 0.0 and not no_dropout and not self.max_crops > 0:
            z = torch.bernoulli((1.0 - self.ucg_rate) * torch.ones(z.shape[0], device=z.device))[:, None] * z
            if tokens is not None:
                tokens = expand_dims_like(torch.bernoulli((1.0 - self.ucg_rate) * torch.ones(tokens.shape[0], device=tokens.device)), tokens) * tokens
        if self.unsqueeze_dim:
            z = z[:, None, :]
        if self.output_tokens:
            assert not self.repeat_to_max_len
            assert not self.pad_to_max_len
            return tokens, z
        if self.repeat_to_max_len:
            if z.dim() == 2:
                z_ = z[:, None, :]
            else:
                z_ = z
            return repeat(z_, 'b 1 d -> b n d', n=self.max_length), z
        elif self.pad_to_max_len:
            assert z.dim() == 3
            z_pad = torch.cat((z, torch.zeros(z.shape[0], self.max_length - z.shape[1], z.shape[2], device=z.device)), 1)
            return z_pad, z_pad[:, 0, ...]
        return z

    def encode_with_vision_transformer(self, img):
        if img.dim() == 5:
            assert self.max_crops == img.shape[1]
            img = rearrange(img, 'b n c h w -> (b n) c h w')
        img = self.preprocess(img)
        if not self.output_tokens:
            assert not self.model.visual.output_tokens
            x = self.model.visual(img)
            tokens = None
        else:
            assert self.model.visual.output_tokens
            x, tokens = self.model.visual(img)
        if self.max_crops > 0:
            x = rearrange(x, '(b n) d -> b n d', n=self.max_crops)
            x = torch.bernoulli((1.0 - self.ucg_rate) * torch.ones(x.shape[0], x.shape[1], 1, device=x.device)) * x
            if tokens is not None:
                tokens = rearrange(tokens, '(b n) t d -> b t (n d)', n=self.max_crops)
                None
        if self.output_tokens:
            return x, tokens
        return x

    def encode(self, text):
        return self(text)


class FrozenCLIPT5Encoder(AbstractEmbModel):

    def __init__(self, clip_version='openai/clip-vit-large-patch14', t5_version='google/t5-v1_1-xl', device='cuda', clip_max_length=77, t5_max_length=77):
        super().__init__()
        self.clip_encoder = FrozenCLIPEmbedder(clip_version, device, max_length=clip_max_length)
        self.t5_encoder = FrozenT5Embedder(t5_version, device, max_length=t5_max_length)
        None

    def encode(self, text):
        return self(text)

    def forward(self, text):
        clip_z = self.clip_encoder.encode(text)
        t5_z = self.t5_encoder.encode(text)
        return [clip_z, t5_z]


class SpatialRescaler(nn.Module):

    def __init__(self, n_stages=1, method='bilinear', multiplier=0.5, in_channels=3, out_channels=None, bias=False, wrap_video=False, kernel_size=1, remap_output=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest', 'linear', 'bilinear', 'trilinear', 'bicubic', 'area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None or remap_output
        if self.remap_output:
            None
            self.channel_mapper = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, padding=kernel_size // 2)
        self.wrap_video = wrap_video

    def forward(self, x):
        if self.wrap_video and x.ndim == 5:
            B, C, T, H, W = x.shape
            x = rearrange(x, 'b c t h w -> b t c h w')
            x = rearrange(x, 'b t c h w -> (b t) c h w')
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)
        if self.wrap_video:
            x = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T, c=C)
            x = rearrange(x, 'b t c h w -> b c t h w')
        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def make_beta_schedule(schedule, n_timestep, linear_start=0.0001, linear_end=0.02):
    if schedule == 'linear':
        betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    return betas.numpy()


class LowScaleEncoder(nn.Module):

    def __init__(self, model_config, linear_start, linear_end, timesteps=1000, max_noise_level=250, output_size=64, scale_factor=1.0):
        super().__init__()
        self.max_noise_level = max_noise_level
        self.model = instantiate_from_config(model_config)
        self.augmentation_schedule = self.register_schedule(timesteps=timesteps, linear_start=linear_start, linear_end=linear_end)
        self.out_size = output_size
        self.scale_factor = scale_factor

    def register_schedule(self, beta_schedule='linear', timesteps=1000, linear_start=0.0001, linear_end=0.02, cosine_s=0.008):
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda : torch.randn_like(x_start))
        return extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def forward(self, x):
        z = self.model.encode(x)
        if isinstance(z, DiagonalGaussianDistribution):
            z = z.sample()
        z = z * self.scale_factor
        noise_level = torch.randint(0, self.max_noise_level, (x.shape[0],), device=x.device).long()
        z = self.q_sample(z, noise_level)
        if self.out_size is not None:
            z = torch.nn.functional.interpolate(z, size=self.out_size, mode='nearest')
        return z, noise_level

    def decode(self, z):
        z = z / self.scale_factor
        return self.model.decode(z)


class ConcatTimestepEmbedderND(AbstractEmbModel):
    """embeds each dimension independently and concatenates them"""

    def __init__(self, outdim):
        super().__init__()
        self.timestep = Timestep(outdim)
        self.outdim = outdim

    def forward(self, x):
        if x.ndim == 1:
            x = x[:, None]
        assert len(x.shape) == 2
        b, dims = x.shape[0], x.shape[1]
        x = rearrange(x, 'b d -> (b d)')
        emb = self.timestep(x)
        emb = rearrange(emb, '(b d) d2 -> b (d d2)', b=b, d=dims, d2=self.outdim)
        return emb


class GaussianEncoder(Encoder, AbstractEmbModel):

    def __init__(self, weight: 'float'=1.0, flatten_output: 'bool'=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.posterior = DiagonalGaussianRegularizer()
        self.weight = weight
        self.flatten_output = flatten_output

    def forward(self, x) ->Tuple[Dict, torch.Tensor]:
        z = super().forward(x)
        z, log = self.posterior(z)
        log['loss'] = log['kl_loss']
        log['weight'] = self.weight
        if self.flatten_output:
            z = rearrange(z, 'b c h w -> b (h w ) c')
        return log, z


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ActNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AttentionPool2d,
     lambda: ([], {'spacial_dim': 4, 'embed_dim': 4, 'num_heads_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DiagonalGaussianRegularizer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Downsample,
     lambda: ([], {'channels': 4, 'use_conv': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FeedForward,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GEGLU,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GroupNorm32,
     lambda: ([], {'num_groups': 1, 'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (IdentityEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (IdentityMap,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LPLayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LPRMSNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MPTMLP,
     lambda: ([], {'d_model': 4, 'expansion_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NLayerDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (RMSNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ScalingLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {})),
    (SiLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SimpleResBlock,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SpatialRescaler,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TimestepBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (TimestepEmbedSequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (TransposedUpsample,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Upsample,
     lambda: ([], {'channels': 4, 'use_conv': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ZeroConv,
     lambda: ([], {'label_nc': 4, 'norm_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (vgg16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

