
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


import numpy as np


import torch


import logging


import pandas as pd


import math


from torch.utils.data.sampler import Sampler


from torch.utils.data.distributed import DistributedSampler


import torch.distributed


from typing import TypeVar


from typing import Optional


from typing import Iterator


from typing import List


import torch.distributed as dist


import torch.utils.data


from scipy.io.wavfile import read


import torch.nn as nn


from torch.distributed import init_process_group


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.utils.data import DistributedSampler


import torch.multiprocessing as mp


import torch.nn.functional as F


from torch.optim.lr_scheduler import LambdaLR


from torch.nn import functional as F


from torch.optim import AdamW


from copy import deepcopy


from functools import partial


from torchvision.utils import make_grid


from typing import Tuple


from typing import Union


from inspect import isfunction


from torch import nn


from torch import einsum


import collections.abc


from itertools import repeat


import warnings


import functools


import random


import collections


import re


from torch.utils.checkpoint import checkpoint


import copy


from torch.nn.init import _calculate_fan_in_and_fan_out


import torch.utils.checkpoint as checkpoint


import torch.distributed.nn


from torch import distributed as dist


from torch import nn as nn


from sklearn.metrics import average_precision_score


from sklearn.metrics import roc_auc_score


from sklearn.metrics import accuracy_score


from collections import OrderedDict


from typing import Callable


from functools import lru_cache


from torchvision.ops.misc import FrozenBatchNorm2d


from torch import optim


import torchvision


from scipy import ndimage


import scipy


import scipy.stats as ss


from scipy.interpolate import interp2d


from scipy.linalg import orth


from collections import namedtuple


import time


import matplotlib.pyplot as plt


from torch import sin


from torch import pow


from torch.nn import Parameter


from scipy.io.wavfile import write


from torch.nn import Conv1d


from torch.nn import ConvTranspose1d


from torch.nn import Conv2d


from torch.nn.utils import weight_norm


from torch.nn.utils import remove_weight_norm


from torch.nn.utils import spectral_norm


import itertools


from torch.utils.tensorboard import SummaryWriter


from torch.nn.parallel import DistributedDataParallel


import matplotlib


import matplotlib.pylab as plt


from scipy import linalg


def dynamic_range_compression_torch(x, C=1, clip_val=1e-05):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


class MelNet(nn.Module):

    def __init__(self, hparams, device='cpu') ->None:
        super().__init__()
        self.n_fft = hparams['fft_size']
        self.num_mels = hparams['audio_num_mel_bins']
        self.sampling_rate = hparams['audio_sample_rate']
        self.hop_size = hparams['hop_size']
        self.win_size = hparams['win_size']
        self.fmin = hparams['fmin']
        self.fmax = hparams['fmax']
        self.device = device
        mel = librosa_mel_fn(self.sampling_rate, self.n_fft, self.num_mels, self.fmin, self.fmax)
        self.mel_basis = torch.from_numpy(mel).float()
        self.hann_window = torch.hann_window(self.win_size)

    def to(self, device, **kwagrs):
        super()
        self.mel_basis = self.mel_basis
        self.hann_window = self.hann_window
        self.device = device

    def forward(self, y, center=False, complex=False):
        if isinstance(y, np.ndarray):
            y = torch.FloatTensor(y)
            if len(y.shape) == 1:
                y = y.unsqueeze(0)
        y = y.clamp(min=-1.0, max=1.0)
        y = torch.nn.functional.pad(y.unsqueeze(1), [int((self.n_fft - self.hop_size) / 2), int((self.n_fft - self.hop_size) / 2)], mode='reflect')
        y = y.squeeze(1)
        spec = torch.stft(y, self.n_fft, hop_length=self.hop_size, win_length=self.win_size, window=self.hann_window, center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=complex)
        if not complex:
            spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-09)
            spec = torch.matmul(self.mel_basis, spec)
            spec = spectral_normalize_torch(spec)
        else:
            B, C, T, _ = spec.shape
            spec = spec.transpose(1, 2)
        return spec


class IdentityFirstStage(torch.nn.Module):

    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-06, affine=True)


def nonlinearity(x):
    return x * torch.sigmoid(x)


class ResnetBlock1D(nn.Module):

    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

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


class AttnBlock1D(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.k = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.v = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.proj_out = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, t, c = q.shape
        q = q.permute(0, 2, 1)
        w_ = torch.bmm(q, k)
        w_ = w_ * int(t) ** -0.5
        w_ = torch.nn.functional.softmax(w_, dim=2)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = self.proj_out(h_)
        return x + h_


class Upsample1D(nn.Module):

    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample1D(nn.Module):

    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = 0, 1
            x = torch.nn.functional.pad(x, pad, mode='constant', value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool1d(x, kernel_size=2, stride=2)
        return x


class Encoder1D(nn.Module):

    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_layers=[], down_layers=[], dropout=0.0, resamp_with_conv=True, in_channels, z_channels, double_z=True, kernel_size=3, **ignore_kwargs):
        """ out_ch is only used in decoder,not used here
        """
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_layers = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        None
        self.down_layers = down_layers
        self.attn_layers = attn_layers
        self.conv_in = torch.nn.Conv1d(in_channels, self.ch, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_layers):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock1D(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout, kernel_size=kernel_size))
                block_in = block_out
                if i_level in attn_layers:
                    attn.append(AttnBlock1D(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level in down_layers:
                down.downsample = Downsample1D(block_in, resamp_with_conv)
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout, kernel_size=kernel_size)
        self.mid.attn_1 = AttnBlock1D(block_in)
        self.mid.block_2 = ResnetBlock1D(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout, kernel_size=kernel_size)
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in, 2 * z_channels if double_z else z_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        temb = None
        hs = [self.conv_in(x)]
        for i_level in range(self.num_layers):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level in self.down_layers:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder1D(nn.Module):

    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_layers=[], down_layers=[], dropout=0.0, kernel_size=3, resamp_with_conv=True, in_channels, z_channels, give_pre_end=False, tanh_out=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_layers = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.down_layers = [(i + 1) for i in down_layers]
        None
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_layers - 1]
        self.conv_in = torch.nn.Conv1d(z_channels, block_in, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock1D(block_in)
        self.mid.block_2 = ResnetBlock1D(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_layers)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock1D(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if i_level in attn_layers:
                    attn.append(AttnBlock1D(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level in self.down_layers:
                up.upsample = Upsample1D(block_in, resamp_with_conv)
            self.up.insert(0, up)
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in, out_ch, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, z):
        self.last_z_shape = z.shape
        temb = None
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        for i_level in reversed(range(self.num_layers)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level in self.down_layers:
                h = self.up[i_level].upsample(h)
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class GEGLU(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class FeedForward(nn.Module):

    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)
        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


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


class CrossAttention(nn.Module):

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class Conv1dGEGLU(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size=9):
        super().__init__()
        self.proj = nn.Conv1d(dim_in, dim_out * 2, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=1)
        return x * F.gelu(gate)


class Conv1dFeedForward(nn.Module):

    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0, kernel_size=9):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(nn.Conv1d(dim, inner_dim, kernel_size=kernel_size, padding=kernel_size // 2), nn.GELU()) if not glu else Conv1dGEGLU(dim, inner_dim)
        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Conv1d(inner_dim, dim_out, kernel_size=kernel_size, padding=kernel_size // 2))

    def forward(self, x):
        return self.net(x)


class BasicTransformerBlock(nn.Module):

    def __init__(self, dim, n_heads, d_head, dropout=0.0, context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.ff = Conv1dFeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x).permute(0, 2, 1)).permute(0, 2, 1) + x
        return x


def zero_module(module):
    """
    Zero out the parameters of a module and return it.zero-initializing the final convolutional layer in each block prior to any residual connections can accelerate training. 
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0.0, context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim) for d in range(depth)])
        self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        x_out = x + x_in
        return x_out


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size, bias=True), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class Conv1DFinalLayer(nn.Module):
    """
    The final layer of CrossAttnDiT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.GroupNorm(16, hidden_size)
        self.conv1d = nn.Conv1d(hidden_size, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.conv1d(x)
        return x


class ConditionEmbedder(nn.Module):

    def __init__(self, hidden_size, context_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(context_dim, hidden_size, bias=True), nn.GELU(approximate='tanh'), nn.Linear(hidden_size, hidden_size, bias=True), nn.LayerNorm(hidden_size))

    def forward(self, x):
        return self.mlp(x)


class TemporalTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0.0, context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim) for d in range(depth)])
        self.proj_out = zero_module(nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, x, context=None):
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c t -> b t c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b t c -> b c t')
        x = self.proj_out(x)
        return x + x_in


class PositionEmbedding(nn.Module):
    MODE_EXPAND = 'MODE_EXPAND'
    MODE_ADD = 'MODE_ADD'
    MODE_CONCAT = 'MODE_CONCAT'

    def __init__(self, num_embeddings, embedding_dim, mode=MODE_ADD):
        super(PositionEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        if self.mode == self.MODE_EXPAND:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings * 2 + 1, embedding_dim))
        else:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        if self.mode == self.MODE_EXPAND:
            indices = torch.clamp(x, -self.num_embeddings, self.num_embeddings) + self.num_embeddings
            return F.embedding(indices.type(torch.LongTensor), self.weight)
        batch_size, seq_len = x.size()[:2]
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.embedding_dim)
        if self.mode == self.MODE_ADD:
            return x + embeddings
        if self.mode == self.MODE_CONCAT:
            return torch.cat((x, embeddings.repeat(batch_size, 1, 1)), dim=-1)
        raise NotImplementedError('Unknown mode: %s' % self.mode)

    def extra_repr(self):
        return 'num_embeddings={}, embedding_dim={}, mode={}'.format(self.num_embeddings, self.embedding_dim, self.mode)


class ConcatDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, in_channels, context_dim, hidden_size=1152, depth=28, num_heads=16, max_len=1000):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        kernel_size = 5
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.c_embedder = ConditionEmbedder(hidden_size, context_dim)
        self.proj_in = nn.Conv1d(in_channels, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pos_emb = PositionEmbedding(num_embeddings=max_len, embedding_dim=hidden_size)
        self.blocks = nn.ModuleList([TemporalTransformer(hidden_size, num_heads, d_head=hidden_size // num_heads, depth=1, context_dim=context_dim) for _ in range(depth)])
        self.final_layer = Conv1DFinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def forward(self, x, t, context, w_cond=None):
        """
        Forward pass of DiT.
        x: (N, C, T) tensor of temporal inputs (latent representations of melspec)
        t: (N,) tensor of diffusion timesteps
        y: (N,max_tokens_len=77, context_dim)
        """
        t = self.t_embedder(t, w_cond=w_cond).unsqueeze(1)
        c = self.c_embedder(context)
        extra_len = c.shape[1] + 1
        x = self.proj_in(x)
        x = rearrange(x, 'b c t -> b t c')
        x = torch.concat([t, c, x], dim=1)
        x = self.pos_emb(x)
        x = rearrange(x, 'b t c -> b c t')
        for block in self.blocks:
            x = block(x)
        x = x[..., extra_len:]
        x = self.final_layer(x)
        return x


class ConcatDiT2MLP(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, in_channels, context_dim, hidden_size=1152, depth=28, num_heads=16, max_len=1000):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        kernel_size = 5
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.c1_embedder = ConditionEmbedder(hidden_size, context_dim)
        self.c2_embedder = ConditionEmbedder(hidden_size, context_dim)
        self.proj_in = nn.Conv1d(in_channels, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pos_emb = PositionEmbedding(num_embeddings=max_len, embedding_dim=hidden_size)
        self.blocks = nn.ModuleList([TemporalTransformer(hidden_size, num_heads, d_head=hidden_size // num_heads, depth=1, context_dim=context_dim) for _ in range(depth)])
        self.final_layer = Conv1DFinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def forward(self, x, t, context, w_cond=None):
        """
        Forward pass of DiT.
        x: (N, C, T) tensor of temporal inputs (latent representations of melspec)
        t: (N,) tensor of diffusion timesteps
        y: (N,max_tokens_len=77, context_dim)
        """
        t = self.t_embedder(t, w_cond=w_cond).unsqueeze(1)
        c1, c2 = context.chunk(2, dim=1)
        c1 = self.c1_embedder(c1)
        c2 = self.c2_embedder(c2)
        c = torch.cat((c1, c2), dim=1)
        extra_len = c.shape[1] + 1
        x = self.proj_in(x)
        x = rearrange(x, 'b c t -> b t c')
        x = torch.concat([t, c, x], dim=1)
        x = self.pos_emb(x)
        x = rearrange(x, 'b t c -> b c t')
        for block in self.blocks:
            x = block(x)
        x = x[..., extra_len:]
        x = self.final_layer(x)
        return x


class ConcatOrderDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, in_channels, context_dim, hidden_size=1152, depth=28, num_heads=16, max_len=1000):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        kernel_size = 5
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.c_embedder = ConditionEmbedder(hidden_size, context_dim)
        self.proj_in = nn.Conv1d(in_channels, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pos_emb = PositionEmbedding(num_embeddings=max_len, embedding_dim=hidden_size)
        self.order_embedding = nn.Embedding(num_embeddings=100, embedding_dim=hidden_size)
        self.blocks = nn.ModuleList([TemporalTransformer(hidden_size, num_heads, d_head=hidden_size // num_heads, depth=1, context_dim=context_dim) for _ in range(depth)])
        self.final_layer = Conv1DFinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def add_order_embedding(self, token_emb, token_ids, orders_list):
        """
        token_emb: shape (N,max_tokens_len=77, hidden_size)
        token_ids: shape (N,max_tokens)
        order_list: [N*list]. len(order_list[i]) == objs_num in text[i] 
        """
        for b, orderl in enumerate(orders_list):
            orderl = torch.LongTensor(orderl)
            order_emb = self.order_embedding(orderl)
            obj2index = []
            cur_obj = 0
            for i in range(token_ids.shape[1]):
                token_id = token_ids[b][i]
                if token_id in [101, 102, 0, 1064]:
                    obj2index.append(-1)
                    if token_id == 1064:
                        cur_obj += 1
                else:
                    obj2index.append(cur_obj)
            for i, order_index in enumerate(obj2index):
                if order_index != -1:
                    token_emb[b][i] += order_emb[order_index]
        return token_emb

    def forward(self, x, t, context):
        """
        Forward pass of DiT.
        x: (N, C, T) tensor of temporal inputs (latent representations of melspec)
        t: (N,) tensor of diffusion timesteps
        context: dict{'token_embedding':(N,max_tokens_len=77, context_dim),'token_ids':tokens:(N,max_tokens_len=77),'orders':orders_list}
        """
        token_embedding = context['token_embedding']
        token_ids = context['token_ids']
        orders = context['orders']
        t = self.t_embedder(t).unsqueeze(1)
        c = self.c_embedder(token_embedding)
        c = self.add_order_embedding(c, token_ids, orders)
        extra_len = c.shape[1] + 1
        x = self.proj_in(x)
        x = rearrange(x, 'b c t -> b t c')
        x = torch.concat([t, c, x], dim=1)
        x = self.pos_emb(x)
        x = rearrange(x, 'b t c -> b c t')
        for block in self.blocks:
            x = block(x)
        x = x[..., extra_len:]
        x = self.final_layer(x)
        return x


class ConcatOrderDiT2(nn.Module):
    """
    Diffusion model with a Transformer backbone. concat by token
    """

    def __init__(self, in_channels, context_dim, hidden_size=1152, depth=28, num_heads=16, max_len=1000):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        kernel_size = 5
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.c_embedder = ConditionEmbedder(hidden_size, context_dim)
        self.proj_in = nn.Conv1d(in_channels, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pos_emb = PositionEmbedding(num_embeddings=max_len, embedding_dim=hidden_size)
        self.max_objs = 10
        self.max_objs_order = 100
        self.order_embedding = nn.Embedding(num_embeddings=self.max_objs_order + 1, embedding_dim=hidden_size)
        self.blocks = nn.ModuleList([TemporalTransformer(hidden_size, num_heads, d_head=hidden_size // num_heads, depth=1, context_dim=context_dim) for _ in range(depth)])
        self.final_layer = Conv1DFinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def concat_order_embedding(self, token_emb, token_ids, orders_list):
        """
        token_emb: shape (N,max_tokens_len=77, hidden_size)
        token_ids: shape (N,max_tokens)
        order_list: [N*list]. len(order_list[i]) == objs_num in text[i] 
        return token_emb: shape (N,max_tokens_len+self.max_objs, hidden_size)
        """
        bsz, t, c = token_emb.shape
        token_emb = list(torch.tensor_split(token_emb, bsz))
        orders_list = deepcopy(orders_list)
        for i in range(bsz):
            token_emb[i] = list(torch.tensor_split(token_emb[i].squeeze(0), t))
        for b, orderl in enumerate(orders_list):
            orderl.append(self.max_objs_order)
            orderl = torch.LongTensor(orderl)
            order_emb = self.order_embedding(orderl)
            order_emb = torch.tensor_split(order_emb, len(orderl))
            obj_insert_index = []
            for i in range(token_ids.shape[1]):
                token_id = token_ids[b][i]
                if token_id == 1064:
                    obj_insert_index.append(i + len(obj_insert_index))
            for i, index in enumerate(obj_insert_index):
                token_emb[b].insert(index, order_emb[i])
            for i in range(self.max_objs - len(orderl) + 1):
                token_emb[b].append(order_emb[-1])
            token_emb[b] = torch.concat(token_emb[b])
        token_emb = torch.stack(token_emb)
        return token_emb

    def forward(self, x, t, context):
        """
        Forward pass of DiT.
        x: (N, C, T) tensor of temporal inputs (latent representations of melspec)
        t: (N,) tensor of diffusion timesteps
        context: dict{'token_embedding':(N,max_tokens_len=77, context_dim),'token_ids':tokens:(N,max_tokens_len=77),'orders':orders_list}
        """
        token_embedding = context['token_embedding']
        token_ids = context['token_ids']
        orders = context['orders']
        t = self.t_embedder(t).unsqueeze(1)
        c = self.c_embedder(token_embedding)
        c = self.concat_order_embedding(c, token_ids, orders)
        extra_len = c.shape[1] + 1
        x = self.proj_in(x)
        x = rearrange(x, 'b c t -> b t c')
        x = torch.concat([t, c, x], dim=1)
        x = self.pos_emb(x)
        x = rearrange(x, 'b t c -> b c t')
        for block in self.blocks:
            x = block(x)
        x = x[..., extra_len:]
        x = self.final_layer(x)
        return x


DEFAULT_DIM_HEAD = 64


Intermediates = namedtuple('Intermediates', ['pre_softmax_attn', 'post_softmax_attn'])


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


class Attention(nn.Module):

    def __init__(self, dim, dim_head=DEFAULT_DIM_HEAD, heads=8, causal=False, mask=None, talking_heads=False, sparse_topk=None, use_entmax15=False, num_mem_kv=0, dropout=0.0, on_attn=False):
        super().__init__()
        if use_entmax15:
            raise NotImplementedError('Check out entmax activation instead of softmax activation!')
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.causal = causal
        self.mask = mask
        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.talking_heads = talking_heads
        if talking_heads:
            self.pre_softmax_proj = nn.Parameter(torch.randn(heads, heads))
            self.post_softmax_proj = nn.Parameter(torch.randn(heads, heads))
        self.sparse_topk = sparse_topk
        self.attn_fn = F.softmax
        self.num_mem_kv = num_mem_kv
        if num_mem_kv > 0:
            self.mem_k = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))
            self.mem_v = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))
        self.attn_on_attn = on_attn
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim * 2), nn.GLU()) if on_attn else nn.Linear(inner_dim, dim)

    def forward(self, x, context=None, mask=None, context_mask=None, rel_pos=None, sinusoidal_emb=None, prev_attn=None, mem=None):
        b, n, _, h, talking_heads, device = *x.shape, self.heads, self.talking_heads, x.device
        kv_input = default(context, x)
        q_input = x
        k_input = kv_input
        v_input = kv_input
        if exists(mem):
            k_input = torch.cat((mem, k_input), dim=-2)
            v_input = torch.cat((mem, v_input), dim=-2)
        if exists(sinusoidal_emb):
            offset = k_input.shape[-2] - q_input.shape[-2]
            q_input = q_input + sinusoidal_emb(q_input, offset=offset)
            k_input = k_input + sinusoidal_emb(k_input)
        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        input_mask = None
        if any(map(exists, (mask, context_mask))):
            q_mask = default(mask, lambda : torch.ones((b, n), device=device).bool())
            k_mask = q_mask if not exists(context) else context_mask
            k_mask = default(k_mask, lambda : torch.ones((b, k.shape[-2]), device=device).bool())
            q_mask = rearrange(q_mask, 'b i -> b () i ()')
            k_mask = rearrange(k_mask, 'b j -> b () () j')
            input_mask = q_mask * k_mask
        if self.num_mem_kv > 0:
            mem_k, mem_v = map(lambda t: repeat(t, 'h n d -> b h n d', b=b), (self.mem_k, self.mem_v))
            k = torch.cat((mem_k, k), dim=-2)
            v = torch.cat((mem_v, v), dim=-2)
            if exists(input_mask):
                input_mask = F.pad(input_mask, (self.num_mem_kv, 0), value=True)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = max_neg_value(dots)
        if exists(prev_attn):
            dots = dots + prev_attn
        pre_softmax_attn = dots
        if talking_heads:
            dots = einsum('b h i j, h k -> b k i j', dots, self.pre_softmax_proj).contiguous()
        if exists(rel_pos):
            dots = rel_pos(dots)
        if exists(input_mask):
            dots.masked_fill_(~input_mask, mask_value)
            del input_mask
        if self.causal:
            i, j = dots.shape[-2:]
            r = torch.arange(i, device=device)
            mask = rearrange(r, 'i -> () () i ()') < rearrange(r, 'j -> () () () j')
            mask = F.pad(mask, (j - i, 0), value=False)
            dots.masked_fill_(mask, mask_value)
            del mask
        if exists(self.sparse_topk) and self.sparse_topk < dots.shape[-1]:
            top, _ = dots.topk(self.sparse_topk, dim=-1)
            vk = top[..., -1].unsqueeze(-1).expand_as(dots)
            mask = dots < vk
            dots.masked_fill_(mask, mask_value)
            del mask
        attn = self.attn_fn(dots, dim=-1)
        post_softmax_attn = attn
        attn = self.dropout(attn)
        if talking_heads:
            attn = einsum('b h i j, h k -> b k i j', attn, self.post_softmax_proj).contiguous()
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        intermediates = Intermediates(pre_softmax_attn=pre_softmax_attn, post_softmax_attn=post_softmax_attn)
        return self.to_out(out), intermediates


class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-08):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TransformerBlock(nn.Module):

    def __init__(self, layer_id: 'int', dim: 'int', n_heads: 'int', n_kv_heads: 'int', multiple_of: 'int', ffn_dim_multiplier: 'float', norm_eps: 'float', qk_norm: 'bool', y_dim: 'int') ->None:
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_heads, n_kv_heads, qk_norm, y_dim)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=4 * dim, multiple_of=multiple_of, ffn_dim_multiplier=ffn_dim_multiplier)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.attention_y_norm = RMSNorm(y_dim, eps=norm_eps)

    def forward(self, x: 'torch.Tensor', x_mask: 'torch.Tensor', y: 'torch.Tensor', y_mask: 'torch.Tensor', freqs_cis: 'torch.Tensor', adaln_input: 'Optional[torch.Tensor]'=None):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention.
                Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and
                feedforward layers.

        """
        if adaln_input is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(6, dim=1)
            h = x + gate_msa.unsqueeze(1) * self.attention(modulate(self.attention_norm(x), shift_msa, scale_msa), x_mask, freqs_cis, self.attention_y_norm(y), y_mask)
            out = h + gate_mlp.unsqueeze(1) * self.feed_forward(modulate(self.ffn_norm(h), shift_mlp, scale_mlp))
        else:
            h = x + self.attention(self.attention_norm(x), x_mask, freqs_cis, self.attention_y_norm(y), y_mask)
            out = h + self.feed_forward(self.ffn_norm(h))
        return out


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TxtFlagLargeDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, in_channels, context_dim, hidden_size=1152, depth=28, num_heads=16, max_len=1000, n_kv_heads=None, multiple_of: 'int'=256, ffn_dim_multiplier: 'Optional[float]'=None, norm_eps=1e-05, qk_norm=None, rope_scaling_factor: 'float'=1.0, ntk_factor: 'float'=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.proj_in = nn.Linear(in_channels, hidden_size, bias=True)
        self.blocks = nn.ModuleList([TransformerBlock(layer_id, hidden_size, num_heads, n_kv_heads, multiple_of, ffn_dim_multiplier, norm_eps, qk_norm, context_dim) for layer_id in range(depth)])
        self.freqs_cis = TxtFlagLargeDiT.precompute_freqs_cis(hidden_size // num_heads, max_len, rope_scaling_factor=rope_scaling_factor, ntk_factor=ntk_factor)
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.rope_scaling_factor = rope_scaling_factor
        self.ntk_factor = ntk_factor
        self.cap_embedder = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, hidden_size, bias=True))

    def forward(self, x, t, context):
        """
        Forward pass of DiT.
        x: (N, C, T) tensor of temporal inputs (latent representations of melspec)
        t: (N,) tensor of diffusion timesteps
        y: (N,max_tokens_len=77, context_dim)
        """
        self.freqs_cis = self.freqs_cis
        x = rearrange(x, 'b c t -> b t c')
        x = self.proj_in(x)
        cap_mask = torch.ones((context.shape[0], context.shape[1]), dtype=torch.int32, device=x.device)
        mask = torch.ones((x.shape[0], x.shape[1]), dtype=torch.int32, device=x.device)
        t = self.t_embedder(t)
        cap_mask_float = cap_mask.float().unsqueeze(-1)
        cap_feats_pool = (context * cap_mask_float).sum(dim=1) / cap_mask_float.sum(dim=1)
        cap_feats_pool = cap_feats_pool
        cap_emb = self.cap_embedder(cap_feats_pool)
        adaln_input = t + cap_emb
        cap_mask = cap_mask.bool()
        for block in self.blocks:
            x = block(x, mask, context, cap_mask, self.freqs_cis[:x.size(1)], adaln_input=adaln_input)
        x = self.final_layer(x, adaln_input)
        x = rearrange(x, 'b t c -> b c t')
        return x

    @staticmethod
    def precompute_freqs_cis(dim: 'int', end: 'int', theta: 'float'=10000.0, rope_scaling_factor: 'float'=1.0, ntk_factor: 'float'=1.0):
        """
        Precompute the frequency tensor for complex exponentials (cis) with
        given dimensions.

        This function calculates a frequency tensor with complex exponentials
        using the given dimension 'dim' and the end index 'end'. The 'theta'
        parameter scales the frequencies. The returned tensor contains complex
        values in complex64 data type.

        Args:
            dim (int): Dimension of the frequency tensor.
            end (int): End index for precomputing frequencies.
            theta (float, optional): Scaling factor for frequency computation.
                Defaults to 10000.0.

        Returns:
            torch.Tensor: Precomputed frequency tensor with complex
                exponentials.
        """
        theta = theta * ntk_factor
        None
        freqs = 1.0 / theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim)
        t = torch.arange(end, device=freqs.device, dtype=torch.float)
        t = t / rope_scaling_factor
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


class TxtFlagLargeImprovedDiTV2(TxtFlagLargeDiT):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, in_channels, context_dim, hidden_size=1152, depth=28, num_heads=16, max_len=1000):
        super().__init__(in_channels, context_dim, hidden_size, depth, num_heads, max_len)
        self.initialize_weights()

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        None


class Upsample(nn.Module):

    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):

    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = 0, 1, 0, 1
            x = torch.nn.functional.pad(x, pad, mode='constant', value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


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

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k)
        w_ = w_ * int(c) ** -0.5
        w_ = torch.nn.functional.softmax(w_, dim=2)
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_


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


def make_attn(in_channels, attn_type='vanilla'):
    assert attn_type in ['vanilla', 'linear', 'none'], f'attn_type {attn_type} unknown'
    None
    if attn_type == 'vanilla':
        return AttnBlock(in_channels)
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


class FixedPositionalEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / 10000 ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_dim=1, offset=0):
        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq) + offset
        sinusoid_inp = torch.einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :]


class GRUGating(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)

    def forward(self, x, residual):
        gated_output = self.gru(rearrange(x, 'b n d -> (b n) d'), rearrange(residual, 'b n d -> (b n) d'))
        return gated_output.reshape_as(x)


LayerIntermediates = namedtuple('Intermediates', ['hiddens', 'attn_intermediates'])


class Residual(nn.Module):

    def forward(self, x, residual):
        return x + residual


class Rezero(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x, **kwargs):
        x, *rest = self.fn(x, **kwargs)
        return x * self.g, *rest


class Scale(nn.Module):

    def __init__(self, value, fn):
        super().__init__()
        self.value = value
        self.fn = fn

    def forward(self, x, **kwargs):
        x, *rest = self.fn(x, **kwargs)
        return x * self.value, *rest


class ScaleNorm(nn.Module):

    def __init__(self, dim, eps=1e-05):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


def always(val):

    def inner(*args, **kwargs):
        return val
    return inner


def equals(val):

    def inner(x):
        return x == val
    return inner


def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return *return_val,


def string_begins_with(prefix, str):
    return str.startswith(prefix)


def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs


def not_equals(val):

    def inner(x):
        return x != val
    return inner


class AttentionLayers(nn.Module):

    def __init__(self, dim, depth, heads=8, causal=False, cross_attend=False, only_cross=False, use_scalenorm=False, use_rmsnorm=False, use_rezero=False, rel_pos_num_buckets=32, rel_pos_max_distance=128, position_infused_attn=False, custom_layers=None, sandwich_coef=None, par_ratio=None, residual_attn=False, cross_residual_attn=False, macaron=False, pre_norm=True, gate_residual=False, **kwargs):
        super().__init__()
        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs)
        attn_kwargs, _ = groupby_prefix_and_trim('attn_', kwargs)
        dim_head = attn_kwargs.get('dim_head', DEFAULT_DIM_HEAD)
        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList([])
        self.has_pos_emb = position_infused_attn
        self.pia_pos_emb = FixedPositionalEmbedding(dim) if position_infused_attn else None
        self.rotary_pos_emb = always(None)
        assert rel_pos_num_buckets <= rel_pos_max_distance, 'number of relative position buckets must be less than the relative position max distance'
        self.rel_pos = None
        self.pre_norm = pre_norm
        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn
        norm_class = ScaleNorm if use_scalenorm else nn.LayerNorm
        norm_class = RMSNorm if use_rmsnorm else norm_class
        norm_fn = partial(norm_class, dim)
        norm_fn = nn.Identity if use_rezero else norm_fn
        branch_fn = Rezero if use_rezero else None
        if cross_attend and not only_cross:
            default_block = 'a', 'c', 'f'
        elif cross_attend and only_cross:
            default_block = 'c', 'f'
        else:
            default_block = 'a', 'f'
        if macaron:
            default_block = ('f',) + default_block
        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(par_ratio):
            par_depth = depth * len(default_block)
            assert 1 < par_ratio <= par_depth, 'par ratio out of range'
            default_block = tuple(filter(not_equals('f'), default_block))
            par_attn = par_depth // par_ratio
            depth_cut = par_depth * 2 // 3
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert len(default_block) <= par_width, 'default block is too large for par_ratio'
            par_block = default_block + ('f',) * (par_width - len(default_block))
            par_head = par_block * par_attn
            layer_types = par_head + ('f',) * (par_depth - len(par_head))
        elif exists(sandwich_coef):
            assert sandwich_coef > 0 and sandwich_coef <= depth, 'sandwich coefficient should be less than the depth'
            layer_types = ('a',) * sandwich_coef + default_block * (depth - sandwich_coef) + ('f',) * sandwich_coef
        else:
            layer_types = default_block * depth
        self.layer_types = layer_types
        self.num_attn_layers = len(list(filter(equals('a'), layer_types)))
        for layer_type in self.layer_types:
            if layer_type == 'a':
                layer = Attention(dim, heads=heads, causal=causal, **attn_kwargs)
            elif layer_type == 'c':
                layer = Attention(dim, heads=heads, **attn_kwargs)
            elif layer_type == 'f':
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)
            else:
                raise Exception(f'invalid layer type {layer_type}')
            if isinstance(layer, Attention) and exists(branch_fn):
                layer = branch_fn(layer)
            if gate_residual:
                residual_fn = GRUGating(dim)
            else:
                residual_fn = Residual()
            self.layers.append(nn.ModuleList([norm_fn(), layer, residual_fn]))

    def forward(self, x, context=None, mask=None, context_mask=None, mems=None, return_hiddens=False):
        hiddens = []
        intermediates = []
        prev_attn = None
        prev_cross_attn = None
        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers
        for ind, (layer_type, (norm, block, residual_fn)) in enumerate(zip(self.layer_types, self.layers)):
            is_last = ind == len(self.layers) - 1
            if layer_type == 'a':
                hiddens.append(x)
                layer_mem = mems.pop(0)
            residual = x
            if self.pre_norm:
                x = norm(x)
            if layer_type == 'a':
                out, inter = block(x, mask=mask, sinusoidal_emb=self.pia_pos_emb, rel_pos=self.rel_pos, prev_attn=prev_attn, mem=layer_mem)
            elif layer_type == 'c':
                out, inter = block(x, context=context, mask=mask, context_mask=context_mask, prev_attn=prev_cross_attn)
            elif layer_type == 'f':
                out = block(x)
            x = residual_fn(out, residual)
            if layer_type in ('a', 'c'):
                intermediates.append(inter)
            if layer_type == 'a' and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == 'c' and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn
            if not self.pre_norm and not is_last:
                x = norm(x)
        if return_hiddens:
            intermediates = LayerIntermediates(hiddens=hiddens, attn_intermediates=intermediates)
            return x, intermediates
        return x


class Encoder(AttentionLayers):

    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on encoder'
        super().__init__(causal=False, **kwargs)


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
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
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

    def forward(self, z):
        self.last_z_shape = z.shape
        temb = None
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class SimpleDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.model = nn.ModuleList([nn.Conv2d(in_channels, in_channels, 1), ResnetBlock(in_channels=in_channels, out_channels=2 * in_channels, temb_channels=0, dropout=0.0), ResnetBlock(in_channels=2 * in_channels, out_channels=4 * in_channels, temb_channels=0, dropout=0.0), ResnetBlock(in_channels=4 * in_channels, out_channels=2 * in_channels, temb_channels=0, dropout=0.0), nn.Conv2d(2 * in_channels, in_channels, 1), Upsample(in_channels, with_conv=True)])
        self.norm_out = Normalize(in_channels)
        self.conv_out = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i in [1, 2, 3]:
                x = layer(x, None)
            else:
                x = layer(x)
        h = self.norm_out(x)
        h = nonlinearity(h)
        x = self.conv_out(h)
        return x


class UpsampleDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, ch, num_res_blocks, resolution, ch_mult=(2, 2), dropout=0.0):
        super().__init__()
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = in_channels
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.res_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            res_block = []
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
            self.res_blocks.append(nn.ModuleList(res_block))
            if i_level != self.num_resolutions - 1:
                self.upsample_blocks.append(Upsample(block_in, True))
                curr_res = curr_res * 2
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = x
        for k, i_level in enumerate(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.res_blocks[i_level][i_block](h, None)
            if i_level != self.num_resolutions - 1:
                h = self.upsample_blocks[k](h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class LatentRescaler(nn.Module):

    def __init__(self, factor, in_channels, mid_channels, out_channels, depth=2):
        super().__init__()
        self.factor = factor
        self.conv_in = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.res_block1 = nn.ModuleList([ResnetBlock(in_channels=mid_channels, out_channels=mid_channels, temb_channels=0, dropout=0.0) for _ in range(depth)])
        self.attn = AttnBlock(mid_channels)
        self.res_block2 = nn.ModuleList([ResnetBlock(in_channels=mid_channels, out_channels=mid_channels, temb_channels=0, dropout=0.0) for _ in range(depth)])
        self.conv_out = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.res_block1:
            x = block(x, None)
        x = torch.nn.functional.interpolate(x, size=(int(round(x.shape[2] * self.factor)), int(round(x.shape[3] * self.factor))))
        x = self.attn(x)
        for block in self.res_block2:
            x = block(x, None)
        x = self.conv_out(x)
        return x


class MergedRescaleEncoder(nn.Module):

    def __init__(self, in_channels, ch, resolution, out_ch, num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True, ch_mult=(1, 2, 4, 8), rescale_factor=1.0, rescale_module_depth=1):
        super().__init__()
        intermediate_chn = ch * ch_mult[-1]
        self.encoder = Encoder(in_channels=in_channels, num_res_blocks=num_res_blocks, ch=ch, ch_mult=ch_mult, z_channels=intermediate_chn, double_z=False, resolution=resolution, attn_resolutions=attn_resolutions, dropout=dropout, resamp_with_conv=resamp_with_conv, out_ch=None)
        self.rescaler = LatentRescaler(factor=rescale_factor, in_channels=intermediate_chn, mid_channels=intermediate_chn, out_channels=out_ch, depth=rescale_module_depth)

    def forward(self, x):
        x = self.encoder(x)
        x = self.rescaler(x)
        return x


class MergedRescaleDecoder(nn.Module):

    def __init__(self, z_channels, out_ch, resolution, num_res_blocks, attn_resolutions, ch, ch_mult=(1, 2, 4, 8), dropout=0.0, resamp_with_conv=True, rescale_factor=1.0, rescale_module_depth=1):
        super().__init__()
        tmp_chn = z_channels * ch_mult[-1]
        self.decoder = Decoder(out_ch=out_ch, z_channels=tmp_chn, attn_resolutions=attn_resolutions, dropout=dropout, resamp_with_conv=resamp_with_conv, in_channels=None, num_res_blocks=num_res_blocks, ch_mult=ch_mult, resolution=resolution, ch=ch)
        self.rescaler = LatentRescaler(factor=rescale_factor, in_channels=z_channels, mid_channels=tmp_chn, out_channels=tmp_chn, depth=rescale_module_depth)

    def forward(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x


class Upsampler(nn.Module):

    def __init__(self, in_size, out_size, in_channels, out_channels, ch_mult=2):
        super().__init__()
        assert out_size >= in_size
        num_blocks = int(np.log2(out_size // in_size)) + 1
        factor_up = 1.0 + out_size % in_size
        None
        self.rescaler = LatentRescaler(factor=factor_up, in_channels=in_channels, mid_channels=2 * in_channels, out_channels=in_channels)
        self.decoder = Decoder(out_ch=out_channels, resolution=out_size, z_channels=in_channels, num_res_blocks=2, attn_resolutions=[], in_channels=None, ch=in_channels, ch_mult=[ch_mult for _ in range(num_blocks)])

    def forward(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x


class Resize(nn.Module):

    def __init__(self, in_channels=None, learned=False, mode='bilinear'):
        super().__init__()
        self.with_conv = learned
        self.mode = mode
        if self.with_conv:
            None
            raise NotImplementedError()
            assert in_channels is not None
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, scale_factor=1.0):
        if scale_factor == 1.0:
            return x
        else:
            x = torch.nn.functional.interpolate(x, mode=self.mode, align_corners=False, scale_factor=scale_factor)
        return x


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
        else:
            sum_dim = list(range(1, len(self.mean.shape)))
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=sum_dim)
            else:
                return 0.5 * torch.sum(torch.pow(self.mean - other.mean, 2) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar, dim=sum_dim)

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config, reload=False):
    if not 'target' in config:
        if config == '__is_first_stage__':
            return None
        elif config == '__is_unconditional__':
            return None
        raise KeyError('Expected key `target` to instantiate.')
    return get_obj_from_str(config['target'], reload=reload)(**config.get('params', dict()))


class FirstStagePostProcessor(nn.Module):

    def __init__(self, ch_mult: 'list', in_channels, pretrained_model: 'nn.Module'=None, reshape=False, n_channels=None, dropout=0.0, pretrained_config=None):
        super().__init__()
        if pretrained_config is None:
            assert pretrained_model is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
            self.pretrained_model = pretrained_model
        else:
            assert pretrained_config is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
            self.instantiate_pretrained(pretrained_config)
        self.do_reshape = reshape
        if n_channels is None:
            n_channels = self.pretrained_model.encoder.ch
        self.proj_norm = Normalize(in_channels, num_groups=in_channels // 2)
        self.proj = nn.Conv2d(in_channels, n_channels, kernel_size=3, stride=1, padding=1)
        blocks = []
        downs = []
        ch_in = n_channels
        for m in ch_mult:
            blocks.append(ResnetBlock(in_channels=ch_in, out_channels=m * n_channels, dropout=dropout))
            ch_in = m * n_channels
            downs.append(Downsample(ch_in, with_conv=False))
        self.model = nn.ModuleList(blocks)
        self.downsampler = nn.ModuleList(downs)

    def instantiate_pretrained(self, config):
        model = instantiate_from_config(config)
        self.pretrained_model = model.eval()
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_with_pretrained(self, x):
        c = self.pretrained_model.encode(x)
        if isinstance(c, DiagonalGaussianDistribution):
            c = c.mode()
        return c

    def forward(self, x):
        z_fs = self.encode_with_pretrained(x)
        z = self.proj_norm(z_fs)
        z = self.proj(z)
        z = nonlinearity(z)
        for submodel, downmodel in zip(self.model, self.downsampler):
            z = submodel(z, temb=None)
            z = downmodel(z)
        if self.do_reshape:
            z = rearrange(z, 'b c h w -> b (h w) c')
        return z


class SiLU(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class HybridConditioner(nn.Module):

    def __init__(self, c_concat_config, c_crossattn_config):
        super().__init__()
        self.concat_conditioner = instantiate_from_config(c_concat_config)
        self.crossattn_conditioner = instantiate_from_config(c_crossattn_config)

    def forward(self, c_concat, c_crossattn):
        c_concat = self.concat_conditioner(c_concat)
        c_crossattn = self.crossattn_conditioner(c_crossattn)
        return {'c_concat': [c_concat], 'c_crossattn': [c_crossattn]}


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


class NLayerDiscriminator1dFeats(NLayerDiscriminator):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input feats
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__(input_nc=input_nc, ndf=64, n_layers=n_layers, use_actnorm=use_actnorm)
        if not use_actnorm:
            norm_layer = nn.BatchNorm1d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm1d
        else:
            use_bias = norm_layer != nn.BatchNorm1d
        kw = 4
        padw = 1
        sequence = [nn.Conv1d(input_nc, input_nc // 2, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = input_nc // 2
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = max(nf_mult_prev // 2 ** n, 8)
            sequence += [nn.Conv1d(nf_mult_prev, nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), norm_layer(nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = max(nf_mult_prev // 2 ** n, 8)
        sequence += [nn.Conv1d(nf_mult_prev, nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), norm_layer(nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = max(nf_mult_prev // 2 ** n, 8)
        sequence += [nn.Conv1d(nf_mult_prev, nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), norm_layer(nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv1d(nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.main = nn.Sequential(*sequence)


class NLayerDiscriminator1dSpecs(NLayerDiscriminator):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=80, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input specs
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__(input_nc=input_nc, ndf=64, n_layers=n_layers, use_actnorm=use_actnorm)
        if not use_actnorm:
            norm_layer = nn.BatchNorm1d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm1d
        else:
            use_bias = norm_layer != nn.BatchNorm1d
        kw = 4
        padw = 1
        sequence = [nn.Conv1d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv1d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        input = input.squeeze(1)
        input = self.main(input)
        return input


class Discriminator2DFactory(nn.Module):

    def __init__(self, time_length, freq_length=80, kernel=(3, 3), c_in=1, hidden_size=128, norm_type='bn', reduction='sum'):
        super(Discriminator2DFactory, self).__init__()
        padding = kernel[0] // 2, kernel[1] // 2

        def discriminator_block(in_filters, out_filters, first=False):
            """
            Input: (B, in, 2H, 2W)
            Output:(B, out, H,  W)
            """
            conv = nn.Conv2d(in_filters, out_filters, kernel, (2, 2), padding)
            if norm_type == 'sn':
                conv = nn.utils.spectral_norm(conv)
            block = [conv, nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if norm_type == 'bn' and not first:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            if norm_type == 'in' and not first:
                block.append(nn.InstanceNorm2d(out_filters, affine=True))
            block = nn.Sequential(*block)
            return block
        self.model = nn.ModuleList([discriminator_block(c_in, hidden_size, first=True), discriminator_block(hidden_size, hidden_size), discriminator_block(hidden_size, hidden_size)])
        self.reduction = reduction
        ds_size = time_length // 2 ** 3, (freq_length + 7) // 2 ** 3
        if reduction != 'none':
            self.adv_layer = nn.Linear(hidden_size * ds_size[0] * ds_size[1], 1)
        else:
            self.adv_layer = nn.Linear(hidden_size * ds_size[1], 1)

    def forward(self, x):
        """

        :param x: [B, C, T, n_bins]
        :return: validity: [B, 1], h: List of hiddens
        """
        h = []
        for l in self.model:
            x = l(x)
            h.append(x)
        if self.reduction != 'none':
            x = x.view(x.shape[0], -1)
            validity = self.adv_layer(x)
        else:
            B, _, T_, _ = x.shape
            x = x.transpose(1, 2).reshape(B, T_, -1)
            validity = self.adv_layer(x)[:, :, 0]
        return validity, h


class MultiWindowDiscriminator(nn.Module):

    def __init__(self, time_lengths, cond_size=0, freq_length=80, kernel=(3, 3), c_in=1, hidden_size=128, norm_type='bn', reduction='sum'):
        super(MultiWindowDiscriminator, self).__init__()
        self.win_lengths = time_lengths
        self.reduction = reduction
        self.conv_layers = nn.ModuleList()
        if cond_size > 0:
            self.cond_proj_layers = nn.ModuleList()
            self.mel_proj_layers = nn.ModuleList()
        for time_length in time_lengths:
            conv_layer = [Discriminator2DFactory(time_length, freq_length, kernel, c_in=c_in, hidden_size=hidden_size, norm_type=norm_type, reduction=reduction)]
            self.conv_layers += conv_layer
            if cond_size > 0:
                self.cond_proj_layers.append(nn.Linear(cond_size, freq_length))
                self.mel_proj_layers.append(nn.Linear(freq_length, freq_length))

    def forward(self, x, x_len, cond=None, start_frames_wins=None):
        """
        Args:
            x (tensor): input mel, (B, c_in, T, n_bins).
            x_length (tensor): len of per mel. (B,).

        Returns:
            tensor : (B).
        """
        validity = []
        if start_frames_wins is None:
            start_frames_wins = [None] * len(self.conv_layers)
        h = []
        for i, start_frames in zip(range(len(self.conv_layers)), start_frames_wins):
            x_clip, c_clip, start_frames = self.clip(x, cond, x_len, self.win_lengths[i], start_frames)
            start_frames_wins[i] = start_frames
            if x_clip is None:
                continue
            if cond is not None:
                x_clip = self.mel_proj_layers[i](x_clip)
                c_clip = self.cond_proj_layers[i](c_clip)[:, None]
                x_clip = x_clip + c_clip
            x_clip, h_ = self.conv_layers[i](x_clip)
            h += h_
            validity.append(x_clip)
        if len(validity) != len(self.conv_layers):
            return None, start_frames_wins, h
        if self.reduction == 'sum':
            validity = sum(validity)
        elif self.reduction == 'stack':
            validity = torch.stack(validity, -1)
        elif self.reduction == 'none':
            validity = torch.cat(validity, -1)
        return validity, start_frames_wins, h

    def clip(self, x, cond, x_len, win_length, start_frames=None):
        """Ramdom clip x to win_length.
        Args:
            x (tensor) : (B, c_in, T, n_bins).
            cond (tensor) : (B, T, H).
            x_len (tensor) : (B,).
            win_length (int): target clip length

        Returns:
            (tensor) : (B, c_in, win_length, n_bins).

        """
        T_start = 0
        T_end = x_len.max() - win_length
        if T_end < 0:
            return None, None, start_frames
        T_end = T_end.item()
        if start_frames is None:
            start_frame = np.random.randint(low=T_start, high=T_end + 1)
            start_frames = [start_frame] * x.size(0)
        else:
            start_frame = start_frames[0]
        x_batch = x[:, :, start_frame:start_frame + win_length]
        c_batch = cond[:, start_frame:start_frame + win_length] if cond is not None else None
        return x_batch, c_batch, start_frames


class Discriminator(nn.Module):

    def __init__(self, time_lengths=[32, 64, 128], freq_length=80, cond_size=0, kernel=(3, 3), c_in=1, hidden_size=128, norm_type='bn', reduction='sum', uncond_disc=True):
        super(Discriminator, self).__init__()
        self.time_lengths = time_lengths
        self.cond_size = cond_size
        self.reduction = reduction
        self.uncond_disc = uncond_disc
        if uncond_disc:
            self.discriminator = MultiWindowDiscriminator(freq_length=freq_length, time_lengths=time_lengths, kernel=kernel, c_in=c_in, hidden_size=hidden_size, norm_type=norm_type, reduction=reduction)
        if cond_size > 0:
            self.cond_disc = MultiWindowDiscriminator(freq_length=freq_length, time_lengths=time_lengths, cond_size=cond_size, kernel=kernel, c_in=c_in, hidden_size=hidden_size, norm_type=norm_type, reduction=reduction)

    def forward(self, x, cond=None, x_len=None, start_frames_wins=None):
        """

        :param x: [B, T, 80]
        :param cond: [B, T, cond_size]
        :param return_y_only:
        :return:
        """
        if len(x.shape) == 3:
            x = x[:, None, :, :]
        if x_len == None:
            x_len = x.sum([1, -1]).ne(0).int().sum([-1])
        ret = {'y_c': None, 'y': None}
        if self.uncond_disc:
            ret['y'], start_frames_wins, ret['h'] = self.discriminator(x, x_len, start_frames_wins=start_frames_wins)
        if self.cond_size > 0 and cond is not None:
            ret['y_c'], start_frames_wins, ret['h_c'] = self.cond_disc(x, x_len, cond, start_frames_wins=start_frames_wins)
        ret['start_frames_wins'] = start_frames_wins
        return ret


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


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        return x


class ConvBlock5x5(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock5x5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        return x


class AttBlock(nn.Module):

    def __init__(self, n_in, n_out, activation='linear', temperature=1.0):
        super(AttBlock, self).__init__()
        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.cla = nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn_att = nn.BatchNorm1d(n_out)

    def forward(self, x):
        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


class Cnn14(nn.Module):

    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, out_emb):
        super(Cnn14, self).__init__()
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, win_length=window_size, window=window, center=center, pad_mode=pad_mode, freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        self.fc1 = nn.Linear(2048, out_emb, bias=True)
        self.fc_audioset = nn.Linear(out_emb, classes_num, bias=True)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)
        """
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}
        return output_dict


class Projection(nn.Module):

    def __init__(self, d_in: 'int', d_out: 'int', p: 'float'=0.5) ->None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


def get_audio_encoder(name: 'str'):
    if name == 'Cnn14':
        return Cnn14
    else:
        raise Exception('The audio encoder name {} is incorrect or not supported'.format(name))


class AudioEncoder(nn.Module):

    def __init__(self, audioenc_name: 'str', d_in: 'int', d_out: 'int', sample_rate: 'int', window_size: 'int', hop_size: 'int', mel_bins: 'int', fmin: 'int', fmax: 'int', classes_num: 'int') ->None:
        super().__init__()
        audio_encoder = get_audio_encoder(audioenc_name)
        self.base = audio_encoder(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, d_in)
        self.projection = Projection(d_in, d_out)

    def forward(self, x):
        out_dict = self.base(x)
        audio_features, audio_classification_output = out_dict['embedding'], out_dict['clipwise_output']
        projected_vec = self.projection(audio_features)
        return projected_vec, audio_classification_output


class TextEncoder(nn.Module):

    def __init__(self, d_out: 'int', text_model: 'str', transformer_embed_dim: 'int') ->None:
        super().__init__()
        self.base = AutoModel.from_pretrained(text_model)
        self.projection = Projection(transformer_embed_dim, d_out)

    def forward(self, x):
        out = self.base(**x)[0]
        out = out[:, 0, :]
        projected_vec = self.projection(out)
        return projected_vec


class CLAP(nn.Module):

    def __init__(self, audioenc_name: 'str', sample_rate: 'int', window_size: 'int', hop_size: 'int', mel_bins: 'int', fmin: 'int', fmax: 'int', classes_num: 'int', out_emb: 'int', text_model: 'str', transformer_embed_dim: 'int', d_proj: 'int'):
        super().__init__()
        self.audio_encoder = AudioEncoder(audioenc_name, out_emb, d_proj, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num)
        self.caption_encoder = TextEncoder(d_proj, text_model, transformer_embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, audio, text):
        audio_embed, _ = self.audio_encoder(audio)
        caption_embed = self.caption_encoder(text)
        return caption_embed, audio_embed, self.logit_scale.exp()


class ClassEmbedder(nn.Module):

    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class SpatialRescaler(nn.Module):

    def __init__(self, n_stages=1, method='bilinear', multiplier=0.5, in_channels=3, out_channels=None, bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest', 'linear', 'bilinear', 'trilinear', 'bicubic', 'area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            None
            self.channel_mapper = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)
        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class DAF(nn.Module):
    """
    直接相加 DirectAddFuse
    """

    def __init__(self):
        super(DAF, self).__init__()

    def forward(self, x, residual):
        return x + residual


class iAFF(nn.Module):
    """
    多特征融合 iAFF
    """

    def __init__(self, channels=64, r=4, type='2D'):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)
        if type == '1D':
            self.local_att = nn.Sequential(nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm1d(inter_channels), nn.ReLU(inplace=True), nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm1d(channels))
            self.global_att = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm1d(inter_channels), nn.ReLU(inplace=True), nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm1d(channels))
            self.local_att2 = nn.Sequential(nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm1d(inter_channels), nn.ReLU(inplace=True), nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm1d(channels))
            self.global_att2 = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm1d(inter_channels), nn.ReLU(inplace=True), nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm1d(channels))
        elif type == '2D':
            self.local_att = nn.Sequential(nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True), nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(channels))
            self.global_att = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True), nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(channels))
            self.local_att2 = nn.Sequential(nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True), nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(channels))
            self.global_att2 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True), nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(channels))
        else:
            raise f'the type is not supported'
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        flag = False
        xa = x + residual
        if xa.size(0) == 1:
            xa = torch.cat([xa, xa], dim=0)
            flag = True
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)
        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        if flag:
            xo = xo[0].unsqueeze(0)
        return xo


class AFF(nn.Module):
    """
    多特征融合 AFF
    """

    def __init__(self, channels=64, r=4, type='2D'):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)
        if type == '1D':
            self.local_att = nn.Sequential(nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm1d(inter_channels), nn.ReLU(inplace=True), nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm1d(channels))
            self.global_att = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm1d(inter_channels), nn.ReLU(inplace=True), nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm1d(channels))
        elif type == '2D':
            self.local_att = nn.Sequential(nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True), nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(channels))
            self.global_att = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True), nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(channels))
        else:
            raise f'the type is not supported.'
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        flag = False
        xa = x + residual
        if xa.size(0) == 1:
            xa = torch.cat([xa, xa], dim=0)
            flag = True
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        if flag:
            xo = xo[0].unsqueeze(0)
        return xo


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


def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, patch_stride=16, enable_fusion=False, fusion_type='None'):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patch_stride = to_2tuple(patch_stride)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.grid_size = img_size[0] // patch_stride[0], img_size[1] // patch_stride[1]
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type
        padding = (patch_size[0] - patch_stride[0]) // 2, (patch_size[1] - patch_stride[1]) // 2
        if self.enable_fusion and self.fusion_type == 'channel_map':
            self.proj = nn.Conv2d(in_chans * 4, embed_dim, kernel_size=patch_size, stride=patch_stride, padding=padding)
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        if self.enable_fusion and self.fusion_type in ['daf_2d', 'aff_2d', 'iaff_2d']:
            self.mel_conv2d = nn.Conv2d(in_chans, embed_dim, kernel_size=(patch_size[0], patch_size[1] * 3), stride=(patch_stride[0], patch_stride[1] * 3), padding=padding)
            if self.fusion_type == 'daf_2d':
                self.fusion_model = DAF()
            elif self.fusion_type == 'aff_2d':
                self.fusion_model = AFF(channels=embed_dim, type='2D')
            elif self.fusion_type == 'iaff_2d':
                self.fusion_model = iAFF(channels=embed_dim, type='2D')

    def forward(self, x, longer_idx=None):
        if self.enable_fusion and self.fusion_type in ['daf_2d', 'aff_2d', 'iaff_2d']:
            global_x = x[:, 0:1, :, :]
            B, C, H, W = global_x.shape
            assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            global_x = self.proj(global_x)
            TW = global_x.size(-1)
            if len(longer_idx) > 0:
                local_x = x[longer_idx, 1:, :, :].contiguous()
                B, C, H, W = local_x.shape
                local_x = local_x.view(B * C, 1, H, W)
                local_x = self.mel_conv2d(local_x)
                local_x = local_x.view(B, C, local_x.size(1), local_x.size(2), local_x.size(3))
                local_x = local_x.permute((0, 2, 3, 1, 4)).contiguous().flatten(3)
                TB, TC, TH, _ = local_x.size()
                if local_x.size(-1) < TW:
                    local_x = torch.cat([local_x, torch.zeros((TB, TC, TH, TW - local_x.size(-1)), device=global_x.device)], dim=-1)
                else:
                    local_x = local_x[:, :, :, :TW]
                global_x[longer_idx] = self.fusion_model(global_x[longer_idx], local_x)
            x = global_x
        else:
            B, C, H, W = x.shape
            assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

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
        """
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
        return x, attn

    def extra_repr(self):
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


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
        input_resolution (tuple[int]): Input resulotion.
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

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_before_mlp='ln'):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm_before_mlp = norm_before_mlp
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.norm_before_mlp == 'ln':
            self.norm2 = nn.LayerNorm(dim)
        elif self.norm_before_mlp == 'bn':
            self.norm2 = lambda x: nn.BatchNorm1d(dim)(x.transpose(1, 2)).transpose(1, 2)
        else:
            raise NotImplementedError
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
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
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows, attn = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn

    def extra_repr(self):
        return f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}'


class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        assert H % 2 == 0 and W % 2 == 0, f'x size ({H}*{W}) are not even.'
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self):
        return f'input_resolution={self.input_resolution}, dim={self.dim}'


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, norm_before_mlp='ln'):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer, norm_before_mlp=norm_before_mlp) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        attns = []
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x, attn = blk(x)
                if not self.training:
                    attns.append(attn.unsqueeze(0))
        if self.downsample is not None:
            x = self.downsample(x)
        if not self.training:
            attn = torch.cat(attns, dim=0)
            attn = torch.mean(attn, dim=0)
        return x, attn

    def extra_repr(self):
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'


def do_mixup(x, mixup_lambda):
    """
    Args:
      x: (batch_size , ...)
      mixup_lambda: (batch_size,)
    Returns:
      out: (batch_size, ...)
    """
    out = (x.transpose(0, -1) * mixup_lambda + torch.flip(x, dims=[0]).transpose(0, -1) * (1 - mixup_lambda)).transpose(0, -1)
    return out


def interpolate(x, ratio):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.

    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    batch_size, time_steps, classes_num = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


class HTSAT_Swin_Transformer(nn.Module):
    """HTSAT based on the Swin Transformer
    Args:
        spec_size (int | tuple(int)): Input Spectrogram size. Default 256
        patch_size (int | tuple(int)): Patch size. Default: 4
        path_stride (iot | tuple(int)): Patch Stride for Frequency and Time Axis. Default: 4
        in_chans (int): Number of input image channels. Default: 1 (mono)
        num_classes (int): Number of classes for classification head. Default: 527
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each HTSAT-Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 8
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        config (module): The configuration Module from config.py
    """

    def __init__(self, spec_size=256, patch_size=4, patch_stride=(4, 4), in_chans=1, num_classes=527, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[4, 8, 16, 32], window_size=8, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, norm_before_mlp='ln', config=None, enable_fusion=False, fusion_type='None', **kwargs):
        super(HTSAT_Swin_Transformer, self).__init__()
        self.config = config
        self.spec_size = spec_size
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.ape = ape
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = len(self.depths)
        self.num_features = int(self.embed_dim * 2 ** (self.num_layers - 1))
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.qkv_bias = qkv_bias
        self.qk_scale = None
        self.patch_norm = patch_norm
        self.norm_layer = norm_layer if self.patch_norm else None
        self.norm_before_mlp = norm_before_mlp
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type
        self.freq_ratio = self.spec_size // self.config.mel_bins
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.interpolate_ratio = 32
        self.spectrogram_extractor = Spectrogram(n_fft=config.window_size, hop_length=config.hop_size, win_length=config.window_size, window=window, center=center, pad_mode=pad_mode, freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=config.sample_rate, n_fft=config.window_size, n_mels=config.mel_bins, fmin=config.fmin, fmax=config.fmax, ref=ref, amin=amin, top_db=top_db, freeze_parameters=True)
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)
        self.bn0 = nn.BatchNorm2d(self.config.mel_bins)
        self.patch_embed = PatchEmbed(img_size=self.spec_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim, norm_layer=self.norm_layer, patch_stride=patch_stride, enable_fusion=self.enable_fusion, fusion_type=self.fusion_type)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.grid_size
        self.patches_resolution = patches_resolution
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=self.drop_rate)
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(self.embed_dim * 2 ** i_layer), input_resolution=(patches_resolution[0] // 2 ** i_layer, patches_resolution[1] // 2 ** i_layer), depth=self.depths[i_layer], num_heads=self.num_heads[i_layer], window_size=self.window_size, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale, drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])], norm_layer=self.norm_layer, downsample=PatchMerging if i_layer < self.num_layers - 1 else None, use_checkpoint=use_checkpoint, norm_before_mlp=self.norm_before_mlp)
            self.layers.append(layer)
        self.norm = self.norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        SF = self.spec_size // 2 ** (len(self.depths) - 1) // self.patch_stride[0] // self.freq_ratio
        self.tscam_conv = nn.Conv2d(in_channels=self.num_features, out_channels=self.num_classes, kernel_size=(SF, 3), padding=(0, 1))
        self.head = nn.Linear(num_classes, num_classes)
        if self.enable_fusion and self.fusion_type in ['daf_1d', 'aff_1d', 'iaff_1d']:
            self.mel_conv1d = nn.Sequential(nn.Conv1d(64, 64, kernel_size=5, stride=3, padding=2), nn.BatchNorm1d(64))
            if self.fusion_type == 'daf_1d':
                self.fusion_model = DAF()
            elif self.fusion_type == 'aff_1d':
                self.fusion_model = AFF(channels=64, type='1D')
            elif self.fusion_type == 'iaff_1d':
                self.fusion_model = iAFF(channels=64, type='1D')
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
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, longer_idx=None):
        frames_num = x.shape[2]
        x = self.patch_embed(x, longer_idx=longer_idx)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for i, layer in enumerate(self.layers):
            x, attn = layer(x)
        x = self.norm(x)
        B, N, C = x.shape
        SF = frames_num // 2 ** (len(self.depths) - 1) // self.patch_stride[0]
        ST = frames_num // 2 ** (len(self.depths) - 1) // self.patch_stride[1]
        x = x.permute(0, 2, 1).contiguous().reshape(B, C, SF, ST)
        B, C, F, T = x.shape
        c_freq_bin = F // self.freq_ratio
        x = x.reshape(B, C, F // c_freq_bin, c_freq_bin, T)
        x = x.permute(0, 1, 3, 2, 4).contiguous().reshape(B, C, c_freq_bin, -1)
        fine_grained_latent_output = torch.mean(x, dim=2)
        fine_grained_latent_output = interpolate(fine_grained_latent_output.permute(0, 2, 1).contiguous(), 8 * self.patch_stride[1])
        latent_output = self.avgpool(torch.flatten(x, 2))
        latent_output = torch.flatten(latent_output, 1)
        x = self.tscam_conv(x)
        x = torch.flatten(x, 2)
        fpx = interpolate(torch.sigmoid(x).permute(0, 2, 1).contiguous(), 8 * self.patch_stride[1])
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        output_dict = {'framewise_output': fpx, 'clipwise_output': torch.sigmoid(x), 'fine_grained_embedding': fine_grained_latent_output, 'embedding': latent_output}
        return output_dict

    def crop_wav(self, x, crop_size, spe_pos=None):
        time_steps = x.shape[2]
        tx = torch.zeros(x.shape[0], x.shape[1], crop_size, x.shape[3])
        for i in range(len(x)):
            if spe_pos is None:
                crop_pos = random.randint(0, time_steps - crop_size - 1)
            else:
                crop_pos = spe_pos
            tx[i][0] = x[i, 0, crop_pos:crop_pos + crop_size, :]
        return tx

    def reshape_wav2img(self, x):
        B, C, T, F = x.shape
        target_T = int(self.spec_size * self.freq_ratio)
        target_F = self.spec_size // self.freq_ratio
        assert T <= target_T and F <= target_F, 'the wav size should less than or equal to the swin input size'
        if T < target_T:
            x = nn.functional.interpolate(x, (target_T, x.shape[3]), mode='bicubic', align_corners=True)
        if F < target_F:
            x = nn.functional.interpolate(x, (x.shape[2], target_F), mode='bicubic', align_corners=True)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], self.freq_ratio, x.shape[3] // self.freq_ratio)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3], x.shape[4])
        return x

    def repeat_wat2img(self, x, cur_pos):
        B, C, T, F = x.shape
        target_T = int(self.spec_size * self.freq_ratio)
        target_F = self.spec_size // self.freq_ratio
        assert T <= target_T and F <= target_F, 'the wav size should less than or equal to the swin input size'
        if T < target_T:
            x = nn.functional.interpolate(x, (target_T, x.shape[3]), mode='bicubic', align_corners=True)
        if F < target_F:
            x = nn.functional.interpolate(x, (x.shape[2], target_F), mode='bicubic', align_corners=True)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x[:, :, :, cur_pos:cur_pos + self.spec_size]
        x = x.repeat(repeats=(1, 1, 4, 1))
        return x

    def forward(self, x: 'torch.Tensor', mixup_lambda=None, infer_mode=False, device=None):
        if self.enable_fusion and x['longer'].sum() == 0:
            if self.training:
                x['longer'][torch.randint(0, x['longer'].shape[0], (1,))] = True
            else:
                x = x['mel_fusion']
                x = x.transpose(1, 3)
                x = self.bn0(x)
                x = x.transpose(1, 3)
                x = self.reshape_wav2img(x)
                output_dict = self.forward_features(x, longer_idx=[])
                return output_dict
        if not self.enable_fusion:
            x = x['waveform']
            x = self.spectrogram_extractor(x)
            x = self.logmel_extractor(x)
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)
            if self.training:
                x = self.spec_augmenter(x)
            if self.training and mixup_lambda is not None:
                x = do_mixup(x, mixup_lambda)
            x = self.reshape_wav2img(x)
            output_dict = self.forward_features(x)
        else:
            longer_list = x['longer']
            x = x['mel_fusion']
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)
            longer_list_idx = torch.where(longer_list)[0]
            if self.fusion_type in ['daf_1d', 'aff_1d', 'iaff_1d']:
                new_x = x[:, 0:1, :, :].clone().contiguous()
                if len(longer_list_idx) > 0:
                    fusion_x_local = x[longer_list_idx, 1:, :, :].clone().contiguous()
                    FB, FC, FT, FF = fusion_x_local.size()
                    fusion_x_local = fusion_x_local.view(FB * FC, FT, FF)
                    fusion_x_local = torch.permute(fusion_x_local, (0, 2, 1)).contiguous()
                    fusion_x_local = self.mel_conv1d(fusion_x_local)
                    fusion_x_local = fusion_x_local.view(FB, FC, FF, fusion_x_local.size(-1))
                    fusion_x_local = torch.permute(fusion_x_local, (0, 2, 1, 3)).contiguous().flatten(2)
                    if fusion_x_local.size(-1) < FT:
                        fusion_x_local = torch.cat([fusion_x_local, torch.zeros((FB, FF, FT - fusion_x_local.size(-1)), device=device)], dim=-1)
                    else:
                        fusion_x_local = fusion_x_local[:, :, :FT]
                    new_x = new_x.squeeze(1).permute((0, 2, 1)).contiguous()
                    new_x[longer_list_idx] = self.fusion_model(new_x[longer_list_idx], fusion_x_local)
                    x = new_x.permute((0, 2, 1)).contiguous()[:, None, :, :]
                else:
                    x = new_x
            elif self.fusion_type in ['daf_2d', 'aff_2d', 'iaff_2d', 'channel_map']:
                x = x
            if self.training:
                x = self.spec_augmenter(x)
            if self.training and mixup_lambda is not None:
                x = do_mixup(x, mixup_lambda)
            x = self.reshape_wav2img(x)
            output_dict = self.forward_features(x, longer_idx=longer_list_idx)
        return output_dict


class MLPLayers(nn.Module):

    def __init__(self, units=[512, 512, 512], nonlin=nn.ReLU(), dropout=0.1):
        super(MLPLayers, self).__init__()
        self.nonlin = nonlin
        self.dropout = dropout
        sequence = []
        for u0, u1 in zip(units[:-1], units[1:]):
            sequence.append(nn.Linear(u0, u1))
            sequence.append(self.nonlin)
            sequence.append(nn.Dropout(self.dropout))
        sequence = sequence[:-2]
        self.sequential = nn.Sequential(*sequence)

    def forward(self, X):
        X = self.sequential(X)
        return X


class LinearProbe(nn.Module):

    def __init__(self, model, mlp, freeze, in_ch, out_ch, act=None):
        """
        Args:
            model: nn.Module
            mlp: bool, if True, then use the MLP layer as the linear probe module
            freeze: bool, if Ture, then freeze all the CLAP model's layers when training the linear probe
            in_ch: int, the output channel from CLAP model
            out_ch: int, the output channel from linear probe (class_num)
            act: torch.nn.functional, the activation function before the loss function
        """
        super().__init__()
        in_ch = 512
        self.clap_model = model
        self.clap_model.text_branch = None
        self.freeze = freeze
        if mlp:
            self.lp_layer = MLPLayers(units=[in_ch, in_ch * 2, out_ch])
        else:
            self.lp_layer = nn.Linear(in_ch, out_ch)
        if self.freeze:
            for param in self.clap_model.parameters():
                param.requires_grad = False
        if act == 'None':
            self.act = None
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'elu':
            self.act = nn.ELU()
        elif act == 'prelu':
            self.act = nn.PReLU(num_parameters=in_ch)
        elif act == 'softmax':
            self.act = nn.Softmax(dim=-1)
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x, mix_lambda=None, device=None):
        """
        Args:
            x: waveform, torch.tensor [batch, t_samples] / batch of mel_spec and longer list
            mix_lambda: torch.tensor [batch], the mixup lambda
        Returns:
            class_prob: torch.tensor [batch, class_num]

        """
        if self.freeze:
            self.clap_model.eval()
        x = self.clap_model.audio_projection(self.clap_model.audio_branch(x, mixup_lambda=mix_lambda, device=device)['embedding'])
        out = self.lp_layer(x)
        if self.act is not None:
            out = self.act(out)
        return out


def gather_features(audio_features, text_features, audio_features_mlp=None, text_features_mlp=None, local_loss=False, gather_with_grad=False, rank=0, world_size=1, use_horovod=False, mlp_loss=False):
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_audio_features = hvd.allgather(audio_features)
            all_text_features = hvd.allgather(text_features)
            if mlp_loss:
                all_audio_features_mlp = hvd.allgather(audio_features_mlp)
                all_text_features_mlp = hvd.allgather(text_features_mlp)
        else:
            with torch.no_grad():
                all_audio_features = hvd.allgather(audio_features)
                all_text_features = hvd.allgather(text_features)
                if mlp_loss:
                    all_audio_features_mlp = hvd.allgather(audio_features_mlp)
                    all_text_features_mlp = hvd.allgather(text_features_mlp)
            if not local_loss:
                gathered_audio_features = list(all_audio_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_audio_features[rank] = audio_features
                gathered_text_features[rank] = text_features
                all_audio_features = torch.cat(gathered_audio_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
                if mlp_loss:
                    gathered_audio_features_mlp = list(all_audio_features_mlp.chunk(world_size, dim=0))
                    gathered_text_features_mlp = list(all_text_features_mlp.chunk(world_size, dim=0))
                    gathered_audio_features_mlp[rank] = audio_features_mlp
                    gathered_text_features_mlp[rank] = text_features_mlp
                    all_audio_features_mlp = torch.cat(gathered_audio_features_mlp, dim=0)
                    all_text_features_mlp = torch.cat(gathered_text_features_mlp, dim=0)
    elif gather_with_grad:
        all_audio_features = torch.cat(torch.distributed.nn.all_gather(audio_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        if mlp_loss:
            all_audio_features_mlp = torch.cat(torch.distributed.nn.all_gather(audio_features_mlp), dim=0)
            all_text_features_mlp = torch.cat(torch.distributed.nn.all_gather(text_features_mlp), dim=0)
    else:
        gathered_audio_features = [torch.zeros_like(audio_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_audio_features, audio_features)
        dist.all_gather(gathered_text_features, text_features)
        if mlp_loss:
            gathered_audio_features_mlp = [torch.zeros_like(audio_features_mlp) for _ in range(world_size)]
            gathered_text_features_mlp = [torch.zeros_like(text_features_mlp) for _ in range(world_size)]
            dist.all_gather(gathered_audio_features_mlp, audio_features_mlp)
            dist.all_gather(gathered_text_features_mlp, text_features_mlp)
        if not local_loss:
            gathered_audio_features[rank] = audio_features
            gathered_text_features[rank] = text_features
            if mlp_loss:
                gathered_audio_features_mlp[rank] = audio_features_mlp
                gathered_text_features_mlp[rank] = text_features_mlp
        all_audio_features = torch.cat(gathered_audio_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)
        if mlp_loss:
            all_audio_features_mlp = torch.cat(gathered_audio_features_mlp, dim=0)
            all_text_features_mlp = torch.cat(gathered_text_features_mlp, dim=0)
    if mlp_loss:
        return all_audio_features, all_text_features, all_audio_features_mlp, all_text_features_mlp
    else:
        return all_audio_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(self, local_loss=False, gather_with_grad=False, cache_labels=False, rank=0, world_size=1, use_horovod=False, mlp_loss=False, weight_loss_kappa=0):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.mlp_loss = mlp_loss
        self.weighted_loss = bool(weight_loss_kappa != 0)
        self.weight_loss_kappa = weight_loss_kappa
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, audio_features, text_features, logit_scale_a, logit_scale_t=None, audio_features_mlp=None, text_features_mlp=None):
        device = audio_features.device
        if self.mlp_loss:
            if self.world_size > 1:
                all_audio_features, all_text_features, all_audio_features_mlp, all_text_features_mlp = gather_features(audio_features=audio_features, text_features=text_features, audio_features_mlp=audio_features_mlp, text_features_mlp=text_features_mlp, local_loss=self.local_loss, gather_with_grad=self.gather_with_grad, rank=self.rank, world_size=self.world_size, use_horovod=self.use_horovod, mlp_loss=self.mlp_loss)
                if self.local_loss:
                    a_logits_per_audio = logit_scale_a * audio_features @ all_text_features_mlp.T
                    a_logits_per_text = logit_scale_a * text_features_mlp @ all_audio_features.T
                    t_logits_per_audio = logit_scale_t * audio_features_mlp @ all_text_features.T
                    t_logits_per_text = logit_scale_t * text_features @ all_audio_features_mlp.T
                else:
                    a_logits_per_audio = logit_scale_a * all_audio_features @ all_text_features_mlp.T
                    a_logits_per_text = a_logits_per_audio.T
                    t_logits_per_audio = logit_scale_t * all_audio_features_mlp @ all_text_features.T
                    t_logits_per_text = t_logits_per_audio.T
            else:
                a_logits_per_audio = logit_scale_a * audio_features @ text_features_mlp.T
                a_logits_per_text = logit_scale_a * text_features_mlp @ audio_features.T
                t_logits_per_audio = logit_scale_t * audio_features_mlp @ text_features.T
                t_logits_per_text = logit_scale_t * text_features @ audio_features_mlp.T
            num_logits = a_logits_per_audio.shape[0]
            if self.prev_num_logits != num_logits or device not in self.labels:
                labels = torch.arange(num_logits, device=device, dtype=torch.long)
                if self.world_size > 1 and self.local_loss:
                    labels = labels + num_logits * self.rank
                if self.cache_labels:
                    self.labels[device] = labels
                    self.prev_num_logits = num_logits
            else:
                labels = self.labels[device]
            if not self.weighted_loss:
                total_loss = (F.cross_entropy(a_logits_per_audio, labels) + F.cross_entropy(a_logits_per_text, labels) + F.cross_entropy(t_logits_per_audio, labels) + F.cross_entropy(t_logits_per_text, labels)) / 4
            else:
                audio_weight = (audio_features @ audio_features.T).detach()
                audio_weight = torch.exp(torch.sum(audio_weight, axis=1) / (self.weight_loss_kappa * len(audio_weight))).detach()
                text_weight = (text_features @ text_features.T).detach()
                text_weight = torch.exp(torch.sum(text_weight, axis=1) / (self.weight_loss_kappa * len(text_features))).detach()
                total_loss = (F.cross_entropy(a_logits_per_audio, labels, weight=audio_weight) + F.cross_entropy(a_logits_per_text, labels, weight=audio_weight) + F.cross_entropy(t_logits_per_audio, labels, weight=text_weight) + F.cross_entropy(t_logits_per_text, labels, weight=text_weight)) / 4
        else:
            if self.world_size > 1:
                all_audio_features, all_text_features = gather_features(audio_features=audio_features, text_features=text_features, local_loss=self.local_loss, gather_with_grad=self.gather_with_grad, rank=self.rank, world_size=self.world_size, use_horovod=self.use_horovod, mlp_loss=self.mlp_loss)
                if self.local_loss:
                    logits_per_audio = logit_scale_a * audio_features @ all_text_features.T
                    logits_per_text = logit_scale_a * text_features @ all_audio_features.T
                else:
                    logits_per_audio = logit_scale_a * all_audio_features @ all_text_features.T
                    logits_per_text = logits_per_audio.T
            else:
                logits_per_audio = logit_scale_a * audio_features @ text_features.T
                logits_per_text = logit_scale_a * text_features @ audio_features.T
            num_logits = logits_per_audio.shape[0]
            if self.prev_num_logits != num_logits or device not in self.labels:
                labels = torch.arange(num_logits, device=device, dtype=torch.long)
                if self.world_size > 1 and self.local_loss:
                    labels = labels + num_logits * self.rank
                if self.cache_labels:
                    self.labels[device] = labels
                    self.prev_num_logits = num_logits
            else:
                labels = self.labels[device]
            if not self.weighted_loss:
                total_loss = (F.cross_entropy(logits_per_audio, labels) + F.cross_entropy(logits_per_text, labels)) / 2
            else:
                audio_weight = (all_audio_features @ all_audio_features.T).detach()
                audio_weight = torch.exp(torch.sum(audio_weight, axis=1) / (self.weight_loss_kappa * len(all_audio_features))).detach()
                text_weight = (all_text_features @ all_text_features.T).detach()
                text_weight = torch.exp(torch.sum(text_weight, axis=1) / (self.weight_loss_kappa * len(all_text_features))).detach()
                total_loss = (F.cross_entropy(logits_per_audio, labels, weight=text_weight) + F.cross_entropy(logits_per_text, labels, weight=audio_weight)) / 2
        return total_loss


def calc_celoss(pred, target):
    target = torch.argmax(target, 1).long()
    return nn.CrossEntropyLoss()(pred, target)


class LPLoss(nn.Module):

    def __init__(self, loss_name):
        super().__init__()
        if loss_name == 'bce':
            self.loss_func = nn.BCEWithLogitsLoss()
        elif loss_name == 'ce':
            self.loss_func = calc_celoss
        elif loss_name == 'mse':
            self.loss_func = nn.MSELoss()
        else:
            raise ValueError(f'the loss func should be at least one of [bce, ce, mse]')

    def forward(self, pred, target):
        loss = self.loss_func(pred, target)
        return loss


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


def freeze_batch_norm_2d(module, module_match={}, name=''):
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`. If `module` is
    itself an instance of either `BatchNorm2d` or `SyncBatchNorm`, it is converted into `FrozenBatchNorm2d` and
    returned. Otherwise, the module is walked recursively and submodules are converted in place.

    Args:
        module (torch.nn.Module): Any PyTorch module.
        module_match (dict): Dictionary of full module names to freeze (all if empty)
        name (str): Full module name (prefix)

    Returns:
        torch.nn.Module: Resulting module

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = '.'.join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, image_size=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        embed_dim = width * 32
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim)
        self.init_parameters()

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def init_parameters(self):
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)
        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith('bn3.weight'):
                    nn.init.zeros_(param)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False
        if freeze_bn_stats:
            freeze_batch_norm_2d(self)

    def stem(self, x):
        for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
            x = self.relu(bn(conv(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: 'torch.Tensor'):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x


class QuickGELU(nn.Module):

    def forward(self, x: 'torch.Tensor'):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: 'int', n_head: 'int', act_layer: 'Callable'=nn.GELU):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)), ('gelu', act_layer()), ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: 'torch.Tensor', attn_mask: 'Optional[torch.Tensor]'=None):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: 'torch.Tensor', attn_mask: 'Optional[torch.Tensor]'=None):
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):

    def __init__(self, width: 'int', layers: 'int', heads: 'int', act_layer: 'Callable'=nn.GELU):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, act_layer=act_layer) for _ in range(layers)])

    def forward(self, x: 'torch.Tensor', attn_mask: 'Optional[torch.Tensor]'=None):
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)
        return x


class VisualTransformer(nn.Module):

    def __init__(self, image_size: 'int', patch_size: 'int', width: 'int', layers: 'int', heads: 'int', output_dim: 'int', act_layer: 'Callable'=nn.GELU):
        super().__init__()
        self.image_size = image_size
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((image_size // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.text_branch = Transformer(width, layers, heads, act_layer=act_layer)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: 'torch.Tensor'):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.text_branch(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


class Cnn6(nn.Module):

    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, enable_fusion=False, fusion_type='None'):
        super(Cnn6, self).__init__()
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, win_length=window_size, window=window, center=center, pad_mode=pad_mode, freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, freeze_parameters=True)
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512)
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None, device=None):
        """
        Input: (batch_size, data_length)"""
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        if self.training:
            x = self.spec_augmenter(x)
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        latent_x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        latent_x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        latent_x = latent_x1 + latent_x2
        latent_x = latent_x.transpose(1, 2)
        latent_x = F.relu_(self.fc1(latent_x))
        latent_output = interpolate(latent_x, 16)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding, 'fine_grained_embedding': latent_output}
        return output_dict


class Cnn10(nn.Module):

    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, enable_fusion=False, fusion_type='None'):
        super(Cnn10, self).__init__()
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, win_length=window_size, window=window, center=center, pad_mode=pad_mode, freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, freeze_parameters=True)
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.fc_audioset = nn.Linear(1024, classes_num, bias=True)
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None, device=None):
        """
        Input: (batch_size, data_length)"""
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        if self.training:
            x = self.spec_augmenter(x)
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        latent_x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        latent_x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        latent_x = latent_x1 + latent_x2
        latent_x = latent_x.transpose(1, 2)
        latent_x = F.relu_(self.fc1(latent_x))
        latent_output = interpolate(latent_x, 32)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding, 'fine_grained_embedding': latent_output}
        return output_dict


_MODEL_CONFIGS = {}


def convert_weights_to_fp16(model: 'nn.Module'):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()
        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f'{s}_proj_weight' for s in ['in', 'q', 'k', 'v']], 'in_proj_bias', 'bias_k', 'bias_v']:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()
        for name in ['text_projection', 'proj']:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()
    model.apply(_convert_weights_to_fp16)


_RN101 = dict(openai='https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt', yfcc15m='https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn101-quickgelu-yfcc15m-3e04b30e.pt')


_RN101_quickgelu = dict(openai='https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt', yfcc15m='https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn101-quickgelu-yfcc15m-3e04b30e.pt')


_RN50 = dict(openai='https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt', yfcc15m='https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-yfcc15m-455df137.pt', cc12m='https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-cc12m-f000538c.pt')


_RN50_quickgelu = dict(openai='https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt', yfcc15m='https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-yfcc15m-455df137.pt', cc12m='https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-cc12m-f000538c.pt')


_RN50x16 = dict(openai='https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt')


_RN50x4 = dict(openai='https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt')


_VITB16 = dict(openai='https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt')


_VITB32 = dict(openai='https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt', laion400m_e31='https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt', laion400m_e32='https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt', laion400m_avg='https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt')


_VITB32_quickgelu = dict(openai='https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt', laion400m_e31='https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt', laion400m_e32='https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt', laion400m_avg='https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt')


_VITL14 = dict(openai='https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt')


_PRETRAINED = {'RN50': _RN50, 'RN50-quickgelu': _RN50_quickgelu, 'RN101': _RN101, 'RN101-quickgelu': _RN101_quickgelu, 'RN50x4': _RN50x4, 'RN50x16': _RN50x16, 'ViT-B-32': _VITB32, 'ViT-B-32-quickgelu': _VITB32_quickgelu, 'ViT-B-16': _VITB16, 'ViT-L-14': _VITL14}


def get_pretrained_url(model: 'str', tag: 'str'):
    if model not in _PRETRAINED:
        return ''
    model_pretrained = _PRETRAINED[model]
    if tag not in model_pretrained:
        return ''
    return model_pretrained[tag]


def list_models():
    """enumerate available model architectures based on config files"""
    return list(_MODEL_CONFIGS.keys())


def build_model_from_openai_state_dict(state_dict: 'dict', model_cfg, enable_fusion: 'bool'=False, fusion_type: 'str'='None'):
    embed_dim = model_cfg['embed_dim']
    audio_cfg = model_cfg['audio_cfg']
    text_cfg = model_cfg['text_cfg']
    context_length = state_dict['positional_embedding'].shape[0]
    vocab_size = state_dict['token_embedding.weight'].shape[0]
    transformer_width = state_dict['ln_final.weight'].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split('.')[2] for k in state_dict if k.startswith(f'transformer.resblocks')))
    audio_cfg = CLAPAudioCfp(**audio_cfg)
    text_cfg = CLAPTextCfg(**text_cfg)
    model = CLAP(embed_dim, audio_cfg=audio_cfg, text_cfg=text_cfg, quick_gelu=True, enable_fusion=enable_fusion, fusion_type=fusion_type)
    state_dict['logit_scale_a'] = state_dict['logit_scale']
    state_dict['logit_scale_t'] = state_dict['logit_scale']
    pop_keys = list(state_dict.keys())[:]
    for key in pop_keys:
        if key.startswith('visual.'):
            state_dict.pop(key, None)
    for key in ['logit_scale', 'input_resolution', 'context_length', 'vocab_size']:
        state_dict.pop(key, None)
    model.load_state_dict(state_dict, strict=False)
    return model.eval()


def list_pretrained_tag_models(tag: 'str'):
    """ return all models having the specified pretrain tag """
    models = []
    for k in _PRETRAINED.keys():
        if tag in _PRETRAINED[k]:
            models.append(k)
    return models


def list_openai_models() ->List[str]:
    """Returns the names of available CLIP models"""
    return list_pretrained_tag_models('openai')


def load_state_dict(checkpoint_path: 'str', map_location='cpu', skip_params=True):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if skip_params:
        if next(iter(state_dict.items()))[0].startswith('module'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def float32_to_int16(x):
    x = np.clip(x, a_min=-1.0, a_max=1.0)
    return (x * 32767.0).astype(np.int16)


h = None


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.0:
        None
    if torch.max(y) > 1.0:
        None
    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax) + '_' + str(y.device)] = torch.from_numpy(mel).float()
        hann_window[str(y.device)] = torch.hann_window(win_size)
    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode='reflect')
    y = y.squeeze(1)
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)], center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-09)
    spec = torch.matmul(mel_basis[str(fmax) + '_' + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def get_audio_features(sample, audio_data, max_len, data_truncating, data_filling, audio_cfg, require_grad=False):
    """
    Calculate and add audio features to sample.
    Sample: a dict containing all the data of current sample.
    audio_data: a tensor of shape (T) containing audio data.
    max_len: the maximum length of audio data.
    data_truncating: the method of truncating data.
    data_filling: the method of filling data.
    audio_cfg: a dict containing audio configuration. Comes from model_cfg['audio_cfg'].
    require_grad: whether to require gradient for audio data.
        This is useful when we want to apply gradient-based classifier-guidance.
    """
    grad_fn = suppress if require_grad else torch.no_grad
    with grad_fn():
        if len(audio_data) > max_len:
            if data_truncating == 'rand_trunc':
                longer = torch.tensor([True])
            elif data_truncating == 'fusion':
                mel = get_mel(audio_data, audio_cfg)
                chunk_frames = max_len // audio_cfg['hop_size'] + 1
                total_frames = mel.shape[0]
                if chunk_frames == total_frames:
                    mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                    sample['mel_fusion'] = mel_fusion
                    longer = torch.tensor([False])
                else:
                    ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
                    if len(ranges[1]) == 0:
                        ranges[1] = [0]
                    if len(ranges[2]) == 0:
                        ranges[2] = [0]
                    idx_front = np.random.choice(ranges[0])
                    idx_middle = np.random.choice(ranges[1])
                    idx_back = np.random.choice(ranges[2])
                    mel_chunk_front = mel[idx_front:idx_front + chunk_frames, :]
                    mel_chunk_middle = mel[idx_middle:idx_middle + chunk_frames, :]
                    mel_chunk_back = mel[idx_back:idx_back + chunk_frames, :]
                    mel_shrink = torchvision.transforms.Resize(size=[chunk_frames, audio_cfg['mel_bins']])(mel[None])[0]
                    mel_fusion = torch.stack([mel_shrink, mel_chunk_front, mel_chunk_middle, mel_chunk_back], dim=0)
                    sample['mel_fusion'] = mel_fusion
                    longer = torch.tensor([True])
            else:
                raise NotImplementedError(f'data_truncating {data_truncating} not implemented')
            overflow = len(audio_data) - max_len
            idx = np.random.randint(0, overflow + 1)
            audio_data = audio_data[idx:idx + max_len]
        else:
            if len(audio_data) < max_len:
                if data_filling == 'repeatpad':
                    n_repeat = int(max_len / len(audio_data))
                    audio_data = audio_data.repeat(n_repeat)
                    audio_data = F.pad(audio_data, (0, max_len - len(audio_data)), mode='constant', value=0)
                elif data_filling == 'pad':
                    audio_data = F.pad(audio_data, (0, max_len - len(audio_data)), mode='constant', value=0)
                elif data_filling == 'repeat':
                    n_repeat = int(max_len / len(audio_data))
                    audio_data = audio_data.repeat(n_repeat + 1)[:max_len]
                else:
                    raise NotImplementedError(f'data_filling {data_filling} not implemented')
            if data_truncating == 'fusion':
                mel = get_mel(audio_data, audio_cfg)
                mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                sample['mel_fusion'] = mel_fusion
            longer = torch.tensor([False])
    sample['longer'] = longer
    sample['waveform'] = audio_data
    return sample


def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


class CLAP_Module(torch.nn.Module):

    def __init__(self, enable_fusion=False, device=None, amodel='HTSAT-tiny', tmodel='roberta') ->None:
        """Initialize CLAP Model

        Parameters
        ----------
        enable_fusion: bool
            if true, it will create the fusion clap model, otherwise non-fusion clap model (default: false) 
        device: str
            if None, it will automatically detect the device (gpu or cpu)
        amodel: str
            audio encoder architecture, default: HTSAT-tiny
        tmodel: str
            text encoder architecture, default: roberta
        """
        super(CLAP_Module, self).__init__()
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        precision = 'fp32'
        if enable_fusion:
            fusion_type = 'aff_2d'
            model, model_cfg = create_model(amodel, tmodel, precision=precision, device=device, enable_fusion=enable_fusion, fusion_type=fusion_type)
        else:
            model, model_cfg = create_model(amodel, tmodel, precision=precision, device=device, enable_fusion=enable_fusion)
        self.enable_fusion = enable_fusion
        self.model = model
        self.model_cfg = model_cfg
        self.tokenize = RobertaTokenizer.from_pretrained('roberta-base')

    def tokenizer(self, text):
        result = self.tokenize(text, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
        return result

    def load_ckpt(self, ckpt=None, model_id=-1):
        """Load the pretrained checkpoint of CLAP model

        Parameters
        ----------
        ckpt: str
            if ckpt is specified, the model will load this ckpt, otherwise the model will download the ckpt from zenodo. 
 
            For fusion model, it will download the 630k+audioset fusion model (id=3). For non-fusion model, it will download the 630k+audioset model (id=1).
        model_id:
            if model_id is specified, you can download our best ckpt, as:
                id = 0 --> 630k non-fusion ckpt 

                id = 1 --> 630k+audioset non-fusion ckpt 

                id = 2 --> 630k fusion ckpt 

                id = 3 --> 630k+audioset fusion ckpt 

            Note that if your model is specied as non-fusion model but you download a fusion model ckpt, you will face an error.
        """
        download_link = 'https://huggingface.co/lukewys/laion_clap/resolve/main/'
        download_names = ['630k-best.pt', '630k-audioset-best.pt', '630k-fusion-best.pt', '630k-audioset-fusion-best.pt']
        if ckpt is not None:
            None
        else:
            None
            if model_id == -1:
                model_id = 3 if self.enable_fusion else 1
            package_dir = os.path.dirname(os.path.realpath(__file__))
            weight_file_name = download_names[model_id]
            ckpt = os.path.join(package_dir, weight_file_name)
            if os.path.exists(ckpt):
                None
            else:
                None
                ckpt = wget.download(download_link + weight_file_name, os.path.dirname(ckpt))
                None
        None
        ckpt = load_state_dict(ckpt, skip_params=True)
        self.model.load_state_dict(ckpt)
        param_names = [n for n, p in self.model.named_parameters()]
        for n in param_names:
            None

    def get_audio_embedding_from_filelist(self, x, use_tensor=False):
        """get audio embeddings from the audio file list

        Parameters
        ----------
        x: List[str] (N,): 
            an audio file list to extract features, audio files can have different lengths (as we have the feature fusion machanism)
        use_tensor: boolean:
            if True, it will return the torch tensor, preserving the gradient (default: False).
        Returns
        ----------
        audio_embed : numpy.darray | torch.Tensor (N,D):
            audio embeddings that extracted from audio files
        """
        self.model.eval()
        audio_input = []
        for f in x:
            audio_waveform, _ = librosa.load(f, sr=48000)
            audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
            audio_waveform = torch.from_numpy(audio_waveform).float()
            temp_dict = {}
            temp_dict = get_audio_features(temp_dict, audio_waveform, 480000, data_truncating='fusion' if self.enable_fusion else 'rand_trunc', data_filling='repeatpad', audio_cfg=self.model_cfg['audio_cfg'], require_grad=audio_waveform.requires_grad)
            audio_input.append(temp_dict)
        audio_embed = self.model.get_audio_embedding(audio_input)
        if not use_tensor:
            audio_embed = audio_embed.detach().cpu().numpy()
        return audio_embed

    def get_audio_embedding_from_data(self, x, use_tensor=False):
        """get audio embeddings from the audio data

        Parameters
        ----------
        x: np.darray | torch.Tensor (N,T): 
            audio data, must be mono audio tracks.
        use_tensor: boolean:
            if True, x should be the tensor input and the output will be the tesnor, preserving the gradient (default: False).      
            Note that if 'use tensor' is set to True, it will not do the quantize of the audio waveform (otherwise the gradient will not be preserved).
        Returns
        ----------
        audio embed: numpy.darray | torch.Tensor (N,D):
            audio embeddings that extracted from audio files
        """
        self.model.eval()
        audio_input = []
        for audio_waveform in x:
            if not use_tensor:
                audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
                audio_waveform = torch.from_numpy(audio_waveform).float()
            temp_dict = {}
            temp_dict = get_audio_features(temp_dict, audio_waveform, 480000, data_truncating='fusion' if self.enable_fusion else 'rand_trunc', data_filling='repeatpad', audio_cfg=self.model_cfg['audio_cfg'], require_grad=audio_waveform.requires_grad)
            audio_input.append(temp_dict)
        audio_embed = self.model.get_audio_embedding(audio_input)
        if not use_tensor:
            audio_embed = audio_embed.detach().cpu().numpy()
        return audio_embed

    def get_text_embedding(self, x, tokenizer=None, use_tensor=False):
        """get text embeddings from texts

        Parameters
        ----------
        x: List[str] (N,): 
            text list 
        tokenizer: func:
            the tokenizer function, if not provided (None), will use the default Roberta tokenizer.
        use_tensor: boolean:
            if True, the output will be the tesnor, preserving the gradient (default: False).      
        Returns
        ----------
        text_embed : numpy.darray | torch.Tensor (N,D):
            text embeddings that extracted from texts
        """
        self.model.eval()
        if tokenizer is not None:
            text_input = tokenizer(x)
        else:
            text_input = self.tokenizer(x)
        text_embed = self.model.get_text_embedding(text_input)
        if not use_tensor:
            text_embed = text_embed.detach().cpu().numpy()
        return text_embed


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class LPIPSWithDiscriminator(nn.Module):

    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0, disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0, perceptual_weight=1.0, use_actnorm=False, disc_conditional=False, disc_loss='hinge'):
        super().__init__()
        assert disc_loss in ['hinge', 'vanilla']
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=use_actnorm).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == 'hinge' else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

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

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx, global_step, last_layer=None, cond=None, split='train', weights=None):
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
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        if optimizer_idx == 0:
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
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
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss
            log = {'{}/total_loss'.format(split): loss.clone().detach().mean(), '{}/logvar'.format(split): self.logvar.detach(), '{}/kl_loss'.format(split): kl_loss.detach().mean(), '{}/nll_loss'.format(split): nll_loss.detach().mean(), '{}/rec_loss'.format(split): rec_loss.detach().mean(), '{}/d_weight'.format(split): d_weight.detach(), '{}/disc_factor'.format(split): torch.tensor(disc_factor), '{}/g_loss'.format(split): g_loss.detach().mean()}
            return loss, log
        if optimizer_idx == 1:
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            log = {'{}/disc_loss'.format(split): d_loss.clone().detach().mean(), '{}/logits_real'.format(split): logits_real.detach().mean(), '{}/logits_fake'.format(split): logits_fake.detach().mean()}
            return d_loss, log


def l1(x, y):
    return torch.abs(x - y)


def l2(x, y):
    return torch.pow(x - y, 2)


def measure_perplexity(predicted_indices, n_embed):
    encodings = F.one_hot(predicted_indices, n_embed).float().reshape(-1, n_embed)
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    cluster_use = torch.sum(avg_probs > 0)
    return perplexity, cluster_use


class VQLPIPSWithDiscriminator(nn.Module):

    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0, disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0, perceptual_weight=1.0, use_actnorm=False, disc_conditional=False, disc_ndf=64, disc_loss='hinge', n_classes=None, perceptual_loss='lpips', pixel_loss='l1'):
        super().__init__()
        assert disc_loss in ['hinge', 'vanilla']
        assert perceptual_loss in ['lpips', 'clips', 'dists']
        assert pixel_loss in ['l1', 'l2']
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        if perceptual_loss == 'lpips':
            None
            self.perceptual_loss = LPIPS().eval()
        else:
            raise ValueError(f'Unknown perceptual loss: >> {perceptual_loss} <<')
        self.perceptual_weight = perceptual_weight
        if pixel_loss == 'l1':
            self.pixel_loss = l1
        else:
            self.pixel_loss = l2
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=use_actnorm, ndf=disc_ndf).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == 'hinge':
            self.disc_loss = hinge_d_loss
        elif disc_loss == 'vanilla':
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        None
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.n_classes = n_classes

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

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx, global_step, last_layer=None, cond=None, split='train', predicted_indices=None):
        if not exists(codebook_loss):
            codebook_loss = torch.tensor([0.0])
        rec_loss = self.pixel_loss(inputs.contiguous(), reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])
        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)
        if optimizer_idx == 0:
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)
            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()
            log = {'{}/total_loss'.format(split): loss.clone().detach().mean(), '{}/quant_loss'.format(split): codebook_loss.detach().mean(), '{}/nll_loss'.format(split): nll_loss.detach().mean(), '{}/rec_loss'.format(split): rec_loss.detach().mean(), '{}/p_loss'.format(split): p_loss.detach().mean(), '{}/d_weight'.format(split): d_weight.detach(), '{}/disc_factor'.format(split): torch.tensor(disc_factor), '{}/g_loss'.format(split): g_loss.detach().mean()}
            if predicted_indices is not None:
                assert self.n_classes is not None
                with torch.no_grad():
                    perplexity, cluster_usage = measure_perplexity(predicted_indices, self.n_classes)
                log[f'{split}/perplexity'] = perplexity
                log[f'{split}/cluster_usage'] = cluster_usage
            return loss, log
        if optimizer_idx == 1:
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            log = {'{}/disc_loss'.format(split): d_loss.clone().detach().mean(), '{}/logits_real'.format(split): logits_real.detach().mean(), '{}/logits_fake'.format(split): logits_fake.detach().mean()}
            return d_loss, log


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)


CKPT_MAP = {'vggishish_lpaps': 'vggishish16.pt', 'vggishish_mean_std_melspec_10s_22050hz': 'train_means_stds_melspec_10s_22050hz.txt', 'melception': 'melception-21-05-10T09-28-40.pt'}


MD5_MAP = {'vggishish_lpaps': '197040c524a07ccacf7715d7080a80bd', 'vggishish_mean_std_melspec_10s_22050hz': 'f449c6fd0e248936c16f6d22492bb625', 'melception': 'a71a41041e945b457c7d3d814bbcf72d'}


URL_MAP = {'vggishish_lpaps': 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/vggishish16.pt', 'vggishish_mean_std_melspec_10s_22050hz': 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/train_means_stds_melspec_10s_22050hz.txt', 'melception': 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/melception-21-05-10T09-28-40.pt'}


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


class ScalingLayer(nn.Module):

    def __init__(self):
        super(ScalingLayer, self).__init__()
        stat_path = get_ckpt_path('vggishish_mean_std_melspec_10s_22050hz', 'ldm/modules/autoencoder/lpaps')
        means, stds = np.loadtxt(stat_path, dtype=np.float32).T
        means = 2 * means - 1
        stds = 2 * stds
        self.register_buffer('shift', torch.from_numpy(means)[None, None, :, None])
        self.register_buffer('scale', torch.from_numpy(stds)[None, None, :, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


class LPAPS(nn.Module):

    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.net = vggishish16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name='vggishish_lpaps'):
        ckpt = get_ckpt_path(name, 'ldm/modules/autoencoder/lpaps')
        self.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')), strict=False)
        None

    @classmethod
    def from_pretrained(cls, name='vggishish_lpaps'):
        if name != 'vggishish_lpaps':
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


class LPAPSWithDiscriminator(nn.Module):

    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0, time_lengths=[16, 32, 64], disc_factor=1.0, disc_weight=1.0, perceptual_weight=1.0, disc_conditional=False, disc_loss='hinge'):
        super().__init__()
        assert disc_loss in ['hinge', 'vanilla']
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        if self.perceptual_weight > 0:
            self.perceptual_loss = LPAPS().eval()
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.discriminator = Discriminator(time_lengths=time_lengths, reduction='stack').apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == 'hinge':
            self.disc_loss = hinge_d_loss
        elif disc_loss == 'vanilla':
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        None
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

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

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx, global_step, last_layer=None, cond=None, split='train', weights=None):
        if len(inputs.shape) == 3:
            inputs, reconstructions = inputs.unsqueeze(1), reconstructions.unsqueeze(1)
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        inputs, reconstructions = inputs.squeeze(1).transpose(1, 2), reconstructions.squeeze(1).transpose(1, 2)
        if optimizer_idx == 0:
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())['y']
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))['y']
            g_loss = -torch.mean(logits_fake)
            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss
            log = {'{}/total_loss'.format(split): loss.clone().detach().mean(), '{}/logvar'.format(split): self.logvar.detach(), '{}/kl_loss'.format(split): kl_loss.detach().mean(), '{}/nll_loss'.format(split): nll_loss.detach().mean(), '{}/rec_loss'.format(split): rec_loss.detach().mean(), '{}/d_weight'.format(split): d_weight.detach(), '{}/disc_factor'.format(split): torch.tensor(disc_factor), '{}/g_loss'.format(split): g_loss.detach().mean()}
            return loss, log
        if optimizer_idx == 1:
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())['y']
                logits_fake = self.discriminator(reconstructions.contiguous().detach())['y']
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))['y']
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))['y']
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            log = {'{}/disc_loss'.format(split): d_loss.clone().detach().mean(), '{}/logits_real'.format(split): logits_real.detach().mean(), '{}/logits_fake'.format(split): logits_fake.detach().mean()}
            return d_loss, log


class DummyLoss(nn.Module):

    def __init__(self):
        super().__init__()


class VQLPAPSWithDiscriminator(nn.Module):

    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0, disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0, perceptual_weight=1.0, use_actnorm=False, disc_conditional=False, disc_ndf=64, disc_loss='hinge', n_classes=None, pixel_loss='l1'):
        super().__init__()
        assert disc_loss in ['hinge', 'vanilla']
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = None
        self.perceptual_weight = perceptual_weight
        if pixel_loss == 'l1':
            self.pixel_loss = l1
        else:
            self.pixel_loss = l2
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=use_actnorm, ndf=disc_ndf).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == 'hinge':
            self.disc_loss = hinge_d_loss
        elif disc_loss == 'vanilla':
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        None
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.n_classes = n_classes

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

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx, global_step, last_layer=None, cond=None, split='train', predicted_indices=None):
        if not exists(codebook_loss):
            codebook_loss = torch.tensor([0.0])
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])
        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)
        if optimizer_idx == 0:
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)
            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()
            log = {'{}/total_loss'.format(split): loss.clone().detach().mean(), '{}/quant_loss'.format(split): codebook_loss.detach().mean(), '{}/nll_loss'.format(split): nll_loss.detach().mean(), '{}/rec_loss'.format(split): rec_loss.detach().mean(), '{}/p_loss'.format(split): p_loss.detach().mean(), '{}/d_weight'.format(split): d_weight.detach(), '{}/disc_factor'.format(split): torch.tensor(disc_factor), '{}/g_loss'.format(split): g_loss.detach().mean()}
            return loss, log
        if optimizer_idx == 1:
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            log = {'{}/disc_loss'.format(split): d_loss.clone().detach().mean(), '{}/logits_real'.format(split): logits_real.detach().mean(), '{}/logits_fake'.format(split): logits_fake.detach().mean()}
            return d_loss, log


class PositionalEncoding(nn.Module):

    def __init__(self, num_hiddens, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, x):
        x = x + self.P[:, :x.shape[1], :]
        return x


class TemporalTransformerSkip(TemporalTransformer):

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0.0, context_dim=None):
        super().__init__(in_channels, n_heads, d_head, depth, dropout, context_dim)
        self.skip_linear = nn.Linear(2 * in_channels, in_channels)

    def forward(self, x, skip, context=None):
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c t -> b t c')
        skip = rearrange(skip, 'b c t -> b t c')
        x = self.skip_linear(torch.cat([x, skip], dim=-1))
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b t c -> b c t')
        x = self.proj_out(x)
        return x + x_in


class AbsolutePositionalEmbedding(nn.Module):

    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)
        self.init_()

    def init_(self):
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, x):
        n = torch.arange(x.shape[1], device=x.device)
        return self.emb(n)[None, :, :]


class TransformerWrapper(nn.Module):

    def __init__(self, *, num_tokens, max_seq_len, attn_layers, emb_dim=None, max_mem_len=0.0, emb_dropout=0.0, num_memory_tokens=None, tie_embedding=False, use_pos_emb=True):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), 'attention layers must be one of Encoder or Decoder'
        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)
        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len) if use_pos_emb and not attn_layers.has_pos_emb else always(0)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)
        self.init_()
        self.to_logits = nn.Linear(dim, num_tokens) if not tie_embedding else lambda t: t @ self.token_emb.weight.t()
        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))
            if hasattr(attn_layers, 'num_memory_tokens'):
                attn_layers.num_memory_tokens = num_memory_tokens

    def init_(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)

    def forward(self, x, return_embeddings=False, mask=None, return_mems=False, return_attn=False, mems=None, **kwargs):
        b, n, device, num_mem = *x.shape, x.device, self.num_memory_tokens
        x = self.token_emb(x)
        x += self.pos_emb(x)
        x = self.emb_dropout(x)
        x = self.project_emb(x)
        if num_mem > 0:
            mem = repeat(self.memory_tokens, 'n d -> b n d', b=b)
            x = torch.cat((mem, x), dim=1)
            if exists(mask):
                mask = F.pad(mask, (num_mem, 0), value=True)
        x, intermediates = self.attn_layers(x, mask=mask, mems=mems, return_hiddens=True, **kwargs)
        x = self.norm(x)
        mem, x = x[:, :num_mem], x[:, num_mem:]
        out = self.to_logits(x) if not return_embeddings else x
        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = list(map(lambda pair: torch.cat(pair, dim=-2), zip(mems, hiddens))) if exists(mems) else hiddens
            new_mems = list(map(lambda t: t[..., -self.max_mem_len:, :].detach(), new_mems))
            return out, new_mems
        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps
        return out


class Snake(nn.Module):
    """
    Implementation of a sine-based periodic activation function
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha: trainable parameter
            alpha is initialized to 1 by default, higher values = higher-frequency.
            alpha will be trained along with the rest of your model.
        """
        super(Snake, self).__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-09

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + 1.0 / (alpha + self.no_div_by_zero) * pow(sin(x * alpha), 2)
        return x


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        """
        super(SnakeBeta, self).__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-09

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + 1.0 / (beta + self.no_div_by_zero) * pow(sin(x * alpha), 2)
        return x


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        filter_ /= filter_.sum()
        filter = filter_.view(1, 1, kernel_size)
    return filter


class LowPassFilter1d(nn.Module):

    def __init__(self, cutoff=0.5, half_width=0.6, stride: 'int'=1, padding: 'bool'=True, padding_mode: 'str'='replicate', kernel_size: 'int'=12):
        super().__init__()
        if cutoff < -0.0:
            raise ValueError('Minimum cutoff must be larger than zero.')
        if cutoff > 0.5:
            raise ValueError('A cutoff above 0.5 does not make sense.')
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer('filter', filter)

    def forward(self, x):
        _, C, _ = x.shape
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        out = F.conv1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
        return out


class DownSample1d(nn.Module):

    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, stride=ratio, kernel_size=self.kernel_size)

    def forward(self, x):
        xx = self.lowpass(x)
        return xx


class UpSample1d(nn.Module):

    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        self.register_buffer('filter', filter)

    def forward(self, x):
        _, C, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode='replicate')
        x = self.ratio * F.conv_transpose1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
        x = x[..., self.pad_left:-self.pad_right]
        return x


class Activation1d(nn.Module):

    def __init__(self, activation, up_ratio: 'int'=2, down_ratio: 'int'=2, up_kernel_size: 'int'=12, down_kernel_size: 'int'=12):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean, std)


class AMPBlock1(torch.nn.Module):

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5), activation=None):
        super(AMPBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2])))])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))])
        self.convs2.apply(init_weights)
        self.num_layers = len(self.convs1) + len(self.convs2)
        if activation == 'snake':
            self.activations = nn.ModuleList([Activation1d(activation=Snake(channels, alpha_logscale=h.snake_logscale)) for _ in range(self.num_layers)])
        elif activation == 'snakebeta':
            self.activations = nn.ModuleList([Activation1d(activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale)) for _ in range(self.num_layers)])
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class AMPBlock2(torch.nn.Module):

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3), activation=None):
        super(AMPBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1])))])
        self.convs.apply(init_weights)
        self.num_layers = len(self.convs)
        if activation == 'snake':
            self.activations = nn.ModuleList([Activation1d(activation=Snake(channels, alpha_logscale=h.snake_logscale)) for _ in range(self.num_layers)])
        elif activation == 'snakebeta':
            self.activations = nn.ModuleList([Activation1d(activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale)) for _ in range(self.num_layers)])
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

    def forward(self, x):
        for c, a in zip(self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class BigVGAN(torch.nn.Module):

    def __init__(self, h):
        super(BigVGAN, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = AMPBlock1 if h.resblock == '1' else AMPBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(nn.ModuleList([weight_norm(ConvTranspose1d(h.upsample_initial_channel // 2 ** i, h.upsample_initial_channel // 2 ** (i + 1), k, u, padding=(k - u) // 2))]))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // 2 ** (i + 1)
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d, activation=h.activation))
        if h.activation == 'snake':
            activation_post = Snake(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post)
        elif h.activation == 'snakebeta':
            activation_post = SnakeBeta(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post)
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        None
        for l in self.ups:
            for l_i in l:
                remove_weight_norm(l_i)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


LRELU_SLOPE = 0.1


class DiscriminatorP(torch.nn.Module):

    def __init__(self, h, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.d_mult = h.discriminator_channel_mult
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([norm_f(Conv2d(1, int(32 * self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(int(32 * self.d_mult), int(128 * self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(int(128 * self.d_mult), int(512 * self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(int(512 * self.d_mult), int(1024 * self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(int(1024 * self.d_mult), int(1024 * self.d_mult), (kernel_size, 1), 1, padding=(2, 0)))])
        self.conv_post = norm_f(Conv2d(int(1024 * self.d_mult), 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - t % self.period
            x = F.pad(x, (0, n_pad), 'reflect')
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):

    def __init__(self, h):
        super(MultiPeriodDiscriminator, self).__init__()
        self.mpd_reshapes = h.mpd_reshapes
        None
        discriminators = [DiscriminatorP(h, rs, use_spectral_norm=h.use_spectral_norm) for rs in self.mpd_reshapes]
        self.discriminators = nn.ModuleList(discriminators)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorR(nn.Module):

    def __init__(self, cfg, resolution):
        super().__init__()
        self.resolution = resolution
        assert len(self.resolution) == 3, 'MRD layer requires list with len=3, got {}'.format(self.resolution)
        self.lrelu_slope = LRELU_SLOPE
        norm_f = weight_norm if cfg.use_spectral_norm == False else spectral_norm
        if hasattr(cfg, 'mrd_use_spectral_norm'):
            None
            norm_f = weight_norm if cfg.mrd_use_spectral_norm == False else spectral_norm
        self.d_mult = cfg.discriminator_channel_mult
        if hasattr(cfg, 'mrd_channel_mult'):
            None
            self.d_mult = cfg.mrd_channel_mult
        self.convs = nn.ModuleList([norm_f(nn.Conv2d(1, int(32 * self.d_mult), (3, 9), padding=(1, 4))), norm_f(nn.Conv2d(int(32 * self.d_mult), int(32 * self.d_mult), (3, 9), stride=(1, 2), padding=(1, 4))), norm_f(nn.Conv2d(int(32 * self.d_mult), int(32 * self.d_mult), (3, 9), stride=(1, 2), padding=(1, 4))), norm_f(nn.Conv2d(int(32 * self.d_mult), int(32 * self.d_mult), (3, 9), stride=(1, 2), padding=(1, 4))), norm_f(nn.Conv2d(int(32 * self.d_mult), int(32 * self.d_mult), (3, 3), padding=(1, 1)))])
        self.conv_post = norm_f(nn.Conv2d(int(32 * self.d_mult), 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []
        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(x, (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode='reflect')
        x = x.squeeze(1)
        x = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False, return_complex=True)
        x = torch.view_as_real(x)
        mag = torch.norm(x, p=2, dim=-1)
        return mag


class MultiResolutionDiscriminator(nn.Module):

    def __init__(self, cfg, debug=False):
        super().__init__()
        self.resolutions = cfg.resolutions
        assert len(self.resolutions) == 3, 'MRD requires list of list with len=3, each element having a list with len=3. got {}'.format(self.resolutions)
        self.discriminators = nn.ModuleList([DiscriminatorR(cfg, resolution) for resolution in self.resolutions])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AFF,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64]), torch.rand([4, 64, 64, 64])], {})),
    (AbsolutePositionalEmbedding,
     lambda: ([], {'dim': 4, 'max_seq_len': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ActNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AttBlock,
     lambda: ([], {'n_in': 4, 'n_out': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (AttentionPool2d,
     lambda: ([], {'spacial_dim': 4, 'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Bottleneck,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ClipLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (ConditionEmbedder,
     lambda: ([], {'hidden_size': 4, 'context_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv1dFeedForward,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (Conv1dGEGLU,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (ConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvBlock5x5,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DAF,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Discriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Downsample,
     lambda: ([], {'in_channels': 4, 'with_conv': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Downsample1D,
     lambda: ([], {'in_channels': 4, 'with_conv': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FeedForward,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FinalLayer,
     lambda: ([], {'hidden_size': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (FixedPositionalEmbedding,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GEGLU,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GroupNorm32,
     lambda: ([], {'num_groups': 1, 'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (IdentityFirstStage,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MultiWindowDiscriminator,
     lambda: ([], {'time_lengths': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (NLayerDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (NLayerDiscriminator1dFeats,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64])], {})),
    (PositionEmbedding,
     lambda: ([], {'num_embeddings': 4, 'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PositionalEncoding,
     lambda: ([], {'num_hiddens': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Projection,
     lambda: ([], {'d_in': 4, 'd_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (QuickGELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RMSNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Residual,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ResidualAttentionBlock,
     lambda: ([], {'d_model': 4, 'n_head': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (Resize,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Rezero,
     lambda: ([], {'fn': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Scale,
     lambda: ([], {'value': 4, 'fn': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ScaleNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SiLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Snake,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SnakeBeta,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SpatialRescaler,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Transformer,
     lambda: ([], {'width': 4, 'layers': 1, 'heads': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (Upsample,
     lambda: ([], {'in_channels': 4, 'with_conv': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Upsample1D,
     lambda: ([], {'in_channels': 4, 'with_conv': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (iAFF,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64]), torch.rand([4, 64, 64, 64])], {})),
]

