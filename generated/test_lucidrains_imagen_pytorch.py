import sys
_module = sys.modules[__name__]
del sys
imagen_pytorch = _module
cli = _module
configs = _module
data = _module
elucidated_imagen = _module
imagen_pytorch = _module
imagen_video = _module
t5 = _module
test = _module
test_trainer = _module
trainer = _module
utils = _module
version = _module
setup = _module

from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchvision, types, typing, uuid, warnings
import operator as op
from dataclasses import dataclass
import numpy as np
from torch import Tensor
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import torch


from functools import partial


from torch import nn


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torchvision import transforms as T


from torch.nn.utils.rnn import pad_sequence


from math import sqrt


from random import random


from typing import List


from typing import Union


from collections import namedtuple


import torch.nn.functional as F


from torch.amp import autocast


from torch.nn.parallel import DistributedDataParallel


import torchvision.transforms as T


import math


from functools import wraps


from torch import einsum


from torch.special import expm1


import functools


from math import ceil


from collections.abc import Iterable


from torch.utils.data import random_split


from torch.optim import Adam


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.optim.lr_scheduler import LambdaLR


from torch.cuda.amp import autocast


from torch.cuda.amp import GradScaler


import numpy as np


from functools import reduce


DEFAULT_T5_NAME = 'google/t5-v1_1-base'


def log(t, eps: 'float'=1e-12):
    return torch.log(t.clamp(min=eps))


@torch.jit.script
def alpha_cosine_log_snr(t, s: 'float'=0.008):
    return -log(torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2 - 1, eps=1e-05)


@torch.jit.script
def beta_linear_log_snr(t):
    return -torch.log(expm1(0.0001 + 10 * t ** 2))


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))


def maybe(fn):

    @wraps(fn)
    def inner(x):
        if not exists(x):
            return x
        return fn(x)
    return inner


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


class GaussianDiffusionContinuousTimes(nn.Module):

    def __init__(self, *, noise_schedule, timesteps=1000):
        super().__init__()
        if noise_schedule == 'linear':
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == 'cosine':
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')
        self.num_timesteps = timesteps

    def get_times(self, batch_size, noise_level, *, device):
        return torch.full((batch_size,), noise_level, device=device, dtype=torch.float32)

    def sample_random_times(self, batch_size, *, device):
        return torch.zeros((batch_size,), device=device).float().uniform_(0, 1)

    def get_condition(self, times):
        return maybe(self.log_snr)(times)

    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1.0, 0.0, self.num_timesteps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    def q_posterior(self, x_start, x_t, t, *, t_next=None):
        t_next = default(t_next, lambda : (t - 1.0 / self.num_timesteps).clamp(min=0.0))
        """ https://openreview.net/attachment?id=2LdBqxc1Yv&name=supplementary_material """
        log_snr = self.log_snr(t)
        log_snr_next = self.log_snr(t_next)
        log_snr, log_snr_next = map(partial(right_pad_dims_to, x_t), (log_snr, log_snr_next))
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)
        c = -expm1(log_snr - log_snr_next)
        posterior_mean = alpha_next * (x_t * (1 - c) / alpha + c * x_start)
        posterior_variance = sigma_next ** 2 * c
        posterior_log_variance_clipped = log(posterior_variance, eps=1e-20)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        dtype = x_start.dtype
        if isinstance(t, float):
            batch = x_start.shape[0]
            t = torch.full((batch,), t, device=x_start.device, dtype=dtype)
        noise = default(noise, lambda : torch.randn_like(x_start))
        log_snr = self.log_snr(t).type(dtype)
        log_snr_padded_dim = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr_padded_dim)
        return alpha * x_start + sigma * noise, log_snr, alpha, sigma

    def q_sample_from_to(self, x_from, from_t, to_t, noise=None):
        shape, device, dtype = x_from.shape, x_from.device, x_from.dtype
        batch = shape[0]
        if isinstance(from_t, float):
            from_t = torch.full((batch,), from_t, device=device, dtype=dtype)
        if isinstance(to_t, float):
            to_t = torch.full((batch,), to_t, device=device, dtype=dtype)
        noise = default(noise, lambda : torch.randn_like(x_from))
        log_snr = self.log_snr(from_t)
        log_snr_padded_dim = right_pad_dims_to(x_from, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr_padded_dim)
        log_snr_to = self.log_snr(to_t)
        log_snr_padded_dim_to = right_pad_dims_to(x_from, log_snr_to)
        alpha_to, sigma_to = log_snr_to_alpha_sigma(log_snr_padded_dim_to)
        return x_from * (alpha_to / alpha) + noise * (sigma_to * alpha - sigma * alpha_to) / alpha

    def predict_start_from_v(self, x_t, t, v):
        log_snr = self.log_snr(t)
        log_snr = right_pad_dims_to(x_t, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        return alpha * x_t - sigma * v

    def predict_start_from_noise(self, x_t, t, noise):
        log_snr = self.log_snr(t)
        log_snr = right_pad_dims_to(x_t, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        return (x_t - sigma * noise) / alpha.clamp(min=1e-08)


Hparams_fields = ['num_sample_steps', 'sigma_min', 'sigma_max', 'sigma_data', 'rho', 'P_mean', 'P_std', 'S_churn', 'S_tmin', 'S_tmax', 'S_noise']


Hparams = namedtuple('Hparams', Hparams_fields)


class NullUnet(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lowres_cond = False
        self.dummy_parameter = nn.Parameter(torch.tensor([0.0]))

    def cast_model_parameters(self, *args, **kwargs):
        return self

    def forward(self, x, *args, **kwargs):
        return x


def cast_tuple(val, length=1):
    if isinstance(val, list):
        val = tuple(val)
    return val if isinstance(val, tuple) else (val,) * length


def Conv2d(dim_in, dim_out, kernel, stride=1, padding=0, **kwargs):
    kernel = cast_tuple(kernel, 2)
    stride = cast_tuple(stride, 2)
    padding = cast_tuple(padding, 2)
    if len(kernel) == 2:
        kernel = 1, *kernel
    if len(stride) == 2:
        stride = 1, *stride
    if len(padding) == 2:
        padding = 0, *padding
    return nn.Conv3d(dim_in, dim_out, kernel, stride=stride, padding=padding, **kwargs)


class CrossEmbedLayer(nn.Module):

    def __init__(self, dim_in, kernel_sizes, dim_out=None, stride=2):
        super().__init__()
        assert all([*map(lambda t: t % 2 == stride % 2, kernel_sizes)])
        dim_out = default(dim_out, dim_in)
        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)
        dim_scales = [int(dim_out / 2 ** i) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]
        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(Conv2d(dim_in, dim_scale, kernel, stride=stride, padding=(kernel - stride) // 2))

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim=1)


def Downsample(dim, dim_out=None):
    dim_out = default(dim_out, dim)
    return nn.Sequential(Rearrange('b c f (h p1) (w p2) -> b (c p1 p2) f h w', p1=2, p2=2), Conv2d(dim * 4, dim_out, 1))


class Identity(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


class LearnedSinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class ChanLayerNorm(nn.Module):

    def __init__(self, dim, stable=False):
        super().__init__()
        self.stable = stable
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        if self.stable:
            x = x / x.amax(dim=1, keepdim=True).detach()
        eps = 1e-05 if x.dtype == torch.float32 else 0.001
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))


class TimeTokenShift(nn.Module):

    def forward(self, x):
        if x.ndim != 5:
            return x
        x, x_shift = x.chunk(2, dim=1)
        x_shift = F.pad(x_shift, (0, 0, 0, 0, 1, -1), value=0.0)
        return torch.cat((x, x_shift), dim=1)


def ChanFeedForward(dim, mult=2, time_token_shift=True):
    hidden_dim = int(dim * mult)
    return Sequential(ChanLayerNorm(dim), Conv2d(dim, hidden_dim, 1, bias=False), nn.GELU(), TimeTokenShift() if time_token_shift else None, ChanLayerNorm(hidden_dim), Conv2d(hidden_dim, dim, 1, bias=False))


class LinearAttention(nn.Module):

    def __init__(self, dim, dim_head=32, heads=8, dropout=0.05, context_dim=None, **kwargs):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = ChanLayerNorm(dim)
        self.nonlin = nn.SiLU()
        self.to_q = nn.Sequential(nn.Dropout(dropout), Conv2d(dim, inner_dim, 1, bias=False), Conv2d(inner_dim, inner_dim, 3, bias=False, padding=1, groups=inner_dim))
        self.to_k = nn.Sequential(nn.Dropout(dropout), Conv2d(dim, inner_dim, 1, bias=False), Conv2d(inner_dim, inner_dim, 3, bias=False, padding=1, groups=inner_dim))
        self.to_v = nn.Sequential(nn.Dropout(dropout), Conv2d(dim, inner_dim, 1, bias=False), Conv2d(inner_dim, inner_dim, 3, bias=False, padding=1, groups=inner_dim))
        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, inner_dim * 2, bias=False)) if exists(context_dim) else None
        self.to_out = nn.Sequential(Conv2d(inner_dim, dim, 1, bias=False), ChanLayerNorm(dim))

    def forward(self, fmap, context=None):
        h, x, y = self.heads, *fmap.shape[-2:]
        fmap = self.norm(fmap)
        q, k, v = map(lambda fn: fn(fmap), (self.to_q, self.to_k, self.to_v))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h=h), (q, k, v))
        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim=-1)
            ck, cv = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (ck, cv))
            k = torch.cat((k, ck), dim=-2)
            v = torch.cat((v, cv), dim=-2)
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)
        q = q * self.scale
        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h=h, x=x, y=y)
        out = self.nonlin(out)
        return self.to_out(out)


class LinearAttentionTransformerBlock(nn.Module):

    def __init__(self, dim, *, depth=1, heads=8, dim_head=32, ff_mult=2, ff_time_token_shift=True, context_dim=None, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([LinearAttention(dim=dim, heads=heads, dim_head=dim_head, context_dim=context_dim), ChanFeedForward(dim=dim, mult=ff_mult, time_token_shift=ff_time_token_shift)]))

    def forward(self, x, context=None):
        for attn, ff in self.layers:
            x = attn(x, context=context) + x
            x = ff(x) + x
        return x


class Parallel(nn.Module):

    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        outputs = [fn(x) for fn in self.fns]
        return sum(outputs)


class LayerNorm(nn.Module):

    def __init__(self, dim, stable=False):
        super().__init__()
        self.stable = stable
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if self.stable:
            x = x / x.amax(dim=-1, keepdim=True).detach()
        eps = 1e-05 if x.dtype == torch.float32 else 0.001
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


def FeedForward(dim, mult=2):
    hidden_dim = int(dim * mult)
    return nn.Sequential(LayerNorm(dim), nn.Linear(dim, hidden_dim, bias=False), nn.GELU(), LayerNorm(hidden_dim), nn.Linear(hidden_dim, dim, bias=False))


def l2norm(t):
    return F.normalize(t, dim=-1)


class PerceiverAttention(nn.Module):

    def __init__(self, *, dim, dim_head=64, heads=8, scale=8):
        super().__init__()
        self.scale = scale
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias=False), nn.LayerNorm(dim))

    def forward(self, x, latents, mask=None):
        x = self.norm(x)
        latents = self.norm_latents(latents)
        b, h = x.shape[0], self.heads
        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale
        sim = einsum('... i d, ... j d  -> ... i j', q, k) * self.scale
        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = F.pad(mask, (0, latents.shape[-2]), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)
        attn = sim.softmax(dim=-1)
        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


def masked_mean(t, *, dim, mask=None):
    if not exists(mask):
        return t.mean(dim=dim)
    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, 'b n -> b n 1')
    masked_t = t.masked_fill(~mask, 0.0)
    return masked_t.sum(dim=dim) / denom.clamp(min=1e-05)


class PerceiverResampler(nn.Module):

    def __init__(self, *, dim, depth, dim_head=64, heads=8, num_latents=64, num_latents_mean_pooled=4, max_seq_len=512, ff_mult=4):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.to_latents_from_mean_pooled_seq = None
        if num_latents_mean_pooled > 0:
            self.to_latents_from_mean_pooled_seq = nn.Sequential(LayerNorm(dim), nn.Linear(dim, dim * num_latents_mean_pooled), Rearrange('b (n d) -> b n d', n=num_latents_mean_pooled))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads), FeedForward(dim=dim, mult=ff_mult)]))

    def forward(self, x, mask=None):
        n, device = x.shape[1], x.device
        pos_emb = self.pos_emb(torch.arange(n, device=device))
        x_with_pos = x + pos_emb
        latents = repeat(self.latents, 'n d -> b n d', b=x.shape[0])
        if exists(self.to_latents_from_mean_pooled_seq):
            meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)
        for attn, ff in self.layers:
            latents = attn(x_with_pos, latents, mask=mask) + latents
            latents = ff(latents) + latents
        return latents


class PixelShuffleUpsample(nn.Module):

    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = Conv2d(dim, dim_out * 4, 1)
        self.net = nn.Sequential(conv, nn.SiLU())
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, f, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, f, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        out = self.net(x)
        frames = x.shape[2]
        out = rearrange(out, 'b c f h w -> (b f) c h w')
        out = self.pixel_shuffle(out)
        return rearrange(out, '(b f) c h w -> b c f h w', f=frames)


class Always:

    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val


class ChanRMSNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.scale * self.gamma


class Conv3d(nn.Module):

    def __init__(self, dim, dim_out=None, kernel_size=3, *, temporal_kernel_size=None, **kwargs):
        super().__init__()
        dim_out = default(dim_out, dim)
        temporal_kernel_size = default(temporal_kernel_size, kernel_size)
        self.spatial_conv = nn.Conv2d(dim, dim_out, kernel_size=kernel_size, padding=kernel_size // 2)
        self.temporal_conv = nn.Conv1d(dim_out, dim_out, kernel_size=temporal_kernel_size) if kernel_size > 1 else None
        self.kernel_size = kernel_size
        if exists(self.temporal_conv):
            nn.init.dirac_(self.temporal_conv.weight.data)
            nn.init.zeros_(self.temporal_conv.bias.data)

    def forward(self, x, ignore_time=False):
        b, c, *_, h, w = x.shape
        is_video = x.ndim == 5
        ignore_time &= is_video
        if is_video:
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        x = self.spatial_conv(x)
        if is_video:
            x = rearrange(x, '(b f) c h w -> b c f h w', b=b)
        if ignore_time or not exists(self.temporal_conv):
            return x
        x = rearrange(x, 'b c f h w -> (b h w) c f')
        if self.kernel_size > 1:
            x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.temporal_conv(x)
        x = rearrange(x, '(b h w) c f -> b c f h w', h=h, w=w)
        return x


class Block(nn.Module):

    def __init__(self, dim, dim_out, norm=True):
        super().__init__()
        self.norm = ChanRMSNorm(dim) if norm else Identity()
        self.activation = nn.SiLU()
        self.project = Conv3d(dim, dim_out, 3, padding=1)

    def forward(self, x, scale_shift=None, ignore_time=False):
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.activation(x)
        return self.project(x, ignore_time=ignore_time)


class CrossAttention(nn.Module):

    def __init__(self, dim, *, context_dim=None, dim_head=64, heads=8, norm_context=False, scale=8):
        super().__init__()
        self.scale = scale
        self.heads = heads
        inner_dim = dim_head * heads
        context_dim = default(context_dim, dim)
        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else Identity()
        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias=False), LayerNorm(dim))

    def forward(self, x, context, mask=None):
        b, n, device = *x.shape[:2], x.device
        x = self.norm(x)
        context = self.norm_context(context)
        q, k, v = self.to_q(x), *self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        nk, nv = map(lambda t: repeat(t, 'd -> b h 1 d', h=self.heads, b=b), self.null_kv.unbind(dim=-2))
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)
        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        max_neg_value = -torch.finfo(sim.dtype).max
        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)
        attn = sim.softmax(dim=-1, dtype=torch.float32)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class GlobalContext(nn.Module):
    """ basically a superior form of squeeze-excitation that is attention-esque """

    def __init__(self, *, dim_in, dim_out):
        super().__init__()
        self.to_k = Conv2d(dim_in, 1, 1)
        hidden_dim = max(3, dim_out // 2)
        self.net = nn.Sequential(Conv2d(dim_in, hidden_dim, 1), nn.SiLU(), Conv2d(hidden_dim, dim_out, 1), nn.Sigmoid())

    def forward(self, x):
        context = self.to_k(x)
        x, context = map(lambda t: rearrange(t, 'b n ... -> b n (...)'), (x, context))
        out = einsum('b i n, b c n -> b c i', context.softmax(dim=-1), x)
        out = rearrange(out, '... -> ... 1 1')
        return self.net(out)


class LinearCrossAttention(CrossAttention):

    def forward(self, x, context, mask=None):
        b, n, device = *x.shape[:2], x.device
        x = self.norm(x)
        context = self.norm_context(context)
        q, k, v = self.to_q(x), *self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v))
        nk, nv = map(lambda t: repeat(t, 'd -> (b h) 1 d', h=self.heads, b=b), self.null_kv.unbind(dim=-2))
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)
        max_neg_value = -torch.finfo(x.dtype).max
        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, 'b n -> b n 1')
            k = k.masked_fill(~mask, max_neg_value)
            v = v.masked_fill(~mask, 0.0)
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)
        q = q * self.scale
        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)
        return self.to_out(out)


class ResnetBlock(nn.Module):

    def __init__(self, dim, dim_out, *, cond_dim=None, time_cond_dim=None, linear_attn=False, use_gca=False, squeeze_excite=False, **attn_kwargs):
        super().__init__()
        self.time_mlp = None
        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_cond_dim, dim_out * 2))
        self.cross_attn = None
        if exists(cond_dim):
            attn_klass = CrossAttention if not linear_attn else LinearCrossAttention
            self.cross_attn = attn_klass(dim=dim_out, context_dim=cond_dim, **attn_kwargs)
        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.gca = GlobalContext(dim_in=dim_out, dim_out=dim_out) if use_gca else Always(1)
        self.res_conv = Conv2d(dim, dim_out, 1) if dim != dim_out else Identity()

    def forward(self, x, time_emb=None, cond=None, ignore_time=False):
        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, ignore_time=ignore_time)
        if exists(self.cross_attn):
            assert exists(cond)
            h = rearrange(h, 'b c ... -> b ... c')
            h, ps = pack([h], 'b * c')
            h = self.cross_attn(h, context=cond) + h
            h, = unpack(h, ps, 'b * c')
            h = rearrange(h, 'b ... c -> b c ...')
        h = self.block2(h, scale_shift=scale_shift, ignore_time=ignore_time)
        h = h * self.gca(h)
        return h + self.res_conv(x)


class DynamicPositionBias(nn.Module):

    def __init__(self, dim, *, heads, depth):
        super().__init__()
        self.mlp = nn.ModuleList([])
        self.mlp.append(nn.Sequential(nn.Linear(1, dim), LayerNorm(dim), nn.SiLU()))
        for _ in range(max(depth - 1, 0)):
            self.mlp.append(nn.Sequential(nn.Linear(dim, dim), LayerNorm(dim), nn.SiLU()))
        self.mlp.append(nn.Linear(dim, heads))

    def forward(self, n, device, dtype):
        i = torch.arange(n, device=device)
        j = torch.arange(n, device=device)
        indices = rearrange(i, 'i -> i 1') - rearrange(j, 'j -> 1 j')
        indices += n - 1
        pos = torch.arange(-n + 1, n, device=device, dtype=dtype)
        pos = rearrange(pos, '... -> ... 1')
        for layer in self.mlp:
            pos = layer(pos)
        bias = pos[indices]
        bias = rearrange(bias, 'i j h -> h i j')
        return bias


class Attention(nn.Module):

    def __init__(self, dim, *, dim_head=64, heads=8, causal=False, context_dim=None, rel_pos_bias=False, rel_pos_bias_mlp_depth=2, init_zero=False, scale=8):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.rel_pos_bias = DynamicPositionBias(dim=dim, heads=heads, depth=rel_pos_bias_mlp_depth) if rel_pos_bias else None
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = LayerNorm(dim)
        self.null_attn_bias = nn.Parameter(torch.randn(heads))
        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))
        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, dim_head * 2)) if exists(context_dim) else None
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias=False), LayerNorm(dim))
        if init_zero:
            nn.init.zeros_(self.to_out[-1].g)

    def forward(self, x, context=None, mask=None, attn_bias=None):
        b, n, device = *x.shape[:2], x.device
        x = self.norm(x)
        q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        nk, nv = map(lambda t: repeat(t, 'd -> b 1 d', b=b), self.null_kv.unbind(dim=-2))
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)
        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim=-1)
            k = torch.cat((ck, k), dim=-2)
            v = torch.cat((cv, v), dim=-2)
        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale
        sim = einsum('b h i d, b j d -> b h i j', q, k) * self.scale
        if not exists(attn_bias) and exists(self.rel_pos_bias):
            attn_bias = self.rel_pos_bias(n, device=device, dtype=q.dtype)
        if exists(attn_bias):
            null_attn_bias = repeat(self.null_attn_bias, 'h -> h n 1', n=n)
            attn_bias = torch.cat((null_attn_bias, attn_bias), dim=-1)
            sim = sim + attn_bias
        max_neg_value = -torch.finfo(sim.dtype).max
        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, max_neg_value)
        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):

    def __init__(self, dim, *, depth=1, heads=8, dim_head=32, ff_mult=2, ff_time_token_shift=True, context_dim=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([Attention(dim=dim, heads=heads, dim_head=dim_head, context_dim=context_dim), ChanFeedForward(dim=dim, mult=ff_mult, time_token_shift=ff_time_token_shift)]))

    def forward(self, x, context=None):
        for attn, ff in self.layers:
            x = rearrange(x, 'b c ... -> b ... c')
            x, ps = pack([x], 'b * c')
            x = attn(x, context=context) + x
            x, = unpack(x, ps, 'b * c')
            x = rearrange(x, 'b ... c -> b c ...')
            x = ff(x) + x
        return x


def Upsample(dim, dim_out=None):
    dim_out = default(dim_out, dim)
    return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), Conv2d(dim, dim_out, 3, padding=1))


def resize_video_to(video, target_image_size, target_frames=None, clamp_range=None, mode='nearest'):
    orig_video_size = video.shape[-1]
    frames = video.shape[2]
    target_frames = default(target_frames, frames)
    target_shape = target_frames, target_image_size, target_image_size
    if tuple(video.shape[-3:]) == target_shape:
        return video
    out = F.interpolate(video, target_shape, mode=mode)
    if exists(clamp_range):
        out = out.clamp(*clamp_range)
    return out


class UpsampleCombiner(nn.Module):

    def __init__(self, dim, *, enabled=False, dim_ins=tuple(), dim_outs=tuple()):
        super().__init__()
        dim_outs = cast_tuple(dim_outs, len(dim_ins))
        assert len(dim_ins) == len(dim_outs)
        self.enabled = enabled
        if not self.enabled:
            self.dim_out = dim
            return
        self.fmap_convs = nn.ModuleList([Block(dim_in, dim_out) for dim_in, dim_out in zip(dim_ins, dim_outs)])
        self.dim_out = dim + (sum(dim_outs) if len(dim_outs) > 0 else 0)

    def forward(self, x, fmaps=None):
        target_size = x.shape[-1]
        fmaps = default(fmaps, tuple())
        if not self.enabled or len(fmaps) == 0 or len(self.fmap_convs) == 0:
            return x
        fmaps = [resize_video_to(fmap, target_size) for fmap in fmaps]
        outs = [conv(fmap) for fmap, conv in zip(fmaps, self.fmap_convs)]
        return torch.cat((x, *outs), dim=1)


T5_CONFIGS = {}


def get_encoded_dim(name):
    if name not in T5_CONFIGS:
        config = T5Config.from_pretrained(name)
        T5_CONFIGS[name] = dict(config=config)
    elif 'config' in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]['config']
    elif 'model' in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]['model'].config
    else:
        assert False
    return config.d_model


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner


print_once = once(print)


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def pack_one_with_inverse(x, pattern):
    packed, packed_shape = pack([x], pattern)

    def inverse(x, inverse_pattern=None):
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(x, packed_shape, inverse_pattern)[0]
    return packed, inverse


def project(x, y):
    x, inverse = pack_one_with_inverse(x, 'b *')
    y, _ = pack_one_with_inverse(y, 'b *')
    dtype = x.dtype
    x, y = x.double(), y.double()
    unit = F.normalize(y, dim=-1)
    parallel = (x * unit).sum(dim=-1, keepdim=True) * unit
    orthogonal = x - parallel
    return inverse(parallel), inverse(orthogonal)


def resize_image_to(image, target_image_size, clamp_range=None, mode='nearest'):
    orig_image_size = image.shape[-1]
    if orig_image_size == target_image_size:
        return image
    out = F.interpolate(image, target_image_size, mode=mode)
    if exists(clamp_range):
        out = out.clamp(*clamp_range)
    return out


def zero_init_(m):
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)


class Pad(nn.Module):

    def __init__(self, padding, value=0.0):
        super().__init__()
        self.padding = padding
        self.value = value

    def forward(self, x):
        return F.pad(x, self.padding, value=self.value)


class RearrangeTimeCentric(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        x = rearrange(x, 'b c f ... -> b ... f c')
        x, ps = pack([x], '* f c')
        x = self.fn(x)
        x, = unpack(x, ps, '* f c')
        x = rearrange(x, 'b ... f c -> b c f ...')
        return x


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


def TemporalDownsample(dim, dim_out=None, stride=2):
    dim_out = default(dim_out, dim)
    return nn.Sequential(Rearrange('b c (f p) h w -> b (c p) f h w', p=stride), Conv2d(dim * stride, dim_out, 1))


class TemporalPixelShuffleUpsample(nn.Module):

    def __init__(self, dim, dim_out=None, stride=2):
        super().__init__()
        self.stride = stride
        dim_out = default(dim_out, dim)
        conv = nn.Conv1d(dim, dim_out * stride, 1)
        self.net = nn.Sequential(conv, nn.SiLU())
        self.pixel_shuffle = Rearrange('b (c r) n -> b c (n r)', r=stride)
        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, f = conv.weight.shape
        conv_weight = torch.empty(o // self.stride, i, f)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r=self.stride)
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b h w) c f')
        out = self.net(x)
        out = self.pixel_shuffle(out)
        return rearrange(out, '(b h w) c f -> b c f h w', h=h, w=w)


def divisible_by(numer, denom):
    return numer % denom == 0


def calc_all_frame_dims(downsample_factors: 'List[int]', frames):
    if not exists(frames):
        return (tuple(),) * len(downsample_factors)
    all_frame_dims = []
    for divisor in downsample_factors:
        assert divisible_by(frames, divisor)
        all_frame_dims.append((frames // divisor,))
    return all_frame_dims


def cast_uint8_images_to_float(images):
    if not images.dtype == torch.uint8:
        return images
    return images / 255


def compact(input_dict):
    return {key: value for key, value in input_dict.items() if exists(value)}


def eval_decorator(fn):

    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


def first(arr, d=None):
    if len(arr) == 0:
        return d
    return arr[0]


def identity(t, *args, **kwargs):
    return t


def maybe_transform_dict_key(input_dict, key, fn):
    if key not in input_dict:
        return input_dict
    copied_dict = input_dict.copy()
    copied_dict[key] = fn(copied_dict[key])
    return copied_dict


def module_device(module):
    return next(module.parameters()).device


def normalize_neg_one_to_one(img):
    return img * 2 - 1


def safe_get_tuple_index(tup, index, default=None):
    if len(tup) <= index:
        return default
    return tup[index]


def scale_video_time(video, downsample_scale=1, mode='nearest'):
    if downsample_scale == 1:
        return video
    image_size, frames = video.shape[-1], video.shape[-3]
    assert divisible_by(frames, downsample_scale), f'trying to temporally downsample a conditioning video frames of length {frames} by {downsample_scale}, however it is not neatly divisible'
    target_frames = frames // downsample_scale
    resized_video = resize_video_to(video, image_size, target_frames=target_frames, mode=mode)
    return resized_video


def get_model(name):
    model = T5EncoderModel.from_pretrained(name)
    return model


MAX_LENGTH = 256


def get_tokenizer(name):
    tokenizer = T5Tokenizer.from_pretrained(name, model_max_length=MAX_LENGTH)
    return tokenizer


def get_model_and_tokenizer(name):
    global T5_CONFIGS
    if name not in T5_CONFIGS:
        T5_CONFIGS[name] = dict()
    if 'model' not in T5_CONFIGS[name]:
        T5_CONFIGS[name]['model'] = get_model(name)
    if 'tokenizer' not in T5_CONFIGS[name]:
        T5_CONFIGS[name]['tokenizer'] = get_tokenizer(name)
    return T5_CONFIGS[name]['model'], T5_CONFIGS[name]['tokenizer']


def t5_encode_tokenized_text(token_ids, attn_mask=None, pad_id=None, name=DEFAULT_T5_NAME):
    assert exists(attn_mask) or exists(pad_id)
    t5, _ = get_model_and_tokenizer(name)
    attn_mask = default(attn_mask, lambda : (token_ids != pad_id).long())
    t5.eval()
    with torch.no_grad():
        output = t5(input_ids=token_ids, attention_mask=attn_mask)
        encoded_text = output.last_hidden_state.detach()
    attn_mask = attn_mask.bool()
    encoded_text = encoded_text.masked_fill(~rearrange(attn_mask, '... -> ... 1'), 0.0)
    return encoded_text


def t5_tokenize(texts: 'List[str]', name=DEFAULT_T5_NAME):
    t5, tokenizer = get_model_and_tokenizer(name)
    if torch.cuda.is_available():
        t5 = t5
    device = next(t5.parameters()).device
    encoded = tokenizer.batch_encode_plus(texts, return_tensors='pt', padding='longest', max_length=MAX_LENGTH, truncation=True)
    input_ids = encoded.input_ids
    attn_mask = encoded.attention_mask
    return input_ids, attn_mask


def t5_encode_text(texts: 'List[str]', name=DEFAULT_T5_NAME, return_attn_mask=False):
    token_ids, attn_mask = t5_tokenize(texts, name=name)
    encoded_text = t5_encode_tokenized_text(token_ids, attn_mask=attn_mask, name=name)
    if return_attn_mask:
        attn_mask = attn_mask.bool()
        return encoded_text, attn_mask
    return encoded_text


def unnormalize_zero_to_one(normed_img):
    return (normed_img + 1) * 0.5


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


def pad_tuple_to_length(t, length, fillvalue=None):
    remain_length = length - len(t)
    if remain_length <= 0:
        return t
    return *t, *((fillvalue,) * remain_length)


def cast_torch_tensor(fn, cast_fp16=False):

    @wraps(fn)
    def inner(model, *args, **kwargs):
        device = kwargs.pop('_device', model.device)
        cast_device = kwargs.pop('_cast_device', True)
        should_cast_fp16 = cast_fp16 and model.cast_half_at_training
        kwargs_keys = kwargs.keys()
        all_args = *args, *kwargs.values()
        split_kwargs_index = len(all_args) - len(kwargs_keys)
        all_args = tuple(map(lambda t: torch.from_numpy(t) if exists(t) and isinstance(t, np.ndarray) else t, all_args))
        if cast_device:
            all_args = tuple(map(lambda t: t if exists(t) and isinstance(t, torch.Tensor) else t, all_args))
        if should_cast_fp16:
            all_args = tuple(map(lambda t: t.half() if exists(t) and isinstance(t, torch.Tensor) and t.dtype != torch.bool else t, all_args))
        args, kwargs_values = all_args[:split_kwargs_index], all_args[split_kwargs_index:]
        kwargs = dict(tuple(zip(kwargs_keys, kwargs_values)))
        out = fn(model, *args, **kwargs)
        return out
    return inner


def cycle(dl):
    while True:
        for data in dl:
            yield data


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


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def find_first(cond, arr):
    for el in arr:
        if cond(el):
            return el
    return None


def split_iterable(it, split_size):
    accum = []
    for ind in range(ceil(len(it) / split_size)):
        start_index = ind * split_size
        accum.append(it[start_index:start_index + split_size])
    return accum


def split(t, split_size=None):
    if not exists(split_size):
        return t
    if isinstance(t, torch.Tensor):
        return t.split(split_size, dim=0)
    if isinstance(t, Iterable):
        return split_iterable(t, split_size)
    return TypeError


def split_args_and_kwargs(*args, split_size=None, **kwargs):
    all_args = *args, *kwargs.values()
    len_all_args = len(all_args)
    first_tensor = find_first(lambda t: isinstance(t, torch.Tensor), all_args)
    assert exists(first_tensor)
    batch_size = len(first_tensor)
    split_size = default(split_size, batch_size)
    num_chunks = ceil(batch_size / split_size)
    dict_len = len(kwargs)
    dict_keys = kwargs.keys()
    split_kwargs_index = len_all_args - dict_len
    split_all_args = [(split(arg, split_size=split_size) if exists(arg) and isinstance(arg, (torch.Tensor, Iterable)) else (arg,) * num_chunks) for arg in all_args]
    chunk_sizes = num_to_groups(batch_size, split_size)
    for chunk_size, *chunked_all_args in tuple(zip(chunk_sizes, *split_all_args)):
        chunked_args, chunked_kwargs_values = chunked_all_args[:split_kwargs_index], chunked_all_args[split_kwargs_index:]
        chunked_kwargs = dict(tuple(zip(dict_keys, chunked_kwargs_values)))
        chunk_size_frac = chunk_size / batch_size
        yield chunk_size_frac, (chunked_args, chunked_kwargs)


def imagen_sample_in_chunks(fn):

    @wraps(fn)
    def inner(self, *args, max_batch_size=None, **kwargs):
        if not exists(max_batch_size):
            return fn(self, *args, **kwargs)
        if self.imagen.unconditional:
            batch_size = kwargs.get('batch_size')
            batch_sizes = num_to_groups(batch_size, max_batch_size)
            outputs = [fn(self, *args, **{**kwargs, 'batch_size': sub_batch_size}) for sub_batch_size in batch_sizes]
        else:
            outputs = [fn(self, *chunked_args, **chunked_kwargs) for _, (chunked_args, chunked_kwargs) in split_args_and_kwargs(*args, split_size=max_batch_size, **kwargs)]
        if isinstance(outputs[0], torch.Tensor):
            return torch.cat(outputs, dim=0)
        return list(map(lambda t: torch.cat(t, dim=0), list(zip(*outputs))))
    return inner


def restore_parts(state_dict_target, state_dict_from):
    for name, param in state_dict_from.items():
        if name not in state_dict_target:
            continue
        if param.size() == state_dict_target[name].size():
            state_dict_target[name].copy_(param)
        else:
            None
    return state_dict_target


def url_to_bucket(url):
    if '://' not in url:
        return url
    _, suffix = url.split('://')
    if prefix in {'gs', 's3'}:
        return suffix.split('/')[0]
    else:
        raise ValueError(f'storage type prefix "{prefix}" is not supported yet')


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ChanLayerNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ChanRMSNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CrossEmbedLayer,
     lambda: ([], {'dim_in': 4, 'kernel_sizes': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NullUnet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Parallel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Residual,
     lambda: ([], {'fn': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TimeTokenShift,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (UpsampleCombiner,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

