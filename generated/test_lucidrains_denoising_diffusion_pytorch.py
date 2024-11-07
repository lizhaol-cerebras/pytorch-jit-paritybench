
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


from functools import wraps


from collections import namedtuple


import torch


from torch import nn


from torch import einsum


import torch.nn.functional as F


import math


import copy


from random import random


from functools import partial


from torch.amp import autocast


from torch import sqrt


from torch.special import expm1


from torch.nn import Module


from torch.nn import ModuleList


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.optim import Adam


from torchvision import transforms as T


from torchvision import utils


from scipy.optimize import linear_sum_assignment


from torch import Tensor


from math import sqrt


import numpy as np


from torch.nn.functional import adaptive_avg_pool2d


from math import ceil


from torch.optim.lr_scheduler import LambdaLR


from typing import Optional


from typing import Union


from typing import Tuple


from math import pi


from math import log as ln


from inspect import isfunction


AttentionConfig = namedtuple('AttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


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


class Attend(nn.Module):

    def __init__(self, dropout=0.0, flash=False, scale=None):
        super().__init__()
        self.dropout = dropout
        self.scale = scale
        self.attn_dropout = nn.Dropout(dropout)
        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'
        self.cpu_config = AttentionConfig(True, True, True)
        self.cuda_config = None
        if not torch.cuda.is_available() or not flash:
            return
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
        device_version = version.parse(f'{device_properties.major}.{device_properties.minor}')
        if device_version > version.parse('8.0'):
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device
        if exists(self.scale):
            default_scale = q.shape[-1]
            q = q * (self.scale / default_scale)
        q, k, v = map(lambda t: t.contiguous(), (q, k, v))
        config = self.cuda_config if is_cuda else self.cpu_config
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return out

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device
        if self.flash:
            return self.flash_attn(q, k, v)
        scale = default(self.scale, q.shape[-1] ** -0.5)
        sim = einsum(f'b h i d, b h j d -> b h i j', q, k) * scale
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = einsum(f'b h i j, b h j d -> b h i d', attn, v)
        return out


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def append_dims(t, dims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * dims))


class RMSNorm(nn.Module):

    def __init__(self, dim, scale=True, normalize_dim=2):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim)) if scale else 1
        self.scale = scale
        self.normalize_dim = normalize_dim

    def forward(self, x):
        normalize_dim = self.normalize_dim
        scale = append_dims(self.g, x.ndim - self.normalize_dim - 1) if self.scale else 1
        return F.normalize(x, dim=normalize_dim) * scale * x.shape[normalize_dim] ** 0.5


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class SinusoidalPosEmb(Module):

    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def divisible_by(numer, denom):
    return numer % denom == 0


class RandomOrLearnedSinusoidalPosEmb(Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class Block(nn.Module):

    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out, normalize_dim=1)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):

    def __init__(self, dim, dim_out, *, time_emb_dim=None):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None
        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.norm = RMSNorm(dim, normalize_dim=1)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), RMSNorm(dim, normalize_dim=1))

    def forward(self, x):
        residual = x
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out) + residual


def l2norm(t):
    return F.normalize(t, dim=-1)


class Attention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32, scale=8, dropout=0.0):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads
        self.norm = RMSNorm(dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def Downsample(dim, dim_out=None, factor=2):
    return nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=factor, p2=factor), nn.Conv2d(dim * factor ** 2, default(dim_out, dim), 1))


class Upsample(nn.Module):

    def __init__(self, dim, dim_out=None, factor=2):
        super().__init__()
        self.factor = factor
        self.factor_squared = factor ** 2
        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * self.factor_squared, 1)
        self.net = nn.Sequential(conv, nn.SiLU(), nn.PixelShuffle(factor))
        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // self.factor_squared, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r=self.factor_squared)
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)


def cast_tuple(t, l=1):
    return (t,) * l if not isinstance(t, tuple) else t


class Unet(Module):

    def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=3, self_condition=False, learned_variance=False, learned_sinusoidal_cond=False, random_fourier_features=False, learned_sinusoidal_dim=16, sinusoidal_pos_emb_theta=10000, attn_dim_head=32, attn_heads=4, full_attn=None, flash_attn=False):
        super().__init__()
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        time_dim = dim * 4
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim
        self.time_mlp = nn.Sequential(sinu_pos_emb, nn.Linear(fourier_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))
        if not full_attn:
            full_attn = *((False,) * (len(dim_mults) - 1)), True
        num_stages = len(dim_mults)
        full_attn = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)
        assert len(full_attn) == len(dim_mults)
        FullAttention = partial(Attention, flash=flash_attn)
        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)
        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= num_resolutions - 1
            attn_klass = FullAttention if layer_full_attn else LinearAttention
            self.downs.append(ModuleList([ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim), ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim), attn_klass(dim_in, dim_head=layer_attn_dim_head, heads=layer_attn_heads), Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)]))
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = FullAttention(mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1])
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == len(in_out) - 1
            attn_klass = FullAttention if layer_full_attn else LinearAttention
            self.ups.append(ModuleList([ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim), ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim), attn_klass(dim_out, dim_head=layer_attn_dim_head, heads=layer_attn_heads), Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)]))
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.final_res_block = ResnetBlock(init_dim * 2, init_dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond=None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda : torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x) + x
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x) + x
            x = upsample(x)
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def logsnr_schedule_cosine(t, logsnr_min=-15, logsnr_max=15):
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * log(torch.tan(t_min + t * (t_max - t_min)))


def logsnr_schedule_shifted(fn, image_d, noise_d):
    shift = 2 * math.log(noise_d / image_d)

    @wraps(fn)
    def inner(*args, **kwargs):
        nonlocal shift
        return fn(*args, **kwargs) + shift
    return inner


def logsnr_schedule_interpolated(fn, image_d, noise_d_low, noise_d_high):
    logsnr_low_fn = logsnr_schedule_shifted(fn, image_d, noise_d_low)
    logsnr_high_fn = logsnr_schedule_shifted(fn, image_d, noise_d_high)

    @wraps(fn)
    def inner(t, *args, **kwargs):
        nonlocal logsnr_low_fn
        nonlocal logsnr_high_fn
        return t * logsnr_low_fn(t, *args, **kwargs) + (1 - t) * logsnr_high_fn(t, *args, **kwargs)
    return inner


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


class GaussianDiffusion(nn.Module):

    def __init__(self, model: 'UViT', *, image_size, channels=3, pred_objective='v', noise_schedule=logsnr_schedule_cosine, noise_d=None, noise_d_low=None, noise_d_high=None, num_sample_steps=500, clip_sample_denoised=True, min_snr_loss_weight=True, min_snr_gamma=5):
        super().__init__()
        assert pred_objective in {'v', 'eps'}, 'whether to predict v-space (progressive distillation paper) or noise'
        self.model = model
        self.channels = channels
        self.image_size = image_size
        self.pred_objective = pred_objective
        assert not all([*map(exists, (noise_d, noise_d_low, noise_d_high))]), 'you must either set noise_d for shifted schedule, or noise_d_low and noise_d_high for shifted and interpolated schedule'
        self.log_snr = noise_schedule
        if exists(noise_d):
            self.log_snr = logsnr_schedule_shifted(self.log_snr, image_size, noise_d)
        if exists(noise_d_low) or exists(noise_d_high):
            assert exists(noise_d_low) and exists(noise_d_high), 'both noise_d_low and noise_d_high must be set'
            self.log_snr = logsnr_schedule_interpolated(self.log_snr, image_size, noise_d_low, noise_d_high)
        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised
        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

    @property
    def device(self):
        return next(self.model.parameters()).device

    def p_mean_variance(self, x, time, time_next):
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)
        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()
        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))
        batch_log_snr = repeat(log_snr, ' -> b', b=x.shape[0])
        pred = self.model(x, batch_log_snr)
        if self.pred_objective == 'v':
            x_start = alpha * x - sigma * pred
        elif self.pred_objective == 'eps':
            x_start = (x - sigma * pred) / alpha
        x_start.clamp_(-1.0, 1.0)
        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
        posterior_variance = squared_sigma_next * c
        return model_mean, posterior_variance

    @torch.no_grad()
    def p_sample(self, x, time, time_next):
        batch, *_, device = *x.shape, x.device
        model_mean, model_variance = self.p_mean_variance(x=x, time=time, time_next=time_next)
        if time_next == 0:
            return model_mean
        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch = shape[0]
        img = torch.randn(shape, device=self.device)
        steps = torch.linspace(1.0, 0.0, self.num_sample_steps + 1, device=self.device)
        for i in tqdm(range(self.num_sample_steps), desc='sampling loop time step', total=self.num_sample_steps):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, times, times_next)
        img.clamp_(-1.0, 1.0)
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size=16):
        return self.p_sample_loop((batch_size, self.channels, self.image_size, self.image_size))

    @autocast('cuda', enabled=False)
    def q_sample(self, x_start, times, noise=None):
        noise = default(noise, lambda : torch.randn_like(x_start))
        log_snr = self.log_snr(times)
        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised = x_start * alpha + noise * sigma
        return x_noised, log_snr

    def p_losses(self, x_start, times, noise=None):
        noise = default(noise, lambda : torch.randn_like(x_start))
        x, log_snr = self.q_sample(x_start=x_start, times=times, noise=noise)
        model_out = self.model(x, log_snr)
        if self.pred_objective == 'v':
            padded_log_snr = right_pad_dims_to(x, log_snr)
            alpha, sigma = padded_log_snr.sigmoid().sqrt(), (-padded_log_snr).sigmoid().sqrt()
            target = alpha * noise - sigma * x_start
        elif self.pred_objective == 'eps':
            target = noise
        loss = F.mse_loss(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')
        snr = log_snr.exp()
        maybe_clip_snr = snr.clone()
        if self.min_snr_loss_weight:
            maybe_clip_snr.clamp_(max=self.min_snr_gamma)
        if self.pred_objective == 'v':
            loss_weight = maybe_clip_snr / (snr + 1)
        elif self.pred_objective == 'eps':
            loss_weight = maybe_clip_snr / snr
        return (loss * loss_weight).mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        img = normalize_to_neg_one_to_one(img)
        times = torch.zeros((img.shape[0],), device=self.device).float().uniform_(0, 1)
        return self.p_losses(img, times, *args, **kwargs)


class MonotonicLinear(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = nn.Linear(*args, **kwargs)

    def forward(self, x):
        return F.linear(x, self.net.weight.abs(), self.net.bias.abs())


class learned_noise_schedule(nn.Module):
    """ described in section H and then I.2 of the supplementary material for variational ddpm paper """

    def __init__(self, *, log_snr_max, log_snr_min, hidden_dim=1024, frac_gradient=1.0):
        super().__init__()
        self.slope = log_snr_min - log_snr_max
        self.intercept = log_snr_max
        self.net = nn.Sequential(Rearrange('... -> ... 1'), MonotonicLinear(1, 1), Residual(nn.Sequential(MonotonicLinear(1, hidden_dim), nn.Sigmoid(), MonotonicLinear(hidden_dim, 1))), Rearrange('... 1 -> ...'))
        self.frac_gradient = frac_gradient

    def forward(self, x):
        frac_gradient = self.frac_gradient
        device = x.device
        out_zero = self.net(torch.zeros_like(x))
        out_one = self.net(torch.ones_like(x))
        x = self.net(x)
        normed = self.slope * ((x - out_zero) / (out_one - out_zero)) + self.intercept
        return normed * frac_gradient + normed.detach() * (1 - frac_gradient)


def alpha_cosine_log_snr(t, s=0.008):
    return -log(torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2 - 1, eps=1e-05)


def beta_linear_log_snr(t):
    return -log(expm1(0.0001 + 10 * t ** 2))


class ContinuousTimeGaussianDiffusion(nn.Module):

    def __init__(self, model, *, image_size, channels=3, noise_schedule='linear', num_sample_steps=500, clip_sample_denoised=True, learned_schedule_net_hidden_dim=1024, learned_noise_schedule_frac_gradient=1.0, min_snr_loss_weight=False, min_snr_gamma=5):
        super().__init__()
        assert model.random_or_learned_sinusoidal_cond
        assert not model.self_condition, 'not supported yet'
        self.model = model
        self.channels = channels
        self.image_size = image_size
        if noise_schedule == 'linear':
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == 'cosine':
            self.log_snr = alpha_cosine_log_snr
        elif noise_schedule == 'learned':
            log_snr_max, log_snr_min = [beta_linear_log_snr(torch.tensor([time])).item() for time in (0.0, 1.0)]
            self.log_snr = learned_noise_schedule(log_snr_max=log_snr_max, log_snr_min=log_snr_min, hidden_dim=learned_schedule_net_hidden_dim, frac_gradient=learned_noise_schedule_frac_gradient)
        else:
            raise ValueError(f'unknown noise schedule {noise_schedule}')
        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised
        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

    @property
    def device(self):
        return next(self.model.parameters()).device

    def p_mean_variance(self, x, time, time_next):
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)
        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()
        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))
        batch_log_snr = repeat(log_snr, ' -> b', b=x.shape[0])
        pred_noise = self.model(x, batch_log_snr)
        if self.clip_sample_denoised:
            x_start = (x - sigma * pred_noise) / alpha
            x_start.clamp_(-1.0, 1.0)
            model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
        else:
            model_mean = alpha_next / alpha * (x - c * sigma * pred_noise)
        posterior_variance = squared_sigma_next * c
        return model_mean, posterior_variance

    @torch.no_grad()
    def p_sample(self, x, time, time_next):
        batch, *_, device = *x.shape, x.device
        model_mean, model_variance = self.p_mean_variance(x=x, time=time, time_next=time_next)
        if time_next == 0:
            return model_mean
        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch = shape[0]
        img = torch.randn(shape, device=self.device)
        steps = torch.linspace(1.0, 0.0, self.num_sample_steps + 1, device=self.device)
        for i in tqdm(range(self.num_sample_steps), desc='sampling loop time step', total=self.num_sample_steps):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, times, times_next)
        img.clamp_(-1.0, 1.0)
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size=16):
        return self.p_sample_loop((batch_size, self.channels, self.image_size, self.image_size))

    @autocast('cuda', enabled=False)
    def q_sample(self, x_start, times, noise=None):
        noise = default(noise, lambda : torch.randn_like(x_start))
        log_snr = self.log_snr(times)
        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised = x_start * alpha + noise * sigma
        return x_noised, log_snr

    def random_times(self, batch_size):
        return torch.zeros((batch_size,), device=self.device).float().uniform_(0, 1)

    def p_losses(self, x_start, times, noise=None):
        noise = default(noise, lambda : torch.randn_like(x_start))
        x, log_snr = self.q_sample(x_start=x_start, times=times, noise=noise)
        model_out = self.model(x, log_snr)
        losses = F.mse_loss(model_out, noise, reduction='none')
        losses = reduce(losses, 'b ... -> b', 'mean')
        if self.min_snr_loss_weight:
            snr = log_snr.exp()
            loss_weight = snr.clamp(min=self.min_snr_gamma) / snr
            losses = losses * loss_weight
        return losses.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        times = self.random_times(b)
        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, times, *args, **kwargs)


class Unet1D(Module):

    def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=3, dropout=0.0, self_condition=False, learned_variance=False, learned_sinusoidal_cond=False, random_fourier_features=False, learned_sinusoidal_dim=16, sinusoidal_pos_emb_theta=10000, attn_dim_head=32, attn_heads=4):
        super().__init__()
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding=3)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        time_dim = dim * 4
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim
        self.time_mlp = nn.Sequential(sinu_pos_emb, nn.Linear(fourier_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))
        resnet_block = partial(ResnetBlock, time_emb_dim=time_dim, dropout=dropout)
        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= num_resolutions - 1
            self.downs.append(ModuleList([resnet_block(dim_in, dim_in), resnet_block(dim_in, dim_in), Residual(PreNorm(dim_in, LinearAttention(dim_in))), Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1)]))
        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head=attn_dim_head, heads=attn_heads)))
        self.mid_block2 = resnet_block(mid_dim, mid_dim)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == len(in_out) - 1
            self.ups.append(ModuleList([resnet_block(dim_out + dim_in, dim_out), resnet_block(dim_out + dim_in, dim_out), Residual(PreNorm(dim_out, LinearAttention(dim_out))), Upsample(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding=1)]))
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv1d(init_dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda : torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)


ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(betas, 0, 0.999)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def identity(t):
    return t


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


class GaussianDiffusion1D(Module):

    def __init__(self, model, *, seq_length, timesteps=1000, sampling_timesteps=None, objective='pred_noise', beta_schedule='cosine', ddim_sampling_eta=0.0, auto_normalize=True):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.seq_length = seq_length
        self.objective = objective
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        register_buffer = lambda name, val: self.register_buffer(name, val)
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
        snr = alphas_cumprod / (1 - alphas_cumprod)
        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)
        register_buffer('loss_weight', loss_weight)
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    def predict_noise_from_start(self, x_t, t, x0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_v(self, x_start, t, noise):
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start

    def predict_start_from_v(self, x_t, t, v):
        return extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond=None, clip_x_start=False, rederive_pred_noise=False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)
        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start
        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: 'int', x_self_cond=None, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, x_self_cond=x_self_cond, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        x_start = None
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
        img = self.unnormalize(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        img = torch.randn(shape, device=device)
        x_start = None
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start=clip_denoised)
            if time_next < 0:
                img = x_start
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
        img = self.unnormalize(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size=16):
        seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, seq_length))

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)
        assert x1.shape == x2.shape
        t_batched = torch.full((b,), t, device=device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))
        img = (1 - lam) * xt1 + lam * xt2
        x_start = None
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)
        return img

    @autocast('cuda', enabled=False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda : torch.randn_like(x_start))
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def p_losses(self, x_start, t, noise=None):
        b, c, n = x_start.shape
        noise = default(noise, lambda : torch.randn_like(x_start))
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()
        model_out = self.model(x, t, x_self_cond)
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        loss = F.mse_loss(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, n, device, seq_length = *img.shape, img.device, self.seq_length
        assert n == seq_length, f'seq length must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)


class ElucidatedDiffusion(nn.Module):

    def __init__(self, net, *, image_size, channels=3, num_sample_steps=32, sigma_min=0.002, sigma_max=80, sigma_data=0.5, rho=7, P_mean=-1.2, P_std=1.2, S_churn=80, S_tmin=0.05, S_tmax=50, S_noise=1.003):
        super().__init__()
        assert net.random_or_learned_sinusoidal_cond
        self.self_condition = net.self_condition
        self.net = net
        self.channels = channels
        self.image_size = image_size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sample_steps = num_sample_steps
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

    @property
    def device(self):
        return next(self.net.parameters()).device

    def c_skip(self, sigma):
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    def preconditioned_network_forward(self, noised_images, sigma, self_cond=None, clamp=False):
        batch, device = noised_images.shape[0], noised_images.device
        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)
        padded_sigma = rearrange(sigma, 'b -> b 1 1 1')
        net_out = self.net(self.c_in(padded_sigma) * noised_images, self.c_noise(sigma), self_cond)
        out = self.c_skip(padded_sigma) * noised_images + self.c_out(padded_sigma) * net_out
        if clamp:
            out = out.clamp(-1.0, 1.0)
        return out

    def sample_schedule(self, num_sample_steps=None):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)
        N = num_sample_steps
        inv_rho = 1 / self.rho
        steps = torch.arange(num_sample_steps, device=self.device, dtype=torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho
        sigmas = F.pad(sigmas, (0, 1), value=0.0)
        return sigmas

    @torch.no_grad()
    def sample(self, batch_size=16, num_sample_steps=None, clamp=True):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)
        shape = batch_size, self.channels, self.image_size, self.image_size
        sigmas = self.sample_schedule(num_sample_steps)
        gammas = torch.where((sigmas >= self.S_tmin) & (sigmas <= self.S_tmax), min(self.S_churn / num_sample_steps, sqrt(2) - 1), 0.0)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))
        init_sigma = sigmas[0]
        images = init_sigma * torch.randn(shape, device=self.device)
        x_start = None
        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas, desc='sampling time step'):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))
            eps = self.S_noise * torch.randn(shape, device=self.device)
            sigma_hat = sigma + gamma * sigma
            images_hat = images + sqrt(sigma_hat ** 2 - sigma ** 2) * eps
            self_cond = x_start if self.self_condition else None
            model_output = self.preconditioned_network_forward(images_hat, sigma_hat, self_cond, clamp=clamp)
            denoised_over_sigma = (images_hat - model_output) / sigma_hat
            images_next = images_hat + (sigma_next - sigma_hat) * denoised_over_sigma
            if sigma_next != 0:
                self_cond = model_output if self.self_condition else None
                model_output_next = self.preconditioned_network_forward(images_next, sigma_next, self_cond, clamp=clamp)
                denoised_prime_over_sigma = (images_next - model_output_next) / sigma_next
                images_next = images_hat + 0.5 * (sigma_next - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma)
            images = images_next
            x_start = model_output_next if sigma_next != 0 else model_output
        images = images.clamp(-1.0, 1.0)
        return unnormalize_to_zero_to_one(images)

    @torch.no_grad()
    def sample_using_dpmpp(self, batch_size=16, num_sample_steps=None):
        """
        thanks to Katherine Crowson (https://github.com/crowsonkb) for figuring it all out!
        https://arxiv.org/abs/2211.01095
        """
        device, num_sample_steps = self.device, default(num_sample_steps, self.num_sample_steps)
        sigmas = self.sample_schedule(num_sample_steps)
        shape = batch_size, self.channels, self.image_size, self.image_size
        images = sigmas[0] * torch.randn(shape, device=device)
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()
        old_denoised = None
        for i in tqdm(range(len(sigmas) - 1)):
            denoised = self.preconditioned_network_forward(images, sigmas[i].item())
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            if not exists(old_denoised) or sigmas[i + 1] == 0:
                denoised_d = denoised
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                gamma = -1 / (2 * r)
                denoised_d = (1 - gamma) * denoised + gamma * old_denoised
            images = sigma_fn(t_next) / sigma_fn(t) * images - (-h).expm1() * denoised_d
            old_denoised = denoised
        images = images.clamp(-1.0, 1.0)
        return unnormalize_to_zero_to_one(images)

    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device=self.device)).exp()

    def forward(self, images):
        batch_size, c, h, w, device, image_size, channels = *images.shape, images.device, self.image_size, self.channels
        assert h == image_size and w == image_size, f'height and width of image must be {image_size}'
        assert c == channels, 'mismatch of image channels'
        images = normalize_to_neg_one_to_one(images)
        sigmas = self.noise_distribution(batch_size)
        padded_sigmas = rearrange(sigmas, 'b -> b 1 1 1')
        noise = torch.randn_like(images)
        noised_images = images + padded_sigmas * noise
        self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                self_cond = self.preconditioned_network_forward(noised_images, sigmas)
                self_cond.detach_()
        denoised = self.preconditioned_network_forward(noised_images, sigmas, self_cond)
        losses = F.mse_loss(denoised, images, reduction='none')
        losses = reduce(losses, 'b ... -> b', 'mean')
        losses = losses * self.loss_weight(sigmas)
        return losses.mean()


class MPSiLU(Module):

    def forward(self, x):
        return F.silu(x) / 0.596


class Gain(Module):

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x * self.gain


class MPCat(Module):

    def __init__(self, t=0.5, dim=-1):
        super().__init__()
        self.t = t
        self.dim = dim

    def forward(self, a, b):
        dim, t = self.dim, self.t
        Na, Nb = a.shape[dim], b.shape[dim]
        C = sqrt((Na + Nb) / ((1.0 - t) ** 2 + t ** 2))
        a = a * (1.0 - t) / sqrt(Na)
        b = b * t / sqrt(Nb)
        return C * torch.cat((a, b), dim=dim)


class MPAdd(Module):

    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, x, res):
        a, b, t = x, res, self.t
        num = a * (1.0 - t) + b * t
        den = sqrt((1 - t) ** 2 + t ** 2)
        return num / den


class PixelNorm(Module):

    def __init__(self, dim, eps=0.0001):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        dim = self.dim
        return l2norm(x, dim=dim, eps=self.eps) * sqrt(x.shape[dim])


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def normalize_weight(weight, eps=0.0001):
    weight, ps = pack_one(weight, 'o *')
    normed_weight = l2norm(weight, eps=eps)
    normed_weight = normed_weight * sqrt(weight.numel() / weight.shape[0])
    return unpack_one(normed_weight, ps, 'o *')


class Conv2d(Module):

    def __init__(self, dim_in, dim_out, kernel_size, eps=0.0001, concat_ones_to_input=False):
        super().__init__()
        weight = torch.randn(dim_out, dim_in + int(concat_ones_to_input), kernel_size, kernel_size)
        self.weight = nn.Parameter(weight)
        self.eps = eps
        self.fan_in = dim_in * kernel_size ** 2
        self.concat_ones_to_input = concat_ones_to_input

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                normed_weight = normalize_weight(self.weight, eps=self.eps)
                self.weight.copy_(normed_weight)
        weight = normalize_weight(self.weight, eps=self.eps) / sqrt(self.fan_in)
        if self.concat_ones_to_input:
            x = F.pad(x, (0, 0, 0, 0, 1, 0), value=1.0)
        return F.conv2d(x, weight, padding='same')


class Linear(Module):

    def __init__(self, dim_in, dim_out, eps=0.0001):
        super().__init__()
        weight = torch.randn(dim_out, dim_in)
        self.weight = nn.Parameter(weight)
        self.eps = eps
        self.fan_in = dim_in

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                normed_weight = normalize_weight(self.weight, eps=self.eps)
                self.weight.copy_(normed_weight)
        weight = normalize_weight(self.weight, eps=self.eps) / sqrt(self.fan_in)
        return F.linear(x, weight)


class MPFourierEmbedding(Module):

    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=False)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        return torch.cat((freqs.sin(), freqs.cos()), dim=-1) * sqrt(2)


class Conv3d(Module):

    def __init__(self, dim_in, dim_out, kernel_size, eps=0.0001, concat_ones_to_input=False):
        super().__init__()
        weight = torch.randn(dim_out, dim_in + int(concat_ones_to_input), kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(weight)
        self.eps = eps
        self.fan_in = dim_in * kernel_size ** 3
        self.concat_ones_to_input = concat_ones_to_input

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                normed_weight = normalize_weight(self.weight, eps=self.eps)
                self.weight.copy_(normed_weight)
        weight = normalize_weight(self.weight, eps=self.eps) / sqrt(self.fan_in)
        if self.concat_ones_to_input:
            x = F.pad(x, (0, 0, 0, 0, 0, 0, 1, 0), value=1.0)
        return F.conv3d(x, weight, padding='same')


class Encoder(Module):

    def __init__(self, dim, dim_out=None, *, emb_dim=None, dropout=0.1, mp_add_t=0.3, has_attn=False, attn_dim_head=64, attn_res_mp_add_t=0.3, attn_flash=False, factorize_space_time_attn=False, downsample=False, downsample_config: Tuple[bool, bool, bool]=(True, True, True)):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.downsample = downsample
        self.downsample_config = downsample_config
        self.downsample_conv = None
        curr_dim = dim
        if downsample:
            self.downsample_conv = Conv3d(curr_dim, dim_out, 1)
            curr_dim = dim_out
        self.pixel_norm = PixelNorm(dim=1)
        self.to_emb = None
        if exists(emb_dim):
            self.to_emb = nn.Sequential(Linear(emb_dim, dim_out), Gain())
        self.block1 = nn.Sequential(MPSiLU(), Conv3d(curr_dim, dim_out, 3))
        self.block2 = nn.Sequential(MPSiLU(), nn.Dropout(dropout), Conv3d(dim_out, dim_out, 3))
        self.res_mp_add = MPAdd(t=mp_add_t)
        self.attn = None
        self.factorized_attn = factorize_space_time_attn
        if has_attn:
            attn_kwargs = dict(dim=dim_out, heads=max(ceil(dim_out / attn_dim_head), 2), dim_head=attn_dim_head, mp_add_t=attn_res_mp_add_t, flash=attn_flash)
            if factorize_space_time_attn:
                self.attn = nn.ModuleList([Attention(**attn_kwargs, only_space=True), Attention(**attn_kwargs, only_time=True)])
            else:
                self.attn = Attention(**attn_kwargs)

    def forward(self, x, emb=None):
        if self.downsample:
            t, h, w = x.shape[-3:]
            resize_factors = tuple(2 if downsample else 1 for downsample in self.downsample_config)
            interpolate_shape = tuple(shape // factor for shape, factor in zip((t, h, w), resize_factors))
            x = F.interpolate(x, interpolate_shape, mode='trilinear')
            x = self.downsample_conv(x)
        x = self.pixel_norm(x)
        res = x.clone()
        x = self.block1(x)
        if exists(emb):
            scale = self.to_emb(emb) + 1
            x = x * rearrange(scale, 'b c -> b c 1 1 1')
        x = self.block2(x)
        x = self.res_mp_add(x, res)
        if exists(self.attn):
            if self.factorized_attn:
                attn_space, attn_time = self.attn
                x = attn_space(x)
                x = attn_time(x)
            else:
                x = self.attn(x)
        return x


class Decoder(Module):

    def __init__(self, dim, dim_out=None, *, emb_dim=None, dropout=0.1, mp_add_t=0.3, has_attn=False, attn_dim_head=64, attn_res_mp_add_t=0.3, attn_flash=False, factorize_space_time_attn=False, upsample=False, upsample_config: Tuple[bool, bool, bool]=(True, True, True)):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.upsample = upsample
        self.upsample_config = upsample_config
        self.needs_skip = not upsample
        self.to_emb = None
        if exists(emb_dim):
            self.to_emb = nn.Sequential(Linear(emb_dim, dim_out), Gain())
        self.block1 = nn.Sequential(MPSiLU(), Conv3d(dim, dim_out, 3))
        self.block2 = nn.Sequential(MPSiLU(), nn.Dropout(dropout), Conv3d(dim_out, dim_out, 3))
        self.res_conv = Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.res_mp_add = MPAdd(t=mp_add_t)
        self.attn = None
        self.factorized_attn = factorize_space_time_attn
        if has_attn:
            attn_kwargs = dict(dim=dim_out, heads=max(ceil(dim_out / attn_dim_head), 2), dim_head=attn_dim_head, mp_add_t=attn_res_mp_add_t, flash=attn_flash)
            if factorize_space_time_attn:
                self.attn = nn.ModuleList([Attention(**attn_kwargs, only_space=True), Attention(**attn_kwargs, only_time=True)])
            else:
                self.attn = Attention(**attn_kwargs)

    def forward(self, x, emb=None):
        if self.upsample:
            t, h, w = x.shape[-3:]
            resize_factors = tuple(2 if upsample else 1 for upsample in self.upsample_config)
            interpolate_shape = tuple(shape * factor for shape, factor in zip((t, h, w), resize_factors))
            x = F.interpolate(x, interpolate_shape, mode='trilinear')
        res = self.res_conv(x)
        x = self.block1(x)
        if exists(emb):
            scale = self.to_emb(emb) + 1
            x = x * rearrange(scale, 'b c -> b c 1 1 1')
        x = self.block2(x)
        x = self.res_mp_add(x, res)
        if exists(self.attn):
            if self.factorized_attn:
                attn_space, attn_time = self.attn
                x = attn_space(x)
                x = attn_time(x)
            else:
                x = self.attn(x)
        return x


def append(arr, el):
    arr.append(el)


def prepend(arr, el):
    arr.insert(0, el)


def xnor(x, y):
    return not x ^ y


class KarrasUnet(Module):
    """
    going by figure 21. config G
    """

    def __init__(self, *, image_size, dim=192, dim_max=768, num_classes=None, channels=4, num_downsamples=3, num_blocks_per_stage=4, attn_res=(16, 8), fourier_dim=16, attn_dim_head=64, attn_flash=False, mp_cat_t=0.5, mp_add_emb_t=0.5, attn_res_mp_add_t=0.3, resnet_mp_add_t=0.3, dropout=0.1, self_condition=False):
        super().__init__()
        self.self_condition = self_condition
        self.channels = channels
        self.image_size = image_size
        input_channels = channels * (2 if self_condition else 1)
        self.input_block = Conv2d(input_channels, dim, 3, concat_ones_to_input=True)
        self.output_block = nn.Sequential(Conv2d(dim, channels, 3), Gain())
        emb_dim = dim * 4
        self.to_time_emb = nn.Sequential(MPFourierEmbedding(fourier_dim), Linear(fourier_dim, emb_dim))
        self.needs_class_labels = exists(num_classes)
        self.num_classes = num_classes
        if self.needs_class_labels:
            self.to_class_emb = Linear(num_classes, 4 * dim)
            self.add_class_emb = MPAdd(t=mp_add_emb_t)
        self.emb_activation = MPSiLU()
        self.num_downsamples = num_downsamples
        attn_res = set(cast_tuple(attn_res))
        block_kwargs = dict(dropout=dropout, emb_dim=emb_dim, attn_dim_head=attn_dim_head, attn_res_mp_add_t=attn_res_mp_add_t, attn_flash=attn_flash)
        self.downs = ModuleList([])
        self.ups = ModuleList([])
        curr_dim = dim
        curr_res = image_size
        self.skip_mp_cat = MPCat(t=mp_cat_t, dim=1)
        prepend(self.ups, Decoder(dim * 2, dim, **block_kwargs))
        assert num_blocks_per_stage >= 1
        for _ in range(num_blocks_per_stage):
            enc = Encoder(curr_dim, curr_dim, **block_kwargs)
            dec = Decoder(curr_dim * 2, curr_dim, **block_kwargs)
            append(self.downs, enc)
            prepend(self.ups, dec)
        for _ in range(self.num_downsamples):
            dim_out = min(dim_max, curr_dim * 2)
            upsample = Decoder(dim_out, curr_dim, has_attn=curr_res in attn_res, upsample=True, **block_kwargs)
            curr_res //= 2
            has_attn = curr_res in attn_res
            downsample = Encoder(curr_dim, dim_out, downsample=True, has_attn=has_attn, **block_kwargs)
            append(self.downs, downsample)
            prepend(self.ups, upsample)
            prepend(self.ups, Decoder(dim_out * 2, dim_out, has_attn=has_attn, **block_kwargs))
            for _ in range(num_blocks_per_stage):
                enc = Encoder(dim_out, dim_out, has_attn=has_attn, **block_kwargs)
                dec = Decoder(dim_out * 2, dim_out, has_attn=has_attn, **block_kwargs)
                append(self.downs, enc)
                prepend(self.ups, dec)
            curr_dim = dim_out
        mid_has_attn = curr_res in attn_res
        self.mids = ModuleList([Decoder(curr_dim, curr_dim, has_attn=mid_has_attn, **block_kwargs), Decoder(curr_dim, curr_dim, has_attn=mid_has_attn, **block_kwargs)])
        self.out_dim = channels

    @property
    def downsample_factor(self):
        return 2 ** self.num_downsamples

    def forward(self, x, time, self_cond=None, class_labels=None):
        assert x.shape[1:] == (self.channels, self.image_size, self.image_size)
        if self.self_condition:
            self_cond = default(self_cond, lambda : torch.zeros_like(x))
            x = torch.cat((self_cond, x), dim=1)
        else:
            assert not exists(self_cond)
        time_emb = self.to_time_emb(time)
        assert xnor(exists(class_labels), self.needs_class_labels)
        if self.needs_class_labels:
            if class_labels.dtype in (torch.int, torch.long):
                class_labels = F.one_hot(class_labels, self.num_classes)
            assert class_labels.shape[-1] == self.num_classes
            class_labels = class_labels.float() * sqrt(self.num_classes)
            class_emb = self.to_class_emb(class_labels)
            time_emb = self.add_class_emb(time_emb, class_emb)
        emb = self.emb_activation(time_emb)
        skips = []
        x = self.input_block(x)
        skips.append(x)
        for encoder in self.downs:
            x = encoder(x, emb=emb)
            skips.append(x)
        for decoder in self.mids:
            x = decoder(x, emb=emb)
        for decoder in self.ups:
            if decoder.needs_skip:
                skip = skips.pop()
                x = self.skip_mp_cat(x, skip)
            x = decoder(x, emb=emb)
        return self.output_block(x)


class MPFeedForward(Module):

    def __init__(self, *, dim, mult=4, mp_add_t=0.3):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(PixelNorm(dim=1), Conv3d(dim, dim_inner, 1), MPSiLU(), Conv3d(dim_inner, dim, 1))
        self.mp_add = MPAdd(t=mp_add_t)

    def forward(self, x):
        res = x
        out = self.net(x)
        return self.mp_add(out, res)


class MPImageTransformer(Module):

    def __init__(self, *, dim, depth, dim_head=64, heads=8, num_mem_kv=4, ff_mult=4, attn_flash=False, residual_mp_add_t=0.3):
        super().__init__()
        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(ModuleList([Attention(dim=dim, heads=heads, dim_head=dim_head, num_mem_kv=num_mem_kv, flash=attn_flash, mp_add_t=residual_mp_add_t), MPFeedForward(dim=dim, mult=ff_mult, mp_add_t=residual_mp_add_t)]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class Conv1d(Module):

    def __init__(self, dim_in, dim_out, kernel_size, eps=0.0001, init_dirac=False, concat_ones_to_input=False):
        super().__init__()
        weight = torch.randn(dim_out, dim_in + int(concat_ones_to_input), kernel_size)
        self.weight = nn.Parameter(weight)
        if init_dirac:
            nn.init.dirac_(self.weight)
        self.eps = eps
        self.fan_in = dim_in * kernel_size
        self.concat_ones_to_input = concat_ones_to_input

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                normed_weight = normalize_weight(self.weight, eps=self.eps)
                self.weight.copy_(normed_weight)
        weight = normalize_weight(self.weight, eps=self.eps) / sqrt(self.fan_in)
        if self.concat_ones_to_input:
            x = F.pad(x, (0, 0, 1, 0), value=1.0)
        return F.conv1d(x, weight, padding='same')


class KarrasUnet1D(Module):
    """
    going by figure 21. config G
    """

    def __init__(self, *, seq_len, dim=192, dim_max=768, num_classes=None, channels=4, num_downsamples=3, num_blocks_per_stage=4, attn_res=(16, 8), fourier_dim=16, attn_dim_head=64, attn_flash=False, mp_cat_t=0.5, mp_add_emb_t=0.5, attn_res_mp_add_t=0.3, resnet_mp_add_t=0.3, dropout=0.1, self_condition=False):
        super().__init__()
        self.self_condition = self_condition
        self.channels = channels
        self.seq_len = seq_len
        input_channels = channels * (2 if self_condition else 1)
        self.input_block = Conv1d(input_channels, dim, 3, concat_ones_to_input=True)
        self.output_block = nn.Sequential(Conv1d(dim, channels, 3), Gain())
        emb_dim = dim * 4
        self.to_time_emb = nn.Sequential(MPFourierEmbedding(fourier_dim), Linear(fourier_dim, emb_dim))
        self.needs_class_labels = exists(num_classes)
        self.num_classes = num_classes
        if self.needs_class_labels:
            self.to_class_emb = Linear(num_classes, 4 * dim)
            self.add_class_emb = MPAdd(t=mp_add_emb_t)
        self.emb_activation = MPSiLU()
        self.num_downsamples = num_downsamples
        attn_res = set(cast_tuple(attn_res))
        block_kwargs = dict(dropout=dropout, emb_dim=emb_dim, attn_dim_head=attn_dim_head, attn_res_mp_add_t=attn_res_mp_add_t, attn_flash=attn_flash)
        self.downs = ModuleList([])
        self.ups = ModuleList([])
        curr_dim = dim
        curr_res = seq_len
        self.skip_mp_cat = MPCat(t=mp_cat_t, dim=1)
        prepend(self.ups, Decoder(dim * 2, dim, **block_kwargs))
        assert num_blocks_per_stage >= 1
        for _ in range(num_blocks_per_stage):
            enc = Encoder(curr_dim, curr_dim, **block_kwargs)
            dec = Decoder(curr_dim * 2, curr_dim, **block_kwargs)
            append(self.downs, enc)
            prepend(self.ups, dec)
        for _ in range(self.num_downsamples):
            dim_out = min(dim_max, curr_dim * 2)
            upsample = Decoder(dim_out, curr_dim, has_attn=curr_res in attn_res, upsample=True, **block_kwargs)
            curr_res //= 2
            has_attn = curr_res in attn_res
            downsample = Encoder(curr_dim, dim_out, downsample=True, has_attn=has_attn, **block_kwargs)
            append(self.downs, downsample)
            prepend(self.ups, upsample)
            prepend(self.ups, Decoder(dim_out * 2, dim_out, has_attn=has_attn, **block_kwargs))
            for _ in range(num_blocks_per_stage):
                enc = Encoder(dim_out, dim_out, has_attn=has_attn, **block_kwargs)
                dec = Decoder(dim_out * 2, dim_out, has_attn=has_attn, **block_kwargs)
                append(self.downs, enc)
                prepend(self.ups, dec)
            curr_dim = dim_out
        mid_has_attn = curr_res in attn_res
        self.mids = ModuleList([Decoder(curr_dim, curr_dim, has_attn=mid_has_attn, **block_kwargs), Decoder(curr_dim, curr_dim, has_attn=mid_has_attn, **block_kwargs)])
        self.out_dim = channels

    @property
    def downsample_factor(self):
        return 2 ** self.num_downsamples

    def forward(self, x, time, self_cond=None, class_labels=None):
        assert x.shape[1:] == (self.channels, self.seq_len)
        if self.self_condition:
            self_cond = default(self_cond, lambda : torch.zeros_like(x))
            x = torch.cat((self_cond, x), dim=1)
        else:
            assert not exists(self_cond)
        time_emb = self.to_time_emb(time)
        assert xnor(exists(class_labels), self.needs_class_labels)
        if self.needs_class_labels:
            if class_labels.dtype in (torch.int, torch.long):
                class_labels = F.one_hot(class_labels, self.num_classes)
            assert class_labels.shape[-1] == self.num_classes
            class_labels = class_labels.float() * sqrt(self.num_classes)
            class_emb = self.to_class_emb(class_labels)
            time_emb = self.add_class_emb(time_emb, class_emb)
        emb = self.emb_activation(time_emb)
        skips = []
        x = self.input_block(x)
        skips.append(x)
        for encoder in self.downs:
            x = encoder(x, emb=emb)
            skips.append(x)
        for decoder in self.mids:
            x = decoder(x, emb=emb)
        for decoder in self.ups:
            if decoder.needs_skip:
                skip = skips.pop()
                x = self.skip_mp_cat(x, skip)
            x = decoder(x, emb=emb)
        return self.output_block(x)


class KarrasUnet3D(Module):
    """
    going by figure 21. config G
    """

    def __init__(self, *, image_size, frames, dim=192, dim_max=768, num_classes=None, channels=4, num_downsamples=3, num_blocks_per_stage: Union[int, Tuple[int, ...]]=4, downsample_types: Optional[Tuple[str, ...]]=None, attn_res=(16, 8), fourier_dim=16, attn_dim_head=64, attn_flash=False, mp_cat_t=0.5, mp_add_emb_t=0.5, attn_res_mp_add_t=0.3, resnet_mp_add_t=0.3, dropout=0.1, self_condition=False, factorize_space_time_attn=False):
        super().__init__()
        self.self_condition = self_condition
        self.channels = channels
        self.frames = frames
        self.image_size = image_size
        input_channels = channels * (2 if self_condition else 1)
        self.input_block = Conv3d(input_channels, dim, 3, concat_ones_to_input=True)
        self.output_block = nn.Sequential(Conv3d(dim, channels, 3), Gain())
        emb_dim = dim * 4
        self.to_time_emb = nn.Sequential(MPFourierEmbedding(fourier_dim), Linear(fourier_dim, emb_dim))
        self.needs_class_labels = exists(num_classes)
        self.num_classes = num_classes
        if self.needs_class_labels:
            self.to_class_emb = Linear(num_classes, 4 * dim)
            self.add_class_emb = MPAdd(t=mp_add_emb_t)
        self.emb_activation = MPSiLU()
        self.num_downsamples = num_downsamples
        downsample_types = default(downsample_types, 'all')
        downsample_types = cast_tuple(downsample_types, num_downsamples)
        assert len(downsample_types) == num_downsamples
        assert all([(t in {'all', 'frame', 'image'}) for t in downsample_types])
        num_blocks_per_stage = cast_tuple(num_blocks_per_stage, num_downsamples)
        if len(num_blocks_per_stage) == num_downsamples:
            first, *_ = num_blocks_per_stage
            num_blocks_per_stage = first, *num_blocks_per_stage
        assert len(num_blocks_per_stage) == num_downsamples + 1
        assert all([(num_blocks >= 1) for num_blocks in num_blocks_per_stage])
        attn_res = set(cast_tuple(attn_res))
        block_kwargs = dict(dropout=dropout, emb_dim=emb_dim, attn_dim_head=attn_dim_head, attn_res_mp_add_t=attn_res_mp_add_t, attn_flash=attn_flash)
        self.downs = ModuleList([])
        self.ups = ModuleList([])
        curr_dim = dim
        curr_image_res = image_size
        curr_frame_res = frames
        self.skip_mp_cat = MPCat(t=mp_cat_t, dim=1)
        prepend(self.ups, Decoder(dim * 2, dim, **block_kwargs))
        init_num_blocks_per_stage, *rest_num_blocks_per_stage = num_blocks_per_stage
        for _ in range(init_num_blocks_per_stage):
            enc = Encoder(curr_dim, curr_dim, **block_kwargs)
            dec = Decoder(curr_dim * 2, curr_dim, **block_kwargs)
            append(self.downs, enc)
            prepend(self.ups, dec)
        for _, layer_num_blocks_per_stage, layer_downsample_type in zip(range(self.num_downsamples), rest_num_blocks_per_stage, downsample_types):
            dim_out = min(dim_max, curr_dim * 2)
            downsample_image = layer_downsample_type in {'all', 'image'}
            downsample_frame = layer_downsample_type in {'all', 'frame'}
            assert not (downsample_image and not divisible_by(curr_image_res, 2))
            assert not (downsample_frame and not divisible_by(curr_frame_res, 2))
            down_and_upsample_config = downsample_frame, downsample_image, downsample_image
            upsample = Decoder(dim_out, curr_dim, has_attn=curr_image_res in attn_res, upsample=True, upsample_config=down_and_upsample_config, factorize_space_time_attn=factorize_space_time_attn, **block_kwargs)
            if downsample_image:
                curr_image_res //= 2
            if downsample_frame:
                curr_frame_res //= 2
            has_attn = curr_image_res in attn_res
            downsample = Encoder(curr_dim, dim_out, downsample=True, downsample_config=down_and_upsample_config, has_attn=has_attn, factorize_space_time_attn=factorize_space_time_attn, **block_kwargs)
            append(self.downs, downsample)
            prepend(self.ups, upsample)
            prepend(self.ups, Decoder(dim_out * 2, dim_out, has_attn=has_attn, **block_kwargs))
            for _ in range(layer_num_blocks_per_stage):
                enc = Encoder(dim_out, dim_out, has_attn=has_attn, **block_kwargs)
                dec = Decoder(dim_out * 2, dim_out, has_attn=has_attn, **block_kwargs)
                append(self.downs, enc)
                prepend(self.ups, dec)
            curr_dim = dim_out
        mid_has_attn = curr_image_res in attn_res
        self.mids = ModuleList([Decoder(curr_dim, curr_dim, has_attn=mid_has_attn, **block_kwargs), Decoder(curr_dim, curr_dim, has_attn=mid_has_attn, **block_kwargs)])
        self.out_dim = channels

    @property
    def downsample_factor(self):
        return 2 ** self.num_downsamples

    def forward(self, x, time, self_cond=None, class_labels=None):
        assert x.shape[1:] == (self.channels, self.frames, self.image_size, self.image_size)
        if self.self_condition:
            self_cond = default(self_cond, lambda : torch.zeros_like(x))
            x = torch.cat((self_cond, x), dim=1)
        else:
            assert not exists(self_cond)
        time_emb = self.to_time_emb(time)
        assert xnor(exists(class_labels), self.needs_class_labels)
        if self.needs_class_labels:
            if class_labels.dtype in (torch.int, torch.long):
                class_labels = F.one_hot(class_labels, self.num_classes)
            assert class_labels.shape[-1] == self.num_classes
            class_labels = class_labels.float() * sqrt(self.num_classes)
            class_emb = self.to_class_emb(class_labels)
            time_emb = self.add_class_emb(time_emb, class_emb)
        emb = self.emb_activation(time_emb)
        skips = []
        x = self.input_block(x)
        skips.append(x)
        for encoder in self.downs:
            x = encoder(x, emb=emb)
            skips.append(x)
        for decoder in self.mids:
            x = decoder(x, emb=emb)
        for decoder in self.ups:
            if decoder.needs_skip:
                skip = skips.pop()
                x = self.skip_mp_cat(x, skip)
            x = decoder(x, emb=emb)
        return self.output_block(x)


NAT = 1.0 / ln(2)


def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(sqrt(2.0 / pi) * (x + 0.044715 * x ** 3)))


def discretized_gaussian_log_likelihood(x, *, means, log_scales, thres=0.999):
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = log(cdf_plus)
    log_one_minus_cdf_min = log(1.0 - cdf_min)
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(x < -thres, log_cdf_plus, torch.where(x > thres, log_one_minus_cdf_min, log(cdf_delta)))
    return log_probs


def meanflat(x):
    return x.mean(dim=tuple(range(1, len(x.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))


class LearnedGaussianDiffusion(GaussianDiffusion):

    def __init__(self, model, vb_loss_weight=0.001, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        assert model.out_dim == model.channels * 2, 'dimension out of unet must be twice the number of channels for learned variance - you can also set the `learned_variance` keyword argument on the Unet to be `True`'
        assert not model.self_condition, 'not supported yet'
        self.vb_loss_weight = vb_loss_weight

    def model_predictions(self, x, t, x_self_cond=None, clip_x_start=False, rederive_pred_noise=False):
        model_output = self.model(x, t)
        model_output, pred_variance = model_output.chunk(2, dim=1)
        maybe_clip = partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, model_output)
        elif self.objective == 'pred_x0':
            pred_noise = self.predict_noise_from_start(x, t, model_output)
            x_start = model_output
        x_start = maybe_clip(x_start)
        return ModelPrediction(pred_noise, x_start, pred_variance)

    def p_mean_variance(self, *, x, t, clip_denoised, model_output=None, **kwargs):
        model_output = default(model_output, lambda : self.model(x, t))
        pred_noise, var_interp_frac_unnormalized = model_output.chunk(2, dim=1)
        min_log = extract(self.posterior_log_variance_clipped, t, x.shape)
        max_log = extract(torch.log(self.betas), t, x.shape)
        var_interp_frac = unnormalize_to_zero_to_one(var_interp_frac_unnormalized)
        model_log_variance = var_interp_frac * max_log + (1 - var_interp_frac) * min_log
        model_variance = model_log_variance.exp()
        x_start = self.predict_start_from_noise(x, t, pred_noise)
        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)
        model_mean, _, _ = self.q_posterior(x_start, x, t)
        return model_mean, model_variance, model_log_variance, x_start

    def p_losses(self, x_start, t, noise=None, clip_denoised=False):
        noise = default(noise, lambda : torch.randn_like(x_start))
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.model(x_t, t)
        true_mean, _, true_log_variance_clipped = self.q_posterior(x_start=x_start, x_t=x_t, t=t)
        model_mean, _, model_log_variance, _ = self.p_mean_variance(x=x_t, t=t, clip_denoised=clip_denoised, model_output=model_output)
        detached_model_mean = model_mean.detach()
        kl = normal_kl(true_mean, true_log_variance_clipped, detached_model_mean, model_log_variance)
        kl = meanflat(kl) * NAT
        decoder_nll = -discretized_gaussian_log_likelihood(x_start, means=detached_model_mean, log_scales=0.5 * model_log_variance)
        decoder_nll = meanflat(decoder_nll) * NAT
        vb_losses = torch.where(t == 0, decoder_nll, kl)
        pred_noise, _ = model_output.chunk(2, dim=1)
        simple_losses = F.mse_loss(pred_noise, noise)
        return simple_losses + vb_losses.mean() * self.vb_loss_weight


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


class FeedForward(nn.Module):

    def __init__(self, dim, cond_dim, mult=4, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(dim, scale=False)
        dim_hidden = dim * mult
        self.to_scale_shift = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim_hidden * 2), Rearrange('b d -> b 1 d'))
        to_scale_shift_linear = self.to_scale_shift[-2]
        nn.init.zeros_(to_scale_shift_linear.weight)
        nn.init.zeros_(to_scale_shift_linear.bias)
        self.proj_in = nn.Sequential(nn.Linear(dim, dim_hidden, bias=False), nn.SiLU())
        self.proj_out = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim_hidden, dim, bias=False))

    def forward(self, x, t):
        x = self.norm(x)
        x = self.proj_in(x)
        scale, shift = self.to_scale_shift(t).chunk(2, dim=-1)
        x = x * (scale + 1) + shift
        return self.proj_out(x)


class Transformer(nn.Module):

    def __init__(self, dim, time_cond_dim, depth, dim_head=32, heads=4, ff_mult=4, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout), FeedForward(dim=dim, mult=ff_mult, cond_dim=time_cond_dim, dropout=dropout)]))

    def forward(self, x, t):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x, t) + x
        return x


class UViT(nn.Module):

    def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), downsample_factor=2, channels=3, vit_depth=6, vit_dropout=0.2, attn_dim_head=32, attn_heads=4, ff_mult=4, learned_sinusoidal_dim=16, init_img_transform: 'callable'=None, final_img_itransform: 'callable'=None, patch_size=1, dual_patchnorm=False):
        super().__init__()
        if exists(init_img_transform) and exists(final_img_itransform):
            init_shape = torch.Size(1, 1, 32, 32)
            mock_tensor = torch.randn(init_shape)
            assert final_img_itransform(init_img_transform(mock_tensor)).shape == init_shape
        self.init_img_transform = default(init_img_transform, identity)
        self.final_img_itransform = default(final_img_itransform, identity)
        input_channels = channels
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)
        self.unpatchify = identity
        input_channels = channels * patch_size ** 2
        needs_patch = patch_size > 1
        if needs_patch:
            if not dual_patchnorm:
                self.init_conv = nn.Conv2d(channels, init_dim, patch_size, stride=patch_size)
            else:
                self.init_conv = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b h w (c p1 p2)', p1=patch_size, p2=patch_size), nn.LayerNorm(input_channels), nn.Linear(input_channels, init_dim), nn.LayerNorm(init_dim), Rearrange('b h w c -> b c h w'))
            self.unpatchify = nn.ConvTranspose2d(input_channels, channels, patch_size, stride=patch_size)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        time_dim = dim * 4
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1
        self.time_mlp = nn.Sequential(sinu_pos_emb, nn.Linear(fourier_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))
        downsample_factor = cast_tuple(downsample_factor, len(dim_mults))
        assert len(downsample_factor) == len(dim_mults)
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, ((dim_in, dim_out), factor) in enumerate(zip(in_out, downsample_factor)):
            is_last = ind >= num_resolutions - 1
            self.downs.append(nn.ModuleList([ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim), ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim), LinearAttention(dim_in), Downsample(dim_in, dim_out, factor=factor)]))
        mid_dim = dims[-1]
        self.vit = Transformer(dim=mid_dim, time_cond_dim=time_dim, depth=vit_depth, dim_head=attn_dim_head, heads=attn_heads, ff_mult=ff_mult, dropout=vit_dropout)
        for ind, ((dim_in, dim_out), factor) in enumerate(zip(reversed(in_out), reversed(downsample_factor))):
            is_last = ind == len(in_out) - 1
            self.ups.append(nn.ModuleList([Upsample(dim_out, dim_in, factor=factor), ResnetBlock(dim_in * 2, dim_in, time_emb_dim=time_dim), ResnetBlock(dim_in * 2, dim_in, time_emb_dim=time_dim), LinearAttention(dim_in)]))
        default_out_dim = input_channels
        self.out_dim = default(out_dim, default_out_dim)
        self.final_res_block = ResnetBlock(init_dim * 2, init_dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    def forward(self, x, time):
        x = self.init_img_transform(x)
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        x = rearrange(x, 'b c h w -> b h w c')
        x, ps = pack([x], 'b * c')
        x = self.vit(x, t)
        x, = unpack(x, ps, 'b * c')
        x = rearrange(x, 'b h w c -> b c h w')
        for upsample, block1, block2, attn in self.ups:
            x = upsample(x)
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        x = self.unpatchify(x)
        return self.final_img_itransform(x)


class VParamContinuousTimeGaussianDiffusion(nn.Module):
    """
    a new type of parameterization in v-space proposed in https://arxiv.org/abs/2202.00512 that
    (1) allows for improved distillation over noise prediction objective and
    (2) noted in imagen-video to improve upsampling unets by removing the color shifting artifacts
    """

    def __init__(self, model, *, image_size, channels=3, num_sample_steps=500, clip_sample_denoised=True):
        super().__init__()
        assert model.random_or_learned_sinusoidal_cond
        assert not model.self_condition, 'not supported yet'
        self.model = model
        self.channels = channels
        self.image_size = image_size
        self.log_snr = alpha_cosine_log_snr
        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised

    @property
    def device(self):
        return next(self.model.parameters()).device

    def p_mean_variance(self, x, time, time_next):
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)
        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()
        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))
        batch_log_snr = repeat(log_snr, ' -> b', b=x.shape[0])
        pred_v = self.model(x, batch_log_snr)
        x_start = alpha * x - sigma * pred_v
        if self.clip_sample_denoised:
            x_start.clamp_(-1.0, 1.0)
        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
        posterior_variance = squared_sigma_next * c
        return model_mean, posterior_variance

    @torch.no_grad()
    def p_sample(self, x, time, time_next):
        batch, *_, device = *x.shape, x.device
        model_mean, model_variance = self.p_mean_variance(x=x, time=time, time_next=time_next)
        if time_next == 0:
            return model_mean
        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch = shape[0]
        img = torch.randn(shape, device=self.device)
        steps = torch.linspace(1.0, 0.0, self.num_sample_steps + 1, device=self.device)
        for i in tqdm(range(self.num_sample_steps), desc='sampling loop time step', total=self.num_sample_steps):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, times, times_next)
        img.clamp_(-1.0, 1.0)
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size=16):
        return self.p_sample_loop((batch_size, self.channels, self.image_size, self.image_size))

    @autocast('cuda', enabled=False)
    def q_sample(self, x_start, times, noise=None):
        noise = default(noise, lambda : torch.randn_like(x_start))
        log_snr = self.log_snr(times)
        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised = x_start * alpha + noise * sigma
        return x_noised, log_snr, alpha, sigma

    def random_times(self, batch_size):
        return torch.zeros((batch_size,), device=self.device).float().uniform_(0, 1)

    def p_losses(self, x_start, times, noise=None):
        noise = default(noise, lambda : torch.randn_like(x_start))
        x, log_snr, alpha, sigma = self.q_sample(x_start=x_start, times=times, noise=noise)
        v = alpha * noise - sigma * x_start
        model_out = self.model(x, log_snr)
        return F.mse_loss(model_out, v)

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        times = self.random_times(b)
        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, times, *args, **kwargs)


class WeightedObjectiveGaussianDiffusion(GaussianDiffusion):

    def __init__(self, model, *args, pred_noise_loss_weight=0.1, pred_x_start_loss_weight=0.1, **kwargs):
        super().__init__(model, *args, **kwargs)
        channels = model.channels
        assert model.out_dim == channels * 2 + 2, 'dimension out (out_dim) of unet must be twice the number of channels + 2 (for the softmax weighted sum) - for channels of 3, this should be (3 * 2) + 2 = 8'
        assert not model.self_condition, 'not supported yet'
        assert not self.is_ddim_sampling, 'ddim sampling cannot be used'
        self.split_dims = channels, channels, 2
        self.pred_noise_loss_weight = pred_noise_loss_weight
        self.pred_x_start_loss_weight = pred_x_start_loss_weight

    def p_mean_variance(self, *, x, t, clip_denoised, model_output=None):
        model_output = self.model(x, t)
        pred_noise, pred_x_start, weights = model_output.split(self.split_dims, dim=1)
        normalized_weights = weights.softmax(dim=1)
        x_start_from_noise = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        x_starts = torch.stack((x_start_from_noise, pred_x_start), dim=1)
        weighted_x_start = einsum('b j h w, b j c h w -> b c h w', normalized_weights, x_starts)
        if clip_denoised:
            weighted_x_start.clamp_(-1.0, 1.0)
        model_mean, model_variance, model_log_variance = self.q_posterior(weighted_x_start, x, t)
        return model_mean, model_variance, model_log_variance

    def p_losses(self, x_start, t, noise=None, clip_denoised=False):
        noise = default(noise, lambda : torch.randn_like(x_start))
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.model(x_t, t)
        pred_noise, pred_x_start, weights = model_output.split(self.split_dims, dim=1)
        noise_loss = F.mse_loss(noise, pred_noise) * self.pred_noise_loss_weight
        x_start_loss = F.mse_loss(x_start, pred_x_start) * self.pred_x_start_loss_weight
        x_start_from_pred_noise = self.predict_start_from_noise(x_t, t, pred_noise)
        x_start_from_pred_noise = x_start_from_pred_noise.clamp(-2.0, 2.0)
        weighted_x_start = einsum('b j h w, b j c h w -> b c h w', weights.softmax(dim=1), torch.stack((x_start_from_pred_noise, pred_x_start), dim=1))
        weighted_x_start_loss = F.mse_loss(x_start, weighted_x_start)
        return weighted_x_start_loss + x_start_loss + noise_loss


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Attend,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Block,
     lambda: ([], {'dim': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Gain,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MPAdd,
     lambda: ([], {'t': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MPCat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MPSiLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MonotonicLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PreNorm,
     lambda: ([], {'dim': 4, 'fn': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RMSNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Residual,
     lambda: ([], {'fn': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResnetBlock,
     lambda: ([], {'dim': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

