
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


import functools


AttentionConfig = namedtuple('AttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


def exists(val):
    return val is not None


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

    def __init__(self, dropout=0.0, flash=False, causal=False):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.causal = causal
        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'
        self.cpu_config = AttentionConfig(True, True, True)
        self.cuda_config = None
        if not torch.cuda.is_available() or not flash:
            return
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda
        q, k, v = map(lambda t: t.contiguous(), (q, k, v))
        config = self.cuda_config if is_cuda else self.cpu_config
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=self.causal)
        return out

    def forward(self, q, k, v, bias=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device
        if self.flash:
            assert not exists(bias)
            return self.flash_attn(q, k, v)
        scale = q.shape[-1] ** -0.5
        sim = einsum(f'b h i d, b h j d -> b h i j', q, k) * scale
        if exists(bias):
            sim = sim + bias
        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = einsum(f'b h i j, b h j d -> b h i d', attn, v)
        return out


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim, theta=10000):
        super().__init__()
        self.theta = theta
        self.dim = dim

    def forward(self, x):
        dtype, device = x.dtype, x.device
        assert dtype == torch.float, 'input to sinusoidal pos emb must be a float type'
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim=-1).type(dtype)


class RMSNorm(nn.Module):

    def __init__(self, chan, dim=1):
        super().__init__()
        self.dim = dim
        self.gamma = nn.Parameter(torch.ones(chan))

    def forward(self, x):
        dim = self.dim
        right_ones = dim + 1 if dim < 0 else x.ndim - 1 - dim
        gamma = self.gamma.reshape(-1, *((1,) * right_ones))
        return F.normalize(x, dim=dim) * x.shape[dim] ** 0.5 * gamma


class GEGLU(nn.Module):

    def forward(self, x):
        x, gate = x.chunk(2, dim=1)
        return x * F.gelu(gate)


def shift_token(t):
    t, t_shift = t.chunk(2, dim=1)
    t_shift = F.pad(t_shift, (0, 0, 0, 0, 1, -1), value=0.0)
    return torch.cat((t, t_shift), dim=1)


class FeedForward(nn.Module):

    def __init__(self, dim, mult=4):
        super().__init__()
        inner_dim = int(dim * mult * 2 / 3)
        self.proj_in = nn.Sequential(nn.Conv3d(dim, inner_dim * 2, 1, bias=False), GEGLU())
        self.proj_out = nn.Sequential(RMSNorm(inner_dim), nn.Conv3d(inner_dim, dim, 1, bias=False))

    def forward(self, x, enable_time=True):
        is_video = x.ndim == 5
        enable_time &= is_video
        if not is_video:
            x = rearrange(x, 'b c h w -> b c 1 h w')
        x = self.proj_in(x)
        if enable_time:
            x = shift_token(x)
        out = self.proj_out(x)
        if not is_video:
            out = rearrange(out, 'b c 1 h w -> b c h w')
        return out


class ContinuousPositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(self, *, dim, heads, num_dims=1, layers=2):
        super().__init__()
        self.num_dims = num_dims
        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(self.num_dims, dim), nn.SiLU()))
        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), nn.SiLU()))
        self.net.append(nn.Linear(dim, heads))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, *dimensions):
        device = self.device
        shape = torch.tensor(dimensions, device=device)
        rel_pos_shape = 2 * shape - 1
        strides = torch.flip(rel_pos_shape, (0,)).cumprod(dim=-1)
        strides = torch.flip(F.pad(strides, (1, -1), value=1), (0,))
        positions = [torch.arange(d, device=device) for d in dimensions]
        grid = torch.stack(torch.meshgrid(*positions, indexing='ij'), dim=-1)
        grid = rearrange(grid, '... c -> (...) c')
        rel_dist = rearrange(grid, 'i c -> i 1 c') - rearrange(grid, 'j c -> 1 j c')
        rel_positions = [torch.arange(-d + 1, d, device=device) for d in dimensions]
        rel_pos_grid = torch.stack(torch.meshgrid(*rel_positions, indexing='ij'), dim=-1)
        rel_pos_grid = rearrange(rel_pos_grid, '... c -> (...) c')
        bias = rel_pos_grid.float()
        for layer in self.net:
            bias = layer(bias)
        rel_dist += shape - 1
        rel_dist *= strides
        rel_dist_indices = rel_dist.sum(dim=-1)
        bias = bias[rel_dist_indices]
        return rearrange(bias, 'i j h -> h i j')


class Attention(nn.Module):

    def __init__(self, dim, dim_head=64, heads=8, flash=False, causal=False):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.attend = Attend(flash=flash, causal=causal)
        self.norm = RMSNorm(dim, dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        nn.init.zeros_(self.to_out.weight.data)

    def forward(self, x, rel_pos_bias=None):
        x = self.norm(x)
        q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        out = self.attend(q, k, v, bias=rel_pos_bias)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def default(val, d):
    return val if exists(val) else d


class PseudoConv3d(nn.Module):

    def __init__(self, dim, dim_out=None, kernel_size=3, *, temporal_kernel_size=None, **kwargs):
        super().__init__()
        dim_out = default(dim_out, dim)
        temporal_kernel_size = default(temporal_kernel_size, kernel_size)
        self.spatial_conv = nn.Conv2d(dim, dim_out, kernel_size=kernel_size, padding=kernel_size // 2)
        self.temporal_conv = nn.Conv1d(dim_out, dim_out, kernel_size=temporal_kernel_size, padding=temporal_kernel_size // 2) if kernel_size > 1 else None
        if exists(self.temporal_conv):
            nn.init.dirac_(self.temporal_conv.weight.data)
            nn.init.zeros_(self.temporal_conv.bias.data)

    def forward(self, x, enable_time=True):
        b, c, *_, h, w = x.shape
        is_video = x.ndim == 5
        enable_time &= is_video
        if is_video:
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        x = self.spatial_conv(x)
        if is_video:
            x = rearrange(x, '(b f) c h w -> b c f h w', b=b)
        if not enable_time or not exists(self.temporal_conv):
            return x
        x = rearrange(x, 'b c f h w -> (b h w) c f')
        x = self.temporal_conv(x)
        x = rearrange(x, '(b h w) c f -> b c f h w', h=h, w=w)
        return x


class SpatioTemporalAttention(nn.Module):

    def __init__(self, dim, *, dim_head=64, heads=8, add_feed_forward=True, ff_mult=4, pos_bias=True, flash=False, causal_time_attn=False):
        super().__init__()
        assert not (flash and pos_bias), 'learned positional attention bias is not compatible with flash attention'
        self.spatial_attn = Attention(dim=dim, dim_head=dim_head, heads=heads, flash=flash)
        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim // 2, heads=heads, num_dims=2) if pos_bias else None
        self.temporal_attn = Attention(dim=dim, dim_head=dim_head, heads=heads, flash=flash, causal=causal_time_attn)
        self.temporal_rel_pos_bias = ContinuousPositionBias(dim=dim // 2, heads=heads, num_dims=1) if pos_bias else None
        self.has_feed_forward = add_feed_forward
        if not add_feed_forward:
            return
        self.ff = FeedForward(dim=dim, mult=ff_mult)

    def forward(self, x, enable_time=True):
        b, c, *_, h, w = x.shape
        is_video = x.ndim == 5
        enable_time &= is_video
        if is_video:
            x = rearrange(x, 'b c f h w -> (b f) (h w) c')
        else:
            x = rearrange(x, 'b c h w -> b (h w) c')
        space_rel_pos_bias = self.spatial_rel_pos_bias(h, w) if exists(self.spatial_rel_pos_bias) else None
        x = self.spatial_attn(x, rel_pos_bias=space_rel_pos_bias) + x
        if is_video:
            x = rearrange(x, '(b f) (h w) c -> b c f h w', b=b, h=h, w=w)
        else:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        if enable_time:
            x = rearrange(x, 'b c f h w -> (b h w) f c')
            time_rel_pos_bias = self.temporal_rel_pos_bias(x.shape[1]) if exists(self.temporal_rel_pos_bias) else None
            x = self.temporal_attn(x, rel_pos_bias=time_rel_pos_bias) + x
            x = rearrange(x, '(b h w) f c -> b c f h w', w=w, h=h)
        if self.has_feed_forward:
            x = self.ff(x, enable_time=enable_time) + x
        return x


class Block(nn.Module):

    def __init__(self, dim, dim_out, kernel_size=3, temporal_kernel_size=None):
        super().__init__()
        self.project = PseudoConv3d(dim, dim_out, 3)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None, enable_time=False):
        x = self.project(x, enable_time=enable_time)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):

    def __init__(self, dim, dim_out, *, timestep_cond_dim=None):
        super().__init__()
        self.timestep_mlp = None
        if exists(timestep_cond_dim):
            self.timestep_mlp = nn.Sequential(nn.SiLU(), nn.Linear(timestep_cond_dim, dim_out * 2))
        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = PseudoConv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, timestep_emb=None, enable_time=True):
        assert not exists(timestep_emb) ^ exists(self.timestep_mlp)
        scale_shift = None
        if exists(self.timestep_mlp) and exists(timestep_emb):
            time_emb = self.timestep_mlp(timestep_emb)
            to_einsum_eq = 'b c 1 1 1' if x.ndim == 5 else 'b c 1 1'
            time_emb = rearrange(time_emb, f'b c -> {to_einsum_eq}')
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift, enable_time=enable_time)
        h = self.block2(h, enable_time=enable_time)
        return h + self.res_conv(x)


class Downsample(nn.Module):

    def __init__(self, dim, downsample_space=True, downsample_time=False, nonlin=False):
        super().__init__()
        assert downsample_space or downsample_time
        self.down_space = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2), nn.Conv2d(dim * 4, dim, 1, bias=False), nn.SiLU() if nonlin else nn.Identity()) if downsample_space else None
        self.down_time = nn.Sequential(Rearrange('b c (f p) h w -> b (c p) f h w', p=2), nn.Conv3d(dim * 2, dim, 1, bias=False), nn.SiLU() if nonlin else nn.Identity()) if downsample_time else None

    def forward(self, x, enable_time=True):
        is_video = x.ndim == 5
        if is_video:
            x = rearrange(x, 'b c f h w -> b f c h w')
            x, ps = pack([x], '* c h w')
        if exists(self.down_space):
            x = self.down_space(x)
        if is_video:
            x, = unpack(x, ps, '* c h w')
            x = rearrange(x, 'b f c h w -> b c f h w')
        if not is_video or not exists(self.down_time) or not enable_time:
            return x
        x = self.down_time(x)
        return x


class Upsample(nn.Module):

    def __init__(self, dim, upsample_space=True, upsample_time=False, nonlin=False):
        super().__init__()
        assert upsample_space or upsample_time
        self.up_space = nn.Sequential(nn.Conv2d(dim, dim * 4, 1), nn.SiLU() if nonlin else nn.Identity(), Rearrange('b (c p1 p2) h w -> b c (h p1) (w p2)', p1=2, p2=2)) if upsample_space else None
        self.up_time = nn.Sequential(nn.Conv3d(dim, dim * 2, 1), nn.SiLU() if nonlin else nn.Identity(), Rearrange('b (c p) f h w -> b c (f p) h w', p=2)) if upsample_time else None
        self.init_()

    def init_(self):
        if exists(self.up_space):
            self.init_conv_(self.up_space[0], 4)
        if exists(self.up_time):
            self.init_conv_(self.up_time[0], 2)

    def init_conv_(self, conv, factor):
        o, *remain_dims = conv.weight.shape
        conv_weight = torch.empty(o // factor, *remain_dims)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r=factor)
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x, enable_time=True):
        is_video = x.ndim == 5
        if is_video:
            x = rearrange(x, 'b c f h w -> b f c h w')
            x, ps = pack([x], '* c h w')
        if exists(self.up_space):
            x = self.up_space(x)
        if is_video:
            x, = unpack(x, ps, '* c h w')
            x = rearrange(x, 'b f c h w -> b c f h w')
        if not is_video or not exists(self.up_time) or not enable_time:
            return x
        x = self.up_time(x)
        return x


def divisible_by(numer, denom):
    return numer % denom == 0


mlist = nn.ModuleList


class SpaceTimeUnet(nn.Module):

    def __init__(self, *, dim, channels=3, dim_mult=(1, 2, 4, 8), self_attns=(False, False, False, True), temporal_compression=(False, True, True, True), resnet_block_depths=(2, 2, 2, 2), attn_dim_head=64, attn_heads=8, condition_on_timestep=True, attn_pos_bias=True, flash_attn=False, causal_time_attn=False):
        super().__init__()
        assert len(dim_mult) == len(self_attns) == len(temporal_compression) == len(resnet_block_depths)
        num_layers = len(dim_mult)
        dims = [dim, *map(lambda mult: mult * dim, dim_mult)]
        dim_in_out = zip(dims[:-1], dims[1:])
        self.frame_multiple = 2 ** sum(tuple(map(int, temporal_compression)))
        self.image_size_multiple = 2 ** num_layers
        self.to_timestep_cond = None
        timestep_cond_dim = dim * 4 if condition_on_timestep else None
        if condition_on_timestep:
            self.to_timestep_cond = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, timestep_cond_dim), nn.SiLU())
        self.downs = mlist([])
        self.ups = mlist([])
        attn_kwargs = dict(dim_head=attn_dim_head, heads=attn_heads, pos_bias=attn_pos_bias, flash=flash_attn, causal_time_attn=causal_time_attn)
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, timestep_cond_dim=timestep_cond_dim)
        self.mid_attn = SpatioTemporalAttention(dim=mid_dim, **attn_kwargs)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, timestep_cond_dim=timestep_cond_dim)
        for _, self_attend, (dim_in, dim_out), compress_time, resnet_block_depth in zip(range(num_layers), self_attns, dim_in_out, temporal_compression, resnet_block_depths):
            assert resnet_block_depth >= 1
            self.downs.append(mlist([ResnetBlock(dim_in, dim_out, timestep_cond_dim=timestep_cond_dim), mlist([ResnetBlock(dim_out, dim_out) for _ in range(resnet_block_depth)]), SpatioTemporalAttention(dim=dim_out, **attn_kwargs) if self_attend else None, Downsample(dim_out, downsample_time=compress_time)]))
            self.ups.append(mlist([ResnetBlock(dim_out * 2, dim_in, timestep_cond_dim=timestep_cond_dim), mlist([ResnetBlock(dim_in + (dim_out if ind == 0 else 0), dim_in) for ind in range(resnet_block_depth)]), SpatioTemporalAttention(dim=dim_in, **attn_kwargs) if self_attend else None, Upsample(dim_out, upsample_time=compress_time)]))
        self.skip_scale = 2 ** -0.5
        self.conv_in = PseudoConv3d(dim=channels, dim_out=dim, kernel_size=7, temporal_kernel_size=3)
        self.conv_out = PseudoConv3d(dim=dim, dim_out=channels, kernel_size=3, temporal_kernel_size=3)

    def forward(self, x, timestep=None, enable_time=True):
        assert not exists(self.to_timestep_cond) ^ exists(timestep)
        is_video = x.ndim == 5
        if enable_time and is_video:
            frames = x.shape[2]
            assert divisible_by(frames, self.frame_multiple), f'number of frames on the video ({frames}) must be divisible by the frame multiple ({self.frame_multiple})'
        height, width = x.shape[-2:]
        assert divisible_by(height, self.image_size_multiple) and divisible_by(width, self.image_size_multiple), f'height and width of the image or video must be a multiple of {self.image_size_multiple}'
        t = self.to_timestep_cond(rearrange(timestep, '... -> (...)')) if exists(timestep) else None
        x = self.conv_in(x, enable_time=enable_time)
        hiddens = []
        for init_block, blocks, maybe_attention, downsample in self.downs:
            x = init_block(x, t, enable_time=enable_time)
            hiddens.append(x.clone())
            for block in blocks:
                x = block(x, enable_time=enable_time)
            if exists(maybe_attention):
                x = maybe_attention(x, enable_time=enable_time)
            hiddens.append(x.clone())
            x = downsample(x, enable_time=enable_time)
        x = self.mid_block1(x, t, enable_time=enable_time)
        x = self.mid_attn(x, enable_time=enable_time)
        x = self.mid_block2(x, t, enable_time=enable_time)
        for init_block, blocks, maybe_attention, upsample in reversed(self.ups):
            x = upsample(x, enable_time=enable_time)
            x = torch.cat((hiddens.pop() * self.skip_scale, x), dim=1)
            x = init_block(x, t, enable_time=enable_time)
            x = torch.cat((hiddens.pop() * self.skip_scale, x), dim=1)
            for block in blocks:
                x = block(x, enable_time=enable_time)
            if exists(maybe_attention):
                x = maybe_attention(x, enable_time=enable_time)
        x = self.conv_out(x, enable_time=enable_time)
        return x


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
    (GEGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PseudoConv3d,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RMSNorm,
     lambda: ([], {'chan': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResnetBlock,
     lambda: ([], {'dim': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

