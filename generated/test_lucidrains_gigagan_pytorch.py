
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


from functools import partial


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torchvision import transforms as T


from torch.autograd import Function


import torch.distributed as dist


from math import log2


from math import sqrt


from random import random


from torchvision import utils


from torch import Tensor


from torch.autograd import grad as torch_grad


from torch.cuda.amp import GradScaler


from torch.optim import AdamW


from torch.optim import Adam


from itertools import islice


from torch.nn import Module


from torch.nn import ModuleList


AttentionConfig = namedtuple('AttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


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

    def __init__(self, dropout=0.0, flash=False):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
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
        is_cuda = q.is_cuda
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
        if self.flash:
            return self.flash_attn(q, k, v)
        scale = q.shape[-1] ** -0.5
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * scale
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        return out


class DiffAugment(nn.Module):

    def __init__(self, *, prob, horizontal_flip, horizontal_flip_prob=0.5):
        super().__init__()
        self.prob = prob
        assert 0 <= prob <= 1.0
        self.horizontal_flip = horizontal_flip
        self.horizontal_flip_prob = horizontal_flip_prob

    def forward(self, images, rgbs: 'List[Tensor]'):
        if random() >= self.prob:
            return images, rgbs
        if random() < self.horizontal_flip_prob:
            images = torch.flip(images, (-1,))
            rgbs = [torch.flip(rgb, (-1,)) for rgb in rgbs]
        return images, rgbs


class ChannelRMSNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        normed = F.normalize(x, dim=1)
        return normed * self.scale * self.gamma


class RMSNorm(Module):

    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        spatial_dims = (1,) * (x.ndim - 2)
        gamma = self.gamma.reshape(-1, *spatial_dims)
        return F.normalize(x, dim=1) * gamma * self.scale


class Blur(nn.Module):

    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class PixelShuffleUpsample(nn.Module):

    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * 4, 1)
        self.net = nn.Sequential(conv, nn.SiLU(), nn.PixelShuffle(2))
        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)


def get_same_padding(size, kernel, dilation, stride):
    return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2


class AdaptiveConv2DMod(nn.Module):

    def __init__(self, dim, dim_out, kernel, *, demod=True, stride=1, dilation=1, eps=1e-08, num_conv_kernels=1):
        super().__init__()
        self.eps = eps
        self.dim_out = dim_out
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.adaptive = num_conv_kernels > 1
        self.weights = nn.Parameter(torch.randn((num_conv_kernels, dim_out, dim, kernel, kernel)))
        self.demod = demod
        nn.init.kaiming_normal_(self.weights, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, fmap, mod: 'Tensor', kernel_mod: 'Tensor | None'=None):
        """
        notation

        b - batch
        n - convs
        o - output
        i - input
        k - kernel
        """
        b, h = fmap.shape[0], fmap.shape[-2]
        if mod.shape[0] != b:
            mod = repeat(mod, 'b ... -> (s b) ...', s=b // mod.shape[0])
        if exists(kernel_mod):
            kernel_mod_has_el = kernel_mod.numel() > 0
            assert self.adaptive or not kernel_mod_has_el
            if kernel_mod_has_el and kernel_mod.shape[0] != b:
                kernel_mod = repeat(kernel_mod, 'b ... -> (s b) ...', s=b // kernel_mod.shape[0])
        weights = self.weights
        if self.adaptive:
            weights = repeat(weights, '... -> b ...', b=b)
            assert exists(kernel_mod) and kernel_mod.numel() > 0
            kernel_attn = kernel_mod.softmax(dim=-1)
            kernel_attn = rearrange(kernel_attn, 'b n -> b n 1 1 1 1')
            weights = reduce(weights * kernel_attn, 'b n ... -> b ...', 'sum')
        mod = rearrange(mod, 'b i -> b 1 i 1 1')
        weights = weights * (mod + 1)
        if self.demod:
            inv_norm = reduce(weights ** 2, 'b o i k1 k2 -> b o 1 1 1', 'sum').clamp(min=self.eps).rsqrt()
            weights = weights * inv_norm
        fmap = rearrange(fmap, 'b c h w -> 1 (b c) h w')
        weights = rearrange(weights, 'b o ... -> (b o) ...')
        padding = get_same_padding(h, self.kernel, self.dilation, self.stride)
        fmap = F.conv2d(fmap, weights, padding=padding, groups=b)
        return rearrange(fmap, '1 (b o) ... -> b o ...', b=b)


class AdaptiveConv1DMod(nn.Module):
    """ 1d version of adaptive conv, for time dimension in videogigagan """

    def __init__(self, dim, dim_out, kernel, *, demod=True, stride=1, dilation=1, eps=1e-08, num_conv_kernels=1):
        super().__init__()
        self.eps = eps
        self.dim_out = dim_out
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.adaptive = num_conv_kernels > 1
        self.weights = nn.Parameter(torch.randn((num_conv_kernels, dim_out, dim, kernel)))
        self.demod = demod
        nn.init.kaiming_normal_(self.weights, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, fmap, mod: 'Tensor', kernel_mod: 'Tensor | None'=None):
        """
        notation

        b - batch
        n - convs
        o - output
        i - input
        k - kernel
        """
        b, t = fmap.shape[0], fmap.shape[-1]
        if mod.shape[0] != b:
            mod = repeat(mod, 'b ... -> (s b) ...', s=b // mod.shape[0])
        if exists(kernel_mod):
            kernel_mod_has_el = kernel_mod.numel() > 0
            assert self.adaptive or not kernel_mod_has_el
            if kernel_mod_has_el and kernel_mod.shape[0] != b:
                kernel_mod = repeat(kernel_mod, 'b ... -> (s b) ...', s=b // kernel_mod.shape[0])
        weights = self.weights
        if self.adaptive:
            weights = repeat(weights, '... -> b ...', b=b)
            assert exists(kernel_mod) and kernel_mod.numel() > 0
            kernel_attn = kernel_mod.softmax(dim=-1)
            kernel_attn = rearrange(kernel_attn, 'b n -> b n 1 1 1')
            weights = reduce(weights * kernel_attn, 'b n ... -> b ...', 'sum')
        mod = rearrange(mod, 'b i -> b 1 i 1')
        weights = weights * (mod + 1)
        if self.demod:
            inv_norm = reduce(weights ** 2, 'b o i k -> b o 1 1', 'sum').clamp(min=self.eps).rsqrt()
            weights = weights * inv_norm
        fmap = rearrange(fmap, 'b c t -> 1 (b c) t')
        weights = rearrange(weights, 'b o ... -> (b o) ...')
        padding = get_same_padding(t, self.kernel, self.dilation, self.stride)
        fmap = F.conv1d(fmap, weights, padding=padding, groups=b)
        return rearrange(fmap, '1 (b o) ... -> b o ...', b=b)


class SelfAttention(nn.Module):

    def __init__(self, dim, dim_head=64, heads=8, dot_product=False):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads
        self.dot_product = dot_product
        self.norm = ChannelRMSNorm(dim)
        self.to_q = nn.Conv2d(dim, dim_inner, 1, bias=False)
        self.to_k = nn.Conv2d(dim, dim_inner, 1, bias=False) if dot_product else None
        self.to_v = nn.Conv2d(dim, dim_inner, 1, bias=False)
        self.null_kv = nn.Parameter(torch.randn(2, heads, dim_head))
        self.to_out = nn.Conv2d(dim_inner, dim, 1, bias=False)

    def forward(self, fmap):
        """
        einstein notation

        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        i - source seq (attend from)
        j - target seq (attend to)
        """
        batch = fmap.shape[0]
        fmap = self.norm(fmap)
        x, y = fmap.shape[-2:]
        h = self.heads
        q, v = self.to_q(fmap), self.to_v(fmap)
        k = self.to_k(fmap) if exists(self.to_k) else q
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h=self.heads), (q, k, v))
        nk, nv = map(lambda t: repeat(t, 'h d -> (b h) 1 d', b=batch), self.null_kv)
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)
        if self.dot_product:
            sim = einsum('b i d, b j d -> b i j', q, k)
        else:
            q_squared = (q * q).sum(dim=-1)
            k_squared = (k * k).sum(dim=-1)
            l2dist_squared = rearrange(q_squared, 'b i -> b i 1') + rearrange(k_squared, 'b j -> b 1 j') - 2 * einsum('b i d, b j d -> b i j', q, k)
            sim = -l2dist_squared
        sim = sim * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', x=x, y=y, h=h)
        return self.to_out(out)


class CrossAttention(nn.Module):

    def __init__(self, dim, dim_context, dim_head=64, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads
        kv_input_dim = default(dim_context, dim)
        self.norm = ChannelRMSNorm(dim)
        self.norm_context = RMSNorm(kv_input_dim)
        self.to_q = nn.Conv2d(dim, dim_inner, 1, bias=False)
        self.to_kv = nn.Linear(kv_input_dim, dim_inner * 2, bias=False)
        self.to_out = nn.Conv2d(dim_inner, dim, 1, bias=False)

    def forward(self, fmap, context, mask=None):
        """
        einstein notation

        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        i - source seq (attend from)
        j - target seq (attend to)
        """
        fmap = self.norm(fmap)
        context = self.norm_context(context)
        x, y = fmap.shape[-2:]
        h = self.heads
        q, k, v = self.to_q(fmap), *self.to_kv(context).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k, v))
        q = rearrange(q, 'b (h d) x y -> (b h) (x y) d', h=self.heads)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        if exists(mask):
            mask = repeat(mask, 'b j -> (b h) 1 j', h=self.heads)
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', x=x, y=y, h=h)
        return self.to_out(out)


class TextAttention(nn.Module):

    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.null_kv = nn.Parameter(torch.randn(2, heads, dim_head))
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    def forward(self, encodings, mask=None):
        """
        einstein notation

        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        i - source seq (attend from)
        j - target seq (attend to)
        """
        batch = encodings.shape[0]
        encodings = self.norm(encodings)
        h = self.heads
        q, k, v = self.to_qkv(encodings).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v))
        nk, nv = map(lambda t: repeat(t, 'h d -> (b h) 1 d', b=batch), self.null_kv)
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = repeat(mask, 'b n -> (b h) 1 n', h=h)
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


def FeedForward(dim, mult=4):
    return nn.Sequential(RMSNorm(dim), nn.Conv2d(dim, dim * mult, 1), nn.GELU(), nn.Conv2d(dim * mult, dim, 1))


class SelfAttentionBlock(nn.Module):

    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4, dot_product=False):
        super().__init__()
        self.attn = SelfAttention(dim=dim, dim_head=dim_head, heads=heads, dot_product=dot_product)
        self.ff = FeedForward(dim=dim, mult=ff_mult, channel_first=True)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, dim_context, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.attn = CrossAttention(dim=dim, dim_context=dim_context, dim_head=dim_head, heads=heads)
        self.ff = FeedForward(dim=dim, mult=ff_mult, channel_first=True)

    def forward(self, x, context, mask=None):
        x = self.attn(x, context=context, mask=mask) + x
        x = self.ff(x) + x
        return x


class Attention(Module):

    def __init__(self, dim, heads=4, dim_head=32, flash=False):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.norm = RMSNorm(dim)
        self.attend = Attend(flash=flash)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h=self.heads), qkv)
        out = self.attend(q, k, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class Transformer(Module):

    def __init__(self, dim, dim_head=64, heads=8, depth=1, flash_attn=True, ff_mult=4):
        super().__init__()
        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(ModuleList([Attention(dim=dim, dim_head=dim_head, heads=heads, flash=flash_attn), FeedForward(dim=dim, mult=ff_mult)]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


def l2norm(t):
    return F.normalize(t, dim=-1)


class EqualLinear(nn.Module):

    def __init__(self, dim, dim_out, lr_mul=1, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim_out, dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim_out))
        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)


def leaky_relu(neg_slope=0.2):
    return nn.LeakyReLU(neg_slope)


class StyleNetwork(nn.Module):

    def __init__(self, dim, depth, lr_mul=0.1, dim_text_latent=0):
        super().__init__()
        self.dim = dim
        self.dim_text_latent = dim_text_latent
        layers = []
        for i in range(depth):
            is_first = i == 0
            dim_in = dim + dim_text_latent if is_first else dim
            layers.extend([EqualLinear(dim_in, dim, lr_mul), leaky_relu()])
        self.net = nn.Sequential(*layers)

    def forward(self, x, text_latent=None):
        x = F.normalize(x, dim=1)
        if self.dim_text_latent > 0:
            assert exists(text_latent)
            x = torch.cat((x, text_latent), dim=-1)
        return self.net(x)


class Noise(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, 1, 1))

    def forward(self, x, noise=None):
        b, _, h, w, device = *x.shape, x.device
        if not exists(noise):
            noise = torch.randn(b, 1, h, w, device=device)
        return x + self.weight * noise


class BaseGenerator(nn.Module):
    pass


def SqueezeExcite(dim, dim_out, reduction=4, dim_min=32):
    dim_hidden = max(dim_out // reduction, dim_min)
    return nn.Sequential(Reduce('b c h w -> b c', 'mean'), nn.Linear(dim, dim_hidden), nn.SiLU(), nn.Linear(dim_hidden, dim_out), nn.Sigmoid(), Rearrange('b c -> b c 1 1'))


def Upsample(*args):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), Blur())


def is_power_of_two(n):
    return log2(n).is_integer()


def safe_unshift(arr):
    if len(arr) == 0:
        return None
    return arr.pop(0)


def conv2d_3x3(dim_in, dim_out):
    return nn.Conv2d(dim_in, dim_out, 3, padding=1)


def divisible_by(numer, denom):
    return numer % denom == 0


class RandomFixedProjection(nn.Module):

    def __init__(self, dim, dim_out, channel_first=True):
        super().__init__()
        weights = torch.randn(dim, dim_out)
        nn.init.kaiming_normal_(weights, mode='fan_out', nonlinearity='linear')
        self.channel_first = channel_first
        self.register_buffer('fixed_weights', weights)

    def forward(self, x):
        if not self.channel_first:
            return x @ self.fixed_weights
        return einsum('b c ..., c d -> b d ...', x, self.fixed_weights)


class Predictor(nn.Module):

    def __init__(self, dim, depth=4, num_conv_kernels=2, unconditional=False):
        super().__init__()
        self.unconditional = unconditional
        self.residual_fn = nn.Conv2d(dim, dim, 1)
        self.residual_scale = 2 ** -0.5
        self.layers = nn.ModuleList([])
        klass = nn.Conv2d if unconditional else partial(AdaptiveConv2DMod, num_conv_kernels=num_conv_kernels)
        klass_kwargs = dict(padding=1) if unconditional else dict()
        for ind in range(depth):
            self.layers.append(nn.ModuleList([klass(dim, dim, 3, **klass_kwargs), leaky_relu(), klass(dim, dim, 3, **klass_kwargs), leaky_relu()]))
        self.to_logits = nn.Conv2d(dim, 1, 1)

    def forward(self, x, mod=None, kernel_mod=None):
        residual = self.residual_fn(x)
        kwargs = dict()
        if not self.unconditional:
            kwargs = dict(mod=mod, kernel_mod=kernel_mod)
        for conv1, activation, conv2, activation in self.layers:
            inner_residual = x
            x = conv1(x, **kwargs)
            x = activation(x)
            x = conv2(x, **kwargs)
            x = activation(x)
            x = x + inner_residual
            x = x * self.residual_scale
        x = x + residual
        return self.to_logits(x)


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


class Downsample(Module):

    def __init__(self, dim, dim_out=None, skip_downsample=False, has_temporal_layers=False):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.skip_downsample = skip_downsample
        self.conv2d = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.has_temporal_layers = has_temporal_layers
        if has_temporal_layers:
            self.conv1d = nn.Conv1d(dim_out, dim_out, 3, padding=1)
            nn.init.dirac_(self.conv1d.weight)
            nn.init.zeros_(self.conv1d.bias)
        self.register_buffer('filter', torch.Tensor([1.0, 2.0, 1.0]))

    def forward(self, x):
        batch = x.shape[0]
        is_input_video = x.ndim == 5
        assert not (is_input_video and not self.has_temporal_layers)
        if is_input_video:
            x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.conv2d(x)
        if is_input_video:
            x = rearrange(x, '(b t) c h w -> b h w c t', b=batch)
            x, ps = pack_one(x, '* c t')
            x = self.conv1d(x)
            x = unpack_one(x, ps, '* c t')
            x = rearrange(x, 'b h w c t -> b c t h w')
        if self.skip_downsample:
            return x, x[:, 0:0]
        before_blur_input = x
        f = self.filter
        N = None
        if is_input_video:
            f = f[N, N, :] * f[N, :, N] * f[:, N, N]
            filter_fn = filter3d
            maxpool_fn = F.max_pool3d
        else:
            f = f[N, :] * f[:, N]
            filter_fn = filter2d
            maxpool_fn = F.max_pool2d
        blurred = filter_fn(x, f[N, ...], normalized=True)
        high_freq_fmap = before_blur_input - blurred
        x = maxpool_fn(x, kernel_size=2)
        return x, high_freq_fmap


def is_unique(arr):
    return len(set(arr)) == len(arr)


TrainDiscrLosses = namedtuple('TrainDiscrLosses', ['divergence', 'multiscale_divergence', 'vision_aided_divergence', 'total_matching_aware_loss', 'gradient_penalty', 'aux_reconstruction'])


TrainGenLosses = namedtuple('TrainGenLosses', ['divergence', 'multiscale_divergence', 'total_vd_divergence', 'contrastive_loss'])


class LinearAttention(Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x):
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
        return self.to_out(out)


class LinearTransformer(Module):

    def __init__(self, dim, dim_head=64, heads=8, depth=1, ff_mult=4):
        super().__init__()
        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(ModuleList([LinearAttention(dim=dim, dim_head=dim_head, heads=heads), FeedForward(dim=dim, mult=ff_mult)]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class PixelShuffleTemporalUpsample(Module):

    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv3d(dim, dim_out * 2, 1)
        self.net = nn.Sequential(conv, nn.SiLU(), Rearrange('b (c p) t h w -> b c (t p) h w', p=2))
        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, t, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 2, i, t, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 2) ...')
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)


def null_iterator():
    while True:
        yield None


class TemporalBlur(Module):

    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = repeat(self.f, 't -> 1 t h w', h=3, w=3)
        return filter3d(x, f, normalized=True)


def interpolate_1d(x, length, mode='bilinear'):
    x = rearrange(x, 'b c t -> b c t 1')
    x = F.interpolate(x, (length, 1), mode=mode)
    return rearrange(x, 'b c t 1 -> b c t')


class TemporalUpsample(Module):

    def __init__(self, dim, dim_out=None):
        super().__init__()
        self.blur = TemporalBlur()

    def forward(self, x):
        assert x.ndim == 5
        time = x.shape[2]
        x = rearrange(x, 'b c t h w -> b h w c t')
        x, ps = pack_one(x, '* c t')
        x = interpolate_1d(x, time * 2, mode='bilinear')
        x = unpack_one(x, ps, '* c t')
        x = rearrange(x, 'b h w c t -> b c t h w')
        x = self.blur(x)
        return x


def cast_tuple(t, length=1):
    if isinstance(t, tuple):
        return t
    return (t,) * length


def fold_space_into_batch(x):
    x = rearrange(x, 'b c t h w -> b h w c t')
    x, ps = pack_one(x, '* c t')

    def split_space_from_batch(out):
        out = unpack_one(x, ps, '* c t')
        out = rearrange(out, 'b h w c t -> b c t h w')
        return out
    return x, split_space_from_batch


def identity(t, *args, **kwargs):
    return t


def pad_dim_to(t, length, dim=0):
    pad_length = length - t.shape[dim]
    zero_pairs = -dim - 1 if dim < 0 else t.ndim - dim - 1
    return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length))


def all_gather_variable_dim(t, dim=0, sizes=None):
    device, world_size = t.device, dist.get_world_size()
    if not exists(sizes):
        size = torch.tensor(t.shape[dim], device=device, dtype=torch.long)
        sizes = [torch.empty_like(size, device=device, dtype=torch.long) for i in range(world_size)]
        dist.all_gather(sizes, size)
        sizes = torch.stack(sizes)
    max_size = sizes.amax().item()
    padded_t = pad_dim_to(t, max_size, dim=dim)
    gathered_tensors = [torch.empty(padded_t.shape, device=device, dtype=padded_t.dtype) for i in range(world_size)]
    dist.all_gather(gathered_tensors, padded_t)
    gathered_tensor = torch.cat(gathered_tensors, dim=dim)
    seq = torch.arange(max_size, device=device)
    mask = rearrange(seq, 'j -> 1 j') < rearrange(sizes, 'i -> i 1')
    mask = rearrange(mask, 'i j -> (i j)')
    seq = torch.arange(mask.shape[-1], device=device)
    indices = seq[mask]
    gathered_tensor = gathered_tensor.index_select(dim, indices)
    return gathered_tensor, sizes


class AllGather(Function):

    @staticmethod
    def forward(ctx, x, dim, sizes):
        is_dist = dist.is_initialized() and dist.get_world_size() > 1
        ctx.is_dist = is_dist
        if not is_dist:
            return x, None
        x, batch_sizes = all_gather_variable_dim(x, dim=dim, sizes=sizes)
        ctx.batch_sizes = batch_sizes.tolist()
        ctx.dim = dim
        return x, batch_sizes

    @staticmethod
    def backward(ctx, grads, _):
        if not ctx.is_dist:
            return grads, None, None
        batch_sizes, rank = ctx.batch_sizes, dist.get_rank()
        grads_by_rank = grads.split(batch_sizes, dim=ctx.dim)
        return grads_by_rank[rank], None, None


all_gather = AllGather.apply


def log(t, eps=1e-20):
    return t.clamp(min=eps).log()


def aux_matching_loss(real, fake):
    """
    making logits negative, as in this framework, discriminator is 0 for real, high value for fake. GANs can have this arbitrarily swapped, as it only matters if the generator and discriminator are opposites
    """
    return (log(1 + (-real).exp()) + log(1 + (-fake).exp())).mean()


def cycle(dl):
    while True:
        for data in dl:
            yield data


def discriminator_hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()


def generator_hinge_loss(fake):
    return fake.mean()


def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []
    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    return wd_params, no_wd_params


def get_optimizer(params, lr=0.0001, wd=0.01, betas=(0.9, 0.99), eps=1e-08, filter_by_requires_grad=True, group_wd_params=True, **kwargs):
    if filter_by_requires_grad:
        params = list(filter(lambda t: t.requires_grad, params))
    if group_wd_params and wd > 0:
        wd_params, no_wd_params = separate_weight_decayable_params(params)
        params = [{'params': wd_params}, {'params': no_wd_params, 'weight_decay': 0}]
    if wd == 0:
        return Adam(params, lr=lr, betas=betas, eps=eps)
    return AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)


def gradient_penalty(images, outputs, grad_output_weights=None, weight=10, scaler: 'GradScaler | None'=None, eps=0.0001):
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    if exists(scaler):
        outputs = [*map(scaler.scale, outputs)]
    if not exists(grad_output_weights):
        grad_output_weights = (1,) * len(outputs)
    maybe_scaled_gradients, *_ = torch_grad(outputs=outputs, inputs=images, grad_outputs=[(torch.ones_like(output) * weight) for output, weight in zip(outputs, grad_output_weights)], create_graph=True, retain_graph=True, only_inputs=True)
    gradients = maybe_scaled_gradients
    if exists(scaler):
        scale = scaler.get_scale()
        inv_scale = 1.0 / max(scale, eps)
        gradients = maybe_scaled_gradients * inv_scale
    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def group_by_num_consecutive(arr, num):
    out = []
    for ind, el in enumerate(arr):
        if ind > 0 and divisible_by(ind, num):
            yield out
            out = []
        out.append(el)
    if len(out) > 0:
        yield out


def mkdir_if_not_exists(path):
    path.mkdir(exist_ok=True, parents=True)


def num_to_groups(num, divisor):
    groups, remainder = divmod(num, divisor)
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Attend,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ChannelRMSNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DiffAugment,
     lambda: ([], {'prob': 0, 'horizontal_flip': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (EqualLinear,
     lambda: ([], {'dim': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Noise,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RMSNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RandomFixedProjection,
     lambda: ([], {'dim': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (StyleNetwork,
     lambda: ([], {'dim': 4, 'depth': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

