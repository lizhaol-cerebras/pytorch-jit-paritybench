
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


from collections import namedtuple


from functools import wraps


import torch


from torch import nn


from torch import einsum


import torch.nn.functional as F


import math


from random import random


from random import randrange


from torch.amp import autocast


from torch import Tensor


from torch.nn import Module


from torch.nn import ModuleList


import re


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.utils.data import Dataset


from torch.utils.data import random_split


EfficientAttentionConfig = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


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

    def __init__(self, causal=False, dropout=0.0, flash=False):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.causal = causal
        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'
        self.cpu_config = EfficientAttentionConfig(True, True, True)
        self.cuda_config = None
        if not torch.cuda.is_available() or not flash:
            return
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = EfficientAttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = EfficientAttentionConfig(False, True, True)

    def get_mask(self, i, j, device):
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)

    def flash_attn(self, q, k, v, mask=None, attn_bias=None):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device
        if k.ndim == 3:
            k = rearrange(k, 'b n d -> b 1 n d')
        if v.ndim == 3:
            v = rearrange(v, 'b n d -> b 1 n d')
        if exists(mask) and mask.ndim != 4:
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask = mask.expand(-1, heads, q_len, -1)
        config = self.cuda_config if is_cuda else self.cpu_config
        causal = self.causal
        if exists(attn_bias):
            mask_value = -torch.finfo(q.dtype).max // 2
            if causal:
                causal_mask = self.get_mask(q_len, k_len, device)
                attn_bias = attn_bias.masked_fill(causal_mask, mask_value)
            if exists(mask):
                attn_bias = attn_bias.masked_fill(~mask, mask_value)
            mask = attn_bias
            causal = False
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0, is_causal=causal)
        return out

    def forward(self, q, k, v, mask=None, attn_bias=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device
        scale = q.shape[-1] ** -0.5
        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'
        if self.flash:
            assert not exists(attn_bias)
            return self.flash_attn(q, k, v, mask=mask)
        sim = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale
        if exists(attn_bias):
            sim = sim + attn_bias
        if self.causal:
            causal_mask = self.get_mask(q_len, k_len, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
        if exists(mask):
            if mask.ndim != 4:
                mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)
        return out


class RotaryEmbedding(Module):

    def __init__(self, dim, theta=10000):
        super().__init__()
        inv_freq = 1.0 / theta ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    @property
    def device(self):
        return next(self.buffers()).device

    @autocast('cuda', enabled=False)
    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs


class T5RelativePositionBias(Module):

    def __init__(self, scale=1.0, num_buckets=32, max_distance=128, heads=8):
        super().__init__()
        self.scale = scale
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, n):
        pos = torch.arange(n, device=self.device).long()
        rel_pos = rearrange(pos, 'j -> 1 j') - rearrange(pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> h i j')
        return bias * self.scale


class Swish(Module):

    def forward(self, x):
        return x * x.sigmoid()


class GLU(Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class DepthWiseConv1d(Module):

    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x, mask=None):
        if exists(mask):
            mask = rearrange(mask, 'b n -> b 1 n')
            x = x.masked_fill(~mask, 0.0)
        x = F.pad(x, self.padding)
        out = self.conv(x)
        if exists(mask):
            out = out.masked_fill(~mask, 0.0)
        return out


class Scale(Module):

    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class ChanLayerNorm(Module):

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-06 if x.dtype == torch.float32 else 0.0001
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * var.clamp(min=eps).rsqrt() * self.gamma


class PreNorm(Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@autocast('cuda', enabled=False)
def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()


def default(val, d):
    return val if exists(val) else d


class Attention(Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, flash=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = Attend(flash=flash, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context=None, mask=None, rotary_emb=None, attn_bias=None, return_values=False, value_residual=None):
        n, device, h, has_context = x.shape[-2], x.device, self.heads, exists(context)
        context = default(context, x)
        q, k, v = self.to_q(x), *self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        if exists(value_residual):
            v = 0.5 * (v + value_residual)
        if exists(rotary_emb):
            q = apply_rotary_pos_emb(rotary_emb, q)
            k = apply_rotary_pos_emb(rotary_emb, k)
        out = self.attend(q, k, v, mask=mask, attn_bias=attn_bias)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        if not return_values:
            return out
        return out, v


class FeedForward(Module):

    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * mult), Swish(), nn.Dropout(dropout), nn.Linear(dim * mult, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return pad, pad - (kernel_size + 1) % 2


class ConformerConvModule(Module):

    def __init__(self, dim, causal=False, expansion_factor=2, kernel_size=31, dropout=0.0):
        super().__init__()
        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)
        self.net1 = nn.Sequential(nn.LayerNorm(dim), Rearrange('b n c -> b c n'), nn.Conv1d(dim, inner_dim * 2, 1), GLU(dim=1))
        self.ds_conv = DepthWiseConv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding)
        self.net2 = nn.Sequential(Swish(), ChanLayerNorm(inner_dim), nn.Conv1d(inner_dim, dim, 1), Rearrange('b c n -> b n c'), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        x = self.net1(x)
        x = self.ds_conv(x, mask=mask)
        return self.net2(x)


class ConformerBlock(Module):

    def __init__(self, *, dim, dim_head=64, heads=8, ff_mult=4, conv_expansion_factor=2, conv_kernel_size=31, attn_dropout=0.0, attn_flash=True, ff_dropout=0.0, conv_dropout=0.0, conv_causal=False, use_gateloop_layers=False):
        super().__init__()
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.gateloop = GateLoop(dim) if use_gateloop_layers else None
        self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, flash=attn_flash)
        self.conv = ConformerConvModule(dim=dim, causal=conv_causal, expansion_factor=conv_expansion_factor, kernel_size=conv_kernel_size, dropout=conv_dropout)
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None, rotary_emb=None, attn_bias=None, attn_value_residual=None, return_values=False):
        x = self.ff1(x) + x
        if exists(self.gateloop):
            x = self.gateloop(x) + x
        attn_out, attn_values = self.attn(x, mask=mask, rotary_emb=rotary_emb, attn_bias=attn_bias, value_residual=attn_value_residual, return_values=True)
        x = attn_out + x
        x = self.conv(x, mask=mask) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        if not return_values:
            return x
        return x, attn_values


class Conformer(Module):

    def __init__(self, dim, *, depth, dim_head=64, heads=8, ff_mult=4, conv_expansion_factor=2, conv_kernel_size=31, attn_dropout=0.0, ff_dropout=0.0, conv_dropout=0.0, conv_causal=False, attn_flash=True, t5_rel_pos_bias=False, use_gateloop_layers=True):
        super().__init__()
        assert not (t5_rel_pos_bias and attn_flash), 'flash attention is not compatible with learned bias'
        self.dim = dim
        self.layers = ModuleList([])
        self.rotary_emb = RotaryEmbedding(dim_head) if not t5_rel_pos_bias else None
        self.rel_pos_bias = T5RelativePositionBias(dim_head ** 0.5, heads=heads) if t5_rel_pos_bias else None
        for _ in range(depth):
            self.layers.append(ConformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult, conv_expansion_factor=conv_expansion_factor, conv_kernel_size=conv_kernel_size, attn_dropout=attn_dropout, ff_dropout=ff_dropout, conv_dropout=conv_dropout, conv_causal=conv_causal, attn_flash=attn_flash, use_gateloop_layers=use_gateloop_layers))

    def forward(self, x, mask=None):
        seq_len = x.shape[-2]
        rotary_emb = self.rotary_emb(seq_len) if exists(self.rotary_emb) else None
        attn_bias = self.rel_pos_bias(seq_len) if exists(self.rel_pos_bias) else None
        attn_value_residual = None
        for block in self.layers:
            x, attn_values = block(x, mask=mask, rotary_emb=rotary_emb, attn_bias=attn_bias, attn_value_residual=attn_value_residual, return_values=True)
            attn_value_residual = default(attn_value_residual, attn_values)
        return x


def divisible_by(numer, denom):
    return numer % denom == 0


class LogitHead(Module):

    def __init__(self, net: 'ConformerWrapper', logit_dim):
        super().__init__()
        self.net = net
        dim = net.dim
        self.to_logits = nn.Linear(dim, logit_dim)

    def forward(self, x):
        embed = self.net(x, return_embeddings=True)
        return self.to_logits(embed)


LossBreakdown = namedtuple('LossBreakdown', ['generator_loss', 'critic_loss'])


def sample_prob(prob):
    return random() < prob


def coin_flip():
    return sample_prob(0.5)


def cosine_schedule(t):
    """ https://arxiv.org/abs/2202.04200 """
    return torch.cos(t * math.pi / 2)


def eval_decorator(fn):

    @wraps(fn)
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


def log(t, eps=1e-10):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return (t / max(temperature, 1e-10) + gumbel_noise(t)).argmax(dim=dim)


def linear_schedule(t):
    return 1 - t


def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)
    return probs


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.0)
        log[key] = old_value + new_value
    return log


def checkpoint_num_steps(checkpoint_path):
    """Returns the number of steps trained from a checkpoint based on the filename.

    Filename format assumed to be something like "/path/to/soundstorm.20000.pt" which is
    for 20k train steps. Returns 20000 in that case.
    """
    results = re.findall('\\d+', str(checkpoint_path))
    if len(results) == 0:
        return 0
    return int(results[-1])


def cycle(dl):
    while True:
        for data in dl:
            yield data


def noop(*args, **kwargs):
    pass


def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Attend,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ChanLayerNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FeedForward,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GLU,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (PreNorm,
     lambda: ([], {'dim': 4, 'fn': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Scale,
     lambda: ([], {'scale': 1.0, 'fn': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

