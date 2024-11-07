import sys
_module = sys.modules[__name__]
del sys
palm_rlhf_pytorch = _module
attention = _module
lora = _module
optimizer = _module
palm = _module
ppo = _module
reward = _module
utils = _module
setup = _module
train = _module

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


from torch import nn


from torch import einsum


import torch.nn.functional as F


from collections import namedtuple


from functools import wraps


from torch.optim import AdamW


from torch.optim import Adam


import math


import copy


from itertools import zip_longest


from functools import partial


from collections import deque


from random import randrange


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.nn.utils.rnn import pad_sequence


import random


import numpy as np


from torch.nn import functional as F


Config = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


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


class Attention(nn.Module):

    def __init__(self, dropout=0.0, causal=False, use_flash_attn=False):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.causal = causal
        self.register_buffer('mask', None, persistent=False)
        self.use_flash_attn = use_flash_attn
        assert not (use_flash_attn and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'
        self.cpu_config = Config(True, True, True)
        self.cuda_config = None
        if not torch.cuda.is_available() or not use_flash_attn:
            return
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = Config(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = Config(False, True, True)

    def get_mask(self, n, device):
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]
        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer('mask', mask, persistent=False)
        return mask

    def flash_attn(self, q, k, v, mask=None):
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda
        k = rearrange(k, 'b ... -> b 1 ...').expand_as(q)
        v = rearrange(v, 'b ... -> b 1 ...').expand_as(q)
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask = mask.expand(-1, heads, q_len, -1)
        config = self.cuda_config if is_cuda else self.cpu_config
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0, is_causal=self.causal)
        return out

    def forward(self, q, k, v, mask=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        n, device = q.shape[-2], q.device
        scale = q.shape[-1] ** -0.5
        if self.use_flash_attn:
            return self.flash_attn(q, k, v, mask=mask)
        sim = einsum('b h i d, b j d -> b h i j', q, k) * scale
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        if self.causal:
            causal_mask = self.get_mask(n, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = einsum('b h i j, b j d -> b h i d', attn, v)
        return out


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class LoRA(nn.Module):

    def __init__(self, dim, dim_out, r=8, alpha=None):
        super().__init__()
        alpha = default(alpha, r)
        self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(dim, r))
        self.B = nn.Parameter(torch.zeros(r, dim_out))

    @property
    def weight(self):
        return self.A @ self.B * self.scale

    def forward(self, x):
        return x @ self.weight


class LayerNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        y = self.fn(x, **kwargs)
        if not any([t.requires_grad for t in (x, y)]):
            return x.add_(y)
        return y + x


class RotaryEmbedding(nn.Module):

    def __init__(self, dim, scale_base=512, use_xpos=True):
        super().__init__()
        inv_freq = 1.0 / 10000 ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freq', inv_freq)
        self.use_xpos = use_xpos
        self.scale_base = scale_base
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        if not self.use_xpos:
            return freqs, torch.ones(1, device=device)
        power = (t - seq_len // 2) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim=-1)
        return freqs, scale


class SwiGLU(nn.Module):

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t, scale=1.0):
    return t * pos.cos() * scale + rotate_half(t) * pos.sin() * scale


def l2norm(t):
    return F.normalize(t, dim=-1)


class ParallelTransformerBlock(nn.Module):

    def __init__(self, dim, dim_head=64, causal=True, heads=8, qk_rmsnorm=False, qk_scale=8, ff_mult=4, attn_dropout=0.0, ff_dropout=0.0, use_xpos=True, xpos_scale_base=512, flash_attn=False):
        super().__init__()
        self.norm = LayerNorm(dim)
        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = attn_inner_dim, dim_head, dim_head, ff_inner_dim * 2
        self.qk_rmsnorm = qk_rmsnorm
        if qk_rmsnorm:
            self.q_scale = nn.Parameter(torch.ones(dim_head))
            self.k_scale = nn.Parameter(torch.ones(dim_head))
        self.attend = Attention(causal=causal, dropout=attn_dropout, use_flash_attn=flash_attn)
        self.heads = heads
        self.scale = dim_head ** -0.5 if not qk_rmsnorm else qk_scale
        self.causal = causal
        self.rotary_emb = RotaryEmbedding(dim_head, scale_base=xpos_scale_base, use_xpos=use_xpos and causal)
        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.flash_attn = flash_attn
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.flash_attn_dropout = attn_dropout
        self.ff_out = nn.Sequential(SwiGLU(), nn.Dropout(ff_dropout), nn.Linear(ff_inner_dim, dim, bias=False))
        self.register_buffer('pos_emb', None, persistent=False)
        self.register_buffer('pos_emb_scale', None, persistent=False)

    def get_rotary_embedding(self, n, device):
        if exists(self.pos_emb) and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n], self.pos_emb_scale[:n]
        pos_emb, scale = self.rotary_emb(n, device=device)
        self.register_buffer('pos_emb', pos_emb, persistent=False)
        self.register_buffer('pos_emb_scale', scale, persistent=False)
        return pos_emb, scale

    def forward(self, x, mask=None, finetune_modules=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        n, device, h = x.shape[1], x.device, self.heads
        x = self.norm(x)
        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)
        lora_q = lora_k = lora_v = lora_o = None
        if exists(finetune_modules):
            lora_q, lora_k, lora_v, lora_o = finetune_modules
            q = q + lora_q(x)
            k = k + lora_k(x)
            v = v + lora_v(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        if self.qk_rmsnorm:
            q, k = map(l2norm, (q, k))
            q = q * self.q_scale
            k = k * self.k_scale
        positions, scale = self.get_rotary_embedding(n, device)
        q = apply_rotary_pos_emb(positions, q, scale)
        k = apply_rotary_pos_emb(positions, k, scale ** -1)
        out = self.attend(q, k, v, mask=mask)
        out = rearrange(out, 'b h n d -> b n (h d)')
        attn_out = self.attn_out(out)
        ff_out = self.ff_out(ff)
        if exists(lora_o):
            attn_out = attn_out + lora_o(out)
        return attn_out + ff_out


def eval_decorator(fn):

    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return (t / max(temperature, 1e-10) + gumbel_noise(t)).argmax(dim=dim)


def identity(t, *args, **kwargs):
    return t


def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


PPOActionCriticReturn = namedtuple('PPOActionCriticReturn', ['actions', 'sequence', 'mask', 'prompt_mask', 'action_logits', 'values'])


def masked_mean(seq, mask=None, dim=1, keepdim=False):
    if not exists(mask):
        return seq.mean(dim=dim)
    if seq.ndim == 3:
        mask = rearrange(mask, 'b n -> b n 1')
    masked_seq = seq.masked_fill(~mask, 0.0)
    numer = masked_seq.sum(dim=dim, keepdim=keepdim)
    denom = mask.sum(dim=dim, keepdim=keepdim)
    masked_mean = numer / denom.clamp(min=0.001)
    masked_mean = masked_mean.masked_fill(denom == 0, 0.0)
    return masked_mean


def shift(t, value=0, shift=1, dim=-1):
    zeros = (0, 0) * (-dim - 1)
    return F.pad(t, (*zeros, shift, -shift), value=value)


Memory = namedtuple('Memory', ['sequence', 'prompt_mask', 'mask', 'action_prob', 'action_log_prob', 'reward', 'value'])


def clipped_value_loss(values, rewards, old_values, clip):
    value_clipped = old_values + (values - old_values).clamp(-clip, clip)
    value_loss_1 = (value_clipped.flatten() - rewards) ** 2
    value_loss_2 = (values.flatten() - rewards) ** 2
    return torch.mean(torch.max(value_loss_1, value_loss_2))


def create_dataloader(data, batch_size, shuffle=True, device=None, **kwargs):
    ds = ExperienceDataset(data, device=device)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **kwargs)


def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []
    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    return wd_params, no_wd_params


def get_optimizer(params, lr=0.0001, wd=0.01, betas=(0.9, 0.99), eps=1e-08, filter_by_requires_grad=False, group_wd_params=True, use_lion=True, **kwargs):
    if filter_by_requires_grad:
        params = list(filter(lambda t: t.requires_grad, params))
    if group_wd_params and wd > 0:
        wd_params, no_wd_params = separate_weight_decayable_params(params)
        params = [{'params': wd_params}, {'params': no_wd_params, 'weight_decay': 0}]
    if use_lion:
        return Lion(params, lr=lr, betas=betas, weight_decay=wd)
    if wd == 0:
        return Adam(params, lr=lr, betas=betas, eps=eps)
    return AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)


def log_prob(prob, indices):
    assert prob.shape[:2] == indices.shape, f'preceding shapes of prob {prob.shape[:2]} and indices {indices.shape} must match'
    return log(prob.gather(-1, indices[..., None])).squeeze(-1)


def masked_entropy(prob, dim=-1, mask=None):
    entropies = (prob * log(prob)).sum(dim=-1)
    return masked_mean(entropies, mask=mask).mean()


def masked_kl_div(prob1, prob2, mask=None, reduce_batch=False):
    """
    need to account for variable sequence lengths, therefore not using the built-in functional version
    """
    kl_divs = (prob1 * (log(prob1) - log(prob2))).sum(dim=-1)
    loss = masked_mean(kl_divs, mask)
    if reduce_batch:
        return loss.mean()
    return loss


def masked_normalize(t, eps=1e-05, mask=None, dim=None):
    dim = default(dim, tuple(range(t.ndim)))
    kwargs = dict(dim=dim, keepdim=True)
    mean = masked_mean(t, mask=mask, **kwargs)
    mean_centered = t - mean
    var = masked_mean(mean_centered ** 2, mask=mask, **kwargs)
    return mean_centered * var.clamp(min=eps).rsqrt()


def pad_sequence_fixed(sequences, *args, **kwargs):
    first_el = sequences[0]
    has_no_dimension = first_el.ndim == 0
    if has_no_dimension:
        sequences = tuple(map(lambda t: t[None], sequences))
    out = pad_sequence(sequences, *args, **kwargs)
    if has_no_dimension:
        out = rearrange(out, '... 1 -> ...')
    return out


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (LayerNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LoRA,
     lambda: ([], {'dim': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Residual,
     lambda: ([], {'fn': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SwiGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

