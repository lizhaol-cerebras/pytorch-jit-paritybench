
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


from torch import nn


from torch import einsum


import torch.nn.functional as F


from collections import namedtuple


from functools import wraps


import math


from functools import partial


from torch import Tensor


from torch.autograd import grad as torch_grad


from torch.nn.utils.rnn import pad_sequence


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from functools import reduce


import warnings


import logging


from torch.optim import AdamW


from torch.optim import Adam


import functools


from itertools import cycle


from itertools import zip_longest


from torch.nn import Module


from torch.nn import ModuleList


from torch.linalg import vector_norm


import re


import copy


from math import sqrt


from random import choice


from collections import Counter


from torch.optim import Optimizer


from torch.optim.lr_scheduler import LambdaLR


from torch.optim.lr_scheduler import _LRScheduler


from torch.utils.data import random_split


Config = namedtuple('Config', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


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

    def __init__(self, dropout=0.0, causal=False, flash=False):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.causal = causal
        self.register_buffer('mask', None, persistent=False)
        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'
        self.cpu_config = Config(True, True, True)
        self.cuda_config = None
        if not torch.cuda.is_available() or not flash:
            return
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = Config(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = Config(False, True, True)

    def flash_attn(self, q, k, v, mask=None):
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda
        k = repeat(k, 'b ... -> b h ...', h=heads)
        v = repeat(v, 'b ... -> b h ...', h=heads)
        causal = self.causal
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask = mask.expand(-1, heads, q_len, -1)
            if causal:
                causal_mask = torch.ones((q_len, k_len), device=q.device, dtype=torch.bool).triu(k_len - q_len + 1)
                mask = mask & ~causal_mask
                causal = False
        config = self.cuda_config if is_cuda else self.cpu_config
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
        n, device = q.shape[-2], q.device
        scale = q.shape[-1] ** -0.5
        if self.flash:
            assert not exists(attn_bias), 'attention bias not supported for flash attention'
            return self.flash_attn(q, k, v, mask=mask)
        sim = einsum('b h i d, b j d -> b h i j', q, k) * scale
        if exists(attn_bias):
            sim = sim + attn_bias
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), device=sim.device, dtype=torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = einsum('b h i j, b j d -> b h i d', attn, v)
        return out


class LayerNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class RelativePositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(self, *, dim, heads, layers=3):
        super().__init__()
        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(1, dim), nn.SiLU()))
        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), nn.SiLU()))
        self.net.append(nn.Linear(dim, heads))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        assert j >= i
        device = self.device
        i_pos = torch.arange(i, device=device) + (j - i)
        j_pos = torch.arange(j, device=device)
        rel_pos = rearrange(i_pos, 'i -> i 1') - rearrange(j_pos, 'j -> 1 j')
        rel_pos += j - 1
        x = torch.arange(-j + 1, j, device=device).float()
        x = rearrange(x, '... -> ... 1')
        for layer in self.net:
            x = layer(x)
        x = x[rel_pos]
        return rearrange(x, 'i j h -> h i j')


class GEGLU(nn.Module):

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def default(val, d):
    return val if exists(val) else d


class Attention(nn.Module):

    def __init__(self, dim, causal=False, dim_head=64, dim_context=None, heads=8, norm_context=False, num_null_kv=0, dropout=0.1, scale=8, flash=False):
        super().__init__()
        self.heads = heads
        self.causal = causal
        inner_dim = dim_head * heads
        dim_context = default(dim_context, dim)
        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()
        self.attn_dropout = nn.Dropout(dropout)
        self.num_null_kv = num_null_kv
        self.null_kv = nn.Parameter(torch.randn(2, num_null_kv, dim_head)) if num_null_kv > 0 else None
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_head * 2, bias=False)
        self.attend = Attend(flash=flash, dropout=dropout, causal=causal)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias=False), nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None, attn_bias=None, prefix_context=None, prefix_context_mask=None, return_kv_cache=False, return_values=False, value_residual: 'Tensor | None'=None, kv_cache=None):
        b, n, _, device = *x.shape, x.device
        if exists(context):
            context = self.context_norm(context)
        kv_input = default(context, x)
        if exists(prefix_context):
            kv_input = torch.cat((prefix_context, kv_input), dim=-2)
            prefix_seq_len = prefix_context.shape[-2]
            if not exists(mask):
                mask = torch.ones((b, n), device=device, dtype=torch.bool)
            if exists(prefix_context_mask):
                mask = torch.cat((prefix_context_mask, mask), dim=-1)
            else:
                mask = F.pad(mask, (prefix_seq_len, 0), value=True)
            if exists(attn_bias):
                attn_bias = F.pad(attn_bias, (prefix_seq_len, 0), value=0.0)
        x = self.norm(x)
        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)
        orig_v = v
        if exists(value_residual):
            v = 0.5 * (v + value_residual)
        if exists(kv_cache):
            ck, cv = kv_cache
            k = torch.cat((ck, k), dim=-2)
            v = torch.cat((cv, v), dim=-2)
        if return_kv_cache:
            kv_cache = torch.stack((k, v))
        if self.num_null_kv > 0:
            null_k, null_v = repeat(self.null_kv, 'kv n d -> kv b n d', b=b).unbind(dim=0)
            k = torch.cat((null_k, k), dim=-2)
            v = torch.cat((null_v, v), dim=-2)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        if exists(mask):
            mask = F.pad(mask, (self.num_null_kv, 0), value=True)
        out = self.attend(q, k, v, attn_bias=attn_bias, mask=mask)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        if not return_kv_cache and not return_values:
            return out
        if return_kv_cache and not return_values:
            return out, kv_cache
        if return_values and not return_kv_cache:
            return out, orig_v
        return out, (kv_cache, orig_v)


def FeedForward(dim, mult=4, dropout=0.1):
    inner_dim = int(dim * 2 * mult / 3)
    return nn.Sequential(LayerNorm(dim), nn.Linear(dim, inner_dim * 2, bias=False), GEGLU(), LayerNorm(inner_dim), nn.Dropout(dropout), nn.Linear(inner_dim, dim, bias=False))


def grad_shrink(t, alpha=0.1):
    return t * alpha + t.detach() * (1 - alpha)


def always(val):

    def inner(*args, **kwargs):
        return val
    return inner


def maybe(fn):
    if not exists(fn):
        return always(None)

    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner


class Transformer(nn.Module):

    def __init__(self, *, dim, depth, heads, dim_context=None, cross_attend=False, attn_dropout=0.0, ff_dropout=0.0, grad_shrink_alpha=0.1, cond_as_self_attn_prefix=False, rel_pos_bias=True, flash_attn=False, add_value_residual=True, **kwargs):
        super().__init__()
        rel_pos_bias = rel_pos_bias and not flash_attn
        assert not (cross_attend and cond_as_self_attn_prefix)
        self.dim_context = default(dim_context, dim)
        self.cond_as_self_attn_prefix = cond_as_self_attn_prefix
        self.grad_shrink = partial(grad_shrink, alpha=grad_shrink_alpha)
        self.layers = nn.ModuleList([])
        self.rel_pos_bias = RelativePositionBias(dim=dim // 2, heads=heads) if rel_pos_bias else None
        for _ in range(depth):
            self.layers.append(nn.ModuleList([Attention(dim=dim, heads=heads, dropout=attn_dropout, flash=flash_attn, causal=True, **kwargs), Attention(dim=dim, heads=heads, dropout=attn_dropout, dim_context=dim_context, flash=flash_attn, num_null_kv=1, norm_context=True, **kwargs) if cross_attend else None, FeedForward(dim=dim, dropout=ff_dropout)]))
        self.norm = LayerNorm(dim)
        self.add_value_residual = add_value_residual

    def forward(self, x, self_attn_mask=None, context=None, context_mask=None, attn_bias=None, return_kv_cache=False, kv_cache=None):
        assert not (self.cond_as_self_attn_prefix and not exists(context))
        assert not (exists(context) and context.shape[-1] != self.dim_context), f'you had specified a conditioning dimension of {self.dim_context}, yet what was received by the transformer has dimension of {context.shape[-1]}'
        n, device = x.shape[1], x.device
        x = self.grad_shrink(x)
        if self.cond_as_self_attn_prefix:
            kv_cache = None
        new_kv_cache = []
        if exists(kv_cache):
            cache_len = kv_cache.shape[-2]
            kv_cache = iter(kv_cache)
        else:
            cache_len = 0
            kv_cache = iter([])
        x = x[:, cache_len:]
        if exists(attn_bias):
            rel_pos_bias = attn_bias
        else:
            rel_pos_bias = maybe(self.rel_pos_bias)(n, n)
        if exists(rel_pos_bias):
            rel_pos_bias = rel_pos_bias[..., cache_len:, :]
        self_attn_kwargs = dict()
        if self.cond_as_self_attn_prefix:
            self_attn_kwargs = dict(prefix_context=context, prefix_context_mask=context_mask)
        self_attn_value_residual = None
        cross_attn_value_residual = None
        for attn, cross_attn, ff in self.layers:
            residual = x
            x, (layer_kv_cache, values) = attn(x, attn_bias=rel_pos_bias, mask=self_attn_mask, kv_cache=next(kv_cache, None), return_kv_cache=True, return_values=True, value_residual=self_attn_value_residual, **self_attn_kwargs)
            if self.add_value_residual:
                self_attn_value_residual = default(self_attn_value_residual, values)
            new_kv_cache.append(layer_kv_cache)
            x = x + residual
            if exists(cross_attn):
                assert exists(context)
                cross_attend_out, values = cross_attn(x, context=context, mask=context_mask, return_values=True, value_residual=cross_attn_value_residual)
                x = cross_attend_out + x
                if self.add_value_residual:
                    cross_attn_value_residual = default(cross_attn_value_residual, values)
            x = ff(x) + x
        x = self.norm(x)
        if not return_kv_cache:
            return x
        return x, torch.stack(new_kv_cache)


DEFAULT_T5_NAME = 'google/t5-v1_1-base'


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
        raise ValueError(f'unknown t5 name {name}')
    return config.d_model


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


MAX_LENGTH = 256


def get_model(name):
    model = T5EncoderModel.from_pretrained(name)
    return model


def get_tokenizer(name):
    tokenizer = T5Tokenizer.from_pretrained(name)
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


def ceil_div(numer, denom):
    return (numer + denom - 1) // denom


def round_down_nearest_multiple(num, divisor):
    return num // divisor * divisor


def remainder_needed_until_multiple(n, mult):
    return ceil_div(n, mult) * mult - n


class FineTransformer(nn.Module):

    def __init__(self, *, num_coarse_quantizers, num_fine_quantizers, codebook_size, dim, depth, heads=8, attn_dropout=0.0, ff_dropout=0.0, t5_name=DEFAULT_T5_NAME, has_condition=False, cond_dim=None, audio_text_condition=False, cond_as_self_attn_prefix=False, cond_drop_prob=0.5, grad_shrink_alpha=0.1, project_coarse_logits=True, pad_id=-1, rel_pos_bias=True, flash_attn=False, **kwargs):
        super().__init__()
        rel_pos_bias = rel_pos_bias and not flash_attn
        if audio_text_condition:
            has_condition = True
            cond_dim = default(cond_dim, dim)
        self.has_condition = has_condition
        self.embed_text = partial(t5_encode_text, name=t5_name)
        self.cond_drop_prob = cond_drop_prob
        self.num_coarse_quantizers = num_coarse_quantizers
        self.coarse_start_token = nn.Parameter(torch.randn(dim))
        self.fine_start_token = nn.Parameter(torch.randn(dim))
        self.coarse_embedding = nn.Embedding(num_coarse_quantizers * codebook_size, dim)
        self.fine_embedding = nn.Embedding(num_fine_quantizers * codebook_size, dim)
        self.coarse_quantize_embedding = nn.Embedding(num_coarse_quantizers, dim)
        self.fine_quantize_embedding = nn.Embedding(num_fine_quantizers, dim)
        self.pad_id = pad_id
        self.eos_id = codebook_size
        text_dim = default(cond_dim, get_encoded_dim(t5_name))
        self.proj_text_embed = nn.Linear(text_dim, dim, bias=False) if text_dim != dim else nn.Identity()
        self.transformer = Transformer(dim=dim, depth=depth, heads=heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout, cross_attend=has_condition and not cond_as_self_attn_prefix, cond_as_self_attn_prefix=cond_as_self_attn_prefix, rel_pos_bias=False, grad_shrink_alpha=grad_shrink_alpha, flash_attn=flash_attn, **kwargs)
        self.null_pos_bias = nn.Parameter(torch.randn(heads, 1, 1)) if rel_pos_bias else None
        pos_bias_mlp_dim = dim // 2
        self.pos_bias_mlp = nn.Sequential(nn.Linear(2, pos_bias_mlp_dim), nn.SiLU(), nn.Linear(pos_bias_mlp_dim, pos_bias_mlp_dim), nn.SiLU(), nn.Linear(pos_bias_mlp_dim, heads)) if rel_pos_bias else None
        self.codebook_size = codebook_size
        self.num_coarse_quantizers = num_coarse_quantizers
        self.num_fine_quantizers = num_fine_quantizers
        self.coarse_logit_weights = nn.Parameter(torch.randn(num_coarse_quantizers, codebook_size, dim)) if project_coarse_logits else None
        self.fine_logit_weights = nn.Parameter(torch.randn(num_fine_quantizers, codebook_size, dim))

    @property
    def device(self):
        return next(self.parameters()).device

    def load(self, path):
        device = self.device
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location=device)
        if 'version' in pkg and version.parse(pkg['version']) < version.parse(__version__):
            None
        self.load_state_dict(pkg['model'])
        return pkg

    def forward_with_cond_scale(self, *args, cond_scale=3, return_kv_cache=False, kv_cache=None, embed_cache=None, **kwargs):
        iter_kv_cache = iter(default(kv_cache, []))
        iter_embed_cache = iter(default(embed_cache, []))
        new_kv_caches = []
        new_embed_caches = []
        (semantic_logits, coarse_logits), (new_kv_cache, new_embed_cache) = self.forward(*args, cond_drop_prob=0.0, return_cache=True, kv_cache=next(iter_kv_cache, None), embed_cache=next(iter_embed_cache, None), **kwargs)
        new_kv_caches.append(new_kv_cache)
        new_embed_caches.append(new_embed_cache)
        if cond_scale == 1 or not self.has_condition:
            if not return_kv_cache:
                return semantic_logits, coarse_logits
            return (semantic_logits, coarse_logits), (torch.stack(new_kv_caches), torch.stack(new_embed_caches))
        (null_semantic_logits, null_coarse_logits), (null_new_kv_cache, null_new_embed_cache) = self.forward(*args, cond_drop_prob=1.0, return_cache=True, kv_cache=next(iter_kv_cache, None), embed_cache=next(iter_embed_cache, None), **kwargs)
        new_kv_caches.append(null_new_kv_cache)
        new_embed_caches.append(null_new_embed_cache)
        scaled_semantic_logits = None
        if exists(null_semantic_logits):
            scaled_semantic_logits = null_semantic_logits + (semantic_logits - null_semantic_logits) * cond_scale
        scaled_coarse_logits = null_coarse_logits + (coarse_logits - null_coarse_logits) * cond_scale
        if not return_kv_cache:
            return scaled_semantic_logits, scaled_coarse_logits
        return (scaled_semantic_logits, scaled_coarse_logits), (torch.stack(new_kv_caches), torch.stack(new_embed_caches))

    def forward(self, coarse_token_ids, fine_token_ids, text: 'list[str] | None'=None, text_embeds=None, cond_drop_prob=None, self_attn_mask=None, kv_cache=None, embed_cache=None, return_cache=False, return_only_fine_logits=False):
        b, device = coarse_token_ids.shape[0], coarse_token_ids.device
        has_text = exists(text) or exists(text_embeds)
        assert not self.has_condition ^ has_text
        text_mask = None
        if not exists(text_embeds) and exists(text):
            with torch.inference_mode():
                text_embeds = self.embed_text(text, output_device=device)
                text_mask = torch.any(text_embeds != 0, dim=-1)
        if exists(text_embeds):
            text_embeds = self.proj_text_embed(text_embeds)
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        if exists(text_mask) and cond_drop_prob > 0:
            keep_mask = prob_mask_like((b,), 1 - cond_drop_prob, device=device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask
        coarse_token_ids, fine_token_ids = map(lambda t: rearrange(t, 'b ... -> b (...)'), (coarse_token_ids, fine_token_ids))
        coarse_self_attn_mask = (coarse_token_ids != self.pad_id) & (coarse_token_ids != self.eos_id)
        coarse_token_ids = coarse_token_ids.masked_fill(~coarse_self_attn_mask, 0)
        fine_token_seq_len = fine_token_ids.shape[-1]
        coarse_self_attn_mask = F.pad(coarse_self_attn_mask, (1, fine_token_seq_len + 1), value=True)
        if exists(self_attn_mask):
            self_attn_mask &= coarse_self_attn_mask
        else:
            self_attn_mask = coarse_self_attn_mask
        b, n = coarse_token_ids.shape
        coarse_length = coarse_token_ids.shape[-1]
        coarse_offsets = torch.arange(self.num_coarse_quantizers, device=device)
        coarse_seq_length = ceil_div(coarse_token_ids.shape[-1], self.num_coarse_quantizers)
        coarse_offsets = repeat(coarse_offsets, 'q -> (n q)', n=coarse_seq_length)
        coarse_offsets = coarse_offsets[:coarse_length]
        coarse_token_ids = coarse_token_ids + rearrange(coarse_offsets, '... -> 1 ...') * self.codebook_size
        fine_length = fine_token_ids.shape[-1]
        fine_offsets = torch.arange(self.num_fine_quantizers, device=device)
        fine_seq_length = ceil_div(fine_token_ids.shape[-1], self.num_fine_quantizers)
        fine_offsets = repeat(fine_offsets, 'q -> (n q)', n=fine_seq_length)
        fine_offsets = fine_offsets[:fine_length]
        fine_token_ids = fine_token_ids + rearrange(fine_offsets, '... -> 1 ...') * self.codebook_size
        coarse_tokens = self.coarse_embedding(coarse_token_ids)
        fine_tokens = self.fine_embedding(fine_token_ids)
        coarse_quantize_tokens = repeat(self.coarse_quantize_embedding.weight, 'q d -> (n q) d', n=ceil_div(coarse_token_ids.shape[-1], self.num_coarse_quantizers))
        coarse_quantize_tokens = coarse_quantize_tokens[:coarse_token_ids.shape[-1], ...]
        coarse_tokens = coarse_tokens + coarse_quantize_tokens
        fine_quantize_tokens = repeat(self.fine_quantize_embedding.weight, 'q d -> (n q) d', n=ceil_div(fine_token_ids.shape[-1], self.num_fine_quantizers))
        fine_quantize_tokens = fine_quantize_tokens[:fine_token_ids.shape[-1], ...]
        fine_tokens = fine_tokens + fine_quantize_tokens
        coarse_start_tokens = repeat(self.coarse_start_token, 'd -> b 1 d', b=b)
        fine_start_tokens = repeat(self.fine_start_token, 'd -> b 1 d', b=b)
        tokens = torch.cat((coarse_start_tokens, coarse_tokens, fine_start_tokens, fine_tokens), dim=1)
        attn_bias = None
        if exists(self.pos_bias_mlp):
            max_seq_len = max(coarse_seq_length, fine_seq_length)
            coarse_pos = torch.arange(coarse_seq_length, device=device)
            fine_pos = torch.arange(fine_seq_length, device=device)
            coarse_pos = repeat(coarse_pos, 'n -> (n q)', q=self.num_coarse_quantizers)[:coarse_length]
            fine_pos = repeat(fine_pos, 'n -> (n q)', q=self.num_fine_quantizers)[:fine_length]
            coarse_pos = F.pad(coarse_pos, (1, 0), value=-1)
            fine_pos = F.pad(fine_pos, (1, 0), value=-1)
            seq_positions = torch.cat((coarse_pos, fine_pos), dim=-1)
            coarse_offsets = F.pad(coarse_offsets, (1, 0), value=0)
            fine_offsets = fine_offsets + self.num_coarse_quantizers
            fine_offsets = F.pad(fine_offsets, (1, 0), value=0)
            seq_offsets = torch.cat((coarse_offsets, fine_offsets), dim=-1)
            pos_mlp_input = torch.stack((seq_positions.clamp(min=0), seq_offsets), dim=-1)
            num_offsets = self.num_fine_quantizers + self.num_coarse_quantizers
            rel_seq_len, rel_offsets = map(lambda n: 2 * n - 1, (max_seq_len, num_offsets))
            rel_dist = rearrange(pos_mlp_input, 'i c -> i 1 c') - rearrange(pos_mlp_input, 'j c -> 1 j c')
            rel_seq_len_range = repeat(torch.arange(rel_seq_len, device=device), 'n -> (n q)', q=rel_offsets)
            rel_offset_range = repeat(torch.arange(rel_offsets, device=device), 'q -> (n q)', n=rel_seq_len)
            mlp_inputs = torch.stack((rel_seq_len_range, rel_offset_range), dim=-1)
            attn_bias = self.pos_bias_mlp(mlp_inputs.float())
            rel_dist_seq_pos, rel_dist_seq_offset = rel_dist.unbind(dim=-1)
            rel_dist_seq_pos += max_seq_len - 1
            rel_dist_seq_offset += num_offsets - 1
            rel_dist_indices = rel_dist_seq_pos * rel_offsets + rel_dist_seq_offset
            attn_bias = attn_bias[rel_dist_indices]
            attn_bias = rearrange(attn_bias, '... h -> h ...')
            is_start_token_seq = seq_positions == -1
            start_token_mask = rearrange(is_start_token_seq, 'i -> i 1') | rearrange(is_start_token_seq, 'j -> 1 j')
            attn_bias = torch.where(start_token_mask, self.null_pos_bias, attn_bias)
        tokens, next_kv_cache = self.transformer(tokens, context=text_embeds, self_attn_mask=self_attn_mask, context_mask=text_mask, attn_bias=attn_bias, kv_cache=kv_cache, return_kv_cache=True)
        if exists(embed_cache):
            tokens = torch.cat((embed_cache, tokens), dim=-2)
        new_embed_cache = tokens
        pred_coarse_tokens, pred_fine_tokens = tokens[:, :n], tokens[:, n + 1:]
        pred_coarse_seq_len = pred_coarse_tokens.shape[1]
        padding = remainder_needed_until_multiple(pred_coarse_seq_len, self.num_coarse_quantizers)
        if padding != 0:
            pred_coarse_tokens = F.pad(pred_coarse_tokens, (0, 0, 0, padding), value=0.0)
        pred_coarse_tokens = rearrange(pred_coarse_tokens, 'b (n q) d -> b n q d', q=self.num_coarse_quantizers)
        coarse_logits = None
        if not return_only_fine_logits and exists(self.coarse_logit_weights):
            coarse_logits = einsum('q c d, b n q d -> b n q c', self.coarse_logit_weights, pred_coarse_tokens)
            coarse_logits = rearrange(coarse_logits, 'b n q c -> b (n q) c')
            coarse_logits = coarse_logits[:, :pred_coarse_seq_len]
        pred_fine_seq_len = pred_fine_tokens.shape[1]
        nq = round_down_nearest_multiple(pred_fine_seq_len, self.num_fine_quantizers)
        pred_fine_tokens_groupable, pred_fine_tokens_remainder = pred_fine_tokens[:, :nq], pred_fine_tokens[:, nq:]
        pred_fine_tokens_groupable = rearrange(pred_fine_tokens_groupable, 'b (n q) d -> b n q d', q=self.num_fine_quantizers)
        fine_logits_groupable = einsum('q c d, b n q d -> b n q c', self.fine_logit_weights, pred_fine_tokens_groupable)
        fine_logits_groupable = rearrange(fine_logits_groupable, 'b n q c -> b (n q) c')
        remainder_num_quantizers = pred_fine_tokens_remainder.shape[1]
        if remainder_num_quantizers > 0:
            fine_logits_remainder = einsum('q c d, b q d -> b q c', self.fine_logit_weights[:remainder_num_quantizers], pred_fine_tokens_remainder)
            fine_logits = torch.cat((fine_logits_groupable, fine_logits_remainder), dim=1)
        else:
            fine_logits = fine_logits_groupable
        logits = coarse_logits, fine_logits
        if not return_cache:
            return logits
        return logits, (next_kv_cache, new_embed_cache)


class AudioConditionerBase(nn.Module):
    pass


def curtail_to_multiple(t, mult, from_left=False):
    data_len = t.shape[-1]
    rounded_seq_len = round_down_nearest_multiple(data_len, mult)
    seq_slice = slice(None, rounded_seq_len) if not from_left else slice(-rounded_seq_len, None)
    return t[..., seq_slice]


class FairseqVQWav2Vec(nn.Module):
    """
    checkpoint path can be found at https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md#vq-wav2vec
    specifically download the kmeans model for now

    $ wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec_kmeans.pt
    """

    def __init__(self, checkpoint_path, target_sample_hz=24000, seq_len_multiple_of=None):
        super().__init__()
        self.target_sample_hz = target_sample_hz
        self.seq_len_multiple_of = seq_len_multiple_of
        path = Path(checkpoint_path)
        assert path.exists(), f'path {checkpoint_path} does not exist'
        checkpoint = torch.load(checkpoint_path)
        load_model_input = {checkpoint_path: checkpoint}
        model, *_ = fairseq.checkpoint_utils.load_model_ensemble_and_task(load_model_input)
        self.model = model[0]
        self.model.eval()
        assert hasattr(self.model, 'vector_quantizer') and hasattr(self.model.vector_quantizer, 'embedding'), 'the vq wav2vec model does not seem to be valid'

    @property
    def groups(self):
        return self.model.vector_quantizer.groups

    @property
    def downsample_factor(self):
        return 80

    @property
    def codebook_size(self):
        return self.model.vector_quantizer.embedding.shape[0]

    @torch.inference_mode()
    def forward(self, wav_input, flatten=True, input_sample_hz=None):
        if exists(input_sample_hz):
            wav_input = resample(wav_input, input_sample_hz, self.target_sample_hz)
        if exists(self.seq_len_multiple_of):
            wav_input = curtail_to_multiple(wav_input, self.seq_len_multiple_of)
        embed = self.model.feature_extractor(wav_input)
        _, codebook_indices = self.model.vector_quantizer.forward_idx(embed)
        if not flatten:
            return codebook_indices
        return rearrange(codebook_indices, 'b ... -> b (...)')


class HubertWithKmeans(nn.Module):
    """
    checkpoint and kmeans can be downloaded at https://github.com/facebookresearch/fairseq/tree/main/examples/hubert
    or you can train your own
    """

    def __init__(self, checkpoint_path, kmeans_path, target_sample_hz=16000, seq_len_multiple_of=None, output_layer=9):
        super().__init__()
        self.target_sample_hz = target_sample_hz
        self.seq_len_multiple_of = seq_len_multiple_of
        self.output_layer = output_layer
        model_path = Path(checkpoint_path)
        kmeans_path = Path(kmeans_path)
        assert model_path.exists(), f'path {checkpoint_path} does not exist'
        assert kmeans_path.exists(), f'path {kmeans_path} does not exist'
        checkpoint = torch.load(checkpoint_path)
        load_model_input = {checkpoint_path: checkpoint}
        model, *_ = fairseq.checkpoint_utils.load_model_ensemble_and_task(load_model_input)
        self.model = model[0]
        self.model.eval()
        kmeans = joblib.load(kmeans_path)
        self.kmeans = kmeans
        self.register_buffer('cluster_centers', torch.from_numpy(kmeans.cluster_centers_))

    @property
    def groups(self):
        return 1

    @property
    def codebook_size(self):
        return self.kmeans.n_clusters

    @property
    def downsample_factor(self):
        return 320

    @torch.inference_mode()
    def forward(self, wav_input, flatten=True, input_sample_hz=None):
        batch, device = wav_input.shape[0], wav_input.device
        if exists(input_sample_hz):
            wav_input = resample(wav_input, input_sample_hz, self.target_sample_hz)
        if exists(self.seq_len_multiple_of):
            wav_input = curtail_to_multiple(wav_input, self.seq_len_multiple_of)
        embed = self.model(wav_input, features_only=True, mask=False, output_layer=self.output_layer)['x']
        batched_cluster_centers = repeat(self.cluster_centers, 'c d -> b c d', b=embed.shape[0])
        dists = -torch.cdist(embed, batched_cluster_centers, p=2)
        clusters = dists.argmax(dim=-1)
        if flatten:
            return clusters
        return rearrange(clusters, 'b ... -> b (...)')


def all_rows_have_eos_id(t, eos_id):
    eos_mask = t == eos_id
    return torch.any(eos_mask, dim=-1).all()


def append_eos_id(ids, eos_id):
    b, device = ids.shape[0], ids.device
    eos_ids = torch.ones(1, device=device).long() * eos_id
    eos_ids = repeat(eos_ids, '1 -> b 1', b=b)
    ids = torch.cat((ids, eos_ids), dim=-1)
    return ids


def batch_unique_consecutive(t, pad_value=0.0):
    unique_arr = [torch.unique_consecutive(el) for el in t.unbind(dim=0)]
    return pad_sequence(unique_arr, batch_first=True, padding_value=pad_value)


def eval_decorator(fn):

    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


def generate_mask_with_prob(shape, mask_prob, device):
    seq = shape[-1]
    rand = torch.randn(shape, device=device)
    rand[:, 0] = -torch.finfo(rand.dtype).max
    num_mask = min(int(seq * mask_prob), seq - 1)
    indices = rand.topk(num_mask, dim=-1).indices
    mask = ~torch.zeros(shape, device=device).scatter(1, indices, 1.0).bool()
    return mask


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return (t / temperature + gumbel_noise(t)).argmax(dim=dim)


def mask_out_after_eos_id(t, eos_id, mask_value=-1, keep_eos=True):
    eos_mask = (t == eos_id).float()
    if keep_eos:
        eos_mask = F.pad(eos_mask, (1, -1))
    after_eos_mask = eos_mask.cumsum(dim=-1) > 0
    return t.masked_fill(after_eos_mask, mask_value)


def safe_cat(*tensors, dim=-2):
    args = [*filter(exists, tensors)]
    if len(args) == 0:
        return None
    elif len(args) == 1:
        return args[0]
    else:
        return torch.cat(args, dim=dim)


def top_k(logits, thres=0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


def get_num_quantizers(model: 'EncodecModel', audio_length=512):
    out = model.encode(torch.randn(1, 1, audio_length))
    return out[0][0].shape[1]


class EncodecWrapper(nn.Module):
    """
    Support pretrained 24kHz Encodec by Meta AI, if you want to skip training SoundStream.

    TODO:
    - see if we need to keep the scaled version and somehow persist the scale factors for when we need to decode? Right
        now I'm just setting self.model.normalize = False to sidestep all of that
    - see if we can use the 48kHz model, which is specifically for music. Right now we're using the 24kHz model because
        that's what was used in MusicLM and avoids any resampling issues.
    -

    """

    def __init__(self, target_sample_hz=24000, strides=(2, 4, 5, 8), num_quantizers=8, bandwidth=6.0):
        super().__init__()
        self.model = EncodecModel.encodec_model_24khz()
        self.model.normalize = False
        self.model.set_target_bandwidth(bandwidth)
        num_quantizers = get_num_quantizers(self.model)
        self.target_sample_hz = target_sample_hz
        assert self.target_sample_hz == 24000, "haven't done anything with non-24kHz yet"
        self.codebook_dim = 128
        self.rq_groups = 1
        self.num_quantizers = num_quantizers
        self.strides = strides
        self.rq = ResidualVQ(dim=128, codebook_size=1024, num_quantizers=num_quantizers)
        for encodec_rq_layer, rq_layer in zip(self.model.quantizer.vq.layers, self.rq.layers):
            encodec_codebook = dict(encodec_rq_layer._codebook.named_buffers()).get('embed')
            vq_codebook = dict(rq_layer._codebook.named_buffers()).get('embed')
            encodec_codebook = rearrange(encodec_codebook, '... -> 1 ...')
            vq_codebook.copy_(encodec_codebook)

    @property
    def seq_len_multiple_of(self):
        return reduce(lambda x, y: x * y, self.strides)

    @property
    def downsample_factor(self):
        return self.seq_len_multiple_of

    def forward(self, x, input_sample_hz=None, return_encoded=False, **kwargs):
        x, ps = pack([x], '* n')
        if exists(input_sample_hz):
            x = resample(x, input_sample_hz, self.target_sample_hz)
        assert not self.model.training, 'Encodec is pretrained and should never be called outside eval mode.'
        wav = rearrange(x, f'b t -> b {self.model.channels} t')
        with torch.inference_mode():
            encoded_frames = self.model.encode(wav)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        codes = rearrange(codes, 'b q n -> b n q')
        emb = None
        if return_encoded:
            emb = self.get_emb_from_indices(codes)
            emb, = unpack(emb, ps, '* n c')
        codes, = unpack(codes, ps, '* n q')
        return emb, codes, None

    def decode_from_codebook_indices(self, quantized_indices):
        assert self.model.sample_rate == 24000, "if changing to 48kHz, that model segments its audio into lengths of 1.0 second with 1% overlap, whereas the 24kHz doesn't segment at all. this means the frame decode logic might change; this is a reminder to double check that."
        frames = self._decode_frame(quantized_indices)
        result = _linear_overlap_add(frames, self.model.segment_stride or 1)
        return rearrange(result, 'b n -> b 1 n')

    def get_emb_from_indices(self, indices):
        codes = rearrange(indices, 'b t q -> q b t')
        emb = self.model.quantizer.decode(codes)
        return rearrange(emb, 'b c n -> b n c')

    def decode(self, emb):
        emb = rearrange(emb, 'b n c -> b c n')
        return self.model.decoder(emb)

    def _decode_frame(self, quantized_indices):
        codes = rearrange(quantized_indices, 'b t q -> q b t')
        emb = self.model.quantizer.decode(codes)
        return self.model.decoder(emb)


class CausalConv1d(Module):

    def __init__(self, chan_in, chan_out, kernel_size, pad_mode='reflect', **kwargs):
        super().__init__()
        kernel_size = kernel_size
        dilation = kwargs.get('dilation', 1)
        stride = kwargs.get('stride', 1)
        self.pad_mode = pad_mode
        self.causal_padding = dilation * (kernel_size - 1) + (1 - stride)
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, **kwargs)

    def forward(self, x):
        x = F.pad(x, (self.causal_padding, 0), mode=self.pad_mode)
        return self.conv(x)


class ChannelTranspose(Module):

    def __init__(self, fn: 'Module'):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        x = rearrange(x, 'b c n -> b n c')
        out = self.fn(x, **kwargs) + x
        return rearrange(out, 'b n c -> b c n')


class ComplexConv2d(Module):

    def __init__(self, dim, dim_out, kernel_size, stride=1, padding=0):
        super().__init__()
        conv = nn.Conv2d(dim, dim_out, kernel_size, dtype=torch.complex64)
        self.weight = nn.Parameter(torch.view_as_real(conv.weight))
        self.bias = nn.Parameter(torch.view_as_real(conv.bias))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        weight, bias = map(torch.view_as_complex, (self.weight, self.bias))
        x = x
        return F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding)


class ModReLU(Module):
    """
    https://arxiv.org/abs/1705.09792
    https://github.com/pytorch/pytorch/issues/47052#issuecomment-718948801
    """

    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return F.relu(torch.abs(x) + self.b) * torch.exp(1.0j * torch.angle(x))


class Residual(Module):

    def __init__(self, fn: 'Module'):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))


def ComplexSTFTResidualUnit(chan_in, chan_out, strides):
    kernel_sizes = tuple(map(lambda t: t + 2, strides))
    paddings = tuple(map(lambda t: t // 2, kernel_sizes))
    return nn.Sequential(Residual(Sequential(ComplexConv2d(chan_in, chan_in, 3, padding=1), ModReLU(), ComplexConv2d(chan_in, chan_in, 3, padding=1))), ComplexConv2d(chan_in, chan_out, kernel_sizes, stride=strides, padding=paddings))


class ComplexSTFTDiscriminator(Module):

    def __init__(self, *, channels=32, strides=((1, 2), (2, 2), (1, 2), (2, 2), (1, 2), (2, 2)), chan_mults=(1, 2, 4, 4, 8, 8), input_channels=1, n_fft=1024, hop_length=256, win_length=1024, stft_normalized=False, stft_window_fn=torch.hann_window, logits_abs=True):
        super().__init__()
        self.init_conv = ComplexConv2d(input_channels, channels, 7, padding=3)
        layer_channels = tuple(map(lambda mult: mult * channels, chan_mults))
        layer_channels = channels, *layer_channels
        layer_channels_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))
        curr_channels = channels
        self.layers = ModuleList([])
        for layer_stride, (chan_in, chan_out) in zip(strides, layer_channels_pairs):
            self.layers.append(ComplexSTFTResidualUnit(chan_in, chan_out, layer_stride))
        self.final_conv = ComplexConv2d(layer_channels[-1], 1, (16, 1))
        self.stft_normalized = stft_normalized
        self.stft_window_fn = stft_window_fn
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.logits_abs = logits_abs

    def forward(self, x, return_intermediates=False):
        x = rearrange(x, 'b 1 n -> b n')
        """
        reference: The content of the paper( https://arxiv.org/pdf/2107.03312.pdf)is as follows:
        The STFT-based discriminator is illustrated in Figure 4
        and operates on a single scale, computing the STFT with a
        window length of W = 1024 samples and a hop length of
        H = 256 samples
        """
        stft_window = self.stft_window_fn(self.win_length, device=x.device)
        x = torch.stft(x, self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=stft_window, normalized=self.stft_normalized, return_complex=True)
        x = rearrange(x, 'b ... -> b 1 ...')
        intermediates = []
        x = self.init_conv(x)
        intermediates.append(x)
        for layer in self.layers:
            x = layer(x)
            intermediates.append(x)
        complex_logits = self.final_conv(x)
        if self.logits_abs:
            complex_logits = complex_logits.abs()
        else:
            complex_logits = torch.view_as_real(complex_logits)
        if not return_intermediates:
            return complex_logits
        return complex_logits, intermediates


class CausalConvTranspose1d(Module):

    def __init__(self, chan_in, chan_out, kernel_size, stride, **kwargs):
        super().__init__()
        self.upsample_factor = stride
        self.padding = kernel_size - 1
        self.conv = nn.ConvTranspose1d(chan_in, chan_out, kernel_size, stride, **kwargs)

    def forward(self, x):
        n = x.shape[-1]
        out = self.conv(x)
        out = out[..., :n * self.upsample_factor]
        return out


class SqueezeExcite(Module):

    def __init__(self, dim, reduction_factor=4, dim_minimum=8):
        super().__init__()
        dim_inner = max(dim_minimum, dim // reduction_factor)
        self.net = nn.Sequential(nn.Conv1d(dim, dim_inner, 1), nn.SiLU(), nn.Conv1d(dim_inner, dim, 1), nn.Sigmoid())

    def forward(self, x):
        seq, device = x.shape[-2], x.device
        cum_sum = x.cumsum(dim=-2)
        denom = torch.arange(1, seq + 1, device=device).float()
        cum_mean = cum_sum / rearrange(denom, 'n -> n 1')
        gate = self.net(cum_mean)
        return x * gate


def ResidualUnit(chan_in, chan_out, dilation, kernel_size=7, squeeze_excite=False, pad_mode='reflect'):
    return Residual(Sequential(CausalConv1d(chan_in, chan_out, kernel_size, dilation=dilation, pad_mode=pad_mode), nn.ELU(), CausalConv1d(chan_out, chan_out, 1, pad_mode=pad_mode), nn.ELU(), SqueezeExcite(chan_out) if squeeze_excite else None))


def DecoderBlock(chan_in, chan_out, stride, cycle_dilations=(1, 3, 9), squeeze_excite=False, pad_mode='reflect'):
    even_stride = stride % 2 == 0
    padding = (stride + (0 if even_stride else 1)) // 2
    output_padding = 0 if even_stride else 1
    residual_unit = partial(ResidualUnit, squeeze_excite=squeeze_excite, pad_mode=pad_mode)
    it = cycle(cycle_dilations)
    return nn.Sequential(CausalConvTranspose1d(chan_in, chan_out, 2 * stride, stride=stride), residual_unit(chan_out, chan_out, next(it)), residual_unit(chan_out, chan_out, next(it)), residual_unit(chan_out, chan_out, next(it)))


def EncoderBlock(chan_in, chan_out, stride, cycle_dilations=(1, 3, 9), squeeze_excite=False, pad_mode='reflect'):
    it = cycle(cycle_dilations)
    residual_unit = partial(ResidualUnit, squeeze_excite=squeeze_excite, pad_mode=pad_mode)
    return nn.Sequential(residual_unit(chan_in, chan_in, next(it)), residual_unit(chan_in, chan_in, next(it)), residual_unit(chan_in, chan_in, next(it)), CausalConv1d(chan_in, chan_out, 2 * stride, stride=stride))


class FiLM(Module):

    def __init__(self, dim, dim_cond):
        super().__init__()
        self.to_cond = nn.Linear(dim_cond, dim * 2)

    def forward(self, x, cond):
        gamma, beta = self.to_cond(cond).chunk(2, dim=-1)
        return x * gamma + beta


class LocalTransformer(Module):

    def __init__(self, *, dim, depth, heads, window_size, dynamic_pos_bias=False, **kwargs):
        super().__init__()
        self.window_size = window_size
        self.layers = ModuleList([])
        self.pos_bias = None
        if dynamic_pos_bias:
            self.pos_bias = DynamicPositionBias(dim=dim // 2, heads=heads)
        for _ in range(depth):
            self.layers.append(ModuleList([LocalMHA(dim=dim, heads=heads, qk_rmsnorm=True, window_size=window_size, use_rotary_pos_emb=not dynamic_pos_bias, gate_values_per_head=True, use_xpos=True, **kwargs), FeedForward(dim=dim)]))

    def forward(self, x):
        w = self.window_size
        attn_bias = self.pos_bias(w, w * 2) if exists(self.pos_bias) else None
        for attn, ff in self.layers:
            x = attn(x, attn_bias=attn_bias) + x
            x = ff(x) + x
        return x


def leaky_relu(p=0.1):
    return nn.LeakyReLU(p)


class MultiScaleDiscriminator(Module):

    def __init__(self, channels=16, layers=4, groups=(4, 16, 64, 256), chan_max=1024, input_channels=1):
        super().__init__()
        self.init_conv = nn.Conv1d(input_channels, channels, 15, padding=7)
        self.conv_layers = ModuleList([])
        curr_channels = channels
        for _, group in zip(range(layers), groups):
            chan_out = min(curr_channels * 4, chan_max)
            self.conv_layers.append(nn.Sequential(nn.Conv1d(curr_channels, chan_out, 41, stride=4, padding=20, groups=group), leaky_relu()))
            curr_channels = chan_out
        self.final_conv = nn.Sequential(nn.Conv1d(curr_channels, curr_channels, 5, padding=2), leaky_relu(), nn.Conv1d(curr_channels, 1, 3, padding=1))

    def forward(self, x, return_intermediates=False):
        x = self.init_conv(x)
        intermediates = []
        for layer in self.conv_layers:
            x = layer(x)
            intermediates.append(x)
        out = self.final_conv(x)
        if not return_intermediates:
            return out
        return out, intermediates


def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)


def filter_by_keys(fn, d):
    return {k: v for k, v in d.items() if fn(k)}


def gradient_penalty(wave, output, weight=10):
    batch_size, device = wave.shape[0], wave.device
    gradients = torch_grad(outputs=output, inputs=wave, grad_outputs=torch.ones_like(output), create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((vector_norm(gradients, dim=1) - 1) ** 2).mean()


def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()


def hinge_gen_loss(fake):
    return -fake.mean()


def map_keys(fn, d):
    return {fn(k): v for k, v in d.items()}


ConstantLRScheduler = partial(LambdaLR, lr_lambda=lambda step: 1.0)


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.0)
        log[key] = old_value + new_value
    return log


def check_one_trainer():
    global ONE_TRAINER_INSTANTIATED
    assert not ONE_TRAINER_INSTANTIATED, 'only one Trainer can be instantiated at a time for training'
    ONE_TRAINER_INSTANTIATED = True


def checkpoint_num_steps(checkpoint_path):
    """Returns the number of steps trained from a checkpoint based on the filename.

    Filename format assumed to be something like "/path/to/semantic.transformer.20000.pt" which is
    for 20k train steps. Returns 20000 in that case.
    """
    results = re.findall('\\d+', str(checkpoint_path))
    if len(results) == 0:
        return 0
    return int(results[-1])


def find_first(cond, arr):
    for el in arr:
        if cond(el):
            return el
    return None


def collate_one_or_multiple_tensors(fn):

    @wraps(fn)
    def inner(data):
        is_one_data = not isinstance(data[0], tuple)
        if is_one_data:
            data = fn(data)
            return data,
        outputs = []
        for datum in zip(*data):
            if is_bearable(datum, Tuple[str, ...]):
                output = list(datum)
            else:
                output = fn(datum)
            outputs.append(output)
        return tuple(outputs)
    return inner


@collate_one_or_multiple_tensors
def curtail_to_shortest_collate(data):
    min_len = min(*[datum.shape[0] for datum in data])
    data = [datum[:min_len] for datum in data]
    return torch.stack(data)


@collate_one_or_multiple_tensors
def pad_to_longest_fn(data):
    return pad_sequence(data, batch_first=True)


def get_dataloader(ds, pad_to_longest=True, **kwargs):
    collate_fn = pad_to_longest_fn if pad_to_longest else curtail_to_shortest_collate
    return DataLoader(ds, collate_fn=collate_fn, **kwargs)


def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []
    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    return wd_params, no_wd_params


def get_optimizer(params, lr=0.0001, wd=0.01, betas=(0.9, 0.99), eps=1e-08, filter_by_requires_grad=False, group_wd_params=True, use_lion=False, **kwargs):
    has_wd = wd > 0
    if filter_by_requires_grad:
        params = list(filter(lambda t: t.requires_grad, params))
    if group_wd_params and has_wd:
        wd_params, no_wd_params = separate_weight_decayable_params(params)
        params = [{'params': wd_params}, {'params': no_wd_params, 'weight_decay': 0}]
    if not has_wd:
        return Adam(params, lr=lr, betas=betas, eps=eps)
    return AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)


def noop(*args, **kwargs):
    pass


def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')


def determine_types(data, config):
    output = []
    for el in data:
        for name, data_type in config.items():
            if is_bearable(el, data_type):
                output.append(name)
                break
        else:
            raise TypeError(f'unable to determine type of {data}')
    return tuple(output)


def dict_values_to_device(d: 'dict', device):
    out = {}
    for k, v in d.items():
        out[k] = v if torch.is_tensor(v) else v
    return out


def has_duplicates(tup):
    counts = dict(Counter(tup))
    return any(filter(lambda count: count > 1, counts.values()))


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (CausalConv1d,
     lambda: ([], {'chan_in': 4, 'chan_out': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (CausalConvTranspose1d,
     lambda: ([], {'chan_in': 4, 'chan_out': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4])], {})),
    (FiLM,
     lambda: ([], {'dim': 4, 'dim_cond': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (GEGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ModReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MultiScaleDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64])], {})),
    (Residual,
     lambda: ([], {'fn': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

