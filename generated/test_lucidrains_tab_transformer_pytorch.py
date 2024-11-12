
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


import torch.nn.functional as F


from torch import nn


from torch import einsum


class GEGLU(nn.Module):

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=16, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        dropped_attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out), attn


class FeedForward(nn.Module):

    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Dropout(dropout), nn.Linear(dim * mult, dim))

    def forward(self, x, **kwargs):
        return self.net(x)


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)), PreNorm(dim, FeedForward(dim, dropout=ff_dropout))]))

    def forward(self, x, return_attn=False):
        post_softmax_attns = []
        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)
            x = x + attn_out
            x = ff(x) + x
        if not return_attn:
            return x
        return x, torch.stack(post_softmax_attns)


class NumericalEmbedder(nn.Module):

    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases


class FTTransformer(nn.Module):

    def __init__(self, *, categories, num_continuous, dim, depth, heads, dim_head=16, dim_out=1, num_special_tokens=2, attn_dropout=0.0, ff_dropout=0.0):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens
        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)
            self.categorical_embeds = nn.Embedding(total_tokens, dim)
        self.num_continuous = num_continuous
        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head, attn_dropout=attn_dropout, ff_dropout=ff_dropout)
        self.to_logits = nn.Sequential(nn.LayerNorm(dim), nn.ReLU(), nn.Linear(dim, dim_out))

    def forward(self, x_categ, x_numer, return_attn=False):
        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
        xs = []
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset
            x_categ = self.categorical_embeds(x_categ)
            xs.append(x_categ)
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)
            xs.append(x_numer)
        x = torch.cat(xs, dim=1)
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x, attns = self.transformer(x, return_attn=True)
        x = x[:, 0]
        logits = self.to_logits(x)
        if not return_attn:
            return logits
        return logits, attns


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class MLP(nn.Module):

    def __init__(self, dims, act=None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= len(dims_pairs) - 1
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)
            if is_last:
                continue
            act = default(act, nn.ReLU())
            layers.append(act)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class TabTransformer(nn.Module):

    def __init__(self, *, categories, num_continuous, dim, depth, heads, dim_head=16, dim_out=1, mlp_hidden_mults=(4, 2), mlp_act=None, num_special_tokens=2, continuous_mean_std=None, attn_dropout=0.0, ff_dropout=0.0, use_shared_categ_embed=True, shared_categ_dim_divisor=8.0):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens
        shared_embed_dim = 0 if not use_shared_categ_embed else int(dim // shared_categ_dim_divisor)
        self.category_embed = nn.Embedding(total_tokens, dim - shared_embed_dim)
        self.use_shared_categ_embed = use_shared_categ_embed
        if use_shared_categ_embed:
            self.shared_category_embed = nn.Parameter(torch.zeros(self.num_categories, shared_embed_dim))
            nn.init.normal_(self.shared_category_embed, std=0.02)
        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)
        self.num_continuous = num_continuous
        if self.num_continuous > 0:
            if exists(continuous_mean_std):
                assert continuous_mean_std.shape == (num_continuous, 2), f'continuous_mean_std must have a shape of ({num_continuous}, 2) where the last dimension contains the mean and variance respectively'
            self.register_buffer('continuous_mean_std', continuous_mean_std)
            self.norm = nn.LayerNorm(num_continuous)
        self.transformer = Transformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head, attn_dropout=attn_dropout, ff_dropout=ff_dropout)
        input_size = dim * self.num_categories + num_continuous
        hidden_dimensions = [(input_size * t) for t in mlp_hidden_mults]
        all_dimensions = [input_size, *hidden_dimensions, dim_out]
        self.mlp = MLP(all_dimensions, act=mlp_act)

    def forward(self, x_categ, x_cont, return_attn=False):
        xs = []
        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset
            categ_embed = self.category_embed(x_categ)
            if self.use_shared_categ_embed:
                shared_categ_embed = repeat(self.shared_category_embed, 'n d -> b n d', b=categ_embed.shape[0])
                categ_embed = torch.cat((categ_embed, shared_categ_embed), dim=-1)
            x, attns = self.transformer(categ_embed, return_attn=True)
            flat_categ = rearrange(x, 'b ... -> b (...)')
            xs.append(flat_categ)
        assert x_cont.shape[1] == self.num_continuous, f'you must pass in {self.num_continuous} values for your continuous input'
        if self.num_continuous > 0:
            if exists(self.continuous_mean_std):
                mean, std = self.continuous_mean_std.unbind(dim=-1)
                x_cont = (x_cont - mean) / std
            normed_cont = self.norm(x_cont)
            xs.append(normed_cont)
        x = torch.cat(xs, dim=-1)
        logits = self.mlp(x)
        if not return_attn:
            return logits
        return logits, attns


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (FeedForward,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GEGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'dims': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PreNorm,
     lambda: ([], {'dim': 4, 'fn': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Residual,
     lambda: ([], {'fn': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

