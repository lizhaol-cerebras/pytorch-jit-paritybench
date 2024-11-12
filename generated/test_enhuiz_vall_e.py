
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


import logging


import random


from collections import defaultdict


from functools import cache


from functools import cached_property


from itertools import groupby


from itertools import zip_longest


from typing import Any


import numpy as np


from torch import Tensor


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


import string


import math


from functools import partial


from typing import Literal


from typing import overload


import torch.nn.functional as F


from torch import einsum


from torch import nn


from torch.distributions import Categorical


from torch.nn.utils.rnn import pad_sequence


from torch.utils.checkpoint import checkpoint


class SinusodialEmbedding(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        exponent = torch.arange(self.d_half, dtype=torch.float32)
        exponent = exponent / self.d_half
        omega = torch.exp(-math.log(10000.0) * exponent)
        self.omega: 'torch.Tensor'
        self.register_buffer('omega', omega, persistent=False)

    @property
    def d_half(self):
        assert self.d_model % 2 == 0, 'Only support even d_model.'
        return self.d_model // 2

    def forward(self, x):
        """
        Args:
            x: (...)
        Returns:
            pe: (... d)
        """
        omega = self.omega
        while omega.dim() <= x.dim():
            omega = omega.unsqueeze(0)
        x = x.unsqueeze(-1)
        x = omega * x
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        return x

    def get_pe(self, n: 'int'):
        """
        Args:
            n: int
        Returns:
            pe: (n d)
        """
        device = self.omega.device
        return self.forward(torch.arange(n, device=device))

    def add_pe(self, x):
        """
        Args:
            x: (b t c)
        """
        e = self.get_pe(x.shape[1])
        e = e[None]
        x = x + e
        return x


class Attention(nn.Module):

    def __init__(self, d_model, n_heads, casual):
        super().__init__()
        assert d_model % n_heads == 0
        dim_head = d_model // n_heads
        self.casual = casual
        self.n_heads = n_heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.to_out = nn.Linear(d_model, d_model)

    def forward(self, x, m):
        """
        Args:
            x: (b t c)
            m: (b t c), 1 is data, 0 is padding
        Returns:
            x: (b t c)
        """
        h = self.n_heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b t (h d) -> b t h d', h=h), (q, k, v))
        e = einsum('b i h d, b j h d -> b i j h', q, k)
        e = e * self.scale
        kpm = m.unsqueeze(1) * m.unsqueeze(2)
        if self.casual:
            kpm = kpm.squeeze(-1).tril().unsqueeze(-1)
        e = e.masked_fill(kpm == 0, -torch.finfo(e.dtype).max)
        a = e.softmax(dim=2)
        o = einsum('b i j h, b j h d -> b i h d', a, v)
        o = o.flatten(-2)
        o = self.to_out(o)
        o = o * m
        return o


class AdaLN(nn.Module):

    def __init__(self, d_model, n_levels, eps=1e-05, k=0.1, c=2):
        super().__init__()
        self.eps = eps
        self.emb = nn.Embedding(n_levels, d_model * 2)
        self.k = k
        self.c = c
        nn.init.zeros_(self.emb.weight)

    def forward(self, x, l):
        logγ, β = self.emb(l).unsqueeze(1).chunk(2, dim=-1)
        h = F.layer_norm(x, x.shape[-1:], eps=self.eps)
        h = self.c * (1 - (self.k * h).detach()) * h
        y = logγ.exp() * h + β
        return y


class PrenormResidual(nn.Module):

    def __init__(self, block, d_model, p_dropout, requires_mask=False, norm_type='ln', n_levels: 'int | None'=None):
        super().__init__()
        self.block = block
        self.requires_mask = requires_mask
        self.norm_type = norm_type
        if norm_type == 'ln':
            self.norm = nn.LayerNorm(d_model)
        elif norm_type == 'adaln':
            assert n_levels is not None
            self.norm = AdaLN(d_model, n_levels)
        else:
            raise NotImplementedError(norm_type)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, m, l):
        """
        Args:
            x: input (b t d)
            m: mask (b t 1), 1 is valuable and 0 is padding
            l: level to use, required only for AdaLN
        """
        nopts = {'l': l} if self.norm_type == 'adaln' else {}
        bopts = {'m': m} if self.requires_mask else {}
        x = x + self.dropout(self.block(self.norm(x, **nopts) * m, **bopts))
        return x * m


class Block(nn.Sequential):

    def __init__(self, d_model, n_heads, p_dropout, casual, norm_type, n_levels):
        super().__init__()
        self.attn = PrenormResidual(Attention(d_model, n_heads, casual), d_model=d_model, p_dropout=p_dropout, requires_mask=True, norm_type=norm_type, n_levels=n_levels)
        self.ffn = PrenormResidual(nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(p_dropout), nn.Linear(d_model * 4, d_model)), d_model=d_model, p_dropout=p_dropout, norm_type=norm_type, n_levels=n_levels)

    def forward(self, x, m, l):
        """
        Args:
            x: (b t c)
            m: (b t 1)
            l: (b)
        """
        poor_in_vram = True
        if x.requires_grad and poor_in_vram:
            x = checkpoint(self.attn, x, m, l)
        else:
            x = self.attn(x, m, l)
        x = self.ffn(x, m, l)
        return x


class Embedding(nn.Embedding):

    def forward(self, x_list: 'list[Tensor]') ->list[Tensor]:
        if len(x_list) == 0:
            return []
        return super().forward(torch.cat(x_list)).split([*map(len, x_list)])


class MultiEmbedding(nn.Module):
    """
    This embedding sums embeddings on different levels.
    """

    def __init__(self, max_n_levels, n_tokens, token_dim):
        super().__init__()
        self.max_n_levels = max_n_levels
        self.n_tokens = n_tokens
        self.weight = nn.Parameter(torch.randn(max_n_levels, n_tokens, token_dim))

    def forward(self, x_list: 'list[Tensor]') ->list[Tensor]:
        if len(x_list) == 0:
            return []
        w = self.weight
        padded_x_list = []
        for xi in x_list:
            xi = F.one_hot(xi, num_classes=self.n_tokens)
            xi = F.pad(xi, (0, 0, 0, w.shape[0] - xi.shape[1]))
            padded_x_list.append(xi)
        x = torch.cat(padded_x_list)
        x = einsum('l k d, n l k -> n d', w, x)
        x_list = x.split([*map(len, x_list)])
        return x_list


def _join(x: 'tuple[Tensor]', sep: 'Tensor'):
    """
    Args:
        x: (k t d)
        sep: (d)
    """
    ret = x[0]
    for i in range(1, len(x)):
        ret = torch.cat((ret, sep[None], x[i]), dim=0)
    return ret


def _create_mask(l, device):
    """1 is valid region and 0 is invalid."""
    seq = torch.arange(max(l), device=device).unsqueeze(0)
    stop = torch.tensor(l, device=device).unsqueeze(1)
    return (seq < stop).float()


def list_to_tensor(x_list: 'list[Tensor]', pattern='t b c -> b t c'):
    """
    Args:
        x_list: [(t d)]
    Returns:
        x: (? ? ?)
        m: (? ? ?), same as x
    """
    l = list(map(len, x_list))
    x = rearrange(pad_sequence(x_list), pattern)
    m = _create_mask(l, x_list[0].device)
    m = m.t().unsqueeze(-1)
    m = rearrange(m, pattern)
    m = m
    return x, m


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (PrenormResidual,
     lambda: ([], {'block': torch.nn.ReLU(), 'd_model': 4, 'p_dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SinusodialEmbedding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

