
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


from functools import partial


import torch


from torch import Tensor


from torch import is_tensor


from torch.utils.data import Dataset


from torch.nn.utils.rnn import pad_sequence


from numpy.lib.format import open_memmap


from math import ceil


from math import pi


from math import sqrt


from torch import nn


from torch import einsum


from torch.nn import Module


from torch.nn import ModuleList


import torch.nn.functional as F


from torch.utils.checkpoint import checkpoint


from torch.cuda.amp import autocast


from torch.utils.data import DataLoader


from torch.optim.lr_scheduler import _LRScheduler


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


class FiLM(Module):

    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.to_gamma = nn.Linear(dim, dim_out, bias=False)
        self.to_beta = nn.Linear(dim, dim_out)
        self.gamma_mult = nn.Parameter(torch.zeros(1))
        self.beta_mult = nn.Parameter(torch.zeros(1))

    def forward(self, x, cond):
        gamma, beta = self.to_gamma(cond), self.to_beta(cond)
        gamma, beta = tuple(rearrange(t, 'b d -> b 1 d') for t in (gamma, beta))
        gamma = 1 + self.gamma_mult * gamma.tanh()
        beta = beta.tanh() * self.beta_mult
        return x * gamma + beta


class PixelNorm(Module):

    def __init__(self, dim, eps=0.0001):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        dim = self.dim
        return F.normalize(x, dim=dim, eps=self.eps) * sqrt(x.shape[dim])


class SqueezeExcite(Module):

    def __init__(self, dim, reduction_factor=4, min_dim=16):
        super().__init__()
        dim_inner = max(dim // reduction_factor, min_dim)
        self.net = nn.Sequential(nn.Linear(dim, dim_inner), nn.SiLU(), nn.Linear(dim_inner, dim), nn.Sigmoid(), Rearrange('b c -> b c 1'))

    def forward(self, x, mask=None):
        if exists(mask):
            x = x.masked_fill(~mask, 0.0)
            num = reduce(x, 'b c n -> b c', 'sum')
            den = reduce(mask.float(), 'b 1 n -> b 1', 'sum')
            avg = num / den.clamp(min=1e-05)
        else:
            avg = reduce(x, 'b c n -> b c', 'mean')
        return x * self.net(avg)


class Block(Module):

    def __init__(self, dim, dim_out=None, dropout=0.0):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = PixelNorm(dim=1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x, mask=None):
        if exists(mask):
            x = x.masked_fill(~mask, 0.0)
        x = self.proj(x)
        if exists(mask):
            x = x.masked_fill(~mask, 0.0)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class ResnetBlock(Module):

    def __init__(self, dim, dim_out=None, *, dropout=0.0):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.block1 = Block(dim, dim_out, dropout=dropout)
        self.block2 = Block(dim_out, dim_out, dropout=dropout)
        self.excite = SqueezeExcite(dim_out)
        self.residual_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, mask=None):
        res = self.residual_conv(x)
        h = self.block1(x, mask=mask)
        h = self.block2(h, mask=mask)
        h = self.excite(h, mask=mask)
        return h + res


def is_tensor_empty(t: 'Tensor'):
    return t.numel() == 0


class GateLoopBlock(Module):

    def __init__(self, dim, *, depth, use_heinsen=True):
        super().__init__()
        self.gateloops = ModuleList([])
        for _ in range(depth):
            gateloop = SimpleGateLoopLayer(dim=dim, use_heinsen=use_heinsen)
            self.gateloops.append(gateloop)

    def forward(self, x, cache=None):
        received_cache = exists(cache)
        if is_tensor_empty(x):
            return x, None
        if received_cache:
            prev, x = x[:, :-1], x[:, -1:]
        cache = default(cache, [])
        cache = iter(cache)
        new_caches = []
        for gateloop in self.gateloops:
            layer_cache = next(cache, None)
            out, new_cache = gateloop(x, cache=layer_cache, return_cache=True)
            new_caches.append(new_cache)
            x = x + out
        if received_cache:
            x = torch.cat((prev, x), dim=-2)
        return x, new_caches


class TorchTyping:

    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: 'str'):
        return self.abstract_dtype[Tensor, shapes]


def identity(t):
    return t


def first(it):
    return it[0]


def l1norm(t):
    return F.normalize(t, dim=-1, p=1)


def l2norm(t):
    return F.normalize(t, dim=-1, p=2)


def derive_angle(x, y, eps=1e-05):
    z = einsum('... d, ... d -> ...', l2norm(x), l2norm(y))
    return z.clip(-1 + eps, 1 - eps).arccos()


def divisible_by(num, den):
    return num % den == 0


def is_odd(n):
    return not divisible_by(n, 2)


def pad_at_dim(t, padding, dim=-1, value=0):
    ndim = t.ndim
    right_dims = ndim - dim - 1 if dim >= 0 else -dim - 1
    zeros = (0, 0) * right_dims
    return F.pad(t, (*zeros, *padding), value=value)


def masked_mean(tensor, mask, dim=-1, eps=1e-05):
    if not exists(mask):
        return tensor.mean(dim=dim)
    mask = rearrange(mask, '... -> ... 1')
    tensor = tensor.masked_fill(~mask, 0.0)
    total_el = mask.sum(dim=dim)
    num = tensor.sum(dim=dim)
    den = total_el.float().clamp(min=eps)
    mean = num / den
    mean = mean.masked_fill(total_el == 0, 0.0)
    return mean


def pad_to_length(t, length, dim=-1, value=0, right=True):
    curr_length = t.shape[dim]
    remainder = length - curr_length
    if remainder <= 0:
        return t
    padding = (0, remainder) if right else (remainder, 0)
    return pad_at_dim(t, padding, dim=dim, value=value)


def safe_cat(tensors, dim):
    tensors = [*filter(exists, tensors)]
    if len(tensors) == 0:
        return None
    elif len(tensors) == 1:
        return first(tensors)
    return torch.cat(tensors, dim=dim)


def set_module_requires_grad_(module: 'Module', requires_grad: 'bool'):
    for param in module.parameters():
        param.requires_grad = requires_grad


def always(value):

    def inner(*args, **kwargs):
        return value
    return inner


def custom_collate(data, pad_id=-1):
    is_dict = isinstance(first(data), dict)
    if is_dict:
        keys = first(data).keys()
        data = [d.values() for d in data]
    output = []
    for datum in zip(*data):
        if is_tensor(first(datum)):
            datum = pad_sequence(datum, batch_first=True, padding_value=pad_id)
        else:
            datum = list(datum)
        output.append(datum)
    output = tuple(output)
    if is_dict:
        output = dict(zip(keys, output))
    return output


def cycle(dl):
    while True:
        for data in dl:
            yield data


def maybe_del(d: 'dict', *keys):
    for key in keys:
        if key not in d:
            continue
        del d[key]


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Block,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (PixelNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
]

