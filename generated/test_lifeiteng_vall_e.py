
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


import logging


import torch


import torch.multiprocessing


import copy


import random


import warnings


from typing import Any


from typing import Dict


from typing import Optional


from typing import Tuple


from typing import Union


import torch.multiprocessing as mp


import torch.nn as nn


from torch import Tensor


from torch.cuda.amp import GradScaler


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.tensorboard import SummaryWriter


from typing import List


import numpy as np


import inspect


from functools import lru_cache


from torch.utils.data import DataLoader


from typing import Callable


from typing import Sequence


import re


from typing import Pattern


from functools import partial


import torch.nn.functional as F


from typing import Iterator


import matplotlib.pyplot as plt


from torch.nn import Linear


from torch.nn import Module


from torch.nn import functional as F


from torch.nn.init import constant_


from torch.nn.init import xavier_normal_


from torch.nn.init import xavier_uniform_


from torch.nn.modules.linear import NonDynamicallyQuantizableLinear


from torch.nn.parameter import Parameter


import math


from collections import defaultdict


from torch.optim import Optimizer


import collections


from functools import reduce


from itertools import repeat


from torch.nn import Embedding as ScaledEmbedding


import numbers


from torch import nn


class ActivationBalancerFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: 'Tensor', scale_factor: 'Tensor', sign_factor: 'Optional[Tensor]', channel_dim: 'int') ->Tensor:
        if channel_dim < 0:
            channel_dim += x.ndim
        ctx.channel_dim = channel_dim
        xgt0 = x > 0
        if sign_factor is None:
            ctx.save_for_backward(xgt0, scale_factor)
        else:
            ctx.save_for_backward(xgt0, scale_factor, sign_factor)
        return x

    @staticmethod
    def backward(ctx, x_grad: 'Tensor') ->Tuple[Tensor, None, None, None]:
        if len(ctx.saved_tensors) == 3:
            xgt0, scale_factor, sign_factor = ctx.saved_tensors
            for _ in range(ctx.channel_dim, x_grad.ndim - 1):
                scale_factor = scale_factor.unsqueeze(-1)
                sign_factor = sign_factor.unsqueeze(-1)
            factor = sign_factor + scale_factor * (xgt0 - 0.5)
        else:
            xgt0, scale_factor = ctx.saved_tensors
            for _ in range(ctx.channel_dim, x_grad.ndim - 1):
                scale_factor = scale_factor.unsqueeze(-1)
            factor = scale_factor * (xgt0 - 0.5)
        neg_delta_grad = x_grad.abs() * factor
        return x_grad - neg_delta_grad, None, None, None


def _compute_scale_factor(x: 'Tensor', channel_dim: 'int', min_abs: 'float', max_abs: 'float', gain_factor: 'float', max_factor: 'float') ->Tensor:
    if channel_dim < 0:
        channel_dim += x.ndim
    sum_dims = [d for d in range(x.ndim) if d != channel_dim]
    x_abs_mean = torch.mean(x.abs(), dim=sum_dims)
    if min_abs == 0.0:
        below_threshold = 0.0
    else:
        below_threshold = ((min_abs - x_abs_mean) * (gain_factor / min_abs)).clamp(min=0, max=max_factor)
    above_threshold = ((x_abs_mean - max_abs) * (gain_factor / max_abs)).clamp(min=0, max=max_factor)
    return below_threshold - above_threshold


def _compute_sign_factor(x: 'Tensor', channel_dim: 'int', min_positive: 'float', max_positive: 'float', gain_factor: 'float', max_factor: 'float') ->Tensor:
    if channel_dim < 0:
        channel_dim += x.ndim
    sum_dims = [d for d in range(x.ndim) if d != channel_dim]
    proportion_positive = torch.mean(x > 0, dim=sum_dims)
    if min_positive == 0.0:
        factor1 = 0.0
    else:
        factor1 = ((min_positive - proportion_positive) * (gain_factor / min_positive)).clamp_(min=0, max=max_factor)
    if max_positive == 1.0:
        factor2 = 0.0
    else:
        factor2 = ((proportion_positive - max_positive) * (gain_factor / (1.0 - max_positive))).clamp_(min=0, max=max_factor)
    sign_factor = factor1 - factor2
    assert not isinstance(sign_factor, float)
    return sign_factor


def _no_op(x: 'Tensor') ->Tensor:
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return x
    else:
        return x.chunk(1, dim=-1)[0]


class ActivationBalancer(torch.nn.Module):
    """
    Modifies the backpropped derivatives of a function to try to encourage, for
    each channel, that it is positive at least a proportion `threshold` of the
    time.  It does this by multiplying negative derivative values by up to
    (1+max_factor), and positive derivative values by up to (1-max_factor),
    interpolated from 1 at the threshold to those extremal values when none
    of the inputs are positive.

    Args:
           num_channels: the number of channels
           channel_dim: the dimension/axis corresponding to the channel, e.g.
               -1, 0, 1, 2; will be interpreted as an offset from x.ndim if negative.
           min_positive: the minimum, per channel, of the proportion of the time
               that (x > 0), below which we start to modify the derivatives.
           max_positive: the maximum, per channel, of the proportion of the time
               that (x > 0), above which we start to modify the derivatives.
           max_factor: the maximum factor by which we modify the derivatives for
              either the sign constraint or the magnitude constraint;
              e.g. with max_factor=0.02, the the derivatives would be multiplied by
              values in the range [0.98..1.02].
           sign_gain_factor: determines the 'gain' with which we increase the
              change in gradient once the constraints on min_positive and max_positive
              are violated.
           scale_gain_factor: determines the 'gain' with which we increase the
              change in gradient once the constraints on min_abs and max_abs
              are violated.
           min_abs:  the minimum average-absolute-value difference from the mean
               value per channel, which we allow, before we start to modify
               the derivatives to prevent this.
           max_abs:  the maximum average-absolute-value difference from the mean
               value per channel, which we allow, before we start to modify
               the derivatives to prevent this.
          min_prob: determines the minimum probability with which we modify the
             gradients for the {min,max}_positive and {min,max}_abs constraints,
             on each forward().  This is done randomly to prevent all layers
             from doing it at the same time.  Early in training we may use
             higher probabilities than this; it will decay to this value.
    """

    def __init__(self, num_channels: 'int', channel_dim: 'int', min_positive: 'float'=0.05, max_positive: 'float'=0.95, max_factor: 'float'=0.04, sign_gain_factor: 'float'=0.01, scale_gain_factor: 'float'=0.02, min_abs: 'float'=0.2, max_abs: 'float'=100.0, min_prob: 'float'=0.1):
        super(ActivationBalancer, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.min_positive = min_positive
        self.max_positive = max_positive
        self.max_factor = max_factor
        self.min_abs = min_abs
        self.max_abs = max_abs
        self.min_prob = min_prob
        self.sign_gain_factor = sign_gain_factor
        self.scale_gain_factor = scale_gain_factor
        self.cpu_count = 0
        self.register_buffer('count', torch.tensor(0, dtype=torch.int64))

    def forward(self, x: 'Tensor') ->Tensor:
        if torch.jit.is_scripting() or not x.requires_grad or torch.jit.is_tracing():
            return _no_op(x)
        count = self.cpu_count
        self.cpu_count += 1
        if random.random() < 0.01:
            self.cpu_count = max(self.cpu_count, self.count.item())
            self.count.fill_(self.cpu_count)
        prob = max(self.min_prob, 0.5 ** (1 + count / 4000.0))
        if random.random() < prob:
            sign_gain_factor = 0.5
            if self.min_positive != 0.0 or self.max_positive != 1.0:
                sign_factor = _compute_sign_factor(x, self.channel_dim, self.min_positive, self.max_positive, gain_factor=self.sign_gain_factor / prob, max_factor=self.max_factor)
            else:
                sign_factor = None
            scale_factor = _compute_scale_factor(x.detach(), self.channel_dim, min_abs=self.min_abs, max_abs=self.max_abs, gain_factor=self.scale_gain_factor / prob, max_factor=self.max_factor)
            return ActivationBalancerFunction.apply(x, scale_factor, sign_factor, self.channel_dim)
        else:
            return _no_op(x)


class BalancedBasicNorm(nn.Module):

    def __init__(self, d_model: 'int', eps: 'float'=1e-05, device=None, dtype=None):
        super(BalancedBasicNorm, self).__init__()
        self.balancer = ActivationBalancer(d_model, channel_dim=-1, min_positive=0.45, max_positive=0.55, max_abs=6.0)
        self.norm = BasicNorm(d_model, eps, device=device, dtype=dtype)

    def forward(self, input: 'Tensor', embedding: 'Any'=None) ->Tensor:
        if isinstance(input, tuple):
            input, embedding = input
            return self.norm((self.balancer(input), embedding))
        assert embedding is None
        return self.norm(self.balancer(input))


class DoubleSwishFunction(torch.autograd.Function):
    """
      double_swish(x) = x * torch.sigmoid(x-1)
    This is a definition, originally motivated by its close numerical
    similarity to swish(swish(x)), where swish(x) =  x * sigmoid(x).

    Memory-efficient derivative computation:
     double_swish(x) = x * s, where s(x) = torch.sigmoid(x-1)
     double_swish'(x) = d/dx double_swish(x) =  x * s'(x) + x' * s(x) = x * s'(x) + s(x).
     Now, s'(x) = s(x) * (1-s(x)).
     double_swish'(x) =  x * s'(x) + s(x).
                      =  x * s(x) * (1-s(x)) + s(x).
                     = double_swish(x) * (1-s(x)) + s(x)
     ... so we just need to remember s(x) but not x itself.
    """

    @staticmethod
    def forward(ctx, x: 'Tensor') ->Tensor:
        requires_grad = x.requires_grad
        x_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x
        s = torch.sigmoid(x - 1.0)
        y = x * s
        if requires_grad:
            deriv = y * (1 - s) + s
            floor = -0.043637
            ceil = 1.2
            d_scaled = (deriv - floor) * (255.0 / (ceil - floor)) + torch.rand_like(deriv)
            if __name__ == '__main__':
                assert d_scaled.min() >= 0.0
                assert d_scaled.max() < 256.0
            d_int = d_scaled
            ctx.save_for_backward(d_int)
        if x.dtype == torch.float16 or torch.is_autocast_enabled():
            y = y
        return y

    @staticmethod
    def backward(ctx, y_grad: 'Tensor') ->Tensor:
        d, = ctx.saved_tensors
        floor = -0.043637
        ceil = 1.2
        d = d * ((ceil - floor) / 255.0) + floor
        return y_grad * d


class DoubleSwish(torch.nn.Module):

    def forward(self, x: 'Tensor') ->Tensor:
        """Return double-swish activation function which is an approximation to Swish(Swish(x)),
        that we approximate closely with x * sigmoid(x-1).
        """
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return x * torch.sigmoid(x - 1.0)
        return DoubleSwishFunction.apply(x)


def BalancedDoubleSwish(d_model, channel_dim=-1, max_abs=10.0, min_prob=0.25) ->nn.Sequential:
    """
    ActivationBalancer -> DoubleSwish
    """
    balancer = ActivationBalancer(d_model, channel_dim=channel_dim, max_abs=max_abs, min_prob=min_prob)
    return nn.Sequential(balancer, DoubleSwish())


class IdentityNorm(nn.Module):

    def __init__(self, d_model: 'int', eps: 'float'=1e-05, device=None, dtype=None) ->None:
        super(IdentityNorm, self).__init__()

    def forward(self, input: 'Tensor', embedding: 'Any'=None) ->Tensor:
        if isinstance(input, tuple):
            return input
        assert embedding is None
        return input


NUM_MEL_BINS = 100


NUM_TEXT_TOKENS = 512


def ScaledLinear(*args, initial_scale: float=1.0, **kwargs) ->nn.Linear:
    """
    Behaves like a constructor of a modified version of nn.Linear
    that gives an easy way to set the default initial parameter scale.

    Args:
        Accepts the standard args and kwargs that nn.Linear accepts
        e.g. in_features, out_features, bias=False.

        initial_scale: you can override this if you want to increase
           or decrease the initial magnitude of the module's output
           (affects the initialization of weight_scale and bias_scale).
           Another option, if you want to do something like this, is
           to re-initialize the parameters.
    """
    ans = nn.Linear(*args, **kwargs)
    with torch.no_grad():
        ans.weight[:] *= initial_scale
        if ans.bias is not None:
            torch.nn.init.uniform_(ans.bias, -0.1 * initial_scale, 0.1 * initial_scale)
    return ans


class SinePositionalEmbedding(nn.Module):

    def __init__(self, dim_model: 'int', dropout: 'float'=0.0, scale: 'bool'=False, alpha: 'bool'=False):
        super().__init__()
        self.dim_model = dim_model
        self.x_scale = math.sqrt(dim_model) if scale else 1.0
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=alpha)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.reverse = False
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, 4000))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe
                return
        pe = torch.zeros(x.size(1), self.dim_model)
        if self.reverse:
            position = torch.arange(x.size(1) - 1, -1, -1.0, dtype=torch.float32).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.detach()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        self.extend_pe(x)
        output = x.unsqueeze(-1) if x.ndim == 2 else x
        output = output * self.x_scale + self.alpha * self.pe[:, :x.size(1)]
        return self.dropout(output)


class TokenEmbedding(nn.Module):

    def __init__(self, dim_model: 'int', vocab_size: 'int', dropout: 'float'=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim_model = dim_model
        self.dropout = torch.nn.Dropout(p=dropout)
        self.word_embeddings = nn.Embedding(self.vocab_size, self.dim_model)

    @property
    def weight(self) ->torch.Tensor:
        return self.word_embeddings.weight

    def embedding(self, index: 'int') ->torch.Tensor:
        return self.word_embeddings.weight[index:index + 1]

    def forward(self, x: 'torch.Tensor'):
        X = self.word_embeddings(x)
        X = self.dropout(X)
        return X


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization"""

    def __init__(self, d_model, norm) ->None:
        super(AdaptiveLayerNorm, self).__init__()
        self.project_layer = nn.Linear(d_model, 2 * d_model)
        self.norm = norm
        self.d_model = d_model
        self.eps = self.norm.eps

    def forward(self, input: 'Tensor', embedding: 'Tensor'=None) ->Tensor:
        if isinstance(input, tuple):
            input, embedding = input
            weight, bias = torch.split(self.project_layer(embedding), split_size_or_sections=self.d_model, dim=-1)
            return weight * self.norm(input) + bias, embedding
        weight, bias = torch.split(self.project_layer(embedding), split_size_or_sections=self.d_model, dim=-1)
        return weight * self.norm(input) + bias


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: 'Tuple[int, ...]'
    eps: 'float'
    elementwise_affine: 'bool'

    def __init__(self, normalized_shape: '_shape_t', eps: 'float'=1e-05, elementwise_affine: 'bool'=True, device=None, dtype=None) ->None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = normalized_shape,
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: 'Tensor', embedding: 'Any'=None) ->Tensor:
        if isinstance(input, tuple):
            input, embedding = input
            return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps), embedding
        assert embedding is None
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self) ->str:
        return '{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class MultiheadAttention(Module):
    """Allows the model to jointly attend to information
    from different representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Multi-Head Attention is defined as:

    .. math::
        \\text{MultiHead}(Q, K, V) = \\text{Concat}(head_1,\\dots,head_h)W^O

    where :math:`head_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    ``forward()`` will use a special optimized implementation if all of the following
    conditions are met:

    - self attention is being computed (i.e., ``query``, ``key``, and ``value`` are the same tensor. This
      restriction will be loosened in the future.)
    - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor argument ``requires_grad``
    - training is disabled (using ``.eval()``)
    - dropout is 0
    - ``add_bias_kv`` is ``False``
    - ``add_zero_attn`` is ``False``
    - ``batch_first`` is ``True`` and the input is batched
    - ``kdim`` and ``vdim`` are equal to ``embed_dim``
    - at most one of ``key_padding_mask`` or ``attn_mask`` is passed
    - if a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ is passed, neither ``key_padding_mask``
      nor ``attn_mask`` is passed

    If the optimized implementation is in use, a
    `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be passed for
    ``query``/``key``/``value`` to represent padding more efficiently than using a
    padding mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_
    will be returned, and an additional speedup proportional to the fraction of the input
    that is padding can be expected.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> # xdoctest: +SKIP
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)

    """
    __constants__ = ['batch_first']
    bias_k: 'Optional[torch.Tensor]'
    bias_v: 'Optional[torch.Tensor]'

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, linear1_cls=Linear, linear2_cls=Linear, device=None, dtype=None) ->None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None
        if linear1_cls == Linear:
            if not self._qkv_same_embed_dim:
                self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
                self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
                self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
                self.register_parameter('in_proj_weight', None)
            else:
                self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
                self.register_parameter('q_proj_weight', None)
                self.register_parameter('k_proj_weight', None)
                self.register_parameter('v_proj_weight', None)
            if bias:
                self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
            else:
                self.register_parameter('in_proj_bias', None)
            self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
            self._reset_parameters()
        else:
            if not self._qkv_same_embed_dim:
                raise NotImplementedError
            else:
                self.in_proj_linear = linear1_cls(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
                self.in_proj_weight = self.in_proj_linear.weight
                self.register_parameter('q_proj_weight', None)
                self.register_parameter('k_proj_weight', None)
                self.register_parameter('v_proj_weight', None)
                if bias:
                    self.in_proj_bias = self.in_proj_linear.bias
                else:
                    self.register_parameter('in_proj_bias', None)
            self.out_proj = linear2_cls(embed_dim, embed_dim, bias=bias, **factory_kwargs)
            if self.bias_k is not None:
                xavier_normal_(self.bias_k)
            if self.bias_v is not None:
                xavier_normal_(self.bias_v)
        self.add_zero_attn = add_zero_attn

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True
        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query: 'Tensor', key: 'Tensor', value: 'Tensor', key_padding_mask: 'Optional[Tensor]'=None, need_weights: 'bool'=True, attn_mask: 'Optional[Tensor]'=None, average_attn_weights: 'bool'=True) ->Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
                or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
                :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
                Queries are compared against key-value pairs to produce the output.
                See "Attention Is All You Need" for more details.
            key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
                or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
                :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
                See "Attention Is All You Need" for more details.
            value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
                ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
                sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
                See "Attention Is All You Need" for more details.
            key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
                to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
                Binary and byte masks are supported.
                For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
                the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
            need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
                Default: ``True``.
            attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
                :math:`(L, S)` or :math:`(N\\cdot\\text{num\\_heads}, L, S)`, where :math:`N` is the batch size,
                :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
                broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
                Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
                corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
                corresponding position is not allowed to attend. For a float mask, the mask values will be added to
                the attention weight.
            average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
                heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
                effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)

        Outputs:
            - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
              :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
              where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
              embedding dimension ``embed_dim``.
            - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
              returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
              :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
              :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
              head of shape :math:`(\\text{num\\_heads}, L, S)` when input is unbatched or :math:`(N, \\text{num\\_heads}, L, S)`.

            .. note::
                `batch_first` argument is ignored for unbatched inputs.
        """
        is_batched = query.dim() == 3
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != torch.bool and not torch.is_floating_point(key_padding_mask):
                raise AssertionError('only bool and floating types of key_padding_mask are supported')
        why_not_fast_path = ''
        if not is_batched:
            why_not_fast_path = f'input not batched; expected query.dim() of 3 but got {query.dim()}'
        elif query is not key or key is not value:
            why_not_fast_path = 'non-self attention was used (query, key, and value are not the same Tensor)'
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = 'training is enabled'
        elif not self.batch_first:
            why_not_fast_path = 'batch_first was not True'
        elif self.bias_k is not None:
            why_not_fast_path = 'self.bias_k was not None'
        elif self.bias_v is not None:
            why_not_fast_path = 'self.bias_v was not None'
        elif self.dropout:
            why_not_fast_path = f'dropout was {self.dropout}, required zero'
        elif self.add_zero_attn:
            why_not_fast_path = 'add_zero_attn was enabled'
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = '_qkv_same_embed_dim was not True'
        elif attn_mask is not None:
            why_not_fast_path = 'attn_mask was not None'
        elif query.is_nested and key_padding_mask is not None:
            why_not_fast_path = 'key_padding_mask is not supported with NestedTensor input'
        elif self.num_heads % 2 == 1:
            why_not_fast_path = 'num_heads is odd'
        elif torch.is_autocast_enabled():
            why_not_fast_path = 'autocast is enabled'
        if not why_not_fast_path:
            tensor_args = query, key, value, self.in_proj_weight, self.in_proj_bias, self.out_proj.weight, self.out_proj.bias
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = 'some Tensor argument has_torch_function'
            elif not all([(x is None or x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]):
                why_not_fast_path = 'some Tensor argument is neither CUDA nor CPU'
            elif torch.is_grad_enabled() and any([(x is not None and x.requires_grad) for x in tensor_args]):
                why_not_fast_path = 'grad is enabled and at least one of query or the input/output projection weights or biases requires_grad'
            if not why_not_fast_path:
                return torch._native_multi_head_attention(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj.weight, self.out_proj.bias, key_padding_mask if key_padding_mask is not None else attn_mask, need_weights, average_attn_weights, 1 if key_padding_mask is not None else 0 if attn_mask is not None else None)
        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, 'MultiheadAttention does not support NestedTensor outside of its fast path. ' + f'The fast path was not hit because {why_not_fast_path}'
        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, use_separate_proj_weight=True, q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight, v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights)
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, average_attn_weights=average_attn_weights)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


def _get_activation_fn(activation: 'str') ->Callable[[Tensor], Tensor]:
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    raise RuntimeError('activation should be relu/gelu, not {}'.format(activation))


class TransformerDecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: 'int', nhead: 'int', dim_feedforward: 'int'=2048, dropout: 'float'=0.1, activation: 'Union[str, Callable[[Tensor], Tensor]]'=F.relu, linear1_self_attention_cls: 'nn.Module'=nn.Linear, linear2_self_attention_cls: 'nn.Module'=nn.Linear, linear1_feedforward_cls: 'nn.Module'=nn.Linear, linear2_feedforward_cls: 'nn.Module'=nn.Linear, batch_first: 'bool'=False, norm_first: 'bool'=False, device=None, dtype=None, layer_norm_cls: 'nn.Module'=LayerNorm, layer_norm_eps: 'float'=1e-05, adaptive_layer_norm=False) ->None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, linear1_cls=linear1_self_attention_cls, linear2_cls=linear2_self_attention_cls, **factory_kwargs)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, linear1_cls=linear1_self_attention_cls, linear2_cls=linear2_self_attention_cls, **factory_kwargs)
        self.linear1 = linear1_feedforward_cls(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = linear2_feedforward_cls(dim_feedforward, d_model, **factory_kwargs)
        self.norm_first = norm_first
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        elif isinstance(activation, partial):
            self.activation = activation(d_model)
        elif activation == BalancedDoubleSwish:
            self.activation = BalancedDoubleSwish(d_model)
        else:
            self.activation = activation
        if adaptive_layer_norm:
            norm1 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)
            norm2 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)
            norm3 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm1 = AdaptiveLayerNorm(d_model, norm1)
            self.norm2 = AdaptiveLayerNorm(d_model, norm2)
            self.norm3 = AdaptiveLayerNorm(d_model, norm3)
        else:
            self.norm1 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)
            if layer_norm_cls == IdentityNorm:
                self.norm3 = BalancedBasicNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            else:
                self.norm3 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)

    def forward(self, tgt: 'Tensor', memory: 'Tensor', tgt_mask: 'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None) ->Tensor:
        """Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt_is_tuple = False
        if isinstance(tgt, tuple):
            x, stage_embedding = tgt
            tgt_is_tuple = True
        else:
            x, stage_embedding = tgt, None
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x, stage_embedding), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x, stage_embedding), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x, stage_embedding))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask), stage_embedding)
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask), stage_embedding)
            x = self.norm3(x + self._ff_block(x), stage_embedding)
        if tgt_is_tuple:
            return x, stage_embedding
        return x

    def _sa_block(self, x: 'Tensor', attn_mask: 'Optional[Tensor]', key_padding_mask: 'Optional[Tensor]') ->Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _mha_block(self, x: 'Tensor', mem: 'Tensor', attn_mask: 'Optional[Tensor]', key_padding_mask: 'Optional[Tensor]') ->Tensor:
        x = self.multihead_attn(x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout2(x)

    def _ff_block(self, x: 'Tensor') ->Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):
    """TransformerEncoder is a stack of N encoder layers. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: 'Tensor', mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, return_layer_states: 'bool'=False) ->Tensor:
        """Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            return_layer_states: return layers' state (optional).

        Shape:
            see the docs in Transformer class.
        """
        if return_layer_states:
            layer_states = []
            output = src
            for mod in self.layers:
                output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
                layer_states.append(output[0])
            if self.norm is not None:
                output = self.norm(output)
            return layer_states, output
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: 'int', nhead: 'int', dim_feedforward: 'int'=2048, dropout: 'float'=0.1, activation: 'Union[str, Callable[[Tensor], Tensor]]'=F.relu, batch_first: 'bool'=False, norm_first: 'bool'=False, device=None, dtype=None, linear1_self_attention_cls: 'nn.Module'=nn.Linear, linear2_self_attention_cls: 'nn.Module'=nn.Linear, linear1_feedforward_cls: 'nn.Module'=nn.Linear, linear2_feedforward_cls: 'nn.Module'=nn.Linear, layer_norm_cls: 'nn.Module'=LayerNorm, layer_norm_eps: 'float'=1e-05, adaptive_layer_norm=False) ->None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, linear1_cls=linear1_self_attention_cls, linear2_cls=linear2_self_attention_cls, **factory_kwargs)
        self.linear1 = linear1_feedforward_cls(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = linear2_feedforward_cls(dim_feedforward, d_model, **factory_kwargs)
        self.norm_first = norm_first
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        elif isinstance(activation, partial):
            activation = activation(d_model)
        elif activation == BalancedDoubleSwish:
            activation = BalancedDoubleSwish(d_model)
        self.activation = activation
        norm1 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)
        if layer_norm_cls == IdentityNorm:
            norm2 = BalancedBasicNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        else:
            norm2 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)
        if adaptive_layer_norm:
            self.norm1 = AdaptiveLayerNorm(d_model, norm1)
            self.norm2 = AdaptiveLayerNorm(d_model, norm2)
        else:
            self.norm1 = norm1
            self.norm2 = norm2

    def __setstate__(self, state):
        super(TransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu

    def forward(self, src: 'Tensor', src_mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None) ->Tensor:
        """Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        x, stage_embedding = src, None
        is_src_tuple = False
        if isinstance(src, tuple):
            x, stage_embedding = src
            is_src_tuple = True
        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(src_key_padding_mask):
                raise AssertionError('only bool and floating types of key_padding_mask are supported')
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x, stage_embedding), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x, stage_embedding))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask), stage_embedding)
            x = self.norm2(x + self._ff_block(x), stage_embedding)
        if is_src_tuple:
            return x, stage_embedding
        return x

    def _sa_block(self, x: 'Tensor', attn_mask: 'Optional[Tensor]', key_padding_mask: 'Optional[Tensor]') ->Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: 'Tensor') ->Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class Transpose(nn.Identity):
    """(N, T, D) -> (N, D, T)"""

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return input.transpose(1, 2)


class Transformer(nn.Module):
    """It implements seq2seq Transformer TTS for debug(No StopPredictor and SpeakerEmbeding)
    Neural Speech Synthesis with Transformer Network
    https://arxiv.org/abs/1809.08895
    """

    def __init__(self, d_model: 'int', nhead: 'int', num_layers: 'int', norm_first: 'bool'=True, add_prenet: 'bool'=False, scaling_xformers: 'bool'=False):
        """
        Args:
          d_model:
            The number of expected features in the input (required).
          nhead:
            The number of heads in the multiheadattention models (required).
          num_layers:
            The number of sub-decoder-layers in the decoder (required).
        """
        super().__init__()
        self.text_embedding = TokenEmbedding(d_model, NUM_TEXT_TOKENS)
        if add_prenet:
            self.encoder_prenet = nn.Sequential(Transpose(), nn.Conv1d(d_model, d_model, kernel_size=5, padding='same'), nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(0.5), nn.Conv1d(d_model, d_model, kernel_size=5, padding='same'), nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(0.5), nn.Conv1d(d_model, d_model, kernel_size=5, padding='same'), nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(0.5), Transpose(), nn.Linear(d_model, d_model))
            self.decoder_prenet = nn.Sequential(nn.Linear(NUM_MEL_BINS, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, d_model))
            assert scaling_xformers is False
        else:
            self.encoder_prenet = nn.Identity()
            if scaling_xformers:
                self.decoder_prenet = ScaledLinear(NUM_MEL_BINS, d_model)
            else:
                self.decoder_prenet = nn.Linear(NUM_MEL_BINS, d_model)
        self.encoder_position = SinePositionalEmbedding(d_model, dropout=0.1, scale=False)
        self.decoder_position = SinePositionalEmbedding(d_model, dropout=0.1, scale=False)
        if scaling_xformers:
            self.encoder = TransformerEncoder(TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4, dropout=0.1, batch_first=True, norm_first=norm_first, linear1_self_attention_cls=ScaledLinear, linear2_self_attention_cls=partial(ScaledLinear, initial_scale=0.01), linear1_feedforward_cls=ScaledLinear, linear2_feedforward_cls=partial(ScaledLinear, initial_scale=0.01), activation=partial(BalancedDoubleSwish, channel_dim=-1, max_abs=10.0, min_prob=0.25), layer_norm_cls=IdentityNorm), num_layers=num_layers, norm=BalancedBasicNorm(d_model) if norm_first else None)
            self.decoder = nn.TransformerDecoder(TransformerDecoderLayer(d_model, nhead, dim_feedforward=d_model * 4, dropout=0.1, batch_first=True, norm_first=norm_first, linear1_self_attention_cls=ScaledLinear, linear2_self_attention_cls=partial(ScaledLinear, initial_scale=0.01), linear1_feedforward_cls=ScaledLinear, linear2_feedforward_cls=partial(ScaledLinear, initial_scale=0.01), activation=partial(BalancedDoubleSwish, channel_dim=-1, max_abs=10.0, min_prob=0.25), layer_norm_cls=IdentityNorm), num_layers=num_layers, norm=BalancedBasicNorm(d_model) if norm_first else None)
            self.predict_layer = ScaledLinear(d_model, NUM_MEL_BINS)
            self.stop_layer = nn.Linear(d_model, 1)
        else:
            self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4, activation=F.relu, dropout=0.1, batch_first=True, norm_first=norm_first), num_layers=num_layers, norm=nn.LayerNorm(d_model) if norm_first else None)
            self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=d_model * 4, activation=F.relu, dropout=0.1, batch_first=True, norm_first=norm_first), num_layers=num_layers, norm=nn.LayerNorm(d_model) if norm_first else None)
            self.predict_layer = nn.Linear(d_model, NUM_MEL_BINS)
            self.stop_layer = nn.Linear(d_model, 1)
        self.stop_accuracy_metric = BinaryAccuracy(threshold=0.5, multidim_average='global')

    def forward(self, x: 'torch.Tensor', x_lens: 'torch.Tensor', y: 'torch.Tensor', y_lens: 'torch.Tensor', reduction: 'str'='sum', train_stage: 'int'=0, **kwargs) ->Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Args:
          x:
            A 2-D tensor of shape (N, S).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (N, T, 8).
          y_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          train_stage:
            Not used in this model.
        Returns:
          Return the predicted audio code matrix, cross-entropy loss and Top-10 accuracy.
        """
        del train_stage
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y_lens.ndim == 1, y_lens.shape
        assert torch.all(x_lens > 0)
        x_mask = make_pad_mask(x_lens)
        x = self.text_embedding(x)
        x = self.encoder_prenet(x)
        x = self.encoder_position(x)
        x = self.encoder(x, src_key_padding_mask=x_mask)
        total_loss, metrics = 0.0, {}
        y_mask = make_pad_mask(y_lens)
        y_mask_float = y_mask.type(torch.float32)
        data_mask = 1.0 - y_mask_float.unsqueeze(-1)

        def pad_y(y):
            y = F.pad(y, (0, 0, 1, 0, 0, 0), value=0).detach()
            return y[:, :-1], y[:, 1:]
        y, targets = pad_y(y * data_mask)
        y_emb = self.decoder_prenet(y)
        y_pos = self.decoder_position(y_emb)
        y_len = y_lens.max()
        tgt_mask = torch.triu(torch.ones(y_len, y_len, device=y.device, dtype=torch.bool), diagonal=1)
        y_dec = self.decoder(y_pos, x, tgt_mask=tgt_mask, memory_key_padding_mask=x_mask)
        predict = self.predict_layer(y_dec)
        total_loss = F.mse_loss(predict, targets, reduction=reduction)
        logits = self.stop_layer(y_dec).squeeze(-1)
        stop_loss = F.binary_cross_entropy_with_logits(logits, y_mask_float.detach(), weight=1.0 + y_mask_float.detach() * 4.0, reduction=reduction)
        metrics['stop_loss'] = stop_loss.detach()
        stop_accuracy = self.stop_accuracy_metric((torch.sigmoid(logits) >= 0.5).type(torch.int64), y_mask.type(torch.int64))
        metrics['stop_accuracy'] = stop_accuracy.item() * y_lens.sum().type(torch.float32)
        return (x, predict), total_loss + 100.0 * stop_loss, metrics

    def inference(self, x: 'torch.Tensor', x_lens: 'torch.Tensor', y: 'Any'=None, **kwargs) ->torch.Tensor:
        """
        Args:
          x:
            A 2-D tensor of shape (1, S).
          x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
        Returns:
          Return the predicted audio code matrix and cross-entropy loss.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert torch.all(x_lens > 0)
        x_mask = make_pad_mask(x_lens)
        x = self.text_embedding(x)
        x = self.encoder_prenet(x)
        x = self.encoder_position(x)
        x = self.encoder(x, src_key_padding_mask=x_mask)
        x_mask = make_pad_mask(x_lens)
        y = torch.zeros([x.shape[0], 1, NUM_MEL_BINS], dtype=torch.float32, device=x.device)
        while True:
            y_emb = self.decoder_prenet(y)
            y_pos = self.decoder_position(y_emb)
            tgt_mask = torch.triu(torch.ones(y.shape[1], y.shape[1], device=y.device, dtype=torch.bool), diagonal=1)
            y_dec = self.decoder(y_pos, x, tgt_mask=tgt_mask, memory_mask=None, memory_key_padding_mask=x_mask)
            predict = self.predict_layer(y_dec[:, -1:])
            logits = self.stop_layer(y_dec[:, -1:]) > 0
            if y.shape[1] > x_lens.max() * 10 or all(logits.cpu().numpy()):
                None
                break
            y = torch.concat([y, predict], dim=1)
        return y[:, 1:]

    def visualize(self, predicts: 'Tuple[torch.Tensor]', batch: 'Dict[str, Union[List, torch.Tensor]]', output_dir: 'str', limit: 'int'=4) ->None:
        visualize(predicts, batch, output_dir, limit=limit)


NUM_AUDIO_TOKENS = 1024


class PromptedFeatures:

    def __init__(self, prompts, features):
        self.prompts = prompts
        self.features = features

    def to(self, device):
        return PromptedFeatures(self.prompts, self.features)

    def sum(self):
        return self.features.sum()

    @property
    def ndim(self):
        return self.features.ndim

    @property
    def data(self):
        return self.prompts, self.features


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf'), min_tokens_to_keep=1):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def topk_sampling(logits, top_k=10, top_p=1.0, temperature=1.0):
    if temperature != 1.0:
        logits = logits / temperature
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return token


class VALLF(nn.Module):
    """It implements https://arxiv.org/abs/2301.02111
    "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers"
    """

    def __init__(self, d_model: 'int', nhead: 'int', num_layers: 'int', norm_first: 'bool'=True, add_prenet: 'bool'=False, decoder_cls: 'Union[nn.TransformerDecoder, nn.TransformerEncoder]'=nn.TransformerDecoder, decoder_layer_cls: 'Union[TransformerDecoderLayer, TransformerEncoderLayer]'=TransformerDecoderLayer, prefix_mode: 'int'=0, share_embedding: 'bool'=True, nar_scale_factor: 'float'=1.0, prepend_bos: 'bool'=False, num_quantizers: 'int'=8):
        """
        Args:
          d_model:
            The number of expected features in the input (required).
          nhead:
            The number of heads in the multiheadattention models (required).
          num_layers:
            The number of sub-decoder-layers in the decoder (required).
        """
        super().__init__()
        nar_d_model = int(d_model * nar_scale_factor)
        self.ar_text_embedding = TokenEmbedding(d_model, NUM_TEXT_TOKENS)
        self.nar_text_embedding = TokenEmbedding(nar_d_model, NUM_TEXT_TOKENS)
        self.ar_audio_prepend_bos = prepend_bos
        self.ar_audio_embedding = TokenEmbedding(d_model, NUM_AUDIO_TOKENS + 1 + int(prepend_bos))
        if add_prenet:
            self.ar_text_prenet = nn.Sequential(Transpose(), nn.Conv1d(d_model, d_model, kernel_size=5, padding='same'), nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(0.5), nn.Conv1d(d_model, d_model, kernel_size=5, padding='same'), nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(0.5), nn.Conv1d(d_model, d_model, kernel_size=5, padding='same'), nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(0.5), Transpose(), nn.Linear(d_model, d_model))
            self.ar_audio_prenet = nn.Sequential(nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(0.25), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.25), nn.Linear(256, d_model))
        else:
            self.ar_text_prenet = nn.Identity()
            self.ar_audio_prenet = nn.Identity()
        self.ar_text_position = SinePositionalEmbedding(d_model, dropout=0.1, scale=False, alpha=True)
        self.ar_audio_position = SinePositionalEmbedding(d_model, dropout=0.1, scale=False, alpha=True)
        self.ar_decoder = decoder_cls(decoder_layer_cls(d_model, nhead, dim_feedforward=d_model * 4, dropout=0.1, batch_first=True, norm_first=norm_first), num_layers=num_layers, norm=LayerNorm(d_model) if norm_first else None)
        self.ar_predict_layer = nn.Linear(d_model, NUM_AUDIO_TOKENS + 1, bias=False)
        self.ar_accuracy_metric = MulticlassAccuracy(NUM_AUDIO_TOKENS + 1, top_k=10, average='micro', multidim_average='global', ignore_index=NUM_AUDIO_TOKENS)
        self.rng = random.Random(0)
        self.num_heads = nhead
        self.prefix_mode = prefix_mode
        self.num_quantizers = num_quantizers
        assert num_quantizers >= 1
        if num_quantizers > 1:
            self.nar_audio_embeddings = nn.ModuleList([TokenEmbedding(nar_d_model, NUM_AUDIO_TOKENS + 1)] + [TokenEmbedding(nar_d_model, NUM_AUDIO_TOKENS) for i in range(num_quantizers - 1)])
            if add_prenet:
                self.nar_text_prenet = nn.Sequential(Transpose(), nn.Conv1d(nar_d_model, nar_d_model, kernel_size=5, padding='same'), nn.BatchNorm1d(nar_d_model), nn.ReLU(), nn.Dropout(0.5), nn.Conv1d(nar_d_model, nar_d_model, kernel_size=5, padding='same'), nn.BatchNorm1d(nar_d_model), nn.ReLU(), nn.Dropout(0.5), nn.Conv1d(nar_d_model, nar_d_model, kernel_size=5, padding='same'), nn.BatchNorm1d(nar_d_model), nn.ReLU(), nn.Dropout(0.5), Transpose(), nn.Linear(nar_d_model, nar_d_model))
                self.nar_audio_prenet = nn.Sequential(nn.Linear(nar_d_model, 256), nn.ReLU(), nn.Dropout(0.25), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.25), nn.Linear(256, nar_d_model))
            else:
                self.nar_text_prenet = nn.Identity()
                self.nar_audio_prenet = nn.Identity()
            self.nar_text_position = SinePositionalEmbedding(nar_d_model, dropout=0.0, scale=False, alpha=False)
            self.nar_audio_position = SinePositionalEmbedding(nar_d_model, dropout=0.1, scale=False, alpha=False)
            self.nar_decoder = decoder_cls(decoder_layer_cls(nar_d_model, int(nhead * nar_scale_factor), dim_feedforward=nar_d_model * 4, dropout=0.1, batch_first=True, norm_first=norm_first, adaptive_layer_norm=True), num_layers=int(num_layers * nar_scale_factor), norm=AdaptiveLayerNorm(nar_d_model, norm=nn.LayerNorm(nar_d_model)) if norm_first else None)
            self.nar_predict_layers = nn.ModuleList([nn.Linear(nar_d_model, NUM_AUDIO_TOKENS, bias=False) for i in range(num_quantizers - 1)])
            self.nar_stage_embeddings = nn.ModuleList([TokenEmbedding(nar_d_model, 1) for i in range(num_quantizers - 1)])
            if share_embedding:
                for j in range(0, num_quantizers - 2):
                    self.nar_predict_layers[j].weight = self.nar_audio_embeddings[j + 2].weight
            self.nar_accuracy_metric = MulticlassAccuracy(NUM_AUDIO_TOKENS + 1, top_k=10, average='micro', multidim_average='global', ignore_index=NUM_AUDIO_TOKENS)

    def stage_parameters(self, stage: 'int'=1) ->Iterator[nn.Parameter]:
        assert stage > 0
        if stage == 1:
            for name, param in self.named_parameters():
                if name.startswith('ar_'):
                    None
                    yield param
        if stage == 2:
            for name, param in self.named_parameters():
                if name.startswith('nar_'):
                    None
                    yield param

    def stage_named_parameters(self, stage: 'int'=1) ->Iterator[Tuple[str, nn.Parameter]]:
        assert stage > 0
        if stage == 1:
            for pair in self.named_parameters():
                if pair[0].startswith('ar_'):
                    yield pair
        if stage == 2:
            for pair in self.named_parameters():
                if pair[0].startswith('nar_'):
                    yield pair

    def pad_y_eos(self, y, y_mask_int, eos_id):
        targets = F.pad(y, (0, 1), value=0) + eos_id * F.pad(y_mask_int, (0, 1), value=1)
        if self.ar_audio_prepend_bos:
            return F.pad(targets[:, :-1], (1, 0), value=NUM_AUDIO_TOKENS + 1), targets
        return targets[:, :-1], targets[:, 1:]

    def _prepare_prompts(self, y, y_lens, codes, nar_stage, y_prompts_codes):
        if self.prefix_mode == 0:
            prefix_len = 0
            y_emb = self.nar_audio_embeddings[0](y)
            for j in range(1, nar_stage):
                y_emb = y_emb + self.nar_audio_embeddings[j](codes[..., j])
        elif self.prefix_mode == 1:
            int_low = (0.25 * y_lens.min()).type(torch.int64).item()
            prefix_len = torch.randint(int_low, int_low * 2, size=()).item()
            prefix_len = min(prefix_len, 225)
            y_prompts = self.nar_audio_embeddings[0](y[:, :prefix_len])
            y_emb = self.nar_audio_embeddings[0](y[:, prefix_len:])
            for j in range(1, self.num_quantizers):
                y_prompts += self.nar_audio_embeddings[j](codes[:, :prefix_len, j])
                if j < nar_stage:
                    y_emb += self.nar_audio_embeddings[j](codes[:, prefix_len:, j])
            y_emb = torch.concat([y_prompts, y_emb], axis=1)
        elif self.prefix_mode in [2, 4]:
            if self.prefix_mode == 2:
                prefix_len = min(225, int(0.25 * y_lens.min().item()))
                y_prompts_codes = []
                for b in range(codes.shape[0]):
                    start = self.rng.randint(0, y_lens[b].item() - prefix_len)
                    y_prompts_codes.append(torch.clone(codes[b, start:start + prefix_len]))
                    codes[b, start:start + prefix_len, nar_stage] = NUM_AUDIO_TOKENS
                y_prompts_codes = torch.stack(y_prompts_codes, dim=0)
            else:
                prefix_len = y_prompts_codes.shape[1]
            y_prompts = self.nar_audio_embeddings[0](y_prompts_codes[..., 0])
            y_emb = self.nar_audio_embeddings[0](y)
            for j in range(1, self.num_quantizers):
                y_prompts += self.nar_audio_embeddings[j](y_prompts_codes[..., j])
                if j < nar_stage:
                    y_emb += self.nar_audio_embeddings[j](codes[..., j])
            y_emb = torch.concat([y_prompts, y_emb], axis=1)
        else:
            raise ValueError
        return y_emb, prefix_len

    def forward(self, x: 'torch.Tensor', x_lens: 'torch.Tensor', y: 'Union[torch.Tensor, PromptedFeatures]', y_lens: 'Union[torch.Tensor, PromptedFeatures]', reduction: 'str'='sum', train_stage: 'int'=0, **kwargs) ->Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Args:
          x:
            A 2-D tensor of shape (N, S).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (N, T, 8).
          y_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          train_stage:
            0: AR & NAR modules, 1: AR modules, 2: NAR modules
        Returns:
          Return the predicted audio code matrix, cross-entropy loss and Top-10 accuracy.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        y_prompts_codes = None
        if isinstance(y, PromptedFeatures):
            y_prompts_codes, y = y.data
            prompts_len, y_lens = y_lens.data
            assert prompts_len.min() == prompts_len.max()
            assert self.prefix_mode == 4
            y_prompts_codes = y_prompts_codes.type(torch.int64)
        assert y.ndim == 3, y.shape
        assert y_lens.ndim == 1, y_lens.shape
        x_mask = make_pad_mask(x_lens)
        text = x
        x = self.ar_text_embedding(text)
        x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)
        total_loss, metrics = 0.0, {}
        y_mask = make_pad_mask(y_lens)
        y_mask_int = y_mask.type(torch.int64)
        codes = y.type(torch.int64) * (1 - y_mask_int.unsqueeze(dim=-1))
        y, targets = self.pad_y_eos(codes[..., 0], y_mask_int, eos_id=NUM_AUDIO_TOKENS)
        if train_stage in [0, 1]:
            y_emb = self.ar_audio_embedding(y)
            y_emb = self.ar_audio_prenet(y_emb)
            y_pos = self.ar_audio_position(y_emb)
            ar_y_mask = y_mask
            if self.ar_audio_prepend_bos:
                ar_y_mask = F.pad(y_mask, (1, 0), value=False)
            y_len = y_lens.max() + int(self.ar_audio_prepend_bos)
            tgt_mask = torch.triu(torch.ones(y_len, y_len, device=y.device, dtype=torch.bool), diagonal=1)
            y_dec, _ = self.ar_decoder((y_pos, None), x, tgt_mask=tgt_mask, tgt_key_padding_mask=ar_y_mask, memory_mask=None, memory_key_padding_mask=x_mask)
            logits = self.ar_predict_layer(y_dec).permute(0, 2, 1)
            total_loss = F.cross_entropy(logits, targets, reduction=reduction)
            metrics['ArTop10Accuracy'] = self.ar_accuracy_metric(logits.detach(), targets).item() * y_lens.sum().type(torch.float32)
        if self.num_quantizers == 1:
            return (x, codes), total_loss, metrics
        if self.ar_audio_prepend_bos:
            y = y[:, 1:]
        if train_stage in [0, 2]:
            num_nar_layers = self.num_quantizers - 1
            nar_stage = self.rng.choices([_k for _k in range(1, self.num_quantizers)], weights=[1.0 / num_nar_layers] * num_nar_layers, k=1)[0]
            x = self.nar_text_embedding(text)
            x = self.nar_text_prenet(x)
            x = self.nar_text_position(x)
            y_emb, prefix_len = self._prepare_prompts(y, y_lens, codes, nar_stage, y_prompts_codes)
            y_len = y_lens.max()
            targets = codes[..., nar_stage] + NUM_AUDIO_TOKENS * y_mask_int
            if self.prefix_mode in [2, 4]:
                targets = targets
                y_mask = F.pad(y_mask, (y_emb.shape[1] - y_len, 0), value=False)
            elif self.prefix_mode == 1:
                targets = targets[:, prefix_len:]
            else:
                assert prefix_len == 0
            y_pos = self.nar_audio_prenet(y_emb)
            y_pos = self.nar_audio_position(y_pos)
            y_dec, _ = self.nar_decoder((y_pos, self.nar_stage_embeddings[nar_stage - 1].weight), x, tgt_mask=None, tgt_key_padding_mask=y_mask, memory_mask=None, memory_key_padding_mask=x_mask)
            if self.prefix_mode != 0:
                y_dec = y_dec[:, prefix_len:]
                if self.prefix_mode == 4:
                    prefix_len = 0
            logits = self.nar_predict_layers[nar_stage - 1](y_dec).permute(0, 2, 1)
            total_length = y_lens.sum().type(torch.float32)
            total_loss += F.cross_entropy(logits, targets, ignore_index=NUM_AUDIO_TOKENS, reduction=reduction) * (total_length / (total_length - prefix_len * x.shape[0]))
            metrics['NarTop10Accuracy'] = self.nar_accuracy_metric(F.pad(logits.detach(), (0, 0, 0, 1, 0, 0), value=logits.min().cpu().item()), targets).item() * total_length
        if train_stage == 0:
            total_loss = total_loss / 2.0
        return (x, codes), total_loss, metrics

    def inference(self, x: 'torch.Tensor', x_lens: 'torch.Tensor', y: 'torch.Tensor', enroll_x_lens: 'Union[torch.Tensor, None]'=None, top_k: 'int'=-100, temperature: 'float'=1.0) ->torch.Tensor:
        """
        Args:
          x:
            A 2-D tensor of shape (1, S).
          x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (1, T, 8).
          top_k: (`optional`) int
            The number of highest probability tokens to keep for top-k-filtering. Default to -100.
          temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        Returns:
          Return the predicted audio code matrix and cross-entropy loss.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y.shape[0] == 1, y.shape
        assert torch.all(x_lens > 0)
        text = x
        x = self.ar_text_embedding(text)
        x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)
        x_mask = make_pad_mask(x_lens)
        prompts = y
        prefix_len = y.shape[1]
        y = prompts[..., 0]
        if self.ar_audio_prepend_bos:
            y = F.pad(y, (1, 0), value=NUM_AUDIO_TOKENS + 1)
        while True:
            y_emb = self.ar_audio_embedding(y)
            y_emb = self.ar_audio_prenet(y_emb)
            y_pos = self.ar_audio_position(y_emb)
            tgt_mask = torch.triu(torch.ones(y.shape[1], y.shape[1], device=y.device, dtype=torch.bool), diagonal=1)
            y_dec, _ = self.ar_decoder((y_pos, None), x, tgt_mask=tgt_mask, memory_mask=None, memory_key_padding_mask=x_mask)
            logits = self.ar_predict_layer(y_dec[:, -1])
            samples = topk_sampling(logits, top_k=top_k, top_p=1.0, temperature=temperature)
            if torch.argmax(logits, dim=-1)[0] == NUM_AUDIO_TOKENS or samples[0, 0] == NUM_AUDIO_TOKENS or y.shape[1] - prefix_len > x_lens.max() * 16:
                if prompts.shape[1] == y.shape[1]:
                    raise SyntaxError("well trained model shouldn't reach here.")
                None
                break
            y = torch.concat([y, samples], dim=1)
        codes = [y[:, prefix_len + int(self.ar_audio_prepend_bos):]]
        if self.num_quantizers == 1:
            return torch.stack(codes, dim=-1)
        y_emb = self.nar_audio_embeddings[0](y[:, int(self.ar_audio_prepend_bos):])
        if self.prefix_mode in [2, 4]:
            enrolled_len = enroll_x_lens.max().item()
            text = torch.concat([text[:, :1], text[:, enrolled_len - 1:]], dim=1)
            assert text.shape[0] == 1
        x = self.nar_text_embedding(text)
        x = self.nar_text_prenet(x)
        x = self.nar_text_position(x)
        if self.prefix_mode != 0:
            for j in range(1, self.num_quantizers):
                y_emb[:, :prefix_len] += self.nar_audio_embeddings[j](prompts[..., j])
        for i, (predict_layer, embedding_layer) in enumerate(zip(self.nar_predict_layers, self.nar_audio_embeddings[1:])):
            y_pos = self.nar_audio_prenet(y_emb)
            y_pos = self.nar_audio_position(y_pos)
            y_dec, _ = self.nar_decoder((y_pos, self.nar_stage_embeddings[i].weight), x, tgt_mask=None, memory_mask=None, memory_key_padding_mask=None)
            logits = predict_layer(y_dec[:, prefix_len:])
            samples = torch.argmax(logits, dim=-1)
            codes.append(samples)
            if i < 6:
                if self.prefix_mode == 0:
                    y_emb[:, :prefix_len] += embedding_layer(prompts[..., i + 1])
                y_emb[:, prefix_len:] += embedding_layer(samples)
        assert len(codes) == self.num_quantizers
        return torch.stack(codes, dim=-1)

    def visualize(self, predicts: 'Tuple[torch.Tensor]', batch: 'Dict[str, Union[List, torch.Tensor]]', output_dir: 'str', limit: 'int'=4) ->None:
        visualize(predicts, batch, output_dir, limit=limit)


class VALLE(VALLF):
    """It implements https://arxiv.org/abs/2301.02111
    "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers"
    """

    def __init__(self, d_model: 'int', nhead: 'int', num_layers: 'int', norm_first: 'bool'=True, add_prenet: 'bool'=False, prefix_mode: 'int'=0, share_embedding: 'bool'=True, nar_scale_factor: 'float'=1.0, **kwargs):
        """
        Args:
          d_model:
            The number of expected features in the input (required).
          nhead:
            The number of heads in the multiheadattention models (required).
          num_layers:
            The number of sub-decoder-layers in the decoder (required).
        """
        super(VALLE, self).__init__(d_model, nhead, num_layers, norm_first=norm_first, add_prenet=add_prenet, decoder_cls=TransformerEncoder, decoder_layer_cls=TransformerEncoderLayer, prefix_mode=prefix_mode, share_embedding=share_embedding, nar_scale_factor=nar_scale_factor, **kwargs)

    def forward(self, x: 'torch.Tensor', x_lens: 'torch.Tensor', y: 'Union[torch.Tensor, PromptedFeatures]', y_lens: 'Union[torch.Tensor, PromptedFeatures]', reduction: 'str'='sum', train_stage: 'int'=0, **kwargs) ->Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Args:
          x:
            A 2-D tensor of shape (N, S).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (N, T, 8).
          y_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          train_stage:
            0: AR & NAR modules, 1: AR modules, 2: NAR modules
        Returns:
          Return the predicted audio code matrix, cross-entropy loss and Top-10 accuracy.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        y_prompts_codes = None
        if isinstance(y, PromptedFeatures):
            y_prompts_codes, y = y.data
            prompts_len, y_lens = y_lens.data
            assert prompts_len.min() == prompts_len.max()
            assert self.prefix_mode == 4
            y_prompts_codes = y_prompts_codes.type(torch.int64)
        assert y.ndim == 3, y.shape
        assert y_lens.ndim == 1, y_lens.shape
        x_mask = make_pad_mask(x_lens)
        y_mask = make_pad_mask(y_lens)
        y_mask_int = y_mask.type(torch.int64)
        text = x
        codes = y.type(torch.int64) * (1 - y_mask_int.unsqueeze(dim=-1))
        y, targets = self.pad_y_eos(codes[..., 0], y_mask_int, eos_id=NUM_AUDIO_TOKENS)
        x_len = x_lens.max()
        metrics = {}
        total_loss = 0.0
        xy_padding_mask = torch.concat([x_mask, y_mask], dim=1)
        if self.ar_audio_prepend_bos:
            ar_xy_padding_mask = torch.concat([x_mask, F.pad(y_mask, (1, 0), value=False)], dim=1)
        else:
            ar_xy_padding_mask = xy_padding_mask
        if train_stage in [0, 1]:
            x = self.ar_text_embedding(text)
            x = self.ar_text_prenet(x)
            x = self.ar_text_position(x)
            y_len = y_lens.max() + int(self.ar_audio_prepend_bos)
            x_attn_mask = F.pad(torch.zeros((x_len, x_len), dtype=torch.bool, device=x.device), (0, y_len), value=True)
            y_attn_mask = F.pad(torch.triu(torch.ones(y_len, y_len, dtype=torch.bool, device=x.device), diagonal=1), (x_len, 0), value=False)
            xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)
            bsz, src_len = x.shape[0], x_len + y_len
            _xy_padding_mask = ar_xy_padding_mask.view(bsz, 1, 1, src_len).expand(-1, self.num_heads, -1, -1).reshape(bsz * self.num_heads, 1, src_len)
            xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)
            new_attn_mask = torch.zeros_like(xy_attn_mask, dtype=x.dtype)
            new_attn_mask.masked_fill_(xy_attn_mask, float('-inf'))
            xy_attn_mask = new_attn_mask
            y_emb = self.ar_audio_embedding(y)
            y_emb = self.ar_audio_prenet(y_emb)
            y_pos = self.ar_audio_position(y_emb)
            xy_pos = torch.concat([x, y_pos], dim=1)
            xy_dec, _ = self.ar_decoder((xy_pos, None), mask=xy_attn_mask)
            logits = self.ar_predict_layer(xy_dec[:, x_len:]).permute(0, 2, 1)
            total_loss = F.cross_entropy(logits, targets, reduction=reduction)
            metrics['ArTop10Accuracy'] = self.ar_accuracy_metric(logits.detach(), targets).item() * y_lens.sum().type(torch.float32)
        if self.num_quantizers == 1:
            return (x, codes), total_loss, metrics
        if self.ar_audio_prepend_bos:
            y = y[:, 1:]
        if train_stage in [0, 2]:
            num_nar_layers = self.num_quantizers - 1
            nar_stage = self.rng.choices([_k for _k in range(1, self.num_quantizers)], weights=[1.0 / num_nar_layers] * num_nar_layers, k=1)[0]
            x = self.nar_text_embedding(text)
            x = self.nar_text_prenet(x)
            x = self.nar_text_position(x)
            y_emb, prefix_len = self._prepare_prompts(y, y_lens, codes, nar_stage, y_prompts_codes)
            y_len = y_lens.max()
            targets = codes[..., nar_stage] + NUM_AUDIO_TOKENS * y_mask_int
            if self.prefix_mode in [2, 4]:
                xy_padding_mask = torch.concat([x_mask, F.pad(y_mask, (y_emb.shape[1] - y_len, 0), value=False)], dim=1)
            elif self.prefix_mode == 1:
                targets = targets[:, prefix_len:]
            y_pos = self.nar_audio_prenet(y_emb)
            y_pos = self.nar_audio_position(y_pos)
            xy_pos = torch.concat([x, y_pos], dim=1)
            xy_dec, _ = self.nar_decoder((xy_pos, self.nar_stage_embeddings[nar_stage - 1].weight), src_key_padding_mask=xy_padding_mask)
            xy_dec = xy_dec[:, x_lens.max() + prefix_len:]
            if self.prefix_mode == 4:
                prefix_len = 0
            logits = self.nar_predict_layers[nar_stage - 1](xy_dec).permute(0, 2, 1)
            total_length = y_lens.sum().type(torch.float32)
            total_loss += F.cross_entropy(logits, targets, ignore_index=NUM_AUDIO_TOKENS, reduction=reduction) * (total_length / (total_length - prefix_len * x.shape[0]))
            metrics['NarTop10Accuracy'] = self.nar_accuracy_metric(F.pad(logits.detach(), (0, 0, 0, 1, 0, 0), value=logits.min().cpu().item()), targets).item() * total_length
        if train_stage == 0:
            total_loss = total_loss / 2.0
        return (x, codes), total_loss, metrics

    def inference(self, x: 'torch.Tensor', x_lens: 'torch.Tensor', y: 'torch.Tensor', enroll_x_lens: 'torch.Tensor', top_k: 'int'=-100, temperature: 'float'=1.0) ->torch.Tensor:
        """
        Args:
          x:
            A 2-D tensor of shape (1, S).
          x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (1, T, 8).
          top_k: (`optional`) int
            The number of highest probability tokens to keep for top-k-filtering. Default to -100.
          temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        Returns:
          Return the predicted audio code matrix.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y.shape[0] == 1, y.shape
        assert torch.all(x_lens > 0)
        text = x
        x = self.ar_text_embedding(text)
        x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)
        text_len = x_lens.max()
        prompts = y
        prefix_len = y.shape[1]
        y = prompts[..., 0]
        if self.ar_audio_prepend_bos:
            y = F.pad(y, (1, 0), value=NUM_AUDIO_TOKENS + 1)
        x_len = x_lens.max()
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)
        while True:
            y_emb = self.ar_audio_embedding(y)
            y_emb = self.ar_audio_prenet(y_emb)
            y_pos = self.ar_audio_position(y_emb)
            xy_pos = torch.concat([x, y_pos], dim=1)
            y_len = y.shape[1]
            x_attn_mask_pad = F.pad(x_attn_mask, (0, y_len), value=True)
            y_attn_mask = F.pad(torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1), (x_len, 0), value=False)
            xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0)
            xy_dec, _ = self.ar_decoder((xy_pos, None), mask=xy_attn_mask)
            logits = self.ar_predict_layer(xy_dec[:, -1])
            samples = topk_sampling(logits, top_k=top_k, top_p=1.0, temperature=temperature)
            if torch.argmax(logits, dim=-1)[0] == NUM_AUDIO_TOKENS or samples[0, 0] == NUM_AUDIO_TOKENS or y.shape[1] - prompts.shape[1] > x_lens.max() * 16:
                if prompts.shape[1] == y.shape[1]:
                    raise SyntaxError("well trained model shouldn't reach here.")
                None
                break
            y = torch.concat([y, samples], dim=1)
        codes = [y[:, prefix_len + int(self.ar_audio_prepend_bos):]]
        if self.num_quantizers == 1:
            return torch.stack(codes, dim=-1)
        y_emb = self.nar_audio_embeddings[0](y[:, int(self.ar_audio_prepend_bos):])
        if self.prefix_mode in [2, 4]:
            enrolled_len = enroll_x_lens.max().item()
            text = torch.concat([text[:, :1], text[:, enrolled_len - 1:]], dim=1)
            text_len = text_len - (enrolled_len - 2)
            assert text.shape[0] == 1
        x = self.nar_text_embedding(text)
        x = self.nar_text_prenet(x)
        x = self.nar_text_position(x)
        if self.prefix_mode == 0:
            for i, (predict_layer, embedding_layer) in enumerate(zip(self.nar_predict_layers, self.nar_audio_embeddings[1:])):
                y_pos = self.nar_audio_prenet(y_emb)
                y_pos = self.nar_audio_position(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)
                xy_dec, _ = self.nar_decoder((xy_pos, self.nar_stage_embeddings[i].weight))
                logits = predict_layer(xy_dec[:, text_len + prefix_len:])
                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)
                if i < self.num_quantizers - 2:
                    y_emb[:, :prefix_len] += embedding_layer(prompts[..., i + 1])
                    y_emb[:, prefix_len:] += embedding_layer(samples)
        else:
            for j in range(1, self.num_quantizers):
                y_emb[:, :prefix_len] += self.nar_audio_embeddings[j](prompts[..., j])
            for i, (predict_layer, embedding_layer) in enumerate(zip(self.nar_predict_layers, self.nar_audio_embeddings[1:])):
                y_pos = self.nar_audio_prenet(y_emb)
                y_pos = self.nar_audio_position(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)
                xy_dec, _ = self.nar_decoder((xy_pos, self.nar_stage_embeddings[i].weight))
                logits = predict_layer(xy_dec[:, text_len + prefix_len:])
                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)
                if i < self.num_quantizers - 2:
                    y_emb[:, prefix_len:] += embedding_layer(samples)
        assert len(codes) == self.num_quantizers
        return torch.stack(codes, dim=-1)

    def continual(self, x: 'torch.Tensor', x_lens: 'torch.Tensor', y: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
          x:
            A 2-D tensor of shape (1, S).
          x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (1, T, 8).
        Returns:
          Return the predicted audio code matrix.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y.shape[0] == 1, y.shape
        assert torch.all(x_lens > 0)
        assert self.num_quantizers == 8
        text = x
        x = self.ar_text_embedding(text)
        x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)
        text_len = x_lens.max()
        prefix_len = min(int(y.shape[1] * 0.5), 3 * 75)
        prompts = y[:, :prefix_len]
        codes = [y[:, prefix_len:, 0]]
        x = self.nar_text_embedding(text)
        x = self.nar_text_prenet(x)
        x = self.nar_text_position(x)
        y_emb = self.nar_audio_embeddings[0](y[..., 0])
        if self.prefix_mode == 0:
            for i, (predict_layer, embedding_layer) in enumerate(zip(self.nar_predict_layers, self.nar_audio_embeddings[1:])):
                y_pos = self.nar_audio_position(y_emb)
                y_pos = self.nar_audio_prenet(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)
                xy_dec, _ = self.nar_decoder((xy_pos, self.nar_stage_embeddings[i].weight))
                logits = predict_layer(xy_dec[:, text_len + prefix_len:])
                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)
                if i < 6:
                    y_emb[:, :prefix_len] += embedding_layer(prompts[..., i + 1])
                    y_emb[:, prefix_len:] += embedding_layer(samples)
        else:
            for j in range(1, 8):
                y_emb[:, :prefix_len] += self.nar_audio_embeddings[j](prompts[..., j])
            for i, (predict_layer, embedding_layer) in enumerate(zip(self.nar_predict_layers, self.nar_audio_embeddings[1:])):
                y_pos = self.nar_audio_prenet(y_emb)
                y_pos = self.nar_audio_position(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)
                xy_dec, _ = self.nar_decoder((xy_pos, self.nar_stage_embeddings[i].weight))
                logits = predict_layer(xy_dec[:, text_len + prefix_len:])
                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)
                if i < 6:
                    y_emb[:, prefix_len:] += embedding_layer(samples)
        assert len(codes) == 8
        return torch.stack(codes, dim=-1)


def random_cast_to_half(x: 'Tensor', min_abs: 'float'=5e-06) ->Tensor:
    """
    A randomized way of casting a floating point value to half precision.
    """
    if x.dtype == torch.float16:
        return x
    x_abs = x.abs()
    is_too_small = x_abs < min_abs
    random_val = min_abs * x.sign() * (torch.rand_like(x) * min_abs < x_abs)
    return torch.where(is_too_small, random_val, x)


class RandomGradFunction(torch.autograd.Function):
    """
    Does nothing in forward pass; in backward pass, gets rid of very small grads using
    randomized approach that preserves expectations (intended to reduce roundoff).
    """

    @staticmethod
    def forward(ctx, x: 'Tensor', min_abs: 'float') ->Tensor:
        ctx.min_abs = min_abs
        return x

    @staticmethod
    def backward(ctx, ans_grad: 'Tensor') ->Tuple[Tensor, None]:
        if ans_grad.dtype == torch.float16:
            return random_cast_to_half(ans_grad, min_abs=ctx.min_abs), None
        else:
            return ans_grad, None


class RandomGrad(torch.nn.Module):
    """
    Gets rid of very small gradients using an expectation-preserving method, intended to increase
    accuracy of training when using amp (automatic mixed precision)
    """

    def __init__(self, min_abs: 'float'=5e-06):
        super(RandomGrad, self).__init__()
        self.min_abs = min_abs

    def forward(self, x: 'Tensor'):
        if torch.jit.is_scripting() or not self.training or torch.jit.is_tracing():
            return x
        else:
            return RandomGradFunction.apply(x, self.min_abs)


class SRLinear(nn.Linear):
    """https://arxiv.org/abs/2303.06296
    Stabilizing Transformer Training by Preventing Attention Entropy Collapse
    """

    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__(in_features, out_features, bias=bias, **kwargs)
        self.register_buffer('u', nn.functional.normalize(torch.randn(in_features), dim=0))
        with torch.no_grad():
            sigma = self.get_sigma()
        self.register_buffer('spectral_norm', sigma)
        self.sigma = nn.Parameter(torch.ones(1))

    def get_sigma(self):
        with torch.no_grad():
            u = self.u
            v = self.weight.mv(u)
            v = nn.functional.normalize(v, dim=0)
            u = self.weight.T.mv(v)
            u = nn.functional.normalize(u, dim=0)
            self.u.data.copy_(u)
        return torch.einsum('c,cd,d->', v, self.weight, u)

    def get_weight(self):
        sigma = self.get_sigma()
        if self.training:
            self.spectral_norm.data.copy_(sigma)
        weight = self.sigma / sigma * self.weight
        return weight

    def forward(self, x):
        return nn.functional.linear(x, self.get_weight(), self.bias)


class SRConv1d(SRLinear):

    def __init__(self, in_features, out_features, kernel_size, stride: 'int'=1, padding: 'str'='same', bias: 'bool'=True, **kwargs):
        in_features = in_features * kernel_size
        super().__init__(in_features, out_features, bias=bias, **kwargs)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        in_features = self.in_features // self.kernel_size
        weight = self.get_weight().view(self.out_features, in_features, self.kernel_size)
        return nn.functional.conv1d(x, weight, bias=self.bias, stride=self.stride, padding=self.padding)


def _diag(x: 'Tensor'):
    if x.ndim == 2:
        return x.diag()
    else:
        batch, dim, dim = x.shape
        x = x.reshape(batch, dim * dim)
        x = x[:, ::dim + 1]
        assert x.shape == (batch, dim)
        return x


def _whitening_metric(x: 'Tensor', num_groups: 'int'):
    """
    Computes the "whitening metric", a value which will be 1.0 if all the eigenvalues of
    of the centered feature covariance are the same within each group's covariance matrix
    and also between groups.
    Args:
        x: a Tensor of shape (*, num_channels)
     num_groups:  the number of groups of channels, a number >=1 that divides num_channels
    Returns:
        Returns a scalar Tensor that will be 1.0 if the data is "perfectly white" and
    greater than 1.0 otherwise.
    """
    assert x.dtype != torch.float16
    x = x.reshape(-1, x.shape[-1])
    num_frames, num_channels = x.shape
    assert num_channels % num_groups == 0
    channels_per_group = num_channels // num_groups
    x = x.reshape(num_frames, num_groups, channels_per_group).transpose(0, 1)
    x = x - x.mean(dim=1, keepdim=True)
    x_covar = torch.matmul(x.transpose(1, 2), x)
    x_covar_mean_diag = _diag(x_covar).mean()
    x_covarsq_mean_diag = (x_covar ** 2).sum() / (num_groups * channels_per_group)
    metric = x_covarsq_mean_diag / (x_covar_mean_diag ** 2 + 1e-20)
    return metric


class WhiteningPenaltyFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: 'Tensor', num_groups: 'int', whitening_limit: 'float', grad_scale: 'float') ->Tensor:
        ctx.save_for_backward(x)
        ctx.num_groups = num_groups
        ctx.whitening_limit = whitening_limit
        ctx.grad_scale = grad_scale
        return x

    @staticmethod
    def backward(ctx, x_grad: 'Tensor'):
        x_orig, = ctx.saved_tensors
        with torch.enable_grad():
            with torch.amp.autocast(enabled=False):
                x_detached = x_orig.detach()
                x_detached.requires_grad = True
                metric = _whitening_metric(x_detached, ctx.num_groups)
                if random.random() < 0.005 or __name__ == '__main__':
                    logging.info(f'Whitening: num_groups={ctx.num_groups}, num_channels={x_orig.shape[-1]}, metric={metric.item():.2f} vs. limit={ctx.whitening_limit}')
                (metric - ctx.whitening_limit).relu().backward()
                penalty_grad = x_detached.grad
                scale = ctx.grad_scale * (x_grad.norm() / (penalty_grad.norm() + 1e-20))
                penalty_grad = penalty_grad * scale
        return x_grad + penalty_grad, None, None, None


class Whiten(nn.Module):

    def __init__(self, num_groups: 'int', whitening_limit: 'float', prob: 'Union[float, Tuple[float, float]]', grad_scale: 'float'):
        """
        Args:
          num_groups: the number of groups to divide the channel dim into before
            whitening.  We will attempt to make the feature covariance
            within each group, after mean subtraction, as "white" as possible,
            while having the same trace across all groups.
         whitening_limit: a value greater than 1.0, that dictates how much
           freedom we have to violate the constraints.  1.0 would mean perfectly
           white, with exactly the same trace across groups; larger values
           give more freedom.  E.g. 2.0.
         prob: the probability with which we apply the gradient modification
           (also affects the grad scale).  May be supplied as a float,
           or as a pair (min_prob, max_prob)

          grad_scale: determines the scale on the gradient term from this object,
            relative to the rest of the gradient on the attention weights.
            E.g. 0.02 (you may want to use smaller values than this if prob is large)
        """
        super(Whiten, self).__init__()
        assert num_groups >= 1
        assert whitening_limit >= 1
        assert grad_scale >= 0
        self.num_groups = num_groups
        self.whitening_limit = whitening_limit
        if isinstance(prob, float):
            assert 0 < prob <= 1
            self.prob = prob
        else:
            self.min_prob, self.max_prob = prob
            assert 0 < self.min_prob < self.max_prob <= 1
            self.prob = self.max_prob
        self.grad_scale = grad_scale

    def forward(self, x: 'Tensor') ->Tensor:
        """
        In the forward pass, this function just returns the input unmodified.
        In the backward pass, it will modify the gradients to ensure that the
        distribution in each group has close to (lambda times I) as the covariance
        after mean subtraction, with the same lambda across groups.
        For whitening_limit > 1, there will be more freedom to violate this
        constraint.

        Args:
           x: the input of shape (*, num_channels)

        Returns:
            x, unmodified.   You should make sure
        you use the returned value, or the graph will be freed
        and nothing will happen in backprop.
        """
        if not x.requires_grad or random.random() > self.prob or self.grad_scale == 0:
            return _no_op(x)
        else:
            if hasattr(self, 'min_prob') and random.random() < 0.25:
                if _whitening_metric(x, self.num_groups) > self.whitening_limit:
                    self.prob = self.max_prob
                else:
                    self.prob = self.min_prob
            return WhiteningPenaltyFunction.apply(x, self.num_groups, self.whitening_limit, self.grad_scale)


class Identity(torch.nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return _no_op(x)


class MaxEigLimiterFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: 'Tensor', coeffs: 'Tensor', direction: 'Tensor', channel_dim: 'int', grad_scale: 'float') ->Tensor:
        ctx.channel_dim = channel_dim
        ctx.grad_scale = grad_scale
        ctx.save_for_backward(x.detach(), coeffs.detach(), direction.detach())
        return x

    @staticmethod
    def backward(ctx, x_grad, *args):
        with torch.enable_grad():
            x_orig, coeffs, new_direction = ctx.saved_tensors
            x_orig.requires_grad = True
            num_channels = x_orig.shape[ctx.channel_dim]
            x = x_orig.transpose(ctx.channel_dim, -1).reshape(-1, num_channels)
            new_direction.requires_grad = False
            x = x - x.mean(dim=0)
            x_var = (x ** 2).mean()
            x_residual = x - coeffs * new_direction
            x_residual_var = (x_residual ** 2).mean()
            variance_proportion = (x_var - x_residual_var) / (x_var + 1e-20)
            variance_proportion.backward()
        x_orig_grad = x_orig.grad
        x_extra_grad = x_orig.grad * ctx.grad_scale * x_grad.norm() / (x_orig_grad.norm() + 1e-20)
        return x_grad + x_extra_grad.detach(), None, None, None, None


class MaxEig(torch.nn.Module):
    """
    Modifies the backpropped derivatives of a function to try to discourage
    that any given direction in activation space accounts for more than
    a specified proportion of the covariance (e.g. 0.2).


    Args:
           num_channels: the number of channels
           channel_dim: the dimension/axis corresponding to the channel, e.g.
               -1, 0, 1, 2; will be interpreted as an offset from x.ndim if negative.
           max_var_per_eig:  the maximum proportion of the variance of the
               features/channels, after mean subtraction, that can come from
               any given eigenvalue.
           min_prob: the minimum probability with which we apply this during any invocation
               of forward(), assuming last time we applied the constraint it was
               not active; supplied for speed.
           scale: determines the scale with which we modify the gradients, relative
               to the existing / unmodified gradients
    """

    def __init__(self, num_channels: 'int', channel_dim: 'int', max_var_per_eig: 'float'=0.2, min_prob: 'float'=0.01, scale: 'float'=0.01):
        super(MaxEig, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.scale = scale
        assert max_var_per_eig == 0.0 or max_var_per_eig > 1.0 / num_channels
        self.max_var_per_eig = max_var_per_eig
        with torch.no_grad():
            direction = torch.arange(num_channels)
            direction = direction / direction.norm()
            self.register_buffer('max_eig_direction', direction)
        self.min_prob = min_prob
        self.cur_prob = 1.0

    def forward(self, x: 'Tensor') ->Tensor:
        if torch.jit.is_scripting() or self.max_var_per_eig <= 0 or random.random() > self.cur_prob or torch.jit.is_tracing():
            return _no_op(x)
        with torch.amp.autocast(enabled=False):
            eps = 1e-20
            orig_x = x
            x = x
            with torch.no_grad():
                x = x.transpose(self.channel_dim, -1).reshape(-1, self.num_channels)
                x = x - x.mean(dim=0)
                new_direction, coeffs = self._find_direction_coeffs(x, self.max_eig_direction)
                x_var = (x ** 2).mean()
                x_residual = x - coeffs * new_direction
                x_residual_var = (x_residual ** 2).mean()
                variance_proportion = (x_var - x_residual_var) / (x_var + 1e-20)
                self._set_direction(0.1 * self.max_eig_direction + new_direction)
            if random.random() < 0.01 or __name__ == '__main__':
                logging.info(f'variance_proportion = {variance_proportion.item()}, shape={tuple(orig_x.shape)}, cur_prob={self.cur_prob}')
            if variance_proportion >= self.max_var_per_eig:
                cur_prob = self.cur_prob
                self.cur_prob = 1.0
                return MaxEigLimiterFunction.apply(orig_x, coeffs, new_direction, self.channel_dim, self.scale)
            else:
                self.cur_prob = 0.75 * self.cur_prob + 0.25 * self.min_prob
                return orig_x

    def _set_direction(self, direction: 'Tensor'):
        """
        Sets self.max_eig_direction to a normalized version of `direction`
        """
        direction = direction.detach()
        direction = direction / direction.norm()
        direction_sum = direction.sum().item()
        if direction_sum - direction_sum == 0:
            self.max_eig_direction[:] = direction
        else:
            logging.info(f'Warning: sum of direction in MaxEig is {direction_sum}, num_channels={{self.num_channels}}, channel_dim={{self.channel_dim}}')

    def _find_direction_coeffs(self, x: 'Tensor', prev_direction: 'Tensor') ->Tuple[Tensor, Tensor, Tensor]:
        """
            Figure out (an approximation to) the proportion of the variance of a set of
            feature vectors that can be attributed to the top eigen-direction.
            Args:
             x: a Tensor of shape (num_frames, num_channels), with num_frames > 1.
          prev_direction:  a Tensor of shape (num_channels,), that is our previous estimate
                   of the top eigen-direction, or a random direction if this is the first
                   iteration.  Does not have to be normalized, but should be nonzero.

        Returns: (cur_direction, coeffs), where:
             cur_direction: a Tensor of shape (num_channels,) that is the current
                estimate of the top eigen-direction.
             coeffs: a Tensor of shape (num_frames, 1) that minimizes, or
                approximately minimizes, (x - coeffs * cur_direction).norm()
        """
        num_frames, num_channels = x.shape
        assert num_channels > 1 and num_frames > 1
        assert prev_direction.shape == (num_channels,)
        coeffs = (x * prev_direction).sum(dim=1, keepdim=True) + 1e-10
        cur_direction = (x * coeffs).sum(dim=0) / ((coeffs ** 2).sum() + 1e-20)
        return cur_direction, coeffs


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ActivationBalancer,
     lambda: ([], {'num_channels': 4, 'channel_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DoubleSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (IdentityNorm,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MultiheadAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (RandomGrad,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SRConv1d,
     lambda: ([], {'in_features': 4, 'out_features': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (SRLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SinePositionalEmbedding,
     lambda: ([], {'dim_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TransformerDecoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (TransformerEncoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (Transpose,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

