
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


import numpy as np


import re


import torch


from copy import deepcopy


import time


import torch.distributed as dist


import random


import warnings


from abc import abstractmethod


from enum import IntEnum


from enum import unique


from logging import warning


import math


from torch.optim import Optimizer


import copy


from matplotlib import pyplot as plt


from torch.utils.data import Dataset


from typing import Any


from typing import Dict


import torchvision


from abc import ABCMeta


from torch import nn as nn


import torch.nn as nn


from torch.nn import functional as F


from collections import OrderedDict


import torch.nn.functional as F


import torch.utils.checkpoint as cp


from torch import nn


from inspect import signature


from torch.nn.parameter import Parameter


from torch.nn import Linear


from torch.nn.init import xavier_uniform_


from torch.nn.init import constant_


from functools import partial


from torch.nn.functional import l1_loss


from torch.nn.functional import mse_loss


from torch.nn.functional import smooth_l1_loss


from torch.cuda.amp import custom_bwd


from torch.cuda.amp import custom_fwd


from torch.autograd.function import Function


from torch.autograd.function import once_differentiable


from torch._C import _infer_size


from torch._C import _add_docstr


from torch.nn import _reduction as _Reduction


from torch.nn.modules import utils


from torch.nn.modules.utils import _single


from torch.nn.modules.utils import _pair


from torch.nn.modules.utils import _triple


from torch.nn.modules.utils import _list_with_default


from torch.nn import grad


from torch._jit_internal import boolean_dispatch


from torch._jit_internal import List


from torch._jit_internal import Optional


from torch._jit_internal import _overload


from torch._jit_internal import Tuple


from torch.nn.functional import linear


from torch.nn.functional import softmax


from torch.nn.functional import dropout


from torch.utils.checkpoint import checkpoint


from typing import Sequence


from torch.nn.init import normal_


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.autograd import Function


from torch.nn import init


from typing import List


from typing import Tuple


from torch import distributed as dist


import itertools


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import CppExtension


from torch.autograd import gradcheck


from torch.nn import BatchNorm1d


from torch.nn import ReLU


def _in_projection(q: 'Tensor', k: 'Tensor', v: 'Tensor', w_q: 'Tensor', w_k: 'Tensor', w_v: 'Tensor', b_q: 'Optional[Tensor]'=None, b_k: 'Optional[Tensor]'=None, b_v: 'Optional[Tensor]'=None) ->Tuple[Tensor, Tensor, Tensor]:
    """
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.
    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`
        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`
    """
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    assert w_q.shape == (Eq, Eq), f'expecting query weights shape of {Eq, Eq}, but got {w_q.shape}'
    assert w_k.shape == (Eq, Ek), f'expecting key weights shape of {Eq, Ek}, but got {w_k.shape}'
    assert w_v.shape == (Eq, Ev), f'expecting value weights shape of {Eq, Ev}, but got {w_v.shape}'
    assert b_q is None or b_q.shape == (Eq,), f'expecting query bias shape of {Eq,}, but got {b_q.shape}'
    assert b_k is None or b_k.shape == (Eq,), f'expecting key bias shape of {Eq,}, but got {b_k.shape}'
    assert b_v is None or b_v.shape == (Eq,), f'expecting value bias shape of {Eq,}, but got {b_v.shape}'
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _in_projection_packed(q: 'Tensor', k: 'Tensor', v: 'Tensor', w: 'Tensor', b: 'Optional[Tensor]'=None) ->List[Tensor]:
    """
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.
    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension
        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            return linear(q, w, b).chunk(3, dim=-1)
        else:
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _scaled_cosine_attention(q: 'Tensor', k: 'Tensor', v: 'Tensor', tau, tau_min, num_heads, attn_mask: 'Optional[Tensor]'=None, extra_attn: 'Optional[Tensor]'=None, dropout_p: 'float'=0.0) ->Tuple[Tensor, Tensor]:
    """
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.
    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.
    Shape:
        - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.
        - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
            have shape :math:`(B, Nt, Ns)`
    """
    B, Nt, E = q.shape
    Ns = k.shape[1]
    if tau is not None:
        q = nn.functional.normalize(q, dim=2)
        k = nn.functional.normalize(k, dim=2)
        attn = torch.bmm(q, k.transpose(-2, -1))
        if tau.ndim == 4:
            assert tau.size(1) == num_heads and attn.size(-1) == Ns
            attn = attn.reshape(B // num_heads, num_heads, Nt, Ns)
            attn = attn / tau.clamp(min=tau_min)
            attn = attn.reshape(B, Nt, Ns)
        else:
            attn = attn / tau.clamp(min=tau_min)
    else:
        q = q / math.sqrt(E)
        attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask
    if extra_attn is not None:
        assert extra_attn.shape == attn.shape, f'{extra_attn.shape} v.s. {attn.shape}'
        attn += extra_attn
    attn = softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)
    output = torch.bmm(attn, v)
    return output, attn


def cosine_multi_head_attention_forward(query: 'Tensor', key: 'Tensor', value: 'Tensor', embed_dim_to_check: 'int', num_heads: 'int', in_proj_weight: 'Tensor', in_proj_bias: 'Optional[Tensor]', bias_k: 'Optional[Tensor]', bias_v: 'Optional[Tensor]', add_zero_attn: 'bool', dropout_p: 'float', out_proj_weight: 'Tensor', out_proj_bias: 'Optional[Tensor]', training: 'bool'=True, key_padding_mask: 'Optional[Tensor]'=None, need_weights: 'bool'=True, attn_mask: 'Optional[Tensor]'=None, use_separate_proj_weight: 'bool'=False, q_proj_weight: 'Optional[Tensor]'=None, k_proj_weight: 'Optional[Tensor]'=None, v_proj_weight: 'Optional[Tensor]'=None, static_k: 'Optional[Tensor]'=None, static_v: 'Optional[Tensor]'=None, extra_attn: 'Optional[Tensor]'=None, tau=None, tau_min=None) ->Tuple[Tensor, Optional[Tensor]]:
    """
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check, f'was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}'
    if isinstance(embed_dim, torch.Tensor):
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f'embed_dim {embed_dim} not divisible by num_heads {num_heads}'
    if use_separate_proj_weight:
        assert key.shape[:2] == value.shape[:2], f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f'key shape {key.shape} does not match value shape {value.shape}'
    if not use_separate_proj_weight:
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, 'use_separate_proj_weight is True but q_proj_weight is None'
        assert k_proj_weight is not None, 'use_separate_proj_weight is True but k_proj_weight is None'
        assert v_proj_weight is not None, 'use_separate_proj_weight is True but v_proj_weight is None'
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn('Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.')
            attn_mask = attn_mask
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, f'Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}'
        if attn_mask.dim() == 2:
            correct_2d_size = tgt_len, src_len
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f'The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.')
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = bsz * num_heads, tgt_len, src_len
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f'The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.')
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn('Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.')
        key_padding_mask = key_padding_mask
    if bias_k is not None and bias_v is not None:
        assert static_k is None, 'bias cannot be added to static key.'
        assert static_v is None, 'bias cannot be added to static value.'
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    else:
        assert static_k.size(0) == bsz * num_heads, f'expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}'
        assert static_k.size(2) == head_dim, f'expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}'
        k = static_k
    if static_v is None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    else:
        assert static_v.size(0) == bsz * num_heads, f'expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}'
        assert static_v.size(2) == head_dim, f'expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}'
        v = static_v
    if add_zero_attn:
        zero_attn_shape = bsz * num_heads, 1, head_dim
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    src_len = k.size(1)
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), f'expecting key_padding_mask shape of {bsz, src_len}, but got {key_padding_mask.shape}'
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float('-inf'))
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float('-inf'))
        attn_mask = new_attn_mask
    if not training:
        dropout_p = 0.0
    attn_output, attn_output_weights = _scaled_cosine_attention(q, k, v, tau, tau_min, num_heads, attn_mask, extra_attn, dropout_p)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    if need_weights:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class CosineMultiheadAttention(nn.MultiheadAttention):
    """Inherit from standard multihead attention, call the customized multi_head_forward function in forward.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None, cosine=True, tau_min=0.01, non_shared_tau=False) ->None:
        super(CosineMultiheadAttention, self).__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim)
        self.batch_first = batch_first
        self.tau_min = tau_min
        if cosine:
            if non_shared_tau:
                self.tau = torch.nn.Parameter(torch.ones(1, num_heads, 1, 1))
            else:
                self.tau = torch.nn.Parameter(torch.ones(1, 1, 1))
        else:
            self.tau = None

    def forward(self, query: 'Tensor', key: 'Tensor', value: 'Tensor', key_padding_mask: 'Optional[Tensor]'=None, need_weights: 'bool'=True, attn_mask: 'Optional[Tensor]'=None, extra_attn=None) ->Tuple[Tensor, Optional[Tensor]]:
        """
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
    Shapes for inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
          source sequence length.
          If a 3D mask: :math:`(N\\cdot\\text{num\\_heads}, L, S)` where N is the batch size, L is the target sequence
          length, S is the source sequence length. ``attn_mask`` ensure that position i is allowed to attend
          the unmasked positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
    Shapes for outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = cosine_multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, use_separate_proj_weight=True, q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight, v_proj_weight=self.v_proj_weight, extra_attn=extra_attn, tau=self.tau, tau_min=self.tau_min)
        else:
            attn_output, attn_output_weights = cosine_multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, extra_attn=extra_attn, tau=self.tau, tau_min=self.tau_min)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


def flat2window(feat, voxel_drop_lvl, flat2win_inds_dict, drop_info):
    """
    Args:
        feat: shape=[N, C], N is the voxel num in the batch.
        voxel_drop_lvl: shape=[N, ]. Indicates drop_level of the window the voxel belongs to.
    Returns:
        feat_3d_dict: contains feat_3d of each drop level. Shape of feat_3d is [num_windows, num_max_tokens, C].
    
    drop_info:
    {1:{'max_tokens':50, 'range':(0, 50)}, }
    """
    dtype = feat.dtype
    device = feat.device
    feat_dim = feat.shape[-1]
    feat_3d_dict = {}
    for dl in drop_info:
        dl_mask = voxel_drop_lvl == dl
        if not dl_mask.any():
            continue
        feat_this_dl = feat[dl_mask]
        this_inds = flat2win_inds_dict[dl][0]
        max_tokens = drop_info[dl]['max_tokens']
        num_windows = (this_inds // max_tokens).max().item() + 1
        feat_3d = torch.zeros((num_windows * max_tokens, feat_dim), dtype=dtype, device=device)
        if this_inds.max() >= num_windows * max_tokens:
            set_trace()
        feat_3d[this_inds] = feat_this_dl
        feat_3d = feat_3d.reshape((num_windows, max_tokens, feat_dim))
        feat_3d_dict[dl] = feat_3d
    return feat_3d_dict


def flat2window_v2(feat, inds_dict):
    assert 'voxel_drop_level' in inds_dict, 'voxel_drop_level should be in inds_dict in v2 function'
    inds_v1 = {k: inds_dict[k] for k in inds_dict if not isinstance(k, str)}
    batching_info = inds_dict['batching_info']
    return flat2window(feat, inds_dict['voxel_drop_level'], inds_v1, batching_info)


def window2flat(feat_3d_dict, inds_dict):
    flat_feat_list = []
    num_all_voxel = 0
    for dl in inds_dict:
        num_all_voxel += inds_dict[dl][0].shape[0]
    dtype = feat_3d_dict[list(feat_3d_dict.keys())[0]].dtype
    device = feat_3d_dict[list(feat_3d_dict.keys())[0]].device
    feat_dim = feat_3d_dict[list(feat_3d_dict.keys())[0]].shape[-1]
    all_flat_feat = torch.zeros((num_all_voxel, feat_dim), device=device, dtype=dtype)
    check_feat = -torch.ones((num_all_voxel,), device=device, dtype=torch.long)
    for dl in feat_3d_dict:
        feat = feat_3d_dict[dl]
        feat_dim = feat.shape[-1]
        inds, flat_pos = inds_dict[dl]
        feat = feat.reshape(-1, feat_dim)
        flat_feat = feat[inds]
        all_flat_feat[flat_pos] = flat_feat
        check_feat[flat_pos] = 0
    assert (check_feat == 0).all()
    return all_flat_feat


def window2flat_v2(feat_3d_dict, inds_dict):
    inds_v1 = {k: inds_dict[k] for k in inds_dict if not isinstance(k, str)}
    return window2flat(feat_3d_dict, inds_v1)


class WindowAttention(nn.Module):

    def __init__(self, d_model, nhead, dropout, batch_first=False, layer_id=None, layer_cfg=dict()):
        super().__init__()
        self.nhead = nhead
        if layer_cfg.get('cosine', False):
            tau_min = layer_cfg.get('tau_min', 0.01)
            self.self_attn = CosineMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False, tau_min=tau_min, cosine=True, non_shared_tau=layer_cfg.get('non_shared_tau', False))
        elif layer_cfg.get('linear', False):
            raise NotImplementedError
            self.self_attn = LinearMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.exe_counter = 0
        self.layer_id = layer_id

    def forward(self, feat_2d, pos_dict, ind_dict, key_padding_dict):
        """
        Args:

        Out:
            shifted_feat_dict: the same type as window_feat_dict
        """
        out_feat_dict = {}
        feat_3d_dict = flat2window_v2(feat_2d, ind_dict)
        for name in feat_3d_dict:
            pos = pos_dict[name]
            feat_3d = feat_3d_dict[name]
            feat_3d = feat_3d.permute(1, 0, 2)
            v = feat_3d
            if pos is not None:
                pos = pos.permute(1, 0, 2)
                assert pos.shape == feat_3d.shape, f'pos_shape: {pos.shape}, feat_shape:{feat_3d.shape}'
                q = k = feat_3d + pos
            else:
                q = k = feat_3d
            key_padding_mask = key_padding_dict[name]
            out_feat_3d, attn_map = self.self_attn(q, k, value=v, key_padding_mask=key_padding_mask)
            out_feat_dict[name] = out_feat_3d.permute(1, 0, 2)
        results = window2flat_v2(out_feat_dict, ind_dict)
        return results


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == 'relu':
        return torch.nn.functional.relu
    if activation == 'gelu':
        return torch.nn.functional.gelu
    if activation == 'glu':
        return torch.nn.functional.glu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')


class BasicShiftBlockV2(nn.Module):
    """ Consist of two encoder layer, shift and shift back.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', batch_first=False, block_id=-100, layer_cfg=dict()):
        super().__init__()
        encoder_1 = EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first, layer_id=block_id * 2 + 0, layer_cfg=layer_cfg)
        encoder_2 = EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first, layer_id=block_id * 2 + 1, layer_cfg=layer_cfg)
        self.encoder_list = nn.ModuleList([encoder_1, encoder_2])

    def forward(self, src, pos_dict_list, ind_dict_list, key_mask_dict_list, using_checkpoint=False):
        num_shifts = len(pos_dict_list)
        assert num_shifts in (1, 2)
        output = src
        for i in range(2):
            this_id = i % num_shifts
            pos_dict = pos_dict_list[this_id]
            ind_dict = ind_dict_list[this_id]
            key_mask_dict = key_mask_dict_list[this_id]
            layer = self.encoder_list[i]
            if using_checkpoint and self.training:
                output = checkpoint(layer, output, pos_dict, ind_dict, key_mask_dict)
            else:
                output = layer(output, pos_dict, ind_dict, key_mask_dict)
        return output


class SSTv2(nn.Module):
    """
    Single-stride Sparse Transformer. 
    Main args:
        d_model (list[int]): the number of filters in first linear layer of each transformer encoder
        dim_feedforward list([int]): the number of filters in first linear layer of each transformer encoder
        output_shape (tuple[int, int]): shape of output bev feature.
        num_attached_conv: the number of convolutions in the end of SST for filling the "empty hold" in BEV feature map.
        conv_kwargs: key arguments of each attached convolution.
        checckpoint_blocks: block IDs (0 to num_blocks - 1) to use checkpoint.
        Note: In PyTorch 1.8, checkpoint function seems not able to receive dict as parameters. Better to use PyTorch >= 1.9.
    """

    def __init__(self, d_model=[], nhead=[], num_blocks=6, dim_feedforward=[], dropout=0.0, activation='gelu', output_shape=None, debug=True, in_channel=None, checkpoint_blocks=[], layer_cfg=dict()):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.checkpoint_blocks = checkpoint_blocks
        if in_channel is not None:
            self.linear0 = nn.Linear(in_channel, d_model[0])
        block_list = []
        for i in range(num_blocks):
            block_list.append(BasicShiftBlockV2(d_model[i], nhead[i], dim_feedforward[i], dropout, activation, batch_first=False, block_id=i, layer_cfg=layer_cfg))
        self.block_list = nn.ModuleList(block_list)
        self._reset_parameters()
        self.output_shape = output_shape
        self.debug = debug

    def forward(self, voxel_info, **kwargs):
        """
        """
        num_shifts = 2
        assert voxel_info['voxel_coors'].dtype == torch.int64, 'data type of coors should be torch.int64!'
        batch_size = voxel_info['voxel_coors'][:, 0].max().item() + 1
        voxel_feat = voxel_info['voxel_feats']
        ind_dict_list = [voxel_info[f'flat2win_inds_shift{i}'] for i in range(num_shifts)]
        padding_mask_list = [voxel_info[f'key_mask_shift{i}'] for i in range(num_shifts)]
        pos_embed_list = [voxel_info[f'pos_dict_shift{i}'] for i in range(num_shifts)]
        output = voxel_feat
        if hasattr(self, 'linear0'):
            output = self.linear0(output)
        for i, block in enumerate(self.block_list):
            output = block(output, pos_embed_list, ind_dict_list, padding_mask_list, using_checkpoint=i in self.checkpoint_blocks)
        output = self.recover_bev(output, voxel_info['voxel_coors'], batch_size)
        output_list = []
        output_list.append(output)
        return output_list

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name and 'tau' not in name:
                nn.init.xavier_uniform_(p)

    def recover_bev(self, voxel_feat, coors, batch_size):
        """
        Args:
            voxel_feat: shape=[N, C]
            coors: [N, 4]
        Return:
            batch_canvas:, shape=[B, C, ny, nx]
        """
        ny, nx = self.output_shape
        feat_dim = voxel_feat.shape[-1]
        batch_canvas = []
        for batch_itt in range(batch_size):
            canvas = torch.zeros(feat_dim, nx * ny, dtype=voxel_feat.dtype, device=voxel_feat.device)
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * nx + this_coors[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_feat[batch_mask, :]
            voxels = voxels.t()
            canvas[:, indices] = voxels
            batch_canvas.append(canvas)
        batch_canvas = torch.stack(batch_canvas, 0)
        batch_canvas = batch_canvas.view(batch_size, feat_dim, ny, nx)
        return batch_canvas


class GatherPoints(Function):
    """Gather Points.

    Gather points with given index.
    """

    @staticmethod
    def forward(ctx, features: 'torch.Tensor', indices: 'torch.Tensor') ->torch.Tensor:
        """forward.

        Args:
            features (Tensor): (B, C, N) features to gather.
            indices (Tensor): (B, M) where M is the number of points.

        Returns:
            Tensor: (B, C, M) where M is the number of points.
        """
        assert features.is_contiguous()
        assert indices.is_contiguous()
        B, npoint = indices.size()
        _, C, N = features.size()
        output = torch.FloatTensor(B, C, npoint)
        gather_points_ext.gather_points_wrapper(B, C, N, npoint, features, indices, output)
        ctx.for_backwards = indices, C, N
        ctx.mark_non_differentiable(indices)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()
        grad_features = torch.FloatTensor(B, C, N).zero_()
        grad_out_data = grad_out.data.contiguous()
        gather_points_ext.gather_points_grad_wrapper(B, C, N, npoint, grad_out_data, idx, grad_features.data)
        return grad_features, None


gather_points = GatherPoints.apply


class GeneralSamplingModule(nn.Module):
    """Sampling Points.

    Sampling points with given index.
    """

    def forward(self, xyz, features, sample_inds):
        """Forward pass.

        Args:
            xyz： (B, N, 3) the coordinates of the features.
            features (Tensor): (B, C, N) features to sample.
            sample_inds (Tensor): (B, M) the given index,
                where M is the number of points.

        Returns:
            Tensor: (B, M, 3) coordinates of sampled features
            Tensor: (B, C, M) the sampled features.
            Tensor: (B, M) the given index.
        """
        xyz_t = xyz.transpose(1, 2).contiguous()
        new_xyz = gather_points(xyz_t, sample_inds).transpose(1, 2).contiguous()
        new_features = gather_points(features, sample_inds).contiguous()
        return new_xyz, new_features, sample_inds


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(nn.Conv1d(input_channel, num_pos_feats, kernel_size=1), nn.BatchNorm1d(num_pos_feats), nn.ReLU(inplace=True), nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


def multi_head_attention_forward(query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias, training=True, key_padding_mask=None, need_weights=True, attn_mask=None, use_separate_proj_weight=False, q_proj_weight=None, k_proj_weight=None, v_proj_weight=None, static_k=None, static_v=None):
    """
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in differnt forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    qkv_same = torch.equal(query, key) and torch.equal(key, value)
    kv_same = torch.equal(key, value)
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert list(query.size()) == [tgt_len, bsz, embed_dim]
    assert key.size() == value.size()
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, 'embed_dim must be divisible by num_heads'
    scaling = float(head_dim) ** -0.5
    if use_separate_proj_weight is not True:
        if qkv_same:
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
        elif kv_same:
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)
            if key is None:
                assert value is None
                k = None
                v = None
            else:
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)
        else:
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)
        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)
        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)
        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:embed_dim * 2])
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[embed_dim * 2:])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling
    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1), dtype=attn_mask.dtype, device=attn_mask.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask, torch.zeros((key_padding_mask.size(0), 1), dtype=key_padding_mask.dtype, device=key_padding_mask.device)], dim=1)
        else:
            assert static_k is None, 'bias cannot be added to static key.'
            assert static_v is None, 'bias cannot be added to static value.'
    else:
        assert bias_k is None
        assert bias_v is None
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k
    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v
    src_len = k.size(1)
    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len
    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1), dtype=attn_mask.dtype, device=attn_mask.device)], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat([key_padding_mask, torch.zeros((key_padding_mask.size(0), 1), dtype=key_padding_mask.dtype, device=key_padding_mask.device)], dim=1)
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]
    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)
        attn_output_weights += attn_mask
    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)
    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    if need_weights:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class MultiheadAttention(nn.Module):
    """Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    .. math::
        \\text{MultiHead}(Q, K, V) = \\text{Concat}(head_1,\\dots,head_h)W^O
        \\text{where} head_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

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

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        """
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if hasattr(self, '_qkv_same_embed_dim') and self._qkv_same_embed_dim is False:
            return multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, use_separate_proj_weight=True, q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight, v_proj_weight=self.v_proj_weight)
        else:
            if not hasattr(self, '_qkv_same_embed_dim'):
                warnings.warn('A new version of MultiheadAttention module has been implemented.                     Please re-train your model with the new module', UserWarning)
            return multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', self_posembed=None, cross_posembed=None, cross_only=False):
        super().__init__()
        self.cross_only = cross_only
        if not self.cross_only:
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        def _get_activation_fn(activation):
            """Return an activation function given a string"""
            if activation == 'relu':
                return F.relu
            if activation == 'gelu':
                return F.gelu
            if activation == 'glu':
                return F.glu
            raise RuntimeError(f'activation should be relu/gelu, not {activation}.')
        self.activation = _get_activation_fn(activation)
        self.self_posembed = self_posembed
        self.cross_posembed = cross_posembed

    def with_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, query, key, query_pos, key_pos, attn_mask=None):
        """
        :param query: B C Pq
        :param key: B C Pk
        :param query_pos: B Pq 3/6
        :param key_pos: B Pk 3/6
        :param value_pos: [B Pq 3/6]
        :return:
        """
        if self.self_posembed is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        if self.cross_posembed is not None:
            key_pos_embed = self.cross_posembed(key_pos).permute(2, 0, 1)
        else:
            key_pos_embed = None
        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)
        if not self.cross_only:
            q = k = v = self.with_pos_embed(query, query_pos_embed)
            query2 = self.self_attn(q, k, value=v)[0]
            query = query + self.dropout1(query2)
            query = self.norm1(query)
        query2 = self.multihead_attn(query=self.with_pos_embed(query, query_pos_embed), key=self.with_pos_embed(key, key_pos_embed), value=self.with_pos_embed(key, key_pos_embed), attn_mask=attn_mask)[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)
        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)
        query = query.permute(1, 2, 0)
        return query


class FFN(nn.Module):

    def __init__(self, in_channels, heads, head_conv=64, final_kernel=1, init_bias=-2.19, conv_cfg=dict(type='Conv1d'), norm_cfg=dict(type='BN1d'), bias='auto', **kwargs):
        super(FFN, self).__init__()
        self.heads = heads
        self.init_bias = init_bias
        for head in self.heads:
            classes, num_conv = self.heads[head]
            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                conv_layers.append(ConvModule(c_in, head_conv, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=bias, conv_cfg=conv_cfg, norm_cfg=norm_cfg))
                c_in = head_conv
            conv_layers.append(build_conv_layer(conv_cfg, head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
            conv_layers = nn.Sequential(*conv_layers)
            self.__setattr__(head, conv_layers)

    def init_weights(self):
        """Initialize weights."""
        for head in self.heads:
            if head == 'heatmap':
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)
            else:
                for m in self.__getattr__(head).modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)

    def forward(self, x):
        """Forward function for SepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            dict[str: torch.Tensor]: contains the following keys:

                -reg （torch.Tensor): 2D regression value with the                     shape of [B, 2, H, W].
                -height (torch.Tensor): Height value with the                     shape of [B, 1, H, W].
                -dim (torch.Tensor): Size value with the shape                     of [B, 3, H, W].
                -rot (torch.Tensor): Rotation value with the                     shape of [B, 1, H, W].
                -vel (torch.Tensor): Velocity value with the                     shape of [B, 2, H, W].
                -heatmap (torch.Tensor): Heatmap with the shape of                     [B, N, H, W].
        """
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)
        return ret_dict


def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.

    Args:
        val (torch.Tensor): The value to be converted.
        offset (float, optional): Offset to set the value range.             Defaults to 0.5.
        period ([type], optional): Period of the value. Defaults to np.pi.

    Returns:
        torch.Tensor: Value in the range of             [-offset * period, (1-offset) * period]
    """
    return val - torch.floor(val / period + offset) * period


def xywhr2xyxyr(boxes_xywhr):
    """Convert a rotated boxes in XYWHR format to XYXYR format.

    Args:
        boxes_xywhr (torch.Tensor): Rotated boxes in XYWHR format.

    Returns:
        torch.Tensor: Converted boxes in XYXYR format.
    """
    boxes = torch.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[:, 2] / 2
    half_h = boxes_xywhr[:, 3] / 2
    boxes[:, 0] = boxes_xywhr[:, 0] - half_w
    boxes[:, 1] = boxes_xywhr[:, 1] - half_h
    boxes[:, 2] = boxes_xywhr[:, 0] + half_w
    boxes[:, 3] = boxes_xywhr[:, 1] + half_h
    boxes[:, 4] = boxes_xywhr[:, 4]
    return boxes


class BaseInstance3DBoxes(object):
    """Base class for 3D Boxes.

    Note:
        The box is bottom centered, i.e. the relative position of origin in
        the box is (0.5, 0.5, 0).

    Args:
        tensor (torch.Tensor | np.ndarray | list): a N x box_dim matrix.
        box_dim (int): Number of the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw).
            Default to 7.
        with_yaw (bool): Whether the box is with yaw rotation.
            If False, the value of yaw will be set to 0 as minmax boxes.
            Default to True.
        origin (tuple[float]): The relative position of origin in the box.
            Default to (0.5, 0.5, 0). This will guide the box be converted to
            (0.5, 0.5, 0) mode.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x box_dim.
        box_dim (int): Integer indicating the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    def __init__(self, tensor, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0)):
        if isinstance(tensor, torch.Tensor):
            device = tensor.device
        else:
            device = torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = tensor.reshape((0, box_dim))
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, tensor.size()
        if tensor.shape[-1] == 6:
            assert box_dim == 6
            fake_rot = tensor.new_zeros(tensor.shape[0], 1)
            tensor = torch.cat((tensor, fake_rot), dim=-1)
            self.box_dim = box_dim + 1
            self.with_yaw = False
        else:
            self.box_dim = box_dim
            self.with_yaw = with_yaw
        self.tensor = tensor.clone()
        if origin != (0.5, 0.5, 0):
            dst = self.tensor.new_tensor((0.5, 0.5, 0))
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    @property
    def volume(self):
        """torch.Tensor: A vector with volume of each box."""
        return self.tensor[:, 3] * self.tensor[:, 4] * self.tensor[:, 5]

    @property
    def dims(self):
        """torch.Tensor: Corners of each box with size (N, 8, 3)."""
        return self.tensor[:, 3:6]

    @property
    def yaw(self):
        """torch.Tensor: A vector with yaw of each box."""
        return self.tensor[:, 6]

    @property
    def height(self):
        """torch.Tensor: A vector with height of each box."""
        return self.tensor[:, 5]

    @property
    def top_height(self):
        """torch.Tensor: A vector with the top height of each box."""
        return self.bottom_height + self.height

    @property
    def bottom_height(self):
        """torch.Tensor: A vector with bottom's height of each box."""
        return self.tensor[:, 2]

    @property
    def center(self):
        """Calculate the center of all the boxes.

        Note:
            In the MMDetection3D's convention, the bottom center is
            usually taken as the default center.

            The relative position of the centers in different kinds of
            boxes are different, e.g., the relative center of a boxes is
            (0.5, 1.0, 0.5) in camera and (0.5, 0.5, 0) in lidar.
            It is recommended to use ``bottom_center`` or ``gravity_center``
            for more clear usage.

        Returns:
            torch.Tensor: A tensor with center of each box.
        """
        return self.bottom_center

    @property
    def bottom_center(self):
        """torch.Tensor: A tensor with center of each box."""
        return self.tensor[:, :3]

    @property
    def gravity_center(self):
        """torch.Tensor: A tensor with center of each box."""
        pass

    @property
    def corners(self):
        """torch.Tensor: a tensor with 8 corners of each box."""
        pass

    @abstractmethod
    def rotate(self, angle, points=None):
        """Rotate boxes with points (optional) with the given angle or         rotation matrix.

        Args:
            angle (float | torch.Tensor | np.ndarray):
                Rotation angle or rotation matrix.
            points (torch.Tensor, numpy.ndarray, :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.
        """
        pass

    @abstractmethod
    def flip(self, bev_direction='horizontal'):
        """Flip the boxes in BEV along given BEV direction."""
        pass

    def translate(self, trans_vector):
        """Translate boxes with the given translation vector.

        Args:
            trans_vector (torch.Tensor): Translation vector of size 1x3.
        """
        if not isinstance(trans_vector, torch.Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        self.tensor[:, :3] += trans_vector

    def in_range_3d(self, box_range):
        """Check whether the boxes are in the given range.

        Args:
            box_range (list | torch.Tensor): The range of box
                (x_min, y_min, z_min, x_max, y_max, z_max)

        Note:
            In the original implementation of SECOND, checking whether
            a box in the range checks whether the points are in a convex
            polygon, we try to reduce the burden for simpler cases.

        Returns:
            torch.Tensor: A binary vector indicating whether each box is                 inside the reference range.
        """
        in_range_flags = (self.tensor[:, 0] > box_range[0]) & (self.tensor[:, 1] > box_range[1]) & (self.tensor[:, 2] > box_range[2]) & (self.tensor[:, 0] < box_range[3]) & (self.tensor[:, 1] < box_range[4]) & (self.tensor[:, 2] < box_range[5])
        return in_range_flags

    @abstractmethod
    def in_range_bev(self, box_range):
        """Check whether the boxes are in the given range.

        Args:
            box_range (list | torch.Tensor): The range of box
                in order of (x_min, y_min, x_max, y_max).

        Returns:
            torch.Tensor: Indicating whether each box is inside                 the reference range.
        """
        pass

    @abstractmethod
    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`Box3DMode`): The target Box mode.
            rt_mat (np.ndarray | torch.Tensor): The rotation and translation
                matrix between different coordinates. Defaults to None.
                The conversion from `src` coordinates to `dst` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`BaseInstance3DBoxes`: The converted box of the same type                 in the `dst` mode.
        """
        pass

    def scale(self, scale_factor):
        """Scale the box with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the boxes.
        """
        self.tensor[:, :6] *= scale_factor
        self.tensor[:, 7:] *= scale_factor

    def limit_yaw(self, offset=0.5, period=np.pi):
        """Limit the yaw to a given period and offset.

        Args:
            offset (float): The offset of the yaw.
            period (float): The expected period.
        """
        self.tensor[:, 6] = limit_period(self.tensor[:, 6], offset, period)

    def nonempty(self, threshold: 'float'=0.0):
        """Find boxes that are non-empty.

        A box is considered empty,
        if either of its side is no larger than threshold.

        Args:
            threshold (float): The threshold of minimal sizes.

        Returns:
            torch.Tensor: A binary vector which represents whether each                 box is empty (False) or non-empty (True).
        """
        box = self.tensor
        size_x = box[..., 3]
        size_y = box[..., 4]
        size_z = box[..., 5]
        keep = (size_x > threshold) & (size_y > threshold) & (size_z > threshold)
        return keep

    def __getitem__(self, item):
        """
        Note:
            The following usage are allowed:
            1. `new_boxes = boxes[3]`:
                return a `Boxes` that contains only one box.
            2. `new_boxes = boxes[2:10]`:
                return a slice of boxes.
            3. `new_boxes = boxes[vector]`:
                where vector is a torch.BoolTensor with `length = len(boxes)`.
                Nonzero elements in the vector will be selected.
            Note that the returned Boxes might share storage with this Boxes,
            subject to Pytorch's indexing semantics.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new object of                  :class:`BaseInstances3DBoxes` after indexing.
        """
        original_type = type(self)
        if isinstance(item, int):
            return original_type(self.tensor[item].view(1, -1), box_dim=self.box_dim, with_yaw=self.with_yaw)
        b = self.tensor[item]
        assert b.dim() == 2, f'Indexing on Boxes with {item} failed to return a matrix!'
        return original_type(b, box_dim=self.box_dim, with_yaw=self.with_yaw)

    def __len__(self):
        """int: Number of boxes in the current object."""
        return self.tensor.shape[0]

    def __repr__(self):
        """str: Return a strings that describes the object."""
        return self.__class__.__name__ + '(\n    ' + str(self.tensor) + ')'

    @classmethod
    def cat(cls, boxes_list):
        """Concatenate a list of Boxes into a single Boxes.

        Args:
            boxes_list (list[:obj:`BaseInstance3DBoxes`]): List of boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: The concatenated Boxes.
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(box, cls) for box in boxes_list)
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0), box_dim=boxes_list[0].tensor.shape[1], with_yaw=boxes_list[0].with_yaw)
        return cat_boxes

    def to(self, device):
        """Convert current boxes to a specific device.

        Args:
            device (str | :obj:`torch.device`): The name of the device.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new boxes object on the                 specific device.
        """
        original_type = type(self)
        return original_type(self.tensor, box_dim=self.box_dim, with_yaw=self.with_yaw)

    def clone(self):
        """Clone the Boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: Box object with the same properties                 as self.
        """
        original_type = type(self)
        return original_type(self.tensor.clone(), box_dim=self.box_dim, with_yaw=self.with_yaw)

    @property
    def device(self):
        """str: The device of the boxes are on."""
        return self.tensor.device

    def __iter__(self):
        """Yield a box as a Tensor of shape (4,) at a time.

        Returns:
            torch.Tensor: A box of shape (4,).
        """
        yield from self.tensor

    @classmethod
    def height_overlaps(cls, boxes1, boxes2, mode='iou'):
        """Calculate height overlaps of two boxes.

        Note:
            This function calculates the height overlaps between boxes1 and
            boxes2,  boxes1 and boxes2 should be in the same type.

        Args:
            boxes1 (:obj:`BaseInstance3DBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`BaseInstance3DBoxes`): Boxes 2 contain M boxes.
            mode (str, optional): Mode of iou calculation. Defaults to 'iou'.

        Returns:
            torch.Tensor: Calculated iou of boxes.
        """
        assert isinstance(boxes1, BaseInstance3DBoxes)
        assert isinstance(boxes2, BaseInstance3DBoxes)
        assert type(boxes1) == type(boxes2), f'"boxes1" and "boxes2" shouldbe in the same type, got {type(boxes1)} and {type(boxes2)}.'
        boxes1_top_height = boxes1.top_height.view(-1, 1)
        boxes1_bottom_height = boxes1.bottom_height.view(-1, 1)
        boxes2_top_height = boxes2.top_height.view(1, -1)
        boxes2_bottom_height = boxes2.bottom_height.view(1, -1)
        heighest_of_bottom = torch.max(boxes1_bottom_height, boxes2_bottom_height)
        lowest_of_top = torch.min(boxes1_top_height, boxes2_top_height)
        overlaps_h = torch.clamp(lowest_of_top - heighest_of_bottom, min=0)
        return overlaps_h

    @classmethod
    def overlaps(cls, boxes1, boxes2, mode='iou'):
        """Calculate 3D overlaps of two boxes.

        Note:
            This function calculates the overlaps between ``boxes1`` and
            ``boxes2``, ``boxes1`` and ``boxes2`` should be in the same type.

        Args:
            boxes1 (:obj:`BaseInstance3DBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`BaseInstance3DBoxes`): Boxes 2 contain M boxes.
            mode (str, optional): Mode of iou calculation. Defaults to 'iou'.

        Returns:
            torch.Tensor: Calculated iou of boxes' heights.
        """
        assert isinstance(boxes1, BaseInstance3DBoxes)
        assert isinstance(boxes2, BaseInstance3DBoxes)
        assert type(boxes1) == type(boxes2), f'"boxes1" and "boxes2" shouldbe in the same type, got {type(boxes1)} and {type(boxes2)}.'
        assert mode in ['iou', 'iof']
        rows = len(boxes1)
        cols = len(boxes2)
        if rows * cols == 0:
            return boxes1.tensor.new(rows, cols)
        overlaps_h = cls.height_overlaps(boxes1, boxes2)
        boxes1_bev = xywhr2xyxyr(boxes1.bev)
        boxes2_bev = xywhr2xyxyr(boxes2.bev)
        overlaps_bev = boxes1_bev.new_zeros((boxes1_bev.shape[0], boxes2_bev.shape[0]))
        iou3d_cuda.boxes_overlap_bev_gpu(boxes1_bev.contiguous(), boxes2_bev.contiguous(), overlaps_bev)
        overlaps_3d = overlaps_bev * overlaps_h
        volume1 = boxes1.volume.view(-1, 1)
        volume2 = boxes2.volume.view(1, -1)
        if mode == 'iou':
            iou3d = overlaps_3d / torch.clamp(volume1 + volume2 - overlaps_3d, min=1e-08)
        else:
            iou3d = overlaps_3d / torch.clamp(volume1, min=1e-08)
        return iou3d

    def new_box(self, data):
        """Create a new box object with data.

        The new box and its tensor has the similar properties             as self and self.tensor, respectively.

        Args:
            data (torch.Tensor | numpy.array | list): Data to be copied.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new bbox object with ``data``,                 the object's other properties are similar to ``self``.
        """
        new_tensor = self.tensor.new_tensor(data) if not isinstance(data, torch.Tensor) else data
        original_type = type(self)
        return original_type(new_tensor, box_dim=self.box_dim, with_yaw=self.with_yaw)


class BasePoints(object):
    """Base class for Points.

    Args:
        tensor (torch.Tensor | np.ndarray | list): a N x points_dim matrix.
        points_dim (int): Number of the dimension of a point.
            Each row is (x, y, z). Default to 3.
        attribute_dims (dict): Dictionary to indicate the meaning of extra
            dimension. Default to None.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x points_dim.
        points_dim (int): Integer indicating the dimension of a point.
            Each row is (x, y, z, ...).
        attribute_dims (bool): Dictionary to indicate the meaning of extra
            dimension. Default to None.
        rotation_axis (int): Default rotation axis for points rotation.
    """

    def __init__(self, tensor, points_dim=3, attribute_dims=None):
        if isinstance(tensor, torch.Tensor):
            device = tensor.device
        else:
            device = torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = tensor.reshape((0, points_dim))
        assert tensor.dim() == 2 and tensor.size(-1) == points_dim, tensor.size()
        self.tensor = tensor
        self.points_dim = points_dim
        self.attribute_dims = attribute_dims
        self.rotation_axis = 0

    @property
    def coord(self):
        """torch.Tensor: Coordinates of each point with size (N, 3)."""
        return self.tensor[:, :3]

    @coord.setter
    def coord(self, tensor):
        """Set the coordinates of each point."""
        try:
            tensor = tensor.reshape(self.shape[0], 3)
        except (RuntimeError, ValueError):
            raise ValueError(f'got unexpected shape {tensor.shape}')
        if not isinstance(tensor, torch.Tensor):
            tensor = self.tensor.new_tensor(tensor)
        self.tensor[:, :3] = tensor

    @property
    def height(self):
        """torch.Tensor: A vector with height of each point."""
        if self.attribute_dims is not None and 'height' in self.attribute_dims.keys():
            return self.tensor[:, self.attribute_dims['height']]
        else:
            return None

    @height.setter
    def height(self, tensor):
        """Set the height of each point."""
        try:
            tensor = tensor.reshape(self.shape[0])
        except (RuntimeError, ValueError):
            raise ValueError(f'got unexpected shape {tensor.shape}')
        if not isinstance(tensor, torch.Tensor):
            tensor = self.tensor.new_tensor(tensor)
        if self.attribute_dims is not None and 'height' in self.attribute_dims.keys():
            self.tensor[:, self.attribute_dims['height']] = tensor
        else:
            if self.attribute_dims is None:
                self.attribute_dims = dict()
            attr_dim = self.shape[1]
            self.tensor = torch.cat([self.tensor, tensor.unsqueeze(1)], dim=1)
            self.attribute_dims.update(dict(height=attr_dim))
            self.points_dim += 1

    @property
    def color(self):
        """torch.Tensor: A vector with color of each point."""
        if self.attribute_dims is not None and 'color' in self.attribute_dims.keys():
            return self.tensor[:, self.attribute_dims['color']]
        else:
            return None

    @color.setter
    def color(self, tensor):
        """Set the color of each point."""
        try:
            tensor = tensor.reshape(self.shape[0], 3)
        except (RuntimeError, ValueError):
            raise ValueError(f'got unexpected shape {tensor.shape}')
        if tensor.max() >= 256 or tensor.min() < 0:
            warnings.warn('point got color value beyond [0, 255]')
        if not isinstance(tensor, torch.Tensor):
            tensor = self.tensor.new_tensor(tensor)
        if self.attribute_dims is not None and 'color' in self.attribute_dims.keys():
            self.tensor[:, self.attribute_dims['color']] = tensor
        else:
            if self.attribute_dims is None:
                self.attribute_dims = dict()
            attr_dim = self.shape[1]
            self.tensor = torch.cat([self.tensor, tensor], dim=1)
            self.attribute_dims.update(dict(color=[attr_dim, attr_dim + 1, attr_dim + 2]))
            self.points_dim += 3

    @property
    def shape(self):
        """torch.Shape: Shape of points."""
        return self.tensor.shape

    def shuffle(self):
        """Shuffle the points.

        Returns:
            torch.Tensor: The shuffled index.
        """
        idx = torch.randperm(self.__len__(), device=self.tensor.device)
        self.tensor = self.tensor[idx]
        return idx

    def rotate(self, rotation, axis=None):
        """Rotate points with the given rotation matrix or angle.

        Args:
            rotation (float, np.ndarray, torch.Tensor): Rotation matrix
                or angle.
            axis (int): Axis to rotate at. Defaults to None.
        """
        if not isinstance(rotation, torch.Tensor):
            rotation = self.tensor.new_tensor(rotation)
        assert rotation.shape == torch.Size([3, 3]) or rotation.numel() == 1, f'invalid rotation shape {rotation.shape}'
        if axis is None:
            axis = self.rotation_axis
        if rotation.numel() == 1:
            rot_sin = torch.sin(rotation)
            rot_cos = torch.cos(rotation)
            if axis == 1:
                rot_mat_T = rotation.new_tensor([[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]])
            elif axis == 2 or axis == -1:
                rot_mat_T = rotation.new_tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
            elif axis == 0:
                rot_mat_T = rotation.new_tensor([[0, rot_cos, -rot_sin], [0, rot_sin, rot_cos], [1, 0, 0]])
            else:
                raise ValueError('axis should in range')
            rot_mat_T = rot_mat_T.T
        elif rotation.numel() == 9:
            rot_mat_T = rotation
        else:
            raise NotImplementedError
        self.tensor[:, :3] = self.tensor[:, :3] @ rot_mat_T
        return rot_mat_T

    @abstractmethod
    def flip(self, bev_direction='horizontal'):
        """Flip the points in BEV along given BEV direction."""
        pass

    def translate(self, trans_vector):
        """Translate points with the given translation vector.

        Args:
            trans_vector (np.ndarray, torch.Tensor): Translation
                vector of size 3 or nx3.
        """
        if not isinstance(trans_vector, torch.Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        trans_vector = trans_vector.squeeze(0)
        if trans_vector.dim() == 1:
            assert trans_vector.shape[0] == 3
        elif trans_vector.dim() == 2:
            assert trans_vector.shape[0] == self.tensor.shape[0] and trans_vector.shape[1] == 3
        else:
            raise NotImplementedError(f'Unsupported translation vector of shape {trans_vector.shape}')
        self.tensor[:, :3] += trans_vector

    def in_range_3d(self, point_range):
        """Check whether the points are in the given range.

        Args:
            point_range (list | torch.Tensor): The range of point
                (x_min, y_min, z_min, x_max, y_max, z_max)

        Note:
            In the original implementation of SECOND, checking whether
            a box in the range checks whether the points are in a convex
            polygon, we try to reduce the burden for simpler cases.

        Returns:
            torch.Tensor: A binary vector indicating whether each point is                 inside the reference range.
        """
        in_range_flags = (self.tensor[:, 0] > point_range[0]) & (self.tensor[:, 1] > point_range[1]) & (self.tensor[:, 2] > point_range[2]) & (self.tensor[:, 0] < point_range[3]) & (self.tensor[:, 1] < point_range[4]) & (self.tensor[:, 2] < point_range[5])
        return in_range_flags

    @abstractmethod
    def in_range_bev(self, point_range):
        """Check whether the points are in the given range.

        Args:
            point_range (list | torch.Tensor): The range of point
                in order of (x_min, y_min, x_max, y_max).

        Returns:
            torch.Tensor: Indicating whether each point is inside                 the reference range.
        """
        pass

    @abstractmethod
    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`CoordMode`): The target Box mode.
            rt_mat (np.ndarray | torch.Tensor): The rotation and translation
                matrix between different coordinates. Defaults to None.
                The conversion from `src` coordinates to `dst` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`BasePoints`: The converted box of the same type                 in the `dst` mode.
        """
        pass

    def scale(self, scale_factor):
        """Scale the points with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the points.
        """
        self.tensor[:, :3] *= scale_factor

    def __getitem__(self, item):
        """
        Note:
            The following usage are allowed:
            1. `new_points = points[3]`:
                return a `Points` that contains only one point.
            2. `new_points = points[2:10]`:
                return a slice of points.
            3. `new_points = points[vector]`:
                where vector is a torch.BoolTensor with `length = len(points)`.
                Nonzero elements in the vector will be selected.
            4. `new_points = points[3:11, vector]`:
                return a slice of points and attribute dims.
            5. `new_points = points[4:12, 2]`:
                return a slice of points with single attribute.
            Note that the returned Points might share storage with this Points,
            subject to Pytorch's indexing semantics.

        Returns:
            :obj:`BasePoints`: A new object of                  :class:`BasePoints` after indexing.
        """
        original_type = type(self)
        if isinstance(item, int):
            return original_type(self.tensor[item].view(1, -1), points_dim=self.points_dim, attribute_dims=self.attribute_dims)
        elif isinstance(item, tuple) and len(item) == 2:
            if isinstance(item[1], slice):
                start = 0 if item[1].start is None else item[1].start
                stop = self.tensor.shape[1] if item[1].stop is None else item[1].stop
                step = 1 if item[1].step is None else item[1].step
                item = list(item)
                item[1] = list(range(start, stop, step))
                item = tuple(item)
            elif isinstance(item[1], int):
                item = list(item)
                item[1] = [item[1]]
                item = tuple(item)
            p = self.tensor[item[0], item[1]]
            keep_dims = list(set(item[1]).intersection(set(range(3, self.tensor.shape[1]))))
            if self.attribute_dims is not None:
                attribute_dims = self.attribute_dims.copy()
                for key in self.attribute_dims.keys():
                    cur_attribute_dims = attribute_dims[key]
                    if isinstance(cur_attribute_dims, int):
                        cur_attribute_dims = [cur_attribute_dims]
                    intersect_attr = list(set(cur_attribute_dims).intersection(set(keep_dims)))
                    if len(intersect_attr) == 1:
                        attribute_dims[key] = intersect_attr[0]
                    elif len(intersect_attr) > 1:
                        attribute_dims[key] = intersect_attr
                    else:
                        attribute_dims.pop(key)
            else:
                attribute_dims = None
        elif isinstance(item, (slice, np.ndarray, torch.Tensor)):
            p = self.tensor[item]
            attribute_dims = self.attribute_dims
        else:
            raise NotImplementedError(f'Invalid slice {item}!')
        assert p.dim() == 2, f'Indexing on Points with {item} failed to return a matrix!'
        return original_type(p, points_dim=p.shape[1], attribute_dims=attribute_dims)

    def __len__(self):
        """int: Number of points in the current object."""
        return self.tensor.shape[0]

    def __repr__(self):
        """str: Return a strings that describes the object."""
        return self.__class__.__name__ + '(\n    ' + str(self.tensor) + ')'

    @classmethod
    def cat(cls, points_list):
        """Concatenate a list of Points into a single Points.

        Args:
            points_list (list[:obj:`BasePoints`]): List of points.

        Returns:
            :obj:`BasePoints`: The concatenated Points.
        """
        assert isinstance(points_list, (list, tuple))
        if len(points_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(points, cls) for points in points_list)
        cat_points = cls(torch.cat([p.tensor for p in points_list], dim=0), points_dim=points_list[0].tensor.shape[1], attribute_dims=points_list[0].attribute_dims)
        return cat_points

    def to(self, device):
        """Convert current points to a specific device.

        Args:
            device (str | :obj:`torch.device`): The name of the device.

        Returns:
            :obj:`BasePoints`: A new boxes object on the                 specific device.
        """
        original_type = type(self)
        return original_type(self.tensor, points_dim=self.points_dim, attribute_dims=self.attribute_dims)

    def clone(self):
        """Clone the Points.

        Returns:
            :obj:`BasePoints`: Box object with the same properties                 as self.
        """
        original_type = type(self)
        return original_type(self.tensor.clone(), points_dim=self.points_dim, attribute_dims=self.attribute_dims)

    @property
    def device(self):
        """str: The device of the points are on."""
        return self.tensor.device

    def __iter__(self):
        """Yield a point as a Tensor of shape (4,) at a time.

        Returns:
            torch.Tensor: A point of shape (4,).
        """
        yield from self.tensor

    def new_point(self, data):
        """Create a new point object with data.

        The new point and its tensor has the similar properties             as self and self.tensor, respectively.

        Args:
            data (torch.Tensor | numpy.array | list): Data to be copied.

        Returns:
            :obj:`BasePoints`: A new point object with ``data``,                 the object's other properties are similar to ``self``.
        """
        new_tensor = self.tensor.new_tensor(data) if not isinstance(data, torch.Tensor) else data
        original_type = type(self)
        return original_type(new_tensor, points_dim=self.points_dim, attribute_dims=self.attribute_dims)

