
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


from copy import copy


from collections import deque


from copy import deepcopy


import torchvision.transforms.functional as F


import copy


import logging


import re


from typing import Dict


from typing import List


import torch.nn as nn


from collections import defaultdict


from typing import Any


from typing import Iterable


from typing import NamedTuple


from typing import Optional


from typing import Tuple


from torch.nn.parallel import DistributedDataParallel


import numpy as np


from torch.nn import functional as F


import warnings


from torchvision.ops.boxes import box_area


from functools import partial


from torch import nn


import torch.nn.functional as F


from torch.autograd import Function


from torch.autograd.function import once_differentiable


from torch.cuda.amp import custom_bwd


from torch.cuda.amp import custom_fwd


from torch.nn.init import xavier_uniform_


from torch.nn.init import constant_


import math


from torch import Tensor


from torch import Size


from typing import Union


from torch.nn.parameter import Parameter


import numbers


from scipy import interpolate


from math import pi


import torch.utils.checkpoint as checkpoint


import torchvision


from collections import OrderedDict


from torchvision.models._utils import IntermediateLayerGetter


import itertools


from scipy.optimize import linear_sum_assignment


import torch.distributed as dist


import torch.nn.init as init


from torch.nn.modules import Module


from typing import Callable


import random


import torch.utils.data as torchdata


from random import choice


from random import randint


from torch.utils.data.sampler import Sampler


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import torchvision.transforms.functional as TransF


import torchvision.transforms as T


from torch.nn.init import uniform_


from torch.nn.init import normal_


import time


from torch.nn.parallel import DataParallel


from torch.cuda.amp import autocast


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.cpp_extension import ROCM_HOME


from torch.autograd import gradcheck


from torch.nn.functional import dropout


from torch.nn.functional import linear


from torch.nn.functional import pad


from torch.nn.functional import softmax


from torch.nn.modules.module import Module


import uuid


def multi_head_attention_forward(query: 'Tensor', key: 'Tensor', value: 'Tensor', embed_dim_to_check: 'int', num_heads: 'int', in_proj_weight: 'Tensor', in_proj_bias: 'Tensor', bias_k: 'Optional[Tensor]', bias_v: 'Optional[Tensor]', add_zero_attn: 'bool', dropout_p: 'float', out_proj_weight: 'Tensor', out_proj_bias: 'Tensor', training: 'bool'=True, key_padding_mask: 'Optional[Tensor]'=None, need_weights: 'bool'=True, attn_mask: 'Optional[Tensor]'=None, use_separate_proj_weight: 'bool'=False, q_proj_weight: 'Optional[Tensor]'=None, k_proj_weight: 'Optional[Tensor]'=None, v_proj_weight: 'Optional[Tensor]'=None, static_k: 'Optional[Tensor]'=None, static_v: 'Optional[Tensor]'=None, out_dim: 'Optional[Tensor]'=None) ->Tuple[Tensor, Optional[Tensor]]:
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
    if not torch.jit.is_scripting():
        tens_ops = query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias
        if any([(type(t) is not Tensor) for t in tens_ops]) and has_torch_function(tens_ops):
            return handle_torch_function(multi_head_attention_forward, tens_ops, query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias, training=training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, use_separate_proj_weight=use_separate_proj_weight, q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight, v_proj_weight=v_proj_weight, static_k=static_k, static_v=static_v)
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)
    head_dim = embed_dim // num_heads
    v_head_dim = out_dim // num_heads
    assert head_dim * num_heads == embed_dim, 'embed_dim must be divisible by num_heads'
    scaling = float(head_dim) ** -0.5
    q = query * scaling
    k = key
    v = value
    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, 'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn('Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.')
            attn_mask = attn_mask
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn('Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.')
        key_padding_mask = key_padding_mask
    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
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
        v = v.contiguous().view(-1, bsz * num_heads, v_head_dim).transpose(0, 1)
    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k
    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == v_head_dim
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
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'))
        else:
            attn_output_weights += attn_mask
    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)
    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)
    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, v_head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, out_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    if need_weights:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class MultiheadAttention(Module):
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
        vdim: total number of features in value. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    bias_k: 'Optional[torch.Tensor]'
    bias_v: 'Optional[torch.Tensor]'

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
        self.out_proj = _LinearWithBias(vdim, vdim)
        self.in_proj_bias = None
        self.in_proj_weight = None
        self.bias_k = self.bias_v = None
        self.q_proj_weight = None
        self.k_proj_weight = None
        self.v_proj_weight = None
        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.out_proj.bias, 0.0)

    def __setstate__(self, state):
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True
        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
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
            attn_mask: 2D or 3D mask that prevents attention to certain positions.
                A 2D mask will be broadcasted for all the batches while a 3D mask allows
                to specify a different mask for the entries of each batch.
        Shape:
            - Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
              the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
              If a ByteTensor is provided, the non-zero positions will be ignored while the position
              with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
              value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
              3D mask :math:`(N*\\text{num_heads}, L, S)` where N is the batch size, L is the target sequence length,
              S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
              positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
              while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
              is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
              is provided, it will be added to the attention weight.
            - Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
              L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, use_separate_proj_weight=True, q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight, v_proj_weight=self.v_proj_weight, out_dim=self.vdim)
        else:
            return multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, out_dim=self.vdim)


class ConditionalSelfAttention(nn.Module):
    """Conditional Self-Attention Module used in Conditional-DETR

    `Conditional DETR for Fast Training Convergence.
    <https://arxiv.org/pdf/2108.06152.pdf>`_


    Args:
        embed_dim (int): The embedding dimension for attention.
        num_heads (int): The number of attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `MultiheadAttention`.
            Default: 0.0.
        batch_first (bool): if `True`, then the input and output tensor will be
            provided as `(bs, n, embed_dim)`. Default: False. `(n, bs, embed_dim)`
    """

    def __init__(self, embed_dim, num_heads, attn_drop=0.0, proj_drop=0.0, batch_first=False, **kwargs):
        super(ConditionalSelfAttention, self).__init__()
        self.query_content_proj = nn.Linear(embed_dim, embed_dim)
        self.query_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.key_content_proj = nn.Linear(embed_dim, embed_dim)
        self.key_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5
        self.batch_first = batch_first

    def forward(self, query, key=None, value=None, identity=None, query_pos=None, key_pos=None, attn_mask=None, key_padding_mask=None, **kwargs):
        """Forward function for `ConditionalSelfAttention`

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_query, embed_dim)`
            key (torch.Tensor): Key embeddings with shape
                `(num_key, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_key, embed_dim)`
            value (torch.Tensor): Value embeddings with the same shape as `key`.
                Same in `torch.nn.MultiheadAttention.forward`. Default: None.
                If None, the `key` will be used.
            identity (torch.Tensor): The tensor, with the same shape as `query``,
                which will be used for identity addition. Default: None.
                If None, `query` will be used.
            query_pos (torch.Tensor): The position embedding for query, with the
                same shape as `query`. Default: None.
            key_pos (torch.Tensor): The position embedding for key. Default: None.
                If None, and `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`.
            attn_mask (torch.Tensor): ByteTensor mask with shape `(num_query, num_key)`.
                Same as `torch.nn.MultiheadAttention.forward`. Default: None.
            key_padding_mask (torch.Tensor): ByteTensor with shape `(bs, num_key)` which
                indicates which elements within `key` to be ignored in attention.
                Default: None.
        """
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key ismissing in {self.__class__.__name__}.')
        assert query_pos is not None and key_pos is not None, 'query_pos and key_pos must be passed into ConditionalAttention Module'
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            query_pos = query_pos.transpose(0, 1)
            key_pos = key_pos.transpose(0, 1)
            identity = identity.transpose(0, 1)
        query_content = self.query_content_proj(query)
        query_pos = self.query_pos_proj(query_pos)
        key_content = self.key_content_proj(key)
        key_pos = self.key_pos_proj(key_pos)
        value = self.value_proj(value)
        N, B, C = query_content.shape
        q = query_content + query_pos
        k = key_content + key_pos
        v = value
        q = q.reshape(N, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        k = k.reshape(N, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        v = v.reshape(N, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn.masked_fill_(attn_mask, float('-inf'))
            else:
                attn += attn_mask
        if key_padding_mask is not None:
            attn = attn.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        if not self.batch_first:
            out = out.transpose(0, 1)
        return identity + self.proj_drop(out)


class ConditionalCrossAttention(nn.Module):
    """Conditional Cross-Attention Module used in Conditional-DETR

    `Conditional DETR for Fast Training Convergence.
    <https://arxiv.org/pdf/2108.06152.pdf>`_


    Args:
        embed_dim (int): The embedding dimension for attention.
        num_heads (int): The number of attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `MultiheadAttention`.
            Default: 0.0.
        batch_first (bool): if `True`, then the input and output tensor will be
            provided as `(bs, n, embed_dim)`. Default: False. `(n, bs, embed_dim)`
    """

    def __init__(self, embed_dim, num_heads, attn_drop=0.0, proj_drop=0.0, batch_first=False, **kwargs):
        super(ConditionalCrossAttention, self).__init__()
        self.query_content_proj = nn.Linear(embed_dim, embed_dim)
        self.query_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.query_pos_sine_proj = nn.Linear(embed_dim, embed_dim)
        self.key_content_proj = nn.Linear(embed_dim, embed_dim)
        self.key_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.num_heads = num_heads
        self.batch_first = batch_first

    def forward(self, query, key=None, value=None, identity=None, query_pos=None, key_pos=None, query_sine_embed=None, is_first_layer=False, attn_mask=None, key_padding_mask=None, **kwargs):
        """Forward function for `ConditionalCrossAttention`

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_query, embed_dim)`
            key (torch.Tensor): Key embeddings with shape
                `(num_key, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_key, embed_dim)`
            value (torch.Tensor): Value embeddings with the same shape as `key`.
                Same in `torch.nn.MultiheadAttention.forward`. Default: None.
                If None, the `key` will be used.
            identity (torch.Tensor): The tensor, with the same shape as x, will
                be used for identity addition. Default: None.
                If None, `query` will be used.
            query_pos (torch.Tensor): The position embedding for query, with the
                same shape as `query`. Default: None.
            key_pos (torch.Tensor): The position embedding for key. Default: None.
                If None, and `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`.
            query_sine_embed (torch.Tensor): None
            is_first_layer (bool): None
            attn_mask (torch.Tensor): ByteTensor mask with shape `(num_query, num_key)`.
                Same as `torch.nn.MultiheadAttention.forward`. Default: None.
            key_padding_mask (torch.Tensor): ByteTensor with shape `(bs, num_key)` which
                indicates which elements within `key` to be ignored in attention.
                Default: None.
        """
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key ismissing in {self.__class__.__name__}.')
        assert query_pos is not None and key_pos is not None, 'query_pos and key_pos must be passed into ConditionalAttention Module'
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            query_pos = query_pos.transpose(0, 1)
            key_pos = key_pos.transpose(0, 1)
            identity = identity.transpose(0, 1)
        query_content = self.query_content_proj(query)
        key_content = self.key_content_proj(key)
        value = self.value_proj(value)
        N, B, C = query_content.shape
        HW, _, _ = key_content.shape
        key_pos = self.key_pos_proj(key_pos)
        if is_first_layer:
            query_pos = self.query_pos_proj(query_pos)
            q = query_content + query_pos
            k = key_content + key_pos
        else:
            q = query_content
            k = key_content
        v = value
        q = q.view(N, B, self.num_heads, C // self.num_heads)
        query_sine_embed = self.query_pos_sine_proj(query_sine_embed).view(N, B, self.num_heads, C // self.num_heads)
        q = torch.cat([q, query_sine_embed], dim=3).view(N, B, C * 2)
        k = k.view(HW, B, self.num_heads, C // self.num_heads)
        key_pos = key_pos.view(HW, B, self.num_heads, C // self.num_heads)
        k = torch.cat([k, key_pos], dim=3).view(HW, B, C * 2)
        q = q.reshape(N, B, self.num_heads, C * 2 // self.num_heads).permute(1, 2, 0, 3)
        k = k.reshape(HW, B, self.num_heads, C * 2 // self.num_heads).permute(1, 2, 0, 3)
        v = v.reshape(HW, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        scale = (C * 2 // self.num_heads) ** -0.5
        q = q * scale
        attn = q @ k.transpose(-2, -1)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn.masked_fill_(attn_mask, float('-inf'))
            else:
                attn += attn_mask
        if key_padding_mask is not None:
            attn = attn.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        if not self.batch_first:
            out = out.transpose(0, 1)
        return identity + self.proj_drop(out)


class ConvNormAct(nn.Module):
    """Utility module that stacks one convolution 2D layer,
    a normalization layer and an activation function.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): Size of the convolving kernel. Default: 1.
        stride (int): Stride of convolution. Default: 1.
        padding (int): Padding added to all four sides of the input. Default: 0.
        dilation (int): Spacing between kernel elements. Default: 1.
        groups (int): Number of blocked connections from input channels
            to output channels. Default: 1.
        bias (bool): if True, adds a learnable bias to the output. Default: True.
        norm_layer (nn.Module): Normalization layer used in `ConvNormAct`. Default: None.
        activation (nn.Module): Activation layer used in `ConvNormAct`. Default: None.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'int'=1, stride: 'int'=1, padding: 'int'=0, dilation: 'int'=1, groups: 'int'=1, bias: 'bool'=True, norm_layer: 'nn.Module'=None, activation: 'nn.Module'=None, **kwargs):
        super(ConvNormAct, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.norm = norm_layer
        self.activation = activation

    def forward(self, x):
        """Forward function for `ConvNormAct`"""
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


class CenterFeatureScaleModule(nn.Module):

    def forward(self, query, center_feature_scale_proj_weight, center_feature_scale_proj_bias):
        center_feature_scale = F.linear(query, weight=center_feature_scale_proj_weight, bias=center_feature_scale_proj_bias).sigmoid()
        return center_feature_scale


def _is_power_of_2(n):
    if not isinstance(n, int) or n < 0:
        raise ValueError('invalid input for _is_power_of_2: {} (type: {})'.format(n, type(n)))
    return n & n - 1 == 0 and n != 0


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()
    raise NotImplementedError(f'build_act_layer does not support {act_layer}')


def build_norm_layer(dim, norm_layer, in_format='channels_last', out_format='channels_last', eps=1e-06):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def _generate_dilation_grids(spatial_shapes, kernel_h, kernel_w, dilation_h, dilation_w, group, device):
    _, H_, W_, _ = spatial_shapes
    points_list = []
    x, y = torch.meshgrid(torch.linspace(-(dilation_w * (kernel_w - 1) // 2), -(dilation_w * (kernel_w - 1) // 2) + (kernel_w - 1) * dilation_w, kernel_w, dtype=torch.float32, device=device), torch.linspace(-(dilation_h * (kernel_h - 1) // 2), -(dilation_h * (kernel_h - 1) // 2) + (kernel_h - 1) * dilation_h, kernel_h, dtype=torch.float32, device=device))
    points_list.extend([x / W_, y / H_])
    grid = torch.stack(points_list, -1).reshape(-1, 1, 2).repeat(1, group, 1).permute(1, 0, 2)
    grid = grid.reshape(1, 1, 1, group * kernel_h * kernel_w, 2)
    return grid


def _get_reference_points(spatial_shapes, device, kernel_h, kernel_w, dilation_h, dilation_w, pad_h=0, pad_w=0, stride_h=1, stride_w=1):
    _, H_, W_, _ = spatial_shapes
    H_out = (H_ - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    W_out = (W_ - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    ref_y, ref_x = torch.meshgrid(torch.linspace(dilation_h * (kernel_h - 1) // 2 + 0.5, dilation_h * (kernel_h - 1) // 2 + 0.5 + (H_out - 1) * stride_h, H_out, dtype=torch.float32, device=device), torch.linspace(dilation_w * (kernel_w - 1) // 2 + 0.5, dilation_w * (kernel_w - 1) // 2 + 0.5 + (W_out - 1) * stride_w, W_out, dtype=torch.float32, device=device))
    ref_y = ref_y.reshape(-1)[None] / H_
    ref_x = ref_x.reshape(-1)[None] / W_
    ref = torch.stack((ref_x, ref_y), -1).reshape(1, H_out, W_out, 1, 2)
    return ref


def dcnv3_core_pytorch(input, offset, mask, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group, group_channels, offset_scale):
    input = F.pad(input, [0, 0, pad_h, pad_h, pad_w, pad_w])
    N_, H_in, W_in, _ = input.shape
    _, H_out, W_out, _ = offset.shape
    ref = _get_reference_points(input.shape, input.device, kernel_h, kernel_w, dilation_h, dilation_w, pad_h, pad_w, stride_h, stride_w)
    grid = _generate_dilation_grids(input.shape, kernel_h, kernel_w, dilation_h, dilation_w, group, input.device)
    spatial_norm = torch.tensor([W_in, H_in]).reshape(1, 1, 1, 2).repeat(1, 1, 1, group * kernel_h * kernel_w)
    sampling_locations = (ref + grid * offset_scale).repeat(N_, 1, 1, 1, 1).flatten(3, 4) + offset * offset_scale / spatial_norm
    P_ = kernel_h * kernel_w
    sampling_grids = 2 * sampling_locations - 1
    input_ = input.view(N_, H_in * W_in, group * group_channels).transpose(1, 2).reshape(N_ * group, group_channels, H_in, W_in)
    sampling_grid_ = sampling_grids.view(N_, H_out * W_out, group, P_, 2).transpose(1, 2).flatten(0, 1)
    sampling_input_ = F.grid_sample(input_, sampling_grid_, mode='bilinear', padding_mode='zeros', align_corners=False)
    mask = mask.view(N_, H_out * W_out, group, P_).transpose(1, 2).reshape(N_ * group, 1, H_out * W_out, P_)
    output = (sampling_input_ * mask).sum(-1).view(N_, group * group_channels, H_out * W_out)
    return output.transpose(1, 2).reshape(N_, H_out, W_out, -1).contiguous()


class DCNv3_pytorch(nn.Module):

    def __init__(self, channels=64, kernel_size=3, dw_kernel_size=None, stride=1, pad=1, dilation=1, group=4, offset_scale=1.0, act_layer='GELU', norm_layer='LN', center_feature_scale=False):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        if not _is_power_of_2(_d_per_group):
            warnings.warn("You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 which is more efficient in our CUDA implementation.")
        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.dw_conv = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=dw_kernel_size, stride=1, padding=(dw_kernel_size - 1) // 2, groups=channels), build_norm_layer(channels, norm_layer, 'channels_first', 'channels_last'), build_act_layer(act_layer))
        self.offset = nn.Linear(channels, group * kernel_size * kernel_size * 2)
        self.mask = nn.Linear(channels, group * kernel_size * kernel_size)
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self._reset_parameters()
        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.0)
        constant_(self.offset.bias.data, 0.0)
        constant_(self.mask.weight.data, 0.0)
        constant_(self.mask.bias.data, 0.0)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(self, input):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        N, H, W, _ = input.shape
        x = self.input_proj(input)
        x_proj = x
        x1 = input.permute(0, 3, 1, 2)
        x1 = self.dw_conv(x1)
        offset = self.offset(x1)
        mask = self.mask(x1).reshape(N, H, W, self.group, -1)
        mask = F.softmax(mask, -1).reshape(N, H, W, -1)
        x = dcnv3_core_pytorch(x, offset, mask, self.kernel_size, self.kernel_size, self.stride, self.stride, self.pad, self.pad, self.dilation, self.dilation, self.group, self.group_channels, self.offset_scale)
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            center_feature_scale = center_feature_scale[..., None].repeat(1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        x = self.output_proj(x)
        return x


class DCNv3Function(Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, input, offset, mask, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group, group_channels, offset_scale, im2col_step):
        ctx.kernel_h = kernel_h
        ctx.kernel_w = kernel_w
        ctx.stride_h = stride_h
        ctx.stride_w = stride_w
        ctx.pad_h = pad_h
        ctx.pad_w = pad_w
        ctx.dilation_h = dilation_h
        ctx.dilation_w = dilation_w
        ctx.group = group
        ctx.group_channels = group_channels
        ctx.offset_scale = offset_scale
        ctx.im2col_step = im2col_step
        output = _C.dcnv3_forward(input, offset, mask, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group, group_channels, offset_scale, ctx.im2col_step)
        ctx.save_for_backward(input, offset, mask)
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        input, offset, mask = ctx.saved_tensors
        grad_input, grad_offset, grad_mask = _C.dcnv3_backward(input, offset, mask, ctx.kernel_h, ctx.kernel_w, ctx.stride_h, ctx.stride_w, ctx.pad_h, ctx.pad_w, ctx.dilation_h, ctx.dilation_w, ctx.group, ctx.group_channels, ctx.offset_scale, grad_output.contiguous(), ctx.im2col_step)
        return grad_input, grad_offset, grad_mask, None, None, None, None, None, None, None, None, None, None, None, None


class DCNv3(nn.Module):

    def __init__(self, channels=64, kernel_size=3, dw_kernel_size=None, stride=1, pad=1, dilation=1, group=4, offset_scale=1.0, act_layer='GELU', norm_layer='LN', center_feature_scale=False):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        if not _is_power_of_2(_d_per_group):
            warnings.warn("You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 which is more efficient in our CUDA implementation.")
        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.dw_conv = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=dw_kernel_size, stride=1, padding=(dw_kernel_size - 1) // 2, groups=channels), build_norm_layer(channels, norm_layer, 'channels_first', 'channels_last'), build_act_layer(act_layer))
        self.offset = nn.Linear(channels, group * kernel_size * kernel_size * 2)
        self.mask = nn.Linear(channels, group * kernel_size * kernel_size)
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self._reset_parameters()
        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.0)
        constant_(self.offset.bias.data, 0.0)
        constant_(self.mask.weight.data, 0.0)
        constant_(self.mask.bias.data, 0.0)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(self, input):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        N, H, W, _ = input.shape
        x = self.input_proj(input)
        x_proj = x
        dtype = x.dtype
        x1 = input.permute(0, 3, 1, 2)
        x1 = self.dw_conv(x1)
        offset = self.offset(x1)
        mask = self.mask(x1).reshape(N, H, W, self.group, -1)
        mask = F.softmax(mask, -1).reshape(N, H, W, -1).type(dtype)
        x = DCNv3Function.apply(x, offset, mask, self.kernel_size, self.kernel_size, self.stride, self.stride, self.pad, self.pad, self.dilation, self.dilation, self.group, self.group_channels, self.offset_scale, 256)
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            center_feature_scale = center_feature_scale[..., None].repeat(1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        x = self.output_proj(x)
        return x


def apply_box_noise(boxes: 'torch.Tensor', box_noise_scale: 'float'=0.4):
    """
    Args:
        boxes (torch.Tensor): Bounding boxes in format ``(x_c, y_c, w, h)`` with
            shape ``(num_boxes, 4)``
        box_noise_scale (float): Scaling factor for box noising. Default: 0.4.
    """
    if box_noise_scale > 0:
        diff = torch.zeros_like(boxes)
        diff[:, :2] = boxes[:, 2:] / 2
        diff[:, 2:] = boxes[:, 2:]
        boxes += torch.mul(torch.rand_like(boxes) * 2 - 1.0, diff) * box_noise_scale
        boxes = boxes.clamp(min=0.0, max=1.0)
    return boxes


def apply_label_noise(labels: 'torch.Tensor', label_noise_prob: 'float'=0.2, num_classes: 'int'=80):
    """
    Args:
        labels (torch.Tensor): Classification labels with ``(num_labels, )``.
        label_noise_prob (float): The probability of the label being noised. Default: 0.2.
        num_classes (int): Number of total categories.

    Returns:
        torch.Tensor: The noised labels the same shape as ``labels``.
    """
    if label_noise_prob > 0:
        p = torch.rand_like(labels.float())
        noised_index = torch.nonzero(p < label_noise_prob).view(-1)
        new_lebels = torch.randint_like(noised_index, 0, num_classes)
        noised_labels = labels.scatter_(0, noised_index, new_lebels)
        return noised_labels
    else:
        return labels


def inverse_sigmoid(x, eps=1e-05):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class GenerateDNQueries(nn.Module):
    """Generate denoising queries for DN-DETR

    Args:
        num_queries (int): Number of total queries in DN-DETR. Default: 300
        num_classes (int): Number of total categories. Default: 80.
        label_embed_dim (int): The embedding dimension for label encoding. Default: 256.
        denoising_groups (int): Number of noised ground truth groups. Default: 5.
        label_noise_prob (float): The probability of the label being noised. Default: 0.2.
        box_noise_scale (float): Scaling factor for box noising. Default: 0.4
        with_indicator (bool): If True, add indicator in noised label/box queries.

    """

    def __init__(self, num_queries: 'int'=300, num_classes: 'int'=80, label_embed_dim: 'int'=256, denoising_groups: 'int'=5, label_noise_prob: 'float'=0.2, box_noise_scale: 'float'=0.4, with_indicator: 'bool'=False):
        super(GenerateDNQueries, self).__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.label_embed_dim = label_embed_dim
        self.denoising_groups = denoising_groups
        self.label_noise_prob = label_noise_prob
        self.box_noise_scale = box_noise_scale
        self.with_indicator = with_indicator
        if with_indicator:
            self.label_encoder = nn.Embedding(num_classes, label_embed_dim - 1)
        else:
            self.label_encoder = nn.Embedding(num_classes, label_embed_dim)

    def generate_query_masks(self, max_gt_num_per_image, device):
        noised_query_nums = max_gt_num_per_image * self.denoising_groups
        tgt_size = noised_query_nums + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size) < 0
        attn_mask[noised_query_nums:, :noised_query_nums] = True
        for i in range(self.denoising_groups):
            if i == 0:
                attn_mask[max_gt_num_per_image * i:max_gt_num_per_image * (i + 1), max_gt_num_per_image * (i + 1):noised_query_nums] = True
            if i == self.denoising_groups - 1:
                attn_mask[max_gt_num_per_image * i:max_gt_num_per_image * (i + 1), :max_gt_num_per_image * i] = True
            else:
                attn_mask[max_gt_num_per_image * i:max_gt_num_per_image * (i + 1), max_gt_num_per_image * (i + 1):noised_query_nums] = True
                attn_mask[max_gt_num_per_image * i:max_gt_num_per_image * (i + 1), :max_gt_num_per_image * i] = True
        return attn_mask

    def forward(self, gt_labels_list, gt_boxes_list):
        """
        Args:
            gt_boxes_list (list[torch.Tensor]): Ground truth bounding boxes per image
                with normalized coordinates in format ``(x, y, w, h)`` in shape ``(num_gts, 4)``
            gt_labels_list (list[torch.Tensor]): Classification labels per image in shape ``(num_gt, )``.
        """
        gt_labels = torch.cat(gt_labels_list)
        gt_boxes = torch.cat(gt_boxes_list)
        gt_labels = gt_labels.repeat(self.denoising_groups, 1).flatten()
        gt_boxes = gt_boxes.repeat(self.denoising_groups, 1)
        device = gt_labels.device
        assert len(gt_labels_list) == len(gt_boxes_list)
        batch_size = len(gt_labels_list)
        gt_nums_per_image = [x.numel() for x in gt_labels_list]
        noised_labels = apply_label_noise(gt_labels, self.label_noise_prob, self.num_classes)
        noised_boxes = apply_box_noise(gt_boxes, self.box_noise_scale)
        noised_boxes = inverse_sigmoid(noised_boxes)
        label_embedding = self.label_encoder(noised_labels)
        query_num = label_embedding.shape[0]
        if self.with_indicator:
            label_embedding = torch.cat([label_embedding, torch.ones([query_num, 1])], 1)
        max_gt_num_per_image = max(gt_nums_per_image)
        noised_query_nums = max_gt_num_per_image * self.denoising_groups
        noised_label_queries = torch.zeros(noised_query_nums, self.label_embed_dim).repeat(batch_size, 1, 1)
        noised_box_queries = torch.zeros(noised_query_nums, 4).repeat(batch_size, 1, 1)
        batch_idx = torch.arange(0, batch_size)
        batch_idx_per_instance = torch.repeat_interleave(batch_idx, torch.tensor(gt_nums_per_image).long())
        batch_idx_per_group = batch_idx_per_instance.repeat(self.denoising_groups, 1).flatten()
        if len(gt_nums_per_image):
            valid_index_per_group = torch.cat([torch.tensor(list(range(num))) for num in gt_nums_per_image])
            valid_index_per_group = torch.cat([(valid_index_per_group + max_gt_num_per_image * i) for i in range(self.denoising_groups)]).long()
        if len(batch_idx_per_group):
            noised_label_queries[batch_idx_per_group, valid_index_per_group] = label_embedding
            noised_box_queries[batch_idx_per_group, valid_index_per_group] = noised_boxes
        attn_mask = self.generate_query_masks(max_gt_num_per_image, device)
        return noised_label_queries, noised_box_queries, attn_mask, self.denoising_groups, max_gt_num_per_image


class GenerateCDNQueries(nn.Module):

    def __init__(self, num_queries: 'int'=300, num_classes: 'int'=80, label_embed_dim: 'int'=256, denoising_nums: 'int'=100, label_noise_prob: 'float'=0.5, box_noise_scale: 'float'=1.0):
        super(GenerateCDNQueries, self).__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.label_embed_dim = label_embed_dim
        self.denoising_nums = denoising_nums
        self.label_noise_prob = label_noise_prob
        self.box_noise_scale = box_noise_scale
        self.label_encoder = nn.Embedding(num_classes, label_embed_dim)

    def forward(self, gt_labels_list, gt_boxes_list):
        denoising_nums = self.denoising_nums * 2


class LayerNorm(nn.Module):
    """LayerNorm which supports both channel_last (default) and channel_first data format.
    The inputs data format should be as follows:
        - channel_last: (bs, h, w, channels)
        - channel_first: (bs, channels, h, w)

    Args:
        normalized_shape (tuple): The size of the input feature dim.
        eps (float): A value added to the denominator for
            numerical stability. Default: True.
        channel_last (bool): Set True for `channel_last` input data
            format. Default: True.
    """

    def __init__(self, normalized_shape, eps=1e-06, channel_last=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.channel_last = channel_last
        self.normalized_shape = normalized_shape,

    def forward(self, x):
        """Forward function for `LayerNorm`"""
        if self.channel_last:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class FFN(nn.Module):

    def __init__(self, d_model, d_ffn, dropout=0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU(True)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm(tgt)
        return tgt


class MultiScaleDeformableAttnFunction(Function):

    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = _C.ms_deform_attn_forward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = _C.ms_deform_attn_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def multi_scale_deformable_attn_pytorch(value: 'torch.Tensor', value_spatial_shapes: 'torch.Tensor', sampling_locations: 'torch.Tensor', attention_weights: 'torch.Tensor') ->torch.Tensor:
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([(H_ * W_) for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(bs, num_heads * embed_dims, num_queries)
    return output.transpose(1, 2).contiguous()


class MultiScaleDeformableAttention(nn.Module):
    """Multi-Scale Deformable Attention Module used in Deformable-DETR

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dim (int): The embedding dimension of Attention. Default: 256.
        num_heads (int): The number of attention heads. Default: 8.
        num_levels (int): The number of feature map used in Attention. Default: 4.
        num_points (int): The number of sampling points for each query
            in each head. Default: 4.
        img2col_steps (int): The step used in image_to_column. Defualt: 64.
            dropout (float): Dropout layer used in output. Default: 0.1.
        batch_first (bool): if ``True``, then the input and output tensor will be
            provided as `(bs, n, embed_dim)`. Default: False. `(n, bs, embed_dim)`
    """

    def __init__(self, embed_dim: 'int'=256, num_heads: 'int'=8, num_levels: 'int'=4, num_points: 'int'=4, img2col_step: 'int'=64, dropout: 'float'=0.1, batch_first: 'bool'=False):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads, but got {} and {}'.format(embed_dim, num_heads))
        head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        if not _is_power_of_2(head_dim):
            warnings.warn("""
                You'd better set d_model in MSDeformAttn to make sure that
                each dim of the attention head a power of 2, which is more efficient.
                """)
        self.im2col_step = img2col_step
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.init_weights()

    def init_weights(self):
        """
        Default initialization for Parameters of Module.
        """
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(self, query: 'torch.Tensor', key: 'Optional[torch.Tensor]'=None, value: 'Optional[torch.Tensor]'=None, identity: 'Optional[torch.Tensor]'=None, query_pos: 'Optional[torch.Tensor]'=None, key_padding_mask: 'Optional[torch.Tensor]'=None, reference_points: 'Optional[torch.Tensor]'=None, spatial_shapes: 'Optional[torch.Tensor]'=None, level_start_index: 'Optional[torch.Tensor]'=None, **kwargs) ->torch.Tensor:
        """Forward Function of MultiScaleDeformableAttention

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)`
            key (torch.Tensor): Key embeddings with shape
                `(num_key, bs, embed_dim)`
            value (torch.Tensor): Value embeddings with shape
                `(num_key, bs, embed_dim)`
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default: None. If None, `query` will be
                used.
            query_pos (torch.Tensor): The position embedding for `query`. Default: None.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with shape `(bs, num_key)`,
                indicating which elements within `key` to be ignored in attention.
            reference_points (torch.Tensor): The normalized reference points
                with shape `(bs, num_query, num_levels, 2)`,
                all elements is range in [0, 1], top-left (0, 0),
                bottom-right (1, 1), including padding are.
                or `(N, Length_{query}, num_levels, 4)`, add additional
                two dimensions `(h, w)` to form reference boxes.
            spatial_shapes (torch.Tensor): Spatial shape of features in different levels.
                With shape `(num_levels, 2)`, last dimension represents `(h, w)`.
            level_start_index (torch.Tensor): The start index of each level. A tensor with
                shape `(num_levels, )` which can be represented as
                `[0, h_0 * w_0, h_0 * w_0 + h_1 * w_1, ...]`.

        Returns:
            torch.Tensor: forward results with shape `(num_query, bs, embed_dim)`
        """
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], float(0))
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, num_query, self.num_heads, self.num_levels, self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError('Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(value if value.dtype == torch.float16 else value, spatial_shapes, level_start_index, sampling_locations, attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)
        if value.dtype == torch.float16:
            output = output
        output = self.output_proj(output)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return self.dropout(output) + identity


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-06
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self, _repr_indent=4):
        head = 'Positional encoding ' + self.__class__.__name__
        body = ['num_pos_feats: {}'.format(self.num_pos_feats), 'temperature: {}'.format(self.temperature), 'normalize: {}'.format(self.normalize), 'scale: {}'.format(self.scale)]
        lines = [head] + [(' ' * _repr_indent + line) for line in body]
        return '\n'.join(lines)


class PositionEmbeddingLearned(nn.Module):
    """
    Position embedding with learnable embedding weights.

    Args:
        num_pos_feats (int): The feature dimension for each position along
            x-axis or y-axis. The final returned dimension for each position
            is 2 times of the input value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default: 50.
        col_num_embed (int, optional): The dictionary size of column embeddings.
            Default: 50.
    """

    def __init__(self, num_pos_feats: 'int'=256, row_num_embed: 'int'=50, col_num_embed: 'int'=50):
        super().__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_pos_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_pos_feats)
        self.num_pos_feats = num_pos_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, mask):
        """Forward function for `PositionEmbeddingLearned`.

        Args:
            mask (torch.Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for the input tensor. Shape as `(bs, h, w)`.

        Returns:
            torch.Tensor: Returned position embedding with
            shape `(bs, num_pos_feats * 2, h, w)`
        """
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_emb = self.col_embed(x)
        y_emb = self.row_embed(y)
        pos = torch.cat([x_emb.unsqueeze(0).repeat(h, 1, 1), y_emb.unsqueeze(1).repeat(1, w, 1)], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos


class BaseTransformerLayer(nn.Module):
    """The implementation of Base `TransformerLayer` used in Transformer. Modified
    from `mmcv <https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/bricks/transformer.py>`_.
    It can be built by directly passing the `Attentions`, `FFNs`, `Norms`
    module, which support more flexible cusomization combined with
    `LazyConfig` system. The `BaseTransformerLayer` also supports `prenorm`
    when you specifying the `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .
    Args:
        attn (list[nn.Module] | nn.Module): nn.Module or a list
            contains the attention module used in TransformerLayer.
        ffn (nn.Module): FFN module used in TransformerLayer.
        norm (nn.Module): Normalization layer used in TransformerLayer.
        operation_order (tuple[str]): The execution order of operation in
            transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying the first element as `norm`.
            Default = None.
    """

    def __init__(self, attn: 'List[nn.Module]', ffn: 'nn.Module', norm: 'nn.Module', operation_order: 'tuple'=None):
        super(BaseTransformerLayer, self).__init__()
        assert set(operation_order).issubset({'self_attn', 'encoder_cross_attn', 'norm', 'cross_attn', 'ffn'})
        num_attn = operation_order.count('self_attn') + operation_order.count('cross_attn') + operation_order.count('encoder_cross_attn')
        if isinstance(attn, nn.Module):
            attn = [copy.deepcopy(attn) for _ in range(num_attn)]
        else:
            assert len(attn) == num_attn, f'The length of attn (nn.Module or List[nn.Module]) {num_attn}is not consistent with the number of attention in operation_order {operation_order}'
        self.num_attn = num_attn
        self.operation_order = operation_order
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = nn.ModuleList()
        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'encoder_cross_attn', 'cross_attn']:
                self.attentions.append(attn[index])
                index += 1
        self.embed_dim = self.attentions[0].embed_dim
        self.ffns = nn.ModuleList()
        num_ffns = operation_order.count('ffn')
        for _ in range(num_ffns):
            self.ffns.append(copy.deepcopy(ffn))
        self.norms = nn.ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(copy.deepcopy(norm))

    def forward(self, query: 'torch.Tensor', key: 'torch.Tensor'=None, value: 'torch.Tensor'=None, query_pos: 'torch.Tensor'=None, key_pos: 'torch.Tensor'=None, attn_masks: 'List[torch.Tensor]'=None, query_key_padding_mask: 'torch.Tensor'=None, key_padding_mask: 'torch.Tensor'=None, reference_points: 'torch.Tensor'=None, **kwargs):
        """Forward function for `BaseTransformerLayer`.
        **kwargs contains the specific arguments of attentions.
        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` or `(bs, num_query, embed_dim)`
                which should be specified follows the attention module used in
                `BaseTransformerLayer`.
            key (torch.Tensor): Key embeddings used in `Attention`.
            value (torch.Tensor): Value embeddings with the same shape as `key`.
            query_pos (torch.Tensor): The position embedding for `query`.
                Default: None.
            key_pos (torch.Tensor): The position embedding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): A list of 2D ByteTensor used
                in calculation the corresponding attention. The length of
                `attn_masks` should be equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape `(bs, num_query)`. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (torch.Tensor): ByteTensor for `key`, with
                shape `(bs, num_key)`. Default: None.
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f'Use same attn_mask in all attentions in {self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of attn_masks {len(attn_masks)} must be equal to the number of attention in operation_order {self.num_attn}'
        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](query, temp_key, temp_value, identity if self.pre_norm else None, query_pos=query_pos, key_pos=query_pos, attn_mask=attn_masks[attn_index], key_padding_mask=query_key_padding_mask, reference_points=reference_points, **kwargs)
                attn_index += 1
                identity = query
            elif layer == 'encoder_cross_attn':
                temp_key = temp_value = value
                query = self.attentions[attn_index](query, temp_key, temp_value, identity if self.pre_norm else None, query_pos=query_pos, key_pos=query_pos, attn_mask=attn_masks[attn_index], key_padding_mask=query_key_padding_mask, reference_points=reference_points, **kwargs)
                attn_index += 1
                identity = query
            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](query, key, value, identity if self.pre_norm else None, query_pos=query_pos, key_pos=key_pos, attn_mask=attn_masks[attn_index], key_padding_mask=key_padding_mask, reference_points=reference_points, **kwargs)
                attn_index += 1
                identity = query
            elif layer == 'ffn':
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1
        return query


class TransformerLayerSequence(nn.Module):
    """Base class for TransformerEncoder and TransformerDecoder, which will copy
    the passed `transformer_layers` module `num_layers` time or save the passed
    list of `transformer_layers` as parameters named ``self.layers``
    which is the type of ``nn.ModuleList``.
    The users should inherit `TransformerLayerSequence` and implemente their
    own forward function.

    Args:
        transformer_layers (list[BaseTransformerLayer] | BaseTransformerLayer): A list
            of BaseTransformerLayer. If it is obj:`BaseTransformerLayer`, it
            would be repeated `num_layers` times to a list[BaseTransformerLayer]
        num_layers (int): The number of `TransformerLayer`. Default: None.
    """

    def __init__(self, transformer_layers=None, num_layers=None):
        super(TransformerLayerSequence, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        if isinstance(transformer_layers, nn.Module):
            for _ in range(num_layers):
                self.layers.append(copy.deepcopy(transformer_layers))
        else:
            assert isinstance(transformer_layers, list) and len(transformer_layers) == num_layers

    def forward(self):
        """Forward function of `TransformerLayerSequence`. The users should inherit
        `TransformerLayerSequence` and implemente their own forward function.
        """
        raise NotImplementedError()


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_head_dim=None, rope=None, xattn=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q_proj = nn.Linear(dim, all_head_dim, bias=False)
        self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
        self.v_proj = nn.Linear(dim, all_head_dim, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.rope = rope
        self.xattn = xattn
        self.proj = nn.Linear(all_head_dim, dim)
        if not HAS_XFORMER:
            self.xattn = False

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.view(B, -1, C)
        N = H * W
        q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)
        k = F.linear(input=x, weight=self.k_proj.weight, bias=None)
        v = F.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias)
        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        q = self.rope(q).type_as(v)
        k = self.rope(k).type_as(v)
        if self.xattn:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            x = xops.memory_efficient_attention(q, k, v)
            x = x.reshape(B, N, -1)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1).type_as(x)
            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = x.view(B, H, W, C)
        return x


class SwiGLU(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.0, norm_layer=nn.LayerNorm, subln=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(self, dim, num_heads, mlp_ratio=4 * 2 / 3, qkv_bias=True, drop_path=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-06), window_size=0, use_residual_block=False, rope=None, xattn=True):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_residual_block (bool): If True, use a residual block after the MLP block.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, rope=rope, xattn=xattn)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = SwiGLU(in_features=dim, hidden_features=int(dim * mlp_ratio), subln=True, norm_layer=norm_layer)
        self.window_size = window_size
        self.use_residual_block = use_residual_block
        if use_residual_block:
            self.residual = ResBottleneckBlock(in_channels=dim, out_channels=dim, bottleneck_channels=dim // 2, norm='LN')

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.use_residual_block:
            x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return x


class LayerNormWithForceFP32(nn.Module):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: '_shape_t'
    eps: 'float'
    elementwise_affine: 'bool'

    def __init__(self, normalized_shape: '_shape_t', eps: 'float'=1e-05, elementwise_affine: 'bool'=True) ->None:
        super(LayerNormWithForceFP32, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = normalized_shape,
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: 'Tensor') ->Tensor:
        return F.layer_norm(input.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps).type_as(input)

    def extra_repr(self) ->Tensor:
        return '{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = dim + shape_len if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


class VisionRotaryEmbedding(nn.Module):

    def __init__(self, dim, pt_seq_len, ft_seq_len=None, custom_freqs=None, freqs_for='lang', theta=10000, max_freq=10, num_freqs=1):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1.0 / theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim)
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len
        freqs_h = torch.einsum('..., f -> ... f', t, freqs)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r=2)
        freqs_w = torch.einsum('..., f -> ... f', t, freqs)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r=2)
        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)
        self.register_buffer('freqs_cos', freqs.cos())
        self.register_buffer('freqs_sin', freqs.sin())
        None

    def forward(self, t, start_index=0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t = t * self.freqs_cos + rotate_half(t) * self.freqs_sin
        return torch.cat((t_left, t, t_right), dim=-1)


class VisionRotaryEmbeddingFast(nn.Module):

    def __init__(self, dim, pt_seq_len=16, ft_seq_len=None, custom_freqs=None, freqs_for='lang', theta=10000, max_freq=10, num_freqs=1):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1.0 / theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim)
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len
        freqs = torch.einsum('..., f -> ... f', t, freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)
        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])
        self.register_buffer('freqs_cos', freqs_cos)
        self.register_buffer('freqs_sin', freqs_sin)
        None

    def forward(self, t):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class Mlp(nn.Module):
    """Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FocalModulation(nn.Module):
    """ Focal Modulation

    Args:
        dim (int): Number of input channels.
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at focal level 1
        focal_factor (int, default=2): Step to increase the focal window
        use_postln (bool, default=False): Whether use post-modulation layernorm
    """

    def __init__(self, dim, proj_drop=0.0, focal_level=2, focal_window=7, focal_factor=2, use_postln=False, use_postln_in_modulation=False, normalize_modulator=False):
        super().__init__()
        self.dim = dim
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator
        self.f = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=True)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()
        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(nn.Sequential(nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, groups=dim, padding=kernel_size // 2, bias=False), nn.GELU()))

    def forward(self, x):
        """ Forward function.

        Args:
            x: input features with shape of (B, H, W, C)
        """
        B, nH, nW, C = x.shape
        x = self.f(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        q, ctx, gates = torch.split(x, (C, C, self.focal_level + 1), 1)
        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * gates[:, l:l + 1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:]
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)
        x_out = q * self.h(ctx_all)
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out


class FocalModulationBlock(nn.Module):
    """ Focal Modulation Block.

    Args:
        dim (int): Number of input channels.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        focal_level (int): number of focal levels
        focal_window (int): focal kernel size at level 1
    """

    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, focal_level=2, focal_window=9, use_postln=False, use_postln_in_modulation=False, normalize_modulator=False, use_layerscale=False, layerscale_value=0.0001):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.use_postln = use_postln
        self.use_layerscale = use_layerscale
        self.norm1 = norm_layer(dim)
        self.modulation = FocalModulation(dim, focal_window=self.focal_window, focal_level=self.focal_level, proj_drop=drop, use_postln_in_modulation=use_postln_in_modulation, normalize_modulator=normalize_modulator)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.H = None
        self.W = None
        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if self.use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, 'input feature has wrong size'
        shortcut = x
        if not self.use_postln:
            x = self.norm1(x)
        x = x.view(B, H, W, C)
        x = self.modulation(x).view(B, H * W, C)
        if self.use_postln:
            x = self.norm1(x)
        x = shortcut + self.drop_path(self.gamma_1 * x)
        if self.use_postln:
            x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
        else:
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, 'input feature has wrong size'
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
        w_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class CrossAttention(nn.Module):
    """ Cross Attention Module
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 8
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: False.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        attn_head_dim (int, optional): Dimension of attention head.
        out_dim (int, optional): Dimension of output.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, attn_head_dim=None, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        assert all_head_dim == dim
        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]
        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = self.k_bias
            v_bias = self.v_bias
        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)
        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)
        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentiveBlock(nn.Module):
    """Attentive Block
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 8
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: False.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop (float, optional): Dropout rate. Default: 0.0.
        attn_drop (float, optional): Attention dropout rate. Default: 0.0.
        drop_path (float | tuple[float], optional): Stochastic depth rate.
            Default: 0.0.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
        attn_head_dim (int, optional): Dimension of attention head. Default: None.
        out_dim (int, optional): Dimension of output. Default: None.
    """

    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer='LN', attn_head_dim=None, out_dim=None):
        super().__init__()
        self.norm1_q = build_norm_layer(dim, norm_layer, eps=1e-06)
        self.norm1_k = build_norm_layer(dim, norm_layer, eps=1e-06)
        self.norm1_v = build_norm_layer(dim, norm_layer, eps=1e-06)
        self.cross_dcn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim, out_dim=out_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x_q, x_kv, pos_q, pos_k, bool_masked_pos, rel_pos_bias=None):
        x_q = self.norm1_q(x_q + pos_q)
        x_k = self.norm1_k(x_kv + pos_k)
        x_v = self.norm1_v(x_kv)
        x = self.cross_dcn(x_q, k=x_k, v=x_v)
        return x


class AttentionPoolingBlock(AttentiveBlock):

    def forward(self, x):
        x_q = x.mean(1, keepdim=True)
        x_kv = x
        pos_q, pos_k = 0, 0
        x = super().forward(x_q, x_kv, pos_q, pos_k, bool_masked_pos=None, rel_pos_bias=None)
        x = x.squeeze(1)
        return x


class StemLayer(nn.Module):
    """ Stem layer of InternImage
    Args:
        in_chans (int): number of input channels
        out_chans (int): number of output channels
        act_layer (str): activation layer
        norm_layer (str): normalization layer
    """

    def __init__(self, in_chans=3, out_chans=96, act_layer='GELU', norm_layer='BN'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans // 2, kernel_size=3, stride=2, padding=1)
        self.norm1 = build_norm_layer(out_chans // 2, norm_layer, 'channels_first', 'channels_first')
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(out_chans // 2, out_chans, kernel_size=3, stride=2, padding=1)
        self.norm2 = build_norm_layer(out_chans, norm_layer, 'channels_first', 'channels_last')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


class DownsampleLayer(nn.Module):
    """ Downsample layer of InternImage
    Args:
        channels (int): number of input channels
        norm_layer (str): normalization layer
    """

    def __init__(self, channels, norm_layer='LN'):
        super().__init__()
        self.conv = nn.Conv2d(channels, 2 * channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = build_norm_layer(2 * channels, norm_layer, 'channels_first', 'channels_last')

    def forward(self, x):
        x = self.conv(x.permute(0, 3, 1, 2))
        x = self.norm(x)
        return x


class MLPLayer(nn.Module):
    """ MLP layer of InternImage
    Args:
        in_features (int): number of input features
        hidden_features (int): number of hidden features
        out_features (int): number of output features
        act_layer (str): activation layer
        drop (float): dropout rate
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer='GELU', drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_act_layer(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InternImageLayer(nn.Module):
    """ Basic layer of InternImage
    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(self, core_op, channels, groups, mlp_ratio=4.0, drop=0.0, drop_path=0.0, act_layer='GELU', norm_layer='LN', post_norm=False, layer_scale=None, offset_scale=1.0, with_cp=False, dw_kernel_size=None, res_post_norm=False, center_feature_scale=False):
        super().__init__()
        self.channels = channels
        self.groups = groups
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp
        self.norm1 = build_norm_layer(channels, 'LN')
        self.post_norm = post_norm
        self.dcn = core_op(channels=channels, kernel_size=3, stride=1, pad=1, dilation=1, group=groups, offset_scale=offset_scale, act_layer=act_layer, norm_layer=norm_layer, dw_kernel_size=dw_kernel_size, center_feature_scale=center_feature_scale)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = build_norm_layer(channels, 'LN')
        self.mlp = MLPLayer(in_features=channels, hidden_features=int(channels * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = layer_scale is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(channels), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(channels), requires_grad=True)
        self.res_post_norm = res_post_norm
        if res_post_norm:
            self.res_post_norm1 = build_norm_layer(channels, 'LN')
            self.res_post_norm2 = build_norm_layer(channels, 'LN')

    def forward(self, x):

        def _inner_forward(x):
            if not self.layer_scale:
                if self.post_norm:
                    x = x + self.drop_path(self.norm1(self.dcn(x)))
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
                elif self.res_post_norm:
                    x = x + self.drop_path(self.res_post_norm1(self.dcn(self.norm1(x))))
                    x = x + self.drop_path(self.res_post_norm2(self.mlp(self.norm2(x))))
                else:
                    x = x + self.drop_path(self.dcn(self.norm1(x)))
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x
            if self.post_norm:
                x = x + self.drop_path(self.gamma1 * self.norm1(self.dcn(x)))
                x = x + self.drop_path(self.gamma2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma1 * self.dcn(self.norm1(x)))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            return x
        if self.with_cp and x.requires_grad:
            x = checkpoint.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class InternImageBlock(nn.Module):
    """ Block of InternImage
    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        depths (list): Depth of each block.
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(self, core_op, channels, depth, groups, downsample=True, mlp_ratio=4.0, drop=0.0, drop_path=0.0, act_layer='GELU', norm_layer='LN', post_norm=False, offset_scale=1.0, layer_scale=None, with_cp=False, dw_kernel_size=None, post_norm_block_ids=None, res_post_norm=False, center_feature_scale=False):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.post_norm = post_norm
        self.center_feature_scale = center_feature_scale
        self.blocks = nn.ModuleList([InternImageLayer(core_op=core_op, channels=channels, groups=groups, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, act_layer=act_layer, norm_layer=norm_layer, post_norm=post_norm, layer_scale=layer_scale, offset_scale=offset_scale, with_cp=with_cp, dw_kernel_size=dw_kernel_size, res_post_norm=res_post_norm, center_feature_scale=center_feature_scale) for i in range(depth)])
        if not self.post_norm or center_feature_scale:
            self.norm = build_norm_layer(channels, 'LN')
        self.post_norm_block_ids = post_norm_block_ids
        if post_norm_block_ids is not None:
            self.post_norms = nn.ModuleList([build_norm_layer(channels, 'LN', eps=1e-06) for _ in post_norm_block_ids])
        self.downsample = DownsampleLayer(channels=channels, norm_layer=norm_layer) if downsample else None

    def forward(self, x, return_wo_downsample=False):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.post_norm_block_ids is not None and i in self.post_norm_block_ids:
                index = self.post_norm_block_ids.index(i)
                x = self.post_norms[index](x)
        if not self.post_norm or self.center_feature_scale:
            x = self.norm(x)
        if return_wo_downsample:
            x_ = x
        if self.downsample is not None:
            x = self.downsample(x)
        if return_wo_downsample:
            return x, x_
        return x


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer('weight', torch.ones(n))
        self.register_buffer('bias', torch.zeros(n))
        self.register_buffer('running_mean', torch.zeros(n))
        self.register_buffer('running_var', torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super(FrozenBatchNorm2d, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-05
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: 'nn.Module', train_backbone: 'bool', num_channels: 'int', return_layers: 'dict'):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, x):
        xs = self.body(x)
        out = {}
        out.update(xs)
        return out


class TorchvisionResNet(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: 'str', train_backbone: 'bool', return_layers: 'dict'={'layer4': 'res5'}, dilation: 'bool'=False):
        backbone = getattr(torchvision.models, name)(replace_stride_with_dilation=[False, False, dilation], pretrained=False, norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_layers)


def binary_cross_entropy_loss_with_logits(inputs, pos_weights, neg_weights, avg_factor):
    p = inputs.sigmoid()
    loss = -pos_weights * p.log() - neg_weights * (1 - p).log()
    return loss.sum() / avg_factor


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-06)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / (area + 1e-06)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() ->int:
    """
    Returns the number of processes.
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


class BaseCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.pos_norm_type = 'softmax'
        self.weight_dict = weight_dict

    def _get_src_permutation_idx(self, indices, pos=True):
        batch_idx = torch.cat([torch.full_like(v[0], i) for i, v in enumerate(indices)])
        src_idx = torch.cat([v[0] for v in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(v[1], i) for i, v in enumerate(indices)])
        tgt_idx = torch.cat([v[1] for v in indices])
        return batch_idx, tgt_idx

    def get_loss(self, outputs: 'dict', targets: 'list', num_boxes: 'float', layer_spec: 'int', specify_indices=None):
        assert 'pred_boxes' in outputs
        assert 'pred_logits' in outputs
        assert layer_spec >= 0
        losses = {}
        pred_boxes = outputs['pred_boxes']
        src_logits = outputs['pred_logits']
        layer_id = layer_spec
        if not specify_indices:
            indices = self.matcher(outputs, targets, 1)
        else:
            indices = specify_indices
        target_boxes = torch.cat([t['boxes'][v[1]] for t, v in zip(targets, indices)], dim=0)
        target_classes_o = torch.cat([t['labels'][v[1]] for t, v in zip(targets, indices)])
        pos_idx = self._get_src_permutation_idx(indices)
        pos_idx_c = pos_idx + (target_classes_o.cpu(),)
        src_boxes = pred_boxes[pos_idx]
        prob = src_logits.sigmoid().float()
        alpha = 0.25
        iou = torch.diag(box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))[0])
        iou = torch.clamp(iou, 0.01)
        t = prob[pos_idx_c] ** alpha * iou ** (1 - alpha)
        t = torch.clamp(t, 0.1).detach()
        gamma = 2
        pos_weights = torch.zeros_like(src_logits)
        pos_weights[pos_idx_c] = (t - prob[pos_idx_c]) ** gamma
        neg_weights = 1 * prob ** gamma
        neg_weights[pos_idx_c] = (1 - t) * prob[pos_idx_c] ** gamma
        loss_ce = binary_cross_entropy_loss_with_logits(src_logits, pos_weights, neg_weights, reduction='mean', avg_factor=num_boxes)
        losses.update({'loss_class': loss_ce})
        loc_weight = 1
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = (loc_weight * loss_bbox).sum() / num_boxes
        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = (loc_weight * loss_giou.view(-1, 1)).sum() / num_boxes
        return losses, indices

    def forward(self, outputs, targets, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device = outputs_without_aux['pred_logits'].device
        if 'aux_outputs' in outputs:
            num_layers = len(outputs['aux_outputs']) + 1
        else:
            num_layers = 1
        losses = {}
        indices_list = []
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        l_dict, indices = self.get_loss(outputs_without_aux, targets, num_boxes, num_layers)
        losses.update(l_dict)
        indices_list.append(indices)
        if 'aux_outputs' in outputs:
            for i in range(num_layers - 1):
                aux_outputs = outputs['aux_outputs'][i]
                l_dict, indices = self.get_loss(aux_outputs, targets, num_boxes, i + 1)
                l_dict = {(k + f'_{i}'): v for k, v in l_dict.items()}
                indices_list.append(indices)
                losses.update(l_dict)
        if return_indices:
            return losses, indices
        else:
            return losses


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -torch.abs(gt_class_logits)


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs (torch.Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (torch.Tensor):
            A float tensor with the same shape as inputs. Stores the binary
            classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
        num_boxes (int): The number of boxes.
    Return:
        torch.Tensor: The computed dice loss.
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


dice_loss_jit = torch.jit.script(dice_loss)


class NestedTensor(object):

    def __init__(self, tensors, mask: 'Optional[Tensor]'):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: 'List[Tensor]') ->NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32))
        max_size.append(max_size_i)
    max_size = tuple(max_size)
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)
        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), 'constant', 1)
        padded_masks.append(padded_mask)
    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)
    return NestedTensor(tensor, mask=mask)


def nested_tensor_from_tensor_list(tensor_list: 'List[Tensor]'):
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            return _onnx_nested_tensor_from_tensor_list(tensor_list)
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
            m[:img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


def sigmoid_ce_loss(inputs: 'torch.Tensor', targets: 'torch.Tensor', num_masks: 'float'):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: 'float'=0.25, gamma: 'float'=2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (torch.Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (torch.Tensor): A float tensor with the same shape as inputs. Stores the binary
            classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
        num_boxes (int): The number of boxes.
        alpha (float, optional): Weighting factor in range (0, 1) to balance
            positive vs negative examples. Default: 0.25.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
            balance easy vs hard examples. Default: 2.

    Returns:
        torch.Tensor: The computed sigmoid focal loss.
    """
    inputs = inputs.float()
    targets = targets.float()
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * (1 - p_t) ** gamma
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_boxes


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, num_points, oversample_ratio, importance_sample_ratio, dn='no', dn_losses=[], panoptic_on=False, semantic_ce_loss=False):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.dn = dn
        self.dn_losses = dn_losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.focal_alpha = 0.25
        self.panoptic_on = panoptic_on
        self.semantic_ce_loss = semantic_ce_loss

    def loss_labels_ce(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'].float()
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1], dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_boxes_panoptic(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_labels = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        isthing = target_labels < 80
        target_boxes = target_boxes[isthing]
        src_boxes = src_boxes[isthing]
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert 'pred_masks' in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs['pred_masks']
        src_masks = src_masks[src_idx]
        masks = [t['masks'] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks
        target_masks = target_masks[tgt_idx]
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(src_masks, lambda logits: calculate_uncertainty(logits), self.num_points, self.oversample_ratio, self.importance_sample_ratio)
            point_labels = point_sample(target_masks, point_coords, align_corners=False).squeeze(1)
        point_logits = point_sample(src_masks, point_coords, align_corners=False).squeeze(1)
        losses = {'loss_mask': sigmoid_ce_loss_jit(point_logits, point_labels, num_masks), 'loss_dice': dice_loss_jit(point_logits, point_labels, num_masks)}
        del src_masks
        del target_masks
        return losses

    def prep_for_dn(self, mask_dict):
        output_known_lbs_bboxes = mask_dict['output_known_lbs_bboxes']
        known_indice = mask_dict['known_indice']
        scalar, pad_size = mask_dict['scalar'], mask_dict['pad_size']
        assert pad_size % scalar == 0
        single_pad = pad_size // scalar
        num_tgt = known_indice.numel()
        return output_known_lbs_bboxes, num_tgt, single_pad, scalar

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for _, tgt in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {'labels': self.loss_labels_ce if self.semantic_ce_loss else self.loss_labels, 'masks': self.loss_masks, 'boxes': self.loss_boxes_panoptic if self.panoptic_on else self.loss_boxes}
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets, mask_dict=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        if self.dn is not 'no' and mask_dict is not None:
            output_known_lbs_bboxes, num_tgt, single_pad, scalar = self.prep_for_dn(mask_dict)
            exc_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.arange(0, len(targets[i]['labels'])).long()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long()
                exc_idx.append((output_idx, tgt_idx))
        indices = self.matcher(outputs_without_aux, targets)
        num_masks = sum(len(t['labels']) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))
        if self.dn != 'no' and mask_dict is not None:
            l_dict = {}
            for loss in self.dn_losses:
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, exc_idx, num_masks * scalar))
            l_dict = {(k + f'_dn'): v for k, v in l_dict.items()}
            losses.update(l_dict)
        elif self.dn != 'no':
            l_dict = dict()
            l_dict['loss_bbox_dn'] = torch.as_tensor(0.0)
            l_dict['loss_giou_dn'] = torch.as_tensor(0.0)
            l_dict['loss_ce_dn'] = torch.as_tensor(0.0)
            if self.dn == 'seg':
                l_dict['loss_mask_dn'] = torch.as_tensor(0.0)
                l_dict['loss_dice_dn'] = torch.as_tensor(0.0)
            losses.update(l_dict)
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {(k + f'_{i}'): v for k, v in l_dict.items()}
                    losses.update(l_dict)
                if 'interm_outputs' in outputs:
                    start = 0
                else:
                    start = 1
                if i >= start:
                    if self.dn != 'no' and mask_dict is not None:
                        out_ = output_known_lbs_bboxes['aux_outputs'][i]
                        l_dict = {}
                        for loss in self.dn_losses:
                            l_dict.update(self.get_loss(loss, out_, targets, exc_idx, num_masks * scalar))
                        l_dict = {(k + f'_dn_{i}'): v for k, v in l_dict.items()}
                        losses.update(l_dict)
                    elif self.dn != 'no':
                        l_dict = dict()
                        l_dict[f'loss_bbox_dn_{i}'] = torch.as_tensor(0.0)
                        l_dict[f'loss_giou_dn_{i}'] = torch.as_tensor(0.0)
                        l_dict[f'loss_ce_dn_{i}'] = torch.as_tensor(0.0)
                        if self.dn == 'seg':
                            l_dict[f'loss_mask_dn_{i}'] = torch.as_tensor(0.0)
                            l_dict[f'loss_dice_dn_{i}'] = torch.as_tensor(0.0)
                        losses.update(l_dict)
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_masks)
                l_dict = {(k + f'_interm'): v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses

    def __repr__(self):
        head = 'Criterion ' + self.__class__.__name__
        body = ['matcher: {}'.format(self.matcher.__repr__(_repr_indent=8)), 'losses: {}'.format(self.losses), 'weight_dict: {}'.format(self.weight_dict), 'num_classes: {}'.format(self.num_classes), 'eos_coef: {}'.format(self.eos_coef), 'num_points: {}'.format(self.num_points), 'oversample_ratio: {}'.format(self.oversample_ratio), 'importance_sample_ratio: {}'.format(self.importance_sample_ratio)]
        _repr_indent = 4
        lines = [head] + [(' ' * _repr_indent + line) for line in body]
        return '\n'.join(lines)


def reduce_loss(loss, reduction):
    """Reduce loss as specified

    Args:
        loss (nn.Tensor): Elementwise loss tensor.
        reduction (str): Specified reduction function chosen from "none",
            "mean" and "sum".

    Return:
        nn.Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    elif reduction == 'mean':
        eps = torch.finfo(loss.dtype).eps
        loss = loss.sum() / (avg_factor + eps)
    elif reduction != 'none':
        raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def cross_entropy(preds, targets, weight=None, class_weight=None, reduction='mean', avg_factor=None, ignore_index=-100, avg_non_ignore=False):
    ignore_index = -100 if ignore_index is None else ignore_index
    loss = F.cross_entropy(preds, targets, weight=class_weight, reduction='none', ignore_index=ignore_index)
    if avg_factor is None and avg_non_ignore and reduction == 'mean':
        avg_factor = targets.numel() - (targets == ignore_index).sum().item()
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


class CrossEntropyLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0, ignore_index=None, avg_non_ignore=False):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore
        if ignore_index is not None and not self.avg_non_ignore and self.reduction == 'mean':
            warnings.warn('Default ``avg_non_ignore`` is False, if you would like to ignore the certain label and average loss over non-ignore labels, which is the same with PyTorch official cross_entropy, set ``avg_non_ignore=True``.')

    def forward(self, preds, targets, weight=None, avg_factor=None, class_weight=None, ignore_index=None, **kwargs):
        if ignore_index is None:
            ignore_index = self.ignore_index
        loss_class = self.loss_weight * cross_entropy(preds, targets, weight, class_weight, reduction=self.reduction, avg_factor=avg_factor, avg_non_ignore=self.avg_non_ignore)
        return loss_class


class DiceLoss(nn.Module):

    def __init__(self, use_sigmoid=True, reduction='mean', loss_weight=1.0, eps=0.001):
        super(DiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(self, preds, targets, weight=None, avg_factor=None):
        if self.use_sigmoid:
            preds = preds.sigmoid()
        loss = self.loss_weight * dice_loss(preds, targets, weight, eps=self.eps, reduction=self.reduction, avg_factor=avg_factor)
        return loss


def focal_loss_with_prob(preds, targets, weight=None, alpha=0.25, gamma=2.0, reduction='mean', avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Different from `sigmoid_focal_loss`, this function accepts probability
    as input.

    Args:
        preds (torch.Tensor): The prediction probability with shape (N, C),
            C is the number of classes.
        targets (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    num_classes = preds.size(1)
    targets = F.one_hot(targets, num_classes=num_classes + 1)
    targets = targets[:, :num_classes]
    targets = targets.type_as(preds)
    p_t = preds * targets + (1 - preds) * (1 - targets)
    ce_loss = F.binary_cross_entropy(preds, targets, reduction='none')
    loss = ce_loss * (1 - p_t) ** gamma
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if weight is not None:
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class FocalLoss(nn.Module):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

    Args:
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', loss_weight=1.0, activated=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

    def forward(self, preds, targets, weight=None, avg_factor=None):
        """Forward function for FocalLoss

        Args:
            preds (torch.Tensor): The prediction probability with shape ``(N, C)``.
                C is the number of classes.
            targets (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """
        if self.activated:
            loss_func = focal_loss_with_prob
        else:
            num_classes = preds.size(1)
            targets = F.one_hot(targets, num_classes=num_classes + 1)
            targets = targets[:, :num_classes]
            loss_func = sigmoid_focal_loss
        loss_class = self.loss_weight * loss_func(preds, targets, weight, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction, avg_factor=avg_factor)
        return loss_class


def giou_loss(preds: 'torch.Tensor', targets: 'torch.Tensor', weight=None, eps: 'float'=1e-06, reduction: 'str'='mean', avg_factor: 'int'=None):
    """`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        preds (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        targets (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    if targets.numel() == 0:
        return preds.sum() * 0
    x1, y1, x2, y2 = preds.unbind(dim=-1)
    x1g, y1g, x2g, y2g = targets.unbind(dim=-1)
    assert (x2 >= x1).all(), 'bad box: x1 larger than x2'
    assert (y2 >= y1).all(), 'bad box: y1 larger than y2'
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)
    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
    iouk = intsctk / (unionk + eps)
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)
    area_c = (xc2 - xc1) * (yc2 - yc1)
    miouk = iouk - (area_c - unionk) / (area_c + eps)
    loss = 1 - miouk
    if weight is not None:
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class GIoULoss(nn.Module):

    def __init__(self, eps: 'float'=1e-06, reduction: 'str'='mean', loss_weight: 'float'=1.0):
        super(GIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, preds, targets, weight=None, avg_factor=None):
        loss_giou = self.loss_weight * giou_loss(preds, targets, weight, eps=self.eps, reduction=self.reduction, avg_factor=avg_factor)
        return loss_giou


def l1_loss(preds, targets, weight=None, reduction: 'str'='mean', avg_factor: 'int'=None):
    if targets.numel() == 0:
        return preds.sum() * 0
    assert preds.size() == targets.size()
    loss = torch.abs(preds - targets)
    if weight is not None:
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class L1Loss(nn.Module):

    def __init__(self, reduction: 'str'='mean', loss_weight: 'float'=1.0):
        super(L1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, preds, targets, weight=None, avg_factor=None):
        loss_bbox = self.loss_weight * l1_loss(preds, targets, weight=weight, reduction=self.reduction, avg_factor=avg_factor)
        return loss_bbox


class FocalLossCost(nn.Module):

    def __init__(self, alpha: 'float'=0.25, gamma: 'float'=2.0, weight: 'float'=1.0, eps: 'float'=1e-08):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.eps = eps

    def forward(self, pred_logits, gt_labels):
        """
        Args:
            pred_logits (nn.Tensor): Predicted classification logits.
            gt_labels (nn.Tensor): Ground truth labels.

        Return:
            nn.Tensor: Focal loss cost matrix with weight in shape
                ``(num_queries, num_gt)``
        """
        alpha = self.alpha
        gamma = self.gamma
        eps = self.eps
        out_prob = pred_logits.sigmoid()
        neg_cost_class = (1 - alpha) * out_prob ** gamma * -(1 - out_prob + eps).log()
        pos_cost_class = alpha * (1 - out_prob) ** gamma * -(out_prob + eps).log()
        cost_class = pos_cost_class[:, gt_labels] - neg_cost_class[:, gt_labels]
        return cost_class * self.weight


class CrossEntropyCost(nn.Module):

    def __init__(self, weight: 'float'=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred_logits, gt_labels):
        """
        Args:
            pred_logits (nn.Tensor): Predicted classification logits.
            gt_labels (nn.Tensor): Ground truth labels.

        Return:
            nn.Tensor: CrossEntropy loss cost matrix with weight in shape
                ``(num_queries, num_gt)``
        """
        out_prob = pred_logits.softmax(-1)
        cost_class = -out_prob[:, gt_labels]
        return cost_class * self.weight


class GIoUCost(nn.Module):

    def __init__(self, weight: 'float'=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred_bboxes, gt_bboxes):
        """
        Args:
            pred_bboxes (nn.Tensor): Predicted bboxes with unnormalized coordinates
                (x1, y1, x2, y2) with shape (num_queries, 4).
            gt_bboxes (nn.Tensor): Ground truth boxes with unnormalized coordinates
                (x1, y1, x2, y2) with shape (num_gt, 4).

        Returns:
            torch.Tensor: GIoU cost with weight
        """
        cost_giou = -generalized_box_iou(pred_bboxes, gt_bboxes)
        return cost_giou * self.weight


class L1Cost(nn.Module):

    def __init__(self, weight: 'float'=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred_bboxes, gt_bboxes):
        """
        Args:
            pred_bboxes (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1] with shape
                (num_queries, 4).
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (cx, cy, w, h) with shape (num_gt, 4).

        Returns:
            torch.Tensor: cost_bbox with weight
        """
        cost_bbox = torch.cdist(pred_bboxes, gt_bboxes, p=1)
        return cost_bbox * self.weight


def batch_dice_loss(inputs: 'torch.Tensor', targets: 'torch.Tensor'):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum('nc,mc->nm', inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(batch_dice_loss)


def batch_sigmoid_ce_loss(inputs: 'torch.Tensor', targets: 'torch.Tensor'):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction='none')
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction='none')
    loss = torch.einsum('nc,mc->nm', pos, targets) + torch.einsum('nc,mc->nm', neg, 1 - targets)
    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(batch_sigmoid_ce_loss)


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: 'float'=1, cost_mask: 'float'=1, cost_dice: 'float'=1, num_points: 'int'=0, cost_box=0, cost_giou=0, panoptic_on=False):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_box = cost_box
        self.cost_giou = cost_giou
        self.panoptic_on = panoptic_on
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, 'all costs cant be 0'
        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets, cost=['cls', 'box', 'mask']):
        """More memory-friendly matching"""
        bs, num_queries = outputs['pred_logits'].shape[:2]
        indices = []
        for b in range(bs):
            out_bbox = outputs['pred_boxes'][b]
            if 'box' in cost:
                tgt_bbox = targets[b]['boxes']
                cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
                cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
            else:
                cost_bbox = torch.tensor(0)
                cost_giou = torch.tensor(0)
            out_prob = outputs['pred_logits'][b].sigmoid()
            tgt_ids = targets[b]['labels']
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * out_prob ** gamma * -(1 - out_prob + 1e-05).log()
            pos_cost_class = alpha * (1 - out_prob) ** gamma * -(out_prob + 1e-05).log()
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            if 'mask' in cost:
                out_mask = outputs['pred_masks'][b]
                tgt_mask = targets[b]['masks']
                out_mask = out_mask[:, None]
                tgt_mask = tgt_mask[:, None]
                point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
                tgt_mask = point_sample(tgt_mask, point_coords.repeat(tgt_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
                out_mask = point_sample(out_mask, point_coords.repeat(out_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
                with autocast(enabled=False):
                    out_mask = out_mask.float()
                    tgt_mask = tgt_mask.float()
                    if out_mask.shape[0] == 0 or tgt_mask.shape[0] == 0:
                        cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
                        cost_dice = batch_dice_loss(out_mask, tgt_mask)
                    else:
                        cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
                        cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
            else:
                cost_mask = torch.tensor(0)
                cost_dice = torch.tensor(0)
            if self.panoptic_on:
                isthing = tgt_ids < 80
                cost_bbox[:, ~isthing] = cost_bbox[:, isthing].mean()
                cost_giou[:, ~isthing] = cost_giou[:, isthing].mean()
                cost_bbox[cost_bbox.isnan()] = 0.0
                cost_giou[cost_giou.isnan()] = 0.0
            C = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice + self.cost_box * cost_bbox + self.cost_giou * cost_giou
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    @torch.no_grad()
    def forward(self, outputs, targets, cost=['cls', 'box', 'mask']):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets, cost)

    def __repr__(self, _repr_indent=4):
        head = 'Matcher ' + self.__class__.__name__
        body = ['cost_class: {}'.format(self.cost_class), 'cost_mask: {}'.format(self.cost_mask), 'cost_dice: {}'.format(self.cost_dice)]
        lines = [head] + [(' ' * _repr_indent + line) for line in body]
        return '\n'.join(lines)


class ChannelMapper(nn.Module):
    """Channel Mapper for reduce/increase channels of backbone features. Modified
    from `mmdet <https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/necks/channel_mapper.py>`_.

    This is used to reduce/increase the channels of backbone features.

    Args:
        input_shape (Dict[str, ShapeSpec]): A dict which contains the backbone features meta infomation,
            e.g. ``input_shape = {"res5": ShapeSpec(channels=2048)}``.
        in_features (List[str]): A list contains the keys which maps the features output from the backbone,
            e.g. ``in_features = ["res"]``.
        out_channels (int): Number of output channels for each scale.
        kernel_size (int, optional): Size of the convolving kernel for each scale.
            Default: 3.
        stride (int, optional): Stride of convolution for each scale. Default: 1.
        bias (bool, optional): If True, adds a learnable bias to the output of each scale.
            Default: True.
        groups (int, optional): Number of blocked connections from input channels to
            output channels for each scale. Default: 1.
        dilation (int, optional): Spacing between kernel elements for each scale.
            Default: 1.
        norm_layer (nn.Module, optional): The norm layer used for each scale. Default: None.
        activation (nn.Module, optional): The activation layer used for each scale. Default: None.
        num_outs (int, optional): Number of output feature maps. There will be ``extra_convs`` when
            ``num_outs`` is larger than the length of ``in_features``. Default: None.

    Examples:
        >>> import torch
        >>> import torch.nn as nn
        >>> from detrex.modeling import ChannelMapper
        >>> from detectron2.modeling import ShapeSpec
        >>> input_features = {
        ... "p0": torch.randn(1, 128, 128, 128),
        ... "p1": torch.randn(1, 256, 64, 64),
        ... "p2": torch.randn(1, 512, 32, 32),
        ... "p3": torch.randn(1, 1024, 16, 16),
        ... }
        >>> input_shapes = {
        ... "p0": ShapeSpec(channels=128),
        ... "p1": ShapeSpec(channels=256),
        ... "p2": ShapeSpec(channels=512),
        ... "p3": ShapeSpec(channels=1024),
        ... }
        >>> in_features = ["p0", "p1", "p2", "p3"]
        >>> neck = ChannelMapper(
        ... input_shapes=input_shapes,
        ... in_features=in_features,
        ... out_channels=256,
        ... norm_layer=nn.GroupNorm(num_groups=32, num_channels=256)
        >>> outputs = neck(input_features)
        >>> for i in range(len(outputs)):
        ... print(f"output[{i}].shape = {outputs[i].shape}")
        output[0].shape = torch.Size([1, 256, 128, 128])
        output[1].shape = torch.Size([1, 256, 64, 64])
        output[2].shape = torch.Size([1, 256, 32, 32])
        output[3].shape = torch.Size([1, 256, 16, 16])
    """

    def __init__(self, input_shapes: 'Dict[str, ShapeSpec]', in_features: 'List[str]', out_channels: 'int', kernel_size: 'int'=3, stride: 'int'=1, bias: 'bool'=True, groups: 'int'=1, dilation: 'int'=1, norm_layer: 'nn.Module'=None, activation: 'nn.Module'=None, num_outs: 'int'=None, **kwargs):
        super(ChannelMapper, self).__init__()
        self.extra_convs = None
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]
        if num_outs is None:
            num_outs = len(input_shapes)
        self.convs = nn.ModuleList()
        for in_channel in in_channels_per_feature:
            self.convs.append(ConvNormAct(in_channels=in_channel, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=bias, groups=groups, dilation=dilation, norm_layer=copy.deepcopy(norm_layer), activation=copy.deepcopy(activation)))
        if num_outs > len(in_channels_per_feature):
            self.extra_convs = nn.ModuleList()
            for i in range(len(in_channels_per_feature), num_outs):
                if i == len(in_channels_per_feature):
                    in_channel = in_channels_per_feature[-1]
                else:
                    in_channel = out_channels
                self.extra_convs.append(ConvNormAct(in_channels=in_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=bias, groups=groups, dilation=dilation, norm_layer=copy.deepcopy(norm_layer), activation=copy.deepcopy(activation)))
        self.input_shapes = input_shapes
        self.in_features = in_features
        self.out_channels = out_channels

    def forward(self, inputs):
        """Forward function for ChannelMapper

        Args:
            inputs (Dict[str, torch.Tensor]): The backbone feature maps.

        Return:
            tuple(torch.Tensor): A tuple of the processed features.
        """
        assert len(inputs) == len(self.convs)
        outs = [self.convs[i](inputs[self.in_features[i]]) for i in range(len(inputs))]
        if self.extra_convs:
            for i in range(len(self.extra_convs)):
                if i == 0:
                    outs.append(self.extra_convs[0](inputs[self.in_features[-1]]))
                else:
                    outs.append(self.extra_convs[i](outs[-1]))
        return tuple(outs)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
    return torch.stack(b, dim=-1)


class AlignDETR(nn.Module):
    """
    AlignDETR is based on DINO.
    Args:
        backbone (nn.Module): backbone module
        position_embedding (nn.Module): position embedding module
        neck (nn.Module): neck module to handle the intermediate outputs features
        transformer (nn.Module): transformer module
        embed_dim (int): dimension of embedding
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 300.
        device (str): Training device. Default: "cuda".
    """

    def __init__(self, backbone: 'nn.Module', position_embedding: 'nn.Module', neck: 'nn.Module', transformer: 'nn.Module', embed_dim: 'int', num_classes: 'int', num_queries: 'int', criterion: 'nn.Module', pixel_mean: 'List[float]'=[123.675, 116.28, 103.53], pixel_std: 'List[float]'=[123.675, 116.28, 103.53], aux_loss: 'bool'=True, select_box_nums_for_evaluation: 'int'=300, device='cuda', dn_number: 'int'=100, label_noise_ratio: 'float'=0.2, box_noise_scale: 'float'=1.0, prior_init=0.01):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.neck = neck
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.transformer = transformer
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        self.criterion = criterion
        self.label_enc = nn.Embedding(num_classes, embed_dim)
        self.dn_number = dn_number
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        prior_prob = prior_init
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)
        num_pred = transformer.decoder.num_layers + 1
        self.class_embed = nn.ModuleList([copy.deepcopy(self.class_embed) for i in range(num_pred)])
        self.bbox_embed = nn.ModuleList([copy.deepcopy(self.bbox_embed) for i in range(num_pred)])
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.bbox_embed = self.bbox_embed
        for bbox_embed_layer in self.bbox_embed:
            nn.init.constant_(bbox_embed_layer.layers[-1].bias.data[2:], 0.0)
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation

    def forward(self, batched_inputs):
        """Forward function of `DINO` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries (anchor boxes in DAB-DETR).
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        images = self.preprocess_image(batched_inputs)
        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]['instances'].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)
        features = self.backbone(images.tensor)
        multi_level_feats = self.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(F.interpolate(img_masks[None], size=feat.shape[-2:]).squeeze(0))
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))
        if self.training:
            gt_instances = [x['instances'] for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_cdn(targets, dn_number=self.dn_number, label_noise_ratio=self.label_noise_ratio, box_noise_scale=self.box_noise_scale, num_queries=self.num_queries, num_classes=self.num_classes, hidden_dim=self.embed_dim, label_enc=self.label_enc)
        else:
            input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None
        query_embeds = input_query_label, input_query_bbox
        inter_states, init_reference, inter_references, enc_state, enc_reference = self.transformer(multi_level_feats, multi_level_masks, multi_level_position_embeddings, query_embeds, attn_masks=[attn_mask, None])
        inter_states[0] += self.label_enc.weight[0, 0] * 0.0
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        if dn_meta is not None:
            outputs_class, outputs_coord = self.dn_post_process(outputs_class, outputs_coord, dn_meta)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            output['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        interm_coord = enc_reference
        interm_class = self.transformer.decoder.class_embed[-1](enc_state)
        output['enc_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
        if self.training:
            loss_dict = self.criterion(output, targets, dn_meta)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output['pred_logits']
            box_pred = output['pred_boxes']
            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get('height', image_size[0])
                width = input_per_image.get('width', image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({'instances': r})
            return processed_results

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def prepare_for_cdn(self, targets, dn_number, label_noise_ratio, box_noise_scale, num_queries, num_classes, hidden_dim, label_enc):
        """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding
            in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
        if dn_number <= 0:
            return None, None, None, None
        dn_number = dn_number * 2
        known = [torch.ones_like(t['labels']) for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            return None, None, None, None
        dn_number = dn_number // int(max(known_num) * 2)
        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t['labels'] for t in targets])
        boxes = torch.cat([t['boxes'] for t in targets])
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)
        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()
        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < label_noise_ratio * 0.5).view(-1)
            new_label = torch.randint_like(chosen_indice, 0, num_classes)
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_padding = int(max(known_num))
        pad_size = int(single_padding * 2 * dn_number)
        positive_idx = torch.tensor(range(len(boxes))).long().unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2
            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2
            rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part, diff) * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]
        m = known_labels_expaned.long()
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)
        padding_label = torch.zeros(pad_size, hidden_dim)
        padding_bbox = torch.zeros(pad_size, 4)
        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)
        map_known_indice = torch.tensor([])
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])
            map_known_indice = torch.cat([(map_known_indice + single_padding * i) for i in range(2 * dn_number)]).long()
        if len(known_bid):
            input_query_label[known_bid.long(), map_known_indice] = input_label_embed
            input_query_bbox[known_bid.long(), map_known_indice] = input_bbox_embed
        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size) < 0
        attn_mask[pad_size:, :pad_size] = True
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_padding * 2 * i:single_padding * 2 * (i + 1), single_padding * 2 * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_padding * 2 * i:single_padding * 2 * (i + 1), :single_padding * i * 2] = True
            else:
                attn_mask[single_padding * 2 * i:single_padding * 2 * (i + 1), single_padding * 2 * (i + 1):pad_size] = True
                attn_mask[single_padding * 2 * i:single_padding * 2 * (i + 1), :single_padding * 2 * i] = True
        dn_meta = {'single_padding': single_padding * 2, 'dn_num': dn_number}
        return input_query_label, input_query_bbox, attn_mask, dn_meta

    def dn_post_process(self, outputs_class, outputs_coord, dn_metas):
        if dn_metas and dn_metas['single_padding'] > 0:
            padding_size = dn_metas['single_padding'] * dn_metas['dn_num']
            output_known_class = outputs_class[:, :, :padding_size, :]
            output_known_coord = outputs_coord[:, :, :padding_size, :]
            outputs_class = outputs_class[:, :, padding_size:, :]
            outputs_coord = outputs_coord[:, :, padding_size:, :]
            out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(output_known_class, output_known_coord)
            dn_metas['output_known_lbs_bboxes'] = out
        return outputs_class, outputs_coord

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x['image']) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode='floor')
        labels = topk_indexes % box_cls.shape[2]
        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(scores, labels, boxes, image_sizes)):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({'labels': gt_classes, 'boxes': gt_boxes})
        return new_targets


def get_local_rank(quality, indices):
    bs = len(indices)
    device = quality.device
    tgt_size = [len(tgt_ind) for _, tgt_ind in indices]
    ind_start = 0
    rank_list = []
    for i in range(bs):
        if tgt_size[i] == 0:
            rank_list.append(torch.zeros(0, dtype=torch.long, device=device))
            continue
        num_tgt = max(indices[i][1]) + 1
        quality_per_img = quality[ind_start:ind_start + tgt_size[i]]
        ind_start += tgt_size[i]
        k = torch.div(tgt_size[i], num_tgt, rounding_mode='floor')
        quality_per_img = quality_per_img.reshape(num_tgt, k)
        ind = quality_per_img.sort(dim=-1, descending=True)[1]
        rank_per_img = torch.zeros_like(quality_per_img, dtype=torch.long, device=device)
        rank_per_img.scatter_(-1, ind, torch.arange(k, device=device, dtype=torch.long).repeat(num_tgt, 1))
        rank_list.append(rank_per_img.flatten())
    return torch.cat(rank_list, 0)


def IA_BCE_loss(src_logits, pos_idx_c, src_boxes, target_boxes, indices, avg_factor, alpha, gamma, w_prime=1):
    prob = src_logits.sigmoid()
    pos_weights = torch.zeros_like(src_logits)
    neg_weights = prob ** gamma
    iou_scores = torch.diag(box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))[0])
    t = prob[pos_idx_c] ** alpha * iou_scores ** (1 - alpha)
    t = torch.clamp(t, 0.01).detach()
    rank = get_local_rank(t, indices)
    if type(w_prime) != int:
        rank_weight = w_prime[rank]
    else:
        rank_weight = w_prime
    t = t * rank_weight
    pos_weights[pos_idx_c] = t
    neg_weights[pos_idx_c] = 1 - t
    loss = -pos_weights * prob.log() - neg_weights * (1 - prob).log()
    return loss.sum() / avg_factor, rank_weight


class ManyToOneCriterion(BaseCriterion):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, match_number, gamma, alpha, tau):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__(num_classes, matcher, weight_dict)
        self.num_classes = num_classes
        self.matcher = matcher
        self.pos_norm_type = 'softmax'
        self.weight_dict = weight_dict
        self.match_number = match_number
        self.gamma = gamma
        self.alpha = alpha
        self.initialize_weight_table(match_number, tau)

    def initialize_weight_table(self, match_number, tau):
        self.weight_table = torch.zeros(len(match_number), max(match_number))
        for layer, n in enumerate(match_number):
            self.weight_table[layer][:n] = torch.exp(-torch.arange(n) / tau)

    def _get_local_rank(self, quality, indices):
        bs = len(indices)
        ind_size = [len(i) for i, _ in indices]
        ind_start = 0
        rank_list = []
        for i in range(bs):
            t = quality[ind_start:ind_start + ind_size[i]]
            ind_start += ind_size[i]
            if t.numel() > 0:
                gt_num = int(max(indices[i][1]) + 1)
                k = ind_size[i] // gt_num
            else:
                gt_num, k = 0, 0
            t = t.reshape(gt_num, k)
            t_ind = t.sort(dim=-1, descending=True)[1]
            rank = torch.zeros_like(t, dtype=torch.long, device=t.device)
            rank.scatter_(-1, t_ind, torch.arange(k, device=t.device, dtype=torch.long).repeat(gt_num, 1))
            rank_list.append(rank.flatten())
        return torch.cat(rank_list, 0)

    def get_loss(self, outputs: 'dict', targets: 'list', num_boxes: 'float', layer_spec: 'int', specify_indices=None):
        assert 'pred_boxes' in outputs
        assert 'pred_logits' in outputs
        assert layer_spec >= 0
        losses = {}
        pred_boxes = outputs['pred_boxes']
        src_logits = outputs['pred_logits']
        layer_id = layer_spec
        device = pred_boxes.device
        if not specify_indices:
            match_n = self.match_number[layer_id]
            indices = self.matcher(outputs, targets, match_n)
        else:
            match_n = 1
            indices = specify_indices
        target_boxes = torch.cat([t['boxes'][v[1]] for t, v in zip(targets, indices)], dim=0)
        target_classes_o = torch.cat([t['labels'][v[1]] for t, v in zip(targets, indices)])
        pos_idx = self._get_src_permutation_idx(indices)
        pos_idx_c = pos_idx + (target_classes_o.cpu(),)
        src_boxes = pred_boxes[pos_idx]
        if not specify_indices:
            w_prime = self.weight_table[layer_spec]
        else:
            w_prime = 1
        loss_class, loc_weight = IA_BCE_loss(src_logits, pos_idx_c, src_boxes, target_boxes, indices, num_boxes, self.alpha, self.gamma, w_prime)
        losses.update({'loss_class': loss_class})
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = (loc_weight * loss_bbox.sum(dim=-1)).sum() / num_boxes
        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = (loc_weight * loss_giou).sum() / num_boxes
        return losses, indices


def coords_fmap2orig(feature, stride):
    """
    transfor one fmap coords to orig coords
    Args
    featurn [batch_size,h,w,c]
    stride int
    Returns
    coords [n,2]
    """
    h, w = feature.shape[1:3]
    shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
    shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = torch.reshape(shift_x, [-1])
    shift_y = torch.reshape(shift_y, [-1])
    coords = torch.stack([shift_x, shift_y], -1) + stride // 2
    return coords


class GenTargets(nn.Module):

    def __init__(self, strides, limit_range):
        super().__init__()
        self.strides = strides
        self.limit_range = limit_range
        assert len(strides) == len(limit_range)

    def forward(self, srcs, gt_boxes, batch_classes):
        """
        inputs
        [0]list [cls_logits,cnt_logits,reg_preds]
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        [1]gt_boxes [batch_size,m,4]  FloatTensor
        [2]classes [batch_size,m]  LongTensor
        Returns
        cls_targets:[batch_size,sum(_h*_w),1]
        cnt_targets:[batch_size,sum(_h*_w),1]
        reg_targets:[batch_size,sum(_h*_w),4]
        """
        cls_logits = srcs
        gt_boxes = gt_boxes
        classes = batch_classes
        cls_targets_all_level = []
        assert len(self.strides) == len(cls_logits)
        for level in range(len(cls_logits)):
            level_out = cls_logits[level]
            level_targets = self._gen_level_targets(level_out, gt_boxes, classes, self.strides[level], self.limit_range[level])
            cls_targets_all_level.append(level_targets)
        return torch.cat(cls_targets_all_level, dim=1)

    def _gen_level_targets(self, out, gt_boxes, classes, stride, limit_range, sample_radiu_ratio=1.5):
        """
        Args
        out list contains [[batch_size,class_num,h,w],[batch_size,1,h,w],[batch_size,4,h,w]]
        gt_boxes [batch_size,m,4]
        classes [batch_size,m]
        stride int
        limit_range list [min,max]
        Returns
        cls_targets,cnt_targets,reg_targets
        """
        cls_logits = out
        batch_size = cls_logits.shape[0]
        class_num = cls_logits.shape[1]
        m = gt_boxes.shape[1]
        cls_logits = cls_logits.permute(0, 2, 3, 1)
        coords = coords_fmap2orig(cls_logits, stride)
        cls_logits = cls_logits.reshape((batch_size, -1, class_num))
        h_mul_w = cls_logits.shape[1]
        x = coords[:, 0]
        y = coords[:, 1]
        l_off = x[None, :, None] - gt_boxes[..., 0][:, None, :]
        t_off = y[None, :, None] - gt_boxes[..., 1][:, None, :]
        r_off = gt_boxes[..., 2][:, None, :] - x[None, :, None]
        b_off = gt_boxes[..., 3][:, None, :] - y[None, :, None]
        ltrb_off = torch.stack([l_off, t_off, r_off, b_off], dim=-1)
        areas = (ltrb_off[..., 0] + ltrb_off[..., 2]) * (ltrb_off[..., 1] + ltrb_off[..., 3])
        off_min = torch.min(ltrb_off, dim=-1)[0]
        off_max = torch.max(ltrb_off, dim=-1)[0]
        mask_in_gtboxes = off_min > 0
        mask_in_level = (off_max > limit_range[0]) & (off_max <= limit_range[1])
        radiu = stride * sample_radiu_ratio
        gt_center_x = (gt_boxes[..., 0] + gt_boxes[..., 2]) / 2
        gt_center_y = (gt_boxes[..., 1] + gt_boxes[..., 3]) / 2
        c_l_off = x[None, :, None] - gt_center_x[:, None, :]
        c_t_off = y[None, :, None] - gt_center_y[:, None, :]
        c_r_off = gt_center_x[:, None, :] - x[None, :, None]
        c_b_off = gt_center_y[:, None, :] - y[None, :, None]
        c_ltrb_off = torch.stack([c_l_off, c_t_off, c_r_off, c_b_off], dim=-1)
        c_off_max = torch.max(c_ltrb_off, dim=-1)[0]
        mask_center = c_off_max < radiu
        mask_pos = mask_in_gtboxes & mask_in_level
        areas[~mask_pos] = 99999999
        areas_min_ind = torch.min(areas, dim=-1)[1]
        classes = torch.broadcast_tensors(classes[:, None, :], areas.long())[0]
        cls_targets = classes[torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_ind.unsqueeze(dim=-1), 1)]
        cls_targets = torch.reshape(cls_targets, (batch_size, -1, 1))
        assert cls_targets.shape == (batch_size, h_mul_w, 1)
        mask_pos_2 = mask_pos.long().sum(dim=-1)
        mask_pos_2 = mask_pos_2 >= 1
        assert mask_pos_2.shape == (batch_size, h_mul_w)
        cls_targets[~mask_pos_2] = 0
        return cls_targets


def focal_loss_from_logits(preds, targets, gamma=2.0, alpha=0.25):
    """
    Args:
    preds: [n,class_num]
    targets: [n,class_num]
    """
    preds = preds.sigmoid()
    pt = preds * targets + (1.0 - preds) * (1.0 - targets)
    w = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = -w * torch.pow(1.0 - pt, gamma) * pt.log()
    return loss.sum()


def compute_cls_loss(preds, targets, mask):
    """
    Args
    preds: list contains five level pred [batch_size,class_num,_h,_w]
    targets: [batch_size,sum(_h*_w),1]
    mask: [batch_size,sum(_h*_w)]
    """
    batch_size = targets.shape[0]
    class_num = preds[0].shape[1]
    mask = mask.unsqueeze(dim=-1)
    num_pos = torch.sum(mask, dim=[1, 2]).clamp_(min=1).float()
    assert preds.shape[:2] == targets.shape[:2]
    loss = []
    for batch_index in range(batch_size):
        pred_pos = preds[batch_index]
        target_pos = targets[batch_index]
        target_pos = (torch.arange(1, class_num + 1, device=target_pos.device)[None, :] == target_pos).float()
        loss.append(focal_loss_from_logits(pred_pos, target_pos).view(1))
    return torch.cat(loss, dim=0) / num_pos


class TwoStageCriterion(SetCriterion):

    def __init__(self, num_classes, matcher, weight_dict, losses=['class', 'boxes'], eos_coef=None, loss_class_type='focal_loss', alpha: 'float'=0.25, gamma: 'float'=2, two_stage_binary_cls=False):
        super().__init__(num_classes, matcher, weight_dict, losses, eos_coef, loss_class_type, alpha, gamma)
        weight_dict['select_loss'] = 1.5
        self.two_stage_binary_cls = two_stage_binary_cls
        strides = [8, 16, 32, 64]
        limit_range = [[-1, 64], [64, 128], [128, 256], [256, 999999]]
        self.target_layer = GenTargets(strides, limit_range)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        if targets is not None:
            batch_size = len(targets)
            temp_labels = []
            gt_boxx = []
            pad_classes_list = []
            pad_classes90_list = []
            pad_boxes_list = []
            labels = []
            for t in targets:
                h, w = t['size']
                boxes = t['boxes']
                boxes = boxes * torch.tensor([w, h, w, h], dtype=torch.float32, device=boxes.device)
                boxes = box_cxcywh_to_xyxy(boxes)
                temp_labels.append(t['labels'])
                gt_boxx.append(boxes)
            for c in temp_labels:
                c = torch.ones(c.shape, device=c.device)
                labels.append(c)
            max_num = 0
            for i in range(batch_size):
                n = labels[i].shape[0]
                if n > max_num:
                    max_num = n
            for i in range(batch_size):
                pad_boxes_list.append(torch.nn.functional.pad(gt_boxx[i], (0, 0, 0, max_num - gt_boxx[i].shape[0]), value=-1))
                pad_classes_list.append(torch.nn.functional.pad(labels[i], (0, max_num - labels[i].shape[0]), value=-1))
            batch_classes = torch.stack(pad_classes_list)
            batch_boxes = torch.stack(pad_boxes_list)
            class_targets = self.target_layer(outputs['srcs'], batch_boxes, batch_classes)
            t_mask_pos = (class_targets > 0).squeeze(dim=-1)
        backbone_mask_prediction = outputs['temp_backbone_mask_prediction']
        select_loss = compute_cls_loss(backbone_mask_prediction, class_targets, t_mask_pos).mean()
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        t_l_dict = dict()
        t_l_dict['select_loss'] = select_loss
        losses.update(t_l_dict)
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {(k + f'_{i}'): v for k, v in l_dict.items()}
                    losses.update(l_dict)
        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            if self.two_stage_binary_cls:
                for bt in targets:
                    bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes)
                l_dict = {(k + '_enc'): v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses


class MixedMatcher(nn.Module):
    """
    MixedMatcher supports multiple matching startegies as one-to-one matching and one-to-many matching.
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).

    Args:
        cost_class (float): The relative weight of the classification error
            in the matching cost. Default: 1.
        cost_bbox (float): The relative weight of the L1 error of the bounding box
            coordinates in the matching cost. Default: 1.
        cost_giou (float): This is the relative weight of the giou loss of
            the bounding box in the matching cost. Default: 1.
        cost_class_type (str): How the classification error is calculated.
            Choose from ``["ce_cost", "focal_loss_cost"]``. Default: "focal_loss_cost".
        alpha (float): Weighting factor in range (0, 1) to balance positive vs
            negative examples in focal loss. Default: 0.25.
        gamma (float): Exponent of modulating factor (1 - p_t) to balance easy vs
            hard examples in focal loss. Default: 2.
    """

    def __init__(self, cost_class: 'float'=1, cost_bbox: 'float'=1, cost_giou: 'float'=1, cost_class_type: 'str'='focal_loss_cost', alpha: 'float'=0.25, gamma: 'float'=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_class_type = cost_class_type
        self.alpha = alpha
        self.gamma = gamma
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, 'all costs cant be 0'
        assert cost_class_type in {'ce_cost', 'focal_loss_cost'}, 'only support ce loss or focal loss for computing class cost'

    @torch.no_grad()
    def forward(self, outputs, targets, gt_copy=1):
        """Forward function for `HungarianMatcher` which performs the matching.

        Args:
            outputs (Dict[str, torch.Tensor]): This is a dict that contains at least these entries:

                - ``"pred_logits"``: Tensor of shape (bs, num_queries, num_classes) with the classification logits.
                - ``"pred_boxes"``: Tensor of shape (bs, num_queries, 4) with the predicted box coordinates.

            targets (List[Dict[str, torch.Tensor]]): This is a list of targets (len(targets) = batch_size),
                where each target is a dict containing:

                - ``"labels"``: Tensor of shape (num_target_boxes, ) (where num_target_boxes is the number of ground-truth objects in the target) containing the class labels.  # noqa
                - ``"boxes"``: Tensor of shape (num_target_boxes, 4) containing the target box coordinates.

        Returns:
            list[torch.Tensor]: A list of size batch_size, containing tuples of `(index_i, index_j)` where:

                - ``index_i`` is the indices of the selected predictions (in order)
                - ``index_j`` is the indices of the corresponding selected targets (in order)

            For each batch element, it holds: `len(index_i) = len(index_j) = min(num_queries, num_target_boxes)`
        """
        bs, num_queries = outputs['pred_logits'].shape[:2]
        if self.cost_class_type == 'ce_cost':
            out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)
        elif self.cost_class_type == 'focal_loss_cost':
            out_prob = outputs['pred_logits'].flatten(0, 1).sigmoid()
        out_bbox = outputs['pred_boxes'].flatten(0, 1)
        tgt_ids = torch.cat([v['labels'] for v in targets])
        tgt_bbox = torch.cat([v['boxes'] for v in targets])
        if self.cost_class_type == 'ce_cost':
            cost_class = -out_prob[:, tgt_ids]
        elif self.cost_class_type == 'focal_loss_cost':
            alpha = self.alpha
            gamma = self.gamma
            neg_cost_class = (1 - alpha) * out_prob ** gamma * -(1 - out_prob + 1e-08).log()
            pos_cost_class = alpha * (1 - out_prob) ** gamma * -(out_prob + 1e-08).log()
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v['boxes']) for v in targets]
        indices = []
        for i, c in enumerate(C.split(sizes, -1)):
            gt_size = c.size(-1)
            if gt_size > 0:
                gt_copy = min(int(num_queries * 0.5 / gt_size), gt_copy)
            src_ind, tgt_ind = linear_sum_assignment(c[i].repeat(1, gt_copy))
            tgt_ind = tgt_ind % gt_size
            tgt_ind, ind = torch.as_tensor(tgt_ind, dtype=torch.int64).sort()
            src_ind = torch.tensor(src_ind, dtype=torch.int64)[ind].view(-1)
            indices.append([src_ind, tgt_ind])
        return indices


class TransformerEncoder(TransformerLayerSequence):

    def __init__(self, embed_dim: 'int'=256, num_heads: 'int'=8, feedforward_dim: 'int'=1024, attn_dropout: 'float'=0.1, ffn_dropout: 'float'=0.1, num_layers: 'int'=6, post_norm: 'bool'=False, num_feature_levels: 'int'=4):
        super(TransformerEncoder, self).__init__(transformer_layers=BaseTransformerLayer(attn=MultiScaleDeformableAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True, num_levels=num_feature_levels), ffn=FFN(embed_dim=embed_dim, feedforward_dim=feedforward_dim, output_dim=embed_dim, num_fcs=2, ffn_drop=ffn_dropout), norm=nn.LayerNorm(embed_dim), operation_order=('self_attn', 'norm', 'ffn', 'norm')), num_layers=num_layers)
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm
        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(self, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, **kwargs):
        for layer in self.layers:
            query = layer(query, key, value, query_pos=query_pos, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, **kwargs)
        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


def _get_clones(module, N, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def gen_sineembed_for_position(pos_tensor):
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError('Unknown pos_tensor shape(-1):{}'.format(pos_tensor.size(-1)))
    return pos


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, d_model=256, query_dim=4, modulate_hw_attn=True, num_feature_levels=1, deformable_decoder=True, decoder_query_perturber=None, dec_layer_number=None, rm_dec_query_scale=True, dec_layer_share=False, dec_layer_dropout_prob=None):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers, layer_share=dec_layer_share)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, 'support return_intermediate only'
        self.query_dim = query_dim
        assert query_dim in [2, 4], 'query_dim should be 2/4 but {}'.format(query_dim)
        self.num_feature_levels = num_feature_levels
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:
            self.query_pos_sine_scale = None
        if rm_dec_query_scale:
            self.query_scale = None
        else:
            raise NotImplementedError
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.bbox_embed = None
        self.class_embed = None
        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder
        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        else:
            self.ref_anchor_head = None
        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None
        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == num_layers
        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()

    def forward(self, tgt, memory, tgt_mask: 'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, refpoints_unsigmoid: 'Optional[Tensor]'=None, level_start_index: 'Optional[Tensor]'=None, spatial_shapes: 'Optional[Tensor]'=None, valid_ratios: 'Optional[Tensor]'=None):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        output = tgt
        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]
        for layer_id, layer in enumerate(self.layers):
            if self.training and self.decoder_query_perturber is not None and layer_id != 0:
                reference_points = self.decoder_query_perturber(reference_points)
            reference_points_input = reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])
            raw_query_pos = self.ref_point_head(query_sine_embed)
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos
            output = layer(tgt=output, tgt_query_pos=query_pos, tgt_query_sine_embed=query_sine_embed, tgt_key_padding_mask=tgt_key_padding_mask, tgt_reference_points=reference_points_input, memory=memory, memory_key_padding_mask=memory_key_padding_mask, memory_level_start_index=level_start_index, memory_spatial_shapes=spatial_shapes, memory_pos=pos, self_attn_mask=tgt_mask, cross_attn_mask=memory_mask)
            if self.bbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()
                reference_points = new_reference_points.detach()
                ref_points.append(new_reference_points)
            intermediate.append(self.norm(output))
        return [[itm_out.transpose(0, 1) for itm_out in intermediate], [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points]]


class Transformer(nn.Module):
    """Transformer module for AlignDETR
    Args:
        encoder (nn.Module): encoder module.
        decoder (nn.Module): decoder module.
        as_two_stage (bool): whether to use two-stage transformer. Default False.
        num_feature_levels (int): number of feature levels. Default 4.
        two_stage_num_proposals (int): number of proposals in two-stage transformer. Default 900.
    """

    def __init__(self, encoder=None, decoder=None, num_feature_levels=4, two_stage_num_proposals=900, learnt_init_query=True):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals
        self.embed_dim = self.encoder.embed_dim
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dim))
        self.learnt_init_query = learnt_init_query
        if self.learnt_init_query:
            self.tgt_embed = nn.Embedding(self.two_stage_num_proposals, self.embed_dim)
        self.enc_output = nn.Linear(self.embed_dim, self.embed_dim)
        self.enc_output_norm = nn.LayerNorm(self.embed_dim)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.normal_(self.level_embeds)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N, S, C = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:_cur + H * W].view(N, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device), torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * 2.0 ** lvl
            proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
            proposals.append(proposal)
            _cur += H * W
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The ratios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has                 shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device), torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid ratios of feature maps of all levels."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, multi_level_feats, multi_level_masks, multi_level_pos_embeds, query_embed, attn_masks, **kwargs):
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = h, w
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in multi_level_masks], 1)
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=feat.device)
        memory = self.encoder(query=feat_flatten, key=None, value=None, query_pos=lvl_pos_embed_flatten, query_key_padding_mask=mask_flatten, spatial_shapes=spatial_shapes, reference_points=reference_points, level_start_index=level_start_index, valid_ratios=valid_ratios, **kwargs)
        output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
        topk = self.two_stage_num_proposals
        topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]
        topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
        reference_points = topk_coords_unact.detach().sigmoid()
        if query_embed[1] is not None:
            reference_points = torch.cat([query_embed[1].sigmoid(), reference_points], 1)
        init_reference_out = reference_points
        target_unact = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
        if self.learnt_init_query:
            target = self.tgt_embed.weight[None].repeat(bs, 1, 1)
        else:
            target = target_unact.detach()
        if query_embed[0] is not None:
            target = torch.cat([query_embed[0], target], 1)
        inter_states, inter_references = self.decoder(query=target, key=memory, value=memory, query_pos=None, key_padding_mask=mask_flatten, reference_points=reference_points, spatial_shapes=spatial_shapes, level_start_index=level_start_index, valid_ratios=valid_ratios, attn_masks=attn_masks, **kwargs)
        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out, target_unact, topk_coords_unact.sigmoid()


class AnchorDETR(nn.Module):
    """Implement DAB-DETR in `DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR
    <https://arxiv.org/abs/2201.12329>`_

    Args:
        backbone (nn.Module): Backbone module for feature extraction.
        in_features (List[str]): Selected backbone output features for transformer module.
        in_channels (int): Dimension of the last feature in `in_features`.
        position_embedding (nn.Module): Position encoding layer for generating position embeddings.
        transformer (nn.Module): Transformer module used for further processing features and input queries.
        embed_dim (int): Hidden dimension for transformer module.
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        freeze_anchor_box_centers (bool): If True, freeze the center param ``(x, y)`` for
            the initialized dynamic anchor boxes in format ``(x, y, w, h)``
            and only train ``(w, h)``. Default: True.
        select_box_nums_for_evaluation (int): Select the top-k confidence predicted boxes for inference.
            Default: 300.
        device (str): Training device. Default: "cuda".
    """

    def __init__(self, backbone: 'nn.Module', in_features: 'List[str]', in_channels: 'int', transformer: 'nn.Module', embed_dim: 'int', criterion: 'nn.Module', aux_loss: 'bool'=True, pixel_mean: 'List[float]'=[123.675, 116.28, 103.53], pixel_std: 'List[float]'=[58.395, 57.12, 57.375], select_box_nums_for_evaluation: 'int'=100, device: 'str'='cuda'):
        super(AnchorDETR, self).__init__()
        self.backbone = backbone
        self.in_features = in_features
        self.transformer = transformer
        self.aux_loss = aux_loss
        self.criterion = criterion
        self.input_proj = nn.Sequential(nn.Conv2d(in_channels, embed_dim, kernel_size=1), nn.GroupNorm(32, embed_dim))
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation

    def forward(self, batched_inputs):
        """Forward function of `DAB-DETR` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries (anchor boxes in DAB-DETR).
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        images = self.preprocess_image(batched_inputs)
        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]['instances'].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)
        features = self.backbone(images.tensor)[self.in_features[-1]]
        features = self.input_proj(features)
        features = features.unsqueeze(1)
        img_masks = F.interpolate(img_masks[None], size=features.shape[-2:])[0]
        outputs_class, outputs_coord = self.transformer(features, img_masks)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            output['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        if self.training:
            gt_instances = [x['instances'] for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output['pred_logits']
            box_pred = output['pred_boxes']
            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get('height', image_size[0])
                width = input_per_image.get('width', image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({'instances': r})
            return processed_results

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def inference(self, box_cls, box_pred, image_sizes):
        """Inference function for DAB-DETR

        Args:
            box_cls (torch.Tensor): tensor of shape ``(batch_size, num_queries, K)``.
                The tensor predicts the classification probability for each query.
            box_pred (torch.Tensor): tensors of shape ``(batch_size, num_queries, 4)``.
                The tensor predicts 4-vector ``(x, y, w, h)`` box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode='floor')
        labels = topk_indexes % box_cls.shape[2]
        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(scores, labels, boxes, image_sizes)):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({'labels': gt_classes, 'boxes': gt_boxes})
        return new_targets

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x['image']) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images


def multi_head_rcda_forward(query_row: 'torch.Tensor', query_col: 'torch.Tensor', key_row: 'torch.Tensor', key_col: 'torch.Tensor', value: 'torch.Tensor', embed_dim_to_check: 'int', num_heads: 'int', in_proj_weight: 'torch.Tensor', in_proj_bias: 'torch.Tensor', bias_k_row: 'Optional[torch.Tensor]', bias_k_col: 'Optional[torch.Tensor]', bias_v: 'Optional[torch.Tensor]', add_zero_attn: 'bool', dropout_p: 'float', out_proj_weight: 'torch.Tensor', out_proj_bias: 'torch.Tensor', training: 'bool'=True, key_padding_mask: 'Optional[torch.Tensor]'=None, need_weights: 'bool'=True, attn_mask: 'Optional[torch.Tensor]'=None, use_separate_proj_weight: 'bool'=False, q_row_proj_weight: 'Optional[torch.Tensor]'=None, q_col_proj_weight: 'Optional[torch.Tensor]'=None, k_row_proj_weight: 'Optional[torch.Tensor]'=None, k_col_proj_weight: 'Optional[torch.Tensor]'=None, v_proj_weight: 'Optional[torch.Tensor]'=None, static_k: 'Optional[torch.Tensor]'=None, static_v: 'Optional[torch.Tensor]'=None) ->Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Args:
        query_row, query_col, key_row, key_col, value: map a query and a set of key-value pairs to an output.
            See "Anchor DETR: Query Design for Transformer-Based Detector" for more details.
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
        attn_mask: 2D or 3D mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer). A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_row_proj_weight, q_col_proj_weight, k_row_proj_weight, k_col_proj_weight, v_proj_weight.
        q_row_proj_weight, q_col_proj_weight, k_row_proj_weight, k_col_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query_row: :math:`(N, L, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - query_col: :math:`(N, L, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key_row: :math:`(N, H, W, E)`, where W is the source sequence row length, N is the batch size, E is
          the embedding dimension.
        - key_col: :math:`(N, H, W, E)`, where H is the source sequence column length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(N, H, W, E)` where HW is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, H, W)`, ByteTensor, where N is the batch size, HW is the source sequence length.
        - attn_mask: Not Implemented
        - static_k: Not Implemented
        - static_v: Not Implemented

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, HW)` where N is the batch size,
          L is the target sequence length, HW is the source sequence length.
    """
    bsz, tgt_len, embed_dim = query_row.size()
    src_len_row = key_row.size()[2]
    src_len_col = key_col.size()[1]
    assert embed_dim == embed_dim_to_check
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, 'embed_dim must be divisible by num_heads'
    scaling = float(head_dim) ** -0.5
    _b = in_proj_bias
    _start = 0
    _end = embed_dim
    _w = in_proj_weight[_start:_end, :]
    if _b is not None:
        _b = _b[_start:_end]
    q_row = F.linear(query_row, _w, _b)
    _b = in_proj_bias
    _start = embed_dim * 1
    _end = embed_dim * 2
    _w = in_proj_weight[_start:_end, :]
    if _b is not None:
        _b = _b[_start:_end]
    q_col = F.linear(query_col, _w, _b)
    _b = in_proj_bias
    _start = embed_dim * 2
    _end = embed_dim * 3
    _w = in_proj_weight[_start:_end, :]
    if _b is not None:
        _b = _b[_start:_end]
    k_row = F.linear(key_row, _w, _b)
    _b = in_proj_bias
    _start = embed_dim * 3
    _end = embed_dim * 4
    _w = in_proj_weight[_start:_end, :]
    if _b is not None:
        _b = _b[_start:_end]
    k_col = F.linear(key_col, _w, _b)
    _b = in_proj_bias
    _start = embed_dim * 4
    _end = None
    _w = in_proj_weight[_start:, :]
    if _b is not None:
        _b = _b[_start:]
    v = F.linear(value, _w, _b)
    q_row = q_row.transpose(0, 1)
    q_col = q_col.transpose(0, 1)
    k_row = k_row.mean(1).transpose(0, 1)
    k_col = k_col.mean(2).transpose(0, 1)
    q_row = q_row * scaling
    q_col = q_col * scaling
    q_row = q_row.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    q_col = q_col.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k_row is not None:
        k_row = k_row.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if k_col is not None:
        k_col = k_col.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().permute(1, 2, 0, 3).reshape(src_len_col, src_len_row, bsz * num_heads, head_dim).permute(2, 0, 1, 3)
    attn_output_weights_row = torch.bmm(q_row, k_row.transpose(1, 2))
    attn_output_weights_col = torch.bmm(q_col, k_col.transpose(1, 2))
    assert list(attn_output_weights_row.size()) == [bsz * num_heads, tgt_len, src_len_row]
    assert list(attn_output_weights_col.size()) == [bsz * num_heads, tgt_len, src_len_col]
    if key_padding_mask is not None:
        mask_row = key_padding_mask[:, 0, :].unsqueeze(1).unsqueeze(2)
        mask_col = key_padding_mask[:, :, 0].unsqueeze(1).unsqueeze(2)
        attn_output_weights_row = attn_output_weights_row.view(bsz, num_heads, tgt_len, src_len_row)
        attn_output_weights_col = attn_output_weights_col.view(bsz, num_heads, tgt_len, src_len_col)
        attn_output_weights_row = attn_output_weights_row.masked_fill(mask_row, float('-inf'))
        attn_output_weights_col = attn_output_weights_col.masked_fill(mask_col, float('-inf'))
        attn_output_weights_row = attn_output_weights_row.view(bsz * num_heads, tgt_len, src_len_row)
        attn_output_weights_col = attn_output_weights_col.view(bsz * num_heads, tgt_len, src_len_col)
    attn_output_weights_col = F.softmax(attn_output_weights_col, dim=-1)
    attn_output_weights_row = F.softmax(attn_output_weights_row, dim=-1)
    attn_output_weights_col = F.dropout(attn_output_weights_col, p=dropout_p, training=training)
    attn_output_weights_row = F.dropout(attn_output_weights_row, p=dropout_p, training=training)
    efficient_compute = True
    if efficient_compute:
        if src_len_col < src_len_row:
            b_ein, q_ein, w_ein = attn_output_weights_row.shape
            b_ein, h_ein, w_ein, c_ein = v.shape
            attn_output_row = torch.matmul(attn_output_weights_row, v.permute(0, 2, 1, 3).reshape(b_ein, w_ein, h_ein * c_ein)).reshape(b_ein, q_ein, h_ein, c_ein).permute(0, 2, 1, 3)
            attn_output = torch.matmul(attn_output_weights_col.permute(1, 0, 2)[:, :, None, :], attn_output_row.permute(2, 0, 1, 3)).squeeze(-2).reshape(tgt_len, bsz, embed_dim)
        else:
            b_ein, q_ein, h_ein = attn_output_weights_col.shape
            b_ein, h_ein, w_ein, c_ein = v.shape
            attn_output_col = torch.matmul(attn_output_weights_col, v.reshape(b_ein, h_ein, w_ein * c_ein)).reshape(b_ein, q_ein, w_ein, c_ein)
            attn_output = torch.matmul(attn_output_weights_row[:, :, None, :], attn_output_col).squeeze(-2).permute(1, 0, 2).reshape(tgt_len, bsz, embed_dim)
    else:
        b_ein, q_ein, h_ein = attn_output_weights_col.shape
        b_ein, h_ein, w_ein, c_ein = v.shape
        attn_output_col = torch.matmul(attn_output_weights_col, v.reshape(b_ein, h_ein, w_ein * c_ein)).reshape(b_ein, q_ein, w_ein, c_ein)
        attn_output = torch.matmul(attn_output_weights_row[:, :, None, :], attn_output_col).squeeze(-2).permute(1, 0, 2).reshape(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    if need_weights:
        return attn_output, torch.einsum('bqw,bqh->qbhw', attn_output_weights_row, attn_output_weights_col).reshape(tgt_len, bsz, num_heads, src_len_col, src_len_row).mean(2)
    else:
        return attn_output, None


class MultiheadRCDA(Module):
    """Allows the model to jointly attend to information
    from different representation subspaces.
    See reference:
        Anchor DETR: Query Design for Transformer-Based Detector

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
        >>> multihead_attn = MultiheadRCDA(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query_row, query_col, key_row, key_col, value)
    """
    __annotations__ = {'bias_k_row': torch._jit_internal.Optional[torch.Tensor], 'bias_k_col': torch._jit_internal.Optional[torch.Tensor], 'bias_v': torch._jit_internal.Optional[torch.Tensor]}
    __constants__ = ['q_row_proj_weight', 'q_col_proj_weight', 'k_row_proj_weight', 'k_col_proj_weight', 'v_proj_weight', 'in_proj_weight']

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadRCDA, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        if self._qkv_same_embed_dim is False:
            self.q_row_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.q_col_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_row_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.k_col_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(5 * embed_dim, embed_dim))
            self.register_parameter('q_row_proj_weight', None)
            self.register_parameter('q_col_proj_weight', None)
            self.register_parameter('k_row_proj_weight', None)
            self.register_parameter('k_col_proj_weight', None)
            self.register_parameter('v_proj_weight', None)
        if bias:
            self.in_proj_bias = Parameter(torch.empty(5 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        if add_bias_kv:
            self.bias_k_row = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_k_col = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k_row = self.bias_k_col = self.bias_v = None
        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            init.xavier_uniform_(self.in_proj_weight)
        else:
            init.xavier_uniform_(self.q_row_proj_weight)
            init.xavier_uniform_(self.q_col_proj_weight)
            init.xavier_uniform_(self.k_row_proj_weight)
            init.xavier_uniform_(self.k_col_proj_weight)
            init.xavier_uniform_(self.v_proj_weight)
        if self.in_proj_bias is not None:
            init.constant_(self.in_proj_bias, 0.0)
            init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k_row is not None:
            init.xavier_normal_(self.bias_k_row)
        if self.bias_k_col is not None:
            init.xavier_normal_(self.bias_k_col)
        if self.bias_v is not None:
            init.xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True
        super(MultiheadRCDA, self).__setstate__(state)

    def forward(self, query_row, query_col, key_row, key_col, value, key_padding_mask=None, need_weights=False, attn_mask=None):
        """
    Args:
        query_row, query_col, key_row, key_col, value: map a query and a set of key-value pairs to an output.
            See "Anchor DETR: Query Design for Transformer-Based Detector" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer). A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shape:
        - Inputs:
        - query_row: :math:`(N, L, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - query_col: :math:`(N, L, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key_row: :math:`(N, H, W, E)`, where W is the source sequence row length, N is the batch size, E is
          the embedding dimension.
        - key_col: :math:`(N, H, W, E)`, where H is the source sequence column length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(N, H, W, E)` where HW is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, H, W)`, ByteTensor, where N is the batch size, HW is the source sequence length.
        - attn_mask: Not Implemented
        - static_k: Not Implemented
        - static_v: Not Implemented

        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, HW)` where N is the batch size,
          L is the target sequence length, HW is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_rcda_forward(query_row, query_col, key_row, key_col, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k_row, self.bias_k_col, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, use_separate_proj_weight=True, q_row_proj_weight=self.q_row_proj_weight, q_col_proj_weight=self.q_col_proj_weight, k_row_proj_weight=self.k_row_proj_weight, k_col_proj_weight=self.k_col_proj_weight, v_proj_weight=self.v_proj_weight)
        else:
            return multi_head_rcda_forward(query_row, query_col, key_row, key_col, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k_row, self.bias_k_col, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask)


def pos2posemb1d(pos, num_pos_feats=256, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
    pos_x = pos[..., None] / dim_t
    posemb = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    return posemb


def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


class TransformerDecoderLayer(nn.Module):

    def __init__(self, embed_dim=256, d_ffn=1024, dropout=0.0, activation='relu', n_heads=8, n_levels=3, attention_type='RCDA'):
        super().__init__()
        self.attention_type = attention_type
        self.attention_type = attention_type
        if attention_type == 'RCDA':
            attention_module = MultiheadRCDA
        elif attention_type == 'nn.MultiheadAttention':
            attention_module = nn.MultiheadAttention
        else:
            raise ValueError(f'unknown {attention_type} attention_type')
        self.cross_attn = attention_module(embed_dim, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        if n_levels > 1:
            self.level_fc = nn.Linear(embed_dim * n_levels, embed_dim)
        self.ffn = FFN(embed_dim, d_ffn, dropout, activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, reference_points, srcs, src_padding_masks=None, adapt_pos2d=None, adapt_pos1d=None, posemb_row=None, posemb_col=None, posemb_2d=None):
        tgt_len = tgt.shape[1]
        query_pos = adapt_pos2d(pos2posemb2d(reference_points))
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        bz, l, c, h, w = srcs.shape
        srcs = srcs.reshape(bz * l, c, h, w).permute(0, 2, 3, 1)
        if self.attention_type == 'RCDA':
            query_pos_x = adapt_pos1d(pos2posemb1d(reference_points[..., 0]))
            query_pos_y = adapt_pos1d(pos2posemb1d(reference_points[..., 1]))
            posemb_row = posemb_row.unsqueeze(1).repeat(1, h, 1, 1)
            posemb_col = posemb_col.unsqueeze(2).repeat(1, 1, w, 1)
            src_row = src_col = srcs
            k_row = src_row + posemb_row
            k_col = src_col + posemb_col
            tgt2 = self.cross_attn((tgt + query_pos_x).repeat(l, 1, 1), (tgt + query_pos_y).repeat(l, 1, 1), k_row, k_col, srcs, key_padding_mask=src_padding_masks)[0].transpose(0, 1)
        else:
            tgt2 = self.cross_attn((tgt + query_pos).repeat(l, 1, 1).transpose(0, 1), (srcs + posemb_2d).reshape(bz * l, h * w, c).transpose(0, 1), srcs.reshape(bz * l, h * w, c).transpose(0, 1), key_padding_mask=src_padding_masks.reshape(bz * l, h * w))[0].transpose(0, 1)
        if l > 1:
            tgt2 = self.level_fc(tgt2.reshape(bz, l, tgt_len, c).permute(0, 2, 3, 1).reshape(bz, tgt_len, c * l))
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt = self.ffn(tgt)
        return tgt


class TransformerEncoderLayerLevel(nn.Module):

    def __init__(self, embed_dim=256, d_ffn=1024, dropout=0.0, activation='relu', n_heads=8):
        super().__init__()
        self.self_attn_level = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, d_ffn, dropout, activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, level_emb=0):
        bz, c, h, w = src.shape
        src = src.permute(0, 2, 3, 1)
        src2 = self.self_attn_level(src.reshape(bz, h * w, c) + level_emb, src.reshape(bz, h * w, c) + level_emb, src.reshape(bz, h * w, c))[0].reshape(bz, h, w, c)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.ffn(src)
        src = src.permute(0, 3, 1, 2)
        return src


class TransformerEncoderLayerSpatial(nn.Module):

    def __init__(self, embed_dim=256, d_ffn=1024, dropout=0.0, activation='relu', n_heads=8, attention_type='RCDA'):
        super().__init__()
        self.attention_type = attention_type
        if attention_type == 'RCDA':
            attention_module = MultiheadRCDA
        elif attention_type == 'nn.MultiheadAttention':
            attention_module = nn.MultiheadAttention
        else:
            raise ValueError(f'unknown {attention_type} attention_type')
        self.self_attn = attention_module(embed_dim, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, d_ffn, dropout, activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, padding_mask=None, posemb_row=None, posemb_col=None, posemb_2d=None):
        bz, c, h, w = src.shape
        src = src.permute(0, 2, 3, 1)
        if self.attention_type == 'RCDA':
            posemb_row = posemb_row.unsqueeze(1).repeat(1, h, 1, 1)
            posemb_col = posemb_col.unsqueeze(2).repeat(1, 1, w, 1)
            src2 = self.self_attn((src + posemb_row).reshape(bz, h * w, c), (src + posemb_col).reshape(bz, h * w, c), src + posemb_row, src + posemb_col, src, key_padding_mask=padding_mask)[0].transpose(0, 1).reshape(bz, h, w, c)
        else:
            src2 = self.self_attn((src + posemb_2d).reshape(bz, h * w, c).transpose(0, 1), (src + posemb_2d).reshape(bz, h * w, c).transpose(0, 1), src.reshape(bz, h * w, c).transpose(0, 1), key_padding_mask=padding_mask.reshape(bz, h * w))[0].transpose(0, 1).reshape(bz, h, w, c)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.ffn(src)
        src = src.permute(0, 3, 1, 2)
        return src


def mask2pos(mask):
    not_mask = ~mask
    y_embed = not_mask[:, :, 0].cumsum(1, dtype=torch.float32)
    x_embed = not_mask[:, 0, :].cumsum(1, dtype=torch.float32)
    y_embed = (y_embed - 0.5) / y_embed[:, -1:]
    x_embed = (x_embed - 0.5) / x_embed[:, -1:]
    return y_embed, x_embed


class AnchorDETRTransformer(nn.Module):

    def __init__(self, embed_dim=256, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.0, activation='relu', num_feature_levels=1, num_query_position=300, num_query_pattern=3, spatial_prior='learned', attention_type='RCDA', num_classes=80):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention_type = attention_type
        encoder_layer = TransformerEncoderLayerSpatial(embed_dim, dim_feedforward, dropout, activation, num_heads, attention_type)
        encoder_layer_level = TransformerEncoderLayerLevel(embed_dim, dim_feedforward, dropout, activation, num_heads)
        decoder_layer = TransformerDecoderLayer(embed_dim, dim_feedforward, dropout, activation, num_heads, num_feature_levels, attention_type)
        if num_feature_levels == 1:
            self.num_encoder_layers_level = 0
        else:
            self.num_encoder_layers_level = num_encoder_layers // 2
        self.num_encoder_layers_spatial = num_encoder_layers - self.num_encoder_layers_level
        self.encoder_layers = _get_clones(encoder_layer, self.num_encoder_layers_spatial)
        self.encoder_layers_level = _get_clones(encoder_layer_level, self.num_encoder_layers_level)
        self.decoder_layers = _get_clones(decoder_layer, num_decoder_layers)
        self.spatial_prior = spatial_prior
        if num_feature_levels > 1:
            self.level_embed = nn.Embedding(num_feature_levels, embed_dim)
        self.num_pattern = num_query_pattern
        self.pattern = nn.Embedding(self.num_pattern, embed_dim)
        self.num_position = num_query_position
        if self.spatial_prior == 'learned':
            self.position = nn.Embedding(self.num_position, 2)
        self.adapt_pos2d = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.adapt_pos1d = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.num_layers = num_decoder_layers
        self.num_classes = num_classes
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self._reset_parameters()

    def _reset_parameters(self):
        num_pred = self.num_layers
        num_classes = self.num_classes
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        if self.spatial_prior == 'learned':
            nn.init.uniform_(self.position.weight.data, 0, 1)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
        self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
        self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])

    def forward(self, srcs, masks):
        bs, l, c, h, w = srcs.shape
        if self.spatial_prior == 'learned':
            reference_points = self.position.weight.unsqueeze(0).repeat(bs, self.num_pattern, 1)
        elif self.spatial_prior == 'grid':
            nx = ny = round(math.sqrt(self.num_position))
            self.num_position = nx * ny
            x = (torch.arange(nx) + 0.5) / nx
            y = (torch.arange(ny) + 0.5) / ny
            xy = torch.meshgrid(x, y)
            reference_points = torch.cat([xy[0].reshape(-1)[..., None], xy[1].reshape(-1)[..., None]], -1)
            reference_points = reference_points.unsqueeze(0).repeat(bs, self.num_pattern, 1)
        else:
            raise ValueError(f'unknown {self.spatial_prior} spatial prior')
        tgt = self.pattern.weight.reshape(1, self.num_pattern, 1, c).repeat(bs, 1, self.num_position, 1).reshape(bs, self.num_pattern * self.num_position, c)
        mask = masks.unsqueeze(1).repeat(1, l, 1, 1).reshape(bs * l, h, w)
        pos_col, pos_row = mask2pos(mask)
        if self.attention_type == 'RCDA':
            posemb_row = self.adapt_pos1d(pos2posemb1d(pos_row))
            posemb_col = self.adapt_pos1d(pos2posemb1d(pos_col))
            posemb_2d = None
        else:
            pos_2d = torch.cat([pos_row.unsqueeze(1).repeat(1, h, 1).unsqueeze(-1), pos_col.unsqueeze(2).repeat(1, 1, w).unsqueeze(-1)], dim=-1)
            posemb_2d = self.adapt_pos2d(pos2posemb2d(pos_2d))
            posemb_row = posemb_col = None
        outputs = srcs.reshape(bs * l, c, h, w)
        for idx in range(len(self.encoder_layers)):
            outputs = self.encoder_layers[idx](outputs, mask, posemb_row, posemb_col, posemb_2d)
        srcs = outputs.reshape(bs, l, c, h, w)
        output = tgt
        outputs_classes = []
        outputs_coords = []
        for lid, layer in enumerate(self.decoder_layers):
            output = layer(output, reference_points, srcs, mask, adapt_pos2d=self.adapt_pos2d, adapt_pos1d=self.adapt_pos1d, posemb_row=posemb_row, posemb_col=posemb_col, posemb_2d=posemb_2d)
            reference = inverse_sigmoid(reference_points)
            outputs_class = self.class_embed[lid](output)
            tmp = self.bbox_embed[lid](output)
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class[None,])
            outputs_coords.append(outputs_coord[None,])
        output = torch.cat(outputs_classes, dim=0), torch.cat(outputs_coords, dim=0)
        return output


class HungarianMatcherGroup(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: 'float'=1, cost_bbox: 'float'=1, cost_giou: 'float'=1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, 'all costs cant be 0'

    def forward(self, outputs, targets, use_focal=True, g_size=1):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs['pred_logits'].shape[:2]
            if use_focal:
                out_prob = outputs['pred_logits'].flatten(0, 1).sigmoid()
            else:
                out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)
            out_bbox = outputs['pred_boxes'].flatten(0, 1)
            if isinstance(targets[0], Instances):
                tgt_ids = torch.cat([gt_per_img.labels for gt_per_img in targets])
                tgt_bbox = torch.cat([gt_per_img.boxes for gt_per_img in targets])
            else:
                tgt_ids = torch.cat([v['labels'] for v in targets])
                tgt_bbox = torch.cat([v['boxes'] for v in targets])
            if use_focal:
                alpha = 0.25
                gamma = 2.0
                neg_cost_class = (1 - alpha) * out_prob ** gamma * -(1 - out_prob + 1e-08).log()
                pos_cost_class = alpha * (1 - out_prob) ** gamma * -(out_prob + 1e-08).log()
                cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            else:
                cost_class = -out_prob[:, tgt_ids]
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C2 = C.view(bs, num_queries // g_size, g_size, -1).cpu()
            if C2.shape[-1] > 0:
                Cmin, _ = C2.max(dim=2)
                if isinstance(targets[0], Instances):
                    sizes = [len(gt_per_img.boxes) for gt_per_img in targets]
                else:
                    sizes = [len(v['boxes']) for v in targets]
                indices = [linear_sum_assignment(c[i]) for i, c in enumerate(Cmin.split(sizes, -1))]
                gindices = []
                for ind in indices:
                    Cindx = np.arange(num_queries).reshape(num_queries // g_size, g_size)
                    gindices.append((Cindx[ind[0]].reshape(-1), ind[1].repeat(g_size)))
            else:
                gindices = [(np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in gindices]


def _filter_predictions_with_area(predictions, area_threshold=100):
    if 'track_instances' in predictions:
        preds = predictions['track_instances']
        wh = preds.boxes[:, 2:4] - preds.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep_idxs = areas > area_threshold
        predictions = copy(predictions)
        predictions['track_instances'] = preds[keep_idxs]
    return predictions


def _filter_predictions_with_confidence(predictions, confidence_threshold=0.5):
    if 'track_instances' in predictions:
        preds = predictions['track_instances']
        keep_idxs = preds.scores > confidence_threshold
        predictions = copy(predictions)
        predictions['track_instances'] = preds[keep_idxs]
    return predictions


class MOT(nn.Module):
    """ Implement CO-MOT: Bridging the Gap Between End-to-end and Non-End-to-end Multi-Object Tracking
    Args:
        backbone (nn.Module): backbone module
        position_embedding (nn.Module): position embedding module
        neck (nn.Module): neck module to handle the intermediate outputs features
        transformer (nn.Module): transformer module
        embed_dim (int): dimension of embedding
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 300.
        device (str): Training device. Default: "cuda".
    """

    def __init__(self, backbone: 'nn.Module', position_embedding: 'nn.Module', neck: 'nn.Module', transformer: 'nn.Module', embed_dim: 'int', num_classes: 'int', num_queries: 'int', criterion: 'nn.Module', track_embed: 'nn.Module', track_base: 'nn.Module', post_process: 'nn.Module', aux_loss: 'bool'=True, device='cuda', g_size=1):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.neck = neck
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.transformer = transformer
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        self.criterion = criterion
        self.device = device
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)
        num_pred = transformer.decoder.num_layers
        self.class_embed = nn.ModuleList([deepcopy(self.class_embed) for i in range(num_pred)])
        self.bbox_embed = nn.ModuleList([deepcopy(self.bbox_embed) for i in range(num_pred)])
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.bbox_embed = self.bbox_embed
        for bbox_embed_layer in self.bbox_embed:
            nn.init.constant_(bbox_embed_layer.layers[-1].bias.data[2:], 0.0)
        self.track_embed = track_embed
        self.post_process = post_process
        self.track_base = track_base
        self.g_size = g_size
        self.position = nn.Embedding(num_queries, 4)
        self.position_offset = nn.Embedding(num_queries * g_size, 4)
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.query_embed_offset = nn.Embedding(num_queries * g_size, embed_dim)
        nn.init.uniform_(self.position.weight.data, 0, 1)
        nn.init.normal_(self.position_offset.weight.data, 0, 1e-05)
        nn.init.normal_(self.query_embed_offset.weight.data, 0, 1e-05)

    def _generate_empty_tracks(self, g_size=1, batch_size=1):
        track_instances = Instances((1, 1))
        num_queries, d_model = self.query_embed.weight.shape
        device = self.query_embed.weight.device
        track_instances.ref_pts = self.position.weight.view(-1, 1, 4).repeat(1, g_size, 1).view(-1, 4) + self.position_offset.weight
        track_instances.query_pos = self.query_embed.weight.view(-1, 1, d_model).repeat(1, g_size, 1).view(-1, d_model) + self.query_embed_offset.weight
        track_instances.ref_pts = track_instances.ref_pts.view(-1, 1, 4).repeat(1, batch_size, 1)
        track_instances.query_pos = track_instances.query_pos.view(-1, 1, d_model).repeat(1, batch_size, 1)
        track_instances.output_embedding = torch.zeros((len(track_instances), batch_size, d_model), device=device)
        track_instances.obj_idxes = torch.full((len(track_instances), batch_size), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full((len(track_instances), batch_size), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros((len(track_instances), batch_size), dtype=torch.long, device=device)
        track_instances.iou = torch.zeros((len(track_instances), batch_size), dtype=torch.float, device=device)
        track_instances.scores = torch.zeros((len(track_instances), batch_size), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros((len(track_instances), batch_size), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros((len(track_instances), batch_size, 4), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros((len(track_instances), batch_size, self.num_classes), dtype=torch.float, device=device)
        track_instances.group_ids = torch.arange(g_size, dtype=torch.long, device=device).repeat(num_queries).view(-1, 1).repeat(1, batch_size)
        track_instances.labels = torch.full((len(track_instances), batch_size), -1, dtype=torch.long, device=device)
        return track_instances

    def clear(self):
        self.track_base.clear()

    def _forward_single_image(self, samples, track_instances, gtboxes=None):
        """Forward function of `MOT`.
        """
        features = self.backbone(samples.tensors)
        img_masks = samples.mask
        multi_level_feats = self.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(F.interpolate(img_masks[None].float(), size=feat.shape[-2:]).squeeze(0))
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))
        input_query_label = track_instances.query_pos
        input_query_bbox = track_instances.ref_pts
        attn_mask = None
        inter_states, init_reference, inter_references, enc_state, enc_reference = self.transformer(multi_level_feats, multi_level_masks, multi_level_position_embeddings, input_query_label, ref_pts=input_query_bbox, attn_mask=attn_mask)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            output['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        output['hs'] = inter_states[-1]
        return output

    def _post_process_single_image(self, frame_res, track_instances, is_last):
        with torch.no_grad():
            if self.training:
                track_scores = frame_res['pred_logits'].sigmoid().max(dim=-1).values
            else:
                track_scores = frame_res['pred_logits'].sigmoid().max(dim=-1).values
        track_instances.scores = track_scores.transpose(0, 1)
        track_instances.pred_logits = frame_res['pred_logits'].transpose(0, 1)
        track_instances.pred_boxes = frame_res['pred_boxes'].transpose(0, 1)
        track_instances.output_embedding = frame_res['hs'].transpose(0, 1)
        if self.training:
            frame_res['track_instances'] = track_instances
            track_instances = self.criterion.match_for_single_frame(frame_res)
        else:
            self.track_base.update(track_instances, g_size=self.g_size)
        tmp = {}
        tmp['track_instances'] = track_instances
        if not is_last:
            out_track_instances = self.track_embed(tmp, g_size=self.g_size)
            frame_res['track_instances'] = out_track_instances
        else:
            frame_res['track_instances'] = None
        return frame_res

    @torch.no_grad()
    def inference_single_image(self, img, ori_img_size, track_instances=None):
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)
        if track_instances is None:
            track_instances = self._generate_empty_tracks(g_size=self.g_size)
        else:
            track_instances = Instances.cat([self._generate_empty_tracks(g_size=self.g_size), track_instances])
        res = self._forward_single_image(img, track_instances=track_instances)
        res = self._post_process_single_image(res, track_instances, False)
        track_instances = res['track_instances']
        track_instances = self.post_process(track_instances, ori_img_size)
        ret = {'track_instances': track_instances}
        if 'ref_pts' in res:
            ref_pts = res['ref_pts']
            img_h, img_w = ori_img_size
            scale_fct = torch.Tensor([img_w, img_h])
            ref_pts = ref_pts * scale_fct[None]
            ret['ref_pts'] = ref_pts
        return ret

    def forward(self, data: 'dict'):

        def fn(frame, gtboxes, track_instances):
            frame = nested_tensor_from_tensor_list(frame)
            frame_res = self._forward_single_image(frame, track_instances, gtboxes)
            return frame_res
        track_instances = None
        if self.training:
            self.criterion.initialize_for_single_clip(data['gt_instances'])
            frames = data['imgs']
            outputs = {'pred_logits': [], 'pred_boxes': []}
            for frame_index, (frame, gt) in enumerate(zip(frames, data['gt_instances'])):
                for f in frame:
                    f.requires_grad = False
                is_last = frame_index == len(frames) - 1
                nbatch = len(frame)
                gtboxes = None
                if track_instances is None:
                    track_instances = self._generate_empty_tracks(g_size=self.g_size, batch_size=nbatch)
                else:
                    track_instances = Instances.cat([self._generate_empty_tracks(g_size=self.g_size, batch_size=nbatch), track_instances])
                if frame_index < len(frames) - 1:
                    args = [frame, gtboxes, track_instances]
                    params = tuple(p for p in self.parameters() if p.requires_grad)
                    frame_res = checkpoint.CheckpointFunction.apply(fn, len(args), *args, *params)
                else:
                    frame = nested_tensor_from_tensor_list(frame)
                    frame_res = self._forward_single_image(frame, track_instances, gtboxes)
                frame_res = self._post_process_single_image(frame_res, track_instances, is_last)
                track_instances = frame_res['track_instances']
                outputs['pred_logits'].append(frame_res['pred_logits'])
                outputs['pred_boxes'].append(frame_res['pred_boxes'])
            outputs['losses_dict'] = self.criterion.losses_dict
            loss_dict = self.criterion(outputs, data)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            assert len(data) == 1
            device = self.device
            outputs = []
            for i, data_ in enumerate(data[0]['data_loader']):
                cur_img, ori_img, proposals, f_path = [d[0] for d in data_]
                cur_img = cur_img
                if track_instances is not None:
                    track_instances.remove('boxes')
                seq_h, seq_w, _ = ori_img.shape
                try:
                    res = self.inference_single_image(cur_img, (seq_h, seq_w), track_instances)
                except:
                    res = self.inference_single_image(cur_img, (seq_h, seq_w), track_instances)
                track_instances = res['track_instances']
                predictions = deepcopy(res)
                if len(predictions['track_instances']):
                    scores = predictions['track_instances'].scores.reshape(-1, self.g_size)
                    keep_idxs = torch.arange(len(predictions['track_instances']), device=scores.device).reshape(-1, self.g_size)
                    keep_idxs = keep_idxs.gather(1, scores.max(-1)[1].reshape(-1, 1)).reshape(-1)
                    predictions['track_instances'] = predictions['track_instances'][keep_idxs]
                predictions = _filter_predictions_with_confidence(predictions, 0.5)
                predictions = _filter_predictions_with_area(predictions)
                outputs.append(predictions['track_instances'])
            return [outputs]

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class ClipMatcher(SetCriterion):
    """This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses: 'List[str]'=['class', 'boxes'], eos_coef: 'float'=0.1, loss_class_type: 'str'='focal_loss', alpha: 'float'=0.25, gamma: 'float'=2.0, g_size=1):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__(num_classes, matcher, weight_dict, losses, eos_coef, loss_class_type, alpha, gamma)
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_loss = True
        self.losses_dict = defaultdict(float)
        self._current_frame_idx = 0
        self.g_size = g_size

    def initialize_for_single_clip(self, gt_instances: 'List[Instances]'):
        self.gt_instances = gt_instances
        self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = defaultdict(float)

    def _step(self):
        self._current_frame_idx += 1

    def calc_loss_for_track_scores(self, track_instances: 'Instances'):
        gt_instances_i = self.gt_instances[self._current_frame_idx]
        outputs = {'pred_logits': track_instances.track_scores[None]}
        device = track_instances.track_scores.device
        num_tracks = len(track_instances)
        src_idx = torch.arange(num_tracks, dtype=torch.long, device=device)
        tgt_idx = track_instances.matched_gt_idxes
        track_losses = self.get_loss('labels', outputs=outputs, gt_instances=[gt_instances_i], indices=[(src_idx, tgt_idx)], num_boxes=1)
        self.losses_dict.update({'frame_{}_track_{}'.format(self._current_frame_idx, key): value for key, value in track_losses.items()})

    def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

    def loss_labels(self, outputs, gt_instances: 'List[Instances]', indices, num_boxes, log=False):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        labels = []
        for gt_per_img, (_, J) in zip(gt_instances, indices):
            labels_per_img = torch.ones_like(J) * self.num_classes
            if len(gt_per_img) > 0:
                labels_per_img[J != -1] = gt_per_img.labels[J[J != -1]]
            labels.append(labels_per_img)
        target_classes_o = torch.cat(labels)
        target_classes[idx] = target_classes_o
        if self.loss_class_type == 'ce_loss':
            loss_class = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        elif self.loss_class_type == 'focal_loss':
            gt_labels_target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[:, :, :-1]
            gt_labels_target = gt_labels_target
            loss_class = sigmoid_focal_loss(src_logits.flatten(1), gt_labels_target.flatten(1), num_boxes=num_boxes, alpha=self.alpha, gamma=self.gamma)
        losses = {'loss_ce': loss_class}
        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, gt_instances: 'List[Instances]', indices: 'List[tuple]', num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([gt_per_img.boxes[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)
        target_obj_ids = torch.cat([gt_per_img.obj_ids[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)
        mask = target_obj_ids != -1
        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes[mask], reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes[mask]), box_cxcywh_to_xyxy(target_boxes[mask])))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {'labels': self.loss_labels, 'boxes': self.loss_boxes}
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def match_for_single_frame(self, outputs: 'dict'):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        gt_instances_i = self.gt_instances[self._current_frame_idx]
        track_instances: 'Instances' = outputs_without_aux['track_instances']
        pred_logits_i = track_instances.pred_logits
        pred_boxes_i = track_instances.pred_boxes
        if not (track_instances.obj_idxes != -1).any():
            outputs_i = {'pred_logits': pred_logits_i.transpose(0, 1), 'pred_boxes': pred_boxes_i.transpose(0, 1)}
            indices = self.matcher(outputs_i, gt_instances_i, g_size=self.g_size)
            indices = [(ind[0], ind[1]) for ind in indices]
            track_instances.matched_gt_idxes[...] = -1
            for i, ind in enumerate(indices):
                track_instances.matched_gt_idxes[ind[0], i] = ind[1]
                track_instances.obj_idxes[ind[0], i] = gt_instances_i[i].obj_ids[ind[1]].long()
                active_idxes = (track_instances.obj_idxes[:, i] >= 0) & (track_instances.matched_gt_idxes[:, i] >= 0)
                active_track_boxes = track_instances.pred_boxes[active_idxes, i]
                if len(active_track_boxes) > 0:
                    gt_boxes = gt_instances_i[i].boxes[track_instances.matched_gt_idxes[active_idxes, i]]
                    active_track_boxes = box_cxcywh_to_xyxy(active_track_boxes)
                    gt_boxes = box_cxcywh_to_xyxy(gt_boxes)
                    track_instances.iou[active_idxes, i] = matched_pairwise_iou(Boxes(active_track_boxes), Boxes(gt_boxes))
            self.num_samples += sum(len(t.boxes) for t in gt_instances_i) * self.g_size
            self.sample_device = pred_logits_i.device
            for loss in self.losses:
                new_track_loss = self.get_loss(loss, outputs=outputs_i, gt_instances=gt_instances_i, indices=indices, num_boxes=1)
                self.losses_dict.update({'frame_{}_{}'.format(self._current_frame_idx, key): value for key, value in new_track_loss.items()})
            if 'aux_outputs' in outputs:
                for i, aux_outputs in enumerate(outputs['aux_outputs']):
                    unmatched_outputs_layer = {'pred_logits': aux_outputs['pred_logits'], 'pred_boxes': aux_outputs['pred_boxes']}
                    matched_indices_layer = self.matcher(unmatched_outputs_layer, gt_instances_i, g_size=self.g_size)
                    matched_indices_layer = [(ind[0], ind[1]) for ind in matched_indices_layer]
                    for loss in self.losses:
                        if loss == 'masks':
                            continue
                        l_dict = self.get_loss(loss, aux_outputs, gt_instances=gt_instances_i, indices=matched_indices_layer, num_boxes=1)
                        self.losses_dict.update({'frame_{}_aux{}_{}'.format(self._current_frame_idx, i, key): value for key, value in l_dict.items()})
        else:
            track_instances.matched_gt_idxes[...] = -1

            def match_for_single_decoder_layer(unmatched_outputs, matcher, untracked_gt_instances, unmatched_track_idxes, untracked_tgt_indexes):
                new_track_indices = matcher(unmatched_outputs, [untracked_gt_instances], g_size=self.g_size)
                src_idx = new_track_indices[0][0]
                tgt_idx = new_track_indices[0][1]
                new_matched_indices = torch.stack([unmatched_track_idxes[src_idx], untracked_tgt_indexes[tgt_idx]], dim=1)
                return new_matched_indices
            for ibn, gt_ins in enumerate(gt_instances_i):
                obj_idxes = gt_ins.obj_ids
                i, j = torch.where(track_instances.obj_idxes[:, ibn:ibn + 1] == obj_idxes)
                track_instances.matched_gt_idxes[i, ibn] = j
                full_track_idxes = torch.arange(len(track_instances), dtype=torch.long, device=pred_logits_i.device)
                matched_track_idxes = track_instances.obj_idxes[:, ibn] >= 0
                prev_matched_indices = torch.stack([full_track_idxes[matched_track_idxes], track_instances.matched_gt_idxes[matched_track_idxes, ibn]], dim=1)
                unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes[:, ibn] == -1]
                tgt_indexes = track_instances.matched_gt_idxes[:, ibn]
                tgt_indexes = tgt_indexes[tgt_indexes != -1]
                tgt_state = torch.zeros(len(gt_ins), device=pred_logits_i.device)
                tgt_state[tgt_indexes] = 1
                full_tgt_idxes = torch.arange(len(gt_ins), device=pred_logits_i.device)
                untracked_tgt_indexes = full_tgt_idxes[tgt_state == 0]
                untracked_gt_instances = gt_ins[untracked_tgt_indexes]
                unmatched_outputs = {'pred_logits': track_instances.pred_logits[unmatched_track_idxes, ibn].unsqueeze(0), 'pred_boxes': track_instances.pred_boxes[unmatched_track_idxes, ibn].unsqueeze(0)}
                new_matched_indices = match_for_single_decoder_layer(unmatched_outputs, self.matcher, untracked_gt_instances, unmatched_track_idxes, untracked_tgt_indexes)
                track_instances.obj_idxes[new_matched_indices[:, 0], ibn] = gt_ins.obj_ids[new_matched_indices[:, 1]].long()
                track_instances.matched_gt_idxes[new_matched_indices[:, 0], ibn] = new_matched_indices[:, 1]
                active_idxes = (track_instances.obj_idxes[:, ibn] >= 0) & (track_instances.matched_gt_idxes[:, ibn] >= 0)
                active_track_boxes = track_instances.pred_boxes[active_idxes, ibn]
                if len(active_track_boxes) > 0:
                    gt_boxes = gt_ins.boxes[track_instances.matched_gt_idxes[active_idxes, ibn]]
                    active_track_boxes = box_cxcywh_to_xyxy(active_track_boxes)
                    gt_boxes = box_cxcywh_to_xyxy(gt_boxes)
                    track_instances.iou[active_idxes, ibn] = matched_pairwise_iou(Boxes(active_track_boxes), Boxes(gt_boxes))
                matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0)
                self.num_samples += len(gt_ins) * self.g_size
                self.sample_device = pred_logits_i.device
                outputs_i = {'pred_logits': pred_logits_i[:, ibn].unsqueeze(0), 'pred_boxes': pred_boxes_i[:, ibn].unsqueeze(0)}
                for loss in self.losses:
                    new_track_loss = self.get_loss(loss, outputs=outputs_i, gt_instances=[gt_ins], indices=[(matched_indices[:, 0], matched_indices[:, 1])], num_boxes=1)
                    for key, value in new_track_loss.items():
                        self.losses_dict['frame_{}_{}'.format(self._current_frame_idx, key)] += value
                if 'aux_outputs' in outputs:
                    for i, aux_outputs in enumerate(outputs['aux_outputs']):
                        unmatched_outputs_layer = {'pred_logits': aux_outputs['pred_logits'][ibn, unmatched_track_idxes].unsqueeze(0), 'pred_boxes': aux_outputs['pred_boxes'][ibn, unmatched_track_idxes].unsqueeze(0)}
                        new_matched_indices_layer = match_for_single_decoder_layer(unmatched_outputs_layer, self.matcher, gt_ins[full_tgt_idxes], unmatched_track_idxes, full_tgt_idxes)
                        matched_indices_layer = torch.cat([new_matched_indices_layer, prev_matched_indices], dim=0)
                        outputs_layer = {'pred_logits': aux_outputs['pred_logits'][ibn].unsqueeze(0), 'pred_boxes': aux_outputs['pred_boxes'][ibn].unsqueeze(0)}
                        for loss in self.losses:
                            if loss == 'masks':
                                continue
                            l_dict = self.get_loss(loss, outputs_layer, gt_instances=[gt_ins], indices=[(matched_indices_layer[:, 0], matched_indices_layer[:, 1])], num_boxes=1)
                            for key, value in l_dict.items():
                                self.losses_dict['frame_{}_aux{}_{}'.format(self._current_frame_idx, i, key)] += value
        self._step()
        return track_instances

    def forward(self, outputs, input_data: 'dict'):
        losses = outputs.pop('losses_dict')
        num_samples = self.get_num_boxes(self.num_samples)
        for loss_name, loss in losses.items():
            losses[loss_name] /= num_samples
        return losses


def pos2posemb(pos, num_pos_feats=64, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
    posemb = pos[..., None] / dim_t
    posemb = torch.stack((posemb[..., 0::2].sin(), posemb[..., 1::2].cos()), dim=-1).flatten(-3)
    return posemb


class DeformableTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.bbox_embed = None
        self.class_embed = None
        self.obj_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios, src_padding_mask=None, mem_bank=None, mem_bank_pad_mask=None, attn_mask=None):
        output = tgt
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            query_pos = pos2posemb(reference_points)
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask, mem_bank, mem_bank_pad_mask, attn_mask)
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        return output, reference_points


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    if activation == 'prelu':
        return nn.PReLU()
    if activation == 'selu':
        return F.selu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')


class DeformableTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation='relu', n_levels=4, n_heads=8, n_points=4, use_deformable_box_attn=False, key_aware_type=None):
        super().__init__()
        if use_deformable_box_attn:
            raise NotImplementedError
        else:
            self.cross_attn = MultiScaleDeformableAttention(embed_dim=d_model, num_levels=n_levels, num_heads=n_heads, num_points=n_points, batch_first=True, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.key_aware_type = key_aware_type
        self.key_aware_proj = None

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    @autocast(enabled=False)
    def forward(self, tgt: 'Optional[Tensor]', tgt_query_pos: 'Optional[Tensor]'=None, tgt_query_sine_embed: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, tgt_reference_points: 'Optional[Tensor]'=None, memory: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, memory_level_start_index: 'Optional[Tensor]'=None, memory_spatial_shapes: 'Optional[Tensor]'=None, memory_pos: 'Optional[Tensor]'=None, self_attn_mask: 'Optional[Tensor]'=None, cross_attn_mask: 'Optional[Tensor]'=None):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
            -
        """
        if self.self_attn is not None:
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
        if self.key_aware_type is not None:
            if self.key_aware_type == 'mean':
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == 'proj_mean':
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError('Unknown key_aware_type: {}'.format(self.key_aware_type))
        tgt2 = self.cross_attn(query=tgt.transpose(0, 1), query_pos=tgt_query_pos.transpose(0, 1), reference_points=tgt_reference_points.transpose(0, 1).contiguous(), value=memory.transpose(0, 1), spatial_shapes=memory_spatial_shapes, level_start_index=memory_level_start_index, key_padding_mask=memory_key_padding_mask).transpose(0, 1)
        tgt = tgt2
        tgt = self.norm1(tgt)
        tgt = self.forward_ffn(tgt)
        return tgt


class DeformableTransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device), torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


def relu_dropout(x, p=0, inplace=False, training=False):
    if not training or p == 0:
        return x.clamp_(min=0) if inplace else x.clamp(min=0)
    mask = (x < 0) | (torch.rand_like(x) > 1 - p)
    return x.masked_fill_(mask, 0).div_(1 - p) if inplace else x.masked_fill(mask, 0).div(1 - p)


class ReLUDropout(torch.nn.Dropout):

    def forward(self, input):
        return relu_dropout(input, p=self.p, training=self.training, inplace=self.inplace)


class DeformableTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation='relu', n_levels=4, n_heads=8, n_points=4, sigmoid_attn=False):
        super().__init__()
        self.self_attn = MultiScaleDeformableAttention(embed_dim=d_model, num_heads=n_heads, num_levels=n_levels, dropout=False, batch_first=True, num_points=n_points, img2col_step=64)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout_relu = ReLUDropout(dropout, True)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout_relu(self.linear1(src)))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        identity = torch.zeros_like(src)
        src2 = self.self_attn(src, query_pos=pos, identity=identity, reference_points=reference_points, value=src, spatial_shapes=spatial_shapes, level_start_index=level_start_index, key_padding_mask=padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.forward_ffn(src)
        return src


class DeformableTransformer(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1, activation='relu', return_intermediate_dec=False, num_feature_levels=4, dec_n_points=4, enc_n_points=4, two_stage=False, two_stage_num_proposals=300, decoder_self_cross=True, sigmoid_attn=False, extra_track_attn=False, memory_bank=False, im2col_step=64):
        super().__init__()
        self.new_frame_adaptor = None
        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points, sigmoid_attn=sigmoid_attn)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, dec_n_points, decoder_self_cross, sigmoid_attn=sigmoid_attn, extra_track_attn=extra_track_attn, memory_bank=memory_bank, im2col_step=im2col_step)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
        proposals = proposals.sigmoid() * scale
        pos = proposals[:, :, :, None] / dim_t
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:_cur + H_ * W_].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device), torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * 2.0 ** lvl
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += H_ * W_
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None, ref_pts=None, mem_bank=None, mem_bank_pad_mask=None, attn_mask=None):
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = h, w
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points_two = topk_coords_unact.sigmoid()
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            _, tgt_two = torch.split(pos_trans_out, c, dim=2)
            tgt = torch.cat([tgt_two, query_embed.unsqueeze(0).expand(bs, -1, -1)[:, topk:]], axis=1)
            reference_points = torch.cat([reference_points_two, ref_pts.unsqueeze(0).expand(bs, -1, -1)[:, topk:]], axis=1)
            init_reference_out = reference_points
        else:
            tgt = query_embed.transpose(0, 1)
            reference_points = ref_pts.transpose(0, 1)
            init_reference_out = reference_points
        hs, inter_references = self.decoder(tgt, reference_points, memory, spatial_shapes, level_start_index, valid_ratios, mask_flatten, mem_bank, mem_bank_pad_mask, attn_mask)
        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, None, None


class ConditionalDETR(nn.Module):
    """Implement Conditional-DETR in `Conditional DETR for Fast Training Convergence
    <https://arxiv.org/abs/2108.06152>`_

    Args:
        backbone (nn.Module): Backbone module for feature extraction.
        in_features (List[str]): Selected backbone output features for transformer module.
        in_channels (int): Dimension of the last feature in `in_features`.
        position_embedding (nn.Module): Position encoding layer for generating position embeddings.
        transformer (nn.Module): Transformer module used for further processing features and input queries.
        embed_dim (int): Hidden dimension for transformer module.
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        select_box_nums_for_evaluation (int): Select the top-k confidence predicted boxes for inference.
            Default: 300.
        device (str): Training device. Default: "cuda".
    """

    def __init__(self, backbone: 'nn.Module', in_features: 'List[str]', in_channels: 'int', position_embedding: 'nn.Module', transformer: 'nn.Module', embed_dim: 'int', num_classes: 'int', num_queries: 'int', criterion: 'nn.Module', aux_loss: 'bool'=True, pixel_mean: 'List[float]'=[123.675, 116.28, 103.53], pixel_std: 'List[float]'=[58.395, 57.12, 57.375], select_box_nums_for_evaluation: 'int'=300, device: 'str'='cuda'):
        super(ConditionalDETR, self).__init__()
        self.backbone = backbone
        self.in_features = in_features
        self.position_embedding = position_embedding
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.transformer = transformer
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(input_dim=embed_dim, hidden_dim=embed_dim, output_dim=4, num_layers=3)
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        self.criterion = criterion
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation
        self.init_weights()

    def init_weights(self):
        """Initialize weights for Conditioanl-DETR."""
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

    def forward(self, batched_inputs):
        """Forward function of `DAB-DETR` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries.
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        images = self.preprocess_image(batched_inputs)
        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]['instances'].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)
        features = self.backbone(images.tensor)[self.in_features[-1]]
        features = self.input_proj(features)
        img_masks = F.interpolate(img_masks[None], size=features.shape[-2:])[0]
        pos_embed = self.position_embedding(img_masks)
        hidden_states, reference = self.transformer(features, img_masks, self.query_embed.weight, pos_embed)
        reference_before_sigmoid = inverse_sigmoid(reference)
        outputs_coords = []
        for lvl in range(hidden_states.shape[0]):
            tmp = self.bbox_embed(hidden_states[lvl])
            tmp[..., :2] += reference_before_sigmoid
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
        outputs_coord = torch.stack(outputs_coords)
        outputs_class = self.class_embed(hidden_states)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            output['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        if self.training:
            gt_instances = [x['instances'] for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output['pred_logits']
            box_pred = output['pred_boxes']
            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get('height', image_size[0])
                width = input_per_image.get('width', image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({'instances': r})
            return processed_results

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def inference(self, box_cls, box_pred, image_sizes):
        """Inference function for DAB-DETR

        Args:
            box_cls (torch.Tensor): tensor of shape ``(batch_size, num_queries, K)``.
                The tensor predicts the classification probability for each query.
            box_pred (torch.Tensor): tensors of shape ``(batch_size, num_queries, 4)``.
                The tensor predicts 4-vector ``(x, y, w, h)`` box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode='floor')
        labels = topk_indexes % box_cls.shape[2]
        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(scores, labels, boxes, image_sizes)):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({'labels': gt_classes, 'boxes': gt_boxes})
        return new_targets

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x['image']) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images


class ConditionalDetrTransformerEncoder(TransformerLayerSequence):

    def __init__(self, embed_dim: 'int'=256, num_heads: 'int'=8, attn_dropout: 'float'=0.1, feedforward_dim: 'int'=2048, ffn_dropout: 'float'=0.1, activation: 'nn.Module'=nn.PReLU(), post_norm: 'bool'=False, num_layers: 'int'=6, batch_first: 'bool'=False):
        super(ConditionalDetrTransformerEncoder, self).__init__(transformer_layers=BaseTransformerLayer(attn=MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_dropout, batch_first=batch_first), ffn=FFN(embed_dim=embed_dim, feedforward_dim=feedforward_dim, ffn_drop=ffn_dropout, activation=activation), norm=nn.LayerNorm(normalized_shape=embed_dim), operation_order=('self_attn', 'norm', 'ffn', 'norm')), num_layers=num_layers)
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm
        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(self, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, **kwargs):
        for layer in self.layers:
            query = layer(query, key, value, query_pos=query_pos, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, **kwargs)
        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


def get_sine_pos_embed(pos_tensor: 'torch.Tensor', num_pos_feats: 'int'=128, temperature: 'int'=10000, exchange_xy: 'bool'=True) ->torch.Tensor:
    """generate sine position embedding from a position tensor

    Args:
        pos_tensor (torch.Tensor): Shape as `(None, n)`.
        num_pos_feats (int): projected shape for each float in the tensor. Default: 128
        temperature (int): The temperature used for scaling
            the position embedding. Default: 10000.
        exchange_xy (bool, optional): exchange pos x and pos y.             For example, input tensor is `[x, y]`, the results will  # noqa 
            be `[pos(y), pos(x)]`. Defaults: True.

    Returns:
        torch.Tensor: Returned position embedding  # noqa 
        with shape `(None, n * num_pos_feats)`.
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)

    def sine_func(x: 'torch.Tensor'):
        sin_x = x * scale / dim_t
        sin_x = torch.stack((sin_x[:, :, 0::2].sin(), sin_x[:, :, 1::2].cos()), dim=3).flatten(2)
        return sin_x
    pos_res = [sine_func(x) for x in pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)]
    if exchange_xy:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
    pos_res = torch.cat(pos_res, dim=2)
    return pos_res


class ConditionalDetrTransformerDecoder(TransformerLayerSequence):

    def __init__(self, embed_dim: 'int'=256, num_heads: 'int'=8, attn_dropout: 'float'=0.0, feedforward_dim: 'int'=2048, ffn_dropout: 'float'=0.0, activation: 'nn.Module'=nn.PReLU(), num_layers: 'int'=None, batch_first: 'bool'=False, post_norm: 'bool'=True, return_intermediate: 'bool'=True):
        super(ConditionalDetrTransformerDecoder, self).__init__(transformer_layers=BaseTransformerLayer(attn=[ConditionalSelfAttention(embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_dropout, batch_first=batch_first), ConditionalCrossAttention(embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_dropout, batch_first=batch_first)], ffn=FFN(embed_dim=embed_dim, feedforward_dim=feedforward_dim, ffn_drop=ffn_dropout, activation=activation), norm=nn.LayerNorm(normalized_shape=embed_dim), operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')), num_layers=num_layers)
        self.return_intermediate = return_intermediate
        self.embed_dim = self.layers[0].embed_dim
        self.query_scale = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)
        self.ref_point_head = MLP(self.embed_dim, self.embed_dim, 2, 2)
        self.bbox_embed = None
        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None
        for idx in range(num_layers - 1):
            self.layers[idx + 1].attentions[1].query_pos_proj = None

    def forward(self, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, **kwargs):
        intermediate = []
        reference_points_before_sigmoid = self.ref_point_head(query_pos)
        reference_points: 'torch.Tensor' = reference_points_before_sigmoid.sigmoid().transpose(0, 1)
        for idx, layer in enumerate(self.layers):
            obj_center = reference_points[..., :2].transpose(0, 1)
            if idx == 0:
                position_transform = 1
            else:
                position_transform = self.query_scale(query)
            query_sine_embed = get_sine_pos_embed(obj_center)
            query_sine_embed = query_sine_embed[..., :self.embed_dim] * position_transform
            query: 'torch.Tensor' = layer(query, key, value, query_pos=query_pos, key_pos=key_pos, query_sine_embed=query_sine_embed, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, is_first_layer=idx == 0, **kwargs)
            if self.return_intermediate:
                if self.post_norm_layer is not None:
                    intermediate.append(self.post_norm_layer(query))
                else:
                    intermediate.append(query)
        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(query)
        if self.return_intermediate:
            return [torch.stack(intermediate).transpose(1, 2), reference_points]
        return query.unsqueeze(0)


class ConditionalDetrTransformer(nn.Module):

    def __init__(self, encoder=None, decoder=None):
        super(ConditionalDetrTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dim = self.encoder.embed_dim
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask, query_embed, pos_embed):
        bs, c, h, w = x.shape
        x = x.view(bs, c, -1).permute(2, 0, 1)
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.view(bs, -1)
        memory = self.encoder(query=x, key=None, value=None, query_pos=pos_embed, query_key_padding_mask=mask)
        target = torch.zeros_like(query_embed)
        hidden_state, references = self.decoder(query=target, key=memory, value=memory, key_pos=pos_embed, query_pos=query_embed)
        return hidden_state, references


class DabDeformableDETR(nn.Module):
    """Implement DAB-Deformable-DETR in `DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR
    <https://arxiv.org/abs/2201.12329>`_.

    Code is modified from the `official github repo
    <https://github.com/IDEA-opensource/DAB-DETR>`_.

    Args:
        backbone (nn.Module): backbone module
        position_embedding (nn.Module): position embedding module
        neck (nn.Module): neck module
        transformer (nn.Module): transformer module
        embed_dim (int): dimension of embedding
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 100.
        device (str): Training device. Default: "cuda".
    """

    def __init__(self, backbone: 'nn.Module', position_embedding: 'nn.Module', neck: 'nn.Module', transformer: 'nn.Module', embed_dim: 'int', num_classes: 'int', num_queries: 'int', criterion: 'nn.Module', pixel_mean: 'List[float]'=[123.675, 116.28, 103.53], pixel_std: 'List[float]'=[58.395, 57.12, 57.375], aux_loss: 'bool'=True, as_two_stage: 'bool'=False, select_box_nums_for_evaluation: 'int'=300, device='cuda'):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.neck = neck
        self.num_queries = num_queries
        if not as_two_stage:
            self.tgt_embed = nn.Embedding(num_queries, embed_dim)
            self.refpoint_embed = nn.Embedding(num_queries, 4)
            nn.init.zeros_(self.tgt_embed.weight)
            nn.init.uniform_(self.refpoint_embed.weight)
            self.refpoint_embed.weight.data[:] = inverse_sigmoid(self.refpoint_embed.weight.data[:]).clamp(-3, 3)
        self.transformer = transformer
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        self.criterion = criterion
        self.as_two_stage = as_two_stage
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)
        num_pred = transformer.decoder.num_layers + 1 if as_two_stage else transformer.decoder.num_layers
        self.class_embed = nn.ModuleList([copy.deepcopy(self.class_embed) for i in range(num_pred)])
        self.bbox_embed = nn.ModuleList([copy.deepcopy(self.bbox_embed) for i in range(num_pred)])
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.bbox_embed = self.bbox_embed
        if self.as_two_stage:
            for bbox_embed_layer in self.bbox_embed:
                nn.init.constant_(bbox_embed_layer.layers[-1].bias.data[2:], 0.0)
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

    def forward(self, batched_inputs):
        """Forward function of `DAB-Deformable-DETR` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries (anchor boxes in DAB-DETR).
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        images = self.preprocess_image(batched_inputs)
        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]['instances'].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)
        features = self.backbone(images.tensor)
        multi_level_feats = self.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(F.interpolate(img_masks[None], size=feat.shape[-2:]).squeeze(0))
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))
        if self.as_two_stage:
            query_embeds = None
        else:
            tgt_embed = self.tgt_embed.weight
            refanchor = self.refpoint_embed.weight
            query_embeds = torch.cat((tgt_embed, refanchor), dim=1)
        inter_states, init_reference, inter_references, enc_state, enc_reference = self.transformer(multi_level_feats, multi_level_masks, multi_level_position_embeddings, query_embeds)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            output['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        if self.as_two_stage:
            interm_coord = enc_reference
            interm_class = self.class_embed[-1](enc_state)
            output['enc_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
        if self.training:
            gt_instances = [x['instances'] for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output['pred_logits']
            box_pred = output['pred_boxes']
            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get('height', image_size[0])
                width = input_per_image.get('width', image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({'instances': r})
            return processed_results

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x['image']) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode='floor')
        labels = topk_indexes % box_cls.shape[2]
        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(scores, labels, boxes, image_sizes)):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({'labels': gt_classes, 'boxes': gt_boxes})
        return new_targets

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class DabDeformableDetrTransformerEncoder(TransformerLayerSequence):

    def __init__(self, embed_dim: 'int'=256, num_heads: 'int'=8, feedforward_dim: 'int'=1024, attn_dropout: 'float'=0.1, ffn_dropout: 'float'=0.1, operation_order: 'tuple'=('self_attn', 'norm', 'ffn', 'norm'), num_layers: 'int'=6, post_norm: 'bool'=False, num_feature_levels: 'int'=4):
        super(DabDeformableDetrTransformerEncoder, self).__init__(transformer_layers=BaseTransformerLayer(attn=MultiScaleDeformableAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True, num_levels=num_feature_levels), ffn=FFN(embed_dim=embed_dim, feedforward_dim=feedforward_dim, output_dim=embed_dim, num_fcs=2, ffn_drop=ffn_dropout), norm=nn.LayerNorm(embed_dim), operation_order=operation_order), num_layers=num_layers)
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm
        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(self, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, **kwargs):
        for layer in self.layers:
            query = layer(query, key, value, query_pos=query_pos, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, **kwargs)
        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class DabDeformableDetrTransformerDecoder(TransformerLayerSequence):

    def __init__(self, embed_dim: 'int'=256, num_heads: 'int'=8, feedforward_dim: 'int'=1024, attn_dropout: 'float'=0.1, ffn_dropout: 'float'=0.1, num_layers: 'int'=6, return_intermediate: 'bool'=True, num_feature_levels: 'int'=4):
        super(DabDeformableDetrTransformerDecoder, self).__init__(transformer_layers=BaseTransformerLayer(attn=[MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_dropout, batch_first=True), MultiScaleDeformableAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True, num_levels=num_feature_levels)], ffn=FFN(embed_dim=embed_dim, feedforward_dim=feedforward_dim, output_dim=embed_dim, ffn_drop=ffn_dropout), norm=nn.LayerNorm(embed_dim), operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')), num_layers=num_layers)
        self.return_intermediate = return_intermediate
        self.query_scale = MLP(embed_dim, embed_dim, embed_dim, 2)
        self.ref_point_head = MLP(2 * embed_dim, embed_dim, embed_dim, 2)
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, reference_points=None, valid_ratios=None, **kwargs):
        output = query
        bs, num_queries, _ = output.size()
        if reference_points.dim() == 2:
            reference_points = reference_points.unsqueeze(0).repeat(bs, 1, 1)
        intermediate = []
        intermediate_reference_points = []
        for layer_idx, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]
            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])
            raw_query_pos = self.ref_point_head(query_sine_embed)
            pos_scale = self.query_scale(output) if layer_idx != 0 else 1
            query_pos = pos_scale * raw_query_pos
            output = layer(output, key, value, query_pos=query_pos, key_pos=key_pos, query_sine_embed=query_sine_embed, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, reference_points=reference_points_input, **kwargs)
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[layer_idx](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        return output, reference_points


class DabDeformableDetrTransformer(nn.Module):
    """Transformer module for DAB-Deformable-DETR

    Args:
        encoder (nn.Module): encoder module.
        decoder (nn.Module): decoder module.
        as_two_stage (bool): whether to use two-stage transformer. Default False.
        num_feature_levels (int): number of feature levels. Default 4.
        two_stage_num_proposals (int): number of proposals in two-stage transformer. Default 100.
            Only used when as_two_stage is True.
    """

    def __init__(self, encoder=None, decoder=None, as_two_stage=False, num_feature_levels=4, two_stage_num_proposals=300):
        super(DabDeformableDetrTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals
        self.embed_dim = self.encoder.embed_dim
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dim))
        if self.as_two_stage:
            self.enc_output = nn.Linear(self.embed_dim, self.embed_dim)
            self.enc_output_norm = nn.LayerNorm(self.embed_dim)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.normal_(self.level_embeds)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N, S, C = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:_cur + H * W].view(N, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device), torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * 2.0 ** lvl
            proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
            proposals.append(proposal)
            _cur += H * W
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The ratios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has                 shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device), torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid ratios of feature maps of all levels."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, multi_level_feats, multi_level_masks, multi_level_pos_embeds, query_embed, **kwargs):
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = h, w
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in multi_level_masks], 1)
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=feat.device)
        memory = self.encoder(query=feat_flatten, key=None, value=None, query_pos=lvl_pos_embed_flatten, query_key_padding_mask=mask_flatten, spatial_shapes=spatial_shapes, reference_points=reference_points, level_start_index=level_start_index, valid_ratios=valid_ratios, **kwargs)
        bs, _, c = memory.shape
        if self.as_two_stage:
            assert query_embed is None, 'query_embed should be None in two-stage'
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            reference_points = topk_coords_unact.detach().sigmoid()
            init_reference_out = reference_points
            target_unact = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target_unact.detach()
        else:
            reference_points = query_embed[..., self.embed_dim:].sigmoid()
            target = query_embed[..., :self.embed_dim]
            target = target.unsqueeze(0).expand(bs, -1, -1)
            init_reference_out = reference_points
        inter_states, inter_references = self.decoder(query=target, key=memory, value=memory, query_pos=None, key_padding_mask=mask_flatten, reference_points=reference_points, spatial_shapes=spatial_shapes, level_start_index=level_start_index, valid_ratios=valid_ratios, **kwargs)
        inter_references_out = inter_references
        if self.as_two_stage:
            return inter_states, init_reference_out, inter_references_out, target_unact, topk_coords_unact.sigmoid()
        return inter_states, init_reference_out, inter_references_out, None, None


class DABDETR(nn.Module):
    """Implement DAB-DETR in `DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR
    <https://arxiv.org/abs/2201.12329>`_

    Args:
        backbone (nn.Module): Backbone module for feature extraction.
        in_features (List[str]): Selected backbone output features for transformer module.
        in_channels (int): Dimension of the last feature in `in_features`.
        position_embedding (nn.Module): Position encoding layer for generating position embeddings.
        transformer (nn.Module): Transformer module used for further processing features and input queries.
        embed_dim (int): Hidden dimension for transformer module.
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        freeze_anchor_box_centers (bool): If True, freeze the center param ``(x, y)`` for
            the initialized dynamic anchor boxes in format ``(x, y, w, h)``
            and only train ``(w, h)``. Default: True.
        select_box_nums_for_evaluation (int): Select the top-k confidence predicted boxes for inference.
            Default: 300.
        device (str): Training device. Default: "cuda".
    """

    def __init__(self, backbone: 'nn.Module', in_features: 'List[str]', in_channels: 'int', position_embedding: 'nn.Module', transformer: 'nn.Module', embed_dim: 'int', num_classes: 'int', num_queries: 'int', criterion: 'nn.Module', aux_loss: 'bool'=True, pixel_mean: 'List[float]'=[123.675, 116.28, 103.53], pixel_std: 'List[float]'=[58.395, 57.12, 57.375], freeze_anchor_box_centers: 'bool'=True, select_box_nums_for_evaluation: 'int'=300, device: 'str'='cuda'):
        super(DABDETR, self).__init__()
        self.backbone = backbone
        self.in_features = in_features
        self.position_embedding = position_embedding
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.transformer = transformer
        self.anchor_box_embed = nn.Embedding(num_queries, 4)
        self.freeze_anchor_box_centers = freeze_anchor_box_centers
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(input_dim=embed_dim, hidden_dim=embed_dim, output_dim=4, num_layers=3)
        self.num_classes = num_classes
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.aux_loss = aux_loss
        self.criterion = criterion
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation
        self.init_weights()

    def init_weights(self):
        """Initialize weights for DAB-DETR."""
        if self.freeze_anchor_box_centers:
            self.anchor_box_embed.weight.data[:, :2].uniform_(0, 1)
            self.anchor_box_embed.weight.data[:, :2] = inverse_sigmoid(self.anchor_box_embed.weight.data[:, :2])
            self.anchor_box_embed.weight.data[:, :2].requires_grad = False
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

    def forward(self, batched_inputs):
        """Forward function of `DAB-DETR` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries (anchor boxes in DAB-DETR).
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        images = self.preprocess_image(batched_inputs)
        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]['instances'].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)
        features = self.backbone(images.tensor)[self.in_features[-1]]
        features = self.input_proj(features)
        img_masks = F.interpolate(img_masks[None], size=features.shape[-2:])[0]
        pos_embed = self.position_embedding(img_masks)
        dynamic_anchor_boxes = self.anchor_box_embed.weight
        hidden_states, reference_boxes = self.transformer(features, img_masks, dynamic_anchor_boxes, pos_embed)
        reference_boxes = inverse_sigmoid(reference_boxes)
        anchor_box_offsets = self.bbox_embed(hidden_states)
        outputs_coord = (reference_boxes + anchor_box_offsets).sigmoid()
        outputs_class = self.class_embed(hidden_states)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            output['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        if self.training:
            gt_instances = [x['instances'] for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output['pred_logits']
            box_pred = output['pred_boxes']
            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get('height', image_size[0])
                width = input_per_image.get('width', image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({'instances': r})
            return processed_results

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def inference(self, box_cls, box_pred, image_sizes):
        """Inference function for DAB-DETR

        Args:
            box_cls (torch.Tensor): tensor of shape ``(batch_size, num_queries, K)``.
                The tensor predicts the classification probability for each query.
            box_pred (torch.Tensor): tensors of shape ``(batch_size, num_queries, 4)``.
                The tensor predicts 4-vector ``(x, y, w, h)`` box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode='floor')
        labels = topk_indexes % box_cls.shape[2]
        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(scores, labels, boxes, image_sizes)):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({'labels': gt_classes, 'boxes': gt_boxes})
        return new_targets

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x['image']) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images


class DabDetrTransformerEncoder(TransformerLayerSequence):

    def __init__(self, embed_dim: 'int'=256, num_heads: 'int'=8, attn_dropout: 'float'=0.1, feedforward_dim: 'int'=2048, ffn_dropout: 'float'=0.1, activation: 'nn.Module'=nn.PReLU(), post_norm: 'bool'=False, num_layers: 'int'=6, batch_first: 'bool'=False):
        super(DabDetrTransformerEncoder, self).__init__(transformer_layers=BaseTransformerLayer(attn=MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_dropout, batch_first=batch_first), ffn=FFN(embed_dim=embed_dim, feedforward_dim=feedforward_dim, ffn_drop=ffn_dropout, activation=activation), norm=nn.LayerNorm(normalized_shape=embed_dim), operation_order=('self_attn', 'norm', 'ffn', 'norm')), num_layers=num_layers)
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm
        self.query_scale = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)
        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(self, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, **kwargs):
        for layer in self.layers:
            position_scales = self.query_scale(query)
            query = layer(query, key, value, query_pos=query_pos * position_scales, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, **kwargs)
        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class DabDetrTransformerDecoder(TransformerLayerSequence):

    def __init__(self, embed_dim: 'int'=256, num_heads: 'int'=8, attn_dropout: 'float'=0.0, feedforward_dim: 'int'=2048, ffn_dropout: 'float'=0.0, activation: 'nn.Module'=nn.PReLU(), num_layers: 'int'=None, modulate_hw_attn: 'bool'=True, batch_first: 'bool'=False, post_norm: 'bool'=True, return_intermediate: 'bool'=True):
        super(DabDetrTransformerDecoder, self).__init__(transformer_layers=BaseTransformerLayer(attn=[ConditionalSelfAttention(embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_dropout, batch_first=batch_first), ConditionalCrossAttention(embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_dropout, batch_first=batch_first)], ffn=FFN(embed_dim=embed_dim, feedforward_dim=feedforward_dim, ffn_drop=ffn_dropout, activation=activation), norm=nn.LayerNorm(normalized_shape=embed_dim), operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')), num_layers=num_layers)
        self.return_intermediate = return_intermediate
        self.embed_dim = self.layers[0].embed_dim
        self.query_scale = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)
        self.ref_point_head = MLP(2 * self.embed_dim, self.embed_dim, self.embed_dim, 2)
        self.bbox_embed = None
        if modulate_hw_attn:
            self.ref_anchor_head = MLP(self.embed_dim, self.embed_dim, 2, 2)
        self.modulate_hw_attn = modulate_hw_attn
        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None
        for idx in range(num_layers - 1):
            self.layers[idx + 1].attentions[1].query_pos_proj = None

    def forward(self, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, anchor_box_embed=None, **kwargs):
        intermediate = []
        reference_boxes = anchor_box_embed.sigmoid()
        intermediate_ref_boxes = [reference_boxes]
        for idx, layer in enumerate(self.layers):
            obj_center = reference_boxes[..., :self.embed_dim]
            query_sine_embed = get_sine_pos_embed(obj_center)
            query_pos = self.ref_point_head(query_sine_embed)
            if idx == 0:
                position_transform = 1
            else:
                position_transform = self.query_scale(query)
            query_sine_embed = query_sine_embed[..., :self.embed_dim] * position_transform
            if self.modulate_hw_attn:
                ref_hw_cond = self.ref_anchor_head(query).sigmoid()
                query_sine_embed[..., self.embed_dim // 2:] *= (ref_hw_cond[..., 0] / obj_center[..., 2]).unsqueeze(-1)
                query_sine_embed[..., :self.embed_dim // 2] *= (ref_hw_cond[..., 1] / obj_center[..., 3]).unsqueeze(-1)
            query = layer(query, key, value, query_pos=query_pos, key_pos=key_pos, query_sine_embed=query_sine_embed, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, is_first_layer=idx == 0, **kwargs)
            if self.bbox_embed is not None:
                offsets = self.bbox_embed(query)
                offsets[..., :self.embed_dim] += inverse_sigmoid(reference_boxes)
                new_reference_boxes = offsets[..., :self.embed_dim].sigmoid()
                if idx != self.num_layers - 1:
                    intermediate_ref_boxes.append(new_reference_boxes)
                reference_boxes = new_reference_boxes.detach()
            if self.return_intermediate:
                if self.post_norm_layer is not None:
                    intermediate.append(self.post_norm_layer(query))
                else:
                    intermediate.append(query)
        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(query)
        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [torch.stack(intermediate).transpose(1, 2), torch.stack(intermediate_ref_boxes).transpose(1, 2)]
            else:
                return [torch.stack(intermediate).transpose(1, 2), reference_boxes.unsqueeze(0).transpose(1, 2)]
        return query.unsqueeze(0)


class DabDetrTransformer(nn.Module):

    def __init__(self, encoder=None, decoder=None):
        super(DabDetrTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dim = self.encoder.embed_dim
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask, anchor_box_embed, pos_embed):
        bs, c, h, w = x.shape
        x = x.view(bs, c, -1).permute(2, 0, 1)
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        anchor_box_embed = anchor_box_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.view(bs, -1)
        memory = self.encoder(query=x, key=None, value=None, query_pos=pos_embed, query_key_padding_mask=mask)
        num_queries = anchor_box_embed.shape[0]
        target = torch.zeros(num_queries, bs, self.embed_dim, device=anchor_box_embed.device)
        hidden_state, reference_boxes = self.decoder(query=target, key=memory, value=memory, key_pos=pos_embed, anchor_box_embed=anchor_box_embed)
        return hidden_state, reference_boxes


class DeformableCriterion(SetCriterion):
    """This class computes the loss for Deformable-DETR
    and two-stage Deformable-DETR
    """

    def __init__(self, num_classes, matcher, weight_dict, losses: 'List[str]'=['class', 'boxes'], eos_coef: 'float'=0.1, loss_class_type: 'str'='focal_loss', alpha: 'float'=0.25, gamma: 'float'=2.0):
        super(DeformableCriterion, self).__init__(num_classes=num_classes, matcher=matcher, weight_dict=weight_dict, losses=losses, eos_coef=eos_coef, loss_class_type=loss_class_type, alpha=alpha, gamma=gamma)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {(k + f'_{i}'): v for k, v in l_dict.items()}
                    losses.update(l_dict)
        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {(k + '_enc'): v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses


class DeformableDETR(nn.Module):
    """Implements the Deformable DETR model.

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2010.04159>`_ .

    Args:
        backbone (nn.Module): the backbone module.
        position_embedding (nn.Module): the position embedding module.
        neck (nn.Module): the neck module.
        transformer (nn.Module): the transformer module.
        embed_dim (int): the dimension of the embedding.
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): whether to use auxiliary loss. Default: True.
        with_box_refine (bool): whether to use box refinement. Default: False.
        as_two_stage (bool): whether to use two-stage. Default: False.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 100.

    """

    def __init__(self, backbone, position_embedding, neck, transformer, embed_dim, num_classes, num_queries, criterion, pixel_mean, pixel_std, aux_loss=True, with_box_refine=False, as_two_stage=False, select_box_nums_for_evaluation=100, device='cuda'):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.neck = neck
        self.num_queries = num_queries
        if not as_two_stage:
            self.query_embedding = nn.Embedding(num_queries, embed_dim * 2)
        self.transformer = transformer
        self.num_classes = num_classes
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self.aux_loss = aux_loss
        self.criterion = criterion
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)
        num_pred = transformer.decoder.num_layers + 1 if as_two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = nn.ModuleList([copy.deepcopy(self.class_embed) for i in range(num_pred)])
            self.bbox_embed = nn.ModuleList([copy.deepcopy(self.bbox_embed) for i in range(num_pred)])
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if as_two_stage:
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]['instances'].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)
        features = self.backbone(images.tensor)
        multi_level_feats = self.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(F.interpolate(img_masks[None], size=feat.shape[-2:]).squeeze(0))
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))
        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight
        inter_states, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, anchors = self.transformer(multi_level_feats, multi_level_masks, multi_level_position_embeddings, query_embeds)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'init_reference': init_reference}
        if self.aux_loss:
            output['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        if self.as_two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            output['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord, 'anchors': anchors}
        if self.training:
            gt_instances = [x['instances'] for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output['pred_logits']
            box_pred = output['pred_boxes']
            if self.criterion.assign_second_stage:
                results = self.nms_inference(box_cls, box_pred, images.image_sizes)
            else:
                results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get('height', image_size[0])
                width = input_per_image.get('width', image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({'instances': r})
            return processed_results

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode='floor')
        labels = topk_indexes % box_cls.shape[2]
        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(scores, labels, boxes, image_sizes)):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def nms_inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []
        bs, n_queries, n_cls = box_cls.shape
        prob = box_cls.sigmoid()
        all_scores = prob.view(bs, n_queries * n_cls)
        all_indexes = torch.arange(n_queries * n_cls)[None].repeat(bs, 1)
        all_boxes = torch.div(all_indexes, box_cls.shape[2], rounding_mode='floor')
        all_labels = all_indexes % box_cls.shape[2]
        boxes = box_cxcywh_to_xyxy(box_pred)
        boxes = torch.gather(boxes, 1, all_boxes.unsqueeze(-1).repeat(1, 1, 4))
        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(all_scores, all_labels, boxes, image_sizes)):
            pre_topk = scores_per_image.topk(10000).indices
            box = box_pred_per_image[pre_topk]
            score = scores_per_image[pre_topk]
            label = labels_per_image[pre_topk]
            keep_index = batched_nms(box, score, label, 0.7)[:100]
            result = Instances(image_size)
            result.pred_boxes = Boxes(box[keep_index])
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = score[keep_index]
            result.pred_classes = label[keep_index]
            results.append(result)
        return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({'labels': gt_classes, 'boxes': gt_boxes})
        return new_targets

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x['image']) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images


class DeformableDetrTransformerEncoder(TransformerLayerSequence):

    def __init__(self, embed_dim: 'int'=256, num_heads: 'int'=8, feedforward_dim: 'int'=1024, attn_dropout: 'float'=0.1, ffn_dropout: 'float'=0.1, num_layers: 'int'=6, post_norm: 'bool'=False, num_feature_levels: 'int'=4):
        super(DeformableDetrTransformerEncoder, self).__init__(transformer_layers=BaseTransformerLayer(attn=MultiScaleDeformableAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True, num_levels=num_feature_levels), ffn=FFN(embed_dim=embed_dim, feedforward_dim=feedforward_dim, output_dim=embed_dim, num_fcs=2, ffn_drop=ffn_dropout), norm=nn.LayerNorm(embed_dim), operation_order=('self_attn', 'norm', 'ffn', 'norm')), num_layers=num_layers)
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm
        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(self, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, **kwargs):
        for layer in self.layers:
            query = layer(query, key, value, query_pos=query_pos, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, **kwargs)
        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class DeformableDetrTransformerDecoder(TransformerLayerSequence):

    def __init__(self, embed_dim: 'int'=256, num_heads: 'int'=8, feedforward_dim: 'int'=1024, attn_dropout: 'float'=0.1, ffn_dropout: 'float'=0.1, num_layers: 'int'=6, return_intermediate: 'bool'=True, num_feature_levels: 'int'=4):
        super(DeformableDetrTransformerDecoder, self).__init__(transformer_layers=BaseTransformerLayer(attn=[MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_dropout, batch_first=True), MultiScaleDeformableAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True, num_levels=num_feature_levels)], ffn=FFN(embed_dim=embed_dim, feedforward_dim=feedforward_dim, output_dim=embed_dim, ffn_drop=ffn_dropout), norm=nn.LayerNorm(embed_dim), operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')), num_layers=num_layers)
        self.return_intermediate = return_intermediate
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, reference_points=None, valid_ratios=None, **kwargs):
        output = query
        intermediate = []
        intermediate_reference_points = []
        for layer_idx, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]
            output = layer(output, key, value, query_pos=query_pos, key_pos=key_pos, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, reference_points=reference_points_input, **kwargs)
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[layer_idx](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        return output, reference_points


class DeformableDetrTransformer(nn.Module):
    """Transformer module for Deformable DETR

    Args:
        encoder (nn.Module): encoder module.
        decoder (nn.Module): decoder module.
        as_two_stage (bool): whether to use two-stage transformer. Default False.
        num_feature_levels (int): number of feature levels. Default 4.
        two_stage_num_proposals (int): number of proposals in two-stage transformer. Default 300.
            Only used when as_two_stage is True.
    """

    def __init__(self, encoder=None, decoder=None, num_feature_levels=4, as_two_stage=False, two_stage_num_proposals=300, assign_first_stage=True):
        super(DeformableDetrTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_feature_levels = num_feature_levels
        self.as_two_stage = as_two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.assign_first_stage = assign_first_stage
        self.embed_dim = self.encoder.embed_dim
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dim))
        if self.as_two_stage:
            self.enc_output = nn.Linear(self.embed_dim, self.embed_dim)
            self.enc_output_norm = nn.LayerNorm(self.embed_dim)
            self.pos_trans = nn.Linear(self.embed_dim * 2, self.embed_dim * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dim * 2)
            self.pix_trans = nn.Linear(self.embed_dim, self.embed_dim)
            self.pix_trans_norm = nn.LayerNorm(self.embed_dim)
        else:
            self.reference_points = nn.Linear(self.embed_dim, 2)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if not self.as_two_stage:
            nn.init.xavier_normal_(self.reference_points.weight.data, gain=1.0)
            nn.init.constant_(self.reference_points.bias.data, 0.0)
        nn.init.normal_(self.level_embeds)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N, S, C = memory.shape
        proposals = []
        _cur = 0
        level_ids = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:_cur + H * W].view(N, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device), torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * 2.0 ** lvl
            proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
            proposals.append(proposal)
            _cur += H * W
            level_ids.append(grid.new_ones(H * W, dtype=torch.long) * lvl)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        level_ids = torch.cat(level_ids)
        return output_memory, output_proposals, level_ids

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The ratios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has                 shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device), torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid ratios of feature maps of all levels."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_proposal_pos_embed(self, proposals, num_pos_feats=128, temperature=10000):
        """Get the position embedding of proposal."""
        scale = 2 * math.pi
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
        proposals = proposals.sigmoid() * scale
        pos = proposals[:, :, :, None] / dim_t
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def forward(self, multi_level_feats, multi_level_masks, multi_level_pos_embeds, query_embed, **kwargs):
        assert self.as_two_stage or query_embed is not None
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = h, w
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in multi_level_masks], 1)
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=feat.device)
        memory = self.encoder(query=feat_flatten, key=None, value=None, query_pos=lvl_pos_embed_flatten, query_key_padding_mask=mask_flatten, spatial_shapes=spatial_shapes, reference_points=reference_points, level_start_index=level_start_index, valid_ratios=valid_ratios, **kwargs)
        bs, _, c = memory.shape
        if self.as_two_stage:
            output_memory, output_proposals, level_ids = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
            topk = self.two_stage_num_proposals
            proposal_logit = enc_outputs_class[..., 0]
            if self.assign_first_stage:
                proposal_boxes = box_cxcywh_to_xyxy(enc_outputs_coord_unact.sigmoid().float()).clamp(0, 1)
                topk_proposals = []
                for b in range(bs):
                    prop_boxes_b = proposal_boxes[b]
                    prop_logits_b = proposal_logit[b]
                    pre_nms_topk = 1000
                    pre_nms_inds = []
                    for lvl in range(len(spatial_shapes)):
                        lvl_mask = level_ids == lvl
                        pre_nms_inds.append(torch.topk(prop_logits_b.sigmoid() * lvl_mask, pre_nms_topk)[1])
                    pre_nms_inds = torch.cat(pre_nms_inds)
                    post_nms_inds = batched_nms(prop_boxes_b[pre_nms_inds], prop_logits_b[pre_nms_inds], level_ids[pre_nms_inds], 0.9)
                    keep_inds = pre_nms_inds[post_nms_inds]
                    if len(keep_inds) < self.two_stage_num_proposals:
                        None
                        keep_inds = torch.topk(proposal_logit[b], topk)[1]
                    q_per_l = topk // len(spatial_shapes)
                    is_level_ordered = level_ids[keep_inds][None] == torch.arange(len(spatial_shapes), device=level_ids.device)[:, None]
                    keep_inds_mask = is_level_ordered & (is_level_ordered.cumsum(1) <= q_per_l)
                    keep_inds_mask = keep_inds_mask.any(0)
                    if keep_inds_mask.sum() < topk:
                        num_to_add = topk - keep_inds_mask.sum()
                        pad_inds = (~keep_inds_mask).nonzero()[:num_to_add]
                        keep_inds_mask[pad_inds] = True
                    keep_inds_topk = keep_inds[keep_inds_mask]
                    topk_proposals.append(keep_inds_topk)
                topk_proposals = torch.stack(topk_proposals)
            else:
                topk_proposals = torch.topk(proposal_logit, topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_pos, query = torch.split(pos_trans_out, c, dim=2)
            topk_feats = torch.stack([output_memory[b][topk_proposals[b]] for b in range(bs)]).detach()
            query = query + self.pix_trans_norm(self.pix_trans(topk_feats))
        else:
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_pos).sigmoid()
            init_reference_out = reference_points
        inter_states, inter_references = self.decoder(query=query, key=None, value=memory, query_pos=query_pos, key_padding_mask=mask_flatten, reference_points=reference_points, spatial_shapes=spatial_shapes, level_start_index=level_start_index, valid_ratios=valid_ratios, **kwargs)
        inter_references_out = inter_references
        if self.as_two_stage:
            return inter_states, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact, output_proposals.sigmoid()
        return inter_states, init_reference_out, inter_references_out, None, None, None


def sample_topk_per_gt(pr_inds, gt_inds, iou, k):
    if len(gt_inds) == 0:
        return pr_inds, gt_inds
    gt_inds2, counts = gt_inds.unique(return_counts=True)
    scores, pr_inds2 = iou[gt_inds2].topk(k, dim=1)
    gt_inds2 = gt_inds2[:, None].repeat(1, k)
    pr_inds3 = torch.cat([pr[:c] for c, pr in zip(counts, pr_inds2)])
    gt_inds3 = torch.cat([gt[:c] for c, gt in zip(counts, gt_inds2)])
    return pr_inds3, gt_inds3


class Stage2Assigner(nn.Module):

    def __init__(self, num_queries, max_k=4):
        super().__init__()
        self.positive_fraction = 0.25
        self.bg_label = 400
        self.batch_size_per_image = num_queries
        self.proposal_matcher = Matcher(thresholds=[0.6], labels=[0, 1], allow_low_quality_matches=True)
        self.k = max_k

    def _sample_proposals(self, matched_idxs: 'torch.Tensor', matched_labels: 'torch.Tensor', gt_classes: 'torch.Tensor'):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            gt_classes[matched_labels == 0] = self.bg_label
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.bg_label
        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(gt_classes, self.batch_size_per_image, self.positive_fraction, self.bg_label)
        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    def forward(self, outputs, targets, return_cost_matrix=False):
        bs = len(targets)
        indices = []
        ious = []
        for b in range(bs):
            iou, _ = box_iou(box_cxcywh_to_xyxy(targets[b]['boxes']), box_cxcywh_to_xyxy(outputs['init_reference'][b].detach()))
            matched_idxs, matched_labels = self.proposal_matcher(iou)
            sampled_idxs, sampled_gt_classes = self._sample_proposals(matched_idxs, matched_labels, targets[b]['labels'])
            pos_pr_inds = sampled_idxs[sampled_gt_classes != self.bg_label]
            pos_gt_inds = matched_idxs[pos_pr_inds]
            pos_pr_inds, pos_gt_inds = self.postprocess_indices(pos_pr_inds, pos_gt_inds, iou)
            indices.append((pos_pr_inds, pos_gt_inds))
            ious.append(iou)
        if return_cost_matrix:
            return indices, ious
        return indices

    def postprocess_indices(self, pr_inds, gt_inds, iou):
        return sample_topk_per_gt(pr_inds, gt_inds, iou, self.k)


class Stage1Assigner(nn.Module):

    def __init__(self, t_low=0.3, t_high=0.7, max_k=4):
        super().__init__()
        self.positive_fraction = 0.5
        self.batch_size_per_image = 256
        self.k = max_k
        self.t_low = t_low
        self.t_high = t_high
        self.anchor_matcher = Matcher(thresholds=[t_low, t_high], labels=[0, -1, 1], allow_low_quality_matches=True)

    def _subsample_labels(self, label):
        """
        Randomly sample a subset of positive and negative examples, and overwrite
        the label vector to the ignore value (-1) for all elements that are not
        included in the sample.

        Args:
            labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
        """
        pos_idx, neg_idx = subsample_labels(label, self.batch_size_per_image, self.positive_fraction, 0)
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    def forward(self, outputs, targets):
        bs = len(targets)
        indices = []
        for b in range(bs):
            anchors = outputs['anchors'][b]
            if len(targets[b]['boxes']) == 0:
                indices.append((torch.tensor([], dtype=torch.long, device=anchors.device), torch.tensor([], dtype=torch.long, device=anchors.device)))
                continue
            iou, _ = box_iou(box_cxcywh_to_xyxy(targets[b]['boxes']), box_cxcywh_to_xyxy(anchors))
            matched_idxs, matched_labels = self.anchor_matcher(iou)
            matched_labels = self._subsample_labels(matched_labels)
            all_pr_inds = torch.arange(len(anchors), device=anchors.device)
            pos_pr_inds = all_pr_inds[matched_labels == 1]
            pos_gt_inds = matched_idxs[pos_pr_inds]
            pos_ious = iou[pos_gt_inds, pos_pr_inds]
            pos_pr_inds, pos_gt_inds = self.postprocess_indices(pos_pr_inds, pos_gt_inds, iou)
            pos_pr_inds, pos_gt_inds = pos_pr_inds, pos_gt_inds
            indices.append((pos_pr_inds, pos_gt_inds))
        return indices

    def postprocess_indices(self, pr_inds, gt_inds, iou):
        return sample_topk_per_gt(pr_inds, gt_inds, iou, self.k)


class DETACriterion(SetCriterion):
    """This class computes the loss for Deformable-DETR
    and two-stage Deformable-DETR
    """

    def __init__(self, num_classes, matcher, weight_dict, losses: 'List[str]'=['class', 'boxes'], eos_coef: 'float'=0.1, loss_class_type: 'str'='focal_loss', alpha: 'float'=0.25, gamma: 'float'=2.0, num_queries: 'int'=300, assign_first_stage: 'bool'=False, assign_second_stage: 'bool'=False):
        super(DETACriterion, self).__init__(num_classes=num_classes, matcher=matcher, weight_dict=weight_dict, losses=losses, eos_coef=eos_coef, loss_class_type=loss_class_type, alpha=alpha, gamma=gamma)
        self.assign_first_stage = assign_first_stage
        self.assign_second_stage = assign_second_stage
        if self.assign_first_stage:
            self.stg1_assigner = Stage1Assigner()
        if self.assign_second_stage:
            self.stg2_assigner = Stage2Assigner(num_queries)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        if self.assign_second_stage:
            indices = self.stg2_assigner(outputs_without_aux, targets)
        else:
            indices = self.matcher(outputs_without_aux, targets)
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if not self.assign_second_stage:
                    indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {(k + f'_{i}'): v for k, v in l_dict.items()}
                    losses.update(l_dict)
        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            if self.assign_first_stage:
                indices = self.stg1_assigner(enc_outputs, bin_targets)
            else:
                indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {(k + '_enc'): v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses


class DETR(nn.Module):
    """Implement DETR in `End-to-End Object Detection with Transformers
    <https://arxiv.org/abs/2005.12872>`_

    Args:
        backbone (nn.Module): Backbone module for feature extraction.
        in_features (List[str]): Selected backbone output features for transformer module.
        in_channels (int): Dimension of the last feature in `in_features`.
        position_embedding (nn.Module): Position encoding layer for generating position embeddings.
        transformer (nn.Module): Transformer module used for further processing features
            and input queries.
        embed_dim (int): Hidden dimension for transformer module.
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        device (str): Training device. Default: "cuda".
    """

    def __init__(self, backbone: 'nn.Module', in_features: 'List[str]', in_channels: 'int', position_embedding: 'nn.Module', transformer: 'nn.Module', embed_dim: 'int', num_classes: 'int', num_queries: 'int', criterion: 'nn.Module', aux_loss: 'bool'=True, pixel_mean: 'List[float]'=[123.675, 116.28, 103.53], pixel_std: 'List[float]'=[58.395, 57.12, 57.375], device: 'str'='cuda'):
        super().__init__()
        self.backbone = backbone
        self.in_features = in_features
        self.position_embedding = position_embedding
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.transformer = transformer
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.class_embed = nn.Linear(embed_dim, num_classes + 1)
        self.bbox_embed = MLP(input_dim=embed_dim, hidden_dim=embed_dim, output_dim=4, num_layers=3)
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        self.criterion = criterion
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

    def forward(self, batched_inputs):
        """Forward function of `DAB-DETR` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries.
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        images = self.preprocess_image(batched_inputs)
        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]['instances'].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)
        features = self.backbone(images.tensor)[self.in_features[-1]]
        features = self.input_proj(features)
        img_masks = F.interpolate(img_masks[None], size=features.shape[-2:])[0]
        pos_embed = self.position_embedding(img_masks)
        hidden_states, _ = self.transformer(features, img_masks, self.query_embed.weight, pos_embed)
        outputs_class = self.class_embed(hidden_states)
        outputs_coord = self.bbox_embed(hidden_states).sigmoid()
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            output['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        if self.training:
            gt_instances = [x['instances'] for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output['pred_logits']
            box_pred = output['pred_boxes']
            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get('height', image_size[0])
                width = input_per_image.get('width', image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({'instances': r})
            return processed_results

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def inference(self, box_cls, box_pred, image_sizes):
        """Inference function for DETR

        Args:
            box_cls (torch.Tensor): tensor of shape ``(batch_size, num_queries, K)``.
                The tensor predicts the classification probability for each query.
            box_pred (torch.Tensor): tensors of shape ``(batch_size, num_queries, 4)``.
                The tensor predicts 4-vector ``(x, y, w, h)`` box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []
        scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)
        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(scores, labels, box_pred, image_sizes)):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({'labels': gt_classes, 'boxes': gt_boxes})
        return new_targets

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x['image']) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images


class DetrTransformerEncoder(TransformerLayerSequence):

    def __init__(self, embed_dim: 'int'=256, num_heads: 'int'=8, attn_dropout: 'float'=0.1, feedforward_dim: 'int'=2048, ffn_dropout: 'float'=0.1, num_layers: 'int'=6, post_norm: 'bool'=True, batch_first: 'bool'=False):
        super(DetrTransformerEncoder, self).__init__(transformer_layers=BaseTransformerLayer(attn=MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_dropout, batch_first=batch_first), ffn=FFN(embed_dim=embed_dim, feedforward_dim=feedforward_dim, ffn_drop=ffn_dropout), norm=nn.LayerNorm(normalized_shape=embed_dim), operation_order=('self_attn', 'norm', 'ffn', 'norm')), num_layers=num_layers)
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm
        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(self, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, **kwargs):
        for layer in self.layers:
            query = layer(query, key, value, query_pos=query_pos, key_pos=key_pos, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, **kwargs)
        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class DetrTransformerDecoder(TransformerLayerSequence):

    def __init__(self, embed_dim: 'int'=256, num_heads: 'int'=8, attn_dropout: 'float'=0.1, feedforward_dim: 'int'=2048, ffn_dropout: 'float'=0.1, num_layers: 'int'=6, post_norm: 'bool'=True, return_intermediate: 'bool'=True, batch_first: 'bool'=False):
        super(DetrTransformerDecoder, self).__init__(transformer_layers=BaseTransformerLayer(attn=MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_dropout, batch_first=batch_first), ffn=FFN(embed_dim=embed_dim, feedforward_dim=feedforward_dim, ffn_drop=ffn_dropout), norm=nn.LayerNorm(normalized_shape=embed_dim), operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')), num_layers=num_layers)
        self.return_intermediate = return_intermediate
        self.embed_dim = self.layers[0].embed_dim
        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(self, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, **kwargs):
        if not self.return_intermediate:
            for layer in self.layers:
                query = layer(query, key, value, query_pos=query_pos, key_pos=key_pos, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, **kwargs)
            if self.post_norm_layer is not None:
                query = self.post_norm_layer(query)[None]
            return query
        intermediate = []
        for layer in self.layers:
            query = layer(query, key, value, query_pos=query_pos, key_pos=key_pos, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, **kwargs)
            if self.return_intermediate:
                if self.post_norm_layer is not None:
                    intermediate.append(self.post_norm_layer(query))
                else:
                    intermediate.append(query)
        return torch.stack(intermediate)


class DetrTransformer(nn.Module):

    def __init__(self, encoder=None, decoder=None):
        super(DetrTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dim = self.encoder.embed_dim
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask, query_embed, pos_embed):
        bs, c, h, w = x.shape
        x = x.view(bs, c, -1).permute(2, 0, 1)
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.view(bs, -1)
        memory = self.encoder(query=x, key=None, value=None, query_pos=pos_embed, query_key_padding_mask=mask)
        target = torch.zeros_like(query_embed)
        decoder_output = self.decoder(query=target, key=memory, value=memory, key_pos=pos_embed, query_pos=query_embed, key_padding_mask=mask)
        decoder_output = decoder_output.transpose(1, 2)
        memory = memory.permute(1, 2, 0).reshape(bs, c, h, w)
        return decoder_output, memory


class DINO(nn.Module):
    """Implement DAB-Deformable-DETR in `DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR
    <https://arxiv.org/abs/2203.03605>`_.

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        backbone (nn.Module): backbone module
        position_embedding (nn.Module): position embedding module
        neck (nn.Module): neck module to handle the intermediate outputs features
        transformer (nn.Module): transformer module
        embed_dim (int): dimension of embedding
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 300.
        device (str): Training device. Default: "cuda".
    """

    def __init__(self, backbone: 'nn.Module', position_embedding: 'nn.Module', neck: 'nn.Module', transformer: 'nn.Module', embed_dim: 'int', num_classes: 'int', num_queries: 'int', criterion: 'nn.Module', pixel_mean: 'List[float]'=[123.675, 116.28, 103.53], pixel_std: 'List[float]'=[58.395, 57.12, 57.375], aux_loss: 'bool'=True, select_box_nums_for_evaluation: 'int'=300, device='cuda', dn_number: 'int'=100, label_noise_ratio: 'float'=0.2, box_noise_scale: 'float'=1.0, input_format: 'Optional[str]'='RGB', vis_period: 'int'=0):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.neck = neck
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.transformer = transformer
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        self.criterion = criterion
        self.label_enc = nn.Embedding(num_classes, embed_dim)
        self.dn_number = dn_number
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.device = device
        self.pixel_mean = torch.Tensor(pixel_mean).view(3, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(3, 1, 1)
        self.normalizer = lambda x: (x - self.pixel_mean) / self.pixel_std
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)
        num_pred = transformer.decoder.num_layers + 1
        self.class_embed = nn.ModuleList([copy.deepcopy(self.class_embed) for i in range(num_pred)])
        self.bbox_embed = nn.ModuleList([copy.deepcopy(self.bbox_embed) for i in range(num_pred)])
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.bbox_embed = self.bbox_embed
        for bbox_embed_layer in self.bbox_embed:
            nn.init.constant_(bbox_embed_layer.layers[-1].bias.data[2:], 0.0)
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation
        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, 'input_format is required for visualization!'

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def forward(self, batched_inputs):
        """Forward function of `DINO` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries (anchor boxes in DAB-DETR).
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        images, img_size = self.preprocess_image(batched_inputs)
        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]['instances'].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = img_size[img_id][0], img_size[img_id][1]
                img_masks[img_id, :img_h, :img_w] = 0
        features = self.backbone(images.tensor)
        multi_level_feats = self.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(F.interpolate(img_masks[None], size=feat.shape[-2:]).squeeze(0))
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))
        if self.training:
            gt_instances = [x['instances'] for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_cdn(targets, dn_number=self.dn_number, label_noise_ratio=self.label_noise_ratio, box_noise_scale=self.box_noise_scale, num_queries=self.num_queries, num_classes=self.num_classes, hidden_dim=self.embed_dim, label_enc=self.label_enc)
        else:
            input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None
        query_embeds = input_query_label, input_query_bbox
        inter_states, init_reference, inter_references, enc_state, enc_reference = self.transformer(multi_level_feats, multi_level_masks, multi_level_position_embeddings, query_embeds, attn_masks=[attn_mask, None])
        inter_states[0] += self.label_enc.weight[0, 0] * 0.0
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        if dn_meta is not None:
            outputs_class, outputs_coord = self.dn_post_process(outputs_class, outputs_coord, dn_meta)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            output['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        interm_coord = enc_reference
        interm_class = self.transformer.decoder.class_embed[-1](enc_state)
        output['enc_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
        if self.training:
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    box_cls = output['pred_logits']
                    box_pred = output['pred_boxes']
                    results = self.inference(box_cls, box_pred, images.image_sizes)
                    self.visualize_training(batched_inputs, results)
            loss_dict = self.criterion(output, targets, dn_meta)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output['pred_logits']
            box_pred = output['pred_boxes']
            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get('height', image_size[0])
                width = input_per_image.get('width', image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({'instances': r})
            return processed_results

    def visualize_training(self, batched_inputs, results):
        storage = get_event_storage()
        max_vis_box = 20
        for input, results_per_image in zip(batched_inputs, results):
            img = input['image']
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input['instances'].gt_boxes)
            anno_img = v_gt.get_image()
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(boxes=results_per_image.pred_boxes[:max_vis_box].tensor.detach().cpu().numpy())
            pred_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, pred_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = 'Left: GT bounding boxes;  Right: Predicted boxes'
            storage.put_image(vis_name, vis_img)
            break

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def prepare_for_cdn(self, targets, dn_number, label_noise_ratio, box_noise_scale, num_queries, num_classes, hidden_dim, label_enc):
        """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding
            in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
        if dn_number <= 0:
            return None, None, None, None
        dn_number = dn_number * 2
        known = [torch.ones_like(t['labels']) for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            return None, None, None, None
        dn_number = dn_number // int(max(known_num) * 2)
        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t['labels'] for t in targets])
        boxes = torch.cat([t['boxes'] for t in targets])
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)
        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()
        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < label_noise_ratio * 0.5).view(-1)
            new_label = torch.randint_like(chosen_indice, 0, num_classes)
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_padding = int(max(known_num))
        pad_size = int(single_padding * 2 * dn_number)
        positive_idx = torch.tensor(range(len(boxes))).long().unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2
            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2
            rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part, diff) * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]
        m = known_labels_expaned.long()
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)
        padding_label = torch.zeros(pad_size, hidden_dim)
        padding_bbox = torch.zeros(pad_size, 4)
        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)
        map_known_indice = torch.tensor([])
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])
            map_known_indice = torch.cat([(map_known_indice + single_padding * i) for i in range(2 * dn_number)]).long()
        if len(known_bid):
            input_query_label[known_bid.long(), map_known_indice] = input_label_embed
            input_query_bbox[known_bid.long(), map_known_indice] = input_bbox_embed
        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size) < 0
        attn_mask[pad_size:, :pad_size] = True
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_padding * 2 * i:single_padding * 2 * (i + 1), single_padding * 2 * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_padding * 2 * i:single_padding * 2 * (i + 1), :single_padding * i * 2] = True
            else:
                attn_mask[single_padding * 2 * i:single_padding * 2 * (i + 1), single_padding * 2 * (i + 1):pad_size] = True
                attn_mask[single_padding * 2 * i:single_padding * 2 * (i + 1), :single_padding * 2 * i] = True
        dn_meta = {'single_padding': single_padding * 2, 'dn_num': dn_number}
        return input_query_label, input_query_bbox, attn_mask, dn_meta

    def dn_post_process(self, outputs_class, outputs_coord, dn_metas):
        if dn_metas and dn_metas['single_padding'] > 0:
            padding_size = dn_metas['single_padding'] * dn_metas['dn_num']
            output_known_class = outputs_class[:, :, :padding_size, :]
            output_known_coord = outputs_coord[:, :, :padding_size, :]
            outputs_class = outputs_class[:, :, padding_size:, :]
            outputs_coord = outputs_coord[:, :, padding_size:, :]
            out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(output_known_class, output_known_coord)
            dn_metas['output_known_lbs_bboxes'] = out
        return outputs_class, outputs_coord

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x['image']) for x in batched_inputs]
        images = [((x - self.pixel_mean) / self.pixel_std) for x in images]
        img_size = [[img.shape[1], img.shape[2]] for img in images]
        max_size = 0
        for img in images:
            _, h, w = img.shape
            if max(h, w) > max_size:
                max_size = max(h, w)
        padding_constraints = copy.deepcopy(self.backbone.padding_constraints)
        if 'square_size' in self.backbone.padding_constraints:
            square_size = self.backbone.padding_constraints['square_size']
            if square_size < max_size and square_size != 0:
                warnings.warn('square_size={}, is smaller than max_size={} in batch'.format(self.backbone.padding_constraints['square_size'], max_size))
                padding_constraints['square_size'] = max_size
        images = ImageList.from_tensors(images, self.backbone.size_divisibility, padding_constraints=padding_constraints)
        return images, img_size

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode='floor')
        labels = topk_indexes % box_cls.shape[2]
        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(scores, labels, boxes, image_sizes)):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({'labels': gt_classes, 'boxes': gt_boxes})
        return new_targets


class DINOTransformerEncoder(TransformerLayerSequence):

    def __init__(self, embed_dim: 'int'=256, num_heads: 'int'=8, feedforward_dim: 'int'=1024, attn_dropout: 'float'=0.1, ffn_dropout: 'float'=0.1, num_layers: 'int'=6, post_norm: 'bool'=False, num_feature_levels: 'int'=4, use_checkpoint: 'bool'=False):
        super(DINOTransformerEncoder, self).__init__(transformer_layers=BaseTransformerLayer(attn=MultiScaleDeformableAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True, num_levels=num_feature_levels), ffn=FFN(embed_dim=embed_dim, feedforward_dim=feedforward_dim, output_dim=embed_dim, num_fcs=2, ffn_drop=ffn_dropout), norm=nn.LayerNorm(embed_dim), operation_order=('self_attn', 'norm', 'ffn', 'norm')), num_layers=num_layers)
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm
        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None
        if use_checkpoint:
            for layer in self.layers:
                layer = checkpoint_wrapper(layer)

    def forward(self, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, **kwargs):
        for layer in self.layers:
            query = layer(query, key, value, query_pos=query_pos, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, **kwargs)
        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class DINOTransformerDecoder(TransformerLayerSequence):

    def __init__(self, embed_dim: 'int'=256, num_heads: 'int'=8, feedforward_dim: 'int'=1024, attn_dropout: 'float'=0.1, ffn_dropout: 'float'=0.1, num_layers: 'int'=6, return_intermediate: 'bool'=True, num_feature_levels: 'int'=4, look_forward_twice: 'bool'=True, use_checkpoint: 'bool'=True):
        super(DINOTransformerDecoder, self).__init__(transformer_layers=BaseTransformerLayer(attn=[MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_dropout, batch_first=True), MultiScaleDeformableAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True, num_levels=num_feature_levels)], ffn=FFN(embed_dim=embed_dim, feedforward_dim=feedforward_dim, output_dim=embed_dim, ffn_drop=ffn_dropout), norm=nn.LayerNorm(embed_dim), operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')), num_layers=num_layers)
        self.return_intermediate = return_intermediate
        self.ref_point_head = MLP(2 * embed_dim, embed_dim, embed_dim, 2)
        self.bbox_embed = None
        self.class_embed = None
        self.look_forward_twice = look_forward_twice
        self.norm = nn.LayerNorm(embed_dim)
        if use_checkpoint:
            for layer in self.layers:
                layer = checkpoint_wrapper(layer)

    def forward(self, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, reference_points=None, valid_ratios=None, **kwargs):
        output = query
        bs, num_queries, _ = output.size()
        if reference_points.dim() == 2:
            reference_points = reference_points.unsqueeze(0).repeat(bs, 1, 1)
        intermediate = []
        intermediate_reference_points = []
        for layer_idx, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]
            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)
            output = layer(output, key, value, query_pos=query_pos, key_pos=key_pos, query_sine_embed=query_sine_embed, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, reference_points=reference_points_input, **kwargs)
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[layer_idx](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                if self.look_forward_twice:
                    intermediate_reference_points.append(new_reference_points)
                else:
                    intermediate_reference_points.append(reference_points)
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        return output, reference_points


class DINOTransformer(nn.Module):
    """Transformer module for DINO

    Args:
        encoder (nn.Module): encoder module.
        decoder (nn.Module): decoder module.
        as_two_stage (bool): whether to use two-stage transformer. Default False.
        num_feature_levels (int): number of feature levels. Default 4.
        two_stage_num_proposals (int): number of proposals in two-stage transformer. Default 900.
    """

    def __init__(self, encoder=None, decoder=None, num_feature_levels=4, two_stage_num_proposals=900, learnt_init_query=True):
        super(DINOTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals
        self.embed_dim = self.encoder.embed_dim
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dim))
        self.learnt_init_query = learnt_init_query
        if self.learnt_init_query:
            self.tgt_embed = nn.Embedding(self.two_stage_num_proposals, self.embed_dim)
        self.enc_output = nn.Linear(self.embed_dim, self.embed_dim)
        self.enc_output_norm = nn.LayerNorm(self.embed_dim)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.normal_(self.level_embeds)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N, S, C = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:_cur + H * W].view(N, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device), torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * 2.0 ** lvl
            proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
            proposals.append(proposal)
            _cur += H * W
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The ratios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has                 shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device), torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid ratios of feature maps of all levels."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, multi_level_feats, multi_level_masks, multi_level_pos_embeds, query_embed, attn_masks, **kwargs):
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = h, w
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in multi_level_masks], 1)
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=feat.device)
        memory = self.encoder(query=feat_flatten, key=None, value=None, query_pos=lvl_pos_embed_flatten, query_key_padding_mask=mask_flatten, spatial_shapes=spatial_shapes, reference_points=reference_points, level_start_index=level_start_index, valid_ratios=valid_ratios, **kwargs)
        output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
        topk = self.two_stage_num_proposals
        topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]
        topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
        reference_points = topk_coords_unact.detach().sigmoid()
        if query_embed[1] is not None:
            reference_points = torch.cat([query_embed[1].sigmoid(), reference_points], 1)
        init_reference_out = reference_points
        target_unact = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
        if self.learnt_init_query:
            target = self.tgt_embed.weight[None].repeat(bs, 1, 1)
        else:
            target = target_unact.detach()
        if query_embed[0] is not None:
            target = torch.cat([query_embed[0], target], 1)
        inter_states, inter_references = self.decoder(query=target, key=memory, value=memory, query_pos=None, key_padding_mask=mask_flatten, reference_points=reference_points, spatial_shapes=spatial_shapes, level_start_index=level_start_index, valid_ratios=valid_ratios, attn_masks=attn_masks, **kwargs)
        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out, target_unact, topk_coords_unact.sigmoid()


class DINOCriterion(TwoStageCriterion):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def forward(self, outputs, targets, dn_metas=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = super(DINOCriterion, self).forward(outputs, targets)
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        aux_num = 0
        if 'aux_outputs' in outputs:
            aux_num = len(outputs['aux_outputs'])
        dn_losses = self.compute_dn_loss(dn_metas, targets, aux_num, num_boxes)
        losses.update(dn_losses)
        return losses

    def compute_dn_loss(self, dn_metas, targets, aux_num, num_boxes):
        """
        compute dn loss in criterion
        Args:
            dn_metas: a dict for dn information
            training: training or inference flag
            aux_num: aux loss number
            focal_alpha:  for focal loss
        """
        losses = {}
        if dn_metas and 'output_known_lbs_bboxes' in dn_metas:
            output_known_lbs_bboxes, dn_num, single_padding = dn_metas['output_known_lbs_bboxes'], dn_metas['dn_num'], dn_metas['single_padding']
            dn_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.arange(0, len(targets[i]['labels'])).long()
                    t = t.unsqueeze(0).repeat(dn_num, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(dn_num)) * single_padding).long().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long()
                dn_idx.append((output_idx, tgt_idx))
            l_dict = {}
            for loss in self.losses:
                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_idx, num_boxes * dn_num, **kwargs))
            l_dict = {(k + '_dn'): v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            losses['loss_bbox_dn'] = torch.as_tensor(0.0)
            losses['loss_giou_dn'] = torch.as_tensor(0.0)
            losses['loss_class_dn'] = torch.as_tensor(0.0)
        for i in range(aux_num):
            l_dict = {}
            if dn_metas and 'output_known_lbs_bboxes' in dn_metas:
                output_known_lbs_bboxes_aux = output_known_lbs_bboxes['aux_outputs'][i]
                for loss in self.losses:
                    kwargs = {}
                    if 'labels' in loss:
                        kwargs = {'log': False}
                    l_dict.update(self.get_loss(loss, output_known_lbs_bboxes_aux, targets, dn_idx, num_boxes * dn_num, **kwargs))
                l_dict = {(k + f'_dn_{i}'): v for k, v in l_dict.items()}
            else:
                l_dict['loss_bbox_dn'] = torch.as_tensor(0.0)
                l_dict['loss_giou_dn'] = torch.as_tensor(0.0)
                l_dict['loss_class_dn'] = torch.as_tensor(0.0)
                l_dict = {(k + f'_{i}'): v for k, v in l_dict.items()}
            losses.update(l_dict)
        return losses


class DNCriterion(SetCriterion):
    """This class computes the loss for DN-DETR."""

    def forward(self, outputs, targets):
        losses = super(DNCriterion, self).forward(outputs, targets)
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        aux_num = 0
        if 'aux_outputs' in outputs:
            aux_num = len(outputs['aux_outputs'])
        dn_losses = self.calculate_dn_loss(outputs, targets, aux_num, num_boxes)
        losses.update(dn_losses)
        return losses

    def calculate_dn_loss(self, outputs, targets, aux_num, num_boxes):
        """
        Calculate dn loss in criterion
        """
        losses = {}
        if outputs and 'denoising_output' in outputs:
            denoising_output, denoising_groups, single_padding = outputs['denoising_output'], outputs['denoising_groups'], outputs['max_gt_num_per_image']
            dn_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.arange(0, len(targets[i]['labels'])).long()
                    t = t.unsqueeze(0).repeat(denoising_groups, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(denoising_groups)).cuda() * single_padding).long().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long()
                dn_idx.append((output_idx, tgt_idx))
            l_dict = {}
            for loss in self.losses:
                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(self.get_loss(loss, denoising_output, targets, dn_idx, num_boxes * denoising_groups, **kwargs))
            l_dict = {(k + '_dn'): v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            losses['loss_bbox_dn'] = torch.as_tensor(0.0)
            losses['loss_giou_dn'] = torch.as_tensor(0.0)
            losses['loss_class_dn'] = torch.as_tensor(0.0)
        for i in range(aux_num):
            l_dict = {}
            if outputs and 'denoising_output' in outputs:
                denoising_output_aux = denoising_output['aux_outputs'][i]
                for loss in self.losses:
                    kwargs = {}
                    if 'labels' in loss:
                        kwargs = {'log': False}
                    l_dict.update(self.get_loss(loss, denoising_output_aux, targets, dn_idx, num_boxes * denoising_groups, **kwargs))
                l_dict = {(k + f'_dn_{i}'): v for k, v in l_dict.items()}
            else:
                l_dict['loss_bbox_dn'] = torch.as_tensor(0.0)
                l_dict['loss_giou_dn'] = torch.as_tensor(0.0)
                l_dict['loss_class_dn'] = torch.as_tensor(0.0)
                l_dict = {(k + f'_{i}'): v for k, v in l_dict.items()}
            losses.update(l_dict)
        return losses


class DNDeformableDETR(nn.Module):
    """Implement DAB-Deformable-DETR in `DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR
    <https://arxiv.org/abs/2203.01305>`_.
    Code is modified from the `official github repo
    <https://github.com/IDEA-opensource/DN-DETR>`_.
    Args:
        backbone (nn.Module): backbone module
        position_embedding (nn.Module): position embedding module
        neck (nn.Module): neck module
        transformer (nn.Module): transformer module
        embed_dim (int): dimension of embedding
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 100.
        device (str): Training device. Default: "cuda".
    """

    def __init__(self, backbone, position_embedding, neck, transformer, num_classes, num_queries, criterion, pixel_mean, pixel_std, embed_dim=256, aux_loss=True, as_two_stage=False, denoising_groups: 'int'=5, label_noise_prob: 'float'=0.2, box_noise_scale: 'float'=0.4, with_indicator: 'bool'=True, select_box_nums_for_evaluation: 'int'=300, device='cuda'):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.neck = neck
        self.num_queries = num_queries
        if not as_two_stage:
            self.tgt_embed = nn.Embedding(num_queries, embed_dim - 1)
            self.refpoint_embed = nn.Embedding(num_queries, 4)
        self.transformer = transformer
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self.num_classes = num_classes
        self.as_two_stage = as_two_stage
        self.denoising_generator = GenerateDNQueries(num_queries=num_queries, num_classes=num_classes + 1, label_embed_dim=embed_dim, denoising_groups=denoising_groups, label_noise_prob=label_noise_prob, box_noise_scale=box_noise_scale, with_indicator=with_indicator)
        self.with_indicator = with_indicator
        self.denoising_groups = denoising_groups
        self.label_noise_prob = label_noise_prob
        self.box_noise_scale = box_noise_scale
        if not as_two_stage:
            if self.with_indicator:
                self.tgt_embed = nn.Embedding(num_queries, embed_dim - 1)
            else:
                self.tgt_embed = nn.Embedding(num_queries, embed_dim)
            self.refpoint_embed = nn.Embedding(num_queries, 4)
            nn.init.zeros_(self.tgt_embed.weight)
            nn.init.uniform_(self.refpoint_embed.weight)
            self.refpoint_embed.weight.data[:] = inverse_sigmoid(self.refpoint_embed.weight.data[:]).clamp(-3, 3)
        self.aux_loss = aux_loss
        self.criterion = criterion
        self.as_two_stage = as_two_stage
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)
        num_pred = transformer.decoder.num_layers + 1 if as_two_stage else transformer.decoder.num_layers
        self.class_embed = nn.ModuleList([copy.deepcopy(self.class_embed) for i in range(num_pred)])
        self.bbox_embed = nn.ModuleList([copy.deepcopy(self.bbox_embed) for i in range(num_pred)])
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.bbox_embed = self.bbox_embed
        if self.as_two_stage:
            for bbox_embed_layer in self.bbox_embed:
                nn.init.constant_(bbox_embed_layer.layers[-1].bias.data[2:], 0.0)
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation

    def forward(self, batched_inputs):
        """Forward function of `DN-Deformable-DETR` which excepts a list of dict as inputs.
        Args:
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries (anchor boxes in DAB-DETR).
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        images = self.preprocess_image(batched_inputs)
        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]['instances'].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)
        features = self.backbone(images.tensor)
        multi_level_feats = self.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(F.interpolate(img_masks[None], size=feat.shape[-2:]).squeeze(0))
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))
        if self.training:
            gt_instances = [x['instances'] for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            gt_labels_list = [t['labels'] for t in targets]
            gt_boxes_list = [t['boxes'] for t in targets]
        else:
            targets = None
        matching_label_query = self.tgt_embed.weight
        if self.with_indicator:
            indicator_for_matching_part = torch.zeros([self.num_queries, 1])
            matching_label_query = torch.cat([matching_label_query, indicator_for_matching_part], 1)
        matching_label_query = matching_label_query.repeat(batch_size, 1, 1)
        matching_box_query = self.refpoint_embed.weight.repeat(batch_size, 1, 1)
        if targets is None:
            input_label_query = matching_label_query
            input_box_query = matching_box_query
            attn_mask = None
            denoising_groups = self.denoising_groups
            max_gt_num_per_image = 0
        else:
            noised_label_queries, noised_box_queries, attn_mask, denoising_groups, max_gt_num_per_image = self.denoising_generator(gt_labels_list, gt_boxes_list)
            input_label_query = torch.cat([noised_label_queries, matching_label_query], 1)
            input_box_query = torch.cat([noised_box_queries, matching_box_query], 1)
        inter_states, init_reference, inter_references, enc_state, enc_reference = self.transformer(multi_level_feats, multi_level_masks, multi_level_position_embeddings, input_label_query, input_box_query, attn_masks=[attn_mask, None])
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        output = {'denoising_groups': torch.tensor(denoising_groups), 'max_gt_num_per_image': torch.tensor(max_gt_num_per_image)}
        outputs_class, outputs_coord = self.dn_post_process(outputs_class, outputs_coord, output)
        output.update({'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]})
        if self.aux_loss:
            output['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        if self.as_two_stage:
            interm_coord = enc_reference
            interm_class = self.class_embed[-1](enc_state)
            output['enc_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
        if self.training:
            gt_instances = [x['instances'] for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output['pred_logits']
            box_pred = output['pred_boxes']
            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get('height', image_size[0])
                width = input_per_image.get('width', image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({'instances': r})
            return processed_results

    def dn_post_process(self, outputs_class, outputs_coord, output):
        if output and output['max_gt_num_per_image'] > 0:
            padding_size = output['max_gt_num_per_image'] * output['denoising_groups']
            output_known_class = outputs_class[:, :, :padding_size, :]
            output_known_coord = outputs_coord[:, :, :padding_size, :]
            outputs_class = outputs_class[:, :, padding_size:, :]
            outputs_coord = outputs_coord[:, :, padding_size:, :]
            out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(output_known_class, output_known_coord)
            output['denoising_output'] = out
        return outputs_class, outputs_coord

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x['image']) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes
        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode='floor')
        labels = topk_indexes % box_cls.shape[2]
        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(scores, labels, boxes, image_sizes)):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({'labels': gt_classes, 'boxes': gt_boxes})
        return new_targets


class DNDeformableDetrTransformerEncoder(TransformerLayerSequence):

    def __init__(self, embed_dim: 'int'=256, num_heads: 'int'=8, feedforward_dim: 'int'=1024, attn_dropout: 'float'=0.1, ffn_dropout: 'float'=0.1, operation_order: 'tuple'=('self_attn', 'norm', 'ffn', 'norm'), num_layers: 'int'=6, post_norm: 'bool'=False, num_feature_levels: 'int'=4):
        super(DNDeformableDetrTransformerEncoder, self).__init__(transformer_layers=BaseTransformerLayer(attn=MultiScaleDeformableAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True, num_levels=num_feature_levels), ffn=FFN(embed_dim=embed_dim, feedforward_dim=feedforward_dim, output_dim=embed_dim, num_fcs=2, ffn_drop=ffn_dropout), norm=nn.LayerNorm(embed_dim), operation_order=operation_order), num_layers=num_layers)
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm
        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(self, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, **kwargs):
        for layer in self.layers:
            query = layer(query, key, value, query_pos=query_pos, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, **kwargs)
        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class DNDeformableDetrTransformerDecoder(TransformerLayerSequence):

    def __init__(self, embed_dim: 'int'=256, num_heads: 'int'=8, feedforward_dim: 'int'=1024, attn_dropout: 'float'=0.1, ffn_dropout: 'float'=0.1, num_layers: 'int'=6, return_intermediate: 'bool'=True, num_feature_levels: 'int'=4):
        super(DNDeformableDetrTransformerDecoder, self).__init__(transformer_layers=BaseTransformerLayer(attn=[MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_dropout, batch_first=True), MultiScaleDeformableAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True, num_levels=num_feature_levels)], ffn=FFN(embed_dim=embed_dim, feedforward_dim=feedforward_dim, output_dim=embed_dim, ffn_drop=ffn_dropout), norm=nn.LayerNorm(embed_dim), operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')), num_layers=num_layers)
        self.return_intermediate = return_intermediate
        self.query_scale = MLP(embed_dim, embed_dim, embed_dim, 2)
        self.ref_point_head = MLP(2 * embed_dim, embed_dim, embed_dim, 2)
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, reference_points=None, valid_ratios=None, **kwargs):
        output = query
        bs, num_queries, _ = output.size()
        intermediate = []
        intermediate_reference_points = []
        for layer_idx, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]
            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])
            raw_query_pos = self.ref_point_head(query_sine_embed)
            pos_scale = self.query_scale(output) if layer_idx != 0 else 1
            query_pos = pos_scale * raw_query_pos
            output = layer(output, key, value, query_pos=query_pos, key_pos=key_pos, query_sine_embed=query_sine_embed, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, reference_points=reference_points_input, **kwargs)
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[layer_idx](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        return output, reference_points


class DNDeformableDetrTransformer(nn.Module):

    def __init__(self, encoder=None, decoder=None, as_two_stage=False, num_feature_levels=4, two_stage_num_proposals=300):
        super(DNDeformableDetrTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals
        self.embed_dim = self.encoder.embed_dim
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dim))
        if self.as_two_stage:
            self.enc_output = nn.Linear(self.embed_dim, self.embed_dim)
            self.enc_outpout_norm = nn.LayerNorm(self.embed_dim)
            self.pos_trans = nn.Linear(self.embed_dim * 2, self.embed_dim * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dim)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.normal_(self.level_embeds)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N, S, C = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:_cur + H * W].view(N, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device), torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * 2.0 ** lvl
            proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
            proposals.append(proposal)
            _cur += H * W
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.
        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The ratios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has                 shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device), torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid ratios of feature maps of all levels."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_proposal_pos_embed(self, proposals, num_pos_feats=128, temperature=10000):
        """Get the position embedding of proposal."""
        scale = 2 * math.pi
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        proposals = proposals.sigmoid() * scale
        pos = proposals[:, :, :, None] / dim_t
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def forward(self, multi_level_feats, multi_level_masks, multi_level_pos_embeds, input_label_query, input_box_query, attn_masks, **kwargs):
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = h, w
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in multi_level_masks], 1)
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=feat.device)
        memory = self.encoder(query=feat_flatten, key=None, value=None, query_pos=lvl_pos_embed_flatten, query_key_padding_mask=mask_flatten, spatial_shapes=spatial_shapes, reference_points=reference_points, level_start_index=level_start_index, valid_ratios=valid_ratios, **kwargs)
        bs, _, c = memory.shape
        if self.as_two_stage:
            assert input_box_query is None, 'query_embed should be None in two-stage'
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            reference_points = topk_coords_unact.detach().sigmoid()
            init_reference_out = reference_points
            target_unact = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target_unact.detach()
        else:
            reference_points = input_box_query.sigmoid()
            target = input_label_query
            init_reference_out = reference_points
        inter_states, inter_references = self.decoder(query=target, key=memory, value=memory, query_pos=None, attn_masks=attn_masks, key_padding_mask=mask_flatten, reference_points=reference_points, spatial_shapes=spatial_shapes, level_start_index=level_start_index, valid_ratios=valid_ratios, **kwargs)
        inter_references_out = inter_references
        if self.as_two_stage:
            return inter_states, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return inter_states, init_reference_out, inter_references_out, None, None


class DNDETR(nn.Module):
    """Implement DN-DETR in `DN-DETR: Dynamic Anchor Boxes are Better Queries for DETR
    <https://arxiv.org/abs/2201.12329>`_

    Args:
        backbone (nn.Module): Backbone module for feature extraction.
        in_features (List[str]): Selected backbone output features for transformer module.
        in_channels (int): Dimension of the last feature in `in_features`.
        position_embedding (nn.Module): Position encoding layer for generating position embeddings.
        transformer (nn.Module): Transformer module used for further processing features and input queries.
        embed_dim (int): Hidden dimension for transformer module.
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        freeze_anchor_box_centers (bool): If True, freeze the center param ``(x, y)`` for
            the initialized dynamic anchor boxes in format ``(x, y, w, h)``
            and only train ``(w, h)``. Default: True.
        select_box_nums_for_evaluation (int): Select the top-k confidence predicted boxes for inference.
            Default: 300.
        denoising_groups (int): Number of groups for noised ground truths. Default: 5.
        label_noise_prob (float): The probability of the label being noised. Default: 0.2.
        box_noise_scale (float): Scaling factor for box noising. Default: 0.4.
        with_indicator (bool): If True, add indicator in denoising queries part and matching queries part.
            Default: True.
        device (str): Training device. Default: "cuda".
    """

    def __init__(self, backbone: 'nn.Module', in_features: 'List[str]', in_channels: 'int', position_embedding: 'nn.Module', transformer: 'nn.Module', embed_dim: 'int', num_classes: 'int', num_queries: 'int', criterion: 'nn.Module', aux_loss: 'bool'=True, pixel_mean: 'List[float]'=[123.675, 116.28, 103.53], pixel_std: 'List[float]'=[58.395, 57.12, 57.375], freeze_anchor_box_centers: 'bool'=True, select_box_nums_for_evaluation: 'int'=300, denoising_groups: 'int'=5, label_noise_prob: 'float'=0.2, box_noise_scale: 'float'=0.4, with_indicator: 'bool'=True, device='cuda'):
        super(DNDETR, self).__init__()
        self.backbone = backbone
        self.in_features = in_features
        self.position_embedding = position_embedding
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.denoising_generator = GenerateDNQueries(num_queries=num_queries, num_classes=num_classes + 1, label_embed_dim=embed_dim, denoising_groups=denoising_groups, label_noise_prob=label_noise_prob, box_noise_scale=box_noise_scale, with_indicator=with_indicator)
        self.denoising_groups = denoising_groups
        self.label_noise_prob = label_noise_prob
        self.box_noise_scale = box_noise_scale
        self.transformer = transformer
        self.anchor_box_embed = nn.Embedding(num_queries, 4)
        self.num_queries = num_queries
        self.freeze_anchor_box_centers = freeze_anchor_box_centers
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(input_dim=embed_dim, hidden_dim=embed_dim, output_dim=4, num_layers=3)
        self.num_classes = num_classes
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.aux_loss = aux_loss
        self.criterion = criterion
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation
        self.init_weights()

    def init_weights(self):
        """Initialize weights for DN-DETR"""
        if self.freeze_anchor_box_centers:
            self.anchor_box_embed.weight.data[:, :2].uniform_(0, 1)
            self.anchor_box_embed.weight.data[:, :2] = inverse_sigmoid(self.anchor_box_embed.weight.data[:, :2])
            self.anchor_box_embed.weight.data[:, :2].requires_grad = False
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

    def forward(self, batched_inputs):
        """Forward function of `DN-DETR` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries (anchor boxes in DAB-DETR).
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        images = self.preprocess_image(batched_inputs)
        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]['instances'].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)
        features = self.backbone(images.tensor)[self.in_features[-1]]
        features = self.input_proj(features)
        img_masks = F.interpolate(img_masks[None], size=features.shape[-2:])[0]
        pos_embed = self.position_embedding(img_masks)
        if self.training:
            gt_instances = [x['instances'] for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            gt_labels_list = [t['labels'] for t in targets]
            gt_boxes_list = [t['boxes'] for t in targets]
        else:
            targets = None
        matching_label_query = self.denoising_generator.label_encoder(torch.tensor(self.num_classes)).repeat(self.num_queries, 1)
        indicator_for_matching_part = torch.zeros([self.num_queries, 1])
        matching_label_query = torch.cat([matching_label_query, indicator_for_matching_part], 1).repeat(batch_size, 1, 1)
        matching_box_query = self.anchor_box_embed.weight.repeat(batch_size, 1, 1)
        if targets is None:
            input_label_query = matching_label_query.transpose(0, 1)
            input_box_query = matching_box_query.transpose(0, 1)
            attn_mask = None
            denoising_groups = self.denoising_groups
            max_gt_num_per_image = 0
        else:
            noised_label_queries, noised_box_queries, attn_mask, denoising_groups, max_gt_num_per_image = self.denoising_generator(gt_labels_list, gt_boxes_list)
            input_label_query = torch.cat([noised_label_queries, matching_label_query], 1).transpose(0, 1)
            input_box_query = torch.cat([noised_box_queries, matching_box_query], 1).transpose(0, 1)
        hidden_states, reference_boxes = self.transformer(features, img_masks, input_box_query, pos_embed, target=input_label_query, attn_mask=[attn_mask, None])
        reference_boxes = inverse_sigmoid(reference_boxes)
        anchor_box_offsets = self.bbox_embed(hidden_states)
        outputs_coord = (reference_boxes + anchor_box_offsets).sigmoid()
        outputs_class = self.class_embed(hidden_states)
        output = {'denoising_groups': torch.tensor(denoising_groups), 'max_gt_num_per_image': torch.tensor(max_gt_num_per_image)}
        outputs_class, outputs_coord = self.dn_post_process(outputs_class, outputs_coord, output)
        output.update({'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]})
        if self.aux_loss:
            output['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        if self.training:
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output['pred_logits']
            box_pred = output['pred_boxes']
            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get('height', image_size[0])
                width = input_per_image.get('width', image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({'instances': r})
            return processed_results

    def dn_post_process(self, outputs_class, outputs_coord, output):
        if output and output['max_gt_num_per_image'] > 0:
            padding_size = output['max_gt_num_per_image'] * output['denoising_groups']
            output_known_class = outputs_class[:, :, :padding_size, :]
            output_known_coord = outputs_coord[:, :, :padding_size, :]
            outputs_class = outputs_class[:, :, padding_size:, :]
            outputs_coord = outputs_coord[:, :, padding_size:, :]
            out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(output_known_class, output_known_coord)
            output['denoising_output'] = out
        return outputs_class, outputs_coord

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def inference(self, box_cls, box_pred, image_sizes):
        """Inference function for DN-DETR

        Args:
            box_cls (torch.Tensor): tensor of shape ``(batch_size, num_queries, K)``.
                The tensor predicts the classification probability for each query.
            box_pred (torch.Tensor): tensors of shape ``(batch_size, num_queries, 4)``.
                The tensor predicts 4-vector ``(x, y, w, h)`` box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode='floor')
        labels = topk_indexes % box_cls.shape[2]
        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(scores, labels, boxes, image_sizes)):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({'labels': gt_classes, 'boxes': gt_boxes})
        return new_targets

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x['image']) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images


class DNDetrTransformerEncoder(TransformerLayerSequence):

    def __init__(self, embed_dim: 'int'=256, num_heads: 'int'=8, attn_dropout: 'float'=0.0, feedforward_dim: 'int'=2048, ffn_dropout: 'float'=0.0, activation: 'nn.Module'=nn.PReLU(), num_layers: 'int'=None, post_norm: 'bool'=False, batch_first: 'bool'=False):
        super(DNDetrTransformerEncoder, self).__init__(transformer_layers=BaseTransformerLayer(attn=MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_dropout, batch_first=batch_first), ffn=FFN(embed_dim=embed_dim, feedforward_dim=feedforward_dim, ffn_drop=ffn_dropout, activation=activation), norm=nn.LayerNorm(normalized_shape=embed_dim), operation_order=('self_attn', 'norm', 'ffn', 'norm')), num_layers=num_layers)
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm
        self.query_scale = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)
        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(self, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, **kwargs):
        for layer in self.layers:
            position_scales = self.query_scale(query)
            query = layer(query, key, value, query_pos=query_pos * position_scales, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, **kwargs)
        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class DNDetrTransformerDecoder(TransformerLayerSequence):

    def __init__(self, embed_dim: 'int'=256, num_heads: 'int'=8, attn_dropout: 'float'=0.0, feedforward_dim: 'int'=2048, ffn_dropout: 'float'=0.0, activation: 'nn.Module'=nn.PReLU(), num_layers: 'int'=None, modulate_hw_attn: 'bool'=True, post_norm: 'bool'=True, return_intermediate: 'bool'=True, batch_first: 'bool'=False):
        super(DNDetrTransformerDecoder, self).__init__(transformer_layers=BaseTransformerLayer(attn=[ConditionalSelfAttention(embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_dropout, batch_first=batch_first), ConditionalCrossAttention(embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_dropout, batch_first=batch_first)], ffn=FFN(embed_dim=embed_dim, feedforward_dim=feedforward_dim, ffn_drop=ffn_dropout, activation=activation), norm=nn.LayerNorm(normalized_shape=embed_dim), operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')), num_layers=num_layers)
        self.return_intermediate = return_intermediate
        self.embed_dim = self.layers[0].embed_dim
        self.query_scale = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)
        self.ref_point_head = MLP(2 * self.embed_dim, self.embed_dim, self.embed_dim, 2)
        self.bbox_embed = None
        if modulate_hw_attn:
            self.ref_anchor_head = MLP(self.embed_dim, self.embed_dim, 2, 2)
        self.modulate_hw_attn = modulate_hw_attn
        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None
        for idx in range(num_layers - 1):
            self.layers[idx + 1].attentions[1].query_pos_proj = None

    def forward(self, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, anchor_box_embed=None, **kwargs):
        intermediate = []
        reference_points = anchor_box_embed.sigmoid()
        refpoints = [reference_points]
        for idx, layer in enumerate(self.layers):
            obj_center = reference_points[..., :self.embed_dim]
            query_sine_embed = get_sine_pos_embed(obj_center)
            query_pos = self.ref_point_head(query_sine_embed)
            if idx == 0:
                position_transform = 1
            else:
                position_transform = self.query_scale(query)
            query_sine_embed = query_sine_embed[..., :self.embed_dim] * position_transform
            if self.modulate_hw_attn:
                ref_hw_cond = self.ref_anchor_head(query).sigmoid()
                query_sine_embed[..., self.embed_dim // 2:] *= (ref_hw_cond[..., 0] / obj_center[..., 2]).unsqueeze(-1)
                query_sine_embed[..., :self.embed_dim // 2] *= (ref_hw_cond[..., 1] / obj_center[..., 3]).unsqueeze(-1)
            query = layer(query, key, value, query_pos=query_pos, key_pos=key_pos, query_sine_embed=query_sine_embed, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, is_first_layer=idx == 0, **kwargs)
            if self.bbox_embed is not None:
                temp = self.bbox_embed(query)
                temp[..., :self.embed_dim] += inverse_sigmoid(reference_points)
                new_reference_points = temp[..., :self.embed_dim].sigmoid()
                if idx != self.num_layers - 1:
                    refpoints.append(new_reference_points)
                reference_points = new_reference_points.detach()
            if self.return_intermediate:
                if self.post_norm_layer is not None:
                    intermediate.append(self.post_norm_layer(query))
                else:
                    intermediate.append(query)
        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(query)
        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [torch.stack(intermediate).transpose(1, 2), torch.stack(refpoints).transpose(1, 2)]
            else:
                return [torch.stack(intermediate).transpose(1, 2), reference_points.unsqueeze(0).transpose(1, 2)]
        return query.unsqueeze(0)


class DNDetrTransformer(nn.Module):

    def __init__(self, encoder=None, decoder=None):
        super(DNDetrTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dim = self.encoder.embed_dim
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask, anchor_box_embed, pos_embed, target=None, attn_mask=None):
        bs, c, h, w = x.shape
        x = x.view(bs, c, -1).permute(2, 0, 1)
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        mask = mask.view(bs, -1)
        memory = self.encoder(query=x, key=None, value=None, query_pos=pos_embed, query_key_padding_mask=mask)
        hidden_state, references = self.decoder(query=target, key=memory, value=memory, key_pos=pos_embed, attn_masks=attn_mask, anchor_box_embed=anchor_box_embed)
        return hidden_state, references


class FOCUS_DETRCriterion(TwoStageCriterion):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def forward(self, outputs, targets, dn_metas=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = super(FOCUS_DETRCriterion, self).forward(outputs, targets)
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        aux_num = 0
        if 'aux_outputs' in outputs:
            aux_num = len(outputs['aux_outputs'])
        dn_losses = self.compute_dn_loss(dn_metas, targets, aux_num, num_boxes)
        losses.update(dn_losses)
        return losses

    def compute_dn_loss(self, dn_metas, targets, aux_num, num_boxes):
        """
        compute dn loss in criterion
        Args:
            dn_metas: a dict for dn information
            training: training or inference flag
            aux_num: aux loss number
            focal_alpha:  for focal loss
        """
        losses = {}
        if dn_metas and 'output_known_lbs_bboxes' in dn_metas:
            output_known_lbs_bboxes, dn_num, single_padding = dn_metas['output_known_lbs_bboxes'], dn_metas['dn_num'], dn_metas['single_padding']
            dn_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.arange(0, len(targets[i]['labels'])).long()
                    t = t.unsqueeze(0).repeat(dn_num, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(dn_num)) * single_padding).long().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long()
                dn_idx.append((output_idx, tgt_idx))
            l_dict = {}
            for loss in self.losses:
                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_idx, num_boxes * dn_num, **kwargs))
            l_dict = {(k + '_dn'): v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            losses['loss_bbox_dn'] = torch.as_tensor(0.0)
            losses['loss_giou_dn'] = torch.as_tensor(0.0)
            losses['loss_class_dn'] = torch.as_tensor(0.0)
        for i in range(aux_num):
            l_dict = {}
            if dn_metas and 'output_known_lbs_bboxes' in dn_metas:
                output_known_lbs_bboxes_aux = output_known_lbs_bboxes['aux_outputs'][i]
                for loss in self.losses:
                    kwargs = {}
                    if 'labels' in loss:
                        kwargs = {'log': False}
                    l_dict.update(self.get_loss(loss, output_known_lbs_bboxes_aux, targets, dn_idx, num_boxes * dn_num, **kwargs))
                l_dict = {(k + f'_dn_{i}'): v for k, v in l_dict.items()}
            else:
                l_dict['loss_bbox_dn'] = torch.as_tensor(0.0)
                l_dict['loss_giou_dn'] = torch.as_tensor(0.0)
                l_dict['loss_class_dn'] = torch.as_tensor(0.0)
                l_dict = {(k + f'_{i}'): v for k, v in l_dict.items()}
            losses.update(l_dict)
        return losses


class FOCUS_DETR(nn.Module):
    """Implement DAB-Deformable-DETR in `DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR
    <https://arxiv.org/abs/2203.03605>`_.

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        backbone (nn.Module): backbone module
        position_embedding (nn.Module): position embedding module
        neck (nn.Module): neck module to handle the intermediate outputs features
        transformer (nn.Module): transformer module
        embed_dim (int): dimension of embedding
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 300.
        device (str): Training device. Default: "cuda".
    """

    def __init__(self, backbone: 'nn.Module', position_embedding: 'nn.Module', neck: 'nn.Module', transformer: 'nn.Module', embed_dim: 'int', num_classes: 'int', num_queries: 'int', criterion: 'nn.Module', pixel_mean: 'List[float]'=[123.675, 116.28, 103.53], pixel_std: 'List[float]'=[58.395, 57.12, 57.375], aux_loss: 'bool'=True, select_box_nums_for_evaluation: 'int'=300, device='cuda', dn_number: 'int'=100, label_noise_ratio: 'float'=0.2, box_noise_scale: 'float'=1.0):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.neck = neck
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.transformer = transformer
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        self.criterion = criterion
        self.label_enc = nn.Embedding(num_classes, embed_dim)
        self.dn_number = dn_number
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)
        num_pred = transformer.decoder.num_layers + 1
        self.class_embed = nn.ModuleList([copy.deepcopy(self.class_embed) for i in range(num_pred)])
        self.enhance_MCSP_layerlist = nn.ModuleList([self.class_embed[transformer.encoder.num_layers] for i in range(transformer.encoder.num_layers)])
        self.bbox_embed = nn.ModuleList([copy.deepcopy(self.bbox_embed) for i in range(num_pred)])
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
        self.transformer.encoder.enhance_MCSP = self.enhance_MCSP_layerlist
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.bbox_embed = self.bbox_embed
        for bbox_embed_layer in self.bbox_embed:
            nn.init.constant_(bbox_embed_layer.layers[-1].bias.data[2:], 0.0)
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation

    def forward(self, batched_inputs):
        """Forward function of `DINO` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.
        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries (anchor boxes in DAB-DETR).
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        images = self.preprocess_image(batched_inputs)
        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]['instances'].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)
        features = self.backbone(images.tensor)
        multi_level_feats = self.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(F.interpolate(img_masks[None], size=feat.shape[-2:]).squeeze(0))
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))
        if self.training:
            gt_instances = [x['instances'] for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_cdn(targets, dn_number=self.dn_number, label_noise_ratio=self.label_noise_ratio, box_noise_scale=self.box_noise_scale, num_queries=self.num_queries, num_classes=self.num_classes, hidden_dim=self.embed_dim, label_enc=self.label_enc)
        else:
            input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None
        query_embeds = input_query_label, input_query_bbox
        inter_states, init_reference, inter_references, enc_state, enc_reference, temp_backbone_mask_prediction = self.transformer(multi_level_feats, multi_level_masks, multi_level_position_embeddings, query_embeds, attn_masks=[attn_mask, None])
        inter_states[0] += self.label_enc.weight[0, 0] * 0.0
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        if dn_meta is not None:
            outputs_class, outputs_coord = self.dn_post_process(outputs_class, outputs_coord, dn_meta)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            output['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        output['temp_backbone_mask_prediction'] = temp_backbone_mask_prediction
        output['srcs'] = multi_level_feats
        interm_coord = enc_reference
        interm_class = self.transformer.decoder.class_embed[-1](enc_state)
        output['enc_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
        if self.training:
            loss_dict = self.criterion(output, targets, dn_meta)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output['pred_logits']
            box_pred = output['pred_boxes']
            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get('height', image_size[0])
                width = input_per_image.get('width', image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({'instances': r})
            return processed_results

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def prepare_for_cdn(self, targets, dn_number, label_noise_ratio, box_noise_scale, num_queries, num_classes, hidden_dim, label_enc):
        """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding
            in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
        if dn_number <= 0:
            return None, None, None, None
        dn_number = dn_number * 2
        known = [torch.ones_like(t['labels']) for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            return None, None, None, None
        dn_number = dn_number // int(max(known_num) * 2)
        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t['labels'] for t in targets])
        boxes = torch.cat([t['boxes'] for t in targets])
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)
        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()
        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < label_noise_ratio * 0.5).view(-1)
            new_label = torch.randint_like(chosen_indice, 0, num_classes)
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_padding = int(max(known_num))
        pad_size = int(single_padding * 2 * dn_number)
        positive_idx = torch.tensor(range(len(boxes))).long().unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2
            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2
            rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part, diff) * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]
        m = known_labels_expaned.long()
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)
        padding_label = torch.zeros(pad_size, hidden_dim)
        padding_bbox = torch.zeros(pad_size, 4)
        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)
        map_known_indice = torch.tensor([])
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])
            map_known_indice = torch.cat([(map_known_indice + single_padding * i) for i in range(2 * dn_number)]).long()
        if len(known_bid):
            input_query_label[known_bid.long(), map_known_indice] = input_label_embed
            input_query_bbox[known_bid.long(), map_known_indice] = input_bbox_embed
        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size) < 0
        attn_mask[pad_size:, :pad_size] = True
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_padding * 2 * i:single_padding * 2 * (i + 1), single_padding * 2 * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_padding * 2 * i:single_padding * 2 * (i + 1), :single_padding * i * 2] = True
            else:
                attn_mask[single_padding * 2 * i:single_padding * 2 * (i + 1), single_padding * 2 * (i + 1):pad_size] = True
                attn_mask[single_padding * 2 * i:single_padding * 2 * (i + 1), :single_padding * 2 * i] = True
        dn_meta = {'single_padding': single_padding * 2, 'dn_num': dn_number}
        return input_query_label, input_query_bbox, attn_mask, dn_meta

    def dn_post_process(self, outputs_class, outputs_coord, dn_metas):
        if dn_metas and dn_metas['single_padding'] > 0:
            padding_size = dn_metas['single_padding'] * dn_metas['dn_num']
            output_known_class = outputs_class[:, :, :padding_size, :]
            output_known_coord = outputs_coord[:, :, :padding_size, :]
            outputs_class = outputs_class[:, :, padding_size:, :]
            outputs_coord = outputs_coord[:, :, padding_size:, :]
            out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(output_known_class, output_known_coord)
            dn_metas['output_known_lbs_bboxes'] = out
        return outputs_class, outputs_coord

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x['image']) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode='floor')
        labels = topk_indexes % box_cls.shape[2]
        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(scores, labels, boxes, image_sizes)):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({'labels': gt_classes, 'boxes': gt_boxes, 'size': targets_per_image.image_size})
        return new_targets


class Focus_DETR_BaseTransformerLayer(nn.Module):
    """The implementation of Base `TransformerLayer` used in Transformer. Modified
    from `mmcv <https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/bricks/transformer.py>`_.
    It can be built by directly passing the `Attentions`, `FFNs`, `Norms`
    module, which support more flexible cusomization combined with
    `LazyConfig` system. The `BaseTransformerLayer` also supports `prenorm`
    when you specifying the `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .
    Args:
        attn (list[nn.Module] | nn.Module): nn.Module or a list
            contains the attention module used in TransformerLayer.
        ffn (nn.Module): FFN module used in TransformerLayer.
        norm (nn.Module): Normalization layer used in TransformerLayer.
        operation_order (tuple[str]): The execution order of operation in
            transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying the first element as `norm`.
            Default = None.
    """

    def __init__(self, attn: 'List[nn.Module]', ffn: 'nn.Module', norm: 'nn.Module', operation_order: 'tuple'=None):
        super(Focus_DETR_BaseTransformerLayer, self).__init__()
        assert set(operation_order).issubset({'self_attn', 'OESM', 'encoder_cross_attn', 'norm', 'cross_attn', 'ffn'})
        self.topk_sa = 300
        num_attn = operation_order.count('self_attn') + operation_order.count('cross_attn') + operation_order.count('encoder_cross_attn') + operation_order.count('OESM')
        if isinstance(attn, nn.Module):
            attn = [copy.deepcopy(attn) for _ in range(num_attn)]
        else:
            assert len(attn) == num_attn, f'The length of attn (nn.Module or List[nn.Module]) {num_attn}is not consistent with the number of attention in operation_order {operation_order}'
        self.num_attn = num_attn
        self.operation_order = operation_order
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = nn.ModuleList()
        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'OESM', 'encoder_cross_attn', 'cross_attn']:
                self.attentions.append(attn[index])
                index += 1
        self.embed_dim = self.attentions[0].embed_dim
        self.ffns = nn.ModuleList()
        num_ffns = operation_order.count('ffn')
        for _ in range(num_ffns):
            self.ffns.append(copy.deepcopy(ffn))
        self.norms = nn.ModuleList()
        self.norm2 = nn.LayerNorm(self.embed_dim)
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(copy.deepcopy(norm))

    def forward(self, foreground_pre_layer: 'torch.Tensor', score_tgt: 'torch.Tensor', query: 'torch.Tensor', key: 'torch.Tensor'=None, value: 'torch.Tensor'=None, query_pos: 'torch.Tensor'=None, key_pos: 'torch.Tensor'=None, attn_masks: 'List[torch.Tensor]'=None, query_key_padding_mask: 'torch.Tensor'=None, key_padding_mask: 'torch.Tensor'=None, reference_points: 'torch.Tensor'=None, **kwargs):
        """Forward function for `BaseTransformerLayer`.
        **kwargs contains the specific arguments of attentions.
        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` or `(bs, num_query, embed_dim)`
                which should be specified follows the attention module used in
                `BaseTransformerLayer`.
            key (torch.Tensor): Key embeddings used in `Attention`.
            value (torch.Tensor): Value embeddings with the same shape as `key`.
            query_pos (torch.Tensor): The position embedding for `query`.
                Default: None.
            key_pos (torch.Tensor): The position embedding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): A list of 2D ByteTensor used
                in calculation the corresponding attention. The length of
                `attn_masks` should be equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape `(bs, num_query)`. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (torch.Tensor): ByteTensor for `key`, with
                shape `(bs, num_key)`. Default: None.
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f'Use same attn_mask in all attentions in {self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of attn_masks {len(attn_masks)} must be equal to the number of attention in operation_order {self.num_attn}'
        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](query, temp_key, temp_value, identity if self.pre_norm else None, query_pos=query_pos, key_pos=query_pos, attn_mask=attn_masks[attn_index], key_padding_mask=query_key_padding_mask, reference_points=reference_points, **kwargs)
                attn_index += 1
                identity = query
            elif layer == 'OESM':
                ori_tgt = query
                mc_score = score_tgt.max(-1)[0] * foreground_pre_layer
                select_tgt_index = torch.topk(mc_score, self.topk_sa, dim=1)[1]
                select_tgt = torch.gather(query, 1, select_tgt_index.unsqueeze(-1).repeat(1, 1, 256))
                select_pos = torch.gather(query_pos, 1, select_tgt_index.unsqueeze(-1).repeat(1, 1, 256))
                temp_key = temp_value = select_tgt
                tgt2 = self.attentions[attn_index](select_tgt, temp_key, temp_value, identity if self.pre_norm else None, query_pos=select_pos, key_pos=select_pos, reference_points=reference_points, **kwargs)
                tgt2 = self.norm2(tgt2)
                query = ori_tgt.scatter(1, select_tgt_index.unsqueeze(-1).repeat(1, 1, tgt2.size(-1)), tgt2)
                attn_index += 1
                identity = query
            elif layer == 'encoder_cross_attn':
                temp_key = temp_value = value
                query = self.attentions[attn_index](query, temp_key, temp_value, identity if self.pre_norm else None, query_pos=query_pos, key_pos=query_pos, attn_mask=attn_masks[attn_index], key_padding_mask=query_key_padding_mask, reference_points=reference_points, **kwargs)
                attn_index += 1
                identity = query
            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](query, key, value, identity if self.pre_norm else None, query_pos=query_pos, key_pos=key_pos, attn_mask=attn_masks[attn_index], key_padding_mask=key_padding_mask, reference_points=reference_points, **kwargs)
                attn_index += 1
                identity = query
            elif layer == 'ffn':
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1
        return query


class FOCUS_DETRTransformerEncoder(TransformerLayerSequence):

    def __init__(self, embed_dim: 'int'=256, num_heads: 'int'=8, feedforward_dim: 'int'=1024, attn_dropout: 'float'=0.1, ffn_dropout: 'float'=0.1, num_layers: 'int'=6, post_norm: 'bool'=False, num_feature_levels: 'int'=4):
        super(FOCUS_DETRTransformerEncoder, self).__init__(transformer_layers=Focus_DETR_BaseTransformerLayer(attn=[MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_dropout, batch_first=True), MultiScaleDeformableAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True, num_levels=num_feature_levels)], ffn=FFN(embed_dim=embed_dim, feedforward_dim=feedforward_dim, output_dim=embed_dim, num_fcs=2, ffn_drop=ffn_dropout), norm=nn.LayerNorm(embed_dim), operation_order=('OESM', 'encoder_cross_attn', 'norm', 'ffn', 'norm')), num_layers=num_layers)
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm
        self.enhance_MCSP = None
        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(self, backbone_mask_prediction, focus_token_nums, foreground_inds, reference_points, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, **kwargs):
        B_, N_, S_, P_ = reference_points.shape
        ori_reference_points = reference_points
        ori_pos = query_pos
        output = query
        for layer_id, layer in enumerate(self.layers):
            query = torch.gather(output, 1, foreground_inds[layer_id].unsqueeze(-1).repeat(1, 1, output.size(-1)))
            query_pos = torch.gather(ori_pos, 1, foreground_inds[layer_id].unsqueeze(-1).repeat(1, 1, query_pos.size(-1)))
            foreground_pre_layer = torch.gather(backbone_mask_prediction, 1, foreground_inds[layer_id])
            reference_points = torch.gather(ori_reference_points.view(B_, N_, -1), 1, foreground_inds[layer_id].unsqueeze(-1).repeat(1, 1, S_ * P_)).view(B_, -1, S_, P_)
            dropflag = False
            score_tgt = self.enhance_MCSP[layer_id](query)
            query = layer(foreground_pre_layer, score_tgt, query, key, value, query_pos=query_pos, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, reference_points=reference_points, **kwargs)
            outputs = []
            for i in range(foreground_inds[layer_id].shape[0]):
                outputs.append(output[i].scatter(0, foreground_inds[layer_id][i][:focus_token_nums[i]].unsqueeze(-1).repeat(1, query.size(-1)), query[i][:focus_token_nums[i]]))
            output = torch.stack(outputs)
        if self.post_norm_layer is not None:
            output = self.post_norm_layer(output)
        return output


class FOCUS_DETRTransformerDecoder(TransformerLayerSequence):

    def __init__(self, embed_dim: 'int'=256, num_heads: 'int'=8, feedforward_dim: 'int'=1024, attn_dropout: 'float'=0.1, ffn_dropout: 'float'=0.1, num_layers: 'int'=6, return_intermediate: 'bool'=True, num_feature_levels: 'int'=4, look_forward_twice=True):
        super(FOCUS_DETRTransformerDecoder, self).__init__(transformer_layers=BaseTransformerLayer(attn=[MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_dropout, batch_first=True), MultiScaleDeformableAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True, num_levels=num_feature_levels)], ffn=FFN(embed_dim=embed_dim, feedforward_dim=feedforward_dim, output_dim=embed_dim, ffn_drop=ffn_dropout), norm=nn.LayerNorm(embed_dim), operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')), num_layers=num_layers)
        self.return_intermediate = return_intermediate
        self.ref_point_head = MLP(2 * embed_dim, embed_dim, embed_dim, 2)
        self.bbox_embed = None
        self.class_embed = None
        self.look_forward_twice = look_forward_twice
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value, reference_points=None, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, valid_ratios=None, **kwargs):
        output = query
        bs, num_queries, _ = output.size()
        if reference_points.dim() == 2:
            reference_points = reference_points.unsqueeze(0).repeat(bs, 1, 1)
        intermediate = []
        intermediate_reference_points = []
        for layer_idx, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]
            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)
            output = layer(output, key, value, query_pos=query_pos, key_pos=key_pos, query_sine_embed=query_sine_embed, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, reference_points=reference_points_input, **kwargs)
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[layer_idx](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                if self.look_forward_twice:
                    intermediate_reference_points.append(new_reference_points)
                else:
                    intermediate_reference_points.append(reference_points)
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        return output, reference_points


class MaskPredictor(nn.Module):

    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.layer1 = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, h_dim), nn.GELU())
        self.layer2 = nn.Sequential(nn.Linear(h_dim, h_dim // 2), nn.GELU(), nn.Linear(h_dim // 2, h_dim // 4), nn.GELU(), nn.Linear(h_dim // 4, 1))

    def forward(self, x):
        z = self.layer1(x)
        z_local, z_global = torch.split(z, self.h_dim // 2, dim=-1)
        z_global = z_global.mean(dim=1, keepdim=True).expand(-1, z_local.shape[1], -1)
        z = torch.cat([z_local, z_global], dim=-1)
        out = self.layer2(z)
        return out


class FOCUS_DETRTransformer(nn.Module):
    """Transformer module for FOCUS_DETR

    Args:
        encoder (nn.Module): encoder module.
        decoder (nn.Module): decoder module.
        as_two_stage (bool): whether to use two-stage transformer. Default False.
        num_feature_levels (int): number of feature levels. Default 4.
        two_stage_num_proposals (int): number of proposals in two-stage transformer. Default 900.
    """

    def __init__(self, encoder=None, decoder=None, num_feature_levels=4, two_stage_num_proposals=900, learnt_init_query=True):
        super(FOCUS_DETRTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.focus_rho = 0.5
        self.cascade_set = torch.Tensor([1.0, 0.8, 0.6, 0.6, 0.4, 0.2])
        self.alpha = nn.Parameter(data=torch.Tensor(3), requires_grad=True)
        self.alpha.data.uniform_(-0.3, 0.3)
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals
        self.embed_dim = self.encoder.embed_dim
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dim))
        self.learnt_init_query = learnt_init_query
        if self.learnt_init_query:
            self.tgt_embed = nn.Embedding(self.two_stage_num_proposals, self.embed_dim)
        self.enc_output = nn.Linear(self.embed_dim, self.embed_dim)
        self.enc_output_norm = nn.LayerNorm(self.embed_dim)
        self.enc_mask_predictor = MaskPredictor(self.embed_dim, self.embed_dim)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.normal_(self.level_embeds)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The ratios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has                 shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device), torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid ratios of feature maps of all levels."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes, process_output=True):
        """Make region proposals for each multi-scale features considering their shapes and padding masks,
        and project & normalize the encoder outputs corresponding to these proposals.
            - center points: relative grid coordinates in the range of [0.01, 0.99] (additional mask)
            - width/height:  2^(layer_id) * s (s=0.05) / see the appendix A.4

        Tensor shape example:
            Args:
                memory: torch.Size([2, 15060, 256])
                memory_padding_mask: torch.Size([2, 15060])
                spatial_shape: torch.Size([4, 2])
            Returns:
                output_memory: torch.Size([2, 15060, 256])
                    - same shape with memory ( + additional mask + linear layer + layer norm )
                output_proposals: torch.Size([2, 15060, 4])
                    - x, y, w, h
        """
        N_, S_, C_ = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:_cur + H_ * W_].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device), torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * 2.0 ** lvl
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += H_ * W_
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
        output_memory = memory
        if process_output:
            output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
            output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals, (~memory_padding_mask).sum(axis=-1)

    def upsamplelike(self, inputs):
        src, size = inputs
        return F.interpolate(src, size, mode='bilinear', align_corners=True)

    def forward(self, multi_level_feats, multi_level_masks, multi_level_pos_embeds, query_embed, attn_masks, **kwargs):
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = h, w
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in multi_level_masks], 1)
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=feat.device)
        if self.focus_rho:
            backbone_output_memory, backbone_output_proposals, valid_token_nums = self.gen_encoder_output_proposals(feat_flatten + lvl_pos_embed_flatten, mask_flatten, spatial_shapes, process_output=bool(self.focus_rho))
            self.valid_token_nums = valid_token_nums
            focus_token_nums = (valid_token_nums * self.focus_rho).int() + 1
            foreground_topk = int(max(focus_token_nums))
            self.focus_token_nums = focus_token_nums
            encoder_foreground_topk = self.cascade_set * foreground_topk
            foreground_score = []
            for i in range(self.num_feature_levels):
                if i == 0:
                    backbone_lvl = backbone_output_memory[:, level_start_index[self.num_feature_levels - 1]:, :]
                    score_prediction_lvl = self.enc_mask_predictor(backbone_lvl).reshape(bs, 1, spatial_shapes[self.num_feature_levels - 1][0], spatial_shapes[self.num_feature_levels - 1][1])
                    foreground_score.append(score_prediction_lvl.view(bs, -1, 1))
                else:
                    backbone_lvl = backbone_output_memory[:, level_start_index[self.num_feature_levels - i - 1]:level_start_index[self.num_feature_levels - i - 0], :]
                    up_score = self.upsamplelike((score_prediction_lvl, (spatial_shapes[self.num_feature_levels - i - 1][0], spatial_shapes[self.num_feature_levels - i - 1][1])))
                    re_backbone_lvl = backbone_lvl.reshape(bs, spatial_shapes[self.num_feature_levels - i - 1][0], spatial_shapes[self.num_feature_levels - i - 1][1], -1).permute(0, 3, 1, 2)
                    backbone_lvl = backbone_lvl + (re_backbone_lvl * up_score * self.alpha[i - 1]).permute(0, 2, 3, 1).reshape(bs, -1, self.embed_dim)
                    score_prediction_lvl = self.enc_mask_predictor(backbone_lvl).reshape(bs, 1, spatial_shapes[self.num_feature_levels - i - 1][0], spatial_shapes[self.num_feature_levels - i - 1][1])
                    foreground_score.append(score_prediction_lvl)
            backbone_mask_prediction = torch.cat([foreground_score[3 - i].view(bs, -1, 1) for i in range(len(foreground_score))], dim=1)
            temp_backbone_mask_prediction = backbone_mask_prediction
            backbone_mask_prediction = backbone_mask_prediction.squeeze(-1)
            backbone_mask_prediction = backbone_mask_prediction.masked_fill(mask_flatten, backbone_mask_prediction.min())
            foreground_inds = []
            for i in range(len(self.cascade_set)):
                foreground_proposal = torch.topk(backbone_mask_prediction, int(encoder_foreground_topk[i]), dim=1)[1]
                foreground_inds.append(foreground_proposal)
        enc_topk_proposals = enc_refpoint_embed = None
        memory = self.encoder(backbone_mask_prediction=backbone_mask_prediction, focus_token_nums=focus_token_nums, foreground_inds=foreground_inds, reference_points=reference_points, query=feat_flatten, key=None, value=feat_flatten, query_pos=lvl_pos_embed_flatten, query_key_padding_mask=mask_flatten, spatial_shapes=spatial_shapes, level_start_index=level_start_index, valid_ratios=valid_ratios, **kwargs)
        output_memory, output_proposals, _ = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
        topk = self.two_stage_num_proposals
        topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]
        topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
        reference_points = topk_coords_unact.detach().sigmoid()
        if query_embed[1] is not None:
            reference_points = torch.cat([query_embed[1].sigmoid(), reference_points], 1)
        init_reference_out = reference_points
        target_unact = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
        if self.learnt_init_query:
            target = self.tgt_embed.weight[None].repeat(bs, 1, 1)
        else:
            target = target_unact.detach()
        if query_embed[0] is not None:
            target = torch.cat([query_embed[0], target], 1)
        inter_states, inter_references = self.decoder(query=target, key=memory, value=memory, reference_points=reference_points, query_pos=None, key_padding_mask=mask_flatten, spatial_shapes=spatial_shapes, level_start_index=level_start_index, valid_ratios=valid_ratios, attn_masks=attn_masks, **kwargs)
        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out, target_unact, topk_coords_unact.sigmoid(), temp_backbone_mask_prediction


class GroupConditionalSelfAttention(nn.Module):
    """Conditional Self-Attention Module used in Group-DETR

    `Conditional DETR for Fast Training Convergence.
    <https://arxiv.org/pdf/2108.06152.pdf>`_


    Args:
        embed_dim (int): The embedding dimension for attention.
        num_heads (int): The number of attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `MultiheadAttention`.
            Default: 0.0.
        batch_first (bool): if `True`, then the input and output tensor will be
            provided as `(bs, n, embed_dim)`. Default: False. `(n, bs, embed_dim)`
    """

    def __init__(self, embed_dim, num_heads, attn_drop=0.0, proj_drop=0.0, group_nums=11, batch_first=False, **kwargs):
        super(GroupConditionalSelfAttention, self).__init__()
        self.query_content_proj = nn.Linear(embed_dim, embed_dim)
        self.query_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.key_content_proj = nn.Linear(embed_dim, embed_dim)
        self.key_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.group_nums = group_nums
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5
        self.batch_first = batch_first

    def forward(self, query, key=None, value=None, identity=None, query_pos=None, key_pos=None, attn_mask=None, key_padding_mask=None, **kwargs):
        """Forward function for `ConditionalSelfAttention`

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_query, embed_dim)`
            key (torch.Tensor): Key embeddings with shape
                `(num_key, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_key, embed_dim)`
            value (torch.Tensor): Value embeddings with the same shape as `key`.
                Same in `torch.nn.MultiheadAttention.forward`. Default: None.
                If None, the `key` will be used.
            identity (torch.Tensor): The tensor, with the same shape as `query``,
                which will be used for identity addition. Default: None.
                If None, `query` will be used.
            query_pos (torch.Tensor): The position embedding for query, with the
                same shape as `query`. Default: None.
            key_pos (torch.Tensor): The position embedding for key. Default: None.
                If None, and `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`.
            attn_mask (torch.Tensor): ByteTensor mask with shape `(num_query, num_key)`.
                Same as `torch.nn.MultiheadAttention.forward`. Default: None.
            key_padding_mask (torch.Tensor): ByteTensor with shape `(bs, num_key)` which
                indicates which elements within `key` to be ignored in attention.
                Default: None.
        """
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key ismissing in {self.__class__.__name__}.')
        assert query_pos is not None and key_pos is not None, 'query_pos and key_pos must be passed into ConditionalAttention Module'
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            query_pos = query_pos.transpose(0, 1)
            key_pos = key_pos.transpose(0, 1)
            identity = identity.transpose(0, 1)
        query_content = self.query_content_proj(query)
        query_pos = self.query_pos_proj(query_pos)
        key_content = self.key_content_proj(key)
        key_pos = self.key_pos_proj(key_pos)
        value = self.value_proj(value)
        N, B, C = query_content.shape
        q = query_content + query_pos
        k = key_content + key_pos
        v = value
        if self.training:
            q = torch.cat(q.split(N // self.group_nums, dim=0), dim=1)
            k = torch.cat(k.split(N // self.group_nums, dim=0), dim=1)
            v = torch.cat(v.split(N // self.group_nums, dim=0), dim=1)
        q = q.reshape(N, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        k = k.reshape(N, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        v = v.reshape(N, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn.masked_fill_(attn_mask, float('-inf'))
            else:
                attn += attn_mask
        if key_padding_mask is not None:
            attn = attn.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        if not self.batch_first:
            out = out.transpose(0, 1)
        if self.training:
            out = torch.cat(out.split(B, dim=1), dim=0)
        return identity + self.proj_drop(out)


class GroupSetCriterion(nn.Module):
    """This class computes the loss for Group DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, group_nums: 'int'=11, losses: 'List[str]'=['class', 'boxes'], alpha: 'float'=0.25, gamma: 'float'=2.0):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.group_nums = group_nums
        self.weight_dict = weight_dict
        self.losses = losses
        self.alpha = alpha
        self.gamma = gamma

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1], dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_class = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes=num_boxes, alpha=self.alpha, gamma=self.gamma) * src_logits.shape[1]
        losses = {'loss_class': loss_class}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for _, tgt in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {'class': self.loss_labels, 'boxes': self.loss_boxes}
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        group_nums = self.group_nums if self.training else 1
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets, group_nums=group_nums)
        num_boxes = sum(len(t['labels']) for t in targets) * group_nums
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, group_nums=group_nums)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {(k + f'_{i}'): v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class GroupDETR(nn.Module):
    """Implement Group-DETR upon Conditional-DETR in
    `Group DETR: Fast DETR Training with Group-Wise One-to-Many Assignment
    <https://arxiv.org/abs/2207.13085>`_

    Args:
        backbone (nn.Module): Backbone module for feature extraction.
        in_features (List[str]): Selected backbone output features for transformer module.
        in_channels (int): Dimension of the last feature in `in_features`.
        position_embedding (nn.Module): Position encoding layer for generating position embeddings.
        transformer (nn.Module): Transformer module used for further processing features and input queries.
        embed_dim (int): Hidden dimension for transformer module.
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        group_num (int): The number of query groups used in GroupDETR. Default: 11.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        select_box_nums_for_evaluation (int): Select the top-k confidence predicted boxes for inference.
            Default: 300.
        device (str): Training device. Default: "cuda".
    """

    def __init__(self, backbone: 'nn.Module', in_features: 'List[str]', in_channels: 'int', position_embedding: 'nn.Module', transformer: 'nn.Module', embed_dim: 'int', num_classes: 'int', num_queries: 'int', criterion: 'nn.Module', aux_loss: 'bool'=True, group_nums: 'int'=11, pixel_mean: 'List[float]'=[123.675, 116.28, 103.53], pixel_std: 'List[float]'=[58.395, 57.12, 57.375], select_box_nums_for_evaluation: 'int'=300, device: 'str'='cuda'):
        super(GroupDETR, self).__init__()
        self.backbone = backbone
        self.in_features = in_features
        self.position_embedding = position_embedding
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.transformer = transformer
        self.query_embed = nn.Embedding(num_queries * group_nums, embed_dim)
        self.num_queries = num_queries
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(input_dim=embed_dim, hidden_dim=embed_dim, output_dim=4, num_layers=3)
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        self.criterion = criterion
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation
        self.init_weights()

    def init_weights(self):
        """Initialize weights for Conditioanl-DETR."""
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

    def forward(self, batched_inputs):
        """Forward function of `DAB-DETR` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries.
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        images = self.preprocess_image(batched_inputs)
        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]['instances'].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)
        features = self.backbone(images.tensor)[self.in_features[-1]]
        features = self.input_proj(features)
        img_masks = F.interpolate(img_masks[None], size=features.shape[-2:])[0]
        pos_embed = self.position_embedding(img_masks)
        if self.training:
            query_embed_weight = self.query_embed.weight
        else:
            query_embed_weight = self.query_embed.weight[:self.num_queries]
        hidden_states, reference = self.transformer(features, img_masks, query_embed_weight, pos_embed)
        reference_before_sigmoid = inverse_sigmoid(reference)
        outputs_coords = []
        for lvl in range(hidden_states.shape[0]):
            tmp = self.bbox_embed(hidden_states[lvl])
            tmp[..., :2] += reference_before_sigmoid
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
        outputs_coord = torch.stack(outputs_coords)
        outputs_class = self.class_embed(hidden_states)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            output['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        if self.training:
            gt_instances = [x['instances'] for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output['pred_logits']
            box_pred = output['pred_boxes']
            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get('height', image_size[0])
                width = input_per_image.get('width', image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({'instances': r})
            return processed_results

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def inference(self, box_cls, box_pred, image_sizes):
        """Inference function for DAB-DETR

        Args:
            box_cls (torch.Tensor): tensor of shape ``(batch_size, num_queries, K)``.
                The tensor predicts the classification probability for each query.
            box_pred (torch.Tensor): tensors of shape ``(batch_size, num_queries, 4)``.
                The tensor predicts 4-vector ``(x, y, w, h)`` box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode='floor')
        labels = topk_indexes % box_cls.shape[2]
        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(scores, labels, boxes, image_sizes)):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({'labels': gt_classes, 'boxes': gt_boxes})
        return new_targets

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x['image']) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images


class GroupDetrTransformerEncoder(TransformerLayerSequence):

    def __init__(self, embed_dim: 'int'=256, num_heads: 'int'=8, attn_dropout: 'float'=0.1, feedforward_dim: 'int'=2048, ffn_dropout: 'float'=0.1, activation: 'nn.Module'=nn.PReLU(), post_norm: 'bool'=False, num_layers: 'int'=6, batch_first: 'bool'=False):
        super(GroupDetrTransformerEncoder, self).__init__(transformer_layers=BaseTransformerLayer(attn=MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_dropout, batch_first=batch_first), ffn=FFN(embed_dim=embed_dim, feedforward_dim=feedforward_dim, ffn_drop=ffn_dropout, activation=activation), norm=nn.LayerNorm(normalized_shape=embed_dim), operation_order=('self_attn', 'norm', 'ffn', 'norm')), num_layers=num_layers)
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm
        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(self, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, **kwargs):
        for layer in self.layers:
            query = layer(query, key, value, query_pos=query_pos, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, **kwargs)
        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class GroupDetrTransformerDecoder(TransformerLayerSequence):

    def __init__(self, embed_dim: 'int'=256, num_heads: 'int'=8, attn_dropout: 'float'=0.0, feedforward_dim: 'int'=2048, ffn_dropout: 'float'=0.0, activation: 'nn.Module'=nn.PReLU(), group_nums: 'int'=11, num_layers: 'int'=None, batch_first: 'bool'=False, post_norm: 'bool'=True, return_intermediate: 'bool'=True):
        super(GroupDetrTransformerDecoder, self).__init__(transformer_layers=BaseTransformerLayer(attn=[GroupConditionalSelfAttention(embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_dropout, group_nums=group_nums, batch_first=batch_first), ConditionalCrossAttention(embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_dropout, batch_first=batch_first)], ffn=FFN(embed_dim=embed_dim, feedforward_dim=feedforward_dim, ffn_drop=ffn_dropout, activation=activation), norm=nn.LayerNorm(normalized_shape=embed_dim), operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')), num_layers=num_layers)
        self.return_intermediate = return_intermediate
        self.embed_dim = self.layers[0].embed_dim
        self.query_scale = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)
        self.ref_point_head = MLP(self.embed_dim, self.embed_dim, 2, 2)
        self.bbox_embed = None
        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None
        for idx in range(num_layers - 1):
            self.layers[idx + 1].attentions[1].query_pos_proj = None

    def forward(self, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, **kwargs):
        intermediate = []
        reference_points_before_sigmoid = self.ref_point_head(query_pos)
        reference_points: 'torch.Tensor' = reference_points_before_sigmoid.sigmoid().transpose(0, 1)
        for idx, layer in enumerate(self.layers):
            obj_center = reference_points[..., :2].transpose(0, 1)
            if idx == 0:
                position_transform = 1
            else:
                position_transform = self.query_scale(query)
            query_sine_embed = get_sine_pos_embed(obj_center)
            query_sine_embed = query_sine_embed[..., :self.embed_dim] * position_transform
            query: 'torch.Tensor' = layer(query, key, value, query_pos=query_pos, key_pos=key_pos, query_sine_embed=query_sine_embed, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, is_first_layer=idx == 0, **kwargs)
            if self.return_intermediate:
                if self.post_norm_layer is not None:
                    intermediate.append(self.post_norm_layer(query))
                else:
                    intermediate.append(query)
        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(query)
        if self.return_intermediate:
            return [torch.stack(intermediate).transpose(1, 2), reference_points]
        return query.unsqueeze(0)


class GroupDetrTransformer(nn.Module):

    def __init__(self, encoder=None, decoder=None):
        super(GroupDetrTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dim = self.encoder.embed_dim
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask, query_embed, pos_embed):
        bs, c, h, w = x.shape
        x = x.view(bs, c, -1).permute(2, 0, 1)
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.view(bs, -1)
        memory = self.encoder(query=x, key=None, value=None, query_pos=pos_embed, query_key_padding_mask=mask)
        target = torch.zeros_like(query_embed)
        hidden_state, references = self.decoder(query=target, key=memory, value=memory, key_pos=pos_embed, query_pos=query_embed)
        return hidden_state, references


class GroupHungarianMatcher(nn.Module):
    """HugarianMatcher supports Group-DETR

    Args:
        cost_class (float): The relative weight of the classification error
            in the matching cost. Default: 1.
        cost_bbox (float): The relative weight of the L1 error of the bounding box
            coordinates in the matching cost. Default: 1.
        cost_giou (float): This is the relative weight of the giou loss of
            the bounding box in the matching cost. Default: 1.
    """

    def __init__(self, cost_class: 'float'=1, cost_bbox: 'float'=1, cost_giou: 'float'=1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, 'all costs cant be 0'

    @torch.no_grad()
    def forward(self, outputs, targets, group_nums=1):
        """Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
            group_nums: Number of groups used for matching.
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs['pred_logits'].shape[:2]
        out_prob = outputs['pred_logits'].flatten(0, 1).sigmoid()
        out_bbox = outputs['pred_boxes'].flatten(0, 1)
        tgt_ids = torch.cat([v['labels'] for v in targets])
        tgt_bbox = torch.cat([v['boxes'] for v in targets])
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * out_prob ** gamma * -(1 - out_prob + 1e-08).log()
        pos_cost_class = alpha * (1 - out_prob) ** gamma * -(out_prob + 1e-08).log()
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v['boxes']) for v in targets]
        indices = []
        g_num_queries = num_queries // group_nums
        C_list = C.split(g_num_queries, dim=1)
        for g_i in range(group_nums):
            C_g = C_list[g_i]
            indices_g = [linear_sum_assignment(c[i]) for i, c in enumerate(C_g.split(sizes, -1))]
            if g_i == 0:
                indices = indices_g
            else:
                indices = [(np.concatenate([indice1[0], indice2[0] + g_num_queries * g_i]), np.concatenate([indice1[1], indice2[1]])) for indice1, indice2 in zip(indices, indices_g)]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class HDeformableDETR(nn.Module):
    """Implements the Deformable DETR model.

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2010.04159>`_ .

    Args:
        backbone (nn.Module): the backbone module.
        position_embedding (nn.Module): the position embedding module.
        neck (nn.Module): the neck module.
        transformer (nn.Module): the transformer module.
        embed_dim (int): the dimension of the embedding.
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): whether to use auxiliary loss. Default: True.
        with_box_refine (bool): whether to use box refinement. Default: False.
        as_two_stage (bool): whether to use two-stage. Default: False.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 100.

    """

    def __init__(self, backbone, position_embedding, neck, transformer, embed_dim, num_classes, num_queries_one2one, num_queries_one2many, criterion, pixel_mean, pixel_std, aux_loss=True, with_box_refine=False, as_two_stage=False, select_box_nums_for_evaluation=100, device='cuda', mixed_selection=True, k_one2many=6, lambda_one2many=1.0):
        super().__init__()
        num_queries = num_queries_one2one + num_queries_one2many
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.neck = neck
        self.num_queries = num_queries
        if not as_two_stage:
            self.query_embedding = nn.Embedding(num_queries, embed_dim * 2)
        elif mixed_selection:
            self.query_embedding = nn.Embedding(num_queries, embed_dim)
        self.transformer = transformer
        self.num_classes = num_classes
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self.aux_loss = aux_loss
        self.criterion = criterion
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)
        num_pred = transformer.decoder.num_layers + 1 if as_two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = nn.ModuleList([copy.deepcopy(self.class_embed) for i in range(num_pred)])
            self.bbox_embed = nn.ModuleList([copy.deepcopy(self.bbox_embed) for i in range(num_pred)])
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if as_two_stage:
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.num_queries_one2one = num_queries_one2one
        self.mixed_selection = mixed_selection
        self.k_one2many = k_one2many
        self.lambda_one2many = lambda_one2many

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]['instances'].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)
            save_num_queries = self.num_queries
            save_two_stage_num_proposals = self.transformer.two_stage_num_proposals
            self.num_queries = self.num_queries_one2one
            self.transformer.two_stage_num_proposals = self.num_queries
        features = self.backbone(images.tensor)
        multi_level_feats = self.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(F.interpolate(img_masks[None], size=feat.shape[-2:]).squeeze(0))
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))
        query_embeds = None
        if not self.as_two_stage or self.mixed_selection:
            query_embeds = self.query_embedding.weight[0:self.num_queries, :]
        """ attention mask to prevent information leakage
        """
        self_attn_mask = torch.zeros([self.num_queries, self.num_queries]).bool()
        self_attn_mask[self.num_queries_one2one:, 0:self.num_queries_one2one] = True
        self_attn_mask[0:self.num_queries_one2one, self.num_queries_one2one:] = True
        inter_states, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(multi_level_feats, multi_level_masks, multi_level_position_embeddings, query_embeds, self_attn_mask)
        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_classes_one2many = []
        outputs_coords_one2many = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes_one2one.append(outputs_class[:, 0:self.num_queries_one2one])
            outputs_classes_one2many.append(outputs_class[:, self.num_queries_one2one:])
            outputs_coords_one2one.append(outputs_coord[:, 0:self.num_queries_one2one])
            outputs_coords_one2many.append(outputs_coord[:, self.num_queries_one2one:])
        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)
        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)
        output = {'pred_logits': outputs_classes_one2one[-1], 'pred_boxes': outputs_coords_one2one[-1], 'pred_logits_one2many': outputs_classes_one2many[-1], 'pred_boxes_one2many': outputs_coords_one2many[-1]}
        if self.aux_loss:
            output['aux_outputs'] = self._set_aux_loss(outputs_classes_one2one, outputs_coords_one2one)
            output['aux_outputs_one2many'] = self._set_aux_loss(outputs_classes_one2many, outputs_coords_one2many)
        if self.as_two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            output['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        if self.training:
            gt_instances = [x['instances'] for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            if self.k_one2many > 0:
                loss_dict = self.train_hybrid(output, targets, self.k_one2many, self.criterion, self.lambda_one2many)
            else:
                loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            new_dict = dict()
            for key, value in weight_dict.items():
                new_dict[key] = value
                new_dict[key + '_one2many'] = value
            weight_dict = new_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output['pred_logits']
            box_pred = output['pred_boxes']
            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get('height', image_size[0])
                width = input_per_image.get('width', image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({'instances': r})
            self.num_queries = save_num_queries
            self.transformer.two_stage_num_proposals = save_two_stage_num_proposals
            return processed_results

    def train_hybrid(self, outputs, targets, k_one2many, criterion, lambda_one2many):
        loss_dict = criterion(outputs, targets)
        multi_targets = copy.deepcopy(targets)
        for target in multi_targets:
            target['boxes'] = target['boxes'].repeat(k_one2many, 1)
            target['labels'] = target['labels'].repeat(k_one2many)
        outputs_one2many = dict()
        outputs_one2many['pred_logits'] = outputs['pred_logits_one2many']
        outputs_one2many['pred_boxes'] = outputs['pred_boxes_one2many']
        outputs_one2many['aux_outputs'] = outputs['aux_outputs_one2many']
        loss_dict_one2many = criterion(outputs_one2many, multi_targets)
        for key, value in loss_dict_one2many.items():
            if key + '_one2many' in loss_dict.keys():
                loss_dict[key + '_one2many'] += value * lambda_one2many
            else:
                loss_dict[key + '_one2many'] = value * lambda_one2many
        return loss_dict

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode='floor')
        labels = topk_indexes % box_cls.shape[2]
        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(scores, labels, boxes, image_sizes)):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({'labels': gt_classes, 'boxes': gt_boxes})
        return new_targets

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x['image']) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images


class HDeformableDetrTransformerEncoder(TransformerLayerSequence):

    def __init__(self, embed_dim: 'int'=256, num_heads: 'int'=8, feedforward_dim: 'int'=1024, attn_dropout: 'float'=0.1, ffn_dropout: 'float'=0.1, num_layers: 'int'=6, post_norm: 'bool'=False, num_feature_levels: 'int'=4):
        super(HDeformableDetrTransformerEncoder, self).__init__(transformer_layers=BaseTransformerLayer(attn=MultiScaleDeformableAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True, num_levels=num_feature_levels), ffn=FFN(embed_dim=embed_dim, feedforward_dim=feedforward_dim, output_dim=embed_dim, num_fcs=2, ffn_drop=ffn_dropout), norm=nn.LayerNorm(embed_dim), operation_order=('self_attn', 'norm', 'ffn', 'norm')), num_layers=num_layers)
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm
        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(self, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, **kwargs):
        for layer in self.layers:
            query = layer(query, key, value, query_pos=query_pos, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, **kwargs)
        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class HDeformableDetrTransformerDecoder(TransformerLayerSequence):

    def __init__(self, embed_dim: 'int'=256, num_heads: 'int'=8, feedforward_dim: 'int'=1024, attn_dropout: 'float'=0.1, ffn_dropout: 'float'=0.1, num_layers: 'int'=6, return_intermediate: 'bool'=True, num_feature_levels: 'int'=4, look_forward_twice=True):
        super(HDeformableDetrTransformerDecoder, self).__init__(transformer_layers=BaseTransformerLayer(attn=[MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_dropout, batch_first=True), MultiScaleDeformableAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True, num_levels=num_feature_levels)], ffn=FFN(embed_dim=embed_dim, feedforward_dim=feedforward_dim, output_dim=embed_dim, ffn_drop=ffn_dropout), norm=nn.LayerNorm(embed_dim), operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')), num_layers=num_layers)
        self.return_intermediate = return_intermediate
        self.bbox_embed = None
        self.class_embed = None
        self.look_forward_twice = look_forward_twice

    def forward(self, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, reference_points=None, valid_ratios=None, **kwargs):
        output = query
        intermediate = []
        intermediate_reference_points = []
        for layer_idx, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]
            output = layer(output, key, value, query_pos=query_pos, key_pos=key_pos, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, reference_points=reference_points_input, **kwargs)
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[layer_idx](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(new_reference_points if self.look_forward_twice else reference_points)
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        return output, reference_points


class HDeformableDetrTransformer(nn.Module):
    """Transformer module for Deformable DETR

    Args:
        encoder (nn.Module): encoder module.
        decoder (nn.Module): decoder module.
        as_two_stage (bool): whether to use two-stage transformer. Default False.
        num_feature_levels (int): number of feature levels. Default 4.
        two_stage_num_proposals (int): number of proposals in two-stage transformer. Default 300.
            Only used when as_two_stage is True.
    """

    def __init__(self, encoder=None, decoder=None, num_feature_levels=4, as_two_stage=False, two_stage_num_proposals=300, mixed_selection=True):
        super(HDeformableDetrTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_feature_levels = num_feature_levels
        self.as_two_stage = as_two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.embed_dim = self.encoder.embed_dim
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dim))
        if self.as_two_stage:
            self.enc_output = nn.Linear(self.embed_dim, self.embed_dim)
            self.enc_output_norm = nn.LayerNorm(self.embed_dim)
            self.pos_trans = nn.Linear(self.embed_dim * 2, self.embed_dim * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dim * 2)
        else:
            self.reference_points = nn.Linear(self.embed_dim, 2)
        self.mixed_selection = mixed_selection
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if not self.as_two_stage:
            nn.init.xavier_normal_(self.reference_points.weight.data, gain=1.0)
            nn.init.constant_(self.reference_points.bias.data, 0.0)
        nn.init.normal_(self.level_embeds)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N, S, C = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:_cur + H * W].view(N, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device), torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * 2.0 ** lvl
            proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
            proposals.append(proposal)
            _cur += H * W
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has                 shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device), torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all  level."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_proposal_pos_embed(self, proposals, num_pos_feats=128, temperature=10000):
        """Get the position embedding of proposal."""
        scale = 2 * math.pi
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
        proposals = proposals.sigmoid() * scale
        pos = proposals[:, :, :, None] / dim_t
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def forward(self, multi_level_feats, multi_level_masks, multi_level_pos_embeds, query_embed, self_attn_mask, **kwargs):
        assert self.as_two_stage or query_embed is not None
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = h, w
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in multi_level_masks], 1)
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=feat.device)
        memory = self.encoder(query=feat_flatten, key=None, value=None, query_pos=lvl_pos_embed_flatten, query_key_padding_mask=mask_flatten, spatial_shapes=spatial_shapes, reference_points=reference_points, level_start_index=level_start_index, valid_ratios=valid_ratios, **kwargs)
        bs, _, c = memory.shape
        if self.as_two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            if not self.mixed_selection:
                query_pos, query = torch.split(pos_trans_out, c, dim=2)
            else:
                query = query_embed.unsqueeze(0).expand(bs, -1, -1)
                query_pos, _ = torch.split(pos_trans_out, c, dim=2)
        else:
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_pos).sigmoid()
            init_reference_out = reference_points
        inter_states, inter_references = self.decoder(query=query, key=None, value=memory, query_pos=query_pos, key_padding_mask=mask_flatten, reference_points=reference_points, spatial_shapes=spatial_shapes, level_start_index=level_start_index, valid_ratios=valid_ratios, attn_masks=[self_attn_mask, None], **kwargs)
        inter_references_out = inter_references
        if self.as_two_stage:
            return inter_states, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return inter_states, init_reference_out, inter_references_out, None, None


class FocalNet(nn.Module):
    """ FocalNet backbone.

    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        drop_rate (float): Dropout rate.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        focal_levels (Sequence[int]): Number of focal levels at four stages
        focal_windows (Sequence[int]): Focal window sizes at first focal level at four stages
        use_conv_embed (bool): Whether use overlapped convolution for patch embedding
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, pretrain_img_size=1600, patch_size=4, in_chans=3, embed_dim=96, depths=[2, 2, 6, 2], mlp_ratio=4.0, drop_rate=0.0, drop_path_rate=0.2, norm_layer=nn.LayerNorm, patch_norm=True, out_indices=(0, 1, 2, 3), frozen_stages=-1, focal_levels=[2, 2, 2, 2], focal_windows=[9, 9, 9, 9], use_conv_embed=False, use_postln=False, use_postln_in_modulation=False, use_layerscale=False, normalize_modulator=False, use_checkpoint=False):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None, use_conv_embed=use_conv_embed, is_stem=True)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), depth=depths[i_layer], mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=PatchEmbed if i_layer < self.num_layers - 1 else None, focal_window=focal_windows[i_layer], focal_level=focal_levels[i_layer], use_conv_embed=use_conv_embed, use_postln=use_postln, use_postln_in_modulation=use_postln_in_modulation, normalize_modulator=normalize_modulator, use_layerscale=use_layerscale, use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Forward function."""
        tic = time.time()
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        outs = {}
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs['res{}'.format(i + 2)] = out
        toc = time.time()
        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(FocalNet, self).train(mode)
        self._freeze_stages()


class PatchMerging(nn.Module):
    """Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        x = x.view(B, H, W, C)
        pad_input = H % 2 == 1 or W % 2 == 1
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class SwinTransformer(nn.Module):
    """Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, pretrain_img_size=224, patch_size=4, in_chans=3, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.2, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, out_indices=(0, 1, 2, 3), frozen_stages=-1, use_checkpoint=False):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=PatchMerging if i_layer < self.num_layers - 1 else None, use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False
        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        outs = {}
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs['res{}'.format(i + 2)] = out
        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()


class MaskDINOHead(nn.Module):

    def __init__(self, input_shape: 'Dict[str, ShapeSpec]', *, num_classes: int, pixel_decoder: nn.Module, loss_weight: float=1.0, ignore_value: int=-1, transformer_predictor: nn.Module):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]
        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight
        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.num_classes = num_classes

    def forward(self, features, mask=None, targets=None):
        return self.layers(features, mask, targets=targets)

    def layers(self, features, mask=None, targets=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features, mask)
        predictions = self.predictor(multi_scale_features, mask_features, mask, targets=targets)
        return predictions


class MSDeformAttnTransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device), torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


class MSDeformAttnTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation='relu', n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = MultiScaleDeformableAttention(embed_dim=d_model, num_levels=n_levels, num_heads=n_heads, num_points=n_points, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        src2 = self.self_attn(query=src, query_pos=pos, reference_points=reference_points, value=src, spatial_shapes=spatial_shapes, level_start_index=level_start_index, key_padding_mask=padding_mask)
        src = src2
        src = self.norm1(src)
        src = self.forward_ffn(src)
        return src


class MSDeformAttnTransformerEncoderOnly(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=1024, dropout=0.1, activation='relu', num_feature_levels=4, enc_n_points=4):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points)
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds):
        enable_mask = 0
        if masks is not None:
            for src in srcs:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = h, w
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        return memory, spatial_shapes, level_start_index


class MaskDINOEncoder(nn.Module):
    """
    This is the multi-scale encoder in detection models, also named as pixel decoder in segmentation models.
    """

    def __init__(self, input_shape: 'Dict[str, ShapeSpec]', *, transformer_dropout: float, transformer_nheads: int, transformer_dim_feedforward: int, transformer_enc_layers: int, conv_dim: int, mask_dim: int, norm: Optional[Union[str, Callable]]=None, transformer_in_features: List[str], common_stride: int, num_feature_levels: int, total_num_feature_levels: int, feature_order: str):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()
        transformer_input_shape = {k: v for k, v in input_shape.items() if k in transformer_in_features}
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        self.feature_strides = [v.stride for k, v in input_shape]
        self.feature_channels = [v.channels for k, v in input_shape]
        self.feature_order = feature_order
        if feature_order == 'low2high':
            transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: -x[1].stride)
        else:
            transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
        self.transformer_in_features = [k for k, v in transformer_input_shape]
        transformer_in_channels = [v.channels for k, v in transformer_input_shape]
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape]
        self.maskdino_num_feature_levels = num_feature_levels
        self.total_num_feature_levels = total_num_feature_levels
        self.common_stride = common_stride
        self.transformer_num_feature_levels = len(self.transformer_in_features)
        self.low_resolution_index = transformer_in_channels.index(max(transformer_in_channels))
        self.high_resolution_index = 0 if self.feature_order == 'low2high' else -1
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(nn.Conv2d(in_channels, conv_dim, kernel_size=1), nn.GroupNorm(32, conv_dim)))
            in_channels = max(transformer_in_channels)
            for _ in range(self.total_num_feature_levels - self.transformer_num_feature_levels):
                input_proj_list.append(nn.Sequential(nn.Conv2d(in_channels, conv_dim, kernel_size=3, stride=2, padding=1), nn.GroupNorm(32, conv_dim)))
                in_channels = conv_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([nn.Sequential(nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1), nn.GroupNorm(32, conv_dim))])
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        self.transformer = MSDeformAttnTransformerEncoderOnly(d_model=conv_dim, dropout=transformer_dropout, nhead=transformer_nheads, dim_feedforward=transformer_dim_feedforward, num_encoder_layers=transformer_enc_layers, num_feature_levels=self.total_num_feature_levels)
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.mask_dim = mask_dim
        self.mask_features = Conv2d(conv_dim, mask_dim, kernel_size=1, stride=1, padding=0)
        weight_init.c2_xavier_fill(self.mask_features)
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))
        lateral_convs = []
        output_convs = []
        use_bias = norm == ''
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            lateral_norm = get_norm(norm, conv_dim)
            output_norm = get_norm(norm, conv_dim)
            lateral_conv = Conv2d(in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm)
            output_conv = Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=use_bias, norm=output_norm, activation=F.relu)
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            self.add_module('adapter_{}'.format(idx + 1), lateral_conv)
            self.add_module('layer_{}'.format(idx + 1), output_conv)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    @autocast(enabled=False)
    def forward_features(self, features, masks):
        srcsl = []
        srcs = []
        posl = []
        pos = []
        if self.total_num_feature_levels > self.transformer_num_feature_levels:
            smallest_feat = features[self.transformer_in_features[self.low_resolution_index]].float()
            _len_srcs = self.transformer_num_feature_levels
            for l in range(_len_srcs, self.total_num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](smallest_feat)
                else:
                    src = self.input_proj[l](srcsl[-1])
                srcsl.append(src)
                posl.append(self.pe_layer(src))
        srcsl = srcsl[::-1]
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))
        srcs.extend(srcsl) if self.feature_order == 'low2high' else srcsl.extend(srcs)
        pos.extend(posl) if self.feature_order == 'low2high' else posl.extend(pos)
        if self.feature_order != 'low2high':
            srcs = srcsl
            pos = posl
        y, spatial_shapes, level_start_index = self.transformer(srcs, masks, pos)
        bs = y.shape[0]
        split_size_or_sections = [None] * self.total_num_feature_levels
        for i in range(self.total_num_feature_levels):
            if i < self.total_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)
        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x = features[f].float()
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            y = cur_fpn + F.interpolate(out[self.high_resolution_index], size=cur_fpn.shape[-2:], mode='bilinear', align_corners=False)
            y = output_conv(y)
            out.append(y)
        for o in out:
            if num_cur_levels < self.total_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1
        return self.mask_features(out[-1]), out[0], multi_scale_features


def gen_encoder_output_proposals(memory: 'Tensor', memory_padding_mask: 'Tensor', spatial_shapes: 'Tensor'):
    """
    Input:
        - memory: bs, \\sum{hw}, d_model
        - memory_padding_mask: bs, \\sum{hw}
        - spatial_shapes: nlevel, 2
    Output:
        - output_memory: bs, \\sum{hw}, d_model
        - output_proposals: bs, \\sum{hw}, 4
    """
    N_, S_, C_ = memory.shape
    base_scale = 4.0
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur:_cur + H_ * W_].view(N_, H_, W_, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
        grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device), torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
        wh = torch.ones_like(grid) * 0.05 * 2.0 ** lvl
        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += H_ * W_
    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
    output_memory = memory
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
    return output_memory, output_proposals


class MaskDINODecoder(nn.Module):

    def __init__(self, in_channels, mask_classification=True, *, num_classes: int, hidden_dim: int, num_queries: int, nheads: int, dim_feedforward: int, dec_layers: int, mask_dim: int, enforce_input_project: bool, two_stage: bool, dn: str, noise_scale: float, dn_num: int, initialize_box_type: bool, initial_pred: bool, learn_tgt: bool, total_num_feature_levels: int=4, dropout: float=0.0, activation: str='relu', nhead: int=8, dec_n_points: int=4, return_intermediate_dec: bool=True, query_dim: int=4, dec_layer_share: bool=False, semantic_ce_loss: bool=False):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
            d_model: transformer dimension
            dim_feedforward: feed forward hidden dimension
            dropout: dropout rate
            activation: activation function
            nhead: num heads in multi-head attention
            dec_n_points: number of sampling points in decoder
            return_intermediate_dec: return the intermediate results of decoder
            query_dim: 4 -> (x, y, w, h)
            dec_layer_share: whether to share each decoder layer
        """
        super().__init__()
        assert mask_classification, 'Only support mask classification model'
        self.mask_classification = mask_classification
        self.num_feature_levels = total_num_feature_levels
        self.initial_pred = initial_pred
        self.dn = dn
        self.learn_tgt = learn_tgt
        self.noise_scale = noise_scale
        self.dn_num = dn_num
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.two_stage = two_stage
        self.initialize_box_type = initialize_box_type
        self.total_num_feature_levels = total_num_feature_levels
        self.num_queries = num_queries
        self.semantic_ce_loss = semantic_ce_loss
        if not two_stage or self.learn_tgt:
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
        if not two_stage and initialize_box_type == 'no':
            self.query_embed = nn.Embedding(num_queries, 4)
        if two_stage:
            self.enc_output = nn.Linear(hidden_dim, hidden_dim)
            self.enc_output_norm = nn.LayerNorm(hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())
        self.num_classes = num_classes
        assert self.mask_classification, 'why not class embedding?'
        if self.mask_classification:
            if self.semantic_ce_loss:
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            else:
                self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.label_enc = nn.Embedding(num_classes, hidden_dim)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.decoder_norm = decoder_norm = nn.LayerNorm(hidden_dim)
        decoder_layer = DeformableTransformerDecoderLayer(hidden_dim, dim_feedforward, dropout, activation, self.num_feature_levels, nhead, dec_n_points)
        self.decoder = TransformerDecoder(decoder_layer, self.num_layers, decoder_norm, return_intermediate=return_intermediate_dec, d_model=hidden_dim, query_dim=query_dim, num_feature_levels=self.num_feature_levels, dec_layer_share=dec_layer_share)
        self.hidden_dim = hidden_dim
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        box_embed_layerlist = [_bbox_embed for i in range(self.num_layers)]
        bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.decoder.bbox_embed = bbox_embed

    def prepare_for_dn(self, targets, tgt, refpoint_emb, batch_size):
        if self.training:
            scalar, noise_scale = self.dn_num, self.noise_scale
            known = [torch.ones_like(t['labels']) for t in targets]
            know_idx = [torch.nonzero(t) for t in known]
            known_num = [sum(k) for k in known]
            if max(known_num) > 0:
                scalar = scalar // int(max(known_num))
            else:
                scalar = 0
            if scalar == 0:
                input_query_label = None
                input_query_bbox = None
                attn_mask = None
                mask_dict = None
                return input_query_label, input_query_bbox, attn_mask, mask_dict
            unmask_bbox = unmask_label = torch.cat(known)
            labels = torch.cat([t['labels'] for t in targets])
            boxes = torch.cat([t['boxes'] for t in targets])
            batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
            known_label_indice = torch.nonzero(unmask_label)
            known_label_indice = known_label_indice.view(-1)
            known_bbox_indice = torch.nonzero(unmask_bbox)
            known_bbox_indice = known_bbox_indice.view(-1)
            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            known_indice = known_indice.repeat(scalar, 1).view(-1)
            known_labels = labels.repeat(scalar, 1).view(-1)
            known_bid = batch_idx.repeat(scalar, 1).view(-1)
            known_bboxs = boxes.repeat(scalar, 1)
            known_labels_expaned = known_labels.clone()
            known_bbox_expand = known_bboxs.clone()
            if noise_scale > 0:
                p = torch.rand_like(known_labels_expaned.float())
                chosen_indice = torch.nonzero(p < noise_scale * 0.5).view(-1)
                new_label = torch.randint_like(chosen_indice, 0, self.num_classes)
                known_labels_expaned.scatter_(0, chosen_indice, new_label)
            if noise_scale > 0:
                diff = torch.zeros_like(known_bbox_expand)
                diff[:, :2] = known_bbox_expand[:, 2:] / 2
                diff[:, 2:] = known_bbox_expand[:, 2:]
                known_bbox_expand += torch.mul(torch.rand_like(known_bbox_expand) * 2 - 1.0, diff) * noise_scale
                known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)
            m = known_labels_expaned.long()
            input_label_embed = self.label_enc(m)
            input_bbox_embed = inverse_sigmoid(known_bbox_expand)
            single_pad = int(max(known_num))
            pad_size = int(single_pad * scalar)
            padding_label = torch.zeros(pad_size, self.hidden_dim)
            padding_bbox = torch.zeros(pad_size, 4)
            if not refpoint_emb is None:
                input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
                input_query_bbox = torch.cat([padding_bbox, refpoint_emb], dim=0).repeat(batch_size, 1, 1)
            else:
                input_query_label = padding_label.repeat(batch_size, 1, 1)
                input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)
            map_known_indice = torch.tensor([])
            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])
                map_known_indice = torch.cat([(map_known_indice + single_pad * i) for i in range(scalar)]).long()
            if len(known_bid):
                input_query_label[known_bid.long(), map_known_indice] = input_label_embed
                input_query_bbox[known_bid.long(), map_known_indice] = input_bbox_embed
            tgt_size = pad_size + self.num_queries
            attn_mask = torch.ones(tgt_size, tgt_size) < 0
            attn_mask[pad_size:, :pad_size] = True
            for i in range(scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            mask_dict = {'known_indice': torch.as_tensor(known_indice).long(), 'batch_idx': torch.as_tensor(batch_idx).long(), 'map_known_indice': torch.as_tensor(map_known_indice).long(), 'known_lbs_bboxes': (known_labels, known_bboxs), 'know_idx': know_idx, 'pad_size': pad_size, 'scalar': scalar}
        else:
            if not refpoint_emb is None:
                input_query_label = tgt.repeat(batch_size, 1, 1)
                input_query_bbox = refpoint_emb.repeat(batch_size, 1, 1)
            else:
                input_query_label = None
                input_query_bbox = None
            attn_mask = None
            mask_dict = None
        if not input_query_bbox is None:
            input_query_label = input_query_label
            input_query_bbox = input_query_bbox
        return input_query_label, input_query_bbox, attn_mask, mask_dict

    def dn_post_process(self, outputs_class, outputs_coord, mask_dict, outputs_mask):
        assert mask_dict['pad_size'] > 0
        output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]
        output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]
        outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]
        if outputs_mask is not None:
            output_known_mask = outputs_mask[:, :, :mask_dict['pad_size'], :]
            outputs_mask = outputs_mask[:, :, mask_dict['pad_size']:, :]
        out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1], 'pred_masks': output_known_mask[-1]}
        out['aux_outputs'] = self._set_aux_loss(output_known_class, output_known_mask, output_known_coord)
        mask_dict['output_known_lbs_bboxes'] = out
        return outputs_class, outputs_coord, outputs_mask

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def pred_box(self, reference, hs, ref0=None):
        if ref0 is None:
            outputs_coord_list = []
        else:
            outputs_coord_list = [ref0]
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.decoder.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)
        return outputs_coord_list

    def forward(self, x, mask_features, masks, targets=None):
        assert len(x) == self.num_feature_levels
        size_list = []
        enable_mask = 0
        if masks is not None:
            for src in x:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool) for src in x]
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for i in range(self.num_feature_levels):
            idx = self.num_feature_levels - 1 - i
            bs, c, h, w = x[idx].shape
            size_list.append(x[i].shape[-2:])
            spatial_shapes.append(x[idx].shape[-2:])
            src_flatten.append(self.input_proj[idx](x[idx]).flatten(2).transpose(1, 2))
            mask_flatten.append(masks[i].flatten(1))
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        predictions_class = []
        predictions_mask = []
        if self.two_stage:
            output_memory, output_proposals = gen_encoder_output_proposals(src_flatten, mask_flatten, spatial_shapes)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            enc_outputs_class_unselected = self.class_embed(output_memory)
            enc_outputs_coord_unselected = self.decoder.bbox_embed[0](output_memory) + output_proposals
            topk = self.num_queries
            topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]
            refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            refpoint_embed = refpoint_embed_undetach.detach()
            tgt_undetach = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.hidden_dim))
            outputs_class, outputs_mask = self.forward_prediction_heads(tgt_undetach.transpose(0, 1), mask_features)
            tgt = tgt_undetach.detach()
            if self.learn_tgt:
                tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            interm_outputs = dict()
            interm_outputs['pred_logits'] = outputs_class
            interm_outputs['pred_boxes'] = refpoint_embed_undetach.sigmoid()
            interm_outputs['pred_masks'] = outputs_mask
            if self.initialize_box_type != 'no':
                assert self.initial_pred
                flaten_mask = outputs_mask.detach().flatten(0, 1)
                h, w = outputs_mask.shape[-2:]
                if self.initialize_box_type == 'bitmask':
                    refpoint_embed = BitMasks(flaten_mask > 0).get_bounding_boxes().tensor
                elif self.initialize_box_type == 'mask2box':
                    refpoint_embed = box_ops.masks_to_boxes(flaten_mask > 0)
                else:
                    assert NotImplementedError
                refpoint_embed = box_ops.box_xyxy_to_cxcywh(refpoint_embed) / torch.as_tensor([w, h, w, h], dtype=torch.float)
                refpoint_embed = refpoint_embed.reshape(outputs_mask.shape[0], outputs_mask.shape[1], 4)
                refpoint_embed = inverse_sigmoid(refpoint_embed)
        elif not self.two_stage:
            tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            refpoint_embed = self.query_embed.weight[None].repeat(bs, 1, 1)
        tgt_mask = None
        mask_dict = None
        if self.dn != 'no' and self.training:
            assert targets is not None
            input_query_label, input_query_bbox, tgt_mask, mask_dict = self.prepare_for_dn(targets, None, None, x[0].shape[0])
            if mask_dict is not None:
                tgt = torch.cat([input_query_label, tgt], dim=1)
        if self.initial_pred:
            outputs_class, outputs_mask = self.forward_prediction_heads(tgt.transpose(0, 1), mask_features)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
        if self.dn != 'no' and self.training and mask_dict is not None:
            refpoint_embed = torch.cat([input_query_bbox, refpoint_embed], dim=1)
        hs, references = self.decoder(tgt=tgt.transpose(0, 1), memory=src_flatten.transpose(0, 1), memory_key_padding_mask=mask_flatten, pos=None, refpoints_unsigmoid=refpoint_embed.transpose(0, 1), level_start_index=level_start_index, spatial_shapes=spatial_shapes, valid_ratios=valid_ratios, tgt_mask=tgt_mask)
        for i, output in enumerate(hs):
            outputs_class, outputs_mask = self.forward_prediction_heads(output.transpose(0, 1), mask_features)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
        if self.initial_pred:
            out_boxes = self.pred_box(references, hs, refpoint_embed.sigmoid())
            assert len(predictions_class) == self.num_layers + 1
        else:
            out_boxes = self.pred_box(references, hs)
        if mask_dict is not None:
            predictions_mask = torch.stack(predictions_mask)
            predictions_class = torch.stack(predictions_class)
            predictions_class, out_boxes, predictions_mask = self.dn_post_process(predictions_class, out_boxes, mask_dict, predictions_mask)
            predictions_class, predictions_mask = list(predictions_class), list(predictions_mask)
        elif self.training:
            predictions_class[-1] += 0.0 * self.label_enc.weight.sum()
        out = {'pred_logits': predictions_class[-1], 'pred_masks': predictions_mask[-1], 'pred_boxes': out_boxes[-1], 'aux_outputs': self._set_aux_loss(predictions_class if self.mask_classification else None, predictions_mask, out_boxes)}
        if self.two_stage:
            out['interm_outputs'] = interm_outputs
        return out, mask_dict

    def forward_prediction_heads(self, output, mask_features):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_features)
        return outputs_class, outputs_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, out_boxes=None):
        if out_boxes is None:
            return [{'pred_logits': a, 'pred_masks': b} for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]
        else:
            return [{'pred_logits': a, 'pred_masks': b, 'pred_boxes': c} for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], out_boxes[:-1])]


class PnPDETR(nn.Module):
    """Implement DETR in `End-to-End Object Detection with Transformers
    <https://arxiv.org/abs/2005.12872>`_

    Args:
        backbone (nn.Module): Backbone module for feature extraction.
        in_features (List[str]): Selected backbone output features for transformer module.
        in_channels (int): Dimension of the last feature in `in_features`.
        position_embedding (nn.Module): Position encoding layer for generating position embeddings.
        transformer (nn.Module): Transformer module used for further processing features
            and input queries.
        embed_dim (int): Hidden dimension for transformer module.
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        device (str): Training device. Default: "cuda".
    """

    def __init__(self, backbone: 'nn.Module', in_features: 'List[str]', in_channels: 'int', position_embedding: 'nn.Module', transformer: 'nn.Module', embed_dim: 'int', num_classes: 'int', num_queries: 'int', test_time_sample_ratio: 'float', criterion: 'nn.Module', aux_loss: 'bool'=True, pixel_mean: 'List[float]'=[123.675, 116.28, 103.53], pixel_std: 'List[float]'=[58.395, 57.12, 57.375], device: 'str'='cuda'):
        super().__init__()
        self.backbone = backbone
        self.in_features = in_features
        self.position_embedding = position_embedding
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.transformer = transformer
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.class_embed = nn.Linear(embed_dim, num_classes + 1)
        self.bbox_embed = MLP(input_dim=embed_dim, hidden_dim=embed_dim, output_dim=4, num_layers=3)
        self.num_classes = num_classes
        self.test_time_sample_ratio = test_time_sample_ratio
        self.aux_loss = aux_loss
        self.criterion = criterion
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

    def forward(self, batched_inputs):
        """Forward function of `DAB-DETR` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries.
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        images = self.preprocess_image(batched_inputs)
        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]['instances'].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)
        features = self.backbone(images.tensor)[self.in_features[-1]]
        features = self.input_proj(features)
        img_masks = F.interpolate(img_masks[None], size=features.shape[-2:])[0]
        pos_embed = self.position_embedding(img_masks)
        hidden_states, sample_reg_loss = self.transformer(features, img_masks, self.query_embed.weight, pos_embed, self.test_time_sample_ratio)
        outputs_class = self.class_embed(hidden_states)
        outputs_coord = self.bbox_embed(hidden_states).sigmoid()
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            output['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        output['sample_reg_loss'] = sample_reg_loss
        if self.training:
            gt_instances = [x['instances'] for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            loss_dict = self.criterion(output, targets)
            loss_dict['sample_reg_loss'] = output['sample_reg_loss']
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output['pred_logits']
            box_pred = output['pred_boxes']
            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get('height', image_size[0])
                width = input_per_image.get('width', image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({'instances': r})
            return processed_results

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def inference(self, box_cls, box_pred, image_sizes):
        """Inference function for DETR

        Args:
            box_cls (torch.Tensor): tensor of shape ``(batch_size, num_queries, K)``.
                The tensor predicts the classification probability for each query.
            box_pred (torch.Tensor): tensors of shape ``(batch_size, num_queries, 4)``.
                The tensor predicts 4-vector ``(x, y, w, h)`` box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []
        scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)
        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(scores, labels, box_pred, image_sizes)):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({'labels': gt_classes, 'boxes': gt_boxes})
        return new_targets

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x['image']) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images


class PnPDetrTransformerEncoder(TransformerLayerSequence):

    def __init__(self, embed_dim: 'int'=256, num_heads: 'int'=8, attn_dropout: 'float'=0.1, feedforward_dim: 'int'=2048, ffn_dropout: 'float'=0.1, num_layers: 'int'=6, post_norm: 'bool'=True, batch_first: 'bool'=False):
        super(PnPDetrTransformerEncoder, self).__init__(transformer_layers=BaseTransformerLayer(attn=MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_dropout, batch_first=batch_first), ffn=FFN(embed_dim=embed_dim, feedforward_dim=feedforward_dim, ffn_drop=ffn_dropout), norm=nn.LayerNorm(normalized_shape=embed_dim), operation_order=('self_attn', 'norm', 'ffn', 'norm')), num_layers=num_layers)
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm
        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(self, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, **kwargs):
        for layer in self.layers:
            query = layer(query, key, value, query_pos=query_pos, key_pos=key_pos, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, **kwargs)
        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class PnPDetrTransformerDecoder(TransformerLayerSequence):

    def __init__(self, embed_dim: 'int'=256, num_heads: 'int'=8, attn_dropout: 'float'=0.1, feedforward_dim: 'int'=2048, ffn_dropout: 'float'=0.1, num_layers: 'int'=6, post_norm: 'bool'=True, return_intermediate: 'bool'=True, batch_first: 'bool'=False):
        super(PnPDetrTransformerDecoder, self).__init__(transformer_layers=BaseTransformerLayer(attn=MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_dropout, batch_first=batch_first), ffn=FFN(embed_dim=embed_dim, feedforward_dim=feedforward_dim, ffn_drop=ffn_dropout), norm=nn.LayerNorm(normalized_shape=embed_dim), operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')), num_layers=num_layers)
        self.return_intermediate = return_intermediate
        self.embed_dim = self.layers[0].embed_dim
        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(self, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, **kwargs):
        if not self.return_intermediate:
            for layer in self.layers:
                query = layer(query, key, value, query_pos=query_pos, key_pos=key_pos, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, **kwargs)
            if self.post_norm_layer is not None:
                query = self.post_norm_layer(query)[None]
            return query
        intermediate = []
        for layer in self.layers:
            query = layer(query, key, value, query_pos=query_pos, key_pos=key_pos, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, **kwargs)
            if self.return_intermediate:
                if self.post_norm_layer is not None:
                    intermediate.append(self.post_norm_layer(query))
                else:
                    intermediate.append(query)
        return torch.stack(intermediate)


class SortSampler(nn.Module):

    def __init__(self, topk_ratio, input_dim, score_pred_net='2layer-fc', kproj_net='2layer-fc', unsample_abstract_number=30, pos_embed_kproj=False):
        super().__init__()
        self.topk_ratio = topk_ratio
        if score_pred_net == '2layer-fc-256':
            self.score_pred_net = nn.Sequential(nn.Conv2d(input_dim, input_dim, 1), nn.ReLU(), nn.Conv2d(input_dim, 1, 1))
        elif score_pred_net == '2layer-fc-16':
            self.score_pred_net = nn.Sequential(nn.Conv2d(input_dim, 16, 1), nn.ReLU(), nn.Conv2d(16, 1, 1))
        elif score_pred_net == '1layer-fc':
            self.score_pred_net = nn.Conv2d(input_dim, 1, 1)
        else:
            raise ValueError
        self.norm_feature = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.unsample_abstract_number = unsample_abstract_number
        if kproj_net == '2layer-fc':
            self.k_proj = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, unsample_abstract_number))
        elif kproj_net == '1layer-fc':
            self.k_proj = nn.Linear(input_dim, unsample_abstract_number)
        else:
            raise ValueError
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.pos_embed_kproj = pos_embed_kproj

    def forward(self, src, mask, pos_embed, sample_ratio):
        bs, c, h, w = src.shape
        sample_weight = self.score_pred_net(src).sigmoid().view(bs, -1)
        sample_weight_clone = sample_weight.clone().detach()
        sample_weight_clone[mask] = -1.0
        if sample_ratio == None:
            sample_ratio = self.topk_ratio
        sample_lens = ((~mask).sum(1) * sample_ratio).int()
        max_sample_num = sample_lens.max()
        mask_topk = torch.arange(max_sample_num).expand(len(sample_lens), max_sample_num) > (sample_lens - 1).unsqueeze(1)
        min_sample_num = sample_lens.min()
        sort_order = sample_weight_clone.sort(descending=True, dim=1)[1]
        sort_confidence_topk = sort_order[:, :max_sample_num]
        sort_confidence_topk_remaining = sort_order[:, min_sample_num:]
        src = src.flatten(2).permute(2, 0, 1)
        src = self.norm_feature(src)
        src_sample_remaining = src.gather(0, sort_confidence_topk_remaining.permute(1, 0)[..., None].expand(-1, -1, c))
        mask_unsampled = torch.arange(mask.size(1)).expand(len(sample_lens), mask.size(1)) < sample_lens.unsqueeze(1)
        mask_unsampled = mask_unsampled | mask.gather(1, sort_order)
        mask_unsampled = mask_unsampled[:, min_sample_num:]
        if self.pos_embed_kproj:
            pos_embed_sample_remaining = pos_embed.gather(0, sort_confidence_topk_remaining.permute(1, 0)[..., None].expand(-1, -1, c))
            kproj = self.k_proj(src_sample_remaining + pos_embed_sample_remaining)
        else:
            kproj = self.k_proj(src_sample_remaining)
        kproj = kproj.masked_fill(mask_unsampled.permute(1, 0).unsqueeze(2), float('-inf')).permute(1, 2, 0).softmax(-1)
        abs_unsampled_points = torch.bmm(kproj, self.v_proj(src_sample_remaining).permute(1, 0, 2)).permute(1, 0, 2)
        abs_unsampled_pos_embed = torch.bmm(kproj, pos_embed.gather(0, sort_confidence_topk_remaining.permute(1, 0)[..., None].expand(-1, -1, c)).permute(1, 0, 2)).permute(1, 0, 2)
        abs_unsampled_mask = mask.new_zeros(mask.size(0), abs_unsampled_points.size(0))
        sample_reg_loss = sample_weight.gather(1, sort_confidence_topk).mean()
        src_sampled = src.gather(0, sort_confidence_topk.permute(1, 0)[..., None].expand(-1, -1, c)) * sample_weight.gather(1, sort_confidence_topk).permute(1, 0).unsqueeze(-1)
        pos_embed_sampled = pos_embed.gather(0, sort_confidence_topk.permute(1, 0)[..., None].expand(-1, -1, c))
        mask_sampled = mask_topk
        src = torch.cat([src_sampled, abs_unsampled_points])
        pos_embed = torch.cat([pos_embed_sampled, abs_unsampled_pos_embed])
        mask = torch.cat([mask_sampled, abs_unsampled_mask], dim=1)
        assert ((~mask).sum(1) == sample_lens + self.unsample_abstract_number).all()
        return src, sample_reg_loss, sort_confidence_topk, mask, pos_embed


class PnPDetrTransformer(nn.Module):

    def __init__(self, encoder=None, decoder=None, sample_topk_ratio=1 / 3.0, score_pred_net='2layer-fc-256', kproj_net='2layer-fc', unsample_abstract_number=30, pos_embed_kproj=False):
        super(PnPDetrTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dim = self.encoder.embed_dim
        self.sampler = SortSampler(sample_topk_ratio, self.embed_dim, score_pred_net=score_pred_net, kproj_net=kproj_net, unsample_abstract_number=unsample_abstract_number, pos_embed_kproj=pos_embed_kproj)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask, query_embed, pos_embed, sample_ratio):
        bs, c, h, w = x.shape
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.view(bs, -1)
        x, sample_reg_loss, sort_confidence_topk, mask, pos_embed = self.sampler(x, mask, pos_embed, sample_ratio)
        memory = self.encoder(query=x, key=None, value=None, query_pos=pos_embed, query_key_padding_mask=mask)
        target = torch.zeros_like(query_embed)
        decoder_output = self.decoder(query=target, key=memory, value=memory, key_pos=pos_embed, query_pos=query_embed, key_padding_mask=mask)
        decoder_output = decoder_output.transpose(1, 2)
        return decoder_output, sample_reg_loss


class DabDetrTransformerDecoder_qr(TransformerLayerSequence):

    def __init__(self, embed_dim: 'int'=256, num_heads: 'int'=8, attn_dropout: 'float'=0.0, feedforward_dim: 'int'=2048, ffn_dropout: 'float'=0.0, activation: 'nn.Module'=nn.PReLU(), num_layers: 'int'=None, modulate_hw_attn: 'bool'=True, batch_first: 'bool'=False, post_norm: 'bool'=True, return_intermediate: 'bool'=True):
        super(DabDetrTransformerDecoder_qr, self).__init__(transformer_layers=BaseTransformerLayer(attn=[ConditionalSelfAttention(embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_dropout, batch_first=batch_first), ConditionalCrossAttention(embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_dropout, batch_first=batch_first)], ffn=FFN(embed_dim=embed_dim, feedforward_dim=feedforward_dim, ffn_drop=ffn_dropout, activation=activation), norm=nn.LayerNorm(normalized_shape=embed_dim), operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')), num_layers=num_layers)
        self.return_intermediate = return_intermediate
        self.embed_dim = self.layers[0].embed_dim
        self.query_scale = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)
        self.ref_point_head = MLP(2 * self.embed_dim, self.embed_dim, self.embed_dim, 2)
        self.bbox_embed = None
        if modulate_hw_attn:
            self.ref_anchor_head = MLP(self.embed_dim, self.embed_dim, 2, 2)
        self.modulate_hw_attn = modulate_hw_attn
        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None
        for idx in range(num_layers - 1):
            self.layers[idx + 1].attentions[1].query_pos_proj = None
        self.start_q = [0, 0, 1, 2, 4, 7, 12]
        self.end_q = [1, 2, 4, 7, 12, 20, 33]

    def forward(self, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, anchor_box_embed=None, **kwargs):
        train_mode = anchor_box_embed.requires_grad
        if train_mode:
            result = self.forward_sqr_train(query, key, value, query_pos, key_pos, attn_masks, query_key_padding_mask, key_padding_mask, anchor_box_embed, **kwargs)
        else:
            result = self.forward_regular(query, key, value, query_pos, key_pos, attn_masks, query_key_padding_mask, key_padding_mask, anchor_box_embed, **kwargs)
        return result

    def forward_sqr_train(self, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, anchor_box_embed=None, **kwargs):
        batchsize = key.shape[1]
        intermediate = []
        intermediate_ref_boxes = []
        reference_boxes = anchor_box_embed.sigmoid()
        query_list_reserve = [query]
        reference_boxes_list_reserve = [reference_boxes]
        for idx, layer in enumerate(self.layers):
            start_q = self.start_q[idx]
            end_q = self.end_q[idx]
            query_list = query_list_reserve.copy()[start_q:end_q]
            reference_boxes_list = reference_boxes_list_reserve.copy()[start_q:end_q]
            query = torch.cat(query_list, dim=1)
            reference_boxes = torch.cat(reference_boxes_list, dim=1)
            fakesetsize = int(query.shape[1] / batchsize)
            k_ = key.repeat(1, fakesetsize, 1)
            v_ = value.repeat(1, fakesetsize, 1)
            key_pos_ = key_pos.repeat(1, fakesetsize, 1)
            intermediate_ref_boxes.append(reference_boxes)
            if idx != 0:
                reference_boxes = reference_boxes.detach()
            obj_center = reference_boxes[..., :self.embed_dim]
            query_sine_embed = get_sine_pos_embed(obj_center)
            query_pos = self.ref_point_head(query_sine_embed)
            if idx == 0:
                position_transform = 1
            else:
                position_transform = self.query_scale(query)
            query_sine_embed = query_sine_embed[..., :self.embed_dim] * position_transform
            if self.modulate_hw_attn:
                ref_hw_cond = self.ref_anchor_head(query).sigmoid()
                query_sine_embed[..., self.embed_dim // 2:] *= (ref_hw_cond[..., 0] / obj_center[..., 2]).unsqueeze(-1)
                query_sine_embed[..., :self.embed_dim // 2] *= (ref_hw_cond[..., 1] / obj_center[..., 3]).unsqueeze(-1)
            query = layer(query, k_, v_, query_pos=query_pos, key_pos=key_pos_, query_sine_embed=query_sine_embed, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, is_first_layer=idx == 0, **kwargs)
            if self.bbox_embed is not None:
                offsets = self.bbox_embed(query)
                offsets[..., :self.embed_dim] += inverse_sigmoid(reference_boxes)
                reference_boxes = offsets[..., :self.embed_dim].sigmoid()
            if self.return_intermediate:
                if self.post_norm_layer is not None:
                    intermediate.append(self.post_norm_layer(query))
                else:
                    intermediate.append(query)
            query_list_reserve.extend([_ for _ in torch.split(query, batchsize, dim=1)])
            reference_boxes_list_reserve.extend([_ for _ in torch.split(reference_boxes, batchsize, dim=1)])
        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(query)
        intermediate = [i for s in [list(torch.split(k, batchsize, dim=1)) for k in intermediate] for i in s]
        intermediate_ref_boxes = [i for s in [list(torch.split(k, batchsize, dim=1)) for k in intermediate_ref_boxes] for i in s]
        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [torch.stack(intermediate).transpose(1, 2), torch.stack(intermediate_ref_boxes).transpose(1, 2)]
            else:
                return [torch.stack(intermediate).transpose(1, 2), reference_boxes.unsqueeze(0).transpose(1, 2)]
        return query.unsqueeze(0)

    def forward_regular(self, query, key, value, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, anchor_box_embed=None, **kwargs):
        intermediate = []
        intermediate_ref_boxes = []
        reference_boxes = anchor_box_embed.sigmoid()
        for idx, layer in enumerate(self.layers):
            intermediate_ref_boxes.append(reference_boxes)
            if idx != 0:
                reference_boxes = reference_boxes.detach()
            obj_center = reference_boxes[..., :self.embed_dim]
            query_sine_embed = get_sine_pos_embed(obj_center)
            query_pos = self.ref_point_head(query_sine_embed)
            if idx == 0:
                position_transform = 1
            else:
                position_transform = self.query_scale(query)
            query_sine_embed = query_sine_embed[..., :self.embed_dim] * position_transform
            if self.modulate_hw_attn:
                ref_hw_cond = self.ref_anchor_head(query).sigmoid()
                query_sine_embed[..., self.embed_dim // 2:] *= (ref_hw_cond[..., 0] / obj_center[..., 2]).unsqueeze(-1)
                query_sine_embed[..., :self.embed_dim // 2] *= (ref_hw_cond[..., 1] / obj_center[..., 3]).unsqueeze(-1)
            query = layer(query, key, value, query_pos=query_pos, key_pos=key_pos, query_sine_embed=query_sine_embed, attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask, key_padding_mask=key_padding_mask, is_first_layer=idx == 0, **kwargs)
            if self.bbox_embed is not None:
                offsets = self.bbox_embed(query)
                offsets[..., :self.embed_dim] += inverse_sigmoid(reference_boxes)
                reference_boxes = offsets[..., :self.embed_dim].sigmoid()
            if self.return_intermediate:
                if self.post_norm_layer is not None:
                    intermediate.append(self.post_norm_layer(query))
                else:
                    intermediate.append(query)
        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(query)
        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [torch.stack(intermediate).transpose(1, 2), torch.stack(intermediate_ref_boxes).transpose(1, 2)]
            else:
                return [torch.stack(intermediate).transpose(1, 2), reference_boxes.unsqueeze(0).transpose(1, 2)]
        return query.unsqueeze(0)


class DeformablePositionEmbeddingSine(nn.Module):
    """Position Embedding used in Deformable-DETR"""

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-06
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class DABPositionEmbeddingSine(nn.Module):
    """Position Embedding used in DAB-DETR"""

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-06
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class DABPositionEmbeddingLearned(nn.Module):
    """Position Embedding Learned used in DAB-DETR"""

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, mask):
        h, w = mask.shape[-2:]
        i = torch.arange(w, device=mask.device)
        j = torch.arange(h, device=mask.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([x_emb.unsqueeze(0).repeat(h, 1, 1), y_emb.unsqueeze(1).repeat(1, w, 1)], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos


class OriginalConditionalAttentionEncoder(nn.Module):
    """Original implementation of Conditional Self-Attention

    Remove norm and dropout layer for test simplicity
    """

    def __init__(self, d_model, nhead):
        super().__init__()
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=0.0, vdim=d_model)

    def forward(self, tgt, query_pos):
        q_content = self.sa_qcontent_proj(tgt)
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(tgt)
        q = q_content + q_pos
        k = k_content + k_pos
        tgt2 = self.self_attn(q, k, v)
        return tgt2


class OriginalConditionalAttentionDecoder(nn.Module):
    """Original implementation of Conditional Attention Decoder

    Remove norm and dropout layer for test simplicity
    """

    def __init__(self, d_model, nhead):
        super().__init__()
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=0.0, vdim=d_model)
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model * 2, nhead, dropout=0.0, vdim=d_model)
        self.nhead = nhead

    def forward(self, tgt, memory, query_pos, pos, query_sine_embed, is_first=True):
        q_content = self.sa_qcontent_proj(tgt)
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(tgt)
        q = q_content + q_pos
        k = k_content + k_pos
        tgt2 = self.self_attn(q, k, v)[0]
        tgt = tgt + tgt2
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)
        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape
        k_pos = self.ca_kpos_proj(pos)
        if is_first:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content
        q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model // self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)
        tgt2 = self.cross_attn(query=q, key=k, value=v)[0]
        return tgt2 + tgt


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AttentionPoolingBlock,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (AttentiveBlock,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (CenterFeatureScaleModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (ConvNormAct,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CrossEntropyCost,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)], {})),
    (CrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (DABPositionEmbeddingLearned,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DownsampleLayer,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FFN,
     lambda: ([], {'d_model': 4, 'd_ffn': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FocalLossCost,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)], {})),
    (FocalModulation,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FrozenBatchNorm2d,
     lambda: ([], {'n': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GenerateCDNQueries,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (InternImageBlock,
     lambda: ([], {'core_op': torch.nn.ReLU, 'channels': 4, 'depth': 1, 'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InternImageLayer,
     lambda: ([], {'core_op': torch.nn.ReLU, 'channels': 4, 'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (L1Cost,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (L1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNormWithForceFP32,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLPLayer,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MaskPredictor,
     lambda: ([], {'in_dim': 4, 'h_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PositionEmbeddingLearned,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PositionEmbeddingSine,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ReLUDropout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (StemLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (SwiGLU,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (to_channels_first,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (to_channels_last,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

