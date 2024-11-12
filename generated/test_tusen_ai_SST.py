
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


import math


from torch.utils.data import DistributedSampler as _DistributedSampler


from torch.utils.data import Sampler


import copy


from functools import partial


from torch.utils.data import DataLoader


import random


import warnings


import time


import torch.distributed as dist


from abc import abstractmethod


from enum import IntEnum


from enum import unique


import torch.nn.functional as F


from torch import Tensor


from math import pi as PI


from matplotlib import pyplot as plt


from torch.utils.data import Dataset


from collections import OrderedDict


from scipy import signal


from abc import ABCMeta


from torch import nn as nn


import torch.nn as nn


from torch.utils.checkpoint import checkpoint


from torch import nn


from torch.nn import functional as F


from scipy.sparse.csgraph import connected_components


from torch.nn.functional import l1_loss


from torch.nn.functional import mse_loss


from torch.nn.functional import smooth_l1_loss


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


from torch.autograd import Function


from typing import List


from typing import Tuple


from torch import distributed as dist


from torch.autograd.function import Function


from torch.nn import init


from torch.nn.parameter import Parameter


import itertools


import matplotlib.pyplot as plt


import matplotlib.cm as cm


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.nn import BatchNorm1d


from torch.nn import ReLU


from torch.autograd import gradcheck


from collections import defaultdict


import torch.multiprocessing as mp


from torch.profiler import profile


from torch.profiler import record_function


from torch.profiler import ProfilerActivity


class SIR(nn.Module):

    def __init__(self, num_blocks=5, in_channels=[], feat_channels=[], rel_mlp_hidden_dims=[], with_rel_mlp=True, with_distance=False, with_cluster_center=False, norm_cfg=dict(type='LN', eps=0.001), mode='max', xyz_normalizer=[1.0, 1.0, 1.0], act='relu', dropout=0, unique_once=False):
        super().__init__()
        self.num_blocks = num_blocks
        self.unique_once = unique_once
        block_list = []
        for i in range(num_blocks):
            return_point_feats = i != num_blocks - 1
            kwargs = dict(type='SIRLayer', in_channels=in_channels[i], feat_channels=feat_channels[i], with_distance=with_distance, with_cluster_center=with_cluster_center, with_rel_mlp=with_rel_mlp, rel_mlp_hidden_dims=rel_mlp_hidden_dims[i], with_voxel_center=False, voxel_size=[0.1, 0.1, 0.1], point_cloud_range=[-74.88, -74.88, -2, 74.88, 74.88, 4], norm_cfg=norm_cfg, mode=mode, fusion_layer=None, return_point_feats=return_point_feats, return_inv=False, rel_dist_scaler=10.0, xyz_normalizer=xyz_normalizer, act=act, dropout=dropout)
            encoder = builder.build_voxel_encoder(kwargs)
            block_list.append(encoder)
        self.block_list = nn.ModuleList(block_list)

    def forward(self, points, features, coors, f_cluster=None):
        if self.unique_once:
            new_coors, unq_inv = torch.unique(coors, return_inverse=True, return_counts=False, dim=0)
        else:
            new_coors = unq_inv = None
        out_feats = features
        cluster_feat_list = []
        for i, block in enumerate(self.block_list):
            in_feats = torch.cat([points, out_feats], 1)
            if i < self.num_blocks - 1:
                out_feats, out_cluster_feats = block(in_feats, coors, f_cluster, unq_inv_once=unq_inv, new_coors_once=new_coors)
                cluster_feat_list.append(out_cluster_feats)
            if i == self.num_blocks - 1:
                out_feats, out_cluster_feats, out_coors = block(in_feats, coors, f_cluster, return_both=True, unq_inv_once=unq_inv, new_coors_once=new_coors)
                cluster_feat_list.append(out_cluster_feats)
        final_cluster_feats = torch.cat(cluster_feat_list, dim=1)
        return out_feats, final_cluster_feats, out_coors


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


def flat2window(feat, voxel_drop_lvl, flat2win_inds_dict, drop_info, padding=0):
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
        padding = torch.tensor(padding, dtype=dtype, device=device)
        feat_3d = torch.ones((num_windows * max_tokens, feat_dim), dtype=dtype, device=device) * padding
        feat_3d[this_inds] = feat_this_dl
        feat_3d = feat_3d.reshape((num_windows, max_tokens, feat_dim))
        feat_3d_dict[dl] = feat_3d
    return feat_3d_dict


def flat2window_v2(feat, inds_dict, padding=0):
    assert 'voxel_drop_level' in inds_dict, 'voxel_drop_level should be in inds_dict in v2 function'
    inds_v1 = {k: inds_dict[k] for k in inds_dict if not isinstance(k, str)}
    batching_info = inds_dict['batching_info']
    return flat2window(feat, inds_dict['voxel_drop_level'], inds_v1, batching_info, padding=padding)


def window2flat(feat_3d_dict, inds_dict):
    flat_feat_list = []
    num_all_voxel = 0
    for dl in inds_dict:
        num_all_voxel += inds_dict[dl][0].shape[0]
    dtype = feat_3d_dict[list(feat_3d_dict.keys())[0]].dtype
    device = feat_3d_dict[list(feat_3d_dict.keys())[0]].device
    feat_dim = feat_3d_dict[list(feat_3d_dict.keys())[0]].shape[-1]
    all_flat_feat = torch.zeros((num_all_voxel, feat_dim), device=device, dtype=dtype)
    for dl in feat_3d_dict:
        feat = feat_3d_dict[dl]
        feat_dim = feat.shape[-1]
        inds, flat_pos = inds_dict[dl]
        feat = feat.reshape(-1, feat_dim)
        flat_feat = feat[inds]
        all_flat_feat[flat_pos] = flat_feat
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


class SRABlock(nn.Module):
    """ Consist of two encoder layer, shift and shift back.
    """

    def __init__(self, key, d_model, nhead, dim_feedforward, window_shape, dropout=0.1, activation='relu', batch_first=False, block_id=-100):
        super().__init__()
        encoder_1 = EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first, layer_id=block_id * 2 + 0)
        encoder_2 = EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first, layer_id=block_id * 2 + 1)
        self.encoder_list = nn.ModuleList([encoder_1, encoder_2])
        self.window_shape = window_shape
        self.key = key

    def forward(self, input, batching_info, using_checkpoint=False):
        assert isinstance(input, SRATensor)
        output = input
        if not output.ready:
            output.setup(batching_info, self.key, self.window_shape, 10000)
        for i in range(2):
            layer = self.encoder_list[i]
            if using_checkpoint:
                output = checkpoint(layer, output, i == 1)
            else:
                output = layer(output, i == 1)
        return output


class SST(nn.Module):

    def __init__(self, d_model=[], nhead=[], num_blocks=6, dim_feedforward=[], dropout=0.1, activation='relu', output_shape=None, num_attached_conv=0, conv_in_channel=64, conv_out_channel=64, norm_cfg=dict(type='BN', eps=0.001, momentum=0.01), conv_cfg=dict(type='Conv2d', bias=False), debug=True, batch_first=False, batching_info=None, no_pos_embed=False, normalize_pos=False, pos_temperature=10000, window_shape=None, init_sparse_shape=None, in_channel=None, conv_kwargs=dict(kernel_size=3, dilation=2, padding=2, stride=1), checkpoint_blocks=[], key='single_unique_key', fp16=True):
        super().__init__()
        assert isinstance(batching_info, tuple)
        self.batching_info = batching_info
        self.no_pos_embed = no_pos_embed
        self.pos_temperature = pos_temperature
        self.d_model = d_model
        self.window_shape = window_shape
        self.key = key
        self.normalize_pos = normalize_pos
        self.nhead = nhead
        self.checkpoint_blocks = checkpoint_blocks
        self.init_sparse_shape = init_sparse_shape
        self.fp16 = fp16
        if in_channel is not None:
            self.linear0 = nn.Linear(in_channel, d_model[0])
        block_list = []
        for i in range(num_blocks):
            block_list.append(SRABlock(key, d_model[i], nhead[i], dim_feedforward[i], window_shape, dropout, activation, batch_first=False, block_id=i))
        self.block_list = nn.ModuleList(block_list)
        self._reset_parameters()
        self.output_shape = output_shape
        self.debug = debug
        self.num_attached_conv = num_attached_conv
        if num_attached_conv > 0:
            conv_list = []
            for i in range(num_attached_conv):
                if isinstance(conv_kwargs, dict):
                    conv_kwargs_i = conv_kwargs
                elif isinstance(conv_kwargs, list):
                    assert len(conv_kwargs) == num_attached_conv
                    conv_kwargs_i = conv_kwargs[i]
                if i > 0:
                    conv_in_channel = conv_out_channel
                conv = build_conv_layer(conv_cfg, in_channels=conv_in_channel, out_channels=conv_out_channel, **conv_kwargs_i)
                if norm_cfg is None:
                    convnormrelu = nn.Sequential(conv, nn.ReLU(inplace=True))
                else:
                    convnormrelu = nn.Sequential(conv, build_norm_layer(norm_cfg, conv_out_channel)[1], nn.ReLU(inplace=True))
                conv_list.append(convnormrelu)
            self.conv_layer = nn.ModuleList(conv_list)

    def forward(self, input_tuple):
        """
        Note that, batch_first is set to True
        Args:
        feat_3d_list: list[Tensor of shape(bs, max_num_token, embed_dim)]
        Outs:
        output: list[Tensor of shape: (bs, embed_dim, h, w)]
                output tensor is in bev view
        """
        voxel_feats, voxel_coors, batch_size = input_tuple
        voxel_coors = voxel_coors.long()
        if self.fp16:
            voxel_feats = voxel_feats
        if self.training:
            batching_info = self.batching_info[0]
        else:
            batching_info = self.batching_info[1]
        device = voxel_feats.device
        if hasattr(self, 'linear0'):
            voxel_feats = self.linear0(voxel_feats)
        output = SRATensor(voxel_feats, voxel_coors, self.init_sparse_shape, batch_size)
        for i, block in enumerate(self.block_list):
            output = block(output, batching_info, using_checkpoint=i in self.checkpoint_blocks)
        output = self._window2bev_old(output.features, output.indices, batch_size)
        if self.num_attached_conv > 0:
            for conv in self.conv_layer:
                output = conv(output)
        output_list = []
        output_list.append(output)
        return output_list

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name:
                nn.init.xavier_uniform_(p)

    def _window2bev_old(self, voxel_feat, coors, batch_size):
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


class BasicShiftBlock(nn.Module):
    """ Consist of two encoder layer, shift and shift back.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', batch_first=False, block_id=-100):
        super().__init__()
        encoder_1 = EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first, layer_id=block_id * 2 + 0)
        encoder_2 = EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first, layer_id=block_id * 2 + 1)
        self.encoder_list = nn.ModuleList([encoder_1, encoder_2])

    def forward(self, src, pos_dict_list, ind_dict_list, voxel_drop_level_list, key_mask_dict_list, drop_info, using_checkpoint=False):
        num_shifts = len(pos_dict_list)
        assert num_shifts in (1, 2)
        output = src
        for i in range(2):
            this_id = i % num_shifts
            pos_dict = pos_dict_list[this_id]
            ind_dict = ind_dict_list[this_id]
            voxel_drop = voxel_drop_level_list[this_id]
            key_mask_dict = key_mask_dict_list[this_id]
            layer = self.encoder_list[i]
            if using_checkpoint:
                output = checkpoint(layer, output, pos_dict, ind_dict, voxel_drop, key_mask_dict, drop_info)
            else:
                output = layer(output, pos_dict, ind_dict, voxel_drop, key_mask_dict, drop_info)
        return output


class SSTv1(nn.Module):
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

    def __init__(self, d_model=[], nhead=[], num_blocks=6, dim_feedforward=[], dropout=0.0, activation='gelu', output_shape=None, num_attached_conv=2, conv_in_channel=64, conv_out_channel=64, norm_cfg=dict(type='naiveSyncBN2d', eps=0.001, momentum=0.01), conv_cfg=dict(type='Conv2d', bias=False), debug=True, drop_info=None, normalize_pos=False, pos_temperature=10000, window_shape=None, in_channel=None, conv_kwargs=dict(kernel_size=3, dilation=2, padding=2, stride=1), checkpoint_blocks=[]):
        super().__init__()
        assert drop_info is not None
        self.meta_drop_info = drop_info
        self.pos_temperature = pos_temperature
        self.d_model = d_model
        self.window_shape = window_shape
        self.normalize_pos = normalize_pos
        self.nhead = nhead
        self.checkpoint_blocks = checkpoint_blocks
        if in_channel is not None:
            self.linear0 = nn.Linear(in_channel, d_model[0])
        block_list = []
        for i in range(num_blocks):
            block_list.append(BasicShiftBlock(d_model[i], nhead[i], dim_feedforward[i], dropout, activation, batch_first=False, block_id=i))
        self.block_list = nn.ModuleList(block_list)
        self._reset_parameters()
        self.output_shape = output_shape
        self.debug = debug
        self.num_attached_conv = num_attached_conv
        if num_attached_conv > 0:
            conv_list = []
            for i in range(num_attached_conv):
                if isinstance(conv_kwargs, dict):
                    conv_kwargs_i = conv_kwargs
                elif isinstance(conv_kwargs, list):
                    assert len(conv_kwargs) == num_attached_conv
                    conv_kwargs_i = conv_kwargs[i]
                if i > 0:
                    conv_in_channel = conv_out_channel
                conv = build_conv_layer(conv_cfg, in_channels=conv_in_channel, out_channels=conv_out_channel, **conv_kwargs_i)
                if norm_cfg is None:
                    convnormrelu = nn.Sequential(conv, nn.ReLU(inplace=True))
                else:
                    convnormrelu = nn.Sequential(conv, build_norm_layer(norm_cfg, conv_out_channel)[1], nn.ReLU(inplace=True))
                conv_list.append(convnormrelu)
            self.conv_layer = nn.ModuleList(conv_list)

    def forward(self, input_tuple):
        """
        """
        voxel_feat, ind_dict_list, voxel_info = input_tuple
        assert voxel_info['coors'].dtype == torch.int64, 'data type of coors should be torch.int64!'
        self.set_drop_info()
        device = voxel_info['coors'].device
        batch_size = voxel_info['coors'][:, 0].max().item() + 1
        num_shifts = len(ind_dict_list)
        padding_mask_list = [self.get_key_padding_mask(ind_dict_list[i], voxel_info[f'voxel_drop_level_shift{i}'], device) for i in range(num_shifts)]
        pos_embed_list = [self.get_pos_embed(_t, voxel_info[f'coors_in_win_shift{i}'], voxel_info[f'voxel_drop_level_shift{i}'], voxel_feat.dtype, voxel_info.get(f'voxel_win_level_shift{i}', None)) for i, _t in enumerate(ind_dict_list)]
        voxel_drop_level_list = [voxel_info[f'voxel_drop_level_shift{i}'] for i in range(num_shifts)]
        output = voxel_feat
        if hasattr(self, 'linear0'):
            output = self.linear0(output)
        for i, block in enumerate(self.block_list):
            output = block(output, pos_embed_list, ind_dict_list, voxel_drop_level_list, padding_mask_list, self.drop_info, using_checkpoint=i in self.checkpoint_blocks)
        output = self.recover_bev(output, voxel_info['coors'], batch_size)
        if self.num_attached_conv > 0:
            for conv in self.conv_layer:
                output = conv(output)
        output_list = []
        output_list.append(output)
        return output_list

    def get_key_padding_mask(self, ind_dict, voxel_drop_lvl, device):
        num_all_voxel = len(voxel_drop_lvl)
        key_padding = torch.ones((num_all_voxel, 1)).bool()
        window_key_padding_dict = flat2window(key_padding, voxel_drop_lvl, ind_dict, self.drop_info)
        for key, value in window_key_padding_dict.items():
            window_key_padding_dict[key] = value.logical_not().squeeze(2)
        return window_key_padding_dict

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name:
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

    def get_pos_embed(self, ind_dict, coors_in_win, voxel_drop_level, dtype, voxel_window_level):
        """
        Args:
        """
        win_x, win_y = self.window_shape
        x, y = coors_in_win[:, 0] - win_x / 2, coors_in_win[:, 1] - win_y / 2
        assert (x >= -win_x / 2 - 0.0001).all()
        assert (x <= win_x / 2 - 1 + 0.0001).all()
        if self.normalize_pos:
            x = x / win_x * 2 * 3.1415
            y = y / win_y * 2 * 3.1415
        pos_length = self.d_model[0] // 2
        assert self.d_model[0] == self.d_model[1] == self.d_model[-1], 'If you want to use different d_model, Please implement corresponding pos embendding.'
        inv_freq = torch.arange(pos_length, dtype=torch.float32, device=coors_in_win.device)
        inv_freq = self.pos_temperature ** (2 * (inv_freq // 2) / pos_length)
        embed_x = x[:, None] / inv_freq[None, :]
        embed_y = y[:, None] / inv_freq[None, :]
        embed_x = torch.stack([embed_x[:, ::2].sin(), embed_x[:, 1::2].cos()], dim=-1).flatten(1)
        embed_y = torch.stack([embed_y[:, ::2].sin(), embed_y[:, 1::2].cos()], dim=-1).flatten(1)
        pos_embed_2d = torch.cat([embed_x, embed_y], dim=-1)
        window_pos_emb_dict = flat2window(pos_embed_2d, voxel_drop_level, ind_dict, self.drop_info)
        return window_pos_emb_dict

    def set_drop_info(self):
        if hasattr(self, 'drop_info'):
            return
        meta = self.meta_drop_info
        if isinstance(meta, tuple):
            if self.training:
                self.drop_info = meta[0]
            else:
                self.drop_info = meta[1]
        else:
            self.drop_info = meta


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

    def __init__(self, d_model=[], nhead=[], num_blocks=6, dim_feedforward=[], dropout=0.0, activation='gelu', output_shape=None, num_attached_conv=2, conv_in_channel=64, conv_out_channel=64, norm_cfg=dict(type='naiveSyncBN2d', eps=0.001, momentum=0.01), conv_cfg=dict(type='Conv2d', bias=False), debug=True, in_channel=None, to_bev=True, conv_kwargs=dict(kernel_size=3, dilation=2, padding=2, stride=1), checkpoint_blocks=[], layer_cfg=dict(), conv_shortcut=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.checkpoint_blocks = checkpoint_blocks
        self.conv_shortcut = conv_shortcut
        self.to_bev = to_bev
        if in_channel is not None:
            self.linear0 = nn.Linear(in_channel, d_model[0])
        block_list = []
        for i in range(num_blocks):
            block_list.append(BasicShiftBlockV2(d_model[i], nhead[i], dim_feedforward[i], dropout, activation, batch_first=False, block_id=i, layer_cfg=layer_cfg))
        self.block_list = nn.ModuleList(block_list)
        self._reset_parameters()
        self.output_shape = output_shape
        self.debug = debug
        self.num_attached_conv = num_attached_conv
        if num_attached_conv > 0:
            conv_list = []
            for i in range(num_attached_conv):
                if isinstance(conv_kwargs, dict):
                    conv_kwargs_i = conv_kwargs
                elif isinstance(conv_kwargs, list):
                    assert len(conv_kwargs) == num_attached_conv
                    conv_kwargs_i = conv_kwargs[i]
                if i > 0:
                    conv_in_channel = conv_out_channel
                conv = build_conv_layer(conv_cfg, in_channels=conv_in_channel, out_channels=conv_out_channel, **conv_kwargs_i)
                if norm_cfg is None:
                    convnormrelu = nn.Sequential(conv, nn.ReLU(inplace=True))
                else:
                    convnormrelu = nn.Sequential(conv, build_norm_layer(norm_cfg, conv_out_channel)[1], nn.ReLU(inplace=True))
                conv_list.append(convnormrelu)
            self.conv_layer = nn.ModuleList(conv_list)

    def forward(self, voxel_info):
        """
        """
        num_shifts = 2
        assert voxel_info['voxel_coors'].dtype == torch.int64, 'data type of coors should be torch.int64!'
        device = voxel_info['voxel_coors'].device
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
        if self.to_bev:
            output = self.recover_bev(output, voxel_info['voxel_coors'], batch_size)
        output_list = []
        if self.num_attached_conv > 0:
            assert self.to_bev
            for conv in self.conv_layer:
                temp = conv(output)
                if temp.shape == output.shape and self.conv_shortcut:
                    output = temp + output
                else:
                    output = temp
        if not self.to_bev:
            output = {'voxel_feats': output, 'voxel_coors': voxel_info['voxel_coors']}
        if len(output_list) == 0:
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


class PointsObjClsModule(nn.Module):
    """object candidate point prediction from seed point features.

    Args:
        in_channel (int): number of channels of seed point features.
        num_convs (int): number of conv layers.
            Default: 3.
        conv_cfg (dict): Config of convolution.
            Default: dict(type='Conv1d').
        norm_cfg (dict): Config of normalization.
            Default: dict(type='BN1d').
        act_cfg (dict): Config of activation.
            Default: dict(type='ReLU').
    """

    def __init__(self, in_channel, num_convs=3, conv_cfg=dict(type='Conv1d'), norm_cfg=dict(type='BN1d'), act_cfg=dict(type='ReLU')):
        super().__init__()
        conv_channels = [in_channel for _ in range(num_convs - 1)]
        conv_channels.append(1)
        self.mlp = nn.Sequential()
        prev_channels = in_channel
        for i in range(num_convs):
            self.mlp.add_module(f'layer{i}', ConvModule(prev_channels, conv_channels[i], 1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg if i < num_convs - 1 else None, act_cfg=act_cfg if i < num_convs - 1 else None, bias=True, inplace=True))
            prev_channels = conv_channels[i]

    def forward(self, seed_features):
        """Forward pass.

        Args:
            seed_features (torch.Tensor): seed features, dims:
                (batch_size, feature_dim, num_seed)

        Returns:
            torch.Tensor: objectness logits, dim:
                (batch_size, 1, num_seed)
        """
        return self.mlp(seed_features)


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


EPS = 1e-06


class FurthestPointSampling(Function):
    """Furthest Point Sampling.

    Uses iterative furthest point sampling to select a set of features whose
    corresponding points have the furthest distance.
    """

    @staticmethod
    def forward(ctx, points_xyz: 'torch.Tensor', num_points: 'int') ->torch.Tensor:
        """forward.

        Args:
            points_xyz (Tensor): (B, N, 3) where N > num_points.
            num_points (int): Number of points in the sampled set.

        Returns:
             Tensor: (B, num_points) indices of the sampled points.
        """
        assert points_xyz.is_contiguous()
        B, N = points_xyz.size()[:2]
        output = torch.IntTensor(B, num_points)
        temp = torch.FloatTensor(B, N).fill_(10000000000.0)
        furthest_point_sample_ext.furthest_point_sampling_wrapper(B, N, num_points, points_xyz, temp, output)
        ctx.mark_non_differentiable(output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class DFPS_Sampler(nn.Module):
    """DFPS_Sampling.

    Using Euclidean distances of points for FPS.
    """

    def __init__(self):
        super(DFPS_Sampler, self).__init__()

    def forward(self, points, features, npoint):
        """Sampling points with D-FPS."""
        fps_idx = furthest_point_sample(points.contiguous(), npoint)
        return fps_idx


def calc_square_dist(point_feat_a, point_feat_b, norm=True):
    """Calculating square distance between a and b.

    Args:
        point_feat_a (Tensor): (B, N, C) Feature vector of each point.
        point_feat_b (Tensor): (B, M, C) Feature vector of each point.
        norm (Bool): Whether to normalize the distance.
            Default: True.

    Returns:
        Tensor: (B, N, M) Distance between each pair points.
    """
    length_a = point_feat_a.shape[1]
    length_b = point_feat_b.shape[1]
    num_channel = point_feat_a.shape[-1]
    a_square = torch.sum(point_feat_a.unsqueeze(dim=2).pow(2), dim=-1)
    b_square = torch.sum(point_feat_b.unsqueeze(dim=1).pow(2), dim=-1)
    a_square = a_square.repeat((1, 1, length_b))
    b_square = b_square.repeat((1, length_a, 1))
    coor = torch.matmul(point_feat_a, point_feat_b.transpose(1, 2))
    dist = a_square + b_square - 2 * coor
    if norm:
        dist = torch.sqrt(dist) / num_channel
    return dist


class FurthestPointSamplingWithDist(Function):
    """Furthest Point Sampling With Distance.

    Uses iterative furthest point sampling to select a set of features whose
    corresponding points have the furthest distance.
    """

    @staticmethod
    def forward(ctx, points_dist: 'torch.Tensor', num_points: 'int') ->torch.Tensor:
        """forward.

        Args:
            points_dist (Tensor): (B, N, N) Distance between each point pair.
            num_points (int): Number of points in the sampled set.

        Returns:
             Tensor: (B, num_points) indices of the sampled points.
        """
        assert points_dist.is_contiguous()
        B, N, _ = points_dist.size()
        output = points_dist.new_zeros([B, num_points], dtype=torch.int32)
        temp = points_dist.new_zeros([B, N]).fill_(10000000000.0)
        furthest_point_sample_ext.furthest_point_sampling_with_dist_wrapper(B, N, num_points, points_dist, temp, output)
        ctx.mark_non_differentiable(output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample_with_dist = FurthestPointSamplingWithDist.apply


class FFPS_Sampler(nn.Module):
    """FFPS_Sampler.

    Using feature distances for FPS.
    """

    def __init__(self):
        super(FFPS_Sampler, self).__init__()

    def forward(self, points, features, npoint):
        """Sampling points with F-FPS."""
        assert features is not None, 'feature input to FFPS_Sampler should not be None'
        features_for_fps = torch.cat([points, features.transpose(1, 2)], dim=2)
        features_dist = calc_square_dist(features_for_fps, features_for_fps, norm=False)
        fps_idx = furthest_point_sample_with_dist(features_dist, npoint)
        return fps_idx


class FS_Sampler(nn.Module):
    """FS_Sampling.

    Using F-FPS and D-FPS simultaneously.
    """

    def __init__(self):
        super(FS_Sampler, self).__init__()

    def forward(self, points, features, npoint):
        """Sampling points with FS_Sampling."""
        assert features is not None, 'feature input to FS_Sampler should not be None'
        features_for_fps = torch.cat([points, features.transpose(1, 2)], dim=2)
        features_dist = calc_square_dist(features_for_fps, features_for_fps, norm=False)
        fps_idx_ffps = furthest_point_sample_with_dist(features_dist, npoint)
        fps_idx_dfps = furthest_point_sample(points, npoint)
        fps_idx = torch.cat([fps_idx_ffps, fps_idx_dfps], dim=1)
        return fps_idx


def get_sampler_type(sampler_type):
    """Get the type and mode of points sampler.

    Args:
        sampler_type (str): The type of points sampler.
            The valid value are "D-FPS", "F-FPS", or "FS".

    Returns:
        class: Points sampler type.
    """
    if sampler_type == 'D-FPS':
        sampler = DFPS_Sampler
    elif sampler_type == 'F-FPS':
        sampler = FFPS_Sampler
    elif sampler_type == 'FS':
        sampler = FS_Sampler
    else:
        raise ValueError(f'Only "sampler_type" of "D-FPS", "F-FPS", or "FS" are supported, got {sampler_type}')
    return sampler


def aligned_3d_nms(boxes, scores, classes, thresh):
    """3d nms for aligned boxes.

    Args:
        boxes (torch.Tensor): Aligned box with shape [n, 6].
        scores (torch.Tensor): Scores of each box.
        classes (torch.Tensor): Class of each box.
        thresh (float): Iou threshold for nms.

    Returns:
        torch.Tensor: Indices of selected boxes.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    z1 = boxes[:, 2]
    x2 = boxes[:, 3]
    y2 = boxes[:, 4]
    z2 = boxes[:, 5]
    area = (x2 - x1) * (y2 - y1) * (z2 - z1)
    zero = boxes.new_zeros(1)
    score_sorted = torch.argsort(scores)
    pick = []
    while score_sorted.shape[0] != 0:
        last = score_sorted.shape[0]
        i = score_sorted[-1]
        pick.append(i)
        xx1 = torch.max(x1[i], x1[score_sorted[:last - 1]])
        yy1 = torch.max(y1[i], y1[score_sorted[:last - 1]])
        zz1 = torch.max(z1[i], z1[score_sorted[:last - 1]])
        xx2 = torch.min(x2[i], x2[score_sorted[:last - 1]])
        yy2 = torch.min(y2[i], y2[score_sorted[:last - 1]])
        zz2 = torch.min(z2[i], z2[score_sorted[:last - 1]])
        classes1 = classes[i]
        classes2 = classes[score_sorted[:last - 1]]
        inter_l = torch.max(zero, xx2 - xx1)
        inter_w = torch.max(zero, yy2 - yy1)
        inter_h = torch.max(zero, zz2 - zz1)
        inter = inter_l * inter_w * inter_h
        iou = inter / (area[i] + area[score_sorted[:last - 1]] - inter)
        iou = iou * (classes1 == classes2).float()
        score_sorted = score_sorted[torch.nonzero(iou <= thresh, as_tuple=False).flatten()]
    indices = boxes.new_tensor(pick, dtype=torch.long)
    return indices


def build_loss(cfg):
    """Build loss function."""
    return LOSSES.build(cfg)


def filter_almost_empty(pts_coors, min_points=5):
    if min_points > 0:
        new_coors, unq_inv, unq_cnt = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)
        cnt_per_point = unq_cnt[unq_inv]
        valid_mask = cnt_per_point >= min_points
    else:
        valid_mask = torch.ones(len(pts_coors), device=pts_coors.device, dtype=torch.bool)
    return valid_mask


def find_connected_componets(points, batch_idx, dist):
    device = points.device
    bsz = batch_idx.max().item() + 1
    base = 0
    components_inds = torch.zeros_like(batch_idx) - 1
    for i in range(bsz):
        batch_mask = batch_idx == i
        if batch_mask.any():
            this_points = points[batch_mask]
            dist_mat = this_points[:, None, :2] - this_points[None, :, :2]
            dist_mat = (dist_mat ** 2).sum(2) ** 0.5
            adj_mat = dist_mat < dist
            adj_mat = adj_mat.cpu().numpy()
            c_inds = connected_components(adj_mat, directed=False)[1]
            c_inds = torch.from_numpy(c_inds).int() + base
            base = c_inds.max().item() + 1
            components_inds[batch_mask] = c_inds
    assert len(torch.unique(components_inds)) == components_inds.max().item() + 1
    return components_inds


def find_connected_componets_gpu(points, batch_idx, dist):
    assert len(points) > 0
    assert cc_gpu is not None
    components_inds = cc_gpu(points, batch_idx, dist, 100, 2, False)
    assert len(torch.unique(components_inds)) == components_inds.max().item() + 1
    return components_inds


def find_connected_componets_single_batch(points, batch_idx, dist):
    device = points.device
    this_points = points
    dist_mat = this_points[:, None, :2] - this_points[None, :, :2]
    dist_mat = (dist_mat ** 2).sum(2) ** 0.5
    adj_mat = dist_mat < dist
    adj_mat = adj_mat.cpu().numpy()
    c_inds = connected_components(adj_mat, directed=False)[1]
    c_inds = torch.from_numpy(c_inds).int()
    return c_inds


def modify_cluster_by_class(cluster_inds_list):
    new_list = []
    for i, inds in enumerate(cluster_inds_list):
        cls_pad = inds.new_ones((len(inds),)) * i
        inds = torch.cat([cls_pad[:, None], inds], 1)
        new_list.append(inds)
    return new_list


def scatter_v2(feat, coors, mode, return_inv=True, min_points=0, unq_inv=None, new_coors=None):
    assert feat.size(0) == coors.size(0)
    if mode == 'avg':
        mode = 'mean'
    if unq_inv is None and min_points > 0:
        new_coors, unq_inv, unq_cnt = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)
    elif unq_inv is None:
        new_coors, unq_inv = torch.unique(coors, return_inverse=True, return_counts=False, dim=0)
    else:
        assert new_coors is not None, 'please pass new_coors for interface consistency, caller: {}'.format(traceback.extract_stack()[-2][2])
    if min_points > 0:
        cnt_per_point = unq_cnt[unq_inv]
        valid_mask = cnt_per_point >= min_points
        feat = feat[valid_mask]
        coors = coors[valid_mask]
        new_coors, unq_inv, unq_cnt = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)
    if mode == 'max':
        new_feat, argmax = torch_scatter.scatter_max(feat, unq_inv, dim=0)
    elif mode in ('mean', 'sum'):
        new_feat = torch_scatter.scatter(feat, unq_inv, dim=0, reduce=mode)
    else:
        raise NotImplementedError
    if not return_inv:
        return new_feat, new_coors
    else:
        return new_feat, new_coors, unq_inv


class ClusterAssigner(torch.nn.Module):
    """ Generating cluster centers for each class and assign each point to cluster centers
    """

    def __init__(self, cluster_voxel_size, min_points, point_cloud_range, connected_dist, class_names=['Car', 'Cyclist', 'Pedestrian'], gpu_clustering=(False, False)):
        super().__init__()
        self.cluster_voxel_size = cluster_voxel_size
        self.min_points = min_points
        self.connected_dist = connected_dist
        self.point_cloud_range = point_cloud_range
        self.class_names = class_names
        self.gpu_clustering = gpu_clustering

    @torch.no_grad()
    def forward(self, points_list, batch_idx_list, gt_bboxes_3d=None, gt_labels_3d=None, origin_points=None):
        gt_bboxes_3d = None
        gt_labels_3d = None
        assert self.num_classes == len(self.class_names)
        cluster_inds_list, valid_mask_list = multi_apply(self.forward_single_class, points_list, batch_idx_list, self.class_names, origin_points)
        cluster_inds_list = modify_cluster_by_class(cluster_inds_list)
        return cluster_inds_list, valid_mask_list

    def forward_single_class(self, points, batch_idx, class_name, origin_points):
        batch_idx = batch_idx.int()
        if isinstance(self.cluster_voxel_size, dict):
            cluster_vsize = self.cluster_voxel_size[class_name]
        elif isinstance(self.cluster_voxel_size, list):
            cluster_vsize = self.cluster_voxel_size[self.class_names.index(class_name)]
        else:
            cluster_vsize = self.cluster_voxel_size
        voxel_size = torch.tensor(cluster_vsize, device=points.device)
        pc_range = torch.tensor(self.point_cloud_range, device=points.device)
        coors = torch.div(points - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').int()
        coors = torch.cat([batch_idx[:, None], coors], dim=1)
        valid_mask = filter_almost_empty(coors, min_points=self.min_points)
        if not valid_mask.any():
            valid_mask = ~valid_mask
        points = points[valid_mask]
        batch_idx = batch_idx[valid_mask]
        coors = coors[valid_mask]
        sampled_centers, voxel_coors, inv_inds = scatter_v2(points, coors, mode='avg', return_inv=True)
        if isinstance(self.connected_dist, dict):
            dist = self.connected_dist[class_name]
        elif isinstance(self.connected_dist, list):
            dist = self.connected_dist[self.class_names.index(class_name)]
        else:
            dist = self.connected_dist
        if self.training:
            cluster_inds = find_connected_componets(sampled_centers, voxel_coors[:, 0], dist)
        elif self.gpu_clustering[1]:
            cluster_inds = find_connected_componets_gpu(sampled_centers, voxel_coors[:, 0], dist)
        else:
            cluster_inds = find_connected_componets_single_batch(sampled_centers, voxel_coors[:, 0], dist)
        assert len(cluster_inds) == len(sampled_centers)
        cluster_inds_per_point = cluster_inds[inv_inds]
        cluster_inds_per_point = torch.stack([batch_idx, cluster_inds_per_point], 1)
        return cluster_inds_per_point, valid_mask


def fps(points, N):
    idx = furthest_point_sample(points.unsqueeze(0), N)
    idx = idx.squeeze(0).long()
    points = points[idx]
    return points


class SSGAssigner(torch.nn.Module):
    """ Generating cluster centers for each class and assign each point to cluster centers
    """

    def __init__(self, cluster_voxel_size, point_cloud_range, radius, num_fps, class_names=['Car', 'Cyclist', 'Pedestrian']):
        super().__init__()
        self.cluster_voxel_size = cluster_voxel_size
        self.radius = radius
        self.num_fps = num_fps
        self.point_cloud_range = point_cloud_range
        self.class_names = class_names

    @torch.no_grad()
    def forward(self, points_list, batch_idx_list, gt_bboxes_3d=None, gt_labels_3d=None, origin_points=None):
        gt_bboxes_3d = None
        gt_labels_3d = None
        assert self.num_classes == len(self.class_names)
        cluster_inds_list, valid_mask_list = multi_apply(self.forward_single_class, points_list, batch_idx_list, self.class_names, origin_points)
        cluster_inds_list = modify_cluster_by_class(cluster_inds_list)
        return cluster_inds_list, valid_mask_list

    def forward_single_class(self, points, batch_idx, class_name, origin_points):
        if isinstance(self.cluster_voxel_size, dict):
            cluster_vsize = self.cluster_voxel_size[class_name]
        elif isinstance(self.cluster_voxel_size, list):
            cluster_vsize = self.cluster_voxel_size[self.class_names.index(class_name)]
        else:
            cluster_vsize = self.cluster_voxel_size
        if isinstance(self.radius, dict):
            radius = self.radius[class_name]
        elif isinstance(self.radius, list):
            radius = self.radius[self.class_names.index(class_name)]
        else:
            radius = self.radius
        voxel_size = torch.tensor(cluster_vsize, device=points.device)
        pc_range = torch.tensor(self.point_cloud_range, device=points.device)
        coors = torch.div(points - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').long()
        coors = coors[:, [2, 1, 0]]
        coors = torch.cat([batch_idx[:, None], coors], dim=1)
        voxels, _, inv_inds = scatter_v2(points, coors, mode='avg', return_inv=True)
        num_fps = self.num_fps[class_name]
        if num_fps >= len(voxels):
            key_points = voxels
        else:
            key_points = fps(voxels, self.num_fps[class_name])
        k_dist_mat = key_points[:, None, :2] - key_points[None, :, :2]
        k_dist_mat = (k_dist_mat ** 2).sum(2) ** 0.5
        dist_mask = k_dist_mat < radius * 2 + 0.01
        triangle1 = torch.arange(len(key_points))[None, :].expand(len(key_points), -1)
        triangle2 = triangle1.T
        triangle_mask = triangle1 <= triangle2
        dist_mask[triangle_mask] = False
        invalid_keypoints_mask = dist_mask.any(0)
        key_points = key_points[~invalid_keypoints_mask]
        dist_mat = key_points[:, None, :2] - voxels[None, :, :2]
        dist_mat = (dist_mat ** 2).sum(2) ** 0.5
        in_radius_mask = dist_mat < radius
        assert (in_radius_mask.sum(0) <= 1).all()
        valid_centers_mask = in_radius_mask.sum(0) == 1
        assert valid_centers_mask.any()
        pos = torch.nonzero(in_radius_mask)
        cluster_inds = pos[:, 0]
        col_inds = pos[:, 1]
        sorted_col_inds, order = torch.sort(col_inds)
        cluster_inds = cluster_inds[order]
        assert (sorted_col_inds == torch.nonzero(valid_centers_mask).reshape(-1)).all()
        cluster_inds_full = cluster_inds.new_zeros(len(voxels)) - 1
        cluster_inds_full[valid_centers_mask] = cluster_inds
        cluster_inds_per_point = cluster_inds_full[inv_inds]
        valid_pts_mask = cluster_inds_per_point > -1
        cluster_inds_per_point = torch.stack([batch_idx, cluster_inds_per_point], 1)
        cluster_inds_per_point = cluster_inds_per_point[valid_pts_mask]
        return cluster_inds_per_point, valid_pts_mask


def ssg_single_sample(points, num_fps, radius):
    """
    a little complicated
    """
    if num_fps >= len(points):
        key_points = points
    else:
        key_points = fps(points, num_fps)
    k_dist_mat = key_points[:, None, :2] - key_points[None, :, :2]
    k_dist_mat = (k_dist_mat ** 2).sum(2) ** 0.5
    dist_mask = k_dist_mat < radius * 2 + 0.01
    triangle1 = torch.arange(len(key_points))[None, :].expand(len(key_points), -1)
    triangle2 = triangle1.T
    triangle_mask = triangle1 <= triangle2
    dist_mask[triangle_mask] = False
    invalid_keypoints_mask = dist_mask.any(0)
    key_points = key_points[~invalid_keypoints_mask]
    dist_mat = key_points[:, None, :2] - points[None, :, :2]
    dist_mat = (dist_mat ** 2).sum(2) ** 0.5
    in_radius_mask = dist_mat < radius
    assert (in_radius_mask.sum(0) <= 1).all()
    valid_centers_mask = in_radius_mask.sum(0) == 1
    assert valid_centers_mask.any()
    pos = torch.nonzero(in_radius_mask)
    cluster_inds = pos[:, 0]
    col_inds = pos[:, 1]
    sorted_col_inds, order = torch.sort(col_inds)
    cluster_inds = cluster_inds[order]
    assert (sorted_col_inds == torch.nonzero(valid_centers_mask).reshape(-1)).all()
    cluster_inds_full = cluster_inds.new_zeros(len(points)) - 1
    cluster_inds_full[valid_centers_mask] = cluster_inds
    return cluster_inds_full


def ssg(points, batch_idx, num_fps, radius):
    device = points.device
    bsz = batch_idx.max().item() + 1
    base = 0
    components_inds = torch.zeros_like(batch_idx) - 2
    for i in range(bsz):
        batch_mask = batch_idx == i
        if batch_mask.any():
            this_points = points[batch_mask]
            this_inds = ssg_single_sample(this_points, num_fps, radius)
            this_inds[this_inds > -1] += base
            base = this_inds.max().item() + 1
            components_inds[batch_mask] = this_inds
    assert (components_inds > -2).all()
    return components_inds


class HybridAssigner(torch.nn.Module):
    """ Generating cluster centers for each class and assign each point to cluster centers
    """

    def __init__(self, point_cloud_range, cfg_per_class, class_names=['Car', 'Cyclist', 'Pedestrian']):
        super().__init__()
        self.point_cloud_range = point_cloud_range
        self.class_names = class_names
        self.cfg_per_class = cfg_per_class

    @torch.no_grad()
    def forward(self, points_list, batch_idx_list, gt_bboxes_3d=None, gt_labels_3d=None, origin_points=None):
        gt_bboxes_3d = None
        gt_labels_3d = None
        assert self.num_classes == len(self.class_names)
        cluster_inds_list, valid_mask_list = multi_apply(self.forward_single_class, points_list, batch_idx_list, self.class_names, origin_points)
        cluster_inds_list = modify_cluster_by_class(cluster_inds_list)
        return cluster_inds_list, valid_mask_list

    def forward_single_class(self, points, batch_idx, class_name, origin_points):
        """
        Dispatcher
        """
        assigner_type = self.cfg_per_class[class_name]['assigner_type']
        if assigner_type == 'ssg':
            return self.forward_ssg(points, batch_idx, class_name, origin_points)
        elif assigner_type == 'ccl':
            return self.forward_ccl(points, batch_idx, class_name, origin_points)

    def forward_ssg(self, points, batch_idx, class_name, origin_points):
        cluster_vsize = self.cfg_per_class[class_name]['cluster_voxel_size']
        radius = self.cfg_per_class[class_name]['radius']
        num_fps = self.cfg_per_class[class_name]['num_fps']
        voxel_size = torch.tensor(cluster_vsize, device=points.device)
        pc_range = torch.tensor(self.point_cloud_range, device=points.device)
        coors = torch.div(points - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').long()
        coors = coors[:, [2, 1, 0]]
        coors = torch.cat([batch_idx[:, None], coors], dim=1)
        voxels, voxel_coors, inv_inds = scatter_v2(points, coors, mode='avg', return_inv=True)
        cluster_inds_full = ssg(voxels, voxel_coors[:, 0], num_fps, radius)
        cluster_inds_per_point = cluster_inds_full[inv_inds]
        valid_pts_mask = cluster_inds_per_point > -1
        cluster_inds_per_point = torch.stack([batch_idx, cluster_inds_per_point], 1)
        cluster_inds_per_point = cluster_inds_per_point[valid_pts_mask]
        return cluster_inds_per_point, valid_pts_mask

    def forward_ccl(self, points, batch_idx, class_name, origin_points):
        cluster_vsize = self.cfg_per_class[class_name]['cluster_voxel_size']
        min_points = self.cfg_per_class[class_name]['min_points']
        dist = self.cfg_per_class[class_name]['connected_dist']
        voxel_size = torch.tensor(cluster_vsize, device=points.device)
        pc_range = torch.tensor(self.point_cloud_range, device=points.device)
        coors = torch.div(points - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').long()
        coors = coors[:, [2, 1, 0]]
        coors = torch.cat([batch_idx[:, None], coors], dim=1)
        valid_mask = filter_almost_empty(coors, min_points=min_points)
        if not valid_mask.any():
            valid_mask = ~valid_mask
        points = points[valid_mask]
        batch_idx = batch_idx[valid_mask]
        coors = coors[valid_mask]
        sampled_centers, voxel_coors, inv_inds = scatter_v2(points, coors, mode='avg', return_inv=True)
        cluster_inds = find_connected_componets(sampled_centers, voxel_coors[:, 0], dist)
        assert len(cluster_inds) == len(sampled_centers)
        cluster_inds_per_point = cluster_inds[inv_inds]
        cluster_inds_per_point = torch.stack([batch_idx, cluster_inds_per_point], 1)
        return cluster_inds_per_point, valid_mask


class TimestampEncoder(torch.nn.Module):
    """ Generating cluster centers for each class and assign each point to cluster centers
    """

    def __init__(self, strategy_cfg):
        super().__init__()
        self.strategy_cfg = strategy_cfg

    def forward(self, point, pts_frame_inds):
        assert (point[:, -1] < 200).all()
        assert (pts_frame_inds >= 0).all()
        assert (pts_frame_inds <= 200).all(), 'Only holds on WOD'
        stra = self.strategy_cfg['strategy']
        return getattr(self, stra)(point, pts_frame_inds)

    def scalar(self, point, pts_frame_inds):
        n = self.strategy_cfg['normalizer']
        ts_embed = point[:, -1:] / n
        out = torch.cat([point, ts_embed], 1)
        return out


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

    def __init__(self, tensor, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0), moved=False):
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
        self.moved = moved

    @property
    def volume(self):
        """torch.Tensor: A vector with volume of each box."""
        return self.tensor[:, 3] * self.tensor[:, 4] * self.tensor[:, 5]

    @property
    def area(self):
        """torch.Tensor: A vector with volume of each box."""
        return self.tensor[:, 3] * self.tensor[:, 4]

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
            dst (:obj:`BoxMode`): The target Box mode.
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
        if self.tensor.size(1) > 7:
            self.tensor[:, [7, 8]] *= scale_factor

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
            :obj:`BaseInstances3DBoxes`: A new object of                  :class:`BaseInstances3DBoxes` after indexing.
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
            boxes_list (list[:obj:`BaseInstances3DBoxes`]): List of boxes.

        Returns:
            :obj:`BaseInstances3DBoxes`: The concatenated Boxes.
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
            boxes1 (:obj:`BaseInstanceBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`BaseInstanceBoxes`): Boxes 2 contain M boxes.
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
            boxes1 (:obj:`BaseInstanceBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`BaseInstanceBoxes`): Boxes 2 contain M boxes.
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

    def new_box(self, data, moved=False):
        """Create a new box object with data.

        The new box and its tensor has the similar properties             as self and self.tensor, respectively.

        Args:
            data (torch.Tensor | numpy.array | list): Data to be copied.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new bbox object with ``data``,                 the object's other properties are similar to ``self``.
        """
        new_tensor = self.tensor.new_tensor(data) if not isinstance(data, torch.Tensor) else data
        original_type = type(self)
        return original_type(new_tensor, box_dim=self.box_dim, with_yaw=self.with_yaw, moved=moved)

    def to_ndarray(self):
        self.original_device = self.tensor.device
        self.tensor = self.tensor.cpu().numpy()

    def to_tensor(self):
        self.tensor = torch.from_numpy(self.tensor)

