
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


import itertools


import time


from typing import Any


from typing import Dict


from typing import List


import torch


from typing import Tuple


from typing import Union


from functools import partial


from typing import Optional


from warnings import warn


from torch import nn


from torch.distributed import destroy_process_group


from torch.distributed import init_process_group


from torch.optim import Optimizer


from torch.utils.data import DataLoader


from torch.utils.data import DistributedSampler


import math


from itertools import chain


import uuid


import torch.distributed.launcher as pet


from torch.utils.data import Dataset


import re


from typing import Generator


from typing import Mapping


from typing import TextIO


import torch.nn.functional as F


from torch.utils.data import Dataset as TorchDataset


import numpy as np


from torch import Tensor


from copy import deepcopy


from torch import tensor


import torch.optim as optim


import torch._dynamo.testing


from torch.nn.functional import normalize


import torchvision


from torch import randn


import copy


import torch.nn as nn


from torch.distributed import launcher


from torch.distributed._composable.fsdp import fully_shard


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


from torch.testing._internal.common_fsdp import FSDPTest


from torch.testing._internal.common_fsdp import MLP


import logging


from torch._C._profiler import _ExperimentalConfig


from torch.distributed.elastic.multiprocessing.errors import record


from torch.distributed.run import get_args_parser as get_torchrun_args_parser


from torch.distributed.run import run


from typing import Callable


from torch.nn.utils.rnn import pad_sequence


from torch.nn import functional as F


from torch.distributed._tensor import distribute_tensor


from torch.distributed._tensor import DTensor


from torchvision.transforms.v2 import functional as F


from enum import Enum


from collections import OrderedDict


from torch._subclasses.fake_tensor import FakeTensorConverter


from torch._subclasses.fake_tensor import FakeTensorMode


from torch.optim.lr_scheduler import LambdaLR


import functools


from typing import Literal


from typing import Protocol


from typing import Set


from collections import defaultdict


from typing import NamedTuple


from torch.autograd.graph import saved_tensors_hooks


from typing import cast


from typing import Type


import torch.distributed as dist


from torch.distributed._composable.fsdp import CPUOffloadPolicy


from torch.distributed._tensor.placement_types import DTensorSpec


from torch.distributed._tensor.placement_types import TensorMeta


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import _CHECKPOINT_WRAPPED_MODULE


from torch.distributed.checkpoint.state_dict import _init_optim_state


from torch.distributed.fsdp import ShardingStrategy


from torch.distributed.fsdp.wrap import ModuleWrapPolicy


import torch.distributed


from torch.profiler import tensorboard_trace_handler


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl


from torch.utils.checkpoint import checkpoint


import string


from typing import Iterable


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing


from torch.optim.lr_scheduler import LRScheduler


from numpy import ndarray


import random


import warnings


from functools import lru_cache


from functools import wraps


from typing import TypeVar


from torch import distributed as dist


def _reshape_for_broadcast(freqs_cis: 'torch.Tensor', x: 'torch.Tensor') ->torch.Tensor:
    """
    freqs_cis: complex - (seq_len, head_dim / 2)
    x: complex - (bsz, seq_len, head_dim / 2)
    """
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (freqs_cis.shape, (x.shape[1], x.shape[-1]))
    shape = [(d if i == 1 or i == ndim - 1 else 1) for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: 'torch.Tensor', xk: 'torch.Tensor', freqs_cis: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = _reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(keys: 'torch.Tensor', values: 'torch.Tensor', repeats: 'int'):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=2)
    values = torch.repeat_interleave(values, repeats=repeats, dim=2)
    return keys, values


class Attention(nn.Module):

    def __init__(self, n_heads: 'int', head_dim: 'int', dim: 'int', n_kv_heads: 'int'):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.repeats = self.n_heads // self.n_kv_heads
        self.scale = head_dim ** -0.5
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(self, x: 'torch.Tensor', freqs_cis: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        key, value = repeat_kv(xk, xv, self.repeats)
        query = xq.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(2, 3)) * self.scale
        None
        if mask is not None:
            scores += mask[None, None, ...]
        None
        scores = scores.float()
        scores = nn.functional.softmax(scores, dim=-1).type_as(query)
        output = torch.matmul(scores, value)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class RMSNorm(nn.Module):
    """
    Implements Root Mean Square Normalization introduced in
    https://arxiv.org/abs/1910.07467.

    Reference implementation (used for correctness verification)
    can be found here:
    https://github.com/facebookresearch/llama/blob/main/llama/model.py

    Args:
        dim (int): embedding size
        eps (float): small value to avoid division by zero. Default: 1e-6
    """

    def __init__(self, dim: 'int', eps: 'float'=1e-06) ->None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor to normalize

        Returns:
            torch.Tensor: The normalized and scaled tensor having the same shape as ``x``.
        """
        x_fp32 = x.float()
        x_normed = (x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)
        return x_normed * self.scale


class FeedForward(nn.Module):
    """This class implements the feed-forward network derived from Llama2.

    Args:
        gate_proj (nn.Module): Projection from input dim to hidden dim, fed through activation
            and multiplied by up_proj.
        down_proj (nn.Module): Final projection to output dim.
        up_proj (Optional[nn.Module]): Projection from input dim to hidden dim, multiplied by
            activation(gate_proj).
        activation (nn.Module): Activation function to use. Default is nn.SiLU().
    """

    def __init__(self, *, gate_proj: nn.Module, down_proj: nn.Module, up_proj: Optional[nn.Module]=None, activation: nn.Module=nn.SiLU()):
        super().__init__()
        self.w1 = gate_proj
        self.w2 = down_proj
        self.w3 = up_proj
        self.activation = activation

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape ``(..., in_dim)``, where ``in_dim`` is the
                input dimension of both ``gate_proj`` and ``up_proj``.

        Returns:
            torch.Tensor: output tensor with shape ``(..., out_dim)``, where ``out_dim`` is the                 output dimension of ``down_proj``.
        """
        h = self.activation(self.w1(x))
        if self.w3 is not None:
            h = h * self.w3(x)
        h = self.w2(h)
        return h


class TransformerBlock(nn.Module):

    def __init__(self, n_heads: 'int', head_dim: 'int', dim: 'int', n_kv_heads: 'int', hidden_dim: 'int', norm_eps: 'float'):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.attention = Attention(n_heads=n_heads, head_dim=head_dim, dim=dim, n_kv_heads=n_kv_heads)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=hidden_dim)
        self.attention_norm = RMSNorm(dim=dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(self, x: 'torch.Tensor', freqs_cis: 'torch.Tensor', mask: 'Optional[torch.Tensor]') ->torch.Tensor:
        r = self.attention.forward(self.attention_norm(x), freqs_cis, mask)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


def precompute_freqs_cis(dim: 'int', end: 'int', theta: 'float'=10000.0) ->torch.Tensor:
    freqs = 1.0 / theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


class Transformer(nn.Module):

    def __init__(self, vocab_size: 'int', n_layers: 'int', n_heads: 'int', head_dim: 'int', dim: 'int', n_kv_heads: 'int', hidden_dim: 'int', max_seq_len: 'int', rope_base: 'int', norm_eps: 'float'):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        assert self.vocab_size > 0
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = torch.nn.ModuleList([TransformerBlock(n_heads=n_heads, head_dim=head_dim, dim=dim, n_kv_heads=n_kv_heads, hidden_dim=hidden_dim, norm_eps=norm_eps) for _ in range(n_layers)])
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(head_dim, max_seq_len * 2, theta=rope_base)

    def forward(self, input_ids: 'torch.Tensor', positions: 'torch.Tensor'):
        _, seqlen = input_ids.shape
        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[positions]
        mask: 'Optional[torch.Tensor]' = None
        if input_ids.shape[1] > 1:
            seqlen = input_ids.shape[1]
            tensor = torch.full((seqlen, seqlen), dtype=h.dtype, fill_value=1, device=h.device)
            mask = torch.tril(tensor, diagonal=0)
            mask = torch.triu(mask, diagonal=-1)
            mask = torch.log(mask)
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        return self.output(self.norm(h)).float()


class RMSNormRef(torch.nn.Module):

    def __init__(self, dim: 'int', eps: 'float'=1e-05):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class _Wrapper(nn.Module):
    """
    For testing the merged checkpoint which requires that the LoRA layer has a parent.
    """

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x)


class _DoraReference(nn.Module):
    """
    DoRA linear layer reference.

    Paper: https://arxiv.org/abs/2402.09353

    Based on the code from:
    https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py
    https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/dora.py

    For more info, see the discussion here:
    https://github.com/huggingface/peft/pull/1474
    """

    def __init__(self, dtype: 'torch.dtype', in_dim: 'int', out_dim: 'int', rank: 'int', alpha: 'float', dropout: 'float'=0.0, use_bias: 'bool'=False, quantize_base: 'bool'=False, use_dora: 'bool'=True):
        super().__init__()
        self.use_bias = use_bias
        self.quantize_base = quantize_base
        self.use_dora = use_dora
        linear = nn.Linear(in_features=in_dim, out_features=out_dim, bias=use_bias, dtype=dtype)
        weight = linear.weight if not quantize_base else to_nf4(linear.weight)
        bias = None
        if use_bias:
            if quantize_base:
                raise NotImplementedError()
            bias = linear.bias
        self.register_parameter('weight', nn.Parameter(weight))
        self.register_parameter('bias', nn.Parameter(bias) if bias is not None else None)
        self.lora_a = nn.Linear(in_dim, rank, bias=False, dtype=dtype)
        self.lora_b = nn.Linear(rank, out_dim, bias=False, dtype=dtype)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.scaling = alpha / rank
        if use_dora:
            self.lora_magnitude = nn.Parameter(torch.randn(out_dim, dtype=dtype))
        self.dropout = nn.Dropout(p=dropout)

    def initialize_dora(self):
        weight = self.weight
        lora_weight = self.lora_b.weight @ self.lora_a.weight
        weight_norm = self._get_weight_norm(weight, lora_weight)
        self.lora_magnitude = nn.Parameter(weight_norm, requires_grad=True)

    def forward(self, x):
        result = self._base_forward(x)
        torch_result_dtype = result.dtype
        x = x
        if not self.use_dora:
            result = result + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling
        else:
            x = self.dropout(x)
            result = result + self._dora_forward(x)
        result = result
        None
        return result

    def _base_forward(self, x):
        if self.quantize_base:
            return linear_nf4(input=x, weight=self.weight)
        return F.linear(x, self.weight, self.bias)

    def _dora_forward(self, x):
        lora_result = self.lora_b(self.lora_a(x))
        x_eye = torch.eye(self.lora_a.weight.shape[1], device=self.lora_a.weight.device, dtype=x.dtype)
        lora_weight = self.lora_b(self.lora_a(x_eye)).T
        magnitude = self.lora_magnitude
        weight = self.weight
        weight_norm = self._get_weight_norm(weight, lora_weight.detach())
        weight_norm = weight_norm.detach()
        mag_norm_scale = (magnitude / weight_norm).view(1, -1)
        result_dora = (mag_norm_scale - 1) * F.linear(x, weight) + mag_norm_scale * lora_result * self.scaling
        return result_dora

    def _get_weight_norm(self, weight, lora_weight):
        weight = weight + self.scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1)
        return weight_norm


class FeedForwardRef(nn.Module):

    def __init__(self, dim: 'int', hidden_dim: 'int', multiple_of: 'int'=256, ffn_dim_multiplier: 'Optional[float]'=None):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class FusedMultiHeadAttention(nn.Module):
    """Multi-headed grouped query self-attention (GQA) layer introduced
    in https://arxiv.org/pdf/2305.13245v1.pdf.

    GQA is a version of multiheaded attention (MHA) which uses fewer
    key/value heads than query heads by grouping n query heads for each
    key and value head. Multi-Query Attention is an extreme
    version where we have a single key and value head shared by all
    query heads.

    Following is an example of MHA, GQA and MQA with num_heads = 4

    (credit for the documentation:
    https://github.com/Lightning-AI/lit-gpt/blob/main/lit_gpt/config.py).


    ::

        ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
        │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
        └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        │    │    │    │         │        │                 │
        ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
        │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
        └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
        ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
        │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
        └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
        ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
                MHA                    GQA                   MQA
        n_kv_heads =4          n_kv_heads=2           n_kv_heads=1

    Args:
        embed_dim (int): embedding dimension for the model
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. User should ensure
            `num_heads` % `num_kv_heads` == 0. For standard MHA set `num_kv_heads` == `num_heads`,
            for GQA `num_kv_heads` < `num_heads`, and for MQA set `num_kv_heads` == 1.
        head_dim (int): dimension of each head, calculated by ``embed_dim`` // ``num_heads``.
        qkv_proj (nn.Module): projection layer for query, key and value.
        output_proj (nn.Module): projection layer for output.
        pos_embeddings (nn.Module): positional embeddings layer, e.g. RotaryPositionalEmbeddings.
        kv_cache (Optional[KVCache]): KVCache object used to cache key and value.
            If not specified, then no caching is used.
        max_seq_len (int): maximum sequence length supported by the model.
            This is needed to compute the RoPE Cache. Default: 4096.
        attn_dropout (float): dropout value passed onto the
            scaled_dot_product_attention function. This argument is ignored if the
            self.training is False. Default value is 0.0.

    Raises:
        ValueError: If `num_heads` % `num_kv_heads` != 0
        ValueError: If `embed_dim` % `num_heads` != 0
        ValueError: If `attn_dropout` < 0 or > 1
    """

    def __init__(self, embed_dim: 'int', num_heads: 'int', num_kv_heads: 'int', head_dim: 'int', qkv_proj: 'nn.Module', output_proj: 'nn.Module', pos_embeddings: 'nn.Module', kv_cache: 'Optional[KVCache]'=None, max_seq_len: 'int'=4096, attn_dropout: 'float'=0.0) ->None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(f'num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})')
        if embed_dim % num_heads != 0:
            raise ValueError(f'embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})')
        if attn_dropout < 0 or attn_dropout > 1:
            raise ValueError(f'attn_dropout ({embed_dim}) must be between 0.0 and 1.0')
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.kv_cache = kv_cache
        self.qkv_proj = qkv_proj
        self.output_proj = output_proj
        self.pos_embeddings = pos_embeddings

    def forward(self, x: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None, curr_pos: 'int'=0) ->torch.Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            mask (Optional[torch.Tensor]): boolean mask, defaults to None.
            curr_pos (int): current position in the sequence, defaults to 0.

        Returns:
            Tensor: output tensor with attention applied

        Raises:
            ValueError: if seq_len of x is bigger than max_seq_len

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - n_kv: num kv heads
            - d: embed dim
            - h_d: head dim
            - qkv_d: qkv_dim computed as (n_h + 2 * n_kv) * h_d

        TODO:
            - Return the attention weights
            - Make application of positional embeddings optional
        """
        bsz, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f'seq_len ({seq_len}) of input tensor should be smaller than max_seq_len ({self.max_seq_len})')
        qkv = self.qkv_proj(x)
        q_per_kv = self.num_heads // self.num_kv_heads
        total_qkv = q_per_kv + 2
        qkv = qkv.view(bsz, seq_len, self.num_kv_heads, total_qkv, self.head_dim)
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=3)
        if self.num_heads != self.num_kv_heads:
            k = k.expand(bsz, seq_len, self.num_kv_heads, q_per_kv, self.head_dim)
            v = v.expand(bsz, seq_len, self.num_kv_heads, q_per_kv, self.head_dim)
        q = q.reshape(bsz, seq_len, -1, self.head_dim)
        k = k.reshape(bsz, seq_len, -1, self.head_dim)
        v = v.reshape(bsz, seq_len, -1, self.head_dim)
        q = self.pos_embeddings(q, curr_pos)
        k = self.pos_embeddings(k, curr_pos)
        if self.kv_cache is not None:
            k, v = self.kv_cache.update(bsz=bsz, seq_len=seq_len, curr_pos=curr_pos, k_val=k, v_val=v)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        output = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_dropout, is_causal=self.kv_cache is None)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.output_proj(output)


class LoRALayer:

    def __init__(self, r: 'int', lora_alpha: 'int', lora_dropout: 'float', merge_weights: 'bool'):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights


class LoRALinearRef(nn.Linear, LoRALayer):

    def __init__(self, in_features: 'int', out_features: 'int', r: 'int'=0, lora_alpha: 'int'=1, lora_dropout: 'float'=0.0, fan_in_fan_out: 'bool'=False, merge_weights: 'bool'=True, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: 'bool'=True):

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        elif self.merge_weights and not self.merged:
            if self.r > 0:
                self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: 'torch.Tensor'):

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class DummyCrossAttentionLayer(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.cache_enabled = False
        self.encoder_max_seq_len = None

    def setup_caches(self, batch_size, dtype, encoder_max_seq_len, decoder_max_seq_len):
        self.cache_enabled = True
        self.encoder_max_seq_len = encoder_max_seq_len

    def caches_are_enabled(self):
        return self.cache_enabled

    def reset_cache(self):
        self.cache_enabled = False

    def forward(self, x):
        return self.linear(x)


class DummySelfAttentionLayer(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.cache_enabled = False
        self.decoder_max_seq_len = None

    def setup_caches(self, batch_size, dtype, encoder_max_seq_len, decoder_max_seq_len):
        self.cache_enabled = True
        self.decoder_max_seq_len = decoder_max_seq_len

    def caches_are_enabled(self):
        return self.cache_enabled

    def reset_cache(self):
        self.cache_enabled = False

    def forward(self, x):
        return self.linear(x)


def register_fusion_module(module: 'nn.Module'):
    """Add the method fusion_params to an nn.Module that
    marks all of the Modules parameters as fusion params.
    This can be used for a layer or an entire model that is
    added to combine two or more pretrained models.

    For example, you might want to add a projection head
    head onto an encoder to learn a projection from the
    pre-trained encodings to the decoder's embedding space. This
    is typical with both Deep Fusion and Early Fusion models.

    Example:
        >>> projection_head = FeedForward(...)
        >>> register_fusion_module(projection_head))
        >>> encoder = nn.Sequential(clip_vit_224(), projection_head)

    Args:
        module (nn.Module): module to add the fusion_params method to
    """

    def fusion_params(self) ->List[str]:
        """
        Return parameters of fusion layer.
        """
        return [k for k, v in self.named_parameters()]
    module.fusion_params = functools.partial(fusion_params, module)


class DummyModel(nn.Module):

    def __init__(self, dim, vocab_size):
        super().__init__()
        self.cache_enabled = False
        self.embed = nn.Embedding(vocab_size, dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, vocab_size)
        register_fusion_module(self.output)

    def setup_caches(self, batch_size, dtype, *args, **kwargs):
        self.cache_enabled = True

    def caches_are_setup(self):
        return self.cache_enabled

    def reset_caches(self):
        self.cache_enabled = False

    def forward(self, tokens, mask, encoder_input, encoder_mask, input_pos):
        x = self.embed(tokens)
        if encoder_input is not None:
            q = self.q(x)
            k = self.k(encoder_input)
            v = self.v(encoder_input)
            x += nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=encoder_mask)
        x = self.output(x)
        return x


class AdapterModule(Protocol):
    """
    Interface for an ``nn.Module`` containing adapter weights.
    Note that an adapter module does not have to explicitly implement this protocol,
    but it must define the ``adapter_params(self)`` method.
    """

    def adapter_params(self) ->List[str]:
        """
        Return a list of strings corresponding to the names of the ``nn.Parameter`` s in
        the model coming from the adapter.
        E.g. if an nn.Module has adapter ``self.proj = nn.Linear(in_dim, out_dim)``,
        then adapter_params should return ``['proj.weight', 'proj.bias']``.

        See LoRALinear's :func:`~torchtune.modules.peft.LoRALinear.adapter_params` for an example.
        """
        pass


class DummyAdapterModule(nn.Module, AdapterModule):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.adapter = nn.Linear(in_dim, out_dim, bias=False)
        self.linear = nn.Linear(in_dim, out_dim)

    def adapter_params(self):
        return ['adapter.weight']

    def forward(self, x):
        return self.adapter(x) + self.non_adapter(x)


class DummyAdapterParentModel(nn.Module, AdapterModule):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dummy_adapter_module = DummyAdapterModule(in_dim, out_dim)
        self.parent_adapter = nn.Linear(in_dim, out_dim)
        self.parent_base_model = nn.Linear(in_dim, out_dim)

    def adapter_params(self):
        return ['parent_adapter.weight', 'parent_adapter.bias']

    def forward(self, x):
        return self.dummy_adapter_module(x) + self.parent_adapter(x) + self.parent_base_model(x)


class TokenPositionalEmbedding(nn.Module):
    """
    Token positional embedding for images, different for every token in an image.

    Notice that tile is different from patch (token). For details, please check the documentation of
    :class:`torchtune.modules.vision_transformer.VisionTransformer`.

    Args:
        embed_dim (int): The dimensionality of each token embedding.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each.
    """

    def __init__(self, embed_dim: 'int', tile_size: 'int', patch_size: 'int') ->None:
        super().__init__()
        patch_grid_size = tile_size // patch_size
        n_tokens_per_tile = patch_grid_size ** 2 + 1
        scale = embed_dim ** -0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn((n_tokens_per_tile, embed_dim)))

    def forward(self, x: 'torch.Tensor', *args: Tuple[Any]) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): torch.Tensor with shape (..., n_tokens_per_tile, embed_dim)
            *args (Tuple[Any]): Optional args.

        Returns:
            torch.Tensor: The input tensor with added positional embeddings.
        """
        return x + self.positional_embedding


class TiledTokenPositionalEmbedding(nn.Module):
    """

    Token positional embedding for tiled images, different for every tile, different for every token.

    There are two positional embeddings in this module:

    * local_token_positional_embedding: same for every tile, different for every token. Equivalent         to :class:`torchtune.models.clip._position_embeddings.TokenPositionalEmbedding`, but gated.
    * global_token_positional_embedding: different for every tile, different for every token.

    Notice that tile is different from patch (token). For details, please check the documentation of
    :class:`torchtune.modules.vision_transformer.VisionTransformer`.

    Args:
        max_num_tiles (int): The maximum number of tiles an image can be divided into.
        embed_dim (int): The dimensionality of each token embedding.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each.
    """

    def __init__(self, max_num_tiles: 'int', embed_dim: 'int', tile_size: 'int', patch_size: 'int') ->None:
        super().__init__()
        patch_grid_size = tile_size // patch_size
        self.n_tokens_per_tile = patch_grid_size ** 2 + 1
        scale = embed_dim ** -0.5
        self.local_token_positional_embedding = nn.Parameter(scale * torch.randn((self.n_tokens_per_tile, embed_dim)))
        self.global_token_positional_embedding = nn.Parameter(scale * torch.randn(max_num_tiles, max_num_tiles, self.n_tokens_per_tile, embed_dim))
        self.gate = nn.Parameter(torch.zeros(1))
        self._register_load_state_dict_pre_hook(self._load_state_dict_hook)

    @torch.no_grad()
    def _load_state_dict_hook(self, state_dict: 'Dict[str, Any]', prefix: 'str', *args: Tuple[Any], **kwargs: Dict[str, Any]) ->None:
        """
        Interpolates positional embeddings to accomodate different number of tiles
        and tokens per tile, in case the model was instantiated with different
        settings than the one you are loading the state dict from.

        For more info, please check self._resize_local_position_embedding and
        self._resize_global_position_embedding functions.

        Args:
            state_dict (Dict[str, Any]): The state dict to load.
            prefix (str): The prefix of the state dict.
            *args (Tuple[Any]): Additional positional arguments.
            **kwargs (Dict[str, Any]): Additional keyword arguments.

        Raises:
            ValueError: if loaded local or global embedding n_tokens_per_tile is not derived
                from a squared grid.
            ValueError: if after interpolation, the shape of the loaded local embedding
                is not compatible with the current embedding.
            ValueError: if after interpolation, the shape of the loaded global embedding
                is not compatible with the current embedding.
        """
        inpt_local_pos_embed = state_dict.get(prefix + 'local_token_positional_embedding')
        if inpt_local_pos_embed is not None:
            if isinstance(inpt_local_pos_embed, DTensor):
                local_embed_is_sharded = True
                local_embed_device_mesh = inpt_local_pos_embed.device_mesh
                local_embed_placements = inpt_local_pos_embed.placements
                inpt_local_pos_embed = inpt_local_pos_embed.full_tensor()
            else:
                local_embed_is_sharded = False
            inpt_n_tokens_per_tile, inpt_embed_dim = inpt_local_pos_embed.shape
            if math.sqrt(inpt_n_tokens_per_tile - 1) % 1 != 0:
                raise ValueError(f'Loaded local positional embedding has shape inpt_n_tokens_per_tile={inpt_n_tokens_per_tile!r}, which indicates a grid_size that is not squared. This is currently not supported.')
            tgt_n_tokens_per_tile, tgt_embed_dim = self.local_token_positional_embedding.shape
            inpt_local_pos_embed = self._resize_local_position_embedding(local_pos_embed=inpt_local_pos_embed, tgt_patch_grid_size=int(math.sqrt(tgt_n_tokens_per_tile - 1)))
            if local_embed_is_sharded:
                inpt_local_pos_embed = distribute_tensor(inpt_local_pos_embed, device_mesh=local_embed_device_mesh, placements=local_embed_placements)
            state_dict[prefix + 'local_token_positional_embedding'] = inpt_local_pos_embed
            if inpt_local_pos_embed.shape != self.local_token_positional_embedding.shape:
                raise ValueError(f'Loaded local positional embedding has shape {inpt_local_pos_embed.shape}, after interpolation. Expected shape {self.local_token_positional_embedding.shape}.')
        inpt_global_pos_embed = state_dict.get(prefix + 'global_token_positional_embedding')
        if inpt_global_pos_embed is not None:
            if isinstance(inpt_global_pos_embed, DTensor):
                global_embed_is_sharded = True
                global_embed_device_mesh = inpt_global_pos_embed.device_mesh
                global_embed_placements = inpt_global_pos_embed.placements
                inpt_global_pos_embed = inpt_global_pos_embed.full_tensor()
            else:
                global_embed_is_sharded = False
            _, _, inpt_n_tokens_per_tile, _ = inpt_global_pos_embed.shape
            if math.sqrt(inpt_n_tokens_per_tile - 1) % 1 != 0:
                raise ValueError(f'Loaded local positional embedding has shape inpt_n_tokens_per_tile={inpt_n_tokens_per_tile!r}, which indicates a grid_size that is not squared. This is currently not supported.')
            tgt_max_num_tiles_x, tgt_max_num_tiles_y, tgt_n_tokens_per_tile, tgt_embed_dim = self.global_token_positional_embedding.shape
            inpt_global_pos_embed = self._resize_global_position_embedding(global_pos_embed=inpt_global_pos_embed, tgt_max_num_tiles=tgt_max_num_tiles_x, tgt_patch_grid_size=int(math.sqrt(tgt_n_tokens_per_tile - 1)))
            if global_embed_is_sharded:
                inpt_global_pos_embed = distribute_tensor(inpt_global_pos_embed, device_mesh=global_embed_device_mesh, placements=global_embed_placements)
            state_dict[prefix + 'global_token_positional_embedding'] = inpt_global_pos_embed
            if inpt_global_pos_embed.shape != self.global_token_positional_embedding.shape:
                raise ValueError(f'Loaded global positional embedding has shape {inpt_global_pos_embed.shape}, after interpolation. Expected shape {self.global_token_positional_embedding.shape}.')

    @staticmethod
    def _resize_local_position_embedding(local_pos_embed: 'torch.Tensor', tgt_patch_grid_size: 'int') ->torch.Tensor:
        """
        Interpolates the local position embedding for a vision encoder to accommodate
        a different number of tokens per tile. This is the only dimension that
        changes during interpolation.

        Args:
            local_pos_embed (torch.Tensor): The position embeddings tensor to be resized. It
                has shape [n_tokens_per_tile, emb_dim], where the first token is the CLS token
                and n_tokens_per_tile = patch_grid_size**2 + 1.
            tgt_patch_grid_size (int): The target size of each patch grid, i.e.,
                the square root of the number of tokens per tile, excluding the class token.

        Returns:
            torch.Tensor: The resized position embeddings tensor of shape
                [tgt_n_tokens_per_tile, dim], where tgt_n_tokens_per_tile = tgt_patch_grid_size**2 + 1.

        Example:
            >>> import torch
            >>> import math
            >>> local_pos_embed = torch.randn((10*10+1, 64))  # Example input tensor
            >>> tgt_patch_grid_size = 20  # Target number of tokens per tile
            >>> resized_pos_embed = _resize_local_position_embedding(local_pos_embed, tgt_patch_grid_size)
            >>> print(resized_pos_embed.shape)
            torch.Size([20*20+1, 64])
        """
        inpt_n_tokens_per_tile, inpt_embed_dim = local_pos_embed.shape
        inpt_patch_grid_size = int(math.sqrt(inpt_n_tokens_per_tile - 1))
        cls_token, local_pos_embed = local_pos_embed[[0]], local_pos_embed[1:]
        local_pos_embed = local_pos_embed.reshape(1, inpt_patch_grid_size, inpt_patch_grid_size, -1).permute(0, 3, 1, 2)
        local_pos_embed = F.interpolate(local_pos_embed, size=[tgt_patch_grid_size, tgt_patch_grid_size], mode='bilinear', align_corners=True)
        local_pos_embed = local_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, inpt_embed_dim)
        local_pos_embed = local_pos_embed.squeeze(0)
        local_pos_embed = torch.cat([cls_token, local_pos_embed], dim=0)
        return local_pos_embed

    @staticmethod
    def _resize_global_position_embedding(global_pos_embed: 'torch.Tensor', tgt_max_num_tiles: 'int', tgt_patch_grid_size: 'int') ->torch.Tensor:
        """
        Interpolates the global position embedding for a vision encoder to accommodate new grid dimensions.
        The embedding dimension is not changed during interpolation, only max_num_tiles and num_tokens_per_tile.

        Args:
            global_pos_embed (torch.Tensor): The input global position embeddings tensor of shape
                [max_num_tiles_x, max_num_tiles_y, num_tokens_per_tile, embed_dim],
                where num_tokens_per_tile = inpt_patch_grid_size * inpt_patch_grid_size + 1 (CLS token), and
                max_num_tiles_x == max_num_tiles_y.
            tgt_max_num_tiles (int): The target maximum number of tiles along one dimension (assumed square grid).
            tgt_patch_grid_size (int): The target size of each patch grid, i.e., the square root of the number of tokens
                per tile, excluding the class token.


        Returns:
            torch.Tensor: The resized global position embeddings tensor of shape
                [tgt_max_num_tiles, tgt_max_num_tiles, tgt_patch_grid_size * tgt_patch_grid_size + 1, embed_dim].

        Example:
            >>> import torch
            >>> global_pos_embed = torch.arange(3*3*(2*2+1)*4).reshape((3, 3, 2*2+1, 4))  # Example input tensor
            >>> tgt_max_num_tiles = 2  # Target maximum number of tiles
            >>> tgt_patch_grid_size = 3  # Target patch grid size
            >>> resized_global_pos_embed = (
            >>> _resize_global_position_embedding(global_pos_embed, tgt_max_num_tiles, tgt_patch_grid_size))
            >>> print(resized_global_pos_embed.shape)
            torch.Size([2, 2, 3*3+1, 4])
        """
        pos_embed = global_pos_embed[:, :, 1:, :]
        cls_embed = global_pos_embed[:, :, [0], :]
        max_num_tiles_x, max_num_tiles_y, n_tokens_per_tile, embed_dim = pos_embed.shape
        inpt_patch_grid_size = int(math.sqrt(n_tokens_per_tile))
        pos_embed = pos_embed.reshape(max_num_tiles_x, max_num_tiles_y, inpt_patch_grid_size, inpt_patch_grid_size, embed_dim)
        pos_embed = pos_embed.permute(0, 2, 1, 3, 4).contiguous()
        pos_embed = pos_embed.reshape(max_num_tiles_x * inpt_patch_grid_size, max_num_tiles_y * inpt_patch_grid_size, embed_dim)
        pos_embed = pos_embed.unsqueeze(0)
        tgt_size = int(tgt_max_num_tiles * tgt_patch_grid_size), int(tgt_max_num_tiles * tgt_patch_grid_size)
        pos_embed = pos_embed.permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=tgt_size, mode='bilinear', align_corners=True)
        pos_embed = pos_embed.permute(0, 2, 3, 1).squeeze(0)
        pos_embed = pos_embed.view(tgt_max_num_tiles, tgt_patch_grid_size, tgt_max_num_tiles, tgt_patch_grid_size, embed_dim)
        pos_embed = pos_embed.permute(0, 2, 1, 3, 4).contiguous()
        pos_embed = pos_embed.view(tgt_max_num_tiles, tgt_max_num_tiles, int(tgt_patch_grid_size ** 2), embed_dim)
        cls_embed = cls_embed.permute(2, 3, 0, 1)
        cls_embed_resized = F.interpolate(cls_embed, size=(tgt_max_num_tiles, tgt_max_num_tiles), mode='bilinear', align_corners=True)
        cls_embed = cls_embed_resized.permute(2, 3, 0, 1)
        global_pos_embed = torch.cat([cls_embed, pos_embed], dim=2)
        return global_pos_embed

    def forward(self, x: 'torch.Tensor', aspect_ratio: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): torch.Tensor with shape
                (bsz * n_imgs, n_tiles, n_tokens_per_tile, embed_dim).
            aspect_ratio (torch.Tensor): torch.Tensor with shape (bsz * n_imgs, 2),
                where aspect_ratio[k] represents the aspect ratio of the k^th image
                of the batch before tile-cropping,  e.g. aspect_ratio[k] = (2,1).
        Returns:
            torch.Tensor: The input tensor with added positional embeddings.
        """
        bsz_and_n_imgs, n_tiles, n_tokens_per_tile, embed_dim = x.shape
        x = x + self.local_token_positional_embedding * (1 - self.gate.tanh())
        x = x.view(bsz_and_n_imgs, n_tiles, n_tokens_per_tile, embed_dim)
        for batch_idx, (n_tiles_h, n_tiles_w) in enumerate(aspect_ratio):
            n_non_padded_tiles = int(n_tiles_h * n_tiles_w)
            pos_embed = self.global_token_positional_embedding[:n_tiles_h, :n_tiles_w, :, :]
            pos_embed = pos_embed.reshape(n_non_padded_tiles, self.n_tokens_per_tile, embed_dim)
            pos_embed = pos_embed * self.gate.tanh()
            x[batch_idx, :n_non_padded_tiles, :, :] += pos_embed
        return x


class TilePositionalEmbedding(nn.Module):
    """
    Positional embedding for tiles, different for every tile, same for every token within a tile.

    Notice that tile is different from patch (token). For details, please check the documentation of
    :class:`torchtune.modules.vision_transformer.VisionTransformer`.

    Args:
        max_num_tiles (int): The maximum number of tiles an image can be divided into.
        embed_dim (int): The dimensionality of each tile embedding.
    """

    def __init__(self, max_num_tiles: 'int', embed_dim: 'int'):
        super().__init__()
        self.embed_dim = embed_dim
        scale = embed_dim ** -0.5
        self.embedding = nn.Parameter(scale * torch.randn(max_num_tiles, max_num_tiles, 1, embed_dim))
        self.gate = nn.Parameter(torch.zeros(1))
        self._register_load_state_dict_pre_hook(self._load_state_dict_hook)

    @torch.no_grad()
    def _load_state_dict_hook(self, state_dict: 'Dict[str, Any]', prefix: 'str', *args: Tuple[Any], **kwargs: Dict[str, Any]):
        """
        Interpolates positional embeddings to accomodate different number of tiles,
        in case the model was instantiated with different
        settings than the one you are loading the state dict from.

        For more info, check self._dynamic_resize function.

        Args:
            state_dict (Dict[str, Any]): The state dict to load.
            prefix (str): The prefix of the state dict.
            *args (Tuple[Any]): Additional positional arguments.
            **kwargs (Dict[str, Any]): Additional keyword arguments.

        Raises:
            ValueError: if the shape of the loaded embedding is not compatible with the current embedding.
            ValueError: if max_num_tiles_x, max_num_tiles_y are not equal.
            ValueError: if after interpolation, the shape of the loaded embedding is not compatible with the current embedding.
        """
        embedding = state_dict.get(prefix + 'embedding')
        if embedding is not None:
            if isinstance(embedding, DTensor):
                embedding_is_sharded = True
                device_mesh = embedding.device_mesh
                placements = embedding.placements
                embedding = embedding.full_tensor()
            else:
                embedding_is_sharded = False
            tgt_max_num_tiles_x, tgt_max_num_tiles_y, tgt_num_tokens, tgt_emb = self.embedding.shape
            inpt_max_num_tiles_x, inpt_max_num_tiles_y, inpt_num_tokens, inpt_emb = state_dict[prefix + 'embedding'].shape
            if inpt_num_tokens != tgt_num_tokens or inpt_emb != tgt_emb:
                raise ValueError(f"Expected embedding shape to be (..., num_tokens, tgt_emb) to match but found shapes {self.embedding.shape} and {state_dict[prefix + 'embedding'].shape}")
            if inpt_max_num_tiles_x != inpt_max_num_tiles_y:
                raise ValueError(f'Expected max_num_tiles_x, max_num_tiles_y to be equal but found, but found(max_num_tiles_x, max_num_tiles_y, 1, embed_dim) = {self.embedding.shape}')
            embedding_new = self._resize_position_embedding(embedding, tgt_max_num_tiles=tgt_max_num_tiles_x)
            if embedding_is_sharded:
                embedding_new = distribute_tensor(embedding_new, device_mesh=device_mesh, placements=placements)
            state_dict[prefix + 'embedding'] = embedding_new
            if embedding_new.shape != self.embedding.shape:
                raise ValueError(f'Expected embedding shape and embedding_new.shape to match but found shapes {self.embedding.shape} and {embedding_new.shape}')

    @staticmethod
    def _resize_position_embedding(embedding: 'torch.Tensor', tgt_max_num_tiles: 'int') ->torch.Tensor:
        """
        Interpolates positional embeddings to accomodate a different max_num_tiles. These
        are the only dimensions that changes during interpolation.

        Args:
            embedding (torch.Tensor): torch.Tensor with shape (max_num_tiles, max_num_tiles, 1, embed_dim
            tgt_max_num_tiles (int): The number of tiles to resize to.

        Returns:
            torch.Tensor: The resized embedding.

        Example:
            >>> import torch
            >>> # create dummy embedding
            >>> embedding = torch.arange(2*2*2*2).reshape(2, 2, 2, 2).float()
            >>> resized_embed = _dynamic_resize(embedding, tgt_max_num_tiles=1)
            >>> print(resized_embed.shape)
            >>> torch.Size([1, 1, 2, 2])
        """
        embedding = embedding.permute(2, 3, 0, 1)
        embedding = F.interpolate(embedding, size=(tgt_max_num_tiles, tgt_max_num_tiles), mode='bilinear', align_corners=True)
        embedding = embedding.permute(2, 3, 0, 1)
        return embedding

    def forward(self, x: 'torch.Tensor', aspect_ratio: 'torch.Tensor') ->torch.Tensor:
        """
        args:
            x (torch.Tensor): torch.Tensor with shape (bsz * n_imgs, n_tiles, n_tokens_per_tile, embed_dim).
            aspect_ratio (torch.Tensor): torch.Tensor with shape (bsz * n_imgs, 2),
                representing the aspect ratio of the image before tile-cropping, e.g. (2,1).
        returns:
            torch.Tensor: The input tensor with added positional embeddings.
        """
        for batch_idx, (n_tiles_h, n_tiles_w) in enumerate(aspect_ratio):
            n_non_padded_tiles = int(n_tiles_h * n_tiles_w)
            pos_embed = self.embedding[:n_tiles_h, :n_tiles_w, :, :]
            pos_embed = pos_embed.reshape(n_non_padded_tiles, 1, self.embed_dim)
            x[batch_idx, :n_non_padded_tiles, :, :] += pos_embed * self.gate.tanh()
        return x


def tile_crop(image: 'torch.Tensor', tile_size: 'int') ->torch.Tensor:
    """
    Divides a tensor into equally sized tiles. The tensor should be divisible by tile_size.

    Args:
        image (torch.Tensor): Input image to crop into tiles.
        tile_size (int): Size of each tile.

    Returns:
        torch.Tensor: torch.Tensor of shape [num_tiles, channel_size, tile_size, tile_size]

    Examples:
        >>> image = torch.rand(3, 200, 300)
        >>> tiles = tile_crop(image, tile_size=50)
        >>> tiles.shape # 4x6 = 24 tiles
        torch.Size([24, 3, 50, 50])

        >>> image = torch.rand(3, 400, 600)
        >>> tiles = tile_crop(image, tile_size=200)
        >>> tiles.shape # 2x3 = 6 tiles
        torch.Size([6, 3, 200, 200])
    """
    channel_size, height, width = image.shape
    assert height % tile_size == 0 and width % tile_size == 0, f'Image size {height}x{width} is not divisible by tile size {tile_size}'
    tiles_height = height // tile_size
    tiles_width = width // tile_size
    reshaped = image.view(channel_size, tiles_height, tile_size, tiles_width, tile_size)
    transposed = reshaped.permute(1, 3, 0, 2, 4)
    tiles = transposed.contiguous().view(tiles_height * tiles_width, channel_size, tile_size, tile_size)
    return tiles


class _CLIPImageTransform(torch.nn.Module):

    def __init__(self, resample: 'str', image_mean: 'Optional[List[float]]', image_std: 'Optional[List[float]]', tile_size: 'int', max_num_tiles: 'int', antialias: 'bool'):
        super().__init__()
        self.resample = resample
        self.image_mean = image_mean
        self.image_std = image_std
        self.tile_size = tile_size
        self.max_num_tiles = max_num_tiles
        self.antialias = antialias
        self.tile_crop = tile_crop
        self.pad = torch.nn.functional.pad

    def check_variable_bounds_for_export(self, target_size: 'List[int]', canvas_size: 'List[int]', lower: 'int', upper: 'int') ->None:
        """
        Performs torch._checks used to export the model. For eager mode usage, please disregard.
        The check mitigates data dependent errors that may occur during torch.export. It installs a
        deferred runtime assert, instead of a compile-time guard. Data dependent errors usually occur
        in models with data-dependent control flow, eg. via .item(), tolist(), nonzero(). For more
        context: https://docs.google.com/document/d/1HSuTTVvYH1pTew89Rtpeu84Ht3nQEFTYhAX3Ypa_xJs/edit
        """
        for var in canvas_size:
            torch._check(var >= lower)
            torch._check(var <= upper)
        for i in range(len(target_size)):
            torch._check(target_size[i] >= lower)
            torch._check(target_size[i] <= canvas_size[i])

    def forward(self, image: 'torch.Tensor', target_size: 'torch.Tensor', canvas_size: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the core transformations involved in CLIPImageTransform;
        1. Resize the image to target_size.
        2. Pad the image to canvas_size.
        3. Normalize the image using image_mean and image_std.
        4. Reshape the image tensor into [n, channels, tile_size, tile_size].
        Args:
            image (torch.Tensor): image as a 3D tensor in form [C, H, W].
            target_size (torch.Tensor): tensor of shape (2,) containing the target_height and target_width for resize.
            canvas_size (torch.Tensor): tensor of shape (2,) containing the canvas_height and canvas_width for padding.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Image tensor of shape [n, channels, tile_size, tile_size]
                and aspect ratio tensor of shape [1, 2].
        """
        target_h, target_w = target_size.tolist()
        canvas_h, canvas_w = canvas_size.tolist()
        self.check_variable_bounds_for_export(target_size=[target_h, target_w], canvas_size=[canvas_h, canvas_w], lower=2, upper=self.tile_size * self.max_num_tiles)
        image = torchvision.transforms._functional_tensor.resize(image, size=[target_h, target_w], interpolation=self.resample, antialias=self.antialias)
        padding = [0, canvas_w - target_w, 0, canvas_h - target_h]
        output = self.pad(image, padding)
        if self.image_mean is not None and self.image_std is not None:
            output = F.normalize(output, self.image_mean, self.image_std)
        tiles = self.tile_crop(output, self.tile_size)
        aspect_ratio = canvas_size // self.tile_size
        return tiles, aspect_ratio


class GemmaNormEmbeddings(nn.Embedding):
    """Module with Embedding and normalization specific to Gemma.
    Gemma requires normalization right after the embeddings. By merging both
    steps in a single module, we can utilize directly
    :class:`~torch.modules.TransformerDecoder`.

    For more details about the embedding module, please see
    https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html

    Args:
        num_embeddings (int): size of the dictionary of embeddings.
        embedding_dim (int): the size of each embedding vector.
        *args: Variable length argument list to be passed to the Embedding module.
        **kwargs: Arbitrary keyword arguments to be passed to the Embedding module.

    Example:
        >>> import torch
        >>> from torchtune.models.gemma import GemmaNormEmbeddings
        >>> embeddings = GemmaNormEmbeddings(2, 4)
        >>> x = torch.randint(0, 2, (1, 3)) # ids can be 0 or 1
        >>> print(x)
        >>> print(embeddings(x))
        >>> print(embeddings(x).shape)
        tensor([[1, 0, 0]])
        tensor([[[-0.2152, -2.1914,  2.8491, -0.4824],
                 [-3.6621, -1.0267,  1.5947, -1.7349],
                 [-3.6621, -1.0267,  1.5947, -1.7349]]], grad_fn=<MulBackward0>)
        torch.Size([1, 3, 4])
    """

    def __init__(self, num_embeddings: 'int', embedding_dim: 'int', *args, **kwargs):
        super().__init__(num_embeddings, embedding_dim, *args, **kwargs)
        self.embedding_dim = embedding_dim

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = super().forward(x)
        return x * torch.tensor(self.embedding_dim ** 0.5, dtype=x.dtype)


class GemmaRMSNorm(nn.Module):

    def __init__(self, dim: 'int', eps: 'float'=1e-06):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.scale.float())
        return output.type_as(x)


class KVCache(nn.Module):
    """
    Standalone ``nn.Module`` containing a kv-cache to cache past key and values during inference.

    Args:
        batch_size (int): batch size model will be run with
        max_seq_len (int): maximum sequence length model will be run with
        num_heads (int): number of heads. We take num_heads instead of num_kv_heads because
            the cache is created after we've expanded the key and value tensors to have the
            same shape as the query tensor. See attention.py for more details
        head_dim (int): per-attention head embedding dimension
        dtype (torch.dtype): dtype for the caches
    """

    def __init__(self, batch_size: 'int', max_seq_len: 'int', num_heads: 'int', head_dim: 'int', dtype: 'torch.dtype') ->None:
        super().__init__()
        cache_shape = batch_size, num_heads, max_seq_len, head_dim
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype), persistent=False)
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype), persistent=False)
        self.register_buffer('cache_pos', torch.arange(0, cache_shape[2]), persistent=False)
        self.batch_size = batch_size

    def reset(self) ->None:
        """Reset the cache to zero."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_pos -= self.size

    @property
    def size(self) ->int:
        return self.cache_pos[0].item()

    def update(self, k_val: 'torch.Tensor', v_val: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        """Update KV cache with the new ``k_val``, ``v_val`` and return the updated cache.

        Note:
            When updating the KV cache, it is assumed that subsequent updates should update key-value
            positions in consecutive sequence positions. If you wish to update cache values which have
            already been filled, use ``.reset()``, which will reset the cache to the zero-th position.

        Example:
            >>> cache = KVCache(batch_size=2, max_seq_len=16, num_heads=4, head_dim=32, dtype=torch.bfloat16)
            >>> keys, values = torch.ones((2, 4, 8, 32)), torch.ones((2, 4, 8, 32))
            >>> cache.update(keys, values)
            >>> # now positions 0 through 7 are filled
            >>> cache.size
            >>> 8
            >>> keys, values = torch.ones((2, 4, 1, 32)), torch.ones((2, 4, 1, 32))
            >>> cache.update(keys, values)
            >>> # this will fill at position 8
            >>> cache.size
            >>> 9

        Args:
            k_val (torch.Tensor): Current key tensor with shape [B, H, S, D]
            v_val (torch.Tensor): Current value tensor with shape [B, H, S, D]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated key and value cache tensors, respectively.

        Raises:
            AssertionError: if the sequence length of ``k_val`` is longer than the maximum cache sequence length.
            ValueError: if the batch size of the new key (or value) tensor is greater than the batch size
                used during cache setup.
        """
        bsz, _, seq_len, _ = k_val.shape
        if bsz > self.k_cache.shape[0]:
            raise ValueError(f'The current cache has been setup with a batch size of {self.k_cache.shape[0]}, but found new key tensors with batch size {k_val.shape[0]}!')
        assert self.cache_pos[0] + seq_len <= self.k_cache.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, self.cache_pos[:seq_len]] = k_val
        v_out[:, :, self.cache_pos[:seq_len]] = v_val
        self.cache_pos += seq_len
        return k_out, v_out


def get_logger(level: 'Optional[str]'=None) ->logging.Logger:
    """
    Get a logger with a stream handler.

    Args:
        level (Optional[str]): The logging level. See https://docs.python.org/3/library/logging.html#levels for list of levels.

    Example:
        >>> logger = get_logger("INFO")
        >>> logger.info("Hello world!")
        INFO:torchtune.utils._logging:Hello world!

    Returns:
        logging.Logger: The logger.
    """
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logger.addHandler(logging.StreamHandler())
    if level is not None:
        level = getattr(logging, level.upper())
        logger.setLevel(level)
    return logger


logger = get_logger('DEBUG')


class TanhSoftCapping(nn.Module):

    def __init__(self, capping_value: 'float') ->None:
        super().__init__()
        self.capping_value = capping_value

    def forward(self, attn_weights):
        attn_weights = attn_weights / self.capping_value
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * self.capping_value
        return attn_weights


class Gemma2FinalNorm(nn.Module):
    """
    Combines RMSNorm and SoftCapping
    """

    def __init__(self, capping_value: 'float', embed_dim: 'int', eps: 'float') ->None:
        super().__init__()
        self.capping_value = capping_value
        self.rms_norm = GemmaRMSNorm(embed_dim, eps=eps)
        self.logit_capping = TanhSoftCapping(capping_value)

    def forward(self, x):
        x = self.rms_norm(x)
        x = self.logit_capping(x)
        return x


class Llama3ScaledRoPE(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864 with additional
    scaling from https://github.com/meta-llama/llama-models/blob/dc42f22a3b05502e7296402b019a51f57fa045c9/models/llama3_1.

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Default scaling factors are from the following Meta-Llama code:
    https://github.com/meta-llama/llama-models/blob/dc42f22a3b05502e7296402b019a51f57fa045c9/models/llama3_1/api/model.py#L41

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
        scale_factor (int): scaling factor for theta. Default: 8
        low_freq_factor (int): low frequency factor for scaling theta. Default: 1
        high_freq_factor (int): high frequency factor for scaling theta. Default: 4
        old_context_len (int): old context length for scaling theta. Default: 8192
    """

    def __init__(self, dim: 'int', max_seq_len: 'int'=4096, base: 'int'=10000, scale_factor: 'int'=8, low_freq_factor: 'int'=1, high_freq_factor: 'int'=4, old_context_len: 'int'=8192) ->None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.scale_factor = scale_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = old_context_len
        self.is_cache_built = False
        self.rope_init()

    def rope_init(self):
        """
        Warning: this is called in recipes before torch.compile,
        so that the cache is built in advance.
        """
        freqs = 1.0 / self.base ** (torch.arange(0, self.dim, 2)[:self.dim // 2].float() / self.dim)
        if freqs.is_meta:
            return
        theta = self.apply_scaling(freqs, self.scale_factor, self.low_freq_factor, self.high_freq_factor, self.old_context_len)
        self.register_buffer('theta', theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)
        self.is_cache_built = True

    def build_rope_cache(self, max_seq_len: 'int'=4096) ->None:
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)
        idx_theta = torch.einsum('i, j -> ij', seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer('cache', cache, persistent=False)

    def apply_scaling(self, freqs: 'torch.Tensor', scale_factor: 'int', low_freq_factor: 'int', high_freq_factor: 'int', old_context_len: 'int'):
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        new_freqs = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / scale_factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
        return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

    def forward(self, x: 'torch.Tensor', *, input_pos: Optional[torch.Tensor]=None) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [b, s, n_h, h_d]
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim

        Raises:
            RuntimeError: if RoPE cache is not initialized prior to forward call
        """
        if not self.is_cache_built:
            raise RuntimeError('RoPE cache is not built. Please call rope_init() first.')
        seq_len = x.size(1)
        rope_cache = self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
        x_out = torch.stack([xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1], xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1]], -1)
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


class Llama3VisionProjectionHead(nn.Module):
    """Projection transformer to adapt the output of a
    pretrained frozen encoder (CLIP) to a pretrained decoder model.
    For example, nn.Sequential(CLIP(), Llama3VisionProjectionHead()).

    Args:
        layers (nn.Module): Transformer Decoder layers
        output (nn.Module): Output linear layer. Input dim is
            (num_hidden + 1) * encoder_dim and output is decoder_dim.
        num_hidden_inputs (int): Number of expected hidden state inputs
    """

    def __init__(self, layers: 'nn.Module', output: 'nn.Module', num_hidden_inputs: 'int'=0) ->None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.output = output
        self.num_hidden = num_hidden_inputs

    def forward(self, x: 'torch.Tensor', hidden_states: 'Optional[List[torch.Tensor]]'=None) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape [b x i x t x e x d]
            hidden_states (Optional[List[torch.Tensor]]): list of hidden states
                from the encoder. Each hidden state has the same shape as x.

        Returns:
            Tensor: output tensor of a sequence of embedings [b x s x d]
                where sequence length is num_imgs*num_tiles+num_embeds

        Notation used for tensor shapes:
            - b: batch size
            - i: number of images
            - t: number of tiles (where a single image is broken into multiple tiles)
            - e: number of embeds per tile (e.g. CLS embed + patch embeds, etc.)
            - s: sequence length computed by i*t*e
            - d: embed dim
        """
        bsz, imgs, tiles, embeds, dim = x.shape
        x = x.view(bsz * imgs, tiles * embeds, dim)
        for layer in self.layers:
            x = layer(x)
        x = x.view(bsz, imgs, tiles, embeds, dim)
        if self.num_hidden > 0:
            hidden_states = torch.stack(hidden_states, dim=-1)
            hidden_states = hidden_states.view(bsz, imgs, tiles, embeds, -1)
            x = torch.cat([x, hidden_states], dim=-1)
        x = self.output(x).reshape(bsz, imgs * tiles * embeds, -1)
        return x


class Llama3VisionEncoder(nn.Module):
    """Vision encoder model for Llama 3.2 Vision. This combines a pretrained
    vision encoder with a learnable projection head. The projection head
    is converted to a fusion module and supports fusion utils.

    Args:
        clip (nn.Module): CLIP encoder vision model
        projection_head (nn.Module): projection_head that takes embeddings
            with dimension encoder_dim as input and outputs embeddings of
            size decoder_dim.
    """

    def __init__(self, clip: 'nn.Module', projection_head: 'nn.Module') ->None:
        super().__init__()
        self.clip = clip
        self.projection = projection_head
        register_fusion_module(self.projection)

    def forward(self, images: 'torch.Tensor', aspect_ratio: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """
        Args:
            images (torch.Tensor): Image tensor with shape [b x i x t x c x w x h]
            aspect_ratio (Optional[torch.Tensor]): Tensor with shape [b x i x 2]. If all
                images have a single tile, i.e. they were not tile-cropped, it should be None.
                Used to calculate the positional embeddings for the tiles.

        Returns:
            Tensor: output tensor of a sequence of embedings [b x s x d]
                where sequence length is num_imgs*num_tiles+num_embeds

         Notation used for tensor shapes:
            - b: batch size
            - i: number of images
            - t: number of tiles (where a single image is broken into multiple tiles)
            - c: number of image channels (e.g. rgb = 3)
            - w: image width
            - h: image height
            - s: sequence length computed by i*t*clip_embeds_per_tile
            - d: embed dim
        """
        x, hidden_states = self.clip(images, aspect_ratio)
        x = self.projection(x, hidden_states)
        return x


class Phi3RotaryPositionalEmbeddings(nn.Module):
    """
    RoPE Embeddings used in the Phi3 model.
    Ref: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct

    This class is not numerically equivalent to the RoPE Embedding module
    used by Llama2 and Llama3.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim`` // ``num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(self, dim: 'int', max_seq_len: 'int'=4096, base: 'int'=10000) ->None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / self.base ** (torch.arange(0, self.dim, 2)[:self.dim // 2].float() / self.dim)
        self.register_buffer('theta', theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: 'int'=4096) ->None:
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)
        idx_theta = torch.einsum('i, j -> ij', seq_idx, self.theta).float()
        freqs = torch.cat([idx_theta, idx_theta], dim=-1)
        cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
        self.register_buffer('cache', cache, persistent=False)

    def forward(self, x: 'torch.Tensor', input_pos: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [b, s, n_h, h_d]
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim

        TODO: The implementation below can be made more efficient
        for inference.
        """
        seq_len = x.size(1)
        head_dim = x.size(-1)
        rope_cache = self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        rope_cache = rope_cache.view(-1, seq_len, 1, head_dim * 2)
        cos = rope_cache[..., :head_dim]
        sin = rope_cache[..., head_dim:]
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        rotated = torch.cat((-x2, x1), dim=-1)
        x_out = x * cos + rotated * sin
        return x_out.type_as(x)


class Qwen2RotaryPositionalEmbeddings(nn.Module):
    """
    RoPE Embeddings used in the Qwen2 model.
    Ref: https://huggingface.co/Qwen/Qwen2-7B-Instruct

    This class is not numerically equivalent to the RoPE Embedding module
    used by Llama2 and Llama3.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim`` // ``num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (float): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(self, dim: 'int', max_seq_len: 'int'=4096, base: 'float'=1000000.0) ->None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / self.base ** (torch.arange(0, self.dim, 2)[:self.dim // 2].float() / self.dim)
        self.register_buffer('theta', theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: 'int'=4096) ->None:
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)
        idx_theta = torch.einsum('i, j -> ij', seq_idx, self.theta).float()
        freqs = torch.cat([idx_theta, idx_theta], dim=-1)
        cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
        self.register_buffer('cache', cache, persistent=False)

    def forward(self, x: 'torch.Tensor', input_pos: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [b, s, n_h, h_d]
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim

        TODO: The implementation below can be made more efficient
        for inference.
        """
        seq_len = x.size(1)
        head_dim = x.size(-1)
        rope_cache = self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        rope_cache = rope_cache.view(-1, seq_len, 1, head_dim * 2)
        cos = rope_cache[..., :head_dim]
        sin = rope_cache[..., head_dim:]
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        rotated = torch.cat((-x2, x1), dim=-1)
        x_out = x * cos + rotated * sin
        return x_out.type_as(x)


def torch_version_ge(version: 'str') ->bool:
    """
    Check if torch version is greater than or equal to the given version.

    Args:
        version (str): The torch version to compare against

    Returns:
        bool: True if torch version is greater than or equal to the given version.

    Example:
        >>> print(torch.__version__)
        2.4.0
        >>> torch_version_ge("2.0")
        True
    """
    return version in torch.__version__ or torch.__version__ >= version


_SUPPORTS_FLEX_ATTENTION = torch_version_ge('2.5.0') and torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 5)


def log_rank_zero(logger: 'logging.Logger', msg: 'str', level: 'int'=logging.INFO) ->None:
    """
    Logs a message only on rank zero.

    Args:
        logger (logging.Logger): The logger.
        msg (str): The warning message.
        level (int): The logging level. See https://docs.python.org/3/library/logging.html#levels for values.
            Defaults to ``logging.INFO``.
    """
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank != 0:
        return
    logger.log(level, msg)


@lru_cache(None)
def log_once(logger: 'logging.Logger', msg: 'str', level: 'int'=logging.INFO) ->None:
    """
    Logs a message only once. LRU cache is used to ensure a specific message is
    logged only once, similar to how :func:`~warnings.warn` works when the ``once``
    rule is set via command-line or environment variable.

    Args:
        logger (logging.Logger): The logger.
        msg (str): The warning message.
        level (int): The logging level. See https://docs.python.org/3/library/logging.html#levels for values.
            Defaults to ``logging.INFO``.
    """
    log_rank_zero(logger=logger, msg=msg, level=level)


def _sdpa_or_flex_attention() ->Callable:
    """
    Helper function to decide when to call flex attention or SDPA. It will use
    flex attention if ALL of the following conditions are met, otherwise it will
    default to SDPA:
    - torch version >= 2.5.0
    - we are sample packing, therefore mask is a BlockMask
    - torch.cuda.get_device_capability() >= (7, 5)
    """
    if _SUPPORTS_FLEX_ATTENTION:

        def _attention_call(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', mask: 'Optional[_MaskType]', dropout_p: 'float', is_causal: 'bool') ->torch.Tensor:
            if isinstance(mask, BlockMask):
                log_once(_log, 'Using flex attention for attention computation since a BlockMask was passed in.', level=logging.DEBUG)
                if dropout_p > 0.0:
                    raise ValueError('Flex attention does not support dropout. Please set dropout to 0.0.')
                return compile_friendly_flex_attention(q, k, v, block_mask=mask)
            else:
                if mask is not None:
                    mask = mask[:, None, :, :]
                return nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=is_causal)
    else:

        def _attention_call(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', mask: 'Optional[_MaskType]', dropout_p: 'float', is_causal: 'bool') ->torch.Tensor:
            if mask is not None:
                mask = mask[:, None, :, :]
            return nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=is_causal)
    return _attention_call


class Fp32LayerNorm(nn.LayerNorm):
    """
    Wrapper around :class:`~torch.nn.LayerNorm` to support mixed-precision training.
    """

    def __init__(self, *args: Any, **kwargs: Any) ->None:
        super().__init__(*args, **kwargs)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The normalized output tensor having the same shape as ``x``.
        """
        output = nn.functional.layer_norm(x.float(), self.normalized_shape, self.weight.float() if self.weight is not None else None, self.bias.float() if self.bias is not None else None, self.eps)
        return output.type_as(x)


class CEWithChunkedOutputLoss(torch.nn.Module):
    """
    Cross-entropy with chunked outputs that saves memory by only upcasting one chunk at a time.

    Whenever the model is trained with bf16, before running CE, we have to upcast
    it to fp32 for better accuracy and stability. When upcasting happens, the memory usage doubles.
    Models like llama3 have large vocabulary size and, therefore, have a large output
    tensor of shape ``(bsz, num_tokens, vocab_size)``. If we chunk on the token level, you can still compute
    the cross entropy normally, but upcasting only one chunk at a time saves considerable memory.

    The CE and upcasting have to be compiled together for better performance.
    When using this class, we recommend using :func:`torch.compile` only on the method ``compute_cross_entropy``.
    The gains from chunking won't be realized if you compile the entire class.

    For more details, please refer to: https://github.com/pytorch/torchtune/pull/1390
    """

    def __init__(self, num_output_chunks: 'int'=8, ignore_index: 'int'=-100):
        super().__init__()
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index

    def compute_cross_entropy(self, logits: 'torch.Tensor', labels: 'torch.Tensor', normalize: 'bool'=True) ->torch.Tensor:
        """
        Upcast logits to fp32 and compute cross entropy loss.
        """
        return F.cross_entropy(logits.float(), labels, ignore_index=self.ignore_index, reduction='sum')

    def forward(self, logits: 'List[torch.Tensor]', labels: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            logits (List[torch.Tensor]): List of chunked logits of length
                ``self.num_output_chunks``, where each chunk has shape
                ``(batch_size, num_tokens / num_output_chunks, vocab_size)``.
            labels (torch.Tensor): Ground truth labels of shape ``(batch_size, num_tokens)``.

        Returns:
            torch.Tensor: Cross entropy loss of shape (1,).

        Example:
            >>> loss_fn = ChunkedCrossEntropyLoss()
            >>>
            >>> h = torch.tensor([bsz, num_tokens, dim])
            >>> output_chunks = [model.output(chunk) for chunk in h.chunk(num_chunks, dim=1)]
            >>>
            >>> labels = torch.tensor([bsz, num_tokens])
            >>> loss = loss_fn(output_chunks, labels)
        """
        total_elements = (labels != self.ignore_index).sum()
        labels = [target_chunk.reshape(-1) for target_chunk in labels.chunk(self.num_output_chunks, dim=1)]
        logits = [logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits]
        total_loss = 0.0
        for logits_chunk, labels_chunk in zip(logits, labels):
            total_loss += self.compute_cross_entropy(logits_chunk, labels_chunk)
        return total_loss / total_elements


class ForwardKLLoss(torch.nn.Module):
    """
    The Kullback-Leibler divergence loss for valid indexes.
    Implementation of https://github.com/jongwooko/distillm/blob/17c0f98bc263b1861a02d5df578c84aea652ee65/distillm/losses.py

    Args:
        ignore_index (int):  Specifies a target value that is ignored and does not contribute to the input gradient.
            The loss is divided over non-ignored targets.
            Default: -100.
    """

    def __init__(self, ignore_index: 'int'=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, student_logits: 'torch.Tensor', teacher_logits: 'torch.Tensor', labels: 'torch.Tensor', normalize: 'bool'=True) ->torch.Tensor:
        """
        Args:
            student_logits (torch.Tensor): logits from student model of shape
                (batch_size*num_tokens, vocab_size).
            teacher_logits (torch.Tensor): logits from teacher model of shape
                (batch_size*num_tokens, vocab_size).
            labels (torch.Tensor): Ground truth labels of shape
                (batch_size, vocab_size).
            normalize (bool): Whether to normalize the loss by the number of unmasked elements.

        Returns:
            torch.Tensor: KL divergence loss of shape (1,).
        """
        teacher_prob = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(student_logits)
        student_logprob = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_prob * student_logprob, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (labels != self.ignore_index).int()
        if not normalize:
            return -torch.sum(x * mask.view(-1), dim=0)
        if torch.sum(mask.view(-1), dim=0) == 0:
            return torch.tensor(0.0, device=x.device)
        return -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)


class ForwardKLWithChunkedOutputLoss(torch.nn.Module):
    """
    Forward KL with chunked outputs that saves memory by only upcasting one chunk at a time.

    Since the model is trained with bf16, before computing KL divergence, we have to upcast
    it to fp32 for better accuracy and stability. When upcasting happens, the memory usage doubles.
    Models like llama3 have large vocabulary size and, therefore, have a large output
    result (bsz, num_tokens, vocab_size). If we chunk on the token level, you can still compute
    the cross entropy normally, but upcasting only one chunk at a time saves considerable memory.

    Args:
        num_output_chunks (int): Number of chunks to chunk the output into. Each chunk has shape
            (batch_size, num_tokens / num_output_chunks, vocab_size).
            Default: 8
        ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.
            The loss is divided over non-ignored targets.
            Default: -100
    """

    def __init__(self, num_output_chunks: 'int'=8, ignore_index: 'int'=-100):
        super().__init__()
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index
        self.fkl_loss = ForwardKLLoss(ignore_index)

    def forward(self, student_logits: 'List[torch.Tensor]', teacher_logits: 'List[torch.Tensor]', labels: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            student_logits (List[torch.Tensor]): List of chunked logits from student model of length
                ``self.num_output_chunks``, where each chunk has shape
                (batch_size, num_tokens / num_output_chunks, vocab_size).
            teacher_logits (List[torch.Tensor]): List of chunked logits from teacher model of length
                ``self.num_output_chunks``, where each chunk has shape
                (batch_size, num_tokens / num_output_chunks, vocab_size).
            labels (torch.Tensor): Ground truth labels of shape (batch_size, num_tokens).

        Returns:
            torch.Tensor: KL divergence loss of shape (1,).

        Example:
            >>> loss_fn = ForwardKLWithChunkedOutputLoss()
            >>>
            >>> h = torch.tensor([bsz, num_tokens, dim])
            >>> output_chunks = [model.output(chunk) for chunk in h.chunk(num_chunks, dim=1)]
            >>> teacher_chunks = [teacher_model.output(chunk) for chunk in h.chunk(num_chunks, dim=1)]
            >>> labels = torch.tensor([bsz, num_tokens])
            >>> loss = loss_fn(output_chunks, teacher_chunks, labels)
        """
        teacher_logits = [teacher_logits_chunk.reshape(-1, teacher_logits_chunk.size(-1)) for teacher_logits_chunk in teacher_logits]
        student_logits = [student_logits_chunk.reshape(-1, student_logits_chunk.size(-1)) for student_logits_chunk in student_logits]
        mask = (labels != self.ignore_index).int()
        labels = [target_chunk.reshape(-1) for target_chunk in labels.chunk(self.num_output_chunks, dim=1)]
        total_fkl_loss = 0.0
        for student_chunk, teacher_chunk, label_chunk in zip(student_logits, teacher_logits, labels):
            total_fkl_loss += self.fkl_loss(student_chunk, teacher_chunk, label_chunk, normalize=False)
        return total_fkl_loss / torch.sum(mask.view(-1), dim=0)


class FrozenNF4Linear(nn.Linear):
    """
    A linear layer similar to ``torch.nn.Linear`` but uses a quantized
    NF4Tensor as its weight. This class also freezes its ``weight`` parameter
    and is meant to be used as the base Linear layer for modeling
    use cases such as QLoRA where base model parameters are frozen.

    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        device (Optional[torch.device]): device to use for the underlying weight. If ``None``, uses the default
            device given by `torch.get_default_device()`.
        bias (bool): whether to include bias in the linear layer. Default: False
        **kwargs: any additional arguments to pass to the underlying Linear layer.

    """

    def __init__(self, in_dim: 'int', out_dim: 'int', device: 'Optional[torch.device]'=None, bias: 'bool'=False, **kwargs):
        super().__init__(in_dim, out_dim, device=device, bias=bias, **kwargs)
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        nf4_weight = to_nf4(self.weight)
        torch.utils.swap_tensors(self.weight, torch.nn.Parameter(nf4_weight, requires_grad=False))

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        """
        Runs linear operation with input tensor as given by `input`. Computation happens in higher
        precision, though only the nf4 weight is saved for backward for gradient computation to ensure
        additional memory is not used.
        Args:
            input (torch.Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        out = linear_nf4(input=input, weight=self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


class FusionLayer(nn.Module):
    """Fusion layer as introduced in `Flamingo: a Visual Language Model for Few-Shot Learning <https://arxiv.org/abs/2204.14198>`_.

    Deep Fusion model architectures combine pretrained encoder models with pretrained
    language models by infusing the encoder outputs into the middle layers of the LLM.
    This allows the language model to interpret the enocder outputs as text and
    "understand" any modality for which you can train an encoder. To enable the language model
    to adapt to the encoder outputs, the FusionLayer fuses a new learnable layer to an existing
    decoder (language model) layer. This additional layer can take the encoder embeddings and
    learn to combine them with the token embeddings from the decoder. The module supports fusing
    the new layer before or after the original, in Flamingo the new layer is fused before the original.

    The original layer is wrapped in FusionLayer such that it maintains its original state_dict
    key and the pre-trained checkpoint isn't broken. The new layer parameters are available
    through ``fusion_params`` to separately control if they're trainable or not.

    Example:
        >>> # Original decoder style transformer
        >>> layer = nn.TransformerSelfAttentionLayer(...)
        >>> model = TransformerDecoder(layers=layer, num_layers=32, ...)
        >>>
        >>> # Fuse a cross attention layer to each self attention layer to adapt for the encoder
        >>> fusion_layer = nn.TransformerCrossAttentionLayer(...)
        >>> fused_layer = FusionLayer(layer, fusion_layer)
        >>> model = TransformerDecoder(layers=fused_layer, num_layers=32, ...)
        >>>
        >>> # Original decoder state_dict still works
        >>> model.load_state_dict(..., strict=False)

    Args:
        layer (nn.Module): original decoder layer
        fusion_layer (nn.Module): new fusion layer
        fusion_first (bool): boolean to insert fusion layer before or after the decoder layer.
    """

    def __init__(self, layer: 'nn.Module', fusion_layer: 'nn.Module', fusion_first: 'bool'=True):
        super().__init__()
        self.layer = layer
        self.fusion_layer = fusion_layer
        self.fusion_first = fusion_first
        self._register_state_dict_hook(FusionLayer._state_dict_hook)
        self._register_load_state_dict_pre_hook(FusionLayer._load_state_dict_hook, with_module=True)

    def _state_dict_hook(self, state_dict, prefix, *args, **kwargs):
        """Remove "layer" from the original layer in the state_dict
        name. This keeps the orginal state dict name for the layer
        from before fusing with the FusionLayer.

        [!Note] This update changes the order of the OrderedDict
        """
        keys = list(state_dict.keys())
        for key in keys:
            local_key = key[len(prefix):]
            if local_key.startswith('layer'):
                new_key = prefix + local_key.replace('layer.', '')
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

    def _load_state_dict_hook(self, state_dict, prefix, *args, **kwargs):
        """Apply extra "layer" prefix to the state_dict key to
        account for the FusionLayer wrapping.
        """
        keys = list(state_dict.keys())
        for key in keys:
            local_key = key[len(prefix):]
            if not local_key.startswith('fusion_layer'):
                new_key = prefix + 'layer.' + local_key
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

    def setup_caches(self, batch_size: 'int', dtype: 'torch.dtype', *, encoder_max_seq_len: int, decoder_max_seq_len: int) ->None:
        """Setup key value cache for both layers.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            encoder_max_seq_len (int): maximum cache sequence length for cross-attention layer.
            decoder_max_seq_len (int): maximum cache sequence length for self-attention layer.
        """
        self.layer.setup_caches(batch_size, dtype, encoder_max_seq_len=encoder_max_seq_len, decoder_max_seq_len=decoder_max_seq_len)
        self.fusion_layer.setup_caches(batch_size, dtype, encoder_max_seq_len=encoder_max_seq_len, decoder_max_seq_len=decoder_max_seq_len)

    def caches_are_setup(self) ->bool:
        """
        Check if the key value caches are setup on ``self.layer``.
        See :func:~torchtune.modules.TransformerDecoder.caches_are_setup`.
        """
        return self.layer.caches_are_setup()

    def caches_are_enabled(self) ->bool:
        """
        Checks if the key value caches on ``self.layer`` are enabled.
        See :func:~torchtune.modules.TransformerDecoder.caches_are_enabled`.
        """
        return self.layer.caches_are_enabled()

    def reset_cache(self):
        """Reset both layers' key value caches."""
        self.layer.reset_cache()
        self.fusion_layer.reset_cache()

    def fusion_params(self) ->List[str]:
        """
        Return parameters of fusion layer.
        """
        fusion_params = [f'fusion_layer.{k}' for k, v in self.fusion_layer.named_parameters()]
        return fusion_params

    def forward(self, x: 'torch.Tensor', **kwargs: Dict) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            **kwargs (Dict): all additional layer args

        Returns:
            Tensor: output tensor with same shape as input
                [batch_size x seq_length x embed_dim]`

        """
        if self.fusion_first:
            x = self.fusion_layer(x, **kwargs)
            x = self.layer(x, **kwargs)
        else:
            x = self.layer(x, **kwargs)
            x = self.fusion_layer(x, **kwargs)
        return x


class FusionEmbedding(nn.Module):
    """Fusion embedding supports training additional special tokens while keeping
    the original embedding frozen. When fusing new models with a language model,
    there may be some additional tokens needed to support the fused language model. For
    example, adding a vision encoder might necessitate additional tokens like ``<|image|>``
    to indicate an images position in text and require learning an embedding for this token.
    The FusionEmbedding keeps the original embeddings frozen while learning a much smaller
    second embedding for the additional tokens. During forward this module routes
    the tokens to the appropriate embedding table.

    Use this as a drop-in replacement for :class:`torch.nn.Embedding` in your model.

    Example:
        >>> embedding = FusionEmbedding(vocab_size=100, fusion_vocab_size=10, embed_dim=128)
        >>> model = TransformerDecoder(tok_embeddings=embedding, ...)
        >>>
        >>> # Original model state_dict still works
        >>> model.load_state_dict(..., strict=False)

    .. note::
        This module assumes all tokens in the range [0, vocab_size) are part of the
        original embedding table and all new tokens in the range
        [vocab_size, vocab_size + fusion_vocab_size)

    Args:
        vocab_size (int): language model vocab size
        fusion_vocab_size (int): additional tokens for the fused model
        embed_dim (int): embedding dimension of the two embedding tables
    """

    def __init__(self, vocab_size: 'int', fusion_vocab_size: 'int', embed_dim: 'int') ->None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fusion_embedding = nn.Embedding(fusion_vocab_size, embed_dim)
        self.dim = embed_dim
        self.num_embeddings = vocab_size + fusion_vocab_size
        self._register_state_dict_hook(FusionEmbedding._state_dict_hook)
        self._register_load_state_dict_pre_hook(FusionEmbedding._load_state_dict_hook, with_module=True)

    def _state_dict_hook(self, destination, prefix, keep_vars):
        """Remove "embedding" from the original embedding in the state_dict
        name. This keeps the orginal state dict name for the embedding
        from before fusing with the FusionEmbedding.

        [!Note] This update changes the order of the OrderedDict
        """
        key = prefix + 'embedding.weight'
        new_key = prefix + 'weight'
        destination[new_key] = destination[key]
        del destination[key]

    def _load_state_dict_hook(self, state_dict, prefix, *args, **kwargs):
        """Apply extra "embedding" prefix to the state_dict key to
        account for the FusionEmbedding wrapping.
        """
        if state_dict:
            key = prefix + 'weight'
            new_key = prefix + 'embedding.weight'
            state_dict[new_key] = state_dict[key]
            del state_dict[key]

    def fusion_params(self) ->List[str]:
        """
        Return fusion embedding parameters.
        """
        fusion_params = ['fusion_embedding.weight']
        return fusion_params

    def _fused_embed(self, bs, seq_len):
        """
        Return an empty tensor the shape of the combined embedding.
        """
        device = self.embedding.weight.device
        dtype = self.embedding.weight.dtype
        return torch.empty(bs, seq_len, self.dim, device=device, dtype=dtype)

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            input (torch.Tensor): input integer tensor with shape
                [batch_size x seq_length]

        Returns:
            Tensor: output tensor embedding with shape
                [batch_size x seq_length x embed_dim]`

        """
        bs, seq_len = input.size()
        vocab_size = self.embedding.num_embeddings
        mask = input < vocab_size
        tokens = torch.masked_select(input, mask)
        fusion_tokens = torch.masked_select(input, ~mask) - vocab_size
        embeds = self.embedding(tokens)
        fusion_embeds = self.fusion_embedding(fusion_tokens)
        out = self._fused_embed(bs, seq_len)
        mask = mask.unsqueeze(-1).expand(bs, seq_len, self.dim)
        out = out.masked_scatter(mask, embeds)
        out = out.masked_scatter(~mask, fusion_embeds)
        return out


def get_fusion_params(model: 'nn.Module') ->Dict[str, nn.Parameter]:
    """
    Return the subset of parameters from a model that correspond to fused
    modules. Assumes that any fusion class has defined the
    :func:`~torchtune.modules.model_fusion.FusionLayer.fusion_params` method.

    Args:
        model (nn.Module): Instance of model class containing some
            fusion params.

    Returns:
        Dict[str, nn.Parameter]: the subset of model's state dict containing
            only adapter parameters.

    """
    fusion_params = {}
    for k, v in model.named_modules():
        if hasattr(v, 'fusion_params') and callable(v.fusion_params):
            current_fusion_params = v.fusion_params()
            for n, p in v.named_parameters(recurse=True):
                if n in current_fusion_params:
                    full_key = f'{k}.{n}' if k else n
                    fusion_params.update({full_key: p})
                    current_fusion_params.remove(n)
            assert current_fusion_params == [], f'Fusion params {current_adapter_params} not converted'
    return fusion_params


def set_trainable_params(model: 'nn.Module', adapter_params: 'Dict[str, Any]') ->None:
    """
    Set trainable parameters for an nn.Module based on a state dict of adapter parameters.

    Args:
        model (nn.Module): Instance of model class containing some adapter params.
        adapter_params (Dict[str, Any]): State dict mapping adapter key names to their
            respective nn.Parameters (i.e. outputs of :func:`~torchtune.modules.peft.get_adapter_params`.)

    Returns:
        None
    """
    for k, v in model.named_parameters():
        v.requires_grad_(k in adapter_params)


class DeepFusionModel(nn.Module):
    """DeepFusion is a type of fused model architecture where a pretrained encoder is combined
    with a pretrained decoder (LLM). This is a popular architecture for multimodal models, with
    a full overview available in `The Evolution of Multimodal Model Architectures <https://arxiv.org/abs/2405.17927>`_.

    This module has the same methods and forward signature as :class:`~torchtune.modules.TransformerDecoder` and can be used
    interchangeably where :class:`~torchtune.modules.TransformerDecoder` is. It combines the encoder with the decoder as a
    single module for checkpointing and finetuning. It is expected that the encoder and decoder
    are already defined with any extra learnable ``fusion_params``: learnable parameters to help
    adapt the pre-trained encoder to the pre-trained decoder.

    Example:
        >>> # decoder is a TransformerDecoder (e.g. llama3_8b) with fused cross attention layers
        >>> embed = FusionEmbedding(...)
        >>> layer = FusionLayer(
        ...     layer=TransformerSelfAttentionLayer(...),
        ...     fusion_layer=TransformerCrossAttentionLayer(...),
        ... )
        >>> decoder = TransformerDecoder(tok_embeddings=embed, layers=layer, num_layers=32, ...)
        >>>
        >>> # encoder is pre-trained encoder (e.g. clip_vit_224) with an added projection head
        >>> projection_head = FeedForward(...)
        >>> register_fusion_module(projection_head))
        >>> encoder = nn.Sequential(clip_vit_224(), projection_head)
        >>>
        >>> # DeepFusionModel combines the encoder and decoder
        >>> model = DeepFusionModel(decoder, encoder)
        >>>
        >>> # Load full fused checkpoints (e.g. a Flamingo checkpoint)
        >>> model.load_state_dict(...)
        >>>
        >>> # Or load pretrained individual models (fusion_params are not loaded)
        >>> model.encoder.load_state_dict(..., strict=False)
        >>> model.decoder.load_state_dict(..., strict=False)
        >>>
        >>> # Forward pass
        >>> output = model(tokens, mask, encoder_input, encoder_mask, input_pos)

    Args:
        decoder (TransformerDecoder): decoder module
        encoder (nn.Module): encoder module
        decoder_trainable (bool): whether to train or freeze the decoder. Default is False.
        encoder_trainable (bool): whether to train or freeze the encoder. Default is False.
        fusion_trainable (bool): whether to train the fusion parameters. Default is True.

    """

    def __init__(self, decoder: 'TransformerDecoder', encoder: 'nn.Module', *, decoder_trainable: bool=False, encoder_trainable: bool=False, fusion_trainable: bool=True):
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder
        trainable_params = set()
        if encoder_trainable:
            trainable_params |= {f'encoder.{n}' for n, p in self.encoder.named_parameters()}
        if decoder_trainable:
            trainable_params |= {f'decoder.{n}' for n, p in self.decoder.named_parameters()}
        if fusion_trainable:
            trainable_params |= set(get_fusion_params(self))
        else:
            trainable_params -= set(get_fusion_params(self))
        set_trainable_params(self, trainable_params)

    def set_num_output_chunks(self, num_output_chunks: 'int') ->None:
        """Used to save memory in combination with :class:`~torchtune.modules.loss.CEWithChunkedOutputLoss`.
        This should be called before the first forward pass, in the recipe."""
        self.decoder.set_num_output_chunks(num_output_chunks)

    def setup_caches(self, batch_size: 'int', dtype: 'torch.dtype', *, encoder_max_seq_len: int=None, decoder_max_seq_len: int=None):
        """
        Sets up key-value attention caches for inference for ``self.decoder``.
        For each layer in ``self.decoder.layers``:
        - :class:`torchtune.modules.TransformerSelfAttentionLayer` will use ``decoder_max_seq_len``.
        - :class:`torchtune.modules.TransformerCrossAttentionLayer` will use ``encoder_max_seq_len``.
        - :class:`torchtune.modules.fusion.FusionLayer` will use both ``decoder_max_seq_len`` and ``encoder_max_seq_len``.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            encoder_max_seq_len (int): maximum encoder cache sequence length.
            decoder_max_seq_len (int): maximum decoder cache sequence length.
        """
        self.decoder.setup_caches(batch_size, dtype, encoder_max_seq_len=encoder_max_seq_len, decoder_max_seq_len=decoder_max_seq_len)

    def caches_are_setup(self) ->bool:
        """
        Check if the key value caches are setup. This means ``setup_caches`` has been called, and
        the relevant attention modules in the model have created their ``KVCache``.
        """
        return self.decoder.caches_are_setup()

    def caches_are_enabled(self) ->bool:
        """
        Checks if the key value caches are enabled. Once KV-caches have been setup, the relevant
        attention modules will be "enabled" and all forward passes will update the caches. This behaviour
        can be disabled without altering the state of the KV-caches by "disabling" the KV-caches
        using :func:`~torchtune.modules.common_utils.disable_kv_cache`, upon which ``caches_are_enabled`` would return False.
        """
        return self.decoder.caches_are_enabled()

    def reset_caches(self):
        """
        Resets KV-cache buffers on relevant attention modules to zero, and reset cache positions to zero,
        without deleting or reallocating cache tensors.
        """
        self.decoder.reset_caches()

    def forward(self, tokens: 'torch.Tensor', *, mask: Optional[torch.Tensor]=None, encoder_input: Optional[Dict]=None, encoder_mask: Optional[torch.Tensor]=None, input_pos: Optional[torch.Tensor]=None) ->Union[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            tokens (torch.Tensor): input tensor with shape ``[b x s]``
            mask (Optional[torch.Tensor]): Optional boolean tensor which contains the attention mask
                with shape ``[b x s x s]``. This is applied after the query-key multiplication and
                before the softmax. A value of True in row i and column j means token i attends
                to token j. A value of False means token i does not attend to token j. If no
                mask is specified, a causal mask is used by default. Default is None.
            encoder_input (Optional[Dict]): Optional input for the encoder.
            encoder_mask (Optional[torch.Tensor]):  Boolean tensor defining a relational matrix between
                tokens and encoder embeddings. A True value at position i,j means token i can attend
                to embedding j in the decoder. Mask has shape ``[b x s x s_e]``. Default is None.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape ``[b x s]``.
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Note: At the very first step of inference, when the model is provided with a prompt,
        ``input_pos`` would contain the positions of all of the tokens in the prompt
        (eg: ``torch.arange(prompt_length)``). This is because we will need to compute the
        KV values for each position.

        Returns:
            Tensor: output tensor with shape ``[b x s x v]`` or a list of layer                 output tensors defined by ``output_hidden_states`` with the                 final output tensor appended to the list.

        Notation used for tensor shapes:
            - b: batch size
            - s: token sequence length
            - s_e: encoder sequence length
            - v: vocab size
            - d: token embed dim
            - d_e: encoder embed dim
            - m_s: max seq len
        """
        encoder_embed = None
        if encoder_input is not None:
            encoder_embed = self.encoder(**encoder_input)
        output = self.decoder(tokens=tokens, mask=mask, encoder_input=encoder_embed, encoder_mask=encoder_mask, input_pos=input_pos)
        return output


def _lora_a_init_params(x: 'nn.Linear') ->None:
    """
    Initialize LoRA A weight to Kaiming uniform.
    """
    nn.init.kaiming_uniform_(x.weight, a=math.sqrt(5))


def _lora_b_init_params(x: 'nn.Linear') ->None:
    """
    Initialize LoRA B weight to zeros.
    """
    nn.init.zeros_(x.weight)


class DoRALinear(nn.Module, AdapterModule):
    """DoRA linear layer as introduced in
    `DoRA: Weight-Decomposed Low-Rank Adaptation of Large Language Models <https://arxiv.org/abs/2402.09353>`_.

    DoRA (Weight-Decomposed Low-Rank Adaptation) fine-tunes a layer by decomposing the pre-trained weights
    into two components: magnitude and direction. The magnitude component is a learnable scalar vector
    that scales each output channel, while the direction component, modified via LoRA, adjusts the orientation
    of weights. By scaling the LoRA update component :math:`BAx` with the `magnitude` vector, DoRA allows the model
    to apply distinct scaling adjustments across different output dimensions.

    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        rank (int): rank of the low-rank approximation
        alpha (float): scaling factor for the low-rank approximation
        dropout (float): dropout probability. Default: 0.0
        use_bias (bool): whether to include bias in the original linear layer.
            Default: False
        quantize_base (bool): Whether to quantize base linear weight or not.
            Default: False

    """

    def __init__(self, in_dim: 'int', out_dim: 'int', rank: 'int', alpha: 'float', dropout: 'float'=0.0, use_bias: 'bool'=False, quantize_base: 'bool'=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.scaling = alpha / rank
        self.use_bias = use_bias
        self._quantize_base = quantize_base
        weight, bias = self._create_weight_and_bias()
        self.register_parameter('weight', nn.Parameter(weight))
        self.register_parameter('bias', nn.Parameter(bias) if bias is not None else None)
        self.disabled = False
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.lora_a = nn.Linear(in_features=in_dim, out_features=rank, bias=False)
        self.lora_b = nn.Linear(in_features=rank, out_features=out_dim, bias=False)
        self.magnitude = nn.Parameter(torch.empty(out_dim))
        self.initialize_parameters()

    def initialize_parameters(self):
        _lora_a_init_params(self.lora_a)
        _lora_b_init_params(self.lora_b)

    @torch.no_grad()
    def initialize_dora_magnitude(self):
        """
        DoRA initializes the magnitude vector such that its outputs are initially
        identical to standard LoRA's outputs.
        """
        base_weight = self.weight
        lora_weight = self.lora_b.weight @ self.lora_a.weight
        weight_norm = self._get_weight_norm(base_weight, lora_weight)
        self.magnitude.copy_(weight_norm)

    def _create_weight_and_bias(self):
        """
        Creates a linear weight and bias tensor, using NF4 dtype if we're quantizing
        (indicated via quantize_base=True).
        """
        in_dim, out_dim, use_bias = self.in_dim, self.out_dim, self.use_bias
        linear = nn.Linear(in_features=in_dim, out_features=out_dim, bias=use_bias)
        weight = linear.weight if not self._quantize_base else to_nf4(linear.weight)
        bias = None
        if self.use_bias:
            bias = linear.bias
        return weight, bias

    def _get_weight_norm(self, weight, lora_weight):
        weight = weight + self.scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1)
        return weight_norm

    def adapter_params(self) ->List[str]:
        """
        Return lora_a.weight and lora_b.weight as adapter params.
        If bias is enabled, also return lora_a.bias and lora_b.bias.
        """
        adapter_params = ['lora_a.weight', 'lora_b.weight', 'magnitude']
        return adapter_params

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape ``(..., in_dim)``

        Returns:
            Tensor: output tensor with shape ``(..., out_dim)``
        """
        if self._quantize_base:
            base_out = linear_nf4(input=x, weight=self.weight)
            if self.use_bias:
                base_out = base_out + self.bias
        else:
            base_out = F.linear(x, self.weight, self.bias)
        if self.disabled:
            return base_out
        x = self.dropout(x)
        lora_out = self.lora_b(self.lora_a(x))
        x_eye = torch.eye(self.lora_a.weight.shape[1], device=self.lora_a.weight.device, dtype=x.dtype)
        lora_weight = self.lora_b(self.lora_a(x_eye)).T
        magnitude = self.magnitude
        weight = self.weight
        weight_norm = self._get_weight_norm(weight, lora_weight.detach())
        weight_norm = weight_norm.detach()
        mag_norm_scale = (magnitude / weight_norm).view(1, -1)
        dora_out = (mag_norm_scale - 1) * base_out + mag_norm_scale * lora_out * self.scaling
        return dora_out + base_out


class LoRALinear(nn.Module, AdapterModule):
    """LoRA linear layer as introduced in `LoRA: Low-Rank Adaptation of Large Language Models <https://arxiv.org/abs/2106.09685>`_.

    LoRA perturbs a given layer via a low-rank approximation where only
    the rank decomposition matrices are trainable. In a linear layer instead of
    :math:`x \\mapsto W_0x` a LoRALinear layer is defined as
    :math:`x \\mapsto W_0x + (\\alpha / r)BAx`, where :math:`r` is the rank of
    the matrices :math:`A` and :math:`B` and :math:`\\alpha` is a scaling factor.
    As in the original implementation, we support dropout before multiplication
    by the low-rank matrices.

    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        rank (int): rank of the low-rank approximation
        alpha (float): scaling factor for the low-rank approximation
        dropout (float): dropout probability. Default: 0.0
        use_bias (bool): whether to include bias in the original linear layer.
            Default: False
        quantize_base (bool): Whether to quantize base linear weight or not.
            Default: False
    """

    def __init__(self, in_dim: 'int', out_dim: 'int', rank: 'int', alpha: 'float', dropout: 'float'=0.0, use_bias: 'bool'=False, quantize_base: 'bool'=False):
        super().__init__()
        self.in_dim = in_dim
        self.rank = rank
        self.alpha = alpha
        self.out_dim = out_dim
        self.use_bias = use_bias
        self._quantize_base = quantize_base
        weight, bias = self._create_weight_and_bias()
        self.disabled = False
        self.register_parameter('weight', nn.Parameter(weight))
        self.register_parameter('bias', nn.Parameter(bias) if bias is not None else None)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.lora_a = nn.Linear(in_features=in_dim, out_features=rank, bias=False)
        self.lora_b = nn.Linear(in_features=rank, out_features=out_dim, bias=False)
        self.merged = False
        self.initialize_parameters()

    def initialize_parameters(self):
        _lora_a_init_params(self.lora_a)
        _lora_b_init_params(self.lora_b)

    def _create_weight_and_bias(self):
        """
        Creates a linear weight and bias tensor, using NF4 dtype if we're quantizing
        (indicated via quantize_base=True).
        """
        in_dim, out_dim, use_bias = self.in_dim, self.out_dim, self.use_bias
        linear = nn.Linear(in_features=in_dim, out_features=out_dim, bias=use_bias)
        weight = linear.weight if not self._quantize_base else to_nf4(linear.weight)
        bias = None
        if self.use_bias:
            bias = linear.bias
        return weight, bias

    def adapter_params(self) ->List[str]:
        """
        Return lora_a.weight and lora_b.weight as adapter params.
        If bias is enabled, also return lora_a.bias and lora_b.bias.
        """
        adapter_params = ['lora_a.weight', 'lora_b.weight']
        return adapter_params

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape ``(..., in_dim)``

        Returns:
            torch.Tensor: output tensor with shape ``(..., out_dim)``

        """
        if self._quantize_base:
            out = linear_nf4(input=x, weight=self.weight)
            if self.use_bias:
                out = out + self.bias
        else:
            out = F.linear(x, self.weight, self.bias)
        if self.disabled:
            return out
        lora_out = self.lora_a(self.dropout(x))
        lora_out = self.alpha / self.rank * self.lora_b(lora_out)
        return out + lora_out


class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(self, dim: 'int', max_seq_len: 'int'=4096, base: 'int'=10000) ->None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def reset_parameters(self):
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / self.base ** (torch.arange(0, self.dim, 2)[:self.dim // 2].float() / self.dim)
        self.register_buffer('theta', theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: 'int'=4096) ->None:
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)
        idx_theta = torch.einsum('i, j -> ij', seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer('cache', cache, persistent=False)

    def forward(self, x: 'torch.Tensor', *, input_pos: Optional[torch.Tensor]=None) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                ``[b, s, n_h, h_d]``
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        seq_len = x.size(1)
        rope_cache = self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
        x_out = torch.stack([xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1], xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1]], -1)
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


class TanhGate(nn.Module):
    """Implements a basic learnable gate to scale layer outputs"""

    def __init__(self) ->None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor to gate

        Returns:
            torch.Tensor: The output tensor after gating. Has the same shape as ``x``.
        """
        return x * self.scale.tanh()


class Linear(nn.Module):
    """
    nn.Module used in :func:`~torchtune.modules.tied_linear.TiedLinear`, added to work with the hooks
    :class:`~torchtune.training._activation_offloading.NoOpManager` that ignore activation
    offloading context manager.

    Without this class, we can't add NoOp hooks, and we will offload the activation of
    the tied linear layer, which is slow.

    For more information, see how NoOpManager is called in the recipes.
    """

    def forward(self, x: 'torch.Tensor', weight: 'torch.Tensor'):
        return F.linear(x, weight)


class TransformerCrossAttentionLayer(nn.Module):
    """
    Cross attention Transformer layer following the same conventions as the TransformerSelfAttentionLayer.
    Normalization is applied before the attention **and** FF layer.

    Args:
        attn (MultiHeadAttention): Attention module.
        mlp (nn.Module): Feed-forward module.
        ca_norm (Optional[nn.Module]): Normalization to be applied before cross-attention.
        mlp_norm (Optional[nn.Module]): Normalization to be applied before the feed-forward layer.
        ca_scale (Optional[nn.Module]): Module to scale cross-attention output.
        mlp_scale (Optional[nn.Module]): Module to scale the feed-forward output.

    Raises:
        AssertionError: if attn.pos_embeddings is set.
    """

    def __init__(self, attn: 'MultiHeadAttention', mlp: 'nn.Module', *, ca_norm: Optional[nn.Module]=None, mlp_norm: Optional[nn.Module]=None, ca_scale: Optional[nn.Module]=None, mlp_scale: Optional[nn.Module]=None) ->None:
        super().__init__()
        if attn.pos_embeddings is not None:
            raise AssertionError("Doesn't support positional embeddings for cross attention,                 because q and k are different sequences.")
        self.attn = attn
        self.mlp = mlp
        self.ca_norm = ca_norm or nn.Identity()
        self.mlp_norm = mlp_norm or nn.Identity()
        self.ca_scale = ca_scale or nn.Identity()
        self.mlp_scale = mlp_scale or nn.Identity()

    def setup_caches(self, batch_size: 'int', dtype: 'torch.dtype', *, encoder_max_seq_len: int, decoder_max_seq_len: int) ->None:
        """Setup key value caches for attention calculation.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            encoder_max_seq_len (int): maximum cache sequence length.
            decoder_max_seq_len (int): this parameter is ignored in this layer.
        """
        self.attn.setup_cache(batch_size, dtype, encoder_max_seq_len)

    def caches_are_setup(self) ->bool:
        """
        Check if the key value caches are setup on ``self.attn``.
        See :func:~torchtune.modules.TransformerDecoder.caches_are_setup`.
        """
        return self.attn.kv_cache is not None

    def caches_are_enabled(self) ->bool:
        """
        Checks if the key value caches on ``self.attn`` are enabled.
        See :func:~torchtune.modules.TransformerDecoder.caches_are_enabled`.
        """
        return self.attn.cache_enabled

    def reset_cache(self):
        """Reset the key value caches."""
        self.attn.reset_cache()

    def _skip_mask(self, mask: 'Optional[torch.Tensor]') ->Optional[torch.Tensor]:
        """Some tokens in x may not attend to any encoder inputs
        due to the cross attention mask (encoder_mask). This results in
        a full row of the attention matrix being masked out.

        In the example below, the word "the" is masked from every embedding.
        The False value means a token can't attend to an embedding.

        .. code-block:: text

            |emb||emb||emb|
        |The| F    F    F
        |red| T    F    T
        |car| F    T    T

        This results in no inputs into the softmax layer which causes a NaN.
        The skip mask is used to mask the outputs of attention and
        mlp resulting in the token being skipped.

        The above example would result in a skip mask of: [[True], [False], [False]]
        which specifies which tokens to fully mask out.

        """
        if mask is None:
            return None
        if mask.dtype == torch.bool:
            mask = ~mask
        else:
            mask = torch.isneginf(mask)
        mask = torch.all(mask, dim=-1, keepdim=True)
        return mask

    def forward(self, x: 'torch.Tensor', *, encoder_input: Optional[torch.Tensor]=None, encoder_mask: Optional[torch.Tensor]=None, **kwargs: Dict) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            encoder_input (Optional[torch.Tensor]): Optional input embeds from the encoder. Shape
                [batch_size x token_sequence x embed_dim]
            encoder_mask (Optional[torch.Tensor]):  Boolean tensor defining a relational matrix between
                tokens and encoder embeddings. A True value at position i,j means token i can attend
                to embedding j in the decoder. Mask has shape [batch_size x token_sequence x embed_sequence].
                Default is None.
            **kwargs (Dict): transformer layer inputs not relevant to self attention.

        Returns:
            torch.Tensor: output tensor with same shape as input
                [batch_size x seq_length x embed_dim]
        """
        empty_cache = not self.caches_are_enabled() or self.attn.kv_cache.size == 0
        if encoder_input is None and empty_cache:
            return x
        skip_mask = self._skip_mask(encoder_mask)
        if encoder_mask is not None:
            encoder_mask = encoder_mask.masked_fill(skip_mask, True)
        attn_out = self.attn(self.ca_norm(x), encoder_input, mask=encoder_mask)
        if skip_mask is not None:
            attn_out = attn_out.masked_fill(skip_mask, 0)
        h = self.ca_scale(attn_out) + x
        mlp_out = self.mlp(self.mlp_norm(h))
        if skip_mask is not None:
            mlp_out = mlp_out.masked_fill(skip_mask, 0)
        out = h + self.mlp_scale(mlp_out)
        return out


def _get_clones(module: 'nn.Module', n: 'int') ->nn.ModuleList:
    """
    Return a list of ``n`` identical layers.

    Args:
        module (nn.Module): module to be cloned
        n (int): number of clones

    Returns:
        nn.ModuleList: list of ``n`` identical layers
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


T = TypeVar('T', bound=type)


def deprecated(msg: 'str'='') ->Callable[[T], T]:
    """
    Decorator to mark an object as deprecated and print additional message.

    Args:
        msg (str): additional information to print after warning.

    Returns:
        Callable[[T], T]: the decorated object.
    """

    @lru_cache(maxsize=1)
    def warn(obj):
        warnings.warn(f'{obj.__name__} is deprecated and will be removed in future versions. ' + msg, category=FutureWarning, stacklevel=3)

    def decorator(obj):

        @wraps(obj)
        def wrapper(*args, **kwargs):
            warn(obj)
            return obj(*args, **kwargs)
        return wrapper
    return decorator


class CLSEmbedding(nn.Module):
    """
    Adds a CLS token to every tile in an image.

    Notice that tile is different from patch (token). An image is divided into tiles during pre-processing,
    and patches are the outcome of the convolution in the ViT applied to each tile.

    Args:
        embed_dim (int): The dimensionality of the input patch embedding.
    """

    def __init__(self, embed_dim: 'int') ->None:
        super().__init__()
        scale = embed_dim ** -0.5
        self.weight = nn.Parameter(scale * torch.randn(embed_dim))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        bsz_and_n_imgs, n_tiles, n_tokens, embed_dim = x.shape
        cls_emb = self.weight.broadcast_to(bsz_and_n_imgs, n_tiles, 1, embed_dim)
        return torch.cat([cls_emb, x], dim=2)


class VisionTransformer(nn.Module):
    """
    Implementation of the ViT architecture (https://arxiv.org/abs/2010.11929),
    with support for tile-cropped images, outputting of hidden layers and optional CLS projection.

    ViT is a transformer architecture that takes in images and outputs N embedded tokens that
    represent this image. Each image is divided into **patches** by a convolution.
    These patches are flattened and subsequently treated as **tokens** by the transformer.

    To further enhance the performance of ViT and avoid downscaling images, we support tile-cropped images,
    which are images divided into **tiles** during the preprocessing stage. For example, instead of
    downscaling an 800x400 image to fit 400x400, we may crop it into two 400x400 tiles,
    if the ``tile_size=400``. For details on preprocessing, please refer to
    :class:`torchtune.models.clip._transforms.CLIPImageTransform`.

    Each of these tiles is further broken down into patches by a convolution operation. For example, if
    your ``patch_size=40``, then each (400, 400) tile will become a grid of 10x10 patches, and your whole image will have
    num_tiles * n_tokens -> num_tiles * (10x10 patches + 1 CLS token) -> num_tiles * 101.

    Before the transformer layers, a CLS token is added to each tile as the first token.
    In transformers, a token called CLS is a special token that is added to the beginning of each sequence.
    This token can be used to represent the whole input, instead of using a pooling operation, for example.

    To help the model "see" the whole image, we use positional embeddings. If your image
    was tile-cropped, then you need to use tile positional embeddings:

    - token_pos_embedding (tiled): :class:`torchtune.models.clip._position_embeddings.TiledTokenPositionalEmbedding`
    - pre_tile_pos_embed: :class:`torchtune.models.clip._position_embeddings.TilePositionalEmbedding`
    - post_tile_pos_embed: :class:`torchtune.models.clip._position_embeddings.TilePositionalEmbedding`

    Otherwise, pre and post tile_pos_embed should be None and all you need is a simple
    token positional embedding:

    - token_pos_embedding (not tiled): :class:`torchtune.models.clip._position_embeddings.TokenPositionalEmbedding`

    All images will be considered as a stack of tiles, even if your image was not tile-cropped. In such cases,
    your image would be composed of a single tile.

    In summary:

    1) An image is broken down into tiles during preprocessing.
    2) In the ViT, the tiles will be broken down into patches.
    3) The patches will be flattened and transformed. We call them tokens, because that's how the transformer sees them.


    Image: shape (8x8)

    .. code-block:: text

        |  1 |  2 |  3 |  4 |  5 |  6 |  7 |  8 |
        |  9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
        | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 |
        | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 |
        | 33 | 34 | 35 | 36 | 37 | 38 | 39 | 40 |
        | 41 | 42 | 43 | 44 | 45 | 46 | 47 | 48 |
        | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 |
        | 57 | 58 | 59 | 60 | 61 | 62 | 63 | 64 |

    Tiles: shape (4,4,4) # (num_tiles, tile_size, tile_size)

    .. code-block:: text

        |  1 |  2 |  3 |  4 |    |  5 |  6 |  7 |  8 |
        |  9 | 10 | 11 | 12 |    | 13 | 14 | 15 | 16 |
        | 17 | 18 | 19 | 20 |    | 21 | 22 | 23 | 24 |
        | 25 | 26 | 27 | 28 |    | 29 | 30 | 31 | 32 |

        | 33 | 34 | 35 | 36 |    | 37 | 38 | 39 | 40 |
        | 41 | 42 | 43 | 44 |    | 45 | 46 | 47 | 48 |
        | 49 | 50 | 51 | 52 |    | 53 | 54 | 55 | 56 |
        | 57 | 58 | 59 | 60 |    | 61 | 62 | 63 | 64 |

    Patches: shape (4,4,2,2) # (num_tiles, num_patches_per_tile, patch_size, patch_size)

    .. code-block:: text

        |  1 |  2 |    |  3 |  4 |    |  5 |  6 |    |  7 |  8 |
        |  9 | 10 |    | 11 | 12 |    | 13 | 14 |    | 15 | 16 |

        | 17 | 18 |    | 19 | 20 |    | 21 | 22 |    | 23 | 24 |
        | 25 | 26 |    | 27 | 28 |    | 29 | 30 |    | 31 | 32 |

        | 33 | 34 |    | 35 | 36 |    | 37 | 38 |    | 39 | 40 |
        | 41 | 42 |    | 43 | 44 |    | 45 | 46 |    | 47 | 48 |

        | 49 | 50 |    | 51 | 52 |    | 53 | 54 |    | 55 | 56 |
        | 57 | 58 |    | 59 | 60 |    | 61 | 62 |    | 63 | 64 |

    token: shape (4, 4, 4) # (num_tiles, num_patches_per_tile, emb_dim)

    .. code-block:: text

        |  1 |  2 |  9 |  10 |    |  3 |  4 |  11 |  12 |    |  17 |  18 |  25 |  26 |    | 19 | 20 |  27 |  28 |
        | ... continuation of data ...
        | ... continuation of data ...
        | 37 | 38 | 45 |  46 |    | 39 |  40 | 47 |  48 |    | 53 | 54 |  61 |  62 |    | 55 | 56 |  63 |  64 |

    For the positional embeddings:

    Same for every tile, different for every token.

    - :class:`torchtune.models.clip._position_embeddings.TokenPositionalEmbedding`
    - :class:`torchtune.models.clip._position_embeddings.TiledTokenPositionalEmbedding`

    .. code-block:: text

        |  1 |  2 |  3 |  4 |    |  1 |  2 |  3 |  4 |
        |  9 | 10 | 11 | 12 |    |  9 | 10 | 11 | 12 |
        | 17 | 18 | 19 | 20 |    | 17 | 18 | 19 | 20 |
        | 25 | 26 | 27 | 28 |    | 25 | 26 | 27 | 28 |

        |  1 |  2 |  3 |  4 |    |  1 |  2 |  3 |  4 |
        |  9 | 10 | 11 | 12 |    |  9 | 10 | 11 | 12 |
        | 17 | 18 | 19 | 20 |    | 17 | 18 | 19 | 20 |
        | 25 | 26 | 27 | 28 |    | 25 | 26 | 27 | 28 |

    Different for every tile, different for every token.

    - :class:`torchtune.models.clip._position_embeddings.TiledTokenPositionalEmbedding`

    .. code-block:: text

        |  1 |  2 |    |  3 |  4 |    |  5 |  6 |    |  7 |  8 |
        |  9 | 10 |    | 11 | 12 |    | 13 | 14 |    | 15 | 16 |

        | 17 | 18 |    | 19 | 20 |    | 21 | 22 |    | 23 | 24 |
        | 25 | 26 |    | 27 | 28 |    | 29 | 30 |    | 31 | 32 |

        | 33 | 34 |    | 35 | 36 |    | 37 | 38 |    | 39 | 40 |
        | 41 | 42 |    | 43 | 44 |    | 45 | 46 |    | 47 | 48 |

        | 49 | 50 |    | 51 | 52 |    | 53 | 54 |    | 55 | 56 |
        | 57 | 58 |    | 59 | 60 |    | 61 | 62 |    | 63 | 64 |

    different for every tile, same for every token within a tile.

    - :class:`torchtune.models.clip._position_embeddings.TilePositionalEmbedding`

    .. code-block:: text

        |  1 |  1 |  1 |  1 |    |  2 |  2 |  2 |  3 |
        |  1 |  1 |  1 |  1 |    |  2 |  2 |  2 |  3 |
        |  1 |  1 |  1 |  1 |    |  2 |  2 |  2 |  3 |
        |  1 |  1 |  1 |  1 |    |  2 |  2 |  2 |  3 |

        |  3 |  3 |  3 |  3 |    |  4 |  4 |  4 |  4 |
        |  3 |  3 |  3 |  3 |    |  4 |  4 |  4 |  4 |
        |  3 |  3 |  3 |  3 |    |  4 |  4 |  4 |  4 |
        |  3 |  3 |  3 |  3 |    |  4 |  4 |  4 |  4 |

    Args:
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
            with shape (40, 40) each.
        num_layers (int): The number of transformer layers.
        embed_dim (int): The dimensionality of each patch embedding (token).
        layer (nn.Module): The transformer layer module.
        token_pos_embedding (nn.Module): The token positional embedding module.
        pre_tile_pos_embed (Optional[nn.Module]): The pre-tile positional embedding module. It should be
            None if your image was not tile-cropped in advance.
        post_tile_pos_embed (Optional[nn.Module]): The post-tile positional embedding module. It should be
            None if your image was not tile-cropped in advance.
        cls_projection (Optional[nn.Module]): The CLS projection module. It should take an input tensor
            of shape (bsz * n_tiles, n_tokens, embed_dim) and output a tensor of shape
            (bsz * n_tiles, cls_output_dim). If provided, only the CLS token projection will be
            outputted, instead of all tokens.
        out_indices (Optional[List[int]]): The indices of hidden layers to return.
            If provided, it will return the intermediate results of the transformer layers
            before they go through a next layer. For example, ``out_indices=[0,3]`` will
            return the tokens before they go through the first and fourth layers.
        in_channels (int): The number of image input channels.

    Raises:
        ValueError: If `tile_size` is not greater than 0.
        ValueError: If `patch_size` is not greater than 0.
        ValueError: If `len(out_indices)` is greater than `num_layers`.
    """

    def __init__(self, patch_size: 'int', tile_size: 'int', num_layers: 'int', embed_dim: 'int', layer: 'nn.Module', token_pos_embedding: 'nn.Module', pre_tile_pos_embed: 'Optional[nn.Module]'=None, post_tile_pos_embed: 'Optional[nn.Module]'=None, cls_projection: 'Optional[nn.Module]'=None, out_indices: 'Optional[List[int]]'=None, in_channels: 'int'=3) ->None:
        super().__init__()
        if tile_size <= 0:
            raise ValueError('tile_size must be > 0')
        if patch_size <= 0:
            raise ValueError('patch_size must be > 0')
        if out_indices and len(out_indices) > num_layers:
            raise ValueError(f'len(out_indices) must be <= num_layers. Got out_indices={out_indices!r} and num_layers={num_layers!r}')
        patch_grid_size = tile_size // patch_size
        self.patches_per_tile = patch_grid_size ** 2
        self.out_indices = out_indices
        if not out_indices:
            self.out_indices = []
        self.pre_tile_pos_embed = pre_tile_pos_embed
        self.post_tile_pos_embed = post_tile_pos_embed
        self.token_pos_embedding = token_pos_embedding
        self.cls_projection = cls_projection
        self.layers = _get_clones(layer, num_layers)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), bias=False)
        self.ln_post = Fp32LayerNorm(embed_dim)
        self.ln_pre = Fp32LayerNorm(embed_dim)
        self.cls_token_embedding = CLSEmbedding(embed_dim)

    def get_image_tokens_per_tile(self):
        return self.patches_per_tile + 1

    def forward(self, images: 'torch.Tensor', aspect_ratio: 'Optional[torch.Tensor]'=None) ->Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Processes images and returns the tokens and hidden states.

        Multiple images per sample: we add a dimension n_imgs to the input. This is useful when a single
        sample constains multiple images, for example:

        - sample 1: "<image> what animal is this?"
        - sample 2: "I like <image> more than <image>"

        In this case, sample 1 has one image, and sample 2 has two images. max_n_imgs = max(2,1) = 2.
        So your input should have shape (bsz=2, n_imgs=2, num_tiles, n_channels, tile_size, tile_size).

        Notice that to batch it, you will have to pad n_imgs to max_n_imgs and max_num_tiles.

        Args:
            images (torch.Tensor): torch.Tensor with shape (bsz, n_imgs, n_tiles, n_channels, tile_size, tile_size).
            aspect_ratio (Optional[torch.Tensor]): torch.Tensor with shape (bsz, n_imgs, 2). If all
                images have a single tile, i.e. they were not tile-cropped, it should be None.
                Used to calculate the positional embeddings for the tiles.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: A tuple: (x, hidden_states),
                where x is a torch.tensor of shape (bsz, n_imgs, n_tiles, n_tokens, embed_dim) and
                hidden_states has shape is a list of len(out_indices) torch.tensor with shape
                (bsz, n_imgs, n_tiles, n_tokens, embed_dim).

        Raises:
            ValueError: If aspect_ratio is None, but n_tiles > 1 in the batch.

        Examples:

            >>> from torchtune.modules.transforms.vision_utils.tile_crop import tile_crop
            >>> from torchtune.modules import VisionTransformer
            >>>
            >>> num_channels = 3
            >>> image_size = (800,400)
            >>> tile_size = 400
            >>> patch_size=40
            >>> patch_grid_size = tile_size // patch_size
            >>>
            >>> # for details about preprocessing, please check
            >>> # torchtune.models.clip._transforms.CLIPImageTransform
            >>>
            >>> # create a random image
            >>> image = torch.rand(num_channels, image_size[0], image_size[1])
            >>>
            >>> # (num_tiles, nch, h, w) -> (2, 3, 400, 400)
            >>> tile_cropped_image = tile_crop(image, tile_size)
            >>> aspect_ratio = torch.tensor([2,1])
            >>>
            >>> # make it a batch of 1 image
            >>> batch_image = tile_cropped_image.unsqueeze(0)
            >>> batch_aspect_ratio = aspect_ratio.unsqueeze(0)
            >>>
            >>> # make it have only 1 image per sample
            >>> batch_image = tile_cropped_image.unsqueeze(1)
            >>> batch_aspect_ratio = aspect_ratio.unsqueeze(1)
            >>>
            >>> # For a detailed example, please check
            >>> # torchtune.models.clip._position_embeddings.clip_vision_encoder
            >>> # model = VisionTransformer(
            ... #           out_indices = [1,2,3,4,5],
            ... #           patch_size=40,
            ... #           patch_grid_size = patch_grid_size,
            ... #           embed_dim = 32,
            ... #           num_layers = 6,
            ... #           in_channels = num_channels,
            ... #           ...)
            >>>
            >>> x, hidden_states = model(images = batch_image, aspect_ratio = batch_aspect_ratio)
            >>>
            >>> # (bsz, n_imgs, num_tiles, num_patches_per_tile + CLS token, embed_dim)
            >>> print(x.shape)
            torch.Size([1, 1, 2, 101, 32])
            >>>
            >>> # list with tensors of shape (bsz, n_imgs, num_tiles, num_patches_per_tile + CLS token, embed_dim)
            >>> print(len(hidden_states))
            5
        """
        hidden_states = []
        bsz, n_imgs, n_tiles, nch, w, h = images.shape
        bsz_and_n_imgs = bsz * n_imgs
        if aspect_ratio is None:
            aspect_ratio = torch.ones((bsz_and_n_imgs, 2), dtype=torch.int, device=images.device)
            if n_tiles > 1:
                raise ValueError(f'aspect_ratio was not provided, but found n_tiles>1 for images.shape={images.shape!r}. Please provide aspect_ratio.')
        images = images.reshape(bsz_and_n_imgs * n_tiles, nch, w, h)
        aspect_ratio = aspect_ratio.reshape(bsz_and_n_imgs, 2)
        x = self.conv(images)
        x = x.reshape(bsz_and_n_imgs, n_tiles, -1, self.patches_per_tile).permute(0, 1, 3, 2)
        bsz_and_n_imgs, n_tiles, n_tokens, embed_dim = x.shape
        if self.pre_tile_pos_embed:
            x = self.pre_tile_pos_embed(x, aspect_ratio)
        x = self.cls_token_embedding(x)
        n_tokens += 1
        x = self.token_pos_embedding(x, aspect_ratio)
        x = self.ln_pre(x)
        x = x.reshape(bsz_and_n_imgs, n_tiles * n_tokens, embed_dim)
        for layer_idx, transformer_layer in enumerate(self.layers):
            if layer_idx in self.out_indices:
                h = x.reshape(bsz, n_imgs, n_tiles, n_tokens, embed_dim)
                hidden_states.append(h)
            x = transformer_layer(x)
        x = self.ln_post(x)
        if self.post_tile_pos_embed:
            x = x.reshape(bsz_and_n_imgs, n_tiles, n_tokens, embed_dim)
            x = self.post_tile_pos_embed(x, aspect_ratio)
        x = x.reshape(bsz, n_imgs, n_tiles, n_tokens, embed_dim)
        if self.cls_projection:
            x = self.cls_projection(x)
        return x, hidden_states


class CLSProjection(nn.Module):
    """
    Linear projection of the CLS token.

    Args:
        embed_dim (int): The dimensionality of the input patch embedding.
        cls_output_dim (int): The dimensionality of the output projection.
    """

    def __init__(self, embed_dim: 'int', cls_output_dim: 'int') ->None:
        super().__init__()
        scale = embed_dim ** -0.5
        self.cls_output_dim = cls_output_dim
        self.weight = nn.Parameter(scale * torch.randn(embed_dim, cls_output_dim))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        bsz, n_imgs, n_tiles, n_tokens, embed_dim = x.shape
        x = x.reshape(bsz * n_imgs * n_tiles, n_tokens, embed_dim)
        x = x[:, 0, :] @ self.weight
        x = x.reshape(bsz, n_imgs, n_tiles, 1, self.cls_output_dim)
        return x


class DPOLoss(nn.Module):
    """
    Direct Preference Optimization (DPO) Loss module: https://arxiv.org/abs/2305.18290
    Simply stated from the paper:

        Intuitively, the DPO update increases the relative log probability of preferred to dispreferred responses,
        but it incorporates a dynamic, per-example importance weight that prevents
        the model degeneration that we find occurs with a naive probability ratio objective.

    Based on the implementation in HF's TRL library:
    https://github.com/huggingface/trl/blob/5d1deb1445828cfd0e947cb3a7925b1c03a283fc/trl/trainer/dpo_trainer.py#L844

    DPO retains similarities to PPO (https://arxiv.org/abs/2009.01325), where it optimizes a policy
    (language) model to align with human preferences, and regularizes the loss function using a baseline
    reference (the frozen, initial language model) to prevent over-fitting to the preference dataset.
    It differs from PPO by optimizing the policy model directly using labelled preference data, rather
    than using an additional reward model to provide feedback.
    This significantly simplifies training and reduces compute overhead.

    Args:
        beta (float): Temperature parameter for the DPO loss, typically in the range of 0.1 to 0.5. Default is 0.1.
        label_smoothing (float): Parameter encoding uncertainty about the labels. Default is 0.
    """

    def __init__(self, beta: 'float'=0.1, label_smoothing: 'float'=0.0):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(self, policy_chosen_logps: 'torch.Tensor', policy_rejected_logps: 'torch.Tensor', reference_chosen_logps: 'torch.Tensor', reference_rejected_logps: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps (torch.Tensor): Log probabilities of the policy model
                for the chosen responses. Shape: (batch_size)
            policy_rejected_logps (torch.Tensor): Log probabilities of the policy model
                for the rejected responses. Shape: (batch_size)
            reference_chosen_logps (torch.Tensor): Log probabilities of the reference model
                for the chosen responses. Shape: (batch_size)
            reference_rejected_logps (torch.Tensor): Log probabilities of the reference model
                for the rejected responses. Shape: (batch_size)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple of three tensors:
                - losses: The DPO loss for each example in the batch.
                - chosen_rewards: Rewards for the chosen responses.
                - rejected_rewards: Rewards for the rejected responses.

        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios
        losses = -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing) - F.logsigmoid(-self.beta * logits) * self.label_smoothing
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        return losses, chosen_rewards, rejected_rewards


class RSOLoss(nn.Module):
    """
    Statistical Rejection Sampling Optimization (RSO) or "hinge" loss module: https://arxiv.org/abs/2309.06657.
    Intuition from the paper:

        DPO is a logistic regression on human preference data, and SLiC (https://arxiv.org/abs/2305.10425) is almost
        equivalent to a support vector machine (SVM) with hinge loss. [RSO] improve[s] SLiC as the SVM counter part of DPO.

    Based on the implementation in HF's TRL library:
    https://github.com/huggingface/trl/blob/4dce042a3863db1d375358e8c8092b874b02934b/trl/trainer/dpo_trainer.py#L1141

    Args:
        gamma (float): Equivalent temperature parameter (from DPO) for the RSO loss.
    """

    def __init__(self, gamma: 'float'=0.1):
        super().__init__()
        self.gamma = gamma

    def forward(self, policy_chosen_logps: 'torch.Tensor', policy_rejected_logps: 'torch.Tensor', reference_chosen_logps: 'torch.Tensor', reference_rejected_logps: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the RSO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps (torch.Tensor): Log probabilities of the policy model
                for the chosen responses. Shape: (batch_size)
            policy_rejected_logps (torch.Tensor): Log probabilities of the policy model
                for the rejected responses. Shape: (batch_size)
            reference_chosen_logps (torch.Tensor): Log probabilities of the reference model
                for the chosen responses. Shape: (batch_size)
            reference_rejected_logps (torch.Tensor): Log probabilities of the reference model
                for the rejected responses. Shape: (batch_size)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple of three tensors:
                - losses: The RSO loss for each example in the batch.
                - chosen_rewards: Rewards for the chosen responses.
                - rejected_rewards: Rewards for the rejected responses.

        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios
        losses = torch.relu(1 - self.gamma * logits)
        chosen_rewards = self.gamma * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.gamma * (policy_rejected_logps - reference_rejected_logps).detach()
        return losses, chosen_rewards, rejected_rewards


class SimPOLoss(nn.Module):
    """
    SimPO: Simple Preference Optimization with a Reference-Free Reward: https://arxiv.org/abs/2405.14734.
    Intuition from the paper:

        The effectiveness of SimPO is attributed to a key design: using the average log probability of a sequence as
        the implicit reward. Additionally, we introduce a target reward margin to the Bradley-Terry objective to
        encourage a larger margin between the winning and losing responses, further enhancing the algorithm's performance.

    Based on the TRL implementation:
    https://github.com/huggingface/trl/blob/98ad01ddfd1e1b67ec018014b83cba40e0caea66/trl/trainer/cpo_trainer.py#L603

    SimPO is pretty much identitcal to DPO but uses average logprobs to eliminate the need for a reference model to regularize
    the policy during training. It also uses a target reward margin to guide the policy towards better responses.
    This is kind of the same intuition as in :class:`~torchtune.rlhf.loss.IPOLoss`, but instead of optimizing against
    a margin between the reference policy and policy models, we're optimizing against a margin between the chosen and
    rejected responses.

    Args:
        beta (float): Equivalent temperature scaling parameter to DPO loss, typically in the range of 2.0 to 2.5. Default is 2.0.
        gamma (float): Target reward margin hyperparameter, typically we have ``gamma in (0, 1.5]``.
            Default is 0.5.
        label_smoothing (float): Parameter encoding uncertainty about the labels. Default is 0.
    """

    def __init__(self, beta: 'float'=2.0, gamma: 'float'=0.5, label_smoothing: 'float'=0.0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, policy_chosen_logps: 'torch.Tensor', policy_rejected_logps: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the SimPO loss for a batch chosen and rejected average log probabilities.

        Args:
            policy_chosen_logps (torch.Tensor): Average log probabilities of the policy model
                for the chosen responses with shape [b,].
            policy_rejected_logps (torch.Tensor): Average log probabilities of the policy model
                for the rejected responses with shape [b,].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]; A tuple of three tensors with shape [b,]:
                - losses: The SimPO loss for each example in the batch.
                - chosen_rewards: Rewards for the chosen responses.
                - rejected_rewards: Rewards for the rejected responses.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        gamma_logratios = self.gamma / self.beta
        logits = pi_logratios - gamma_logratios
        losses = -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing) - F.logsigmoid(-self.beta * logits) * self.label_smoothing
        chosen_rewards = self.beta * policy_chosen_logps.detach()
        rejected_rewards = self.beta * policy_rejected_logps.detach()
        return losses, chosen_rewards, rejected_rewards


class PPOLoss(nn.Module):
    """
    Proximal Policy Optimization (PPO) Loss module.
    This implementation uses the following references:

    https://arxiv.org/abs/1707.06347 eqn. 7

    https://github.com/vwxyzjn/lm-human-preference-details/blob/ccc19538e817e98a60d3253242ac15e2a562cb49/lm_human_preference_details/train_policy_accelerate.py#L719

    https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L68-L75


    Args:
        epsilon (float): clipping range for PPO update.
        value_clip_range (float): clipping range for value function update.
        value_coeff (float): coefficient for the value function loss contribution.
    """

    def __init__(self, epsilon: 'float'=0.1, value_clip_range: 'float'=0.2, value_coeff: 'float'=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.value_clip_range = value_clip_range
        self.value_coeff = value_coeff

    def forward(self, pi_old_logprobs: 'torch.Tensor', pi_logprobs: 'torch.Tensor', advantages: 'torch.Tensor', phi_old_values: 'torch.Tensor', phi_values: 'torch.Tensor', returns: 'torch.Tensor', padding_masks: 'Optional[torch.Tensor]'=None, value_padding_masks: 'Optional[torch.Tensor]'=None) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Forward pass of the PPO loss module.

        Args:
            pi_old_logprobs (torch.Tensor): Log probabilities of the old policy.
            pi_logprobs (torch.Tensor): Log probabilities of the current policy.
            advantages (torch.Tensor): Advantage values.
            phi_old_values (torch.Tensor): Value predictions of the old value function.
            phi_values (torch.Tensor): Value predictions of the current value function.
            returns (torch.Tensor): Return values.
            padding_masks (Optional[torch.Tensor]): Padding token masks of the same shape as ``pi_logprobs``,
                where True indicates the corresponding loss values should participage in policy loss calculation.
            value_padding_masks (Optional[torch.Tensor]): Padding token masks of the same shape as ``pi_logprobs``,
                where True indicates the corresponding loss values should participage in value loss calculation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple of five tensors:
                - loss: The total PPO loss.
                - policy_loss: The policy function loss.
                - value_loss: The value function loss.
                - ratios: The ratio between the current and old policy probabilities.
                - clipfrac: The fraction of ratios that were clipped.

        """
        ratios = torch.exp(pi_logprobs - pi_old_logprobs)
        clipped_ratios = torch.clamp(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon)
        policy_losses_clipped = -advantages * clipped_ratios
        policy_losses_unclipped = -advantages * ratios
        clipfrac = (policy_losses_clipped > policy_losses_unclipped).float()
        clipfrac = clipfrac.mean() if padding_masks is None else rlhf.masked_mean(clipfrac, padding_masks)
        policy_loss = torch.maximum(policy_losses_clipped, policy_losses_unclipped)
        policy_loss = policy_loss.mean() if padding_masks is None else rlhf.masked_mean(policy_loss, padding_masks)
        values_clipped = torch.clamp(phi_values, phi_old_values - self.value_clip_range, phi_old_values + self.value_clip_range)
        value_loss = torch.maximum((phi_values - returns) ** 2, (values_clipped - returns) ** 2)
        value_loss = 0.5 * value_loss.mean() if value_padding_masks is None else 0.5 * rlhf.masked_mean(value_loss, value_padding_masks)
        loss = policy_loss + value_loss * self.value_coeff
        return loss, policy_loss.detach(), value_loss.detach(), ratios.mean().detach(), clipfrac.detach()


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (CLSEmbedding,
     lambda: ([], {'embed_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CLSProjection,
     lambda: ([], {'embed_dim': 4, 'cls_output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (DummyCrossAttentionLayer,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DummySelfAttentionLayer,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FeedForward,
     lambda: ([], {'gate_proj': torch.nn.ReLU(), 'down_proj': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FeedForwardRef,
     lambda: ([], {'dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Fp32LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FusionLayer,
     lambda: ([], {'layer': torch.nn.ReLU(), 'fusion_layer': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Gemma2FinalNorm,
     lambda: ([], {'capping_value': 4, 'embed_dim': 4, 'eps': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GemmaRMSNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Llama3ScaledRoPE,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Llama3VisionProjectionHead,
     lambda: ([], {'layers': [torch.nn.ReLU()], 'output': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (PPOLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Phi3RotaryPositionalEmbeddings,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Qwen2RotaryPositionalEmbeddings,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RMSNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RMSNormRef,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RSOLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (RotaryPositionalEmbeddings,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TanhGate,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TanhSoftCapping,
     lambda: ([], {'capping_value': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TokenPositionalEmbedding,
     lambda: ([], {'embed_dim': 4, 'tile_size': 4, 'patch_size': 4}),
     lambda: ([torch.rand([4, 4, 2, 4])], {})),
    (_Wrapper,
     lambda: ([], {'layer': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

