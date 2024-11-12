
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


import uuid


from typing import Literal


from typing import Optional


import torch


from abc import ABC


from abc import abstractmethod


from typing import Callable


from typing import Union


import time


import itertools


import warnings


from typing import Tuple


import torch._dynamo.config


import torch._inductor.config


from functools import reduce


from math import gcd


import torch.nn as nn


from torch import Tensor


from torch.nn import functional as F


import logging


import torch.nn.functional as F


import math


from typing import Any


from typing import Dict


from torch.utils.data import DataLoader


from typing import List


from typing import Type


from typing import Mapping


import numpy as np


import pandas as pd


from torch.utils.data import Dataset


import inspect


import re


from time import perf_counter as timer


from torch import nn


class KVCache(nn.Module):

    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype):
        super().__init__()
        cache_shape = max_batch_size, n_heads, max_seq_length, head_dim
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val
        return k_out, v_out


def find_multiple(n: 'int', *args: Tuple[int]) ->int:
    k = reduce(lambda x, y: x * y // gcd(x, y), args + (1,))
    if n % k == 0:
        return n
    return n + k - n % k


def get_default_dtype() ->str:
    """Compute default 'dtype' based on GPU architecture"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_properties = torch.cuda.get_device_properties(i)
            dtype = 'float16' if device_properties.major <= 7 else 'bfloat16'
    else:
        dtype = 'float16'
    None
    return dtype


transformer_configs = {'metavoice-1B': dict(n_layer=24, n_head=16, dim=2048, vocab_size=2562)}


class RMSNorm(nn.Module):

    def __init__(self, ndim: 'int', eps: 'float'):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x) * self.weight


class Attention(nn.Module):

    def __init__(self, config: 'ModelArgs'):
        super().__init__()
        assert config.dim % config.n_head == 0
        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim

    def forward(self, x: 'Tensor', mask: 'Tensor', input_pos: 'Optional[Tensor]'=None) ->Tensor:
        bsz, seqlen, _ = x.shape
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)
        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)
        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)
        return y


class SwiGLU(nn.Module):

    def __init__(self, in_dim, out_dim, bias) ->None:
        super().__init__()
        self.w1 = nn.Linear(in_dim, out_dim, bias=bias)
        self.w3 = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return F.silu(self.w1(x)) * self.w3(x)


class FeedForward(nn.Module):

    def __init__(self, config: 'ModelArgs') ->None:
        super().__init__()
        self.swiglu = SwiGLU(config)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: 'Tensor') ->Tensor:
        return self.w2(self.swiglu(x))


class TransformerBlock(nn.Module):

    def __init__(self, config: 'ModelArgs') ->None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: 'Tensor', input_pos: 'Tensor', mask: 'Tensor') ->Tensor:
        h = x + self.attention(self.attention_norm(x), mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):

    def __init__(self, config: 'ModelArgs') ->None:
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.pos_embeddings = nn.Embedding(config.block_size, config.dim)
        self.speaker_cond_pos = nn.Linear(config.speaker_emb_dim, config.dim, bias=False)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.mask_cache: 'Optional[Tensor]' = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_spk_cond_mask(self):
        self.spk_cond_mask = torch.zeros((2, 1, self.config.dim), dtype=torch.bool)
        self.spk_cond_mask[0] = 1

    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim, dtype=self.config.dtype)
        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))

    def forward(self, idx: 'Tensor', spk_emb: 'Tensor', input_pos: 'Tensor') ->Tensor:
        mask = self.causal_mask[None, None, input_pos]
        x = self.tok_embeddings(idx) + self.pos_embeddings(input_pos) + self.speaker_cond_pos(spk_emb) * self.spk_cond_mask
        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: 'str'):
        return cls(ModelArgs.from_name(name))


class WeightOnlyInt8Linear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: 'int'
    out_features: 'int'
    weight: 'torch.Tensor'

    def __init__(self, in_features: 'int', out_features: 'int', bias: 'bool'=True, device=None, dtype=None) ->None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('weight', torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer('scales', torch.ones(out_features, dtype=torch.bfloat16))

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return F.linear(input, self.weight) * self.scales


def linear_forward_int4(x, weight_int4pack, scales_and_zeros, out_features, groupsize):
    origin_x_size = x.size()
    x = x.reshape(-1, origin_x_size[-1])
    c = torch.ops.aten._weight_int4pack_mm(x, weight_int4pack, groupsize, scales_and_zeros)
    new_shape = origin_x_size[:-1] + (out_features,)
    c = c.reshape(new_shape)
    return c


class WeightOnlyInt4Linear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: 'int'
    out_features: 'int'
    weight: 'torch.Tensor'

    def __init__(self, in_features: 'int', out_features: 'int', bias=True, device=None, dtype=None, groupsize: 'int'=128, inner_k_tiles: 'int'=8, padding: 'bool'=True, use_cuda=True) ->None:
        super().__init__()
        self.padding = padding
        if padding:
            self.origin_in_features = in_features
            in_features = find_multiple(in_features, 1024)
        self.in_features = in_features
        self.out_features = out_features
        assert not bias, 'require bias=False'
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles
        assert out_features % 8 == 0, 'require out_features % 8 == 0'
        assert in_features % (inner_k_tiles * 16) == 0, 'require in_features % (innerKTiles * 16) == 0'
        if use_cuda:
            self.register_buffer('weight', torch.empty((out_features // 8, in_features // (inner_k_tiles * 16), 32, inner_k_tiles // 2), dtype=torch.int32))
        else:
            self.register_buffer('weight', torch.empty((out_features, in_features // 2), dtype=torch.uint8))
        self.register_buffer('scales_and_zeros', torch.empty((in_features // groupsize, out_features, 2), dtype=torch.bfloat16))

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        input = input
        if self.padding:
            import torch.nn.functional as F
            input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return linear_forward_int4(input, self.weight, self.scales_and_zeros, self.out_features, self.groupsize)


class SelfAttention(nn.Module):

    def __init__(self, config):
        """
        Initializes the SelfAttention module.

        Args:
            config: An object containing the configuration parameters for the SelfAttention module.
        """
        super().__init__()
        self._validate_config(config)
        self._initialize_parameters(config)

    def empty_kv_cache(self, batch_size: 'int', kv_cache_maxlen: 'int', dtype: 'torch.dtype'):
        """
        Empties the key-value cache.

        Args:
            batch_size: The batch size.
            kv_cache_maxlen: The maximum length of the key-value cache.
            dtype: The data type of the cache.

        Raises:
            Exception: If trying to empty the KV cache when it is disabled.
        """
        if self.kv_cache_enabled is False:
            raise Exception('Trying to empty KV cache when it is disabled')
        self.register_buffer('kv_cache', torch.zeros(2, batch_size, kv_cache_maxlen, self.n_head, self.n_embd // self.n_head, dtype=dtype, device=self.c_attn.weight.device), persistent=False)
        self.kv_cache_first_empty_index = 0

    def _initialize_parameters(self, config):
        """
        Initializes the parameters of the SelfAttention module.

        Args:
            config: An object containing the configuration parameters for the SelfAttention module.
        """
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.causal = config.causal
        self.attn_kernel_type = config.attn_kernel_type
        self.attn_dropout = nn.Dropout(config.dropout)
        self.kv_cache_enabled = False

    def _validate_config(self, config):
        """
        Validates the configuration parameters.

        Args:
            config: An object containing the configuration parameters for the SelfAttention module.

        Raises:
            AssertionError: If the embedding dimension is not divisible by the number of heads.
        """
        assert config.n_embd % config.n_head == 0, 'Embedding dimension must be divisible by number of heads'

    def _update_kv_cache(self, q, k, v):
        """
        Updates the key-value cache.

        Args:
            q: The query tensor.
            k: The key tensor.
            v: The value tensor.

        Returns:
            The updated key and value tensors.

        Raises:
            AssertionError: If the dimensions of the query, key, and value tensors are not compatible.
        """
        q_time, k_time, v_time = q.shape[1], k.shape[1], v.shape[1]
        if self.kv_cache_first_empty_index == 0:
            assert q_time == k_time and q_time == v_time
        else:
            assert q_time == 1, f'Only one query at a time is supported, but got q_time={q_time} for kv_cache_first_empty_index={self.kv_cache_first_empty_index}'
        self.kv_cache[0, :, self.kv_cache_first_empty_index:self.kv_cache_first_empty_index + q_time] = k
        self.kv_cache[1, :, self.kv_cache_first_empty_index:self.kv_cache_first_empty_index + q_time] = v
        self.kv_cache_first_empty_index += q_time
        k = self.kv_cache[0, :, :self.kv_cache_first_empty_index]
        v = self.kv_cache[1, :, :self.kv_cache_first_empty_index]
        return k, v

    def _torch_attn(self, c_x: 'torch.Tensor') ->torch.Tensor:
        """
        Performs attention using the torch.nn.functional.scaled_dot_product_attention function.

        Args:
            c_x: The input tensor.

        Returns:
            The output tensor.
        """
        q, k, v = c_x.split(1, dim=2)
        q = q.squeeze(2)
        k = k.squeeze(2)
        v = v.squeeze(2)
        is_causal_attn_mask = self.causal and (not self.kv_cache_enabled or self.kv_cache_first_empty_index == 0)
        if self.kv_cache_enabled:
            k, v = self._update_kv_cache(q, k, v)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=is_causal_attn_mask).transpose(1, 2)
        return y

    def forward(self, x):
        """
        Performs the forward pass of the SelfAttention module.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        B, T, C = x.size()
        c_x = self.c_attn(x).view(B, T, 3, self.n_head, C // self.n_head)
        if self.attn_kernel_type == 'torch_attn':
            y = self._torch_attn(c_x)
        else:
            raise Exception(f'Unknown attention kernel type: {self.attn_kernel_type}')
        y = y.contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-05)


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.non_linearity = config.nonlinearity_type
        hidden_dim = 4 * config.n_embd
        if config.nonlinearity_type == 'gelu':
            self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
            self.gelu = nn.GELU()
            self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        elif config.nonlinearity_type == 'swiglu':
            if config.swiglu_multiple_of is None:
                raise Exception('SwiGLU requires swiglu_multiple_of to be set')
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = config.swiglu_multiple_of * math.ceil(hidden_dim / config.swiglu_multiple_of)
            self.swiglu = SwiGLU(config.n_embd, hidden_dim, bias=config.bias)
            self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        else:
            raise Exception(f'Unknown nonlinearity type: {config.nonlinearity_type}')
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        if self.non_linearity == 'gelu':
            x = self.c_fc(x)
            x = self.gelu(x)
        elif self.non_linearity == 'swiglu':
            x = self.swiglu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Block class represents a single block in the model.

    Args:
        config (object): Configuration object containing parameters for the block.

    Attributes:
        ln_1 (object): Layer normalization for the attention layer.
        ln_2 (object): Layer normalization for the feed-forward layer.
        attn (object): Self-attention layer.
        mlp (object): Multi-layer perceptron layer.

    Methods:
        forward(x): Performs forward pass through the block.
    """

    def __init__(self, config):
        super().__init__()
        if config.norm_type == 'rmsnorm':
            if config.rmsnorm_eps is None:
                raise Exception('RMSNorm requires rmsnorm_eps to be set')
            self.ln_1 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
            self.ln_2 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        elif config.norm_type == 'layernorm':
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        else:
            raise Exception(f'Unknown norm type: {config.norm_type}')
        self.attn = SelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        """
        Performs forward pass through the block.

        Args:
            x (tensor): Input tensor.

        Returns:
            tensor: Output tensor after passing through the block.
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


def top_p_sample(prob_dist: 'torch.Tensor', top_p: 'float'):
    sorted_probs, sorted_indices = torch.sort(prob_dist, descending=True, dim=-1)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cum_sum_probs > top_p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0
    sorted_indices_to_remove = sorted_indices_to_remove.bool()
    sorted_probs[sorted_indices_to_remove] = 0
    reversed_indices = torch.argsort(sorted_indices)
    prob_dist = torch.gather(sorted_probs, -1, reversed_indices)
    prob_dist = prob_dist / prob_dist.sum(dim=-1, keepdim=True)
    return prob_dist


class CausalInferenceMixin:
    """
    Mixin class for performing inference in a causal language model.

    This mixin provides methods for predicting the next token in a sequence, sampling from the model,
    and applying token prediction masks.

    Attributes:
        None

    Methods:
        _sample_next_token: Predicts the next token in the sequence.
        _create_token_pred_mask: Creates a token prediction mask based on sequence lengths.
        _apply_token_pred_mask: Applies a token prediction mask to the next token predictions.
        _sample_batch: Samples a batch of tokens from the model.
        _sort_for_batching: Sorts the input sequences for efficient batching.
        _causal_sample: Generates a sequence of tokens using causal sampling.

    """

    @torch.no_grad()
    def _sample_next_token(self, *, idx: torch.Tensor, speaker_embs: Optional[torch.Tensor], temperature: float, top_k: Optional[int], top_p: Optional[float], guidance_scale: Optional[Tuple[float, float]]) ->torch.Tensor:
        """
        Predict the next token in the sequence.

        Args:
            idx (torch.Tensor): Initial sequence indices of shape (batch, num_hierarchies, time).
            speaker_embs (Optional[torch.Tensor]): Speaker embeddings. Set to `None` if using an unconditional model.
            temperature (float): Sampling temperature.
            top_k (Optional[int]): Top-k filtering threshold. Set to `None` to disable top-k filtering.
            top_p (Optional[float]): Nucleus sampling threshold. Set to `None` to disable it.
            guidance_scale (Optional[float]): Scale factor for the guidance loss. Set to `None` to disable guidance.

        Returns:
            torch.Tensor: Next index in the sequence after sampling. Shape: (batch, num_hierarchies).
        """
        if top_k is not None and top_p is not None:
            raise ValueError('Only one of top_k and top_p can be set')
        idx_cond = idx if idx.size(-1) <= self.config.block_size else idx[:, :, -self.config.block_size:]
        list_logits, _ = self(idx_cond, speaker_embs=speaker_embs)
        if guidance_scale is not None:
            spkemb_guidance_scale, prompt_guidance_scale = guidance_scale
            assert spkemb_guidance_scale >= 1
            assert prompt_guidance_scale >= 1
            base_scale = spkemb_guidance_scale + prompt_guidance_scale - 1
            for i, logits in enumerate(list_logits):
                if prompt_guidance_scale > 1:
                    logits_cond, logits_uncond_spkemb, logits_uncond_prompt = logits.split(logits.shape[0] // 3, dim=0)
                else:
                    logits_cond, logits_uncond_spkemb = logits.split(logits.shape[0] // 2, dim=0)
                    logits_uncond_prompt = 0
                list_logits[i] = base_scale * logits_cond + (1 - spkemb_guidance_scale) * logits_uncond_spkemb + (1 - prompt_guidance_scale) * logits_uncond_prompt
        list_logits = [(logits[:, -1, :] / temperature) for logits in list_logits]
        if top_k is not None:
            for i in range(len(list_logits)):
                logits = list_logits[i]
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                list_logits[i] = logits
        probs = [F.softmax(logits, dim=-1) for logits in list_logits]
        if top_p is not None:
            for i in range(len(probs)):
                probs[i] = top_p_sample(probs[i], top_p)
        idx_next = [torch.multinomial(prob, num_samples=1) for prob in probs]
        idx_next = torch.cat(idx_next, dim=-1)
        return idx_next

    @torch.no_grad()
    def _create_token_pred_mask(self, idx: 'torch.Tensor', seq_lens: 'list[int]') ->torch.Tensor:
        """
        Creates a token prediction mask based on sequence lengths.

        Args:
            idx (torch.Tensor): Initial sequence indices of shape (batch, num_hierarchies, time).
            seq_lens (list[int]): List of sequence lengths for each sequence in idx.

        Returns:
            torch.Tensor: Token prediction mask of shape (batch, time).
        """
        token_pred_mask = torch.zeros((idx.shape[0], idx.shape[-1]), dtype=torch.bool, device=idx.device)
        for i in range(len(seq_lens)):
            token_pred_mask[i, :seq_lens[i]] = True
        assert (token_pred_mask[:, :min(seq_lens)] == 1).all()
        return token_pred_mask

    @torch.no_grad()
    def _apply_token_pred_mask(self, *, idx_next: torch.Tensor, orig_input_at_t: torch.Tensor, token_pred_mask_at_t: torch.Tensor) ->torch.Tensor:
        """
        Applies a token prediction mask to the next token predictions.

        Args:
            idx_next (torch.Tensor): Next token predictions of shape (batch, num_hierarchies).
            orig_input_at_t (torch.Tensor): Original input at time step t of shape (batch, num_hierarchies).
            token_pred_mask_at_t (torch.Tensor): Token prediction mask at time step t of shape (batch, 1).

        Returns:
            torch.Tensor: Updated next token predictions after applying the token prediction mask.
        """
        idx_next = idx_next * ~token_pred_mask_at_t + orig_input_at_t * token_pred_mask_at_t
        return idx_next

    @torch.no_grad()
    def _sample_batch(self, *, idx: torch.Tensor, max_new_tokens: int, seq_lens: list[int], temperature: float, top_k: Optional[int], top_p: Optional[float], speaker_embs: Optional[torch.Tensor], guidance_scale: Optional[Tuple[float, float]], end_of_audio_token: int, end_of_text_token: int):
        """
        Samples a batch of tokens from the model.

        Args:
            idx (torch.Tensor): Initial sequence indices of shape (batch, num_hierarchies, time).
            max_new_tokens (int): Maximum number of NEW tokens to generate (in addition to largest sequence in idx).
            seq_lens (list[int]): List of sequence lengths for each sequence in idx.
            temperature (float): Sampling temperature.
            top_k (Optional[int]): Top-k filtering threshold. Set to `None` to disable top-k filtering.
            top_p (Optional[float]): Nucleus sampling threshold. Set to `None` to disable it.
            speaker_embs (Optional[torch.Tensor]): Speaker embeddings. Set to `None` if using an unconditional model.
            guidance_scale (Optional[float]): Scale factor for the guidance loss. Set to `None` to disable guidance.

        Returns:
            torch.Tensor: Generated sequence indices of shape (batch, num_hierarchies, time).
        """
        assert max(seq_lens) <= idx.shape[-1]
        token_pred_mask = self._create_token_pred_mask(idx, seq_lens)
        input = torch.clone(idx)
        min_seq_lens = min(seq_lens)
        idx = idx[:, :, :min_seq_lens]
        idx_out = torch.full((idx.shape[0], idx.shape[1], idx.shape[2] + max_new_tokens), end_of_audio_token, dtype=idx.dtype, device=idx.device)
        idx_out[:, :, :min_seq_lens] = idx
        terminated = idx.new_zeros(idx.shape[0], dtype=torch.bool)
        if guidance_scale is not None:
            _, prompt_guidance_scale = guidance_scale
            if speaker_embs is None:
                raise Exception('Guidance is only supported for conditional models')
            speaker_embs = list(speaker_embs) + [None] * speaker_embs.shape[0] + (list(speaker_embs) if prompt_guidance_scale > 1 else [])
        for timestep in tqdm.tqdm(range(min_seq_lens, min_seq_lens + max_new_tokens), desc='tokens: '):
            if terminated.all():
                break
            if self.kv_cache_enabled is True and timestep > min_seq_lens:
                idx_input = idx_out[:, :, [timestep - 1]]
            else:
                idx_input = idx_out[:, :, :timestep]
            if guidance_scale is not None:
                _, prompt_guidance_scale = guidance_scale
                if timestep == min_seq_lens:
                    None
                idx_input = idx_input.unsqueeze(0).repeat(3 if prompt_guidance_scale > 1 else 2, 1, 1, 1).reshape(-1, idx_input.shape[1], idx_input.shape[2])
                if prompt_guidance_scale > 1:
                    idx_input_uncond = idx_input[idx_input.shape[0] // 3 * 2:]
                    idx_input_uncond = idx_input_uncond.view(-1)
                    idx_input_uncond[idx_input_uncond > end_of_audio_token] = end_of_text_token
            idx_next = self._sample_next_token(idx=idx_input, speaker_embs=speaker_embs, temperature=temperature, top_k=top_k, top_p=top_p, guidance_scale=guidance_scale)
            assert idx_next.shape[0] == idx.shape[0]
            if timestep < token_pred_mask.shape[-1]:
                idx_next = self._apply_token_pred_mask(idx_next=idx_next, orig_input_at_t=input[:, :, timestep], token_pred_mask_at_t=token_pred_mask[:, [timestep]])
            is_endofaudio = (idx_next == end_of_audio_token).any(dim=-1)
            terminated = terminated | is_endofaudio
            idx_next[terminated] = end_of_audio_token
            idx_out[:, :, timestep] = idx_next
        return idx_out

    @torch.no_grad()
    def _sort_for_batching(self, *, idx: torch.Tensor, seq_lens: list[int], speaker_embs: Optional[torch.Tensor], batch_size: int, max_new_tokens: int) ->Tuple[list[int], list[int], torch.Tensor, list[int], Optional[torch.Tensor], int]:
        """
        Sorts the input sequences for efficient batching.

        Args:
            idx (torch.Tensor): Initial sequence indices of shape (batch, num_hierarchies, time).
            seq_lens (list[int]): List of sequence lengths for each sequence in idx.
            speaker_embs (Optional[torch.Tensor]): Speaker embeddings. Set to `None` if using an unconditional model.
            batch_size (int): Batch size for sampling. idx is split into batches of this size for sampling.
            max_new_tokens (int): Maximum number of NEW tokens to generate (in addition to largest sequence in idx).

        Returns:
            Tuple[list[int], list[int], torch.Tensor, list[int], Optional[torch.Tensor], int]:
                - sorted_indices (list[int]): List of indices of the input sequences that transform it into sorted order.
                - invert_sorted_indices (list[int]): List of indices to invert the sorted sequences back to the original order.
                - idx (torch.Tensor): Input sequence indices in sorted order.
                - seq_lens (list[int]): Sequence lengths in sorted order.
                - speaker_embs (Optional[torch.Tensor]): speaker embeddings in sorted order.
                - max_token_len (int): Effective maximum number of tokens to generate.
        """
        assert len(seq_lens) == idx.shape[0]
        assert max(seq_lens) <= idx.shape[-1]
        sorted_indices = np.argsort(seq_lens)
        inverted_sorted_indices = np.zeros(len(seq_lens), dtype=np.int32)
        inverted_sorted_indices[sorted_indices] = np.arange(len(seq_lens), dtype=np.int32)
        idx = idx[sorted_indices]
        seq_lens = [seq_lens[i] for i in sorted_indices]
        speaker_embs = speaker_embs[sorted_indices] if speaker_embs is not None else None
        max_token_len = 0
        for start_index in range(0, len(seq_lens), batch_size):
            end_index = min(start_index + batch_size, len(seq_lens))
            batch_seq_lens = seq_lens[start_index:end_index]
            max_token_len = max(max_token_len, min(batch_seq_lens) + max_new_tokens)
        return sorted_indices, inverted_sorted_indices, idx, seq_lens, speaker_embs, max_token_len

    @torch.no_grad()
    def _causal_sample(self, *, idx: torch.Tensor, max_new_tokens: int, seq_lens: list[int], temperature: float, top_k: Optional[int], top_p: Optional[float], speaker_embs: Optional[torch.Tensor], batch_size: int, guidance_scale: Optional[Tuple[float, float]]=None, dtype: torch.dtype=torch.bfloat16, end_of_audio_token: int, end_of_text_token: int) ->torch.Tensor:
        """
        Generates a sequence of tokens using causal sampling.

        Args:
            idx (torch.Tensor): Initial sequence indices of shape (batch, num_hierarchies, time).
            max_new_tokens (int): Maximum number of NEW tokens to generate (in addition to largest sequence in idx).
            seq_lens (list[int]): List of sequence lengths for each sequence in idx.
            temperature (float): Sampling temperature.
            top_k (Optional[int]): Top-k filtering threshold. Set to `None` to disable top-k filtering.
            top_p (Optional[float]): Nucleus sampling threshold. Set to `None` to disable it.
            speaker_embs (Optional[torch.Tensor]): Speaker embeddings. Set to `None` if using an unconditional model.
            batch_size (int): Batch size for sampling. idx is split into batches of this size for sampling.
            guidance_scale (Optional[float]): Scale factor for the guidance loss. Set to `None` to disable guidance.

        Returns:
            torch.Tensor: Generated sequence indices of shape (batch, num_hierarchies, time).
        """
        _, invert_sorted_indices, idx, seq_lens, speaker_embs, max_token_len = self._sort_for_batching(idx=idx, seq_lens=seq_lens, speaker_embs=speaker_embs, batch_size=batch_size, max_new_tokens=max_new_tokens)
        return_idx = torch.zeros((len(seq_lens), idx.size(1), max_token_len), dtype=torch.long, device=idx.device)
        for start_index in tqdm.tqdm(range(0, len(seq_lens), batch_size), desc='batch: '):
            end_index = min(start_index + batch_size, len(seq_lens))
            kv_batch_size = end_index - start_index
            if guidance_scale is not None:
                if guidance_scale[1] > 1:
                    kv_batch_size = 3 * kv_batch_size
                else:
                    kv_batch_size = 2 * kv_batch_size
            if self.kv_cache_enabled:
                self.empty_kv_cache(batch_size=kv_batch_size, kv_cache_maxlen=self.config.block_size, dtype=dtype)
            batch_seq_lens = seq_lens[start_index:end_index]
            batch_max_new_tokens = max_token_len - min(batch_seq_lens)
            batch_idx = idx[start_index:end_index]
            batch_speaker_embs = speaker_embs[start_index:end_index] if speaker_embs is not None else None
            batch_idx = self._sample_batch(idx=batch_idx, max_new_tokens=batch_max_new_tokens, seq_lens=batch_seq_lens, temperature=temperature, top_k=top_k, top_p=top_p, speaker_embs=batch_speaker_embs, guidance_scale=guidance_scale, end_of_audio_token=end_of_audio_token, end_of_text_token=end_of_text_token)
            return_idx[start_index:end_index] = batch_idx
        return return_idx[invert_sorted_indices]

    def empty_kv_cache(self, *, batch_size: int, kv_cache_maxlen: int, dtype: torch.dtype):
        """
        Empties key-value (KV) cache for causal attention.

        Args:
            batch_size (int): The batch size.
            kv_cache_maxlen (int): The maximum length of the KV cache.
            dtype (torch.dtype): The data type of the KV cache.

        Raises:
            Exception: If KV cache is enabled for non-causal attention.

        """
        if self.kv_cache_enabled is False:
            raise Exception('KV cache is not enabled')
        if self.config.causal is False:
            raise Exception('KV cache is not supported for non-causal attention')
        self.kv_pos = 0
        for block in self.transformer.h:
            block.attn.empty_kv_cache(batch_size=batch_size, kv_cache_maxlen=kv_cache_maxlen, dtype=dtype)

    def enable_kv_cache(self):
        """
        Enables key-value (KV) cache for causal attention.

        Raises:
            Exception: If KV cache is enabled for non-causal attention.

        """
        if self.config.causal is False:
            raise Exception('KV cache is not supported for non-causal attention')
        self.kv_cache_enabled = True
        for block in self.transformer.h:
            block.attn.kv_cache_enabled = True

    def disable_kv_cache(self):
        """
        Disables the key-value cache for the transformer and all its blocks.
        """
        self.kv_cache_enabled = False
        for block in self.transformer.h:
            block.attn.kv_cache_enabled = False
            block.attn.kv_cache = None
            block.attn.kv_cache_first_empty_index = 0

    @torch.no_grad()
    def _slow_causal_sampling_loop(self, idx: 'torch.Tensor', max_new_tokens: 'int', temperature: 'float'=1.0, top_k: 'Optional[int]'=None, top_p: 'Optional[float]'=None, speaker_embs: 'Optional[torch.Tensor]'=None, guidance_scale: 'Optional[float]'=None):
        """
        Old non-batched version of causal sampling. Kept for testing / reference.

        Take a conditioning sequence of indices idx (LongTensor of shape (b,n_head,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        assert idx.dim() == 3, 'idx must be a batch of sequences of hierarchical tokens'
        assert idx.size(0) == 1, 'can only do one sequence at a time for now'
        assert top_p is None, 'nucleus sampling not supported yet with _slow_causal_sampling_loop'
        if self.config.causal is not True:
            raise Exception('Causal sampling is only supported for causal models')
        if self.kv_cache_enabled:
            None
            self.empty_kv_cache(batch_size=1, kv_cache_maxlen=self.config.block_size, dtype=torch.bfloat16)
        for i in range(max_new_tokens):
            idx_cond = idx if idx.size(-1) <= self.config.block_size else idx[:, -self.config.block_size:]
            if self.kv_cache_enabled:
                if i > 0:
                    idx_cond = idx_cond[:, :, -1:]
            list_logits, _ = self(idx_cond, speaker_embs=speaker_embs)
            if guidance_scale is not None:
                list_logits_uncond, _ = self(idx_cond, speaker_embs=None)
                list_logits = [(guidance_scale * logits + (1 - guidance_scale) * logits_uncond) for logits, logits_uncond in zip(list_logits, list_logits_uncond)]
            list_logits = [(logits[:, -1, :] / temperature) for logits in list_logits]
            if top_k is not None:
                for i in range(len(list_logits)):
                    logits = list_logits[i]
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                    list_logits[i] = logits
            probs = [F.softmax(logits, dim=-1) for logits in list_logits]
            idx_next = torch.tensor([torch.multinomial(prob, num_samples=1) for prob in probs], device=idx.device)
            idx = torch.cat((idx, idx_next.unsqueeze(0).unsqueeze(-1)), dim=2)
        return idx


END_OF_TEXT_TOKEN = 1537


class NonCausalInferenceMixin:
    """
    Mixin class for non-causal inference in a language model.

    This class provides methods for performing non-causal sampling using a language model.
    """

    @torch.no_grad()
    def _non_causal_sample(self, *, idx: torch.Tensor, speaker_embs: Optional[torch.Tensor], temperature: float, top_k: int):
        """
        Perform non-causal sampling.

        Args:
            idx (torch.Tensor): Input tensor of shape (batch_size, num_in_hierarchies, sequence_length).
            speaker_embs (Optional[torch.Tensor]): Speaker embeddings tensor of shape (batch_size, embedding_size).
            temperature (float): Temperature parameter for scaling the logits.
            top_k (int): Number of top options to consider.

        Returns:
            torch.Tensor: Sampled output tensor of shape (batch_size, num_out_hierarchies, sequence_length).
        """
        b, c, t = idx.size()
        assert t == self.config.block_size, f'input size {t} != config.block_size {self.config.block_size}'
        list_logits, _ = self(idx, speaker_embs=speaker_embs)
        list_logits = [(logits / temperature) for logits in list_logits]
        if top_k is not None:
            for i in range(len(list_logits)):
                logits = list_logits[i]
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, :, [-1]]] = -float('Inf')
                list_logits[i] = logits
                assert logits.shape[0] == b and logits.shape[1] == t
        probs = [F.softmax(logits, dim=-1) for logits in list_logits]
        assert probs[0].shape[0] == b and probs[0].shape[1] == t
        outs = []
        for b_prob in probs:
            out = [torch.multinomial(prob, num_samples=1).transpose(0, 1).unsqueeze(0) for prob in b_prob]
            assert len(out) == b and out[0].shape[0] == 1 and out[0].shape[1] == 1 and out[0].shape[2] == t
            out = torch.cat(out, dim=0)
            assert out.shape[0] == b and out.shape[1] == 1 and out.shape[2] == t
            outs.append(out)
        out = torch.cat(outs, dim=1)
        assert out.shape[0] == b and out.shape[2] == t
        return out


def _check_speaker_emb_dims(speaker_embs: 'Union[list, torch.Tensor]', expected_speaker_emb_dim: 'int', expected_batch_size: 'int') ->Union[torch.Tensor, list]:
    """
    Checks that the speaker embedding dimensions are correct, and reshapes them if necessary.
    """
    if type(speaker_embs) == list:
        b_se = len(speaker_embs)
        for i, s in enumerate(speaker_embs):
            if s is not None:
                emb_dim = s.shape[-1]
                if s.ndim == 1:
                    speaker_embs[i] = speaker_embs[i].unsqueeze(0)
    else:
        if speaker_embs.ndim == 2:
            speaker_embs = speaker_embs[:, None, :]
        b_se, num_examples, emb_dim = speaker_embs.size()
    assert b_se == expected_batch_size, f'Batch size mismatch: {b_se} != {expected_batch_size}'
    assert emb_dim == expected_speaker_emb_dim, f'Speaker embedding dimension mismatch: {emb_dim} != {expected_speaker_emb_dim}'
    return speaker_embs


def _select_spkemb(spkemb, mask):
    _, examples, _ = spkemb.shape
    mask = torch.nn.functional.one_hot(mask.long(), num_classes=examples)
    spkemb = spkemb.transpose(1, 2)
    mask = mask.transpose(1, 2)
    return torch.bmm(spkemb, mask).transpose(1, 2)


class GPT(nn.Module, NonCausalInferenceMixin, CausalInferenceMixin):

    def __init__(self, config: 'GPTConfig', speaker_emb_dim: 'Optional[int]'=None):
        """
        Initialize the GPT model.

        Args:
            config (GPTConfig): Configuration object for the model.
            speaker_emb_dim (Optional[int]): Dimension of the speaker embedding. Default is None.
        """
        super().__init__()
        assert config.vocab_sizes is not None
        assert config.block_size is not None
        self.config = config
        self.kv_cache_enabled = False
        self.kv_pos = 0
        self.speaker_emb_dim = speaker_emb_dim
        self.spk_emb_on_text = config.spk_emb_on_text
        if self.config.causal is True and self.spk_emb_on_text is False:
            None
            None
            None
        if self.config.causal is False and self.spk_emb_on_text is False:
            raise Exception('Cannot use speaker embedding masking with non-causal model. This is unexpected. Check for relevant changes required in code before proceeding.')
        if config.norm_type == 'rmsnorm':
            if config.rmsnorm_eps is None:
                raise Exception('RMSNorm requires rmsnorm_eps to be set')
            ln_f = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        elif config.norm_type == 'layernorm':
            ln_f = LayerNorm(config.n_embd, bias=config.bias)
        else:
            raise Exception(f'Unknown norm type: {config.norm_type}')
        self.transformer = nn.ModuleDict(dict(wtes=nn.ModuleList([nn.Embedding(vsize, config.n_embd) for vsize in config.vocab_sizes]), wpe=nn.Embedding(config.block_size, config.n_embd), drop=nn.Dropout(config.dropout), h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]), ln_f=ln_f))
        if speaker_emb_dim is not None:
            self.speaker_cond_pos = nn.Linear(speaker_emb_dim, config.n_embd, bias=False)
        self.lm_heads = nn.ModuleList()
        if config.target_vocab_sizes is not None:
            assert config.causal is False
        else:
            assert config.causal is True
        for vsize in (config.vocab_sizes if config.target_vocab_sizes is None else config.target_vocab_sizes):
            self.lm_heads.append(nn.Linear(config.n_embd, vsize, bias=False))
        if config.target_vocab_sizes is None:
            for i in range(len(config.vocab_sizes)):
                self.lm_heads[i].weight = self.transformer.wtes[i].weight
            assert len(self.lm_heads) == len(self.transformer.wtes), f'Number of heads ({len(self.lm_heads)}) must match number of one-hot embedding matrics ({len(self.transformer.wtes)}).'
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        None

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _mask_spk_emb_on_text(self, idx: 'torch.Tensor', spk_emb: 'torch.Tensor') ->torch.Tensor:
        """
        This is in a separate function so we can test it easily.
        """
        is_end_of_text = idx[:, 0, :] == END_OF_TEXT_TOKEN
        mask = (torch.cumsum(is_end_of_text, dim=-1) > 0).float()
        spk_emb = spk_emb * mask[:, :, None]
        return spk_emb

    def forward(self, idx, targets=None, speaker_embs=None, speaker_emb_mask=None, loss_reduce: "Literal['mean', 'none']"='mean'):
        device = idx.device
        b, num_hierarchies, t = idx.size()
        if speaker_embs is not None:
            speaker_embs = _check_speaker_emb_dims(speaker_embs=speaker_embs, expected_speaker_emb_dim=self.speaker_emb_dim, expected_batch_size=b)
        assert t <= self.config.block_size, f'Cannot forward sequence of length {t}, block size is only {self.config.block_size}'
        if self.kv_cache_enabled:
            if self.kv_pos == 0:
                pos = torch.arange(0, t, dtype=torch.long, device=device)
                self.kv_pos += t
            else:
                assert t == 1, 'KV cache is only supported for single token inputs'
                pos = torch.tensor([self.kv_pos], dtype=torch.long, device=device)
                self.kv_pos += 1
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
        assert num_hierarchies == len(self.transformer.wtes), f'Input tensor has {num_hierarchies} hierarchies, but model has {len(self.transformer.wtes)} set of input embeddings.'
        tok_emb = torch.zeros((b, t, self.config.n_embd), device=device)
        for i, wte in enumerate(self.transformer.wtes):
            tok_emb += wte(idx[:, i, :])
        pos_emb = self.transformer.wpe(pos)
        spk_emb = 0.0
        if speaker_embs is not None:
            if type(speaker_embs) == list:
                assert speaker_emb_mask is None
                assert self.training is False
                assert self.spk_emb_on_text is True
                spk_emb = []
                for speaker_emb_row in speaker_embs:
                    if speaker_emb_row is not None:
                        spk_emb.append(self.speaker_cond_pos(speaker_emb_row.unsqueeze(0)))
                        assert spk_emb[-1].shape == (1, 1, self.config.n_embd), f'spk_emb[-1].shape={spk_emb[-1].shape}'
                    else:
                        spk_emb.append(torch.zeros((1, 1, self.config.n_embd), device=device, dtype=pos_emb.dtype))
                spk_emb = torch.cat(spk_emb, dim=0)
                assert spk_emb.ndim == 3 and spk_emb.shape[1] == 1 and spk_emb.shape[0] == b, f'spk_emb.ndim={spk_emb.ndim}, spk_emb.shape={spk_emb.shape}, len(speaker_embs)={len(speaker_embs)}'
            else:
                speakers_embedded = self.speaker_cond_pos(speaker_embs)
                if speaker_emb_mask is not None:
                    spk_emb = _select_spkemb(speakers_embedded, speaker_emb_mask)
                    assert spk_emb.shape == (b, t, self.config.n_embd)
                else:
                    spk_emb = speakers_embedded
                    assert spk_emb.ndim == 3 and spk_emb.shape[1] == 1
                if self.training and self.config.spkemb_dropout > 0.0:
                    dropout = torch.ones_like(speakers_embedded) * (torch.rand(speakers_embedded.shape[0], 1, 1, device=device) >= self.config.spkemb_dropout)
                    spk_emb = torch.where(dropout == 0, torch.zeros_like(speakers_embedded), speakers_embedded)
            if self.spk_emb_on_text is False:
                assert speaker_emb_mask is None, 'Not implemented for spk_emb_on_text=False'
                spk_emb = self._mask_spk_emb_on_text(idx, spk_emb)
        x = self.transformer.drop(tok_emb + pos_emb + spk_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            list_logits = [lm_head(x) for lm_head in self.lm_heads]
            losses = [F.cross_entropy(logits.view(-1, logits.size(-1)), targets[:, i, :].contiguous().view(-1), ignore_index=-1, reduction=loss_reduce) for i, logits in enumerate(list_logits)]
            losses = torch.stack(losses)
            if loss_reduce == 'mean':
                losses = losses.mean()
            else:
                losses = rearrange(losses, 'h (b t) -> b h t', h=len(self.lm_heads), b=b, t=t)
        else:
            if self.config.causal:
                list_logits = [lm_head(x[:, [-1], :]) for lm_head in self.lm_heads]
            else:
                list_logits = [lm_head(x) for lm_head in self.lm_heads]
            losses = None
        return list_logits, losses

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [{'params': decay_params, 'weight_decay': weight_decay}, {'params': nodecay_params, 'weight_decay': 0.0}]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        None
        None
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        None
        return optimizer

    @torch.no_grad()
    def generate(self, idx: 'torch.Tensor', max_new_tokens: 'int', seq_lens: 'Optional[list]'=None, temperature: 'float'=1.0, top_k: 'Optional[int]'=None, top_p: 'Optional[float]'=None, speaker_embs: 'Optional[torch.Tensor]'=None, batch_size: 'Optional[int]'=None, guidance_scale: 'Optional[Tuple[float, float]]'=None, dtype: 'torch.dtype'=torch.bfloat16, end_of_audio_token: 'int'=99999, end_of_text_token: 'int'=99999):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,num_hierarchies,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        assert idx.dim() == 3, 'idx must be a batch of sequences of hierarchical tokens'
        if self.config.causal:
            if seq_lens is None or batch_size is None:
                raise Exception('seq_lens and batch_size must be provided for causal sampling')
            return self._causal_sample(idx=idx, max_new_tokens=max_new_tokens, seq_lens=seq_lens, temperature=temperature, top_k=top_k, top_p=top_p, speaker_embs=speaker_embs, batch_size=batch_size, guidance_scale=guidance_scale, dtype=dtype, end_of_audio_token=end_of_audio_token, end_of_text_token=end_of_text_token)
        else:
            if seq_lens is not None:
                raise Exception('seq_lens is not supported yet for non-causal sampling')
            if batch_size is None:
                raise Exception('batch_size must be provided for non-causal sampling')
            if guidance_scale is not None:
                raise Exception('guidance_scale is not supported for non-causal sampling')
            if top_p is not None:
                raise Exception('top_p is not supported for non-causal sampling')
            out = []
            for start_index in tqdm.tqdm(range(0, idx.shape[0], batch_size), desc='Non-causal batching'):
                end_index = min(start_index + batch_size, idx.shape[0])
                out.append(self._non_causal_sample(idx=idx[start_index:end_index], speaker_embs=speaker_embs[start_index:end_index] if speaker_embs is not None else None, temperature=temperature, top_k=top_k))
            return torch.cat(out, dim=0)


mel_n_channels = 40


mel_window_step = 10


model_embedding_size = 256


model_hidden_size = 256


model_num_layers = 3


partials_n_frames = 160


sampling_rate = 16000


class SpeakerEncoder(nn.Module):

    def __init__(self, weights_fpath: 'Optional[str]'=None, device: 'Optional[Union[str, torch.device]]'=None, verbose: 'bool'=True, eval: 'bool'=False):
        super().__init__()
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        start = timer()
        checkpoint = torch.load(weights_fpath, map_location='cpu')
        self.load_state_dict(checkpoint['model_state'], strict=False)
        self
        if eval:
            self.eval()
        if verbose:
            None

    def forward(self, mels: 'torch.FloatTensor'):
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

    @staticmethod
    def compute_partial_slices(n_samples: 'int', rate, min_coverage):
        samples_per_frame = int(sampling_rate * mel_window_step / 1000)
        n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
        frame_step = int(np.round(sampling_rate / rate / samples_per_frame))
        wav_slices, mel_slices = [], []
        steps = max(1, n_frames - partials_n_frames + frame_step + 1)
        for i in range(0, steps, frame_step):
            mel_range = np.array([i, i + partials_n_frames])
            wav_range = mel_range * samples_per_frame
            mel_slices.append(slice(*mel_range))
            wav_slices.append(slice(*wav_range))
        last_wav_range = wav_slices[-1]
        coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
        if coverage < min_coverage and len(mel_slices) > 1:
            mel_slices = mel_slices[:-1]
            wav_slices = wav_slices[:-1]
        return wav_slices, mel_slices

    def embed_utterance(self, wav: 'np.ndarray', return_partials=False, rate=1.3, min_coverage=0.75, numpy: 'bool'=True):
        wav_slices, mel_slices = self.compute_partial_slices(len(wav), rate, min_coverage)
        max_wave_length = wav_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), 'constant')
        mel = audio.wav_to_mel_spectrogram(wav)
        mels = np.array([mel[s] for s in mel_slices])
        mels = torch.from_numpy(mels)
        with torch.no_grad():
            partial_embeds = self(mels)
        if numpy:
            raw_embed = np.mean(partial_embeds.cpu().numpy(), axis=0)
            embed = raw_embed / np.linalg.norm(raw_embed, 2)
        else:
            raw_embed = partial_embeds.mean(dim=0)
            embed = raw_embed / torch.linalg.norm(raw_embed, 2)
        if return_partials:
            return embed, partial_embeds, wav_slices
        return embed

    def embed_speaker(self, wavs: 'List[np.ndarray]', **kwargs):
        raw_embed = np.mean([self.embed_utterance(wav, return_partials=False, **kwargs) for wav in wavs], axis=0)
        return raw_embed / np.linalg.norm(raw_embed, 2)

    def embed_utterance_from_file(self, fpath: 'str', numpy: 'bool') ->torch.Tensor:
        wav_tgt, _ = librosa.load(fpath, sr=sampling_rate)
        wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
        embedding = self.embed_utterance(wav_tgt, numpy=numpy)
        return embedding


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (LayerNorm,
     lambda: ([], {'ndim': 4, 'bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RMSNorm,
     lambda: ([], {'ndim': 4, 'eps': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SwiGLU,
     lambda: ([], {'in_dim': 4, 'out_dim': 4, 'bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

