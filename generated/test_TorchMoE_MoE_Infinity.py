
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


from typing import Tuple


from typing import List


import numpy as np


import random


import torch.distributed as dist


import torch.distributed.rpc as rpc


import time


from torch.distributed import rpc


from typing import Any


from typing import Union


from typing import Dict


import torch.nn as nn


import copy


import uuid


from collections import Counter


from scipy.spatial.distance import cosine


from typing import Optional


import torch.nn.functional as F


import inspect


import math


import warnings


import re


import torch.utils.checkpoint


from torch import nn


from torch.nn import BCEWithLogitsLoss


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


import functools


from typing import Type


from typing import Callable


import logging


from collections import OrderedDict


from collections import defaultdict


from abc import ABC


from abc import abstractmethod


from torch.utils import cpp_extension


def get_arctic_linear(input_dim, output_dim, bias=False, use_deepspeed_implementation=False, ds_optimized_lora_config=None, ds_optimized_quantization_config=None, ds_optimized_base_weight_sharding=False, dtype=torch.bfloat16):
    """Can return deepspeed optimized linear if available.
    Args:
        input_dim, output_dim, bias, dtype: self explanatory (same as from nn.Linear)
        ds_optimized_lora_config: config of type ds_linear.LoRAConfig that contains lora specific parameter if we want to add lora to this layer.
        ds_optimized_quantization_config: config of type ds_linear.QuantizationConfig.
        ds_optimized_base_weight_sharding: bool. If true, the base weight for lora (provided ds_optimized_lora_config is not None) will be sharded across all available gpus
        in a tensor parallel way.
    """
    if is_deepspeed_available():
        if ds_optimized_lora_config is not None:
            ds_optimized_lora_config: 'ds_linear.LoRAConfig' = copy.deepcopy(ds_optimized_lora_config)
            ds_optimized_lora_config.base_weight_sharding = torch.distributed.get_world_size() if ds_optimized_base_weight_sharding else 1
        return ds_linear.OptimizedLinear(input_dim, output_dim, bias, ds_optimized_lora_config, ds_optimized_quantization_config, dtype=dtype)
    return nn.Linear(input_dim, output_dim, bias=bias, dtype=dtype)


class ArcticMLP(nn.Module):

    def __init__(self, config: 'ArcticConfig', use_deepspeed_implementation=False, ds_optimized_lora_config=None, ds_optimized_quantization_config=None, shard_base_weights_if_doing_lora=False, is_residual_mlp=False):
        """MLP class for Arctic supporting vanilla linear layers as well as some deepspeed optimizations.
        ds_optimized_lora_config: config of type ds_linear.LoRAConfig that contains lora specific parameter if we want to add lora to this layer.
        ds_optimized_quantization_config: config of type ds_linear.QuantizationConfig.
        ds_optimized_base_weight_sharding: bool. If true, the base weight for lora (provided ds_optimized_lora_config is not None) will be sharded across all available gpus
        in a tensor parallel way.
        is_residual_mlp: bool. If true, this is MLP inside arctic residual layer which has ffn_dim the same as full intermediate_size.
        """
        super(ArcticMLP, self).__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size if not is_residual_mlp else self.hidden_dim
        self.w1 = get_arctic_linear(self.hidden_dim, self.ffn_dim, False, use_deepspeed_implementation=use_deepspeed_implementation, ds_optimized_lora_config=ds_optimized_lora_config, ds_optimized_quantization_config=ds_optimized_quantization_config, ds_optimized_base_weight_sharding=shard_base_weights_if_doing_lora, dtype=torch.bfloat16)
        self.w2 = get_arctic_linear(self.ffn_dim, self.hidden_dim, False, use_deepspeed_implementation=use_deepspeed_implementation, ds_optimized_lora_config=ds_optimized_lora_config, ds_optimized_quantization_config=ds_optimized_quantization_config, ds_optimized_base_weight_sharding=shard_base_weights_if_doing_lora, dtype=torch.bfloat16)
        self.w3 = get_arctic_linear(self.hidden_dim, self.ffn_dim, False, use_deepspeed_implementation=use_deepspeed_implementation, ds_optimized_lora_config=ds_optimized_lora_config, ds_optimized_quantization_config=ds_optimized_quantization_config, ds_optimized_base_weight_sharding=shard_base_weights_if_doing_lora, dtype=torch.bfloat16)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class SyncArcticMoeBlock(nn.Module):
    archer_config: 'ArcherConfig' = None
    layer_id: 'int' = None

    def __init__(self, config: 'ArcticConfig', layer_id: 'int', **kwargs):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.layer_id = layer_id
        self.top_k = config.num_experts_per_tok
        self.is_moe_layer = (layer_id + 1) % config.moe_layer_frequency == 0
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([ArcticMLP(config) for i in range(self.num_experts)])
        self.archer_tracer = None
        self.archer_engine = None
        self.expert_tensor_ids: 'Dict[int, int]' = None

    def forward(self, hidden_states: 'torch.Tensor') ->Tuple[torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights
        expert_index = selected_experts.reshape(batch_size, sequence_length, self.top_k)
        for i in range(batch_size):
            seq_id = self.seq_id_list[i]
            expert_matrix = self.expert_predictor.predict(seq_id, expert_index[i], self.layer_id)
            self.expert_prefetcher.prefetch_experts(self.layer_id, expert_matrix)
        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        return final_hidden_states, expert_mask


class MoeMLP(nn.Module):

    def __init__(self, hidden_dim: 'int', ffn_dim: 'int') ->None:
        super().__init__()
        self.linear_v = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.linear_1 = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.linear = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.act_fn = nn.GELU()

    def forward(self, hidden_states: 'torch.Tensor') ->torch.Tensor:
        current_hidden_states = self.act_fn(self.linear(hidden_states)) * self.linear_v(hidden_states)
        current_hidden_states = self.linear_1(current_hidden_states)
        return current_hidden_states


class SyncGrokMoeBlock(nn.Module):
    archer_config: 'ArcherConfig' = None
    layer_id: 'int' = None

    def __init__(self, hidden_dim: 'int', ffn_dim: 'int', num_experts: 'int', top_k: 'int'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([MoeMLP(hidden_dim, ffn_dim) for _ in range(self.num_experts)])
        self.archer_tracer = None
        self.archer_engine = None
        self.expert_tensor_ids: 'Dict[int, int]' = None

    def forward(self, hidden_states: 'torch.Tensor') ->Tuple[torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights
        expert_index = selected_experts.reshape(batch_size, sequence_length, self.top_k)
        for i in range(batch_size):
            seq_id = self.seq_id_list[i]
            expert_matrix = self.expert_predictor.predict(seq_id, expert_index[i], self.layer_id)
            self.expert_prefetcher.prefetch_experts(self.layer_id, expert_matrix)
        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] == 0:
                continue
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states)
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class SyncMixtralSparseMoeBlock(nn.Module):
    archer_config: 'ArcherConfig' = None
    layer_id: 'int' = None

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([MixtralBlockSparseTop2MLP(config) for _ in range(self.num_experts)])
        self.archer_tracer = None
        self.archer_engine = None
        self.expert_tensor_ids: 'Dict[int, int]' = None

    def forward(self, hidden_states: 'torch.Tensor') ->torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        num_tokens = batch_size * sequence_length
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights
        router_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
        routing_weights_mask = (routing_weights[:, :, None] * router_mask).permute(0, 2, 1)
        router_mask = router_mask.permute(0, 2, 1)
        router_mask = torch.logical_or(router_mask[:, :, 0], router_mask[:, :, 1])
        routing_weights_mask = torch.sum(routing_weights_mask, dim=-1)
        expert_index = selected_experts.reshape(batch_size, sequence_length, self.top_k)
        for i in range(batch_size):
            seq_id = self.seq_id_list[i]
            expert_matrix = self.expert_predictor.predict(seq_id, expert_index[i], self.layer_id)
            self.expert_prefetcher.prefetch_experts(self.layer_id, expert_matrix)
        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
        for expert_idx in range(self.num_experts):
            token_indices = router_mask[:, expert_idx]
            current_state = hidden_states[token_indices, :]
            if token_indices.any():
                current_hidden_states = self.experts[expert_idx](current_state) * routing_weights_mask[token_indices, expert_idx][:, None]
                final_hidden_states[token_indices, :] += current_hidden_states
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class ArcticRMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-06):
        """
        ArcticRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class ArcticRotaryEmbedding(nn.Module):

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


DEEPSPEED_LORA_CONFIG = 'deepspeed_lora'


DEEPSPEED_QUANTIZATION_CONFIG = 'deepspeed_quantization'


QUANTIZATION_CONFIG = 'ds_quantization_config'


USE_DEEPSPEED_MOE_ARG = 'use_deepspeed_moe_implementation'


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


def repeat_kv(hidden_states: 'torch.Tensor', n_rep: 'int') ->torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class ArcticAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: 'ArcticConfig', layer_idx: 'Optional[int]'=None, **kwargs):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(f'Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` when creating this class.')
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.use_deepspeed_implementation = USE_DEEPSPEED_MOE_ARG in kwargs and kwargs[USE_DEEPSPEED_MOE_ARG]
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(f'hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads}).')
        deepspeed_quantization = kwargs.get(DEEPSPEED_QUANTIZATION_CONFIG)
        deepspeed_lora_config = kwargs.get(DEEPSPEED_LORA_CONFIG)
        quantization_config = kwargs.get(QUANTIZATION_CONFIG, None)
        self.q_proj = get_arctic_linear(self.hidden_size, self.num_heads * self.head_dim, bias=False, use_deepspeed_implementation=self.use_deepspeed_implementation, ds_optimized_lora_config=deepspeed_lora_config, ds_optimized_quantization_config=quantization_config, ds_optimized_base_weight_sharding=True, dtype=torch.bfloat16)
        self.k_proj = get_arctic_linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, use_deepspeed_implementation=self.use_deepspeed_implementation, ds_optimized_lora_config=deepspeed_lora_config, ds_optimized_quantization_config=quantization_config, ds_optimized_base_weight_sharding=True, dtype=torch.bfloat16)
        self.v_proj = get_arctic_linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, use_deepspeed_implementation=self.use_deepspeed_implementation, ds_optimized_lora_config=deepspeed_lora_config, ds_optimized_quantization_config=quantization_config, ds_optimized_base_weight_sharding=True, dtype=torch.bfloat16)
        self.o_proj = get_arctic_linear(self.hidden_size, self.hidden_size, bias=False, use_deepspeed_implementation=self.use_deepspeed_implementation, ds_optimized_lora_config=deepspeed_lora_config, ds_optimized_quantization_config=quantization_config, ds_optimized_base_weight_sharding=True, dtype=torch.bfloat16)
        self.rotary_emb = ArcticRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta)

    def _shape(self, tensor: 'torch.Tensor', seq_len: 'int', bsz: 'int'):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]'=None, position_ids: 'Optional[torch.LongTensor]'=None, past_key_value: 'Optional[Cache]'=None, output_attentions: 'bool'=False, use_cache: 'bool'=False, **kwargs) ->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if 'padding_mask' in kwargs:
            warnings.warn('Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`')
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} for auto-regressive decoding with k/v caching, please make sure to initialize the attention class with a layer index.')
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None:
            cache_kwargs = {'sin': sin, 'cos': cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(f'Attention weights should be of size {bsz, self.num_heads, q_len, kv_seq_len}, but is {attn_weights.size()}')
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(f'Attention mask should be of size {bsz, 1, q_len, kv_seq_len}, but is {attention_mask.size()}')
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {bsz, self.num_heads, q_len, self.head_dim}, but is {attn_output.size()}')
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


class ArcticFlashAttention2(ArcticAttention):
    """
    Arctic flash attention module. This module inherits from `ArcticAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]'=None, position_ids: 'Optional[torch.LongTensor]'=None, past_key_value: 'Optional[Cache]'=None, output_attentions: 'bool'=False, use_cache: 'bool'=False, **kwargs):
        if 'padding_mask' in kwargs:
            warnings.warn('Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`')
            attention_mask = kwargs.pop('padding_mask')
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} for auto-regressive decoding with k/v caching, please make sure to initialize the attention class with a layer index.')
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        use_sliding_windows = _flash_supports_window_size and getattr(self.config, 'sliding_window', None) is not None and kv_seq_len > self.config.sliding_window
        if not _flash_supports_window_size:
            logger.warning_once('The current flash attention version does not support sliding window attention, for a more memory efficient implementation make sure to upgrade flash-attn library.')
        if past_key_value is not None:
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            if getattr(self.config, 'sliding_window', None) is not None and kv_seq_len > self.config.sliding_window and cache_has_contents:
                slicing_tokens = 1 - self.config.sliding_window
                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]
                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()
                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(f'past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got {past_key.shape}')
                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)
            cache_kwargs = {'sin': sin, 'cos': cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, '_pre_quantization_dtype'):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            logger.warning_once(f'The input hidden states seems to be silently casted in float32, this might be related to the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in {target_dtype}.')
            query_states = query_states
            key_states = key_states
            value_states = value_states
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        attn_output = self._flash_attention_forward(query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate, use_sliding_windows=use_sliding_windows)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None, use_sliding_windows=False):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.
        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            causal = self.is_causal and query_length != 1
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(query_states, key_states, value_states, attention_mask, query_length)
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(query_states, key_states, value_states, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_in_batch_q, max_seqlen_k=max_seqlen_in_batch_k, dropout_p=dropout, softmax_scale=softmax_scale, causal=causal)
            else:
                attn_output_unpad = flash_attn_varlen_func(query_states, key_states, value_states, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_in_batch_q, max_seqlen_k=max_seqlen_in_batch_k, dropout_p=dropout, softmax_scale=softmax_scale, causal=causal, window_size=(self.config.sliding_window, self.config.sliding_window))
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        elif not use_sliding_windows:
            attn_output = flash_attn_func(query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal)
        else:
            attn_output = flash_attn_func(query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal, window_size=(self.config.sliding_window, self.config.sliding_window))
        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len:]
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        if query_length == kv_seq_len:
            query_layer = index_first_axis(query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=query_layer.device)
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)
        return query_layer, key_layer, value_layer, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_in_batch_q, max_seqlen_in_batch_k)


class ArcticSdpaAttention(ArcticAttention):
    """
    Arctic attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `ArcticAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]'=None, position_ids: 'Optional[torch.LongTensor]'=None, past_key_value: 'Optional[Cache]'=None, output_attentions: 'bool'=False, use_cache: 'bool'=False) ->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            logger.warning_once('ArcticModel is using ArcticSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.')
            return super().forward(hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache)
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None:
            cache_kwargs = {'sin': sin, 'cos': cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(f'Attention mask should be of size {bsz, 1, q_len, kv_seq_len}, but is {attention_mask.size()}')
        if query_states.device.type == 'cuda' and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        attn_output = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=self.attention_dropout if self.training else 0.0, is_causal=self.is_causal and attention_mask is None and q_len > 1)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value


MOE_EXPERT_PARALLEL_SIZE_ARG = 'moe_expert_parallel_size'


ARCTIC_START_DOCSTRING = """
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`ArcticConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

