
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


from typing import Dict


from typing import List


from typing import Optional


from typing import Tuple


from typing import Union


import torch


import torch.nn as nn


import torch.nn.functional as F


from typing import Sequence


import copy


from torch.utils.data import Dataset


from typing import Any


import torch.utils.checkpoint


from torch import nn


from torch.nn import BCEWithLogitsLoss


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


import torch.distributed as dist


import random


import numpy as np


def rmsnorm_func(hidden_states, weight, variance_epsilon):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return weight * hidden_states


class LlamaRMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-06):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.register_buffer('variance_epsilon', torch.tensor(eps), persistent=False)

    def forward(self, hidden_states):
        return rmsnorm_func(hidden_states, self.weight, self.variance_epsilon)


class FlashRotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(self, dim: 'int', base=10000.0, interleaved=False, scale_base=None, scaling_factor=1.0, pos_idx_in_fp32=True, device=None):
        """
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
                of 1st half and 2nd half (GPT-NeoX style).
            pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1] are in fp32,
                otherwise they might be in lower precision.
                This option was added because previously (before 2023-07-02), when we construct
                the position indices, we use the dtype of self.inv_freq. In most cases this would
                be fp32, but if the model is trained in pure bf16 (not mixed precision), then
                self.inv_freq would be bf16, and the position indices are also in bf16.
                Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
                embeddings for some positions will coincide.
                To maintain compatibility with models previously trained in pure bf16,
                we add this option.
            scaling_factor: RotaryEmbedding extended with linear scaling.
        """
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.interleaved = interleaved
        self.scale_base = scale_base
        self.scaling_factor = scaling_factor
        scale = (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim) if scale_base is not None else None
        self.register_buffer('scale', scale)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _compute_inv_freq(self, device=None):
        return 1 / self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        if seqlen > self._seq_len_cached or self._cos_cached.device != device or self._cos_cached.dtype != dtype or self.training and self._cos_cached.is_inference():
            self._seq_len_cached = seqlen
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                t /= self.scaling_factor
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self.inv_freq
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                t /= self.scaling_factor
                inv_freq = self.inv_freq
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs)
                self._sin_cached = torch.sin(freqs)
            else:
                power = (torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2) / self.scale_base
                scale = self.scale ** power.unsqueeze(-1)
                self._cos_cached = torch.cos(freqs) * scale
                self._sin_cached = torch.sin(freqs) * scale
                self._cos_k_cached = torch.cos(freqs) / scale
                self._sin_k_cached = torch.sin(freqs) / scale

    def forward(self, q: 'torch.Tensor', k: 'torch.Tensor', seqlen_offset: 'int'=0, unpadded_lengths: 'Optional[Tuple[torch.Tensor]]'=None) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        q: (batch, seqlen, nheads, headdim)
        k: (batch, seqlen, nheads, headdim)
        seqlen_offset: can be used in generation where the qkv being passed in is only the last
        token in the batch.
        """
        if unpadded_lengths is not None:
            cu_seqlens, max_seqlen = unpadded_lengths
        else:
            cu_seqlens, max_seqlen = None, q.shape[1]
        self._update_cos_sin_cache(max_seqlen + seqlen_offset, device=q.device, dtype=q.dtype)
        if self.scale is None:
            return apply_rotary_emb_func(q, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:], self.interleaved, True, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen), apply_rotary_emb_func(k, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:], self.interleaved, True, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        else:
            assert False


class LlamaMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


@torch.jit.script
def repeat_kv(hidden_states: 'torch.Tensor', n_rep: 'int') ->torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    final_shape = list(hidden_states.shape[:-2]) + [-1] + [hidden_states.shape[-1]]
    expand_shape = [-1] * (len(hidden_states.shape) - 1) + [n_rep] + [-1]
    hidden_states = hidden_states.unsqueeze(-1).expand(expand_shape)
    return hidden_states.reshape(final_shape)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: 'LlamaConfig'):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(f'hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads}).')
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.register_buffer('norm_factor', torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)), persistent=False)
        if not getattr(self.config, 'rope_scaling', None):
            scaling_factor = 1
        else:
            scaling_type = self.config.rope_scaling['type']
            scaling_factor = self.config.rope_scaling['factor']
            assert scaling_type == 'linear'
        theta = getattr(self.config, 'rope_theta', 10000)
        self.rotary_emb = FlashRotaryEmbedding(self.head_dim, base=theta, interleaved=False, scaling_factor=scaling_factor)
        self.distributed_attn_func = flash_attn_kvpacked_func

    def _shape(self, tensor: 'torch.Tensor', seq_len: 'int', bsz: 'int'):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]'=None, position_ids: 'Optional[torch.LongTensor]'=None, past_key_value: 'Optional[Tuple[torch.Tensor]]'=None, output_attentions: 'bool'=False, use_cache: 'bool'=False, unpadded_lengths: 'Optional[Tuple[torch.Tensor]]'=None) ->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        h_size = hidden_states.size(-1)
        has_layer_past = past_key_value is not None
        if has_layer_past:
            past_kv = past_key_value[0]
            past_len = past_key_value[1]
        else:
            past_len = 0
        if position_ids is not None:
            past_len += position_ids.min()
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = q.view(*q.shape[:-1], self.num_heads, self.head_dim)
        k = k.view(*k.shape[:-1], self.num_key_value_heads, self.head_dim)
        v = v.view(*v.shape[:-1], self.num_key_value_heads, self.head_dim)
        q, k = self.rotary_emb(q, k, past_len, unpadded_lengths)
        kv = torch.stack([k, v], -3)
        kv = repeat_kv(kv, self.num_key_value_groups)
        if has_layer_past:
            new_len = past_len + q.size(1)
            if new_len > past_kv.size(1):
                past_kv = torch.cat([past_kv, torch.empty(hidden_states.size(0), 256, 2, kv.size(3), kv.size(4), dtype=kv.dtype, device=kv.device)], 1)
            past_kv[:, past_len:new_len] = kv
            kv = past_kv[:, :new_len]
        else:
            past_kv = kv
        past_key_value = (past_kv, past_len + q.size(1)) if use_cache else None
        if unpadded_lengths is not None:
            assert attention_mask is not None
            cu_seqlens, max_seqlen = unpadded_lengths
            attn_outputs = flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, dropout_p=0.0, softmax_scale=1.0 / self.norm_factor, causal=True, return_attn_probs=output_attentions)
        else:
            attn_outputs = flash_attn_kvpacked_func(q, kv, dropout_p=0.0, softmax_scale=1.0 / self.norm_factor, causal=True, return_attn_probs=output_attentions)
        attn_output = attn_outputs[0] if output_attentions else attn_outputs
        attn_output = attn_output.reshape(*attn_output.shape[:-2], h_size)
        attn_weights = attn_outputs[2] if output_attentions else None
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(nn.Module):

    def __init__(self, config: 'LlamaConfig'):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._fsdp_wrap = True

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]'=None, position_ids: 'Optional[torch.LongTensor]'=None, past_key_value: 'Optional[Tuple[torch.Tensor]]'=None, unpadded_lengths: 'Optional[Tuple[torch.Tensor]]'=None, output_attentions: 'Optional[bool]'=False, use_cache: 'Optional[bool]'=False) ->Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache, unpadded_lengths=unpadded_lengths)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = hidden_states,
        if output_attentions:
            outputs += self_attn_weights,
        if use_cache:
            outputs += present_key_value,
        return outputs


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (LlamaRMSNorm,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]
