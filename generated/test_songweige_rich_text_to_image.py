
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


import numpy as np


import torchvision


import math


import random


import time


from torchvision import transforms


from typing import Any


from typing import Dict


from typing import Optional


import torch.nn.functional as F


from torch import nn


from typing import Callable


from typing import Union


import collections


import torch.nn as nn


from functools import partial


import inspect


from typing import List


from typing import Tuple


import torch.utils.checkpoint


import matplotlib as mpl


import matplotlib.pyplot as plt


from sklearn.cluster import SpectralClustering


import torchvision.transforms as transforms


import abc


import torch.nn.functional as nnf


class AdaLayerNorm(nn.Module):
    """
    Norm layer modified to incorporate timestep embeddings.
    """

    def __init__(self, embedding_dim, num_embeddings):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep)))
        scale, shift = torch.chunk(emb, 2)
        x = self.norm(x) * (1 + scale) + shift
        return x


class AdaLayerNormZero(nn.Module):
    """
    Norm layer adaptive layer norm zero (adaLN-Zero).
    """

    def __init__(self, embedding_dim, num_embeddings):
        super().__init__()
        self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-06)

    def forward(self, x, timestep, class_labels, hidden_dtype=None):
        emb = self.linear(self.silu(self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AttnAddedKVProcessor:
    """
    Processor for performing attention-related computations with extra learnable key and value matrices for the text
    encoder.
    """

    def __call__(self, attn: 'Attention', hidden_states, encoder_hidden_states=None, attention_mask=None):
        residual = hidden_states
        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj)
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj)
        if not attn.only_cross_attention:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)
        else:
            key = encoder_hidden_states_key_proj
            value = encoder_hidden_states_value_proj
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
        hidden_states = hidden_states + residual
        return hidden_states


class AttnAddedKVProcessor2_0:
    """
    Processor for performing scaled dot-product attention (enabled by default if you're using PyTorch 2.0), with extra
    learnable key and value matrices for the text encoder.
    """

    def __init__(self):
        if not hasattr(F, 'scaled_dot_product_attention'):
            raise ImportError('AttnAddedKVProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.')

    def __call__(self, attn: 'Attention', hidden_states, encoder_hidden_states=None, attention_mask=None):
        residual = hidden_states
        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size, out_dim=4)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query, out_dim=4)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj, out_dim=4)
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj, out_dim=4)
        if not attn.only_cross_attention:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            key = attn.head_to_batch_dim(key, out_dim=4)
            value = attn.head_to_batch_dim(value, out_dim=4)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        else:
            key = encoder_hidden_states_key_proj
            value = encoder_hidden_states_value_proj
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, residual.shape[1])
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
        hidden_states = hidden_states + residual
        return hidden_states


class AttnProcessor:
    """
    Default processor for performing attention-related computations.
    """

    def __call__(self, attn: 'Attention', hidden_states, real_attn_probs=None, attn_weights=None, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        if real_attn_probs is None:
            attention_probs = attn.get_attention_scores(query, key, attention_mask, attn_weights=attn_weights)
        else:
            attention_probs = real_attn_probs
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        attention_probs_avg = attn.reshape_batch_dim_to_heads_and_average(attention_probs)
        return hidden_states, [attention_probs_avg, attention_probs]


class AttnProcessor2_0:
    """
    Default processor for performing attention-related computations.
    """

    def __call__(self, attn: 'Attention', hidden_states, real_attn_probs=None, attn_weights=None, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        if real_attn_probs is None:
            attention_probs = attn.get_attention_scores(query, key, attention_mask, attn_weights=attn_weights)
        else:
            attention_probs = real_attn_probs
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        attention_probs_avg = attn.reshape_batch_dim_to_heads_and_average(attention_probs)
        return hidden_states, [attention_probs_avg, attention_probs]


class CustomDiffusionAttnProcessor(nn.Module):
    """
    Processor for implementing attention for the Custom Diffusion method.

    Args:
        train_kv (`bool`, defaults to `True`):
            Whether to newly train the key and value matrices corresponding to the text features.
        train_q_out (`bool`, defaults to `True`):
            Whether to newly train query matrices corresponding to the latent image features.
        hidden_size (`int`, *optional*, defaults to `None`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*, defaults to `None`):
            The number of channels in the `encoder_hidden_states`.
        out_bias (`bool`, defaults to `True`):
            Whether to include the bias parameter in `train_q_out`.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
    """

    def __init__(self, train_kv=True, train_q_out=True, hidden_size=None, cross_attention_dim=None, out_bias=True, dropout=0.0):
        super().__init__()
        self.train_kv = train_kv
        self.train_q_out = train_q_out
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        if self.train_kv:
            self.to_k_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
            self.to_v_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        if self.train_q_out:
            self.to_q_custom_diffusion = nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_out_custom_diffusion = nn.ModuleList([])
            self.to_out_custom_diffusion.append(nn.Linear(hidden_size, hidden_size, bias=out_bias))
            self.to_out_custom_diffusion.append(nn.Dropout(dropout))

    def __call__(self, attn: 'Attention', hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if self.train_q_out:
            query = self.to_q_custom_diffusion(hidden_states)
        else:
            query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            crossattn = False
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        if self.train_kv:
            key = self.to_k_custom_diffusion(encoder_hidden_states)
            value = self.to_v_custom_diffusion(encoder_hidden_states)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        if crossattn:
            detach = torch.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :] * 0.0
            key = detach * key + (1 - detach) * key.detach()
            value = detach * value + (1 - detach) * value.detach()
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        if self.train_q_out:
            hidden_states = self.to_out_custom_diffusion[0](hidden_states)
            hidden_states = self.to_out_custom_diffusion[1](hidden_states)
        else:
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class CustomDiffusionXFormersAttnProcessor(nn.Module):
    """
    Processor for implementing memory efficient attention using xFormers for the Custom Diffusion method.

    Args:
    train_kv (`bool`, defaults to `True`):
        Whether to newly train the key and value matrices corresponding to the text features.
    train_q_out (`bool`, defaults to `True`):
        Whether to newly train query matrices corresponding to the latent image features.
    hidden_size (`int`, *optional*, defaults to `None`):
        The hidden size of the attention layer.
    cross_attention_dim (`int`, *optional*, defaults to `None`):
        The number of channels in the `encoder_hidden_states`.
    out_bias (`bool`, defaults to `True`):
        Whether to include the bias parameter in `train_q_out`.
    dropout (`float`, *optional*, defaults to 0.0):
        The dropout probability to use.
    attention_op (`Callable`, *optional*, defaults to `None`):
        The base
        [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to use
        as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best operator.
    """

    def __init__(self, train_kv=True, train_q_out=False, hidden_size=None, cross_attention_dim=None, out_bias=True, dropout=0.0, attention_op: 'Optional[Callable]'=None):
        super().__init__()
        self.train_kv = train_kv
        self.train_q_out = train_q_out
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.attention_op = attention_op
        if self.train_kv:
            self.to_k_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
            self.to_v_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        if self.train_q_out:
            self.to_q_custom_diffusion = nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_out_custom_diffusion = nn.ModuleList([])
            self.to_out_custom_diffusion.append(nn.Linear(hidden_size, hidden_size, bias=out_bias))
            self.to_out_custom_diffusion.append(nn.Dropout(dropout))

    def __call__(self, attn: 'Attention', hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if self.train_q_out:
            query = self.to_q_custom_diffusion(hidden_states)
        else:
            query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            crossattn = False
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        if self.train_kv:
            key = self.to_k_custom_diffusion(encoder_hidden_states)
            value = self.to_v_custom_diffusion(encoder_hidden_states)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        if crossattn:
            detach = torch.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :] * 0.0
            key = detach * key + (1 - detach) * key.detach()
            value = detach * value + (1 - detach) * value.detach()
        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale)
        hidden_states = hidden_states
        hidden_states = attn.batch_to_head_dim(hidden_states)
        if self.train_q_out:
            hidden_states = self.to_out_custom_diffusion[0](hidden_states)
            hidden_states = self.to_out_custom_diffusion[1](hidden_states)
        else:
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class LoRALinearLayer(nn.Module):

    def __init__(self, in_features, out_features, rank=4, network_alpha=None):
        super().__init__()
        if rank > min(in_features, out_features):
            raise ValueError(f'LoRA rank {rank} must be less or equal than {min(in_features, out_features)}')
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        self.network_alpha = network_alpha
        self.rank = rank
        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype
        down_hidden_states = self.down(hidden_states)
        up_hidden_states = self.up(down_hidden_states)
        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank
        return up_hidden_states


class LoRAAttnAddedKVProcessor(nn.Module):
    """
    Processor for implementing the LoRA attention mechanism with extra learnable key and value matrices for the text
    encoder.

    Args:
        hidden_size (`int`, *optional*):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*, defaults to `None`):
            The number of channels in the `encoder_hidden_states`.
        rank (`int`, defaults to 4):
            The dimension of the LoRA update matrices.

    """

    def __init__(self, hidden_size, cross_attention_dim=None, rank=4, network_alpha=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank
        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.add_k_proj_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.add_v_proj_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

    def __call__(self, attn: 'Attention', hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0):
        residual = hidden_states
        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        query = attn.head_to_batch_dim(query)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states) + scale * self.add_k_proj_lora(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states) + scale * self.add_v_proj_lora(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj)
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj)
        if not attn.only_cross_attention:
            key = attn.to_k(hidden_states) + scale * self.to_k_lora(hidden_states)
            value = attn.to_v(hidden_states) + scale * self.to_v_lora(hidden_states)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)
        else:
            key = encoder_hidden_states_key_proj
            value = encoder_hidden_states_value_proj
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
        hidden_states = hidden_states + residual
        return hidden_states


class LoRAAttnProcessor(nn.Module):
    """
    Processor for implementing the LoRA attention mechanism.

    Args:
        hidden_size (`int`, *optional*):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the `encoder_hidden_states`.
        rank (`int`, defaults to 4):
            The dimension of the LoRA update matrices.
        network_alpha (`int`, *optional*):
            Equivalent to `alpha` but it's usage is specific to Kohya (A1111) style LoRAs.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, rank=4, network_alpha=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank
        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

    def __call__(self, attn: 'Attention', hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0, temb=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        query = attn.head_to_batch_dim(query)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class LoRAAttnProcessor2_0(nn.Module):
    """
    Processor for implementing the LoRA attention mechanism using PyTorch 2.0's memory-efficient scaled dot-product
    attention.

    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the `encoder_hidden_states`.
        rank (`int`, defaults to 4):
            The dimension of the LoRA update matrices.
        network_alpha (`int`, *optional*):
            Equivalent to `alpha` but it's usage is specific to Kohya (A1111) style LoRAs.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, rank=4, network_alpha=None):
        super().__init__()
        if not hasattr(F, 'scaled_dot_product_attention'):
            raise ImportError('AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.')
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank
        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

    def __call__(self, attn: 'Attention', hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        inner_dim = hidden_states.shape[-1]
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(encoder_hidden_states)
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class LoRAXFormersAttnProcessor(nn.Module):
    """
    Processor for implementing the LoRA attention mechanism with memory efficient attention using xFormers.

    Args:
        hidden_size (`int`, *optional*):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the `encoder_hidden_states`.
        rank (`int`, defaults to 4):
            The dimension of the LoRA update matrices.
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
        network_alpha (`int`, *optional*):
            Equivalent to `alpha` but it's usage is specific to Kohya (A1111) style LoRAs.

    """

    def __init__(self, hidden_size, cross_attention_dim, rank=4, attention_op: 'Optional[Callable]'=None, network_alpha=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank
        self.attention_op = attention_op
        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

    def __call__(self, attn: 'Attention', hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0, temb=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        query = attn.head_to_batch_dim(query).contiguous()
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(encoder_hidden_states)
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class SlicedAttnAddedKVProcessor:
    """
    Processor for implementing sliced attention with extra learnable key and value matrices for the text encoder.

    Args:
        slice_size (`int`, *optional*):
            The number of steps to compute attention. Uses as many slices as `attention_head_dim // slice_size`, and
            `attention_head_dim` must be a multiple of the `slice_size`.
    """

    def __init__(self, slice_size):
        self.slice_size = slice_size

    def __call__(self, attn: "'Attention'", hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        dim = query.shape[-1]
        query = attn.head_to_batch_dim(query)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj)
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj)
        if not attn.only_cross_attention:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)
        else:
            key = encoder_hidden_states_key_proj
            value = encoder_hidden_states_value_proj
        batch_size_attention, query_tokens, _ = query.shape
        hidden_states = torch.zeros((batch_size_attention, query_tokens, dim // attn.heads), device=query.device, dtype=query.dtype)
        for i in range(batch_size_attention // self.slice_size):
            start_idx = i * self.slice_size
            end_idx = (i + 1) * self.slice_size
            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]
            attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None
            attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)
            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])
            hidden_states[start_idx:end_idx] = attn_slice
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
        hidden_states = hidden_states + residual
        return hidden_states


class SlicedAttnProcessor:
    """
    Processor for implementing sliced attention.

    Args:
        slice_size (`int`, *optional*):
            The number of steps to compute attention. Uses as many slices as `attention_head_dim // slice_size`, and
            `attention_head_dim` must be a multiple of the `slice_size`.
    """

    def __init__(self, slice_size):
        self.slice_size = slice_size

    def __call__(self, attn: 'Attention', hidden_states, encoder_hidden_states=None, attention_mask=None):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        dim = query.shape[-1]
        query = attn.head_to_batch_dim(query)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        batch_size_attention, query_tokens, _ = query.shape
        hidden_states = torch.zeros((batch_size_attention, query_tokens, dim // attn.heads), device=query.device, dtype=query.dtype)
        for i in range(batch_size_attention // self.slice_size):
            start_idx = i * self.slice_size
            end_idx = (i + 1) * self.slice_size
            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]
            attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None
            attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)
            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])
            hidden_states[start_idx:end_idx] = attn_slice
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class SpatialNorm(nn.Module):
    """
    Spatially conditioned normalization as defined in https://arxiv.org/abs/2209.09002
    """

    def __init__(self, f_channels, zq_channels):
        super().__init__()
        self.norm_layer = nn.GroupNorm(num_channels=f_channels, num_groups=32, eps=1e-06, affine=True)
        self.conv_y = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, f, zq):
        f_size = f.shape[-2:]
        zq = F.interpolate(zq, size=f_size, mode='nearest')
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f


class XFormersAttnAddedKVProcessor:
    """
    Processor for implementing memory efficient attention using xFormers.

    Args:
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(self, attention_op: 'Optional[Callable]'=None):
        self.attention_op = attention_op

    def __call__(self, attn: 'Attention', hidden_states, encoder_hidden_states=None, attention_mask=None):
        residual = hidden_states
        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj)
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj)
        if not attn.only_cross_attention:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)
        else:
            key = encoder_hidden_states_key_proj
            value = encoder_hidden_states_value_proj
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale)
        hidden_states = hidden_states
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
        hidden_states = hidden_states + residual
        return hidden_states


class XFormersAttnProcessor:
    """
    Processor for implementing memory efficient attention using xFormers.

    Args:
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(self, attention_op: 'Optional[Callable]'=None):
        self.attention_op = attention_op

    def __call__(self, attn: 'Attention', hidden_states: 'torch.FloatTensor', encoder_hidden_states: 'Optional[torch.FloatTensor]'=None, attention_mask: 'Optional[torch.FloatTensor]'=None, temb: 'Optional[torch.FloatTensor]'=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, key_tokens, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale)
        hidden_states = hidden_states
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class ApproximateGELU(nn.Module):
    """
    The approximate form of Gaussian Error Linear Unit (GELU)

    For more details, see section 2: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, dim_in: 'int', dim_out: 'int'):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)


class GEGLU(nn.Module):
    """
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: 'int', dim_out: 'int'):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def gelu(self, gate):
        if gate.device.type != 'mps':
            return F.gelu(gate)
        return F.gelu(gate.to(dtype=torch.float32))

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class GELU(nn.Module):
    """
    GELU activation function with tanh approximation support with `approximate="tanh"`.
    """

    def __init__(self, dim_in: 'int', dim_out: 'int', approximate: 'str'='none'):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)
        self.approximate = approximate

    def gelu(self, gate):
        if gate.device.type != 'mps':
            return F.gelu(gate, approximate=self.approximate)
        return F.gelu(gate.to(dtype=torch.float32), approximate=self.approximate)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    """
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
    """

    def __init__(self, dim: 'int', dim_out: 'Optional[int]'=None, mult: 'int'=4, dropout: 'float'=0.0, activation_fn: 'str'='geglu', final_dropout: 'bool'=False):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        if activation_fn == 'gelu':
            act_fn = GELU(dim, inner_dim)
        if activation_fn == 'gelu-approximate':
            act_fn = GELU(dim, inner_dim, approximate='tanh')
        elif activation_fn == 'geglu':
            act_fn = GEGLU(dim, inner_dim)
        elif activation_fn == 'geglu-approximate':
            act_fn = ApproximateGELU(dim, inner_dim)
        self.net = nn.ModuleList([])
        self.net.append(act_fn)
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out))
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class AdaGroupNorm(nn.Module):
    """
    GroupNorm layer modified to incorporate timestep embeddings.
    """

    def __init__(self, embedding_dim: 'int', out_dim: 'int', num_groups: 'int', act_fn: 'Optional[str]'=None, eps: 'float'=1e-05):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        if act_fn is None:
            self.act = None
        else:
            self.act = get_activation(act_fn)
        self.linear = nn.Linear(embedding_dim, out_dim * 2)

    def forward(self, x, emb):
        if self.act:
            emb = self.act(emb)
        emb = self.linear(emb)
        emb = emb[:, :, None, None]
        scale, shift = emb.chunk(2, dim=1)
        x = F.group_norm(x, self.num_groups, eps=self.eps)
        x = x * (1 + scale) + shift
        return x


class DualTransformer2DModel(nn.Module):
    """
    Dual transformer wrapper that combines two `Transformer2DModel`s for mixed inference.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            Pass if the input is continuous. The number of channels in the input and output.
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.1): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of encoder_hidden_states dimensions to use.
        sample_size (`int`, *optional*): Pass if the input is discrete. The width of the latent images.
            Note that this is fixed at training time as it is used for learning a number of position embeddings. See
            `ImagePositionalEmbeddings`.
        num_vector_embeds (`int`, *optional*):
            Pass if the input is discrete. The number of classes of the vector embeddings of the latent pixels.
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*): Pass if at least one of the norm_layers is `AdaLayerNorm`.
            The number of diffusion steps used during training. Note that this is fixed at training time as it is used
            to learn a number of embeddings that are added to the hidden states. During inference, you can denoise for
            up to but not more than steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the TransformerBlocks' attention should contain a bias parameter.
    """

    def __init__(self, num_attention_heads: 'int'=16, attention_head_dim: 'int'=88, in_channels: 'Optional[int]'=None, num_layers: 'int'=1, dropout: 'float'=0.0, norm_num_groups: 'int'=32, cross_attention_dim: 'Optional[int]'=None, attention_bias: 'bool'=False, sample_size: 'Optional[int]'=None, num_vector_embeds: 'Optional[int]'=None, activation_fn: 'str'='geglu', num_embeds_ada_norm: 'Optional[int]'=None):
        super().__init__()
        self.transformers = nn.ModuleList([Transformer2DModel(num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim, in_channels=in_channels, num_layers=num_layers, dropout=dropout, norm_num_groups=norm_num_groups, cross_attention_dim=cross_attention_dim, attention_bias=attention_bias, sample_size=sample_size, num_vector_embeds=num_vector_embeds, activation_fn=activation_fn, num_embeds_ada_norm=num_embeds_ada_norm) for _ in range(2)])
        self.mix_ratio = 0.5
        self.condition_lengths = [77, 257]
        self.transformer_index_for_condition = [1, 0]

    def forward(self, hidden_states, encoder_hidden_states, timestep=None, attention_mask=None, cross_attention_kwargs=None, return_dict: 'bool'=True):
        """
        Args:
            hidden_states ( When discrete, `torch.LongTensor` of shape `(batch size, num latent pixels)`.
                When continuous, `torch.FloatTensor` of shape `(batch size, channel, height, width)`): Input
                hidden_states
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.long`, *optional*):
                Optional timestep to be applied as an embedding in AdaLayerNorm's. Used to indicate denoising step.
            attention_mask (`torch.FloatTensor`, *optional*):
                Optional attention mask to be applied in Attention
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.transformer_2d.Transformer2DModelOutput`] or `tuple`:
            [`~models.transformer_2d.Transformer2DModelOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        input_states = hidden_states
        encoded_states = []
        tokens_start = 0
        for i in range(2):
            condition_state = encoder_hidden_states[:, tokens_start:tokens_start + self.condition_lengths[i]]
            transformer_index = self.transformer_index_for_condition[i]
            encoded_state = self.transformers[transformer_index](input_states, encoder_hidden_states=condition_state, timestep=timestep, cross_attention_kwargs=cross_attention_kwargs, return_dict=False)[0]
            encoded_states.append(encoded_state - input_states)
            tokens_start += self.condition_lengths[i]
        output_states = encoded_states[0] * self.mix_ratio + encoded_states[1] * (1 - self.mix_ratio)
        output_states = output_states + input_states
        if not return_dict:
            return output_states,
        return Transformer2DModelOutput(sample=output_states)


CrossAttentionLayers = ['down_blocks.1.attentions.0.transformer_blocks.0.attn2', 'down_blocks.2.attentions.0.transformer_blocks.0.attn2', 'down_blocks.2.attentions.1.transformer_blocks.0.attn2', 'mid_block.attentions.0.transformer_blocks.0.attn2', 'up_blocks.1.attentions.0.transformer_blocks.0.attn2', 'up_blocks.1.attentions.1.transformer_blocks.0.attn2', 'up_blocks.1.attentions.2.transformer_blocks.0.attn2', 'up_blocks.2.attentions.1.transformer_blocks.0.attn2']


SelfAttentionLayers = ['down_blocks.0.attentions.0.transformer_blocks.0.attn1', 'down_blocks.0.attentions.1.transformer_blocks.0.attn1', 'down_blocks.1.attentions.0.transformer_blocks.0.attn1', 'down_blocks.1.attentions.1.transformer_blocks.0.attn1', 'down_blocks.2.attentions.0.transformer_blocks.0.attn1', 'down_blocks.2.attentions.1.transformer_blocks.0.attn1', 'mid_block.attentions.0.transformer_blocks.0.attn1', 'up_blocks.1.attentions.0.transformer_blocks.0.attn1', 'up_blocks.1.attentions.1.transformer_blocks.0.attn1', 'up_blocks.1.attentions.2.transformer_blocks.0.attn1', 'up_blocks.2.attentions.0.transformer_blocks.0.attn1', 'up_blocks.2.attentions.1.transformer_blocks.0.attn1', 'up_blocks.2.attentions.2.transformer_blocks.0.attn1', 'up_blocks.3.attentions.0.transformer_blocks.0.attn1', 'up_blocks.3.attentions.1.transformer_blocks.0.attn1', 'up_blocks.3.attentions.2.transformer_blocks.0.attn1']


AttentionProcessor = Union[AttnProcessor, AttnProcessor2_0, XFormersAttnProcessor, SlicedAttnProcessor, AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0, XFormersAttnAddedKVProcessor, LoRAAttnProcessor, LoRAXFormersAttnProcessor, LoRAAttnProcessor2_0, LoRAAttnAddedKVProcessor, CustomDiffusionAttnProcessor, CustomDiffusionXFormersAttnProcessor]


class Downsample2D(nn.Module):
    """A 2D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
    """

    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, name='conv'):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name
        if use_conv:
            conv = nn.Conv2d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)
        if name == 'conv':
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == 'Conv2d_0':
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            pad = 0, 1, 0, 1
            hidden_states = F.pad(hidden_states, pad, mode='constant', value=0)
        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)
        return hidden_states


class Upsample2D(nn.Module):
    """A 2D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
    """

    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name='conv'):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        conv = None
        if use_conv_transpose:
            conv = nn.ConvTranspose2d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)
        if name == 'conv':
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(hidden_states)
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode='nearest')
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode='nearest')
        if dtype == torch.bfloat16:
            hidden_states = hidden_states
        if self.use_conv:
            if self.name == 'conv':
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)
        return hidden_states


def upfirdn2d_native(tensor, kernel, up=1, down=1, pad=(0, 0)):
    up_x = up_y = up
    down_x = down_y = down
    pad_x0 = pad_y0 = pad[0]
    pad_x1 = pad_y1 = pad[1]
    _, channel, in_h, in_w = tensor.shape
    tensor = tensor.reshape(-1, in_h, in_w, 1)
    _, in_h, in_w, minor = tensor.shape
    kernel_h, kernel_w = kernel.shape
    out = tensor.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)
    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out
    out = out[:, max(-pad_y0, 0):out.shape[1] - max(-pad_y1, 0), max(-pad_x0, 0):out.shape[2] - max(-pad_x1, 0), :]
    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1, in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
    return out.view(-1, channel, out_h, out_w)


def downsample_2d(hidden_states, kernel=None, factor=2, gain=1):
    """Downsample2D a batch of 2D images with the given filter.
    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]` and downsamples each image with the
    given filter. The filter is normalized so that if the input pixels are constant, they will be scaled by the
    specified `gain`. Pixels outside the image are assumed to be zero, and the filter is padded with zeros so that its
    shape is a multiple of the downsampling factor.

    Args:
        hidden_states: Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
        kernel: FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to average pooling.
        factor: Integer downsampling factor (default: 2).
        gain: Scaling factor for signal magnitude (default: 1.0).

    Returns:
        output: Tensor of the shape `[N, C, H // factor, W // factor]`
    """
    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor
    kernel = torch.tensor(kernel, dtype=torch.float32)
    if kernel.ndim == 1:
        kernel = torch.outer(kernel, kernel)
    kernel /= torch.sum(kernel)
    kernel = kernel * gain
    pad_value = kernel.shape[0] - factor
    output = upfirdn2d_native(hidden_states, kernel, down=factor, pad=((pad_value + 1) // 2, pad_value // 2))
    return output


def upsample_2d(hidden_states, kernel=None, factor=2, gain=1):
    """Upsample2D a batch of 2D images with the given filter.
    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]` and upsamples each image with the given
    filter. The filter is normalized so that if the input pixels are constant, they will be scaled by the specified
    `gain`. Pixels outside the image are assumed to be zero, and the filter is padded with zeros so that its shape is
    a: multiple of the upsampling factor.

    Args:
        hidden_states: Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
        kernel: FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to nearest-neighbor upsampling.
        factor: Integer upsampling factor (default: 2).
        gain: Scaling factor for signal magnitude (default: 1.0).

    Returns:
        output: Tensor of the shape `[N, C, H * factor, W * factor]`
    """
    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor
    kernel = torch.tensor(kernel, dtype=torch.float32)
    if kernel.ndim == 1:
        kernel = torch.outer(kernel, kernel)
    kernel /= torch.sum(kernel)
    kernel = kernel * (gain * factor ** 2)
    pad_value = kernel.shape[0] - factor
    output = upfirdn2d_native(hidden_states, kernel, up=factor, pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2))
    return output


class ResnetBlock2D(nn.Module):
    """
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        groups_out (`int`, *optional*, default to None):
            The number of groups to use for the second normalization layer. if set to None, same as `groups`.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
        non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
        time_embedding_norm (`str`, *optional*, default to `"default"` ): Time scale shift config.
            By default, apply timestep embedding conditioning with a simple shift mechanism. Choose "scale_shift" or
            "ada_group" for a stronger conditioning with scale and shift.
        kernel (`torch.FloatTensor`, optional, default to None): FIR filter, see
            [`~models.resnet.FirUpsample2D`] and [`~models.resnet.FirDownsample2D`].
        output_scale_factor (`float`, *optional*, default to be `1.0`): the scale factor to use for the output.
        use_in_shortcut (`bool`, *optional*, default to `True`):
            If `True`, add a 1x1 nn.conv2d layer for skip-connection.
        up (`bool`, *optional*, default to `False`): If `True`, add an upsample layer.
        down (`bool`, *optional*, default to `False`): If `True`, add a downsample layer.
        conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
            `conv_shortcut` output.
        conv_2d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
            If None, same as `out_channels`.
    """

    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, temb_channels=512, groups=32, groups_out=None, pre_norm=True, eps=1e-06, non_linearity='swish', skip_time_act=False, time_embedding_norm='default', kernel=None, output_scale_factor=1.0, use_in_shortcut=None, up=False, down=False, conv_shortcut_bias: bool=True, conv_2d_out_channels: Optional[int]=None):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act
        if groups_out is None:
            groups_out = groups
        if self.time_embedding_norm == 'ada_group':
            self.norm1 = AdaGroupNorm(temb_channels, in_channels, groups, eps=eps)
        elif self.time_embedding_norm == 'spatial':
            self.norm1 = SpatialNorm(in_channels, temb_channels)
        else:
            self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels is not None:
            if self.time_embedding_norm == 'default':
                self.time_emb_proj = torch.nn.Linear(temb_channels, out_channels)
            elif self.time_embedding_norm == 'scale_shift':
                self.time_emb_proj = torch.nn.Linear(temb_channels, 2 * out_channels)
            elif self.time_embedding_norm == 'ada_group' or self.time_embedding_norm == 'spatial':
                self.time_emb_proj = None
            else:
                raise ValueError(f'unknown time_embedding_norm : {self.time_embedding_norm} ')
        else:
            self.time_emb_proj = None
        if self.time_embedding_norm == 'ada_group':
            self.norm2 = AdaGroupNorm(temb_channels, out_channels, groups_out, eps=eps)
        elif self.time_embedding_norm == 'spatial':
            self.norm2 = SpatialNorm(out_channels, temb_channels)
        else:
            self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = torch.nn.Conv2d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = get_activation(non_linearity)
        self.upsample = self.downsample = None
        if self.up:
            if kernel == 'fir':
                fir_kernel = 1, 3, 3, 1
                self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
            elif kernel == 'sde_vp':
                self.upsample = partial(F.interpolate, scale_factor=2.0, mode='nearest')
            else:
                self.upsample = Upsample2D(in_channels, use_conv=False)
        elif self.down:
            if kernel == 'fir':
                fir_kernel = 1, 3, 3, 1
                self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
            elif kernel == 'sde_vp':
                self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
            else:
                self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name='op')
        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut
        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = torch.nn.Conv2d(in_channels, conv_2d_out_channels, kernel_size=1, stride=1, padding=0, bias=conv_shortcut_bias)

    def forward(self, input_tensor, temb, inject_states=None):
        hidden_states = input_tensor
        if self.time_embedding_norm == 'ada_group' or self.time_embedding_norm == 'spatial':
            hidden_states = self.norm1(hidden_states, temb)
        else:
            hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        if self.upsample is not None:
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)
        hidden_states = self.conv1(hidden_states)
        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]
        if temb is not None and self.time_embedding_norm == 'default':
            hidden_states = hidden_states + temb
        if self.time_embedding_norm == 'ada_group' or self.time_embedding_norm == 'spatial':
            hidden_states = self.norm2(hidden_states, temb)
        else:
            hidden_states = self.norm2(hidden_states)
        if temb is not None and self.time_embedding_norm == 'scale_shift':
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)
        if inject_states is not None:
            output_tensor = (input_tensor + inject_states) / self.output_scale_factor
        else:
            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        return output_tensor, hidden_states


class CrossAttnDownBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, transformer_layers_per_block: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, num_attention_heads=1, cross_attention_dim=1280, output_scale_factor=1.0, downsample_padding=1, add_downsample=True, dual_cross_attention=False, use_linear_projection=False, only_cross_attention=False, upcast_attention=False):
        super().__init__()
        resnets = []
        attentions = []
        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
            if not dual_cross_attention:
                attentions.append(Transformer2DModel(num_attention_heads, out_channels // num_attention_heads, in_channels=out_channels, num_layers=transformer_layers_per_block, cross_attention_dim=cross_attention_dim, norm_num_groups=resnet_groups, use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention, upcast_attention=upcast_attention))
            else:
                attentions.append(DualTransformer2DModel(num_attention_heads, out_channels // num_attention_heads, in_channels=out_channels, num_layers=1, cross_attention_dim=cross_attention_dim, norm_num_groups=resnet_groups))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name='op')])
        else:
            self.downsamplers = None
        self.gradient_checkpointing = False

    def forward(self, hidden_states: 'torch.FloatTensor', temb: 'Optional[torch.FloatTensor]'=None, encoder_hidden_states: 'Optional[torch.FloatTensor]'=None, attention_mask: 'Optional[torch.FloatTensor]'=None, cross_attention_kwargs: 'Optional[Dict[str, Any]]'=None, encoder_attention_mask: 'Optional[torch.FloatTensor]'=None):
        output_states = ()
        for resnet, attn in zip(self.resnets, self.attentions):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):

                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward
                ckpt_kwargs: 'Dict[str, Any]' = {'use_reentrant': False} if is_torch_version('>=', '1.11.0') else {}
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb, **ckpt_kwargs)
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(attn, return_dict=False), hidden_states, encoder_hidden_states, None, None, cross_attention_kwargs, attention_mask, encoder_attention_mask, **ckpt_kwargs)[0]
            else:
                hidden_states, _ = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs, attention_mask=attention_mask, encoder_attention_mask=encoder_attention_mask, return_dict=False)[0]
            output_states = output_states + (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)
        return hidden_states, output_states


class CrossAttnUpBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', prev_output_channel: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, transformer_layers_per_block: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, num_attention_heads=1, cross_attention_dim=1280, output_scale_factor=1.0, add_upsample=True, dual_cross_attention=False, use_linear_projection=False, only_cross_attention=False, upcast_attention=False):
        super().__init__()
        resnets = []
        attentions = []
        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        for i in range(num_layers):
            res_skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=resnet_in_channels + res_skip_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
            if not dual_cross_attention:
                attentions.append(Transformer2DModel(num_attention_heads, out_channels // num_attention_heads, in_channels=out_channels, num_layers=transformer_layers_per_block, cross_attention_dim=cross_attention_dim, norm_num_groups=resnet_groups, use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention, upcast_attention=upcast_attention))
            else:
                attentions.append(DualTransformer2DModel(num_attention_heads, out_channels // num_attention_heads, in_channels=out_channels, num_layers=1, cross_attention_dim=cross_attention_dim, norm_num_groups=resnet_groups))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None
        self.gradient_checkpointing = False

    def forward(self, hidden_states: 'torch.FloatTensor', res_hidden_states_tuple: 'Tuple[torch.FloatTensor, ...]', temb: 'Optional[torch.FloatTensor]'=None, encoder_hidden_states: 'Optional[torch.FloatTensor]'=None, cross_attention_kwargs: 'Optional[Dict[str, Any]]'=None, upsample_size: 'Optional[int]'=None, attention_mask: 'Optional[torch.FloatTensor]'=None, encoder_attention_mask: 'Optional[torch.FloatTensor]'=None):
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):

                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward
                ckpt_kwargs: 'Dict[str, Any]' = {'use_reentrant': False} if is_torch_version('>=', '1.11.0') else {}
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb, **ckpt_kwargs)
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(attn, return_dict=False), hidden_states, encoder_hidden_states, None, None, cross_attention_kwargs, attention_mask, encoder_attention_mask, **ckpt_kwargs)[0]
            else:
                hidden_states, _ = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs, attention_mask=attention_mask, encoder_attention_mask=encoder_attention_mask, return_dict=False)[0]
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
        return hidden_states


class DownBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, output_scale_factor=1.0, add_downsample=True, downsample_padding=1):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name='op')])
        else:
            self.downsamplers = None
        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None):
        output_states = ()
        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                if is_torch_version('>=', '1.11.0'):
                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb, use_reentrant=False)
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states, _ = resnet(hidden_states, temb)
            output_states = output_states + (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)
        return hidden_states, output_states


class UNetMidBlock2DCrossAttn(nn.Module):

    def __init__(self, in_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, transformer_layers_per_block: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, num_attention_heads=1, output_scale_factor=1.0, cross_attention_dim=1280, dual_cross_attention=False, use_linear_projection=False, upcast_attention=False):
        super().__init__()
        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        resnets = [ResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm)]
        attentions = []
        for _ in range(num_layers):
            if not dual_cross_attention:
                attentions.append(Transformer2DModel(num_attention_heads, in_channels // num_attention_heads, in_channels=in_channels, num_layers=transformer_layers_per_block, cross_attention_dim=cross_attention_dim, norm_num_groups=resnet_groups, use_linear_projection=use_linear_projection, upcast_attention=upcast_attention))
            else:
                attentions.append(DualTransformer2DModel(num_attention_heads, in_channels // num_attention_heads, in_channels=in_channels, num_layers=1, cross_attention_dim=cross_attention_dim, norm_num_groups=resnet_groups))
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: 'torch.FloatTensor', temb: 'Optional[torch.FloatTensor]'=None, encoder_hidden_states: 'Optional[torch.FloatTensor]'=None, attention_mask: 'Optional[torch.FloatTensor]'=None, cross_attention_kwargs: 'Optional[Dict[str, Any]]'=None, encoder_attention_mask: 'Optional[torch.FloatTensor]'=None) ->torch.FloatTensor:
        hidden_states, _ = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs, attention_mask=attention_mask, encoder_attention_mask=encoder_attention_mask, return_dict=False)[0]
            hidden_states, _ = resnet(hidden_states, temb)
        return hidden_states


class UNetMidBlock2DSimpleCrossAttn(nn.Module):

    def __init__(self, in_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, attention_head_dim=1, output_scale_factor=1.0, cross_attention_dim=1280, skip_time_act=False, only_cross_attention=False, cross_attention_norm=None):
        super().__init__()
        self.has_cross_attention = True
        self.attention_head_dim = attention_head_dim
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.num_heads = in_channels // self.attention_head_dim
        resnets = [ResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm, skip_time_act=skip_time_act)]
        attentions = []
        for _ in range(num_layers):
            processor = AttnAddedKVProcessor2_0() if hasattr(F, 'scaled_dot_product_attention') else AttnAddedKVProcessor()
            attentions.append(Attention(query_dim=in_channels, cross_attention_dim=in_channels, heads=self.num_heads, dim_head=self.attention_head_dim, added_kv_proj_dim=cross_attention_dim, norm_num_groups=resnet_groups, bias=True, upcast_softmax=True, only_cross_attention=only_cross_attention, cross_attention_norm=cross_attention_norm, processor=processor))
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm, skip_time_act=skip_time_act))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: 'torch.FloatTensor', temb: 'Optional[torch.FloatTensor]'=None, encoder_hidden_states: 'Optional[torch.FloatTensor]'=None, attention_mask: 'Optional[torch.FloatTensor]'=None, cross_attention_kwargs: 'Optional[Dict[str, Any]]'=None, encoder_attention_mask: 'Optional[torch.FloatTensor]'=None):
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        if attention_mask is None:
            mask = None if encoder_hidden_states is None else encoder_attention_mask
        else:
            mask = attention_mask
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=mask, **cross_attention_kwargs)
            hidden_states, _ = resnet(hidden_states, temb)
        return hidden_states


class UpBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', prev_output_channel: 'int', out_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, output_scale_factor=1.0, add_upsample=True):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            res_skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=resnet_in_channels + res_skip_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None
        self.gradient_checkpointing = False

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                if is_torch_version('>=', '1.11.0'):
                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb, use_reentrant=False)
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states, _ = resnet(hidden_states, temb)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
        return hidden_states


class AttnDownBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, attention_head_dim=1, output_scale_factor=1.0, downsample_padding=1, downsample_type='conv'):
        super().__init__()
        resnets = []
        attentions = []
        self.downsample_type = downsample_type
        if attention_head_dim is None:
            logger.warn(f'It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}.')
            attention_head_dim = out_channels
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
            attentions.append(Attention(out_channels, heads=out_channels // attention_head_dim, dim_head=attention_head_dim, rescale_output_factor=output_scale_factor, eps=resnet_eps, norm_num_groups=resnet_groups, residual_connection=True, bias=True, upcast_softmax=True, _from_deprecated_attn_block=True))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if downsample_type == 'conv':
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name='op')])
        elif downsample_type == 'resnet':
            self.downsamplers = nn.ModuleList([ResnetBlock2D(in_channels=out_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm, down=True)])
        else:
            self.downsamplers = None

    def forward(self, hidden_states, temb=None, upsample_size=None):
        output_states = ()
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states, _ = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)
            output_states = output_states + (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                if self.downsample_type == 'resnet':
                    hidden_states = downsampler(hidden_states, temb=temb)
                else:
                    hidden_states = downsampler(hidden_states)
            output_states += hidden_states,
        return hidden_states, output_states


class AttnDownEncoderBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, attention_head_dim=1, output_scale_factor=1.0, add_downsample=True, downsample_padding=1):
        super().__init__()
        resnets = []
        attentions = []
        if attention_head_dim is None:
            logger.warn(f'It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}.')
            attention_head_dim = out_channels
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, temb_channels=None, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
            attentions.append(Attention(out_channels, heads=out_channels // attention_head_dim, dim_head=attention_head_dim, rescale_output_factor=output_scale_factor, eps=resnet_eps, norm_num_groups=resnet_groups, residual_connection=True, bias=True, upcast_softmax=True, _from_deprecated_attn_block=True))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name='op')])
        else:
            self.downsamplers = None

    def forward(self, hidden_states):
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states, _ = resnet(hidden_states, temb=None)
            hidden_states = attn(hidden_states)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


class FirDownsample2D(nn.Module):
    """A 2D FIR downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        fir_kernel (`tuple`, default `(1, 3, 3, 1)`):
            kernel for the FIR filter.
    """

    def __init__(self, channels=None, out_channels=None, use_conv=False, fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_channels = out_channels if out_channels else channels
        if use_conv:
            self.Conv2d_0 = nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.fir_kernel = fir_kernel
        self.use_conv = use_conv
        self.out_channels = out_channels

    def _downsample_2d(self, hidden_states, weight=None, kernel=None, factor=2, gain=1):
        """Fused `Conv2d()` followed by `downsample_2d()`.
        Padding is performed only once at the beginning, not between the operations. The fused op is considerably more
        efficient than performing the same calculation using standard TensorFlow ops. It supports gradients of
        arbitrary order.

        Args:
            hidden_states: Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
            weight:
                Weight tensor of the shape `[filterH, filterW, inChannels, outChannels]`. Grouped convolution can be
                performed by `inChannels = x.shape[0] // numGroups`.
            kernel: FIR filter of the shape `[firH, firW]` or `[firN]` (separable). The default is `[1] *
            factor`, which corresponds to average pooling.
            factor: Integer downsampling factor (default: 2).
            gain: Scaling factor for signal magnitude (default: 1.0).

        Returns:
            output: Tensor of the shape `[N, C, H // factor, W // factor]` or `[N, H // factor, W // factor, C]`, and
            same datatype as `x`.
        """
        assert isinstance(factor, int) and factor >= 1
        if kernel is None:
            kernel = [1] * factor
        kernel = torch.tensor(kernel, dtype=torch.float32)
        if kernel.ndim == 1:
            kernel = torch.outer(kernel, kernel)
        kernel /= torch.sum(kernel)
        kernel = kernel * gain
        if self.use_conv:
            _, _, convH, convW = weight.shape
            pad_value = kernel.shape[0] - factor + (convW - 1)
            stride_value = [factor, factor]
            upfirdn_input = upfirdn2d_native(hidden_states, torch.tensor(kernel, device=hidden_states.device), pad=((pad_value + 1) // 2, pad_value // 2))
            output = F.conv2d(upfirdn_input, weight, stride=stride_value, padding=0)
        else:
            pad_value = kernel.shape[0] - factor
            output = upfirdn2d_native(hidden_states, torch.tensor(kernel, device=hidden_states.device), down=factor, pad=((pad_value + 1) // 2, pad_value // 2))
        return output

    def forward(self, hidden_states):
        if self.use_conv:
            downsample_input = self._downsample_2d(hidden_states, weight=self.Conv2d_0.weight, kernel=self.fir_kernel)
            hidden_states = downsample_input + self.Conv2d_0.bias.reshape(1, -1, 1, 1)
        else:
            hidden_states = self._downsample_2d(hidden_states, kernel=self.fir_kernel, factor=2)
        return hidden_states


class AttnSkipDownBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_pre_norm: 'bool'=True, attention_head_dim=1, output_scale_factor=np.sqrt(2.0), add_downsample=True):
        super().__init__()
        self.attentions = nn.ModuleList([])
        self.resnets = nn.ModuleList([])
        if attention_head_dim is None:
            logger.warn(f'It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}.')
            attention_head_dim = out_channels
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            self.resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=min(in_channels // 4, 32), groups_out=min(out_channels // 4, 32), dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
            self.attentions.append(Attention(out_channels, heads=out_channels // attention_head_dim, dim_head=attention_head_dim, rescale_output_factor=output_scale_factor, eps=resnet_eps, norm_num_groups=32, residual_connection=True, bias=True, upcast_softmax=True, _from_deprecated_attn_block=True))
        if add_downsample:
            self.resnet_down = ResnetBlock2D(in_channels=out_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=min(out_channels // 4, 32), dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm, use_in_shortcut=True, down=True, kernel='fir')
            self.downsamplers = nn.ModuleList([FirDownsample2D(out_channels, out_channels=out_channels)])
            self.skip_conv = nn.Conv2d(3, out_channels, kernel_size=(1, 1), stride=(1, 1))
        else:
            self.resnet_down = None
            self.downsamplers = None
            self.skip_conv = None

    def forward(self, hidden_states, temb=None, skip_sample=None):
        output_states = ()
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states, _ = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)
            output_states += hidden_states,
        if self.downsamplers is not None:
            hidden_states = self.resnet_down(hidden_states, temb)
            for downsampler in self.downsamplers:
                skip_sample = downsampler(skip_sample)
            hidden_states = self.skip_conv(skip_sample) + hidden_states
            output_states += hidden_states,
        return hidden_states, output_states, skip_sample


class DownEncoderBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, output_scale_factor=1.0, add_downsample=True, downsample_padding=1):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, temb_channels=None, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name='op')])
        else:
            self.downsamplers = None

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


class KAttentionBlock(nn.Module):
    """
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
    """

    def __init__(self, dim: 'int', num_attention_heads: 'int', attention_head_dim: 'int', dropout: 'float'=0.0, cross_attention_dim: 'Optional[int]'=None, attention_bias: 'bool'=False, upcast_attention: 'bool'=False, temb_channels: 'int'=768, add_self_attention: 'bool'=False, cross_attention_norm: 'Optional[str]'=None, group_size: 'int'=32):
        super().__init__()
        self.add_self_attention = add_self_attention
        if add_self_attention:
            self.norm1 = AdaGroupNorm(temb_channels, dim, max(1, dim // group_size))
            self.attn1 = Attention(query_dim=dim, heads=num_attention_heads, dim_head=attention_head_dim, dropout=dropout, bias=attention_bias, cross_attention_dim=None, cross_attention_norm=None)
        self.norm2 = AdaGroupNorm(temb_channels, dim, max(1, dim // group_size))
        self.attn2 = Attention(query_dim=dim, cross_attention_dim=cross_attention_dim, heads=num_attention_heads, dim_head=attention_head_dim, dropout=dropout, bias=attention_bias, upcast_attention=upcast_attention, cross_attention_norm=cross_attention_norm)

    def _to_3d(self, hidden_states, height, weight):
        return hidden_states.permute(0, 2, 3, 1).reshape(hidden_states.shape[0], height * weight, -1)

    def _to_4d(self, hidden_states, height, weight):
        return hidden_states.permute(0, 2, 1).reshape(hidden_states.shape[0], -1, height, weight)

    def forward(self, hidden_states: 'torch.FloatTensor', encoder_hidden_states: 'Optional[torch.FloatTensor]'=None, emb: 'Optional[torch.FloatTensor]'=None, attention_mask: 'Optional[torch.FloatTensor]'=None, cross_attention_kwargs: 'Optional[Dict[str, Any]]'=None, encoder_attention_mask: 'Optional[torch.FloatTensor]'=None):
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        if self.add_self_attention:
            norm_hidden_states = self.norm1(hidden_states, emb)
            height, weight = norm_hidden_states.shape[2:]
            norm_hidden_states = self._to_3d(norm_hidden_states, height, weight)
            attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None, attention_mask=attention_mask, **cross_attention_kwargs)
            attn_output = self._to_4d(attn_output, height, weight)
            hidden_states = attn_output + hidden_states
        norm_hidden_states = self.norm2(hidden_states, emb)
        height, weight = norm_hidden_states.shape[2:]
        norm_hidden_states = self._to_3d(norm_hidden_states, height, weight)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask if encoder_hidden_states is None else encoder_attention_mask, **cross_attention_kwargs)
        attn_output = self._to_4d(attn_output, height, weight)
        hidden_states = attn_output + hidden_states
        return hidden_states


class KDownsample2D(nn.Module):

    def __init__(self, pad_mode='reflect'):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor([[1 / 8, 3 / 8, 3 / 8, 1 / 8]])
        self.pad = kernel_1d.shape[1] // 2 - 1
        self.register_buffer('kernel', kernel_1d.T @ kernel_1d, persistent=False)

    def forward(self, inputs):
        inputs = F.pad(inputs, (self.pad,) * 4, self.pad_mode)
        weight = inputs.new_zeros([inputs.shape[1], inputs.shape[1], self.kernel.shape[0], self.kernel.shape[1]])
        indices = torch.arange(inputs.shape[1], device=inputs.device)
        kernel = self.kernel[None, :].expand(inputs.shape[1], -1, -1)
        weight[indices, indices] = kernel
        return F.conv2d(inputs, weight, stride=2)


class KCrossAttnDownBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', temb_channels: 'int', cross_attention_dim: 'int', dropout: 'float'=0.0, num_layers: 'int'=4, resnet_group_size: 'int'=32, add_downsample=True, attention_head_dim: 'int'=64, add_self_attention: 'bool'=False, resnet_eps: 'float'=1e-05, resnet_act_fn: 'str'='gelu'):
        super().__init__()
        resnets = []
        attentions = []
        self.has_cross_attention = True
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            groups = in_channels // resnet_group_size
            groups_out = out_channels // resnet_group_size
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, dropout=dropout, temb_channels=temb_channels, groups=groups, groups_out=groups_out, eps=resnet_eps, non_linearity=resnet_act_fn, time_embedding_norm='ada_group', conv_shortcut_bias=False))
            attentions.append(KAttentionBlock(out_channels, out_channels // attention_head_dim, attention_head_dim, cross_attention_dim=cross_attention_dim, temb_channels=temb_channels, attention_bias=True, add_self_attention=add_self_attention, cross_attention_norm='layer_norm', group_size=resnet_group_size))
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)
        if add_downsample:
            self.downsamplers = nn.ModuleList([KDownsample2D()])
        else:
            self.downsamplers = None
        self.gradient_checkpointing = False

    def forward(self, hidden_states: 'torch.FloatTensor', temb: 'Optional[torch.FloatTensor]'=None, encoder_hidden_states: 'Optional[torch.FloatTensor]'=None, attention_mask: 'Optional[torch.FloatTensor]'=None, cross_attention_kwargs: 'Optional[Dict[str, Any]]'=None, encoder_attention_mask: 'Optional[torch.FloatTensor]'=None):
        output_states = ()
        for resnet, attn in zip(self.resnets, self.attentions):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):

                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward
                ckpt_kwargs: 'Dict[str, Any]' = {'use_reentrant': False} if is_torch_version('>=', '1.11.0') else {}
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb, **ckpt_kwargs)
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(attn, return_dict=False), hidden_states, encoder_hidden_states, temb, attention_mask, cross_attention_kwargs, encoder_attention_mask, **ckpt_kwargs)
            else:
                hidden_states, _ = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, emb=temb, attention_mask=attention_mask, cross_attention_kwargs=cross_attention_kwargs, encoder_attention_mask=encoder_attention_mask)
            if self.downsamplers is None:
                output_states += None,
            else:
                output_states += hidden_states,
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states, output_states


class KDownBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=4, resnet_eps: 'float'=1e-05, resnet_act_fn: 'str'='gelu', resnet_group_size: 'int'=32, add_downsample=False):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            groups = in_channels // resnet_group_size
            groups_out = out_channels // resnet_group_size
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, dropout=dropout, temb_channels=temb_channels, groups=groups, groups_out=groups_out, eps=resnet_eps, non_linearity=resnet_act_fn, time_embedding_norm='ada_group', conv_shortcut_bias=False))
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList([KDownsample2D()])
        else:
            self.downsamplers = None
        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None):
        output_states = ()
        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                if is_torch_version('>=', '1.11.0'):
                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb, use_reentrant=False)
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states, _ = resnet(hidden_states, temb)
            output_states += hidden_states,
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states, output_states


class ResnetDownsampleBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, output_scale_factor=1.0, add_downsample=True, skip_time_act=False):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm, skip_time_act=skip_time_act))
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList([ResnetBlock2D(in_channels=out_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm, skip_time_act=skip_time_act, down=True)])
        else:
            self.downsamplers = None
        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None):
        output_states = ()
        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                if is_torch_version('>=', '1.11.0'):
                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb, use_reentrant=False)
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states, _ = resnet(hidden_states, temb)
            output_states = output_states + (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, temb)
            output_states = output_states + (hidden_states,)
        return hidden_states, output_states


class SimpleCrossAttnDownBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, attention_head_dim=1, cross_attention_dim=1280, output_scale_factor=1.0, add_downsample=True, skip_time_act=False, only_cross_attention=False, cross_attention_norm=None):
        super().__init__()
        self.has_cross_attention = True
        resnets = []
        attentions = []
        self.attention_head_dim = attention_head_dim
        self.num_heads = out_channels // self.attention_head_dim
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm, skip_time_act=skip_time_act))
            processor = AttnAddedKVProcessor2_0() if hasattr(F, 'scaled_dot_product_attention') else AttnAddedKVProcessor()
            attentions.append(Attention(query_dim=out_channels, cross_attention_dim=out_channels, heads=self.num_heads, dim_head=attention_head_dim, added_kv_proj_dim=cross_attention_dim, norm_num_groups=resnet_groups, bias=True, upcast_softmax=True, only_cross_attention=only_cross_attention, cross_attention_norm=cross_attention_norm, processor=processor))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList([ResnetBlock2D(in_channels=out_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm, skip_time_act=skip_time_act, down=True)])
        else:
            self.downsamplers = None
        self.gradient_checkpointing = False

    def forward(self, hidden_states: 'torch.FloatTensor', temb: 'Optional[torch.FloatTensor]'=None, encoder_hidden_states: 'Optional[torch.FloatTensor]'=None, attention_mask: 'Optional[torch.FloatTensor]'=None, cross_attention_kwargs: 'Optional[Dict[str, Any]]'=None, encoder_attention_mask: 'Optional[torch.FloatTensor]'=None):
        output_states = ()
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        if attention_mask is None:
            mask = None if encoder_hidden_states is None else encoder_attention_mask
        else:
            mask = attention_mask
        for resnet, attn in zip(self.resnets, self.attentions):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):

                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(attn, return_dict=False), hidden_states, encoder_hidden_states, mask, cross_attention_kwargs)[0]
            else:
                hidden_states, _ = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=mask, **cross_attention_kwargs)
            output_states = output_states + (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, temb)
            output_states = output_states + (hidden_states,)
        return hidden_states, output_states


class SkipDownBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_pre_norm: 'bool'=True, output_scale_factor=np.sqrt(2.0), add_downsample=True, downsample_padding=1):
        super().__init__()
        self.resnets = nn.ModuleList([])
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            self.resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=min(in_channels // 4, 32), groups_out=min(out_channels // 4, 32), dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
        if add_downsample:
            self.resnet_down = ResnetBlock2D(in_channels=out_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=min(out_channels // 4, 32), dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm, use_in_shortcut=True, down=True, kernel='fir')
            self.downsamplers = nn.ModuleList([FirDownsample2D(out_channels, out_channels=out_channels)])
            self.skip_conv = nn.Conv2d(3, out_channels, kernel_size=(1, 1), stride=(1, 1))
        else:
            self.resnet_down = None
            self.downsamplers = None
            self.skip_conv = None

    def forward(self, hidden_states, temb=None, skip_sample=None):
        output_states = ()
        for resnet in self.resnets:
            hidden_states, _ = resnet(hidden_states, temb)
            output_states += hidden_states,
        if self.downsamplers is not None:
            hidden_states = self.resnet_down(hidden_states, temb)
            for downsampler in self.downsamplers:
                skip_sample = downsampler(skip_sample)
            hidden_states = self.skip_conv(skip_sample) + hidden_states
            output_states += hidden_states,
        return hidden_states, output_states, skip_sample


def get_down_block(down_block_type, num_layers, in_channels, out_channels, temb_channels, add_downsample, resnet_eps, resnet_act_fn, transformer_layers_per_block=1, num_attention_heads=None, resnet_groups=None, cross_attention_dim=None, downsample_padding=None, dual_cross_attention=False, use_linear_projection=False, only_cross_attention=False, upcast_attention=False, resnet_time_scale_shift='default', resnet_skip_time_act=False, resnet_out_scale_factor=1.0, cross_attention_norm=None, attention_head_dim=None, downsample_type=None):
    if attention_head_dim is None:
        logger.warn(f'It is recommended to provide `attention_head_dim` when calling `get_down_block`. Defaulting `attention_head_dim` to {num_attention_heads}.')
        attention_head_dim = num_attention_heads
    down_block_type = down_block_type[7:] if down_block_type.startswith('UNetRes') else down_block_type
    if down_block_type == 'DownBlock2D':
        return DownBlock2D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, add_downsample=add_downsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups, downsample_padding=downsample_padding, resnet_time_scale_shift=resnet_time_scale_shift)
    elif down_block_type == 'ResnetDownsampleBlock2D':
        return ResnetDownsampleBlock2D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, add_downsample=add_downsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups, resnet_time_scale_shift=resnet_time_scale_shift, skip_time_act=resnet_skip_time_act, output_scale_factor=resnet_out_scale_factor)
    elif down_block_type == 'AttnDownBlock2D':
        if add_downsample is False:
            downsample_type = None
        else:
            downsample_type = downsample_type or 'conv'
        return AttnDownBlock2D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups, downsample_padding=downsample_padding, attention_head_dim=attention_head_dim, resnet_time_scale_shift=resnet_time_scale_shift, downsample_type=downsample_type)
    elif down_block_type == 'CrossAttnDownBlock2D':
        if cross_attention_dim is None:
            raise ValueError('cross_attention_dim must be specified for CrossAttnDownBlock2D')
        return CrossAttnDownBlock2D(num_layers=num_layers, transformer_layers_per_block=transformer_layers_per_block, in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, add_downsample=add_downsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups, downsample_padding=downsample_padding, cross_attention_dim=cross_attention_dim, num_attention_heads=num_attention_heads, dual_cross_attention=dual_cross_attention, use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention, upcast_attention=upcast_attention, resnet_time_scale_shift=resnet_time_scale_shift)
    elif down_block_type == 'SimpleCrossAttnDownBlock2D':
        if cross_attention_dim is None:
            raise ValueError('cross_attention_dim must be specified for SimpleCrossAttnDownBlock2D')
        return SimpleCrossAttnDownBlock2D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, add_downsample=add_downsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups, cross_attention_dim=cross_attention_dim, attention_head_dim=attention_head_dim, resnet_time_scale_shift=resnet_time_scale_shift, skip_time_act=resnet_skip_time_act, output_scale_factor=resnet_out_scale_factor, only_cross_attention=only_cross_attention, cross_attention_norm=cross_attention_norm)
    elif down_block_type == 'SkipDownBlock2D':
        return SkipDownBlock2D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, add_downsample=add_downsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, downsample_padding=downsample_padding, resnet_time_scale_shift=resnet_time_scale_shift)
    elif down_block_type == 'AttnSkipDownBlock2D':
        return AttnSkipDownBlock2D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, add_downsample=add_downsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, attention_head_dim=attention_head_dim, resnet_time_scale_shift=resnet_time_scale_shift)
    elif down_block_type == 'DownEncoderBlock2D':
        return DownEncoderBlock2D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, add_downsample=add_downsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups, downsample_padding=downsample_padding, resnet_time_scale_shift=resnet_time_scale_shift)
    elif down_block_type == 'AttnDownEncoderBlock2D':
        return AttnDownEncoderBlock2D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, add_downsample=add_downsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups, downsample_padding=downsample_padding, attention_head_dim=attention_head_dim, resnet_time_scale_shift=resnet_time_scale_shift)
    elif down_block_type == 'KDownBlock2D':
        return KDownBlock2D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, add_downsample=add_downsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn)
    elif down_block_type == 'KCrossAttnDownBlock2D':
        return KCrossAttnDownBlock2D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, add_downsample=add_downsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, cross_attention_dim=cross_attention_dim, attention_head_dim=attention_head_dim, add_self_attention=True if not add_downsample else False)
    raise ValueError(f'{down_block_type} does not exist.')


class FirUpsample2D(nn.Module):
    """A 2D FIR upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        fir_kernel (`tuple`, default `(1, 3, 3, 1)`):
            kernel for the FIR filter.
    """

    def __init__(self, channels=None, out_channels=None, use_conv=False, fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_channels = out_channels if out_channels else channels
        if use_conv:
            self.Conv2d_0 = nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.use_conv = use_conv
        self.fir_kernel = fir_kernel
        self.out_channels = out_channels

    def _upsample_2d(self, hidden_states, weight=None, kernel=None, factor=2, gain=1):
        """Fused `upsample_2d()` followed by `Conv2d()`.

        Padding is performed only once at the beginning, not between the operations. The fused op is considerably more
        efficient than performing the same calculation using standard TensorFlow ops. It supports gradients of
        arbitrary order.

        Args:
            hidden_states: Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
            weight: Weight tensor of the shape `[filterH, filterW, inChannels,
                outChannels]`. Grouped convolution can be performed by `inChannels = x.shape[0] // numGroups`.
            kernel: FIR filter of the shape `[firH, firW]` or `[firN]`
                (separable). The default is `[1] * factor`, which corresponds to nearest-neighbor upsampling.
            factor: Integer upsampling factor (default: 2).
            gain: Scaling factor for signal magnitude (default: 1.0).

        Returns:
            output: Tensor of the shape `[N, C, H * factor, W * factor]` or `[N, H * factor, W * factor, C]`, and same
            datatype as `hidden_states`.
        """
        assert isinstance(factor, int) and factor >= 1
        if kernel is None:
            kernel = [1] * factor
        kernel = torch.tensor(kernel, dtype=torch.float32)
        if kernel.ndim == 1:
            kernel = torch.outer(kernel, kernel)
        kernel /= torch.sum(kernel)
        kernel = kernel * (gain * factor ** 2)
        if self.use_conv:
            convH = weight.shape[2]
            convW = weight.shape[3]
            inC = weight.shape[1]
            pad_value = kernel.shape[0] - factor - (convW - 1)
            stride = factor, factor
            output_shape = (hidden_states.shape[2] - 1) * factor + convH, (hidden_states.shape[3] - 1) * factor + convW
            output_padding = output_shape[0] - (hidden_states.shape[2] - 1) * stride[0] - convH, output_shape[1] - (hidden_states.shape[3] - 1) * stride[1] - convW
            assert output_padding[0] >= 0 and output_padding[1] >= 0
            num_groups = hidden_states.shape[1] // inC
            weight = torch.reshape(weight, (num_groups, -1, inC, convH, convW))
            weight = torch.flip(weight, dims=[3, 4]).permute(0, 2, 1, 3, 4)
            weight = torch.reshape(weight, (num_groups * inC, -1, convH, convW))
            inverse_conv = F.conv_transpose2d(hidden_states, weight, stride=stride, output_padding=output_padding, padding=0)
            output = upfirdn2d_native(inverse_conv, torch.tensor(kernel, device=inverse_conv.device), pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2 + 1))
        else:
            pad_value = kernel.shape[0] - factor
            output = upfirdn2d_native(hidden_states, torch.tensor(kernel, device=hidden_states.device), up=factor, pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2))
        return output

    def forward(self, hidden_states):
        if self.use_conv:
            height = self._upsample_2d(hidden_states, self.Conv2d_0.weight, kernel=self.fir_kernel)
            height = height + self.Conv2d_0.bias.reshape(1, -1, 1, 1)
        else:
            height = self._upsample_2d(hidden_states, kernel=self.fir_kernel, factor=2)
        return height


class AttnSkipUpBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', prev_output_channel: 'int', out_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_pre_norm: 'bool'=True, attention_head_dim=1, output_scale_factor=np.sqrt(2.0), add_upsample=True):
        super().__init__()
        self.attentions = nn.ModuleList([])
        self.resnets = nn.ModuleList([])
        for i in range(num_layers):
            res_skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            self.resnets.append(ResnetBlock2D(in_channels=resnet_in_channels + res_skip_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=min(resnet_in_channels + res_skip_channels // 4, 32), groups_out=min(out_channels // 4, 32), dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
        if attention_head_dim is None:
            logger.warn(f'It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `out_channels`: {out_channels}.')
            attention_head_dim = out_channels
        self.attentions.append(Attention(out_channels, heads=out_channels // attention_head_dim, dim_head=attention_head_dim, rescale_output_factor=output_scale_factor, eps=resnet_eps, norm_num_groups=32, residual_connection=True, bias=True, upcast_softmax=True, _from_deprecated_attn_block=True))
        self.upsampler = FirUpsample2D(in_channels, out_channels=out_channels)
        if add_upsample:
            self.resnet_up = ResnetBlock2D(in_channels=out_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=min(out_channels // 4, 32), groups_out=min(out_channels // 4, 32), dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm, use_in_shortcut=True, up=True, kernel='fir')
            self.skip_conv = nn.Conv2d(out_channels, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.skip_norm = torch.nn.GroupNorm(num_groups=min(out_channels // 4, 32), num_channels=out_channels, eps=resnet_eps, affine=True)
            self.act = nn.SiLU()
        else:
            self.resnet_up = None
            self.skip_conv = None
            self.skip_norm = None
            self.act = None

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, skip_sample=None):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states, _ = resnet(hidden_states, temb)
        hidden_states = self.attentions[0](hidden_states)
        if skip_sample is not None:
            skip_sample = self.upsampler(skip_sample)
        else:
            skip_sample = 0
        if self.resnet_up is not None:
            skip_sample_states = self.skip_norm(hidden_states)
            skip_sample_states = self.act(skip_sample_states)
            skip_sample_states = self.skip_conv(skip_sample_states)
            skip_sample = skip_sample + skip_sample_states
            hidden_states = self.resnet_up(hidden_states, temb)
        return hidden_states, skip_sample


class AttnUpBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', prev_output_channel: 'int', out_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, attention_head_dim=1, output_scale_factor=1.0, upsample_type='conv'):
        super().__init__()
        resnets = []
        attentions = []
        self.upsample_type = upsample_type
        if attention_head_dim is None:
            logger.warn(f'It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}.')
            attention_head_dim = out_channels
        for i in range(num_layers):
            res_skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=resnet_in_channels + res_skip_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
            attentions.append(Attention(out_channels, heads=out_channels // attention_head_dim, dim_head=attention_head_dim, rescale_output_factor=output_scale_factor, eps=resnet_eps, norm_num_groups=resnet_groups, residual_connection=True, bias=True, upcast_softmax=True, _from_deprecated_attn_block=True))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if upsample_type == 'conv':
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        elif upsample_type == 'resnet':
            self.upsamplers = nn.ModuleList([ResnetBlock2D(in_channels=out_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm, up=True)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states, _ = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                if self.upsample_type == 'resnet':
                    hidden_states = upsampler(hidden_states, temb=temb)
                else:
                    hidden_states = upsampler(hidden_states)
        return hidden_states


class AttnUpDecoderBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, attention_head_dim=1, output_scale_factor=1.0, add_upsample=True, temb_channels=None):
        super().__init__()
        resnets = []
        attentions = []
        if attention_head_dim is None:
            logger.warn(f'It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `out_channels`: {out_channels}.')
            attention_head_dim = out_channels
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=input_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
            attentions.append(Attention(out_channels, heads=out_channels // attention_head_dim, dim_head=attention_head_dim, rescale_output_factor=output_scale_factor, eps=resnet_eps, norm_num_groups=resnet_groups if resnet_time_scale_shift != 'spatial' else None, spatial_norm_dim=temb_channels if resnet_time_scale_shift == 'spatial' else None, residual_connection=True, bias=True, upcast_softmax=True, _from_deprecated_attn_block=True))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states, temb=None):
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb=temb)
            hidden_states = attn(hidden_states, temb=temb)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


class KUpsample2D(nn.Module):

    def __init__(self, pad_mode='reflect'):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor([[1 / 8, 3 / 8, 3 / 8, 1 / 8]]) * 2
        self.pad = kernel_1d.shape[1] // 2 - 1
        self.register_buffer('kernel', kernel_1d.T @ kernel_1d, persistent=False)

    def forward(self, inputs):
        inputs = F.pad(inputs, ((self.pad + 1) // 2,) * 4, self.pad_mode)
        weight = inputs.new_zeros([inputs.shape[1], inputs.shape[1], self.kernel.shape[0], self.kernel.shape[1]])
        indices = torch.arange(inputs.shape[1], device=inputs.device)
        kernel = self.kernel[None, :].expand(inputs.shape[1], -1, -1)
        weight[indices, indices] = kernel
        return F.conv_transpose2d(inputs, weight, stride=2, padding=self.pad * 2 + 1)


class KCrossAttnUpBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=4, resnet_eps: 'float'=1e-05, resnet_act_fn: 'str'='gelu', resnet_group_size: 'int'=32, attention_head_dim=1, cross_attention_dim: 'int'=768, add_upsample: 'bool'=True, upcast_attention: 'bool'=False):
        super().__init__()
        resnets = []
        attentions = []
        is_first_block = in_channels == out_channels == temb_channels
        is_middle_block = in_channels != out_channels
        add_self_attention = True if is_first_block else False
        self.has_cross_attention = True
        self.attention_head_dim = attention_head_dim
        k_in_channels = out_channels if is_first_block else 2 * out_channels
        k_out_channels = in_channels
        num_layers = num_layers - 1
        for i in range(num_layers):
            in_channels = k_in_channels if i == 0 else out_channels
            groups = in_channels // resnet_group_size
            groups_out = out_channels // resnet_group_size
            if is_middle_block and i == num_layers - 1:
                conv_2d_out_channels = k_out_channels
            else:
                conv_2d_out_channels = None
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, conv_2d_out_channels=conv_2d_out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=groups, groups_out=groups_out, dropout=dropout, non_linearity=resnet_act_fn, time_embedding_norm='ada_group', conv_shortcut_bias=False))
            attentions.append(KAttentionBlock(k_out_channels if i == num_layers - 1 else out_channels, k_out_channels // attention_head_dim if i == num_layers - 1 else out_channels // attention_head_dim, attention_head_dim, cross_attention_dim=cross_attention_dim, temb_channels=temb_channels, attention_bias=True, add_self_attention=add_self_attention, cross_attention_norm='layer_norm', upcast_attention=upcast_attention))
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)
        if add_upsample:
            self.upsamplers = nn.ModuleList([KUpsample2D()])
        else:
            self.upsamplers = None
        self.gradient_checkpointing = False

    def forward(self, hidden_states: 'torch.FloatTensor', res_hidden_states_tuple: 'Tuple[torch.FloatTensor, ...]', temb: 'Optional[torch.FloatTensor]'=None, encoder_hidden_states: 'Optional[torch.FloatTensor]'=None, cross_attention_kwargs: 'Optional[Dict[str, Any]]'=None, upsample_size: 'Optional[int]'=None, attention_mask: 'Optional[torch.FloatTensor]'=None, encoder_attention_mask: 'Optional[torch.FloatTensor]'=None):
        res_hidden_states_tuple = res_hidden_states_tuple[-1]
        if res_hidden_states_tuple is not None:
            hidden_states = torch.cat([hidden_states, res_hidden_states_tuple], dim=1)
        for resnet, attn in zip(self.resnets, self.attentions):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):

                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward
                ckpt_kwargs: 'Dict[str, Any]' = {'use_reentrant': False} if is_torch_version('>=', '1.11.0') else {}
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb, **ckpt_kwargs)
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(attn, return_dict=False), hidden_states, encoder_hidden_states, temb, attention_mask, cross_attention_kwargs, encoder_attention_mask, **ckpt_kwargs)[0]
            else:
                hidden_states, _ = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, emb=temb, attention_mask=attention_mask, cross_attention_kwargs=cross_attention_kwargs, encoder_attention_mask=encoder_attention_mask)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


class KUpBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=5, resnet_eps: 'float'=1e-05, resnet_act_fn: 'str'='gelu', resnet_group_size: 'Optional[int]'=32, add_upsample=True):
        super().__init__()
        resnets = []
        k_in_channels = 2 * out_channels
        k_out_channels = in_channels
        num_layers = num_layers - 1
        for i in range(num_layers):
            in_channels = k_in_channels if i == 0 else out_channels
            groups = in_channels // resnet_group_size
            groups_out = out_channels // resnet_group_size
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=k_out_channels if i == num_layers - 1 else out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=groups, groups_out=groups_out, dropout=dropout, non_linearity=resnet_act_fn, time_embedding_norm='ada_group', conv_shortcut_bias=False))
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList([KUpsample2D()])
        else:
            self.upsamplers = None
        self.gradient_checkpointing = False

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
        res_hidden_states_tuple = res_hidden_states_tuple[-1]
        if res_hidden_states_tuple is not None:
            hidden_states = torch.cat([hidden_states, res_hidden_states_tuple], dim=1)
        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                if is_torch_version('>=', '1.11.0'):
                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb, use_reentrant=False)
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states, _ = resnet(hidden_states, temb)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


class ResnetUpsampleBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', prev_output_channel: 'int', out_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, output_scale_factor=1.0, add_upsample=True, skip_time_act=False):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            res_skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=resnet_in_channels + res_skip_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm, skip_time_act=skip_time_act))
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList([ResnetBlock2D(in_channels=out_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm, skip_time_act=skip_time_act, up=True)])
        else:
            self.upsamplers = None
        self.gradient_checkpointing = False

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                if is_torch_version('>=', '1.11.0'):
                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb, use_reentrant=False)
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states, _ = resnet(hidden_states, temb)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, temb)
        return hidden_states


class SimpleCrossAttnUpBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', prev_output_channel: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, attention_head_dim=1, cross_attention_dim=1280, output_scale_factor=1.0, add_upsample=True, skip_time_act=False, only_cross_attention=False, cross_attention_norm=None):
        super().__init__()
        resnets = []
        attentions = []
        self.has_cross_attention = True
        self.attention_head_dim = attention_head_dim
        self.num_heads = out_channels // self.attention_head_dim
        for i in range(num_layers):
            res_skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=resnet_in_channels + res_skip_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm, skip_time_act=skip_time_act))
            processor = AttnAddedKVProcessor2_0() if hasattr(F, 'scaled_dot_product_attention') else AttnAddedKVProcessor()
            attentions.append(Attention(query_dim=out_channels, cross_attention_dim=out_channels, heads=self.num_heads, dim_head=self.attention_head_dim, added_kv_proj_dim=cross_attention_dim, norm_num_groups=resnet_groups, bias=True, upcast_softmax=True, only_cross_attention=only_cross_attention, cross_attention_norm=cross_attention_norm, processor=processor))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList([ResnetBlock2D(in_channels=out_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm, skip_time_act=skip_time_act, up=True)])
        else:
            self.upsamplers = None
        self.gradient_checkpointing = False

    def forward(self, hidden_states: 'torch.FloatTensor', res_hidden_states_tuple: 'Tuple[torch.FloatTensor, ...]', temb: 'Optional[torch.FloatTensor]'=None, encoder_hidden_states: 'Optional[torch.FloatTensor]'=None, upsample_size: 'Optional[int]'=None, attention_mask: 'Optional[torch.FloatTensor]'=None, cross_attention_kwargs: 'Optional[Dict[str, Any]]'=None, encoder_attention_mask: 'Optional[torch.FloatTensor]'=None):
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        if attention_mask is None:
            mask = None if encoder_hidden_states is None else encoder_attention_mask
        else:
            mask = attention_mask
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):

                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(attn, return_dict=False), hidden_states, encoder_hidden_states, mask, cross_attention_kwargs)[0]
            else:
                hidden_states, _ = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=mask, **cross_attention_kwargs)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, temb)
        return hidden_states


class SkipUpBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', prev_output_channel: 'int', out_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_pre_norm: 'bool'=True, output_scale_factor=np.sqrt(2.0), add_upsample=True, upsample_padding=1):
        super().__init__()
        self.resnets = nn.ModuleList([])
        for i in range(num_layers):
            res_skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            self.resnets.append(ResnetBlock2D(in_channels=resnet_in_channels + res_skip_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=min((resnet_in_channels + res_skip_channels) // 4, 32), groups_out=min(out_channels // 4, 32), dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
        self.upsampler = FirUpsample2D(in_channels, out_channels=out_channels)
        if add_upsample:
            self.resnet_up = ResnetBlock2D(in_channels=out_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=min(out_channels // 4, 32), groups_out=min(out_channels // 4, 32), dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm, use_in_shortcut=True, up=True, kernel='fir')
            self.skip_conv = nn.Conv2d(out_channels, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.skip_norm = torch.nn.GroupNorm(num_groups=min(out_channels // 4, 32), num_channels=out_channels, eps=resnet_eps, affine=True)
            self.act = nn.SiLU()
        else:
            self.resnet_up = None
            self.skip_conv = None
            self.skip_norm = None
            self.act = None

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, skip_sample=None):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states, _ = resnet(hidden_states, temb)
        if skip_sample is not None:
            skip_sample = self.upsampler(skip_sample)
        else:
            skip_sample = 0
        if self.resnet_up is not None:
            skip_sample_states = self.skip_norm(hidden_states)
            skip_sample_states = self.act(skip_sample_states)
            skip_sample_states = self.skip_conv(skip_sample_states)
            skip_sample = skip_sample + skip_sample_states
            hidden_states = self.resnet_up(hidden_states, temb)
        return hidden_states, skip_sample


class UpDecoderBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, output_scale_factor=1.0, add_upsample=True, temb_channels=None):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=input_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states, temb=None):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


def get_up_block(up_block_type, num_layers, in_channels, out_channels, prev_output_channel, temb_channels, add_upsample, resnet_eps, resnet_act_fn, transformer_layers_per_block=1, num_attention_heads=None, resnet_groups=None, cross_attention_dim=None, dual_cross_attention=False, use_linear_projection=False, only_cross_attention=False, upcast_attention=False, resnet_time_scale_shift='default', resnet_skip_time_act=False, resnet_out_scale_factor=1.0, cross_attention_norm=None, attention_head_dim=None, upsample_type=None):
    if attention_head_dim is None:
        logger.warn(f'It is recommended to provide `attention_head_dim` when calling `get_up_block`. Defaulting `attention_head_dim` to {num_attention_heads}.')
        attention_head_dim = num_attention_heads
    up_block_type = up_block_type[7:] if up_block_type.startswith('UNetRes') else up_block_type
    if up_block_type == 'UpBlock2D':
        return UpBlock2D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, prev_output_channel=prev_output_channel, temb_channels=temb_channels, add_upsample=add_upsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups, resnet_time_scale_shift=resnet_time_scale_shift)
    elif up_block_type == 'ResnetUpsampleBlock2D':
        return ResnetUpsampleBlock2D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, prev_output_channel=prev_output_channel, temb_channels=temb_channels, add_upsample=add_upsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups, resnet_time_scale_shift=resnet_time_scale_shift, skip_time_act=resnet_skip_time_act, output_scale_factor=resnet_out_scale_factor)
    elif up_block_type == 'CrossAttnUpBlock2D':
        if cross_attention_dim is None:
            raise ValueError('cross_attention_dim must be specified for CrossAttnUpBlock2D')
        return CrossAttnUpBlock2D(num_layers=num_layers, transformer_layers_per_block=transformer_layers_per_block, in_channels=in_channels, out_channels=out_channels, prev_output_channel=prev_output_channel, temb_channels=temb_channels, add_upsample=add_upsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups, cross_attention_dim=cross_attention_dim, num_attention_heads=num_attention_heads, dual_cross_attention=dual_cross_attention, use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention, upcast_attention=upcast_attention, resnet_time_scale_shift=resnet_time_scale_shift)
    elif up_block_type == 'SimpleCrossAttnUpBlock2D':
        if cross_attention_dim is None:
            raise ValueError('cross_attention_dim must be specified for SimpleCrossAttnUpBlock2D')
        return SimpleCrossAttnUpBlock2D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, prev_output_channel=prev_output_channel, temb_channels=temb_channels, add_upsample=add_upsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups, cross_attention_dim=cross_attention_dim, attention_head_dim=attention_head_dim, resnet_time_scale_shift=resnet_time_scale_shift, skip_time_act=resnet_skip_time_act, output_scale_factor=resnet_out_scale_factor, only_cross_attention=only_cross_attention, cross_attention_norm=cross_attention_norm)
    elif up_block_type == 'AttnUpBlock2D':
        if add_upsample is False:
            upsample_type = None
        else:
            upsample_type = upsample_type or 'conv'
        return AttnUpBlock2D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, prev_output_channel=prev_output_channel, temb_channels=temb_channels, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups, attention_head_dim=attention_head_dim, resnet_time_scale_shift=resnet_time_scale_shift, upsample_type=upsample_type)
    elif up_block_type == 'SkipUpBlock2D':
        return SkipUpBlock2D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, prev_output_channel=prev_output_channel, temb_channels=temb_channels, add_upsample=add_upsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, resnet_time_scale_shift=resnet_time_scale_shift)
    elif up_block_type == 'AttnSkipUpBlock2D':
        return AttnSkipUpBlock2D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, prev_output_channel=prev_output_channel, temb_channels=temb_channels, add_upsample=add_upsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, attention_head_dim=attention_head_dim, resnet_time_scale_shift=resnet_time_scale_shift)
    elif up_block_type == 'UpDecoderBlock2D':
        return UpDecoderBlock2D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, add_upsample=add_upsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups, resnet_time_scale_shift=resnet_time_scale_shift, temb_channels=temb_channels)
    elif up_block_type == 'AttnUpDecoderBlock2D':
        return AttnUpDecoderBlock2D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, add_upsample=add_upsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups, attention_head_dim=attention_head_dim, resnet_time_scale_shift=resnet_time_scale_shift, temb_channels=temb_channels)
    elif up_block_type == 'KUpBlock2D':
        return KUpBlock2D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, add_upsample=add_upsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn)
    elif up_block_type == 'KCrossAttnUpBlock2D':
        return KCrossAttnUpBlock2D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, add_upsample=add_upsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, cross_attention_dim=cross_attention_dim, attention_head_dim=attention_head_dim)
    raise ValueError(f'{up_block_type} does not exist.')


class RegionDiffusion(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.num_train_timesteps = 1000
        self.clip_gradient = False
        None
        model_id = 'runwayml/stable-diffusion-v1-5'
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder='vae')
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder='tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder='text_encoder')
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder='unet')
        self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', num_train_timesteps=self.num_train_timesteps, skip_prk_steps=True, steps_offset=1)
        self.alphas_cumprod = self.scheduler.alphas_cumprod
        self.masks = []
        self.attention_maps = None
        self.selfattn_maps = None
        self.crossattn_maps = None
        self.color_loss = torch.nn.functional.mse_loss
        self.forward_hooks = []
        self.forward_replacement_hooks = []
        None

    def get_text_embeds(self, prompt, negative_prompt):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids)[0]
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def get_text_embeds_list(self, prompts):
        text_embeddings = []
        for prompt in prompts:
            text_input = self.tokenizer([prompt], padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
            with torch.no_grad():
                text_embeddings.append(self.text_encoder(text_input.input_ids)[0])
        return text_embeddings

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None, use_guidance=False, text_format_dict={}, inject_selfattn=0, inject_background=0):
        if latents is None:
            latents = torch.randn((1, self.unet.in_channels, height // 8, width // 8), device=self.device)
        if inject_selfattn > 0 or inject_background > 0:
            latents_reference = latents.clone().detach()
        self.scheduler.set_timesteps(num_inference_steps)
        n_styles = text_embeddings.shape[0] - 1
        assert n_styles == len(self.masks)
        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                with torch.no_grad():
                    feat_inject_step = t > (1 - inject_selfattn) * 1000
                    background_inject_step = i == int(inject_background * len(self.scheduler.timesteps)) and inject_background > 0
                    noise_pred_uncond_cur = self.unet(latents, t, encoder_hidden_states=text_embeddings[:1])['sample']
                    self.register_fontsize_hooks(text_format_dict)
                    noise_pred_text_cur = self.unet(latents, t, encoder_hidden_states=text_embeddings[-1:])['sample']
                    self.remove_fontsize_hooks()
                    if inject_selfattn > 0 or inject_background > 0:
                        noise_pred_uncond_refer = self.unet(latents_reference, t, encoder_hidden_states=text_embeddings[:1])['sample']
                        self.register_selfattn_hooks(feat_inject_step)
                        noise_pred_text_refer = self.unet(latents_reference, t, encoder_hidden_states=text_embeddings[-1:])['sample']
                        self.remove_selfattn_hooks()
                    noise_pred_uncond = noise_pred_uncond_cur * self.masks[-1]
                    noise_pred_text = noise_pred_text_cur * self.masks[-1]
                    for style_i, mask in enumerate(self.masks[:-1]):
                        self.register_replacement_hooks(feat_inject_step)
                        noise_pred_text_cur = self.unet(latents, t, encoder_hidden_states=text_embeddings[style_i + 1:style_i + 2])['sample']
                        self.remove_replacement_hooks()
                        noise_pred_uncond = noise_pred_uncond + noise_pred_uncond_cur * mask
                        noise_pred_text = noise_pred_text + noise_pred_text_cur * mask
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                if inject_selfattn > 0 or inject_background > 0:
                    noise_pred_refer = noise_pred_uncond_refer + guidance_scale * (noise_pred_text_refer - noise_pred_uncond_refer)
                    latents_reference = self.scheduler.step(torch.cat([noise_pred, noise_pred_refer]), t, torch.cat([latents, latents_reference]))['prev_sample']
                    latents, latents_reference = torch.chunk(latents_reference, 2, dim=0)
                else:
                    latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
                if use_guidance and t < text_format_dict['guidance_start_step']:
                    with torch.enable_grad():
                        if not latents.requires_grad:
                            latents.requires_grad = True
                        latents_0 = self.predict_x0(latents, noise_pred, t)
                        latents_inp = 1 / 0.18215 * latents_0
                        imgs = self.vae.decode(latents_inp).sample
                        imgs = (imgs / 2 + 0.5).clamp(0, 1)
                        loss_total = 0.0
                        for attn_map, rgb_val in zip(text_format_dict['color_obj_atten'], text_format_dict['target_RGB']):
                            avg_rgb = (imgs * attn_map[:, 0]).sum(2).sum(2) / attn_map[:, 0].sum()
                            loss = self.color_loss(avg_rgb, rgb_val[:, :, 0, 0]) * 100
                            loss_total += loss
                        loss_total.backward()
                    latents = (latents - latents.grad * text_format_dict['color_guidance_weight'] * text_format_dict['color_obj_atten_all']).detach().clone()
                if background_inject_step:
                    latents = latents_reference * self.masks[-1] + latents * (1 - self.masks[-1])
        return latents

    def predict_x0(self, x_t, eps_t, t):
        alpha_t = self.scheduler.alphas_cumprod[t]
        return (x_t - eps_t * torch.sqrt(1 - alpha_t)) / torch.sqrt(alpha_t)

    def produce_attn_maps(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]
        text_embeddings = self.get_text_embeds(prompts, negative_prompts)
        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)
        self.scheduler.set_timesteps(num_inference_steps)
        self.remove_replacement_hooks()
        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                latent_model_input = torch.cat([latents] * 2)
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        imgs = self.decode_latents(latents)
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')
        return imgs

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def encode_imgs(self, imgs):
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None, text_format_dict={}, use_guidance=False, inject_selfattn=0, inject_background=0):
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]
        text_embeds = self.get_text_embeds(prompts, negative_prompts)
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, use_guidance=use_guidance, text_format_dict=text_format_dict, inject_selfattn=inject_selfattn, inject_background=inject_background)
        imgs = self.decode_latents(latents)
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')
        return imgs

    def reset_attention_maps(self):
        """Function to reset attention maps.
        We reset attention maps because we append them while getting hooks
        to visualize attention maps for every step.
        """
        for key in self.selfattn_maps:
            self.selfattn_maps[key] = []
        for key in self.crossattn_maps:
            self.crossattn_maps[key] = []

    def register_evaluation_hooks(self):
        """Function for registering hooks during evaluation.
        We mainly store activation maps averaged over queries.
        """
        self.forward_hooks = []

        def save_activations(activations, name, module, inp, out):
            """
            PyTorch Forward hook to save outputs at each forward pass.
            """
            if 'attn2' in name:
                assert out[1].shape[-1] == 77
                activations[name].append(out[1].detach().cpu())
            else:
                assert out[1].shape[-1] != 77
        attention_dict = collections.defaultdict(list)
        for name, module in self.unet.named_modules():
            leaf_name = name.split('.')[-1]
            if 'attn' in leaf_name:
                self.forward_hooks.append(module.register_forward_hook(partial(save_activations, attention_dict, name)))
        self.attention_maps = attention_dict

    def register_selfattn_hooks(self, feat_inject_step=False):
        """Function for registering hooks during evaluation.
        We mainly store activation maps averaged over queries.
        """
        self.selfattn_forward_hooks = []

        def save_activations(activations, name, module, inp, out):
            """
            PyTorch Forward hook to save outputs at each forward pass.
            """
            if 'attn2' in name:
                assert out[1][1].shape[-1] == 77
            else:
                assert out[1][1].shape[-1] != 77
                activations[name] = out[1][1].detach()

        def save_resnet_activations(activations, name, module, inp, out):
            """
            PyTorch Forward hook to save outputs at each forward pass.
            """
            assert out[1].shape[-1] == 16
            activations[name] = out[1].detach()
        attention_dict = collections.defaultdict(list)
        for name, module in self.unet.named_modules():
            leaf_name = name.split('.')[-1]
            if 'attn' in leaf_name and feat_inject_step:
                self.selfattn_forward_hooks.append(module.register_forward_hook(partial(save_activations, attention_dict, name)))
            if name == 'up_blocks.1.resnets.1' and feat_inject_step:
                self.selfattn_forward_hooks.append(module.register_forward_hook(partial(save_resnet_activations, attention_dict, name)))
        self.self_attention_maps_cur = attention_dict

    def register_replacement_hooks(self, feat_inject_step=False):
        """Function for registering hooks to replace self attention.
        """
        self.forward_replacement_hooks = []

        def replace_activations(name, module, args):
            """
            PyTorch Forward hook to save outputs at each forward pass.
            """
            if 'attn1' in name:
                modified_args = args[0], self.self_attention_maps_cur[name]
                return modified_args

        def replace_resnet_activations(name, module, args):
            """
            PyTorch Forward hook to save outputs at each forward pass.
            """
            modified_args = args[0], args[1], self.self_attention_maps_cur[name]
            return modified_args
        for name, module in self.unet.named_modules():
            leaf_name = name.split('.')[-1]
            if 'attn' in leaf_name and feat_inject_step:
                self.forward_replacement_hooks.append(module.register_forward_pre_hook(partial(replace_activations, name)))
            if name == 'up_blocks.1.resnets.1' and feat_inject_step:
                self.forward_replacement_hooks.append(module.register_forward_pre_hook(partial(replace_resnet_activations, name)))

    def register_tokenmap_hooks(self):
        """Function for registering hooks during evaluation.
        We mainly store activation maps averaged over queries.
        """
        self.forward_hooks = []

        def save_activations(selfattn_maps, crossattn_maps, n_maps, name, module, inp, out):
            """
            PyTorch Forward hook to save outputs at each forward pass.
            """
            if name in n_maps:
                n_maps[name] += 1
            else:
                n_maps[name] = 1
            if 'attn2' in name:
                assert out[1][0].shape[-1] == 77
                if name in CrossAttentionLayers and n_maps[name] > 10:
                    if name in crossattn_maps:
                        crossattn_maps[name] += out[1][0].detach().cpu()[1:2]
                    else:
                        crossattn_maps[name] = out[1][0].detach().cpu()[1:2]
            else:
                assert out[1][0].shape[-1] != 77
                if name in SelfAttentionLayers and n_maps[name] > 10:
                    if name in crossattn_maps:
                        selfattn_maps[name] += out[1][0].detach().cpu()[1:2]
                    else:
                        selfattn_maps[name] = out[1][0].detach().cpu()[1:2]
        selfattn_maps = collections.defaultdict(list)
        crossattn_maps = collections.defaultdict(list)
        n_maps = collections.defaultdict(list)
        for name, module in self.unet.named_modules():
            leaf_name = name.split('.')[-1]
            if 'attn' in leaf_name:
                self.forward_hooks.append(module.register_forward_hook(partial(save_activations, selfattn_maps, crossattn_maps, n_maps, name)))
        self.selfattn_maps = selfattn_maps
        self.crossattn_maps = crossattn_maps
        self.n_maps = n_maps

    def remove_tokenmap_hooks(self):
        for hook in self.forward_hooks:
            hook.remove()
        self.selfattn_maps = None
        self.crossattn_maps = None
        self.n_maps = None

    def remove_evaluation_hooks(self):
        for hook in self.forward_hooks:
            hook.remove()
        self.attention_maps = None

    def remove_replacement_hooks(self):
        for hook in self.forward_replacement_hooks:
            hook.remove()

    def remove_selfattn_hooks(self):
        for hook in self.selfattn_forward_hooks:
            hook.remove()

    def register_fontsize_hooks(self, text_format_dict={}):
        """Function for registering hooks to replace self attention.
        """
        self.forward_fontsize_hooks = []

        def adjust_attn_weights(name, module, args):
            """
            PyTorch Forward hook to save outputs at each forward pass.
            """
            if 'attn2' in name:
                modified_args = args[0], None, attn_weights
                return modified_args
        if 'word_pos' in text_format_dict and text_format_dict['word_pos'] is not None and 'font_size' in text_format_dict and text_format_dict['font_size'] is not None:
            attn_weights = {'word_pos': text_format_dict['word_pos'], 'font_size': text_format_dict['font_size']}
        else:
            attn_weights = None
        for name, module in self.unet.named_modules():
            leaf_name = name.split('.')[-1]
            if 'attn' in leaf_name and attn_weights is not None:
                self.forward_fontsize_hooks.append(module.register_forward_pre_hook(partial(adjust_attn_weights, name)))

    def remove_fontsize_hooks(self):
        for hook in self.forward_fontsize_hooks:
            hook.remove()


class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
    """

    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name='conv'):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.conv = None
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, inputs):
        assert inputs.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(inputs)
        outputs = F.interpolate(inputs, scale_factor=2.0, mode='nearest')
        if self.use_conv:
            outputs = self.conv(outputs)
        return outputs


class Downsample1D(nn.Module):
    """A 1D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
    """

    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, name='conv'):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name
        if use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.conv = nn.AvgPool1d(kernel_size=stride, stride=stride)

    def forward(self, inputs):
        assert inputs.shape[1] == self.channels
        return self.conv(inputs)


def rearrange_dims(tensor):
    if len(tensor.shape) == 2:
        return tensor[:, :, None]
    if len(tensor.shape) == 3:
        return tensor[:, :, None, :]
    elif len(tensor.shape) == 4:
        return tensor[:, :, 0, :]
    else:
        raise ValueError(f'`len(tensor)`: {len(tensor)} has to be 2, 3 or 4.')


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.conv1d = nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.group_norm = nn.GroupNorm(n_groups, out_channels)
        self.mish = nn.Mish()

    def forward(self, inputs):
        intermediate_repr = self.conv1d(inputs)
        intermediate_repr = rearrange_dims(intermediate_repr)
        intermediate_repr = self.group_norm(intermediate_repr)
        intermediate_repr = rearrange_dims(intermediate_repr)
        output = self.mish(intermediate_repr)
        return output


class ResidualTemporalBlock1D(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, kernel_size=5):
        super().__init__()
        self.conv_in = Conv1dBlock(inp_channels, out_channels, kernel_size)
        self.conv_out = Conv1dBlock(out_channels, out_channels, kernel_size)
        self.time_emb_act = nn.Mish()
        self.time_emb = nn.Linear(embed_dim, out_channels)
        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) if inp_channels != out_channels else nn.Identity()

    def forward(self, inputs, t):
        """
        Args:
            inputs : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]

        returns:
            out : [ batch_size x out_channels x horizon ]
        """
        t = self.time_emb_act(t)
        t = self.time_emb(t)
        out = self.conv_in(inputs) + rearrange_dims(t)
        out = self.conv_out(out)
        return out + self.residual_conv(inputs)


class TemporalConvLayer(nn.Module):
    """
    Temporal convolutional layer that can be used for video (sequence of images) input Code mostly copied from:
    https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/models/multi_modal/video_synthesis/unet_sd.py#L1016
    """

    def __init__(self, in_dim, out_dim=None, dropout=0.0):
        super().__init__()
        out_dim = out_dim or in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv1 = nn.Sequential(nn.GroupNorm(32, in_dim), nn.SiLU(), nn.Conv3d(in_dim, out_dim, (3, 1, 1), padding=(1, 0, 0)))
        self.conv2 = nn.Sequential(nn.GroupNorm(32, out_dim), nn.SiLU(), nn.Dropout(dropout), nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)))
        self.conv3 = nn.Sequential(nn.GroupNorm(32, out_dim), nn.SiLU(), nn.Dropout(dropout), nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)))
        self.conv4 = nn.Sequential(nn.GroupNorm(32, out_dim), nn.SiLU(), nn.Dropout(dropout), nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)))
        nn.init.zeros_(self.conv4[-1].weight)
        nn.init.zeros_(self.conv4[-1].bias)

    def forward(self, hidden_states, num_frames=1):
        hidden_states = hidden_states[None, :].reshape((-1, num_frames) + hidden_states.shape[1:]).permute(0, 2, 1, 3, 4)
        identity = hidden_states
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.conv3(hidden_states)
        hidden_states = self.conv4(hidden_states)
        hidden_states = identity + hidden_states
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape((hidden_states.shape[0] * hidden_states.shape[2], -1) + hidden_states.shape[3:])
        return hidden_states


class UNetMidBlock2D(nn.Module):

    def __init__(self, in_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, add_attention: 'bool'=True, attention_head_dim=1, output_scale_factor=1.0):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention
        resnets = [ResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm)]
        attentions = []
        if attention_head_dim is None:
            logger.warn(f'It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {in_channels}.')
            attention_head_dim = in_channels
        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(Attention(in_channels, heads=in_channels // attention_head_dim, dim_head=attention_head_dim, rescale_output_factor=output_scale_factor, eps=resnet_eps, norm_num_groups=resnet_groups if resnet_time_scale_shift == 'default' else None, spatial_norm_dim=temb_channels if resnet_time_scale_shift == 'spatial' else None, residual_connection=True, bias=True, upcast_softmax=True, _from_deprecated_attn_block=True))
            else:
                attentions.append(None)
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states, temb=temb)
            hidden_states, _ = resnet(hidden_states, temb)
        return hidden_states


class CLIP_fx:

    def __init__(self, name='ViT-B/32', device='cuda'):
        self.model, _ = clip.load(name, device=device)
        self.model.eval()
        self.name = 'clip_' + name.lower().replace('-', '_').replace('/', '_')

    def __call__(self, img_t):
        img_x = img_t / 255.0
        T_norm = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        img_x = T_norm(img_x)
        assert torch.is_tensor(img_x)
        if len(img_x.shape) == 3:
            img_x = img_x.unsqueeze(0)
        B, C, H, W = img_x.shape
        with torch.no_grad():
            z = self.model.encode_image(img_x)
        return z

    def encode_text(self, text_t):
        text = clip.tokenize(text_t)
        z_text = self.model.encode_text(text).float()
        return z_text


def img_preprocess_clip(img_np):
    x = Image.fromarray(img_np.astype(np.uint8)).convert('RGB')
    T = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224)])
    return np.asarray(T(x)).clip(0, 255).astype(np.uint8)


class CLIPEncoder(nn.Module):

    def __init__(self, clip_version='ViT-B/32', pretrained='', device='cuda'):
        super().__init__()
        self.clip_version = clip_version
        if not pretrained:
            if self.clip_version == 'ViT-H-14':
                self.pretrained = 'laion2b_s32b_b79k'
            elif self.clip_version == 'ViT-g-14':
                self.pretrained = 'laion2b_s12b_b42k'
            else:
                self.pretrained = 'openai'
        self.model = CLIP_fx(name=clip_version)
        self.device = device

    @torch.no_grad()
    def get_clip_score(self, text, image):
        if not isinstance(text, (list, tuple)):
            text = [text]
        text_features = self.model.encode_text(text).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        if isinstance(image, str):
            image = torchvision.io.read_image(image)
        image = img_preprocess_clip(image)
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1).type(torch.float32)
        image_features = self.model(image).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.T
        return similarity


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AdaGroupNorm,
     lambda: ([], {'embedding_dim': 4, 'out_dim': 4, 'num_groups': 1}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (ApproximateGELU,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Downsample1D,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (Downsample2D,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FeedForward,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FirDownsample2D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FirUpsample2D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GEGLU,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GELU,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (KDownsample2D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (KUpsample2D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LoRALinearLayer,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Upsample1D,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Upsample2D,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

