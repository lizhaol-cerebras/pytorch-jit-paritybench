
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


import math


from typing import Any


from typing import Optional


from typing import Tuple


from typing import Union


import torch


import torch.utils.checkpoint


from torch import nn


import re


import logging


import numpy as np


from torchvision import transforms


import random


import time


import warnings


from torch.utils.data import Dataset


from torch.utils.data import Subset


from functools import partial


import torch.distributed as dist


from torch.utils.data import DataLoader


from torch.utils.data.distributed import DistributedSampler


from torch import distributed as dist


from queue import Queue


import uuid


import itertools


import pandas as pd


from sklearn.metrics import accuracy_score


from sklearn.metrics import precision_score


from sklearn.metrics import recall_score


from sklearn.metrics import confusion_matrix


from typing import List


import inspect


import torch.nn.functional as F


from abc import ABC


from abc import abstractmethod


import torch.nn as nn


from torch.nn import CrossEntropyLoss


import copy


from typing import TYPE_CHECKING


from typing import Callable


from typing import Generator


from torch.utils.data import Sampler


from typing import Dict


from typing import Sequence


import logging.handlers


from collections import defaultdict


import torchvision.transforms as transforms


from torchvision.transforms import InterpolationMode


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.data.sampler import Sampler


def get_abs_pos(abs_pos, tgt_size):
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype
    if src_size != tgt_size:
        return F.interpolate(abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2), size=(tgt_size, tgt_size), mode='bicubic', align_corners=False).permute(0, 2, 3, 1).flatten(0, 2)
    else:
        return abs_pos


class MplugOwlVisionEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        else:
            self.cls_token = None
        self.patch_embed = nn.Conv2d(in_channels=3, out_channels=self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size, bias=False)
        if self.cls_token is not None:
            self.num_patches = (self.image_size // self.patch_size) ** 2
            self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.hidden_size))
        else:
            self.num_patches = 256
            self.position_embedding = nn.Parameter(torch.randn(256, self.hidden_size))
        self.pre_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: 'torch.FloatTensor') ->torch.Tensor:
        batch_size = pixel_values.size(0)
        image_embeds = self.patch_embed(pixel_values)
        image_embeds = image_embeds.flatten(2).transpose(1, 2)
        if self.cls_token is not None:
            class_embeds = self.cls_token.expand(batch_size, 1, -1)
            embeddings = torch.cat([class_embeds, image_embeds], dim=1)
            embeddings = embeddings + self.position_embedding[:, :embeddings.size(1)]
        else:
            embeddings = image_embeds
            embeddings = embeddings + get_abs_pos(self.position_embedding, embeddings.size(1))
        embeddings = self.pre_layernorm(embeddings)
        return embeddings


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: 'torch.Tensor'):
        output = torch.nn.functional.layer_norm(x.float(), self.normalized_shape, self.weight.float() if self.weight is not None else None, self.bias.float() if self.bias is not None else None, self.eps)
        return output.type_as(x)


flash_attn_func = None


class MplugOwlVisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(f'hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads}).')
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(config.attention_dropout)
        self.query_key_value = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)

    def _shape(self, tensor: 'torch.Tensor', seq_len: 'int', bsz: 'int'):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: 'torch.Tensor', head_mask: 'Optional[torch.Tensor]'=None, output_attentions: 'Optional[bool]'=False) ->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, seq_len, embed_dim = hidden_states.size()
        mixed_qkv = self.query_key_value(hidden_states)
        mixed_qkv = mixed_qkv.reshape(bsz, seq_len, self.num_heads, 3, embed_dim // self.num_heads).permute(3, 0, 2, 1, 4)
        query_states, key_states, value_states = mixed_qkv[0], mixed_qkv[1], mixed_qkv[2]
        if False:
            query_states = query_states.permute(0, 2, 1, 3).contiguous()
            query_states = query_states.view(query_states.size(0) * query_states.size(1), query_states.size(2), -1)
            key_states = key_states.permute(0, 2, 1, 3).contiguous()
            key_states = key_states.view(key_states.size(0) * key_states.size(1), key_states.size(2), -1)
            value_states = value_states.permute(0, 2, 1, 3).contiguous()
            value_states = value_states.view(value_states.size(0) * value_states.size(1), value_states.size(2), -1)
            cu_seqlens = torch.arange(0, (bsz + 1) * seq_len, step=seq_len, dtype=torch.int32, device=query_states.device)
            context_layer = flash_attn_func(query_states, key_states, value_states, cu_seqlens, cu_seqlens, seq_len, seq_len, self.dropout if self.training else 0.0, softmax_scale=self.scale, causal=False, return_attn_probs=False)
            context_layer = context_layer.view(bsz, seq_len, context_layer.size(1), context_layer.size(2))
        else:
            attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
            attention_scores = attention_scores * self.scale
            attention_probs = torch.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)
            if head_mask is not None:
                attention_probs = attention_probs * head_mask
            context_layer = torch.matmul(attention_probs, value_states).permute(0, 2, 1, 3)
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)
        output = self.dense(context_layer)
        outputs = (output, attention_probs) if output_attentions else (output, None)
        return outputs


class QuickGELU(nn.Module):

    def forward(self, x: 'torch.Tensor'):
        return x * torch.sigmoid(1.702 * x)


class MplugOwlMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: 'torch.Tensor') ->torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class MplugOwlVisionEncoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MplugOwlVisionAttention(config)
        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MplugOwlMLP(config)
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'torch.Tensor', output_attentions: 'Optional[bool]'=False) ->Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights = self.self_attn(hidden_states=hidden_states, head_mask=attention_mask, output_attentions=output_attentions)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        outputs = hidden_states,
        if output_attentions:
            outputs += attn_weights,
        return outputs


class MplugOwlVisualAbstractorMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        in_features = config.hidden_size
        self.act = nn.SiLU()
        self.w1 = nn.Linear(in_features, config.intermediate_size)
        self.w2 = nn.Linear(config.intermediate_size, in_features)
        self.w3 = nn.Linear(in_features, config.intermediate_size)
        self.ffn_ln = nn.LayerNorm(config.intermediate_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: 'torch.Tensor') ->torch.Tensor:
        hidden_states = self.act(self.w1(hidden_states)) * self.w3(hidden_states)
        hidden_states = self.ffn_ln(hidden_states)
        hidden_states = self.w2(hidden_states)
        return hidden_states


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class MplugOwlVisualAbstractorMultiHeadAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.encoder_hidden_size, self.all_head_size)
        self.value = nn.Linear(config.encoder_hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.save_attention = False
        grids = config.grid_size
        self.register_buffer('q_pos_embed', torch.from_numpy(get_1d_sincos_pos_embed_from_grid(config.hidden_size, np.arange(config.num_learnable_queries, dtype=np.float32))).float())
        self.register_buffer('k_pos_embed', torch.from_numpy(get_2d_sincos_pos_embed(config.hidden_size, grids, cls_token=config.use_cls_token)).float())

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        qk_pos_embed = torch.cat([self.q_pos_embed, self.k_pos_embed], dim=0).unsqueeze(0)
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states + qk_pos_embed))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        attention_mask = encoder_attention_mask
        mixed_query_layer = self.query(hidden_states + self.q_pos_embed.unsqueeze(0))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        past_key_value = key_layer, value_layer
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        if self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)
        attention_probs_dropped = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask
        context_layer = torch.matmul(attention_probs_dropped, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        outputs = outputs + (past_key_value,)
        return outputs


class MplugOwlVisualAbstractorCrossOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        dim = config.encoder_hidden_size
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MplugOwlVisualAbstractorMLP(config)

    def forward(self, hidden_states: 'torch.Tensor', input_tensor: 'torch.Tensor') ->torch.Tensor:
        input_tensor = input_tensor + self.out_proj(hidden_states)
        input_tensor = input_tensor + self.mlp(self.norm2(input_tensor))
        return input_tensor


class MplugOwlVisualAbstractorAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = MplugOwlVisualAbstractorMultiHeadAttention(config)
        self.output = MplugOwlVisualAbstractorCrossOutput(config)
        self.pruned_heads = set()
        self.norm1 = nn.LayerNorm(config.encoder_hidden_size)
        self.normk = nn.LayerNorm(config.encoder_hidden_size)
        self.add_pos_embed = config.add_v2t_pos_emb
        if self.add_pos_embed:
            self.q_pos_embed = nn.Parameter(torch.from_numpy(get_1d_sincos_pos_embed_from_grid(config.encoder_hidden_size, np.arange(config.num_learnable_queries, dtype=np.float32))).float()).requires_grad_(False)
            self.k_pos_embed = nn.Parameter(torch.from_numpy(get_2d_sincos_pos_embed(config.encoder_hidden_size, config.grid_size, cls_token=config.cls_token)).float()).requires_grad_(False)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads)
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.out_proj, index, dim=1)
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'Optional[torch.FloatTensor]'=None, head_mask: 'Optional[torch.FloatTensor]'=None, encoder_hidden_states: 'Optional[torch.FloatTensor]'=None, encoder_attention_mask: 'Optional[torch.FloatTensor]'=None, past_key_value: 'Optional[Tuple[Tuple[torch.FloatTensor]]]'=None, output_attentions: 'Optional[bool]'=False) ->Tuple[torch.Tensor]:
        hidden_states = self.norm1(hidden_states)
        encoder_hidden_states = self.normk(encoder_hidden_states)
        encoder_hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        encoder_attention_mask = torch.cat([attention_mask, encoder_attention_mask], dim=-1)
        self_outputs = self.attention(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class MplugOwlVisualAbstractorLayer(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.layer_idx = layer_idx
        self.crossattention = MplugOwlVisualAbstractorAttention(config)
        self.has_cross_attention = True

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, output_attentions=False):
        if encoder_hidden_states is None:
            raise ValueError('encoder_hidden_states must be given for cross-attention layers')
        cross_attention_outputs = self.crossattention(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions=output_attentions)
        query_attention_output = cross_attention_outputs[0]
        outputs = query_attention_output,
        return outputs


class MplugOwlVisualAbstractorEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([MplugOwlVisualAbstractorLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = True

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, output_attentions=False, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        for i in range(self.config.num_hidden_layers):
            layer_module = self.layers[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if getattr(self.config, 'gradient_checkpointing', False) and self.training:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)
                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(layer_module), hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions)
            hidden_states = layer_outputs[0]
        return BaseModelOutput(last_hidden_state=hidden_states)


class MplugOwlVisionLocalTemporal(nn.Module):

    def __init__(self, config):
        super(MplugOwlVisionLocalTemporal, self).__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = 1 + (self.image_size // self.patch_size) ** 2
        self.hidden_size = config.hidden_size
        d_bottleneck = self.hidden_size // 2
        self.ln = LayerNormFp32(self.hidden_size)
        self.down_proj = nn.Conv3d(self.hidden_size, d_bottleneck, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv3d(d_bottleneck, d_bottleneck, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), groups=d_bottleneck)
        self.up_proj = nn.Conv3d(d_bottleneck, self.hidden_size, kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.up_proj.weight, 0)
        nn.init.constant_(self.up_proj.bias, 0)
        self.activation_func = QuickGELU()

    def forward(self, x):
        T = x.size(1)
        H = int((self.num_patches - 1) ** 0.5)
        cls_token, x = x[:, :, 0:1], x[:, :, 1:]
        x = self.ln(x)
        x = einops.rearrange(x, 'b t (h w) c -> b c t h w', h=H)
        x = self.down_proj(x)
        if self.conv.weight.dtype == torch.bfloat16:
            x = torch.nn.functional.conv3d(x.half(), self.conv.weight.half(), bias=self.conv.bias.half(), stride=1, padding=(1, 0, 0), groups=self.conv.weight.shape[0])
        else:
            x = self.conv(x)
        x = self.activation_func(x)
        x = self.up_proj(x)
        x = einops.rearrange(x, 'b c t h w -> b t (h w) c')
        x = torch.cat([cls_token, x], dim=2)
        return x


class MultiwayNetwork(nn.Module):

    def __init__(self, module_provider, num_multiway=2, out_features=None):
        super(MultiwayNetwork, self).__init__()
        self.multiway = torch.nn.ModuleList([module_provider() for _ in range(num_multiway)])
        self.out_features = out_features

    def forward(self, hidden_states, multiway_indices):
        if len(self.multiway) == 1:
            return self.multiway[0](hidden_states)
        if self.out_features:
            output_hidden_states = torch.empty(hidden_states.size(0), hidden_states.size(1), self.out_features, dtype=hidden_states.dtype)
        else:
            output_hidden_states = torch.empty_like(hidden_states)
        for idx, subway in enumerate(self.multiway):
            local_indices = multiway_indices.eq(idx).nonzero(as_tuple=True)
            hidden = hidden_states[local_indices].unsqueeze(1).contiguous()
            if hidden.numel():
                output = subway(hidden)
                if isinstance(output, tuple):
                    output = output[0]
                output = output.squeeze(1)
                output_hidden_states[local_indices] = output
        return output_hidden_states.contiguous()


def _rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


apply_rotary_emb_func = None


def apply_rotary_pos_emb(t, freqs):
    """ Apply rotary embedding to the first rotary_dim of the iput
    Arguments:
      t (tensor(batch_size, seq_len, n_head, head_dim)):
        the input embedding/hidden states
      freqs (list[tensor(1, seq_len, 1, rotary_dim), tensor(1, seq_len, 1, rotary_dim)]):
        the cached cos/sin position embeddings
    """
    rot_dim = freqs[0].shape[-1]
    cos, sin = freqs
    t_float = t.float()
    if apply_rotary_emb_func is not None and t.is_cuda:
        cos = cos.squeeze(0).squeeze(1)[:, :rot_dim // 2]
        sin = sin.squeeze(0).squeeze(1)[:, :rot_dim // 2]
        return apply_rotary_emb_func(t_float, cos, sin).type_as(t)
    else:
        t_rot, t_pass = t_float[..., :rot_dim], t_float[..., rot_dim:]
        t_rot = t_rot * cos + _rotate_half(t_rot) * sin
        return torch.cat((t_rot, t_pass), dim=-1).type_as(t)


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


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: 'LlamaConfig'):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(f'hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads}).')
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = MultiwayNetwork(module_provider=partial(nn.Linear, in_features=self.hidden_size, out_features=self.num_key_value_heads * self.head_dim, bias=config.attention_bias))
        self.v_proj = MultiwayNetwork(module_provider=partial(nn.Linear, in_features=self.hidden_size, out_features=self.num_key_value_heads * self.head_dim, bias=config.attention_bias))
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta)
        else:
            scaling_type = self.config.rope_scaling['type']
            scaling_factor = self.config.rope_scaling['factor']
            if scaling_type == 'linear':
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor, base=self.rope_theta)
            elif scaling_type == 'dynamic':
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor, base=self.rope_theta)
            else:
                raise ValueError(f'Unknown RoPE scaling type {scaling_type}')

    def _shape(self, tensor: 'torch.Tensor', seq_len: 'int', bsz: 'int'):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: 'torch.Tensor', modality_indicators: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]'=None, position_ids: 'Optional[torch.LongTensor]'=None, past_key_value: 'Optional[Tuple[torch.Tensor]]'=None, output_attentions: 'bool'=False, use_cache: 'bool'=False, padding_mask: 'Optional[torch.LongTensor]'=None) ->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states, modality_indicators)
        value_states = self.v_proj(hidden_states, modality_indicators)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None
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
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {bsz, self.num_heads, q_len, self.head_dim}, but is {attn_output.size()}')
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(nn.Module):

    def __init__(self, config: 'LlamaConfig'):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        mlp_kwargs = {'config': config, 'hidden_size': config.hidden_size, 'intermediate_size': config.intermediate_size, 'hidden_act': config.hidden_act}
        valid_params = set(inspect.signature(LlamaMLP.__init__).parameters.keys()) - {'self'}
        mlp_kwargs = {k: v for k, v in mlp_kwargs.items() if k in valid_params}
        self.mlp = LlamaMLP(**mlp_kwargs)
        self.input_layernorm = MultiwayNetwork(module_provider=partial(LlamaRMSNorm, hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attention_layernorm = MultiwayNetwork(module_provider=partial(LlamaRMSNorm, hidden_size=config.hidden_size, eps=config.rms_norm_eps))

    def forward(self, hidden_states: 'torch.Tensor', modality_indicators: 'torch.Tensor'=None, attention_mask: 'Optional[torch.Tensor]'=None, position_ids: 'Optional[torch.LongTensor]'=None, past_key_value: 'Optional[Tuple[torch.Tensor]]'=None, output_attentions: 'Optional[bool]'=False, use_cache: 'Optional[bool]'=False) ->Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
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
        hidden_states = self.input_layernorm(hidden_states, modality_indicators)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(hidden_states=hidden_states, modality_indicators=modality_indicators, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states, modality_indicators)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = hidden_states,
        if output_attentions:
            outputs += self_attn_weights,
        if use_cache:
            outputs += present_key_value,
        return outputs


flash_attn_unpadded_func = None


class FlashSelfAttention(torch.nn.Module):

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        assert flash_attn_unpadded_func is not None, 'Please install FlashAttention first, e.g., with pip install flash-attn'
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def unpad_input(self, hidden_states, attention_mask):
        valid_mask = attention_mask.squeeze(1).squeeze(1).eq(0)
        seqlens_in_batch = valid_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(valid_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
        hidden_states = hidden_states[indices]
        return hidden_states, indices, cu_seqlens, max_seqlen_in_batch

    def pad_input(self, hidden_states, indices, batch, seqlen):
        output = torch.zeros(batch * seqlen, *hidden_states.shape[1:], device=hidden_states.device, dtype=hidden_states.dtype)
        output[indices] = hidden_states
        return rearrange(output, '(b s) ... -> b s ...', b=batch)

    def forward(self, q, k, v, attention_mask=None):
        assert all(i.dtype in [torch.float16, torch.bfloat16] for i in (q, k, v))
        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]
        seqlen_out = seqlen_q
        if flash_attn_func is not None and batch_size == 1:
            dropout_p = self.dropout_p if self.training else 0
            output = flash_attn_func(q, k, v, dropout_p, softmax_scale=self.softmax_scale, causal=self.causal)
            return output
        q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q.device)
        if batch_size > 1 and attention_mask is not None:
            k, indices_k, cu_seqlens_k, seqlen_k = self.unpad_input(k, attention_mask)
            if q.size(0) == v.size(0):
                q = q[indices_k]
                cu_seqlens_q = cu_seqlens_k
                seqlen_q = seqlen_k
            v = v[indices_k]
        else:
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=q.device)
        if self.training:
            assert seqlen_k == seqlen_q
            is_causal = self.causal
            dropout_p = self.dropout_p
        else:
            is_causal = seqlen_q == seqlen_k
            dropout_p = 0
        output = flash_attn_unpadded_func(q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k, dropout_p, softmax_scale=self.softmax_scale, causal=is_causal)
        if batch_size > 1 and attention_mask is not None and seqlen_q == seqlen_k:
            output = self.pad_input(output, indices_k, batch_size, seqlen_out)
        else:
            new_shape = (batch_size, output.shape[0] // batch_size) + output.shape[1:]
            output = output.view(new_shape)
        return output


SUPPORT_TORCH2 = hasattr(torch, '__version__') and int(torch.__version__.split('.')[0]) >= 2


_ERROR_INPUT_CPU_QUERY_WITH_FLASH_ATTN_ACTIVATED = """We detect you have activated flash attention support, but running model computation on CPU. Please make sure that your input data has been placed on GPU. If you actually want to run CPU computation, please following the readme and set device_map="cpu" to disable flash attention when loading the model (calling AutoModelForCausalLM.from_pretrained).
检测到您的模型已激活了flash attention支持，但正在执行CPU运算任务。如使用flash attention，请您确认模型输入已经传到GPU上。如果您确认要执行CPU运算，请您在载入模型（调用AutoModelForCausalLM.from_pretrained）时，按照readme说法，指定device_map="cpu"以禁用flash attention。
"""


def dequantize_cache_torch(qdata, scale, zero):
    data = scale * (qdata - zero)
    return data


def quantize_cache_v(fdata, bits, qmax, qmin):
    qtype = torch.uint8
    device = fdata.device
    shape = fdata.shape
    fdata_cal = torch.flatten(fdata, 2)
    fmax = torch.amax(fdata_cal, dim=-1, keepdim=True)
    fmin = torch.amin(fdata_cal, dim=-1, keepdim=True)
    if qmax.device != fmax.device:
        qmax = qmax
        qmin = qmin
    scale = (fmax - fmin) / (qmax - qmin)
    zero = qmin - fmin / scale
    scale = scale.unsqueeze(-1).repeat(1, 1, shape[2], 1).contiguous()
    zero = zero.unsqueeze(-1).repeat(1, 1, shape[2], 1).contiguous()
    res_data = fdata / scale + zero
    qdata = torch.clamp(res_data, qmin, qmax)
    return qdata.contiguous(), scale, zero


class QWenAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.register_buffer('masked_bias', torch.tensor(-10000.0), persistent=False)
        self.seq_length = config.seq_length
        self.hidden_size = config.hidden_size
        self.split_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.use_flash_attn = config.use_flash_attn
        self.scale_attn_weights = True
        self.projection_size = config.kv_channels * config.num_attention_heads
        assert self.projection_size % config.num_attention_heads == 0
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.multiway = config.multiway
        if self.multiway:
            self.c_attn = MultiwayNetwork(module_provider=partial(nn.Linear, in_features=config.hidden_size, out_features=3 * self.projection_size), out_features=3 * self.projection_size)
        else:
            self.c_attn = nn.Linear(config.hidden_size, 3 * self.projection_size)
        self.c_proj = nn.Linear(config.hidden_size, self.projection_size, bias=not config.no_bias)
        self.is_fp32 = not (config.bf16 or config.fp16)
        if self.use_flash_attn and flash_attn_unpadded_func is not None and not self.is_fp32:
            self.core_attention_flash = FlashSelfAttention(causal=True, attention_dropout=config.attn_dropout_prob)
        self.bf16 = config.bf16
        self.use_dynamic_ntk = config.use_dynamic_ntk
        self.use_logn_attn = config.use_logn_attn
        logn_list = [(math.log(i, self.seq_length) if i > self.seq_length else 1) for i in range(1, 32768)]
        logn_tensor = torch.tensor(logn_list)[None, :, None, None]
        self.register_buffer('logn_tensor', logn_tensor, persistent=False)
        self.attn_dropout = nn.Dropout(config.attn_dropout_prob)
        self.softmax_in_fp32 = config.softmax_in_fp32 if hasattr(config, 'softmax_in_fp32') else False
        self.use_cache_quantization = config.use_cache_quantization if hasattr(config, 'use_cache_quantization') else False
        self.use_cache_kernel = config.use_cache_kernel if hasattr(config, 'use_cache_kernel') else False
        cache_dtype = torch.float
        if self.bf16:
            cache_dtype = torch.bfloat16
        elif config.fp16:
            cache_dtype = torch.float16
        self.cache_qmax = torch.tensor(torch.iinfo(torch.uint8).max, dtype=cache_dtype)
        self.cache_qmin = torch.tensor(torch.iinfo(torch.uint8).min, dtype=cache_dtype)
        if config.use_cache_quantization and config.use_cache_kernel:
            module_root = pathlib.Path(__file__).parent
            src_files = 'cache_autogptq_cuda_256.cpp', 'cache_autogptq_cuda_kernel_256.cu'
            if any(not (module_root / src).is_file() for src in src_files):
                warnings.warn('KV cache kernel source files (.cpp and .cu) not found.')
                self.cache_kernels = None
            else:
                try:
                    self.cache_kernels = cache_autogptq_cuda_256
                except ImportError:
                    warnings.warn('Failed to import KV cache kernels.')
                    self.cache_kernels = None

    def _attn(self, query, key, value, causal_mask=None, attention_mask=None, head_mask=None):
        device = query.device
        if self.use_cache_quantization:
            qk, qk_scale, qk_zero = key
            if self.use_cache_kernel and self.cache_kernels is not None:
                shape = query.shape[:-1] + (qk.shape[-2],)
                attn_weights = torch.zeros(shape, dtype=torch.float16, device=device)
                self.cache_kernels.vecquant8matmul_batched_faster_old(query.contiguous() if query.dtype == torch.float16 else query.contiguous(), qk.transpose(-1, -2).contiguous(), attn_weights, qk_scale.contiguous() if qk_scale.dtype == torch.float16 else qk_scale.contiguous(), qk_zero.contiguous() if qk_zero.dtype == torch.float16 else qk_zero.contiguous())
            else:
                key = dequantize_cache_torch(qk, qk_scale, qk_zero)
                attn_weights = torch.matmul(query, key.transpose(-1, -2))
        else:
            attn_weights = torch.matmul(query, key.transpose(-1, -2))
        if self.scale_attn_weights:
            if self.use_cache_quantization:
                size_temp = value[0].size(-1)
            else:
                size_temp = value.size(-1)
            attn_weights = attn_weights / size_temp ** 0.5
        mask_value = torch.finfo(attn_weights.dtype).min
        if causal_mask is not None:
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        if self.softmax_in_fp32:
            attn_weights = nn.functional.softmax(attn_weights.float(), dim=-1)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.type(query.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        if self.use_cache_quantization:
            qv, qv_scale, qv_zero = value
            if self.use_cache_kernel and self.cache_kernels is not None:
                shape = attn_weights.shape[:-1] + (query.shape[-1],)
                attn_output = torch.zeros(shape, dtype=torch.float16, device=device)
                self.cache_kernels.vecquant8matmul_batched_column_compression_faster_old(attn_weights.contiguous() if attn_weights.dtype == torch.float16 else attn_weights.contiguous(), qv.contiguous(), attn_output, qv_scale.contiguous() if qv_scale.dtype == torch.float16 else qv_scale.contiguous(), qv_zero.contiguous() if qv_zero.dtype == torch.float16 else qv_zero.contiguous())
                if attn_output.dtype != query.dtype:
                    attn_output = attn_output
                    attn_weights = attn_weights
            else:
                value = dequantize_cache_torch(qv, qv_scale, qv_zero)
                attn_output = torch.matmul(attn_weights, value)
        else:
            attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2)
        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        tensor = tensor.contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(self, hidden_states: 'Optional[Tuple[torch.FloatTensor]]', modality_indicators=None, rotary_pos_emb_list: 'Optional[List[List[torch.Tensor]]]'=None, layer_past: 'Optional[Tuple[torch.Tensor]]'=None, attention_mask: 'Optional[torch.FloatTensor]'=None, head_mask: 'Optional[torch.FloatTensor]'=None, encoder_hidden_states: 'Optional[torch.Tensor]'=None, encoder_attention_mask: 'Optional[torch.FloatTensor]'=None, output_attentions: 'Optional[bool]'=False, use_cache: 'Optional[bool]'=False):
        if self.multiway:
            mixed_x_layer = self.c_attn(hidden_states, modality_indicators)
        else:
            mixed_x_layer = self.c_attn(hidden_states)
        query, key, value = mixed_x_layer.split(self.split_size, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        if rotary_pos_emb_list is not None:
            cur_len = query.shape[1]
            if len(rotary_pos_emb_list) == 1:
                rotary_pos_emb = rotary_pos_emb_list[0]
                rotary_pos_emb = [i[:, -cur_len:, :, :] for i in rotary_pos_emb]
                rotary_pos_emb = (rotary_pos_emb,) * 2
                q_pos_emb, k_pos_emb = rotary_pos_emb
                query = apply_rotary_pos_emb(query, q_pos_emb)
                key = apply_rotary_pos_emb(key, k_pos_emb)
            else:
                query_list = []
                key_list = []
                for i, rotary_pos_emb in enumerate(rotary_pos_emb_list):
                    rotary_pos_emb = [i[:, -cur_len:, :, :] for i in rotary_pos_emb]
                    rotary_pos_emb = (rotary_pos_emb,) * 2
                    q_pos_emb, k_pos_emb = rotary_pos_emb
                    query_list += [apply_rotary_pos_emb(query[i:i + 1, :, :], q_pos_emb)]
                    key_list += [apply_rotary_pos_emb(key[i:i + 1, :, :], k_pos_emb)]
                query = torch.cat(query_list, dim=0)
                key = torch.cat(key_list, dim=0)
        if self.use_cache_quantization:
            key = quantize_cache_v(key.permute(0, 2, 1, 3), bits=8, qmin=self.cache_qmin, qmax=self.cache_qmax)
            value = quantize_cache_v(value.permute(0, 2, 1, 3), bits=8, qmin=self.cache_qmin, qmax=self.cache_qmax)
        if layer_past is not None:
            past_key, past_value = layer_past[0], layer_past[1]
            if self.use_cache_quantization:
                key = torch.cat((past_key[0], key[0]), dim=2), torch.cat((past_key[1], key[1]), dim=2), torch.cat((past_key[2], key[2]), dim=2)
                value = torch.cat((past_value[0], value[0]), dim=2), torch.cat((past_value[1], value[1]), dim=2), torch.cat((past_value[2], value[2]), dim=2)
            else:
                key = torch.cat((past_key, key), dim=1)
                value = torch.cat((past_value, value), dim=1)
        if use_cache:
            present = key, value
        else:
            present = None
        key_size = key[0].size(2) if self.use_cache_quantization else key.size(1)
        if key_size > self.seq_length and self.use_logn_attn and not self.training:
            if self.use_cache_quantization:
                seq_start = key[0].size(2) - query.size(1)
                seq_end = key[0].size(2)
            else:
                seq_start = key.size(1) - query.size(1)
                seq_end = key.size(1)
            logn_tensor = self.logn_tensor[:, seq_start:seq_end, :, :].type_as(query)
            query = query * logn_tensor.expand_as(query)
        if self.use_flash_attn and flash_attn_unpadded_func is not None and not self.is_fp32 and query.is_cuda:
            q, k, v = query, key, value
            attn_output = self.core_attention_flash(q, k, v, attention_mask=attention_mask)
        else:
            key_size = key[0].size(2) if self.use_cache_quantization else key.size(1)
            if query.size(1) == key_size:
                causal_mask = torch.tril(torch.ones((key_size, key_size), dtype=torch.bool, device=query.device)).view(1, 1, key_size, key_size)
            else:
                causal_mask = None
            query = query.permute(0, 2, 1, 3)
            if not self.use_cache_quantization:
                key = key.permute(0, 2, 1, 3)
                value = value.permute(0, 2, 1, 3)
            if causal_mask is None and self.use_flash_attn and flash_attn_unpadded_func is not None and not self.is_fp32 and not query.is_cuda:
                raise Exception(_ERROR_INPUT_CPU_QUERY_WITH_FLASH_ATTN_ACTIVATED)
            if not self.use_cache_quantization and SUPPORT_TORCH2:
                if attention_mask is not None:
                    attention_mask = attention_mask.expand(-1, -1, query.size(2), -1)
                    if causal_mask is not None:
                        attention_mask = attention_mask.masked_fill(~causal_mask, torch.finfo(query.dtype).min)
                else:
                    attention_mask = causal_mask
                attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask).transpose(1, 2)
                attn_weight = None
            else:
                attn_output, attn_weight = self._attn(query, key, value, causal_mask, attention_mask, head_mask)
        context_layer = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(context_layer)
        outputs = attn_output, present
        if output_attentions:
            if self.use_flash_attn and flash_attn_unpadded_func is not None and not self.is_fp32:
                raise ValueError('Cannot output attentions while using flash-attn')
            elif not self.use_cache_quantization and SUPPORT_TORCH2:
                raise ValueError('Cannot output attentions while using scaled_dot_product_attention')
            else:
                outputs += attn_weight,
        return outputs


class QWenMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size // 2, bias=not config.no_bias)
        self.w2 = nn.Linear(config.hidden_size, config.intermediate_size // 2, bias=not config.no_bias)
        ff_dim_in = config.intermediate_size // 2
        self.c_proj = nn.Linear(ff_dim_in, config.hidden_size, bias=not config.no_bias)

    def forward(self, hidden_states):
        a1 = self.w1(hidden_states)
        a2 = self.w2(hidden_states)
        intermediate_parallel = a1 * F.silu(a2)
        output = self.c_proj(intermediate_parallel)
        return output


rms_norm = None


class RMSNorm(torch.nn.Module):

    def __init__(self, dim: 'int', eps: 'float'=1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if rms_norm is not None and x.is_cuda:
            return rms_norm(x, self.weight, self.eps)
        else:
            output = self._norm(x.float()).type_as(x)
            return output * self.weight


class QWenBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        self.bf16 = config.bf16
        self.ln_1 = RMSNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = QWenAttention(config)
        self.ln_2 = RMSNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = QWenMLP(config)

    def forward(self, hidden_states: 'Optional[Tuple[torch.FloatTensor]]', modality_indicators=None, rotary_pos_emb_list: 'Optional[List[List[torch.Tensor]]]'=None, layer_past: 'Optional[Tuple[torch.Tensor]]'=None, attention_mask: 'Optional[torch.FloatTensor]'=None, head_mask: 'Optional[torch.FloatTensor]'=None, encoder_hidden_states: 'Optional[torch.Tensor]'=None, encoder_attention_mask: 'Optional[torch.FloatTensor]'=None, use_cache: 'Optional[bool]'=False, output_attentions: 'Optional[bool]'=False):
        layernorm_output = self.ln_1(hidden_states)
        attn_outputs = self.attn(layernorm_output, modality_indicators=modality_indicators, rotary_pos_emb_list=rotary_pos_emb_list, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        residual = hidden_states
        layernorm_input = attn_output + residual
        layernorm_output = self.ln_2(layernorm_input)
        residual = layernorm_input
        mlp_output = self.mlp(layernorm_output)
        hidden_states = residual + mlp_output
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        return outputs


class RotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        if importlib.util.find_spec('einops') is None:
            raise RuntimeError('einops is required for Rotary Embedding')
        self._rotary_pos_emb_cache = None
        self._seq_len_cached = 0
        self._ntk_alpha_cached = 1.0
        self._ntk_alpha_cached_list = [1.0]

    def update_rotary_pos_emb_cache(self, seqlen, ntk_alpha=1.0):
        if seqlen > self._seq_len_cached or ntk_alpha != self._ntk_alpha_cached:
            base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))
            self.inv_freq = 1.0 / base ** (torch.arange(0, self.dim, 2, device=self.inv_freq.device).float() / self.dim)
            self._seq_len_cached = max(2 * seqlen, 16)
            self._ntk_alpha_cached = ntk_alpha
            seq = torch.arange(self._seq_len_cached, device=self.inv_freq.device)
            freqs = torch.outer(seq.type_as(self.inv_freq), self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            emb = rearrange(emb, 'n d -> 1 n 1 d')
            cos, sin = emb.cos(), emb.sin()
            self._rotary_pos_emb_cache = [cos, sin]

    def forward(self, max_seq_len, ntk_alpha=1.0):
        self.update_rotary_pos_emb_cache(max_seq_len, ntk_alpha)
        cos, sin = self._rotary_pos_emb_cache
        return [cos[:, :max_seq_len], sin[:, :max_seq_len]]


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (LayerNormFp32,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MplugOwlVisionAttention,
     lambda: ([], {'config': SimpleNamespace(hidden_size=4, num_attention_heads=4, attention_dropout=0.5)}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (MplugOwlVisualAbstractorCrossOutput,
     lambda: ([], {'config': SimpleNamespace(encoder_hidden_size=4, hidden_size=4, intermediate_size=4, layer_norm_eps=1)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MplugOwlVisualAbstractorMLP,
     lambda: ([], {'config': SimpleNamespace(hidden_size=4, intermediate_size=4, layer_norm_eps=1)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (QWenMLP,
     lambda: ([], {'config': SimpleNamespace(hidden_size=4, intermediate_size=4, no_bias=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (QuickGELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RMSNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

