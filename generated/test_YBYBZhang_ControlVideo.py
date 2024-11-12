
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


import torch


import torchvision


import torch.nn as nn


import torch.nn.functional as F


from typing import Optional


from typing import Callable


import math


from torch import nn


from typing import Any


from typing import Dict


from typing import List


from typing import Tuple


from typing import Union


from torch.nn import functional as F


import inspect


import torch.utils.checkpoint


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True), nn.PReLU(out_planes))


class IFBlock(nn.Module):

    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(conv(in_planes, c // 2, 3, 2, 1), conv(c // 2, c, 3, 2, 1))
        self.convblock0 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock1 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock2 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock3 = nn.Sequential(conv(c, c), conv(c, c))
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(c, c // 2, 4, 2, 1), nn.PReLU(c // 2), nn.ConvTranspose2d(c // 2, 4, 4, 2, 1))
        self.conv2 = nn.Sequential(nn.ConvTranspose2d(c, c // 2, 4, 2, 1), nn.PReLU(c // 2), nn.ConvTranspose2d(c // 2, 1, 4, 2, 1))

    def forward(self, x, flow, scale=1):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode='bilinear', align_corners=False, recompute_scale_factor=False)
        flow = F.interpolate(flow, scale_factor=1.0 / scale, mode='bilinear', align_corners=False, recompute_scale_factor=False) * 1.0 / scale
        feat = self.conv0(torch.cat((x, flow), 1))
        feat = self.convblock0(feat) + feat
        feat = self.convblock1(feat) + feat
        feat = self.convblock2(feat) + feat
        feat = self.convblock3(feat) + feat
        flow = self.conv1(feat)
        mask = self.conv2(feat)
        flow = F.interpolate(flow, scale_factor=scale, mode='bilinear', align_corners=False, recompute_scale_factor=False) * scale
        mask = F.interpolate(mask, scale_factor=scale, mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return flow, mask


class IndividualAttention(nn.Module):
    """
    A cross attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(self, query_dim: 'int', cross_attention_dim: 'Optional[int]'=None, heads: 'int'=8, dim_head: 'int'=64, dropout: 'float'=0.0, bias=False, upcast_attention: 'bool'=False, upcast_softmax: 'bool'=False, added_kv_proj_dim: 'Optional[int]'=None, norm_num_groups: 'Optional[int]'=None):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.sliceable_head_dim = heads
        self._slice_size = None
        self._use_memory_efficient_attention_xformers = False
        self.added_kv_proj_dim = added_kv_proj_dim
        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=inner_dim, num_groups=norm_num_groups, eps=1e-05, affine=True)
        else:
            self.group_norm = None
        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)
        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def set_attention_slice(self, slice_size):
        if slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(f'slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.')
        self._slice_size = slice_size

    def _attention(self, query, key, value, attention_mask=None):
        if self.upcast_attention:
            query = query.float()
            key = key.float()
        attention_scores = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device), query, key.transpose(-1, -2), beta=0, alpha=self.scale)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        if self.upcast_softmax:
            attention_scores = attention_scores.float()
        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _sliced_attention(self, query, key, value, sequence_length, dim, attention_mask):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros((batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype)
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]
            if self.upcast_attention:
                query_slice = query_slice.float()
                key_slice = key_slice.float()
            attn_slice = torch.baddbmm(torch.empty(slice_size, query.shape[1], key.shape[1], dtype=query_slice.dtype, device=query.device), query_slice, key_slice.transpose(-1, -2), beta=0, alpha=self.scale)
            if attention_mask is not None:
                attn_slice = attn_slice + attention_mask[start_idx:end_idx]
            if self.upcast_softmax:
                attn_slice = attn_slice.float()
            attn_slice = attn_slice.softmax(dim=-1)
            attn_slice = attn_slice
            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])
            hidden_states[start_idx:end_idx] = attn_slice
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _memory_efficient_attention_xformers(self, query, key, value, attention_mask):
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        batch_size, sequence_length, _ = hidden_states.shape
        encoder_hidden_states = encoder_hidden_states
        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)
        if self.added_kv_proj_dim is not None:
            raise NotImplementedError
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        curr_frame_index = torch.arange(video_length)
        key = rearrange(key, '(b f) d c -> b f d c', f=video_length)
        key = key[:, curr_frame_index]
        key = rearrange(key, 'b f d c -> (b f) d c')
        value = rearrange(value, '(b f) d c -> b f d c', f=video_length)
        value = value[:, curr_frame_index]
        value = rearrange(value, 'b f d c -> (b f) d c')
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)
        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            hidden_states = hidden_states
        elif self._slice_size is None or query.shape[0] // self._slice_size == 1:
            hidden_states = self._attention(query, key, value, attention_mask)
        else:
            hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class BasicTransformerBlock(nn.Module):

    def __init__(self, dim: 'int', num_attention_heads: 'int', attention_head_dim: 'int', dropout=0.0, cross_attention_dim: 'Optional[int]'=None, activation_fn: 'str'='geglu', num_embeds_ada_norm: 'Optional[int]'=None, attention_bias: 'bool'=False, only_cross_attention: 'bool'=False, upcast_attention: 'bool'=False):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        self.attn1 = IndividualAttention(query_dim=dim, heads=num_attention_heads, dim_head=attention_head_dim, dropout=dropout, bias=attention_bias, cross_attention_dim=cross_attention_dim if only_cross_attention else None, upcast_attention=upcast_attention)
        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        if cross_attention_dim is not None:
            self.attn2 = CrossAttention(query_dim=dim, cross_attention_dim=cross_attention_dim, heads=num_attention_heads, dim_head=attention_head_dim, dropout=dropout, bias=attention_bias, upcast_attention=upcast_attention)
        else:
            self.attn2 = None
        if cross_attention_dim is not None:
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        else:
            self.norm2 = None
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)
        self.norm_temp = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: 'bool', attention_op: 'Optional[Callable]'=None):
        if not is_xformers_available():
            None
            raise ModuleNotFoundError('Refer to https://github.com/facebookresearch/xformers for more information on how to install xformers', name='xformers')
        elif not torch.cuda.is_available():
            raise ValueError("torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only available for GPU ")
        else:
            try:
                _ = xformers.ops.memory_efficient_attention(torch.randn((1, 2, 40), device='cuda'), torch.randn((1, 2, 40), device='cuda'), torch.randn((1, 2, 40), device='cuda'))
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            if self.attn2 is not None:
                self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None):
        norm_hidden_states = self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        if self.only_cross_attention:
            hidden_states = self.attn1(norm_hidden_states, encoder_hidden_states, attention_mask=attention_mask) + hidden_states
        else:
            hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask, video_length=video_length) + hidden_states
        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            hidden_states = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask) + hidden_states
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states


class FullyFrameAttention(nn.Module):
    """
    A cross attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(self, query_dim: 'int', cross_attention_dim: 'Optional[int]'=None, heads: 'int'=8, dim_head: 'int'=64, dropout: 'float'=0.0, bias=False, upcast_attention: 'bool'=False, upcast_softmax: 'bool'=False, added_kv_proj_dim: 'Optional[int]'=None, norm_num_groups: 'Optional[int]'=None):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.sliceable_head_dim = heads
        self._slice_size = None
        self._use_memory_efficient_attention_xformers = False
        self.added_kv_proj_dim = added_kv_proj_dim
        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=inner_dim, num_groups=norm_num_groups, eps=1e-05, affine=True)
        else:
            self.group_norm = None
        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)
        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def set_attention_slice(self, slice_size):
        if slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(f'slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.')
        self._slice_size = slice_size

    def _attention(self, query, key, value, attention_mask=None):
        if self.upcast_attention:
            query = query.float()
            key = key.float()
        attention_scores = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device), query, key.transpose(-1, -2), beta=0, alpha=self.scale)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        if self.upcast_softmax:
            attention_scores = attention_scores.float()
        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _sliced_attention(self, query, key, value, sequence_length, dim, attention_mask):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros((batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype)
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]
            if self.upcast_attention:
                query_slice = query_slice.float()
                key_slice = key_slice.float()
            attn_slice = torch.baddbmm(torch.empty(slice_size, query.shape[1], key.shape[1], dtype=query_slice.dtype, device=query.device), query_slice, key_slice.transpose(-1, -2), beta=0, alpha=self.scale)
            if attention_mask is not None:
                attn_slice = attn_slice + attention_mask[start_idx:end_idx]
            if self.upcast_softmax:
                attn_slice = attn_slice.float()
            attn_slice = attn_slice.softmax(dim=-1)
            attn_slice = attn_slice
            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])
            hidden_states[start_idx:end_idx] = attn_slice
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _memory_efficient_attention_xformers(self, query, key, value, attention_mask):
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, inter_frame=False):
        batch_size, sequence_length, _ = hidden_states.shape
        encoder_hidden_states = encoder_hidden_states
        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = rearrange(query, '(b f) d c -> b (f d) c', f=video_length)
        query = self.reshape_heads_to_batch_dim(query)
        if self.added_kv_proj_dim is not None:
            raise NotImplementedError
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        if inter_frame:
            key = rearrange(key, '(b f) d c -> b f d c', f=video_length)[:, [0, -1]]
            value = rearrange(value, '(b f) d c -> b f d c', f=video_length)[:, [0, -1]]
            key = rearrange(key, 'b f d c -> b (f d) c')
            value = rearrange(value, 'b f d c -> b (f d) c')
        else:
            key = rearrange(key, '(b f) d c -> b (f d) c', f=video_length)
            value = rearrange(value, '(b f) d c -> b (f d) c', f=video_length)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)
        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            hidden_states = hidden_states
        elif self._slice_size is None or query.shape[0] // self._slice_size == 1:
            hidden_states = self._attention(query, key, value, attention_mask)
        else:
            hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        hidden_states = rearrange(hidden_states, 'b (f d) c -> (b f) d c', f=video_length)
        return hidden_states


class InflatedConv3d(nn.Conv2d):

    def forward(self, x):
        video_length = x.shape[2]
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        x = super().forward(x)
        x = rearrange(x, '(b f) c h w -> b c f h w', f=video_length)
        return x


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(self, conditioning_embedding_channels: 'int', conditioning_channels: 'int'=3, block_out_channels: 'Tuple[int]'=(16, 32, 96, 256)):
        super().__init__()
        self.conv_in = InflatedConv3d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([])
        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(InflatedConv3d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))
        self.conv_out = zero_module(InflatedConv3d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1))

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)
        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)
        embedding = self.conv_out(embedding)
        return embedding


class Mish(torch.nn.Module):

    def forward(self, hidden_states):
        return hidden_states * torch.tanh(torch.nn.functional.softplus(hidden_states))


class ResnetBlock3D(nn.Module):

    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, temb_channels=512, groups=32, groups_out=None, pre_norm=True, eps=1e-06, non_linearity='swish', time_embedding_norm='default', output_scale_factor=1.0, use_in_shortcut=None):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = output_scale_factor
        if groups_out is None:
            groups_out = groups
        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = InflatedConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels is not None:
            if self.time_embedding_norm == 'default':
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == 'scale_shift':
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(f'unknown time_embedding_norm : {self.time_embedding_norm} ')
            self.time_emb_proj = torch.nn.Linear(temb_channels, time_emb_proj_out_channels)
        else:
            self.time_emb_proj = None
        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = InflatedConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if non_linearity == 'swish':
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == 'mish':
            self.nonlinearity = Mish()
        elif non_linearity == 'silu':
            self.nonlinearity = nn.SiLU()
        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut
        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = InflatedConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)
        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None, None]
        if temb is not None and self.time_embedding_norm == 'default':
            hidden_states = hidden_states + temb
        hidden_states = self.norm2(hidden_states)
        if temb is not None and self.time_embedding_norm == 'scale_shift':
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)
        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        return output_tensor


class UNetMidBlock3DCrossAttn(nn.Module):

    def __init__(self, in_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, attn_num_head_channels=1, output_scale_factor=1.0, cross_attention_dim=1280, dual_cross_attention=False, use_linear_projection=False, upcast_attention=False):
        super().__init__()
        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        resnets = [ResnetBlock3D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm)]
        attentions = []
        for _ in range(num_layers):
            if dual_cross_attention:
                raise NotImplementedError
            attentions.append(Transformer3DModel(attn_num_head_channels, in_channels // attn_num_head_channels, in_channels=in_channels, num_layers=1, cross_attention_dim=cross_attention_dim, norm_num_groups=resnet_groups, use_linear_projection=use_linear_projection, upcast_attention=upcast_attention))
            resnets.append(ResnetBlock3D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, attention_mask=None, inter_frame=False):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, inter_frame=inter_frame).sample
            hidden_states = resnet(hidden_states, temb)
        return hidden_states


class Downsample3D(nn.Module):

    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, name='conv'):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name
        if use_conv:
            conv = InflatedConv3d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            raise NotImplementedError
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
            raise NotImplementedError
        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)
        return hidden_states


class CrossAttnDownBlock3D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, attn_num_head_channels=1, cross_attention_dim=1280, output_scale_factor=1.0, downsample_padding=1, add_downsample=True, dual_cross_attention=False, use_linear_projection=False, only_cross_attention=False, upcast_attention=False):
        super().__init__()
        resnets = []
        attentions = []
        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock3D(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
            if dual_cross_attention:
                raise NotImplementedError
            attentions.append(Transformer3DModel(attn_num_head_channels, out_channels // attn_num_head_channels, in_channels=out_channels, num_layers=1, cross_attention_dim=cross_attention_dim, norm_num_groups=resnet_groups, use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention, upcast_attention=upcast_attention))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample3D(out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name='op')])
        else:
            self.downsamplers = None
        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, attention_mask=None, inter_frame=False):
        output_states = ()
        for resnet, attn in zip(self.resnets, self.attentions):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None, inter_frame=None):

                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict, inter_frame=inter_frame)
                        else:
                            return module(*inputs)
                    return custom_forward
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(attn, return_dict=False, inter_frame=inter_frame), hidden_states, encoder_hidden_states)[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, inter_frame=inter_frame).sample
            output_states += hidden_states,
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += hidden_states,
        return hidden_states, output_states


class DownBlock3D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, output_scale_factor=1.0, add_downsample=True, downsample_padding=1):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock3D(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample3D(out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name='op')])
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
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)
            output_states += hidden_states,
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += hidden_states,
        return hidden_states, output_states


class Upsample3D(nn.Module):

    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name='conv'):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        conv = None
        if use_conv_transpose:
            raise NotImplementedError
        elif use_conv:
            conv = InflatedConv3d(self.channels, self.out_channels, 3, padding=1)
        if name == 'conv':
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv_transpose:
            raise NotImplementedError
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=[1.0, 2.0, 2.0], mode='nearest')
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


class CrossAttnUpBlock3D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', prev_output_channel: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, attn_num_head_channels=1, cross_attention_dim=1280, output_scale_factor=1.0, add_upsample=True, dual_cross_attention=False, use_linear_projection=False, only_cross_attention=False, upcast_attention=False):
        super().__init__()
        resnets = []
        attentions = []
        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        for i in range(num_layers):
            res_skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock3D(in_channels=resnet_in_channels + res_skip_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
            if dual_cross_attention:
                raise NotImplementedError
            attentions.append(Transformer3DModel(attn_num_head_channels, out_channels // attn_num_head_channels, in_channels=out_channels, num_layers=1, cross_attention_dim=cross_attention_dim, norm_num_groups=resnet_groups, use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention, upcast_attention=upcast_attention))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample3D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None
        self.gradient_checkpointing = False

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, encoder_hidden_states=None, upsample_size=None, attention_mask=None, inter_frame=False):
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None, inter_frame=None):

                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict, inter_frame=inter_frame)
                        else:
                            return module(*inputs)
                    return custom_forward
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(attn, return_dict=False, inter_frame=inter_frame), hidden_states, encoder_hidden_states)[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, inter_frame=inter_frame).sample
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
        return hidden_states


class UpBlock3D(nn.Module):

    def __init__(self, in_channels: 'int', prev_output_channel: 'int', out_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, output_scale_factor=1.0, add_upsample=True):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            res_skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock3D(in_channels=resnet_in_channels + res_skip_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample3D(out_channels, use_conv=True, out_channels=out_channels)])
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
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
        return hidden_states


class TemporalConv1d(nn.Conv1d):

    def forward(self, x):
        b, c, f, h, w = x.shape
        y = rearrange(x.clone(), 'b c f h w -> (b h w) c f')
        y = super().forward(y)
        y = rearrange(y, '(b h w) c f -> b c f h w', b=b, h=h, w=w)
        return y


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (IFBlock,
     lambda: ([], {'in_planes': 4}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {})),
    (Mish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

