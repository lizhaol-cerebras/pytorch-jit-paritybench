
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


import warnings


from typing import List


from typing import Optional


import pandas as pd


import numpy as np


import torch


from torch import Tensor


from torch.nn.functional import interpolate


import torch.utils.checkpoint as checkpoint


from torch import nn


from typing import Any


from typing import Dict


from typing import Tuple


from typing import Union


import torch.nn as nn


import torch.utils.checkpoint


from torch.nn import LayerNorm


import logging


import math


import torch.nn.functional as F


from torch.nn import Parameter


import torch.distributed as distributed


from torch.autograd import Variable


from itertools import product


from itertools import permutations


from itertools import combinations_with_replacement


from itertools import chain


import inspect


from typing import Callable


import copy


import re


import random


import torchvision


import torchvision.transforms as T


from itertools import islice


from torch.utils.data import Dataset


from torch.nn import functional as F


from torchvision.transforms.functional import to_tensor


from itertools import groupby


from typing import Set


from typing import Type


def use_temporal(module, num_frames, x):
    if num_frames == 1:
        if isinstance(module, TransformerTemporalModel):
            return {'sample': x}
        else:
            return x


def custom_checkpoint(module, mode=None):
    if mode == None:
        raise ValueError('Mode for gradient checkpointing cannot be none.')
    custom_forward = None
    if mode == 'resnet':

        def custom_forward(hidden_states, temb):
            inputs = module(hidden_states, temb)
            return inputs
    if mode == 'attn':

        def custom_forward(hidden_states, encoder_hidden_states=None, cross_attention_kwargs=None):
            inputs = module(hidden_states, encoder_hidden_states, cross_attention_kwargs)
            return inputs
    if mode == 'temp':

        def custom_forward(hidden_states, num_frames=None):
            inputs = use_temporal(module, num_frames, hidden_states)
            if inputs is None:
                inputs = module(hidden_states, num_frames=num_frames)
            return inputs
    return custom_forward


g_c = checkpoint.checkpoint


def cross_attn_g_c(attn, temp_attn, resnet, temp_conv, hidden_states, encoder_hidden_states, cross_attention_kwargs, temb, num_frames, inverse_temp=False):

    def ordered_g_c(idx):
        if idx == 0:
            return g_c(custom_checkpoint(attn, mode='attn'), hidden_states, encoder_hidden_states, cross_attention_kwargs, use_reentrant=False)['sample']
        if idx == 1:
            return g_c(custom_checkpoint(temp_attn, mode='temp'), hidden_states, num_frames, use_reentrant=False)['sample']
        if idx == 2:
            return g_c(custom_checkpoint(resnet, mode='resnet'), hidden_states, temb, use_reentrant=False)
        if idx == 3:
            return g_c(custom_checkpoint(temp_conv, mode='temp'), hidden_states, num_frames, use_reentrant=False)
    if not inverse_temp:
        for idx in [0, 1, 2, 3]:
            hidden_states = ordered_g_c(idx)
    else:
        for idx in [2, 3, 0, 1]:
            hidden_states = ordered_g_c(idx)
    return hidden_states


def up_down_g_c(resnet, temp_conv, hidden_states, temb, num_frames):
    hidden_states = g_c(custom_checkpoint(resnet, mode='resnet'), hidden_states, temb, use_reentrant=False)
    hidden_states = g_c(custom_checkpoint(temp_conv, mode='temp'), hidden_states, num_frames, use_reentrant=False)
    return hidden_states


class UNetMidBlock3DCrossAttn(nn.Module):

    def __init__(self, in_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, attn_num_head_channels=1, output_scale_factor=1.0, cross_attention_dim=1280, dual_cross_attention=False, use_linear_projection=True, upcast_attention=False):
        super().__init__()
        self.gradient_checkpointing = False
        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        resnets = [ResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm)]
        temp_convs = [TemporalConvLayer(in_channels, in_channels)]
        attentions = []
        temp_attentions = []
        for _ in range(num_layers):
            attentions.append(Transformer2DModel(in_channels // attn_num_head_channels, attn_num_head_channels, in_channels=in_channels, num_layers=1, cross_attention_dim=cross_attention_dim, norm_num_groups=resnet_groups, use_linear_projection=use_linear_projection, upcast_attention=upcast_attention))
            temp_attentions.append(TransformerTemporalModel(in_channels // attn_num_head_channels, attn_num_head_channels, in_channels=in_channels, num_layers=1, cross_attention_dim=cross_attention_dim, norm_num_groups=resnet_groups))
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
            temp_convs.append(TemporalConvLayer(in_channels, in_channels))
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, attention_mask=None, num_frames=1, cross_attention_kwargs=None):
        if self.gradient_checkpointing:
            hidden_states = up_down_g_c(self.resnets[0], self.temp_convs[0], hidden_states, temb, num_frames)
        else:
            hidden_states = self.resnets[0](hidden_states, temb)
            hidden_states = self.temp_convs[0](hidden_states, num_frames=num_frames)
        for attn, temp_attn, resnet, temp_conv in zip(self.attentions, self.temp_attentions, self.resnets[1:], self.temp_convs[1:]):
            if self.gradient_checkpointing:
                hidden_states = cross_attn_g_c(attn, temp_attn, resnet, temp_conv, hidden_states, encoder_hidden_states, cross_attention_kwargs, temb, num_frames)
            else:
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs).sample
                if num_frames > 1:
                    hidden_states = temp_attn(hidden_states, num_frames=num_frames).sample
                hidden_states = resnet(hidden_states, temb)
                if num_frames > 1:
                    hidden_states = temp_conv(hidden_states, num_frames=num_frames)
        return hidden_states


class CrossAttnDownBlock3D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, attn_num_head_channels=1, cross_attention_dim=1280, output_scale_factor=1.0, downsample_padding=1, add_downsample=True, dual_cross_attention=False, use_linear_projection=False, only_cross_attention=False, upcast_attention=False):
        super().__init__()
        resnets = []
        attentions = []
        temp_attentions = []
        temp_convs = []
        self.gradient_checkpointing = False
        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
            temp_convs.append(TemporalConvLayer(out_channels, out_channels))
            attentions.append(Transformer2DModel(out_channels // attn_num_head_channels, attn_num_head_channels, in_channels=out_channels, num_layers=1, cross_attention_dim=cross_attention_dim, norm_num_groups=resnet_groups, use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention, upcast_attention=upcast_attention))
            temp_attentions.append(TransformerTemporalModel(out_channels // attn_num_head_channels, attn_num_head_channels, in_channels=out_channels, num_layers=1, cross_attention_dim=cross_attention_dim, norm_num_groups=resnet_groups))
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name='op')])
        else:
            self.downsamplers = None

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, attention_mask=None, num_frames=1, cross_attention_kwargs=None):
        output_states = ()
        for resnet, temp_conv, attn, temp_attn in zip(self.resnets, self.temp_convs, self.attentions, self.temp_attentions):
            if self.gradient_checkpointing:
                hidden_states = cross_attn_g_c(attn, temp_attn, resnet, temp_conv, hidden_states, encoder_hidden_states, cross_attention_kwargs, temb, num_frames, inverse_temp=True)
            else:
                hidden_states = resnet(hidden_states, temb)
                if num_frames > 1:
                    hidden_states = temp_conv(hidden_states, num_frames=num_frames)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs).sample
                if num_frames > 1:
                    hidden_states = temp_attn(hidden_states, num_frames=num_frames).sample
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
        temp_convs = []
        self.gradient_checkpointing = False
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
            temp_convs.append(TemporalConvLayer(out_channels, out_channels))
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name='op')])
        else:
            self.downsamplers = None

    def forward(self, hidden_states, temb=None, num_frames=1):
        output_states = ()
        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            if self.gradient_checkpointing:
                hidden_states = up_down_g_c(resnet, temp_conv, hidden_states, temb, num_frames)
            else:
                hidden_states = resnet(hidden_states, temb)
                if num_frames > 1:
                    hidden_states = temp_conv(hidden_states, num_frames=num_frames)
            output_states += hidden_states,
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += hidden_states,
        return hidden_states, output_states


class CrossAttnUpBlock3D(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', prev_output_channel: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, attn_num_head_channels=1, cross_attention_dim=1280, output_scale_factor=1.0, add_upsample=True, dual_cross_attention=False, use_linear_projection=False, only_cross_attention=False, upcast_attention=False):
        super().__init__()
        resnets = []
        temp_convs = []
        attentions = []
        temp_attentions = []
        self.gradient_checkpointing = False
        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        for i in range(num_layers):
            res_skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=resnet_in_channels + res_skip_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
            temp_convs.append(TemporalConvLayer(out_channels, out_channels))
            attentions.append(Transformer2DModel(out_channels // attn_num_head_channels, attn_num_head_channels, in_channels=out_channels, num_layers=1, cross_attention_dim=cross_attention_dim, norm_num_groups=resnet_groups, use_linear_projection=use_linear_projection, only_cross_attention=only_cross_attention, upcast_attention=upcast_attention))
            temp_attentions.append(TransformerTemporalModel(out_channels // attn_num_head_channels, attn_num_head_channels, in_channels=out_channels, num_layers=1, cross_attention_dim=cross_attention_dim, norm_num_groups=resnet_groups))
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, encoder_hidden_states=None, upsample_size=None, attention_mask=None, num_frames=1, cross_attention_kwargs=None):
        for resnet, temp_conv, attn, temp_attn in zip(self.resnets, self.temp_convs, self.attentions, self.temp_attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            if self.gradient_checkpointing:
                hidden_states = cross_attn_g_c(attn, temp_attn, resnet, temp_conv, hidden_states, encoder_hidden_states, cross_attention_kwargs, temb, num_frames, inverse_temp=True)
            else:
                hidden_states = resnet(hidden_states, temb)
                if num_frames > 1:
                    hidden_states = temp_conv(hidden_states, num_frames=num_frames)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs).sample
                if num_frames > 1:
                    hidden_states = temp_attn(hidden_states, num_frames=num_frames).sample
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
        return hidden_states


class UpBlock3D(nn.Module):

    def __init__(self, in_channels: 'int', prev_output_channel: 'int', out_channels: 'int', temb_channels: 'int', dropout: 'float'=0.0, num_layers: 'int'=1, resnet_eps: 'float'=1e-06, resnet_time_scale_shift: 'str'='default', resnet_act_fn: 'str'='swish', resnet_groups: 'int'=32, resnet_pre_norm: 'bool'=True, output_scale_factor=1.0, add_upsample=True):
        super().__init__()
        resnets = []
        temp_convs = []
        self.gradient_checkpointing = False
        for i in range(num_layers):
            res_skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=resnet_in_channels + res_skip_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
            temp_convs.append(TemporalConvLayer(out_channels, out_channels))
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None, num_frames=1):
        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            if self.gradient_checkpointing:
                hidden_states = up_down_g_c(resnet, temp_conv, hidden_states, temb, num_frames)
            else:
                hidden_states = resnet(hidden_states, temb)
                if num_frames > 1:
                    hidden_states = temp_conv(hidden_states, num_frames=num_frames)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
        return hidden_states


class GradMultiply(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class SamePad(nn.Module):

    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, :-self.remove]
        return x


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        return x * self.act(x)


class GLU_Linear(nn.Module):

    def __init__(self, input_dim, output_dim, glu_type='sigmoid', bias_in_glu=True):
        super(GLU_Linear, self).__init__()
        self.glu_type = glu_type
        self.output_dim = output_dim
        if glu_type == 'sigmoid':
            self.glu_act = torch.nn.Sigmoid()
        elif glu_type == 'swish':
            self.glu_act = Swish()
        elif glu_type == 'relu':
            self.glu_act = torch.nn.ReLU()
        elif glu_type == 'gelu':
            self.glu_act = torch.nn.GELU()
        if bias_in_glu:
            self.linear = nn.Linear(input_dim, output_dim * 2, True)
        else:
            self.linear = nn.Linear(input_dim, output_dim * 2, False)

    def forward(self, x):
        x = self.linear(x)
        if self.glu_type == 'bilinear':
            x = x[:, :, 0:self.output_dim] * x[:, :, self.output_dim:self.output_dim * 2]
        else:
            x = x[:, :, 0:self.output_dim] * self.glu_act(x[:, :, self.output_dim:self.output_dim * 2])
        return x


def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """
    if p <= 0:
        return module
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))
    is_conv = module.weight.ndim == 4
    if not is_conv:
        assert module.weight.size(1) % block_size == 0, 'Input features must be a multiple of block sizes'
    elif module.kernel_size == (1, 1):
        assert module.in_channels % block_size == 0, 'Input channels must be a multiple of block sizes'
    else:
        k = module.kernel_size[0] * module.kernel_size[1]
        assert k % block_size == 0, 'Kernel size must be a multiple of block size'

    def _forward_pre_hook(mod, input):
        if mod.training:
            if not is_conv:
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)
                mask = torch.zeros(in_features // block_size * out_features, device=weight.device)
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)
            else:
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(int(in_channels // block_size * out_channels), device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(weight.size(0), weight.size(1), device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
            mask = mask
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)
    module.register_forward_pre_hook(_forward_pre_hook)
    return module


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, self_attention=False, encoder_decoder_attention=False, q_noise=0.0, qn_block_size=8, has_relative_attention_bias=False, num_buckets=32, max_distance=128, gru_rel_pos=False, rescale_init=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)
        self.has_relative_attention_bias = has_relative_attention_bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
        self.head_dim = embed_dim // num_heads
        self.q_head_dim = self.head_dim
        self.k_head_dim = self.head_dim
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.scaling = self.head_dim ** -0.5
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert not self.self_attention or self.qkv_same_dim, 'Self-attention requires query, key and value to be of the same size'
        k_bias = True
        if rescale_init:
            k_bias = False
        k_embed_dim = embed_dim
        q_embed_dim = embed_dim
        self.k_proj = quant_noise(nn.Linear(self.kdim, k_embed_dim, bias=k_bias), q_noise, qn_block_size)
        self.v_proj = quant_noise(nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.q_proj = quant_noise(nn.Linear(embed_dim, q_embed_dim, bias=bias), q_noise, qn_block_size)
        self.out_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        self.add_zero_attn = add_zero_attn
        self.gru_rel_pos = gru_rel_pos
        if self.gru_rel_pos:
            self.grep_linear = nn.Linear(self.q_head_dim, 8)
            self.grep_a = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
        if self.has_relative_attention_bias:
            nn.init.xavier_normal_(self.relative_attention_bias.weight)

    def _relative_positions_bucket(self, relative_positions, bidirectional=True):
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        relative_buckets = 0
        if bidirectional:
            num_buckets = num_buckets // 2
            relative_buckets += (relative_positions > 0) * num_buckets
            relative_positions = torch.abs(relative_positions)
        else:
            relative_positions = -torch.min(relative_positions, torch.zeros_like(relative_positions))
        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact
        relative_postion_if_large = max_exact + torch.log(relative_positions.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        relative_postion_if_large = torch.min(relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1))
        relative_buckets += torch.where(is_small, relative_positions, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(relative_position, bidirectional=True)
        relative_position_bucket = relative_position_bucket
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1])
        return values

    def forward(self, query, key: 'Optional[Tensor]', value: 'Optional[Tensor]', key_padding_mask: 'Optional[Tensor]'=None, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]'=None, need_weights: 'bool'=True, static_kv: 'bool'=False, attn_mask: 'Optional[Tensor]'=None, before_softmax: 'bool'=False, need_head_weights: 'bool'=False, position_bias: 'Optional[Tensor]'=None) ->Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True
        is_tpu = query.device.type == 'xla'
        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]
        if self.has_relative_attention_bias and position_bias is None:
            position_bias = self.compute_bias(tgt_len, src_len)
            position_bias = position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1).view(bsz * self.num_heads, tgt_len, src_len)
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and 'prev_key' in saved_state:
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None
        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling
        alpha = 32
        q *= 1 / alpha
        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.q_head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.k_head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if saved_state is not None:
            if 'prev_key' in saved_state:
                _prev_key = saved_state['prev_key']
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if 'prev_value' in saved_state:
                _prev_value = saved_state['prev_value']
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: 'Optional[Tensor]' = None
            if 'prev_key_padding_mask' in saved_state:
                prev_key_padding_mask = saved_state['prev_key_padding_mask']
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(key_padding_mask=key_padding_mask, prev_key_padding_mask=prev_key_padding_mask, batch_size=bsz, src_len=k.size(1), static_kv=static_kv)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_key_padding_mask'] = key_padding_mask
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        assert k.size(1) == src_len
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = (attn_weights - attn_weights.max(dim=-1, keepdim=True)[0]) * alpha
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if before_softmax:
            return attn_weights, v, position_bias
        if position_bias is not None:
            attn_mask_rel_pos = position_bias
            if self.gru_rel_pos == 1:
                query_layer = q.view(bsz, self.num_heads, tgt_len, self.q_head_dim) * alpha / self.scaling
                _B, _H, _L, __ = query_layer.size()
                gate_a, gate_b = torch.sigmoid(self.grep_linear(query_layer).view(_B, _H, _L, 2, 4).sum(-1, keepdim=False)).chunk(2, dim=-1)
                gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                attn_mask_rel_pos = gate_a_1.view(bsz * self.num_heads, tgt_len, 1) * position_bias
            attn_mask_rel_pos = attn_mask_rel_pos.view(attn_weights.size())
            attn_weights = attn_weights + attn_mask_rel_pos
        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)
        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: 'Optional[Tensor]' = None
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                attn_weights = attn_weights.mean(dim=0)
        return attn, attn_weights, position_bias

    @staticmethod
    def _append_prev_key_padding_mask(key_padding_mask: 'Optional[Tensor]', prev_key_padding_mask: 'Optional[Tensor]', batch_size: 'int', src_len: 'int', static_kv: 'bool') ->Optional[Tensor]:
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), key_padding_mask.float()], dim=1)
        elif prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros((batch_size, src_len - prev_key_padding_mask.size(1)), device=prev_key_padding_mask.device)
                new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), filler.float()], dim=1)
            else:
                new_key_padding_mask = prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros((batch_size, src_len - key_padding_mask.size(1)), device=key_padding_mask.device)
                new_key_padding_mask = torch.cat([filler.float(), key_padding_mask.float()], dim=1)
            else:
                new_key_padding_mask = key_padding_mask.float()
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def _get_input_buffer(self, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]') ->Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, 'attn_state')
        if result is not None:
            return result
        else:
            empty_result: 'Dict[str, Optional[Tensor]]' = {}
            return empty_result

    def _set_input_buffer(self, incremental_state: 'Dict[str, Dict[str, Optional[Tensor]]]', buffer: 'Dict[str, Optional[Tensor]]'):
        return self.set_incremental_state(incremental_state, 'attn_state', buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: 'int', src_len: 'int', bsz: 'int'):
        return attn_weights


def gelu(x: 'torch.Tensor') ->torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)


def gelu_accurate(x):
    if not hasattr(gelu_accurate, '_a'):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return 0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))


def get_activation_fn(activation: 'str'):
    """Returns the activation function corresponding to `activation`"""
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return gelu
    elif activation == 'gelu_fast':
        warnings.warn('--activation-fn=gelu_fast has been renamed to gelu_accurate')
        return gelu_accurate
    elif activation == 'gelu_accurate':
        return gelu_accurate
    elif activation == 'tanh':
        return torch.tanh
    elif activation == 'linear':
        return lambda x: x
    elif activation == 'glu':
        return lambda x: x
    else:
        raise RuntimeError('--activation-fn {} not supported'.format(activation))


class TransformerSentenceEncoderLayer(nn.Module):

    def __init__(self, embedding_dim: 'float'=768, ffn_embedding_dim: 'float'=3072, num_attention_heads: 'float'=8, dropout: 'float'=0.1, attention_dropout: 'float'=0.1, activation_dropout: 'float'=0.1, activation_fn: 'str'='relu', layer_norm_first: 'bool'=False, deep_norm: 'bool'=False, has_relative_attention_bias: 'bool'=False, num_buckets: 'int'=0, max_distance: 'int'=0, rescale_init: 'bool'=False, gru_rel_pos: 'bool'=False, encoder_layers: 'int'=0) ->None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_name = activation_fn
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(self.embedding_dim, num_attention_heads, dropout=attention_dropout, self_attention=True, has_relative_attention_bias=has_relative_attention_bias, num_buckets=num_buckets, max_distance=max_distance, rescale_init=rescale_init, gru_rel_pos=gru_rel_pos)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.layer_norm_first = layer_norm_first
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        if self.activation_name == 'glu':
            self.fc1 = GLU_Linear(self.embedding_dim, ffn_embedding_dim, 'swish')
        else:
            self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        self.final_layer_norm = LayerNorm(self.embedding_dim)
        self.deep_norm = deep_norm
        if self.deep_norm:
            self.deep_norm_alpha = math.pow(2 * encoder_layers, 1 / 4)
        else:
            self.deep_norm_alpha = 1

    def forward(self, x: 'torch.Tensor', self_attn_mask: 'torch.Tensor'=None, self_attn_padding_mask: 'torch.Tensor'=None, need_weights: 'bool'=False, pos_bias=None):
        residual = x
        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn, pos_bias = self.self_attn(query=x, key=x, value=x, key_padding_mask=self_attn_padding_mask, need_weights=False, attn_mask=self_attn_mask, position_bias=pos_bias)
            x = self.dropout1(x)
            x = residual + x
            residual = x
            x = self.final_layer_norm(x)
            if self.activation_name == 'glu':
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn, pos_bias = self.self_attn(query=x, key=x, value=x, key_padding_mask=self_attn_padding_mask, need_weights=need_weights, attn_mask=self_attn_mask, position_bias=pos_bias)
            x = self.dropout1(x)
            x = residual * self.deep_norm_alpha + x
            x = self.self_attn_layer_norm(x)
            residual = x
            if self.activation_name == 'glu':
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual * self.deep_norm_alpha + x
            x = self.final_layer_norm(x)
        return x, attn, pos_bias


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02))
    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class TransformerEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        self.pos_conv = nn.Conv1d(self.embedding_dim, self.embedding_dim, kernel_size=args.conv_pos, padding=args.conv_pos // 2, groups=args.conv_pos_groups)
        dropout = 0
        std = math.sqrt(4 * (1.0 - dropout) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)
        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name='weight', dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())
        if hasattr(args, 'relative_position_embedding'):
            self.relative_position_embedding = args.relative_position_embedding
            self.num_buckets = args.num_buckets
            self.max_distance = args.max_distance
        else:
            self.relative_position_embedding = False
            self.num_buckets = 0
            self.max_distance = 0
        self.layers = nn.ModuleList([TransformerSentenceEncoderLayer(embedding_dim=self.embedding_dim, ffn_embedding_dim=args.encoder_ffn_embed_dim, num_attention_heads=args.encoder_attention_heads, dropout=self.dropout, attention_dropout=args.attention_dropout, activation_dropout=args.activation_dropout, activation_fn=args.activation_fn, layer_norm_first=args.layer_norm_first, deep_norm=args.deep_norm, has_relative_attention_bias=self.relative_position_embedding, num_buckets=self.num_buckets, max_distance=self.max_distance, gru_rel_pos=args.gru_rel_pos, encoder_layers=args.encoder_layers) for i in range(args.encoder_layers)])
        if self.relative_position_embedding:
            for i in range(1, args.encoder_layers):
                del self.layers[i].self_attn.relative_attention_bias
                self.layers[i].self_attn.relative_attention_bias = self.layers[0].self_attn.relative_attention_bias
        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop
        self.apply(init_bert_params)
        if args.deep_norm:
            deep_norm_beta = math.pow(8 * args.encoder_layers, -1 / 4)
            for i in range(args.encoder_layers):
                nn.init.xavier_normal_(self.layers[i].self_attn.k_proj.weight, gain=1)
                nn.init.xavier_normal_(self.layers[i].self_attn.v_proj.weight, gain=deep_norm_beta)
                nn.init.xavier_normal_(self.layers[i].self_attn.q_proj.weight, gain=1)
                nn.init.xavier_normal_(self.layers[i].self_attn.out_proj.weight, gain=deep_norm_beta)
                nn.init.xavier_normal_(self.layers[i].fc1.weight, gain=deep_norm_beta)
                nn.init.xavier_normal_(self.layers[i].fc2.weight, gain=deep_norm_beta)
        self.layer_wise_gradient_decay_ratio = getattr(args, 'layer_wise_gradient_decay_ratio', 1)

    def forward(self, x, padding_mask=None, layer=None):
        x, layers_sum, layers = self.extract_features(x, padding_mask, layer)
        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)
        return x, layers_sum, layers

    def extract_features(self, x, padding_mask=None, tgt_layer=None):
        if padding_mask is not None:
            x[padding_mask] = 0
        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x += x_conv
        if not self.layer_norm_first:
            x = self.layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)
        layers = []
        layer_results = []
        z = None
        if tgt_layer is not None:
            layer_results.append((x, z))
        r = None
        pos_bias = None
        for i, layer in enumerate(self.layers):
            if self.layer_wise_gradient_decay_ratio != 1.0:
                x = GradMultiply.apply(x, self.layer_wise_gradient_decay_ratio)
            dropout_probability = np.random.random()
            if not self.training or dropout_probability > self.layerdrop:
                x, z, pos_bias = layer(x, self_attn_padding_mask=padding_mask, need_weights=False, pos_bias=pos_bias)
            if tgt_layer is not None:
                layer_results.append((x, z))
            if i == tgt_layer:
                r = x
                break
            layers.append(x.transpose(0, 1))
        if r is not None:
            x = r
        x = x.transpose(0, 1)
        layers_cat = torch.cat(layers, dim=2)
        return x, layers_cat, layers


class BEATs(nn.Module):

    def __init__(self, cfg: 'BEATsConfig') ->None:
        super().__init__()
        logger.info(f'BEATs Config: {cfg.__dict__}')
        self.cfg = cfg
        self.embed = cfg.embed_dim
        self.post_extract_proj = nn.Linear(self.embed, cfg.encoder_embed_dim) if self.embed != cfg.encoder_embed_dim else None
        self.input_patch_size = cfg.input_patch_size
        self.patch_embedding = nn.Conv2d(1, self.embed, kernel_size=self.input_patch_size, stride=self.input_patch_size, bias=cfg.conv_bias)
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        assert not cfg.deep_norm or not cfg.layer_norm_first
        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)
        if cfg.finetuned_model:
            self.predictor_dropout = nn.Dropout(cfg.predictor_dropout)
            self.predictor = nn.Linear(cfg.encoder_embed_dim, cfg.predictor_class)
        else:
            self.predictor = None

    def forward_padding_mask(self, features: 'torch.Tensor', padding_mask: 'torch.Tensor') ->torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def preprocess(self, source: 'torch.Tensor', fbank_mean: 'float'=15.41663, fbank_std: 'float'=6.55582) ->torch.Tensor:
        fbanks = []
        for waveform in source:
            waveform = waveform.unsqueeze(0) * 2 ** 15
            fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)
        fbank = (fbank - fbank_mean) / (2 * fbank_std)
        return fbank

    def extract_features(self, source: 'torch.Tensor', padding_mask: 'Optional[torch.Tensor]'=None, fbank_mean: 'float'=15.41663, fbank_std: 'float'=6.55582):
        fbank = self.preprocess(source, fbank_mean=fbank_mean, fbank_std=fbank_std)
        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(fbank, padding_mask)
        fbank = fbank.unsqueeze(1)
        features = self.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        x = self.dropout_input(features)
        x, layers_sum, layers = self.encoder(x, padding_mask=padding_mask)
        if self.predictor is not None:
            x = self.predictor_dropout(x)
            logits = self.predictor(x)
            if padding_mask is not None and padding_mask.any():
                logits[padding_mask] = 0
                logits = logits.sum(dim=1)
                logits = logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(logits)
            else:
                logits = logits.mean(dim=1)
            lprobs = torch.sigmoid(logits)
            return lprobs, padding_mask
        else:
            return x, layers_sum, layers


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)
    return samples[indices]


def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device
    means = sample_vectors(samples, num_clusters)
    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') - rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim=-1)
        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)
        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]
        if use_cosine_sim:
            new_means = l2norm(new_means)
        means = torch.where(zero_mask[..., None], means, new_means)
    return means, bins


class EmbeddingEMA(nn.Module):

    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-05, kmeans_init=True, codebook_init_path=''):
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.eps = eps
        if codebook_init_path == '':
            if not kmeans_init:
                weight = torch.randn(num_tokens, codebook_dim)
                weight = l2norm(weight)
            else:
                weight = torch.zeros(num_tokens, codebook_dim)
            self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        else:
            None
            codebook_ckpt_weight = torch.load(codebook_init_path, map_location='cpu')
            weight = codebook_ckpt_weight.clone()
            self.register_buffer('initted', torch.Tensor([True]))
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)
        self.update = True

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return
        None
        embed, cluster_size = kmeans(data, self.num_tokens, 10, use_cosine_sim=True)
        self.weight.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg):
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized)


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=1 - decay)


def norm_ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=1 - decay)
    moving_avg.data.copy_(l2norm(moving_avg.data))


class NormEMAVectorQuantizer(nn.Module):

    def __init__(self, n_embed, embedding_dim, beta, decay=0.99, eps=1e-05, statistic_code_usage=True, kmeans_init=False, codebook_init_path=''):
        super().__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.beta = beta
        self.decay = decay
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps, kmeans_init, codebook_init_path)
        self.statistic_code_usage = statistic_code_usage
        if statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(n_embed))
        if distributed.is_available() and distributed.is_initialized():
            None
            self.all_reduce_fn = distributed.all_reduce
        else:
            self.all_reduce_fn = nn.Identity()

    def reset_cluster_size(self, device):
        if self.statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(self.num_tokens))
            self.cluster_size = self.cluster_size

    def forward(self, z):
        z = l2norm(z)
        z_flattened = z.reshape(-1, self.codebook_dim)
        self.embedding.init_embed_(z_flattened)
        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + self.embedding.weight.pow(2).sum(dim=1) - 2 * torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight)
        encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(encoding_indices).view(z.shape)
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)
        if not self.training:
            with torch.no_grad():
                cluster_size = encodings.sum(0)
                self.all_reduce_fn(cluster_size)
                ema_inplace(self.cluster_size, cluster_size, self.decay)
        if self.training and self.embedding.update:
            bins = encodings.sum(0)
            self.all_reduce_fn(bins)
            ema_inplace(self.cluster_size, bins, self.decay)
            zero_mask = bins == 0
            bins = bins.masked_fill(zero_mask, 1.0)
            embed_sum = z_flattened.t() @ encodings
            self.all_reduce_fn(embed_sum)
            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
            embed_normalized = l2norm(embed_normalized)
            embed_normalized = torch.where(zero_mask[..., None], self.embedding.weight, embed_normalized)
            norm_ema_inplace(self.embedding.weight, embed_normalized, self.decay)
        loss = self.beta * F.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()
        return z_q, loss, encoding_indices


class Tokenizers(nn.Module):

    def __init__(self, cfg: 'TokenizersConfig') ->None:
        super().__init__()
        logger.info(f'Tokenizers Config: {cfg.__dict__}')
        self.cfg = cfg
        self.embed = cfg.embed_dim
        self.post_extract_proj = nn.Linear(self.embed, cfg.encoder_embed_dim) if self.embed != cfg.encoder_embed_dim else None
        self.input_patch_size = cfg.input_patch_size
        self.patch_embedding = nn.Conv2d(1, self.embed, kernel_size=self.input_patch_size, stride=self.input_patch_size, bias=cfg.conv_bias)
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        assert not cfg.deep_norm or not cfg.layer_norm_first
        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)
        self.quantize = NormEMAVectorQuantizer(n_embed=cfg.quant_n, embedding_dim=cfg.quant_dim, beta=1.0, kmeans_init=True, decay=0.99)
        self.quant_n = cfg.quant_n
        self.quantize_layer = nn.Sequential(nn.Linear(cfg.encoder_embed_dim, cfg.encoder_embed_dim), nn.Tanh(), nn.Linear(cfg.encoder_embed_dim, cfg.quant_dim))

    def forward_padding_mask(self, features: 'torch.Tensor', padding_mask: 'torch.Tensor') ->torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def preprocess(self, source: 'torch.Tensor', fbank_mean: 'float'=15.41663, fbank_std: 'float'=6.55582) ->torch.Tensor:
        fbanks = []
        for waveform in source:
            waveform = waveform.unsqueeze(0) * 2 ** 15
            fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)
        fbank = (fbank - fbank_mean) / (2 * fbank_std)
        return fbank

    def extract_labels(self, source: 'torch.Tensor', padding_mask: 'Optional[torch.Tensor]'=None, fbank_mean: 'float'=15.41663, fbank_std: 'float'=6.55582):
        fbank = self.preprocess(source, fbank_mean=fbank_mean, fbank_std=fbank_std)
        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(fbank, padding_mask)
        fbank = fbank.unsqueeze(1)
        features = self.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        x = self.dropout_input(features)
        x, layer_results = self.encoder(x, padding_mask=padding_mask)
        quantize_input = self.quantize_layer(x)
        quantize_feature, embed_loss, embed_ind = self.quantize(quantize_input)
        return embed_ind


class Unary(nn.Module):

    def __init__(self, embed_size):
        """
            Captures local entity information
        :param embed_size:  the embedding dimension
        """
        super(Unary, self).__init__()
        self.embed = nn.Conv1d(embed_size, embed_size, 1)
        self.feature_reduce = nn.Conv1d(embed_size, 1, 1)

    def forward(self, X):
        X = X.transpose(1, 2)
        X_embed = self.embed(X)
        X_nl_embed = F.dropout(F.relu(X_embed))
        X_poten = self.feature_reduce(X_nl_embed)
        return X_poten.squeeze(1)


class Pairwise(nn.Module):

    def __init__(self, embed_x_size, x_spatial_dim=None, embed_y_size=None, y_spatial_dim=None):
        """
            Captures interaction between utilities or entities of the same utility
        :param embed_x_size: the embedding dimension of the first utility
        :param x_spatial_dim: the spatial dimension of the first utility for batch norm and weighted marginalization
        :param embed_y_size: the embedding dimension of the second utility (none for self-interactions)
        :param y_spatial_dim: the spatial dimension of the second utility for batch norm and weighted marginalization
        """
        super(Pairwise, self).__init__()
        embed_y_size = embed_y_size if y_spatial_dim is not None else embed_x_size
        self.y_spatial_dim = y_spatial_dim if y_spatial_dim is not None else x_spatial_dim
        self.embed_size = max(embed_x_size, embed_y_size)
        self.x_spatial_dim = x_spatial_dim
        self.embed_X = nn.Conv1d(embed_x_size, self.embed_size, 1)
        self.embed_Y = nn.Conv1d(embed_y_size, self.embed_size, 1)
        if x_spatial_dim is not None:
            self.normalize_S = nn.BatchNorm1d(self.x_spatial_dim * self.y_spatial_dim)
            self.margin_X = nn.Conv1d(self.y_spatial_dim, 1, 1)
            self.margin_Y = nn.Conv1d(self.x_spatial_dim, 1, 1)

    def forward(self, X, Y=None):
        X_t = X.transpose(1, 2)
        Y_t = Y.transpose(1, 2) if Y is not None else X_t
        X_embed = self.embed_X(X_t)
        Y_embed = self.embed_Y(Y_t)
        X_norm = F.normalize(X_embed)
        Y_norm = F.normalize(Y_embed)
        S = X_norm.transpose(1, 2).bmm(Y_norm)
        if self.x_spatial_dim is not None:
            S = self.normalize_S(S.view(-1, self.x_spatial_dim * self.y_spatial_dim)).view(-1, self.x_spatial_dim, self.y_spatial_dim)
            X_poten = self.margin_X(S.transpose(1, 2)).transpose(1, 2).squeeze(2)
            Y_poten = self.margin_Y(S).transpose(1, 2).squeeze(2)
        else:
            X_poten = S.mean(dim=2, keepdim=False)
            Y_poten = S.mean(dim=1, keepdim=False)
        if Y is None:
            return X_poten
        else:
            return X_poten, Y_poten


class Atten(nn.Module):

    def __init__(self, util_e, sharing_factor_weights=[], prior_flag=False, sizes=[], size_force=False, pairwise_flag=True, unary_flag=True, self_flag=True):
        """
            The class performs an attention on a given list of utilities representation.
        :param util_e: the embedding dimensions
        :param sharing_factor_weights:  To share weights, provide a dict of tuples:
         {idx: (num_utils, connected utils)
         Note, for efficiency, the shared utils (i.e., history, are connected to ans
          and question only.
         TODO: connections between shared utils
        :param prior_flag: is prior factor provided
        :param sizes: the spatial simension (used for batch-norm and weighted marginalization)
        :param size_force: force spatial size with adaptive avg pooling.
        :param pairwise_flag: use pairwise interaction between utilities
        :param unary_flag: use local information
        :param self_flag: use self interactions between utilitie's entities
        """
        super(Atten, self).__init__()
        self.util_e = util_e
        self.prior_flag = prior_flag
        self.n_utils = len(util_e)
        self.spatial_pool = nn.ModuleDict()
        self.un_models = nn.ModuleList()
        self.self_flag = self_flag
        self.pairwise_flag = pairwise_flag
        self.unary_flag = unary_flag
        self.size_force = size_force
        if len(sizes) == 0:
            sizes = [None for _ in util_e]
        self.sharing_factor_weights = sharing_factor_weights
        for idx, e_dim in enumerate(util_e):
            self.un_models.append(Unary(e_dim))
            if self.size_force:
                self.spatial_pool[str(idx)] = nn.AdaptiveAvgPool1d(sizes[idx])
        self.pp_models = nn.ModuleDict()
        for (idx1, e_dim_1), (idx2, e_dim_2) in combinations_with_replacement(enumerate(util_e), 2):
            if self.self_flag and idx1 == idx2:
                self.pp_models[str(idx1)] = Pairwise(e_dim_1, sizes[idx1])
            elif pairwise_flag:
                if idx1 in self.sharing_factor_weights:
                    if idx2 not in self.sharing_factor_weights[idx1][1]:
                        continue
                if idx2 in self.sharing_factor_weights:
                    if idx1 not in self.sharing_factor_weights[idx2][1]:
                        continue
                self.pp_models[str((idx1, idx2))] = Pairwise(e_dim_1, sizes[idx1], e_dim_2, sizes[idx2])
        self.reduce_potentials = nn.ModuleList()
        self.num_of_potentials = dict()
        self.default_num_of_potentials = 0
        if self.self_flag:
            self.default_num_of_potentials += 1
        if self.unary_flag:
            self.default_num_of_potentials += 1
        if self.prior_flag:
            self.default_num_of_potentials += 1
        for idx in range(self.n_utils):
            self.num_of_potentials[idx] = self.default_num_of_potentials
        """
         All other utilities
        """
        if pairwise_flag:
            for idx, (num_utils, connected_utils) in sharing_factor_weights:
                for c_u in connected_utils:
                    self.num_of_potentials[c_u] += num_utils
                    self.num_of_potentials[idx] += 1
            for k in self.num_of_potentials:
                if k not in self.sharing_factor_weights:
                    self.num_of_potentials[k] += self.n_utils - 1 - len(sharing_factor_weights)
        for idx in range(self.n_utils):
            self.reduce_potentials.append(nn.Conv1d(self.num_of_potentials[idx], 1, 1, bias=False))

    def forward(self, utils, priors=None):
        assert self.n_utils == len(utils)
        assert priors is None and not self.prior_flag or priors is not None and self.prior_flag and len(priors) == self.n_utils
        b_size = utils[0].size(0)
        util_factors = dict()
        attention = list()
        if self.size_force:
            for i, (num_utils, _) in self.sharing_factor_weights.items():
                if str(i) not in self.spatial_pool.keys():
                    continue
                else:
                    high_util = utils[i]
                    high_util = high_util.view(num_utils * b_size, high_util.size(2), high_util.size(3))
                    high_util = high_util.transpose(1, 2)
                    utils[i] = self.spatial_pool[str(i)](high_util).transpose(1, 2)
            for i in range(self.n_utils):
                if i in self.sharing_factor_weights or str(i) not in self.spatial_pool.keys():
                    continue
                utils[i] = utils[i].transpose(1, 2)
                utils[i] = self.spatial_pool[str(i)](utils[i]).transpose(1, 2)
                if self.prior_flag and priors[i] is not None:
                    priors[i] = self.spatial_pool[str(i)](priors[i].unsqueeze(1)).squeeze(1)
        for i, (num_utils, connected_list) in self.sharing_factor_weights:
            if self.unary_flag:
                util_factors.setdefault(i, []).append(self.un_models[i](utils[i]))
            if self.self_flag:
                util_factors.setdefault(i, []).append(self.pp_models[str(i)](utils[i]))
            if self.pairwise_flag:
                for j in connected_list:
                    other_util = utils[j]
                    expanded_util = other_util.unsqueeze(1).expand(b_size, num_utils, other_util.size(1), other_util.size(2)).contiguous().view(b_size * num_utils, other_util.size(1), other_util.size(2))
                    if i < j:
                        factor_ij, factor_ji = self.pp_models[str((i, j))](utils[i], expanded_util)
                    else:
                        factor_ji, factor_ij = self.pp_models[str((j, i))](expanded_util, utils[i])
                    util_factors[i].append(factor_ij)
                    util_factors.setdefault(j, []).append(factor_ji.view(b_size, num_utils, factor_ji.size(1)))
        for i in range(self.n_utils):
            if i in self.sharing_factor_weights:
                continue
            if self.unary_flag:
                util_factors.setdefault(i, []).append(self.un_models[i](utils[i]))
            if self.self_flag:
                util_factors.setdefault(i, []).append(self.pp_models[str(i)](utils[i]))
        if self.pairwise_flag:
            for i, j in combinations_with_replacement(range(self.n_utils), 2):
                if i in self.sharing_factor_weights or j in self.sharing_factor_weights:
                    continue
                if i == j:
                    continue
                else:
                    factor_ij, factor_ji = self.pp_models[str((i, j))](utils[i], utils[j])
                    util_factors.setdefault(i, []).append(factor_ij)
                    util_factors.setdefault(j, []).append(factor_ji)
        for i in range(self.n_utils):
            if self.prior_flag:
                prior = priors[i] if priors[i] is not None else torch.zeros_like(util_factors[i][0], requires_grad=False)
                util_factors[i].append(prior)
            util_factors[i] = torch.cat([(p if len(p.size()) == 3 else p.unsqueeze(1)) for p in util_factors[i]], dim=1)
            util_factors[i] = self.reduce_potentials[i](util_factors[i]).squeeze(1)
            util_factors[i] = F.softmax(util_factors[i], dim=1).unsqueeze(2)
            attention.append(torch.bmm(utils[i].transpose(1, 2), util_factors[i]).squeeze(2))
        return attention


class NaiveAttention(nn.Module):

    def __init__(self):
        """
            Used for ablation analysis - removing attention.
        """
        super(NaiveAttention, self).__init__()

    def forward(self, utils, priors):
        atten = []
        spatial_atten = []
        for u, p in zip(utils, priors):
            if type(u) is tuple:
                u = u[1]
                num_elements = u.shape[0]
                if p is not None:
                    u = u.view(-1, u.shape[-2], u.shape[-1])
                    p = p.view(-1, p.shape[-2], p.shape[-1])
                    spatial_atten.append(torch.bmm(p.transpose(1, 2), u).squeeze(2).view(num_elements, -1, u.shape[-2], u.shape[-1]))
                else:
                    spatial_atten.append(u.mean(2))
                continue
            if p is not None:
                atten.append(torch.bmm(u.transpose(1, 2), p.unsqueeze(2)).squeeze(2))
            else:
                atten.append(u.mean(1))
        return atten, spatial_atten


class FGA(nn.Module):

    def __init__(self, vocab_size, word_embed_dim, hidden_ques_dim, hidden_ans_dim, hidden_hist_dim, hidden_cap_dim, hidden_img_dim):
        """
        Factor Graph Attention
        :param vocab_size: vocabulary size
        :param word_embed_dim
        :param hidden_ques_dim:
        :param hidden_ans_dim:
        :param hidden_hist_dim:
        :param img_features_dim:
        """
        super(FGA, self).__init__()
        None
        self.hidden_ques_dim = hidden_ques_dim
        self.hidden_ans_dim = hidden_ans_dim
        self.hidden_cap_dim = hidden_cap_dim
        self.hidden_img_dim = hidden_img_dim
        self.hidden_hist_dim = hidden_hist_dim
        self.word_embedddings = nn.Embedding(vocab_size + 1 + 1, word_embed_dim, padding_idx=0)
        self.lstm_ques = nn.LSTM(word_embed_dim, self.hidden_ques_dim, batch_first=True)
        self.lstm_ans = nn.LSTM(word_embed_dim, self.hidden_ans_dim, batch_first=True)
        self.lstm_hist_ques = nn.LSTM(word_embed_dim, self.hidden_hist_dim, batch_first=True)
        self.lstm_hist_ans = nn.LSTM(word_embed_dim, self.hidden_hist_dim, batch_first=True)
        self.lstm_hist_cap = nn.LSTM(word_embed_dim, self.hidden_cap_dim, batch_first=True)
        self.qahistnet = nn.Sequential(nn.Linear(self.hidden_hist_dim * 2, self.hidden_hist_dim), nn.ReLU(inplace=True))
        self.concat_dim = self.hidden_ques_dim + self.hidden_ans_dim + self.hidden_ans_dim + self.hidden_img_dim + self.hidden_cap_dim + self.hidden_hist_dim * 9
        self.simnet = nn.Sequential(nn.Linear(self.concat_dim, self.concat_dim // 2, bias=False), nn.BatchNorm1d(self.concat_dim // 2), nn.ReLU(inplace=True), nn.Linear(self.concat_dim // 2, self.concat_dim // 4, bias=False), nn.BatchNorm1d(self.concat_dim // 4), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(self.concat_dim // 4, 1))
        sharing_factor_weights = {(4): (9, [0, 1]), (5): (9, [0, 1])}
        self.mul_atten = Atten(util_e=[self.hidden_ans_dim, self.hidden_ques_dim, self.hidden_cap_dim, self.hidden_img_dim, self.hidden_hist_dim, self.hidden_hist_dim], sharing_factor_weights=sharing_factor_weights, sizes=[100, 21, 41, 37, 21, 21], prior_flag=True, pairwise_flag=True)

    def forward(self, input_ques, input_ans, input_hist_ques, input_hist_ans, input_hist_cap, input_ques_length, input_ans_length, input_cap_length, i_e):
        """

        :param input_ques:
        :param input_ans:
        :param input_hist_ques:
        :param input_hist_ans:
        :param input_hist_cap:
        :param input_ques_length:
        :param input_ans_length:
        :param input_cap_length:
        :param i_e:
        :return:
        """
        n_options = input_ans.size()[1]
        batch_size = input_ques.size()[0]
        nqa_per_dial, nwords_per_qa = input_hist_ques.size()[1], input_hist_ques.size()[2]
        nwords_per_cap = input_hist_cap.size()[1]
        max_length_input_ans = input_ans.size()[-1]
        assert batch_size == input_hist_ques.size()[0] == input_hist_ans.size()[0] == input_ques.size()[0] == input_ans.size()[0] == input_hist_cap.size()[0]
        assert nqa_per_dial == input_hist_ques.size()[1] == input_hist_ans.size()[1]
        assert nwords_per_qa == input_hist_ques.size()[2] == input_hist_ans.size()[2]
        q_we = self.word_embedddings(input_ques)
        a_we = self.word_embedddings(input_ans.view(-1, max_length_input_ans))
        hq_we = self.word_embedddings(input_hist_ques.view(-1, nwords_per_qa))
        ha_we = self.word_embedddings(input_hist_ans.view(-1, nwords_per_qa))
        c_we = self.word_embedddings(input_hist_cap.view(-1, nwords_per_cap))
        """
            q_we = batch x 20 x embed_ques_dim
            a_we = 100*batch x 20 x embed_ans_dim
            hq_we = batch*nqa_per_dial, nwords_per_qa, embed_hist_dim
            ha_we = batch*nqa_per_dial, nwords_per_qa, embed_hist_dim
            c_we = batch*ncap_per_dial, nwords_per_cap, embed_hist_dim
        """
        self.lstm_ques.flatten_parameters()
        self.lstm_ans.flatten_parameters()
        self.lstm_hist_ques.flatten_parameters()
        self.lstm_hist_ans.flatten_parameters()
        self.lstm_hist_cap.flatten_parameters()
        i_feat = i_e
        q_seq, self.hidden_ques = self.lstm_ques(q_we)
        a_seq, self.hidden_ans = self.lstm_ans(a_we)
        hq_seq, self.hidden_hist_ques = self.lstm_hist_ques(hq_we)
        ha_seq, self.hidden_hist_ans = self.lstm_hist_ans(ha_we)
        cap_seq, self.hidden_cap = self.lstm_hist_cap(c_we)
        """
            length is used for attention prior
        """
        q_len = input_ques_length.data - 1
        c_len = input_cap_length.data.view(-1) - 1
        ans_index = torch.arange(0, n_options * batch_size).long()
        ans_len = input_ans_length.data.view(-1) - 1
        ans_seq = a_seq[ans_index, ans_len, :]
        ans_seq = ans_seq.view(batch_size, n_options, self.hidden_ans_dim)
        batch_index = torch.arange(0, batch_size).long()
        q_prior = torch.zeros(batch_size, q_seq.size(1))
        q_prior[batch_index, q_len] = 100
        c_prior = torch.zeros(batch_size, cap_seq.size(1))
        c_prior[batch_index, c_len] = 100
        ans_prior = torch.ones(batch_size, ans_seq.size(1))
        img_prior = torch.ones(batch_size, i_feat.size(1))
        ans_atten, ques_atten, cap_atten, img_atten, hq_atten, ha_atten = self.mul_atten([ans_seq, q_seq, cap_seq, i_feat, hq_seq, ha_seq], priors=[ans_prior, q_prior, c_prior, img_prior, None, None])
        """
            expand to answers based
        """
        ques_atten = torch.unsqueeze(ques_atten, 1).expand(batch_size, n_options, self.hidden_ques_dim)
        cap_atten = torch.unsqueeze(cap_atten, 1).expand(batch_size, n_options, self.hidden_cap_dim)
        img_atten = torch.unsqueeze(img_atten, 1).expand(batch_size, n_options, self.hidden_img_dim)
        ans_atten = torch.unsqueeze(ans_atten, 1).expand(batch_size, n_options, self.hidden_ans_dim)
        """
            combine history
        """
        input_qahistnet = torch.cat((hq_atten, ha_atten), 1)
        output_qahistnet = self.qahistnet(input_qahistnet)
        output_qahistnet = output_qahistnet.view(batch_size, nqa_per_dial * self.hidden_hist_dim)
        output_qahistnet = torch.unsqueeze(output_qahistnet, 1).expand(batch_size, n_options, nqa_per_dial * self.hidden_hist_dim)
        input_qa = torch.cat((ans_seq, ques_atten, ans_atten, img_atten, output_qahistnet, cap_atten), 2)
        input_qa = input_qa.view(batch_size * n_options, self.concat_dim)
        out_scores = self.simnet(input_qa)
        out_scores = out_scores.squeeze(dim=1)
        out_scores = out_scores.view(batch_size, n_options)
        return out_scores


class TempEmbedder(nn.Module):

    def __init__(self, input_size=128, hidden_size=768, output_size=1024):
        super(TempEmbedder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gelu = nn.GELU()
        self.fga = Atten(util_e=[output_size], pairwise_flag=False)

    def forward(self, audio_embs):
        audio_embs = self.fc1(audio_embs)
        audio_embs = self.gelu(audio_embs)
        audio_embs = self.fc2(audio_embs)
        attend = self.fga([audio_embs])[0]
        return attend


class PositionalEncoding(nn.Module):

    def __init__(self, hidden_size, max_seq_len=16):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
        pe = torch.zeros(max_seq_len, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CLIPVisionEmbeddings(nn.Module):

    def __init__(self, config: 'CLIPVisionConfig'):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        self.patch_embedding = nn.Conv2d(in_channels=config.num_channels, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False)
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer('position_ids', torch.arange(self.num_positions).expand((1, -1)))

    def forward(self, pixel_values: 'torch.FloatTensor') ->torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class CLIPTextEmbeddings(nn.Module):

    def __init__(self, config: 'CLIPTextConfig'):
        super().__init__()
        embed_dim = config.hidden_size
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids: 'Optional[torch.LongTensor]'=None, position_ids: 'Optional[torch.LongTensor]'=None, inputs_embeds: 'Optional[torch.FloatTensor]'=None, audio_token: 'Optional[torch.FloatTensor]'=None, temp_token: 'Optional[torch.FloatTensor]'=None, local_windows: 'Optional[list]'=None) ->torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)
            indices = torch.where(input_ids == 49408)
            if not temp_token is None and indices[0].shape[0]:
                inputs_embeds[indices] = temp_token.view(temp_token.shape[0] * temp_token.shape[1], 1024)
            indices = torch.where(input_ids == 49409)
            if not local_windows[0] is None and indices[0].shape[0]:
                inputs_embeds[indices] = local_windows[0].view(local_windows[0].shape[0] * local_windows[0].shape[1], 1024)
            indices = torch.where(input_ids == 49410)
            if not local_windows[1] is None and indices[0].shape[0]:
                inputs_embeds[indices] = local_windows[1].view(local_windows[1].shape[0] * local_windows[0].shape[1], 1024)
            indices = torch.where(input_ids == 49411)
            if not local_windows[2] is None and indices[0].shape[0]:
                inputs_embeds[indices] = local_windows[2].view(local_windows[2].shape[0] * local_windows[0].shape[1], 1024)
            indices = torch.where(input_ids == 49412)
            if not local_windows[3] is None and indices[0].shape[0]:
                inputs_embeds[indices] = local_windows[3].view(local_windows[3].shape[0] * local_windows[0].shape[1], 1024)
            indices = torch.where(input_ids == 49413)
            if not audio_token is None and indices[0].shape[0]:
                inputs_embeds[indices] = audio_token.view(audio_token.shape[0] * audio_token.shape[1], 1024)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings
        return embeddings


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads}).')
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: 'torch.Tensor', seq_len: 'int', bsz: 'int'):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]'=None, causal_attention_mask: 'Optional[torch.Tensor]'=None, output_attentions: 'Optional[bool]'=False) ->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, embed_dim = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        proj_shape = bsz * self.num_heads, -1, self.head_dim
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(f'Attention weights should be of size {bsz * self.num_heads, tgt_len, src_len}, but is {attn_weights.size()}')
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f'Attention mask should be of size {bsz, 1, tgt_len, src_len}, but is {causal_attention_mask.size()}')
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f'Attention mask should be of size {bsz, 1, tgt_len, src_len}, but is {attention_mask.size()}')
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {bsz, self.num_heads, tgt_len, self.head_dim}, but is {attn_output.size()}')
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped


class CLIPMLP(nn.Module):

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


class CLIPEncoderLayer(nn.Module):

    def __init__(self, config: 'CLIPConfig'):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'torch.Tensor', causal_attention_mask: 'torch.Tensor', output_attentions: 'Optional[bool]'=False) ->Tuple[torch.FloatTensor]:
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
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, causal_attention_mask=causal_attention_mask, output_attentions=output_attentions)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = hidden_states,
        if output_attentions:
            outputs += attn_weights,
        return outputs


CLIP_TEXT_INPUTS_DOCSTRING = """
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


def _expand_mask(mask: 'torch.Tensor', dtype: 'torch.dtype', tgt_len: 'Optional[int]'=None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask, torch.finfo(dtype).min)


def _make_causal_mask(input_ids_shape: 'torch.Size', dtype: 'torch.dtype', device: 'torch.device', past_key_values_length: 'int'=0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


CLIP_VISION_INPUTS_DOCSTRING = """
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2, kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch = x.shape[0]
        fft_dim = -2, -1
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm='ortho')
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1) + ffted.size()[3:])
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))
        ffted = ffted.view((batch, -1, 2) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm='ortho')
        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(SpectralTransform, self).__init__()
        self.stride = stride
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False), nn.BatchNorm2d(out_channels // 2), nn.ReLU(inplace=True))
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)
        output = self.conv2(x + output)
        return output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_type='reflect', gated=False):
        super(FFC, self).__init__()
        assert stride == 1 or stride == 2, 'Stride should be 1 or 2.'
        self.stride = stride
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg
        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2)
        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0
        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)
            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1
        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)
        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin=0, ratio_gout=0, stride=1, padding=0, dilation=1, groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride, padding, dilation, groups, bias)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)
        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResnetBlock(nn.Module):

    def __init__(self, dim, ratio_gin, ratio_gout):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=1, dilation=1, ratio_gin=ratio_gin, ratio_gout=ratio_gout)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=1, dilation=1, ratio_gin=ratio_gin, ratio_gout=ratio_gout)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        return out


class ConcatTupleLayer(nn.Module):

    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)


class LargeMaskInpainting(nn.Module):

    def __init__(self, input_nc=4, output_nc=3, ngf=64, n_downsampling=3, n_blocks=18, max_features=1024):
        super().__init__()
        model = [nn.ReflectionPad2d(3), FFC_BN_ACT(input_nc, ngf, kernel_size=7)]
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [FFC_BN_ACT(min(max_features, ngf * mult), min(max_features, ngf * mult * 2), kernel_size=3, stride=2, padding=1, ratio_gout=0.75 if i == n_downsampling - 1 else 0)]
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(min(max_features, ngf * 2 ** n_downsampling), ratio_gin=0.75, ratio_gout=0.75)
            model += [cur_resblock]
        model += [ConcatTupleLayer()]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult), min(max_features, int(ngf * mult / 2)), kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(min(max_features, int(ngf * mult / 2))), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7), nn.Sigmoid()]
        self.model = nn.Sequential(*model)

    def forward(self, img, mask):
        masked_img = img * (1 - mask)
        masked_img = torch.cat([masked_img, mask], dim=1)
        pred = self.model(masked_img)
        inpainted = mask * pred + (1 - mask) * img
        return inpainted


class LoraInjectedLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=False, r=4, dropout_p=0.1, scale=1.0):
        super().__init__()
        if r > min(in_features, out_features):
            None
            r = min(in_features, out_features)
        self.r = r
        self.linear = nn.Linear(in_features, out_features, bias)
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.lora_up = nn.Linear(r, out_features, bias=False)
        self.scale = scale
        self.selector = nn.Identity()
        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        return self.linear(input) + self.dropout(self.lora_up(self.selector(self.lora_down(input)))) * self.scale

    def realize_as_lora(self):
        return self.lora_up.weight.data * self.scale, self.lora_down.weight.data

    def set_selector_from_diag(self, diag: 'torch.Tensor'):
        assert diag.shape == (self.r,)
        self.selector = nn.Linear(self.r, self.r, bias=False)
        self.selector.weight.data = torch.diag(diag)
        self.selector.weight.data = self.selector.weight.data.to(self.lora_up.weight.device)


class LoraInjectedConv2d(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size, stride=1, padding=0, dilation=1, groups: 'int'=1, bias: 'bool'=True, r: 'int'=4, dropout_p: 'float'=0.1, scale: 'float'=1.0):
        super().__init__()
        if r > min(in_channels, out_channels):
            None
            r = min(in_channels, out_channels)
        self.r = r
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.lora_down = nn.Conv2d(in_channels=in_channels, out_channels=r, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.lora_up = nn.Conv2d(in_channels=r, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.selector = nn.Identity()
        self.scale = scale
        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        return self.conv(input) + self.dropout(self.lora_up(self.selector(self.lora_down(input)))) * self.scale

    def realize_as_lora(self):
        return self.lora_up.weight.data * self.scale, self.lora_down.weight.data

    def set_selector_from_diag(self, diag: 'torch.Tensor'):
        assert diag.shape == (self.r,)
        self.selector = nn.Conv2d(in_channels=self.r, out_channels=self.r, kernel_size=1, stride=1, padding=0, bias=False)
        self.selector.weight.data = torch.diag(diag)
        self.selector.weight.data = self.selector.weight.data.to(self.lora_up.weight.device)


class LoraInjectedConv3d(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: '(3, 1, 1)', padding: '(1, 0, 0)', bias: 'bool'=False, r: 'int'=4, dropout_p: 'float'=0, scale: 'float'=1.0):
        super().__init__()
        if r > min(in_channels, out_channels):
            None
            r = min(in_channels, out_channels)
        self.r = r
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.lora_down = nn.Conv3d(in_channels=in_channels, out_channels=r, kernel_size=kernel_size, bias=False, padding=padding)
        self.dropout = nn.Dropout(dropout_p)
        self.lora_up = nn.Conv3d(in_channels=r, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.selector = nn.Identity()
        self.scale = scale
        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        return self.conv(input) + self.dropout(self.lora_up(self.selector(self.lora_down(input)))) * self.scale

    def realize_as_lora(self):
        return self.lora_up.weight.data * self.scale, self.lora_down.weight.data

    def set_selector_from_diag(self, diag: 'torch.Tensor'):
        assert diag.shape == (self.r,)
        self.selector = nn.Conv3d(in_channels=self.r, out_channels=self.r, kernel_size=1, stride=1, padding=0, bias=False)
        self.selector.weight.data = torch.diag(diag)
        self.selector.weight.data = self.selector.weight.data.to(self.lora_up.weight.device)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (CLIPAttention,
     lambda: ([], {'config': SimpleNamespace(hidden_size=4, num_attention_heads=4, attention_dropout=0.5)}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (CLIPVisionEmbeddings,
     lambda: ([], {'config': SimpleNamespace(hidden_size=4, image_size=4, patch_size=4, num_channels=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FFC_BN_ACT,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FourierUnit,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GLU_Linear,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 0, 4])], {})),
    (LoraInjectedConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LoraInjectedConv3d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LoraInjectedLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MultiheadAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (NaiveAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (NormEMAVectorQuantizer,
     lambda: ([], {'n_embed': 4, 'embedding_dim': 4, 'beta': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Pairwise,
     lambda: ([], {'embed_x_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (PositionalEncoding,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SamePad,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SpectralTransform,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Unary,
     lambda: ([], {'embed_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
]

