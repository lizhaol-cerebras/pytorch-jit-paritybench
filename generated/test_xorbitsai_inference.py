
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


import typing


from typing import TYPE_CHECKING


from typing import Any


from typing import Dict


from typing import Iterator


from typing import List


from typing import Optional


from typing import Union


import functools


import inspect


import queue


import time


import types


import uuid


from typing import AsyncGenerator


from typing import AsyncIterator


from typing import Callable


from typing import Generator


import logging


import torch


import numpy as np


from collections import defaultdict


from typing import Literal


from typing import Tuple


from typing import no_type_check


import re


from collections import deque


import itertools


import warnings


import abc


from abc import abstractmethod


from functools import lru_cache


from typing import TypedDict


import torch.nn as nn


from torch import Tensor


from torch.nn import functional as F


from typing import Iterable


from functools import partial


from collections.abc import Sequence


import random


from torch.utils.data import DataLoader


from copy import deepcopy


import torch.distributed as dist


from torch.distributed.elastic.multiprocessing.errors import record


import math


from torch.utils.data import IterableDataset


from torch.nn.utils.rnn import pad_sequence


import torch.nn.functional as F


from torch.nn.utils import weight_norm


import typing as tp


from scipy.signal import get_window


from torch.nn import Conv1d


from torch.nn import ConvTranspose1d


from torch.nn.utils import remove_weight_norm


from torch.distributions.uniform import Uniform


from torch import nn


from torch.nn.utils.rnn import unpad_sequence


from torch import sin


from torch import pow


from torch.nn import Parameter


import torch.utils.checkpoint as ckpt


from torch.optim.lr_scheduler import _LRScheduler


import torch.optim as optim


from torch.utils.tensorboard import SummaryWriter


from torch.nn.utils import clip_grad_norm_


import torchvision.transforms


import torchvision


import torchvision.transforms.functional


import copy


from typing import Type


from typing import Final


from typing import Sequence


from typing import Set


from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype


from torch.utils._foreach_utils import _has_foreach_support


from torch.utils.data import Dataset


from itertools import chain


from random import Random


from torch.distributed import get_rank


from torch.distributed import get_world_size


from torch.distributed import is_initialized


from torch.utils.data import get_worker_info


from math import prod


from torch.nn.utils.parametrizations import weight_norm


from torch.nn.utils.parametrize import remove_parametrizations


from torch.utils.checkpoint import checkpoint


import matplotlib


from matplotlib import pyplot as plt


from typing import Annotated


import torch._dynamo.config


import torch._inductor.config


import string


from abc import ABC


from torch.nn import CrossEntropyLoss


import matplotlib.pyplot as plt


from torch.utils.data.dataloader import DataLoader


import torch.utils.data


from scipy.io.wavfile import read


from torch.nn import AvgPool1d


from torch.nn import Conv2d


from torch.nn.utils import spectral_norm


import matplotlib.pylab as plt


from time import perf_counter


from math import ceil


from torch.nn.init import trunc_normal_


from torchvision import transforms


import logging.handlers


from torch.distributions import Categorical


def compress(tensor, config):
    """Simulate group-wise quantization."""
    if not config.enabled:
        return tensor
    group_size, num_bits, group_dim, symmetric = config.group_size, config.num_bits, config.group_dim, config.symmetric
    assert num_bits <= 8
    original_shape = tensor.shape
    num_groups = (original_shape[group_dim] + group_size - 1) // group_size
    new_shape = original_shape[:group_dim] + (num_groups, group_size) + original_shape[group_dim + 1:]
    pad_len = (group_size - original_shape[group_dim] % group_size) % group_size
    if pad_len != 0:
        pad_shape = original_shape[:group_dim] + (pad_len,) + original_shape[group_dim + 1:]
        tensor = torch.cat([tensor, torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)], dim=group_dim)
    data = tensor.view(new_shape)
    if symmetric:
        B = 2 ** (num_bits - 1) - 1
        scale = B / torch.max(data.abs(), dim=group_dim + 1, keepdim=True)[0]
        data = data * scale
        data = data.clamp_(-B, B).round_()
        return data, scale, original_shape
    else:
        B = 2 ** num_bits - 1
        mn = torch.min(data, dim=group_dim + 1, keepdim=True)[0]
        mx = torch.max(data, dim=group_dim + 1, keepdim=True)[0]
        scale = B / (mx - mn)
        data = data - mn
        data.mul_(scale)
        data = data.clamp_(0, B).round_()
        return data, mn, scale, original_shape


def decompress(packed_data, config):
    """Simulate group-wise dequantization."""
    if not config.enabled:
        return packed_data
    group_size, _, group_dim, symmetric = config.group_size, config.num_bits, config.group_dim, config.symmetric
    if symmetric:
        data, scale, original_shape = packed_data
        data = data / scale
    else:
        data, mn, scale, original_shape = packed_data
        data = data / scale
        data.add_(mn)
    pad_len = (group_size - original_shape[group_dim] % group_size) % group_size
    if pad_len:
        padded_original_shape = original_shape[:group_dim] + (original_shape[group_dim] + pad_len,) + original_shape[group_dim + 1:]
        data = data.reshape(padded_original_shape)
        indices = [slice(0, x) for x in original_shape]
        return data[indices].contiguous()
    else:
        return data.view(original_shape)


class CLinear(nn.Module):
    """Compressed Linear Layer."""

    def __init__(self, weight=None, bias=None, device=None):
        super().__init__()
        if weight is None:
            self.weight = None
        elif isinstance(weight, Tensor):
            self.weight = compress(weight.data, default_compression_config)
        else:
            self.weight = weight
        self.bias = bias

    def forward(self, input: 'Tensor') ->Tensor:
        weight = decompress(self.weight, default_compression_config)
        if self.bias is None:
            return F.linear(input, weight)
        return F.linear(input, weight, self.bias)


class Attention(nn.Module):
    fused_attn: 'Final[bool]'

    def __init__(self, dim: 'int', num_heads: 'int'=8, qkv_bias: 'bool'=False, qk_norm: 'bool'=False, attn_drop: 'float'=0.0, proj_drop: 'float'=0.0, norm_layer: 'nn.Module'=nn.LayerNorm) ->None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0.0 else nn.Identity()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, in_features, out_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        """
        super().__init__()
        self.in_features = out_features if isinstance(out_features, list) else [out_features]
        self.proj = LoRACompatibleLinear(in_features, out_features)
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(self.in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(self.in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-09

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        x = self.proj(x)
        if self.alpha_logscale:
            alpha = torch.exp(self.alpha)
            beta = torch.exp(self.beta)
        else:
            alpha = self.alpha
            beta = self.beta
        x = x + 1.0 / (beta + self.no_div_by_zero) * torch.pow(torch.sin(x * alpha), 2)
        return x


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
        elif activation_fn == 'snakebeta':
            act_fn = SnakeBeta(dim, inner_dim)
        self.net = nn.ModuleList([])
        self.net.append(act_fn)
        self.net.append(nn.Dropout(dropout))
        self.net.append(LoRACompatibleLinear(inner_dim, dim_out))
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class Block1D(torch.nn.Module):

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv1d(dim, dim_out, 3, padding=1), torch.nn.GroupNorm(groups, dim_out), nn.Mish())

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class Downsample1D(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv = torch.nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class ResnetBlock1D(torch.nn.Module):

    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = torch.nn.Sequential(nn.Mish(), torch.nn.Linear(time_emb_dim, dim_out))
        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)
        self.res_conv = torch.nn.Conv1d(dim, dim_out, 1)

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class SinusoidalPosEmb(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, 'SinusoidalPosEmb requires dim to be even'

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):

    def __init__(self, in_channels: 'int', time_embed_dim: 'int', act_fn: 'str'='silu', out_dim: 'int'=None, post_act_fn: 'Optional[str]'=None, cond_proj_dim=None):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None
        self.act = get_activation(act_fn)
        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)
        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)
        if self.act is not None:
            sample = self.act(sample)
        sample = self.linear_2(sample)
        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


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

    def __init__(self, channels, use_conv=False, use_conv_transpose=True, out_channels=None, name='conv'):
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


class ConditionalDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, channels=(256, 256), dropout=0.05, attention_head_dim=64, n_blocks=1, num_mid_blocks=2, num_heads=4, act_fn='snake'):
        """
        This decoder requires an input with the same shape of the target. So, if your text content
        is shorter or longer than the outputs, please re-sampling it before feeding to the decoder.
        """
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(in_channels=in_channels, time_embed_dim=time_embed_dim, act_fn='silu')
        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        output_channel = in_channels
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList([BasicTransformerBlock(dim=output_channel, num_attention_heads=num_heads, attention_head_dim=attention_head_dim, dropout=dropout, activation_fn=act_fn) for _ in range(n_blocks)])
            downsample = Downsample1D(output_channel) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))
        for i in range(num_mid_blocks):
            input_channel = channels[-1]
            out_channels = channels[-1]
            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList([BasicTransformerBlock(dim=output_channel, num_attention_heads=num_heads, attention_head_dim=attention_head_dim, dropout=dropout, activation_fn=act_fn) for _ in range(n_blocks)])
            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))
        channels = channels[::-1] + (channels[0],)
        for i in range(len(channels) - 1):
            input_channel = channels[i] * 2
            output_channel = channels[i + 1]
            is_last = i == len(channels) - 2
            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList([BasicTransformerBlock(dim=output_channel, num_attention_heads=num_heads, attention_head_dim=attention_head_dim, dropout=dropout, activation_fn=act_fn) for _ in range(n_blocks)])
            upsample = Upsample1D(output_channel, use_conv_transpose=True) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))
        self.final_block = Block1D(channels[-1], channels[-1])
        self.final_proj = nn.Conv1d(channels[-1], self.out_channels, 1)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, mask, mu, t, spks=None, cond=None):
        """Forward pass of the UNet1DConditional model.

        Args:
            x (torch.Tensor): shape (batch_size, in_channels, time)
            mask (_type_): shape (batch_size, 1, time)
            t (_type_): shape (batch_size)
            spks (_type_, optional): shape: (batch_size, condition_channels). Defaults to None.
            cond (_type_, optional): placeholder for future use. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        t = self.time_embeddings(t)
        t = self.time_mlp(t)
        x = pack([x, mu], 'b * t')[0]
        if spks is not None:
            spks = repeat(spks, 'b c -> b c t', t=x.shape[-1])
            x = pack([x, spks], 'b * t')[0]
        if cond is not None:
            x = pack([x, cond], 'b * t')[0]
        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            x = rearrange(x, 'b c t -> b t c').contiguous()
            attn_mask = torch.matmul(mask_down.transpose(1, 2).contiguous(), mask_down)
            for transformer_block in transformer_blocks:
                x = transformer_block(hidden_states=x, attention_mask=attn_mask, timestep=t)
            x = rearrange(x, 'b t c -> b c t').contiguous()
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]
        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = rearrange(x, 'b c t -> b t c').contiguous()
            attn_mask = torch.matmul(mask_mid.transpose(1, 2).contiguous(), mask_mid)
            for transformer_block in transformer_blocks:
                x = transformer_block(hidden_states=x, attention_mask=attn_mask, timestep=t)
            x = rearrange(x, 'b t c -> b c t').contiguous()
        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = pack([x[:, :, :skip.shape[-1]], skip], 'b * t')[0]
            x = resnet(x, mask_up, t)
            x = rearrange(x, 'b c t -> b t c').contiguous()
            attn_mask = torch.matmul(mask_up.transpose(1, 2).contiguous(), mask_up)
            for transformer_block in transformer_blocks:
                x = transformer_block(hidden_states=x, attention_mask=attn_mask, timestep=t)
            x = rearrange(x, 'b t c -> b c t').contiguous()
            x = upsample(x * mask_up)
        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        return output * mask


def make_pad_mask(lengths: 'torch.Tensor', max_len: 'int'=0) ->torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


class InterpolateRegulator(nn.Module):

    def __init__(self, channels: 'int', sampling_ratios: 'Tuple', out_channels: 'int'=None, groups: 'int'=1):
        super().__init__()
        self.sampling_ratios = sampling_ratios
        out_channels = out_channels or channels
        model = nn.ModuleList([])
        if len(sampling_ratios) > 0:
            for _ in sampling_ratios:
                module = nn.Conv1d(channels, channels, 3, 1, 1)
                norm = nn.GroupNorm(groups, channels)
                act = nn.Mish()
                model.extend([module, norm, act])
        model.append(nn.Conv1d(channels, out_channels, 1, 1))
        self.model = nn.Sequential(*model)

    def forward(self, x, ylens=None):
        mask = (~make_pad_mask(ylens)).unsqueeze(-1)
        x = F.interpolate(x.transpose(1, 2).contiguous(), size=ylens.max(), mode='linear')
        out = self.model(x).transpose(1, 2).contiguous()
        olens = ylens
        return out * mask, olens

    def inference(self, x1, x2, mel_len1, mel_len2):
        if x2.shape[1] > 40:
            x2_head = F.interpolate(x2[:, :20].transpose(1, 2).contiguous(), size=34, mode='linear')
            x2_mid = F.interpolate(x2[:, 20:-20].transpose(1, 2).contiguous(), size=mel_len2 - 34 * 2, mode='linear')
            x2_tail = F.interpolate(x2[:, -20:].transpose(1, 2).contiguous(), size=34, mode='linear')
            x2 = torch.concat([x2_head, x2_mid, x2_tail], dim=2)
        else:
            x2 = F.interpolate(x2.transpose(1, 2).contiguous(), size=mel_len2, mode='linear')
        if x1.shape[1] != 0:
            x1 = F.interpolate(x1.transpose(1, 2).contiguous(), size=mel_len1, mode='linear')
            x = torch.concat([x1, x2], dim=2)
        else:
            x = x2
        out = self.model(x).transpose(1, 2).contiguous()
        return out, mel_len1 + mel_len2


class ConvRNNF0Predictor(nn.Module):

    def __init__(self, num_class: 'int'=1, in_channels: 'int'=80, cond_channels: 'int'=512):
        super().__init__()
        self.num_class = num_class
        self.condnet = nn.Sequential(weight_norm(nn.Conv1d(in_channels, cond_channels, kernel_size=3, padding=1)), nn.ELU(), weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)), nn.ELU(), weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)), nn.ELU(), weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)), nn.ELU(), weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)), nn.ELU())
        self.classifier = nn.Linear(in_features=cond_channels, out_features=self.num_class)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.condnet(x)
        x = x.transpose(1, 2)
        return torch.abs(self.classifier(x).squeeze(-1))


class Snake(nn.Module):
    """
    Implementation of a sine-based periodic activation function
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha: trainable parameter
            alpha is initialized to 1 by default, higher values = higher-frequency.
            alpha will be trained along with the rest of your model.
        """
        super(Snake, self).__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-09

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + 1.0 / (alpha + self.no_div_by_zero) * pow(sin(x * alpha), 2)
        return x


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean, std)


class ResBlock(torch.nn.Module):
    """Residual block module in HiFiGAN/BigVGAN."""

    def __init__(self, channels: 'int'=512, kernel_size: 'int'=3, dilations: 'tp.List[int]'=[1, 3, 5]):
        super(ResBlock, self).__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for dilation in dilations:
            self.convs1.append(weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation, padding=get_padding(kernel_size, dilation))))
            self.convs2.append(weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))))
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)
        self.activations1 = nn.ModuleList([Snake(channels, alpha_logscale=False) for _ in range(len(self.convs1))])
        self.activations2 = nn.ModuleList([Snake(channels, alpha_logscale=False) for _ in range(len(self.convs2))])

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        for idx in range(len(self.convs1)):
            xt = self.activations1[idx](x)
            xt = self.convs1[idx](xt)
            xt = self.activations2[idx](xt)
            xt = self.convs2[idx](xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for idx in range(len(self.convs1)):
            remove_weight_norm(self.convs1[idx])
            remove_weight_norm(self.convs2[idx])


class SineGen(torch.nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    @torch.no_grad()
    def forward(self, f0):
        """
        :param f0: [B, 1, sample_len], Hz
        :return: [B, 1, sample_len]
        """
        F_mat = torch.zeros((f0.size(0), self.harmonic_num + 1, f0.size(-1)))
        for i in range(self.harmonic_num + 1):
            F_mat[:, i:i + 1, :] = f0 * (i + 1) / self.sampling_rate
        theta_mat = 2 * np.pi * (torch.cumsum(F_mat, dim=-1) % 1)
        u_dist = Uniform(low=-np.pi, high=np.pi)
        phase_vec = u_dist.sample(sample_shape=(f0.size(0), self.harmonic_num + 1, 1))
        phase_vec[:, 0, :] = 0
        sine_waves = self.sine_amp * torch.sin(theta_mat + phase_vec)
        uv = self._f02uv(f0)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    """ SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(self, sampling_rate, upsample_scale, harmonic_num=0, sine_amp=0.1, add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x.transpose(1, 2))
            sine_wavs = sine_wavs.transpose(1, 2)
            uv = uv.transpose(1, 2)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


class HiFTGenerator(nn.Module):
    """
    HiFTNet Generator: Neural Source Filter + ISTFTNet
    https://arxiv.org/abs/2309.09493
    """

    def __init__(self, in_channels: 'int'=80, base_channels: 'int'=512, nb_harmonics: 'int'=8, sampling_rate: 'int'=22050, nsf_alpha: 'float'=0.1, nsf_sigma: 'float'=0.003, nsf_voiced_threshold: 'float'=10, upsample_rates: 'tp.List[int]'=[8, 8], upsample_kernel_sizes: 'tp.List[int]'=[16, 16], istft_params: 'tp.Dict[str, int]'={'n_fft': 16, 'hop_len': 4}, resblock_kernel_sizes: 'tp.List[int]'=[3, 7, 11], resblock_dilation_sizes: 'tp.List[tp.List[int]]'=[[1, 3, 5], [1, 3, 5], [1, 3, 5]], source_resblock_kernel_sizes: 'tp.List[int]'=[7, 11], source_resblock_dilation_sizes: 'tp.List[tp.List[int]]'=[[1, 3, 5], [1, 3, 5]], lrelu_slope: 'float'=0.1, audio_limit: 'float'=0.99, f0_predictor: 'torch.nn.Module'=None):
        super(HiFTGenerator, self).__init__()
        self.out_channels = 1
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.istft_params = istft_params
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.m_source = SourceModuleHnNSF(sampling_rate=sampling_rate, upsample_scale=np.prod(upsample_rates) * istft_params['hop_len'], harmonic_num=nb_harmonics, sine_amp=nsf_alpha, add_noise_std=nsf_sigma, voiced_threshod=nsf_voiced_threshold)
        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(upsample_rates) * istft_params['hop_len'])
        self.conv_pre = weight_norm(Conv1d(in_channels, base_channels, 7, 1, padding=3))
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(ConvTranspose1d(base_channels // 2 ** i, base_channels // 2 ** (i + 1), k, u, padding=(k - u) // 2)))
        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        downsample_rates = [1] + upsample_rates[::-1][:-1]
        downsample_cum_rates = np.cumprod(downsample_rates)
        for i, (u, k, d) in enumerate(zip(downsample_cum_rates[::-1], source_resblock_kernel_sizes, source_resblock_dilation_sizes)):
            if u == 1:
                self.source_downs.append(Conv1d(istft_params['n_fft'] + 2, base_channels // 2 ** (i + 1), 1, 1))
            else:
                self.source_downs.append(Conv1d(istft_params['n_fft'] + 2, base_channels // 2 ** (i + 1), u * 2, u, padding=u // 2))
            self.source_resblocks.append(ResBlock(base_channels // 2 ** (i + 1), k, d))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = base_channels // 2 ** (i + 1)
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))
        self.conv_post = weight_norm(Conv1d(ch, istft_params['n_fft'] + 2, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft_window = torch.from_numpy(get_window('hann', istft_params['n_fft'], fftbins=True).astype(np.float32))
        self.f0_predictor = f0_predictor

    def _f02source(self, f0: 'torch.Tensor') ->torch.Tensor:
        f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        har_source, _, _ = self.m_source(f0)
        return har_source.transpose(1, 2)

    def _stft(self, x):
        spec = torch.stft(x, self.istft_params['n_fft'], self.istft_params['hop_len'], self.istft_params['n_fft'], window=self.stft_window, return_complex=True)
        spec = torch.view_as_real(spec)
        return spec[..., 0], spec[..., 1]

    def _istft(self, magnitude, phase):
        magnitude = torch.clip(magnitude, max=100.0)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        inverse_transform = torch.istft(torch.complex(real, img), self.istft_params['n_fft'], self.istft_params['hop_len'], self.istft_params['n_fft'], window=self.stft_window)
        return inverse_transform

    def forward(self, x: 'torch.Tensor', cache_source: 'torch.Tensor'=torch.zeros(1, 1, 0)) ->torch.Tensor:
        f0 = self.f0_predictor(x)
        s = self._f02source(f0)
        if cache_source.shape[2] == 0:
            s[:, :, :cache_source.shape[2]] = cache_source
        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        magnitude = torch.exp(x[:, :self.istft_params['n_fft'] // 2 + 1, :])
        phase = torch.sin(x[:, self.istft_params['n_fft'] // 2 + 1:, :])
        x = self._istft(magnitude, phase)
        x = torch.clamp(x, -self.audio_limit, self.audio_limit)
        return x, s

    def remove_weight_norm(self):
        None
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        self.source_module.remove_weight_norm()
        for l in self.source_downs:
            remove_weight_norm(l)
        for l in self.source_resblocks:
            l.remove_weight_norm()

    @torch.inference_mode()
    def inference(self, mel: 'torch.Tensor', cache_source: 'torch.Tensor'=torch.zeros(1, 1, 0)) ->torch.Tensor:
        return self.forward(x=mel, cache_source=cache_source)


IGNORE_ID = -1


class LabelSmoothingLoss(nn.Module):
    """Label-smoothing loss.

    In a standard CE loss, the label's data distribution is:
    [0,1,2] ->
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    In the smoothing version CE Loss,some probabilities
    are taken from the true label prob (1.0) and are divided
    among other labels.

    e.g.
    smoothing=0.1
    [0,1,2] ->
    [
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ]

    Args:
        size (int): the number of class
        padding_idx (int): padding class id which will be ignored for loss
        smoothing (float): smoothing rate (0.0 means the conventional CE)
        normalize_length (bool):
            normalize loss by sequence length if True
            normalize loss by batch size if False
    """

    def __init__(self, size: 'int', padding_idx: 'int', smoothing: 'float', normalize_length: 'bool'=False):
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.normalize_length = normalize_length

    def forward(self, x: 'torch.Tensor', target: 'torch.Tensor') ->torch.Tensor:
        """Compute loss between x and target.

        The model outputs and data labels tensors are flatten to
        (batch*seqlen, class) shape and a mask is applied to the
        padding part which should not be calculated for loss.

        Args:
            x (torch.Tensor): prediction (batch, seqlen, class)
            target (torch.Tensor):
                target signal masked with self.padding_id (batch, seqlen)
        Returns:
            loss (torch.Tensor) : The KL loss, scalar float value
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (self.size - 1))
        ignore = target == self.padding_idx
        total = len(target) - ignore.sum().item()
        target = target.masked_fill(ignore, 0)
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom


def th_accuracy(pad_outputs: 'torch.Tensor', pad_targets: 'torch.Tensor', ignore_label: 'int') ->torch.Tensor:
    """Calculate accuracy.

    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        torch.Tensor: Accuracy value (0.0 - 1.0).

    """
    pad_pred = pad_outputs.view(pad_targets.size(0), pad_targets.size(1), pad_outputs.size(1)).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return (numerator / denominator).detach()


class TransformerLM(torch.nn.Module):

    def __init__(self, text_encoder_input_size: 'int', llm_input_size: 'int', llm_output_size: 'int', text_token_size: 'int', speech_token_size: 'int', text_encoder: 'torch.nn.Module', llm: 'torch.nn.Module', sampling: 'Callable', length_normalized_loss: 'bool'=True, lsm_weight: 'float'=0.0, spk_embed_dim: 'int'=192):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.speech_token_size = speech_token_size
        self.text_embedding = torch.nn.Embedding(text_token_size, text_encoder_input_size)
        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = nn.Linear(self.text_encoder.output_size(), llm_input_size)
        self.sos_eos = 0
        self.task_id = 1
        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 1)
        self.criterion_ce = LabelSmoothingLoss(size=speech_token_size + 1, padding_idx=IGNORE_ID, smoothing=lsm_weight, normalize_length=length_normalized_loss)
        self.speech_embedding = torch.nn.Embedding(speech_token_size, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)
        self.sampling = sampling

    def encode(self, text: 'torch.Tensor', text_lengths: 'torch.Tensor'):
        encoder_out, encoder_mask = self.text_encoder(text, text_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(self, sos_eos_emb, embedding, text_token, text_token_len, task_id_emb, speech_token, speech_token_len):
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        lm_input = [torch.concat([sos_eos_emb.squeeze(dim=0), embedding[i], text_token[i], task_id_emb.squeeze(dim=0), speech_token[i]], dim=0) for i in range(len(text_token))]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        return lm_input, lm_input_len

    def forward(self, batch: 'dict', device: 'torch.device') ->Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch['text_token']
        text_token_len = batch['text_token_len']
        speech_token = batch['speech_token']
        speech_token_len = batch['speech_token_len']
        embedding = batch['embedding']
        lm_target = [torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + speech_token[i, :speech_token_len[i]].tolist() + [self.speech_token_size]) for i in range(text_token.size(0))]
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID)
        text_token = self.text_embedding(text_token)
        text_token, text_token_len = self.encode(text_token, text_token_len)
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1)
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        speech_token = self.speech_embedding(speech_token)
        lm_input, lm_input_len = self.pad_unpad_sequence(sos_eos_emb, embedding, text_token, text_token_len, task_id_emb, speech_token, speech_token_len)
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len)
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 1), lm_target, ignore_label=IGNORE_ID)
        return {'loss': loss, 'acc': acc}

    def sampling_ids(self, weighted_scores: 'torch.Tensor', decoded_tokens: 'List', sampling: 'int', ignore_eos: 'bool'=True):
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if not ignore_eos or self.speech_token_size not in top_ids:
                break
        return top_ids

    @torch.inference_mode()
    def inference(self, text: 'torch.Tensor', text_len: 'torch.Tensor', prompt_text: 'torch.Tensor', prompt_text_len: 'torch.Tensor', prompt_speech_token: 'torch.Tensor', prompt_speech_token_len: 'torch.Tensor', embedding: 'torch.Tensor', sampling: 'int'=25, max_token_text_ratio: 'float'=20, min_token_text_ratio: 'float'=2) ->Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.text_embedding(text)
        text, text_len = self.encode(text, text_len)
        if embedding.shape[0] != 0:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = embedding.unsqueeze(dim=1)
        else:
            embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype)
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype)
        lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)
        out_tokens = []
        offset = 0
        att_cache, cnn_cache = torch.zeros((0, 0, 0, 0), device=lm_input.device), torch.zeros((0, 0, 0, 0), device=lm_input.device)
        for i in range(max_len):
            y_pred, att_cache, cnn_cache = self.llm.forward_chunk(lm_input, offset=0, required_cache_size=-1, att_cache=att_cache, cnn_cache=cnn_cache, att_mask=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)))
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.speech_token_size:
                break
            yield torch.tensor([[top_ids]], dtype=torch.int64, device=device)
            out_tokens.append(top_ids)
            offset += lm_input.size(1)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)


class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Return Swish activation function."""
        return x * torch.sigmoid(x)


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head: 'int', n_feat: 'int', dropout_rate: 'float', key_bias: 'bool'=True):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return q, k, v

    def forward_attention(self, value: 'torch.Tensor', scores: 'torch.Tensor', mask: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool)) ->torch.Tensor:
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask.size(2) > 0:
            mask = mask.unsqueeze(1).eq(0)
            mask = mask[:, :, :, :scores.size(-1)]
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x)

    def forward(self, query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor', mask: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool), pos_emb: 'torch.Tensor'=torch.empty(0), cache: 'torch.Tensor'=torch.zeros((0, 0, 0, 0))) ->Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                CosyVoice.
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`


        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`

        """
        q, k, v = self.forward_qkv(query, key, value)
        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        new_cache = torch.cat((k, v), dim=-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head: 'int', n_feat: 'int', dropout_rate: 'float', key_bias: 'bool'=True):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate, key_bias)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x: 'torch.Tensor') ->torch.Tensor:
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, 2*time1-1).
            time1 means the length of query vector.

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(x.size()[0], x.size()[1], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[:, :, :, :x.size(-1) // 2 + 1]
        return x

    def forward(self, query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor', mask: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool), pos_emb: 'torch.Tensor'=torch.empty(0), cache: 'torch.Tensor'=torch.zeros((0, 0, 0, 0))) ->Tuple[torch.Tensor, torch.Tensor]:
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, time2, size).
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)
        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        new_cache = torch.cat((k, v), dim=-1)
        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        if matrix_ac.shape != matrix_bd.shape:
            matrix_bd = self.rel_shift(matrix_bd)
        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model."""

    def __init__(self, channels: 'int', kernel_size: 'int'=15, activation: 'nn.Module'=nn.ReLU(), norm: 'str'='batch_norm', causal: 'bool'=False, bias: 'bool'=True):
        """Construct an ConvolutionModule object.
        Args:
            channels (int): The number of channels of conv layers.
            kernel_size (int): Kernel size of conv layers.
            causal (int): Whether use causal convolution or not
        """
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=bias)
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
        self.depthwise_conv = nn.Conv1d(channels, channels, kernel_size, stride=1, padding=padding, groups=channels, bias=bias)
        assert norm in ['batch_norm', 'layer_norm']
        if norm == 'batch_norm':
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.use_layer_norm = True
            self.norm = nn.LayerNorm(channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.activation = activation

    def forward(self, x: 'torch.Tensor', mask_pad: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool), cache: 'torch.Tensor'=torch.zeros((0, 0, 0))) ->Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (#batch, channels, cache_t),
                (0, 0, 0) meas fake cache.
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        x = x.transpose(1, 2)
        if mask_pad.size(2) > 0:
            x.masked_fill_(~mask_pad, 0.0)
        if self.lorder > 0:
            if cache.size(2) == 0:
                x = nn.functional.pad(x, (self.lorder, 0), 'constant', 0.0)
            else:
                assert cache.size(0) == x.size(0)
                assert cache.size(1) == x.size(1)
                x = torch.cat((cache, x), dim=2)
            assert x.size(2) > self.lorder
            new_cache = x[:, :, -self.lorder:]
        else:
            new_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)
        if mask_pad.size(2) > 0:
            x.masked_fill_(~mask_pad, 0.0)
        return x.transpose(1, 2), new_cache


COSYVOICE_ACTIVATION_CLASSES = {'hardtanh': torch.nn.Hardtanh, 'tanh': torch.nn.Tanh, 'relu': torch.nn.ReLU, 'selu': torch.nn.SELU, 'swish': getattr(torch.nn, 'SiLU', Swish), 'gelu': torch.nn.GELU}


COSYVOICE_ATTENTION_CLASSES = {'selfattn': MultiHeadedAttention, 'rel_selfattn': RelPositionMultiHeadedAttention}


class EspnetRelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module (new implementation).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    See : Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.

    """

    def __init__(self, d_model: 'int', dropout_rate: 'float', max_len: 'int'=5000):
        """Construct an PositionalEncoding object."""
        super(EspnetRelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: 'torch.Tensor'):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe
                return
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model))
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe

    def forward(self, x: 'torch.Tensor', offset: 'Union[int, torch.Tensor]'=0) ->Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.position_encoding(size=x.size(1), offset=offset)
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset: 'Union[int, torch.Tensor]', size: 'int') ->torch.Tensor:
        """ For getting encoding in a streaming fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            offset (int or torch.tensor): start offset
            size (int): required size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
        pos_emb = self.pe[:, self.pe.size(1) // 2 - size + 1:self.pe.size(1) // 2 + size]
        return pos_emb


class PositionalEncoding(torch.nn.Module):
    """Positional encoding.

    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(self, d_model: 'int', dropout_rate: 'float', max_len: 'int'=5000, reverse: 'bool'=False):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len
        self.pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x: 'torch.Tensor', offset: 'Union[int, torch.Tensor]'=0) ->Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
            offset (int, torch.tensor): position offset

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            torch.Tensor: for compatibility to RelPositionalEncoding
        """
        self.pe = self.pe
        pos_emb = self.position_encoding(offset, x.size(1), False)
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset: 'Union[int, torch.Tensor]', size: 'int', apply_dropout: 'bool'=True) ->torch.Tensor:
        """ For getting encoding in a streaming fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            offset (int or torch.tensor): start offset
            size (int): required size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
        if isinstance(offset, int):
            assert offset + size <= self.max_len
            pos_emb = self.pe[:, offset:offset + size]
        elif isinstance(offset, torch.Tensor) and offset.dim() == 0:
            assert offset + size <= self.max_len
            pos_emb = self.pe[:, offset:offset + size]
        else:
            assert torch.max(offset) + size <= self.max_len
            index = offset.unsqueeze(1) + torch.arange(0, size)
            flag = index > 0
            index = index * flag
            pos_emb = F.embedding(index, self.pe[0])
        if apply_dropout:
            pos_emb = self.dropout(pos_emb)
        return pos_emb


class LearnablePositionalEncoding(PositionalEncoding):
    """ Learnable position encoding used in openai-whisper.decoder
    """

    def __init__(self, d_model: 'int', dropout_rate: 'float', max_len: 'int'=448):
        super().__init__(d_model, dropout_rate, max_len)
        self.pe = torch.nn.Parameter(torch.empty(1, max_len, d_model))
        self.xscale = 1.0


class NoPositionalEncoding(torch.nn.Module):
    """ No position encoding
    """

    def __init__(self, d_model: 'int', dropout_rate: 'float'):
        super().__init__()
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x: 'torch.Tensor', offset: 'Union[int, torch.Tensor]'=0) ->Tuple[torch.Tensor, torch.Tensor]:
        """ Just return zero vector for interface compatibility
        """
        pos_emb = torch.zeros(1, x.size(1), self.d_model)
        return self.dropout(x), pos_emb

    def position_encoding(self, offset: 'Union[int, torch.Tensor]', size: 'int') ->torch.Tensor:
        return torch.zeros(1, size, self.d_model)


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(self, d_model: 'int', dropout_rate: 'float', max_len: 'int'=5000):
        """Initialize class."""
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def forward(self, x: 'torch.Tensor', offset: 'Union[int, torch.Tensor]'=0) ->Tuple[torch.Tensor, torch.Tensor]:
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).
        """
        self.pe = self.pe
        x = x * self.xscale
        pos_emb = self.position_encoding(offset, x.size(1), False)
        return self.dropout(x), self.dropout(pos_emb)


class WhisperPositionalEncoding(PositionalEncoding):
    """ Sinusoids position encoding used in openai-whisper.encoder
    """

    def __init__(self, d_model: 'int', dropout_rate: 'float', max_len: 'int'=1500):
        super().__init__(d_model, dropout_rate, max_len)
        self.xscale = 1.0
        log_timescale_increment = np.log(10000) / (d_model // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(d_model // 2))
        scaled_time = torch.arange(max_len)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        delattr(self, 'pe')
        self.register_buffer('pe', pe.unsqueeze(0))


COSYVOICE_EMB_CLASSES = {'embed': PositionalEncoding, 'abs_pos': PositionalEncoding, 'rel_pos': RelPositionalEncoding, 'rel_pos_espnet': EspnetRelPositionalEncoding, 'no_pos': NoPositionalEncoding, 'abs_pos_whisper': WhisperPositionalEncoding, 'embed_learnable_pe': LearnablePositionalEncoding}


class DecoderLayer(nn.Module):
    """Single decoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        src_attn (torch.nn.Module): Inter-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
            If `None` is passed, Inter-attention is not used, such as
            CIF, GPT, and other decoder only model.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
    """

    def __init__(self, size: 'int', self_attn: 'nn.Module', src_attn: 'Optional[nn.Module]', feed_forward: 'nn.Module', dropout_rate: 'float', normalize_before: 'bool'=True):
        """Construct an DecoderLayer object."""
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size, eps=1e-05)
        self.norm2 = nn.LayerNorm(size, eps=1e-05)
        self.norm3 = nn.LayerNorm(size, eps=1e-05)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before

    def forward(self, tgt: 'torch.Tensor', tgt_mask: 'torch.Tensor', memory: 'torch.Tensor', memory_mask: 'torch.Tensor', cache: 'Optional[torch.Tensor]'=None) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (torch.Tensor): Mask for input tensor
                (#batch, maxlen_out).
            memory (torch.Tensor): Encoded memory
                (#batch, maxlen_in, size).
            memory_mask (torch.Tensor): Encoded memory mask
                (#batch, maxlen_in).
            cache (torch.Tensor): cached tensors.
                (#batch, maxlen_out - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, maxlen_out, size).
            torch.Tensor: Mask for output tensor (#batch, maxlen_out).
            torch.Tensor: Encoded memory (#batch, maxlen_in, size).
            torch.Tensor: Encoded memory mask (#batch, maxlen_in).

        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            assert cache.shape == (tgt.shape[0], tgt.shape[1] - 1, self.size), '{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}'
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = tgt_mask[:, -1:, :]
        x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)[0])
        if not self.normalize_before:
            x = self.norm1(x)
        if self.src_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.norm2(x)
            x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask)[0])
            if not self.normalize_before:
                x = self.norm2(x)
        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)
        if cache is not None:
            x = torch.cat([cache, x], dim=1)
        return x, tgt_mask, memory, memory_mask


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """

    def __init__(self, idim: 'int', hidden_units: 'int', dropout_rate: 'float', activation: 'torch.nn.Module'=torch.nn.ReLU()):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim)

    def forward(self, xs: 'torch.Tensor') ->torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


def subsequent_mask(size: 'int', device: 'torch.device'=torch.device('cpu')) ->torch.Tensor:
    """Create mask for subsequent steps (size, size).

    This mask is used only in decoder which works in an auto-regressive mode.
    This means the current step could only do attention with its left steps.

    In encoder, fully attention is used when streaming is not necessary and
    the sequence is not long. In this  case, no attention mask is needed.

    When streaming is need, chunk-based attention is used in encoder. See
    subsequent_chunk_mask for the chunk-based attention mask.

    Args:
        size (int): size of mask
        str device (str): "cpu" or "cuda" or torch.Tensor.device
        dtype (torch.device): result dtype

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_mask(3)
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    """
    arange = torch.arange(size, device=device)
    mask = arange.expand(size, size)
    arange = arange.unsqueeze(-1)
    mask = mask <= arange
    return mask


class TransformerDecoder(torch.nn.Module):
    """Base class of Transfomer decoder module.
    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        src_attention: if false, encoder-decoder cross attention is not
                       applied, such as CIF model
        key_bias: whether use bias in attention.linear_k, False for whisper models.
        gradient_checkpointing: rerunning a forward-pass segment for each
            checkpointed segment during backward.
        tie_word_embedding: Tie or clone module weights depending of whether we are
            using TorchScript or not
    """

    def __init__(self, vocab_size: 'int', encoder_output_size: 'int', attention_heads: 'int'=4, linear_units: 'int'=2048, num_blocks: 'int'=6, dropout_rate: 'float'=0.1, positional_dropout_rate: 'float'=0.1, self_attention_dropout_rate: 'float'=0.0, src_attention_dropout_rate: 'float'=0.0, input_layer: 'str'='embed', use_output_layer: 'bool'=True, normalize_before: 'bool'=True, src_attention: 'bool'=True, key_bias: 'bool'=True, activation_type: 'str'='relu', gradient_checkpointing: 'bool'=False, tie_word_embedding: 'bool'=False):
        super().__init__()
        attention_dim = encoder_output_size
        activation = COSYVOICE_ACTIVATION_CLASSES[activation_type]()
        self.embed = torch.nn.Sequential(torch.nn.Identity() if input_layer == 'no_pos' else torch.nn.Embedding(vocab_size, attention_dim), COSYVOICE_EMB_CLASSES[input_layer](attention_dim, positional_dropout_rate))
        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(attention_dim, eps=1e-05)
        self.use_output_layer = use_output_layer
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        else:
            self.output_layer = torch.nn.Identity()
        self.num_blocks = num_blocks
        self.decoders = torch.nn.ModuleList([DecoderLayer(attention_dim, COSYVOICE_ATTENTION_CLASSES['selfattn'](attention_heads, attention_dim, self_attention_dropout_rate, key_bias), COSYVOICE_ATTENTION_CLASSES['selfattn'](attention_heads, attention_dim, src_attention_dropout_rate, key_bias) if src_attention else None, PositionwiseFeedForward(attention_dim, linear_units, dropout_rate, activation), dropout_rate, normalize_before) for _ in range(self.num_blocks)])
        self.gradient_checkpointing = gradient_checkpointing
        self.tie_word_embedding = tie_word_embedding

    def forward(self, memory: 'torch.Tensor', memory_mask: 'torch.Tensor', ys_in_pad: 'torch.Tensor', ys_in_lens: 'torch.Tensor', r_ys_in_pad: 'torch.Tensor'=torch.empty(0), reverse_weight: 'float'=0.0) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: not used in transformer decoder, in order to unify api
                with bidirectional decoder
            reverse_weight: not used in transformer decoder, in order to unify
                api with bidirectional decode
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                    vocab_size) if use_output_layer is True,
                torch.tensor(0.0), in order to unify api with bidirectional decoder
                olens: (batch, )
        NOTE(xcsong):
            We pass the `__call__` method of the modules instead of `forward` to the
            checkpointing API because `__call__` attaches all the hooks of the module.
            https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2
        """
        tgt = ys_in_pad
        maxlen = tgt.size(1)
        tgt_mask = ~make_pad_mask(ys_in_lens, maxlen).unsqueeze(1)
        tgt_mask = tgt_mask
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        tgt_mask = tgt_mask & m
        x, _ = self.embed(tgt)
        if self.gradient_checkpointing and self.training:
            x = self.forward_layers_checkpointed(x, tgt_mask, memory, memory_mask)
        else:
            x = self.forward_layers(x, tgt_mask, memory, memory_mask)
        if self.normalize_before:
            x = self.after_norm(x)
        if self.use_output_layer:
            x = self.output_layer(x)
        olens = tgt_mask.sum(1)
        return x, torch.tensor(0.0), olens

    def forward_layers(self, x: 'torch.Tensor', tgt_mask: 'torch.Tensor', memory: 'torch.Tensor', memory_mask: 'torch.Tensor') ->torch.Tensor:
        for layer in self.decoders:
            x, tgt_mask, memory, memory_mask = layer(x, tgt_mask, memory, memory_mask)
        return x

    @torch.jit.unused
    def forward_layers_checkpointed(self, x: 'torch.Tensor', tgt_mask: 'torch.Tensor', memory: 'torch.Tensor', memory_mask: 'torch.Tensor') ->torch.Tensor:
        for layer in self.decoders:
            x, tgt_mask, memory, memory_mask = ckpt.checkpoint(layer.__call__, x, tgt_mask, memory, memory_mask)
        return x

    def forward_one_step(self, memory: 'torch.Tensor', memory_mask: 'torch.Tensor', tgt: 'torch.Tensor', tgt_mask: 'torch.Tensor', cache: 'Optional[List[torch.Tensor]]'=None) ->Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.
            This is only used for decoding.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask, (batch, 1, maxlen_in)
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x, _ = self.embed(tgt)
        new_cache = []
        for i, decoder in enumerate(self.decoders):
            if cache is None:
                c = None
            else:
                c = cache[i]
            x, tgt_mask, memory, memory_mask = decoder(x, tgt_mask, memory, memory_mask, cache=c)
            new_cache.append(x)
        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.use_output_layer:
            y = torch.log_softmax(self.output_layer(y), dim=-1)
        return y, new_cache

    def tie_or_clone_weights(self, jit_mode: 'bool'=True):
        """Tie or clone module weights (between word_emb and output_layer)
            depending of whether we are using TorchScript or not"""
        if not self.use_output_layer:
            return
        if jit_mode:
            logging.info('clone emb.weight to output.weight')
            self.output_layer.weight = torch.nn.Parameter(self.embed[0].weight.clone())
        else:
            logging.info('tie emb.weight with output.weight')
            self.output_layer.weight = self.embed[0].weight
        if getattr(self.output_layer, 'bias', None) is not None:
            self.output_layer.bias.data = torch.nn.functional.pad(self.output_layer.bias.data, (0, self.output_layer.weight.shape[0] - self.output_layer.bias.shape[0]), 'constant', 0)


class BiTransformerDecoder(torch.nn.Module):
    """Base class of Transfomer decoder module.
    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        r_num_blocks: the number of right to left decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        key_bias: whether use bias in attention.linear_k, False for whisper models.
    """

    def __init__(self, vocab_size: 'int', encoder_output_size: 'int', attention_heads: 'int'=4, linear_units: 'int'=2048, num_blocks: 'int'=6, r_num_blocks: 'int'=0, dropout_rate: 'float'=0.1, positional_dropout_rate: 'float'=0.1, self_attention_dropout_rate: 'float'=0.0, src_attention_dropout_rate: 'float'=0.0, input_layer: 'str'='embed', use_output_layer: 'bool'=True, normalize_before: 'bool'=True, key_bias: 'bool'=True, gradient_checkpointing: 'bool'=False, tie_word_embedding: 'bool'=False):
        super().__init__()
        self.tie_word_embedding = tie_word_embedding
        self.left_decoder = TransformerDecoder(vocab_size, encoder_output_size, attention_heads, linear_units, num_blocks, dropout_rate, positional_dropout_rate, self_attention_dropout_rate, src_attention_dropout_rate, input_layer, use_output_layer, normalize_before, key_bias=key_bias, gradient_checkpointing=gradient_checkpointing, tie_word_embedding=tie_word_embedding)
        self.right_decoder = TransformerDecoder(vocab_size, encoder_output_size, attention_heads, linear_units, r_num_blocks, dropout_rate, positional_dropout_rate, self_attention_dropout_rate, src_attention_dropout_rate, input_layer, use_output_layer, normalize_before, key_bias=key_bias, gradient_checkpointing=gradient_checkpointing, tie_word_embedding=tie_word_embedding)

    def forward(self, memory: 'torch.Tensor', memory_mask: 'torch.Tensor', ys_in_pad: 'torch.Tensor', ys_in_lens: 'torch.Tensor', r_ys_in_pad: 'torch.Tensor', reverse_weight: 'float'=0.0) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: padded input token ids, int64 (batch, maxlen_out),
                used for right to left decoder
            reverse_weight: used for right to left decoder
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                    vocab_size) if use_output_layer is True,
                r_x: x: decoded token score (right to left decoder)
                    before softmax (batch, maxlen_out, vocab_size)
                    if use_output_layer is True,
                olens: (batch, )
        """
        l_x, _, olens = self.left_decoder(memory, memory_mask, ys_in_pad, ys_in_lens)
        r_x = torch.tensor(0.0)
        if reverse_weight > 0.0:
            r_x, _, olens = self.right_decoder(memory, memory_mask, r_ys_in_pad, ys_in_lens)
        return l_x, r_x, olens

    def forward_one_step(self, memory: 'torch.Tensor', memory_mask: 'torch.Tensor', tgt: 'torch.Tensor', tgt_mask: 'torch.Tensor', cache: 'Optional[List[torch.Tensor]]'=None) ->Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.
            This is only used for decoding.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask, (batch, 1, maxlen_in)
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        return self.left_decoder.forward_one_step(memory, memory_mask, tgt, tgt_mask, cache)

    def tie_or_clone_weights(self, jit_mode: 'bool'=True):
        """Tie or clone module weights (between word_emb and output_layer)
            depending of whether we are using TorchScript or not"""
        self.left_decoder.tie_or_clone_weights(jit_mode)
        self.right_decoder.tie_or_clone_weights(jit_mode)


class BaseSubsampling(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset: 'Union[int, torch.Tensor]', size: 'int') ->torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)


class Conv1dSubsampling2(BaseSubsampling):
    """Convolutional 1D subsampling (to 1/2 length).
       It is designed for Whisper, ref:
       https://github.com/openai/whisper/blob/main/whisper/model.py

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: 'int', odim: 'int', dropout_rate: 'float', pos_enc_class: 'torch.nn.Module'):
        """Construct an Conv1dSubsampling2 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv1d(idim, odim, kernel_size=3, padding=1), torch.nn.GELU(), torch.nn.Conv1d(odim, odim, kernel_size=3, stride=2, padding=1), torch.nn.GELU())
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 2
        self.right_context = 4

    def forward(self, x: 'torch.Tensor', x_mask: 'torch.Tensor', offset: 'Union[int, torch.Tensor]'=0) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.
            torch.Tensor: positional encoding

        """
        time = x.size(1)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, (time + 1) % 2::2]


class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: 'int', odim: 'int', dropout_rate: 'float', pos_enc_class: 'torch.nn.Module'):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(1, odim, 3, 2), torch.nn.ReLU(), torch.nn.Conv2d(odim, odim, 3, 2), torch.nn.ReLU())
        self.out = torch.nn.Sequential(torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 4
        self.right_context = 6

    def forward(self, x: 'torch.Tensor', x_mask: 'torch.Tensor', offset: 'Union[int, torch.Tensor]'=0) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
            torch.Tensor: positional encoding

        """
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 2::2]


class Conv2dSubsampling6(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/6 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
    """

    def __init__(self, idim: 'int', odim: 'int', dropout_rate: 'float', pos_enc_class: 'torch.nn.Module'):
        """Construct an Conv2dSubsampling6 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(1, odim, 3, 2), torch.nn.ReLU(), torch.nn.Conv2d(odim, odim, 5, 3), torch.nn.ReLU())
        self.linear = torch.nn.Linear(odim * (((idim - 1) // 2 - 2) // 3), odim)
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 6
        self.right_context = 10

    def forward(self, x: 'torch.Tensor', x_mask: 'torch.Tensor', offset: 'Union[int, torch.Tensor]'=0) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 6.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 6.
            torch.Tensor: positional encoding
        """
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.linear(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 4::3]


class Conv2dSubsampling8(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/8 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: 'int', odim: 'int', dropout_rate: 'float', pos_enc_class: 'torch.nn.Module'):
        """Construct an Conv2dSubsampling8 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(1, odim, 3, 2), torch.nn.ReLU(), torch.nn.Conv2d(odim, odim, 3, 2), torch.nn.ReLU(), torch.nn.Conv2d(odim, odim, 3, 2), torch.nn.ReLU())
        self.linear = torch.nn.Linear(odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim)
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 8
        self.right_context = 14

    def forward(self, x: 'torch.Tensor', x_mask: 'torch.Tensor', offset: 'Union[int, torch.Tensor]'=0) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.
            torch.Tensor: positional encoding
        """
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.linear(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 2::2][:, :, 2::2]


class EmbedinigNoSubsampling(BaseSubsampling):
    """Embedding input without subsampling
    """

    def __init__(self, idim: 'int', odim: 'int', dropout_rate: 'float', pos_enc_class: 'torch.nn.Module'):
        super().__init__()
        self.embed = torch.nn.Embedding(idim, odim)
        self.pos_enc = pos_enc_class

    def forward(self, x: 'torch.Tensor', x_mask: 'torch.Tensor', offset: 'Union[int, torch.Tensor]'=0) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .

        """
        x = self.embed(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


class LegacyLinearNoSubsampling(BaseSubsampling):
    """Linear transform the input without subsampling

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: 'int', odim: 'int', dropout_rate: 'float', pos_enc_class: 'torch.nn.Module'):
        """Construct an linear object."""
        super().__init__()
        self.out = torch.nn.Sequential(torch.nn.Linear(idim, odim), torch.nn.LayerNorm(odim, eps=1e-05), torch.nn.Dropout(dropout_rate), torch.nn.ReLU())
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(self, x: 'torch.Tensor', x_mask: 'torch.Tensor', offset: 'Union[int, torch.Tensor]'=0) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .

        """
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


class LinearNoSubsampling(BaseSubsampling):
    """Linear transform the input without subsampling

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: 'int', odim: 'int', dropout_rate: 'float', pos_enc_class: 'torch.nn.Module'):
        """Construct an linear object."""
        super().__init__()
        self.out = torch.nn.Sequential(torch.nn.Linear(idim, odim), torch.nn.LayerNorm(odim, eps=1e-05), torch.nn.Dropout(dropout_rate))
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(self, x: 'torch.Tensor', x_mask: 'torch.Tensor', offset: 'Union[int, torch.Tensor]'=0) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .

        """
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


COSYVOICE_SUBSAMPLE_CLASSES = {'linear': LinearNoSubsampling, 'linear_legacy': LegacyLinearNoSubsampling, 'embed': EmbedinigNoSubsampling, 'conv1d2': Conv1dSubsampling2, 'conv2d': Conv2dSubsampling4, 'conv2d6': Conv2dSubsampling6, 'conv2d8': Conv2dSubsampling8, 'paraformer_dummy': torch.nn.Identity}


def subsequent_chunk_mask(size: 'int', chunk_size: 'int', num_left_chunks: 'int'=-1, device: 'torch.device'=torch.device('cpu')) ->torch.Tensor:
    """Create mask for subsequent steps (size, size) with chunk size,
       this is for streaming encoder

    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks
        device (torch.device): "cpu" or "cuda" or torch.Tensor.device

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    for i in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
        ending = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, start:ending] = True
    return ret


def add_optional_chunk_mask(xs: 'torch.Tensor', masks: 'torch.Tensor', use_dynamic_chunk: 'bool', use_dynamic_left_chunk: 'bool', decoding_chunk_size: 'int', static_chunk_size: 'int', num_decoding_left_chunks: 'int', enable_full_context: 'bool'=True):
    """ Apply optional mask for encoder.

    Args:
        xs (torch.Tensor): padded input, (B, L, D), L for max length
        mask (torch.Tensor): mask for xs, (B, 1, L)
        use_dynamic_chunk (bool): whether to use dynamic chunk or not
        use_dynamic_left_chunk (bool): whether to use dynamic left chunk for
            training.
        decoding_chunk_size (int): decoding chunk size for dynamic chunk, it's
            0: default for training, use random dynamic chunk.
            <0: for decoding, use full chunk.
            >0: for decoding, use fixed chunk size as set.
        static_chunk_size (int): chunk size for static chunk training/decoding
            if it's greater than 0, if use_dynamic_chunk is true,
            this parameter will be ignored
        num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
            >=0: use num_decoding_left_chunks
            <0: use all left chunks
        enable_full_context (bool):
            True: chunk size is either [1, 25] or full context(max_len)
            False: chunk size ~ U[1, 25]

    Returns:
        torch.Tensor: chunk mask of the input xs.
    """
    if use_dynamic_chunk:
        max_len = xs.size(1)
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            chunk_size = torch.randint(1, max_len, (1,)).item()
            num_left_chunks = -1
            if chunk_size > max_len // 2 and enable_full_context:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_left_chunks = torch.randint(0, max_left_chunks, (1,)).item()
        chunk_masks = subsequent_chunk_mask(xs.size(1), chunk_size, num_left_chunks, xs.device)
        chunk_masks = chunk_masks.unsqueeze(0)
        chunk_masks = masks & chunk_masks
    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(xs.size(1), static_chunk_size, num_left_chunks, xs.device)
        chunk_masks = chunk_masks.unsqueeze(0)
        chunk_masks = masks & chunk_masks
    else:
        chunk_masks = masks
    return chunk_masks


class BaseEncoder(torch.nn.Module):

    def __init__(self, input_size: 'int', output_size: 'int'=256, attention_heads: 'int'=4, linear_units: 'int'=2048, num_blocks: 'int'=6, dropout_rate: 'float'=0.1, positional_dropout_rate: 'float'=0.1, attention_dropout_rate: 'float'=0.0, input_layer: 'str'='conv2d', pos_enc_layer_type: 'str'='abs_pos', normalize_before: 'bool'=True, static_chunk_size: 'int'=0, use_dynamic_chunk: 'bool'=False, global_cmvn: 'torch.nn.Module'=None, use_dynamic_left_chunk: 'bool'=False, gradient_checkpointing: 'bool'=False):
        """
        Args:
            input_size (int): input dim
            output_size (int): dimension of attention
            attention_heads (int): the number of heads of multi head attention
            linear_units (int): the hidden units number of position-wise feed
                forward
            num_blocks (int): the number of decoder blocks
            dropout_rate (float): dropout rate
            attention_dropout_rate (float): dropout rate in attention
            positional_dropout_rate (float): dropout rate after adding
                positional encoding
            input_layer (str): input layer type.
                optional [linear, conv2d, conv2d6, conv2d8]
            pos_enc_layer_type (str): Encoder positional encoding layer type.
                opitonal [abs_pos, scaled_abs_pos, rel_pos, no_pos]
            normalize_before (bool):
                True: use layer_norm before each sub-block of a layer.
                False: use layer_norm after each sub-block of a layer.
            static_chunk_size (int): chunk size for static chunk training and
                decoding
            use_dynamic_chunk (bool): whether use dynamic chunk size for
                training or not, You can only use fixed chunk(chunk_size > 0)
                or dyanmic chunk size(use_dynamic_chunk = True)
            global_cmvn (Optional[torch.nn.Module]): Optional GlobalCMVN module
            use_dynamic_left_chunk (bool): whether use dynamic left chunk in
                dynamic chunk training
            key_bias: whether use bias in attention.linear_k, False for whisper models.
            gradient_checkpointing: rerunning a forward-pass segment for each
                checkpointed segment during backward.
        """
        super().__init__()
        self._output_size = output_size
        self.global_cmvn = global_cmvn
        self.embed = COSYVOICE_SUBSAMPLE_CLASSES[input_layer](input_size, output_size, dropout_rate, COSYVOICE_EMB_CLASSES[pos_enc_layer_type](output_size, positional_dropout_rate))
        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-05)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.gradient_checkpointing = gradient_checkpointing

    def output_size(self) ->int:
        return self._output_size

    def forward(self, xs: 'torch.Tensor', xs_lens: 'torch.Tensor', decoding_chunk_size: 'int'=0, num_decoding_left_chunks: 'int'=-1) ->Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        NOTE(xcsong):
            We pass the `__call__` method of the modules instead of `forward` to the
            checkpointing API because `__call__` attaches all the hooks of the module.
            https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2
        """
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks
        chunk_masks = add_optional_chunk_mask(xs, masks, self.use_dynamic_chunk, self.use_dynamic_left_chunk, decoding_chunk_size, self.static_chunk_size, num_decoding_left_chunks)
        if self.gradient_checkpointing and self.training:
            xs = self.forward_layers_checkpointed(xs, chunk_masks, pos_emb, mask_pad)
        else:
            xs = self.forward_layers(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks

    def forward_layers(self, xs: 'torch.Tensor', chunk_masks: 'torch.Tensor', pos_emb: 'torch.Tensor', mask_pad: 'torch.Tensor') ->torch.Tensor:
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs

    @torch.jit.unused
    def forward_layers_checkpointed(self, xs: 'torch.Tensor', chunk_masks: 'torch.Tensor', pos_emb: 'torch.Tensor', mask_pad: 'torch.Tensor') ->torch.Tensor:
        for layer in self.encoders:
            xs, chunk_masks, _, _ = ckpt.checkpoint(layer.__call__, xs, chunk_masks, pos_emb, mask_pad)
        return xs

    @torch.jit.export
    def forward_chunk(self, xs: 'torch.Tensor', offset: 'int', required_cache_size: 'int', att_cache: 'torch.Tensor'=torch.zeros(0, 0, 0, 0), cnn_cache: 'torch.Tensor'=torch.zeros(0, 0, 0, 0), att_mask: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool)) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Forward just one chunk

        Args:
            xs (torch.Tensor): chunk input, with shape (b=1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate +                         subsample.right_context + 1`
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (elayers, b=1, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`

        Returns:
            torch.Tensor: output of current input xs,
                with shape (b=1, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                dynamic shape (elayers, head, ?, d_k * 2)
                depending on required_cache_size.
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.

        """
        assert xs.size(0) == 1
        tmp_masks = torch.ones(1, xs.size(1), device=xs.device, dtype=torch.bool)
        tmp_masks = tmp_masks.unsqueeze(1)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, _ = self.embed(xs, tmp_masks, offset)
        elayers, cache_t1 = att_cache.size(0), att_cache.size(2)
        chunk_size = xs.size(1)
        attention_key_size = cache_t1 + chunk_size
        pos_emb = self.embed.position_encoding(offset=offset - cache_t1, size=attention_key_size)
        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = attention_key_size
        else:
            next_cache_start = max(attention_key_size - required_cache_size, 0)
        r_att_cache = []
        r_cnn_cache = []
        for i, layer in enumerate(self.encoders):
            xs, _, new_att_cache, new_cnn_cache = layer(xs, att_mask, pos_emb, att_cache=att_cache[i:i + 1] if elayers > 0 else att_cache, cnn_cache=cnn_cache[i] if cnn_cache.size(0) > 0 else cnn_cache)
            r_att_cache.append(new_att_cache[:, :, next_cache_start:, :])
            r_cnn_cache.append(new_cnn_cache.unsqueeze(0))
        if self.normalize_before:
            xs = self.after_norm(xs)
        r_att_cache = torch.cat(r_att_cache, dim=0)
        r_cnn_cache = torch.cat(r_cnn_cache, dim=0)
        return xs, r_att_cache, r_cnn_cache

    @torch.jit.unused
    def forward_chunk_by_chunk(self, xs: 'torch.Tensor', decoding_chunk_size: 'int', num_decoding_left_chunks: 'int'=-1) ->Tuple[torch.Tensor, torch.Tensor]:
        """ Forward input chunk by chunk with chunk_size like a streaming
            fashion

        Here we should pay special attention to computation cache in the
        streaming style forward chunk by chunk. Three things should be taken
        into account for computation in the current network:
            1. transformer/conformer encoder layers output cache
            2. convolution in conformer
            3. convolution in subsampling

        However, we don't implement subsampling cache for:
            1. We can control subsampling module to output the right result by
               overlapping input instead of cache left context, even though it
               wastes some computation, but subsampling only takes a very
               small fraction of computation in the whole model.
            2. Typically, there are several covolution layers with subsampling
               in subsampling module, it is tricky and complicated to do cache
               with different convolution layers with different subsampling
               rate.
            3. Currently, nn.Sequential is used to stack all the convolution
               layers in subsampling, we need to rewrite it to make it work
               with cache, which is not preferred.
        Args:
            xs (torch.Tensor): (1, max_len, dim)
            chunk_size (int): decoding chunk size
        """
        assert decoding_chunk_size > 0
        assert self.static_chunk_size > 0 or self.use_dynamic_chunk
        subsampling = self.embed.subsampling_rate
        context = self.embed.right_context + 1
        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        num_frames = xs.size(1)
        att_cache: 'torch.Tensor' = torch.zeros((0, 0, 0, 0), device=xs.device)
        cnn_cache: 'torch.Tensor' = torch.zeros((0, 0, 0, 0), device=xs.device)
        outputs = []
        offset = 0
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks
        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + decoding_window, num_frames)
            chunk_xs = xs[:, cur:end, :]
            y, att_cache, cnn_cache = self.forward_chunk(chunk_xs, offset, required_cache_size, att_cache, cnn_cache)
            outputs.append(y)
            offset += y.size(1)
        ys = torch.cat(outputs, 1)
        masks = torch.ones((1, 1, ys.size(1)), device=ys.device, dtype=torch.bool)
        return ys, masks


class RelativePositionBias(nn.Module):

    def __init__(self, config: 'T5Config', bidirectional: 'bool'):
        self.bidirectional = bidirectional
        self.num_buckets = config.relative_attention_num_buckets
        self.max_distance = config.relative_attention_max_distance
        self.n_heads = config.num_heads
        self.embeddings = nn.Embedding(self.num_buckets, self.n_heads)

    @staticmethod
    def _relative_position_bucket(rpos, bidirectional, num_buckets, max_distance):
        num_buckets = num_buckets // 2 if bidirectional else num_buckets
        max_exact = num_buckets // 2
        abspos = rpos.abs()
        is_small = abspos < max_exact
        scale = (num_buckets - max_exact) / math.log(max_distance / max_exact)
        buckets_large = (mx.log(abspos / max_exact) * scale).astype(mx.int16)
        buckets_large = mx.minimum(max_exact + buckets_large, num_buckets - 1)
        buckets = mx.where(is_small, abspos, buckets_large)
        if bidirectional:
            buckets = buckets + (rpos > 0) * num_buckets
        else:
            buckets = buckets * (rpos < 0)
        return buckets

    def __call__(self, query_length: 'int', key_length: 'int', offset: 'int'=0):
        """Compute binned relative position bias"""
        context_position = mx.arange(offset, query_length)[:, None]
        memory_position = mx.arange(key_length)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(relative_position, bidirectional=self.bidirectional, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.embeddings(relative_position_bucket)
        return values.transpose(2, 0, 1)


class DenseActivation(nn.Module):

    def __init__(self, config: 'T5Config'):
        super().__init__()
        mlp_dims = config.d_ff or config.d_model * 4
        self.gated = config.feed_forward_proj.startswith('gated')
        if self.gated:
            self.wi_0 = nn.Linear(config.d_model, mlp_dims, bias=False)
            self.wi_1 = nn.Linear(config.d_model, mlp_dims, bias=False)
        else:
            self.wi = nn.Linear(config.d_model, mlp_dims, bias=False)
        self.wo = nn.Linear(mlp_dims, config.d_model, bias=False)
        activation = config.feed_forward_proj.removeprefix('gated-')
        if activation == 'relu':
            self.act = nn.relu
        elif activation == 'gelu':
            self.act = nn.gelu
        elif activation == 'silu':
            self.act = nn.silu
        else:
            raise ValueError(f'Unknown activation: {activation}')

    def __call__(self, x):
        if self.gated:
            hidden_act = self.act(self.wi_0(x))
            hidden_linear = self.wi_1(x)
            x = hidden_act * hidden_linear
        else:
            x = self.act(self.wi(x))
        return self.wo(x)


class Linear(nn.Linear):

    def forward(self, x: 'Tensor') ->Tensor:
        return F.linear(x, self.weight, None if self.bias is None else self.bias)


class MultiHeadAttention(nn.Module):

    def __init__(self, n_state: 'int', n_head: 'int'):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(self, x: 'Tensor', xa: 'Optional[Tensor]'=None, mask: 'Optional[Tensor]'=None, kv_cache: 'Optional[dict]'=None):
        q = self.query(x)
        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]
        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(self, q: 'Tensor', k: 'Tensor', v: 'Tensor', mask: 'Optional[Tensor]'=None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()
        w = F.softmax(qk, dim=-1)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class TransformerEncoderLayer(nn.Module):

    def __init__(self, config: 'T5Config'):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ln1 = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ln2 = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dense = DenseActivation(config)

    def __call__(self, x, mask):
        y = self.ln1(x)
        y, _ = self.attention(y, y, y, mask=mask)
        x = x + y
        y = self.ln2(x)
        y = self.dense(y)
        return x + y


class TransformerEncoder(nn.Module):

    def __init__(self, config: 'T5Config'):
        super().__init__()
        self.layers = [TransformerEncoderLayer(config) for i in range(config.num_layers)]
        self.ln = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.relative_attention_bias = RelativePositionBias(config, bidirectional=True)

    def __call__(self, x: 'mx.array'):
        pos_bias = self.relative_attention_bias(x.shape[1], x.shape[1])
        pos_bias = pos_bias.astype(x.dtype)
        for layer in self.layers:
            x = layer(x, mask=pos_bias)
        return self.ln(x)


class ConformerEncoderLayer(nn.Module):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module
             instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: use layer_norm after each sub-block.
    """

    def __init__(self, size: 'int', self_attn: 'torch.nn.Module', feed_forward: 'Optional[nn.Module]'=None, feed_forward_macaron: 'Optional[nn.Module]'=None, conv_module: 'Optional[nn.Module]'=None, dropout_rate: 'float'=0.1, normalize_before: 'bool'=True):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size, eps=1e-05)
        self.norm_mha = nn.LayerNorm(size, eps=1e-05)
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-05)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = nn.LayerNorm(size, eps=1e-05)
            self.norm_final = nn.LayerNorm(size, eps=1e-05)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

    def forward(self, x: 'torch.Tensor', mask: 'torch.Tensor', pos_emb: 'torch.Tensor', mask_pad: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool), att_cache: 'torch.Tensor'=torch.zeros((0, 0, 0, 0)), cnn_cache: 'torch.Tensor'=torch.zeros((0, 0, 0, 0))) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time，time),
                (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): positional encoding, must not be None
                for ConformerEncoderLayer.
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1，time), (0, 0, 0) means fake mask.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
                (#batch=1, size, cache_t2)
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time).
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cahce tensor (#batch, size, cache_t2).
        """
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb, att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)
            if not self.normalize_before:
                x = self.norm_conv(x)
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)
        if self.conv_module is not None:
            x = self.norm_final(x)
        return x, mask, new_att_cache, new_cnn_cache


class ConformerEncoder(BaseEncoder):
    """Conformer encoder module."""

    def __init__(self, input_size: 'int', output_size: 'int'=256, attention_heads: 'int'=4, linear_units: 'int'=2048, num_blocks: 'int'=6, dropout_rate: 'float'=0.1, positional_dropout_rate: 'float'=0.1, attention_dropout_rate: 'float'=0.0, input_layer: 'str'='conv2d', pos_enc_layer_type: 'str'='rel_pos', normalize_before: 'bool'=True, static_chunk_size: 'int'=0, use_dynamic_chunk: 'bool'=False, global_cmvn: 'torch.nn.Module'=None, use_dynamic_left_chunk: 'bool'=False, positionwise_conv_kernel_size: 'int'=1, macaron_style: 'bool'=True, selfattention_layer_type: 'str'='rel_selfattn', activation_type: 'str'='swish', use_cnn_module: 'bool'=True, cnn_module_kernel: 'int'=15, causal: 'bool'=False, cnn_module_norm: 'str'='batch_norm', key_bias: 'bool'=True, gradient_checkpointing: 'bool'=False):
        """Construct ConformerEncoder

        Args:
            input_size to use_dynamic_chunk, see in BaseEncoder
            positionwise_conv_kernel_size (int): Kernel size of positionwise
                conv1d layer.
            macaron_style (bool): Whether to use macaron style for
                positionwise layer.
            selfattention_layer_type (str): Encoder attention layer type,
                the parameter has no effect now, it's just for configure
                compatibility.
            activation_type (str): Encoder activation function type.
            use_cnn_module (bool): Whether to use convolution module.
            cnn_module_kernel (int): Kernel size of convolution module.
            causal (bool): whether to use causal convolution or not.
            key_bias: whether use bias in attention.linear_k, False for whisper models.
        """
        super().__init__(input_size, output_size, attention_heads, linear_units, num_blocks, dropout_rate, positional_dropout_rate, attention_dropout_rate, input_layer, pos_enc_layer_type, normalize_before, static_chunk_size, use_dynamic_chunk, global_cmvn, use_dynamic_left_chunk, gradient_checkpointing)
        activation = COSYVOICE_ACTIVATION_CLASSES[activation_type]()
        encoder_selfattn_layer_args = attention_heads, output_size, attention_dropout_rate, key_bias
        positionwise_layer_args = output_size, linear_units, dropout_rate, activation
        convolution_layer_args = output_size, cnn_module_kernel, activation, cnn_module_norm, causal
        self.encoders = torch.nn.ModuleList([ConformerEncoderLayer(output_size, COSYVOICE_ATTENTION_CLASSES[selfattention_layer_type](*encoder_selfattn_layer_args), PositionwiseFeedForward(*positionwise_layer_args), PositionwiseFeedForward(*positionwise_layer_args) if macaron_style else None, ConvolutionModule(*convolution_layer_args) if use_cnn_module else None, dropout_rate, normalize_before) for _ in range(num_blocks)])


class MoEFFNLayer(torch.nn.Module):
    """
    Mixture of expert with Positionwise feed forward layer
    See also figure 1 in https://arxiv.org/pdf/2305.15663.pdf
    The output dim is same with the input dim.

    Modified from https://github.com/Lightning-AI/lit-gpt/pull/823
                  https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219
    Args:
        n_expert: number of expert.
        n_expert_per_token: The actual number of experts used for each frame
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """

    def __init__(self, n_expert: 'int', n_expert_per_token: 'int', idim: 'int', hidden_units: 'int', dropout_rate: 'float', activation: 'torch.nn.Module'=torch.nn.ReLU()):
        super(MoEFFNLayer, self).__init__()
        self.gate = torch.nn.Linear(idim, n_expert, bias=False)
        self.experts = torch.nn.ModuleList(PositionwiseFeedForward(idim, hidden_units, dropout_rate, activation) for _ in range(n_expert))
        self.n_expert_per_token = n_expert_per_token

    def forward(self, xs: 'torch.Tensor') ->torch.Tensor:
        """Foward function.
        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)

        """
        B, L, D = xs.size()
        xs = xs.view(-1, D)
        router = self.gate(xs)
        logits, indices = torch.topk(router, self.n_expert_per_token)
        weights = torch.nn.functional.softmax(logits, dim=1, dtype=torch.float)
        output = torch.zeros_like(xs)
        for i, expert in enumerate(self.experts):
            mask = indices == i
            batch_idx, ith_expert = torch.where(mask)
            output[batch_idx] += weights[batch_idx, ith_expert, None] * expert(xs[batch_idx])
        return output.view(B, L, D)


class CLIPVisionTower(nn.Module):

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, ignore_mismatched_sizes=True)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images, output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class HybridVisionTower(nn.Module):

    def __init__(self, high_res_cfg: 'Dict', low_res_cfg: 'Dict', freeze_high: 'bool'=False, freeze_low: 'bool'=False, concat_type: "Literal['feature', 'sequence', 'add', 'tuple']"='tuple', **ignore_kwargs):
        super().__init__()
        self.vision_tower_high = CLIPVisionTower(**high_res_cfg)
        self.vision_tower_low = CLIPVisionTower(**low_res_cfg)
        self.low_res_size = low_res_cfg['image_size']
        self.concat_type = concat_type
        self.high_layer_norm = nn.LayerNorm(high_res_cfg.get('output_dim', 1024))
        self.low_layer_norm = nn.LayerNorm(low_res_cfg.get('output_dim', 1024))
        if freeze_high:
            for p_name, p in self.vision_tower_high.named_parameters():
                p.requires_grad = False
            self.vision_tower_high = self.vision_tower_high.eval()
        else:
            for p_name, p in self.vision_tower_high.named_parameters():
                if 'downsamples' in p_name or 'neck' in p_name:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        if freeze_low:
            for p in self.vision_tower_low.parameters():
                p.requires_grad = False
            self.vision_tower_low = self.vision_tower_low.eval()
        self.resize = torchvision.transforms.Resize(self.low_res_size, antialias=True)

    def forward(self, images: 'torch.Tensor'):
        """

        Args:
            images (torch.Tensor): [bs, 3, H, W]

        Returns:
            res (torch.Tensor): [bs, t, c]
        """
        high_images = images
        low_images = self.resize(images)
        high_res = self.vision_tower_high(high_images)
        high_res = rearrange(high_res, 'b c h w -> b (h w) c')
        low_res = self.vision_tower_low(low_images)
        if self.concat_type == 'feature':
            images_features = torch.cat([high_res, low_res], dim=-1)
        elif self.concat_type == 'sequence':
            images_features = torch.cat([high_res, low_res], dim=1)
        elif self.concat_type == 'add':
            images_features = high_res + low_res
        elif self.concat_type == 'tuple':
            images_features = high_res, low_res
        else:
            raise ValueError('Currently only support `feature`, `sequence`, `add` and `tuple` concat type.')
        return images_features


class MlpProjector(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.projector_type == 'identity':
            modules = nn.Identity()
        elif cfg.projector_type == 'linear':
            modules = nn.Linear(cfg.input_dim, cfg.n_embed)
        elif cfg.projector_type == 'mlp_gelu':
            mlp_depth = cfg.get('depth', 1)
            modules = [nn.Linear(cfg.input_dim, cfg.n_embed)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)
        elif cfg.projector_type == 'low_high_hybrid_split_mlp_gelu':
            mlp_depth = cfg.get('depth', 1)
            self.high_up_proj = nn.Linear(cfg.input_dim, cfg.n_embed // 2)
            self.low_up_proj = nn.Linear(cfg.input_dim, cfg.n_embed // 2)
            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)
        else:
            raise ValueError(f'Unknown projector type: {cfg.projector_type}')
        self.layers = modules

    def forward(self, x_or_tuple: 'Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]'):
        """

        Args:
            x_or_tuple (Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:  if it is a tuple of torch.Tensor,
                then it comes from the hybrid vision encoder, and x = high_res_x, low_res_x);
                otherwise it is the feature from the single vision encoder.

        Returns:
            x (torch.Tensor): [b, s, c]
        """
        if isinstance(x_or_tuple, tuple):
            high_x, low_x = x_or_tuple
            high_x = self.high_up_proj(high_x)
            low_x = self.low_up_proj(low_x)
            x = torch.concat([high_x, low_x], dim=-1)
        else:
            x = x_or_tuple
        return self.layers(x)


class MLPBlock(nn.Module):

    def __init__(self, embedding_dim: 'int', mlp_dim: 'int', act: 'Type[nn.Module]'=nn.GELU) ->None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):

    def __init__(self, num_channels: 'int', eps: 'float'=1e-06) ->None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def drop_path(x, drop_prob: 'float'=0.0, training: 'bool'=False, scale_by_keep: 'bool'=True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: 'float'=0.0, scale_by_keep: 'bool'=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class LayerScale(nn.Module):

    def __init__(self, dim: 'int', init_values: 'float'=1e-05, inplace: 'bool'=False) ->None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(self, kernel_size: 'Tuple[int, int]'=(16, 16), stride: 'Tuple[int, int]'=(16, 16), padding: 'Tuple[int, int]'=(0, 0), in_chans: 'int'=3, embed_dim: 'int'=768) ->None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


class ImageEncoderViT(nn.Module):

    def __init__(self, img_size: 'int'=1024, patch_size: 'int'=16, in_chans: 'int'=3, embed_dim: 'int'=768, depth: 'int'=12, num_heads: 'int'=12, mlp_ratio: 'float'=4.0, out_chans: 'int'=256, qkv_bias: 'bool'=True, norm_layer: 'Type[nn.Module]'=nn.LayerNorm, act_layer: 'Type[nn.Module]'=nn.GELU, use_abs_pos: 'bool'=True, use_rel_pos: 'bool'=False, rel_pos_zero_init: 'bool'=True, window_size: 'int'=0, global_attn_indexes: 'Tuple[int, ...]'=(), downsample_channels: 'Tuple[int, ...]'=(512, 1024)) ->None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
            downsample_channels (list): Channels for downsampling layers.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed(kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), in_chans=in_chans, embed_dim=embed_dim)
        self.pos_embed: 'Optional[nn.Parameter]' = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim))
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer, act_layer=act_layer, use_rel_pos=use_rel_pos, rel_pos_zero_init=rel_pos_zero_init, window_size=window_size if i not in global_attn_indexes else 0, input_size=(img_size // patch_size, img_size // patch_size))
            self.blocks.append(block)
        self.neck = nn.Sequential(nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False), LayerNorm2d(out_chans), nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False), LayerNorm2d(out_chans))
        in_channels = out_chans
        downsamples = []
        for i in range(len(downsample_channels)):
            out_channels = downsample_channels[i]
            downsamples.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False))
            in_channels = out_channels
        self.downsamples = nn.Sequential(*downsamples)
        self.sam_hd = True
        if self.sam_hd:
            self.hd_alpha_downsamples = nn.Parameter(torch.zeros(1))
            self.neck_hd = copy.deepcopy(self.neck)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        global_features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.sam_hd and blk.window_size == 0:
                global_features.append(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        x_dtype = x.dtype
        x = F.interpolate(x.float(), size=(96, 96), mode='bilinear', align_corners=False)
        x = self.downsamples(x)
        if self.sam_hd:
            first_global_feature = self.neck_hd(global_features[0].permute(0, 3, 1, 2))
            x_dtype = first_global_feature.dtype
            first_global_feature = F.interpolate(first_global_feature.float(), size=(96, 96), mode='bilinear', align_corners=False)
            first_global_feature = self.downsamples(first_global_feature)
            x = x + first_global_feature * self.hd_alpha_downsamples
        return x


def init_weights_vit_timm(module: 'nn.Module', name: 'str'='') ->None:
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def get_extra_padding_for_conv1d(x: 'torch.Tensor', kernel_size: 'int', stride: 'int', padding_total: 'int'=0) ->int:
    """See `pad_for_conv1d`."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(x: 'torch.Tensor', paddings: 'tuple[int, int]', mode: 'str'='zeros', value: 'float'=0.0):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right
    before the reflection happen.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


class FishConvNet(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, groups=1):
        super(FishConvNet, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, groups=groups)
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.dilation = dilation

    def forward(self, x):
        pad = self.kernel_size - self.stride
        extra_padding = get_extra_padding_for_conv1d(x, self.kernel_size, self.stride, pad)
        x = pad1d(x, (pad, extra_padding), mode='constant', value=0)
        return self.conv(x).contiguous()

    def weight_norm(self, name='weight', dim=0):
        self.conv = weight_norm(self.conv, name=name, dim=dim)
        return self

    def remove_weight_norm(self):
        self.conv = remove_parametrizations(self.conv)
        return self


def unpad1d(x: 'torch.Tensor', paddings: 'tuple[int, int]'):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert padding_left + padding_right <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


class FishTransConvNet(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1):
        super(FishTransConvNet, self).__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        x = self.conv(x)
        pad = self.kernel_size - self.stride
        padding_right = math.ceil(pad)
        padding_left = pad - padding_right
        x = unpad1d(x, (padding_left, padding_right))
        return x.contiguous()

    def weight_norm(self, name='weight', dim=0):
        self.conv = weight_norm(self.conv, name=name, dim=dim)
        return self

    def remove_weight_norm(self):
        self.conv = remove_parametrizations(self.conv)
        return self


LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.h = h
        self.convs1 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2])))])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ParallelBlock(nn.Module):

    def __init__(self, channels: 'int', kernel_sizes: 'tuple[int]'=(3, 7, 11), dilation_sizes: 'tuple[tuple[int]]'=((1, 3, 5), (1, 3, 5), (1, 3, 5))):
        super().__init__()
        assert len(kernel_sizes) == len(dilation_sizes)
        self.blocks = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilation_sizes):
            self.blocks.append(ResBlock1(channels, k, d))

    def forward(self, x):
        return torch.stack([block(x) for block in self.blocks], dim=0).mean(dim=0)

    def remove_parametrizations(self):
        for block in self.blocks:
            block.remove_parametrizations()


class HiFiGANGenerator(nn.Module):

    def __init__(self, *, hop_length: int=512, upsample_rates: tuple[int]=(8, 8, 2, 2, 2), upsample_kernel_sizes: tuple[int]=(16, 16, 8, 2, 2), resblock_kernel_sizes: tuple[int]=(3, 7, 11), resblock_dilation_sizes: tuple[tuple[int]]=((1, 3, 5), (1, 3, 5), (1, 3, 5)), num_mels: int=128, upsample_initial_channel: int=512, pre_conv_kernel_size: int=7, post_conv_kernel_size: int=7, post_activation: Callable=partial(nn.SiLU, inplace=True)):
        super().__init__()
        assert prod(upsample_rates) == hop_length, f'hop_length must be {prod(upsample_rates)}'
        self.conv_pre = FishConvNet(num_mels, upsample_initial_channel, pre_conv_kernel_size, stride=1).weight_norm()
        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)
        self.noise_convs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(FishTransConvNet(upsample_initial_channel // 2 ** i, upsample_initial_channel // 2 ** (i + 1), k, stride=u).weight_norm())
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // 2 ** (i + 1)
            self.resblocks.append(ParallelBlock(ch, resblock_kernel_sizes, resblock_dilation_sizes))
        self.activation_post = post_activation()
        self.conv_post = FishConvNet(ch, 1, post_conv_kernel_size, stride=1).weight_norm()
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.silu(x, inplace=True)
            x = self.ups[i](x)
            if self.training and self.checkpointing:
                x = checkpoint(self.resblocks[i], x, use_reentrant=False)
            else:
                x = self.resblocks[i](x)
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_parametrizations(self):
        for up in self.ups:
            remove_parametrizations(up, tensor_name='weight')
        for block in self.resblocks:
            block.remove_parametrizations()
        remove_parametrizations(self.conv_pre, tensor_name='weight')
        remove_parametrizations(self.conv_post, tensor_name='weight')


class LayerNorm(nn.LayerNorm):

    def forward(self, x: 'Tensor') ->Tensor:
        return super().forward(x.float()).type(x.dtype)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        kernel_size (int): Kernel size for depthwise conv. Default: 7.
        dilation (int): Dilation for depthwise conv. Default: 1.
    """

    def __init__(self, dim: 'int', drop_path: 'float'=0.0, layer_scale_init_value: 'float'=1e-06, mlp_ratio: 'float'=4.0, kernel_size: 'int'=7, dilation: 'int'=1):
        super().__init__()
        self.dwconv = FishConvNet(dim, dim, kernel_size=kernel_size, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-06)
        self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, apply_residual: 'bool'=True):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1)
        x = self.drop_path(x)
        if apply_residual:
            x = input + x
        return x


class ConvNeXtEncoder(nn.Module):

    def __init__(self, input_channels: 'int'=3, depths: 'list[int]'=[3, 3, 9, 3], dims: 'list[int]'=[96, 192, 384, 768], drop_path_rate: 'float'=0.0, layer_scale_init_value: 'float'=1e-06, kernel_size: 'int'=7):
        super().__init__()
        assert len(depths) == len(dims)
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(FishConvNet(input_channels, dims[0], kernel_size=7), LayerNorm(dims[0], eps=1e-06, data_format='channels_first'))
        self.downsample_layers.append(stem)
        for i in range(len(depths) - 1):
            mid_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-06, data_format='channels_first'), nn.Conv1d(dims[i], dims[i + 1], kernel_size=1))
            self.downsample_layers.append(mid_layer)
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(depths)):
            stage = nn.Sequential(*[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value, kernel_size=kernel_size) for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]
        self.norm = LayerNorm(dims[-1], eps=1e-06, data_format='channels_first')
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        for i in range(len(self.downsample_layers)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x)


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


class FireflyArchitecture(nn.Module):

    def __init__(self, backbone: 'nn.Module', head: 'nn.Module', quantizer: 'nn.Module', spec_transform: 'nn.Module'):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.quantizer = quantizer
        self.spec_transform = spec_transform
        self.downsample_factor = math.prod(self.quantizer.downsample_factor)

    def forward(self, x: 'torch.Tensor', template=None, mask=None) ->torch.Tensor:
        if self.spec_transform is not None:
            x = self.spec_transform(x)
        x = self.backbone(x)
        if mask is not None:
            x = x * mask
        if self.quantizer is not None:
            vq_result = self.quantizer(x)
            x = vq_result.z
            if mask is not None:
                x = x * mask
        x = self.head(x, template=template)
        if x.ndim == 2:
            x = x[:, None, :]
        if self.vq is not None:
            return x, vq_result
        return x

    def encode(self, audios, audio_lengths):
        audios = audios.float()
        mels = self.spec_transform(audios)
        mel_lengths = audio_lengths // self.spec_transform.hop_length
        mel_masks = sequence_mask(mel_lengths, mels.shape[2])
        mel_masks_float_conv = mel_masks[:, None, :].float()
        mels = mels * mel_masks_float_conv
        encoded_features = self.backbone(mels) * mel_masks_float_conv
        feature_lengths = mel_lengths // self.downsample_factor
        return self.quantizer.encode(encoded_features), feature_lengths

    def decode(self, indices, feature_lengths) ->torch.Tensor:
        mel_masks = sequence_mask(feature_lengths * self.downsample_factor, indices.shape[2] * self.downsample_factor)
        mel_masks_float_conv = mel_masks[:, None, :].float()
        audio_lengths = feature_lengths * self.downsample_factor * self.spec_transform.hop_length
        audio_masks = sequence_mask(audio_lengths, indices.shape[2] * self.downsample_factor * self.spec_transform.hop_length)
        audio_masks_float_conv = audio_masks[:, None, :].float()
        z = self.quantizer.decode(indices) * mel_masks_float_conv
        x = self.head(z) * audio_masks_float_conv
        return x, audio_lengths

    def remove_parametrizations(self):
        if hasattr(self.backbone, 'remove_parametrizations'):
            self.backbone.remove_parametrizations()
        if hasattr(self.head, 'remove_parametrizations'):
            self.head.remove_parametrizations()

    @property
    def device(self):
        return next(self.parameters()).device


@dataclass
class FSQResult:
    z: 'torch.Tensor'
    codes: 'torch.Tensor'
    latents: 'torch.Tensor'


class DownsampleFiniteScalarQuantize(nn.Module):

    def __init__(self, input_dim: 'int'=512, n_codebooks: 'int'=9, n_groups: 'int'=1, levels: 'tuple[int]'=(8, 5, 5, 5), downsample_factor: 'tuple[int]'=(2, 2), downsample_dims: 'tuple[int] | None'=None):
        super().__init__()
        if downsample_dims is None:
            downsample_dims = [input_dim for _ in range(len(downsample_factor))]
        all_dims = (input_dim,) + tuple(downsample_dims)
        self.residual_fsq = GroupedResidualFSQ(dim=all_dims[-1], levels=levels, num_quantizers=n_codebooks, groups=n_groups)
        self.downsample_factor = downsample_factor
        self.downsample_dims = downsample_dims
        self.downsample = nn.Sequential(*[nn.Sequential(FishConvNet(all_dims[idx], all_dims[idx + 1], kernel_size=factor, stride=factor), ConvNeXtBlock(dim=all_dims[idx + 1])) for idx, factor in enumerate(downsample_factor)])
        self.upsample = nn.Sequential(*[nn.Sequential(FishTransConvNet(all_dims[idx + 1], all_dims[idx], kernel_size=factor, stride=factor), ConvNeXtBlock(dim=all_dims[idx])) for idx, factor in reversed(list(enumerate(downsample_factor)))])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, z) ->FSQResult:
        original_shape = z.shape
        z = self.downsample(z)
        quantized, indices = self.residual_fsq(z.mT)
        result = FSQResult(z=quantized.mT, codes=indices.mT, latents=z)
        result.z = self.upsample(result.z)
        diff = original_shape[-1] - result.z.shape[-1]
        left = diff // 2
        right = diff - left
        if diff > 0:
            result.z = F.pad(result.z, (left, right))
        elif diff < 0:
            result.z = result.z[..., left:-right]
        return result

    def encode(self, z):
        z = self.downsample(z)
        _, indices = self.residual_fsq(z.mT)
        indices = rearrange(indices, 'g b l r -> b (g r) l')
        return indices

    def decode(self, indices: 'torch.Tensor'):
        indices = rearrange(indices, 'b (g r) l -> g b l r', g=self.residual_fsq.groups)
        z_q = self.residual_fsq.get_output_from_indices(indices)
        z_q = self.upsample(z_q.mT)
        return z_q


class LinearSpectrogram(nn.Module):

    def __init__(self, n_fft=2048, win_length=2048, hop_length=512, center=False, mode='pow2_sqrt'):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.mode = mode
        self.register_buffer('window', torch.hann_window(win_length), persistent=False)

    def forward(self, y: 'Tensor') ->Tensor:
        if y.ndim == 3:
            y = y.squeeze(1)
        y = torch.nn.functional.pad(y.unsqueeze(1), ((self.win_length - self.hop_length) // 2, (self.win_length - self.hop_length + 1) // 2), mode='reflect').squeeze(1)
        spec = torch.stft(y, self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window, center=self.center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
        spec = torch.view_as_real(spec)
        if self.mode == 'pow2_sqrt':
            spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-06)
        return spec


class LogMelSpectrogram(nn.Module):

    def __init__(self, sample_rate=44100, n_fft=2048, win_length=2048, hop_length=512, n_mels=128, center=False, f_min=0.0, f_max=None):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or float(sample_rate // 2)
        self.spectrogram = LinearSpectrogram(n_fft, win_length, hop_length, center)
        fb = F.melscale_fbanks(n_freqs=self.n_fft // 2 + 1, f_min=self.f_min, f_max=self.f_max, n_mels=self.n_mels, sample_rate=self.sample_rate, norm='slaney', mel_scale='slaney')
        self.register_buffer('fb', fb, persistent=False)

    def compress(self, x: 'Tensor') ->Tensor:
        return torch.log(torch.clamp(x, min=1e-05))

    def decompress(self, x: 'Tensor') ->Tensor:
        return torch.exp(x)

    def apply_mel_scale(self, x: 'Tensor') ->Tensor:
        return torch.matmul(x.transpose(-1, -2), self.fb).transpose(-1, -2)

    def forward(self, x: 'Tensor', return_linear: 'bool'=False, sample_rate: 'int'=None) ->Tensor:
        if sample_rate is not None and sample_rate != self.sample_rate:
            x = F.resample(x, orig_freq=sample_rate, new_freq=self.sample_rate)
        linear = self.spectrogram(x)
        x = self.apply_mel_scale(linear)
        x = self.compress(x)
        if return_linear:
            return x, self.compress(linear)
        return x


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

    def __init__(self, in_features: 'int', out_features: 'int', bias=True, device=None, dtype=None, groupsize: 'int'=128, inner_k_tiles: 'int'=8, padding: 'bool'=True) ->None:
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
        self.register_buffer('weight', torch.empty((out_features // 8, in_features // (inner_k_tiles * 16), 32, inner_k_tiles // 2), dtype=torch.int32))
        self.register_buffer('scales_and_zeros', torch.empty((in_features // groupsize, out_features, 2), dtype=torch.bfloat16))

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        input = input
        if self.padding:
            import torch.nn.functional as F
            input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return linear_forward_int4(input, self.weight, self.scales_and_zeros, self.out_features, self.groupsize)


class IdentityMap(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {'mm_projector_type': 'identity'}


class SimpleResBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)
        self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class Denoiser(torch.nn.Module):
    """Removes model bias from audio produced with waveglow"""

    def __init__(self, vocoder, filter_length=1024, n_overlap=4, win_length=1024, mode='zeros'):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = int(filter_length / n_overlap)
        self.win_length = win_length
        dtype, device = next(vocoder.parameters()).dtype, next(vocoder.parameters()).device
        self.device = device
        if mode == 'zeros':
            mel_input = torch.zeros((1, 80, 88), dtype=dtype, device=device)
        elif mode == 'normal':
            mel_input = torch.randn((1, 80, 88), dtype=dtype, device=device)
        else:
            raise Exception(f'Mode {mode} if not supported')

        def stft_fn(audio, n_fft, hop_length, win_length, window):
            spec = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
            spec = torch.view_as_real(spec)
            return torch.sqrt(spec.pow(2).sum(-1)), torch.atan2(spec[..., -1], spec[..., 0])
        self.stft = lambda x: stft_fn(audio=x, n_fft=self.filter_length, hop_length=self.hop_length, win_length=self.win_length, window=torch.hann_window(self.win_length, device=device))
        self.istft = lambda x, y: torch.istft(torch.complex(x * torch.cos(y), x * torch.sin(y)), n_fft=self.filter_length, hop_length=self.hop_length, win_length=self.win_length, window=torch.hann_window(self.win_length, device=device))
        with torch.no_grad():
            bias_audio = vocoder(mel_input).float().squeeze(0)
            bias_spec, _ = self.stft(bias_audio)
        self.register_buffer('bias_spec', bias_spec[:, :, 0][:, :, None])

    @torch.inference_mode()
    def forward(self, audio, strength=0.0005):
        audio_spec, audio_angles = self.stft(audio)
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        audio_denoised = self.istft(audio_spec_denoised, audio_angles)
        return audio_denoised


class ResBlock2(torch.nn.Module):

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.h = h
        self.convs = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1])))])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):

    def __init__(self, h):
        super().__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(ConvTranspose1d(h.upsample_initial_channel // 2 ** i, h.upsample_initial_channel // 2 ** (i + 1), k, u, padding=(k - u) // 2)))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // 2 ** (i + 1)
            for _, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        None
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(torch.nn.Module):

    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)))])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - t % self.period
            x = F.pad(x, (0, n_pad), 'reflect')
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(2), DiscriminatorP(3), DiscriminatorP(5), DiscriminatorP(7), DiscriminatorP(11)])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for _, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):

    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([norm_f(Conv1d(1, 128, 15, 1, padding=7)), norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)), norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)), norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)), norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)), norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)), norm_f(Conv1d(1024, 1024, 5, 1, padding=2))])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorS(use_spectral_norm=True), DiscriminatorS(), DiscriminatorS()])
        self.meanpools = nn.ModuleList([AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class ResnetBlock(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(num_groups=32, dims=in_channels, eps=1e-06, affine=True, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, dims=out_channels, eps=1e-06, affine=True, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Linear(in_channels, out_channels)

    def __call__(self, x):
        h = x
        h = self.norm1(h)
        h = nn.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nn.silu(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class Upsample(nn.Module):

    def __init__(self, in_channels: 'int'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def __call__(self, x: 'mx.array'):
        x = upsample_nearest(x, (2, 2))
        x = self.conv(x)
        return x


class Decoder(nn.Module):

    def __init__(self, ch: 'int', out_ch: 'int', ch_mult: 'list[int]', num_res_blocks: 'int', in_channels: 'int', resolution: 'int', z_channels: 'int'):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = 1, z_channels, curr_res, curr_res
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = {}
        self.mid['block_1'] = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid['attn_1'] = AttnBlock(block_in)
        self.mid['block_2'] = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.up = []
        for i_level in reversed(range(self.num_resolutions)):
            block = []
            attn = []
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = {}
            up['block'] = block
            up['attn'] = attn
            if i_level != 0:
                up['upsample'] = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)
        self.norm_out = nn.GroupNorm(num_groups=32, dims=block_in, eps=1e-06, affine=True, pytorch_compatible=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def __call__(self, z: 'mx.array'):
        h = self.conv_in(z)
        h = self.mid['block_1'](h)
        h = self.mid['attn_1'](h)
        h = self.mid['block_2'](h)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level]['block'][i_block](h)
                if len(self.up[i_level]['attn']) > 0:
                    h = self.up[i_level]['attn'][i_block](h)
            if i_level != 0:
                h = self.up[i_level]['upsample'](h)
        h = self.norm_out(h)
        h = nn.silu(h)
        h = self.conv_out(h)
        return h


class BASECFM(torch.nn.Module, ABC):

    def __init__(self, n_feats, cfm_params, n_spks=1, spk_emb_dim=128):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        if hasattr(cfm_params, 'sigma_min'):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 0.0001
        self.estimator = None

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        sol = []
        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, mu, t, spks, cond)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return sol[-1]

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        z = torch.randn_like(x1)
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z
        loss = F.mse_loss(self.estimator(y, mask, mu, t.squeeze(), spks), u, reduction='sum') / (torch.sum(mask) * u.shape[1])
        return loss, y


class CFM(BASECFM):

    def __init__(self, in_channels, out_channel, cfm_params, decoder_params, n_spks=1, spk_emb_dim=64):
        super().__init__(n_feats=in_channels, cfm_params=cfm_params, n_spks=n_spks, spk_emb_dim=spk_emb_dim)
        in_channels = in_channels + (spk_emb_dim if n_spks > 1 else 0)
        self.estimator = Decoder(in_channels=in_channels, out_channels=out_channel, **decoder_params)


class ConvReluNorm(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        self.conv_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.conv_layers.append(torch.nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask


class DurationPredictor(nn.Module):

    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.p_dropout = p_dropout
        self.drop = torch.nn.Dropout(p_dropout)
        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = torch.nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = torch.nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class RotaryPositionalEmbeddings(nn.Module):
    """
    ## RoPE module

    Rotary encoding transforms pairs of features by rotating in the 2D plane.
    That is, it organizes the $d$ features as $rac{d}{2}$ pairs.
    Each pair can be considered a coordinate in a 2D plane, and the encoding will rotate it
    by an angle depending on the position of the token.
    """

    def __init__(self, d: 'int', base: 'int'=10000):
        """
        * `d` is the number of features $d$
        * `base` is the constant used for calculating $\\Theta$
        """
        super().__init__()
        self.base = base
        self.d = int(d)
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: 'torch.Tensor'):
        """
        Cache $\\cos$ and $\\sin$ values
        """
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return
        seq_len = x.shape[0]
        theta = 1.0 / self.base ** (torch.arange(0, self.d, 2).float() / self.d)
        seq_idx = torch.arange(seq_len, device=x.device).float()
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]

    def _neg_half(self, x: 'torch.Tensor'):
        d_2 = self.d // 2
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x: 'torch.Tensor'):
        """
        * `x` is the Tensor at the head of a key or a query with shape `[seq_len, batch_size, n_heads, d]`
        """
        x = rearrange(x, 'b h t d -> t b h d')
        self._build_cache(x)
        x_rope, x_pass = x[..., :self.d], x[..., self.d:]
        neg_half_x = self._neg_half(x_rope)
        x_rope = x_rope * self.cos_cached[:x.shape[0]] + neg_half_x * self.sin_cached[:x.shape[0]]
        return rearrange(torch.cat((x_rope, x_pass), dim=-1), 't b h d -> b h t d')


class FFN(nn.Module):

    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = torch.nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.drop = torch.nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class Downsample(nn.Module):

    def __init__(self, in_channels: 'int'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def __call__(self, x: 'mx.array'):
        x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        x = self.conv(x)
        return x


class Encoder(nn.Module):

    def __init__(self, resolution: 'int', in_channels: 'int', ch: 'int', ch_mult: 'list[int]', num_res_blocks: 'int', z_channels: 'int'):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = []
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = []
            attn = []
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = {}
            down['block'] = block
            down['attn'] = attn
            if i_level != self.num_resolutions - 1:
                down['downsample'] = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)
        self.mid = {}
        self.mid['block_1'] = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid['attn_1'] = AttnBlock(block_in)
        self.mid['block_2'] = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.norm_out = nn.GroupNorm(num_groups=32, dims=block_in, eps=1e-06, affine=True, pytorch_compatible=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def __call__(self, x: 'mx.array'):
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level]['block'][i_block](hs[-1])
                if len(self.down[i_level]['attn']) > 0:
                    h = self.down[i_level]['attn'][i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level]['downsample'](hs[-1]))
        h = hs[-1]
        h = self.mid['block_1'](h)
        h = self.mid['attn_1'](h)
        h = self.mid['block_2'](h)
        h = self.norm_out(h)
        h = nn.silu(h)
        h = self.conv_out(h)
        return h


class TextEncoder(nn.Module):

    def __init__(self, encoder_type, encoder_params, duration_predictor_params, n_vocab, n_spks=1, spk_emb_dim=128):
        super().__init__()
        self.encoder_type = encoder_type
        self.n_vocab = n_vocab
        self.n_feats = encoder_params.n_feats
        self.n_channels = encoder_params.n_channels
        self.spk_emb_dim = spk_emb_dim
        self.n_spks = n_spks
        self.emb = torch.nn.Embedding(n_vocab, self.n_channels)
        torch.nn.init.normal_(self.emb.weight, 0.0, self.n_channels ** -0.5)
        if encoder_params.prenet:
            self.prenet = ConvReluNorm(self.n_channels, self.n_channels, self.n_channels, kernel_size=5, n_layers=3, p_dropout=0.5)
        else:
            self.prenet = lambda x, x_mask: x
        self.encoder = Encoder(encoder_params.n_channels + (spk_emb_dim if n_spks > 1 else 0), encoder_params.filter_channels, encoder_params.n_heads, encoder_params.n_layers, encoder_params.kernel_size, encoder_params.p_dropout)
        self.proj_m = torch.nn.Conv1d(self.n_channels + (spk_emb_dim if n_spks > 1 else 0), self.n_feats, 1)
        self.proj_w = DurationPredictor(self.n_channels + (spk_emb_dim if n_spks > 1 else 0), duration_predictor_params.filter_channels_dp, duration_predictor_params.kernel_size, duration_predictor_params.p_dropout)

    def forward(self, x, x_lengths, spks=None):
        """Run forward pass to the transformer based encoder and duration predictor

        Args:
            x (torch.Tensor): text input
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): text input lengths
                shape: (batch_size,)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size,)

        Returns:
            mu (torch.Tensor): average output of the encoder
                shape: (batch_size, n_feats, max_text_length)
            logw (torch.Tensor): log duration predicted by the duration predictor
                shape: (batch_size, 1, max_text_length)
            x_mask (torch.Tensor): mask for the text input
                shape: (batch_size, 1, max_text_length)
        """
        x = self.emb(x) * math.sqrt(self.n_channels)
        x = torch.transpose(x, 1, -1)
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1)
        x = self.prenet(x, x_mask)
        if self.n_spks > 1:
            x = torch.cat([x, spks.unsqueeze(-1).repeat(1, 1, x.shape[-1])], dim=1)
        x = self.encoder(x, x_mask)
        mu = self.proj_m(x) * x_mask
        x_dp = torch.detach(x)
        logw = self.proj_w(x_dp, x_mask)
        return mu, logw, x_mask


class DiagonalGaussian(nn.Module):

    def __call__(self, z: 'mx.array'):
        mean, logvar = mx.split(z, 2, axis=-1)
        if self.training:
            std = mx.exp(0.5 * logvar)
            eps = mx.random.normal(shape=z.shape, dtype=z.dtype)
            return mean + std * eps
        else:
            return mean


class AutoEncoder(nn.Module):

    def __init__(self, params: 'AutoEncoderParams'):
        super().__init__()
        self.encoder = Encoder(resolution=params.resolution, in_channels=params.in_channels, ch=params.ch, ch_mult=params.ch_mult, num_res_blocks=params.num_res_blocks, z_channels=params.z_channels)
        self.decoder = Decoder(resolution=params.resolution, in_channels=params.in_channels, ch=params.ch, out_ch=params.out_ch, ch_mult=params.ch_mult, num_res_blocks=params.num_res_blocks, z_channels=params.z_channels)
        self.reg = DiagonalGaussian()
        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    def sanitize(self, weights):
        new_weights = {}
        for k, w in weights.items():
            if w.ndim == 4:
                w = w.transpose(0, 2, 3, 1)
                w = w.reshape(-1).reshape(w.shape)
                if w.shape[1:3] == (1, 1):
                    w = w.squeeze((1, 2))
            new_weights[k] = w
        return new_weights

    def encode(self, x: 'mx.array'):
        z = self.reg(self.encoder(x))
        z = self.scale_factor * (z - self.shift_factor)
        return z

    def decode(self, z: 'mx.array'):
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)

    def __call__(self, x: 'mx.array'):
        return self.decode(self.encode(x))


class CLIPEncoderLayer(nn.Module):
    """The transformer encoder layer from CLIP."""

    def __init__(self, model_dims: 'int', num_heads: 'int', activation: 'str'):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(model_dims)
        self.layer_norm2 = nn.LayerNorm(model_dims)
        self.attention = nn.MultiHeadAttention(model_dims, num_heads, bias=True)
        self.linear1 = nn.Linear(model_dims, 4 * model_dims)
        self.linear2 = nn.Linear(4 * model_dims, model_dims)
        self.act = _ACTIVATIONS[activation]

    def __call__(self, x, attn_mask=None):
        y = self.layer_norm1(x)
        y = self.attention(y, y, y, attn_mask)
        x = y + x
        y = self.layer_norm2(x)
        y = self.linear1(y)
        y = self.act(y)
        y = self.linear2(y)
        x = y + x
        return x


class CLIPTextModel(nn.Module):
    """Implements the text encoder transformer from CLIP."""

    def __init__(self, config: 'CLIPTextModelConfig'):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.model_dims)
        self.position_embedding = nn.Embedding(config.max_length, config.model_dims)
        self.layers = [CLIPEncoderLayer(config.model_dims, config.num_heads, config.hidden_act) for i in range(config.num_layers)]
        self.final_layer_norm = nn.LayerNorm(config.model_dims)

    def _get_mask(self, N, dtype):
        indices = mx.arange(N)
        mask = indices[:, None] < indices[None]
        mask = mask.astype(dtype) * (-60000.0 if dtype == mx.float16 else -1000000000.0)
        return mask

    def sanitize(self, weights):
        new_weights = {}
        for key, w in weights.items():
            if key.startswith('text_model.'):
                key = key[11:]
            if key.startswith('embeddings.'):
                key = key[11:]
            if key.startswith('encoder.'):
                key = key[8:]
            if 'self_attn.' in key:
                key = key.replace('self_attn.', 'attention.')
            if 'q_proj.' in key:
                key = key.replace('q_proj.', 'query_proj.')
            if 'k_proj.' in key:
                key = key.replace('k_proj.', 'key_proj.')
            if 'v_proj.' in key:
                key = key.replace('v_proj.', 'value_proj.')
            if 'mlp.fc1' in key:
                key = key.replace('mlp.fc1', 'linear1')
            if 'mlp.fc2' in key:
                key = key.replace('mlp.fc2', 'linear2')
            new_weights[key] = w
        return new_weights

    def __call__(self, x):
        B, N = x.shape
        eos_tokens = x.argmax(-1)
        x = self.token_embedding(x)
        x = x + self.position_embedding.weight[:N]
        mask = self._get_mask(N, x.dtype)
        hidden_states = []
        for l in self.layers:
            x = l(x, mask)
            hidden_states.append(x)
        x = self.final_layer_norm(x)
        last_hidden_state = x
        pooled_output = x[mx.arange(len(x)), eos_tokens]
        return CLIPOutput(pooled_output=pooled_output, last_hidden_state=last_hidden_state, hidden_states=hidden_states)


def _rope(pos: 'mx.array', dim: 'int', theta: 'float'):
    scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
    omega = 1.0 / theta ** scale
    x = pos[..., None] * omega
    cosx = mx.cos(x)
    sinx = mx.sin(x)
    pe = mx.stack([cosx, -sinx, sinx, cosx], axis=-1)
    pe = pe.reshape(*pe.shape[:-1], 2, 2)
    return pe


class EmbedND(nn.Module):

    def __init__(self, dim: 'int', theta: 'int', axes_dim: 'List[int]'):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def __call__(self, ids: 'mx.array'):
        n_axes = ids.shape[-1]
        pe = mx.concatenate([_rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], axis=-3)
        return pe[:, None]


def _apply_rope(x, pe):
    s = x.shape
    x = x.reshape(*s[:-1], -1, 1, 2)
    x = _ab_plus_cd(x[..., 0], pe[..., 0], x[..., 1], pe[..., 1])
    return x.reshape(s)


def _attention(q: 'mx.array', k: 'mx.array', v: 'mx.array', pe: 'mx.array'):
    B, H, L, D = q.shape
    q = _apply_rope(q, pe)
    k = _apply_rope(k, pe)
    x = mx.fast.scaled_dot_product_attention(q, k, v, scale=D ** -0.5)
    return x.transpose(0, 2, 1, 3).reshape(B, L, -1)


@dataclass
class ModulationOut:
    shift: 'mx.array'
    scale: 'mx.array'
    gate: 'mx.array'


class Modulation(nn.Module):

    def __init__(self, dim: 'int', double: 'bool'):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def __call__(self, x: 'mx.array') ->Tuple[ModulationOut, Optional[ModulationOut]]:
        x = self.lin(nn.silu(x))
        xs = mx.split(x[:, None, :], self.multiplier, axis=-1)
        mod1 = ModulationOut(*xs[:3])
        mod2 = ModulationOut(*xs[3:]) if self.is_double else None
        return mod1, mod2


class SingleStreamBlock(nn.Module):

    def __init__(self, hidden_size: 'int', num_heads: 'int', mlp_ratio: 'float'=4.0, qk_scale: 'Optional[float]'=None):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
        self.norm = QKNorm(head_dim)
        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, affine=False, eps=1e-06)
        self.mlp_act = nn.GELU(approx='tanh')
        self.modulation = Modulation(hidden_size, double=False)

    def __call__(self, x: 'mx.array', vec: 'mx.array', pe: 'mx.array'):
        B, L, _ = x.shape
        H = self.num_heads
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        q, k, v, mlp = mx.split(self.linear1(x_mod), [self.hidden_size, 2 * self.hidden_size, 3 * self.hidden_size], axis=-1)
        q = q.reshape(B, L, H, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, H, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, H, -1).transpose(0, 2, 1, 3)
        q, k = self.norm(q, k)
        y = _attention(q, k, v, pe)
        y = self.linear2(mx.concatenate([y, self.mlp_act(mlp)], axis=2))
        return x + mod.gate * y


class LastLayer(nn.Module):

    def __init__(self, hidden_size: 'int', patch_size: 'int', out_channels: 'int'):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, affine=False, eps=1e-06)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def __call__(self, x: 'mx.array', vec: 'mx.array'):
        shift, scale = mx.split(self.adaLN_modulation(vec), 2, axis=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


class LoRALinear(nn.Module):

    @staticmethod
    def from_base(linear: 'nn.Linear', r: 'int'=8, dropout: 'float'=0.0, scale: 'float'=1.0):
        output_dims, input_dims = linear.weight.shape
        lora_lin = LoRALinear(input_dims=input_dims, output_dims=output_dims, r=r, dropout=dropout, scale=scale)
        lora_lin.linear = linear
        return lora_lin

    def fuse(self):
        linear = self.linear
        bias = 'bias' in linear
        weight = linear.weight
        dtype = weight.dtype
        output_dims, input_dims = weight.shape
        fused_linear = nn.Linear(input_dims, output_dims, bias=bias)
        lora_b = self.scale * self.lora_b.T
        lora_a = self.lora_a.T
        fused_linear.weight = weight + (lora_b @ lora_a).astype(dtype)
        if bias:
            fused_linear.bias = linear.bias
        return fused_linear

    def __init__(self, input_dims: 'int', output_dims: 'int', r: 'int'=8, dropout: 'float'=0.0, scale: 'float'=1.0, bias: 'bool'=False):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale
        scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(low=-scale, high=scale, shape=(input_dims, r))
        self.lora_b = mx.zeros(shape=(r, output_dims))

    def __call__(self, x):
        y = self.linear(x)
        z = self.dropout(x) @ self.lora_a @ self.lora_b
        return y + (self.scale * z).astype(x.dtype)


def timestep_embedding(t: 'mx.array', dim: 'int', max_period: 'int'=10000, time_factor: 'float'=1000.0):
    half = dim // 2
    freqs = mx.arange(0, half, dtype=mx.float32) / half
    freqs = freqs * -math.log(max_period)
    freqs = mx.exp(freqs)
    x = (time_factor * t)[:, None] * freqs[None]
    x = mx.concatenate([mx.cos(x), mx.sin(x)], axis=-1)
    return x.astype(t.dtype)


_ENCODER_REPLACEMENT_PATTERNS = [('.layer.0.SelfAttention.', '.attention.'), ('.layer.1.DenseReluDense.', '.dense.')]


_SHARED_REPLACEMENT_PATTERNS = [('.block.', '.layers.'), ('.k.', '.key_proj.'), ('.o.', '.out_proj.'), ('.q.', '.query_proj.'), ('.v.', '.value_proj.'), ('shared.', 'wte.'), ('lm_head.', 'lm_head.linear.'), ('.layer.0.layer_norm.', '.ln1.'), ('.layer.1.layer_norm.', '.ln2.'), ('.layer.2.layer_norm.', '.ln3.'), ('.final_layer_norm.', '.ln.'), ('layers.0.layer.0.SelfAttention.relative_attention_bias.', 'relative_attention_bias.embeddings.')]


class T5Encoder(nn.Module):

    def __init__(self, config: 'T5Config'):
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = TransformerEncoder(config)

    def sanitize(self, weights):
        new_weights = {}
        for k, w in weights.items():
            for old, new in _SHARED_REPLACEMENT_PATTERNS:
                k = k.replace(old, new)
            if k.startswith('encoder.'):
                for old, new in _ENCODER_REPLACEMENT_PATTERNS:
                    k = k.replace(old, new)
            new_weights[k] = w
        return new_weights

    def __call__(self, inputs: 'mx.array'):
        return self.encoder(self.wte(inputs))


class Identity(torch.nn.Identity):

    def forward(self, input: 'Tensor', **kwargs) ->Tensor:
        return super().forward(input)


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


def get_abs_pos(abs_pos, tgt_size):
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype
    if src_size != tgt_size:
        return F.interpolate(abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2), size=(tgt_size, tgt_size), mode='bicubic', align_corners=False).permute(0, 2, 3, 1).flatten(0, 2)
    else:
        return abs_pos


class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(self, grid_size, embed_dim, num_heads, kv_dim=None, norm_layer=partial(nn.LayerNorm, eps=1e-06)):
        super().__init__()
        self.num_queries = grid_size ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.pos_embed = nn.Parameter(torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, grid_size)).float()).requires_grad_(False)
        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query, std=0.02)
        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)
        self.ln_post = norm_layer(embed_dim)
        self.proj = nn.Parameter(embed_dim ** -0.5 * torch.randn(embed_dim, embed_dim))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, attn_mask=None):
        pos_embed = get_abs_pos(self.pos_embed, x.size(1))
        x = self.kv_proj(x)
        x = self.ln_kv(x).permute(1, 0, 2)
        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(self._repeat(q, N) + self.pos_embed.unsqueeze(1), x + pos_embed.unsqueeze(1), x, attn_mask=attn_mask)[0]
        x = out.permute(1, 0, 2)
        x = self.ln_post(x)
        x = x @ self.proj
        return x

    def _repeat(self, query, N: 'int'):
        return query.unsqueeze(1).repeat(1, N, 1)


class Conv1d(nn.Conv1d):

    def _conv_forward(self, x: 'Tensor', weight: 'Tensor', bias: 'Optional[Tensor]') ->Tensor:
        return super()._conv_forward(x, weight, None if bias is None else bias)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, n_state: 'int', n_head: 'int', cross_attention: 'bool'=False):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)
        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None
        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(self, x: 'Tensor', xa: 'Optional[Tensor]'=None, mask: 'Optional[Tensor]'=None, kv_cache: 'Optional[dict]'=None):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class AudioEncoder(nn.Module):

    def __init__(self, n_mels: 'int', n_ctx: 'int', n_state: 'int', n_head: 'int', n_layer: 'int'):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer('positional_embedding', sinusoids(n_ctx, n_state))
        self.blocks: 'Iterable[ResidualAttentionBlock]' = nn.ModuleList([ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)])
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: 'Tensor'):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        assert x.shape[1:] == self.positional_embedding.shape, 'incorrect audio shape'
        x = x + self.positional_embedding
        for block in self.blocks:
            x = block(x)
        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):

    def __init__(self, n_vocab: 'int', n_ctx: 'int', n_state: 'int', n_head: 'int', n_layer: 'int'):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
        self.blocks: 'Iterable[ResidualAttentionBlock]' = nn.ModuleList([ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)])
        self.ln = LayerNorm(n_state)
        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer('mask', mask, persistent=False)

    def forward(self, x: 'Tensor', xa: 'Tensor', kv_cache: 'Optional[dict]'=None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = self.token_embedding(x) + self.positional_embedding[offset:offset + x.shape[-1]]
        x = x
        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)
        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight, 0, 1)).float()
        return logits


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Conv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (ConvNeXtBlock,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (ConvolutionModule,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (DiscriminatorP,
     lambda: ([], {'period': 4}),
     lambda: ([torch.rand([4, 1, 4])], {})),
    (DiscriminatorS,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64])], {})),
    (Downsample1D,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (EspnetRelPositionalEncoding,
     lambda: ([], {'d_model': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FishConvNet,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (FishTransConvNet,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (IdentityMap,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LabelSmoothingLoss,
     lambda: ([], {'size': 4, 'padding_idx': 4, 'smoothing': 4}),
     lambda: ([torch.rand([64, 4, 4, 4]), torch.ones([1024], dtype=torch.int64)], {})),
    (LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm2d,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerScale,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LearnablePositionalEncoding,
     lambda: ([], {'d_model': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLPBlock,
     lambda: ([], {'embedding_dim': 4, 'mlp_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MoEFFNLayer,
     lambda: ([], {'n_expert': 4, 'n_expert_per_token': 4, 'idim': 4, 'hidden_units': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (MultiHeadAttention,
     lambda: ([], {'n_state': 4, 'n_head': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (MultiHeadedAttention,
     lambda: ([], {'n_head': 4, 'n_feat': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MultiScaleDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64]), torch.rand([4, 1, 64])], {})),
    (NoPositionalEncoding,
     lambda: ([], {'d_model': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PatchEmbed,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PositionwiseFeedForward,
     lambda: ([], {'idim': 4, 'hidden_units': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RelPositionalEncoding,
     lambda: ([], {'d_model': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 4])], {})),
    (ResBlock1,
     lambda: ([], {'h': 4, 'channels': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (ResBlock2,
     lambda: ([], {'h': 4, 'channels': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (Resampler,
     lambda: ([], {'grid_size': 4, 'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (ResidualAttentionBlock,
     lambda: ([], {'n_state': 4, 'n_head': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (SimpleResBlock,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Snake,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Upsample1D,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (WhisperPositionalEncoding,
     lambda: ([], {'d_model': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

