
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


from torchvision import datasets


from torchvision import transforms


from torchvision.transforms.functional import InterpolationMode


from torch.utils.data import Dataset


import numpy as np


import random


import math


from collections import OrderedDict


from typing import Optional


from typing import Tuple


from typing import Union


import torch.nn.functional as F


from torch import nn


import warnings


from torch import Tensor


from torch.nn.init import constant_


from torch.nn.init import xavier_normal_


from torch.nn.init import xavier_uniform_


from torch.nn.parameter import Parameter


from torch.nn.modules import Module


from torch.nn.modules.linear import NonDynamicallyQuantizableLinear


from torch.overrides import has_torch_function


from torch.overrides import handle_torch_function


from torch.nn.functional import _in_projection_packed


from torch.nn.functional import linear


from copy import deepcopy


import torch.nn as nn


from typing import Iterable


from typing import Any


from typing import List


from torch import optim as optim


import time


import torch.backends.cudnn as cudnn


from collections import defaultdict


from collections import deque


import torch.distributed as dist


from sklearn import metrics


def multi_head_attention_forward(query: 'Tensor', key: 'Tensor', value: 'Tensor', embed_dim_to_check: 'int', num_heads: 'int', in_proj_weight: 'Tensor', in_proj_bias: 'Optional[Tensor]', bias_k: 'Optional[Tensor]', bias_v: 'Optional[Tensor]', add_zero_attn: 'bool', dropout_p: 'float', out_proj_weight: 'Tensor', out_proj_bias: 'Optional[Tensor]', training: 'bool'=True, key_padding_mask: 'Optional[Tensor]'=None, need_weights: 'bool'=True, attn_mask: 'Optional[Tensor]'=None, use_separate_proj_weight: 'bool'=False, q_proj_weight: 'Optional[Tensor]'=None, k_proj_weight: 'Optional[Tensor]'=None, v_proj_weight: 'Optional[Tensor]'=None, static_k: 'Optional[Tensor]'=None, static_v: 'Optional[Tensor]'=None) ->Tuple[Tensor, Optional[Tensor]]:
    tens_ops = query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias
    if has_torch_function(tens_ops):
        return handle_torch_function(multi_head_attention_forward, tens_ops, query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias, training=training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, use_separate_proj_weight=use_separate_proj_weight, q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight, v_proj_weight=v_proj_weight, static_k=static_k, static_v=static_v)
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
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        assert static_k.size(0) == bsz * num_heads, f'expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}'
        assert static_k.size(2) == head_dim, f'expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}'
        k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
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
    attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    if need_weights:
        return attn_output, attn_output_weights
    else:
        return attn_output, None


class MultiheadAttention(Module):
    __constants__ = ['batch_first']
    bias_k: 'Optional[torch.Tensor]'
    bias_v: 'Optional[torch.Tensor]'

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None) ->None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)
        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None
        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True
        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query: 'Tensor', key: 'Tensor', value: 'Tensor', key_padding_mask: 'Optional[Tensor]'=None, need_weights: 'bool'=True, attn_mask: 'Optional[Tensor]'=None) ->Tuple[Tensor, Optional[Tensor]]:
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, use_separate_proj_weight=True, q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight, v_proj_weight=self.v_proj_weight)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([('-1', nn.AvgPool2d(stride)), ('0', nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)), ('1', nn.BatchNorm2d(planes * self.expansion))]))

    def forward(self, x: 'torch.Tensor'):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):

    def __init__(self, spacial_dim: 'int', embed_dim: 'int', num_heads: 'int', output_dim: 'int'=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(query=x, key=x, value=x, embed_dim_to_check=x.shape[-1], num_heads=self.num_heads, q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight, in_proj_weight=None, in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]), bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0, out_proj_weight=self.c_proj.weight, out_proj_bias=self.c_proj.bias, use_separate_proj_weight=True, training=self.training, need_weights=False)
        return x[0]


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: 'torch.Tensor'):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):

    def forward(self, x: 'torch.Tensor'):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: 'int', n_head: 'int', attn_mask: 'torch.Tensor'=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)), ('gelu', QuickGELU()), ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: 'torch.Tensor', need_weights=False, attn_mask: 'torch.Tensor'=None):
        attn_mask = self.attn_mask if self.attn_mask is not None else attn_mask
        return self.attn(x, x, x, need_weights=need_weights, attn_mask=attn_mask)

    def save_attention_map(self, attn):
        self.attn_map = attn

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def forward(self, x: 'torch.Tensor', save_attn=False, attn_mask: 'torch.Tensor'=None):
        attn_output, attn_output_weights = self.attention(self.ln_1(x), need_weights=save_attn, attn_mask=attn_mask)
        x = x + attn_output
        x_ffn = self.mlp(self.ln_2(x))
        x = x + x_ffn
        return x


class Transformer(nn.Module):

    def __init__(self, width: 'int', layers: 'int', heads: 'int', attn_mask: 'torch.Tensor'=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: 'torch.Tensor', attn_mask=None):
        for i, block in enumerate(self.resblocks):
            if i == self.layers - 1:
                x = block(x, save_attn=True, attn_mask=attn_mask)
            else:
                x = block(x, attn_mask=attn_mask)
        return x


class VisionTransformer_MIM(nn.Module):

    def __init__(self, input_resolution: 'int', patch_size: 'int', width: 'int', layers: 'int', heads: 'int', output_dim: 'int', mask: 'bool'):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.heads = heads
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.has_decoder = mask
        if self.has_decoder:
            self.mask_token_embedding = nn.Parameter(torch.zeros(1, 1, width))
            self.ln_patch = LayerNorm(width)
            nn.init.normal_(self.mask_token_embedding, std=0.02)
            self.decoder = nn.Sequential(nn.Conv2d(in_channels=width, out_channels=self.patch_size ** 2 * 3, kernel_size=1), nn.PixelShuffle(self.patch_size))

    def forward(self, x: 'torch.Tensor', mask=None):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        if mask is not None:
            assert self.has_decoder, 'model does not have decoder'
            mask_token_embedding = self.mask_token_embedding.expand(x.shape[0], x.shape[1], -1)
            w = mask.flatten(1).unsqueeze(-1).type_as(mask_token_embedding)
            x = x * (1 - w) + mask_token_embedding * w
        x = torch.cat([self.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        if mask is not None:
            x_cls = self.ln_post(x[:, 0, :])
            x_cls = F.normalize(x_cls @ self.proj, dim=-1)
            logits = 100 * x_cls @ self.classifier
            x_patch = self.ln_patch(x[:, 1:, :])
            B, L, C = x_patch.shape
            H = W = int(L ** 0.5)
            x_patch_reshape = x_patch.permute(0, 2, 1).reshape(B, C, H, W)
            x_rec = self.decoder(x_patch_reshape)
            x_mask = F.normalize(x_patch @ self.proj, dim=-1)
            loss_align = torch.sum((x_mask - x_cls.unsqueeze(1)).pow(2), dim=-1, keepdim=True)
            loss_align = loss_align[w.bool()].view(w.size(0), -1)
            mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
            return logits, x_rec, loss_align, mask
        else:
            x_cls = self.ln_post(x[:, 0, :])
            x_proj = F.normalize(x_cls @ self.proj, dim=-1)
            logits = 100 * x_proj @ self.classifier
            return logits


class CLIP(nn.Module):

    def __init__(self, embed_dim: 'int', image_resolution: 'int', vision_layers: 'Union[Tuple[int, int, int, int], int]', vision_width: 'int', vision_patch_size: 'int', mask: 'bool', context_length: 'int', vocab_size: 'int', transformer_width: 'int', transformer_heads: 'int', transformer_layers: 'int'):
        super().__init__()
        self.context_length = context_length
        vision_heads = vision_width // 64
        self.visual = VisionTransformer_MIM(input_resolution=image_resolution, patch_size=vision_patch_size, width=vision_width, layers=vision_layers, heads=vision_heads, output_dim=embed_dim, mask=mask)
        self.transformer = Transformer(width=transformer_width, layers=transformer_layers, heads=transformer_heads, attn_mask=self.build_attention_mask())
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        proj_std = self.transformer.width ** -0.5 * (2 * self.transformer.layers) ** -0.5
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x


_MODELS = {'ViT-B/32': 'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt', 'ViT-B/16': 'https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt', 'ViT-L/14': 'https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt'}


def _download(url: 'str', root: 'str'):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    expected_sha256 = url.split('/')[-2]
    download_target = os.path.join(root, filename)
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f'{download_target} exists and is not a regular file')
    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, 'rb').read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f'{download_target} exists, but the SHA256 checksum does not match; re-downloading the file')
    with urllib.request.urlopen(url) as source, open(download_target, 'wb') as output:
        with tqdm(total=int(source.info().get('Content-Length')), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))
    if hashlib.sha256(open(download_target, 'rb').read()).hexdigest() != expected_sha256:
        raise RuntimeError(f'Model has been downloaded but the SHA256 checksum does not not match')
    return download_target


def available_models() ->List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def build_model(state_dict: 'dict', mask: 'bool'=False):
    vision_width = state_dict['visual.conv1.weight'].shape[0]
    vision_layers = len([k for k in state_dict.keys() if k.startswith('visual.') and k.endswith('.attn.in_proj_weight')])
    vision_patch_size = state_dict['visual.conv1.weight'].shape[-1]
    grid_size = round((state_dict['visual.positional_embedding'].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    embed_dim = state_dict['text_projection'].shape[1]
    context_length = state_dict['positional_embedding'].shape[0]
    vocab_size = state_dict['token_embedding.weight'].shape[0]
    transformer_width = state_dict['ln_final.weight'].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split('.')[2] for k in state_dict if k.startswith(f'transformer.resblocks')))
    model = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size, mask, context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)
    for key in ['input_resolution', 'context_length', 'vocab_size']:
        if key in state_dict:
            del state_dict[key]
    msg = model.load_state_dict(state_dict, strict=False)
    None
    None
    return model


def load(name: 'str', device: 'Union[str, torch.device]'='cuda' if torch.cuda.is_available() else 'cpu', jit: 'bool'=False, download_root: 'str'=None, mask: 'bool'=False):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser('~/.cache/clip'))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f'Model {name} not found; available models = {available_models()}')
    try:
        model = torch.jit.load(model_path, map_location=device if jit else 'cpu').eval()
        state_dict = None
    except RuntimeError:
        if jit:
            warnings.warn(f'File {model_path} is not a JIT archive. Loading as a state dict instead')
            jit = False
        state_dict = torch.load(model_path, map_location='cpu')
    if not jit:
        model = build_model(state_dict or model.state_dict(), mask=mask)
        if str(device) == 'cpu':
            model.float()
        return model
    device_holder = torch.jit.trace(lambda : torch.ones([]), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes('prim::Constant') if 'Device' in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, 'graph') else []
        except RuntimeError:
            graphs = []
        if hasattr(module, 'forward1'):
            graphs.append(module.forward1.graph)
        for graph in graphs:
            for node in graph.findAllNodes('prim::Constant'):
                if 'value' in node.attributeNames() and str(node['value']).startswith('cuda'):
                    node.copyAttributes(device_node)
    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)
    if str(device) == 'cpu':
        float_holder = torch.jit.trace(lambda : torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode('aten::to').inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, 'graph') else []
            except RuntimeError:
                graphs = []
            if hasattr(module, 'forward1'):
                graphs.append(module.forward1.graph)
            for graph in graphs:
                for node in graph.findAllNodes('aten::to'):
                    inputs = list(node.inputs())
                    for i in [1, 2]:
                        if inputs[i].node()['value'] == 5:
                            inputs[i].node().copyAttributes(float_node)
        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)
        model.float()
    return model


def tokenize(texts: 'Union[str, List[str]]', context_length: 'int'=77, truncate: 'bool'=False) ->torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]
    sot_token = _tokenizer.encoder['<|startoftext|>']
    eot_token = _tokenizer.encoder['<|endoftext|>']
    all_tokens = [([sot_token] + _tokenizer.encode(text) + [eot_token]) for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f'Input {texts[i]} is too long for context length {context_length}')
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result


class clip_classifier(nn.Module):

    def __init__(self, args, is_train=False):
        super().__init__()
        self.templates = json.load(open(args.template, 'r'))
        self.templates = self.templates[args.dataset]
        classnames = json.load(open(args.classname, 'r'))
        self.classnames = classnames[args.dataset]
        self.model = load(args.clip_model, jit=False, mask=args.mask)
        self.model.float()
        self.init_classifier_weights(args)
        self.model = self.model

    def init_classifier_weights(self, args):
        None
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(self.classnames):
                texts = [template.format(classname) for template in self.templates]
                texts = tokenize(texts)
                class_embeddings = self.model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
        self.model.visual.classifier = nn.Parameter(torch.stack(zeroshot_weights, dim=1))
        del self.model.transformer, self.model.token_embedding, self.model.positional_embedding, self.model.ln_final, self.model.text_projection, self.model.logit_scale
        return

    def forward(self, images, **kwargs):
        return self.model.visual(images, **kwargs)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AttentionPool2d,
     lambda: ([], {'spacial_dim': 4, 'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Bottleneck,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (QuickGELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualAttentionBlock,
     lambda: ([], {'d_model': 4, 'n_head': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (Transformer,
     lambda: ([], {'width': 4, 'layers': 1, 'heads': 4}),
     lambda: ([torch.rand([4, 4])], {})),
]

