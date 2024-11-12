
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


from typing import Any


from typing import Union


from typing import List


import torch


from torchvision.transforms import Compose


from torchvision.transforms import Resize


from torchvision.transforms import CenterCrop


from torchvision.transforms import ToTensor


from torchvision.transforms import Normalize


from collections import OrderedDict


from typing import Tuple


import numpy as np


import torch.nn.functional as F


from torch import nn


import math


from typing import Optional


from torch import Tensor


from torch.nn.init import xavier_uniform_


from torch.nn.init import constant_


from torch.nn.init import xavier_normal_


from torch.nn.parameter import Parameter


from torch.nn import functional as F


from torch.utils.data import Dataset


from torchvision import transforms


import random


from torchvision.transforms.functional import crop


import torch.nn as nn


import torchvision.transforms as T


from torchvision.transforms import InterpolationMode


from torchvision import transforms as T


from torch.optim import lr_scheduler


import torchvision


import torch.nn


import scipy.interpolate


from collections.abc import Sequence


import numbers


from torchvision.transforms.functional import InterpolationMode


from torchvision.transforms.functional import _interpolation_modes_from_int


from torchvision.transforms.functional import get_image_num_channels


from torchvision.transforms.functional import get_image_size


from torchvision.transforms.functional import perspective


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


def multi_head_attention_forward(query: 'Tensor', key: 'Tensor', value: 'Tensor', embed_dim_to_check: 'int', num_heads: 'int', in_proj_weight: 'Tensor', in_proj_bias: 'Tensor', bias_k: 'Optional[Tensor]', bias_v: 'Optional[Tensor]', add_zero_attn: 'bool', dropout_p: 'float', out_proj_weight: 'Tensor', out_proj_bias: 'Tensor', training: 'bool'=True, key_padding_mask: 'Optional[Tensor]'=None, need_weights: 'bool'=True, attn_mask: 'Optional[Tensor]'=None, use_separate_proj_weight: 'bool'=False, q_proj_weight: 'Optional[Tensor]'=None, k_proj_weight: 'Optional[Tensor]'=None, v_proj_weight: 'Optional[Tensor]'=None, static_k: 'Optional[Tensor]'=None, static_v: 'Optional[Tensor]'=None, attention_probs_forward_hook=None, attention_probs_backwards_hook=None) ->Tuple[Tensor, Optional[Tensor]]:
    if not torch.jit.is_scripting():
        tens_ops = query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias
        if any([(type(t) is not Tensor) for t in tens_ops]) and F.has_torch_function(tens_ops):
            return F.handle_torch_function(multi_head_attention_forward, tens_ops, query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias, training=training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, use_separate_proj_weight=use_separate_proj_weight, q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight, v_proj_weight=v_proj_weight, static_k=static_k, static_v=static_v)
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, 'embed_dim must be divisible by num_heads'
    scaling = float(head_dim) ** -0.5
    if not use_separate_proj_weight:
        if torch.equal(query, key) and torch.equal(key, value):
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
        elif torch.equal(key, value):
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)
            if key is None:
                assert value is None
                k = None
                v = None
            else:
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)
        else:
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)
        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)
        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)
        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:embed_dim * 2])
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[embed_dim * 2:])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling
    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, 'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn('Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.')
            attn_mask = attn_mask
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn('Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.')
        key_padding_mask = key_padding_mask
    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, 'bias cannot be added to static key.'
            assert static_v is None, 'bias cannot be added to static value.'
    else:
        assert bias_k is None
        assert bias_v is None
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k
    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v
    src_len = k.size(1)
    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len
    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'))
        else:
            attn_output_weights += attn_mask
    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)
    if attention_probs_forward_hook is not None and attention_probs_backwards_hook is not None:
        attention_probs_forward_hook(attn_output_weights)
        attn_output_weights.register_hook(attention_probs_backwards_hook)
    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    if need_weights:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


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
        x, _ = multi_head_attention_forward(query=x, key=x, value=x, embed_dim_to_check=x.shape[-1], num_heads=self.num_heads, q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight, in_proj_weight=None, in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]), bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0, out_proj_weight=self.c_proj.weight, out_proj_bias=self.c_proj.bias, use_separate_proj_weight=True, training=self.training, need_weights=False)
        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):

        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: 'torch.Tensor'):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):

    def forward(self, x: 'torch.Tensor'):
        return x * torch.sigmoid(1.702 * x)


class _LinearWithBias(torch.nn.Linear):
    bias: 'Tensor'

    def __init__(self, in_features: 'int', out_features: 'int') ->None:
        super().__init__(in_features, out_features, bias=True)


class MultiheadAttention(torch.nn.Module):
    """Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \\text{MultiHead}(Q, K, V) = \\text{Concat}(head_1,\\dots,head_h)W^O
        \\text{where} head_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    bias_k: 'Optional[torch.Tensor]'
    bias_v: 'Optional[torch.Tensor]'

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)
        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)
        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
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

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, attention_probs_forward_hook=None, attention_probs_backwards_hook=None):
        """
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.

        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, use_separate_proj_weight=True, q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight, v_proj_weight=self.v_proj_weight, attention_probs_forward_hook=attention_probs_forward_hook, attention_probs_backwards_hook=attention_probs_backwards_hook)
        else:
            return multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, attention_probs_forward_hook=attention_probs_forward_hook, attention_probs_backwards_hook=attention_probs_backwards_hook)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: 'int', n_head: 'int', attn_mask: 'torch.Tensor'=None):
        super().__init__()
        self.attn = MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)), ('gelu', QuickGELU()), ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.attn_probs = None
        self.attn_grad = None

    def set_attn_probs(self, attn_probs):
        self.attn_probs = attn_probs

    def set_attn_grad(self, attn_grad):
        self.attn_grad = attn_grad

    def attention(self, x: 'torch.Tensor'):
        self.attn_mask = self.attn_mask if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, attention_probs_forward_hook=self.set_attn_probs, attention_probs_backwards_hook=self.set_attn_grad)[0]

    def forward(self, x: 'torch.Tensor'):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):

    def __init__(self, width: 'int', layers: 'int', heads: 'int', attn_mask: 'torch.Tensor'=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: 'torch.Tensor'):
        return self.resblocks(x)


class VisionTransformer(nn.Module):

    def __init__(self, input_resolution: 'int', patch_size: 'int', width: 'int', layers: 'int', heads: 'int', output_dim: 'int'):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def interpolate_pos_encoding(self, x, w, h):
        positional_embedding = self.positional_embedding.unsqueeze(0)
        patch_size = self.conv1.kernel_size[0]
        npatch = x.shape[1] - 1
        N = positional_embedding.shape[1] - 1
        if npatch == N and w == h:
            return positional_embedding
        class_pos_embed = positional_embedding[:, 0]
        patch_pos_embed = positional_embedding[:, 1:]
        dim = x.shape[-1]
        w0 = w // patch_size
        h0 = h // patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2), scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)), mode='bicubic')
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x: 'torch.Tensor'):
        x = self.transformer_first_blocks_forward(x)
        x = self.transformer.resblocks[-1](x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x

    def transformer_first_blocks_forward(self, x):
        h, w = x.shape[-2:]
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        positional_embedding = self.interpolate_pos_encoding(x, w, h)
        x = x + positional_embedding
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer.resblocks[:-1](x)
        return x

    @staticmethod
    def attn_cosine_sim(x, eps=1e-08):
        norm = x.norm(dim=2, keepdim=True)
        factor = torch.clamp(norm @ norm.permute(0, 2, 1), min=eps)
        sim_matrix = x @ x.permute(0, 2, 1) / factor
        return sim_matrix


class VisualTransformer(nn.Module):

    def __init__(self, input_resolution: 'int', patch_size: 'int', width: 'int', layers: 'int', heads: 'int', output_dim: 'int'):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: 'torch.Tensor'):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x


class CLIP(nn.Module):

    def __init__(self, embed_dim: 'int', image_resolution: 'int', vision_layers: 'Union[Tuple[int, int, int, int], int]', vision_width: 'int', vision_patch_size: 'int', context_length: 'int', vocab_size: 'int', transformer_width: 'int', transformer_heads: 'int', transformer_layers: 'int'):
        super().__init__()
        self.context_length = context_length
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(layers=vision_layers, output_dim=embed_dim, heads=vision_heads, input_resolution=image_resolution, width=vision_width)
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(input_resolution=image_resolution, patch_size=vision_patch_size, width=vision_width, layers=vision_layers, heads=vision_heads, output_dim=embed_dim)
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
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith('bn3.weight'):
                        nn.init.zeros_(param)
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

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        return logits_per_image, logits_per_text


class Concat(nn.Module):

    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))
        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]
        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)
            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2:diff2 + target_shape2, diff3:diff3 + target_shape3])
        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class GenNoise(nn.Module):

    def __init__(self, dim2):
        super(GenNoise, self).__init__()
        self.dim2 = dim2

    def forward(self, input):
        a = list(input.size())
        a[1] = self.dim2
        b = torch.zeros(a).type_as(input.data)
        b.normal_()
        x = torch.autograd.Variable(b)
        return x


class Swish(nn.Module):
    """
    https://arxiv.org/abs/1710.05941
    The hype was so huge that I could not help but try it
    """

    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


class PixelNormLayer(nn.Module):
    """
    Pixelwise feature vector normalization.
    """

    def __init__(self, eps=1e-08):
        super(PixelNormLayer, self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-08)

    def __repr__(self):
        return self.__class__.__name__ + '(eps = %s)' % self.eps


class DecorrelatedColorsToRGB(nn.Module):
    """Converts from a decorrelated color space to RGB. See
    https://github.com/eps696/aphantasia/blob/master/aphantasia/image.py. Usually intended
    to be followed by a sigmoid.
    """

    def __init__(self, inv_color_scale=1.6):
        super().__init__()
        color_correlation_svd_sqrt = torch.tensor([[0.26, 0.09, 0.02], [0.27, 0.0, -0.05], [0.27, -0.09, 0.03]])
        color_correlation_svd_sqrt /= torch.tensor([inv_color_scale, 1.0, 1.0])
        max_norm_svd_sqrt = color_correlation_svd_sqrt.norm(dim=0).max()
        color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
        self.register_buffer('colcorr_t', color_correlation_normalized.T)

    def inverse(self, image):
        colcorr_t_inv = torch.linalg.inv(self.colcorr_t)
        return torch.einsum('nchw,cd->ndhw', image, colcorr_t_inv)

    def forward(self, image):
        if image.dim() == 4:
            image_rgb, alpha = image[:, :3], image[:, 3].unsqueeze(1)
            image_rgb = torch.einsum('nchw,cd->ndhw', image_rgb, self.colcorr_t)
            image = torch.cat([image_rgb, alpha], dim=1)
        else:
            image = torch.einsum('nchw,cd->ndhw', image, self.colcorr_t)
        return image


def get_kernel(factor, kernel_type, phase, kernel_width, support=None, sigma=None):
    assert kernel_type in ['lanczos', 'gauss', 'box']
    if phase == 0.5 and kernel_type != 'box':
        kernel = np.zeros([kernel_width - 1, kernel_width - 1])
    else:
        kernel = np.zeros([kernel_width, kernel_width])
    if kernel_type == 'box':
        assert phase == 0.5, 'Box filter is always half-phased'
        kernel[:] = 1.0 / (kernel_width * kernel_width)
    elif kernel_type == 'gauss':
        assert sigma, 'sigma is not specified'
        assert phase != 0.5, 'phase 1/2 for gauss not implemented'
        center = (kernel_width + 1.0) / 2.0
        None
        sigma_sq = sigma * sigma
        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                di = (i - center) / 2.0
                dj = (j - center) / 2.0
                kernel[i - 1][j - 1] = np.exp(-(di * di + dj * dj) / (2 * sigma_sq))
                kernel[i - 1][j - 1] = kernel[i - 1][j - 1] / (2.0 * np.pi * sigma_sq)
    elif kernel_type == 'lanczos':
        assert support, 'support is not specified'
        center = (kernel_width + 1) / 2.0
        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                if phase == 0.5:
                    di = abs(i + 0.5 - center) / factor
                    dj = abs(j + 0.5 - center) / factor
                else:
                    di = abs(i - center) / factor
                    dj = abs(j - center) / factor
                pi_sq = np.pi * np.pi
                val = 1
                if di != 0:
                    val = val * support * np.sin(np.pi * di) * np.sin(np.pi * di / support)
                    val = val / (np.pi * np.pi * di * di)
                if dj != 0:
                    val = val * support * np.sin(np.pi * dj) * np.sin(np.pi * dj / support)
                    val = val / (np.pi * np.pi * dj * dj)
                kernel[i - 1][j - 1] = val
    else:
        assert False, 'wrong method name'
    kernel /= kernel.sum()
    return kernel


class Downsampler(nn.Module):
    """
    http://www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
    """

    def __init__(self, n_planes, factor, kernel_type, phase=0, kernel_width=None, support=None, sigma=None, preserve_size=False):
        super(Downsampler, self).__init__()
        assert phase in [0, 0.5], 'phase should be 0 or 0.5'
        if kernel_type == 'lanczos2':
            support = 2
            kernel_width = 4 * factor + 1
            kernel_type_ = 'lanczos'
        elif kernel_type == 'lanczos3':
            support = 3
            kernel_width = 6 * factor + 1
            kernel_type_ = 'lanczos'
        elif kernel_type == 'gauss12':
            kernel_width = 7
            sigma = 1 / 2
            kernel_type_ = 'gauss'
        elif kernel_type == 'gauss1sq2':
            kernel_width = 9
            sigma = 1.0 / np.sqrt(2)
            kernel_type_ = 'gauss'
        elif kernel_type in ['lanczos', 'gauss', 'box']:
            kernel_type_ = kernel_type
        else:
            assert False, 'wrong name kernel'
        self.kernel = get_kernel(factor, kernel_type_, phase, kernel_width, support=support, sigma=sigma)
        downsampler = nn.Conv2d(n_planes, n_planes, kernel_size=self.kernel.shape, stride=factor, padding=0)
        downsampler.weight.data[:] = 0
        downsampler.bias.data[:] = 0
        kernel_torch = torch.from_numpy(self.kernel)
        for i in range(n_planes):
            downsampler.weight.data[i, i] = kernel_torch
        self.downsampler_ = downsampler
        if preserve_size:
            if self.kernel.shape[0] % 2 == 1:
                pad = int((self.kernel.shape[0] - 1) / 2.0)
            else:
                pad = int((self.kernel.shape[0] - factor) / 2.0)
            self.padding = nn.ReplicationPad2d(pad)
        self.preserve_size = preserve_size

    def forward(self, input):
        if self.preserve_size:
            x = self.padding(input)
        else:
            x = input
        self.x = x
        return self.downsampler_(x)


def compose_text_with_templates(text: 'str', templates) ->list:
    return [template.format(text) for template in templates]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ClipExtractor(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model = clip.load(cfg['clip_model_name'], device=device)[0]
        self.model = model.eval().requires_grad_(False)
        self.clip_input_size = 224
        self.clip_normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        self.basic_transform = T.Compose([T.Resize(self.clip_input_size, max_size=380), self.clip_normalize])
        self.augs = T.Compose([T.RandomHorizontalFlip(p=0.5), T.RandomApply([T.RandomAffine(degrees=15, translate=(0.1, 0.1), fill=cfg['clip_affine_transform_fill'], interpolation=InterpolationMode.BILINEAR)], p=0.8), T.RandomPerspective(distortion_scale=0.4, p=0.5, interpolation=InterpolationMode.BILINEAR, fill=cfg['clip_affine_transform_fill']), T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.7), T.RandomGrayscale(p=0.15)])
        self.n_aug = cfg['n_aug']

    def augment_input(self, input, n_aug=None, clip_input_size=None):
        if n_aug is None:
            n_aug = self.n_aug
        if clip_input_size is None:
            clip_input_size = self.clip_input_size
        cutouts = []
        cutout = T.Resize(clip_input_size, max_size=320)(input)
        cutout_h, cutout_w = cutout.shape[-2:]
        cutout = self.augs(cutout)
        cutouts.append(cutout)
        sideY, sideX = input.shape[2:4]
        for _ in range(n_aug - 1):
            s = torch.zeros(1).uniform_(0.6, 1).item()
            h = int(sideY * s)
            w = int(sideX * s)
            cutout = T.RandomCrop(size=(h, w))(input)
            cutout = T.Resize((cutout_h, cutout_w))(cutout)
            cutout = self.augs(cutout)
            cutouts.append(cutout)
        cutouts = torch.cat(cutouts)
        return cutouts

    def get_image_embedding(self, x, aug=True):
        if aug:
            views = self.augment_input(x)
        else:
            views = self.basic_transform(x)
        if type(views) == list:
            image_embeds = []
            for view in views:
                image_embeds.append(self.encode_image(self.clip_normalize(view)))
            image_embeds = torch.cat(image_embeds)
        else:
            image_embeds = self.encode_image(self.clip_normalize(views))
        return image_embeds

    def encode_image(self, x):
        return self.model.encode_image(x)

    def get_text_embedding(self, text, template, average_embeddings=False):
        if type(text) == str:
            text = [text]
        embeddings = []
        for prompt in text:
            with torch.no_grad():
                embedding = self.model.encode_text(clip.tokenize(compose_text_with_templates(prompt, template)))
            embeddings.append(embedding)
        embeddings = torch.cat(embeddings)
        if average_embeddings:
            embeddings = embeddings.mean(dim=0, keepdim=True)
        return embeddings

    def get_self_sim(self, x):
        x = self.basic_transform(x)
        return self.model.calculate_self_sim(x)


class ClipRelevancy(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = clip.load('ViT-B/32', device=device, jit=False)[0]
        clip_input_size = 224
        self.preprocess = T.Compose([T.Resize((clip_input_size, clip_input_size)), T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
        input_prompts = cfg['bootstrap_text']
        if type(input_prompts) == str:
            input_prompts = [input_prompts]
        self.text = clip.tokenize(input_prompts)
        if self.cfg['use_negative_bootstrap']:
            input_negative_prompts = cfg['bootstrap_negative_text']
            if type(input_negative_prompts) == str:
                input_negative_prompts = [input_negative_prompts]
            self.bootstrap_negative_text = clip.tokenize(input_negative_prompts)

    def image_relevance(self, image_relevance):
        patch_size = 32
        h = w = 224
        image_relevance = image_relevance.reshape(1, 1, h // patch_size, w // patch_size)
        image_relevance = torch.nn.functional.interpolate(image_relevance, size=(h, w), mode='bilinear')
        image_relevance = image_relevance.reshape(h, w)
        image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
        return image_relevance

    def interpret(self, image, negative=False):
        text = self.text if not negative else self.bootstrap_negative_text
        batch_size = text.shape[0]
        images = image.repeat(batch_size, 1, 1, 1)
        logits_per_image, logits_per_text = self.model(images, text)
        probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
        index = [i for i in range(batch_size)]
        one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
        one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot * logits_per_image)
        self.model.zero_grad()
        image_attn_blocks = list(dict(self.model.visual.transformer.resblocks.named_children()).values())
        num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
        R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype)
        R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        for i, blk in enumerate(image_attn_blocks):
            if i <= self.cfg['relevancy_num_layers']:
                continue
            grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
            cam = blk.attn_probs.detach()
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            R = R + torch.bmm(cam, R)
        image_relevance = R[:, 0, 1:]
        return image_relevance

    def forward(self, img, preprocess=True, negative=False):
        if preprocess:
            img = self.preprocess(img)
        R_image = self.interpret(img, negative=negative)
        res = []
        for el in R_image:
            res.append(self.image_relevance(el).float())
        res = torch.stack(res, dim=0)
        return res


def act(act_fun='LeakyReLU'):
    """
    Either string defining an activation function or module (e.g. nn.ReLU)
    """
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':
        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
        else:
            assert False
        stride = 1
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)
    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)


def bn(num_features):
    return nn.BatchNorm2d(num_features)


_norm = bn


def norm(channels):
    return _norm(channels)


def skip(num_input_channels=2, num_output_channels=3, num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], filter_size_down=3, filter_size_up=3, filter_skip_size=1, need_sigmoid=True, need_tanh=False, need_bias=True, pad='reflection', upsample_mode='bilinear', downsample_mode='stride', act_fun='LeakyReLU', need1x1_up=True, decorr_rgb=True):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)
    n_scales = len(num_channels_down)
    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode = [upsample_mode] * n_scales
    if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
        downsample_mode = [downsample_mode] * n_scales
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
        filter_size_down = [filter_size_down] * n_scales
    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up = [filter_size_up] * n_scales
    last_scale = n_scales - 1
    cur_depth = None
    model = nn.Sequential()
    model_tmp = model
    input_depth = num_input_channels
    for i in range(len(num_channels_down)):
        deeper = nn.Sequential()
        skip = nn.Sequential()
        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)
        model_tmp.add(norm(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))
        if num_channels_skip[i] != 0:
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(norm(num_channels_skip[i]))
            skip.add(act(act_fun))
        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add(norm(num_channels_down[i]))
        deeper.add(act(act_fun))
        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(norm(num_channels_down[i]))
        deeper.add(act(act_fun))
        deeper_main = nn.Sequential()
        if i == len(num_channels_down) - 1:
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]
        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))
        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(norm(num_channels_up[i]))
        model_tmp.add(act(act_fun))
        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            if i > 0:
                model_tmp.add(norm(num_channels_up[i]))
            model_tmp.add(act(act_fun))
        input_depth = num_channels_down[i]
        model_tmp = deeper_main
    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if decorr_rgb:
        model.add(DecorrelatedColorsToRGB())
    if need_sigmoid:
        model.add(nn.Sigmoid())
    elif need_tanh:
        model.add(nn.Tanh())
    return model


def define_G(cfg):
    netG = skip(3, 4, num_channels_down=[cfg['skip_n33d']] * cfg['num_scales'] if isinstance(cfg['skip_n33d'], int) else cfg['skip_n33d'], num_channels_up=[cfg['skip_n33u']] * cfg['num_scales'] if isinstance(cfg['skip_n33u'], int) else cfg['skip_n33u'], num_channels_skip=[cfg['skip_n11']] * cfg['num_scales'] if isinstance(cfg['skip_n11'], int) else cfg['skip_n11'], need_bias=True)
    return netG


class Model(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.netG = define_G(cfg)

    def render(self, net_output, bg_image=None):
        assert net_output.min() >= 0 and net_output.max() <= 1
        edit = net_output[:, :3]
        alpha = net_output[:, 3].unsqueeze(1).repeat(1, 3, 1, 1)
        greenscreen = torch.zeros_like(edit)
        greenscreen[:, 1, :, :] = 177 / 255
        greenscreen[:, 2, :, :] = 64 / 255
        edit_on_greenscreen = alpha * edit + (1 - alpha) * greenscreen
        outputs = {'edit': edit, 'alpha': alpha, 'edit_on_greenscreen': edit_on_greenscreen}
        if bg_image is not None:
            outputs['composite'] = (1 - alpha) * bg_image + alpha * edit
        return outputs

    def forward(self, input):
        outputs = {}
        if 'input_crop' in input:
            outputs['output_crop'] = self.render(self.netG(input['input_crop']), bg_image=input['input_crop'])
        if 'input_image' in input:
            outputs['output_image'] = self.render(self.netG(input['input_image']), bg_image=input['input_image'])
        for outer_key in outputs.keys():
            for key, value in outputs[outer_key].items():
                outputs[outer_key][key] = [value[0]]
        return outputs


def positionalEncoding_vec(in_tensor, b):
    proj = torch.einsum('ij, k -> ijk', in_tensor, b)
    mapped_coords = torch.cat((torch.sin(proj), torch.cos(proj)), dim=1)
    output = mapped_coords.transpose(2, 1).contiguous().view(mapped_coords.size(0), -1)
    return output


class IMLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=256, use_positional=True, positional_dim=10, skip_layers=[4, 6], num_layers=8, verbose=True, use_tanh=True, apply_softmax=False):
        super(IMLP, self).__init__()
        self.verbose = verbose
        self.use_tanh = use_tanh
        self.apply_softmax = apply_softmax
        if apply_softmax:
            self.softmax = nn.Softmax()
        if use_positional:
            encoding_dimensions = 2 * input_dim * positional_dim
            self.b = torch.tensor([(2 ** j * np.pi) for j in range(positional_dim)], requires_grad=False)
        else:
            encoding_dimensions = input_dim
        self.hidden = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                input_dims = encoding_dimensions
            elif i in skip_layers:
                input_dims = hidden_dim + encoding_dimensions
            else:
                input_dims = hidden_dim
            if i == num_layers - 1:
                self.hidden.append(nn.Linear(input_dims, output_dim, bias=True))
            else:
                self.hidden.append(nn.Linear(input_dims, hidden_dim, bias=True))
        self.skip_layers = skip_layers
        self.num_layers = num_layers
        self.positional_dim = positional_dim
        self.use_positional = use_positional
        if self.verbose:
            None

    def forward(self, x):
        if self.use_positional:
            if self.b.device != x.device:
                self.b = self.b
            pos = positionalEncoding_vec(x, self.b)
            x = pos
        input = x.detach().clone()
        for i, layer in enumerate(self.hidden):
            if i > 0:
                x = F.relu(x)
            if i in self.skip_layers:
                x = torch.cat((x, input), 1)
            x = layer(x)
        if self.use_tanh:
            x = torch.tanh(x)
        if self.apply_softmax:
            x = self.softmax(x)
        return x


class VideoModel(Model):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.net_preprocess = transforms.Compose([])

    @staticmethod
    def resize_crops(crops, resize_factor):
        return torchvision.transforms.functional.resize(crops, [crops.shape[-2] // resize_factor, crops.shape[-1] // resize_factor], InterpolationMode.BILINEAR, antialias=True)

    def process_crops(self, uv_values, crops, original_crops, alpha=None):
        resized_crops = []
        cnn_output_crops = []
        render_dict = {'edit': [], 'alpha': [], 'edit_on_greenscreen': [], 'composite': []}
        atlas_crop = crops[0]
        for i in range(3):
            grid_sampled_atlas_crop = F.grid_sample(atlas_crop, uv_values[i], mode='bilinear', align_corners=self.config['align_corners']).clamp(min=0.0, max=1.0)
            resized_crops.append(grid_sampled_atlas_crop)
        cnn_output = self.netG(atlas_crop)
        cnn_output_crops.append(cnn_output[:, :3])
        rendered_atlas_crops = self.render(cnn_output, bg_image=atlas_crop)
        for key, value in rendered_atlas_crops.items():
            for i in range(3):
                sampled_frame_crop = F.grid_sample(value, uv_values[i], mode='bilinear', align_corners=self.config['align_corners']).clamp(min=0.0, max=1.0)
                if alpha is not None:
                    sampled_frame_crop = sampled_frame_crop * alpha[i]
                    if key == 'edit_on_greenscreen':
                        greenscreen = torch.zeros_like(sampled_frame_crop)
                        greenscreen[:, 1, :, :] = 177 / 255
                        greenscreen[:, 2, :, :] = 64 / 255
                        sampled_frame_crop += (1 - alpha[i]) * greenscreen
                render_dict[key].append(sampled_frame_crop.squeeze(0))
        frame_index = random.randint(0, 2)
        rec_crop = original_crops[frame_index]
        resized_crops.append(rec_crop)
        cnn_output = self.netG(rec_crop)
        if alpha is not None:
            alpha_crop = alpha[frame_index]
            cnn_output = cnn_output * alpha_crop
        cnn_output_crops.append(cnn_output[:, :3])
        rendered_frame_crop = self.render(cnn_output, bg_image=original_crops[frame_index])
        for key, value in rendered_frame_crop.items():
            render_dict[key].append(value.squeeze(0))
        return render_dict, resized_crops, cnn_output_crops

    def process_atlas(self, atlas):
        atlas_edit = self.netG(atlas)
        rendered_atlas = self.render(atlas_edit, bg_image=atlas)
        return rendered_atlas

    def forward(self, input_dict):
        inputs = input_dict['global_crops']
        outputs = {'background': {}, 'foreground': {}}
        if self.config['finetune_foreground']:
            if self.config['multiply_foreground_alpha']:
                alpha = inputs['foreground_alpha']
            else:
                alpha = None
            foreground_outputs, resized_crops, cnn_output_crops = self.process_crops(inputs['foreground_uvs'], inputs['foreground_atlas_crops'], inputs['original_foreground_crops'], alpha=alpha)
            outputs['foreground']['output_crop'] = foreground_outputs
            outputs['foreground']['cnn_inputs'] = resized_crops
            outputs['foreground']['cnn_outputs'] = cnn_output_crops
            if 'input_image' in input_dict.keys():
                outputs['foreground']['output_image'] = self.process_atlas(input_dict['input_image'])
        elif self.config['finetune_background']:
            background_outputs, resized_crops, cnn_output_crops = self.process_crops(inputs['background_uvs'], inputs['background_atlas_crops'], inputs['original_background_crops'])
            outputs['background']['output_crop'] = background_outputs
            outputs['background']['cnn_inputs'] = resized_crops
            outputs['background']['cnn_outputs'] = cnn_output_crops
            if 'input_image' in input_dict.keys():
                outputs['background']['output_image'] = self.process_atlas(input_dict['input_image'])
        return outputs


class RandomSizeCrop(object):

    def __init__(self, min_cover):
        super(RandomSizeCrop, self).__init__()
        self.min_cover = min_cover

    def __call__(self, img):
        if self.min_cover == 1:
            return img
        if isinstance(img, torch.Tensor):
            h, w = img.shape[-2:]
        else:
            w, h = img.size[-2:]
        s = np.random.uniform(self.min_cover, 1)
        size_h = int(h * s)
        size_w = int(w * s)
        return T.RandomCrop((size_h, size_w))(img)


def get_augmentations_template():
    templates = ['photo of {}.', 'high quality photo of {}.', 'a photo of {}.', 'the photo of {}.', 'image of {}.', 'an image of {}.', 'high quality image of {}.', 'a high quality image of {}.', 'the {}.', 'a {}.', '{}.', '{}', '{}!', '{}...']
    return templates


def get_screen_template():
    return ['{} over a green screen.', '{} in front of a green screen.']


def cosine_loss(x, y, scaling=1.2):
    return scaling * (1 - F.cosine_similarity(x, y).mean())


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2).mean()


def get_text_criterion(cfg):
    if cfg['text_criterion'] == 'spherical':
        text_criterion = spherical_dist_loss
    elif cfg['text_criterion'] == 'cosine':
        text_criterion = cosine_loss
    else:
        return NotImplementedError('text criterion [%s] is not implemented', cfg['text_criterion'])
    return text_criterion


class LossG(torch.nn.Module):

    def __init__(self, cfg, clip_extractor):
        super().__init__()
        self.cfg = cfg
        template = get_augmentations_template()
        self.src_e = clip_extractor.get_text_embedding(cfg['src_text'], template)
        self.target_comp_e = clip_extractor.get_text_embedding(cfg['comp_text'], template)
        self.target_greenscreen_e = clip_extractor.get_text_embedding(cfg['screen_text'], get_screen_template())
        self.clip_extractor = clip_extractor
        self.text_criterion = get_text_criterion(cfg)
        if cfg['bootstrap_epoch'] > 0 and cfg['lambda_bootstrap'] > 0:
            self.relevancy_extractor = ClipRelevancy(cfg)
            self.relevancy_criterion = torch.nn.MSELoss()
            self.lambda_bootstrap = cfg['lambda_bootstrap']

    def forward(self, outputs, inputs):
        losses = {}
        loss_G = 0
        all_outputs_composite = []
        all_outputs_greenscreen = []
        all_outputs_edit = []
        all_outputs_alpha = []
        all_inputs = []
        for out, ins in zip(['output_crop', 'output_image'], ['input_crop', 'input_image']):
            if out not in outputs:
                continue
            all_outputs_composite += outputs[out]['composite']
            all_outputs_greenscreen += outputs[out]['edit_on_greenscreen']
            all_outputs_edit += outputs[out]['edit']
            all_outputs_alpha += outputs[out]['alpha']
            all_inputs += inputs[ins]
        if inputs['step'] < self.cfg['bootstrap_epoch'] and self.cfg['lambda_bootstrap'] > 0:
            losses['loss_bootstrap'] = self.calculate_relevancy_loss(all_outputs_alpha, all_inputs)
            if self.cfg['bootstrap_scheduler'] == 'linear':
                lambda_bootstrap = self.cfg['lambda_bootstrap'] * (1 - (inputs['step'] + 1) / self.cfg['bootstrap_epoch'])
            elif self.cfg['bootstrap_scheduler'] == 'exponential':
                lambda_bootstrap = self.lambda_bootstrap * 0.99
                self.lambda_bootstrap = lambda_bootstrap
            elif self.cfg['bootstrap_scheduler'] == 'none':
                lambda_bootstrap = self.lambda_bootstrap
            else:
                raise ValueError('Unknown bootstrap scheduler')
            lambda_bootstrap = max(lambda_bootstrap, self.cfg['lambda_bootstrap_min'])
            loss_G += losses['loss_bootstrap'] * lambda_bootstrap
        if self.cfg['lambda_structure'] > 0:
            losses['loss_structure'] = self.calculate_structure_loss(all_outputs_composite, all_inputs)
            loss_G += losses['loss_structure'] * self.cfg['lambda_structure']
        if self.cfg['lambda_composition'] > 0:
            losses['loss_comp_clip'] = self.calculate_clip_loss(all_outputs_composite, self.target_comp_e)
            losses['loss_comp_dir'] = self.calculate_clip_dir_loss(all_inputs, all_outputs_composite, self.target_comp_e)
            loss_G += (losses['loss_comp_clip'] + losses['loss_comp_dir']) * self.cfg['lambda_composition']
        if self.cfg['lambda_sparsity'] > 0:
            total, l0, l1 = self.calculate_alpha_reg(all_outputs_alpha)
            losses['loss_sparsity'] = total
            losses['loss_sparsity_l0'] = l0
            losses['loss_sparsity_l1'] = l1
            loss_G += losses['loss_sparsity'] * self.cfg['lambda_sparsity']
        if self.cfg['lambda_screen'] > 0:
            losses['loss_screen'] = self.calculate_clip_loss(all_outputs_greenscreen, self.target_greenscreen_e)
            loss_G += losses['loss_screen'] * self.cfg['lambda_screen']
        losses['loss'] = loss_G
        return losses

    def calculate_alpha_reg(self, prediction):
        """
        Calculate the alpha sparsity term: linear combination between L1 and pseudo L0 penalties
        """
        l1_loss = 0.0
        for el in prediction:
            l1_loss += el.mean()
        l1_loss = l1_loss / len(prediction)
        loss = self.cfg['lambda_alpha_l1'] * l1_loss
        l0_loss = 0.0
        for el in prediction:
            l0_loss += torch.mean((torch.sigmoid(el * 5.0) - 0.5) * 2.0)
        l0_loss = l0_loss / len(prediction)
        loss += self.cfg['lambda_alpha_l0'] * l0_loss
        return loss, l0_loss, l1_loss

    def calculate_clip_loss(self, outputs, target_embeddings):
        n_embeddings = np.random.randint(1, len(target_embeddings) + 1)
        target_embeddings = target_embeddings[torch.randint(len(target_embeddings), (n_embeddings,))]
        loss = 0.0
        for img in outputs:
            img_e = self.clip_extractor.get_image_embedding(img.unsqueeze(0))
            for target_embedding in target_embeddings:
                loss += self.text_criterion(img_e, target_embedding.unsqueeze(0))
        loss /= len(outputs) * len(target_embeddings)
        return loss

    def calculate_clip_dir_loss(self, inputs, outputs, target_embeddings):
        n_embeddings = np.random.randint(1, min(len(self.src_e), len(target_embeddings)) + 1)
        idx = torch.randint(min(len(self.src_e), len(target_embeddings)), (n_embeddings,))
        src_embeddings = self.src_e[idx]
        target_embeddings = target_embeddings[idx]
        target_dirs = target_embeddings - src_embeddings
        loss = 0.0
        for in_img, out_img in zip(inputs, outputs):
            in_e = self.clip_extractor.get_image_embedding(in_img.unsqueeze(0))
            out_e = self.clip_extractor.get_image_embedding(out_img.unsqueeze(0))
            for target_dir in target_dirs:
                loss += 1 - torch.nn.CosineSimilarity()(out_e - in_e, target_dir.unsqueeze(0)).mean()
        loss /= len(outputs) * len(target_dirs)
        return loss

    def calculate_structure_loss(self, outputs, inputs):
        loss = 0.0
        for input, output in zip(inputs, outputs):
            with torch.no_grad():
                target_self_sim = self.clip_extractor.get_self_sim(input.unsqueeze(0))
            current_self_sim = self.clip_extractor.get_self_sim(output.unsqueeze(0))
            loss = loss + torch.nn.MSELoss()(current_self_sim, target_self_sim)
        loss = loss / len(outputs)
        return loss

    def calculate_relevancy_loss(self, alpha, input_img):
        positive_relevance_loss = 0.0
        for curr_alpha, curr_img in zip(alpha, input_img):
            x = torch.stack([curr_alpha, curr_img], dim=0)
            x = T.Compose([RandomSizeCrop(min_cover=self.cfg['bootstrapping_min_cover']), T.Resize((224, 224))])(x)
            curr_alpha, curr_img = x[0].unsqueeze(0), x[1].unsqueeze(0)
            positive_relevance = self.relevancy_extractor(curr_img)
            positive_relevance_loss = self.relevancy_criterion(curr_alpha[0], positive_relevance.repeat(3, 1, 1))
            if self.cfg['use_negative_bootstrap']:
                negative_relevance = self.relevancy_extractor(curr_img, negative=True)
                relevant_values = negative_relevance > self.cfg['bootstrap_negative_map_threshold']
                negative_alpha_local = (1 - curr_alpha) * relevant_values.unsqueeze(1)
                negative_relevance_local = negative_relevance * relevant_values
                negative_relevance_loss = self.relevancy_criterion(negative_alpha_local, negative_relevance_local.unsqueeze(1).repeat(1, 3, 1, 1))
                positive_relevance_loss += negative_relevance_loss
        positive_relevance_loss = positive_relevance_loss / len(alpha)
        return positive_relevance_loss


class AtlasLoss(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.clip_extractor = ClipExtractor(config)
        common_config = {key: config[key] for key in ['lambda_composition', 'lambda_sparsity', 'lambda_screen', 'lambda_alpha_l1', 'lambda_alpha_l0', 'text_criterion', 'clip_model_name', 'bootstrap_epoch', 'lambda_bootstrap', 'relevancy_num_layers', 'lambda_structure', 'bootstrap_text', 'bootstrap_scheduler', 'bootstrapping_min_cover', 'use_negative_bootstrap', 'bootstrap_negative_text', 'bootstrap_negative_map_threshold', 'lambda_bootstrap_min', 'device']}
        texts_config = {'screen_text': config['screen_text'], 'comp_text': config['comp_text'], 'src_text': config['src_text']}
        common_config.update(texts_config)
        self.loss = LossG(common_config, self.clip_extractor)
        self.config = config

    def forward(self, outputs, inputs):
        losses = {}
        if self.config['finetune_background']:
            inputs['input_crop'] = [el.squeeze(0) for el in outputs['background']['cnn_inputs']]
            losses['background'] = self.loss(outputs['background'], inputs)
        elif self.config['finetune_foreground']:
            inputs['input_crop'] = [el.squeeze(0) for el in outputs['foreground']['cnn_inputs']]
            losses['foreground'] = self.loss(outputs['foreground'], inputs)
        return losses


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
    (DecorrelatedColorsToRGB,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GenNoise,
     lambda: ([], {'dim2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (IMLP,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MultiheadAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (PixelNormLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (QuickGELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualAttentionBlock,
     lambda: ([], {'d_model': 4, 'n_head': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Transformer,
     lambda: ([], {'width': 4, 'layers': 1, 'heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (VisionTransformer,
     lambda: ([], {'input_resolution': 4, 'patch_size': 4, 'width': 4, 'layers': 1, 'heads': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (_LinearWithBias,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

