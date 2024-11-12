
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


import math


import torch


import torch.nn as nn


from torch.nn import functional as F


from torch import nn


import torch.nn.functional as F


from torch.nn import Parameter


import torch.onnx.operators


import matplotlib


import re


import random


import numpy as np


import torch.distributed as dist


import torch.utils.data


import torch.optim


import time


from collections import defaultdict


import types


from torch.nn import DataParallel


from torch.nn.parallel import DistributedDataParallel


import itertools


from functools import wraps


from torch.cuda._utils import _get_device_index


import copy


import torch.multiprocessing as mp


from torch.optim.optimizer import Optimizer


DEFAULT_MAX_TARGET_POSITIONS = 2000


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(init_size, embedding_dim, padding_idx)
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None, timestep=None, **kwargs):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.shape[:2]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = SinusoidalPositionalEmbedding.get_embedding(max_pos, self.embedding_dim, self.padding_idx)
        self.weights = self.weights
        if incremental_state is not None:
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)
        positions = utils.make_positions(input, self.padding_idx)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(100000.0)


class ConvTBC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ConvTBC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.weight = torch.nn.Parameter(torch.Tensor(self.kernel_size, in_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

    def forward(self, input):
        return torch.conv_tbc(input.contiguous(), self.weight, self.bias, self.padding)


def LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, export=False):
    if not export and torch.cuda.is_available():
        try:
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class EncConvLayer(nn.Module):

    def __init__(self, c, kernel_size, dropout):
        super().__init__()
        self.layer_norm = LayerNorm(c)
        conv = ConvTBC(c, c, kernel_size, padding=kernel_size // 2)
        std = math.sqrt(4 * (1.0 - dropout) / (kernel_size * c))
        nn.init.normal_(conv.weight, mean=0, std=std)
        nn.init.constant_(conv.bias, 0)
        self.conv = nn.utils.weight_norm(conv, dim=2)
        self.dropout = dropout

    def forward(self, x, encoder_padding_mask=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm.training = layer_norm_training
        residual = x
        if encoder_padding_mask is not None:
            x = x.masked_fill(encoder_padding_mask.t().unsqueeze(-1), 0)
        x = self.layer_norm(x)
        x = self.conv(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, self.training)
        x = x + residual
        return x


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class EncLSTMLayer(nn.Module):

    def __init__(self, c, dropout):
        super().__init__()
        self.c = c
        self.layer_norm = LayerNorm(c)
        self.lstm = nn.LSTM(c, c, 1, bidirectional=True)
        self.out_proj = Linear(2 * c, c)
        self.dropout = dropout

    def forward(self, x, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm.training = layer_norm_training
        self.lstm.flatten_parameters()
        residual = x
        x = self.layer_norm(x)
        x, _ = self.lstm(x)
        x = self.out_proj(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        return x


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, self_attention=False, encoder_decoder_attention=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.scaling = self.head_dim ** -0.5
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert not self.self_attention or self.qkv_same_dim, 'Self-attention requires query, key and value to be of the same size'
        if self.qkv_same_dim:
            self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        else:
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        self.add_zero_attn = add_zero_attn
        self.reset_parameters()
        self.enable_torch_version = False
        if hasattr(F, 'multi_head_attention_forward'):
            self.enable_torch_version = True
        else:
            self.enable_torch_version = False

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
            nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None, need_weights=True, static_kv=False, attn_mask=None, before_softmax=False, need_head_weights=False, enc_dec_attn_constraint_mask=None):
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
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if self.enable_torch_version and incremental_state is None and not static_kv:
            if self.qkv_same_dim:
                return F.multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, self.training, key_padding_mask, need_weights, attn_mask)
            else:
                return F.multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, torch.empty([0]), self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, self.training, key_padding_mask, need_weights, attn_mask, use_separate_proj_weight=True, q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight, v_proj_weight=self.v_proj_weight)
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None
        if self.self_attention:
            q, k, v = self.in_proj_qkv(query)
        elif self.encoder_decoder_attention:
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.in_proj_k(key)
                v = self.in_proj_v(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling
        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if saved_state is not None:
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            if 'prev_key_padding_mask' in saved_state and saved_state['prev_key_padding_mask'] is not None:
                prev_key_padding_mask = saved_state['prev_key_padding_mask']
                if static_kv:
                    key_padding_mask = prev_key_padding_mask
                else:
                    key_padding_mask = torch.cat((prev_key_padding_mask, key_padding_mask), dim=1)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_key_padding_mask'] = key_padding_mask
            self._set_input_buffer(incremental_state, saved_state)
        src_len = k.size(1)
        if key_padding_mask is not None and key_padding_mask.shape == torch.Size([]):
            key_padding_mask = None
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask
        if enc_dec_attn_constraint_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(enc_dec_attn_constraint_mask.unsqueeze(2).bool(), float('-inf'))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_logits = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        if before_softmax:
            return attn_weights, v
        attn_weights_float = utils.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                attn_weights = attn_weights.mean(dim=0)
        else:
            attn_weights = None
        return attn, (attn_weights, attn_logits)

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_q(self, query):
        if self.qkv_same_dim:
            return self._in_proj(query, end=self.embed_dim)
        else:
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[:self.embed_dim]
            return F.linear(query, self.q_proj_weight, bias)

    def in_proj_k(self, key):
        if self.qkv_same_dim:
            return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)
        else:
            weight = self.k_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[self.embed_dim:2 * self.embed_dim]
            return F.linear(key, weight, bias)

    def in_proj_v(self, value):
        if self.qkv_same_dim:
            return self._in_proj(value, start=2 * self.embed_dim)
        else:
            weight = self.v_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[2 * self.embed_dim:]
            return F.linear(value, weight, bias)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(self, incremental_state, 'attn_state') or {}

    def _set_input_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(self, incremental_state, 'attn_state', buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len, src_len, bsz):
        return attn_weights

    def clear_buffer(self, incremental_state=None):
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                del saved_state['prev_key']
            if 'prev_value' in saved_state:
                del saved_state['prev_value']
            self._set_input_buffer(incremental_state, saved_state)


class NewTransformerFFNLayer(nn.Module):

    def __init__(self, hidden_size, filter_size, padding='SAME', kernel_size=1, dropout=0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout
        if padding == 'SAME':
            self.ffn_1 = nn.Conv1d(hidden_size, filter_size, kernel_size, padding=kernel_size // 2)
        elif padding == 'LEFT':
            self.ffn_1 = nn.Sequential(nn.ConstantPad1d((kernel_size - 1, 0), 0.0), nn.Conv1d(hidden_size, filter_size, kernel_size))
        self.ffn_2 = Linear(filter_size, hidden_size)

    def forward(self, x, incremental_state=None):
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_input' in saved_state:
                prev_input = saved_state['prev_input']
                x = torch.cat((prev_input, x), dim=0)
            x = x[-self.kernel_size:]
            saved_state['prev_input'] = x
            self._set_input_buffer(incremental_state, saved_state)
        x = self.ffn_1(x.permute(1, 2, 0)).permute(2, 0, 1)
        x = x * self.kernel_size ** -0.5
        if incremental_state is not None:
            x = x[-1:]
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.ffn_2(x)
        return x

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(self, incremental_state, 'f') or {}

    def _set_input_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(self, incremental_state, 'f', buffer)

    def clear_buffer(self, incremental_state):
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_input' in saved_state:
                del saved_state['prev_input']
            self._set_input_buffer(incremental_state, saved_state)


class EncLocalSALayer(nn.Module):

    def __init__(self, c, num_heads, dropout, attention_dropout=0.1, relu_dropout=0.1):
        super().__init__()
        self.c = c
        self.dropout = dropout
        self.layer_norm1 = LayerNorm(c)
        self.self_attn = MultiheadAttention(self.c, num_heads, self_attention=True, dropout=attention_dropout, bias=False)
        self.layer_norm2 = LayerNorm(c)
        self.ffn = NewTransformerFFNLayer(c, 4 * c, kernel_size=9, dropout=relu_dropout)
        self.chunk_size = 101

    def forward(self, x, encoder_padding_mask=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
        residual = x
        x = self.layer_norm1(x)
        states = []
        T = x.shape[0]
        all_neg_inf = utils.fill_with_neg_inf2(x.new(T, T))
        half_chunk_size = self.chunk_size // 2
        attn_mask = torch.triu(all_neg_inf, half_chunk_size + 1) + torch.tril(all_neg_inf, -half_chunk_size - 1)
        encoder_padding_mask = encoder_padding_mask.data
        for i in range(0, x.shape[0], half_chunk_size + 1):
            k_start = max(0, i - half_chunk_size)
            k_end = min(x.shape[0], i + self.chunk_size)
            kv = x[k_start:k_end]
            q = x[i:i + half_chunk_size + 1]
            q_nonpadding = (1 - encoder_padding_mask[:, i:i + half_chunk_size + 1].float()).data
            encoder_padding_mask_ = encoder_padding_mask[:, k_start:k_end].data
            encoder_padding_mask_[q_nonpadding.sum(-1) == 0, :] = 0
            x_, _ = self.self_attn(query=q, key=kv, value=kv, key_padding_mask=encoder_padding_mask_, attn_mask=attn_mask[i:i + half_chunk_size + 1, k_start:k_end])
            x_ = x_ * (1 - q_nonpadding.T[:, :, None])
            states.append(x_)
        x = torch.cat(states)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        residual = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        return x


class EncSALayer(nn.Module):

    def __init__(self, c, num_heads, dropout, attention_dropout=0.1, relu_dropout=0.1, kernel_size=9, padding='SAME'):
        super().__init__()
        self.c = c
        self.dropout = dropout
        self.layer_norm1 = LayerNorm(c)
        self.self_attn = MultiheadAttention(self.c, num_heads, self_attention=True, dropout=attention_dropout, bias=False)
        self.layer_norm2 = LayerNorm(c)
        self.ffn = NewTransformerFFNLayer(c, 4 * c, kernel_size=kernel_size, dropout=relu_dropout, padding=padding)

    def forward(self, x, encoder_padding_mask=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
        residual = x
        x = self.layer_norm1(x)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        residual = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        return x


hparams = {}


OPERATIONS_ENCODER = {(1): lambda c, dropout: EncConvLayer(c, 1, dropout), (2): lambda c, dropout: EncConvLayer(c, 5, dropout), (3): lambda c, dropout: EncConvLayer(c, 9, dropout), (4): lambda c, dropout: EncConvLayer(c, 13, dropout), (5): lambda c, dropout: EncConvLayer(c, 17, dropout), (6): lambda c, dropout: EncConvLayer(c, 21, dropout), (7): lambda c, dropout: EncConvLayer(c, 25, dropout), (8): lambda c, dropout, k=None: EncSALayer(c, 2, dropout=dropout, attention_dropout=0.0, relu_dropout=dropout, kernel_size=k if k is not None else hparams['enc_ffn_kernel_size'], padding=hparams['ffn_padding']), (9): lambda c, dropout: EncSALayer(c, 4, dropout), (10): lambda c, dropout: EncSALayer(c, 8, dropout), (11): lambda c, dropout: EncLocalSALayer(c, 2, dropout), (12): lambda c, dropout: EncLSTMLayer(c, dropout)}


class TransformerEncoderLayer(nn.Module):

    def __init__(self, layer, hidden_size, dropout, kernel_size=None):
        super().__init__()
        self.layer = layer
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.op = OPERATIONS_ENCODER[layer](hidden_size, dropout, kernel_size)

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)


class BaseDecoder(nn.Module):

    def __init__(self, arch, hidden_size=None, dropout=None, num_layers=None, last_ln=True):
        super().__init__()
        self.arch = arch
        self.num_layers = hparams['dec_layers'] if num_layers is None else num_layers
        if hidden_size is not None:
            embed_dim = self.hidden_size = hidden_size
        else:
            embed_dim = self.hidden_size = hparams['hidden_size']
        if dropout is not None:
            self.dropout = dropout
        else:
            self.dropout = hparams['dropout']
        self.max_source_positions = DEFAULT_MAX_TARGET_POSITIONS
        self.padding_idx = 0
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim, self.padding_idx, init_size=self.max_source_positions + self.padding_idx + 1)
        self.layers = nn.ModuleList([])
        self.layers.extend([TransformerEncoderLayer(self.arch[i], self.hidden_size, self.dropout, kernel_size=hparams['dec_ffn_kernel_size']) for i in range(self.num_layers)])
        self.layer_norm = nn.LayerNorm(embed_dim) if last_ln else None

    def forward(self, x, require_w=False):
        """
        :param x: [B, T, C]
        :param require_w: True if this module needs to return weight matrix
        :return: [B, T, C]
        """
        padding_mask = x.abs().sum(-1).eq(0).data
        positions = self.pos_embed_alpha * self.embed_positions(x[..., 0])
        x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)
        attn_w = []
        if require_w:
            for layer in self.layers:
                x, attn_w_i = layer(x, encoder_padding_mask=padding_mask, require_w=require_w)
                attn_w.append(attn_w_i)
        else:
            for layer in self.layers:
                x = layer(x, encoder_padding_mask=padding_mask)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        x = x.transpose(0, 1)
        return (x, attn_w) if require_w else x


class DurationPredictor(torch.nn.Module):
    """Duration predictor module.
    This is a module of duration predictor described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The duration predictor predicts a duration of each frame in log domain from the hidden embeddings of encoder.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    Note:
        The calculation domain of outputs is different between in `forward` and in `inference`. In `forward`,
        the outputs are calculated in log domain but in `inference`, those are calculated in linear domain.
    """

    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0, padding='SAME'):
        """Initilize duration predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.
        """
        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0), torch.nn.ReLU(), LayerNorm(n_chans, dim=1), torch.nn.Dropout(dropout_rate))]
        self.linear = torch.nn.Linear(n_chans, 1)

    def _forward(self, xs, x_masks=None, is_inference=False):
        xs = xs.transpose(1, -1)
        for f in self.conv:
            if self.padding == 'SAME':
                xs = F.pad(xs, [self.kernel_size // 2, self.kernel_size // 2])
            elif self.padding == 'LEFT':
                xs = F.pad(xs, [self.kernel_size - 1, 0])
            xs = f(xs)
            if x_masks is not None:
                xs = xs * (1 - x_masks.float())[:, None, :]
        xs = self.linear(xs.transpose(1, -1)).squeeze(-1)
        if is_inference:
            xs = torch.clamp(torch.round(xs.exp() - self.offset), min=0).long()
        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)
        return xs

    def forward(self, xs, x_masks=None):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).
        """
        return self._forward(xs, x_masks, False)

    def inference(self, xs, x_masks=None):
        """Inference duration.
        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).
        """
        return self._forward(xs, x_masks, True)


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


def pad_list(xs, pad_value, max_len=None):
    """Perform padding for the list of tensors.
    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.
    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).
    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])
    """
    n_batch = len(xs)
    if max_len is None:
        max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :min(xs[i].size(0), max_len)] = xs[i][:max_len]
    return pad


class LengthRegulator(torch.nn.Module):
    """Length regulator module for feed-forward Transformer.
    This is a module of length regulator described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The length regulator expands char or phoneme-level embedding features to frame-level by repeating each
    feature based on the corresponding predicted durations.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    """

    def __init__(self, pad_value=0.0):
        """Initilize length regulator module.
        Args:
            pad_value (float, optional): Value used for padding.
        """
        super(LengthRegulator, self).__init__()
        self.pad_value = pad_value

    def forward(self, ds, ilens, alpha=1.0, max_len=None):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of sequences of char or phoneme embeddings (B, Tmax, D).
            ds (LongTensor): Batch of durations of each frame (B, T).
            ilens (LongTensor): Batch of input lengths (B,).
            alpha (float, optional): Alpha value to control speed of speech.
        Returns:
            Tensor: replicated input tensor based on durations (B, T*, D).
        """
        assert alpha > 0
        if alpha != 1.0:
            ds = torch.round(ds.float() * alpha).long()
        ds = [d[:ilen] for d, ilen in zip(ds, ilens)]
        mel2ph = [(self._repeat_one_sequence(torch.arange(len(d)), d) + 1) for d in ds]
        return pad_list(mel2ph, 0, max_len).long()

    def _repeat_one_sequence(self, x, d):
        """Repeat each frame according to duration.
        Examples:
            >>> x = torch.tensor([[1], [2], [3]])
            tensor([[1],
                    [2],
                    [3]])
            >>> d = torch.tensor([1, 2, 3])
            tensor([1, 2, 3])
            >>> self._repeat_one_sequence(x, d)
            tensor([[1],
                    [2],
                    [2],
                    [3],
                    [3],
                    [3]])
        """
        if d.sum() == 0:
            logging.warn('all of the predicted durations are 0. fill 0 with 1.')
            d = d.fill_(1)
        return torch.cat([x_.repeat(int(d_), 1) for x_, d_ in zip(x, d) if d_ != 0], dim=0)


DEFAULT_MAX_SOURCE_POSITIONS = 2000


class TransformerEncoder(nn.Module):

    def __init__(self, arch, embed_tokens, last_ln=True, num_layers=None):
        super().__init__()
        self.arch = arch
        self.num_layers = hparams['enc_layers'] if num_layers is None else num_layers
        self.hidden_size = hparams['hidden_size']
        self.embed_tokens = embed_tokens
        self.padding_idx = embed_tokens.padding_idx
        embed_dim = embed_tokens.embedding_dim
        self.dropout = hparams['dropout']
        self.embed_scale = math.sqrt(embed_dim)
        self.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim, self.padding_idx, init_size=self.max_source_positions + self.padding_idx + 1)
        self.layers = nn.ModuleList([])
        self.layers.extend([TransformerEncoderLayer(self.arch[i], self.hidden_size, self.dropout) for i in range(self.num_layers)])
        self.last_ln = last_ln
        if last_ln:
            self.layer_norm = LayerNorm(embed_dim)

    def forward_embedding(self, src_tokens):
        embed = self.embed_scale * self.embed_tokens(src_tokens)
        positions = self.embed_positions(src_tokens)
        x = embed + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def forward(self, src_tokens):
        """

        :param src_tokens: [B, T]
        :return: {
            'encoder_out': [T x B x C]
            'encoder_padding_mask': [B x T]
            'encoder_embedding': [B x T x C]
            'attn_w': []
        }
        """
        x, encoder_embedding = self.forward_embedding(src_tokens)
        x = x.transpose(0, 1)
        encoder_padding_mask = src_tokens.eq(self.padding_idx).data
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=encoder_padding_mask)
        if self.last_ln:
            x = self.layer_norm(x)
            x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        return {'encoder_out': x, 'encoder_padding_mask': encoder_padding_mask, 'encoder_embedding': encoder_embedding, 'attn_w': []}


class BaseModel(nn.Module):

    def __init__(self, arch, dictionary, out_dims=None):
        super().__init__()
        self.dictionary = dictionary
        self.padding_idx = dictionary.pad()
        if isinstance(arch, str):
            self.arch = list(map(int, arch.strip().split()))
        else:
            assert isinstance(arch, (list, tuple))
            self.arch = arch
        self.enc_layers = hparams['enc_layers']
        self.dec_layers = hparams['dec_layers']
        self.enc_arch = self.arch[:self.enc_layers]
        self.dec_arch = self.arch[self.enc_layers:self.enc_layers + self.dec_layers]
        self.hidden_size = hparams['hidden_size']
        self.encoder_embed_tokens = self.build_embedding(self.dictionary, self.hidden_size)
        self.encoder = TransformerEncoder(self.enc_arch, self.encoder_embed_tokens)
        self.decoder = BaseDecoder(self.dec_arch) if hparams['dec_layers'] > 0 else None
        self.mel_out = Linear(self.hidden_size, hparams['audio_num_mel_bins'] if out_dims is None else out_dims, bias=True)
        if hparams['use_spk_id']:
            self.spk_embed_proj = Embedding(hparams['num_spk'], self.hidden_size)
        else:
            self.spk_embed_proj = Linear(256, self.hidden_size, bias=True)
        self.dur_predictor = DurationPredictor(self.hidden_size, n_chans=hparams['predictor_hidden'], dropout_rate=0.5, padding=hparams['ffn_padding'], kernel_size=hparams['dur_predictor_kernel'])
        self.length_regulator = LengthRegulator()

    def build_embedding(self, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        emb = Embedding(num_embeddings, embed_dim, self.padding_idx)
        return emb

    def forward(self, src_tokens):
        raise NotImplementedError


class DurationPredictorLoss(torch.nn.Module):
    """Loss function module for duration predictor.
    The loss value is Calculated in log domain to make it Gaussian.
    """

    def __init__(self, offset=1.0, reduction='none'):
        """Initilize duration predictor loss module.
        Args:
            offset (float, optional): Offset value to avoid nan in log domain.
            reduction (str): Reduction type in loss calculation.
        """
        super(DurationPredictorLoss, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction=reduction)
        self.offset = offset

    def forward(self, outputs, targets, nonpadding):
        """Calculate forward propagation.
        Args:
            outputs (Tensor): Batch of prediction durations in log domain (B, T)
            targets (LongTensor): Batch of groundtruth durations in linear domain (B, T)
        Returns:
            Tensor: Mean squared error loss value.
        Note:
            `outputs` is in log domain but `targets` is in linear domain.
        """
        targets = torch.log(targets.float() + self.offset)
        loss = self.criterion(outputs, targets.float())
        loss = (loss * nonpadding).sum() / nonpadding.sum()
        return loss


class ConvNorm(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Conv2dBlock(nn.Module):

    def __init__(self, idim, odim, kernel_size, stride, padding, norm_fn, acti_fn, dropout_rate=0.0):
        super().__init__()
        self.conv_norm = nn.Sequential(nn.Conv2d(idim, odim, kernel_size, stride, padding), nn.BatchNorm2d(odim))
        if acti_fn == 'relu':
            self.conv_norm.add_module('acti_relu', nn.ReLU(inplace=True))
        self.conv_norm.add_module('do', nn.Dropout(dropout_rate))

    def forward(self, inputs):
        return self.conv_norm(inputs)


class ConvTranspose2dBlock(nn.Module):

    def __init__(self, idim, odim, kernel_size, stride, padding, output_padding, norm_fn, acti_fn, dropout_rate=0.0):
        super().__init__()
        self.trans_conv = nn.Sequential(nn.ConvTranspose2d(idim, odim, kernel_size, stride, padding, output_padding=output_padding), nn.BatchNorm2d(odim))
        if acti_fn == 'relu':
            self.trans_conv.add_module('acti_relu', nn.ReLU(inplace=True))
        self.trans_conv.add_module('do', nn.Dropout(dropout_rate))

    def forward(self, inputs):
        return self.trans_conv(inputs)


class ImageDecoder2D(nn.Module):

    def __init__(self, img_size, input_dim, norm_fn='none', acti_fn='relu'):
        super().__init__()
        self.mini_map_h, self.mini_map_w = self.get_size(img_size, 4)
        self.fc = nn.Linear(input_dim, self.mini_map_h * self.mini_map_w * 256)
        self.dconv1 = ConvTranspose2dBlock(384, 196, 5, stride=2, padding=2, output_padding=1, norm_fn=norm_fn, acti_fn=acti_fn)
        self.dconv2 = ConvTranspose2dBlock(260, 128, 5, stride=2, padding=2, output_padding=1, norm_fn=norm_fn, acti_fn=acti_fn)
        self.dconv3 = ConvTranspose2dBlock(160, 80, 5, stride=2, padding=2, output_padding=1, norm_fn=norm_fn, acti_fn=acti_fn)
        self.dconv4 = ConvTranspose2dBlock(96, 48, 5, stride=2, padding=2, output_padding=1, norm_fn=norm_fn, acti_fn=acti_fn)
        self.dconv5 = Conv2dBlock(48, 16, 5, stride=1, padding=2, norm_fn=norm_fn, acti_fn=acti_fn)
        self.dconv6 = nn.Conv2d(16, 3, 5, stride=1, padding=2)
        """
        self.dconv1 = mn.Sequential(nn.ConvTranspose2d(384,196,5,stride = 2, padding = 2,output_padding - 1),nn.Rl(
            inplace=True))
        self.dconv2 = nn.Sequential(mn.convTranspose2d(260,128,5,stride = 2,, padding = 2,output_padding = 1),nn.ReLu(
            implace=Truel)
        self.dconv3 = nn.Sequential(nn.ConVvTranspose2d(160,80,5,stride = 2, padding = 2,output_padding - 1), nm.RelLu(
            inplace=rue)
        self.dconv4 = nn.Sequential(nn.CconvTranspose2d(96,48,5,stride = 2,padding = 2,output_padting = 1),mn.Relu(
            inplace=True)
        self.dconv5 = nn.Sequential(nn.conv2d(48,16,5,stride = 1,padding = 2),nn.ReLu(inplace=True))
        """

    def get_size(self, img_size, num_layers):
        return tuple([math.ceil(_ / 2 ** num_layers) for _ in img_size])

    def forward(self, concat_z, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4):
        out = self.fc(concat_z)
        out = out.contiguous().view(out.shape[0], 256, self.mini_map_h, self.mini_map_w)
        out = F.relu(out, inplace=True)
        out = torch.cat([out, img_e_conv4], dim=1)
        out = self.dconv1(out)
        out = torch.cat([out, img_e_conv3], dim=1)
        out = self.dconv2(out)
        out = torch.cat([out, img_e_conv2], dim=1)
        out = self.dconv3(out)
        out = torch.cat([out, img_e_conv1], dim=1)
        out = self.dconv4(out)
        out = self.dconv5(out)
        out = self.dconv6(out)
        return torch.tanh(out)


class Conv3dBlock(nn.Module):

    def __init__(self, idim, odim, kernel_size, stride, padding, norm_fn, acti_fn, dropout_rate=0.0):
        super().__init__()
        self.conv_norm = nn.Sequential(nn.Conv3d(idim, odim, kernel_size, stride, padding), nn.BatchNorm3d(odim))
        if acti_fn == 'relu':
            self.conv_norm.add_module('acti_relu', nn.ReLU(inplace=True))
        self.conv_norm.add_module('do', nn.Dropout(dropout_rate))

    def forward(self, inputs):
        return self.conv_norm(inputs)


class ConvTranspose3dBlock(nn.Module):

    def __init__(self, idim, odim, kernel_size, stride, padding, output_padding, norm_fn, acti_fn, dropout_rate=0.0):
        super().__init__()
        self.trans_conv = nn.Sequential(nn.ConvTranspose3d(idim, odim, kernel_size, stride, padding, output_padding=output_padding), nn.BatchNorm3d(odim))
        if acti_fn == 'relu':
            self.trans_conv.add_module('acti_relu', nn.ReLU(inplace=True))
        self.trans_conv.add_module('do', nn.Dropout(dropout_rate))

    def forward(self, inputs):
        return self.trans_conv(inputs)


class ImageDecoder3D(nn.Module):

    def __init__(self, img_channel, img_size, input_dim, norm_fn='none', acti_fn='relu'):
        super().__init__()
        self.mini_map_h, self.mini_map_w = self.get_size(img_size, 4)
        self.fc = nn.Linear(input_dim, self.mini_map_h * self.mini_map_w * 256)
        self.dconv1 = ConvTranspose3dBlock(384, 196, (1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2), output_padding=(0, 1, 1), norm_fn=norm_fn, acti_fn=acti_fn)
        self.dconv2 = ConvTranspose3dBlock(260, 128, (1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2), output_padding=(0, 1, 1), norm_fn=norm_fn, acti_fn=acti_fn)
        self.dconv3 = ConvTranspose3dBlock(160, 80, (1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2), output_padding=(0, 1, 1), norm_fn=norm_fn, acti_fn=acti_fn)
        self.dconv4 = ConvTranspose3dBlock(96, 48, (1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2), output_padding=(0, 1, 1), norm_fn=norm_fn, acti_fn=acti_fn)
        self.dconv5 = Conv3dBlock(48, 16, (1, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2), norm_fn=norm_fn, acti_fn=acti_fn)
        self.dconv6 = nn.Conv3d(16, img_channel, (1, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2))

    def get_size(self, img_size, num_layers):
        return tuple([math.ceil(_ / 2 ** num_layers) for _ in img_size])

    def forward(self, concat_z, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4):
        out = self.fc(concat_z)
        out = out.contiguous().view(out.shape[0], out.shape[1], 256, self.mini_map_h, self.mini_map_w)
        out = out.permute(0, 2, 1, 3, 4)
        out = F.relu(out, inplace=True)
        out = torch.cat([out, img_e_conv4], dim=1)
        out = self.dconv1(out)
        out = torch.cat([out, img_e_conv3], dim=1)
        out = self.dconv2(out)
        out = torch.cat([out, img_e_conv2], dim=1)
        out = self.dconv3(out)
        out = torch.cat([out, img_e_conv1], dim=1)
        out = self.dconv4(out)
        out = self.dconv5(out)
        out = self.dconv6(out)
        return torch.tanh(out)


class ImageEncoder(nn.Module):

    def __init__(self, img_channel, img_size, hidden_size, if_tanh=False, only_lip=False, norm_fn='none', acti_fn='relu'):
        super().__init__()
        self.if_tanh = if_tanh
        img_c = img_channel if only_lip else 6
        self.conv1 = Conv2dBlock(img_c, 16, 5, stride=2, padding=2, norm_fn=norm_fn, acti_fn=acti_fn)
        self.conv2 = Conv2dBlock(16, 32, 5, stride=2, padding=2, norm_fn=norm_fn, acti_fn=acti_fn)
        self.conv3 = Conv2dBlock(32, 64, 5, stride=2, padding=2, norm_fn=norm_fn, acti_fn=acti_fn)
        self.conv4 = Conv2dBlock(64, 128, 5, stride=2, padding=2, norm_fn=norm_fn, acti_fn=acti_fn)
        """
        self.conv1=nn.Sequential(nn.Conv2d(img_c, 16, 5, stride=2, padding=2), nn.ReLu(inplace=True))
        self.conv2=nn.Sequential(nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.ReLu(inplace=True))
        self.conv3=nn.Sequential(nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLu(inplace=True))
        self.conv4=nn.Sequential(nn.Conv2d(64, 128, 5, stride=2, padding=2), nn.ReLu(inplace=True))
        """
        mini_map_h, mini_map_w = self.get_size(img_size, 4)
        self.fc = nn.Linear(mini_map_h * mini_map_w * 128, hidden_size)

    def get_size(self, img_size, num_layers):
        return tuple([math.ceil(_ / 2 ** num_layers) for _ in img_size])

    def forward(self, inputs):
        img_e_conv1 = self.conv1(inputs)
        img_e_conv2 = self.conv2(img_e_conv1)
        img_e_conv3 = self.conv3(img_e_conv2)
        img_e_conv4 = self.conv4(img_e_conv3)
        img_e_fc_5 = img_e_conv4.contiguous().view(img_e_conv4.shape[0], -1)
        img_e_fc_5 = self.fc(img_e_fc_5)
        if self.if_tanh:
            img_e_fc_5 = F.tanh(img_e_fc_5)
        return img_e_fc_5, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4


class FastLip(BaseModel):

    def __init__(self, arch, dictionary, out_dims=None):
        super().__init__(arch, dictionary, out_dims)
        self.img_encoder = ImageEncoder(hparams['img_channel'], [hparams['img_h'], hparams['img_w']], hparams['hidden_size'], if_tanh=True, only_lip=True)
        if hparams['imgdecoder_2D']:
            self.img_decoder = ImageDecoder2D([hparams['img_h'], hparams['img_w']], hparams['hidden_size'] * 2)
        else:
            self.img_decoder = ImageDecoder3D(hparams['img_channel'], [hparams['img_h'], hparams['img_w']], hparams['hidden_size'] * 2)

    def forward(self, src_tokens, vid2ph=None, guide_face=None, skip_decoder=False):
        """

        :param src_tokens: [B, T]
        :param vid2ph:
        :param guide_face: [B, img_h, img_w, C]
        :return: {
            'lip_out': [B, T_s, ?],
            'dur': [B, T_t],
            'w_st_pred': [heads, B, tokens], 'w_st': [heads, B, tokens],
            'encoder_out_noref': [B, T_t, H]
        }
        """
        ret = {}
        encoder_outputs = self.encoder(src_tokens)
        encoder_out = encoder_outputs['encoder_out']
        src_nonpadding = (src_tokens > 0).float().permute(1, 0)[:, :, None]
        encoder_out = encoder_out * src_nonpadding
        dur_input = encoder_out.transpose(0, 1)
        if hparams['predictor_sg']:
            dur_input = dur_input.detach()
        if vid2ph is None:
            dur = self.dur_predictor.inference(dur_input, src_tokens == 0)
            if not hparams['sep_dur_loss']:
                dur[src_tokens == self.dictionary.seg()] = 0
            vid2ph = self.length_regulator(dur, (src_tokens != 0).sum(-1))[..., 0]
        else:
            ret['dur'] = self.dur_predictor(dur_input, src_tokens == 0)
        ret['vid2ph'] = vid2ph
        decoder_inp = F.pad(encoder_out, [0, 0, 0, 0, 1, 0])
        vid2ph_ = vid2ph.permute([1, 0])[..., None].repeat([1, 1, encoder_out.shape[-1]]).contiguous()
        decoder_inp = torch.gather(decoder_inp, 0, vid2ph_).transpose(0, 1)
        ret['decoder_inp_origin'] = decoder_inp
        guide_face = guide_face.permute(0, 3, 1, 2)
        guide_face_embed, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4 = self.img_encoder(guide_face)
        guide_face_embed = guide_face_embed[:, None, :]
        decoder_inp += guide_face_embed
        decoder_inp = decoder_inp * (vid2ph != 0).float()[:, :, None]
        ret['decoder_inp'] = decoder_inp
        if skip_decoder:
            return ret
        x = decoder_inp
        if hparams['dec_layers'] > 0:
            x = self.decoder(x)
        if hparams['imgdecoder_2D']:
            B = x.shape[0]
            seq_len = x.shape[1]
            BB = B * seq_len
            concat_z = torch.cat([guide_face_embed.repeat([1, seq_len, 1]), x], dim=-1).reshape([BB, -1])
            img_e_conv1 = img_e_conv1[:, None, :, :, :].repeat([1, seq_len, 1, 1, 1])
            img_e_conv1 = img_e_conv1.reshape([BB, img_e_conv1.shape[-3], img_e_conv1.shape[-2], img_e_conv1.shape[-1]])
            img_e_conv2 = img_e_conv2[:, None, :, :, :].repeat([1, seq_len, 1, 1, 1])
            img_e_conv2 = img_e_conv2.reshape([BB, img_e_conv2.shape[-3], img_e_conv2.shape[-2], img_e_conv2.shape[-1]])
            img_e_conv3 = img_e_conv3[:, None, :, :, :].repeat([1, seq_len, 1, 1, 1])
            img_e_conv3 = img_e_conv3.reshape([BB, img_e_conv3.shape[-3], img_e_conv3.shape[-2], img_e_conv3.shape[-1]])
            img_e_conv4 = img_e_conv4[:, None, :, :, :].repeat([1, seq_len, 1, 1, 1])
            img_e_conv4 = img_e_conv4.reshape([BB, img_e_conv4.shape[-3], img_e_conv4.shape[-2], img_e_conv4.shape[-1]])
            output_frames = self.img_decoder(concat_z, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4)
            output_frames = output_frames.reshape([B, seq_len, output_frames.shape[-3], output_frames.shape[-2], output_frames.shape[-1]])
            x = output_frames.permute(0, 1, 3, 4, 2)
        else:
            seq_len = x.shape[1]
            concat_z = torch.cat([guide_face_embed.repeat([1, seq_len, 1]), x], dim=-1)
            img_e_conv1 = img_e_conv1[:, :, None, :, :].repeat([1, 1, seq_len, 1, 1])
            img_e_conv2 = img_e_conv2[:, :, None, :, :].repeat([1, 1, seq_len, 1, 1])
            img_e_conv3 = img_e_conv3[:, :, None, :, :].repeat([1, 1, seq_len, 1, 1])
            img_e_conv4 = img_e_conv4[:, :, None, :, :].repeat([1, 1, seq_len, 1, 1])
            output_frames = self.img_decoder(concat_z, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4)
            x = output_frames.permute(0, 2, 3, 4, 1)
        x = x * (vid2ph != 0).float()[:, :, None, None, None]
        ret['lip_out'] = x
        return ret


class SelfAttention(nn.Module):

    def __init__(self, hid_dim, n_heads, dropout=0.1, gaussian_bias=False, gaussian_tao=None, gaus_init_l=3000):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0
        self.w_q = Linear(hid_dim, hid_dim)
        self.w_k = Linear(hid_dim, hid_dim)
        self.w_v = Linear(hid_dim, hid_dim)
        self.gaussian_bias = gaussian_bias
        if gaussian_bias:
            self.tao = nn.Parameter(torch.FloatTensor(n_heads))
            nn.init.constant_(self.tao, gaussian_tao)
            self.bias_matrix = torch.Tensor([[(-abs(i - j) ** 2 / 2.0) for i in range(gaus_init_l)] for j in range(gaus_init_l)])
        self.fc = Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.sqrt_d = (hid_dim // n_heads) ** -0.5

    def forward(self, query, key, value, mask=None, require_w=False):
        for m in [query, key, value]:
            m.transpose_(0, 1)
        bsz, length, emb_dim = query.shape
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads)
        Q = Q.permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        QK = torch.matmul(Q, K.transpose(2, 3)) * self.sqrt_d
        if self.gaussian_bias:
            L = QK.size(-1)
            if L <= self.bias_matrix.size(0):
                gaussian_mask = self.bias_matrix[:L, :L].repeat(self.n_heads, 1, 1)
            else:
                gaussian_mask = torch.tensor([[(-abs(i - j) ** 2 / 2.0) for i in range(L)] for j in range(L)]).repeat(self.n_heads, 1, 1)
                None
            gaussian_mask = torch.mul(gaussian_mask, torch.pow(self.tao, -4)[:, None, None])
            QK += gaussian_mask.repeat(bsz, 1, 1, 1)
        if mask is not None:
            """
            attn weight size: b*n_h, L, L
            mask size: b, L -> b, 1, 1, L + 
            attn_weight(b,n_head,L,L).masked_fill(mask)
            """
            QK = QK.masked_fill(mask[:, None, None, :], float('-inf'))
        attn_weight = torch.softmax(QK, dim=-1)
        attention = self.dropout(attn_weight)
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        x.transpose_(0, 1)
        return (x, attn_weight[:1, :1, ...]) if require_w else (x, None)


class CyclicalPositionEmb(nn.Module):

    def __init__(self, K, emb_size):
        super(CyclicalPositionEmb, self).__init__()
        self.fc = Linear(K, emb_size)

    def forward(self, x):
        """
        :param x: B * T * 1
        :return: x
        """
        pass


class ConvAttentionLayer(nn.Module):

    def __init__(self, c, hidden_size, dropout=0.0):
        super().__init__()
        self.in_projection = Linear(c, hidden_size)
        self.out_projection = Linear(hidden_size, c)
        self.dropout = dropout

    def forward(self, x, key, value, encoder_padding_mask=None, enc_dec_attn_constraint_mask=None):
        query = self.in_projection(x)
        attn_weights = torch.bmm(query.transpose(0, 1), key.transpose(0, 1).transpose(1, 2))
        if encoder_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(encoder_padding_mask.unsqueeze(1), float('-inf')).type_as(attn_weights)
        if enc_dec_attn_constraint_mask is not None:
            attn_weights = attn_weights.masked_fill(enc_dec_attn_constraint_mask.bool(), float('-inf')).type_as(attn_weights)
        attn_logits = attn_weights
        sz = attn_weights.size()
        attn_scores = F.softmax(attn_weights.view(sz[0] * sz[1], sz[2]), dim=1)
        attn_scores = attn_scores.view(sz)
        attn_scores = F.dropout(attn_scores, p=self.dropout, training=self.training)
        attn = torch.bmm(attn_scores, value.transpose(0, 1)).transpose(0, 1)
        s = value.size(0)
        if encoder_padding_mask is None:
            attn = attn * (s * math.sqrt(1.0 / s))
        else:
            s = s - encoder_padding_mask.type_as(attn).sum(dim=1, keepdim=True)
            s = s.transpose(0, 1).unsqueeze(-1)
            attn = attn * (s * s.rsqrt())
        attn = self.out_projection(attn)
        return attn, attn_scores, attn_logits


class LinearizedConvolution(ConvTBC):
    """An optimized version of nn.Conv1d.

    At training time, this module uses ConvTBC, which is an optimized version
    of Conv1d. At inference time, it optimizes incremental generation (i.e.,
    one time step at a time) by replacing the convolutions with linear layers.
    Note that the input order changes from training to inference.
    """

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self._linearized_weight = None
        self.register_backward_hook(self._clear_linearized_weight)

    def forward(self, input, incremental_state=None):
        """
        Args:
            incremental_state: Used to buffer signal; if not None, then input is
                expected to contain a single frame. If the input order changes
                between time steps, call reorder_incremental_state.
        Input:
            Time x Batch x Channel
        """
        if incremental_state is None:
            output = super().forward(input)
            if self.kernel_size > 1 and self.padding > 0:
                output = output[:-self.padding, :, :]
            return output
        weight = self._get_linearized_weight()
        kw = self.kernel_size
        input = input.transpose(0, 1)
        bsz = input.size(0)
        if kw > 1:
            input = input.data
            input_buffer = self._get_input_buffer(incremental_state)
            if input_buffer is None:
                input_buffer = input.new(bsz, kw, input.size(2)).zero_()
                self._set_input_buffer(incremental_state, input_buffer)
            else:
                input_buffer[:, :-1, :] = input_buffer[:, 1:, :].clone()
            input_buffer[:, -1, :] = input[:, -1, :]
            input = input_buffer
        with torch.no_grad():
            output = F.linear(input.view(bsz, -1), weight, self.bias)
        return output.view(bsz, 1, -1).transpose(0, 1)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(self, incremental_state, 'input_buffer')

    def _set_input_buffer(self, incremental_state, new_buffer):
        return utils.set_incremental_state(self, incremental_state, 'input_buffer', new_buffer)

    def _get_linearized_weight(self):
        if self._linearized_weight is None:
            kw = self.kernel_size
            weight = self.weight.transpose(2, 1).transpose(1, 0).contiguous()
            assert weight.size() == (self.out_channels, kw, self.in_channels)
            self._linearized_weight = weight.view(self.out_channels, -1)
        return self._linearized_weight

    def _clear_linearized_weight(self, *args):
        self._linearized_weight = None

    def clear_buffer(self, input, incremental_state=None):
        if incremental_state is not None:
            self._set_input_buffer(incremental_state, None)


class DecConvLayer(nn.Module):

    def __init__(self, c, kernel_size, dropout, attention_dropout=0.1):
        super().__init__()
        self.layer_norm1 = LayerNorm(c)
        conv = LinearizedConvolution(c, c, kernel_size, padding=kernel_size - 1)
        std = math.sqrt(4 * (1.0 - dropout) / (kernel_size * c))
        nn.init.normal_(conv.weight, mean=0, std=std)
        nn.init.constant_(conv.bias, 0)
        self.conv = nn.utils.weight_norm(conv, dim=2)
        self.layer_norm2 = LayerNorm(c)
        self.attention = MultiheadAttention(c, 1, dropout=attention_dropout, encoder_decoder_attention=True, bias=False)
        self.dropout = dropout

    def forward(self, x, encoder_out=None, encoder_padding_mask=None, incremental_state=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
        residual = x
        x = self.layer_norm1(x)
        x = self.conv(x, incremental_state=incremental_state)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = self.layer_norm2(x)
        x, attn = self.attention(query=x, key=encoder_out, value=encoder_out, key_padding_mask=encoder_padding_mask, incremental_state=incremental_state, static_kv=True, enc_dec_attn_constraint_mask=utils.get_incremental_state(self, incremental_state, 'enc_dec_attn_constraint_mask'))
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        attn_logits = attn[1]
        return x, attn_logits

    def clear_buffer(self, input, encoder_out=None, encoder_padding_mask=None, incremental_state=None):
        self.conv.clear_buffer(input, incremental_state)

    def set_buffer(self, name, tensor, incremental_state):
        return utils.set_incremental_state(self, incremental_state, name, tensor)


class DecSALayer(nn.Module):

    def __init__(self, c, num_heads, dropout, attention_dropout=0.1, relu_dropout=0.1, kernel_size=9):
        super().__init__()
        self.c = c
        self.dropout = dropout
        self.layer_norm1 = LayerNorm(c)
        self.self_attn = MultiheadAttention(c, num_heads, self_attention=True, dropout=attention_dropout, bias=False)
        self.layer_norm2 = LayerNorm(c)
        self.encoder_attn = MultiheadAttention(c, num_heads, encoder_decoder_attention=True, dropout=attention_dropout, bias=False)
        self.layer_norm3 = LayerNorm(c)
        self.ffn = NewTransformerFFNLayer(c, 4 * c, padding='LEFT', kernel_size=kernel_size, dropout=relu_dropout)

    def forward(self, x, encoder_out=None, encoder_padding_mask=None, incremental_state=None, self_attn_mask=None, self_attn_padding_mask=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
            self.layer_norm3.training = layer_norm_training
        residual = x
        x = self.layer_norm1(x)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=self_attn_padding_mask, incremental_state=incremental_state, attn_mask=self_attn_mask)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        residual = x
        x = self.layer_norm2(x)
        x, attn = self.encoder_attn(query=x, key=encoder_out, value=encoder_out, key_padding_mask=encoder_padding_mask, incremental_state=incremental_state, static_kv=True, enc_dec_attn_constraint_mask=utils.get_incremental_state(self, incremental_state, 'enc_dec_attn_constraint_mask'))
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        residual = x
        x = self.layer_norm3(x)
        x = self.ffn(x, incremental_state=incremental_state)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        attn_logits = attn[1]
        return x, attn_logits

    def clear_buffer(self, input, encoder_out=None, encoder_padding_mask=None, incremental_state=None):
        self.encoder_attn.clear_buffer(incremental_state)
        self.ffn.clear_buffer(incremental_state)

    def set_buffer(self, name, tensor, incremental_state):
        return utils.set_incremental_state(self, incremental_state, name, tensor)


class LSTMAttentionLayer(nn.Module):

    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False, dropout=0.0):
        super().__init__()
        self.input_proj = Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = Linear(input_embed_dim + source_embed_dim, output_embed_dim, bias=bias)
        self.dropout = dropout

    def forward(self, input, source_hids, encoder_padding_mask=None, enc_dec_attn_constraint_mask=None):
        x = self.input_proj(input)
        attn_weights = torch.bmm(x.transpose(0, 1), source_hids.transpose(0, 1).transpose(1, 2))
        if encoder_padding_mask is not None:
            attn_weights = attn_weights.float().masked_fill_(encoder_padding_mask.unsqueeze(1), float('-inf')).type_as(attn_weights)
        if enc_dec_attn_constraint_mask is not None:
            attn_weights = attn_weights.float().masked_fill_(enc_dec_attn_constraint_mask.bool(), float('-inf')).type_as(attn_weights)
        attn_logits = attn_weights
        sz = attn_weights.size()
        attn_scores = F.softmax(attn_weights.view(sz[0] * sz[1], sz[2]), dim=1)
        attn_scores = attn_scores.view(sz)
        attn_scores = F.dropout(attn_scores, p=self.dropout, training=self.training)
        attn = torch.bmm(attn_scores, source_hids.transpose(0, 1)).transpose(0, 1)
        x = torch.tanh(self.output_proj(torch.cat((attn, input), dim=-1)))
        return x, attn_scores, attn_logits


class DecLSTMLayer(nn.Module):

    def __init__(self, c, dropout, attention_dropout=0.1):
        super().__init__()
        self.c = c
        self.layer_norm1 = LayerNorm(c)
        self.lstm = nn.LSTM(c, c, 1, dropout=dropout)
        self.layer_norm2 = LayerNorm(c)
        self.attention = MultiheadAttention(c, 1, dropout=attention_dropout, encoder_decoder_attention=True, bias=False)
        self.dropout = dropout

    def forward(self, x, encoder_out=None, encoder_padding_mask=None, incremental_state=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
        self.lstm.flatten_parameters()
        if incremental_state is not None:
            x = x[-1:, :, :]
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells = cached_state
        else:
            prev_hiddens = encoder_out.mean(dim=0, keepdim=True)
            prev_cells = encoder_out.mean(dim=0, keepdim=True)
        residual = x
        x = self.layer_norm1(x)
        x, hidden = self.lstm(x, (prev_hiddens, prev_cells))
        hiddens, cells = hidden
        x = residual + x
        x = self.layer_norm2(x)
        x, attn = self.attention(query=x, key=encoder_out, value=encoder_out, key_padding_mask=encoder_padding_mask, incremental_state=incremental_state, static_kv=True, enc_dec_attn_constraint_mask=utils.get_incremental_state(self, incremental_state, 'enc_dec_attn_constraint_mask'))
        x = F.dropout(x, self.dropout, training=self.training)
        if incremental_state is not None:
            prev_hiddens = hiddens
            prev_cells = cells
            utils.set_incremental_state(self, incremental_state, 'cached_state', (prev_hiddens, prev_cells))
        x = residual + x
        attn_logits = attn[1]
        return x, attn_logits

    def clear_buffer(self, input, encoder_out=None, encoder_padding_mask=None, incremental_state=None):
        if incremental_state is not None:
            prev_hiddens = encoder_out.mean(dim=0, keepdim=True)
            prev_cells = encoder_out.mean(dim=0, keepdim=True)
            utils.set_incremental_state(self, incremental_state, 'cached_state', (prev_hiddens, prev_cells))

    def set_buffer(self, name, tensor, incremental_state):
        return utils.set_incremental_state(self, incremental_state, name, tensor)


def _find_tensors(obj):
    """
    Recursively find all tensors contained in the specified object.
    """
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return itertools.chain(*map(_find_tensors, obj))
    if isinstance(obj, dict):
        return itertools.chain(*map(_find_tensors, obj.values()))
    return []


class DDP(DistributedDataParallel):
    """
    Override the forward call in lightning so it goes to training and validation step respectively
    """

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def forward(self, *inputs, **kwargs):
        self._sync_params()
        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                if self.module.training:
                    output = self.module.training_step(*inputs[0], **kwargs[0])
                elif self.module.testing:
                    output = self.module.test_step(*inputs[0], **kwargs[0])
                else:
                    output = self.module.validation_step(*inputs[0], **kwargs[0])
            else:
                outputs = self.parallel_apply(self._module_copies[:len(inputs)], inputs, kwargs)
                output = self.gather(outputs, self.output_device)
        else:
            output = self.module(*inputs, **kwargs)
        if torch.is_grad_enabled():
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        return output


class DP(DataParallel):
    """
    Override the forward call in lightning so it goes to training and validation step respectively
    """

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        for t in itertools.chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError('module must have its parameters and buffers on device {} (device_ids[0]) but found one of them on device: {}'.format(self.src_device_obj, t.device))
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            if self.module.training:
                return self.module.training_step(*inputs[0], **kwargs[0])
            elif self.module.testing:
                return self.module.test_step(*inputs[0], **kwargs[0])
            else:
                return self.module.validation_step(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])


class GradientAccumulationScheduler:

    def __init__(self, scheduling: 'dict'):
        if scheduling == {}:
            raise TypeError('Empty dict cannot be interpreted correct')
        for key in scheduling.keys():
            if not isinstance(key, int) or not isinstance(scheduling[key], int):
                raise TypeError('All epoches and accumulation factor must be integers')
        minimal_epoch = min(scheduling.keys())
        if minimal_epoch < 1:
            msg = f'Epochs indexing from 1, epoch {minimal_epoch} cannot be interpreted correct'
            raise IndexError(msg)
        elif minimal_epoch != 1:
            scheduling.update({(1): 1})
        self.scheduling = scheduling
        self.epochs = sorted(scheduling.keys())

    def on_epoch_begin(self, epoch, trainer):
        epoch += 1
        for i in reversed(range(len(self.epochs))):
            if epoch >= self.epochs[i]:
                trainer.accumulate_grad_batches = self.scheduling.get(self.epochs[i])
                break


class BaseTrainer:

    def __init__(self, logger=True, checkpoint_callback=True, default_save_path=None, gradient_clip_val=0, process_position=0, gpus=-1, log_gpu_memory=None, show_progress_bar=True, track_grad_norm=-1, check_val_every_n_epoch=1, accumulate_grad_batches=1, max_updates=1000, min_epochs=1, val_check_interval=1.0, log_save_interval=100, row_log_interval=10, print_nan_grads=False, weights_summary='full', num_sanity_val_steps=100, resume_from_checkpoint=None):
        self.log_gpu_memory = log_gpu_memory
        self.gradient_clip_val = gradient_clip_val
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.track_grad_norm = track_grad_norm
        self.on_gpu = True if gpus and torch.cuda.is_available() else False
        self.process_position = process_position
        self.weights_summary = weights_summary
        self.max_updates = max_updates
        self.min_epochs = min_epochs
        self.num_sanity_val_steps = num_sanity_val_steps
        self.print_nan_grads = print_nan_grads
        self.resume_from_checkpoint = resume_from_checkpoint
        self.default_save_path = default_save_path
        self.total_batch_idx = 0
        self.running_loss = []
        self.avg_loss = 0
        self.batch_idx = 0
        self.tqdm_metrics = {}
        self.callback_metrics = {}
        self.num_val_batches = 0
        self.num_training_batches = 0
        self.num_test_batches = 0
        self.get_train_dataloader = None
        self.get_test_dataloaders = None
        self.get_val_dataloaders = None
        self.is_iterable_train_dataloader = False
        self.model = None
        self.testing = False
        self.disable_validation = False
        self.lr_schedulers = []
        self.optimizers = None
        self.global_step = 0
        self.current_epoch = 0
        self.total_batches = 0
        self.checkpoint_callback = checkpoint_callback
        self.checkpoint_callback.save_function = self.save_checkpoint
        self.weights_save_path = self.checkpoint_callback.filepath
        self.configure_accumulated_gradients(accumulate_grad_batches)
        self.data_parallel_device_ids = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        self.root_gpu = self.data_parallel_device_ids[0]
        self.use_ddp = False
        self.use_dp = False
        self.single_gpu = False
        self.distributed_backend = 'ddp' if self.num_gpus > 0 else 'dp'
        self.set_distributed_mode(self.distributed_backend)
        self.proc_rank = 0
        self.world_size = 1
        self.node_rank = 0
        self.show_progress_bar = show_progress_bar
        self.log_save_interval = log_save_interval
        self.val_check_interval = val_check_interval
        self.logger = logger
        self.logger.rank = 0
        self.row_log_interval = row_log_interval

    @property
    def num_gpus(self):
        gpus = self.data_parallel_device_ids
        if gpus is None:
            return 0
        else:
            return len(gpus)

    @property
    def data_parallel(self):
        return self.use_dp or self.use_ddp

    def get_model(self):
        is_dp_module = isinstance(self.model, (DDP, DP))
        model = self.model.module if is_dp_module else self.model
        return model

    def fit(self, model):
        if self.use_ddp:
            mp.spawn(self.ddp_train, nprocs=self.num_gpus, args=(model,))
        elif self.use_dp:
            self.dp_train(model)
        elif self.single_gpu:
            self.single_gpu_train(model)
        else:
            assert False, 'GPU not found'
        return 1

    def init_optimizers(self, optimizers):
        if isinstance(optimizers, Optimizer):
            return [optimizers], []
        elif len(optimizers) == 2 and isinstance(optimizers[0], list):
            optimizers, lr_schedulers = optimizers
            return optimizers, lr_schedulers
        elif isinstance(optimizers, list) or isinstance(optimizers, tuple):
            return optimizers, []

    def run_pretrain_routine(self, model):
        """Sanity check a few things before starting actual training.

        :param model:
        """
        ref_model = model
        if self.data_parallel:
            ref_model = model.module
        ref_model.trainer = self
        self.copy_trainer_model_properties(ref_model)
        if self.logger is not None:
            ref_model.logger = self.logger
            self.logger.save()
        if self.use_ddp:
            dist.barrier()
        self.get_dataloaders(ref_model)
        self.model = model
        self.restore_weights(model)
        if self.testing:
            self.run_evaluation(test=True)
            return
        self.disable_validation = self.num_val_batches == 0
        ref_model.on_sanity_check_start()
        ref_model.on_train_start()
        if not self.disable_validation and self.num_sanity_val_steps > 0:
            pbar = tqdm.tqdm(desc='Validation sanity check', total=self.num_sanity_val_steps * len(self.get_val_dataloaders()), leave=False, position=2 * self.process_position, disable=not self.show_progress_bar, dynamic_ncols=True, unit='batch')
            self.main_progress_bar = pbar
            self.val_progress_bar = tqdm.tqdm(disable=True)
            self.evaluate(model, self.get_val_dataloaders(), self.num_sanity_val_steps, self.testing)
            self.main_progress_bar.close()
            self.val_progress_bar.close()
        pbar = tqdm.tqdm(leave=True, position=2 * self.process_position, disable=not self.show_progress_bar, dynamic_ncols=True, unit='batch', file=sys.stdout)
        self.main_progress_bar = pbar
        if self.on_gpu:
            torch.cuda.empty_cache()
        self.train()

    def test(self, model):
        self.testing = True
        self.fit(model)

    @property
    def training_tqdm_dict(self):
        tqdm_dict = {'step': '{}'.format(self.global_step)}
        tqdm_dict.update(self.tqdm_metrics)
        return tqdm_dict

    def restore_weights(self, model):
        """
        To restore weights we have two cases.
        First, attempt to restore hpc weights. If successful, don't restore
        other weights.

        Otherwise, try to restore actual weights
        :param model:
        :return:
        """
        if self.on_gpu:
            torch.cuda.empty_cache()
        if self.resume_from_checkpoint is not None:
            self.restore(self.resume_from_checkpoint, on_gpu=self.on_gpu)
        else:
            self.restore_state_if_checkpoint_exists(model)
        if self.use_ddp:
            dist.barrier()
        if self.on_gpu:
            torch.cuda.empty_cache()

    def restore_state_if_checkpoint_exists(self, model):
        did_restore = False
        no_ckpt_callback = self.checkpoint_callback is None or not self.checkpoint_callback
        if no_ckpt_callback or not os.path.exists(self.checkpoint_callback.filepath):
            return did_restore
        last_steps = -1
        last_ckpt_name = None
        checkpoints = os.listdir(self.checkpoint_callback.filepath)
        for name in checkpoints:
            if '.ckpt' in name:
                if 'steps_' in name:
                    steps = name.split('steps_')[1]
                    steps = int(re.sub('[^0-9]', '', steps))
                    if steps > last_steps:
                        last_steps = steps
                        last_ckpt_name = name
        if last_ckpt_name is not None:
            last_ckpt_path = os.path.join(self.checkpoint_callback.filepath, last_ckpt_name)
            self.restore(last_ckpt_path, self.on_gpu)
            logging.info(f'model and trainer restored from checkpoint: {last_ckpt_path}')
            did_restore = True
        return did_restore

    def restore(self, checkpoint_path, on_gpu):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = self.get_model()
        model.load_state_dict(checkpoint['state_dict'])
        if on_gpu:
            model
        self.restore_training_state(checkpoint)
        model.global_step = self.global_step
        del checkpoint
        try:
            if dist.is_initialized() and dist.get_rank() > 0:
                return
        except Exception as e:
            None
            return

    def restore_training_state(self, checkpoint):
        """
        Restore trainer state.
        Model will get its change to update
        :param checkpoint:
        :return:
        """
        if self.checkpoint_callback is not None and self.checkpoint_callback is not False:
            self.checkpoint_callback.best = checkpoint['checkpoint_callback_best']
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['epoch']
        optimizer_states = checkpoint['optimizer_states']
        for optimizer, opt_state in zip(self.optimizers, optimizer_states):
            optimizer.load_state_dict(opt_state)
            if self.root_gpu is not None:
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v
        lr_schedulers = checkpoint['lr_schedulers']
        for scheduler, lrs_state in zip(self.lr_schedulers, lr_schedulers):
            scheduler.load_state_dict(lrs_state)

    def _atomic_save(self, checkpoint, filepath):
        """Saves a checkpoint atomically, avoiding the creation of incomplete checkpoints.

        This will create a temporary checkpoint with a suffix of ``.part``, then copy it to the final location once
        saving is finished.

        Args:
            checkpoint (object): The object to save.
                Built to be used with the ``dump_checkpoint`` method, but can deal with anything which ``torch.save``
                accepts.
            filepath (str|pathlib.Path): The path to which the checkpoint will be saved.
                This points to the file that the checkpoint will be stored in.
        """
        tmp_path = str(filepath) + '.part'
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, filepath)

    def save_checkpoint(self, filepath):
        checkpoint = self.dump_checkpoint()
        self._atomic_save(checkpoint, filepath)

    def dump_checkpoint(self):
        checkpoint = {'epoch': self.current_epoch, 'global_step': self.global_step}
        if self.checkpoint_callback is not None and self.checkpoint_callback is not False:
            checkpoint['checkpoint_callback_best'] = self.checkpoint_callback.best
        optimizer_states = []
        for i, optimizer in enumerate(self.optimizers):
            optimizer_states.append(optimizer.state_dict())
        checkpoint['optimizer_states'] = optimizer_states
        lr_schedulers = []
        for i, scheduler in enumerate(self.lr_schedulers):
            lr_schedulers.append(scheduler.state_dict())
        checkpoint['lr_schedulers'] = lr_schedulers
        model = self.get_model()
        checkpoint['state_dict'] = model.state_dict()
        model.on_save_checkpoint(checkpoint)
        return checkpoint

    def copy_trainer_model_properties(self, model):
        if isinstance(model, DP):
            ref_model = model.module
        elif isinstance(model, DDP):
            ref_model = model.module
        else:
            ref_model = model
        for m in [model, ref_model]:
            m.trainer = self
            m.on_gpu = self.on_gpu
            m.use_dp = self.use_dp
            m.use_ddp = self.use_ddp
            m.testing = self.testing
            m.single_gpu = self.single_gpu

    def transfer_batch_to_gpu(self, batch, gpu_id):
        if callable(getattr(batch, 'cuda', None)):
            return batch
        elif callable(getattr(batch, 'to', None)):
            return batch
        elif isinstance(batch, list):
            for i, x in enumerate(batch):
                batch[i] = self.transfer_batch_to_gpu(x, gpu_id)
            return batch
        elif isinstance(batch, tuple):
            batch = list(batch)
            for i, x in enumerate(batch):
                batch[i] = self.transfer_batch_to_gpu(x, gpu_id)
            return tuple(batch)
        elif isinstance(batch, dict):
            for k, v in batch.items():
                batch[k] = self.transfer_batch_to_gpu(v, gpu_id)
            return batch
        return batch

    def single_gpu_train(self, model):
        self.optimizers, self.lr_schedulers = self.init_optimizers(model.configure_optimizers())
        model
        self.run_pretrain_routine(model)

    def dp_train(self, model):
        self.optimizers, self.lr_schedulers = self.init_optimizers(model.configure_optimizers())
        model
        device_ids = self.data_parallel_device_ids
        model = DP(model, device_ids=device_ids)
        self.run_pretrain_routine(model)

    def set_distributed_mode(self, distributed_backend):
        if self.num_gpus == 0:
            return
        elif self.num_gpus == 1:
            self.single_gpu = True
            self.use_dp = False
            self.use_ddp = False
            self.root_gpu = 0
            self.data_parallel_device_ids = [0]
        elif distributed_backend is not None:
            self.use_dp = distributed_backend == 'dp'
            self.use_ddp = distributed_backend == 'ddp'
        elif distributed_backend is None:
            self.use_dp = True
            self.use_ddp = False
        logging.info(f'gpu available: {torch.cuda.is_available()}, used: {self.on_gpu}')

    def ddp_train(self, gpu_idx, model):
        """
        Entry point into a DP thread
        :param gpu_idx:
        :param model:
        :param cluster_obj:
        :return:
        """
        self.node_rank = 0
        self.show_progress_bar = self.show_progress_bar and self.node_rank == 0 and gpu_idx == 0
        if self.use_ddp:
            self.proc_rank = self.node_rank * self.num_gpus + gpu_idx
            self.world_size = self.num_gpus
        if self.logger is not None:
            self.logger.rank = self.proc_rank
        model.trainer = self
        model.init_ddp_connection(self.proc_rank, self.world_size)
        self.optimizers, self.lr_schedulers = self.init_optimizers(model.configure_optimizers())
        if self.distributed_backend == 'ddp':
            torch.cuda.set_device(gpu_idx)
        model
        self.copy_trainer_model_properties(model)
        self.root_gpu = gpu_idx
        if self.distributed_backend == 'ddp':
            device_ids = [gpu_idx]
        else:
            device_ids = None
        model = model.configure_ddp(model, device_ids)
        self.run_pretrain_routine(model)

    def resolve_root_node_address(self, root_node):
        if '[' in root_node:
            name = root_node.split('[')[0]
            number = root_node.split(',')[0]
            if '-' in number:
                number = number.split('-')[0]
            number = re.sub('[^0-9]', '', number)
            root_node = name + number
        return root_node

    def log_metrics(self, metrics, grad_norm_dic, step=None):
        """Logs the metric dict passed in.

        :param metrics:
        :param grad_norm_dic:
        """
        metrics['epoch'] = self.current_epoch
        metrics.update(grad_norm_dic)
        scalar_metrics = self.metrics_to_scalars(metrics)
        step = step if step is not None else self.global_step
        if self.proc_rank == 0 and self.logger is not None:
            self.logger.log_metrics(scalar_metrics, step=step)
            self.logger.save()

    def add_tqdm_metrics(self, metrics):
        for k, v in metrics.items():
            if type(v) is torch.Tensor:
                v = v.item()
            self.tqdm_metrics[k] = v

    def metrics_to_scalars(self, metrics):
        new_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if type(v) is dict:
                v = self.metrics_to_scalars(v)
            new_metrics[k] = v
        return new_metrics

    def process_output(self, output, train=False):
        """Reduces output according to the training mode.

        Separates loss from logging and tqdm metrics
        :param output:
        :return:
        """
        callback_metrics = {}
        for k, v in output.items():
            if k not in ['progress_bar', 'log', 'hiddens']:
                callback_metrics[k] = v
        if train and self.use_dp:
            num_gpus = self.num_gpus
            callback_metrics = self.reduce_distributed_output(callback_metrics, num_gpus)
        for k, v in callback_metrics.items():
            if isinstance(v, torch.Tensor):
                callback_metrics[k] = v.item()
        try:
            progress_output = output['progress_bar']
            if train and self.use_dp:
                num_gpus = self.num_gpus
                progress_output = self.reduce_distributed_output(progress_output, num_gpus)
            progress_bar_metrics = progress_output
        except Exception:
            progress_bar_metrics = {}
        try:
            log_output = output['log']
            if train and self.use_dp:
                num_gpus = self.num_gpus
                log_output = self.reduce_distributed_output(log_output, num_gpus)
            log_metrics = log_output
        except Exception:
            log_metrics = {}
        loss = None
        if train:
            try:
                loss = output['loss']
            except Exception:
                if type(output) is torch.Tensor:
                    loss = output
                else:
                    raise RuntimeError('No `loss` value in the dictionary returned from `model.training_step()`.')
            if self.use_dp:
                loss = self.reduce_distributed_output(loss, self.num_gpus)
        hiddens = output.get('hiddens')
        callback_metrics.update(progress_bar_metrics)
        callback_metrics.update(log_metrics)
        for k, v in callback_metrics.items():
            if isinstance(v, torch.Tensor):
                callback_metrics[k] = v.item()
        return loss, progress_bar_metrics, log_metrics, callback_metrics, hiddens

    def reduce_distributed_output(self, output, num_gpus):
        if num_gpus <= 1:
            return output
        if type(output) is torch.Tensor:
            return output.mean()
        for k, v in output.items():
            if isinstance(output[k], dict):
                output[k] = self.reduce_distributed_output(output[k], num_gpus)
            elif isinstance(output[k], torch.Tensor) and output[k].dim() == 0:
                pass
            elif output[k].size(0) == num_gpus:
                reduced = torch.mean(output[k])
                output[k] = reduced
        return output

    def clip_gradients(self):
        if self.gradient_clip_val > 0:
            model = self.get_model()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_val)

    def print_nan_gradients(self):
        model = self.get_model()
        for param in model.parameters():
            if param.grad is not None and torch.isnan(param.grad.float()).any():
                logging.info(param, param.grad)

    def configure_accumulated_gradients(self, accumulate_grad_batches):
        self.accumulate_grad_batches = None
        if isinstance(accumulate_grad_batches, dict):
            self.accumulation_scheduler = GradientAccumulationScheduler(accumulate_grad_batches)
        elif isinstance(accumulate_grad_batches, int):
            schedule = {(1): accumulate_grad_batches}
            self.accumulation_scheduler = GradientAccumulationScheduler(schedule)
        else:
            raise TypeError('Gradient accumulation supports only int and dict types')

    def get_dataloaders(self, model):
        if not self.testing:
            self.init_train_dataloader(model)
            self.init_val_dataloader(model)
        self.init_test_dataloader(model)
        if self.use_ddp:
            dist.barrier()
            if not self.testing:
                self.get_train_dataloader()
                self.get_val_dataloaders()
            self.get_test_dataloaders()

    def init_train_dataloader(self, model):
        self.fisrt_epoch = True
        self.get_train_dataloader = model.train_dataloader
        if isinstance(self.get_train_dataloader(), torch.utils.data.DataLoader):
            self.num_training_batches = len(self.get_train_dataloader())
            self.num_training_batches = int(self.num_training_batches)
        else:
            self.num_training_batches = float('inf')
            self.is_iterable_train_dataloader = True
        if isinstance(self.val_check_interval, int):
            self.val_check_batch = self.val_check_interval
        else:
            self._percent_range_check('val_check_interval')
            self.val_check_batch = int(self.num_training_batches * self.val_check_interval)
            self.val_check_batch = max(1, self.val_check_batch)

    def init_val_dataloader(self, model):
        self.get_val_dataloaders = model.val_dataloader
        self.num_val_batches = 0
        if self.get_val_dataloaders() is not None:
            if isinstance(self.get_val_dataloaders()[0], torch.utils.data.DataLoader):
                self.num_val_batches = sum(len(dataloader) for dataloader in self.get_val_dataloaders())
                self.num_val_batches = int(self.num_val_batches)
            else:
                self.num_val_batches = float('inf')

    def init_test_dataloader(self, model):
        self.get_test_dataloaders = model.test_dataloader
        if self.get_test_dataloaders() is not None:
            if isinstance(self.get_test_dataloaders()[0], torch.utils.data.DataLoader):
                self.num_test_batches = sum(len(dataloader) for dataloader in self.get_test_dataloaders())
                self.num_test_batches = int(self.num_test_batches)
            else:
                self.num_test_batches = float('inf')

    def evaluate(self, model, dataloaders, max_batches, test=False):
        """Run evaluation code.

        :param model: PT model
        :param dataloaders: list of PT dataloaders
        :param max_batches: Scalar
        :param test: boolean
        :return:
        """
        model.zero_grad()
        model.eval()
        self.copy_trainer_model_properties(model)
        torch.set_grad_enabled(False)
        if test:
            self.get_model().test_start()
        outputs = []
        for dataloader_idx, dataloader in enumerate(dataloaders):
            dl_outputs = []
            for batch_idx, batch in enumerate(dataloader):
                if batch is None:
                    continue
                if batch_idx >= max_batches:
                    break
                output = self.evaluation_forward(model, batch, batch_idx, dataloader_idx, test)
                dl_outputs.append(output)
                if test:
                    self.test_progress_bar.update(1)
                else:
                    self.val_progress_bar.update(1)
            outputs.append(dl_outputs)
        if len(dataloaders) == 1:
            outputs = outputs[0]
        model = self.get_model()
        if test:
            eval_results_ = model.test_end(outputs)
        else:
            eval_results_ = model.validation_end(outputs)
        if eval_results_ is not None:
            eval_results = eval_results_
        model.train()
        torch.set_grad_enabled(True)
        return eval_results

    def run_evaluation(self, test=False):
        model = self.get_model()
        model.on_pre_performance_check()
        if test:
            dataloaders = self.get_test_dataloaders()
            max_batches = self.num_test_batches
        else:
            dataloaders = self.get_val_dataloaders()
            max_batches = self.num_val_batches
        position = 2 * self.process_position + (not test)
        desc = 'Testing' if test else 'Validating'
        pbar = tqdm.tqdm(desc=desc, total=max_batches, leave=test, position=position, disable=not self.show_progress_bar, dynamic_ncols=True, unit='batch', file=sys.stdout)
        setattr(self, f"{'test' if test else 'val'}_progress_bar", pbar)
        eval_results = self.evaluate(self.model, dataloaders, max_batches, test)
        _, prog_bar_metrics, log_metrics, callback_metrics, _ = self.process_output(eval_results)
        self.add_tqdm_metrics(prog_bar_metrics)
        self.log_metrics(log_metrics, {})
        self.callback_metrics.update(callback_metrics)
        model.on_post_performance_check()
        tqdm_metrics = self.training_tqdm_dict
        if not test:
            self.main_progress_bar.set_postfix(**tqdm_metrics)
        if test:
            self.test_progress_bar.close()
        else:
            self.val_progress_bar.close()
        if self.proc_rank == 0 and self.checkpoint_callback is not None and not test:
            self.checkpoint_callback.on_epoch_end(epoch=self.current_epoch, logs=self.callback_metrics)

    def evaluation_forward(self, model, batch, batch_idx, dataloader_idx, test=False):
        args = [batch, batch_idx]
        if test and len(self.get_test_dataloaders()) > 1:
            args.append(dataloader_idx)
        elif not test and len(self.get_val_dataloaders()) > 1:
            args.append(dataloader_idx)
        if self.use_ddp or self.use_dp:
            output = model(*args)
            return output
        if self.single_gpu:
            root_gpu = 0
            if isinstance(self.data_parallel_device_ids, list):
                root_gpu = self.data_parallel_device_ids[0]
            batch = self.transfer_batch_to_gpu(batch, root_gpu)
            args[0] = batch
        if test:
            output = model.test_step(*args)
        else:
            output = model.validation_step(*args)
        return output

    def train(self):
        model = self.get_model()
        for epoch in range(self.current_epoch, 1000000):
            if self.use_ddp and hasattr(self.get_train_dataloader().sampler, 'set_epoch'):
                self.get_train_dataloader().sampler.set_epoch(epoch)
            model = self.get_model()
            model.current_epoch = epoch
            self.current_epoch = epoch
            total_val_batches = 0
            if not self.disable_validation:
                is_val_epoch = (self.current_epoch + 1) % self.check_val_every_n_epoch == 0
                val_checks_per_epoch = self.num_training_batches // self.val_check_batch
                val_checks_per_epoch = val_checks_per_epoch if is_val_epoch else 0
                total_val_batches = self.num_val_batches * val_checks_per_epoch
            self.total_batches = self.num_training_batches + total_val_batches
            self.batch_loss_value = 0
            if self.is_iterable_train_dataloader:
                num_iterations = None
            else:
                num_iterations = self.total_batches
            desc = f'Epoch {epoch + 1}' if not self.is_iterable_train_dataloader else ''
            self.main_progress_bar.set_description(desc)
            self.accumulation_scheduler.on_epoch_begin(epoch, self)
            self.run_training_epoch()
            if self.lr_schedulers is not None:
                for lr_scheduler in self.lr_schedulers:
                    lr_scheduler.step(epoch=self.current_epoch)
        self.main_progress_bar.close()
        model.on_train_end()
        if self.logger is not None:
            self.logger.finalize('success')

    def run_training_epoch(self):
        if self.is_function_implemented('on_epoch_start'):
            model = self.get_model()
            model.on_epoch_start()
        for batch_idx, batch in enumerate(self.get_train_dataloader()):
            if batch_idx >= self.num_training_batches:
                break
            self.batch_idx = batch_idx
            model = self.get_model()
            model.global_step = self.global_step
            output = self.run_training_batch(batch, batch_idx)
            batch_result, grad_norm_dic, batch_step_metrics = output
            early_stop_epoch = batch_result == -1
            should_check_val = not self.disable_validation and self.global_step % self.val_check_batch == 0 and not self.fisrt_epoch
            self.fisrt_epoch = False
            if should_check_val:
                self.run_evaluation(test=self.testing)
            should_save_log = (batch_idx + 1) % self.log_save_interval == 0 or early_stop_epoch
            if should_save_log:
                if self.proc_rank == 0 and self.logger is not None:
                    self.logger.save()
            should_log_metrics = batch_idx % self.row_log_interval == 0 or early_stop_epoch
            if should_log_metrics:
                self.log_metrics(batch_step_metrics, grad_norm_dic)
            self.global_step += 1
            self.total_batch_idx += 1
            if early_stop_epoch:
                break
            if self.global_step > self.max_updates:
                None
                exit()
        if self.is_function_implemented('on_epoch_end'):
            model = self.get_model()
            model.on_epoch_end()

    def run_training_batch(self, batch, batch_idx):
        grad_norm_dic = {}
        all_callback_metrics = []
        all_log_metrics = []
        if batch is None:
            return 0, grad_norm_dic, {}
        if self.is_function_implemented('on_batch_start'):
            model_ref = self.get_model()
            response = model_ref.on_batch_start(batch)
            if response == -1:
                return -1, grad_norm_dic, {}
        splits = [batch]
        self.hiddens = None
        for split_idx, split_batch in enumerate(splits):
            self.split_idx = split_idx
            for opt_idx, optimizer in enumerate(self.optimizers):
                if len(self.optimizers) > 1:
                    for param in self.get_model().parameters():
                        param.requires_grad = False
                    for group in optimizer.param_groups:
                        for param in group['params']:
                            param.requires_grad = True

                def optimizer_closure():
                    output = self.training_forward(split_batch, batch_idx, opt_idx, self.hiddens)
                    closure_loss = output[0]
                    progress_bar_metrics = output[1]
                    log_metrics = output[2]
                    callback_metrics = output[3]
                    self.hiddens = output[4]
                    if closure_loss is None:
                        return None
                    closure_loss = closure_loss / self.accumulate_grad_batches
                    model_ref = self.get_model()
                    if closure_loss.requires_grad:
                        model_ref.backward(closure_loss, optimizer)
                    all_callback_metrics.append(callback_metrics)
                    self.add_tqdm_metrics(progress_bar_metrics)
                    all_log_metrics.append(log_metrics)
                    if self.is_function_implemented('on_after_backward'):
                        model_ref = self.get_model()
                        model_ref.on_after_backward()
                    return closure_loss
                loss = optimizer_closure()
                if loss is None:
                    continue
                if self.print_nan_grads:
                    self.print_nan_gradients()
                self.batch_loss_value += loss.item()
                if (self.batch_idx + 1) % self.accumulate_grad_batches == 0:
                    if batch_idx % self.row_log_interval == 0:
                        if self.track_grad_norm > 0:
                            model = self.get_model()
                            grad_norm_dic = model.grad_norm(self.track_grad_norm)
                    self.clip_gradients()
                    model = self.get_model()
                    model.optimizer_step(self.current_epoch, batch_idx, optimizer, opt_idx)
                    self.running_loss.append(self.batch_loss_value)
                    self.batch_loss_value = 0
                    self.avg_loss = np.mean(self.running_loss[-100:])
        if self.is_function_implemented('on_batch_end'):
            model = self.get_model()
            model.on_batch_end()
        self.main_progress_bar.update(1)
        self.main_progress_bar.set_postfix(**self.training_tqdm_dict)
        all_log_metrics = {k: v for d in all_log_metrics for k, v in d.items()}
        self.callback_metrics.update({k: v for d in all_callback_metrics for k, v in d.items()})
        return 0, grad_norm_dic, all_log_metrics

    def training_forward(self, batch, batch_idx, opt_idx, hiddens):
        """
        Handle forward for each training case (distributed, single gpu, etc...)
        :param batch:
        :param batch_idx:
        :return:
        """
        args = [batch, batch_idx, opt_idx]
        if self.use_ddp or self.use_dp:
            output = self.model(*args)
        elif self.single_gpu:
            gpu_id = 0
            if isinstance(self.data_parallel_device_ids, list):
                gpu_id = self.data_parallel_device_ids[0]
            batch = self.transfer_batch_to_gpu(copy.copy(batch), gpu_id)
            args[0] = batch
            output = self.model.training_step(*args)
        else:
            output = self.model.training_step(*args)
        model_ref = self.get_model()
        output_ = model_ref.training_end(output)
        if output_ is not None:
            output = output_
        output = self.process_output(output, train=True)
        return output

    def is_function_implemented(self, f_name):
        model = self.get_model()
        f_op = getattr(model, f_name, None)
        return callable(f_op)

    def _percent_range_check(self, name):
        value = getattr(self, name)
        msg = f'`{name}` must lie in the range [0.0, 1.0], but got {value:.3f}.'
        if name == 'val_check_interval':
            msg += ' If you want to disable validation set `val_percent_check` to 0.0 instead.'
        if not 0.0 <= value <= 1.0:
            raise ValueError(msg)


def data_loader(fn):
    """
    Decorator to make any fx with this use the lazy property
    :param fn:
    :return:
    """
    wraps(fn)
    attr_name = '_lazy_' + fn.__name__

    def _get_data_loader(self):
        try:
            value = getattr(self, attr_name)
        except AttributeError:
            try:
                value = fn(self)
                if value is not None and not isinstance(value, list) and fn.__name__ in ['test_dataloader', 'val_dataloader']:
                    value = [value]
            except AttributeError as e:
                traceback.print_exc()
                error = f'{fn.__name__}: An AttributeError was encountered: ' + str(e)
                raise RuntimeError(error) from e
            setattr(self, attr_name, value)
        return value
    return _get_data_loader


class Args:

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)


def set_hparams(use_cmd=True, config='', exp_name='', hparams_str=''):
    if use_cmd:
        parser = argparse.ArgumentParser(description='neural music')
        parser.add_argument('--config', type=str, default='', help='location of the data corpus')
        parser.add_argument('--exp_name', type=str, default='', help='exp_name')
        parser.add_argument('--hparams', type=str, default='', help='location of the data corpus')
        parser.add_argument('--infer', action='store_true', help='infer')
        parser.add_argument('--validate', action='store_true', help='validate')
        parser.add_argument('--reset', action='store_true', help='reset hparams')
        parser.add_argument('--debug', action='store_true', help='debug')
        args, unknown = parser.parse_known_args()
    else:
        args = Args(config=config, exp_name=exp_name, hparams=hparams_str, infer=False, validate=False, reset=False, debug=False)
    args_work_dir = ''
    if args.exp_name != '':
        args.work_dir = args.exp_name
        args_work_dir = f'checkpoints/{args.work_dir}'

    def load_config(config_fn):
        with open(config_fn) as f:
            hparams_ = yaml.safe_load(f)
        if 'base_config' in hparams_:
            ret_hparams = load_config(hparams_['base_config'])
            ret_hparams.update(hparams_)
        else:
            ret_hparams = hparams_
        return ret_hparams
    global hparams
    assert args.config != '' or args_work_dir != ''
    saved_hparams = {}
    if args_work_dir != 'checkpoints/':
        ckpt_config_path = f'{args_work_dir}/config.yaml'
        if os.path.exists(ckpt_config_path):
            try:
                with open(ckpt_config_path) as f:
                    saved_hparams.update(yaml.safe_load(f))
            except:
                pass
        if args.config == '':
            args.config = ckpt_config_path
    hparams.update(load_config(args.config))
    hparams['work_dir'] = args_work_dir
    if not args.reset:
        hparams.update(saved_hparams)
    if args.hparams != '':
        for new_hparam in args.hparams.split(','):
            k, v = new_hparam.split('=')
            if v in ['True', 'False'] or type(hparams[k]) == bool:
                hparams[k] = eval(v)
            else:
                hparams[k] = type(hparams[k])(v)
    if args_work_dir != '' and (not os.path.exists(ckpt_config_path) or args.reset):
        os.makedirs(hparams['work_dir'], exist_ok=True)
        with open(ckpt_config_path, 'w') as f:
            yaml.safe_dump(hparams, f)
    hparams['infer'] = args.infer
    hparams['debug'] = args.debug
    hparams['validate'] = args.validate


class BaseTask(nn.Module):

    def __init__(self, *args, **kwargs):
        super(BaseTask, self).__init__(*args, **kwargs)
        self.current_epoch = 0
        self.global_step = 0
        self.loaded_optimizer_states_dict = {}
        self.trainer = None
        self.logger = None
        self.on_gpu = False
        self.use_dp = False
        self.use_ddp = False
        self.example_input_array = None
        self.max_tokens = hparams['max_tokens']
        self.max_sentences = hparams['max_sentences']
        self.max_eval_tokens = hparams['max_eval_tokens']
        if self.max_eval_tokens == -1:
            hparams['max_eval_tokens'] = self.max_eval_tokens = self.max_tokens
        self.max_eval_sentences = hparams['max_eval_sentences']
        if self.max_eval_sentences == -1:
            hparams['max_eval_sentences'] = self.max_eval_sentences = self.max_sentences
        None
        for i, (k, v) in enumerate(sorted(hparams.items())):
            None
        None
        self.model = None
        self.training_losses_meter = None

    def build_model(self):
        raise NotImplementedError

    def load_ckpt(self, ckpt_base_dir, model_name='model', force=True, strict=True):
        base_dir = ckpt_base_dir
        checkpoint_path = sorted(glob.glob(f'{base_dir}/model_ckpt_steps_*.ckpt'), key=lambda x: int(re.findall(f'{base_dir}/model_ckpt_steps_(\\d+).ckpt', x)[0]))
        if len(checkpoint_path) > 0:
            checkpoint_path = checkpoint_path[0]
            fake_task = nn.Module()
            fake_task.__setattr__(model_name, self.__getattr__(model_name))
            state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
            if not strict:
                new_model_state_dict = self.state_dict()
                unmatched_keys = []
                for key, param in state_dict.items():
                    if key in new_model_state_dict:
                        new_param = new_model_state_dict[key]
                        if new_param.shape != param.shape:
                            unmatched_keys.append(key)
                for key in unmatched_keys:
                    del state_dict[key]
            fake_task.load_state_dict(state_dict, strict=strict)
            None
        else:
            e_msg = f'| ckpt not found in {base_dir}.'
            if force:
                assert False, e_msg
            else:
                None

    def on_epoch_start(self):
        self.training_losses_meter = {'total_loss': utils.AvgrageMeter()}

    def _training_step(self, sample, batch_idx, optimizer_idx):
        """

        :param sample:
        :param batch_idx:
        :return: total loss: torch.Tensor, loss_log: dict
        """
        raise NotImplementedError

    def training_step(self, sample, batch_idx, optimizer_idx=-1):
        loss_ret = self._training_step(sample, batch_idx, optimizer_idx)
        self.opt_idx = optimizer_idx
        if loss_ret is None:
            return {'loss': None}
        total_loss, log_outputs = loss_ret
        log_outputs = utils.tensors_to_scalars(log_outputs)
        for k, v in log_outputs.items():
            if k not in self.training_losses_meter:
                self.training_losses_meter[k] = utils.AvgrageMeter()
            if not np.isnan(v):
                self.training_losses_meter[k].update(v)
        self.training_losses_meter['total_loss'].update(total_loss.item())
        try:
            log_outputs['lr'] = self.scheduler.get_lr()
            if isinstance(log_outputs['lr'], list):
                log_outputs['lr'] = log_outputs['lr'][0]
        except:
            pass
        progress_bar_log = log_outputs
        tb_log = {f'tr/{k}': v for k, v in log_outputs.items()}
        return {'loss': total_loss, 'progress_bar': progress_bar_log, 'log': tb_log}

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.step()
        optimizer.zero_grad()
        self.scheduler.step(self.global_step // hparams['accumulate_grad_batches'])

    def on_epoch_end(self):
        loss_outputs = {k: round(v.avg, 4) for k, v in self.training_losses_meter.items()}
        None

    def validation_step(self, sample, batch_idx):
        """

        :param sample:
        :param batch_idx:
        :return: output: dict
        """
        raise NotImplementedError

    def _validation_end(self, outputs):
        """

        :param outputs:
        :return: loss_output: dict
        """
        raise NotImplementedError

    def validation_end(self, outputs):
        loss_output = self._validation_end(outputs)
        None
        return {'log': {f'val/{k}': v for k, v in loss_output.items()}, 'val_loss': loss_output['total_loss']}

    def build_scheduler(self, optimizer):
        raise NotImplementedError

    def build_optimizer(self, model):
        raise NotImplementedError

    def configure_optimizers(self):
        set_hparams()
        self.model = self.build_model()
        None
        optm = self.build_optimizer(self.model)
        self.scheduler = self.build_scheduler(optm)
        return [optm]

    def test_start(self):
        pass

    def test_step(self, sample, batch_idx):
        return self.validation_step(sample, batch_idx)

    def test_end(self, outputs):
        return self.validation_end(outputs)

    @classmethod
    def start(cls):
        set_hparams()
        os.environ['MASTER_PORT'] = str(random.randint(15000, 30000))
        random.seed(hparams['seed'])
        np.random.seed(hparams['seed'])
        task = cls()
        trainer = BaseTrainer(checkpoint_callback=LatestModelCheckpoint(filepath=hparams['work_dir'], verbose=True, monitor='val_loss', mode='min', num_keep=5, period=1 if hparams['save_ckpt'] else 100000), logger=TensorBoardLogger(save_dir=hparams['work_dir'], name='lightning_logs', version='lastest'), gradient_clip_val=hparams['clip_grad_norm'], val_check_interval=hparams['val_check_interval'], row_log_interval=hparams['log_interval'], max_updates=hparams['max_updates'], num_sanity_val_steps=hparams['num_sanity_val_steps'] if not hparams['validate'] else 10000, accumulate_grad_batches=hparams['accumulate_grad_batches'])
        if not hparams['infer']:
            trainer.checkpoint_callback.task = task
            trainer.fit(task)
        else:
            trainer.test(task)

    def configure_ddp(self, model, device_ids):
        model = DDP(model, device_ids=device_ids, find_unused_parameters=True)
        if dist.get_rank() != 0 and not hparams['debug']:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
        random.seed(hparams['seed'])
        np.random.seed(hparams['seed'])
        return model

    def training_end(self, *args, **kwargs):
        return None

    def init_ddp_connection(self, proc_rank, world_size):
        default_port = 12910
        try:
            default_port = os.environ['MASTER_PORT']
        except Exception:
            os.environ['MASTER_PORT'] = str(default_port)
        root_node = '127.0.0.2'
        root_node = self.trainer.resolve_root_node_address(root_node)
        os.environ['MASTER_ADDR'] = root_node
        dist.init_process_group('nccl', rank=proc_rank, world_size=world_size)

    @data_loader
    def train_dataloader(self):
        return None

    @data_loader
    def test_dataloader(self):
        return None

    @data_loader
    def val_dataloader(self):
        return None

    def on_load_checkpoint(self, checkpoint):
        pass

    def on_save_checkpoint(self, checkpoint):
        pass

    def on_sanity_check_start(self):
        pass

    def on_train_start(self):
        pass

    def on_train_end(self):
        pass

    def on_batch_start(self, batch):
        pass

    def on_batch_end(self):
        pass

    def on_pre_performance_check(self):
        pass

    def on_post_performance_check(self):
        pass

    def on_before_zero_grad(self, optimizer):
        pass

    def on_after_backward(self):
        pass

    def backward(self, loss, optimizer):
        loss.backward()

    def grad_norm(self, norm_type):
        results = {}
        total_norm = 0
        for name, p in self.named_parameters():
            if p.requires_grad:
                try:
                    param_norm = p.grad.data.norm(norm_type)
                    total_norm += param_norm ** norm_type
                    norm = param_norm ** (1 / norm_type)
                    grad = round(norm.data.cpu().numpy().flatten()[0], 3)
                    results['grad_{}_norm_{}'.format(norm_type, name)] = grad
                except Exception:
                    pass
        total_norm = total_norm ** (1.0 / norm_type)
        grad = round(total_norm.data.cpu().numpy().flatten()[0], 3)
        results['grad_{}_norm_total'.format(norm_type)] = grad
        return results


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, prefix, hparams, shuffle):
        super().__init__()
        self.hparams = hparams
        self.shuffle = shuffle
        self.data_dir = data_dir
        self.prefix = prefix
        self.sort_by_len = hparams['sort_by_len']
        self.sizes = None

    @property
    def _sizes(self):
        return self.sizes

    def __getitem__(self, index):
        raise NotImplementedError

    def collater(self, samples):
        raise NotImplementedError

    def __len__(self):
        return len(self._sizes)

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        size = min(self._sizes[index], hparams['max_frames'])
        return size

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
            if self.sort_by_len:
                indices = indices[np.argsort(np.array(self._sizes)[indices], kind='mergesort')]
        else:
            indices = np.arange(len(self))
        return indices

    @property
    def num_workers(self):
        return int(os.getenv('NUM_WORKERS', 1))


class IndexedDataset:

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.data_file = None
        self.data_offsets = np.load(f'{path}.idx', allow_pickle=True).item()['offsets']
        self.data_file = open(f'{path}.data', 'rb', buffering=0)

    def check_index(self, i):
        if i < 0 or i >= len(self.data_offsets) - 1:
            raise IndexError('index out of range')

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    def __getitem__(self, i):
        self.check_index(i)
        self.data_file.seek(self.data_offsets[i])
        b = self.data_file.read(self.data_offsets[i + 1] - self.data_offsets[i])
        item = pickle.loads(b)
        return item

    def __len__(self):
        return len(self.data_offsets) - 1


class GridDataset(BaseDataset):
    """A dataset that provides helpers for batching."""

    def __init__(self, data_dir, phone_encoder, prefix, hparams, shuffle=False):
        super().__init__(data_dir, prefix, hparams, shuffle)
        self.phone_encoder = phone_encoder
        self.data = None
        self.idx2key = np.load(f'{self.data_dir}/{self.prefix}_all_keys.npy')
        self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
        if self.prefix == 'test':
            self.idx2key = self.idx2key[:hparams['cut_test_set']]
            self.sizes = self.sizes[:hparams['cut_test_set']]
            pass
        else:
            raise NotImplementedError
        self.indexed_ds = None

    def _get_item(self, index):
        if hasattr(self, 'avail_idxs') and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        vid2ph = None
        phone = torch.LongTensor(item['phone'][:hparams['max_input_tokens']])
        guide_face = torch.Tensor(item['guide_face'])
        if len(guide_face.shape) == 2:
            guide_face = guide_face[:, :, None]
        sample = {'id': index, 'utt_id': item['item_name'], 'text': item['txt'], 'source': phone, 'vid2ph': vid2ph, 'guide_face': guide_face}
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        pad_idx = self.phone_encoder.pad()
        id = torch.LongTensor([s['id'] for s in samples])
        utt_ids = [s['utt_id'] for s in samples]
        text = [s['text'] for s in samples]
        src_tokens = utils.collate_1d([s['source'] for s in samples], pad_idx)
        vid2ph = None
        face_lst = [s['guide_face'].reshape(-1) for s in samples]
        guide_face = utils.collate_1d(face_lst, 0.0)
        guide_face = guide_face.reshape(guide_face.shape[0], hparams['img_h'], hparams['img_w'], hparams['img_channel'])
        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
        ntokens = sum(len(s['source']) for s in samples)
        batch = {'id': id, 'utt_id': utt_ids, 'nsamples': len(samples), 'ntokens': ntokens, 'text': text, 'src_tokens': src_tokens, 'vid2ph': vid2ph, 'src_lengths': src_lengths, 'guide_face': guide_face}
        return batch


class RSQRTSchedule(object):

    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer
        self.constant_lr = hparams['lr']
        self.warmup_updates = hparams['warmup_updates']
        self.hidden_size = hparams['hidden_size']
        self.lr = hparams['lr']
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
        self.step(0)

    def step(self, num_updates):
        constant_lr = self.constant_lr
        warmup = min(num_updates / self.warmup_updates, 1.0)
        rsqrt_decay = max(self.warmup_updates, num_updates) ** -0.5
        rsqrt_hidden = self.hidden_size ** -0.5
        self.lr = max(constant_lr * warmup * rsqrt_decay * rsqrt_hidden, 1e-07)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


EOS = '<EOS>'


PAD = '<pad>'


UNK = '<UNK>'


RESERVED_TOKENS = [PAD, EOS, UNK]


NUM_RESERVED_TOKENS = len(RESERVED_TOKENS)


SEG = '|'


def strip_ids(ids, ids_to_strip):
    """Strip ids_to_strip from the end ids."""
    ids = list(ids)
    while ids and ids[-1] in ids_to_strip:
        ids.pop()
    return ids


class TextEncoder(object):
    """Base class for converting from ints to/from human readable strings."""

    def __init__(self, num_reserved_ids=NUM_RESERVED_TOKENS):
        self._num_reserved_ids = num_reserved_ids

    @property
    def num_reserved_ids(self):
        return self._num_reserved_ids

    def encode(self, s):
        """Transform a human-readable string into a sequence of int ids.

        The ids should be in the range [num_reserved_ids, vocab_size). Ids [0,
        num_reserved_ids) are reserved.

        EOS is not appended.

        Args:
        s: human-readable string to be converted.

        Returns:
        ids: list of integers
        """
        return [(int(w) + self._num_reserved_ids) for w in s.split()]

    def decode(self, ids, strip_extraneous=False):
        """Transform a sequence of int ids into a human-readable string.

        EOS is not expected in ids.

        Args:
        ids: list of integers to be converted.
        strip_extraneous: bool, whether to strip off extraneous tokens
            (EOS and PAD).

        Returns:
        s: human-readable string.
        """
        if strip_extraneous:
            ids = strip_ids(ids, list(range(self._num_reserved_ids or 0)))
        return ' '.join(self.decode_list(ids))

    def decode_list(self, ids):
        """Transform a sequence of int ids into a their string versions.

        This method supports transforming individual input/output ids to their
        string versions so that sequence to/from text conversions can be visualized
        in a human readable format.

        Args:
        ids: list of integers to be converted.

        Returns:
        strs: list of human-readable string.
        """
        decoded_ids = []
        for id_ in ids:
            if 0 <= id_ < self._num_reserved_ids:
                decoded_ids.append(RESERVED_TOKENS[int(id_)])
            else:
                decoded_ids.append(id_ - self._num_reserved_ids)
        return [str(d) for d in decoded_ids]

    @property
    def vocab_size(self):
        raise NotImplementedError()


class TokenTextEncoder(TextEncoder):
    """Encoder based on a user-supplied vocabulary (file or list)."""

    def __init__(self, vocab_filename, reverse=False, vocab_list=None, replace_oov=None, num_reserved_ids=NUM_RESERVED_TOKENS):
        """Initialize from a file or list, one token per line.

        Handling of reserved tokens works as follows:
        - When initializing from a list, we add reserved tokens to the vocab.
        - When initializing from a file, we do not add reserved tokens to the vocab.
        - When saving vocab files, we save reserved tokens to the file.

        Args:
            vocab_filename: If not None, the full filename to read vocab from. If this
                is not None, then vocab_list should be None.
            reverse: Boolean indicating if tokens should be reversed during encoding
                and decoding.
            vocab_list: If not None, a list of elements of the vocabulary. If this is
                not None, then vocab_filename should be None.
            replace_oov: If not None, every out-of-vocabulary token seen when
                encoding will be replaced by this string (which must be in vocab).
            num_reserved_ids: Number of IDs to save for reserved tokens like <EOS>.
        """
        super(TokenTextEncoder, self).__init__(num_reserved_ids=num_reserved_ids)
        self._reverse = reverse
        self._replace_oov = replace_oov
        if vocab_filename:
            self._init_vocab_from_file(vocab_filename)
        else:
            assert vocab_list is not None
            self._init_vocab_from_list(vocab_list)
        self.pad_index = self._token_to_id[PAD]
        self.eos_index = self._token_to_id[EOS]
        self.unk_index = self._token_to_id[UNK]
        self.seg_index = self._token_to_id[SEG] if SEG in self._token_to_id else self.eos_index

    def encode(self, s):
        """Converts a space-separated string of tokens to a list of ids."""
        sentence = s
        tokens = sentence.strip().split()
        if self._replace_oov is not None:
            tokens = [(t if t in self._token_to_id else self._replace_oov) for t in tokens]
        ret = [self._token_to_id[tok] for tok in tokens]
        return ret[::-1] if self._reverse else ret

    def decode(self, ids, strip_eos=False, strip_padding=False):
        if strip_padding and self.pad() in list(ids):
            pad_pos = list(ids).index(self.pad())
            ids = ids[:pad_pos]
        if strip_eos and self.eos() in list(ids):
            eos_pos = list(ids).index(self.eos())
            ids = ids[:eos_pos]
        return ' '.join(self.decode_list(ids))

    def decode_list(self, ids):
        seq = reversed(ids) if self._reverse else ids
        return [self._safe_id_to_token(i) for i in seq]

    @property
    def vocab_size(self):
        return len(self._id_to_token)

    def __len__(self):
        return self.vocab_size

    def _safe_id_to_token(self, idx):
        return self._id_to_token.get(idx, 'ID_%d' % idx)

    def _init_vocab_from_file(self, filename):
        """Load vocab from a file.

        Args:
        filename: The file to load vocabulary from.
        """
        with open(filename) as f:
            tokens = [token.strip() for token in f.readlines()]

        def token_gen():
            for token in tokens:
                yield token
        self._init_vocab(token_gen(), add_reserved_tokens=False)

    def _init_vocab_from_list(self, vocab_list):
        """Initialize tokens from a list of tokens.

        It is ok if reserved tokens appear in the vocab list. They will be
        removed. The set of tokens in vocab_list should be unique.

        Args:
        vocab_list: A list of tokens.
        """

        def token_gen():
            for token in vocab_list:
                if token not in RESERVED_TOKENS:
                    yield token
        self._init_vocab(token_gen())

    def _init_vocab(self, token_generator, add_reserved_tokens=True):
        """Initialize vocabulary with tokens from token_generator."""
        self._id_to_token = {}
        non_reserved_start_index = 0
        if add_reserved_tokens:
            self._id_to_token.update(enumerate(RESERVED_TOKENS))
            non_reserved_start_index = len(RESERVED_TOKENS)
        self._id_to_token.update(enumerate(token_generator, start=non_reserved_start_index))
        self._token_to_id = dict((v, k) for k, v in six.iteritems(self._id_to_token))

    def pad(self):
        return self.pad_index

    def eos(self):
        return self.eos_index

    def unk(self):
        return self.unk_index

    def seg(self):
        return self.seg_index

    def store_to_file(self, filename):
        """Write vocab file to disk.

        Vocab files have one token per line. The file ends in a newline. Reserved
        tokens are written to the vocab file as well.

        Args:
        filename: Full path of the file to store the vocab to.
        """
        with open(filename, 'w') as f:
            for i in range(len(self._id_to_token)):
                f.write(self._id_to_token[i] + '\n')


class FastLipGenTask(BaseTask):

    def __init__(self):
        self.arch = hparams['arch']
        if isinstance(self.arch, str):
            self.arch = list(map(int, self.arch.strip().split()))
        if self.arch is not None:
            self.num_heads = utils.get_num_heads(self.arch[hparams['enc_layers']:])
        self.phone_encoder = self.build_phone_encoder(hparams['data_dir'])
        super(FastLipGenTask, self).__init__()
        self.dur_loss_fn = DurationPredictorLoss()
        self.mse_loss_fn = torch.nn.MSELoss()
        self.ds_cls = GridDataset
        self.item2wav = {os.path.splitext(os.path.basename(v))[0]: v for v in glob.glob('./data/grid/grid_wavs/*.wav')}
        self.padding_idx = self.phone_encoder.pad()
        self.eos_idx = self.phone_encoder.eos()
        self.seg_idx = self.phone_encoder.seg()
        self.saving_result_pool = None
        self.saving_results_futures = None
        self.stats = {}

    @data_loader
    def test_dataloader(self):
        test_dataset = self.ds_cls(hparams['data_dir'], self.phone_encoder, hparams['test_set_name'], hparams, shuffle=False)
        return self.build_dataloader(test_dataset, False, self.max_eval_tokens, self.max_eval_sentences)

    def build_model(self):
        arch = self.arch
        model = FastLip(arch, self.phone_encoder)
        return model

    def build_dataloader(self, dataset, shuffle, max_tokens=None, max_sentences=None, required_batch_size_multiple=-1, endless=False):
        if required_batch_size_multiple == -1:
            required_batch_size_multiple = torch.cuda.device_count()

        def shuffle_batches(batches):
            np.random.shuffle(batches)
            return batches
        if max_tokens is not None:
            max_tokens *= torch.cuda.device_count()
        if max_sentences is not None:
            max_sentences *= torch.cuda.device_count()
        indices = dataset.ordered_indices()
        batch_sampler = utils.batch_by_size(indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences, required_batch_size_multiple=required_batch_size_multiple)
        if shuffle:
            batches = shuffle_batches(list(batch_sampler))
            if endless:
                batches = [b for _ in range(1000) for b in shuffle_batches(list(batch_sampler))]
        else:
            batches = batch_sampler
            if endless:
                batches = [b for _ in range(1000) for b in batches]
        num_workers = dataset.num_workers
        if self.trainer.use_ddp:
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
            batches = [x[rank::num_replicas] for x in batches if len(x) % num_replicas == 0]
        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collater, batch_sampler=batches, num_workers=num_workers, pin_memory=False)

    def build_phone_encoder(self, data_dir):
        phone_list_file = os.path.join(data_dir, 'phone_set.json')
        phone_list = json.load(open(phone_list_file))
        return TokenTextEncoder(None, vocab_list=phone_list)

    def build_scheduler(self, optimizer):
        return RSQRTSchedule(optimizer)

    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.AdamW(model.parameters(), lr=hparams['lr'], betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']), weight_decay=hparams['weight_decay'])
        return optimizer

    def test_step(self, sample, batch_idx):
        test_guide_face = sample['guide_face']
        src_tokens = sample['src_tokens']
        if hparams['profile_infer']:
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            vid2ph = sample['vid2ph']
        else:
            vid2ph = None
        with utils.Timer('fs', print_time=hparams['profile_infer']):
            if hparams['vid_use_gt_dur']:
                outputs = self.model(src_tokens, sample['vid2ph'], test_guide_face)
            else:
                outputs = self.model(src_tokens, vid2ph, test_guide_face)
        sample['outputs'] = outputs['lip_out']
        sample['vid2ph_pred'] = outputs['vid2ph']
        return self.after_infer(sample)

    def after_infer(self, predictions):
        if self.saving_result_pool is None and not hparams['profile_infer']:
            self.saving_result_pool = Pool(8)
        predictions = utils.unpack_dict_to_list(predictions)
        t = tqdm(predictions)
        for num_predictions, prediction in enumerate(t):
            for k, v in prediction.items():
                if type(v) is torch.Tensor:
                    prediction[k] = v.cpu().numpy()
            utt_id = prediction.get('utt_id')
            text = prediction.get('text')
            outputs = self.remove_padding_3D(prediction['outputs'])
            outputs = self.tensor2img(outputs)
            gen_dir = os.path.join(hparams['work_dir'], f"generated_{self.trainer.global_step}_{hparams['gen_dir_name']}")
            if not hparams['profile_infer']:
                os.makedirs(gen_dir, exist_ok=True)
                self.saving_result_pool.apply_async(self.save_result, args=[outputs, f'P', utt_id, text, gen_dir, self.item2wav])
                t.set_description(f'Pred_shape: {outputs.shape}')
            else:
                if 'gen_wav_time' not in self.stats:
                    self.stats['gen_wav_time'] = 0
                self.stats['gen_lip_frames'] += outputs.shape[0]
                None
        return {}

    def test_end(self, outputs):
        self.saving_result_pool.close()
        self.saving_result_pool.join()
        return {}

    def test_start(self):
        pass

    def remove_padding_3D(self, x, padding_idx=0):
        if x is None:
            return None
        assert len(x.shape) in [4]
        return x[np.abs(x).sum(-1).sum(-1).sum(-1) != padding_idx]

    @staticmethod
    def save_result(vid, prefix, utt_id, text, gen_dir, item2wav):
        base_fn = f'[{prefix}][{utt_id}]'
        if text is not None:
            TXT = text.replace(':', '%3A')[:80]
            base_fn += '_'.join(TXT.split(' '))
        os.makedirs(f'{gen_dir}/{base_fn}', exist_ok=True)
        destination_path = f'{gen_dir}/{base_fn}'
        for idx, frame in enumerate(vid):
            io.imsave(destination_path + '/' + '{0:03d}.png'.format(idx), frame)
        imgclip = ImageSequenceClip(list(vid), fps=hparams['vid_fps'])
        if utt_id in item2wav.keys() and os.path.exists(item2wav[utt_id]):
            wav_clip = AudioFileClip(item2wav[utt_id], fps=16000)
            imgclip = imgclip.set_audio(wav_clip)
        imgclip.write_videofile(gen_dir + f'/{prefix}+{utt_id}.avi', codec='png', fps=hparams['vid_fps'], audio_fps=16000, logger=None)
        return {}

    @staticmethod
    def tensor2img(vid):
        vid = (vid * 255.0).clip(min=0.0, max=255.0)
        vid = vid.astype(np.uint8)
        return vid


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Conv2dBlock,
     lambda: ([], {'idim': 4, 'odim': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4, 'norm_fn': 4, 'acti_fn': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv3dBlock,
     lambda: ([], {'idim': 4, 'odim': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4, 'norm_fn': 4, 'acti_fn': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (ConvAttentionLayer,
     lambda: ([], {'c': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (ConvNorm,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (ConvTBC,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (CyclicalPositionEmb,
     lambda: ([], {'K': 4, 'emb_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DP,
     lambda: ([], {'module': torch.nn.ReLU()}),
     lambda: ([], {'input': torch.rand([4, 4])})),
    (DurationPredictorLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (EncLSTMLayer,
     lambda: ([], {'c': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4])], {})),
    (LSTMAttentionLayer,
     lambda: ([], {'input_embed_dim': 4, 'source_embed_dim': 4, 'output_embed_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (LinearizedConvolution,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (MultiheadAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (NewTransformerFFNLayer,
     lambda: ([], {'hidden_size': 4, 'filter_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (SelfAttention,
     lambda: ([], {'hid_dim': 4, 'n_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
]

