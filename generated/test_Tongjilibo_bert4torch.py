
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


import torch


from torch import nn


import torch.nn.functional as F


from typing import Union


from typing import Optional


from typing import List


from typing import Tuple


from typing import Literal


import torch.nn as nn


import numpy as np


import inspect


import copy


import random


import warnings


import torch.distributed as dist


from typing import Callable


import re


from torch.utils.checkpoint import checkpoint


from typing import Any


import torch.utils.checkpoint


from torch.nn.init import _calculate_fan_in_and_fan_out


from functools import partial


from torch import Tensor


from torch.nn.functional import *


from torch.nn.modules.activation import *


from torch.nn.init import trunc_normal_


from torch.nn.init import constant_


from torch.nn.init import xavier_normal_


from torch.nn.init import xavier_uniform_


import logging


import time


from functools import lru_cache


import torchvision


from torchvision import io


from torchvision import transforms


from torchvision.transforms import InterpolationMode


from torch.optim.lr_scheduler import LambdaLR


from torch.optim.optimizer import Optimizer


from typing import Dict


from copy import deepcopy


from torch.nn import Linear


from torch.nn import Embedding


from torch.nn.parameter import Parameter


from torch.nn.utils.rnn import pad_sequence


from typing import Iterable


from typing import Set


from torch.utils.checkpoint import CheckpointFunction


import collections


from collections import OrderedDict


from torch.nn.modules import Module


from torch.utils.data import DataLoader


import torch.optim as optim


from torch.nn.functional import softmax


from torch.nn import Module


from sklearn.metrics import mean_squared_error


from sklearn.metrics import mean_absolute_error


from typing import Sequence


import pandas as pd


from torch.utils.data import Dataset


from itertools import groupby


from torch import optim


from torch.distributions.bernoulli import Bernoulli


from torch.utils.tensorboard import SummaryWriter


from sklearn.metrics import accuracy_score


from sklearn.metrics import f1_score


from collections import Counter


from scipy.optimize import linear_sum_assignment


from sklearn.metrics import classification_report


import tensorflow as tf


from torch import device


from sklearn.model_selection import StratifiedKFold


from torch.optim import Adam


from torch.utils.data import TensorDataset


from sklearn.metrics.pairwise import paired_cosine_distances


from sklearn.metrics import roc_auc_score


from sklearn.decomposition import PCA


from scipy.stats import spearmanr


from scipy.stats import pearsonr


from sklearn.metrics.pairwise import paired_euclidean_distances


from sklearn.metrics.pairwise import paired_manhattan_distances


import scipy.stats


from collections import defaultdict


from sklearn.metrics import precision_recall_fscore_support


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from collections import deque


from random import sample


import functools


import torch.onnx


from torch.optim import AdamW


from torch.utils.data.distributed import DistributedSampler


from itertools import cycle


def is_causal_mask(attention_mask: 'torch.Tensor'):
    """判断一个矩阵是不是下三角阵"""
    return bool(torch.all(torch.tril(attention_mask) == attention_mask))


class MultiHeadAttention(nn.Module):
    """多头注意力
    :param hidden_size: int, 隐含层神经元个数
    :param num_attention_heads: int, 多头注意力的多头数
    :param attention_probs_dropout_prob: float，softmax后的dropout rate
    :param dropout_rate: float, pos_dropout对应的dropout rate, 目前仅在deverta中使用，默认为0.1
    :param attention_scale: bool, 是否对attention_scores进行缩放，默认为True
    :param output_attentions: bool，是否返回attention_scores，默认为False
    :param bias: bool, qkvo的weight是否包含bias，默认为True
    :param rope_scaling: dict, rope的position encoding的参数，默认为None
    :param _attn_implementation: Literal枚举值，计算attention score的方式，支持'sdpa', 'xformers', 'flash_attn_2', "eager"等, 默认为None
    :param use_logn_attn: bool，是否使用use_logn_attn, 默认为None
    :param layer_idx: int，transformer block的层序号
    """

    def __init__(self, hidden_size: 'int', num_attention_heads: 'int', attention_probs_dropout_prob: 'float', dropout_rate: 'float'=0.1, attention_scale: 'bool'=True, output_attentions: 'bool'=False, bias: 'bool'=True, rope_scaling: 'dict'=None, _attn_implementation: "Literal['sdpa', 'xformers', 'flash_attn_2', 'eager']"='eager', use_logn_attn: 'bool'=None, layer_idx: 'int'=None, num_key_value_heads: 'int'=None, **kwargs):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.dropout_rate = dropout_rate
        self.is_decoder = kwargs.get('is_decoder', False)
        self.attention_scale = attention_scale
        self.output_attentions = output_attentions
        self.bias = bias
        self.rope_scaling = rope_scaling or dict()
        self.layer_idx = layer_idx
        self.sliding_window = kwargs.get('sliding_window')
        self.max_window_layers = kwargs.get('max_window_layers')
        self._attn_implementation = _attn_implementation
        self.use_logn_attn = use_logn_attn
        self.max_position = kwargs.get('max_position')
        self.attention_head_size = kwargs.get('attention_head_size', int(hidden_size / num_attention_heads))
        self.attention_key_size = kwargs.get('attention_key_size', self.attention_head_size)
        q_inner_dim = self.attention_key_size * num_attention_heads
        k_inner_dim = q_inner_dim
        v_inner_dim = self.attention_head_size * num_attention_heads
        if num_key_value_heads is not None:
            self.num_key_value_heads = num_key_value_heads
            k_inner_dim_tmp = self.attention_head_size * self.num_key_value_heads
            v_inner_dim_tmp = k_inner_dim_tmp
        if kwargs.get('longlora_group_size') is not None:
            self.longlora_group_size = kwargs.get('longlora_group_size')
        self.q = nn.Linear(hidden_size, q_inner_dim, bias=bias)
        self.k = nn.Linear(hidden_size, k_inner_dim_tmp if hasattr(self, 'num_key_value_heads') else k_inner_dim, bias=bias)
        self.v = nn.Linear(hidden_size, v_inner_dim_tmp if hasattr(self, 'num_key_value_heads') else v_inner_dim, bias=bias)
        self.o = nn.Linear(v_inner_dim, hidden_size, bias=bias)
        self.dropout = nn.Dropout(attention_probs_dropout_prob) if attention_probs_dropout_prob > 0 else lambda x: x
        self.init_position_encoding(**kwargs)

    def init_position_encoding(self, **kwargs):
        """初始化相对位置编码"""
        pass

    def _get_qkv_states(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, position_ids):
        """获取qkv states，主要是为了下游继承"""
        query_states = self.transpose_for_q_scores(self.q(hidden_states))
        if encoder_hidden_states is not None and past_key_value is not None:
            key_states, value_states = past_key_value
            attention_mask = encoder_attention_mask
        elif encoder_hidden_states is not None:
            key_states = self.transpose_for_k_scores(self.k(encoder_hidden_states))
            value_states = self.transpose_for_v_scores(self.v(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_states = self.transpose_for_k_scores(self.k(hidden_states))
            value_states = self.transpose_for_v_scores(self.v(hidden_states))
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self.transpose_for_k_scores(self.k(hidden_states))
            value_states = self.transpose_for_v_scores(self.v(hidden_states))
        return query_states, key_states, value_states, attention_mask

    def forward(self, hidden_states: 'Optional[torch.Tensor]'=None, attention_mask: 'Optional[torch.FloatTensor]'=None, encoder_hidden_states: 'Optional[torch.FloatTensor]'=None, encoder_attention_mask: 'Optional[torch.FloatTensor]'=None, past_key_value: 'Optional[Tuple[Tuple[torch.FloatTensor]]]'=None, position_ids=None, **model_kwargs):
        """
        :param hidden_states: [batch_size, seq_q, hidden_size]
        :param attention_mask: [batch_size, 1, 1, seq_q] 或者 [batch_size, 1, seq_q, seq_q]
        :param encoder_hidden_states: [batch_size, seq_k, hidden_size]
        :param encoder_attention_mask: [batch_size, 1, 1, seq_k]
        :param past_key_value: ([batch_size, num_attention_heads, key_len_cache, attention_head_size], ...)
        """
        query_states, key_states, value_states, attention_mask = self._get_qkv_states(hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, position_ids)
        if self.use_logn_attn:
            query_states *= ((position_ids + 1)[:, None, :, None].log() / np.log(self.max_position)).clip(1)
        if self.is_decoder and not self.training:
            past_key_value = key_states, value_states
        if hasattr(self, 'num_key_value_heads') and self.num_key_value_heads > 1:
            key_states = self.repeat_kv(key_states)
            value_states = self.repeat_kv(value_states)
        if hasattr(self, 'longlora_group_size'):
            query_states, key_states, value_states, attention_mask = self.longlora_shift(query_states, key_states, value_states, attention_mask)
        attention_scores = None
        if self._attn_implementation == 'xformers' and self.training:
            context_layer = xops.memory_efficient_attention(query_states, key_states, value_states, attn_bias=xops.LowerTriangularMask())
        elif self._attn_implementation in {True, 'sdpa'}:
            is_causal = query_states.shape[2] == key_states.shape[2] and is_causal_mask(attention_mask)
            context_layer = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=None if is_causal else attention_mask.bool(), dropout_p=self.attention_probs_dropout_prob if self.training else 0.0, is_causal=is_causal)
        elif self._attn_implementation == 'flash_attn_2':
            context_layer = self.flash_attention_forward(query_states, key_states, value_states, past_key_value, attention_mask, hidden_states.shape[1])
        else:
            context_layer, attention_scores = self.torch_attention_forward(query_states, key_states, value_states, attention_mask)
        if hasattr(self, 'longlora_group_size'):
            bsz, q_len = hidden_states.shape[:2]
            context_layer = context_layer.transpose(1, 2).contiguous()
            context_layer = context_layer.reshape(bsz, q_len, self.num_attention_heads, self.attention_head_size)
            context_layer[:, :, self.num_attention_heads // 2:] = context_layer[:, :, self.num_attention_heads // 2:].roll(self.longlora_group_size // 2, dims=1)
            context_layer = context_layer.reshape(bsz, q_len, self.hidden_size)
        else:
            context_layer = context_layer.permute(0, 2, 1, 3)
            new_context_layer_shape = context_layer.size()[:-2] + (context_layer.size()[-2] * context_layer.size()[-1],)
            context_layer = context_layer.reshape(*new_context_layer_shape)
        outputs = (self.o(context_layer), attention_scores) if self.output_attentions else (self.o(context_layer),)
        return outputs + (past_key_value,) if self.is_decoder else outputs

    def repeat_kv(self, hidden_states):
        hidden_states = hidden_states.unsqueeze(2)
        hidden_states = hidden_states.expand(-1, -1, self.num_attention_heads // self.num_key_value_heads, -1, -1)
        hidden_states = hidden_states.contiguous().view(hidden_states.shape[:1] + (self.num_attention_heads,) + hidden_states.shape[-2:])
        return hidden_states

    def longlora_shift(self, query_states, key_states, value_states, attention_mask):
        """longlora中对qkv和mask进行修改: https://github.com/dvlab-research/LongLoRA"""

        def shift(qkv, bsz, q_len, group_size, num_heads, head_dim):
            qkv[:, num_heads // 2:] = qkv[:, num_heads // 2:].roll(-group_size // 2, dims=2)
            qkv = qkv.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, num_heads, head_dim).transpose(1, 2)
            return qkv
        bsz, _, q_len, _ = query_states.shape
        num_group = q_len // self.longlora_group_size
        query_states = shift(query_states, bsz, q_len, self.longlora_group_size, self.num_attention_heads, self.attention_head_size)
        key_states = shift(key_states, bsz, q_len, self.longlora_group_size, self.num_attention_heads, self.attention_head_size)
        value_states = shift(value_states, bsz, q_len, self.longlora_group_size, self.num_attention_heads, self.attention_head_size)
        attention_mask = attention_mask[:, :, :self.longlora_group_size, :self.longlora_group_size].repeat(num_group, 1, 1, 1)
        return query_states, key_states, value_states, attention_mask

    def transpose_for_q_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_key_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_k_scores(self, x):
        if hasattr(self, 'num_key_value_heads'):
            new_x_shape = x.size()[:-1] + (self.num_key_value_heads, self.attention_key_size)
        else:
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_key_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_v_scores(self, x):
        if hasattr(self, 'num_key_value_heads'):
            new_x_shape = x.size()[:-1] + (self.num_key_value_heads, self.attention_head_size)
        else:
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def apply_attention_scale(self, attention_scores):
        """方便子类继承"""
        return attention_scores / math.sqrt(self.attention_head_size)

    def apply_relative_pos_emb(self, query_states, key_states, attention_scores):
        return attention_scores

    def torch_attention_forward(self, query_states: 'torch.FloatTensor', key_states: 'torch.FloatTensor', value_states: 'torch.FloatTensor', attention_mask: 'torch.Tensor'):
        """qkv attention: torch原生实现"""
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        attention_scores = self.apply_relative_pos_emb(query_states, key_states, attention_scores)
        if self.attention_scale:
            attention_scores = self.apply_attention_scale(attention_scores)
        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask) * torch.finfo(query_states.dtype).min
            attention_scores = attention_scores + attention_mask
        attention_probs = F.softmax(attention_scores, dim=-1, dtype=query_states.dtype)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_states)
        return context_layer, attention_scores

    def flash_attention_forward(self, query_states: 'torch.FloatTensor', key_states: 'torch.FloatTensor', value_states: 'torch.FloatTensor', past_key_value: 'Union[Tuple[torch.Tensor]]', attention_mask: 'torch.Tensor', query_length: 'int', softmax_scale: 'float'=None):
        """ flash_attn，参考transformers中的调用
        """

        def _get_unpad_data(attention_mask):
            seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            max_seqlen_in_batch = seqlens_in_batch.max().item()
            cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
            return indices, cu_seqlens, max_seqlen_in_batch

        def _upad_input(self, query_states, key_states, value_states, attention_mask, query_length):
            indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
            batch_size, kv_seq_len, num_key_value_heads, head_dim = key_states.shape
            key_states = index_first_axis(key_states.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
            value_states = index_first_axis(value_states.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
            if query_length == kv_seq_len:
                query_states = index_first_axis(query_states.reshape(batch_size * kv_seq_len, self.num_attention_heads, head_dim), indices_k)
                cu_seqlens_q = cu_seqlens_k
                max_seqlen_in_batch_q = max_seqlen_in_batch_k
                indices_q = indices_k
            elif query_length == 1:
                max_seqlen_in_batch_q = 1
                cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=query_states.device)
                indices_q = cu_seqlens_q[:-1]
                query_states = query_states.squeeze(1)
            else:
                attention_mask = attention_mask[:, -query_length:]
                query_states, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_states, attention_mask)
            return query_states, key_states, value_states, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_in_batch_q, max_seqlen_in_batch_k)

        def _use_sliding_windows():
            if self.max_window_layers is not None and self.layer_idx >= self.max_window_layers:
                return False
            kv_seq_len = key_states.shape[1]
            use_sliding_windows = _flash_supports_window_size and self.sliding_window is not None and kv_seq_len > self.sliding_window
            if use_sliding_windows and not _flash_supports_window_size:
                log_warn_once('The current flash attention version does not support sliding window attention, for a more memory efficient implementation make sure to upgrade flash-attn library.')
                use_sliding_windows = False
            if use_sliding_windows and past_key_value is not None and past_key_value[0].shape[2] > 0:
                use_sliding_windows = True
            return use_sliding_windows

        def _run_sliding_windows(key_states, value_states, past_key_value, attention_mask):
            """sliding_window部分"""
            slicing_tokens = -self.sliding_window
            past_key = past_key_value[0][:, :, slicing_tokens:, :].contiguous()
            past_value = past_key_value[1][:, :, slicing_tokens:, :].contiguous()
            past_key_value = past_key, past_value
            if past_key.shape[-2] != self.sliding_window:
                raise ValueError(f'past key must have a shape of (`batch_size, num_heads, self.sliding_window-1, head_dim`), got {past_key.shape}')
            if attention_mask is not None:
                attention_mask = attention_mask[:, :, slicing_tokens:, slicing_tokens:]
            key_states = key_states[:, slicing_tokens:, :, :].contiguous()
            value_states = value_states[:, slicing_tokens:, :, :].contiguous()
            return key_states, value_states, past_key_value, attention_mask

        def _transpose(query_states, key_states, value_states):
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            return query_states, key_states, value_states
        dropout = 0.0 if not self.training else self.attention_probs_dropout_prob
        is_causal = is_causal_mask(attention_mask)
        if not is_causal and attention_mask.shape[1:3] == torch.Size([1, 1]):
            query_states, key_states, value_states = _transpose(query_states, key_states, value_states)
            use_sliding_windows = _use_sliding_windows()
            if use_sliding_windows:
                key_states, value_states, past_key_value, attention_mask = _run_sliding_windows(key_states, value_states, past_key_value, attention_mask)
            attn_mask = attention_mask[:, 0, 0, :]
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = _upad_input(self, query_states, key_states, value_states, attn_mask, query_length)
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
            attn_output_unpad = flash_attn_varlen_func(query_states, key_states, value_states, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_in_batch_q, max_seqlen_k=max_seqlen_in_batch_k, dropout_p=dropout, softmax_scale=softmax_scale, causal=False, window_size=(self.sliding_window, self.sliding_window) if use_sliding_windows else (-1, -1))
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        elif is_causal:
            query_states, key_states, value_states = _transpose(query_states, key_states, value_states)
            use_sliding_windows = _use_sliding_windows()
            if use_sliding_windows:
                key_states, value_states, past_key_value, attention_mask = _run_sliding_windows(key_states, value_states, past_key_value, attention_mask)
            attn_output = flash_attn_func(query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=True, window_size=(self.sliding_window, self.sliding_window) if use_sliding_windows else (-1, -1))
        elif is_causal:
            log_warn_once('Flash Attention only support key_padding_mask, use torch_attention_forward instead.')
            attn_output, _ = self.torch_attention_forward(query_states, key_states, value_states, attention_mask)
            self._attn_implementation = None
            return attn_output
        return attn_output.transpose(1, 2)


class DebertaV2PositionsEncoding(nn.Module):
    """deberta用的相对位置编码
    来自论文：https://arxiv.org/abs/2006.03654
    """

    def __init__(self, qlen, klen, position_buckets, max_position):
        super(DebertaV2PositionsEncoding, self).__init__()
        q_ids = torch.arange(0, qlen)
        k_ids = torch.arange(0, klen)
        rel_pos_ids = q_ids[:, None] - k_ids[None, :]
        if position_buckets > 0 and max_position > 0:
            rel_pos_ids = self.make_log_bucket_position(rel_pos_ids, position_buckets, max_position)
        rel_pos_ids = rel_pos_ids
        rel_pos_ids = rel_pos_ids[:qlen, :]
        rel_pos_ids = rel_pos_ids.unsqueeze(0)
        self.register_buffer('relative_position', rel_pos_ids)

    @staticmethod
    def make_log_bucket_position(relative_pos, bucket_size, max_position):
        sign = torch.sign(relative_pos)
        mid = bucket_size // 2
        abs_pos = torch.where((relative_pos < mid) & (relative_pos > -mid), torch.tensor(mid - 1).type_as(relative_pos), torch.abs(relative_pos))
        log_pos = torch.ceil(torch.log(abs_pos / mid) / torch.log(torch.tensor((max_position - 1) / mid)) * (mid - 1)) + mid
        bucket_pos = torch.where(abs_pos <= mid, relative_pos.type_as(log_pos), log_pos * sign)
        return bucket_pos

    def forward(self, qlen, klen):
        return self.relative_position[:, :qlen, :klen]


class DebertaV2Attention(MultiHeadAttention):

    def init_position_encoding(self, **kwargs):
        self.share_att_key = kwargs.get('share_att_key', False)
        self.position_buckets = kwargs.get('position_buckets', -1)
        self.max_relative_positions = kwargs.get('max_relative_positions', -1)
        if self.max_relative_positions < 1:
            self.max_relative_positions = kwargs.get('max_position_embeddings')
        self.pos_ebd_size = self.max_relative_positions
        if self.position_buckets > 0:
            self.pos_ebd_size = self.position_buckets
        self.pos_att_type = kwargs.get('pos_att_type', [])
        self.relative_positions = DebertaV2PositionsEncoding(qlen=self.max_position, klen=self.max_position, position_buckets=kwargs.get('position_buckets'), max_position=self.max_position)
        self.relative_positions_encoding = nn.Embedding(self.max_position, self.hidden_size)
        self.norm_rel_ebd = [x.strip() for x in kwargs.get('norm_rel_ebd', 'none').lower().split('|')]
        if 'layer_norm' in self.norm_rel_ebd:
            self.layernorm = nn.LayerNorm(self.hidden_size, kwargs.get('layer_norm_eps', 1e-12), elementwise_affine=True)
        self.pos_dropout = nn.Dropout(self.dropout_rate)

    def apply_relative_pos_emb(self, query_states, key_states, attention_scores):
        if not hasattr(self, 'relative_positions_encoding'):
            return attention_scores
        self.attention_scale = False
        scale_factor = 1
        if 'c2p' in self.pos_att_type:
            scale_factor += 1
        if 'p2c' in self.pos_att_type:
            scale_factor += 1
        scale = torch.sqrt(torch.tensor(query_states.size(-1), dtype=torch.float) * scale_factor)
        attention_scores = attention_scores / scale
        rel_embeddings = self.pos_dropout(self.layernorm(self.relative_positions_encoding.weight))
        relations_keys = self.relative_positions(attention_scores.shape[-2], attention_scores.shape[-1])
        rel_att = self.apply_deberta_pos_emb(query_states, key_states, relations_keys, rel_embeddings, scale_factor)
        attention_scores = attention_scores + rel_att
        return attention_scores

    def apply_deberta_pos_emb(self, query_states: 'torch.FloatTensor', key_states: 'torch.FloatTensor', relative_pos, rel_embeddings, scale_factor):
        """deberta_v2使用，和原版区别是query_states是4维, 原disentangled_attention_bias"""
        btz, n_head, q_len, d_head = query_states.size()
        k_len = key_states.size(-2)
        if relative_pos is None:
            relative_pos = self.relative_positions(q_len, k_len)
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        elif relative_pos.dim() != 4:
            raise ValueError(f'Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}')
        att_span = self.pos_ebd_size
        relative_pos = relative_pos.long()
        rel_embeddings = rel_embeddings[0:att_span * 2, :].unsqueeze(0)
        if self.share_att_key:
            pos_query_states = self.transpose_for_q_scores(self.q(rel_embeddings)).repeat(btz, 1, 1, 1)
            pos_key_states = self.transpose_for_k_scores(self.k(rel_embeddings)).repeat(btz, 1, 1, 1)
        else:
            pass
        score = 0
        if 'c2p' in self.pos_att_type:
            scale = torch.sqrt(torch.tensor(d_head, dtype=torch.float) * scale_factor)
            c2p_att = torch.matmul(query_states, pos_key_states.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_pos.expand([btz, n_head, q_len, k_len]))
            score += c2p_att / scale
        if 'p2c' in self.pos_att_type:
            scale = torch.sqrt(torch.tensor(d_head, dtype=torch.float) * scale_factor)
            if k_len != q_len:
                r_pos = self.relative_positions(k_len, k_len)
                r_pos = r_pos.unsqueeze(0)
            else:
                r_pos = relative_pos
            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = torch.matmul(key_states, pos_query_states.transpose(-1, -2))
            p2c_att = torch.gather(p2c_att, dim=-1, index=p2c_pos.squeeze(0).expand([btz, n_head, k_len, k_len])).transpose(-1, -2)
            score += p2c_att / scale
        return score


class ALiBiPositionsEncoding(nn.Module):
    """ALiBi: Attention with Linear Biases
       https://github.com/ofirpress/attention_with_linear_biases
    """

    def __init__(self, n_head, **kwargs) ->None:
        super().__init__()
        self.n_head = n_head
        self.max_cache_pos = -1

    def _get_interleave(self, n):

        def _get_interleave_power_of_2(n):
            start = 2 ** -2 ** -(math.log2(n) - 3)
            ratio = start
            return [(start * ratio ** i) for i in range(n)]
        if math.log2(n).is_integer():
            return _get_interleave_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return _get_interleave_power_of_2(closest_power_of_2) + self._get_interleave(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

    def _gen_alibi_mask(self, seq_len):
        slopes = torch.Tensor(self._get_interleave(self.n_head))
        alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(seq_len).unsqueeze(0).unsqueeze(0).expand(self.n_head, -1, -1)
        alibi = alibi.view(self.n_head, 1, seq_len)
        return alibi

    def forward(self, key_layer):
        """
        key_layer: [btz, n_head, q_len, hdsz]
        """
        seq_length_with_past = key_layer.shape[2]
        if seq_length_with_past > self.max_cache_pos:
            self.max_cache_pos = seq_length_with_past
            self.future_mask = self._gen_alibi_mask(seq_length_with_past)
        mask = self.future_mask[:self.n_head, :seq_length_with_past, :seq_length_with_past]
        return mask.unsqueeze(0)


class AlibiAttention(MultiHeadAttention):
    """alibi相对位置编码"""

    def init_position_encoding(self, **kwargs):
        self.relative_positions_encoding = ALiBiPositionsEncoding(self.num_attention_heads)

    def apply_relative_pos_emb(self, query_states, key_states, attention_scores):
        attention_scores = self.apply_alibi_pos_emb(attention_scores, key_states)
        return attention_scores

    def apply_alibi_pos_emb(self, attention_scores, key_states):
        """ 执行alibi相对位置编码，单独拎出来主要是falcon是在+之后再执行attention_scale的 """
        attention_scores = self.apply_attention_scale(attention_scores)
        key_position_scores_r_t = self.relative_positions_encoding(key_states)
        attention_scores = attention_scores + key_position_scores_r_t
        attention_scores = torch.max(attention_scores, torch.tensor(torch.finfo(attention_scores.dtype).min))
        self.attention_scale = False
        return attention_scores


def get_sinusoid_encoding_table(n_position: 'int', d_hid: 'int', base: 'float'=10000.0, ntk_alpha: 'float'=1.0, scaling_factor: 'float'=1.0, padding_idx: 'Optional[int]'=None):
    """ sinusoid编码
        
        :param n_position: int, 位置长度
        :param d_hid: int, 位置编码长度
        :param padding_idx: padding的token_ids
        :param ntk_alpha: int, 要扩展的倍数
        :param scaling_factor: int, chatglm中32k的插值
        :return: [seq_len, d_hid]
    """
    if ntk_alpha is not None and ntk_alpha != 1:
        base = base * ntk_alpha ** (d_hid / (d_hid - 2))
    inv_freq = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(base) / d_hid))
    if scaling_factor is not None and scaling_factor != 1:
        inv_freq = inv_freq / scaling_factor
    position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
    embeddings_table = torch.zeros(n_position, d_hid)
    embeddings_table[:, 0::2] = torch.sin(position * inv_freq)
    embeddings_table[:, 1::2] = torch.cos(position * inv_freq)
    return embeddings_table
    position_ids = torch.arange(0, n_position).unsqueeze(1)
    position_ids = position_ids.expand(-1, d_hid)
    indices = torch.arange(0, d_hid)
    position_ids = position_ids * torch.pow(10000, -2 * torch.true_divide(torch.floor_divide(indices, 2), d_hid))
    position_ids[:, ::2] = torch.sin(position_ids[:, ::2])
    position_ids[:, 1::2] = torch.cos(position_ids[:, 1::2])
    return position_ids


class NezhaPositionsEncoding(nn.Module):
    """nezha用的google相对位置编码
    来自论文：https://arxiv.org/abs/1803.02155
    """

    def __init__(self, qlen, klen, embedding_size, max_relative_position=127):
        super(NezhaPositionsEncoding, self).__init__()
        vocab_size = max_relative_position * 2 + 1
        distance_mat = torch.arange(klen)[None, :] - torch.arange(qlen)[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
        final_mat = distance_mat_clipped + max_relative_position
        embeddings_table = get_sinusoid_encoding_table(vocab_size, embedding_size)
        position_embeddings = nn.Embedding.from_pretrained(embeddings_table, freeze=True)(final_mat)
        self.register_buffer('position_embeddings', position_embeddings)

    def forward(self, qlen, klen):
        return self.position_embeddings[:qlen, :klen, :]


class NezhaTypicalRelativeAttention(MultiHeadAttention):

    def init_position_encoding(self, **kwargs):
        self.relative_positions_encoding = NezhaPositionsEncoding(qlen=self.max_position, klen=self.max_position, embedding_size=self.attention_head_size, max_relative_position=kwargs.get('max_relative_position'))

    def apply_relative_pos_emb(self, query_states, key_states, attention_scores):
        if not hasattr(self, 'relative_positions_encoding'):
            return attention_scores
        relations_keys = self.relative_positions_encoding(attention_scores.shape[-1], attention_scores.shape[-1])
        key_position_scores_r_t = torch.einsum('bnih,ijh->bnij', query_states, relations_keys)
        attention_scores = attention_scores + key_position_scores_r_t
        return attention_scores

    def torch_attention_forward(self, query_states: 'torch.FloatTensor', key_states: 'torch.FloatTensor', value_states: 'torch.FloatTensor', attention_mask: 'torch.Tensor'):
        """qkv attention: torch原生实现"""
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        attention_scores = self.apply_relative_pos_emb(query_states, key_states, attention_scores)
        if self.attention_scale:
            attention_scores = self.apply_attention_scale(attention_scores)
        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask) * torch.finfo(query_states.dtype).min
            attention_scores = attention_scores + attention_mask
        attention_probs = F.softmax(attention_scores, dim=-1, dtype=query_states.dtype)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_states)
        relations_values = self.relative_positions_encoding(attention_scores.shape[-1], attention_scores.shape[-1])
        value_position_scores_r_t = torch.einsum('bnij,ijh->bnih', attention_probs, relations_values)
        context_layer = context_layer + value_position_scores_r_t
        return context_layer, attention_scores


class RopePositionEncoding(nn.Module):
    """旋转式位置编码: https://kexue.fm/archives/8265

    :param embedding_size: embedding的大小
    :param rope_rank: 排序的方式，目前支持'adjacent', 'updown/rotate_half'
    :param ntk_alpha: ntk外推的alpha
    :param scaling_factor: 对position_ids进行缩放的尺度参数
    :param rope_theta: rope中使用的base大小
    """

    def __init__(self, embedding_size: 'int', max_position: 'int'=2048, rope_rank: "Literal['adjacent', 'updown', 'rotate_half']"='adjacent', scaling_factor: 'float'=1.0, rope_theta: 'float'=10000.0, device=None, **kwargs):
        super(RopePositionEncoding, self).__init__()
        self.embedding_size = embedding_size
        self.rope_rank = rope_rank or 'adjacent'
        assert self.rope_rank in {'adjacent', 'updown', 'rotate_half'}, "rank kwarg only support 'adjacent/updown/rotate_half' "
        self.ntk_alpha = 1.0
        self.scaling_factor = scaling_factor
        self.rope_theta = rope_theta or 10000.0
        self.max_position = max_position
        self.max_seq_len_cache = max_position
        self._set_cos_sin_cache(max_position, device=device, dtype=torch.get_default_dtype())

    def get_sinusoid_encoding_table(self, n_position: 'int', d_hid: 'int', base: 'float'=10000.0, ntk_alpha: 'float'=1.0, scaling_factor: 'float'=1.0, padding_idx: 'Optional[int]'=None):
        """这里重新定义主要是为了后续继承"""
        return get_sinusoid_encoding_table(n_position, d_hid, base, ntk_alpha=ntk_alpha, scaling_factor=scaling_factor)

    def _set_cos_sin_cache(self, seq_len, device=None, dtype=None):
        position_embeddings = self.get_sinusoid_encoding_table(seq_len, self.embedding_size, base=self.rope_theta, ntk_alpha=self.ntk_alpha, scaling_factor=self.scaling_factor)
        if self.rope_rank == 'adjacent':
            cos_cache = position_embeddings[:, 1::2].repeat_interleave(2, dim=-1)
            sin_cache = position_embeddings[:, ::2].repeat_interleave(2, dim=-1)
        elif self.rope_rank in {'updown', 'rotate_half'}:
            cos_cache = position_embeddings[:, 1::2].repeat(1, 2)
            sin_cache = position_embeddings[:, ::2].repeat(1, 2)
        self._register_buffer(cos_cache, sin_cache, device, dtype)

    def _register_buffer(self, cos_cache, sin_cache, device=None, dtype=None):
        if device is not None:
            cos_cache, sin_cache = cos_cache, sin_cache
        if dtype is not None:
            cos_cache, sin_cache = cos_cache, sin_cache
        self.register_buffer('cos_cache', cos_cache, persistent=False)
        self.register_buffer('sin_cache', sin_cache, persistent=False)

    def rotate_and_compute(self, x, cos, sin):
        if self.rope_rank == 'adjacent':
            x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)
        elif self.rope_rank in {'updown', 'rotate_half'}:
            x2 = torch.cat([-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]], dim=-1)
        return x * cos + x2 * sin

    def forward(self, qk: 'Union[torch.Tensor, List[torch.Tensor]]', position_ids: 'torch.Tensor'=None, seq_len: 'int'=None, seq_dim: 'int'=-2):
        """修改了原有的q和k重复走一遍embedding，实现加速"""
        if isinstance(qk, list):
            device, dtype = qk[0].device, qk[0].dtype
        else:
            device, dtype = qk.device, qk.dtype
        if seq_len is None:
            if position_ids is not None:
                seq_len = position_ids.max() + 1
            elif isinstance(qk, list):
                seq_len = qk[0].shape[seq_dim]
            elif isinstance(qk, torch.Tensor):
                seq_len = qk.shape[seq_dim]
        if seq_len > self.max_seq_len_cache:
            self.max_seq_len_cache = seq_len
            self._set_cos_sin_cache(seq_len, device, dtype)
        if self.cos_cache.dtype != dtype or self.cos_cache.device != device:
            self._register_buffer(self.cos_cache, self.sin_cache, device, dtype)
        if position_ids is not None:
            cos = F.embedding(position_ids, self.cos_cache)
            sin = F.embedding(position_ids, self.sin_cache)
        else:
            cos = self.cos_cache[:seq_len]
            sin = self.sin_cache[:seq_len]
        if cos.dim() < qk[0].dim() if isinstance(qk, list) else qk.dim():
            cos = cos.unsqueeze(seq_dim - 1)
            sin = sin.unsqueeze(seq_dim - 1)
        if isinstance(qk, list):
            return [self.rotate_and_compute(x, cos, sin) for x in qk]
        else:
            return self.rotate_and_compute(qk, cos, sin)


class RopeDynamicNTKScalingPositionEncoding(RopePositionEncoding):
    """使用Dynamic NTK scaling的rope"""

    def __init__(self, embedding_size: 'int', max_position: 'int'=2048, rope_rank: "Literal['adjacent', 'updown', 'rotate_half']"='adjacent', scaling_factor: 'float'=1.0, rope_theta: 'float'=10000.0, **kwargs):
        self.scaling_factor_raw = scaling_factor
        scaling_factor = 1.0
        super().__init__(embedding_size, max_position, rope_rank, scaling_factor, rope_theta, **kwargs)

    def _set_cos_sin_cache(self, seq_len, device=None, dtype=None):
        self.ntk_alpha = self.scaling_factor_raw * seq_len / self.max_position - (self.scaling_factor_raw - 1)
        return super()._set_cos_sin_cache(seq_len, device, dtype)


class RopeDynamicNTKScalingQwenPositionEncoding(RopePositionEncoding):
    """使用Dynamic NTK scaling的rope (Qwen版)"""

    def _set_cos_sin_cache(self, seq_len, device=None, dtype=None):
        context_value = math.log(seq_len / self.max_position, 2) + 1
        ntk_alpha = max(2 ** math.ceil(context_value) - 1, 1)
        if ntk_alpha != self.ntk_alpha:
            self.ntk_alpha = ntk_alpha
        return super()._set_cos_sin_cache(seq_len, device, dtype)


class RopeGlmPositionEncoding(RopePositionEncoding):
    """GLM对应的rope编码"""

    def __init__(self, embedding_size: 'int', max_position: 'int'=2048, rope_rank: "Literal['adjacent', 'updown', 'rotate_half']"='adjacent', scaling_factor: 'float'=1, rope_theta: 'float'=10000, device=None, **kwargs):
        super().__init__(embedding_size // 2, max_position, rope_rank, scaling_factor, rope_theta, device, **kwargs)

    def forward(self, qk: 'Union[torch.Tensor, List[torch.Tensor]]', position_ids: 'torch.Tensor'=None, seq_len: 'int'=None, seq_dim: 'int'=-2):
        query_states, key_states = qk
        q1, q2 = query_states.chunk(2, dim=query_states.ndim - 1)
        k1, k2 = key_states.chunk(2, dim=key_states.ndim - 1)
        if len(position_ids.shape) == 3:
            q1, k1 = super().forward([q1, k1], position_ids[:, 0, :], seq_len)
            q2, k2 = super().forward([q2, k2], position_ids[:, 1, :], seq_len)
        else:
            q1, k1 = super().forward([q1, k1], position_ids, seq_len)
        query_states = torch.concat([q1, q2], dim=q1.ndim - 1)
        key_states = torch.concat([k1, k2], dim=k1.ndim - 1)
        return query_states, key_states


class RopeLinearScalingPositionEncoding(RopePositionEncoding):
    """使用linear scaling的rope, scaling_factor != 1的时候生效"""
    pass


class RopeLlama3PositionEncoding(RopePositionEncoding):
    """使用llama3的rope"""

    def __init__(self, embedding_size: 'int', max_position: 'int'=2048, rope_rank: "Literal['adjacent', 'updown', 'rotate_half']"='adjacent', scaling_factor: 'float'=1.0, rope_theta: 'float'=10000.0, **kwargs):
        self.low_freq_factor = kwargs['low_freq_factor']
        self.high_freq_factor = kwargs['high_freq_factor']
        self.old_context_len = kwargs['original_max_position_embeddings']
        super().__init__(embedding_size, max_position, rope_rank, scaling_factor, rope_theta, **kwargs)

    def get_sinusoid_encoding_table(self, n_position: 'int', d_hid: 'int', base: 'float'=10000, ntk_alpha: 'float'=1, scaling_factor: 'float'=1, padding_idx: 'Optional[int]'=None):
        if ntk_alpha is not None and ntk_alpha != 1:
            base = base * ntk_alpha ** (d_hid / (d_hid - 2))
        scaling_factor = scaling_factor or 1
        position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
        inv_freq = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(base) / d_hid))
        low_freq_wavelen = self.old_context_len / self.low_freq_factor
        high_freq_wavelen = self.old_context_len / self.high_freq_factor
        new_freqs = []
        for freq in inv_freq:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / scaling_factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (self.old_context_len / wavelen - self.low_freq_factor) / (self.high_freq_factor - self.low_freq_factor)
                new_freqs.append((1 - smooth) * freq / scaling_factor + smooth * freq)
        inv_freq = torch.tensor(new_freqs, dtype=inv_freq.dtype, device=inv_freq.device)
        embeddings_table = torch.zeros(n_position, d_hid)
        if scaling_factor is not None and scaling_factor != 1:
            position = position / scaling_factor
        embeddings_table[:, 0::2] = torch.sin(position * inv_freq)
        embeddings_table[:, 1::2] = torch.cos(position * inv_freq)
        return embeddings_table


class RopeMropePositionEncoding(RopePositionEncoding):
    """qwen2vl中使用"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mrope_section = kwargs.get('mrope_section')

    def forward(self, qk: 'Union[torch.Tensor, List[torch.Tensor]]', position_ids: 'torch.Tensor'=None, seq_len: 'int'=None, seq_dim: 'int'=-2):
        if isinstance(qk, list):
            device, dtype = qk[0].device, qk[0].dtype
        else:
            device, dtype = qk.device, qk.dtype
        if seq_len is None:
            if position_ids is not None:
                seq_len = position_ids.max() + 1
            elif isinstance(qk, list):
                seq_len = qk[0].shape[seq_dim]
            elif isinstance(qk, torch.Tensor):
                seq_len = qk.shape[seq_dim]
        if seq_len > self.max_seq_len_cache:
            self.max_seq_len_cache = seq_len
            self._set_cos_sin_cache(seq_len, device, dtype)
        if self.cos_cache.dtype != dtype or self.cos_cache.device != device:
            self._register_buffer(self.cos_cache, self.sin_cache, device, dtype)
        if position_ids is not None:
            cos = F.embedding(position_ids, self.cos_cache)
            sin = F.embedding(position_ids, self.sin_cache)
        else:
            cos = self.cos_cache[:seq_len]
            sin = self.sin_cache[:seq_len]
        if cos.dim() < qk[0].dim() if isinstance(qk, list) else qk.dim():
            cos = cos.unsqueeze(seq_dim - 1)
            sin = sin.unsqueeze(seq_dim - 1)
        mrope_section = self.mrope_section * 2
        cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(1)
        sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(1)
        if isinstance(qk, list):
            return [self.rotate_and_compute(x, cos, sin) for x in qk]
        else:
            return self.rotate_and_compute(qk, cos, sin)


class RopeYarnPositionEncoding(RopePositionEncoding):
    """DeepSeekV2中使用"""

    def __init__(self, embedding_size, max_position=2048, rope_rank: "Literal['adjacent', 'updown', 'rotate_half']"='adjacent', scaling_factor=1.0, rope_theta=10000, original_max_position_embeddings=4096, beta_fast=32, beta_slow=1, mscale=1, mscale_all_dim=0, **kwargs):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(embedding_size, max_position, rope_rank, scaling_factor, rope_theta, **kwargs)

    @staticmethod
    def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
        return dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def yarn_find_correction_range(self, low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
        low = math.floor(self.yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
        high = math.ceil(self.yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
        return max(low, 0), min(high, dim - 1)

    @staticmethod
    def yarn_get_mscale(scale=1, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    @staticmethod
    def yarn_linear_ramp_mask(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    def _set_cos_sin_cache(self, seq_len, device='cpu', dtype=torch.float32):
        self.max_seq_len_cache = seq_len
        dim = self.embedding_size
        freq_extra = 1.0 / self.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        freq_inter = 1.0 / (self.scaling_factor * self.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        low, high = self.yarn_find_correction_range(self.beta_fast, self.beta_slow, dim, self.rope_theta, self.original_max_position_embeddings)
        inv_freq_mask = 1.0 - self.yarn_linear_ramp_mask(low, high, dim // 2)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        _mscale = float(self.yarn_get_mscale(self.scaling_factor, self.mscale) / self.yarn_get_mscale(self.scaling_factor, self.mscale_all_dim))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cache', emb.cos() * _mscale, persistent=False)
        self.register_buffer('sin_cache', emb.sin() * _mscale, persistent=False)

    def rotate_and_compute(self, x, cos, sin):
        b, h, s, d = x.shape
        x = x.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        x2 = torch.cat([-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]], dim=-1)
        return x * cos + x2 * sin


ROPE_ENCODGING_MAP = {None: RopePositionEncoding, 'linear': RopeLinearScalingPositionEncoding, 'dynamic': RopeDynamicNTKScalingPositionEncoding, 'dynamic_qwen': RopeDynamicNTKScalingQwenPositionEncoding, 'llama3': RopeLlama3PositionEncoding, 'yarn': RopeYarnPositionEncoding, 'mrope': RopeMropePositionEncoding, 'glm': RopeGlmPositionEncoding}


class RopeAttention(MultiHeadAttention):

    def init_position_encoding(self, **kwargs):
        rope_scaling = copy.deepcopy(self.rope_scaling)
        scaling_type = rope_scaling.pop('rope_type', rope_scaling.pop('type', None))
        scaling_factor = rope_scaling.pop('factor', None)
        rope_theta = kwargs.get('rope_theta')
        rope_rank = kwargs.get('rope_rank')
        if scaling_type is None:
            assert scaling_factor is None, f'Args `rope_scaling.factor` not supported in standard rope'
        elif scaling_type in {'linear', 'dynamic'}:
            assert scaling_factor is not None and scaling_factor != 1, f'Args `rope_scaling.factor`={scaling_factor} which is illegal'
        self.relative_positions_encoding = ROPE_ENCODGING_MAP[scaling_type](embedding_size=self.attention_head_size, max_position=self.max_position, rope_rank=rope_rank, scaling_factor=scaling_factor, rope_theta=rope_theta, **rope_scaling)

    def _get_qkv_states(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, position_ids):
        query_states = self.transpose_for_q_scores(self.q(hidden_states))
        key_states = self.transpose_for_k_scores(self.k(hidden_states))
        value_states = self.transpose_for_v_scores(self.v(hidden_states))
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        query_states, key_states = self.relative_positions_encoding([query_states, key_states], position_ids, kv_seq_len)
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        return query_states, key_states, value_states, attention_mask


def _gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def linear_act(x):
    return x


def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


def swiglu(x, dim=-1):
    x = torch.chunk(x, 2, dim=dim)
    return silu(x[0]) * x[1]


ACT2FN = {'relu': nn.functional.relu, 'silu': silu, 'swish': silu, 'swiglu': swiglu, 'gelu': gelu, 'tanh': torch.tanh, 'gelu_new': _gelu_new, 'gelu_fast': gelu_fast, 'quick_gelu': quick_gelu, 'mish': mish, 'linear': linear_act, 'sigmoid': torch.sigmoid, 'softmax': nn.Softmax(dim=-1)}


def get_activation(activation_string):
    """根据activation_string返回对应的激活函数

    :param activation_string: str, 传入的激活函数名
    :return: Any
    """
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f'function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}')


class GatedAttention(nn.Module):
    """门控注意力单元
    链接：https://arxiv.org/abs/2202.10447
    介绍：https://kexue.fm/archives/8934
    说明：没有加入加性相对位置编码
    参考pytorch项目：https://github.com/lucidrains/FLASH-pytorch
    """

    def __init__(self, hidden_size, attention_key_size, intermediate_size, attention_probs_dropout_prob, hidden_act, is_dropout=False, attention_scale=True, bias=True, normalization='softmax_plus', **kwargs):
        super().__init__()
        self.intermediate_size = intermediate_size
        self.attention_head_size = attention_key_size
        self.attention_scale = attention_scale
        self.is_dropout = is_dropout
        self.normalization = normalization
        self.hidden_fn = get_activation(hidden_act)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.i_dense = nn.Linear(hidden_size, self.intermediate_size * 2 + attention_key_size, bias=bias)
        self.offsetscale = self.OffsetScale(attention_key_size, heads=2, bias=bias)
        self.o_dense = nn.Linear(self.intermediate_size, hidden_size, bias=bias)
        self.p_bias = kwargs.get('p_bias')
        if self.p_bias == 'rotary':
            self.relative_positions_encoding = RopePositionEncoding(embedding_size=self.attention_head_size, **kwargs)

    def forward(self, hidden_states, attention_mask):
        hidden_states = self.hidden_fn(self.i_dense(hidden_states))
        u, v, qk = hidden_states.split([self.intermediate_size, self.intermediate_size, self.attention_head_size], dim=-1)
        q, k = self.offsetscale(qk)
        if self.p_bias == 'rotary':
            q = self.relative_positions_encoding(q)
            k = self.relative_positions_encoding(k)
        attention_scores = torch.einsum('b i d, b j d -> b i j', q, k)
        if self.attention_scale:
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask) * -1000000000000.0
            attention_scores = attention_scores + attention_mask.squeeze(1)
        attention_scores = self.attention_normalize(attention_scores, -1, self.normalization)
        if self.is_dropout:
            attention_scores = self.dropout(attention_scores)
        out = self.o_dense(u * torch.einsum('b i j, b j d -> b i d', attention_scores, v))
        return out

    def attention_normalize(self, a, dim=-1, method='softmax'):
        """不同的注意力归一化方案
        softmax：常规/标准的指数归一化；
        squared_relu：来自 https://arxiv.org/abs/2202.10447 ；
        softmax_plus：来自 https://kexue.fm/archives/8823 。
        """
        if method == 'softmax':
            return F.softmax(a, dim=dim)
        else:
            mask = (a > -100000000000.0).float()
            l = torch.maximum(torch.sum(mask, dim=dim, keepdims=True), torch.tensor(1))
            if method == 'squared_relu':
                return F.relu(a) ** 2 / l
            elif method == 'softmax_plus':
                return F.softmax(a * torch.log(l) / torch.log(torch.tensor(512.0)), dim=dim)
        return a


    class OffsetScale(nn.Module):
        """仿射变换"""

        def __init__(self, head_size, heads=1, bias=True):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(heads, head_size))
            self.bias = bias
            if bias:
                self.beta = nn.Parameter(torch.zeros(heads, head_size))
            nn.init.normal_(self.gamma, std=0.02)

        def forward(self, x):
            out = torch.einsum('... d, h d -> ... h d', x, self.gamma)
            if self.bias:
                out = out + self.beta
            return out.unbind(dim=-2)


class TransformerxlMultiHeadAttn(MultiHeadAttention):
    """Transformer_XL式相对位置编码RelPartialLearnableMultiHeadAttn, 这里修改成了MultiHeadAttention的batch_first代码格式"""

    def __init__(self, *args, r_w_bias=None, r_r_bias=None, r_s_bias=None, **kwargs):
        super().__init__(*args, **kwargs)
        segment_vocab_size = kwargs.get('segment_vocab_size')
        if r_r_bias is None or r_w_bias is None:
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.num_attention_heads, self.attention_head_size))
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.num_attention_heads, self.attention_head_size))
            if segment_vocab_size > 0:
                self.r_s_bias = nn.Parameter(torch.FloatTensor(self.num_attention_heads, self.attention_head_size))
        else:
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias
            self.r_s_bias = r_s_bias
        if segment_vocab_size > 0:
            self.seg_embed = nn.Parameter(torch.FloatTensor(segment_vocab_size, self.num_attention_heads, self.attention_head_size))
        self.r = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        self.rel_shift_opt = kwargs.get('rel_shift_opt')

    @staticmethod
    def rel_shift(x, zero_triu=False):
        """transformer_xl使用, 向左shift让右上角都是0, 对角线是同一个值, x: [btz, n_head, q_len, k_len]"""
        q_len, k_len = x.size(2), x.size(-1)
        zero_pad = torch.zeros((*x.size()[:2], q_len, 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(*x.size()[:2], k_len + 1, q_len)
        x = x_padded[:, :, 1:, :].view_as(x)
        if zero_triu:
            ones = torch.ones((q_len, k_len), device=x.device)
            x = x * torch.tril(ones, k_len - q_len)[None, None, :, :]
        return x

    @staticmethod
    def rel_shift_bnij(x, klen=-1):
        """ xlnet使用"""
        x_size = x.shape
        x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        x = x[:, :, 1:, :]
        x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        return x

    def forward(self, w, cat, r, attention_mask=None, seg_mat=None):
        qlen, rlen, bsz = w.size(1), r.size(0), w.size(0)
        mixed_query_layer = self.q(cat)[:, -qlen:, :]
        mixed_key_layer = self.k(cat)
        mixed_value_layer = self.v(cat)
        w_head_q = self.transpose_for_q_scores(mixed_query_layer)
        w_head_k = self.transpose_for_k_scores(mixed_key_layer)
        w_head_v = self.transpose_for_v_scores(mixed_value_layer)
        r_head_k = self.r(r)
        r_head_k = r_head_k.view(rlen, self.num_attention_heads, self.attention_head_size)
        rw_head_q = w_head_q + self.r_w_bias.unsqueeze(1)
        AC = torch.einsum('bnid,bnjd->bnij', (rw_head_q, w_head_k))
        rr_head_q = w_head_q + self.r_r_bias.unsqueeze(1)
        BD = torch.einsum('bnid,jnd->bnij', (rr_head_q, r_head_k))
        BD = self.rel_shift_bnij(BD, klen=AC.shape[3]) if self.rel_shift_opt == 'xlnet' else self.rel_shift(BD)
        if hasattr(self, 'seg_embed') and self.r_r_bias is not None:
            seg_mat = F.one_hot(seg_mat, 2).float()
            EF = torch.einsum('bnid,snd->ibns', w_head_q + self.r_s_bias.unsqueeze(1), self.seg_embed)
            EF = torch.einsum('bijs,ibns->bnij', seg_mat, EF)
        else:
            EF = 0
        attention_scores = AC + BD + EF
        if self.attention_scale:
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None and attention_mask.any().item():
            attention_mask = 1.0 - attention_mask
            attention_scores = attention_scores.float().masked_fill(attention_mask.bool(), -1e+30).type_as(attention_mask)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, w_head_v)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (self.o(context_layer), attention_scores) if self.output_attentions else (self.o(context_layer),)
        return outputs


class LayerNorm(nn.Module):

    def __init__(self, hidden_size: 'int', eps: 'float'=1e-12, conditional_size: 'Union[bool, int]'=False, weight: 'bool'=True, bias: 'bool'=True, norm_mode: "Literal['normal', 'torch_buildin', 'rmsnorm']"='normal', **kwargs):
        """ layernorm层，自行实现是为了兼容conditianal layernorm，使得可以做条件文本生成、条件分类等任务

            :param hidden_size: int, layernorm的神经元个数
            :param eps: float
            :param conditional_size: int, condition layernorm的神经元个数; 详情：https://spaces.ac.cn/archives/7124
            :param weight: bool, 是否包含权重
            :param bias: bool, 是否包含偏置
            :param norm_mode: str, `normal`, `rmsnorm`, `torch_buildin`
        """
        super(LayerNorm, self).__init__()
        assert norm_mode in {'normal', 'rmsnorm', 'torch_buildin'}, f'Args norm_mode:{norm_mode} not supported'
        self.normalized_shape = hidden_size,
        self.norm_mode = norm_mode
        self.eps = eps
        self.conditional_size = conditional_size
        if weight:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        if conditional_size:
            self.dense1 = nn.Linear(conditional_size, hidden_size, bias=False)
            self.dense1.weight.data.uniform_(0, 0)
            self.dense2 = nn.Linear(conditional_size, hidden_size, bias=False)
            self.dense2.weight.data.uniform_(0, 0)

    def forward(self, hidden_states: 'torch.FloatTensor', cond: 'Optional[torch.Tensor]'=None):
        if isinstance(hidden_states, (list, tuple)):
            cond = hidden_states[1] if self.conditional_size else None
            hidden_states = hidden_states[0]
        if self.norm_mode == 'torch_buildin':
            return F.layer_norm(hidden_states, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.norm_mode == 'rmsnorm':
            hidden_states_fp32 = hidden_states.float()
            variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
            o = (hidden_states_fp32 * torch.rsqrt(variance + self.eps)).type_as(hidden_states)
        else:
            u = hidden_states.mean(-1, keepdim=True)
            s = (hidden_states - u).pow(2).mean(-1, keepdim=True)
            o = (hidden_states - u) / torch.sqrt(s + self.eps)
        if not hasattr(self, 'weight'):
            self.weight = 1
        if self.conditional_size and cond is not None:
            for _ in range(len(hidden_states.shape) - len(cond.shape)):
                cond = cond.unsqueeze(dim=1)
            output = (self.weight + self.dense1(cond)) * o + self.dense2(cond)
        else:
            output = self.weight * o
        if hasattr(self, 'bias') and self.bias is not None:
            output += self.bias
        return output if output.dtype == hidden_states.dtype else output.type_as(hidden_states)


class DeepseekV2Attention(MultiHeadAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_lora_rank = kwargs.get('q_lora_rank')
        self.kv_lora_rank = kwargs.get('kv_lora_rank')
        self.qk_nope_head_dim = kwargs.get('qk_nope_head_dim')
        self.qk_rope_head_dim = kwargs.get('qk_rope_head_dim')
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        layer_norm_eps = kwargs.get('layer_norm_eps', 1e-06)
        if self.q_lora_rank is None:
            self.q = nn.Linear(self.hidden_size, self.num_attention_heads * self.q_head_dim, bias=self.bias)
        else:
            del self.q
            self.q_a = nn.Linear(self.hidden_size, self.q_lora_rank, bias=self.bias)
            self.q_a_layernorm = LayerNorm(self.q_lora_rank, norm_mode='rmsnorm', eps=layer_norm_eps, bias=self.bias)
            self.q_b = nn.Linear(self.q_lora_rank, self.attention_key_size * self.q_head_dim, bias=self.bias)
        del self.k, self.v
        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=self.bias)
        self.kv_a_layernorm = LayerNorm(self.kv_lora_rank, norm_mode='rmsnorm', eps=layer_norm_eps, bias=self.bias)
        self.kv_b = nn.Linear(self.kv_lora_rank, self.num_attention_heads * (self.q_head_dim - self.qk_rope_head_dim + self.attention_head_size), bias=self.bias)
        self.o = nn.Linear(self.num_attention_heads * self.attention_head_size, self.hidden_size, bias=self.bias)
        self.softmax_scale = self.q_head_dim ** -0.5
        if self.rope_scaling is not None:
            mscale_all_dim = self.rope_scaling.get('mscale_all_dim', 0)
            scaling_factor = self.rope_scaling['factor']
            if mscale_all_dim:
                mscale = 1.0 if scaling_factor <= 1 else 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
                self.softmax_scale = self.softmax_scale * mscale * mscale

    def init_position_encoding(self, **kwargs):
        """这里dim为qk_rope_head_dim所以重新初始化了"""
        rope_scaling = copy.deepcopy(self.rope_scaling)
        scaling_type = rope_scaling.pop('rope_type', rope_scaling.pop('type', None))
        scaling_factor = rope_scaling.pop('factor', None)
        rope_theta = kwargs.get('rope_theta')
        rope_rank = kwargs.get('rope_rank')
        self.relative_positions_encoding = ROPE_ENCODGING_MAP[scaling_type](embedding_size=kwargs.get('qk_rope_head_dim'), max_position=self.max_position, rope_rank=rope_rank, scaling_factor=scaling_factor, rope_theta=rope_theta, **rope_scaling)

    def apply_attention_scale(self, attention_scores):
        return attention_scores * self.softmax_scale

    def _get_qkv_states(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, position_ids):
        bsz, q_len, _ = hidden_states.size()
        if self.q_lora_rank is None:
            q = self.q(hidden_states)
        else:
            q = self.q_b(self.q_a_layernorm(self.q_a(hidden_states)))
        q = q.view(bsz, q_len, self.num_attention_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = self.kv_b(self.kv_a_layernorm(compressed_kv)).view(bsz, q_len, self.num_attention_heads, self.qk_nope_head_dim + self.attention_head_size).transpose(1, 2)
        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.attention_head_size], dim=-1)
        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        q_pe, k_pe = self.relative_positions_encoding([q_pe, k_pe], position_ids, kv_seq_len)
        k_pe: 'torch.Tensor'
        query_states = k_pe.new_empty(bsz, self.num_attention_heads, q_len, self.q_head_dim)
        query_states[:, :, :, :self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim:] = q_pe
        key_states = k_pe.new_empty(bsz, self.num_attention_heads, q_len, self.q_head_dim)
        key_states[:, :, :, :self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim:] = k_pe
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        return query_states, key_states, value_states, attention_mask


class T5PositionsEncoding(nn.Module):
    """Google T5的相对位置编码
    来自论文：https://arxiv.org/abs/1910.10683
    """

    def __init__(self, qlen, klen, relative_attention_num_buckets, is_decoder=False):
        super(T5PositionsEncoding, self).__init__()
        context_position = torch.arange(qlen, dtype=torch.long)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position = self._relative_position_bucket(relative_position, bidirectional=not is_decoder, num_buckets=relative_attention_num_buckets)
        self.register_buffer('relative_position', relative_position)

    def forward(self, qlen, klen):
        return self.relative_position[:qlen, :klen]

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """直接来源于transformer
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret


class T5Attention(MultiHeadAttention):

    def init_position_encoding(self, **kwargs):
        self.relative_positions = T5PositionsEncoding(qlen=self.max_position, klen=self.max_position, relative_attention_num_buckets=kwargs.get('relative_attention_num_buckets'), is_decoder=kwargs.get('is_decoder'))
        self.relative_positions_encoding = nn.Embedding(kwargs.get('relative_attention_num_buckets'), self.num_attention_heads)

    def apply_relative_pos_emb(self, query_states, key_states, attention_scores):
        if not hasattr(self, 'relative_positions_encoding'):
            return attention_scores
        relations_keys = self.relative_positions(attention_scores.shape[-1], attention_scores.shape[-1])
        key_position_scores_r_t = self.relative_positions_encoding(relations_keys).permute([2, 0, 1]).unsqueeze(0)
        key_position_scores_r_t = key_position_scores_r_t[:, :, -attention_scores.shape[-2]:, :]
        attention_scores = attention_scores + key_position_scores_r_t
        return attention_scores


class SinusoidalPositionEncoding(nn.Module):
    """定义Sin-Cos位置Embedding
    """

    def __init__(self, max_position, embedding_size):
        super(SinusoidalPositionEncoding, self).__init__()
        self.position_embeddings = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(max_position, embedding_size), freeze=True)

    def forward(self, position_ids):
        return self.position_embeddings(position_ids)


class BertEmbeddings(nn.Module):
    """embeddings层
       构造word, position and token_type embeddings, 一般是token、position、segment三者embedding之和
    """

    def __init__(self, vocab_size: 'int', embedding_size: 'int', hidden_size: 'int', max_position: 'int', segment_vocab_size: 'int', shared_segment_embeddings: 'bool', dropout_rate: 'float', conditional_size: 'Union[bool, int]'=False, pad_token_id: 'int'=0, **kwargs):
        super(BertEmbeddings, self).__init__()
        self.shared_segment_embeddings = shared_segment_embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_token_id)
        if kwargs.get('p_bias') == 'sinusoid':
            self.position_embeddings = SinusoidalPositionEncoding(max_position, embedding_size)
        elif kwargs.get('p_bias') in {'rotary', 'typical_relative', 't5_relative', 'MultiHeadAttention', 'deberta_v2', 'alibi'}:
            pass
        elif max_position > 0:
            self.position_embeddings = nn.Embedding(max_position, embedding_size)
        self.hierarchical_position = kwargs.get('hierarchical_position')
        if segment_vocab_size > 0 and not shared_segment_embeddings and kwargs.get('use_segment_embedding', True):
            self.segment_embeddings = nn.Embedding(segment_vocab_size, embedding_size)
        self.emb_scale = kwargs.get('emb_scale', 1)
        self.layerNorm = LayerNorm(embedding_size, eps=kwargs.get('layer_norm_eps', 1e-12), conditional_size=conditional_size, **kwargs)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else lambda x: x
        if embedding_size != hidden_size:
            self.embedding_hidden_mapping_in = nn.Linear(embedding_size, hidden_size)

    def apply_hierarchical_pos_embedding(self, position_index):
        """层次分解位置代码: https://spaces.ac.cn/archives/7947"""
        alpha = 0.4 if self.hierarchical_position is True else self.hierarchical_position
        embeddings = self.position_embeddings.weight - alpha * self.position_embeddings.weight[:1]
        embeddings = embeddings / (1 - alpha)
        btz, seqlen = position_index.shape
        position_index_reshape = position_index.flatten()[:, None]
        embeddings_x = take_along_dim(embeddings, torch_div(position_index_reshape, embeddings.size(0), rounding_mode='trunc'), dim=0)
        embeddings_y = take_along_dim(embeddings, position_index_reshape % embeddings.size(0), dim=0)
        position_embeddings = alpha * embeddings_x + (1 - alpha) * embeddings_y
        return position_embeddings.reshape(btz, seqlen, -1)

    def apply_embeddings(self, token_ids: 'torch.Tensor', segment_ids: 'torch.Tensor', position_ids: 'torch.Tensor', additional_embs: 'Union[Tuple[torch.Tensor], List[torch.Tensor]]'=None, **kwargs):
        """单独拆分出来，方便下游继承和修改"""
        if not token_ids.requires_grad and token_ids.dtype in {torch.long, torch.int}:
            words_embeddings = self.word_embeddings(token_ids)
        else:
            words_embeddings = token_ids
        if hasattr(self, 'segment_embeddings'):
            segment_ids = torch.zeros_like(token_ids) if segment_ids is None else segment_ids
            segment_embeddings = self.segment_embeddings(segment_ids)
            embeddings = words_embeddings + segment_embeddings
        elif self.shared_segment_embeddings:
            segment_ids = torch.zeros_like(token_ids) if segment_ids is None else segment_ids
            segment_embeddings = self.word_embeddings(segment_ids)
            embeddings = words_embeddings + segment_embeddings
        else:
            embeddings = words_embeddings
        if hasattr(self, 'position_embeddings') and position_ids is not None:
            if position_ids.shape[0] == 1:
                position_ids = position_ids.repeat(token_ids.shape[0], 1)
            if self.hierarchical_position is not None and position_ids.shape[1] > self.position_embeddings.weight.shape[0]:
                position_embeddings = self.apply_hierarchical_pos_embedding(position_ids)
            else:
                position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        if additional_embs is not None:
            for emb in additional_embs:
                embeddings += emb
        return embeddings

    def forward(self, token_ids: 'torch.Tensor'=None, segment_ids: 'torch.Tensor'=None, position_ids: 'torch.Tensor'=None, conditional_emb: 'Optional[torch.Tensor]'=None, additional_embs: 'Union[Tuple[torch.Tensor], List[torch.Tensor]]'=None, attention_mask: 'torch.Tensor'=None, **kwargs):
        embeddings = self.apply_embeddings(token_ids, segment_ids, position_ids, additional_embs, **kwargs)
        if self.emb_scale != 1:
            embeddings = embeddings * self.emb_scale
        if hasattr(self, 'layerNorm'):
            embeddings = self.layerNorm(embeddings, conditional_emb)
        if attention_mask is not None:
            embeddings *= attention_mask[:, 0, 0, :, None]
        if hasattr(self, 'dropout'):
            embeddings = self.dropout(embeddings)
        if hasattr(self, 'embedding_hidden_mapping_in'):
            embeddings = self.embedding_hidden_mapping_in(embeddings)
        return embeddings


class PositionWiseFeedForward(nn.Module):

    def __init__(self, hidden_size: 'int', intermediate_size: 'int', dropout_rate: 'float'=0.5, hidden_act: 'str'='gelu', is_dropout: 'bool'=False, bias: 'bool'=True, **kwargs):
        super(PositionWiseFeedForward, self).__init__()
        self.is_dropout = is_dropout
        self.intermediate_act_fn = get_activation(hidden_act)
        self.intermediateDense = nn.Linear(hidden_size, intermediate_size * 2 if hidden_act == 'swiglu' else intermediate_size, bias=bias)
        self.outputDense = nn.Linear(intermediate_size, hidden_size, bias=bias)
        if self.is_dropout:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        if self.is_dropout:
            x = self.dropout(self.intermediate_act_fn(self.intermediateDense(x)))
        else:
            x = self.intermediate_act_fn(self.intermediateDense(x))
        x = self.outputDense(x)
        return x


class LlamaFeedForward(nn.Module):
    """FeedForward和Bert的不一致，Bert只有两个全连接, LLaMA和Qwen使用"""

    def __init__(self, dim: 'int', intermediate_size: 'int', hidden_act='silu', bias=False, **kwargs):
        super().__init__()
        self.intermediateDense = nn.Linear(dim, intermediate_size, bias=bias)
        self.outputDense = nn.Linear(intermediate_size, dim, bias=bias)
        self.intermediateDense2 = nn.Linear(dim, intermediate_size, bias=bias)
        self.intermediate_act_fn = get_activation(hidden_act)

    def forward(self, x):
        return self.outputDense(self.intermediate_act_fn(self.intermediateDense(x)) * self.intermediateDense2(x))


class T5PositionWiseFeedForward(PositionWiseFeedForward):
    """参考transformer包: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py"""

    def __init__(self, hidden_size, intermediate_size, **kwargs):
        super().__init__(hidden_size, intermediate_size, **kwargs)
        self.intermediateDense = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.intermediateDense1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.outputDense = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        x_gelu = self.intermediate_act_fn(self.intermediateDense(x))
        x_linear = self.intermediateDense1(x)
        x = x_gelu * x_linear
        if self.is_dropout:
            x = self.dropout(x)
        x = self.outputDense(x)
        return x


class CRF(nn.Module):
    """Conditional random field: https://github.com/lonePatient/BERT-NER-Pytorch/blob/master/models/layers/crf.py
    """

    def __init__(self, num_tags: 'int', init_transitions: 'Optional[List[np.ndarray]]'=None, freeze=False) ->None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        if init_transitions is None and not freeze:
            self.start_transitions = nn.Parameter(torch.empty(num_tags))
            self.end_transitions = nn.Parameter(torch.empty(num_tags))
            self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
            nn.init.uniform_(self.start_transitions, -0.1, 0.1)
            nn.init.uniform_(self.end_transitions, -0.1, 0.1)
            nn.init.uniform_(self.transitions, -0.1, 0.1)
        elif init_transitions is not None:
            transitions = torch.tensor(init_transitions[0], dtype=torch.float)
            start_transitions = torch.tensor(init_transitions[1], dtype=torch.float)
            end_transitions = torch.tensor(init_transitions[2], dtype=torch.float)
            if not freeze:
                self.transitions = nn.Parameter(transitions)
                self.start_transitions = nn.Parameter(start_transitions)
                self.end_transitions = nn.Parameter(end_transitions)
            else:
                self.register_buffer('transitions', transitions)
                self.register_buffer('start_transitions', start_transitions)
                self.register_buffer('end_transitions', end_transitions)

    def __repr__(self) ->str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(self, emissions: 'torch.Tensor', mask: 'torch.ByteTensor', tags: 'torch.LongTensor', reduction: 'str'='mean') ->torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
            emissions: [btz, seq_len, num_tags]
            mask: [btz, seq_len]
            tags: [btz, seq_len]
        """
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, tags=tags, mask=mask)
        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        llh = denominator - numerator
        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        return llh.sum() / mask.float().sum()

    def decode(self, emissions: 'torch.Tensor', mask: 'Optional[torch.ByteTensor]'=None, nbest: 'Optional[int]'=None, pad_tag: 'Optional[int]'=None) ->List[List[List[int]]]:
        """Find the most likely tag sequence using Viterbi algorithm."""
        if nbest is None:
            nbest = 1
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.uint8, device=emissions.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, mask=mask)
        best_path = self._viterbi_decode_nbest(emissions, mask, nbest, pad_tag)
        return best_path[0] if nbest == 1 else best_path

    def _validate(self, emissions: 'torch.Tensor', tags: 'Optional[torch.LongTensor]'=None, mask: 'Optional[torch.ByteTensor]'=None) ->None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(f'expected last dimension of emissions is {self.num_tags}, got {emissions.size(2)}')
        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(f'the first two dimensions of emissions and tags must match, got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')
        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(f'the first two dimensions of emissions and mask must match, got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq_bf = mask[:, 0].all()
            if not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(self, emissions: 'torch.Tensor', tags: 'torch.LongTensor', mask: 'torch.ByteTensor') ->torch.Tensor:
        batch_size, seq_length = tags.shape
        mask = mask.float()
        score = self.start_transitions[tags[:, 0]]
        score += emissions[torch.arange(batch_size), 0, tags[:, 0]]
        for i in range(1, seq_length):
            score += self.transitions[tags[:, i - 1], tags[:, i]] * mask[:, i]
            score += emissions[torch.arange(batch_size), i, tags[:, i]] * mask[:, i]
        seq_ends = mask.long().sum(dim=1) - 1
        last_tags = tags[torch.arange(batch_size), seq_ends]
        score += self.end_transitions[last_tags]
        return score

    def _compute_normalizer(self, emissions: 'torch.Tensor', mask: 'torch.ByteTensor') ->torch.Tensor:
        seq_length = emissions.size(1)
        score = self.start_transitions + emissions[:, 0]
        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[:, i].unsqueeze(1).bool(), next_score, score)
        score += self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode_nbest(self, emissions: 'torch.FloatTensor', mask: 'torch.ByteTensor', nbest: 'int', pad_tag: 'Optional[int]'=None) ->List[List[List[int]]]:
        if pad_tag is None:
            pad_tag = 0
        device = emissions.device
        batch_size, seq_length = mask.shape
        score = self.start_transitions + emissions[:, 0]
        history_idx = torch.zeros((batch_size, seq_length, self.num_tags, nbest), dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags, nbest), dtype=torch.long, device=device)
        oor_tag = torch.full((batch_size, seq_length, nbest), pad_tag, dtype=torch.long, device=device)
        for i in range(1, seq_length):
            if i == 1:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[:, i].unsqueeze(1)
                next_score = broadcast_score + self.transitions + broadcast_emission
            else:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[:, i].unsqueeze(1).unsqueeze(2)
                next_score = broadcast_score + self.transitions.unsqueeze(1) + broadcast_emission
            next_score, indices = next_score.view(batch_size, -1, self.num_tags).topk(nbest, dim=1)
            if i == 1:
                score = score.unsqueeze(-1).expand(-1, -1, nbest)
                indices = indices * nbest
            next_score = next_score.transpose(2, 1)
            indices = indices.transpose(2, 1)
            score = torch.where(mask[:, i].unsqueeze(-1).unsqueeze(-1).bool(), next_score, score)
            indices = torch.where(mask[:, i].unsqueeze(-1).unsqueeze(-1).bool(), indices, oor_idx)
            history_idx[:, i - 1] = indices
        end_score = score + self.end_transitions.unsqueeze(-1)
        _, end_tag = end_score.view(batch_size, -1).topk(nbest, dim=1)
        seq_ends = mask.long().sum(dim=1) - 1
        history_idx.scatter_(1, seq_ends.view(-1, 1, 1, 1).expand(-1, 1, self.num_tags, nbest), end_tag.view(-1, 1, 1, nbest).expand(-1, 1, self.num_tags, nbest))
        best_tags_arr = torch.zeros((batch_size, seq_length, nbest), dtype=torch.long, device=device)
        best_tags = torch.arange(nbest, dtype=torch.long, device=device).view(1, -1).expand(batch_size, -1)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[:, idx].view(batch_size, -1), 1, best_tags)
            best_tags_arr[:, idx] = torch_div(best_tags.data.view(batch_size, -1), nbest, rounding_mode='floor')
        return torch.where(mask.unsqueeze(-1).bool(), best_tags_arr, oor_tag).permute(2, 0, 1)


class GlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    参考：https://kexue.fm/archives/8373

    :param hidden_size: 即模型最顶层输出的hidden_size
    :param heads: 在实体识别和关系提取中，分别代表着实体个数和关系个数
    :param head_size: 即每个heads的神经元个数，点积时候使用，相当于attention
    """

    def __init__(self, hidden_size: 'int', heads: 'int', head_size: 'int', RoPE: 'bool'=True, use_bias: 'bool'=True, tril_mask: 'bool'=True):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.dense = nn.Linear(hidden_size, heads * head_size * 2, bias=use_bias)
        if self.RoPE:
            self.position_embedding = RopePositionEncoding(head_size)

    def forward(self, inputs: 'torch.Tensor', mask: 'torch.Tensor'=None):
        """ 
        :param inputs: shape=[..., hdsz]
        :param mask: shape=[btz, seq_len], padding部分为0
        """
        sequence_output = self.dense(inputs)
        sequence_output = torch.stack(torch.chunk(sequence_output, self.heads, dim=-1), dim=-2)
        qw, kw = sequence_output[..., :self.head_size], sequence_output[..., self.head_size:]
        if self.RoPE:
            qw = self.position_embedding(qw.transpose(1, -2)).transpose(1, -2)
            kw = self.position_embedding(kw.transpose(1, -2)).transpose(1, -2)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        if mask is not None:
            attention_mask1 = 1 - mask.unsqueeze(1).unsqueeze(3)
            attention_mask2 = 1 - mask.unsqueeze(1).unsqueeze(2)
            logits = logits.masked_fill(attention_mask1.bool(), value=-float('inf'))
            logits = logits.masked_fill(attention_mask2.bool(), value=-float('inf'))
        if self.tril_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1000000000000.0
        return logits / self.head_size ** 0.5


class EfficientGlobalPointer(nn.Module):
    """更加参数高效的GlobalPointer
    参考：https://kexue.fm/archives/8877
    这里实现和GlobalPointer相似，而未采用原版的奇偶位来取qw和kw，个人理解两种方式是无区别的
    """

    def __init__(self, hidden_size: 'int', heads: 'int', head_size: 'int', RoPE: 'bool'=True, use_bias: 'bool'=True, tril_mask: 'bool'=True):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.p_dense = nn.Linear(hidden_size, head_size * 2, bias=use_bias)
        self.q_dense = nn.Linear(head_size * 2, heads * 2, bias=use_bias)
        if self.RoPE:
            self.position_embedding = RopePositionEncoding(head_size)

    def forward(self, inputs: 'torch.Tensor', mask: 'torch.Tensor'=None):
        """ 
        :param inputs: shape=[..., hdsz]
        :param mask: shape=[btz, seq_len], padding部分为0
        """
        sequence_output = self.p_dense(inputs)
        qw, kw = sequence_output[..., :self.head_size], sequence_output[..., self.head_size:]
        if self.RoPE:
            qw = self.position_embedding(qw)
            kw = self.position_embedding(kw)
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.head_size ** 0.5
        bias_input = self.q_dense(sequence_output)
        bias = torch.stack(torch.chunk(bias_input, self.heads, dim=-1), dim=-2).transpose(1, 2) / 2
        logits = logits.unsqueeze(1) + bias[..., :1] + bias[..., 1:].transpose(2, 3)
        if mask is not None:
            attention_mask1 = 1 - mask.unsqueeze(1).unsqueeze(3)
            attention_mask2 = 1 - mask.unsqueeze(1).unsqueeze(2)
            logits = logits.masked_fill(attention_mask1.bool(), value=-float('inf'))
            logits = logits.masked_fill(attention_mask2.bool(), value=-float('inf'))
        if self.tril_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1000000000000.0
        return logits


class AdaptiveEmbedding(nn.Module):
    """Transformer_XL的自适应embedding, 实现不同区间使用不同的维度
    可以实现如高频词用比如1024或512维，低频词用256或64维, 再用Linear层project到相同的维数
    """

    def __init__(self, vocab_size, embedding_size, hidden_size, cutoffs, div_val=1, sample_softmax=False, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.cutoffs = cutoffs + [vocab_size]
        self.div_val = div_val
        self.hidden_size = hidden_size
        self.emb_scale = hidden_size ** 0.5
        self.cutoff_ends = [0] + self.cutoffs
        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(nn.Embedding(vocab_size, embedding_size, sparse=sample_softmax > 0))
            if hidden_size != embedding_size:
                self.emb_projs.append(nn.Parameter(torch.FloatTensor(hidden_size, embedding_size)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = embedding_size // div_val ** i
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.FloatTensor(hidden_size, d_emb_i)))

    def forward(self, token_ids):
        if self.div_val == 1:
            embed = self.emb_layers[0](token_ids)
            if self.hidden_size != self.embedding_size:
                embed = nn.functional.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = token_ids.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.hidden_size], dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()
                if indices_i.numel() == 0:
                    continue
                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = nn.functional.linear(emb_i, self.emb_projs[i])
                emb_flat.index_copy_(0, indices_i, emb_i)
            embed_shape = token_ids.size() + (self.hidden_size,)
            embed = emb_flat.view(embed_shape)
        embed.mul_(self.emb_scale)
        return embed


class BlockIdentity(nn.Module):
    """ A placeholder identity operator that is argument-insensitive. """

    def __init__(self, *args, **kwargs):
        super(BlockIdentity, self).__init__()

    def forward(self, *args, **kwargs):
        if len(args) > 0 and len(kwargs) > 0:
            return args, kwargs
        elif len(args) > 0:
            return args[0] if len(args) == 1 else args
        elif len(kwargs) > 0:
            return kwargs
        else:
            return None


class TplinkerHandshakingKernel(nn.Module):
    """Tplinker的HandshakingKernel实现"""

    def __init__(self, hidden_size, shaking_type, inner_enc_type=''):
        super().__init__()
        self.shaking_type = shaking_type
        if shaking_type == 'cat':
            self.combine_fc = nn.Linear(hidden_size * 2, hidden_size)
        elif shaking_type == 'cat_plus':
            self.combine_fc = nn.Linear(hidden_size * 3, hidden_size)
        elif shaking_type == 'cln':
            self.tp_cln = LayerNorm(hidden_size, conditional_size=hidden_size)
        elif shaking_type == 'cln_plus':
            self.tp_cln = LayerNorm(hidden_size, conditional_size=hidden_size)
            self.inner_context_cln = LayerNorm(hidden_size, conditional_size=hidden_size)
        self.inner_enc_type = inner_enc_type
        if inner_enc_type == 'mix_pooling':
            self.lamtha = nn.Parameter(torch.rand(hidden_size))
        elif inner_enc_type == 'lstm':
            self.inner_context_lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, bidirectional=False, batch_first=True)

    def enc_inner_hiddens(self, seq_hiddens, inner_enc_type='lstm'):

        def pool(seqence, pooling_type):
            if pooling_type == 'mean_pooling':
                pooling = torch.mean(seqence, dim=-2)
            elif pooling_type == 'max_pooling':
                pooling, _ = torch.max(seqence, dim=-2)
            elif pooling_type == 'mix_pooling':
                pooling = self.lamtha * torch.mean(seqence, dim=-2) + (1 - self.lamtha) * torch.max(seqence, dim=-2)[0]
            return pooling
        if 'pooling' in inner_enc_type:
            inner_context = torch.stack([pool(seq_hiddens[:, :i + 1, :], inner_enc_type) for i in range(seq_hiddens.size()[1])], dim=1)
        elif inner_enc_type == 'lstm':
            inner_context, _ = self.inner_context_lstm(seq_hiddens)
        return inner_context

    def forward(self, seq_hiddens):
        """
        :param seq_hiddens: (batch_size, seq_len, hidden_size)
        :return: shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size) (32, 5+4+3+2+1, 5)
        """
        seq_len = seq_hiddens.size()[-2]
        shaking_hiddens_list = []
        for ind in range(seq_len):
            hidden_each_step = seq_hiddens[:, ind, :]
            visible_hiddens = seq_hiddens[:, ind:, :]
            repeat_hiddens = hidden_each_step[:, None, :].repeat(1, seq_len - ind, 1)
            if self.shaking_type == 'cat':
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens], dim=-1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == 'cat_plus':
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens, inner_context], dim=-1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == 'cln':
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
            elif self.shaking_type == 'cln_plus':
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
                shaking_hiddens = self.inner_context_cln([shaking_hiddens, inner_context])
            shaking_hiddens_list.append(shaking_hiddens)
        long_shaking_hiddens = torch.cat(shaking_hiddens_list, dim=1)
        return long_shaking_hiddens


class MixUp(nn.Module):
    """mixup方法实现
    
    :param method: str, 可选'embed', ’encoder‘分别表示在embedding和encoder层面做mixup, None表示mix后续处理, ’hidden‘表示对隐含层做mixup
    :param alpha: float
    :param layer_mix: None or int, 需要mix的隐含层index
    """

    def __init__(self, method='encoder', alpha=1.0, layer_mix=None):
        super().__init__()
        assert method in {'embed', 'encoder', 'hidden', None}
        self.method = method
        self.alpha = alpha
        self.perm_index = None
        self.lam = 0
        self.layer_mix = layer_mix

    def get_perm(self, inputs):
        if isinstance(inputs, torch.Tensor):
            return inputs[self.perm_index]
        elif isinstance(inputs, (list, tuple)):
            return [(inp[self.perm_index] if isinstance(inp, torch.Tensor) else inp) for inp in inputs]

    def mix_up(self, output, output1):
        if isinstance(output, torch.Tensor):
            return self.lam * output + (1.0 - self.lam) * output1
        elif isinstance(output, (list, tuple)):
            output_final = []
            for i in range(len(output)):
                if output[i] is None:
                    output_final.append(output[i])
                elif not output[i].requires_grad and output[i].dtype in {torch.long, torch.int}:
                    output_final.append(torch.max(output[i], output1[i]))
                else:
                    output_final.append(self.lam * output[i] + (1.0 - self.lam) * output1[i])
            return output_final
        else:
            raise ValueError('Illegal model output')

    def encode(self, model, inputs):
        batch_size = inputs[0].shape[0]
        device = inputs[0].device
        self.lam = np.random.beta(self.alpha, self.alpha)
        self.perm_index = torch.randperm(batch_size)
        if self.method is None:
            output = model(inputs)
            output1 = self.get_perm(output)
            return [output, output1]
        elif self.method == 'encoder':
            output = model(inputs)
            output1 = self.get_perm(output)
            output_final = self.mix_up(output, output1)
        elif self.method == 'embed':
            output = model.apply_embeddings(inputs)
            output1 = self.get_perm(output)
            output_final = self.mix_up(output, output1)
            output_final = model.apply_main_layers(output_final)
            output_final = model.apply_final_layers(output_final)
        elif self.method == 'hidden':
            if self.layer_mix is None:
                try:
                    layer_mix = random.randint(0, len(model.encoderLayer))
                except:
                    warnings.warn('LayerMix random failded')
                    layer_mix = 0
            else:
                layer_mix = self.layer_mix

            def apply_on_layer_end(l_i, output):
                if l_i == layer_mix:
                    output1 = self.get_perm(output)
                    return self.mix_up(output, output1)
                else:
                    return output
            model.apply_on_layer_end = apply_on_layer_end
            output_final = model(inputs)
        return output_final

    def forward(self, criterion, y_pred, y_true):
        """计算loss
        """
        y_true1 = y_true[self.perm_index]
        return self.lam * criterion(y_pred, y_true) + (1 - self.lam) * criterion(y_pred, y_true1)


class ConvLayer(nn.Module):
    """deberta_v2中使用"""

    def __init__(self, hidden_size, dropout_rate=0.1, layer_norm_eps=1e-12, conv_kernel_size=3, conv_groups=1, conv_act='tanh', **kwargs):
        super().__init__()
        kernel_size = conv_kernel_size
        groups = conv_groups
        self.conv_act = conv_act
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=(kernel_size - 1) // 2, groups=groups)
        self.LayerNorm = nn.LayerNorm(hidden_size, layer_norm_eps)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, residual_states, input_mask):
        out = self.conv(hidden_states.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        rmask = (1 - input_mask).bool()
        out.masked_fill_(rmask.unsqueeze(-1).expand(out.size()), 0)
        out = get_activation(self.conv_act)(self.dropout(out))
        layer_norm_input = residual_states + out
        output = self.LayerNorm(layer_norm_input)
        if input_mask is None:
            output_states = output
        else:
            if input_mask.dim() != layer_norm_input.dim():
                if input_mask.dim() == 4:
                    input_mask = input_mask.squeeze(1).squeeze(1)
                input_mask = input_mask.unsqueeze(2)
            input_mask = input_mask
            output_states = output * input_mask
        return output_states


class MultiSampleDropout(nn.Module):
    """multisample dropout (wut): https://arxiv.org/abs/1905.09788"""

    def __init__(self, hidden_size, num_labels, K=5, p=0.5):
        super().__init__()
        self.K = K
        self.dropout = nn.Dropout(p)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input):
        logits = torch.stack([self.classifier(self.dropout(input)) for _ in range(self.K)], dim=0)
        logits = torch.mean(logits, dim=0)
        return logits


class BottleneckAdapterLayer(nn.Module):
    """BottleneckAdapterLayer"""

    def __init__(self, adapter_input_size, bottleneck_size, adapter_non_linearity='gelu'):
        super().__init__()
        self.adapter_input_size = adapter_input_size
        self.bottleneck_size = bottleneck_size
        self.non_linearity = get_activation(adapter_non_linearity)
        self.down_proj = nn.Linear(self.adapter_input_size, self.bottleneck_size)
        self.up_proj = nn.Linear(self.bottleneck_size, self.adapter_input_size)
        self.init_weights()

    def init_weights(self, init_mean=0.0, init_std=0.01):
        self.down_proj.weight.data.normal_(mean=init_mean, std=init_std)
        self.down_proj.bias.data.zero_()
        self.up_proj.weight.data.normal_(mean=init_mean, std=init_std)
        self.up_proj.bias.data.zero_()

    def forward(self, x):
        output = self.up_proj(self.non_linearity(self.down_proj(x)))
        output = x + output
        return output


class NormHead(nn.Module):
    """normalized lm_head，目前是Baichuan2使用"""

    def __init__(self, hidden_size, vocab_size, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((vocab_size, hidden_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.first_flag = True

    def forward(self, hidden_states):
        if self.training:
            norm_weight = nn.functional.normalize(self.weight)
        elif self.first_flag:
            self.first_flag = False
            self.weight.data = nn.functional.normalize(self.weight)
            norm_weight = self.weight
        else:
            norm_weight = self.weight
        return nn.functional.linear(hidden_states, norm_weight)


class MoEGate(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.topk_method = config.get('topk_method', 'greedy')
        self.routed_scaling_factor = config.get('routed_scaling_factor', 1.0)
        self.n_group = config.get('n_group')
        self.topk_group = config.get('topk_group')
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) ->None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32), None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        if self.topk_method == 'greedy':
            topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        elif self.topk_method == 'group_limited_greedy':
            group_scores = scores.view(bsz * seq_len, self.n_group, -1).max(dim=-1).values
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = group_mask.unsqueeze(-1).expand(bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group).reshape(bsz * seq_len, -1)
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)
            topk_weight, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        else:
            topk_weight = topk_weight * self.routed_scaling_factor
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss, 
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class DeepseekMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, **config):
        super().__init__()
        config = DottableDict(config)
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        if hasattr(config, 'ep_size') and config.ep_size > 1:
            assert config.ep_size == dist.get_world_size()
            self.ep_size = config.ep_size
            self.experts_per_rank = config.n_routed_experts // config.ep_size
            self.ep_rank = dist.get_rank()
            self.experts = nn.ModuleList([(LlamaFeedForward(config.hidden_size, intermediate_size=config.moe_intermediate_size) if i >= self.ep_rank * self.experts_per_rank and i < (self.ep_rank + 1) * self.experts_per_rank else None) for i in range(config.n_routed_experts)])
        else:
            self.ep_size = 1
            self.experts_per_rank = config.n_routed_experts
            self.ep_rank = 0
            self.experts = nn.ModuleList([LlamaFeedForward(config.hidden_size, intermediate_size=config.moe_intermediate_size) for _ in range(config.n_routed_experts)])
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = LlamaFeedForward(config.hidden_size, intermediate_size=intermediate_size)

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        sorted_tokens_shape = sorted_tokens.shape
        if self.ep_size > 1:
            tokens_per_ep_rank = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
            tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
            dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)
            output_splits = tokens_per_expert_group.view(self.ep_size, -1).sum(1).cpu().numpy().tolist()
            gathered_tokens = sorted_tokens.new_empty(tokens_per_expert_group.sum(dim=0).cpu().item(), sorted_tokens.shape[1])
            input_split_sizes = tokens_per_ep_rank.cpu().numpy().tolist()
            dist.all_to_all(list(gathered_tokens.split(output_splits)), list(sorted_tokens.split(input_split_sizes)))
            tokens_per_expert_post_gather = tokens_per_expert_group.view(self.ep_size, self.experts_per_rank).sum(dim=0)
            gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
            s = 0
            for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
                gatherd_idxs[s:s + k] = i % self.experts_per_rank
                s += k
            gatherd_idxs = gatherd_idxs.argsort()
            sorted_tokens = gathered_tokens[gatherd_idxs]
            tokens_per_expert = tokens_per_expert_post_gather
        tokens_per_expert = tokens_per_expert.cpu().numpy()
        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx
        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        if self.ep_size > 1:
            new_x = torch.empty_like(outs)
            new_x[gatherd_idxs] = outs
            gathered_tokens = new_x.new_empty(*sorted_tokens_shape)
            dist.all_to_all(list(gathered_tokens.split(input_split_sizes)), list(new_x.split(output_splits)))
            outs = gathered_tokens
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = new_x.view(*topk_ids.shape, -1).type(topk_weight.dtype).mul_(topk_weight.unsqueeze(dim=-1)).sum(dim=1).type(new_x.dtype)
        return final_out


class XlnetPositionsEncoding(nn.Module):
    """Xlnet, transformer_xl使用的相对位置编码
       和SinusoidalPositionEncoding区别是一个是间隔排列, 一个是前后排列
    """

    def __init__(self, embedding_size: 'int'):
        super().__init__()
        self.demb = embedding_size
        inv_freq = 1 / 10000 ** (torch.arange(0.0, embedding_size, 2.0) / embedding_size)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb


ATTENTION_MAP = {'MultiHeadAttention': MultiHeadAttention, 'GatedAttention': GatedAttention, 'TransformerxlMultiHeadAttn': TransformerxlMultiHeadAttn, 'DeepseekV2Attention': DeepseekV2Attention, 'DebertaV2Attention': DebertaV2Attention, 'AlibiAttention': AlibiAttention, 'NezhaTypicalRelativeAttention': NezhaTypicalRelativeAttention, 'RopeAttention': RopeAttention, 'T5Attention': T5Attention, 'deberta_v2': DebertaV2Attention, 'alibi': AlibiAttention, 'typical_relative': NezhaTypicalRelativeAttention, 'rotary': RopeAttention, 't5_relative': T5Attention}


class BertLayer(nn.Module):
    """Transformer层:
        顺序为: Attention --> Add --> LayerNorm --> Feed Forward --> Add --> LayerNorm

        :param hidden_size: int, 隐含层神经元个数
        :param num_attention_heads: int, 多头注意力的多头数
        :param attention_probs_dropout_prob: float，softmax后的dropout rate
        :param dropout_rate: float, 残差连接中对multiHeadAttention或者mlp添加dropout的rate
        :param intermediate_size: int, mlp中间隐含层的神经元个数，一般是hidden_size的数倍
        :param hidden_act: str，激活函数的种类
        :param is_dropout: bool, mlp中是否使用dropout层，默认为False
        :param conditional_size: bool/int，LayerNorm时候是否使用条件LayerNorm, 默认为False
        :param pre_layernorm: bool, layernorm是pre还是post，bert是post，现在大模型基本都是pre, 默认为False表示post_layernorm
        :param apply_residual_post_layernorm: bool，残差连接时候是使用layernorm前的还是后的hidden_states, 默认为False表示使用layernorm前的

        注意:
        1. 以上都不计dropout层，并不代表没有dropout，每一层的dropout使用略有不同，注意区分
        2. 原始的Transformer的encoder中的Feed Forward层一共有两层linear，
        3. config.intermediate_size的大小不仅是第一层linear的输出尺寸，也是第二层linear的输入尺寸
    """

    def __init__(self, hidden_size: 'int', num_attention_heads: 'int', dropout_rate: 'float', attention_probs_dropout_prob: 'float', intermediate_size: 'int', hidden_act: 'str', is_dropout: 'bool'=False, conditional_size: 'Union[bool, int]'=False, pre_layernorm: 'bool'=False, apply_residual_post_layernorm: 'bool'=False, **kwargs):
        super(BertLayer, self).__init__()
        self.dropout_rate = dropout_rate
        layer_norm_eps = kwargs.get('layer_norm_eps', 1e-12)
        self.pre_layernorm = pre_layernorm
        self.apply_residual_post_layernorm = apply_residual_post_layernorm
        self.is_decoder = kwargs.get('is_decoder', False)
        self.add_cross_attention = kwargs.get('add_cross_attention', False)
        self.attn_type = kwargs.get('attn_type', kwargs.get('p_bias', 'MultiHeadAttention'))
        self.mlp_type = kwargs.get('mlp_type', 'PositionWiseFeedForward')
        self.multiHeadAttention = ATTENTION_MAP[self.attn_type](hidden_size, num_attention_heads, attention_probs_dropout_prob, dropout_rate, **kwargs)
        self.attnLayerNorm = LayerNorm(hidden_size, eps=layer_norm_eps, conditional_size=conditional_size, **kwargs)
        if self.mlp_type == 'PositionWiseFeedForward':
            self.feedForward = PositionWiseFeedForward(hidden_size, intermediate_size, dropout_rate, hidden_act, is_dropout=is_dropout, **kwargs)
        elif self.mlp_type == 'LlamaFeedForward':
            self.feedForward = LlamaFeedForward(hidden_size, intermediate_size, hidden_act, kwargs['bias'])
        else:
            raise ValueError(f'mlp_type={self.mlp_type} not supported')
        self.ffnLayerNorm = LayerNorm(hidden_size, eps=layer_norm_eps, conditional_size=conditional_size, **kwargs)
        if self.add_cross_attention and self.is_decoder:
            self.crossAttention = ATTENTION_MAP[self.attn_type](hidden_size, num_attention_heads, attention_probs_dropout_prob, dropout_rate, **kwargs)
            self.crossLayerNorm = LayerNorm(hidden_size, eps=layer_norm_eps, conditional_size=conditional_size, **kwargs)

    def forward(self, hidden_states: 'torch.FloatTensor'=None, attention_mask: 'torch.Tensor'=None, position_ids: 'torch.FloatTensor'=None, conditional_emb: 'Optional[torch.Tensor]'=None, encoder_hidden_states=None, encoder_attention_mask: 'Optional[torch.FloatTensor]'=None, past_key_value: 'Optional[Tuple[Tuple[torch.FloatTensor]]]'=None, cross_past_key_value: 'Optional[Tuple[Tuple[torch.FloatTensor]]]'=None, **model_kwargs):
        return_tensors = dict()
        if self.pre_layernorm:
            x = self.attnLayerNorm(hidden_states, conditional_emb)
        else:
            x = hidden_states
        self_attn_output = self.multiHeadAttention(x, attention_mask, past_key_value=past_key_value, position_ids=position_ids)
        residual = x if self.apply_residual_post_layernorm else hidden_states
        hidden_states = self.dropout_add(self_attn_output[0], residual)
        if not self.pre_layernorm:
            hidden_states = self.attnLayerNorm(hidden_states, conditional_emb)
        if self.is_decoder and encoder_hidden_states is not None:
            if self.pre_layernorm:
                x = self.crossLayerNorm(hidden_states, conditional_emb)
            else:
                x = hidden_states
            cross_attn_output = self.crossAttention(x, None, encoder_hidden_states, encoder_attention_mask, cross_past_key_value, position_ids=position_ids)
            residual = x if self.apply_residual_post_layernorm else hidden_states
            hidden_states = self.dropout_add(cross_attn_output[0], residual)
            if model_kwargs.get('use_states', False):
                return_tensors['cross_past_key_value'] = cross_attn_output[-1]
            if not self.pre_layernorm:
                hidden_states = self.crossLayerNorm(hidden_states, conditional_emb)
        if self.pre_layernorm:
            x = self.ffnLayerNorm(hidden_states, conditional_emb)
        else:
            x = hidden_states
        feedforward_output = self.feedForward(x)
        residual = x if self.apply_residual_post_layernorm else hidden_states
        hidden_states = self.dropout_add(feedforward_output, residual)
        if not self.pre_layernorm:
            hidden_states = self.ffnLayerNorm(hidden_states, conditional_emb)
        if self.is_decoder and model_kwargs.get('use_states', False):
            return_tensors['past_key_value'] = self_attn_output[-1]
        return_tensors['hidden_states'] = hidden_states
        return return_tensors

    def dropout_add(self, x: 'torch.Tensor', residual: 'torch.Tensor') ->torch.Tensor:
        out = F.dropout(x, p=self.dropout_rate, training=self.training)
        out = residual + out
        return out


class T5Layer(BertLayer):
    """T5的Encoder的主体是基于Self-Attention的模块
    顺序：LN --> Att --> Add --> LN --> FFN --> Add
    """

    def __init__(self, *args, version='t5.1.0', **kwargs):
        super().__init__(*args, **kwargs)
        if version.endswith('t5.1.1'):
            self.feedForward = T5PositionWiseFeedForward(**kwargs)
        if self.add_cross_attention and self.is_decoder and hasattr(self.crossAttention, 'relative_positions_encoding'):
            del self.crossAttention.relative_positions_encoding
            del self.crossAttention.relative_positions

    def forward(self, hidden_states=None, attention_mask=None, conditional_emb=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, cross_past_key_value=None, **model_kwargs):
        x = self.attnLayerNorm(hidden_states, conditional_emb)
        self_attn_output = self.multiHeadAttention(x, attention_mask, past_key_value=past_key_value)
        hidden_states = self.dropout_add(self_attn_output[0], hidden_states)
        if self.is_decoder and encoder_hidden_states is not None:
            x = self.crossLayerNorm(hidden_states, conditional_emb)
            cross_attn_output = self.crossAttention(x, None, encoder_hidden_states, encoder_attention_mask, cross_past_key_value)
            hidden_states = self.dropout_add(cross_attn_output[0], hidden_states)
            if model_kwargs.get('use_states', False):
                model_kwargs['cross_past_key_value'] = cross_attn_output[-1]
        x = self.ffnLayerNorm(hidden_states, conditional_emb)
        ffn_output = self.feedForward(x)
        hidden_states = self.dropout_add(ffn_output, hidden_states)
        if self.is_decoder and model_kwargs.get('use_states', False):
            model_kwargs['past_key_value'] = self_attn_output[-1]
        model_kwargs['hidden_states'] = hidden_states
        return model_kwargs


class XlnetLayer(BertLayer):
    """Transformer_XL层
    顺序为: Attention --> Add --> LayerNorm --> Feed Forward --> Add --> LayerNorm
    """

    def __init__(self, hidden_size, num_attention_heads, dropout_rate, attention_probs_dropout_prob, intermediate_size, hidden_act, **kwargs):
        super().__init__(hidden_size, num_attention_heads, dropout_rate, attention_probs_dropout_prob, intermediate_size, hidden_act, **kwargs)
        self.pre_layernorm = kwargs.get('pre_layernorm')
        self.multiHeadAttention = TransformerxlMultiHeadAttn(hidden_size, num_attention_heads, attention_probs_dropout_prob, bias=False, **kwargs)

    def forward(self, hidden_states=None, segment_ids=None, pos_emb=None, attention_mask=None, mems_i=None, conditional_emb=None, **model_kwargs):
        hidden_states_cat = torch.cat([mems_i, hidden_states], 1) if mems_i is not None else hidden_states
        if self.pre_layernorm:
            hidden_states_cat = self.attnLayerNorm(hidden_states_cat, conditional_emb)
        self_attn_output = self.multiHeadAttention(hidden_states, hidden_states_cat, pos_emb, attention_mask, segment_ids)
        hidden_states = self.dropout_add(self_attn_output[0], hidden_states)
        if not self.pre_layernorm:
            hidden_states = self.attnLayerNorm(hidden_states, conditional_emb)
        x = self.ffnLayerNorm(hidden_states, conditional_emb) if self.pre_layernorm else hidden_states
        self_attn_output2 = self.feedForward(x)
        hidden_states = self.dropout_add(self_attn_output2, hidden_states)
        if not self.pre_layernorm:
            hidden_states = self.ffnLayerNorm(hidden_states, conditional_emb)
        model_kwargs['hidden_states'] = hidden_states
        return model_kwargs


class MiniCPMLayer(BertLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_depth = kwargs.get('scale_depth')
        self.num_hidden_layers = kwargs['num_hidden_layers']

    def dropout_add(self, x: 'torch.Tensor', residual: 'torch.Tensor') ->torch.Tensor:
        return residual + x * (self.scale_depth / math.sqrt(self.num_hidden_layers))


class FalconParallelAttnLayer(BertLayer):
    """适用于Falcon的transformer block
    主要区别是attention和feedForward是平行的
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attnLayerNorm.bias = nn.Parameter(torch.zeros(kwargs['hidden_size']))
        del self.ffnLayerNorm

    def forward(self, hidden_states=None, attention_mask=None, position_ids=None, conditional_emb=None, past_key_value=None, **model_kwargs):
        x = self.attnLayerNorm(hidden_states, conditional_emb)
        self_attn_output = self.multiHeadAttention(x, attention_mask, past_key_value=past_key_value, position_ids=position_ids)
        feedforward_output = self.feedForward(x)
        feedforward_output += self_attn_output[0]
        hidden_states = self.dropout_add(feedforward_output, hidden_states)
        if self.is_decoder and model_kwargs.get('use_states', False):
            model_kwargs['past_key_value'] = self_attn_output[-1]
        model_kwargs['hidden_states'] = hidden_states
        return model_kwargs


class GlmLayer(BertLayer):
    """顺序：LN --> Att --> Add --> LN --> FFN --> Add"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_hidden_layers = kwargs['num_hidden_layers']
        hidden_size, eps = kwargs['hidden_size'], kwargs.get('layer_norm_eps', 1e-05)
        self.attnLayerNorm = torch.nn.LayerNorm(hidden_size, eps=eps)
        self.ffnLayerNorm = torch.nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, hidden_states=None, attention_mask=None, past_key_value=None, **model_kwargs):
        x = self.attnLayerNorm(hidden_states)
        alpha = (2 * self.num_hidden_layers) ** 0.5
        self_attn_output = self.multiHeadAttention(x, attention_mask, past_key_value=past_key_value, **model_kwargs)
        hidden_states = x * alpha + self_attn_output[0]
        x = self.ffnLayerNorm(hidden_states)
        hidden_states = x * alpha + self.feedForward(x)
        if self.is_decoder and model_kwargs.get('use_states', False):
            model_kwargs['past_key_value'] = self_attn_output[-1]
        model_kwargs['hidden_states'] = hidden_states
        return model_kwargs


class Glm2Layer(BertLayer):
    """顺序：LN --> Att --> Add --> LN --> FFN --> Add"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_size, eps = kwargs['hidden_size'], kwargs.get('layer_norm_eps', 1e-05)
        self.attnLayerNorm = LayerNorm(hidden_size, eps=eps, norm_mode='rmsnorm', bias=False)
        self.ffnLayerNorm = LayerNorm(hidden_size, eps=eps, norm_mode='rmsnorm', bias=False)
        self.multiHeadAttention.o.register_parameter('bias', None)
        self.feedForward.intermediateDense.register_parameter('bias', None)
        self.feedForward.outputDense.register_parameter('bias', None)


class Gpt2MlLayer(BertLayer):
    """未定义在layer.py中是因为该层针对gpt2_ml模型，不可复用；
    顺序：Att --> Add --> LN --> FFN --> Add --> LN
    """

    def forward(self, hidden_states=None, attention_mask=None, conditional_emb=None, past_key_value=None, **model_kwargs):
        self_attn_output = self.multiHeadAttention(hidden_states, attention_mask, past_key_value=past_key_value)
        hidden_states = self.dropout_add(self_attn_output[0], hidden_states)
        x = self.attnLayerNorm(hidden_states, conditional_emb)
        ffn_output = self.feedForward(x)
        hidden_states = self.dropout_add(ffn_output, hidden_states)
        hidden_states = self.ffnLayerNorm(hidden_states, conditional_emb)
        if self.is_decoder and model_kwargs.get('use_states', False):
            model_kwargs['past_key_value'] = self_attn_output[-1]
        model_kwargs['hidden_states'] = hidden_states
        return model_kwargs


class GAULayer(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.gau = GatedAttention(**kwargs)
        self.dropout_rate = kwargs.get('dropout_rate')
        self.attnLayerNorm = LayerNorm(**kwargs)

    def forward(self, hidden_states=None, attention_mask=None, conditional_emb=None, **model_kwargs):
        gau_hidden_states = self.gau(hidden_states, attention_mask)
        hidden_states = hidden_states + F.dropout(gau_hidden_states, p=self.dropout_rate, training=self.training)
        hidden_states = self.attnLayerNorm(hidden_states, conditional_emb)
        model_kwargs['hidden_states'] = hidden_states
        return model_kwargs


class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""

    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        :param input: torch.Tensor, shape=[N, C]
        :param target: torch.Tensor, shape=[N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index)
        return loss


class LabelSmoothingCrossEntropy(nn.Module):

    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        :param output: torch.Tensor, shape=[N, C]
        :param target: torch.Tensor, shape=[N, ]
        """
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction, ignore_index=self.ignore_index)


class MultilabelCategoricalCrossentropy(nn.Module):
    """多标签分类的交叉熵；
    说明：y_true和y_pred的shape一致，y_true的元素非0即1， 1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax！预测阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解本文。
    参考：https://kexue.fm/archives/7359
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pred, y_true):
        """
        :param y_true: torch.Tensor, [..., num_classes]
        :param y_pred: torch.Tensor: [..., num_classes]
        """
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_pos = y_pred - (1 - y_true) * 1000000000000.0
        y_pred_neg = y_pred - y_true * 1000000000000.0
        y_pred_pos = torch.cat([y_pred_pos, torch.zeros_like(y_pred_pos[..., :1])], dim=-1)
        y_pred_neg = torch.cat([y_pred_neg, torch.zeros_like(y_pred_neg[..., :1])], dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        return (pos_loss + neg_loss).mean()


class SparseMultilabelCategoricalCrossentropy(nn.Module):
    """稀疏版多标签分类的交叉熵；
       请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax，预测阶段则输出y_pred大于0的类；
       详情请看：https://kexue.fm/archives/7359，https://kexue.fm/archives/8888
    """

    def __init__(self, mask_zero=False, epsilon=1e-07, **kwargs):
        super().__init__(**kwargs)
        self.mask_zero = mask_zero
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        """
        :param y_true: shape=[..., num_positive]
        :param y_pred: shape=[..., num_classes]
        """
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred = torch.cat([y_pred, zeros], dim=-1)
        if self.mask_zero:
            infs = zeros + float('inf')
            y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = torch.gather(y_pred, dim=-1, index=y_true)
        y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
        if self.mask_zero:
            y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
            y_pos_2 = torch.gather(y_pred, dim=-1, index=y_true)
        pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
        all_loss = torch.logsumexp(y_pred, dim=-1)
        aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
        aux_loss = torch.clamp(1 - torch.exp(aux_loss), self.epsilon, 1)
        neg_loss = all_loss + torch.log(aux_loss)
        return pos_loss + neg_loss


class ContrastiveLoss(nn.Module):
    """对比损失：减小正例之间的距离，增大正例和反例之间的距离
    公式：labels * distance_matrix.pow(2) + (1-labels)*F.relu(margin-distance_matrix).pow(2)
    https://www.sbert.net/docs/package_reference/losses.html

    :param margin: float, 距离参数，distance>margin时候不参加梯度回传，默认为0.5
    :param size_average: bool, 是否对loss在样本维度上求均值，默认为True
    :param online: bool, 是否使用OnlineContrastiveLoss, 即仅计算困难样本的loss, 默认为False
    """

    def __init__(self, margin=0.5, size_average=True, online=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average
        self.online = online

    def forward(self, distances, labels, pos_id=1, neg_id=0):
        """
        :param distances: torch.Tensor, 样本对之间的预测距离, shape=[btz]
        :param labels: torch.Tensor, 样本对之间的真实距离, shape=[btz]
        :param pos_id: int, 正样本的label
        :param neg_id: int, 负样本的label
        """
        if not self.online:
            losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
            return losses.mean() if self.size_average else losses.sum()
        else:
            negs = distances[labels == neg_id]
            poss = distances[labels == pos_id]
            negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
            positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]
            positive_loss = positive_pairs.pow(2).sum()
            negative_loss = F.relu(self.margin - negative_pairs).pow(2).sum()
            return positive_loss + negative_loss


class RDropLoss(nn.Module):
    """R-Drop的Loss实现，官方项目：https://github.com/dropreg/R-Drop

    :param alpha: float, 控制rdrop的loss的比例
    :param rank: str, 指示y_pred的排列方式, 支持['adjacent', 'updown']
    """

    def __init__(self, alpha=4, rank='adjacent'):
        super().__init__()
        self.alpha = alpha
        assert rank in {'adjacent', 'updown'}, "rank kwarg only support 'adjacent' and 'updown' "
        self.rank = rank
        self.loss_sup = nn.CrossEntropyLoss()
        self.loss_rdrop = nn.KLDivLoss(reduction='none')

    def forward(self, *args):
        """支持两种方式: 一种是y_pred, y_true, 另一种是y_pred1, y_pred2, y_true

        :param y_pred: torch.Tensor, 第一种方式的样本预测值, shape=[btz*2, num_labels]
        :param y_true: torch.Tensor, 样本真实值, 第一种方式shape=[btz*2,], 第二种方式shape=[btz,]
        :param y_pred1: torch.Tensor, 第二种方式的样本预测值, shape=[btz, num_labels]
        :param y_pred2: torch.Tensor, 第二种方式的样本预测值, shape=[btz, num_labels]
        """
        assert len(args) in {2, 3}, 'RDropLoss only support 2 or 3 input args'
        if len(args) == 2:
            y_pred, y_true = args
            loss_sup = self.loss_sup(y_pred, y_true)
            if self.rank == 'adjacent':
                y_pred1 = y_pred[1::2]
                y_pred2 = y_pred[::2]
            elif self.rank == 'updown':
                half_btz = y_true.shape[0] // 2
                y_pred1 = y_pred[:half_btz]
                y_pred2 = y_pred[half_btz:]
        else:
            y_pred1, y_pred2, y_true = args
            loss_sup = (self.loss_sup(y_pred1, y_true) + self.loss_sup(y_pred2, y_true)) / 2
        loss_rdrop1 = self.loss_rdrop(F.log_softmax(y_pred1, dim=-1), F.softmax(y_pred2, dim=-1))
        loss_rdrop2 = self.loss_rdrop(F.log_softmax(y_pred2, dim=-1), F.softmax(y_pred1, dim=-1))
        return loss_sup + torch.mean(loss_rdrop1 + loss_rdrop2) / 4 * self.alpha


class UDALoss(nn.Module):
    """UDALoss，使用时候需要继承一下，因为forward需要使用到global_step和total_steps
    https://arxiv.org/abs/1904.12848

    :param tsa_schedule: str, tsa策略，可选['linear_schedule', 'exp_schedule', 'log_schedule']
    :param start_p: float, tsa生效概率下限, 默认为0
    :param end_p: float, tsa生效概率上限, 默认为1
    :param return_all_loss: bool, 是否返回所有的loss，默认为True
    :return: loss
    """

    def __init__(self, tsa_schedule=None, start_p=0, end_p=1, return_all_loss=True):
        super().__init__()
        self.loss_sup = nn.CrossEntropyLoss()
        self.loss_unsup = nn.KLDivLoss(reduction='batchmean')
        self.tsa_schedule = tsa_schedule
        self.start = start_p
        self.end = end_p
        if self.tsa_schedule:
            assert self.tsa_schedule in {'linear_schedule', 'exp_schedule', 'log_schedule'}, 'tsa_schedule config illegal'
        self.return_all_loss = return_all_loss

    def forward(self, y_pred, y_true_sup, global_step, total_steps):
        """ y_pred由[pred_sup, true_unsup, pred_unsup]三部分组成
        
        :param y_pred: torch.Tensor, 样本预测值, shape=[btz_sup+btz_unsup*2, num_labels]
        :param y_true_sup: torch.Tensor, 样本真实值, shape=[btz_sup,]
        :param global_step: int, 当前训练步数
        :param total_steps: int, 训练总步数
        """
        sup_size = y_true_sup.size(0)
        unsup_size = (y_pred.size(0) - sup_size) // 2
        y_pred_sup = y_pred[:sup_size]
        if self.tsa_schedule is None:
            loss_sup = self.loss_sup(y_pred_sup, y_true_sup)
        else:
            threshold = self.get_tsa_threshold(self.tsa_schedule, global_step, total_steps, self.start, self.end)
            true_prob = torch.gather(F.softmax(y_pred_sup, dim=-1), dim=1, index=y_true_sup[:, None])
            sel_rows = true_prob.lt(threshold).sum(dim=-1).gt(0)
            loss_sup = self.loss_sup(y_pred_sup[sel_rows], y_true_sup[sel_rows]) if sel_rows.sum() > 0 else 0
        y_true_unsup = y_pred[sup_size:sup_size + unsup_size]
        y_true_unsup = F.softmax(y_true_unsup.detach(), dim=-1)
        y_pred_unsup = F.log_softmax(y_pred[sup_size + unsup_size:], dim=-1)
        loss_unsup = self.loss_unsup(y_pred_unsup, y_true_unsup)
        if self.return_all_loss:
            return loss_sup + loss_unsup, loss_sup, loss_unsup
        else:
            return loss_sup + loss_unsup

    @staticmethod
    def get_tsa_threshold(schedule, global_step, num_train_steps, start, end):
        training_progress = global_step / num_train_steps
        if schedule == 'linear_schedule':
            threshold = training_progress
        elif schedule == 'exp_schedule':
            scale = 5
            threshold = math.exp((training_progress - 1) * scale)
        elif schedule == 'log_schedule':
            scale = 5
            threshold = 1 - math.exp(-training_progress * scale)
        return threshold * (end - start) + start


class TemporalEnsemblingLoss(nn.Module):
    """TemporalEnsembling的实现，思路是在监督loss的基础上，增加一个mse的一致性损失loss

       - 官方项目：https://github.com/s-laine/tempens
       - pytorch第三方实现：https://github.com/ferretj/temporal-ensembling
       - 使用的时候，train_dataloader的shffle必须未False
    """

    def __init__(self, epochs, max_val=10.0, ramp_up_mult=-5.0, alpha=0.5, max_batch_num=100, hist_device='cpu'):
        super().__init__()
        self.loss_sup = nn.CrossEntropyLoss()
        self.max_epochs = epochs
        self.max_val = max_val
        self.ramp_up_mult = ramp_up_mult
        self.alpha = alpha
        self.max_batch_num = max_batch_num
        self.hist_unsup = []
        self.hist_sup = []
        self.hist_device = hist_device
        self.hist_input_y = []
        assert (self.alpha >= 0) & (self.alpha < 1)

    def forward(self, y_pred_sup, y_pred_unsup, y_true_sup, epoch, bti):
        """
        :param y_pred_sup: torch.Tensor, 监督学习样本预测值, shape=[btz, num_labels]
        :param y_pred_unsup: torch.Tensor, 无监督学习样本预测值, shape=[btz, num_labels]
        :param y_true_sup: int, 监督学习样本真实值, shape=[btz,]
        :param epoch: int, 当前epoch
        :param bti: int, 当前batch的序号
        """
        self.same_batch_check(y_pred_sup, y_pred_unsup, y_true_sup, bti)
        if self.max_batch_num is None or bti < self.max_batch_num:
            self.init_hist(bti, y_pred_sup, y_pred_unsup)
            sup_ratio = float(len(y_pred_sup)) / (len(y_pred_sup) + len(y_pred_unsup))
            w = self.weight_schedule(epoch, sup_ratio)
            sup_loss, unsup_loss = self.temporal_loss(y_pred_sup, y_pred_unsup, y_true_sup, bti)
            self.hist_unsup[bti] = self.update(self.hist_unsup[bti], y_pred_unsup.detach(), epoch)
            self.hist_sup[bti] = self.update(self.hist_sup[bti], y_pred_sup.detach(), epoch)
            return sup_loss + w * unsup_loss, sup_loss, w * unsup_loss
        else:
            return self.loss_sup(y_pred_sup, y_true_sup)

    def same_batch_check(self, y_pred_sup, y_pred_unsup, y_true_sup, bti):
        """检测数据的前几个batch必须是一致的, 这里写死是10个"""
        if bti >= 10:
            return
        if bti >= len(self.hist_input_y):
            self.hist_input_y.append(y_true_sup)
        else:
            err_msg = 'TemporalEnsemblingLoss requests the same sort dataloader, you may need to set train_dataloader shuffle=False'
            assert self.hist_input_y[bti].equal(y_true_sup), err_msg

    def update(self, hist, y_pred, epoch):
        """更新历史logit，利用alpha门控来控制比例
        """
        Z = self.alpha * hist + (1.0 - self.alpha) * y_pred
        output = Z * (1.0 / (1.0 - self.alpha ** (epoch + 1)))
        return output

    def weight_schedule(self, epoch, sup_ratio):
        max_val = self.max_val * sup_ratio
        if epoch == 0:
            return 0.0
        elif epoch >= self.max_epochs:
            return max_val
        return max_val * np.exp(self.ramp_up_mult * (1.0 - float(epoch) / self.max_epochs) ** 2)

    def temporal_loss(self, y_pred_sup, y_pred_unsup, y_true_sup, bti):

        def mse_loss(out1, out2):
            quad_diff = torch.sum((F.softmax(out1, dim=1) - F.softmax(out2, dim=1)) ** 2)
            return quad_diff / out1.data.nelement()
        sup_loss = self.loss_sup(y_pred_sup, y_true_sup)
        unsup_loss = mse_loss(y_pred_unsup, self.hist_unsup[bti])
        unsup_loss += mse_loss(y_pred_sup, self.hist_sup[bti])
        return sup_loss, unsup_loss

    def init_hist(self, bti, y_pred_sup, y_pred_unsup):
        if bti >= len(self.hist_sup):
            self.hist_sup.append(torch.zeros_like(y_pred_sup))
            self.hist_unsup.append(torch.zeros_like(y_pred_unsup))


class CausalLMLoss(nn.CrossEntropyLoss):
    """Causal语言模型的Loss

    :param offset: 是否对logit和labels做错位处理, 取决于在做数据时候是否已经offset过
    :param logits_index: 如果model.forward()返回了多个, 则logits对应的index
    """

    def __init__(self, *args, offset=False, logits_index=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.offset = offset
        self.logits_index = logits_index

    def forward(self, logits: 'Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]', labels: 'Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]') ->torch.Tensor:
        """
        logits: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]], 形状为[btz, seq_len, vocab_size]
        labels: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
            1) token_ids: [btz, seq_len]
            2) token_ids: [btz, seq_len]  + mask: [btz, seq_len]
        """
        if isinstance(logits, (List, Tuple)):
            logits = logits[self.logits_index]
        assert len(logits.shape) == 3, 'Args `logits` size should be [btz, seq_len, vocab_size]'
        raw_dtyps = logits.dtype
        logits = logits
        mask = None
        if isinstance(labels, (List, Tuple)):
            for item in labels[1:]:
                mask = item if mask is None else mask * item
            labels = labels[0]
        if self.offset:
            logits = logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            if mask is not None:
                mask = mask[:, 1:].contiguous()
        logits = logits.reshape(-1, logits.shape[-1])
        if mask is not None:
            labels = labels * mask
        labels = labels.flatten()
        loss = super().forward(logits, labels)
        return loss


def load_state_dict_into_meta_model(model: 'nn.Module', state_dict: 'dict', device_map: 'dict'=None, dtype: 'Union[str, torch.dtype]'=None, offload_folder=None, state_dict_folder=None, state_dict_index=None, is_safetensors: 'bool'=False):
    """ 把state_dict导入meta_model
    为了代码简洁，这里device_map需要外部手动指定, 形式如{'embeddings.word_embeddings': 0, 'LayerNormFinal': 0, 'lm_head': 0}
    """
    for param_name, param in state_dict.items():
        module_name = param_name
        set_module_kwargs = {'value': param}
        if device_map is None or device_map == 'cpu':
            param_device = 'cpu'
        else:
            while len(module_name) > 0 and module_name not in device_map:
                module_name = '.'.join(module_name.split('.')[:-1])
            if module_name == '' and '' not in device_map:
                raise ValueError(f"{param_name} doesn't have any device set.")
            param_device = device_map[module_name]
        if param_device == 'disk':
            if not is_safetensors:
                offload_index = offload_weight(param, param_name, offload_folder, offload_index)
        elif param_device == 'cpu' and state_dict_index is not None:
            state_dict_index = offload_weight(param, param_name, state_dict_folder, state_dict_index)
        else:
            set_module_kwargs['dtype'] = dtype or param.dtype
            set_module_tensor_to_device(model, param_name, param_device, **set_module_kwargs)


class CacheTensor:

    def __init__(self, *args, **kwargs):
        self.tensor = torch.empty(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.tensor = self.tensor

    def data_ptr(self):
        return self.tensor.data_ptr()


def extract_weight_to_half(weight: 'torch.Tensor', scale_list: 'torch.Tensor', source_bit_width: 'int'):
    if source_bit_width == 8:
        func = kernels.int8WeightExtractionHalf
    elif source_bit_width == 4:
        func = kernels.int4WeightExtractionHalf
    else:
        assert False, 'Unsupported bit-width'
    with torch.device(weight.device):
        n, m = weight.size(0), weight.size(1)
        out = torch.empty(n, m * (8 // source_bit_width), dtype=torch.half, device='cuda')
        stream = torch.cuda.current_stream()
        gridDim = n, 1, 1
        blockDim = min(round_up(m, 32), 1024), 1, 1
        func(gridDim, blockDim, 0, stream, [ctypes.c_void_p(weight.data_ptr()), ctypes.c_void_p(scale_list.data_ptr()), ctypes.c_void_p(out.data_ptr()), ctypes.c_int32(n), ctypes.c_int32(m)])
        return out


class W8A16Linear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp: 'torch.Tensor', quant_w: 'torch.Tensor', scale_w: 'torch.Tensor', weight_bit_width):
        ctx.inp_shape = inp.size()
        ctx.weight_bit_width = weight_bit_width
        out_features = quant_w.size(0)
        inp = inp.contiguous().view(-1, inp.size(-1))
        weight = extract_weight_to_half(quant_w, scale_w, weight_bit_width)
        ctx.weight_shape = weight.size()
        output = inp.mm(weight.t())
        ctx.save_for_backward(inp, quant_w, scale_w)
        return output.view(*(ctx.inp_shape[:-1] + (out_features,)))

    @staticmethod
    def backward(ctx, grad_output: 'torch.Tensor'):
        inp, quant_w, scale_w = ctx.saved_tensors
        weight = extract_weight_to_half(quant_w, scale_w, ctx.weight_bit_width)
        grad_output = grad_output.contiguous().view(-1, weight.size(0))
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(inp)
        return grad_input.view(ctx.inp_shape), grad_weight.view(ctx.weight_shape), None, None


def compress_int4_weight(weight: 'torch.Tensor'):
    with torch.device(weight.device):
        n, m = weight.size(0), weight.size(1)
        assert m % 2 == 0
        m = m // 2
        out = torch.empty(n, m, dtype=torch.int8, device='cuda')
        stream = torch.cuda.current_stream()
        gridDim = n, 1, 1
        blockDim = min(round_up(m, 32), 1024), 1, 1
        kernels.int4WeightCompression(gridDim, blockDim, 0, stream, [ctypes.c_void_p(weight.data_ptr()), ctypes.c_void_p(out.data_ptr()), ctypes.c_int32(n), ctypes.c_int32(m)])
        return out


class QuantizedLinear(Linear):

    def __init__(self, weight_bit_width: 'int', weight_tensor=None, bias_tensor=None, quantized_weight=None, quantized_weight_scale=None, quantization_cache=None, empty_init=False, *args, **kwargs):
        super(QuantizedLinear, self).__init__(*args, **kwargs)
        self.weight_bit_width = weight_bit_width
        self.quantization_cache = quantization_cache
        if quantized_weight is not None and quantized_weight_scale is not None:
            del self.weight
            self.weight = Parameter(quantized_weight, requires_grad=False)
            self.weight_scale = Parameter(quantized_weight_scale, requires_grad=False)
        else:
            shape = self.weight.shape
            del self.weight
            if weight_tensor is None or empty_init:
                self.weight = torch.empty(shape[0], shape[1] * weight_bit_width // 8, dtype=torch.int8, device=kwargs['device'])
                self.weight_scale = torch.empty(shape[0], dtype=kwargs['dtype'], device=kwargs['device'])
            else:
                self.weight_scale = weight_tensor.abs().max(dim=-1).values / (2 ** (weight_bit_width - 1) - 1)
                self.weight = torch.round(weight_tensor / self.weight_scale[:, None])
                if weight_bit_width == 4:
                    self.weight = compress_int4_weight(self.weight)
            self.weight = Parameter(self.weight, requires_grad=False)
            self.weight_scale = Parameter(self.weight_scale, requires_grad=False)
        if bias_tensor is not None:
            self.bias = Parameter(bias_tensor, requires_grad=False)
        else:
            self.bias = None

    def reset_parameters(self):
        """To accelerate initialization"""
        pass

    def forward(self, input):
        output = W8A16Linear.apply(input, self.weight, self.weight_scale, self.weight_bit_width)
        if self.bias is not None:
            output = output + self.bias
        return output

    def _apply(self, fn):
        self_obj = super()._apply(fn)
        if self.quantization_cache is not None:
            self.quantization_cache
            self.quantization_cache
        return self_obj


BAR_TYPE = ['░▝▗▖▘▚▞▛▙█', '░▖▘▝▗▚▞█', ' ▖▘▝▗▚▞█', '░▒█', ' >=', ' ▏▎▍▌▋▊▉█░▏▎▍▌▋▊▉█']


def quantize_cpm_kernels(model: 'nn.Module', quantization_bit: 'int'=8, use_quantization_cache: 'bool'=False, empty_init: 'bool'=False, target_modules: 'Union[str, List]'=None, **kwargs):
    """从chagglm-6b移植过来的的量化，方便以int8和int4进行推理
    源链接：https://huggingface.co/THUDM/chatglm-6b/blob/main/quantization.py
    
    Replace fp16 linear with quantized linear
    这里修改了hard code, 可以适配其他模型
    target_modules: str/list, 指定对某些层做量化
    """
    if not is_package_available('cpm_kernels'):
        raise ModuleNotFoundError('Module `cpm_kernels` not found, you may use `pip install cpm_kernels`')
    modules_trans = {}
    for name, module in model.named_modules():
        if target_modules is None and isinstance(module, Linear):
            modules_trans[name] = module
        elif target_modules is not None and isinstance(module, Linear):
            if isinstance(target_modules, str):
                target_module_found = re.fullmatch(target_modules, name)
            else:
                target_module_found = any(name.endswith(target_key) for target_key in target_modules)
            if target_module_found:
                modules_trans[name] = module
    current_device = torch.cuda.current_device()
    dtype = torch.half
    QuantizedLinearWithPara = partial(QuantizedLinear, weight_bit_width=quantization_bit, bias=True, dtype=dtype, empty_init=empty_init)
    cache = dict()
    for name, module in tqdm(modules_trans.items(), desc='Quantize linear layers'):
        cache_name = re.sub('\\.[0-9]+\\.', '.', name)
        if use_quantization_cache and cache_name not in cache:
            n, m = module.weight.size(0), module.weight.size(1)
            cache[cache_name] = CacheTensor(n, m, dtype=dtype, device=current_device, requires_grad=False)
        module_quant = QuantizedLinearWithPara(weight_tensor=module.weight, bias_tensor=module.bias, in_features=module.in_features, out_features=module.out_features, device=module.weight.device, quantization_cache=cache.get(cache_name))
        del module
        name_new = list(name)
        for iter_ in re.finditer('\\.[0-9]+\\.', name):
            iter_str = name[iter_.start():iter_.end()]
            name_new[iter_.start():iter_.end()] = [''] * (iter_.end() - iter_.start())
            name_new[iter_.start()] = '[' + iter_str[1:-1] + '].'
        exec('model.' + ''.join(name_new) + ' = module_quant')
    return model


class BERT_BASE(nn.Module):
    """模型基类
    """

    def __init__(self, vocab_size: 'int', hidden_size: 'int', num_hidden_layers: 'int', num_attention_heads: 'int', intermediate_size: 'int', hidden_act: 'str', dropout_rate: 'float'=None, attention_probs_dropout_prob: 'float'=None, embedding_size: 'int'=None, attention_head_size: 'int'=None, attention_key_size: 'int'=None, initializer_range: 'float'=0.02, sequence_length: 'int'=None, keep_tokens: 'List[int]'=None, compound_tokens: 'List[int]'=None, residual_attention_scores: 'bool'=False, keep_hidden_layers: 'List[int]'=None, hierarchical_position: 'Union[bool, float]'=None, gradient_checkpoint: 'bool'=False, output_all_encoded_layers: 'bool'=False, tie_word_embeddings: 'bool'=False, return_dict: 'bool'=False, **kwargs):
        super(BERT_BASE, self).__init__()
        if keep_tokens is not None:
            vocab_size = len(keep_tokens)
        if compound_tokens is not None:
            vocab_size += len(compound_tokens)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size or self.hidden_size // self.num_attention_heads
        self.attention_key_size = attention_key_size or self.attention_head_size
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate or 0
        self.attention_probs_dropout_prob = attention_probs_dropout_prob or 0
        self.hidden_act = hidden_act
        self.embedding_size = embedding_size or hidden_size
        self.initializer_range = initializer_range
        self.sequence_length = sequence_length
        self.keep_tokens = keep_tokens
        self.compound_tokens = compound_tokens
        self.attention_bias = None
        self.position_bias = None
        self.attention_scores = None
        self.residual_attention_scores = residual_attention_scores
        self.keep_hidden_layers = set(range(num_hidden_layers)) if keep_hidden_layers is None else set(keep_hidden_layers)
        self.hierarchical_position = hierarchical_position
        self.gradient_checkpoint = gradient_checkpoint
        self.quantized = False
        self.output_all_encoded_layers = output_all_encoded_layers
        self.add_trainer = kwargs['add_trainer']
        self.tie_word_embeddings = tie_word_embeddings or kwargs.get('tie_emb_prj_weight', False)
        self.return_dict = return_dict

    def tie_weights(self):
        pass

    def gradient_checkpointing_enable(self):
        self.gradient_checkpoint = True

    def enable_input_require_grads(self):
        """transformer移植来
        Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
        the model weights fixed.
        """

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)
        self._require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)

    def disable_input_require_grads(self):
        """transformer移植来
        Removes the `_require_grads_hook`.
        """
        self._require_grads_hook.remove()

    def get_kw(self, *args, **kwargs):
        """把self.属性设置到kwargs中, 方便传参"""
        for arg in args:
            kwargs[arg] = getattr(self, arg)
        return kwargs

    def args_segmentate(self, inputs, **model_kwargs):
        """解析输入，转成list，tuple类型"""
        if len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
            return inputs[0]
        return inputs

    def forward(self, *inputs, **model_kwargs):
        """定义模型的训练流程
        
        :param inputs: List[torch.Tensor], 默认顺序是[token_ids, segment_ids(若有), position_ids(若有), custom_attention_mask(若有), conditional_input(若有)]
        :return: List[torch.Tensor] or torch.Tensor, 模型输出，默认顺序为[last_hidden_state/all_encoded_layers, pooled_output(若有), mlm_scores(若有), nsp_scores(若有)]
        """
        inputs = self.args_segmentate(inputs, **model_kwargs)
        model_kwargs = self.apply_embeddings(*inputs, **model_kwargs)
        model_kwargs = self.apply_main_layers(**model_kwargs)
        outputs = self.apply_final_layers(**model_kwargs)
        return outputs

    @torch.no_grad()
    def predict(self, *inputs, **model_kwargs):
        """定义模型的预测流程
        
        :param inputs: List[torch.Tensor], 默认顺序是[token_ids, segment_ids(若有), position_ids(若有), custom_attention_mask(若有), conditional_input(若有)]
        :return: List[torch.Tensor] or torch.Tensor, 模型输出，默认顺序为[last_hidden_state/all_encoded_layers, pooled_output(若有), mlm_scores(若有), nsp_scores(若有)]
        """
        if self.training:
            self.eval()
        return self.forward(*inputs, **model_kwargs)

    def init_model_weights(self, module):
        """ 初始化权重 """
        if isinstance(module, (nn.Linear, nn.Embedding)) and module.weight.requires_grad:
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            if hasattr(module, 'bias') and module.bias is not None and module.bias.requires_grad:
                module.bias.data.zero_()
            if hasattr(module, 'weight') and module.weight.requires_grad:
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None and module.bias.requires_grad:
            module.bias.data.zero_()

    def init_meta_weights(self, module):
        """meta weights初始化, 主要是在量化里面用到
        """
        if hasattr(module, 'weight') and module.weight.device == torch.device('meta'):
            module.to_empty(device='cpu')

    def variable_mapping(self):
        """构建pytorch层与checkpoint的变量名之间的映射表"""
        return {}

    def load_variable(self, *args, **kwargs):
        raise NotImplementedError

    def load_embeddings(self, embeddings):
        """根据keep_tokens和compound_tokens对embedding进行修改"""
        if self.keep_tokens is not None:
            embeddings = embeddings[self.keep_tokens]
        if self.compound_tokens is not None:
            ext_embeddings = []
            for item in self.compound_tokens:
                try:
                    ext_embeddings.append(torch.mean(embeddings[item], 0) * torch.ones_like(embeddings[item]))
                except IndexError:
                    ext_embeddings.append(torch.mean(embeddings, 0, keepdim=True))
                    warnings.warn(f'Initialize ext_embeddings from compound_tokens not in embedding index')
            embeddings = torch.cat([embeddings] + ext_embeddings, 0)
        return embeddings

    def load_trans_ckpt(self, checkpoint: 'str'):
        """加载ckpt并转换
           1. 支持.safe_tensors + .bin
           2. 方便后续各个模型继承并做一些预处理, 如对qkv权重进行split
        """
        return load_checkpoint(checkpoint)

    def from_pretrained_single(self, checkpoint: 'Union[str, os.PathLike]'=None, mapping: 'Union[dict, Callable]'=None, skip_init: 'bool'=False, device_map: 'dict'=None, torch_dtype=None, verbose=1):
        """加载预训练模型(单个权重文件)，根据mapping从checkpoint加载权重"""
        ckpt_state_dict = self.load_trans_ckpt(checkpoint)
        mapping = mapping or self.variable_mapping()
        model_params = set([i[0] for i in self.named_parameters()])
        if isinstance(mapping, dict):
            for layer_name in model_params:
                if layer_name in ckpt_state_dict and layer_name not in mapping:
                    mapping.update({layer_name: layer_name})
        elif isinstance(mapping, Callable):
            mapping = {mapping(k): k for k in ckpt_state_dict}
        else:
            raise TypeError(f'Args `mapping`={type(mapping)} not supported')
        state_dict_new = {}
        missing_keys = []
        over_keys = set(ckpt_state_dict.keys())
        needed_keys = []
        model_state_dict = self.state_dict()
        for new_key, old_key in mapping.items():
            if new_key not in model_state_dict:
                continue
            if old_key in ckpt_state_dict:
                state_dict_new[new_key] = self.load_variable(ckpt_state_dict[old_key], old_key, new_key)
                if old_key in over_keys:
                    over_keys.remove(old_key)
            else:
                missing_keys.append(old_key)
            needed_keys.append(old_key)
        over_keys = list(over_keys)
        del ckpt_state_dict
        gc.collect()
        self._print_mismatch_keys(missing_keys, over_keys, verbose)
        if not skip_init:
            self.load_state_dict(state_dict_new, strict=False)
        else:
            load_state_dict_into_meta_model(self, state_dict_new, device_map=device_map, dtype=torch_dtype, is_safetensors=checkpoint.endswith('.safetensors'))
        del state_dict_new
        gc.collect()
        return missing_keys, over_keys, needed_keys

    def from_pretrained(self, checkpoints: 'Union[str, os.PathLike, list]', mapping: 'Union[dict, Callable]'=None, skip_init: 'bool'=False, device_map: 'dict'=None, torch_dtype=None, verbose=1, **kwargs):
        """加载预训练模型(单个/多个ckpt)"""
        if isinstance(checkpoints, str):
            self.from_pretrained_single(checkpoints, mapping=mapping, skip_init=skip_init, device_map=device_map, torch_dtype=torch_dtype, verbose=verbose)
        elif isinstance(checkpoints, (tuple, list)):
            all_missing_keys, all_over_keys = [], []
            tqdm_checkpoints = tqdm(checkpoints)
            for checkpoint in tqdm_checkpoints:
                tqdm_checkpoints.set_description(f'Loading {os.path.basename(checkpoint)}')
                missing_keys, over_keys, needed_keys = self.from_pretrained_single(checkpoint, mapping=mapping, skip_init=skip_init, device_map=device_map, torch_dtype=torch_dtype, verbose=0)
                all_missing_keys.extend(missing_keys)
                all_over_keys.extend(over_keys)
                if checkpoint == checkpoints[-1]:
                    tqdm_checkpoints.set_description('Loading checkpoint shards')
            all_missing_keys = set(all_missing_keys).difference(set(needed_keys))
            all_over_keys = set(all_over_keys).difference(set(needed_keys))
            self._print_mismatch_keys(all_missing_keys, all_over_keys, verbose)
        else:
            raise ValueError('Args `checkpoint_path` only support `str` or `list(str)` format')
        if device_map is not None and is_accelerate_available():
            device_map_kwargs = {'device_map': device_map, 'offload_dir': kwargs.get('offload_folder'), 'offload_index': kwargs.get('offload_index'), 'offload_buffers': kwargs.get('offload_buffers', False), 'skip_keys': 'past_key_values'}
            dispatch_model(self, **device_map_kwargs)

    def _get_no_split_modules(self, device_map: 'str'):
        """
        Get the modules of the model that should not be spit when using device_map. We iterate through the modules to
        get the underlying `_no_split_modules`.

        Args:
            device_map (`str`):
                The device map value. Options are ["auto", "balanced", "balanced_low_0", "sequential"]

        Returns:
            `List[str]`: List of modules that should not be split
        """
        _no_split_modules = set()
        modules_to_check = [self]
        while len(modules_to_check) > 0:
            module = modules_to_check.pop(-1)
            if module.__class__.__name__ not in _no_split_modules:
                if isinstance(module, BERT_BASE):
                    if module._no_split_modules is None:
                        raise ValueError(f"{module.__class__.__name__} does not support `device_map='{device_map}'`. To implement support, the model class needs to implement the `_no_split_modules` attribute.")
                    else:
                        _no_split_modules = _no_split_modules | set(module._no_split_modules)
                modules_to_check += list(module.children())
        return list(_no_split_modules)

    @staticmethod
    def _print_mismatch_keys(missing_keys, over_keys, verbose):
        """打印mismatch keys"""
        if verbose != 0:
            for key in missing_keys:
                log_warn(f'`{key}` not found in pretrained checkpoints')
        if verbose > 1:
            for key in over_keys:
                log_warn(f'`{key}` only exists in pretrained checkpoints but not in model parameters')

    def save_trans_ckpt(self):
        """对state_dict进行转换
        1. load_trans_ckpt的逆操作
        2. 方便后续各个模型继承并做一些预处理, 如合并qkv权重
        """
        return self.state_dict()

    def save_pretrained(self, save_path: 'str', weight_map: 'dict'=None, mapping: 'Union[dict, Callable]'=None, write_to_disk: 'bool'=True, ignore_tied_parameters=False):
        """按照预训练模型的key来保存模型, 可供transformers包加载
           1. 按照variable_mapping()逆向来保存权重
           2. 各个模型存在save_trans_ckpt()的也要执行, 如部分大模型需要把q,k,v合并为qkv

           :param save_path: str, 保存的文件/文件夹路径
           :param weight_map: dict, 部分大模型会有pytorch_model.bin.index.json文件, 对应其中的weight_map字段
                              可`from bert4torch.snippets import JsonConfig
                                 weight_map = JsonConfig(config_path).weight_map`来加载
           :param mapping: dict/func, 一般来说为None, 也允许用户自行指定映射关系（一般不需要）
           :param write_to_disk: bool, 是否写入硬盘，一般都是True, 该参数主要是为了在Trainer().save_pretrained
           :param ignore_tied_parameters: bool, 保存时候忽视tied_parameters
        """
        state_dict = self.save_trans_ckpt()
        if ignore_tied_parameters:
            named_tied_parameters = find_tied_parameters(self)
            tied_parameters = [tied_parameter for _, tied_parameters in named_tied_parameters.items() for tied_parameter in tied_parameters]
            log_info(f'Remove tied parameters: {tied_parameters}')
            for tied_parameter in tied_parameters:
                if tied_parameter in state_dict:
                    state_dict.pop(tied_parameter)
        mapping = mapping or self.variable_mapping()
        for k in list(state_dict.keys()):
            if isinstance(mapping, dict):
                state_dict[mapping.get(k, k)] = state_dict.pop(k)
            elif isinstance(mapping, Callable):
                state_dict[mapping(k)] = state_dict.pop(k)
        save_dir = None if re.search('\\.[a-zA-z0-9]+$', save_path) else save_path
        if write_to_disk and hasattr(self, 'checkpoint_path') and self.checkpoint_path is not None and save_dir:
            if isinstance(self.checkpoint_path, str):
                checkpoint_dir = os.path.dirname(self.checkpoint_path) if os.path.isfile(self.checkpoint_path) else self.checkpoint_path
            elif isinstance(self.checkpoint_path, (tuple, list)):
                checkpoint_dir = os.path.dirname(self.checkpoint_path[0]) if os.path.isfile(self.checkpoint_path[0]) else self.checkpoint_path[0]
            else:
                raise TypeError(f'`self.checkpoint_path` only support str,tuple,list')
            copytree(checkpoint_dir, save_dir, ignore_copy_files=['\\.bin$', '\\.safetensors$'], dirs_exist_ok=True)
            bin_index_json = [os.path.join(checkpoint_dir, i) for i in os.listdir(checkpoint_dir) if i.endswith('.index.json')]
            bin_index_json = bin_index_json[0] if bin_index_json else ''
            if save_dir is not None and os.path.exists(bin_index_json):
                weight_map = weight_map or JsonConfig(bin_index_json).get('weight_map')
        if weight_map is None:
            if write_to_disk:
                save_checkpoint(state_dict, os.path.join(save_dir, 'pytorch_model.bin') if save_dir else save_path)
            else:
                return state_dict
        else:
            ckpt2param = dict()
            for param_name, save_file in weight_map.items():
                if save_file not in ckpt2param:
                    ckpt2param[save_file] = set([param_name])
                else:
                    ckpt2param[save_file].add(param_name)
            for save_file, param_names in ckpt2param.items():
                single_ckpt = {}
                for k in list(state_dict.keys()):
                    if k in param_names:
                        single_ckpt[k] = state_dict.pop(k)
                save_checkpoint(single_ckpt, os.path.join(save_dir or save_path, save_file))

    def apply_embeddings(self, *inputs, **model_kwargs):
        raise NotImplementedError

    def apply_main_layers(self, *inputs, **model_kwargs):
        raise NotImplementedError

    def apply_final_layers(self, *inputs, **model_kwargs):
        raise NotImplementedError

    def apply_on_layer_begin(self, l_i, **model_kwargs):
        """新增对layer block输入进行操作的函数"""
        if model_kwargs.get('past_key_values') is not None:
            model_kwargs['past_key_value'] = model_kwargs['past_key_values'][l_i]
        if 'encoder_hidden_states' in model_kwargs and model_kwargs.get('cross_past_key_values') is not None:
            model_kwargs['cross_past_key_value'] = model_kwargs['cross_past_key_values'][l_i]
        return model_kwargs

    def apply_on_layer_end(self, l_i, **model_kwargs):
        """新增对layer block输出进行操作的函数, 目前仅在MixUp中使用"""
        if model_kwargs.get('use_states') is not True:
            return model_kwargs
        if model_kwargs.get('past_key_value') is not None:
            if 'past_key_values' not in model_kwargs or model_kwargs.get('past_key_values') is None:
                model_kwargs['past_key_values'] = [None] * self.num_hidden_layers
            model_kwargs['past_key_values'][l_i] = model_kwargs['past_key_value']
        if model_kwargs.get('cross_past_key_value') is not None:
            if 'cross_past_key_values' not in model_kwargs or model_kwargs.get('cross_past_key_values') is None:
                model_kwargs['cross_past_key_values'] = [None] * self.num_hidden_layers
            model_kwargs['cross_past_key_values'][l_i] = model_kwargs['cross_past_key_value']
        return model_kwargs

    def compute_attention_bias(self, inputs=None):
        """定义每一层的Attention Bias"""
        return self.attention_bias

    def compute_position_bias(self, inputs=None):
        """定义每一层的Position Bias（一般相对位置编码用）"""
        return self.position_bias

    def set_outputs(self, outputs):
        """设置output和oututs属性"""
        if not isinstance(outputs, list):
            outputs = [outputs]
        outputs = outputs[:]
        self.outputs = outputs
        if len(outputs) > 1:
            self.output = outputs
        else:
            self.output = outputs[0]

    def quantize(self, quantization_method: "Literal['cpm_kernels', 'load_in_8bit', 'load_in_4bit']", **kwargs):
        """量化
        
        Examples:
        ```python
        >>> # 1. bitsandbytes的load_in_8bit量化
        >>> model = model.quantize(quantization_method='load_in_8bit', llm_int8_skip_modules=['model.embeddings.word_embeddings', 'lm_head'])
        
        >>> # 2. bitsandbytes的load_in_4bit量化
        >>> from transformers import BitsAndBytesConfig
        >>> q_config = BitsAndBytesConfig(load_in_4bit=True,
        ...                             bnb_4bit_quant_type='nf4',
        ...                             bnb_4bit_use_double_quant=True,
        ...                             bnb_4bit_compute_dtype=torch.float16,  # 可选 torch.float32, torch.float16, torch.bfloat16
        ...                             llm_int8_skip_modules=['model.embeddings.word_embeddings', 'lm_head']
        ...                             )
        >>> model = model.quantize(quantization_method='load_in_4bit', quantization_config=q_config)
        
        >>> # 3. cpm_kernels量化
        >>> model = model.quantize(quantization_method='cpm_kernels', quantization_bit=8)
        ```
        """
        if self.quantized:
            None
            return self
        new_kwargs = copy.deepcopy(kwargs)
        if 'model' in new_kwargs:
            new_kwargs.pop('model')
        if quantization_method == 'cpm_kernels':
            self.half()
            self = quantize_cpm_kernels(self, **new_kwargs)
        elif quantization_method in {'load_in_8bit', 'load_in_4bit'}:
            load_in_8bit = True if quantization_method == 'load_in_8bit' else False
            load_in_4bit = True if quantization_method == 'load_in_4bit' else False
            self = quantize_load_in_kbit(self, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, **new_kwargs)
        else:
            raise ValueError('Please check args `quantization_method`')
        self.quantized = True
        torch.cuda.empty_cache()
        return self

    def add_adapter(self, adapter_method='bottleneck', bottlenect_size=64):
        """增加adapter层"""
        self = add_adapter(self, adapter_method, bottlenect_size)
        self.print_trainable_parameters()
        return self

    def get_peft_model(self, peft_config, adapter_name='default'):
        """hf的peft库：https://github.com/huggingface/peft
        peft的接口LoraModel接口有变，这里使用v0.0.3
        """
        self.peft_config = {adapter_name: peft_config}
        if isinstance(peft_config, peft.LoraConfig):
            model = peft.LoraModel(self, self.peft_config, adapter_name)
        elif isinstance(peft_config, peft.AdaLoraConfig):
            model = peft.AdaLoraModel(self, self.peft_config, adapter_name)
        else:
            raise ValueError(f'{type(peft_config)} has not been supported')
        self = add_trainer(model) if self.add_trainer else model
        self.print_trainable_parameters()
        return self

    def print_trainable_parameters(self):
        """打印可训练的参数量"""
        print_trainable_parameters(self)

    @property
    def device(self) ->torch.device:
        """获取model所在的device"""
        return get_parameter_device(self)


TRANSFORMER_BLOCKS = {'BertLayer': BertLayer, 'MiniCPMLayer': MiniCPMLayer, 'FalconParallelAttnLayer': FalconParallelAttnLayer, 'GlmLayer': GlmLayer, 'Glm2Layer': Glm2Layer, 'T5Layer': T5Layer, 'GAU_Layer': GAULayer, 'Gpt2MlLayer': Gpt2MlLayer, 'XlnetLayer': XlnetLayer}


def create_position_ids_start_at_padding(input_ids, padding_idx, past_key_values_length=0, start_padding_idx=True):
    """生成padding_ids, 从padding_idx+1开始。忽略填充符号"""
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + (padding_idx if start_padding_idx else 0)


def modify_variable_mapping(original_func, **new_dict):
    """对variable_mapping的返回值（字典）进行修改
    """

    def wrapper(*args, **kwargs):
        result = original_func(*args, **kwargs)
        result.update(new_dict)
        return result
    return wrapper


def old_checkpoint(function, model_kwargs):
    """ 兼容torch<1.11.0时仅允许输入输出是位置参数
    通过闭包来对返回参数进行控制
    """

    def create_custom_forward(module):

        def custom_forward(*inputs):
            outputs = module(*inputs)
            if isinstance(outputs, dict):
                setattr(create_custom_forward, 'outputs_keys', [v for v in outputs.keys()])
                return tuple(outputs.values())
            else:
                return outputs
        return custom_forward
    args = []
    __args = inspect.getargspec(type(function).forward)
    arg_names, arg_defaults = __args[0][1:], __args[-1]
    for i, arg_name in enumerate(arg_names):
        args.append(model_kwargs.get(arg_name, arg_defaults[i]))
    preserve = model_kwargs.pop('preserve_rng_state', True)
    outputs = CheckpointFunction.apply(create_custom_forward(function), preserve, *args)
    if hasattr(create_custom_forward, 'outputs_keys'):
        return dict(zip(create_custom_forward.outputs_keys, outputs))
    else:
        return outputs


class BERT(BERT_BASE):
    """构建BERT模型
    """
    _no_split_modules = ['BertLayer']

    def __init__(self, max_position: 'int', segment_vocab_size: 'int'=2, with_pool: 'bool'=False, with_nsp: 'bool'=False, with_mlm: 'bool'=False, custom_position_ids: "Literal[True, False, 'start_at_padding']"=False, custom_attention_mask: 'bool'=False, shared_segment_embeddings: 'bool'=False, conditional_size: 'Union[bool, int]'=None, additional_embs: 'Union[bool, torch.Tensor, List[torch.Tensor]]'=False, is_dropout: 'bool'=False, pad_token_id: 'int'=0, layer_type: 'str'='BertLayer', **kwargs):
        super(BERT, self).__init__(**kwargs)
        self.max_position = max_position
        self.segment_vocab_size = segment_vocab_size
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.custom_position_ids = custom_position_ids
        self.custom_attention_mask = custom_attention_mask
        self.shared_segment_embeddings = shared_segment_embeddings
        self.is_dropout = is_dropout
        self.pad_token_id = pad_token_id
        if self.with_nsp and not self.with_pool:
            self.with_pool = True
        self.additional_embs = additional_embs
        self.conditional_size = conditional_size
        self.embeddings = BertEmbeddings(**self.get_kw(*self._embedding_args, **kwargs))
        self.encoderLayer = nn.ModuleList([(TRANSFORMER_BLOCKS[layer_type](layer_idx=layer_idx, **self.get_kw(*self._layer_args, **kwargs)) if layer_idx in self.keep_hidden_layers else BlockIdentity()) for layer_idx in range(self.num_hidden_layers)])
        if self.with_pool:
            self.pooler = nn.Linear(self.hidden_size, self.hidden_size)
            self.pooler_activation = nn.Tanh() if self.with_pool is True else get_activation(self.with_pool)
            if self.with_nsp:
                self.nsp = nn.Linear(self.hidden_size, 2)
        else:
            self.pooler = None
            self.pooler_activation = None
        if self.with_mlm:
            self.mlmDense = nn.Linear(self.hidden_size, self.embedding_size)
            self.transform_act_fn = get_activation(self.hidden_act)
            self.mlmLayerNorm = LayerNorm(self.embedding_size, eps=1e-12, conditional_size=self.conditional_size)
            self.mlmDecoder = nn.Linear(self.embedding_size, self.vocab_size, bias=False)
            self.mlmBias = nn.Parameter(torch.zeros(self.vocab_size))
            self.mlmDecoder.bias = self.mlmBias
            self.tie_weights()
        self.model_type = 'bert'

    @property
    def _embedding_args(self):
        args = ['vocab_size', 'embedding_size', 'hidden_size', 'max_position', 'segment_vocab_size', 'shared_segment_embeddings', 'dropout_rate', 'conditional_size']
        return args

    @property
    def _layer_args(self):
        args = ['hidden_size', 'num_attention_heads', 'dropout_rate', 'attention_probs_dropout_prob', 'intermediate_size', 'hidden_act', 'is_dropout', 'conditional_size', 'max_position']
        return args

    def tie_weights(self):
        """权重的tie"""
        if self.tie_word_embeddings is True and self.with_mlm:
            self.mlmDecoder.weight = self.embeddings.word_embeddings.weight
            self.mlmDecoder.bias = self.mlmBias

    def get_input_embeddings(self):
        """获取word_embeddings"""
        return self.embeddings.word_embeddings

    def layer_forward(self, layer, model_kwargs, use_reentrant=False):
        """transformer block的forward"""
        if self.gradient_checkpoint and self.training:
            if use_reentrant is True or version.parse(torch.__version__) < version.parse('1.11.0'):
                return old_checkpoint(layer, model_kwargs)
            else:
                return checkpoint(layer, use_reentrant=use_reentrant, **model_kwargs)
        else:
            return layer(**model_kwargs)

    def preprare_embeddings_inputs(self, *inputs: Union[tuple, list], **model_kwargs):
        """解析准备进embedding层的的输入"""
        index_ = 0
        if model_kwargs.get('input_ids') is not None:
            token_ids = model_kwargs['input_ids']
        elif model_kwargs.get('token_ids') is not None:
            token_ids = model_kwargs['token_ids']
        else:
            token_ids = inputs[0]
            index_ += 1
        if model_kwargs.get('segment_ids') is not None:
            segment_ids = model_kwargs['segment_ids']
        elif model_kwargs.get('token_type_ids') is not None:
            segment_ids = model_kwargs['token_type_ids']
        elif self.segment_vocab_size > 0:
            segment_ids = inputs[index_]
            index_ += 1
        else:
            segment_ids = None
        past_key_values_length = 0
        if model_kwargs.get('past_key_values_length') is not None:
            past_key_values_length = model_kwargs['past_key_values_length']
        elif model_kwargs.get('past_key_values') is not None:
            past_key_values_length = model_kwargs.get('past_key_values')[0][0].shape[2]
        if model_kwargs.get('position_ids') is not None:
            position_ids = model_kwargs['position_ids']
        elif self.custom_position_ids is True:
            position_ids = inputs[index_]
            index_ += 1
        elif self.custom_position_ids == 'start_at_padding':
            position_ids = create_position_ids_start_at_padding(token_ids, self.pad_token_id, past_key_values_length)
        else:
            position_ids = torch.arange(token_ids.shape[1], dtype=torch.long, device=token_ids.device).unsqueeze(0) + past_key_values_length
        model_kwargs['position_ids'] = position_ids
        if model_kwargs.get('attention_mask') is not None:
            attention_mask = model_kwargs['attention_mask']
        elif self.custom_attention_mask:
            attention_mask = inputs[index_].long()
            index_ += 1
        elif not token_ids.requires_grad and token_ids.dtype in {torch.long, torch.int}:
            attention_mask = (token_ids != self.pad_token_id).long()
            if self.pad_token_id < 0:
                token_ids = token_ids * attention_mask
        else:
            attention_mask = self.attention_mask_cache
        self.attention_mask_cache = attention_mask
        model_kwargs['pad_attention_mask'] = attention_mask
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        self.compute_attention_bias([token_ids, segment_ids])
        if self.attention_bias is not None:
            attention_mask = attention_mask * self.attention_bias
        try:
            attention_mask = attention_mask
        except StopIteration:
            attention_mask = attention_mask
        if model_kwargs.get('past_key_values') is not None and attention_mask.shape[-1] < model_kwargs.get('past_key_values')[0][0].shape[2] + token_ids.shape[1]:
            pad_length = model_kwargs.get('past_key_values')[0][0].shape[2] + token_ids.shape[1] - attention_mask.shape[-1]
            pre_attention_mask = torch.ones(attention_mask.shape[:3] + torch.Size([pad_length]))
            attention_mask = torch.cat([pre_attention_mask, attention_mask], dim=-1)
        if model_kwargs.get('conditional_emb') is not None:
            conditional_emb = model_kwargs['conditional_emb']
        elif self.conditional_size is not None:
            conditional_emb = inputs[index_]
            index_ += 1
        else:
            conditional_emb = None
        if model_kwargs.get('additional_embs') is not None:
            additional_embs = model_kwargs['additional_embs']
        elif self.additional_embs is True:
            additional_embs = inputs[index_]
            index_ += 1
        else:
            additional_embs = None
        additional_embs = [additional_embs] if isinstance(additional_embs, torch.Tensor) else additional_embs
        if len(inputs[index_:]) >= 2:
            model_kwargs['encoder_hidden_states'], model_kwargs['encoder_attention_mask'] = inputs[index_], inputs[index_ + 1]
        return token_ids, segment_ids, position_ids, conditional_emb, additional_embs, attention_mask, model_kwargs

    def apply_embeddings(self, *inputs: Union[tuple, list], **model_kwargs):
        """BERT的embedding，可接受"位置参数/关键字参数"形式

        :param inputs: List[torch.Tensor], 默认顺序是[token_ids, segment_ids(若有), position_ids(若有), custom_attention_mask(若有), conditional_input(若有), additional_input(若有)]
        :param model_kwargs: Dict[torch.Tensor], 字典输入项，和inputs是二选一的
        :return: Dict[torch.Tensor], [hidden_states, attention_mask, conditional_emb, ...]
        """
        token_ids, segment_ids, position_ids, conditional_emb, additional_embs, attention_mask, model_kwargs = self.preprare_embeddings_inputs(*inputs, **model_kwargs)
        hidden_states = self.embeddings(token_ids, segment_ids, position_ids, conditional_emb, additional_embs)
        model_kwargs.update({'hidden_states': hidden_states, 'attention_mask': attention_mask, 'conditional_emb': conditional_emb})
        return model_kwargs

    def apply_main_layers(self, **model_kwargs):
        """BERT的主体是基于Self-Attention的模块；
        顺序:Att --> Add --> LN --> FFN --> Add --> LN
        
        :param model_kwargs: Dict[torch.Tensor], 包含hidden_states, attention_mask, conditional_emb等
        :return: Dict[torch.Tensor], [encoded_layers, conditional_emb]
        """
        encoded_layers = [model_kwargs['hidden_states']]
        for l_i, layer_module in enumerate(self.encoderLayer):
            model_kwargs = self.apply_on_layer_begin(l_i, **model_kwargs)
            outputs = self.layer_forward(layer_module, model_kwargs)
            model_kwargs.update(outputs)
            hidden_states = model_kwargs['hidden_states']
            model_kwargs = self.apply_on_layer_end(l_i, **model_kwargs)
            if self.output_all_encoded_layers:
                encoded_layers.append(hidden_states)
        if not self.output_all_encoded_layers:
            encoded_layers.append(hidden_states)
        model_kwargs['encoded_layers'] = encoded_layers
        return model_kwargs

    def apply_final_layers(self, **model_kwargs):
        """根据剩余参数决定输出

        :param model_kwargs: Dict[torch.Tensor], 包含encoded_layers, conditional_emb等
        :return: List[torch.Tensor]|torch.Tensor|Dict[torch.Tensor], 模型输出，包含last_hidden_state/all_encoded_layers, pooled_output(若有), mlm_scores(若有), nsp_scores(若有)
        """
        encoded_layers, conditional_emb = model_kwargs['encoded_layers'], model_kwargs.get('conditional_emb', None)
        last_hidden_state = encoded_layers[-1]
        if self.with_pool:
            pooled_output = self.pooler_activation(self.pooler(last_hidden_state[:, 0]))
        else:
            pooled_output = None
        if self.with_pool and self.with_nsp:
            nsp_scores = self.nsp(pooled_output)
        else:
            nsp_scores = None
        if self.with_mlm:
            mlm_hidden_state = self.mlmDense(last_hidden_state)
            mlm_hidden_state = self.transform_act_fn(mlm_hidden_state)
            mlm_hidden_state = self.mlmLayerNorm(mlm_hidden_state, conditional_emb)
            mlm_scores = self.mlmDecoder(mlm_hidden_state)
            mlm_activation = get_activation('linear' if self.with_mlm is True else self.with_mlm)
            mlm_scores = mlm_activation(mlm_scores)
        else:
            mlm_scores = None
        if not self.output_all_encoded_layers:
            return self.gen_outputs(locals(), last_hidden_state, pooled_output, mlm_scores, nsp_scores)
        else:
            return self.gen_outputs(locals(), encoded_layers, pooled_output, mlm_scores, nsp_scores)

    def gen_outputs(self, locals_dict, *args):
        """ 生成outputs list/dict两种形式"""
        if not self.return_dict:
            outputs = [value for value in args if value is not None]
            return outputs if len(outputs) > 1 else outputs[0]
        else:
            outputs = DottableDict()
            for arg in args:
                if arg is None:
                    continue
                for name, value in locals_dict.items():
                    if value is arg:
                        outputs[name] = arg
                        break
            return outputs

    def load_trans_ckpt(self, checkpoint):
        """加载ckpt, 方便后续继承并做一些预处理
        这么写的原因是下游很多模型从BERT继承，这样下游可以默认使用BERT_BASE的load_trans_ckpt
        """
        state_dict = super().load_trans_ckpt(checkpoint)
        if hasattr(self, 'model_type') and self.model_type == 'bert':
            mapping_reverse = {v: k for k, v in self.variable_mapping().items()}
            mapping = {}
            for key in state_dict.keys():
                if '.gamma' in key:
                    value = key.replace('.gamma', '.weight')
                    mapping[mapping_reverse[value]] = key
                if '.beta' in key:
                    value = key.replace('.beta', '.bias')
                    mapping[mapping_reverse[value]] = key
            if 'cls.predictions.bias' in state_dict and 'cls.predictions.decoder.bias' not in state_dict:
                mapping['mlmDecoder.bias'] = 'cls.predictions.bias'
            self.variable_mapping = modify_variable_mapping(self.variable_mapping, **mapping)
        return state_dict

    def load_variable(self, variable, old_key, new_key, prefix='bert'):
        """加载单个变量的函数, 这里的名称均为映射前的"""
        if old_key in {f'{prefix}.embeddings.word_embeddings.weight', 'cls.predictions.bias', 'cls.predictions.decoder.weight', 'cls.predictions.decoder.bias'}:
            return self.load_embeddings(variable)
        elif new_key in {'embeddings.word_embeddings.weight', 'mlmBias', 'mlmDecoder.weight', 'mlmDecoder.bias'}:
            return self.load_embeddings(variable)
        else:
            return variable

    def variable_mapping(self, prefix='bert'):
        """权重映射字典，格式为{new_key: old_key}"""
        mapping = {'embeddings.word_embeddings.weight': f'{prefix}.embeddings.word_embeddings.weight', 'embeddings.position_embeddings.weight': f'{prefix}.embeddings.position_embeddings.weight', 'embeddings.segment_embeddings.weight': f'{prefix}.embeddings.token_type_embeddings.weight', 'embeddings.layerNorm.weight': f'{prefix}.embeddings.LayerNorm.weight', 'embeddings.layerNorm.bias': f'{prefix}.embeddings.LayerNorm.bias', 'pooler.weight': f'{prefix}.pooler.dense.weight', 'pooler.bias': f'{prefix}.pooler.dense.bias', 'nsp.weight': 'cls.seq_relationship.weight', 'nsp.bias': 'cls.seq_relationship.bias', 'mlmDense.weight': 'cls.predictions.transform.dense.weight', 'mlmDense.bias': 'cls.predictions.transform.dense.bias', 'mlmLayerNorm.weight': 'cls.predictions.transform.LayerNorm.weight', 'mlmLayerNorm.bias': 'cls.predictions.transform.LayerNorm.bias', 'mlmBias': 'cls.predictions.bias', 'mlmDecoder.weight': 'cls.predictions.decoder.weight', 'mlmDecoder.bias': 'cls.predictions.decoder.bias'}
        for i in range(self.num_hidden_layers):
            prefix_i = f'{prefix}.encoder.layer.%d.' % i
            mapping.update({f'encoderLayer.{i}.multiHeadAttention.q.weight': prefix_i + 'attention.self.query.weight', f'encoderLayer.{i}.multiHeadAttention.q.bias': prefix_i + 'attention.self.query.bias', f'encoderLayer.{i}.multiHeadAttention.k.weight': prefix_i + 'attention.self.key.weight', f'encoderLayer.{i}.multiHeadAttention.k.bias': prefix_i + 'attention.self.key.bias', f'encoderLayer.{i}.multiHeadAttention.v.weight': prefix_i + 'attention.self.value.weight', f'encoderLayer.{i}.multiHeadAttention.v.bias': prefix_i + 'attention.self.value.bias', f'encoderLayer.{i}.multiHeadAttention.o.weight': prefix_i + 'attention.output.dense.weight', f'encoderLayer.{i}.multiHeadAttention.o.bias': prefix_i + 'attention.output.dense.bias', f'encoderLayer.{i}.attnLayerNorm.weight': prefix_i + 'attention.output.LayerNorm.weight', f'encoderLayer.{i}.attnLayerNorm.bias': prefix_i + 'attention.output.LayerNorm.bias', f'encoderLayer.{i}.feedForward.intermediateDense.weight': prefix_i + 'intermediate.dense.weight', f'encoderLayer.{i}.feedForward.intermediateDense.bias': prefix_i + 'intermediate.dense.bias', f'encoderLayer.{i}.feedForward.outputDense.weight': prefix_i + 'output.dense.weight', f'encoderLayer.{i}.feedForward.outputDense.bias': prefix_i + 'output.dense.bias', f'encoderLayer.{i}.ffnLayerNorm.weight': prefix_i + 'output.LayerNorm.weight', f'encoderLayer.{i}.ffnLayerNorm.bias': prefix_i + 'output.LayerNorm.bias'})
        if self.embedding_size != self.hidden_size:
            mapping.update({'embeddings.embedding_hidden_mapping_in.weight': f'bert.encoder.embedding_hidden_mapping_in.weight', 'embeddings.embedding_hidden_mapping_in.bias': f'bert.encoder.embedding_hidden_mapping_in.bias'})
        return mapping


def delete_arguments(*arguments):
    """装饰器，为类方法删除参数（主要用于类的__init__方法）"""

    def actual_decorator(func):

        def new_func(self, *args, **kwargs):
            for k in arguments:
                if k in kwargs:
                    raise TypeError("%s got an unexpected keyword argument '%s'" % (self.__class__.__name__, k))
            return func(self, *args, **kwargs)
        return new_func
    return actual_decorator


def insert_arguments(**arguments):
    """装饰器，为类方法增加参数（主要用于类的__init__方法）"""

    def actual_decorator(func):

        def new_func(self, *args, **kwargs):
            for k, v in arguments.items():
                if k in kwargs:
                    v = kwargs.pop(k)
                setattr(self, k, v)
            return func(self, *args, **kwargs)
        return new_func
    return actual_decorator


class ELECTRA(BERT):
    """Google推出的ELECTRA模型；
    链接：https://arxiv.org/abs/2003.10555
    """

    @insert_arguments(with_discriminator=False)
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, max_position, **kwargs):
        super(ELECTRA, self).__init__(max_position, **kwargs)
        self.model_type = 'electra'
        if self.with_discriminator:
            self.dense = nn.Linear(self.hidden_size, self.hidden_size)
            self.dense_act = get_activation(self.hidden_act)
            self.dense_prediction = nn.Linear(self.hidden_size, 1)
            self.dense_prediction_act = get_activation('sigmoid') if self.with_discriminator is True else get_activation(self.with_discriminator)

    def apply_final_layers(self, **model_kwargs):
        outputs = super().apply_final_layers(**model_kwargs)
        last_hidden_state = outputs['last_hidden_state'] if self.return_dict else outputs
        if self.with_discriminator:
            logits = self.dense_act(self.dense(last_hidden_state))
            logits = self.dense_prediction_act(self.dense_prediction(logits))
            if self.return_dict:
                outputs['logits'] = logits
            else:
                outputs.append(logits)
        return outputs

    def variable_mapping(self):
        mapping = super(ELECTRA, self).variable_mapping(prefix='electra')
        mapping.update({'dense.weight': 'discriminator_predictions.dense.weight', 'dense.bias': 'discriminator_predictions.dense.bias', 'dense_prediction.weight': 'discriminator_predictions.dense_prediction.weight', 'dense_prediction.bias': 'discriminator_predictions.dense_prediction.bias'})
        for del_key in ['pooler.weight', 'pooler.bias', 'nsp.weight', 'nsp.bias', 'mlmDense.weight', 'mlmDense.bias', 'mlmLayerNorm.weight', 'mlmLayerNorm.bias', 'mlmBias', 'mlmDecoder.weight', 'mlmDecoder.bias']:
            del mapping[del_key]
        return mapping


class ERNIE(BERT):
    """百度文心 https://github.com/PaddlePaddle/ERNIE"""

    def __init__(self, *args, **kwargs):
        super(ERNIE, self).__init__(*args, **kwargs)
        self.use_task_id = kwargs.get('use_task_id')
        self.embeddings = self.ErnieEmbeddings(**self.get_kw(*self._embedding_args, **kwargs))
        self.model_type = 'ernie'

    def variable_mapping(self):
        mapping = super(ERNIE, self).variable_mapping(prefix='ernie')
        mapping.update({'mlmDecoder.weight': 'ernie.embeddings.word_embeddings.weight', 'mlmDecoder.bias': 'cls.predictions.bias'})
        for k, v in mapping.items():
            if 'LayerNorm.weight' in v or 'LayerNorm.bias' in v:
                v1 = v.replace('.weight', '.gamma').replace('.bias', '.beta')
                mapping[k] = v1
        for del_key in ['nsp.weight', 'nsp.bias']:
            del mapping[del_key]
        if self.use_task_id:
            mapping['embeddings.task_type_embeddings.weight'] = 'ernie.embeddings.task_type_embeddings.weight'
        return mapping


    class ErnieEmbeddings(BertEmbeddings):

        def __init__(self, vocab_size, embedding_size, *args, **kwargs):
            super().__init__(vocab_size, embedding_size, *args, **kwargs)
            self.use_task_id = kwargs.get('use_task_id')
            if self.use_task_id:
                self.task_type_embeddings = nn.Embedding(kwargs.get('task_type_vocab_size'), embedding_size)

        def apply_embeddings(self, token_ids, segment_ids, position_ids, additional_embs, **kwargs):
            embeddings = super().apply_embeddings(token_ids, segment_ids, position_ids, additional_embs, **kwargs)
            task_type_ids = kwargs.get('task_type_ids')
            if self.use_task_id:
                if task_type_ids is None:
                    task_type_ids = torch.zeros(token_ids.shape, dtype=torch.long, device=embeddings.device)
                task_type_embeddings = self.task_type_embeddings(task_type_ids)
                embeddings += task_type_embeddings
            return embeddings


class SiglipVisionEmbeddings(nn.Module):

    def __init__(self, config: 'SiglipVisionConfig'):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Conv2d(in_channels=config.num_channels, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, padding='valid')
        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def forward(self, pixel_values: 'torch.FloatTensor', patch_attention_mask: 'torch.BoolTensor', tgt_sizes: 'Optional[torch.IntTensor]'=None) ->torch.Tensor:
        batch_size = pixel_values.size(0)
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
        max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
        boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
        position_ids = torch.full(size=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0)
        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            if tgt_sizes is not None:
                nb_patches_h = tgt_sizes[batch_idx][0]
                nb_patches_w = tgt_sizes[batch_idx][1]
            else:
                nb_patches_h = p_attn_mask[:, 0].sum()
                nb_patches_w = p_attn_mask[0].sum()
            fractional_coords_h = torch.arange(0, 1 - 1e-06, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-06, 1 / nb_patches_w)
            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids
        position_ids = position_ids
        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


class SiglipAttention(nn.Module):
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

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]'=None, output_attentions: 'Optional[bool]'=False) ->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        batch_size, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_v_seq_len = key_states.shape[-2]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(f'Attention weights should be of size {batch_size, self.num_heads, q_len, k_v_seq_len}, but is {attn_weights.size()}')
        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(f'Attention mask should be of size {batch_size, 1, q_len, k_v_seq_len}, but is {attention_mask.size()}')
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {batch_size, self.num_heads, q_len, self.head_dim}, but is {attn_output.size()}')
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


log_config = {'DEBUG': {'level': 10, 'color': 'purple'}, 'INFO': {'level': 20, 'color': 'green'}, 'TRAIN': {'level': 21, 'color': 'cyan'}, 'EVAL': {'level': 22, 'color': 'blue'}, 'WARNING': {'level': 30, 'color': 'yellow'}, 'ERROR': {'level': 40, 'color': 'red'}, 'CRITICAL': {'level': 50, 'color': 'bold_red'}}


class SiglipFlashAttention2(SiglipAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'Optional[torch.LongTensor]'=None, position_ids: 'Optional[torch.LongTensor]'=None, past_key_value: 'Optional[Tuple[torch.Tensor]]'=None, output_attentions: 'bool'=False, use_cache: 'bool'=False, **kwargs) ->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        dropout_rate = self.dropout if self.training else 0.0
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
        attn_output = self._flash_attention_forward(query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate)
        attn_output = attn_output.reshape(bsz, q_len, self.embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights

    def _flash_attention_forward(self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None):
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
        """
        causal = self.is_causal and query_length != 1
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(query_states, key_states, value_states, attention_mask, query_length)
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
            attn_output_unpad = flash_attn_varlen_func(query_states, key_states, value_states, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_in_batch_q, max_seqlen_k=max_seqlen_in_batch_k, dropout_p=dropout, softmax_scale=softmax_scale, causal=causal)
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal)
        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape
        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
        if query_length == kv_seq_len:
            query_layer = index_first_axis(query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k)
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


class SiglipMLP(nn.Module):

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


class SiglipEncoderLayer(nn.Module):

    def __init__(self, config: 'SiglipVisionConfig'):
        super().__init__()
        self.embed_dim = config.hidden_size
        self._use_flash_attention_2 = config._attn_implementation == 'flash_attention_2'
        self.self_attn = SiglipAttention(config) if not self._use_flash_attention_2 else SiglipFlashAttention2(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'torch.Tensor', output_attentions: 'Optional[bool]'=False) ->Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = hidden_states,
        if output_attentions:
            outputs += attn_weights,
        return outputs


def get_1d_sincos_pos_embed_from_grid_new(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (H, W)
    out: (H, W, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    out = np.einsum('hw,d->hwd', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=-1)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=-1)
    return emb


def get_2d_sincos_pos_embed(embed_dim, image_size):
    """
    image_size: image_size or (image_height, image_width)
    return:
    pos_embed: [image_height, image_width, embed_dim]
    """
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size, grid_w_size = image_size[0], image_size[1]
    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
       given learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (batch_size, num_queries, embed_dim)
    """

    def __init__(self, num_queries, embed_dim, num_heads, kv_dim=None, norm_layer=partial(nn.LayerNorm, eps=1e-06), adaptive=False, max_size=(70, 70)):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.adaptive = adaptive
        self.max_size = max_size
        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()
        self.attn = MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)
        self.ln_post = norm_layer(embed_dim)
        self.proj = nn.Parameter(embed_dim ** -0.5 * torch.randn(embed_dim, embed_dim))
        self._set_2d_pos_cache(self.max_size)

    def _set_2d_pos_cache(self, max_size, device='cpu'):
        if is_deepspeed_zero3_enabled():
            device = 'cuda'
        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.embed_dim, max_size)).float()
        self.register_buffer('pos_embed', pos_embed, persistent=False)

    def _adjust_pos_cache(self, tgt_sizes, device):
        max_h = torch.max(tgt_sizes[:, 0])
        max_w = torch.max(tgt_sizes[:, 1])
        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = [max(max_h, self.max_size[0]), max(max_w, self.max_size[1])]
            self._set_2d_pos_cache(self.max_size, device)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, tgt_sizes=None):
        assert x.shape[0] == tgt_sizes.shape[0]
        bs = x.shape[0]
        device = x.device
        dtype = x.dtype
        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]
        self._adjust_pos_cache(tgt_sizes, device=device)
        max_patch_len = torch.max(patch_len)
        key_padding_mask = torch.zeros((bs, max_patch_len), dtype=torch.bool, device=device)
        pos_embed = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i]
            pos_embed.append(self.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)))
            key_padding_mask[i, patch_len[i]:] = True
        pos_embed = torch.nn.utils.rnn.pad_sequence(pos_embed, batch_first=True, padding_value=0.0).permute(1, 0, 2)
        x = self.kv_proj(x)
        x = self.ln_kv(x).permute(1, 0, 2)
        q = self.ln_q(self.query)
        out = self.attn(self._repeat(q, bs), x + pos_embed, x, key_padding_mask=key_padding_mask)[0]
        x = out.permute(1, 0, 2)
        x = self.ln_post(x)
        x = x @ self.proj
        return x

    def _repeat(self, query, N: 'int'):
        return query.unsqueeze(1).repeat(1, N, 1)


def _canonical_mask(mask: 'Optional[Tensor]', mask_name: 'str', other_type: 'Optional[DType]', other_name: 'str', target_type: 'DType', check_other: 'bool'=True) ->Optional[Tensor]:
    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = torch.is_floating_point(mask)
        if _mask_dtype != torch.bool and not _mask_is_float:
            raise AssertionError(f'only bool and floating types of {mask_name} are supported')
        if check_other and other_type is not None:
            if _mask_dtype != other_type:
                warnings.warn(f'Support for mismatched {mask_name} and {other_name} is deprecated. Use same type for both instead.')
        if not _mask_is_float:
            mask = torch.zeros_like(mask, dtype=target_type).masked_fill_(mask, float('-inf'))
    return mask


def _in_projection(q: 'Tensor', k: 'Tensor', v: 'Tensor', w_q: 'Tensor', w_k: 'Tensor', w_v: 'Tensor', b_q: 'Optional[Tensor]'=None, b_k: 'Optional[Tensor]'=None, b_v: 'Optional[Tensor]'=None) ->Tuple[Tensor, Tensor, Tensor]:
    """
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.
    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`
        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`
    """
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    assert w_q.shape == (Eq, Eq), f'expecting query weights shape of {Eq, Eq}, but got {w_q.shape}'
    assert w_k.shape == (Eq, Ek), f'expecting key weights shape of {Eq, Ek}, but got {w_k.shape}'
    assert w_v.shape == (Eq, Ev), f'expecting value weights shape of {Eq, Ev}, but got {w_v.shape}'
    assert b_q is None or b_q.shape == (Eq,), f'expecting query bias shape of {Eq,}, but got {b_q.shape}'
    assert b_k is None or b_k.shape == (Eq,), f'expecting key bias shape of {Eq,}, but got {b_k.shape}'
    assert b_v is None or b_v.shape == (Eq,), f'expecting value bias shape of {Eq,}, but got {b_v.shape}'
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _in_projection_packed(q: 'Tensor', k: 'Tensor', v: 'Tensor', w: 'Tensor', b: 'Optional[Tensor]'=None) ->List[Tensor]:
    """
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.
    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension
        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            proj = linear(q, w, b)
            proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
            return proj[0], proj[1], proj[2]
        else:
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            q_proj = linear(q, w_q, b_q)
            kv_proj = linear(k, w_kv, b_kv)
            kv_proj = kv_proj.unflatten(-1, (2, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
            return q_proj, kv_proj[0], kv_proj[1]
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _mha_shape_check(query: 'Tensor', key: 'Tensor', value: 'Tensor', key_padding_mask: 'Optional[Tensor]', attn_mask: 'Optional[Tensor]', num_heads: 'int'):
    if query.dim() == 3:
        is_batched = True
        assert key.dim() == 3 and value.dim() == 3, f'For batched (3-D) `query`, expected `key` and `value` to be 3-D but found {key.dim()}-D and {value.dim()}-D tensors respectively'
        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 2, f'For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D but found {key_padding_mask.dim()}-D tensor instead'
        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), f'For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D but found {attn_mask.dim()}-D tensor instead'
    elif query.dim() == 2:
        is_batched = False
        assert key.dim() == 2 and value.dim() == 2, f'For unbatched (2-D) `query`, expected `key` and `value` to be 2-D but found {key.dim()}-D and {value.dim()}-D tensors respectively'
        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 1, f'For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D but found {key_padding_mask.dim()}-D tensor instead'
        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), f'For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D but found {attn_mask.dim()}-D tensor instead'
            if attn_mask.dim() == 3:
                expected_shape = num_heads, query.shape[0], key.shape[0]
                assert attn_mask.shape == expected_shape, f'Expected `attn_mask` shape to be {expected_shape} but got {attn_mask.shape}'
    else:
        raise AssertionError(f'query should be unbatched 2D or batched 3D tensor but received {query.dim()}-D query tensor')
    return is_batched


def _none_or_dtype(input: 'Optional[Tensor]') ->Optional[DType]:
    if input is None:
        return None
    elif isinstance(input, torch.Tensor):
        return input.dtype
    raise RuntimeError('input to _none_or_dtype() must be None or torch.Tensor')


class MultiheadAttention(nn.MultiheadAttention):

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None):
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first, device, dtype)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)

    def forward(self, query: 'Tensor', key: 'Tensor', value: 'Tensor', key_padding_mask: 'Optional[Tensor]'=None, need_weights: 'bool'=True, attn_mask: 'Optional[Tensor]'=None, average_attn_weights: 'bool'=True, is_causal: 'bool'=False) ->Tuple[Tensor, Optional[Tensor]]:
        why_not_fast_path = ''
        if attn_mask is not None and torch.is_floating_point(attn_mask) or key_padding_mask is not None and torch.is_floating_point(key_padding_mask):
            why_not_fast_path = 'floating-point masks are not supported for fast path.'
        is_batched = query.dim() == 3
        key_padding_mask = _canonical_mask(mask=key_padding_mask, mask_name='key_padding_mask', other_type=F._none_or_dtype(attn_mask), other_name='attn_mask', target_type=query.dtype)
        attn_mask = _canonical_mask(mask=attn_mask, mask_name='attn_mask', other_type=None, other_name='', target_type=query.dtype, check_other=False)
        if not is_batched:
            why_not_fast_path = f'input not batched; expected query.dim() of 3 but got {query.dim()}'
        elif query is not key or key is not value:
            why_not_fast_path = 'non-self attention was used (query, key, and value are not the same Tensor)'
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is None:
            why_not_fast_path = 'in_proj_weight was None'
        elif query.dtype != self.in_proj_weight.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = 'training is enabled'
        elif self.num_heads % 2 != 0:
            why_not_fast_path = 'self.num_heads is not even'
        elif not self.batch_first:
            why_not_fast_path = 'batch_first was not True'
        elif self.bias_k is not None:
            why_not_fast_path = 'self.bias_k was not None'
        elif self.bias_v is not None:
            why_not_fast_path = 'self.bias_v was not None'
        elif self.add_zero_attn:
            why_not_fast_path = 'add_zero_attn was enabled'
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = '_qkv_same_embed_dim was not True'
        elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
            why_not_fast_path = 'supplying both src_key_padding_mask and src_mask at the same time                                  is not supported with NestedTensor input'
        elif torch.is_autocast_enabled():
            why_not_fast_path = 'autocast is enabled'
        if not why_not_fast_path:
            tensor_args = query, key, value, self.in_proj_weight, self.in_proj_bias, self.out_proj.weight, self.out_proj.bias
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = 'some Tensor argument has_torch_function'
            elif _is_make_fx_tracing():
                why_not_fast_path = 'we are running make_fx tracing'
            elif not all(_check_arg_device(x) for x in tensor_args):
                why_not_fast_path = f"some Tensor argument's device is neither one of cpu, cuda or {torch.utils.backend_registration._privateuse1_backend_name}"
            elif torch.is_grad_enabled() and any(_arg_requires_grad(x) for x in tensor_args):
                why_not_fast_path = 'grad is enabled and at least one of query or the input/output projection weights or biases requires_grad'
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)
                if self.in_proj_bias is not None and self.in_proj_weight is not None:
                    return torch._native_multi_head_attention(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj.weight, self.out_proj.bias, merged_mask, need_weights, average_attn_weights, mask_type)
        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, 'MultiheadAttention does not support NestedTensor outside of its fast path. ' + f'The fast path was not hit because {why_not_fast_path}'
        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))
        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = self.multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, use_separate_proj_weight=True, q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight, v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights, is_causal=is_causal)
        else:
            attn_output, attn_output_weights = self.multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, average_attn_weights=average_attn_weights, is_causal=is_causal)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def multi_head_attention_forward(self, query: 'Tensor', key: 'Tensor', value: 'Tensor', embed_dim_to_check: 'int', num_heads: 'int', in_proj_weight: 'Optional[Tensor]', in_proj_bias: 'Optional[Tensor]', bias_k: 'Optional[Tensor]', bias_v: 'Optional[Tensor]', add_zero_attn: 'bool', dropout_p: 'float', out_proj_weight: 'Tensor', out_proj_bias: 'Optional[Tensor]', training: 'bool'=True, key_padding_mask: 'Optional[Tensor]'=None, need_weights: 'bool'=True, attn_mask: 'Optional[Tensor]'=None, use_separate_proj_weight: 'bool'=False, q_proj_weight: 'Optional[Tensor]'=None, k_proj_weight: 'Optional[Tensor]'=None, v_proj_weight: 'Optional[Tensor]'=None, static_k: 'Optional[Tensor]'=None, static_v: 'Optional[Tensor]'=None, average_attn_weights: 'bool'=True, is_causal: 'bool'=False) ->Tuple[Tensor, Optional[Tensor]]:
        tens_ops = query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias
        is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)
        if not is_batched:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(0)
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        key_padding_mask = _canonical_mask(mask=key_padding_mask, mask_name='key_padding_mask', other_type=_none_or_dtype(attn_mask), other_name='attn_mask', target_type=query.dtype)
        if is_causal and attn_mask is None:
            raise RuntimeError('Need attn_mask if specifying the is_causal hint. You may use the Transformer module method `generate_square_subsequent_mask` to create this mask.')
        if is_causal and key_padding_mask is None and not need_weights:
            attn_mask = None
        else:
            attn_mask = _canonical_mask(mask=attn_mask, mask_name='attn_mask', other_type=None, other_name='', target_type=query.dtype, check_other=False)
            if key_padding_mask is not None:
                is_causal = False
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
            assert in_proj_weight is not None, 'use_separate_proj_weight is False but in_proj_weight is None'
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
        q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        if static_k is None:
            k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        else:
            assert static_k.size(0) == bsz * num_heads, f'expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}'
            assert static_k.size(2) == head_dim, f'expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}'
            k = static_k
        if static_v is None:
            v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
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
            else:
                attn_mask = attn_mask + key_padding_mask
        if not training:
            dropout_p = 0.0
        if need_weights:
            B, Nt, E = q.shape
            q_scaled = q / math.sqrt(E)
            assert not (is_causal and attn_mask is None), 'FIXME: is_causal not implemented for need_weights'
            if attn_mask is not None:
                attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
            else:
                attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
            attn_output_weights = softmax(attn_output_weights, dim=-1)
            if dropout_p > 0.0:
                attn_output_weights = dropout(attn_output_weights, p=dropout_p)
            attn_output = torch.bmm(attn_output_weights, v)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            attn_output = self.out_proj(attn_output)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)
            if not is_batched:
                attn_output = attn_output.squeeze(1)
                attn_output_weights = attn_output_weights.squeeze(0)
            return attn_output, attn_output_weights
        else:
            if attn_mask is not None:
                if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(0)
                else:
                    attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)
            q = q.view(bsz, num_heads, tgt_len, head_dim)
            k = k.view(bsz, num_heads, src_len, head_dim)
            v = v.view(bsz, num_heads, src_len, head_dim)
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
            attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
            attn_output = self.out_proj(attn_output)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
            if not is_batched:
                attn_output = attn_output.squeeze(1)
            return attn_output, None


class Encoder(BERT):

    def __init__(self, *args, **kwargs):
        kwargs['vocab_size'] = kwargs.get('src_vocab_size', kwargs['vocab_size'])
        super().__init__(*args, **kwargs)
        self.encoder_attention_mask = None
        self.model_type = 'encoder'

    def forward(self, *inputs, **model_kwargs):
        """因为encoder需要返回encoder_attention_mask，因此这里从新定义一下，多返回一个参数
        """
        outputs, model_kwargs = super().forward(*inputs, use_states=True, **model_kwargs)
        return ([outputs] if isinstance(outputs, torch.Tensor) else outputs) + [model_kwargs['attention_mask']]


class LogitsProcessorList(list):
    """
    This class can be used to create a list of [`LogitsProcessor`] or [`LogitsWarper`] to subsequently process a
    `scores` input tensor. This class inherits from list and adds a specific *__call__* method to apply each
    [`LogitsProcessor`] or [`LogitsWarper`] to the inputs.
    """

    def __call__(self, input_ids: 'torch.LongTensor', scores: 'torch.FloatTensor', **kwargs) ->torch.FloatTensor:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            kwargs (`Dict[str, Any]`, *optional*):
                Additional kwargs that are specific to a logits processor.

        Return:
            `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`:
                The processed prediction scores.

        """
        for processor in self:
            scores = processor(input_ids, scores, **kwargs)
        return scores


LOGITS_PROCESSOR_INPUTS_DOCSTRING = """
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""


def get_max_input_seqlen(input_seqlen: 'torch.Tensor'):
    return input_seqlen.max().item() if len(input_seqlen) > 1 else input_seqlen.item()


def sequence_padding(inputs: 'Union[List[np.ndarray], List[List], List[torch.Tensor]]', length: 'Union[List, int]'=None, value: 'int'=0, seq_dims: 'int'=1, mode: "Literal['pre', 'left', 'post', 'right']"='post'):
    """将序列padding到同一长度"""
    if isinstance(inputs[0], (np.ndarray, list)):
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]
        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]
        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode in {'post', 'right'}:
                    pad_width[i] = 0, length[i] - np.shape(x)[i]
                elif mode in {'pre', 'left'}:
                    pad_width[i] = length[i] - np.shape(x)[i], 0
                else:
                    raise ValueError('"mode" argument must be "post/right" or "pre/left".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)
        return np.array(outputs)
    elif isinstance(inputs[0], torch.Tensor):
        assert mode in {'post', 'right'}, '"mode" argument must be "post/right" when element is torch.Tensor'
        if length is not None:
            inputs = [i[:length] for i in inputs]
        return pad_sequence(inputs, padding_value=value, batch_first=True)
    else:
        raise ValueError('"input" argument must be tensor/list/ndarray.')


class TokenizerDictOutput(dict):
    """tokenizer以字典方式输出"""

    def to(self, device):
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self[k] = v
        return self


class TokenizerListOutput(list):
    """tokenizer以列表方式输出"""

    def to(self, device):
        for i, v in enumerate(self):
            if isinstance(v, torch.Tensor):
                self[i] = v
        return self


class Trie(object):
    """自定义Trie树对象，用来保存知识库
    """

    def __init__(self, value_key=-1):
        self.data = {}
        self.value_key = str(value_key)

    def __setitem__(self, key, value):
        """传入一对(key, value)到前缀树中
        """
        data = self.data
        for k in key:
            k = str(k)
            if k not in data:
                data[k] = {}
            data = data[k]
        if self.value_key in data:
            if data[self.value_key] != value:
                data[self.value_key] += '\t' + value
        else:
            data[self.value_key] = value

    def __getitem__(self, key):
        """获取key对应的value
        """
        data = self.data
        for k in key:
            k = str(k)
            data = data[k]
        return data[self.value_key]

    def next_ones(self, prefix):
        """获取prefix后一位的容许集
        """
        data = self.data
        for k in prefix:
            k = str(k)
            data = data[k]
        return [k for k in data if k != self.value_key]

    def keys(self, prefix=None, data=None):
        """获取以prefix开头的所有key
        """
        data = data or self.data
        prefix = prefix or []
        for k in prefix:
            k = str(k)
            if k not in data:
                return []
            data = data[k]
        results = []
        for k in data:
            if k == self.value_key:
                results.append([])
            else:
                results.extend([([k] + j) for j in self.keys(None, data[k])])
        return [(prefix + i) for i in results]

    def save(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False)

    def load(self, filename):
        with open(filename, encoding='utf-8') as f:
            self.data = json.load(f)


def is_string(s):
    """判断是否是字符串"""
    return isinstance(s, basestring)


def truncate_sequences(sequences: 'Iterable[List[int]]', maxlen: 'int', indices: 'Union[int, List[int], Tuple[int]]'=-1):
    """截断各个sequences以保证总长度至不超过maxlen, 原地修改，优先从最长的sequence开始截断
    :param sequences: List[List[int]], 需要截断的序列
    :param maxlen: int, 所有序列的总长度
    :param indices: int/List[int]/Tuple[int] 每次去掉的token_id的index

    ### Example
    ```python
    from bert4torch.snippets import truncate_sequences
    seq = [list(range(20)), list(range(30))]
    res = truncate_sequences(seq, maxlen=11, indices=-1)
    print(res, seq)
    # 输出：[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5]] [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5]]
    ```
    """
    sequences = [s for s in sequences if s]
    if not isinstance(indices, (list, tuple)):
        indices = [indices] * len(sequences)
    assert len(indices) == len(sequences)
    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > maxlen:
            i = np.argmax(lengths)
            sequences[i].pop(indices[i])
        else:
            return sequences


class TokenizerBase(object):
    """分词器基类
    """

    def __init__(self, token_start: 'str'='[CLS]', token_end: 'str'='[SEP]', token_unk: 'str'='[UNK]', token_pad: 'str'='[PAD]', token_mask: 'str'='[MASK]', add_special_tokens: 'Union[str, tuple, list]'=None, pre_tokenize: 'Callable'=None, token_translate: 'Dict'=None):
        """参数说明：
        token_unk: 未知词标记
        token_end: 句子切分标记，当只有一句话作为输入时，此标记知识作为结束符；当有两句话作为输入时，此标记作为分隔符、最后一句话的结束符
        pad_token: padding填充标记
        token_start: 分类标记，位于整个序列的第一个
        mask_token: mask标记
        pre_tokenize: 外部传入的分词函数，用作对文本进行预分词。如果传入pre_tokenize，则先执行pre_tokenize(text)，然后在它的基础上执行原本的tokenize函数；
        token_translate: 映射字典，主要用在tokenize之后，将某些特殊的token替换为对应的token。
        """
        self._token_pad = self.pad_token = token_pad
        self._token_unk = self.unk_token = token_unk
        self._token_mask = self.mask_token = token_mask
        self._token_start = self.start_token = token_start
        self._token_end = self.end_token = token_end
        self.never_split = [i for i in [self._token_unk, self._token_end, self._token_pad, self._token_start, self._token_mask] if isinstance(i, str)]
        if add_special_tokens is not None:
            if isinstance(add_special_tokens, (tuple, list)):
                self.never_split.extend(add_special_tokens)
            elif isinstance(add_special_tokens, str):
                self.never_split.append(add_special_tokens)
        self.tokens_trie = self._create_trie(self.never_split)
        self._pre_tokenize = pre_tokenize
        self._token_translate = token_translate or {}
        self._token_translate_inv = {v: k for k, v in self._token_translate.items()}

    def _create_trie(self, unique_no_split_tokens):
        trie = Trie()
        for token in unique_no_split_tokens:
            trie.add(token)
        return trie

    def tokenize(self, text: 'str', maxlen: 'int'=None) ->List[str]:
        """分词函数
        """
        tokens = [(self._token_translate.get(token) or token) for token in self._tokenize(text)]
        if self._token_start is not None:
            tokens.insert(0, self._token_start)
        if self._token_end is not None:
            tokens.append(self._token_end)
        if maxlen is not None:
            index = int(self._token_end is not None) + 1
            truncate_sequences([tokens], maxlen, -index)
        return tokens

    def token_to_id(self, token):
        """token转换为对应的id
        """
        raise NotImplementedError

    def tokens_to_ids(self, tokens: 'List[str]') ->List[int]:
        """token序列转换为对应的id序列
        """
        return [self.token_to_id(token) for token in tokens]

    def _encode(self, first_text: 'str', second_text: 'str'=None, maxlen: 'int'=None, pattern: 'str'='S*E*E', truncate_from: "Literal['left', 'right']"='right', return_offsets: "Literal['transformers', True, False]"=False):
        """输出文本对应token id和segment id
        """
        first_tokens = self.tokenize(first_text) if is_string(first_text) else first_text
        if second_text is None:
            second_tokens = None
        elif is_string(second_text):
            second_tokens = self.tokenize(second_text)
        else:
            second_tokens = second_text
        if maxlen is not None:
            if truncate_from == 'right':
                index = -int(self._token_end is not None) - 1
            elif truncate_from == 'left':
                index = int(self._token_start is not None)
            else:
                index = truncate_from
            if second_text is not None and pattern == 'S*E*E':
                maxlen += 1
            truncate_sequences([first_tokens, second_tokens], maxlen, index)
        first_token_ids = self.tokens_to_ids(first_tokens)
        first_segment_ids = [0] * len(first_token_ids)
        if second_text is not None:
            if pattern == 'S*E*E':
                idx = int(bool(self._token_start))
                second_tokens = second_tokens[idx:]
            second_token_ids = self.tokens_to_ids(second_tokens)
            second_segment_ids = [1] * len(second_token_ids)
            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)
        encode_output = [first_token_ids, first_segment_ids]
        if return_offsets != False:
            offset = self.rematch(first_text, first_tokens)
            if second_text is not None:
                offset += self.rematch(second_text, second_tokens)
            if return_offsets == 'transformers':
                encode_output.append([([0, 0] if not k else [k[0], k[-1] + 1]) for k in offset])
            else:
                encode_output.append(offset)
        return encode_output

    def __call__(self, *args: Any, **kwds: Any) ->Any:
        return self.encode(*args, **kwds)

    def encode(self, first_texts: 'Union[str, List[str]]', second_texts: 'Union[str, List[str]]'=None, maxlen: 'int'=None, pattern: 'str'='S*E*E', truncate_from: "Literal['left', 'right']"='right', return_offsets: "Literal['transformers', True, False]"=False, return_tensors: "Literal[True, 'pt', 'np']"=None, return_dict: 'bool'=False, **kwargs) ->Union[List[Union[List, np.ndarray, torch.Tensor]], Dict[str, Union[List, np.ndarray, torch.Tensor]]]:
        """可以处理多条或者单条
        :param first_texts: 需要encode的文本/文本列表
        :param second_texts: 需要encode的文本/文本列表二, 一般需要切分segment_ids使用, 默认未None
        :param maxlen: 允许的最大长度，
        :param pattern: 
        :param truncate_from: 超长时候是截断左侧还是截断右侧, 默认截断右侧right
        :param return_offsets: 是否返回char和token之间的位置映射关系
        :param return_tensors: 是否以tensor的形式返回
        :param return_dict: 是否以dict形式返回

        Returns
        :param input_ids: [CLS] + first_text + [SEP] + second_text
        :param attantion_mask: [1, 1, 1,..., 0, 0, 0]
        """
        maxlen = maxlen or kwargs.get('max_length')
        return_list = False if isinstance(first_texts, str) else True
        first_texts = [first_texts] if isinstance(first_texts, str) else first_texts
        second_texts = [second_texts] if isinstance(second_texts, str) else second_texts
        first_token_ids, first_segment_ids, offsets = [], [], []
        if second_texts is None:
            second_texts = [None] * len(first_texts)
        assert len(first_texts) == len(second_texts), 'first_texts and second_texts should be same length'
        for first_text, second_text in zip(first_texts, second_texts):
            outputs = self._encode(first_text, second_text, maxlen, pattern, truncate_from, return_offsets)
            first_token_ids.append(outputs[0])
            first_segment_ids.append(outputs[1])
            if len(outputs) >= 3:
                offsets.append(outputs[2])
        encode_outputs = OrderedDict()
        encode_outputs['input_ids'] = first_token_ids
        encode_outputs['token_type_ids'] = first_segment_ids
        if return_offsets:
            encode_outputs['offset'] = offsets
        if return_tensors in {True, 'pt', 'np'}:
            for key, value in encode_outputs.items():
                if key in {'input_ids', 'token_type_ids'}:
                    encode_outputs[key] = sequence_padding(value, value=self.pad_token_id)
                    if return_tensors == 'pt':
                        encode_outputs[key] = torch.tensor(encode_outputs[key], dtype=torch.long)
        elif not return_list:
            encode_outputs = OrderedDict({key: item[0] for key, item in encode_outputs.items()})
        if return_dict:
            return TokenizerDictOutput(encode_outputs)
        else:
            return TokenizerListOutput([value for value in encode_outputs.values()])

    def id_to_token(self, i):
        """id序列为对应的token
        """
        raise NotImplementedError

    def ids_to_tokens(self, ids):
        """id序列转换为对应的token序列
        """
        return [self.id_to_token(int(i)) for i in ids]

    def decode(self, ids):
        """转为可读文本
        """
        raise NotImplementedError

    def _tokenize(self, text):
        """基本分词函数
        """
        raise NotImplementedError

    def rematch(self, text: 'str', tokens: 'List[str]') ->List[List]:
        return []


class DecoderBase(BERT_BASE):
    passed_kwargs = {'use_states', 'position_ids', 'past_token_ids', 'pad_attention_mask', 'attention_mask', 'past_key_values', 'cross_past_key_values'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_inputs_for_generation(self, *inputs, **states):
        """为后续forward定义所需参数，方便继承"""
        return states

    def _update_model_kwargs_for_generation(self, model_kwargs: 'dict'):
        """需要返回给下一次generate使用到的要素，方便继承"""
        if 'states' in model_kwargs:
            return model_kwargs['states']
        return {k: v for k, v in model_kwargs.items() if k in self.passed_kwargs}

    def forward(self, *inputs, **model_kwargs):
        """定义模型的训练流程
        
        :param inputs: List[torch.Tensor], 默认顺序是[token_ids, segment_ids(若有), position_ids(若有), custom_attention_mask(若有), conditional_input(若有)]
        :return: List[torch.Tensor] or torch.Tensor, 模型输出，默认顺序为[last_hidden_state/all_encoded_layers, pooled_output(若有), mlm_scores(若有), nsp_scores(若有)]
        """
        inputs = self.args_segmentate(inputs, **model_kwargs)
        model_kwargs = self.apply_embeddings(*inputs, **model_kwargs)
        model_kwargs = self.apply_main_layers(**model_kwargs)
        outputs = self.apply_final_layers(**model_kwargs)
        if model_kwargs.get('use_states', False):
            return outputs, self._update_model_kwargs_for_generation(model_kwargs)
        else:
            return outputs

    def _prepare_generation(self, **kwargs):
        if not hasattr(self, 'generation'):
            self.generation = SeqGeneration(self, **kwargs)

    def generate(self, input_ids: 'Union[str, list, torch.Tensor]', **kwargs):
        """单条样本生成 / batch样本生成，use_states=True时要求pad_mode='pre'
        """
        self._prepare_generation(**kwargs)
        return self.generation.generate(input_ids, **kwargs)

    def stream_generate(self, input_ids: 'Union[str, torch.Tensor]', **kwargs):
        """单条样本stream输出预测的结果"""
        self._prepare_generation(**kwargs)
        yield from self.generation.stream_generate(input_ids, **kwargs)


class LM_Mask(object):
    """定义下三角Attention Mask（语言模型用）"""

    def compute_attention_bias(self, inputs=None):
        """通过idxs序列的比较来得到对应的mask"""
        token_ids = inputs[0]
        seq_len = token_ids.shape[1]
        attention_bias = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.long, device=inputs[0].device), diagonal=0)
        self.attention_bias = attention_bias.unsqueeze(0).unsqueeze(1)
        return self.attention_bias


class Decoder(LM_Mask, BERT, DecoderBase):
    """所有decoder模型的基类(含大模型)

    :param logit_scale: bool, 是否对lm_logits进行缩放
    :param final_layernorm: bool, 对last_hidden_state是否进行层归一化
    :param convert_lm_logits_dtype: bool, 是否对lm_logits进行dtype转换
    """

    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    @insert_arguments(with_lm=True)
    def __init__(self, *args, logit_scale: bool=False, final_layernorm: bool=False, convert_lm_logits_dtype: Literal['float16', 'float32', 'float64', 'bfloat16', None]=None, **kwargs):
        kwargs['vocab_size'] = kwargs.get('tgt_vocab_size', kwargs['vocab_size'])
        kwargs['is_decoder'] = True
        super().__init__(*args, **kwargs)
        self.is_decoder = True
        self.model_type = 'decoder'
        self.decoderLayer = self.encoderLayer
        del self.encoderLayer
        self.final_layernorm = final_layernorm
        mapping = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32, 'float64': torch.float64}
        self.convert_lm_logits_dtype = mapping[convert_lm_logits_dtype] if convert_lm_logits_dtype is not None else None
        if self.with_lm:
            self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
            self.final_activation = get_activation('linear' if self.with_lm is True else self.with_lm)
        self.tie_weights()
        if isinstance(logit_scale, bool) and logit_scale:
            self.logit_scale = self.hidden_size ** -0.5
        elif not isinstance(logit_scale, bool) and isinstance(logit_scale, (int, float)):
            self.logit_scale = logit_scale
        if self.final_layernorm:
            self.LayerNormFinal = LayerNorm(self.hidden_size, eps=kwargs.get('layer_norm_eps', 1e-12), conditional_size=self.conditional_size, norm_mode=kwargs.get('norm_mode', 'normal'), weight=kwargs.get('weight', True), bias=kwargs.get('bias', True))

    def tie_weights(self):
        if self.tie_word_embeddings and self.with_lm:
            self.lm_head.weight = self.embeddings.word_embeddings.weight

    def apply_main_layers(self, **model_kwargs):
        """Dencoder主体是基于Self-Attention、Cross-Attention的模块；
        顺序：Att1 --> Add --> LN --> Att2 --> Add -->  LN --> FFN --> Add --> LN
        """
        decoded_layers = [model_kwargs['hidden_states']]
        for l_i, layer_module in enumerate(self.decoderLayer):
            model_kwargs = self.apply_on_layer_begin(l_i, **model_kwargs)
            outputs = self.layer_forward(layer_module, model_kwargs)
            model_kwargs.update(outputs)
            hidden_states = model_kwargs['hidden_states']
            model_kwargs = self.apply_on_layer_end(l_i, **model_kwargs)
            if self.output_all_encoded_layers:
                decoded_layers.append(hidden_states)
        if not self.output_all_encoded_layers:
            decoded_layers.append(hidden_states)
        model_kwargs['decoded_layers'] = decoded_layers
        return model_kwargs

    def apply_final_layers(self, **model_kwargs):
        last_hidden_state = model_kwargs['decoded_layers'][-1]
        if self.final_layernorm:
            last_hidden_state = self.LayerNormFinal(last_hidden_state)
        if self.with_lm:
            lm_logits = self.lm_head(last_hidden_state)
            lm_logits = lm_logits * self.logit_scale if hasattr(self, 'logit_scale') else lm_logits
            lm_logits = self.final_activation(lm_logits)
            if self.convert_lm_logits_dtype is not None:
                lm_logits = lm_logits
            return self.gen_outputs(locals(), last_hidden_state, lm_logits) if self.return_dict else lm_logits
        elif not self.return_dict:
            return last_hidden_state
        else:
            return self.gen_outputs(locals(), last_hidden_state)

    def load_variable(self, variable, old_key, new_key, prefix='decoder'):
        """加载单个变量的函数, 这里的名称均为映射前的"""
        mapping = self.variable_mapping()
        if old_key in {f'{prefix}.embeddings.word_embeddings.weight', f'{prefix}.lm_head.weight'}:
            return self.load_embeddings(variable)
        elif new_key in {'embeddings.word_embeddings.weight', 'lm_head.weight'}:
            return self.load_embeddings(variable)
        else:
            return variable

    def variable_mapping(self, prefix='decoder'):
        raw_mapping = super().variable_mapping(prefix=prefix)
        mapping = {}
        for k, v in raw_mapping.items():
            mapping[k.replace('encoderLayer', 'decoderLayer')] = v
        if self.final_layernorm:
            mapping.update({'LayerNormFinal.weight': f'{prefix}.LayerNormFinal.weight', 'LayerNormFinal.bias': f'{prefix}.LayerNormFinal.bias'})
        if self.with_lm and not self.tie_word_embeddings:
            mapping.update({'lm_head.weight': f'{prefix}.lm_head.weight'})
        return mapping


class Transformer(DecoderBase):
    """encoder-decoder结构
    :param tie_word_embeddings: bool, decoder的word_embeddings和lm_head的权重共享
    :param tie_word_embeddings_encoder_decoder: bool, encoder和decoder之间的word_embedding权重共享
    """

    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, tie_word_embeddings: bool=False, tie_word_embeddings_encoder_decoder: bool=False, **kwargs):
        super(Transformer, self).__init__(*args, **kwargs)
        self.max_position = kwargs['max_position']
        self.tie_word_embeddings = kwargs['tie_word_embeddings'] = tie_word_embeddings
        self.tie_word_embeddings_encoder_decoder = tie_word_embeddings_encoder_decoder
        self.is_encoder_decoder = True
        self.model_type = 'transformer'
        self.encoder = Encoder(*args, **kwargs)
        self.decoder = Decoder(*args, add_cross_attention=True, **kwargs)

    def tie_weights(self):
        self.decoder.tie_weights()
        if self.tie_word_embeddings_encoder_decoder:
            assert self.encoder.vocab_size == self.decoder.vocab_size, 'To share word embedding, the vocab size of src/tgt shall be the same.'
            self.decoder.embeddings.word_embeddings.weight = self.encoder.embeddings.word_embeddings.weight

    def forward(self, *inputs):
        inputs = self.args_segmentate(inputs)
        encoder_input, decoder_input = inputs[:2]
        encoder_hidden_states, encoder_attention_mask = self.encoder(encoder_input)
        decoder_outputs = self.decoder(decoder_input + [encoder_hidden_states, encoder_attention_mask])
        return [encoder_hidden_states] + [decoder_outputs]

    def _prepare_generation(self, **kwargs):
        if not hasattr(self, 'generation'):
            self.generation = Seq2SeqGeneration(self, **kwargs)


class Transformer_XL(BERT):
    """构建transformer-xl模型, 已加载；
    项目: https://github.com/kimiyoung/transformer-xl；
    不同点:  
    1) 简化了原有的AdaptiveEmbedding(可选)和未使用ProjectedAdaptiveLogSoftmax, 直接输出last_hidden_state；
    2) mems修改了transformer中初始化为zero_tensor, 改为包含最后一层, 原项目初始化为empty_tensor；
    3) SinusoidalPositionEncoding一般是sincos间隔排列, 这里是先sin后cos；
    4) attention_mask在multi_attn中使用中使用1e30来替代原来的1000。
    """

    @delete_arguments('with_pool', 'with_nsp', 'with_mlm')
    @insert_arguments(with_lm=False)
    def __init__(self, *args, mem_len=0, same_length=False, clamp_len=-1, **kwargs):
        kwargs.update({'p_bias': 'MultiHeadAttention'})
        self.attn_type = kwargs.pop('attn_type', 0)
        self.mem_len, self.same_length, self.clamp_len = mem_len, same_length, clamp_len
        super().__init__(*args, **kwargs)
        if kwargs.get('adaptive_embedding'):
            cutoffs, div_val, sample_softmax = kwargs.get('cutoffs', []), kwargs.get('div_val', 1), kwargs.get('sample_softmax', False)
            self.embeddings = AdaptiveEmbedding(self.vocab_size, self.embedding_size, self.hidden_size, cutoffs, div_val, sample_softmax)
        else:
            self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.pos_embeddings = XlnetPositionsEncoding(self.embedding_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        if not kwargs.get('untie_r'):
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.num_attention_heads, self.attention_head_size))
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.num_attention_heads, self.attention_head_size))
            if self.segment_vocab_size > 0:
                self.r_s_bias = nn.Parameter(torch.FloatTensor(self.num_attention_heads, self.attention_head_size))
        else:
            self.r_w_bias, self.r_r_bias = None, None
            self.r_s_bias = None
        self.encoderLayer = nn.ModuleList([(XlnetLayer(layer_idx=layer_idx, r_s_bias=None, **self.get_kw('r_w_bias', 'r_r_bias', *self._layer_args, **kwargs)) if layer_idx in self.keep_hidden_layers else BlockIdentity()) for layer_idx in range(self.num_hidden_layers)])
        if self.with_lm:
            self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=True)
        self.model_type = 'transformer_xl'

    def init_mems(self, bsz):
        """初始化mems, 用于记忆mlen的各层隐含层状态"""
        if isinstance(self.mem_len, (int, float)) and self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for _ in range(self.num_hidden_layers + 1):
                empty = torch.zeros(bsz, self.mem_len, self.hidden_size, dtype=param.dtype, device=param.device)
                mems.append(empty)
            return mems
        else:
            return None

    def _update_mems(self, hids, mlen, qlen):
        """更新mems"""
        if self.mems is None:
            return None
        assert len(hids) == len(self.mems), 'len(hids) != len(mems)'
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([self.mems[i], hids[i]], dim=1)
                new_mems.append(cat[:, beg_idx:end_idx].detach())
        self.mems = new_mems

    def relative_positional_encoding(self, qlen, klen, device):
        pos_seq = torch.arange(klen - 1, -1, -1.0, device=device, dtype=torch.long)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.dropout(self.pos_embeddings(pos_seq))
        return pos_emb

    def create_mask(self, word_emb, qlen, klen, mlen):
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            mask_shift_len = qlen - mask_len if mask_len > 0 else qlen
            attention_mask = 1 - (torch.triu(all_ones, 1 + mlen) + torch.tril(all_ones, -mask_shift_len)).byte()
        else:
            attention_mask = torch.tril(word_emb.new_ones(qlen, klen), diagonal=mlen).byte()
        attention_mask = attention_mask[None, None, :, :]
        return attention_mask

    def apply_embeddings(self, *inputs, **model_kwargs):
        """接受的inputs输入: [token_ids, segment_ids], 暂不支持条件LayerNorm输入"""
        assert isinstance(inputs, (tuple, list)), f'Inputs only support list,tuple format but passed {type(inputs)}'
        index_ = 0
        if model_kwargs.get('input_ids') is not None:
            token_ids = model_kwargs['input_ids']
        elif model_kwargs.get('token_ids') is not None:
            token_ids = model_kwargs['token_ids']
        else:
            token_ids = inputs[0]
            index_ += 1
        self.mems = self.init_mems(token_ids.size(0))
        word_emb = self.dropout(self.embeddings(token_ids))
        index_ = 1
        btz, qlen = token_ids.shape[:2]
        mlen = self.mems[0].size(1) if self.mems is not None else 0
        klen = mlen + qlen
        pos_emb = self.relative_positional_encoding(qlen, klen, word_emb.device)
        if model_kwargs.get('segment_ids') is not None:
            segment_ids = model_kwargs['segment_ids']
        elif model_kwargs.get('token_type_ids') is not None:
            segment_ids = model_kwargs['token_type_ids']
        elif self.segment_vocab_size > 0:
            segment_ids = inputs[index_]
            index_ += 1
        else:
            segment_ids = None
        if segment_ids is not None:
            if mlen > 0:
                mem_pad = torch.zeros([btz, mlen], dtype=torch.long, device=word_emb.device)
                cat_ids = torch.cat([mem_pad, segment_ids], dim=1)
            else:
                cat_ids = segment_ids
            segment_ids = (segment_ids[:, :, None] != cat_ids[:, None]).long()
        if self.attn_type in {'uni', 0}:
            non_tgt_mask = self.create_mask(word_emb, qlen, klen, mlen)
        elif self.attn_type == 'bi':
            attention_mask = (token_ids != self.pad_token_id).long().unsqueeze(1).unsqueeze(2)
            non_tgt_mask = torch.eye(qlen)[None, None, :, :]
            non_tgt_mask = (1 - attention_mask - non_tgt_mask <= 0).long()
        model_kwargs.update({'hidden_states': word_emb, 'segment_ids': segment_ids, 'pos_emb': pos_emb, 'attention_mask': non_tgt_mask})
        return model_kwargs

    def apply_main_layers(self, **model_kwargs):
        encoded_layers = [model_kwargs['hidden_states']]
        for l_i, layer_module in enumerate(self.encoderLayer):
            mems_i = None if self.mems is None else self.mems[l_i]
            model_kwargs['mems_i'] = mems_i
            model_kwargs = self.apply_on_layer_begin(l_i, **model_kwargs)
            outputs = self.layer_forward(layer_module, model_kwargs)
            model_kwargs.update(outputs)
            hidden_states = model_kwargs['hidden_states']
            model_kwargs = self.apply_on_layer_end(l_i, **model_kwargs)
            encoded_layers.append(hidden_states)
        hidden_states = self.dropout(hidden_states)
        qlen = hidden_states.size(1)
        mlen = self.mems[0].size(0) if self.mems is not None else 0
        self._update_mems(encoded_layers, mlen, qlen)
        if not self.output_all_encoded_layers:
            encoded_layers = encoded_layers[:1] + [hidden_states]
        model_kwargs['encoded_layers'] = encoded_layers
        return model_kwargs

    def load_variable(self, variable, old_key, new_key):
        if self.keep_tokens is not None or self.compound_tokens is not None:
            raise ValueError('Custom keep_tokens and compound_tokens is not yet supported in Transformer_XL')
        return variable

    def load_trans_ckpt(self, checkpoint):
        state_dict = super().load_trans_ckpt(checkpoint)
        for i in range(self.num_hidden_layers):
            qkv_net = state_dict.pop(f'transformer.layers.{i}.dec_attn.qkv_net.weight')
            for k, v in zip(['q', 'k', 'v'], qkv_net.chunk(3, dim=0)):
                state_dict[f'encoderLayer.{i}.multiHeadAttention.{k}.weight'] = v
        return state_dict

    def save_trans_ckpt(self):
        state_dict = self.state_dict()
        for i in range(self.num_hidden_layers):
            qkv = []
            old_key = 'encoderLayer.{}.multiHeadAttention.{}.weight'
            for i_k in ['q', 'k', 'v']:
                if old_key.format(i, i_k) in state_dict:
                    qkv.append(state_dict.pop(old_key.format(i, i_k)))
            if qkv:
                state_dict[f'transformer.layers.{i}.dec_attn.qkv_net.weight'] = torch.cat(qkv)
        return state_dict

    def variable_mapping(self):
        mapping = {'embeddings.emb_layers.0.weight': 'transformer.word_emb.emb_layers.0.weight', 'embeddings.emb_layers.1.weight': 'transformer.word_emb.emb_layers.1.weight', 'embeddings.emb_layers.2.weight': 'transformer.word_emb.emb_layers.2.weight', 'embeddings.emb_layers.3.weight': 'transformer.word_emb.emb_layers.3.weight', 'embeddings.emb_projs.0': 'transformer.word_emb.emb_projs.0', 'embeddings.emb_projs.1': 'transformer.word_emb.emb_projs.1', 'embeddings.emb_projs.2': 'transformer.word_emb.emb_projs.2', 'embeddings.emb_projs.3': 'transformer.word_emb.emb_projs.3'}
        for i in range(self.num_hidden_layers):
            mapping.update({f'encoderLayer.{i}.multiHeadAttention.r_r_bias': f'transformer.layers.{i}.dec_attn.r_r_bias', f'encoderLayer.{i}.multiHeadAttention.r_w_bias': f'transformer.layers.{i}.dec_attn.r_w_bias', f'encoderLayer.{i}.multiHeadAttention.o.weight': f'transformer.layers.{i}.dec_attn.o_net.weight', f'encoderLayer.{i}.attnLayerNorm.weight': f'transformer.layers.{i}.dec_attn.layer_norm.weight', f'encoderLayer.{i}.attnLayerNorm.bias': f'transformer.layers.{i}.dec_attn.layer_norm.bias', f'encoderLayer.{i}.multiHeadAttention.r.weight': f'transformer.layers.{i}.dec_attn.r_net.weight', f'encoderLayer.{i}.feedForward.intermediateDense.weight': f'transformer.layers.{i}.pos_ff.CoreNet.0.weight', f'encoderLayer.{i}.feedForward.intermediateDense.bias': f'transformer.layers.{i}.pos_ff.CoreNet.0.bias', f'encoderLayer.{i}.feedForward.outputDense.weight': f'transformer.layers.{i}.pos_ff.CoreNet.3.weight', f'encoderLayer.{i}.feedForward.outputDense.bias': f'transformer.layers.{i}.pos_ff.CoreNet.3.bias', f'encoderLayer.{i}.ffnLayerNorm.weight': f'transformer.layers.{i}.pos_ff.layer_norm.weight', f'encoderLayer.{i}.ffnLayerNorm.bias': f'transformer.layers.{i}.pos_ff.layer_norm.bias'})
        return mapping


class UIE(BERT):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_size = self.hidden_size
        self.linear_start = nn.Linear(hidden_size, 1)
        self.linear_end = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        if kwargs.get('use_task_id') and kwargs.get('use_task_id'):
            task_type_embeddings = nn.Embedding(kwargs.get('task_type_vocab_size'), self.hidden_size)
            self.embeddings.task_type_embeddings = task_type_embeddings

            def hook(module, input, output):
                return output + task_type_embeddings(torch.zeros(input[0].size(), dtype=torch.int64, device=input[0].device))
            self.embeddings.word_embeddings.register_forward_hook(hook)

    def forward(self, token_ids, token_type_ids):
        outputs = super().forward([token_ids, token_type_ids])
        sequence_output = outputs[0]
        start_logits = self.linear_start(sequence_output)
        start_logits = torch.squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits)
        end_logits = self.linear_end(sequence_output)
        end_logits = torch.squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits)
        return start_prob, end_prob

    @torch.no_grad()
    def predict(self, token_ids, token_type_ids):
        self.eval()
        start_prob, end_prob = self.forward(token_ids, token_type_ids)
        return start_prob, end_prob


class XLNET(Transformer_XL):
    """构建xlnet模型, 这里做了简化, 只用来finetune, 即没有perm_mask, target_mapping这些输入；
       接受的inputs输入: [token_ids, segment_ids]
    """

    def __init__(self, *args, bi_data=False, **kwargs):
        self.attn_type = kwargs.get('attn_type', 'bi')
        self.bi_data = bi_data
        kwargs['rel_shift_opt'] = 'xlnet'
        super().__init__(*args, **kwargs)
        self.model_type = 'xlnet'

    def relative_positional_encoding(self, qlen, klen, device):
        if self.attn_type == 'bi':
            beg, end = klen, -qlen
        elif self.attn_type == 'uni':
            beg, end = klen, -1
        else:
            raise ValueError(f'Unknown `attn_type` {self.attn_type}.')
        pos_seq = torch.arange(beg, end, -1.0, device=device, dtype=torch.long)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        fwd_pos_emb = self.pos_embeddings(pos_seq)
        if self.bi_data:
            pos_seq = torch.arange(-beg, -end, -1.0, device=device, dtype=torch.long)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            bwd_pos_emb = self.pos_embeddings(pos_seq)
            pos_emb = torch.cat([fwd_pos_emb, bwd_pos_emb], dim=0)
        else:
            pos_emb = fwd_pos_emb
        pos_emb = self.dropout(pos_emb)
        return pos_emb

    def apply_final_layers(self, **model_kwargs):
        outputs = super().apply_final_layers(**model_kwargs)
        last_hidden_state = outputs['last_hidden_state'] if self.return_dict else outputs
        if self.with_lm:
            lm_logits = self.lm_head(last_hidden_state)
            return self.gen_outputs(locals(), last_hidden_state, lm_logits) if self.return_dict else [last_hidden_state, lm_logits]
        elif not self.return_dict:
            return last_hidden_state
        else:
            return self.gen_outputs(locals(), last_hidden_state)

    def load_variable(self, variable, old_key, new_key):
        if old_key in {f'transformer.word_embedding.weight', 'lm_loss.weight', 'lm_loss.bias'}:
            return self.load_embeddings(variable)
        else:
            return variable

    def load_trans_ckpt(self, checkpoint):
        state_dict = torch.load(checkpoint, map_location='cpu')
        for i in range(self.num_hidden_layers):
            prefix_i = f'transformer.layer.%d.' % i
            mapping = {(prefix_i + 'rel_attn.q'): f'encoderLayer.{i}.multiHeadAttention.q.weight', (prefix_i + 'rel_attn.k'): f'encoderLayer.{i}.multiHeadAttention.k.weight', (prefix_i + 'rel_attn.v'): f'encoderLayer.{i}.multiHeadAttention.v.weight', (prefix_i + 'rel_attn.r'): f'encoderLayer.{i}.multiHeadAttention.r.weight'}
            for old_key, new_key in mapping.items():
                if state_dict.get(old_key) is not None:
                    variable = state_dict.pop(old_key)
                    state_dict[new_key] = variable.reshape(variable.shape[0], -1).T
            old_key = prefix_i + 'rel_attn.o'
            if state_dict.get(old_key) is not None:
                variable = state_dict.pop(old_key)
                state_dict[f'encoderLayer.{i}.multiHeadAttention.o.weight'] = variable.reshape(variable.shape[0], -1)
        return state_dict

    def save_trans_ckpt(self):
        state_dict = self.state_dict()
        for i in range(self.num_hidden_layers):
            prefix_i = f'transformer.layer.%d.' % i
            mapping = {f'encoderLayer.{i}.multiHeadAttention.q.weight': prefix_i + 'rel_attn.q', f'encoderLayer.{i}.multiHeadAttention.k.weight': prefix_i + 'rel_attn.k', f'encoderLayer.{i}.multiHeadAttention.v.weight': prefix_i + 'rel_attn.v', f'encoderLayer.{i}.multiHeadAttention.r.weight': prefix_i + 'rel_attn.r'}
            for old_key, new_key in mapping.items():
                if state_dict.get(old_key) is not None:
                    variable = state_dict.pop(old_key)
                    state_dict[new_key] = variable.T.reshape(-1, self.num_attention_heads, self.attention_head_size)
            old_key = f'encoderLayer.{i}.multiHeadAttention.o.weight'
            if state_dict.get(old_key) is not None:
                variable = state_dict.pop(old_key)
                state_dict[prefix_i + 'rel_attn.o'] = variable.reshape(-1, self.num_attention_heads, self.attention_head_size)
        return state_dict

    def variable_mapping(self):
        mapping = {'embeddings.weight': f'transformer.word_embedding.weight', 'lm_head.weight': 'lm_loss.weight', 'lm_head.bias': 'lm_loss.bias'}
        for i in range(self.num_hidden_layers):
            prefix_i = f'transformer.layer.%d.' % i
            mapping.update({f'encoderLayer.{i}.multiHeadAttention.r_r_bias': prefix_i + 'rel_attn.r_r_bias', f'encoderLayer.{i}.multiHeadAttention.r_s_bias': prefix_i + 'rel_attn.r_s_bias', f'encoderLayer.{i}.multiHeadAttention.r_w_bias': prefix_i + 'rel_attn.r_w_bias', f'encoderLayer.{i}.multiHeadAttention.seg_embed': prefix_i + 'rel_attn.seg_embed', f'encoderLayer.{i}.attnLayerNorm.weight': prefix_i + 'rel_attn.layer_norm.weight', f'encoderLayer.{i}.attnLayerNorm.bias': prefix_i + 'rel_attn.layer_norm.bias', f'encoderLayer.{i}.feedForward.intermediateDense.weight': prefix_i + 'ff.layer_1.weight', f'encoderLayer.{i}.feedForward.intermediateDense.bias': prefix_i + 'ff.layer_1.bias', f'encoderLayer.{i}.feedForward.outputDense.weight': prefix_i + 'ff.layer_2.weight', f'encoderLayer.{i}.feedForward.outputDense.bias': prefix_i + 'ff.layer_2.bias', f'encoderLayer.{i}.ffnLayerNorm.weight': prefix_i + 'ff.layer_norm.weight', f'encoderLayer.{i}.ffnLayerNorm.bias': prefix_i + 'ff.layer_norm.bias'})
        return mapping


class QuantizedEmbedding(Embedding):

    def __init__(self, weight_bit_width: 'int', weight_tensor=None, quantized_weight=None, quantized_weight_scale=None, empty_init=False, *args, **kwargs):
        super(QuantizedEmbedding, self).__init__(*args, **kwargs)
        self.weight_bit_width = weight_bit_width
        if quantized_weight is not None and quantized_weight_scale is not None:
            del self.weight
            self.weight = Parameter(quantized_weight, requires_grad=False)
            self.weight_scale = Parameter(quantized_weight_scale, requires_grad=False)
        else:
            shape = self.weight.shape
            del self.weight
            if weight_tensor is None or empty_init:
                self.weight = torch.empty(shape[0], shape[1] * weight_bit_width // 8, dtype=torch.int8, device=kwargs['device'])
                self.weight_scale = torch.empty(shape[0], dtype=kwargs['dtype'], device=kwargs['device'])
            else:
                self.weight_scale = weight_tensor.abs().max(dim=-1).values / (2 ** (weight_bit_width - 1) - 1)
                self.weight = torch.round(weight_tensor / self.weight_scale[:, None])
                if weight_bit_width == 4:
                    self.weight = compress_int4_weight(self.weight)
            self.weight = Parameter(self.weight, requires_grad=False)
            self.weight_scale = Parameter(self.weight_scale, requires_grad=False)

    def forward(self, input):
        original_weight = extract_weight_to_half(weight=self.weight, scale_list=self.weight_scale, source_bit_width=self.weight_bit_width)
        output = F.embedding(input, original_weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        return output


class PrefixEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        self.shape_4 = config.hidden_size // config.num_attention_heads
        if config.get('num_key_value_heads') is not None:
            self.shape_3 = config.num_key_value_heads
            embed_size = self.shape_3 * self.shape_4
        else:
            self.shape_3 = config.num_attention_heads
            embed_size = config.hidden_size
        if self.prefix_projection:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(torch.nn.Linear(config.hidden_size, config.hidden_size), torch.nn.Tanh(), torch.nn.Linear(config.hidden_size, config.num_hidden_layers * embed_size * 2))
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * embed_size * 2)

    def forward(self, prefix: 'torch.Tensor'):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class MyLoss(nn.Module):

    def forward(self, y_pred, y_true_sup):
        y_pred_sup = y_pred[:y_true_sup.shape[0]]
        return F.cross_entropy(y_pred_sup, y_true_sup)


class CrossEntropyLoss(nn.CrossEntropyLoss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, outputs, y_true):
        y_pred = outputs[-1]
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        return super().forward(y_pred, y_true)


class BERT_THESEUS(BERT):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layer = BertLayer(self.hidden_size, self.num_attention_heads, self.dropout_rate, self.attention_probs_dropout_prob, self.intermediate_size, self.hidden_act, is_dropout=False, conditional_size=self.conditional_size)
        self.encoderLayer = nn.ModuleList(nn.ModuleList([copy.deepcopy(layer) for _ in range(self.num_hidden_layers)]))
        self.scc_n_layer = 6
        self.scc_layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.scc_n_layer)])
        self.compress_ratio = self.num_hidden_layers // self.scc_n_layer
        self.bernoulli = None

    def set_replacing_rate(self, replacing_rate):
        if not 0 < replacing_rate <= 1:
            raise Exception('Replace rate must be in the range (0, 1]!')
        self.bernoulli = Bernoulli(torch.tensor([replacing_rate]))

    def apply_main_layers(self, **model_kwargs):
        """BERT的主体是基于Self-Attention的模块
        顺序:Att --> Add --> LN --> FFN --> Add --> LN
        v0.2.8以后输入输出是以字典形式，这里进行修改
        """
        hidden_states, attention_mask, conditional_emb = model_kwargs['hidden_states'], model_kwargs['attention_mask'], model_kwargs['conditional_emb']
        encoded_layers = [hidden_states]
        if self.training:
            inference_layers = []
            for i in range(self.scc_n_layer):
                if self.bernoulli.sample() == 1:
                    inference_layers.append(self.scc_layer[i])
                else:
                    for offset in range(self.compress_ratio):
                        inference_layers.append(self.encoderLayer[i * self.compress_ratio + offset])
        else:
            inference_layers = self.scc_layer
        for i, layer_module in enumerate(inference_layers):
            outputs = layer_module(hidden_states, attention_mask, conditional_emb)
            hidden_states = outputs['hidden_states']
            model_kwargs.update(outputs)
            if self.output_all_encoded_layers:
                encoded_layers.append(hidden_states)
        if not self.output_all_encoded_layers:
            encoded_layers.append(hidden_states)
        model_kwargs['encoded_layers'] = encoded_layers
        return model_kwargs


class ALBERT(BERT):

    def __init__(self, *args, **kwargs):
        super(ALBERT, self).__init__(*args, **kwargs)
        self.encoderLayer = nn.ModuleList([self.encoderLayer[0]])
        self.model_type = 'albert'

    def apply_main_layers(self, **model_kwargs):
        """BERT的主体是基于Self-Attention的模块（和BERT区别是始终使用self.encoderLayer[0]）；
        顺序:Att --> Add --> LN --> FFN --> Add --> LN
        """
        encoded_layers = [model_kwargs['hidden_states']]
        for l_i in range(self.num_hidden_layers):
            model_kwargs = self.apply_on_layer_begin(l_i, **model_kwargs)
            layer_module = self.encoderLayer[0]
            outputs = self.layer_forward(layer_module, model_kwargs)
            model_kwargs.update(outputs)
            hidden_states = model_kwargs['hidden_states']
            model_kwargs = self.apply_on_layer_end(l_i, **model_kwargs)
            if self.output_all_encoded_layers:
                encoded_layers.append(hidden_states)
        if not self.output_all_encoded_layers:
            encoded_layers.append(hidden_states)
        model_kwargs['encoded_layers'] = encoded_layers
        return model_kwargs

    def variable_mapping(self):
        mapping = {'embeddings.word_embeddings.weight': 'albert.embeddings.word_embeddings.weight', 'embeddings.position_embeddings.weight': 'albert.embeddings.position_embeddings.weight', 'embeddings.segment_embeddings.weight': 'albert.embeddings.token_type_embeddings.weight', 'embeddings.layerNorm.weight': 'albert.embeddings.LayerNorm.weight', 'embeddings.layerNorm.bias': 'albert.embeddings.LayerNorm.bias', 'embeddings.embedding_hidden_mapping_in.weight': 'albert.encoder.embedding_hidden_mapping_in.weight', 'embeddings.embedding_hidden_mapping_in.bias': 'albert.encoder.embedding_hidden_mapping_in.bias', 'pooler.weight': 'albert.pooler.weight', 'pooler.bias': 'albert.pooler.bias', 'nsp.weight': 'sop_classifier.classifier.weight', 'nsp.bias': 'sop_classifier.classifier.bias', 'mlmDense.weight': 'predictions.dense.weight', 'mlmDense.bias': 'predictions.dense.bias', 'mlmLayerNorm.weight': 'predictions.LayerNorm.weight', 'mlmLayerNorm.bias': 'predictions.LayerNorm.bias', 'mlmBias': 'predictions.bias', 'mlmDecoder.weight': 'predictions.decoder.weight', 'mlmDecoder.bias': 'predictions.decoder.bias'}
        i = 0
        prefix_i = f'albert.encoder.albert_layer_groups.{i}.albert_layers.{i}.'
        mapping.update({f'encoderLayer.{i}.multiHeadAttention.q.weight': prefix_i + 'attention.query.weight', f'encoderLayer.{i}.multiHeadAttention.q.bias': prefix_i + 'attention.query.bias', f'encoderLayer.{i}.multiHeadAttention.k.weight': prefix_i + 'attention.key.weight', f'encoderLayer.{i}.multiHeadAttention.k.bias': prefix_i + 'attention.key.bias', f'encoderLayer.{i}.multiHeadAttention.v.weight': prefix_i + 'attention.value.weight', f'encoderLayer.{i}.multiHeadAttention.v.bias': prefix_i + 'attention.value.bias', f'encoderLayer.{i}.multiHeadAttention.o.weight': prefix_i + 'attention.dense.weight', f'encoderLayer.{i}.multiHeadAttention.o.bias': prefix_i + 'attention.dense.bias', f'encoderLayer.{i}.attnLayerNorm.weight': prefix_i + 'attention.LayerNorm.weight', f'encoderLayer.{i}.attnLayerNorm.bias': prefix_i + 'attention.LayerNorm.bias', f'encoderLayer.{i}.feedForward.intermediateDense.weight': prefix_i + 'ffn.weight', f'encoderLayer.{i}.feedForward.intermediateDense.bias': prefix_i + 'ffn.bias', f'encoderLayer.{i}.feedForward.outputDense.weight': prefix_i + 'ffn_output.weight', f'encoderLayer.{i}.feedForward.outputDense.bias': prefix_i + 'ffn_output.bias', f'encoderLayer.{i}.ffnLayerNorm.weight': prefix_i + 'full_layer_layer_norm.weight', f'encoderLayer.{i}.ffnLayerNorm.bias': prefix_i + 'full_layer_layer_norm.bias'})
        return mapping

    def load_variable(self, variable, old_key, new_key):
        if old_key in {'albert.embeddings.word_embeddings.weight', 'predictions.bias', 'predictions.decoder.weight', 'predictions.decoder.bias'}:
            return self.load_embeddings(variable)
        elif old_key == 'sop_classifier.classifier.weight':
            return variable.T
        else:
            return variable


class ALBERT_Unshared(ALBERT):

    def __init__(self, *args, **kwargs):
        super(ALBERT_Unshared, self).__init__(*args, **kwargs)
        self.encoderLayer = nn.ModuleList([copy.deepcopy(self.encoderLayer[0]) for _ in range(self.num_hidden_layers)])
        self.model_type = 'albert_unshared'

    def apply_main_layers(self, **model_kwargs):
        """BERT的主体是基于Self-Attention的模块（和ALBERT区别是所有层权重独立）；这里就是调用BERT类的方法
        顺序:Att --> Add --> LN --> FFN --> Add --> LN
        """
        return BERT.apply_main_layers(self, **model_kwargs)

    def variable_mapping(self):
        mapping = super().variable_mapping()
        prefix_0 = f'albert.encoder.albert_layer_groups.0.albert_layers.0.'
        for i in range(1, self.num_hidden_layers):
            mapping.update({f'encoderLayer.{i}.multiHeadAttention.q.weight': prefix_0 + 'attention.query.weight', f'encoderLayer.{i}.multiHeadAttention.q.bias': prefix_0 + 'attention.query.bias', f'encoderLayer.{i}.multiHeadAttention.k.weight': prefix_0 + 'attention.key.weight', f'encoderLayer.{i}.multiHeadAttention.k.bias': prefix_0 + 'attention.key.bias', f'encoderLayer.{i}.multiHeadAttention.v.weight': prefix_0 + 'attention.value.weight', f'encoderLayer.{i}.multiHeadAttention.v.bias': prefix_0 + 'attention.value.bias', f'encoderLayer.{i}.multiHeadAttention.o.weight': prefix_0 + 'attention.dense.weight', f'encoderLayer.{i}.multiHeadAttention.o.bias': prefix_0 + 'attention.dense.bias', f'encoderLayer.{i}.attnLayerNorm.weight': prefix_0 + 'attention.LayerNorm.weight', f'encoderLayer.{i}.attnLayerNorm.bias': prefix_0 + 'attention.LayerNorm.bias', f'encoderLayer.{i}.feedForward.intermediateDense.weight': prefix_0 + 'ffn.weight', f'encoderLayer.{i}.feedForward.intermediateDense.bias': prefix_0 + 'ffn.bias', f'encoderLayer.{i}.feedForward.outputDense.weight': prefix_0 + 'ffn_output.weight', f'encoderLayer.{i}.feedForward.outputDense.bias': prefix_0 + 'ffn_output.bias', f'encoderLayer.{i}.ffnLayerNorm.weight': prefix_0 + 'full_layer_layer_norm.weight', f'encoderLayer.{i}.ffnLayerNorm.bias': prefix_0 + 'full_layer_layer_norm.bias'})
        return mapping


class BART(Transformer):
    """BART: encoder-decoder结构
    decoder: tie_word_embeddings=True
    encoder-decoder: tie_word_embeddings_encoder_decoder=False
    """

    def __init__(self, *args, tie_word_embeddings: bool=True, **kwargs):
        kwargs['tie_word_embeddings'] = tie_word_embeddings
        kwargs['logit_scale'] = kwargs.get('logit_scale', False)
        self.postion_offset = 2
        kwargs['max_position'] = kwargs['max_position'] + self.postion_offset
        super(BART, self).__init__(*args, **kwargs)
        self.model_type = 'bart'

    def load_variable(self, variable, old_key, new_key):
        if old_key in {'shared.weight', 'encoder.embed_tokens.weight', 'decoder.embed_tokens.weight'}:
            return self.load_embeddings(variable)
        else:
            return variable

    def load_trans_ckpt(self, checkpoint):
        state_dict = super().load_trans_ckpt(checkpoint)
        mapping_reverse = {v: k for k, v in self.variable_mapping().items()}
        mapping = {}
        for k in list(state_dict.keys()):
            if k in {'final_logits_bias', 'lm_head.weight', 'shared.weight', 'model.shared.weight'}:
                continue
            if 'model.' in k:
                k_new = k.replace('model.', '')
                mapping[mapping_reverse[k_new]] = k
            else:
                k_new = k
            if 'embed_positions.weight' in k:
                state_dict[k] = torch.cat([state_dict[k][self.postion_offset:], state_dict[k][:self.postion_offset]])
        self.variable_mapping = modify_variable_mapping(self.variable_mapping, **mapping)
        return state_dict

    def save_trans_ckpt(self):
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if 'embeddings.position_embeddings.weight' in k:
                state_dict[k] = torch.cat([state_dict[k][-self.postion_offset:], state_dict[k][:-self.postion_offset]])
        return state_dict

    def variable_mapping(self):
        mapping = {'encoder.embeddings.word_embeddings.weight': 'encoder.embed_tokens.weight', 'encoder.embeddings.position_embeddings.weight': 'encoder.embed_positions.weight', 'encoder.embeddings.layerNorm.weight': 'encoder.layernorm_embedding.weight', 'encoder.embeddings.layerNorm.bias': 'encoder.layernorm_embedding.bias', 'decoder.embeddings.word_embeddings.weight': 'decoder.embed_tokens.weight', 'decoder.embeddings.position_embeddings.weight': 'decoder.embed_positions.weight', 'decoder.embeddings.layerNorm.weight': 'decoder.layernorm_embedding.weight', 'decoder.embeddings.layerNorm.bias': 'decoder.layernorm_embedding.bias'}
        for i in range(self.num_hidden_layers):
            mapping.update({f'encoder.encoderLayer.{i}.multiHeadAttention.q.weight': f'encoder.layers.{i}.self_attn.q_proj.weight', f'encoder.encoderLayer.{i}.multiHeadAttention.q.bias': f'encoder.layers.{i}.self_attn.q_proj.bias', f'encoder.encoderLayer.{i}.multiHeadAttention.k.weight': f'encoder.layers.{i}.self_attn.k_proj.weight', f'encoder.encoderLayer.{i}.multiHeadAttention.k.bias': f'encoder.layers.{i}.self_attn.k_proj.bias', f'encoder.encoderLayer.{i}.multiHeadAttention.v.weight': f'encoder.layers.{i}.self_attn.v_proj.weight', f'encoder.encoderLayer.{i}.multiHeadAttention.v.bias': f'encoder.layers.{i}.self_attn.v_proj.bias', f'encoder.encoderLayer.{i}.multiHeadAttention.o.weight': f'encoder.layers.{i}.self_attn.out_proj.weight', f'encoder.encoderLayer.{i}.multiHeadAttention.o.bias': f'encoder.layers.{i}.self_attn.out_proj.bias', f'encoder.encoderLayer.{i}.attnLayerNorm.weight': f'encoder.layers.{i}.self_attn_layer_norm.weight', f'encoder.encoderLayer.{i}.attnLayerNorm.bias': f'encoder.layers.{i}.self_attn_layer_norm.bias', f'encoder.encoderLayer.{i}.feedForward.intermediateDense.weight': f'encoder.layers.{i}.fc1.weight', f'encoder.encoderLayer.{i}.feedForward.intermediateDense.bias': f'encoder.layers.{i}.fc1.bias', f'encoder.encoderLayer.{i}.feedForward.outputDense.weight': f'encoder.layers.{i}.fc2.weight', f'encoder.encoderLayer.{i}.feedForward.outputDense.bias': f'encoder.layers.{i}.fc2.bias', f'encoder.encoderLayer.{i}.ffnLayerNorm.weight': f'encoder.layers.{i}.final_layer_norm.weight', f'encoder.encoderLayer.{i}.ffnLayerNorm.bias': f'encoder.layers.{i}.final_layer_norm.bias', f'decoder.decoderLayer.{i}.multiHeadAttention.q.weight': f'decoder.layers.{i}.self_attn.q_proj.weight', f'decoder.decoderLayer.{i}.multiHeadAttention.q.bias': f'decoder.layers.{i}.self_attn.q_proj.bias', f'decoder.decoderLayer.{i}.multiHeadAttention.k.weight': f'decoder.layers.{i}.self_attn.k_proj.weight', f'decoder.decoderLayer.{i}.multiHeadAttention.k.bias': f'decoder.layers.{i}.self_attn.k_proj.bias', f'decoder.decoderLayer.{i}.multiHeadAttention.v.weight': f'decoder.layers.{i}.self_attn.v_proj.weight', f'decoder.decoderLayer.{i}.multiHeadAttention.v.bias': f'decoder.layers.{i}.self_attn.v_proj.bias', f'decoder.decoderLayer.{i}.multiHeadAttention.o.weight': f'decoder.layers.{i}.self_attn.out_proj.weight', f'decoder.decoderLayer.{i}.multiHeadAttention.o.bias': f'decoder.layers.{i}.self_attn.out_proj.bias', f'decoder.decoderLayer.{i}.attnLayerNorm.weight': f'decoder.layers.{i}.self_attn_layer_norm.weight', f'decoder.decoderLayer.{i}.attnLayerNorm.bias': f'decoder.layers.{i}.self_attn_layer_norm.bias', f'decoder.decoderLayer.{i}.crossAttention.q.weight': f'decoder.layers.{i}.encoder_attn.q_proj.weight', f'decoder.decoderLayer.{i}.crossAttention.q.bias': f'decoder.layers.{i}.encoder_attn.q_proj.bias', f'decoder.decoderLayer.{i}.crossAttention.k.weight': f'decoder.layers.{i}.encoder_attn.k_proj.weight', f'decoder.decoderLayer.{i}.crossAttention.k.bias': f'decoder.layers.{i}.encoder_attn.k_proj.bias', f'decoder.decoderLayer.{i}.crossAttention.v.weight': f'decoder.layers.{i}.encoder_attn.v_proj.weight', f'decoder.decoderLayer.{i}.crossAttention.v.bias': f'decoder.layers.{i}.encoder_attn.v_proj.bias', f'decoder.decoderLayer.{i}.crossAttention.o.weight': f'decoder.layers.{i}.encoder_attn.out_proj.weight', f'decoder.decoderLayer.{i}.crossAttention.o.bias': f'decoder.layers.{i}.encoder_attn.out_proj.bias', f'decoder.decoderLayer.{i}.crossLayerNorm.weight': f'decoder.layers.{i}.encoder_attn_layer_norm.weight', f'decoder.decoderLayer.{i}.crossLayerNorm.bias': f'decoder.layers.{i}.encoder_attn_layer_norm.bias', f'decoder.decoderLayer.{i}.feedForward.intermediateDense.weight': f'decoder.layers.{i}.fc1.weight', f'decoder.decoderLayer.{i}.feedForward.intermediateDense.bias': f'decoder.layers.{i}.fc1.bias', f'decoder.decoderLayer.{i}.feedForward.outputDense.weight': f'decoder.layers.{i}.fc2.weight', f'decoder.decoderLayer.{i}.feedForward.outputDense.bias': f'decoder.layers.{i}.fc2.bias', f'decoder.decoderLayer.{i}.ffnLayerNorm.weight': f'decoder.layers.{i}.final_layer_norm.weight', f'decoder.decoderLayer.{i}.ffnLayerNorm.bias': f'decoder.layers.{i}.final_layer_norm.bias'})
        return mapping


CHAT_START_DOCSTRING = """
    :param checkpoint_path: str, 模型权重地址，可以是所在文件夹、文件地址、文件地址列表
    :param precision: bool, 精度, 'double', 'float', 'half', 'float16', 'bfloat16'
    :param quantization_config: dict, 模型量化使用到的参数, eg. {'quantization_method':'cpm_kernels', 'quantization_bit':8}
    :param generation_config: dict, genrerate使用到的参数, eg. {'mode':'random_sample', 'max_length':2048, 'default_rtype':'logits', 'use_states':True}
    :param create_model_at_startup: bool, 是否在启动的时候加载模型, 默认为True
    :param system: Optional[str]=None, 模型使用的system信息, 仅部分模型可用, 且openai api格式的不需要设置该参数
"""


def _is_control(char):
    """Checks whether `chars` is a control character."""
    if char == '\t' or char == '\n' or char == '\r':
        return False
    cat = unicodedata.category(char)
    if cat.startswith('C'):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    if cp >= 33 and cp <= 47 or cp >= 58 and cp <= 64 or cp >= 91 and cp <= 96 or cp >= 123 and cp <= 126:
        return True
    cat = unicodedata.category(char)
    if cat.startswith('P'):
        return True
    return False


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    if char == ' ' or char == '\t' or char == '\n' or char == '\r':
        return True
    cat = unicodedata.category(char)
    if cat == 'Zs':
        return True
    return False


def whitespace_tokenize(text):
    """去除文本中的空白符"""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True, never_split=('[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]')):
        """Constructs a BasicTokenizer.
        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text: 'str'):
        """文本切分成token"""
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))
        output_tokens = whitespace_tokenize(' '.join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize('NFD', text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == 'Mn':
                continue
            output.append(char)
        return ''.join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return [''.join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(' ')
                output.append(char)
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        if cp >= 19968 and cp <= 40959 or cp >= 13312 and cp <= 19903 or cp >= 131072 and cp <= 173791 or cp >= 173824 and cp <= 177983 or cp >= 177984 and cp <= 178207 or cp >= 178208 and cp <= 183983 or cp >= 63744 and cp <= 64255 or cp >= 194560 and cp <= 195103:
            return True
        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 65533 or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token='[UNK]', max_input_chars_per_word=100, do_tokenize_unk=False):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
        self.do_tokenize_unk = do_tokenize_unk

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token if self.do_tokenize_unk else token)
                continue
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = ''.join(chars[start:end])
                    if start > 0:
                        substr = '##' + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
            if self.do_tokenize_unk and is_bad:
                output_tokens.append(self.unk_token)
            elif not self.do_tokenize_unk and is_bad:
                output_tokens.append(substr)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

