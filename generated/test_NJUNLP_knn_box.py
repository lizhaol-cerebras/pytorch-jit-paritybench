
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


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


import torch


import math


from torch.nn.modules.loss import _Loss


from typing import Any


from typing import Dict


from typing import Optional


from torch import Tensor


from typing import Callable


from typing import Tuple


from torch import nn


from torch.nn import Parameter


import logging


from collections.abc import Iterable


import re


import itertools as it


import warnings


from collections import deque


from collections import namedtuple


import pandas as pd


from torch.utils.data import Dataset


from itertools import groupby


from torch.utils.data import DataLoader


from collections import Counter


import collections


from collections import OrderedDict


from typing import Union


from torch.serialization import default_restore_location


import inspect


from typing import List


from typing import BinaryIO


from torch.utils.data.dataloader import default_collate


import itertools


import torch.utils.data


from functools import lru_cache


import queue


import time


from enum import Enum


from typing import Sequence


from collections import defaultdict


from typing import Type


import random


from typing import Mapping


import torch.distributed as dist


from functools import partial


from functools import wraps


import copy


from typing import Iterator


import uuid


from torch.autograd import Variable


from numbers import Number


from typing import NamedTuple


import functools


from torch.nn.modules.utils import _single


from torch.autograd import Function


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.nn.modules.utils import _pair


from torch.nn.modules.conv import _ConvNd


import torch.onnx.operators


from itertools import repeat


import torch.optim


from itertools import chain


import types


import torch.optim.lr_scheduler


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


from typing import Set


from itertools import accumulate


from sklearn.decomposition import PCA


from random import randint


from random import sample


import torch.optim as optim


from torch.optim import lr_scheduler


from sklearn.cluster import Birch


from sklearn.cluster import DBSCAN


from inspect import currentframe


from inspect import getframeinfo


class LatentLayersKLLoss(_Loss):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, layer_samples, lang_idx, update_num, sample_size):
        prior = self.args.prior
        samples = layer_samples[lang_idx]
        eps = 1e-07
        if prior == 'uniform':
            kl_loss = (samples * (torch.log(samples + eps) - math.log(0.5))).sum(-1)
        elif prior == 'agged_posterior':
            y_t = torch.stack([x.detach() for x in layer_samples], dim=0)
            agged_q = torch.sum(y_t, dim=0)
            row_norm = agged_q.sum(-1)
            normed_agg_q = agged_q / row_norm
            kl_loss = (samples * (torch.log(samples + eps) - torch.log(normed_agg_q + eps))).sum(-1)
        else:
            raise NotImplementedError('The specified prior is not implemented.')
        kl_loss /= layer_samples[0].size()[0]
        kl_weight = min(self.args.sparsity_weight, (update_num - self.args.soft_update) * self.args.sparsity_weight / self.args.anneal_updates)
        kl_loss *= kl_weight * sample_size
        return kl_loss


class LatentLayersSparsityLoss(_Loss):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def is_valid(self, update_num):
        if self.args.target_layers <= 0:
            return False
        return update_num > self.args.soft_update + self.args.anneal_updates

    def forward(self, layer_samples_list, update_num, sample_size):
        batch_loss = 0
        share_loss = 0
        global_sparsity_loss = 0
        layer_samples = torch.stack(layer_samples_list, dim=0)
        if (self.args.target_layers > 0 or self.args.share_weight > 0) and update_num > self.args.soft_update + self.args.anneal_updates:
            if update_num < self.args.anneal_updates + self.args.soft_update:
                weight_anneal = 0
            elif update_num < 2 * self.args.anneal_updates + self.args.soft_update:
                weight_anneal = (update_num - self.args.soft_update - self.args.anneal_updates) * self.args.share_weight / self.args.anneal_updates
            else:
                weight_anneal = 1
            layer_utilization = torch.sum(layer_samples, dim=0)
            layer_utilization /= layer_samples.size()[0]
            if self.args.share_weight > 0:
                share_loss = sum(-1.0 * v * math.log(v) for v in layer_utilization if v > 0)
                batch_loss += weight_anneal * self.args.share_weight * sample_size * share_loss
            if self.args.target_layers > 0:
                expeted_layers = sum(layer_utilization)
                global_sparsity_loss = (expeted_layers - self.args.target_layers) ** 2
                batch_loss += weight_anneal * self.args.share_weight * sample_size * global_sparsity_loss
        return batch_loss


class LayerSelect(nn.Module):
    """Compute samples (from a Gumbel-Sigmoid distribution) which is used as
    either (soft) weighting or (hard) selection of residual connection.
    https://arxiv.org/abs/2009.13102
    """

    def __init__(self, num_layers, num_logits, args):
        super(LayerSelect, self).__init__()
        self.args = args
        self.layer_logits = torch.nn.Parameter(torch.Tensor(num_logits, num_layers), requires_grad=True)
        self.hard_select = not (hasattr(args, 'soft_select') and args.soft_select)
        self.tau = getattr(args, 'sampling_tau', 5)
        self.detach_grad = False
        self.layer_samples = [None] * num_logits

    @staticmethod
    def add_args(parser):
        parser.add_argument('--soft-select', action='store_true', help='use soft samples in training an inference')
        parser.add_argument('--sampling-tau', type=float, help='sampling temperature')

    def sample(self, logit_idx):
        """To leverage the efficiency of distributed training, samples for all
        layers are computed at once for each logit_idx. Logits are parameters
        learnt independent of each other.

        Args:
            logit_idx: The index of logit parameters used for sampling.
        """
        assert logit_idx is not None
        self.samples = self._gumbel_sigmoid(self.layer_logits[logit_idx, :].detach() if self.detach_grad else self.layer_logits[logit_idx, :], dim=-1, tau=self.tau, hard=self.hard_select)
        self.layer_samples[logit_idx] = self.samples

    def forward(self, i):
        sample = self.samples[i]
        return sample

    def _gumbel_sigmoid(self, logits, tau=1, hard=False, eps=1e-10, dim=-1, threshold=0.5):
        gumbels1 = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        gumbels2 = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        gumbels1 = (logits + gumbels1 - gumbels2) / tau
        y_soft = gumbels1.sigmoid()
        if hard:
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).masked_fill(y_soft > threshold, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        return ret


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


class FairseqIncrementalState(object):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_incremental_state()

    def init_incremental_state(self):
        self._incremental_state_id = str(uuid.uuid4())

    def _get_full_incremental_state_key(self, key: 'str') ->str:
        return '{}.{}'.format(self._incremental_state_id, key)

    def get_incremental_state(self, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]', key: 'str') ->Optional[Dict[str, Optional[Tensor]]]:
        """Helper for getting incremental state for an nn.Module."""
        full_key = self._get_full_incremental_state_key(key)
        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(self, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]', key: 'str', value: 'Dict[str, Optional[Tensor]]') ->Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
        """Helper for setting incremental state for an nn.Module."""
        if incremental_state is not None:
            full_key = self._get_full_incremental_state_key(key)
            incremental_state[full_key] = value
        return incremental_state


def with_incremental_state(cls):
    cls.__bases__ = (FairseqIncrementalState,) + tuple(b for b in cls.__bases__ if b != FairseqIncrementalState)
    return cls


@with_incremental_state
class MultiheadLinearAttention(nn.Module):
    """Multi-headed linformer attention.

    Projects the key and values down to the compressed dimension, before computing self-attention.

    See "Linformer: Self-Attention with Linear Complexity" for more details.
    """

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, self_attention=False, encoder_decoder_attention=False, q_noise=0.0, qn_block_size=8, compressed=1, max_seq_len=256, shared_kv_compressed=0, shared_compress_layer=None, freeze_compress=0):
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
        self.k_proj = quant_noise(nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.v_proj = quant_noise(nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.q_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        if shared_compress_layer is None:
            self.compress_seq_len = max_seq_len // compressed
            self.compress_k = nn.Linear(max_seq_len, self.compress_seq_len, bias=False)
            if shared_kv_compressed == 0:
                self.compress_v = nn.Linear(max_seq_len, self.compress_seq_len, bias=False)
            self.layerwise_sharing = False
        else:
            self.compress_k = shared_compress_layer
            if shared_kv_compressed == 0:
                self.compress_v = shared_compress_layer
            self.layerwise_sharing = True
        self.shared_kv_compressed = shared_kv_compressed
        self.out_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        self.add_zero_attn = add_zero_attn
        self.reset_parameters()
        if freeze_compress == 1:
            self.compress_k.weight.requires_grad = False
            if shared_kv_compressed == 0:
                self.compress_v.weight.requires_grad = False
        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
            if not self.layerwise_sharing:
                nn.init.xavier_uniform_(self.compress_k.weight, gain=1 / math.sqrt(2))
                if self.shared_kv_compressed == 0:
                    nn.init.xavier_uniform_(self.compress_v.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)
            if not self.layerwise_sharing:
                nn.init.xavier_uniform_(self.compress_k.weight)
                if self.shared_kv_compressed == 0:
                    nn.init.xavier_uniform_(self.compress_v.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key: 'Optional[Tensor]', value: 'Optional[Tensor]', key_padding_mask: 'Optional[Tensor]'=None, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]'=None, need_weights: 'bool'=True, static_kv: 'bool'=False, attn_mask: 'Optional[Tensor]'=None, before_softmax: 'bool'=False, need_head_weights: 'bool'=False) ->Tuple[Tensor, Optional[Tensor]]:
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
            k_input = query.permute(1, 2, 0).contiguous()
            k_input = F.linear(k_input, self.compress_k.weight[:, 0:tgt_len]).permute(2, 0, 1).contiguous()
            k = self.k_proj(k_input)
            v_input = query.permute(1, 2, 0).contiguous()
            if self.shared_kv_compressed == 0:
                v_input = F.linear(v_input, self.compress_v.weight[:, 0:tgt_len]).permute(2, 0, 1).contiguous()
            if self.shared_kv_compressed == 1:
                v_input = F.linear(v_input, self.compress_k.weight[:, 0:tgt_len]).permute(2, 0, 1).contiguous()
            v = self.v_proj(v_input)
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
                _prev_key = saved_state['prev_key']
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
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
            key_padding_mask = MultiheadLinearAttention._append_prev_key_padding_mask(key_padding_mask=key_padding_mask, prev_key_padding_mask=prev_key_padding_mask, batch_size=bsz, src_len=k.size(1), static_kv=static_kv)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_key_padding_mask'] = key_padding_mask
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        src_len = k.size(1)
        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = MultiheadLinearAttention.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask
        if before_softmax:
            return attn_weights, v
        attn_weights_float = utils.softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: 'Optional[Tensor]' = None
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                attn_weights = attn_weights.mean(dim=0)
        return attn, attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(key_padding_mask: 'Optional[Tensor]', prev_key_padding_mask: 'Optional[Tensor]', batch_size: 'int', src_len: 'int', static_kv: 'bool') ->Optional[Tensor]:
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), key_padding_mask.float()], dim=1)
        elif prev_key_padding_mask is not None:
            filler = torch.zeros((batch_size, src_len - prev_key_padding_mask.size(1)), device=prev_key_padding_mask.device)
            new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), filler.float()], dim=1)
        elif key_padding_mask is not None:
            filler = torch.zeros((batch_size, src_len - key_padding_mask.size(1)), device=key_padding_mask.device)
            new_key_padding_mask = torch.cat([filler.float(), key_padding_mask.float()], dim=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(self, incremental_state: 'Dict[str, Dict[str, Optional[Tensor]]]', new_order: 'Tensor'):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(0) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(self, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]') ->Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, 'attn_state')
        if result is not None:
            return result
        else:
            empty_result: 'Dict[str, Optional[Tensor]]' = {}
            return empty_result

    def _set_input_buffer(self, incremental_state: 'Dict[str, Dict[str, Optional[Tensor]]]', buffer: 'Dict[str, Optional[Tensor]]'):
        return self.set_incremental_state(incremental_state, 'attn_state', buffer)

    def apply_sparse_mask(attn_weights, tgt_len: 'int', src_len: 'int', bsz: 'int'):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != '' else ''
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + 'in_proj_weight'):
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + 'q_proj.weight'] = state_dict[k][:dim]
                items_to_add[prefix + 'k_proj.weight'] = state_dict[k][dim:2 * dim]
                items_to_add[prefix + 'v_proj.weight'] = state_dict[k][2 * dim:]
                keys_to_remove.append(k)
                k_bias = prefix + 'in_proj_bias'
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + 'q_proj.bias'] = state_dict[k_bias][:dim]
                    items_to_add[prefix + 'k_proj.bias'] = state_dict[k_bias][dim:2 * dim]
                    items_to_add[prefix + 'v_proj.bias'] = state_dict[k_bias][2 * dim:]
                    keys_to_remove.append(prefix + 'in_proj_bias')
        for k in keys_to_remove:
            del state_dict[k]
        for key, value in items_to_add.items():
            state_dict[key] = value


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def safe_cumprod(tensor, dim: 'int', eps: 'float'=1e-10):
    """
    An implementation of cumprod to prevent precision issue.
    cumprod(x)
    = [x1, x1x2, x1x2x3, ....]
    = [exp(log(x1)), exp(log(x1) + log(x2)), exp(log(x1) + log(x2) + log(x3)), ...]
    = exp(cumsum(log(x)))
    """
    if (tensor + eps < 0).any().item():
        raise RuntimeError('Safe cumprod can only take non-negative tensors as input.Consider use torch.cumprod if you want to calculate negative values.')
    log_tensor = torch.log(tensor + eps)
    cumsum_log_tensor = torch.cumsum(log_tensor, dim)
    exp_cumsum_log_tensor = torch.exp(cumsum_log_tensor)
    return exp_cumsum_log_tensor


def exclusive_cumprod(tensor, dim: 'int', eps: 'float'=1e-10):
    """
    Implementing exclusive cumprod.
    There is cumprod in pytorch, however there is no exclusive mode.
    cumprod(x) = [x1, x1x2, x2x3x4, ..., prod_{i=1}^n x_i]
    exclusive means cumprod(x) = [1, x1, x1x2, x1x2x3, ..., prod_{i=1}^{n-1} x_i]
    """
    tensor_size = list(tensor.size())
    tensor_size[dim] = 1
    return_tensor = safe_cumprod(torch.cat([torch.ones(tensor_size).type_as(tensor), tensor], dim=dim), dim=dim, eps=eps)
    if dim == 0:
        return return_tensor[:-1]
    elif dim == 1:
        return return_tensor[:, :-1]
    elif dim == 2:
        return return_tensor[:, :, :-1]
    else:
        raise RuntimeError('Cumprod on dimension 3 and more is not implemented')


@with_incremental_state
class MonotonicAttention(nn.Module):
    """
    Abstract class of monotonic attentions
    """

    def __init__(self, args):
        self.eps = args.attention_eps
        self.mass_preservation = args.mass_preservation
        self.noise_mean = args.noise_mean
        self.noise_var = args.noise_var
        self.energy_bias_init = args.energy_bias_init
        self.energy_bias = nn.Parameter(self.energy_bias_init * torch.ones([1])) if args.energy_bias is True else 0

    @staticmethod
    def add_args(parser):
        parser.add_argument('--no-mass-preservation', action='store_false', dest='mass_preservation', help='Do not stay on the last token when decoding')
        parser.add_argument('--mass-preservation', action='store_true', dest='mass_preservation', help='Stay on the last token when decoding')
        parser.set_defaults(mass_preservation=True)
        parser.add_argument('--noise-var', type=float, default=1.0, help='Variance of discretness noise')
        parser.add_argument('--noise-mean', type=float, default=0.0, help='Mean of discretness noise')
        parser.add_argument('--energy-bias', action='store_true', default=False, help='Bias for energy')
        parser.add_argument('--energy-bias-init', type=float, default=-2.0, help='Initial value of the bias for energy')
        parser.add_argument('--attention-eps', type=float, default=1e-06, help='Epsilon when calculating expected attention')

    def p_choose(self, *args):
        raise NotImplementedError

    def input_projections(self, *args):
        raise NotImplementedError

    def attn_energy(self, q_proj, k_proj, key_padding_mask=None):
        """
        Calculating monotonic energies

        ============================================================
        Expected input size
        q_proj: bsz * num_heads, tgt_len, self.head_dim
        k_proj: bsz * num_heads, src_len, self.head_dim
        key_padding_mask: bsz, src_len
        attn_mask: tgt_len, src_len
        """
        bsz, tgt_len, embed_dim = q_proj.size()
        bsz = bsz // self.num_heads
        src_len = k_proj.size(1)
        attn_energy = torch.bmm(q_proj, k_proj.transpose(1, 2)) + self.energy_bias
        attn_energy = attn_energy.view(bsz, self.num_heads, tgt_len, src_len)
        if key_padding_mask is not None:
            attn_energy = attn_energy.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2).bool(), float('-inf'))
        return attn_energy

    def expected_alignment_train(self, p_choose, key_padding_mask):
        """
        Calculating expected alignment for MMA
        Mask is not need because p_choose will be 0 if masked

        q_ij = (1 − p_{ij−1})q_{ij−1} + a+{i−1j}
        a_ij = p_ij q_ij

        parellel solution:
        ai = p_i * cumprod(1 − pi) * cumsum(a_i / cumprod(1 − pi))

        ============================================================
        Expected input size
        p_choose: bsz * num_heads, tgt_len, src_len
        """
        bsz_num_heads, tgt_len, src_len = p_choose.size()
        cumprod_1mp = exclusive_cumprod(1 - p_choose, dim=2, eps=self.eps)
        cumprod_1mp_clamp = torch.clamp(cumprod_1mp, self.eps, 1.0)
        init_attention = p_choose.new_zeros([bsz_num_heads, 1, src_len])
        init_attention[:, :, 0] = 1.0
        previous_attn = [init_attention]
        for i in range(tgt_len):
            alpha_i = (p_choose[:, i] * cumprod_1mp[:, i] * torch.cumsum(previous_attn[i][:, 0] / cumprod_1mp_clamp[:, i], dim=1)).clamp(0, 1.0)
            previous_attn.append(alpha_i.unsqueeze(1))
        alpha = torch.cat(previous_attn[1:], dim=1)
        if self.mass_preservation:
            alpha[:, :, -1] = 1 - alpha[:, :, :-1].sum(dim=-1).clamp(0.0, 1.0)
        assert not torch.isnan(alpha).any(), 'NaN detected in alpha.'
        return alpha

    def expected_alignment_infer(self, p_choose, key_padding_mask, incremental_state):
        """
        Calculating mo alignment for MMA during inference time

        ============================================================
        Expected input size
        p_choose: bsz * num_heads, tgt_len, src_len
        key_padding_mask: bsz * src_len
        incremental_state: dict
        """
        bsz_num_heads, tgt_len, src_len = p_choose.size()
        assert tgt_len == 1
        p_choose = p_choose[:, 0, :]
        monotonic_cache = self._get_monotonic_buffer(incremental_state)
        bsz = bsz_num_heads // self.num_heads
        prev_monotonic_step = monotonic_cache.get('step', p_choose.new_zeros([bsz, self.num_heads]).long())
        bsz, num_heads = prev_monotonic_step.size()
        assert num_heads == self.num_heads
        assert bsz * num_heads == bsz_num_heads
        p_choose = p_choose.view(bsz, num_heads, src_len)
        if key_padding_mask is not None:
            src_lengths = src_len - key_padding_mask.sum(dim=1, keepdim=True).long()
        else:
            src_lengths = prev_monotonic_step.new_ones(bsz, 1) * src_len
        src_lengths = src_lengths.expand_as(prev_monotonic_step)
        new_monotonic_step = prev_monotonic_step
        step_offset = 0
        if key_padding_mask is not None:
            if key_padding_mask[:, 0].any():
                step_offset = key_padding_mask.sum(dim=-1, keepdim=True)
        max_steps = src_lengths - 1 if self.mass_preservation else src_lengths
        finish_read = new_monotonic_step.eq(max_steps)
        while finish_read.sum().item() < bsz * self.num_heads:
            p_choose_i = p_choose.gather(2, (step_offset + new_monotonic_step).unsqueeze(2).clamp(0, src_len - 1)).squeeze(2)
            action = (p_choose_i < 0.5).type_as(prev_monotonic_step).masked_fill(finish_read, 0)
            new_monotonic_step += action
            finish_read = new_monotonic_step.eq(max_steps) | (action == 0)
        monotonic_cache['step'] = new_monotonic_step
        alpha = p_choose.new_zeros([bsz * self.num_heads, src_len]).scatter(1, (step_offset + new_monotonic_step).view(bsz * self.num_heads, 1).clamp(0, src_len - 1), 1)
        if not self.mass_preservation:
            alpha = alpha.masked_fill((new_monotonic_step == max_steps).view(bsz * self.num_heads, 1), 0)
        alpha = alpha.unsqueeze(1)
        self._set_monotonic_buffer(incremental_state, monotonic_cache)
        return alpha

    def v_proj_output(self, value):
        raise NotImplementedError

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None, *args, **kwargs):
        tgt_len, bsz, embed_dim = query.size()
        src_len = value.size(0)
        p_choose = self.p_choose(query, key, key_padding_mask)
        if incremental_state is not None:
            alpha = self.expected_alignment_infer(p_choose, key_padding_mask, incremental_state)
        else:
            alpha = self.expected_alignment_train(p_choose, key_padding_mask)
        beta = self.expected_attention(alpha, query, key, value, key_padding_mask, incremental_state)
        attn_weights = beta
        v_proj = self.v_proj_output(value)
        attn = torch.bmm(attn_weights.type_as(v_proj), v_proj)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        beta = beta.view(bsz, self.num_heads, tgt_len, src_len)
        alpha = alpha.view(bsz, self.num_heads, tgt_len, src_len)
        p_choose = p_choose.view(bsz, self.num_heads, tgt_len, src_len)
        return attn, {'alpha': alpha, 'beta': beta, 'p_choose': p_choose}

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        super().reorder_incremental_state(incremental_state, new_order)
        input_buffer = self._get_monotonic_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_monotonic_buffer(incremental_state, input_buffer)

    def _get_monotonic_buffer(self, incremental_state):
        return utils.get_incremental_state(self, incremental_state, 'monotonic') or {}

    def _set_monotonic_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(self, incremental_state, 'monotonic', buffer)

    def get_pointer(self, incremental_state):
        return utils.get_incremental_state(self, incremental_state, 'monotonic') or {}

    def get_fastest_pointer(self, incremental_state):
        return self.get_pointer(incremental_state)['step'].max(0)[0]

    def set_pointer(self, incremental_state, p_choose):
        curr_pointer = self.get_pointer(incremental_state)
        if len(curr_pointer) == 0:
            buffer = torch.zeros_like(p_choose)
        else:
            buffer = self.get_pointer(incremental_state)['step']
        buffer += (p_choose < 0.5).type_as(buffer)
        utils.set_incremental_state(self, incremental_state, 'monotonic', {'step': buffer})


logger = logging.getLogger('fairseq_cli.validate')


class FairseqDropout(nn.Module):

    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x, inplace: 'bool'=False):
        if self.training or self.apply_during_inference:
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x

    def make_generation_fast_(self, name: 'str', retain_dropout: 'bool'=False, retain_dropout_modules: 'Optional[List[str]]'=None, **kwargs):
        if retain_dropout:
            if retain_dropout_modules is not None and self.module_name is None:
                logger.warning('Cannot enable dropout during inference for module {} because module_name was not set'.format(name))
            elif retain_dropout_modules is None or self.module_name in retain_dropout_modules:
                logger.info('Enabling dropout during inference for module: {}'.format(name))
                self.apply_during_inference = True
            else:
                logger.info('Disabling dropout for module: {}'.format(name))


@with_incremental_state
class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, self_attention=False, encoder_decoder_attention=False, q_noise=0.0, qn_block_size=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.scaling = self.head_dim ** -0.5
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert not self.self_attention or self.qkv_same_dim, 'Self-attention requires query, key and value to be of the same size'
        self.k_proj = quant_noise(nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.v_proj = quant_noise(nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.q_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.out_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        self.add_zero_attn = add_zero_attn
        self.reset_parameters()
        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

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

    def forward(self, query, key: 'Optional[Tensor]', value: 'Optional[Tensor]', key_padding_mask: 'Optional[Tensor]'=None, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]'=None, need_weights: 'bool'=True, static_kv: 'bool'=False, attn_mask: 'Optional[Tensor]'=None, before_softmax: 'bool'=False, need_head_weights: 'bool'=False) ->Tuple[Tensor, Optional[Tensor]]:
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
        if not self.onnx_trace and not self.tpu and incremental_state is None and not static_kv and not torch.jit.is_scripting():
            assert key is not None and value is not None
            return F.multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, torch.empty([0]), torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)), self.bias_k, self.bias_v, self.add_zero_attn, self.dropout_module.p, self.out_proj.weight, self.out_proj.bias, self.training or self.dropout_module.apply_during_inference, key_padding_mask, need_weights, attn_mask, use_separate_proj_weight=True, q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight)
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
                _prev_key = saved_state['prev_key']
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
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
        src_len = k.size(1)
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
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not self.tpu:
                attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if before_softmax:
            return attn_weights, v
        attn_weights_float = utils.softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)
        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: 'Optional[Tensor]' = None
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                attn_weights = attn_weights.mean(dim=0)
        return attn, attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(key_padding_mask: 'Optional[Tensor]', prev_key_padding_mask: 'Optional[Tensor]', batch_size: 'int', src_len: 'int', static_kv: 'bool') ->Optional[Tensor]:
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), key_padding_mask.float()], dim=1)
        elif prev_key_padding_mask is not None:
            filler = torch.zeros((batch_size, src_len - prev_key_padding_mask.size(1)), device=prev_key_padding_mask.device)
            new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), filler.float()], dim=1)
        elif key_padding_mask is not None:
            filler = torch.zeros((batch_size, src_len - key_padding_mask.size(1)), device=key_padding_mask.device)
            new_key_padding_mask = torch.cat([filler.float(), key_padding_mask.float()], dim=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(self, incremental_state: 'Dict[str, Dict[str, Optional[Tensor]]]', new_order: 'Tensor'):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(0) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

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

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != '' else ''
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + 'in_proj_weight'):
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + 'q_proj.weight'] = state_dict[k][:dim]
                items_to_add[prefix + 'k_proj.weight'] = state_dict[k][dim:2 * dim]
                items_to_add[prefix + 'v_proj.weight'] = state_dict[k][2 * dim:]
                keys_to_remove.append(k)
                k_bias = prefix + 'in_proj_bias'
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + 'q_proj.bias'] = state_dict[k_bias][:dim]
                    items_to_add[prefix + 'k_proj.bias'] = state_dict[k_bias][dim:2 * dim]
                    items_to_add[prefix + 'v_proj.bias'] = state_dict[k_bias][2 * dim:]
                    keys_to_remove.append(prefix + 'in_proj_bias')
        for k in keys_to_remove:
            del state_dict[k]
        for key, value in items_to_add.items():
            state_dict[key] = value


class MonotonicMultiheadAttentionHard(MonotonicAttention, MultiheadAttention):

    def __init__(self, args):
        MultiheadAttention.__init__(self, embed_dim=args.decoder_embed_dim, num_heads=args.decoder_attention_heads, kdim=getattr(args, 'encoder_embed_dim', None), vdim=getattr(args, 'encoder_embed_dim', None), dropout=args.attention_dropout, encoder_decoder_attention=True)
        MonotonicAttention.__init__(self, args)
        self.k_in_proj = {'monotonic': self.k_proj}
        self.q_in_proj = {'monotonic': self.q_proj}
        self.v_in_proj = {'output': self.v_proj}

    def input_projections(self, query, key, value, name):
        """
        Prepare inputs for multihead attention

        ============================================================
        Expected input size
        query: tgt_len, bsz, embed_dim
        key: src_len, bsz, embed_dim
        value: src_len, bsz, embed_dim
        name: monotonic or soft
        """
        if query is not None:
            bsz = query.size(1)
            q = self.q_in_proj[name](query)
            q *= self.scaling
            q = q.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        else:
            q = None
        if key is not None:
            bsz = key.size(1)
            k = self.k_in_proj[name](key)
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        else:
            k = None
        if value is not None:
            bsz = value.size(1)
            v = self.v_in_proj[name](value)
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        else:
            v = None
        return q, k, v

    def p_choose(self, query, key, key_padding_mask=None):
        """
        Calculating step wise prob for reading and writing
        1 to read, 0 to write

        ============================================================
        Expected input size
        query: bsz, tgt_len, embed_dim
        key: bsz, src_len, embed_dim
        value: bsz, src_len, embed_dim
        key_padding_mask: bsz, src_len
        attn_mask: bsz, src_len
        query: bsz, tgt_len, embed_dim
        """
        q_proj, k_proj, _ = self.input_projections(query, key, None, 'monotonic')
        attn_energy = self.attn_energy(q_proj, k_proj, key_padding_mask)
        noise = 0
        if self.training:
            noise = torch.normal(self.noise_mean, self.noise_var, attn_energy.size()).type_as(attn_energy)
        p_choose = torch.sigmoid(attn_energy + noise)
        _, _, tgt_len, src_len = p_choose.size()
        return p_choose.view(-1, tgt_len, src_len)

    def expected_attention(self, alpha, *args):
        """
        For MMA-H, beta = alpha
        """
        return alpha

    def v_proj_output(self, value):
        _, _, v_proj = self.input_projections(None, None, value, 'output')
        return v_proj


def lengths_to_padding_mask(lens: 'torch.LongTensor') ->torch.BoolTensor:
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask


def lengths_to_mask(lens: 'torch.LongTensor') ->torch.BoolTensor:
    return ~lengths_to_padding_mask(lens)


class MonotonicMultiheadAttentionInfiniteLookback(MonotonicMultiheadAttentionHard):

    def __init__(self, args):
        super().__init__(args)
        self.init_soft_attention()

    def init_soft_attention(self):
        self.k_proj_soft = nn.Linear(self.kdim, self.embed_dim, bias=True)
        self.q_proj_soft = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_in_proj['soft'] = self.k_proj_soft
        self.q_in_proj['soft'] = self.q_proj_soft
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_in_proj['soft'].weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_in_proj['soft'].weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_in_proj['soft'].weight)
            nn.init.xavier_uniform_(self.q_in_proj['soft'].weight)

    def expected_attention(self, alpha, query, key, value, key_padding_mask, incremental_state):
        bsz_x_num_heads, tgt_len, src_len = alpha.size()
        bsz = int(bsz_x_num_heads / self.num_heads)
        q, k, _ = self.input_projections(query, key, None, 'soft')
        soft_energy = self.attn_energy(q, k, key_padding_mask)
        assert list(soft_energy.size()) == [bsz, self.num_heads, tgt_len, src_len]
        soft_energy = soft_energy.view(bsz * self.num_heads, tgt_len, src_len)
        if incremental_state is not None:
            monotonic_cache = self._get_monotonic_buffer(incremental_state)
            monotonic_step = monotonic_cache['step'] + 1
            step_offset = 0
            if key_padding_mask is not None:
                if key_padding_mask[:, 0].any():
                    step_offset = key_padding_mask.sum(dim=-1, keepdim=True)
            monotonic_step += step_offset
            mask = lengths_to_mask(monotonic_step.view(-1), soft_energy.size(2), 1).unsqueeze(1)
            soft_energy = soft_energy.masked_fill(~mask.bool(), float('-inf'))
            soft_energy = soft_energy - soft_energy.max(dim=2, keepdim=True)[0]
            exp_soft_energy = torch.exp(soft_energy)
            exp_soft_energy_sum = exp_soft_energy.sum(dim=2)
            beta = exp_soft_energy / exp_soft_energy_sum.unsqueeze(2)
        else:
            soft_energy = soft_energy - soft_energy.max(dim=2, keepdim=True)[0]
            exp_soft_energy = torch.exp(soft_energy)
            exp_soft_energy_cumsum = torch.cumsum(exp_soft_energy, dim=2)
            if key_padding_mask is not None:
                if key_padding_mask.any():
                    exp_soft_energy_cumsum = exp_soft_energy_cumsum.view(-1, self.num_heads, tgt_len, src_len).masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(1), self.eps).view(-1, tgt_len, src_len)
            inner_items = alpha / exp_soft_energy_cumsum
            beta = exp_soft_energy * torch.cumsum(inner_items.flip(dims=[2]), dim=2).flip(dims=[2])
            beta = self.dropout_module(beta)
        assert not torch.isnan(beta).any(), 'NaN detected in beta.'
        return beta


def convert_padding_direction(src_tokens, padding_idx, right_to_left: 'bool'=False, left_to_right: 'bool'=False):
    assert right_to_left ^ left_to_right
    pad_mask = src_tokens.eq(padding_idx)
    if not pad_mask.any():
        return src_tokens
    if left_to_right and not pad_mask[:, 0].any():
        return src_tokens
    if right_to_left and not pad_mask[:, -1].any():
        return src_tokens
    max_len = src_tokens.size(1)
    buffered = torch.empty(0).long()
    if max_len > 0:
        torch.arange(max_len, out=buffered)
    range = buffered.type_as(src_tokens).expand_as(src_tokens)
    num_pads = pad_mask.long().sum(dim=1, keepdim=True)
    if right_to_left:
        index = torch.remainder(range - num_pads, max_len)
    else:
        index = torch.remainder(range + num_pads, max_len)
    return src_tokens.gather(1, index)


class MonotonicMultiheadAttentionWaitk(MonotonicMultiheadAttentionInfiniteLookback):

    def __init__(self, args):
        super().__init__(args)
        self.q_in_proj['soft'] = self.q_in_proj['monotonic']
        self.k_in_proj['soft'] = self.k_in_proj['monotonic']
        self.waitk_lagging = args.waitk_lagging
        assert self.waitk_lagging > 0, f'Lagging has to been larger than 0, get {self.waitk_lagging}.'

    @staticmethod
    def add_args(parser):
        super(MonotonicMultiheadAttentionWaitk, MonotonicMultiheadAttentionWaitk).add_args(parser)
        parser.add_argument('--waitk-lagging', type=int, required=True, help='Wait k lagging')

    def p_choose(self, query, key, key_padding_mask=None, attn_mask=None, incremental_state=None):
        """
        query: bsz, tgt_len
        key: bsz, src_len
        key_padding_mask: bsz, src_len
        """
        src_len, bsz, _ = key.size()
        tgt_len, bsz, _ = query.size()
        p_choose = query.new_ones(bsz, tgt_len, src_len)
        p_choose = torch.tril(p_choose, diagonal=self.waitk_lagging - 1)
        p_choose = torch.triu(p_choose, diagonal=self.waitk_lagging - 1)
        if key_padding_mask is not None and key_padding_mask[:, 0].eq(1).any():
            p_choose = p_choose.masked_fill(key_padding_mask.float().flip(1).unsqueeze(1).bool(), -1)
            p_choose = convert_padding_direction(p_choose.view(-1, src_len).long(), padding_idx=-1, right_to_left=True)
            p_choose = p_choose.view(bsz, tgt_len, src_len).type_as(query)
            p_choose[p_choose.eq(-1)] = 0
        p_choose = p_choose.contiguous().unsqueeze(1).expand(-1, self.num_heads, -1, -1).contiguous().view(-1, tgt_len, src_len)
        return p_choose


class MeanPoolGatingNetwork(torch.nn.Module):
    """A simple mean-pooling gating network for selecting experts.

    This module applies mean pooling over an encoder's output and returns
    reponsibilities for each expert. The encoder format is expected to match
    :class:`fairseq.models.transformer.TransformerEncoder`.
    """

    def __init__(self, embed_dim, num_experts, dropout=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.fc1 = torch.nn.Linear(embed_dim, embed_dim)
        self.dropout = torch.nn.Dropout(dropout) if dropout is not None else None
        self.fc2 = torch.nn.Linear(embed_dim, num_experts)

    def forward(self, encoder_out):
        if not (hasattr(encoder_out, 'encoder_out') and hasattr(encoder_out, 'encoder_padding_mask') and encoder_out.encoder_out.size(2) == self.embed_dim):
            raise ValueError('Unexpected format for encoder_out')
        encoder_padding_mask = encoder_out.encoder_padding_mask
        encoder_out = encoder_out.encoder_out.transpose(0, 1)
        if encoder_padding_mask is not None:
            encoder_out = encoder_out.clone()
            encoder_out[encoder_padding_mask] = 0
            ntokens = torch.sum(~encoder_padding_mask, dim=1, keepdim=True)
            x = torch.sum(encoder_out, dim=1) / ntokens.type_as(encoder_out)
        else:
            x = torch.mean(encoder_out, dim=1)
        x = torch.tanh(self.fc1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1, dtype=torch.float32).type_as(x)


def eval_str_list(x, type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(type, x))
    except TypeError:
        return [type(x)]


def gen_parser_from_dataclass(parser: 'ArgumentParser', dataclass_instance: 'FairseqDataclass', delete_default: 'bool'=False) ->None:
    """convert a dataclass instance to tailing parser arguments"""
    import re

    def argparse_name(name: 'str'):
        if name == 'data':
            return name
        if name == '_name':
            return None
        return '--' + name.replace('_', '-')

    def interpret_dc_type(field_type):
        if isinstance(field_type, str):
            raise RuntimeError()
        typestring = str(field_type)
        if re.match('(typing.|^)Union\\[(.*), NoneType\\]$', typestring):
            return field_type.__args__[0]
        return field_type

    def get_kwargs_from_dc(dataclass_instance: 'FairseqDataclass', k: 'str') ->Dict[str, Any]:
        """k: dataclass attributes"""
        field_type = dataclass_instance._get_type(k)
        inter_type = interpret_dc_type(field_type)
        if isinstance(inter_type, type) and issubclass(inter_type, List):
            field_default = dataclass_instance._get_default_factory(k)
        else:
            field_default = dataclass_instance._get_default(k)
        if isinstance(inter_type, type) and issubclass(inter_type, Enum):
            field_choices = [t.value for t in list(inter_type)]
        else:
            field_choices = None
        field_help = dataclass_instance._get_help(k)
        field_const = dataclass_instance._get_argparse_const(k)
        kwargs = {}
        if isinstance(field_default, str) and field_default.startswith('${'):
            kwargs['default'] = field_default
        else:
            if field_default is MISSING:
                kwargs['required'] = True
            if field_choices is not None:
                kwargs['choices'] = field_choices
            if isinstance(inter_type, type) and issubclass(inter_type, List) or 'List' in str(inter_type):
                if 'int' in str(inter_type):
                    kwargs['type'] = lambda x: eval_str_list(x, int)
                elif 'float' in str(inter_type):
                    kwargs['type'] = lambda x: eval_str_list(x, float)
                elif 'str' in str(inter_type):
                    kwargs['type'] = lambda x: eval_str_list(x, str)
                else:
                    raise NotImplementedError()
                if field_default is not MISSING:
                    kwargs['default'] = ','.join(map(str, field_default))
            elif isinstance(inter_type, type) and issubclass(inter_type, Enum) or 'Enum' in str(inter_type):
                kwargs['type'] = str
                if field_default is not MISSING:
                    if isinstance(field_default, Enum):
                        kwargs['default'] = field_default.value
                    else:
                        kwargs['default'] = field_default
            elif inter_type is bool:
                kwargs['action'] = 'store_false' if field_default is True else 'store_true'
                kwargs['default'] = field_default
            else:
                kwargs['type'] = inter_type
                if field_default is not MISSING:
                    kwargs['default'] = field_default
        kwargs['help'] = field_help
        if field_const is not None:
            kwargs['const'] = field_const
            kwargs['nargs'] = '?'
        return kwargs
    for k in dataclass_instance._get_all_attributes():
        field_name = argparse_name(dataclass_instance._get_name(k))
        if field_name is None:
            continue
        kwargs = get_kwargs_from_dc(dataclass_instance, k)
        field_args = [field_name]
        alias = dataclass_instance._get_argparse_alias(k)
        if alias is not None:
            field_args.append(alias)
        if 'default' in kwargs:
            if isinstance(kwargs['default'], str) and kwargs['default'].startswith('${'):
                if kwargs['help'] is None:
                    continue
                else:
                    del kwargs['default']
            if delete_default:
                del kwargs['default']
        try:
            parser.add_argument(*field_args, **kwargs)
        except ArgumentError:
            pass


def prune_state_dict(state_dict, args):
    """Prune the given state_dict if desired for LayerDrop
    (https://arxiv.org/abs/1909.11556).

    Training with LayerDrop allows models to be robust to pruning at inference
    time. This function prunes state_dict to allow smaller models to be loaded
    from a larger model and re-maps the existing state_dict for this to occur.

    It's called by functions that load models from checkpoints and does not
    need to be called directly.
    """
    if not args or args.arch == 'ptt_transformer':
        return state_dict
    encoder_layers_to_keep = args.encoder_layers_to_keep if 'encoder_layers_to_keep' in vars(args) else None
    decoder_layers_to_keep = args.decoder_layers_to_keep if 'decoder_layers_to_keep' in vars(args) else None
    if not encoder_layers_to_keep and not decoder_layers_to_keep:
        return state_dict
    logger.info('Pruning model to specified layer configuration - this works best if the model was trained with LayerDrop')

    def create_pruning_pass(layers_to_keep, layer_name):
        keep_layers = sorted([int(layer_string) for layer_string in layers_to_keep.split(',')])
        mapping_dict = {}
        for i in range(len(keep_layers)):
            mapping_dict[str(keep_layers[i])] = str(i)
        regex = re.compile('^{layer}.*\\.layers\\.(\\d+)'.format(layer=layer_name))
        return {'substitution_regex': regex, 'mapping_dict': mapping_dict}
    pruning_passes = []
    if encoder_layers_to_keep:
        pruning_passes.append(create_pruning_pass(encoder_layers_to_keep, 'encoder'))
    if decoder_layers_to_keep:
        pruning_passes.append(create_pruning_pass(decoder_layers_to_keep, 'decoder'))
    new_state_dict = {}
    for layer_name in state_dict.keys():
        match = re.search('\\.layers\\.(\\d+)\\.', layer_name)
        if not match:
            new_state_dict[layer_name] = state_dict[layer_name]
            continue
        original_layer_number = match.group(1)
        for pruning_pass in pruning_passes:
            if original_layer_number in pruning_pass['mapping_dict'] and pruning_pass['substitution_regex'].search(layer_name):
                new_layer_number = pruning_pass['mapping_dict'][original_layer_number]
                substitution_match = pruning_pass['substitution_regex'].search(layer_name)
                new_state_key = layer_name[:substitution_match.start(1)] + new_layer_number + layer_name[substitution_match.end(1):]
                new_state_dict[new_state_key] = state_dict[layer_name]
    if 'encoder_layers_to_keep' in vars(args):
        args.encoder_layers_to_keep = None
    if 'decoder_layers_to_keep' in vars(args):
        args.decoder_layers_to_keep = None
    return new_state_dict


class BaseFairseqModel(nn.Module):
    """Base class for fairseq models."""

    def __init__(self):
        super().__init__()
        self._is_generation_fast = False

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        dc = getattr(cls, '__dataclass', None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc())

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        raise NotImplementedError('Model must implement the build_model method')

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample['target']

    def get_normalized_probs(self, net_output: 'Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]]', log_probs: 'bool', sample: 'Optional[Dict[str, Tensor]]'=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

    def get_normalized_probs_scriptable(self, net_output: 'Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]]', log_probs: 'bool', sample: 'Optional[Dict[str, Tensor]]'=None):
        """Scriptable helper function for get_normalized_probs in ~BaseFairseqModel"""
        if hasattr(self, 'decoder'):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
        elif torch.is_tensor(net_output):
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    def extract_features(self, *args, **kwargs):
        """Similar to *forward* but only return features."""
        return self(*args, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return None

    def load_state_dict(self, state_dict, strict=True, args=None):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        self.upgrade_state_dict(state_dict)
        new_state_dict = prune_state_dict(state_dict, args)
        return super().load_state_dict(new_state_dict, strict)

    def upgrade_state_dict(self, state_dict):
        """Upgrade old state dicts to work with newer code."""
        self.upgrade_state_dict_named(state_dict, '')

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade old state dicts to work with newer code.

        Args:
            state_dict (dict): state dictionary to upgrade, in place
            name (str): the state dict key corresponding to the current module
        """
        assert state_dict is not None

        def do_upgrade(m, prefix):
            if len(prefix) > 0:
                prefix += '.'
            for n, c in m.named_children():
                name = prefix + n
                if hasattr(c, 'upgrade_state_dict_named'):
                    c.upgrade_state_dict_named(state_dict, name)
                elif hasattr(c, 'upgrade_state_dict'):
                    c.upgrade_state_dict(state_dict)
                do_upgrade(c, name)
        do_upgrade(self, name)

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""

        def _apply(m):
            if hasattr(m, 'set_num_updates') and m != self:
                m.set_num_updates(num_updates)
        self.apply(_apply)

    def prepare_for_inference_(self, args):
        """Prepare model for inference."""
        kwargs = {}
        kwargs['beamable_mm_beam_size'] = None if getattr(args, 'no_beamable_mm', False) else getattr(args, 'beam', 5)
        kwargs['need_attn'] = getattr(args, 'print_alignment', False)
        if hasattr(args, 'retain_dropout'):
            kwargs['retain_dropout'] = args.retain_dropout
            kwargs['retain_dropout_modules'] = getattr(args, 'retain_dropout_modules', None)
        self.make_generation_fast_(**kwargs)

    def make_generation_fast_(self, **kwargs):
        """
        Legacy entry point to optimize model for faster generation.
        Prefer prepare_for_inference_.
        """
        if self._is_generation_fast:
            return
        self._is_generation_fast = True

        def apply_remove_weight_norm(module):
            try:
                nn.utils.remove_weight_norm(module)
            except (AttributeError, ValueError):
                return
        self.apply(apply_remove_weight_norm)

        def apply_make_generation_fast_(module, prefix):
            if len(prefix) > 0:
                prefix += '.'
            base_func = BaseFairseqModel.make_generation_fast_
            for n, m in module.named_modules():
                if m != self and hasattr(m, 'make_generation_fast_') and m.make_generation_fast_.__func__ is not base_func:
                    name = prefix + n
                    m.make_generation_fast_(name=name, **kwargs)
        apply_make_generation_fast_(self, '')

        def train(mode=True):
            if mode:
                raise RuntimeError('cannot train after make_generation_fast')
        self.eval()
        self.train = train

    def prepare_for_onnx_export_(self, **kwargs):
        """Make model exportable via ONNX trace."""
        seen = set()

        def apply_prepare_for_onnx_export_(module):
            if module != self and hasattr(module, 'prepare_for_onnx_export_') and module not in seen:
                seen.add(module)
                module.prepare_for_onnx_export_(**kwargs)
        self.apply(apply_prepare_for_onnx_export_)

    def prepare_for_tpu_(self, **kwargs):
        """Optionally modify model for use on TPUs."""
        seen = set()

        def apply_prepare_for_tpu_(module):
            if module != self and hasattr(module, 'prepare_for_tpu_') and module not in seen:
                seen.add(module)
                module.prepare_for_tpu_(**kwargs)
        self.apply(apply_prepare_for_tpu_)

    @classmethod
    def upgrade_args(cls, args):
        if hasattr(args, 'max_sentences') and not hasattr(args, 'batch_size'):
            args.batch_size = args.max_sentences

    @classmethod
    def from_pretrained(cls, model_name_or_path, checkpoint_file='model.pt', data_name_or_path='.', **kwargs):
        """
        Load a :class:`~fairseq.models.FairseqModel` from a pre-trained model
        file. Downloads and caches the pre-trained model file if needed.

        The base implementation returns a
        :class:`~fairseq.hub_utils.GeneratorHubInterface`, which can be used to
        generate translations or sample from language models. The underlying
        :class:`~fairseq.models.FairseqModel` can be accessed via the
        *generator.models* attribute.

        Other models may override this to implement custom hub interfaces.

        Args:
            model_name_or_path (str): either the name of a pre-trained model to
                load or a path/URL to a pre-trained model state dict
            checkpoint_file (str, optional): colon-separated list of checkpoint
                files in the model archive to ensemble (default: 'model.pt')
            data_name_or_path (str, optional): point args.data to the archive
                at the given path/URL. Can start with '.' or './' to reuse the
                model archive path.
        """
        x = hub_utils.from_pretrained(model_name_or_path, checkpoint_file, data_name_or_path, archive_map=cls.hub_models(), **kwargs)
        cls.upgrade_args(x['args'])
        logger.info(x['args'])
        return hub_utils.GeneratorHubInterface(x['args'], x['task'], x['models'])

    @classmethod
    def hub_models(cls):
        return {}


class ZeroPad1d(nn.Module):

    def __init__(self, pad_left, pad_right):
        super().__init__()
        self.pad_left = pad_left
        self.pad_right = pad_right

    def forward(self, x):
        return F.pad(x, (self.pad_left, self.pad_right))


class Fp32GroupNorm(nn.GroupNorm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(input.float(), self.num_groups, self.weight.float() if self.weight is not None else None, self.bias.float() if self.bias is not None else None, self.eps)
        return output.type_as(input)


class Fp32LayerNorm(nn.LayerNorm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(input.float(), self.normalized_shape, self.weight.float() if self.weight is not None else None, self.bias.float() if self.bias is not None else None, self.eps)
        return output.type_as(input)


class TransposeLast(nn.Module):

    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)


def norm_block(is_layer_norm, dim, affine=True):
    if is_layer_norm:
        mod = nn.Sequential(TransposeLast(), Fp32LayerNorm(dim, elementwise_affine=affine), TransposeLast())
    else:
        mod = Fp32GroupNorm(1, dim, affine=affine)
    return mod


class ConvAggegator(nn.Module):

    def __init__(self, conv_layers, embed, dropout, skip_connections, residual_scale, non_affine_group_norm, conv_bias, zero_pad, activation):
        super().__init__()

        def block(n_in, n_out, k, stride):
            ka = k // 2
            kb = ka - 1 if k % 2 == 0 else ka
            pad = ZeroPad1d(ka + kb, 0) if zero_pad else nn.ReplicationPad1d((ka + kb, 0))
            return nn.Sequential(pad, nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias), nn.Dropout(p=dropout), norm_block(False, n_out, affine=not non_affine_group_norm), activation)
        in_d = embed
        self.conv_layers = nn.ModuleList()
        self.residual_proj = nn.ModuleList()
        for dim, k, stride in conv_layers:
            if in_d != dim and skip_connections:
                self.residual_proj.append(nn.Conv1d(in_d, dim, 1, bias=False))
            else:
                self.residual_proj.append(None)
            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim
        self.conv_layers = nn.Sequential(*self.conv_layers)
        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x):
        for rproj, conv in zip(self.residual_proj, self.conv_layers):
            residual = x
            x = conv(x)
            if self.skip_connections:
                if rproj is not None:
                    residual = rproj(residual)
                x = (x + residual) * self.residual_scale
        return x


class ConvFeatureExtractionModel(nn.Module):

    def __init__(self, conv_layers: 'List[Tuple[int, int, int]]', dropout: 'float'=0.0, mode: 'str'='default', conv_bias: 'bool'=False):
        super().__init__()
        assert mode in {'default', 'layer_norm'}

        def block(n_in, n_out, k, stride, is_layer_norm=False, is_group_norm=False, conv_bias=False):

            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv
            assert (is_layer_norm and is_group_norm) == False, 'layer norm and group norm are exclusive'
            if is_layer_norm:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.Sequential(TransposeLast(), Fp32LayerNorm(dim, elementwise_affine=True), TransposeLast()), nn.GELU())
            elif is_group_norm:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), Fp32GroupNorm(dim, dim, affine=True), nn.GELU())
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())
        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, 'invalid conv definition: ' + str(cl)
            dim, k, stride = cl
            self.conv_layers.append(block(in_d, dim, k, stride, is_layer_norm=mode == 'layer_norm', is_group_norm=mode == 'default' and i == 0, conv_bias=conv_bias))
            in_d = dim

    def forward(self, x):
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x


class GumbelVectorQuantizer(nn.Module):

    def __init__(self, dim, num_vars, temp, groups, combine_groups, vq_dim, time_first, activation=nn.GELU(), weight_proj_depth=1, weight_proj_factor=1):
        """Vector quantization using gumbel softmax

        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            temp: temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor)
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
            vq_dim: dimensionality of the resulting quantized vector
            time_first: if true, expect input in BxTxC format, otherwise in BxCxT
            activation: what activation to use (should be a module). this is only used if weight_proj_depth is > 1
            weight_proj_depth: number of layers (with activation in between) to project input before computing logits
            weight_proj_factor: this is used only if weight_proj_depth is > 1. scales the inner dimensionality of
                                projections by this factor
        """
        super().__init__()
        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.time_first = time_first
        assert vq_dim % groups == 0, f'dim {vq_dim} must be divisible by groups {groups} for concatenation'
        var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1
        self.vars = nn.Parameter(torch.FloatTensor(1, num_groups * num_vars, var_dim))
        nn.init.uniform_(self.vars)
        if weight_proj_depth > 1:

            def block(input_dim, output_dim):
                return nn.Sequential(nn.Linear(input_dim, output_dim), activation)
            inner_dim = self.input_dim * weight_proj_factor
            self.weight_proj = nn.Sequential(*[block(self.input_dim if i == 0 else inner_dim, inner_dim) for i in range(weight_proj_depth - 1)], nn.Linear(inner_dim, groups * num_vars))
        else:
            self.weight_proj = nn.Linear(self.input_dim, groups * num_vars)
            nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
            nn.init.zeros_(self.weight_proj.bias)
        assert len(temp) == 3, temp
        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp
        self.codebook_indices = None

    def set_num_updates(self, num_updates):
        self.curr_temp = max(self.max_temp * self.temp_decay ** num_updates, self.min_temp)

    def get_codebook_indices(self):
        if self.codebook_indices is None:
            from itertools import product
            p = [range(self.num_vars)] * self.groups
            inds = list(product(*p))
            self.codebook_indices = torch.tensor(inds, dtype=torch.long, device=self.vars.device).flatten()
            if not self.combine_groups:
                self.codebook_indices = self.codebook_indices.view(self.num_vars ** self.groups, -1)
                for b in range(1, self.groups):
                    self.codebook_indices[:, b] += self.num_vars * b
                self.codebook_indices = self.codebook_indices.flatten()
        return self.codebook_indices

    def codebook(self):
        indices = self.get_codebook_indices()
        return self.vars.squeeze(0).index_select(0, indices).view(self.num_vars ** self.groups, -1)

    def sample_from_codebook(self, b, n):
        indices = self.get_codebook_indices()
        indices = indices.view(-1, self.groups)
        cb_size = indices.size(0)
        assert n < cb_size, f'sample size {n} is greater than size of codebook {cb_size}'
        sample_idx = torch.randint(low=0, high=cb_size, size=(b * n,))
        indices = indices[sample_idx]
        z = self.vars.squeeze(0).index_select(0, indices.flatten()).view(b, n, -1)
        return z

    def to_codebook_index(self, indices):
        res = indices.new_full(indices.shape[:-1], 0)
        for i in range(self.groups):
            exponent = self.groups - i - 1
            res += indices[..., i] * self.num_vars ** exponent
        return res

    def forward_idx(self, x):
        res = self.forward(x, produce_targets=True)
        return res['x'], res['targets']

    def forward(self, x, produce_targets=False):
        result = {'num_vars': self.num_vars * self.groups}
        if not self.time_first:
            x = x.transpose(1, 2)
        bsz, tsz, fsz = x.shape
        x = x.reshape(-1, fsz)
        x = self.weight_proj(x)
        x = x.view(bsz * tsz * self.groups, -1)
        _, k = x.max(-1)
        hard_x = x.new_zeros(*x.shape).scatter_(-1, k.view(-1, 1), 1.0).view(bsz * tsz, self.groups, -1)
        hard_probs = torch.mean(hard_x.float(), dim=0)
        result['code_perplexity'] = torch.exp(-torch.sum(hard_probs * torch.log(hard_probs + 1e-07), dim=-1)).sum()
        avg_probs = torch.softmax(x.view(bsz * tsz, self.groups, -1).float(), dim=-1).mean(dim=0)
        result['prob_perplexity'] = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-07), dim=-1)).sum()
        result['temp'] = self.curr_temp
        if self.training:
            x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=True).type_as(x)
        else:
            x = hard_x
        x = x.view(bsz * tsz, -1)
        vars = self.vars
        if self.combine_groups:
            vars = vars.repeat(1, self.groups, 1)
        if produce_targets:
            result['targets'] = x.view(bsz * tsz * self.groups, -1).argmax(dim=-1).view(bsz, tsz, self.groups).detach()
        x = x.unsqueeze(-1) * vars
        x = x.view(bsz * tsz, self.groups, self.num_vars, -1)
        x = x.sum(-2)
        x = x.view(bsz, tsz, -1)
        if not self.time_first:
            x = x.transpose(1, 2)
        result['x'] = x
        return result


class KmeansVectorQuantizer(nn.Module):

    def __init__(self, dim, num_vars, groups, combine_groups, vq_dim, time_first, gamma=0.25):
        """Vector quantization using straight pass-through estimator (i.e. kmeans)

        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
            vq_dim: dimensionality of the resulting quantized vector
            time_first: if true, expect input in BxTxC format, otherwise in BxCxT
            gamma: commitment loss coefficient
        """
        super().__init__()
        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.vq_dim = vq_dim
        self.time_first = time_first
        assert vq_dim % groups == 0, f'dim {vq_dim} must be divisible by groups {groups} for concatenation'
        self.var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1
        self.embedding = nn.Parameter(0.01 * torch.randn(num_vars, num_groups, self.var_dim))
        self.projection = nn.Sequential(nn.Conv1d(dim, dim, kernel_size=1, groups=groups, bias=False), Fp32GroupNorm(groups, dim))
        self.gamma = gamma
        self.mse_mean = nn.MSELoss(reduction='mean')

    def _pass_grad(self, x, y):
        """Manually set gradient for backward pass.
        for y = f(x), ensure that during the backward pass,
        dL/dy = dL/dx regardless of f(x).
        Returns:
            y, with the gradient forced to be dL/dy = dL/dx.
        """
        return y.detach() + (x - x.detach())

    @property
    def expand_embedding(self):
        if self.combine_groups:
            return self.embedding.expand(self.num_vars, self.groups, self.var_dim)
        return self.embedding

    def forward_idx(self, x):
        res = self.forward(x, produce_targets=True)
        return res['x'], res['targets']

    def forward(self, x, produce_targets=False):
        result = {'num_vars': self.num_vars}
        if self.time_first:
            x = x.transpose(1, 2)
        bsz, fsz, tsz = x.shape
        ze = self.projection(x)
        ze_ = ze.view(bsz, self.groups, self.var_dim, tsz).permute(0, 3, 1, 2)
        d = (ze_.unsqueeze(0) - self.expand_embedding.unsqueeze(1).unsqueeze(1)).view(self.num_vars, bsz, tsz, self.groups, -1).norm(dim=-1, p=2)
        idx = d.argmin(dim=0)
        zq = torch.stack([self.expand_embedding[idx[..., group], group] for group in range(self.groups)], dim=-2).view(bsz, tsz, self.groups * self.var_dim).permute(0, 2, 1)
        assert ze.shape == zq.shape, (ze.shape, zq.shape)
        x = self._pass_grad(ze, zq)
        hard_x = idx.new_zeros(bsz * tsz * self.groups, self.num_vars).scatter_(-1, idx.view(-1, 1), 1.0).view(bsz * tsz, self.groups, -1)
        hard_probs = torch.mean(hard_x.float(), dim=0)
        result['code_perplexity'] = torch.exp(-torch.sum(hard_probs * torch.log(hard_probs + 1e-07), dim=-1)).sum()
        if produce_targets:
            result['targets'] = idx
        if self.time_first:
            x = x.transpose(1, 2)
        result['x'] = x
        ze = ze.float()
        zq = zq.float()
        latent_loss = self.mse_mean(zq, ze.detach())
        commitment_loss = self.mse_mean(ze, zq.detach())
        result['kmeans_loss'] = latent_loss + self.gamma * commitment_loss
        return result


def buffered_arange(max):
    if not hasattr(buffered_arange, 'buf'):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


class Wav2VecPredictionsModel(nn.Module):

    def __init__(self, in_dim, out_dim, prediction_steps, n_negatives, cross_sample_negatives, sample_distance, dropout, offset, balanced_classes, infonce):
        super().__init__()
        self.n_negatives = n_negatives
        self.cross_sample_negatives = cross_sample_negatives
        self.sample_distance = sample_distance
        self.project_to_steps = nn.ConvTranspose2d(in_dim, out_dim, (1, prediction_steps))
        self.dropout = nn.Dropout(p=dropout)
        self.offset = offset
        self.balanced_classes = balanced_classes
        self.infonce = infonce

    def sample_negatives(self, y):
        bsz, fsz, tsz = y.shape
        y = y.transpose(0, 1)
        y = y.contiguous().view(fsz, -1)
        cross_high = tsz * bsz
        high = tsz if self.sample_distance is None else min(tsz, self.sample_distance)
        assert high > 1
        neg_idxs = torch.randint(low=0, high=high, size=(bsz, self.n_negatives * tsz))
        with torch.no_grad():
            if self.n_negatives > 0:
                tszs = buffered_arange(tsz).unsqueeze(-1).expand(-1, self.n_negatives).flatten()
                neg_idxs = torch.randint(low=0, high=high - 1, size=(bsz, self.n_negatives * tsz))
                neg_idxs[neg_idxs >= tszs] += 1
            if self.cross_sample_negatives > 0:
                tszs = buffered_arange(tsz).unsqueeze(-1).expand(-1, self.cross_sample_negatives).flatten()
                cross_neg_idxs = torch.randint(low=0, high=cross_high - 1, size=(bsz, self.cross_sample_negatives * tsz))
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1
        if self.n_negatives > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs
        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)
        negs = y[..., neg_idxs.view(-1)]
        negs = negs.view(fsz, bsz, self.n_negatives + self.cross_sample_negatives, tsz).permute(2, 1, 0, 3)
        return negs

    def forward(self, x, y):
        x = x.unsqueeze(-1)
        x = self.project_to_steps(x)
        x = self.dropout(x)
        negatives = self.sample_negatives(y)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)
        copies = targets.size(0)
        bsz, dim, tsz, steps = x.shape
        steps = min(steps, tsz - self.offset)
        predictions = x.new(bsz * copies * (tsz - self.offset + 1) * steps - (steps + 1) * steps // 2 * copies * bsz)
        if self.infonce:
            labels = predictions.new_full((predictions.shape[0] // copies,), 0, dtype=torch.long)
        else:
            labels = torch.zeros_like(predictions)
        weights = torch.full_like(labels, 1 / self.n_negatives) if self.balanced_classes and not self.infonce else None
        start = end = 0
        for i in range(steps):
            offset = i + self.offset
            end = start + (tsz - offset) * bsz * copies
            if self.infonce:
                predictions[start:end] = torch.einsum('bct,nbct->tbn', x[..., :-offset, i], targets[..., offset:]).flatten()
            else:
                pos_num = (end - start) // copies
                predictions[start:end] = torch.einsum('bct,nbct->nbt', x[..., :-offset, i], targets[..., offset:]).flatten()
                labels[start:start + pos_num] = 1.0
                if weights is not None:
                    weights[start:start + pos_num] = 1.0
            start = end
        assert end == predictions.numel(), '{} != {}'.format(end, predictions.numel())
        if self.infonce:
            predictions = predictions.view(-1, copies)
        elif weights is not None:
            labels = labels, weights
        return predictions, labels


ARCH_CONFIG_REGISTRY = {}


ARCH_MODEL_INV_REGISTRY = {}


ARCH_MODEL_NAME_REGISTRY = {}


ARCH_MODEL_REGISTRY = {}


MODEL_REGISTRY = {}


def register_model_architecture(model_name, arch_name):
    """
    New model architectures can be added to fairseq with the
    :func:`register_model_architecture` function decorator. After registration,
    model architectures can be selected with the ``--arch`` command-line
    argument.

    For example::

        @register_model_architecture('lstm', 'lstm_luong_wmt_en_de')
        def lstm_luong_wmt_en_de(args):
            args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1000)
            (...)

    The decorated function should take a single argument *args*, which is a
    :class:`argparse.Namespace` of arguments parsed from the command-line. The
    decorated function should modify these arguments in-place to match the
    desired architecture.

    Args:
        model_name (str): the name of the Model (Model must already be
            registered)
        arch_name (str): the name of the model architecture (``--arch``)
    """

    def arch_override_from_yaml(args, arch):
        root_dir = os.path.dirname(os.path.dirname(fairseq.__file__))
        yaml_path = os.path.join(root_dir, 'config/model/{}.yaml'.format(arch))
        if not os.path.exists(yaml_path):
            raise RuntimeError(f'yaml file {yaml_path} does not exist!')
        arch_cfg = OmegaConf.load(yaml_path)
        for k, v in arch_cfg.items():
            setattr(args, k, getattr(args, k, v))

    def register_model_arch_fn(fn):
        if model_name not in MODEL_REGISTRY:
            raise ValueError('Cannot register model architecture for unknown model type ({})'.format(model_name))
        if arch_name in ARCH_MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate model architecture ({})'.format(arch_name))
        if not callable(fn):
            raise ValueError('Model architecture must be callable ({})'.format(arch_name))
        ARCH_MODEL_REGISTRY[arch_name] = MODEL_REGISTRY[model_name]
        ARCH_MODEL_NAME_REGISTRY[arch_name] = model_name
        ARCH_MODEL_INV_REGISTRY.setdefault(model_name, []).append(arch_name)
        if type(fn) is type and issubclass(fn, BaseFairseqModel):
            ARCH_CONFIG_REGISTRY[arch_name] = lambda args: arch_override_from_yaml(args, arch=arch_name)
        else:
            ARCH_CONFIG_REGISTRY[arch_name] = fn
        return fn
    return register_model_arch_fn


class Wav2VecModel(BaseFairseqModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--prediction-steps', type=int, metavar='N', help='number of steps ahead to predict')
        parser.add_argument('--sample-distance', type=int, metavar='N', help='sample distance from target. does not work properly with cross-sampling')
        parser.add_argument('--cross-sample-negatives', type=int, metavar='N', help='num of cross sampled negatives')
        parser.add_argument('--num-negatives', type=int, metavar='N', help='number of negative examples')
        parser.add_argument('--conv-feature-layers', type=str, metavar='EXPR', help='convolutional feature extraction layers [(dim, kernel_size, stride), ...]')
        parser.add_argument('--conv-aggregator-layers', type=str, metavar='EXPR', help='convolutional feature extraction layers [(dim, kernel_size, stride), ...]')
        parser.add_argument('--dropout', type=float, metavar='D', help='dropout to apply within the model')
        parser.add_argument('--dropout-features', type=float, metavar='D', help='dropout to apply to the features')
        parser.add_argument('--dropout-agg', type=float, metavar='D', help='dropout to apply after aggregation step')
        parser.add_argument('--encoder', type=str, choices=['cnn'], help='type of encoder to use')
        parser.add_argument('--aggregator', type=str, choices=['cnn', 'gru'], help='type of aggregator to use')
        parser.add_argument('--gru-dim', type=int, metavar='N', help='GRU dimensionality')
        parser.add_argument('--no-conv-bias', action='store_true', help='if set, does not learn bias for conv layers')
        parser.add_argument('--agg-zero-pad', action='store_true', help='if set, zero pads in aggregator instead of repl pad')
        parser.add_argument('--skip-connections-feat', action='store_true', help='if set, adds skip connections to the feature extractor')
        parser.add_argument('--skip-connections-agg', action='store_true', help='if set, adds skip connections to the aggregator')
        parser.add_argument('--residual-scale', type=float, metavar='D', help='scales residual by sqrt(value)')
        parser.add_argument('--log-compression', action='store_true', help='if set, adds a log compression to feature extractor')
        parser.add_argument('--balanced-classes', action='store_true', help='if set, loss is scaled to balance for number of negatives')
        parser.add_argument('--project-features', choices=['none', 'same', 'new'], help='if not none, features are projected using the (same or new) aggregator')
        parser.add_argument('--non-affine-group-norm', action='store_true', help='if set, group norm is not affine')
        parser.add_argument('--offset', help='if set, introduces an offset from target to predictions. if set to "auto", it is computed automatically from the receptive field')
        parser.add_argument('--activation', type=str, choices=['relu', 'gelu'], help='which activation function to use')
        parser.add_argument('--vq-type', type=str, choices=['none', 'gumbel', 'kmeans'], help='which type of quantizer to use')
        parser.add_argument('--vq-vars', type=int, metavar='N', help='if set, project to this many vector quantized variables per group')
        parser.add_argument('--vq-groups', type=int, metavar='N', help='number of groups of latent variables')
        parser.add_argument('--vq-dim', type=int, metavar='N', help='uses this dimensionality for quantized vectors')
        parser.add_argument('--vq-depth', type=int, metavar='N', help='number of layers for vq weight projection')
        parser.add_argument('--combine-groups', action='store_true', help='if set, variables are shared among groups')
        parser.add_argument('--vq-temp', type=str, metavar='TEMP', help='temperature for latent variable sampling with gumbel softmax. should be a tuple of 3 values (start, end, decay)')
        parser.add_argument('--vq-gamma', type=float, metavar='D', help='gamma parameter for kmeans style vector quantization')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_wav2vec_architecture(args)
        model = Wav2VecModel(args)
        logger.info(model)
        return model

    def __init__(self, args):
        super().__init__()
        self.prediction_steps = args.prediction_steps
        offset = args.offset
        if args.activation == 'relu':
            activation = nn.ReLU()
        elif args.activation == 'gelu':
            activation = nn.GELU()
        else:
            raise Exception('unknown activation ' + args.activation)
        if args.encoder == 'cnn':
            feature_enc_layers = eval(args.conv_feature_layers)
            self.feature_extractor = ConvFeatureExtractionModel(conv_layers=feature_enc_layers, dropout=0.0, log_compression=args.log_compression, skip_connections=args.skip_connections_feat, residual_scale=args.residual_scale, non_affine_group_norm=args.non_affine_group_norm, activation=activation)
            embed = feature_enc_layers[-1][0]
        else:
            raise Exception('unknown encoder type ' + args.encoder)
        self.vector_quantizer = None
        if args.vq_type == 'gumbel':
            self.vector_quantizer = GumbelVectorQuantizer(dim=embed, num_vars=args.vq_vars, temp=eval(args.vq_temp), groups=args.vq_groups, combine_groups=args.combine_groups, vq_dim=args.vq_dim if args.vq_dim > 0 else embed, time_first=False, activation=activation, weight_proj_depth=args.vq_depth, weight_proj_factor=2)
        elif args.vq_type == 'kmeans':
            self.vector_quantizer = KmeansVectorQuantizer(dim=embed, num_vars=args.vq_vars, groups=args.vq_groups, combine_groups=args.combine_groups, vq_dim=args.vq_dim if args.vq_dim > 0 else embed, time_first=False, gamma=args.vq_gamma)
        else:
            assert args.vq_type == 'none' or args.vq_type is None, 'Unknown quantizer type'
        if args.offset == 'auto':
            assert args.encoder == 'cnn'
            jin = 0
            rin = 0
            for _, k, stride in feature_enc_layers:
                if rin == 0:
                    rin = k
                rin = rin + (k - 1) * jin
                if jin == 0:
                    jin = stride
                else:
                    jin *= stride
            offset = math.ceil(rin / jin)
        offset = int(offset)

        def make_aggregator():
            if args.aggregator == 'cnn':
                agg_layers = eval(args.conv_aggregator_layers)
                agg_dim = agg_layers[-1][0]
                feature_aggregator = ConvAggegator(conv_layers=agg_layers, embed=embed, dropout=args.dropout, skip_connections=args.skip_connections_agg, residual_scale=args.residual_scale, non_affine_group_norm=args.non_affine_group_norm, conv_bias=not args.no_conv_bias, zero_pad=args.agg_zero_pad, activation=activation)
            elif args.aggregator == 'gru':
                agg_dim = args.gru_dim
                feature_aggregator = nn.Sequential(TransposeLast(), nn.GRU(input_size=embed, hidden_size=agg_dim, num_layers=1, dropout=args.dropout), TransposeLast(deconstruct_idx=0))
            else:
                raise Exception('unknown aggregator type ' + args.aggregator)
            return feature_aggregator, agg_dim
        self.feature_aggregator, agg_dim = make_aggregator()
        self.wav2vec_predictions = Wav2VecPredictionsModel(in_dim=agg_dim, out_dim=embed, prediction_steps=args.prediction_steps, n_negatives=args.num_negatives, cross_sample_negatives=args.cross_sample_negatives, sample_distance=args.sample_distance, dropout=args.dropout, offset=offset, balanced_classes=args.balanced_classes, infonce=args.infonce)
        self.dropout_feats = nn.Dropout(p=args.dropout_features)
        self.dropout_agg = nn.Dropout(p=args.dropout_agg)
        if args.project_features == 'none':
            self.project_features = None
        elif args.project_features == 'same':
            self.project_features = self.feature_aggregator
        elif args.project_features == 'new':
            self.project_features, _ = make_aggregator()

    def forward(self, source):
        result = {}
        features = self.feature_extractor(source)
        if self.vector_quantizer:
            q_res = self.vector_quantizer(features)
            features = q_res['x']
            for k in q_res.keys():
                if k != 'x':
                    result[k] = q_res[k]
        x = self.dropout_feats(features)
        x = self.feature_aggregator(x)
        x = self.dropout_agg(x)
        if self.project_features is not None:
            features = self.project_features(features)
        x, targets = self.wav2vec_predictions(x, features)
        result['cpc_logits'] = x
        result['cpc_targets'] = targets
        return result

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

    def max_positions(self):
        """Maximum length supported by the model."""
        return sys.maxsize

    def get_logits(self, net_output):
        logits = net_output['cpc_logits']
        return logits

    def get_targets(self, sample, net_output):
        t = net_output['cpc_targets']
        if isinstance(t, tuple):
            t = t[0]
        return t.contiguous()

    def get_target_weights(self, targets, net_output):
        targets = net_output['cpc_targets']
        if isinstance(targets, tuple) and targets[-1] is not None:
            return targets[-1]
        return None

    def get_extra_losses(self, net_output):
        loss = None
        if 'prob_perplexity' in net_output:
            loss = net_output['num_vars'] - net_output['prob_perplexity']
        elif 'kmeans_loss' in net_output:
            loss = net_output['kmeans_loss']
        return loss


class PretrainedWav2VecModel(nn.Module):

    def __init__(self, fname):
        super().__init__()
        checkpoint = torch.load(fname)
        self.args = checkpoint['args']
        model = Wav2VecModel.build_model(self.args, None)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        self.model = model

    def forward(self, x):
        with torch.no_grad():
            z = self.model.feature_extractor(x)
            if isinstance(z, tuple):
                z = z[0]
            c = self.model.feature_aggregator(z)
        return z, c


class FairseqCriterion(_Loss):

    def __init__(self, task):
        super().__init__()
        self.task = task
        if hasattr(task, 'target_dictionary'):
            tgt_dict = task.target_dictionary
            self.padding_idx = tgt_dict.pad() if tgt_dict is not None else -100

    @classmethod
    def add_args(cls, parser):
        """Add criterion-specific arguments to the parser."""
        dc = getattr(cls, '__dataclass', None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc())

    @classmethod
    def build_criterion(cls, args, task):
        """Construct a criterion from command-line args."""
        init_args = {}
        for p in inspect.signature(cls).parameters.values():
            if p.kind == p.POSITIONAL_ONLY or p.kind == p.VAR_POSITIONAL or p.kind == p.VAR_KEYWORD:
                raise NotImplementedError('{} not supported'.format(p.kind))
            assert p.kind in {p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY}
            if p.name == 'task':
                init_args['task'] = task
            elif hasattr(args, p.name):
                init_args[p.name] = getattr(args, p.name)
            elif p.default != p.empty:
                pass
            else:
                raise NotImplementedError('Unable to infer Criterion arguments, please implement {}.build_criterion'.format(cls.__name__))
        return cls(**init_args)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        raise NotImplementedError

    @staticmethod
    def aggregate_logging_outputs(logging_outputs: 'List[Dict[str, Any]]') ->Dict[str, Any]:
        """Aggregate logging outputs from data parallel training."""
        utils.deprecation_warning('The aggregate_logging_outputs API is deprecated. Please use the reduce_metrics API instead.')
        raise NotImplementedError

    @classmethod
    def reduce_metrics(cls, logging_outputs: 'List[Dict[str, Any]]') ->None:
        """Aggregate logging outputs from data parallel training."""
        utils.deprecation_warning('Criterions should implement the reduce_metrics API. Falling back to deprecated aggregate_logging_outputs API.')
        agg_logging_outputs = cls.aggregate_logging_outputs(logging_outputs)
        for k, v in agg_logging_outputs.items():
            if k in {'nsentences', 'ntokens', 'sample_size'}:
                continue
            metrics.log_scalar(k, v)

    @staticmethod
    def logging_outputs_can_be_summed() ->bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False


class LegacyFairseqCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(task=task)
        self.args = args
        utils.deprecation_warning('Criterions should take explicit arguments instead of an argparse.Namespace object, please update your criterion by extending FairseqCriterion instead of LegacyFairseqCriterion.')

    @classmethod
    def build_criterion(cls, args, task):
        """Construct a criterion from command-line args."""
        return cls(args, task)


def label_smoothed_nll_loss(lprobs, target, epsilon=0.002, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss


class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, ignore_prefix_size=0, report_accuracy=False):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0.0, type=float, metavar='D', help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report-accuracy', action='store_true', help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int, help='Ignore first N tokens')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {'loss': loss.data, 'nll_loss': nll_loss.data, 'ntokens': sample['ntokens'], 'nsentences': sample['target'].size(0), 'sample_size': sample_size}
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output['n_correct'] = utils.item(n_correct.data)
            logging_output['total'] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, 'batch_first', False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce)
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask)))
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) ->None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        total = utils.item(sum(log.get('total', 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar('total', total)
            n_correct = utils.item(sum(log.get('n_correct', 0) for log in logging_outputs))
            metrics.log_scalar('n_correct', n_correct)
            metrics.log_derived('accuracy', lambda meters: round(meters['n_correct'].sum * 100.0 / meters['total'].sum, 3) if meters['total'].sum > 0 else float('nan'))

    @staticmethod
    def logging_outputs_can_be_summed() ->bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


def compute_cross_entropy_loss(logits, targets, ignore_index=-100):
    """
    Function to compute the cross entropy loss. The default value of
    ignore_index is the same as the default value for F.cross_entropy in
    pytorch.
    """
    assert logits.size(0) == targets.size(-1), "Logits and Targets tensor shapes don't match up"
    loss = F.nll_loss(F.log_softmax(logits, -1, dtype=torch.float32), targets, reduction='sum', ignore_index=ignore_index)
    return loss


class LegacyMaskedLmLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    This optionally also computes the next sentence prediction (NSP) loss and
    adds it to the overall loss based on the specified args. There are three
    cases to consider:
        1) Generic MLM training without NSP loss. In this case sentence_targets
           and sentence_logits are both None.
        2) BERT training without NSP loss. In this case sentence_targets is
           not None but sentence_logits is None and we should not be computing
           a sentence level loss.
        3) BERT training with NSP loss. In this case both sentence_targets and
           sentence_logits are not None and we should be computing a sentence
           level loss. The weight of the sentence level loss is specified as
           an argument.
    """

    def __init__(self, task, masked_lm_only, nsp_loss_weight):
        super().__init__(task)
        self.masked_lm_only = masked_lm_only
        self.nsp_loss_weight = nsp_loss_weight

    @staticmethod
    def add_args(parser):
        """Args for MaskedLM Loss"""
        parser.add_argument('--masked-lm-only', default=False, action='store_true', help='compute MLM loss only')
        parser.add_argument('--nsp-loss-weight', default=1.0, type=float, help='weight for next sentence prediction loss (default 1)')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        lm_logits, output_metadata = model(**sample['net_input'])
        lm_logits = lm_logits.view(-1, lm_logits.size(-1))
        lm_targets = sample['lm_target'].view(-1)
        lm_loss = compute_cross_entropy_loss(lm_logits, lm_targets, self.padding_idx)
        ntokens = utils.strip_pad(lm_targets, self.padding_idx).numel()
        loss = lm_loss / ntokens
        nsentences = sample['nsentences']
        sentence_loss = None
        if not self.masked_lm_only:
            sentence_logits = output_metadata['sentence_logits']
            sentence_targets = sample['sentence_target'].view(-1)
            nsentences = sentence_targets.size(0)
            if sentence_logits is not None:
                sentence_loss = compute_cross_entropy_loss(sentence_logits, sentence_targets)
                loss += self.nsp_loss_weight * (sentence_loss / nsentences)
        sample_size = 1
        logging_output = {'loss': utils.item(loss.data) if reduce else loss.data, 'lm_loss': utils.item(lm_loss.data) if reduce else lm_loss.data, 'sentence_loss': (utils.item(sentence_loss.data) if reduce else sentence_loss.data) if sentence_loss is not None else 0.0, 'ntokens': ntokens, 'nsentences': nsentences, 'sample_size': sample_size}
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) ->None:
        """Aggregate logging outputs from data parallel training."""
        lm_loss_sum = sum(log.get('lm_loss', 0) for log in logging_outputs)
        sentence_loss_sum = sum(log.get('sentence_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_loss = sum(log.get('loss', 0) for log in logging_outputs)
        metrics.log_scalar('loss', agg_loss / sample_size / math.log(2) if sample_size > 0 else 0.0, sample_size, round=3)
        metrics.log_scalar('lm_loss', lm_loss_sum / ntokens / math.log(2) if ntokens > 0 else 0.0, ntokens, round=3)
        metrics.log_scalar('sentence_loss', sentence_loss_sum / nsentences / math.log(2) if nsentences > 0 else 0.0, nsentences, round=3)
        metrics.log_scalar('nll_loss', lm_loss_sum / ntokens / math.log(2) if ntokens > 0 else 0.0, ntokens, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() ->bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


class MaskedLmLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, task, tpu=False):
        super().__init__(task)
        self.tpu = tpu

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        masked_tokens = sample['target'].ne(self.padding_idx)
        sample_size = masked_tokens.int().sum()
        if self.tpu:
            masked_tokens = None
        elif masked_tokens.device == torch.device('cpu'):
            if not masked_tokens.any():
                masked_tokens = None
        else:
            masked_tokens = torch.where(masked_tokens.any(), masked_tokens, masked_tokens.new([True]))
        logits = model(**sample['net_input'], masked_tokens=masked_tokens)[0]
        targets = model.get_targets(sample, [logits])
        if masked_tokens is not None:
            targets = targets[masked_tokens]
        loss = modules.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='sum', ignore_index=self.padding_idx)
        logging_output = {'loss': loss if self.tpu else loss.data, 'ntokens': sample['ntokens'], 'nsentences': sample['nsentences'], 'sample_size': sample_size}
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) ->None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() ->bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


class LabelSmoothedDualImitationCriterion(FairseqCriterion):

    def __init__(self, task, label_smoothing):
        super().__init__(task)
        self.label_smoothing = label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0.0, type=float, metavar='D', help='epsilon for label smoothing, 0 means no label smoothing')

    def _compute_loss(self, outputs, targets, masks=None, label_smoothing=0.0, name='loss', factor=1.0):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        def mean_ds(x: 'Tensor', dim=None) ->Tensor:
            return x.float().mean().type_as(x) if dim is None else x.float().mean(dim).type_as(x)
        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]
        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets, reduction='none')
            else:
                losses = F.kl_div(logits, targets, reduction='none')
                losses = losses.sum(-1)
            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
            else:
                loss = nll_loss
        loss = loss * factor
        return {'name': name, 'loss': loss, 'nll_loss': nll_loss, 'factor': factor}

    def _custom_loss(self, loss, name='loss', factor=1.0):
        return {'name': name, 'loss': loss, 'factor': factor}

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample['nsentences'], sample['ntokens']
        src_tokens, src_lengths = sample['net_input']['src_tokens'], sample['net_input']['src_lengths']
        tgt_tokens, prev_output_tokens = sample['target'], sample['prev_target']
        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
        losses, nll_loss = [], []
        for obj in outputs:
            if outputs[obj].get('loss', None) is None:
                _losses = self._compute_loss(outputs[obj].get('out'), outputs[obj].get('tgt'), outputs[obj].get('mask', None), outputs[obj].get('ls', 0.0), name=obj + '-loss', factor=outputs[obj].get('factor', 1.0))
            else:
                _losses = self._custom_loss(outputs[obj].get('loss'), name=obj + '-loss', factor=outputs[obj].get('factor', 1.0))
            losses += [_losses]
            if outputs[obj].get('nll_loss', False):
                nll_loss += [_losses.get('nll_loss', 0.0)]
        loss = sum(l['loss'] for l in losses)
        nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 else loss.new_tensor(0)
        sample_size = 1
        logging_output = {'loss': loss.data, 'nll_loss': nll_loss.data, 'ntokens': ntokens, 'nsentences': nsentences, 'sample_size': sample_size}
        for l in losses:
            logging_output[l['name']] = utils.item(l['loss'].data / l['factor']) if reduce else l[['loss']].data / l['factor']
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) ->None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))
        loss = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get('nll_loss', 0) for log in logging_outputs))
        metrics.log_scalar('loss', loss / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))
        for key in logging_outputs[0]:
            if key[-5:] == '-loss':
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(key[:-5], val / sample_size / math.log(2) if sample_size > 0 else 0.0, sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() ->bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


class SentencePredictionCriterion(FairseqCriterion):

    def __init__(self, task, classification_head_name, regression_target):
        super().__init__(task)
        self.classification_head_name = classification_head_name
        self.regression_target = regression_target

    @staticmethod
    def add_args(parser):
        parser.add_argument('--classification-head-name', default='sentence_classification_head', help='name of the classification head to use')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert hasattr(model, 'classification_heads') and self.classification_head_name in model.classification_heads, 'model must provide sentence classification head for --criterion=sentence_prediction'
        logits, _ = model(**sample['net_input'], features_only=True, classification_head_name=self.classification_head_name)
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()
        if not self.regression_target:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            loss = F.nll_loss(lprobs, targets, reduction='sum')
        else:
            logits = logits.view(-1).float()
            targets = targets.float()
            loss = F.mse_loss(logits, targets, reduction='sum')
        logging_output = {'loss': loss.data, 'ntokens': sample['ntokens'], 'nsentences': sample_size, 'sample_size': sample_size}
        if not self.regression_target:
            preds = logits.argmax(dim=1)
            logging_output['ncorrect'] = (preds == targets).sum()
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) ->None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy', 100.0 * ncorrect / nsentences, nsentences, round=1)

    @staticmethod
    def logging_outputs_can_be_summed() ->bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


class SentenceRankingCriterion(FairseqCriterion):

    def __init__(self, task, ranking_head_name, save_predictions, num_classes):
        super().__init__(task)
        self.ranking_head_name = ranking_head_name
        if save_predictions is not None:
            self.prediction_h = open(save_predictions, 'w')
        else:
            self.prediction_h = None
        self.num_classes = num_classes

    def __del__(self):
        if self.prediction_h is not None:
            self.prediction_h.close()

    @staticmethod
    def add_args(parser):
        parser.add_argument('--save-predictions', metavar='FILE', help='file to save predictions to')
        parser.add_argument('--ranking-head-name', default='sentence_classification_head', help='name of the ranking head to use')

    def forward(self, model, sample, reduce=True):
        """Compute ranking loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert hasattr(model, 'classification_heads') and self.ranking_head_name in model.classification_heads, 'model must provide sentence ranking head for --criterion=sentence_ranking'
        scores = []
        for idx in range(self.num_classes):
            score, _ = model(**sample['net_input{idx}'.format(idx=idx + 1)], classification_head_name=self.ranking_head_name)
            scores.append(score)
        logits = torch.cat(scores, dim=1)
        sample_size = logits.size(0)
        if 'target' in sample:
            targets = model.get_targets(sample, [logits]).view(-1)
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            loss = F.nll_loss(lprobs, targets, reduction='sum')
        else:
            targets = None
            loss = torch.tensor(0.0, requires_grad=True)
        if self.prediction_h is not None:
            preds = logits.argmax(dim=1)
            for i, (id, pred) in enumerate(zip(sample['id'].tolist(), preds.tolist())):
                if targets is not None:
                    label = targets[i].item()
                    None
                else:
                    None
        logging_output = {'loss': loss.data, 'ntokens': sample['ntokens'], 'nsentences': sample_size, 'sample_size': sample_size}
        if targets is not None:
            logging_output['ncorrect'] = (logits.argmax(dim=1) == targets).sum()
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) ->None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy', 100.0 * ncorrect / nsentences, nsentences, round=1)

    @staticmethod
    def logging_outputs_can_be_summed() ->bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


def safe_round(number, ndigits):
    if hasattr(number, '__round__'):
        return round(number, ndigits)
    elif torch is not None and torch.is_tensor(number) and number.numel() == 1:
        return safe_round(number.item(), ndigits)
    elif np is not None and np.ndim(number) == 0 and hasattr(number, 'item'):
        return safe_round(number.item(), ndigits)
    else:
        return number


class Wav2vecCriterion(FairseqCriterion):

    def __init__(self, task, infonce=False, loss_weights=None, log_keys=None):
        super().__init__(task)
        self.infonce = infonce
        self.loss_weights = None if loss_weights is None else eval(loss_weights)
        self.log_keys = [] if log_keys is None else eval(log_keys)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--infonce', action='store_true', help='if set, uses cross entropy instead of binary cross entropy (i.e. InfoNCE loss)')
        parser.add_argument('--loss-weights', type=str, default=None, help='weights for additional loss terms (not first one)')
        parser.add_argument('--log-keys', type=str, default=None, help='output keys to log')

    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        logits = model.get_logits(net_output).float()
        target = model.get_targets(sample, net_output)
        weights = None
        if hasattr(model, 'get_target_weights') and not self.infonce:
            weights = model.get_target_weights(target, net_output)
            if torch.is_tensor(weights):
                weights = weights.float()
        losses = []
        if self.infonce:
            loss = F.cross_entropy(logits, target, reduction='sum' if reduce else 'none')
        else:
            loss = F.binary_cross_entropy_with_logits(logits, target.float(), weights, reduction='sum' if reduce else 'none')
        sample_size = target.numel() if self.infonce else target.long().sum().item()
        losses.append(loss.detach().clone())
        if self.loss_weights is not None:
            assert hasattr(model, 'get_extra_losses')
            extra_losses = model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(self.loss_weights), f'{len(extra_losses)}, {len(self.loss_weights)}'
            for p, coef in zip(extra_losses, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    losses.append(p)
        logging_output = {'loss': loss.item() if reduce else loss, 'ntokens': sample_size, 'nsentences': sample['id'].numel(), 'sample_size': sample_size}
        for lk in self.log_keys:
            if lk in net_output:
                logging_output[lk] = float(net_output[lk])
        if len(losses) > 1:
            for i, l in enumerate(losses):
                logging_output[f'loss_{i}'] = l.item()
        if self.infonce:
            with torch.no_grad():
                if logits.numel() == 0:
                    corr = 0
                    count = 0
                else:
                    assert logits.dim() > 1, logits.shape
                    max = logits.argmax(-1) == 0
                    min = logits.argmin(-1) == 0
                    both = max & min
                    corr = max.long().sum().item() - both.long().sum().item()
                    count = max.numel()
                logging_output['correct'] = corr
                logging_output['count'] = count
        if log_pred:
            logging_output['logits'] = logits.cpu().numpy()
            logging_output['target'] = target.cpu().numpy()
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) ->None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        nsentences = utils.item(sum(log.get('nsentences', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('ntokens', ntokens)
        metrics.log_scalar('nsentences', nsentences)
        correct = sum(log.get('correct', 0) for log in logging_outputs)
        metrics.log_scalar('_correct', correct)
        total = sum(log.get('count', 0) for log in logging_outputs)
        metrics.log_scalar('_total', total)
        if total > 0:
            metrics.log_derived('accuracy', lambda meters: safe_round(meters['_correct'].sum / meters['_total'].sum, 5) if meters['_total'].sum > 0 else float('nan'))
        builtin_keys = {'loss', 'ntokens', 'nsentences', 'sample_size', 'correct', 'count'}
        for k in logging_outputs[0]:
            if k not in builtin_keys:
                val = sum(log.get(k, 0) for log in logging_outputs) / len(logging_outputs)
                if k.startswith('loss'):
                    metrics.log_scalar(k, val / sample_size / math.log(2), sample_size)
                else:
                    metrics.log_scalar(k, val, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() ->bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False


class GeneratorHubInterface(nn.Module):
    """
    PyTorch Hub interface for generating sequences from a pre-trained
    translation or language model.
    """

    def __init__(self, args, task, models):
        super().__init__()
        self.args = args
        self.task = task
        self.models = nn.ModuleList(models)
        self.src_dict = task.source_dictionary
        self.tgt_dict = task.target_dictionary
        for model in self.models:
            model.prepare_for_inference_(args)
        self.align_dict = utils.load_align_dict(getattr(args, 'replace_unk', None))
        self.tokenizer = encoders.build_tokenizer(args)
        self.bpe = encoders.build_bpe(args)
        self.max_positions = utils.resolve_max_positions(self.task.max_positions(), *[model.max_positions() for model in models])
        self.register_buffer('_float_tensor', torch.tensor([0], dtype=torch.float))

    @property
    def device(self):
        return self._float_tensor.device

    def translate(self, sentences: 'List[str]', beam: 'int'=5, verbose: 'bool'=False, **kwargs) ->List[str]:
        return self.sample(sentences, beam, verbose, **kwargs)

    def sample(self, sentences: 'List[str]', beam: 'int'=1, verbose: 'bool'=False, **kwargs) ->List[str]:
        if isinstance(sentences, str):
            return self.sample([sentences], beam=beam, verbose=verbose, **kwargs)[0]
        tokenized_sentences = [self.encode(sentence) for sentence in sentences]
        batched_hypos = self.generate(tokenized_sentences, beam, verbose, **kwargs)
        return [self.decode(hypos[0]['tokens']) for hypos in batched_hypos]

    def score(self, sentences: 'List[str]', **kwargs):
        if isinstance(sentences, str):
            return self.score([sentences], **kwargs)[0]
        tokenized_sentences = [self.encode(sentence) for sentence in sentences]
        return [hypos[0] for hypos in self.generate(tokenized_sentences, score_reference=True, **kwargs)]

    def generate(self, tokenized_sentences: 'List[torch.LongTensor]', beam: 'int'=5, verbose: 'bool'=False, skip_invalid_size_inputs=False, inference_step_args=None, **kwargs) ->List[List[Dict[str, torch.Tensor]]]:
        if torch.is_tensor(tokenized_sentences) and tokenized_sentences.dim() == 1:
            return self.generate(tokenized_sentences.unsqueeze(0), beam=beam, verbose=verbose, **kwargs)[0]
        gen_args = copy.copy(self.args)
        gen_args.beam = beam
        for k, v in kwargs.items():
            setattr(gen_args, k, v)
        generator = self.task.build_generator(self.models, gen_args)
        inference_step_args = inference_step_args or {}
        results = []
        for batch in self._build_batches(tokenized_sentences, skip_invalid_size_inputs):
            batch = utils.apply_to_sample(lambda t: t, batch)
            translations = self.task.inference_step(generator, self.models, batch, **inference_step_args)
            for id, hypos in zip(batch['id'].tolist(), translations):
                results.append((id, hypos))
        outputs = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]
        if verbose:

            def getarg(name, default):
                return getattr(gen_args, name, getattr(self.args, name, default))
            for source_tokens, target_hypotheses in zip(tokenized_sentences, outputs):
                src_str_with_unk = self.string(source_tokens)
                logger.info('S\t{}'.format(src_str_with_unk))
                for hypo in target_hypotheses:
                    hypo_str = self.decode(hypo['tokens'])
                    logger.info('H\t{}\t{}'.format(hypo['score'], hypo_str))
                    logger.info('P\t{}'.format(' '.join(map(lambda x: '{:.4f}'.format(x), hypo['positional_scores'].tolist()))))
                    if hypo['alignment'] is not None and getarg('print_alignment', False):
                        logger.info('A\t{}'.format(' '.join(['{}-{}'.format(src_idx, tgt_idx) for src_idx, tgt_idx in hypo['alignment']])))
        return outputs

    def encode(self, sentence: 'str') ->torch.LongTensor:
        sentence = self.tokenize(sentence)
        sentence = self.apply_bpe(sentence)
        return self.binarize(sentence)

    def decode(self, tokens: 'torch.LongTensor') ->str:
        sentence = self.string(tokens)
        sentence = self.remove_bpe(sentence)
        return self.detokenize(sentence)

    def tokenize(self, sentence: 'str') ->str:
        if self.tokenizer is not None:
            sentence = self.tokenizer.encode(sentence)
        return sentence

    def detokenize(self, sentence: 'str') ->str:
        if self.tokenizer is not None:
            sentence = self.tokenizer.decode(sentence)
        return sentence

    def apply_bpe(self, sentence: 'str') ->str:
        if self.bpe is not None:
            sentence = self.bpe.encode(sentence)
        return sentence

    def remove_bpe(self, sentence: 'str') ->str:
        if self.bpe is not None:
            sentence = self.bpe.decode(sentence)
        return sentence

    def binarize(self, sentence: 'str') ->torch.LongTensor:
        return self.src_dict.encode_line(sentence, add_if_not_exist=False).long()

    def string(self, tokens: 'torch.LongTensor') ->str:
        return self.tgt_dict.string(tokens)

    def _build_batches(self, tokens: 'List[List[int]]', skip_invalid_size_inputs: 'bool') ->Iterator[Dict[str, Any]]:
        lengths = torch.LongTensor([t.numel() for t in tokens])
        batch_iterator = self.task.get_batch_iterator(dataset=self.task.build_dataset_for_inference(tokens, lengths), max_tokens=self.args.max_tokens, max_sentences=self.args.batch_size, max_positions=self.max_positions, ignore_invalid_inputs=skip_invalid_size_inputs, disable_iterator_cache=True).next_epoch_itr(shuffle=False)
        return batch_iterator


def LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, export=False):
    if torch.jit.is_scripting():
        export = True
    if not export and torch.cuda.is_available() and has_fused_layernorm:
        return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: 'int', embedding_dim: 'int', padding_idx: 'int'):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.onnx_trace = False
        if self.padding_idx is not None:
            self.max_positions = self.num_embeddings - self.padding_idx - 1
        else:
            self.max_positions = self.num_embeddings

    def forward(self, input: 'Tensor', incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]'=None, positions: 'Optional[Tensor]'=None):
        """Input is expected to be of size [bsz x seqlen]."""
        assert positions is None or self.padding_idx is None, 'If positions is pre-computed then padding_idx should not be set.'
        if positions is None:
            if incremental_state is not None:
                positions = torch.zeros((1, 1), device=input.device, dtype=input.dtype).fill_(int(self.padding_idx + input.size(1)))
            else:
                positions = utils.make_positions(input, self.padding_idx, onnx_trace=self.onnx_trace)
        return F.embedding(positions, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(init_size, embedding_dim, padding_idx)
        self.onnx_trace = False
        self.register_buffer('_float_tensor', torch.FloatTensor(1))
        self.max_positions = int(100000.0)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    @staticmethod
    def get_embedding(num_embeddings: 'int', embedding_dim: 'int', padding_idx: 'Optional[int]'=None):
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

    def forward(self, input, incremental_state: 'Optional[Any]'=None, timestep: 'Optional[Tensor]'=None, positions: 'Optional[Any]'=None):
        """Input is expected to be of size [bsz x seqlen]."""
        bspair = torch.onnx.operators.shape_as_tensor(input)
        bsz, seq_len = bspair[0], bspair[1]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = SinusoidalPositionalEmbedding.get_embedding(max_pos, self.embedding_dim, self.padding_idx)
        self.weights = self.weights
        if incremental_state is not None:
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            if self.onnx_trace:
                return self.weights.index_select(index=self.padding_idx + pos, dim=0).unsqueeze(1).repeat(bsz, 1, 1)
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)
        positions = utils.make_positions(input, self.padding_idx, onnx_trace=self.onnx_trace)
        if self.onnx_trace:
            flat_embeddings = self.weights.detach().index_select(0, positions.view(-1))
            embedding_shape = torch.cat((bsz.view(1), seq_len.view(1), torch.tensor([-1], dtype=torch.long)))
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(flat_embeddings, embedding_shape)
            return embeddings
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()


def PositionalEmbedding(num_embeddings: 'int', embedding_dim: 'int', padding_idx: 'int', learned: 'bool'=False):
    if learned:
        if padding_idx is not None:
            num_embeddings = num_embeddings + padding_idx + 1
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        if padding_idx is not None:
            nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, init_size=num_embeddings + padding_idx + 1)
    return m


class TransformerEncoderEmbedding(nn.Module):
    """ Encoder Embedding + Positional Embedding """

    def __init__(self, args, embed_tokens):
        super().__init__()
        self.dropout = args.dropout
        self.max_source_positions = args.max_source_positions
        self.embed_tokens = embed_tokens
        if isinstance(embed_tokens, nn.ModuleList):
            self.padding_idx = embed_tokens[0].padding_idx
            embed_dim = sum(e.embedding_dim for e in embed_tokens)
        else:
            self.padding_idx = embed_tokens.padding_idx
            embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(args.max_source_positions, embed_dim, self.padding_idx, learned=args.encoder_learned_pos) if not args.no_token_positional_embeddings else None
        if getattr(args, 'layernorm_embedding', False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

    def forward(self, input):
        src_tokens = input[0]
        prev_output_tokens = input[2]
        if isinstance(self.embed_tokens, nn.ModuleList):
            x_embed_list = []
            for embed_tokens_part in self.embed_tokens:
                x_embed_list.append(embed_tokens_part(src_tokens))
            embedded = torch.cat(x_embed_list, dim=-1)
        else:
            embedded = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * embedded
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding:
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        return x, encoder_padding_mask, prev_output_tokens


class TransformerEncoderLayerNorm(nn.Module):
    """
    Layer norm at the the end of all encoder layers if
    args.encoder_enormalize_before = True
    """

    def __init__(self, args, embed_dim):
        super().__init__()
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(self, input):
        x = input[0]
        encoder_padding_mask = input[1]
        prev_output_tokens = input[2]
        if self.layer_norm:
            x = self.layer_norm(x)
        return x, encoder_padding_mask, prev_output_tokens


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class TransformerDecoderEmbedding(nn.Module):
    """ Decoder Embedding + Positional Embedding """

    def __init__(self, args, embed_tokens):
        super().__init__()
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed
        input_embed_dim = sum(e.embedding_dim for e in embed_tokens) if isinstance(embed_tokens, nn.ModuleList) else embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_output_dim
        padding_idx = embed_tokens[0].padding_idx if isinstance(embed_tokens, nn.ModuleList) else embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions
        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None
        self.embed_positions = PositionalEmbedding(args.max_target_positions, embed_dim, padding_idx, learned=args.decoder_learned_pos) if not args.no_token_positional_embeddings else None

    def forward(self, input):
        mt_task = False
        if isinstance(input, tuple):
            if len(input) == 3:
                encoder_out = input[0]
                encoder_padding_mask = input[1]
                prev_output_tokens = input[2]
                incremental_state = None
                mt_task = True
            else:
                prev_output_tokens = input[0]
                encoder_out = None
                encoder_padding_mask = None
                incremental_state = None
        else:
            prev_output_tokens = input
            encoder_out = None
            encoder_padding_mask = None
            incremental_state = None
        positions = self.embed_positions(prev_output_tokens, incremental_state=incremental_state) if self.embed_positions is not None else None
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        if isinstance(self.embed_tokens, nn.ModuleList):
            x_embed_list = []
            for embed_tokens_part in self.embed_tokens:
                x_embed_list.append(embed_tokens_part(prev_output_tokens))
            x = self.embed_scale * torch.cat(x_embed_list, dim=-1)
        else:
            x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)
        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)
        if mt_task:
            return x, encoder_out, encoder_padding_mask
        return x


class TiedLinear(nn.Module):

    def __init__(self, weight, transpose):
        super().__init__()
        self.weight = weight
        self.transpose = transpose

    def forward(self, input):
        return F.linear(input, self.weight.t() if self.transpose else self.weight)


class TiedHeadModule(nn.Module):

    def __init__(self, weights, input_dim, num_classes, q_noise, qn_block_size):
        super().__init__()
        tied_emb, _ = weights
        self.num_words, emb_dim = tied_emb.size()
        self.word_proj = quant_noise(TiedLinear(tied_emb, transpose=False), q_noise, qn_block_size)
        if input_dim != emb_dim:
            self.word_proj = nn.Sequential(quant_noise(nn.Linear(input_dim, emb_dim, bias=False), q_noise, qn_block_size), self.word_proj)
        self.class_proj = quant_noise(nn.Linear(input_dim, num_classes, bias=False), q_noise, qn_block_size)
        self.out_dim = self.num_words + num_classes
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def forward(self, input):
        inp_sz = functools.reduce(operator.mul, input.shape[:-1], 1)
        out = self._float_tensor.new(inp_sz, self.out_dim)
        out[:, :self.num_words] = self.word_proj(input.view(inp_sz, -1))
        out[:, self.num_words:] = self.class_proj(input.view(inp_sz, -1))
        return out


class AdaptiveSoftmax(nn.Module):
    """
    This is an implementation of the efficient softmax approximation for
    graphical processing units (GPU), described in the paper "Efficient softmax
    approximation for GPUs" (http://arxiv.org/abs/1609.04309).
    """

    def __init__(self, vocab_size, input_dim, cutoff, dropout, factor=4.0, adaptive_inputs=None, tie_proj=False, q_noise=0, qn_block_size=8):
        super().__init__()
        if vocab_size > cutoff[-1]:
            cutoff = cutoff + [vocab_size]
        else:
            assert vocab_size == cutoff[-1], 'cannot specify cutoff larger than vocab size'
        output_dim = cutoff[0] + len(cutoff) - 1
        self.vocab_size = vocab_size
        self.cutoff = cutoff
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.input_dim = input_dim
        self.factor = factor
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size
        self.lsm = nn.LogSoftmax(dim=1)
        if adaptive_inputs is not None:
            self.head = TiedHeadModule(adaptive_inputs.weights_for_band(0), input_dim, len(cutoff) - 1, self.q_noise, self.qn_block_size)
        else:
            self.head = quant_noise(nn.Linear(input_dim, output_dim, bias=False), self.q_noise, self.qn_block_size)
        self._make_tail(adaptive_inputs, tie_proj)

        def init_weights(m):
            if hasattr(m, 'weight') and not isinstance(m, TiedLinear) and not isinstance(m, TiedHeadModule):
                nn.init.xavier_uniform_(m.weight)
        self.apply(init_weights)
        self.register_buffer('version', torch.LongTensor([1]))

    def _make_tail(self, adaptive_inputs=None, tie_proj=False):
        self.tail = nn.ModuleList()
        for i in range(len(self.cutoff) - 1):
            dim = int(self.input_dim // self.factor ** (i + 1))
            tied_emb, tied_proj = adaptive_inputs.weights_for_band(i + 1) if adaptive_inputs is not None else (None, None)
            if tied_proj is not None:
                if tie_proj:
                    proj = quant_noise(TiedLinear(tied_proj, transpose=True), self.q_noise, self.qn_block_size)
                else:
                    proj = quant_noise(nn.Linear(tied_proj.size(0), tied_proj.size(1), bias=False), self.q_noise, self.qn_block_size)
            else:
                proj = quant_noise(nn.Linear(self.input_dim, dim, bias=False), self.q_noise, self.qn_block_size)
            if tied_emb is None:
                out_proj = nn.Linear(dim, self.cutoff[i + 1] - self.cutoff[i], bias=False)
            else:
                out_proj = TiedLinear(tied_emb, transpose=False)
            m = nn.Sequential(proj, nn.Dropout(self.dropout_module.p), quant_noise(out_proj, self.q_noise, self.qn_block_size))
            self.tail.append(m)

    def upgrade_state_dict_named(self, state_dict, name):
        version_name = name + '.version'
        if version_name not in state_dict:
            raise Exception('This version of the model is no longer supported')

    def adapt_target(self, target):
        """
        In order to be efficient, the AdaptiveSoftMax does not compute the
        scores for all the word of the vocabulary for all the examples. It is
        thus necessary to call the method adapt_target of the AdaptiveSoftMax
        layer inside each forward pass.
        """
        target = target.view(-1)
        new_target = [target.clone()]
        target_idxs = []
        for i in range(len(self.cutoff) - 1):
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))
            new_target[0][mask] = self.cutoff[0] + i
            if mask.any():
                target_idxs.append(mask.nonzero(as_tuple=False).squeeze(1))
                new_target.append(target[mask].add(-self.cutoff[i]))
            else:
                target_idxs.append(None)
                new_target.append(None)
        return new_target, target_idxs

    def forward(self, input, target):
        """
        Args:
            input: (b x t x d)
            target: (b x t)
        Returns:
            2 lists: output for each cutoff section and new targets by cut off
        """
        input = input.contiguous().view(-1, input.size(-1))
        input = self.dropout_module(input)
        new_target, target_idxs = self.adapt_target(target)
        output = [self.head(input)]
        for i in range(len(target_idxs)):
            if target_idxs[i] is not None:
                output.append(self.tail[i](input.index_select(0, target_idxs[i])))
            else:
                output.append(None)
        return output, new_target

    def get_log_prob(self, input, target):
        """
        Computes the log probabilities for all the words of the vocabulary,
        given a 2D tensor of hidden vectors.
        """
        bsz, length, dim = input.size()
        input = input.contiguous().view(-1, dim)
        if target is not None:
            _, target_idxs = self.adapt_target(target)
        else:
            target_idxs = None
        head_y = self.head(input)
        log_probs = head_y.new_zeros(input.size(0), self.vocab_size)
        head_sz = self.cutoff[0] + len(self.tail)
        log_probs[:, :head_sz] = self.lsm(head_y)
        tail_priors = log_probs[:, self.cutoff[0]:head_sz].clone()
        for i in range(len(self.tail)):
            start = self.cutoff[i]
            end = self.cutoff[i + 1]
            if target_idxs is None:
                tail_out = log_probs[:, start:end]
                tail_out.copy_(self.tail[i](input))
                log_probs[:, start:end] = self.lsm(tail_out).add_(tail_priors[:, i, None])
            elif target_idxs[i] is not None:
                idxs = target_idxs[i]
                tail_out = log_probs[idxs, start:end]
                tail_out.copy_(self.tail[i](input[idxs]))
                log_probs[idxs, start:end] = self.lsm(tail_out).add_(tail_priors[idxs, i, None])
        log_probs = log_probs.view(bsz, length, -1)
        return log_probs


class TransformerDecoderOutputLayer(nn.Module):

    def __init__(self, args, embed_tokens, dictionary):
        super().__init__()
        self.share_input_output_embed = args.share_decoder_input_output_embed
        self.embed_tokens = embed_tokens
        self.output_embed_dim = args.decoder_output_dim
        embed_dim = args.decoder_embed_dim
        self.project_out_dim = Linear(embed_dim, self.output_embed_dim, bias=False) if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights else None
        self.adaptive_softmax = None
        if args.adaptive_softmax_cutoff is not None:
            assert not isinstance(embed_tokens, nn.ModuleList)
            self.adaptive_softmax = AdaptiveSoftmax(len(dictionary), self.output_embed_dim, options.eval_str_list(args.adaptive_softmax_cutoff, type=int), dropout=args.adaptive_softmax_dropout, adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None, factor=args.adaptive_softmax_factor, tie_proj=args.tie_adaptive_proj)
        elif not self.share_input_output_embed:
            self.embed_tokens = nn.Parameter(torch.Tensor(len(dictionary), self.output_embed_dim))
            nn.init.normal_(self.embed_tokens, mean=0, std=self.output_embed_dim ** -0.5)
        if args.decoder_normalize_before and not getattr(args, 'no_decoder_final_norm', False):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(self, input, apply_final_proj=True):
        if isinstance(input, tuple):
            x = input[0]
        else:
            x = input
        if self.layer_norm:
            x = self.layer_norm(x)
        x = x.transpose(0, 1)
        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
        if apply_final_proj:
            x = self.output_layer(x)
        return x

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            if self.share_input_output_embed:
                if isinstance(self.embed_tokens, nn.ModuleList):
                    output = None
                    for i, emb in enumerate(self.embed_tokens):
                        sidx = i * emb.embedding_dim
                        eidx = (i + 1) * emb.embedding_dim
                        if output is None:
                            output = F.linear(features[:, :, sidx:eidx], emb.weight)
                        else:
                            output += F.linear(features[:, :, sidx:eidx], emb.weight)
                    return output
                else:
                    return F.linear(features, self.embed_tokens.weight)
            else:
                return F.linear(features, self.embed_tokens)
        else:
            return features


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8)
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.activation_fn = utils.get_activation_fn(activation=getattr(args, 'activation_fn', 'relu'))
        activation_dropout_p = getattr(args, 'activation_dropout', 0)
        if activation_dropout_p == 0:
            activation_dropout_p = getattr(args, 'relu_dropout', 0)
        self.activation_dropout_module = FairseqDropout(float(activation_dropout_p), module_name=self.__class__.__name__)
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(self.embed_dim, args.encoder_ffn_embed_dim, self.quant_noise, self.quant_noise_block_size)
        self.fc2 = self.build_fc2(args.encoder_ffn_embed_dim, self.embed_dim, self.quant_noise, self.quant_noise_block_size)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(embed_dim, args.encoder_attention_heads, dropout=args.attention_dropout, self_attention=True, q_noise=self.quant_noise, qn_block_size=self.quant_noise_block_size)

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {'0': 'self_attn_layer_norm', '1': 'final_layer_norm'}
        for old, new in layer_norm_map.items():
            for m in ('weight', 'bias'):
                k = '{}.layer_norms.{}.{}'.format(name, old, m)
                if k in state_dict:
                    state_dict['{}.{}.{}'.format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: 'Optional[Tensor]'=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask, -100000000.0)
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask, attn_mask=attn_mask)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8)
        self.cross_self_attention = getattr(args, 'cross_self_attention', False)
        self.self_attn = self.build_self_attention(self.embed_dim, args, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn)
        self.activation_fn = utils.get_activation_fn(activation=str(args.activation_fn) if getattr(args, 'activation_fn', None) is not None else 'relu')
        activation_dropout_p = getattr(args, 'activation_dropout', 0)
        if activation_dropout_p == 0:
            activation_dropout_p = getattr(args, 'relu_dropout', 0)
        self.activation_dropout_module = FairseqDropout(float(activation_dropout_p), module_name=self.__class__.__name__)
        self.normalize_before = args.decoder_normalize_before
        export = getattr(args, 'char_inputs', False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.fc1 = self.build_fc1(self.embed_dim, args.decoder_ffn_embed_dim, self.quant_noise, self.quant_noise_block_size)
        self.fc2 = self.build_fc2(args.decoder_ffn_embed_dim, self.embed_dim, self.quant_noise, self.quant_noise_block_size)
        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True
        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        return MultiheadAttention(embed_dim, args.decoder_attention_heads, dropout=args.attention_dropout, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn, self_attention=not getattr(args, 'cross_self_attention', False), q_noise=self.quant_noise, qn_block_size=self.quant_noise_block_size)

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(embed_dim, args.decoder_attention_heads, kdim=getattr(args, 'encoder_embed_dim', None), vdim=getattr(args, 'encoder_embed_dim', None), dropout=args.attention_dropout, encoder_decoder_attention=True, q_noise=self.quant_noise, qn_block_size=self.quant_noise_block_size)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(self, x, encoder_out: 'Optional[torch.Tensor]'=None, encoder_padding_mask: 'Optional[torch.Tensor]'=None, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]'=None, prev_self_attn_state: 'Optional[List[torch.Tensor]]'=None, prev_attn_state: 'Optional[List[torch.Tensor]]'=None, self_attn_mask: 'Optional[torch.Tensor]'=None, self_attn_padding_mask: 'Optional[torch.Tensor]'=None, need_attn: 'bool'=False, need_head_weights: 'bool'=False):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: 'Dict[str, Optional[Tensor]]' = {'prev_key': prev_key, 'prev_value': prev_value}
            if len(prev_self_attn_state) >= 3:
                saved_state['prev_key_padding_mask'] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (incremental_state is not None and _self_attn_input_buffer is not None and 'prev_key' in _self_attn_input_buffer):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat((x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1)
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(encoder_out.size(1), encoder_out.size(0))
                self_attn_padding_mask = torch.cat((encoder_padding_mask, self_attn_padding_mask), dim=1)
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x
        x, attn = self.self_attn(query=x, key=y, value=y, key_padding_mask=self_attn_padding_mask, incremental_state=incremental_state, need_weights=False, attn_mask=self_attn_mask)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: 'Dict[str, Optional[Tensor]]' = {'prev_key': prev_key, 'prev_value': prev_value}
                if len(prev_attn_state) >= 3:
                    saved_state['prev_key_padding_mask'] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            x, attn = self.encoder_attn(query=x, key=encoder_out, value=encoder_out, key_padding_mask=encoder_padding_mask, incremental_state=incremental_state, static_kv=True, need_weights=need_attn or not self.training and self.need_attn, need_head_weights=need_head_weights)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [saved_state['prev_key'], saved_state['prev_value'], saved_state['prev_key_padding_mask']]
            else:
                self_attn_state = [saved_state['prev_key'], saved_state['prev_value']]
            return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: 'bool'=False, **kwargs):
        self.need_attn = need_attn


class ModelParallelRobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = ColumnParallelLinear(embed_dim, embed_dim, gather_output=True)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)
        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        if masked_tokens is not None:
            features = features[masked_tokens, :]
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = copy_to_model_parallel_region(x)
        x = F.linear(x, self.weight)
        x = gather_from_model_parallel_region(x).contiguous()
        x = x + self.bias
        return x


class ModelParallelRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout):
        super().__init__()
        self.dense = ColumnParallelLinear(input_dim, inner_dim, gather_output=True)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@with_incremental_state
class ModelParallelMultiheadAttention(nn.Module):
    """Model parallel Multi-headed attention.
    This performs the Multi-headed attention over multiple gpus.

    See "Megatron-LM: https://arxiv.org/pdf/1909.08053.pdf" for more details.
    """

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0.0, bias=True, self_attention=False, encoder_decoder_attention=False):
        super().__init__()
        if not has_megatron_submodule:
            raise ImportError('\n\nPlease install the megatron submodule:\n\n  git submodule update --init fairseq/model_parallel/megatron')
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.model_parallel_size = get_model_parallel_world_size()
        self.num_heads_partition = num_heads // self.model_parallel_size
        assert self.num_heads_partition * self.model_parallel_size == num_heads, 'Number of heads must be divisible by model parallel size'
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.scaling = self.head_dim ** -0.5
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert not self.self_attention or self.qkv_same_dim, 'Self-attention requires query, key and value to be of the same size'
        self.k_proj = ColumnParallelLinear(self.kdim, embed_dim, bias=bias, gather_output=False)
        self.v_proj = ColumnParallelLinear(self.vdim, embed_dim, bias=bias, gather_output=False)
        self.q_proj = ColumnParallelLinear(embed_dim, embed_dim, bias=bias, gather_output=False)
        self.out_proj = RowParallelLinear(embed_dim, embed_dim, bias=bias, input_is_parallel=True)
        self.tpu = False

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def forward(self, query, key: 'Optional[Tensor]', value: 'Optional[Tensor]', key_padding_mask: 'Optional[Tensor]'=None, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]'=None, static_kv: 'bool'=False, attn_mask: 'Optional[Tensor]'=None, **unused_kwargs) ->Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
        """
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
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
        q = q.contiguous().view(tgt_len, bsz * self.num_heads_partition, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads_partition, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads_partition, self.head_dim).transpose(0, 1)
        if saved_state is not None:
            if 'prev_key' in saved_state:
                _prev_key = saved_state['prev_key']
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads_partition, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if 'prev_value' in saved_state:
                _prev_value = saved_state['prev_value']
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads_partition, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: 'Optional[Tensor]' = None
            if 'prev_key_padding_mask' in saved_state:
                prev_key_padding_mask = saved_state['prev_key_padding_mask']
            assert k is not None and v is not None
            key_padding_mask = ModelParallelMultiheadAttention._append_prev_key_padding_mask(key_padding_mask=key_padding_mask, prev_key_padding_mask=prev_key_padding_mask, batch_size=bsz, src_len=k.size(1), static_kv=static_kv)
            saved_state['prev_key'] = k.view(bsz, self.num_heads_partition, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads_partition, -1, self.head_dim)
            saved_state['prev_key_padding_mask'] = key_padding_mask
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        src_len = k.size(1)
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads_partition, tgt_len, src_len]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads_partition, tgt_len, src_len)
            if not self.tpu:
                attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads_partition, tgt_len, src_len)
        attn_weights_float = utils.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        with get_cuda_rng_tracker().fork():
            attn_probs = self.dropout_module(attn_weights)
        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads_partition, tgt_len, self.head_dim]
        embed_dim_partition = embed_dim // self.model_parallel_size
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim_partition)
        attn = self.out_proj(attn)
        attn_weights: 'Optional[Tensor]' = None
        return attn, attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(key_padding_mask: 'Optional[Tensor]', prev_key_padding_mask: 'Optional[Tensor]', batch_size: 'int', src_len: 'int', static_kv: 'bool') ->Optional[Tensor]:
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), key_padding_mask.float()], dim=1)
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(batch_size, src_len - prev_key_padding_mask.size(1))
            if prev_key_padding_mask.is_cuda:
                filler = filler
            new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), filler.float()], dim=1)
        elif key_padding_mask is not None:
            filler = torch.zeros(batch_size, src_len - key_padding_mask.size(1))
            if key_padding_mask.is_cuda:
                filler = filler
            new_key_padding_mask = torch.cat([filler.float(), key_padding_mask.float()], dim=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def reorder_incremental_state(self, incremental_state: 'Dict[str, Dict[str, Optional[Tensor]]]', new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                if input_buffer[k] is not None:
                    input_buffer[k] = input_buffer[k].index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(self, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]') ->Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, 'attn_state')
        if result is not None:
            return result
        else:
            empty_result: 'Dict[str, Optional[Tensor]]' = {}
            return empty_result

    def _set_input_buffer(self, incremental_state: 'Dict[str, Dict[str, Optional[Tensor]]]', buffer: 'Dict[str, Optional[Tensor]]'):
        return self.set_incremental_state(incremental_state, 'attn_state', buffer)


class BARTHubInterface(nn.Module):
    """A simple PyTorch Hub interface to BART.

    Usage: https://github.com/pytorch/fairseq/tree/master/examples/bart
    """

    def __init__(self, args, task, model):
        super().__init__()
        self.args = args
        self.task = task
        self.model = model
        self.bpe = encoders.build_bpe(args)
        self.max_positions = min(utils.resolve_max_positions(self.task.max_positions(), self.model.max_positions()))
        self.register_buffer('_float_tensor', torch.tensor([0], dtype=torch.float))

    @property
    def device(self):
        return self._float_tensor.device

    def encode(self, sentence: 'str', *addl_sentences, no_separator=True) ->torch.LongTensor:
        """
        BPE-encode a sentence (or multiple sentences).

        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`).

        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> 1 2 3 </s>`

        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::

            >>> bart.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> bart.encode(' world').tolist()
            [0, 232, 2]
            >>> bart.encode('world').tolist()
            [0, 8331, 2]
        """
        tokens = self.bpe.encode(sentence)
        if len(tokens.split(' ')) > self.max_positions - 2:
            tokens = ' '.join(tokens.split(' ')[:self.max_positions - 2])
        bpe_sentence = '<s> ' + tokens + ' </s>'
        for s in addl_sentences:
            bpe_sentence += ' </s>' if not no_separator else ''
            bpe_sentence += ' ' + self.bpe.encode(s) + ' </s>'
        tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False)
        return tokens.long()

    def decode(self, tokens: 'torch.LongTensor'):
        assert tokens.dim() == 1
        tokens = tokens.cpu().numpy()
        if tokens[0] == self.task.source_dictionary.bos():
            tokens = tokens[1:]
        eos_mask = tokens == self.task.source_dictionary.eos()
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [self.bpe.decode(self.task.source_dictionary.string(s)) for s in sentences]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    def _build_sample(self, src_tokens: 'List[torch.LongTensor]'):
        dataset = self.task.build_dataset_for_inference(src_tokens, [x.numel() for x in src_tokens])
        sample = dataset.collater(dataset)
        sample = utils.apply_to_sample(lambda tensor: tensor, sample)
        return sample

    def sample(self, sentences: 'List[str]', beam: 'int'=1, verbose: 'bool'=False, **kwargs) ->str:
        input = [self.encode(sentence) for sentence in sentences]
        hypos = self.generate(input, beam, verbose, **kwargs)
        return [self.decode(x['tokens']) for x in hypos]

    def generate(self, tokens: 'List[torch.LongTensor]', beam: 'int'=5, verbose: 'bool'=False, **kwargs) ->torch.LongTensor:
        sample = self._build_sample(tokens)
        gen_args = copy.copy(self.args)
        gen_args.beam = beam
        for k, v in kwargs.items():
            setattr(gen_args, k, v)
        generator = self.task.build_generator([self.model], gen_args)
        translations = self.task.inference_step(generator, [self.model], sample, prefix_tokens=sample['net_input']['src_tokens'].new_zeros((len(tokens), 1)).fill_(self.task.source_dictionary.bos()))
        if verbose:
            src_str_with_unk = self.string(tokens)
            logger.info('S\t{}'.format(src_str_with_unk))

        def getarg(name, default):
            return getattr(gen_args, name, getattr(self.args, name, default))
        hypos = [x[0] for x in translations]
        hypos = [v for _, v in sorted(zip(sample['id'].tolist(), hypos))]
        return hypos

    def extract_features(self, tokens: 'torch.LongTensor', return_all_hiddens: 'bool'=False) ->torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > min(self.model.max_positions()):
            raise ValueError('tokens exceeds maximum length: {} > {}'.format(tokens.size(-1), self.model.max_positions()))
        tokens,
        prev_output_tokens = tokens.clone()
        prev_output_tokens[:, 0] = tokens.gather(1, (tokens.ne(self.task.source_dictionary.pad()).sum(dim=1) - 1).unsqueeze(-1)).squeeze()
        prev_output_tokens[:, 1:] = tokens[:, :-1]
        features, extra = self.model(src_tokens=tokens, src_lengths=None, prev_output_tokens=prev_output_tokens, features_only=True, return_all_hiddens=return_all_hiddens)
        if return_all_hiddens:
            inner_states = extra['inner_states']
            return [inner_state.transpose(0, 1) for inner_state in inner_states]
        else:
            return features

    def register_classification_head(self, name: 'str', num_classes: 'int'=None, embedding_size: 'int'=None, **kwargs):
        self.model.register_classification_head(name, num_classes=num_classes, embedding_size=embedding_size, **kwargs)

    def predict(self, head: 'str', tokens: 'torch.LongTensor', return_logits: 'bool'=False):
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        features = self.extract_features(tokens)
        sentence_representation = features[tokens.eq(self.task.source_dictionary.eos()), :].view(features.size(0), -1, features.size(-1))[:, -1, :]
        logits = self.model.classification_heads[head](sentence_representation)
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)


class BARTClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout, do_spectral_norm=False):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)
        if do_spectral_norm:
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class FairseqDecoder(nn.Module):
    """Base class for decoders."""

    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary
        self.onnx_trace = False

    def forward(self, prev_output_tokens, encoder_out=None, **kwargs):
        """
        Args:
            prev_output_tokens (LongTensor): shifted output tokens of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (dict, optional): output from the encoder, used for
                encoder-side attention

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        x = self.output_layer(x)
        return x, extra

    def extract_features(self, prev_output_tokens, encoder_out=None, **kwargs):
        """
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        raise NotImplementedError

    def output_layer(self, features, **kwargs):
        """
        Project features to the default output size, e.g., vocabulary size.

        Args:
            features (Tensor): features returned by *extract_features*.
        """
        raise NotImplementedError

    def get_normalized_probs(self, net_output: 'Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]]', log_probs: 'bool', sample: 'Optional[Dict[str, Tensor]]'=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        if hasattr(self, 'adaptive_softmax') and self.adaptive_softmax is not None:
            if sample is not None:
                assert 'target' in sample
                target = sample['target']
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out
        logits = net_output[0]
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)

    def max_positions(self):
        """Maximum input length supported by the decoder."""
        return 1000000.0

    def upgrade_state_dict(self, state_dict):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True


class FairseqEncoder(nn.Module):
    """Base class for encoders."""

    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of shape
                `(batch)`
        """
        raise NotImplementedError

    def forward_torchscript(self, net_input: 'Dict[str, Tensor]'):
        """A TorchScript-compatible version of forward.

        Encoders which use additional arguments may want to override
        this method for TorchScript compatibility.
        """
        if torch.jit.is_scripting():
            return self.forward(src_tokens=net_input['src_tokens'], src_lengths=net_input['src_lengths'])
        else:
            return self.forward_non_torchscript(net_input)

    @torch.jit.unused
    def forward_non_torchscript(self, net_input: 'Dict[str, Tensor]'):
        encoder_input = {k: v for k, v in net_input.items() if k != 'prev_output_tokens'}
        return self.forward(**encoder_input)

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to `new_order`.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            `encoder_out` rearranged according to `new_order`
        """
        raise NotImplementedError

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return 1000000.0

    def upgrade_state_dict(self, state_dict):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""

        def _apply(m):
            if hasattr(m, 'set_num_updates') and m != self:
                m.set_num_updates(num_updates)
        self.apply(_apply)


@with_incremental_state
class FairseqIncrementalDecoder(FairseqDecoder):
    """Base class for incremental decoders.

    Incremental decoding is a special mode at inference time where the Model
    only receives a single timestep of input corresponding to the previous
    output token (for teacher forcing) and must produce the next output
    *incrementally*. Thus the model must cache any long-term state that is
    needed about the sequence, e.g., hidden states, convolutional states, etc.

    Compared to the standard :class:`FairseqDecoder` interface, the incremental
    decoder interface allows :func:`forward` functions to take an extra keyword
    argument (*incremental_state*) that can be used to cache state across
    time-steps.

    The :class:`FairseqIncrementalDecoder` interface also defines the
    :func:`reorder_incremental_state` method, which is used during beam search
    to select and reorder the incremental state based on the selection of beams.

    To learn more about how incremental decoding works, refer to `this blog
    <http://www.telesens.co/2019/04/21/understanding-incremental-decoding-in-fairseq/>`_.
    """

    def __init__(self, dictionary):
        super().__init__(dictionary)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        """
        Args:
            prev_output_tokens (LongTensor): shifted output tokens of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (dict, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict, optional): dictionary used for storing
                state during :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        raise NotImplementedError

    def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        """
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        raise NotImplementedError

    def reorder_incremental_state(self, incremental_state: 'Dict[str, Dict[str, Optional[Tensor]]]', new_order: 'Tensor'):
        """Reorder incremental state.

        This will be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """
        pass

    def reorder_incremental_state_scripting(self, incremental_state: 'Dict[str, Dict[str, Optional[Tensor]]]', new_order: 'Tensor'):
        """Main entry point for reordering the incremental state.

        Due to limitations in TorchScript, we call this function in
        :class:`fairseq.sequence_generator.SequenceGenerator` instead of
        calling :func:`reorder_incremental_state` directly.
        """
        for module in self.modules():
            if hasattr(module, 'reorder_incremental_state'):
                result = module.reorder_incremental_state(incremental_state, new_order)
                if result is not None:
                    incremental_state = result

    def set_beam_size(self, beam_size):
        """Sets the beam size in the decoder and all children."""
        if getattr(self, '_beam_size', -1) != beam_size:
            seen = set()

            def apply_set_beam_size(module):
                if module != self and hasattr(module, 'set_beam_size') and module not in seen:
                    seen.add(module)
                    module.set_beam_size(beam_size)
            self.apply(apply_set_beam_size)
            self._beam_size = beam_size


class FairseqEncoderDecoderModel(BaseFairseqModel):
    """Base class for encoder-decoder models.

    Args:
        encoder (FairseqEncoder): the encoder
        decoder (FairseqDecoder): the decoder
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        assert isinstance(self.encoder, FairseqEncoder)
        assert isinstance(self.decoder, FairseqDecoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

    def extract_features(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        features = self.decoder.extract_features(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return features

    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return self.encoder.max_positions(), self.decoder.max_positions()

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()


class FairseqModel(FairseqEncoderDecoderModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        utils.deprecation_warning('FairseqModel is deprecated, please use FairseqEncoderDecoderModel or BaseFairseqModel instead', stacklevel=4)


class FairseqMultiModel(BaseFairseqModel):
    """Base class for combining multiple encoder-decoder models."""

    def __init__(self, encoders, decoders):
        super().__init__()
        assert encoders.keys() == decoders.keys()
        self.keys = list(encoders.keys())
        for key in self.keys:
            assert isinstance(encoders[key], FairseqEncoder)
            assert isinstance(decoders[key], FairseqDecoder)
        self.models = nn.ModuleDict({key: FairseqEncoderDecoderModel(encoders[key], decoders[key]) for key in self.keys})

    @staticmethod
    def build_shared_embeddings(dicts: 'Dict[str, Dictionary]', langs: 'List[str]', embed_dim: 'int', build_embedding: 'callable', pretrained_embed_path: 'Optional[str]'=None):
        """
        Helper function to build shared embeddings for a set of languages after
        checking that all dicts corresponding to those languages are equivalent.

        Args:
            dicts: Dict of lang_id to its corresponding Dictionary
            langs: languages that we want to share embeddings for
            embed_dim: embedding dimension
            build_embedding: callable function to actually build the embedding
            pretrained_embed_path: Optional path to load pretrained embeddings
        """
        shared_dict = dicts[langs[0]]
        if any(dicts[lang] != shared_dict for lang in langs):
            raise ValueError('--share-*-embeddings requires a joined dictionary: --share-encoder-embeddings requires a joined source dictionary, --share-decoder-embeddings requires a joined target dictionary, and --share-all-embeddings requires a joint source + target dictionary.')
        return build_embedding(shared_dict, embed_dim, pretrained_embed_path)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        raise NotImplementedError

    def max_positions(self):
        """Maximum length supported by the model."""
        return {key: (self.models[key].encoder.max_positions(), self.models[key].decoder.max_positions()) for key in self.keys}

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return min(model.decoder.max_positions() for model in self.models.values())

    @property
    def encoder(self):
        return self.models[self.keys[0]].encoder

    @property
    def decoder(self):
        return self.models[self.keys[0]].decoder

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

    def load_state_dict(self, state_dict, strict=True, args=None):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        self.upgrade_state_dict(state_dict)
        new_state_dict = prune_state_dict(state_dict, args)
        return super().load_state_dict(new_state_dict, strict)


class FairseqLanguageModel(BaseFairseqModel):
    """Base class for decoder-only models.

    Args:
        decoder (FairseqDecoder): the decoder
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        assert isinstance(self.decoder, FairseqDecoder)

    def forward(self, src_tokens, **kwargs):
        """
        Run the forward pass for a decoder-only model.

        Feeds a batch of tokens through the decoder to predict the next tokens.

        Args:
            src_tokens (LongTensor): tokens on which to condition the decoder,
                of shape `(batch, tgt_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, seq_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        return self.decoder(src_tokens, **kwargs)

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

    def extract_features(self, src_tokens, **kwargs):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, seq_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        return self.decoder.extract_features(src_tokens, **kwargs)

    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return self.decoder.max_positions()

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()

    @property
    def supported_targets(self):
        return {'future'}


class FairseqEncoderModel(BaseFairseqModel):
    """Base class for encoder-only models.

    Args:
        encoder (FairseqEncoder): the encoder
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        assert isinstance(self.encoder, FairseqEncoder)

    def forward(self, src_tokens, src_lengths, **kwargs):
        """
        Run the forward pass for a encoder-only model.

        Feeds a batch of tokens through the encoder to generate features.

        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`

        Returns:
            the encoder's output, typically of shape `(batch, src_len, features)`
        """
        return self.encoder(src_tokens, src_lengths, **kwargs)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        encoder_out = net_output['encoder_out']
        if torch.is_tensor(encoder_out):
            logits = encoder_out.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    def max_positions(self):
        """Maximum length supported by the model."""
        return self.encoder.max_positions()


class Downsample(nn.Module):
    """
    Selects every nth element, where n is the index
    """

    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[::self.index + 1]


def GatedLinear(in_features, out_features, dropout=0.0, bias=True):
    """Weight-normalized Linear layer (input: B x T x C) with interspersed GLU units"""
    return nn.Sequential(Linear(in_features, out_features * 4, dropout, bias), nn.GLU(), Linear(out_features * 2, out_features * 2, dropout, bias), nn.GLU(), Linear(out_features, out_features, dropout, bias))


class ScalarBias(torch.autograd.Function):
    """
    Adds a vector of scalars, used in self-attention mechanism to allow
    the model to optionally attend to this vector instead of the past
    """

    @staticmethod
    def forward(ctx, input, dim, bias_init):
        size = list(input.size())
        size[dim] += 1
        output = input.new(*size).fill_(bias_init)
        output.narrow(dim, 1, size[dim] - 1).copy_(input)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad):
        return grad.narrow(ctx.dim, 1, grad.size(ctx.dim) - 1), None, None


def scalar_bias(input, dim, bias_init=0):
    return ScalarBias.apply(input, dim, bias_init)


class SingleHeadAttention(nn.Module):
    """
    Single-head attention that supports Gating and Downsampling
    """

    def __init__(self, out_channels, embed_dim, head_dim, head_index, dropout=0.0, bias=True, project_input=True, gated=False, downsample=False, num_heads=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.head_index = head_index
        self.head_dim = head_dim
        self.project_input = project_input
        self.gated = gated
        self.downsample = downsample
        self.num_heads = num_heads
        self.projection = None
        k_layers = []
        v_layers = []
        if self.downsample:
            k_layers.append(Downsample(self.head_index))
            v_layers.append(Downsample(self.head_index))
            out_proj_size = self.head_dim
        else:
            out_proj_size = self.head_dim * self.num_heads
        if self.gated:
            k_layers.append(GatedLinear(self.embed_dim, out_proj_size, bias=bias))
            self.in_proj_q = GatedLinear(self.embed_dim, out_proj_size, bias=bias)
            v_layers.append(GatedLinear(self.embed_dim, out_proj_size, bias=bias))
        else:
            k_layers.append(Linear(self.embed_dim, out_proj_size, bias=bias))
            self.in_proj_q = Linear(self.embed_dim, out_proj_size, bias=bias)
            v_layers.append(Linear(self.embed_dim, out_proj_size, bias=bias))
        self.in_proj_k = nn.Sequential(*k_layers)
        self.in_proj_v = nn.Sequential(*v_layers)
        if self.downsample:
            self.out_proj = Linear(out_proj_size, self.head_dim, bias=bias)
        else:
            self.out_proj = Linear(out_proj_size, out_channels, bias=bias)
        self.scaling = self.head_dim ** -0.5

    def forward(self, query, key, value, mask_future_timesteps=False, key_padding_mask=None, use_scalar_bias=False):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        src_len, bsz, out_channels = key.size()
        tgt_len = query.size(0)
        assert list(query.size()) == [tgt_len, bsz, out_channels]
        assert key.size() == value.size()
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        if self.downsample:
            size = bsz
        else:
            size = bsz * self.num_heads
        k = key
        v = value
        q = query
        if self.project_input:
            q = self.in_proj_q(q)
            k = self.in_proj_k(k)
            v = self.in_proj_v(v)
            src_len = k.size()[0]
        q *= self.scaling
        if not self.downsample:
            q = q.view(tgt_len, size, self.head_dim)
            k = k.view(src_len, size, self.head_dim)
            v = v.view(src_len, size, self.head_dim)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if mask_future_timesteps:
            assert query.size() == key.size(), 'mask_future_timesteps only applies to self-attention'
            attn_weights *= torch.tril(attn_weights.data.new([1]).expand(tgt_len, tgt_len).clone(), diagonal=-1)[:, ::self.head_index + 1 if self.downsample else 1].unsqueeze(0)
            attn_weights += torch.triu(attn_weights.data.new([-math.inf]).expand(tgt_len, tgt_len).clone(), diagonal=0)[:, ::self.head_index + 1 if self.downsample else 1].unsqueeze(0)
        tgt_size = tgt_len
        if use_scalar_bias:
            attn_weights = scalar_bias(attn_weights, 2)
            v = scalar_bias(v, 1)
            tgt_size += 1
        if key_padding_mask is not None:
            if key_padding_mask.max() > 0:
                if self.downsample:
                    attn_weights = attn_weights.view(bsz, 1, tgt_len, src_len)
                else:
                    attn_weights = attn_weights.view(size, self.num_heads, tgt_len, src_len)
                attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), -math.inf)
                attn_weights = attn_weights.view(size, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_module(attn_weights)
        attn = torch.bmm(attn_weights, v)
        if self.downsample:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.head_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn = self.out_proj(attn)
        return attn, attn_weights


class DownsampledMultiHeadAttention(nn.ModuleList):
    """
    Multi-headed attention with Gating and Downsampling
    """

    def __init__(self, out_channels, embed_dim, num_heads, dropout=0.0, bias=True, project_input=True, gated=False, downsample=False):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.downsample = downsample
        self.gated = gated
        self.project_input = project_input
        assert self.head_dim * num_heads == embed_dim
        if self.downsample:
            attention_heads = []
            for index in range(self.num_heads):
                attention_heads.append(SingleHeadAttention(out_channels, self.embed_dim, self.head_dim, index, dropout, bias, self.project_input, self.gated, self.downsample, self.num_heads))
            super().__init__(modules=attention_heads)
            self.out_proj = Linear(embed_dim, out_channels, bias=bias)
        else:
            super().__init__()
            self.attention_module = SingleHeadAttention(out_channels, self.embed_dim, self.head_dim, 1, dropout, bias, self.project_input, self.gated, self.downsample, self.num_heads)

    def forward(self, query, key, value, mask_future_timesteps=False, key_padding_mask=None, use_scalar_bias=False):
        src_len, bsz, embed_dim = key.size()
        tgt_len = query.size(0)
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()
        tgt_size = tgt_len
        if use_scalar_bias:
            tgt_size += 1
        attn = []
        attn_weights = []
        if self.downsample:
            for attention_head_number in range(self.num_heads):
                _attn, _attn_weight = self[attention_head_number](query, key, value, mask_future_timesteps, key_padding_mask, use_scalar_bias)
                attn.append(_attn)
                attn_weights.append(_attn_weight)
            full_attn = torch.cat(attn, dim=2)
            full_attn = self.out_proj(full_attn)
            return full_attn, attn_weights[0].clone()
        else:
            _attn, _attn_weight = self.attention_module(query, key, value, mask_future_timesteps, key_padding_mask, use_scalar_bias)
            attn.append(_attn)
            attn_weights.append(_attn_weight)
            full_attn = torch.cat(attn, dim=2)
            full_attn_weights = torch.cat(attn_weights)
            full_attn_weights = full_attn_weights.view(bsz, self.num_heads, tgt_size, src_len)
            full_attn_weights = full_attn_weights.sum(dim=1) / self.num_heads
            return full_attn, full_attn_weights


class ConvTBC(torch.nn.Module):
    """1D convolution over an input of shape (time x batch x channel)

    The implementation uses gemm to perform the convolution. This implementation
    is faster than cuDNN for small kernel sizes.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ConvTBC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.padding = _single(padding)
        self.weight = torch.nn.Parameter(torch.Tensor(self.kernel_size[0], in_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

    def forward(self, input):
        return torch.conv_tbc(input.contiguous(), self.weight, self.bias, self.padding[0])

    def __repr__(self):
        s = '{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, padding={padding}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


@with_incremental_state
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

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = ConvTBC.state_dict(self, destination, prefix, keep_vars=keep_vars)
        if prefix + '_linearized_weight' in state:
            del state[prefix + '_linearized_weight']
        return state

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != '' else ''
        if prefix + '_linearized_weight' in state_dict:
            del state_dict[prefix + '_linearized_weight']

    def forward(self, input, incremental_state=None):
        """
        Args:
            incremental_state: Used to buffer signal; if not None, then input is
                expected to contain a single frame. If the input order changes
                between time steps, call reorder_incremental_state.
        Input:
            Time x Batch x Channel during training
            Batch x Time x Channel during inference
        """
        if incremental_state is None:
            output = super().forward(input)
            if self.kernel_size[0] > 1 and self.padding[0] > 0:
                output = output[:-self.padding[0], :, :]
            return output
        weight = self._get_linearized_weight()
        kw = self.kernel_size[0]
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
        return output.view(bsz, 1, -1)

    def reorder_incremental_state(self, incremental_state, new_order):
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            input_buffer = input_buffer.index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(self, incremental_state, 'input_buffer')

    def _set_input_buffer(self, incremental_state, new_buffer):
        return utils.set_incremental_state(self, incremental_state, 'input_buffer', new_buffer)

    def _get_linearized_weight(self):
        if self._linearized_weight is None:
            kw = self.kernel_size[0]
            weight = self.weight.transpose(2, 1).transpose(1, 0).contiguous()
            assert weight.size() == (self.out_channels, kw, self.in_channels)
            self._linearized_weight = torch.nn.Parameter(weight.view(self.out_channels, -1))
        return self._linearized_weight

    def _clear_linearized_weight(self, *args):
        self._linearized_weight = None


def LinearizedConv1d(in_channels, out_channels, kernel_size, dropout=0.0, **kwargs):
    """Weight-normalized Conv1d layer optimized for decoding"""
    m = LinearizedConvolution(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt(4 * (1.0 - dropout) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return m


class SelfAttention(nn.Module):

    def __init__(self, out_channels, embed_dim, num_heads, project_input=False, gated=False, downsample=False):
        super().__init__()
        self.attention = DownsampledMultiHeadAttention(out_channels, embed_dim, num_heads, dropout=0, bias=True, project_input=project_input, gated=gated, downsample=downsample)
        self.in_proj_q = Linear(out_channels, embed_dim)
        self.in_proj_k = Linear(out_channels, embed_dim)
        self.in_proj_v = Linear(out_channels, embed_dim)
        self.ln = LayerNorm(out_channels)

    def forward(self, x):
        residual = x
        query = self.in_proj_q(x)
        key = self.in_proj_k(x)
        value = self.in_proj_v(x)
        x, _ = self.attention(query, key, value, mask_future_timesteps=True, use_scalar_bias=True)
        return self.ln(x + residual)


@with_incremental_state
class FConvDecoder(FairseqDecoder):
    """Convolutional decoder"""

    def __init__(self, dictionary, embed_dim=512, out_embed_dim=256, max_positions=1024, convolutions=((512, 3),) * 8, attention=True, dropout=0.1, selfattention=False, attention_nheads=1, selfattention_nheads=1, project_input=False, gated_attention=False, downsample=False, pretrained=False, trained_decoder=None):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([2]))
        self.pretrained = pretrained
        self.pretrained_decoder = trained_decoder
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.need_attn = True
        in_channels = convolutions[0][0]

        def expand_bool_array(val):
            if isinstance(val, bool):
                return [val] * len(convolutions)
            return val
        attention = expand_bool_array(attention)
        selfattention = expand_bool_array(selfattention)
        if not isinstance(attention, list) or len(attention) != len(convolutions):
            raise ValueError('Attention is expected to be a list of booleans of length equal to the number of layers.')
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        self.embed_positions = PositionalEmbedding(max_positions, embed_dim, padding_idx)
        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.selfattention = nn.ModuleList()
        self.attproj = nn.ModuleList()
        for i, (out_channels, kernel_size) in enumerate(convolutions):
            self.projections.append(Linear(in_channels, out_channels) if in_channels != out_channels else None)
            self.convolutions.append(LinearizedConv1d(in_channels, out_channels * 2, kernel_size, padding=kernel_size - 1, dropout=dropout))
            self.attention.append(DownsampledMultiHeadAttention(out_channels, embed_dim, attention_nheads, project_input=project_input, gated=False, downsample=False) if attention[i] else None)
            self.attproj.append(Linear(out_channels, embed_dim, dropout=dropout) if attention[i] else None)
            self.selfattention.append(SelfAttention(out_channels, embed_dim, selfattention_nheads, project_input=project_input, gated=gated_attention, downsample=downsample) if selfattention[i] else None)
            in_channels = out_channels
        self.fc2 = Linear(in_channels, out_embed_dim)
        self.fc3 = Linear(out_embed_dim, num_embeddings, dropout=dropout)
        if self.pretrained:
            self.gate1 = nn.Sequential(Linear(out_embed_dim * 2, out_embed_dim), nn.Sigmoid())
            self.gate2 = nn.Sequential(Linear(out_embed_dim * 2, out_embed_dim), nn.Sigmoid())
            self.joining = nn.Sequential(Linear(out_embed_dim * 2, out_embed_dim * 2), LayerNorm(out_embed_dim * 2), nn.GLU(), Linear(out_embed_dim, out_embed_dim * 2), LayerNorm(out_embed_dim * 2), nn.GLU(), Linear(out_embed_dim, out_embed_dim), LayerNorm(out_embed_dim))
            self.pretrained_outputs = {}

            def save_output():

                def hook(a, b, output):
                    self.pretrained_outputs['out'] = output
                return hook
            self.pretrained_decoder.fc2.register_forward_hook(save_output())

    def forward(self, prev_output_tokens, encoder_out):
        trained_encoder_out = encoder_out['pretrained'] if self.pretrained else None
        encoder_out = encoder_out['encoder']['encoder_out']
        encoder_a, encoder_b = self._split_encoder_out(encoder_out)
        positions = self.embed_positions(prev_output_tokens)
        x = self.embed_tokens(prev_output_tokens) + positions
        x = self.dropout_module(x)
        target_embedding = x.transpose(0, 1)
        x = self.fc1(x)
        x = x.transpose(0, 1)
        avg_attn_scores = None
        for proj, conv, attention, selfattention, attproj in zip(self.projections, self.convolutions, self.attention, self.selfattention, self.attproj):
            residual = x if proj is None else proj(x)
            x = self.dropout_module(x)
            x = conv(x)
            x = F.glu(x, dim=2)
            if attention is not None:
                r = x
                x, attn_scores = attention(attproj(x) + target_embedding, encoder_a, encoder_b)
                x = x + r
                if not self.training and self.need_attn:
                    if avg_attn_scores is None:
                        avg_attn_scores = attn_scores
                    else:
                        avg_attn_scores.add_(attn_scores)
            if selfattention is not None:
                x = selfattention(x)
            x = (x + residual) * math.sqrt(0.5)
        x = x.transpose(0, 1)
        x = self.fc2(x)
        x = self.dropout_module(x)
        if not self.pretrained:
            x = self.fc3(x)
        if self.pretrained:
            trained_x, _ = self.pretrained_decoder.forward(prev_output_tokens, trained_encoder_out)
            y = torch.cat([x, self.pretrained_outputs['out']], dim=-1)
            gate1 = self.gate1(y)
            gate2 = self.gate2(y)
            gated_x1 = gate1 * x
            gated_x2 = gate2 * self.pretrained_outputs['out']
            fusion = torch.cat([gated_x1, gated_x2], dim=-1)
            fusion = self.joining(fusion)
            fusion_output = self.fc3(fusion)
            return fusion_output, avg_attn_scores
        else:
            return x, avg_attn_scores

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def _split_encoder_out(self, encoder_out):
        """Split and transpose encoder outputs."""
        encoder_a, encoder_b = encoder_out
        encoder_a = encoder_a.transpose(0, 1).contiguous()
        encoder_b = encoder_b.transpose(0, 1).contiguous()
        result = encoder_a, encoder_b
        return result


class GradMultiply(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class FConvEncoder(FairseqEncoder):
    """Convolutional encoder"""

    def __init__(self, dictionary, embed_dim=512, max_positions=1024, convolutions=((512, 3),) * 20, dropout=0.1, attention=False, attention_nheads=1):
        super().__init__(dictionary)
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.num_attention_layers = None
        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        self.embed_positions = PositionalEmbedding(max_positions, embed_dim, self.padding_idx)

        def expand_bool_array(val):
            if isinstance(val, bool):
                return [val] * len(convolutions)
            return val
        attention = expand_bool_array(attention)
        in_channels = convolutions[0][0]
        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.attproj = nn.ModuleList()
        for i, (out_channels, kernel_size) in enumerate(convolutions):
            self.projections.append(Linear(in_channels, out_channels) if in_channels != out_channels else None)
            self.convolutions.append(ConvTBC(in_channels, out_channels * 2, kernel_size, dropout=dropout))
            self.attention.append(SelfAttention(out_channels, embed_dim, attention_nheads) if attention[i] else None)
            in_channels = out_channels
        self.fc2 = Linear(in_channels, embed_dim)

    def forward(self, src_tokens, src_lengths):
        x = self.embed_tokens(src_tokens) + self.embed_positions(src_tokens)
        x = self.dropout_module(x)
        input_embedding = x.transpose(0, 1)
        x = self.fc1(x)
        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()
        if not encoder_padding_mask.any():
            encoder_padding_mask = None
        x = x.transpose(0, 1)
        for proj, conv, attention in zip(self.projections, self.convolutions, self.attention):
            residual = x if proj is None else proj(x)
            if encoder_padding_mask is not None:
                x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)
            x = self.dropout_module(x)
            padding_l = (conv.kernel_size[0] - 1) // 2
            padding_r = conv.kernel_size[0] // 2
            x = F.pad(x, (0, 0, 0, 0, padding_l, padding_r))
            x = conv(x)
            x = F.glu(x, dim=2)
            if attention is not None:
                x = attention(x)
            x = (x + residual) * math.sqrt(0.5)
        x = x.transpose(1, 0)
        x = self.fc2(x)
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.t()
            x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)
        x = GradMultiply.apply(x, 1.0 / (2.0 * self.num_attention_layers))
        y = (x + input_embedding.transpose(0, 1)) * math.sqrt(0.5)
        return {'encoder_out': (x, y), 'encoder_padding_mask': encoder_padding_mask}

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out['encoder_out'] = tuple(eo.index_select(0, new_order) for eo in encoder_out['encoder_out'])
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].index_select(0, new_order)
        if 'pretrained' in encoder_out:
            encoder_out['pretrained']['encoder_out'] = tuple(eo.index_select(0, new_order) for eo in encoder_out['pretrained']['encoder_out'])
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions


class FConvModel(FairseqEncoderDecoderModel):
    """
    A fully convolutional model, i.e. a convolutional encoder and a
    convolutional decoder, as described in `"Convolutional Sequence to Sequence
    Learning" (Gehring et al., 2017) <https://arxiv.org/abs/1705.03122>`_.

    Args:
        encoder (FConvEncoder): the encoder
        decoder (FConvDecoder): the decoder

    The Convolutional model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.fconv_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):

        def moses_subword(path):
            return {'path': path, 'tokenizer': 'moses', 'bpe': 'subword_nmt'}
        return {'conv.wmt14.en-fr': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt14.v2.en-fr.fconv-py.tar.bz2'), 'conv.wmt14.en-de': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-de.fconv-py.tar.bz2'), 'conv.wmt17.en-de': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt17.v2.en-de.fconv-py.tar.bz2')}

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.encoder.num_attention_layers = sum(layer is not None for layer in decoder.attention)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D', help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N', help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR', help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-layers', type=str, metavar='EXPR', help='encoder layers [(dim, kernel_size), ...]')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N', help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR', help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-layers', type=str, metavar='EXPR', help='decoder layers [(dim, kernel_size), ...]')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N', help='decoder output embedding dimension')
        parser.add_argument('--decoder-attention', type=str, metavar='EXPR', help='decoder attention [True, ...]')
        parser.add_argument('--share-input-output-embed', action='store_true', help='share input and output embeddings (requires --decoder-out-embed-dim and --decoder-embed-dim to be equal)')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)
        encoder_embed_dict = None
        if args.encoder_embed_path:
            encoder_embed_dict = utils.parse_embedding(args.encoder_embed_path)
            utils.print_embed_overlap(encoder_embed_dict, task.source_dictionary)
        decoder_embed_dict = None
        if args.decoder_embed_path:
            decoder_embed_dict = utils.parse_embedding(args.decoder_embed_path)
            utils.print_embed_overlap(decoder_embed_dict, task.target_dictionary)
        encoder = FConvEncoder(dictionary=task.source_dictionary, embed_dim=args.encoder_embed_dim, embed_dict=encoder_embed_dict, convolutions=eval(args.encoder_layers), dropout=args.dropout, max_positions=args.max_source_positions)
        decoder = FConvDecoder(dictionary=task.target_dictionary, embed_dim=args.decoder_embed_dim, embed_dict=decoder_embed_dict, convolutions=eval(args.decoder_layers), out_embed_dim=args.decoder_out_embed_dim, attention=eval(args.decoder_attention), dropout=args.dropout, max_positions=args.max_target_positions, share_embed=args.share_input_output_embed)
        return FConvModel(encoder, decoder)


class AttentionLayer(nn.Module):

    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        super().__init__()
        self.input_proj = Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = Linear(input_embed_dim + source_embed_dim, output_embed_dim, bias=bias)

    def forward(self, input, source_hids, encoder_padding_mask):
        x = self.input_proj(input)
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(encoder_padding_mask, float('-inf')).type_as(attn_scores)
        attn_scores = F.softmax(attn_scores, dim=0)
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)
        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores


class CompositeEncoder(FairseqEncoder):
    """
    A wrapper around a dictionary of :class:`FairseqEncoder` objects.

    We run forward on each encoder and return a dictionary of outputs. The first
    encoder's dictionary is used for initialization.

    Args:
        encoders (dict): a dictionary of :class:`FairseqEncoder` objects.
    """

    def __init__(self, encoders):
        super().__init__(next(iter(encoders.values())).dictionary)
        self.encoders = encoders
        for key in self.encoders:
            self.add_module(key, self.encoders[key])

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of shape
                `(batch)`

        Returns:
            dict:
                the outputs from each Encoder
        """
        encoder_out = {}
        for key in self.encoders:
            encoder_out[key] = self.encoders[key](src_tokens, src_lengths)
        return encoder_out

    def reorder_encoder_out(self, encoder_out, new_order):
        """Reorder encoder output according to new_order."""
        for key in self.encoders:
            encoder_out[key] = self.encoders[key].reorder_encoder_out(encoder_out[key], new_order)
        return encoder_out

    def max_positions(self):
        return min(self.encoders[key].max_positions() for key in self.encoders)

    def upgrade_state_dict(self, state_dict):
        for key in self.encoders:
            self.encoders[key].upgrade_state_dict(state_dict)
        return state_dict


class FConvModelSelfAtt(FairseqEncoderDecoderModel):

    @classmethod
    def hub_models(cls):
        return {'conv.stories.pretrained': {'path': 'https://dl.fbaipublicfiles.com/fairseq/models/stories_checkpoint.tar.gz', 'checkpoint_file': 'pretrained_checkpoint.pt', 'tokenizer': 'nltk'}, 'conv.stories': {'path': 'https://dl.fbaipublicfiles.com/fairseq/models/stories_checkpoint.tar.gz', 'checkpoint_file': 'fusion_checkpoint.pt', 'tokenizer': 'nltk', 'pretrained': 'True', 'pretrained_checkpoint': './pretrained_checkpoint.pt'}, 'data.stories': 'https://dl.fbaipublicfiles.com/fairseq/data/stories_test.tar.bz2'}

    def __init__(self, encoder, decoder, pretrained_encoder=None):
        super().__init__(encoder, decoder)
        self.encoder.num_attention_layers = sum(layer is not None for layer in decoder.attention)
        self.pretrained_encoder = pretrained_encoder
        if self.pretrained_encoder is None:
            encoders = {'encoder': encoder}
        else:
            encoders = {'encoder': encoder, 'pretrained': self.pretrained_encoder}
        self.encoder = CompositeEncoder(encoders)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D', help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N', help='encoder embedding dimension')
        parser.add_argument('--encoder-layers', type=str, metavar='EXPR', help='encoder layers [(dim, kernel_size), ...]')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N', help='decoder embedding dimension')
        parser.add_argument('--decoder-layers', type=str, metavar='EXPR', help='decoder layers [(dim, kernel_size), ...]')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N', help='decoder output embedding dimension')
        parser.add_argument('--decoder-attention', type=str, metavar='EXPR', help='decoder attention [True, ...]')
        parser.add_argument('--self-attention', type=str, metavar='EXPR', help='decoder self-attention layers, ex: [True] + [False]*5')
        parser.add_argument('--multihead-attention-nheads', type=int, help='Number of heads to use in attention')
        parser.add_argument('--multihead-self-attention-nheads', type=int, help='Number of heads to use in self-attention')
        parser.add_argument('--encoder-attention', type=str, metavar='EXPR', help='encoder attention [True, ...]')
        parser.add_argument('--encoder-attention-nheads', type=int, help='Number of heads to use in encoder attention')
        parser.add_argument('--project-input', type=str, metavar='EXPR', help='Use projections in self-attention [True, ...]')
        parser.add_argument('--gated-attention', type=str, metavar='EXPR', help='Use GLU layers in self-attention projections [True, ...]')
        parser.add_argument('--downsample', type=str, metavar='EXPR', help='Use downsampling in self-attention [True, ...]')
        parser.add_argument('--pretrained-checkpoint', metavar='DIR', help='path to load checkpoint from pretrained model')
        parser.add_argument('--pretrained', type=str, metavar='EXPR', help='use pretrained model when training [True, ...]')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        trained_encoder, trained_decoder = None, None
        pretrained = eval(args.pretrained)
        if pretrained:
            logger.info('loading pretrained model')
            if not os.path.exists(args.pretrained_checkpoint):
                new_pretrained_checkpoint = os.path.join(args.data, args.pretrained_checkpoint)
                if os.path.exists(new_pretrained_checkpoint):
                    args.pretrained_checkpoint = new_pretrained_checkpoint
            trained_model = checkpoint_utils.load_model_ensemble(filenames=[args.pretrained_checkpoint], task=task)[0][0]
            trained_decoder = list(trained_model.children())[1]
            trained_encoder = list(trained_model.children())[0]
            for param in trained_decoder.parameters():
                param.requires_grad = False
            for param in trained_encoder.parameters():
                param.requires_grad = False
        encoder = FConvEncoder(task.source_dictionary, embed_dim=args.encoder_embed_dim, convolutions=eval(args.encoder_layers), dropout=args.dropout, max_positions=args.max_source_positions, attention=eval(args.encoder_attention), attention_nheads=args.encoder_attention_nheads)
        decoder = FConvDecoder(task.target_dictionary, embed_dim=args.decoder_embed_dim, convolutions=eval(args.decoder_layers), out_embed_dim=args.decoder_out_embed_dim, attention=eval(args.decoder_attention), dropout=args.dropout, max_positions=args.max_target_positions, selfattention=eval(args.self_attention), attention_nheads=args.multihead_attention_nheads, selfattention_nheads=args.multihead_self_attention_nheads, project_input=eval(args.project_input), gated_attention=eval(args.gated_attention), downsample=eval(args.downsample), pretrained=pretrained, trained_decoder=trained_decoder)
        model = FConvModelSelfAtt(encoder, decoder, trained_encoder)
        return model

    @property
    def pretrained(self):
        return self.pretrained_encoder is not None


DEFAULT_MAX_TARGET_POSITIONS = 1024


class HuggingFaceGPT2LanguageModel(FairseqLanguageModel):

    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--embed-dim', type=int, metavar='N', help='embedding dimension')
        parser.add_argument('--num-attention-heads', type=int, metavar='N', help='num attention heads')
        parser.add_argument('--num-layers', type=int, metavar='N', help='num layers')
        parser.add_argument('--dropout', type=float, metavar='D', help='dropout probability for all fully connected layers in the embeddings, encoder, and pooler')
        parser.add_argument('--attention-dropout', type=float, metavar='D', help='dropout probability for attention weights')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        default_architecture(args)
        return cls(HuggingFaceGPT2Decoder(args, task))


def unfold1d(x, kernel_size, padding_l, pad_value=0):
    """unfold T x B x C to T x B x C x K"""
    if kernel_size > 1:
        T, B, C = x.size()
        x = F.pad(x, (0, 0, 0, 0, padding_l, kernel_size - 1 - padding_l), value=pad_value)
        x = x.as_strided((T, B, C, kernel_size), (B * C, C, 1, B * C))
    else:
        x = x.unsqueeze(3)
    return x


@with_incremental_state
class DynamicConv1dTBC(nn.Module):
    """Dynamic lightweight convolution taking T x B x C inputs
    Args:
        input_size: # of channels of the input
        kernel_size: convolution channels
        padding_l: padding to the left when using "same" padding
        num_heads: number of heads used. The weight is of shape (num_heads, 1, kernel_size)
        weight_dropout: the drop rate of the DropConnect to drop the weight
        weight_softmax: normalize the weight with softmax before the convolution
        renorm_padding: re-normalize the filters to ignore the padded part (only the non-padding parts sum up to 1)
        bias: use bias
        conv_bias: bias of the convolution
        query_size: specified when feeding a different input as the query
        in_proj: project the input and generate the filter together

    Shape:
        Input: TxBxC, i.e. (timesteps, batch_size, input_size)
        Output: TxBxC, i.e. (timesteps, batch_size, input_size)

    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias:   the learnable bias of the module of shape `(input_size)`
    """

    def __init__(self, input_size, kernel_size=1, padding_l=None, num_heads=1, weight_dropout=0.0, weight_softmax=False, renorm_padding=False, bias=False, conv_bias=False, query_size=None, in_proj=False):
        super().__init__()
        self.input_size = input_size
        self.query_size = input_size if query_size is None else query_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_dropout_module = FairseqDropout(weight_dropout, module_name=self.__class__.__name__)
        self.weight_softmax = weight_softmax
        self.renorm_padding = renorm_padding
        if in_proj:
            self.weight_linear = Linear(self.input_size, self.input_size + num_heads * kernel_size * 1)
        else:
            self.weight_linear = Linear(self.query_size, num_heads * kernel_size * 1, bias=bias)
        if conv_bias:
            self.conv_bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.conv_bias = None
        self.reset_parameters()

    @property
    def in_proj(self):
        return self.weight_linear.out_features == self.input_size + self.num_heads * self.kernel_size

    def reset_parameters(self):
        self.weight_linear.reset_parameters()
        if self.conv_bias is not None:
            nn.init.constant_(self.conv_bias, 0.0)

    def forward(self, x, incremental_state=None, query=None, unfold=None):
        """Assuming the input, x, of the shape T x B x C and producing an output in the shape T x B x C
        args:
            x: Input of shape T x B x C, i.e. (timesteps, batch_size, input_size)
            incremental_state: A dict to keep the state
            unfold: unfold the input or not. If not, we use the matrix trick instead
            query: use the specified query to predict the conv filters
        """
        unfold = x.size(0) > 512 if unfold is None else unfold
        unfold = unfold or incremental_state is not None
        assert query is None or not self.in_proj
        if query is None:
            query = x
        if unfold:
            output = self._forward_unfolded(x, incremental_state, query)
        else:
            output = self._forward_expanded(x, incremental_state, query)
        if self.conv_bias is not None:
            output = output + self.conv_bias.view(1, 1, -1)
        return output

    def _forward_unfolded(self, x, incremental_state, query):
        """The conventional implementation of convolutions.
        Unfolding the input by having a window shifting to the right."""
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size
        if self.in_proj:
            proj = self.weight_linear(x)
            x = proj.narrow(2, 0, self.input_size).contiguous()
            weight = proj.narrow(2, self.input_size, H * K).contiguous().view(T * B * H, -1)
        else:
            weight = self.weight_linear(query).view(T * B * H, -1)
        assert not self.renorm_padding or incremental_state is not None
        if incremental_state is not None:
            input_buffer = self._get_input_buffer(incremental_state)
            if input_buffer is None:
                input_buffer = x.new()
            x_unfold = torch.cat([input_buffer, x.unsqueeze(3)], dim=3)
            if self.kernel_size > 1:
                self._set_input_buffer(incremental_state, x_unfold[:, :, :, -self.kernel_size + 1:])
            x_unfold = x_unfold.view(T * B * H, R, -1)
        else:
            padding_l = self.padding_l
            if K > T and padding_l == K - 1:
                weight = weight.narrow(1, K - T, T)
                K, padding_l = T, T - 1
            x_unfold = unfold1d(x, K, padding_l, 0)
            x_unfold = x_unfold.view(T * B * H, R, K)
        if self.weight_softmax and not self.renorm_padding:
            weight = F.softmax(weight, dim=1)
        weight = weight.narrow(1, 0, K)
        if incremental_state is not None:
            weight = weight[:, -x_unfold.size(2):]
            K = weight.size(1)
        if self.weight_softmax and self.renorm_padding:
            weight = F.softmax(weight, dim=1)
        weight = self.weight_dropout_module(weight, inplace=False)
        output = torch.bmm(x_unfold, weight.unsqueeze(2))
        output = output.view(T, B, C)
        return output

    def _forward_expanded(self, x, incremental_stat, query):
        """Turn the convolution filters into band matrices and do matrix multiplication.
        This is faster when the sequence is short, but less memory efficient.
        This is not used in the decoder during inference.
        """
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size
        if self.in_proj:
            proj = self.weight_linear(x)
            x = proj.narrow(2, 0, self.input_size).contiguous()
            weight = proj.narrow(2, self.input_size, H * K).contiguous().view(T * B * H, -1)
        else:
            weight = self.weight_linear(query).view(T * B * H, -1)
        if not self.renorm_padding:
            if self.weight_softmax:
                weight = F.softmax(weight, dim=1)
            weight = self.weight_dropout_module(weight, inplace=False)
        weight = weight.narrow(1, 0, K).contiguous()
        weight = weight.view(T, B * H, K).transpose(0, 1)
        x = x.view(T, B * H, R).transpose(0, 1)
        if self.weight_softmax and self.renorm_padding:
            weight_expanded = weight.new(B * H, T, T + K - 1).fill_(float('-inf'))
            weight_expanded.as_strided((B * H, T, K), (T * (T + K - 1), T + K, 1)).copy_(weight)
            weight_expanded = weight_expanded.narrow(2, self.padding_l, T)
            weight_expanded = F.softmax(weight_expanded, dim=2)
            weight_expanded = self.weight_dropout_module(weight_expanded, inplace=False)
        else:
            P = self.padding_l
            if K > T and P == K - 1:
                weight = weight.narrow(2, K - T, T)
                K, P = T, T - 1
            weight_expanded = weight.new_zeros(B * H, T, T + K - 1, requires_grad=False)
            weight_expanded.as_strided((B * H, T, K), (T * (T + K - 1), T + K, 1)).copy_(weight)
            weight_expanded = weight_expanded.narrow(2, P, T)
        output = torch.bmm(weight_expanded, x)
        output = output.transpose(0, 1).contiguous().view(T, B, C)
        return output

    def reorder_incremental_state(self, incremental_state, new_order):
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            input_buffer = input_buffer.index_select(1, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(self, incremental_state, 'input_buffer')

    def _set_input_buffer(self, incremental_state, new_buffer):
        return utils.set_incremental_state(self, incremental_state, 'input_buffer', new_buffer)

    def extra_repr(self):
        s = '{}, kernel_size={}, padding_l={}, num_heads={}, weight_softmax={}, conv_bias={}, renorm_padding={}, in_proj={}'.format(self.input_size, self.kernel_size, self.padding_l, self.num_heads, self.weight_softmax, self.conv_bias is not None, self.renorm_padding, self.in_proj)
        if self.query_size != self.input_size:
            s += ', query_size={}'.format(self.query_size)
        if self.weight_dropout_module.p > 0.0:
            s += ', weight_dropout={}'.format(self.weight_dropout_module.p)
        return s


class dynamicconvFunction(Function):

    @staticmethod
    def forward(ctx, x, weights, padding_l):
        ctx.padding_l = padding_l
        outputs = dynamicconv_cuda.forward(x, weights, padding_l)
        variables = [x, weights]
        ctx.save_for_backward(*variables)
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output):
        outputs = dynamicconv_cuda.backward(grad_output.contiguous(), ctx.padding_l, *ctx.saved_tensors)
        grad_input, grad_weights = outputs
        return grad_input, grad_weights, None


@with_incremental_state
class DynamicconvLayer(nn.Module):

    def __init__(self, input_size, kernel_size=1, padding_l=None, weight_softmax=False, num_heads=1, weight_dropout=0.0, bias=False, renorm_padding=False, conv_bias=False, query_size=None):
        super(DynamicconvLayer, self).__init__()
        self.input_size = input_size
        self.query_size = input_size if query_size is None else query_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_softmax = weight_softmax
        self.weight_dropout_module = FairseqDropout(weight_dropout, module_name=self.__class__.__name__)
        self.renorm_padding = renorm_padding
        self.bias = bias
        self.weight_linear = nn.Linear(input_size, num_heads * kernel_size, bias)
        if conv_bias:
            self.conv_bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.conv_bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_linear.weight)
        if self.conv_bias is not None:
            nn.init.constant_(self.conv_bias, 0.0)
            nn.init.constant_(self.weight_linaer.bias, 0.0)

    def forward(self, x, incremental_state=None, query=None, unfold=None):
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        if incremental_state is not None:
            unfold = x.size(0) > 512 if unfold is None else unfold
            unfold = unfold or incremental_state is not None
            assert query is None
            if query is None:
                query = x
            if unfold:
                output = self._forward_unfolded(x, incremental_state, query)
            else:
                output = self._forward_expanded(x, incremental_state, query)
            if self.conv_bias is not None:
                output = output + self.conv_bias.view(1, 1, -1)
            return output
        else:
            weight = self.weight_linear(x).view(T, B, H, K)
            if self.weight_softmax:
                weight = F.softmax(weight, dim=-1)
            if self.weight_dropout_module.p:
                weight = self.weight_dropout_module(weight)
            weight = weight.permute(1, 2, 3, 0).contiguous()
            self.filters = weight
            x = x.permute(1, 2, 0).contiguous()
            output = dynamicconvFunction.apply(x, weight, self.padding_l).permute(2, 0, 1)
            if self.conv_bias is not None:
                output = output + self.conv_bias.view(1, 1, -1)
            return output

    def reorder_incremental_state(self, incremental_state, new_order):
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            input_buffer = input_buffer.index_select(1, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(self, incremental_state, 'input_buffer')

    def _set_input_buffer(self, incremental_state, new_buffer):
        return utils.set_incremental_state(self, incremental_state, 'input_buffer', new_buffer)

    def _forward_unfolded(self, x, incremental_state, query):
        """The conventional implementation of convolutions.
        Unfolding the input by having a window shifting to the right."""
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size
        weight = self.weight_linear(query).view(T * B * H, -1)
        assert not self.renorm_padding or incremental_state is not None
        if incremental_state is not None:
            input_buffer = self._get_input_buffer(incremental_state)
            if input_buffer is None:
                input_buffer = x.new()
            x_unfold = torch.cat([input_buffer, x.unsqueeze(3)], dim=3)
            if self.kernel_size > 1:
                self._set_input_buffer(incremental_state, x_unfold[:, :, :, -self.kernel_size + 1:])
            x_unfold = x_unfold.view(T * B * H, R, -1)
        else:
            padding_l = self.padding_l
            if K > T and padding_l == K - 1:
                weight = weight.narrow(1, K - T, T)
                K, padding_l = T, T - 1
            x_unfold = unfold1d(x, K, padding_l, 0)
            x_unfold = x_unfold.view(T * B * H, R, K)
        if self.weight_softmax and not self.renorm_padding:
            weight = F.softmax(weight, dim=1)
        weight = weight.narrow(1, 0, K)
        if incremental_state is not None:
            weight = weight[:, -x_unfold.size(2):]
            K = weight.size(1)
        if self.weight_softmax and self.renorm_padding:
            weight = F.softmax(weight, dim=1)
        weight = self.weight_dropout_module(weight, inplace=False)
        output = torch.bmm(x_unfold, weight.unsqueeze(2))
        output = output.view(T, B, C)
        return output

    def _forward_expanded(self, x, incremental_stat, query):
        """Turn the convolution filters into band matrices and do matrix multiplication.
        This is faster when the sequence is short, but less memory efficient.
        This is not used in the decoder during inference.
        """
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size
        weight = self.weight_linear(query).view(T * B * H, -1)
        if not self.renorm_padding:
            if self.weight_softmax:
                weight = F.softmax(weight, dim=1)
            weight = self.weight_dropout_module(weight, inplace=False)
        weight = weight.narrow(1, 0, K).contiguous()
        weight = weight.view(T, B * H, K).transpose(0, 1)
        x = x.view(T, B * H, R).transpose(0, 1)
        if self.weight_softmax and self.renorm_padding:
            weight_expanded = weight.new(B * H, T, T + K - 1).fill_(float('-inf'))
            weight_expanded.as_strided((B * H, T, K), (T * (T + K - 1), T + K, 1)).copy_(weight)
            weight_expanded = weight_expanded.narrow(2, self.padding_l, T)
            weight_expanded = F.softmax(weight_expanded, dim=2)
            weight_expanded = self.weight_dropout_module(weight_expanded, inplace=False)
        else:
            P = self.padding_l
            if K > T and P == K - 1:
                weight = weight.narrow(2, K - T, T)
                K, P = T, T - 1
            weight_expanded = weight.new_zeros(B * H, T, T + K - 1, requires_grad=False)
            weight_expanded.as_strided((B * H, T, K), (T * (T + K - 1), T + K, 1)).copy_(weight)
            weight_expanded = weight_expanded.narrow(2, P, T)
        output = torch.bmm(weight_expanded, x)
        output = output.transpose(0, 1).contiguous().view(T, B, C)
        return output


def DynamicConv(input_size, kernel_size=1, padding_l=None, num_heads=1, weight_dropout=0.0, weight_softmax=False, renorm_padding=False, bias=False, conv_bias=False, query_size=None, in_proj=False):
    if torch.cuda.is_available():
        try:
            return DynamicconvLayer(input_size, kernel_size=kernel_size, padding_l=padding_l, num_heads=num_heads, weight_dropout=weight_dropout, weight_softmax=weight_softmax, bias=bias)
        except ImportError as e:
            None
    return DynamicConv1dTBC(input_size, kernel_size=kernel_size, padding_l=padding_l, num_heads=num_heads, weight_dropout=weight_dropout, weight_softmax=weight_softmax, bias=bias)


class lightconvFunction(Function):

    @staticmethod
    def forward(ctx, x, weights, padding_l):
        ctx.padding_l = padding_l
        outputs = lightconv_cuda.forward(x, weights, padding_l)
        variables = [x, weights]
        ctx.save_for_backward(*variables)
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output):
        outputs = lightconv_cuda.backward(grad_output.contiguous(), ctx.padding_l, *ctx.saved_tensors)
        grad_input, grad_weights = outputs
        return grad_input, grad_weights, None


@with_incremental_state
class LightconvLayer(nn.Module):

    def __init__(self, input_size, kernel_size=1, padding_l=None, weight_softmax=False, num_heads=1, weight_dropout=0.0, bias=False):
        super(LightconvLayer, self).__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_softmax = weight_softmax
        self.weight_dropout_module = FairseqDropout(weight_dropout, module_name=self.__class__.__name__)
        self.weight = nn.Parameter(torch.Tensor(num_heads, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.bias = None
        self.reset_parameters()

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != '' else ''
        for k, v in state_dict.items():
            if k.endswith(prefix + 'weight'):
                if v.dim() == 3 and v.size(1) == 1:
                    state_dict[k] = v.squeeze(1)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x, incremental_state=None):
        if incremental_state is not None:
            T, B, C = x.size()
            K, H = self.kernel_size, self.num_heads
            R = C // H
            input_buffer = self._get_input_buffer(incremental_state)
            if input_buffer is None:
                input_buffer = x.new()
            x_unfold = torch.cat([input_buffer, x.unsqueeze(3)], dim=3)
            if self.kernel_size > 1:
                self._set_input_buffer(incremental_state, x_unfold[:, :, :, -self.kernel_size + 1:])
            x_unfold = x_unfold.view(T * B * H, R, -1)
            weight = self.weight
            if self.weight_softmax:
                weight = F.softmax(weight.float(), dim=1).type_as(weight)
            weight = weight[:, -x_unfold.size(2):]
            K = weight.size(1)
            weight = weight.view(1, H, K).expand(T * B, H, K).contiguous().view(T * B * H, K, 1)
            weight = self.weight_dropout_module(weight)
            output = torch.bmm(x_unfold, weight)
            output = output.view(T, B, C)
            return output
        else:
            x = x.permute(1, 2, 0).contiguous()
            weight = self.weight
            if self.weight_softmax:
                weight = F.softmax(self.weight, -1)
            if self.weight_dropout_module.p:
                weight = self.weight_dropout_module(weight)
            return lightconvFunction.apply(x, weight, self.padding_l).permute(2, 0, 1)

    def reorder_incremental_state(self, incremental_state, new_order):
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            input_buffer = input_buffer.index_select(1, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(self, incremental_state, 'input_buffer')

    def _set_input_buffer(self, incremental_state, new_buffer):
        return utils.set_incremental_state(self, incremental_state, 'input_buffer', new_buffer)

    def half(self):
        return self._apply(lambda t: t.half() if t.is_floating_point() else t)


@with_incremental_state
class LightweightConv1dTBC(nn.Module):
    """Lightweight Convolution assuming the input is TxBxC
    Args:
        input_size: # of channels of the input
        kernel_size: convolution channels
        padding_l: padding to the left when using "same" padding
        num_heads: number of heads used. The weight is of shape (num_heads, 1, kernel_size)
        weight_dropout: the drop rate of the DropConnect to drop the weight
        weight_softmax: normalize the weight with softmax before the convolution
        bias: use bias

    Shape:
        Input: TxBxC, i.e. (timesteps, batch_size, input_size)
        Output: TxBxC, i.e. (timesteps, batch_size, input_size)

    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias:   the learnable bias of the module of shape `(input_size)`
    """

    def __init__(self, input_size, kernel_size=1, padding_l=None, num_heads=1, weight_dropout=0.0, weight_softmax=False, bias=False):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_dropout_module = FairseqDropout(weight_dropout, module_name=self.__class__.__name__)
        self.weight_softmax = weight_softmax
        self.weight = nn.Parameter(torch.Tensor(num_heads, 1, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.bias = None
        self.reset_parameters()
        self.onnx_trace = False

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x, incremental_state=None, unfold=False):
        """Assuming the input, x, of the shape T x B x C and producing an output in the shape T x B x C
        args:
            x: Input of shape T x B x C, i.e. (timesteps, batch_size, input_size)
            incremental_state: A dict to keep the state
            unfold: unfold the input or not. If not, we use the matrix trick instead
        """
        unfold = unfold or incremental_state is not None
        if unfold:
            output = self._forward_unfolded(x, incremental_state)
        else:
            output = self._forward_expanded(x, incremental_state)
        if self.bias is not None:
            output = output + self.bias.view(1, 1, -1)
        return output

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def _forward_unfolded(self, x, incremental_state):
        """The conventional implementation of convolutions.
        Unfolding the input by having a window shifting to the right."""
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size
        weight = self.weight.view(H, K)
        if incremental_state is not None:
            input_buffer = self._get_input_buffer(incremental_state)
            if input_buffer is None:
                input_buffer = x.new()
            x_unfold = torch.cat([input_buffer, x.unsqueeze(3)], dim=3)
            if self.kernel_size > 1:
                self._set_input_buffer(incremental_state, x_unfold[:, :, :, -self.kernel_size + 1:])
            x_unfold = x_unfold.view(T * B * H, R, -1)
        else:
            x_unfold = unfold1d(x, self.kernel_size, self.padding_l, 0)
            x_unfold = x_unfold.view(T * B * H, R, K)
        if self.weight_softmax:
            weight = utils.softmax(weight, dim=1, onnx_trace=self.onnx_trace).type_as(weight)
        if incremental_state is not None:
            weight = weight[:, -x_unfold.size(2):]
            K = weight.size(1)
        weight = weight.view(1, H, K).expand(T * B, H, K).contiguous().view(T * B * H, K, 1)
        weight = self.weight_dropout_module(weight)
        output = torch.bmm(x_unfold, weight)
        output = output.view(T, B, C)
        return output

    def _forward_expanded(self, x, incremental_state):
        """Turn the convolution filters into band matrices and do matrix multiplication.
        This is faster when the sequence is short, but less memory efficient.
        This is not used in the decoder during inference.
        """
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size
        weight = self.weight.view(H, K)
        if self.weight_softmax:
            weight = utils.softmax(weight, dim=1, onnx_trace=self.onnx_trace).type_as(weight)
        weight = weight.view(1, H, K).expand(T * B, H, K).contiguous()
        weight = weight.view(T, B * H, K).transpose(0, 1)
        x = x.view(T, B * H, R).transpose(0, 1)
        P = self.padding_l
        if K > T and P == K - 1:
            weight = weight.narrow(2, K - T, T)
            K, P = T, T - 1
        weight_expanded = weight.new_zeros(B * H, T, T + K - 1, requires_grad=False)
        weight_expanded.as_strided((B * H, T, K), (T * (T + K - 1), T + K, 1)).copy_(weight)
        weight_expanded = weight_expanded.narrow(2, P, T)
        weight_expanded = self.weight_dropout_module(weight_expanded)
        output = torch.bmm(weight_expanded, x)
        output = output.transpose(0, 1).contiguous().view(T, B, C)
        return output

    def reorder_incremental_state(self, incremental_state, new_order):
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            input_buffer = input_buffer.index_select(1, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(self, incremental_state, 'input_buffer')

    def _set_input_buffer(self, incremental_state, new_buffer):
        return utils.set_incremental_state(self, incremental_state, 'input_buffer', new_buffer)

    def extra_repr(self):
        s = '{}, kernel_size={}, padding_l={}, num_heads={}, weight_softmax={}, bias={}'.format(self.input_size, self.kernel_size, self.padding_l, self.num_heads, self.weight_softmax, self.bias is not None)
        if self.weight_dropout_module.p > 0.0:
            s += ', weight_dropout={}'.format(self.weight_dropout_module.p)
        return s


def LightweightConv(input_size, kernel_size=1, padding_l=None, num_heads=1, weight_dropout=0.0, weight_softmax=False, bias=False):
    if torch.cuda.is_available():
        try:
            return LightconvLayer(input_size, kernel_size=kernel_size, padding_l=padding_l, num_heads=num_heads, weight_dropout=weight_dropout, weight_softmax=weight_softmax, bias=bias)
        except ImportError as e:
            None
    return LightweightConv1dTBC(input_size, kernel_size=kernel_size, padding_l=padding_l, num_heads=num_heads, weight_dropout=weight_dropout, weight_softmax=weight_softmax, bias=bias)


class LightConvDecoderLayer(nn.Module):
    """Decoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
        kernel_size: kernel size of the convolution
    """

    def __init__(self, args, no_encoder_attn=False, kernel_size=0):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.conv_dim = args.decoder_conv_dim
        if args.decoder_glu:
            self.linear1 = Linear(self.embed_dim, 2 * self.conv_dim)
            self.act = nn.GLU()
        else:
            self.linear1 = Linear(self.embed_dim, self.conv_dim)
            self.act = None
        if args.decoder_conv_type == 'lightweight':
            self.conv = LightweightConv(self.conv_dim, kernel_size, padding_l=kernel_size - 1, weight_softmax=args.weight_softmax, num_heads=args.decoder_attention_heads, weight_dropout=args.weight_dropout)
        elif args.decoder_conv_type == 'dynamic':
            self.conv = DynamicConv(self.conv_dim, kernel_size, padding_l=kernel_size - 1, weight_softmax=args.weight_softmax, num_heads=args.decoder_attention_heads, weight_dropout=args.weight_dropout)
        else:
            raise NotImplementedError
        self.linear2 = Linear(self.conv_dim, self.embed_dim)
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.relu_dropout_module = FairseqDropout(args.relu_dropout, module_name=self.__class__.__name__)
        self.input_dropout_module = FairseqDropout(args.input_dropout, module_name=self.__class__.__name__)
        self.normalize_before = args.decoder_normalize_before
        self.conv_layer_norm = LayerNorm(self.embed_dim)
        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttention(self.embed_dim, args.decoder_attention_heads, dropout=args.attention_dropout, encoder_decoder_attention=True)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

    def forward(self, x, encoder_out, encoder_padding_mask, incremental_state, prev_conv_state=None, prev_attn_state=None, conv_mask=None, conv_padding_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.conv_layer_norm, x, before=True)
        if prev_conv_state is not None:
            if incremental_state is None:
                incremental_state = {}
            self.conv._set_input_buffer(incremental_state, prev_conv_state)
        x = self.input_dropout_module(x)
        x = self.linear1(x)
        if self.act is not None:
            x = self.act(x)
        x = self.conv(x, incremental_state=incremental_state)
        x = self.linear2(x)
        x = self.dropout_module(x)
        x = residual + x
        x = self.maybe_layer_norm(self.conv_layer_norm, x, after=True)
        attn = None
        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {'prev_key': prev_key, 'prev_value': prev_value}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            x, attn = self.encoder_attn(query=x, key=encoder_out, value=encoder_out, key_padding_mask=encoder_padding_mask, incremental_state=incremental_state, static_kv=True, need_weights=not self.training and self.need_attn)
            x = self.dropout_module(x)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)
        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = F.relu(self.fc1(x))
        x = self.relu_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def extra_repr(self):
        return 'dropout={}, relu_dropout={}, input_dropout={}, normalize_before={}'.format(self.dropout_module.p, self.relu_dropout_module.p, self.input_dropout_module.p, self.normalize_before)


class LightConvDecoder(FairseqIncrementalDecoder):
    """
    LightConv decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`LightConvDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, final_norm=True):
        super().__init__(dictionary)
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.share_input_output_embed = args.share_decoder_input_output_embed
        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        output_embed_dim = args.decoder_output_dim
        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions
        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None
        self.embed_positions = PositionalEmbedding(args.max_target_positions, embed_dim, padding_idx, learned=args.decoder_learned_pos) if not args.no_token_positional_embeddings else None
        self.layers = nn.ModuleList([])
        self.layers.extend([LightConvDecoderLayer(args, no_encoder_attn, kernel_size=args.decoder_kernel_size_list[i]) for i in range(args.decoder_layers)])
        self.adaptive_softmax = None
        self.project_out_dim = Linear(embed_dim, output_embed_dim, bias=False) if embed_dim != output_embed_dim and not args.tie_adaptive_weights else None
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(len(dictionary), output_embed_dim, utils.eval_str_list(args.adaptive_softmax_cutoff, type=int), dropout=args.adaptive_softmax_dropout, adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None, factor=args.adaptive_softmax_factor, tie_proj=args.tie_adaptive_proj)
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=output_embed_dim ** -0.5)
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """
        positions = self.embed_positions(prev_output_tokens, incremental_state=incremental_state) if self.embed_positions is not None else None
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)
        if positions is not None:
            x += positions
        x = self.dropout_module(x)
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]
        for layer in self.layers:
            x, attn = layer(x, encoder_out['encoder_out'] if encoder_out is not None else None, encoder_out['encoder_padding_mask'] if encoder_out is not None else None, incremental_state)
            inner_states.append(x)
        if self.normalize:
            x = self.layer_norm(x)
        x = x.transpose(0, 1)
        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
        if self.adaptive_softmax is None:
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = F.linear(x, self.embed_out)
        return x, {'attn': attn, 'inner_states': inner_states}

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]


class LightConvEncoderLayer(nn.Module):
    """Encoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        kernel_size: kernel size of the convolution
    """

    def __init__(self, args, kernel_size=0):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.conv_dim = args.encoder_conv_dim
        padding_l = kernel_size // 2 if kernel_size % 2 == 1 else ((kernel_size - 1) // 2, kernel_size // 2)
        if args.encoder_glu:
            self.linear1 = Linear(self.embed_dim, 2 * self.conv_dim)
            self.act = nn.GLU()
        else:
            self.linear1 = Linear(self.embed_dim, self.conv_dim)
            self.act = None
        if args.encoder_conv_type == 'lightweight':
            self.conv = LightweightConv(self.conv_dim, kernel_size, padding_l=padding_l, weight_softmax=args.weight_softmax, num_heads=args.encoder_attention_heads, weight_dropout=args.weight_dropout)
        elif args.encoder_conv_type == 'dynamic':
            self.conv = DynamicConv(self.conv_dim, kernel_size, padding_l=padding_l, weight_softmax=args.weight_softmax, num_heads=args.encoder_attention_heads, weight_dropout=args.weight_dropout)
        else:
            raise NotImplementedError
        self.linear2 = Linear(self.conv_dim, self.embed_dim)
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.relu_dropout_module = FairseqDropout(args.relu_dropout, module_name=self.__class__.__name__)
        self.input_dropout_module = FairseqDropout(args.input_dropout, module_name=self.__class__.__name__)
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x = self.input_dropout_module(x)
        x = self.linear1(x)
        if self.act is not None:
            x = self.act(x)
        if encoder_padding_mask is not None:
            x = x.masked_fill(encoder_padding_mask.transpose(0, 1).unsqueeze(2), 0)
        x = self.conv(x)
        x = self.linear2(x)
        x = self.dropout_module(x)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)
        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = self.relu_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

    def extra_repr(self):
        return 'dropout={}, relu_dropout={}, input_dropout={}, normalize_before={}'.format(self.dropout_module.p, self.relu_dropout_module.p, self.input_dropout_module.p, self.normalize_before)


class LightConvEncoder(FairseqEncoder):
    """
    LightConv encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`LightConvEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions
        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(args.max_source_positions, embed_dim, self.padding_idx, learned=args.encoder_learned_pos) if not args.no_token_positional_embeddings else None
        self.layers = nn.ModuleList([])
        self.layers.extend([LightConvEncoderLayer(args, kernel_size=args.encoder_kernel_size_list[i]) for i in range(args.encoder_layers)])
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, src_tokens, **unused):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = self.dropout_module(x)
        x = x.transpose(0, 1)
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
        if self.normalize:
            x = self.layer_norm(x)
        return {'encoder_out': x, 'encoder_padding_mask': encoder_padding_mask}

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)


class LightConvModel(FairseqEncoderDecoderModel):
    """
    LightConv and DynamicConv model from `"Pay Less Attention with Lightweight and Dynamic Convolutions" (Wu, et al, 2019)
    <https://openreview.net/pdf?id=SkVhlh09tX>`_.
    To use LightConv please set ``--encoder-conv-type lightweight --decoder-conv-type lightweight``
    To use DynamicConv please set ``--encoder-conv-type dynamic --decoder-conv-type dynamic``

    Args:
        encoder (LightConvEncoder): the encoder
        decoder (LightConvDecoder): the decoder

    The LightConv model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.lightconv_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):

        def moses_subword(path):
            return {'path': path, 'tokenizer': 'moses', 'bpe': 'subword_nmt'}
        return {'lightconv.no_glu.iwslt14.de-en': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/dynamicconv/iwslt14.de-en.lightconv.tar.gz'), 'dynamicconv.no_glu.iwslt14.de-en': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/dynamicconv/iwslt14.de-en.dynamicconv.tar.gz'), 'lightconv.no_glu.wmt16.en-de': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/dynamicconv/wmt16.en-de.joined-dict.lightconv.tar.gz'), 'dynamicconv.no_glu.wmt16.en-de': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/dynamicconv/wmt16.en-de.joined-dict.dynamicconv.tar.gz'), 'lightconv.glu.wmt16.en-de': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/dynamicconv/wmt16.en-de.joined-dict.lightconv-glu.tar.gz'), 'dynamicconv.glu.wmt16.en-de': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/dynamicconv/wmt16.en-de.joined-dict.dynamicconv-glu.tar.gz'), 'lightconv.glu.wmt17.en-de': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/dynamicconv/wmt16.en-de.joined-dict.lightconv-glu.tar.gz'), 'dynamicconv.glu.wmt17.en-de': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/dynamicconv/wmt16.en-de.joined-dict.dynamicconv-glu.tar.gz'), 'lightconv.glu.wmt14.en-fr': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/dynamicconv/wmt14.en-fr.joined-dict.lightconv-glu.tar.gz'), 'dynamicconv.glu.wmt14.en-fr': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/dynamicconv/wmt14.en-fr.joined-dict.dynamicconv-glu.tar.gz'), 'lightconv.glu.wmt17.zh-en': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/dynamicconv/wmt17.zh-en.lightconv-glu.tar.gz'), 'dynamicconv.glu.wmt17.zh-en': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/dynamicconv/wmt17.zh-en.dynamicconv-glu.tar.gz')}

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D', help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D', help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D', help='dropout probability after ReLU in FFN')
        parser.add_argument('--input-dropout', type=float, metavar='D', help='dropout probability of the inputs')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR', help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N', help='encoder embedding dimension')
        parser.add_argument('--encoder-conv-dim', type=int, metavar='N', help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N', help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N', help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N', help='num encoder attention heads or LightConv/DynamicConv heads')
        parser.add_argument('--encoder-normalize-before', action='store_true', help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true', help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR', help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N', help='decoder embedding dimension')
        parser.add_argument('--decoder-conv-dim', type=int, metavar='N', help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N', help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N', help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N', help='num decoder attention heads or LightConv/DynamicConv heads')
        parser.add_argument('--decoder-learned-pos', action='store_true', help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true', help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true', help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true', help='share encoder, decoder and output embeddings (requires shared dictionary and embed dim)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR', help='comma separated list of adaptive softmax cutoff points. Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D', help='sets adaptive softmax dropout for the tail projections')
        """LightConv and DynamicConv arguments"""
        parser.add_argument('--encoder-kernel-size-list', type=lambda x: utils.eval_str_list(x, int), help='list of kernel size (default: "[3,7,15,31,31,31,31]")')
        parser.add_argument('--decoder-kernel-size-list', type=lambda x: utils.eval_str_list(x, int), help='list of kernel size (default: "[3,7,15,31,31,31]")')
        parser.add_argument('--encoder-glu', type=utils.eval_bool, help='glu after in proj')
        parser.add_argument('--decoder-glu', type=utils.eval_bool, help='glu after in proj')
        parser.add_argument('--encoder-conv-type', default='dynamic', type=str, choices=['dynamic', 'lightweight'], help='type of convolution')
        parser.add_argument('--decoder-conv-type', default='dynamic', type=str, choices=['dynamic', 'lightweight'], help='type of convolution')
        parser.add_argument('--weight-softmax', default=True, type=utils.eval_bool)
        parser.add_argument('--weight-dropout', type=float, metavar='D', help='dropout probability for conv weights')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)
        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb
        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise RuntimeError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise RuntimeError('--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and args.decoder_embed_path != args.encoder_embed_path:
                raise RuntimeError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(src_dict, args.encoder_embed_dim, args.encoder_embed_path)
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(src_dict, args.encoder_embed_dim, args.encoder_embed_path)
            decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim, args.decoder_embed_path)
        encoder = LightConvEncoder(args, src_dict, encoder_embed_tokens)
        decoder = LightConvDecoder(args, tgt_dict, decoder_embed_tokens)
        return LightConvModel(encoder, decoder)


DEFAULT_MAX_SOURCE_POSITIONS = 1024


class MLPAttention(nn.Module):
    """The original attention from Badhanau et al. (2014)

    https://arxiv.org/abs/1409.0473, based on a Multi-Layer Perceptron.
    The attention score between position i in the encoder and position j in the
    decoder is: alpha_ij = V_a * tanh(W_ae * enc_i + W_ad * dec_j + b_a)
    """

    def __init__(self, decoder_hidden_state_dim, context_dim, attention_dim):
        super().__init__()
        self.context_dim = context_dim
        self.attention_dim = attention_dim
        self.encoder_proj = nn.Linear(context_dim, self.attention_dim, bias=True)
        self.decoder_proj = nn.Linear(decoder_hidden_state_dim, self.attention_dim, bias=False)
        self.to_scores = nn.Linear(self.attention_dim, 1, bias=False)

    def forward(self, decoder_state, source_hids, encoder_padding_mask):
        """The expected input dimensions are:
        decoder_state: bsz x decoder_hidden_state_dim
        source_hids: src_len x bsz x context_dim
        encoder_padding_mask: src_len x bsz
        """
        src_len, bsz, _ = source_hids.size()
        flat_source_hids = source_hids.view(-1, self.context_dim)
        encoder_component = self.encoder_proj(flat_source_hids)
        encoder_component = encoder_component.view(src_len, bsz, self.attention_dim)
        decoder_component = self.decoder_proj(decoder_state).unsqueeze(0)
        hidden_att = torch.tanh((decoder_component + encoder_component).view(-1, self.attention_dim))
        attn_scores = self.to_scores(hidden_att).view(src_len, bsz)
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(encoder_padding_mask, float('-inf')).type_as(attn_scores)
        normalized_masked_attn_scores = F.softmax(attn_scores, dim=0)
        attn_weighted_context = (source_hids * normalized_masked_attn_scores.unsqueeze(2)).sum(dim=0)
        return attn_weighted_context, normalized_masked_attn_scores


class LSTMDecoder(FairseqIncrementalDecoder):

    def __init__(self, dictionary, embed_dim, num_layers, hidden_size, dropout, encoder_output_dim, attention_dim, output_layer_dim):
        """
        Args:
            dictionary: target text dictionary.
            embed_dim: embedding dimension for target tokens.
            num_layers: number of LSTM layers.
            hidden_size: hidden size for LSTM layers.
            dropout: dropout probability. Dropout can be applied to the
                embeddings, the LSTM layers, and the context vector.
            encoder_output_dim: encoder output dimension (hidden size of
                encoder LSTM).
            attention_dim: attention dimension for MLP attention.
            output_layer_dim: size of the linear layer prior to output
                projection.
        """
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(num_embeddings, embed_dim, padding_idx)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.layers = nn.ModuleList()
        for layer_id in range(num_layers):
            input_size = embed_dim if layer_id == 0 else encoder_output_dim
            self.layers.append(nn.LSTMCell(input_size=input_size, hidden_size=hidden_size))
        self.context_dim = encoder_output_dim
        self.attention = MLPAttention(decoder_hidden_state_dim=hidden_size, context_dim=encoder_output_dim, attention_dim=attention_dim)
        self.deep_output_layer = nn.Linear(hidden_size + encoder_output_dim + embed_dim, output_layer_dim)
        self.output_projection = nn.Linear(output_layer_dim, num_embeddings)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        encoder_padding_mask = encoder_out['encoder_padding_mask']
        encoder_outs = encoder_out['encoder_out']
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()
        srclen = encoder_outs.size(0)
        embeddings = self.embed_tokens(prev_output_tokens)
        x = embeddings
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.transpose(0, 1)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells = cached_state
        else:
            prev_hiddens = [encoder_out['encoder_out'].mean(dim=0)] * self.num_layers
            prev_cells = [x.new_zeros(bsz, self.hidden_size)] * self.num_layers
        attn_scores = x.new_zeros(bsz, srclen)
        attention_outs = []
        outs = []
        for j in range(seqlen):
            input = x[j, :, :]
            attention_out = None
            for i, layer in enumerate(self.layers):
                hidden, cell = layer(input, (prev_hiddens[(i - 1) % self.num_layers], prev_cells[(i - 1) % self.num_layers]))
                if self.dropout is not None:
                    hidden = self.dropout(hidden)
                prev_hiddens[i] = hidden
                prev_cells[i] = cell
                if attention_out is None:
                    attention_out, attn_scores = self.attention(hidden, encoder_outs, encoder_padding_mask)
                    if self.dropout is not None:
                        attention_out = self.dropout(attention_out)
                    attention_outs.append(attention_out)
                input = attention_out
            outs.append(hidden)
        utils.set_incremental_state(self, incremental_state, 'cached_state', (prev_hiddens, prev_cells))
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)
        attention_outs_concat = torch.cat(attention_outs, dim=0).view(seqlen, bsz, self.context_dim)
        x = x.transpose(0, 1)
        attention_outs_concat = attention_outs_concat.transpose(0, 1)
        x = torch.cat((x, attention_outs_concat, embeddings), dim=2)
        x = self.deep_output_layer(x)
        x = torch.tanh(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.output_projection(x)
        return x, None

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)
        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


class LSTMEncoder(FairseqEncoder):
    """LSTM encoder."""

    def __init__(self, dictionary, embed_dim=512, hidden_size=512, num_layers=1, dropout_in=0.1, dropout_out=0.1, bidirectional=False, left_pad=True, pretrained_embed=None, padding_idx=None, max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS):
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in_module = FairseqDropout(dropout_in, module_name=self.__class__.__name__)
        self.dropout_out_module = FairseqDropout(dropout_out, module_name=self.__class__.__name__)
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.max_source_positions = max_source_positions
        num_embeddings = len(dictionary)
        self.padding_idx = padding_idx if padding_idx is not None else dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed
        self.lstm = LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=self.dropout_out_module.p if num_layers > 1 else 0.0, bidirectional=bidirectional)
        self.left_pad = left_pad
        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens: 'Tensor', src_lengths: 'Tensor', enforce_sorted: 'bool'=True):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of
                shape `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of
                shape `(batch)`
            enforce_sorted (bool, optional): if True, `src_tokens` is
                expected to contain sequences sorted by length in a
                decreasing order. If False, this condition is not
                required. Default: True.
        """
        if self.left_pad:
            src_tokens = utils.convert_padding_direction(src_tokens, torch.zeros_like(src_tokens).fill_(self.padding_idx), left_to_right=True)
        bsz, seqlen = src_tokens.size()
        x = self.embed_tokens(src_tokens)
        x = self.dropout_in_module(x)
        x = x.transpose(0, 1)
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.cpu(), enforce_sorted=enforce_sorted)
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_idx * 1.0)
        x = self.dropout_out_module(x)
        assert list(x.size()) == [seqlen, bsz, self.output_units]
        if self.bidirectional:
            final_hiddens = self.combine_bidir(final_hiddens, bsz)
            final_cells = self.combine_bidir(final_cells, bsz)
        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()
        return tuple((x, final_hiddens, final_cells, encoder_padding_mask))

    def combine_bidir(self, outs, bsz: 'int'):
        out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
        return out.view(self.num_layers, bsz, -1)

    def reorder_encoder_out(self, encoder_out, new_order):
        return tuple((encoder_out[0].index_select(1, new_order), encoder_out[1].index_select(1, new_order), encoder_out[2].index_select(1, new_order), encoder_out[3].index_select(1, new_order)))

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions


class LSTMModel(FairseqEncoderDecoderModel):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D', help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N', help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR', help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-freeze-embed', action='store_true', help='freeze encoder embeddings')
        parser.add_argument('--encoder-hidden-size', type=int, metavar='N', help='encoder hidden size')
        parser.add_argument('--encoder-layers', type=int, metavar='N', help='number of encoder layers')
        parser.add_argument('--encoder-bidirectional', action='store_true', help='make all layers of encoder bidirectional')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N', help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR', help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-freeze-embed', action='store_true', help='freeze decoder embeddings')
        parser.add_argument('--decoder-hidden-size', type=int, metavar='N', help='decoder hidden size')
        parser.add_argument('--decoder-layers', type=int, metavar='N', help='number of decoder layers')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N', help='decoder output embedding dimension')
        parser.add_argument('--decoder-attention', type=str, metavar='BOOL', help='decoder attention')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR', help='comma separated list of adaptive softmax cutoff points. Must be used with adaptive_loss criterion')
        parser.add_argument('--share-decoder-input-output-embed', default=False, action='store_true', help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', default=False, action='store_true', help='share encoder, decoder and output embeddings (requires shared dictionary and embed dim)')
        parser.add_argument('--encoder-dropout-in', type=float, metavar='D', help='dropout probability for encoder input embedding')
        parser.add_argument('--encoder-dropout-out', type=float, metavar='D', help='dropout probability for encoder output')
        parser.add_argument('--decoder-dropout-in', type=float, metavar='D', help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, metavar='D', help='dropout probability for decoder output')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)
        if args.encoder_layers != args.decoder_layers:
            raise ValueError('--encoder-layers must match --decoder-layers')
        max_source_positions = getattr(args, 'max_source_positions', DEFAULT_MAX_SOURCE_POSITIONS)
        max_target_positions = getattr(args, 'max_target_positions', DEFAULT_MAX_TARGET_POSITIONS)

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)
        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim)
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = Embedding(num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad())
        if args.share_all_embeddings:
            if task.source_dictionary != task.target_dictionary:
                raise ValueError('--share-all-embeddings requires a joint dictionary')
            if args.decoder_embed_path and args.decoder_embed_path != args.encoder_embed_path:
                raise ValueError('--share-all-embed not compatible with --decoder-embed-path')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError('--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            pretrained_decoder_embed = pretrained_encoder_embed
            args.share_decoder_input_output_embed = True
        else:
            pretrained_decoder_embed = None
            if args.decoder_embed_path:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(args.decoder_embed_path, task.target_dictionary, args.decoder_embed_dim)
        if args.share_decoder_input_output_embed and args.decoder_embed_dim != args.decoder_out_embed_dim:
            raise ValueError('--share-decoder-input-output-embeddings requires --decoder-embed-dim to match --decoder-out-embed-dim')
        if args.encoder_freeze_embed:
            pretrained_encoder_embed.weight.requires_grad = False
        if args.decoder_freeze_embed:
            pretrained_decoder_embed.weight.requires_grad = False
        encoder = LSTMEncoder(dictionary=task.source_dictionary, embed_dim=args.encoder_embed_dim, hidden_size=args.encoder_hidden_size, num_layers=args.encoder_layers, dropout_in=args.encoder_dropout_in, dropout_out=args.encoder_dropout_out, bidirectional=args.encoder_bidirectional, pretrained_embed=pretrained_encoder_embed, max_source_positions=max_source_positions)
        decoder = LSTMDecoder(dictionary=task.target_dictionary, embed_dim=args.decoder_embed_dim, hidden_size=args.decoder_hidden_size, out_embed_dim=args.decoder_out_embed_dim, num_layers=args.decoder_layers, dropout_in=args.decoder_dropout_in, dropout_out=args.decoder_dropout_out, attention=utils.eval_bool(args.decoder_attention), encoder_output_units=encoder.output_units, pretrained_embed=pretrained_decoder_embed, share_input_output_embed=args.share_decoder_input_output_embed, adaptive_softmax_cutoff=utils.eval_str_list(args.adaptive_softmax_cutoff, type=int) if args.criterion == 'adaptive_loss' else None, max_target_positions=max_target_positions, residuals=False)
        return cls(encoder, decoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]'=None):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, incremental_state=incremental_state)
        return decoder_out


class LayerDropModuleList(nn.ModuleList):
    """
    A LayerDrop implementation based on :class:`torch.nn.ModuleList`.

    We refresh the choice of which layers to drop every time we iterate
    over the LayerDropModuleList instance. During evaluation we always
    iterate over all layers.

    Usage::

        layers = LayerDropList(p=0.5, modules=[layer1, layer2, layer3])
        for layer in layers:  # this might iterate over layers 1 and 3
            x = layer(x)
        for layer in layers:  # this might iterate over all layers
            x = layer(x)
        for layer in layers:  # this might not iterate over any layers
            x = layer(x)

    Args:
        p (float): probability of dropping out each layer
        modules (iterable, optional): an iterable of modules to add
    """

    def __init__(self, p, modules=None):
        super().__init__(modules)
        self.p = p

    def __iter__(self):
        dropout_probs = torch.empty(len(self)).uniform_()
        for i, m in enumerate(super().__iter__()):
            if not self.training or dropout_probs[i] > self.p:
                yield m


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(self, embedding_dim: 'int'=768, ffn_embedding_dim: 'int'=3072, num_attention_heads: 'int'=8, dropout: 'float'=0.1, attention_dropout: 'float'=0.1, activation_dropout: 'float'=0.1, activation_fn: 'str'='relu', export: 'bool'=False, q_noise: 'float'=0.0, qn_block_size: 'int'=8, init_fn: 'Callable'=None) ->None:
        super().__init__()
        if init_fn is not None:
            init_fn()
        self.embedding_dim = embedding_dim
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.activation_dropout_module = FairseqDropout(activation_dropout, module_name=self.__class__.__name__)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = self.build_self_attention(self.embedding_dim, num_attention_heads, dropout=attention_dropout, self_attention=True, q_noise=q_noise, qn_block_size=qn_block_size)
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)
        self.fc1 = self.build_fc1(self.embedding_dim, ffn_embedding_dim, q_noise=q_noise, qn_block_size=qn_block_size)
        self.fc2 = self.build_fc2(ffn_embedding_dim, self.embedding_dim, q_noise=q_noise, qn_block_size=qn_block_size)
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(self, embed_dim, num_attention_heads, dropout, self_attention, q_noise, qn_block_size):
        return MultiheadAttention(embed_dim, num_attention_heads, dropout=dropout, self_attention=True, q_noise=q_noise, qn_block_size=qn_block_size)

    def forward(self, x: 'torch.Tensor', self_attn_mask: 'Optional[torch.Tensor]'=None, self_attn_padding_mask: 'Optional[torch.Tensor]'=None):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        x, attn = self.self_attn(query=x, key=x, value=x, key_padding_mask=self_attn_padding_mask, need_weights=False, attn_mask=self_attn_mask)
        x = self.dropout_module(x)
        x = residual + x
        x = self.self_attn_layer_norm(x)
        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn


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
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.v_proj.weight.data.normal_(mean=0.0, std=0.02)


class TransformerSentenceEncoder(nn.Module):
    """
    Implementation for a Bi-directional Transformer based Sentence Encoder used
    in BERT/XLM style pre-trained models.

    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    TransformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape T x B x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(self, padding_idx: 'int', vocab_size: 'int', num_encoder_layers: 'int'=6, embedding_dim: 'int'=768, ffn_embedding_dim: 'int'=3072, num_attention_heads: 'int'=8, dropout: 'float'=0.1, attention_dropout: 'float'=0.1, activation_dropout: 'float'=0.1, layerdrop: 'float'=0.0, max_seq_len: 'int'=256, num_segments: 'int'=2, use_position_embeddings: 'bool'=True, offset_positions_by_padding: 'bool'=True, encoder_normalize_before: 'bool'=False, apply_bert_init: 'bool'=False, activation_fn: 'str'='relu', learned_pos_embedding: 'bool'=True, embed_scale: 'float'=None, freeze_embeddings: 'bool'=False, n_trans_layers_to_freeze: 'int'=0, export: 'bool'=False, traceable: 'bool'=False, q_noise: 'float'=0.0, qn_block_size: 'int'=8) ->None:
        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_segments = num_segments
        self.use_position_embeddings = use_position_embeddings
        self.apply_bert_init = apply_bert_init
        self.learned_pos_embedding = learned_pos_embedding
        self.traceable = traceable
        self.tpu = False
        self.embed_tokens = self.build_embedding(self.vocab_size, self.embedding_dim, self.padding_idx)
        self.embed_scale = embed_scale
        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(nn.Linear(self.embedding_dim, self.embedding_dim, bias=False), q_noise, qn_block_size)
        else:
            self.quant_noise = None
        self.segment_embeddings = nn.Embedding(self.num_segments, self.embedding_dim, padding_idx=None) if self.num_segments > 0 else None
        self.embed_positions = PositionalEmbedding(self.max_seq_len, self.embedding_dim, padding_idx=self.padding_idx if offset_positions_by_padding else None, learned=self.learned_pos_embedding) if self.use_position_embeddings else None
        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend([self.build_transformer_sentence_encoder_layer(embedding_dim=self.embedding_dim, ffn_embedding_dim=ffn_embedding_dim, num_attention_heads=num_attention_heads, dropout=self.dropout_module.p, attention_dropout=attention_dropout, activation_dropout=activation_dropout, activation_fn=activation_fn, export=export, q_noise=q_noise, qn_block_size=qn_block_size) for _ in range(num_encoder_layers)])
        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None
        if self.apply_bert_init:
            self.apply(init_bert_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False
        if freeze_embeddings:
            freeze_module_params(self.embed_tokens)
            freeze_module_params(self.segment_embeddings)
            freeze_module_params(self.embed_positions)
            freeze_module_params(self.emb_layer_norm)
        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def build_transformer_sentence_encoder_layer(self, embedding_dim, ffn_embedding_dim, num_attention_heads, dropout, attention_dropout, activation_dropout, activation_fn, export, q_noise, qn_block_size):
        return TransformerSentenceEncoderLayer(embedding_dim=embedding_dim, ffn_embedding_dim=ffn_embedding_dim, num_attention_heads=num_attention_heads, dropout=dropout, attention_dropout=attention_dropout, activation_dropout=activation_dropout, activation_fn=activation_fn, export=export, q_noise=q_noise, qn_block_size=qn_block_size)

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def forward(self, tokens: 'torch.Tensor', segment_labels: 'torch.Tensor'=None, last_state_only: 'bool'=False, positions: 'Optional[torch.Tensor]'=None, token_embeddings: 'Optional[torch.Tensor]'=None) ->Tuple[torch.Tensor, torch.Tensor]:
        padding_mask = tokens.eq(self.padding_idx)
        if not self.traceable and not self.tpu and not padding_mask.any():
            padding_mask = None
        if token_embeddings is not None:
            x = token_embeddings
        else:
            x = self.embed_tokens(tokens)
        if self.embed_scale is not None:
            x = x * self.embed_scale
        if self.embed_positions is not None:
            x = x + self.embed_positions(tokens, positions=positions)
        if self.segment_embeddings is not None and segment_labels is not None:
            x = x + self.segment_embeddings(segment_labels)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)
        x = self.dropout_module(x)
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        x = x.transpose(0, 1)
        inner_states = []
        if not last_state_only:
            inner_states.append(x)
        for layer in self.layers:
            x, _ = layer(x, self_attn_padding_mask=padding_mask)
            if not last_state_only:
                inner_states.append(x)
        sentence_rep = x[0, :, :]
        if last_state_only:
            inner_states = [x]
        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            return inner_states, sentence_rep


class MaskedLMEncoder(FairseqEncoder):
    """
    Encoder for Masked Language Modelling.
    """

    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.padding_idx = dictionary.pad()
        self.vocab_size = dictionary.__len__()
        self.max_positions = args.max_positions
        self.sentence_encoder = TransformerSentenceEncoder(padding_idx=self.padding_idx, vocab_size=self.vocab_size, num_encoder_layers=args.encoder_layers, embedding_dim=args.encoder_embed_dim, ffn_embedding_dim=args.encoder_ffn_embed_dim, num_attention_heads=args.encoder_attention_heads, dropout=args.dropout, attention_dropout=args.attention_dropout, activation_dropout=args.act_dropout, max_seq_len=self.max_positions, num_segments=args.num_segment, use_position_embeddings=not args.no_token_positional_embeddings, encoder_normalize_before=args.encoder_normalize_before, apply_bert_init=args.apply_bert_init, activation_fn=args.activation_fn, learned_pos_embedding=args.encoder_learned_pos)
        self.share_input_output_embed = args.share_encoder_input_output_embed
        self.embed_out = None
        self.sentence_projection_layer = None
        self.sentence_out_dim = args.sentence_class_num
        self.lm_output_learned_bias = None
        self.load_softmax = not getattr(args, 'remove_head', False)
        self.masked_lm_pooler = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
        self.pooler_activation = utils.get_activation_fn(args.pooler_activation_fn)
        self.lm_head_transform_weight = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
        self.activation_fn = utils.get_activation_fn(args.activation_fn)
        self.layer_norm = LayerNorm(args.encoder_embed_dim)
        self.lm_output_learned_bias = None
        if self.load_softmax:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(self.vocab_size))
            if not self.share_input_output_embed:
                self.embed_out = nn.Linear(args.encoder_embed_dim, self.vocab_size, bias=False)
            if args.sent_loss:
                self.sentence_projection_layer = nn.Linear(args.encoder_embed_dim, self.sentence_out_dim, bias=False)

    def forward(self, src_tokens, segment_labels=None, masked_tokens=None, **unused):
        """
        Forward pass for Masked LM encoder. This first computes the token
        embedding using the token embedding matrix, position embeddings (if
        specified) and segment embeddings (if specified).

        Here we assume that the sentence representation corresponds to the
        output of the classification_token (see bert_task or cross_lingual_lm
        task for more details).
        Args:
            - src_tokens: B x T matrix representing sentences
            - segment_labels: B x T matrix representing segment label for tokens
        Returns:
            - a tuple of the following:
                - logits for predictions in format B x T x C to be used in
                  softmax afterwards
                - a dictionary of additional data, where 'pooled_output' contains
                  the representation for classification_token and 'inner_states'
                  is a list of internal model states used to compute the
                  predictions (similar in ELMO). 'sentence_logits'
                  is the prediction logit for NSP task and is only computed if
                  this is specified in the input arguments.
        """
        inner_states, sentence_rep = self.sentence_encoder(src_tokens, segment_labels=segment_labels)
        x = inner_states[-1].transpose(0, 1)
        if masked_tokens is not None:
            x = x[masked_tokens, :]
        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))
        pooled_output = self.pooler_activation(self.masked_lm_pooler(sentence_rep))
        if self.share_input_output_embed and hasattr(self.sentence_encoder.embed_tokens, 'weight'):
            x = F.linear(x, self.sentence_encoder.embed_tokens.weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)
        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias
        sentence_logits = None
        if self.sentence_projection_layer:
            sentence_logits = self.sentence_projection_layer(pooled_output)
        return x, {'inner_states': inner_states, 'pooled_output': pooled_output, 'sentence_logits': sentence_logits}

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        if isinstance(self.sentence_encoder.embed_positions, SinusoidalPositionalEmbedding):
            state_dict[name + '.sentence_encoder.embed_positions._float_tensor'] = torch.FloatTensor(1)
        if not self.load_softmax:
            for k in list(state_dict.keys()):
                if 'embed_out.weight' in k or 'sentence_projection_layer.weight' in k or 'lm_output_learned_bias' in k:
                    del state_dict[k]
        return state_dict


class MaskedLMModel(FairseqEncoderModel):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args
        if getattr(args, 'apply_bert_init', False):
            self.apply(init_bert_params)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D', help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D', help='dropout probability for attention weights')
        parser.add_argument('--act-dropout', type=float, metavar='D', help='dropout probability after activation in FFN')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N', help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N', help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N', help='num encoder attention heads')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N', help='encoder embedding dimension')
        parser.add_argument('--share-encoder-input-output-embed', action='store_true', help='share encoder input and output embeddings')
        parser.add_argument('--encoder-learned-pos', action='store_true', help='use learned positional embeddings in the encoder')
        parser.add_argument('--no-token-positional-embeddings', action='store_true', help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--num-segment', type=int, metavar='N', help='num segment in the input')
        parser.add_argument('--max-positions', type=int, help='number of positional embeddings to learn')
        parser.add_argument('--sentence-class-num', type=int, metavar='N', help='number of classes for sentence task')
        parser.add_argument('--sent-loss', action='store_true', help='if set, calculate sentence level predictions')
        parser.add_argument('--apply-bert-init', action='store_true', help='use custom param initialization for BERT')
        parser.add_argument('--activation-fn', choices=utils.get_available_activation_fns(), help='activation function to use')
        parser.add_argument('--pooler-activation-fn', choices=utils.get_available_activation_fns(), help='Which activation function to use for pooler layer.')
        parser.add_argument('--encoder-normalize-before', action='store_true', help='apply layernorm before each encoder block')

    def forward(self, src_tokens, segment_labels=None, **kwargs):
        return self.encoder(src_tokens, segment_labels=segment_labels, **kwargs)

    def max_positions(self):
        return self.encoder.max_positions

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)
        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample
        logger.info(args)
        encoder = MaskedLMEncoder(args, task.dictionary)
        return cls(args, encoder)


class _EnsembleModelEncoder(object):

    def __init__(self, models):
        self.models = models

    def reorder_encoder_out(self, encoder_outs, new_order):
        encoder_outs = [model.encoder.reorder_encoder_out(encoder_out, new_order) for model, encoder_out in zip(self.models, encoder_outs)]
        return encoder_outs


class BasicEnsembleModel(torch.nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.bos = self.models[0].decoder.dictionary.bos()
        self.eos = self.models[0].decoder.dictionary.eos()
        self.pad = self.models[0].decoder.dictionary.pad()
        self.unk = self.models[0].decoder.dictionary.unk()
        self.encoder = _EnsembleModelEncoder(self.models)

    def has_encoder(self):
        return hasattr(self.models[0], 'encoder')

    def max_decoder_positions(self):
        return min(m.max_decoder_positions() for m in self.models)

    @torch.no_grad()
    def forward_encoder(self, encoder_input):
        if not self.has_encoder():
            return None
        return [model.forward_encoder(encoder_input) for model in self.models]

    @torch.no_grad()
    def forward_decoder(self, *inputs):
        raise NotImplementedError

    def initialize_output_tokens(self, *inputs):
        raise NotImplementedError


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


def _apply_del_words(in_tokens, in_scores, in_attn, word_del_pred, padding_idx, bos_idx, eos_idx):
    in_masks = in_tokens.ne(padding_idx)
    bos_eos_masks = in_tokens.eq(bos_idx) | in_tokens.eq(eos_idx)
    max_len = in_tokens.size(1)
    word_del_pred.masked_fill_(~in_masks, 1)
    word_del_pred.masked_fill_(bos_eos_masks, 0)
    reordering = new_arange(in_tokens).masked_fill_(word_del_pred, max_len).sort(1)[1]
    out_tokens = in_tokens.masked_fill(word_del_pred, padding_idx).gather(1, reordering)
    out_scores = None
    if in_scores is not None:
        out_scores = in_scores.masked_fill(word_del_pred, 0).gather(1, reordering)
    out_attn = None
    if in_attn is not None:
        _mask = word_del_pred[:, :, None].expand_as(in_attn)
        _reordering = reordering[:, :, None].expand_as(in_attn)
        out_attn = in_attn.masked_fill(_mask, 0.0).gather(1, _reordering)
    return out_tokens, out_scores, out_attn


def _apply_ins_masks(in_tokens, in_scores, mask_ins_pred, padding_idx, unk_idx, eos_idx):
    in_masks = in_tokens.ne(padding_idx)
    in_lengths = in_masks.sum(1)
    in_tokens.masked_fill_(~in_masks, eos_idx)
    mask_ins_pred.masked_fill_(~in_masks[:, 1:], 0)
    out_lengths = in_lengths + mask_ins_pred.sum(1)
    out_max_len = out_lengths.max()
    out_masks = new_arange(out_lengths, out_max_len)[None, :] < out_lengths[:, None]
    reordering = (mask_ins_pred + in_masks[:, 1:].long()).cumsum(1)
    out_tokens = in_tokens.new_zeros(in_tokens.size(0), out_max_len).fill_(padding_idx).masked_fill_(out_masks, unk_idx)
    out_tokens[:, 0] = in_tokens[:, 0]
    out_tokens.scatter_(1, reordering, in_tokens[:, 1:])
    out_scores = None
    if in_scores is not None:
        in_scores.masked_fill_(~in_masks, 0)
        out_scores = in_scores.new_zeros(*out_tokens.size())
        out_scores[:, 0] = in_scores[:, 0]
        out_scores.scatter_(1, reordering, in_scores[:, 1:])
    return out_tokens, out_scores


def _apply_ins_words(in_tokens, in_scores, word_ins_pred, word_ins_scores, unk_idx):
    word_ins_masks = in_tokens.eq(unk_idx)
    out_tokens = in_tokens.masked_scatter(word_ins_masks, word_ins_pred[word_ins_masks])
    if in_scores is not None:
        out_scores = in_scores.masked_scatter(word_ins_masks, word_ins_scores[word_ins_masks])
    else:
        out_scores = None
    return out_tokens, out_scores


def _fill(x, mask, y, padding_idx):
    """
    Filling tensor x with y at masked positions (dim=0).
    """
    if x is None:
        return y
    assert x.dim() == y.dim() and mask.size(0) == x.size(0)
    assert x.dim() == 2 or x.dim() == 3 and x.size(2) == y.size(2)
    n_selected = mask.sum()
    assert n_selected == y.size(0)
    if n_selected == x.size(0):
        return y
    if x.size(1) < y.size(1):
        dims = [x.size(0), y.size(1) - x.size(1)]
        if x.dim() == 3:
            dims.append(x.size(2))
        x = torch.cat([x, x.new_zeros(*dims).fill_(padding_idx)], 1)
        x[mask] = y
    elif x.size(1) > y.size(1):
        x[mask] = padding_idx
        if x.dim() == 2:
            x[mask, :y.size(1)] = y
        else:
            x[mask, :y.size(1), :] = y
    else:
        x[mask] = y
    return x


def _skip(x, mask):
    """
    Getting sliced (dim=0) tensor by mask. Supporting tensor and list/dict of tensors.
    """
    if isinstance(x, int):
        return x
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        if x.size(0) == mask.size(0):
            return x[mask]
        elif x.size(1) == mask.size(0):
            return x[:, mask]
    if isinstance(x, list):
        return [_skip(x_i, mask) for x_i in x]
    if isinstance(x, dict):
        return {k: _skip(v, mask) for k, v in x.items()}
    raise NotImplementedError


def _skip_encoder_out(encoder, encoder_out, mask):
    if not mask.any():
        return encoder_out
    else:
        return encoder.reorder_encoder_out(encoder_out, mask.nonzero(as_tuple=False).squeeze())


class EnsembleLevT(BasicEnsembleModel):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__(models)

    @torch.no_grad()
    def forward_decoder(self, decoder_out, encoder_outs, eos_penalty=0.0, max_ratio=None, **kwargs):
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        attn = decoder_out.attn
        bsz = output_tokens.size(0)
        if max_ratio is None:
            max_lens = output_tokens.new().fill_(255)
        else:
            if encoder_outs[0].encoder_padding_mask is None:
                src_lens = encoder_outs[0].encoder_out.new(bsz).fill_(encoder_outs[0].encoder_out.size(1))
            else:
                src_lens = (~encoder_outs[0].encoder_padding_mask).sum(1)
            max_lens = (src_lens * max_ratio).clamp(min=10).long()
        can_del_word = output_tokens.ne(self.pad).sum(1) > 2
        if can_del_word.sum() != 0:
            output_tokens, output_scores, attn = self.forward_word_del(encoder_outs, output_tokens, output_scores, attn, can_del_word)
        can_ins_mask = output_tokens.ne(self.pad).sum(1) < max_lens
        if can_ins_mask.sum() != 0:
            output_tokens, output_scores = self.forward_mask_ins(encoder_outs, output_tokens, output_scores, can_ins_mask, eos_penalty, max_lens)
        can_ins_word = output_tokens.eq(self.unk).sum(1) > 0
        if can_ins_word.sum() != 0:
            output_tokens, output_scores, attn = self.forward_word_ins(encoder_outs, output_tokens, output_scores, attn, can_ins_word)
        cut_off = output_tokens.ne(self.pad).sum(1).max()
        output_tokens = output_tokens[:, :cut_off]
        output_scores = output_scores[:, :cut_off]
        attn = None if attn is None else attn[:, :cut_off, :]
        return decoder_out._replace(output_tokens=output_tokens, output_scores=output_scores, attn=attn, history=None)

    def forward_word_del(self, encoder_outs, output_tokens, output_scores, attn, can_del_word):
        word_del_score_avg = []
        word_del_attn_avg = []
        for model, encoder_out in zip(self.models, encoder_outs):
            word_del_out, word_del_attn = model.decoder.forward_word_del(_skip(output_tokens, can_del_word), _skip_encoder_out(model.encoder, encoder_out, can_del_word))
            word_del_score = F.log_softmax(word_del_out, 2)
            word_del_score_avg.append(word_del_score)
            word_del_attn_avg.append(word_del_attn)
        word_del_score_avg = torch.logsumexp(torch.stack(word_del_score_avg, dim=0), dim=0) - math.log(len(self.models))
        word_del_pred = word_del_score_avg.max(-1)[1].bool()
        if word_del_attn_avg[0] is not None:
            word_del_attn_avg = torch.stack(word_del_attn_avg, dim=0) / len(self.models)
        else:
            word_del_attn_avg = None
        _tokens, _scores, _attn = _apply_del_words(output_tokens[can_del_word], output_scores[can_del_word], word_del_attn_avg, word_del_pred, self.pad, self.bos, self.eos)
        output_tokens = _fill(output_tokens, can_del_word, _tokens, self.pad)
        output_scores = _fill(output_scores, can_del_word, _scores, 0)
        attn = _fill(attn, can_del_word, _attn, 0.0)
        return output_tokens, output_scores, attn

    def forward_mask_ins(self, encoder_outs, output_tokens, output_scores, can_ins_mask, eos_penalty, max_lens):
        mask_ins_score_avg = []
        for model, encoder_out in zip(self.models, encoder_outs):
            mask_ins_out, _ = model.decoder.forward_mask_ins(_skip(output_tokens, can_ins_mask), _skip_encoder_out(model.encoder, encoder_out, can_ins_mask))
            mask_ins_score = F.log_softmax(mask_ins_out, 2)
            if eos_penalty > 0.0:
                mask_ins_score[:, :, 0] -= eos_penalty
            mask_ins_score_avg.append(mask_ins_score)
        mask_ins_score_avg = torch.logsumexp(torch.stack(mask_ins_score_avg, dim=0), dim=0) - math.log(len(self.models))
        mask_ins_pred = mask_ins_score_avg.max(-1)[1]
        mask_ins_pred = torch.min(mask_ins_pred, max_lens[can_ins_mask, None].expand_as(mask_ins_pred))
        _tokens, _scores = _apply_ins_masks(output_tokens[can_ins_mask], output_scores[can_ins_mask], mask_ins_pred, self.pad, self.unk, self.eos)
        output_tokens = _fill(output_tokens, can_ins_mask, _tokens, self.pad)
        output_scores = _fill(output_scores, can_ins_mask, _scores, 0)
        return output_tokens, output_scores

    def forward_word_ins(self, encoder_outs, output_tokens, output_scores, attn, can_ins_word):
        word_ins_score_avg = []
        word_ins_attn_avg = []
        for model, encoder_out in zip(self.models, encoder_outs):
            word_ins_out, word_ins_attn = model.decoder.forward_word_ins(_skip(output_tokens, can_ins_word), _skip_encoder_out(model.encoder, encoder_out, can_ins_word))
            word_ins_score = F.log_softmax(word_ins_out, 2)
            word_ins_score_avg.append(word_ins_score)
            word_ins_attn_avg.append(word_ins_attn)
        word_ins_score_avg = torch.logsumexp(torch.stack(word_ins_score_avg, dim=0), dim=0) - math.log(len(self.models))
        if word_ins_attn_avg[0] is not None:
            word_ins_attn_avg = torch.stack(word_ins_attn_avg, dim=0) / len(self.models)
        else:
            word_ins_attn_avg = None
        word_ins_score_max, word_ins_pred = word_ins_score_avg.max(-1)
        _tokens, _scores = _apply_ins_words(output_tokens[can_ins_word], output_scores[can_ins_word], word_ins_pred, word_ins_score_max, self.unk)
        output_tokens = _fill(output_tokens, can_ins_word, _tokens, self.pad)
        output_scores = _fill(output_scores, can_ins_word, _scores, 0)
        attn = _fill(attn, can_ins_word, word_ins_attn, 0.0)
        return output_tokens, output_scores, attn

    def initialize_output_tokens(self, encoder_outs, src_tokens):
        return self.models[0].initialize_output_tokens(encoder_outs[0], src_tokens)


class RobertaHubInterface(nn.Module):
    """A simple PyTorch Hub interface to RoBERTa.

    Usage: https://github.com/pytorch/fairseq/tree/master/examples/roberta
    """

    def __init__(self, args, task, model):
        super().__init__()
        self.args = args
        self.task = task
        self.model = model
        self.bpe = encoders.build_bpe(args)
        self.register_buffer('_float_tensor', torch.tensor([0], dtype=torch.float))

    @property
    def device(self):
        return self._float_tensor.device

    def encode(self, sentence: 'str', *addl_sentences, no_separator=False) ->torch.LongTensor:
        """
        BPE-encode a sentence (or multiple sentences).

        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`) and we use an
        extra end-of-sentence (`</s>`) as a separator.

        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> </s> 1 2 3 </s>`

        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::

            >>> roberta.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> roberta.encode(' world').tolist()
            [0, 232, 2]
            >>> roberta.encode('world').tolist()
            [0, 8331, 2]
        """
        bpe_sentence = '<s> ' + self.bpe.encode(sentence) + ' </s>'
        for s in addl_sentences:
            bpe_sentence += ' </s>' if not no_separator else ''
            bpe_sentence += ' ' + self.bpe.encode(s) + ' </s>'
        tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False, add_if_not_exist=False)
        return tokens.long()

    def decode(self, tokens: 'torch.LongTensor'):
        assert tokens.dim() == 1
        tokens = tokens.numpy()
        if tokens[0] == self.task.source_dictionary.bos():
            tokens = tokens[1:]
        eos_mask = tokens == self.task.source_dictionary.eos()
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [self.bpe.decode(self.task.source_dictionary.string(s)) for s in sentences]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    def extract_features(self, tokens: 'torch.LongTensor', return_all_hiddens: 'bool'=False) ->torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > self.model.max_positions():
            raise ValueError('tokens exceeds maximum length: {} > {}'.format(tokens.size(-1), self.model.max_positions()))
        features, extra = self.model(tokens, features_only=True, return_all_hiddens=return_all_hiddens)
        if return_all_hiddens:
            inner_states = extra['inner_states']
            return [inner_state.transpose(0, 1) for inner_state in inner_states]
        else:
            return features

    def register_classification_head(self, name: 'str', num_classes: 'int'=None, embedding_size: 'int'=None, **kwargs):
        self.model.register_classification_head(name, num_classes=num_classes, embedding_size=embedding_size, **kwargs)

    def predict(self, head: 'str', tokens: 'torch.LongTensor', return_logits: 'bool'=False):
        features = self.extract_features(tokens)
        logits = self.model.classification_heads[head](features)
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)

    def extract_features_aligned_to_words(self, sentence: 'str', return_all_hiddens: 'bool'=False) ->torch.Tensor:
        """Extract RoBERTa features, aligned to spaCy's word-level tokenizer."""
        nlp = alignment_utils.spacy_nlp()
        tokenizer = alignment_utils.spacy_tokenizer()
        bpe_toks = self.encode(sentence)
        spacy_toks = tokenizer(sentence)
        spacy_toks_ws = [t.text_with_ws for t in tokenizer(sentence)]
        alignment = alignment_utils.align_bpe_to_words(self, bpe_toks, spacy_toks_ws)
        features = self.extract_features(bpe_toks, return_all_hiddens=return_all_hiddens)
        features = features.squeeze(0)
        aligned_feats = alignment_utils.align_features_to_words(self, features, alignment)
        doc = Doc(nlp.vocab, words=['<s>'] + [x.text for x in spacy_toks] + ['</s>'], spaces=[True] + [x.endswith(' ') for x in spacy_toks_ws[:-1]] + [True, False])
        assert len(doc) == aligned_feats.size(0)
        doc.user_token_hooks['vector'] = lambda token: aligned_feats[token.i]
        return doc

    def fill_mask(self, masked_input: 'str', topk: 'int'=5):
        masked_token = '<mask>'
        assert masked_token in masked_input and masked_input.count(masked_token) == 1, "Please add one {0} token for the input, eg: 'He is a {0} guy'".format(masked_token)
        text_spans = masked_input.split(masked_token)
        text_spans_bpe = ' {0} '.format(masked_token).join([self.bpe.encode(text_span.rstrip()) for text_span in text_spans]).strip()
        tokens = self.task.source_dictionary.encode_line('<s> ' + text_spans_bpe + ' </s>', append_eos=False, add_if_not_exist=False)
        masked_index = (tokens == self.task.mask_idx).nonzero()
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        with utils.model_eval(self.model):
            features, extra = self.model(tokens.long(), features_only=False, return_all_hiddens=False)
        logits = features[0, masked_index, :].squeeze()
        prob = logits.softmax(dim=0)
        values, index = prob.topk(k=topk, dim=0)
        topk_predicted_token_bpe = self.task.source_dictionary.string(index)
        topk_filled_outputs = []
        for index, predicted_token_bpe in enumerate(topk_predicted_token_bpe.split(' ')):
            predicted_token = self.bpe.decode(predicted_token_bpe)
            if predicted_token_bpe.startswith('▁'):
                predicted_token = ' ' + predicted_token
            if ' {0}'.format(masked_token) in masked_input:
                topk_filled_outputs.append((masked_input.replace(' {0}'.format(masked_token), predicted_token), values[index].item(), predicted_token))
            else:
                topk_filled_outputs.append((masked_input.replace(masked_token, predicted_token), values[index].item(), predicted_token))
        return topk_filled_outputs

    def disambiguate_pronoun(self, sentence: 'str') ->bool:
        """
        Usage::

            >>> disambiguate_pronoun('The _trophy_ would not fit in the brown suitcase because [it] was too big.')
            True

            >>> disambiguate_pronoun('The trophy would not fit in the brown suitcase because [it] was too big.')
            'The trophy'
        """
        assert hasattr(self.task, 'disambiguate_pronoun'), 'roberta.disambiguate_pronoun() requires a model trained with the WSC task.'
        with utils.model_eval(self.model):
            return self.task.disambiguate_pronoun(self.model, sentence, use_cuda=self.device.type == 'cuda')


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout, q_noise=0, qn_block_size=8, do_spectral_norm=False):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = apply_quant_noise_(nn.Linear(inner_dim, num_classes), q_noise, qn_block_size)
        if do_spectral_norm:
            if q_noise != 0:
                raise NotImplementedError('Attempting to use Spectral Normalization with Quant Noise. This is not officially supported')
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)
        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        if masked_tokens is not None:
            features = features[masked_tokens, :]
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = F.linear(x, self.weight) + self.bias
        return x


class RobertaEncoder(FairseqEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.args = args
        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(','))
        self.sentence_encoder = TransformerSentenceEncoder(padding_idx=dictionary.pad(), vocab_size=len(dictionary), num_encoder_layers=args.encoder_layers, embedding_dim=args.encoder_embed_dim, ffn_embedding_dim=args.encoder_ffn_embed_dim, num_attention_heads=args.encoder_attention_heads, dropout=args.dropout, attention_dropout=args.attention_dropout, activation_dropout=args.activation_dropout, layerdrop=args.encoder_layerdrop, max_seq_len=args.max_positions, num_segments=0, encoder_normalize_before=True, apply_bert_init=True, activation_fn=args.activation_fn, q_noise=args.quant_noise_pq, qn_block_size=args.quant_noise_pq_block_size)
        args.untie_weights_roberta = getattr(args, 'untie_weights_roberta', False)
        self.lm_head = RobertaLMHead(embed_dim=args.encoder_embed_dim, output_dim=len(dictionary), activation_fn=args.activation_fn, weight=self.sentence_encoder.embed_tokens.weight if not args.untie_weights_roberta else None)

    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, masked_tokens=None, **unused):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """
        x, extra = self.extract_features(src_tokens, return_all_hiddens=return_all_hiddens)
        if not features_only:
            x = self.output_layer(x, masked_tokens=masked_tokens)
        return x, extra

    def extract_features(self, src_tokens, return_all_hiddens=False, **kwargs):
        inner_states, _ = self.sentence_encoder(src_tokens, last_state_only=not return_all_hiddens, token_embeddings=kwargs.get('token_embeddings', None))
        features = inner_states[-1].transpose(0, 1)
        return features, {'inner_states': inner_states if return_all_hiddens else None}

    def output_layer(self, features, masked_tokens=None, **unused):
        return self.lm_head(features, masked_tokens)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions


class RobertaModel(FairseqEncoderModel):

    @classmethod
    def hub_models(cls):
        return {'roberta.base': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz', 'roberta.large': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz', 'roberta.large.mnli': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz', 'roberta.large.wsc': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.wsc.tar.gz'}

    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args
        self.apply(init_bert_params)
        self.classification_heads = nn.ModuleDict()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--encoder-layers', type=int, metavar='L', help='num encoder layers')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='H', help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='F', help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='A', help='num encoder attention heads')
        parser.add_argument('--activation-fn', choices=utils.get_available_activation_fns(), help='activation function to use')
        parser.add_argument('--pooler-activation-fn', choices=utils.get_available_activation_fns(), help='activation function to use for pooler layer')
        parser.add_argument('--encoder-normalize-before', action='store_true', help='apply layernorm before each encoder block')
        parser.add_argument('--dropout', type=float, metavar='D', help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D', help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', type=float, metavar='D', help='dropout probability after activation in FFN')
        parser.add_argument('--pooler-dropout', type=float, metavar='D', help='dropout probability in the masked_lm pooler layers')
        parser.add_argument('--max-positions', type=int, help='number of positional embeddings to learn')
        parser.add_argument('--load-checkpoint-heads', action='store_true', help='(re-)register and load heads when loading checkpoints')
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0, help='LayerDrop probability for encoder')
        parser.add_argument('--encoder-layers-to-keep', default=None, help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0, help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8, help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0, help='scalar quantization noise and scalar quantization at training time')
        parser.add_argument('--untie-weights-roberta', action='store_true', help='Untie weights between embeddings and classifiers in RoBERTa')
        parser.add_argument('--spectral-norm-classification-head', action='store_true', default=False, help='Apply spectral normalization on the classification head')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)
        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample
        encoder = RobertaEncoder(args, task.source_dictionary)
        return cls(args, encoder)

    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, classification_head_name=None, **kwargs):
        if classification_head_name is not None:
            features_only = True
        x, extra = self.encoder(src_tokens, features_only, return_all_hiddens, **kwargs)
        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        return x, extra

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning('re-registering head "{}" with num_classes {} (prev: {}) and inner_dim {} (prev: {})'.format(name, num_classes, prev_num_classes, inner_dim, prev_inner_dim))
        self.classification_heads[name] = RobertaClassificationHead(input_dim=self.args.encoder_embed_dim, inner_dim=inner_dim or self.args.encoder_embed_dim, num_classes=num_classes, activation_fn=self.args.pooler_activation_fn, pooler_dropout=self.args.pooler_dropout, q_noise=self.args.quant_noise_pq, qn_block_size=self.args.quant_noise_pq_block_size, do_spectral_norm=self.args.spectral_norm_classification_head)

    @property
    def supported_targets(self):
        return {'self'}

    @classmethod
    def from_pretrained(cls, model_name_or_path, checkpoint_file='model.pt', data_name_or_path='.', bpe='gpt2', **kwargs):
        x = hub_utils.from_pretrained(model_name_or_path, checkpoint_file, data_name_or_path, archive_map=cls.hub_models(), bpe=bpe, load_checkpoint_heads=True, **kwargs)
        cls.upgrade_args(x['args'])
        logger.info(x['args'])
        return RobertaHubInterface(x['args'], x['task'], x['models'][0])

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != '' else ''
        for k in list(state_dict.keys()):
            if k.startswith(prefix + 'decoder'):
                new_k = prefix + 'encoder' + k[len(prefix + 'decoder'):]
                state_dict[new_k] = state_dict[k]
                del state_dict[k]
        super().upgrade_state_dict_named(state_dict, name)
        current_head_names = [] if not hasattr(self, 'classification_heads') else self.classification_heads.keys()
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue
            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
            inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)
            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            elif head_name not in current_head_names:
                logger.warning('deleting classification head ({}) from checkpoint not present in current model: {}'.format(head_name, k))
                keys_to_delete.append(k)
            elif num_classes != self.classification_heads[head_name].out_proj.out_features or inner_dim != self.classification_heads[head_name].dense.out_features:
                logger.warning('deleting classification head ({}) from checkpoint with different dimensions than current model: {}'.format(head_name, k))
                keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    logger.info('Overwriting ' + prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v


class BerardEncoder(FairseqEncoder):

    def __init__(self, input_layers: 'List[int]', conv_layers: 'List[Tuple[int]]', in_channels: 'int', input_feat_per_channel: 'int', num_blstm_layers: 'int', lstm_size: 'int', dropout: 'float'):
        """
        Args:
            input_layers: list of linear layer dimensions. These layers are
                applied to the input features and are followed by tanh and
                possibly dropout.
            conv_layers: list of conv2d layer configurations. A configuration is
                a tuple (out_channels, conv_kernel_size, stride).
            in_channels: number of input channels.
            input_feat_per_channel: number of input features per channel. These
                are speech features, typically 40 or 80.
            num_blstm_layers: number of bidirectional LSTM layers.
            lstm_size: size of the LSTM hidden (and cell) size.
            dropout: dropout probability. Dropout can be applied after the
                linear layers and LSTM layers but not to the convolutional
                layers.
        """
        super().__init__(None)
        self.input_layers = nn.ModuleList()
        in_features = input_feat_per_channel
        for out_features in input_layers:
            if dropout > 0:
                self.input_layers.append(nn.Sequential(nn.Linear(in_features, out_features), nn.Dropout(p=dropout)))
            else:
                self.input_layers.append(nn.Linear(in_features, out_features))
            in_features = out_features
        self.in_channels = in_channels
        self.input_dim = input_feat_per_channel
        self.conv_kernel_sizes_and_strides = []
        self.conv_layers = nn.ModuleList()
        lstm_input_dim = input_layers[-1]
        for conv_layer in conv_layers:
            out_channels, conv_kernel_size, conv_stride = conv_layer
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, conv_kernel_size, stride=conv_stride, padding=conv_kernel_size // 2))
            self.conv_kernel_sizes_and_strides.append((conv_kernel_size, conv_stride))
            in_channels = out_channels
            lstm_input_dim //= conv_stride
        lstm_input_dim *= conv_layers[-1][0]
        self.lstm_size = lstm_size
        self.num_blstm_layers = num_blstm_layers
        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=lstm_size, num_layers=num_blstm_layers, dropout=dropout, bidirectional=True)
        self.output_dim = 2 * lstm_size
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        """
        Args
            src_tokens: padded tensor (B, T, C * feat)
            src_lengths: tensor of original lengths of input utterances (B,)
        """
        bsz, max_seq_len, _ = src_tokens.size()
        x = src_tokens.view(bsz, max_seq_len, self.in_channels, self.input_dim).transpose(1, 2).contiguous()
        for input_layer in self.input_layers:
            x = input_layer(x)
            x = torch.tanh(x)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        bsz, _, output_seq_len, _ = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous().view(output_seq_len, bsz, -1)
        input_lengths = src_lengths.clone()
        for k, s in self.conv_kernel_sizes_and_strides:
            p = k // 2
            input_lengths = (input_lengths.float() + 2 * p - k) / s + 1
            input_lengths = input_lengths.floor().long()
        packed_x = nn.utils.rnn.pack_padded_sequence(x, input_lengths)
        h0 = x.new(2 * self.num_blstm_layers, bsz, self.lstm_size).zero_()
        c0 = x.new(2 * self.num_blstm_layers, bsz, self.lstm_size).zero_()
        packed_outs, _ = self.lstm(packed_x, (h0, c0))
        x, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_outs)
        if self.dropout is not None:
            x = self.dropout(x)
        encoder_padding_mask = lengths_to_padding_mask(output_lengths).t()
        return {'encoder_out': x, 'encoder_padding_mask': encoder_padding_mask}

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out['encoder_out'] = encoder_out['encoder_out'].index_select(1, new_order)
        encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].index_select(1, new_order)
        return encoder_out


class BerardModel(FairseqEncoderDecoderModel):
    """Implementation of a model similar to https://arxiv.org/abs/1802.04200

    Paper title: End-to-End Automatic Speech Translation of Audiobooks
    An implementation is available in tensorflow at
    https://github.com/eske/seq2seq
    Relevant files in this implementation are the config
    (https://github.com/eske/seq2seq/blob/master/config/LibriSpeech/AST.yaml)
    and the model code
    (https://github.com/eske/seq2seq/blob/master/translate/models.py).
    The encoder and decoder try to be close to the original implementation.
    The attention is an MLP as in Bahdanau et al.
    (https://arxiv.org/abs/1409.0473).
    There is no state initialization by averaging the encoder outputs.
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--input-layers', type=str, metavar='EXPR', help='List of linear layer dimensions. These layers are applied to the input features and are followed by tanh and possibly dropout.')
        parser.add_argument('--dropout', type=float, metavar='D', help='Dropout probability to use in the encoder/decoder. Note that this parameters control dropout in various places, there is no fine-grained control for dropout for embeddings vs LSTM layers for example.')
        parser.add_argument('--in-channels', type=int, metavar='N', help='Number of encoder input channels. Typically value is 1.')
        parser.add_argument('--conv-layers', type=str, metavar='EXPR', help='List of conv layers (format: (channels, kernel, stride)).')
        parser.add_argument('--num-blstm-layers', type=int, metavar='N', help='Number of encoder bi-LSTM layers.')
        parser.add_argument('--lstm-size', type=int, metavar='N', help='LSTM hidden size.')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N', help='Embedding dimension of the decoder target tokens.')
        parser.add_argument('--decoder-hidden-dim', type=int, metavar='N', help='Decoder LSTM hidden dimension.')
        parser.add_argument('--decoder-num-layers', type=int, metavar='N', help='Number of decoder LSTM layers.')
        parser.add_argument('--attention-dim', type=int, metavar='N', help='Hidden layer dimension in MLP attention.')
        parser.add_argument('--output-layer-dim', type=int, metavar='N', help='Hidden layer dim for linear layer prior to output projection.')
        parser.add_argument('--load-pretrained-encoder-from', type=str, metavar='STR', help='model to take encoder weights from (for initialization)')
        parser.add_argument('--load-pretrained-decoder-from', type=str, metavar='STR', help='model to take decoder weights from (for initialization)')

    @classmethod
    def build_encoder(cls, args, task):
        encoder = BerardEncoder(input_layers=literal_eval(args.input_layers), conv_layers=literal_eval(args.conv_layers), in_channels=args.input_channels, input_feat_per_channel=args.input_feat_per_channel, num_blstm_layers=args.num_blstm_layers, lstm_size=args.lstm_size, dropout=args.dropout)
        if getattr(args, 'load_pretrained_encoder_from', None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(component=encoder, checkpoint=args.load_pretrained_encoder_from)
        return encoder

    @classmethod
    def build_decoder(cls, args, task):
        decoder = LSTMDecoder(dictionary=task.target_dictionary, embed_dim=args.decoder_embed_dim, num_layers=args.decoder_num_layers, hidden_size=args.decoder_hidden_dim, dropout=args.dropout, encoder_output_dim=2 * args.lstm_size, attention_dim=args.attention_dim, output_layer_dim=args.output_layer_dim)
        if getattr(args, 'load_pretrained_decoder_from', None):
            decoder = checkpoint_utils.load_pretrained_component_from_model(component=decoder, checkpoint=args.load_pretrained_decoder_from)
        return decoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        encoder = cls.build_encoder(args, task)
        decoder = cls.build_decoder(args, task)
        return cls(encoder, decoder)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        lprobs = super().get_normalized_probs(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs


class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(self, in_channels: 'int', mid_channels: 'int', out_channels: 'int', kernel_sizes: 'List[int]'=(3, 3)):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(nn.Conv1d(in_channels if i == 0 else mid_channels // 2, mid_channels if i < self.n_layers - 1 else out_channels * 2, k, stride=2, padding=k // 2) for i, k in enumerate(kernel_sizes))

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()
        x = src_tokens.transpose(1, 2).contiguous()
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        return x, self.get_out_seq_lens_tensor(src_lengths)


EncoderOut = NamedTuple('EncoderOut', [('encoder_out', Tensor), ('encoder_padding_mask', Optional[Tensor]), ('encoder_embedding', Optional[Tensor]), ('encoder_states', Optional[List[Tensor]]), ('src_tokens', Optional[Tensor]), ('src_lengths', Optional[Tensor])])


class S2TTransformerEncoder(FairseqEncoder):
    """Speech-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args):
        super().__init__(None)
        self.dropout_module = FairseqDropout(p=args.dropout, module_name=self.__class__.__name__)
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1
        self.subsample = Conv1dSubsampler(args.input_feat_per_channel * args.input_channels, args.conv_channels, args.encoder_embed_dim, [int(k) for k in args.conv_kernel_sizes.split(',')])
        self.embed_positions = PositionalEmbedding(args.max_source_positions, args.encoder_embed_dim, self.padding_idx)
        self.transformer_layers = nn.ModuleList([TransformerEncoderLayer(args) for _ in range(args.encoder_layers)])
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

    def forward(self, src_tokens, src_lengths):
        x, input_lengths = self.subsample(src_tokens, src_lengths)
        x = self.embed_scale * x
        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        x += positions
        x = self.dropout_module(x)
        for layer in self.transformer_layers:
            x = layer(x, encoder_padding_mask)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return EncoderOut(encoder_out=x, encoder_padding_mask=encoder_padding_mask, encoder_embedding=None, encoder_states=None, src_tokens=None, src_lengths=None)

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: 'EncoderOut', new_order):
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """
        encoder_padding_mask: 'Optional[Tensor]' = encoder_out.encoder_padding_mask
        encoder_embedding: 'Optional[Tensor]' = encoder_out.encoder_embedding
        new_encoder_out = encoder_out.encoder_out if encoder_out.encoder_out is None else encoder_out.encoder_out.index_select(1, new_order)
        new_encoder_padding_mask = encoder_padding_mask if encoder_padding_mask is None else encoder_padding_mask.index_select(0, new_order)
        new_encoder_embedding = encoder_embedding if encoder_embedding is None else encoder_embedding.index_select(0, new_order)
        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)
        return EncoderOut(encoder_out=new_encoder_out, encoder_padding_mask=new_encoder_padding_mask, encoder_embedding=new_encoder_embedding, encoder_states=encoder_states, src_tokens=None, src_lengths=None)


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)
        self.dropout = args.decoder_dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed
        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_embed_dim
        args.encoder_embed_dim = embed_dim
        self.layerdrop = args.decoder_layerdrop
        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions
        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None
        self.embed_positions = PositionalEmbedding(args.max_target_positions, embed_dim, padding_idx, learned=args.decoder_learned_pos) if not args.no_token_positional_embeddings else None
        args = copy.deepcopy(args)
        args.dropout = args.decoder_dropout
        args.attention_dropout = args.decoder_attention_dropout
        args.activation_dropout = args.decoder_activation_dropout
        self.layers = nn.ModuleList([])
        self.layers.extend([TransformerDecoderLayer(args, no_encoder_attn) for _ in range(args.decoder_layers)])
        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)
        if args.decoder_normalize_before and not getattr(args, 'no_decoder_final_norm', False):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        prev_output_tokens = prev_output_tokens.long()
        x, extra = self.extract_features(prev_output_tokens, encoder_out, incremental_state)
        x = self.output_layer(x)
        return x, extra

    def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        positions = self.embed_positions(prev_output_tokens, incremental_state=incremental_state) if self.embed_positions is not None else None
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)
        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]
        for layer in self.layers:
            dropout_probability = np.random.random()
            if not self.training or dropout_probability > self.layerdrop:
                x, attn, _ = layer(x, encoder_out['encoder_out'] if encoder_out is not None else None, encoder_out['encoder_padding_mask'] if encoder_out is not None else None, incremental_state, self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None)
                inner_states.append(x)
        if self.layer_norm:
            x = self.layer_norm(x)
        x = x.transpose(0, 1)
        return x, {'attn': attn, 'inner_states': inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.share_input_output_embed:
            return F.linear(features, self.embed_tokens.weight)
        else:
            return F.linear(features, self.embed_out)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device or self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


class TransformerDecoderScriptable(TransformerDecoder):

    def extract_features(self, prev_output_tokens, encoder_out: 'Optional[EncoderOut]'=None, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]'=None, full_context_alignment: 'bool'=False, alignment_layer: 'Optional[int]'=None, alignment_heads: 'Optional[int]'=None):
        x, _ = self.extract_features_scriptable(prev_output_tokens, encoder_out, incremental_state, full_context_alignment, alignment_layer, alignment_heads)
        return x, None


class S2TTransformerModel(FairseqEncoderDecoderModel):
    """Adapted Transformer model (https://arxiv.org/abs/1706.03762) for
    speech-to-text tasks. The Transformer encoder/decoder remains the same.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as downsample input
    sequence for computational efficiency."""

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--conv-kernel-sizes', type=str, metavar='N', help='kernel sizes of Conv1d subsampling layers')
        parser.add_argument('--conv-channels', type=int, metavar='N', help='# of channels in Conv1d subsampling layers')
        parser.add_argument('--activation-fn', type=str, default='relu', choices=utils.get_available_activation_fns(), help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D', help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D', help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D', help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N', help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N', help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N', help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N', help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true', help='apply layernorm before each encoder block')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N', help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N', help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N', help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N', help='num decoder attention heads')
        parser.add_argument('--decoder-normalize-before', action='store_true', help='apply layernorm before each decoder block')
        parser.add_argument('--layernorm-embedding', action='store_true', help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true', help='if True, dont scale embeddings')
        parser.add_argument('--load-pretrained-encoder-from', type=str, metavar='STR', help='model to take encoder weights from (for initialization)')

    @classmethod
    def build_encoder(cls, args):
        encoder = S2TTransformerEncoder(args)
        if getattr(args, 'load_pretrained_encoder_from', None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(component=encoder, checkpoint=args.load_pretrained_encoder_from)
            logger.info(f'loaded pretrained encoder from: {args.load_pretrained_encoder_from}')
        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        return TransformerDecoderScriptable(args, task.target_dictionary, embed_tokens)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)
        decoder_embed_tokens = build_embedding(task.target_dictionary, args.decoder_embed_dim)
        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        return cls(encoder, decoder)

    def get_normalized_probs(self, net_output: 'Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]]', log_probs: 'bool', sample: 'Optional[Dict[str, Tensor]]'=None):
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(prev_output_tokens=prev_output_tokens, encoder_out=encoder_out)
        return decoder_out


class Wav2VecEncoder(FairseqEncoder):

    def __init__(self, args, tgt_dict=None):
        self.apply_mask = args.apply_mask
        arg_overrides = {'dropout': args.dropout, 'activation_dropout': args.activation_dropout, 'dropout_input': args.dropout_input, 'attention_dropout': args.attention_dropout, 'mask_length': args.mask_length, 'mask_prob': args.mask_prob, 'mask_selection': args.mask_selection, 'mask_other': args.mask_other, 'no_mask_overlap': args.no_mask_overlap, 'mask_channel_length': args.mask_channel_length, 'mask_channel_prob': args.mask_channel_prob, 'mask_channel_selection': args.mask_channel_selection, 'mask_channel_other': args.mask_channel_other, 'no_mask_channel_overlap': args.no_mask_channel_overlap, 'encoder_layerdrop': args.layerdrop, 'feature_grad_mult': args.feature_grad_mult}
        if getattr(args, 'w2v_args', None) is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(args.w2v_path, arg_overrides)
            w2v_args = state['args']
        else:
            state = None
            w2v_args = args.w2v_args
        assert args.normalize == w2v_args.normalize, 'Fine-tuning works best when data normalization is the same'
        w2v_args.data = args.data
        task = tasks.setup_task(w2v_args)
        model = task.build_model(w2v_args)
        if state is not None and not args.no_pretrained_weights:
            model.load_state_dict(state['model'], strict=True)
        model.remove_pretraining_modules()
        super().__init__(task.source_dictionary)
        d = w2v_args.encoder_embed_dim
        self.w2v_model = model
        self.final_dropout = nn.Dropout(args.final_dropout)
        self.freeze_finetune_updates = args.freeze_finetune_updates
        self.num_updates = 0
        if tgt_dict is not None:
            self.proj = Linear(d, len(tgt_dict))
        elif getattr(args, 'decoder_embed_dim', d) != d:
            self.proj = Linear(d, args.decoder_embed_dim)
        else:
            self.proj = None

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, **kwargs):
        w2v_args = {'source': source, 'padding_mask': padding_mask, 'mask': self.apply_mask and self.training}
        ft = self.freeze_finetune_updates <= self.num_updates
        with (torch.no_grad() if not ft else contextlib.ExitStack()):
            x, padding_mask = self.w2v_model.extract_features(**w2v_args)
            if tbc:
                x = x.transpose(0, 1)
        x = self.final_dropout(x)
        if self.proj:
            x = self.proj(x)
        return {'encoder_out': x, 'encoder_padding_mask': padding_mask, 'padding_mask': padding_mask}

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


def add_common_args(parser):
    parser.add_argument('--w2v-path', help='path to wav2vec 2.0 model')
    parser.add_argument('--no-pretrained-weights', action='store_true', help='if true, does not load pretrained weights')
    parser.add_argument('--dropout-input', type=float, metavar='D', help='dropout to apply to the input (after feat extr)')
    parser.add_argument('--final-dropout', type=float, metavar='D', help='dropout after transformer and before final projection')
    parser.add_argument('--apply-mask', action='store_true', help='apply masking during fine-tuning')
    parser.add_argument('--dropout', type=float, metavar='D', help='dropout probability inside wav2vec 2.0 model')
    parser.add_argument('--attention-dropout', type=float, metavar='D', help='dropout probability for attention weights inside wav2vec 2.0 model')
    parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D', help='dropout probability after activation in FFN inside wav2vec 2.0 model')
    parser.add_argument('--mask-length', type=int, help='repeat the mask indices multiple times')
    parser.add_argument('--mask-prob', type=float, help='probability of replacing a token with mask')
    parser.add_argument('--mask-selection', type=str, choices=['static', 'uniform', 'normal', 'poisson'], help='how to choose masks')
    parser.add_argument('--mask-other', type=float, help="stdev of the mask length in case of 'normal' selection strategy")
    parser.add_argument('--no-mask-overlap', action='store_true', help='whether to allow masks to overlap')
    parser.add_argument('--mask-channel-length', type=int, help='repeat the mask indices multiple times')
    parser.add_argument('--mask-channel-prob', type=float, help='probability of replacing a token with mask')
    parser.add_argument('--mask-channel-selection', type=str, choices=['static', 'uniform', 'normal', 'poisson'], help='how to choose masks')
    parser.add_argument('--mask-channel-other', type=float, help="stdev of the mask length in case of 'normal' selection strategy")
    parser.add_argument('--no-mask-channel-overlap', action='store_true', help='whether to allow masks to overlap')
    parser.add_argument('--freeze-finetune-updates', default=0, type=int, help='dont finetune wav2vec for this many updates')
    parser.add_argument('--feature-grad-mult', default=None, type=float, help='reset feature grad mult in wav2vec 2.0 to this')
    parser.add_argument('--layerdrop', default=0.0, type=float, help='probability of dropping a layer in wav2vec 2.0')


class TransformerModel(FairseqEncoderDecoderModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        add_common_args(parser)
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N', help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N', help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N', help='num decoder layers')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', help='decoder layerdrop chance')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N', help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true', help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true', help='apply layernorm before each decoder block')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true', help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--decoder-dropout', type=float, metavar='D', help='dropout probability in the decoder')
        parser.add_argument('--decoder-attention-dropout', type=float, metavar='D', help='dropout probability for attention weights inside the decoder')
        parser.add_argument('--decoder-activation-dropout', type=float, metavar='D', help='dropout probability after activation in FFN inside the decoder')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)
        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 2048
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 2048
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb
        decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim)
        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return TransformerModel(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args):
        return Wav2VecEncoder(args)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(args, tgt_dict, embed_tokens)

    def forward(self, **kwargs):
        encoder_out = self.encoder(tbc=False, **kwargs)
        decoder_out = self.decoder(encoder_out=encoder_out, **kwargs)
        return decoder_out

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict


class SamePad(nn.Module):

    def __init__(self, kernel_size):
        super().__init__()
        self.remove = kernel_size % 2 == 0

    def forward(self, x):
        if self.remove:
            x = x[:, :, :-1]
        return x


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
        self.layers = nn.ModuleList([TransformerSentenceEncoderLayer(embedding_dim=self.embedding_dim, ffn_embedding_dim=args.encoder_ffn_embed_dim, num_attention_heads=args.encoder_attention_heads, dropout=self.dropout, attention_dropout=args.attention_dropout, activation_dropout=args.activation_dropout, activation_fn=args.activation_fn, layer_norm_first=args.layer_norm_first) for _ in range(args.encoder_layers)])
        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop
        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None):
        x = self.extract_features(x, padding_mask)
        if self.layer_norm_first:
            x = self.layer_norm(x)
        return x

    def extract_features(self, x, padding_mask=None):
        if padding_mask is not None:
            x[padding_mask] = 0
        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x += x_conv
        if not self.layer_norm_first:
            x = self.layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)
        layer_results = []
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or dropout_probability > self.layerdrop:
                x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
                layer_results.append(x)
        x = x.transpose(0, 1)
        return x

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


def compute_mask_indices(shape: 'Tuple[int, int]', padding_mask: 'Optional[torch.Tensor]', mask_prob: 'float', mask_length: 'int', mask_type: 'str'='static', mask_other: 'float'=0.0, min_masks: 'int'=0, no_overlap: 'bool'=False, min_space: 'int'=0) ->np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
    """
    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)
    all_num_mask = int(mask_prob * all_sz / float(mask_length) + np.random.rand())
    all_num_mask = max(min_masks, all_num_mask)
    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(mask_prob * sz / float(mask_length) + np.random.rand())
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask
        if mask_type == 'static':
            lengths = np.full(num_mask, mask_length)
        elif mask_type == 'uniform':
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == 'normal':
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == 'poisson':
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception('unknown mask selection ' + mask_type)
        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)
        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))
                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts
            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter((e - s if e - s >= length + min_space else 0 for s, e in parts), np.int)
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1
            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)
            mask_idc = np.asarray([(mask_idc[j] + offset) for j in range(len(mask_idc)) for offset in range(lengths[j])])
        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))
    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True
    return mask


class Wav2Vec2Model(BaseFairseqModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--extractor-mode', choices=['default', 'layer_norm'], help='mode for feature extractor. default has a single group norm with d groups in the first conv block, whereas layer_norm has layer norms in every block (meant to use with --normalize)')
        parser.add_argument('--encoder-layers', type=int, metavar='L', help='num encoder layers in the transformer')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='H', help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='F', help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='A', help='num encoder attention heads')
        parser.add_argument('--activation-fn', choices=utils.get_available_activation_fns(), help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D', help='dropout probability for the transformer')
        parser.add_argument('--attention-dropout', type=float, metavar='D', help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', type=float, metavar='D', help='dropout probability after activation in FFN')
        parser.add_argument('--final-dim', type=int, metavar='D', help='project final representations and targets to this many dimensions')
        parser.add_argument('--layer-norm-first', action='store_true', help='apply layernorm first in the transformer')
        parser.add_argument('--encoder-layerdrop', type=float, help='probability of dropping a tarnsformer layer')
        parser.add_argument('--conv-feature-layers', type=str, metavar='EXPR', help='convolutional feature extraction layers [(dim, kernel_size, stride), ...]')
        parser.add_argument('--logit-temp', type=float, help='temperature to divide logits by')
        parser.add_argument('--quantize-targets', action='store_true', help='use quantized targets')
        parser.add_argument('--quantize-input', action='store_true', help='use quantized inputs')
        parser.add_argument('--same-quantizer', action='store_true', help='use same quantizer for inputs and targets')
        parser.add_argument('--feature-grad-mult', type=float, help='multiply feature extractor var grads by this')
        parser.add_argument('--latent-vars', type=int, metavar='N', help='number of latent variables V in each group of the codebook')
        parser.add_argument('--latent-groups', type=int, metavar='N', help='number of groups G of latent variables in the codebook')
        parser.add_argument('--latent-dim', type=int, metavar='N', help='if set, uses this dimensionality for latent variables. otherwise uses final_dim / latent_groups')
        parser.add_argument('--mask-length', type=int, help='mask length')
        parser.add_argument('--mask-prob', type=float, help='probability of replacing a token with mask')
        parser.add_argument('--mask-selection', type=str, choices=['static', 'uniform', 'normal', 'poisson'], help='how to choose masks')
        parser.add_argument('--mask-other', type=float, help='secondary mask argument (used for more complex distributions), see help in compute_mask_indices')
        parser.add_argument('--no-mask-overlap', action='store_true', help='whether to allow masks to overlap')
        parser.add_argument('--mask-min-space', type=int, help='min space between spans (if no overlap is enabled)')
        parser.add_argument('--mask-channel-length', type=int, help='repeat the mask indices multiple times')
        parser.add_argument('--mask-channel-prob', type=float, help='probability of replacing a token with mask')
        parser.add_argument('--mask-channel-selection', type=str, choices=['static', 'uniform', 'normal', 'poisson'], help='how to choose masks')
        parser.add_argument('--mask-channel-other', type=float, help='secondary mask argument (used for more complex distributions), see help in compute_mask_indices')
        parser.add_argument('--no-mask-channel-overlap', action='store_true', help='whether to allow masks to overlap')
        parser.add_argument('--mask-channel-min-space', type=int, help='min space between spans (if no overlap is enabled)')
        parser.add_argument('--dropout-input', type=float, metavar='D', help='dropout to apply to the input (after feat extr)')
        parser.add_argument('--dropout-features', type=float, metavar='D', help='dropout to apply to the features (after feat extr)')
        parser.add_argument('--num-negatives', type=int, metavar='N', help='number of negative examples')
        parser.add_argument('--negatives-from-everywhere', action='store_true', help='sample negatives from everywhere, not just masked states')
        parser.add_argument('--cross-sample-negatives', type=int, metavar='N', help='num of cross sampled negatives')
        parser.add_argument('--codebook-negatives', type=int, metavar='N', help='num of codebook sampled negatives')
        parser.add_argument('--conv-pos', type=int, metavar='N', help='number of filters for convolutional positional embeddings')
        parser.add_argument('--conv-pos-groups', type=int, metavar='N', help='number of groups for convolutional positional embedding')
        parser.add_argument('--latent-temp', type=str, metavar='D', help='temperature for latent variable sampling. can be tuple of 3 values (start, end, decay)')
        parser.add_argument('--target-glu', action='store_true', help='adds projection + glu to targets')
        parser.add_argument('--conv-bias', action='store_true', help='include bias in conv encoder')

    def __init__(self, args):
        super().__init__()
        self.args = args
        feature_enc_layers = eval(args.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]
        self.feature_extractor = ConvFeatureExtractionModel(conv_layers=feature_enc_layers, dropout=0.0, mode=args.extractor_mode, conv_bias=args.conv_bias)
        self.post_extract_proj = nn.Linear(self.embed, args.encoder_embed_dim) if self.embed != args.encoder_embed_dim and not args.quantize_input else None
        self.mask_prob = args.mask_prob
        self.mask_selection = args.mask_selection
        self.mask_other = args.mask_other
        self.mask_length = args.mask_length
        self.no_mask_overlap = args.no_mask_overlap
        self.mask_min_space = args.mask_min_space
        self.mask_channel_prob = args.mask_channel_prob
        self.mask_channel_selection = args.mask_channel_selection
        self.mask_channel_other = args.mask_channel_other
        self.mask_channel_length = args.mask_channel_length
        self.no_mask_channel_overlap = args.no_mask_channel_overlap
        self.mask_channel_min_space = args.mask_channel_min_space
        self.dropout_input = nn.Dropout(args.dropout_input)
        self.dropout_features = nn.Dropout(args.dropout_features)
        self.feature_grad_mult = args.feature_grad_mult
        self.quantizer = None
        self.input_quantizer = None
        self.n_negatives = args.num_negatives
        self.cross_sample_negatives = args.cross_sample_negatives
        self.codebook_negatives = args.codebook_negatives
        self.negatives_from_everywhere = args.negatives_from_everywhere
        self.logit_temp = args.logit_temp
        final_dim = args.final_dim if args.final_dim > 0 else args.encoder_embed_dim
        if args.quantize_targets:
            vq_dim = args.latent_dim if args.latent_dim > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(dim=self.embed, num_vars=args.latent_vars, temp=eval(args.latent_temp), groups=args.latent_groups, combine_groups=False, vq_dim=vq_dim, time_first=True)
            self.project_q = nn.Linear(vq_dim, final_dim)
        else:
            self.project_q = nn.Linear(self.embed, final_dim)
        if args.quantize_input:
            if args.same_quantizer and self.quantizer is not None:
                vq_dim = final_dim
                self.input_quantizer = self.quantizer
            else:
                vq_dim = args.latent_dim if args.latent_dim > 0 else args.encoder_embed_dim
                self.input_quantizer = GumbelVectorQuantizer(dim=self.embed, num_vars=args.latent_vars, temp=eval(args.latent_temp), groups=args.latent_groups, combine_groups=False, vq_dim=vq_dim, time_first=True)
            self.project_inp = nn.Linear(vq_dim, args.encoder_embed_dim)
        self.mask_emb = nn.Parameter(torch.FloatTensor(args.encoder_embed_dim).uniform_())
        self.encoder = TransformerEncoder(args)
        self.layer_norm = LayerNorm(self.embed)
        self.target_glu = None
        if args.target_glu:
            self.target_glu = nn.Sequential(nn.Linear(final_dim, final_dim * 2), nn.GLU())
        self.final_proj = nn.Linear(args.encoder_embed_dim, final_dim)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    @classmethod
    def build_model(cls, args, task=None):
        """Build a new model instance."""
        base_architecture(args)
        return cls(args)

    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices((B, T), padding_mask, self.mask_prob, self.mask_length, self.mask_selection, self.mask_other, min_masks=2, no_overlap=self.no_mask_overlap, min_space=self.mask_min_space)
            mask_indices = torch.from_numpy(mask_indices)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None
        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices((B, C), None, self.mask_channel_prob, self.mask_channel_length, self.mask_channel_selection, self.mask_channel_other, no_overlap=self.no_mask_channel_overlap, min_space=self.mask_channel_min_space)
            mask_channel_indices = torch.from_numpy(mask_channel_indices).unsqueeze(1).expand(-1, T, -1)
            x[mask_channel_indices] = 0
        return x, mask_indices

    def sample_negatives(self, y, num):
        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)
        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)
        cross_high = tsz * bsz
        high = tsz
        with torch.no_grad():
            assert high > 1, f'{bsz, tsz, fsz}'
            if self.n_negatives > 0:
                tszs = buffered_arange(num).unsqueeze(-1).expand(-1, self.n_negatives).flatten()
                neg_idxs = torch.randint(low=0, high=high - 1, size=(bsz, self.n_negatives * num))
                neg_idxs[neg_idxs >= tszs] += 1
            if self.cross_sample_negatives > 0:
                tszs = buffered_arange(num).unsqueeze(-1).expand(-1, self.cross_sample_negatives).flatten()
                cross_neg_idxs = torch.randint(low=0, high=cross_high - 1, size=(bsz, self.cross_sample_negatives * num))
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1
        if self.n_negatives > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs
        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)
        negs = y[neg_idxs.view(-1)]
        negs = negs.view(bsz, num, self.n_negatives + self.cross_sample_negatives, fsz).permute(2, 0, 1, 3)
        return negs, neg_idxs

    def compute_preds(self, x, y, negatives):
        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)
        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float('-inf')
        return logits

    def forward(self, source, padding_mask=None, mask=True, features_only=False):
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        features_pen = features.float().pow(2).mean()
        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()
        if padding_mask is not None:
            extra = padding_mask.size(1) % features.size(1)
            if extra > 0:
                padding_mask = padding_mask[:, :-extra]
            padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
            padding_mask = padding_mask.all(-1)
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)
        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None
        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets=False)
            features = q['x']
            num_vars = q['num_vars']
            code_ppl = q['code_perplexity']
            prob_ppl = q['prob_perplexity']
            curr_temp = q['temp']
            features = self.project_inp(features)
        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask)
            if mask_indices is not None:
                y = unmasked_features[mask_indices].view(unmasked_features.size(0), -1, unmasked_features.size(-1))
            else:
                y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None
        x = self.encoder(x, padding_mask=padding_mask)
        if features_only:
            return {'x': x, 'padding_mask': padding_mask}
        if self.quantizer:
            q = self.quantizer(y, produce_targets=False)
            y = q['x']
            num_vars = q['num_vars']
            code_ppl = q['code_perplexity']
            prob_ppl = q['prob_perplexity']
            curr_temp = q['temp']
            y = self.project_q(y)
            if self.negatives_from_everywhere:
                neg_cands, *_ = self.quantizer(unmasked_features, produce_targets=False)
                negs, _ = self.sample_negatives(neg_cands, y.size(1))
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(y, y.size(1))
            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(y.size(0) * y.size(1), self.codebook_negatives)
                cb_negs = cb_negs.view(self.codebook_negatives, y.size(0), y.size(1), -1)
                cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            y = self.project_q(y)
            if self.negatives_from_everywhere:
                negs, _ = self.sample_negatives(unmasked_features, y.size(1))
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(y, y.size(1))
        x = x[mask_indices].view(x.size(0), -1, x.size(-1))
        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)
        x = self.final_proj(x)
        x = self.compute_preds(x, y, negs)
        result = {'x': x, 'padding_mask': padding_mask, 'features_pen': features_pen}
        if prob_ppl is not None:
            result['prob_perplexity'] = prob_ppl
            result['code_perplexity'] = code_ppl
            result['num_vars'] = num_vars
            result['temp'] = curr_temp
        return result

    def quantize(self, x):
        assert self.quantizer is not None
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return self.quantizer.forward_idx(x)

    def extract_features(self, source, padding_mask, mask=False):
        res = self.forward(source, padding_mask, mask=mask, features_only=True)
        return res['x'], res['padding_mask']

    def get_logits(self, net_output):
        logits = net_output['x']
        logits = logits.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))
        return logits

    def get_targets(self, sample, net_output, expand_steps=True):
        x = net_output['x']
        return x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)

    def get_extra_losses(self, net_output):
        pen = []
        if 'prob_perplexity' in net_output:
            pen.append((net_output['num_vars'] - net_output['prob_perplexity']) / net_output['num_vars'])
        if 'features_pen' in net_output:
            pen.append(net_output['features_pen'])
        return pen

    def remove_pretraining_modules(self):
        self.quantizer = None
        self.project_q = None
        self.target_glu = None
        self.final_proj = None


class Wav2VecCtc(BaseFairseqModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        add_common_args(parser)

    def __init__(self, w2v_encoder, args):
        super().__init__()
        self.w2v_encoder = w2v_encoder
        self.args = args

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)
        w2v_encoder = Wav2VecEncoder(args, task.target_dictionary)
        return cls(w2v_encoder, args)

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output['encoder_out']
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x


class AdaptiveInput(nn.Module):

    def __init__(self, vocab_size: 'int', padding_idx: 'int', initial_dim: 'int', factor: 'float', output_dim: 'int', cutoff: 'List[int]', q_noise: 'float'=0, qn_block_size: 'int'=8):
        super().__init__()
        if vocab_size > cutoff[-1]:
            cutoff = cutoff + [vocab_size]
        else:
            assert vocab_size == cutoff[-1], 'cannot specify cutoff larger than vocab size'
        self.cutoff = cutoff
        self.embedding_dim = output_dim
        self.padding_idx = padding_idx
        self.embeddings = nn.ModuleList()
        for i in range(len(self.cutoff)):
            prev = self.cutoff[i - 1] if i > 0 else 0
            size = self.cutoff[i] - prev
            dim = int(initial_dim // factor ** i)
            seq = nn.Sequential(nn.Embedding(size, dim, self.padding_idx), quant_noise(nn.Linear(dim, output_dim, bias=False), q_noise, qn_block_size))
            self.embeddings.append(seq)
            self.padding_idx = None
        self.padding_idx = padding_idx

        def init_weights(m):
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=m.weight.shape[1] ** -0.5)
                nn.init.constant_(m.weight[padding_idx], 0)
            elif hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight)
        self.apply(init_weights)
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def weights_for_band(self, band: 'int'):
        return self.embeddings[band][0].weight, self.embeddings[band][1].weight

    def forward(self, input: 'torch.Tensor'):
        result = self._float_tensor.new(input.shape + (self.embedding_dim,))
        for i in range(len(self.cutoff)):
            mask = input.lt(self.cutoff[i])
            if i > 0:
                mask.mul_(input.ge(self.cutoff[i - 1]))
                chunk_input = input[mask] - self.cutoff[i - 1]
            else:
                chunk_input = input[mask]
            if mask.any():
                result[mask] = self.embeddings[i](chunk_input)
        return result


class BeamableMM(nn.Module):
    """This module provides an optimized MM for beam decoding with attention.

    It leverage the fact that the source-side of the input is replicated beam
    times and the target-side of the input is of width one. This layer speeds up
    inference by replacing the inputs {(bsz x 1 x nhu), (bsz x sz2 x nhu)}
    with smaller inputs {(bsz/beam x beam x nhu), (bsz/beam x sz2 x nhu)}.
    """

    def __init__(self, beam_size=None):
        super(BeamableMM, self).__init__()
        self.beam_size = beam_size

    def forward(self, input1, input2):
        if not self.training and self.beam_size is not None and input1.dim() == 3 and input1.size(1) == 1:
            bsz, beam = input1.size(0), self.beam_size
            input1 = input1[:, 0, :].unfold(0, beam, beam).transpose(2, 1)
            input2 = input2.unfold(0, beam, beam)[:, :, :, 0]
            if input1.size(0) == 1:
                output = torch.mm(input1[0, :, :], input2[0, :, :])
            else:
                output = input1.bmm(input2)
            return output.view(bsz, 1, -1)
        else:
            return input1.bmm(input2)

    def set_beam_size(self, beam_size):
        self.beam_size = beam_size


CHAR_EOS_IDX = 257


CHAR_PAD_IDX = 0


class PathManager:
    """
    Wrapper for insulating OSS I/O (using Python builtin operations) from
    fvcore's PathManager abstraction (for transparently handling various
    internal backends).
    """

    @staticmethod
    def open(path: 'str', mode: 'str'='r', buffering: 'int'=-1, encoding: 'Optional[str]'=None, errors: 'Optional[str]'=None, newline: 'Optional[str]'=None):
        if FVCorePathManager:
            return FVCorePathManager.open(path=path, mode=mode, buffering=buffering, encoding=encoding, errors=errors, newline=newline)
        return open(path, mode=mode, buffering=buffering, encoding=encoding, errors=errors, newline=newline)

    @staticmethod
    def copy(src_path: 'str', dst_path: 'str', overwrite: 'bool'=False) ->bool:
        if FVCorePathManager:
            return FVCorePathManager.copy(src_path=src_path, dst_path=dst_path, overwrite=overwrite)
        return shutil.copyfile(src_path, dst_path)

    @staticmethod
    def get_local_path(path: 'str', **kwargs) ->str:
        if FVCorePathManager:
            return FVCorePathManager.get_local_path(path, **kwargs)
        return path

    @staticmethod
    def exists(path: 'str') ->bool:
        if FVCorePathManager:
            return FVCorePathManager.exists(path)
        return os.path.exists(path)

    @staticmethod
    def isfile(path: 'str') ->bool:
        if FVCorePathManager:
            return FVCorePathManager.isfile(path)
        return os.path.isfile(path)

    @staticmethod
    def ls(path: 'str') ->List[str]:
        if FVCorePathManager:
            return FVCorePathManager.ls(path)
        return os.listdir(path)

    @staticmethod
    def mkdirs(path: 'str') ->None:
        if FVCorePathManager:
            return FVCorePathManager.mkdirs(path)
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def rm(path: 'str') ->None:
        if FVCorePathManager:
            return FVCorePathManager.rm(path)
        os.remove(path)

    @staticmethod
    def chmod(path: 'str', mode: 'int') ->None:
        if 'manifold' not in path:
            os.chmod(path, mode)

    @staticmethod
    def register_handler(handler) ->None:
        if FVCorePathManager:
            return FVCorePathManager.register_handler(handler=handler)

    @staticmethod
    def copy_from_local(local_path: 'str', dst_path: 'str', overwrite: 'bool'=False, **kwargs) ->None:
        if FVCorePathManager:
            return FVCorePathManager.copy_from_local(local_path=local_path, dst_path=dst_path, overwrite=overwrite, **kwargs)
        return shutil.copyfile(local_path, dst_path)


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)


SPACE_NORMALIZER = re.compile('\\s+')


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(' ', line)
    line = line.strip()
    return line.split()


class Dictionary(object):
    """A mapping from symbols to consecutive integers"""

    def __init__(self, *, bos='<s>', pad='<pad>', eos='</s>', unk='<unk>', extra_special_symbols=None):
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def string(self, tensor, bpe_symbol=None, escape_unk=False, extra_symbols_to_ignore=None, unk_string=None):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return '\n'.join(self.string(t, bpe_symbol, escape_unk, extra_symbols_to_ignore) for t in tensor)
        extra_symbols_to_ignore = set(extra_symbols_to_ignore or [])
        extra_symbols_to_ignore.add(self.eos())

        def token_string(i):
            if i == self.unk():
                if unk_string is not None:
                    return unk_string
                else:
                    return self.unk_string(escape_unk)
            else:
                return self[i]
        if hasattr(self, 'bos_index'):
            extra_symbols_to_ignore.add(self.bos())
        sent = ' '.join(token_string(i) for i in tensor if utils.item(i) not in extra_symbols_to_ignore)
        return data_utils.post_process(sent, bpe_symbol)

    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <<unk>>"""
        if escape:
            return '<{}>'.format(self.unk_word)
        else:
            return self.unk_word

    def add_symbol(self, word, n=1, overwrite=False):
        """Adds a word to the dictionary"""
        if word in self.indices and not overwrite:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def update(self, new_dict):
        """Updates counts from new dictionary."""
        for word in new_dict.symbols:
            idx2 = new_dict.indices[word]
            if word in self.indices:
                idx = self.indices[word]
                self.count[idx] = self.count[idx] + new_dict.count[idx2]
            else:
                idx = len(self.symbols)
                self.indices[word] = idx
                self.symbols.append(word)
                self.count.append(new_dict.count[idx2])

    def finalize(self, threshold=-1, nwords=-1, padding_factor=8):
        """Sort symbols by frequency in descending order, ignoring special ones.

        Args:
            - threshold defines the minimum word count
            - nwords defines the total number of words in the final dictionary,
                including special symbols
            - padding_factor can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        if nwords <= 0:
            nwords = len(self)
        new_indices = dict(zip(self.symbols[:self.nspecial], range(self.nspecial)))
        new_symbols = self.symbols[:self.nspecial]
        new_count = self.count[:self.nspecial]
        c = Counter(dict(sorted(zip(self.symbols[self.nspecial:], self.count[self.nspecial:]))))
        for symbol, count in c.most_common(nwords - self.nspecial):
            if count >= threshold:
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(count)
            else:
                break
        assert len(new_symbols) == len(new_indices)
        self.count = list(new_count)
        self.symbols = list(new_symbols)
        self.indices = new_indices
        self.pad_to_multiple_(padding_factor)

    def pad_to_multiple_(self, padding_factor):
        """Pad Dictionary size to be a multiple of *padding_factor*."""
        if padding_factor > 1:
            i = 0
            while len(self) % padding_factor != 0:
                symbol = 'madeupword{:04d}'.format(i)
                self.add_symbol(symbol, n=0)
                i += 1

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.bos_index

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index

    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f)
        return d

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, str):
            try:
                with open(PathManager.get_local_path(f), 'r', encoding='utf-8') as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception('Incorrect encoding detected in {}, please rebuild the dataset'.format(f))
            return
        lines = f.readlines()
        indices_start_line = self._load_meta(lines)
        for line in lines[indices_start_line:]:
            try:
                line, field = line.rstrip().rsplit(' ', 1)
                if field == '#fairseq:overwrite':
                    overwrite = True
                    line, field = line.rsplit(' ', 1)
                else:
                    overwrite = False
                count = int(field)
                word = line
                if word in self and not overwrite:
                    raise RuntimeError("Duplicate word found when loading Dictionary: '{}'. Duplicate words can overwrite earlier ones by adding the #fairseq:overwrite flag at the end of the corresponding row in the dictionary file. If using the Camembert model, please download an updated copy of the model file.".format(word))
                self.add_symbol(word, n=count, overwrite=overwrite)
            except ValueError:
                raise ValueError("Incorrect dictionary format, expected '<token> <cnt> [flags]'")

    def _save(self, f, kv_iterator):
        if isinstance(f, str):
            PathManager.mkdirs(os.path.dirname(f))
            with PathManager.open(f, 'w', encoding='utf-8') as fd:
                return self.save(fd)
        for k, v in kv_iterator:
            None

    def _get_meta(self):
        return [], []

    def _load_meta(self, lines):
        return 0

    def save(self, f):
        """Stores dictionary into a text file"""
        ex_keys, ex_vals = self._get_meta()
        self._save(f, zip(ex_keys + self.symbols[self.nspecial:], ex_vals + self.count[self.nspecial:]))

    def dummy_sentence(self, length):
        t = torch.Tensor(length).uniform_(self.nspecial + 1, len(self)).long()
        t[-1] = self.eos()
        return t

    def encode_line(self, line, line_tokenizer=tokenize_line, add_if_not_exist=True, consumer=None, append_eos=True, reverse_order=False):
        words = line_tokenizer(line)
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)
        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = self.add_symbol(word)
            else:
                idx = self.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = self.eos_index
        return ids

    @staticmethod
    def _add_file_to_dictionary_single_worker(filename, tokenize, eos_word, worker_id=0, num_workers=1):
        counter = Counter()
        with open(PathManager.get_local_path(filename), 'r', encoding='utf-8') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_workers
            offset = worker_id * chunk_size
            end = offset + chunk_size
            f.seek(offset)
            if offset > 0:
                safe_readline(f)
            line = f.readline()
            while line:
                for word in tokenize(line):
                    counter.update([word])
                counter.update([eos_word])
                if f.tell() > end:
                    break
                line = f.readline()
        return counter

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize, num_workers):

        def merge_result(counter):
            for w, c in sorted(counter.items()):
                dict.add_symbol(w, c)
        if num_workers > 1:
            pool = Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(pool.apply_async(Dictionary._add_file_to_dictionary_single_worker, (filename, tokenize, dict.eos_word, worker_id, num_workers)))
            pool.close()
            pool.join()
            for r in results:
                merge_result(r.get())
        else:
            merge_result(Dictionary._add_file_to_dictionary_single_worker(filename, tokenize, dict.eos_word))


class Highway(torch.nn.Module):
    """
    A `Highway layer <https://arxiv.org/abs/1505.00387>`_.
    Adopted from the AllenNLP implementation.
    """

    def __init__(self, input_dim: 'int', num_layers: 'int'=1):
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList([nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)])
        self.activation = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            nn.init.constant_(layer.bias[self.input_dim:], 1)
            nn.init.constant_(layer.bias[:self.input_dim], 0)
            nn.init.xavier_normal_(layer.weight)

    def forward(self, x: 'torch.Tensor'):
        for layer in self.layers:
            projection = layer(x)
            proj_x, gate = projection.chunk(2, dim=-1)
            proj_x = self.activation(proj_x)
            gate = torch.sigmoid(gate)
            x = gate * x + (gate.new_tensor([1]) - gate) * proj_x
        return x


class CharacterTokenEmbedder(torch.nn.Module):

    def __init__(self, vocab: 'Dictionary', filters: 'List[Tuple[int, int]]', char_embed_dim: 'int', word_embed_dim: 'int', highway_layers: 'int', max_char_len: 'int'=50, char_inputs: 'bool'=False):
        super(CharacterTokenEmbedder, self).__init__()
        self.onnx_trace = False
        self.embedding_dim = word_embed_dim
        self.max_char_len = max_char_len
        self.char_embeddings = nn.Embedding(257, char_embed_dim, padding_idx=0)
        self.symbol_embeddings = nn.Parameter(torch.FloatTensor(2, word_embed_dim))
        self.eos_idx, self.unk_idx = 0, 1
        self.char_inputs = char_inputs
        self.convolutions = nn.ModuleList()
        for width, out_c in filters:
            self.convolutions.append(nn.Conv1d(char_embed_dim, out_c, kernel_size=width))
        last_dim = sum(f[1] for f in filters)
        self.highway = Highway(last_dim, highway_layers) if highway_layers > 0 else None
        self.projection = nn.Linear(last_dim, word_embed_dim)
        assert vocab is not None or char_inputs, 'vocab must be set if not using char inputs'
        self.vocab = None
        if vocab is not None:
            self.set_vocab(vocab, max_char_len)
        self.reset_parameters()

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def set_vocab(self, vocab, max_char_len):
        word_to_char = torch.LongTensor(len(vocab), max_char_len)
        truncated = 0
        for i in range(len(vocab)):
            if i < vocab.nspecial:
                char_idxs = [0] * max_char_len
            else:
                chars = vocab[i].encode()
                char_idxs = [(c + 1) for c in chars] + [0] * (max_char_len - len(chars))
            if len(char_idxs) > max_char_len:
                truncated += 1
                char_idxs = char_idxs[:max_char_len]
            word_to_char[i] = torch.LongTensor(char_idxs)
        if truncated > 0:
            logger.info('truncated {} words longer than {} characters'.format(truncated, max_char_len))
        self.vocab = vocab
        self.word_to_char = word_to_char

    @property
    def padding_idx(self):
        return Dictionary().pad() if self.vocab is None else self.vocab.pad()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.char_embeddings.weight)
        nn.init.xavier_normal_(self.symbol_embeddings)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.constant_(self.char_embeddings.weight[self.char_embeddings.padding_idx], 0.0)
        nn.init.constant_(self.projection.bias, 0.0)

    def forward(self, input: 'torch.Tensor'):
        if self.char_inputs:
            chars = input.view(-1, self.max_char_len)
            pads = chars[:, 0].eq(CHAR_PAD_IDX)
            eos = chars[:, 0].eq(CHAR_EOS_IDX)
            if eos.any():
                if self.onnx_trace:
                    chars = torch.where(eos.unsqueeze(1), chars.new_zeros(1), chars)
                else:
                    chars[eos] = 0
            unk = None
        else:
            flat_words = input.view(-1)
            chars = self.word_to_char[flat_words.type_as(self.word_to_char)].type_as(input)
            pads = flat_words.eq(self.vocab.pad())
            eos = flat_words.eq(self.vocab.eos())
            unk = flat_words.eq(self.vocab.unk())
        word_embs = self._convolve(chars)
        if self.onnx_trace:
            if pads.any():
                word_embs = torch.where(pads.unsqueeze(1), word_embs.new_zeros(1), word_embs)
            if eos.any():
                word_embs = torch.where(eos.unsqueeze(1), self.symbol_embeddings[self.eos_idx], word_embs)
            if unk is not None and unk.any():
                word_embs = torch.where(unk.unsqueeze(1), self.symbol_embeddings[self.unk_idx], word_embs)
        else:
            if pads.any():
                word_embs[pads] = 0
            if eos.any():
                word_embs[eos] = self.symbol_embeddings[self.eos_idx]
            if unk is not None and unk.any():
                word_embs[unk] = self.symbol_embeddings[self.unk_idx]
        return word_embs.view(input.size()[:2] + (-1,))

    def _convolve(self, char_idxs: 'torch.Tensor'):
        char_embs = self.char_embeddings(char_idxs)
        char_embs = char_embs.transpose(1, 2)
        conv_result = []
        for conv in self.convolutions:
            x = conv(char_embs)
            x, _ = torch.max(x, -1)
            x = F.relu(x)
            conv_result.append(x)
        x = torch.cat(conv_result, dim=-1)
        if self.highway is not None:
            x = self.highway(x)
        x = self.projection(x)
        return x


def logsumexp(x, dim=1):
    return torch.logsumexp(x.float(), dim=dim).type_as(x)


class DynamicCRF(nn.Module):
    """Dynamic CRF layer is used to approximate the traditional
    Conditional Random Fields (CRF)
    $P(y | x) = 1/Z(x) exp(sum_i s(y_i, x) + sum_i t(y_{i-1}, y_i, x))$

    where in this function, we assume the emition scores (s) are given,
    and the transition score is a |V| x |V| matrix $M$

    in the following two aspects:
     (1) it used a low-rank approximation for the transition matrix:
         $M = E_1 E_2^T$
     (2) it used a beam to estimate the normalizing factor Z(x)
    """

    def __init__(self, num_embedding, low_rank=32, beam_size=64):
        super().__init__()
        self.E1 = nn.Embedding(num_embedding, low_rank)
        self.E2 = nn.Embedding(num_embedding, low_rank)
        self.vocb = num_embedding
        self.rank = low_rank
        self.beam = beam_size

    def extra_repr(self):
        return 'vocab_size={}, low_rank={}, beam_size={}'.format(self.vocb, self.rank, self.beam)

    def forward(self, emissions, targets, masks, beam=None):
        """
        Compute the conditional log-likelihood of a sequence of target tokens given emission scores

        Args:
            emissions (`~torch.Tensor`): Emission score are usually the unnormalized decoder output
                ``(batch_size, seq_len, vocab_size)``. We assume batch-first
            targets (`~torch.LongTensor`): Sequence of target token indices
                ``(batch_size, seq_len)
            masks (`~torch.ByteTensor`): Mask tensor with the same size as targets

        Returns:
            `~torch.Tensor`: approximated log-likelihood
        """
        numerator = self._compute_score(emissions, targets, masks)
        denominator = self._compute_normalizer(emissions, targets, masks, beam)
        return numerator - denominator

    def forward_decoder(self, emissions, masks=None, beam=None):
        """
        Find the most likely output sequence using Viterbi algorithm.

        Args:
            emissions (`~torch.Tensor`): Emission score are usually the unnormalized decoder output
                ``(batch_size, seq_len, vocab_size)``. We assume batch-first
            masks (`~torch.ByteTensor`): Mask tensor with the same size as targets

        Returns:
            `~torch.LongTensor`: decoded sequence from the CRF model
        """
        return self._viterbi_decode(emissions, masks, beam)

    def _compute_score(self, emissions, targets, masks=None):
        batch_size, seq_len = targets.size()
        emission_scores = emissions.gather(2, targets[:, :, None])[:, :, 0]
        transition_scores = (self.E1(targets[:, :-1]) * self.E2(targets[:, 1:])).sum(2)
        scores = emission_scores
        scores[:, 1:] += transition_scores
        if masks is not None:
            scores = scores * masks.type_as(scores)
        return scores.sum(-1)

    def _compute_normalizer(self, emissions, targets=None, masks=None, beam=None):
        beam = beam if beam is not None else self.beam
        batch_size, seq_len = emissions.size()[:2]
        if targets is not None:
            _emissions = emissions.scatter(2, targets[:, :, None], np.float('inf'))
            beam_targets = _emissions.topk(beam, 2)[1]
            beam_emission_scores = emissions.gather(2, beam_targets)
        else:
            beam_emission_scores, beam_targets = emissions.topk(beam, 2)
        beam_transition_score1 = self.E1(beam_targets[:, :-1])
        beam_transition_score2 = self.E2(beam_targets[:, 1:])
        beam_transition_matrix = torch.bmm(beam_transition_score1.view(-1, beam, self.rank), beam_transition_score2.view(-1, beam, self.rank).transpose(1, 2))
        beam_transition_matrix = beam_transition_matrix.view(batch_size, -1, beam, beam)
        score = beam_emission_scores[:, 0]
        for i in range(1, seq_len):
            next_score = score[:, :, None] + beam_transition_matrix[:, i - 1]
            next_score = logsumexp(next_score, dim=1) + beam_emission_scores[:, i]
            if masks is not None:
                score = torch.where(masks[:, i:i + 1], next_score, score)
            else:
                score = next_score
        return logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions, masks=None, beam=None):
        beam = beam if beam is not None else self.beam
        batch_size, seq_len = emissions.size()[:2]
        beam_emission_scores, beam_targets = emissions.topk(beam, 2)
        beam_transition_score1 = self.E1(beam_targets[:, :-1])
        beam_transition_score2 = self.E2(beam_targets[:, 1:])
        beam_transition_matrix = torch.bmm(beam_transition_score1.view(-1, beam, self.rank), beam_transition_score2.view(-1, beam, self.rank).transpose(1, 2))
        beam_transition_matrix = beam_transition_matrix.view(batch_size, -1, beam, beam)
        traj_tokens, traj_scores = [], []
        finalized_tokens, finalized_scores = [], []
        score = beam_emission_scores[:, 0]
        dummy = torch.arange(beam, device=score.device).expand(*score.size()).contiguous()
        for i in range(1, seq_len):
            traj_scores.append(score)
            _score = score[:, :, None] + beam_transition_matrix[:, i - 1]
            _score, _index = _score.max(dim=1)
            _score = _score + beam_emission_scores[:, i]
            if masks is not None:
                score = torch.where(masks[:, i:i + 1], _score, score)
                index = torch.where(masks[:, i:i + 1], _index, dummy)
            else:
                score, index = _score, _index
            traj_tokens.append(index)
        best_score, best_index = score.max(dim=1)
        finalized_tokens.append(best_index[:, None])
        finalized_scores.append(best_score[:, None])
        for idx, scs in zip(reversed(traj_tokens), reversed(traj_scores)):
            previous_index = finalized_tokens[-1]
            finalized_tokens.append(idx.gather(1, previous_index))
            finalized_scores.append(scs.gather(1, previous_index))
        finalized_tokens.reverse()
        finalized_tokens = torch.cat(finalized_tokens, 1)
        finalized_tokens = beam_targets.gather(2, finalized_tokens[:, :, None])[:, :, 0]
        finalized_scores.reverse()
        finalized_scores = torch.cat(finalized_scores, 1)
        finalized_scores[:, 1:] = finalized_scores[:, 1:] - finalized_scores[:, :-1]
        return finalized_scores, finalized_tokens


class LightweightConv1d(nn.Module):
    """Lightweight Convolution assuming the input is BxCxT
    This is just an example that explains LightConv clearer than the TBC version.
    We don't use this module in the model.

    Args:
        input_size: # of channels of the input and output
        kernel_size: convolution channels
        padding: padding
        num_heads: number of heads used. The weight is of shape
            `(num_heads, 1, kernel_size)`
        weight_softmax: normalize the weight with softmax before the convolution

    Shape:
        Input: BxCxT, i.e. (batch_size, input_size, timesteps)
        Output: BxCxT, i.e. (batch_size, input_size, timesteps)

    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias: the learnable bias of the module of shape `(input_size)`
    """

    def __init__(self, input_size, kernel_size=1, padding=0, num_heads=1, weight_softmax=False, bias=False, weight_dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.padding = padding
        self.weight_softmax = weight_softmax
        self.weight = nn.Parameter(torch.Tensor(num_heads, 1, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.bias = None
        self.weight_dropout_module = FairseqDropout(weight_dropout, module_name=self.__class__.__name__)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, input):
        """
        input size: B x C x T
        output size: B x C x T
        """
        B, C, T = input.size()
        H = self.num_heads
        weight = self.weight
        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)
        weight = self.weight_dropout_module(weight)
        input = input.view(-1, H, T)
        output = F.conv1d(input, weight, padding=self.padding, groups=self.num_heads)
        output = output.view(B, C, T)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)
        return output


class PQConv2d(nn.Module):
    """
    Quantized counterpart of nn.Conv2d module. Stores the centroid, the assignments
    and the non-quantized biases. The full weight is re-instantiated at each forward
    pass and autograd automatically computes the gradients with respect to the
    centroids.

    Args:
        - centroids: centroids of size n_centroids x block_size
        - assignments: assignments of the centroids to the subvectors
          of size self.out_channels x n_blocks
        - bias: the non-quantized bias, must be either torch.Tensor or None

    Remarks:
        - We refer the reader to the official documentation of the nn.Conv2d module
          for the other arguments and the behavior of the module.
        - Performance tests on GPU show that this implementation is 10% slower than
          the non-quantized nn.Conv2d module for a standard training loop.
        - During the backward, the gradients are averaged by cluster and not summed.
          This explains the hook registered to the centroids.
    """

    def __init__(self, centroids, assignments, bias, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros'):
        super(PQConv2d, self).__init__()
        self.block_size = centroids.size(1)
        self.n_centroids = centroids.size(0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        if in_channels // groups * np.prod(self.kernel_size) % self.block_size != 0:
            raise ValueError('Wrong PQ sizes')
        if len(assignments) % out_channels != 0:
            raise ValueError('Wrong PQ sizes')
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.centroids = nn.Parameter(centroids, requires_grad=True)
        self.register_buffer('assignments', assignments)
        self.register_buffer('counts', torch.bincount(assignments).type_as(centroids))
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter('bias', None)
        self.centroids.register_hook(lambda x: x / self.counts[:, None])

    @property
    def weight(self):
        return self.centroids[self.assignments].reshape(-1, self.out_channels, self.block_size).permute(1, 0, 2).reshape(self.out_channels, self.in_channels // self.groups, *self.kernel_size)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self):
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        s += ', n_centroids={n_centroids}, block_size={block_size}'
        return s.format(**self.__dict__)


class PQEmbedding(nn.Module):
    """
    Quantized counterpart of nn.Embedding module. Stores the centroids and
    the assignments. The full weight is re-instantiated at each forward
    pass.

    Args:
        - centroids: centroids of size n_centroids x block_size
        - assignments: assignments of the centroids to the subvectors
          of size self.out_features x n_blocks
        - bias: the non-quantized bias

    Remarks:
        - We refer the reader to the official documentation of the nn.Embedding module
          for the other arguments and the behavior of the module
        - Performance tests on GPU show that this implementation is 10% slower than
          the non-quantized nn.Embedding module for a standard training loop.
    """

    def __init__(self, centroids, assignments, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None):
        super(PQEmbedding, self).__init__()
        self.block_size = centroids.size(1)
        self.n_centroids = centroids.size(0)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        if self.embedding_dim % self.block_size != 0:
            raise ValueError('Wrong PQ sizes')
        if len(assignments) % self.num_embeddings != 0:
            raise ValueError('Wrong PQ sizes')
        self.centroids = nn.Parameter(centroids, requires_grad=True)
        self.register_buffer('assignments', assignments)
        self.register_buffer('counts', torch.bincount(assignments).type_as(centroids))

    @property
    def weight(self):
        return self.centroids[self.assignments].reshape(-1, self.num_embeddings, self.block_size).permute(1, 0, 2).flatten(1, 2)

    def forward(self, input):
        return F.embedding(input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

    def extra_repr(self):
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        s += ', n_centroids={n_centroids}, block_size={block_size}'
        return s.format(**self.__dict__)


class PQLinear(nn.Module):
    """
    Quantized counterpart of nn.Linear module. Stores the centroid, the assignments
    and the non-quantized biases. The full weight is re-instantiated at each forward
    pass.

    Args:
        - centroids: centroids of size n_centroids x block_size
        - assignments: assignments of the centroids to the subvectors
          of size self.out_features x n_blocks
        - bias: the non-quantized bias

    Remarks:
        - We refer the reader to the official documentation of the nn.Linear module
          for the other arguments and the behavior of the module
        - Performance tests on GPU show that this implementation is 15% slower than
          the non-quantized nn.Linear module for a standard training loop.
    """

    def __init__(self, centroids, assignments, bias, in_features, out_features):
        super(PQLinear, self).__init__()
        self.block_size = centroids.size(1)
        self.n_centroids = centroids.size(0)
        self.in_features = in_features
        self.out_features = out_features
        if self.in_features % self.block_size != 0:
            raise ValueError('Wrong PQ sizes')
        if len(assignments) % self.out_features != 0:
            raise ValueError('Wrong PQ sizes')
        self.centroids = nn.Parameter(centroids, requires_grad=True)
        self.register_buffer('assignments', assignments)
        self.register_buffer('counts', torch.bincount(assignments).type_as(centroids))
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter('bias', None)

    @property
    def weight(self):
        return self.centroids[self.assignments].reshape(-1, self.out_features, self.block_size).permute(1, 0, 2).flatten(1, 2)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self):
        return f'in_features={self.in_features},                 out_features={self.out_features},                 n_centroids={self.n_centroids},                 block_size={self.block_size},                 bias={self.bias is not None}'


def emulate_int(w, bits, method, scale=None, zero_point=None):
    q = globals()[f'emulate_int{bits}_{method}']
    return q(w, scale=scale, zero_point=zero_point)


class IntConv2d(_ConvNd):
    """
    Quantized counterpart of the nn.Conv2d module that applies QuantNoise during training.

    Args:
        - standard nn.Conv2d parameters
        - p: amount of noise to inject (0 = no quantization, 1 = quantize all the weights)
        - bits: number of bits
        - method: choose among {"tensor", "histogram", "channel"}
        - update_step: recompute scale and zero_point every update_steps iterations

    Remarks:
        - We use the straight-thgourh estimator so that the gradients
          back-propagate nicely in the network, this is implemented with
          the detach() trick
        - Parameters scale and zero_point are recomputed every update_step
          forward pass to reduce the overhead
        - At test time, the weights are fully quantized
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', p=0, bits=8, method='histogram', update_step=1000):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(IntConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, padding_mode)
        self.p = p
        self.bits = bits
        self.method = method
        self.update_step = update_step
        self.counter = 0

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode), weight, self.bias, self.stride, _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, input):
        p = self.p if self.training else 1
        if self.counter % self.update_step == 0:
            self.scale = None
            self.zero_point = None
        self.counter += 1
        weight_quantized, self.scale, self.zero_point = emulate_int(self.weight.detach(), bits=self.bits, method=self.method, scale=self.scale, zero_point=self.zero_point)
        mask = torch.zeros_like(self.weight)
        mask.bernoulli_(1 - p)
        noise = (weight_quantized - self.weight).masked_fill(mask.bool(), 0)
        clamp_low = -self.scale * self.zero_point
        clamp_high = self.scale * (2 ** self.bits - 1 - self.zero_point)
        weight = torch.clamp(self.weight, clamp_low.item(), clamp_high.item()) + noise.detach()
        output = self._conv_forward(input, weight)
        return output

    def extra_repr(self):
        return 'in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}, dilation={}, groups={}, bias={}, quant_noise={}, bits={}, method={}'.format(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias is not None, self.p, self.bits, self.method)


class IntEmbedding(nn.Module):
    """
    Quantized counterpart of the nn.Embedding module that applies QuantNoise during training.

    Args:
        - num_embeddings: number of tokens
        - embedding_dim: embedding dimension
        - p: amount of noise to inject (0 = no quantization, 1 = quantize all the weights)
        - bits: number of bits
        - method: choose among {"tensor", "histogram", "channel"}
        - update_step: recompute scale and zero_point every update_steps iterations

    Remarks:
        - We use the straight-through estimator so that the gradients
          back-propagate nicely in the network, this is implemented with
          the detach() trick
        - Parameters scale and zero_point are recomputed every update_step
          forward pass to reduce the overhead
        - At test time, the weights are fully quantized
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, p=0, update_step=1000, bits=8, method='histogram'):
        super(IntEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], 'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = nn.Parameter(_weight)
        self.sparse = sparse
        self.p = p
        self.bits = bits
        self.method = method
        self.update_step = update_step
        self.counter = 0

    def reset_parameters(self):
        nn.init.normal_(self.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input):
        p = self.p if self.training else 1
        if self.counter % self.update_step == 0:
            self.scale = None
            self.zero_point = None
        self.counter += 1
        weight_quantized, self.scale, self.zero_point = emulate_int(self.weight.detach(), bits=self.bits, method=self.method, scale=self.scale, zero_point=self.zero_point)
        mask = torch.zeros_like(self.weight)
        mask.bernoulli_(1 - p)
        noise = (weight_quantized - self.weight).masked_fill(mask.bool(), 0)
        clamp_low = -self.scale * self.zero_point
        clamp_high = self.scale * (2 ** self.bits - 1 - self.zero_point)
        weight = torch.clamp(self.weight, clamp_low.item(), clamp_high.item()) + noise.detach()
        output = F.embedding(input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        return output

    def extra_repr(self):
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        s += 'quant_noise={p}, bits={bits}, method={method}'
        return s.format(**self.__dict__)


class IntLinear(nn.Module):
    """
    Quantized counterpart of the nn.Linear module that applies QuantNoise during training.

    Args:
        - in_features: input features
        - out_features: output features
        - bias: bias or not
        - p: amount of noise to inject (0 = no quantization, 1 = quantize all the weights)
        - bits: number of bits
        - method: choose among {"tensor", "histogram", "channel"}
        - update_step: recompute scale and zero_point every update_steps iterations

    Remarks:
        - We use the straight-through estimator so that the gradients
          back-propagate nicely in the network, this is implemented with
          the detach() trick.
        - Parameters scale and zero_point are recomputed every update_step
          forward pass to reduce the overhead
        - At test time, the weights are fully quantized
    """

    def __init__(self, in_features, out_features, bias=True, p=0, update_step=3000, bits=8, method='histogram'):
        super(IntLinear, self).__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.chosen_bias = bias
        if self.chosen_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.p = p
        self.bits = bits
        self.method = method
        self.update_step = update_step
        self.counter = 0

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.chosen_bias:
            nn.init.constant_(self.bias, 0.0)
        return

    def forward(self, input):
        p = self.p if self.training else 1
        if self.counter % self.update_step == 0:
            self.scale = None
            self.zero_point = None
        self.counter += 1
        weight_quantized, self.scale, self.zero_point = emulate_int(self.weight.detach(), bits=self.bits, method=self.method, scale=self.scale, zero_point=self.zero_point)
        mask = torch.zeros_like(self.weight)
        mask.bernoulli_(1 - p)
        noise = (weight_quantized - self.weight).masked_fill(mask.bool(), 0)
        clamp_low = -self.scale * self.zero_point
        clamp_high = self.scale * (2 ** self.bits - 1 - self.zero_point)
        weight = torch.clamp(self.weight, clamp_low.item(), clamp_high.item()) + noise.detach()
        output = F.linear(input, weight, self.bias)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, quant_noise={}, bits={}, method={}'.format(self.in_features, self.out_features, self.bias is not None, self.p, self.bits, self.method)


class SparseMultiheadAttention(MultiheadAttention):
    """Sparse Multi-Headed Attention.

    "Generating Long Sequences with Sparse Transformers". Implements
    fixed factorized self attention, where l=stride and c=expressivity.
    A(1) includes all words in the stride window and A(2) takes a summary of c
    words from the end of each stride window.
    If is_bidirectional=False, we do not include any words past the current word,
    as in the paper.
    """

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, self_attention=False, encoder_decoder_attention=False, stride=32, expressivity=8, is_bidirectional=True):
        super().__init__(embed_dim, num_heads, kdim, vdim, dropout, bias, add_bias_kv, add_zero_attn, self_attention, encoder_decoder_attention)
        self.is_bidirectional = is_bidirectional
        self.stride = stride
        self.expressivity = expressivity
        assert self.stride > 0 and self.stride >= self.expressivity

    def compute_checkpoint(self, word_index):
        if word_index % self.stride == 0 and word_index != 0:
            checkpoint_index = word_index - self.expressivity
        else:
            checkpoint_index = math.floor(word_index / self.stride) * self.stride + self.stride - self.expressivity
        return checkpoint_index

    def compute_subset_summaries(self, absolute_max):
        checkpoint_index = self.compute_checkpoint(0)
        subset_two = set()
        while checkpoint_index <= absolute_max - 1:
            summary = set(range(checkpoint_index, min(checkpoint_index + self.expressivity + 1, absolute_max)))
            subset_two = subset_two.union(summary)
            checkpoint_index = self.compute_checkpoint(checkpoint_index + self.stride)
        return subset_two

    def compute_fixed_attention_subset(self, word_index, tgt_len):
        if not self.is_bidirectional:
            absolute_max = word_index + 1
        else:
            absolute_max = tgt_len
        rounded_index = math.floor((word_index + self.stride) / self.stride) * self.stride
        if word_index % self.stride == 0 and word_index != 0:
            subset_one = set(range(word_index - self.stride, min(absolute_max, word_index + 1)))
        else:
            subset_one = set(range(max(0, rounded_index - self.stride), min(absolute_max, rounded_index + 1)))
        subset_two = set()
        if not self.is_bidirectional:
            subset_two = self.compute_subset_summaries(absolute_max)
        return subset_one.union(subset_two)

    def buffered_sparse_mask(self, tensor, tgt_len, src_len):
        assert tgt_len > self.stride
        sparse_mask = torch.empty((tgt_len, src_len)).float().fill_(float('-inf'))
        subset_summaries = set()
        if self.is_bidirectional:
            subset_summaries = self.compute_subset_summaries(tgt_len)
        for i in range(tgt_len):
            fixed_attention_subset = self.compute_fixed_attention_subset(i, tgt_len)
            fixed_attention_subset = fixed_attention_subset.union(subset_summaries)
            included_word_indices = torch.LongTensor(list(fixed_attention_subset))
            sparse_mask[i].index_fill_(0, included_word_indices, 0)
        return sparse_mask.type_as(tensor)

    def apply_sparse_mask(self, attn_weights, tgt_len, src_len, bsz):
        sparse_mask = self.buffered_sparse_mask(attn_weights, tgt_len, src_len)
        sparse_mask = sparse_mask.unsqueeze(0).expand(bsz * self.num_heads, tgt_len, src_len)
        attn_weights += sparse_mask


def infer_conv_output_dim(conv_op, input_dim, sample_inchannel):
    sample_seq_len = 200
    sample_bsz = 10
    x = torch.randn(sample_bsz, sample_inchannel, sample_seq_len, input_dim)
    x = conv_op(x)
    x = x.transpose(1, 2)
    bsz, seq = x.size()[:2]
    per_channel_dim = x.size()[3]
    return x.contiguous().view(bsz, seq, -1).size(-1), per_channel_dim


class VGGBlock(torch.nn.Module):
    """
    VGG motibated cnn module https://arxiv.org/pdf/1409.1556.pdf

    Args:
        in_channels: (int) number of input channels (typically 1)
        out_channels: (int) number of output channels
        conv_kernel_size: convolution channels
        pooling_kernel_size: the size of the pooling window to take a max over
        num_conv_layers: (int) number of convolution layers
        input_dim: (int) input dimension
        conv_stride: the stride of the convolving kernel.
            Can be a single number or a tuple (sH, sW)  Default: 1
        padding: implicit paddings on both sides of the input.
            Can be a single number or a tuple (padH, padW). Default: None
        layer_norm: (bool) if layer norm is going to be applied. Default: False

    Shape:
        Input: BxCxTxfeat, i.e. (batch_size, input_size, timesteps, features)
        Output: BxCxTxfeat, i.e. (batch_size, input_size, timesteps, features)
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size, pooling_kernel_size, num_conv_layers, input_dim, conv_stride=1, padding=None, layer_norm=False):
        assert input_dim is not None, 'Need input_dim for LayerNorm and infer_conv_output_dim'
        super(VGGBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_kernel_size = _pair(conv_kernel_size)
        self.pooling_kernel_size = _pair(pooling_kernel_size)
        self.num_conv_layers = num_conv_layers
        self.padding = tuple(e // 2 for e in self.conv_kernel_size) if padding is None else _pair(padding)
        self.conv_stride = _pair(conv_stride)
        self.layers = nn.ModuleList()
        for layer in range(num_conv_layers):
            conv_op = nn.Conv2d(in_channels if layer == 0 else out_channels, out_channels, self.conv_kernel_size, stride=self.conv_stride, padding=self.padding)
            self.layers.append(conv_op)
            if layer_norm:
                conv_output_dim, per_channel_dim = infer_conv_output_dim(conv_op, input_dim, in_channels if layer == 0 else out_channels)
                self.layers.append(nn.LayerNorm(per_channel_dim))
                input_dim = per_channel_dim
            self.layers.append(nn.ReLU())
        if self.pooling_kernel_size is not None:
            pool_op = nn.MaxPool2d(kernel_size=self.pooling_kernel_size, ceil_mode=True)
            self.layers.append(pool_op)
            self.total_output_dim, self.output_dim = infer_conv_output_dim(pool_op, input_dim, out_channels)

    def forward(self, x):
        for i, _ in enumerate(self.layers):
            x = self.layers[i](x)
        return x


class Search(nn.Module):

    def __init__(self, tgt_dict):
        super().__init__()
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.src_lengths = torch.tensor(-1)
        self.supports_constraints = False
        self.stop_on_max_len = False

    def step(self, step, lprobs, scores, prev_output_tokens=None, original_batch_idxs=None):
        """Take a single search step.

        Args:
            step: the current search step, starting at 0
            lprobs: (bsz x input_beam_size x vocab_size)
                the model's log-probabilities over the vocabulary at the current step
            scores: (bsz x input_beam_size x step)
                the historical model scores of each hypothesis up to this point
            prev_output_tokens: (bsz x step)
                the previously generated oputput tokens
            original_batch_idxs: (bsz)
                the tensor with the batch indices, in the range [0, bsz)
                this is useful in case there has been applied a re-ordering
                and we need to know the orignal indices

        Return: A tuple of (scores, indices, beams) where:
            scores: (bsz x output_beam_size)
                the scores of the chosen elements; output_beam_size can be
                larger than input_beam_size, e.g., we may return
                2*input_beam_size to account for EOS
            indices: (bsz x output_beam_size)
                the indices of the chosen elements
            beams: (bsz x output_beam_size)
                the hypothesis ids of the chosen elements, in the range [0, input_beam_size)
        """
        raise NotImplementedError

    @torch.jit.export
    def set_src_lengths(self, src_lengths):
        self.src_lengths = src_lengths

    @torch.jit.export
    def init_constraints(self, batch_constraints: 'Optional[Tensor]', beam_size: 'int'):
        """Initialize constraint states for constrained decoding (if supported).

        Args:
            batch_constraints: (torch.Tensor, optional)
                the list of constraints, in packed form
            beam_size: (int)
                the beam size
        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        pass

    def prune_sentences(self, batch_idxs: 'Tensor'):
        """
        Removes constraint states for completed sentences (if supported).
        This is called from sequence_generator._generate() when sentences are
        deleted from the batch.

        Args:
            batch_idxs: Indices of *sentences* whose constraint state should be *kept*.
        """
        pass

    def update_constraints(self, active_hypos: 'Tensor'):
        """
        Updates the constraint states by selecting the beam items that are retained.
        This is called at each time step of sequence_generator._generate() when
        the set of 2 * {beam_size} candidate hypotheses are reduced to the beam size.

        Args:
            active_hypos: (batch size, beam size)
              list of integers denoting, for each sentence, which beam candidate items
              should be kept.
        """
        pass


class BeamSearch(Search):

    def __init__(self, tgt_dict):
        super().__init__(tgt_dict)
        self.constraint_states = None

    @torch.jit.export
    def step(self, step: 'int', lprobs, scores: 'Optional[Tensor]', prev_output_tokens: 'Optional[Tensor]'=None, original_batch_idxs: 'Optional[Tensor]'=None):
        bsz, beam_size, vocab_size = lprobs.size()
        if step == 0:
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:
            assert scores is not None
            lprobs = lprobs + scores[:, :, step - 1].unsqueeze(-1)
        top_prediction = torch.topk(lprobs.view(bsz, -1), k=min(beam_size * 2, lprobs.view(bsz, -1).size(1) - 1))
        scores_buf = top_prediction[0]
        indices_buf = top_prediction[1]
        beams_buf = indices_buf // vocab_size
        indices_buf = indices_buf.fmod(vocab_size)
        return scores_buf, indices_buf, beams_buf


class PrefixConstrainedBeamSearch(Search):

    def __init__(self, tgt_dict, prefix_allowed_tokens_fn):
        super().__init__(tgt_dict)
        self.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self.stop_on_max_len = True

    @torch.jit.export
    def apply_mask(self, x, prev_output_tokens, original_batch_idxs):
        beam_size = x.shape[0] // original_batch_idxs.shape[0]
        original_batch_idxs = original_batch_idxs.unsqueeze(-1).repeat((1, beam_size)).flatten().tolist()
        mask = torch.full_like(x, -math.inf)
        for sent_i, (sent, batch_i) in enumerate(zip(prev_output_tokens, original_batch_idxs)):
            mask[sent_i, :, self.prefix_allowed_tokens_fn(batch_i, sent)] = 0
        return mask

    @torch.jit.export
    def step(self, step: 'int', lprobs: 'Tensor', scores: 'Tensor', prev_output_tokens: 'Tensor', original_batch_idxs: 'Tensor'):
        bsz, beam_size, vocab_size = lprobs.size()
        lprobs += self.apply_mask(lprobs.view(bsz * beam_size, 1, vocab_size), prev_output_tokens, original_batch_idxs).view(bsz, beam_size, vocab_size)
        if step == 0:
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:
            assert scores is not None
            lprobs = lprobs + scores[:, :, step - 1].unsqueeze(-1)
        top_prediction = torch.topk(lprobs.view(bsz, -1), k=min(beam_size, lprobs.view(bsz, -1).size(1) - 1))
        scores_buf = top_prediction[0]
        indices_buf = top_prediction[1]
        beams_buf = indices_buf // vocab_size
        indices_buf = indices_buf.fmod(vocab_size)
        return scores_buf, indices_buf, beams_buf


class ConstraintSequence:

    def __init__(self, sequences: 'List[List[int]]'):
        """Represents a set of possibly multitoken constraints by
        concatenating them and internally recording the end points.
        """
        self.sequences = []
        self.endpoints = []
        self.num_tokens = 0
        self.tokens = set()
        for sequence in sequences:
            for token in sequence:
                self.tokens.add(token)
            self.num_tokens += len(sequence)
            self.endpoints += [(False) for x in range(len(sequence) - 1)] + [True]
            self.sequences += sequence

    def __getitem__(self, key: 'int'):
        return self.sequences[key]

    def __len__(self):
        return len(self.sequences)

    def __str__(self):
        return str(self.sequences)


class ConstraintState:

    def __init__(self):
        pass


def unpack_constraints(constraint_tensor: 'torch.Tensor') ->List[torch.Tensor]:
    """
    Transforms *one row* of a packed constraint tensor (e.g., for one
    sentence in the batch) into a list of constraint tensors.
    """
    constraint_list = []
    num_constraints = constraint_tensor[0]
    constraints = constraint_tensor.tolist()
    offset = 1
    for i in range(num_constraints):
        where = constraints.index(0, offset)
        constraint_list.append(constraint_tensor[offset:where])
        offset = where + 1
    return constraint_list


class OrderedConstraintState(ConstraintState):
    """
    Records progress through the set of linear nonbranching constraints with gaps.
    """

    def __init__(self, sequence: 'ConstraintSequence', state: 'int'=-1):
        self.sequence = sequence
        self.state = state

    @staticmethod
    def create(constraint_tensor: 'torch.Tensor'):
        constraint_list = unpack_constraints(constraint_tensor)
        return OrderedConstraintState(ConstraintSequence(constraint_list), -1)

    def __str__(self):
        return f'{self.state}/{self.bank}x{self.num_completed}'

    def __copy__(self):
        return OrderedConstraintState(self.sequence, self.state)

    def copy(self):
        return self.__copy__()

    @property
    def num_completed(self):
        if self.state == -1:
            return 0
        count = len(list(filter(lambda x: x, self.sequence.endpoints[0:self.state + 1])))
        return count

    @property
    def is_root(self):
        return self.state == -1

    @property
    def name(self):
        if self.state == -1:
            return 'ROOT'
        else:
            return str(self.sequence[self.state])

    @property
    def bank(self) ->int:
        return self.state + 1

    @property
    def finished(self):
        return self.state + 1 == len(self.sequence)

    @property
    def token_counts(self):
        return self.sequence.token_counts()

    @property
    def tokens(self):
        return self.sequence.tokens

    @property
    def num_constraint_tokens(self):
        return sum(self.token_counts.values())

    def next_tokens(self) ->Set[int]:
        """Returns the list of tokens that could come next.
        These are (a) all tokens extending the root state and, for
        non-root states, additionally all tokens extending the current
        state."""
        tokens = set()
        if self.state > 0:
            tokens.add(self.sequence[0])
        if not self.finished:
            tokens.add(self.sequence[self.state + 1])
        return tokens

    def advance(self, token: 'int'):
        """Reads in a token and advances the state. Here's how it works.

        We can advance to the next state if:
        - there is a matching child
        - its path isn't blocked

        A path is blocked when all constraints that are descendants of
        that node have already been generated, in the current state.

        If we are not able to advance from the current state, we "fall
        off the graph" and return to the root state. There, we again
        try to advance, checking the same criteria.

        In any case, when falling off the graph, we need to do some
        bookkeeping. We:
        - check whether any constraints were met (all prefixes of
          current state)
        - if one is found, mark it as completed
        - adjust visited nodes accordingly
        """
        token = int(token)
        if self.finished:
            next_state = self.copy()
        elif self.sequence[self.state + 1] == token:
            next_state = OrderedConstraintState(self.sequence, self.state + 1)
        elif self.sequence.endpoints[self.state]:
            next_state = self.copy()
        elif token == self.sequence[0]:
            next_state = OrderedConstraintState(self.sequence, 0)
        else:
            next_state = OrderedConstraintState(self.sequence, -1)
        return next_state


class ConstraintNode:
    """
    Represents a node in a trie managing unordered constraints.
    """

    def __init__(self, token: 'int'=None, parent=None):
        self.token = int(token) if token is not None else None
        self.parent = parent
        self.terminal = 0
        self.children = {}
        self.num_constraints = 0

    @property
    def id(self):
        return self.token

    def __str__(self):
        term = self.terminal != 0
        return f'[{self.token}].{term}#{self.num_constraints}'

    def __getitem__(self, key: 'int'):
        return self.children.get(key, None)

    def next_tokens(self) ->Set[int]:
        """The set of child labels."""
        return set(self.children.keys())

    @staticmethod
    def create(constraints: 'List[List[int]]'):
        root = ConstraintNode()
        for sequence in constraints:
            root.add_sequence(sequence)
        return root

    @staticmethod
    def print_graph(node: "'ConstraintNode'"):
        if len(node.children) == 0:
            return str(node)
        else:
            s = f'({node}'
            for child in node.children.values():
                s += ' ' + ConstraintNode.print_graph(child)
            s += ')'
            return s

    def token_counts(self) ->Counter:
        """Returns a counter of the number of times each token is used
        in a constraint.
        """
        token_counts = Counter()
        kids = list(self.children.values())
        while len(kids) > 0:
            kid = kids.pop()
            token_counts[kid.id] += kid.num_constraints
            kids += list(kid.children.values())
        return token_counts

    def tokens(self) ->Set[int]:
        """Returns the set of tokens in constraints."""
        return set(self.token_counts().keys())

    def add_sequence(self, sequence: 'List[int]'):
        """Adds a constraint, represented as a list of integers, to
        the trie."""
        assert len(sequence) > 0
        token = int(sequence[0])
        if token not in self.children:
            self.children[token] = ConstraintNode(token, parent=self)
        node = self.children[token]
        if len(sequence) == 1:
            node.terminal += 1
            node.num_constraints += 1
            parent = node.parent
            while parent is not None:
                parent.num_constraints += 1
                parent = parent.parent
        else:
            node.add_sequence(sequence[1:])


class UnorderedConstraintState(ConstraintState):
    """
    Records progress through the set of constraints for each item in the beam
    using a trie.
    """

    def __init__(self, node: 'ConstraintNode', copy_from: "'ConstraintState'"=None):
        self.node = node
        if copy_from is None:
            self.root = node
            self.completed = Counter()
            self.generated = Counter()
            self.needed_tokens = self.root.tokens()
        else:
            self.completed = Counter(copy_from.completed)
            self.generated = Counter(copy_from.generated)
            self.root = copy_from.root
        if self.node != self.root:
            self.generated[node] += 1

    @staticmethod
    def create(constraint_tensor: 'torch.Tensor'):
        constraint_list = unpack_constraints(constraint_tensor)
        constraint_trie_root = ConstraintNode.create(constraint_list)
        return UnorderedConstraintState(constraint_trie_root)

    def __str__(self):
        gen_str = ','.join([str(node) for node in self.generated])
        return f'{self.name}/{self.bank}({gen_str})x{self.num_completed}'

    def __copy__(self):
        copied_state = UnorderedConstraintState(self.node, copy_from=self)
        return copied_state

    def copy(self):
        return self.__copy__()

    @property
    def name(self):
        if self.node.id is None:
            return 'ROOT'
        else:
            return str(self.node.id)

    @property
    def is_root(self):
        return self.node == self.root

    @property
    def bank(self):
        return sum(self.generated.values())

    @property
    def num_completed(self):
        """The number of constraints (not constraint tokens) that are completed.
        In addition to the already-completed states, we need to account for the
        current state, which might get marked as completed when another token
        is generated.
        """
        in_final = self.node.terminal and self.completed[self.node] < self.node.terminal
        return sum(self.completed.values()) + in_final

    @property
    def finished(self):
        return self.root.num_constraints - self.num_completed == 0

    @property
    def token_counts(self):
        return self.root.token_counts()

    @property
    def tokens(self):
        return self.root.tokens()

    @property
    def num_constraint_tokens(self):
        return sum(self.token_counts.values())

    def next_tokens(self) ->Set[int]:
        """Returns the list of tokens that could come next.
        These are (a) all tokens extending the root state and, for
        non-root states, additionally all tokens extending the current
        state."""
        if self.node != self.root:
            return self.root.next_tokens().union(self.node.next_tokens())
        else:
            return self.root.next_tokens()

    def advance(self, token: 'int'):
        """Reads in a token and advances the state. Here's how it works.

        We can advance to the next state if:
        - there is a matching child
        - its path isn't blocked

        A path is blocked when all constraints that are descendants of
        that node have already been generated, in the current state.

        If we are not able to advance from the current state, we "fall
        off the graph" and return to the root state. There, we again
        try to advance, checking the same criteria.

        In any case, when falling off the graph, we need to do some
        bookkeeping. We:
        - check whether any constraints were met (all prefixes of
          current state)
        - if one is found, mark it as completed
        - adjust visited nodes accordingly
        """
        token = int(token)
        next_state = None
        child = self.node[token]
        if child is not None and self.generated[child] < child.num_constraints:
            next_state = UnorderedConstraintState(child, copy_from=self)

        def rewind():
            """If we're mid-trie and an "illegal" token is chosen next, we need
            to reset our state to the root state. However, along the way, we need
            to check whether a prefix of the current trie state represents a state
            we could mark as completed.
            """
            node = self.node
            while node != self.root:
                if node.terminal and self.completed[node] < node.terminal:
                    next_state.completed[node] += 1
                    return
                next_state.generated[node] -= 1
                node = node.parent
        if next_state is None and token in self.root.next_tokens():
            child = self.root[token]
            if self.generated[child] < child.num_constraints:
                next_state = UnorderedConstraintState(child, copy_from=self)
            else:
                next_state = UnorderedConstraintState(self.root, copy_from=self)
            rewind()
        elif next_state is None:
            next_state = UnorderedConstraintState(self.root, copy_from=self)
            rewind()
        return next_state


class LexicallyConstrainedBeamSearch(Search):
    """Implements lexically constrained beam search as described in

        Fast Lexically Constrained Decoding with Dynamic Beam
        Allocation for Neural Machine Translation.  Post & Vilar,
        NAACL 2018.  https://www.aclweb.org/anthology/N18-1119/

    and

        Improved Lexically Constrained Decoding for Translation and
        Monolingual Rewriting. Hu et al, NAACL
        2019. https://www.aclweb.org/anthology/N19-1090/

    This is accomplished by maintaining, for each beam hypothesis, a
    ConstraintState object (see constraints.py) that tracks which
    constraints have been generated and using this information to
    shape the beam for each input sentence.
    """

    def __init__(self, tgt_dict, representation):
        super().__init__(tgt_dict)
        self.representation = representation
        self.vocab_size = len(tgt_dict)
        self.num_cands = 0
        self.supports_constraints = True

    @torch.jit.export
    def init_constraints(self, batch_constraints: 'Optional[Tensor]', beam_size: 'int'):
        self.constraint_states = []
        for constraint_tensor in batch_constraints:
            if self.representation == 'ordered':
                constraint_state = OrderedConstraintState.create(constraint_tensor)
            elif self.representation == 'unordered':
                constraint_state = UnorderedConstraintState.create(constraint_tensor)
            self.constraint_states.append([constraint_state for i in range(beam_size)])

    @torch.jit.export
    def prune_sentences(self, batch_idxs: 'Tensor'):
        self.constraint_states = [self.constraint_states[i] for i in batch_idxs.tolist()]

    @torch.jit.export
    def update_constraints(self, active_hypos: 'Tensor'):
        if self.constraint_states:
            batch_size = active_hypos.size(0)
            for sentid in range(batch_size):
                self.constraint_states[sentid] = [self.constraint_states[sentid][i] for i in active_hypos[sentid]]

    @torch.jit.export
    def step(self, step: 'int', lprobs: 'Tensor', scores: 'Optional[Tensor]', prev_output_tokens: 'Optional[Tensor]'=None, original_batch_idxs: 'Optional[Tensor]'=None):
        """
        A constrained step builds a large candidates list from the following:
        - the top 2 * {beam_size} items over the whole beam
        - for each item in the beam
          - the top {each_k} (default 1)
          - all next constraints
        We then compute the constrained state of each beam item, and assign
        stripe codes: 0 to the best in each bank, 1 to the 2nd-best, and so
        on. We then sort by (stripe, score), and truncate the list at
        2 * beam size.

        Args:
            step: the decoder step
            lprobs: (batch size, beam size, target vocab)
                the target-vocab distributions for each item in the beam.
        Retrun: A tuple of (scores, indices, beams, constraints) where:
            scores: (batch, output beam size)
                the scores of the chosen elements
            indices: (batch, output beam size)
                the target vocab indices of the chosen elements
            beams: (batch, output beam size)
                the 0-indexed hypothesis ids of the chosen elements
            constraints: (batch, output beam size)
                the new constraint states
        """
        each_k = 1
        device = lprobs.device
        batch_size, beam_size, vocab_size = lprobs.size()
        self.num_cands = min(beam_size * 2, lprobs.view(batch_size, -1).size(1) - 1)
        constraint_states = self.constraint_states
        if constraint_states and step > 0:
            not_finished_indices = []
            for sentno, sent_constraints in enumerate(constraint_states):
                for beamno, state in enumerate(sent_constraints):
                    index = sentno * beam_size + beamno
                    if not state.finished:
                        not_finished_indices.append(index)
            not_finished_indices = torch.tensor(not_finished_indices)
            if not_finished_indices.numel() > 0:
                lprobs.view(batch_size * beam_size, -1)[not_finished_indices, self.eos] = -math.inf
        if step == 0:
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:
            assert scores is not None
            lprobs = lprobs + scores[:, :, step - 1].unsqueeze(-1)
        top_prediction = torch.topk(lprobs.view(batch_size, -1), self.num_cands)
        scores_buf, indices_buf = top_prediction
        beams_buf = indices_buf // vocab_size
        indices_buf = indices_buf.fmod(vocab_size)
        if not constraint_states:
            return scores_buf, indices_buf, beams_buf
        if step > 0:
            top_scores, top_indices = torch.topk(lprobs.view(batch_size * beam_size, -1), k=each_k, dim=1)
            top_scores = top_scores.view(batch_size, -1)
            top_indices = top_indices.view(batch_size, -1)
            scores_buf = torch.cat((scores_buf, top_scores), dim=1)
            indices_buf = torch.cat((indices_buf, top_indices), dim=1)
            new_beams = torch.arange(0, beam_size, device=device).repeat(batch_size, 1)
            beams_buf = torch.cat((beams_buf, new_beams), dim=1)
        new_scores_buf = torch.zeros((batch_size, 2 * beam_size), device=device)
        new_indices_buf = torch.zeros((batch_size, 2 * beam_size), device=device).long()
        new_beams_buf = torch.zeros((batch_size, 2 * beam_size), device=device).long()
        for sentno, states in enumerate(constraint_states):
            scores, indices, beams, new_states = self.step_sentence(step, sentno, lprobs[sentno], constraint_states[sentno], beams_buf[sentno].clone(), indices_buf[sentno].clone(), scores_buf[sentno].clone())
            new_scores_buf[sentno] = scores
            new_indices_buf[sentno] = indices
            new_beams_buf[sentno] = beams
            self.constraint_states[sentno] = new_states
        return new_scores_buf, new_indices_buf, new_beams_buf

    @torch.jit.export
    def step_sentence(self, step: 'int', sentno: 'int', lprobs: 'Tensor', constraint_states: 'List[List[ConstraintState]]', beams_buf: 'Tensor', indices_buf: 'Tensor', scores_buf: 'Tensor'):
        """Does per-sentence processing. Adds all constraints for each
        hypothesis to the list of candidates; then removes duplicates,
        sorts, and dynamically stripes across the banks. All tensor inputs
        are collapsed to those pertaining to a single input sentence.
        """
        device = lprobs.device
        for beamno, state in enumerate(constraint_states):
            next_tokens = torch.tensor(list(state.next_tokens()), device=device).long()
            if next_tokens.numel() != 0:
                indices_buf = torch.cat((indices_buf, next_tokens))
                next_beams = torch.tensor(beamno, device=device).repeat(next_tokens.size(0)).long()
                beams_buf = torch.cat((beams_buf, next_beams))
                next_values = lprobs[beamno].take(next_tokens.view(-1))
                scores_buf = torch.cat((scores_buf, next_values))
            if step == 0:
                break
        cands_size = indices_buf.size(0)
        constraint_states = [constraint_states[beams_buf[i]].advance(indices_buf[i]) for i in range(cands_size)]
        banks = torch.tensor([state.bank for state in constraint_states], device=device)
        num_constraint_tokens = len(state.tokens)
        MAX_SCORE = -100
        sort_key = (num_constraint_tokens - banks) * MAX_SCORE + scores_buf
        sort_values, sort_indices = sort_key.sort(dim=0, descending=True)
        scores_buf = scores_buf[sort_indices]
        indices_buf = indices_buf[sort_indices]
        beams_buf = beams_buf[sort_indices]
        banks = banks[sort_indices]
        constraint_states = [constraint_states[i] for i in sort_indices]

        def roll(t):
            """Rolls a 1d tensor left by 1.

            [0, 1, 2, 3, 4] becomes [4, 0, 1, 2, 3]
            """
            return torch.cat((t[-1].unsqueeze(0), t[0:-1]), dim=0)
        uniques_mask = beams_buf * (self.vocab_size + 1) + indices_buf
        uniques_mask = roll(uniques_mask) != uniques_mask
        scores_buf = torch.masked_select(scores_buf, uniques_mask)
        indices_buf = torch.masked_select(indices_buf, uniques_mask)
        beams_buf = torch.masked_select(beams_buf, uniques_mask)
        banks = torch.masked_select(banks, uniques_mask)
        i = 1
        for mask in uniques_mask[1:]:
            if not mask:
                constraint_states.pop(i)
            i += mask
        stripe_offsets = [(offset * (len(banks) + 1)) for offset in range(len(banks) + 1)]
        stripes = torch.zeros_like(banks)
        cur_bank_count = -1
        cur_bank = banks[0]
        for i, bank in enumerate(banks):
            if bank != cur_bank:
                cur_bank_count = 0
                cur_bank = bank
            else:
                cur_bank_count += 1
            stripes[i] = num_constraint_tokens - bank + stripe_offsets[cur_bank_count]
        sort_values, sort_indices = stripes.sort(dim=0)
        scores_buf = scores_buf[sort_indices]
        indices_buf = indices_buf[sort_indices]
        beams_buf = beams_buf[sort_indices]
        constraint_states = [constraint_states[i] for i in sort_indices]
        scores_buf = scores_buf[:self.num_cands]
        indices_buf = indices_buf[:self.num_cands]
        beams_buf = beams_buf[:self.num_cands]
        return scores_buf, indices_buf, beams_buf, constraint_states


class LengthConstrainedBeamSearch(Search):

    def __init__(self, tgt_dict, min_len_a, min_len_b, max_len_a, max_len_b):
        super().__init__(tgt_dict)
        self.min_len_a = min_len_a
        self.min_len_b = min_len_b
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.beam = BeamSearch(tgt_dict)
        self.needs_src_lengths = True

    def step(self, step: 'int', lprobs, scores, prev_output_tokens: 'Optional[Tensor]'=None, original_batch_idxs: 'Optional[Tensor]'=None):
        min_lens = self.min_len_a * self.src_lengths + self.min_len_b
        max_lens = self.max_len_a * self.src_lengths + self.max_len_b
        lprobs[step < min_lens, :, self.eos] = -math.inf
        lprobs[step >= max_lens, :, self.eos] = 0
        return self.beam.step(step, lprobs, scores)


class DiverseBeamSearch(Search):
    """Diverse Beam Search.

    See "Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence
    Models" for details.

    We only implement the Hamming Diversity penalty here, which performed best
    in the original paper.
    """

    def __init__(self, tgt_dict, num_groups, diversity_strength):
        super().__init__(tgt_dict)
        self.num_groups = num_groups
        self.diversity_strength = -diversity_strength
        self.beam = BeamSearch(tgt_dict)

    @torch.jit.export
    def step(self, step: 'int', lprobs, scores, prev_output_tokens: 'Optional[Tensor]'=None, original_batch_idxs: 'Optional[Tensor]'=None):
        bsz, beam_size, vocab_size = lprobs.size()
        if beam_size % self.num_groups != 0:
            raise ValueError('DiverseBeamSearch requires --beam to be divisible by the number of groups')
        diversity_buf = torch.zeros(lprobs[:, 0, :].size())
        scores_G, indices_G, beams_G = [], [], []
        for g in range(self.num_groups):
            lprobs_g = lprobs[:, g::self.num_groups, :]
            scores_g = scores[:, g::self.num_groups, :] if step > 0 else None
            if g > 0:
                lprobs_g = torch.add(lprobs_g, other=diversity_buf.unsqueeze(1), alpha=self.diversity_strength)
            else:
                lprobs_g = lprobs_g.contiguous()
            scores_buf, indices_buf, beams_buf = self.beam.step(step, lprobs_g, scores_g)
            beams_buf.mul_(self.num_groups).add_(g)
            scores_G.append(scores_buf.clone())
            indices_G.append(indices_buf.clone())
            beams_G.append(beams_buf.clone())
            diversity_buf.scatter_add_(1, indices_buf, torch.ones(indices_buf.size()))
        scores_buf = torch.stack(scores_G, dim=2).view(bsz, -1)
        indices_buf = torch.stack(indices_G, dim=2).view(bsz, -1)
        beams_buf = torch.stack(beams_G, dim=2).view(bsz, -1)
        return scores_buf, indices_buf, beams_buf


class Sampling(Search):
    sampling_topk: 'int'
    sampling_topp: 'float'

    def __init__(self, tgt_dict, sampling_topk=-1, sampling_topp=-1.0):
        super().__init__(tgt_dict)
        self.sampling_topk = sampling_topk
        self.sampling_topp = sampling_topp

    def _sample_topp(self, lprobs):
        """Sample among the smallest set of elements whose cumulative probability mass exceeds p.

        See `"The Curious Case of Neural Text Degeneration"
        (Holtzman et al., 2019) <https://arxiv.org/abs/1904.09751>`_.

        Args:
            lprobs: (bsz x input_beam_size x vocab_size)
                the model's log-probabilities over the vocabulary at the current step

        Return: A tuple of (trimed_probs, truncated_indices) where:
            trimed_probs: (bsz x input_beam_size x ?)
                the model's probabilities over the elements selected to sample from. The
                width of the third dimension is determined by top-P.
            truncated_indices: (bsz x input_beam_size x ?)
                the indices of the chosen elements.
        """
        probs = lprobs.exp_()
        sorted_probs, sorted_indices = probs.sort(descending=True)
        cumsum_probs = sorted_probs.cumsum(dim=2)
        mask = cumsum_probs.lt(self.sampling_topp)
        cumsum_mask = mask.cumsum(dim=2)
        last_included = cumsum_mask[:, :, -1:]
        last_included.clamp_(0, mask.size()[2] - 1)
        mask = mask.scatter_(2, last_included, 1)
        max_dim = last_included.max()
        truncated_mask = mask[:, :, :max_dim + 1]
        truncated_probs = sorted_probs[:, :, :max_dim + 1]
        truncated_indices = sorted_indices[:, :, :max_dim + 1]
        trim_mask = ~truncated_mask
        trimed_probs = truncated_probs.masked_fill_(trim_mask, 0)
        return trimed_probs, truncated_indices

    @torch.jit.export
    def step(self, step: 'int', lprobs, scores, prev_output_tokens: 'Optional[Tensor]'=None, original_batch_idxs: 'Optional[Tensor]'=None):
        bsz, beam_size, vocab_size = lprobs.size()
        if step == 0:
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        if self.sampling_topp > 0:
            probs, top_indices = self._sample_topp(lprobs)
        elif self.sampling_topk > 0:
            lprobs, top_indices = lprobs.topk(self.sampling_topk)
            probs = lprobs.exp_()
        else:
            probs = lprobs.exp_()
            top_indices = torch.empty(0)
        if step == 0:
            indices_buf = torch.multinomial(probs.view(bsz, -1), beam_size, replacement=True).view(bsz, beam_size)
        else:
            indices_buf = torch.multinomial(probs.view(bsz * beam_size, -1), 1, replacement=True).view(bsz, beam_size)
        if step == 0:
            probs = probs.expand(bsz, beam_size, -1)
        scores_buf = torch.gather(probs, dim=2, index=indices_buf.unsqueeze(-1))
        scores_buf = scores_buf.log_().view(bsz, -1)
        if self.sampling_topk > 0 or self.sampling_topp > 0:
            indices_buf = torch.gather(top_indices.expand(bsz, beam_size, -1), dim=2, index=indices_buf.unsqueeze(-1)).squeeze(2)
        if step == 0:
            beams_buf = indices_buf.new_zeros(bsz, beam_size)
        else:
            beams_buf = torch.arange(0, beam_size).repeat(bsz, 1)
            scores_buf.add_(torch.gather(scores[:, :, step - 1], dim=1, index=beams_buf))
        return scores_buf, indices_buf, beams_buf


class DiverseSiblingsSearch(Search):
    """
    Beam search with diverse siblings.

    See "A Simple, Fast Diverse Decoding Algorithm for Neural Generation" for details.
    https://arxiv.org/abs/1611.08562

    1/ Calculate hypotheses for each beam
    2/ Intra-sibling ordering
    3/ Rewrite scores
    4/ Choose top K hypotheses

    if diversity_rate == 0 is equivalent to BeamSearch
    """

    def __init__(self, tgt_dict, diversity_rate):
        super().__init__(tgt_dict)
        self.diversity_rate = diversity_rate
        self.beam = BeamSearch(tgt_dict)

    def step(self, step: 'int', lprobs, scores, prev_output_tokens: 'Optional[Tensor]'=None, original_batch_idxs: 'Optional[Tensor]'=None):
        bsz, beam_size, vocab_size = lprobs.size()
        k = min(beam_size * 2, lprobs.view(bsz, -1).size(1) - 1)
        s_list: 'List[Tensor]'
        i_list: 'List[Tensor]'
        s_list = [torch.empty(0) for i in range(beam_size)]
        i_list = [torch.LongTensor() for i in range(beam_size)]
        sibling_score = torch.arange(1, k + 1) * self.diversity_rate
        if step == 0:
            return self.beam.step(step, lprobs, scores)
        lprobs.add_(scores[:, :, step - 1].unsqueeze(-1))
        for i in range(beam_size):
            torch.topk(lprobs[:, i, :].view(bsz, -1), k, out=(s_list[i], i_list[i]))
            i_list[i].fmod_(vocab_size)
            s_list[i].sub_(sibling_score)
        indices = torch.stack(i_list, dim=1).view(bsz, -1)
        final_scores = torch.empty(0)
        final_indices = torch.LongTensor()
        final_beams = torch.LongTensor()
        final_scores, final_indices = torch.topk(torch.stack(s_list, dim=1).view(bsz, -1), k)
        final_beams = final_indices // k
        for i in range(bsz):
            final_indices[i] = indices[i][final_indices[i]]
        return final_scores, final_indices, final_beams


class EnsembleModel(nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models_size = len(models)
        self.single_model = models[0]
        self.models = nn.ModuleList(models)
        self.has_incremental: 'bool' = False
        if all(hasattr(m, 'decoder') and isinstance(m.decoder, FairseqIncrementalDecoder) for m in models):
            self.has_incremental = True

    def forward(self):
        pass

    def has_encoder(self):
        return hasattr(self.single_model, 'encoder')

    def has_incremental_states(self):
        return self.has_incremental

    def max_decoder_positions(self):
        return min([m.max_decoder_positions() for m in self.models])

    @torch.jit.export
    def forward_encoder(self, net_input: 'Dict[str, Tensor]'):
        if not self.has_encoder():
            return None
        return [model.encoder.forward_torchscript(net_input) for model in self.models]

    @torch.jit.export
    def forward_decoder(self, tokens, encoder_outs: 'List[EncoderOut]', incremental_states: 'List[Dict[str, Dict[str, Optional[Tensor]]]]', temperature: 'float'=1.0, knn_parameter=None, save_knn_informations=None, sample=None):
        log_probs = []
        avg_attn: 'Optional[Tensor]' = None
        encoder_out: 'Optional[EncoderOut]' = None
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            if self.has_incremental_states():
                decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out, incremental_state=incremental_states[i], knn_parameter=knn_parameter, save_knn_informations=save_knn_informations)
            else:
                decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out, knn_parameter=knn_parameter, save_knn_informations=save_knn_informations)
            attn: 'Optional[Tensor]' = None
            knn_probs: 'Optional[Tensor]' = None
            decoder_len = len(decoder_out)
            if decoder_len > 1 and decoder_out[1] is not None:
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]['attn']
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                if attn is not None:
                    attn = attn[:, -1, :]
            decoder_out_tuple = decoder_out[0][:, -1:, :].div_(temperature), None if decoder_len <= 1 else decoder_out[1]
            probs, extra = model.get_normalized_probs(decoder_out_tuple, log_probs=True, sample=sample)
            probs = probs[:, -1, :]
            if self.models_size == 1:
                return probs, attn, extra
            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(self.models_size)
        extra = {}
        if avg_attn is not None:
            avg_attn.div_(self.models_size)
        return avg_probs, avg_attn, extra

    @torch.jit.export
    def reorder_encoder_out(self, encoder_outs: 'Optional[List[EncoderOut]]', new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_outs: 'List[EncoderOut]' = []
        if not self.has_encoder():
            return new_outs
        for i, model in enumerate(self.models):
            assert encoder_outs is not None
            new_outs.append(model.encoder.reorder_encoder_out(encoder_outs[i], new_order))
        return new_outs

    @torch.jit.export
    def reorder_incremental_state(self, incremental_states: 'List[Dict[str, Dict[str, Optional[Tensor]]]]', new_order):
        if not self.has_incremental_states():
            return
        for i, model in enumerate(self.models):
            model.decoder.reorder_incremental_state_scripting(incremental_states[i], new_order)


class SequenceGenerator(nn.Module):

    def __init__(self, models, tgt_dict, beam_size=1, max_len_a=0, max_len_b=200, min_len=1, normalize_scores=True, len_penalty=1.0, unk_penalty=0.0, temperature=1.0, match_source_len=False, no_repeat_ngram_size=0, search_strategy=None, eos=None, symbols_to_strip_from_output=None, lm_model=None, lm_weight=1.0):
        """Generates translations of a given source sentence.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        super().__init__()
        if isinstance(models, EnsembleModel):
            self.model = models
        else:
            self.model = EnsembleModel(models)
        self.tgt_dict = tgt_dict
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.symbols_to_strip_from_output = symbols_to_strip_from_output.union({self.eos}) if symbols_to_strip_from_output is not None else {self.eos}
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.no_repeat_ngram_size = no_repeat_ngram_size
        assert temperature > 0, '--temperature must be greater than 0'
        self.search = search.BeamSearch(tgt_dict) if search_strategy is None else search_strategy
        self.should_set_src_lengths = hasattr(self.search, 'needs_src_lengths') and self.search.needs_src_lengths
        self.model.eval()
        self.lm_model = lm_model
        self.lm_weight = lm_weight
        if self.lm_model is not None:
            self.lm_model.eval()

    def cuda(self):
        self.model
        return self

    @torch.no_grad()
    def forward(self, sample: 'Dict[str, Dict[str, Tensor]]', prefix_tokens: 'Optional[Tensor]'=None, bos_token: 'Optional[int]'=None):
        """Generate a batch of translations.

        Args:
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, prefix_tokens, bos_token=bos_token)

    def generate_batched_itr(self, data_itr, beam_size=None, cuda=False, timer=None):
        """Iterate over a batched dataset and yield individual translations.
        Args:
            cuda (bool, optional): use GPU for generation
            timer (StopwatchMeter, optional): time generations
        """
        for sample in data_itr:
            s = utils.move_to_cuda(sample) if cuda else sample
            if 'net_input' not in s:
                continue
            input = s['net_input']
            encoder_input = {k: v for k, v in input.items() if k != 'prev_output_tokens'}
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(encoder_input)
            if timer is not None:
                timer.stop(sum(len(h[0]['tokens']) for h in hypos))
            for i, id in enumerate(s['id'].data):
                src = utils.strip_pad(input['src_tokens'].data[i, :], self.pad)
                ref = utils.strip_pad(s['target'].data[i, :], self.pad) if s['target'] is not None else None
                yield id, src, ref, hypos[i]

    @torch.no_grad()
    def generate(self, models, sample: 'Dict[str, Dict[str, Tensor]]', **kwargs):
        """Generate translations. Match the api of other fairseq generators.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            constraints (torch.LongTensor, optional): force decoder to include
                the list of constraints
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, **kwargs)

    def _generate(self, sample: 'Dict[str, Dict[str, Tensor]]', prefix_tokens: 'Optional[Tensor]'=None, constraints: 'Optional[Tensor]'=None, bos_token: 'Optional[int]'=None):
        incremental_states = torch.jit.annotate(List[Dict[str, Dict[str, Optional[Tensor]]]], [torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}) for i in range(self.model.models_size)])
        net_input = sample['net_input']
        if 'src_tokens' in net_input:
            src_tokens = net_input['src_tokens']
            src_lengths = (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        elif 'source' in net_input:
            src_tokens = net_input['source']
            src_lengths = net_input['padding_mask'].size(-1) - net_input['padding_mask'].sum(-1) if net_input['padding_mask'] is not None else torch.tensor(src_tokens.size(-1))
        else:
            raise Exception('expected src_tokens or source in net input')
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size
        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError("Target-side constraints were provided, but search method doesn't support them")
        self.search.init_constraints(constraints, beam_size)
        max_len: 'int' = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(int(self.max_len_a * src_len + self.max_len_b), self.model.max_decoder_positions() - 1)
        assert self.min_len <= max_len, 'min_len cannot be larger than max_len, please adjust these!'
        encoder_outs = self.model.forward_encoder(net_input)
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        assert encoder_outs is not None
        scores = torch.zeros(bsz * beam_size, max_len + 1).float()
        tokens = torch.zeros(bsz * beam_size, max_len + 2).long().fill_(self.pad)
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn: 'Optional[Tensor]' = None
        cands_to_ignore = torch.zeros(bsz, beam_size).eq(-1)
        finalized = torch.jit.annotate(List[List[Dict[str, Tensor]]], [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)])
        finished = [(False) for i in range(bsz)]
        num_remaining_sent = bsz
        cand_size = 2 * beam_size
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)
        reorder_state: 'Optional[Tensor]' = None
        batch_idxs: 'Optional[Tensor]' = None
        original_batch_idxs: 'Optional[Tensor]' = None
        if 'id' in sample and isinstance(sample['id'], Tensor):
            original_batch_idxs = sample['id']
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens)
        for step in range(max_len + 1):
            if reorder_state is not None:
                if batch_idxs is not None:
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(encoder_outs, reorder_state)
            lprobs, avg_attn_scores = self.model.forward_decoder(tokens[:, :step + 1], encoder_outs, incremental_states, self.temperature)
            if self.lm_model is not None:
                lm_out = self.lm_model(tokens[:, :step + 1])
                probs = self.lm_model.get_normalized_probs(lm_out, log_probs=True, sample=None)
                probs = probs[:, -1, :] * self.lm_weight
                lprobs += probs
            lprobs[lprobs != lprobs] = torch.tensor(-math.inf)
            lprobs[:, self.pad] = -math.inf
            lprobs[:, self.unk] -= self.unk_penalty
            if step >= max_len:
                lprobs[:, :self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf
            if prefix_tokens is not None and step < prefix_tokens.size(1) and step < max_len:
                lprobs, tokens, scores = self._prefix_tokens(step, lprobs, scores, tokens, prefix_tokens, beam_size)
            elif step < self.min_len:
                lprobs[:, self.eos] = -math.inf
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(bsz * beam_size, avg_attn_scores.size(1), max_len + 2)
                attn[:, :, step + 1].copy_(avg_attn_scores)
            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0)
            eos_scores = torch.empty(0)
            if self.should_set_src_lengths:
                self.search.set_src_lengths(src_lengths)
            if self.no_repeat_ngram_size > 0:
                lprobs = self._no_repeat_ngram(tokens, lprobs, bsz, beam_size, step)
            cand_scores, cand_indices, cand_beams = self.search.step(step, lprobs.view(bsz, -1, self.vocab_size), scores.view(bsz, beam_size, -1)[:, :, :step], tokens[:, :step + 1], original_batch_idxs)
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0)
            eos_bbsz_idx = torch.masked_select(cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size])
            finalized_sents: 'List[int]' = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size])
                finalized_sents = self.finalize_hypos(step, eos_bbsz_idx, eos_scores, tokens, scores, finalized, finished, beam_size, attn, src_lengths, max_len)
                num_remaining_sent -= len(finalized_sents)
            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)
                batch_mask = torch.ones(bsz, dtype=torch.bool, device=cand_indices.device)
                batch_mask[finalized_sents] = False
                batch_idxs = torch.arange(bsz, device=cand_indices.device).masked_select(batch_mask)
                self.search.prune_sentences(batch_idxs)
                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]
                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, attn.size(1), -1)
                bsz = new_bsz
            else:
                batch_idxs = None
            eos_mask[:, :beam_size] = ~(~cands_to_ignore & ~eos_mask[:, :beam_size])
            active_mask = torch.add(eos_mask.type_as(cand_offsets) * cand_size, cand_offsets[:eos_mask.size(1)])
            new_cands_to_ignore, active_hypos = torch.topk(active_mask, k=beam_size, dim=1, largest=False)
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            assert (~cands_to_ignore).any(dim=1).all()
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)
            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)
            tokens[:, :step + 1] = torch.index_select(tokens[:, :step + 1], dim=0, index=active_bbsz_idx)
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(cand_indices, dim=1, index=active_hypos)
            if step > 0:
                scores[:, :step] = torch.index_select(scores[:, :step], dim=0, index=active_bbsz_idx)
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(cand_scores, dim=1, index=active_hypos)
            self.search.update_constraints(active_hypos)
            if attn is not None:
                attn[:, :, :step + 2] = torch.index_select(attn[:, :, :step + 2], dim=0, index=active_bbsz_idx)
            reorder_state = active_bbsz_idx
        for sent in range(len(finalized)):
            scores = torch.tensor([float(elem['score'].item()) for elem in finalized[sent]])
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(List[Dict[str, Tensor]], finalized[sent])
        return finalized

    def _prefix_tokens(self, step: 'int', lprobs, scores, tokens, prefix_tokens, beam_size: 'int'):
        """Handle prefix tokens"""
        prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
        prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
        prefix_mask = prefix_toks.ne(self.pad)
        lprobs[prefix_mask] = torch.tensor(-math.inf)
        lprobs[prefix_mask] = lprobs[prefix_mask].scatter(-1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask])
        eos_mask = prefix_toks.eq(self.eos)
        if eos_mask.any():
            first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[:, 0, 1:step + 1]
            eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
            target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
            assert (first_beam == target_prefix).all()
            tokens = self.replicate_first_beam(tokens, eos_mask_batch_dim, beam_size)
            scores = self.replicate_first_beam(scores, eos_mask_batch_dim, beam_size)
            lprobs = self.replicate_first_beam(lprobs, eos_mask_batch_dim, beam_size)
        return lprobs, tokens, scores

    def replicate_first_beam(self, tensor, mask, beam_size: 'int'):
        tensor = tensor.view(-1, beam_size, tensor.size(-1))
        tensor[mask] = tensor[mask][:, :1, :]
        return tensor.view(-1, tensor.size(-1))

    def finalize_hypos(self, step: 'int', bbsz_idx, eos_scores, tokens, scores, finalized: 'List[List[Dict[str, Tensor]]]', finished: 'List[bool]', beam_size: 'int', attn: 'Optional[Tensor]', src_lengths, max_len: 'int'):
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        A sentence is finalized when {beam_size} finished items have been collected for it.

        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        Args:
            bbsz_idx (Tensor):
        """
        assert bbsz_idx.numel() == eos_scores.numel()
        tokens_clone = tokens.index_select(0, bbsz_idx)[:, 1:step + 2]
        tokens_clone[:, step] = self.eos
        attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step + 2] if attn is not None else None
        pos_scores = scores.index_select(0, bbsz_idx)[:, :step + 1]
        pos_scores[:, step] = eos_scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]
        if self.normalize_scores:
            eos_scores /= (step + 1) ** self.len_penalty
        cum_unfin: 'List[int]' = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)
        sents_seen: 'Dict[str, Optional[Tensor]]' = {}
        for i in range(bbsz_idx.size()[0]):
            idx = bbsz_idx[i]
            score = eos_scores[i]
            unfin_idx = idx // beam_size
            sent = unfin_idx + cum_unfin[unfin_idx]
            seen = str(sent.item()) + '_' + str(unfin_idx.item())
            if seen not in sents_seen:
                sents_seen[seen] = None
            if self.match_source_len and step > src_lengths[unfin_idx]:
                score = torch.tensor(-math.inf)
            if len(finalized[sent]) < beam_size:
                if attn_clone is not None:
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)
                finalized[sent].append({'tokens': tokens_clone[i], 'score': score, 'attention': hypo_attn, 'alignment': torch.empty(0), 'positional_scores': pos_scores[i]})
        newly_finished: 'List[int]' = []
        for seen in sents_seen.keys():
            sent: 'int' = int(float(seen.split('_')[0]))
            unfin_idx: 'int' = int(float(seen.split('_')[1]))
            if not finished[sent] and self.is_finished(step, unfin_idx, max_len, len(finalized[sent]), beam_size):
                finished[sent] = True
                newly_finished.append(unfin_idx)
        return newly_finished

    def is_finished(self, step: 'int', unfin_idx: 'int', max_len: 'int', finalized_sent_len: 'int', beam_size: 'int'):
        """
        Check whether decoding for a sentence is finished, which
        occurs when the list of finalized sentences has reached the
        beam size, or when we reach the maximum length.
        """
        assert finalized_sent_len <= beam_size
        if finalized_sent_len == beam_size or step == max_len:
            return True
        return False

    def calculate_banned_tokens(self, tokens, step: 'int', gen_ngrams: 'List[Dict[str, List[int]]]', no_repeat_ngram_size: 'int', bbsz_idx: 'int'):
        tokens_list: 'List[int]' = tokens[bbsz_idx, step + 2 - no_repeat_ngram_size:step + 1].tolist()
        ngram_index = ','.join([str(x) for x in tokens_list])
        return gen_ngrams[bbsz_idx].get(ngram_index, torch.jit.annotate(List[int], []))

    def transpose_list(self, l: 'List[List[int]]'):
        min_len = min([len(x) for x in l])
        l2 = [[row[i] for row in l] for i in range(min_len)]
        return l2

    def _no_repeat_ngram(self, tokens, lprobs, bsz: 'int', beam_size: 'int', step: 'int'):
        gen_ngrams: 'List[Dict[str, List[int]]]' = [torch.jit.annotate(Dict[str, List[int]], {}) for bbsz_idx in range(bsz * beam_size)]
        cpu_tokens = tokens.cpu()
        for bbsz_idx in range(bsz * beam_size):
            gen_tokens: 'List[int]' = cpu_tokens[bbsz_idx].tolist()
            for ngram in self.transpose_list([gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]):
                key = ','.join([str(x) for x in ngram[:-1]])
                gen_ngrams[bbsz_idx][key] = gen_ngrams[bbsz_idx].get(key, torch.jit.annotate(List[int], [])) + [ngram[-1]]
        if step + 2 - self.no_repeat_ngram_size >= 0:
            banned_tokens = [self.calculate_banned_tokens(tokens, step, gen_ngrams, self.no_repeat_ngram_size, bbsz_idx) for bbsz_idx in range(bsz * beam_size)]
        else:
            banned_tokens = [torch.jit.annotate(List[int], []) for bbsz_idx in range(bsz * beam_size)]
        for bbsz_idx in range(bsz * beam_size):
            lprobs[bbsz_idx][torch.tensor(banned_tokens[bbsz_idx]).long()] = torch.tensor(-math.inf)
        return lprobs


class EnsembleModelWithAlignment(EnsembleModel):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__(models)

    def forward_align(self, src_tokens, src_lengths, prev_output_tokens):
        avg_attn = None
        for model in self.models:
            decoder_out = model(src_tokens, src_lengths, prev_output_tokens)
            attn = decoder_out[1]['attn'][0]
            if avg_attn is None:
                avg_attn = attn
            else:
                avg_attn.add_(attn)
        if len(self.models) > 1:
            avg_attn.div_(len(self.models))
        return avg_attn


class KNNSequenceGenerator(nn.Module):

    def __init__(self, models, tgt_dict, beam_size=1, max_len_a=0, max_len_b=200, min_len=1, normalize_scores=True, len_penalty=1.0, unk_penalty=0.0, temperature=1.0, match_source_len=False, no_repeat_ngram_size=0, search_strategy=None, eos=None, symbols_to_strip_from_output=None, lm_model=None, lm_weight=1.0):
        """Generates translations of a given source sentence.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        super().__init__()
        if isinstance(models, EnsembleModel):
            self.model = models
        else:
            self.model = EnsembleModel(models)
        self.tgt_dict = tgt_dict
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.symbols_to_strip_from_output = symbols_to_strip_from_output.union({self.eos}) if symbols_to_strip_from_output is not None else {self.eos}
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.no_repeat_ngram_size = no_repeat_ngram_size
        assert temperature > 0, '--temperature must be greater than 0'
        self.search = search.BeamSearch(tgt_dict) if search_strategy is None else search_strategy
        self.should_set_src_lengths = hasattr(self.search, 'needs_src_lengths') and self.search.needs_src_lengths
        self.model.eval()
        self.lm_model = lm_model
        self.lm_weight = lm_weight
        if self.lm_model is not None:
            self.lm_model.eval()

    def cuda(self):
        self.model
        return self

    @torch.no_grad()
    def forward(self, sample: 'Dict[str, Dict[str, Tensor]]', prefix_tokens: 'Optional[Tensor]'=None, bos_token: 'Optional[int]'=None):
        """Generate a batch of translations.

        Args:
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, prefix_tokens, bos_token=bos_token)

    def generate_batched_itr(self, data_itr, beam_size=None, cuda=False, timer=None):
        """Iterate over a batched dataset and yield individual translations.
        Args:
            cuda (bool, optional): use GPU for generation
            timer (StopwatchMeter, optional): time generations
        """
        for sample in data_itr:
            s = utils.move_to_cuda(sample) if cuda else sample
            if 'net_input' not in s:
                continue
            input = s['net_input']
            encoder_input = {k: v for k, v in input.items() if k != 'prev_output_tokens'}
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(encoder_input)
            if timer is not None:
                timer.stop(sum(len(h[0]['tokens']) for h in hypos))
            for i, id in enumerate(s['id'].data):
                src = utils.strip_pad(input['src_tokens'].data[i, :], self.pad)
                ref = utils.strip_pad(s['target'].data[i, :], self.pad) if s['target'] is not None else None
                yield id, src, ref, hypos[i]

    @torch.no_grad()
    def generate(self, models, sample: 'Dict[str, Dict[str, Tensor]]', **kwargs):
        """Generate translations. Match the api of other fairseq generators.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            constraints (torch.LongTensor, optional): force decoder to include
                the list of constraints
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, **kwargs)

    def _generate(self, sample: 'Dict[str, Dict[str, Tensor]]', prefix_tokens: 'Optional[Tensor]'=None, constraints: 'Optional[Tensor]'=None, bos_token: 'Optional[int]'=None):
        incremental_states = torch.jit.annotate(List[Dict[str, Dict[str, Optional[Tensor]]]], [torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}) for i in range(self.model.models_size)])
        net_input = sample['net_input']
        knn_parameter = sample['knn_parameter']
        save_knn_informations = sample['save_knn_informations']
        if 'src_tokens' in net_input:
            src_tokens = net_input['src_tokens']
            src_lengths = (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        elif 'source' in net_input:
            src_tokens = net_input['source']
            src_lengths = net_input['padding_mask'].size(-1) - net_input['padding_mask'].sum(-1) if net_input['padding_mask'] is not None else torch.tensor(src_tokens.size(-1))
        else:
            raise Exception('expected src_tokens or source in net input')
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size
        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError("Target-side constraints were provided, but search method doesn't support them")
        self.search.init_constraints(constraints, beam_size)
        max_len: 'int' = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(int(self.max_len_a * src_len + self.max_len_b), self.model.max_decoder_positions() - 1)
        assert self.min_len <= max_len, 'min_len cannot be larger than max_len, please adjust these!'
        encoder_outs = self.model.forward_encoder(net_input)
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        assert encoder_outs is not None
        scores = torch.zeros(bsz * beam_size, max_len + 1).float()
        tokens = torch.zeros(bsz * beam_size, max_len + 2).long().fill_(self.pad)
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn: 'Optional[Tensor]' = None
        knn_probs_record = None
        neural_probs_record = None
        combined_probs_record = None
        query_point_record = None
        knn_neighbors_values_record = None
        knn_neighbors_keys_record = None
        knn_l2_distance_record = None
        knn_sentence_ids_record = None
        knn_token_positions_record = None
        cands_to_ignore = torch.zeros(bsz, beam_size).eq(-1)
        finalized = torch.jit.annotate(List[List[Dict[str, Tensor]]], [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)])
        finished = [(False) for i in range(bsz)]
        num_remaining_sent = bsz
        cand_size = 2 * beam_size
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)
        reorder_state: 'Optional[Tensor]' = None
        batch_idxs: 'Optional[Tensor]' = None
        original_batch_idxs: 'Optional[Tensor]' = None
        if 'id' in sample and isinstance(sample['id'], Tensor):
            original_batch_idxs = sample['id']
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens)
        for step in range(max_len + 1):
            if reorder_state is not None:
                if batch_idxs is not None:
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(encoder_outs, reorder_state)
            lprobs, avg_attn_scores, extra = self.model.forward_decoder(tokens[:, :step + 1], encoder_outs, incremental_states, self.temperature, knn_parameter, save_knn_informations, sample)
            neural_probs = extra.get('neural_probs')
            combined_probs = extra.get('combined_probs')
            query_point = extra.get('query_point')
            knn_neighbors_values = extra.get('knn_neighbors_values')
            knn_neighbors_keys = extra.get('knn_neighbors_keys')
            knn_l2_distance = extra.get('knn_l2_distance')
            knn_sentence_ids = extra.get('knn_sentence_ids')
            knn_token_positions = extra.get('knn_token_positions')
            if self.lm_model is not None:
                lm_out = self.lm_model(tokens[:, :step + 1])
                probs = self.lm_model.get_normalized_probs(lm_out, log_probs=True, sample=None)
                probs = probs[:, -1, :] * self.lm_weight
                lprobs += probs
            lprobs[lprobs != lprobs] = torch.tensor(-math.inf)
            lprobs[:, self.pad] = -math.inf
            lprobs[:, self.unk] -= self.unk_penalty
            if step >= max_len:
                lprobs[:, :self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf
            if prefix_tokens is not None and step < prefix_tokens.size(1) and step < max_len:
                lprobs, tokens, scores = self._prefix_tokens(step, lprobs, scores, tokens, prefix_tokens, beam_size)
            elif step < self.min_len:
                lprobs[:, self.eos] = -math.inf
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(bsz * beam_size, avg_attn_scores.size(1), max_len + 2)
                attn[:, :, step + 1].copy_(avg_attn_scores)
            if neural_probs is not None:
                if neural_probs_record is None:
                    neural_probs_record = torch.empty(bsz * beam_size, max_len + 2, len(self.tgt_dict))
                neural_probs_record[:, step + 1, :] = neural_probs.squeeze(1)
            if combined_probs is not None:
                if combined_probs_record is None:
                    combined_probs_record = torch.empty(bsz * beam_size, max_len + 2, len(self.tgt_dict))
                combined_probs_record[:, step + 1, :] = combined_probs.squeeze(1)
            if query_point is not None:
                if query_point_record is None:
                    query_point_record = torch.empty(bsz * beam_size, max_len + 2, query_point.shape[-1])
                query_point_record[:, step + 1, :] = query_point.squeeze(1)
            if knn_neighbors_values is not None:
                if knn_neighbors_values_record is None:
                    knn_neighbors_values_record = torch.empty(bsz * beam_size, max_len + 2, int(sample['knn_parameter']['k']), dtype=torch.int32)
                knn_neighbors_values_record[:, step + 1, :] = knn_neighbors_values.squeeze(1)
            if knn_neighbors_keys is not None:
                if knn_neighbors_keys_record is None:
                    knn_neighbors_keys_record = torch.empty(bsz * beam_size, max_len + 2, sample['knn_parameter']['k'], knn_neighbors_keys.shape[-1])
                knn_neighbors_keys_record[:, step + 1, :] = knn_neighbors_keys.squeeze(1)
            if knn_l2_distance is not None:
                if knn_l2_distance_record is None:
                    knn_l2_distance_record = torch.empty(bsz * beam_size, max_len + 2, int(sample['knn_parameter']['k']))
                knn_l2_distance_record[:, step + 1, :] = knn_l2_distance.squeeze(1)
            if knn_sentence_ids is not None:
                if knn_sentence_ids_record is None:
                    knn_sentence_ids_record = torch.empty(bsz * beam_size, max_len + 2, int(sample['knn_parameter']['k']), dtype=torch.int32)
                knn_sentence_ids_record[:, step + 1, :] = knn_sentence_ids.squeeze(1)
            if knn_token_positions is not None:
                if knn_token_positions_record is None:
                    knn_token_positions_record = torch.empty(bsz * beam_size, max_len + 2, int(sample['knn_parameter']['k']), dtype=torch.int32)
                knn_token_positions_record[:, step + 1, :] = knn_token_positions.squeeze(1)
            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0)
            eos_scores = torch.empty(0)
            if self.should_set_src_lengths:
                self.search.set_src_lengths(src_lengths)
            if self.no_repeat_ngram_size > 0:
                lprobs = self._no_repeat_ngram(tokens, lprobs, bsz, beam_size, step)
            cand_scores, cand_indices, cand_beams = self.search.step(step, lprobs.view(bsz, -1, self.vocab_size), scores.view(bsz, beam_size, -1)[:, :, :step], tokens[:, :step + 1], original_batch_idxs)
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0)
            eos_bbsz_idx = torch.masked_select(cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size])
            finalized_sents: 'List[int]' = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size])
                finalized_sents = self.finalize_hypos(step, eos_bbsz_idx, eos_scores, tokens, scores, finalized, finished, beam_size, attn, src_lengths, max_len, neural_probs_record, combined_probs_record, query_point_record, knn_neighbors_keys_record, knn_neighbors_values_record, knn_l2_distance_record, knn_sentence_ids_record, knn_token_positions_record)
                num_remaining_sent -= len(finalized_sents)
            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)
                batch_mask = torch.ones(bsz, dtype=torch.bool, device=cand_indices.device)
                batch_mask[finalized_sents] = False
                batch_idxs = torch.arange(bsz, device=cand_indices.device).masked_select(batch_mask)
                self.search.prune_sentences(batch_idxs)
                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]
                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, attn.size(1), -1)
                if neural_probs_record is not None:
                    neural_probs_record = neural_probs_record.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, neural_probs_record.size(1), -1)
                if combined_probs_record is not None:
                    combined_probs_record = combined_probs_record.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, combined_probs_record.size(1), -1)
                if query_point_record is not None:
                    query_point_record = query_point_record.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, query_point_record.size(1), -1)
                if knn_neighbors_keys_record is not None:
                    knn_neighbors_keys_record = knn_neighbors_keys_record.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, knn_neighbors_keys_record.size(1), -1)
                if knn_neighbors_values_record is not None:
                    knn_neighbors_values_record = knn_neighbors_values_record.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, knn_neighbors_values_record.size(1), -1)
                if knn_l2_distance_record is not None:
                    knn_l2_distance_record = knn_l2_distance_record.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, knn_l2_distance_record.size(1), -1)
                if knn_sentence_ids_record is not None:
                    knn_sentence_ids_record = knn_sentence_ids_record.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, knn_sentence_ids_record.size(1), -1)
                if knn_token_positions_record is not None:
                    knn_token_positions_record = knn_token_positions_record.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, knn_token_positions_record.size(1), -1)
                bsz = new_bsz
            else:
                batch_idxs = None
            eos_mask[:, :beam_size] = ~(~cands_to_ignore & ~eos_mask[:, :beam_size])
            active_mask = torch.add(eos_mask.type_as(cand_offsets) * cand_size, cand_offsets[:eos_mask.size(1)])
            new_cands_to_ignore, active_hypos = torch.topk(active_mask, k=beam_size, dim=1, largest=False)
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            assert (~cands_to_ignore).any(dim=1).all()
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)
            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)
            tokens[:, :step + 1] = torch.index_select(tokens[:, :step + 1], dim=0, index=active_bbsz_idx)
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(cand_indices, dim=1, index=active_hypos)
            if step > 0:
                scores[:, :step] = torch.index_select(scores[:, :step], dim=0, index=active_bbsz_idx)
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(cand_scores, dim=1, index=active_hypos)
            self.search.update_constraints(active_hypos)
            if attn is not None:
                attn[:, :, :step + 2] = torch.index_select(attn[:, :, :step + 2], dim=0, index=active_bbsz_idx)
            if neural_probs_record is not None:
                neural_probs_record[:, :step + 2, :] = torch.index_select(neural_probs_record[:, :step + 2, :], dim=0, index=active_bbsz_idx)
            if combined_probs_record is not None:
                combined_probs_record[:, :step + 2, :] = torch.index_select(combined_probs_record[:, :step + 2, :], dim=0, index=active_bbsz_idx)
            if query_point_record is not None:
                query_point_record[:, :step + 2, :] = torch.index_select(query_point_record[:, :step + 2, :], dim=0, index=active_bbsz_idx)
            if knn_neighbors_keys_record is not None:
                knn_neighbors_keys_record[:, :step + 2, :] = torch.index_select(knn_neighbors_keys_record[:, :step + 2, :], dim=0, index=active_bbsz_idx)
            if knn_neighbors_values_record is not None:
                knn_neighbors_values_record[:, :step + 2, :] = torch.index_select(knn_neighbors_values_record[:, :step + 2, :], dim=0, index=active_bbsz_idx)
            if knn_l2_distance_record is not None:
                knn_l2_distance_record[:, :step + 2, :] = torch.index_select(knn_l2_distance_record[:, :step + 2, :], dim=0, index=active_bbsz_idx)
            if knn_sentence_ids_record is not None:
                knn_sentence_ids_record[:, :step + 2, :] = torch.index_select(knn_sentence_ids_record[:, :step + 2, :], dim=0, index=active_bbsz_idx)
            if knn_token_positions_record is not None:
                knn_token_positions_record[:, :step + 2, :] = torch.index_select(knn_token_positions_record[:, :step + 2, :], dim=0, index=active_bbsz_idx)
            reorder_state = active_bbsz_idx
        for sent in range(len(finalized)):
            scores = torch.tensor([float(elem['score'].item()) for elem in finalized[sent]])
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(List[Dict[str, Tensor]], finalized[sent])
        return finalized

    def _prefix_tokens(self, step: 'int', lprobs, scores, tokens, prefix_tokens, beam_size: 'int'):
        """Handle prefix tokens"""
        prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
        prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
        prefix_mask = prefix_toks.ne(self.pad)
        lprobs[prefix_mask] = torch.tensor(-math.inf)
        lprobs[prefix_mask] = lprobs[prefix_mask].scatter(-1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask])
        eos_mask = prefix_toks.eq(self.eos)
        if eos_mask.any():
            first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[:, 0, 1:step + 1]
            eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
            target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
            assert (first_beam == target_prefix).all()
            tokens = self.replicate_first_beam(tokens, eos_mask_batch_dim, beam_size)
            scores = self.replicate_first_beam(scores, eos_mask_batch_dim, beam_size)
            lprobs = self.replicate_first_beam(lprobs, eos_mask_batch_dim, beam_size)
        return lprobs, tokens, scores

    def replicate_first_beam(self, tensor, mask, beam_size: 'int'):
        tensor = tensor.view(-1, beam_size, tensor.size(-1))
        tensor[mask] = tensor[mask][:, :1, :]
        return tensor.view(-1, tensor.size(-1))

    def finalize_hypos(self, step: 'int', bbsz_idx, eos_scores, tokens, scores, finalized: 'List[List[Dict[str, Tensor]]]', finished: 'List[bool]', beam_size: 'int', attn: 'Optional[Tensor]', src_lengths, max_len: 'int', neural_probs_record: 'Optional[Tensor]', combined_probs_record: 'Optional[Tensor]', query_point_record, knn_neighbors_keys_record, knn_neighbors_values_record, knn_l2_distance_record, knn_sentence_ids_record, knn_token_positions_record):
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        A sentence is finalized when {beam_size} finished items have been collected for it.

        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        Args:
            bbsz_idx (Tensor):
        """
        assert bbsz_idx.numel() == eos_scores.numel()
        tokens_clone = tokens.index_select(0, bbsz_idx)[:, 1:step + 2]
        tokens_clone[:, step] = self.eos
        attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step + 2] if attn is not None else None
        neural_prob_record_clone = neural_probs_record.index_select(0, bbsz_idx)[:, 1:step + 2, :] if neural_probs_record is not None else None
        combined_prob_record_clone = combined_probs_record.index_select(0, bbsz_idx)[:, 1:step + 2, :] if combined_probs_record is not None else None
        query_point_record_clone = query_point_record.index_select(0, bbsz_idx)[:, 1:step + 2, :] if query_point_record is not None else None
        knn_neighbors_keys_record_clone = knn_neighbors_keys_record.index_select(0, bbsz_idx)[:, 1:step + 2, :] if knn_neighbors_keys_record is not None else None
        knn_neighbors_values_record_clone = knn_neighbors_values_record.index_select(0, bbsz_idx)[:, 1:step + 2, :] if knn_neighbors_values_record is not None else None
        knn_l2_distance_record_clone = knn_l2_distance_record.index_select(0, bbsz_idx)[:, 1:step + 2, :] if knn_l2_distance_record is not None else None
        knn_sentence_ids_record_clone = knn_sentence_ids_record.index_select(0, bbsz_idx)[:, 1:step + 2, :] if knn_sentence_ids_record is not None else None
        knn_token_positions_record_clone = knn_token_positions_record.index_select(0, bbsz_idx)[:, 1:step + 2, :] if knn_token_positions_record is not None else None
        pos_scores = scores.index_select(0, bbsz_idx)[:, :step + 1]
        pos_scores[:, step] = eos_scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]
        if self.normalize_scores:
            eos_scores /= (step + 1) ** self.len_penalty
        cum_unfin: 'List[int]' = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)
        sents_seen: 'Dict[str, Optional[Tensor]]' = {}
        for i in range(bbsz_idx.size()[0]):
            idx = bbsz_idx[i]
            score = eos_scores[i]
            unfin_idx = idx // beam_size
            sent = unfin_idx + cum_unfin[unfin_idx]
            seen = str(sent.item()) + '_' + str(unfin_idx.item())
            if seen not in sents_seen:
                sents_seen[seen] = None
            if self.match_source_len and step > src_lengths[unfin_idx]:
                score = torch.tensor(-math.inf)
            if len(finalized[sent]) < beam_size:
                if attn_clone is not None:
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)
                if neural_prob_record_clone is not None:
                    hypo_neural_prob_record = neural_prob_record_clone[i]
                else:
                    hypo_neural_prob_record = torch.empty(0)
                if combined_prob_record_clone is not None:
                    hypo_combined_prob_record = combined_prob_record_clone[i]
                else:
                    hypo_combined_prob_record = torch.empty(0)
                if combined_prob_record_clone is not None:
                    hypo_combined_prob_record = combined_prob_record_clone[i]
                else:
                    hypo_combined_prob_record = torch.empty(0)
                if query_point_record_clone is not None:
                    hypo_query_point_record = query_point_record_clone[i]
                else:
                    hypo_query_point_record = torch.empty(0)
                if knn_neighbors_keys_record_clone is not None:
                    hypo_knn_neighbors_keys_record = knn_neighbors_keys_record_clone[i]
                else:
                    hypo_knn_neighbors_keys_record = torch.empty(0)
                if knn_neighbors_values_record_clone is not None:
                    hypo_knn_neighbors_values_record = knn_neighbors_values_record_clone[i]
                else:
                    hypo_knn_neighbors_values_record = torch.empty(0)
                if knn_l2_distance_record_clone is not None:
                    hypo_knn_l2_distance_record = knn_l2_distance_record_clone[i]
                else:
                    hypo_knn_l2_distance_record = torch.empty(0)
                if knn_sentence_ids_record_clone is not None:
                    hypo_knn_sentence_ids_record = knn_sentence_ids_record_clone[i]
                else:
                    hypo_knn_sentence_ids_record = torch.empty(0)
                if knn_token_positions_record_clone is not None:
                    hypo_knn_token_positions_record = knn_token_positions_record_clone[i]
                else:
                    hypo_knn_token_positions_record = torch.empty(0)
                finalized[sent].append({'tokens': tokens_clone[i], 'score': score, 'attention': hypo_attn, 'alignment': torch.empty(0), 'positional_scores': pos_scores[i], 'neural_probs': hypo_neural_prob_record, 'combined_probs': hypo_combined_prob_record, 'query_point': hypo_query_point_record, 'knn_neighbors_keys': hypo_knn_neighbors_keys_record, 'knn_neighbors_values': hypo_knn_neighbors_values_record, 'knn_l2_distance': hypo_knn_l2_distance_record, 'knn_sentence_ids': hypo_knn_sentence_ids_record, 'knn_token_positions': hypo_knn_token_positions_record})
        newly_finished: 'List[int]' = []
        for seen in sents_seen.keys():
            sent: 'int' = int(float(seen.split('_')[0]))
            unfin_idx: 'int' = int(float(seen.split('_')[1]))
            if not finished[sent] and self.is_finished(step, unfin_idx, max_len, len(finalized[sent]), beam_size):
                finished[sent] = True
                newly_finished.append(unfin_idx)
        return newly_finished

    def is_finished(self, step: 'int', unfin_idx: 'int', max_len: 'int', finalized_sent_len: 'int', beam_size: 'int'):
        """
        Check whether decoding for a sentence is finished, which
        occurs when the list of finalized sentences has reached the
        beam size, or when we reach the maximum length.
        """
        assert finalized_sent_len <= beam_size
        if finalized_sent_len == beam_size or step == max_len:
            return True
        return False

    def calculate_banned_tokens(self, tokens, step: 'int', gen_ngrams: 'List[Dict[str, List[int]]]', no_repeat_ngram_size: 'int', bbsz_idx: 'int'):
        tokens_list: 'List[int]' = tokens[bbsz_idx, step + 2 - no_repeat_ngram_size:step + 1].tolist()
        ngram_index = ','.join([str(x) for x in tokens_list])
        return gen_ngrams[bbsz_idx].get(ngram_index, torch.jit.annotate(List[int], []))

    def transpose_list(self, l: 'List[List[int]]'):
        min_len = min([len(x) for x in l])
        l2 = [[row[i] for row in l] for i in range(min_len)]
        return l2

    def _no_repeat_ngram(self, tokens, lprobs, bsz: 'int', beam_size: 'int', step: 'int'):
        gen_ngrams: 'List[Dict[str, List[int]]]' = [torch.jit.annotate(Dict[str, List[int]], {}) for bbsz_idx in range(bsz * beam_size)]
        cpu_tokens = tokens.cpu()
        for bbsz_idx in range(bsz * beam_size):
            gen_tokens: 'List[int]' = cpu_tokens[bbsz_idx].tolist()
            for ngram in self.transpose_list([gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]):
                key = ','.join([str(x) for x in ngram[:-1]])
                gen_ngrams[bbsz_idx][key] = gen_ngrams[bbsz_idx].get(key, torch.jit.annotate(List[int], [])) + [ngram[-1]]
        if step + 2 - self.no_repeat_ngram_size >= 0:
            banned_tokens = [self.calculate_banned_tokens(tokens, step, gen_ngrams, self.no_repeat_ngram_size, bbsz_idx) for bbsz_idx in range(bsz * beam_size)]
        else:
            banned_tokens = [torch.jit.annotate(List[int], []) for bbsz_idx in range(bsz * beam_size)]
        for bbsz_idx in range(bsz * beam_size):
            lprobs[bbsz_idx][torch.tensor(banned_tokens[bbsz_idx]).long()] = torch.tensor(-math.inf)
        return lprobs


class SequenceGeneratorWithAlignment(KNNSequenceGenerator):

    def __init__(self, models, tgt_dict, left_pad_target=False, **kwargs):
        """Generates translations of a given source sentence.

        Produces alignments following "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            left_pad_target (bool, optional): Whether or not the
                hypothesis should be left padded or not when they are
                teacher forced for generating alignments.
        """
        super().__init__(EnsembleModelWithAlignment(models), tgt_dict, **kwargs)
        self.left_pad_target = left_pad_target

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        finalized = super()._generate(sample, **kwargs)
        src_tokens = sample['net_input']['src_tokens']
        bsz = src_tokens.shape[0]
        beam_size = self.beam_size
        src_tokens, src_lengths, prev_output_tokens, tgt_tokens = self._prepare_batch_for_alignment(sample, finalized)
        if any(getattr(m, 'full_context_alignment', False) for m in self.model.models):
            attn = self.model.forward_align(src_tokens, src_lengths, prev_output_tokens)
        else:
            attn = [finalized[i // beam_size][i % beam_size]['attention'].transpose(1, 0) for i in range(bsz * beam_size)]
        if src_tokens.device != 'cpu':
            src_tokens = src_tokens
            tgt_tokens = tgt_tokens
            attn = [i for i in attn]
        for i in range(bsz * beam_size):
            alignment = utils.extract_hard_alignment(attn[i], src_tokens[i], tgt_tokens[i], self.pad, self.eos)
            finalized[i // beam_size][i % beam_size]['alignment'] = alignment
        return finalized

    def _prepare_batch_for_alignment(self, sample, hypothesis):
        src_tokens = sample['net_input']['src_tokens']
        bsz = src_tokens.shape[0]
        src_tokens = src_tokens[:, None, :].expand(-1, self.beam_size, -1).contiguous().view(bsz * self.beam_size, -1)
        src_lengths = sample['net_input']['src_lengths']
        src_lengths = src_lengths[:, None].expand(-1, self.beam_size).contiguous().view(bsz * self.beam_size)
        prev_output_tokens = data_utils.collate_tokens([beam['tokens'] for example in hypothesis for beam in example], self.pad, self.eos, self.left_pad_target, move_eos_to_beginning=True)
        tgt_tokens = data_utils.collate_tokens([beam['tokens'] for example in hypothesis for beam in example], self.pad, self.eos, self.left_pad_target, move_eos_to_beginning=False)
        return src_tokens, src_lengths, prev_output_tokens, tgt_tokens


class MetaKNetwork(nn.Module):
    """ meta k network of robust knn-mt """

    def __init__(self, max_k=32, midsize=32, midsize_dc=4, topk_wp=8, k_trainable=True, lambda_trainable=True, temperature_trainable=True, relative_label_count=False, device='cuda:0', **kwargs):
        super().__init__()
        self.max_k = max_k
        self.k_trainable = k_trainable
        self.lambda_trainable = lambda_trainable
        self.temperature_trainable = temperature_trainable
        self.relative_label_count = relative_label_count
        self.device = device
        self.mask_for_label_count = None
        self.midsize = midsize
        self.midsize_dc = midsize_dc
        self.topk_wp = topk_wp
        self.distance_func = nn.Sequential(nn.Linear(self.max_k * 2 + self.topk_wp, 1))
        self.distance_fc1 = nn.Sequential(nn.Linear(2, self.midsize_dc), nn.Tanh(), nn.Linear(self.midsize_dc, 1))
        self.distance_fc2 = nn.Sequential(nn.Linear(self.max_k * 2, self.midsize), nn.Tanh(), nn.Linear(self.midsize, 2))

    def forward(self, tgt_index: 'torch.Tensor', knn_dists: 'torch.Tensor', knn_key_feature: 'torch.Tensor', network_probs: 'torch.Tensor', network_select_probs: 'torch.Tensor'):
        B, S, K = knn_dists.size()
        label_counts = self._get_label_count_segment(tgt_index, self.relative_label_count)
        all_key_feature = torch.cat([knn_key_feature.log().unsqueeze(-1), network_select_probs.log().unsqueeze(-1)], -1)
        top_prob, top_idx = torch.topk(network_probs, self.topk_wp)
        knn_feat = torch.cat([knn_dists, label_counts.float()], -1)
        noise_logit = self.distance_fc1(all_key_feature).squeeze(-1)
        sim_lambda = self.distance_func(torch.cat([top_prob.log(), knn_key_feature.log(), network_select_probs.log()], -1))
        lambda_logit = self.distance_fc2(knn_feat.view(B, S, -1))
        knn_lambda = torch.softmax(torch.cat([lambda_logit[:, :, :1], sim_lambda], -1), -1)[:, :, :1]
        tempe = torch.sigmoid(lambda_logit[:, :, 1:2])
        probs = torch.softmax(-knn_dists * tempe + noise_logit, -1)
        return {'probs': probs, 'knn_lambda': knn_lambda}

    def _get_label_count_segment(self, vals, relative=False):
        """ this function return the label counts for different range of k nearest neighbor 
            [[0:0], [0:1], [0:2], ..., ]
        """
        if self.mask_for_label_count is None:
            mask_for_label_count = torch.empty((self.max_k, self.max_k)).fill_(1)
            mask_for_label_count = torch.triu(mask_for_label_count, diagonal=1).bool()
            mask_for_label_count.requires_grad = False
            self.mask_for_label_count = mask_for_label_count
        B, S, K = vals.size()
        expand_vals = vals.unsqueeze(-2).expand(B, S, K, K)
        expand_vals = expand_vals.masked_fill(self.mask_for_label_count, value=-1)
        labels_sorted, _ = expand_vals.sort(dim=-1)
        labels_sorted[:, :, :, 1:] *= (labels_sorted[:, :, :, 1:] - labels_sorted[:, :, :, :-1] != 0).long()
        retrieve_label_counts = labels_sorted.ne(0).sum(-1)
        retrieve_label_counts[:, :, :-1] -= 1
        if relative:
            retrieve_label_counts[:, :, 1:] = retrieve_label_counts[:, :, 1:] - retrieve_label_counts[:, :, :-1]
        return retrieve_label_counts


def calculate_combined_prob(knn_prob, neural_model_logit, lambda_, log_probs):
    """ 
    How vanilla knn-mt calculate the combining probability.
    """
    neural_model_prob = F.softmax(neural_model_logit, dim=-1)
    combined_probs = knn_prob * lambda_ + neural_model_prob * (1 - lambda_)
    extra = {}
    extra['neural_probs'] = neural_model_prob
    extra['unlog_combined_probs'] = combined_probs
    if log_probs:
        combined_probs = torch.log(combined_probs)
    return combined_probs, extra


def calculate_knn_prob(vals, distances, probability_dim, temperature, device, **kwargs):
    """
    How vanilla knn-mt calculates knn probs using retrieved vals and distances.
    """
    scaled_dists = -distances / temperature
    knn_weights = torch.softmax(scaled_dists, dim=-1)
    B, S, K = vals.size()
    knn_probs = torch.zeros(B, S, probability_dim, device=device)
    knn_probs.scatter_add_(dim=-1, index=vals, src=knn_weights)
    return knn_probs


def read_config(path):
    """
    read the config file under the `path` folder

    Args:
        path:
            folder where the config file is stored
    
    Returns:
        dict
    """
    config_file = os.path.join(path, 'config.json')
    with open(config_file, encoding='utf-8', mode='r') as f:
        return json.load(f)


def write_config(path, config):
    """
    write the config file to the `path` folder

    Args:
        path:
            folder where the config file is stored
    
    Returns:
        dict
    """
    with open(os.path.join(path, 'config.json'), encoding='utf-8', mode='w') as f:
        json.dump(config, f, indent=6)


class AdaptiveCombiner(nn.Module):
    """ Adaptive knn-mt Combiner """

    def __init__(self, max_k, probability_dim, k_trainable=True, lambda_trainable=True, temperature_trainable=True, **kwargs):
        super().__init__()
        self.meta_k_network = MetaKNetwork(max_k, k_trainable, lambda_trainable, temperature_trainable, **kwargs)
        self.max_k = max_k
        self.probability_dim = probability_dim
        self.k_trainable = k_trainable
        self.lambda_trainable = lambda_trainable
        self.temperature_trainable = temperature_trainable
        self.kwargs = kwargs
        self.mask_for_distance = None
        assert self.lambda_trainable or 'lambda_' in kwargs, 'if lambda is not trainable, you should provide a fixed lambda_ value'
        assert self.temperature_trainable or 'temperature' in kwargs, 'if temperature is not trainable, you should provide a fixed temperature'
        self.k = None if self.k_trainable else kwargs['k']
        self.lambda_ = None if self.lambda_trainable else kwargs['lambda_']
        self.temperature = None if self.temperature_trainable else kwargs['temperature']

    def get_knn_prob(self, vals, distances, device='cuda:0'):
        metak_outputs = self.meta_k_network(vals, distances)
        if self.lambda_trainable:
            self.lambda_ = metak_outputs['lambda_net_output']
        if self.temperature_trainable:
            self.temperature = metak_outputs['temperature_net_output']
        if self.k_trainable:
            if not hasattr(self, 'mask_for_distance') or self.mask_for_distance is None:
                self.mask_for_distance = self._generate_mask_for_distance(self.max_k, device)
            k_probs = metak_outputs['k_net_output']
            B, S, K = vals.size()
            R_K = k_probs.size(-1)
            distances = distances.unsqueeze(-2).expand(B, S, R_K, K)
            distances = distances * self.mask_for_distance
            if self.temperature_trainable:
                temperature = self.temperature.unsqueeze(-1).expand(B, S, R_K, K)
            else:
                temperature = self.temperature
            distances = -distances / temperature
            knn_weight = torch.softmax(distances, dim=-1)
            weight_sum_knn_weight = torch.matmul(k_probs.unsqueeze(-2), knn_weight).squeeze(-2)
            knn_prob = torch.zeros(B, S, self.probability_dim, device=device)
            knn_prob.scatter_add_(src=weight_sum_knn_weight.float(), index=vals, dim=-1)
        else:
            knn_prob = calculate_knn_prob(vals, distances, self.probability_dim, self.temperature, device=device)
        return knn_prob

    def get_combined_prob(self, knn_prob, neural_model_logit, log_probs=False):
        """ get combined probs of knn_prob and neural_model_prob """
        return calculate_combined_prob(knn_prob, neural_model_logit, self.lambda_, log_probs)

    def dump(self, path):
        """ dump the adaptive knn-mt to disk """
        config = {}
        config['max_k'] = self.max_k
        config['probability_dim'] = self.probability_dim
        config['k_trainable'] = self.k_trainable
        config['lambda_trainable'] = self.lambda_trainable
        config['temperature_trainable'] = self.temperature_trainable
        for k, v in self.kwargs.items():
            config[k] = v
        write_config(path, config)
        torch.save(self.state_dict(), os.path.join(path, 'adaptive_combiner.pt'))

    @classmethod
    def load(cls, path):
        """ load the adaptive knn-mt from disk """
        config = read_config(path)
        adaptive_combiner = cls(**config)
        adaptive_combiner.load_state_dict(torch.load(os.path.join(path, 'adaptive_combiner.pt')))
        return adaptive_combiner

    @staticmethod
    def _generate_mask_for_distance(max_k, device):
        k_mask = torch.empty((max_k, max_k)).fill_(999.0)
        k_mask = torch.triu(k_mask, diagonal=1) + 1
        power_index = torch.tensor([(pow(2, i) - 1) for i in range(0, int(math.log(max_k, 2)) + 1)])
        k_mask = k_mask[power_index]
        k_mask.requires_grad = False
        k_mask = k_mask
        return k_mask


class BandwidthEstimator(nn.Module):

    def __init__(self, query_dim, device='cuda:0'):
        super().__init__()
        self.fc = nn.Linear(query_dim * 2, 1)

    def forward(self, query, average_key):
        x = torch.cat((query, average_key), dim=-1)
        x = self.fc(x)
        x = torch.exp(x)
        return x


class WeightEstimator(nn.Module):
    """ model to get the lamba weight"""

    def __init__(self, query_dim, device='cuda:0'):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(query_dim * 2, query_dim), nn.ReLU(), nn.Linear(query_dim, 1), nn.Sigmoid())

    def forward(self, query, weighted_sum_key):
        x = torch.cat((query, weighted_sum_key), dim=-1)
        return self.model(x)


class KernelSmoothedCombiner(nn.Module):
    """
    combiner for kernel smoothed knn-mt
    """

    def __init__(self, query_dim, probability_dim, device='cuda:0', kernel_type='laplacian'):
        super().__init__()
        self.bandwidth_estimator = BandwidthEstimator(query_dim=query_dim, device=device)
        self.weight_estimator = WeightEstimator(query_dim=query_dim, device=device)
        self.device = device
        self.query_dim = query_dim
        self.probability_dim = probability_dim
        self.kernel_type = kernel_type
        self.lambda_ = None

    def get_knn_prob(self, query, keys, vals, distances, device='cuda:0', **kwargs):
        """caculate the knn prob """
        if self.training:
            keys = keys[..., 1:, :]
            vals = vals[..., 1:]
            distances = distances[..., 1:]
        average_key = torch.mean(keys, dim=-2)
        query = query.float()
        average_key = average_key.float()
        bandwidth = self.bandwidth_estimator(query, average_key)
        if self.kernel_type == 'gaussian':
            scaled_dists = -distances / bandwidth
        else:
            scaled_dists = -torch.sqrt(distances) / bandwidth
        knn_weights = torch.softmax(scaled_dists, dim=-1).unsqueeze(-1)
        weighted_sum_key = knn_weights.repeat(*([1] * (knn_weights.dim() - 1)), keys.size(-1)) * keys
        weighted_sum_key = torch.sum(weighted_sum_key, dim=-2)
        B, S, K = vals.size()
        knn_probs = torch.zeros(B, S, self.probability_dim, device=device)
        knn_probs.scatter_add_(dim=-1, index=vals, src=knn_weights.squeeze(-1))
        self.lambda_ = self.weight_estimator(query, weighted_sum_key)
        return knn_probs

    def get_combined_prob(self, knn_prob, neural_model_logit, log_probs=False):
        """ 
        strategy of combine probability 
        """
        return calculate_combined_prob(knn_prob, neural_model_logit, self.lambda_, log_probs)

    def dump(self, path):
        """
        dump a kernel smoothed combiner to disk"""
        if not os.path.exists(path):
            os.makedirs(path)
        config = {}
        config['query_dim'] = self.query_dim
        config['probability_dim'] = self.probability_dim
        config['kernel_type'] = self.kernel_type
        write_config(path, config)
        torch.save(self.state_dict(), os.path.join(path, 'kernel_smoothed_combiner.pt'))

    @classmethod
    def load(cls, path):
        """
        load kernel smoothed combiner from disk"""
        config = read_config(path)
        kernel_smoothed_combiner = cls(**config)
        kernel_smoothed_combiner.load_state_dict(torch.load(os.path.join(path, 'kernel_smoothed_combiner.pt')))
        return kernel_smoothed_combiner


class RobustCombiner(nn.Module):
    """ Robust knn-mt Combiner """

    def __init__(self, max_k, probability_dim, midsize=32, midsize_dc=4, topk_wp=8, **kwargs):
        super().__init__()
        self.max_k = max_k
        self.probability_dim = probability_dim
        self.midsize = midsize
        self.midsize_dc = midsize_dc
        self.topk_wp = topk_wp
        self.kwargs = kwargs
        self.mask_for_distance = None
        self.meta_k_network = MetaKNetwork(max_k=self.max_k, midsize=self.midsize, midsize_dc=self.midsize_dc, topk_wp=self.topk_wp, **kwargs)

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""

        def _apply(m):
            if hasattr(m, 'set_num_updates') and m != self:
                m.set_num_updates(num_updates)
        self.apply(_apply)

    def get_knn_prob(self, tgt_index: 'torch.Tensor', knn_dists: 'torch.Tensor', knn_key_feature: 'torch.Tensor', network_probs: 'torch.Tensor', network_select_probs: 'torch.Tensor', device='cuda:0'):
        metak_outputs = self.meta_k_network(tgt_index=tgt_index, knn_dists=knn_dists, knn_key_feature=knn_key_feature, network_probs=network_probs, network_select_probs=network_select_probs)
        self.lambda_ = metak_outputs['knn_lambda']
        knn_prob = torch.zeros(*network_probs.shape, device=device)
        knn_prob.scatter_add_(dim=-1, index=tgt_index, src=metak_outputs['probs'])
        return knn_prob

    def get_combined_prob(self, knn_prob, neural_model_logit, log_probs=False):
        """ get combined probs of knn_prob and neural_model_prob """
        return calculate_combined_prob(knn_prob, neural_model_logit, self.lambda_, log_probs)

    def dump(self, path):
        """ dump the robust knn-mt to disk """
        config = {}
        config['max_k'] = self.max_k
        config['probability_dim'] = self.probability_dim
        config['midsize'] = self.midsize
        config['midsize_dc'] = self.midsize_dc
        for k, v in self.kwargs.items():
            config[k] = v
        write_config(path, config)
        torch.save(self.state_dict(), os.path.join(path, 'robust_combiner.pt'))

    @classmethod
    def load(cls, path):
        """ load the robust knn-mt from disk """
        config = read_config(path)
        robust_combiner = cls(**config)
        robust_combiner.load_state_dict(torch.load(os.path.join(path, 'robust_combiner.pt')))
        return robust_combiner

    @staticmethod
    def _generate_mask_for_distance(max_k, device):
        k_mask = torch.empty((max_k, max_k)).fill_(999.0)
        k_mask = torch.triu(k_mask, diagonal=1) + 1
        power_index = torch.tensor([(pow(2, i) - 1) for i in range(0, int(math.log(max_k, 2)) + 1)])
        k_mask = k_mask[power_index]
        k_mask.requires_grad = False
        k_mask = k_mask
        return k_mask


class Memmap:
    """
    automatic capacity expansion memmap.
    If you create a Memmap with mode "w+" for write, you needn't declare it's shpae and dtype,
    Memmap will inference its shape and dtype the first time you call `add`.
    If you create a Memmap with mode "r" for read, you must give dtype and shape infomation on creation.

    Usage:
        # Create and Write a Memmap
        mmap = Memmap("/home/keys", mode="w+")
        a = torch.rand(10,64)
        mmap.add(a) 
        b = np.random.randn(38, 64)
        mmap.dump() # dump the file to disk

        # Read a Existed Memmap
        mmap = Memmap("/home/vals", mode="r", dtype=int, shape=(20000,))
    """

    def __init__(self, filename, mode='r', dtype=None, shape=None):
        self.filename = filename
        self.mode = mode
        file_exists = os.path.exists(filename)
        if mode == 'r' or mode == 'r+':
            assert file_exists, "The memmap file %s dosen't exist" % filename
            assert dtype is not None, 'must specify dtype when read a memmap'
            assert shape is not None, 'must specify shape when read a memmap'
            if isinstance(shape, list):
                shape = tuple(shape)
            self.data = np.memmap(filename, dtype=self.convert_data_type(dtype), shape=shape, mode=mode)
            self.size = shape[0]
            self.dtype = dtype
        else:
            self.data = None
            self.size = 0
            self.dtype = None

    @property
    def shape(self):
        """
        return the logical shape of a memmap.
        These function dont count redundant preallocated entries.

        for example, if we allocate [1000,5,8] space but the real entry size is 500,
        we will return [500, 5, 8] here.
        """
        return tuple([self.size] + list(self.data.shape[1:]))

    def add(self, data):
        assert self.mode == 'r+' or self.mode == 'w+', "You can't write to a Memmap with {} mode.".format(self.mode)
        if self.data is None:
            preallocated_shape = list(data.shape) if data.shape else [1]
            preallocated_shape[0] = 300000
            preallocated_shape = tuple(preallocated_shape)
            self.dtype = self.convert_data_type(data.dtype)
            self.data = np.memmap(self.filename, dtype=self.dtype, shape=preallocated_shape, mode=self.mode)
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        assert data.dtype == self.data.dtype, 'Inconsistent data types when add to memmap, require %s but add %s' % (str(self.data.dtype), str(data.dtype))
        assert data.shape[1:] == self.data.shape[1:], 'Inconsistent data dimension when add to memmap, require %s but add %s' % (str(self.data.shape[1:]), str(data.shape[1:]))
        data_shape = data.shape if data.shape else (1,)
        need_resize = False
        now_capacity = self.data.shape[0]
        new_capacity = now_capacity
        while data_shape[0] + self.size >= new_capacity:
            need_resize = True
            if new_capacity < 5000000:
                new_capacity = 2 * new_capacity
            else:
                new_capacity = int(new_capacity * 1.5)
        if need_resize:
            new_shape = [new_capacity] + list(self.data.shape[1:])
            new_shape = tuple(new_shape)
            new_memory_footprint = self.data.dtype.itemsize
            for x in new_shape:
                new_memory_footprint *= x
            self.data.base.resize(new_memory_footprint)
            self.data.flush()
            self.data = np.memmap(self.filename, dtype=self.dtype, mode='r+', shape=new_shape)
        self.data[self.size:self.size + data_shape[0]] = data
        self.size += data_shape[0]

    def drop_redundant(self):
        """
        trim the memmap, discard redundant preallocated entries
        """
        if self.size != self.data.shape[0]:
            new_shape = self.shape
            new_memory_footprint = self.data.dtype.itemsize
            for x in new_shape:
                new_memory_footprint *= x
            self.data.base.resize(new_memory_footprint)
            self.data.flush()
            self.data = np.memmap(self.filename, dtype=self.dtype, mode='r+', shape=new_shape)

    def dump(self):
        """ 
        when we dump the Memmap to disk, we dicard redundant preallocated entries.
        It means we trim the memmap to `self.size` entries
        """
        self.drop_redundant()

    @staticmethod
    def convert_data_type(data_type):
        """ convert an input data dtype to numpy compatible dtype """
        data_type_convert_dict = {np.float32: np.float32, np.float16: np.float16, np.dtype('float32'): np.float32, np.dtype('float16'): np.float16, torch.float: np.float32, torch.float32: np.float32, torch.float16: np.float16, 'float32': np.float32, 'float16': np.float16, str(np.float32): np.float32, str(np.float16): np.float16, str(torch.float): np.float32, str(torch.float32): np.float32, str(torch.float16): np.float16, np.int16: int, np.int32: int, np.int64: int, np.dtype('int64'): int, np.dtype('int32'): int, np.dtype('int16'): int, np.int_: int, int: int, torch.int64: int, torch.int32: int, torch.int: int, str(np.int16): int, str(np.int32): int, str(np.int64): int, str(np.int_): int, "<class 'int'>": int, str(torch.int64): int, str(torch.int32): int, str(torch.int): int, 'int': int}
        assert data_type in data_type_convert_dict, 'Unsupported data type when convert dtype for memmap!'
        return data_type_convert_dict[data_type]


class Datastore:
    """
    implement vanilla datastore
    """

    def __init__(self, path, datas=None, **kwargs):
        """
        Args:
            path(`str`):
                the directory to save datastore files
            datas(`dict`):
                the dict of inner data
            data_infos(`dict`):
                The infomations of datastore inner data
        
        """
        self.path = path
        self.datas = datas if datas is not None else {}
        if not os.path.exists(path):
            os.makedirs(path)

    def __getitem__(self, name):
        """ access  inner data
        Usage:
            ds = Datastore(path="/home/datastore")
            a = torch.rand(3,1024)
            ds["keys"].add(a)
            b = torch.rand(3,1)
            ds["vals"].add(b)
        """
        if name not in self.datas:
            self.datas[name] = Memmap(filename=os.path.join(self.path, name + '.npy'), mode='w+')
        return self.datas[name]

    def __setitem__(self, name, data):
        """ set inner data directory
        Usage:
            ds = Datastore(path="/home/datastore")
            mp = Memmap("/home/vals.npy", mode="r")
            ds["vals"] = mp
        """
        assert isinstance(data, Memmap), '__setitme__ is designed for set Memmap object'
        self.datas[name] = data

    def __delitem__(self, name):
        """ delete a inner data """
        if name in self.datas:
            del self.datas[name]

    def set_pad_mask(self, mask):
        """ 
        save the pad mask 
        """
        self.mask = mask

    def get_pad_mask(self):
        """
        get the saved mask
        """
        assert hasattr(self, 'mask'), 'You should set pad mask first!'
        return self.mask

    @classmethod
    def load(cls, path, load_list):
        """
        load the datastore from the `path` folder

        Args:
            path(`str`):
                folder where the datastore files is stored
            load_list(`list`):
                specify the data name which we want to load
        Return:
            Datastore object(`Datastore`)
        """
        datas = {}
        config = read_config(path)
        for name in load_list:
            assert name in config['data_list'], "You haven't save {} but you list it in load_list".format(name)
            if os.path.exists(os.path.join(path, name + '.npy')):
                _info = config['data_infos'][name]
                datas[name] = Memmap(filename=os.path.join(path, name + '.npy'), shape=_info['shape'], dtype=_info['dtype'], mode='r+')
        return cls(path, datas)

    def dump(self, verbose=True, dump_list=None):
        """
        store the datastore files and config file to disk.
        
        Args:
            verbose: whether to display detailed infomation
            dump_list: specify the data names which you want to dump. if dump_list is None, dump all data
        """
        config = {}
        config['data_list'] = []
        config['data_infos'] = {}
        for name in self.datas.keys():
            config['data_list'].append(name)
            config['data_infos'][name] = {'name': name, 'shape': self.datas[name].shape, 'dtype': str(self.datas[name].dtype)}
            if dump_list is None or name in dump_list:
                self.datas[name].dump()
                if verbose:
                    None
        write_config(self.path, config)

    def load_faiss_index(self, filename, move_to_gpu=True, verbose=True):
        """
        load faiss index from disk

        Args:
            filename: the prefix of faiss_index file, for example `keys.faiss_index`, filename is `keys`
            move_to_gpu: wether move the faiss index to GPU
        """
        index_path = os.path.join(self.path, filename + '.faiss_index')
        config = read_config(self.path)
        if not hasattr(self, 'faiss_index') or self.faiss_index is None:
            self.faiss_index = {}
        self.faiss_index[filename] = load_faiss_index(path=index_path, n_probe=32, move_to_gpu=move_to_gpu, verbose=verbose)

    def build_faiss_index(self, name, verbose=True, do_pca=False, pca_dim=256, use_gpu=True):
        """
        build faiss index for a data.
        the output file named name+.faiss_index

        Args:
            name: The data name which need to build faiss index
            verbose: display detailed message
            do_pca: wether do a PCA when building faiss index
            pca_dim: if use PCA, the PCA output dim
        """
        if not isinstance(self.datas[name], Memmap):
            None
            os.exit(1)
        build_faiss_index(self.datas[name].data, self.datas[name].shape, os.path.join(self.path, name + '.faiss_index'), do_pca=do_pca, pca_dim=pca_dim, use_gpu=use_gpu, verbose=verbose)


class ReductionNetwork(nn.Module):
    """ network to compress dimension """

    def __init__(self, dictionary_len, input_dim, output_dim, dropout=0.0, train_mode=True):
        super().__init__()
        self.dictionary_len = dictionary_len
        self.reduction_layer = nn.Sequential(nn.Linear(input_dim, input_dim // 4), nn.Tanh(), nn.Dropout(p=dropout), nn.Linear(input_dim // 4, output_dim))
        nn.init.xavier_normal_(self.reduction_layer[0].weight, gain=0.01)
        nn.init.xavier_normal_(self.reduction_layer[-1].weight, gain=0.1)
        if train_mode:
            self.word_predict_layer = nn.Linear(output_dim, self.dictionary_len, bias=False)
            nn.init.normal_(self.word_predict_layer.weight, mean=0, std=output_dim ** -0.5)

    def forward(self, data, dr_loss_ratio, nce_loss_ratio, wp_loss_ratio, device='cuda:0'):
        """ forward data to get loss

        The final loss is:
            loss = dr_loss_ratio*dr_loss + nce_loss_ratio*nce_loss + wp_loss_ratio*wp_loss
        """
        assert dr_loss_ratio + nce_loss_ratio + wp_loss_ratio == 1.0, "ERROR: loss ratio's sum must equal to 1.0"
        pivot_samples = data['pivot_samples']
        positive_samples = data['positive_samples']
        negative_samples = data['negative_samples']
        pivot_ids = data['pivot_ids']
        positive_ids = data['positive_ids']
        negative_ids = data['negative_ids']
        batch_size = pivot_ids.shape[0]
        stack_data = torch.cat([pivot_samples, positive_samples, negative_samples], dim=0)
        stack_ids = torch.cat([pivot_ids, positive_ids, negative_ids], dim=0)
        reducted_data = self.reduction_layer(stack_data)
        reducted_pivot_data, reducted_positive_data, reducted_negative_data = reducted_data[:batch_size], reducted_data[batch_size:2 * batch_size], reducted_data[2 * batch_size:3 * batch_size]
        dr_loss = 0.0
        if dr_loss_ratio != 0.0:
            pos_dis = nn.MSELoss(reduce=False)(reducted_pivot_data, reducted_positive_data).sum(-1)
            margin = 10.0

            def hingle_loss(pivot_data, negative_data, margin):
                neg_dis = nn.MSELoss(reduce=False)(pivot_data, negative_data).sum(-1)
                neg_dis = (neg_dis < margin).float() * neg_dis + (neg_dis >= margin).float() * margin
                return neg_dis
            neg_dis = hingle_loss(reducted_pivot_data, reducted_negative_data, margin)
            soft_pos = 1.0
            soft_neg = 1.0
            soft_pos_loss = soft_pos * pos_dis
            soft_neg_loss = soft_neg * (margin / (neg_dis + 0.001))
            dr_loss = (soft_pos_loss + soft_neg_loss).mean()
        nce_loss = 0
        if nce_loss_ratio != 0.0:
            nce_distance_pos = -(reducted_positive_data[:, None, :] * reducted_pivot_data[None, :, :]).sum(-1)
            nce_distance = nce_distance_pos
            """
            NOTE the simplest nce is to optimize among positive pairs in a batch, but sampling of positive
            pairs ignore tokens of low frequence Which make the optimization only done for high-frequence vocab.
            To address this, we optimize positive pairs nce loss along with negative pairs
            """
            nce_distance_pos = -(reducted_positive_data[:, None, :] * reducted_pivot_data[None, :, :]).sum(-1)
            nce_distance_neg = -(reducted_negative_data[:, None, :] * reducted_pivot_data[None, :, :]).sum(-1)
            nce_distance = torch.cat([nce_distance_pos, nce_distance_neg], axis=1)
            nce_lprobs = torch.nn.functional.log_softmax(-nce_distance, dim=-1)
            nce_target = torch.arange(end=batch_size)
            nce_loss = label_smoothed_nll_loss(nce_lprobs, nce_target, 0.001, reduce=True)
            nce_loss = nce_loss / float(batch_size)
        wp_loss = 0
        if wp_loss_ratio != 0.0:
            logits = self.word_predict_layer(reducted_data)
            word_probs = nn.functional.log_softmax(logits, dim=-1)
            word_predict_loss = label_smoothed_nll_loss(word_probs, stack_ids, 0.001, reduce=True)
            wp_loss = word_predict_loss / float(batch_size)
        loss = dr_loss_ratio * dr_loss + nce_loss_ratio * nce_loss + wp_loss_ratio * wp_loss
        return loss


class PckDatastore(Datastore, nn.Module):
    """ Implementation of PCK-MT datastore """

    def __init__(self, path, dictionary_len, reduction_network_input_dim=None, reduction_network_output_dim=None, datas=None, **kwargs):
        Datastore.__init__(self, path, datas, **kwargs)
        nn.Module.__init__(self)
        self.dictionary_len = dictionary_len
        if reduction_network_input_dim and reduction_network_output_dim and dictionary_len:
            self.reduction_network = ReductionNetwork(dictionary_len, reduction_network_input_dim, reduction_network_output_dim, train_mode=False)
            self.reduction_network_input_dim = reduction_network_input_dim
            self.reduction_network_output_dim = reduction_network_output_dim

    def prune_size(self, output_path, n_of_4_gram=4, prune_style='random', sample_rate=0.1, minimum_sample=2, thread_num=30):
        """ prune the datastore size """
        start_time = time.time()
        ppl_mask = (self.datas['ids_4_gram'].data != 0).astype(np.float32)
        """e.g., for a phrase 'it is a lovely dog' (which ends with 'dog'),
        we collect normalized ppls of all n-grams:
        - ppl of 'dog' = ppls[:1] / 1
        - ppl of 'lovely dog' = ppls[:2].sum() / 2 
        - ppl of 'a lovely dog' = ppls[:3].sum() / 3
        - ppl of 'is a lovely dog' = ppls[:4].sum() / 4
        - ppl of 'it is a lovely dog' = ppls[:5].sum() / 5
        """
        n_gram_uniform_ppl = -np.log(self['probs_4_gram'].data * ppl_mask + 1e-05)
        n_gram_uniform_ppl = np.concatenate([(n_gram_uniform_ppl[:, :i + 1].sum(-1, keepdims=True) / (i + 1)) for i in range(n_gram_uniform_ppl.shape[-1])], axis=-1)
        None
        tgt_entropy = self['entropy'].data
        if 1 <= n_of_4_gram <= 4:
            n_gram_uniform_ppl = np.min(n_gram_uniform_ppl, axis=-1)
            linear_hash_weight = np.array([0] + [math.exp(i + 1) for i in range(n_of_4_gram - 1)])
            ids_n_gram_hash = (self['ids_4_gram'].data[:, :n_of_4_gram] @ linear_hash_weight[:, None])[:, 0]
            ids_n_gram_hash = ids_n_gram_hash / np.power(np.log10(ids_n_gram_hash + 1.0) + 1, 10)
            ids_n_gram_hash = ids_n_gram_hash
            n_gram = ids_n_gram_hash + self['ids_4_gram'].data[:, 0]
            del ids_n_gram_hash
        else:
            raise NotImplementedError('not implemented for n = %d' % n_of_4_gram)
        table_n_gram_counter = Counter(n_gram)
        table_n_gram = list(table_n_gram_counter.keys())
        table_n_gram_idx_dict = {}
        for k in table_n_gram:
            table_n_gram_idx_dict[k] = np.zeros(table_n_gram_counter[k], dtype=np.int64)
        for idx, gram in enumerate(n_gram):
            if table_n_gram_counter[gram] <= 0:
                continue
            table_n_gram_counter[gram] -= 1
            table_n_gram_idx_dict[gram][table_n_gram_counter[gram]] = idx
        del table_n_gram_counter
        None
        """
        NOTE: about table_n_gram_idx_dict
        For a trainset that contains 6 sentences:
            I:   'this is a good place'
            II:  'it is rainy.'
            III: 'he is good'
            IV:  'i think he is excellent'
            V:   'yes he is'
            VI:  'is it ?'
        We build the datastore:
        0-this, 1-is, 2-a, 3-good, 4-place,
        5-it, 6-is, 7-rainy,
        8-he, 9-is, 10-good,
        11-i, 12-think, 13-he, 14-is, 15-excellent,
        16-yes, 17-he, 18-is,
        19-is, 20-it, 21-?
        the 1-gram list of "is":  [
            [1('this is')],
            [6('it is')],
            [9('he is')],
            [14('he is')],
            [18('he is')],
            [19('is')]
        ]
        the 2-gram list of "is" that ends with the token "is": [
            [1 ('this is')],
            [6 ('it is')],
            [9, 14, 18 ('he is')],
            [19 ('[padding] is')]
        ]
        etc.
        """
        None
        thread_width = len(table_n_gram_idx_dict) // thread_num + 1
        pool = Pool(processes=thread_num)
        table_n_gram_idx_dict_keys = list(table_n_gram_idx_dict.keys())
        results = [pool.apply_async(func=self._n_gram_prune_thread_inner_table_n_gram_idx_dict, args=(dict([(k, table_n_gram_idx_dict[k]) for k in table_n_gram_idx_dict_keys[i * thread_width:min((i + 1) * thread_width, len(table_n_gram_idx_dict))]]), prune_style, minimum_sample, sample_rate, n_gram_uniform_ppl if 'ppl' in prune_style else None, tgt_entropy if 'entropy' in prune_style else None)) for i in range(thread_num)]
        pool.close()
        pool.join()
        table_n_gram_idx_dict = {}
        for res in results:
            table_n_gram_idx_dict.update(res.get())
        table_n_gram_idx_dict_keys = list(table_n_gram_idx_dict.keys())
        pool = Pool(processes=thread_num)
        thread_width = len(table_n_gram_idx_dict) // thread_num + 1
        None
        results = [pool.apply_async(func=self._collect_pruned_n_grams_thread, args=(dict([(k, table_n_gram_idx_dict[k]) for k in table_n_gram_idx_dict_keys[i * thread_width:min((i + 1) * thread_width, len(table_n_gram_idx_dict))]]),)) for i in range(thread_num)]
        pool.close()
        pool.join()
        output_datastore = Datastore(path=output_path)
        for res in results:
            vals_l, dbidx_l, tgt_lens_l, src_lens_l = res.get()
            vals_l = [val for vals in vals_l for val in vals]
            keys_l = [self['keys'].data[dbidx] for dbidxs in dbidx_l for dbidx in dbidxs]
            vals = np.array(vals_l, dtype=self['vals'].data.dtype)
            keys = np.array(keys_l, dtype=self['keys'].data.dtype)
            output_datastore['keys'].add(keys)
            output_datastore['vals'].add(vals)
        output_datastore.dump()
        output_datastore.build_faiss_index('keys')
        None

    @classmethod
    def random_sample(cls, keys, nums):
        """random sample if keys' size bigger than nums """
        assert type(keys) in [list, np.ndarray], type(keys)
        if isinstance(keys, list):
            if len(keys) > nums:
                return random.sample(keys, nums)
            else:
                return keys
        elif keys.shape[0] > nums:
            return keys[np.random.choice(keys.shape[0], nums, replace=False)]
        else:
            return keys

    @classmethod
    def _n_gram_prune_thread_inner_table_n_gram_idx_dict(cls, table_n_gram_idx_dict, prune_style, minimum_sample, sample_rate, n_gram_uniform_ppl=None, tgt_entropy=None):
        """prune the items which has same n-gram hash code.
            the prune policy has: random, ppl, tgt_entropy
        """
        for n_gram_str_symbol, np_idxs in table_n_gram_idx_dict.items():
            selected_num = max(minimum_sample, int(sample_rate * np_idxs.shape[0]))
            if np_idxs.shape[0] <= selected_num:
                continue
            if prune_style == 'random':
                table_n_gram_idx_dict[n_gram_str_symbol] = cls.random_sample(np_idxs, selected_num)
            elif 'ppl' in prune_style:
                ppl_group = n_gram_uniform_ppl[np_idxs]
                if prune_style == 'prune_high_ppl':
                    mask = np.argpartition(ppl_group, selected_num)[:selected_num]
                elif prune_style == 'prune_low_ppl':
                    mask = np.argpartition(ppl_group, -selected_num)[-selected_num:]
                elif prune_style == 'prune_half_low_half_high_ppl':
                    mask1 = np.argpartition(ppl_group, selected_num // 2)[:selected_num // 2]
                    mask2 = np.argpartition(ppl_group, -selected_num // 2)[-selected_num // 2:]
                    mask = np.concatenate((mask1, mask2), axis=0)
                elif prune_style == 'prune_similar_ppl':
                    mask = cls.ppl_split_and_sample(ppl_group, sample_rate=sample_rate)
                table_n_gram_idx_dict[n_gram_str_symbol] = np_idxs[mask]
            elif 'entropy' in prune_style:
                entropy_group = tgt_entropy[np_idxs]
                if prune_style == 'prune_high_entropy':
                    mask = np.argpartition(entropy_group, selected_num)[:selected_num]
                elif prune_style == 'prune_low_entropy':
                    mask = np.argpartition(entropy_group, -selected_num)[-selected_num:]
                elif prune_style == 'prune_half_low_half_high_entropy':
                    mask1 = np.argpartition(entropy_group, selected_num // 2)[:selected_num // 2]
                    mask2 = np.argpartition(entropy_group, -selected_num // 2)[-selected_num // 2:]
                    mask = np.concatenate((mask1, mask2), axis=0)
                elif prune_style == 'prune_similar_entropy':
                    mask = cls.ppl_split_and_sample(entropy_group, sample_rate=sample_rate)
                table_n_gram_idx_dict[n_gram_str_symbol] = np_idxs[mask]
            else:
                raise NotImplementedError('not implemented prune_style = %s' % prune_style)
        return table_n_gram_idx_dict

    @classmethod
    def ppl_split_and_sample(cls, ppl_group: 'np.array', sample_rate: 'float'=0.3, translation_cost_threshold: 'float'=1.5, minimum_sample: 'int'=2):
        if ppl_group.shape[0] > 10000.0:
            sc = Birch(n_clusters=None, threshold=translation_cost_threshold)
            clustering = sc.fit(ppl_group[:, None])
            labels = clustering.labels_
            ppl_clusters = [[] for _ in range(labels.max() + 1)]
            for n in range(labels.shape[0]):
                if labels[n] == -1:
                    continue
                ppl_clusters[labels[n]].append(n)
            for i, clusters in enumerate(ppl_clusters):
                clusters = np.array(clusters)
                sample_nums = max(min(minimum_sample, clusters.shape[0]), int(sample_rate * clusters.shape[0]))
                clusters = cls.random_sample(clusters, sample_nums)
                ppl_clusters[i] = clusters
            for n in range(labels.shape[0]):
                if labels[n] == -1:
                    ppl_clusters.append(np.array([n], dtype=np.int))
            ppl_clusters = [ppl_index for ppl_index in ppl_clusters if ppl_index.shape[0] > 0]
            mask = np.hstack(ppl_clusters)
            assert mask.shape[0] <= ppl_group.shape[0]
            return mask
        else:
            ppl_affinity = ppl_group[None] - ppl_group[:, None]
            ppl_similar = np.abs(ppl_affinity) <= translation_cost_threshold
            ppl_idx_clusters = []
            idx_empty = np.arange(ppl_similar.shape[0])
            while ppl_similar.sum() != 0.0:
                ppl_similar_numbers = ppl_similar.astype(np.float32).sum(-1)
                ppl_max_similar_idx = np.argmax(ppl_similar_numbers)
                select_mask = ppl_similar[ppl_max_similar_idx]
                ppl_idx_clusters.append(idx_empty[select_mask])
                ppl_similar = ppl_similar[~select_mask]
                ppl_similar = ppl_similar[:, ~select_mask]
                idx_empty = idx_empty[~select_mask]
            for i, clusters in enumerate(ppl_idx_clusters):
                sample_nums = max(min(minimum_sample, clusters.shape[0]), int(sample_rate * clusters.shape[0]))
                clusters = cls.random_sample(clusters, sample_nums)
                ppl_idx_clusters[i] = clusters
            mask = np.hstack(ppl_idx_clusters)
            assert mask.shape[0] <= ppl_group.shape[0], ppl_idx_clusters
            return mask

    @classmethod
    def _collect_pruned_n_grams_thread(cls, table_n_gram_idx_dict):
        """ 
        for a dict {"3.12":[1,4,3], "4.52":[6,9], "3.89":[11,2]}
        we return: 
            val_list: [[3,3,3],[4,4],[3,3]]
            dbidx_list: [[1,4,3],[6,9],[11,2]]
        """
        len_d = len(table_n_gram_idx_dict)
        val_list = [[] for _ in range(len_d)]
        dbidx_list = [[] for _ in range(len_d)]
        for i, (n_gram_str_symbol, np_idxs) in enumerate(table_n_gram_idx_dict.items()):
            np_idxs = table_n_gram_idx_dict[n_gram_str_symbol]
            vocab_id = int(n_gram_str_symbol)
            val_list[i] = [vocab_id] * np_idxs.shape[0]
            dbidx_list[i] = np_idxs.tolist()
        return val_list, dbidx_list, None, None

    def train_reduction_network(self, triplet_dataset, batch_size, dr_loss_ratio, nce_loss_ratio, wp_loss_ratio, lr, min_lr, patience, max_update, log_path, valid_interval, device='cuda:0'):
        """ a simple function to train reduction network"""
        assert self.training, 'Pytorch is not on trainning mode'
        assert max_update > valid_interval, 'max_update must bigger than valid_interval'
        tb_writer = None
        try:
            tb_writer = SummaryWriter(log_path)
        except:
            None
        dataloader = DataLoader(dataset=triplet_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
        valid_dataloader = DataLoader(dataset=triplet_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
        None
        self.reduction_network
        self.reduction_network.train()
        optimizer = optim.Adam(self.reduction_network.parameters(), lr, betas=(0.9, 0.98))
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, min_lr=min_lr, factor=0.5)
        min_valid_loss = 10000000.0
        best_checkpoint = None
        no_improved_cnt = 0
        pbar = tqdm(total=max_update)
        update_step = 0
        valid_losses = []
        valid_cnt = 0
        break_flag = False
        while True:
            if break_flag:
                break
            for data in dataloader:
                if update_step >= max_update:
                    break_flag = True
                    break
                if (update_step + 1) % valid_interval == 0:
                    valid_losses = []
                    for valid_data in valid_dataloader:
                        with torch.no_grad():
                            valid_loss = self.reduction_network(valid_data, dr_loss_ratio, nce_loss_ratio, wp_loss_ratio, device)
                            valid_losses.append(valid_loss.item())
                    avg_valid_loss = sum(valid_losses) / len(valid_losses)
                    if tb_writer:
                        tb_writer.add_scalar('valid_loss', avg_valid_loss, update_step)
                    None
                    if avg_valid_loss < min_valid_loss:
                        None
                        best_checkpoint = self.reduction_network.state_dict()
                        min_valid_loss = avg_valid_loss
                        no_improved_cnt = 0
                    else:
                        no_improved_cnt += 1
                        None
                        if no_improved_cnt >= patience:
                            None
                            break_flag = True
                            break
                    scheduler.step(avg_valid_loss)
                    update_step += 1
                    pbar.update(1)
                train_loss = self.reduction_network(data, dr_loss_ratio, nce_loss_ratio, wp_loss_ratio, device)
                if tb_writer:
                    tb_writer.add_scalar('train_loss', train_loss.item(), update_step)
                pbar.update(1)
                pbar.set_postfix(step=update_step, loss=train_loss.item())
                optimizer.zero_grad()
                train_loss.backward()
                nn.utils.clip_grad_norm_(self.reduction_network.parameters(), 1.0)
                optimizer.step()
                update_step += 1
        self.reduction_network.load_state_dict(best_checkpoint)
        None
        None

    def vector_reduct(self, x, device='cuda:0'):
        """ reduct the input x with reduct network """
        self.reduction_network = self.reduction_network
        self.reduction_network.eval()
        x = x
        assert x.size()[-1] == self.reduction_network_input_dim, 'Error: The vector size is not correct!'
        with torch.no_grad():
            reducted_x = self.reduction_network.reduction_layer(x)
        return reducted_x

    def reconstruct_keys_with_reduction_network(self, output_dir, batch_size=100):
        None
        start_idx = 0
        key_size = self['keys'].size
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        new_keys = Memmap(os.path.join(output_dir, 'keys.npy'), mode='w+')
        while start_idx < key_size:
            end_idx = min(start_idx + batch_size, key_size)
            original_key = self['keys'].data[start_idx:end_idx]
            original_key = torch.tensor(original_key, dtype=torch.float)
            reduct_key = self.vector_reduct(original_key)
            new_keys.add(reduct_key.half())
            start_idx = end_idx
        None
        return new_keys

    @classmethod
    def load(cls, path, load_list, load_network):
        """
        load the datastore from the `path` folder

        Args:
            path(`str`):
                folder where the datastore files is stored
            load_list(`list`):
                specify the data name which we want to load
        Return:
            Datastore object(`Datastore`)
        """
        datas = {}
        config = read_config(path)
        for name in load_list:
            assert name in config['data_list'], "You haven't save {} but you list it in load_list".format(name)
            if os.path.exists(os.path.join(path, name + '.npy')):
                _info = config['data_infos'][name]
                datas[name] = Memmap(filename=os.path.join(path, name + '.npy'), shape=_info['shape'], dtype=_info['dtype'], mode='r+')
        dictionary_len = config['dictionary_len']
        if load_network:
            reduction_network_input_dim = config['reduction_network_input_dim']
            reduction_network_output_dim = config['reduction_network_output_dim']
            pck_datastore = cls(path, dictionary_len, reduction_network_input_dim, reduction_network_output_dim, datas)
            pck_datastore.load_state_dict(torch.load(os.path.join(path, 'reduct_network.pt')), strict=False)
        else:
            pck_datastore = cls(path, dictionary_len, None, None, datas)
        return pck_datastore

    def dump(self, verbose=True, dump_list=None, dump_network=False):
        """
        store the datastore files and config file to disk.
        
        Args:
            verbose: whether to display detailed infomation
            dump_list: specify the data names which you want to dump. if dump_list is None, dump all data
        """
        config = {}
        config['data_list'] = []
        config['data_infos'] = {}
        for name in self.datas.keys():
            config['data_list'].append(name)
            config['data_infos'][name] = {'name': name, 'shape': self.datas[name].shape, 'dtype': str(self.datas[name].dtype)}
            if dump_list is None or name in dump_list:
                self.datas[name].dump()
                if verbose:
                    None
        config['dictionary_len'] = self.dictionary_len
        if dump_network:
            config['reduction_network_input_dim'] = self.reduction_network_input_dim
            config['reduction_network_output_dim'] = self.reduction_network_output_dim
            torch.save(self.state_dict(), os.path.join(self.path, 'reduct_network.pt'))
        write_config(self.path, config)

    def set_target(self, x):
        self.tgt_ids = x

    def get_target(self):
        return self.tgt_ids


def retrieve_k_nearest(query, faiss_index, k):
    """
    use faiss to retrieve k nearest item
    """
    query_shape = list(query.size())
    distances, indices = faiss_index.search(query.detach().cpu().float().reshape(-1, query_shape[-1]).numpy(), k)
    distances = torch.tensor(distances, device=query.device).view(*query_shape[:-1], k)
    indices = torch.tensor(indices, device=query.device).view(*query_shape[:-1], k)
    return {'distances': distances, 'indices': indices}


class Retriever:

    def __init__(self, datastore, k):
        self.datastore = datastore
        self.k = k
        self.results = None

    def retrieve(self, query, return_list=['vals', 'distances'], k=None):
        """ 
        retrieve the datastore, save and return results 
        if parameter k is provided, it will suppress self.k
        """
        k = k if k is not None else self.k
        if not hasattr(self.datastore, 'faiss_index') or self.datastore.faiss_index is None or 'keys' not in self.datastore.faiss_index:
            self.datastore.load_faiss_index('keys', move_to_gpu=True)
        query = query.detach()
        faiss_results = retrieve_k_nearest(query, self.datastore.faiss_index['keys'], k)
        ret = {}
        if 'distances' in return_list:
            ret['distances'] = faiss_results['distances']
        if 'indices' in return_list:
            ret['indices'] = faiss_results['indices']
        if 'k' in return_list:
            ret['k'] = k
        if 'query' in return_list:
            ret['query'] = query
        indices = faiss_results['indices'].cpu().numpy()
        for data_name in return_list:
            if data_name not in ['distances', 'indices', 'k', 'query']:
                assert data_name in self.datastore.datas, 'You must load the {} of datastore first'.format(data_name)
                ret[data_name] = torch.tensor(self.datastore[data_name].data[indices], device=query.device)
        self.results = ret
        return ret


_global_vars = {}


def global_vars():
    return _global_vars


def select_keys_with_pad_mask(keys, mask):
    """
    use the mask to chose keys 

    Args:
        keys: (batch_sz, seq, dim)
        mask: (batch_sz, seq)
    
    Return: (*, dim)
    """
    mask_shape = mask.size()
    mask = mask.unsqueeze(-1).repeat(*([1] * len(mask_shape) + [keys.size(-1)]))
    return keys.masked_select(mask).view(-1, keys.size(-1))


class AdaptiveKNNMTDecoder(TransformerDecoder):
    """
    The adaptive knn-mt Decoder, equipped with Datastore, Retriever and AdaptiveCombiner.
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        """
        we override this function to create knn-related module.
        In other words, create datastore, retriever and combiner.
        """
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        if args.knn_mode == 'build_datastore':
            if 'datastore' not in global_vars():
                global_vars()['datastore'] = Datastore(args.knn_datastore_path)
            self.datastore = global_vars()['datastore']
        else:
            self.datastore = Datastore.load(args.knn_datastore_path, load_list=['vals'])
            self.datastore.load_faiss_index('keys')
            self.retriever = Retriever(datastore=self.datastore, k=args.knn_max_k)
            if args.knn_mode == 'train_metak':
                self.combiner = AdaptiveCombiner(max_k=args.knn_max_k, probability_dim=len(dictionary), k_trainable=args.knn_k_type == 'trainable', lambda_trainable=args.knn_lambda_type == 'trainable', lamda_=args.knn_lambda, temperature_trainable=args.knn_temperature_type == 'trainable', temperature=args.knn_temperature)
            elif args.knn_mode == 'inference':
                self.combiner = AdaptiveCombiner.load(args.knn_combiner_path)

    def forward(self, prev_output_tokens, encoder_out: 'Optional[EncoderOut]'=None, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]'=None, features_only: 'bool'=False, full_context_alignment: 'bool'=False, alignment_layer: 'Optional[int]'=None, alignment_heads: 'Optional[int]'=None, src_lengths: 'Optional[Any]'=None, return_all_hiddens: 'bool'=False):
        """
        we overwrite this function to do something else besides forward the TransformerDecoder.
        
        when the action mode is `building datastore`, we save keys to datastore.
        when the action mode is `inference`, we retrieve the datastore with hidden state.
        """
        x, extra = self.extract_features(prev_output_tokens, encoder_out=encoder_out, incremental_state=incremental_state, full_context_alignment=full_context_alignment, alignment_layer=alignment_layer, alignment_heads=alignment_heads)
        if self.args.knn_mode == 'build_datastore':
            keys = select_keys_with_pad_mask(x, self.datastore.get_pad_mask())
            self.datastore['keys'].add(keys.half())
        elif self.args.knn_mode == 'inference' or self.args.knn_mode == 'train_metak':
            self.retriever.retrieve(x, return_list=['vals', 'distances'])
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def get_normalized_probs(self, net_output: 'Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]]', log_probs: 'bool', sample: 'Optional[Dict[str, Tensor]]'=None):
        """
        we overwrite this function to change the probability calculation process.
        step 1. 
            calculate the knn probability based on retrieved resultes
        step 2.
            combine the knn probability with NMT's probability 
        """
        if self.args.knn_mode == 'inference' or self.args.knn_mode == 'train_metak':
            knn_prob = self.combiner.get_knn_prob(**self.retriever.results, device=net_output[0].device)
            combined_prob, _ = self.combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
            return combined_prob
        else:
            return super().get_normalized_probs(net_output, log_probs, sample)


def disable_model_grad(model):
    """ disable whole model's gradient """
    for name, param in model.named_parameters():
        param.requires_grad = False


def enable_module_grad(model, module_name):
    """ enable a module's gridient caclulation by module name"""
    for name, param in model.named_parameters():
        if module_name in name:
            param.requires_grad = True


class AdaptiveKNNMT(TransformerModel):
    """
    The adaptive knn-mt model.
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        if args.knn_mode == 'train_metak':
            disable_model_grad(self)
            enable_module_grad(self, 'combiner')

    @staticmethod
    def add_args(parser):
        """
        add knn-mt related args here
        """
        TransformerModel.add_args(parser)
        parser.add_argument('--knn-mode', choices=['build_datastore', 'train_metak', 'inference'], help='choose the action mode')
        parser.add_argument('--knn-datastore-path', type=str, metavar='STR', help='the directory of save or load datastore')
        parser.add_argument('--knn-max-k', type=int, metavar='N', default=8, help='The hyper-parameter max k of adaptive knn-mt')
        parser.add_argument('--knn-k-type', choices=['fixed', 'trainable'], default='trainable', help='trainable k or fixed k, if choose `fixed`, we use all theentries returned by retriever to calculate knn probs, i.e. directly use --knn-max-k as k')
        parser.add_argument('--knn-lambda-type', choices=['fixed', 'trainable'], default='trainable', help='trainable lambda or fixed lambda')
        parser.add_argument('--knn-lambda', type=float, default=0.7, help='if use a fixed lambda, provide it with --knn-lambda')
        parser.add_argument('--knn-temperature-type', choices=['fixed', 'trainable'], default='trainable', help='trainable temperature or fixed temperature')
        parser.add_argument('--knn-temperature', type=float, default=10, help='if use a fixed temperature, provide it with --knn-temperature')
        parser.add_argument('--knn-combiner-path', type=str, metavar='STR', default='/home/', help='The directory to save/load adaptiveCombiner')
        parser.add_argument('--build-faiss-index-with-cpu', action='store_true', default=False, help='use faiss-cpu instead of faiss-gpu (useful when gpu memory is small)')

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        """
        we override this function, replace the TransformerDecoder with AdaptiveKNNMTDecoder
        """
        return AdaptiveKNNMTDecoder(args, tgt_dict, embed_tokens, no_encoder_attn=getattr(args, 'no_cross_attention', False))


class GreedyMergeKNNMTEncoder(TransformerEncoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.args = args

    def forward(self, src_tokens, src_lengths, return_all_hiddens: 'bool'=False, token_embeddings: 'Optional[torch.Tensor]'=None):
        ret = super().forward(src_tokens, src_lengths, return_all_hiddens, token_embeddings)
        if self.args.enable_cache:
            global_vars()['new_batch_comes'] = True
        return ret


def calculate_knn_prob_with_merge_weight(vals, distances, merge_weights, probability_dim, temperature, device, **kwargs):
    """ 
    when the key-value pair has a merge weight.
    used by greedy-merge knn-mt
    """
    scaled_dists = -distances / temperature + torch.log(merge_weights.float())
    knn_weights = torch.softmax(scaled_dists, dim=-1)
    B, S, K = vals.size()
    knn_probs = torch.zeros(B, S, probability_dim, device=device)
    knn_probs.scatter_add_(dim=-1, index=vals, src=knn_weights)
    return knn_probs


class CacheCombiner:
    """
    Combiner use with CacheRetriever.
    """

    def __init__(self, lambda_, temperature, probability_dim):
        self.lambda_ = lambda_
        self.temperature = temperature
        self.probability_dim = probability_dim

    def get_knn_prob(self, cache, query, vals, distances, query_idx_which_use_cache, query_idx_which_use_datastore, cached_probs, original_query_shape, merge_weights=None, device='cuda:0', **kwargs):
        """ get knn probs.
        for those query which use cache, directly use cached probabilty.
        for those query which use datastore, calculate the probabilty with vals and distances.
        """
        assert query.size(0) == vals.size(0), 'Error'
        assert distances.size(0) == vals.size(0), 'Error'
        if merge_weights is not None:
            datastore_retrieved_probs = calculate_knn_prob_with_merge_weight(vals, distances, merge_weights, self.probability_dim, self.temperature, device)
        else:
            datastore_retrieved_probs = calculate_knn_prob(vals, distances, self.probability_dim, self.temperature, device)
        probabilities_shape = list(original_query_shape[:-1]) + [self.probability_dim]
        knn_probs = torch.zeros(*probabilities_shape, device=device)
        knn_probs = knn_probs.view(-1, self.probability_dim)
        if query_idx_which_use_cache.numel() > 0:
            knn_probs[query_idx_which_use_cache] = cached_probs
        if query_idx_which_use_datastore.numel() > 0:
            knn_probs[query_idx_which_use_datastore] = datastore_retrieved_probs
        knn_probs = knn_probs.view(*probabilities_shape)
        if query_idx_which_use_datastore.numel() > 0:
            if cache['queries'] is None:
                cache['queries'] = query
                cache['probs'] = datastore_retrieved_probs
            else:
                cache['queries'] = torch.cat((cache['queries'], query), dim=0)
                cache['probs'] = torch.cat((cache['probs'], datastore_retrieved_probs), dim=0)
        return knn_probs

    def get_combined_prob(self, knn_prob, neural_model_logit, log_probs=False):
        return calculate_combined_prob(knn_prob, neural_model_logit, self.lambda_, log_probs)


class CacheRetriever:

    def __init__(self, datastore, k):
        self.datastore = datastore
        self.k = k
        self.results = None
        self.cache = {'queries': None, 'probs': None}

    def retrieve(self, query, return_list=['keys', 'vals', 'distances'], cache_threshold=6.0):
        """
        retrieve the datastore and results with a cache. 
        note: for those queries which use cache, only return it's cached probs
        """
        if not hasattr(self.datastore, 'faiss_index') or self.datastore.faiss_index is None or 'keys' not in self.datastore.faiss_index:
            self.datastore.load_faiss_index('keys')
        ret = {}
        query = query.detach()
        ret['original_query_shape'] = query.size()
        ret['cache'] = self.cache
        query = query.view(-1, query.size(-1))
        if self.cache['queries'] is not None:
            distance_matrix = torch.cdist(query, self.cache['queries'], p=2)
            min_distance, min_indices = distance_matrix.min(dim=-1)
            mask = min_distance <= cache_threshold
            query_idx_which_use_cache = mask.nonzero(as_tuple=True)[0]
            cached_probs = self.cache['probs'][min_indices[query_idx_which_use_cache]]
            ret['query_idx_which_use_cache'] = query_idx_which_use_cache
            ret['cached_probs'] = cached_probs
            ret['query_idx_which_use_datastore'] = (~mask).nonzero(as_tuple=True)[0]
            query_using_datastore = query[ret['query_idx_which_use_datastore']]
        else:
            ret['query_idx_which_use_cache'] = torch.empty(0)
            ret['cached_probs'] = torch.empty(0)
            ret['query_idx_which_use_datastore'] = torch.arange(start=0, end=query.size(0), device=query.device)
            query_using_datastore = query
        query = query_using_datastore
        faiss_results = retrieve_k_nearest(query, self.datastore.faiss_index['keys'], self.k)
        if 'distances' in return_list:
            ret['distances'] = faiss_results['distances']
        if 'indices' in return_list:
            ret['indices'] = faiss_results['indices']
        if 'k' in return_list:
            ret['k'] = k
        if 'query' in return_list:
            ret['query'] = query
        indices = faiss_results['indices'].cpu().numpy()
        for data_name in return_list:
            if data_name not in ['distances', 'indices', 'k', 'query']:
                assert data_name in self.datastore.datas, 'You must load the `{}` of datastore first'.format(data_name)
                ret[data_name] = torch.tensor(self.datastore[data_name].data[indices], device=query.device)
        self.results = ret
        return ret

    def clear_cache(self):
        """ clear the cache """
        self.cache['queries'] = None
        self.cache['vals'] = None


class Combiner:
    """
    A simple Combiner used by vanilla knn-mt
    """

    def __init__(self, lambda_, temperature, probability_dim):
        self.lambda_ = lambda_
        self.temperature = temperature
        self.probability_dim = probability_dim

    def get_knn_prob(self, vals, distances, temperature=None, device='cuda:0', **kwargs):
        """
        calculate knn prob for vanilla knn-mt
        parameter temperature will suppress self.parameter
        """
        temperature = temperature if temperature is not None else self.temperature
        return calculate_knn_prob(vals, distances, self.probability_dim, temperature, device, **kwargs)

    def get_combined_prob(self, knn_prob, neural_model_logit, lambda_=None, log_probs=False):
        """ 
        strategy of combine probability of vanilla knn-mt
        If parameter `lambda_` is given, it will suppress the self.lambda_ 
        """
        lambda_ = lambda_ if lambda_ is not None else self.lambda_
        return calculate_combined_prob(knn_prob, neural_model_logit, lambda_, log_probs)


class GreedyMergeDatastore(Datastore):
    """
    implement greedy merge datastore
    """

    def prune(self, merge_neighbors=2, batch_size=4096, verbose=True):
        """
        prune the datastore using greedy merge strategy.
        """
        None
        start = time.time()
        neighbors = self._collect_neighbors(merge_neighbors, batch_size, verbose)
        weights = np.memmap(os.path.join(self.path, 'total_merge_weights.npy'), dtype=int, mode='w+', shape=(self['vals'].size,))
        weights[:] = 1
        random_order = list(range(self['vals'].size))
        random.shuffle(random_order)
        None
        start = time.time()
        with tqdm.tqdm(total=len(random_order)) as pbar:
            for i, id_ in enumerate(random_order):
                pbar.update(1)
                if weights[id_] <= 0:
                    continue
                for k, v in enumerate(neighbors[id_]):
                    if id_ != v and weights[v] == 1 and self['vals'].data[v] == self['vals'].data[id_]:
                        weights[v] = 0
                        weights[id_] += 1
        None
        pruned_datastore_size = int((weights > 0).sum())
        if verbose:
            None
        None
        start = time.time()
        self['new_keys'] = Memmap(os.path.join(self.path, 'new_keys.npy'), mode='w+')
        self['new_vals'] = Memmap(os.path.join(self.path, 'new_vals.npy'), mode='w+')
        self['merge_weights'] = Memmap(os.path.join(self.path, 'merge_weights.npy'), mode='w+')
        with tqdm.tqdm(total=weights.shape[0]) as pbar:
            for i, wgh in enumerate(weights):
                pbar.update(1)
                if wgh > 0:
                    self['new_keys'].add(self['keys'].data[i].reshape(1, -1))
                    self['new_vals'].add(self['vals'].data[i].reshape(1))
                    self['merge_weights'].add(wgh)
        self['keys'] = self['new_keys']
        self['vals'] = self['new_vals']
        del self['new_keys']
        del self['new_vals']
        del weights
        os.remove(os.path.join(self.path, 'keys.npy'))
        os.remove(os.path.join(self.path, 'vals.npy'))
        os.remove(os.path.join(self.path, 'total_merge_weights.npy'))
        os.remove(os.path.join(self.path, 'neighbors_' + str(merge_neighbors) + '.npy'))
        os.rename(os.path.join(self.path, 'new_keys.npy'), os.path.join(self.path, 'keys.npy'))
        os.rename(os.path.join(self.path, 'new_vals.npy'), os.path.join(self.path, 'vals.npy'))
        self['keys'].filename = os.path.join(self.path, 'keys.npy')
        self['vals'].filename = os.path.join(self.path, 'vals.npy')
        None
        None

    def _collect_neighbors(self, merge_neighbors=2, batch_size=4096, verbose=True):
        """
        collect the neighbors of original datastore's entry
        
        Args:
            merge_neighbors: merge how many neighbors
        """
        if not hasattr(self, 'faiss_index') or self.faiss_index is None:
            self.load_faiss_index('keys', verbose=False)
        self['keys'].drop_redundant()
        self['vals'].drop_redundant()
        neighbors = np.memmap(os.path.join(self.path, f'neighbors_{merge_neighbors}.npy'), dtype=np.int32, mode='w+', shape=(self['vals'].size, merge_neighbors + 1))
        if verbose:
            None
            start_time = time.time()
        batches = []
        cnt = 0
        offset = 0
        for i in tqdm.tqdm(range(0, self['vals'].size)):
            batches.append(self['keys'].data[i])
            cnt += 1
            if cnt % batch_size == 0 or i == self['vals'].size - 1:
                dists, knns = self.faiss_index['keys'].search(np.array(batches).astype(np.float32), merge_neighbors + 1)
                neighbors[offset:offset + knns.shape[0]] = knns
                cnt = 0
                batches = []
                offset += knns.shape[0]
        del self.faiss_index['keys']
        if verbose:
            None
        return neighbors


class MergeWeightCombiner(Combiner):
    """ 
    used by greedy merge knn-mt [when enable_cache=False, use_merge_weight=True]
    """

    def get_knn_prob(self, vals, distances, merge_weights, device='cuda:0', **kwargs):
        return calculate_knn_prob_with_merge_weight(vals, distances, merge_weights, self.probability_dim, self.temperature, device, **kwargs)


class GreedyMergeKNNMTDecoder(TransformerDecoder):
    """
    The greedy merge knn-mt Decoder, equipped with knn datastore, retriever and combiner.

    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        """
        we override this function to create knn-related module.
        In other words, create datastore, retriever and combiner.
        """
        TransformerDecoder.__init__(self, args, dictionary, embed_tokens, no_encoder_attn)
        if args.knn_mode == 'build_datastore':
            if 'datastore' not in global_vars():
                global_vars()['datastore'] = GreedyMergeDatastore(args.knn_datastore_path)
            self.datastore = global_vars()['datastore']
        elif args.knn_mode == 'inference':
            load_list = ['vals']
            if self.args.use_merge_weights:
                load_list.append('merge_weights')
            self.datastore = GreedyMergeDatastore.load(args.knn_datastore_path, load_list=load_list)
            self.datastore.load_faiss_index('keys', move_to_gpu=False)
            if args.enable_cache:
                self.retriever = CacheRetriever(datastore=self.datastore, k=args.knn_k)
                self.combiner = CacheCombiner(lambda_=args.knn_lambda, temperature=args.knn_temperature, probability_dim=len(dictionary))
            else:
                self.retriever = Retriever(datastore=self.datastore, k=args.knn_k)
                if args.use_merge_weights:
                    self.combiner = MergeWeightCombiner(lambda_=args.knn_lambda, temperature=args.knn_temperature, probability_dim=len(dictionary))
                else:
                    self.combiner = Combiner(lambda_=args.knn_lambda, temperature=args.knn_temperature, probability_dim=len(dictionary))

    def forward(self, prev_output_tokens, encoder_out: 'Optional[EncoderOut]'=None, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]'=None, features_only: 'bool'=False, full_context_alignment: 'bool'=False, alignment_layer: 'Optional[int]'=None, alignment_heads: 'Optional[int]'=None, src_lengths: 'Optional[Any]'=None, return_all_hiddens: 'bool'=False):
        """
        we overwrite this function to do something else besides forward the TransformerDecoder.
        
        when the action mode is `building datastore`, we save keys to datastore.
        when the action mode is `inference`, we retrieve the datastore with hidden state.
        """
        x, extra = self.extract_features(prev_output_tokens, encoder_out=encoder_out, incremental_state=incremental_state, full_context_alignment=full_context_alignment, alignment_layer=alignment_layer, alignment_heads=alignment_heads)
        if self.args.knn_mode == 'build_datastore':
            keys = select_keys_with_pad_mask(x, self.datastore.get_pad_mask())
            self.datastore['keys'].add(keys.half())
        elif self.args.knn_mode == 'inference':
            if self.args.enable_cache:
                if global_vars()['new_batch_comes']:
                    self.retriever.clear_cache()
                global_vars()['new_batch_comes'] = False
            return_list = ['vals', 'distances']
            if self.args.enable_cache:
                return_list.append('query')
            if self.args.use_merge_weights:
                return_list.append('merge_weights')
            if self.args.enable_cache:
                self.retriever.retrieve(x, return_list=return_list, cache_threshold=self.args.cache_threshold)
            else:
                self.retriever.retrieve(x, return_list=return_list)
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def get_normalized_probs(self, net_output: 'Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]]', log_probs: 'bool', sample: 'Optional[Dict[str, Tensor]]'=None):
        """
        we overwrite this function to change the probability calculation process.
        step 1. 
            calculate the knn probability based on retrieve resultes
        step 2.
            combine the knn probability with NMT's probability 
        """
        if self.args.knn_mode == 'inference':
            knn_prob = self.combiner.get_knn_prob(**self.retriever.results, use_merge_weights=self.args.use_merge_weights, device=net_output[0].device)
            combined_prob, _ = self.combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
            return combined_prob
        else:
            return super().get_normalized_probs(net_output, log_probs, sample)


class KernelSmoothedKNNMTDecoder(TransformerDecoder):
    """
    The adaptive knn-mt Decoder, equipped with Datastore, Retriever and AdaptiveCombiner.
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        """
        we override this function to create knn-related module.
        In other words, create datastore, retriever and combiner.
        """
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        if args.knn_mode == 'build_datastore':
            if 'datastore' not in global_vars():
                global_vars()['datastore'] = Datastore(args.knn_datastore_path)
            self.datastore = global_vars()['datastore']
        else:
            self.datastore = Datastore.load(args.knn_datastore_path, load_list=['keys', 'vals'])
            self.datastore.load_faiss_index('keys')
            self.retriever = Retriever(datastore=self.datastore, k=args.knn_k)
            if args.knn_mode == 'train_kster':
                self.combiner = KernelSmoothedCombiner(probability_dim=len(dictionary), query_dim=args.decoder_output_dim)
            elif args.knn_mode == 'inference':
                self.combiner = KernelSmoothedCombiner.load(args.knn_combiner_path)

    def forward(self, prev_output_tokens, encoder_out: 'Optional[EncoderOut]'=None, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]'=None, features_only: 'bool'=False, full_context_alignment: 'bool'=False, alignment_layer: 'Optional[int]'=None, alignment_heads: 'Optional[int]'=None, src_lengths: 'Optional[Any]'=None, return_all_hiddens: 'bool'=False):
        """
        we overwrite this function to do something else besides forward the TransformerDecoder.
        
        when the action mode is `building datastore`, we save keys to datastore.
        when the action mode is `inference`, we retrieve the datastore with hidden state.
        """
        x, extra = self.extract_features(prev_output_tokens, encoder_out=encoder_out, incremental_state=incremental_state, full_context_alignment=full_context_alignment, alignment_layer=alignment_layer, alignment_heads=alignment_heads)
        if self.args.knn_mode == 'build_datastore':
            keys = select_keys_with_pad_mask(x, self.datastore.get_pad_mask())
            self.datastore['keys'].add(keys.half())
        elif self.args.knn_mode == 'inference' or self.args.knn_mode == 'train_kster':
            self.retriever.retrieve(x, return_list=['vals', 'keys', 'query', 'distances'])
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def get_normalized_probs(self, net_output: 'Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]]', log_probs: 'bool', sample: 'Optional[Dict[str, Tensor]]'=None):
        """
        we overwrite this function to change the probability calculation process.
        step 1. 
            calculate the knn probability based on retrieved resultes
        step 2.
            combine the knn probability with NMT's probability 
        """
        if self.args.knn_mode == 'inference' or self.args.knn_mode == 'train_kster':
            knn_prob = self.combiner.get_knn_prob(**self.retriever.results, device=net_output[0].device)
            combined_prob, _ = self.combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
            return combined_prob
        else:
            return super().get_normalized_probs(net_output, log_probs, sample)


class KernelSmoothedKNNMT(TransformerModel):
    """
    The kernel smoothed knn-mt model.
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        if args.knn_mode == 'train_kster':
            disable_model_grad(self)
            enable_module_grad(self, 'combiner')

    @staticmethod
    def add_args(parser):
        """
        add knn-mt related args here
        """
        TransformerModel.add_args(parser)
        parser.add_argument('--knn-mode', choices=['build_datastore', 'train_kster', 'inference'], help='choose the action mode')
        parser.add_argument('--knn-datastore-path', type=str, metavar='STR', help='the directory of save or load datastore')
        parser.add_argument('--knn-k', type=int, metavar='N', default=8, help='The hyper-parameter k of adaptive knn-mt')
        parser.add_argument('--knn-combiner-path', type=str, metavar='STR', default='/home/', help='The directory to save/load KernelSmoothedCombiner')
        parser.add_argument('--build-faiss-index-with-cpu', action='store_true', default=False, help='use faiss-cpu instead of faiss-gpu (useful when gpu memory is small)')

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        """
        we override this function, replace the TransformerDecoder with AdaptiveKNNMTDecoder
        """
        return KernelSmoothedKNNMTDecoder(args, tgt_dict, embed_tokens, no_encoder_attn=getattr(args, 'no_cross_attention', False))


class PckKNNMTDecoder(TransformerDecoder):
    """
    The pck knn-mt Decoder, equipped with knn datastore, retriever and combiner.
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        """
        we override this function to create knn-related module.
        In other words, create datastore, retriever and combiner.
        """
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        if args.knn_mode == 'build_datastore':
            if 'datastore' not in global_vars():
                global_vars()['datastore'] = PckDatastore(path=args.knn_datastore_path, dictionary_len=len(self.dictionary))
            self.datastore = global_vars()['datastore']
        else:
            self.datastore = PckDatastore.load(args.knn_datastore_path, load_list=['vals'], load_network=True)
            self.datastore.load_faiss_index('keys')
            self.retriever = Retriever(datastore=self.datastore, k=args.knn_max_k)
            if args.knn_mode == 'train_metak':
                self.combiner = AdaptiveCombiner(max_k=args.knn_max_k, probability_dim=len(dictionary), k_trainable=args.knn_k_type == 'trainable', lambda_trainable=args.knn_lambda_type == 'trainable', lamda_=args.knn_lambda, temperature_trainable=args.knn_temperature_type == 'trainable', temperature=args.knn_temperature)
            elif args.knn_mode == 'inference':
                self.combiner = AdaptiveCombiner.load(args.knn_combiner_path)

    def forward(self, prev_output_tokens, encoder_out: 'Optional[EncoderOut]'=None, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]'=None, features_only: 'bool'=False, full_context_alignment: 'bool'=False, alignment_layer: 'Optional[int]'=None, alignment_heads: 'Optional[int]'=None, src_lengths: 'Optional[Any]'=None, return_all_hiddens: 'bool'=False):
        """
        we overwrite this function to do something else besides forward the TransformerDecoder.
        
        when the action mode is `building datastore`, we save keys to datastore.
        when the action mode is `inference`, we retrieve the datastore with hidden state.
        """
        x, extra = self.extract_features(prev_output_tokens, encoder_out=encoder_out, incremental_state=incremental_state, full_context_alignment=full_context_alignment, alignment_layer=alignment_layer, alignment_heads=alignment_heads)
        if self.args.knn_mode == 'build_datastore':

            def get_4_gram(target):
                """
                Args:
                    target: [B, T]
                Return: [B, T, 4]
                """
                batch_size = target.size(0)
                target = target[:, :, None]
                target_pad_1 = torch.cat((torch.zeros((batch_size, 1, 1), device=x.device, dtype=torch.long), target[:, :-1]), 1)
                target_pad_2 = torch.cat((torch.zeros((batch_size, 2, 1), device=x.device, dtype=torch.long), target[:, :-2]), 1)
                target_pad_3 = torch.cat((torch.zeros((batch_size, 3, 1), device=x.device, dtype=torch.long), target[:, :-3]), 1)
                return torch.cat((target, target_pad_1, target_pad_2, target_pad_3), -1)

            def get_tgt_probs(probs, target):
                """ 
                Args:
                    probs: [B, T, dictionary]
                    target: [B, T]
                Return: [B, T]
                """
                B, T, C = probs.size(0), probs.size(1), probs.size(2)
                one_hot = torch.arange(0, C)[None, None].repeat(B, T, 1) == target[:, :, None]
                return (probs * one_hot.float()).sum(-1)

            def get_4_gram_probs(target_prob):
                """
                Args:
                    target_prob: [B, T]
                Return: [B, T, 4]
                """
                target_prob = target_prob[:, :, None]
                target_pad_1 = torch.cat((target_prob[:, :1].repeat(1, 1, 1), target_prob[:, :-1]), 1)
                target_pad_2 = torch.cat((target_prob[:, :1].repeat(1, 2, 1), target_prob[:, :-2]), 1)
                target_pad_3 = torch.cat((target_prob[:, :1].repeat(1, 3, 1), target_prob[:, :-3]), 1)
                return torch.cat((target_prob, target_pad_1, target_pad_2, target_pad_3), -1)

            def get_entropy(probs):
                """probs: [B, T, dictionary]"""
                return -(probs * torch.log(probs + 1e-07)).sum(-1)
            output_logit = self.output_layer(x)
            output_probs = F.softmax(output_logit, dim=-1)
            target = self.datastore.get_target()
            ids_4_gram = get_4_gram(target)
            target_prob = get_tgt_probs(output_probs, target)
            probs_4_gram = get_4_gram_probs(target_prob)
            entropy = get_entropy(output_probs)
            pad_mask = self.datastore.get_pad_mask()
            keys = select_keys_with_pad_mask(x, pad_mask)
            ids_4_gram = select_keys_with_pad_mask(ids_4_gram, pad_mask)
            probs_4_gram = select_keys_with_pad_mask(probs_4_gram, pad_mask)
            entropy = entropy.masked_select(pad_mask)
            self.datastore['keys'].add(keys.half())
            self.datastore['ids_4_gram'].add(ids_4_gram)
            self.datastore['probs_4_gram'].add(probs_4_gram)
            self.datastore['entropy'].add(entropy)
        elif self.args.knn_mode == 'train_metak' or self.args.knn_mode == 'inference':
            self.retriever.retrieve(self.datastore.vector_reduct(x), return_list=['vals', 'distances'])
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def get_normalized_probs(self, net_output: 'Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]]', log_probs: 'bool', sample: 'Optional[Dict[str, Tensor]]'=None):
        """
        we overwrite this function to change the probability calculation process.
        step 1.
            calculate the knn probability based on retrieve resultes
        step 2.
            combine the knn probability with NMT's probability 
        """
        if self.args.knn_mode == 'inference' or self.args.knn_mode == 'train_metak':
            knn_prob = self.combiner.get_knn_prob(**self.retriever.results, device=net_output[0].device)
            combined_prob, _ = self.combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
            return combined_prob
        else:
            return super().get_normalized_probs(net_output, log_probs, sample)


class PckKNNMT(AdaptiveKNNMT):
    """
    The  pck knn-mt model.
    """

    @staticmethod
    def add_args(parser):
        """
        add pck knn-mt related args here
        """
        AdaptiveKNNMT.add_args(parser)
        parser.add_argument('--knn-reduct-dim', type=int, metavar='N', default=64, help='reducted dimension of datastore')

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        """
        we override this function, replace the TransformerDecoder with PckKNNMTDecoder
        """
        return PckKNNMTDecoder(args, tgt_dict, embed_tokens, no_encoder_attn=getattr(args, 'no_cross_attention', False))


class PlacKNNMTDecoder(TransformerDecoder):
    """
    The plac knn-mt Decoder, equipped with knn datastore, retriever and combiner.
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        """
        we override this function to create knn-related module.
        In other words, create datastore, retriever and combiner.
        """
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        assert args.knn_mode == 'save_mt_pred', 'PLAC Model can only be used to save mt model predictions'
        if 'datastore' not in global_vars():
            global_vars()['datastore'] = Datastore(args.knn_datastore_path)
        self.datastore = global_vars()['datastore']

    def forward(self, prev_output_tokens, encoder_out: 'Optional[EncoderOut]'=None, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]'=None, features_only: 'bool'=False, full_context_alignment: 'bool'=False, alignment_layer: 'Optional[int]'=None, alignment_heads: 'Optional[int]'=None, src_lengths: 'Optional[Any]'=None, return_all_hiddens: 'bool'=False):
        """
        we overwrite this function to do something else besides forward the TransformerDecoder.
        
        when the action mode is `building datastore`, we save keys to datastore.
        when the action mode is `inference`, we retrieve the datastore with hidden state.
        """
        x, extra = self.extract_features(prev_output_tokens, encoder_out=encoder_out, incremental_state=incremental_state, full_context_alignment=full_context_alignment, alignment_layer=alignment_layer, alignment_heads=alignment_heads)
        assert self.args.knn_mode == 'save_mt_pred', 'PLAC Model can only be used to save mt model predictions'
        keys = select_keys_with_pad_mask(x, self.datastore.get_pad_mask())
        self.datastore['keys'].add(keys.half())
        if not features_only:
            x = self.output_layer(x)
        mt_preds = x.argmax(dim=-1, keepdim=True)
        mt_preds = select_keys_with_pad_mask(mt_preds, self.datastore.get_pad_mask())
        self.datastore['mt_preds'].add(mt_preds.squeeze(-1))
        return x, extra


class PlacKNNMT(TransformerModel):
    """
    The plac knn-mt model.
    """

    @staticmethod
    def add_args(parser):
        """
        add knn-mt related args here
        """
        TransformerModel.add_args(parser)
        parser.add_argument('--knn-mode', choices=['save_mt_pred'], help='choose the action mode')
        parser.add_argument('--knn-datastore-path', type=str, metavar='STR', help='the directory of save or load datastore')
        parser.add_argument('--build-faiss-index-with-cpu', action='store_true', default=False, help='use faiss-cpu instead of faiss-gpu (useful when gpu memory is small)')

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        """
        we override this function, replace the TransformerDecoder with PlacKNNMTDecoder
        """
        return PlacKNNMTDecoder(args, tgt_dict, embed_tokens, no_encoder_attn=getattr(args, 'no_cross_attention', False))


class RobustKNNMTDecoder(TransformerDecoder):
    """
    The robust knn-mt Decoder, equipped with Datastore, Retriever and RobustCombiner.
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        """
        we override this function to create knn-related module.
        In other words, create datastore, retriever and combiner.
        """
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.update_num = 0
        if args.knn_mode == 'build_datastore':
            if 'datastore' not in global_vars():
                global_vars()['datastore'] = Datastore(args.knn_datastore_path)
            self.datastore = global_vars()['datastore']
        else:
            self.datastore = Datastore.load(args.knn_datastore_path, load_list=['keys', 'vals'])
            self.datastore.load_faiss_index('keys')
            self.retriever = Retriever(datastore=self.datastore, k=args.knn_max_k)
            if args.knn_mode == 'train_metak':
                self.combiner = RobustCombiner(max_k=args.knn_max_k, midsize=args.robust_wp_hidden_size, midsize_dc=args.robust_dc_hidden_size, topk_wp=args.robust_wp_topk, probability_dim=len(dictionary))
            elif args.knn_mode == 'inference':
                self.combiner = RobustCombiner.load(args.knn_combiner_path)

    def forward(self, prev_output_tokens, encoder_out: 'Optional[EncoderOut]'=None, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]'=None, features_only: 'bool'=False, full_context_alignment: 'bool'=False, alignment_layer: 'Optional[int]'=None, alignment_heads: 'Optional[int]'=None, src_lengths: 'Optional[Any]'=None, return_all_hiddens: 'bool'=False, target: 'Optional[Tensor]'=None):
        """
        we overwrite this function to do something else besides forward the TransformerDecoder.
        
        when the action mode is `building datastore`, we save keys to datastore.
        when the action mode is `inference`, we retrieve the datastore with hidden state.
        """
        x, extra = self.extract_features(prev_output_tokens, encoder_out=encoder_out, incremental_state=incremental_state, full_context_alignment=full_context_alignment, alignment_layer=alignment_layer, alignment_heads=alignment_heads)
        if self.args.knn_mode == 'build_datastore':
            keys = select_keys_with_pad_mask(x, self.datastore.get_pad_mask())
            self.datastore['keys'].add(keys.half())
        elif self.args.knn_mode == 'inference' or self.args.knn_mode == 'train_metak':
            self.retriever.retrieve(x, return_list=['vals', 'query', 'distances', 'keys'])
        extra.update({'last_hidden': x, 'target': target})
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def get_normalized_probs(self, net_output: 'Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]]', log_probs: 'bool', sample: 'Optional[Dict[str, Tensor]]'=None):
        """
        we overwrite this function to change the probability calculation process.
        step 1. 
            calculate the knn probability based on retrieved resultes
        step 2.
            combine the knn probability with NMT's probability 
        """
        if self.args.knn_mode == 'inference' or self.args.knn_mode == 'train_metak':
            network_probs = utils.softmax(net_output[0], dim=-1, onnx_trace=self.onnx_trace)
            knn_dists = self.retriever.results['distances']
            tgt_index = self.retriever.results['vals']
            knn_key = self.retriever.results['keys']
            queries = self.retriever.results['query']
            knn_dists = torch.sum((knn_key - queries.unsqueeze(-2).detach()) ** 2, dim=-1)
            knn_dists, new_index = torch.sort(knn_dists, dim=-1)
            tgt_index = tgt_index.gather(dim=-1, index=new_index)
            knn_key = knn_key.gather(dim=-2, index=new_index.unsqueeze(-1).expand(knn_key.shape))
            B, S, K = knn_dists.size()
            network_select_probs = network_probs.gather(index=tgt_index, dim=-1)
            if self.training:
                target = net_output[1]['target']
                last_hidden = net_output[1]['last_hidden']
                random_rate = self.args.robust_training_alpha0
                noise_var = self.args.robust_training_sigma
                e = self.args.robust_training_beta
                random_rate = random_rate * math.exp(-self.update_num / e)
                noise_mask = (tgt_index == target.unsqueeze(-1)).any(-1, True)
                rand_mask = ((torch.rand(B, S, 1) < random_rate) & (target.unsqueeze(-1) != 1)).long()
                rand_mask2 = ((torch.rand(B, S, 1) < random_rate) & (target.unsqueeze(-1) != 1) & ~noise_mask).float()
                with torch.no_grad():
                    knn_key = knn_key + torch.randn_like(knn_key) * rand_mask.unsqueeze(-1) * noise_var
                    new_key = last_hidden + torch.randn_like(last_hidden) * noise_var
                    noise_knn_key = torch.cat([new_key.unsqueeze(-2), knn_key.float()[:, :, :-1, :]], -2)
                    noise_tgt_index = torch.cat([target.unsqueeze(-1), tgt_index[:, :, :-1]], -1)
                    tgt_index = noise_tgt_index * rand_mask2.long() + tgt_index * (1 - rand_mask2.long())
                    knn_key = noise_knn_key * rand_mask2.unsqueeze(-1) + knn_key * (1 - rand_mask2.unsqueeze(-1))
                    knn_probs = utils.softmax(self.output_layer(knn_key.float()), dim=-1, onnx_trace=self.onnx_trace)
                    knn_key_feature = knn_probs.gather(-1, index=tgt_index.unsqueeze(-1)).squeeze(-1)
                    noise_knn_dists = torch.sum((knn_key - last_hidden.unsqueeze(-2).detach()) ** 2, dim=3)
                    dup_knn_dists = noise_knn_dists
                    new_dists, dist_index = torch.sort(dup_knn_dists, dim=-1)
                    new_index = dist_index
                    knn_dists = new_dists
                    tgt_index = tgt_index.gather(-1, new_index)
                    network_select_probs = network_probs.gather(index=tgt_index, dim=-1)
                    knn_key_feature = knn_key_feature.gather(-1, new_index)
            else:
                knn_probs = utils.softmax(self.output_layer(knn_key.float()), dim=-1, onnx_trace=self.onnx_trace)
                knn_key_feature = knn_probs.gather(-1, index=tgt_index.unsqueeze(-1)).squeeze(-1)
            knn_prob = self.combiner.get_knn_prob(tgt_index=tgt_index, knn_dists=knn_dists, knn_key_feature=knn_key_feature, network_probs=network_probs, network_select_probs=network_select_probs, device=net_output[0].device)
            combined_prob, _ = self.combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
            return combined_prob
        else:
            return super().get_normalized_probs(net_output, log_probs, sample)

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self.update_num = num_updates


class RobustKNNMT(TransformerModel):
    """
    The robust knn-mt model.
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        if args.knn_mode == 'train_metak':
            disable_model_grad(self)
            enable_module_grad(self, 'combiner')

    @staticmethod
    def add_args(parser):
        """
        add knn-mt related args here
        """
        TransformerModel.add_args(parser)
        parser.add_argument('--knn-mode', choices=['build_datastore', 'train_metak', 'inference'], help='choose the action mode')
        parser.add_argument('--knn-datastore-path', type=str, metavar='STR', help='the directory of save or load datastore')
        parser.add_argument('--knn-max-k', type=int, metavar='N', default=8, help='The hyper-parameter max k of robust knn-mt')
        parser.add_argument('--knn-combiner-path', type=str, metavar='STR', default='/home/', help='The directory to save/load robustCombiner')
        parser.add_argument('--build-faiss-index-with-cpu', action='store_true', default=False, help='use faiss-cpu instead of faiss-gpu (useful when gpu memory is small)')
        parser.add_argument('--robust-training-sigma', type=float, default=0.01, help='the noise vector is sampled from a Gaussian distribution with variance sigma^2')
        parser.add_argument('--robust-training-alpha0', type=float, default=1.0, help='alpha0 control the initial value of the perturbation ratio (alpha)')
        parser.add_argument('--robust-training-beta', type=int, default=1000, help='beta control the declining speed of the perturbation ratio (alpha)')
        parser.add_argument('--robust-dc-hidden-size', type=int, default=4, help='the hidden size of DC network')
        parser.add_argument('--robust-wp-hidden-size', type=int, default=32, help='the hidden size of WP network')
        parser.add_argument('--robust-wp-topk', type=int, default=8, help='WP network uses the k highest probabilities of the NMT distribution as input')

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        """
        we override this function, replace the TransformerDecoder with RobustKNNMTDecoder
        """
        return RobustKNNMTDecoder(args, tgt_dict, embed_tokens, no_encoder_attn=getattr(args, 'no_cross_attention', False))

    def forward(self, src_tokens, src_lengths, prev_output_tokens, return_all_hiddens: 'bool'=True, features_only: 'bool'=False, alignment_layer: 'Optional[int]'=None, alignment_heads: 'Optional[int]'=None, target: 'Optional[Tensor]'=None):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, features_only=features_only, alignment_layer=alignment_layer, alignment_heads=alignment_heads, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens, target=target)
        return decoder_out


class LabelSmoothedCrossEntropyCriterionForRobust(LabelSmoothedCrossEntropyCriterion):

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'], target=sample['target'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {'loss': loss.data, 'nll_loss': nll_loss.data, 'ntokens': sample['ntokens'], 'nsentences': sample['target'].size(0), 'sample_size': sample_size}
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output['n_correct'] = utils.item(n_correct.data)
            logging_output['total'] = utils.item(total.data)
        return loss, sample_size, logging_output


class SimpleScalableCombiner:
    """
    A Combiner used by simple and scalable knn-mt
    """

    def __init__(self, temperature, probability_dim):
        self.lambda_ = None
        self.temperature = temperature
        self.probability_dim = probability_dim

    def get_knn_prob(self, vals, distances, temperature=None, device='cuda:0', **kwargs):
        """
        calculate knn prob for vanilla knn-mt
        parameter temperature will suppress self.parameter
        """
        temperature = temperature if temperature is not None else self.temperature
        scaled_dists = -distances / temperature
        min_distance, _ = distances.min(dim=-1, keepdim=True)
        self.lambda_ = torch.nn.functional.relu(1 - min_distance / self.temperature)
        knn_weights = torch.softmax(scaled_dists, dim=-1)
        B, S, K = vals.size()
        knn_probs = torch.zeros(B, S, self.probability_dim, device=device)
        knn_probs.scatter_add_(dim=-1, index=vals, src=knn_weights)
        return knn_probs

    def get_combined_prob(self, knn_prob, neural_model_logit, lambda_=None, log_probs=False):
        """ 
        strategy of combine probability of vanilla knn-mt
        If parameter `lambda_` is given, it will suppress the self.lambda_ 
        """
        lambda_ = lambda_ if lambda_ is not None else self.lambda_
        return calculate_combined_prob(knn_prob, neural_model_logit, lambda_, log_probs)


class SimpleScalableRetriever:

    def __init__(self, k):
        self.k = k
        self.results = None

    def retrieve(self, query, encoder_out_hash, return_list=['vals', 'distances'], k=None):
        """ retrieve from dynamic datastore
        Args:
            query: [batch, 1, dim]
            encoder_out_hash: [batch]
        """
        k = self.k if k is None else k
        ret = {}
        batch_size, seq_len, dim = query.shape
        if 'keys' in return_list:
            ret['keys'] = torch.empty(batch_size, seq_len, k, dim, dtype=query.dtype, device=query.device)
        if 'vals' in return_list:
            ret['vals'] = torch.empty(batch_size, seq_len, k, dtype=torch.int64, device=query.device)
        if 'distances' in return_list:
            ret['distances'] = torch.empty(batch_size, seq_len, k, dtype=torch.float, device=query.device)
        for sentence_idx, sentence in enumerate(query):
            for pos, q in enumerate(sentence):
                try:
                    dynamic_datastore = global_vars()['encoderout_to_kv'][float(encoder_out_hash[sentence_idx])]
                except:
                    None
                    nearest_dis = 100000.0
                    nearest_key = ''
                    for h in global_vars()['encoderout_to_kv'].keys():
                        if abs(float(encoder_out_hash[sentence_idx]) - h) < nearest_dis:
                            nearest_dis = abs(float(encoder_out_hash[sentence_idx]) - h)
                            nearest_key = h
                    dynamic_datastore = global_vars()['encoderout_to_kv'][nearest_key]
                if dynamic_datastore is not None:
                    distances = torch.cdist(q.unsqueeze(0), dynamic_datastore['keys'])
                    distances = distances.squeeze(0)
                    sorted_distances, indices = torch.sort(distances)
                    if indices.shape[0] < k:
                        None
                        sorted_distances = torch.cat((sorted_distances, 100000 * torch.ones(k - sorted_distances.shape[0], dtype=torch.float, device=query.device)))
                        indices = torch.cat((indices, torch.zeros(k - indices.shape[0], dtype=torch.int64, device=query.device)))
                    if 'keys' in return_list:
                        ret['keys'][sentence_idx][pos] = dynamic_datastore['keys'][indices[:k]]
                    if 'vals' in return_list:
                        ret['vals'][sentence_idx][pos] = dynamic_datastore['vals'][indices[:k]]
                    if 'distances' in return_list:
                        ret['distances'][sentence_idx][pos] = torch.square(sorted_distances[:k])
                else:
                    if 'keys' in return_list:
                        ret['keys'][sentence_idx][pos] = torch.zeros(k, dim, dtype=query.dtype, device=query.device)
                    if 'vals' in return_list:
                        ret['vals'][sentence_idx][pos] = torch.zeros(k, dtype=torch.int64, device=query.device)
                    if 'distances' in return_list:
                        ret['distances'][sentence_idx][pos] = 100000 * torch.ones(k, dtype=torch.float, device=query.device)
        self.results = ret
        return ret


def vector_hash_func(x):
    """ Hash torch.tensor
    Args:
        x: [batch, seq_len, dim]
    Return:
        str[batch]
    """
    return np.around(x[:, 0, 0].cpu().numpy(), decimals=6)


class SimpleScalableKNNMTDecoder(TransformerDecoder):
    """
    The simple and scalable knn-mt Decoder """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        """
        we override this function to create knn-related module.
        In other words, create datastore, retriever and combiner.
        """
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        if args.knn_mode == 'inference':
            self.retriever = SimpleScalableRetriever(k=args.knn_k)
            self.combiner = SimpleScalableCombiner(temperature=args.knn_temperature, probability_dim=len(dictionary))

    def forward(self, prev_output_tokens, encoder_out: 'Optional[EncoderOut]'=None, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]'=None, features_only: 'bool'=False, full_context_alignment: 'bool'=False, alignment_layer: 'Optional[int]'=None, alignment_heads: 'Optional[int]'=None, src_lengths: 'Optional[Any]'=None, return_all_hiddens: 'bool'=False):
        """
        we overwrite this function to do something else besides forward the TransformerDecoder.
        
        when the action mode is `inference`, we retrieve the elasticsearch database
        with source tokens.
        """
        x, extra = self.extract_features(prev_output_tokens, encoder_out=encoder_out, incremental_state=incremental_state, full_context_alignment=full_context_alignment, alignment_layer=alignment_layer, alignment_heads=alignment_heads)
        if self.args.knn_mode == 'inference':
            encoder_out_hash = vector_hash_func(encoder_out[0].transpose(0, 1))
            self.retriever.retrieve(query=x, encoder_out_hash=encoder_out_hash, return_list=['vals', 'distances'])
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def get_normalized_probs(self, net_output: 'Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]]', log_probs: 'bool', sample: 'Optional[Dict[str, Tensor]]'=None):
        """
        we overwrite this function to change the probability calculation process.
        step 1. 
            calculate the knn probability based on retrieve resultes
        step 2.
            combine the knn probability with NMT's probability 
        """
        if self.args.knn_mode == 'inference':
            knn_prob = self.combiner.get_knn_prob(**self.retriever.results, device=net_output[0].device)
            combined_prob, _ = self.combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
            return combined_prob
        else:
            return super().get_normalized_probs(net_output, log_probs, sample)


def collate(source, target, pad_idx=1, eos_idx=2, left_pad_source=True, left_pad_target=False):
    """collate tokens to get a mini-batch
    Args:
        source: list of torch.tensor
        target: list of torch.tensor
    """

    def merge(tokens, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(tokens, pad_idx, eos_idx, left_pad, move_eos_to_beginning=move_eos_to_beginning, pad_to_length=None, pad_to_multiple=1)
    src_tokens = merge(tokens=source, left_pad=left_pad_source, move_eos_to_beginning=False)
    src_lengths = torch.LongTensor([s.ne(pad_idx).long().sum() for s in source])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    src_tokens = src_tokens.index_select(0, sort_order)
    tgt_tokens = merge(tokens=target, left_pad=False, move_eos_to_beginning=False)
    tgt_tokens = tgt_tokens.index_select(0, sort_order)
    prev_output_tokens = merge(tokens=target, left_pad=False, move_eos_to_beginning=True)
    prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    return {'src_tokens': src_tokens, 'src_lengths': src_lengths, 'tgt_tokens': tgt_tokens, 'prev_output_tokens': prev_output_tokens, 'sort_order': sort_order}


def filter_pad_tokens(tokens, pad_idx=1):
    """
    given a int tensor, 
    return all no pad element and the mask,
    1 represent no-pad, 0 represent pad
    """
    mask = tokens.ne(pad_idx)
    tokens = tokens.masked_select(mask)
    return tokens, mask


class SimpleScalableKNNMTEncoder(TransformerEncoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.args = args

    def forward(self, src_tokens, src_lengths, return_all_hiddens: 'bool'=False, token_embeddings: 'Optional[torch.Tensor]'=None):
        encoder_out = super().forward(src_tokens, src_lengths, return_all_hiddens, token_embeddings)
        if self.args.knn_mode == 'inference':
            with torch.no_grad():
                self._build_dynamic_datastore(encoder_out, src_tokens)
        return encoder_out

    def _build_dynamic_datastore(self, encoder_out, src_tokens):
        """ build dynamic datastore when forwarding encoder
        Args: 
            encoder_out
            src_tokens
        Return: mapping from encoder_out_hash to (key vector, val vector)
        """
        es = elasticsearch.Elasticsearch(['http://localhost:' + str(self.args.elastic_port)], request_timeout=3600)
        tracer = logging.getLogger('elasticsearch')
        tracer.setLevel(logging.CRITICAL)
        requests = []
        for src in src_tokens:
            src = [str(token_id) for token_id in src.cpu().numpy().tolist() if token_id != 1 and token_id != 2]
            src = ' '.join(src)
            req_head = {'index': self.args.elastic_index_name}
            req_body = {'query': {'match': {'source_tokens': src}}, 'from': 0, 'size': 64}
            requests.extend([req_head, req_body])
        resp = es.msearch(body=requests)['responses']
        retrieve_results = [None] * len(src_tokens)
        for i in range(len(src_tokens)):
            hits = resp[i]['hits']['hits']
            retrieve_results[i] = [[], []]
            for sentence in hits:
                retrieve_results[i][0].append(sentence['_source']['source_tokens'])
                retrieve_results[i][1].append(sentence['_source']['target_tokens'])
        for i in range(len(src_tokens)):
            similarities = []
            for retrieve_src in retrieve_results[i][0]:
                retrieve_src_list = [int(num) for num in retrieve_src.split()]
                edit_distance = editdistance.eval([tok for tok in src_tokens[i].cpu().numpy().tolist()[:-1] if tok != 1 and tok != 2], retrieve_src_list)
                similarities.append(1.0 - float(edit_distance) / max(len(src_tokens[i]) - 1, len(retrieve_src_list)))
            indices = np.argsort(-np.array(similarities))
            retrieve_results[i][0] = [retrieve_results[i][0][idx] for idx in indices[:self.args.reserve_top_m]]
            retrieve_results[i][1] = [retrieve_results[i][1][idx] for idx in indices[:self.args.reserve_top_m]]
        encoder_out_hash = vector_hash_func(encoder_out[0].transpose(0, 1))
        global_vars()['encoderout_to_kv'] = {}
        for i in range(len(src_tokens)):
            if len(retrieve_results[i][0]) == 0:
                global_vars()['encoderout_to_kv'][float(encoder_out_hash[i])] = None
                None
                None
                continue
            source = [torch.LongTensor([int(token) for token in s.split()] + [self.dictionary.eos()]) for s in retrieve_results[i][0]]
            target = [torch.LongTensor([int(token) for token in s.split()] + [self.dictionary.eos()]) for s in retrieve_results[i][1]]
            batch = collate(source, target, pad_idx=self.dictionary.pad(), eos_idx=self.dictionary.eos(), left_pad_source=self.args.left_pad_source, left_pad_target=self.args.left_pad_target)
            if torch.cuda.is_available() and not self.args.cpu:
                batch = utils.move_to_cuda(batch)
            global_vars()['sk_mt_model'].args.knn_mode = '-'
            model_decoder_out = global_vars()['sk_mt_model'](batch['src_tokens'], batch['src_lengths'], batch['prev_output_tokens'], return_all_hiddens=False, features_only=True)[0]
            global_vars()['sk_mt_model'].args.knn_mode = 'inference'
            non_pad_tokens, mask = filter_pad_tokens(batch['tgt_tokens'])
            keys = select_keys_with_pad_mask(model_decoder_out, mask)
            global_vars()['encoderout_to_kv'][float(encoder_out_hash[i])] = {'keys': keys, 'vals': non_pad_tokens}


class SimpleScalableKNNMT(TransformerModel):
    """
    The simple and scalable knn-mt model.
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        if 'sk_mt_model' not in global_vars():
            global_vars()['sk_mt_model'] = self

    @staticmethod
    def add_args(parser):
        """
        add knn-mt related args here
        """
        TransformerModel.add_args(parser)
        parser.add_argument('--knn-mode', choices=['inference'], help='choose the action mode')
        parser.add_argument('--knn-k', type=int, metavar='N', default=2, help='The hyper-parameter k of  knn-mt')
        parser.add_argument('--knn-temperature', type=float, metavar='D', default=100, help='The hyper-parameter temperature of  knn-mt')
        parser.add_argument('--reserve-top-m', type=int, default=16, help='reserve top-m retrieve results')
        parser.add_argument('--elastic-index-name', type=str, help='The elasticsearch                             index name which to retrieve from')
        parser.add_argument('--elastic-port', type=int, default=9200, help='The port of elasticsearch service.')

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        """
        we override this function, replace the TransformerDecoder with SimpleScalableKNNMTEncoder
        """
        return SimpleScalableKNNMTEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        """
        we override this function, replace the TransformerDecoder with SimpleScalableKNNMTDecoder
        """
        return SimpleScalableKNNMTDecoder(args, tgt_dict, embed_tokens, no_encoder_attn=getattr(args, 'no_cross_attention', False))


class VanillaKNNMTDecoder(TransformerDecoder):
    """
    The vanilla knn-mt Decoder, equipped with knn datastore, retriever and combiner.
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        """
        we override this function to create knn-related module.
        In other words, create datastore, retriever and combiner.
        """
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        if args.knn_mode == 'build_datastore':
            if 'datastore' not in global_vars():
                global_vars()['datastore'] = Datastore(args.knn_datastore_path)
            self.datastore = global_vars()['datastore']
        elif args.knn_mode == 'inference':
            self.datastore = Datastore.load(args.knn_datastore_path, load_list=['vals'])
            self.datastore.load_faiss_index('keys')
            self.retriever = Retriever(datastore=self.datastore, k=args.knn_k)
            self.combiner = Combiner(lambda_=args.knn_lambda, temperature=args.knn_temperature, probability_dim=len(dictionary))

    def forward(self, prev_output_tokens, encoder_out: 'Optional[EncoderOut]'=None, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]'=None, features_only: 'bool'=False, full_context_alignment: 'bool'=False, alignment_layer: 'Optional[int]'=None, alignment_heads: 'Optional[int]'=None, src_lengths: 'Optional[Any]'=None, return_all_hiddens: 'bool'=False):
        """
        we overwrite this function to do something else besides forward the TransformerDecoder.
        
        when the action mode is `building datastore`, we save keys to datastore.
        when the action mode is `inference`, we retrieve the datastore with hidden state.
        """
        x, extra = self.extract_features(prev_output_tokens, encoder_out=encoder_out, incremental_state=incremental_state, full_context_alignment=full_context_alignment, alignment_layer=alignment_layer, alignment_heads=alignment_heads)
        if self.args.knn_mode == 'build_datastore':
            keys = select_keys_with_pad_mask(x, self.datastore.get_pad_mask())
            self.datastore['keys'].add(keys.half())
        elif self.args.knn_mode == 'inference':
            self.retriever.retrieve(x, return_list=['vals', 'distances'])
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def get_normalized_probs(self, net_output: 'Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]]', log_probs: 'bool', sample: 'Optional[Dict[str, Tensor]]'=None):
        """
        we overwrite this function to change the probability calculation process.
        step 1. 
            calculate the knn probability based on retrieve resultes
        step 2.
            combine the knn probability with NMT's probability 
        """
        if self.args.knn_mode == 'inference':
            knn_prob = self.combiner.get_knn_prob(**self.retriever.results, device=net_output[0].device)
            combined_prob, _ = self.combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
            return combined_prob
        else:
            return super().get_normalized_probs(net_output, log_probs, sample)


class VanillaKNNMT(TransformerModel):
    """
    The vanilla knn-mt model.
    """

    @staticmethod
    def add_args(parser):
        """
        add knn-mt related args here
        """
        TransformerModel.add_args(parser)
        parser.add_argument('--knn-mode', choices=['build_datastore', 'inference'], help='choose the action mode')
        parser.add_argument('--knn-datastore-path', type=str, metavar='STR', help='the directory of save or load datastore')
        parser.add_argument('--knn-k', type=int, metavar='N', default=8, help='The hyper-parameter k of vanilla knn-mt')
        parser.add_argument('--knn-lambda', type=float, metavar='D', default=0.7, help='The hyper-parameter lambda of vanilla knn-mt')
        parser.add_argument('--knn-temperature', type=float, metavar='D', default=10, help='The hyper-parameter temperature of vanilla knn-mt')
        parser.add_argument('--build-faiss-index-with-cpu', action='store_true', default=False, help='use faiss-cpu instead of faiss-gpu (useful when gpu memory is small)')

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        """
        we override this function, replace the TransformerDecoder with VanillaKNNMTDecoder
        """
        return VanillaKNNMTDecoder(args, tgt_dict, embed_tokens, no_encoder_attn=getattr(args, 'no_cross_attention', False))


class VanillaKNNMTVisualDecoder(TransformerDecoder):
    """
    The vanilla knn-mt Decoder with visualization, equipped with knn datastore, retriever and combiner.
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        """
        we override this function to create knn-related module.
        In other words, create datastore, retriever and combiner.
        """
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        if args.knn_mode == 'build_datastore':
            if 'datastore' not in global_vars():
                global_vars()['datastore'] = Datastore(args.knn_datastore_path)
            self.datastore = global_vars()['datastore']
        elif args.knn_mode == 'inference':
            self.datastore = Datastore.load(args.knn_datastore_path, load_list=['keys', 'vals', 'sentence_ids', 'token_positions'])
            self.datastore.load_faiss_index('keys')
            self.retriever = Retriever(datastore=self.datastore, k=args.knn_k)
            self.combiner = Combiner(lambda_=args.knn_lambda, temperature=args.knn_temperature, probability_dim=len(dictionary))

    def forward(self, prev_output_tokens, encoder_out: 'Optional[EncoderOut]'=None, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]'=None, features_only: 'bool'=False, full_context_alignment: 'bool'=False, alignment_layer: 'Optional[int]'=None, alignment_heads: 'Optional[int]'=None, src_lengths: 'Optional[Any]'=None, return_all_hiddens: 'bool'=False, **kwargs):
        """
        we overwrite this function to do something else besides forward the TransformerDecoder.
        
        when the action mode is `building datastore`, we save keys to datastore.
        when the action mode is `inference`, we retrieve the datastore with hidden state.
        """
        x, extra = self.extract_features(prev_output_tokens, encoder_out=encoder_out, incremental_state=incremental_state, full_context_alignment=full_context_alignment, alignment_layer=alignment_layer, alignment_heads=alignment_heads)
        if self.args.knn_mode == 'build_datastore':
            keys = select_keys_with_pad_mask(x, self.datastore.get_pad_mask())
            self.datastore['keys'].add(keys.half())
        elif self.args.knn_mode == 'inference':
            k = int(kwargs['knn_parameter']['k'])
            self.retriever.retrieve(x, k=k, return_list=['keys', 'vals', 'query', 'distances', 'sentence_ids', 'token_positions'])
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def get_normalized_probs(self, net_output: 'Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]]', log_probs: 'bool', sample: 'Optional[Dict[str, Tensor]]'=None):
        """
        we overwrite this function to change the probability calculation process.
        step 1. 
            calculate the knn probability based on retrieve resultes
        step 2.
            combine the knn probability with NMT's probability 
        
        compared to vanilla knn-mt, the visual version `get_noramlized_probs` need to return some extra infomation 
        when do inference
        """
        if self.args.knn_mode == 'inference':
            extra = {}
            lambda_ = sample['knn_parameter']['lambda']
            temperature = sample['knn_parameter']['temperature']
            knn_prob = self.combiner.get_knn_prob(**self.retriever.results, temperature=temperature, device=net_output[0].device)
            combined_prob, extra_combiner_info = self.combiner.get_combined_prob(knn_prob, net_output[0], lambda_=lambda_, log_probs=log_probs)
            extra['neural_probs'] = extra_combiner_info['neural_probs']
            extra['combined_probs'] = extra_combiner_info['unlog_combined_probs']
            extra['query_point'] = self.retriever.results['query']
            extra['knn_neighbors_values'] = self.retriever.results['vals']
            extra['knn_neighbors_keys'] = self.retriever.results['keys']
            extra['knn_l2_distance'] = self.retriever.results['distances']
            extra['knn_sentence_ids'] = self.retriever.results['sentence_ids']
            extra['knn_token_positions'] = self.retriever.results['token_positions']
            return combined_prob, extra
        else:
            return super().get_normalized_probs(net_output, log_probs, sample)


class VanillaKNNMTVisual(TransformerModel):
    """
    The vanilla knn-mt model with visualization.
    """

    @staticmethod
    def add_args(parser):
        """
        add knn-mt related args here
        """
        TransformerModel.add_args(parser)
        parser.add_argument('--knn-mode', choices=['build_datastore', 'inference'], help='choose the action mode')
        parser.add_argument('--knn-datastore-path', type=str, metavar='STR', help='the directory of save or load datastore')
        parser.add_argument('--knn-k', type=int, metavar='N', default=8, help='The hyper-parameter k of vanilla knn-mt')
        parser.add_argument('--knn-lambda', type=float, metavar='D', default=0.7, help='The hyper-parameter lambda of vanilla knn-mt')
        parser.add_argument('--knn-temperature', type=float, metavar='D', default=10, help='The hyper-parameter temperature of vanilla knn-mt')
        parser.add_argument('--build-faiss-index-with-cpu', action='store_true', default=False, help='use faiss-cpu instead of faiss-gpu (useful when gpu memory is small)')

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        """
        we override this function, replace the TransformerDecoder with VanillaKNNMTVisualDecoder
        """
        return VanillaKNNMTVisualDecoder(args, tgt_dict, embed_tokens, no_encoder_attn=getattr(args, 'no_cross_attention', False))


def lengths_to_encoder_padding_mask(lengths, batch_first=False):
    """
    convert lengths (a 1-D Long/Int tensor) to 2-D binary tensor

    Args:
        lengths: a (B, )-shaped tensor

    Return:
        max_length: maximum length of B sequences
        encoder_padding_mask: a (max_length, B) binary mask, where
        [t, b] = 0 for t < lengths[b] and 1 otherwise

    TODO:
        kernelize this function if benchmarking shows this function is slow
    """
    max_lengths = torch.max(lengths).item()
    bsz = lengths.size(0)
    encoder_padding_mask = torch.arange(max_lengths).view(1, max_lengths).expand(bsz, -1) >= lengths.view(bsz, 1).expand(-1, max_lengths)
    if not batch_first:
        return encoder_padding_mask.t(), max_lengths
    else:
        return encoder_padding_mask, max_lengths


class DummyEncoder(FairseqEncoder):

    def __init__(self):
        super().__init__(None)

    def forward(self, src_tokens, src_lengths):
        mask, max_len = lengths_to_encoder_padding_mask(src_lengths)
        return {'encoder_out': src_tokens, 'encoder_padding_mask': mask}


class DummyEncoderModel(FairseqEncoderModel):

    def __init__(self, encoder):
        super().__init__(encoder)

    @classmethod
    def build_model(cls, args, task):
        return cls(DummyEncoder())

    def get_logits(self, net_output):
        return torch.log(torch.div(net_output['encoder_out'], 1 - net_output['encoder_out']))

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        lprobs = super().get_normalized_probs(net_output, log_probs, sample=sample)
        lprobs.batch_first = True
        return lprobs


class ModelWithSharedParameter(nn.Module):

    def __init__(self):
        super(ModelWithSharedParameter, self).__init__()
        self.embedding = nn.Embedding(1000, 200)
        self.FC1 = nn.Linear(200, 200)
        self.FC2 = nn.Linear(200, 200)
        self.FC2.weight = nn.Parameter(self.FC1.weight)
        self.FC2.bias = nn.Parameter(self.FC1.bias)
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.FC2(self.ReLU(self.FC1(input))) + self.FC1(input)


class Model(nn.Module):

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        return output


class TestEncoder(FairseqEncoder):

    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.args = args

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        return EncoderOut(encoder_out=src_tokens, encoder_padding_mask=None, encoder_embedding=None, encoder_states=None, src_tokens=None, src_lengths=None)

    def reorder_encoder_out(self, encoder_out, new_order):
        return EncoderOut(encoder_out=encoder_out.encoder_out.index_select(0, new_order), encoder_padding_mask=None, encoder_embedding=None, encoder_states=None, src_tokens=None, src_lengths=None)


class TestIncrementalDecoder(FairseqIncrementalDecoder):

    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        assert hasattr(args, 'beam_probs') or hasattr(args, 'probs')
        args.max_decoder_positions = getattr(args, 'max_decoder_positions', 100)
        self.args = args

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None):
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bbsz = prev_output_tokens.size(0)
        vocab = len(self.dictionary)
        src_len = encoder_out.encoder_out.size(1)
        tgt_len = prev_output_tokens.size(1)
        if incremental_state is not None:
            step = utils.get_incremental_state(self, incremental_state, 'step')
            if step is None:
                step = 0
            utils.set_incremental_state(self, incremental_state, 'step', step + 1)
            steps = [step]
        else:
            steps = list(range(tgt_len))
        if hasattr(self.args, 'probs'):
            assert self.args.probs.dim() == 3, 'expected probs to have size bsz*steps*vocab'
            probs = self.args.probs.index_select(1, torch.LongTensor(steps))
        else:
            probs = torch.FloatTensor(bbsz, len(steps), vocab).zero_()
            for i, step in enumerate(steps):
                if step < len(self.args.beam_probs):
                    probs[:, i, self.dictionary.eos():] = self.args.beam_probs[step]
                else:
                    probs[:, i, self.dictionary.eos()] = 1.0
        attn = torch.rand(bbsz, tgt_len, src_len)
        dev = prev_output_tokens.device
        return probs, {'attn': [attn]}

    def get_normalized_probs(self, net_output, log_probs, _):
        probs = net_output[0]
        if log_probs:
            return probs.log()
        else:
            return probs

    def max_positions(self):
        return self.args.max_decoder_positions


class TestModel(FairseqEncoderDecoderModel):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, args, task):
        encoder = TestEncoder(args, task.source_dictionary)
        decoder = TestIncrementalDecoder(args, task.target_dictionary)
        return cls(encoder, decoder)


class TestReshapingEncoder(FairseqEncoder):

    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.args = args

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        b_sz, t_sz = src_tokens.shape
        padding_needed = t_sz % 2
        x = src_tokens
        if padding_needed > 0:
            padding_needed = 2 - padding_needed
            x = F.pad(x, (0, padding_needed))
        return EncoderOut(encoder_out=x.view(b_sz, -1, 2), encoder_padding_mask=None, encoder_embedding=None, encoder_states=None, src_tokens=None, src_lengths=None)

    def reorder_encoder_out(self, encoder_out, new_order):
        return EncoderOut(encoder_out=encoder_out.encoder_out.index_select(0, new_order), encoder_padding_mask=None, encoder_embedding=None, encoder_states=None, src_tokens=None, src_lengths=None)


class TestReshapingModel(FairseqEncoderDecoderModel):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, args, task):
        encoder = TestReshapingEncoder(args, task.source_dictionary)
        decoder = TestIncrementalDecoder(args, task.target_dictionary)
        return cls(encoder, decoder)


class TestAdditionalInputEncoder(FairseqEncoder):

    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.args = args

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        assert 'fancy_other_input' in kwargs
        assert kwargs['fancy_other_input'] is not None
        return EncoderOut(encoder_out=src_tokens, encoder_padding_mask=None, encoder_embedding=None, encoder_states=None, src_tokens=None, src_lengths=None)

    def reorder_encoder_out(self, encoder_out, new_order):
        return EncoderOut(encoder_out=encoder_out.encoder_out.index_select(0, new_order), encoder_padding_mask=None, encoder_embedding=None, encoder_states=None, src_tokens=None, src_lengths=None)


class TestAdditionalInputModel(FairseqEncoderDecoderModel):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, args, task):
        encoder = TestAdditionalInputEncoder(args, task.source_dictionary)
        decoder = TestIncrementalDecoder(args, task.target_dictionary)
        return cls(encoder, decoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AdaptiveSoftmax,
     lambda: ([], {'vocab_size': 4, 'input_dim': 4, 'cutoff': [4, 4], 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (BandwidthEstimator,
     lambda: ([], {'query_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (BeamableMM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (Conv1dSubsampler,
     lambda: ([], {'in_channels': 4, 'mid_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ConvTBC,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (Downsample,
     lambda: ([], {'index': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DownsampledMultiHeadAttention,
     lambda: ([], {'out_channels': 4, 'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (FairseqDropout,
     lambda: ([], {'p': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Fp32GroupNorm,
     lambda: ([], {'num_groups': 1, 'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Fp32LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GumbelVectorQuantizer,
     lambda: ([], {'dim': 4, 'num_vars': 4, 'temp': [4, 4, 4], 'groups': 1, 'combine_groups': 1, 'vq_dim': 4, 'time_first': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (Highway,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (KmeansVectorQuantizer,
     lambda: ([], {'dim': 4, 'num_vars': 4, 'groups': 1, 'combine_groups': 1, 'vq_dim': 4, 'time_first': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (LightweightConv1d,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (LinearizedConvolution,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (Model,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MonotonicMultiheadAttentionHard,
     lambda: ([], {'args': SimpleNamespace(decoder_embed_dim=4, decoder_attention_heads=4, encoder_embed_dim=4, attention_dropout=0.5, attention_eps=4, mass_preservation=4, noise_mean=4, noise_var=4, energy_bias_init=4, energy_bias=4)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (MonotonicMultiheadAttentionInfiniteLookback,
     lambda: ([], {'args': SimpleNamespace(decoder_embed_dim=4, decoder_attention_heads=4, encoder_embed_dim=4, attention_dropout=0.5, attention_eps=4, mass_preservation=4, noise_mean=4, noise_var=4, energy_bias_init=4, energy_bias=4)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (MonotonicMultiheadAttentionWaitk,
     lambda: ([], {'args': SimpleNamespace(decoder_embed_dim=4, decoder_attention_heads=4, encoder_embed_dim=4, attention_dropout=0.5, attention_eps=4, mass_preservation=4, noise_mean=4, noise_var=4, energy_bias_init=4, energy_bias=4, waitk_lagging=4)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (MultiheadAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (SamePad,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SelfAttention,
     lambda: ([], {'out_channels': 4, 'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (SingleHeadAttention,
     lambda: ([], {'out_channels': 4, 'embed_dim': 4, 'head_dim': 4, 'head_index': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (SparseMultiheadAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (TestEncoder,
     lambda: ([], {'args': SimpleNamespace(), 'dictionary': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TestReshapingEncoder,
     lambda: ([], {'args': SimpleNamespace(), 'dictionary': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (TransformerEncoderLayerNorm,
     lambda: ([], {'args': SimpleNamespace(encoder_normalize_before=4), 'embed_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TransposeLast,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (VGGBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'conv_kernel_size': 4, 'pooling_kernel_size': 4, 'num_conv_layers': 1, 'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Wav2VecCtc,
     lambda: ([], {'w2v_encoder': torch.nn.ReLU(), 'args': SimpleNamespace()}),
     lambda: ([], {'input': torch.rand([4, 4])})),
    (Wav2VecPredictionsModel,
     lambda: ([], {'in_dim': 4, 'out_dim': 4, 'prediction_steps': 4, 'n_negatives': 4, 'cross_sample_negatives': 4, 'sample_distance': 4, 'dropout': 0.5, 'offset': 4, 'balanced_classes': 4, 'infonce': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (WeightEstimator,
     lambda: ([], {'query_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ZeroPad1d,
     lambda: ([], {'pad_left': 4, 'pad_right': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

