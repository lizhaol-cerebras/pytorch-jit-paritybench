
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


import copy


import logging


from torch.utils.data import DataLoader


from torch.utils.dlpack import to_dlpack


import numpy as np


from typing import List


import math


from typing import Optional


from typing import Tuple


import torch.nn as nn


import torch.nn.functional as F


from torch.utils.dlpack import from_dlpack


import numpy


import time


from torch.utils.data import datapipes


from torch.utils.data.datapipes.iter import IterableWrapper


from functools import partial


from torch.utils.data import Dataset


from torch.utils.data import IterableDataset


from typing import Dict


import matplotlib.pyplot as plt


import matplotlib.font_manager as fm


import random


from typing import Union


import torch.utils.checkpoint as ckpt


import torch.distributed as dist


from torch.distributed.elastic.multiprocessing.errors import record


import collections


from collections.abc import Callable


from torch.utils.data import IterDataPipe


from torch.utils.data import functional_datapipe


from torch.utils.data.datapipes.iter import Mapper


from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES


from torch.utils.data.datapipes.iter.sharding import ShardingFilterIterDataPipe


from torch.utils.data.datapipes.utils.common import _check_unpickable_fn


from torch.nn.utils.rnn import pad_sequence


from torch import nn


from typing import Any


from torch.nn.modules.conv import _ConvNd


from torch.nn.modules.conv import _size_2_t


from torch.nn.modules.conv import Union


from torch.nn.modules.conv import _pair


from torch.nn.modules.conv import Tensor


from torch.nn.modules.conv import Optional


from collections import defaultdict


import re


from collections import OrderedDict


from torch.nn import BatchNorm1d


from torch.nn import LayerNorm


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


from torch.distributed.fsdp import FullStateDictConfig


from torch.distributed.fsdp import StateDictType


from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy


from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


import warnings


from torch.optim.lr_scheduler import _LRScheduler


import torch.optim as optim


from torch.nn.utils import clip_grad_norm_


from torch.distributed.fsdp import CPUOffload


from torch.distributed.fsdp import MixedPrecision


from torch.distributed.fsdp import sharded_grad_scaler


from torch.distributed.fsdp import ShardingStrategy


class Fbank(torch.nn.Module):

    def __init__(self, opts):
        super(Fbank, self).__init__()
        self.fbank = kaldifeat.Fbank(opts)

    def forward(self, waves: 'List[torch.Tensor]'):
        return self.fbank(waves)


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


class FrameReducer(nn.Module):
    """The encoder output is first used to calculate
    the CTC posterior probability; then for each output frame,
    if its blank posterior is bigger than some thresholds,
    it will be simply discarded from the encoder output.
    """

    def __init__(self, blank_threshlod: 'float'=0.95):
        super().__init__()
        self.blank_threshlod = blank_threshlod

    def forward(self, x: 'torch.Tensor', x_lens: 'torch.Tensor', ctc_output: 'torch.Tensor', y_lens: 'Optional[torch.Tensor]'=None, blank_id: 'int'=0) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:
              The shared encoder output with shape [N, T, C].
            x_lens:
              A tensor of shape (batch_size,) containing the number of frames in
              `x` before padding.
            ctc_output:
              The CTC output with shape [N, T, vocab_size].
            y_lens:
              A tensor of shape (batch_size,) containing the number of frames in
              `y` before padding.
            blank_id:
              The blank id of ctc_output.
        Returns:
            out:
              The frame reduced encoder output with shape [N, T', C].
            out_lens:
              A tensor of shape (batch_size,) containing the number of frames in
              `out` before padding.
        """
        N, T, C = x.size()
        padding_mask = make_pad_mask(x_lens, x.size(1))
        non_blank_mask = (ctc_output[:, :, blank_id] < math.log(self.blank_threshlod)) * ~padding_mask
        if y_lens is not None:
            limit_lens = T - y_lens
            max_limit_len = limit_lens.max().int()
            fake_limit_indexes = torch.topk(ctc_output[:, :, blank_id], max_limit_len).indices
            T = torch.arange(max_limit_len).expand_as(fake_limit_indexes)
            T = torch.remainder(T, limit_lens.unsqueeze(1))
            limit_indexes = torch.gather(fake_limit_indexes, 1, T)
            limit_mask = torch.full_like(non_blank_mask, False, device=x.device).scatter_(1, limit_indexes, True)
            non_blank_mask = non_blank_mask | ~limit_mask
        out_lens = non_blank_mask.sum(dim=1)
        max_len = out_lens.max()
        pad_lens_list = torch.full_like(out_lens, max_len.item(), device=x.device) - out_lens
        max_pad_len = pad_lens_list.max()
        out = F.pad(x, (0, 0, 0, max_pad_len))
        valid_pad_mask = ~make_pad_mask(pad_lens_list)
        total_valid_mask = torch.concat([non_blank_mask, valid_pad_mask], dim=1)
        out = out[total_valid_mask].reshape(N, -1, C)
        return out, out_lens


class SinusoidalPositionEncoder(torch.nn.Module):
    """https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/modules/embedding.py#L387
    """

    def __int__(self):
        pass

    def encode(self, positions: 'torch.Tensor', depth: 'int', dtype: 'torch.dtype'=torch.float32) ->torch.Tensor:
        batch_size = positions.size(0)
        positions = positions.type(dtype)
        device = positions.device
        log_timescale_increment = torch.log(torch.tensor([10000], dtype=dtype, device=device)) / (depth / 2 - 1)
        inv_timescales = torch.exp(torch.arange(depth / 2, device=device).type(dtype) * -log_timescale_increment)
        inv_timescales = torch.reshape(inv_timescales, [batch_size, -1])
        scaled_time = torch.reshape(positions, [1, -1, 1]) * torch.reshape(inv_timescales, [1, 1, -1])
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        return encoding

    def forward(self, x):
        _, timesteps, input_dim = x.size()
        positions = torch.arange(1, timesteps + 1, device=x.device)[None, :]
        position_encoding = self.encode(positions, input_dim, x.dtype)
        return x + position_encoding, position_encoding


IGNORE_ID = -1


@torch.no_grad()
def sampler(logits: 'torch.Tensor', temperatures: 'Union[torch.Tensor, None]', top_ps: 'torch.Tensor', top_ks: 'torch.Tensor') ->torch.Tensor:
    assert logits.size(1) == 1
    logits = logits.squeeze(1)
    if temperatures is None:
        return torch.argmax(logits, dim=-1).squeeze(dim=-1)
    logits.div_(temperatures.unsqueeze(dim=1))
    probs = torch.softmax(logits, dim=-1, dtype=torch.float)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    top_ps_mask = probs_sum - probs_sort > top_ps.unsqueeze(dim=1)
    probs_sort = torch.where(top_ps_mask, 0, probs_sort)
    top_ks_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device)
    top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
    top_ks_mask = top_ks_mask >= top_ks.unsqueeze(dim=1)
    probs_sort = torch.where(top_ks_mask, 0, probs_sort)
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    probs = torch.gather(probs_sort, dim=-1, index=torch.argsort(probs_idx, dim=-1))
    next_token_ids = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(dim=-1)
    return next_token_ids


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


class CausalLM(torch.nn.Module):

    def __init__(self, vocab_size: 'int', decoder: 'DecoderOnly', special_tokens: 'dict', tie_word_embedding: 'bool'=False, linear_bias: 'bool'=False, ignore_id: 'int'=IGNORE_ID, lsm_weight: 'float'=0.0, reduction: 'str'='mean') ->None:
        super().__init__()
        del special_tokens
        self.embed = torch.nn.Embedding(vocab_size, decoder.hidden_size)
        self.out = torch.nn.Linear(decoder.hidden_size, vocab_size, bias=linear_bias)
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.criterion_att = torch.nn.CrossEntropyLoss(ignore_index=ignore_id, label_smoothing=lsm_weight, reduction=reduction)
        self.tie_word_embedding = tie_word_embedding
        self.ignore_id = ignore_id

    @torch.jit.unused
    def forward(self, batch: 'dict', device: 'torch.device') ->Dict[str, Optional[torch.Tensor]]:
        """ Forward for training
        """
        text = batch['feats']
        target = batch['target']
        text_length = batch['feats_lengths']
        mask = ~make_pad_mask(text_length, max_len=text.size(1)).unsqueeze(1)
        causal_mask = subsequent_mask(mask.size(-1), device=mask.device).unsqueeze(0)
        att_mask = causal_mask & mask
        embeding = self.embed(text)
        decoder_out = self.out(self.decoder(embeding, att_mask)[0])
        loss = self.criterion_att(decoder_out.view(-1, self.vocab_size), target.view(-1))
        acc = th_accuracy(decoder_out.view(-1, self.vocab_size), target, ignore_label=self.ignore_id)
        return {'loss': loss, 'ppl': torch.exp(loss.detach()), 'th_accuracy': acc}

    def tie_or_clone_weights(self, jit_mode: 'bool'):
        if not self.tie_word_embedding:
            return
        if jit_mode:
            self.out.weight = torch.nn.Parameter(self.embed.weight.clone())
        else:
            self.out.weight = self.embed.weight

    @torch.jit.unused
    @torch.inference_mode()
    def generate(self, prompts_tokens: 'List[List[int]]', device: 'torch.device', stop_tokens: 'List[int]', dtype: 'torch.dtype'=torch.float32, output_len: 'int'=100, temperature: 'Union[float, None]'=0.95, top_p: 'float'=1.0, top_k: 'int'=100) ->List[List[int]]:
        """Generates responses for given prompts using Gemma model."""
        batch_size = len(prompts_tokens)
        min_prompt_len = min(len(p) for p in prompts_tokens)
        max_prompt_len = max(len(p) for p in prompts_tokens)
        max_seq_len = max_prompt_len + output_len
        assert max_seq_len <= self.decoder.pos_enc.max_len
        kv_caches = []
        for _ in range(len(self.decoder.decoders)):
            size = batch_size, 0, self.decoder.n_kv_head, self.decoder.head_dim
            k_cache = torch.zeros(size=size, dtype=dtype, device=device)
            v_cache = torch.zeros(size=size, dtype=dtype, device=device)
            kv_caches.append((k_cache, v_cache))
        token_ids_tensor = torch.full((batch_size, max_seq_len), IGNORE_ID, dtype=torch.int64, device=device)
        input_token_ids_tensor = torch.full((batch_size, min_prompt_len), IGNORE_ID, dtype=torch.int64, device=device)
        for i, p in enumerate(prompts_tokens):
            token_ids_tensor[i, :len(p)] = torch.tensor(p)
            input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(p[:min_prompt_len])
        prompt_mask_tensor = token_ids_tensor != IGNORE_ID
        input_positions_tensor = torch.arange(0, min_prompt_len, dtype=torch.int64)
        mask_tensor = torch.ones((1, 1, max_seq_len, max_seq_len), dtype=torch.bool)
        mask_tensor = torch.tril(mask_tensor)
        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        att_mask = curr_mask_tensor.squeeze(1)[:, :min_prompt_len, :min_prompt_len]
        output_positions_tensor = torch.LongTensor([min_prompt_len - 1])
        temperatures_tensor = None if not temperature else torch.FloatTensor([temperature] * batch_size)
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size)
        output_index = torch.tensor(min_prompt_len, dtype=torch.int64)
        input_token_embeding = self.embed(input_token_ids_tensor)
        offset = torch.tensor([0] * len(prompts_tokens))
        input_offset = offset
        stop_tokens_tensor = torch.tensor(stop_tokens, device=device)
        for i in range(max_seq_len - min_prompt_len):
            decoder_out, kv_caches = self.decoder(input_token_embeding, att_mask, input_offset, kv_caches)
            decoder_out = self.out(decoder_out)
            decoder_out = decoder_out.index_select(1, output_positions_tensor)
            next_token_ids = sampler(decoder_out, temperatures_tensor, top_ps_tensor, top_ks_tensor)
            curr_prompt_mask = prompt_mask_tensor.index_select(1, output_index).squeeze(dim=1)
            curr_token_ids = token_ids_tensor.index_select(1, output_index).squeeze(dim=1)
            output_token_ids = torch.where(curr_prompt_mask, curr_token_ids, next_token_ids).unsqueeze(dim=1)
            token_ids_tensor.index_copy_(1, output_index, output_token_ids)
            input_token_ids_tensor = output_token_ids
            input_token_embeding = self.embed(input_token_ids_tensor)
            input_positions_tensor = output_index.unsqueeze(dim=-1)
            curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
            att_mask = curr_mask_tensor.squeeze(1)[:, :output_index + 1, :output_index + 1]
            output_positions_tensor = torch.tensor(0, dtype=torch.int64)
            input_offset = offset + output_index.unsqueeze(-1)
            output_index = output_index + 1
            if all(torch.isin(next_token_ids, stop_tokens_tensor)):
                break
        token_ids = token_ids_tensor.tolist()
        results = []
        for i, tokens in enumerate(token_ids):
            trimmed_output = tokens[len(prompts_tokens[i]):len(prompts_tokens[i]) + output_len]
            for stop_token in stop_tokens:
                try:
                    eos_index = trimmed_output.index(stop_token)
                    trimmed_output = trimmed_output[:eos_index]
                    break
                except Exception:
                    continue
            results.append(trimmed_output)
        return results


T_CACHE = Tuple[torch.Tensor, torch.Tensor]


class RMSNorm(torch.nn.Module):
    """ https://arxiv.org/pdf/1910.07467.pdf
    """

    def __init__(self, dim: 'int', eps: 'float'=1e-06, add_unit_offset: 'bool'=True):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.add_unit_offset = add_unit_offset

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        x = self._norm(x.float()).type_as(x)
        if self.add_unit_offset:
            return x * (1 + self.weight)
        else:
            return x * self.weight


WENET_NORM_CLASSES = {'layer_norm': LayerNorm, 'batch_norm': BatchNorm1d, 'rms_norm': RMSNorm}


class TransformerEncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
    """

    def __init__(self, size: 'int', self_attn: 'torch.nn.Module', feed_forward: 'torch.nn.Module', dropout_rate: 'float', normalize_before: 'bool'=True, layer_norm_type: 'str'='layer_norm', norm_eps: 'float'=1e-05, rms_norm_offset: 'bool'=True):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        assert layer_norm_type in ['layer_norm', 'rms_norm']
        norm_class = WENET_NORM_CLASSES[layer_norm_type]
        if layer_norm_type == 'rms_norm':
            norm_class = partial(norm_class, add_unit_offset=rms_norm_offset)
        self.norm1 = norm_class(size, eps=norm_eps)
        self.norm2 = norm_class(size, eps=norm_eps)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

    def forward(self, x: 'torch.Tensor', mask: 'torch.Tensor', pos_emb: 'torch.Tensor', mask_pad: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool), att_cache: 'T_CACHE'=(torch.zeros((0, 0, 0, 0)), torch.zeros((0, 0, 0, 0))), cnn_cache: 'torch.Tensor'=torch.zeros((0, 0, 0, 0))) ->Tuple[torch.Tensor, torch.Tensor, T_CACHE, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time，time),
                (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): just for interface compatibility
                to ConformerEncoderLayer
            mask_pad (torch.Tensor): does not used in transformer layer,
                just for unified api with conformer.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
                (#batch=1, size, cache_t2), not used here, it's for interface
                compatibility to ConformerEncoderLayer.
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time).
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cahce tensor (#batch=1, size, cache_t2).

        """
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb, cache=att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm1(x)
        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)
        fake_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        return x, mask, new_att_cache, fake_cnn_cache


class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Return Swish activation function."""
        return x * torch.sigmoid(x)


WENET_ACTIVATION_CLASSES = {'hardtanh': torch.nn.Hardtanh, 'tanh': torch.nn.Tanh, 'relu': torch.nn.ReLU, 'selu': torch.nn.SELU, 'swish': getattr(torch.nn, 'SiLU', Swish), 'gelu': torch.nn.GELU}


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.
    if n_kv_head != None and n_kv_head != n_head
    see: https://arxiv.org/pdf/1911.02150.pdf
         https://arxiv.org/pdf/2305.13245.pdf

    Example:
        case 1: n_kv_head == None, head_dim == None, MultiHead attention (MHSA)
        case 2: n_kv_head=1, n_head = 16, MultiQuery attention (MQA)
        case 3: nv_kv_head=2, n_head = 16, GroupedQuery attention (GQA)

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head: 'int', n_feat: 'int', dropout_rate: 'float', query_bias: 'bool'=True, key_bias: 'bool'=True, value_bias: 'bool'=True, use_sdpa: 'bool'=False, n_kv_head: 'Optional[int]'=None, head_dim: 'Optional[int]'=None):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        self.inner_dim = n_feat if head_dim is None else head_dim * n_head
        if n_kv_head is not None:
            assert head_dim is not None
            self.inner_kv_dim = head_dim * n_kv_head
            n_kv_head = n_kv_head
        else:
            self.inner_kv_dim = self.inner_dim
            n_kv_head = n_head
        self.d_k = self.inner_dim // n_head
        assert self.d_k == self.inner_kv_dim // n_kv_head
        self.h = n_head
        self.h_kv = n_kv_head
        self.linear_q = nn.Linear(n_feat, self.inner_dim, bias=query_bias)
        self.linear_k = nn.Linear(n_feat, self.inner_kv_dim, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, self.inner_kv_dim, bias=value_bias)
        self.linear_out = nn.Linear(self.inner_dim, n_feat, bias=query_bias)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.use_sdpa = use_sdpa
        self.dropout_rate = dropout_rate

    def _forward_linearx(self, name: 'str', x: 'torch.Tensor', head_first: 'bool'=True) ->torch.Tensor:
        assert x.ndim >= 3
        if name == 'query':
            x = self.linear_q(x)
            x_shape = x.size()
            x_shape = x_shape[:-1] + torch.Size([self.h, self.d_k])
        elif name == 'key':
            x = self.linear_k(x)
            x_shape = x.size()
            x_shape = x_shape[:-1] + torch.Size([self.h_kv, self.d_k])
        else:
            assert name == 'value'
            x = self.linear_v(x)
            x_shape = x.size()
            x_shape = x_shape[:-1] + torch.Size([self.h_kv, self.d_k])
        x = x.view(x_shape)
        if head_first:
            x = x.transpose(-3, -2)
        return x

    def forward_qkv(self, query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, ..., time1, size).
            key (torch.Tensor): Key tensor (#batch, ..., time2, size).
            value (torch.Tensor): Value tensor (#batch, ..., time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, ..., n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, ..., n_head_kv, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, ..., n_head_kv, time2, d_k).

        """
        q = self._forward_linearx('query', query)
        k = self._forward_linearx('key', key)
        v = self._forward_linearx('value', value)
        return q, k, v

    def forward_attention(self, value: 'torch.Tensor', scores: 'torch.Tensor', mask: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool)) ->torch.Tensor:
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value, size
                (#batch, ..., n_head, time2, d_k).
            scores (torch.Tensor): Attention score, size
                (#batch, ..., n_head, time1, time2).
            mask (torch.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, ..., time1, time2), (0, ..., 0, 0) means fake mask.

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        if mask.size(-1) > 0:
            mask = mask.unsqueeze(-3).eq(0)
            mask = mask[..., :scores.size(-1)]
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores.float(), dim=-1).type_as(value).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores.float(), dim=-1).type_as(value)
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(-3, -2).contiguous()
        x_shape = x.size()[:-2] + torch.Size([self.h * self.d_k])
        x = x.view(x_shape)
        return self.linear_out(x)

    def _update_kv_and_cache(self, k: 'torch.Tensor', v: 'torch.Tensor', cache: 'T_CACHE', head_first: 'bool'=True) ->Tuple[torch.Tensor, torch.Tensor, T_CACHE]:
        new_cache = cache
        seq_axis = -2 if head_first else -3
        head_axis = -3 if head_first else -2
        if not self.training:
            key_cache, value_cache = cache
            if key_cache.size(0) > 0:
                k = torch.cat([key_cache, k], dim=seq_axis)
            if value_cache.size(0) > 0:
                v = torch.cat([value_cache, v], dim=seq_axis)
            new_cache = k, v
        if self.h_kv != self.h and self.h_kv != 1:
            n_repeat = self.h // self.h_kv
            k_shape = k.size()
            repeat_axis = head_axis + 1
            k = k.unsqueeze(head_axis).expand(k_shape[:repeat_axis] + torch.Size([n_repeat]) + k_shape[repeat_axis:]).reshape(k_shape[:head_axis] + torch.Size([self.h_kv * n_repeat]) + k_shape[repeat_axis:])
            v_shape = v.size()
            v = v.unsqueeze(head_axis).expand(v_shape[:repeat_axis] + torch.Size([n_repeat]) + v_shape[repeat_axis:]).reshape(v_shape[:head_axis] + torch.Size([self.h_kv * n_repeat]) + v_shape[repeat_axis:])
        return k, v, new_cache

    def forward(self, query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor', mask: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool), pos_emb: 'torch.Tensor'=torch.empty(0), cache: 'T_CACHE'=(torch.zeros(0, 0, 0, 0), torch.zeros(0, 0, 0, 0))) ->Tuple[torch.Tensor, T_CACHE]:
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
                Wenet.
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
        k, v, new_cache = self._update_kv_and_cache(k, v, cache)
        if not self.use_sdpa:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            return self.forward_attention(v, scores, mask), new_cache
        else:
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask.unsqueeze(1), dropout_p=self.dropout_rate, scale=1 / math.sqrt(self.d_k))
            output = output.transpose(1, 2).contiguous().view(query.size(0), -1, self.h * self.d_k)
            return self.linear_out(output), new_cache


class GroupedRelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper:
        https://arxiv.org/abs/1901.02860
        https://arxiv.org/abs/2109.01163
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head, n_feat, dropout_rate, group_size=3):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.group_size = group_size
        self.d_k = n_feat // n_head
        self.n_feat = n_feat
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k * self.group_size))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k * self.group_size))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x, zero_triu: 'bool'=False):
        """Compute relative positinal encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, size).
            zero_triu (bool): If true, return the lower triangular part of
                the matrix.
        Returns:
            torch.Tensor: Output tensor.
        """
        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(x.size()[0], x.size()[1], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)
        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]
        return x

    def pad4group(self, Q, K, V, P, mask, group_size: 'int'=3):
        """
        q: (#batch, time1, size) -> (#batch, head, time1, size/head)
        k,v: (#batch, time2, size) -> (#batch, head, time2, size/head)
        p: (#batch, time2, size)
        """
        overflow_Q = Q.size(2) % group_size
        overflow_KV = K.size(2) % group_size
        padding_Q = (group_size - overflow_Q) * int(overflow_Q // (overflow_Q + 1e-17))
        padding_KV = (group_size - overflow_KV) * int(overflow_KV // (overflow_KV + 1e-17))
        batch_size, _, seq_len_KV, _ = K.size()
        Q = F.pad(Q, (0, 0, 0, padding_Q), value=0.0)
        K = F.pad(K, (0, 0, 0, padding_KV), value=0.0)
        V = F.pad(V, (0, 0, 0, padding_KV), value=0.0)
        if mask is not None and mask.size(2) > 0:
            mask = mask[:, ::group_size, ::group_size]
        Q = Q.transpose(1, 2).contiguous().view(batch_size, -1, self.h, self.d_k * group_size).transpose(1, 2)
        K = K.transpose(1, 2).contiguous().view(batch_size, -1, self.h, self.d_k * group_size).transpose(1, 2)
        V = V.transpose(1, 2).contiguous().view(batch_size, -1, self.h, self.d_k * group_size).transpose(1, 2)
        P_batch_size = P.size(0)
        overflow_P = P.size(1) % group_size
        padding_P = group_size - overflow_P if overflow_P else 0
        P = F.pad(P, (0, 0, 0, padding_P), value=0.0)
        P = P.view(P_batch_size, -1, self.h, self.d_k * group_size).transpose(1, 2)
        return Q, K, V, P, mask, padding_Q

    def forward_attention(self, value: 'torch.Tensor', scores: 'torch.Tensor', mask: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool), padding_q: 'Optional[int]'=None) ->torch.Tensor:
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.
            padding_q : for GroupedAttention in efficent conformer

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
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.n_feat)
        if padding_q is not None:
            x = x[:, :x.size(1) - padding_q]
        return self.linear_out(x)

    def forward(self, query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor', mask: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool), pos_emb: 'torch.Tensor'=torch.empty(0), cache: 'torch.Tensor'=torch.zeros((0, 0, 0, 0))) ->Tuple[torch.Tensor, torch.Tensor]:
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
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
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)
        p = self.linear_pos(pos_emb)
        batch_size, seq_len_KV, _ = k.size()
        q = q.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        new_cache = torch.cat((k, v), dim=-1)
        if mask is not None and mask.size(2) > 0:
            time2 = mask.size(2)
            k = k[:, :, -time2:, :]
            v = v[:, :, -time2:, :]
        q, k, v, p, mask, padding_q = self.pad4group(q, k, v, p, mask, self.group_size)
        q = q.transpose(1, 2)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k * self.group_size)
        return self.forward_attention(v, scores, mask, padding_q), new_cache


class MultiHeadedCrossAttention(MultiHeadedAttention):

    def __init__(self, n_head: 'int', n_feat: 'int', dropout_rate: 'float', query_bias: 'bool'=True, key_bias: 'bool'=True, value_bias: 'bool'=True, use_sdpa: 'bool'=False, n_kv_head: 'Optional[int]'=None, head_dim: 'Optional[int]'=None):
        super().__init__(n_head, n_feat, dropout_rate, query_bias, key_bias, value_bias, use_sdpa, n_kv_head, head_dim)

    def forward(self, query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor', mask: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool), pos_emb: 'torch.Tensor'=torch.empty(0), cache: 'T_CACHE'=(torch.zeros((0, 0, 0, 0)), torch.zeros((0, 0, 0, 0)))) ->Tuple[torch.Tensor, T_CACHE]:
        del pos_emb
        key_cache, value_cache = cache
        assert key_cache.size(0) == value_cache.size(0)
        if key_cache.size(0) > 0:
            assert not self.training
            q = self._forward_linearx('query', query)
            k, v = key_cache, value_cache
        else:
            q, k, v = self.forward_qkv(query, key, value)
        new_cache = (k, v) if not self.training else cache
        if self.h_kv != self.h and self.h_kv != 1:
            k = torch.repeat_interleave(k, self.h // self.h_kv, dim=-3)
            v = torch.repeat_interleave(v, self.h // self.h_kv, dim=-3)
        B = query.size(0)
        Beams = 1
        if B != k.size(0):
            assert not self.training
            Beams = B // k.size(0)
            B = k.size(0)
            q = q.view(B, Beams, q.size(-3), q.size(-2), q.size(-1))
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            mask = mask.unsqueeze(1)
        if not self.use_sdpa:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            output = self.forward_attention(v, scores, mask)
        else:
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask.unsqueeze(1), dropout_p=self.dropout_rate, scale=1 / math.sqrt(self.d_k))
            output = output.transpose(-2, -3).contiguous()
            output_shape = output.size()[:-2] + torch.Size([self.h * self.d_k])
            output = output.view(output_shape)
            output = self.linear_out(output)
        if query.size(0) != B:
            assert not self.training
            output_shape = torch.Size([B * Beams]) + output.size()[2:]
            output = output.view(output_shape)
        return output, new_cache


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head: 'int', n_feat: 'int', dropout_rate: 'float', query_bias: 'bool'=True, key_bias: 'bool'=True, value_bias: 'bool'=True, use_sdpa: 'bool'=False, n_kv_head: 'Optional[int]'=None, head_dim: 'Optional[int]'=None):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate, query_bias, key_bias, value_bias, use_sdpa, n_kv_head, head_dim)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x, zero_triu: 'bool'=False):
        """Compute relative positinal encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, size).
            zero_triu (bool): If true, return the lower triangular part of
                the matrix.
        Returns:
            torch.Tensor: Output tensor.
        """
        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(x.size()[0], x.size()[1], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)
        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]
        return x

    def forward(self, query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor', mask: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool), pos_emb: 'torch.Tensor'=torch.empty(0), cache: 'T_CACHE'=(torch.zeros((0, 0, 0, 0)), torch.zeros((0, 0, 0, 0)))) ->Tuple[torch.Tensor, T_CACHE]:
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
        k, v, new_cache = self._update_kv_and_cache(k, v, cache)
        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        if not self.use_sdpa:
            matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
            scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
            return self.forward_attention(v, scores, mask), new_cache
        else:
            assert mask.dtype != torch.bool
            mask = mask.unsqueeze(1)
            mask = (matrix_bd + mask) / math.sqrt(self.d_k)
            output = torch.nn.functional.scaled_dot_product_attention(q_with_bias_u, k, v, attn_mask=mask, dropout_p=self.dropout_rate, scale=1 / math.sqrt(self.d_k))
            output = output.transpose(1, 2).contiguous().view(query.size(0), -1, self.h * self.d_k)
            return self.linear_out(output), new_cache


def google_apply_rotary_emb(x: 'torch.Tensor', freqs_cis: 'torch.Tensor') ->torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    x_ = torch.view_as_complex(torch.stack(torch.chunk(x.float(), 2, dim=-1), dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1)
    return x_out


def llama_apply_rotary_emb(x: 'torch.Tensor', freqs_cis: 'torch.Tensor') ->torch.Tensor:
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x)


WENET_APPLY_ROTARY_EMB = {'google': google_apply_rotary_emb, 'llama': llama_apply_rotary_emb}


class RopeMultiHeadedAttention(MultiHeadedAttention):

    def __init__(self, n_head: 'int', n_feat: 'int', dropout_rate: 'float', query_bias: 'bool'=True, key_bias: 'bool'=True, value_bias: 'bool'=True, use_sdpa: 'bool'=False, n_kv_head: 'Optional[int]'=None, head_dim: 'Optional[int]'=None, style='google'):
        super().__init__(n_head, n_feat, dropout_rate, query_bias, key_bias, value_bias, use_sdpa, n_kv_head, head_dim)
        self.style = style

    def forward(self, query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor', mask: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool), pos_emb: 'torch.Tensor'=torch.empty(0), cache: 'T_CACHE'=(torch.zeros((0, 0, 0, 0)), torch.zeros(0, 0, 0, 0))) ->Tuple[torch.Tensor, T_CACHE]:
        """Compute rope scaled dot product attention.

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
                Wenet.
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`


        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`

        """
        q = self._forward_linearx('query', query, head_first=False)
        k = self._forward_linearx('key', key, head_first=False)
        v = self._forward_linearx('value', value, head_first=False)
        q = WENET_APPLY_ROTARY_EMB[self.style](q, pos_emb)
        k = WENET_APPLY_ROTARY_EMB[self.style](k, pos_emb)
        k, v, new_cache = self._update_kv_and_cache(k, v, cache, head_first=False)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if not self.use_sdpa:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            return self.forward_attention(v, scores, mask), new_cache
        else:
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask.unsqueeze(1), dropout_p=self.dropout_rate, scale=1 / math.sqrt(self.d_k))
            output = output.transpose(1, 2).contiguous().view(query.size(0), -1, self.h * self.d_k)
            return self.linear_out(output), new_cache


class ShawRelPositionMultiHeadedAttention(MultiHeadedAttention):
    """ https://arxiv.org/pdf/1803.02155.pdf
    """

    def __init__(self, n_head: 'int', n_feat: 'int', dropout_rate: 'float', query_bias: 'bool'=True, key_bias: 'bool'=True, value_bias: 'bool'=True, use_sdpa: 'bool'=False, n_kv_head: 'Optional[int]'=None, head_dim: 'Optional[int]'=None):
        del n_kv_head, head_dim
        super().__init__(n_head, n_feat, dropout_rate, query_bias, key_bias, value_bias, use_sdpa, None, None)
        self.max_right_rel_pos = 8
        self.max_left_rel_pos = 64
        self.rel_k_embed = torch.nn.Embedding(self.max_left_rel_pos + self.max_right_rel_pos + 1, self.d_k)

    def _relative_indices(self, keys: 'torch.Tensor') ->torch.Tensor:
        indices = torch.arange(keys.size(2), device=keys.device).unsqueeze(0)
        rel_indices = indices - indices.transpose(0, 1)
        rel_indices = torch.clamp(rel_indices, -self.max_left_rel_pos, self.max_right_rel_pos)
        return rel_indices + self.max_left_rel_pos

    def forward(self, query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor', mask: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool), pos_emb: 'torch.Tensor'=torch.empty(0), cache: 'T_CACHE'=(torch.zeros((0, 0, 0, 0)), torch.zeros(0, 0, 0, 0))) ->Tuple[torch.Tensor, T_CACHE]:
        del pos_emb
        q, k, v = self.forward_qkv(query, key, value)
        k, v, new_cache = self._update_kv_and_cache(k, v, cache)
        rel_k = self.rel_k_embed(self._relative_indices(k))
        rel_k = rel_k[-q.size(2):]
        rel_att_weights = torch.einsum('bhld,lrd->bhlr', q, rel_k)
        if not self.use_sdpa:
            scores = (torch.matmul(q, k.transpose(-2, -1)) + rel_att_weights) / math.sqrt(self.d_k)
            return self.forward_attention(v, scores, mask), new_cache
        else:
            assert mask.dtype != torch.bool
            mask = mask.unsqueeze(1)
            mask = (rel_att_weights + mask) / math.sqrt(self.d_k)
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout_rate, scale=1 / math.sqrt(self.d_k))
            output = output.transpose(1, 2).contiguous().view(query.size(0), -1, self.h * self.d_k)
            return self.linear_out(output), new_cache


WENET_ATTENTION_CLASSES = {'selfattn': MultiHeadedAttention, 'rel_selfattn': RelPositionMultiHeadedAttention, 'grouped_rel_selfattn': GroupedRelPositionMultiHeadedAttention, 'crossattn': MultiHeadedCrossAttention, 'shaw_rel_selfattn': ShawRelPositionMultiHeadedAttention, 'rope_abs_selfattn': RopeMultiHeadedAttention}


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
        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: 'torch.Tensor', offset: 'Union[int, torch.Tensor]'=0) ->Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
            offset (int, torch.tensor): position offset

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            torch.Tensor: for compatibility to RelPositionalEncoding
        """
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


class ParaformerPositinoalEncoding(WhisperPositionalEncoding):
    """ Sinusoids position encoding used in paraformer.encoder
    """

    def __init__(self, depth: 'int', d_model: 'int', dropout_rate: 'float'=0.1, max_len: 'int'=1500):
        super().__init__(depth, dropout_rate, max_len)
        self.xscale = d_model ** 0.5


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
        x = x * self.xscale
        pos_emb = self.position_encoding(offset, x.size(1), False)
        return self.dropout(x), self.dropout(pos_emb)


def precompute_freqs_cis(dim: 'int', end: 'int', theta: 'float'=10000.0) ->torch.Tensor:
    """Precomputes the frequency cis."""
    freqs = 1.0 / theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


class RopePositionalEncoding(PositionalEncoding):

    def __init__(self, d_model: 'int', head_dim: 'int', dropout_rate: 'float', max_len: 'int'=1500, rope_theta=10000.0, scale: 'bool'=True):
        super().__init__(d_model, dropout_rate=dropout_rate, max_len=max_len)
        delattr(self, 'pe')
        self.max_len = max_len * 2
        pe = precompute_freqs_cis(head_dim, self.max_len, rope_theta)
        self.register_buffer('pe', torch.view_as_real(pe.unsqueeze(0)))
        self.dropout_rate = dropout_rate
        self.scale = scale

    def forward(self, x: 'torch.Tensor', offset: 'Union[int, torch.Tensor]'=0) ->Tuple[torch.Tensor, torch.Tensor]:
        pos_emb = self.position_encoding(offset, x.size(1), True)
        pos_emb = pos_emb.unsqueeze(2)
        if self.scale:
            x = x * self.xscale
        return self.dropout(x), pos_emb

    def position_encoding(self, offset: 'Union[int, torch.Tensor]', size: 'int', apply_dropout: 'bool'=True) ->torch.Tensor:
        pe = torch.view_as_complex(self.pe)
        if isinstance(offset, int):
            assert offset + size <= self.max_len
            pos_emb = pe[:, offset:offset + size]
        else:
            assert torch.max(offset) + size <= self.max_len
            index = offset.unsqueeze(1) + torch.arange(0, size)
            flag = index > 0
            index = index * flag
            pos_emb = F.embedding(index, pe[0])
        if apply_dropout:
            pos_emb = self.dropout_complex(pos_emb)
        return pos_emb

    def dropout_complex(self, x):
        mask = torch.nn.functional.dropout(torch.ones_like(x.real), training=self.training, p=self.dropout_rate)
        return x * mask


WENET_EMB_CLASSES = {'embed': PositionalEncoding, 'abs_pos': PositionalEncoding, 'rel_pos': RelPositionalEncoding, 'no_pos': NoPositionalEncoding, 'abs_pos_whisper': WhisperPositionalEncoding, 'embed_learnable_pe': LearnablePositionalEncoding, 'abs_pos_paraformer': ParaformerPositinoalEncoding, 'rope_pos': RopePositionalEncoding}


class GatedVariantsMLP(torch.nn.Module):
    """ https://arxiv.org/pdf/2002.05202.pdf
    """

    def __init__(self, idim: 'int', hidden_units: 'int', dropout_rate: 'float', activation: 'torch.nn.Module'=torch.nn.GELU(), bias: 'bool'=True, *dummy_args, **dummy_kwargs):
        """Construct a PositionwiseFeedForward object."""
        super(GatedVariantsMLP, self).__init__()
        self.gate = torch.nn.Linear(idim, hidden_units, bias=False)
        self.activation = activation
        self.w_1 = torch.nn.Linear(idim, hidden_units, bias=bias)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim, bias=bias)

    def forward(self, x) ->torch.Tensor:
        """Foward function.
        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)

        """
        gate = self.activation(self.gate(x))
        up = self.w_1(x)
        fuse = gate * up
        return self.w_2(self.dropout(fuse))


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

    def __init__(self, idim: 'int', hidden_units: 'int', dropout_rate: 'float', activation: 'torch.nn.Module'=torch.nn.ReLU(), bias: 'bool'=True, *dummy_args, **dummy_kwargs):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units, bias=bias)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim, bias=bias)

    def forward(self, xs: 'torch.Tensor') ->torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class MoEFFNLayer(torch.nn.Module):
    """
    Mixture of expert with Positionwise feed forward layer
    See also figure 1 in https://arxiv.org/pdf/2305.15663.pdf
    The output dim is same with the input dim.

    Modified from https://github.com/Lightning-AI/lit-gpt/pull/823
                  https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219
    Args:
        n_expert: number of expert.
        n_expert_activated: The actual number of experts used for each frame
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """

    def __init__(self, idim: 'int', hidden_units: 'int', dropout_rate: 'float', activation: 'torch.nn.Module'=torch.nn.ReLU(), bias: 'bool'=False, n_expert: 'int'=8, n_expert_activated: 'int'=2):
        super(MoEFFNLayer, self).__init__()
        self.gate = torch.nn.Linear(idim, n_expert, bias=False)
        self.experts = torch.nn.ModuleList(PositionwiseFeedForward(idim, hidden_units, dropout_rate, activation, bias=bias) for _ in range(n_expert))
        self.n_expert = n_expert
        self.n_expert_activated = n_expert_activated

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
        logits, selected_experts = torch.topk(router, self.n_expert_activated)
        weights = torch.nn.functional.softmax(logits, dim=1, dtype=torch.float)
        output = torch.zeros_like(xs)
        for i, expert in enumerate(self.experts):
            mask = selected_experts == i
            token_ids, ith_expert = torch.where(mask)
            output[token_ids] += weights[token_ids, ith_expert, None] * expert(xs[token_ids])
        return output.view(B, L, D)


WENET_MLP_CLASSES = {'position_wise_feed_forward': PositionwiseFeedForward, 'moe': MoEFFNLayer, 'gated': GatedVariantsMLP}


def mask_to_bias(mask: 'torch.Tensor', dtype: 'torch.dtype') ->torch.Tensor:
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask
    mask = (1.0 - mask) * -10000000000.0
    return mask


class DecoderOnly(torch.nn.Module):

    def __init__(self, n_kv_head: 'int', head_dim: 'int', hidden_size: 'int', attention_heads: 'int'=4, linear_units: 'int'=2048, num_blocks: 'int'=6, dropout_rate: 'float'=0.1, positional_dropout_rate: 'float'=0.1, attention_dropout_rate: 'float'=0.0, normalize_before: 'bool'=True, query_bias: 'bool'=False, key_bias: 'bool'=False, value_bias: 'bool'=False, mlp_bias: 'bool'=False, activation_type: 'str'='gelu', gelu_approximate: 'Union[str, None]'=None, max_position_embeding: 'int'=8192, mlp_type: 'str'='gated', layer_norm_type: 'str'='rms_norm', norm_eps: 'float'=1e-05, rms_norm_offset: 'bool'=True, selfattention_layer_type: 'str'='rope_abs_selfattn', use_sdpa: 'bool'=False, gradient_checkpointing: 'bool'=False, rope_theta: 'float'=10000.0, rope_style: 'str'='google', scale_embed: 'bool'=True) ->None:
        super().__init__()
        assert selfattention_layer_type in ['rope_abs_selfattn']
        self.pos_enc = WENET_EMB_CLASSES['rope_pos'](hidden_size, head_dim, max_len=max_position_embeding, dropout_rate=positional_dropout_rate, rope_theta=rope_theta, scale=scale_embed)
        if activation_type == 'gelu' and gelu_approximate is not None:
            activation = WENET_ACTIVATION_CLASSES['gelu'](approximate=gelu_approximate)
        else:
            activation = WENET_ACTIVATION_CLASSES[activation_type]()
        mlp_class = WENET_MLP_CLASSES[mlp_type]
        self.num_blocks = num_blocks
        self.decoders = torch.nn.ModuleList([TransformerEncoderLayer(hidden_size, WENET_ATTENTION_CLASSES[selfattention_layer_type](attention_heads, hidden_size, attention_dropout_rate, query_bias, key_bias, value_bias, use_sdpa, n_kv_head, head_dim, style=rope_style), mlp_class(hidden_size, linear_units, dropout_rate, activation, mlp_bias), dropout_rate, normalize_before, layer_norm_type=layer_norm_type, norm_eps=norm_eps, rms_norm_offset=rms_norm_offset) for _ in range(self.num_blocks)])
        self.pre_norm = normalize_before
        self.final_norm: 'Optional[torch.nn.Module]' = None
        if self.pre_norm:
            norm_class = WENET_NORM_CLASSES[layer_norm_type]
            if layer_norm_type == 'rms_norm':
                norm_class = partial(norm_class, add_unit_offset=rms_norm_offset)
            self.final_norm = norm_class(hidden_size, eps=norm_eps)
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self._hidden_size = hidden_size
        self.use_sdpa = use_sdpa
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, input: 'torch.Tensor', att_mask: 'torch.Tensor', input_position: 'Union[int, torch.Tensor]'=0, kv_caches: 'Optional[List[T_CACHE]]'=None) ->Tuple[torch.Tensor, Union[List[T_CACHE], None]]:
        xs, pos_emb = self.pos_enc(input, offset=input_position)
        if self.use_sdpa:
            att_mask = mask_to_bias(att_mask, xs.dtype)
        if self.gradient_checkpointing and self.training:
            xs = self.forward_layers_checkpointed(xs, att_mask, pos_emb)
        else:
            xs, kv_caches = self.forward_layers(xs, att_mask, pos_emb, kv_caches)
        if self.pre_norm and self.final_norm is not None:
            xs = self.final_norm(xs)
        return xs, kv_caches

    def forward_layers(self, xs: 'torch.Tensor', att_mask: 'torch.Tensor', pos_emb: 'torch.Tensor', kv_caches: 'Optional[List[T_CACHE]]'=None) ->Tuple[torch.Tensor, Union[List[T_CACHE], None]]:
        if self.training:
            for i, layer in enumerate(self.decoders):
                xs, _, _, _ = layer(xs, att_mask, pos_emb)
            new_kv_caches = kv_caches
        else:
            assert kv_caches is not None
            new_kv_caches = []
            for i, layer in enumerate(self.decoders):
                xs, _, new_kv_cache, _ = layer(xs, att_mask, pos_emb, att_cache=(kv_caches[i][0], kv_caches[i][1]))
                new_kv_caches.append(new_kv_cache)
        return xs, new_kv_caches

    @torch.jit.ignore(drop=True)
    def forward_layers_checkpointed(self, xs: 'torch.Tensor', att_mask: 'torch.Tensor', pos_emb: 'torch.Tensor') ->torch.Tensor:
        for layer in self.decoders:
            xs, _, _, _ = ckpt.checkpoint(layer.__call__, xs, att_mask, pos_emb)
        return xs

    @property
    def hidden_size(self):
        return self._hidden_size


def to_numpy(tensors):
    out = []
    if type(tensors) == torch.tensor:
        tensors = [tensors]
    for tensor in tensors:
        if tensor.requires_grad:
            tensor = tensor.detach().cpu().numpy()
        else:
            tensor = tensor.cpu().numpy()
        out.append(tensor)
    return out


class BPULayerNorm(torch.nn.Module):
    """Refactor torch.nn.LayerNorm to meet 4-D dataflow."""

    def __init__(self, module, chunk_size=8, run_on_bpu=False):
        super().__init__()
        original = copy.deepcopy(module)
        self.hidden = module.weight.size(0)
        self.chunk_size = chunk_size
        self.run_on_bpu = run_on_bpu
        if self.run_on_bpu:
            self.weight = torch.nn.Parameter(module.weight.reshape(1, self.hidden, 1, 1).repeat(1, 1, 1, chunk_size))
            self.bias = torch.nn.Parameter(module.bias.reshape(1, self.hidden, 1, 1).repeat(1, 1, 1, chunk_size))
            self.negtive = torch.nn.Parameter(torch.ones((1, self.hidden, 1, chunk_size)) * -1.0)
            self.eps = torch.nn.Parameter(torch.zeros((1, self.hidden, 1, chunk_size)) + module.eps)
            self.mean_conv_1 = torch.nn.Conv2d(self.hidden, 1, 1, bias=False)
            self.mean_conv_1.weight = torch.nn.Parameter(torch.ones(self.hidden, self.hidden, 1, 1) / (1.0 * self.hidden))
            self.mean_conv_2 = torch.nn.Conv2d(self.hidden, 1, 1, bias=False)
            self.mean_conv_2.weight = torch.nn.Parameter(torch.ones(self.hidden, self.hidden, 1, 1) / (1.0 * self.hidden))
        else:
            self.norm = module
        self.check_equal(original)

    def check_equal(self, module):
        random_data = torch.randn(1, self.chunk_size, self.hidden)
        orig_out = module(random_data)
        new_out = self.forward(random_data.transpose(1, 2).unsqueeze(2))
        np.testing.assert_allclose(to_numpy(orig_out), to_numpy(new_out.squeeze(2).transpose(1, 2)), rtol=0.01, atol=0.001)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        if self.run_on_bpu:
            u = self.mean_conv_1(x)
            numerator = x + u * self.negtive
            s = torch.pow(numerator, 2)
            s = self.mean_conv_2(s)
            denominator = torch.sqrt(s + self.eps)
            x = torch.div(numerator, denominator)
            x = x * self.weight + self.bias
        else:
            x = x.squeeze(2).transpose(1, 2).contiguous()
            x = self.norm(x)
            x = x.transpose(1, 2).contiguous().unsqueeze(2)
        return x


class BPUIdentity(torch.nn.Module):
    """Refactor torch.nn.Identity().
       For inserting BPU node whose input == output.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.identity_conv = torch.nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
        torch.nn.init.dirac_(self.identity_conv.weight.data, groups=channels)
        self.check_equal()

    def check_equal(self):
        random_data = torch.randn(1, self.channels, 1, 10)
        result = self.forward(random_data)
        np.testing.assert_allclose(to_numpy(random_data), to_numpy(result), rtol=0.01, atol=0.001)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Identity with 4-D dataflow, input == output.
        Args:
            x (torch.Tensor): (batch, in_channel, 1, time)

        Returns:
            (torch.Tensor): (batch, in_channel, 1, time).
        """
        return self.identity_conv(x)


class BPULinear(torch.nn.Module):
    """Refactor torch.nn.Linear or pointwise_conv"""

    def __init__(self, module, is_pointwise_conv=False):
        super().__init__()
        original = copy.deepcopy(module)
        self.idim = module.weight.size(1)
        self.odim = module.weight.size(0)
        self.is_pointwise_conv = is_pointwise_conv
        self.linear = torch.nn.Conv2d(self.idim, self.odim, 1, 1)
        if is_pointwise_conv:
            self.linear.weight = torch.nn.Parameter(module.weight.unsqueeze(-1))
        else:
            self.linear.weight = torch.nn.Parameter(module.weight.unsqueeze(2).unsqueeze(3))
        self.linear.bias = module.bias
        self.check_equal(original)

    def check_equal(self, module):
        random_data = torch.randn(1, 8, self.idim)
        if self.is_pointwise_conv:
            random_data = random_data.transpose(1, 2)
        original_result = module(random_data)
        if self.is_pointwise_conv:
            random_data = random_data.transpose(1, 2)
            original_result = original_result.transpose(1, 2)
        random_data = random_data.transpose(1, 2).unsqueeze(2)
        new_result = self.forward(random_data)
        np.testing.assert_allclose(to_numpy(original_result), to_numpy(new_result.squeeze(2).transpose(1, 2)), rtol=0.01, atol=0.001)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Linear with 4-D dataflow.
        Args:
            x (torch.Tensor): (batch, in_channel, 1, time)
        Returns:
            (torch.Tensor): (batch, out_channel, 1, time).
        """
        return self.linear(x)


class BPUGlobalCMVN(torch.nn.Module):
    """Refactor wenet/transformer/cmvn.py::GlobalCMVN"""

    def __init__(self, module):
        super().__init__()
        self.norm_var = module.norm_var
        self.mean = module.mean.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        self.istd = module.istd.unsqueeze(-1).unsqueeze(0).unsqueeze(0)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """CMVN with 4-D dataflow.
        Args:
            x (torch.Tensor): (batch, 1, mel_dim, time)
        Returns:
            (torch.Tensor): normalized feature with same shape.
        """
        x = x - self.mean
        if self.norm_var:
            x = x * self.istd
        return x


class BPUConv2dSubsampling8(torch.nn.Module):
    """Refactor wenet/transformer/subsampling.py::Conv2dSubsampling8

    NOTE(xcsong): Only support pos_enc_class == NoPositionalEncoding
    """

    def __init__(self, module):
        super().__init__()
        original = copy.deepcopy(module)
        self.right_context = module.right_context
        self.subsampling_rate = module.subsampling_rate
        assert isinstance(module.pos_enc, NoPositionalEncoding)
        self.conv = module.conv
        for idx in [0, 2, 4]:
            self.conv[idx].weight = torch.nn.Parameter(module.conv[idx].weight.transpose(2, 3))
        self.linear = torch.nn.ModuleList()
        odim = module.linear.weight.size(0)
        freq = module.linear.weight.size(1) // odim
        self.odim, self.freq = odim, freq
        weight = module.linear.weight.reshape(odim, odim, freq, 1)
        self.split_size = []
        num_split = (freq - 1) // 7 + 1
        slice_begin = 0
        for idx in range(num_split):
            kernel_size = min(freq, (idx + 1) * 7) - idx * 7
            conv_ele = torch.nn.Conv2d(odim, odim, (kernel_size, 1), (kernel_size, 1))
            conv_ele.weight = torch.nn.Parameter(weight[:, :, slice_begin:slice_begin + kernel_size, :])
            conv_ele.bias = torch.nn.Parameter(torch.zeros_like(conv_ele.bias))
            self.linear.append(conv_ele)
            self.split_size.append(kernel_size)
            slice_begin += kernel_size
        self.linear[0].bias = torch.nn.Parameter(module.linear.bias)
        self.check_equal(original)

    def check_equal(self, module):
        random_data = torch.randn(1, 67, 80)
        mask = torch.zeros(1, 1, 67)
        original_result, _, _ = module(random_data, mask)
        random_data = random_data.transpose(1, 2).unsqueeze(0)
        new_result = self.forward(random_data)
        np.testing.assert_allclose(to_numpy(original_result), to_numpy(new_result.squeeze(2).transpose(1, 2)), rtol=0.01, atol=0.001)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Subsample x with 4-D dataflow.
        Args:
            x (torch.Tensor): Input tensor (#batch, 1, mel_dim, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, odim, 1, time'),
                where time' = time // 8.
        """
        x = self.conv(x)
        x_out = torch.zeros(x.size(0), self.odim, 1, x.size(3))
        x = torch.split(x, self.split_size, dim=2)
        for idx, (x_part, layer) in enumerate(zip(x, self.linear)):
            x_out += layer(x_part)
        return x_out


class BPUMultiHeadedAttention(torch.nn.Module):
    """Refactor wenet/transformer/attention.py::MultiHeadedAttention

    NOTE(xcsong): Only support attention_class == MultiHeadedAttention,
        we do not consider RelPositionMultiHeadedAttention currently.
    """

    def __init__(self, module, chunk_size, left_chunks):
        super().__init__()
        original = copy.deepcopy(module)
        self.d_k = module.d_k
        self.h = module.h
        n_feat = self.d_k * self.h
        self.chunk_size = chunk_size
        self.left_chunks = left_chunks
        self.time = chunk_size * (left_chunks + 1)
        self.activation = torch.nn.Softmax(dim=-1)
        self.linear_q = BPULinear(module.linear_q)
        self.linear_k = BPULinear(module.linear_k)
        self.linear_v = BPULinear(module.linear_v)
        self.linear_out = BPULinear(module.linear_out)
        self.register_buffer('denom', torch.full((1, self.h, 1, 1), 1.0 / math.sqrt(self.d_k)))
        self.check_equal(original)

    def check_equal(self, module):
        random_data = torch.randn(1, self.chunk_size, self.d_k * self.h)
        mask = torch.ones((1, self.h, self.chunk_size, self.time), dtype=torch.bool)
        cache = torch.zeros(1, self.h, self.chunk_size * self.left_chunks, self.d_k * 2)
        original_out, original_cache = module(random_data, random_data, random_data, mask[:, 0, :, :], torch.empty(0), cache)
        random_data = random_data.transpose(1, 2).unsqueeze(2)
        cache = cache.reshape(1, self.h, self.d_k * 2, self.chunk_size * self.left_chunks)
        new_out, new_cache = self.forward(random_data, random_data, random_data, mask, cache)
        np.testing.assert_allclose(to_numpy(original_out), to_numpy(new_out.squeeze(2).transpose(1, 2)), rtol=0.01, atol=0.001)
        np.testing.assert_allclose(to_numpy(original_cache), to_numpy(new_cache.transpose(2, 3)), rtol=0.01, atol=0.001)

    def forward(self, q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', mask: 'torch.Tensor', cache: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot product attention.

        Args:
            q (torch.Tensor): Query tensor (#batch, size, 1, chunk_size).
            k (torch.Tensor): Key tensor (#batch, size, 1, chunk_size).
            v (torch.Tensor): Value tensor (#batch, size, 1, chunk_size).
            mask (torch.Tensor): Mask tensor,
                (#batch, head, chunk_size, cache_t + chunk_size).
            cache (torch.Tensor): Cache tensor
                (1, head, d_k * 2, cache_t),
                where `cache_t == chunk_size * left_chunks`.


        Returns:
            torch.Tensor: Output tensor (#batch, size, 1, chunk_size).
            torch.Tensor: Cache tensor
                (1, head, d_k * 2, cache_t + chunk_size)
                where `cache_t == chunk_size * left_chunks`
        """
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)
        q = q.view(1, self.h, self.d_k, self.chunk_size)
        k = k.view(1, self.h, self.d_k, self.chunk_size)
        v = v.view(1, self.h, self.d_k, self.chunk_size)
        q = q.transpose(2, 3)
        k_cache, v_cache = torch.split(cache, cache.size(2) // 2, dim=2)
        k = torch.cat((k_cache, k), dim=3)
        v = torch.cat((v_cache, v), dim=3)
        new_cache = torch.cat((k, v), dim=2)
        scores = torch.matmul(q, k) * self.denom
        mask = mask.eq(0)
        scores = scores.masked_fill(mask, -float('inf'))
        attn = self.activation(scores).masked_fill(mask, 0.0)
        attn = attn.transpose(2, 3)
        x = torch.matmul(v, attn)
        x = x.view(1, self.d_k * self.h, 1, self.chunk_size)
        x_out = self.linear_out(x)
        return x_out, new_cache


class BPUConvolution(torch.nn.Module):
    """Refactor wenet/transformer/convolution.py::ConvolutionModule

    NOTE(xcsong): Only suport use_layer_norm == False
    """

    def __init__(self, module):
        super().__init__()
        original = copy.deepcopy(module)
        self.lorder = module.lorder
        self.use_layer_norm = False
        self.activation = module.activation
        channels = module.pointwise_conv1.weight.size(1)
        self.channels = channels
        kernel_size = module.depthwise_conv.weight.size(2)
        assert module.use_layer_norm is False
        self.pointwise_conv1 = BPULinear(module.pointwise_conv1, True)
        self.depthwise_conv = torch.nn.Conv2d(channels, channels, (1, kernel_size), stride=1, groups=channels)
        self.depthwise_conv.weight = torch.nn.Parameter(module.depthwise_conv.weight.unsqueeze(-2))
        self.depthwise_conv.bias = torch.nn.Parameter(module.depthwise_conv.bias)
        self.norm = torch.nn.BatchNorm2d(channels)
        self.norm.training = False
        self.norm.num_features = module.norm.num_features
        self.norm.eps = module.norm.eps
        self.norm.momentum = module.norm.momentum
        self.norm.weight = torch.nn.Parameter(module.norm.weight)
        self.norm.bias = torch.nn.Parameter(module.norm.bias)
        self.norm.running_mean = module.norm.running_mean
        self.norm.running_var = module.norm.running_var
        self.pointwise_conv2 = BPULinear(module.pointwise_conv2, True)
        self.identity = BPUIdentity(channels)
        self.check_equal(original)

    def check_equal(self, module):
        random_data = torch.randn(1, 8, self.channels)
        cache = torch.zeros((1, self.channels, self.lorder))
        original_out, original_cache = module(random_data, cache=cache)
        random_data = random_data.transpose(1, 2).unsqueeze(2)
        cache = cache.unsqueeze(2)
        new_out, new_cache = self.forward(random_data, cache)
        np.testing.assert_allclose(to_numpy(original_out), to_numpy(new_out.squeeze(2).transpose(1, 2)), rtol=0.01, atol=0.001)
        np.testing.assert_allclose(to_numpy(original_cache), to_numpy(new_cache.squeeze(2)), rtol=0.01, atol=0.001)

    def forward(self, x: 'torch.Tensor', cache: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, channels, 1, chunk_size).
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (#batch, channels, 1, cache_t).
        Returns:
            torch.Tensor: Output tensor (#batch, channels, 1, chunk_size).
            torch.Tensor: Cache tensor (#batch, channels, 1, cache_t).
        """
        x = torch.cat((self.identity(cache), self.identity(x)), dim=3)
        new_cache = x[:, :, :, -self.lorder:]
        x = self.pointwise_conv1(x)
        x = torch.nn.functional.glu(x, dim=1)
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))
        x = self.pointwise_conv2(x)
        return x, new_cache


class BPUFFN(torch.nn.Module):
    """Refactor wenet/transformer/positionwise_feed_forward.py::PositionwiseFeedForward
    """

    def __init__(self, module):
        super().__init__()
        original = copy.deepcopy(module)
        self.activation = module.activation
        self.w_1 = BPULinear(module.w_1)
        self.w_2 = BPULinear(module.w_2)
        self.check_equal(original)

    def check_equal(self, module):
        random_data = torch.randn(1, 8, self.w_1.idim)
        original_out = module(random_data)
        random_data = random_data.transpose(1, 2).unsqueeze(2)
        new_out = self.forward(random_data)
        np.testing.assert_allclose(to_numpy(original_out), to_numpy(new_out.squeeze(2).transpose(1, 2)), rtol=0.01, atol=0.001)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, D, 1, L)
        Returns:
            output tensor, (B, D, 1, L)
        """
        return self.w_2(self.activation(self.w_1(x)))


class BPUConformerEncoderLayer(torch.nn.Module):
    """Refactor wenet/transformer/encoder_layer.py::ConformerEncoderLayer
    """

    def __init__(self, module, chunk_size, left_chunks, ln_run_on_bpu=False):
        super().__init__()
        original = copy.deepcopy(module)
        self.size = module.size
        assert module.normalize_before is True
        assert module.concat_after is False
        self.feed_forward_macaron = BPUFFN(module.feed_forward_macaron)
        self.self_attn = BPUMultiHeadedAttention(module.self_attn, chunk_size, left_chunks)
        self.conv_module = BPUConvolution(module.conv_module)
        self.feed_forward = BPUFFN(module.feed_forward)
        self.norm_ff = BPULayerNorm(module.norm_ff, chunk_size, ln_run_on_bpu)
        self.norm_mha = BPULayerNorm(module.norm_mha, chunk_size, ln_run_on_bpu)
        self.norm_ff_macron = BPULayerNorm(module.norm_ff_macaron, chunk_size, ln_run_on_bpu)
        self.norm_conv = BPULayerNorm(module.norm_conv, chunk_size, ln_run_on_bpu)
        self.norm_final = BPULayerNorm(module.norm_final, chunk_size, ln_run_on_bpu)
        self.register_buffer('ff_scale', torch.full((1, self.size, 1, 1), module.ff_scale))
        self.check_equal(original)

    def check_equal(self, module):
        time1 = self.self_attn.chunk_size
        time2 = self.self_attn.time
        h, d_k = self.self_attn.h, self.self_attn.d_k
        random_x = torch.randn(1, time1, self.size)
        att_mask = torch.ones(1, h, time1, time2)
        att_cache = torch.zeros(1, h, time2 - time1, d_k * 2)
        cnn_cache = torch.zeros(1, self.size, self.conv_module.lorder)
        original_x, _, original_att_cache, original_cnn_cache = module(random_x, att_mask[:, 0, :, :], torch.empty(0), att_cache=att_cache, cnn_cache=cnn_cache)
        random_x = random_x.transpose(1, 2).unsqueeze(2)
        att_cache = att_cache.reshape(1, h, d_k * 2, time2 - time1)
        cnn_cache = cnn_cache.unsqueeze(2)
        new_x, new_att_cache, new_cnn_cache = self.forward(random_x, att_mask, att_cache, cnn_cache)
        np.testing.assert_allclose(to_numpy(original_att_cache), to_numpy(new_att_cache.transpose(2, 3)), rtol=0.01, atol=0.001)
        np.testing.assert_allclose(to_numpy(original_x), to_numpy(new_x.squeeze(2).transpose(1, 2)), rtol=0.01, atol=0.001)
        np.testing.assert_allclose(to_numpy(original_cnn_cache), to_numpy(new_cnn_cache.squeeze(2)), rtol=0.01, atol=0.001)

    def forward(self, x: 'torch.Tensor', att_mask: 'torch.Tensor', att_cache: 'torch.Tensor', cnn_cache: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, size, 1, chunk_size)
            att_mask (torch.Tensor): Mask tensor for the input
                (#batch, head, chunk_size, cache_t1 + chunk_size),
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, d_k * 2, cache_t1), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
                (#batch=1, size, 1, cache_t2)
        Returns:
            torch.Tensor: Output tensor (#batch, size, 1, chunk_size).
            torch.Tensor: att_cache tensor,
                (1, head, d_k * 2, cache_t1 + chunk_size).
            torch.Tensor: cnn_cahce tensor (#batch, size, 1, cache_t2).
        """
        residual = x
        x = self.norm_ff_macron(x)
        x = residual + self.ff_scale * self.feed_forward_macaron(x)
        residual = x
        x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(x, x, x, att_mask, att_cache)
        x = residual + x_att
        residual = x
        x = self.norm_conv(x)
        x, new_cnn_cache = self.conv_module(x, cnn_cache)
        x = residual + x
        residual = x
        x = self.norm_ff(x)
        x = residual + self.ff_scale * self.feed_forward(x)
        x = self.norm_final(x)
        return x, new_att_cache, new_cnn_cache


class BPUConformerEncoder(torch.nn.Module):
    """Refactor wenet/transformer/encoder.py::ConformerEncoder
    """

    def __init__(self, module, chunk_size, left_chunks, ln_run_on_bpu=False):
        super().__init__()
        original = copy.deepcopy(module)
        output_size = module.output_size()
        self._output_size = module.output_size()
        self.after_norm = module.after_norm
        self.chunk_size = chunk_size
        self.left_chunks = left_chunks
        self.head = module.encoders[0].self_attn.h
        self.layers = len(module.encoders)
        self.global_cmvn = BPUGlobalCMVN(module.global_cmvn)
        self.embed = BPUConv2dSubsampling8(module.embed)
        self.encoders = torch.nn.ModuleList()
        for layer in module.encoders:
            self.encoders.append(BPUConformerEncoderLayer(layer, chunk_size, left_chunks, ln_run_on_bpu))
        self.identity_cnncache = BPUIdentity(output_size)
        self.check_equal(original)

    def check_equal(self, module):
        time1 = self.encoders[0].self_attn.chunk_size
        time2 = self.encoders[0].self_attn.time
        layers = self.layers
        h, d_k = self.head, self.encoders[0].self_attn.d_k
        decoding_window = (self.chunk_size - 1) * module.embed.subsampling_rate + module.embed.right_context + 1
        lorder = self.encoders[0].conv_module.lorder
        random_x = torch.randn(1, decoding_window, 80)
        att_mask = torch.ones(1, h, time1, time2)
        att_cache = torch.zeros(layers, h, time2 - time1, d_k * 2)
        cnn_cache = torch.zeros(layers, 1, self._output_size, lorder)
        orig_x, orig_att_cache, orig_cnn_cache = module.forward_chunk(random_x, 0, time2 - time1, att_mask=att_mask[:, 0, :, :], att_cache=att_cache, cnn_cache=cnn_cache)
        random_x = random_x.unsqueeze(0)
        att_cache = att_cache.reshape(1, h * layers, d_k * 2, time2 - time1)
        cnn_cache = cnn_cache.reshape(1, self._output_size, layers, lorder)
        new_x, new_att_cache, new_cnn_cache = self.forward(random_x, att_cache, cnn_cache, att_mask)
        caches = torch.split(new_att_cache, h, dim=1)
        caches = [c.transpose(2, 3) for c in caches]
        np.testing.assert_allclose(to_numpy(orig_att_cache), to_numpy(torch.cat(caches, dim=0)), rtol=0.01, atol=0.001)
        np.testing.assert_allclose(to_numpy(orig_x), to_numpy(new_x.squeeze(2).transpose(1, 2)), rtol=0.01, atol=0.001)
        np.testing.assert_allclose(to_numpy(orig_cnn_cache), to_numpy(new_cnn_cache.transpose(0, 2).transpose(1, 2)), rtol=0.01, atol=0.001)

    def forward(self, xs: 'torch.Tensor', att_cache: 'torch.Tensor', cnn_cache: 'torch.Tensor', att_mask: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Forward just one chunk

        Args:
            xs (torch.Tensor): chunk input, with shape (b=1, 1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate +                         subsample.right_context + 1`
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (1, head * elayers, d_k * 2, cache_t1), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (1, hidden-dim, elayers, cache_t2), where
                `cache_t2 == cnn.lorder - 1`
            att_mask (torch.Tensor): Mask tensor for the input
                (#batch, head, chunk_size, cache_t1 + chunk_size),

        Returns:
            torch.Tensor: output of current input xs,
                with shape (b=1, hidden-dim, 1, chunk_size).
            torch.Tensor: new attention cache required for next chunk, with
                same shape as the original att_cache.
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.
        """
        xs = xs.transpose(2, 3)
        xs = self.global_cmvn(xs)
        xs = self.embed(xs)
        att_cache = torch.split(att_cache, self.head, dim=1)
        cnn_cache = self.identity_cnncache(cnn_cache)
        cnn_cache = torch.split(cnn_cache, 1, dim=2)
        r_att_cache = []
        r_cnn_cache = []
        for i, layer in enumerate(self.encoders):
            xs, new_att_cache, new_cnn_cache = layer(xs, att_mask, att_cache=att_cache[i], cnn_cache=cnn_cache[i])
            r_att_cache.append(new_att_cache[:, :, :, self.chunk_size:])
            r_cnn_cache.append(new_cnn_cache)
        r_att_cache = torch.cat(r_att_cache, dim=1)
        r_cnn_cache = self.identity_cnncache(torch.cat(r_cnn_cache, dim=2))
        xs = xs.squeeze(2).transpose(1, 2).contiguous()
        xs = self.after_norm(xs)
        xs = xs.transpose(1, 2).contiguous().unsqueeze(2)
        return xs, r_att_cache, r_cnn_cache


class BPUCTC(torch.nn.Module):
    """Refactor wenet/transformer/ctc.py::CTC
    """

    def __init__(self, module):
        super().__init__()
        original = copy.deepcopy(module)
        self.idim = module.ctc_lo.weight.size(1)
        num_class = module.ctc_lo.weight.size(0)
        self.ctc_lo = torch.nn.ModuleList()
        self.split_size = []
        num_split = (num_class - 1) // 2048 + 1
        for idx in range(num_split):
            out_channel = min(num_class, (idx + 1) * 2048) - idx * 2048
            conv_ele = torch.nn.Conv2d(self.idim, out_channel, 1, 1)
            self.ctc_lo.append(conv_ele)
            self.split_size.append(out_channel)
        orig_weight = torch.split(module.ctc_lo.weight, self.split_size, dim=0)
        orig_bias = torch.split(module.ctc_lo.bias, self.split_size, dim=0)
        for i, (w, b) in enumerate(zip(orig_weight, orig_bias)):
            w = w.unsqueeze(2).unsqueeze(3)
            self.ctc_lo[i].weight = torch.nn.Parameter(w)
            self.ctc_lo[i].bias = torch.nn.Parameter(b)
        self.check_equal(original)

    def check_equal(self, module):
        random_data = torch.randn(1, 100, self.idim)
        original_result = module.ctc_lo(random_data)
        random_data = random_data.transpose(1, 2).unsqueeze(2)
        new_result = self.forward(random_data)
        np.testing.assert_allclose(to_numpy(original_result), to_numpy(new_result.squeeze(2).transpose(1, 2)), rtol=0.01, atol=0.001)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """frame activations, without softmax.

        Args:
            Tensor x: 4d tensor (B, hidden_dim, 1, chunk_size)
        Returns:
            torch.Tensor: (B, num_class, 1, chunk_size)
        """
        out = []
        for i, layer in enumerate(self.ctc_lo):
            out.append(layer(x))
        out = torch.cat(out, dim=1)
        return out


class Encoder(torch.nn.Module):

    def __init__(self, encoder: 'BaseEncoder', ctc: 'CTC', beam_size: 'int'=10):
        super().__init__()
        self.encoder = encoder
        self.ctc = ctc
        self.beam_size = beam_size

    def forward(self, speech: 'torch.Tensor', speech_lengths: 'torch.Tensor'):
        """Encoder
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        Returns:
            encoder_out: B x T x F
            encoder_out_lens: B
            ctc_log_probs: B x T x V
            beam_log_probs: B x T x beam_size
            beam_log_probs_idx: B x T x beam_size
        """
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths, -1, -1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        ctc_log_probs = self.ctc.log_softmax(encoder_out)
        encoder_out_lens = encoder_out_lens.int()
        beam_log_probs, beam_log_probs_idx = torch.topk(ctc_log_probs, self.beam_size, dim=2)
        return encoder_out, encoder_out_lens, ctc_log_probs, beam_log_probs, beam_log_probs_idx


class StreamingEncoder(torch.nn.Module):

    def __init__(self, model, required_cache_size, beam_size, transformer=False, return_ctc_logprobs=False):
        super().__init__()
        self.ctc = model.ctc
        self.subsampling_rate = model.encoder.embed.subsampling_rate
        self.embed = model.encoder.embed
        self.global_cmvn = model.encoder.global_cmvn
        self.required_cache_size = required_cache_size
        self.beam_size = beam_size
        self.encoder = model.encoder
        self.transformer = transformer
        self.return_ctc_logprobs = return_ctc_logprobs

    def forward(self, chunk_xs, chunk_lens, offset, att_cache, cnn_cache, cache_mask):
        """Streaming Encoder
        Args:
            xs (torch.Tensor): chunk input, with shape (b, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate +                         subsample.right_context + 1`
            offset (torch.Tensor): offset with shape (b, 1)
                        1 is retained for triton deployment
            required_cache_size (int): cache size required for next chunk
                compuation
                > 0: actual cache size
                <= 0: not allowed in streaming gpu encoder                   `
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (b, elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (b, elayers, b, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`
            cache_mask: (torch.Tensor): cache mask with shape (b, required_cache_size)
                 in a batch of request, each request may have different
                 history cache. Cache mask is used to indidate the effective
                 cache for each request
        Returns:
            torch.Tensor: log probabilities of ctc output and cutoff by beam size
                with shape (b, chunk_size, beam)
            torch.Tensor: index of top beam size probabilities for each timestep
                with shape (b, chunk_size, beam)
            torch.Tensor: output of current input xs,
                with shape (b, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                same shape (b, elayers, head, cache_t1, d_k * 2)
                as the original att_cache
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.
            torch.Tensor: new cache mask, with same shape as the original
                cache mask
        """
        offset = offset.squeeze(1)
        T = chunk_xs.size(1)
        chunk_mask = ~make_pad_mask(chunk_lens, T).unsqueeze(1)
        chunk_mask = chunk_mask
        att_cache = torch.transpose(att_cache, 0, 1)
        cnn_cache = torch.transpose(cnn_cache, 0, 1)
        xs = self.global_cmvn(chunk_xs)
        xs, pos_emb, chunk_mask = self.embed(xs, chunk_mask, offset)
        cache_size = att_cache.size(3)
        masks = torch.cat((cache_mask, chunk_mask), dim=2)
        index = offset - cache_size
        pos_emb = self.embed.position_encoding(index, cache_size + xs.size(1))
        pos_emb = pos_emb
        next_cache_start = -self.required_cache_size
        r_cache_mask = masks[:, :, next_cache_start:]
        r_att_cache = []
        r_cnn_cache = []
        for i, layer in enumerate(self.encoder.encoders):
            i_kv_cache = att_cache[i]
            size = att_cache.size(-1) // 2
            kv_cache = i_kv_cache[:, :, :, :size], i_kv_cache[:, :, :, size:]
            xs, _, new_kv_cache, new_cnn_cache = layer(xs, masks, pos_emb, att_cache=kv_cache, cnn_cache=cnn_cache[i])
            new_att_cache = torch.cat(new_kv_cache, dim=-1)
            r_att_cache.append(new_att_cache[:, :, next_cache_start:, :].unsqueeze(1))
            if not self.transformer:
                r_cnn_cache.append(new_cnn_cache.unsqueeze(1))
        if self.encoder.normalize_before:
            chunk_out = self.encoder.after_norm(xs)
        else:
            chunk_out = xs
        r_att_cache = torch.cat(r_att_cache, dim=1)
        if not self.transformer:
            r_cnn_cache = torch.cat(r_cnn_cache, dim=1)
        log_ctc_probs = self.ctc.log_softmax(chunk_out)
        log_probs, log_probs_idx = torch.topk(log_ctc_probs, self.beam_size, dim=2)
        log_probs = log_probs
        r_offset = offset + chunk_out.shape[1]
        chunk_out_lens = chunk_lens // self.subsampling_rate
        r_offset = r_offset.unsqueeze(1)
        if self.return_ctc_logprobs:
            return log_ctc_probs, chunk_out, chunk_out_lens, r_offset, r_att_cache, r_cnn_cache, r_cache_mask
        else:
            return log_probs, log_probs_idx, chunk_out, chunk_out_lens, r_offset, r_att_cache, r_cnn_cache, r_cache_mask


class StreamingSqueezeformerEncoder(torch.nn.Module):

    def __init__(self, model, required_cache_size, beam_size):
        super().__init__()
        self.ctc = model.ctc
        self.subsampling_rate = model.encoder.embed.subsampling_rate
        self.embed = model.encoder.embed
        self.global_cmvn = model.encoder.global_cmvn
        self.required_cache_size = required_cache_size
        self.beam_size = beam_size
        self.encoder = model.encoder
        self.reduce_idx = model.encoder.reduce_idx
        self.recover_idx = model.encoder.recover_idx
        if self.reduce_idx is None:
            self.time_reduce = None
        elif self.recover_idx is None:
            self.time_reduce = 'normal'
        else:
            self.time_reduce = 'recover'
            assert len(self.reduce_idx) == len(self.recover_idx)

    def calculate_downsampling_factor(self, i: 'int') ->int:
        if self.reduce_idx is None:
            return 1
        else:
            reduce_exp, recover_exp = 0, 0
            for exp, rd_idx in enumerate(self.reduce_idx):
                if i >= rd_idx:
                    reduce_exp = exp + 1
            if self.recover_idx is not None:
                for exp, rc_idx in enumerate(self.recover_idx):
                    if i >= rc_idx:
                        recover_exp = exp + 1
            return int(2 ** (reduce_exp - recover_exp))

    def forward(self, chunk_xs, chunk_lens, offset, att_cache, cnn_cache, cache_mask):
        """Streaming Encoder
        Args:
            xs (torch.Tensor): chunk input, with shape (b, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate +                         subsample.right_context + 1`
            offset (torch.Tensor): offset with shape (b, 1)
                        1 is retained for triton deployment
            required_cache_size (int): cache size required for next chunk
                compuation
                > 0: actual cache size
                <= 0: not allowed in streaming gpu encoder                   `
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (b, elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (b, elayers, b, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`
            cache_mask: (torch.Tensor): cache mask with shape (b, required_cache_size)
                 in a batch of request, each request may have different
                 history cache. Cache mask is used to indidate the effective
                 cache for each request
        Returns:
            torch.Tensor: log probabilities of ctc output and cutoff by beam size
                with shape (b, chunk_size, beam)
            torch.Tensor: index of top beam size probabilities for each timestep
                with shape (b, chunk_size, beam)
            torch.Tensor: output of current input xs,
                with shape (b, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                same shape (b, elayers, head, cache_t1, d_k * 2)
                as the original att_cache
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.
            torch.Tensor: new cache mask, with same shape as the original
                cache mask
        """
        offset = offset.squeeze(1)
        T = chunk_xs.size(1)
        chunk_mask = ~make_pad_mask(chunk_lens, T).unsqueeze(1)
        chunk_mask = chunk_mask
        att_cache = torch.transpose(att_cache, 0, 1)
        cnn_cache = torch.transpose(cnn_cache, 0, 1)
        xs = self.global_cmvn(chunk_xs)
        xs, pos_emb, chunk_mask = self.embed(xs, chunk_mask, offset)
        elayers, cache_size = att_cache.size(0), att_cache.size(3)
        att_mask = torch.cat((cache_mask, chunk_mask), dim=2)
        index = offset - cache_size
        pos_emb = self.embed.position_encoding(index, cache_size + xs.size(1))
        pos_emb = pos_emb
        next_cache_start = -self.required_cache_size
        r_cache_mask = att_mask[:, :, next_cache_start:]
        r_att_cache = []
        r_cnn_cache = []
        mask_pad = torch.ones(1, xs.size(1), device=xs.device, dtype=torch.bool)
        mask_pad = mask_pad.unsqueeze(1)
        max_att_len: 'int' = 0
        recover_activations: 'List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]' = []
        index = 0
        xs_lens = torch.tensor([xs.size(1)], device=xs.device, dtype=torch.int)
        xs = self.encoder.preln(xs)
        for i, layer in enumerate(self.encoder.encoders):
            if self.reduce_idx is not None:
                if self.time_reduce is not None and i in self.reduce_idx:
                    recover_activations.append((xs, att_mask, pos_emb, mask_pad))
                    xs, xs_lens, att_mask, mask_pad = self.encoder.time_reduction_layer(xs, xs_lens, att_mask, mask_pad)
                    pos_emb = pos_emb[:, ::2, :]
                    if self.encoder.pos_enc_layer_type == 'rel_pos_repaired':
                        pos_emb = pos_emb[:, :xs.size(1) * 2 - 1, :]
                    index += 1
            if self.recover_idx is not None:
                if self.time_reduce == 'recover' and i in self.recover_idx:
                    index -= 1
                    recover_tensor, recover_att_mask, recover_pos_emb, recover_mask_pad = recover_activations[index]
                    xs = xs.unsqueeze(2).repeat(1, 1, 2, 1).flatten(1, 2)
                    xs = self.encoder.time_recover_layer(xs)
                    recoverd_t = recover_tensor.size(1)
                    xs = recover_tensor + xs[:, :recoverd_t, :].contiguous()
                    att_mask = recover_att_mask
                    pos_emb = recover_pos_emb
                    mask_pad = recover_mask_pad
            factor = self.calculate_downsampling_factor(i)
            xs, _, new_att_cache, new_cnn_cache = layer(xs, att_mask, pos_emb, att_cache=att_cache[i][:, :, ::factor, :][:, :, :pos_emb.size(1) - xs.size(1), :] if elayers > 0 else att_cache[:, :, ::factor, :], cnn_cache=cnn_cache[i] if cnn_cache.size(0) > 0 else cnn_cache)
            cached_att = new_att_cache[:, :, next_cache_start // factor:, :]
            cached_cnn = new_cnn_cache.unsqueeze(1)
            cached_att = cached_att.unsqueeze(3).repeat(1, 1, 1, factor, 1).flatten(2, 3)
            if i == 0:
                max_att_len = cached_att.size(2)
            r_att_cache.append(cached_att[:, :, :max_att_len, :].unsqueeze(1))
            r_cnn_cache.append(cached_cnn)
        chunk_out = xs
        r_att_cache = torch.cat(r_att_cache, dim=1)
        r_cnn_cache = torch.cat(r_cnn_cache, dim=1)
        log_ctc_probs = self.ctc.log_softmax(chunk_out)
        log_probs, log_probs_idx = torch.topk(log_ctc_probs, self.beam_size, dim=2)
        log_probs = log_probs
        r_offset = offset + chunk_out.shape[1]
        chunk_out_lens = chunk_lens // self.subsampling_rate
        r_offset = r_offset.unsqueeze(1)
        return log_probs, log_probs_idx, chunk_out, chunk_out_lens, r_offset, r_att_cache, r_cnn_cache, r_cache_mask


class StreamingEfficientConformerEncoder(torch.nn.Module):

    def __init__(self, model, required_cache_size, beam_size):
        super().__init__()
        self.ctc = model.ctc
        self.subsampling_rate = model.encoder.embed.subsampling_rate
        self.embed = model.encoder.embed
        self.global_cmvn = model.encoder.global_cmvn
        self.required_cache_size = required_cache_size
        self.beam_size = beam_size
        self.encoder = model.encoder
        self.stride_layer_idx = model.encoder.stride_layer_idx
        self.stride = model.encoder.stride
        self.num_blocks = model.encoder.num_blocks
        self.cnn_module_kernel = model.encoder.cnn_module_kernel

    def calculate_downsampling_factor(self, i: 'int') ->int:
        factor = 1
        for idx, stride_idx in enumerate(self.stride_layer_idx):
            if i > stride_idx:
                factor *= self.stride[idx]
        return factor

    def forward(self, chunk_xs, chunk_lens, offset, att_cache, cnn_cache, cache_mask):
        """Streaming Encoder
        Args:
            chunk_xs (torch.Tensor): chunk input, with shape (b, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate +                         subsample.right_context + 1`
            chunk_lens (torch.Tensor):
            offset (torch.Tensor): offset with shape (b, 1)
                        1 is retained for triton deployment
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (b, elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (b, elayers, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`
            cache_mask: (torch.Tensor): cache mask with shape (b, required_cache_size)
                 in a batch of request, each request may have different
                 history cache. Cache mask is used to indidate the effective
                 cache for each request
        Returns:
            torch.Tensor: log probabilities of ctc output and cutoff by beam size
                with shape (b, chunk_size, beam)
            torch.Tensor: index of top beam size probabilities for each timestep
                with shape (b, chunk_size, beam)
            torch.Tensor: output of current input xs,
                with shape (b, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                same shape (b, elayers, head, cache_t1, d_k * 2)
                as the original att_cache
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.
            torch.Tensor: new cache mask, with same shape as the original
                cache mask
        """
        offset = offset.squeeze(1)
        offset *= self.calculate_downsampling_factor(self.num_blocks + 1)
        T = chunk_xs.size(1)
        chunk_mask = ~make_pad_mask(chunk_lens, T).unsqueeze(1)
        chunk_mask = chunk_mask
        att_cache = torch.transpose(att_cache, 0, 1)
        cnn_cache = torch.transpose(cnn_cache, 0, 1)
        xs = self.global_cmvn(chunk_xs)
        xs, pos_emb, chunk_mask = self.embed(xs, chunk_mask, offset)
        cache_size = att_cache.size(3)
        masks = torch.cat((cache_mask, chunk_mask), dim=2)
        att_mask = torch.cat((cache_mask, chunk_mask), dim=2)
        index = offset - cache_size
        pos_emb = self.embed.position_encoding(index, cache_size + xs.size(1))
        pos_emb = pos_emb
        next_cache_start = -self.required_cache_size
        r_cache_mask = masks[:, :, next_cache_start:]
        r_att_cache = []
        r_cnn_cache = []
        mask_pad = chunk_mask
        max_att_len, max_cnn_len = 0, 0
        for i, layer in enumerate(self.encoder.encoders):
            factor = self.calculate_downsampling_factor(i)
            att_cache_trunc = 0
            if xs.size(1) + att_cache.size(3) / factor > pos_emb.size(1):
                att_cache_trunc = xs.size(1) + att_cache.size(3) // factor - pos_emb.size(1) + 1
            xs, _, new_att_cache, new_cnn_cache = layer(xs, att_mask, pos_emb, mask_pad=mask_pad, att_cache=att_cache[i][:, :, ::factor, :][:, :, att_cache_trunc:, :], cnn_cache=cnn_cache[i, :, :, :] if cnn_cache.size(0) > 0 else cnn_cache)
            if i in self.stride_layer_idx:
                efficient_index = self.stride_layer_idx.index(i)
                att_mask = att_mask[:, ::self.stride[efficient_index], ::self.stride[efficient_index]]
                mask_pad = mask_pad[:, ::self.stride[efficient_index], ::self.stride[efficient_index]]
                pos_emb = pos_emb[:, ::self.stride[efficient_index], :]
            new_att_cache = new_att_cache[:, :, next_cache_start // factor:, :]
            new_cnn_cache = new_cnn_cache.unsqueeze(1)
            new_att_cache = new_att_cache.unsqueeze(3).repeat(1, 1, 1, factor, 1).flatten(2, 3)
            new_cnn_cache = F.pad(new_cnn_cache, (self.cnn_module_kernel - 1 - new_cnn_cache.size(3), 0))
            if i == 0:
                max_att_len = new_att_cache.size(2)
                max_cnn_len = new_cnn_cache.size(3)
            r_att_cache.append(new_att_cache[:, :, -max_att_len:, :].unsqueeze(1))
            r_cnn_cache.append(new_cnn_cache[:, :, :, -max_cnn_len:])
        if self.encoder.normalize_before:
            chunk_out = self.encoder.after_norm(xs)
        else:
            chunk_out = xs
        r_att_cache = torch.cat(r_att_cache, dim=1)
        r_cnn_cache = torch.cat(r_cnn_cache, dim=1)
        log_ctc_probs = self.ctc.log_softmax(chunk_out)
        log_probs, log_probs_idx = torch.topk(log_ctc_probs, self.beam_size, dim=2)
        log_probs = log_probs
        r_offset = offset + chunk_out.shape[1]
        chunk_out_lens = chunk_lens // self.subsampling_rate // self.calculate_downsampling_factor(self.num_blocks + 1)
        chunk_out_lens += 1
        r_offset = r_offset.unsqueeze(1)
        return log_probs, log_probs_idx, chunk_out, chunk_out_lens, r_offset, r_att_cache, r_cnn_cache, r_cache_mask


class Decoder(torch.nn.Module):

    def __init__(self, decoder: 'TransformerDecoder', ctc_weight: 'float'=0.5, reverse_weight: 'float'=0.0, beam_size: 'int'=10, decoder_fastertransformer: 'bool'=False):
        super().__init__()
        self.decoder = decoder
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight
        self.beam_size = beam_size
        self.decoder_fastertransformer = decoder_fastertransformer

    def forward(self, encoder_out: 'torch.Tensor', encoder_lens: 'torch.Tensor', hyps_pad_sos_eos: 'torch.Tensor', hyps_lens_sos: 'torch.Tensor', r_hyps_pad_sos_eos: 'torch.Tensor', ctc_score: 'torch.Tensor'):
        """Encoder
        Args:
            encoder_out: B x T x F
            encoder_lens: B
            hyps_pad_sos_eos: B x beam x (T2+1),
                        hyps with sos & eos and padded by ignore id
            hyps_lens_sos: B x beam, length for each hyp with sos
            r_hyps_pad_sos_eos: B x beam x (T2+1),
                    reversed hyps with sos & eos and padded by ignore id
            ctc_score: B x beam, ctc score for each hyp
        Returns:
            decoder_out: B x beam x T2 x V
            r_decoder_out: B x beam x T2 x V
            best_index: B
        """
        B, T, F = encoder_out.shape
        bz = self.beam_size
        B2 = B * bz
        encoder_out = encoder_out.repeat(1, bz, 1).view(B2, T, F)
        encoder_mask = ~make_pad_mask(encoder_lens, T).unsqueeze(1)
        encoder_mask = encoder_mask.repeat(1, bz, 1).view(B2, 1, T)
        T2 = hyps_pad_sos_eos.shape[2] - 1
        hyps_pad = hyps_pad_sos_eos.view(B2, T2 + 1)
        hyps_lens = hyps_lens_sos.view(B2)
        hyps_pad_sos = hyps_pad[:, :-1].contiguous()
        hyps_pad_eos = hyps_pad[:, 1:].contiguous()
        r_hyps_pad = r_hyps_pad_sos_eos.view(B2, T2 + 1)
        r_hyps_pad_sos = r_hyps_pad[:, :-1].contiguous()
        r_hyps_pad_eos = r_hyps_pad[:, 1:].contiguous()
        decoder_out, r_decoder_out, _ = self.decoder(encoder_out, encoder_mask, hyps_pad_sos, hyps_lens, r_hyps_pad_sos, self.reverse_weight)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        V = decoder_out.shape[-1]
        decoder_out = decoder_out.view(B2, T2, V)
        mask = ~make_pad_mask(hyps_lens, T2)
        index = torch.unsqueeze(hyps_pad_eos * mask, 2)
        score = decoder_out.gather(2, index).squeeze(2)
        score = score * mask
        decoder_out = decoder_out.view(B, bz, T2, V)
        if self.reverse_weight > 0:
            r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
            r_decoder_out = r_decoder_out.view(B2, T2, V)
            index = torch.unsqueeze(r_hyps_pad_eos * mask, 2)
            r_score = r_decoder_out.gather(2, index).squeeze(2)
            r_score = r_score * mask
            score = score * (1 - self.reverse_weight) + self.reverse_weight * r_score
            r_decoder_out = r_decoder_out.view(B, bz, T2, V)
        score = torch.sum(score, axis=1)
        score = torch.reshape(score, (B, bz)) + self.ctc_weight * ctc_score
        best_index = torch.argmax(score, dim=1)
        if self.decoder_fastertransformer:
            return decoder_out, best_index
        else:
            return best_index


class ConvolutionalSpatialGatingUnit(torch.nn.Module):
    """Convolutional Spatial Gating Unit (CSGU)."""

    def __init__(self, size: 'int', kernel_size: 'int', dropout_rate: 'float', use_linear_after_conv: 'bool', gate_activation: 'str', causal: 'bool'=True):
        super().__init__()
        n_channels = size // 2
        self.norm = nn.LayerNorm(n_channels)
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
        self.conv = torch.nn.Conv1d(n_channels, n_channels, kernel_size, 1, padding, groups=n_channels)
        if use_linear_after_conv:
            self.linear = torch.nn.Linear(n_channels, n_channels)
        else:
            self.linear = None
        if gate_activation == 'identity':
            self.act = torch.nn.Identity()
        else:
            self.act = WENET_ACTIVATION_CLASSES[gate_activation]()
        self.dropout = torch.nn.Dropout(dropout_rate)

    def espnet_initialization_fn(self):
        torch.nn.init.normal_(self.conv.weight, std=1e-06)
        torch.nn.init.ones_(self.conv.bias)
        if self.linear is not None:
            torch.nn.init.normal_(self.linear.weight, std=1e-06)
            torch.nn.init.ones_(self.linear.bias)

    def forward(self, x: 'torch.Tensor', cache: 'torch.Tensor'=torch.zeros((0, 0, 0))) ->Tuple[torch.Tensor, torch.Tensor]:
        """Forward method

        Args:
            x (torch.Tensor): (batch, time, channels)
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (#batch, channels, cache_t),
                (0, 0, 0) meas fake cache.

        Returns:
            out (torch.Tensor): (batch, time, channels/2)
        """
        x_r, x_g = x.chunk(2, dim=-1)
        x_g = x_g.transpose(1, 2)
        if self.lorder > 0:
            if cache.size(2) == 0:
                x_g = nn.functional.pad(x_g, (self.lorder, 0), 'constant', 0.0)
            else:
                assert cache.size(0) == x_g.size(0)
                assert cache.size(1) == x_g.size(1)
                x_g = torch.cat((cache, x_g), dim=2)
            assert x_g.size(2) > self.lorder
            new_cache = x_g[:, :, -self.lorder:]
        else:
            new_cache = torch.zeros((0, 0, 0), dtype=x_g.dtype, device=x_g.device)
        x_g = x_g.transpose(1, 2)
        x_g = self.norm(x_g)
        x_g = self.conv(x_g.transpose(1, 2)).transpose(1, 2)
        if self.linear is not None:
            x_g = self.linear(x_g)
        x_g = self.act(x_g)
        out = x_r * x_g
        out = self.dropout(out)
        return out, new_cache


class ConvolutionalGatingMLP(torch.nn.Module):
    """Convolutional Gating MLP (cgMLP)."""

    def __init__(self, size: 'int', linear_units: 'int', kernel_size: 'int', dropout_rate: 'float', use_linear_after_conv: 'bool', gate_activation: 'str', causal: 'bool'=True):
        super().__init__()
        self.channel_proj1 = torch.nn.Sequential(torch.nn.Linear(size, linear_units), torch.nn.GELU())
        self.csgu = ConvolutionalSpatialGatingUnit(size=linear_units, kernel_size=kernel_size, dropout_rate=dropout_rate, use_linear_after_conv=use_linear_after_conv, gate_activation=gate_activation, causal=causal)
        self.channel_proj2 = torch.nn.Linear(linear_units // 2, size)

    def forward(self, x: 'torch.Tensor', mask: 'torch.Tensor', cache: 'torch.Tensor'=torch.zeros((0, 0, 0))) ->Tuple[torch.Tensor, torch.Tensor]:
        """Forward method

        Args:
            x (torch.Tensor): (batch, time, channels)
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask. Not used yet
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (#batch, channels, cache_t),
                (0, 0, 0) meas fake cache.

        Returns:
            out (torch.Tensor): (batch, time, channels/2)
        """
        xs_pad = x
        xs_pad = self.channel_proj1(xs_pad)
        xs_pad, new_cnn_cache = self.csgu(xs_pad, cache)
        xs_pad = self.channel_proj2(xs_pad)
        out = xs_pad
        return out, new_cnn_cache


class LayerDropModuleList(torch.nn.ModuleList):
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

    Limitations:
        1 can work with ddp when layer's gradient checkpoint disabled
        2 can't work with ddp when layer's gradient checkpoint enables
        3 can work with fsdp
        4 can work with deepspeed
    """

    def __init__(self, p: 'List[float]', modules=None):
        super().__init__(modules)
        assert len(p) == len(self)
        self.p = p

    def __iter__(self):
        dropout_probs = torch.empty(len(self)).uniform_()
        for i, m in enumerate(super().__iter__()):
            if not self.training or dropout_probs[i] > self.p[i]:
                yield m


class BranchformerEncoderLayer(torch.nn.Module):
    """Branchformer encoder layer module.

    Args:
        size (int): model dimension
        attn: standard self-attention or efficient attention, optional
        cgmlp: ConvolutionalGatingMLP, optional
        dropout_rate (float): dropout probability
        merge_method (str): concat, learned_ave, fixed_ave
        cgmlp_weight (float): weight of the cgmlp branch, between 0 and 1,
            used if merge_method is fixed_ave
        attn_branch_drop_rate (float): probability of dropping the attn branch,
            used if merge_method is learned_ave
        stochastic_depth_rate (float): stochastic depth probability
    """

    def __init__(self, size: 'int', attn: 'Optional[torch.nn.Module]', cgmlp: 'Optional[torch.nn.Module]', dropout_rate: 'float', merge_method: 'str', cgmlp_weight: 'float'=0.5, attn_branch_drop_rate: 'float'=0.0, stochastic_depth_rate: 'float'=0.0):
        super().__init__()
        assert attn is not None or cgmlp is not None, 'At least one branch should be valid'
        self.size = size
        self.attn = attn
        self.cgmlp = cgmlp
        self.merge_method = merge_method
        self.cgmlp_weight = cgmlp_weight
        self.attn_branch_drop_rate = attn_branch_drop_rate
        self.stochastic_depth_rate = stochastic_depth_rate
        self.use_two_branches = attn is not None and cgmlp is not None
        if attn is not None:
            self.norm_mha = nn.LayerNorm(size)
        if cgmlp is not None:
            self.norm_mlp = nn.LayerNorm(size)
        self.norm_final = nn.LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.pooling_proj1 = torch.nn.Linear(size, 1)
        self.pooling_proj2 = torch.nn.Linear(size, 1)
        self.weight_proj1 = torch.nn.Linear(size, 1)
        self.weight_proj2 = torch.nn.Linear(size, 1)
        if self.use_two_branches:
            if self.merge_method == 'concat':
                self.merge_proj = torch.nn.Linear(size + size, size)
            elif self.merge_method == 'learned_ave':
                self.merge_proj = torch.nn.Linear(size, size)
            elif self.merge_method == 'fixed_ave':
                assert 0.0 <= cgmlp_weight <= 1.0, 'cgmlp weight should be between 0.0 and 1.0'
                if cgmlp_weight == 0.0:
                    self.use_two_branches = False
                    self.cgmlp = None
                    self.norm_mlp = None
                elif cgmlp_weight == 1.0:
                    self.use_two_branches = False
                    self.attn = None
                    self.norm_mha = None
                self.merge_proj = torch.nn.Linear(size, size)
            else:
                raise ValueError(f'unknown merge method: {merge_method}')
        else:
            self.merge_proj = torch.nn.Identity()

    def _forward(self, x: 'torch.Tensor', mask: 'torch.Tensor', pos_emb: 'torch.Tensor', mask_pad: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool), att_cache: 'T_CACHE'=(torch.zeros((0, 0, 0, 0)), torch.zeros(0, 0, 0, 0)), cnn_cache: 'torch.Tensor'=torch.zeros((0, 0, 0, 0)), stoch_layer_coeff: 'float'=1.0) ->Tuple[torch.Tensor, torch.Tensor, T_CACHE, torch.Tensor]:
        x1 = x
        x2 = x
        if self.attn is not None:
            x1 = self.norm_mha(x1)
            x_att, new_att_cache = self.attn(x1, x1, x1, mask, pos_emb, att_cache)
            x1 = self.dropout(x_att)
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.cgmlp is not None:
            x2 = self.norm_mlp(x2)
            x2, new_cnn_cache = self.cgmlp(x2, mask_pad, cnn_cache)
            x2 = self.dropout(x2)
        if self.use_two_branches:
            if self.merge_method == 'concat':
                x = x + stoch_layer_coeff * self.dropout(self.merge_proj(torch.cat([x1, x2], dim=-1)))
            elif self.merge_method == 'learned_ave':
                if self.training and self.attn_branch_drop_rate > 0 and torch.rand(1).item() < self.attn_branch_drop_rate:
                    w1, w2 = torch.tensor(0.0), torch.tensor(1.0)
                else:
                    score1 = self.pooling_proj1(x1).transpose(1, 2) / self.size ** 0.5
                    score1 = score1.masked_fill(mask_pad.eq(0), -float('inf'))
                    score1 = torch.softmax(score1, dim=-1).masked_fill(mask_pad.eq(0), 0.0)
                    pooled1 = torch.matmul(score1, x1).squeeze(1)
                    weight1 = self.weight_proj1(pooled1)
                    score2 = self.pooling_proj2(x2).transpose(1, 2) / self.size ** 0.5
                    score2 = score2.masked_fill(mask_pad.eq(0), -float('inf'))
                    score2 = torch.softmax(score2, dim=-1).masked_fill(mask_pad.eq(0), 0.0)
                    pooled2 = torch.matmul(score2, x2).squeeze(1)
                    weight2 = self.weight_proj2(pooled2)
                    merge_weights = torch.softmax(torch.cat([weight1, weight2], dim=-1), dim=-1)
                    merge_weights = merge_weights.unsqueeze(-1).unsqueeze(-1)
                    w1, w2 = merge_weights[:, 0], merge_weights[:, 1]
                x = x + stoch_layer_coeff * self.dropout(self.merge_proj(w1 * x1 + w2 * x2))
            elif self.merge_method == 'fixed_ave':
                x = x + stoch_layer_coeff * self.dropout(self.merge_proj((1.0 - self.cgmlp_weight) * x1 + self.cgmlp_weight * x2))
            else:
                raise RuntimeError(f'unknown merge method: {self.merge_method}')
        elif self.attn is None:
            x = x + stoch_layer_coeff * self.dropout(self.merge_proj(x2))
        elif self.cgmlp is None:
            x = x + stoch_layer_coeff * self.dropout(self.merge_proj(x1))
        else:
            raise RuntimeError('Both branches are not None, which is unexpected.')
        x = self.norm_final(x)
        return x, mask, new_att_cache, new_cnn_cache

    def forward(self, x: 'torch.Tensor', mask: 'torch.Tensor', pos_emb: 'torch.Tensor', mask_pad: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool), att_cache: 'T_CACHE'=(torch.zeros((0, 0, 0, 0)), torch.zeros(0, 0, 0, 0)), cnn_cache: 'torch.Tensor'=torch.zeros((0, 0, 0, 0))) ->Tuple[torch.Tensor, torch.Tensor, T_CACHE, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (Union[Tuple, torch.Tensor]): Input tensor  (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time, time).
            pos_emb (torch.Tensor): positional encoding, must not be None
                for BranchformerEncoderLayer.
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1，time), (0, 0, 0) means fake mask.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in cgmlp layer
                (#batch=1, size, cache_t2)

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time.
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cahce tensor (#batch, size, cache_t2).
        """
        stoch_layer_coeff = 1.0
        if self.training:
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)
        return self._forward(x, mask, pos_emb, mask_pad, att_cache, cnn_cache, stoch_layer_coeff)


class EBranchformerEncoderLayer(torch.nn.Module):
    """E-Branchformer encoder layer module.

    Args:
        size (int): model dimension
        attn: standard self-attention or efficient attention
        cgmlp: ConvolutionalGatingMLP
        feed_forward: feed-forward module, optional
        feed_forward: macaron-style feed-forward module, optional
        dropout_rate (float): dropout probability
        merge_conv_kernel (int): kernel size of the depth-wise conv in merge module
    """

    def __init__(self, size: 'int', attn: 'torch.nn.Module', cgmlp: 'torch.nn.Module', feed_forward: 'Optional[torch.nn.Module]', feed_forward_macaron: 'Optional[torch.nn.Module]', dropout_rate: 'float', merge_conv_kernel: 'int'=3, causal: 'bool'=True, stochastic_depth_rate=0.0):
        super().__init__()
        self.size = size
        self.attn = attn
        self.cgmlp = cgmlp
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.ff_scale = 1.0
        if self.feed_forward is not None:
            self.norm_ff = nn.LayerNorm(size)
        if self.feed_forward_macaron is not None:
            self.ff_scale = 0.5
            self.norm_ff_macaron = nn.LayerNorm(size)
        self.norm_mha = nn.LayerNorm(size)
        self.norm_mlp = nn.LayerNorm(size)
        self.norm_final = nn.LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout_rate)
        if causal:
            padding = 0
            self.lorder = merge_conv_kernel - 1
        else:
            assert (merge_conv_kernel - 1) % 2 == 0
            padding = (merge_conv_kernel - 1) // 2
            self.lorder = 0
        self.depthwise_conv_fusion = torch.nn.Conv1d(size + size, size + size, kernel_size=merge_conv_kernel, stride=1, padding=padding, groups=size + size, bias=True)
        self.merge_proj = torch.nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate

    def _forward(self, x: 'torch.Tensor', mask: 'torch.Tensor', pos_emb: 'torch.Tensor', mask_pad: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool), att_cache: 'T_CACHE'=(torch.zeros((0, 0, 0, 0)), torch.zeros(0, 0, 0, 0)), cnn_cache: 'torch.Tensor'=torch.zeros((0, 0, 0, 0)), stoch_layer_coeff: 'float'=1.0) ->Tuple[torch.Tensor, torch.Tensor, T_CACHE, torch.Tensor]:
        if self.feed_forward_macaron is not None:
            residual = x
            x = self.norm_ff_macaron(x)
            x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(self.feed_forward_macaron(x))
        x1 = x
        x2 = x
        x1 = self.norm_mha(x1)
        x_att, new_att_cache = self.attn(x1, x1, x1, mask, pos_emb, att_cache)
        x1 = self.dropout(x_att)
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        x2 = self.norm_mlp(x2)
        x2, new_cnn_cache = self.cgmlp(x2, mask_pad, cnn_cache)
        x2 = self.dropout(x2)
        x_concat = torch.cat([x1, x2], dim=-1)
        x_tmp = x_concat.transpose(1, 2)
        if self.lorder > 0:
            x_tmp = nn.functional.pad(x_tmp, (self.lorder, 0), 'constant', 0.0)
            assert x_tmp.size(2) > self.lorder
        x_tmp = self.depthwise_conv_fusion(x_tmp)
        x_tmp = x_tmp.transpose(1, 2)
        x = x + stoch_layer_coeff * self.dropout(self.merge_proj(x_concat + x_tmp))
        if self.feed_forward is not None:
            residual = x
            x = self.norm_ff(x)
            x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(self.feed_forward(x))
        x = self.norm_final(x)
        return x, mask, new_att_cache, new_cnn_cache

    def forward(self, x: 'torch.Tensor', mask: 'torch.Tensor', pos_emb: 'torch.Tensor', mask_pad: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool), att_cache: 'T_CACHE'=(torch.zeros((0, 0, 0, 0)), torch.zeros(0, 0, 0, 0)), cnn_cache: 'torch.Tensor'=torch.zeros((0, 0, 0, 0))) ->Tuple[torch.Tensor, torch.Tensor, T_CACHE, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (Union[Tuple, torch.Tensor]): Input tensor  (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time, time).
            pos_emb (torch.Tensor): positional encoding, must not be None
                for BranchformerEncoderLayer.
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1，time), (0, 0, 0) means fake mask.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in cgmlp layer
                (#batch=1, size, cache_t2)

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time.
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cahce tensor (#batch, size, cache_t2).
        """
        stoch_layer_coeff = 1.0
        if self.training:
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)
        return self._forward(x, mask, pos_emb, mask_pad, att_cache, cnn_cache, stoch_layer_coeff)


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model."""

    def __init__(self, channels: 'int', kernel_size: 'int'=15, activation: 'nn.Module'=nn.ReLU(), norm: 'str'='batch_norm', causal: 'bool'=False, bias: 'bool'=True, norm_eps: 'float'=1e-05):
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
        assert norm in ['batch_norm', 'layer_norm', 'rms_norm']
        if norm == 'batch_norm':
            self.use_layer_norm = False
            self.norm = WENET_NORM_CLASSES['batch_norm'](channels, eps=norm_eps)
        else:
            self.use_layer_norm = True
            self.norm = WENET_NORM_CLASSES[norm](channels, eps=norm_eps)
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

    def __init__(self, size: 'int', self_attn: 'torch.nn.Module', feed_forward: 'Optional[nn.Module]'=None, feed_forward_macaron: 'Optional[nn.Module]'=None, conv_module: 'Optional[nn.Module]'=None, dropout_rate: 'float'=0.1, normalize_before: 'bool'=True, layer_norm_type: 'str'='layer_norm', norm_eps: 'float'=1e-05):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        assert layer_norm_type in ['layer_norm', 'rms_norm']
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = WENET_NORM_CLASSES[layer_norm_type](size, eps=norm_eps)
        self.norm_mha = WENET_NORM_CLASSES[layer_norm_type](size, eps=norm_eps)
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = WENET_NORM_CLASSES[layer_norm_type](size, eps=norm_eps)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = WENET_NORM_CLASSES[layer_norm_type](size, eps=norm_eps)
            self.norm_final = WENET_NORM_CLASSES[layer_norm_type](size, eps=norm_eps)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

    def forward(self, x: 'torch.Tensor', mask: 'torch.Tensor', pos_emb: 'torch.Tensor', mask_pad: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool), att_cache: 'T_CACHE'=(torch.zeros((0, 0, 0, 0)), torch.zeros((0, 0, 0, 0))), cnn_cache: 'torch.Tensor'=torch.zeros((0, 0, 0, 0))) ->Tuple[torch.Tensor, torch.Tensor, T_CACHE, torch.Tensor]:
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


class StrideConformerEncoderLayer(nn.Module):
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

    def __init__(self, size: 'int', self_attn: 'torch.nn.Module', feed_forward: 'Optional[nn.Module]'=None, feed_forward_macaron: 'Optional[nn.Module]'=None, conv_module: 'Optional[nn.Module]'=None, pointwise_conv_layer: 'Optional[nn.Module]'=None, dropout_rate: 'float'=0.1, normalize_before: 'bool'=True):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.pointwise_conv_layer = pointwise_conv_layer
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
        self.concat_linear = nn.Linear(size + size, size)

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
        new_cnn_cache = torch.tensor([0.0], dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            if self.pointwise_conv_layer is not None:
                residual = residual.transpose(1, 2)
                residual = self.pointwise_conv_layer(residual)
                residual = residual.transpose(1, 2)
                assert residual.size(0) == x.size(0)
                assert residual.size(1) == x.size(1)
                assert residual.size(2) == x.size(2)
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


class Conv2dSubsampling2(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: 'int', odim: 'int', dropout_rate: 'float', pos_enc_class: 'torch.nn.Module'):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(1, odim, 3, 2), torch.nn.ReLU())
        self.out = torch.nn.Sequential(torch.nn.Linear(odim * ((idim - 1) // 2), odim))
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 2
        self.right_context = 2

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
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2]


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


class DepthwiseConv2dSubsampling4(BaseSubsampling):
    """Depthwise Convolutional 2D subsampling (to 1/4 length).

        Args:
            idim (int): Input dimension.
            odim (int): Output dimension.
            pos_enc_class (nn.Module): position encoding class.
            dw_stride (int): Whether do depthwise convolution.
            input_size (int): filter bank dimension.

        """

    def __init__(self, idim: 'int', odim: 'int', pos_enc_class: 'torch.nn.Module', dw_stride: 'bool'=False, input_size: 'int'=80, input_dropout_rate: 'float'=0.1, init_weights: 'bool'=True):
        super(DepthwiseConv2dSubsampling4, self).__init__()
        self.idim = idim
        self.odim = odim
        self.pw_conv = nn.Conv2d(in_channels=idim, out_channels=odim, kernel_size=3, stride=2)
        self.act1 = nn.ReLU()
        self.dw_conv = nn.Conv2d(in_channels=odim, out_channels=odim, kernel_size=3, stride=2, groups=odim if dw_stride else 1)
        self.act2 = nn.ReLU()
        self.pos_enc = pos_enc_class
        self.input_proj = nn.Sequential(nn.Linear(odim * (((input_size - 1) // 2 - 1) // 2), odim), nn.Dropout(p=input_dropout_rate))
        if init_weights:
            linear_max = (odim * input_size / 4) ** -0.5
            torch.nn.init.uniform_(self.input_proj.state_dict()['0.weight'], -linear_max, linear_max)
            torch.nn.init.uniform_(self.input_proj.state_dict()['0.bias'], -linear_max, linear_max)
        self.subsampling_rate = 4
        self.right_context = 6

    def forward(self, x: 'torch.Tensor', x_mask: 'torch.Tensor', offset: 'int'=0) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)
        x = self.pw_conv(x)
        x = self.act1(x)
        x = self.dw_conv(x)
        x = self.act2(x)
        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(b, t, c * f)
        x, pos_emb = self.pos_enc(x, offset)
        x = self.input_proj(x)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2]


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


class StackNFramesSubsampling(BaseSubsampling):

    def __init__(self, idim: 'int', odim: 'int', dropout_rate: 'float', pos_enc_class: 'torch.nn.Module', stride: 'int'=2):
        super().__init__()
        del dropout_rate
        self.pos_enc_class = pos_enc_class
        self.stride = stride
        self.idim = idim
        self.norm = torch.nn.LayerNorm(idim * stride, eps=1e-05)
        self.out = torch.nn.Linear(idim * stride, odim)

    def forward(self, x: 'torch.Tensor', x_mask: 'torch.Tensor', offset: 'Union[int, torch.Tensor]'=0) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // stride.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // stride.
            torch.Tensor: positional encoding
        """
        with torch.no_grad():
            b, s, _ = x.size()
            seq_len = x_mask.sum(-1).view(b)
            r = s % self.stride
            s -= r
            x = x[:, :s, :]
            seq_len = torch.where(seq_len > s, s, seq_len)
            seq_len = seq_len // self.stride
            new_mask = ~make_pad_mask(seq_len, max_len=s // self.stride)
            x = x.view(b, s // self.stride, self.idim * self.stride)
            _, pos_emb = self.pos_enc_class(x, offset)
        x = self.norm(x)
        x = self.out(x)
        return x, pos_emb, new_mask.unsqueeze(1)

    def position_encoding(self, offset: 'Union[int, torch.Tensor]', size: 'int') ->torch.Tensor:
        return self.pos_enc_class.position_encoding(offset, size)


WENET_SUBSAMPLE_CLASSES = {'linear': LinearNoSubsampling, 'embed': EmbedinigNoSubsampling, 'conv1d2': Conv1dSubsampling2, 'conv2d2': Conv2dSubsampling2, 'conv2d': Conv2dSubsampling4, 'dwconv2d4': DepthwiseConv2dSubsampling4, 'conv2d6': Conv2dSubsampling6, 'conv2d8': Conv2dSubsampling8, 'paraformer_dummy': torch.nn.Identity, 'stack_n_frames': StackNFramesSubsampling}


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


def add_optional_chunk_mask(xs: 'torch.Tensor', masks: 'torch.Tensor', use_dynamic_chunk: 'bool', use_dynamic_left_chunk: 'bool', decoding_chunk_size: 'int', static_chunk_size: 'int', num_decoding_left_chunks: 'int', enable_full_context: 'bool'=True, max_chunk_size: 'int'=25):
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
            True: chunk size is either [1, max_chunk_size] or full context(max_len)
            False: chunk size ~ U[1, max_chunk_size]

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
                chunk_size = chunk_size % max_chunk_size + 1
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


class EfficientConformerEncoder(torch.nn.Module):
    """Conformer encoder module."""

    def __init__(self, input_size: 'int', output_size: 'int'=256, attention_heads: 'int'=4, linear_units: 'int'=2048, num_blocks: 'int'=6, dropout_rate: 'float'=0.1, positional_dropout_rate: 'float'=0.1, attention_dropout_rate: 'float'=0.0, input_layer: 'str'='conv2d', pos_enc_layer_type: 'str'='rel_pos', normalize_before: 'bool'=True, static_chunk_size: 'int'=0, use_dynamic_chunk: 'bool'=False, global_cmvn: 'torch.nn.Module'=None, use_dynamic_left_chunk: 'bool'=False, macaron_style: 'bool'=True, activation_type: 'str'='swish', use_cnn_module: 'bool'=True, cnn_module_kernel: 'int'=15, causal: 'bool'=False, cnn_module_norm: 'str'='batch_norm', stride_layer_idx: 'Optional[Union[int, List[int]]]'=3, stride: 'Optional[Union[int, List[int]]]'=2, group_layer_idx: 'Optional[Union[int, List[int], tuple]]'=(0, 1, 2, 3), group_size: 'int'=3, stride_kernel: 'bool'=True, **kwargs):
        """Construct Efficient Conformer Encoder

        Args:
            input_size to use_dynamic_chunk, see in BaseEncoder
            macaron_style (bool): Whether to use macaron style for
                positionwise layer.
            activation_type (str): Encoder activation function type.
            use_cnn_module (bool): Whether to use convolution module.
            cnn_module_kernel (int): Kernel size of convolution module.
            causal (bool): whether to use causal convolution or not.
            stride_layer_idx (list): layer id with StrideConv, start from 0
            stride (list): stride size of each StrideConv in efficient conformer
            group_layer_idx (list): layer id with GroupedAttention, start from 0
            group_size (int): group size of every GroupedAttention layer
            stride_kernel (bool): default True. True: recompute cnn kernels with stride.
        """
        super().__init__()
        self._output_size = output_size
        logging.info(f'input_layer = {input_layer}, subsampling_class = {WENET_SUBSAMPLE_CLASSES[input_layer]}')
        self.global_cmvn = global_cmvn
        self.embed = WENET_SUBSAMPLE_CLASSES[input_layer](input_size, output_size, dropout_rate, WENET_EMB_CLASSES[pos_enc_layer_type](output_size, positional_dropout_rate))
        self.input_layer = input_layer
        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-05)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        activation = WENET_ACTIVATION_CLASSES[activation_type]()
        self.num_blocks = num_blocks
        self.attention_heads = attention_heads
        self.cnn_module_kernel = cnn_module_kernel
        self.global_chunk_size = 0
        self.chunk_feature_map = 0
        self.stride_layer_idx = [stride_layer_idx] if type(stride_layer_idx) == int else stride_layer_idx
        self.stride = [stride] if type(stride) == int else stride
        self.group_layer_idx = [group_layer_idx] if type(group_layer_idx) == int else group_layer_idx
        self.grouped_size = group_size
        assert len(self.stride) == len(self.stride_layer_idx)
        self.cnn_module_kernels = [cnn_module_kernel]
        for i in self.stride:
            if stride_kernel:
                self.cnn_module_kernels.append(self.cnn_module_kernels[-1] // i)
            else:
                self.cnn_module_kernels.append(self.cnn_module_kernels[-1])
        logging.info(f'stride_layer_idx= {self.stride_layer_idx}, stride = {self.stride}, cnn_module_kernel = {self.cnn_module_kernels}, group_layer_idx = {self.group_layer_idx}, grouped_size = {self.grouped_size}')
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = output_size, linear_units, dropout_rate, activation
        convolution_layer = ConvolutionModule
        index = 0
        layers = []
        for i in range(num_blocks):
            if i in self.group_layer_idx:
                encoder_selfattn_layer = WENET_ATTENTION_CLASSES['grouped_rel_selfattn']
                encoder_selfattn_layer_args = attention_heads, output_size, attention_dropout_rate, self.grouped_size
            else:
                if pos_enc_layer_type == 'no_pos':
                    encoder_selfattn_layer = WENET_ATTENTION_CLASSES['selfattn']
                else:
                    encoder_selfattn_layer = WENET_ATTENTION_CLASSES['rel_selfattn']
                encoder_selfattn_layer_args = attention_heads, output_size, attention_dropout_rate
            if i in self.stride_layer_idx:
                convolution_layer_args_stride = output_size, self.cnn_module_kernels[index], activation, cnn_module_norm, causal, True, self.stride[index]
                layers.append(StrideConformerEncoderLayer(output_size, encoder_selfattn_layer(*encoder_selfattn_layer_args), positionwise_layer(*positionwise_layer_args), positionwise_layer(*positionwise_layer_args) if macaron_style else None, convolution_layer(*convolution_layer_args_stride) if use_cnn_module else None, torch.nn.AvgPool1d(kernel_size=self.stride[index], stride=self.stride[index], padding=0, ceil_mode=True, count_include_pad=False), dropout_rate, normalize_before))
                index = index + 1
            else:
                convolution_layer_args_normal = output_size, self.cnn_module_kernels[index], activation, cnn_module_norm, causal
                layers.append(ConformerEncoderLayer(output_size, encoder_selfattn_layer(*encoder_selfattn_layer_args), positionwise_layer(*positionwise_layer_args), positionwise_layer(*positionwise_layer_args) if macaron_style else None, convolution_layer(*convolution_layer_args_normal) if use_cnn_module else None, dropout_rate, normalize_before))
        self.encoders = torch.nn.ModuleList(layers)

    def set_global_chunk_size(self, chunk_size):
        """Used in ONNX export.
        """
        logging.info(f'set global chunk size: {chunk_size}, default is 0.')
        self.global_chunk_size = chunk_size
        if self.embed.subsampling_rate == 2:
            self.chunk_feature_map = 2 * self.global_chunk_size + 1
        elif self.embed.subsampling_rate == 6:
            self.chunk_feature_map = 6 * self.global_chunk_size + 5
        elif self.embed.subsampling_rate == 8:
            self.chunk_feature_map = 8 * self.global_chunk_size + 7
        else:
            self.chunk_feature_map = 4 * self.global_chunk_size + 3

    def output_size(self) ->int:
        return self._output_size

    def calculate_downsampling_factor(self, i: 'int') ->int:
        factor = 1
        for idx, stride_idx in enumerate(self.stride_layer_idx):
            if i > stride_idx:
                factor *= self.stride[idx]
        return factor

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
        """
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks
        chunk_masks = add_optional_chunk_mask(xs, masks, self.use_dynamic_chunk, self.use_dynamic_left_chunk, decoding_chunk_size, self.static_chunk_size, num_decoding_left_chunks)
        index = 0
        for i, layer in enumerate(self.encoders):
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
            if i in self.stride_layer_idx:
                masks = masks[:, :, ::self.stride[index]]
                chunk_masks = chunk_masks[:, ::self.stride[index], ::self.stride[index]]
                mask_pad = masks
                pos_emb = pos_emb[:, ::self.stride[index], :]
                index = index + 1
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks

    def forward_chunk(self, xs: 'torch.Tensor', offset: 'int', required_cache_size: 'int', att_cache: 'torch.Tensor'=torch.zeros(0, 0, 0, 0), cnn_cache: 'torch.Tensor'=torch.zeros(0, 0, 0, 0), att_mask: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool)) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Forward just one chunk

        Args:
            xs (torch.Tensor): chunk input
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
            att_mask : mask matrix of self attention

        Returns:
            torch.Tensor: output of current input xs
            torch.Tensor: subsampling cache required for next chunk computation
            List[torch.Tensor]: encoder layers output cache required for next
                chunk computation
            List[torch.Tensor]: conformer cnn cache

        """
        assert xs.size(0) == 1
        offset *= self.calculate_downsampling_factor(self.num_blocks + 1)
        chunk_masks = torch.ones(1, xs.size(1), device=xs.device, dtype=torch.bool)
        chunk_masks = chunk_masks.unsqueeze(1)
        real_len = 0
        if self.global_chunk_size > 0:
            real_len = xs.size(1)
            pad_len = self.chunk_feature_map - real_len
            xs = F.pad(xs, (0, 0, 0, pad_len), value=0.0)
            chunk_masks = F.pad(chunk_masks, (0, pad_len), value=0.0)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, chunk_masks = self.embed(xs, chunk_masks, offset)
        elayers, cache_t1 = att_cache.size(0), att_cache.size(2)
        chunk_size = xs.size(1)
        attention_key_size = cache_t1 + chunk_size
        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = attention_key_size
        else:
            next_cache_start = max(attention_key_size - required_cache_size, 0)
        r_att_cache = []
        r_cnn_cache = []
        mask_pad = torch.ones(1, xs.size(1), device=xs.device, dtype=torch.bool)
        mask_pad = mask_pad.unsqueeze(1)
        if self.global_chunk_size > 0:
            pos_emb = self.embed.position_encoding(offset=max(offset - cache_t1, 0), size=cache_t1 + self.global_chunk_size)
            att_mask[:, :, -self.global_chunk_size:] = chunk_masks
            mask_pad = chunk_masks
        else:
            pos_emb = self.embed.position_encoding(offset=offset - cache_t1, size=attention_key_size)
        max_att_len, max_cnn_len = 0, 0
        for i, layer in enumerate(self.encoders):
            factor = self.calculate_downsampling_factor(i)
            att_cache_trunc = 0
            if xs.size(1) + att_cache.size(2) / factor > pos_emb.size(1):
                att_cache_trunc = xs.size(1) + att_cache.size(2) // factor - pos_emb.size(1) + 1
            xs, _, new_att_cache, new_cnn_cache = layer(xs, att_mask, pos_emb, mask_pad=mask_pad, att_cache=att_cache[i:i + 1, :, ::factor, :][:, :, att_cache_trunc:, :], cnn_cache=cnn_cache[i, :, :, :] if cnn_cache.size(0) > 0 else cnn_cache)
            if i in self.stride_layer_idx:
                efficient_index = self.stride_layer_idx.index(i)
                att_mask = att_mask[:, ::self.stride[efficient_index], ::self.stride[efficient_index]]
                mask_pad = mask_pad[:, ::self.stride[efficient_index], ::self.stride[efficient_index]]
                pos_emb = pos_emb[:, ::self.stride[efficient_index], :]
            new_att_cache = new_att_cache[:, :, next_cache_start // factor:, :]
            new_cnn_cache = new_cnn_cache.unsqueeze(0)
            new_att_cache = new_att_cache.repeat_interleave(repeats=factor, dim=2)
            new_cnn_cache = F.pad(new_cnn_cache, (self.cnn_module_kernel - 1 - new_cnn_cache.size(3), 0))
            if i == 0:
                max_att_len = new_att_cache.size(2)
                max_cnn_len = new_cnn_cache.size(3)
            r_att_cache.append(new_att_cache[:, :, -max_att_len:, :])
            r_cnn_cache.append(new_cnn_cache[:, :, :, -max_cnn_len:])
        if self.normalize_before:
            xs = self.after_norm(xs)
        r_att_cache = torch.cat(r_att_cache, dim=0)
        r_cnn_cache = torch.cat(r_cnn_cache, dim=0)
        if self.global_chunk_size > 0 and real_len:
            chunk_real_len = real_len // self.embed.subsampling_rate // self.calculate_downsampling_factor(self.num_blocks + 1)
            xs = xs[:, :chunk_real_len + 1, :]
        return xs, r_att_cache, r_cnn_cache

    def forward_chunk_by_chunk(self, xs: 'torch.Tensor', decoding_chunk_size: 'int', num_decoding_left_chunks: 'int'=-1, use_onnx=False) ->Tuple[torch.Tensor, torch.Tensor]:
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
               with cache, which is not prefered.
        Args:
            xs (torch.Tensor): (1, max_len, dim)
            decoding_chunk_size (int): decoding chunk size
            num_decoding_left_chunks (int):
            use_onnx (bool): True for simulating ONNX model inference.
        """
        assert decoding_chunk_size > 0
        assert self.static_chunk_size > 0 or self.use_dynamic_chunk
        subsampling = self.embed.subsampling_rate
        context = self.embed.right_context + 1
        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        num_frames = xs.size(1)
        outputs = []
        offset = 0
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks
        if use_onnx:
            logging.info('Simulating for ONNX runtime ...')
            att_cache: 'torch.Tensor' = torch.zeros((self.num_blocks, self.attention_heads, required_cache_size, self.output_size() // self.attention_heads * 2), device=xs.device)
            cnn_cache: 'torch.Tensor' = torch.zeros((self.num_blocks, 1, self.output_size(), self.cnn_module_kernel - 1), device=xs.device)
            self.set_global_chunk_size(chunk_size=decoding_chunk_size)
        else:
            logging.info('Simulating for JIT runtime ...')
            att_cache: 'torch.Tensor' = torch.zeros((0, 0, 0, 0), device=xs.device)
            cnn_cache: 'torch.Tensor' = torch.zeros((0, 0, 0, 0), device=xs.device)
        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + decoding_window, num_frames)
            logging.info(f'-->> frame chunk msg: cur={cur}, end={end}, num_frames={end - cur}, decoding_window={decoding_window}')
            if use_onnx:
                att_mask: 'torch.Tensor' = torch.ones((1, 1, required_cache_size + decoding_chunk_size), dtype=torch.bool, device=xs.device)
                if cur == 0:
                    att_mask[:, :, :required_cache_size] = 0
            else:
                att_mask: 'torch.Tensor' = torch.ones((0, 0, 0), dtype=torch.bool, device=xs.device)
            chunk_xs = xs[:, cur:end, :]
            y, att_cache, cnn_cache = self.forward_chunk(chunk_xs, offset, required_cache_size, att_cache, cnn_cache, att_mask)
            outputs.append(y)
            offset += y.size(1)
        ys = torch.cat(outputs, 1)
        masks = torch.ones(1, 1, ys.size(1), device=ys.device, dtype=torch.bool)
        return ys, masks


class LoRALayer:

    def __init__(self, r: 'int', lora_alpha: 'int', lora_dropout: 'float', merge_weights: 'bool'):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = self.identity
        self.merged = False
        self.merge_weights = merge_weights

    def identity(self, x):
        return x


class Embedding(nn.Embedding, LoRALayer):

    def __init__(self, num_embeddings: 'int', embedding_dim: 'int', r: 'int'=0, lora_alpha: 'int'=1, merge_weights: 'bool'=True, **kwargs):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0, merge_weights=merge_weights)
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: 'bool'=True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    temp = (self.lora_B @ self.lora_A).transpose(0, 1)
                    self.weight.data -= temp * self.scaling
                self.merged = False
        elif self.merge_weights and not self.merged:
            if self.r > 0:
                temp = (self.lora_B @ self.lora_A).transpose(0, 1)
                self.weight.data += temp * self.scaling
            self.merged = True

    def forward(self, x: 'torch.Tensor'):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            after_A = F.embedding(x, self.lora_A.transpose(0, 1), self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
            result += after_A @ self.lora_B.transpose(0, 1) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)


class Linear(nn.Linear, LoRALayer):

    def __init__(self, in_features: 'int', out_features: 'int', r: 'int'=0, lora_alpha: 'int'=1, lora_dropout: 'float'=0.0, fan_in_fan_out: 'bool'=False, merge_weights: 'bool'=True, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    def train(self, mode: 'bool'=True):
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    temp = self.T(self.lora_B @ self.lora_A)
                    self.weight.data -= temp * self.scaling
                self.merged = False
        elif self.merge_weights and not self.merged:
            if self.r > 0:
                temp = self.T(self.lora_B @ self.lora_A)
                self.weight.data += temp * self.scaling
            self.merged = True

    def forward(self, x: 'torch.Tensor'):
        if self.r > 0 and not self.merged:
            result = F.linear(x, self.T(self.weight), bias=self.bias)
            result += self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1) * self.scaling
            return result
        else:
            return F.linear(x, self.T(self.weight), bias=self.bias)


class MergedLinear(nn.Linear, LoRALayer):

    def __init__(self, in_features: 'int', out_features: 'int', r: 'int'=0, lora_alpha: 'int'=1, lora_dropout: 'float'=0.0, enable_lora: 'List[bool]'=None, fan_in_fan_out: 'bool'=False, merge_weights: 'bool'=True, **kwargs):
        if enable_lora is None:
            enable_lora = [False]
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, 'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
            self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.size()[1:]))
        result[self.lora_ind] = x
        return result

    def T(self, w):
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    def merge_AB(self):
        delta_w = F.conv1d(self.lora_A.unsqueeze(0), self.lora_B.unsqueeze(-1), groups=sum(self.enable_lora)).squeeze(0)
        return self.T(delta_w)

    def train(self, mode: 'bool'=True):
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        elif self.merge_weights and not self.merged:
            if self.r > 0 and any(self.enable_lora):
                self.weight.data += self.merge_AB() * self.scaling
            self.merged = True

    def forward(self, x: 'torch.Tensor'):
        if self.merged:
            return F.linear(x, self.T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, self.T(self.weight), bias=self.bias)
            if self.r > 0:
                temp = self.T(self.merge_AB().T)
                result += self.lora_dropout(x) @ temp * self.scaling
            return result


class ConvLoRA(nn.Module, LoRALayer):

    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0.0, merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        if r > 0:
            self.lora_A = nn.Parameter(self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size)))
            self.lora_B = nn.Parameter(self.conv.weight.new_zeros((out_channels // self.conv.groups * kernel_size, r * kernel_size)))
            self.scaling = self.lora_alpha / self.r
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        elif self.merge_weights and not self.merged:
            if self.r > 0:
                self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(x, self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling, self.conv.bias)
        return self.conv(x)


class Conv2d(ConvLoRA):

    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)


class Conv1d(ConvLoRA):

    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)


class Conv3d(ConvLoRA):

    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(nn.Conv3d, *args, **kwargs)


def cif(hidden: 'torch.Tensor', alphas: 'torch.Tensor', threshold: 'float'):
    batch_size, len_time, hidden_size = hidden.size()
    integrate = torch.zeros([batch_size], device=hidden.device)
    frame = torch.zeros([batch_size, hidden_size], device=hidden.device)
    list_fires = []
    list_frames = []
    for t in range(len_time):
        alpha = alphas[:, t]
        distribution_completion = torch.ones([batch_size], device=hidden.device) - integrate
        integrate += alpha
        list_fires.append(integrate)
        fire_place = integrate >= threshold
        integrate = torch.where(fire_place, integrate - torch.ones([batch_size], device=hidden.device), integrate)
        cur = torch.where(fire_place, distribution_completion, alpha)
        remainds = alpha - cur
        frame += cur[:, None] * hidden[:, t, :]
        list_frames.append(frame)
        frame = torch.where(fire_place[:, None].repeat(1, hidden_size), remainds[:, None] * hidden[:, t, :], frame)
    fires = torch.stack(list_fires, 1)
    frames = torch.stack(list_frames, 1)
    list_ls = []
    len_labels = torch.round(alphas.sum(-1)).int()
    max_label_len = len_labels.max()
    for b in range(batch_size):
        fire = fires[b, :]
        l = torch.index_select(frames[b, :, :], 0, torch.nonzero(fire >= threshold).squeeze())
        pad_l = torch.zeros([int(max_label_len - l.size(0)), hidden_size], device=hidden.device)
        list_ls.append(torch.cat([l, pad_l], 0))
    return torch.stack(list_ls, 0), fires


class Cif(nn.Module):

    def __init__(self, idim, l_order, r_order, threshold=1.0, dropout=0.1, smooth_factor=1.0, noise_threshold=0.0, tail_threshold=0.45, residual=True, cnn_groups=0):
        super().__init__()
        self.pad = nn.ConstantPad1d((l_order, r_order), 0.0)
        self.cif_conv1d = nn.Conv1d(idim, idim, l_order + r_order + 1, groups=idim if cnn_groups == 0 else cnn_groups)
        self.cif_output = nn.Linear(idim, 1)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.threshold = threshold
        self.smooth_factor = smooth_factor
        self.noise_threshold = noise_threshold
        self.tail_threshold = tail_threshold
        self.residual = residual

    def forward(self, hidden, target_label: 'Optional[torch.Tensor]'=None, mask: 'torch.Tensor'=torch.tensor(0), ignore_id: 'int'=-1, mask_chunk_predictor: 'Optional[torch.Tensor]'=None, target_label_length: 'Optional[torch.Tensor]'=None) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = hidden
        context = h.transpose(1, 2)
        queries = self.pad(context)
        memory = self.cif_conv1d(queries)
        if self.residual:
            output = memory + context
        else:
            output = memory
        output = self.dropout(output)
        output = output.transpose(1, 2)
        output = torch.relu(output)
        output = self.cif_output(output)
        alphas = torch.sigmoid(output)
        alphas = torch.nn.functional.relu(alphas * self.smooth_factor - self.noise_threshold)
        if mask is not None:
            mask = mask.transpose(-1, -2)
            alphas = alphas * mask
        if mask_chunk_predictor is not None:
            alphas = alphas * mask_chunk_predictor
        alphas = alphas.squeeze(-1)
        mask = mask.squeeze(-1)
        if target_label_length is not None:
            target_length = target_label_length
        elif target_label is not None:
            target_length = (target_label != ignore_id).float().sum(-1)
        else:
            target_length = None
        token_num = alphas.sum(-1)
        if target_length is not None:
            alphas *= (target_length / token_num)[:, None].repeat(1, alphas.size(1))
        elif self.tail_threshold > 0.0:
            hidden, alphas, token_num = self.tail_process_fn(hidden, alphas, token_num, mask=mask)
        acoustic_embeds, cif_peak = cif(hidden, alphas, self.threshold)
        if target_length is None and self.tail_threshold > 0.0:
            token_num_int = torch.max(token_num).type(torch.int32).item()
            acoustic_embeds = acoustic_embeds[:, :token_num_int, :]
        return acoustic_embeds, token_num, alphas, cif_peak

    def tail_process_fn(self, hidden: 'torch.Tensor', alphas: 'torch.Tensor', token_num: 'Optional[torch.Tensor]'=None, mask: 'Optional[torch.Tensor]'=None) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, d = hidden.size()
        if mask is not None:
            zeros_t = torch.zeros((b, 1), dtype=torch.float32, device=alphas.device)
            mask = mask
            ones_t = torch.ones_like(zeros_t)
            mask_1 = torch.cat([mask, zeros_t], dim=1)
            mask_2 = torch.cat([ones_t, mask], dim=1)
            mask = mask_2 - mask_1
            tail_threshold = mask * self.tail_threshold
            alphas = torch.cat([alphas, zeros_t], dim=1)
            alphas = torch.add(alphas, tail_threshold)
        else:
            tail_threshold_tensor = torch.tensor([self.tail_threshold], dtype=alphas.dtype)
            tail_threshold_tensor = torch.reshape(tail_threshold_tensor, (1, 1))
            alphas = torch.cat([alphas, tail_threshold_tensor], dim=1)
        zeros = torch.zeros((b, 1, d), dtype=hidden.dtype)
        hidden = torch.cat([hidden, zeros], dim=1)
        token_num = alphas.sum(dim=-1)
        token_num_floor = torch.floor(token_num)
        return hidden, alphas, token_num_floor

    def gen_frame_alignments(self, alphas: 'torch.Tensor'=None, encoder_sequence_length: 'torch.Tensor'=None):
        batch_size, maximum_length = alphas.size()
        int_type = torch.int32
        is_training = self.training
        if is_training:
            token_num = torch.round(torch.sum(alphas, dim=1)).type(int_type)
        else:
            token_num = torch.floor(torch.sum(alphas, dim=1)).type(int_type)
        max_token_num = torch.max(token_num).item()
        alphas_cumsum = torch.cumsum(alphas, dim=1)
        alphas_cumsum = torch.floor(alphas_cumsum).type(int_type)
        alphas_cumsum = alphas_cumsum[:, None, :].repeat(1, max_token_num, 1)
        index = torch.ones([batch_size, max_token_num], dtype=int_type)
        index = torch.cumsum(index, dim=1)
        index = index[:, :, None].repeat(1, 1, maximum_length)
        index_div = torch.floor(torch.true_divide(alphas_cumsum, index)).type(int_type)
        index_div_bool_zeros = index_div.eq(0)
        index_div_bool_zeros_count = torch.sum(index_div_bool_zeros, dim=-1) + 1
        index_div_bool_zeros_count = torch.clamp(index_div_bool_zeros_count, 0, encoder_sequence_length.max())
        token_num_mask = ~make_pad_mask(token_num, max_len=max_token_num)
        index_div_bool_zeros_count *= token_num_mask
        index_div_bool_zeros_count_tile = index_div_bool_zeros_count[:, :, None].repeat(1, 1, maximum_length)
        ones = torch.ones_like(index_div_bool_zeros_count_tile)
        zeros = torch.zeros_like(index_div_bool_zeros_count_tile)
        ones = torch.cumsum(ones, dim=2)
        cond = index_div_bool_zeros_count_tile == ones
        index_div_bool_zeros_count_tile = torch.where(cond, zeros, ones)
        index_div_bool_zeros_count_tile_bool = index_div_bool_zeros_count_tile.type(torch.bool)
        index_div_bool_zeros_count_tile = 1 - index_div_bool_zeros_count_tile_bool.type(int_type)
        index_div_bool_zeros_count_tile_out = torch.sum(index_div_bool_zeros_count_tile, dim=1)
        index_div_bool_zeros_count_tile_out = index_div_bool_zeros_count_tile_out.type(int_type)
        predictor_mask = (~make_pad_mask(encoder_sequence_length, max_len=encoder_sequence_length.max())).type(int_type)
        index_div_bool_zeros_count_tile_out = index_div_bool_zeros_count_tile_out * predictor_mask
        predictor_alignments = index_div_bool_zeros_count_tile_out
        predictor_alignments_length = predictor_alignments.sum(-1).type(encoder_sequence_length.dtype)
        return predictor_alignments.detach(), predictor_alignments_length.detach()


class MAELoss(nn.Module):

    def __init__(self, normalize_length=False):
        super(MAELoss, self).__init__()
        self.normalize_length = normalize_length
        self.criterion = torch.nn.L1Loss(reduction='sum')

    def forward(self, token_length, pre_token_length):
        loss_token_normalizer = token_length.size(0)
        if self.normalize_length:
            loss_token_normalizer = token_length.sum().type(torch.float32)
        loss = self.criterion(token_length, pre_token_length)
        loss = loss / loss_token_normalizer
        return loss


class LFR(torch.nn.Module):

    def __init__(self, m: 'int'=7, n: 'int'=6) ->None:
        """
        Actually, this implements stacking frames and skipping frames.
        if m = 1 and n = 1, just return the origin features.
        if m = 1 and n > 1, it works like skipping.
        if m > 1 and n = 1, it works like stacking but only support right frames.
        if m > 1 and n > 1, it works like LFR.

        """
        super().__init__()
        self.m = m
        self.n = n
        self.left_padding_nums = math.ceil((self.m - 1) // 2)

    def forward(self, input: 'torch.Tensor', input_lens: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        orign_type = input_lens.dtype
        input_lens = input_lens
        B, _, D = input.size()
        n_lfr = torch.ceil(input_lens / self.n)
        prepad_nums = input_lens + self.left_padding_nums
        right_padding_nums = torch.where(self.m >= prepad_nums - self.n * (n_lfr - 1), self.m - (prepad_nums - self.n * (n_lfr - 1)), 0)
        T_all = self.left_padding_nums + input_lens + right_padding_nums
        new_len = T_all // self.n
        T_all_max = T_all.max().int()
        tail_frames_index = (input_lens - 1).view(B, 1, 1).repeat(1, 1, D)
        tail_frames = torch.gather(input, 1, tail_frames_index)
        tail_frames = tail_frames.repeat(1, right_padding_nums.max().int(), 1)
        head_frames = input[:, 0:1, :].repeat(1, self.left_padding_nums, 1)
        input = torch.cat([head_frames, input, tail_frames], dim=1)
        index = torch.arange(T_all_max, device=input.device, dtype=input_lens.dtype).unsqueeze(0).repeat(B, 1)
        index_mask = index < (self.left_padding_nums + input_lens).unsqueeze(1)
        tail_index_mask = torch.logical_not(index >= T_all.unsqueeze(1)) & index_mask
        tail = torch.ones(T_all_max, dtype=input_lens.dtype, device=input.device).unsqueeze(0).repeat(B, 1) * (T_all_max - 1)
        indices = torch.where(torch.logical_or(index_mask, tail_index_mask), index, tail)
        input = torch.gather(input, 1, indices.unsqueeze(2).repeat(1, 1, D))
        input = input.unfold(1, self.m, step=self.n).transpose(2, 3)
        new_len = new_len
        return input.reshape(B, -1, D * self.m), new_len


class PositionwiseFeedForwardDecoderSANM(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, adim=None, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForwardDecoderSANM, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim if adim is None else adim, bias=False)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation
        self.norm = torch.nn.LayerNorm(hidden_units)

    def forward(self, x):
        """Forward function."""
        return self.w_2(self.norm(self.dropout(self.activation(self.w_1(x)))))


class _Decoders3(torch.nn.Module):
    """Paraformer has a decoder3"""

    def __init__(self, hidden: 'int', pos_clss: 'torch.nn.Module') ->None:
        super().__init__()
        self.feed_forward = pos_clss
        self.norm1 = torch.nn.LayerNorm(hidden)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self.feed_forward(self.norm1(x))


class Predictor(torch.nn.Module):

    def __init__(self, idim, l_order, r_order, threshold=1.0, dropout=0.1, smooth_factor=1.0, noise_threshold=0.0, tail_threshold=0.45, residual=True, cnn_groups=0, smooth_factor2=0.25, noise_threshold2=0.01, upsample_times=3):
        super().__init__()
        self.predictor = Cif(idim, l_order, r_order, threshold, dropout, smooth_factor, noise_threshold, tail_threshold, residual, cnn_groups)
        self.smooth_factor2 = smooth_factor2
        self.noise_threshold2 = noise_threshold
        self.upsample_times = upsample_times
        self.noise_threshold2 = noise_threshold2
        self.tp_upsample_cnn = torch.nn.ConvTranspose1d(idim, idim, self.upsample_times, self.upsample_times)
        self.tp_blstm = torch.nn.LSTM(idim, idim, 1, bias=True, batch_first=True, dropout=0.0, bidirectional=True)
        self.tp_output = torch.nn.Linear(idim * 2, 1)

    def forward(self, hidden, target_label: 'Optional[torch.Tensor]'=None, mask: 'torch.Tensor'=torch.tensor(0), ignore_id: 'int'=-1, mask_chunk_predictor: 'Optional[torch.Tensor]'=None, target_label_length: 'Optional[torch.Tensor]'=None):
        acoustic_embeds, token_num, alphas, cif_peak = self.predictor(hidden, target_label, mask, ignore_id, mask_chunk_predictor, target_label_length)
        output, (_, _) = self.tp_blstm(self.tp_upsample_cnn(hidden.transpose(1, 2)).transpose(1, 2))
        tp_alphas = torch.sigmoid(self.tp_output(output))
        tp_alphas = torch.nn.functional.relu(tp_alphas * self.smooth_factor2 - self.noise_threshold2)
        mask = mask.repeat(1, self.upsample_times, 1).transpose(-1, -2).reshape(tp_alphas.shape[0], -1)
        mask = mask.unsqueeze(-1)
        tp_alphas = tp_alphas * mask
        tp_alphas = tp_alphas.squeeze(-1)
        tp_token_num = tp_alphas.sum(-1)
        return acoustic_embeds, token_num, alphas, cif_peak, tp_alphas, tp_token_num, mask


class Conv2dValid(_ConvNd):
    """
    Conv2d operator for VALID mode padding.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: '_size_2_t', stride: '_size_2_t'=1, padding: 'Union[str, _size_2_t]'=0, dilation: '_size_2_t'=1, groups: 'int'=1, bias: 'bool'=True, padding_mode: 'str'='zeros', device=None, dtype=None, valid_trigx: 'bool'=False, valid_trigy: 'bool'=False) ->None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(Conv2dValid, self).__init__(in_channels, out_channels, kernel_size_, stride_, padding_, dilation_, False, _pair(0), groups, bias, padding_mode, **factory_kwargs)
        self.valid_trigx = valid_trigx
        self.valid_trigy = valid_trigy

    def _conv_forward(self, input: 'Tensor', weight: 'Tensor', bias: 'Optional[Tensor]'):
        validx, validy = 0, 0
        if self.valid_trigx:
            validx = (input.size(-2) * (self.stride[-2] - 1) - 1 + self.kernel_size[-2]) // 2
        if self.valid_trigy:
            validy = (input.size(-1) * (self.stride[-1] - 1) - 1 + self.kernel_size[-1]) // 2
        return F.conv2d(input, weight, bias, self.stride, (validx, validy), self.dilation, self.groups)

    def forward(self, input: 'Tensor') ->Tensor:
        return self._conv_forward(input, self.weight, self.bias)


class SqueezeformerEncoderLayer(nn.Module):
    """Encoder layer module.
        Args:
            size (int): Input dimension.
            self_attn (torch.nn.Module): Self-attention module instance.
                `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
                instance can be used as the argument.
            feed_forward1 (torch.nn.Module): Feed-forward module instance.
                `PositionwiseFeedForward` instance can be used as the argument.
            conv_module (torch.nn.Module): Convolution module instance.
                `ConvlutionModule` instance can be used as the argument.
            feed_forward2 (torch.nn.Module): Feed-forward module instance.
                `PositionwiseFeedForward` instance can be used as the argument.
            dropout_rate (float): Dropout rate.
            normalize_before (bool):
                True: use layer_norm before each sub-block.
                False: use layer_norm after each sub-block.
        """

    def __init__(self, size: 'int', self_attn: 'torch.nn.Module', feed_forward1: 'Optional[nn.Module]'=None, conv_module: 'Optional[nn.Module]'=None, feed_forward2: 'Optional[nn.Module]'=None, normalize_before: 'bool'=False, dropout_rate: 'float'=0.1, concat_after: 'bool'=False):
        super(SqueezeformerEncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.layer_norm1 = nn.LayerNorm(size)
        self.ffn1 = feed_forward1
        self.layer_norm2 = nn.LayerNorm(size)
        self.conv_module = conv_module
        self.layer_norm3 = nn.LayerNorm(size)
        self.ffn2 = feed_forward2
        self.layer_norm4 = nn.LayerNorm(size)
        self.normalize_before = normalize_before
        self.dropout = nn.Dropout(dropout_rate)
        self.concat_after = concat_after
        if concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        else:
            self.concat_linear = nn.Identity()

    def forward(self, x: 'torch.Tensor', mask: 'torch.Tensor', pos_emb: 'torch.Tensor', mask_pad: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool), att_cache: 'torch.Tensor'=torch.zeros((0, 0, 0, 0)), cnn_cache: 'torch.Tensor'=torch.zeros((0, 0, 0, 0))) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x
        if self.normalize_before:
            x = self.layer_norm1(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb, att_cache)
        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.layer_norm1(x)
        residual = x
        if self.normalize_before:
            x = self.layer_norm2(x)
        x = self.ffn1(x)
        x = residual + self.dropout(x)
        if not self.normalize_before:
            x = self.layer_norm2(x)
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        residual = x
        if self.normalize_before:
            x = self.layer_norm3(x)
        x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
        x = residual + self.dropout(x)
        if not self.normalize_before:
            x = self.layer_norm3(x)
        residual = x
        if self.normalize_before:
            x = self.layer_norm4(x)
        x = self.ffn2(x)
        x = residual + self.dropout(x)
        if not self.normalize_before:
            x = self.layer_norm4(x)
        return x, mask, new_att_cache, new_cnn_cache


class TimeReductionLayer1D(nn.Module):
    """
    Modified NeMo,
    Squeezeformer Time Reduction procedure.
    Downsamples the audio by `stride` in the time dimension.
    Args:
        channel (int): input dimension of
                       MultiheadAttentionMechanism and PositionwiseFeedForward
        out_dim (int): Output dimension of the module.
        kernel_size (int): Conv kernel size for
                           depthwise convolution in convolution module
        stride (int): Downsampling factor in time dimension.
    """

    def __init__(self, channel: 'int', out_dim: 'int', kernel_size: 'int'=5, stride: 'int'=2):
        super(TimeReductionLayer1D, self).__init__()
        self.channel = channel
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = max(0, self.kernel_size - self.stride)
        self.dw_conv = nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, stride=stride, padding=self.padding, groups=channel)
        self.pw_conv = nn.Conv1d(in_channels=channel, out_channels=out_dim, kernel_size=1, stride=1, padding=0, groups=1)
        self.init_weights()

    def init_weights(self):
        dw_max = self.kernel_size ** -0.5
        pw_max = self.channel ** -0.5
        torch.nn.init.uniform_(self.dw_conv.weight, -dw_max, dw_max)
        torch.nn.init.uniform_(self.dw_conv.bias, -dw_max, dw_max)
        torch.nn.init.uniform_(self.pw_conv.weight, -pw_max, pw_max)
        torch.nn.init.uniform_(self.pw_conv.bias, -pw_max, pw_max)

    def forward(self, xs, xs_lens: 'torch.Tensor', mask: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool), mask_pad: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool)):
        xs = xs.transpose(1, 2)
        xs = xs.masked_fill(mask_pad.eq(0), 0.0)
        xs = self.dw_conv(xs)
        xs = self.pw_conv(xs)
        xs = xs.transpose(1, 2)
        B, T, D = xs.size()
        mask = mask[:, ::self.stride, ::self.stride]
        mask_pad = mask_pad[:, :, ::self.stride]
        L = mask_pad.size(-1)
        if L - T < 0:
            xs = xs[:, :L - T, :].contiguous()
        else:
            dummy_pad = torch.zeros(B, L - T, D, device=xs.device)
            xs = torch.cat([xs, dummy_pad], dim=1)
        xs_lens = torch.div(xs_lens + 1, 2, rounding_mode='trunc')
        return xs, xs_lens, mask, mask_pad


class TimeReductionLayer2D(nn.Module):

    def __init__(self, kernel_size: 'int'=5, stride: 'int'=2, encoder_dim: 'int'=256):
        super(TimeReductionLayer2D, self).__init__()
        self.encoder_dim = encoder_dim
        self.kernel_size = kernel_size
        self.dw_conv = Conv2dValid(in_channels=encoder_dim, out_channels=encoder_dim, kernel_size=(kernel_size, 1), stride=stride, valid_trigy=True)
        self.pw_conv = Conv2dValid(in_channels=encoder_dim, out_channels=encoder_dim, kernel_size=1, stride=1, valid_trigx=False, valid_trigy=False)
        self.kernel_size = kernel_size
        self.stride = stride
        self.init_weights()

    def init_weights(self):
        dw_max = self.kernel_size ** -0.5
        pw_max = self.encoder_dim ** -0.5
        torch.nn.init.uniform_(self.dw_conv.weight, -dw_max, dw_max)
        torch.nn.init.uniform_(self.dw_conv.bias, -dw_max, dw_max)
        torch.nn.init.uniform_(self.pw_conv.weight, -pw_max, pw_max)
        torch.nn.init.uniform_(self.pw_conv.bias, -pw_max, pw_max)

    def forward(self, xs: 'torch.Tensor', xs_lens: 'torch.Tensor', mask: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool), mask_pad: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool)) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        xs = xs.masked_fill(mask_pad.transpose(1, 2).eq(0), 0.0)
        xs = xs.unsqueeze(2)
        padding1 = self.kernel_size - self.stride
        xs = F.pad(xs, (0, 0, 0, 0, 0, padding1, 0, 0), mode='constant', value=0.0)
        xs = self.dw_conv(xs.permute(0, 3, 1, 2))
        xs = self.pw_conv(xs).permute(0, 3, 2, 1).squeeze(1).contiguous()
        tmp_length = xs.size(1)
        xs_lens = torch.div(xs_lens + 1, 2, rounding_mode='trunc')
        padding2 = max(0, (xs_lens.max() - tmp_length).data.item())
        batch_size, hidden = xs.size(0), xs.size(-1)
        dummy_pad = torch.zeros(batch_size, padding2, hidden, device=xs.device)
        xs = torch.cat([xs, dummy_pad], dim=1)
        mask = mask[:, ::2, ::2]
        mask_pad = mask_pad[:, :, ::2]
        return xs, xs_lens, mask, mask_pad


class TimeReductionLayerStream(nn.Module):
    """
    Squeezeformer Time Reduction procedure.
    Downsamples the audio by `stride` in the time dimension.
    Args:
        channel (int): input dimension of
            MultiheadAttentionMechanism and PositionwiseFeedForward
        out_dim (int): Output dimension of the module.
        kernel_size (int): Conv kernel size for
            depthwise convolution in convolution module
        stride (int): Downsampling factor in time dimension.
    """

    def __init__(self, channel: 'int', out_dim: 'int', kernel_size: 'int'=1, stride: 'int'=2):
        super(TimeReductionLayerStream, self).__init__()
        self.channel = channel
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dw_conv = nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, stride=stride, padding=0, groups=channel)
        self.pw_conv = nn.Conv1d(in_channels=channel, out_channels=out_dim, kernel_size=1, stride=1, padding=0, groups=1)
        self.init_weights()

    def init_weights(self):
        dw_max = self.kernel_size ** -0.5
        pw_max = self.channel ** -0.5
        torch.nn.init.uniform_(self.dw_conv.weight, -dw_max, dw_max)
        torch.nn.init.uniform_(self.dw_conv.bias, -dw_max, dw_max)
        torch.nn.init.uniform_(self.pw_conv.weight, -pw_max, pw_max)
        torch.nn.init.uniform_(self.pw_conv.bias, -pw_max, pw_max)

    def forward(self, xs, xs_lens: 'torch.Tensor', mask: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool), mask_pad: 'torch.Tensor'=torch.ones((0, 0, 0), dtype=torch.bool)):
        xs = xs.transpose(1, 2)
        xs = xs.masked_fill(mask_pad.eq(0), 0.0)
        xs = self.dw_conv(xs)
        xs = self.pw_conv(xs)
        xs = xs.transpose(1, 2)
        B, T, D = xs.size()
        mask = mask[:, ::self.stride, ::self.stride]
        mask_pad = mask_pad[:, :, ::self.stride]
        L = mask_pad.size(-1)
        if L - T < 0:
            xs = xs[:, :L - T, :].contiguous()
        else:
            dummy_pad = torch.zeros(B, L - T, D, device=xs.device)
            xs = torch.cat([xs, dummy_pad], dim=1)
        xs_lens = torch.div(xs_lens + 1, 2, rounding_mode='trunc')
        return xs, xs_lens, mask, mask_pad


class SqueezeformerEncoder(nn.Module):

    def __init__(self, input_size: 'int'=80, encoder_dim: 'int'=256, output_size: 'int'=256, attention_heads: 'int'=4, num_blocks: 'int'=12, reduce_idx: 'Optional[Union[int, List[int]]]'=5, recover_idx: 'Optional[Union[int, List[int]]]'=11, feed_forward_expansion_factor: 'int'=4, dw_stride: 'bool'=False, input_dropout_rate: 'float'=0.1, pos_enc_layer_type: 'str'='rel_pos', time_reduction_layer_type: 'str'='conv1d', do_rel_shift: 'bool'=True, feed_forward_dropout_rate: 'float'=0.1, attention_dropout_rate: 'float'=0.1, cnn_module_kernel: 'int'=31, cnn_norm_type: 'str'='batch_norm', dropout: 'float'=0.1, causal: 'bool'=False, adaptive_scale: 'bool'=True, activation_type: 'str'='swish', init_weights: 'bool'=True, global_cmvn: 'torch.nn.Module'=None, normalize_before: 'bool'=False, use_dynamic_chunk: 'bool'=False, concat_after: 'bool'=False, static_chunk_size: 'int'=0, use_dynamic_left_chunk: 'bool'=False):
        """Construct SqueezeformerEncoder

        Args:
            input_size to use_dynamic_chunk, see in Transformer BaseEncoder.
            encoder_dim (int): The hidden dimension of encoder layer.
            output_size (int): The output dimension of final projection layer.
            attention_heads (int): Num of attention head in attention module.
            num_blocks (int): Num of encoder layers.
            reduce_idx Optional[Union[int, List[int]]]:
                reduce layer index, from 40ms to 80ms per frame.
            recover_idx Optional[Union[int, List[int]]]:
                recover layer index, from 80ms to 40ms per frame.
            feed_forward_expansion_factor (int): Enlarge coefficient of FFN.
            dw_stride (bool): Whether do depthwise convolution
                              on subsampling module.
            input_dropout_rate (float): Dropout rate of input projection layer.
            pos_enc_layer_type (str): Self attention type.
            time_reduction_layer_type (str): Conv1d or Conv2d reduction layer.
            do_rel_shift (bool): Whether to do relative shift
                                 operation on rel-attention module.
            cnn_module_kernel (int): Kernel size of CNN module.
            activation_type (str): Encoder activation function type.
            use_cnn_module (bool): Whether to use convolution module.
            cnn_module_kernel (int): Kernel size of convolution module.
            adaptive_scale (bool): Whether to use adaptive scale.
            init_weights (bool): Whether to initialize weights.
            causal (bool): whether to use causal convolution or not.
        """
        super(SqueezeformerEncoder, self).__init__()
        self.global_cmvn = global_cmvn
        self.reduce_idx: 'Optional[Union[int, List[int]]]' = [reduce_idx] if type(reduce_idx) == int else reduce_idx
        self.recover_idx: 'Optional[Union[int, List[int]]]' = [recover_idx] if type(recover_idx) == int else recover_idx
        self.check_ascending_list()
        if reduce_idx is None:
            self.time_reduce = None
        else:
            if recover_idx is None:
                self.time_reduce = 'normal'
            else:
                self.time_reduce = 'recover'
                assert len(self.reduce_idx) == len(self.recover_idx)
            self.reduce_stride = 2
        self._output_size = output_size
        self.normalize_before = normalize_before
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.pos_enc_layer_type = pos_enc_layer_type
        activation = WENET_ACTIVATION_CLASSES[activation_type]()
        if pos_enc_layer_type != 'rel_pos':
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = attention_heads, output_size, attention_dropout_rate
        else:
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = attention_heads, encoder_dim, attention_dropout_rate, do_rel_shift, adaptive_scale, init_weights
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = encoder_dim, encoder_dim * feed_forward_expansion_factor, feed_forward_dropout_rate, activation, adaptive_scale, init_weights
        convolution_layer = ConvolutionModule
        convolution_layer_args = encoder_dim, cnn_module_kernel, activation, cnn_norm_type, causal, True, adaptive_scale, init_weights
        self.embed = DepthwiseConv2dSubsampling4(1, encoder_dim, RelPositionalEncoding(encoder_dim, dropout_rate=0.1), dw_stride, input_size, input_dropout_rate, init_weights)
        self.preln = nn.LayerNorm(encoder_dim)
        self.encoders = torch.nn.ModuleList([SqueezeformerEncoderLayer(encoder_dim, encoder_selfattn_layer(*encoder_selfattn_layer_args), positionwise_layer(*positionwise_layer_args), convolution_layer(*convolution_layer_args), positionwise_layer(*positionwise_layer_args), normalize_before, dropout, concat_after) for _ in range(num_blocks)])
        if time_reduction_layer_type == 'conv1d':
            time_reduction_layer = TimeReductionLayer1D
            time_reduction_layer_args = {'channel': encoder_dim, 'out_dim': encoder_dim}
        elif time_reduction_layer_type == 'stream':
            time_reduction_layer = TimeReductionLayerStream
            time_reduction_layer_args = {'channel': encoder_dim, 'out_dim': encoder_dim}
        else:
            time_reduction_layer = TimeReductionLayer2D
            time_reduction_layer_args = {'encoder_dim': encoder_dim}
        self.time_reduction_layer = time_reduction_layer(**time_reduction_layer_args)
        self.time_recover_layer = nn.Linear(encoder_dim, encoder_dim)
        self.final_proj = None
        if output_size != encoder_dim:
            self.final_proj = nn.Linear(encoder_dim, output_size)

    def output_size(self) ->int:
        return self._output_size

    def forward(self, xs: 'torch.Tensor', xs_lens: 'torch.Tensor', decoding_chunk_size: 'int'=0, num_decoding_left_chunks: 'int'=-1) ->Tuple[torch.Tensor, torch.Tensor]:
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks
        chunk_masks = add_optional_chunk_mask(xs, masks, self.use_dynamic_chunk, self.use_dynamic_left_chunk, decoding_chunk_size, self.static_chunk_size, num_decoding_left_chunks)
        xs_lens = mask_pad.squeeze(1).sum(1)
        xs = self.preln(xs)
        recover_activations: 'List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]' = []
        index = 0
        for i, layer in enumerate(self.encoders):
            if self.reduce_idx is not None:
                if self.time_reduce is not None and i in self.reduce_idx:
                    recover_activations.append((xs, chunk_masks, pos_emb, mask_pad))
                    xs, xs_lens, chunk_masks, mask_pad = self.time_reduction_layer(xs, xs_lens, chunk_masks, mask_pad)
                    pos_emb = pos_emb[:, ::2, :]
                    index += 1
            if self.recover_idx is not None:
                if self.time_reduce == 'recover' and i in self.recover_idx:
                    index -= 1
                    recover_tensor, recover_chunk_masks, recover_pos_emb, recover_mask_pad = recover_activations[index]
                    xs = xs.unsqueeze(2).repeat(1, 1, 2, 1).flatten(1, 2)
                    xs = self.time_recover_layer(xs)
                    recoverd_t = recover_tensor.size(1)
                    xs = recover_tensor + xs[:, :recoverd_t, :].contiguous()
                    chunk_masks = recover_chunk_masks
                    pos_emb = recover_pos_emb
                    mask_pad = recover_mask_pad
                    xs = xs.masked_fill(~mask_pad[:, 0, :].unsqueeze(-1), 0.0)
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        if self.final_proj is not None:
            xs = self.final_proj(xs)
        return xs, masks

    def check_ascending_list(self):
        if self.reduce_idx is not None:
            assert self.reduce_idx == sorted(self.reduce_idx), 'reduce_idx should be int or ascending list'
        if self.recover_idx is not None:
            assert self.recover_idx == sorted(self.recover_idx), 'recover_idx should be int or ascending list'

    def calculate_downsampling_factor(self, i: 'int') ->int:
        if self.reduce_idx is None:
            return 1
        else:
            reduce_exp, recover_exp = 0, 0
            for exp, rd_idx in enumerate(self.reduce_idx):
                if i >= rd_idx:
                    reduce_exp = exp + 1
            if self.recover_idx is not None:
                for exp, rc_idx in enumerate(self.recover_idx):
                    if i >= rc_idx:
                        recover_exp = exp + 1
            return int(2 ** (reduce_exp - recover_exp))

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
        mask_pad = torch.ones(1, xs.size(1), device=xs.device, dtype=torch.bool)
        mask_pad = mask_pad.unsqueeze(1)
        max_att_len: 'int' = 0
        recover_activations: 'List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]' = []
        index = 0
        xs_lens = torch.tensor([xs.size(1)], device=xs.device, dtype=torch.int)
        xs = self.preln(xs)
        for i, layer in enumerate(self.encoders):
            if self.reduce_idx is not None:
                if self.time_reduce is not None and i in self.reduce_idx:
                    recover_activations.append((xs, att_mask, pos_emb, mask_pad))
                    xs, xs_lens, att_mask, mask_pad = self.time_reduction_layer(xs, xs_lens, att_mask, mask_pad)
                    pos_emb = pos_emb[:, ::2, :]
                    index += 1
            if self.recover_idx is not None:
                if self.time_reduce == 'recover' and i in self.recover_idx:
                    index -= 1
                    recover_tensor, recover_att_mask, recover_pos_emb, recover_mask_pad = recover_activations[index]
                    xs = xs.unsqueeze(2).repeat(1, 1, 2, 1).flatten(1, 2)
                    xs = self.time_recover_layer(xs)
                    recoverd_t = recover_tensor.size(1)
                    xs = recover_tensor + xs[:, :recoverd_t, :].contiguous()
                    att_mask = recover_att_mask
                    pos_emb = recover_pos_emb
                    mask_pad = recover_mask_pad
                    if att_mask.size(1) != 0:
                        xs = xs.masked_fill(~att_mask[:, 0, :].unsqueeze(-1), 0.0)
            factor = self.calculate_downsampling_factor(i)
            xs, _, new_att_cache, new_cnn_cache = layer(xs, att_mask, pos_emb, att_cache=att_cache[i:i + 1][:, :, ::factor, :][:, :, :pos_emb.size(1) - xs.size(1), :] if elayers > 0 else att_cache[:, :, ::factor, :], cnn_cache=cnn_cache[i] if cnn_cache.size(0) > 0 else cnn_cache)
            cached_att = new_att_cache[:, :, next_cache_start // factor:, :]
            cached_cnn = new_cnn_cache.unsqueeze(0)
            cached_att = cached_att.unsqueeze(3).repeat(1, 1, 1, factor, 1).flatten(2, 3)
            if i == 0:
                max_att_len = cached_att.size(2)
            r_att_cache.append(cached_att[:, :, :max_att_len, :])
            r_cnn_cache.append(cached_cnn)
        r_att_cache = torch.cat(r_att_cache, dim=0)
        r_cnn_cache = torch.cat(r_cnn_cache, dim=0)
        if self.final_proj is not None:
            xs = self.final_proj(xs)
        return xs, r_att_cache, r_cnn_cache

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
               with cache, which is not prefered.
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


def compute_mask_indices_v2(shape, padding_mask, mask_prob: 'float', mask_length: 'int', mask_type: 'str'='static', mask_other: 'float'=0.0, min_masks: 'int'=2, no_overlap: 'bool'=False, min_space: 'int'=1, device=torch.device('cpu')):
    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)
    padding_mask = padding_mask.cpu().numpy()
    all_num_mask = int(mask_prob * all_sz / float(mask_length) + np.random.rand())
    all_num_mask = max(min_masks, all_num_mask)
    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None and not isinstance(padding_mask, bytes):
            sz = all_sz - padding_mask[i].sum()
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

            def arrange(s, e, length, keep_length, mask_idc):
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
                parts.extend(arrange(s, e, length, min_length, mask_idc))
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
    mask = torch.from_numpy(mask)
    return mask


def make_non_pad_mask(lengths: 'torch.Tensor') ->torch.Tensor:
    """Make mask tensor containing indices of non-padded part.

    The sequences in a batch may have different lengths. To enable
    batch computing, padding is need to make all sequence in same
    size. To avoid the padding part pass value to context dependent
    block such as attention or convolution , this padding part is
    masked.

    This pad_mask is used in both encoder and decoder.

    1 for non-padded part and 0 for padded part.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
    """
    return ~make_pad_mask(lengths)


def quantize_vector(latent: 'torch.Tensor', codebook: 'torch.Tensor'):
    """
    Symbols in comments:
    B: batch_size.
    D: latent_dim.
    C: num_latent_classes per group
    G: num of codebook groups.

    Args:
        latent: [B, D]
        codebook: [C, G, D // G]

    Returns:
        (quantized, codes, onehot).
         - quantized: [B, D]
         - codes:     [B, G]
         - onehot:    [B, G, C]
    """
    assert len(codebook.size()) == 3
    b, d = latent.size()
    c, g, _ = codebook.size()
    assert d % g == 0
    latent = latent.reshape(b, g, d // g)
    distance = torch.sum(latent ** 2, -1, keepdim=True) - 2 * torch.einsum('bgd,cgd->bgc', latent, codebook) + torch.sum(codebook.permute([2, 1, 0]) ** 2, 0, keepdim=True)
    codes = torch.argmin(distance, dim=-1)
    one_hot = torch.nn.functional.one_hot(codes, c).type(codebook.dtype)
    quantized = torch.einsum('bgc,cgd->bgd', one_hot, codebook)
    quantized = torch.reshape(quantized, [b, d])
    return quantized, codes, one_hot


class BestRQModel(torch.nn.Module):

    def __init__(self, encoder: 'torch.nn.Module', num_mel_bins: 'int'=80, embedding_dim: 'int'=16, num_embeddings: 'int'=8192, num_codebooks: 'int'=1, mask_prob: 'float'=0.01, mask_length: 'int'=10, min_masks: 'int'=2, norm_epsilon: 'float'=1e-05, out_bias: 'bool'=False, features_regularization_weight: 'float'=0.01) ->None:
        super().__init__()
        assert mask_prob > 0.0
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.min_masks = min_masks
        self.num_codebooks = num_codebooks
        self.num_embeddings = num_embeddings
        self.features_regularization_weight = features_regularization_weight
        self.encoder = encoder
        self.encoder_top_n_out = torch.nn.parameter.Parameter(torch.empty(self.num_codebooks, self.encoder.output_size(), num_embeddings))
        torch.nn.init.trunc_normal_(self.encoder_top_n_out, std=0.02)
        self.out_bias = out_bias
        if self.out_bias:
            self.encoder_top_n_out_bias = torch.nn.parameter.Parameter(torch.empty(self.num_codebooks, num_embeddings))
            torch.nn.init.zeros_(self.encoder_top_n_out_bias)
        self.stack_frames = self.encoder.embed.right_context + 1
        self.stride = self.encoder.embed.subsampling_rate
        input_dim = num_mel_bins * self.stride
        self.projection = torch.nn.parameter.Parameter(torch.empty(input_dim, embedding_dim * self.num_codebooks), requires_grad=False)
        torch.nn.init.xavier_uniform_(self.projection)
        self.embeddings = torch.nn.parameter.Parameter(torch.empty(num_embeddings, self.num_codebooks, embedding_dim), requires_grad=False)
        torch.nn.init.normal_(self.embeddings)
        self.embeddings /= self.embeddings.norm(dim=-1, p=2, keepdim=True) + 1e-08
        self.reset_encoder_parameter()

    def reset_encoder_parameter(self):

        def _reset_parameter(module: 'torch.nn.Module'):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                    torch.nn.init.uniform_(module.bias, a=-k, b=k)
            elif isinstance(module, torch.Tensor):
                torch.nn.init.trunc_normal_(module)
            else:
                raise NotImplementedError('other module not support now')
        encoders = self.encoder.encoders
        for _, layer in enumerate(encoders):
            self_attn = layer.self_attn
            _reset_parameter(self_attn.linear_q)
            _reset_parameter(self_attn.linear_k)
            _reset_parameter(self_attn.linear_v)
            _reset_parameter(self_attn.linear_out)
            if isinstance(self_attn, RelPositionMultiHeadedAttention):
                _reset_parameter(self_attn.pos_bias_u)
                _reset_parameter(self_attn.pos_bias_v)
            if isinstance(layer, ConformerEncoderLayer):
                conv1, conv2 = layer.conv_module.pointwise_conv1, layer.conv_module.depthwise_conv
                _reset_parameter(conv1)
                _reset_parameter(conv2)

    def forward(self, batch: 'Dict', device: 'torch.device'):
        xs = batch['feats']
        xs_lens = batch['feats_lengths']
        input = xs
        features_pen: 'Optional[torch.Tensor]' = None
        if self.features_regularization_weight != 0.0:
            features_pen = input.pow(2).mean()
        xs, code_ids_mask = self._apply_mask_signal(xs, xs_lens)
        unmasked_xs = self._stack_features(input, xs_lens)
        masked_xs = xs
        target_ids = self._nearest_embedding_idx(unmasked_xs)
        target_ids = target_ids[:, :code_ids_mask.size(1), :]
        out, out_mask = self.encoder(masked_xs, xs_lens)
        out = out.unsqueeze(1)
        top_n_out = self.encoder_top_n_out.unsqueeze(0)
        out = torch.matmul(out, top_n_out)
        if self.out_bias:
            out = out + self.encoder_top_n_out_bias.unsqueeze(0).unsqueeze(2)
        masks = out_mask.squeeze(1) * code_ids_mask
        loss = self._compute_loss(out, target_ids, mask=masks)
        if self.features_regularization_weight != 0.0:
            loss = loss + self.features_regularization_weight * features_pen
        num_codes = masks.sum() * self.num_codebooks
        uniq_num_codes = torch.tensor(torch.unique(target_ids * masks.unsqueeze(2)).numel()).detach()
        ids_corr = out.argmax(dim=-1, keepdim=False).transpose(1, 2) == target_ids
        codes_acc = (ids_corr * masks.unsqueeze(2)).sum() / num_codes
        return {'codes_acc': codes_acc, 'features_l2': features_pen, 'loss': loss, 'num_codes': num_codes, 'uniq_num_codes': uniq_num_codes, 'th_accuracy': codes_acc}

    def _apply_mask_signal(self, input: 'torch.Tensor', input_lens: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        device = input.device
        B, T, _ = input.size()
        padding_mask = make_pad_mask(input_lens)
        padding_mask_stride = padding_mask.unfold(1, size=self.stack_frames, step=self.stride)
        padding_mask, _ = torch.max(padding_mask_stride, dim=-1)
        masks = compute_mask_indices_v2(padding_mask.size(), padding_mask, self.mask_prob, self.mask_length, min_masks=self.min_masks, device=device)
        subsampling_mask = masks
        bool_stride_mask = torch.ones_like(padding_mask_stride, device=device)
        mask_stride = torch.where(masks.unsqueeze(-1), bool_stride_mask, False)
        masks = mask_stride[:, :, :self.stride].flatten(start_dim=1)
        masks_padding = torch.zeros(B, T, device=device, dtype=padding_mask.dtype)
        masks_padding[:, :masks.size(-1)] = masks
        masks = masks_padding
        masks_expand = masks.unsqueeze(-1)
        mask_emb = torch.normal(mean=0, std=0.1, size=(1, 1, input.size(2)))
        xs = torch.where(masks_expand, mask_emb, input)
        return xs, subsampling_mask

    def _stack_features(self, input: 'torch.Tensor', input_lens: 'torch.Tensor') ->torch.Tensor:
        stack_input = input.unfold(1, size=self.stride, step=self.stride)
        stack_input = stack_input.transpose(-1, -2)
        b, n, f, d = stack_input.size()
        stack_input = stack_input.reshape(b, n, f * d)
        mask = make_non_pad_mask(input_lens)
        stack_mask = mask.unfold(1, size=self.stride, step=self.stride)
        stack_mask, _ = torch.min(stack_mask, dim=-1)
        stack_input = stack_input * stack_mask.unsqueeze(2)
        mean = stack_input.sum(1, keepdim=True) / stack_mask.sum(dim=1, keepdim=True).unsqueeze(1)
        std = torch.sqrt(((stack_input - mean) ** 2).sum(dim=1, keepdim=True) / stack_mask.sum(dim=1, keepdim=True).unsqueeze(1))
        norm_stack_input = (stack_input - mean) / (std + 1e-05)
        return norm_stack_input

    def _compute_loss(self, input: 'torch.Tensor', target: 'torch.Tensor', mask: 'torch.Tensor') ->torch.Tensor:
        logits = input.transpose(1, 2).contiguous().view(-1, input.size(-1))
        loss = torch.nn.functional.cross_entropy(logits, target.contiguous().view(-1), reduction='none')
        loss = (loss * mask.view(-1)).sum() / mask.sum()
        return loss

    def _nearest_embedding_idx(self, xs: 'torch.Tensor') ->torch.Tensor:
        xs = torch.matmul(xs, self.projection)
        xs = xs / (xs.norm(dim=-1, p=2, keepdim=True) + 1e-08)
        codebooks = self.embeddings
        B, T, C = xs.size()
        xs_flatten = xs.view(B * T, C)
        _, codes, _ = quantize_vector(xs_flatten, codebooks)
        return codes.reshape(B, T, -1)


def gumbel(shape: 'torch.Size', dtype: 'torch.dtype', device: 'torch.device'):
    """Sample Gumbel random values with given shape and float dtype.

    The values are distributed according to the probability density function:

    .. math::
     f(x) = e^{-(x + e^{-x})}

    Args:
      shape (torch.Size): pdf shape
      dtype (torch.dtype): pdf value dtype

    Returns:
       A random array with the specified shape and dtype.
    """
    return -torch.log(-torch.log(torch.empty(shape, device=device).uniform_(torch.finfo(dtype).tiny, 1.0)))


class Wav2vecGumbelVectorQuantizer(torch.nn.Module):

    def __init__(self, features_dim: 'int'=256, num_codebooks: 'int'=2, num_embeddings: 'int'=8192, embedding_dim: 'int'=16, hard: 'bool'=False) ->None:
        super().__init__()
        self.num_groups = num_codebooks
        self.num_codevectors_per_group = num_embeddings
        assert embedding_dim % num_codebooks == 0.0
        self.embeddings = torch.nn.parameter.Parameter(torch.empty(1, num_codebooks * num_embeddings, embedding_dim // num_codebooks), requires_grad=True)
        torch.nn.init.uniform_(self.embeddings)
        self.weight_proj = torch.nn.Linear(features_dim, num_codebooks * num_embeddings)
        self.hard = hard

    @staticmethod
    def _compute_perplexity(probs, mask=None):
        if mask is not None:
            mask_extended = torch.broadcast_to(mask.flatten()[:, None, None], probs.shape)
            probs = torch.where(mask_extended, probs, torch.zeros_like(probs))
            marginal_probs = probs.sum(dim=0) / mask.sum()
        else:
            marginal_probs = probs.mean(dim=0)
        perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-07), dim=-1)).sum()
        return perplexity

    def forward(self, input: 'torch.Tensor', input_mask: 'torch.Tensor', temperature: 'float'=1.0) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, _ = input.size()
        hidden = self.weight_proj(input)
        hidden = hidden.reshape(b * t * self.num_groups, -1)
        if not self.hard:
            gumbels = gumbel(hidden.size(), hidden.dtype, hidden.device)
            codevector_probs = torch.nn.functional.softmax((hidden + gumbels) / temperature, dim=-1)
            codevector_soft_dist = torch.nn.functional.softmax(hidden.reshape(b * t, self.num_groups, -1), dim=-1)
            perplexity = self._compute_perplexity(codevector_soft_dist, input_mask)
        else:
            codevector_idx = hidden.argmax(axis=-1)
            codevector_probs = torch.nn.functional.one_hot(codevector_idx, hidden.shape[-1]) * 1.0
            codevector_probs = codevector_probs.reshape(b * t, self.num_groups, -1)
            perplexity = self._compute_perplexity(codevector_probs, input_mask)
        targets_idx = codevector_probs.argmax(-1).reshape(b, t, -1)
        codevector_probs = codevector_probs.reshape(b * t, -1)
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.embeddings
        codevectors = codevectors_per_group.reshape(b * t, self.num_groups, self.num_codevectors_per_group, -1)
        codevectors = codevectors.sum(-2).reshape(b, t, -1)
        return codevectors, perplexity, targets_idx


def _compute_contrastive_loss(quantized_features: 'torch.Tensor', features: 'torch.Tensor', negative_indices: 'torch.Tensor', mask_time_indices: 'torch.Tensor', logits_temp: 'float', num_negatives: 'int'=1):
    batch_size, sequence_length, hidden_size = quantized_features.shape
    quantized_negatives = quantized_features.view(-1, hidden_size)[negative_indices.view(-1)]
    quantized_negatives = quantized_negatives.view(batch_size, sequence_length, num_negatives, hidden_size).permute(2, 0, 1, 3)
    target_features = torch.cat([quantized_features.unsqueeze(0), quantized_negatives], dim=0)
    loss_logits = F.cosine_similarity(features, target_features, dim=-1)
    loss_logits = loss_logits / logits_temp
    neg_is_pos = (quantized_features == quantized_negatives).all(-1)
    neg_is_pos = torch.cat([torch.full((1,) + loss_logits.shape[1:], False, device=neg_is_pos.device), neg_is_pos], dim=0)
    loss_logits = torch.where(neg_is_pos, -1000000000.0, loss_logits)
    predictions = loss_logits.permute(2, 1, 0).reshape(-1, loss_logits.shape[0])
    targets = ((1 - mask_time_indices.long()) * -100).transpose(1, 0).flatten()
    target_mask = torch.where(targets >= 0, 1.0, 0.0)
    contrastive_loss = F.cross_entropy(predictions, targets.long(), reduction='none') * target_mask
    contrastive_loss = contrastive_loss.sum()
    return contrastive_loss


def _sample_negative_indices(features_shape: 'Tuple', num_negatives: 'int', device: 'torch.device', mask_time_indices: 'Optional[torch.Tensor]'=None):
    """
    Sample `num_negatives` vectors from feature vectors.
    """
    batch_size, sequence_length = features_shape
    sequence_length_range = torch.arange(sequence_length, device=device)
    sampled_negative_indices = torch.zeros((batch_size, sequence_length, num_negatives), dtype=sequence_length_range.dtype, device=device)
    mask_time_indices = mask_time_indices.bool() if mask_time_indices is not None else torch.ones(features_shape, dtype=torch.bool, device=device)
    for batch_idx in range(batch_size):
        high = mask_time_indices[batch_idx].sum() - 1
        mapped_masked_indices = sequence_length_range[mask_time_indices[batch_idx]]
        feature_indices = torch.arange(high + 1).unsqueeze(1).expand(high + 1, num_negatives)
        sampled_indices = torch.randint(0, high, size=(high + 1, num_negatives))
        sampled_indices[sampled_indices >= feature_indices] += 1
        sampled_negative_indices[batch_idx][mask_time_indices[batch_idx]] = mapped_masked_indices[sampled_indices]
        sampled_negative_indices[batch_idx] += batch_idx * sequence_length
    return sampled_negative_indices.reshape(batch_size, -1)


class W2VBERTModel(torch.nn.Module):

    def __init__(self, encoder: 'Union[ConformerEncoder, TransformerEncoder]', embedding_dim: 'int'=256, num_embeddings: 'int'=320, num_codebooks: 'int'=1, mask_prob: 'float'=0.065, mask_length: 'int'=10, min_masks: 'int'=2, num_negatives: 'int'=100, features_regularization_weight: 'float'=0.01, max_gumbel_temperature: 'float'=2.0, min_gumbel_temperature: 'float'=0.1, gumbel_temperature_decay: 'float'=0.999995, contrastive_logits_temperature: 'float'=0.1, diversity_weight: 'float'=0.0, bias: 'bool'=True, contrastive_blocks: 'int'=6, masked_blocks: 'int'=6, contrastive_weight: 'float'=1.0, mlm_weight: 'float'=1.0, warmup_steps: 'int'=25000) ->None:
        """ Wrap encoder to train using W2V-BERT's style

        Described in:
        https://arxiv.org/pdf/2108.06209v2.pdf

        Args:
            encoder: wenet's encoder,
                     only support conformer and transformer now
            embedding_dim: codebooks embedding dim
            num_embeddings: numbers of each codebook
            num_codebooks: numbers of codebooks i.e groups of codebook
            mask_prob: probs of mask
            mask_length: spans of masks
            min_masks: min masks for each audio
            num_negatives: numbers of negatives of each masks
            features_regularization_weight: l2 regularization weight
            max_gumbel_temperature: maximum temperature for gumbel softmax
            min_gumbel_temperature: minimum temperature for gumbel softmax
            gumbel_temperature_decay:
                decay of gumbel temperature during training
            contrastive_logits_temperature:
                the temperature in the contrastive loss.
        """
        super().__init__()
        assert mask_prob > 0.0
        assert contrastive_blocks > 0 and masked_blocks > 0 and contrastive_blocks + masked_blocks == len(encoder.encoders)
        self.contrastive_blocks = contrastive_blocks
        self.masked_blocks = masked_blocks
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.min_masks = min_masks
        self.num_negatives = num_negatives
        self.features_regularization_weight = features_regularization_weight
        self.diversity_weight = diversity_weight
        self.contrastive_weight = contrastive_weight
        self.mlm_weight = mlm_weight
        self.warmup_steps = warmup_steps
        self.encoder = encoder
        self.num_codebooks = num_codebooks
        self.quantizer = Wav2vecGumbelVectorQuantizer(self.encoder.output_size(), num_codebooks=num_codebooks, num_embeddings=num_embeddings, embedding_dim=embedding_dim, hard=False)
        self.max_gumbel_temp = max_gumbel_temperature
        self.min_gumbel_temp = min_gumbel_temperature
        self.gumbel_temp_decay = gumbel_temperature_decay
        self.num_codevectors_per_group = num_embeddings
        self.num_codevector_groups = num_codebooks
        self.contrastive_logits_temp = contrastive_logits_temperature
        self.encoder_top_n_out = torch.nn.parameter.Parameter(torch.empty(num_codebooks, self.encoder.output_size(), num_embeddings))
        torch.nn.init.trunc_normal_(self.encoder_top_n_out, std=0.02)
        self.bias = bias
        if bias:
            self.encoder_top_n_out_bias = torch.nn.parameter.Parameter(torch.empty(num_codebooks, num_embeddings))
            torch.nn.init.zeros_(self.encoder_top_n_out_bias)
        self.reset_encoder_parameter()

    def reset_encoder_parameter(self):

        def _reset_parameter(module: 'torch.nn.Module'):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                    torch.nn.init.uniform_(module.bias, a=-k, b=k)
            elif isinstance(module, torch.Tensor):
                torch.nn.init.trunc_normal_(module)
            else:
                raise NotImplementedError('other module not support now')
        encoders = self.encoder.encoders
        for _, layer in enumerate(encoders):
            self_attn = layer.self_attn
            _reset_parameter(self_attn.linear_q)
            _reset_parameter(self_attn.linear_k)
            _reset_parameter(self_attn.linear_v)
            _reset_parameter(self_attn.linear_out)
            if isinstance(self_attn, RelPositionMultiHeadedAttention):
                _reset_parameter(self_attn.pos_bias_u)
                _reset_parameter(self_attn.pos_bias_v)
            if isinstance(layer, ConformerEncoderLayer):
                conv1, conv2 = layer.conv_module.pointwise_conv1, layer.conv_module.depthwise_conv
                _reset_parameter(conv1)
                _reset_parameter(conv2)

    @torch.jit.unused
    def forward(self, batch: 'Dict', device: 'torch.device'):
        steps = batch.get('steps', None)
        xs = batch['feats']
        xs_lens = batch['feats_lengths']
        assert xs.size(0) == xs_lens.size(0)
        assert steps is not None
        xs, pos_emb, masks = self._forward_subsampling(xs, xs_lens)
        unmasked_xs = xs
        masked_xs, masked_masks = self._apply_mask(xs, masks.squeeze(1))
        contrastive_vec, mlm_vec, out_mask = self._forward_encoder_blocks(masked_xs, masks, pos_emb, masks)
        gumbel_temperature = max(self.max_gumbel_temp * self.gumbel_temp_decay ** steps, self.min_gumbel_temp)
        quantized_features, codevector_perplexity, targets_ids = self.quantizer(unmasked_xs, masks.squeeze(1), gumbel_temperature)
        sampled_negative_indices = _sample_negative_indices(xs.size()[:-1], self.num_negatives, masked_masks.device, masked_masks)
        loss_contrastive = _compute_contrastive_loss(quantized_features, contrastive_vec, sampled_negative_indices, masked_masks, self.contrastive_logits_temp, self.num_negatives)
        loss = loss_contrastive
        sample_size = masked_masks.sum()
        loss_diversity: 'Optional[torch.Tensor]' = None
        if self.diversity_weight != 0.0:
            loss_diversity = (self.num_codevector_groups * self.num_codevectors_per_group - codevector_perplexity) / (self.num_codevectors_per_group * self.num_codevector_groups)
            loss_diversity = loss_diversity * sample_size
            loss = loss + self.diversity_weight * loss_diversity
        loss = loss / sample_size
        features_pen: 'Optional[torch.Tensor]' = None
        if self.features_regularization_weight != 0.0:
            features_pen = xs.pow(2).mean()
            loss = loss + self.features_regularization_weight * features_pen
        out = mlm_vec.unsqueeze(1)
        top_n_out = self.encoder_top_n_out.unsqueeze(0)
        out = torch.matmul(out, top_n_out)
        if self.bias:
            out = out + self.encoder_top_n_out_bias.unsqueeze(0).unsqueeze(2)
        num_codes = masked_masks.sum() * self.num_codebooks
        loss_mlm = self._compute_mlm_loss(out, targets_ids, mask=out_mask.squeeze(1) * masked_masks)
        ids_corr = out.argmax(dim=-1, keepdim=False).transpose(1, 2) == targets_ids
        codes_acc = (ids_corr * masked_masks.unsqueeze(2)).sum() / num_codes
        mlm_weight = self.mlm_weight if steps >= self.warmup_steps else 0.1 + 0.9 * (steps / self.warmup_steps)
        loss = self.contrastive_weight * loss + mlm_weight * loss_mlm
        return {'code_ppl': codevector_perplexity.detach(), 'features_l2': features_pen, 'codes_acc': codes_acc.detach(), 'loss': loss, 'loss_contrastive': loss_contrastive / sample_size, 'loss_diversity': loss_diversity, 'loss_mlm': loss_mlm}

    def _apply_mask(self, xs: 'torch.Tensor', xs_masks: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        masks = compute_mask_indices_v2(xs.size()[:-1], ~xs_masks, self.mask_prob, self.mask_length, min_masks=self.min_masks, device=xs.device)
        masks_expand = masks.unsqueeze(-1)
        mask_emb = torch.normal(mean=0, std=0.1, size=xs.size(), device=xs.device)
        xs = torch.where(masks_expand, mask_emb, xs)
        return xs, masks

    def _compute_mlm_loss(self, input: 'torch.Tensor', target: 'torch.Tensor', mask: 'torch.Tensor') ->torch.Tensor:
        log_probs = torch.log_softmax(input, dim=-1).transpose(1, 2)
        per_example_n_loss = -log_probs.gather(3, target.unsqueeze(3)).squeeze(3)
        numerator = torch.sum(per_example_n_loss * mask.unsqueeze(2))
        denominator = torch.sum(mask) + 1e-05
        loss = numerator / (denominator * self.num_codebooks)
        return loss

    def _forward_subsampling(self, xs: 'torch.Tensor', xs_lens: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        masks = make_non_pad_mask(xs_lens).unsqueeze(1)
        if self.encoder.global_cmvn is not None:
            xs = self.encoder.global_cmvn(xs)
        xs, pos_emb, masks = self.encoder.embed(xs, masks)
        return xs, pos_emb, masks

    def _forward_encoder_blocks(self, xs: 'torch.Tensor', xs_masks: 'torch.Tensor', pos_emb: 'torch.Tensor', mask_pad: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        masks = xs_masks
        xs: 'torch.Tensor'
        for layer in self.encoder.encoders[:self.contrastive_blocks]:
            xs, masks, _, _ = layer(xs, xs_masks, pos_emb, mask_pad)
        contrastive_vec = xs
        for layer in self.encoder.encoders[self.contrastive_blocks:]:
            xs, masks, _, _ = layer(xs, xs_masks, pos_emb, mask_pad)
        masked_vec = xs
        if self.encoder.normalize_before:
            xs = self.encoder.after_norm(xs)
            masked_vec = xs
        return contrastive_vec, masked_vec, masks


class Wav2vec2Model(torch.nn.Module):

    def __init__(self, encoder: 'Union[ConformerEncoder, TransformerEncoder]', embedding_dim: 'int'=256, num_embeddings: 'int'=320, num_codebooks: 'int'=1, mask_prob: 'float'=0.065, mask_length: 'int'=10, min_masks: 'int'=2, num_negatives: 'int'=100, features_regularization_weight: 'float'=0.01, max_gumbel_temperature: 'float'=2.0, min_gumbel_temperature: 'float'=0.1, gumbel_temperature_decay: 'float'=0.999995, contrastive_logits_temperature: 'float'=0.1, diversity_weight: 'float'=0.0) ->None:
        """ Wrap encoder to train using wav2vec2's style

        Args:
            encoder: wenet's encoder,
                     only support conformer and transformer now
            embedding_dim: codebooks embedding dim
            num_embeddings: numbers of each codebook
            num_codebooks: numbers of codebooks i.e groups of codebook
            mask_prob: probs of mask
            mask_length: spans of masks
            min_maks: min masks for each audio
            num_negatives: numbers of negatives of each masks
            features_regularization_weight: l2 regularization weight
            max_gumbel_temperature: maximum temperature for gumbel softmax
            min_gumbel_temperature: minimum temperature for gumbel softmax
            gumbel_temperature_decay:
                decay of gumbel temperature during training
            contrastive_logits_temperature:
                the temperature in the contrastive loss.
        """
        super().__init__()
        assert mask_prob > 0.0
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.min_masks = min_masks
        self.num_negatives = num_negatives
        self.features_regularization_weight = features_regularization_weight
        self.diversity_weight = diversity_weight
        self.encoder = encoder
        self.quantizer = Wav2vecGumbelVectorQuantizer(self.encoder.output_size(), num_codebooks=num_codebooks, num_embeddings=num_embeddings, embedding_dim=embedding_dim, hard=False)
        self.max_gumbel_temp = max_gumbel_temperature
        self.min_gumbel_temp = min_gumbel_temperature
        self.gumbel_temp_decay = gumbel_temperature_decay
        self.num_codevectors_per_group = num_embeddings
        self.num_codevector_groups = num_codebooks
        self.contrastive_logits_temp = contrastive_logits_temperature
        self.mask_emb = torch.nn.parameter.Parameter(torch.empty(self.encoder.output_size()).uniform_(), requires_grad=True)
        self.reset_encoder_parameter()

    def reset_encoder_parameter(self):

        def _reset_parameter(module: 'torch.nn.Module'):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                    torch.nn.init.uniform_(module.bias, a=-k, b=k)
            elif isinstance(module, torch.Tensor):
                torch.nn.init.trunc_normal_(module)
            else:
                raise NotImplementedError('other module not support now')
        encoders = self.encoder.encoders
        for _, layer in enumerate(encoders):
            self_attn = layer.self_attn
            _reset_parameter(self_attn.linear_q)
            _reset_parameter(self_attn.linear_k)
            _reset_parameter(self_attn.linear_v)
            _reset_parameter(self_attn.linear_out)
            if isinstance(self_attn, RelPositionMultiHeadedAttention):
                _reset_parameter(self_attn.pos_bias_u)
                _reset_parameter(self_attn.pos_bias_v)
            if isinstance(layer, ConformerEncoderLayer):
                conv1, conv2 = layer.conv_module.pointwise_conv1, layer.conv_module.depthwise_conv
                _reset_parameter(conv1)
                _reset_parameter(conv2)

    @torch.jit.unused
    def forward(self, batch: 'Dict', device: 'torch.device'):
        steps = batch.get('steps', None)
        xs = batch['feats']
        xs_lens = batch['feats_lengths']
        assert xs.size(0) == xs_lens.size(0)
        assert steps is not None
        xs, pos_emb, masks = self._forward_subsampling(xs, xs_lens)
        unmasked_xs = xs
        masked_xs, masked_masks = self._apply_mask(xs, masks.squeeze(1))
        out, _ = self._forward_encoder_blocks(masked_xs, masks, pos_emb, masks)
        gumbel_temperature = max(self.max_gumbel_temp * self.gumbel_temp_decay ** steps, self.min_gumbel_temp)
        quantized_features, codevector_perplexity, _ = self.quantizer(unmasked_xs, masks.squeeze(1), gumbel_temperature)
        sampled_negative_indices = _sample_negative_indices(xs.size()[:-1], self.num_negatives, masked_masks.device, masked_masks)
        loss_contrastive = _compute_contrastive_loss(quantized_features, out, sampled_negative_indices, masked_masks, self.contrastive_logits_temp, self.num_negatives)
        loss = loss_contrastive
        sample_size = masked_masks.sum()
        loss_diversity: 'Optional[torch.Tensor]' = None
        if self.diversity_weight != 0.0:
            loss_diversity = (self.num_codevector_groups * self.num_codevectors_per_group - codevector_perplexity) / (self.num_codevectors_per_group * self.num_codevector_groups)
            loss_diversity = loss_diversity * sample_size
            loss = loss + self.diversity_weight * loss_diversity
        loss = loss / sample_size
        features_pen: 'Optional[torch.Tensor]' = None
        if self.features_regularization_weight != 0.0:
            features_pen = xs.pow(2).mean()
            loss = loss + self.features_regularization_weight * features_pen
        return {'code_ppl': codevector_perplexity.detach(), 'features_l2': features_pen, 'loss': loss, 'loss_contrastive': loss_contrastive / sample_size, 'loss_diversity': loss_diversity}

    def _apply_mask(self, xs: 'torch.Tensor', xs_masks: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        masks = compute_mask_indices_v2(xs.size()[:-1], ~xs_masks, self.mask_prob, self.mask_length, min_masks=self.min_masks, device=xs.device)
        masks_expand = masks.unsqueeze(-1)
        mask_emb = self.mask_emb.view(1, 1, -1)
        xs = torch.where(masks_expand, mask_emb, xs)
        return xs, masks

    def _forward_subsampling(self, xs: 'torch.Tensor', xs_lens: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        masks = make_non_pad_mask(xs_lens).unsqueeze(1)
        if self.encoder.global_cmvn is not None:
            xs = self.encoder.global_cmvn(xs)
        xs, pos_emb, masks = self.encoder.embed(xs, masks)
        return xs, pos_emb, masks

    def _forward_encoder_blocks(self, xs: 'torch.Tensor', xs_masks: 'torch.Tensor', pos_emb: 'torch.Tensor', mask_pad: 'torch.Tensor'):
        masks = xs_masks
        for layer in self.encoder.encoders:
            xs, masks, _, _ = layer(xs, xs_masks, pos_emb, mask_pad)
        if self.encoder.normalize_before:
            xs = self.encoder.after_norm(xs)
        return xs, masks


class TransducerJoint(torch.nn.Module):

    def __init__(self, vocab_size: 'int', enc_output_size: 'int', pred_output_size: 'int', join_dim: 'int', prejoin_linear: 'bool'=True, postjoin_linear: 'bool'=False, joint_mode: 'str'='add', activation: 'str'='tanh', hat_joint: 'bool'=False, dropout_rate: 'float'=0.1, hat_activation: 'str'='tanh'):
        assert joint_mode in ['add']
        super().__init__()
        self.activatoin = WENET_ACTIVATION_CLASSES[activation]()
        self.prejoin_linear = prejoin_linear
        self.postjoin_linear = postjoin_linear
        self.joint_mode = joint_mode
        if not self.prejoin_linear and not self.postjoin_linear:
            assert enc_output_size == pred_output_size == join_dim
        self.enc_ffn: 'Optional[nn.Linear]' = None
        self.pred_ffn: 'Optional[nn.Linear]' = None
        if self.prejoin_linear:
            self.enc_ffn = nn.Linear(enc_output_size, join_dim)
            self.pred_ffn = nn.Linear(pred_output_size, join_dim)
        self.post_ffn: 'Optional[nn.Linear]' = None
        if self.postjoin_linear:
            self.post_ffn = nn.Linear(join_dim, join_dim)
        self.hat_joint = hat_joint
        self.vocab_size = vocab_size
        self.ffn_out: 'Optional[torch.nn.Linear]' = None
        if not self.hat_joint:
            self.ffn_out = nn.Linear(join_dim, vocab_size)
        self.blank_pred: 'Optional[torch.nn.Module]' = None
        self.token_pred: 'Optional[torch.nn.Module]' = None
        if self.hat_joint:
            self.blank_pred = torch.nn.Sequential(torch.nn.Tanh(), torch.nn.Dropout(dropout_rate), torch.nn.Linear(join_dim, 1), torch.nn.LogSigmoid())
            self.token_pred = torch.nn.Sequential(WENET_ACTIVATION_CLASSES[hat_activation](), torch.nn.Dropout(dropout_rate), torch.nn.Linear(join_dim, self.vocab_size - 1))

    def forward(self, enc_out: 'torch.Tensor', pred_out: 'torch.Tensor', pre_project: 'bool'=True) ->torch.Tensor:
        """
        Args:
            enc_out (torch.Tensor): [B, T, E]
            pred_out (torch.Tensor): [B, T, P]
        Return:
            [B,T,U,V]
        """
        if pre_project and self.prejoin_linear and self.enc_ffn is not None and self.pred_ffn is not None:
            enc_out = self.enc_ffn(enc_out)
            pred_out = self.pred_ffn(pred_out)
        if enc_out.ndim != 4:
            enc_out = enc_out.unsqueeze(2)
        if pred_out.ndim != 4:
            pred_out = pred_out.unsqueeze(1)
        _ = self.joint_mode
        out = enc_out + pred_out
        if self.postjoin_linear and self.post_ffn is not None:
            out = self.post_ffn(out)
        if not self.hat_joint and self.ffn_out is not None:
            out = self.activatoin(out)
            out = self.ffn_out(out)
            return out
        else:
            assert self.blank_pred is not None
            assert self.token_pred is not None
            blank_logp = self.blank_pred(out)
            scale_logp = torch.clamp(1 - torch.exp(blank_logp), min=1e-06)
            label_logp = self.token_pred(out).log_softmax(dim=-1)
            label_logp = torch.log(scale_logp) + label_logp
            out = torch.cat((blank_logp, label_logp), dim=-1)
            return out


class PredictorBase(torch.nn.Module):

    def __init__(self) ->None:
        super().__init__()

    def init_state(self, batch_size: 'int', device: 'torch.device', method: 'str'='zero') ->List[torch.Tensor]:
        _, _, _ = batch_size, method, device
        raise NotImplementedError('this is a base precictor')

    def batch_to_cache(self, cache: 'List[torch.Tensor]') ->List[List[torch.Tensor]]:
        _ = cache
        raise NotImplementedError('this is a base precictor')

    def cache_to_batch(self, cache: 'List[List[torch.Tensor]]') ->List[torch.Tensor]:
        _ = cache
        raise NotImplementedError('this is a base precictor')

    def output_size(self):
        raise NotImplementedError('this is a base precictor')

    def forward(self, input: 'torch.Tensor', cache: 'Optional[List[torch.Tensor]]'=None):
        _, _ = input, cache
        raise NotImplementedError('this is a base precictor')

    def forward_step(self, input: 'torch.Tensor', padding: 'torch.Tensor', cache: 'List[torch.Tensor]') ->Tuple[torch.Tensor, List[torch.Tensor]]:
        _, _, _ = input, padding, cache
        raise NotImplementedError('this is a base precictor')


def ApplyPadding(input, padding, pad_value) ->torch.Tensor:
    """
    Args:
        input:   [bs, max_time_step, dim]
        padding: [bs, max_time_step]
    """
    return padding * pad_value + input * (1 - padding)


WENET_RNN_CLASSES = {'rnn': torch.nn.RNN, 'lstm': torch.nn.LSTM, 'gru': torch.nn.GRU}


class RNNPredictor(PredictorBase):

    def __init__(self, voca_size: 'int', embed_size: 'int', output_size: 'int', embed_dropout: 'float', hidden_size: 'int', num_layers: 'int', bias: 'bool'=True, rnn_type: 'str'='lstm', dropout: 'float'=0.1) ->None:
        super().__init__()
        self.n_layers = num_layers
        self.hidden_size = hidden_size
        self._output_size = output_size
        self.embed = nn.Embedding(voca_size, embed_size)
        self.dropout = nn.Dropout(embed_dropout)
        self.rnn = WENET_RNN_CLASSES[rnn_type](input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias, batch_first=True, dropout=dropout)
        self.projection = nn.Linear(hidden_size, output_size)

    def output_size(self):
        return self._output_size

    def forward(self, input: 'torch.Tensor', cache: 'Optional[List[torch.Tensor]]'=None) ->torch.Tensor:
        """
        Args:
            input (torch.Tensor): [batch, max_time).
            padding (torch.Tensor): [batch, max_time]
            cache : rnn predictor cache[0] == state_m
                    cache[1] == state_c
        Returns:
            output: [batch, max_time, output_size]
        """
        embed = self.embed(input)
        embed = self.dropout(embed)
        states: 'Optional[Tuple[torch.Tensor, torch.Tensor]]' = None
        if cache is None:
            state = self.init_state(batch_size=input.size(0), device=input.device)
            states = state[0], state[1]
        else:
            assert len(cache) == 2
            states = cache[0], cache[1]
        out, (m, c) = self.rnn(embed, states)
        out = self.projection(out)
        _, _ = m, c
        return out

    def batch_to_cache(self, cache: 'List[torch.Tensor]') ->List[List[torch.Tensor]]:
        """
        Args:
           cache: [state_m, state_c]
               state_ms: [1*n_layers, bs, ...]
               state_cs: [1*n_layers, bs, ...]
        Returns:
           new_cache: [[state_m_1, state_c_1], [state_m_2, state_c_2]...]
        """
        assert len(cache) == 2
        state_ms = cache[0]
        state_cs = cache[1]
        assert state_ms.size(1) == state_cs.size(1)
        new_cache: 'List[List[torch.Tensor]]' = []
        for state_m, state_c in zip(torch.split(state_ms, 1, dim=1), torch.split(state_cs, 1, dim=1)):
            new_cache.append([state_m, state_c])
        return new_cache

    def cache_to_batch(self, cache: 'List[List[torch.Tensor]]') ->List[torch.Tensor]:
        """
        Args:
            cache : [[state_m_1, state_c_1], [state_m_1, state_c_1]...]

        Returns:
            new_caceh: [state_ms, state_cs],
                state_ms: [1*n_layers, bs, ...]
                state_cs: [1*n_layers, bs, ...]
        """
        state_ms = torch.cat([states[0] for states in cache], dim=1)
        state_cs = torch.cat([states[1] for states in cache], dim=1)
        return [state_ms, state_cs]

    def init_state(self, batch_size: 'int', device: 'torch.device', method: 'str'='zero') ->List[torch.Tensor]:
        assert batch_size > 0
        _ = method
        return [torch.zeros(1 * self.n_layers, batch_size, self.hidden_size, device=device), torch.zeros(1 * self.n_layers, batch_size, self.hidden_size, device=device)]

    def forward_step(self, input: 'torch.Tensor', padding: 'torch.Tensor', cache: 'List[torch.Tensor]') ->Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            input (torch.Tensor): [batch_size, time_step=1]
            padding (torch.Tensor): [batch_size,1], 1 is padding value
            cache : rnn predictor cache[0] == state_m
                    cache[1] == state_c
        """
        assert len(cache) == 2
        state_m, state_c = cache[0], cache[1]
        embed = self.embed(input)
        embed = self.dropout(embed)
        out, (m, c) = self.rnn(embed, (state_m, state_c))
        out = self.projection(out)
        m = ApplyPadding(m, padding.unsqueeze(0), state_m)
        c = ApplyPadding(c, padding.unsqueeze(0), state_c)
        return out, [m, c]


class EmbeddingPredictor(PredictorBase):
    """Embedding predictor

    Described in:
    https://arxiv.org/pdf/2109.07513.pdf

    embed-> proj -> layer norm -> swish
    """

    def __init__(self, voca_size: 'int', embed_size: 'int', output_size: 'int', embed_dropout: 'float', n_head: 'int', history_size: 'int'=2, activation: 'str'='swish', bias: 'bool'=False, layer_norm_epsilon: 'float'=1e-05) ->None:
        super().__init__()
        assert output_size == embed_size
        self.num_heads = n_head
        self.embed_size = embed_size
        self.context_size = history_size + 1
        self.pos_embed = torch.nn.Linear(embed_size * self.context_size, self.num_heads, bias=bias)
        self.embed = nn.Embedding(voca_size, self.embed_size)
        self.embed_dropout = nn.Dropout(p=embed_dropout)
        self.ffn = nn.Linear(self.embed_size, self.embed_size)
        self.norm = nn.LayerNorm(self.embed_size, eps=layer_norm_epsilon)
        self.activatoin = WENET_ACTIVATION_CLASSES[activation]()

    def output_size(self):
        return self.embed_size

    def init_state(self, batch_size: 'int', device: 'torch.device', method: 'str'='zero') ->List[torch.Tensor]:
        assert batch_size > 0
        _ = method
        return [torch.zeros(batch_size, self.context_size - 1, self.embed_size, device=device)]

    def batch_to_cache(self, cache: 'List[torch.Tensor]') ->List[List[torch.Tensor]]:
        """
        Args:
            cache : [history]
                history: [bs, ...]
        Returns:
            new_ache : [[history_1], [history_2], [history_3]...]
        """
        assert len(cache) == 1
        cache_0 = cache[0]
        history: 'List[List[torch.Tensor]]' = []
        for h in torch.split(cache_0, 1, dim=0):
            history.append([h])
        return history

    def cache_to_batch(self, cache: 'List[List[torch.Tensor]]') ->List[torch.Tensor]:
        """
        Args:
            cache : [[history_1], [history_2], [history3]...]

        Returns:
            new_caceh: [history],
                history: [bs, ...]
        """
        history = torch.cat([h[0] for h in cache], dim=0)
        return [history]

    def forward(self, input: 'torch.Tensor', cache: 'Optional[List[torch.Tensor]]'=None):
        """ forward for training
        """
        input = self.embed(input)
        input = self.embed_dropout(input)
        if cache is None:
            zeros = self.init_state(input.size(0), device=input.device)[0]
        else:
            assert len(cache) == 1
            zeros = cache[0]
        input = torch.cat((zeros, input), dim=1)
        input = input.unfold(1, self.context_size, 1).permute(0, 1, 3, 2)
        multi_head_pos = self.pos_embed.weight.view(self.num_heads, self.embed_size, self.context_size)
        input_expand = input.unsqueeze(2)
        multi_head_pos = multi_head_pos.permute(0, 2, 1)
        weight = input_expand * multi_head_pos
        weight = weight.sum(dim=-1, keepdim=False).unsqueeze(3)
        output = weight.matmul(input_expand).squeeze(dim=3)
        output = output.sum(dim=2)
        output = output / (self.num_heads * self.context_size)
        output = self.ffn(output)
        output = self.norm(output)
        output = self.activatoin(output)
        return output

    def forward_step(self, input: 'torch.Tensor', padding: 'torch.Tensor', cache: 'List[torch.Tensor]') ->Tuple[torch.Tensor, List[torch.Tensor]]:
        """ forward step for inference
        Args:
            input (torch.Tensor): [batch_size, time_step=1]
            padding (torch.Tensor): [batch_size,1], 1 is padding value
            cache: for embedding predictor, cache[0] == history
        """
        assert input.size(1) == 1
        assert len(cache) == 1
        history = cache[0]
        assert history.size(1) == self.context_size - 1
        input = self.embed(input)
        input = self.embed_dropout(input)
        context_input = torch.cat((history, input), dim=1)
        input_expand = context_input.unsqueeze(1).unsqueeze(2)
        multi_head_pos = self.pos_embed.weight.view(self.num_heads, self.embed_size, self.context_size)
        multi_head_pos = multi_head_pos.permute(0, 2, 1)
        weight = input_expand * multi_head_pos
        weight = weight.sum(dim=-1, keepdim=False).unsqueeze(3)
        output = weight.matmul(input_expand).squeeze(dim=3)
        output = output.sum(dim=2)
        output = output / (self.num_heads * self.context_size)
        output = self.ffn(output)
        output = self.norm(output)
        output = self.activatoin(output)
        new_cache = context_input[:, 1:, :]
        return output, [new_cache]


class ConvPredictor(PredictorBase):

    def __init__(self, voca_size: 'int', embed_size: 'int', output_size: 'int', embed_dropout: 'float', history_size: 'int'=2, activation: 'str'='relu', bias: 'bool'=False, layer_norm_epsilon: 'float'=1e-05) ->None:
        super().__init__()
        assert embed_size == output_size
        assert history_size >= 0
        self.embed_size = embed_size
        self.context_size = history_size + 1
        self.embed = nn.Embedding(voca_size, self.embed_size)
        self.embed_dropout = nn.Dropout(p=embed_dropout)
        self.conv = nn.Conv1d(in_channels=embed_size, out_channels=embed_size, kernel_size=self.context_size, padding=0, groups=embed_size, bias=bias)
        self.norm = nn.LayerNorm(embed_size, eps=layer_norm_epsilon)
        self.activatoin = WENET_ACTIVATION_CLASSES[activation]()

    def output_size(self):
        return self.embed_size

    def init_state(self, batch_size: 'int', device: 'torch.device', method: 'str'='zero') ->List[torch.Tensor]:
        assert batch_size > 0
        assert method == 'zero'
        return [torch.zeros(batch_size, self.context_size - 1, self.embed_size, device=device)]

    def cache_to_batch(self, cache: 'List[List[torch.Tensor]]') ->List[torch.Tensor]:
        """
        Args:
            cache : [[history_1], [history_2], [history3]...]

        Returns:
            new_caceh: [history],
                history: [bs, ...]
        """
        history = torch.cat([h[0] for h in cache], dim=0)
        return [history]

    def batch_to_cache(self, cache: 'List[torch.Tensor]') ->List[List[torch.Tensor]]:
        """
        Args:
            cache : [history]
                history: [bs, ...]
        Returns:
            new_ache : [[history_1], [history_2], [history_3]...]
        """
        assert len(cache) == 1
        cache_0 = cache[0]
        history: 'List[List[torch.Tensor]]' = []
        for h in torch.split(cache_0, 1, dim=0):
            history.append([h])
        return history

    def forward(self, input: 'torch.Tensor', cache: 'Optional[List[torch.Tensor]]'=None):
        """ forward for training
        """
        input = self.embed(input)
        input = self.embed_dropout(input)
        if cache is None:
            zeros = self.init_state(input.size(0), device=input.device)[0]
        else:
            assert len(cache) == 1
            zeros = cache[0]
        input = torch.cat((zeros, input), dim=1)
        input = input.permute(0, 2, 1)
        out = self.conv(input).permute(0, 2, 1)
        out = self.activatoin(self.norm(out))
        return out

    def forward_step(self, input: 'torch.Tensor', padding: 'torch.Tensor', cache: 'List[torch.Tensor]') ->Tuple[torch.Tensor, List[torch.Tensor]]:
        """ forward step for inference
        Args:
            input (torch.Tensor): [batch_size, time_step=1]
            padding (torch.Tensor): [batch_size,1], 1 is padding value
            cache: for embedding predictor, cache[0] == history
        """
        assert input.size(1) == 1
        assert len(cache) == 1
        history = cache[0]
        assert history.size(1) == self.context_size - 1
        input = self.embed(input)
        input = self.embed_dropout(input)
        context_input = torch.cat((history, input), dim=1)
        input = context_input.permute(0, 2, 1)
        out = self.conv(input).permute(0, 2, 1)
        out = self.activatoin(self.norm(out))
        new_cache = context_input[:, 1:, :]
        return out, [new_cache]


class DecodeResult:

    def __init__(self, tokens: 'List[int]', score: 'float'=0.0, confidence: 'float'=0.0, tokens_confidence: 'List[float]'=None, times: 'List[int]'=None, nbest: 'List[List[int]]'=None, nbest_scores: 'List[float]'=None, nbest_times: 'List[List[int]]'=None):
        """
        Args:
            tokens: decode token list
            score: the total decode score of this result
            confidence: the total confidence of this result, it's in 0~1
            tokens_confidence: confidence of each token
            times: timestamp of each token, list of (start, end)
            nbest: nbest result
            nbest_scores: score of each nbest
            nbest_times:
        """
        self.tokens = tokens
        self.score = score
        self.confidence = confidence
        self.tokens_confidence = tokens_confidence
        self.times = times
        self.nbest = nbest
        self.nbest_scores = nbest_scores
        self.nbest_times = nbest_times


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


def pad_list(xs: 'List[torch.Tensor]', pad_value: 'int'):
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
    max_len = max([len(item) for item in xs])
    batchs = len(xs)
    ndim = xs[0].ndim
    if ndim == 1:
        pad_res = torch.zeros(batchs, max_len, dtype=xs[0].dtype, device=xs[0].device)
    elif ndim == 2:
        pad_res = torch.zeros(batchs, max_len, xs[0].shape[1], dtype=xs[0].dtype, device=xs[0].device)
    elif ndim == 3:
        pad_res = torch.zeros(batchs, max_len, xs[0].shape[1], xs[0].shape[2], dtype=xs[0].dtype, device=xs[0].device)
    else:
        raise ValueError(f'Unsupported ndim: {ndim}')
    pad_res.fill_(pad_value)
    for i in range(batchs):
        pad_res[i, :len(xs[i])] = xs[i]
    return pad_res


def add_sos_eos(ys_pad: 'torch.Tensor', sos: 'int', eos: 'int', ignore_id: 'int') ->Tuple[torch.Tensor, torch.Tensor]:
    """Add <sos> and <eos> labels.

    Args:
        ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)
        sos (int): index of <sos>
        eos (int): index of <eeos>
        ignore_id (int): index of padding

    Returns:
        ys_in (torch.Tensor) : (B, Lmax + 1)
        ys_out (torch.Tensor) : (B, Lmax + 1)

    Examples:
        >>> sos_id = 10
        >>> eos_id = 11
        >>> ignore_id = -1
        >>> ys_pad
        tensor([[ 1,  2,  3,  4,  5],
                [ 4,  5,  6, -1, -1],
                [ 7,  8,  9, -1, -1]], dtype=torch.int32)
        >>> ys_in,ys_out=add_sos_eos(ys_pad, sos_id , eos_id, ignore_id)
        >>> ys_in
        tensor([[10,  1,  2,  3,  4,  5],
                [10,  4,  5,  6, 11, 11],
                [10,  7,  8,  9, 11, 11]])
        >>> ys_out
        tensor([[ 1,  2,  3,  4,  5, 11],
                [ 4,  5,  6, 11, -1, -1],
                [ 7,  8,  9, 11, -1, -1]])
    """
    _sos = torch.tensor([sos], dtype=torch.long, requires_grad=False, device=ys_pad.device)
    _eos = torch.tensor([eos], dtype=torch.long, requires_grad=False, device=ys_pad.device)
    ys = [y[y != ignore_id] for y in ys_pad]
    ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)


def add_whisper_tokens(special_tokens, ys_pad: 'torch.Tensor', ignore_id: 'int', tasks: 'List[str]', no_timestamp: 'bool', langs: 'List[str]', use_prev: 'bool') ->Tuple[torch.Tensor, torch.Tensor]:
    """Add whisper-style tokens.

    ([PREV] -> [previous text tokens or hotwords]).optional --
      ┌------------------------------------------------------↲
      ↓
    [sot] -> [language id] -> [transcribe] -> [begin time] -> [text tokens] -> [end time] -> ... -> [eot]    # noqa
        |          |                |-------> [no timestamps] -> [text tokens] ----------------------↑       # noqa
        |          |                                                                                 |       # noqa
        |          |--------> [translate]  -> [begin time] -> [text tokens] -> [end time] -> ... --->|       # noqa
        |                           |-------> [no timestamps] -> [text tokens] --------------------->|       # noqa
        |                                                                                            |       # noqa
        |--> [no speech(VAD)] ---------------------------------------------------------------------->|       # noqa

    Args:
        special_tokens: get IDs of special tokens
        ignore_id (int): index of padding
        no_timestamp (bool): whether to add timestamps tokens
        tasks (List[str]): list of task tags
        langs (List[str]): list of language tags

    Returns:
        ys_in (torch.Tensor) : (B, Lmax + ?)
        ys_out (torch.Tensor) : (B, Lmax + ?)

    """
    assert len(langs) == ys_pad.size(0)
    assert len(tasks) == ys_pad.size(0)
    if use_prev:
        _prev = [special_tokens['sot_prev']]
        raise NotImplementedError
    else:
        _prev = []
    _sot = []
    for task, lang in zip(tasks, langs):
        if task == 'transcribe':
            task_id = special_tokens['transcribe']
        elif task == 'translate':
            task_id = special_tokens['translate']
        elif task == 'vad':
            task_id = special_tokens['no_speech']
        else:
            raise NotImplementedError('unsupported task {}'.format(task))
        language_id = special_tokens['sot'] + 1 + WHISPER_LANGS.index(lang)
        prefix = _prev + [special_tokens['sot'], language_id, task_id]
        if task == 'transcribe' or task == 'translate':
            if no_timestamp:
                prefix.append(special_tokens['no_timestamps'])
            else:
                prefix.append(special_tokens['timestamp_begin'])
                raise NotImplementedError
        elif task == 'vad':
            prefix.append(special_tokens['no_speech'])
        else:
            raise NotImplementedError
        prefix = torch.tensor(prefix, dtype=torch.long, requires_grad=False, device=ys_pad.device)
        _sot.append(prefix)
    _eot = torch.tensor([special_tokens['eot']], dtype=torch.long, requires_grad=False, device=ys_pad.device)
    ys = [y[y != ignore_id] for y in ys_pad]
    ys_in = [torch.cat([prefix, y], dim=0) for prefix, y in zip(_sot, ys)]
    ys_out = [torch.cat([prefix[1:], y, _eot], dim=0) for prefix, y in zip(_sot, ys)]
    return pad_list(ys_in, special_tokens['eot']), pad_list(ys_out, ignore_id)


def mask_finished_preds(pred: 'torch.Tensor', flag: 'torch.Tensor', eos: 'int') ->torch.Tensor:
    """
    If a sequence is finished, all of its branch should be <eos>

    Args:
        pred (torch.Tensor): A int array with shape
            (batch_size * beam_size, beam_size).
        flag (torch.Tensor): A bool array with shape
            (batch_size * beam_size, 1).

    Returns:
        torch.Tensor: (batch_size * beam_size).
    """
    beam_size = pred.size(-1)
    finished = flag.repeat([1, beam_size])
    return pred.masked_fill_(finished, eos)


def mask_finished_scores(score: 'torch.Tensor', flag: 'torch.Tensor') ->torch.Tensor:
    """
    If a sequence is finished, we only allow one alive branch. This function
    aims to give one branch a zero score and the rest -inf score.

    Args:
        score (torch.Tensor): A real value array with shape
            (batch_size * beam_size, beam_size).
        flag (torch.Tensor): A bool array with shape
            (batch_size * beam_size, 1).

    Returns:
        torch.Tensor: (batch_size * beam_size, beam_size).
    """
    beam_size = score.size(-1)
    zero_mask = torch.zeros_like(flag, dtype=torch.bool)
    if beam_size > 1:
        unfinished = torch.cat((zero_mask, flag.repeat([1, beam_size - 1])), dim=1)
        finished = torch.cat((flag, zero_mask.repeat([1, beam_size - 1])), dim=1)
    else:
        unfinished = zero_mask
        finished = flag
    score.masked_fill_(unfinished, -float('inf'))
    score.masked_fill_(finished, 0)
    return score


def attention_beam_search(model, encoder_out: 'torch.Tensor', encoder_mask: 'torch.Tensor', beam_size: 'int'=10, length_penalty: 'float'=0.0, infos: 'Dict[str, List[str]]'=None) ->List[DecodeResult]:
    device = encoder_out.device
    batch_size = encoder_out.shape[0]
    maxlen = encoder_out.size(1)
    encoder_dim = encoder_out.size(2)
    running_size = batch_size * beam_size
    if getattr(model, 'special_tokens', None) is not None and 'transcribe' in model.special_tokens:
        tasks, langs = infos['tasks'], infos['langs']
        tasks = [t for t in tasks for _ in range(beam_size)]
        langs = [l for l in langs for _ in range(beam_size)]
        hyps = torch.ones([running_size, 0], dtype=torch.long, device=device)
        hyps, _ = add_whisper_tokens(model.special_tokens, hyps, model.ignore_id, tasks=tasks, no_timestamp=True, langs=langs, use_prev=False)
    else:
        hyps = torch.ones([running_size, 1], dtype=torch.long, device=device).fill_(model.sos)
    prefix_len = hyps.size(1)
    scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1), dtype=torch.float)
    scores = scores.to(device).repeat([batch_size]).unsqueeze(1)
    end_flag = torch.zeros_like(scores, dtype=torch.bool, device=device)
    cache = {'self_att_cache': {}, 'cross_att_cache': {}}
    if model.decoder.use_sdpa:
        encoder_mask = mask_to_bias(encoder_mask, encoder_out.dtype)
    if hasattr(model, 'decode_maxlen'):
        maxlen = model.decode_maxlen
    for i in range(prefix_len, maxlen + 1):
        if end_flag.sum() == running_size:
            break
        hyps_mask = subsequent_mask(i).unsqueeze(0).repeat(running_size, 1, 1)
        if model.decoder.use_sdpa:
            hyps_mask = mask_to_bias(hyps_mask, encoder_out.dtype)
        logp = model.decoder.forward_one_step(encoder_out, encoder_mask, hyps, hyps_mask, cache)
        top_k_logp, top_k_index = logp.topk(beam_size)
        top_k_logp = mask_finished_scores(top_k_logp, end_flag)
        top_k_index = mask_finished_preds(top_k_index, end_flag, model.eos)
        scores = scores + top_k_logp
        scores = scores.view(batch_size, beam_size * beam_size)
        scores, offset_k_index = scores.topk(k=beam_size)
        cache_index = (offset_k_index // beam_size).view(-1)
        base_cache_index = (torch.arange(batch_size, device=device).view(-1, 1).repeat([1, beam_size]) * beam_size).view(-1)
        cache_index = base_cache_index + cache_index
        cache['self_att_cache'] = {i_layer: (torch.index_select(value[0], dim=0, index=cache_index), torch.index_select(value[1], dim=0, index=cache_index)) for i_layer, value in cache['self_att_cache'].items()}
        torch.cuda.empty_cache()
        scores = scores.view(-1, 1)
        base_k_index = torch.arange(batch_size, device=device).view(-1, 1).repeat([1, beam_size])
        base_k_index = base_k_index * beam_size * beam_size
        best_k_index = base_k_index.view(-1) + offset_k_index.view(-1)
        best_k_pred = torch.index_select(top_k_index.view(-1), dim=-1, index=best_k_index)
        best_hyps_index = best_k_index // beam_size
        last_best_k_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
        hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)), dim=1)
        end_flag = torch.eq(hyps[:, -1], model.eos).view(-1, 1)
    scores = scores.view(batch_size, beam_size)
    lengths = hyps.ne(model.eos).sum(dim=1).view(batch_size, beam_size).float()
    scores = scores / lengths.pow(length_penalty)
    best_scores, best_index = scores.max(dim=-1)
    best_hyps_index = best_index + torch.arange(batch_size, dtype=torch.long, device=device) * beam_size
    best_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
    best_hyps = best_hyps[:, prefix_len:]
    results = []
    for i in range(batch_size):
        hyp = best_hyps[i]
        hyp = hyp[hyp != model.eos]
        results.append(DecodeResult(hyp.tolist()))
    return results


def attention_rescoring(model, ctc_prefix_results: 'List[DecodeResult]', encoder_outs: 'torch.Tensor', encoder_lens: 'torch.Tensor', ctc_weight: 'float'=0.0, reverse_weight: 'float'=0.0, infos: 'Dict[str, List[str]]'=None) ->List[DecodeResult]:
    """
        Args:
            ctc_prefix_results(List[DecodeResult]): ctc prefix beam search results
    """
    sos, eos = model.sos_symbol(), model.eos_symbol()
    device = encoder_outs.device
    assert encoder_outs.shape[0] == len(ctc_prefix_results)
    batch_size = encoder_outs.shape[0]
    results = []
    for b in range(batch_size):
        encoder_out = encoder_outs[b, :encoder_lens[b], :].unsqueeze(0)
        hyps = ctc_prefix_results[b].nbest
        ctc_scores = ctc_prefix_results[b].nbest_scores
        hyps_pad = pad_sequence([torch.tensor(hyp, device=device, dtype=torch.long) for hyp in hyps], True, model.ignore_id)
        hyps_lens = torch.tensor([len(hyp) for hyp in hyps], device=device, dtype=torch.long)
        if getattr(model, 'special_tokens', None) is not None and 'transcribe' in model.special_tokens:
            prev_len = hyps_pad.size(1)
            hyps_pad, _ = add_whisper_tokens(model.special_tokens, hyps_pad, model.ignore_id, tasks=[infos['tasks'][b]] * len(hyps), no_timestamp=True, langs=[infos['langs'][b]] * len(hyps), use_prev=False)
            cur_len = hyps_pad.size(1)
            hyps_lens = hyps_lens + cur_len - prev_len
            prefix_len = 4
        else:
            hyps_pad, _ = add_sos_eos(hyps_pad, sos, eos, model.ignore_id)
            hyps_lens = hyps_lens + 1
            prefix_len = 1
        decoder_out, r_decoder_out = model.forward_attention_decoder(hyps_pad, hyps_lens, encoder_out, reverse_weight)
        best_score = -float('inf')
        best_index = 0
        confidences = []
        tokens_confidences = []
        for i, hyp in enumerate(hyps):
            score = 0.0
            tc = []
            for j, w in enumerate(hyp):
                s = decoder_out[i][j + (prefix_len - 1)][w]
                score += s
                tc.append(math.exp(s))
            score += decoder_out[i][len(hyp) + (prefix_len - 1)][eos]
            if reverse_weight > 0 and r_decoder_out.dim() > 0:
                r_score = 0.0
                for j, w in enumerate(hyp):
                    s = r_decoder_out[i][len(hyp) - j - 1 + (prefix_len - 1)][w]
                    r_score += s
                    tc[j] = (tc[j] + math.exp(s)) / 2
                r_score += r_decoder_out[i][len(hyp) + (prefix_len - 1)][eos]
                score = score * (1 - reverse_weight) + r_score * reverse_weight
            confidences.append(math.exp(score / (len(hyp) + 1)))
            score += ctc_scores[i] * ctc_weight
            if score > best_score:
                best_score = score.item()
                best_index = i
            tokens_confidences.append(tc)
        results.append(DecodeResult(hyps[best_index], best_score, confidence=confidences[best_index], times=ctc_prefix_results[b].nbest_times[best_index], tokens_confidence=tokens_confidences[best_index]))
    return results


def remove_duplicates_and_blank(hyp: 'List[int]', blank_id: 'int'=0) ->List[int]:
    new_hyp: 'List[int]' = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank_id:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp


def ctc_greedy_search(ctc_probs: 'torch.Tensor', ctc_lens: 'torch.Tensor', blank_id: 'int'=0) ->List[DecodeResult]:
    batch_size = ctc_probs.shape[0]
    maxlen = ctc_probs.size(1)
    topk_prob, topk_index = ctc_probs.topk(1, dim=2)
    topk_index = topk_index.view(batch_size, maxlen)
    mask = make_pad_mask(ctc_lens, maxlen)
    topk_index = topk_index.masked_fill_(mask, blank_id)
    hyps = [hyp.tolist() for hyp in topk_index]
    scores = topk_prob.max(1)
    results = []
    for hyp in hyps:
        r = DecodeResult(remove_duplicates_and_blank(hyp, blank_id))
        results.append(r)
    return results


def log_add(*args) ->float:
    """
    Stable log add
    """
    if all(a == -float('inf') for a in args):
        return -float('inf')
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp


class PrefixScore:
    """ For CTC prefix beam search """

    def __init__(self, s: 'float'=float('-inf'), ns: 'float'=float('-inf'), v_s: 'float'=float('-inf'), v_ns: 'float'=float('-inf'), context_state: 'ContextState'=None, context_score: 'float'=0.0):
        self.s = s
        self.ns = ns
        self.v_s = v_s
        self.v_ns = v_ns
        self.cur_token_prob = float('-inf')
        self.times_s = []
        self.times_ns = []
        self.context_state = context_state
        self.context_score = context_score
        self.has_context = False

    def score(self):
        return log_add(self.s, self.ns)

    def viterbi_score(self):
        return self.v_s if self.v_s > self.v_ns else self.v_ns

    def times(self):
        return self.times_s if self.v_s > self.v_ns else self.times_ns

    def total_score(self):
        return self.score() + self.context_score

    def copy_context(self, prefix_score):
        self.context_score = prefix_score.context_score
        self.context_state = prefix_score.context_state

    def update_context(self, context_graph, prefix_score, word_id):
        self.copy_context(prefix_score)
        score, context_state = context_graph.forward_one_step(prefix_score.context_state, word_id)
        self.context_score += score
        self.context_state = context_state


def ctc_prefix_beam_search(ctc_probs: 'torch.Tensor', ctc_lens: 'torch.Tensor', beam_size: 'int', context_graph: 'ContextGraph'=None, blank_id: 'int'=0) ->List[DecodeResult]:
    """
        Returns:
            List[List[List[int]]]: nbest result for each utterance
    """
    batch_size = ctc_probs.shape[0]
    results = []
    for i in range(batch_size):
        ctc_prob = ctc_probs[i]
        num_t = ctc_lens[i]
        cur_hyps = [(tuple(), PrefixScore(s=0.0, ns=-float('inf'), v_s=0.0, v_ns=0.0, context_state=None if context_graph is None else context_graph.root, context_score=0.0))]
        for t in range(0, num_t):
            logp = ctc_prob[t]
            next_hyps = defaultdict(lambda : PrefixScore())
            top_k_logp, top_k_index = logp.topk(beam_size)
            for u in top_k_index:
                u = u.item()
                prob = logp[u].item()
                for prefix, prefix_score in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if u == blank_id:
                        next_score = next_hyps[prefix]
                        next_score.s = log_add(next_score.s, prefix_score.score() + prob)
                        next_score.v_s = prefix_score.viterbi_score() + prob
                        next_score.times_s = prefix_score.times().copy()
                        if context_graph and not next_score.has_context:
                            next_score.copy_context(prefix_score)
                            next_score.has_context = True
                    elif u == last:
                        next_score1 = next_hyps[prefix]
                        next_score1.ns = log_add(next_score1.ns, prefix_score.ns + prob)
                        if next_score1.v_ns < prefix_score.v_ns + prob:
                            next_score1.v_ns = prefix_score.v_ns + prob
                            if next_score1.cur_token_prob < prob:
                                next_score1.cur_token_prob = prob
                                next_score1.times_ns = prefix_score.times_ns.copy()
                                next_score1.times_ns[-1] = t
                        if context_graph and not next_score1.has_context:
                            next_score1.copy_context(prefix_score)
                            next_score1.has_context = True
                        n_prefix = prefix + (u,)
                        next_score2 = next_hyps[n_prefix]
                        next_score2.ns = log_add(next_score2.ns, prefix_score.s + prob)
                        if next_score2.v_ns < prefix_score.v_s + prob:
                            next_score2.v_ns = prefix_score.v_s + prob
                            next_score2.cur_token_prob = prob
                            next_score2.times_ns = prefix_score.times_s.copy()
                            next_score2.times_ns.append(t)
                        if context_graph and not next_score2.has_context:
                            next_score2.update_context(context_graph, prefix_score, u)
                            next_score2.has_context = True
                    else:
                        n_prefix = prefix + (u,)
                        next_score = next_hyps[n_prefix]
                        next_score.ns = log_add(next_score.ns, prefix_score.score() + prob)
                        if next_score.v_ns < prefix_score.viterbi_score() + prob:
                            next_score.v_ns = prefix_score.viterbi_score() + prob
                            next_score.cur_token_prob = prob
                            next_score.times_ns = prefix_score.times().copy()
                            next_score.times_ns.append(t)
                        if context_graph and not next_score.has_context:
                            next_score.update_context(context_graph, prefix_score, u)
                            next_score.has_context = True
            next_hyps = sorted(next_hyps.items(), key=lambda x: x[1].total_score(), reverse=True)
            cur_hyps = next_hyps[:beam_size]
        if context_graph is not None:
            for i, hyp in enumerate(cur_hyps):
                context_score, new_context_state = context_graph.finalize(hyp[1].context_state)
                cur_hyps[i][1].context_score = context_score
                cur_hyps[i][1].context_state = new_context_state
        nbest = [y[0] for y in cur_hyps]
        nbest_scores = [y[1].total_score() for y in cur_hyps]
        nbest_times = [y[1].times() for y in cur_hyps]
        best = nbest[0]
        best_score = nbest_scores[0]
        best_time = nbest_times[0]
        results.append(DecodeResult(tokens=best, score=best_score, times=best_time, nbest=nbest, nbest_scores=nbest_scores, nbest_times=nbest_times))
    return results


def reverse_pad_list(ys_pad: 'torch.Tensor', ys_lens: 'torch.Tensor', pad_value: 'float'=-1.0) ->torch.Tensor:
    """Reverse padding for the list of tensors.

    Args:
        ys_pad (tensor): The padded tensor (B, Tokenmax).
        ys_lens (tensor): The lens of token seqs (B)
        pad_value (int): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tokenmax).

    Examples:
        >>> x
        tensor([[1, 2, 3, 4], [5, 6, 7, 0], [8, 9, 0, 0]])
        >>> pad_list(x, 0)
        tensor([[4, 3, 2, 1],
                [7, 6, 5, 0],
                [9, 8, 0, 0]])

    """
    r_ys_pad = pad_sequence([torch.flip(y.int()[:i], [0]) for y, i in zip(ys_pad, ys_lens)], True, pad_value)
    return r_ys_pad


class ASRModel(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(self, vocab_size: 'int', encoder: 'BaseEncoder', decoder: 'TransformerDecoder', ctc: 'CTC', ctc_weight: 'float'=0.5, ignore_id: 'int'=IGNORE_ID, reverse_weight: 'float'=0.0, lsm_weight: 'float'=0.0, length_normalized_loss: 'bool'=False, special_tokens: 'Optional[dict]'=None, apply_non_blank_embedding: 'bool'=False):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        super().__init__()
        self.sos = vocab_size - 1 if special_tokens is None else special_tokens.get('<sos>', vocab_size - 1)
        self.eos = vocab_size - 1 if special_tokens is None else special_tokens.get('<eos>', vocab_size - 1)
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight
        self.apply_non_blank_embedding = apply_non_blank_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.ctc = ctc
        self.criterion_att = LabelSmoothingLoss(size=vocab_size, padding_idx=ignore_id, smoothing=lsm_weight, normalize_length=length_normalized_loss)

    @torch.jit.unused
    def forward(self, batch: 'dict', device: 'torch.device') ->Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss"""
        speech = batch['feats']
        speech_lengths = batch['feats_lengths']
        text = batch['target']
        text_lengths = batch['target_lengths']
        assert text_lengths.dim() == 1, text_lengths.shape
        assert speech.shape[0] == speech_lengths.shape[0] == text.shape[0] == text_lengths.shape[0], (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        if self.ctc_weight != 0.0:
            loss_ctc, ctc_probs = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)
        else:
            loss_ctc, ctc_probs = None, None
        if self.apply_non_blank_embedding:
            assert self.ctc_weight != 0
            assert ctc_probs is not None
            encoder_out, encoder_mask = self.filter_blank_embedding(ctc_probs, encoder_out)
        if self.ctc_weight != 1.0:
            loss_att, acc_att = self._calc_att_loss(encoder_out, encoder_mask, text, text_lengths, {'langs': batch['langs'], 'tasks': batch['tasks']})
        else:
            loss_att = None
            acc_att = None
        if loss_ctc is None:
            loss = loss_att
        elif loss_att is None:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
        return {'loss': loss, 'loss_att': loss_att, 'loss_ctc': loss_ctc, 'th_accuracy': acc_att}

    def tie_or_clone_weights(self, jit_mode: 'bool'=True):
        self.decoder.tie_or_clone_weights(jit_mode)

    @torch.jit.unused
    def _forward_ctc(self, encoder_out: 'torch.Tensor', encoder_mask: 'torch.Tensor', text: 'torch.Tensor', text_lengths: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        loss_ctc, ctc_probs = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)
        return loss_ctc, ctc_probs

    def filter_blank_embedding(self, ctc_probs: 'torch.Tensor', encoder_out: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        batch_size = encoder_out.size(0)
        maxlen = encoder_out.size(1)
        top1_index = torch.argmax(ctc_probs, dim=2)
        indices = []
        for j in range(batch_size):
            indices.append(torch.tensor([i for i in range(maxlen) if top1_index[j][i] != 0]))
        select_encoder_out = [torch.index_select(encoder_out[i, :, :], 0, indices[i]) for i in range(batch_size)]
        select_encoder_out = pad_sequence(select_encoder_out, batch_first=True, padding_value=0)
        xs_lens = torch.tensor([len(indices[i]) for i in range(batch_size)])
        T = select_encoder_out.size(1)
        encoder_mask = ~make_pad_mask(xs_lens, T).unsqueeze(1)
        encoder_out = select_encoder_out
        return encoder_out, encoder_mask

    def _calc_att_loss(self, encoder_out: 'torch.Tensor', encoder_mask: 'torch.Tensor', ys_pad: 'torch.Tensor', ys_pad_lens: 'torch.Tensor', infos: 'Dict[str, List[str]]'=None) ->Tuple[torch.Tensor, torch.Tensor]:
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1
        r_ys_pad = reverse_pad_list(ys_pad, ys_pad_lens, float(self.ignore_id))
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, self.sos, self.eos, self.ignore_id)
        decoder_out, r_decoder_out, _ = self.decoder(encoder_out, encoder_mask, ys_in_pad, ys_in_lens, r_ys_in_pad, self.reverse_weight)
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        r_loss_att = torch.tensor(0.0)
        if self.reverse_weight > 0.0:
            r_loss_att = self.criterion_att(r_decoder_out, r_ys_out_pad)
        loss_att = loss_att * (1 - self.reverse_weight) + r_loss_att * self.reverse_weight
        acc_att = th_accuracy(decoder_out.view(-1, self.vocab_size), ys_out_pad, ignore_label=self.ignore_id)
        return loss_att, acc_att

    def _forward_encoder(self, speech: 'torch.Tensor', speech_lengths: 'torch.Tensor', decoding_chunk_size: 'int'=-1, num_decoding_left_chunks: 'int'=-1, simulate_streaming: 'bool'=False) ->Tuple[torch.Tensor, torch.Tensor]:
        if simulate_streaming and decoding_chunk_size > 0:
            encoder_out, encoder_mask = self.encoder.forward_chunk_by_chunk(speech, decoding_chunk_size=decoding_chunk_size, num_decoding_left_chunks=num_decoding_left_chunks)
        else:
            encoder_out, encoder_mask = self.encoder(speech, speech_lengths, decoding_chunk_size=decoding_chunk_size, num_decoding_left_chunks=num_decoding_left_chunks)
        return encoder_out, encoder_mask

    @torch.jit.unused
    def ctc_logprobs(self, encoder_out: 'torch.Tensor', blank_penalty: 'float'=0.0, blank_id: 'int'=0):
        if blank_penalty > 0.0:
            logits = self.ctc.ctc_lo(encoder_out)
            logits[:, :, blank_id] -= blank_penalty
            ctc_probs = logits.log_softmax(dim=2)
        else:
            ctc_probs = self.ctc.log_softmax(encoder_out)
        return ctc_probs

    def decode(self, methods: 'List[str]', speech: 'torch.Tensor', speech_lengths: 'torch.Tensor', beam_size: 'int', decoding_chunk_size: 'int'=-1, num_decoding_left_chunks: 'int'=-1, ctc_weight: 'float'=0.0, simulate_streaming: 'bool'=False, reverse_weight: 'float'=0.0, context_graph: 'ContextGraph'=None, blank_id: 'int'=0, blank_penalty: 'float'=0.0, length_penalty: 'float'=0.0, infos: 'Dict[str, List[str]]'=None) ->Dict[str, List[DecodeResult]]:
        """ Decode input speech

        Args:
            methods:(List[str]): list of decoding methods to use, which could
                could contain the following decoding methods, please refer paper:
                https://arxiv.org/pdf/2102.01547.pdf
                   * ctc_greedy_search
                   * ctc_prefix_beam_search
                   * atttention
                   * attention_rescoring
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
            reverse_weight (float): right to left decoder weight
            ctc_weight (float): ctc score weight

        Returns: dict results of all decoding methods
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        encoder_out, encoder_mask = self._forward_encoder(speech, speech_lengths, decoding_chunk_size, num_decoding_left_chunks, simulate_streaming)
        encoder_lens = encoder_mask.squeeze(1).sum(1)
        ctc_probs = self.ctc_logprobs(encoder_out, blank_penalty, blank_id)
        results = {}
        if 'attention' in methods:
            results['attention'] = attention_beam_search(self, encoder_out, encoder_mask, beam_size, length_penalty, infos)
        if 'ctc_greedy_search' in methods:
            results['ctc_greedy_search'] = ctc_greedy_search(ctc_probs, encoder_lens, blank_id)
        if 'ctc_prefix_beam_search' in methods:
            ctc_prefix_result = ctc_prefix_beam_search(ctc_probs, encoder_lens, beam_size, context_graph, blank_id)
            results['ctc_prefix_beam_search'] = ctc_prefix_result
        if 'attention_rescoring' in methods:
            if 'ctc_prefix_beam_search' in results:
                ctc_prefix_result = results['ctc_prefix_beam_search']
            else:
                ctc_prefix_result = ctc_prefix_beam_search(ctc_probs, encoder_lens, beam_size, context_graph, blank_id)
            if self.apply_non_blank_embedding:
                encoder_out, _ = self.filter_blank_embedding(ctc_probs, encoder_out)
            results['attention_rescoring'] = attention_rescoring(self, ctc_prefix_result, encoder_out, encoder_lens, ctc_weight, reverse_weight, infos)
        return results

    @torch.jit.export
    def subsampling_rate(self) ->int:
        """ Export interface for c++ call, return subsampling_rate of the
            model
        """
        return self.encoder.embed.subsampling_rate

    @torch.jit.export
    def right_context(self) ->int:
        """ Export interface for c++ call, return right_context of the model
        """
        return self.encoder.embed.right_context

    @torch.jit.export
    def sos_symbol(self) ->int:
        """ Export interface for c++ call, return sos symbol id of the model
        """
        return self.sos

    @torch.jit.export
    def eos_symbol(self) ->int:
        """ Export interface for c++ call, return eos symbol id of the model
        """
        return self.eos

    @torch.jit.export
    def forward_encoder_chunk(self, xs: 'torch.Tensor', offset: 'int', required_cache_size: 'int', att_cache: 'torch.Tensor'=torch.zeros(0, 0, 0, 0), cnn_cache: 'torch.Tensor'=torch.zeros(0, 0, 0, 0)) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Export interface for c++ call, give input chunk xs, and return
            output from time 0 to current chunk.

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
        return self.encoder.forward_chunk(xs, offset, required_cache_size, att_cache, cnn_cache)

    @torch.jit.export
    def ctc_activation(self, xs: 'torch.Tensor') ->torch.Tensor:
        """ Export interface for c++ call, apply linear transform and log
            softmax before ctc
        Args:
            xs (torch.Tensor): encoder output

        Returns:
            torch.Tensor: activation before ctc

        """
        return self.ctc.log_softmax(xs)

    @torch.jit.export
    def is_bidirectional_decoder(self) ->bool:
        """
        Returns:
            torch.Tensor: decoder output
        """
        if hasattr(self.decoder, 'right_decoder'):
            return True
        else:
            return False

    @torch.jit.export
    def forward_attention_decoder(self, hyps: 'torch.Tensor', hyps_lens: 'torch.Tensor', encoder_out: 'torch.Tensor', reverse_weight: 'float'=0) ->Tuple[torch.Tensor, torch.Tensor]:
        """ Export interface for c++ call, forward decoder with multiple
            hypothesis from ctc prefix beam search and one encoder output
        Args:
            hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad sos at the begining
            hyps_lens (torch.Tensor): length of each hyp in hyps
            encoder_out (torch.Tensor): corresponding encoder output
            r_hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad eos at the begining which is used fo right to left decoder
            reverse_weight: used for verfing whether used right to left decoder,
            > 0 will use.

        Returns:
            torch.Tensor: decoder output
        """
        assert encoder_out.size(0) == 1
        num_hyps = hyps.size(0)
        assert hyps_lens.size(0) == num_hyps
        encoder_out = encoder_out.repeat(num_hyps, 1, 1)
        encoder_mask = torch.ones(num_hyps, 1, encoder_out.size(1), dtype=torch.bool, device=encoder_out.device)
        r_hyps_lens = hyps_lens - 1
        r_hyps = hyps[:, 1:]
        max_len = torch.max(r_hyps_lens)
        index_range = torch.arange(0, max_len, 1)
        seq_len_expand = r_hyps_lens.unsqueeze(1)
        seq_mask = seq_len_expand > index_range
        index = seq_len_expand - 1 - index_range
        index = index * seq_mask
        r_hyps = torch.gather(r_hyps, 1, index)
        r_hyps = torch.where(seq_mask, r_hyps, self.eos)
        r_hyps = torch.cat([hyps[:, 0:1], r_hyps], dim=1)
        decoder_out, r_decoder_out, _ = self.decoder(encoder_out, encoder_mask, hyps, hyps_lens, r_hyps, reverse_weight)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        return decoder_out, r_decoder_out


class GlobalCMVN(torch.nn.Module):

    def __init__(self, mean: 'torch.Tensor', istd: 'torch.Tensor', norm_var: 'bool'=True):
        """
        Args:
            mean (torch.Tensor): mean stats
            istd (torch.Tensor): inverse std, std which is 1.0 / std
        """
        super().__init__()
        assert mean.shape == istd.shape
        self.norm_var = norm_var
        self.register_buffer('mean', mean)
        self.register_buffer('istd', istd)

    def forward(self, x: 'torch.Tensor'):
        """
        Args:
            x (torch.Tensor): (batch, max_len, feat_dim)

        Returns:
            (torch.Tensor): normalized feature
        """
        x = x - self.mean
        if self.norm_var:
            x = x * self.istd
        return x


class CTC(torch.nn.Module):
    """CTC module"""

    def __init__(self, odim: 'int', encoder_output_size: 'int', dropout_rate: 'float'=0.0, reduce: 'bool'=True, blank_id: 'int'=0):
        """ Construct CTC module
        Args:
            odim: dimension of outputs
            encoder_output_size: number of encoder projection units
            dropout_rate: dropout rate (0.0 ~ 1.0)
            reduce: reduce the CTC loss into a scalar
            blank_id: blank label.
        """
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.ctc_lo = torch.nn.Linear(eprojs, odim)
        reduction_type = 'sum' if reduce else 'none'
        self.ctc_loss = torch.nn.CTCLoss(blank=blank_id, reduction=reduction_type, zero_infinity=True)

    def forward(self, hs_pad: 'torch.Tensor', hlens: 'torch.Tensor', ys_pad: 'torch.Tensor', ys_lens: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
        ys_hat = ys_hat.transpose(0, 1)
        ys_hat = ys_hat.log_softmax(2)
        loss = self.ctc_loss(ys_hat, ys_pad, hlens, ys_lens)
        loss = loss / ys_hat.size(1)
        ys_hat = ys_hat.transpose(0, 1)
        return loss, ys_hat

    def log_softmax(self, hs_pad: 'torch.Tensor') ->torch.Tensor:
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad: 'torch.Tensor') ->torch.Tensor:
        """argmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)


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

    def __init__(self, size: 'int', self_attn: 'nn.Module', src_attn: 'Optional[nn.Module]', feed_forward: 'nn.Module', dropout_rate: 'float', normalize_before: 'bool'=True, layer_norm_type: 'str'='layer_norm', norm_eps: 'float'=1e-05):
        """Construct an DecoderLayer object."""
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        assert layer_norm_type in ['layer_norm', 'rms_norm']
        self.norm1 = WENET_NORM_CLASSES[layer_norm_type](size, eps=norm_eps)
        self.norm2 = WENET_NORM_CLASSES[layer_norm_type](size, eps=norm_eps)
        self.norm3 = WENET_NORM_CLASSES[layer_norm_type](size, eps=norm_eps)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before

    def forward(self, tgt: 'torch.Tensor', tgt_mask: 'torch.Tensor', memory: 'torch.Tensor', memory_mask: 'torch.Tensor', cache: 'Optional[Dict[str, Optional[T_CACHE]]]'=None) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        if cache is not None:
            att_cache = cache['self_att_cache']
            cross_att_cache = cache['cross_att_cache']
        else:
            att_cache, cross_att_cache = None, None
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        if att_cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
            att_cache = torch.empty(0, 0, 0, 0), torch.empty(0, 0, 0, 0)
        else:
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = tgt_mask[:, -1:, :]
        x, new_att_cache = self.self_attn(tgt_q, tgt_q, tgt_q, tgt_q_mask, cache=att_cache)
        if cache is not None:
            cache['self_att_cache'] = new_att_cache
        x = residual + self.dropout(x)
        if not self.normalize_before:
            x = self.norm1(x)
        if self.src_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.norm2(x)
            if cross_att_cache is None:
                cross_att_cache = torch.empty(0, 0, 0, 0), torch.empty(0, 0, 0, 0)
            x, new_cross_cache = self.src_attn(x, memory, memory, memory_mask, cache=cross_att_cache)
            if cache is not None:
                cache['cross_att_cache'] = new_cross_cache
            x = residual + self.dropout(x)
            if not self.normalize_before:
                x = self.norm2(x)
        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)
        return x, tgt_mask, memory, memory_mask


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
        query_bias: whether use bias in attention.linear_q
        key_bias: whether use bias in attention.linear_k, False for whisper models.
        value_bias: whether use bias in attention.linear_v
        gradient_checkpointing: rerunning a forward-pass segment for each
            checkpointed segment during backward.
        tie_word_embedding: Tie or clone module weights depending of whether we are
            using TorchScript or not
    """

    def __init__(self, vocab_size: 'int', encoder_output_size: 'int', attention_heads: 'int'=4, linear_units: 'int'=2048, num_blocks: 'int'=6, dropout_rate: 'float'=0.1, positional_dropout_rate: 'float'=0.1, self_attention_dropout_rate: 'float'=0.0, src_attention_dropout_rate: 'float'=0.0, input_layer: 'str'='embed', use_output_layer: 'bool'=True, normalize_before: 'bool'=True, src_attention: 'bool'=True, query_bias: 'bool'=True, key_bias: 'bool'=True, value_bias: 'bool'=True, activation_type: 'str'='relu', gradient_checkpointing: 'bool'=False, tie_word_embedding: 'bool'=False, use_sdpa: 'bool'=False, layer_norm_type: 'str'='layer_norm', norm_eps: 'float'=1e-05, n_kv_head: 'Optional[int]'=None, head_dim: 'Optional[int]'=None, mlp_type: 'str'='position_wise_feed_forward', mlp_bias: 'bool'=True, n_expert: 'int'=8, n_expert_activated: 'int'=2):
        super().__init__()
        attention_dim = encoder_output_size
        activation = WENET_ACTIVATION_CLASSES[activation_type]()
        self.embed = torch.nn.Sequential(torch.nn.Identity() if input_layer == 'no_pos' else torch.nn.Embedding(vocab_size, attention_dim), WENET_EMB_CLASSES[input_layer](attention_dim, positional_dropout_rate))
        assert layer_norm_type in ['layer_norm', 'rms_norm']
        self.normalize_before = normalize_before
        self.after_norm = WENET_NORM_CLASSES[layer_norm_type](attention_dim, eps=norm_eps)
        self.use_output_layer = use_output_layer
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        else:
            self.output_layer = torch.nn.Identity()
        self.num_blocks = num_blocks
        mlp_class = WENET_MLP_CLASSES[mlp_type]
        self.decoders = torch.nn.ModuleList([DecoderLayer(attention_dim, WENET_ATTENTION_CLASSES['selfattn'](attention_heads, attention_dim, self_attention_dropout_rate, query_bias, key_bias, value_bias, use_sdpa, n_kv_head, head_dim), WENET_ATTENTION_CLASSES['crossattn'](attention_heads, attention_dim, src_attention_dropout_rate, query_bias, key_bias, value_bias, use_sdpa, n_kv_head, head_dim) if src_attention else None, mlp_class(attention_dim, linear_units, dropout_rate, activation, mlp_bias, n_expert=n_expert, n_expert_activated=n_expert_activated), dropout_rate, normalize_before, layer_norm_type, norm_eps) for _ in range(self.num_blocks)])
        self.gradient_checkpointing = gradient_checkpointing
        self.tie_word_embedding = tie_word_embedding
        self.use_sdpa = use_sdpa

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
        if self.use_sdpa:
            tgt_mask = mask_to_bias(tgt_mask, memory.dtype)
            memory_mask = mask_to_bias(memory_mask, memory.dtype)
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
            x, tgt_mask, memory, memory_mask = ckpt.checkpoint(layer.__call__, x, tgt_mask, memory, memory_mask, use_reentrant=False)
        return x

    def forward_one_step(self, memory: 'torch.Tensor', memory_mask: 'torch.Tensor', tgt: 'torch.Tensor', tgt_mask: 'torch.Tensor', cache: 'Dict[str, Dict[str, T_CACHE]]') ->torch.Tensor:
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
        update_cross_att_cache = True
        if len(cache['cross_att_cache']) != 0:
            assert len(cache['cross_att_cache']) == self.num_blocks
            update_cross_att_cache = False
        for i, decoder in enumerate(self.decoders):
            layer_i = 'layer_{}'.format(i)
            self_att_cache = cache['self_att_cache'].get(layer_i, None)
            cross_att_cache = cache['cross_att_cache'].get(layer_i, None)
            c = {'self_att_cache': self_att_cache, 'cross_att_cache': cross_att_cache}
            x, tgt_mask, memory, memory_mask = decoder(x, tgt_mask, memory, memory_mask, cache=c)
            assert c['self_att_cache'] is not None
            assert c['cross_att_cache'] is not None
            cache['self_att_cache'][layer_i] = c['self_att_cache']
            if update_cross_att_cache:
                cache['cross_att_cache'][layer_i] = c['cross_att_cache']
        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.use_output_layer:
            y = torch.log_softmax(self.output_layer(y), dim=-1)
        return y

    def tie_or_clone_weights(self, jit_mode: 'bool'=True):
        """Tie or clone module weights (between word_emb and output_layer)
            depending of whether we are using TorchScript or not"""
        rank = int(os.environ.get('RANK', 0))
        if not self.use_output_layer:
            return
        if not self.tie_word_embedding:
            return
        if jit_mode:
            if rank == 0:
                logging.info('clone emb.weight to output.weight')
            self.output_layer.weight = torch.nn.Parameter(self.embed[0].weight.clone())
        else:
            if rank == 0:
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

    def __init__(self, vocab_size: 'int', encoder_output_size: 'int', attention_heads: 'int'=4, linear_units: 'int'=2048, num_blocks: 'int'=6, r_num_blocks: 'int'=0, dropout_rate: 'float'=0.1, positional_dropout_rate: 'float'=0.1, self_attention_dropout_rate: 'float'=0.0, src_attention_dropout_rate: 'float'=0.0, input_layer: 'str'='embed', use_output_layer: 'bool'=True, normalize_before: 'bool'=True, src_attention: 'bool'=True, query_bias: 'bool'=True, key_bias: 'bool'=True, value_bias: 'bool'=True, activation_type: 'str'='relu', gradient_checkpointing: 'bool'=False, tie_word_embedding: 'bool'=False, use_sdpa: 'bool'=False, layer_norm_type: 'str'='layer_norm', norm_eps: 'float'=1e-05, n_kv_head: 'Optional[int]'=None, head_dim: 'Optional[int]'=None, mlp_type: 'str'='position_wise_feed_forward', mlp_bias: 'bool'=True, n_expert: 'int'=8, n_expert_activated: 'int'=2):
        super().__init__()
        self.use_sdpa = use_sdpa
        self.tie_word_embedding = tie_word_embedding
        self.left_decoder = TransformerDecoder(vocab_size, encoder_output_size, attention_heads, linear_units, num_blocks, dropout_rate, positional_dropout_rate, self_attention_dropout_rate, src_attention_dropout_rate, input_layer, use_output_layer, normalize_before, src_attention=src_attention, query_bias=query_bias, key_bias=key_bias, value_bias=value_bias, activation_type=activation_type, gradient_checkpointing=gradient_checkpointing, tie_word_embedding=tie_word_embedding, use_sdpa=use_sdpa, layer_norm_type=layer_norm_type, norm_eps=norm_eps, n_kv_head=n_kv_head, head_dim=head_dim, mlp_type=mlp_type, mlp_bias=mlp_bias, n_expert=n_expert, n_expert_activated=n_expert_activated)
        self.right_decoder = TransformerDecoder(vocab_size, encoder_output_size, attention_heads, linear_units, r_num_blocks, dropout_rate, positional_dropout_rate, self_attention_dropout_rate, src_attention_dropout_rate, input_layer, use_output_layer, normalize_before, src_attention=src_attention, query_bias=query_bias, key_bias=key_bias, value_bias=value_bias, activation_type=activation_type, gradient_checkpointing=gradient_checkpointing, tie_word_embedding=tie_word_embedding, use_sdpa=use_sdpa, layer_norm_type=layer_norm_type, norm_eps=norm_eps, n_kv_head=n_kv_head, head_dim=head_dim, mlp_type=mlp_type, mlp_bias=mlp_bias, n_expert=n_expert, n_expert_activated=n_expert_activated)

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


class BaseEncoder(torch.nn.Module):

    def __init__(self, input_size: 'int', output_size: 'int'=256, attention_heads: 'int'=4, linear_units: 'int'=2048, num_blocks: 'int'=6, dropout_rate: 'float'=0.1, positional_dropout_rate: 'float'=0.1, attention_dropout_rate: 'float'=0.0, input_layer: 'str'='conv2d', pos_enc_layer_type: 'str'='abs_pos', normalize_before: 'bool'=True, static_chunk_size: 'int'=0, use_dynamic_chunk: 'bool'=False, global_cmvn: 'torch.nn.Module'=None, use_dynamic_left_chunk: 'bool'=False, gradient_checkpointing: 'bool'=False, use_sdpa: 'bool'=False, layer_norm_type: 'str'='layer_norm', norm_eps: 'float'=1e-05):
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
            query_bias: whether use bias in attention.linear_q
            key_bias: whether use bias in attention.linear_k, False for whisper models.
            value_bias: whether use bias in attention.linear_v
            gradient_checkpointing: rerunning a forward-pass segment for each
                checkpointed segment during backward.
            use_sdpa: whether to use SDPA, currently only support transformer for now
        """
        super().__init__()
        self._output_size = output_size
        self.global_cmvn = global_cmvn
        pos_emb_class = WENET_EMB_CLASSES[pos_enc_layer_type]
        self.embed = WENET_SUBSAMPLE_CLASSES[input_layer](input_size, output_size, dropout_rate, pos_emb_class(output_size, positional_dropout_rate) if pos_enc_layer_type != 'rope_pos' else pos_emb_class(output_size, output_size // attention_heads, positional_dropout_rate))
        assert layer_norm_type in ['layer_norm', 'rms_norm']
        self.normalize_before = normalize_before
        self.after_norm = WENET_NORM_CLASSES[layer_norm_type](output_size, eps=norm_eps)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.gradient_checkpointing = gradient_checkpointing
        self.use_sdpa = use_sdpa

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
        chunk_masks = add_optional_chunk_mask(xs, masks, self.use_dynamic_chunk, self.use_dynamic_left_chunk, decoding_chunk_size, self.static_chunk_size, num_decoding_left_chunks, max_chunk_size=int(100.0 / self.embed.subsampling_rate))
        if self.use_sdpa:
            chunk_masks = mask_to_bias(chunk_masks, xs.dtype)
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
            xs, chunk_masks, _, _ = ckpt.checkpoint(layer.__call__, xs, chunk_masks, pos_emb, mask_pad, use_reentrant=False)
        return xs

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
            if elayers == 0:
                kv_cache = att_cache, att_cache
            else:
                i_kv_cache = att_cache[i:i + 1]
                size = att_cache.size(-1) // 2
                kv_cache = i_kv_cache[:, :, :, :size], i_kv_cache[:, :, :, size:]
            xs, _, new_kv_cache, new_cnn_cache = layer(xs, att_mask, pos_emb, att_cache=kv_cache, cnn_cache=cnn_cache[i] if cnn_cache.size(0) > 0 else cnn_cache)
            new_att_cache = torch.cat(new_kv_cache, dim=-1)
            r_att_cache.append(new_att_cache[:, :, next_cache_start:, :])
            r_cnn_cache.append(new_cnn_cache.unsqueeze(0))
        if self.normalize_before:
            xs = self.after_norm(xs)
        r_att_cache = torch.cat(r_att_cache, dim=0)
        r_cnn_cache = torch.cat(r_cnn_cache, dim=0)
        return xs, r_att_cache, r_cnn_cache

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
               with cache, which is not prefered.
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


class TransformerEncoder(BaseEncoder):
    """Transformer encoder module."""

    def __init__(self, input_size: 'int', output_size: 'int'=256, attention_heads: 'int'=4, linear_units: 'int'=2048, num_blocks: 'int'=6, dropout_rate: 'float'=0.1, positional_dropout_rate: 'float'=0.1, attention_dropout_rate: 'float'=0.0, input_layer: 'str'='conv2d', pos_enc_layer_type: 'str'='abs_pos', normalize_before: 'bool'=True, static_chunk_size: 'int'=0, use_dynamic_chunk: 'bool'=False, global_cmvn: 'torch.nn.Module'=None, use_dynamic_left_chunk: 'bool'=False, query_bias: 'bool'=True, key_bias: 'bool'=True, value_bias: 'bool'=True, activation_type: 'str'='relu', gradient_checkpointing: 'bool'=False, use_sdpa: 'bool'=False, layer_norm_type: 'str'='layer_norm', norm_eps: 'float'=1e-05, n_kv_head: 'Optional[int]'=None, head_dim: 'Optional[int]'=None, selfattention_layer_type: 'str'='selfattn', mlp_type: 'str'='position_wise_feed_forward', mlp_bias: 'bool'=True, n_expert: 'int'=8, n_expert_activated: 'int'=2):
        """ Construct TransformerEncoder

        See Encoder for the meaning of each parameter.
        """
        super().__init__(input_size, output_size, attention_heads, linear_units, num_blocks, dropout_rate, positional_dropout_rate, attention_dropout_rate, input_layer, pos_enc_layer_type, normalize_before, static_chunk_size, use_dynamic_chunk, global_cmvn, use_dynamic_left_chunk, gradient_checkpointing, use_sdpa, layer_norm_type, norm_eps)
        assert selfattention_layer_type in ['selfattn', 'rope_abs_selfattn']
        activation = WENET_ACTIVATION_CLASSES[activation_type]()
        mlp_class = WENET_MLP_CLASSES[mlp_type]
        self.encoders = torch.nn.ModuleList([TransformerEncoderLayer(output_size, WENET_ATTENTION_CLASSES[selfattention_layer_type](attention_heads, output_size, attention_dropout_rate, query_bias, key_bias, value_bias, use_sdpa, n_kv_head, head_dim), mlp_class(output_size, linear_units, dropout_rate, activation, mlp_bias, n_expert=n_expert, n_expert_activated=n_expert_activated), dropout_rate, normalize_before, layer_norm_type=layer_norm_type, norm_eps=norm_eps) for _ in range(num_blocks)])


class ConformerEncoder(BaseEncoder):
    """Conformer encoder module."""

    def __init__(self, input_size: 'int', output_size: 'int'=256, attention_heads: 'int'=4, linear_units: 'int'=2048, num_blocks: 'int'=6, dropout_rate: 'float'=0.1, positional_dropout_rate: 'float'=0.1, attention_dropout_rate: 'float'=0.0, input_layer: 'str'='conv2d', pos_enc_layer_type: 'str'='rel_pos', normalize_before: 'bool'=True, static_chunk_size: 'int'=0, use_dynamic_chunk: 'bool'=False, global_cmvn: 'torch.nn.Module'=None, use_dynamic_left_chunk: 'bool'=False, positionwise_conv_kernel_size: 'int'=1, macaron_style: 'bool'=True, selfattention_layer_type: 'str'='rel_selfattn', activation_type: 'str'='swish', use_cnn_module: 'bool'=True, cnn_module_kernel: 'int'=15, causal: 'bool'=False, cnn_module_norm: 'str'='batch_norm', query_bias: 'bool'=True, key_bias: 'bool'=True, value_bias: 'bool'=True, conv_bias: 'bool'=True, gradient_checkpointing: 'bool'=False, use_sdpa: 'bool'=False, layer_norm_type: 'str'='layer_norm', norm_eps: 'float'=1e-05, n_kv_head: 'Optional[int]'=None, head_dim: 'Optional[int]'=None, mlp_type: 'str'='position_wise_feed_forward', mlp_bias: 'bool'=True, n_expert: 'int'=8, n_expert_activated: 'int'=2):
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
        super().__init__(input_size, output_size, attention_heads, linear_units, num_blocks, dropout_rate, positional_dropout_rate, attention_dropout_rate, input_layer, pos_enc_layer_type, normalize_before, static_chunk_size, use_dynamic_chunk, global_cmvn, use_dynamic_left_chunk, gradient_checkpointing, use_sdpa, layer_norm_type, norm_eps)
        activation = WENET_ACTIVATION_CLASSES[activation_type]()
        encoder_selfattn_layer_args = attention_heads, output_size, attention_dropout_rate, query_bias, key_bias, value_bias, use_sdpa, n_kv_head, head_dim
        positionwise_layer_args = output_size, linear_units, dropout_rate, activation, mlp_bias, n_expert, n_expert_activated
        convolution_layer_args = output_size, cnn_module_kernel, activation, cnn_module_norm, causal, conv_bias
        mlp_class = WENET_MLP_CLASSES[mlp_type]
        self.encoders = torch.nn.ModuleList([ConformerEncoderLayer(output_size, WENET_ATTENTION_CLASSES[selfattention_layer_type](*encoder_selfattn_layer_args), mlp_class(*positionwise_layer_args), mlp_class(*positionwise_layer_args) if macaron_style else None, ConvolutionModule(*convolution_layer_args) if use_cnn_module else None, dropout_rate, normalize_before, layer_norm_type=layer_norm_type, norm_eps=norm_eps) for _ in range(num_blocks)])


class Whisper(ASRModel):

    def __init__(self, vocab_size: 'int', encoder: 'TransformerEncoder', decoder: 'TransformerDecoder', ctc: 'CTC'=None, ctc_weight: 'float'=0.5, ignore_id: 'int'=IGNORE_ID, reverse_weight: 'float'=0.0, lsm_weight: 'float'=0.0, length_normalized_loss: 'bool'=False, special_tokens: 'dict'=None):
        super().__init__(vocab_size, encoder, decoder, ctc, ctc_weight, ignore_id, reverse_weight, lsm_weight, length_normalized_loss, special_tokens)
        assert reverse_weight == 0.0
        self.sos = special_tokens['sot']
        self.eos = special_tokens['eot']
        self.decode_maxlen = self.decoder.embed[1].max_len

    def set_alignment_heads(self, dump: 'bytes'):
        raise NotImplementedError

    @property
    def is_multilingual(self):
        return self.vocab_size >= 51865

    @property
    def num_languages(self):
        return self.vocab_size - 51765 - int(self.is_multilingual)

    def _calc_att_loss(self, encoder_out: 'torch.Tensor', encoder_mask: 'torch.Tensor', ys_pad: 'torch.Tensor', ys_pad_lens: 'torch.Tensor', infos: 'Dict[str, List[str]]') ->Tuple[torch.Tensor, float]:
        prev_len = ys_pad.size(1)
        ys_in_pad, ys_out_pad = add_whisper_tokens(self.special_tokens, ys_pad, self.ignore_id, tasks=infos['tasks'], no_timestamp=True, langs=infos['langs'], use_prev=False)
        cur_len = ys_in_pad.size(1)
        ys_in_lens = ys_pad_lens + cur_len - prev_len
        decoder_out, r_decoder_out, _ = self.decoder(encoder_out, encoder_mask, ys_in_pad, ys_in_lens)
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(decoder_out.view(-1, self.vocab_size), ys_out_pad, ignore_label=self.ignore_id)
        return loss_att, acc_att


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BPUIdentity,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (Conv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv2dValid,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Conv3d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvolutionModule,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (GatedVariantsMLP,
     lambda: ([], {'idim': 4, 'hidden_units': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GlobalCMVN,
     lambda: ([], {'mean': torch.rand([4, 4]), 'istd': torch.rand([4, 4])}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LFR,
     lambda: ([], {}),
     lambda: ([torch.ones([4, 4, 4], dtype=torch.int64), torch.ones([4], dtype=torch.int64)], {})),
    (LabelSmoothingLoss,
     lambda: ([], {'size': 4, 'padding_idx': 4, 'smoothing': 4}),
     lambda: ([torch.rand([64, 4, 4, 4]), torch.ones([1024], dtype=torch.int64)], {})),
    (LearnablePositionalEncoding,
     lambda: ([], {'d_model': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MAELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MergedLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MoEFFNLayer,
     lambda: ([], {'idim': 4, 'hidden_units': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (MultiHeadedAttention,
     lambda: ([], {'n_head': 4, 'n_feat': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MultiHeadedCrossAttention,
     lambda: ([], {'n_head': 4, 'n_feat': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (NoPositionalEncoding,
     lambda: ([], {'d_model': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ParaformerPositinoalEncoding,
     lambda: ([], {'depth': 1, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 0])], {})),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PositionwiseFeedForward,
     lambda: ([], {'idim': 4, 'hidden_units': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PositionwiseFeedForwardDecoderSANM,
     lambda: ([], {'idim': 4, 'hidden_units': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RMSNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RelPositionalEncoding,
     lambda: ([], {'d_model': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RopePositionalEncoding,
     lambda: ([], {'d_model': 4, 'head_dim': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ShawRelPositionMultiHeadedAttention,
     lambda: ([], {'n_head': 4, 'n_feat': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (SinusoidalPositionEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TransducerJoint,
     lambda: ([], {'vocab_size': 4, 'enc_output_size': 4, 'pred_output_size': 4, 'join_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (WhisperPositionalEncoding,
     lambda: ([], {'d_model': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

