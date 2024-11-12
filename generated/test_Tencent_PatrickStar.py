
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


from torch.utils.data import SequentialSampler


import logging


import time


from torch.utils.data import DataLoader


from sklearn.model_selection import train_test_split


import warnings


from typing import Any


from typing import Iterable


from typing import List


from typing import Tuple


import math


from torch import nn


from torch.nn import BCEWithLogitsLoss


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


import copy


from math import floor as floor


import numpy as np


from torch.utils.checkpoint import checkpoint


from typing import Optional


from time import sleep


import functools


from copy import deepcopy


import torch.nn as nn


from abc import ABC


from abc import abstractmethod


from collections import OrderedDict


import itertools


import torch.distributed as dist


from torch.utils.cpp_extension import BuildExtension


import matplotlib.pyplot as plt


import matplotlib.patches as patches


from torch.multiprocessing import Process


from torch.nn import Embedding as TorchEmbedding


class BertSelfAttention(nn.Module):

    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, 'embedding_size'):
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(config, 'position_embedding_type', 'absolute')
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        if self.is_decoder:
            past_key_value = key_layer, value_layer
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding
            if self.position_embedding_type == 'relative_key':
                relative_position_scores = torch.einsum('bhld,lrd->bhlr', query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == 'relative_key_query':
                relative_position_scores_query = torch.einsum('bhld,lrd->bhlr', query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum('bhrd,lrd->bhlr', key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads)
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


def partition_uniform(num_items, num_parts):
    parts = [0] * (num_parts + 1)
    if num_items <= num_parts:
        for p in range(num_parts + 1):
            parts[p] = min(p, num_items)
        return parts
    chunksize = floor(num_items / num_parts)
    for p in range(num_parts):
        parts[p] = min(chunksize * p, num_items)
    parts[num_parts] = num_items
    return parts


def split_tensor_along_last_dim(tensor, partitions, contiguous_split_chunks=False):
    """Split a tensor along its last dimension. Adapted from Megatron-LM.
    Arguments:
        tensor: input tensor.
        partitions: list of partition sizes to supply to torch.split
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    last_dim = tensor.dim() - 1
    tensor_list = torch.split(tensor, partitions, dim=last_dim)
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)
    return tensor_list


class TiledLinear(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True, in_splits=1, out_splits=1, input_is_already_split=False, combine_out_splits=True, linear_cls=torch.nn.Linear, init_linear=None, **kwargs):
        """A replacement for ``torch.nn.Linear`` that works with ZeRO-3 to reduce
        memory requirements via tiling.
        TiledLinear breaks the input and output dimensions of a linear layer
        into tiles that are processed in sequence. This class enables huge
        linear layers when combined with ZeRO-3 because inactive tiles can be
        partitioned and offloaded.
        .. note::
            We recommend using as few tiles as necessary. Tiling
            significantly reduces memory usage, but can reduce throughput
            for inexpensive layers. This due to the smaller kernels having
            less parallelism and lower arithmetic intensity, while
            introducing more frequent synchronization and communication.
        Args:
            in_features (int): See ``torch.nn.Linear``
            out_features (int): See ``torch.nn.Linear``
            bias (bool, optional): See ``torch.nn.Linear``
            in_splits (int, optional): The number of tiles along the input dimension. Defaults to 1.
            out_splits (int, optional): The number of tiles along the output dimension. Defaults to 1.
            input_is_already_split (bool, optional): If set to ``True``, assume that the ``input_`` in
                to ``forward()`` is already split into ``in_splits`` chunks. Defaults to ``False``.
            combine_out_splits (bool, optional): If set to ``False``, do not combine the ``out_splits`` outputs
                into a single tensor. Defaults to ``True``.
            linear_cls (class, optional): The underlying class to build individual tiles.
                Defaults to ``torch.nn.Linear``.
            init_linear (``torch.nn.Linear``, optional): If set, copy the parameters of
                ``init_linear``. Useful for debugging. Defaults to ``None``.
            kwargs (dict, optional): additional keyword arguments to provide to ``linear_cls()``.
        Raises:
            RuntimeError: ``in_splits`` must be within the range [1, in_features).
            RuntimeError: ``out_splits`` must be within the range of [1, out_features).
        """
        super().__init__()
        if in_splits < 1 or in_splits > in_features:
            raise RuntimeError('in splits must be in range [1, in_features].')
        if out_splits < 1 or out_splits > out_features:
            raise RuntimeError('out splits must be in range [1, out_features].')
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.out_splits = out_splits
        self.in_splits = in_splits
        self.input_is_already_split = input_is_already_split
        self.combine_out_splits = combine_out_splits
        self.in_parts = partition_uniform(num_items=in_features, num_parts=in_splits)
        self.out_parts = partition_uniform(num_items=out_features, num_parts=out_splits)
        assert len(self.out_parts) == out_splits + 1
        assert len(self.in_parts) == in_splits + 1
        assert self.out_parts[0] == 0
        assert self.out_parts[out_splits] == out_features
        assert self.in_parts[in_splits] == in_features
        self.linears = torch.nn.ModuleList()
        for out_id in range(out_splits):
            self.linears.append(torch.nn.ModuleList())
            local_out_dim = self.out_parts[out_id + 1] - self.out_parts[out_id]
            for in_id in range(in_splits):
                local_bias = bias if in_id == in_splits - 1 else False
                local_in_dim = self.in_parts[in_id + 1] - self.in_parts[in_id]
                local = linear_cls(local_in_dim, local_out_dim, bias=local_bias, **kwargs)
                self.linears[out_id].append(local)
        if init_linear is not None:
            self.copy_params_from(init_linear)

    @torch.no_grad()
    def copy_params_from(self, other):
        """Copy the weight and bias data from ``other``.
        This is especially useful for reproducible initialization and testing.
        Equivalent to:
        .. code-block:: python
            with torch.no_grad():
                self.weight.copy_(other.weight)
                if self.bias is not None:
                    self.bias.copy_(other.bias)
        .. note::
            If ZeRO-3 is enabled, this is a collective operation and the
            updated parameters of data-parallel rank 0 will be visible on all
            ranks. See :class:`deepspeed.zero.GatheredParameters` for more
            information.
        Args:
            other (``torch.nn.Linear``): the linear layer to copy from.
        """
        assert hasattr(other, 'weight')
        assert other.weight.size() == (self.out_features, self.in_features)
        if self.use_bias:
            assert hasattr(other, 'bias')
            assert other.bias is not None
            assert other.bias.size() == (self.out_features,)
        else:
            assert other.bias is None
        for row in range(self.out_splits):
            rstart = self.out_parts[row]
            rstop = self.out_parts[row + 1]
            for col in range(self.in_splits):
                cstart = self.in_parts[col]
                cstop = self.in_parts[col + 1]
                local = self.linears[row][col]
                global_weight = other.weight[rstart:rstop, cstart:cstop]
                local.weight.copy_(global_weight)
            if local.bias is not None:
                local.bias.data.copy_(other.bias[rstart:rstop].data)

    def forward(self, input_):
        if self.in_splits > 1 and not self.input_is_already_split:
            input_parts = partition_uniform(input_.shape[-1], self.in_splits)
            split_sizes = [(input_parts[p + 1] - input_parts[p]) for p in range(self.in_splits)]
            inputs = self._split_global_input(input_, split_sizes)
        elif self.in_splits > 1:
            inputs = input_
            assert len(inputs) == self.in_splits, f'Col splits {self.in_splits} does not match input splits {len(inputs)}'
        else:
            inputs = [input_]
        outputs = [None] * self.out_splits
        for out_id in range(self.out_splits):
            for in_id in range(self.in_splits):
                local_output = self.linears[out_id][in_id](inputs[in_id])
                outputs[out_id] = self._reduce_local_output(in_id=in_id, out_id=out_id, current_out=outputs[out_id], new_out=local_output)
        if self.combine_out_splits:
            return self._combine_output_splits(outputs)
        return outputs

    def _split_global_input(self, input, split_sizes):
        """Partition an input tensor along the last dimension, aligned with given splits.
        Subclasses should override this method to account for new input types.
        Args:
            input (List[Tensor]): The tensor to partition along the last dimension.
            split_sizes (List[int]): The size of each partition.
        Returns:
            List[Any]: A list of the chunks of ``input``.
        """
        return split_tensor_along_last_dim(input, split_sizes)

    def _reduce_local_output(self, in_id, out_id, current_out, new_out):
        """Reduce (sum) a new local result into the existing local results.
        Subclasses should override this method.
        For a given ``out_id``, this method is called ``in_id-1`` times. The first input
        split is a simple assignment.
        Args:
            in_id (int): The input split that produced ``new_out``.
            out_id (int): The output split that produced ``new_out``.
            current_out (Any): The reduced form of all previous ``out_id`` results.
            new_out (Any): The local result from forward (``in_id``, ``out_id``)e
        Returns:
            Any: The combined result of ``current_out`` and ``new_out``.
        """
        if current_out is None:
            return new_out.clone()
        else:
            return current_out + new_out

    def _combine_output_splits(self, outputs):
        """Join the splits of the output into a single result.
        Args:
            outputs (List[Any]): The reduced outputs for each output split.
        Returns:
            Any: The combined outputs.
        """
        assert len(outputs) == self.out_splits
        return torch.cat(outputs, dim=-1)


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super().__init__()
        if global_opt_flags.USE_TILE:
            self.dense = TiledLinear(in_features=config.hidden_size, out_features=config.intermediate_size, linear_cls=nn.Linear, in_splits=1, out_splits=4)
        else:
            self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        if global_opt_flags.USE_TILE:
            self.dense = TiledLinear(in_features=config.intermediate_size, out_features=config.hidden_size, linear_cls=nn.Linear, in_splits=4, out_splits=1)
        else:
            self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f'{self} should be used as a decoder model if cross attention is added')
            self.crossattention = BertAttention(config, position_embedding_type='absolute')
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions, past_key_value=self_attn_past_key_value)
        attention_output = self_attention_outputs[0]
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]
        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, 'crossattention'):
                raise ValueError(f'If `encoder_hidden_states` are passed,                        {self} has to be instantiated with cross-attention                             layers by setting `config.add_cross_attention=True`')
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, cross_attn_past_key_value, output_attentions)
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value
        layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
        outputs = (layer_output,) + outputs
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class Encoder(torch.nn.Module):

    def __init__(self, hidden_dim, is_ckp=False):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.Linear(hidden_dim, hidden_dim))
        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear5 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.is_ckp = is_ckp

    def forward(self, x):
        h2 = self.linear1(x)
        if self.is_ckp:
            h3 = checkpoint(self.linear3, h2)
        else:
            h3 = self.linear3(h2)
        h4 = self.linear4(h3)
        h5 = self.linear5(h4)
        return h5


class SimpleModel(torch.nn.Module):

    def __init__(self, hidden_dim, seq_len, is_ckp=False, is_share_param=False):
        super(SimpleModel, self).__init__()
        config = BertConfig()
        config.vocab_size = 25
        config.max_position_embeddings = seq_len
        config.hidden_size = hidden_dim
        self.embeddings_1 = BertEmbeddings(config)
        self._is_share_param = is_share_param
        if is_share_param:
            self.embeddings_2 = self.embeddings_1
        else:
            self.embeddings_2 = BertEmbeddings(config)
        self.encoder = Encoder(hidden_dim, is_ckp)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        h1 = self.embeddings_1(x)
        h2 = self.embeddings_2(x)
        h3 = h1 + h2
        h3 = self.encoder(h3)
        return self.cross_entropy_loss(h3[:, 0], y)


class LoggerFactory:

    @staticmethod
    def create_logger(name=None, level=logging.WARNING):
        """create a logger
        Args:
            name (str): name of the logger
            level: level of logger
        Raises:
            ValueError is name is None
        """
        if name is None:
            raise ValueError('name for logger cannot be None')
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        logger_ = logging.getLogger(name)
        logger_.setLevel(level)
        logger_.propagate = False
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setFormatter(formatter)
        logger_.addHandler(RichHandler())
        return logger_


class _CopyInputToCPU(torch.autograd.Function):

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_):
        logger.debug(f'Copy input to cpu and {input_.dtype}.')
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        target_device = torch.device(f'cuda:{torch.cuda.current_device()}')
        logger.debug('Copy grad_output to cuda.')
        return grad_output


def copy_to_cpu(input_):
    return _CopyInputToCPU.apply(input_)


class _CopyActToGPU(torch.autograd.Function):

    @staticmethod
    def symbolic(graph, input_):
        target_device = torch.device(f'cuda:{torch.cuda.current_device()}')
        return input_

    @staticmethod
    def forward(ctx, input_):
        target_device = torch.device(f'cuda:{torch.cuda.current_device()}')
        logger.debug(f'Copy grad_output to cuda, input dtype {input_.dtype}.')
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.float()


def copy_to_gpu(input_):
    return _CopyActToGPU.apply(input_)


class Embedding(nn.Embedding):
    """CPU Embedding.

    If `use_cpu` is set, the embedding operations will
    be performed on CPU.
    """
    use_cpu = False
    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cpu = Embedding.use_cpu
        Embedding.instances.append(self)

    def forward(self, input_):
        if self.use_cpu:
            input_ = copy_to_cpu(input_)
        else:
            input_ = copy_to_gpu(input_)
        output = super().forward(input_)
        if self.use_cpu:
            output = copy_to_gpu(output)
        return output


class DynamicLossScaler:
    """
    Class that manages dynamic loss scaling.  It is recommended to use :class:`DynamicLossScaler`
    indirectly, by supplying ``dynamic_loss_scale=True`` to the constructor of
    :class:`Fp16Optimizer`.  However, it's important to understand how :class:`DynamicLossScaler`
    operates, because the default options can be changed using the
    the ``dynamic_loss_args`` argument to :class:`Fp16Optimizer`'s constructor.
    Loss scaling is designed to combat the problem of underflowing gradients encountered at long
    times when training fp16 networks.  Dynamic loss scaling begins by attempting a very high loss
    scale.  Ironically, this may result in OVERflowing gradients.  If overflowing gradients are
    encountered, :class:`DynamicLossScaler` informs :class:`Fp16Optimizer` that an overflow has
    occurred.
    :class:`Fp16Optimizer` then skips the update step for this particular iteration/minibatch,
    and :class:`DynamicLossScaler` adjusts the loss scale to a lower value.
    If a certain number of iterations occur without overflowing gradients detected,
    :class:`DynamicLossScaler` increases the loss scale once more.
    In this way :class:`DynamicLossScaler` attempts to "ride the edge" of
    always using the highest loss scale possible without incurring overflow.
    Args:
        init_scale (float, optional, default=2**32):  Initial loss scale attempted by :class:`DynamicLossScaler.`
        scale_factor (float, optional, default=2.0):  Factor used when adjusting the loss scale. If an overflow
        is encountered, the loss scale is readjusted to loss scale/``scale_factor``.
        If ``scale_window`` consecutive iterations take place without an overflow,
        the loss scale is readjusted to loss_scale*``scale_factor``.
        scale_window (int, optional, default=1000):  Number of consecutive iterations
        without an overflow to wait before increasing the loss scale.
    """

    def __init__(self, init_scale=2 ** 32, scale_factor=2.0, scale_window=1000, min_scale=1, delayed_shift=1, consecutive_hysteresis=False):
        self.cur_scale = init_scale
        self.cur_iter = 0
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.min_scale = min_scale
        self.delayed_shift = delayed_shift
        self.cur_hysteresis = delayed_shift
        self.consecutive_hysteresis = consecutive_hysteresis

    def has_overflow(self, param):
        if DynamicLossScaler._has_inf_or_nan(param.grad):
            return True
        return False

    def _has_inf_or_nan(x):
        try:
            cpu_sum = float(x.float().sum())
        except RuntimeError as instance:
            if 'value cannot be converted' not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
                return True
            return False

    def update_scale(self, overflow):
        if not hasattr(self, 'min_scale'):
            self.min_scale = 1
        if not hasattr(self, 'delayed_shift'):
            self.delayed_shift = 1
        if not hasattr(self, 'cur_hysteresis'):
            self.cur_hysteresis = 1
        if not hasattr(self, 'consecutive_hysteresis'):
            self.consecutive_hysteresis = True
        if overflow:
            if self.delayed_shift == 1 or self.cur_hysteresis == 1:
                self.cur_scale = max(self.cur_scale / self.scale_factor, self.min_scale)
            else:
                self.cur_hysteresis -= 1
            self.last_overflow_iter = self.cur_iter
        else:
            if self.consecutive_hysteresis:
                self.cur_hysteresis = self.delayed_shift
            if (self.cur_iter - self.last_overflow_iter) % self.scale_window == 0:
                if not self.consecutive_hysteresis:
                    self.cur_hysteresis = self.delayed_shift
                self.cur_scale *= self.scale_factor
        self.cur_iter += 1

    @property
    def loss_scale(self):
        return self.cur_scale

    def backward(self, loss, retain_graph=False):
        scaled_loss = loss * self.loss_scale
        scaled_loss.backward(retain_graph=retain_graph)

