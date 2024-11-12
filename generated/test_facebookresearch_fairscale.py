
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


from collections import namedtuple


import torch


from torch.utils.data import DataLoader


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data import Dataset


import time


from torch import nn


from torch.cuda import Event


import logging


import math


import warnings


from torch.distributed import rpc


import torch.multiprocessing as mp


import torch.nn as nn


from torch.optim.optimizer import Optimizer


from functools import reduce


import numpy as np


from torch.optim import Adam


from torch.utils.data.dataloader import DataLoader


from torchvision.datasets import FakeData


from torchvision.transforms import ToTensor


import torch.distributed as dist


from torch.nn.parallel import DistributedDataParallel as DDP


from collections import defaultdict


from enum import Enum


from typing import Any


from typing import List


from typing import Optional


from typing import cast


import torch.autograd.profiler as profiler


from torch.cuda.amp import GradScaler as TorchGradScaler


from torch.utils.data import BatchSampler


from torch.utils.data import Sampler


from torchvision.datasets import MNIST


from torchvision.transforms import Compose


from torchvision.transforms import Resize


from typing import Dict


from typing import Tuple


from typing import Union


from torch.autograd.profiler import record_function


from torch.distributed import ProcessGroup


from typing import Set


import torch.fx


from torch.fx.node import Node


import functools


from typing import Callable


from typing import Iterable


from torch.autograd import Variable


from torch.nn.modules import Module


from typing import Iterator


from abc import ABC


from abc import abstractmethod


from math import log as mlog


from collections import deque


from functools import partial


from typing import ClassVar


from typing import Deque


import collections


from typing import MutableMapping


from torch import Tensor


from torch.distributed.nn import RemoteModule


from types import TracebackType


from typing import Type


import inspect


import torch.nn.functional as F


from enum import auto


from functools import lru_cache


from typing import NamedTuple


from typing import Sequence


from torch.utils.hooks import RemovableHandle


import copy


import random


from typing import TYPE_CHECKING


from typing import Generator


import numpy


from collections import OrderedDict


from torch.nn.utils.rnn import PackedSequence


import collections.abc as abc


from math import inf


import re


import torch.utils.checkpoint as torch_checkpoint


from torch.nn.modules.batchnorm import _BatchNorm


from itertools import groupby


import typing


from typing import Mapping


from torch.nn.parameter import Parameter


from itertools import chain


import torch.nn.init as init


from torch.cuda import _lazy_call


from torch.utils.checkpoint import detach_variable


from torch.nn import Module


from torch.nn import ModuleList


import itertools


from typing import TypeVar


from torch import ByteTensor


import torch.autograd


from queue import Empty as QueueEmpty


from queue import Queue


import torch.cuda.comm


import torch.cuda


from torch.distributed.distributed_c10d import _get_global_rank


from typing import FrozenSet


from torch.optim import SGD


from torch.optim import Optimizer


from torch.cuda import FloatTensor


from torch.cuda.amp.common import amp_definitely_not_available


from torch.cuda.amp.grad_scaler import GradScaler as TorchGradScaler


from torch.optim.sgd import SGD


from torch.autograd import profiler


from torch.nn import Parameter


import torch.distributed


import torch.nn


import torch.distributed.autograd as dist_autograd


from torch.distributed.optim import DistributedOptimizer


import torch.distributed.rpc as rpc


import torch.optim as optim


from torch.nn.parallel import DistributedDataParallel


from torch.utils.checkpoint import checkpoint as torch_checkpoint_wrapper


from torch.nn import BatchNorm2d


from torch.nn import LayerNorm


from torch.nn import Linear


from torch.nn import Sequential


from itertools import product


from time import time


from torch.optim import Adadelta


from torch.cuda.amp import GradScaler


from torch.nn import AdaptiveAvgPool2d


from torch.nn import Conv2d


from torch.nn import CrossEntropyLoss


from torch.nn import ReLU


from torch.nn import Sigmoid


from torch.nn import SyncBatchNorm


from copy import deepcopy


from torch.utils.checkpoint import checkpoint as torch_checkpoint


from torch import optim


from sklearn.datasets import make_blobs


from torch.cuda.amp.autocast_mode import autocast


import torchvision


import torchvision.transforms as transforms


from torch.optim.lr_scheduler import LambdaLR


class EmbeddingLayer(nn.Embedding):
    """Wrapped nn.Embedding layer to allow for weight initialization."""

    def __init__(self, ntoken, ninp, initrange):
        super().__init__(ntoken, ninp)
        self.ninp_sqrt = math.sqrt(ninp)
        self.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        return super().forward(src) * self.ninp_sqrt


class PositionalEncodingLayer(nn.Module):
    """PositionalEncoding layer for a given Transformer model."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncodingLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class FeedForwardLayer(nn.Module):
    """FeedForward layer for a given Transformer model."""

    def __init__(self, d_model, dim_feedforward, activation, dropout) ->None:
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = activation
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.linear2(self.dropout1(self.activation(self.linear1(x)))))


class _AllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx: 'Any', group: 'dist.ProcessGroup', input: 'Tensor') ->Tensor:
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx: 'Any', *grad_output: Tensor) ->Tuple[None, Tensor]:
        return None, _AllToAll.apply(ctx.group, *grad_output)


def gumbel_rsample(shape: 'Tuple', device: 'torch.device') ->Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample
        gumbel_map[device] = gumbel
    return gumbel(shape)


def one_hot(tensor: 'torch.Tensor', num_classes: 'int') ->Tensor:
    """Workaround for https://github.com/pytorch/pytorch/issues/55579"""
    assert num_classes > 0, 'num_classes must be a positive integer'
    ret = torch.zeros(tensor.shape + (num_classes,), device=tensor.device, dtype=tensor.dtype)
    ret.scatter_(-1, tensor.unsqueeze(-1), 1)
    return ret


def top2gating(logits: 'torch.Tensor') ->Tuple[Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    gates = F.softmax(logits, dim=1, dtype=torch.float)
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    capacity = 2 * num_tokens // num_experts
    assert num_tokens % num_experts == 0
    indices1_s = torch.argmax(gates, dim=1)
    mask1 = one_hot(indices1_s, num_classes=num_experts)
    logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float('-inf'))
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = one_hot(indices2_s, num_classes=num_experts)
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    locations2 += torch.sum(mask1, dim=0, keepdim=True)
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.mean(me * ce)
    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)
    gates1_s = (gates * mask1).sum(dim=1)
    gates2_s = (gates * mask2).sum(dim=1)
    denom_s = gates1_s + gates2_s
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s
    gates1 = gates1_s.unsqueeze(-1) * mask1
    gates2 = gates2_s.unsqueeze(-1) * mask2
    locations1_sc = one_hot(locations1_s, num_classes=capacity)
    locations2_sc = one_hot(locations2_s, num_classes=capacity)
    combine1_sec = gates1.unsqueeze(2) * locations1_sc.unsqueeze(1)
    combine2_sec = gates2.unsqueeze(2) * locations2_sc.unsqueeze(1)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()
    return l_aux, combine_weights, dispatch_mask


class Top2Gate(torch.nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """
    wg: 'torch.nn.Linear'

    def __init__(self, model_dim: 'int', num_experts: 'int') ->None:
        super().__init__()
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)

    def forward(self, input: 'torch.Tensor') ->Tuple[Tensor, Tensor, Tensor]:
        logits = self.wg(input)
        return top2gating(logits)


class TransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).
        is_moe: if ``True``, the feedforward layer will have MOE enabled.
        num_local_experts: number of local experts for MOE.


    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=nn.ReLU(), layer_norm_eps=1e-05, norm_first=False, is_moe=False, num_local_experts=1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.is_moe = is_moe
        if is_moe:
            world_size = 1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size()
            num_global_experts = num_local_experts * world_size
            self.gate = Top2Gate(d_model, num_global_experts)
            experts = nn.ModuleList([FeedForwardLayer(d_model, dim_feedforward, activation, dropout) for _ in range(num_local_experts)])
            self.moe_layer = MOELayer(self.gate, experts)
        else:
            self.ff_block = FeedForwardLayer(d_model, dim_feedforward, activation, dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout(x)

    def _ff_block(self, x):
        if self.is_moe:
            return self.moe_layer(x)
        else:
            return self.ff_block(x)


class TransformerDecoderLayer(TransformerEncoderLayer):
    """TransformerDecoder layer which inherits from TransformerEncoderLayer."""

    def __init__(self, ninp, nhead, nhid, dropout, is_moe=False, num_local_experts=1):
        super().__init__(ninp, nhead, nhid, dropout, is_moe=is_moe, num_local_experts=num_local_experts)
        self.src_mask = None

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src))
            self.src_mask = mask
        return super().forward(src, self.src_mask)


class LinearLayer(nn.Linear):
    """Wrapped nn.Linear layer to allow for weight initialization."""

    def __init__(self, ninp, ntoken, initrange):
        super().__init__(ninp, ntoken)
        self.bias.data.zero_()
        self.weight.data.uniform_(-initrange, initrange)


class TransformerLMSequntial(nn.Sequential):
    """A small language model based on the design of GPT-2 using nn.Sequeitnal
    for compatability with Pipe"""

    def __init__(self, ntokens, ninp, nhead, nhid, dropout, initrange, ndecoder):
        layers = [EmbeddingLayer(ntokens, ninp, initrange), PositionalEncodingLayer(ninp, dropout)]
        for _ in range(ndecoder):
            layers.append(TransformerDecoderLayer(ninp, nhead, nhid, dropout))
        layers.append(LinearLayer(ninp, ntokens, initrange))
        super(TransformerLMSequntial, self).__init__(*layers)


class TransformerLM(nn.Sequential):
    """A GPT-2 based nn.Sequential language model."""

    def __init__(self, ntokens, ninp, nhead, nhid, dropout, initrange, ndecoder, is_moe=False, num_local_experts=1):
        layers = [EmbeddingLayer(ntokens, ninp, initrange), PositionalEncodingLayer(ninp, dropout)]
        for _ in range(ndecoder):
            layers.append(TransformerDecoderLayer(ninp, nhead, nhid, dropout, is_moe, num_local_experts))
        layers.append(LinearLayer(ninp, ntokens, initrange))
        super(TransformerLM, self).__init__(*layers)


BROADCAST_BUCKET_SIZE = 10 * 1024 * 1024


class EventRecorder(object):

    def stop(self) ->None:
        pass


class Edge(object):

    def __init__(self, local_master_rank: 'int', dest: 'int', src: 'int', local_rank: 'int') ->None:
        self.src = src
        self.dest = dest
        self.process_group = dist.new_group([src, dest])
        if local_master_rank in [self.src, self.dest] and local_rank == 0:
            initializer_tensor = torch.Tensor([1])
            dist.all_reduce(initializer_tensor, group=self.process_group)
            initializer_tensor = torch.Tensor([1]).half()
            dist.all_reduce(initializer_tensor, group=self.process_group)


class GraphManager(ABC):

    def __init__(self, rank: 'int', world_size: 'int', nprocs_per_node: 'int'=1, local_rank: 'int'=0, peers_per_itr: 'int'=1) ->None:
        assert int(peers_per_itr) >= 1
        self.rank = rank
        self.world_size = world_size
        self.phone_book: 'List[List[Edge]]' = [[] for _ in range(self.world_size)]
        self._peers_per_itr = peers_per_itr
        self._group_indices = list(range(peers_per_itr))
        self.nprocs_per_node = nprocs_per_node
        self.local_rank = local_rank
        self._make_graph()

    @property
    def peers_per_itr(self) ->int:
        return self._peers_per_itr

    @peers_per_itr.setter
    def peers_per_itr(self, v: 'int') ->None:
        self._peers_per_itr = v
        self._group_indices = list(range(v))

    @abstractmethod
    def _make_graph(self) ->None:
        """
        Returns a nested list of peers; the outer-list is indexed by rank,
        the inner list denotes the set of peers that 'rank' can send
        messages to at any point in time
        """
        raise NotImplementedError

    def _add_peers(self, rank: 'int', peers: 'List[int]') ->None:
        for peer in peers:
            if peer not in self.phone_book[rank]:
                self.phone_book[rank].append(Edge(local_master_rank=self.rank * self.nprocs_per_node, dest=peer * self.nprocs_per_node, src=rank * self.nprocs_per_node, local_rank=self.local_rank))

    @abstractmethod
    def is_regular_graph(self) ->bool:
        """Whether each node has the same number of in-peers as out-peers"""
        raise NotImplementedError

    @abstractmethod
    def is_bipartite_graph(self) ->bool:
        """Whether graph is bipartite or not"""
        raise NotImplementedError

    @abstractmethod
    def is_passive(self, rank: 'Optional[int]'=None) ->bool:
        """Whether 'rank' is a passive node or not"""
        raise NotImplementedError

    @abstractmethod
    def is_dynamic_graph(self) ->bool:
        """Whether the graph-type is dynamic (as opposed to static)"""
        raise NotImplementedError

    def get_peers(self, rotate: 'bool'=False) ->Tuple[List[int], List[int]]:
        """Returns the out and in-peers corresponding to 'self.rank'"""
        if rotate:
            self._rotate_group_indices()
        out_peers, in_peers = [], []
        for group_index in self._group_indices:
            out_peers.append(self.phone_book[self.rank][group_index].dest)
            for rank, peers in enumerate(self.phone_book):
                if rank == self.rank:
                    continue
                if self.rank * self.nprocs_per_node == peers[group_index].dest:
                    in_peers.append(rank)
        return out_peers, in_peers

    def get_edges(self, rotate: 'bool'=False) ->Tuple[List[Edge], List[Edge]]:
        """Returns the pairwise process groups between rank and the out and
        in-peers corresponding to 'self.rank'"""
        if rotate:
            self._rotate_group_indices()
        out_edges, in_edges = [], []
        for group_index in self._group_indices:
            out_edges.append(self.phone_book[self.rank][group_index])
            for rank, edges in enumerate(self.phone_book):
                if rank == self.rank:
                    continue
                if self.rank * self.nprocs_per_node == edges[group_index].dest:
                    in_edges.append(self.phone_book[rank][group_index])
        return out_edges, in_edges

    def _rotate_group_indices(self) ->None:
        """Incerement group indices to point to the next out-peer"""
        increment = self.peers_per_itr
        for i, group_index in enumerate(self._group_indices):
            self._group_indices[i] = int((group_index + increment) % len(self.phone_book[self.rank]))

    def _rotate_forward(self, r: 'int', p: 'int') ->int:
        """Helper function returns peer that is p hops ahead of r"""
        return (r + p) % self.world_size

    def _rotate_backward(self, r: 'int', p: 'int') ->int:
        """Helper function returns peer that is p hops behind r"""
        return (r - p) % self.world_size


HEARTBEAT_TIMEOUT = 300


class MixingManager(ABC):

    def __init__(self, graph: 'GraphManager', device: 'Optional[torch.device]') ->None:
        self.graph_manager = graph
        self.device = device

    def is_regular(self) ->bool:
        """
        Whether there is bias accumulated in local entry of stationary
        distribution of mixing matrix
        """
        return self.graph_manager.is_regular_graph() and self.is_uniform()

    @abstractmethod
    def is_uniform(self) ->bool:
        """Whether mixing weights are distributed uniformly over peers"""
        raise NotImplementedError

    @abstractmethod
    def get_mixing_weights(self, residual_adjusted: 'bool'=True) ->Dict[Union[str, int], torch.Tensor]:
        """Create mixing weight dictionary using uniform allocation"""
        raise NotImplementedError


class MultiProcessAdapter(logging.LoggerAdapter):
    """
    Creates an adapter to make logging for multiple processes cleaner
    """

    def process(self, msg: 'str', kwargs: 'Any') ->Tuple[str, MutableMapping[str, Any]]:
        process_num = kwargs.pop('process_num', self.extra['process_num'])
        return f'process: {process_num} {msg}', kwargs


class UniformMixing(MixingManager):

    def get_mixing_weights(self, residual_adjusted: 'bool'=True) ->Dict[Union[str, int], torch.Tensor]:
        """Create mixing weight dictionary using uniform allocation"""
        mixing_weights: 'Dict[Union[str, int], torch.Tensor]' = {}
        out_peers, _ = self.graph_manager.get_peers()
        w = torch.tensor([1.0 / (len(out_peers) + 1.0)], device=self.device)
        mixing_weights['lo'] = w.clone()
        w_op = w if not residual_adjusted else w / mixing_weights['lo']
        mixing_weights['uniform'] = w_op.clone()
        for op in out_peers:
            mixing_weights[op] = w_op.clone()
        return mixing_weights

    def is_uniform(self) ->bool:
        return True


class dist_backend(str, Enum):
    UNDEFINED = 'undefined'
    TCP = 'tcp'
    MPI = 'mpi'
    GLOO = 'gloo'
    NCCL = 'nccl'


class Gossiper(object):
    """Generic gossip averaging object for multi-peer communication

    Args:
        msg (torch.Tensor): message used to initialize recv buffer
        graph (GraphManager): Subclass of GraphManager
        device: (torch.Device) device on which to initialize recv buffer
        mixing (MixingManager): Subclass of MixingManager
        logger (logging.Logger): Module used to log results
        rank (int): Rank of the current process
        world_size (int): World size of the current process
    """

    def __init__(self, msg: 'torch.Tensor', graph: 'GraphManager', device: 'Optional[torch.device]'=None, mixing: 'MixingManager'=None, logger: 'logging.Logger'=None, rank: 'Optional[int]'=None, world_size: 'Optional[int]'=None) ->None:
        """
        Initialize generic averaging class designed for multi-peer comms
        """
        self.logger = logger
        if rank is None or world_size is None:
            assert dist.is_initialized()
            assert dist.get_backend() != dist_backend.GLOO
            assert dist.get_backend() != dist_backend.NCCL
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        self.rank = rank
        self.world_size = world_size
        assert isinstance(graph, GraphManager)
        self._graph_manager = graph
        self.peers_per_itr_device = torch.tensor([self._graph_manager.peers_per_itr], device=device, dtype=msg.dtype)
        self.passive = self._graph_manager.is_passive()
        self.refresh_peers_(rotate=False)
        if mixing is None:
            mixing = UniformMixing(self._graph_manager, device)
        assert isinstance(mixing, MixingManager)
        self._mixing_manager = mixing
        self.refresh_mixing_weights_()
        self.regular = self._mixing_manager.is_regular()
        self.device = device if device is not None else msg.device
        self.out_msg_buffer: 'List[Tuple[dist.Work, torch.Tensor]]' = []
        self.in_msg_buffer = msg.clone().detach_()
        self._ps_weight: 'torch.Tensor' = torch.ones(1, dtype=msg.dtype).detach_().to(self.device)
        if not self.regular:
            self.in_msg_buffer = torch.cat([self.in_msg_buffer, self.ps_weight])
        if self.device.type == 'cpu':
            try:
                self.in_msg_buffer = self.in_msg_buffer.pin_memory()
            except Exception as e:
                if self.logger is not None:
                    self.logger.error(e)
                else:
                    raise
        self.placeholder = self.in_msg_buffer.clone()

    @property
    def ps_weight(self) ->torch.Tensor:
        return self._ps_weight

    @ps_weight.setter
    def ps_weight(self, v: 'torch.Tensor') ->None:
        self._ps_weight.data[0] = v

    @property
    def peers_per_itr(self) ->int:
        return self._graph_manager.peers_per_itr

    @peers_per_itr.setter
    def peers_per_itr(self, v: 'int') ->None:
        self._graph_manager.peers_per_itr = v

    def refresh_peers_(self, rotate: 'Optional[bool]'=None) ->None:
        """Update in- and out-peers"""
        if rotate is None:
            rotate = self._graph_manager.is_dynamic_graph()
        assert not (rotate and not self._graph_manager.is_dynamic_graph())
        self.out_edges, self.in_edges = self._graph_manager.get_edges(rotate)

    def refresh_mixing_weights_(self, residual_adjusted: 'bool'=False) ->None:
        """Update mixing-matrix weights"""
        self.mixing_weights = self._mixing_manager.get_mixing_weights(residual_adjusted)

    def mix_out_msg_(self, out_msg: 'torch.Tensor', ps_weight: 'torch.Tensor') ->Iterator[torch.Tensor]:
        """Returns a generator mixing messages on the fly"""
        self.refresh_mixing_weights_(residual_adjusted=True)
        self.ps_weight = ps_weight
        if not self.regular:
            out_msg = torch.cat([out_msg, cast(torch.Tensor, self.ps_weight.type(out_msg.dtype))])
        if self._mixing_manager.is_uniform():
            weight = self.mixing_weights['uniform']
            out_msg *= weight.type(out_msg.dtype)
            for _ in self.out_edges:
                yield out_msg
        else:
            for out_edge in self.out_edges:
                weight = self.mixing_weights[out_edge.dest]
                yield out_msg.mul(weight.type(out_msg.dtype))

    def clean_msg_buffers_(self) ->None:
        """Clean outgoing message buffer"""
        while len(self.out_msg_buffer) > 0:
            req, msg = self.out_msg_buffer.pop()
            req.wait()
            msg.set_()

    def parse_in_msg_buffer(self) ->Tuple[torch.Tensor, torch.Tensor]:
        """Parse in-msg buffer and return msg and ps-weight separately"""
        msg = self.in_msg_buffer
        if not self.regular:
            return msg.narrow(0, 0, len(msg) - 1), msg[-1]
        else:
            return msg, self.ps_weight * self.peers_per_itr_device

    def mix(self, out_msg: 'torch.Tensor', ps_weight: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        """Single gossip step"""
        raise NotImplementedError


class PushPull(Gossiper):
    """Doubly-stochastic consensus averaging module"""

    def mix(self, out_msg: 'torch.Tensor', ps_weight: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        assert out_msg.device.type == self.device.type
        if self.logger is not None:
            self.logger.debug('in/out -peers {}/{}'.format(self.in_edges, self.out_edges))
        mixed_out_msgs = self.mix_out_msg_(out_msg, ps_weight)
        if len(self.in_edges) == 1 and len(self.out_edges) == 1:
            out_edge, in_edge = self.out_edges[0], self.in_edges[0]
            msg = next(mixed_out_msgs)
            if not self.passive:
                dist.broadcast(tensor=msg, src=out_edge.src, group=out_edge.process_group)
                dist.broadcast(tensor=self.in_msg_buffer, src=in_edge.src, group=in_edge.process_group)
            else:
                dist.broadcast(tensor=self.in_msg_buffer, src=in_edge.src, group=in_edge.process_group)
                dist.broadcast(tensor=msg, src=out_edge.src, group=out_edge.process_group)
        else:
            self.in_msg_buffer.zero_()
            for out_edge, in_edge in zip(self.out_edges, self.in_edges):
                msg = next(mixed_out_msgs)
                if not self.passive:
                    dist.broadcast(tensor=msg, src=out_edge.src, group=out_edge.process_group)
                    dist.broadcast(tensor=self.placeholder, src=in_edge.src, group=in_edge.process_group)
                else:
                    dist.broadcast(tensor=self.placeholder, src=in_edge.src, group=in_edge.process_group)
                    dist.broadcast(tensor=msg, src=out_edge.src, group=out_edge.process_group)
                self.in_msg_buffer.add_(self.placeholder)
        self.refresh_peers_()
        self.clean_msg_buffers_()
        return self.parse_in_msg_buffer()


class PushSum(Gossiper):
    """1-peer Push-Sum consensus averaging module"""

    def mix(self, out_msg: 'torch.Tensor', ps_weight: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        """Consensus averaging step"""
        assert out_msg.device.type == self.device.type
        if self.logger is not None:
            self.logger.debug('in/out -peers {}/{}'.format(self.in_edges, self.out_edges))
        mixed_out_msgs = self.mix_out_msg_(out_msg, ps_weight)
        for out_edge in self.out_edges:
            msg = next(mixed_out_msgs)
            assert self.rank == out_edge.src
            req = dist.broadcast(tensor=msg, src=out_edge.src, group=out_edge.process_group, async_op=True)
            self.out_msg_buffer.append((req, msg))
        if len(self.in_edges) == 1:
            in_edge = self.in_edges[0]
            dist.broadcast(tensor=self.in_msg_buffer, src=in_edge.src, group=in_edge.process_group)
        else:
            self.in_msg_buffer.zero_()
            for in_edge in self.in_edges:
                dist.broadcast(tensor=self.placeholder, src=in_edge.src, group=in_edge.process_group)
                self.in_msg_buffer.add_(self.placeholder)
        self.refresh_peers_()
        self.clean_msg_buffers_()
        return self.parse_in_msg_buffer()


class SlowMoBaseAlgorithm(str, Enum):
    LOCALSGD = 'localsgd'
    SGP = 'sgp'


def flatten_tensors(tensors: 'List[torch.Tensor]') ->torch.Tensor:
    """
    Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually
    Args:
        tensors (Iterable[Tensor]): dense tensors to flatten
    Returns:
        A 1D buffer containing input tensors
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def group_by_dtype(tensors: 'List[torch.Tensor]') ->Dict[torch.dtype, List[torch.Tensor]]:
    """
    Returns a dict mapping from the tensor dtype to a list containing all
    tensors of that dtype.
    Arg:
        tensors (Iterable[Tensor]): list of tensors
    """
    tensors_by_dtype = collections.defaultdict(list)
    for tensor in tensors:
        tensors_by_dtype[tensor.dtype].append(tensor)
    return tensors_by_dtype


def unflatten_tensors(flat: 'torch.Tensor', tensors: 'List[torch.Tensor]') ->List[torch.Tensor]:
    """
    View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by flatten_dense_tensors.
    Args:
        flat (Tensor): flattened dense tensors to unflatten
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
            unflatten flat
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return outputs


def communicate(tensors: 'List[torch.Tensor]', communication_op: 'Any', logger: 'logging.Logger'=None) ->None:
    """
    Communicate a list of tensors
    Args:
        tensors (Iterable[Tensor]): list of tensors
        communication_op: a method or partial object which takes a tensor as
            input and communicates it. It can be a partial object around
            something like torch.distributed.all_reduce
    """
    tensors_by_dtype = group_by_dtype(tensors)
    for tensors_with_same_dtype in tensors_by_dtype.values():
        flat_tensor = flatten_tensors(tensors_with_same_dtype)
        if logger is not None:
            logger.debug('Flatten completed')
        communication_op(tensor=flat_tensor)
        if logger is not None:
            logger.debug('Commmunication completed')
        with torch.no_grad():
            for f, t in zip(unflatten_tensors(flat_tensor, tensors_with_same_dtype), tensors_with_same_dtype):
                t.copy_(f)
        if logger is not None:
            logger.debug('Unflatten completed')


def create_and_record_event() ->torch.Event:
    event = torch.Event(enable_timing=True)
    event.record()
    return event


MAX_LEN_DEQUEUE = 10 ** 4


deque_with_max_len_fixed = partial(deque, maxlen=MAX_LEN_DEQUEUE)


class CudaEventRecorder(EventRecorder):
    """Allows profiling in an easy-to-use manner. CudaEventRecorder can be used
    in a loop. When it is used in a loop (or when an event recorder is created
    multiple times with the same name), get_timings returns the statistics of the
    timings since the last reset. Note: in case the number of timings is greater than
    10,000, only the last 10,000 timings are used to calculate the statistics.

    Usage:
    >>> event_recorder1 = CudaEventRecorder('1')
    >>> # Sequence of events whose time is to be measured
    >>> event_recorder1.stop()
    >>> event_recorder2 = CudaEventRecorder('2')
    >>> # Sequence of events whose time is to be measured
    >>> event_recorder2.stop()
    >>> print(CudaEventRecorder.get_timings())

    Args:
        event_name (str): The name by which the cuda event is to be referred later on

    """
    event_recorders: "ClassVar[Dict[str, Deque['CudaEventRecorder']]]" = defaultdict(deque_with_max_len_fixed)
    all_event_recorders: "ClassVar[Dict[str, Deque['CudaEventRecorder']]]" = defaultdict(deque_with_max_len_fixed)

    def __init__(self, event_name: 'str') ->None:
        self.event_name = event_name
        self.start_event = create_and_record_event()
        self.end_event: 'Optional[torch.cuda.Event]' = None
        CudaEventRecorder.event_recorders[event_name].append(self)
        CudaEventRecorder.all_event_recorders[event_name].append(self)

    def stop(self) ->None:
        self.end_event = create_and_record_event()

    def find_time_elapsed(self) ->float:
        if self.end_event is None:
            raise Exception(f'stopEvent was not called for event with name {self.event_name}')
        self.end_event.synchronize()
        return self.start_event.elapsed_time(self.end_event)

    @classmethod
    def reset(cls) ->None:
        cls.event_recorders = defaultdict(deque_with_max_len_fixed)

    @classmethod
    def get_common_timings(cls, event_recorders: "Dict[str, Deque['CudaEventRecorder']]", description: 'str') ->str:
        all_timings_str = f'{description}:\n'
        for event_name, event_recorder_list in event_recorders.items():
            time_taken_list = [event_recorder.find_time_elapsed() for event_recorder in event_recorder_list]
            all_timings_str += '{}: Time taken: avg: {}, std: {}, count: {}\n'.format(event_name, statistics.mean(time_taken_list), statistics.pstdev(time_taken_list), len(time_taken_list))
        return all_timings_str

    @classmethod
    def get_timings(cls) ->str:
        """Returns the timings since last reset was called"""
        return cls.get_common_timings(cls.event_recorders, 'Timings since last reset')

    @classmethod
    def get_all_timings(cls) ->str:
        """Returns the statistics of all the timings"""
        return cls.get_common_timings(cls.all_event_recorders, 'All timings')


class DummyCudaEventRecorder(EventRecorder):
    pass


def create_event_recorder(event_name: 'str', dummy: 'bool'=False) ->EventRecorder:
    if not dummy:
        return CudaEventRecorder(event_name)
    return DummyCudaEventRecorder()


def create_process_group(ranks: 'List[int]') ->torch.distributed.ProcessGroup:
    """
    Creates and intializes a new process group. Assumes init_process_group
    has already been called
    Arguments:
        ranks (list<int>): ranks corresponding to the processes which should
            belong the created process group
    Returns:
        New process group
    """
    new_group = dist.new_group(ranks=ranks)
    init_tensor_fp32, init_tensor_fp16 = torch.zeros(1), torch.zeros(1).half()
    for init_tensor in [init_tensor_fp32, init_tensor_fp16]:
        if torch.cuda.is_available():
            init_tensor = init_tensor
        if dist.get_rank() in ranks:
            dist.all_reduce(init_tensor, group=new_group)
        torch.cuda.synchronize()
    return new_group


def make_logger(rank: 'int', verbose: 'bool'=True) ->logging.Logger:
    """
    Return a logger for writing to stdout
    Args:
        rank (int): rank of node making logger
        verbose (bool): whether to set log-level to INFO; o.w. WARNING
    Returns:
        Python logger
    """
    logger = logging.getLogger(__name__)
    if logger not in HANDLER_AND_LEVEL_SET:
        console = logging.StreamHandler(stream=sys.stdout)
        format_str = '{}'.format(rank)
        format_str += ': %(levelname)s -- %(threadName)s -- %(message)s'
        console.setFormatter(logging.Formatter(format_str))
        logger.addHandler(console)
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        HANDLER_AND_LEVEL_SET.add(logger)
    return logger


class SlowMoDistributedDataParallel(Module):
    """Wraps an arbitrary :class:`nn.Module <torch.nn.Module>` module and allows
    it to be run on multiple GPUs (distributed) in a data parallel setting.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the batch
    dimension. The module is replicated on each machine and each device, and
    each such replica handles a portion of the input. After the optimizer update,
    it synchronizes the parameters on the different nodes using SlowMo
    (https://arxiv.org/abs/1910.00643).

    Please make sure to read the documentation for slowmo_memory_efficient parameter as
    it contains a non-trivial trick in order to optimize our implementation.

    Please refer to the documentation of ``torch.nn.parallel.DistributedDataParallel``
    for other useful tips for using this container.

    Parameters:
        module (Module):
            module to be parallelized
        nprocs_per_node (int):
            Number of processes per node (one per GPU). This needs to be specified for optimal accuracy and speed.
            Syncing across GPUs in a node is extremely fast, which we utilize for performance optimization
        broadcast_buffers (bool):
            Flag that enables syncing (broadcasting) buffers (example - batchnorm buffers) of the module at beginning
            of the ``forward`` function. Setting it to False would result in better performance due to less
            communication on the network but might result in a reduced accuracy (default: ``True``)
        slowmo_base_algorithm (SlowMoBaseAlgorithm):
            The base algorithm to be used for approximately averaging the different parameters across nodes.  The base
            algorithm is responsible for increasing the efficiency of this module. The base algorithm, combined with
            SlowMo, results in significant speedups without accuracy loss. Either Stochastic Gradient Push
            (SlowMoBaseAlgorithm.SGP) (https://arxiv.org/abs/1811.10792) or LocalSGD (SlowMoBaseAlgorithm.LOCALSGD)
            (https://arxiv.org/abs/1808.07217) can be used here (default: SlowMoBaseAlgorithm.LOCALSGD)
    SlowMo Parameters:
        slowmo_momentum (float):
            This specifies the value of slowmo momentum to be used (read https://arxiv.org/abs/1910.00643 for more
            details). This parameter might need to be tuned and the optimal value varies according to the use case and
            the number of nodes being run on. The optimal value typically increases with the number of nodes. On
            training transfomers on the WMT 16 En-De dataset, we have found the optimal values to be 0 for less than 4
            nodes, 0.2 for 4 nodes, 0.5 for 8 nodes and 0.6 for 16 nodes (default: 0.5)
        slowmo_memory_efficient (bool):
            If enabled, use a memory efficient implementation of SlowMo. The basic implementation of SlowMo occupies
            extra memory equal to double the memory occupied by the model parameters. The memory efficient
            implementation shards that memory across a certain number of shards which is specified as a parameter
            below.
            In addition, slowmo_memory_efficient leads to extra communication with throughput equivalent to an
            allreduce, and performs an allreduce as a side-effect. In order to optimize the implementation, we skip
            the typical allreduce when slowmo_base_algorithm is localsgd and the localsgd step and slowmo step occur
            on the same iteration. Also, we skip the gossip step when slowmo_base_algorithm is sgp. We can skip these
            because the memory-efficient slowmo step does an allreduce as a side effect. Due to this skipping, when
            slowmo_base_algorithm is localsgd, we recommend setting slowmo_frequency to be a multiple of
            localsgd_frequency.
            We recommend setting this parameter to True when slowmo_base_algorithm is localsgd. In case of sgp, there
            is a tradeoff between extra memory usage which is double the memory occupied by the parameters, and extra
            time spent which is half the time taken up by an allreduce every slowmo_frequency iterations and we
            suggest setting it to False (default: True)
        slowmo_frequency (int):
            This specifies how often (number of iterations) slow momentum is to be performed. We recommend keeping
            slowmo_frequency as a multiple of localsgd_frequency. Please look at the documentation of
            slowmo_memory_efficient for the reasoning (default: 48)
        slowmo_lr (float):
            This specifies the value of slowmo learning rate to be used (read https://arxiv.org/abs/1910.00643 for
            more details). We do not recommend changing this (default: 1.0)
        slowmo_num_shards (int):
            The number of shards between which slow momentum parameters are distributed. This is only used when
            memory_efficient is set to True.
            The number of shards should scale with the number of parameters in the model. Increasing the number of
            shards decreases the memory used per node for storing the slow momentum parameters. However, if the shard
            size per node is too small, it results in a communication overhead (default: 32)
    LocalSGD Parameters:
        localsgd_frequency (int):
            LocalSGD typically averages the parameters once every few iterations. This parameter specifices the
            frequency of averaging.  We recommend keeping slowmo_frequency as a multiple of localsgd_frequency. Please
            look at the documentation of slowmo_memory_efficient for the reasoning (default: 3)
    SGP Parameters:
        graph (Optional[GraphManager):
            Graph to be used for gossip communication. This is used to specify the interaction graph between the
            different nodes (default: None)
        mixing (Optional[MixingManager]):
            Mixing manager to be used for gossip communication. This is used to specify weights given to outgoing and
            incoming messages (default: None)
        push_sum (bool):
            Whether to use PushSum or PushPull gossip (default: True)
        overlap (bool):
            Whether to use the overlap form of SGP. This feature is currently disabled until further testing is done
            for its use (default: False)
        synch_freq (int):
            How often (number of iterations) to synchronize for overlap SGP. A value of 0 means to synchronize overlap
            SGP every iteration (default: 0)
        use_streams (bool):
            Whether to use CUDA streams to speed up SGP overlap (default: True)
        slowmo_sgp_average_params (bool):
            Whether to completely average the parameters when slowmo is done instead of a partial averaging that
            happens every iteration (default: False)
    Debugging Parameters:
        verbose (bool):
            Prints various logs which are useful for debugging (default: False)
        profile_mode (bool):
            Prints the time taken by different parts of the code, which can help in finding bottlenecks (default: False)
    Parameters for Advanced Users:
        process_rank (Optional[int]):
            Rank of the current process in the process group (default: None)
        process_world_size (Optional[int]):
            Size of the process group (default: None)
        global_group (Optional[torch.distributed.ProcessGroup]):
            Global process group initialized by init_process_group (default: None)
        master_group (Optional[torch.distributed.ProcessGroup]):
            Process group which only contains the master GPUs of each node (default: None)
        local_node_group (Optional[torch.distributed.ProcessGroup]):
            Process group which only contains the GPUs local to the current node (default: None)
        comm_device: (Optional[torch.device]):
            The torch.device on which torch tensors are to be placed before communication (default: None)

    Example:
        >>> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
        >>> net = fairscale.data_parallel.SlowMoDistributedDataParallel(model, nprocs_per_node=8)
        >>> loss = criterion(net(inputs), targets)
        >>> loss.backward()
        >>> optimizer.step()
        >>> net.perform_slowmo(optimizer)
    """

    def __init__(self, module: 'torch.nn.Module', nprocs_per_node: 'int', broadcast_buffers: 'bool'=True, slowmo_base_algorithm: 'SlowMoBaseAlgorithm'=SlowMoBaseAlgorithm.LOCALSGD, slowmo_momentum: 'float'=0.5, slowmo_memory_efficient: 'bool'=True, slowmo_frequency: 'int'=48, slowmo_lr: 'float'=1.0, slowmo_num_shards: 'int'=32, localsgd_frequency: 'int'=3, graph: 'Optional[GraphManager]'=None, mixing: 'Optional[MixingManager]'=None, push_sum: 'bool'=True, overlap: 'bool'=False, synch_freq: 'int'=0, use_streams: 'bool'=True, slowmo_sgp_average_params: 'bool'=False, verbose: 'bool'=False, profile_mode: 'bool'=False, process_rank: 'Optional[int]'=None, process_world_size: 'Optional[int]'=None, global_group: 'Optional[torch.distributed.ProcessGroup]'=None, master_group: 'Optional[torch.distributed.ProcessGroup]'=None, local_node_group: 'Optional[torch.distributed.ProcessGroup]'=None, comm_device: 'Optional[torch.device]'=None) ->None:
        super(SlowMoDistributedDataParallel, self).__init__()
        assert os.environ.get('NCCL_BLOCKING_WAIT', '0') == '0'
        assert nprocs_per_node >= 1
        self.nprocs_per_node = nprocs_per_node
        if process_world_size is None or process_rank is None:
            assert dist.is_initialized()
            process_rank = dist.get_rank()
            process_world_size = dist.get_world_size()
        assert process_world_size is not None and process_rank is not None
        self.process_rank = process_rank
        self.process_world_size = process_world_size
        self._initialize_logger(verbose, self.process_rank)
        logical_rank, logical_world_size = self._maybe_create_process_groups(self.process_rank, self.process_world_size, nprocs_per_node, global_group, master_group, local_node_group)
        self.logical_rank = logical_rank
        self.logical_world_size = logical_world_size
        self.module = module
        self.broadcast_buffers = broadcast_buffers
        first_param_dtype = next(self.module.parameters()).dtype
        self.broadcast_bucket_size = BROADCAST_BUCKET_SIZE
        self.module_buffers = list(self.module.buffers())
        if comm_device is None:
            cpu_comm = dist.get_backend() == 'gloo'
            comm_device = torch.device('cpu') if cpu_comm else torch.device('cuda')
        self._cpu_comm = comm_device.type == 'cpu'
        self.dist_config = {'verbose': verbose, 'comm_device': comm_device, 'logical_rank': logical_rank, 'process_rank': self.process_rank, 'logical_world_size': logical_world_size, 'cpu_comm': self._cpu_comm}
        self.profile_mode = profile_mode
        self.num_updates = 0
        self.portion_start: 'Optional[int]' = None
        self.slowmo = slowmo_lr != 1 or slowmo_momentum != 0
        self.slowmo_lr = slowmo_lr if self.slowmo else 1
        self.slowmo_momentum = slowmo_momentum if self.slowmo else 0
        self.slowmo_frequency = slowmo_frequency
        self.slowmo_sgp_average_params = slowmo_sgp_average_params
        self.localsgd = slowmo_base_algorithm == SlowMoBaseAlgorithm.LOCALSGD
        self.sgp = slowmo_base_algorithm == SlowMoBaseAlgorithm.SGP
        self.localsgd_frequency = localsgd_frequency
        self.ef1: 'Optional[List[torch.Tensor]]' = None
        self.global_momentum_buffers_initialized = False
        if self.master_group is None:
            assert self.localsgd or self.sgp
            self.localsgd = self.sgp = False
            self.logger.warning('Disabling LocalSGD and SGP since a local allreduce will suffice')
        if self.slowmo and not self.localsgd and not self.sgp:
            self.logger.warning('SlowMo is being used without LocalSGD and SGP')
        self.slowmo_memory_efficient = slowmo_memory_efficient
        self.slowmo_num_shards = min(self.process_world_size, slowmo_num_shards) if self.slowmo_memory_efficient else 1
        self.is_current_node_a_slowmo_shard = self.process_rank < self.slowmo_num_shards if self.slowmo_memory_efficient else True
        self.nprocs_per_node_device = torch.tensor([self.nprocs_per_node], device=comm_device, dtype=first_param_dtype)
        if self.sgp:
            self._sgp_init(module=module, first_param_dtype=first_param_dtype, logical_rank=logical_rank, logical_world_size=logical_world_size, comm_device=comm_device, graph=graph, mixing=mixing, push_sum=push_sum, overlap=overlap, synch_freq=synch_freq, use_streams=use_streams, slowmo_sgp_average_params=slowmo_sgp_average_params)
        self._register_hooks()
        self.logger.debug('Initialization of SlowMoDistributedDataParallel complete')

    def _initialize_logger(self, verbose: 'bool', process_rank: 'int') ->None:
        """Initializes the logger"""
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger = cast(logging.Logger, MultiProcessAdapter(self.logger, {'process_num': process_rank}))

    def _maybe_create_process_groups(self, process_rank: 'int', process_world_size: 'int', nprocs_per_node: 'int', global_group: 'Optional[torch.distributed.ProcessGroup]', master_group: 'Optional[torch.distributed.ProcessGroup]', local_node_group: 'Optional[torch.distributed.ProcessGroup]') ->Tuple[int, int]:
        """Creates the process groups required for the SlowMo implementation"""
        self.local_rank = process_rank % self.nprocs_per_node
        assert process_world_size % self.nprocs_per_node == 0
        logical_world_size = process_world_size // self.nprocs_per_node
        logical_rank = process_rank // self.nprocs_per_node
        self._maybe_initialize_global_group(global_group, process_world_size)
        self._maybe_initialize_local_node_group(local_node_group, process_rank, logical_world_size)
        self._maybe_initialize_master_group(master_group, process_rank, process_world_size, nprocs_per_node)
        self.logger.debug('Initialization of all process groups complete')
        return logical_rank, logical_world_size

    def _maybe_initialize_global_group(self, global_group: 'Optional[torch.distributed.ProcessGroup]', process_world_size: 'int') ->None:
        if global_group is None:
            all_processes = list(range(process_world_size))
            self.global_group = create_process_group(all_processes)
            self.logger.debug('Initialization of global group complete')
        else:
            self.global_group = global_group
        self.logger.debug('Global group set')
        self.process_group = self.global_group

    def _maybe_initialize_master_group(self, master_group: 'Optional[torch.distributed.ProcessGroup]', process_rank: 'int', process_world_size: 'int', nprocs_per_node: 'int') ->None:
        if master_group is not None:
            self.master_group: 'Optional[torch.distributed.ProcessGroup]' = master_group
            return
        if self.nprocs_per_node > 1:
            self.logger.debug('Initializing master process group')
            master_nodes = [i for i in range(process_world_size) if i % nprocs_per_node == 0]
            self.master_group = create_process_group(master_nodes) if len(master_nodes) > 1 else None
            if self.master_group is not None and process_rank in master_nodes:
                self.logger.debug('Initialization of master group complete')
        else:
            self.master_group = self.global_group

    def _maybe_initialize_local_node_group(self, local_node_group: 'Optional[torch.distributed.ProcessGroup]', process_rank: 'int', logical_world_size: 'int') ->None:
        if self.nprocs_per_node == 1:
            self.local_node_group = None
            return
        if local_node_group is not None:
            self.local_node_group = local_node_group
            return
        self.logger.debug('Initializing local process groups')
        for node in range(logical_world_size):
            node_processes_ranks = list(range(node * self.nprocs_per_node, (node + 1) * self.nprocs_per_node))
            new_local_group = create_process_group(node_processes_ranks)
            if process_rank in node_processes_ranks:
                self.local_node_group = new_local_group
        assert self.local_node_group is not None
        self.logger.debug('Initialization of local groups complete')

    def forward(self, *inputs: Any, **kwargs: Any) ->Union[torch.Tensor, List[torch.Tensor]]:
        """Forward pass performed in parallel across all devices on node"""
        return self.module(*inputs, **kwargs)

    def _sync_params(self) ->None:
        """Synchronize parameters across devices (intra-node)"""
        if self.local_node_group is None:
            return
        params = cast(List[torch.Tensor], list(self.module.parameters()))
        communication_op = functools.partial(dist.broadcast, src=self.logical_rank * self.nprocs_per_node, group=self.local_node_group)
        communicate(params, communication_op)
        self.logger.debug('Intra-node param sync complete')

    def _sync_buffers(self) ->None:
        """Synchronize buffers across nodes"""
        if self.broadcast_buffers and len(self.module_buffers) > 0:
            self._distributed_broadcast_coalesced(self.process_group, self.module_buffers, self.broadcast_bucket_size)
        self.logger.debug('Intra-node buffer sync complete')

    def _distributed_broadcast_coalesced(self, process_group: 'torch.distributed.ProcessGroup', tensors: 'List[torch.Tensor]', buffer_size: 'int') ->None:
        dist._broadcast_coalesced(process_group, tensors, buffer_size)

    def _create_event_recorder(self, event_name: 'str') ->EventRecorder:
        """Creates an cuda event recorder which helps in profiling"""
        return create_event_recorder(event_name, dummy=not self.profile_mode)

    def _fp16_fp32_iterator(self, optimizer: 'torch.optim.Optimizer', fp32_params: 'Optional[torch.Tensor]') ->Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterator for those fp16 parameters which have a fp32 copy"""
        if hasattr(optimizer, '_amp_stash') and hasattr(optimizer._amp_stash, 'fp16_groups'):
            for p_fp16_group, p_fp32_group in zip(optimizer._amp_stash.fp16_groups, optimizer._amp_stash.fp32_from_fp16_groups):
                for p_fp16, p_fp32 in zip(p_fp16_group, p_fp32_group):
                    yield p_fp16, p_fp32
        elif fp32_params is not None:
            if isinstance(fp32_params, dict):
                fp32_params_list = list(fp32_params.values())
                assert len(fp32_params_list) == 1
                fp32_params = fp32_params_list[0]
            if isinstance(fp32_params, list):
                for p, fp32_param in zip(self.parameters(), fp32_params):
                    yield p.view(-1), fp32_param
            else:
                offset = 0
                for p in self.parameters():
                    yield p.view(-1), fp32_params[offset:offset + p.numel()]
                    offset += p.numel()

    def _should_perform_slowmo(self) ->bool:
        return self.slowmo and (self.num_updates + 1) % self.slowmo_frequency == 0

    def _should_perform_localsgd(self) ->bool:
        return self.localsgd and (self.num_updates + 1) % self.localsgd_frequency == 0

    def _skip_averaging_memory_efficient_slowmo(self) ->bool:
        return self.slowmo_memory_efficient and self._should_perform_slowmo()

    def _should_perform_sgp_common(self) ->bool:
        return self.sgp and not self.overlap and not self._skip_averaging_memory_efficient_slowmo()

    def _should_perform_sgp(self) ->bool:
        return self._should_perform_sgp_common() and not self.overlap

    def _should_perform_sgp_overlap(self) ->bool:
        return self._should_perform_sgp_common() and self.overlap

    def _should_use_error_feedback(self, fp16_fp32_list: 'List[Tuple[torch.Tensor, torch.Tensor]]') ->bool:
        return bool(fp16_fp32_list) and (self._should_perform_sgp() or self._should_allreduce_params())

    def _should_allreduce_params(self) ->bool:
        return self.sgp and self._should_perform_slowmo() and self.slowmo_sgp_average_params or self._should_perform_localsgd() and not self._skip_averaging_memory_efficient_slowmo()

    def _maybe_pre_communicate_error_feedback(self, fp16_fp32_list: 'List[Tuple[torch.Tensor, torch.Tensor]]') ->None:
        ef_rec = self._create_event_recorder('Error feedback')
        if self._should_use_error_feedback(fp16_fp32_list):
            with torch.no_grad():
                for p_fp16, p_fp32 in fp16_fp32_list:
                    if self._should_allreduce_params():
                        p_fp16.div_(self.logical_world_size)
                        p_fp16.mul_(self.logical_world_size)
                    p_fp32 -= p_fp16.float()
                if self.ef1 is not None:
                    for idx, (_, p_fp32) in enumerate(fp16_fp32_list):
                        p_fp32 += self.ef1[idx]
                        p_fp32.div_(2)
        ef_rec.stop()
        self.logger.debug('Error feedback completed')

    def _maybe_post_communicate_error_feedback(self, fp16_fp32_list: 'List[Tuple[torch.Tensor, torch.Tensor]]') ->None:
        ef_unroll_rec = self._create_event_recorder('Sync and error feedback unroll rec')
        if self._should_use_error_feedback(fp16_fp32_list):
            with torch.no_grad():
                for p, p_fp32 in fp16_fp32_list:
                    p_fp32 += p.float()
        ef_unroll_rec.stop()
        self.logger.debug('Error feedback unroll completed')

    def _maybe_perform_sgp(self) ->None:
        sgp_rec = self._create_event_recorder('SGP')
        if self._should_perform_sgp():
            if not self._should_allreduce_params():
                self._sgp_transfer_params()
                self._sgp_query_gossip_queue()
                torch.cuda.synchronize()
            self.logger.debug('SGP completed')
        sgp_rec.stop()

    def _maybe_allreduce(self) ->None:
        localsgd_rec = self._create_event_recorder('Localsgd communication time')
        if self._should_allreduce_params():
            communication_op = functools.partial(dist.all_reduce, group=self.master_group)
            params = cast(List[torch.Tensor], list(self.parameters()))
            with torch.no_grad():
                for p in params:
                    p.div_(self.logical_world_size)
            self.logger.debug('Params normalized before localsgd step')
            communicate(params, communication_op, self.logger)
            torch.cuda.synchronize()
            self.logger.debug('Allreduce completed')
        localsgd_rec.stop()

    def _maybe_sync_locally(self) ->None:
        if self._should_perform_sgp() or self._should_allreduce_params():
            self._sync_params()
            torch.cuda.synchronize()

    def _maybe_perform_slowmo(self, optimizer: 'torch.optim.Optimizer') ->None:
        slowmo_rec = self._create_event_recorder('Slowmo')
        if self._should_perform_slowmo():
            self._global_momentum_step(optimizer)
        slowmo_rec.stop()
        self.logger.debug('Global momentum step completed')

    def _maybe_copy_back_fp32_parameters(self, fp16_fp32_list: 'List[Tuple[torch.Tensor, torch.Tensor]]') ->None:
        ef_copy_rec = self._create_event_recorder('Error feedback copy back')
        if (self._should_perform_sgp() or self._should_allreduce_params() or self._should_perform_slowmo()) and fp16_fp32_list:
            with torch.no_grad():
                for idx, (p_fp16, p_fp32) in enumerate(fp16_fp32_list):
                    p_fp16.copy_(p_fp32)
        ef_copy_rec.stop()
        self.logger.debug('Error feedback copy-back completed')

    def _maybe_sgp_overlap_pre_communicate_error_feedback(self, fp16_fp32_list: 'List[Tuple[torch.Tensor, torch.Tensor]]') ->None:
        if self._should_perform_sgp_overlap() and fp16_fp32_list:
            if self.ef1 is None:
                self.ef1 = [p_fp32.clone().detach_() for _, p_fp32 in fp16_fp32_list]
            with torch.no_grad():
                assert self.ef1 is not None
                for ef1, (p_fp16, p_fp32) in zip(self.ef1, fp16_fp32_list):
                    ef1.copy_(p_fp32 - p_fp16.float())

    def perform_slowmo(self, optimizer: 'torch.optim.Optimizer', fp32_params: 'Optional[torch.Tensor]'=None) ->None:
        """This is to be called after optimizer.step(). It performs the approximate averaging using
        the base algorithm (SGP/ LocalSGD) and the slow momentum step. Since LocalSGD and the slow
        momentum step are not performed every iteration, it only performs those when needed.

        It is recommended to call ``model.zero_grad(set_to_none=True)`` just before calling this function. This
        is because ``model.zero_grad(set_to_none=True)`` frees up the memory occupied by the gradients, some of which
        may be reused by this function.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer being used for training the model
            fp32_params (Optional[torch.Tensor]): To be used when performing fp16 training. Needs to be
                        set to the fp16 copy of the parameters (default: None)
        """
        if not self.global_momentum_buffers_initialized:
            self._init_global_momentum_buffers(optimizer)
        fp16_fp32_list = list(self._fp16_fp32_iterator(optimizer, fp32_params))
        self.logger.debug('Created a list of fp16 and fp32 corresponding parameters')
        self.logger.debug('Booleans set. Values - self._should_perform_slowmo()=%r, self._should_perform_localsgd()=%r, self._should_allreduce_params()=%r', self._should_perform_slowmo(), self._should_perform_localsgd(), self._should_allreduce_params())
        self.logger.debug('Step number(0-indexed)=%d', self.num_updates)
        if self.num_updates == 0 and fp32_params is None and not hasattr(optimizer, '_amp_stash') and any(p.dtype == torch.float16 for p in self.parameters()):
            self.logger.warning('WARNING: please set fp32_params in perform_slowmo() in order to avoid accuracy loss')
        self._maybe_pre_communicate_error_feedback(fp16_fp32_list)
        self._maybe_perform_sgp()
        self._maybe_allreduce()
        self._maybe_sync_locally()
        self._maybe_post_communicate_error_feedback(fp16_fp32_list)
        self._maybe_perform_slowmo(optimizer)
        self._maybe_copy_back_fp32_parameters(fp16_fp32_list)
        self._maybe_sgp_overlap_pre_communicate_error_feedback(fp16_fp32_list)
        self.num_updates += 1

    def _init_global_momentum_buffers(self, optimizer: 'torch.optim.Optimizer') ->None:
        """Initializes the slow momentum buffers"""
        self.global_momentum_buffers_initialized = True
        if not self.slowmo:
            return
        total_elements = 0
        params_dtype = None
        for group in optimizer.param_groups:
            for p in group['params']:
                total_elements += p.numel()
                if params_dtype is None:
                    params_dtype, params_device = p.dtype, p.device
                assert p.dtype == params_dtype == torch.float32
                assert p.device == params_device
        self.world_portion_length = (total_elements + self.slowmo_num_shards - 1) // self.slowmo_num_shards
        if not self.is_current_node_a_slowmo_shard:
            return
        self.portion_start = self.process_rank * self.world_portion_length if self.slowmo_memory_efficient else 0
        self.portion_end = min((self.process_rank + 1) * self.world_portion_length, total_elements) if self.slowmo_memory_efficient else total_elements
        self.old_params = torch.empty(self.world_portion_length, dtype=params_dtype).detach()
        offset = 0
        for group in optimizer.param_groups:
            for p in group['params']:
                numel = p.numel()
                if offset + numel > self.portion_start and offset < self.portion_end:
                    overall_start = max(self.portion_start, offset)
                    overall_end = min(self.portion_end, offset + numel)
                    p_start = overall_start - offset
                    p_end = overall_end - offset
                    buffer_start = overall_start - self.portion_start
                    buffer_end = overall_end - self.portion_start
                    current_p = p.view(-1)[p_start:p_end]
                    current_p_old = self.old_params[buffer_start:buffer_end]
                    current_p_old.copy_(current_p)
                offset += numel
        self.global_momentum_buffer = torch.zeros_like(self.old_params).detach()

    def _distributed_comm(self, optimizer: 'torch.optim.Optimizer', mode: 'str') ->None:
        """Performs the communication needed for the efficient SlowMo implementation"""
        offset = 0
        slowmo_comm_lists: 'List[List[torch.Tensor]]' = [[] for _ in range(self.slowmo_num_shards)]
        with torch.no_grad():
            for group in optimizer.param_groups:
                for p in group['params']:
                    numel = p.numel()
                    if mode == 'gather':
                        p /= self.process_world_size
                    current_start = offset
                    while current_start < offset + numel:
                        main_node = current_start // self.world_portion_length
                        main_node_end = (main_node + 1) * self.world_portion_length
                        current_end = min(offset + numel, main_node_end)
                        p_start = current_start - offset
                        p_end = current_end - offset
                        slowmo_comm_lists[main_node].append(p.view(-1)[p_start:p_end])
                        current_start = current_end
                    offset += numel
            for slowmo_rank, slowmo_comm_list in enumerate(slowmo_comm_lists):
                if mode == 'gather':
                    communication_op = functools.partial(dist.reduce, dst=slowmo_rank)
                elif mode == 'scatter':
                    communication_op = functools.partial(dist.broadcast, src=slowmo_rank)
                communicate(slowmo_comm_list, communication_op)

    def _global_momentum_step(self, optimizer: 'torch.optim.Optimizer') ->None:
        """Performs the slow momentum step"""
        if not self.slowmo:
            return
        if not self.global_momentum_buffers_initialized:
            self._init_global_momentum_buffers(optimizer)
        if self.slowmo_memory_efficient:
            self._distributed_comm(optimizer, mode='gather')
        if self.is_current_node_a_slowmo_shard:
            self._perform_local_optimization(optimizer)
        if self.slowmo_memory_efficient:
            self._distributed_comm(optimizer, mode='scatter')

    def _perform_local_optimization(self, optimizer: 'torch.optim.Optimizer') ->None:
        """Performs the slow momentum on the local shard"""
        assert self.portion_start is not None
        with torch.no_grad():
            offset = 0
            for group in optimizer.param_groups:
                for p in group['params']:
                    numel = p.numel()
                    if offset + numel > self.portion_start and offset < self.portion_end:
                        overall_start = max(self.portion_start, offset)
                        overall_end = min(self.portion_end, offset + numel)
                        p_start = overall_start - offset
                        p_end = overall_end - offset
                        buffer_start = overall_start - self.portion_start
                        buffer_end = overall_end - self.portion_start
                        current_p = p.view(-1)[p_start:p_end]
                        current_p_gmb = self.global_momentum_buffer[buffer_start:buffer_end]
                        current_p_old = self.old_params[buffer_start:buffer_end]
                        current_p_gmb.mul_(self.slowmo_momentum).sub_(current_p, alpha=1 / group['lr']).add_(current_p_old, alpha=1 / group['lr'])
                        current_p_old.add_(current_p_gmb, alpha=-group['lr'] * self.slowmo_lr)
                        current_p.copy_(current_p_old)
                    offset += numel

    def _register_hooks(self) ->None:
        """
        Registers push-sum de-bias/bias hooks in pre-forward/post-backward
        passes in all leaf modules
        """
        self.register_forward_pre_hook(self.__make_forward_pre_hook())
        self.register_backward_hook(self.__make_backward_hook())

    def __make_backward_hook(self) ->Callable[..., None]:
        self.logger.debug('making backward hook')

        def hook(*unused: Any) ->None:
            if self.local_node_group is not None:
                grads = []
                for p in self.module.parameters():
                    if not p.requires_grad or p.grad is None:
                        continue
                    p.grad.div_(self.nprocs_per_node)
                    grads.append(p.grad)
                self.logger.debug('Gradients ready for syncing')
                communication_op = functools.partial(dist.all_reduce, group=self.local_node_group)
                communicate(grads, communication_op, self.logger)
                self.logger.debug('Gradient sync during backward pass in local_group complete')
            if self.sgp:
                self._sgp_ps_numerator()
                if self.gossip_enable and self.overlap and not self._skip_averaging_memory_efficient_slowmo():
                    self._sgp_query_gossip_queue()

        def queue_hook(*unused: Any) ->None:
            Variable._execution_engine.queue_callback(hook)
        return queue_hook

    def __make_forward_pre_hook(self) ->Callable[..., None]:
        self.logger.debug('making forward pre-hook')

        def hook(*unused: Any) ->None:
            """Query gossip queue and de-bias during forward pass"""
            self._sync_buffers()
            if self.sgp:
                if self.gossip_enable and self.overlap and not self._skip_averaging_memory_efficient_slowmo():
                    self._sgp_transfer_params()
                self._sgp_unbias()
        return hook

    def _sgp_init(self, module: 'torch.nn.Module', first_param_dtype: 'torch.dtype', logical_rank: 'int', logical_world_size: 'int', comm_device: 'Optional[torch.device]'=None, graph: 'Optional[GraphManager]'=None, mixing: 'Optional[MixingManager]'=None, push_sum: 'bool'=True, overlap: 'bool'=False, synch_freq: 'int'=0, use_streams: 'bool'=True, slowmo_sgp_average_params: 'bool'=False) ->None:
        """Perform initialization for Stochastic Gradient Push base algorithm"""
        if graph is None:
            graph = NPDDEGraph(logical_rank, logical_world_size, self.nprocs_per_node, self.local_rank)
        if mixing is None:
            mixing = UniformMixing(graph, comm_device)
        self.dist_config.update({'graph': graph, 'mixing': mixing, 'push_sum': push_sum})
        self.overlap = overlap
        assert not self.overlap
        self.synch_freq = synch_freq
        self.asynch = synch_freq > 0
        self.ps_weight = torch.ones(1, device=comm_device, dtype=first_param_dtype)
        self.is_sgp_ps_numerator = False
        self.gossip_enable = True
        self.gossiping = False
        self.params_mixed = True
        self.gossip_ps_factor = torch.zeros(1, device=comm_device, dtype=first_param_dtype)
        self.gossip_ps_weight = self.ps_weight.clone()
        self.gossip_params = []
        self.gossip_device_buffer = []
        for p in module.parameters():
            cp = cast(torch.nn.Parameter, p.clone().detach_())
            cp = cast(torch.nn.Parameter, cp.cpu().pin_memory() if self._cpu_comm else cp)
            self.gossip_params.append(cp)
            self.gossip_device_buffer.append(cp)
        self.gossip_lock = threading.Lock()
        self.gossip_flag = threading.Event()
        self.train_flag = threading.Event()
        if cast(torch.device, self.dist_config['comm_device']).type != 'cpu' and use_streams:
            self.gossip_stream = torch.Stream()
        else:
            self.gossip_stream = torch.cuda.current_stream()
        if self.process_rank % self.nprocs_per_node == 0:
            self.gossip_thread = threading.Thread(target=SlowMoDistributedDataParallel._sgp_gossip_target, args=(self.dist_config, self.gossip_flag, self.train_flag, self.gossip_lock, self.gossip_params, self.gossip_device_buffer, self.gossip_ps_weight, self.gossip_ps_factor, self.gossip_stream))
            self.gossip_thread.daemon = True
            self.gossip_thread.name = 'Gossip-Thread'
            self.gossip_thread.start()
        else:
            self.gossip_flag.set()
        self.gossip_flag.wait()
        self.gossip_flag.clear()
        self.lazy_mixing = not self.asynch and cast(MixingManager, self.dist_config['mixing']).is_regular()
        self.lazy_ps_factor = self.gossip_ps_factor.clone()
        self.logger.debug('lazy mixing: %r', self.lazy_mixing)

    def state_dict(self) ->Dict[str, Union[torch.Tensor, bool]]:
        state_dict = super(SlowMoDistributedDataParallel, self).state_dict()
        if self.sgp:
            state_dict['ps_weight'] = self.ps_weight.cpu()
            state_dict['is_sgp_ps_numerator'] = self.is_sgp_ps_numerator
        return state_dict

    def load_state_dict(self, state_dict: 'Dict[str, Union[torch.Tensor, bool]]') ->None:
        if self.sgp:
            assert isinstance(state_dict, dict)
            self.ps_weight = cast(torch.Tensor, state_dict.pop('ps_weight'))
            self.is_sgp_ps_numerator = cast(bool, state_dict.pop('is_sgp_ps_numerator'))
        super(SlowMoDistributedDataParallel, self).load_state_dict(cast(Dict[str, torch.Tensor], state_dict))

    def _sgp_ps_numerator(self) ->None:
        """Convert model params to ps-numerator"""
        if not self.is_sgp_ps_numerator:
            if not self.lazy_mixing:
                ps_weight = self.ps_weight
                with torch.no_grad():
                    for p in self.module.parameters():
                        p.mul_(cast(torch.Tensor, ps_weight.type(p.dtype)))
            self.is_sgp_ps_numerator = True

    def _sgp_unbias(self) ->None:
        """Convert model params to de-biased estimate"""
        if self.is_sgp_ps_numerator:
            if not self.lazy_mixing:
                ps_weight = self.ps_weight
                with torch.no_grad():
                    for p in self.module.parameters():
                        p.div_(cast(torch.Tensor, ps_weight.type(p.dtype)))
            self.is_sgp_ps_numerator = False

    def train(self, mode: 'bool'=True) ->'SlowMoDistributedDataParallel':
        super(SlowMoDistributedDataParallel, self).train(mode)
        if self.sgp:
            self.gossip_enable = True
        return self

    def eval(self) ->'SlowMoDistributedDataParallel':
        super(SlowMoDistributedDataParallel, self).eval()
        if self.sgp:
            self.gossip_enable = False
            self._sgp_query_gossip_queue(non_blocking=self.asynch)
        return self

    def _sgp_query_gossip_queue(self, non_blocking: 'bool'=False) ->bool:
        """Check gossip-queue for push-sum residuals and update model"""
        if not self.gossip_enable:
            return False
        self.logger.debug('querying gossip queue')
        if not self.gossiping:
            if self.process_rank % self.nprocs_per_node == 0:
                self.logger.warning('not gossiping right now')
            return False
        if not non_blocking and not self.gossip_flag.wait(timeout=HEARTBEAT_TIMEOUT):
            raise RuntimeError('Gossip flag timeout')
            sys.exit()
        if self.gossip_flag.is_set():
            self.logger.debug('received gossip flag')
            if self.gossip_ps_weight[0] == -1:
                self.gossip_flag.clear()
                self.params_mixed = True
                self.gossiping = False
                self._sgp_transfer_params(mix=False)
                return False
            self.lazy_ps_factor.copy_(self.gossip_ps_factor)
            self._sgp_ps_numerator()
            self.ps_weight += self.gossip_ps_weight
            if self.lazy_mixing:
                self.ps_weight *= self.lazy_ps_factor
            with torch.no_grad():
                for p, r in zip(self.module.parameters(), self.gossip_device_buffer):
                    p.add_(r)
                    if self.lazy_mixing:
                        p.mul_(cast(torch.Tensor, self.lazy_ps_factor.type(p.dtype)))
            self.logger.debug('updated ps-weight %f', self.ps_weight)
            self.logger.debug('updated model params')
            self.gossip_flag.clear()
            self.params_mixed = True
            self.gossiping = False
            return True
        return False

    def _sgp_transfer_params(self, mix: 'bool'=True) ->bool:
        """Transfers COPY of model parameters to gossip queue"""
        if not self.gossip_enable or self.process_rank % self.nprocs_per_node != 0:
            return False
        self.logger.debug('transferring model params')
        if not self.params_mixed:
            self.logger.warning('params not mixed')
            return False
        mix = mix and not self.lazy_mixing
        self._sgp_ps_numerator()
        if mix:
            self.ps_weight *= self.gossip_ps_factor
        self.gossip_ps_weight.copy_(self.ps_weight)
        with torch.no_grad():
            for p, gossip_device_buffer_elem in zip(self.module.parameters(), self.gossip_device_buffer):
                if mix:
                    p.mul_(cast(torch.Tensor, self.gossip_ps_factor.type(p.dtype)))
                gossip_device_buffer_elem.copy_(p)
        self.gossip_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.gossip_stream):
            for b, gp in zip(self.gossip_device_buffer, self.gossip_params):
                gp.copy_(b, non_blocking=True)
        self.logger.debug('transferred model params')
        self.params_mixed = False
        self.gossiping = True
        self.train_flag.set()
        return True

    @staticmethod
    def _sgp_gossip_into_receive_buffer(send_buffer: 'List[torch.Tensor]', gossiper: 'Gossiper', receive_buffer: 'List[torch.Tensor]', gossip_ps_weight: 'torch.Tensor', gossip_lock: 'threading.Lock', dist_config: 'Dict[Any, Any]') ->Tuple[torch.Tensor, torch.Tensor]:
        out_msg = flatten_tensors(send_buffer)
        with gossip_lock:
            in_msg, ps_weight = gossiper.mix(out_msg, gossip_ps_weight)
            ps_factor = gossiper.mixing_weights['lo']
        with torch.no_grad():
            for r, g in zip(unflatten_tensors(in_msg, send_buffer), receive_buffer):
                if dist_config['cpu_comm']:
                    g.copy_(r, non_blocking=True)
                else:
                    g.copy_(r)
        return ps_weight, ps_factor

    @staticmethod
    def _sgp_gossip_target(dist_config: 'Dict[Any, Any]', gossip_flag: 'threading.Event', train_flag: 'threading.Event', gossip_lock: 'threading.Lock', gossip_params: 'List[torch.Tensor]', gossip_device_buffer: 'List[torch.Tensor]', gossip_ps_weight: 'torch.Tensor', gossip_ps_factor: 'torch.Tensor', gossip_stream: 'torch.cuda.Stream') ->None:
        """Gossip thread, which performs push-sum on model params"""
        logger = make_logger(dist_config['logical_rank'], dist_config['verbose'])
        gossip_params_by_dtype = group_by_dtype(gossip_params)
        gossip_device_buffer_by_dtype = group_by_dtype(gossip_device_buffer)
        gossipers = {}
        gossiper_class = PushSum if dist_config['push_sum'] else PushPull
        for dtype in gossip_params_by_dtype:
            gossipers[dtype] = gossiper_class(flatten_tensors(gossip_params_by_dtype[dtype]), device=cast(torch.device, dist_config['comm_device']), graph=cast(GraphManager, dist_config['graph']), mixing=cast(MixingManager, dist_config['mixing']), rank=dist_config['process_rank'], world_size=dist_config['logical_world_size'], logger=logger)
        dist_config['gossipers'] = gossipers
        gossip_ps_factor.copy_(gossipers[list(gossipers)[0]].mixing_weights['lo'])
        gossip_flag.set()
        while True:
            train_flag.wait()
            logger.debug('received train-flag')
            try:
                with torch.cuda.stream(gossip_stream):
                    for dtype in gossip_params_by_dtype:
                        ps_weight, ps_factor = SlowMoDistributedDataParallel._sgp_gossip_into_receive_buffer(gossip_params_by_dtype[dtype], gossipers[dtype], gossip_device_buffer_by_dtype[dtype], gossip_ps_weight, gossip_lock, dist_config)
                    gossip_ps_weight.copy_(ps_weight)
                    gossip_ps_factor.copy_(ps_factor)
            except RuntimeError as e:
                logger.warning('received runtime error {}'.format(e))
                for gossiper in gossipers.values():
                    gossiper.clean_msg_buffers_()
                gossip_ps_weight.fill_(-1)
            finally:
                gossip_stream.synchronize()
                train_flag.clear()
                gossip_flag.set()


class MultiInputSequential(nn.Module):
    """A variation of nn.Sequential, that allows the first module in the sequence accepts
    multiple inputs. To be used internally by _split_module
    """

    def __init__(self, *modules: nn.Module) ->None:
        super().__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, *inputs: Tuple[Tensor]) ->Tensor:
        input = self.modules_list[0](*inputs)
        for module in self.modules_list[1:]:
            input = module(input)
        return input


ConsumerType = TypeVar('ConsumerType')


def RemoteSequential(rref_list: 'List[rpc.RRef]') ->MultiInputSequential:
    return MultiInputSequential(*(r.local_value() for r in rref_list))


Tensors = Tuple[Tensor, ...]


TensorOrTensors = Union[Tensor, Tensors]


class Batch:
    """An abstraction of an atomic tensor or a tuple of tensors. This
    eliminates every boilerplate code to classify an atomic tensor or a tuple
    of tensors.
    ::

        x = generate_tensor_or_tensors()
        x = Batch(x)

        # in-place update
        x[0] = F.apply(x[0])
        x[:] = F.apply(*x)

        # f(x) if x is a tensor.
        # f(*x) if x is a tuple of tensors.
        # y is also a batch.
        y = x.call(f)

    """

    def __init__(self, value: 'TensorOrTensors', index: 'int') ->None:
        self.value = value
        self.atomic = torch.is_tensor(value)
        self.__index = index

    @property
    def index(self) ->int:
        return self.__index

    @property
    def tensor(self) ->Tensor:
        """Retrieves the underlying tensor."""
        if not self.atomic:
            raise AttributeError('not atomic batch')
        return cast(Tensor, self.value)

    @property
    def tensors(self) ->Tensors:
        """Retrieves the underlying tensors."""
        if self.atomic:
            raise AttributeError('batch is atomic')
        return cast(Tensors, self.value)

    @property
    def tensor_or_tensors(self) ->TensorOrTensors:
        """Retrieves the underlying tensor or tensors regardless of type."""
        return self.value

    def call(self, function: 'Function') ->'Batch':
        """Calls a function by the underlying tensor or tensors. It also wraps
        the output with :class:`Batch`.
        """
        return Batch(function(self.value), self.index)

    def __repr__(self) ->str:
        return f'Batch[atomic={self.atomic!r}]({self.value!r})'

    def __iter__(self) ->Iterator[Tensor]:
        if self.atomic:
            yield self.tensor
        else:
            yield from self.tensors

    def __len__(self) ->int:
        return 1 if self.atomic else len(self.tensors)

    def __getitem__(self, index: 'int') ->Tensor:
        if not self.atomic:
            return self.tensors[index]
        if index != 0:
            raise IndexError('atomic batch allows index 0 only')
        return self.tensor

    @typing.overload
    def __setitem__(self, index: 'int', value: 'Tensor') ->None:
        ...

    @typing.overload
    def __setitem__(self, index: 'slice', value: 'Tensors') ->None:
        ...

    def __setitem__(self, index: 'Union[int, slice]', value: 'TensorOrTensors') ->None:
        if isinstance(index, int):
            value = cast(Tensor, value)
            self._setitem_by_index(index, value)
        else:
            value = cast(Tensors, value)
            self._setitem_by_slice(index, value)

    def _setitem_by_index(self, index: 'int', value: 'Tensor') ->None:
        if not self.atomic:
            i = index
            self.value = self.value[:i] + (value,) + self.value[i + 1:]
            return
        if index != 0:
            raise IndexError('atomic batch allows index 0 only')
        self.value = value

    def _setitem_by_slice(self, index: 'slice', value: 'Tensors') ->None:
        if not index.start is index.stop is index.step is None:
            raise NotImplementedError('only slice [:] supported')
        if not self.atomic:
            self.value = value
            return
        if len(value) != 1:
            raise IndexError('atomic batch cannot be replaced with multiple tensors')
        self.value = value[0]


class CPUStreamType:
    pass


AbstractStream = Union[torch.Stream, CPUStreamType]


CPUStream = CPUStreamType()


def default_stream(device: 'torch.device') ->AbstractStream:
    """:func:`torch.cuda.default_stream` for either CPU or CUDA device."""
    if device.type != 'cuda':
        return CPUStream
    return torch.cuda.default_stream(device)


def as_cuda(stream: 'AbstractStream') ->torch.Stream:
    """Casts the given stream as :class:`torch.cuda.Stream`."""
    return cast(torch.Stream, stream)


def is_cuda(stream: 'Optional[AbstractStream]') ->bool:
    """Returns ``True`` if the given stream is a valid CUDA stream."""
    return stream is not CPUStream


def get_phony(device: 'torch.device', *, requires_grad: bool) ->Tensor:
    """Gets a phony. Phony is tensor without space. It is useful to make
    arbitrary dependency in a autograd graph because it doesn't require any
    gradient accumulation.

    .. note::

        Phonies for each device are cached. If an autograd function gets a phony
        internally, the phony must be detached to be returned. Otherwise, the
        autograd engine will mutate the cached phony in-place::

            class Phonify(torch.autograd.Function):
                @staticmethod
                def forward(ctx, input):
                    phony = get_phony(input.device, requires_grad=False)
                    return phony.detach()  # detach() is necessary.

    """
    key = device, requires_grad
    try:
        phony = _phonies[key]
    except KeyError:
        with use_stream(default_stream(device)):
            phony = torch.empty(1, device=device, requires_grad=requires_grad)
        _phonies[key] = phony
    return phony


class Fork(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "'Fork'", input: 'Tensor') ->Tuple[Tensor, Tensor]:
        phony = get_phony(input.device, requires_grad=False)
        return input.detach(), phony.detach()

    @staticmethod
    def backward(ctx: "'Fork'", grad_input: 'Tensor', grad_grad: 'Tensor') ->Tensor:
        return grad_input


def fork(input: 'Tensor') ->Tuple[Tensor, Tensor]:
    """Branches out from an autograd lane of the given tensor."""
    if torch.is_grad_enabled() and input.requires_grad:
        input, phony = Fork.apply(input)
    else:
        phony = get_phony(input.device, requires_grad=False)
    return input, phony


class Join(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "'Join'", input: 'Tensor', phony: 'Tensor') ->Tensor:
        return input.detach()

    @staticmethod
    def backward(ctx: "'Join'", grad_input: 'Tensor') ->Tuple[Tensor, None]:
        return grad_input, None


def join(input: 'Tensor', phony: 'Tensor') ->Tensor:
    """Merges two autograd lanes."""
    if torch.is_grad_enabled() and (input.requires_grad or phony.requires_grad):
        input = Join.apply(input, phony)
    return input


MOVING_DENIED = TypeError('denied to move parameters and buffers, because Pipe should manage device placement')


def save_rng_states(device: 'torch.device', rng_states: 'Deque[RNGStates]') ->None:
    """:meth:`Checkpoint.forward` captures the current PyTorch's random number
    generator states at CPU and GPU to reuse in :meth:`Recompute.backward`.

    .. seealso:: :ref:`Referential Transparency`

    """
    cpu_rng_state = torch.get_rng_state()
    gpu_rng_state: 'Optional[ByteTensor]'
    if device.type == 'cuda':
        gpu_rng_state = torch.get_rng_state(device)
    else:
        gpu_rng_state = None
    rng_states.clear()
    rng_states.append((cpu_rng_state, gpu_rng_state))


class Checkpoint(torch.autograd.Function):

    @staticmethod
    def forward(ctx: 'Context', phony: 'Tensor', recomputed: 'Deque[Recomputed]', rng_states: 'Deque[RNGStates]', function: 'Function', input_atomic: 'bool', *input: Tensor) ->TensorOrTensors:
        ctx.recomputed = recomputed
        ctx.rng_states = rng_states
        save_rng_states(input[0].device, ctx.rng_states)
        ctx.function = function
        ctx.input_atomic = input_atomic
        ctx.save_for_backward(*input)
        with torch.no_grad(), enable_checkpointing():
            output = function(input[0] if input_atomic else input)
        return output

    @staticmethod
    def backward(ctx: 'Context', *grad_output: Tensor) ->Tuple[Optional[Tensor], ...]:
        output, input_leaf = ctx.recomputed.pop()
        if isinstance(output, tuple):
            tensors = output
        else:
            tensors = output,
        if any(y.requires_grad for y in tensors):
            tensors = tuple([x for x in tensors if x.requires_grad])
            torch.autograd.backward(tensors, grad_output)
        grad_input: 'List[Optional[Tensor]]' = [None, None, None, None, None]
        grad_input.extend(x.grad for x in input_leaf)
        return tuple(grad_input)


class Recompute(torch.autograd.Function):

    @staticmethod
    def forward(ctx: 'Context', phony: 'Tensor', recomputed: 'Deque[Recomputed]', rng_states: 'Deque[RNGStates]', function: 'Function', input_atomic: 'bool', *input: Tensor) ->Tensor:
        ctx.recomputed = recomputed
        ctx.rng_states = rng_states
        ctx.function = function
        ctx.input_atomic = input_atomic
        ctx.save_for_backward(*input)
        return phony

    @staticmethod
    def backward(ctx: 'Context', *grad_output: Tensor) ->Tuple[None, ...]:
        input = ctx.saved_tensors
        input_leaf = tuple(x.detach().requires_grad_(x.requires_grad) for x in input)
        with restore_rng_states(input[0].device, ctx.rng_states):
            with torch.enable_grad(), enable_recomputing():
                output = ctx.function(input_leaf[0] if ctx.input_atomic else input_leaf)
        ctx.recomputed.append((output, input_leaf))
        grad_input: 'List[None]' = [None, None, None, None, None]
        grad_input.extend(None for _ in ctx.saved_tensors)
        return tuple(grad_input)


class Checkpointing:
    """Generates a pair of :class:`Checkpoint` and :class:`Recompute`."""

    def __init__(self, function: 'Function', batch: 'Batch') ->None:
        self.function = function
        self.batch = batch
        self.recomputed: 'Deque[Recomputed]' = deque(maxlen=1)
        self.rng_states: 'Deque[RNGStates]' = deque(maxlen=1)

    def checkpoint(self) ->Batch:
        """Returns a batch applied by :class:`Checkpoint`."""
        input_atomic = self.batch.atomic
        input = tuple(self.batch)
        phony = get_phony(self.batch[0].device, requires_grad=True)
        output = Checkpoint.apply(phony, self.recomputed, self.rng_states, self.function, input_atomic, *input)
        if isinstance(output, tuple):
            output = tuple([(x if x.is_floating_point() else x.detach()) for x in output])
        return Batch(output, self.batch.index)

    def recompute(self, batch: 'Batch') ->None:
        """Applies :class:`Recompute` to the batch in place."""
        input_atomic = self.batch.atomic
        input = tuple(self.batch)
        batch[0], phony = fork(batch[0])
        phony = Recompute.apply(phony, self.recomputed, self.rng_states, self.function, input_atomic, *input)
        batch[0] = join(batch[0], phony)


ExcInfo = Tuple[Type[BaseException], BaseException, TracebackType]


class Task:
    """A task represents how to compute a micro-batch on a partition.

    It consists of two parts: :meth:`compute` and :meth:`finalize`.
    :meth:`compute` should be executed in worker threads concurrently.
    :meth:`finalize` should be executed after when worker threads complete to
    execute :meth:`compute`.

    :meth:`compute` might be boosted by worker threads. Because it produces
    several CUDA API calls by user code. In PyTorch, parallel CUDA API calls
    are not serialized through GIL. So more than one CUDA API call can be
    produced at the same time.

    """

    def __init__(self, stream: 'Optional[AbstractStream]', *, compute: Callable[[], Batch], finalize: Optional[Callable[[Batch], None]]) ->None:
        self.stream = stream
        self._compute = compute
        self._finalize = finalize
        self._grad_enabled = torch.is_grad_enabled()

    def compute(self) ->Batch:
        with use_stream(self.stream), torch.set_grad_enabled(self._grad_enabled):
            return self._compute()

    def finalize(self, batch: 'Batch') ->None:
        if self._finalize is None:
            return
        with use_stream(self.stream), torch.set_grad_enabled(self._grad_enabled):
            self._finalize(batch)


def worker(in_queue: 'InQueue', out_queue: 'OutQueue', device: 'torch.device') ->None:
    """The main loop of a worker thread."""
    with use_device(device):
        while True:
            task = in_queue.get()
            if task is None:
                break
            try:
                batch = task.compute()
            except Exception:
                exc_info = cast(ExcInfo, sys.exc_info())
                out_queue.put((False, exc_info))
                continue
            out_queue.put((True, (task, batch)))
    done = False, None
    out_queue.put(done)


def current_stream(device: 'torch.device') ->AbstractStream:
    """:func:`torch.cuda.current_stream` for either CPU or CUDA device."""
    if device.type != 'cuda':
        return CPUStream
    return torch.cuda.current_stream(device)


def torch_version(version: 'str'=torch.__version__) ->Tuple[int, ...]:
    numbering = re.search('^(\\d+).(\\d+).(\\d+)([^\\+]*)(\\+\\S*)?$', version)
    if not numbering:
        return tuple()
    global _logged
    if numbering.group(4) and not _logged:
        logging.warning(f'Pytorch pre-release version {version} - assuming intent to test it')
        _logged = True
    return tuple(int(numbering.group(n)) for n in range(1, 4))


def check_pytorch_version() ->None:
    if torch_version() < (1, 8, 0):
        raise Exception('DistributedPipeline requires PyTorch version 1.8 or higher')


def _reshape_inputs(input: 'torch.Tensor', target: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
    """Convert 3D inputs to 2D for this kernel"""
    if len(input.shape) == 3:
        input = input.reshape(-1, input.shape[2])
    if len(target.shape) == 2:
        target = target.reshape(-1)
    return input, target


class BaselineSoftmax(nn.Module):
    """Baseline softmax that does an output linear projection and a softmax.


        We also support LMCL (Large Margin Cosine Loss) from the CosFace paper. See
        more detailed comment in the MEVO class below.

        This is intended to be used with an embedding layer with shared weights.

    Args:
        proj_weight (nn.Parameter):
            The shared weight.
        tile_factor (int):
            Unused. It is here to make kernel init easier with MEVO.
        log_softmax (bool):
            If True, use log_softmax instead of softmax.
        margin (float):
            Used in LMCL (when scale != None). See MEVO comments for
            more details.
        scale (Optional[float]):
            Used in LMCL. If scale is None, LMCL is turned off. See
            MEVO comments for more details.

    """

    def __init__(self, proj_weight: 'nn.Parameter', tile_factor: 'int'=0, log_softmax: 'bool'=True, margin: 'float'=0.35, scale: 'Optional[float]'=None):
        super().__init__()
        out_dim, in_dim = proj_weight.shape
        assert 'cuda' in str(proj_weight.device), 'weight should be on GPU'
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        assert proj_weight.dtype in [torch.float16, torch.float32]
        if proj_weight.dtype == torch.float16:
            self.fc = self.fc.half()
        self.fc.weight = proj_weight
        assert self.fc.weight.dtype in [torch.float16, torch.float32], self.fc.weight.dtype
        self.fp16 = self.fc.weight.dtype == torch.float16
        self.log_softmax = log_softmax
        self.margin = margin
        self.scale = scale

    def lmcl_pre_softmax(self, input: 'torch.Tensor', target: 'torch.Tensor') ->torch.Tensor:
        x = F.normalize(input, dim=1)
        w = F.normalize(self.fc.weight, dim=1)
        logits = torch.einsum('nc,kc->nk', x, w)
        row_ind = torch.arange(x.shape[0], dtype=torch.long)
        col_ind = target
        logits[row_ind, col_ind] -= self.margin
        logits *= self.scale
        return logits

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor') ->torch.Tensor:
        """Forward function that computes softmax output with the input and target."""
        assert isinstance(input, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        input, target = _reshape_inputs(input, target)
        if self.fp16:
            assert input.dtype == torch.float16
        if self.scale is not None:
            x = self.lmcl_pre_softmax(input, target)
        else:
            x = self.fc(input)
        if self.log_softmax:
            x = F.log_softmax(x, dim=-1, dtype=torch.float32)
        else:
            x = F.softmax(x, dim=-1, dtype=torch.float32)
        assert x.dtype == torch.float32
        return x


class BaselineSoftmaxNllLoss(BaselineSoftmax):
    """Baseline that does an output projection, a softmax & a NLL loss (cross-entropy).

    See BaselineSoftmax above. Constructor is the same. Only difference is in the
    forward function.

    This class is used for testing and benchmarking.
    """

    def __init__(self, proj_weight: 'nn.Parameter', tile_factor: 'int'=0, log_softmax: 'bool'=True, margin: 'float'=0.35, scale: 'Optional[float]'=None):
        super().__init__(proj_weight, tile_factor, log_softmax, margin, scale)

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor') ->torch.Tensor:
        """Forward that directly compute the loss."""
        assert isinstance(input, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        input, target = _reshape_inputs(input, target)
        x = super().forward(input, target)
        return F.nll_loss(x, target, reduction='sum')


DEBUG = False


class BackwardTriggerFn(torch.autograd.Function):
    """A backward trigger function."""

    @staticmethod
    def forward(ctx: 'Any', w: 'torch.Tensor', trigger_tensor: 'torch.Tensor') ->torch.Tensor:
        """We take a weight tensor and the trigger as inputs and output the weight directly."""
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            None
        ctx.save_for_backward(w, trigger_tensor)
        return w

    @staticmethod
    def backward(ctx: 'Any', *args: Any) ->Any:
        """We return zero grad for the trigger only."""
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            None
        assert len(args) == 1
        w, trigger = ctx.saved_tensors
        assert w.requires_grad
        assert trigger.requires_grad
        return None, torch.zeros_like(trigger)


class BackwardTrigger(nn.Module):
    """A backward trigger module.

    This module takes a parameter as an input and create a linked parameter
    from a newly created trigger parameter.

    The way to use it in a module's ``__init__'' and ``forward'' functions:

    ```
    def __init__():
      ...
      self.trigger = BackwardTrigger(some_layer.weight)
      ...

    def forward():
      w = self.trigger()
      ... continue to use w ...
    ```

    As a resule, the trigger's backward hook will be called at the end of
    the backward for the module that uses this trigger.
    """

    def __init__(self, linked_param: 'torch.Tensor'):
        super().__init__()
        assert isinstance(linked_param, nn.Parameter)
        self.trigger = nn.Parameter(torch.rand(1, dtype=linked_param.dtype, device=linked_param.device))
        self.trigger._linked_param = linked_param

    def forward(self) ->torch.Tensor:
        return BackwardTriggerFn.apply(self.trigger._linked_param, self.trigger)


def lmcl_matmul(i: 'torch.Tensor', w: 'torch.Tensor', tgt: 'torch.Tensor', w_idx: 'int', margin: 'float', scale: 'Optional[float]') ->torch.Tensor:
    """LMCL variation of matmul with normalization, margin and scale."""
    logits = torch.matmul(F.normalize(i, dim=1), F.normalize(w, dim=1).T)
    mask = torch.arange(w_idx * w.shape[0], (w_idx + 1) * w.shape[0], dtype=torch.long, device=i.device).expand(i.shape[0], -1)
    logits[mask == tgt.reshape(-1, 1)] -= margin
    logits *= scale
    return logits


class GetMaxFunction(torch.autograd.Function):
    """Custom checkpointed function to get max-per-token from an input and a weight"""

    @staticmethod
    def get_max(i: 'torch.Tensor', w: 'torch.Tensor', tgt: 'torch.Tensor', w_idx: 'int', full_precision: 'bool', margin: 'float', scale: 'Optional[float]') ->torch.Tensor:
        """
        Throughout this code:

          i: input data with shape = (split-of-tokens, d_model)
          w: weight data with shape = (split-of-vocabs, d_model)
          tgt: target prediction data with shape = (split-of-tokens,)
        """
        if scale is not None:
            _m = lmcl_matmul(i, w, tgt, w_idx, margin, scale)
        else:
            _m = torch.matmul(i, w.T)
        if full_precision:
            _m = _m.float()
        _m = _m.max(dim=1)[0]
        return _m

    @staticmethod
    def forward(ctx: 'Any', i: 'torch.Tensor', w: 'torch.Tensor', tgt: 'torch.Tensor', kernel_obj: "'MemoryEfficientVocabOutput'", w_idx: 'int', w_split_size: 'int', split_dim: 'int') ->torch.Tensor:
        """Forward function that computes the max, without saving activations."""
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            None
        ctx.save_for_backward(i, w, tgt)
        ctx.kernel_obj = kernel_obj
        ctx.w_idx = w_idx
        ctx.w_split_size = w_split_size
        ctx.args = {}
        assert split_dim == 0
        with torch.no_grad():
            return GetMaxFunction.get_max(i, w, tgt, w_idx, kernel_obj.fp_max, kernel_obj.margin, kernel_obj.scale)

    @staticmethod
    def backward(ctx: 'Any', *args: Any) ->Any:
        """Recompute the forward max and backward grad.

        Accumulate the grad to the right split of the full grad.
        """
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            None
        assert len(args) == 1
        assert ctx.kernel_obj.proj_weight.grad is not None
        i, w, tgt = ctx.saved_tensors
        assert i.requires_grad
        assert w.requires_grad
        i = i.detach().requires_grad_(True)
        w = w.detach().requires_grad_(True)
        with torch.enable_grad():
            maxs = GetMaxFunction.get_max(i, w, tgt, ctx.w_idx, ctx.kernel_obj.fp_max, ctx.kernel_obj.margin, ctx.kernel_obj.scale)
        torch.autograd.backward(maxs, *args)
        assert w.grad is not None
        with torch.no_grad():
            grads = torch.split(ctx.kernel_obj.proj_weight.grad, ctx.w_split_size)
            grads[ctx.w_idx].add_(w.grad)
        return i.grad, None, None, None, None, None, None


class GetSumFunction(torch.autograd.Function):
    """Custom checkpointed function to get sum-per-token from an input and a weight."""

    @staticmethod
    def get_sum(i: 'torch.Tensor', w: 'torch.Tensor', tgt: 'torch.Tensor', maxs: 'torch.Tensor', w_idx: 'int', full_precision: 'bool', margin: 'float', scale: 'Optional[float]') ->torch.Tensor:
        if scale is not None:
            _s = lmcl_matmul(i, w, tgt, w_idx, margin, scale)
        else:
            _s = torch.matmul(i, w.T)
        if full_precision:
            _s = _s.float()
        _s = (_s - maxs.reshape(-1, 1)).exp().sum(dim=1)
        return _s

    @staticmethod
    def forward(ctx: 'Any', i: 'torch.Tensor', w: 'torch.Tensor', tgt: 'torch.Tensor', maxs: 'torch.Tensor', kernel_obj: "'MemoryEfficientVocabOutput'", w_idx: 'int', w_split_size: 'int', split_dim: 'int') ->torch.Tensor:
        """Forward function that computes the sum, without saving activations."""
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            None
        ctx.save_for_backward(i, w, tgt, maxs)
        ctx.kernel_obj = kernel_obj
        ctx.w_idx = w_idx
        ctx.w_split_size = w_split_size
        assert split_dim == 0
        with torch.no_grad():
            return GetSumFunction.get_sum(i, w, tgt, maxs, w_idx, kernel_obj.fp_sum, kernel_obj.margin, kernel_obj.scale)

    @staticmethod
    def backward(ctx: 'Any', *args: Any) ->Any:
        """Recompute the forward sum and backward grad.

        Accumulate the grad to the right split of the full grad.
        """
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            None
        assert len(args) == 1
        assert ctx.kernel_obj.proj_weight.grad is not None
        i, w, tgt, maxs = ctx.saved_tensors
        assert i.requires_grad
        assert w.requires_grad
        assert maxs.requires_grad
        i = i.detach().requires_grad_(True)
        w = w.detach().requires_grad_(True)
        maxs = maxs.detach().requires_grad_(True)
        with torch.enable_grad():
            sums = GetSumFunction.get_sum(i, w, tgt, maxs, ctx.w_idx, ctx.kernel_obj.fp_sum, ctx.kernel_obj.margin, ctx.kernel_obj.scale)
        torch.autograd.backward(sums, *args)
        assert w.grad is not None
        with torch.no_grad():
            grads = torch.split(ctx.kernel_obj.proj_weight.grad, ctx.w_split_size)
            grads[ctx.w_idx].add_(w.grad)
        return i.grad, None, None, maxs.grad, None, None, None, None


class TargetScoreFunction(torch.autograd.Function):
    """Custom checkpointed function to compute the target score."""

    @staticmethod
    def get_target_score(i: 'torch.Tensor', w: 'torch.Tensor', target: 'torch.Tensor', full_precision: 'bool', margin: 'float', scale: 'Optional[float]') ->torch.Tensor:
        tokens, d_model = i.shape
        assert d_model == w.shape[1]
        tw = w.gather(dim=0, index=target.reshape(target.shape[0], 1).expand(target.shape[0], d_model))
        assert tw.shape == (tokens, d_model)
        if scale is not None:
            target_score = F.normalize(i, dim=1) * F.normalize(tw, dim=1)
        else:
            target_score = i * tw
        if full_precision:
            target_score = target_score.float()
        target_score = target_score.sum(dim=1)
        if scale is not None:
            target_score -= margin
            target_score *= scale
        return target_score

    @staticmethod
    def forward(ctx: 'Any', i: 'torch.Tensor', w: 'torch.Tensor', target: 'torch.Tensor', kernel_obj: "'MemoryEfficientVocabOutput'") ->torch.Tensor:
        """Forward, without activations."""
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            None
        ctx.save_for_backward(i, w, target)
        ctx.kernel_obj = kernel_obj
        with torch.no_grad():
            x = TargetScoreFunction.get_target_score(i, w, target, kernel_obj.fp_target, kernel_obj.margin, kernel_obj.scale)
        return x

    @staticmethod
    def backward(ctx: 'Any', *args: Any) ->Any:
        """Forward and backward again, assign or accumulate the gradients."""
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            None
        assert len(args) == 1
        i, w, target = ctx.saved_tensors
        assert i.requires_grad
        assert w.requires_grad
        assert not target.requires_grad
        i = i.detach().requires_grad_(True)
        w = w.detach().requires_grad_(True)
        with torch.enable_grad():
            scores = TargetScoreFunction.get_target_score(i, w, target, ctx.kernel_obj.fp_target, ctx.kernel_obj.margin, ctx.kernel_obj.scale)
        torch.autograd.backward(scores, *args)
        if ctx.kernel_obj.proj_weight.grad is not None:
            ctx.kernel_obj.proj_weight.grad.add_(w.grad)
        else:
            ctx.kernel_obj.proj_weight.grad = w.grad
        return i.grad, None, None, None


def _next_power_of_2_or_max(n: 'int', max_n: 'int') ->int:
    """Return the smallest power of 2 greater than or equal to n, with a limit.

    Useful when used in splitting a tensor into chunks with power-of-2 sizes.
    """
    if n == 0:
        return 1
    orig_n = n
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    assert n >= orig_n, f'{n} vs. {orig_n}'
    assert bin(n).count('1') == 1, bin(n)
    if n > max_n:
        return max_n
    return n


class MemoryEfficientVocabOutput(nn.Module):
    """Fused fc + softmax + nll_loss in a tiled fashion.

        MEVO uses much less memory but is quite a bit slower.

        MEVO also implements the LMCL (Large Margin Cosine Loss) function introduced by
        highly cited
        `CosFace: Large Margin Cosine Loss for Deep Face Recognition [Wang et al.]`_.

        .. _`CosFace: Large Margin Cosine Loss for Deep Face Recognition [Wang et al.]`: https://arxiv.org/abs/1801.09414

        LMCL can be turned on using the ``margin`` and ``scale`` parameters below. These
        hyperparameters most likely require tuning, depending on the number of classes etc.

        MEVO LMCL can be suitable for face recognition and image retrieval tasks, esp. when
        the number prediction target classes is large. MEVO is slower but can use much
        less GPU memory in that case, which enables training with larger batches. We
        hope this is helpful but we strongly recommend users (AI researchers
        and engineers) to carefully consider their applications of this technology. This
        types of technology should not be used by small group of people exclusively to
        potentially harm the general public.

    Args:
        proj_weight (nn.Parameter):
            Sharing this weight with an embedding layer.
        tile_factor (int):
            Number of splits to use on the input sequence and vocab dimensions.
            Default: 16
        reduction (str):
            Reduction OP (sum or mean).
            Default: sum
        margin (float):
            Hyperparameter of the separation margin between classes. See the
            appendix of the CosFace paper for a formula on how to compute its
            value properly. The default value is unlikely to be suitable in all
            cases.
            Default: 0.35
        scale (Optional[float]):
            Hyperparameter of the feature-vector-scaling for LMCL. When not
            supplied, LMCL is turned off. See the appendix of the CosFace paper for
            a formula on how to compute its value properly.
            Default: None
    """

    def __init__(self, proj_weight: 'nn.Parameter', tile_factor: 'int'=16, reduction: 'str'='sum', margin: 'float'=0.35, scale: 'Optional[float]'=None):
        super().__init__()
        self.proj_weight = proj_weight
        self.tf_in, self.tf_w = tile_factor, tile_factor
        self.fp_max = True
        self.fp_sum = True
        self.fp_target = True
        self.log_softmax = True
        self.reduction = reduction
        assert self.reduction in ['sum', 'mean']
        self.margin = margin
        self.scale = scale
        self.trigger = BackwardTrigger(self.proj_weight)
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            None

    def get_target_nlprob(self, i: 'torch.Tensor', w: 'torch.Tensor', target: 'torch.Tensor', debase_max: 'torch.Tensor', exp_sums: 'torch.Tensor') ->torch.Tensor:
        """Get target's negative log probability."""
        target_score = TargetScoreFunction.apply(i, w, target, self)
        prob = (target_score - debase_max).exp() / exp_sums
        if self.log_softmax:
            prob = prob.log()
        return -prob.sum()

    def eval_forward(self, input: 'torch.Tensor') ->torch.Tensor:
        """Eval time forward that doesn't fuse the softmax and NLL Loss kernels."""
        return torch.matmul(input, self.proj_weight.T)

    def forward(self, input: 'torch.Tensor', target: 'Optional[torch.Tensor]') ->torch.Tensor:
        if not self.training and target is None:
            return self.eval_forward(input)
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            cur_mem = round(torch.cuda.memory_allocated() / 1024 / 1024)
            mem = round(torch.cuda.max_memory_allocated() / 1024 / 1024)
            None
        assert isinstance(input, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        if torch.is_grad_enabled():
            assert input.requires_grad
        input, target = _reshape_inputs(input, target)
        tokens, d_model = input.shape
        t2, = target.shape
        vocab, d2 = self.proj_weight.shape
        assert d_model == d2, f'incorrect shape {d_model} vs {d2}'
        assert tokens == t2, f'incorrect shape {tokens} vs {t2}'
        split_dim = 0
        input_split_size = _next_power_of_2_or_max(tokens // self.tf_in, tokens)
        weight_split_size = _next_power_of_2_or_max(vocab // self.tf_w, vocab)
        inputs = torch.split(input, input_split_size, split_dim)
        weight = self.trigger()
        weights = torch.split(weight, weight_split_size, split_dim)
        targets = tuple([torch.Tensor()] * len(inputs))
        if self.scale is not None:
            targets = torch.split(target, input_split_size, split_dim)
        maxs = []
        for i, tgt in zip(inputs, targets):
            m = None
            for w_idx, w in enumerate(weights):
                _m = GetMaxFunction.apply(i, w, tgt, self, w_idx, weight_split_size, split_dim)
                if m is None:
                    m = _m
                else:
                    m = torch.max(m, _m)
            assert m is not None
            maxs.append(m)
        maxs_tensor = torch.cat(maxs)
        assert maxs_tensor.shape == (tokens,)
        sums = []
        for i, tgt, debase_max in zip(inputs, targets, maxs):
            s = None
            for w_idx, w in enumerate(weights):
                _s = GetSumFunction.apply(i, w, tgt, debase_max, self, w_idx, weight_split_size, split_dim)
                if s is None:
                    s = _s
                else:
                    s += _s
            assert s is not None
            sums.append(s)
        sums_tensor = torch.cat(sums)
        assert sums_tensor.shape == (tokens,)
        result = self.get_target_nlprob(input, self.proj_weight, target, maxs_tensor, sums_tensor)
        if self.reduction == 'mean':
            result /= tokens
        return result


class ModelShard(nn.Module):
    """
    Wrap one shard of the model, make it possible to load parameters on the
    fly for the FW and BW pass on the given device.
    """

    def __init__(self, cpu_model_shard: 'nn.Module', device: 'torch.device', offload_device: 'torch.device', index: 'int'):
        super().__init__()
        self.model_shard = cpu_model_shard
        self.index = index
        self.device = device
        torch.device(self.device)
        self.offload_device = offload_device
        self.model_shard
        self._cpu_to_gpu_stream = torch.Stream(device=self.device)
        self._gpu_to_cpu_stream = torch.Stream(device=self.device)

    def forward(self, *inputs):
        return self.model_shard(*inputs) if isinstance(inputs, tuple) else self.model_shard(inputs)

    def to(self, device: 'torch.device') ->'ModelShard':
        self.model_shard
        return self

    def train(self, mode: 'bool'=True) ->'ModelShard':
        self.model_shard.train(mode)
        return self

    def to_device(self) ->None:
        self.model_shard

    def forward_load(self, non_blocking: 'bool'=True) ->None:
        with torch.cuda.stream(self._cpu_to_gpu_stream):
            self.model_shard

    def backward_load(self, non_blocking: 'bool'=True) ->None:
        with torch.cuda.stream(self._cpu_to_gpu_stream):
            self.model_shard

    def forward_drop(self, non_blocking: 'bool'=True) ->None:
        with torch.cuda.stream(self._gpu_to_cpu_stream):
            self.model_shard

    def backward_drop(self, non_blocking: 'bool'=True) ->None:
        with torch.cuda.stream(self._gpu_to_cpu_stream):
            self.model_shard


def _conditional_amp_bwd_decorator(orig_func):
    if hasattr(torch.amp, 'custom_bwd'):
        return torch.amp.custom_bwd(orig_func)

    @functools.wraps(orig_func)
    def inner_decorator(*args: Any, **kwargs: Any) ->Any:
        return orig_func(*args, **kwargs)
    return inner_decorator


def _conditional_amp_fwd_decorator(orig_func):
    if hasattr(torch.amp, 'custom_fwd'):
        return torch.amp.custom_fwd(orig_func)

    @functools.wraps(orig_func)
    def inner_decorator(*args: Any, **kwargs: Any) ->Any:
        return orig_func(*args, **kwargs)
    return inner_decorator


_MODEL_PARALLEL_GROUP = None


def get_model_parallel_group() ->torch.distributed.ProcessGroup:
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, 'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP


def ensure_divisibility(numerator: 'int', denominator: 'int') ->None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(numerator, denominator)


def divide_and_check_no_remainder(numerator: 'int', denominator: 'int') ->int:
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(tensor: 'torch.Tensor', num_partitions: 'int', contiguous_split_chunks: 'bool'=False) ->Tuple[torch.Tensor, ...]:
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    last_dim = tensor.dim() - 1
    last_dim_size = divide_and_check_no_remainder(tensor.size()[last_dim], num_partitions)
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)
    return tensor_list


def _split(input_: 'torch.Tensor') ->torch.Tensor:
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    group = get_model_parallel_group()
    if torch.distributed.get_world_size(group=group) == 1:
        return input_
    world_size = torch.distributed.get_world_size(group=group)
    input_list = split_tensor_along_last_dim(input_, world_size)
    rank = torch.distributed.get_rank(group=group)
    output = input_list[rank].contiguous()
    return output


class OffloadModel(nn.Module):
    """Wraps an arbitrary :class:`nn.Sequential <torch.nn.Sequential>` module
    to train by offloading majority of the model parameters to the CPU.
    `OffloadModel` is heavily inspired by the _L2L algorithm and _Zero-Offload.
    ::

        model = get_model()
        offload_model = OffloadModel(model, device,
                                    offload_device=torch.device(“cpu”),
                                    num_slices=3,
                                    checkpoint_activation=True,
                                    num_microbatches=5)

    .. _L2L: https://arxiv.org/abs/2002.05645
    .. _Zero-Offload: https://arxiv.org/abs/2101.06840

    At each step, a layer(or series of layers) are loaded
    onto the GPU for the forward and backward pass with intermediate
    activations being copied onto the GPU as required. Once the forward
    or backward pass is completed for a given shard, it is moved back to
    the CPU again.

    `OffloadModel` supports activation checkpointing which reduces
    the memory footprint. You can also increase the number of
    microbatches which translates to more computation cycles for
    every shard load. This helps offset the cost of moving the shard
    from the CPU to GPU and vice versa.

    Note: OffloadModel currently only supports nn.Sequential models.

    Args:
        module (~torch.nn.Sequential): Module to be offloaded.

        device (torch.device):
            Device where the active model should reside.

        offload_device (torch.device):
            Device where the inactive model should reside.

        num_slices (int):
            Number of slices into which the model should be chunked.

        checkpoint_activation (bool):
            Boolean to indicate if we want to checkpoint intermediate
            activation states on the CPU. Default value is False.

        num_microbatches (int):
            Number of microbatches which should be run per model
            shard on device.
    """

    def __init__(self, model: 'Any', device: 'torch.device', offload_device: 'torch.device'=torch.device('cpu'), num_slices: 'int'=3, checkpoint_activation: 'bool'=False, num_microbatches: 'int'=1):
        super().__init__()
        if not model:
            raise TypeError('`model` argument to `OffloadModel` cannot be None.')
        if not device:
            raise TypeError('`device` argument to `OffloadModel` cannot be None.')
        if not (isinstance(model, nn.Sequential) or type(model) == list):
            raise TypeError('`model` argument to `OffloadModel` must be of type `nn.Sequential`.')
        if not torch.cuda.is_available():
            raise TypeError('CUDA must be available as one of the compute devices for `OffloadModel`.')
        self.device = device
        self.offload_device = offload_device
        self.model_slices: 'List[nn.Module]' = []
        if type(model) == list:
            for i, m in enumerate(model):
                self.model_slices.append(ModelShard(cpu_model_shard=m, device=device, offload_device=offload_device, index=i))
        else:
            splits = _split(model, num_slices)
            for i, split in enumerate(splits):
                self.model_slices.append(ModelShard(cpu_model_shard=nn.Sequential(*split), device=device, offload_device=offload_device, index=i))
        self._model = torch.nn.Sequential(*self.model_slices)
        self._activations: 'List[Tuple]' = []
        if not checkpoint_activation and num_microbatches > 1:
            raise RuntimeError('We currently only support microbatches with activation checkpointing.')
        self._checkpoint_activation = checkpoint_activation
        self._num_microbatches = num_microbatches

    def forward(self, *inputs: Any, **_: Any) ->Any:
        if self._checkpoint_activation:
            return OffloadFunction.apply(*inputs, torch.tensor([], requires_grad=True), self)
        self._activations = []
        for index in range(-1, len(self.model_slices)):
            if index >= 0:
                self._activations[index] = tuple([a for a in list(self._activations[index])])
                inputs = self._activations[index]
                inputs = self.model_slices[index](*inputs)
            inputs = ShardSyncLayer.apply(inputs, index, self.model_slices, self)
            self._activations.append(inputs)
            if index >= 0:
                self._activations[index] = tuple([a.cpu() for a in list(self._activations[index])])
        result = self._activations[-1]
        result = tuple([r for r in result])
        return result[0] if len(result) == 1 else result


def _forward(input: 'Tensor', affine: 'bool', mean: 'Tensor', invstd: 'Tensor', weight: 'Tensor', bias: 'Tensor') ->Tensor:
    if affine:
        return (input - mean) * (invstd * weight.reshape_as(mean)) + bias.reshape_as(mean)
    else:
        return (input - mean) * invstd


class _SyncBatchNormFunction(torch.autograd.Function):
    """
    An autograd function used to avoid storing activations for intermediate results.

    NOTE: Even though the mean and var are passed into this function, we do the entire
    backward, including mean and var, here. We have to calculate statistics outside
    this function in order to avoid multiple all_reduces when using checkpointing.
    """

    @staticmethod
    def forward(ctx, input, weight, bias, affine, mean, invstd, total_count, process_group):
        ctx.save_for_backward(input, weight, bias, mean, invstd, total_count)
        ctx.process_group = process_group
        return _forward(input, affine, mean, invstd, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        needs_input_grad = ctx.needs_input_grad[0]
        needs_weight_grad = ctx.needs_input_grad[1]
        grad_input = None
        grad_weight = None
        grad_bias = None
        input, weight, bias, mean, invstd, total_count = ctx.saved_tensors
        process_group = ctx.process_group
        dim = [d for d in range(input.ndim) if d != 1]
        if needs_input_grad or needs_weight_grad:
            grad_common = torch.sum((input - mean) * grad_output, dim=dim, keepdim=True)
        if needs_input_grad:
            if weight is None:
                grad_input = invstd * grad_output
                grad_mean = -torch.sum(grad_input, dim=dim, keepdim=True)
                grad_invstd = grad_common
            else:
                grad_input = invstd * weight.reshape_as(mean) * grad_output
                grad_mean = -torch.sum(grad_input, dim=dim, keepdim=True)
                grad_invstd = grad_common * weight.reshape_as(mean)
            grad_var = -0.5 * invstd.pow(3) * grad_invstd
            grad_mean += -2 * mean * grad_var
            grad_meansqr = grad_var
            vec = torch.cat([grad_mean, grad_meansqr])
            all_reduce_handle = dist.all_reduce(vec, group=process_group, async_op=True)
        if needs_weight_grad:
            grad_weight = (grad_common * invstd).resize_as(weight)
            grad_bias = torch.sum(grad_output, dim=dim)
        if needs_input_grad:
            all_reduce_handle.wait()
            vec = vec / total_count
            grad_mean, grad_meansqr = vec.chunk(2)
            grad_input += grad_mean
            grad_input += input * (2 * grad_meansqr)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None


def _calculate_stats(input: 'Tensor', eps: 'float', process_group: 'ProcessGroup') ->Tuple[Tensor, Tensor, Tensor, Tensor]:
    dim = [d for d in range(input.ndim) if d != 1]
    count = torch.full((1,), input.numel() // input.size(1), device=input.device, dtype=input.dtype)
    total_count = count.clone()
    all_reduce_handle = dist.all_reduce(total_count, group=process_group, async_op=True)
    mean = torch.mean(input, dim=dim, keepdim=True)
    meansqr = torch.mean(input * input, dim=dim, keepdim=True)
    vec = torch.cat([mean, meansqr])
    all_reduce_handle.wait()
    vec = vec * (count / total_count)
    dist.all_reduce(vec, group=process_group)
    mean, meansqr = vec.chunk(2)
    var = meansqr - mean * mean
    invstd = torch.rsqrt(var + eps)
    return mean, var, invstd, total_count


def _track_running_stats(running_mean: 'Tensor', running_var: 'Tensor', momentum: 'float', mean: 'Tensor', var: 'Tensor', total_count: 'Tensor') ->None:
    unbiased_var = var * (total_count / (total_count - 1))
    running_mean += momentum * (mean.reshape(-1) - running_mean)
    running_var += momentum * (unbiased_var.reshape(-1) - running_var)


def is_checkpointing() ->bool:
    """Whether the current forward propagation is under checkpointing.

    Returns:
        bool: :data:`True` if it's under checkpointing.

    """
    return thread_local.is_checkpointing


def is_recomputing() ->bool:
    """Whether the current forward propagation is under checkpoint
    recomputation. Use this to prevent duplicated side-effects at forward
    propagation::

        class Counter(nn.Module):
            def __init__(self):
                super().__init__()
                self.counter = 0

            def forward(self, input):
                if not is_recomputing():
                    self.counter += 1
                return input

    Returns:
        bool: :data:`True` if it's under checkpoint recomputation.

    .. seealso:: :ref:`Detecting Recomputation`

    """
    return thread_local.is_recomputing


class SyncBatchNorm(torch.nn.BatchNorm2d):
    """
    Fast re-implementation of ``torch.nn.SyncBatchNorm`` that can achieve a speedup
    of 5x or more over the default implementation depending on size of the input
    and number of distributed workers.
    """

    def __init__(self, *args: Tuple[Any, ...], process_group: Optional[ProcessGroup]=None, **kwargs: Dict[str, Any]) ->None:
        super().__init__(*args, **kwargs)
        self._process_group = process_group if process_group is not None else dist.group.WORLD
        self.saved_for_2nd_fwd: 'List[Tuple]' = []
        self.disable_patch_batchnorm = True

    def forward(self, input: 'Tensor') ->Tensor:
        if not dist.is_initialized() or not self.training:
            return super().forward(input)
        wrapped = is_checkpointing() or is_recomputing()
        if not wrapped or is_checkpointing():
            with torch.no_grad():
                mean, var, invstd, total_count = _calculate_stats(input, self.eps, self._process_group)
                if self.track_running_stats:
                    _track_running_stats(self.running_mean, self.running_var, self.momentum, mean, var, total_count)
        if is_checkpointing():
            self.saved_for_2nd_fwd.append((mean, invstd, total_count))
            return _forward(input, self.affine, mean, invstd, self.weight, self.bias)
        if is_recomputing():
            mean, invstd, total_count = self.saved_for_2nd_fwd.pop(0)
        return _SyncBatchNormFunction.apply(input, self.weight, self.bias, self.affine, mean, invstd, total_count, self._process_group)

    @classmethod
    def convert_sync_batchnorm(cls, module: 'torch.nn.Module', process_group: 'Optional[ProcessGroup]'=None) ->torch.nn.Module:
        """Helper function to convert all :attr:`BatchNorm*D` layers in the model to
        :class:`fairscale.experimental.nn.SyncBatchNorm` layers.

        Args:
            module (nn.Module): module containing one or more attr:`BatchNorm*D` layers
            process_group (optional): process group to scope synchronization,
                default is the whole world

        Returns:
            The original :attr:`module` with the converted :class:`torch.nn.SyncBatchNorm`
            layers. If the original :attr:`module` is a :attr:`BatchNorm*D` layer,
            a new :class:`torch.nn.SyncBatchNorm` layer object will be returned
            instead.

        Example::

            >>> # Network with nn.BatchNorm layer
            >>> module = torch.nn.Sequential(
            >>>            torch.nn.Linear(20, 100),
            >>>            torch.nn.BatchNorm1d(100),
            >>>          ).cuda()
            >>> # creating process group (optional)
            >>> # ranks is a list of int identifying rank ids.
            >>> ranks = list(range(8))
            >>> r1, r2 = ranks[:4], ranks[4:]
            >>> # Note: every rank calls into new_group for every
            >>> # process group created, even if that rank is not
            >>> # part of the group.
            >>> process_groups = [torch.distributed.new_group(pids) for pids in [r1, r2]]
            >>> process_group = process_groups[0 if dist.get_rank() <= 3 else 1]
            >>> sync_bn_module = fairscale.experimental.nn.SyncBatchNorm.convert_sync_batchnorm(module, process_group)

        """
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module_output = SyncBatchNorm(module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats, process_group=process_group)
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
            if hasattr(module, 'qconfig'):
                module_output.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_sync_batchnorm(child, process_group))
        del module
        return module_output


class FlatParameter(nn.Parameter):
    """A parameter that is initialized from a list of parameters and can be
    turned into a list of views as needed.
    """

    def __new__(cls, params: 'Sequence[nn.Parameter]', requires_grad: 'bool'=True) ->'FlatParameter':
        """Make an object using the parent's __new__ function."""
        if not isinstance(params, (list, tuple)) or len(params) == 0:
            raise ValueError('An non-empty list or tuple argument is needed')
        if not all(isinstance(p, (nn.Parameter, Tensor)) for p in params):
            raise ValueError('List items need to be Parameter types')
        if any(isinstance(p, FlatParameter) for p in params):
            raise ValueError('Nesting FlatParameter is not supported')
        data = torch.cat([(p.detach().reshape(-1) if isinstance(p, nn.Parameter) else p.reshape(-1)) for p in params], 0)
        return super(FlatParameter, cls).__new__(cls, data, requires_grad=requires_grad)

    def __init__(self, params: 'Sequence[nn.Parameter]', requires_grad: 'bool'=True):
        """Initialize the _param_numels and _param_shapes lists."""
        self._param_numels = [p.numel() for p in params]
        assert self.numel() <= sum(self._param_numels), f'Something wrong with __new__ method, {self.numel()} vs. {sum(self._param_numels)}'
        self._param_shapes = [p.size() for p in params]
        self._param_infos: 'List[Tuple[str, nn.Module, str]]' = []
        self._shared_param_infos: 'List[Tuple[str, str, nn.Module, str, nn.Module, str]]' = []

    def get_param_views(self, external_data: 'Optional[Tensor]'=None) ->Iterator[Tensor]:
        """Return a generator of views that map to the original parameters."""
        assert self.data.numel() <= sum(self._param_numels), f'Incorrect internal state {self.data.numel()} vs. {sum(self._param_numels)}'
        data = external_data if external_data is not None else self
        if data.numel() != sum(self._param_numels):
            raise ValueError(f'Incorrect numel of supplied data: got {data.numel()} but expected {sum(self._param_numels)}')
        return (t.view(s) for t, s in zip(data.split(self._param_numels), self._param_shapes))

    def metadata(self) ->Tuple[List[str], List[torch.Size], List[int]]:
        """Return tuple of (names, shapes, numels) metadata for this flat parameter."""
        names = [('.'.join([m, n]) if m else n) for m, _, n in self._param_infos]
        return names, self._param_shapes, self._param_numels

    def __setstate__(self, state: 'Tuple[Any, Any, Any, Any]') ->None:
        """Use by pickle to set the internal states."""
        self._param_numels, self._param_shapes, self._param_infos, self._shared_param_infos = state
        assert self.numel() <= sum(self._param_numels), f'Incorrect pickling {self.numel()} vs. {sum(self._param_numels)}'

    def __reduce_ex__(self, proto: 'int') ->Tuple[Any, Any, Any]:
        """Support pickling between ranks."""
        return FlatParameter, ([self.data], self.requires_grad), (self._param_numels, self._param_shapes, self._param_infos, self._shared_param_infos)


def replace_by_prefix_(state_dict: "Union[Dict[str, Tensor], 'OrderedDict[str, Tensor]']", old_prefix: 'str', new_prefix: 'str') ->None:
    """
    Replace all keys that match a given old_prefix with a new_prefix (in-place).

    Usage::

        state_dict = {"layer.xyz": torch.tensor(1)}
        replace_by_prefix_(state_dict, "layer.", "module.layer.")
        assert state_dict == {"module.layer.xyz": torch.tensor(1)}
    """
    if old_prefix == new_prefix:
        raise ValueError('old_prefix and new_prefix must be distinct')
    for key in list(state_dict.keys()):
        if not key.startswith(old_prefix):
            continue
        new_key = new_prefix + key[len(old_prefix):]
        state_dict[new_key] = state_dict[key]
        del state_dict[key]


def _post_state_dict_hook(module: 'nn.Module', state_dict: "'OrderedDict[str, Tensor]'", prefix: 'str', *args: Any) ->'OrderedDict[str, Tensor]':
    replace_by_prefix_(state_dict, prefix + '_fpw_module.', prefix)
    return state_dict


_enable_pre_load_state_dict_hook = True


def _pre_load_state_dict_hook(state_dict: "Union[Dict[str, Tensor], 'OrderedDict[str, Tensor]']", prefix: 'str', *args: Any) ->None:
    if not _enable_pre_load_state_dict_hook:
        return
    replace_by_prefix_(state_dict, prefix, prefix + '_fpw_module.')
    flat_param_key = prefix + '_fpw_module.flat_param'
    for k in list(state_dict.keys()):
        if k.startswith(flat_param_key):
            last_part = k.split('.')[-1]
            assert last_part.startswith('flat_param_'), last_part
            replace_by_prefix_(state_dict, k, prefix + last_part)


class ProcessGroupName(str, Enum):
    default = 'default'
    reduce_scatter = 'reduce_scatter'


class Bucket:
    """
    Helper class to simplify the handling of buckets, which unify the underlying storage of multiple tensors
    """

    def __init__(self, size: 'int', dtype: 'torch.dtype', device: 'torch.device') ->None:
        self._params: 'List[torch.Tensor]' = []
        self._param_ids: 'List[int]' = []
        self._fill = 0
        self.buffer: 'torch.Tensor' = torch.zeros(size, dtype=dtype, device=device)

    def to(self, device: 'Optional[Union[int, torch.device]]', dtype: 'Optional[torch.dtype]'=None, non_blocking: 'bool'=False, keep_param_alignment: 'bool'=True) ->'ParamBucket':
        """
        Move the underlying buffer
        """
        assert self.buffer is not None, 'Cannot move a collapsed bucket, please rebuild it'
        self.buffer = self.buffer


class ReduceScatterBucketer:
    """
    Helper for bucketing multiple reduce-scatter operations on small tensors
    into larger reduce-scatter ops to improve communication efficiency.

    Usage::

        bucketer = ReduceScatterBucketer()
        bucketer.reduce_scatter_async(
            small_tensors, callback_fn=lambda result: print("small")
        )
        bucketer.reduce_scatter_async(
            big_tensors, callback_fn=lambda result: print("big")
        )
        bucketer.reduce_scatter_async(
            more_small_tensors, callback_fn=lambda result: print("small2")
        )
        bucketer.flush()  # callbacks only guaranteed to be called after flush()
        # Example output (note that it is out of order, due to bucketing):
        # big
        # small
        # small2

    Args:
        bucket_cap_mb (int, Optional): bucket size for communicating. Buckets
            are sub-divided based on world_size. Values <= 0 disable bucketing.
    """

    def __init__(self, bucket_cap_mb: 'int'=25):
        self.bucket_cap_mb = bucket_cap_mb
        self.buckets: "Dict[Tuple[torch.dtype, torch.device, 'ProcessGroup'], Bucket]" = {}

    @torch.no_grad()
    def reduce_scatter_async(self, input_list: 'List[Tensor]', group: "'ProcessGroup'", callback_fn: 'Optional[Callable]'=None) ->None:
        """
        Reduce-scatter a list of tensors asynchronously, so smaller reductions
        can be bucketed together. The given callback (``callback_fn``) will be
        called with the reduced result at some later time. Call ``flush()`` to
        force all queued ops and callbacks to be executed.

        Note that large inputs will be reduced immediately, and this function
        may also flush the relevant bucket to make room for ``input_list``.

        Args:
            input_list (List[Tensor]): list of tensors to reduce-scatter. List
                should contain ``group.size()`` tensors and each tensor should
                have identical shape, dtype and device.
            group (ProcessGroup): process group for reduction
            callback_fn (Callable, Optional): callback function to call after
                the reduction executes. Function will be called with a single
                argument corresponding to the reduced result.
        """
        world_size = group.size()
        assert len(input_list) == world_size, f'reduce_scatter received {len(input_list)} inputs, expected group.size() ({world_size})'
        first_input = input_list[0]
        first_input_size = first_input.numel()
        bucket_shard_size = self._get_shard_size(first_input.element_size(), world_size)
        if first_input_size > bucket_shard_size:
            output = torch.zeros_like(input_list[0])
            if hasattr(dist, '_reduce_scatter_base') and enable_nccl_base_collectives:
                input_flattened = torch.cat(input_list)
                dist._reduce_scatter_base(output, input_flattened, group=group)
            else:
                dist.reduce_scatter(output, input_list, group=group)
            if callback_fn is not None:
                callback_fn(output)
            return
        bucket = self._get_bucket(first_input, group)
        if first_input_size > bucket.data.size(1) - bucket.offset:
            bucket.flush()
        stacked_input = torch.stack(input_list).view(world_size, first_input_size)
        offset = bucket.offset
        bucket.data[:, offset:offset + first_input_size].copy_(stacked_input)
        bucket.offset += first_input_size
        if callback_fn is not None:
            result_view = bucket.output_shard[offset:offset + first_input_size].view_as(first_input)
            bucket.callbacks.append(functools.partial(callback_fn, result_view))

    @torch.no_grad()
    def flush(self) ->None:
        """Reduce-scatter any partial buckets."""
        for bucket in self.buckets.values():
            bucket.flush()

    @torch.no_grad()
    def teardown(self) ->None:
        """Free buffers from all buckets."""
        for bucket in self.buckets.values():
            bucket.teardown()

    @functools.lru_cache()
    def _get_shard_size(self, element_size: 'int', num_shards: 'int') ->int:
        if self.bucket_cap_mb <= 0:
            return 0
        MB = 1024 * 1024
        bucket_size = self.bucket_cap_mb * MB / element_size
        return int(bucket_size // num_shards)

    def _get_bucket(self, tensor: 'Tensor', group: "'ProcessGroup'") ->Bucket:
        key = tensor.dtype, tensor.device, group
        if key not in self.buckets:
            world_size = group.size()
            shard_size = self._get_shard_size(tensor.element_size(), world_size)
            data = tensor.new_zeros((world_size, shard_size))
            self.buckets[key] = Bucket(data, group)
        self.buckets[key].setup()
        return self.buckets[key]


class TrainingState(Enum):
    """
    Simple enum to indicate what state FSDP is in. Used for asserting
    to make sure APIs are called in the correct state.

    ..note::

        BACKWARD_PRE and BACKWARD_POST states are used to ensure we
        receives backward hooks in the correct order. It is used to catch
        unexpected order of hooks being called (likely due to our
        hook registration logic or autograd engine logic changes).

    TODO (Min): It would be nice to capture the stepping state as well.
        Maybe we can use the model.zero_grad() call, but not sure if it
        is called if optim.zero_grad() is used instead.
        It would be nice to have clear state transition be explicit like:

        zero_grad -> fwd -> bwd -> optionally accum grad by repeating
        fwd/bwd -> stepping -> loop back to zero_grad
    """
    IDLE = auto()
    FORWARD = auto()
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()
    SUMMON_FULL_PARAMS = auto()


def _clean_path(path: 'str') ->str:
    """Remove FSDP related wrapper modules from a given state dict key str path."""
    return '.'.join([split for split in path.split('.') if split not in {'_fsdp_wrapped_module', '_fpw_module'}])


def _get_default_cuda_device(module: 'nn.Module') ->torch.device:
    """Try to infer CUDA device from module parameters."""
    try:
        compute_device = next(module.parameters()).device
        if compute_device.type == 'cuda':
            return compute_device
    except StopIteration:
        pass
    return torch.device('cuda')


def _unpad(shard: 'torch.Tensor', pad: 'int') ->torch.Tensor:
    if pad > 0:
        shard = shard[:-pad]
    return shard


@torch.no_grad()
def alloc_storage_(data: 'torch.Tensor', size: 'torch.Size') ->None:
    """Allocate storage for a tensor."""
    if data.storage().size() == size.numel():
        return
    assert data.storage().size() == 0
    data.storage().resize_(size.numel())


def apply_to_type(type_fn: 'Callable', fn: 'Callable', container: 'Union[torch.Tensor, np.ndarray, Dict, List, Tuple, Set, NamedTuple]') ->Any:
    """Recursively apply to all objects in different kinds of container types that matches a type function."""

    def _apply(x: 'Union[torch.Tensor, np.ndarray, Dict, List, Tuple, Set]') ->Any:
        if type_fn(x):
            return fn(x)
        elif isinstance(x, OrderedDict):
            od = x.__class__()
            for key, value in x.items():
                od[key] = _apply(value)
            return od
        elif isinstance(x, PackedSequence):
            _apply(x.data)
            return x
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            f = getattr(x, '_fields', None)
            if f is None:
                return tuple(_apply(x) for x in x)
            else:
                assert isinstance(f, tuple), 'This needs to be a namedtuple'
                x = cast(NamedTuple, x)
                _dict: 'Dict[str, Any]' = x._asdict()
                _dict = {key: _apply(value) for key, value in _dict.items()}
                return type(x)(**_dict)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x
    return _apply(container)


def apply_to_tensors(fn: 'Callable', container: 'Union[torch.Tensor, Dict, List, Tuple, Set]') ->Any:
    """Recursively apply to all tensor in different kinds of container types."""
    return apply_to_type(torch.is_tensor, fn, container)


def calc_grad_norm(parameters: 'List[torch.nn.Parameter]', p: 'float') ->torch.Tensor:
    """Calculate gradient norm of an iterable of parameters.
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda par: par.grad is not None, parameters))
    if len(parameters) == 0:
        return torch.tensor(0.0)
    p = float(p)
    if p == inf:
        local_norm = max(par.grad.detach().abs().max() for par in parameters)
    else:
        local_norm = torch.norm(torch.stack([torch.norm(par.grad.detach(), p, dtype=torch.float32) for par in parameters]), p)
    return local_norm


def cast_floats_to_right_precision(to_fp16: 'bool', no_grad: 'bool', *args: Any, **kwargs: Any) ->Tuple[Any, Any]:
    """
    Cast floating point Tensors in *args or **kwargs to FP16 or FP32 if they are not.
    We also retain the requires_grad flag so that casting doesn't affect the autograd graph.
    """

    def fn_fp16(x: 'torch.Tensor') ->torch.Tensor:
        if x.dtype is torch.float32:
            y = x.half()
            if x.is_leaf:
                y.requires_grad = x.requires_grad
            return y
        return x

    def fn_fp32(x: 'torch.Tensor') ->torch.Tensor:
        if x.dtype is torch.float16:
            y = x.float()
            if x.is_leaf:
                y.requires_grad = x.requires_grad
            return y
        return x
    fn = fn_fp16 if to_fp16 else fn_fp32
    context = torch.no_grad() if no_grad else contextlib.suppress()
    with context:
        return apply_to_tensors(fn, args), apply_to_tensors(fn, kwargs)


def chunk_and_pad(tensor: 'torch.Tensor', num_chunks: 'int') ->List[torch.Tensor]:
    """Chunk a given Tensor into num_chunks parts and add any necessary padding."""
    chunks = list(torch.flatten(tensor).chunk(num_chunks))
    num_pad_for_partial_chunk = chunks[0].numel() - chunks[-1].numel()
    if num_pad_for_partial_chunk > 0:
        chunks[-1] = F.pad(chunks[-1], [0, num_pad_for_partial_chunk])
    if len(chunks) < num_chunks:
        chunks.extend([torch.zeros_like(chunks[0]) for _ in range(num_chunks - len(chunks))])
    return chunks


def enable_pytorch_sync_bn(module: 'torch.nn.Module') ->None:
    """Call _specify_ddp_gpu_num for all pytorch SyncBN layers so that it
    is happily running even without DDP. E.g. this is used by FSDP.
    """
    for layer in module.modules():
        if isinstance(layer, torch.nn.modules.SyncBatchNorm) and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)


def free_storage_(data: 'torch.Tensor') ->None:
    """Free underlying storage of a Tensor."""
    if data.storage().size() > 0:
        assert data.storage_offset() == 0
        data.storage().resize_(0)

