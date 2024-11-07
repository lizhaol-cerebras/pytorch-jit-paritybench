
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


import random


from collections import Counter


from collections import OrderedDict


import numpy as np


import torch


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import functools


import time


import math


import itertools


from typing import Callable


from typing import Dict


from typing import Iterable


from typing import List


from typing import Optional


from typing import Tuple


from torch import Tensor


from torch import device


from torch import dtype


from torch import nn


from torch.nn import CrossEntropyLoss


from torch.nn import functional as F


import torch.nn.functional as F


import re


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim as optim


import torch.utils.data


import numpy


import collections


import torch.distributed as dist


import logging


import copy


from torch.nn import MSELoss


from torch.optim import Optimizer


from torch.optim.lr_scheduler import LambdaLR


from torch.nn.parameter import Parameter


from torch.optim.lr_scheduler import _LRScheduler


from enum import Enum


from torch.utils.data import TensorDataset


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data import ConcatDataset


from typing import Union


from logging import getLogger


import warnings


from typing import Any


from torch.utils.data import DistributedSampler


from torch.utils.data import Sampler


from collections import namedtuple


from torch.nn.init import xavier_uniform_


from collections import deque


from torch.utils.data import Subset


from torch.optim import AdamW


from torch.utils.data import BatchSampler


from collections import defaultdict


from torch.utils.data.sampler import BatchSampler


from torch.utils.data.sampler import Sampler


from random import choice


from random import randint


from time import time


import pandas as pd


import torch.utils.checkpoint as checkpoint


from abc import ABCMeta


from abc import abstractmethod


from torch.nn.modules.batchnorm import BatchNorm2d


from torchvision.ops import RoIPool


from torchvision.ops.boxes import batched_nms


from torchvision.ops.boxes import nms


from functools import partial


import matplotlib as mpl


import matplotlib.colors as mplc


import matplotlib.figure as mplfigure


from matplotlib.backends.backend_agg import FigureCanvasAgg


from sklearn.metrics import f1_score


import torchvision


import torchvision.transforms as transforms


from torch import autograd


from torch.nn import init


import torch.utils.data as data


import string


from torch.utils.data.dataloader import DataLoader


import tensorflow as tf


from abc import ABC


from typing import NamedTuple


from typing import NewType


from torch.nn.utils.rnn import pad_sequence


from torch.utils.data.dataset import Dataset


from collections import UserDict


from typing import TYPE_CHECKING


from functools import wraps


from types import ModuleType


from typing import BinaryIO


import inspect


import numbers


from types import SimpleNamespace


from typing import Set


import torch.utils.checkpoint


from collections.abc import Sequence


from torch import _softmax_backward_data


from functools import lru_cache


from torch.nn import LayerNorm


from torch.serialization import default_restore_location


from torch.autograd import Function


from torch.nn import SmoothL1Loss


from functools import reduce


from torch.autograd.function import Function


from torch.nn.modules.container import ModuleDict


import enum


from typing import Generator


from typing import Text


from itertools import groupby


import uuid


from collections.abc import Iterable


from typing import Sequence


from logging import StreamHandler


from torch.utils.data.sampler import RandomSampler


from torch.utils.data.sampler import SequentialSampler


from typing import Iterator


from itertools import takewhile


class LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


def find_pruneable_heads_and_indices(heads: 'List[int]', n_heads: 'int', head_size: 'int', already_pruned_heads: 'Set[int]') ->Tuple[Set[int], torch.LongTensor]:
    """
    Finds the heads and their indices taking :obj:`already_pruned_heads` into account.

    Args:
        heads (:obj:`List[int]`): List of the indices of heads to prune.
        n_heads (:obj:`int`): The number of heads in the model.
        head_size (:obj:`int`): The size of each head.
        already_pruned_heads (:obj:`Set[int]`): A set of already pruned heads.

    Returns:
        :obj:`Tuple[Set[int], torch.LongTensor]`: A tuple with the remaining heads and their corresponding indices.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: 'torch.LongTensor' = torch.arange(len(mask))[mask].long()
    return heads, index


def prune_conv1d_layer(layer: 'Conv1D', index: 'torch.LongTensor', dim: 'int'=1) ->Conv1D:
    """
    Prune a Conv1D layer to keep only entries in index. A Conv1D work as a Linear layer (see e.g. BERT) but the weights
    are transposed.

    Used to remove heads.

    Args:
        layer (:class:`~transformers.modeling_utils.Conv1D`): The layer to prune.
        index (:obj:`torch.LongTensor`): The indices to keep in the layer.
        dim (:obj:`int`, `optional`, defaults to 1): The dimension on which to keep the indices.

    Returns:
        :class:`~transformers.modeling_utils.Conv1D`: The pruned layer as a new layer with :obj:`requires_grad=True`.
    """
    index = index
    W = layer.weight.index_select(dim, index).clone().detach()
    if dim == 0:
        b = layer.bias.clone().detach()
    else:
        b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = Conv1D(new_size[1], new_size[0])
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    new_layer.bias.requires_grad = False
    new_layer.bias.copy_(b.contiguous())
    new_layer.bias.requires_grad = True
    return new_layer


class Attention(nn.Module):

    def __init__(self, nx, n_ctx, config, scale=False):
        super().__init__()
        n_state = nx
        assert n_state % config.n_head == 0
        self.register_buffer('bias', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.n_head, self.split_size // self.n_head, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + 2 * self.split_size])
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)
        self.split_size = self.split_size // self.n_head * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        b = self.bias[:, :, :w.size(-2), :w.size(-1)]
        w = w * b + -10000.0 * (1 - b)
        if attention_mask is not None:
            w = w + attention_mask
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        if head_mask is not None:
            w = w * head_mask
        outputs = [torch.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x, attention_mask=None, head_mask=None, output_attentions=False):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
        a = attn_outputs[0]
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        outputs = [a] + attn_outputs[1:]
        return outputs


def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class MLP(nn.Module):

    def __init__(self, n_state, config):
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT_FNS[config.afn]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):

    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        nx = config.n_embd
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)

    def forward(self, x, attention_mask=None, head_mask=None, output_attentions=False):
        attn_outputs = self.attn(x, attention_mask=attention_mask, head_mask=head_mask, output_attentions=output_attentions)
        a = attn_outputs[0]
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        outputs = [h] + attn_outputs[1:]
        return outputs


def is_tf_available():
    return _tf_available


def is_torch_available():
    return _torch_available


def is_tensor(x):
    """ Tests if ``x`` is a :obj:`torch.Tensor`, :obj:`tf.Tensor` or :obj:`np.ndarray`. """
    if is_torch_available():
        import torch
        if isinstance(x, torch.Tensor):
            return True
    if is_tf_available():
        import tensorflow as tf
        if isinstance(x, tf.Tensor):
            return True
    return isinstance(x, np.ndarray)


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a ``__getitem__`` that allows indexing by integer or slice (like
    a tuple) or strings (like a dictionary) that will ignore the ``None`` attributes. Otherwise behaves like a regular
    python dictionary.

    .. warning::
        You can't unpack a :obj:`ModelOutput` directly. Use the :meth:`~transformers.file_utils.ModelOutput.to_tuple`
        method to convert it to a tuple before.
    """

    def __post_init__(self):
        class_fields = fields(self)
        assert len(class_fields), f'{self.__class__.__name__} has no fields.'
        assert all(field.default is None for field in class_fields[1:]), f'{self.__class__.__name__} should not have more than one required field.'
        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])
        if other_fields_are_none and not is_tensor(first_field):
            try:
                iterator = iter(first_field)
                first_field_iterator = True
            except TypeError:
                first_field_iterator = False
            if first_field_iterator:
                for element in iterator:
                    if not isinstance(element, (list, tuple)) or not len(element) == 2 or not isinstance(element[0], str):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f'You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.')

    def setdefault(self, *args, **kwargs):
        raise Exception(f'You cannot use ``setdefault`` on a {self.__class__.__name__} instance.')

    def pop(self, *args, **kwargs):
        raise Exception(f'You cannot use ``pop`` on a {self.__class__.__name__} instance.')

    def update(self, *args, **kwargs):
        raise Exception(f'You cannot use ``update`` on a {self.__class__.__name__} instance.')

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for k, v in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) ->Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


DEPARALLELIZE_DOCSTRING = """
    Moves the model to cpu from a model parallel state.

    Example::

        # On a 4 GPU machine with t5-3b:
        model = T5ForConditionalGeneration.from_pretrained('t5-3b')
        device_map = {0: [0, 1, 2],

                     1: [3, 4, 5, 6, 7, 8, 9],
                     2: [10, 11, 12, 13, 14, 15, 16],
                     3: [17, 18, 19, 20, 21, 22, 23]}
        model.parallelize(device_map) # Splits the model across several devices
        model.deparallelize() # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
"""


class GPT2Config(object):

    def __init__(self, vocab_size_or_config_json_file=50257, n_positions=1024, n_ctx=1024, n_embd=768, n_layer=12, n_head=12, layer_norm_epsilon=1e-05, initializer_range=0.02, lora_attn_dim=0, lora_attn_alpha=128, lora_dropout=0.0, lora_r_dropout=0.0, fix_dropout=0.0):
        self.vocab_size = vocab_size_or_config_json_file
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.lora_attn_dim = lora_attn_dim
        self.lora_attn_alpha = lora_attn_alpha
        self.lora_dropout = lora_dropout
        self.lora_r_dropout = lora_r_dropout
        self.fix_dropout = fix_dropout


DUMMY_INPUTS = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]


class BeamHypotheses(object):

    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1000000000.0

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


FINALIZE_INPUTS_DOCSTRING = """
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size * num_beams, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using any class inheriting from :class:`~transformers.PretrainedTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        final_beam_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_beams)`):
            The final scores of all non-finished beams.
        final_beam_tokens (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_beams)`):
            The last tokens to be added to the non-finished beam_hypotheses.
        final_beam_indices (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_beams)`):
            The beam indices indicating to which beam the :obj:`final_beam_tokens` shall be added.
        pad_token_id (:obj:`int`, `optional`):
            The id of the `padding` token.
        eos_token_id (:obj:`int`, `optional`):
            The id of the `end-of-sequence` token.

    Return:
        :obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`: The generated
        sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or shorter if all
        batches finished early due to the :obj:`eos_token_id`.

"""


PROCESS_INPUTS_DOCSTRING = """
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size * num_beams, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using any class inheriting from :class:`~transformers.PretrainedTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        next_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2 * num_beams)`):
            Current scores of the top :obj:`2 * num_beams` non-finished beam hypotheses.
        next_tokens (:obj:`torch.LongTensor` of shape :obj:`(batch_size, 2 * num_beams)`):
            :obj:`input_ids` of the tokens corresponding to the top :obj:`2 * num_beams` non-finished beam hypotheses.
        next_indices (:obj:`torch.LongTensor` of shape :obj:`(batch_size, 2 * num_beams)`):
            Beam indices indicating to which beam hypothesis the :obj:`next_tokens` correspond.
        pad_token_id (:obj:`int`, `optional`):
            The id of the `padding` token.
        eos_token_id (:obj:`int`, `optional`):
            The id of the `end-of-sequence` token.

    Return:
        :obj:`UserDict`: A dictionary composed of the fields as defined above:

            - **next_beam_scores** (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_beams)`) -- Updated
              scores of all non-finished beams.
            - **next_beam_tokens** (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_beams)`) -- Next tokens
              to be added to the non-finished beam_hypotheses.
            - **next_beam_indices** (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_beams)`) -- Beam indices
              indicating to which beam the next tokens shall be added.

"""


def add_start_docstrings(*docstr):

    def docstring_decorator(fn):
        fn.__doc__ = ''.join(docstr) + (fn.__doc__ if fn.__doc__ is not None else '')
        return fn
    return docstring_decorator


class BeamScorer(ABC):
    """
    Abstract base class for all beam scorers that are used for :meth:`~transformers.PretrainedModel.beam_search` and
    :meth:`~transformers.PretrainedModel.beam_sample`.
    """

    @abstractmethod
    @add_start_docstrings(PROCESS_INPUTS_DOCSTRING)
    def process(self, input_ids: 'torch.LongTensor', next_scores: 'torch.FloatTensor', next_tokens: 'torch.LongTensor', next_indices: 'torch.LongTensor', **kwargs) ->Tuple[torch.Tensor]:
        raise NotImplementedError('This is an abstract method.')

    @abstractmethod
    @add_start_docstrings(FINALIZE_INPUTS_DOCSTRING)
    def finalize(self, input_ids: 'torch.LongTensor', next_scores: 'torch.FloatTensor', next_tokens: 'torch.LongTensor', next_indices: 'torch.LongTensor', **kwargs) ->torch.LongTensor:
        raise NotImplementedError('This is an abstract method.')


class BeamSearchScorer(BeamScorer):
    """
    :class:`transformers.BeamScorer` implementing standard beam search decoding.

    Adapted in part from `Facebook's XLM beam search code
    <https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529>`__.

    Reference for the diverse beam search algorithm and implementation `Ashwin Kalyan's DBS implementation
    <https://github.com/ashwinkalyan/dbs/blob/master/dbs/beam_utils.lua>`__

    Args:
        batch_size (:obj:`int`):
            Batch Size of :obj:`input_ids` for which standard beam search decoding is run in parallel.
        max_length (:obj:`int`):
            The maximum length of the sequence to be generated.
        num_beams (:obj:`int`):
            Number of beams for beam search.
        device (:obj:`torch.device`):
            Defines the device type (*e.g.*, :obj:`"cpu"` or :obj:`"cuda"`) on which this instance of
            :obj:`BeamSearchScorer` will be allocated.
        length_penalty (:obj:`float`, `optional`, defaults to 1.0):
            Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
            model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
            sequences.
        do_early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
        num_beam_hyps_to_keep (:obj:`int`, `optional`, defaults to 1):
            The number of beam hypotheses that shall be returned upon calling
            :meth:`~transformer.BeamSearchScorer.finalize`.
        num_beam_groups (:obj:`int`):
            Number of groups to divide :obj:`num_beams` into in order to ensure diversity among different groups of
            beams. See `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__ for more details.
    """

    def __init__(self, batch_size: 'int', max_length: 'int', num_beams: 'int', device: 'torch.device', length_penalty: 'Optional[float]'=1.0, do_early_stopping: 'Optional[bool]'=False, num_beam_hyps_to_keep: 'Optional[int]'=1, num_beam_groups: 'Optional[int]'=1):
        self.max_length = max_length
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups
        self._is_init = False
        self._beam_hyps = [BeamHypotheses(num_beams=self.num_beams, max_length=self.max_length, length_penalty=self.length_penalty, early_stopping=self.do_early_stopping) for _ in range(batch_size)]
        self._done = torch.tensor([(False) for _ in range(batch_size)], dtype=torch.bool, device=self.device)
        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(f'`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1, one should make use of `greedy_search` instead.')
        if not isinstance(num_beam_groups, int) or num_beam_groups > num_beams or num_beams % num_beam_groups != 0:
            raise ValueError(f'`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` has to be divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}.')

    @property
    def is_done(self) ->bool:
        return self._done.all()

    def process(self, input_ids: 'torch.LongTensor', next_scores: 'torch.FloatTensor', next_tokens: 'torch.LongTensor', next_indices: 'torch.LongTensor', pad_token_id: 'Optional[int]'=None, eos_token_id: 'Optional[int]'=None) ->Tuple[torch.Tensor]:
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        assert batch_size == input_ids.shape[0] // self.group_size
        device = input_ids.device
        next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                assert len(beam_hyp) >= self.num_beams, 'Batch can only be done if at least {} beams have been generated'.format(self.num_beams)
                assert eos_token_id is not None and pad_token_id is not None, 'generated beams >= num_beams -> eos_token_id and pad_token have to be defined'
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])):
                batch_beam_idx = batch_idx * self.group_size + next_index
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    beam_hyp.add(input_ids[batch_beam_idx].clone(), next_score.item())
                else:
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1
                if beam_idx == self.group_size:
                    break
            if beam_idx < self.group_size:
                raise ValueError(f'At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id: {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected.')
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(next_scores[batch_idx].max().item(), cur_len)
        return UserDict({'next_beam_scores': next_beam_scores.view(-1), 'next_beam_tokens': next_beam_tokens.view(-1), 'next_beam_indices': next_beam_indices.view(-1)})

    def finalize(self, input_ids: 'torch.LongTensor', final_beam_scores: 'torch.FloatTensor', final_beam_tokens: 'torch.LongTensor', final_beam_indices: 'torch.LongTensor', pad_token_id: 'Optional[int]'=None, eos_token_id: 'Optional[int]'=None) ->Tuple[torch.LongTensor]:
        batch_size = len(self._beam_hyps)
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add(final_tokens, final_score)
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=self.device, dtype=torch.float32)
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)
                best.append(best_hyp)
                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score
        sent_max_len = min(sent_lengths.max().item() + 1, self.max_length)
        decoded: 'torch.LongTensor' = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, '`pad_token_id` has to be defined'
            decoded.fill_(pad_token_id)
        for i, hypo in enumerate(best):
            decoded[i, :sent_lengths[i]] = hypo
            if sent_lengths[i] < self.max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
        return UserDict({'sequences': decoded, 'sequence_scores': best_scores})


LOGITS_PROCESSOR_INPUTS_DOCSTRING = """
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax
            or scores for each vocabulary token after SoftMax.
        kwargs:
            Additional stopping critera specific kwargs.

    Return:
        :obj:`bool`. :obj:`False` indicates we should continue, :obj:`True` indicates we should stop.

"""


class LogitsProcessor(ABC):
    """Abstract base class for all logit processors that can be applied during generation."""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: 'torch.LongTensor', scores: 'torch.FloatTensor') ->torch.FloatTensor:
        """Torch method for processing logits."""
        raise NotImplementedError(f'{self.__class__} is an abstract class. Only classes inheriting this class can be called.')


def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    return banned_ngrams.get(ngram_idx, [])


def _get_ngrams(ngram_size: 'int', prev_input_ids: 'torch.Tensor', num_hypos: 'int'):
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
    return generated_ngrams


class EncoderNoRepeatNGramLogitsProcessor(LogitsProcessor):
    """
    :class:`transformers.LogitsProcessor` that enforces no repetition of encoder input ids n-grams for the decoder ids.
    See `ParlAI <https://github.com/facebookresearch/ParlAI/blob/master/parlai/core/torch_generator_agent.py#L1350>`__.

    Args:
        encoder_ngram_size (:obj:`int`):
            All ngrams of size :obj:`ngram_size` can only occur within the encoder input ids.
        encoder_input_ids (:obj:`int`):
            The encoder_input_ids that should not be repeated within the decoder ids.
    """

    def __init__(self, encoder_ngram_size: 'int', encoder_input_ids: 'torch.LongTensor'):
        if not isinstance(encoder_ngram_size, int) or encoder_ngram_size <= 0:
            raise ValueError(f'`encoder_ngram_size` has to be a strictly positive integer, but is {encoder_ngram_size}')
        self.ngram_size = encoder_ngram_size
        if len(encoder_input_ids.shape) == 1:
            encoder_input_ids = encoder_input_ids.unsqueeze(0)
        self.batch_size = encoder_input_ids.shape[0]
        self.generated_ngrams = _get_ngrams(encoder_ngram_size, encoder_input_ids, self.batch_size)

    def __call__(self, input_ids: 'torch.LongTensor', scores: 'torch.FloatTensor') ->torch.FloatTensor:
        num_hypos = scores.shape[0]
        num_beams = num_hypos // self.batch_size
        cur_len = input_ids.shape[-1]
        banned_batch_tokens = [_get_generated_ngrams(self.generated_ngrams[hypo_idx // num_beams], input_ids[hypo_idx], self.ngram_size, cur_len) for hypo_idx in range(num_hypos)]
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float('inf')
        return scores


class ForcedBOSTokenLogitsProcessor(LogitsProcessor):
    """
    :class:`~transformers.LogitsProcessor` that enforces the specified token as the first generated token.

    Args:
        bos_token_id (:obj:`int`):
            The id of the token to force as the first generated token.
    """

    def __init__(self, bos_token_id: 'int'):
        self.bos_token_id = bos_token_id

    def __call__(self, input_ids: 'torch.LongTensor', scores: 'torch.FloatTensor') ->torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len == 1:
            num_tokens = scores.shape[1]
            scores[:, [i for i in range(num_tokens) if i != self.bos_token_id]] = -float('inf')
            scores[:, self.bos_token_id] = 0
        return scores


class ForcedEOSTokenLogitsProcessor(LogitsProcessor):
    """
    :class:`~transformers.LogitsProcessor` that enforces the specified token as the last generated token when
    :obj:`max_length` is reached.

    Args:
        max_length (:obj:`int`):
            The maximum length of the sequence to be generated.
        eos_token_id (:obj:`int`):
            The id of the token to force as the last generated token when :obj:`max_length` is reached.
    """

    def __init__(self, max_length: 'int', eos_token_id: 'int'):
        self.max_length = max_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: 'torch.LongTensor', scores: 'torch.FloatTensor') ->torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len == self.max_length - 1:
            num_tokens = scores.shape[1]
            scores[:, [i for i in range(num_tokens) if i != self.eos_token_id]] = -float('inf')
            scores[:, self.eos_token_id] = 0
        return scores


class HammingDiversityLogitsProcessor(LogitsProcessor):
    """
    :class:`transformers.LogitsProcessor` that enforces diverse beam search. Note that this logits processor is only
    effective for :meth:`transformers.PretrainedModel.group_beam_search`. See `Diverse Beam Search: Decoding Diverse
    Solutions from Neural Sequence Models <https://arxiv.org/pdf/1610.02424.pdf>`__ for more details.

    Args:
        diversity_penalty (:obj:`float`):
            This value is subtracted from a beam's score if it generates a token same as any beam from other group at a
            particular time. Note that :obj:`diversity_penalty` is only effective if ``group beam search`` is enabled.
        num_beams (:obj:`int`):
            Number of beams used for group beam search. See `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__ for
            more details.
        num_beam_groups (:obj:`int`):
            Number of groups to divide :obj:`num_beams` into in order to ensure diversity among different groups of
            beams. See `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__ for more details.
    """

    def __init__(self, diversity_penalty: 'float', num_beams: 'int', num_beam_groups: 'int'):
        if not isinstance(diversity_penalty, float) or not diversity_penalty > 0.0:
            raise ValueError('`diversity_penalty` should be a float strictly larger than 0.')
        self._diversity_penalty = diversity_penalty
        if not isinstance(num_beams, int) or num_beams < 2:
            raise ValueError('`num_beams` should be an integer strictly larger than 1.')
        self._num_beams = num_beams
        if not isinstance(num_beam_groups, int) or num_beam_groups < 2:
            raise ValueError('`num_beam_groups` should be an integer strictly larger than 1.')
        if num_beam_groups > num_beams:
            raise ValueError('`beam_groups` has to be smaller or equal to `num_beams`.')
        self._num_sub_beams = num_beams // num_beam_groups

    def __call__(self, input_ids: 'torch.LongTensor', scores: 'torch.FloatTensor', current_tokens: 'torch.LongTensor', beam_group_idx: 'int') ->torch.FloatTensor:
        batch_size = current_tokens.shape[0] // self._num_beams
        group_start_idx = beam_group_idx * self._num_sub_beams
        group_end_idx = min(group_start_idx + self._num_sub_beams, self._num_beams)
        group_size = group_end_idx - group_start_idx
        vocab_size = scores.shape[-1]
        if group_start_idx == 0:
            return scores
        for batch_idx in range(batch_size):
            previous_group_tokens = current_tokens[batch_idx * self._num_beams:batch_idx * self._num_beams + group_start_idx]
            token_frequency = torch.bincount(previous_group_tokens, minlength=vocab_size)
            scores[batch_idx * group_size:(batch_idx + 1) * group_size] -= self._diversity_penalty * token_frequency
        return scores


class LogitsProcessorList(list):
    """
    This class can be used to create a list of :class:`~transformers.LogitsProcessor` or
    :class:`~transformers.LogitsWarper` to subsequently process a :obj:`scores` input tensor. This class inherits from
    list and adds a specific `__call__` method to apply each :class:`~transformers.LogitsProcessor` or
    :class:`~transformers.LogitsProcessor` to the inputs.
    """

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: 'torch.LongTensor', scores: 'torch.FloatTensor', **kwargs) ->torch.FloatTensor:
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                assert all(arg in kwargs for arg in list(function_args.keys())[2:]), f'Make sure that all the required parameters: {list(function_args.keys())} for {processor.__class__} are passed to the logits processor.'
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)
        return scores


class StoppingCriteria(ABC):
    """Abstract base class for all stopping criteria that can be applied during generation."""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: 'torch.LongTensor', score: 'torch.FloatTensor', **kwargs) ->bool:
        raise NotImplementedError('StoppingCriteria needs to be subclassed')


class MaxLengthCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generated number of tokens exceeds :obj:`max_length`.
    Keep in mind for decoder-only type of transformers, this will include the initial prompted tokens.

    Args:
        max_length (:obj:`int`):
            The maximum length that the output sequence can have in number of tokens.
    """

    def __init__(self, max_length: 'int'):
        self.max_length = max_length

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: 'torch.LongTensor', scores: 'torch.FloatTensor', **kwargs) ->bool:
        return input_ids.shape[-1] > self.max_length


class MaxTimeCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generation exceeds some amount of time. By default, the
    time will start being counted when you initialize this function. You can override this by passing an
    :obj:`initial_time`.

    Args:
        max_time (:obj:`float`):
            The maximum allowed time in seconds for the generation.
        initial_time (:obj:`float`, `optional`, defaults to :obj:`time.time()`):
            The start of the generation allowed time.
    """

    def __init__(self, max_time: 'float', initial_timestamp: 'Optional[float]'=None):
        self.max_time = max_time
        self.initial_timestamp = time.time() if initial_timestamp is None else initial_timestamp

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: 'torch.LongTensor', scores: 'torch.FloatTensor', **kwargs) ->bool:
        return time.time() - self.initial_timestamp > self.max_time


class MinLengthLogitsProcessor(LogitsProcessor):
    """
    :class:`transformers.LogitsProcessor` enforcing a min-length by setting EOS probability to 0.

    Args:
        min_length (:obj:`int`):
            The minimum length below which the score of :obj:`eos_token_id` is set to :obj:`-float("Inf")`.
        eos_token_id (:obj:`int`):
            The id of the `end-of-sequence` token.
    """

    def __init__(self, min_length: 'int', eos_token_id: 'int'):
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f'`min_length` has to be a positive integer, but is {min_length}')
        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(f'`eos_token_id` has to be a positive integer, but is {eos_token_id}')
        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: 'torch.LongTensor', scores: 'torch.FloatTensor') ->torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len < self.min_length:
            scores[:, self.eos_token_id] = -float('inf')
        return scores


class NoBadWordsLogitsProcessor(LogitsProcessor):
    """
    :class:`transformers.LogitsProcessor` that enforces that specified sequences will never be sampled.

    Args:
        bad_words_ids (:obj:`List[List[int]]`):
            List of list of token ids that are not allowed to be generated. In order to get the tokens of the words
            that should not appear in the generated text, use :obj:`tokenizer(bad_word,
            add_prefix_space=True).input_ids`.
        eos_token_id (:obj:`int`):
            The id of the `end-of-sequence` token.
    """

    def __init__(self, bad_words_ids: 'Iterable[Iterable[int]]', eos_token_id: 'int'):
        if not isinstance(bad_words_ids, List) or len(bad_words_ids) == 0:
            raise ValueError(f'`bad_words_ids` has to be a non-emtpy list, but is {bad_words_ids}.')
        if any(not isinstance(bad_word_ids, list) for bad_word_ids in bad_words_ids):
            raise ValueError(f'`bad_words_ids` has to be a list of lists, but is {bad_words_ids}.')
        if any(any(not isinstance(token_id, (int, np.integer)) or token_id < 0 for token_id in bad_word_ids) for bad_word_ids in bad_words_ids):
            raise ValueError(f'Each list in `bad_words_ids` has to be a list of positive integers, but is {bad_words_ids}.')
        self.bad_words_ids = list(filter(lambda bad_token_seq: bad_token_seq != [eos_token_id], bad_words_ids))
        for banned_token_seq in self.bad_words_ids:
            assert len(banned_token_seq) > 0, 'Banned words token sequences {} cannot have an empty list'.format(bad_words_ids)

    def __call__(self, input_ids: 'torch.LongTensor', scores: 'torch.FloatTensor') ->torch.FloatTensor:
        banned_tokens = self._calc_banned_bad_words_ids(input_ids)
        scores = self._set_scores_to_inf_for_banned_tokens(scores, banned_tokens)
        return scores

    def _tokens_match(self, prev_tokens: 'torch.LongTensor', tokens: 'List[int]') ->bool:
        if len(tokens) == 0:
            return True
        elif len(tokens) > len(prev_tokens):
            return False
        elif prev_tokens[-len(tokens):].tolist() == tokens:
            return True
        else:
            return False

    def _calc_banned_bad_words_ids(self, prev_input_ids: 'Iterable[int]') ->Iterable[int]:
        banned_tokens = []
        for prev_input_ids_slice in prev_input_ids:
            banned_tokens_slice = []
            for banned_token_seq in self.bad_words_ids:
                if self._tokens_match(prev_input_ids_slice, banned_token_seq[:-1]) is False:
                    continue
                banned_tokens_slice.append(banned_token_seq[-1])
            banned_tokens.append(banned_tokens_slice)
        return banned_tokens

    def _set_scores_to_inf_for_banned_tokens(self, scores: 'torch.Tensor', banned_tokens: 'List[List[int]]') ->None:
        """
        Modifies the scores in place by setting the banned token positions to `-inf`. Banned token is expected to be a
        list of list of banned tokens to ban in the format [[batch index, vocabulary position],...

        Args:
            scores: logits distribution of shape (batch size, vocabulary size)
            banned_tokens: list of list of tokens to ban of length (batch_size)
        """
        banned_mask_list = []
        for idx, batch_banned_tokens in enumerate(banned_tokens):
            for token in batch_banned_tokens:
                banned_mask_list.append([idx, token])
        if not banned_mask_list:
            return scores
        banned_mask = torch.LongTensor(banned_mask_list)
        indices = torch.ones(len(banned_mask))
        banned_mask = torch.sparse.LongTensor(banned_mask.t(), indices, scores.size()).to_dense().bool()
        scores = scores.masked_fill(banned_mask, -float('inf'))
        return scores


def _calc_banned_ngram_tokens(ngram_size: 'int', prev_input_ids: 'torch.Tensor', num_hypos: 'int', cur_len: 'int') ->List[Iterable[int]]:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < ngram_size:
        return [[] for _ in range(num_hypos)]
    generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos)
    banned_tokens = [_get_generated_ngrams(generated_ngrams[hypo_idx], prev_input_ids[hypo_idx], ngram_size, cur_len) for hypo_idx in range(num_hypos)]
    return banned_tokens


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    """
    :class:`transformers.LogitsProcessor` that enforces no repetition of n-grams. See `Fairseq
    <https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345>`__.

    Args:
        ngram_size (:obj:`int`):
            All ngrams of size :obj:`ngram_size` can only occur once.
    """

    def __init__(self, ngram_size: 'int'):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f'`ngram_size` has to be a strictly positive integer, but is {ngram_size}')
        self.ngram_size = ngram_size

    def __call__(self, input_ids: 'torch.LongTensor', scores: 'torch.FloatTensor') ->torch.FloatTensor:
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]
        banned_batch_tokens = _calc_banned_ngram_tokens(self.ngram_size, input_ids, num_batch_hypotheses, cur_len)
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float('inf')
        return scores


class PrefixConstrainedLogitsProcessor(LogitsProcessor):
    """
    :class:`transformers.LogitsProcessor` that enforces contrained generation and is useful for prefix-conditioned
    constrained generation. See `Autoregressive Entity Retrieval <https://arxiv.org/abs/2010.00904>`__ for more
    information.

    Args:
        prefix_allowed_tokens_fn: (:obj:`Callable[[int, torch.Tensor], List[int]]`):
            This function constraints the beam search to allowed tokens only at each step. This function takes 2
            arguments :obj:`inputs_ids` and the batch ID :obj:`batch_id`. It has to return a list with the allowed
            tokens for the next generation step conditioned on the previously generated tokens :obj:`inputs_ids` and
            the batch ID :obj:`batch_id`.
    """

    def __init__(self, prefix_allowed_tokens_fn: 'Callable[[int, torch.Tensor], List[int]]', num_beams: 'int'):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams

    def __call__(self, input_ids: 'torch.LongTensor', scores: 'torch.FloatTensor') ->torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                mask[batch_id * self._num_beams + beam_id, self._prefix_allowed_tokens_fn(batch_id, sent)] = 0
        return scores + mask


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    """
    :class:`transformers.LogitsProcessor` enforcing an exponential penalty on repeated sequences.

    Args:
        repetition_penalty (:obj:`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See `this paper
            <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
    """

    def __init__(self, penalty: 'float'):
        if not isinstance(penalty, float) or not penalty > 0:
            raise ValueError(f'`penalty` has to be a strictly positive float, but is {penalty}')
        self.penalty = penalty

    def __call__(self, input_ids: 'torch.LongTensor', scores: 'torch.FloatTensor') ->torch.FloatTensor:
        score = torch.gather(scores, 1, input_ids)
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
        scores.scatter_(1, input_ids, score)
        return scores


class StoppingCriteriaList(list):

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: 'torch.LongTensor', scores: 'torch.FloatTensor', **kwargs) ->bool:
        return any(criteria(input_ids, scores) for criteria in self)


class LogitsWarper(ABC):
    """Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: 'torch.LongTensor', scores: 'torch.FloatTensor') ->torch.FloatTensor:
        """Torch method for warping logits."""
        raise NotImplementedError(f'{self.__class__} is an abstract class. Only classes inheriting this class can be called.')


class TemperatureLogitsWarper(LogitsWarper):
    """
    :class:`transformers.LogitsWarper` for temperature (exponential scaling output probability distribution).

    Args:
        temperature (:obj:`float`):
            The value used to module the logits distribution.
    """

    def __init__(self, temperature: 'float'):
        if not isinstance(temperature, float) or not temperature > 0:
            raise ValueError(f'`temperature` has to be a strictly positive float, but is {temperature}')
        self.temperature = temperature

    def __call__(self, input_ids: 'torch.Tensor', scores: 'torch.Tensor') ->torch.Tensor:
        scores = scores / self.temperature
        return scores


class TopKLogitsWarper(LogitsWarper):
    """
    :class:`transformers.LogitsWarper` that performs top-k, i.e. restricting to the k highest probability elements.

    Args:
        top_k (:obj:`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (:obj:`float`, `optional`, defaults to :obj:`-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (:obj:`int`, `optional`, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_k: 'int', filter_value: 'float'=-float('Inf'), min_tokens_to_keep: 'int'=1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f'`top_k` has to be a strictly positive integer, but is {top_k}')
        self.top_k = top_k
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: 'torch.LongTensor', scores: 'torch.FloatTensor') ->torch.FloatTensor:
        top_k = min(max(self.top_k, self.min_tokens_to_keep), scores.size(-1))
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TopPLogitsWarper(LogitsWarper):
    """
    :class:`transformers.LogitsWarper` that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <=
    prob_cut_off.

    Args:
        top_p (:obj:`float`):
            If set to < 1, only the most probable tokens with probabilities that add up to :obj:`top_p` or higher are
            kept for generation.
        filter_value (:obj:`float`, `optional`, defaults to :obj:`-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (:obj:`int`, `optional`, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_p: 'float', filter_value: 'float'=-float('Inf'), min_tokens_to_keep: 'int'=1):
        if not isinstance(top_p, float) or (top_p < 0 or top_p > 1.0):
            raise ValueError(f'`top_p` has to be a float > 0 and < 1, but is {top_p}')
        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: 'torch.LongTensor', scores: 'torch.FloatTensor') ->torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        if self.min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :self.min_tokens_to_keep - 1] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


def validate_stopping_criteria(stopping_criteria: 'StoppingCriteriaList', max_length: 'int'):
    found = False
    for stopping_criterium in stopping_criteria:
        if isinstance(stopping_criterium, MaxLengthCriteria):
            found = True
            if stopping_criterium.max_length != max_length:
                warnings.warn('You set different `max_length` for stopping criteria and `max_length` parameter', UserWarning)
    if not found:
        stopping_criteria.append(MaxLengthCriteria(max_length=max_length))


def get_parameter_device(parameter: "Union[nn.Module, GenerationMixin, 'ModuleUtilsMixin']"):
    try:
        return next(parameter.parameters()).device
    except StopIteration:

        def find_tensor_attributes(module: 'nn.Module') ->List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples
        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device


def get_parameter_dtype(parameter: "Union[nn.Module, GenerationMixin, 'ModuleUtilsMixin']"):
    try:
        return next(parameter.parameters()).dtype
    except StopIteration:

        def find_tensor_attributes(module: 'nn.Module') ->List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples
        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype


CONFIG_NAME = 'config.json'


def http_get(url: 'str', temp_file: 'BinaryIO', proxies=None, resume_size=0, headers: 'Optional[Dict[str, str]]'=None):
    """
    Donwload remote file. Do not gobble up errors.
    """
    headers = copy.deepcopy(headers)
    if resume_size > 0:
        headers['Range'] = 'bytes=%d-' % (resume_size,)
    r = requests.get(url, stream=True, proxies=proxies, headers=headers)
    r.raise_for_status()
    content_length = r.headers.get('Content-Length')
    total = resume_size + int(content_length) if content_length is not None else None
    progress = tqdm(unit='B', unit_scale=True, total=total, initial=resume_size, desc='Downloading', disable=bool(logging.get_verbosity() == logging.NOTSET))
    for chunk in r.iter_content(chunk_size=1024):
        if chunk:
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def define_sagemaker_information():
    try:
        instance_data = requests.get(os.environ['ECS_CONTAINER_METADATA_URI']).json()
        dlc_container_used = instance_data['Image']
        dlc_tag = instance_data['Image'].split(':')[1]
    except Exception:
        dlc_container_used = None
        dlc_tag = None
    sagemaker_params = json.loads(os.getenv('SM_FRAMEWORK_PARAMS', '{}'))
    runs_distributed_training = True if 'sagemaker_distributed_dataparallel_enabled' in sagemaker_params else False
    account_id = os.getenv('TRAINING_JOB_ARN').split(':')[4] if 'TRAINING_JOB_ARN' in os.environ else None
    sagemaker_object = {'sm_framework': os.getenv('SM_FRAMEWORK_MODULE', None), 'sm_region': os.getenv('AWS_REGION', None), 'sm_number_gpu': os.getenv('SM_NUM_GPUS', 0), 'sm_number_cpu': os.getenv('SM_NUM_CPUS', 0), 'sm_distributed_training': runs_distributed_training, 'sm_deep_learning_container': dlc_container_used, 'sm_deep_learning_container_tag': dlc_tag, 'sm_account_id': account_id}
    return sagemaker_object


def is_training_run_on_sagemaker():
    return 'SAGEMAKER_JOB_NAME' in os.environ and not DISABLE_TELEMETRY


def http_user_agent(user_agent: 'Union[Dict, str, None]'=None) ->str:
    """
    Formats a user-agent string with basic info about a request.
    """
    ua = 'transformers/{}; python/{}'.format(__version__, sys.version.split()[0])
    if is_torch_available():
        ua += f'; torch/{_torch_version}'
    if is_tf_available():
        ua += f'; tensorflow/{_tf_version}'
    if is_training_run_on_sagemaker():
        ua += '; ' + '; '.join(f'{k}/{v}' for k, v in define_sagemaker_information().items())
    if isinstance(user_agent, dict):
        ua += '; ' + '; '.join(f'{k}/{v}' for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += '; ' + user_agent
    return ua


def url_to_filename(url: 'str', etag: 'Optional[str]'=None) ->str:
    """
    Convert `url` into a hashed filename in a repeatable way. If `etag` is specified, append its hash to the url's,
    delimited by a period. If the url ends with .h5 (Keras HDF5 weights) adds '.h5' to the name so that TF 2.0 can
    identify it as a HDF5 file (see
    https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1380)
    """
    url_bytes = url.encode('utf-8')
    filename = sha256(url_bytes).hexdigest()
    if etag:
        etag_bytes = etag.encode('utf-8')
        filename += '.' + sha256(etag_bytes).hexdigest()
    if url.endswith('.h5'):
        filename += '.h5'
    return filename


def get_from_cache(url: 'str', cache_dir=None, force_download=False, proxies=None, etag_timeout=10, resume_download=False, user_agent: 'Union[Dict, str, None]'=None, use_auth_token: 'Union[bool, str, None]'=None, local_files_only=False) ->Optional[str]:
    """
    Given a URL, look for the corresponding file in the local cache. If it's not there, download it. Then return the
    path to the cached file.

    Return:
        Local path (string) of file or if networking is off, last version of file cached on disk.

    Raises:
        In case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    headers = {'user-agent': http_user_agent(user_agent)}
    if isinstance(use_auth_token, str):
        headers['authorization'] = 'Bearer {}'.format(use_auth_token)
    elif use_auth_token:
        token = HfFolder.get_token()
        if token is None:
            raise EnvironmentError('You specified use_auth_token=True, but a huggingface token was not found.')
        headers['authorization'] = 'Bearer {}'.format(token)
    url_to_download = url
    etag = None
    if not local_files_only:
        try:
            r = requests.head(url, headers=headers, allow_redirects=False, proxies=proxies, timeout=etag_timeout)
            r.raise_for_status()
            etag = r.headers.get('X-Linked-Etag') or r.headers.get('ETag')
            if etag is None:
                raise OSError("Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility.")
            if 300 <= r.status_code <= 399:
                url_to_download = r.headers['Location']
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass
    filename = url_to_filename(url, etag)
    cache_path = os.path.join(cache_dir, filename)
    if etag is None:
        if os.path.exists(cache_path):
            return cache_path
        else:
            matching_files = [file for file in fnmatch.filter(os.listdir(cache_dir), filename.split('.')[0] + '.*') if not file.endswith('.json') and not file.endswith('.lock')]
            if len(matching_files) > 0:
                return os.path.join(cache_dir, matching_files[-1])
            elif local_files_only:
                raise FileNotFoundError("Cannot find the requested files in the cached path and outgoing traffic has been disabled. To enable model look-ups and downloads online, set 'local_files_only' to False.")
            else:
                raise ValueError('Connection error, and we cannot find the requested files in the cached path. Please try again or make sure your Internet connection is on.')
    if os.path.exists(cache_path) and not force_download:
        return cache_path
    lock_path = cache_path + '.lock'
    with FileLock(lock_path):
        if os.path.exists(cache_path) and not force_download:
            return cache_path
        if resume_download:
            incomplete_path = cache_path + '.incomplete'

            @contextmanager
            def _resumable_file_manager() ->'io.BufferedWriter':
                with open(incomplete_path, 'ab') as f:
                    yield f
            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(tempfile.NamedTemporaryFile, mode='wb', dir=cache_dir, delete=False)
            resume_size = 0
        with temp_file_manager() as temp_file:
            logger.info('%s not found in cache or force_download set to True, downloading to %s', url, temp_file.name)
            http_get(url_to_download, temp_file, proxies=proxies, resume_size=resume_size, headers=headers)
        logger.info('storing %s in cache at %s', url, cache_path)
        os.replace(temp_file.name, cache_path)
        logger.info('creating metadata file for %s', cache_path)
        meta = {'url': url, 'etag': etag}
        meta_path = cache_path + '.json'
        with open(meta_path, 'w') as meta_file:
            json.dump(meta, meta_file)
    return cache_path


ENV_VARS_TRUE_VALUES = {'1', 'ON', 'YES', 'TRUE'}


def is_offline_mode():
    return _is_offline_mode


def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ('http', 'https')


def cached_path(url_or_filename, cache_dir=None, force_download=False, proxies=None, resume_download=False, user_agent: 'Union[Dict, str, None]'=None, extract_compressed_file=False, force_extract=False, use_auth_token: 'Union[bool, str, None]'=None, local_files_only=False) ->Optional[str]:
    """
    Given something that might be a URL (or might be a local path), determine which. If it's a URL, download the file
    and cache it, and return the path to the cached file. If it's already a local path, make sure the file exists and
    then return the path

    Args:
        cache_dir: specify a cache directory to save the file to (overwrite the default cache dir).
        force_download: if True, re-download the file even if it's already cached in the cache dir.
        resume_download: if True, resume the download if incompletely received file is found.
        user_agent: Optional string or dict that will be appended to the user-agent on remote requests.
        use_auth_token: Optional string or boolean to use as Bearer token for remote files. If True,
            will get token from ~/.huggingface.
        extract_compressed_file: if True and the path point to a zip or tar file, extract the compressed
            file in a folder along the archive.
        force_extract: if True when extract_compressed_file is True and the archive was already extracted,
            re-extract the archive and override the folder where it was extracted.

    Return:
        Local path (string) of file or if networking is off, last version of file cached on disk.

    Raises:
        In case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if is_offline_mode() and not local_files_only:
        logger.info('Offline mode: forcing local_files_only=True')
        local_files_only = True
    if is_remote_url(url_or_filename):
        output_path = get_from_cache(url_or_filename, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, user_agent=user_agent, use_auth_token=use_auth_token, local_files_only=local_files_only)
    elif os.path.exists(url_or_filename):
        output_path = url_or_filename
    elif urlparse(url_or_filename).scheme == '':
        raise EnvironmentError('file {} not found'.format(url_or_filename))
    else:
        raise ValueError('unable to parse {} as a URL or as a local path'.format(url_or_filename))
    if extract_compressed_file:
        if not is_zipfile(output_path) and not tarfile.is_tarfile(output_path):
            return output_path
        output_dir, output_file = os.path.split(output_path)
        output_extract_dir_name = output_file.replace('.', '-') + '-extracted'
        output_path_extracted = os.path.join(output_dir, output_extract_dir_name)
        if os.path.isdir(output_path_extracted) and os.listdir(output_path_extracted) and not force_extract:
            return output_path_extracted
        lock_path = output_path + '.lock'
        with FileLock(lock_path):
            shutil.rmtree(output_path_extracted, ignore_errors=True)
            os.makedirs(output_path_extracted)
            if is_zipfile(output_path):
                with ZipFile(output_path, 'r') as zip_file:
                    zip_file.extractall(output_path_extracted)
                    zip_file.close()
            elif tarfile.is_tarfile(output_path):
                tar_file = tarfile.open(output_path)
                tar_file.extractall(output_path_extracted)
                tar_file.close()
            else:
                raise EnvironmentError('Archive format of {} could not be identified'.format(output_path))
        return output_path_extracted
    return output_path


HUGGINGFACE_CO_PREFIX = 'https://huggingface.co/{model_id}/resolve/{revision}/{filename}'


PRESET_MIRROR_DICT = {'tuna': 'https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models', 'bfsu': 'https://mirrors.bfsu.edu.cn/hugging-face-models'}


def hf_bucket_url(model_id: 'str', filename: 'str', subfolder: 'Optional[str]'=None, revision: 'Optional[str]'=None, mirror=None) ->str:
    """
    Resolve a model identifier, a file name, and an optional revision id, to a huggingface.co-hosted url, redirecting
    to Cloudfront (a Content Delivery Network, or CDN) for large files.

    Cloudfront is replicated over the globe so downloads are way faster for the end user (and it also lowers our
    bandwidth costs).

    Cloudfront aggressively caches files by default (default TTL is 24 hours), however this is not an issue here
    because we migrated to a git-based versioning system on huggingface.co, so we now store the files on S3/Cloudfront
    in a content-addressable way (i.e., the file name is its hash). Using content-addressable filenames means cache
    can't ever be stale.

    In terms of client-side caching from this library, we base our caching on the objects' ETag. An object' ETag is:
    its sha1 if stored in git, or its sha256 if stored in git-lfs. Files cached locally from transformers before v3.5.0
    are not shared with those new files, because the cached file's name contains a hash of the url (which changed).
    """
    if subfolder is not None:
        filename = f'{subfolder}/{filename}'
    if mirror:
        endpoint = PRESET_MIRROR_DICT.get(mirror, mirror)
        legacy_format = '/' not in model_id
        if legacy_format:
            return f'{endpoint}/{model_id}-{filename}'
        else:
            return f'{endpoint}/{model_id}/{filename}'
    if revision is None:
        revision = 'main'
    return HUGGINGFACE_CO_PREFIX.format(model_id=model_id, revision=revision, filename=filename)


class PretrainedConfig(object):
    """
    Base class for all configuration classes. Handles a few parameters common to all models' configurations as well as
    methods for loading/downloading/saving configurations.

    Note: A configuration file can be loaded and saved to disk. Loading the configuration file and using this file to
    initialize a model does **not** load the model weights. It only affects the model's configuration.

    Class attributes (overridden by derived classes)

        - **model_type** (:obj:`str`): An identifier for the model type, serialized into the JSON file, and used to
          recreate the correct object in :class:`~transformers.AutoConfig`.
        - **is_composition** (:obj:`bool`): Whether the config class is composed of multiple sub-configs. In this case
          the config has to be initialized from two or more configs of type :class:`~transformers.PretrainedConfig`
          like: :class:`~transformers.EncoderDecoderConfig` or :class:`~RagConfig`.
        - **keys_to_ignore_at_inference** (:obj:`List[str]`): A list of keys to ignore by default when looking at
          dictionary outputs of the model during inference.

    Args:
        name_or_path (:obj:`str`, `optional`, defaults to :obj:`""`):
            Store the string that was passed to :func:`~transformers.PreTrainedModel.from_pretrained` or
            :func:`~transformers.TFPreTrainedModel.from_pretrained` as ``pretrained_model_name_or_path`` if the
            configuration was created with such a method.
        output_hidden_states (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the model should return all hidden-states.
        output_attentions (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the model should returns all attentions.
        return_dict (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return a :class:`~transformers.file_utils.ModelOutput` instead of a plain
            tuple.
        is_encoder_decoder (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether the model is used as an encoder/decoder or not.
        is_decoder (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether the model is used as decoder or not (in which case it's used as an encoder).
        add_cross_attention (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether cross-attention layers should be added to the model. Note, this option is only relevant for models
            that can be used as decoder models within the `:class:~transformers.EncoderDecoderModel` class, which
            consists of all models in ``AUTO_MODELS_FOR_CAUSAL_LM``.
        tie_encoder_decoder (:obj:`bool`, `optional`, defaults to :obj:`False`)
            Whether all encoder weights should be tied to their equivalent decoder weights. This requires the encoder
            and decoder model to have the exact same parameter names.
        prune_heads (:obj:`Dict[int, List[int]]`, `optional`, defaults to :obj:`{}`):
            Pruned heads of the model. The keys are the selected layer indices and the associated values, the list of
            heads to prune in said layer.

            For instance ``{1: [0, 2], 2: [2, 3]}`` will prune heads 0 and 2 on layer 1 and heads 2 and 3 on layer 2.
        chunk_size_feed_forward (:obj:`int`, `optional`, defaults to :obj:`0`):
            The chunk size of all feed forward layers in the residual attention blocks. A chunk size of :obj:`0` means
            that the feed forward layer is not chunked. A chunk size of n means that the feed forward layer processes
            :obj:`n` < sequence_length embeddings at a time. For more information on feed forward chunking, see `How
            does Feed Forward Chunking work? <../glossary.html#feed-forward-chunking>`__ .

    Parameters for sequence generation

        - **max_length** (:obj:`int`, `optional`, defaults to 20) -- Maximum length that will be used by default in the
          :obj:`generate` method of the model.
        - **min_length** (:obj:`int`, `optional`, defaults to 10) -- Minimum length that will be used by default in the
          :obj:`generate` method of the model.
        - **do_sample** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Flag that will be used by default in the
          :obj:`generate` method of the model. Whether or not to use sampling ; use greedy decoding otherwise.
        - **early_stopping** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Flag that will be used by default
          in the :obj:`generate` method of the model. Whether to stop the beam search when at least ``num_beams``
          sentences are finished per batch or not.
        - **num_beams** (:obj:`int`, `optional`, defaults to 1) -- Number of beams for beam search that will be used by
          default in the :obj:`generate` method of the model. 1 means no beam search.
        - **num_beam_groups** (:obj:`int`, `optional`, defaults to 1) -- Number of groups to divide :obj:`num_beams`
          into in order to ensure diversity among different groups of beams that will be used by default in the
          :obj:`generate` method of the model. 1 means no group beam search.
        - **diversity_penalty** (:obj:`float`, `optional`, defaults to 0.0) -- Value to control diversity for group
          beam search. that will be used by default in the :obj:`generate` method of the model. 0 means no diversity
          penalty. The higher the penalty, the more diverse are the outputs.
        - **temperature** (:obj:`float`, `optional`, defaults to 1) -- The value used to module the next token
          probabilities that will be used by default in the :obj:`generate` method of the model. Must be strictly
          positive.
        - **top_k** (:obj:`int`, `optional`, defaults to 50) -- Number of highest probability vocabulary tokens to keep
          for top-k-filtering that will be used by default in the :obj:`generate` method of the model.
        - **top_p** (:obj:`float`, `optional`, defaults to 1) -- Value that will be used by default in the
          :obj:`generate` method of the model for ``top_p``. If set to float < 1, only the most probable tokens with
          probabilities that add up to ``top_p`` or higher are kept for generation.
        - **repetition_penalty** (:obj:`float`, `optional`, defaults to 1) -- Parameter for repetition penalty that
          will be used by default in the :obj:`generate` method of the model. 1.0 means no penalty.
        - **length_penalty** (:obj:`float`, `optional`, defaults to 1) -- Exponential penalty to the length that will
          be used by default in the :obj:`generate` method of the model.
        - **no_repeat_ngram_size** (:obj:`int`, `optional`, defaults to 0) -- Value that will be used by default in the
          :obj:`generate` method of the model for ``no_repeat_ngram_size``. If set to int > 0, all ngrams of that size
          can only occur once.
        - **encoder_no_repeat_ngram_size** (:obj:`int`, `optional`, defaults to 0) -- Value that will be used by
          default in the :obj:`generate` method of the model for ``encoder_no_repeat_ngram_size``. If set to int > 0,
          all ngrams of that size that occur in the ``encoder_input_ids`` cannot occur in the ``decoder_input_ids``.
        - **bad_words_ids** (:obj:`List[int]`, `optional`) -- List of token ids that are not allowed to be generated
          that will be used by default in the :obj:`generate` method of the model. In order to get the tokens of the
          words that should not appear in the generated text, use :obj:`tokenizer.encode(bad_word,
          add_prefix_space=True)`.
        - **num_return_sequences** (:obj:`int`, `optional`, defaults to 1) -- Number of independently computed returned
          sequences for each element in the batch that will be used by default in the :obj:`generate` method of the
          model.
        - **output_scores** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Whether the model should return the
          logits when used for generation
        - **return_dict_in_generate** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Whether the model should
          return a :class:`~transformers.file_utils.ModelOutput` instead of a :obj:`torch.LongTensor`
        - **forced_bos_token_id** (:obj:`int`, `optional`) -- The id of the token to force as the first generated token
          after the :obj:`decoder_start_token_id`. Useful for multilingual models like :doc:`mBART
          <../model_doc/mbart>` where the first generated token needs to be the target language token.
        - **forced_eos_token_id** (:obj:`int`, `optional`) -- The id of the token to force as the last generated token
          when :obj:`max_length` is reached.


    Parameters for fine-tuning tasks

        - **architectures** (:obj:`List[str]`, `optional`) -- Model architectures that can be used with the model
          pretrained weights.
        - **finetuning_task** (:obj:`str`, `optional`) -- Name of the task used to fine-tune the model. This can be
          used when converting from an original (TensorFlow or PyTorch) checkpoint.
        - **id2label** (:obj:`Dict[int, str]`, `optional`) -- A map from index (for instance prediction index, or
          target index) to label.
        - **label2id** (:obj:`Dict[str, int]`, `optional`) -- A map from label to index for the model.
        - **num_labels** (:obj:`int`, `optional`) -- Number of labels to use in the last layer added to the model,
          typically for a classification task.
        - **task_specific_params** (:obj:`Dict[str, Any]`, `optional`) -- Additional keyword arguments to store for the
          current task.

    Parameters linked to the tokenizer

        - **tokenizer_class** (:obj:`str`, `optional`) -- The name of the associated tokenizer class to use (if none is
          set, will use the tokenizer associated to the model by default).
        - **prefix** (:obj:`str`, `optional`) -- A specific prompt that should be added at the beginning of each text
          before calling the model.
        - **bos_token_id** (:obj:`int`, `optional`)) -- The id of the `beginning-of-stream` token.
        - **pad_token_id** (:obj:`int`, `optional`)) -- The id of the `padding` token.
        - **eos_token_id** (:obj:`int`, `optional`)) -- The id of the `end-of-stream` token.
        - **decoder_start_token_id** (:obj:`int`, `optional`)) -- If an encoder-decoder model starts decoding with a
          different token than `bos`, the id of that token.
        - **sep_token_id** (:obj:`int`, `optional`)) -- The id of the `separation` token.

    PyTorch specific parameters

        - **torchscript** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Whether or not the model should be
          used with Torchscript.
        - **tie_word_embeddings** (:obj:`bool`, `optional`, defaults to :obj:`True`) -- Whether the model's input and
          output word embeddings should be tied. Note that this is only relevant if the model has a output word
          embedding layer.

    TensorFlow specific parameters

        - **use_bfloat16** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Whether or not the model should use
          BFloat16 scalars (only used by some TensorFlow models).
    """
    model_type: 'str' = ''
    is_composition: 'bool' = False

    def __init__(self, **kwargs):
        self.return_dict = kwargs.pop('return_dict', True)
        self.output_hidden_states = kwargs.pop('output_hidden_states', False)
        self.output_attentions = kwargs.pop('output_attentions', False)
        self.torchscript = kwargs.pop('torchscript', False)
        self.use_bfloat16 = kwargs.pop('use_bfloat16', False)
        self.pruned_heads = kwargs.pop('pruned_heads', {})
        self.tie_word_embeddings = kwargs.pop('tie_word_embeddings', True)
        self.is_encoder_decoder = kwargs.pop('is_encoder_decoder', False)
        self.is_decoder = kwargs.pop('is_decoder', False)
        self.add_cross_attention = kwargs.pop('add_cross_attention', False)
        self.tie_encoder_decoder = kwargs.pop('tie_encoder_decoder', False)
        self.max_length = kwargs.pop('max_length', 20)
        self.min_length = kwargs.pop('min_length', 0)
        self.do_sample = kwargs.pop('do_sample', False)
        self.early_stopping = kwargs.pop('early_stopping', False)
        self.num_beams = kwargs.pop('num_beams', 1)
        self.num_beam_groups = kwargs.pop('num_beam_groups', 1)
        self.diversity_penalty = kwargs.pop('diversity_penalty', 0.0)
        self.temperature = kwargs.pop('temperature', 1.0)
        self.top_k = kwargs.pop('top_k', 50)
        self.top_p = kwargs.pop('top_p', 1.0)
        self.repetition_penalty = kwargs.pop('repetition_penalty', 1.0)
        self.length_penalty = kwargs.pop('length_penalty', 1.0)
        self.no_repeat_ngram_size = kwargs.pop('no_repeat_ngram_size', 0)
        self.encoder_no_repeat_ngram_size = kwargs.pop('encoder_no_repeat_ngram_size', 0)
        self.bad_words_ids = kwargs.pop('bad_words_ids', None)
        self.num_return_sequences = kwargs.pop('num_return_sequences', 1)
        self.chunk_size_feed_forward = kwargs.pop('chunk_size_feed_forward', 0)
        self.output_scores = kwargs.pop('output_scores', False)
        self.return_dict_in_generate = kwargs.pop('return_dict_in_generate', False)
        self.forced_bos_token_id = kwargs.pop('forced_bos_token_id', None)
        self.forced_eos_token_id = kwargs.pop('forced_eos_token_id', None)
        self.architectures = kwargs.pop('architectures', None)
        self.finetuning_task = kwargs.pop('finetuning_task', None)
        self.id2label = kwargs.pop('id2label', None)
        self.label2id = kwargs.pop('label2id', None)
        if self.id2label is not None:
            kwargs.pop('num_labels', None)
            self.id2label = dict((int(key), value) for key, value in self.id2label.items())
        else:
            self.num_labels = kwargs.pop('num_labels', 2)
        self.tokenizer_class = kwargs.pop('tokenizer_class', None)
        self.prefix = kwargs.pop('prefix', None)
        self.bos_token_id = kwargs.pop('bos_token_id', None)
        self.pad_token_id = kwargs.pop('pad_token_id', None)
        self.eos_token_id = kwargs.pop('eos_token_id', None)
        self.sep_token_id = kwargs.pop('sep_token_id', None)
        self.decoder_start_token_id = kwargs.pop('decoder_start_token_id', None)
        self.task_specific_params = kwargs.pop('task_specific_params', None)
        if kwargs.pop('xla_device', None) is not None:
            logger.warn('The `xla_device` argument has been deprecated in v4.4.0 of Transformers. It is ignored and you can safely remove it from your `config.json` file.')
        self._name_or_path = str(kwargs.pop('name_or_path', ''))
        kwargs.pop('transformers_version', None)
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err

    @property
    def name_or_path(self) ->str:
        return self._name_or_path

    @name_or_path.setter
    def name_or_path(self, value):
        self._name_or_path = str(value)

    @property
    def use_return_dict(self) ->bool:
        """
        :obj:`bool`: Whether or not return :class:`~transformers.file_utils.ModelOutput` instead of tuples.
        """
        return self.return_dict and not self.torchscript

    @property
    def num_labels(self) ->int:
        """
        :obj:`int`: The number of labels for classification models.
        """
        return len(self.id2label)

    @num_labels.setter
    def num_labels(self, num_labels: 'int'):
        if self.id2label is None or len(self.id2label) != num_labels:
            self.id2label = {i: 'LABEL_{}'.format(i) for i in range(num_labels)}
            self.label2id = dict(zip(self.id2label.values(), self.id2label.keys()))

    def save_pretrained(self, save_directory: 'Union[str, os.PathLike]'):
        """
        Save a configuration object to the directory ``save_directory``, so that it can be re-loaded using the
        :func:`~transformers.PretrainedConfig.from_pretrained` class method.

        Args:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
        """
        if os.path.isfile(save_directory):
            raise AssertionError('Provided path ({}) should be a directory, not a file'.format(save_directory))
        os.makedirs(save_directory, exist_ok=True)
        output_config_file = os.path.join(save_directory, CONFIG_NAME)
        self.to_json_file(output_config_file, use_diff=True)
        logger.info(f'Configuration saved in {output_config_file}')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: 'Union[str, os.PathLike]', **kwargs) ->'PretrainedConfig':
        """
        Instantiate a :class:`~transformers.PretrainedConfig` (or a derived class) from a pretrained model
        configuration.

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                This can be either:

                - a string, the `model id` of a pretrained model configuration hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like ``bert-base-uncased``, or
                  namespaced under a user or organization name, like ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing a configuration file saved using the
                  :func:`~transformers.PretrainedConfig.save_pretrained` method, e.g., ``./my_model_directory/``.
                - a path or url to a saved configuration JSON `file`, e.g.,
                  ``./my_model_directory/configuration.json``.
            cache_dir (:obj:`str` or :obj:`os.PathLike`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force to (re-)download the configuration files and override the cached versions if
                they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received file. Attempts to resume the download if such a file
                exists.
            proxies (:obj:`Dict[str, str]`, `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            use_auth_token (:obj:`str` or `bool`, `optional`):
                The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
                generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            return_unused_kwargs (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If :obj:`False`, then this function returns just the final configuration object.

                If :obj:`True`, then this functions returns a :obj:`Tuple(config, unused_kwargs)` where `unused_kwargs`
                is a dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e.,
                the part of ``kwargs`` which has not been used to update ``config`` and is otherwise ignored.
            kwargs (:obj:`Dict[str, Any]`, `optional`):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the ``return_unused_kwargs`` keyword parameter.

        .. note::

            Passing :obj:`use_auth_token=True` is required when you want to use a private model.


        Returns:
            :class:`PretrainedConfig`: The configuration object instantiated from this pretrained model.

        Examples::

            # We can't instantiate directly the base class `PretrainedConfig` so let's show the examples on a
            # derived class: BertConfig
            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from huggingface.co and cache.
            config = BertConfig.from_pretrained('./test/saved_model/')  # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
            config = BertConfig.from_pretrained('./test/saved_model/my_configuration.json')
            config = BertConfig.from_pretrained('bert-base-uncased', output_attentions=True, foo=False)
            assert config.output_attentions == True
            config, unused_kwargs = BertConfig.from_pretrained('bert-base-uncased', output_attentions=True,
                                                               foo=False, return_unused_kwargs=True)
            assert config.output_attentions == True
            assert unused_kwargs == {'foo': False}

        """
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path: 'Union[str, os.PathLike]', **kwargs) ->Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a ``pretrained_model_name_or_path``, resolve to a dictionary of parameters, to be used for instantiating a
        :class:`~transformers.PretrainedConfig` using ``from_dict``.



        Parameters:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            :obj:`Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the configuration object.

        """
        cache_dir = kwargs.pop('cache_dir', None)
        force_download = kwargs.pop('force_download', False)
        resume_download = kwargs.pop('resume_download', False)
        proxies = kwargs.pop('proxies', None)
        use_auth_token = kwargs.pop('use_auth_token', None)
        local_files_only = kwargs.pop('local_files_only', False)
        revision = kwargs.pop('revision', None)
        if is_offline_mode() and not local_files_only:
            logger.info('Offline mode: forcing local_files_only=True')
            local_files_only = True
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        else:
            config_file = hf_bucket_url(pretrained_model_name_or_path, filename=CONFIG_NAME, revision=revision, mirror=None)
        try:
            resolved_config_file = cached_path(config_file, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, local_files_only=local_files_only, use_auth_token=use_auth_token)
            config_dict = cls._dict_from_json_file(resolved_config_file)
        except EnvironmentError as err:
            logger.error(err)
            msg = f"Can't load config for '{pretrained_model_name_or_path}'. Make sure that:\n\n- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a {CONFIG_NAME} file\n\n"
            raise EnvironmentError(msg)
        except json.JSONDecodeError:
            msg = "Couldn't reach server at '{}' to download configuration file or configuration file is not a valid JSON file. Please check network or file content here: {}.".format(config_file, resolved_config_file)
            raise EnvironmentError(msg)
        if resolved_config_file == config_file:
            logger.info('loading configuration file {}'.format(config_file))
        else:
            logger.info('loading configuration file {} from cache at {}'.format(config_file, resolved_config_file))
        return config_dict, kwargs

    @classmethod
    def from_dict(cls, config_dict: 'Dict[str, Any]', **kwargs) ->'PretrainedConfig':
        """
        Instantiates a :class:`~transformers.PretrainedConfig` from a Python dictionary of parameters.

        Args:
            config_dict (:obj:`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                :func:`~transformers.PretrainedConfig.get_config_dict` method.
            kwargs (:obj:`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            :class:`PretrainedConfig`: The configuration object instantiated from those parameters.
        """
        return_unused_kwargs = kwargs.pop('return_unused_kwargs', False)
        config = cls(**config_dict)
        if hasattr(config, 'pruned_heads'):
            config.pruned_heads = dict((int(key), value) for key, value in config.pruned_heads.items())
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)
        logger.info('Model config %s', str(config))
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_json_file(cls, json_file: 'Union[str, os.PathLike]') ->'PretrainedConfig':
        """
        Instantiates a :class:`~transformers.PretrainedConfig` from the path to a JSON file of parameters.

        Args:
            json_file (:obj:`str` or :obj:`os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            :class:`PretrainedConfig`: The configuration object instantiated from that JSON file.

        """
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: 'Union[str, os.PathLike]'):
        with open(json_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        return json.loads(text)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return '{} {}'.format(self.__class__.__name__, self.to_json_string())

    def to_diff_dict(self) ->Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()
        default_config_dict = PretrainedConfig().to_dict()
        class_config_dict = self.__class__().to_dict() if not self.is_composition else {}
        serializable_config_dict = {}
        for key, value in config_dict.items():
            if key not in default_config_dict or key == 'transformers_version' or value != default_config_dict[key] or key in class_config_dict and value != class_config_dict[key]:
                serializable_config_dict[key] = value
        return serializable_config_dict

    def to_dict(self) ->Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, 'model_type'):
            output['model_type'] = self.__class__.model_type
        output['transformers_version'] = __version__
        return output

    def to_json_string(self, use_diff: 'bool'=True) ->str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to ``True``, only the difference between the config instance and the default
                ``PretrainedConfig()`` is serialized to JSON string.

        Returns:
            :obj:`str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + '\n'

    def to_json_file(self, json_file_path: 'Union[str, os.PathLike]', use_diff: 'bool'=True):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (:obj:`str` or :obj:`os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to ``True``, only the difference between the config instance and the default
                ``PretrainedConfig()`` is serialized to JSON file.
        """
        with open(json_file_path, 'w', encoding='utf-8') as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    def update(self, config_dict: 'Dict[str, Any]'):
        """
        Updates attributes of this class with attributes from ``config_dict``.

        Args:
            config_dict (:obj:`Dict[str, Any]`): Dictionary of attributes that shall be updated for this class.
        """
        for key, value in config_dict.items():
            setattr(self, key, value)


TF2_WEIGHTS_NAME = 'tf_model.h5'


TF_WEIGHTS_NAME = 'model.ckpt'


WEIGHTS_NAME = 'pytorch_model.bin'


def convert_tf_weight_name_to_pt_weight_name(tf_name, start_prefix_to_remove=''):
    """
    Convert a TF 2.0 model variable name in a pytorch model weight name.

    Conventions for TF2.0 scopes -> PyTorch attribute names conversions:

        - '$1___$2' is replaced by $2 (can be used to duplicate or remove layers in TF2.0 vs PyTorch)
        - '_._' is replaced by a new level separation (can be used to convert TF2.0 lists in PyTorch nn.ModulesList)

    return tuple with:

        - pytorch model weight name
        - transpose: boolean indicating whether TF2.0 and PyTorch weights matrices are transposed with regards to each
          other
    """
    tf_name = tf_name.replace(':0', '')
    tf_name = re.sub('/[^/]*___([^/]*)/', '/\\1/', tf_name)
    tf_name = tf_name.replace('_._', '/')
    tf_name = re.sub('//+', '/', tf_name)
    tf_name = tf_name.split('/')
    if len(tf_name) > 1:
        tf_name = tf_name[1:]
    transpose = bool(tf_name[-1] in ['kernel', 'pointwise_kernel', 'depthwise_kernel'] or 'emb_projs' in tf_name or 'out_projs' in tf_name)
    if tf_name[-1] == 'kernel' or tf_name[-1] == 'embeddings' or tf_name[-1] == 'gamma':
        tf_name[-1] = 'weight'
    if tf_name[-1] == 'beta':
        tf_name[-1] = 'bias'
    if tf_name[-1] == 'pointwise_kernel' or tf_name[-1] == 'depthwise_kernel':
        tf_name[-1] = tf_name[-1].replace('_kernel', '.weight')
    tf_name = '.'.join(tf_name)
    if start_prefix_to_remove:
        tf_name = tf_name.replace(start_prefix_to_remove, '', 1)
    return tf_name, transpose


def load_tf2_weights_in_pytorch_model(pt_model, tf_weights, allow_missing_keys=False):
    """Load TF2.0 symbolic weights in a PyTorch model"""
    try:
        import tensorflow as tf
        import torch
    except ImportError:
        logger.error('Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.')
        raise
    new_pt_params_dict = {}
    current_pt_params_dict = dict(pt_model.named_parameters())
    start_prefix_to_remove = ''
    if not any(s.startswith(pt_model.base_model_prefix) for s in current_pt_params_dict.keys()):
        start_prefix_to_remove = pt_model.base_model_prefix + '.'
    tf_weights_map = {}
    for tf_weight in tf_weights:
        pt_name, transpose = convert_tf_weight_name_to_pt_weight_name(tf_weight.name, start_prefix_to_remove=start_prefix_to_remove)
        tf_weights_map[pt_name] = tf_weight.numpy(), transpose
    all_tf_weights = set(list(tf_weights_map.keys()))
    loaded_pt_weights_data_ptr = {}
    missing_keys_pt = []
    for pt_weight_name, pt_weight in current_pt_params_dict.items():
        if pt_weight.data_ptr() in loaded_pt_weights_data_ptr:
            new_pt_params_dict[pt_weight_name] = loaded_pt_weights_data_ptr[pt_weight.data_ptr()]
            continue
        if pt_weight_name not in tf_weights_map:
            if allow_missing_keys:
                missing_keys_pt.append(pt_weight_name)
                continue
            raise AttributeError('{} not found in TF 2.0 model'.format(pt_weight_name))
        array, transpose = tf_weights_map[pt_weight_name]
        if transpose:
            array = numpy.transpose(array)
        if len(pt_weight.shape) < len(array.shape):
            array = numpy.squeeze(array)
        elif len(pt_weight.shape) > len(array.shape):
            array = numpy.expand_dims(array, axis=0)
        if list(pt_weight.shape) != list(array.shape):
            try:
                array = numpy.reshape(array, pt_weight.shape)
            except AssertionError as e:
                e.args += pt_weight.shape, array.shape
                raise e
        try:
            assert list(pt_weight.shape) == list(array.shape)
        except AssertionError as e:
            e.args += pt_weight.shape, array.shape
            raise e
        new_pt_params_dict[pt_weight_name] = torch.from_numpy(array)
        loaded_pt_weights_data_ptr[pt_weight.data_ptr()] = torch.from_numpy(array)
        all_tf_weights.discard(pt_weight_name)
    missing_keys, unexpected_keys = pt_model.load_state_dict(new_pt_params_dict, strict=False)
    missing_keys += missing_keys_pt
    if len(unexpected_keys) > 0:
        logger.warning(f'Some weights of the TF 2.0 model were not used when initializing the PyTorch model {pt_model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are initializing {pt_model.__class__.__name__} from a TF 2.0 model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a TFBertForPreTraining model).\n- This IS NOT expected if you are initializing {pt_model.__class__.__name__} from a TF 2.0 model that you expect to be exactly identical (e.g. initializing a BertForSequenceClassification model from a TFBertForSequenceClassification model).')
    else:
        logger.warning(f'All TF 2.0 model weights were used when initializing {pt_model.__class__.__name__}.\n')
    if len(missing_keys) > 0:
        logger.warning(f'Some weights of {pt_model.__class__.__name__} were not initialized from the TF 2.0 model and are newly initialized: {missing_keys}\nYou should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.')
    else:
        logger.warning(f'All the weights of {pt_model.__class__.__name__} were initialized from the TF 2.0 model.\nIf your task is similar to the task the model of the checkpoint was trained on, you can already use {pt_model.__class__.__name__} for predictions without further training.')
    logger.info('Weights or buffers not loaded from TF 2.0 model: {}'.format(all_tf_weights))
    return pt_model


def load_tf2_model_in_pytorch_model(pt_model, tf_model, allow_missing_keys=False):
    """Load TF 2.0 model in a pytorch model"""
    weights = tf_model.weights
    return load_tf2_weights_in_pytorch_model(pt_model, weights, allow_missing_keys=allow_missing_keys)


def load_tf_weights(model, resolved_archive_file, _prefix=None):
    """
    Detect missing and unexpected layers and load the TF weights accordingly to their names and shapes.

    Args:
        model (:obj:`tf.keras.models.Model`):
            The model to load the weights into.
        resolved_archive_file (:obj:`str`):
            The location of the H5 file.

    Returns:
        Two lists, one for the missing layers, and another one for the unexpected layers.
    """
    missing_layers = []
    unexpected_layers = []
    with h5py.File(resolved_archive_file, 'r') as f:
        saved_h5_model_layers_name = set(hdf5_format.load_attributes_from_hdf5_group(f, 'layer_names'))
        missing_layers = list(set([layer.name for layer in model.layers]) - saved_h5_model_layers_name)
        unexpected_layers = list(saved_h5_model_layers_name - set([layer.name for layer in model.layers]))
        saved_weight_names_set = set()
        symbolic_weights_names = set()
        weight_value_tuples = []
        for layer in model.layers:
            if layer.name in saved_h5_model_layers_name:
                h5_layer_object = f[layer.name]
                symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
                saved_weights = {}
                for weight_name in hdf5_format.load_attributes_from_hdf5_group(h5_layer_object, 'weight_names'):
                    name = '/'.join(weight_name.split('/')[1:])
                    if _prefix is not None:
                        name = _prefix + '/' + name
                    saved_weights[name] = np.asarray(h5_layer_object[weight_name])
                    saved_weight_names_set.add(name)
                for symbolic_weight in symbolic_weights:
                    if _prefix is not None:
                        delimeter = len(_prefix.split('/'))
                        symbolic_weight_name = '/'.join(symbolic_weight.name.split('/')[:delimeter] + symbolic_weight.name.split('/')[delimeter + 1:])
                    else:
                        symbolic_weight_name = '/'.join(symbolic_weight.name.split('/')[1:])
                    saved_weight_value = saved_weights.get(symbolic_weight_name, None)
                    symbolic_weights_names.add(symbolic_weight_name)
                    if saved_weight_value is not None:
                        if K.int_shape(symbolic_weight) != saved_weight_value.shape:
                            try:
                                array = np.reshape(saved_weight_value, K.int_shape(symbolic_weight))
                            except AssertionError as e:
                                e.args += K.int_shape(symbolic_weight), saved_weight_value.shape
                                raise e
                        else:
                            array = saved_weight_value
                        weight_value_tuples.append((symbolic_weight, array))
    K.batch_set_value(weight_value_tuples)
    missing_layers.extend(list(symbolic_weights_names - saved_weight_names_set))
    unexpected_layers.extend(list(saved_weight_names_set - symbolic_weights_names))
    return missing_layers, unexpected_layers


PATH_TO_TRANSFORMERS = 'src/transformers'


def load_tf2_checkpoint_in_pytorch_model(pt_model, tf_checkpoint_path, tf_inputs=None, allow_missing_keys=False):
    """
    Load TF 2.0 HDF5 checkpoint in a PyTorch model We use HDF5 to easily do transfer learning (see
    https://github.com/tensorflow/tensorflow/blob/ee16fcac960ae660e0e4496658a366e2f745e1f0/tensorflow/python/keras/engine/network.py#L1352-L1357).
    """
    try:
        import tensorflow as tf
        import torch
    except ImportError:
        logger.error('Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.')
        raise
    logger.info('Loading TensorFlow weights from {}'.format(tf_checkpoint_path))
    tf_model_class_name = 'TF' + pt_model.__class__.__name__
    tf_model_class = getattr(transformers, tf_model_class_name)
    tf_model = tf_model_class(pt_model.config)
    if tf_inputs is None:
        tf_inputs = tf_model.dummy_inputs
    if tf_inputs is not None:
        tf_model(tf_inputs, training=False)
    load_tf_weights(tf_model, tf_checkpoint_path)
    return load_tf2_model_in_pytorch_model(pt_model, tf_model, allow_missing_keys=allow_missing_keys)


def unwrap_model(model: 'torch.nn.Module') ->torch.nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (:obj:`torch.nn.Module`): The model to unwrap.
    """
    if hasattr(model, 'module'):
        return unwrap_model(model.module)
    else:
        return model


def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    """Load tf checkpoints in a pytorch model"""
    try:
        import re
        import tensorflow as tf
    except ImportError:
        logger.error('Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.')
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    logger.info('Converting TensorFlow checkpoint from {}'.format(tf_path))
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info('Loading TF weight {} with shape {}'.format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())
    for name, array in zip(names, arrays):
        name = name[6:]
        name = name.split('/')
        pointer = model
        for m_name in name:
            if re.fullmatch('[A-Za-z]+\\d+', m_name):
                scope_names = re.split('(\\d+)', m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == 'w' or scope_names[0] == 'g':
                pointer = getattr(pointer, 'weight')
            elif scope_names[0] == 'b':
                pointer = getattr(pointer, 'bias')
            elif scope_names[0] == 'wpe' or scope_names[0] == 'wte':
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        try:
            assert pointer.shape == array.shape, f'Pointer shape {pointer.shape} and array shape {array.shape} mismatched'
        except AssertionError as e:
            e.args += pointer.shape, array.shape
            raise
        logger.info('Initialize PyTorch weight {}'.format(name))
        pointer.data = torch.from_numpy(array)
    return model


GPT2_INPUTS_DOCSTRING = """
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, input_ids_length)`):
            :obj:`input_ids_length` = ``sequence_length`` if :obj:`past_key_values` is ``None`` else
            ``past_key_values[0][0].shape[-2]`` (``sequence_length`` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If :obj:`past_key_values` is used, only ``input_ids`` that do not have their past calculated should be
            passed as ``input_ids``.

            Indices can be obtained using :class:`~transformers.GPT2Tokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers`):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            :obj:`past_key_values` output below). Can be used to speed up sequential decoding. The ``input_ids`` which
            have their past given to this model should not be passed as ``input_ids`` as they have already been
            computed.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, input_ids_length)`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.

            If :obj:`past_key_values` is used, optionally only the last :obj:`inputs_embeds` have to be input (see
            :obj:`past_key_values`).
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


GPT2_START_DOCSTRING = """

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.GPT2Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""


PARALLELIZE_DOCSTRING = """
    This is an experimental feature and is a subject to change at a moment's notice.

    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.

    Args:
        device_map (:obj:`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the t5 models have the
            following number of attention modules:

                - t5-small: 6
                - t5-base: 12
                - t5-large: 24
                - t5-3b: 24
                - t5-11b: 24

    Example::

            # Here is an example of a device map on a machine with 4 GPUs using t5-3b, which has a total of 24 attention modules:
            model = T5ForConditionalGeneration.from_pretrained('t5-3b')
            device_map = {0: [0, 1, 2],

                         1: [3, 4, 5, 6, 7, 8, 9],
                         2: [10, 11, 12, 13, 14, 15, 16],
                         3: [17, 18, 19, 20, 21, 22, 23]}
            model.parallelize(device_map)
"""


_CHECKPOINT_FOR_DOC = 'xlnet-base-cased'


_CONFIG_FOR_DOC = 'XLNetConfig'


_TOKENIZER_FOR_DOC = 'XLNetTokenizer'


PT_BASE_MODEL_SAMPLE = """
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
"""


PT_CAUSAL_LM_SAMPLE = """
    Example::

        >>> import torch
        >>> from transformers import {tokenizer_class}, {model_class}

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs, labels=inputs["input_ids"])
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""


PT_MASKED_LM_SAMPLE = """
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("The capital of France is {mask}.", return_tensors="pt")
        >>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]

        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""


PT_MULTIPLE_CHOICE_SAMPLE = """
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> choice0 = "It is eaten with a fork and a knife."
        >>> choice1 = "It is eaten while held in the hand."
        >>> labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

        >>> encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors='pt', padding=True)
        >>> outputs = model(**{{k: v.unsqueeze(0) for k,v in encoding.items()}}, labels=labels)  # batch size is 1

        >>> # the linear classifier still needs to be trained
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""


PT_QUESTION_ANSWERING_SAMPLE = """
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        >>> inputs = tokenizer(question, text, return_tensors='pt')
        >>> start_positions = torch.tensor([1])
        >>> end_positions = torch.tensor([3])

        >>> outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
        >>> loss = outputs.loss
        >>> start_scores = outputs.start_logits
        >>> end_scores = outputs.end_logits
"""


PT_SEQUENCE_CLASSIFICATION_SAMPLE = """
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""


PT_TOKEN_CLASSIFICATION_SAMPLE = """
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1

        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""


TF_BASE_MODEL_SAMPLE = """
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> outputs = model(inputs)

        >>> last_hidden_states = outputs.last_hidden_state
"""


TF_CAUSAL_LM_SAMPLE = """
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> outputs = model(inputs)
        >>> logits = outputs.logits
"""


TF_MASKED_LM_SAMPLE = """
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("The capital of France is {mask}.", return_tensors="tf")
        >>> inputs["labels"] = tokenizer("The capital of France is Paris.", return_tensors="tf")["input_ids"]

        >>> outputs = model(inputs)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""


TF_MULTIPLE_CHOICE_SAMPLE = """
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> choice0 = "It is eaten with a fork and a knife."
        >>> choice1 = "It is eaten while held in the hand."

        >>> encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors='tf', padding=True)
        >>> inputs = {{k: tf.expand_dims(v, 0) for k, v in encoding.items()}}
        >>> outputs = model(inputs)  # batch size is 1

        >>> # the linear classifier still needs to be trained
        >>> logits = outputs.logits
"""


TF_QUESTION_ANSWERING_SAMPLE = """
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        >>> input_dict = tokenizer(question, text, return_tensors='tf')
        >>> outputs = model(input_dict)
        >>> start_logits = outputs.start_logits
        >>> end_logits = outputs.end_logits

        >>> all_tokens = tokenizer.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])
        >>> answer = ' '.join(all_tokens[tf.math.argmax(start_logits, 1)[0] : tf.math.argmax(end_logits, 1)[0]+1])
"""


TF_SEQUENCE_CLASSIFICATION_SAMPLE = """
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> inputs["labels"] = tf.reshape(tf.constant(1), (-1, 1)) # Batch size 1

        >>> outputs = model(inputs)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""


TF_TOKEN_CLASSIFICATION_SAMPLE = """
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> input_ids = inputs["input_ids"]
        >>> inputs["labels"] = tf.reshape(tf.constant([1] * tf.size(input_ids).numpy()), (-1, tf.size(input_ids))) # Batch size 1

        >>> outputs = model(inputs)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""


PT_RETURN_INTRODUCTION = """
    Returns:
        :class:`~{full_output_type}` or :obj:`tuple(torch.FloatTensor)`: A :class:`~{full_output_type}` (if
        ``return_dict=True`` is passed or when ``config.return_dict=True``) or a tuple of :obj:`torch.FloatTensor`
        comprising various elements depending on the configuration (:class:`~transformers.{config_class}`) and inputs.

"""


TF_RETURN_INTRODUCTION = """
    Returns:
        :class:`~{full_output_type}` or :obj:`tuple(tf.Tensor)`: A :class:`~{full_output_type}` (if
        ``return_dict=True`` is passed or when ``config.return_dict=True``) or a tuple of :obj:`tf.Tensor` comprising
        various elements depending on the configuration (:class:`~transformers.{config_class}`) and inputs.

"""


def _get_indent(t):
    """Returns the indentation in the first line of t"""
    search = re.search('^(\\s*)\\S', t)
    return '' if search is None else search.groups()[0]


def _convert_output_args_doc(output_args_doc):
    """Convert output_args_doc to display properly."""
    indent = _get_indent(output_args_doc)
    blocks = []
    current_block = ''
    for line in output_args_doc.split('\n'):
        if _get_indent(line) == indent:
            if len(current_block) > 0:
                blocks.append(current_block[:-1])
            current_block = f'{line}\n'
        else:
            current_block += f'{line[2:]}\n'
    blocks.append(current_block[:-1])
    for i in range(len(blocks)):
        blocks[i] = re.sub('^(\\s+)(\\S+)(\\s+)', '\\1- **\\2**\\3', blocks[i])
        blocks[i] = re.sub(':\\s*\\n\\s*(\\S)', ' -- \\1', blocks[i])
    return '\n'.join(blocks)


def _prepare_output_docstrings(output_type, config_class):
    """
    Prepares the return part of the docstring using `output_type`.
    """
    docstrings = output_type.__doc__
    lines = docstrings.split('\n')
    i = 0
    while i < len(lines) and re.search('^\\s*(Args|Parameters):\\s*$', lines[i]) is None:
        i += 1
    if i < len(lines):
        docstrings = '\n'.join(lines[i + 1:])
        docstrings = _convert_output_args_doc(docstrings)
    full_output_type = f'{output_type.__module__}.{output_type.__name__}'
    intro = TF_RETURN_INTRODUCTION if output_type.__name__.startswith('TF') else PT_RETURN_INTRODUCTION
    intro = intro.format(full_output_type=full_output_type, config_class=config_class)
    return intro + docstrings


def add_code_sample_docstrings(*docstr, tokenizer_class=None, checkpoint=None, output_type=None, config_class=None, mask=None):

    def docstring_decorator(fn):
        model_class = fn.__qualname__.split('.')[0]
        is_tf_class = model_class[:2] == 'TF'
        doc_kwargs = dict(model_class=model_class, tokenizer_class=tokenizer_class, checkpoint=checkpoint)
        if 'SequenceClassification' in model_class:
            code_sample = TF_SEQUENCE_CLASSIFICATION_SAMPLE if is_tf_class else PT_SEQUENCE_CLASSIFICATION_SAMPLE
        elif 'QuestionAnswering' in model_class:
            code_sample = TF_QUESTION_ANSWERING_SAMPLE if is_tf_class else PT_QUESTION_ANSWERING_SAMPLE
        elif 'TokenClassification' in model_class:
            code_sample = TF_TOKEN_CLASSIFICATION_SAMPLE if is_tf_class else PT_TOKEN_CLASSIFICATION_SAMPLE
        elif 'MultipleChoice' in model_class:
            code_sample = TF_MULTIPLE_CHOICE_SAMPLE if is_tf_class else PT_MULTIPLE_CHOICE_SAMPLE
        elif 'MaskedLM' in model_class or model_class in ['FlaubertWithLMHeadModel', 'XLMWithLMHeadModel']:
            doc_kwargs['mask'] = '[MASK]' if mask is None else mask
            code_sample = TF_MASKED_LM_SAMPLE if is_tf_class else PT_MASKED_LM_SAMPLE
        elif 'LMHead' in model_class or 'CausalLM' in model_class:
            code_sample = TF_CAUSAL_LM_SAMPLE if is_tf_class else PT_CAUSAL_LM_SAMPLE
        elif 'Model' in model_class or 'Encoder' in model_class:
            code_sample = TF_BASE_MODEL_SAMPLE if is_tf_class else PT_BASE_MODEL_SAMPLE
        else:
            raise ValueError(f"Docstring can't be built for model {model_class}")
        output_doc = _prepare_output_docstrings(output_type, config_class) if output_type is not None else ''
        built_doc = code_sample.format(**doc_kwargs)
        fn.__doc__ = (fn.__doc__ or '') + ''.join(docstr) + output_doc + built_doc
        return fn
    return docstring_decorator


def add_start_docstrings_to_model_forward(*docstr):

    def docstring_decorator(fn):
        class_name = ':class:`~transformers.{}`'.format(fn.__qualname__.split('.')[0])
        intro = '   The {} forward method, overrides the :func:`__call__` special method.'.format(class_name)
        note = '\n\n    .. note::\n        Although the recipe for forward pass needs to be defined within this function, one should call the\n        :class:`Module` instance afterwards instead of this since the former takes care of running the pre and post\n        processing steps while the latter silently ignores them.\n        '
        fn.__doc__ = intro + note + ''.join(docstr) + (fn.__doc__ if fn.__doc__ is not None else '')
        return fn
    return docstring_decorator


def assert_device_map(device_map, num_blocks):
    blocks = list(range(0, num_blocks))
    device_map_blocks = [item for sublist in list(device_map.values()) for item in sublist]
    duplicate_blocks = []
    for i in device_map_blocks:
        if device_map_blocks.count(i) > 1 and i not in duplicate_blocks:
            duplicate_blocks.append(i)
    missing_blocks = [i for i in blocks if i not in device_map_blocks]
    extra_blocks = [i for i in device_map_blocks if i not in blocks]
    assert len(duplicate_blocks) == 0, 'Duplicate attention blocks specified in device_map. Attention blocks must be specified to one device. These attention blocks were specified more than once: ' + str(duplicate_blocks)
    assert len(missing_blocks) == 0, 'There are attention blocks for this model that are not specified in the device_map. Add these attention blocks to a device on the device_map: ' + str(missing_blocks)
    assert len(extra_blocks) == 0, 'The device_map contains more attention blocks than this model has. Remove these from the device_map:' + str(extra_blocks)


def get_device_map(n_layers, devices):
    """Returns a dictionary of layers distributed evenly across all devices."""
    layers = list(range(n_layers))
    n_blocks = int(ceil(n_layers / len(devices)))
    layers_list = list(layers[i:i + n_blocks] for i in range(0, n_layers, n_blocks))
    return dict(zip(devices, layers_list))


class GPT2LMHead(nn.Module):

    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights

    def forward(self, hidden_state):
        lm_logits = self.decoder(hidden_state)
        return lm_logits


class GPT2LMModel(nn.Module):

    def __init__(self, config):
        super(GPT2LMModel, self).__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)
        self.apply(self._init_weights)

    def set_tied(self):
        """ Make sure we are sharing the embeddings"""
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(self, input_ids, lm_labels=None, lm_mask=None, past=None, len_past=None, label_smooth=0.0, is_report_accuracy=False):
        _batch, _len = input_ids.shape
        hidden_states, presents = self.transformer(input_ids, past=past, len_past=len_past)
        lm_logits = self.lm_head(hidden_states)
        if lm_labels is not None:
            if is_report_accuracy:
                _pred_token = torch.argmax(lm_logits, dim=-1)
                _hit = (_pred_token == lm_labels) * lm_mask
                _t1_acc = torch.zeros(_batch, dtype=torch.float, device=input_ids.device)
                _all_acc = torch.zeros(_batch, dtype=torch.float, device=input_ids.device)
                for _b in range(0, _batch):
                    for _i in range(0, _len):
                        if lm_mask[_b, _i] >= 1.0:
                            if _hit[_b, _i] > 0:
                                _t1_acc[_b] = 1.0
                            break
                    _is_succ = True
                    for _i in range(0, _len):
                        if lm_mask[_b, _i] >= 1.0:
                            if _hit[_b, _i] <= 0:
                                _is_succ = False
                                break
                    if _is_succ:
                        _all_acc[_b] = 1.0
            if label_smooth > 0.0001:
                logprobs = torch.nn.functional.log_softmax(lm_logits.view(-1, lm_logits.size(-1)), dim=-1)
                nll_loss = -logprobs.gather(dim=-1, index=lm_labels.view(-1).unsqueeze(1))
                nll_loss = nll_loss.squeeze(1)
                smooth_loss = -logprobs.mean(dim=-1)
                loss = (1.0 - label_smooth) * nll_loss + label_smooth * smooth_loss
                loss = loss.view(_batch, _len)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduce=False)
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1)).view(_batch, _len)
            if lm_mask is None:
                lm_mask = torch.ones(loss.shape, dtype=loss.dtype, device=loss.device)
            loss = loss * lm_mask
            loss = loss.sum() / (lm_mask.sum() + 0.0001)
            if is_report_accuracy:
                return lm_logits, loss, _t1_acc, _all_acc
            else:
                return lm_logits, loss
        return lm_logits, presents

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def load_weight(self, state_dict):
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        state_dict_tmp = copy.deepcopy(state_dict)
        old_keys = []
        new_keys = []
        for key in state_dict_tmp:
            new_key = None
            if key.endswith('.g'):
                new_key = key[:-2] + '.weight'
            elif key.endswith('.b'):
                new_key = key[:-2] + '.bias'
            elif key.endswith('.w'):
                new_key = key[:-2] + '.weight'
            if key.startswith('module.transformer.'):
                new_key = key[len('module.transformer.'):]
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        for n, p in self.transformer.named_parameters():
            if n not in state_dict:
                state_dict[n] = p
        self.transformer.load_state_dict(state_dict, strict=False)
        self.set_tied()


class BertConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.BertModel` or a
    :class:`~transformers.TFBertModel`. It is used to instantiate a BERT model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the BERT `bert-base-uncased <https://huggingface.co/bert-base-uncased>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        position_embedding_type (:obj:`str`, `optional`, defaults to :obj:`"absolute"`):
            Type of position embedding. Choose one of :obj:`"absolute"`, :obj:`"relative_key"`,
            :obj:`"relative_key_query"`. For positional embeddings use :obj:`"absolute"`. For more information on
            :obj:`"relative_key"`, please refer to `Self-Attention with Relative Position Representations (Shaw et al.)
            <https://arxiv.org/abs/1803.02155>`__. For more information on :obj:`"relative_key_query"`, please refer to
            `Method 4` in `Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)
            <https://arxiv.org/abs/2009.13658>`__.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if ``config.is_decoder=True``.
        apply_lora (:obj:`bool`, `optional`):
            apply Lora.
        lora_alpha (:obj:`int`, `optional`):
            lora alpha.
        lora_r (:obj:`int`, `optional`):
            lora r.
        apply_adapter (:obj:`bool`, `optional`):
            apply adapter.
        adapter_type (:obj:`str`, `optional`):
            houlsby or pfeiffer.
        adapter_size (:obj:`int`, `optional`):
            8 16 32 64.

    Examples::

        >>> from transformers import BertModel, BertConfig

        >>> # Initializing a BERT bert-base-uncased style configuration
        >>> configuration = BertConfig()

        >>> # Initializing a model from the bert-base-uncased style configuration
        >>> model = BertModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = 'bert'

    def __init__(self, vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0, gradient_checkpointing=False, position_embedding_type='absolute', use_cache=True, apply_lora=False, lora_alpha=None, lora_r=None, apply_adapter=False, adapter_type=None, adapter_size=None, **kwargs):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.apply_lora = apply_lora
        self.lora_alpha = lora_alpha
        self.lora_r = lora_r
        self.apply_adapter = apply_adapter
        self.adapter_type = adapter_type
        self.adapter_size = adapter_size


BERT_INPUTS_DOCSTRING = """
    Args:
        input_ids (:obj:`np.ndarray`, :obj:`tf.Tensor`, :obj:`List[tf.Tensor]` :obj:`Dict[str, tf.Tensor]` or :obj:`Dict[str, np.ndarray]` and each example must have the shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertTokenizer`. See
            :func:`transformers.PreTrainedTokenizer.__call__` and :func:`transformers.PreTrainedTokenizer.encode` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`np.ndarray` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`np.ndarray` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`__
        position_ids (:obj:`np.ndarray` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`__
        head_mask (:obj:`np.ndarray` or :obj:`tf.Tensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`np.ndarray` or :obj:`tf.Tensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple. This
            argument can be used in eager mode, in graph mode the value will always be set to True.
        training (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


BERT_START_DOCSTRING = """

    This model inherits from :class:`~transformers.TFPreTrainedModel`. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading or saving, resizing the input
    embeddings, pruning heads etc.)

    This model is also a `tf.keras.Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ subclass. Use
    it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage
    and behavior.

    .. note::

        TF 2.0 models accepts two formats as inputs:

        - having all inputs as keyword arguments (like PyTorch models), or
        - having all inputs as a list, tuple or dict in the first positional arguments.

        This second option is useful when using :meth:`tf.keras.Model.fit` method which currently requires having all
        the tensors in the first argument of the model call function: :obj:`model(inputs)`.

        If you choose this second option, there are three possibilities you can use to gather all the input Tensors in
        the first positional argument :

        - a single Tensor with :obj:`input_ids` only and nothing else: :obj:`model(inputs_ids)`
        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
          :obj:`model([input_ids, attention_mask])` or :obj:`model([input_ids, attention_mask, token_type_ids])`
        - a dictionary with one or several input Tensors associated to the input names given in the docstring:
          :obj:`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Args:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.TFPreTrainedModel.from_pretrained` method to load the
            model weights.
"""


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:seq_length + past_key_values_length]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, 'embedding_size'):
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
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


def prune_linear_layer(layer: 'torch.nn.Linear', index: 'torch.LongTensor', dim: 'int'=0) ->torch.nn.Linear:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (:obj:`torch.nn.Linear`): The layer to prune.
        index (:obj:`torch.LongTensor`): The indices to keep in the layer.
        dim (:obj:`int`, `optional`, defaults to 0): The dimension on which to keep the indices.

    Returns:
        :obj:`torch.nn.Linear`: The pruned layer as a new layer with :obj:`requires_grad=True`.
    """
    index = index
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


class BertAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
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


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super().__init__()
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
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def apply_chunking_to_forward(forward_fn: 'Callable[..., torch.Tensor]', chunk_size: 'int', chunk_dim: 'int', *input_tensors) ->torch.Tensor:
    """
    This function chunks the :obj:`input_tensors` into smaller input tensor parts of size :obj:`chunk_size` over the
    dimension :obj:`chunk_dim`. It then applies a layer :obj:`forward_fn` to each chunk independently to save memory.

    If the :obj:`forward_fn` is independent across the :obj:`chunk_dim` this function will yield the same result as
    directly applying :obj:`forward_fn` to :obj:`input_tensors`.

    Args:
        forward_fn (:obj:`Callable[..., torch.Tensor]`):
            The forward function of the model.
        chunk_size (:obj:`int`):
            The chunk size of a chunked tensor: :obj:`num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_dim (:obj:`int`):
            The dimension over which the :obj:`input_tensors` should be chunked.
        input_tensors (:obj:`Tuple[torch.Tensor]`):
            The input tensors of ``forward_fn`` which will be chunked

    Returns:
        :obj:`torch.Tensor`: A tensor with the same shape as the :obj:`forward_fn` would have given if applied`.


    Examples::

        # rename the usual forward() fn to forward_chunk()
        def forward_chunk(self, hidden_states):
            hidden_states = self.decoder(hidden_states)
            return hidden_states

        # implement a chunked forward function
        def forward(self, hidden_states):
            return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
    """
    assert len(input_tensors) > 0, '{} has to be a tuple/list of tensors'.format(input_tensors)
    tensor_shape = input_tensors[0].shape[chunk_dim]
    assert all(input_tensor.shape[chunk_dim] == tensor_shape for input_tensor in input_tensors), 'All input tenors have to be of the same shape'
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    assert num_args_in_forward_chunk_fn == len(input_tensors), 'forward_chunk_fn expects {} arguments, but only {} input tensors are given'.format(num_args_in_forward_chunk_fn, len(input_tensors))
    if chunk_size > 0:
        assert input_tensors[0].shape[chunk_dim] % chunk_size == 0, 'The dimension to be chunked {} has to be a multiple of the chunk size {}'.format(input_tensors[0].shape[chunk_dim], chunk_size)
        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        return torch.cat(output_chunks, dim=chunk_dim)
    return forward_fn(*input_tensors)


class BertLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f'{self} should be used as a decoder model if cross attention is added'
            self.crossattention = BertAttention(config)
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
            assert hasattr(self, 'crossattention'), f'If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`'
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


class BertEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, use_cache=None, output_attentions=False, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if getattr(self.config, 'gradient_checkpointing', False) and self.training:
                if use_cache:
                    logger.warn('`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting `use_cache=False`...')
                    use_cache = False

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)
                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(layer_module), hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += layer_outputs[-1],
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=next_decoder_cache, hidden_states=all_hidden_states, attentions=all_self_attentions, cross_attentions=all_cross_attentions)


class BertPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error('Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.')
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info('Converting TensorFlow checkpoint from {}'.format(tf_path))
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info('Loading TF weight {} with shape {}'.format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    for name, array in zip(names, arrays):
        name = name.split('/')
        if any(n in ['adam_v', 'adam_m', 'AdamWeightDecayOptimizer', 'AdamWeightDecayOptimizer_1', 'global_step'] for n in name):
            logger.info('Skipping {}'.format('/'.join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch('[A-Za-z]+_\\d+', m_name):
                scope_names = re.split('_(\\d+)', m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == 'kernel' or scope_names[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif scope_names[0] == 'output_bias' or scope_names[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif scope_names[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif scope_names[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info('Skipping {}'.format('/'.join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape, f'Pointer shape {pointer.shape} and array shape {array.shape} mismatched'
        except AssertionError as e:
            e.args += pointer.shape, array.shape
            raise
        logger.info('Initialize PyTorch weight {}'.format(name))
        pointer.data = torch.from_numpy(array)
    return model


class Bert(nn.Module):
    """This class is not really necessary and should probably disappear."""

    def __init__(self):
        super().__init__()
        config = BertConfig.from_pretrained('bert-base-uncased')
        self.model = BertModel(config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        self.eval()
        with torch.no_grad():
            encoder_outputs, _ = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, **kwargs)
        return encoder_outputs


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super().__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if step:
            emb = emb + self.pe[:, step][:, None, :]
        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]


MAX_SIZE = 5000


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1, use_final_linear=True):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        super().__init__()
        self.head_count = head_count
        self.linear_keys = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.use_final_linear = use_final_linear
        if self.use_final_linear:
            self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None, layer_cache=None, type=None, predefined_graph_1=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :

           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)
        if layer_cache is not None:
            if type == 'self':
                query, key, value = self.linear_query(query), self.linear_keys(query), self.linear_values(query)
                key = shape(key)
                value = shape(value)
                if layer_cache is not None:
                    device = key.device
                    if layer_cache['self_keys'] is not None:
                        key = torch.cat((layer_cache['self_keys'], key), dim=2)
                    if layer_cache['self_values'] is not None:
                        value = torch.cat((layer_cache['self_values'], value), dim=2)
                    layer_cache['self_keys'] = key
                    layer_cache['self_values'] = value
            elif type == 'context':
                query = self.linear_query(query)
                if layer_cache is not None:
                    if layer_cache['memory_keys'] is None:
                        key, value = self.linear_keys(key), self.linear_values(value)
                        key = shape(key)
                        value = shape(value)
                    else:
                        key, value = layer_cache['memory_keys'], layer_cache['memory_values']
                    layer_cache['memory_keys'] = key
                    layer_cache['memory_values'] = value
                else:
                    key, value = self.linear_keys(key), self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)
        query = shape(query)
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))
        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e+18)
        attn = self.softmax(scores)
        if predefined_graph_1 is not None:
            attn_masked = attn[:, -1] * predefined_graph_1
            attn_masked = attn_masked / (torch.sum(attn_masked, 2).unsqueeze(2) + 1e-09)
            attn = torch.cat([attn[:, :-1], attn_masked.unsqueeze(1)], 1)
        drop_attn = self.dropout(attn)
        if self.use_final_linear:
            context = unshape(torch.matmul(drop_attn, value))
            output = self.final_linear(context)
            return output
        else:
            context = torch.matmul(drop_attn, value)
            return context


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    """A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-06)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the PositionwiseFeedForward.
      dropout (float): dropout probability(0-1.0).
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.context_attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-06)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-06)
        self.drop = nn.Dropout(dropout)
        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        self.register_buffer('mask', mask)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask, previous_input=None, layer_cache=None, step=None):
        """
        Args:
            inputs (`FloatTensor`): `[batch_size x 1 x model_dim]`
            memory_bank (`FloatTensor`): `[batch_size x src_len x model_dim]`
            src_pad_mask (`LongTensor`): `[batch_size x 1 x src_len]`
            tgt_pad_mask (`LongTensor`): `[batch_size x 1 x 1]`

        Returns:
            (`FloatTensor`, `FloatTensor`, `FloatTensor`):

            * output `[batch_size x 1 x model_dim]`
            * attn `[batch_size x 1 x src_len]`
            * all_input `[batch_size x current_step x model_dim]`

        """
        dec_mask = torch.gt(tgt_pad_mask + self.mask[:, :tgt_pad_mask.size(1), :tgt_pad_mask.size(1)], 0)
        input_norm = self.layer_norm_1(inputs)
        all_input = input_norm
        if previous_input is not None:
            all_input = torch.cat((previous_input, input_norm), dim=1)
            dec_mask = None
        query = self.self_attn(all_input, all_input, input_norm, mask=dec_mask, layer_cache=layer_cache, type='self')
        query = self.drop(query) + inputs
        query_norm = self.layer_norm_2(query)
        mid = self.context_attn(memory_bank, memory_bank, query_norm, mask=src_pad_mask, layer_cache=layer_cache, type='context')
        output = self.feed_forward(self.drop(mid) + query)
        return output, all_input

    def _get_attn_subsequent_mask(self, size):
        """
        Get an attention mask to avoid using the subsequent info.

        Args:
            size: int

        Returns:
            (`LongTensor`):

            * subsequent_mask `[1 x size x size]`
        """
        attn_shape = 1, size, size
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.

    Modules need to implement this to utilize beam search decoding.
    """

    def detach(self):
        """ Need to document this """
        self.hidden = tuple([_.detach() for _ in self.hidden])
        self.input_feed = self.input_feed.detach()

    def beam_update(self, idx, positions, beam_size):
        """ Need to document this """
        for e in self._all:
            sizes = e.size()
            br = sizes[1]
            if len(sizes) == 3:
                sent_states = e.view(sizes[0], beam_size, br // beam_size, sizes[2])[:, :, idx]
            else:
                sent_states = e.view(sizes[0], beam_size, br // beam_size, sizes[2], sizes[3])[:, :, idx]
            sent_states.data.copy_(sent_states.data.index_select(1, positions))

    def map_batch_fn(self, fn):
        raise NotImplementedError()


class TransformerDecoderState(DecoderState):
    """ Transformer Decoder state base class """

    def __init__(self, src):
        """
        Args:
            src (FloatTensor): a sequence of source words tensors
                    with optional feature tensors, of size (len x batch).
        """
        self.src = src
        self.previous_input = None
        self.previous_layer_inputs = None
        self.cache = None

    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        if self.previous_input is not None and self.previous_layer_inputs is not None:
            return self.previous_input, self.previous_layer_inputs, self.src
        else:
            return self.src,

    def detach(self):
        if self.previous_input is not None:
            self.previous_input = self.previous_input.detach()
        if self.previous_layer_inputs is not None:
            self.previous_layer_inputs = self.previous_layer_inputs.detach()
        self.src = self.src.detach()

    def update_state(self, new_input, previous_layer_inputs):
        state = TransformerDecoderState(self.src)
        state.previous_input = new_input
        state.previous_layer_inputs = previous_layer_inputs
        return state

    def _init_cache(self, memory_bank, num_layers):
        self.cache = {}
        for l in range(num_layers):
            layer_cache = {'memory_keys': None, 'memory_values': None}
            layer_cache['self_keys'] = None
            layer_cache['self_values'] = None
            self.cache['layer_{}'.format(l)] = layer_cache

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.src = self.src.data.repeat(1, beam_size, 1)

    def map_batch_fn(self, fn):

        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)
        self.src = fn(self.src, 0)
        if self.cache is not None:
            _recursive_map(self.cache)


class TransformerDecoder(nn.Module):
    """
    The Transformer decoder from "Attention is All You Need".

    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       dropout (float): dropout parameters
       embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
       attn_type (str): if using a separate copy attention
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings, vocab_size):
        super().__init__()
        self.decoder_type = 'transformer'
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout, self.embeddings.embedding_dim)
        self.transformer_layers = nn.ModuleList([TransformerDecoderLayer(d_model, heads, d_ff, dropout) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-06)

    def forward(self, input_ids, encoder_hidden_states=None, state=None, attention_mask=None, memory_lengths=None, step=None, cache=None, encoder_attention_mask=None):
        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        memory_bank = encoder_hidden_states
        """
        tgt = input_ids
        memory_bank = encoder_hidden_states
        memory_mask = encoder_attention_mask
        src_words = state.src
        src_batch, src_len = src_words.size()
        padding_idx = self.embeddings.padding_idx
        tgt_words = tgt
        tgt_batch, tgt_len = tgt_words.size()
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1).expand(tgt_batch, tgt_len, tgt_len)
        if memory_mask is not None:
            src_len = memory_mask.size(-1)
            src_pad_mask = memory_mask.expand(src_batch, tgt_len, src_len)
        else:
            src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1).expand(src_batch, tgt_len, src_len)
        emb = self.embeddings(input_ids)
        output = self.pos_emb(emb, step)
        assert emb.dim() == 3
        if state.cache is None:
            saved_inputs = []
        for i in range(self.num_layers):
            prev_layer_input = None
            if state.cache is None:
                if state.previous_input is not None:
                    prev_layer_input = state.previous_layer_inputs[i]
            output, all_input = self.transformer_layers[i](output, memory_bank, src_pad_mask, tgt_pad_mask, previous_input=prev_layer_input, layer_cache=state.cache['layer_{}'.format(i)] if state.cache is not None else None, step=step)
            if state.cache is None:
                saved_inputs.append(all_input)
        if state.cache is None:
            saved_inputs = torch.stack(saved_inputs)
        output = self.layer_norm(output)
        if state.cache is None:
            state = state.update_state(tgt, saved_inputs)
        return output, state

    def init_decoder_state(self, src, memory_bank, with_cache=False):
        """ Init decoder state """
        state = TransformerDecoderState(src)
        if with_cache:
            state._init_cache(memory_bank, self.num_layers)
        return state


class BertHighway(nn.Module):
    """A module to provide a shortcut
    from (the output of one non-final BertLayer in BertEncoder) to (cross-entropy computation in BertForSequenceClassification)
    """

    def __init__(self, config):
        super().__init__()
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, encoder_outputs):
        pooler_input = encoder_outputs[0]
        pooler_output = self.pooler(pooler_input)
        bmodel_output = (pooler_input, pooler_output) + encoder_outputs[1:]
        pooled_output = bmodel_output[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits, pooled_output


class HighwayException(Exception):

    def __init__(self, message, exit_layer):
        self.message = message
        self.exit_layer = exit_layer


def entropy(x):
    """Calculate entropy of a pre-softmax logit Tensor"""
    exp_x = torch.exp(x)
    A = torch.sum(exp_x, dim=1)
    B = torch.sum(x * exp_x, dim=1)
    return torch.log(A) - B / A


class DeeBertEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.highway = nn.ModuleList([BertHighway(config) for _ in range(config.num_hidden_layers)])
        self.early_exit_entropy = [(-1) for _ in range(config.num_hidden_layers)]

    def set_early_exit_entropy(self, x):
        if type(x) is float or type(x) is int:
            for i in range(len(self.early_exit_entropy)):
                self.early_exit_entropy[i] = x
        else:
            self.early_exit_entropy = x

    def init_highway_pooler(self, pooler):
        loaded_model = pooler.state_dict()
        for highway in self.highway:
            for name, param in highway.pooler.state_dict().items():
                param.copy_(loaded_model[name])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        all_highway_exits = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask)
            hidden_states = layer_outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
            current_outputs = hidden_states,
            if self.output_hidden_states:
                current_outputs = current_outputs + (all_hidden_states,)
            if self.output_attentions:
                current_outputs = current_outputs + (all_attentions,)
            highway_exit = self.highway[i](current_outputs)
            if not self.training:
                highway_logits = highway_exit[0]
                highway_entropy = entropy(highway_logits)
                highway_exit = highway_exit + (highway_entropy,)
                all_highway_exits = all_highway_exits + (highway_exit,)
                if highway_entropy < self.early_exit_entropy[i]:
                    new_output = (highway_logits,) + current_outputs[1:] + (all_highway_exits,)
                    raise HighwayException(new_output, i + 1)
            else:
                all_highway_exits = all_highway_exits + (highway_exit,)
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        outputs = hidden_states,
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        outputs = outputs + (all_highway_exits,)
        return outputs


class RetrievalQAEmbedder(torch.nn.Module):

    def __init__(self, sent_encoder, dim):
        super(RetrievalQAEmbedder, self).__init__()
        self.sent_encoder = sent_encoder
        self.output_dim = 128
        self.project_q = torch.nn.Linear(dim, self.output_dim, bias=False)
        self.project_a = torch.nn.Linear(dim, self.output_dim, bias=False)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def embed_sentences_checkpointed(self, input_ids, attention_mask, checkpoint_batch_size=-1):
        if checkpoint_batch_size < 0 or input_ids.shape[0] < checkpoint_batch_size:
            return self.sent_encoder(input_ids, attention_mask=attention_mask)[1]
        else:
            device = input_ids.device
            input_shape = input_ids.size()
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            head_mask = [None] * self.sent_encoder.config.num_hidden_layers
            extended_attention_mask: 'torch.Tensor' = self.sent_encoder.get_extended_attention_mask(attention_mask, input_shape, device)

            def partial_encode(*inputs):
                encoder_outputs = self.sent_encoder.encoder(inputs[0], attention_mask=inputs[1], head_mask=head_mask)
                sequence_output = encoder_outputs[0]
                pooled_output = self.sent_encoder.pooler(sequence_output)
                return pooled_output
            embedding_output = self.sent_encoder.embeddings(input_ids=input_ids, position_ids=None, token_type_ids=token_type_ids, inputs_embeds=None)
            pooled_output_list = []
            for b in range(math.ceil(input_ids.shape[0] / checkpoint_batch_size)):
                b_embedding_output = embedding_output[b * checkpoint_batch_size:(b + 1) * checkpoint_batch_size]
                b_attention_mask = extended_attention_mask[b * checkpoint_batch_size:(b + 1) * checkpoint_batch_size]
                pooled_output = checkpoint.checkpoint(partial_encode, b_embedding_output, b_attention_mask)
                pooled_output_list.append(pooled_output)
            return torch.cat(pooled_output_list, dim=0)

    def embed_questions(self, q_ids, q_mask, checkpoint_batch_size=-1):
        q_reps = self.embed_sentences_checkpointed(q_ids, q_mask, checkpoint_batch_size)
        return self.project_q(q_reps)

    def embed_answers(self, a_ids, a_mask, checkpoint_batch_size=-1):
        a_reps = self.embed_sentences_checkpointed(a_ids, a_mask, checkpoint_batch_size)
        return self.project_a(a_reps)

    def forward(self, q_ids, q_mask, a_ids, a_mask, checkpoint_batch_size=-1):
        device = q_ids.device
        q_reps = self.embed_questions(q_ids, q_mask, checkpoint_batch_size)
        a_reps = self.embed_answers(a_ids, a_mask, checkpoint_batch_size)
        compare_scores = torch.mm(q_reps, a_reps.t())
        loss_qa = self.ce_loss(compare_scores, torch.arange(compare_scores.shape[1]))
        loss_aq = self.ce_loss(compare_scores.t(), torch.arange(compare_scores.shape[0]))
        loss = (loss_qa + loss_aq) / 2
        return loss


class LoRALayer:

    def __init__(self, r: 'int', lora_alpha: 'int', lora_dropout: 'float', merge_weights: 'bool'):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights


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


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = 'p5'

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from C5 feature.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.num_levels = 2
        self.in_feature = 'res5'
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


def get_norm(norm, out_channels):
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {'BN': BatchNorm2d, 'GN': lambda channels: nn.GroupNorm(32, channels), 'nnSyncBN': nn.SyncBatchNorm, '': lambda x: x}[norm]
    return norm(out_channels)


class BasicStem(nn.Module):

    def __init__(self, in_channels=3, out_channels=64, norm='BN', caffe_maxpool=False):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False, norm=get_norm(norm, out_channels))
        self.caffe_maxpool = caffe_maxpool

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        if self.caffe_maxpool:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0, ceil_mode=True)
        else:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

    @property
    def out_channels(self):
        return self.conv1.out_channels

    @property
    def stride(self):
        return 4


class ResNetBlockBase(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        return self


class BottleneckBlock(ResNetBlockBase):

    def __init__(self, in_channels, out_channels, bottleneck_channels, stride=1, num_groups=1, norm='BN', stride_in_1x1=False, dilation=1):
        super().__init__(in_channels, out_channels, stride)
        if in_channels != out_channels:
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, norm=get_norm(norm, out_channels))
        else:
            self.shortcut = None
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        self.conv1 = Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=stride_1x1, bias=False, norm=get_norm(norm, bottleneck_channels))
        self.conv2 = Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride_3x3, padding=1 * dilation, bias=False, groups=num_groups, dilation=dilation, norm=get_norm(norm, bottleneck_channels))
        self.conv3 = Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False, norm=get_norm(norm, out_channels))

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)
        out = F.relu_(out)
        out = self.conv3(out)
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        out += shortcut
        out = F.relu_(out)
        return out


class ShapeSpec(namedtuple('_ShapeSpec', ['channels', 'height', 'width', 'stride'])):

    def __new__(cls, *, channels=None, height=None, width=None, stride=None):
        return super().__new__(cls, channels, height, width, stride)


class Backbone(nn.Module, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):
        pass

    @property
    def size_divisibility(self):
        """
        Some backbones require the input height and width to be divisible by a specific integer. This is
        typically true for encoder / decoder type networks with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific input size divisibility is required.
        """
        return 0

    def output_shape(self):
        return {name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]) for name in self._out_features}

    @property
    def out_features(self):
        """deprecated"""
        return self._out_features

    @property
    def out_feature_strides(self):
        """deprecated"""
        return {f: self._out_feature_strides[f] for f in self._out_features}

    @property
    def out_feature_channels(self):
        """deprecated"""
        return {f: self._out_feature_channels[f] for f in self._out_features}


class ResNet(Backbone):

    def __init__(self, stem, stages, num_classes=None, out_features=None):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[ResNetBlock]]): several (typically 4) stages, each contains multiple :class:`ResNetBlockBase`.
            num_classes (None or int): if None, will not perform classification.
            out_features (list[str]): name of the layers whose outputs should be returned in forward. Can be anything in:
            "stem", "linear", or "res2" ... If None, will return the output of the last layer.
        """
        super(ResNet, self).__init__()
        self.stem = stem
        self.num_classes = num_classes
        current_stride = self.stem.stride
        self._out_feature_strides = {'stem': current_stride}
        self._out_feature_channels = {'stem': self.stem.out_channels}
        self.stages_and_names = []
        for i, blocks in enumerate(stages):
            for block in blocks:
                assert isinstance(block, ResNetBlockBase), block
                curr_channels = block.out_channels
            stage = nn.Sequential(*blocks)
            name = 'res' + str(i + 2)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            self._out_feature_strides[name] = current_stride = int(current_stride * np.prod([k.stride for k in blocks]))
            self._out_feature_channels[name] = blocks[-1].out_channels
        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)
            nn.init.normal_(self.linear.weight, stddev=0.01)
            name = 'linear'
        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, 'Available children: {}'.format(', '.join(children))

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if 'stem' in self._out_features:
            outputs['stem'] = x
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = self.linear(x)
            if 'linear' in self._out_features:
                outputs['linear'] = x
        return outputs

    def output_shape(self):
        return {name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]) for name in self._out_features}

    @staticmethod
    def make_stage(block_class, num_blocks, first_stride=None, *, in_channels, out_channels, **kwargs):
        """
        Usually, layers that produce the same feature map spatial size
        are defined as one "stage".
        Under such definition, stride_per_block[1:] should all be 1.
        """
        if first_stride is not None:
            assert 'stride' not in kwargs and 'stride_per_block' not in kwargs
            kwargs['stride_per_block'] = [first_stride] + [1] * (num_blocks - 1)
        blocks = []
        for i in range(num_blocks):
            curr_kwargs = {}
            for k, v in kwargs.items():
                if k.endswith('_per_block'):
                    assert len(v) == num_blocks, f"Argument '{k}' of make_stage should have the same length as num_blocks={num_blocks}."
                    newk = k[:-len('_per_block')]
                    assert newk not in kwargs, f'Cannot call make_stage with both {k} and {newk}!'
                    curr_kwargs[newk] = v[i]
                else:
                    curr_kwargs[k] = v
            blocks.append(block_class(in_channels=in_channels, out_channels=out_channels, **curr_kwargs))
            in_channels = out_channels
        return blocks


def assign_boxes_to_levels(box_lists: 'List[torch.Tensor]', min_level: 'int', max_level: 'int', canonical_box_size: 'int', canonical_level: 'int'):
    box_sizes = torch.sqrt(torch.cat([boxes.area() for boxes in box_lists]))
    level_assignments = torch.floor(canonical_level + torch.log2(box_sizes / canonical_box_size + 1e-08))
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    return level_assignments - min_level


def _fmt_box_list(box_tensor, batch_index: 'int'):
    repeated_index = torch.full((len(box_tensor), 1), batch_index, dtype=box_tensor.dtype, device=box_tensor.device)
    return torch.cat((repeated_index, box_tensor), dim=1)


def convert_boxes_to_pooler_format(box_lists: 'List[torch.Tensor]'):
    pooler_fmt_boxes = torch.cat([_fmt_box_list(box_list, i) for i, box_list in enumerate(box_lists)], dim=0)
    return pooler_fmt_boxes


class ROIPooler(nn.Module):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(self, output_size, scales, sampling_ratio, canonical_box_size=224, canonical_level=4):
        super().__init__()
        min_level = -math.log2(scales[0])
        max_level = -math.log2(scales[-1])
        assert math.isclose(min_level, int(min_level)) and math.isclose(max_level, int(max_level))
        assert len(scales) == max_level - min_level + 1, 'not pyramid'
        assert 0 < min_level and min_level <= max_level
        if isinstance(output_size, int):
            output_size = output_size, output_size
        assert len(output_size) == 2 and isinstance(output_size[0], int) and isinstance(output_size[1], int)
        if len(scales) > 1:
            assert min_level <= canonical_level and canonical_level <= max_level
        assert canonical_box_size > 0
        self.output_size = output_size
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        self.level_poolers = nn.ModuleList(RoIPool(output_size, spatial_scale=scale) for scale in scales)
        self.canonical_level = canonical_level
        self.canonical_box_size = canonical_box_size

    def forward(self, feature_maps, boxes):
        """
        Args:
            feature_maps: List[torch.Tensor(N,C,W,H)]
            box_lists: list[torch.Tensor])
        Returns:
            A tensor of shape(N*B, Channels, output_size, output_size)
        """
        x = [v for v in feature_maps.values()]
        num_level_assignments = len(self.level_poolers)
        assert len(x) == num_level_assignments and len(boxes) == x[0].size(0)
        pooler_fmt_boxes = convert_boxes_to_pooler_format(boxes)
        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)
        level_assignments = assign_boxes_to_levels(boxes, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level)
        num_boxes = len(pooler_fmt_boxes)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]
        dtype, device = x[0].dtype, x[0].device
        output = torch.zeros((num_boxes, num_channels, output_size, output_size), dtype=dtype, device=device)
        for level, (x_level, pooler) in enumerate(zip(x, self.level_poolers)):
            inds = torch.nonzero(level_assignments == level).squeeze(1)
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            output[inds] = pooler(x_level, pooler_fmt_boxes_level)
        return output


class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4, use_attr=False, num_attrs=-1):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int)
            cls_agnostic_bbox_reg (bool)
            box_dim (int)
        """
        super().__init__()
        if not isinstance(input_size, int):
            input_size = np.prod(input_size)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)
        self.use_attr = use_attr
        if use_attr:
            """
            Modifications for VG in RoI heads
            Embedding: {num_classes + 1} --> {input_size // 8}
            Linear: {input_size + input_size // 8} --> {input_size // 4}
            Linear: {input_size // 4} --> {num_attrs + 1}
            """
            self.cls_embedding = nn.Embedding(num_classes + 1, input_size // 8)
            self.fc_attr = nn.Linear(input_size + input_size // 8, input_size // 4)
            self.attr_score = nn.Linear(input_size // 4, num_attrs + 1)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for item in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(item.bias, 0)

    def forward(self, roi_features):
        if roi_features.dim() > 2:
            roi_features = torch.flatten(roi_features, start_dim=1)
        scores = self.cls_score(roi_features)
        proposal_deltas = self.bbox_pred(roi_features)
        if self.use_attr:
            _, max_class = scores.max(-1)
            cls_emb = self.cls_embedding(max_class)
            roi_features = torch.cat([roi_features, cls_emb], -1)
            roi_features = self.fc_attr(roi_features)
            roi_features = F.relu(roi_features)
            attr_scores = self.attr_score(roi_features)
            return scores, attr_scores, proposal_deltas
        else:
            return scores, proposal_deltas


class Res5ROIHeads(nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.
    It contains logic of cropping the regions, extract per-region features
    (by the res-5 block in this case), and make per-region predictions.
    """

    def __init__(self, cfg, input_shape):
        super().__init__()
        self.batch_size_per_image = cfg.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.ROI_HEADS.POSITIVE_FRACTION
        self.in_features = cfg.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt = cfg.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels = {k: v.channels for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg = cfg.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.stage_channel_factor = 2 ** 3
        self.out_channels = cfg.RESNETS.RES2_OUT_CHANNELS * self.stage_channel_factor
        pooler_resolution = cfg.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = 1.0 / self.feature_strides[self.in_features[0]],
        sampling_ratio = cfg.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        res5_halve = cfg.ROI_BOX_HEAD.RES5HALVE
        use_attr = cfg.ROI_BOX_HEAD.ATTR
        num_attrs = cfg.ROI_BOX_HEAD.NUM_ATTRS
        self.pooler = ROIPooler(output_size=pooler_resolution, scales=pooler_scales, sampling_ratio=sampling_ratio)
        self.res5 = self._build_res5_block(cfg)
        if not res5_halve:
            """
            Modifications for VG in RoI heads:
            1. Change the stride of conv1 and shortcut in Res5.Block1 from 2 to 1
            2. Modifying all conv2 with (padding: 1 --> 2) and (dilation: 1 --> 2)
            """
            self.res5[0].conv1.stride = 1, 1
            self.res5[0].shortcut.stride = 1, 1
            for i in range(3):
                self.res5[i].conv2.padding = 2, 2
                self.res5[i].conv2.dilation = 2, 2
        self.box_predictor = FastRCNNOutputLayers(self.out_channels, self.num_classes, self.cls_agnostic_bbox_reg, use_attr=use_attr, num_attrs=num_attrs)

    def _build_res5_block(self, cfg):
        stage_channel_factor = self.stage_channel_factor
        num_groups = cfg.RESNETS.NUM_GROUPS
        width_per_group = cfg.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels = num_groups * width_per_group * stage_channel_factor
        out_channels = self.out_channels
        stride_in_1x1 = cfg.RESNETS.STRIDE_IN_1X1
        norm = cfg.RESNETS.NORM
        blocks = ResNet.make_stage(BottleneckBlock, 3, first_stride=2, in_channels=out_channels // 2, bottleneck_channels=bottleneck_channels, out_channels=out_channels, num_groups=num_groups, norm=norm, stride_in_1x1=stride_in_1x1)
        return nn.Sequential(*blocks)

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def forward(self, features, proposal_boxes, gt_boxes=None):
        if self.training:
            """
            see https://github.com/airsplay/py-bottom-up-attention/                    blob/master/detectron2/modeling/roi_heads/roi_heads.py
            """
            raise NotImplementedError()
        assert not proposal_boxes[0].requires_grad
        box_features = self._shared_roi_transform(features, proposal_boxes)
        feature_pooled = box_features.mean(dim=[2, 3])
        obj_logits, attr_logits, pred_proposal_deltas = self.box_predictor(feature_pooled)
        return obj_logits, attr_logits, pred_proposal_deltas, feature_pooled


def _create_grid_offsets(size: 'List[int]', stride: 'int', offset: 'float', device):
    grid_height, grid_width = size
    shifts_x = torch.arange(offset * stride, grid_width * stride, step=stride, dtype=torch.float32, device=device)
    shifts_y = torch.arange(offset * stride, grid_height * stride, step=stride, dtype=torch.float32, device=device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return shift_x, shift_y


class AnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set of anchors.
    """

    def __init__(self, cfg, input_shape: 'List[ShapeSpec]'):
        super().__init__()
        sizes = cfg.ANCHOR_GENERATOR.SIZES
        aspect_ratios = cfg.ANCHOR_GENERATOR.ASPECT_RATIOS
        self.strides = [x.stride for x in input_shape]
        self.offset = cfg.ANCHOR_GENERATOR.OFFSET
        assert 0.0 <= self.offset < 1.0, self.offset
        """
        sizes (list[list[int]]): sizes[i] is the list of anchor sizes for feat map i
            1. given in absolute lengths in units of the input image;
            2. they do not dynamically scale if the input image size changes.
        aspect_ratios (list[list[float]])
        strides (list[int]): stride of each input feature.
        """
        self.num_features = len(self.strides)
        self.cell_anchors = nn.ParameterList(self._calculate_anchors(sizes, aspect_ratios))
        self._spacial_feat_dim = 4

    def _calculate_anchors(self, sizes, aspect_ratios):
        if len(sizes) == 1:
            sizes *= self.num_features
        if len(aspect_ratios) == 1:
            aspect_ratios *= self.num_features
        assert self.num_features == len(sizes)
        assert self.num_features == len(aspect_ratios)
        cell_anchors = [self.generate_cell_anchors(s, a).float() for s, a in zip(sizes, aspect_ratios)]
        return cell_anchors

    @property
    def box_dim(self):
        return self._spacial_feat_dim

    @property
    def num_cell_anchors(self):
        """
        Returns:
            list[int]: Each int is the number of anchors at every pixel location, on that feature map.
        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides, self.cell_anchors):
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, base_anchors.device)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))
        return anchors

    def generate_cell_anchors(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
        """
        anchors are continuous geometric rectangles
        centered on one feature map point sample.
        We can later build the set of anchors
        for the entire feature map by tiling these tensors
        """
        anchors = []
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        return nn.Parameter(torch.Tensor(anchors))

    def forward(self, features):
        """
        Args:
            features List[torch.Tensor]: list of feature maps on which to generate anchors.
        Returns:
            torch.Tensor: a list of #image elements.
        """
        num_images = features[0].size(0)
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
        anchors_over_all_feature_maps = torch.stack(anchors_over_all_feature_maps)
        return anchors_over_all_feature_maps.unsqueeze(0).repeat_interleave(num_images, dim=0)


class RPNHead(nn.Module):
    """
    RPN classification and regression heads. Uses a 3x3 conv to produce a shared
    hidden state from which one 1x1 conv predicts objectness logits for each anchor
    and a second 1x1 conv predicts bounding-box deltas specifying how to deform
    each anchor into an object proposal.
    """

    def __init__(self, cfg, input_shape: 'List[ShapeSpec]'):
        super().__init__()
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, 'Each level must have the same channel!'
        in_channels = in_channels[0]
        anchor_generator = AnchorGenerator(cfg, input_shape)
        num_cell_anchors = anchor_generator.num_cell_anchors
        box_dim = anchor_generator.box_dim
        assert len(set(num_cell_anchors)) == 1, 'Each level must have the same number of cell anchors'
        num_cell_anchors = num_cell_anchors[0]
        if cfg.PROPOSAL_GENERATOR.HIDDEN_CHANNELS == -1:
            hid_channels = in_channels
        else:
            hid_channels = cfg.PROPOSAL_GENERATOR.HIDDEN_CHANNELS
        self.conv = nn.Conv2d(in_channels, hid_channels, kernel_size=3, stride=1, padding=1)
        self.objectness_logits = nn.Conv2d(hid_channels, num_cell_anchors, kernel_size=1, stride=1)
        self.anchor_deltas = nn.Conv2d(hid_channels, num_cell_anchors * box_dim, kernel_size=1, stride=1)
        for layer in [self.conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of feature maps
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = F.relu(self.conv(x))
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas


class Box2BoxTransform(object):
    """
    This R-CNN transformation scales the box's width and height
    by exp(dw), exp(dh) and shifts a box's center by the offset
    (dx * width, dy * height).
    """

    def __init__(self, weights: 'Tuple[float, float, float, float]', scale_clamp: 'float'=None):
        """
        Args:
            weights (4-element tuple): Scaling factors that are applied to the
                (dx, dy, dw, dh) deltas. In Fast R-CNN, these were originally set
                such that the deltas have unit variance; now they are treated as
                hyperparameters of the system.
            scale_clamp (float): When predicting deltas, the predicted box scaling
                factors (dw and dh) are clamped such that they are <= scale_clamp.
        """
        self.weights = weights
        if scale_clamp is not None:
            self.scale_clamp = scale_clamp
        else:
            """
            Value for clamping large dw and dh predictions.
            The heuristic is that we clamp such that dw and dh are no larger
            than what would transform a 16px box into a 1000px box
            (based on a small anchor, 16px, and a typical image size, 1000px).
            """
            self.scale_clamp = math.log(1000.0 / 16)

    def get_deltas(self, src_boxes, target_boxes):
        """
        Get box regression transformation deltas (dx, dy, dw, dh) that can be used
        to transform the `src_boxes` into the `target_boxes`. That is, the relation
        ``target_boxes == self.apply_deltas(deltas, src_boxes)`` is true (unless
        any delta is too large and is clamped).
        Args:
            src_boxes (Tensor): source boxes, e.g., object proposals
            target_boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)
        src_widths = src_boxes[:, 2] - src_boxes[:, 0]
        src_heights = src_boxes[:, 3] - src_boxes[:, 1]
        src_ctr_x = src_boxes[:, 0] + 0.5 * src_widths
        src_ctr_y = src_boxes[:, 1] + 0.5 * src_heights
        target_widths = target_boxes[:, 2] - target_boxes[:, 0]
        target_heights = target_boxes[:, 3] - target_boxes[:, 1]
        target_ctr_x = target_boxes[:, 0] + 0.5 * target_widths
        target_ctr_y = target_boxes[:, 1] + 0.5 * target_heights
        wx, wy, ww, wh = self.weights
        dx = wx * (target_ctr_x - src_ctr_x) / src_widths
        dy = wy * (target_ctr_y - src_ctr_y) / src_heights
        dw = ww * torch.log(target_widths / src_widths)
        dh = wh * torch.log(target_heights / src_heights)
        deltas = torch.stack((dx, dy, dw, dh), dim=1)
        assert (src_widths > 0).all().item(), 'Input boxes to Box2BoxTransform are not valid!'
        return deltas

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.
        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        wx, wy, ww, wh = self.weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]
        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
        return pred_boxes


class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be matched to zero or more predicted elements.
    The matching is determined by the MxN match_quality_matrix, that characterizes
    how well each (ground-truth, prediction)-pair match each other. For example,
    if the elements are boxes, this matrix may contain box intersection-over-union
    overlap values.
    The matcher returns (a) a vector of length N containing the index of the
    ground-truth element m in [0, M) that matches to prediction n in [0, N).
    (b) a vector of length N containing the labels for each prediction.
    """

    def __init__(self, thresholds: 'List[float]', labels: 'List[int]', allow_low_quality_matches: 'bool'=False):
        """
        Args:
            thresholds (list): a list of thresholds used to stratify predictions
                into levels.
            labels (list): a list of values to label predictions belonging at
                each level. A label can be one of {-1, 0, 1} signifying
                {ignore, negative class, positive class}, respectively.
            allow_low_quality_matches (bool): if True, produce additional matches or predictions with maximum match quality lower than high_threshold.
                For example, thresholds = [0.3, 0.5] labels = [0, -1, 1] All predictions with iou < 0.3 will be marked with 0 and
                thus will be considered as false positives while training. All predictions with 0.3 <= iou < 0.5 will be marked with -1 and
                thus will be ignored. All predictions with 0.5 <= iou will be marked with 1 and thus will be considered as true positives.
        """
        thresholds = thresholds[:]
        assert thresholds[0] > 0
        thresholds.insert(0, -float('inf'))
        thresholds.append(float('inf'))
        assert all([(low <= high) for low, high in zip(thresholds[:-1], thresholds[1:])])
        assert all([(label_i in [-1, 0, 1]) for label_i in labels])
        assert len(labels) == len(thresholds) - 1
        self.thresholds = thresholds
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the pairwise quality between M ground-truth elements and N predicted
                elements. All elements must be >= 0 (due to the us of `torch.nonzero` for selecting indices in :meth:`set_low_quality_matches_`).
        Returns:
            matches (Tensor[int64]): a vector of length N, where matches[i] is a matched ground-truth index in [0, M)
            match_labels (Tensor[int8]): a vector of length N, where pred_labels[i] indicates true or false positive or ignored
        """
        assert match_quality_matrix.dim() == 2
        if match_quality_matrix.numel() == 0:
            default_matches = match_quality_matrix.new_full((match_quality_matrix.size(1),), 0, dtype=torch.int64)
            default_match_labels = match_quality_matrix.new_full((match_quality_matrix.size(1),), self.labels[0], dtype=torch.int8)
            return default_matches, default_match_labels
        assert torch.all(match_quality_matrix >= 0)
        matched_vals, matches = match_quality_matrix.max(dim=0)
        match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)
        for l, low, high in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):
            low_high = (matched_vals >= low) & (matched_vals < high)
            match_labels[low_high] = l
        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(match_labels, match_quality_matrix)
        return matches, match_labels

    def set_low_quality_matches_(self, match_labels, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth G find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth G.
        This function implements the RPN assignment case (i)
        in Sec. 3.1.2 of Faster R-CNN.
        """
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        of_quality_inds = match_quality_matrix == highest_quality_foreach_gt[:, None]
        if of_quality_inds.dim() == 0:
            _, pred_inds_with_highest_quality = of_quality_inds.unsqueeze(0).nonzero().unbind(1)
        else:
            _, pred_inds_with_highest_quality = of_quality_inds.nonzero().unbind(1)
        match_labels[pred_inds_with_highest_quality] = 1


class RPNOutputs(object):

    def __init__(self, box2box_transform, anchor_matcher, batch_size_per_image, positive_fraction, images, pred_objectness_logits, pred_anchor_deltas, anchors, boundary_threshold=0, gt_boxes=None, smooth_l1_beta=0.0):
        """
        Args:
            box2box_transform (Box2BoxTransform): :class:`Box2BoxTransform` instance for anchor-proposal transformations.
            anchor_matcher (Matcher): :class:`Matcher` instance for matching anchors to ground-truth boxes; used to determine training labels.
            batch_size_per_image (int): number of proposals to sample when training
            positive_fraction (float): target fraction of sampled proposals that should be positive
            images (ImageList): :class:`ImageList` instance representing N input images
            pred_objectness_logits (list[Tensor]): A list of L elements. Element i is a tensor of shape (N, A, Hi, W)
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape (N, A*4, Hi, Wi)
            anchors (list[torch.Tensor]): nested list of boxes. anchors[i][j] at (n, l) stores anchor array for feature map l
            boundary_threshold (int): if >= 0, then anchors that extend beyond the image boundary by more than boundary_thresh are not used in training.
            gt_boxes (list[Boxes], optional): A list of N elements.
            smooth_l1_beta (float): The transition point between L1 and L2 lossn. When set to 0, the loss becomes L1. When +inf, it is ignored
        """
        self.box2box_transform = box2box_transform
        self.anchor_matcher = anchor_matcher
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.pred_objectness_logits = pred_objectness_logits
        self.pred_anchor_deltas = pred_anchor_deltas
        self.anchors = anchors
        self.gt_boxes = gt_boxes
        self.num_feature_maps = len(pred_objectness_logits)
        self.num_images = len(images)
        self.boundary_threshold = boundary_threshold
        self.smooth_l1_beta = smooth_l1_beta

    def _get_ground_truth(self):
        raise NotImplementedError()

    def predict_proposals(self):
        proposals = []
        anchors = self.anchors.transpose(0, 1)
        for anchors_i, pred_anchor_deltas_i in zip(anchors, self.pred_anchor_deltas):
            B = anchors_i.size(-1)
            N, _, Hi, Wi = pred_anchor_deltas_i.shape
            anchors_i = anchors_i.flatten(start_dim=0, end_dim=1)
            pred_anchor_deltas_i = pred_anchor_deltas_i.view(N, -1, B, Hi, Wi).permute(0, 3, 4, 1, 2).reshape(-1, B)
            proposals_i = self.box2box_transform.apply_deltas(pred_anchor_deltas_i, anchors_i)
            proposals.append(proposals_i.view(N, -1, B))
        proposals = torch.stack(proposals)
        return proposals

    def predict_objectness_logits(self):
        """
        Returns:
            pred_objectness_logits (list[Tensor]) -> (N, Hi*Wi*A).
        """
        pred_objectness_logits = [score.permute(0, 2, 3, 1).reshape(self.num_images, -1) for score in self.pred_objectness_logits]
        return pred_objectness_logits


def _clip_box(tensor, box_size: 'Tuple[int, int]'):
    assert torch.isfinite(tensor).all(), 'Box tensor contains infinite or NaN!'
    h, w = box_size
    tensor[:, 0].clamp_(min=0, max=w)
    tensor[:, 1].clamp_(min=0, max=h)
    tensor[:, 2].clamp_(min=0, max=w)
    tensor[:, 3].clamp_(min=0, max=h)


def _nonempty_boxes(box, threshold: 'float'=0.0) ->torch.Tensor:
    widths = box[:, 2] - box[:, 0]
    heights = box[:, 3] - box[:, 1]
    keep = (widths > threshold) & (heights > threshold)
    return keep


def find_top_rpn_proposals(proposals, pred_objectness_logits, images, image_sizes, nms_thresh, pre_nms_topk, post_nms_topk, min_box_side_len, training):
    """Args:
        proposals (list[Tensor]): (L, N, Hi*Wi*A, 4).
        pred_objectness_logits: tensors of length L.
        nms_thresh (float): IoU threshold to use for NMS
        pre_nms_topk (int): before nms
        post_nms_topk (int): after nms
        min_box_side_len (float): minimum proposal box side
        training (bool): True if proposals are to be used in training,
    Returns:
        results (List[Dict]): stores post_nms_topk object proposals for image i.
    """
    num_images = len(images)
    device = proposals[0].device
    topk_scores = []
    topk_proposals = []
    level_ids = []
    batch_idx = torch.arange(num_images, device=device)
    for level_id, proposals_i, logits_i in zip(itertools.count(), proposals, pred_objectness_logits):
        Hi_Wi_A = logits_i.shape[1]
        num_proposals_i = min(pre_nms_topk, Hi_Wi_A)
        logits_i, idx = logits_i.sort(descending=True, dim=1)
        topk_scores_i = logits_i[batch_idx, :num_proposals_i]
        topk_idx = idx[batch_idx, :num_proposals_i]
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]
        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        level_ids.append(torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device))
    topk_scores = torch.cat(topk_scores, dim=1)
    topk_proposals = torch.cat(topk_proposals, dim=1)
    level_ids = torch.cat(level_ids, dim=0)
    results = []
    for n, image_size in enumerate(image_sizes):
        boxes = topk_proposals[n]
        scores_per_img = topk_scores[n]
        _clip_box(boxes, image_size)
        keep = _nonempty_boxes(boxes, threshold=min_box_side_len)
        lvl = level_ids
        if keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], level_ids[keep]
        keep = batched_nms(boxes, scores_per_img, lvl, nms_thresh)
        keep = keep[:post_nms_topk]
        res = boxes[keep], scores_per_img[keep]
        results.append(res)
    return results


class RPN(nn.Module):
    """
    Region Proposal Network, introduced by the Faster R-CNN paper.
    """

    def __init__(self, cfg, input_shape: 'Dict[str, ShapeSpec]'):
        super().__init__()
        self.min_box_side_len = cfg.PROPOSAL_GENERATOR.MIN_SIZE
        self.in_features = cfg.RPN.IN_FEATURES
        self.nms_thresh = cfg.RPN.NMS_THRESH
        self.batch_size_per_image = cfg.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_fraction = cfg.RPN.POSITIVE_FRACTION
        self.smooth_l1_beta = cfg.RPN.SMOOTH_L1_BETA
        self.loss_weight = cfg.RPN.LOSS_WEIGHT
        self.pre_nms_topk = {(True): cfg.RPN.PRE_NMS_TOPK_TRAIN, (False): cfg.RPN.PRE_NMS_TOPK_TEST}
        self.post_nms_topk = {(True): cfg.RPN.POST_NMS_TOPK_TRAIN, (False): cfg.RPN.POST_NMS_TOPK_TEST}
        self.boundary_threshold = cfg.RPN.BOUNDARY_THRESH
        self.anchor_generator = AnchorGenerator(cfg, [input_shape[f] for f in self.in_features])
        self.box2box_transform = Box2BoxTransform(weights=cfg.RPN.BBOX_REG_WEIGHTS)
        self.anchor_matcher = Matcher(cfg.RPN.IOU_THRESHOLDS, cfg.RPN.IOU_LABELS, allow_low_quality_matches=True)
        self.rpn_head = RPNHead(cfg, [input_shape[f] for f in self.in_features])

    def training(self, images, image_shapes, features, gt_boxes):
        pass

    def inference(self, outputs, images, image_shapes, features, gt_boxes=None):
        outputs = find_top_rpn_proposals(outputs.predict_proposals(), outputs.predict_objectness_logits(), images, image_shapes, self.nms_thresh, self.pre_nms_topk[self.training], self.post_nms_topk[self.training], self.min_box_side_len, self.training)
        results = []
        for img in outputs:
            im_boxes, img_box_logits = img
            img_box_logits, inds = img_box_logits.sort(descending=True)
            im_boxes = im_boxes[inds]
            results.append((im_boxes, img_box_logits))
        proposal_boxes, logits = tuple(map(list, zip(*results)))
        return proposal_boxes, logits

    def forward(self, images, image_shapes, features, gt_boxes=None):
        """
        Args:
            images (torch.Tensor): input images of length `N`
            features (dict[str: Tensor])
            gt_instances
        """
        features = [features[f] for f in self.in_features]
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        anchors = self.anchor_generator(features)
        outputs = RPNOutputs(self.box2box_transform, self.anchor_matcher, self.batch_size_per_image, self.positive_fraction, images, pred_objectness_logits, pred_anchor_deltas, anchors, self.boundary_threshold, gt_boxes, self.smooth_l1_beta)
        if self.training:
            raise NotImplementedError()
            return self.training(outputs, images, image_shapes, features, gt_boxes)
        else:
            return self.inference(outputs, images, image_shapes, features, gt_boxes)


class Config:
    _pointer = {}

    def __init__(self, dictionary: 'dict', name: 'str'='root', level=0):
        self._name = name
        self._level = level
        d = {}
        for k, v in dictionary.items():
            if v is None:
                raise ValueError()
            k = copy.deepcopy(k)
            v = copy.deepcopy(v)
            if isinstance(v, dict):
                v = Config(v, name=k, level=level + 1)
            d[k] = v
            setattr(self, k, v)
        self._pointer = d

    def __repr__(self):
        return str(list(self._pointer.keys()))

    def __setattr__(self, key, val):
        self.__dict__[key] = val
        self.__dict__[key.upper()] = val
        levels = key.split('.')
        last_level = len(levels) - 1
        pointer = self._pointer
        if len(levels) > 1:
            for i, l in enumerate(levels):
                if hasattr(self, l) and isinstance(getattr(self, l), Config):
                    setattr(getattr(self, l), '.'.join(levels[i:]), val)
                if l == last_level:
                    pointer[l] = val
                else:
                    pointer = pointer[l]

    def to_dict(self):
        return self._pointer

    def dump_yaml(self, data, file_name):
        with open(f'{file_name}', 'w') as stream:
            dump(data, stream)

    def dump_json(self, data, file_name):
        with open(f'{file_name}', 'w') as stream:
            json.dump(data, stream)

    @staticmethod
    def load_yaml(config):
        with open(config) as stream:
            data = load(stream, Loader=Loader)
        return data

    def __str__(self):
        t = '    '
        if self._name != 'root':
            r = f'{t * (self._level - 1)}{self._name}:\n'
        else:
            r = ''
        level = self._level
        for i, (k, v) in enumerate(self._pointer.items()):
            if isinstance(v, Config):
                r += f'{t * self._level}{v}\n'
                self._level += 1
            else:
                r += f'{t * self._level}{k}: {v} ({type(v).__name__})\n'
            self._level = level
        return r[:-1]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: 'str', **kwargs):
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        return cls(config_dict)

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path: 'str', **kwargs):
        cache_dir = kwargs.pop('cache_dir', None)
        force_download = kwargs.pop('force_download', False)
        resume_download = kwargs.pop('resume_download', False)
        proxies = kwargs.pop('proxies', None)
        local_files_only = kwargs.pop('local_files_only', False)
        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        else:
            config_file = hf_bucket_url(pretrained_model_name_or_path, filename=CONFIG_NAME, use_cdn=False)
        try:
            resolved_config_file = cached_path(config_file, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, local_files_only=local_files_only)
            if resolved_config_file is None:
                raise EnvironmentError
            config_file = Config.load_yaml(resolved_config_file)
        except EnvironmentError:
            msg = "Can't load config for"
            raise EnvironmentError(msg)
        if resolved_config_file == config_file:
            None
        else:
            None
        return Config.load_yaml(resolved_config_file), kwargs


def do_nms(boxes, scores, image_shape, score_thresh, nms_thresh, mind, maxd):
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    boxes = boxes.reshape(-1, 4)
    _clip_box(boxes, image_shape)
    boxes = boxes.view(-1, num_bbox_reg_classes, 4)
    max_scores, max_classes = scores.max(1)
    num_objs = boxes.size(0)
    boxes = boxes.view(-1, 4)
    idxs = torch.arange(num_objs) * num_bbox_reg_classes + max_classes
    max_boxes = boxes[idxs]
    keep = nms(max_boxes, max_scores, nms_thresh)
    keep = keep[:maxd]
    if keep.shape[-1] >= mind and keep.shape[-1] <= maxd:
        max_boxes, max_scores = max_boxes[keep], max_scores[keep]
        classes = max_classes[keep]
        return max_boxes, max_scores, classes, keep
    else:
        return None


class ROIOutputs(object):

    def __init__(self, cfg, training=False):
        self.smooth_l1_beta = cfg.ROI_BOX_HEAD.SMOOTH_L1_BETA
        self.box2box_transform = Box2BoxTransform(weights=cfg.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
        self.training = training
        self.score_thresh = cfg.ROI_HEADS.SCORE_THRESH_TEST
        self.min_detections = cfg.MIN_DETECTIONS
        self.max_detections = cfg.MAX_DETECTIONS
        nms_thresh = cfg.ROI_HEADS.NMS_THRESH_TEST
        if not isinstance(nms_thresh, list):
            nms_thresh = [nms_thresh]
        self.nms_thresh = nms_thresh

    def _predict_boxes(self, proposals, box_deltas, preds_per_image):
        num_pred = box_deltas.size(0)
        B = proposals[0].size(-1)
        K = box_deltas.size(-1) // B
        box_deltas = box_deltas.view(num_pred * K, B)
        proposals = torch.cat(proposals, dim=0).unsqueeze(-2).expand(num_pred, K, B)
        proposals = proposals.reshape(-1, B)
        boxes = self.box2box_transform.apply_deltas(box_deltas, proposals)
        return boxes.view(num_pred, K * B).split(preds_per_image, dim=0)

    def _predict_objs(self, obj_logits, preds_per_image):
        probs = F.softmax(obj_logits, dim=-1)
        probs = probs.split(preds_per_image, dim=0)
        return probs

    def _predict_attrs(self, attr_logits, preds_per_image):
        attr_logits = attr_logits[..., :-1].softmax(-1)
        attr_probs, attrs = attr_logits.max(-1)
        return attr_probs.split(preds_per_image, dim=0), attrs.split(preds_per_image, dim=0)

    @torch.no_grad()
    def inference(self, obj_logits, attr_logits, box_deltas, pred_boxes, features, sizes, scales=None):
        preds_per_image = [p.size(0) for p in pred_boxes]
        boxes_all = self._predict_boxes(pred_boxes, box_deltas, preds_per_image)
        obj_scores_all = self._predict_objs(obj_logits, preds_per_image)
        attr_probs_all, attrs_all = self._predict_attrs(attr_logits, preds_per_image)
        features = features.split(preds_per_image, dim=0)
        final_results = []
        zipped = zip(boxes_all, obj_scores_all, attr_probs_all, attrs_all, sizes)
        for i, (boxes, obj_scores, attr_probs, attrs, size) in enumerate(zipped):
            for nms_t in self.nms_thresh:
                outputs = do_nms(boxes, obj_scores, size, self.score_thresh, nms_t, self.min_detections, self.max_detections)
                if outputs is not None:
                    max_boxes, max_scores, classes, ids = outputs
                    break
            if scales is not None:
                scale_yx = scales[i]
                max_boxes[:, 0::2] *= scale_yx[1]
                max_boxes[:, 1::2] *= scale_yx[0]
            final_results.append((max_boxes, classes, max_scores, attrs[ids], attr_probs[ids], features[i][ids]))
        boxes, classes, class_probs, attrs, attr_probs, roi_features = map(list, zip(*final_results))
        return boxes, classes, class_probs, attrs, attr_probs, roi_features

    def training(self, obj_logits, attr_logits, box_deltas, pred_boxes, features, sizes):
        pass

    def __call__(self, obj_logits, attr_logits, box_deltas, pred_boxes, features, sizes, scales=None):
        if self.training:
            raise NotImplementedError()
        return self.inference(obj_logits, attr_logits, box_deltas, pred_boxes, features, sizes, scales=scales)


def build_backbone(cfg):
    input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
    norm = cfg.RESNETS.NORM
    stem = BasicStem(in_channels=input_shape.channels, out_channels=cfg.RESNETS.STEM_OUT_CHANNELS, norm=norm, caffe_maxpool=cfg.MODEL.MAX_POOL)
    freeze_at = cfg.BACKBONE.FREEZE_AT
    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False
    out_features = cfg.RESNETS.OUT_FEATURES
    depth = cfg.RESNETS.DEPTH
    num_groups = cfg.RESNETS.NUM_GROUPS
    width_per_group = cfg.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels = cfg.RESNETS.STEM_OUT_CHANNELS
    out_channels = cfg.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1 = cfg.RESNETS.STRIDE_IN_1X1
    res5_dilation = cfg.RESNETS.RES5_DILATION
    assert res5_dilation in {1, 2}, 'res5_dilation cannot be {}.'.format(res5_dilation)
    num_blocks_per_stage = {(50): [3, 4, 6, 3], (101): [3, 4, 23, 3], (152): [3, 8, 36, 3]}[depth]
    stages = []
    out_stage_idx = [{'res2': 2, 'res3': 3, 'res4': 4, 'res5': 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or stage_idx == 5 and dilation == 2 else 2
        stage_kargs = {'num_blocks': num_blocks_per_stage[idx], 'first_stride': first_stride, 'in_channels': in_channels, 'bottleneck_channels': bottleneck_channels, 'out_channels': out_channels, 'num_groups': num_groups, 'norm': norm, 'stride_in_1x1': stride_in_1x1, 'dilation': dilation}
        stage_kargs['block_class'] = BottleneckBlock
        blocks = ResNet.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        if freeze_at >= stage_idx:
            for block in blocks:
                block.freeze()
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features)


def load_checkpoint(ckp):
    r = OrderedDict()
    with open(ckp, 'rb') as f:
        ckp = pkl.load(f)['model']
    for k in copy.deepcopy(list(ckp.keys())):
        v = ckp.pop(k)
        if isinstance(v, np.ndarray):
            v = torch.tensor(v)
        else:
            assert isinstance(v, torch.tensor), type(v)
        r[k] = v
    return r


def norm_box(boxes, raw_sizes):
    if not isinstance(boxes, torch.Tensor):
        normalized_boxes = boxes.copy()
    else:
        normalized_boxes = boxes.clone()
    normalized_boxes[:, :, (0, 2)] /= raw_sizes[:, 1]
    normalized_boxes[:, :, (1, 3)] /= raw_sizes[:, 0]
    return normalized_boxes


def pad_list_tensors(list_tensors, preds_per_image, max_detections=None, return_tensors=None, padding=None, pad_value=0, location=None):
    """
    location will always be cpu for np tensors
    """
    if location is None:
        location = 'cpu'
    assert return_tensors in {'pt', 'np', None}
    assert padding in {'max_detections', 'max_batch', None}
    new = []
    if padding is None:
        if return_tensors is None:
            return list_tensors
        elif return_tensors == 'pt':
            if not isinstance(list_tensors, torch.Tensor):
                return torch.stack(list_tensors)
            else:
                return list_tensors
        elif not isinstance(list_tensors, list):
            return np.array(list_tensors)
        else:
            return list_tensors
    if padding == 'max_detections':
        assert max_detections is not None, 'specify max number of detections per batch'
    elif padding == 'max_batch':
        max_detections = max(preds_per_image)
    for i in range(len(list_tensors)):
        too_small = False
        tensor_i = list_tensors.pop(0)
        if tensor_i.ndim < 2:
            too_small = True
            tensor_i = tensor_i.unsqueeze(-1)
        assert isinstance(tensor_i, torch.Tensor)
        tensor_i = F.pad(input=tensor_i, pad=(0, 0, 0, max_detections - preds_per_image[i]), mode='constant', value=pad_value)
        if too_small:
            tensor_i = tensor_i.squeeze(-1)
        if return_tensors is None:
            if location == 'cpu':
                tensor_i = tensor_i.cpu()
            tensor_i = tensor_i.tolist()
        if return_tensors == 'np':
            if location == 'cpu':
                tensor_i = tensor_i.cpu()
            tensor_i = tensor_i.numpy()
        elif location == 'cpu':
            tensor_i = tensor_i.cpu()
        new.append(tensor_i)
    if return_tensors == 'np':
        return np.stack(new, axis=0)
    elif return_tensors == 'pt' and not isinstance(new, torch.Tensor):
        return torch.stack(new, dim=0)
    else:
        return list_tensors


class GeneralizedRCNN(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = RPN(cfg, self.backbone.output_shape())
        self.roi_heads = Res5ROIHeads(cfg, self.backbone.output_shape())
        self.roi_outputs = ROIOutputs(cfg)
        self

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop('config', None)
        state_dict = kwargs.pop('state_dict', None)
        cache_dir = kwargs.pop('cache_dir', None)
        from_tf = kwargs.pop('from_tf', False)
        force_download = kwargs.pop('force_download', False)
        resume_download = kwargs.pop('resume_download', False)
        proxies = kwargs.pop('proxies', None)
        local_files_only = kwargs.pop('local_files_only', False)
        use_cdn = kwargs.pop('use_cdn', True)
        if not isinstance(config, Config):
            config_path = config if config is not None else pretrained_model_name_or_path
            config = Config.from_pretrained(config_path, cache_dir=cache_dir, force_download=force_download, resume_download=resume_download, proxies=proxies, local_files_only=local_files_only)
        if pretrained_model_name_or_path is not None:
            if os.path.isdir(pretrained_model_name_or_path):
                if os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                else:
                    raise EnvironmentError('Error no file named {} found in directory {} '.format(WEIGHTS_NAME, pretrained_model_name_or_path))
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
            elif os.path.isfile(pretrained_model_name_or_path + '.index'):
                assert from_tf, 'We found a TensorFlow checkpoint at {}, please set from_tf to True to load from this checkpoint'.format(pretrained_model_name_or_path + '.index')
                archive_file = pretrained_model_name_or_path + '.index'
            else:
                archive_file = hf_bucket_url(pretrained_model_name_or_path, filename=WEIGHTS_NAME, use_cdn=use_cdn)
            try:
                resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, local_files_only=local_files_only)
                if resolved_archive_file is None:
                    raise EnvironmentError
            except EnvironmentError:
                msg = f"Can't load weights for '{pretrained_model_name_or_path}'."
                raise EnvironmentError(msg)
            if resolved_archive_file == archive_file:
                None
            else:
                None
        else:
            resolved_archive_file = None
        model = cls(config)
        if state_dict is None:
            try:
                try:
                    state_dict = torch.load(resolved_archive_file, map_location='cpu')
                except Exception:
                    state_dict = load_checkpoint(resolved_archive_file)
            except Exception:
                raise OSError('Unable to load weights from pytorch checkpoint file. If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. ')
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata
        model_to_load = model
        model_to_load.load_state_dict(state_dict)
        if model.__class__.__name__ != model_to_load.__class__.__name__:
            base_model_state_dict = model_to_load.state_dict().keys()
            head_model_state_dict_without_base_prefix = [key.split(cls.base_model_prefix + '.')[-1] for key in model.state_dict().keys()]
            missing_keys.extend(head_model_state_dict_without_base_prefix - base_model_state_dict)
        if len(unexpected_keys) > 0:
            None
        else:
            None
        if len(missing_keys) > 0:
            None
        else:
            None
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(model.__class__.__name__, '\n\t'.join(error_msgs)))
        model.eval()
        return model

    def forward(self, images, image_shapes, gt_boxes=None, proposals=None, scales_yx=None, **kwargs):
        """
        kwargs:
            max_detections (int), return_tensors {"np", "pt", None}, padding {None,
            "max_detections"}, pad_value (int), location = {"cuda", "cpu"}
        """
        if self.training:
            raise NotImplementedError()
        return self.inference(images=images, image_shapes=image_shapes, gt_boxes=gt_boxes, proposals=proposals, scales_yx=scales_yx, **kwargs)

    @torch.no_grad()
    def inference(self, images, image_shapes, gt_boxes=None, proposals=None, scales_yx=None, **kwargs):
        original_sizes = image_shapes * scales_yx
        features = self.backbone(images)
        if proposals is None:
            proposal_boxes, _ = self.proposal_generator(images, image_shapes, features, gt_boxes)
        else:
            assert proposals is not None
        obj_logits, attr_logits, box_deltas, feature_pooled = self.roi_heads(features, proposal_boxes, gt_boxes)
        boxes, classes, class_probs, attrs, attr_probs, roi_features = self.roi_outputs(obj_logits=obj_logits, attr_logits=attr_logits, box_deltas=box_deltas, pred_boxes=proposal_boxes, features=feature_pooled, sizes=image_shapes, scales=scales_yx)
        subset_kwargs = {'max_detections': kwargs.get('max_detections', None), 'return_tensors': kwargs.get('return_tensors', None), 'pad_value': kwargs.get('pad_value', 0), 'padding': kwargs.get('padding', None)}
        preds_per_image = torch.tensor([p.size(0) for p in boxes])
        boxes = pad_list_tensors(boxes, preds_per_image, **subset_kwargs)
        classes = pad_list_tensors(classes, preds_per_image, **subset_kwargs)
        class_probs = pad_list_tensors(class_probs, preds_per_image, **subset_kwargs)
        attrs = pad_list_tensors(attrs, preds_per_image, **subset_kwargs)
        attr_probs = pad_list_tensors(attr_probs, preds_per_image, **subset_kwargs)
        roi_features = pad_list_tensors(roi_features, preds_per_image, **subset_kwargs)
        subset_kwargs['padding'] = None
        preds_per_image = pad_list_tensors(preds_per_image, None, **subset_kwargs)
        sizes = pad_list_tensors(image_shapes, None, **subset_kwargs)
        normalized_boxes = norm_box(boxes, original_sizes)
        return OrderedDict({'obj_ids': classes, 'obj_probs': class_probs, 'attr_ids': attrs, 'attr_probs': attr_probs, 'boxes': boxes, 'sizes': sizes, 'preds_per_image': preds_per_image, 'roi_features': roi_features, 'normalized_boxes': normalized_boxes})


POOLING_BREAKDOWN = {(1): (1, 1), (2): (2, 1), (3): (3, 1), (4): (2, 2), (5): (5, 1), (6): (3, 2), (7): (7, 1), (8): (4, 2), (9): (3, 3)}


class ImageEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        model = torchvision.models.resnet152(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d(POOLING_BREAKDOWN[args.num_image_embeds])

    def forward(self, x):
        out = self.pool(self.model(x))
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out


class MagnitudeBinarizer(object):
    """
    Magnitude Binarizer.
    Computes a binary mask M from a real value matrix S such that `M_{i,j} = 1` if and only if `S_{i,j}`
    is among the k% highest values of |S| (absolute value).

    Implementation is inspired from https://github.com/NervanaSystems/distiller/blob/2291fdcc2ea642a98d4e20629acb5a9e2e04b4e6/distiller/pruning/automated_gradual_pruner.py#L24
    """

    @staticmethod
    def apply(inputs: 'torch.tensor', threshold: 'float'):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input matrix from which the binarizer computes the binary mask.
                This input marix is typically the weight matrix.
            threshold (`float`)
                The percentage of weights to keep (the rest is pruned).
                `threshold` is a float between 0 and 1.
        Returns:
            mask (`torch.FloatTensor`)
                Binary matrix of the same size as `inputs` acting as a mask (1 - the associated weight is
                retained, 0 - the associated weight is pruned).
        """
        mask = inputs.clone()
        _, idx = inputs.abs().flatten().sort(descending=True)
        j = int(threshold * inputs.numel())
        flat_out = mask.flatten()
        flat_out[idx[j:]] = 0
        flat_out[idx[:j]] = 1
        return mask


class ThresholdBinarizer(autograd.Function):
    """
    Thresholdd binarizer.
    Computes a binary mask M from a real value matrix S such that `M_{i,j} = 1` if and only if `S_{i,j} > 	au`
    where `	au` is a real value threshold.

    Implementation is inspired from:
        https://github.com/arunmallya/piggyback
        Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights
        Arun Mallya, Dillon Davis, Svetlana Lazebnik
    """

    @staticmethod
    def forward(ctx, inputs: 'torch.tensor', threshold: 'float', sigmoid: 'bool'):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input matrix from which the binarizer computes the binary mask.
            threshold (`float`)
                The threshold value (in R).
            sigmoid (`bool`)
                If set to ``True``, we apply the sigmoid function to the `inputs` matrix before comparing to `threshold`.
                In this case, `threshold` should be a value between 0 and 1.
        Returns:
            mask (`torch.FloatTensor`)
                Binary matrix of the same size as `inputs` acting as a mask (1 - the associated weight is
                retained, 0 - the associated weight is pruned).
        """
        nb_elems = inputs.numel()
        nb_min = int(0.005 * nb_elems) + 1
        if sigmoid:
            mask = (torch.sigmoid(inputs) > threshold).type(inputs.type())
        else:
            mask = (inputs > threshold).type(inputs.type())
        if mask.sum() < nb_min:
            k_threshold = inputs.flatten().kthvalue(max(nb_elems - nb_min, 1)).values
            mask = (inputs > k_threshold).type(inputs.type())
        return mask

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None, None


class TopKBinarizer(autograd.Function):
    """
    Top-k Binarizer.
    Computes a binary mask M from a real value matrix S such that `M_{i,j} = 1` if and only if `S_{i,j}`
    is among the k% highest values of S.

    Implementation is inspired from:
        https://github.com/allenai/hidden-networks
        What's hidden in a randomly weighted neural network?
        Vivek Ramanujan*, Mitchell Wortsman*, Aniruddha Kembhavi, Ali Farhadi, Mohammad Rastegari
    """

    @staticmethod
    def forward(ctx, inputs: 'torch.tensor', threshold: 'float'):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input matrix from which the binarizer computes the binary mask.
            threshold (`float`)
                The percentage of weights to keep (the rest is pruned).
                `threshold` is a float between 0 and 1.
        Returns:
            mask (`torch.FloatTensor`)
                Binary matrix of the same size as `inputs` acting as a mask (1 - the associated weight is
                retained, 0 - the associated weight is pruned).
        """
        mask = inputs.clone()
        _, idx = inputs.flatten().sort(descending=True)
        j = int(threshold * inputs.numel())
        flat_out = mask.flatten()
        flat_out[idx[j:]] = 0
        flat_out[idx[:j]] = 1
        return mask

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None


class MaskedLinear(nn.Linear):
    """
    Fully Connected layer with on the fly adaptive mask.
    If needed, a score matrix is created to store the importance of each associated weight.
    """

    def __init__(self, in_features: 'int', out_features: 'int', bias: 'bool'=True, mask_init: 'str'='constant', mask_scale: 'float'=0.0, pruning_method: 'str'='topK'):
        """
        Args:
            in_features (`int`)
                Size of each input sample
            out_features (`int`)
                Size of each output sample
            bias (`bool`)
                If set to ``False``, the layer will not learn an additive bias.
                Default: ``True``
            mask_init (`str`)
                The initialization method for the score matrix if a score matrix is needed.
                Choices: ["constant", "uniform", "kaiming"]
                Default: ``constant``
            mask_scale (`float`)
                The initialization parameter for the chosen initialization method `mask_init`.
                Default: ``0.``
            pruning_method (`str`)
                Method to compute the mask.
                Choices: ["topK", "threshold", "sigmoied_threshold", "magnitude", "l0"]
                Default: ``topK``
        """
        super(MaskedLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        assert pruning_method in ['topK', 'threshold', 'sigmoied_threshold', 'magnitude', 'l0']
        self.pruning_method = pruning_method
        if self.pruning_method in ['topK', 'threshold', 'sigmoied_threshold', 'l0']:
            self.mask_scale = mask_scale
            self.mask_init = mask_init
            self.mask_scores = nn.Parameter(torch.Tensor(self.weight.size()))
            self.init_mask()

    def init_mask(self):
        if self.mask_init == 'constant':
            init.constant_(self.mask_scores, val=self.mask_scale)
        elif self.mask_init == 'uniform':
            init.uniform_(self.mask_scores, a=-self.mask_scale, b=self.mask_scale)
        elif self.mask_init == 'kaiming':
            init.kaiming_uniform_(self.mask_scores, a=math.sqrt(5))

    def forward(self, input: 'torch.tensor', threshold: 'float'):
        if self.pruning_method == 'topK':
            mask = TopKBinarizer.apply(self.mask_scores, threshold)
        elif self.pruning_method in ['threshold', 'sigmoied_threshold']:
            sig = 'sigmoied' in self.pruning_method
            mask = ThresholdBinarizer.apply(self.mask_scores, threshold, sig)
        elif self.pruning_method == 'magnitude':
            mask = MagnitudeBinarizer.apply(self.weight, threshold)
        elif self.pruning_method == 'l0':
            l, r, b = -0.1, 1.1, 2 / 3
            if self.training:
                u = torch.zeros_like(self.mask_scores).uniform_().clamp(0.0001, 0.9999)
                s = torch.sigmoid((u.log() - (1 - u).log() + self.mask_scores) / b)
            else:
                s = torch.sigmoid(self.mask_scores)
            s_bar = s * (r - l) + l
            mask = s_bar.clamp(min=0.0, max=1.0)
        weight_thresholded = mask * self.weight
        return F.linear(input, weight_thresholded, self.bias)


def _denominator(t_slice_shape, precision, unroll=1):

    def fwd(qs, ks):

        def body(p, qk):
            q, k = qk
            p += k
            x = jnp.einsum('...m,...m->...', q, p, precision=precision)
            return p, x
        p = jnp.zeros(t_slice_shape)
        p, R = lax.scan(body, p, (qs, ks), unroll=unroll)
        return R, (qs, ks, p)

    def bwd(qkp, R_ct):

        def body(carry, qkx):
            p, p_ct = carry
            q, k, x_ct = qkx
            q_ct = jnp.einsum('...,...m->...m', x_ct, p, precision=precision)
            p_ct += jnp.einsum('...,...m->...m', x_ct, q, precision=precision)
            k_ct = p_ct
            p -= k
            return (p, p_ct), (q_ct, k_ct)
        qs, ks, p = qkp
        _, (qs_ct, ks_ct) = lax.scan(body, (p, jnp.zeros_like(p)), (qs, ks, R_ct), reverse=True, unroll=unroll)
        return qs_ct, ks_ct

    @jax.custom_vjp
    def _denominator_impl(qs, ks):
        R, _ = fwd(qs, ks)
        return R
    _denominator_impl.defvjp(fwd, bwd)
    return _denominator_impl


def _invert_perm(perm):
    perm_inv = [0] * len(perm)
    for i, j in enumerate(perm):
        perm_inv[j] = i
    return tuple(perm_inv)


def _numerator(z_slice_shape, precision, unroll=1):

    def fwd(qs, ks, vs):

        def body(p, qkv):
            q, k, v = qkv
            p += jnp.einsum('...m,...d->...md', k, v, precision=precision)
            X_slice = jnp.einsum('...m,...md->...d', q, p, precision=precision)
            return p, X_slice
        init_value = jnp.zeros(z_slice_shape)
        p, W = lax.scan(body, init_value, (qs, ks, vs), unroll=unroll)
        return W, (p, qs, ks, vs)

    def bwd(pqkv, W_ct):

        def body(carry, qkv_xct):
            p, p_ct = carry
            q, k, v, x_ct = qkv_xct
            q_ct = jnp.einsum('...d,...md->...m', x_ct, p, precision=precision)
            p_ct += jnp.einsum('...d,...m->...md', x_ct, q, precision=precision)
            k_ct = jnp.einsum('...md,...d->...m', p_ct, v, precision=precision)
            v_ct = jnp.einsum('...md,...m->...d', p_ct, k, precision=precision)
            p -= jnp.einsum('...m,...d->...md', k, v, precision=precision)
            return (p, p_ct), (q_ct, k_ct, v_ct)
        p, qs, ks, vs = pqkv
        _, (qs_ct, ks_ct, vs_ct) = lax.scan(body, (p, jnp.zeros_like(p)), (qs, ks, vs, W_ct), reverse=True, unroll=unroll)
        return qs_ct, ks_ct, vs_ct

    @jax.custom_vjp
    def _numerator_impl(qs, ks, vs):
        W, _ = fwd(qs, ks, vs)
        return W
    _numerator_impl.defvjp(fwd, bwd)
    return _numerator_impl


class RandomMatrix(object):
    """
    Abstract class providing a method for constructing 2D random arrays. Class is responsible for constructing 2D
    random arrays.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_2d_array(self):
        raise NotImplementedError('Abstract method')


class GaussianOrthogonalRandomMatrix(RandomMatrix):
    """
    Class providing a method to create Gaussian orthogonal matrix. Class is responsible for constructing 2D Gaussian
    orthogonal arrays.
    """

    def __init__(self, nb_rows, nb_columns, key, scaling=0):
        self.nb_rows = nb_rows
        self.nb_columns = nb_columns
        self.key = key
        self.scaling = scaling

    def get_2d_array(self):
        nb_full_blocks = int(self.nb_rows / self.nb_columns)
        block_list = []
        rng = self.key
        for _ in range(nb_full_blocks):
            rng, rng_input = jax.random.split(rng)
            unstructured_block = random.normal(rng_input, (self.nb_columns, self.nb_columns))
            q, _ = jnp.linalg.qr(unstructured_block)
            q = jnp.transpose(q)
            block_list.append(q)
        remaining_rows = self.nb_rows - nb_full_blocks * self.nb_columns
        if remaining_rows > 0:
            rng, rng_input = jax.random.split(rng)
            unstructured_block = random.normal(rng_input, (self.nb_columns, self.nb_columns))
            q, _ = jnp.linalg.qr(unstructured_block)
            q = jnp.transpose(q)
            block_list.append(q[0:remaining_rows])
        final_matrix = jnp.vstack(block_list)
        if self.scaling == 0:
            multiplier = jnp.linalg.norm(random.normal(self.key, (self.nb_rows, self.nb_columns)), axis=1)
        elif self.scaling == 1:
            multiplier = jnp.sqrt(float(self.nb_columns)) * jnp.ones(self.nb_rows)
        else:
            raise ValueError('Scaling must be one of {0, 1}. Was %s' % self._scaling)
        return jnp.matmul(jnp.diag(multiplier), final_matrix)


class GaussianUnstructuredRandomMatrix(RandomMatrix):

    def __init__(self, nb_rows, nb_columns, key):
        self.nb_rows = nb_rows
        self.nb_columns = nb_columns
        self.key = key

    def get_2d_array(self):
        return random.normal(self.key, (self.nb_rows, self.nb_columns))


def nonnegative_softmax_kernel_feature_creator(data, projection_matrix, attention_dims_t, batch_dims_t, precision, is_query, normalize_data=True, eps=0.0001):
    """
    Constructs nonnegative kernel features for fast softmax attention

    Args:
      data: input for which features are computes
      projection_matrix: random matrix used to compute features
      attention_dims_t: tuple of attention dimensions
      batch_dims_t: tuple of batch dimensions
      precision: precision parameter
      is_query: predicate indicating whether input data corresponds to queries or
        keys
      normalize_data: predicate indicating whether data should be normalized,
      eps: numerical stabilizer

    Returns:
      Random features for fast softmax attention.
    """
    del attention_dims_t
    if normalize_data:
        data_normalizer = 1.0 / jnp.sqrt(jnp.sqrt(data.shape[-1]))
    else:
        data_normalizer = 1.0
    ratio = 1.0 / jnp.sqrt(projection_matrix.shape[0])
    data_mod_shape = data.shape[0:len(batch_dims_t)] + projection_matrix.shape
    data_thick_random_matrix = jnp.zeros(data_mod_shape) + projection_matrix
    data_dash = lax.dot_general(data_normalizer * data, data_thick_random_matrix, (((data.ndim - 1,), (data_thick_random_matrix.ndim - 1,)), (batch_dims_t, batch_dims_t)), precision=precision)
    diag_data = jnp.square(data)
    diag_data = jnp.sum(diag_data, axis=data.ndim - 1)
    diag_data = diag_data / 2.0 * data_normalizer * data_normalizer
    diag_data = jnp.expand_dims(diag_data, axis=data.ndim - 1)
    if is_query:
        last_dims_t = len(data_dash.shape) - 1,
        data_dash = ratio * (jnp.exp(data_dash - diag_data - jnp.max(data_dash, axis=last_dims_t, keepdims=True)) + eps)
    else:
        data_dash = ratio * (jnp.exp(data_dash - diag_data - jnp.max(data_dash)) + eps)
    return data_dash


def sincos_softmax_kernel_feature_creator(data, projection_matrix, attention_dims_t, batch_dims_t, precision, normalize_data=True):
    """
    Constructs kernel sin-cos features for fast softmax attention

    Args:
      data: input for which features are computes
      projection_matrix: random matrix used to compute features
      attention_dims_t: tuple of attention dimensions
      batch_dims_t: tuple of batch dimensions
      precision: precision parameter
      normalize_data: predicate indicating whether data should be normalized

    Returns:
      Random features for fast softmax attention.
    """
    if normalize_data:
        data_normalizer = 1.0 / jnp.sqrt(jnp.sqrt(data.shape[-1]))
    else:
        data_normalizer = 1.0
    ratio = 1.0 / jnp.sqrt(projection_matrix.shape[0])
    data_mod_shape = data.shape[0:len(batch_dims_t)] + projection_matrix.shape
    data_thick_random_matrix = jnp.zeros(data_mod_shape) + projection_matrix
    data_dash = lax.dot_general(data_normalizer * data, data_thick_random_matrix, (((data.ndim - 1,), (data_thick_random_matrix.ndim - 1,)), (batch_dims_t, batch_dims_t)), precision=precision)
    data_dash_cos = ratio * jnp.cos(data_dash)
    data_dash_sin = ratio * jnp.sin(data_dash)
    data_dash = jnp.concatenate((data_dash_cos, data_dash_sin), axis=-1)
    diag_data = jnp.square(data)
    diag_data = jnp.sum(diag_data, axis=data.ndim - 1)
    diag_data = diag_data / 2.0 * data_normalizer * data_normalizer
    diag_data = jnp.expand_dims(diag_data, axis=data.ndim - 1)
    data_renormalizer = jnp.max(diag_data, attention_dims_t, keepdims=True)
    diag_data -= data_renormalizer
    diag_data = jnp.exp(diag_data)
    data_prime = data_dash * diag_data
    return data_prime


def make_fast_softmax_attention(qkv_dim, renormalize_attention=True, numerical_stabilizer=1e-06, nb_features=256, ortho_features=True, ortho_scaling=0.0, redraw_features=True, unidirectional=False, nonnegative_features=True, lax_scan_unroll=1):
    """Construct a fast softmax attention method."""
    logging.info('Fast softmax attention: %s features and orthogonal=%s, renormalize=%s', nb_features, ortho_features, renormalize_attention)
    if ortho_features:
        matrix_creator = functools.partial(GaussianOrthogonalRandomMatrix, nb_features, qkv_dim, scaling=ortho_scaling)
    else:
        matrix_creator = functools.partial(GaussianUnstructuredRandomMatrix, nb_features, qkv_dim)
    if nonnegative_features:

        def kernel_feature_creator(data, projection_matrix, attention_dims_t, batch_dims_t, precision, is_query, normalize_data=True):
            return nonnegative_softmax_kernel_feature_creator(data, projection_matrix, attention_dims_t, batch_dims_t, precision, is_query, normalize_data, numerical_stabilizer)
    else:

        def kernel_feature_creator(data, projection_matrix, attention_dims_t, batch_dims_t, precision, is_query, normalize_data=True):
            del is_query
            return sincos_softmax_kernel_feature_creator(data, projection_matrix, attention_dims_t, batch_dims_t, precision, normalize_data)
    attention_fn = FastAttentionviaLowRankDecomposition(matrix_creator, kernel_feature_creator, renormalize_attention=renormalize_attention, numerical_stabilizer=numerical_stabilizer, redraw_features=redraw_features, unidirectional=unidirectional, lax_scan_unroll=lax_scan_unroll).dot_product_attention
    return attention_fn


class ClassificationHead(torch.nn.Module):
    """Classification Head for  transformer encoders"""

    def __init__(self, class_size, embed_size):
        super().__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        self.mlp = torch.nn.Linear(embed_size, class_size)

    def forward(self, hidden_state):
        logits = self.mlp(hidden_state)
        return logits


EPSILON = 1e-10


VOCAB_FILES_NAMES = {'vocab_file': 'vocab.json', 'tokenizer_config_file': 'tokenizer_config.json'}


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


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings. The reversible bpe codes work on unicode
    strings. This means you need a large # of unicode characters in your vocab if you want to avoid UNKs. When you're
    at something like a 10B token dataset you end up needing around 5K for decent coverage. This is a signficant
    percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup tables between utf-8 bytes and unicode
    strings. And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    Return set of symbol pairs in a word. Word is represented as tuple of symbols (symbols being variable-length
    strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:

    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip([tuple(k) for k in bpe_merges], range(len(bpe_merges))))
        self.cache = {}
        self.random = random.Random(0)
        self.pat = re.compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+")

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            return token
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except Exception:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def split_to_words(self, text):
        return list(re.findall(self.pat, text))

    def encode(self, text):
        bpe_tokens = []
        for token in self.split_to_words(text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text


def get_encoder(encoder, vocab):
    return Encoder(encoder=encoder, bpe_merges=vocab)


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, 'r', encoding='utf-8') as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip('\n')
        vocab[token] = index
    return vocab


class GPT2Tokenizer(object):
    """
    A wrapper of GPT2 tokenizer with similar interface as BERT tokenizer

    Args:
        vocab_file (:obj:`str`, optional):
            The local path of vocabulary package or the release name of vocabulary in `DeBERTa GitHub releases
            <https://github.com/microsoft/DeBERTa/releases>`_, e.g. "bpe_encoder", default: `None`.

            If it's `None`, then it will download the vocabulary in the latest release from GitHub. The vocabulary file
            is a state dictionary with three items, "dict_map", "vocab", "encoder" which correspond to three files used
            in `RoBERTa`, i.e. `dict.txt`, `vocab.txt` and `encoder.json`. The difference between our wrapped GPT2
            tokenizer and RoBERTa wrapped tokenizer are,

            - Special tokens, unlike `RoBERTa` which use `<s>`, `</s>` as the `start` token and `end` token of a
              sentence. We use `[CLS]` and `[SEP]` as the `start` and `end` token of input sentence which is the same
              as `BERT`.

            - We remapped the token ids in our dictionary with regarding to the new special tokens, `[PAD]` => 0,
              `[CLS]` => 1, `[SEP]` => 2, `[UNK]` => 3, `[MASK]` => 50264

        special_tokens (:obj:`list`, optional):
            List of special tokens to be added to the end of the vocabulary.
    """

    def __init__(self, vocab_file=None, special_tokens=None):
        self.pad_token = '[PAD]'
        self.sep_token = '[SEP]'
        self.unk_token = '[UNK]'
        self.cls_token = '[CLS]'
        self.symbols = []
        self.count = []
        self.indices = {}
        self.pad_token_id = self.add_symbol(self.pad_token)
        self.cls_token_id = self.add_symbol(self.cls_token)
        self.sep_token_id = self.add_symbol(self.sep_token)
        self.unk_token_id = self.add_symbol(self.unk_token)
        self.gpt2_encoder = load_vocab(vocab_file)
        self.bpe = get_encoder(self.gpt2_encoder['encoder'], self.gpt2_encoder['vocab'])
        for w, n in self.gpt2_encoder['dict_map']:
            self.add_symbol(w, n)
        self.mask_token = '[MASK]'
        self.mask_id = self.add_symbol(self.mask_token)
        self.special_tokens = ['[MASK]', '[SEP]', '[PAD]', '[UNK]', '[CLS]']
        if special_tokens is not None:
            for t in special_tokens:
                self.add_special_token(t)
        self.vocab = self.indices
        self.ids_to_tokens = self.symbols

    def tokenize(self, text):
        """
        Convert an input text to tokens.

        Args:
          text (:obj:`str`): input text to be tokenized.

        Returns:
          A list of byte tokens where each token represent the byte id in GPT2 byte dictionary

        Example::
          >>> tokenizer = GPT2Tokenizer()
          >>> text = "Hello world!"
          >>> tokens = tokenizer.tokenize(text)
          >>> print(tokens)
          ['15496', '995', '0']
        """
        bpe = self._encode(text)
        return [t for t in bpe.split(' ') if t]

    def convert_tokens_to_ids(self, tokens):
        """
        Convert list of tokens to ids

        Args:
          tokens (:obj:`list<str>`): list of tokens

        Returns:
          List of ids
        """
        return [self.vocab[t] for t in tokens]

    def convert_ids_to_tokens(self, ids):
        """
        Convert list of ids to tokens

        Args:
          ids (:obj:`list<int>`): list of ids

        Returns:
          List of tokens
        """
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    def split_to_words(self, text):
        return self.bpe.split_to_words(text)

    def decode(self, tokens):
        """
        Decode list of tokens to text strings

        Args:
          tokens (:obj:`list<str>`): list of tokens.

        Returns:
          Text string corresponds to the input tokens.

        Example::
          >>> tokenizer = GPT2Tokenizer()
          >>> text = "Hello world!"
          >>> tokens = tokenizer.tokenize(text)
          >>> print(tokens)
          ['15496', '995', '0']
          >>> tokenizer.decode(tokens)
          'Hello world!'
        """
        return self.bpe.decode([int(t) for t in tokens if t not in self.special_tokens])

    def add_special_token(self, token):
        """
        Adds a special token to the dictionary

        Args:
          token (:obj:`str`): Tthe new token/word to be added to the vocabulary.

        Returns:
          The id of new token in the vocabulary.

        """
        self.special_tokens.append(token)
        return self.add_symbol(token)

    def part_of_whole_word(self, token, is_bos=False):
        if is_bos:
            return True
        s = self._decode(token)
        if len(s) == 1 and (_is_whitespace(list(s)[0]) or _is_control(list(s)[0]) or _is_punctuation(list(s)[0])):
            return False
        return not s.startswith(' ')

    def sym(self, id):
        return self.ids_to_tokens[id]

    def id(self, sym):
        return self.vocab[sym]

    def _encode(self, x: 'str') ->str:
        return ' '.join(map(str, self.bpe.encode(x)))

    def _decode(self, x: 'str') ->str:
        return self.bpe.decode(map(int, x.split()))

    def add_symbol(self, word, n=1):
        """
        Adds a word to the dictionary

        Args:
          word (:obj:`str`): Tthe new token/word to be added to the vocabulary.
          n (int, optional): The frequency of the word.

        Returns:
          The id of the new word.

        """
        if word in self.indices:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def save_pretrained(self, path: 'str', filename_prefix: 'str'=None):
        import torch
        filename = VOCAB_FILES_NAMES[list(VOCAB_FILES_NAMES.keys())[0]]
        if filename_prefix is not None:
            filename = filename_prefix + '-' + filename
        full_path = os.path.join(path, filename)
        torch.save(self.gpt2_encoder, full_path)
        return full_path,


class Discriminator(torch.nn.Module):
    """Transformer encoder followed by a Classification Head"""

    def __init__(self, class_size, pretrained_model='gpt2-medium', cached_mode=False, device='cpu'):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
        self.encoder = GPT2LMHeadModel.from_pretrained(pretrained_model)
        self.embed_size = self.encoder.transformer.config.hidden_size
        self.classifier_head = ClassificationHead(class_size=class_size, embed_size=self.embed_size)
        self.cached_mode = cached_mode
        self.device = device

    def get_classifier(self):
        return self.classifier_head

    def train_custom(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier_head.train()

    def avg_representation(self, x):
        mask = x.ne(0).unsqueeze(2).repeat(1, 1, self.embed_size).float().detach()
        hidden = self.encoder.transformer(x)['last_hidden_state']
        masked_hidden = hidden * mask
        avg_hidden = torch.sum(masked_hidden, dim=1) / (torch.sum(mask, dim=1).detach() + EPSILON)
        return avg_hidden

    def forward(self, x):
        if self.cached_mode:
            avg_hidden = x
        else:
            avg_hidden = self.avg_representation(x)
        logits = self.classifier_head(avg_hidden)
        probs = F.log_softmax(logits, dim=-1)
        return probs


class PoolerStartLogits(nn.Module):
    """
    Compute SQuAD start logits from sequence hidden states.

    Args:
        config (:class:`~transformers.PretrainedConfig`):
            The config used by the model, will be used to grab the :obj:`hidden_size` of the model.
    """

    def __init__(self, config: 'PretrainedConfig'):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states: 'torch.FloatTensor', p_mask: 'Optional[torch.FloatTensor]'=None) ->torch.FloatTensor:
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            p_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len)`, `optional`):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.

        Returns:
            :obj:`torch.FloatTensor`: The start logits for SQuAD.
        """
        x = self.dense(hidden_states).squeeze(-1)
        if p_mask is not None:
            if get_parameter_dtype(self) == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e+30 * p_mask
        return x


class PoolerEndLogits(nn.Module):
    """
    Compute SQuAD end logits from sequence hidden states.

    Args:
        config (:class:`~transformers.PretrainedConfig`):
            The config used by the model, will be used to grab the :obj:`hidden_size` of the model and the
            :obj:`layer_norm_eps` to use.
    """

    def __init__(self, config: 'PretrainedConfig'):
        super().__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense_1 = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states: 'torch.FloatTensor', start_states: 'Optional[torch.FloatTensor]'=None, start_positions: 'Optional[torch.LongTensor]'=None, p_mask: 'Optional[torch.FloatTensor]'=None) ->torch.FloatTensor:
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            start_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`, `optional`):
                The hidden states of the first tokens for the labeled span.
            start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                The position of the first token for the labeled span.
            p_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len)`, `optional`):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.

        .. note::

            One of ``start_states`` or ``start_positions`` should be not obj:`None`. If both are set,
            ``start_positions`` overrides ``start_states``.

        Returns:
            :obj:`torch.FloatTensor`: The end logits for SQuAD.
        """
        assert start_states is not None or start_positions is not None, 'One of start_states, start_positions should be not None'
        if start_positions is not None:
            slen, hsz = hidden_states.shape[-2:]
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)
            start_states = hidden_states.gather(-2, start_positions)
            start_states = start_states.expand(-1, slen, -1)
        x = self.dense_0(torch.cat([hidden_states, start_states], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)
        if p_mask is not None:
            if get_parameter_dtype(self) == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e+30 * p_mask
        return x


class PoolerAnswerClass(nn.Module):
    """
    Compute SQuAD 2.0 answer class from classification and start tokens hidden states.

    Args:
        config (:class:`~transformers.PretrainedConfig`):
            The config used by the model, will be used to grab the :obj:`hidden_size` of the model.
    """

    def __init__(self, config):
        super().__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.dense_1 = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: 'torch.FloatTensor', start_states: 'Optional[torch.FloatTensor]'=None, start_positions: 'Optional[torch.LongTensor]'=None, cls_index: 'Optional[torch.LongTensor]'=None) ->torch.FloatTensor:
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            start_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`, `optional`):
                The hidden states of the first tokens for the labeled span.
            start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                The position of the first token for the labeled span.
            cls_index (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Position of the CLS token for each sentence in the batch. If :obj:`None`, takes the last token.

        .. note::

            One of ``start_states`` or ``start_positions`` should be not obj:`None`. If both are set,
            ``start_positions`` overrides ``start_states``.

        Returns:
            :obj:`torch.FloatTensor`: The SQuAD 2.0 answer class.
        """
        hsz = hidden_states.shape[-1]
        assert start_states is not None or start_positions is not None, 'One of start_states, start_positions should be not None'
        if start_positions is not None:
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)
            start_states = hidden_states.gather(-2, start_positions).squeeze(-2)
        if cls_index is not None:
            cls_index = cls_index[:, None, None].expand(-1, -1, hsz)
            cls_token_state = hidden_states.gather(-2, cls_index).squeeze(-2)
        else:
            cls_token_state = hidden_states[:, -1, :]
        x = self.dense_0(torch.cat([start_states, cls_token_state], dim=-1))
        x = self.activation(x)
        x = self.dense_1(x).squeeze(-1)
        return x


def replace_return_docstrings(output_type=None, config_class=None):

    def docstring_decorator(fn):
        docstrings = fn.__doc__
        lines = docstrings.split('\n')
        i = 0
        while i < len(lines) and re.search('^\\s*Returns?:\\s*$', lines[i]) is None:
            i += 1
        if i < len(lines):
            lines[i] = _prepare_output_docstrings(output_type, config_class)
            docstrings = '\n'.join(lines)
        else:
            raise ValueError(f"The function {fn} should have an empty 'Return:' or 'Returns:' in its docstring as placeholder, current docstring is:\n{docstrings}")
        fn.__doc__ = docstrings
        return fn
    return docstring_decorator


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError('function {} not found in ACT2FN mapping {}'.format(activation_string, list(ACT2FN.keys())))


class SequenceSummary(nn.Module):
    """
    Compute a single vector summary of a sequence hidden states.

    Args:
        config (:class:`~transformers.PretrainedConfig`):
            The config used by the model. Relevant arguments in the config class of the model are (refer to the actual
            config class of your model for the default values it uses):

            - **summary_type** (:obj:`str`) -- The method to use to make this summary. Accepted values are:

                - :obj:`"last"` -- Take the last token hidden state (like XLNet)
                - :obj:`"first"` -- Take the first token hidden state (like Bert)
                - :obj:`"mean"` -- Take the mean of all tokens hidden states
                - :obj:`"cls_index"` -- Supply a Tensor of classification token position (GPT/GPT-2)
                - :obj:`"attn"` -- Not implemented now, use multi-head attention

            - **summary_use_proj** (:obj:`bool`) -- Add a projection after the vector extraction.
            - **summary_proj_to_labels** (:obj:`bool`) -- If :obj:`True`, the projection outputs to
              :obj:`config.num_labels` classes (otherwise to :obj:`config.hidden_size`).
            - **summary_activation** (:obj:`Optional[str]`) -- Set to :obj:`"tanh"` to add a tanh activation to the
              output, another string or :obj:`None` will add no activation.
            - **summary_first_dropout** (:obj:`float`) -- Optional dropout probability before the projection and
              activation.
            - **summary_last_dropout** (:obj:`float`)-- Optional dropout probability after the projection and
              activation.
    """

    def __init__(self, config: 'PretrainedConfig'):
        super().__init__()
        self.summary_type = getattr(config, 'summary_type', 'last')
        if self.summary_type == 'attn':
            raise NotImplementedError
        self.summary = Identity()
        if hasattr(config, 'summary_use_proj') and config.summary_use_proj:
            if hasattr(config, 'summary_proj_to_labels') and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)
        activation_string = getattr(config, 'summary_activation', None)
        self.activation: 'Callable' = get_activation(activation_string) if activation_string else Identity()
        self.first_dropout = Identity()
        if hasattr(config, 'summary_first_dropout') and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)
        self.last_dropout = Identity()
        if hasattr(config, 'summary_last_dropout') and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

    def forward(self, hidden_states: 'torch.FloatTensor', cls_index: 'Optional[torch.LongTensor]'=None) ->torch.FloatTensor:
        """
        Compute a single vector summary of a sequence hidden states.

        Args:
            hidden_states (:obj:`torch.FloatTensor` of shape :obj:`[batch_size, seq_len, hidden_size]`):
                The hidden states of the last layer.
            cls_index (:obj:`torch.LongTensor` of shape :obj:`[batch_size]` or :obj:`[batch_size, ...]` where ... are optional leading dimensions of :obj:`hidden_states`, `optional`):
                Used if :obj:`summary_type == "cls_index"` and takes the last token of the sequence as classification
                token.

        Returns:
            :obj:`torch.FloatTensor`: The summary of the sequence hidden states.
        """
        if self.summary_type == 'last':
            output = hidden_states[:, -1]
        elif self.summary_type == 'first':
            output = hidden_states[:, 0]
        elif self.summary_type == 'mean':
            output = hidden_states.mean(dim=1)
        elif self.summary_type == 'cls_index':
            if cls_index is None:
                cls_index = torch.full_like(hidden_states[..., :1, :], hidden_states.shape[-2] - 1, dtype=torch.long)
            else:
                cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                cls_index = cls_index.expand((-1,) * (cls_index.dim() - 1) + (hidden_states.size(-1),))
            output = hidden_states.gather(-2, cls_index).squeeze(-2)
        elif self.summary_type == 'attn':
            raise NotImplementedError
        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)
        return output


class Adapter(nn.Module):

    def __init__(self, dim, r, act):
        super().__init__()
        self.adapter_A = nn.Linear(dim, r)
        self.act = get_activation(act)
        self.adapter_B = nn.Linear(r, dim)

    def forward(self, x, residual):
        result = self.adapter_A(x)
        result = self.act(result)
        result = self.adapter_B(result)
        return result + residual


class AlbertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:seq_length + past_key_values_length]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class AlbertAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, 'embedding_size'):
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pruned_heads = set()
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_attention_heads, self.attention_head_size, self.pruned_heads)
        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)
        self.dense = prune_linear_layer(self.dense, index, dim=1)
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
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
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attention_dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        w = self.dense.weight.t().view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
        b = self.dense.bias
        projected_context_layer = torch.einsum('bfnd,ndh->bfh', context_layer, w) + b
        projected_context_layer_dropout = self.output_dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
        return (layernormed_context_layer, attention_probs) if output_attentions else (layernormed_context_layer,)


class AlbertLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.full_layer_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = AlbertAttention(config)
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, output_hidden_states=False):
        attention_output = self.attention(hidden_states, attention_mask, head_mask, output_attentions)
        ffn_output = apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output[0])
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
        return (hidden_states,) + attention_output[1:]

    def ff_chunk(self, attention_output):
        ffn_output = self.ffn(attention_output)
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        return ffn_output


class AlbertLayerGroup(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.albert_layers = nn.ModuleList([AlbertLayer(config) for _ in range(config.inner_group_num)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, output_hidden_states=False):
        layer_hidden_states = ()
        layer_attentions = ()
        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(hidden_states, attention_mask, head_mask[layer_index], output_attentions)
            hidden_states = layer_output[0]
            if output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)
            if output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)
        outputs = hidden_states,
        if output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        if output_attentions:
            outputs = outputs + (layer_attentions,)
        return outputs


class AlbertTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.ModuleList([AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i in range(self.config.num_hidden_layers):
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))
            layer_group_output = self.albert_layer_groups[group_idx](hidden_states, attention_mask, head_mask[group_idx * layers_per_group:(group_idx + 1) * layers_per_group], output_attentions, output_hidden_states)
            hidden_states = layer_group_output[0]
            if output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions)


class AlbertConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.AlbertModel` or a
    :class:`~transformers.TFAlbertModel`. It is used to instantiate an ALBERT model according to the specified
    arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the ALBERT `xxlarge <https://huggingface.co/albert-xxlarge-v2>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30000):
            Vocabulary size of the ALBERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.AlbertModel` or
            :class:`~transformers.TFAlbertModel`.
        embedding_size (:obj:`int`, `optional`, defaults to 128):
            Dimensionality of vocabulary embeddings.
        hidden_size (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_hidden_groups (:obj:`int`, `optional`, defaults to 1):
            Number of groups for the hidden layers, parameters in the same group are shared.
        num_attention_heads (:obj:`int`, `optional`, defaults to 64):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 16384):
            The dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        inner_group_num (:obj:`int`, `optional`, defaults to 1):
            The number of inner repetition of attention and ffn.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.AlbertModel` or
            :class:`~transformers.TFAlbertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        classifier_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for attached classifiers.
        position_embedding_type (:obj:`str`, `optional`, defaults to :obj:`"absolute"`):
            Type of position embedding. Choose one of :obj:`"absolute"`, :obj:`"relative_key"`,
            :obj:`"relative_key_query"`. For positional embeddings use :obj:`"absolute"`. For more information on
            :obj:`"relative_key"`, please refer to `Self-Attention with Relative Position Representations (Shaw et al.)
            <https://arxiv.org/abs/1803.02155>`__. For more information on :obj:`"relative_key_query"`, please refer to
            `Method 4` in `Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)
            <https://arxiv.org/abs/2009.13658>`__.

    Examples::

        >>> from transformers import AlbertConfig, AlbertModel
        >>> # Initializing an ALBERT-xxlarge style configuration
        >>> albert_xxlarge_configuration = AlbertConfig()

        >>> # Initializing an ALBERT-base style configuration
        >>> albert_base_configuration = AlbertConfig(
        ...      hidden_size=768,
        ...      num_attention_heads=12,
        ...      intermediate_size=3072,
        ...  )

        >>> # Initializing a model from the ALBERT-base style configuration
        >>> model = AlbertModel(albert_xxlarge_configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = 'albert'

    def __init__(self, vocab_size=30000, embedding_size=128, hidden_size=4096, num_hidden_layers=12, num_hidden_groups=1, num_attention_heads=64, intermediate_size=16384, inner_group_num=1, hidden_act='gelu_new', hidden_dropout_prob=0, attention_probs_dropout_prob=0, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, classifier_dropout_prob=0.1, position_embedding_type='absolute', pad_token_id=0, bos_token_id=2, eos_token_id=3, **kwargs):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_groups = num_hidden_groups
        self.num_attention_heads = num_attention_heads
        self.inner_group_num = inner_group_num
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.classifier_dropout_prob = classifier_dropout_prob
        self.position_embedding_type = position_embedding_type


ALBERT_INPUTS_DOCSTRING = """
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.AlbertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.__call__` and :meth:`transformers.PreTrainedTokenizer.encode` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


ALBERT_START_DOCSTRING = """

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Args:
        config (:class:`~transformers.AlbertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""


def load_tf_weights_in_albert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model."""
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error('Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.')
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info('Converting TensorFlow checkpoint from {}'.format(tf_path))
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info('Loading TF weight {} with shape {}'.format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    for name, array in zip(names, arrays):
        None
    for name, array in zip(names, arrays):
        original_name = name
        name = name.replace('module/', '')
        name = name.replace('ffn_1', 'ffn')
        name = name.replace('bert/', 'albert/')
        name = name.replace('attention_1', 'attention')
        name = name.replace('transform/', '')
        name = name.replace('LayerNorm_1', 'full_layer_layer_norm')
        name = name.replace('LayerNorm', 'attention/LayerNorm')
        name = name.replace('transformer/', '')
        name = name.replace('intermediate/dense/', '')
        name = name.replace('ffn/intermediate/output/dense/', 'ffn_output/')
        name = name.replace('/output/', '/')
        name = name.replace('/self/', '/')
        name = name.replace('pooler/dense', 'pooler')
        name = name.replace('cls/predictions', 'predictions')
        name = name.replace('predictions/attention', 'predictions')
        name = name.replace('embeddings/attention', 'embeddings')
        name = name.replace('inner_group_', 'albert_layers/')
        name = name.replace('group_', 'albert_layer_groups/')
        if len(name.split('/')) == 1 and ('output_bias' in name or 'output_weights' in name):
            name = 'classifier/' + name
        if 'seq_relationship' in name:
            name = name.replace('seq_relationship/output_', 'sop_classifier/classifier/')
            name = name.replace('weights', 'weight')
        name = name.split('/')
        if 'adam_m' in name or 'adam_v' in name or 'AdamWeightDecayOptimizer' in name or 'AdamWeightDecayOptimizer_1' in name or 'global_step' in name:
            logger.info('Skipping {}'.format('/'.join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch('[A-Za-z]+_\\d+', m_name):
                scope_names = re.split('_(\\d+)', m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == 'kernel' or scope_names[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif scope_names[0] == 'output_bias' or scope_names[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif scope_names[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif scope_names[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info('Skipping {}'.format('/'.join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape, f'Pointer shape {pointer.shape} and array shape {array.shape} mismatched'
        except AssertionError as e:
            e.args += pointer.shape, array.shape
            raise
        None
        pointer.data = torch.from_numpy(array)
    return model


class AlbertMLMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(config.embedding_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size)
        self.activation = ACT2FN[config.hidden_act]
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        prediction_scores = hidden_states
        return prediction_scores


class AlbertSOPHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, pooled_output):
        dropout_pooled_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_pooled_output)
        return logits


class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: 'int', embedding_dim: 'int'):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids_shape: 'torch.Size', past_key_values_length: 'int'=0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions + self.offset)


class BartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim: 'int', num_heads: 'int', dropout: 'float'=0.0, is_decoder: 'bool'=False, bias: 'bool'=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).'
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: 'torch.Tensor', seq_len: 'int', bsz: 'int'):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: 'torch.Tensor', key_value_states: 'Optional[torch.Tensor]'=None, past_key_value: 'Optional[Tuple[torch.Tensor]]'=None, attention_mask: 'Optional[torch.Tensor]'=None, layer_head_mask: 'Optional[torch.Tensor]'=None, output_attentions: 'bool'=False) ->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        if self.is_decoder:
            past_key_value = key_states, value_states
        proj_shape = bsz * self.num_heads, -1, self.head_dim
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len), f'Attention weights should be of size {bsz * self.num_heads, tgt_len, src_len}, but is {attn_weights.size()}'
        if attention_mask is not None:
            assert attention_mask.size() == (bsz, 1, tgt_len, src_len), f'Attention mask should be of size {bsz, 1, tgt_len, src_len}, but is {attention_mask.size()}'
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        if layer_head_mask is not None:
            assert layer_head_mask.size() == (self.num_heads,), f'Head mask for a single layer should be of size {self.num_heads,}, but is {layer_head_mask.size()}'
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim), f'`attn_output` should be of size {bsz, self.num_heads, tgt_len, self.head_dim}, but is {attn_output.size()}'
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped, past_key_value


class BartEncoderLayer(nn.Module):

    def __init__(self, config: 'BartConfig'):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BartAttention(embed_dim=self.embed_dim, num_heads=config.encoder_attention_heads, dropout=config.attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'torch.Tensor', layer_head_mask: 'torch.Tensor', output_attentions: 'bool'=False):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask, output_attentions=output_attentions)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        if hidden_states.dtype == torch.float16 and (torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        outputs = hidden_states,
        if output_attentions:
            outputs += attn_weights,
        return outputs


class BartDecoderLayer(nn.Module):

    def __init__(self, config: 'BartConfig'):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BartAttention(embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout, is_decoder=True)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(self.embed_dim, config.decoder_attention_heads, dropout=config.attention_dropout, is_decoder=True)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]'=None, encoder_hidden_states: 'Optional[torch.Tensor]'=None, encoder_attention_mask: 'Optional[torch.Tensor]'=None, layer_head_mask: 'Optional[torch.Tensor]'=None, encoder_layer_head_mask: 'Optional[torch.Tensor]'=None, past_key_value: 'Optional[Tuple[torch.Tensor]]'=None, output_attentions: 'Optional[bool]'=False, use_cache: 'Optional[bool]'=True):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            encoder_layer_head_mask (:obj:`torch.FloatTensor`): mask for encoder attention heads in a given layer of
                size `(config.encoder_attention_heads,)`.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        hidden_states, self_attn_weights, present_key_value = self.self_attn(hidden_states=hidden_states, past_key_value=self_attn_past_key_value, attention_mask=attention_mask, layer_head_mask=layer_head_mask, output_attentions=output_attentions)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(hidden_states=hidden_states, key_value_states=encoder_hidden_states, attention_mask=encoder_attention_mask, layer_head_mask=encoder_layer_head_mask, past_key_value=cross_attn_past_key_value, output_attentions=output_attentions)
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            present_key_value = present_key_value + cross_attn_present_key_value
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        outputs = hidden_states,
        if output_attentions:
            outputs += self_attn_weights, cross_attn_weights
        if use_cache:
            outputs += present_key_value,
        return outputs


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim: 'int', inner_dim: 'int', num_classes: 'int', pooler_dropout: 'float'):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: 'torch.Tensor'):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class BartConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.BartModel`. It is used to
    instantiate a BART model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BART `facebook/bart-large
    <https://huggingface.co/facebook/bart-large>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 50265):
            Vocabulary size of the BART model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.BartModel` or
            :class:`~transformers.TFBartModel`.
        d_model (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        encoder_layers (:obj:`int`, `optional`, defaults to 12):
            Number of encoder layers.
        decoder_layers (:obj:`int`, `optional`, defaults to 12):
            Number of decoder layers.
        encoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        classifier_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for classifier.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        encoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the encoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        decoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the decoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        scale_embedding (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Scale embeddings by diving by sqrt(d_model).
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        num_labels: (:obj:`int`, `optional`, defaults to 3):
            The number of labels to use in :class:`~transformers.BartForSequenceClassification`.
        forced_eos_token_id (:obj:`int`, `optional`, defaults to 2):
            The id of the token to force as the last generated token when :obj:`max_length` is reached. Usually set to
            :obj:`eos_token_id`.

    Example::

        >>> from transformers import BartModel, BartConfig

        >>> # Initializing a BART facebook/bart-large style configuration
        >>> configuration = BartConfig()

        >>> # Initializing a model from the facebook/bart-large style configuration
        >>> model = BartModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = 'bart'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(self, vocab_size=50265, max_position_embeddings=1024, encoder_layers=12, encoder_ffn_dim=4096, encoder_attention_heads=16, decoder_layers=12, decoder_ffn_dim=4096, decoder_attention_heads=16, encoder_layerdrop=0.0, decoder_layerdrop=0.0, activation_function='gelu', d_model=1024, dropout=0.1, attention_dropout=0.0, activation_dropout=0.0, init_std=0.02, classifier_dropout=0.0, scale_embedding=False, gradient_checkpointing=False, use_cache=True, num_labels=3, pad_token_id=1, bos_token_id=0, eos_token_id=2, is_encoder_decoder=True, decoder_start_token_id=2, forced_eos_token_id=2, **kwargs):
        super().__init__(num_labels=num_labels, pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, is_encoder_decoder=is_encoder_decoder, decoder_start_token_id=decoder_start_token_id, forced_eos_token_id=forced_eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_embedding = scale_embedding
        if self.forced_bos_token_id is None and kwargs.get('force_bos_token_to_be_generated', False):
            self.forced_bos_token_id = self.bos_token_id
            warnings.warn(f'Please make sure the config includes `forced_bos_token_id={self.bos_token_id}` in future versions.The config can simply be saved and uploaded again to be fixed.')

    @property
    def num_attention_heads(self) ->int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) ->int:
        return self.d_model


def _expand_mask(mask: 'torch.Tensor', dtype: 'torch.dtype', tgt_len: 'Optional[int]'=None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


def _make_causal_mask(input_ids_shape: 'torch.Size', dtype: 'torch.dtype', past_key_values_length: 'int'=0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float('-inf'))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


BART_INPUTS_DOCSTRING = """
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using :class:`~transformers.BartTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BartTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__

            Bart uses the :obj:`eos_token_id` as the starting token for :obj:`decoder_input_ids` generation. If
            :obj:`past_key_values` is used, optionally only the last :obj:`decoder_input_ids` have to be input (see
            :obj:`past_key_values`).

            For translation and summarization training, :obj:`decoder_input_ids` should be provided. If no
            :obj:`decoder_input_ids` is provided, the model will create this tensor by shifting the :obj:`input_ids` to
            the right for denoising pre-training following the paper.
        decoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.

            If you want to change padding behavior, you should read :func:`modeling_bart._prepare_decoder_inputs` and
            modify to your needs. See diagram 1 in `the paper <https://arxiv.org/abs/1910.13461>`__ for more
            information on the default strategy.
        head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the heas is **masked**.

        decoder_head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`):
            Tuple consists of (:obj:`last_hidden_state`, `optional`: :obj:`hidden_states`, `optional`:
            :obj:`attentions`) :obj:`last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`,
            `optional`) is a sequence of hidden-states at the output of the last layer of the encoder. Used in the
            cross-attention of the decoder.
        past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size, sequence_length)`.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded
            representation. If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_inputs_embeds`
            have to be input (see :obj:`past_key_values`). This is useful if you want more control over how to convert
            :obj:`decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If :obj:`decoder_input_ids` and :obj:`decoder_inputs_embeds` are both unset, :obj:`decoder_inputs_embeds`
            takes the value of :obj:`inputs_embeds`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


BART_START_DOCSTRING = """
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.BartConfig`):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""


def shift_tokens_right(input_ids: 'torch.Tensor', pad_token_id: 'int', decoder_start_token_id: 'int'):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    assert pad_token_id is not None, 'self.model.config.pad_token_id has to be defined.'
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


BART_GENERATION_EXAMPLE = """
    Summarization example::

        >>> from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

        >>> model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        >>> tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

        >>> ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')

        >>> # Generate Summary
        >>> summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
        >>> print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

    Mask filling example::

        >>> from transformers import BartTokenizer, BartForConditionalGeneration
        >>> tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        >>> TXT = "My friends are <mask> but they eat too many carbs."

        >>> model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        >>> input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
        >>> logits = model(input_ids).logits

        >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
        >>> probs = logits[0, masked_index].softmax(dim=0)
        >>> values, predictions = probs.topk(5)

        >>> tokenizer.decode(predictions).split()
"""


def add_end_docstrings(*docstr):

    def docstring_decorator(fn):
        fn.__doc__ = fn.__doc__ + ''.join(docstr)
        return fn
    return docstring_decorator


class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertGenerationEmbeddings(nn.Module):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:seq_length + past_key_values_length]
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertGenerationConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a
    :class:`~transformers.BertGenerationPreTrainedModel`. It is used to instantiate a BertGeneration model according to
    the specified arguments, defining the model architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 50358):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.BertGeneration`.
        hidden_size (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often called feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If :obj:`True`, use gradient checkpointing to save memory at the expense of slower backward pass.
        position_embedding_type (:obj:`str`, `optional`, defaults to :obj:`"absolute"`):
            Type of position embedding. Choose one of :obj:`"absolute"`, :obj:`"relative_key"`,
            :obj:`"relative_key_query"`. For positional embeddings use :obj:`"absolute"`. For more information on
            :obj:`"relative_key"`, please refer to `Self-Attention with Relative Position Representations (Shaw et al.)
            <https://arxiv.org/abs/1803.02155>`__. For more information on :obj:`"relative_key_query"`, please refer to
            `Method 4` in `Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)
            <https://arxiv.org/abs/2009.13658>`__.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if ``config.is_decoder=True``.

    Examples::

        >>> from transformers import BertGenerationConfig, BertGenerationEncoder

        >>> # Initializing a BertGeneration config
        >>> configuration = BertGenerationConfig()

        >>> # Initializing a model from the config
        >>> model = BertGenerationEncoder(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = 'bert-generation'

    def __init__(self, vocab_size=50358, hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0, bos_token_id=2, eos_token_id=1, gradient_checkpointing=False, position_embedding_type='absolute', use_cache=True, **kwargs):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache


BERT_GENERATION_INPUTS_DOCSTRING = """
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertGenerationTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.__call__` and :meth:`transformers.PreTrainedTokenizer.encode` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


BERT_GENERATION_START_DOCSTRING = """

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.BertGenerationConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""


class BertGenerationOnlyLMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        logits = self.decoder(hidden_states)
        return logits


class BlenderbotLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: 'int', embedding_dim: 'int'):
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, input_ids_shape: 'torch.Size', past_key_values_length: 'int'=0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions)


class BlenderbotAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim: 'int', num_heads: 'int', dropout: 'float'=0.0, is_decoder: 'bool'=False, bias: 'bool'=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).'
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: 'torch.Tensor', seq_len: 'int', bsz: 'int'):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: 'torch.Tensor', key_value_states: 'Optional[torch.Tensor]'=None, past_key_value: 'Optional[Tuple[torch.Tensor]]'=None, attention_mask: 'Optional[torch.Tensor]'=None, layer_head_mask: 'Optional[torch.Tensor]'=None, output_attentions: 'bool'=False) ->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        if self.is_decoder:
            past_key_value = key_states, value_states
        proj_shape = bsz * self.num_heads, -1, self.head_dim
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len), f'Attention weights should be of size {bsz * self.num_heads, tgt_len, src_len}, but is {attn_weights.size()}'
        if attention_mask is not None:
            assert attention_mask.size() == (bsz, 1, tgt_len, src_len), f'Attention mask should be of size {bsz, 1, tgt_len, src_len}, but is {attention_mask.size()}'
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        if layer_head_mask is not None:
            assert layer_head_mask.size() == (self.num_heads,), f'Head mask for a single layer should be of size {self.num_heads,}, but is {layer_head_mask.size()}'
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim), f'`attn_output` should be of size {bsz, self.num_heads, tgt_len, self.head_dim}, but is {attn_output.size()}'
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped, past_key_value


class BlenderbotEncoderLayer(nn.Module):

    def __init__(self, config: 'BlenderbotConfig'):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BlenderbotAttention(embed_dim=self.embed_dim, num_heads=config.encoder_attention_heads, dropout=config.attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'torch.Tensor', layer_head_mask: 'torch.Tensor', output_attentions: 'bool'=False):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask, output_attentions=output_attentions)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16 and (torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        outputs = hidden_states,
        if output_attentions:
            outputs += attn_weights,
        return outputs


class BlenderbotDecoderLayer(nn.Module):

    def __init__(self, config: 'BlenderbotConfig'):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BlenderbotAttention(embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout, is_decoder=True)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BlenderbotAttention(self.embed_dim, config.decoder_attention_heads, dropout=config.attention_dropout, is_decoder=True)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]'=None, encoder_hidden_states: 'Optional[torch.Tensor]'=None, encoder_attention_mask: 'Optional[torch.Tensor]'=None, layer_head_mask: 'Optional[torch.Tensor]'=None, encoder_layer_head_mask: 'Optional[torch.Tensor]'=None, past_key_value: 'Optional[Tuple[torch.Tensor]]'=None, output_attentions: 'Optional[bool]'=False, use_cache: 'Optional[bool]'=True):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            encoder_layer_head_mask (:obj:`torch.FloatTensor`): mask for encoder attention heads in a given layer of
                size `(config.encoder_attention_heads,)`.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        hidden_states, self_attn_weights, present_key_value = self.self_attn(hidden_states=hidden_states, past_key_value=self_attn_past_key_value, attention_mask=attention_mask, layer_head_mask=layer_head_mask, output_attentions=output_attentions)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(hidden_states=hidden_states, key_value_states=encoder_hidden_states, attention_mask=encoder_attention_mask, layer_head_mask=layer_head_mask, past_key_value=cross_attn_past_key_value, output_attentions=output_attentions)
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            present_key_value = present_key_value + cross_attn_present_key_value
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        outputs = hidden_states,
        if output_attentions:
            outputs += self_attn_weights, cross_attn_weights
        if use_cache:
            outputs += present_key_value,
        return outputs


class BlenderbotConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.BlenderbotModel`. It is used
    to instantiate an Blenderbot model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Blenderbot
    `facebook/blenderbot-3B <https://huggingface.co/facebook/blenderbot-3B>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 50265):
            Vocabulary size of the Blenderbot model. Defines the number of different tokens that can be represented by
            the :obj:`inputs_ids` passed when calling :class:`~transformers.BlenderbotModel` or
            :class:`~transformers.TFBlenderbotModel`.
        d_model (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        encoder_layers (:obj:`int`, `optional`, defaults to 12):
            Number of encoder layers.
        decoder_layers (:obj:`int`, `optional`, defaults to 12):
            Number of decoder layers.
        encoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        classifier_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for classifier.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        encoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the encoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        decoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the decoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        scale_embedding (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Scale embeddings by diving by sqrt(d_model).
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models)
        forced_eos_token_id (:obj:`int`, `optional`, defaults to 2):
            The id of the token to force as the last generated token when :obj:`max_length` is reached. Usually set to
            :obj:`eos_token_id`.

    Example::

        >>> from transformers import BlenderbotModel, BlenderbotConfig

        >>> # Initializing a Blenderbot facebook/blenderbot-3B style configuration
        >>> configuration = BlenderbotConfig()

        >>> # Initializing a model from the facebook/blenderbot-3B style configuration
        >>> model = BlenderbotModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = 'blenderbot'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(self, vocab_size=8008, max_position_embeddings=128, encoder_layers=2, encoder_ffn_dim=10240, encoder_attention_heads=32, decoder_layers=24, decoder_ffn_dim=10240, decoder_attention_heads=32, encoder_layerdrop=0.0, decoder_layerdrop=0.0, use_cache=True, is_encoder_decoder=True, activation_function='gelu', d_model=2560, dropout=0.1, attention_dropout=0.0, activation_dropout=0.0, init_std=0.02, decoder_start_token_id=1, classifier_dropout=0.0, scale_embedding=False, gradient_checkpointing=False, pad_token_id=0, bos_token_id=1, eos_token_id=2, encoder_no_repeat_ngram_size=3, forced_eos_token_id=2, **kwargs):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, is_encoder_decoder=is_encoder_decoder, decoder_start_token_id=decoder_start_token_id, encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size, forced_eos_token_id=forced_eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_embedding = scale_embedding

    @property
    def num_attention_heads(self) ->int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) ->int:
        return self.d_model


BLENDERBOT_INPUTS_DOCSTRING = """
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using :class:`~transformers.BlenderbotTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BlenderbotTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__

            Blenderbot uses the :obj:`bos_token_id` as the starting token for :obj:`decoder_input_ids` generation. If
            :obj:`past_key_values` is used, optionally only the last :obj:`decoder_input_ids` have to be input (see
            :obj:`past_key_values`).
        decoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.

            If you want to change padding behavior, you should read :func:`modeling_blenderbot._prepare_decoder_inputs`
            and modify to your needs. See diagram 1 in `the paper <https://arxiv.org/abs/1910.13461>`__ for more
            information on the default strategy.
        head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the heas is **masked**.

        decoder_head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`):
            Tuple consists of (:obj:`last_hidden_state`, `optional`: :obj:`hidden_states`, `optional`:
            :obj:`attentions`) :obj:`last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`,
            `optional`) is a sequence of hidden-states at the output of the last layer of the encoder. Used in the
            cross-attention of the decoder.
        past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size, sequence_length)`.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded
            representation. If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_inputs_embeds`
            have to be input (see :obj:`past_key_values`). This is useful if you want more control over how to convert
            :obj:`decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If :obj:`decoder_input_ids` and :obj:`decoder_inputs_embeds` are both unset, :obj:`decoder_inputs_embeds`
            takes the value of :obj:`inputs_embeds`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


BLENDERBOT_START_DOCSTRING = """
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.BlenderbotConfig`):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""


BLENDERBOT_SMALL_INPUTS_DOCSTRING = """
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using :class:`~transformers.BlenderbotSmallTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BlenderbotSmallTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__

            BlenderbotSmall uses the :obj:`bos_token_id` as the starting token for :obj:`decoder_input_ids` generation.
            If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_input_ids` have to be input (see
            :obj:`past_key_values`).
        decoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.

            If you want to change padding behavior, you should read
            :func:`modeling_blenderbot_small._prepare_decoder_inputs` and modify to your needs. See diagram 1 in `the
            paper <https://arxiv.org/abs/1910.13461>`__ for more information on the default strategy.
        head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the heas is **masked**.

        decoder_head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`):
            Tuple consists of (:obj:`last_hidden_state`, `optional`: :obj:`hidden_states`, `optional`:
            :obj:`attentions`) :obj:`last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`,
            `optional`) is a sequence of hidden-states at the output of the last layer of the encoder. Used in the
            cross-attention of the decoder.
        past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size, sequence_length)`.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded
            representation. If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_inputs_embeds`
            have to be input (see :obj:`past_key_values`). This is useful if you want more control over how to convert
            :obj:`decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If :obj:`decoder_input_ids` and :obj:`decoder_inputs_embeds` are both unset, :obj:`decoder_inputs_embeds`
            takes the value of :obj:`inputs_embeds`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


BLENDERBOT_SMALL_START_DOCSTRING = """
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.BlenderbotSmallConfig`):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""


class BlenderbotSmallAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim: 'int', num_heads: 'int', dropout: 'float'=0.0, is_decoder: 'bool'=False, bias: 'bool'=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).'
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: 'torch.Tensor', seq_len: 'int', bsz: 'int'):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: 'torch.Tensor', key_value_states: 'Optional[torch.Tensor]'=None, past_key_value: 'Optional[Tuple[torch.Tensor]]'=None, attention_mask: 'Optional[torch.Tensor]'=None, layer_head_mask: 'Optional[torch.Tensor]'=None, output_attentions: 'bool'=False) ->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        if self.is_decoder:
            past_key_value = key_states, value_states
        proj_shape = bsz * self.num_heads, -1, self.head_dim
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len), f'Attention weights should be of size {bsz * self.num_heads, tgt_len, src_len}, but is {attn_weights.size()}'
        if attention_mask is not None:
            assert attention_mask.size() == (bsz, 1, tgt_len, src_len), f'Attention mask should be of size {bsz, 1, tgt_len, src_len}, but is {attention_mask.size()}'
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        if layer_head_mask is not None:
            assert layer_head_mask.size() == (self.num_heads,), f'Head mask for a single layer should be of size {self.num_heads,}, but is {layer_head_mask.size()}'
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim), f'`attn_output` should be of size {bsz, self.num_heads, tgt_len, self.head_dim}, but is {attn_output.size()}'
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped, past_key_value


class BlenderbotSmallDecoderLayer(nn.Module):

    def __init__(self, config: 'BlenderbotSmallConfig'):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BlenderbotSmallAttention(embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout, is_decoder=True)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BlenderbotSmallAttention(self.embed_dim, config.decoder_attention_heads, dropout=config.attention_dropout, is_decoder=True)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]'=None, encoder_hidden_states: 'Optional[torch.Tensor]'=None, encoder_attention_mask: 'Optional[torch.Tensor]'=None, layer_head_mask: 'Optional[torch.Tensor]'=None, encoder_layer_head_mask: 'Optional[torch.Tensor]'=None, past_key_value: 'Optional[Tuple[torch.Tensor]]'=None, output_attentions: 'Optional[bool]'=False, use_cache: 'Optional[bool]'=True):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            encoder_layer_head_mask (:obj:`torch.FloatTensor`): mask for encoder attention heads in a given layer of
                size `(config.encoder_attention_heads,)`.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        hidden_states, self_attn_weights, present_key_value = self.self_attn(hidden_states=hidden_states, past_key_value=self_attn_past_key_value, attention_mask=attention_mask, layer_head_mask=layer_head_mask, output_attentions=output_attentions)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(hidden_states=hidden_states, key_value_states=encoder_hidden_states, attention_mask=encoder_attention_mask, layer_head_mask=encoder_layer_head_mask, past_key_value=cross_attn_past_key_value, output_attentions=output_attentions)
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            present_key_value = present_key_value + cross_attn_present_key_value
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        outputs = hidden_states,
        if output_attentions:
            outputs += self_attn_weights, cross_attn_weights
        if use_cache:
            outputs += present_key_value,
        return outputs


class BlenderbotSmallLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: 'int', embedding_dim: 'int'):
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, input_ids_shape: 'torch.Size', past_key_values_length: 'int'=0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions)


class BlenderbotSmallConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.BlenderbotSmallModel`. It is
    used to instantiate an BlenderbotSmall model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the BlenderbotSmall
    `facebook/blenderbot_small-90M <https://huggingface.co/facebook/blenderbot_small-90M>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 50265):
            Vocabulary size of the BlenderbotSmall model. Defines the number of different tokens that can be
            represented by the :obj:`inputs_ids` passed when calling :class:`~transformers.BlenderbotSmallModel` or
            :class:`~transformers.TFBlenderbotSmallModel`.
        d_model (:obj:`int`, `optional`, defaults to 512):
            Dimensionality of the layers and the pooler layer.
        encoder_layers (:obj:`int`, `optional`, defaults to 8):
            Number of encoder layers.
        decoder_layers (:obj:`int`, `optional`, defaults to 8):
            Number of decoder layers.
        encoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (:obj:`int`, `optional`, defaults to 2048):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (:obj:`int`, `optional`, defaults to 2048):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        classifier_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for classifier.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        encoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the encoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        decoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the decoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        scale_embedding (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Scale embeddings by diving by sqrt(d_model).
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models)
        forced_eos_token_id (:obj:`int`, `optional`, defaults to 2):
            The id of the token to force as the last generated token when :obj:`max_length` is reached. Usually set to
            :obj:`eos_token_id`.

    Example::

        >>> from transformers import BlenderbotSmallModel, BlenderbotSmallConfig

        >>> # Initializing a BlenderbotSmall facebook/blenderbot_small-90M style configuration
        >>> configuration = BlenderbotSmallConfig()

        >>> # Initializing a model from the facebook/blenderbot_small-90M style configuration
        >>> model = BlenderbotSmallModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = 'blenderbot-small'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(self, vocab_size=50265, max_position_embeddings=512, encoder_layers=8, encoder_ffn_dim=2048, encoder_attention_heads=16, decoder_layers=8, decoder_ffn_dim=2048, decoder_attention_heads=16, encoder_layerdrop=0.0, decoder_layerdrop=0.0, use_cache=True, is_encoder_decoder=True, activation_function='gelu', d_model=512, dropout=0.1, attention_dropout=0.0, activation_dropout=0.0, init_std=0.02, decoder_start_token_id=1, classifier_dropout=0.0, scale_embedding=False, gradient_checkpointing=False, pad_token_id=0, bos_token_id=1, eos_token_id=2, forced_eos_token_id=2, **kwargs):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, is_encoder_decoder=is_encoder_decoder, decoder_start_token_id=decoder_start_token_id, forced_eos_token_id=forced_eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_embedding = scale_embedding

    @property
    def num_attention_heads(self) ->int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) ->int:
        return self.d_model


class BlenderbotSmallEncoderLayer(nn.Module):

    def __init__(self, config: 'BlenderbotSmallConfig'):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BlenderbotSmallAttention(embed_dim=self.embed_dim, num_heads=config.encoder_attention_heads, dropout=config.attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: 'torch.Tensor', attention_mask: 'torch.Tensor', layer_head_mask: 'torch.Tensor', output_attentions: 'bool'=False):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask, output_attentions=output_attentions)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        if hidden_states.dtype == torch.float16 and (torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        outputs = hidden_states,
        if output_attentions:
            outputs += attn_weights,
        return outputs


BLENDERBOT_GENERATION_EXAMPLE = """
    Conversation example::

        >>> from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
        >>> mname = 'facebook/blenderbot-400M-distill'
        >>> model = BlenderbotForConditionalGeneration.from_pretrained(mname)
        >>> tokenizer = BlenderbotTokenizer.from_pretrained(mname)
        >>> UTTERANCE = "My friends are cool but they eat too many carbs."
        >>> print("Human: ", UTTERANCE)
        >>> inputs = tokenizer([UTTERANCE], return_tensors='pt')
        >>> reply_ids = model.generate(**inputs)
        >>> print("Bot: ", tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0])

        >>> REPLY = "I'm not sure"
        >>> print("Human: ", REPLY)
        >>> NEXT_UTTERANCE = (
        ... "My friends are cool but they eat too many carbs.</s> <s>That's unfortunate. "
        ... "Are they trying to lose weight or are they just trying to be healthier?</s> "
        ... "<s> I'm not sure."
        ... )
        >>> inputs = tokenizer([NEXT_UTTERANCE], return_tensors='pt')
        >>> next_reply_ids = model.generate(**inputs)
        >>> print("Bot: ", tokenizer.batch_decode(next_reply_ids, skip_special_tokens=True)[0])
"""


BLENDERBOT_SMALL_GENERATION_EXAMPLE = """
    Conversation example::

        >>> from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration
        >>> mname = 'facebook/blenderbot_small-90M'
        >>> model = BlenderbotSmallForConditionalGeneration.from_pretrained(mname)
        >>> tokenizer = BlenderbotSmallTokenizer.from_pretrained(mname)
        >>> UTTERANCE = "My friends are cool but they eat too many carbs."
        >>> print("Human: ", UTTERANCE)
        >>> inputs = tokenizer([UTTERANCE], return_tensors='pt')
        >>> inputs.pop("token_type_ids")
        >>> reply_ids = model.generate(**inputs)
        >>> print("Bot: ", tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0])
        what kind of carbs do they eat? i don't know much about carbs.

        >>> REPLY = "I'm not sure"
        >>> print("Human: ", REPLY)
        >>> NEXT_UTTERANCE = (
        ... "My friends are cool but they eat too many carbs.</s> "
        ... "<s>what kind of carbs do they eat? i don't know much about carbs.</s> "
        ... "<s>I'm not sure."
        ... )
        >>> inputs = tokenizer([NEXT_UTTERANCE], return_tensors='pt')
        >>> inputs.pop("token_type_ids")
        >>> next_reply_ids = model.generate(**inputs)
        >>> print("Bot: ", tokenizer.batch_decode(next_reply_ids, skip_special_tokens=True)[0])
"""


class ConvBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ConvBertConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.ConvBertModel`. It is used to
    instantiate an ConvBERT model according to the specified arguments, defining the model architecture. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the ConvBERT `conv-bert-base
    <https://huggingface.co/YituTech/conv-bert-base>`__ architecture. Configuration objects inherit from
    :class:`~transformers.PretrainedConfig` and can be used to control the model outputs. Read the documentation from
    :class:`~transformers.PretrainedConfig` for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the ConvBERT model. Defines the number of different tokens that can be represented by
            the :obj:`inputs_ids` passed when calling :class:`~transformers.ConvBertModel` or
            :class:`~transformers.TFConvBertModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.ConvBertModel`
            or :class:`~transformers.TFConvBertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        head_ratio (:obj:`int`, `optional`, defaults to 2):
            Ratio gamma to reduce the number of attention heads.
        num_groups (:obj:`int`, `optional`, defaults to 1):
            The number of groups for grouped linear layers for ConvBert model
        conv_kernel_size (:obj:`int`, `optional`, defaults to 9):
            The size of the convolutional kernel.


    Example::
        >>> from transformers import ConvBertModel, ConvBertConfig
        >>> # Initializing a ConvBERT convbert-base-uncased style configuration
        >>> configuration = ConvBertConfig()
        >>> # Initializing a model from the convbert-base-uncased style configuration
        >>> model = ConvBertModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = 'convbert'

    def __init__(self, vocab_size=30522, hidden_size=768, is_encoder_decoder=False, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=1, bos_token_id=0, eos_token_id=2, embedding_size=768, head_ratio=2, conv_kernel_size=9, num_groups=1, **kwargs):
        super().__init__(pad_token_id=pad_token_id, is_encoder_decoder=is_encoder_decoder, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.embedding_size = embedding_size
        self.head_ratio = head_ratio
        self.conv_kernel_size = conv_kernel_size
        self.num_groups = num_groups


def load_tf_weights_in_convbert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import tensorflow as tf
    except ImportError:
        logger.error('Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.')
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info('Converting TensorFlow checkpoint from {}'.format(tf_path))
    init_vars = tf.train.list_variables(tf_path)
    tf_data = {}
    for name, shape in init_vars:
        logger.info('Loading TF weight {} with shape {}'.format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        tf_data[name] = array
    param_mapping = {'embeddings.word_embeddings.weight': 'electra/embeddings/word_embeddings', 'embeddings.position_embeddings.weight': 'electra/embeddings/position_embeddings', 'embeddings.token_type_embeddings.weight': 'electra/embeddings/token_type_embeddings', 'embeddings.LayerNorm.weight': 'electra/embeddings/LayerNorm/gamma', 'embeddings.LayerNorm.bias': 'electra/embeddings/LayerNorm/beta', 'embeddings_project.weight': 'electra/embeddings_project/kernel', 'embeddings_project.bias': 'electra/embeddings_project/bias'}
    if config.num_groups > 1:
        group_dense_name = 'g_dense'
    else:
        group_dense_name = 'dense'
    for j in range(config.num_hidden_layers):
        param_mapping[f'encoder.layer.{j}.attention.self.query.weight'] = f'electra/encoder/layer_{j}/attention/self/query/kernel'
        param_mapping[f'encoder.layer.{j}.attention.self.query.bias'] = f'electra/encoder/layer_{j}/attention/self/query/bias'
        param_mapping[f'encoder.layer.{j}.attention.self.key.weight'] = f'electra/encoder/layer_{j}/attention/self/key/kernel'
        param_mapping[f'encoder.layer.{j}.attention.self.key.bias'] = f'electra/encoder/layer_{j}/attention/self/key/bias'
        param_mapping[f'encoder.layer.{j}.attention.self.value.weight'] = f'electra/encoder/layer_{j}/attention/self/value/kernel'
        param_mapping[f'encoder.layer.{j}.attention.self.value.bias'] = f'electra/encoder/layer_{j}/attention/self/value/bias'
        param_mapping[f'encoder.layer.{j}.attention.self.key_conv_attn_layer.depthwise.weight'] = f'electra/encoder/layer_{j}/attention/self/conv_attn_key/depthwise_kernel'
        param_mapping[f'encoder.layer.{j}.attention.self.key_conv_attn_layer.pointwise.weight'] = f'electra/encoder/layer_{j}/attention/self/conv_attn_key/pointwise_kernel'
        param_mapping[f'encoder.layer.{j}.attention.self.key_conv_attn_layer.bias'] = f'electra/encoder/layer_{j}/attention/self/conv_attn_key/bias'
        param_mapping[f'encoder.layer.{j}.attention.self.conv_kernel_layer.weight'] = f'electra/encoder/layer_{j}/attention/self/conv_attn_kernel/kernel'
        param_mapping[f'encoder.layer.{j}.attention.self.conv_kernel_layer.bias'] = f'electra/encoder/layer_{j}/attention/self/conv_attn_kernel/bias'
        param_mapping[f'encoder.layer.{j}.attention.self.conv_out_layer.weight'] = f'electra/encoder/layer_{j}/attention/self/conv_attn_point/kernel'
        param_mapping[f'encoder.layer.{j}.attention.self.conv_out_layer.bias'] = f'electra/encoder/layer_{j}/attention/self/conv_attn_point/bias'
        param_mapping[f'encoder.layer.{j}.attention.output.dense.weight'] = f'electra/encoder/layer_{j}/attention/output/dense/kernel'
        param_mapping[f'encoder.layer.{j}.attention.output.LayerNorm.weight'] = f'electra/encoder/layer_{j}/attention/output/LayerNorm/gamma'
        param_mapping[f'encoder.layer.{j}.attention.output.dense.bias'] = f'electra/encoder/layer_{j}/attention/output/dense/bias'
        param_mapping[f'encoder.layer.{j}.attention.output.LayerNorm.bias'] = f'electra/encoder/layer_{j}/attention/output/LayerNorm/beta'
        param_mapping[f'encoder.layer.{j}.intermediate.dense.weight'] = f'electra/encoder/layer_{j}/intermediate/{group_dense_name}/kernel'
        param_mapping[f'encoder.layer.{j}.intermediate.dense.bias'] = f'electra/encoder/layer_{j}/intermediate/{group_dense_name}/bias'
        param_mapping[f'encoder.layer.{j}.output.dense.weight'] = f'electra/encoder/layer_{j}/output/{group_dense_name}/kernel'
        param_mapping[f'encoder.layer.{j}.output.dense.bias'] = f'electra/encoder/layer_{j}/output/{group_dense_name}/bias'
        param_mapping[f'encoder.layer.{j}.output.LayerNorm.weight'] = f'electra/encoder/layer_{j}/output/LayerNorm/gamma'
        param_mapping[f'encoder.layer.{j}.output.LayerNorm.bias'] = f'electra/encoder/layer_{j}/output/LayerNorm/beta'
    for param in model.named_parameters():
        param_name = param[0]
        retriever = attrgetter(param_name)
        result = retriever(model)
        tf_name = param_mapping[param_name]
        value = torch.from_numpy(tf_data[tf_name])
        logger.info(f'TF: {tf_name}, PT: {param_name} ')
        if tf_name.endswith('/kernel'):
            if not tf_name.endswith('/intermediate/g_dense/kernel'):
                if not tf_name.endswith('/output/g_dense/kernel'):
                    value = value.T
        if tf_name.endswith('/depthwise_kernel'):
            value = value.permute(1, 2, 0)
        if tf_name.endswith('/pointwise_kernel'):
            value = value.permute(2, 1, 0)
        if tf_name.endswith('/conv_attn_key/bias'):
            value = value.unsqueeze(-1)
        result.data = value
    return model


class SeparableConv1D(nn.Module):
    """This class implements separable convolution, i.e. a depthwise and a pointwise layer"""

    def __init__(self, config, input_filters, output_filters, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Conv1d(input_filters, input_filters, kernel_size=kernel_size, groups=input_filters, padding=kernel_size // 2, bias=False)
        self.pointwise = nn.Conv1d(input_filters, output_filters, kernel_size=1, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_filters, 1))
        self.depthwise.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.pointwise.weight.data.normal_(mean=0.0, std=config.initializer_range)

    def forward(self, hidden_states):
        x = self.depthwise(hidden_states)
        x = self.pointwise(x)
        x += self.bias
        return x


class ConvBertSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, 'embedding_size'):
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        new_num_attention_heads = config.num_attention_heads // config.head_ratio
        if new_num_attention_heads < 1:
            self.head_ratio = config.num_attention_heads
            self.num_attention_heads = 1
        else:
            self.num_attention_heads = new_num_attention_heads
            self.head_ratio = config.head_ratio
        self.conv_kernel_size = config.conv_kernel_size
        assert config.hidden_size % self.num_attention_heads == 0, 'hidden_size should be divisible by num_attention_heads'
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.key_conv_attn_layer = SeparableConv1D(config, config.hidden_size, self.all_head_size, self.conv_kernel_size)
        self.conv_kernel_layer = nn.Linear(self.all_head_size, self.num_attention_heads * self.conv_kernel_size)
        self.conv_out_layer = nn.Linear(config.hidden_size, self.all_head_size)
        self.unfold = nn.Unfold(kernel_size=[self.conv_kernel_size, 1], padding=[int((self.conv_kernel_size - 1) / 2), 0])
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        batch_size = hidden_states.size(0)
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        conv_out_layer = self.conv_out_layer(hidden_states)
        conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
        conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        conv_out_layer = nn.functional.unfold(conv_out_layer, kernel_size=[self.conv_kernel_size, 1], dilation=1, padding=[(self.conv_kernel_size - 1) // 2, 0], stride=1)
        conv_out_layer = conv_out_layer.transpose(1, 2).reshape(batch_size, -1, self.all_head_size, self.conv_kernel_size)
        conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
        context_layer = torch.cat([context_layer, conv_out], 2)
        new_context_layer_shape = context_layer.size()[:-2] + (self.head_ratio * self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class ConvBertSelfOutput(nn.Module):

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


class ConvBertAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = ConvBertSelfAttention(config)
        self.output = ConvBertSelfOutput(config)
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

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, output_attentions=False):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class GroupedLinearLayer(nn.Module):

    def __init__(self, input_size, output_size, num_groups):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_groups = num_groups
        self.group_in_dim = self.input_size // self.num_groups
        self.group_out_dim = self.output_size // self.num_groups
        self.weight = nn.Parameter(torch.Tensor(self.num_groups, self.group_in_dim, self.group_out_dim))
        self.bias = nn.Parameter(torch.Tensor(output_size))

    def forward(self, hidden_states):
        batch_size = list(hidden_states.size())[0]
        x = torch.reshape(hidden_states, [-1, self.num_groups, self.group_in_dim])
        x = x.permute(1, 0, 2)
        x = torch.matmul(x, self.weight)
        x = x.permute(1, 0, 2)
        x = torch.reshape(x, [batch_size, -1, self.output_size])
        x = x + self.bias
        return x


class ConvBertIntermediate(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.num_groups == 1:
            self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        else:
            self.dense = GroupedLinearLayer(input_size=config.hidden_size, output_size=config.intermediate_size, num_groups=config.num_groups)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ConvBertOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.num_groups == 1:
            self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        else:
            self.dense = GroupedLinearLayer(input_size=config.intermediate_size, output_size=config.hidden_size, num_groups=config.num_groups)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ConvBertLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ConvBertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f'{self} should be used as a decoder model if cross attention is added'
            self.crossattention = ConvBertAttention(config)
        self.intermediate = ConvBertIntermediate(config)
        self.output = ConvBertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(self, 'crossattention'), f'If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`'
            cross_attention_outputs = self.crossattention(attention_output, encoder_attention_mask, head_mask, encoder_hidden_states, output_attentions)
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]
        layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class ConvBertEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ConvBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            if getattr(self.config, 'gradient_checkpointing', False):

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)
                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(layer_module), hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None)
        return BaseModelOutputWithCrossAttentions(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions, cross_attentions=all_cross_attentions)


class ConvBertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


CONVBERT_INPUTS_DOCSTRING = """
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.ConvBertTokenizer`. See
            :func:`transformers.PreTrainedTokenizer.encode` and :func:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`{0}`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:


            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:


            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:


            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


CONVBERT_START_DOCSTRING = """
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config (:class:`~transformers.ConvBertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""


class ConvBertGeneratorPredictions(nn.Module):
    """Prediction module for the generator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(config.embedding_size)
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)

    def forward(self, generator_hidden_states):
        hidden_states = self.dense(generator_hidden_states)
        hidden_states = get_activation('gelu')(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class ConvBertClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config

    def forward(self, hidden_states, **kwargs):
        x = hidden_states[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class MultiHeadAttention(nn.Module):
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, config):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = config.attention_dropout
        assert self.dim % self.n_heads == 0
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        attention_head_size = self.dim // self.n_heads
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.n_heads, attention_head_size, self.pruned_heads)
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        self.n_heads = self.n_heads - len(heads)
        self.dim = attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input, mask, kv=None, cache=None, head_mask=None, output_attentions=False):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen if cache is None else cache['slen'] + qlen
        else:
            klen = kv.size(1)
        n_heads = self.n_heads
        dim_per_head = self.dim // n_heads
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        q = shape(self.q_lin(input))
        if kv is None:
            k = shape(self.k_lin(input))
            v = shape(self.v_lin(input))
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = shape(self.k_lin(k))
            v = shape(self.v_lin(v))
        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = torch.cat([k_, k], dim=2)
                    v = torch.cat([v_, v], dim=2)
                else:
                    k, v = cache[self.layer_id]
            cache[self.layer_id] = k, v
        q = q / math.sqrt(dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))
        mask = (mask == 0).view(mask_reshape).expand_as(scores)
        scores.masked_fill_(mask, -float('inf'))
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        if head_mask is not None:
            weights = weights * head_mask
        context = torch.matmul(weights, v)
        context = unshape(context)
        outputs = self.out_lin(context),
        if output_attentions:
            outputs = outputs + (weights,)
        return outputs


class EncoderLayer(nn.Module):

    def __init__(self, config: 'FSMTConfig'):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Attention(self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask, layer_head_mask, output_attentions=False):
        """
        Args:
            x (:obj:`torch.Tensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (:obj:`torch.ByteTensor`): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        x, attn_weights = self.self_attn(query=x, key=x, key_padding_mask=encoder_padding_mask, layer_head_mask=layer_head_mask, output_attentions=output_attentions)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)
        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn_weights


class CTRLConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.CTRLModel` or a
    :class:`~transformers.TFCTRLModel`. It is used to instantiate a CTRL model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the `ctrl <https://huggingface.co/ctrl>`__ architecture from SalesForce.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 246534):
            Vocabulary size of the CTRL model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.CTRLModel` or
            :class:`~transformers.TFCTRLModel`.
        n_positions (:obj:`int`, `optional`, defaults to 256):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_ctx (:obj:`int`, `optional`, defaults to 256):
            Dimensionality of the causal mask (usually same as n_positions).
        n_embd (:obj:`int`, `optional`, defaults to 1280):
            Dimensionality of the embeddings and hidden states.
        dff (:obj:`int`, `optional`, defaults to 8192):
            Dimensionality of the inner dimension of the feed forward networks (FFN).
        n_layer (:obj:`int`, `optional`, defaults to 48):
            Number of hidden layers in the Transformer encoder.
        n_head (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        resid_pdrop (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (:obj:`int`, `optional`, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (:obj:`float`, `optional`, defaults to 1e-6):
            The epsilon to use in the layer normalization layers
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models).


    Examples::

        >>> from transformers import CTRLModel, CTRLConfig

        >>> # Initializing a CTRL configuration
        >>> configuration = CTRLConfig()

        >>> # Initializing a model from the configuration
        >>> model = CTRLModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = 'ctrl'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(self, vocab_size=246534, n_positions=256, n_ctx=256, n_embd=1280, dff=8192, n_layer=48, n_head=16, resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1, layer_norm_epsilon=1e-06, initializer_range=0.02, summary_type='cls_index', summary_use_proj=True, summary_activation=None, summary_proj_to_labels=True, summary_first_dropout=0.1, use_cache=True, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.dff = dff
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.use_cache = use_cache

    @property
    def max_position_embeddings(self):
        return self.n_positions

    @property
    def hidden_size(self):
        return self.n_embd

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer


CTRL_INPUTS_DOCSTRING = """
    Args:
        input_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, input_ids_length)`):
            :obj:`input_ids_length` = ``sequence_length`` if ``past`` is ``None`` else ``past[0].shape[-2]``
            (``sequence_length`` of input past key value states).

            Indices of input sequence tokens in the vocabulary.

            If :obj:`past` is used, only input IDs that do not have their past calculated should be passed as
            ``input_ids``.

            Indices can be obtained using :class:`~transformers.CTRLTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.__call__` and :meth:`transformers.PreTrainedTokenizer.encode` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        past (:obj:`List[tf.Tensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model (see
            :obj:`past` output below). Can be used to speed up sequential decoding. The token ids which have their past
            given to this model should not be passed as input ids as they have already been computed.
        attention_mask (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`__
        position_ids (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`__
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, ``past`` key value states are returned and can be used to speed up decoding (see
            ``past``).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple. This
            argument can be used in eager mode, in graph mode the value will always be set to True.
        training (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


CTRL_START_DOCSTRING = """

    This model inherits from :class:`~transformers.TFPreTrainedModel`. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading or saving, resizing the input
    embeddings, pruning heads etc.)

    This model is also a `tf.keras.Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ subclass. Use
    it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage
    and behavior.

    .. note::

        TF 2.0 models accepts two formats as inputs:

        - having all inputs as keyword arguments (like PyTorch models), or
        - having all inputs as a list, tuple or dict in the first positional arguments.

        This second option is useful when using :meth:`tf.keras.Model.fit` method which currently requires having all
        the tensors in the first argument of the model call function: :obj:`model(inputs)`.

        If you choose this second option, there are three possibilities you can use to gather all the input Tensors in
        the first positional argument :

        - a single Tensor with :obj:`input_ids` only and nothing else: :obj:`model(inputs_ids)`
        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
          :obj:`model([input_ids, attention_mask])` or :obj:`model([input_ids, attention_mask, token_type_ids])`
        - a dictionary with one or several input Tensors associated to the input names given in the docstring:
          :obj:`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Parameters:
        config (:class:`~transformers.CTRLConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""


def angle_defn(pos, i, d_model_size):
    angle_rates = 1 / np.power(10000, 2 * (i // 2) / d_model_size)
    return pos * angle_rates


def positional_encoding(position, d_model_size):
    angle_rads = angle_defn(np.arange(position)[:, np.newaxis], np.arange(d_model_size)[np.newaxis, :], d_model_size)
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = tf.convert_to_tensor(np.concatenate([sines, cosines], axis=-1))
    return pos_encoding


class DropoutContext(object):

    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True


def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None
    if dropout > 0 and mask is None:
        mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).bool()
    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask
    return mask, dropout


class XDropout(torch.autograd.Function):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""

    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:
            mask, = ctx.saved_tensors
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            return grad_output, None


class StableDropout(torch.nn.Module):
    """
    Optimized dropout module for stabilizing the training

    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        """
        Call the module

        Args:
            x (:obj:`torch.tensor`): The input tensor to apply dropout
        """
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x

    def clear_context(self):
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob


class ContextPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states):
        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        return self.config.hidden_size


class DebertaLayerNorm(nn.Module):
    """LayerNorm module in the TF style (epsilon inside the square root)."""

    def __init__(self, size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_type = hidden_states.dtype
        hidden_states = hidden_states.float()
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states
        y = self.weight * hidden_states + self.bias
        return y


class DebertaSelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class XSoftmax(torch.autograd.Function):
    """
    Masked Softmax which is optimized for saving memory

    Args:
        input (:obj:`torch.tensor`): The input tensor that will apply softmax.
        mask (:obj:`torch.IntTensor`): The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
        dim (int): The dimension that will apply softmax

    Example::

          >>> import torch
          >>> from transformers.models.deberta_v2.modeling_deberta_v2 import XSoftmax

          >>> # Make a tensor
          >>> x = torch.randn([4,20,100])

          >>> # Create a mask
          >>> mask = (x>0).int()

          >>> y = XSoftmax.apply(x, mask, dim=-1)
    """

    @staticmethod
    def forward(self, input, mask, dim):
        self.dim = dim
        rmask = ~mask.bool()
        output = input.masked_fill(rmask, float('-inf'))
        output = torch.softmax(output, self.dim)
        output.masked_fill_(rmask, 0)
        self.save_for_backward(output)
        return output

    @staticmethod
    def backward(self, grad_output):
        output, = self.saved_tensors
        inputGrad = _softmax_backward_data(grad_output, output, self.dim, output)
        return inputGrad, None, None


def make_log_bucket_position(relative_pos, bucket_size, max_position):
    sign = np.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = np.where((relative_pos < mid) & (relative_pos > -mid), mid - 1, np.abs(relative_pos))
    log_pos = np.ceil(np.log(abs_pos / mid) / np.log((max_position - 1) / mid) * (mid - 1)) + mid
    bucket_pos = np.where(abs_pos <= mid, relative_pos, log_pos * sign).astype(np.int)
    return bucket_pos


def build_relative_position(query_size, key_size, bucket_size=-1, max_position=-1):
    """
    Build relative position according to the query and key

    We assume the absolute position of query :math:`P_q` is range from (0, query_size) and the absolute position of key
    :math:`P_k` is range from (0, key_size), The relative positions from query to key is :math:`R_{q \\rightarrow k} =
    P_q - P_k`

    Args:
        query_size (int): the length of query
        key_size (int): the length of key
        bucket_size (int): the size of position bucket
        max_position (int): the maxium allowed absolute positoin

    Return:
        :obj:`torch.LongTensor`: A tensor with shape [1, query_size, key_size]

    """
    q_ids = np.arange(0, query_size)
    k_ids = np.arange(0, key_size)
    rel_pos_ids = q_ids[:, None] - np.tile(k_ids, (q_ids.shape[0], 1))
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    rel_pos_ids = torch.tensor(rel_pos_ids, dtype=torch.long)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids


class DisentangledSelfAttention(torch.nn.Module):
    """
    Disentangled self-attention module

    Parameters:
        config (:obj:`DebertaV2Config`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            `BertConfig`, for more details, please refer :class:`~transformers.DebertaV2Config`

    """

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        _attention_head_size = config.hidden_size // config.num_attention_heads
        self.attention_head_size = getattr(config, 'attention_head_size', _attention_head_size)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        if config.apply_lora:
            self.query_proj = lora.Linear(config.hidden_size, self.all_head_size, r=config.lora_r, lora_alpha=config.lora_alpha, merge_weights=False)
        else:
            self.query_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        if config.apply_lora:
            self.value_proj = lora.Linear(config.hidden_size, self.all_head_size, r=config.lora_r, lora_alpha=config.lora_alpha, merge_weights=False)
        else:
            self.value_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.share_att_key = getattr(config, 'share_att_key', False)
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []
        self.relative_attention = getattr(config, 'relative_attention', False)
        if self.relative_attention:
            self.position_buckets = getattr(config, 'position_buckets', -1)
            self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets
            self.pos_dropout = StableDropout(config.hidden_dropout_prob)
            if not self.share_att_key:
                if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                    self.pos_key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
                if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                    self.pos_query_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = StableDropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, attention_heads):
        new_x_shape = x.size()[:-1] + (attention_heads, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))

    def forward(self, hidden_states, attention_mask, return_att=False, query_states=None, relative_pos=None, rel_embeddings=None):
        """
        Call the module

        Args:
            hidden_states (:obj:`torch.FloatTensor`):
                Input states to the module usually the output from previous layer, it will be the Q,K and V in
                `Attention(Q,K,V)`

            attention_mask (:obj:`torch.ByteTensor`):
                An attention mask matrix of shape [`B`, `N`, `N`] where `B` is the batch size, `N` is the maximum
                sequence length in which element [i,j] = `1` means the `i` th token in the input can attend to the `j`
                th token.

            return_att (:obj:`bool`, optional):
                Whether return the attention matrix.

            query_states (:obj:`torch.FloatTensor`, optional):
                The `Q` state in `Attention(Q,K,V)`.

            relative_pos (:obj:`torch.LongTensor`):
                The relative position encoding between the tokens in the sequence. It's of shape [`B`, `N`, `N`] with
                values ranging in [`-max_relative_positions`, `max_relative_positions`].

            rel_embeddings (:obj:`torch.FloatTensor`):
                The embedding of relative distances. It's a tensor of shape [:math:`2 \\times
                \\text{max_relative_positions}`, `hidden_size`].


        """
        if query_states is None:
            query_states = hidden_states
        q = self.query_proj(query_states)
        k = self.key_proj(hidden_states)
        v = self.value_proj(hidden_states)
        query_layer = self.transpose_for_scores(q, self.num_attention_heads)
        key_layer = self.transpose_for_scores(k, self.num_attention_heads)
        value_layer = self.transpose_for_scores(v, self.num_attention_heads)
        rel_att = None
        scale_factor = 1
        if 'c2p' in self.pos_att_type:
            scale_factor += 1
        if 'p2c' in self.pos_att_type:
            scale_factor += 1
        if 'p2p' in self.pos_att_type:
            scale_factor += 1
        scale = math.sqrt(query_layer.size(-1) * scale_factor)
        attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2)) / scale
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_attention_bias(query_layer, key_layer, relative_pos, rel_embeddings, scale_factor)
        if rel_att is not None:
            attention_scores = attention_scores + rel_att
        attention_scores = attention_scores
        attention_scores = attention_scores.view(-1, self.num_attention_heads, attention_scores.size(-2), attention_scores.size(-1))
        attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.bmm(attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer)
        context_layer = context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1)).permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if return_att:
            return context_layer, attention_probs
        else:
            return context_layer

    def manually_gather(self, input, index):
        assert input.dim() == 3
        assert index.dim() == 2
        assert input.size(1) == index.size(0)
        index = index + torch.arange(start=0, end=index.size(0) * input.shape[-1], step=input.shape[-1], device=index.device).view(-1, 1)
        return torch.index_select(input.view(input.shape[0], -1), dim=-1, index=index.view(-1)).view(input.shape[0], index.shape[0], index.shape[1])

    def disentangled_attention_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(q, key_layer.size(-2), bucket_size=self.position_buckets, max_position=self.max_relative_positions)
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        elif relative_pos.dim() != 4:
            raise ValueError(f'Relative postion ids must be of dim 2 or 3 or 4. {relative_pos.dim()}')
        att_span = self.pos_ebd_size
        relative_pos = relative_pos.long()
        rel_embeddings = rel_embeddings[self.pos_ebd_size - att_span:self.pos_ebd_size + att_span, :].unsqueeze(0)
        if self.share_att_key:
            pos_query_layer = self.transpose_for_scores(nn.Linear.forward(self.query_proj, rel_embeddings), self.num_attention_heads).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
            pos_key_layer = self.transpose_for_scores(self.key_proj(rel_embeddings), self.num_attention_heads).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
        else:
            if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                pos_key_layer = self.transpose_for_scores(self.pos_key_proj(rel_embeddings), self.num_attention_heads).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
            if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                pos_query_layer = self.transpose_for_scores(self.pos_query_proj(rel_embeddings), self.num_attention_heads).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
        score = 0
        if 'c2p' in self.pos_att_type:
            scale = math.sqrt(pos_key_layer.size(-1) * scale_factor)
            c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = self.manually_gather(c2p_att, c2p_pos.squeeze(0).squeeze(0))
            score += c2p_att / scale
        if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
            scale = math.sqrt(pos_query_layer.size(-1) * scale_factor)
            if key_layer.size(-2) != query_layer.size(-2):
                r_pos = build_relative_position(key_layer.size(-2), key_layer.size(-2), bucket_size=self.position_buckets, max_position=self.max_relative_positions)
                r_pos = r_pos.unsqueeze(0)
            else:
                r_pos = relative_pos
            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            if query_layer.size(-2) != key_layer.size(-2):
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)
        if 'p2c' in self.pos_att_type:
            p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_att = self.manually_gather(p2c_att, p2c_pos.squeeze(0).squeeze(0)).transpose(-1, -2)
            if query_layer.size(-2) != key_layer.size(-2):
                p2c_att = torch.gather(p2c_att, dim=-2, index=pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2))))
            score += p2c_att / scale
        if 'p2p' in self.pos_att_type:
            pos_query = pos_query_layer[:, :, att_span:, :]
            p2p_att = torch.matmul(pos_query, pos_key_layer.transpose(-1, -2))
            p2p_att = p2p_att.expand(query_layer.size()[:2] + p2p_att.size()[2:])
            if query_layer.size(-2) != key_layer.size(-2):
                p2p_att = torch.gather(p2p_att, dim=-2, index=pos_index.expand(query_layer.size()[:2] + (pos_index.size(-2), p2p_att.size(-1))))
            p2p_att = torch.gather(p2p_att, dim=-1, index=c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)]))
            score += p2p_att
        return score


class DebertaAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = DisentangledSelfAttention(config)
        self.output = DebertaSelfOutput(config)
        self.config = config

    def forward(self, hidden_states, attention_mask, return_att=False, query_states=None, relative_pos=None, rel_embeddings=None):
        self_output = self.self(hidden_states, attention_mask, return_att, query_states=query_states, relative_pos=relative_pos, rel_embeddings=rel_embeddings)
        if return_att:
            self_output, att_matrix = self_output
        if query_states is None:
            query_states = hidden_states
        attention_output = self.output(self_output, query_states)
        if return_att:
            return attention_output, att_matrix
        else:
            return attention_output


class DebertaIntermediate(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class DebertaOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DebertaLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = DebertaAttention(config)
        self.intermediate = DebertaIntermediate(config)
        self.output = DebertaOutput(config)

    def forward(self, hidden_states, attention_mask, return_att=False, query_states=None, relative_pos=None, rel_embeddings=None):
        attention_output = self.attention(hidden_states, attention_mask, return_att=return_att, query_states=query_states, relative_pos=relative_pos, rel_embeddings=rel_embeddings)
        if return_att:
            attention_output, att_matrix = attention_output
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if return_att:
            return layer_output, att_matrix
        else:
            return layer_output


class DebertaEncoder(nn.Module):
    """Modified BertEncoder with relative position bias support"""

    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([DebertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.relative_attention = getattr(config, 'relative_attention', False)
        if self.relative_attention:
            self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.rel_embeddings = nn.Embedding(self.max_relative_positions * 2, config.hidden_size)

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        return rel_embeddings

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
            attention_mask = attention_mask.byte()
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        return attention_mask

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
            relative_pos = build_relative_position(q, hidden_states.size(-2), hidden_states.device)
        return relative_pos

    def forward(self, hidden_states, attention_mask, output_hidden_states=True, output_attentions=False, query_states=None, relative_pos=None, return_dict=True):
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states
        rel_embeddings = self.get_rel_embedding()
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer_module(next_kv, attention_mask, output_attentions, query_states=query_states, relative_pos=relative_pos, rel_embeddings=rel_embeddings)
            if output_attentions:
                hidden_states, att_m = hidden_states
            if query_states is not None:
                query_states = hidden_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = hidden_states
            if output_attentions:
                all_attentions = all_attentions + (att_m,)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions)


class DebertaEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        pad_token_id = getattr(config, 'pad_token_id', 0)
        self.embedding_size = getattr(config, 'embedding_size', config.hidden_size)
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=pad_token_id)
        self.position_biased_input = getattr(config, 'position_biased_input', True)
        if not self.position_biased_input:
            self.position_embeddings = None
        else:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)
        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.embedding_size)
        if self.embedding_size != config.hidden_size:
            self.embed_proj = nn.Linear(self.embedding_size, config.hidden_size, bias=False)
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, mask=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(position_ids.long())
        else:
            position_embeddings = torch.zeros_like(inputs_embeds)
        embeddings = inputs_embeds
        if self.position_biased_input:
            embeddings += position_embeddings
        if self.config.type_vocab_size > 0:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings
        if self.embedding_size != self.config.hidden_size:
            embeddings = self.embed_proj(embeddings)
        embeddings = self.LayerNorm(embeddings)
        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            mask = mask
            embeddings = embeddings * mask
        embeddings = self.dropout(embeddings)
        return embeddings


class DebertaConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.DebertaModel` or a
    :class:`~transformers.TFDebertaModel`. It is used to instantiate a DeBERTa model according to the specified
    arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the DeBERTa `microsoft/deberta-base <https://huggingface.co/microsoft/deberta-base>`__
    architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Arguments:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the DeBERTa model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.DebertaModel` or
            :class:`~transformers.TFDebertaModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"`, :obj:`"gelu"`, :obj:`"tanh"`, :obj:`"gelu_fast"`,
            :obj:`"mish"`, :obj:`"linear"`, :obj:`"sigmoid"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.DebertaModel` or
            :class:`~transformers.TFDebertaModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        relative_attention (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether use relative position encoding.
        max_relative_positions (:obj:`int`, `optional`, defaults to 1):
            The range of relative positions :obj:`[-max_position_embeddings, max_position_embeddings]`. Use the same
            value as :obj:`max_position_embeddings`.
        pad_token_id (:obj:`int`, `optional`, defaults to 0):
            The value used to pad input_ids.
        position_biased_input (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether add absolute position embedding to content embedding.
        pos_att_type (:obj:`List[str]`, `optional`):
            The type of relative position attention, it can be a combination of :obj:`["p2c", "c2p", "p2p"]`, e.g.
            :obj:`["p2c"]`, :obj:`["p2c", "c2p"]`, :obj:`["p2c", "c2p", 'p2p"]`.
        layer_norm_eps (:obj:`float`, optional, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
    """
    model_type = 'deberta'

    def __init__(self, vocab_size=50265, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=0, initializer_range=0.02, layer_norm_eps=1e-07, relative_attention=False, max_relative_positions=-1, pad_token_id=0, position_biased_input=True, pos_att_type=None, pooler_dropout=0, pooler_hidden_act='gelu', **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.relative_attention = relative_attention
        self.max_relative_positions = max_relative_positions
        self.pad_token_id = pad_token_id
        self.position_biased_input = position_biased_input
        if type(pos_att_type) == str:
            pos_att_type = [x.strip() for x in pos_att_type.lower().split('|')]
        self.pos_att_type = pos_att_type
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.pooler_hidden_size = kwargs.get('pooler_hidden_size', hidden_size)
        self.pooler_dropout = pooler_dropout
        self.pooler_hidden_act = pooler_hidden_act


DEBERTA_INPUTS_DOCSTRING = """
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.DebertaV2Tokenizer`. See
            :func:`transformers.PreTrainedTokenizer.encode` and :func:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`{0}`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


DEBERTA_START_DOCSTRING = """
    The DeBERTa model was proposed in `DeBERTa: Decoding-enhanced BERT with Disentangled Attention
    <https://arxiv.org/abs/2006.03654>`_ by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen. It's build on top of
    BERT/RoBERTa with two improvements, i.e. disentangled attention and enhanced mask decoder. With those two
    improvements, it out perform BERT/RoBERTa on a majority of tasks with 80GB pretraining data.

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.```


    Parameters:
        config (:class:`~transformers.DebertaV2Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""


class DebertaPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class DebertaLMPredictionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transform = DebertaPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class DebertaOnlyMLMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = DebertaLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class DebertaV2SelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DebertaV2Attention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = DisentangledSelfAttention(config)
        self.output = DebertaV2SelfOutput(config)
        self.config = config

    def forward(self, hidden_states, attention_mask, return_att=False, query_states=None, relative_pos=None, rel_embeddings=None):
        self_output = self.self(hidden_states, attention_mask, return_att, query_states=query_states, relative_pos=relative_pos, rel_embeddings=rel_embeddings)
        if return_att:
            self_output, att_matrix = self_output
        if query_states is None:
            query_states = hidden_states
        attention_output = self.output(self_output, query_states)
        if return_att:
            return attention_output, att_matrix
        else:
            return attention_output


class DebertaV2Intermediate(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class DebertaV2Output(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DebertaV2Layer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = DebertaV2Attention(config)
        self.intermediate = DebertaV2Intermediate(config)
        self.output = DebertaV2Output(config)

    def forward(self, hidden_states, attention_mask, return_att=False, query_states=None, relative_pos=None, rel_embeddings=None):
        attention_output = self.attention(hidden_states, attention_mask, return_att=return_att, query_states=query_states, relative_pos=relative_pos, rel_embeddings=rel_embeddings)
        if return_att:
            attention_output, att_matrix = attention_output
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if return_att:
            return layer_output, att_matrix
        else:
            return layer_output


class ConvLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        kernel_size = getattr(config, 'conv_kernel_size', 3)
        groups = getattr(config, 'conv_groups', 1)
        self.conv_act = getattr(config, 'conv_act', 'tanh')
        self.conv = torch.nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size, padding=(kernel_size - 1) // 2, groups=groups)
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, residual_states, input_mask):
        out = self.conv(hidden_states.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        rmask = (1 - input_mask).bool()
        out.masked_fill_(rmask.unsqueeze(-1).expand(out.size()), 0)
        out = ACT2FN[self.conv_act](self.dropout(out))
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


class DebertaV2Encoder(nn.Module):
    """Modified BertEncoder with relative position bias support"""

    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([DebertaV2Layer(config) for _ in range(config.num_hidden_layers)])
        self.relative_attention = getattr(config, 'relative_attention', False)
        if self.relative_attention:
            self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.position_buckets = getattr(config, 'position_buckets', -1)
            pos_ebd_size = self.max_relative_positions * 2
            if self.position_buckets > 0:
                pos_ebd_size = self.position_buckets * 2
            self.rel_embeddings = nn.Embedding(pos_ebd_size, config.hidden_size)
        self.norm_rel_ebd = [x.strip() for x in getattr(config, 'norm_rel_ebd', 'none').lower().split('|')]
        if 'layer_norm' in self.norm_rel_ebd:
            self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=True)
        self.conv = ConvLayer(config) if getattr(config, 'conv_kernel_size', 0) > 0 else None

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        if rel_embeddings is not None and 'layer_norm' in self.norm_rel_ebd:
            rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
            attention_mask = attention_mask.byte()
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        return attention_mask

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
            relative_pos = build_relative_position(q, hidden_states.size(-2), bucket_size=self.position_buckets, max_position=self.max_relative_positions)
        return relative_pos

    def forward(self, hidden_states, attention_mask, output_hidden_states=True, output_attentions=False, query_states=None, relative_pos=None, return_dict=True):
        if attention_mask.dim() <= 2:
            input_mask = attention_mask
        else:
            input_mask = (attention_mask.sum(-2) > 0).byte()
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states
        rel_embeddings = self.get_rel_embedding()
        output_states = next_kv
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output_states,)
            output_states = layer_module(next_kv, attention_mask, output_attentions, query_states=query_states, relative_pos=relative_pos, rel_embeddings=rel_embeddings)
            if output_attentions:
                output_states, att_m = output_states
            if i == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states, input_mask)
            if query_states is not None:
                query_states = output_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = output_states
            if output_attentions:
                all_attentions = all_attentions + (att_m,)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (output_states,)
        if not return_dict:
            return tuple(v for v in [output_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=output_states, hidden_states=all_hidden_states, attentions=all_attentions)


class DebertaV2Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        pad_token_id = getattr(config, 'pad_token_id', 0)
        self.embedding_size = getattr(config, 'embedding_size', config.hidden_size)
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=pad_token_id)
        self.position_biased_input = getattr(config, 'position_biased_input', True)
        if not self.position_biased_input:
            self.position_embeddings = None
        else:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)
        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.embedding_size)
        if self.embedding_size != config.hidden_size:
            self.embed_proj = nn.Linear(self.embedding_size, config.hidden_size, bias=False)
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, mask=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(position_ids.long())
        else:
            position_embeddings = torch.zeros_like(inputs_embeds)
        embeddings = inputs_embeds
        if self.position_biased_input:
            embeddings += position_embeddings
        if self.config.type_vocab_size > 0:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings
        if self.embedding_size != self.config.hidden_size:
            embeddings = self.embed_proj(embeddings)
        embeddings = self.LayerNorm(embeddings)
        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            mask = mask
            embeddings = embeddings * mask
        embeddings = self.dropout(embeddings)
        return embeddings


class DebertaV2Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.DebertaV2Model`. It is used
    to instantiate a DeBERTa-v2 model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the DeBERTa
    `microsoft/deberta-v2-xlarge <https://huggingface.co/microsoft/deberta-base>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Arguments:
        vocab_size (:obj:`int`, `optional`, defaults to 128100):
            Vocabulary size of the DeBERTa-v2 model. Defines the number of different tokens that can be represented by
            the :obj:`inputs_ids` passed when calling :class:`~transformers.DebertaV2Model`.
        hidden_size (:obj:`int`, `optional`, defaults to 1536):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 24):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 6144):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"`, :obj:`"gelu"`, :obj:`"tanh"`, :obj:`"gelu_fast"`,
            :obj:`"mish"`, :obj:`"linear"`, :obj:`"sigmoid"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 0):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.DebertaModel` or
            :class:`~transformers.TFDebertaModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-7):
            The epsilon used by the layer normalization layers.
        relative_attention (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether use relative position encoding.
        max_relative_positions (:obj:`int`, `optional`, defaults to -1):
            The range of relative positions :obj:`[-max_position_embeddings, max_position_embeddings]`. Use the same
            value as :obj:`max_position_embeddings`.
        pad_token_id (:obj:`int`, `optional`, defaults to 0):
            The value used to pad input_ids.
        position_biased_input (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether add absolute position embedding to content embedding.
        pos_att_type (:obj:`List[str]`, `optional`):
            The type of relative position attention, it can be a combination of :obj:`["p2c", "c2p", "p2p"]`, e.g.
            :obj:`["p2c"]`, :obj:`["p2c", "c2p"]`, :obj:`["p2c", "c2p", 'p2p"]`.
        layer_norm_eps (:obj:`float`, optional, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        cls_dropout (:obj:`float`, `optional`):
            cls dropout.
        apply_lora (:obj:`bool`, `optional`):
            apply Lora.
        lora_alpha (:obj:`int`, `optional`):
            lora alpha.
        lora_r (:obj:`int`, `optional`):
            lora r.
        rdrop_loss_wgt (:obj:`float`, `optional`, defaults to 0):
            rdrop loss weight.
    """
    model_type = 'deberta-v2'

    def __init__(self, vocab_size=128100, hidden_size=1536, num_hidden_layers=24, num_attention_heads=24, intermediate_size=6144, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=0, initializer_range=0.02, layer_norm_eps=1e-07, relative_attention=False, max_relative_positions=-1, pad_token_id=0, position_biased_input=True, pos_att_type=None, pooler_dropout=0, pooler_hidden_act='gelu', cls_dropout=None, apply_lora=False, lora_alpha=None, lora_r=None, reg_loss_wgt=0.0, masking_prob=0.0, cls_token_id=1, sep_token_id=2, unk_token_id=3, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.relative_attention = relative_attention
        self.max_relative_positions = max_relative_positions
        self.pad_token_id = pad_token_id
        self.position_biased_input = position_biased_input
        if type(pos_att_type) == str:
            pos_att_type = [x.strip() for x in pos_att_type.lower().split('|')]
        self.pos_att_type = pos_att_type
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.pooler_hidden_size = kwargs.get('pooler_hidden_size', hidden_size)
        self.pooler_dropout = pooler_dropout
        self.pooler_hidden_act = pooler_hidden_act
        self.cls_dropout = cls_dropout
        self.apply_lora = apply_lora
        self.lora_alpha = lora_alpha
        self.lora_r = lora_r
        self.reg_loss_wgt = reg_loss_wgt
        self.masking_prob = masking_prob
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.unk_token_id = unk_token_id


class DebertaV2PredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class DebertaV2LMPredictionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transform = DebertaV2PredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class DebertaV2OnlyMLMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = DebertaV2LMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([[(pos / np.power(10000, 2 * (j // 2) / dim)) for j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


class Embeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)
        if config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(n_pos=config.max_position_embeddings, dim=config.dim, out=self.position_embeddings.weight)
        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids):
        """
        Parameters:
            input_ids: torch.tensor(bs, max_seq_length) The token ids to embed.

        Returns: torch.tensor(bs, max_seq_length, dim) The embedded tokens (plus position embeddings, no token_type
        embeddings)
        """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = word_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)
        assert self.dim % self.n_heads == 0
        self.q_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.k_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.out_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        attention_head_size = self.dim // self.n_heads
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.n_heads, attention_head_size, self.pruned_heads)
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        self.n_heads = self.n_heads - len(heads)
        self.dim = attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, query, key, value, mask, head_mask=None, output_attentions=False):
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        dim_per_head = self.dim // self.n_heads
        mask_reshp = bs, 1, 1, k_length

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        q = shape(self.q_lin(query))
        k = shape(self.k_lin(key))
        v = shape(self.v_lin(value))
        q = q / math.sqrt(dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))
        mask = (mask == 0).view(mask_reshp).expand_as(scores)
        scores.masked_fill_(mask, -float('inf'))
        weights = nn.Softmax(dim=-1)(scores)
        weights = self.dropout(weights)
        if head_mask is not None:
            weights = weights * head_mask
        context = torch.matmul(weights, v)
        context = unshape(context)
        context = self.out_lin(context)
        if output_attentions:
            return context, weights
        else:
            return context,


class FFN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.lin1 = nn.Linear(in_features=config.dim, out_features=config.hidden_dim)
        self.lin2 = nn.Linear(in_features=config.hidden_dim, out_features=config.dim)
        assert config.activation in ['relu', 'gelu'], "activation ({}) must be in ['relu', 'gelu']".format(config.activation)
        self.activation = gelu if config.activation == 'gelu' else nn.ReLU()

    def forward(self, input):
        return apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, input)

    def ff_chunk(self, input):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.dim % config.n_heads == 0
        self.attention = MultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)
        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

    def forward(self, x, attn_mask=None, head_mask=None, output_attentions=False):
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim)
            attn_mask: torch.tensor(bs, seq_length)

        Returns:
            sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length) The attention weights ffn_output:
            torch.tensor(bs, seq_length, dim) The output of the transformer block contextualization.
        """
        sa_output = self.attention(query=x, key=x, value=x, mask=attn_mask, head_mask=head_mask, output_attentions=output_attentions)
        if output_attentions:
            sa_output, sa_weights = sa_output
        else:
            assert type(sa_output) == tuple
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)
        ffn_output = self.ffn(sa_output)
        ffn_output = self.output_layer_norm(ffn_output + sa_output)
        output = ffn_output,
        if output_attentions:
            output = (sa_weights,) + output
        return output


class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_layers = config.n_layers
        layer = TransformerBlock(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.n_layers)])

    def forward(self, x, attn_mask=None, head_mask=None, output_attentions=False, output_hidden_states=False, return_dict=None):
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim) Input sequence embedded.
            attn_mask: torch.tensor(bs, seq_length) Attention mask on the sequence.

        Returns:
            hidden_state: torch.tensor(bs, seq_length, dim) Sequence of hidden states in the last (top)
            layer all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
                Tuple of length n_layers with the hidden states from each layer.
                Optional: only if output_hidden_states=True
            all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
                Tuple of length n_layers with the attention weights from each layer
                Optional: only if output_attentions=True
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)
            layer_outputs = layer_module(x=hidden_state, attn_mask=attn_mask, head_mask=head_mask[i], output_attentions=output_attentions)
            hidden_state = layer_outputs[-1]
            if output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)
        if not return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_state, hidden_states=all_hidden_states, attentions=all_attentions)


class DistilBertConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.DistilBertModel` or a
    :class:`~transformers.TFDistilBertModel`. It is used to instantiate a DistilBERT model according to the specified
    arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the DistilBERT `distilbert-base-uncased
    <https://huggingface.co/distilbert-base-uncased>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the DistilBERT model. Defines the number of different tokens that can be represented by
            the :obj:`inputs_ids` passed when calling :class:`~transformers.DistilBertModel` or
            :class:`~transformers.TFDistilBertModel`.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        sinusoidal_pos_embds (:obj:`boolean`, `optional`, defaults to :obj:`False`):
            Whether to use sinusoidal positional embeddings.
        n_layers (:obj:`int`, `optional`, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        n_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        dim (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        hidden_dim (:obj:`int`, `optional`, defaults to 3072):
            The size of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        activation (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        qa_dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probabilities used in the question answering model
            :class:`~transformers.DistilBertForQuestionAnswering`.
        seq_classif_dropout (:obj:`float`, `optional`, defaults to 0.2):
            The dropout probabilities used in the sequence classification and the multiple choice model
            :class:`~transformers.DistilBertForSequenceClassification`.

    Examples::

        >>> from transformers import DistilBertModel, DistilBertConfig

        >>> # Initializing a DistilBERT configuration
        >>> configuration = DistilBertConfig()

        >>> # Initializing a model from the configuration
        >>> model = DistilBertModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = 'distilbert'

    def __init__(self, vocab_size=30522, max_position_embeddings=512, sinusoidal_pos_embds=False, n_layers=6, n_heads=12, dim=768, hidden_dim=4 * 768, dropout=0.1, attention_dropout=0.1, activation='gelu', initializer_range=0.02, qa_dropout=0.1, seq_classif_dropout=0.2, pad_token_id=0, **kwargs):
        super().__init__(**kwargs, pad_token_id=pad_token_id)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.sinusoidal_pos_embds = sinusoidal_pos_embds
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation = activation
        self.initializer_range = initializer_range
        self.qa_dropout = qa_dropout
        self.seq_classif_dropout = seq_classif_dropout

    @property
    def hidden_size(self):
        return self.dim

    @property
    def num_attention_heads(self):
        return self.n_heads

    @property
    def num_hidden_layers(self):
        return self.n_layers


DISTILBERT_INPUTS_DOCSTRING = """
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.DistilBertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


DISTILBERT_START_DOCSTRING = """

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.DistilBertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""


DPRReaderOutput = collections.namedtuple('DPRReaderOutput', ['start_logits', 'end_logits', 'relevance_logits'])


class DPRConfig(PretrainedConfig):
    """
    :class:`~transformers.DPRConfig` is the configuration class to store the configuration of a `DPRModel`.

    This is the configuration class to store the configuration of a :class:`~transformers.DPRContextEncoder`,
    :class:`~transformers.DPRQuestionEncoder`, or a :class:`~transformers.DPRReader`. It is used to instantiate the
    components of the DPR model.

    This class is a subclass of :class:`~transformers.BertConfig`. Please check the superclass for the documentation of
    all kwargs.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the DPR model. Defines the different tokens that can be represented by the `inputs_ids`
            passed to the forward method of :class:`~transformers.BertModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the `token_type_ids` passed into :class:`~transformers.BertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        position_embedding_type (:obj:`str`, `optional`, defaults to :obj:`"absolute"`):
            Type of position embedding. Choose one of :obj:`"absolute"`, :obj:`"relative_key"`,
            :obj:`"relative_key_query"`. For positional embeddings use :obj:`"absolute"`. For more information on
            :obj:`"relative_key"`, please refer to `Self-Attention with Relative Position Representations (Shaw et al.)
            <https://arxiv.org/abs/1803.02155>`__. For more information on :obj:`"relative_key_query"`, please refer to
            `Method 4` in `Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)
            <https://arxiv.org/abs/2009.13658>`__.
        projection_dim (:obj:`int`, `optional`, defaults to 0):
            Dimension of the projection for the context and question encoders. If it is set to zero (default), then no
            projection is done.
    """
    model_type = 'dpr'

    def __init__(self, vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0, gradient_checkpointing=False, position_embedding_type='absolute', projection_dim: 'int'=0, **kwargs):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.projection_dim = projection_dim
        self.position_embedding_type = position_embedding_type


DPR_ENCODERS_INPUTS_DOCSTRING = """
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. To match pretraining, DPR input sequence should be
            formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs (for a pair title+text for example):

            ::

                tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
                token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1

            (b) For single sequences (for a question for example):

            ::

                tokens:         [CLS] the dog is hairy . [SEP]
                token_type_ids:   0   0   0   0  0     0   0

            DPR is a model with absolute position embeddings so it's usually advised to pad the inputs on the right
            rather than the left.

            Indices can be obtained using :class:`~transformers.DPRTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


DPR_START_DOCSTRING = """

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.DPRConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""


DPR_READER_INPUTS_DOCSTRING = """
    Args:
        input_ids: (:obj:`Tuple[torch.LongTensor]` of shapes :obj:`(n_passages, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. It has to be a sequence triplet with 1) the question
            and 2) the passages titles and 3) the passages texts To match pretraining, DPR :obj:`input_ids` sequence
            should be formatted with [CLS] and [SEP] with the format:

                ``[CLS] <question token ids> [SEP] <titles ids> [SEP] <texts ids>``

            DPR is a model with absolute position embeddings so it's usually advised to pad the inputs on the right
            rather than the left.

            Indices can be obtained using :class:`~transformers.DPRReaderTokenizer`. See this class documentation for
            more details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(n_passages, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(n_passages, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


class ElectraEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:seq_length + past_key_values_length]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ElectraSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, 'embedding_size'):
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
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


class ElectraSelfOutput(nn.Module):

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


class ElectraAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = ElectraSelfAttention(config)
        self.output = ElectraSelfOutput(config)
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


class ElectraIntermediate(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ElectraOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ElectraLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ElectraAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f'{self} should be used as a decoder model if cross attention is added'
            self.crossattention = ElectraAttention(config)
        self.intermediate = ElectraIntermediate(config)
        self.output = ElectraOutput(config)

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
            assert hasattr(self, 'crossattention'), f'If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`'
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


class ElectraEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ElectraLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, use_cache=None, output_attentions=False, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if getattr(self.config, 'gradient_checkpointing', False) and self.training:
                if use_cache:
                    logger.warn('`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting `use_cache=False`...')
                    use_cache = False

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)
                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(layer_module), hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += layer_outputs[-1],
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=next_decoder_cache, hidden_states=all_hidden_states, attentions=all_self_attentions, cross_attentions=all_cross_attentions)


class ElectraDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        self.config = config

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = get_activation(self.config.hidden_act)(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze(-1)
        return logits


class ElectraGeneratorPredictions(nn.Module):
    """Prediction module for the generator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(config.embedding_size)
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)

    def forward(self, generator_hidden_states):
        hidden_states = self.dense(generator_hidden_states)
        hidden_states = get_activation('gelu')(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class ElectraConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.ElectraModel` or a
    :class:`~transformers.TFElectraModel`. It is used to instantiate a ELECTRA model according to the specified
    arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the ELECTRA `google/electra-small-discriminator
    <https://huggingface.co/google/electra-small-discriminator>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the ELECTRA model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.ElectraModel` or
            :class:`~transformers.TFElectraModel`.
        embedding_size (:obj:`int`, `optional`, defaults to 128):
            Dimensionality of the encoder layers and the pooler layer.
        hidden_size (:obj:`int`, `optional`, defaults to 256):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 4):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.ElectraModel` or
            :class:`~transformers.TFElectraModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        summary_type (:obj:`str`, `optional`, defaults to :obj:`"first"`):
            Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

            Has to be one of the following options:

                - :obj:`"last"`: Take the last token hidden state (like XLNet).
                - :obj:`"first"`: Take the first token hidden state (like BERT).
                - :obj:`"mean"`: Take the mean of all tokens hidden states.
                - :obj:`"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
                - :obj:`"attn"`: Not implemented now, use multi-head attention.
        summary_use_proj (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

            Whether or not to add a projection after the vector extraction.
        summary_activation (:obj:`str`, `optional`):
            Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

            Pass :obj:`"gelu"` for a gelu activation to the output, any other value will result in no activation.
        summary_last_dropout (:obj:`float`, `optional`, defaults to 0.0):
            Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

            The dropout ratio to be used after the projection and activation.
        position_embedding_type (:obj:`str`, `optional`, defaults to :obj:`"absolute"`):
            Type of position embedding. Choose one of :obj:`"absolute"`, :obj:`"relative_key"`,
            :obj:`"relative_key_query"`. For positional embeddings use :obj:`"absolute"`. For more information on
            :obj:`"relative_key"`, please refer to `Self-Attention with Relative Position Representations (Shaw et al.)
            <https://arxiv.org/abs/1803.02155>`__. For more information on :obj:`"relative_key_query"`, please refer to
            `Method 4` in `Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)
            <https://arxiv.org/abs/2009.13658>`__.

    Examples::

        >>> from transformers import ElectraModel, ElectraConfig

        >>> # Initializing a ELECTRA electra-base-uncased style configuration
        >>> configuration = ElectraConfig()

        >>> # Initializing a model from the electra-base-uncased style configuration
        >>> model = ElectraModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = 'electra'

    def __init__(self, vocab_size=30522, embedding_size=128, hidden_size=256, num_hidden_layers=12, num_attention_heads=4, intermediate_size=1024, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, summary_type='first', summary_use_proj=True, summary_activation='gelu', summary_last_dropout=0.1, pad_token_id=0, position_embedding_type='absolute', **kwargs):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_last_dropout = summary_last_dropout
        self.position_embedding_type = position_embedding_type


ELECTRA_INPUTS_DOCSTRING = """
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.ElectraTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


ELECTRA_START_DOCSTRING = """

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.ElectraConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

