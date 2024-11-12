
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


import numpy as np


import torch.nn.functional as F


import math


import torch


import re


from typing import List


from typing import Union


from typing import Optional


from typing import Any


from typing import Dict


from typing import Tuple


import torch.nn as nn


from torch import Tensor


from torch import nn


from torch.nn import Parameter


import torch.optim.lr_scheduler


from itertools import chain


import itertools


from collections import OrderedDict


from typing import Callable


import collections


from collections import Counter


from torch.utils.data import Dataset


from torch.utils.data.dataloader import default_collate


import random


from torch.nn import functional as F


from typing import Iterable


from collections import defaultdict


from collections import deque


import torch as th


from torch.utils.data import DataLoader


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data.sampler import Sampler


import pandas as pd


from torch.optim import Adagrad


from torch.nn.modules.loss import _Loss


from torch.nn.functional import linear


from torch.nn.functional import softmax


from torch.nn.functional import dropout


from torch.nn.functional import pad


from torch.nn.functional import has_torch_function


from torch.nn.functional import handle_torch_function


from torch.nn.functional import _in_projection_packed


import warnings


import torch.distributed as dist


from collections import namedtuple


from typing import NamedTuple


from functools import partial


from torch.testing._internal.common_utils import TestCase


from collections.abc import Iterable


import itertools as it


from itertools import groupby


import matplotlib.pyplot as plt


import time


import torch.hub


import functools


import inspect


import torch.utils.data


from scipy.interpolate import interp1d


import copy


from functools import reduce


from torch.autograd import Variable


from scipy.signal import get_window


from math import sqrt


import torch.distributions as distr


from scipy.io.wavfile import read


import scipy


import torch.multiprocessing as mp


from torch.utils.data import DistributedSampler


from types import SimpleNamespace


from itertools import starmap


from torch.distributions.categorical import Categorical


from enum import Enum


from enum import auto


from torch import autograd


import sklearn


from torch.utils import benchmark


import typing as tp


from abc import ABC


from abc import abstractmethod


from functools import lru_cache


from typing import BinaryIO


import queue


from typing import Iterator


from typing import Sequence


from typing import Mapping


from functools import wraps


import uuid


from numbers import Number


from torch.nn.parallel import DistributedDataParallel


from torch import device as Device


from itertools import repeat


from torch.nn import Conv1d


from torch.nn import ConvTranspose1d


from torch.nn.utils import remove_weight_norm


from torch.nn.utils import weight_norm


import torch.utils.checkpoint as checkpoint


from torch.nn.modules.utils import _single


from torch.autograd import Function


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from inspect import isfunction


from torch.nn.modules.utils import _pair


from torch.nn.modules.conv import _ConvNd


import torch.onnx.operators


from numpy.random import uniform


import torch.optim


from collections.abc import Collection


import types


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


from typing import Set


from itertools import accumulate


from typing import TYPE_CHECKING


import numpy


from torch.utils import cpp_extension


import string


from copy import deepcopy


from inspect import currentframe


from inspect import getframeinfo


from torch.utils.checkpoint import checkpoint


from torch.cuda.amp import GradScaler


from torch.cuda.amp import autocast


class TdnnBNReLU(nn.Module):
    """A block of Tdnn-BatchNorm-ReLU layers."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = dilation * (kernel_size - 1) // 2
        self.tdnn = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=self.padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)

    def output_lengths(self, in_lengths):
        out_lengths = (in_lengths + 2 * self.padding - self.dilation * (self.kernel_size - 1) + self.stride - 1) // self.stride
        return out_lengths

    def forward(self, src, src_lengths):
        x = src.transpose(1, 2).contiguous()
        x = F.relu(self.bn(self.tdnn(x)))
        x = x.transpose(2, 1).contiguous()
        x_lengths = self.output_lengths(src_lengths)
        padding_mask = ~speech_utils.sequence_mask(x_lengths, x.size(1))
        if padding_mask.any():
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x, x_lengths, padding_mask


def LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, export=False):
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        export = True
    if not export and torch.cuda.is_available() and has_fused_layernorm:
        return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


def deprecation_warning(message, stacklevel=3):
    warnings.warn(message, stacklevel=stacklevel)


def gelu(x: 'torch.Tensor') ->torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)


def gelu_accurate(x):
    if not hasattr(gelu_accurate, '_a'):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return 0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))


def relu_squared(x: 'torch.Tensor'):
    return F.relu(x).pow(2)


def get_activation_fn(activation: 'str') ->Callable:
    """Returns the activation function corresponding to `activation`"""
    if activation == 'relu':
        return F.relu
    elif activation == 'relu_squared':
        return relu_squared
    elif activation == 'gelu':
        return gelu
    elif activation == 'gelu_fast':
        deprecation_warning('--activation-fn=gelu_fast has been renamed to gelu_accurate')
        return gelu_accurate
    elif activation == 'gelu_accurate':
        return gelu_accurate
    elif activation == 'tanh':
        return torch.tanh
    elif activation == 'linear':
        return lambda x: x
    elif activation == 'swish':
        return torch.nn.SiLU
    else:
        raise RuntimeError('--activation-fn {} not supported'.format(activation))


class ConvolutionModule(torch.nn.Module):
    """Convolution block used in the conformer block"""

    def __init__(self, embed_dim, channels, depthwise_kernel_size, dropout, activation_fn='swish', bias=False, export=False):
        """
        Args:
            embed_dim: Embedding dimension
            channels: Number of channels in depthwise conv layers
            depthwise_kernel_size: Depthwise conv layer kernel size
            dropout: dropout value
            activation_fn: Activation function to use after depthwise convolution kernel
            bias: If bias should be added to conv layers
            export: If layernorm should be exported to jit
        """
        super(ConvolutionModule, self).__init__()
        assert (depthwise_kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        self.layer_norm = LayerNorm(embed_dim, export=export)
        self.pointwise_conv1 = torch.nn.Conv1d(embed_dim, 2 * channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.glu = torch.nn.GLU(dim=1)
        self.depthwise_conv = torch.nn.Conv1d(channels, channels, depthwise_kernel_size, stride=1, padding=(depthwise_kernel_size - 1) // 2, groups=channels, bias=bias)
        self.batch_norm = torch.nn.BatchNorm1d(channels)
        self.activation = get_activation_fn(activation_fn)(channels)
        self.pointwise_conv2 = torch.nn.Conv1d(channels, embed_dim, kernel_size=1, stride=1, padding=0, bias=bias)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input of shape B X T X C
        Returns:
          Tensor of shape B X T X C
        """
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


logger = logging.getLogger(__name__)


class FairseqDropout(nn.Module):

    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x, inplace: 'bool'=False):
        if self.p > 0 and (self.training or self.apply_during_inference):
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


class FeedForwardModule(torch.nn.Module):
    """Positionwise feed forward layer used in conformer"""

    def __init__(self, input_feat, hidden_units, dropout1, dropout2, activation_fn='swish', bias=True):
        """
        Args:
            input_feat: Input feature dimension
            hidden_units: Hidden unit dimension
            dropout1: dropout value for layer1
            dropout2: dropout value for layer2
            activation_fn: Name of activation function
            bias: If linear layers should have bias
        """
        super(FeedForwardModule, self).__init__()
        self.layer_norm = LayerNorm(input_feat)
        self.w_1 = torch.nn.Linear(input_feat, hidden_units, bias=bias)
        self.w_2 = torch.nn.Linear(hidden_units, input_feat, bias=bias)
        self.dropout1 = torch.nn.Dropout(dropout1)
        self.dropout2 = torch.nn.Dropout(dropout2)
        self.activation = get_activation_fn(activation_fn)(hidden_units)

    def forward(self, x):
        """
        Args:
            x: Input Tensor of shape  T X B X C
        Returns:
            Tensor of shape T X B X C
        """
        x = self.layer_norm(x)
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.w_2(x)
        return self.dropout2(x)


class FairseqDecoder(nn.Module):
    """Base class for decoders."""

    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary
        self.onnx_trace = False
        self.adaptive_softmax = None

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
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

    def get_normalized_probs_scriptable(self, net_output: 'Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]]', log_probs: 'bool', sample: 'Optional[Dict[str, Tensor]]'=None):
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

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade old state dicts to work with newer code."""
        return state_dict

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True


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


def _mask_for_xformers(mask: 'Tensor', to_dtype: 'Optional[torch.dtype]'=None):
    """
    call to pytorch multihead accepts three mask types:
        - ByteTensor where non-zero means to mask
        - FloatTensor which is an additive mask
        - BoolTensor where True means to mask
    xFormers currently accepts boolean and additive maks. For boolean masks
    the values have opposite meaning. For a BoolTensor True mean to keep the value.
    """
    float_types = [torch.float, torch.float16]
    additive = mask.dtype in float_types
    to_dtype = mask.dtype if to_dtype is None else to_dtype
    to_additive = to_dtype in float_types
    if additive:
        if to_additive:
            return mask
        mask = mask < 0
    if to_additive:
        new_mask = torch.zeros_like(mask, dtype=to_dtype)
        new_mask = new_mask.masked_fill_(mask, -float('inf'))
        return new_mask
    mask = ~mask
    mask = mask
    return mask


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


class MultiheadAttention(FairseqIncrementalDecoder):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, self_attention=False, encoder_decoder_attention=False, dictionary=None, relaxed_attention_weight=0.0, q_noise=0.0, qn_block_size=8, xformers_att_config: 'Optional[str]'=None, xformers_blocksparse_layout: 'Optional[torch.Tensor]'=None, xformers_blocksparse_blocksize: 'Optional[int]'=16, positional_embedding=None):
        super().__init__(dictionary)
        xformers_att_config = utils.eval_str_dict(xformers_att_config)
        self.use_xformers = xformers_att_config is not None
        if self.use_xformers and not _xformers_available:
            raise ImportError('\n\n  Please install xFormers.')
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
        self.relaxed_attention_weight = relaxed_attention_weight
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
        self.beam_size = 1
        self.positional_embedding = positional_embedding
        if self.positional_embedding is not None and not self.positional_embedding.learnable:
            assert self.positional_embedding.embedding_dim == embed_dim
            self.pos_bias_u = nn.Parameter(torch.Tensor(embed_dim))
            self.pos_bias_v = nn.Parameter(torch.Tensor(embed_dim))
            self.pos_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=False), q_noise, qn_block_size)
        else:
            if self.positional_embedding is not None:
                assert positional_embedding.embedding_dim == embed_dim or positional_embedding.embedding_dim == self.head_dim
            self.pos_bias_u = self.pos_bias_v = self.pos_proj = None
        self.reset_parameters()
        if self.use_xformers:
            xformers_att_config['dropout'] = xformers_att_config.get('dropout', dropout)
            xformers_att_config['num_heads'] = xformers_att_config.get('num_heads', num_heads)
            if xformers_blocksparse_layout is not None:
                xformers_att_config['block_size'] = xformers_blocksparse_blocksize
                xformers_att_config['layout'] = xformers_blocksparse_layout
                xformers_att_config['name'] = 'blocksparse'
            self.attention = build_attention(xformers_att_config)
        self.onnx_trace = False
        self.skip_embed_dim_check = False
        self.init_incremental_state()

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

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
        if self.pos_bias_u is not None:
            nn.init.xavier_uniform_(self.pos_bias_u.view(self.num_heads, -1))
        if self.pos_bias_v is not None:
            nn.init.xavier_uniform_(self.pos_bias_v.view(self.num_heads, -1))
        if self.pos_proj is not None:
            nn.init.xavier_uniform_(self.pos_proj.weight, gain=1 / math.sqrt(2))

    def _get_reserve_head_index(self, num_heads_to_keep: 'int'):
        k_proj_heads_norm = []
        q_proj_heads_norm = []
        v_proj_heads_norm = []
        for i in range(self.num_heads):
            start_idx = i * self.head_dim
            end_idx = (i + 1) * self.head_dim
            k_proj_heads_norm.append(torch.sum(torch.abs(self.k_proj.weight[start_idx:end_idx,])).tolist() + torch.sum(torch.abs(self.k_proj.bias[start_idx:end_idx])).tolist())
            q_proj_heads_norm.append(torch.sum(torch.abs(self.q_proj.weight[start_idx:end_idx,])).tolist() + torch.sum(torch.abs(self.q_proj.bias[start_idx:end_idx])).tolist())
            v_proj_heads_norm.append(torch.sum(torch.abs(self.v_proj.weight[start_idx:end_idx,])).tolist() + torch.sum(torch.abs(self.v_proj.bias[start_idx:end_idx])).tolist())
        heads_norm = []
        for i in range(self.num_heads):
            heads_norm.append(k_proj_heads_norm[i] + q_proj_heads_norm[i] + v_proj_heads_norm[i])
        sorted_head_index = sorted(range(self.num_heads), key=lambda k: heads_norm[k], reverse=True)
        reserve_head_index = []
        for i in range(num_heads_to_keep):
            start = sorted_head_index[i] * self.head_dim
            end = (sorted_head_index[i] + 1) * self.head_dim
            reserve_head_index.append((start, end))
        return reserve_head_index

    def _adaptive_prune_heads(self, reserve_head_index: 'List[Tuple[int, int]]'):
        new_q_weight = []
        new_q_bias = []
        new_k_weight = []
        new_k_bias = []
        new_v_weight = []
        new_v_bias = []
        new_out_proj_weight = []
        if self.positional_embedding is not None:
            if self.positional_embedding.learnable:
                new_positional_embedding_weight = []
            else:
                new_pos_bias_u = []
                new_pos_bias_v = []
                new_pos_proj_weight = []
        for ele in reserve_head_index:
            start_idx, end_idx = ele
            new_q_weight.append(self.q_proj.weight[start_idx:end_idx,])
            new_q_bias.append(self.q_proj.bias[start_idx:end_idx])
            new_k_weight.append(self.k_proj.weight[start_idx:end_idx,])
            new_k_bias.append(self.k_proj.bias[start_idx:end_idx])
            new_v_weight.append(self.v_proj.weight[start_idx:end_idx,])
            new_v_bias.append(self.v_proj.bias[start_idx:end_idx])
            new_out_proj_weight.append(self.out_proj.weight[:, start_idx:end_idx])
            if self.positional_embedding is not None:
                if self.positional_embedding.learnable:
                    new_positional_embedding_weight.append(self.positional_embedding.weight[:, start_idx:end_idx])
                else:
                    new_pos_bias_u.append(self.pos_bias_u[start_idx:end_idx])
                    new_pos_bias_v.append(self.pos_bias_v[start_idx:end_idx])
                    new_pos_proj_weight.append(self.pos_proj.weight[start_idx:end_idx])
        new_q_weight = torch.cat(new_q_weight).detach()
        new_k_weight = torch.cat(new_k_weight).detach()
        new_v_weight = torch.cat(new_v_weight).detach()
        new_out_proj_weight = torch.cat(new_out_proj_weight, dim=-1).detach()
        new_q_weight.requires_grad = True
        new_k_weight.requires_grad = True
        new_v_weight.requires_grad = True
        new_out_proj_weight.requires_grad = True
        if self.positional_embedding is not None:
            if self.positional_embedding.learnable:
                new_positional_embedding_weight = torch.cat(new_positional_embedding_weight, dim=-1).detach()
                new_positional_embedding_weight.requires_grad = True
            else:
                new_pos_bias_u = torch.cat(new_pos_bias_u).detach()
                new_pos_bias_u.requires_grad = True
                new_pos_bias_v = torch.cat(new_pos_bias_v).detach()
                new_pos_bias_v.requires_grad = True
                new_pos_proj_weight.torch.cat(new_pos_proj_weight).detach()
                new_pos_proj_weight.requires_grad = True
        new_q_bias = torch.cat(new_q_bias).detach()
        new_q_bias.requires_grad = True
        new_k_bias = torch.cat(new_k_bias).detach()
        new_k_bias.requires_grad = True
        new_v_bias = torch.cat(new_v_bias).detach()
        new_v_bias.requires_grad = True
        self.q_proj.weight = torch.nn.Parameter(new_q_weight)
        self.q_proj.bias = torch.nn.Parameter(new_q_bias)
        self.k_proj.weight = torch.nn.Parameter(new_k_weight)
        self.k_proj.bias = torch.nn.Parameter(new_k_bias)
        self.v_proj.weight = torch.nn.Parameter(new_v_weight)
        self.v_proj.bias = torch.nn.Parameter(new_v_bias)
        self.out_proj.weight = torch.nn.Parameter(new_out_proj_weight)
        if self.positional_embedding is not None:
            if self.positional_embedding.learnable:
                self.positional_embedding.weight = torch.nn.Parameter(new_positional_embedding_weight)
            else:
                self.pos_bias_u = torch.nn.Parameter(new_pos_bias_u)
                self.pos_bias_v = torch.nn.Parameter(new_pos_bias_v)
                self.positional_embedding.weight = new_positional_embedding_weight
                self.pos_proj.weight = torch.nn.Parameter(new_pos_proj_weight)
        self.num_heads = len(reserve_head_index)
        self.embed_dim = self.head_dim * self.num_heads
        self.q_proj.out_features = self.embed_dim
        self.k_proj.out_features = self.embed_dim
        self.v_proj.out_features = self.embed_dim
        if self.positional_embedding is not None:
            if self.positional_embedding.learnable:
                self.positional_embedding.embedding_dim = self.embed_dim
            else:
                self.pos_proj.out_features = self.embed_dim

    def _set_skip_embed_dim_check(self):
        self.skip_embed_dim_check = True

    def _pad_masks(self, key_padding_mask: 'Optional[Tensor]', attn_mask: 'Optional[Tensor]') ->Tuple[Optional[Tensor], Optional[Tensor]]:
        if attn_mask is not None:
            shape = attn_mask.size()[:-1] + torch.Size([1])
            attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(shape)], dim=-1)
        if key_padding_mask is not None:
            shape = key_padding_mask.size()[:-1] + torch.Size([1])
            key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_zeros(shape)], dim=-1)
        return key_padding_mask, attn_mask

    def _add_bias(self, k: 'Tensor', v: 'Tensor', key_padding_mask: 'Optional[Tensor]', attn_mask: 'Optional[Tensor]', bsz: 'int') ->Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        assert self.bias_k is not None
        assert self.bias_v is not None
        k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
        key_padding_mask, attn_mask = self._pad_masks(key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return k, v, key_padding_mask, attn_mask

    def _append_zero_attn(self, k: 'Tensor', v: 'Tensor', key_padding_mask: 'Optional[Tensor]', attn_mask: 'Optional[Tensor]') ->Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        zero_attn_shape = k.size()[:-2] + torch.Size([1]) + k.size()[-1:]
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=-2)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=-2)
        key_padding_mask, attn_mask = self._pad_masks(key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return k, v, key_padding_mask, attn_mask

    def _xformers_attn_forward(self, query, key: 'Optional[Tensor]', value: 'Optional[Tensor]', key_padding_mask: 'Optional[Tensor]'=None, need_weights: 'bool'=True, attn_mask: 'Optional[Tensor]'=None) ->Tuple[Tensor, Optional[Tensor]]:
        tgt_len, bsz, embed_dim = query.size()
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == tgt_len
        if self.self_attention:
            key = query
            value = query
        elif self.encoder_decoder_attention:
            value = key
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        if self.bias_k is not None:
            assert self.bias_v is not None
            k, v, attn_mask, key_padding_mask = self._add_bias(k, v, attn_mask, key_padding_mask, bsz)

        def fold_heads(x):
            return x.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        def split_heads(x):
            return x.contiguous().view(-1, bsz, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
        massage = split_heads if self.attention.requires_head_dimension else fold_heads
        q = massage(q)
        if k is not None:
            k = massage(k)
        if v is not None:
            v = massage(v)
        if self.add_zero_attn:
            k, v, key_padding_mask, attn_mask = self._append_zero_attn(k=k, v=v, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        kwargs = {}
        if attn_mask is not None and self.attention.supports_attention_mask:
            attn_mask = _mask_for_xformers(attn_mask, to_dtype=q.dtype)
            kwargs['att_mask'] = attn_mask
        if key_padding_mask is not None:
            key_padding_mask = _mask_for_xformers(key_padding_mask, to_dtype=torch.bool)
            if not self.attention.requires_separate_masks:
                attn_mask = maybe_merge_masks(attn_mask, key_padding_mask, batch_size=bsz, src_len=k.size(-2), tgt_len=q.size(-2), num_heads=self.num_heads)
                key_padding_mask = None
                kwargs['att_mask'] = attn_mask
            if self.attention.supports_key_padding_mask:
                kwargs['key_padding_mask'] = key_padding_mask
        y = self.attention(q, k, v, **kwargs)
        y = y.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).flatten(start_dim=2, end_dim=3).transpose(0, 1)
        assert list(y.size()) == [tgt_len, bsz, embed_dim]
        y = self.out_proj(y)
        return y, None

    def forward(self, query: 'Tensor', key: 'Optional[Tensor]', value: 'Optional[Tensor]', key_padding_mask: 'Optional[Tensor]'=None, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]'=None, need_weights: 'bool'=True, static_kv: 'bool'=False, attn_mask: 'Optional[Tensor]'=None, before_softmax: 'bool'=False, need_head_weights: 'bool'=False) ->Tuple[Tensor, Optional[Tensor]]:
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
        is_tpu = query.device.type == 'xla'
        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        if not self.skip_embed_dim_check:
            assert embed_dim == self.embed_dim, f'query dim {embed_dim} != {self.embed_dim}'
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert value is not None
                assert src_len, key_bsz == value.shape[:2]
        if not self.onnx_trace and not is_tpu and incremental_state is None and not static_kv and not torch.jit.is_scripting() and not self.skip_embed_dim_check and self.positional_embedding is None:
            assert key is not None and value is not None
            if self.use_xformers:
                return self._xformers_attn_forward(query, key, value, key_padding_mask, need_weights, attn_mask)
            else:
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
                if self.beam_size > 1 and bsz == key.size(1):
                    key = key.view(key.size(0), -1, self.beam_size, key.size(2))[:, :, 0, :]
                    if key_padding_mask is not None:
                        key_padding_mask = key_padding_mask.view(-1, self.beam_size, key_padding_mask.size(1))[:, 0, :]
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        if self.positional_embedding is not None:
            if not self.positional_embedding.learnable:
                q_with_bias_v = (q + self.pos_bias_v) * self.scaling
                q_with_bias_v = q_with_bias_v.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
                q = q + self.pos_bias_u
        q *= self.scaling
        if self.bias_k is not None:
            assert self.bias_v is not None
            k, v, attn_mask, key_padding_mask = self._add_bias(k, v, attn_mask, key_padding_mask, bsz)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        kv_bsz = bsz
        if k is not None:
            kv_bsz = k.size(1)
            k = k.contiguous().view(-1, kv_bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, kv_bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if saved_state is not None:
            if 'prev_key' in saved_state:
                _prev_key = saved_state['prev_key']
                assert _prev_key is not None
                kv_bsz = _prev_key.size(0)
                prev_key = _prev_key.view(kv_bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if 'prev_value' in saved_state:
                _prev_value = saved_state['prev_value']
                assert _prev_value is not None
                assert kv_bsz == _prev_value.size(0)
                prev_value = _prev_value.view(kv_bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: 'Optional[Tensor]' = None
            if 'prev_key_padding_mask' in saved_state:
                prev_key_padding_mask = saved_state['prev_key_padding_mask']
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(key_padding_mask=key_padding_mask, prev_key_padding_mask=prev_key_padding_mask, batch_size=kv_bsz, src_len=k.size(1), static_kv=static_kv)
            saved_state['prev_key'] = k.view(kv_bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(kv_bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_key_padding_mask'] = key_padding_mask
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        assert k.size(1) == src_len
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == kv_bsz
            assert key_padding_mask.size(1) == src_len
        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k, v, key_padding_mask, attn_mask = self._append_zero_attn(k=k, v=v, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.encoder_decoder_attention and bsz != kv_bsz:
            attn_weights = torch.einsum('bxhtd,bhsd->bxhts', q.view((kv_bsz, -1, self.num_heads) + q.size()[1:]), k.view((kv_bsz, self.num_heads) + k.size()[1:]))
            attn_weights = attn_weights.reshape((-1,) + attn_weights.size()[-2:])
        else:
            attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)
        if self.positional_embedding is not None:
            assert not self.encoder_decoder_attention, 'positional embedding is only applicable to self attention'
            assert bsz == kv_bsz, f'{bsz} != {kv_bsz}'
            assert src_len >= tgt_len, f'{src_len} vs {tgt_len}'
            if key_padding_mask is not None:
                pe = self.positional_embedding(~key_padding_mask.bool())
            else:
                pe = self.positional_embedding(k.new_ones([bsz, src_len], dtype=torch.bool))
            if not self.positional_embedding.learnable:
                pe = self.pos_proj(pe)
            if pe.size(-1) == self.head_dim:
                pe = pe.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
            pe = pe.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
            pe = pe.reshape(bsz * self.num_heads, -1, self.head_dim)
            positional_logits = torch.bmm(q_with_bias_v if not self.positional_embedding.learnable else q, pe.transpose(1, 2))
            assert list(positional_logits.size()) == [bsz * self.num_heads, tgt_len, 2 * src_len - 1]
            batch_head_stride, tgt_stride, src_stride = positional_logits.stride()
            positional_logits = positional_logits.as_strided((bsz * self.num_heads, tgt_len, src_len), (batch_head_stride, tgt_stride - src_stride, src_stride), storage_offset=src_stride * (tgt_len - 1))
            attn_weights = attn_weights + positional_logits
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.view(kv_bsz, -1, self.num_heads, tgt_len, src_len)
                attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3), float('-inf'))
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if before_softmax:
            return attn_weights, v
        attn_weights_float = utils.softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights = attn_weights_float.type_as(attn_weights)
        if self.training and self.relaxed_attention_weight > 0.0:
            attn_weights = (1.0 - self.relaxed_attention_weight) * attn_weights + self.relaxed_attention_weight / src_len
        attn_probs = self.dropout_module(attn_weights)
        assert v is not None
        attn: 'Optional[Tensor]' = None
        if self.encoder_decoder_attention and bsz != kv_bsz:
            attn = torch.einsum('bxhts,bhsd->bxhtd', attn_probs.view((kv_bsz, -1, self.num_heads) + attn_probs.size()[1:]), v.view((kv_bsz, self.num_heads) + v.size()[1:]))
            attn = attn.reshape((-1,) + attn.size()[-2:])
        else:
            attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            attn = attn.contiguous().view(tgt_len, bsz, self.embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
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
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros((batch_size, src_len - prev_key_padding_mask.size(1)), device=prev_key_padding_mask.device)
                new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), filler.float()], dim=1)
            else:
                new_key_padding_mask = prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros((batch_size, src_len - key_padding_mask.size(1)), device=key_padding_mask.device)
                new_key_padding_mask = torch.cat([filler.float(), key_padding_mask.float()], dim=1)
            else:
                new_key_padding_mask = key_padding_mask.float()
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(self, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]', new_order: 'Tensor'):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention:
                        if input_buffer_k.size(0) * self.beam_size == new_order.size(0):
                            return incremental_state
                        elif self.beam_size > 1:
                            input_buffer[k] = input_buffer_k.index_select(0, new_order.reshape(-1, self.beam_size)[:, 0] // self.beam_size)
                        else:
                            input_buffer[k] = input_buffer_k.index_select(0, new_order)
                    else:
                        input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def set_beam_size(self, beam_size):
        """Used for effiecient beamable enc-dec attention"""
        self.beam_size = beam_size

    def _get_input_buffer(self, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]') ->Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, 'attn_state')
        if result is not None:
            return result
        else:
            empty_result: 'Dict[str, Optional[Tensor]]' = {}
            return empty_result

    def _set_input_buffer(self, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]', buffer: 'Dict[str, Optional[Tensor]]'):
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


class ConformerWithRelativePositionalEmbeddingEncoderLayerBase(nn.Module):
    """Conformer encoder layer block based on https://arxiv.org/abs/2005.08100, with optional relative positional embedding."""

    def __init__(self, cfg, return_fc=False, positional_embedding: 'Optional[RelativePositionalEmbedding]'=None):
        super().__init__()
        self.cfg = cfg
        self.return_fc = return_fc
        self.embed_dim = cfg.encoder.embed_dim
        self.quant_noise = cfg.quant_noise.pq
        self.quant_noise_block_size = cfg.quant_noise.pq_block_size
        self.ffn1 = FeedForwardModule(input_feat=self.embed_dim, hidden_units=cfg.encoder.ffn_embed_dim, dropout1=cfg.activation_dropout, dropout2=cfg.dropout, activation_fn='swish')
        self.self_attn = self.build_self_attention(self.embed_dim, cfg, positional_embedding=positional_embedding)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
        self.dropout_module = FairseqDropout(cfg.dropout, module_name=self.__class__.__name__)
        self.conv_module = ConvolutionModule(embed_dim=self.embed_dim, channels=self.embed_dim, depthwise_kernel_size=cfg.encoder.depthwise_conv_kernel_size, dropout=cfg.dropout, activation_fn='swish')
        self.ffn2 = FeedForwardModule(input_feat=self.embed_dim, hidden_units=cfg.encoder.ffn_embed_dim, dropout1=cfg.activation_dropout, dropout2=cfg.dropout, activation_fn='swish')
        self.final_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

    def build_self_attention(self, embed_dim, cfg, positional_embedding=None):
        return MultiheadAttention(embed_dim, cfg.encoder.attention_heads, dropout=cfg.attention_dropout, self_attention=True, q_noise=self.quant_noise, qn_block_size=self.quant_noise_block_size, xformers_att_config=cfg.encoder.xformers_att_config, positional_embedding=positional_embedding)

    def forward(self, x, encoder_padding_mask: 'Optional[Tensor]', attn_mask: 'Optional[Tensor]'=None):
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
            attn_mask = attn_mask.masked_fill(attn_mask, -100000000.0 if x.dtype == torch.float32 else -10000.0)
        residual = x
        x = self.ffn1(x)
        x = x * 0.5 + residual
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask, need_weights=False, attn_mask=attn_mask)
        x = self.dropout_module(x)
        x = x + residual
        residual = x
        x = x.transpose(0, 1)
        x = self.conv_module(x)
        x = x.transpose(0, 1)
        x = x + residual
        residual = x
        x = self.ffn2(x)
        fc_result = x
        x = x * 0.5 + residual
        x = self.final_layer_norm(x)
        if self.return_fc and not torch.jit.is_scripting():
            return x, fc_result
        return x


DEFAULT_MAX_SOURCE_POSITIONS = 1024


DEFAULT_MAX_TARGET_POSITIONS = 1024


DEFAULT_MIN_PARAMS_TO_WRAP = int(100000000.0)


_NAME_PARSER = '(decoder|encoder|quant_noise)_(.*)'


def safe_getattr(obj, k, default=None):
    """Returns obj[k] if it exists and is not None, otherwise returns default."""
    if OmegaConf.is_config(obj):
        return obj[k] if k in obj and obj[k] is not None else default
    return getattr(obj, k, default)


def safe_hasattr(obj, k):
    """Returns True if the given key exists and is not None."""
    return getattr(obj, k, None) is not None


class ConformerWithRelativePositionalEmbeddingEncoderLayer(ConformerWithRelativePositionalEmbeddingEncoderLayerBase):

    def __init__(self, args, positional_embedding: 'Optional[RelativePositionalEmbedding]'=None):
        super().__init__(TransformerConfig.from_namespace(args), positional_embedding=positional_embedding)
        self.args = args

    def build_self_attention(self, embed_dim, args, positional_embedding=None):
        return super().build_self_attention(embed_dim, TransformerConfig.from_namespace(args), positional_embedding=positional_embedding)


class LearnedRelativePositionalEmbedding(nn.Embedding):
    """
    This module learns relative positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, embedding_dim: 'int', padding_idx: 'Optional[int]'=None, max_size: 'int'=1024):
        num_embeddings = 2 * max_size - 1
        if padding_idx is not None:
            num_embeddings += padding_idx + 1
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.onnx_trace = False
        if padding_idx is not None:
            self.max_positions = self.num_embeddings - padding_idx - 1
        else:
            self.max_positions = self.num_embeddings
        self.max_size = max_size

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0, std=self.embedding_dim ** -0.5)
        if self.padding_idx is not None:
            nn.init.constant_(self.weight[self.padding_idx], 0.0)

    @property
    def learnable(self) ->bool:
        return True

    def forward(self, input: 'Tensor', incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]'=None, positions: 'Optional[Tensor]'=None):
        """
        Args:
            input (torch.Tensor): input tensor of shape `(batch_size, seq_len)`
            incremental_state (dict, optional): dictionary used for storing state during
                :ref:`Incremental decoding` or training a streaming model. No use here.
            positions (torch.Tenser, optional): position ids passed in

        Returns:
            relative posotional embedding for key of shape `(batch_size, 2*seq_len-1, embed_dim)`,
                where `seq_len` is the length of key
        """
        assert positions is None or self.padding_idx is None, 'If positions is pre-computed then padding_idx should not be set.'
        if positions is None:
            bsz, seq_len = input.size(0), input.size(1)
            start = self.max_positions // 2 - seq_len + 1
            end = self.max_positions // 2 + seq_len
            positions = torch.arange(start, end, device=self.weight.device).expand(bsz, -1)
            if seq_len > self.max_size:
                positions = positions.clamp(min=0, max=self.max_positions - 1)
            if self.padding_idx is not None:
                positions = positions + (self.padding_idx + 1)
        return F.embedding(positions, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)


class SinusoidalRelativePositionalEmbedding(nn.Module):
    """This module produces sinusoidal relative positional embeddings of any length."""

    def __init__(self, embedding_dim, padding_idx: 'Optional[int]'=None, init_size=1024, max_size: 'Optional[int]'=None, scale_embedding=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.embedding_scale = embedding_dim ** -0.5 if scale_embedding else 1.0
        self.weight = self.embedding_scale * SinusoidalRelativePositionalEmbedding.get_embedding(init_size, embedding_dim, padding_idx)
        self.onnx_trace = False
        self.register_buffer('_float_tensor', torch.FloatTensor(1))
        self.max_size = max_size

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    @property
    def learnable(self) ->bool:
        return False

    @staticmethod
    def get_embedding(seq_len: 'int', embedding_dim: 'int', padding_idx: 'Optional[int]'=None):
        """Build sinusoidal relative embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".

        Positive when keys are to the right of the query, and negative otherwise.
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(seq_len, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb_pos = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(seq_len, -1)
        emb_neg = torch.cat([torch.sin(-emb), torch.cos(-emb)], dim=1).view(seq_len, -1)
        if embedding_dim % 2 == 1:
            emb_pos = torch.cat([emb_pos, torch.zeros(seq_len, 1)], dim=1)
            emb_neg = torch.cat([emb_neg, torch.zeros(seq_len, 1)], dim=1)
        emb_neg = torch.flip(emb_neg, [0])
        emb_pos = emb_pos[1:]
        emb = torch.cat([emb_neg, emb_pos], dim=0)
        if padding_idx is not None:
            emb = torch.cat([emb.new_zeros([padding_idx + 1, emb.size(1)]), emb], dim=0)
        return emb

    def forward(self, input: 'Tensor', incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]'=None, positions: 'Optional[Tensor]'=None):
        """
        Args:
            input (torch.Tensor): input tensor of shape `(batch_size, seq_len)`
            incremental_state (dict, optional): dictionary used for storing state during
                :ref:`Incremental decoding` or training a streaming model. No use here.
            positions (torch.Tenser, optional): not used in this function

        Returns:
            relative posotional embedding for key of shape `(batch_size, 2*seq_len-1, embed_dim)`,
                where `seq_len` is the length of key
        """
        bspair = torch.onnx.operators.shape_as_tensor(input)
        bsz, seq_len = bspair[0], bspair[1]
        max_positions = self.weight.size(0)
        if self.padding_idx is not None:
            max_positions -= self.padding_idx + 1
        if self.weight is None or 2 * seq_len - 1 > max_positions and (self.max_size is None or seq_len <= self.max_size):
            self.weight = self.embedding_scale * SinusoidalRelativePositionalEmbedding.get_embedding(seq_len, self.embedding_dim, self.padding_idx)
            max_positions = self.weight.size(0)
            if self.padding_idx is not None:
                max_positions -= self.padding_idx + 1
        self.weight = self.weight
        start = max_positions // 2 - seq_len + 1
        end = max_positions // 2 + seq_len
        positions = torch.arange(start, end)
        if self.max_size is not None and seq_len > self.max_size:
            positions = positions.clamp(min=0, max=max_positions - 1)
        if self.padding_idx is not None:
            positions = positions + (self.padding_idx + 1)
        used_weight = self.weight[positions, :]
        if self.onnx_trace:
            return used_weight.unsqueeze(0).repeat(bsz, 1, 1)
        else:
            return used_weight.expand(bsz, -1, -1)


class BaseAttention(nn.Module):
    """Base class for attention layers."""

    def __init__(self, query_dim, value_dim, embed_dim=None):
        super().__init__()
        self.query_dim = query_dim
        self.value_dim = value_dim
        self.embed_dim = embed_dim
        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        pass

    def forward(self, query, value, key_padding_mask=None, state=None):
        raise NotImplementedError


class BahdanauAttention(BaseAttention):
    """Bahdanau Attention."""

    def __init__(self, query_dim, value_dim, embed_dim, normalize=True):
        super().__init__(query_dim, value_dim, embed_dim)
        self.query_proj = nn.Linear(self.query_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(self.value_dim, embed_dim, bias=False)
        self.v = Parameter(torch.Tensor(embed_dim))
        self.normalize = normalize
        if self.normalize:
            self.b = Parameter(torch.Tensor(embed_dim))
            self.g = Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        self.query_proj.weight.data.uniform_(-0.1, 0.1)
        self.value_proj.weight.data.uniform_(-0.1, 0.1)
        nn.init.uniform_(self.v, -0.1, 0.1)
        if self.normalize:
            nn.init.constant_(self.b, 0.0)
            nn.init.constant_(self.g, math.sqrt(1.0 / self.embed_dim))

    def forward(self, query, value, key_padding_mask=None, state=None):
        projected_query = self.query_proj(query).unsqueeze(0)
        key = self.value_proj(value)
        if self.normalize:
            normed_v = self.g * self.v / torch.norm(self.v)
            attn_scores = (normed_v * torch.tanh(projected_query + key + self.b)).sum(dim=2)
        else:
            attn_scores = self.v * torch.tanh(projected_query + key).sum(dim=2)
        if key_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(key_padding_mask, float('-inf')).type_as(attn_scores)
        attn_scores = F.softmax(attn_scores, dim=0)
        context = (attn_scores.unsqueeze(2) * value).sum(dim=0)
        next_state = attn_scores
        return context, attn_scores, next_state


class LuongAttention(BaseAttention):
    """Luong Attention."""

    def __init__(self, query_dim, value_dim, embed_dim=None, scale=True):
        super().__init__(query_dim, value_dim, embed_dim)
        self.value_proj = nn.Linear(self.value_dim, self.query_dim, bias=False)
        self.scale = scale
        if self.scale:
            self.g = Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        self.value_proj.weight.data.uniform_(-0.1, 0.1)
        if self.scale:
            nn.init.constant_(self.g, 1.0)

    def forward(self, query, value, key_padding_mask=None, state=None):
        query = query.unsqueeze(1)
        key = self.value_proj(value).transpose(0, 1)
        attn_scores = torch.bmm(query, key.transpose(1, 2)).squeeze(1)
        attn_scores = attn_scores.transpose(0, 1)
        if self.scale:
            attn_scores = self.g * attn_scores
        if key_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(key_padding_mask, float('-inf')).type_as(attn_scores)
        attn_scores = F.softmax(attn_scores, dim=0)
        context = (attn_scores.unsqueeze(2) * value).sum(dim=0)
        next_state = attn_scores
        return context, attn_scores, next_state


def Convolution2d(in_channels, out_channels, kernel_size, stride):
    if isinstance(kernel_size, (list, tuple)):
        if len(kernel_size) != 2:
            assert len(kernel_size) == 1
            kernel_size = kernel_size[0], kernel_size[0]
    else:
        assert isinstance(kernel_size, int)
        kernel_size = kernel_size, kernel_size
    if isinstance(stride, (list, tuple)):
        if len(stride) != 2:
            assert len(stride) == 1
            stride = stride[0], stride[0]
    else:
        assert isinstance(stride, int)
        stride = stride, stride
    assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
    padding = (kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2
    m = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
    return m


def write_version_py():
    with open(os.path.join('fairseq', 'version.txt')) as f:
        version = f.read().strip()
    with open(os.path.join('fairseq', 'version.py'), 'w') as f:
        f.write('__version__ = "{}"\n'.format(version))
    return version


class ConvBNReLU(nn.Module):
    """Sequence of convolution-[BatchNorm]-ReLU layers.

    Args:
        out_channels (int): the number of output channels of conv layer
        kernel_sizes (int or tuple): kernel sizes
        strides (int or tuple): strides
        in_channels (int, optional): the number of input channels (default: 1)
        apply_batchnorm (bool, optional): if True apply BatchNorm after each convolution layer (default: True)
    """

    def __init__(self, out_channels, kernel_sizes, strides, in_channels=1, apply_batchnorm=True):
        super().__init__()
        if not has_packaging:
            raise ImportError('Please install packaging with: pip install packaging')
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.in_channels = in_channels
        num_layers = len(out_channels)
        assert num_layers == len(kernel_sizes) and num_layers == len(strides)
        self.convolutions = nn.ModuleList()
        self.batchnorms = nn.ModuleList() if apply_batchnorm else None
        for i in range(num_layers):
            self.convolutions.append(Convolution2d(self.in_channels if i == 0 else self.out_channels[i - 1], self.out_channels[i], self.kernel_sizes[i], self.strides[i]))
            if apply_batchnorm:
                self.batchnorms.append(nn.BatchNorm2d(out_channels[i]))

    def output_lengths(self, in_lengths: 'Union[torch.Tensor, int]'):
        out_lengths = in_lengths
        for stride in self.strides:
            if isinstance(stride, (list, tuple)):
                assert len(stride) > 0
                s = stride[0]
            else:
                assert isinstance(stride, int)
                s = stride
            if version.parse(torch.__version__) >= version.parse('1.8.0') and isinstance(in_lengths, torch.Tensor):
                out_lengths = torch.div(out_lengths + s - 1, s, rounding_mode='floor')
            else:
                out_lengths = (out_lengths + s - 1) // s
        return out_lengths

    def forward(self, src, src_lengths):
        x = src.view(src.size(0), src.size(1), self.in_channels, src.size(2) // self.in_channels).transpose(1, 2)
        if self.batchnorms is not None:
            for conv, bn in zip(self.convolutions, self.batchnorms):
                x = F.relu(bn(conv(x)))
        else:
            for conv in self.convolutions:
                x = F.relu(conv(x))
        x = x.transpose(1, 2)
        x = x.contiguous().view(x.size(0), x.size(1), x.size(2) * x.size(3))
        x_lengths = self.output_lengths(src_lengths)
        padding_mask = ~speech_utils.sequence_mask(x_lengths, x.size(1))
        if padding_mask.any():
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x, x_lengths, padding_mask


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

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade old state dicts to work with newer code."""
        return state_dict

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""

        def _apply(m):
            if hasattr(m, 'set_num_updates') and m != self:
                m.set_num_updates(num_updates)
        self.apply(_apply)


class CTCDecoder(FairseqEncoder):

    def __init__(self, dictionary, in_dim):
        super().__init__(dictionary)
        self.proj = nn.Linear(in_dim, len(dictionary))

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        encoder_out = self.proj(src_tokens)
        return {'encoder_out': encoder_out}


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
        return min([m.max_decoder_positions() for m in self.models if hasattr(m, 'max_decoder_positions')] + [sys.maxsize])

    def set_decoder_beam_size(self, beam_size):
        """Set beam size for efficient beamable enc-dec attention."""
        if beam_size > 1:
            for model in self.models:
                if hasattr(model, 'set_beam_size'):
                    model.set_beam_size(beam_size)

    @torch.jit.export
    def forward_encoder(self, net_input: 'Dict[str, Tensor]'):
        if not self.has_encoder():
            return None
        return [model.encoder.forward_torchscript(net_input) for model in self.models]

    @torch.jit.export
    def forward_decoder(self, tokens, encoder_outs: 'List[Dict[str, List[Tensor]]]', incremental_states: 'List[Dict[str, Dict[str, Optional[Tensor]]]]', temperature: 'float'=1.0):
        log_probs = []
        avg_attn: 'Optional[Tensor]' = None
        encoder_out: 'Optional[Dict[str, List[Tensor]]]' = None
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            if self.has_incremental_states():
                decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out, incremental_state=incremental_states[i])
            elif hasattr(model, 'decoder'):
                decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out)
            else:
                decoder_out = model.forward(tokens)
            attn: 'Optional[Tensor]' = None
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
            probs = model.get_normalized_probs(decoder_out_tuple, log_probs=True, sample=None)
            probs = probs[:, -1, :]
            if self.models_size == 1:
                return probs, attn
            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(self.models_size)
        if avg_attn is not None:
            avg_attn.div_(self.models_size)
        return avg_probs, avg_attn

    @torch.jit.export
    def reorder_encoder_out(self, encoder_outs: 'Optional[List[Dict[str, List[Tensor]]]]', new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_outs: 'List[Dict[str, List[Tensor]]]' = []
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


class GenerateLogProbsForDecoding(nn.Module):

    def __init__(self, models, apply_log_softmax=False):
        """Generate the neural network's output intepreted as log probabilities
        for decoding with Kaldi.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            apply_log_softmax (bool, optional): apply log-softmax on top of the
                network's output (default: False)
        """
        super().__init__()
        if isinstance(models, EnsembleModel):
            self.model = models
        else:
            self.model = EnsembleModel(models)
        self.apply_log_softmax = apply_log_softmax
        self.model.eval()

    def cuda(self):
        self.model
        return self

    @torch.no_grad()
    def generate(self, models, sample: 'Dict[str, Dict[str, Tensor]]', **kwargs):
        """Generate a batch of translations.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
        """
        return self._generate(sample, **kwargs)

    def _generate(self, sample: 'Dict[str, Dict[str, Tensor]]', **kwargs):
        net_input = sample['net_input']
        src_tokens = net_input['src_tokens']
        bsz = src_tokens.size(0)
        encoder_outs = self.model.forward_encoder(net_input)
        logits = encoder_outs[0]['encoder_out'][0].transpose(0, 1).float()
        assert logits.size(0) == bsz
        padding_mask = encoder_outs[0]['encoder_padding_mask'][0].t() if len(encoder_outs[0]['encoder_padding_mask']) > 0 else None
        if self.apply_log_softmax:
            return F.log_softmax(logits, dim=-1), padding_mask
        return logits, padding_mask


class SimpleGreedyDecoder(nn.Module):

    def __init__(self, models, dictionary, max_len_a=0, max_len_b=200, max_len=0, temperature=1.0, eos=None, symbols_to_strip_from_output=None, for_validation=True, **kwargs):
        """Decode given speech audios with the simple greedy search.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            dictionary (~fairseq.data.Dictionary): dictionary
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            max_len (int, optional): the maximum length of the generated output
                (not including end-of-sentence)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            for_validation (bool, optional): indicate whether the decoder is
                used for validation. It affects how max_len is determined, and
                whether a tensor of lprobs is returned. If true, target should be
                not None
        """
        super().__init__()
        if isinstance(models, EnsembleModel):
            self.model = models
        else:
            self.model = EnsembleModel(models)
        self.pad = dictionary.pad()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos() if eos is None else eos
        self.symbols_to_strip_from_output = symbols_to_strip_from_output.union({self.eos}) if symbols_to_strip_from_output is not None else {self.eos}
        self.vocab_size = len(dictionary)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.max_len = max_len or self.model.max_decoder_positions()
        self.temperature = temperature
        assert temperature > 0, '--temperature must be greater than 0'
        self.model.eval()
        self.for_validation = for_validation

    def cuda(self):
        self.model
        return self

    @torch.no_grad()
    def decode(self, models, sample: 'Dict[str, Dict[str, Tensor]]', **kwargs):
        """Generate a batch of 1-best hypotheses. Match the API of other fairseq generators.
        Normally called for validation during training.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._decode(sample, **kwargs)

    @torch.no_grad()
    def _decode(self, sample: 'Dict[str, Dict[str, Tensor]]', bos_token: 'Optional[int]'=None):
        incremental_states = torch.jit.annotate(List[Dict[str, Dict[str, Optional[Tensor]]]], [torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}) for i in range(self.model.models_size)])
        net_input = sample['net_input']
        src_tokens = net_input['src_tokens']
        bsz, src_len = src_tokens.size()[:2]
        encoder_outs = self.model.forward_encoder(net_input)
        target = sample['target']
        assert target is not None or not self.for_validation
        max_encoder_output_length = encoder_outs[0]['encoder_out'][0].size(0)
        max_len = max(max_encoder_output_length, target.size(1)) if self.for_validation else min(int(self.max_len_a * src_len + self.max_len_b), self.max_len - 1)
        tokens = src_tokens.new(bsz, max_len + 2).long().fill_(self.pad)
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        lprobs = encoder_outs[0]['encoder_out'][0].new_full((bsz, target.size(1), self.vocab_size), -np.log(self.vocab_size)) if self.for_validation else None
        attn = None
        for step in range(max_len + 1):
            is_eos = tokens[:, step].eq(self.eos)
            if step > 0 and is_eos.sum() == is_eos.size(0):
                tokens = tokens[:, :step + 1]
                if attn is not None:
                    attn = attn[:, :, :step + 1]
                break
            log_probs, avg_attn_scores = self.model.forward_decoder(tokens[:, :step + 1], encoder_outs, incremental_states, temperature=self.temperature)
            tokens[:, step + 1] = log_probs.argmax(-1)
            if step > 0:
                log_probs[is_eos, :] = -np.log(log_probs.size(1))
                tokens[is_eos, step + 1] = self.eos
            if self.for_validation and step < target.size(1):
                lprobs[:, step, :] = log_probs
            if type(avg_attn_scores) is list:
                avg_attn_scores = avg_attn_scores[0]
            if avg_attn_scores is not None:
                if attn is None:
                    attn = avg_attn_scores.new(bsz, max_encoder_output_length, max_len + 2)
                attn[:, :, step + 1].copy_(avg_attn_scores)
        return tokens[:, 1:], lprobs, attn


class TransducerBaseDecoder(nn.Module):

    def __init__(self, models, dictionary, max_len=0, max_num_expansions_per_step=2, temperature=1.0, eos=None, bos=None, blank=None, model_predicts_eos=False, symbols_to_strip_from_output=None, lm_model=None, lm_weight=1.0, print_alignment=False, **kwargs):
        """Decode given speech audios.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            dictionary (~fairseq.data.Dictionary): dictionary
            max_len (int, optional): the maximum length of the encoder output
                that can emit tokens (default: 0, no limit)
            max_num_expansions_per_step (int, optional): the maximum number of
                non-blank expansions in a single time step (default: 2)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            eos (int, optional): index of eos. Will be dictionary.eos() if None
                (default: None)
            bos (int, optional): index of bos. Will be dictionary.eos() if None
                (default: None)
            blank (int, optional): index of blank. Will be dictionary.bos() if
                None (default: None)
            model_predicts_eos(bool, optional): enable it if the transducer model was
                trained to predict EOS. Probability mass of emitting EOS will be transferred
                to BLANK to alleviate early stop issue during decoding (default: False)
            lm_model (fairseq.models.FairseqLanguageModel, optional): LM model for LM fusion (default: None)
            lm_weight (float, optional): LM weight for LM fusion (default: 1.0)
            print_alignment (bool, optional): if True returns alignments (default: False)
        """
        super().__init__()
        self.model = models[0]
        self.eos = dictionary.eos() if eos is None else eos
        self.bos = dictionary.eos() if bos is None else bos
        self.blank = dictionary.bos() if blank is None else blank
        self.model_predicts_eos = model_predicts_eos
        self.symbols_to_strip_from_output = symbols_to_strip_from_output.union({self.eos, self.bos, self.blank}) if symbols_to_strip_from_output is not None else {self.eos, self.bos, self.blank}
        self.vocab_size = len(dictionary)
        self.beam_size = 1
        self.max_len = max_len
        self.max_num_expansions_per_step = max_num_expansions_per_step
        assert max_num_expansions_per_step > 0, '--max-num-expansions-per-step must be at least 1'
        self.temperature = temperature
        assert temperature > 0, '--temperature must be greater than 0'
        self.print_alignment = print_alignment
        self.model.eval()
        self.lm_model = lm_model
        self.lm_weight = lm_weight
        if self.lm_model is not None:
            self.vocab_nonblank_mask = torch.ones((self.vocab_size,), dtype=torch.bool)
            self.vocab_nonblank_mask[self.blank] = False
            if len(self.lm_model.decoder.dictionary) == self.vocab_size - 1:
                self.no_blank_in_lm = True
                logger.info("the LM's vocabulary has 1 less symbol than that of the ASR model. Assuming it is blank symbol.")
            else:
                assert len(self.lm_model.decoder.dictionary) == self.vocab_size
                self.no_blank_in_lm = False
            self.lm_model.eval()

    def cuda(self):
        self.model
        if self.lm_model is not None:
            self.lm_model
        return self

    @torch.no_grad()
    def decode(self, models, sample: 'Dict[str, Dict[str, Tensor]]', **kwargs) ->Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Generate a batch of 1-best hypotheses. Match the API of other fairseq generators.
        Normally called for validation during training.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, **kwargs)

    @torch.no_grad()
    def generate(self, models, sample: 'Dict[str, Dict[str, Tensor]]', **kwargs) ->List[List[Dict[str, Tensor]]]:
        """API to be invoked from :func:`~fairseq.tasks.fairseq_task.FairseqTask.inference_step()`"""
        bos_token = kwargs.get('bos_token', None)
        tokens, scores, alignments = self._generate(sample, bos_token=bos_token)
        bsz = tokens.size(0)
        finalized = torch.jit.annotate(List[List[Dict[str, Tensor]]], [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)])
        for i in range(bsz):
            finalized[i].append({'tokens': tokens[i, :], 'score': scores[i], 'attention': None, 'alignment': alignments[i, :, :] if self.print_alignment and alignments is not None else None})
        return finalized

    @torch.no_grad()
    def _generate(self, sample: 'Dict[str, Dict[str, Tensor]]', bos_token: 'Optional[int]'=None) ->Tuple[Union[Tensor, List[Tensor]], Union[Tensor, List[Tensor]], Optional[Union[Tensor, List[Tensor]]]]:
        """Implement the algorithm here.
        Should return a tuple of tokens, scores and alignments.

        Args:
            feature (Tensor): feature of shape
                `(batch, feature_length, feature_dim)`
            feature_lens (Tensor, optional): feature lengths of shape `(batch)`

        Returns:
            tokens (LongTensor or List[LongTensor]): token sequences of shape
                `(batch, max_dec_out_length)`
            scores (FloatTensor or List[FloatTensor]): scores of shape `(batch)`
            alignments (LongTensor or List[LongTensor], optional): alignments of
                shape `(batch, max_enc_out_length, max_num_tokens_per_step)`
        """
        raise NotImplementedError


def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False, pad_to_length=None, pad_to_multiple=1, pad_to_bsz=None):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
    res = values[0].new(batch_size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)
    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def is_prefix_tensorized(hyps: 'Hypotheses', are_sorted: 'bool'=False):
    """Returns a mask tensor where the (i, j)-th element indicates if the i-th row of `hyps.sequences`
    is a prefix of the j-th row.

    Args:
        hyps (Hypotheses): sequences of tokens, a tensor of shape `(batch, tgt_len)`.
        are_sorted (bool, optional): True if the hypotheses in `hyps` are already sorted by length (in non-increasing order).

    Returns:
        prefix_relations (Tensor): `prefix_relations[i][j]` is True iff `hyps.sequences[i]` is a prefix of `hyps.sequences[j]`,
            a bool tensor of shape `(batch, batch)`.
    """
    bsz = hyps.size()
    assert bsz > 0
    assert hyps.sequences.size(0) == bsz
    lengths = hyps.sequence_lengths
    prefix_relations = hyps.sequences.new_full((bsz, bsz), False, dtype=torch.bool)

    def check_pair(i, j):
        return lengths[i] < lengths[j] and (hyps.sequences[i, :] == hyps.sequences[j, :])[:lengths[i]].all()
    if are_sorted:
        for j in range(bsz - 1):
            for i in range(j + 1, bsz):
                prefix_relations[i, j] = check_pair(i, j)
    else:
        for j in range(bsz):
            for i in range(bsz):
                prefix_relations[i, j] = check_pair(i, j)
    return prefix_relations


def strip_pad(tensor, pad):
    return tensor[tensor.ne(pad)]


def apply_to_sample(f, sample):
    if hasattr(sample, '__len__') and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, collections.OrderedDict):
            od = collections.OrderedDict((key, _apply(value)) for key, value in x.items())
            od.__dict__ = x.__dict__
            return od
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x
    return _apply(sample)


class TransducerGreedyDecoder(TransducerBaseDecoder):

    def __init__(self, models, dictionary, max_len=0, max_num_expansions_per_step=2, temperature=1.0, eos=None, bos=None, blank=None, model_predicts_eos=False, symbols_to_strip_from_output=None, lm_model=None, lm_weight=1.0, print_alignment=False, **kwargs):
        """Decode given speech audios with the simple greedy search.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            dictionary (~fairseq.data.Dictionary): dictionary
            max_len (int, optional): the maximum length of the encoder output
                that can emit tokens (default: 0, no limit)
            max_num_expansions_per_step (int, optional): the maximum number of
                non-blank expansions in a single time step (default: 2)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            eos (int, optional): index of eos. Will be dictionary.eos() if None
                (default: None)
            bos (int, optional): index of bos. Will be dictionary.eos() if None
                (default: None)
            blank (int, optional): index of blank. Will be dictionary.bos() if
                None (default: None)
            model_predicts_eos(bool, optional): enable it if the transducer model was
                trained to predict EOS. Probability mass of emitting EOS will be transferred
                to BLANK to alleviate early stop issue during decoding (default: False)
            lm_model (fairseq.models.FairseqLanguageModel, optional): LM model for LM fusion (default: None)
            lm_weight (float, optional): LM weight for LM fusion (default: 1.0)
            print_alignment (bool, optional): if True returns alignments (default: False)
        """
        super().__init__(models, dictionary, max_len=max_len, max_num_expansions_per_step=max_num_expansions_per_step, temperature=temperature, eos=eos, bos=bos, blank=blank, model_predicts_eos=model_predicts_eos, symbols_to_strip_from_output=symbols_to_strip_from_output, lm_model=lm_model, lm_weight=lm_weight, print_alignment=print_alignment, **kwargs)
        assert hasattr(self.model.decoder, 'masked_copy_cached_state') and callable(self.model.decoder.masked_copy_cached_state), 'self.model.decoder should implement masked_copy_cached_state()'
        assert hasattr(self.model.decoder, 'initialize_cached_state') and callable(self.model.decoder.initialize_cached_state), 'self.model.decoder should implement initialize_cached_state()'
        if self.lm_model is not None:
            assert hasattr(self.lm_model.decoder, 'masked_copy_cached_state') and callable(self.lm_model.decoder.masked_copy_cached_state), 'self.lm_model.decoder should implement masked_copy_cached_state()'
            assert hasattr(self.model.decoder, 'initialize_cached_state') and callable(self.lm_model.decoder.initialize_cached_state), 'self.lm_model.decoder should implement initialize_cached_state()'

    @torch.no_grad()
    def _generate(self, sample: 'Dict[str, Dict[str, Tensor]]', bos_token: 'Optional[int]'=None) ->Tuple[Tensor, Tensor, Optional[Tensor]]:
        net_input = sample['net_input']
        src_tokens = net_input['src_tokens']
        bsz, src_len = src_tokens.size()[:2]
        encoder_outs = self.model.encoder.forward_torchscript(net_input)
        enc_out = encoder_outs['encoder_out'][0].transpose(0, 1)
        enc_out_lengths = encoder_outs['src_lengths'][0]
        max_enc_out_length = enc_out_lengths.max().item()
        max_len = min(max_enc_out_length, self.max_len) if self.max_len > 0 else max_enc_out_length
        tokens = src_tokens.new_full((bsz, max_len, self.max_num_expansions_per_step + 1), self.blank, dtype=torch.long)
        prev_nonblank_tokens = tokens.new_full((bsz, 1), self.bos if bos_token is None else bos_token)
        scores = enc_out.new_full((bsz, max_len, self.max_num_expansions_per_step + 1), 0.0)
        incremental_state = torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}))
        self.model.decoder.initialize_cached_state(tokens, incremental_state=incremental_state)
        if self.lm_model is not None:
            lm_incremental_state = torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}))
            self.lm_model.decoder.initialize_cached_state(tokens, incremental_state=lm_incremental_state)
        for step in range(max_len):
            blank_mask = step >= enc_out_lengths
            expansion_idx = 0
            while not blank_mask.all() and expansion_idx < self.max_num_expansions_per_step + 1:
                old_cached_state = apply_to_sample(torch.clone, self.model.decoder.get_cached_state(incremental_state))
                dec_out = self.model.decoder.extract_features(prev_nonblank_tokens, incremental_state=incremental_state)[0]
                logits = self.model.joint(enc_out[:, step:step + 1, :], dec_out, apply_output_layer=True).squeeze(2).squeeze(1)
                lprobs = self.model.get_normalized_probs((logits.div_(self.temperature), None), log_probs=True)
                if self.lm_model is not None:
                    old_lm_cached_state = apply_to_sample(torch.clone, self.lm_model.decoder.get_cached_state(lm_incremental_state))
                    lm_prev_nonblank_tokens = torch.where(prev_nonblank_tokens > self.blank, prev_nonblank_tokens - 1, prev_nonblank_tokens) if self.no_blank_in_lm else prev_nonblank_tokens
                    lm_out = self.lm_model(lm_prev_nonblank_tokens, incremental_state=lm_incremental_state)
                    lm_lprobs = self.lm_model.get_normalized_probs(lm_out, log_probs=True).squeeze(1)
                    lprobs_no_blank = lprobs[:, self.vocab_nonblank_mask]
                    if not self.no_blank_in_lm:
                        lm_lprobs = lm_lprobs[:, self.vocab_nonblank_mask]
                    lprobs_with_lm_no_blank = lprobs_no_blank + self.lm_weight * lm_lprobs
                    log_scaling_factor = lprobs_no_blank.exp().sum(1).log() - lprobs_with_lm_no_blank.exp().sum(1).log()
                    lprobs_with_lm_no_blank += log_scaling_factor.unsqueeze(1)
                    lprobs[:, self.vocab_nonblank_mask] = lprobs_with_lm_no_blank
                if self.model_predicts_eos:
                    lprobs[:, self.blank] = torch.logaddexp(lprobs[:, self.blank], lprobs[:, self.eos])
                    lprobs[:, self.eos] = float('-inf')
                if expansion_idx < self.max_num_expansions_per_step:
                    scores[:, step, expansion_idx], tokens[:, step, expansion_idx] = lprobs.max(-1)
                    scores[blank_mask, step, expansion_idx] = 0.0
                    blank_mask_local = tokens[:, step, expansion_idx] == self.blank
                    blank_mask.logical_or_(blank_mask_local)
                    tokens[blank_mask, step, expansion_idx] = self.blank
                    prev_nonblank_tokens[~blank_mask, 0] = tokens[~blank_mask, step, expansion_idx]
                else:
                    scores[~blank_mask, step, expansion_idx] = lprobs[~blank_mask, self.blank]
                    blank_mask.fill_(True)
                self.model.decoder.masked_copy_cached_state(incremental_state, old_cached_state, blank_mask)
                if self.lm_model is not None:
                    self.lm_model.decoder.masked_copy_cached_state(lm_incremental_state, old_lm_cached_state, blank_mask)
                expansion_idx += 1
        alignments = tokens if self.print_alignment else None
        return tokens.view(bsz, -1), scores.view(bsz, -1).sum(-1), alignments


class Aligner(object):
    """
    An alignprocessor align video and text and output a dict of tensors (for a model).
    """

    def __init__(self, config):
        """__init__ needs to be light weight for more workers/threads."""
        self.split = config.split
        self.max_video_len = config.max_video_len
        self.max_len = config.max_len
        tokenizer = AutoTokenizer.from_pretrained(str(config.bert_name), use_fast=config.use_fast)
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id

    def __call__(self, video_id, video_feature, text_feature):
        raise NotImplementedError

    def _build_video_seq(self, video_feature, video_clips=None):
        """
        `video_feature`: available video tokens.
        `video_clips`: video clip sequence to build.
        """
        if not isinstance(video_feature, np.ndarray):
            raise ValueError('unsupported type of video_feature', type(video_feature))
        if video_clips is None:
            video_start = 0
            video_end = min(len(video_feature), self.max_video_len)
            video_clips = {'start': [video_start], 'end': [video_end]}
        vfeats = np.zeros((self.max_video_len, video_feature.shape[1]), dtype=np.float32)
        vmasks = torch.zeros((self.max_video_len,), dtype=torch.bool)
        video_len = 0
        for start, end in zip(video_clips['start'], video_clips['end']):
            clip_len = min(self.max_video_len - video_len, end - start)
            if clip_len > 0:
                vfeats[video_len:video_len + clip_len] = video_feature[start:start + clip_len]
                vmasks[video_len:video_len + clip_len] = 1
                video_len += clip_len
        vfeats = torch.from_numpy(vfeats)
        return vfeats, vmasks

    def _build_text_seq(self, text_feature, text_clip_indexs=None):
        """
        `text_feature`: all available clips.
        `text_clip_indexes`: clip sequence to build.
        """
        if text_clip_indexs is None:
            text_clip_indexs = [0]
        full_caps = []
        if isinstance(text_feature, dict):
            for clip_idx in text_clip_indexs:
                full_caps.extend(text_feature['cap'][clip_idx])
        else:
            full_caps = text_feature
        max_text_len = self.max_len - self.max_video_len - 3
        full_caps = full_caps[:max_text_len]
        full_caps = [self.cls_token_id, self.sep_token_id] + full_caps + [self.sep_token_id]
        text_pad_len = self.max_len - len(full_caps) - self.max_video_len
        padded_full_caps = full_caps + [self.pad_token_id] * text_pad_len
        caps = torch.LongTensor(padded_full_caps)
        cmasks = torch.zeros((len(padded_full_caps),), dtype=torch.bool)
        cmasks[:len(full_caps)] = 1
        return caps, cmasks

    def batch_post_processing(self, batch, video_feature):
        return batch


class STConv3D(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size, stride=1, padding=0, separable=False):
        super(STConv3D, self).__init__()
        self.separable = separable
        self.relu = nn.ReLU(inplace=True)
        assert len(kernel_size) == 3
        if separable and kernel_size[0] != 1:
            spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]
            temporal_kernel_size = [kernel_size[0], 1, 1]
            if isinstance(stride, list) and len(stride) == 3:
                spatial_stride = [1, stride[1], stride[2]]
                temporal_stride = [stride[0], 1, 1]
            else:
                spatial_stride = [1, stride, stride]
                temporal_stride = [stride, 1, 1]
            if isinstance(padding, list) and len(padding) == 3:
                spatial_padding = [0, padding[1], padding[2]]
                temporal_padding = [padding[0], 0, 0]
            else:
                spatial_padding = [0, padding, padding]
                temporal_padding = [padding, 0, 0]
        if separable:
            self.conv1 = nn.Conv3d(input_dim, output_dim, kernel_size=spatial_kernel_size, stride=spatial_stride, padding=spatial_padding, bias=False)
            self.bn1 = nn.BatchNorm3d(output_dim)
            self.conv2 = nn.Conv3d(output_dim, output_dim, kernel_size=temporal_kernel_size, stride=temporal_stride, padding=temporal_padding, bias=False)
            self.bn2 = nn.BatchNorm3d(output_dim)
        else:
            self.conv1 = nn.Conv3d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
            self.bn1 = nn.BatchNorm3d(output_dim)

    def forward(self, input):
        out = self.relu(self.bn1(self.conv1(input)))
        if self.separable:
            out = self.relu(self.bn2(self.conv2(out)))
        return out


class SelfGating(nn.Module):

    def __init__(self, input_dim):
        super(SelfGating, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, input_tensor):
        """Feature gating as used in S3D-G.
      """
        spatiotemporal_average = th.mean(input_tensor, dim=[2, 3, 4])
        weights = self.fc(spatiotemporal_average)
        weights = th.sigmoid(weights)
        return weights[:, :, None, None, None] * input_tensor


class InceptionBlock(nn.Module):

    def __init__(self, input_dim, num_outputs_0_0a, num_outputs_1_0a, num_outputs_1_0b, num_outputs_2_0a, num_outputs_2_0b, num_outputs_3_0b, gating=True):
        super(InceptionBlock, self).__init__()
        self.conv_b0 = STConv3D(input_dim, num_outputs_0_0a, [1, 1, 1])
        self.conv_b1_a = STConv3D(input_dim, num_outputs_1_0a, [1, 1, 1])
        self.conv_b1_b = STConv3D(num_outputs_1_0a, num_outputs_1_0b, [3, 3, 3], padding=1, separable=True)
        self.conv_b2_a = STConv3D(input_dim, num_outputs_2_0a, [1, 1, 1])
        self.conv_b2_b = STConv3D(num_outputs_2_0a, num_outputs_2_0b, [3, 3, 3], padding=1, separable=True)
        self.maxpool_b3 = th.nn.MaxPool3d((3, 3, 3), stride=1, padding=1)
        self.conv_b3_b = STConv3D(input_dim, num_outputs_3_0b, [1, 1, 1])
        self.gating = gating
        self.output_dim = num_outputs_0_0a + num_outputs_1_0b + num_outputs_2_0b + num_outputs_3_0b
        if gating:
            self.gating_b0 = SelfGating(num_outputs_0_0a)
            self.gating_b1 = SelfGating(num_outputs_1_0b)
            self.gating_b2 = SelfGating(num_outputs_2_0b)
            self.gating_b3 = SelfGating(num_outputs_3_0b)

    def forward(self, input):
        """Inception block
      """
        b0 = self.conv_b0(input)
        b1 = self.conv_b1_a(input)
        b1 = self.conv_b1_b(b1)
        b2 = self.conv_b2_a(input)
        b2 = self.conv_b2_b(b2)
        b3 = self.maxpool_b3(input)
        b3 = self.conv_b3_b(b3)
        if self.gating:
            b0 = self.gating_b0(b0)
            b1 = self.gating_b1(b1)
            b2 = self.gating_b2(b2)
            b3 = self.gating_b3(b3)
        return th.cat((b0, b1, b2, b3), dim=1)


class MaxPool3dTFPadding(th.nn.Module):

    def __init__(self, kernel_size, stride=None, padding='SAME'):
        super(MaxPool3dTFPadding, self).__init__()
        if padding == 'SAME':
            padding_shape = self._get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.pad = th.nn.ConstantPad3d(padding_shape, 0)
        self.pool = th.nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def _get_padding_shape(self, filter_shape, stride):

        def _pad_top_bottom(filter_dim, stride_val):
            pad_along = max(filter_dim - stride_val, 0)
            pad_top = pad_along // 2
            pad_bottom = pad_along - pad_top
            return pad_top, pad_bottom
        padding_shape = []
        for filter_dim, stride_val in zip(filter_shape, stride):
            pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
            padding_shape.append(pad_top)
            padding_shape.append(pad_bottom)
        depth_top = padding_shape.pop(0)
        depth_bottom = padding_shape.pop(0)
        padding_shape.append(depth_top)
        padding_shape.append(depth_bottom)
        return tuple(padding_shape)

    def forward(self, inp):
        inp = self.pad(inp)
        out = self.pool(inp)
        return out


class Sentence_Embedding(nn.Module):

    def __init__(self, embd_dim, num_embeddings=66250, word_embedding_dim=300, token_to_word_path='dict.npy', max_words=16, output_dim=2048):
        super(Sentence_Embedding, self).__init__()
        self.word_embd = nn.Embedding(num_embeddings, word_embedding_dim)
        self.fc1 = nn.Linear(word_embedding_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, embd_dim)
        self.word_to_token = {}
        self.max_words = max_words
        token_to_word = np.load(token_to_word_path)
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = th.zeros(size - len(tensor)).long()
            return th.cat((tensor, zero), dim=0)

    def _split_text(self, sentence):
        w = re.findall("[\\w']+", str(sentence))
        return w

    def _words_to_token(self, words):
        words = [self.word_to_token[word] for word in words if word in self.word_to_token]
        if words:
            we = self._zero_pad_tensor_token(th.LongTensor(words), self.max_words)
            return we
        else:
            return th.zeros(self.max_words).long()

    def _words_to_ids(self, x):
        split_x = [self._words_to_token(self._split_text(sent.lower())) for sent in x]
        return th.stack(split_x, dim=0)

    def forward(self, x):
        x = self._words_to_ids(x)
        x = self.word_embd(x)
        x = F.relu(self.fc1(x))
        x = th.max(x, dim=1)[0]
        x = self.fc2(x)
        return {'text_embedding': x}


class S3D(nn.Module):

    def __init__(self, dict_path, num_classes=512, gating=True, space_to_depth=True):
        super(S3D, self).__init__()
        self.num_classes = num_classes
        self.gating = gating
        self.space_to_depth = space_to_depth
        if space_to_depth:
            self.conv1 = STConv3D(24, 64, [2, 4, 4], stride=1, padding=(1, 2, 2), separable=False)
        else:
            self.conv1 = STConv3D(3, 64, [3, 7, 7], stride=2, padding=(1, 3, 3), separable=False)
        self.conv_2b = STConv3D(64, 64, [1, 1, 1], separable=False)
        self.conv_2c = STConv3D(64, 192, [3, 3, 3], padding=1, separable=True)
        self.gating = SelfGating(192)
        self.maxpool_2a = MaxPool3dTFPadding(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')
        self.maxpool_3a = MaxPool3dTFPadding(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')
        self.mixed_3b = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.mixed_3c = InceptionBlock(self.mixed_3b.output_dim, 128, 128, 192, 32, 96, 64)
        self.maxpool_4a = MaxPool3dTFPadding(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME')
        self.mixed_4b = InceptionBlock(self.mixed_3c.output_dim, 192, 96, 208, 16, 48, 64)
        self.mixed_4c = InceptionBlock(self.mixed_4b.output_dim, 160, 112, 224, 24, 64, 64)
        self.mixed_4d = InceptionBlock(self.mixed_4c.output_dim, 128, 128, 256, 24, 64, 64)
        self.mixed_4e = InceptionBlock(self.mixed_4d.output_dim, 112, 144, 288, 32, 64, 64)
        self.mixed_4f = InceptionBlock(self.mixed_4e.output_dim, 256, 160, 320, 32, 128, 128)
        self.maxpool_5a = self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding='SAME')
        self.mixed_5b = InceptionBlock(self.mixed_4f.output_dim, 256, 160, 320, 32, 128, 128)
        self.mixed_5c = InceptionBlock(self.mixed_5b.output_dim, 384, 192, 384, 48, 128, 128)
        self.fc = nn.Linear(self.mixed_5c.output_dim, num_classes)
        self.text_module = Sentence_Embedding(num_classes, token_to_word_path=dict_path)

    def _space_to_depth(self, input):
        """3D space to depth trick for TPU optimization.
      """
        B, C, T, H, W = input.shape
        input = input.view(B, C, T // 2, 2, H // 2, 2, W // 2, 2)
        input = input.permute(0, 3, 5, 7, 1, 2, 4, 6)
        input = input.contiguous().view(B, 8 * C, T // 2, H // 2, W // 2)
        return input

    def forward(self, inputs):
        """Defines the S3DG base architecture."""
        if self.space_to_depth:
            inputs = self._space_to_depth(inputs)
        net = self.conv1(inputs)
        if self.space_to_depth:
            net = net[:, :, 1:, 1:, 1:]
        net = self.maxpool_2a(net)
        net = self.conv_2b(net)
        net = self.conv_2c(net)
        if self.gating:
            net = self.gating(net)
        net = self.maxpool_3a(net)
        net = self.mixed_3b(net)
        net = self.mixed_3c(net)
        net = self.maxpool_4a(net)
        net = self.mixed_4b(net)
        net = self.mixed_4c(net)
        net = self.mixed_4d(net)
        net = self.mixed_4e(net)
        net = self.mixed_4f(net)
        net = self.maxpool_5a(net)
        net = self.mixed_5b(net)
        net = self.mixed_5c(net)
        net = th.mean(net, dim=[2, 3, 4])
        return {'video_embedding': self.fc(net), 'mixed_5c': net}


def set_seed(seed=43211):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class MMDataset(Dataset):
    """
    A generic multi-modal dataset.
        Args:
            `meta_processor`: a meta processor,
                handling loading meta data and return video_id and text_id.
            `video_processor`: a video processor,
                handling e.g., decoding, loading .np files.
            `text_processor`: a text processor,
                handling e.g., tokenization.
            `aligner`: combine the video and text feature
                as one training example.
    """

    def __init__(self, meta_processor, video_processor, text_processor, align_processor):
        self.split = meta_processor.split
        self.meta_processor = meta_processor
        self.video_processor = video_processor
        self.text_processor = text_processor
        self.align_processor = align_processor

    def __len__(self):
        return len(self.meta_processor)

    def __getitem__(self, idx):
        if self.split == 'test':
            set_seed(idx)
        video_id, text_id = self.meta_processor[idx]
        video_feature = self.video_processor(video_id)
        text_feature = self.text_processor(text_id)
        output = self.align_processor(video_id, video_feature, text_feature)
        output.update({'idx': idx})
        return output

    def collater(self, samples):
        """This collator is deprecated.
        set self.collator = MMDataset.collater.
        see collator in FairseqMMDataset.
        """
        if len(samples) == 0:
            return {}
        if isinstance(samples[0], dict):
            batch = OrderedDict()
            for key in samples[0]:
                if samples[0][key] is not None:
                    batch[key] = default_collate([sample[key] for sample in samples])
            return batch
        else:
            return default_collate(samples)

    def print_example(self, output):
        None
        if hasattr(self.align_processor, 'subsampling') and self.align_processor.subsampling is not None and self.align_processor.subsampling > 1:
            for key in output:
                if torch.is_tensor(output[key]):
                    output[key] = output[key][0]
        tokenizer = None
        if hasattr(self.text_processor, 'tokenizer'):
            tokenizer = self.text_processor.tokenizer
        elif hasattr(self.align_processor, 'tokenizer'):
            tokenizer = self.align_processor.tokenizer
        if tokenizer is not None:
            caps = output['caps'].tolist()
            if isinstance(caps[0], list):
                caps = caps[0]
            None
            None
        for key, value in output.items():
            if torch.is_tensor(value):
                if len(value.size()) >= 3:
                    None
                    None
                    None
                else:
                    None
        None


class Task(object):
    """
    A task refers to one generic training task (e.g., training one model).
    """

    @classmethod
    def config_task(cls, config):
        """
        determine whether to load a hard-coded task or config from a generic one.
        via if a task string is available in config.
        """
        if config.task is not None:
            task_cls = getattr(tasks, config.task)
            return task_cls(config)
        else:
            return Task(config)

    def __init__(self, config):
        self.config = config
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.model = None
        self.loss_fn = None
        self.eval_fn = None

    def build_dataset(self):
        """TODO (huxu): move processor breakdown to MMDataset."""
        """fill-in `self.train_data`, `self.val_data` and `self.test_data`."""
        meta_processor_cls = getattr(processors, self.config.dataset.meta_processor)
        video_processor_cls = getattr(processors, self.config.dataset.video_processor)
        text_processor_cls = getattr(processors, self.config.dataset.text_processor)
        aligner_cls = getattr(processors, self.config.dataset.aligner)
        if self.config.dataset.train_path is not None:
            self.config.dataset.split = 'train'
            meta_processor = meta_processor_cls(self.config.dataset)
            video_processor = video_processor_cls(self.config.dataset)
            text_processor = text_processor_cls(self.config.dataset)
            aligner = aligner_cls(self.config.dataset)
            self.train_data = MMDataset(meta_processor, video_processor, text_processor, aligner)
            None
            output = self.train_data[0]
            self.train_data.print_example(output)
        if self.config.dataset.val_path is not None:
            self.config.dataset.split = 'valid'
            meta_processor = meta_processor_cls(self.config.dataset)
            video_processor = video_processor_cls(self.config.dataset)
            text_processor = text_processor_cls(self.config.dataset)
            aligner = aligner_cls(self.config.dataset)
            self.val_data = MMDataset(meta_processor, video_processor, text_processor, aligner)
            None
            output = self.val_data[0]
            self.val_data.print_example(output)
        if self.config.dataset.split == 'test':
            meta_processor = meta_processor_cls(self.config.dataset)
            video_processor = video_processor_cls(self.config.dataset)
            text_processor = text_processor_cls(self.config.dataset)
            self.test_data = MMDataset(meta_processor, video_processor, text_processor, aligner)
            None
            output = self.test_data[0]
            self.test_data.print_example(output)

    def build_model(self, checkpoint=None):
        if self.model is None:
            model_cls = getattr(models, self.config.model.model_cls)
            self.model = model_cls(self.config)
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)
        return self.model

    def load_checkpoint(self, checkpoint):
        if self.model is None:
            raise ValueError('model is not initialized.')
        state_dict = torch.load(checkpoint)
        state_dict = self._trim_state_dict(state_dict)
        self.model.load_state_dict(state_dict, strict=False)
        if next(self.model.parameters()).dtype == torch.float16:
            self.model = self.model.float()
        return self.model

    def _trim_state_dict(self, state_dict):
        from collections import OrderedDict
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        if 'model' in state_dict:
            state_dict = state_dict['model']
        ret_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith('mmmodel'):
                key = key[len('mmmodel.'):]
            ret_state_dict[key] = value
        return ret_state_dict

    def build_loss(self):
        if self.loss_fn is None and self.config.loss is not None:
            loss_cls = getattr(losses, self.config.loss.loss_cls)
            self.loss_fn = loss_cls()
        return self.loss_fn

    def flat_subsample(self, tensor):
        size = tensor.size()
        if len(size) >= 2:
            batch_size = size[0] * size[1]
            expanded_size = (batch_size,) + size[2:] if len(size) > 2 else (batch_size,)
            tensor = tensor.view(expanded_size)
        return tensor

    def reshape_subsample(self, sample):
        if hasattr(self.config.dataset, 'subsampling') and self.config.dataset.subsampling is not None and self.config.dataset.subsampling > 1:
            for key in sample:
                if torch.is_tensor(sample[key]):
                    sample[key] = self.flat_subsample(sample[key])
        return sample

    def __call__(self, model, sample):
        loss = None
        loss_scalar = float('inf')
        sample = self.reshape_subsample(sample)
        outputs = self.model(**sample)
        sample.update(outputs)
        if self.loss_fn is not None:
            loss = self.loss_fn(**sample)
            loss_scalar = loss.item()
        batch_size = sample['caps'].size(0)
        sample_size = 1
        return {'loss': loss, 'loss_scalar': loss_scalar, 'max_len': self.config.dataset.max_len, 'batch_size': batch_size, 'sample_size': sample_size}

    def build_dataloader(self):
        """only used for trainer that lacks building loaders."""
        raise NotImplementedError


def recursive_config(config_path):
    """allows for stacking of configs in any depth."""
    config = OmegaConf.load(config_path)
    if config.includes is not None:
        includes = config.includes
        config.pop('includes')
        base_config = recursive_config(includes)
        config = OmegaConf.merge(base_config, config)
    return config


class MMPTModel(nn.Module):
    """An e2e wrapper of inference model.
    """

    @classmethod
    def from_pretrained(cls, config, checkpoint='checkpoint_best.pt'):
        config = recursive_config(config)
        mmtask = Task.config_task(config)
        checkpoint_path = os.path.join(config.eval.save_path, checkpoint)
        mmtask.build_model(checkpoint=checkpoint_path)
        video_encoder = S3D('pretrained_models/s3d_dict.npy', 512)
        video_encoder.load_state_dict(torch.load('pretrained_models/s3d_howto100m.pth'))
        tokenizer = AutoTokenizer.from_pretrained(config.dataset.bert_name, use_fast=config.dataset.use_fast)
        aligner = Aligner(config.dataset)
        return MMPTModel(config, mmtask.model, video_encoder), tokenizer, aligner

    def __init__(self, config, model, video_encoder, **kwargs):
        super().__init__()
        self.max_video_len = config.dataset.max_video_len
        self.video_encoder = video_encoder
        self.model = model

    def forward(self, video_frames, caps, cmasks, return_score=False):
        bsz = video_frames.size(0)
        assert bsz == 1, 'only bsz=1 is supported now.'
        seq_len = video_frames.size(1)
        video_frames = video_frames.view(-1, *video_frames.size()[2:])
        vfeats = self.video_encoder(video_frames.permute(0, 4, 1, 2, 3))
        vfeats = vfeats['video_embedding']
        vfeats = vfeats.view(bsz, seq_len, vfeats.size(-1))
        padding = torch.zeros(bsz, self.max_video_len - seq_len, vfeats.size(-1))
        vfeats = torch.cat([vfeats, padding], dim=1)
        vmasks = torch.cat([torch.ones((bsz, seq_len), dtype=torch.bool), torch.zeros((bsz, self.max_video_len - seq_len), dtype=torch.bool)], dim=1)
        output = self.model(caps, cmasks, vfeats, vmasks)
        if return_score:
            output = {'score': torch.bmm(output['pooled_video'][:, None, :], output['pooled_text'][:, :, None]).squeeze(-1).squeeze(-1)}
        return output


class MMFusion(nn.Module):
    """a MMPT wrapper class for MMBert style models.
    TODO: move isolated mask to a subclass.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        transformer_config = AutoConfig.from_pretrained(config.dataset.bert_name)
        self.hidden_size = transformer_config.hidden_size
        self.is_train = False
        if config.dataset.train_path is not None:
            self.is_train = True
        self.num_hidden_layers = transformer_config.num_hidden_layers
        self.last_iso_layer = 0
        if config.dataset.num_iso_layer is not None:
            self.last_iso_layer = config.dataset.num_iso_layer - 1 + 1
        if config.model.mm_encoder_cls is not None:
            mm_encoder_cls = getattr(transformermodel, config.model.mm_encoder_cls)
            model_config = AutoConfig.from_pretrained(config.dataset.bert_name)
            model_config.max_video_len = config.dataset.max_video_len
            model_config.use_seg_emb = config.model.use_seg_emb
            self.mm_encoder = mm_encoder_cls.from_pretrained(config.dataset.bert_name, config=model_config)
        elif config.model.video_encoder_cls is not None and config.model.text_encoder_cls is not None:
            video_encoder_cls = getattr(transformermodel, config.model.video_encoder_cls)
            model_config = AutoConfig.from_pretrained(config.dataset.bert_name)
            model_config.max_video_len = config.dataset.max_video_len
            if hasattr(model_config, 'num_layers'):
                model_config.num_layers = config.model.num_hidden_video_layers
            else:
                model_config.num_hidden_layers = config.model.num_hidden_video_layers
            self.video_encoder = video_encoder_cls.from_pretrained(config.dataset.bert_name, config=model_config)
            text_encoder_cls = getattr(transformermodel, config.model.text_encoder_cls)
            self.text_encoder = text_encoder_cls.from_pretrained(config.dataset.bert_name)
        else:
            raise ValueError('the encoder must be either MM or two backbones.')

    def forward(self, caps, cmasks, vfeats, vmasks, **kwargs):
        raise NotImplementedError('Please derive MMFusion module.')

    def _mm_on_the_fly(self, cmasks, vmasks, attention_mask):
        """helper function for mask, seg_ids and token_type_ids."""
        if attention_mask is None:
            attention_mask = self._mm_attention_mask(cmasks, vmasks)
        """
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        """
        token_type_ids = torch.cat([torch.zeros((vmasks.size(0), vmasks.size(1) + 2), dtype=torch.long, device=vmasks.device), torch.ones((cmasks.size(0), cmasks.size(1) - 2), dtype=torch.long, device=cmasks.device)], dim=1)
        return attention_mask, token_type_ids

    def _mm_attention_mask(self, cmasks, vmasks):
        assert cmasks.size(0) == vmasks.size(0), '{}, {}, {}, {}'.format(str(cmasks.size()), str(vmasks.size()), str(cmasks.size(0)), str(vmasks.size(0)))
        mm_mask = torch.cat([cmasks[:, :1], vmasks, cmasks[:, 1:]], dim=1)
        if self.last_iso_layer == 0:
            return mm_mask
        else:
            batch_size = cmasks.size(0)
            iso_mask = self._make_iso_mask(batch_size, cmasks, vmasks)
            mm_mask = mm_mask[:, None, :].repeat(1, mm_mask.size(-1), 1)
            iso_mm_masks = []
            iso_mask = iso_mask[:, None, :, :].repeat(1, self.last_iso_layer, 1, 1)
            iso_mm_masks.append(iso_mask)
            if self.last_iso_layer < self.num_hidden_layers:
                mm_mask = mm_mask[:, None, :, :].repeat(1, self.num_hidden_layers - self.last_iso_layer, 1, 1)
                iso_mm_masks.append(mm_mask)
            iso_mm_masks = torch.cat(iso_mm_masks, dim=1)
            return iso_mm_masks

    def _make_iso_mask(self, batch_size, cmasks, vmasks):
        cls_self_mask = torch.cat([torch.ones((batch_size, 1), dtype=torch.bool, device=cmasks.device), torch.zeros((batch_size, cmasks.size(1) + vmasks.size(1) - 1), dtype=torch.bool, device=cmasks.device)], dim=1)
        iso_video_mask = torch.cat([torch.zeros((batch_size, 1), dtype=torch.bool, device=cmasks.device), vmasks, cmasks[:, 1:2], torch.zeros((batch_size, cmasks.size(1) - 2), dtype=torch.bool, device=cmasks.device)], dim=1)
        iso_text_mask = torch.cat([torch.zeros((batch_size, 2 + vmasks.size(1)), dtype=torch.bool, device=cmasks.device), cmasks[:, 2:]], dim=1)
        cls_self_mask = cls_self_mask[:, None, :]
        iso_video_mask = iso_video_mask[:, None, :].repeat(1, vmasks.size(1) + 1, 1)
        iso_text_mask = iso_text_mask[:, None, :].repeat(1, cmasks.size(1) - 2, 1)
        return torch.cat([cls_self_mask, iso_video_mask, iso_text_mask], dim=1)

    def _pooling_vt_layer(self, layered_sequence_output, cmasks, vmasks):
        layer_idx = self.last_iso_layer if self.last_iso_layer > 0 else self.num_hidden_layers
        hidden_state = layered_sequence_output[layer_idx]
        batch_size = cmasks.size(0)
        text_offset = vmasks.size(1) + 2
        video_outputs = hidden_state[:, 1:text_offset]
        video_attention_mask = torch.cat([vmasks, torch.ones((batch_size, 1), dtype=torch.bool, device=vmasks.device)], dim=1)
        assert video_outputs.size(1) == video_attention_mask.size(1)
        pooled_video = torch.sum(video_outputs * video_attention_mask.unsqueeze(-1), dim=1) / video_attention_mask.sum(1, keepdim=True)
        text_attention_mask = cmasks[:, 2:]
        text_outputs = hidden_state[:, text_offset:]
        assert text_outputs.size(1) == text_attention_mask.size(1)
        pooled_text = torch.sum(text_outputs * text_attention_mask.unsqueeze(-1), dim=1) / text_attention_mask.sum(1, keepdim=True)
        return pooled_video, pooled_text


class MMFusionMFMMLM(MMFusion):
    """forward function for MFM and MLM."""

    def forward(self, caps, cmasks, vfeats, vmasks, attention_mask=None, video_label=None, text_label=None, **kwargs):
        output_hidden_states = False if self.is_train else True
        target_vfeats, non_masked_frame_mask = None, None
        if video_label is not None:
            target_vfeats = vfeats.masked_select(video_label.unsqueeze(-1)).view(-1, vfeats.size(-1))
            vfeats[video_label] = 0.0
            non_masked_frame_mask = vmasks.clone()
            non_masked_frame_mask[video_label] = False
        attention_mask, token_type_ids = self._mm_on_the_fly(cmasks, vmasks, attention_mask)
        outputs = self.mm_encoder(input_ids=caps, input_video_embeds=vfeats, attention_mask=attention_mask, token_type_ids=token_type_ids, masked_frame_labels=video_label, target_video_hidden_states=target_vfeats, non_masked_frame_mask=non_masked_frame_mask, masked_lm_labels=text_label, output_hidden_states=output_hidden_states)
        video_logits, text_logits = outputs[0], outputs[1]
        if self.is_train:
            return {'video_logits': video_logits, 'text_logits': text_logits}
        pooled_video, pooled_text = self._pooling_vt_layer(outputs[2], cmasks, vmasks)
        return {'pooled_video': pooled_video, 'pooled_text': pooled_text}


class BertMFMMLMPredictionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, video_hidden_states=None, target_video_hidden_states=None, non_masked_frame_hidden_states=None, text_hidden_states=None):
        video_logits, text_logits = None, None
        if video_hidden_states is not None:
            video_hidden_states = self.transform(video_hidden_states)
            non_masked_frame_logits = torch.mm(video_hidden_states, non_masked_frame_hidden_states.transpose(1, 0))
            masked_frame_logits = torch.bmm(video_hidden_states.unsqueeze(1), target_video_hidden_states.unsqueeze(-1)).squeeze(-1)
            video_logits = torch.cat([masked_frame_logits, non_masked_frame_logits], dim=1)
        if text_hidden_states is not None:
            text_hidden_states = self.transform(text_hidden_states)
            text_logits = self.decoder(text_hidden_states)
        return video_logits, text_logits


class MFMMLMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = BertMFMMLMPredictionHead(config)

    def forward(self, video_hidden_states=None, target_video_hidden_states=None, non_masked_frame_hidden_states=None, text_hidden_states=None):
        video_logits, text_logits = self.predictions(video_hidden_states, target_video_hidden_states, non_masked_frame_hidden_states, text_hidden_states)
        return video_logits, text_logits


class VideoTokenMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        input_dim = config.input_dim if hasattr(config, 'input_dim') else 512
        self.linear1 = nn.Linear(input_dim, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class BertMTMPredictionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, video_hidden_states=None, target_video_hidden_states=None, non_masked_frame_hidden_states=None, text_hidden_states=None):
        non_masked_frame_hidden_states = non_masked_frame_hidden_states.transpose(1, 0)
        video_logits, text_logits = None, None
        if video_hidden_states is not None:
            video_hidden_states = self.transform(video_hidden_states)
            masked_frame_logits = torch.bmm(video_hidden_states.unsqueeze(1), target_video_hidden_states.unsqueeze(-1)).squeeze(-1)
            non_masked_frame_logits = torch.mm(video_hidden_states, non_masked_frame_hidden_states)
            video_on_vocab_logits = self.decoder(video_hidden_states)
            video_logits = torch.cat([masked_frame_logits, non_masked_frame_logits, video_on_vocab_logits], dim=1)
        if text_hidden_states is not None:
            text_hidden_states = self.transform(text_hidden_states)
            text_on_vocab_logits = self.decoder(text_hidden_states)
            text_on_video_logits = torch.mm(text_hidden_states, non_masked_frame_hidden_states)
            text_logits = torch.cat([text_on_vocab_logits, text_on_video_logits], dim=1)
        return video_logits, text_logits


class MTMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = BertMTMPredictionHead(config)

    def forward(self, video_hidden_states=None, target_video_hidden_states=None, non_masked_frame_hidden_states=None, text_hidden_states=None):
        video_logits, text_logits = self.predictions(video_hidden_states, target_video_hidden_states, non_masked_frame_hidden_states, text_hidden_states)
        return video_logits, text_logits


class MMFusionMTM(MMFusionMFMMLM):

    def __init__(self, config, **kwargs):
        super().__init__(config)
        """
        For reproducibility:
        self.mm_encoder will be initialized then discarded.
        """
        model_config = AutoConfig.from_pretrained(config.dataset.bert_name)
        model_config.max_video_len = config.dataset.max_video_len
        model_config.use_seg_emb = config.model.use_seg_emb
        self.mm_encoder = MMBertForMTM.from_pretrained(config.dataset.bert_name, config=model_config)


class MMFusionShare(MMFusion):
    """A retrival wrapper using mm_encoder as both video/text backbone.
    TODO: move formally.
    """

    def forward(self, caps, cmasks, vfeats, vmasks, attention_mask=None, video_label=None, text_label=None, output_hidden_states=False, **kwargs):
        pooled_video = self.forward_video(vfeats, vmasks, caps, cmasks, output_hidden_states)
        pooled_text = self.forward_text(caps, cmasks, output_hidden_states)
        return {'pooled_video': pooled_video, 'pooled_text': pooled_text}

    def forward_video(self, vfeats, vmasks, caps, cmasks, output_hidden_states=False, **kwargs):
        input_ids = caps[:, :2]
        attention_mask = torch.cat([cmasks[:, :1], vmasks, cmasks[:, 1:2]], dim=1)
        token_type_ids = torch.zeros((vmasks.size(0), vmasks.size(1) + 2), dtype=torch.long, device=vmasks.device)
        outputs = self.mm_encoder(input_ids=input_ids, input_video_embeds=vfeats, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        video_outputs = outputs[0]
        if output_hidden_states:
            return video_outputs
        batch_size = cmasks.size(0)
        video_attention_mask = torch.cat([torch.zeros((batch_size, 1), dtype=torch.bool, device=vmasks.device), vmasks, torch.ones((batch_size, 1), dtype=torch.bool, device=vmasks.device)], dim=1)
        assert video_outputs.size(1) == video_attention_mask.size(1)
        video_attention_mask = video_attention_mask.type(video_outputs.dtype) / video_attention_mask.sum(1, keepdim=True)
        pooled_video = torch.bmm(video_outputs.transpose(2, 1), video_attention_mask.unsqueeze(2)).squeeze(-1)
        return pooled_video

    def forward_text(self, caps, cmasks, output_hidden_states=False, **kwargs):
        input_ids = torch.cat([caps[:, :1], caps[:, 2:]], dim=1)
        attention_mask = torch.cat([cmasks[:, :1], cmasks[:, 2:]], dim=1)
        token_type_ids = torch.cat([torch.zeros((cmasks.size(0), 1), dtype=torch.long, device=cmasks.device), torch.ones((cmasks.size(0), cmasks.size(1) - 2), dtype=torch.long, device=cmasks.device)], dim=1)
        outputs = self.mm_encoder(input_ids=input_ids, input_video_embeds=None, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        text_outputs = outputs[0]
        if output_hidden_states:
            return text_outputs
        batch_size = caps.size(0)
        text_attention_mask = torch.cat([torch.zeros((batch_size, 1), dtype=torch.bool, device=cmasks.device), cmasks[:, 2:]], dim=1)
        assert text_outputs.size(1) == text_attention_mask.size(1)
        text_attention_mask = text_attention_mask.type(text_outputs.dtype) / text_attention_mask.sum(1, keepdim=True)
        pooled_text = torch.bmm(text_outputs.transpose(2, 1), text_attention_mask.unsqueeze(2)).squeeze(-1)
        return pooled_text


class MMFusionSeparate(MMFusionShare):

    def forward_video(self, vfeats, vmasks, caps, cmasks, output_hidden_states=False, **kwargs):
        input_ids = caps[:, :2]
        attention_mask = torch.cat([cmasks[:, :1], vmasks, cmasks[:, 1:2]], dim=1)
        token_type_ids = torch.zeros((vmasks.size(0), vmasks.size(1) + 2), dtype=torch.long, device=vmasks.device)
        outputs = self.video_encoder(input_ids=input_ids, input_video_embeds=vfeats, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        video_outputs = outputs[0]
        if output_hidden_states:
            return video_outputs
        batch_size = cmasks.size(0)
        video_attention_mask = torch.cat([torch.zeros((batch_size, 1), dtype=torch.bool, device=vmasks.device), vmasks, torch.ones((batch_size, 1), dtype=torch.bool, device=vmasks.device)], dim=1)
        assert video_outputs.size(1) == video_attention_mask.size(1)
        video_attention_mask = video_attention_mask.type(video_outputs.dtype) / video_attention_mask.sum(1, keepdim=True)
        pooled_video = torch.bmm(video_outputs.transpose(2, 1), video_attention_mask.unsqueeze(2)).squeeze(-1)
        return pooled_video

    def forward_text(self, caps, cmasks, output_hidden_states=False, **kwargs):
        input_ids = torch.cat([caps[:, :1], caps[:, 2:]], dim=1)
        attention_mask = torch.cat([cmasks[:, :1], cmasks[:, 2:]], dim=1)
        token_type_ids = torch.zeros((cmasks.size(0), cmasks.size(1) - 1), dtype=torch.long, device=cmasks.device)
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        text_outputs = outputs[0]
        if output_hidden_states:
            return text_outputs
        batch_size = caps.size(0)
        text_attention_mask = torch.cat([torch.zeros((batch_size, 1), dtype=torch.bool, device=cmasks.device), cmasks[:, 2:]], dim=1)
        assert text_outputs.size(1) == text_attention_mask.size(1)
        text_attention_mask = text_attention_mask.type(text_outputs.dtype) / text_attention_mask.sum(1, keepdim=True)
        pooled_text = torch.bmm(text_outputs.transpose(2, 1), text_attention_mask.unsqueeze(2)).squeeze(-1)
        return pooled_text


class MMFusionJoint(MMFusion):
    """fine-tuning wrapper for retrival task."""

    def forward(self, caps, cmasks, vfeats, vmasks, attention_mask=None, video_label=None, text_label=None, **kwargs):
        output_hidden_states = True
        attention_mask, token_type_ids = self._mm_on_the_fly(cmasks, vmasks, attention_mask)
        separate_forward_split = None if self.is_train else vmasks.size(1) + 2
        outputs = self.mm_encoder(input_ids=caps, input_video_embeds=vfeats, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=output_hidden_states, separate_forward_split=separate_forward_split)
        pooled_video, pooled_text = self._pooling_vt_layer(outputs[2], cmasks, vmasks)
        return {'pooled_video': pooled_video, 'pooled_text': pooled_text}


class MMFusionActionSegmentation(MMFusion):
    """Fine-tuning wrapper for action segmentation.
    TODO: rename this for VLM.
    """

    def forward(self, caps, cmasks, vfeats, vmasks, attention_mask=None, **kwargs):
        caps = caps.view(-1, caps.size(-1))
        cmasks = cmasks.view(-1, cmasks.size(-1))
        vfeats = vfeats.view(-1, vfeats.size(2), vfeats.size(3))
        vmasks = vmasks.view(-1, vmasks.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(2), attention_mask.size(3)) if attention_mask is not None else None
        output_hidden_states = True
        attention_mask, token_type_ids = self._mm_on_the_fly(cmasks, vmasks, attention_mask)
        logits = self.mm_encoder(input_ids=caps, input_video_embeds=vfeats, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=output_hidden_states)
        return {'logits': logits[0][:, 1:vmasks.size(1) + 1]}


class MMFusionActionLocalization(MMFusion):
    """fine-tuning model for retrival task."""

    def __init__(self, config, **kwargs):
        super().__init__(config)
        tokenizer = AutoTokenizer.from_pretrained(config.dataset.bert_name)
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def forward(self, caps, cmasks, vfeats, vmasks, attention_mask=None, **kwargs):
        caps = caps.squeeze(0)
        cmasks = cmasks.squeeze(0)
        vfeats = vfeats.squeeze(0)
        vmasks = vmasks.squeeze(0)
        attention_mask = attention_mask.squeeze(0) if attention_mask is not None else None
        output_hidden_states = True
        dummy_vfeats = torch.zeros((caps.size(0), 1, vfeats.size(-1)), device=vfeats.device, dtype=vfeats.dtype)
        dummy_vmasks = torch.ones((caps.size(0), 1), dtype=torch.bool, device=vfeats.device)
        dummy_caps = torch.LongTensor([[self.cls_token_id, self.sep_token_id, self.pad_token_id, self.sep_token_id]]).repeat(vfeats.size(0), 1)
        dummy_cmasks = torch.BoolTensor([[0, 1, 0, 1]]).repeat(vfeats.size(0), 1)
        attention_mask, token_type_ids = self._mm_on_the_fly(dummy_cmasks, vmasks, None)
        outputs = self.mm_encoder(input_ids=dummy_caps, input_video_embeds=vfeats, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=output_hidden_states)
        layer_idx = self.last_iso_layer if self.last_iso_layer > 0 else self.num_hidden_layers
        video_seq = outputs[2][layer_idx][:, 1:vmasks.size(1) + 1].masked_select(vmasks.unsqueeze(-1)).view(-1, self.hidden_size)
        attention_mask, token_type_ids = self._mm_on_the_fly(cmasks, dummy_vmasks, None)
        outputs = self.mm_encoder(input_ids=caps, input_video_embeds=dummy_vfeats, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=output_hidden_states)
        _, pooled_text = self._pooling_vt_layer(outputs[2], cmasks, dummy_vmasks)
        logits = torch.mm(video_seq, pooled_text.transpose(1, 0))
        return {'logits': logits}


class MMFusionSeparateActionSegmentation(MMFusionSeparate):
    """Fine-tuning wrapper for action segmentation."""

    def forward(self, caps, cmasks, vfeats, vmasks, attention_mask=None, **kwargs):
        caps = caps.view(-1, caps.size(-1))
        cmasks = cmasks.view(-1, cmasks.size(-1))
        vfeats = vfeats.view(-1, vfeats.size(2), vfeats.size(3))
        vmasks = vmasks.view(-1, vmasks.size(-1))
        logits = self.forward_video(vfeats, vmasks, caps, cmasks, output_hidden_states=True)
        return {'logits': logits[:, 1:vmasks.size(1) + 1]}


class MMFusionSeparateActionLocalization(MMFusionSeparate):

    def __init__(self, config, **kwargs):
        super().__init__(config)
        tokenizer = AutoTokenizer.from_pretrained(config.dataset.bert_name)
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def forward(self, caps, cmasks, vfeats, vmasks, **kwargs):
        caps = caps.squeeze(0)
        cmasks = cmasks.squeeze(0)
        vfeats = vfeats.squeeze(0)
        vmasks = vmasks.squeeze(0)
        dummy_caps = torch.LongTensor([[self.cls_token_id, self.sep_token_id, self.pad_token_id, self.sep_token_id]]).repeat(vfeats.size(0), 1)
        dummy_cmasks = torch.BoolTensor([[0, 1, 0, 1]]).repeat(vfeats.size(0), 1)
        outputs = self.forward_video(vfeats, vmasks, dummy_caps, dummy_cmasks, output_hidden_states=True)
        video_seq = outputs[:, 1:vmasks.size(1) + 1].masked_select(vmasks.unsqueeze(-1)).view(-1, self.hidden_size)
        pooled_text = self.forward_text(caps, cmasks, output_hidden_states=False)
        logits = torch.mm(video_seq, pooled_text.transpose(1, 0))
        return {'logits': logits}


class MMFusionShareActionLocalization(MMFusionShare):

    def __init__(self, config, **kwargs):
        super().__init__(config)
        tokenizer = AutoTokenizer.from_pretrained(config.dataset.bert_name)
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def forward(self, caps, cmasks, vfeats, vmasks, **kwargs):
        caps = caps.squeeze(0)
        cmasks = cmasks.squeeze(0)
        vfeats = vfeats.squeeze(0)
        vmasks = vmasks.squeeze(0)
        dummy_caps = torch.LongTensor([[self.cls_token_id, self.sep_token_id, self.pad_token_id, self.sep_token_id]]).repeat(vfeats.size(0), 1)
        dummy_cmasks = torch.BoolTensor([[0, 1, 0, 1]]).repeat(vfeats.size(0), 1)
        outputs = self.forward_video(vfeats, vmasks, dummy_caps, dummy_cmasks, output_hidden_states=True)
        video_seq = outputs[:, 1:vmasks.size(1) + 1].masked_select(vmasks.unsqueeze(-1)).view(-1, self.hidden_size)
        pooled_text = self.forward_text(caps, cmasks, output_hidden_states=False)
        logits = torch.mm(video_seq, pooled_text.transpose(1, 0))
        return {'logits': logits}


class MMFusionNLG(MMFusion):

    def __init__(self, config, **kwargs):
        super().__init__(config)
        if config.model.max_decode_length is not None:
            self.max_length = min(config.model.max_decode_length, config.dataset.max_len - config.dataset.max_video_len - 3)
        else:
            self.max_length = config.dataset.max_len - config.dataset.max_video_len - 3
        self.gen_param = config.gen_param if config.gen_param is not None else {}

    def forward(self, caps, cmasks, vfeats, vmasks, attention_mask, video_label=None, text_label=None, **kwargs):
        """use pre-trained LM header for generation."""
        attention_mask, token_type_ids = self._mm_on_the_fly(cmasks, vmasks, attention_mask)
        outputs = self.mm_encoder(input_ids=caps, input_video_embeds=vfeats, attention_mask=attention_mask, token_type_ids=token_type_ids, masked_lm_labels=text_label)
        return {'logits': outputs[0]}

    @torch.no_grad()
    def generate(self, caps, cmasks, vfeats, vmasks, attention_mask=None, bos_token_id=None, eos_token_id=None, **kwargs):
        assert caps.size(1) == 3
        attention_mask, token_type_ids = self._mm_on_the_fly(cmasks, vmasks, attention_mask)
        output = self.mm_encoder.generate(input_ids=caps, input_video_embeds=vfeats, attention_mask=attention_mask, token_type_ids=token_type_ids, bos_token_id=bos_token_id, eos_token_id=eos_token_id, max_length=self.max_length, **self.gen_param)
        return output


class AlignHead(nn.Module):
    """this will load pre-trained weights for NSP, which is desirable."""

    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, dropout_pooled_output):
        logits = self.seq_relationship(dropout_pooled_output)
        return logits


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len))
    mask = ids < lengths.unsqueeze(1)
    return mask


class GlobalAvgPool(torch.nn.Module):

    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x, lengths=None):
        """Average pooling across time steps (dim=1) with optionally lengths.
        Args:
            x: torch.Tensor of shape (N, T, ...)
            lengths: None or torch.Tensor of shape (N,)
            dim: dimension to pool
        """
        if lengths is None:
            return x.mean(dim=1, keepdim=False)
        else:
            mask = get_mask_from_lengths(lengths).type(x.type())
            mask_shape = list(mask.size()) + [(1) for _ in range(x.ndimension() - 2)]
            mask = mask.reshape(*mask_shape)
            numer = (x * mask).sum(dim=1, keepdim=False)
            denom = mask.sum(dim=1, keepdim=False)
            return numer / denom


class AdaptiveMask(nn.Module):
    """Soft masking function for adaptive size.
    It masks out the last K values of an input. The masking value
    goes from 1 to 0 gradually, so K can be learned with
    back-propagation.
    Args:
        max_size: maximum size (i.e. input dimension)
        ramp_size: size of the ramp going from 0 to 1
        init_val: initial size proportion not to be masked out
        shape: learn multiple sizes independent of each other
    """

    def __init__(self, max_size, ramp_size, init_val=0, shape=(1,)):
        nn.Module.__init__(self)
        self._max_size = max_size
        self._ramp_size = ramp_size
        self.current_val = nn.Parameter(torch.zeros(*shape) + init_val)
        mask_template = torch.linspace(1 - max_size, 0, steps=max_size)
        self.register_buffer('mask_template', mask_template)

    def forward(self, x):
        mask = self.mask_template.float() + self.current_val.float() * self._max_size
        mask = mask / self._ramp_size + 1
        mask = mask.clamp(0, 1)
        if x.size(-1) < self._max_size:
            mask = mask.narrow(-1, self._max_size - x.size(-1), x.size(-1))
        x = (x * mask).type_as(x)
        return x

    def get_current_max_size(self, include_ramp=True):
        current_size = math.ceil(self.current_val.max().item() * self._max_size)
        if include_ramp:
            current_size += self._ramp_size
        current_size = max(0, min(self._max_size, current_size))
        return current_size

    def get_current_avg_size(self, include_ramp=True):
        current_size = math.ceil(self.current_val.float().mean().item() * self._max_size)
        if include_ramp:
            current_size += self._ramp_size
        current_size = max(0, min(self._max_size, current_size))
        return current_size

    def clamp_param(self):
        """this need to be called after each update"""
        self.current_val.data.clamp_(0, 1)


class AdaptiveSpan(nn.Module):
    """Adaptive attention span for Transformerself.
    This module learns an attention span length from data for each
    self-attention head.
    Args:
        attn_span: maximum attention span
        adapt_span_loss: loss coefficient for the span length
        adapt_span_ramp: length of the masking ramp
        adapt_span_init: initial size ratio
        adapt_span_cache: adapt cache size to reduce memory usage
    """

    def __init__(self, attn_span, adapt_span_ramp, adapt_span_init, n_head, adapt_span_layer, **kargs):
        nn.Module.__init__(self)
        self._max_span = attn_span
        self._n_head = n_head
        self._adapt_span_layer = adapt_span_layer
        if self._adapt_span_layer:
            self._mask = AdaptiveMask(max_size=self._max_span, ramp_size=adapt_span_ramp, init_val=adapt_span_init)
        else:
            self._mask = AdaptiveMask(max_size=self._max_span, ramp_size=adapt_span_ramp, init_val=adapt_span_init, shape=(n_head, 1, 1))

    def forward(self, attn, normalize=True):
        """mask attention with the right span"""
        self.clamp_param()
        if self._adapt_span_layer:
            attn = self._mask(attn)
        else:
            B = attn.size(0)
            M = attn.size(1)
            attn = attn.reshape(B // self._n_head, self._n_head, M, -1)
            attn = self._mask(attn)
            attn = attn.view(B, M, -1)
        return attn

    def get_trim_len(self):
        """how much of memory can be trimmed to reduce computation"""
        L = self._max_span
        trim_len = min(L - 1, L - self._mask.get_current_max_size())
        trim_len = math.floor(trim_len / 64) * 64
        return trim_len

    def trim_memory(self, query, key, value, key_pe):
        """trim out unnecessary memory beforehand to reduce computation"""
        trim_len = self.get_trim_len()
        cache_size = key.size(1) - query.size(1)
        trim_len_cache = trim_len - (self._max_span - cache_size)
        if trim_len_cache > 0:
            key = key[:, trim_len_cache:, :]
            value = value[:, trim_len_cache:, :]
        elif trim_len_cache < 0:
            key = F.pad(key, [0, 0, -trim_len_cache, 0])
            value = F.pad(value, [0, 0, -trim_len_cache, 0])
        if trim_len > 0:
            if key_pe is not None:
                key_pe = key_pe[:, :, trim_len:]
        return key, value, key_pe

    def get_cache_size(self):
        """determine how long the cache should be"""
        trim_len = self.get_trim_len()
        return min(self._max_span, self._max_span - trim_len + 64)

    def get_loss(self):
        """a loss term for regularizing the span length"""
        return self._max_span * self._mask.current_val.float().mean()

    def get_current_max_span(self):
        return self._mask.get_current_max_size()

    def get_current_avg_span(self):
        return self._mask.get_current_avg_size()

    def clamp_param(self):
        self._mask.clamp_param()


def _skew(X, pad_value):
    """shift every row 1 step to right"""
    B, M, L = X.size()
    X = F.pad(X, (0, M + 1), value=pad_value)
    X = X.view(B, -1)
    X = X[:, :-M]
    X = X.view(B, M, M + L)
    return X


def _unskew(X):
    """reverse _skew operation"""
    B, M, L = X.size()
    L -= M
    X = X.view(B, -1)
    X = F.pad(X, (0, M))
    X = X.view(B, M, M + L + 1)
    X = X[:, :, :L]
    return X


class SeqAttention(nn.Module):
    """Sequential self-attention layer.
    Each token will attend to its previous fixed number of steps.
    Note that attention doesn't include the current step itself.
    """

    def __init__(self, d_model, n_head, attn_span, dropout, adapt_span_layer, **kargs):
        nn.Module.__init__(self)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.attn_span = attn_span
        self.adaptive_span = AdaptiveSpan(attn_span=attn_span, n_head=n_head, adapt_span_layer=adapt_span_layer, **kargs)

    def forward(self, query, key, value, key_pe):
        key, value, key_pe = self.adaptive_span.trim_memory(query, key, value, key_pe)
        attn_cont = torch.matmul(query, key.transpose(-1, -2))
        attn_cont = _unskew(attn_cont)
        attn_pos = torch.matmul(query, key_pe)
        attn = attn_cont + attn_pos
        attn = attn / math.sqrt(self.d_model)
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        attn = self.adaptive_span(attn)
        attn = self.dropout(attn)
        attn_cont = _skew(attn, 0)
        out = torch.matmul(attn_cont, value)
        return out

    def get_cache_size(self):
        return self.adaptive_span.get_cache_size()


class MultiHeadSeqAttention(nn.Module):

    def __init__(self, d_model, n_head, **kargs):
        nn.Module.__init__(self)
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.attn = SeqAttention(d_model=self.head_dim, n_head=n_head, **kargs)
        self.proj_query = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_normal_(self.proj_query.weight)
        self.proj_out = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_normal_(self.proj_out.weight)
        self.proj_val = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_normal_(self.proj_val.weight)
        self.proj_key = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_normal_(self.proj_key.weight)

    def head_reshape(self, x):
        K = self.n_head
        D = self.head_dim
        x = x.view(x.size()[:-1] + (K, D))
        x = x.transpose(1, 2).contiguous()
        x = x.view(-1, x.size(-2), x.size(-1))
        return x

    def forward(self, query, key, value, key_pe):
        B = query.size(0)
        K = self.n_head
        D = self.head_dim
        M = query.size(1)
        query = self.proj_query(query)
        query = self.head_reshape(query)
        value = self.proj_val(value)
        value = self.head_reshape(value)
        key = self.proj_key(key)
        key = self.head_reshape(key)
        out = self.attn(query, key, value, key_pe)
        out = out.view(B, K, M, D)
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, M, -1)
        out = self.proj_out(out)
        return out


class FeedForwardLayer(nn.Module):

    def __init__(self, d_model, d_inner, dropout, **kargs):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(d_model, d_inner)
        self.fc2 = nn.Linear(d_inner, d_model)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        h1 = F.relu(self.fc1(h))
        h1 = self.dropout(h1)
        h2 = self.fc2(h1)
        return h2


class TransformerSeqLayer(nn.Module):

    def __init__(self, d_model, **kargs):
        nn.Module.__init__(self)
        self.attn = MultiHeadSeqAttention(d_model=d_model, **kargs)
        self.norm1 = LayerNorm(d_model)
        self.ff = FeedForwardLayer(d_model=d_model, **kargs)
        self.norm2 = LayerNorm(d_model)

    def forward(self, h, h_cache, key_pe):
        h_all = torch.cat([h_cache, h], dim=1)
        attn_out = self.attn(h, h_all, h_all, key_pe)
        h = self.norm1(h + attn_out)
        if self.ff is not None:
            ff_out = self.ff(h)
            out = self.norm2(h + ff_out)
        else:
            out = h
        return out

    def get_cache_size(self):
        return self.attn.attn.get_cache_size()


class TransformerSeq(nn.Module):

    def __init__(self, vocab_size, d_model, n_head, n_layer, attn_span, emb_dropout, aux_loss_scaler, adapt_span_layer, **kargs):
        nn.Module.__init__(self)
        self.in_emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.in_emb.weight, mean=0, std=d_model ** -0.5)
        self.out_emb = nn.Linear(d_model, vocab_size)
        self.aux_loss_scaler = aux_loss_scaler
        if emb_dropout > 0:
            self.emb_dropout = nn.Dropout(emb_dropout)
        else:
            self.emb_dropout = None
        self.key_pe = nn.Parameter(torch.randn(1, d_model // n_head, attn_span))
        self.layers = nn.ModuleList()
        self.layers.extend(TransformerSeqLayer(d_model=d_model, n_head=n_head, attn_span=attn_span, adapt_span_layer=adapt_span_layer, **kargs) for _ in range(n_layer))

    def forward(self, x, h_cache, target=None):
        block_size = x.size(1)
        h = self.in_emb(x)
        if self.emb_dropout is not None:
            h = self.emb_dropout(h)
        h_cache_next = []
        for l, layer in enumerate(self.layers):
            cache_size = layer.attn.attn.get_cache_size()
            if cache_size > block_size:
                h_cache_next_l = torch.cat([h_cache[l][:, -cache_size + block_size:, :], h], dim=1).detach()
            else:
                h_cache_next_l = h[:, -cache_size:, :].detach()
            h_cache_next.append(h_cache_next_l)
            h = layer(h, h_cache[l], self.key_pe)
        if self.emb_dropout is not None:
            h = self.emb_dropout(h)
        out = F.log_softmax(self.out_emb(h).float(), dim=-1).type_as(h)
        dummy_loss = None
        return out, h_cache_next, dummy_loss

    def get_aux_loss(self):
        loss = 0.0
        for layer in self.layers:
            loss += layer.attn.attn.adaptive_span.get_loss()
        return self.aux_loss_scaler * loss

    def get_current_max_span(self):
        max_span = 0.0
        for layer in self.layers:
            max_span = max(max_span, layer.attn.attn.adaptive_span.get_current_max_span())
        return max_span

    def get_current_avg_span(self):
        avg_span = 0.0
        for layer in self.layers:
            avg_span += layer.attn.attn.adaptive_span.get_current_avg_span()
        return avg_span / len(self.layers)


class HeadSelectionLoss(_Loss):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.kl_weight = getattr(args, 'kl_weight', 0.0)

    def forward(self, head_samples, sample_sizes, prior=0.5, eps=1e-07):
        """
        head_scores: (num_tasks, num_layers, num_heads)
        sample_sizes: (num_tasks, )
        """
        kl_loss = (head_samples * (torch.log(head_samples + eps) - math.log(prior))).sum(-1).sum(-1)
        kl_loss /= torch.numel(head_samples) / head_samples.size(0)
        kl_loss = self.kl_weight * torch.matmul(kl_loss, sample_sizes)
        return kl_loss


class AttnHeadSelector(nn.Module):
    """
    Latent variable modeling of attention head selection
    """

    def __init__(self, num_tasks, num_layers, total_num_heads, num_heads, select_strategy='group', head_select_temp=5.0):
        super(AttnHeadSelector, self).__init__()
        self.num_tasks = num_tasks
        self.num_layers = num_layers
        self.total_num_heads = total_num_heads
        self.num_heads = num_heads
        self.select_strategy = select_strategy
        self.temp = head_select_temp
        self.head_logits = torch.nn.Parameter(torch.Tensor(self.num_tasks, self.num_layers, total_num_heads), requires_grad=True)
        nn.init.uniform_(self.head_logits, a=math.log(0.01), b=math.log(1.0))

    def gumbel_sample(self, logits, tau=1.0):
        gumbels1 = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        gumbels2 = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        gumbels1 = (logits + gumbels1 - gumbels2) / tau
        y_soft = gumbels1.sigmoid()
        return y_soft

    def subset_select(self, y_soft, topk, dim=-1):
        top_values, top_inds = torch.topk(y_soft, k=topk, dim=dim)
        top_ret = 1.0 - top_values.detach() + top_values
        return top_inds.detach(), top_ret

    def group_selet(self, y_soft, topk, dim=-1):
        top_values, top_inds = torch.max(y_soft.view(self.num_tasks, self.num_layers, -1, topk), dim=2)
        top_inds = top_inds * topk + torch.arange(topk, device=top_inds.device).unsqueeze(0).unsqueeze(1)
        top_ret = 1.0 - top_values.detach() + top_values
        return top_inds.detach(), top_ret

    def head_select(self, task_ids=None):
        self.head_samples = self.gumbel_sample(self.head_logits, tau=self.temp)
        if self.select_strategy == 'subset':
            self.subset_heads, self.subset_weights = self.subset_select(self.head_samples, topk=self.num_heads)
        elif self.select_strategy == 'group':
            self.subset_heads, self.subset_weights = self.group_selet(self.head_samples, topk=self.num_heads)
        else:
            raise ValueError('{} is not supported'.format(self.select_strategy))
        self.batch_subset = self.subset_heads[task_ids, :, :]
        self.batch_weights = self.subset_weights[task_ids, :, :]

    def forward(self, layer_idx):
        assert layer_idx is not None
        batch_subset = self.batch_subset[:, layer_idx, :]
        batch_weights = self.batch_weights[:, layer_idx, :]
        return batch_subset, batch_weights


class BaseRanker(nn.Module):

    def __init__(self, args, task):
        super().__init__()
        self.separator_token = task.dictionary.eos()
        self.padding_idx = task.dictionary.pad()

    def forward(self, src_tokens):
        raise NotImplementedError

    def get_segment_labels(self, src_tokens):
        segment_boundary = (src_tokens == self.separator_token).long()
        segment_labels = segment_boundary.cumsum(dim=1) - segment_boundary - (src_tokens == self.padding_idx).long()
        return segment_labels

    def get_positions(self, src_tokens, segment_labels):
        segment_positions = torch.arange(src_tokens.shape[1]).repeat(src_tokens.shape[0], 1)
        segment_boundary = (src_tokens == self.separator_token).long()
        _, col_idx = (segment_positions * segment_boundary).nonzero(as_tuple=True)
        col_idx = torch.cat([torch.zeros(1).type_as(col_idx), col_idx])
        offset = torch.cat([torch.zeros(1).type_as(segment_boundary), segment_boundary.sum(dim=1).cumsum(dim=0)[:-1]])
        segment_positions -= col_idx[segment_labels + offset.unsqueeze(1)] * (segment_labels != 0)
        padding_mask = src_tokens.ne(self.padding_idx)
        segment_positions = (segment_positions + 1) * padding_mask.type_as(segment_positions) + self.padding_idx
        return segment_positions


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
        self.padding_idx = padding_idx if padding_idx is not None else 0
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
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size
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

    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02))
    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


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
        self.embed_tokens = self.build_embedding(self.vocab_size, self.embedding_dim, self.padding_idx)
        self.embed_scale = embed_scale
        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(nn.Linear(self.embedding_dim, self.embedding_dim, bias=False), q_noise, qn_block_size)
        else:
            self.quant_noise = None
        self.segment_embeddings = nn.Embedding(self.num_segments, self.embedding_dim, padding_idx=None) if self.num_segments > 0 else None
        self.embed_positions = PositionalEmbedding(self.max_seq_len, self.embedding_dim, padding_idx=self.padding_idx if offset_positions_by_padding else None, learned=self.learned_pos_embedding) if self.use_position_embeddings else None
        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None
        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend([self.build_transformer_sentence_encoder_layer(embedding_dim=self.embedding_dim, ffn_embedding_dim=ffn_embedding_dim, num_attention_heads=num_attention_heads, dropout=self.dropout_module.p, attention_dropout=attention_dropout, activation_dropout=activation_dropout, activation_fn=activation_fn, export=export, q_noise=q_noise, qn_block_size=qn_block_size) for _ in range(num_encoder_layers)])
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

    def forward(self, tokens: 'torch.Tensor', segment_labels: 'torch.Tensor'=None, last_state_only: 'bool'=False, positions: 'Optional[torch.Tensor]'=None, token_embeddings: 'Optional[torch.Tensor]'=None, attn_mask: 'Optional[torch.Tensor]'=None) ->Tuple[torch.Tensor, torch.Tensor]:
        is_tpu = tokens.device.type == 'xla'
        padding_mask = tokens.eq(self.padding_idx)
        if not self.traceable and not is_tpu and not padding_mask.any():
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
            x, _ = layer(x, self_attn_padding_mask=padding_mask, self_attn_mask=attn_mask)
            if not last_state_only:
                inner_states.append(x)
        sentence_rep = x[0, :, :]
        if last_state_only:
            inner_states = [x]
        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            return inner_states, sentence_rep


def update_init_roberta_model_state(state):
    """
   update the state_dict of a Roberta model for initializing
   weights of the BertRanker
   """
    for k in list(state.keys()):
        if '.lm_head.' in k or 'version' in k:
            del state[k]
            continue
        assert k.startswith('encoder.sentence_encoder.') or k.startswith('decoder.sentence_encoder.'), f'Cannot recognize parameter name {k}'
        if 'layernorm_embedding' in k:
            new_k = k.replace('.layernorm_embedding.', '.emb_layer_norm.')
            state[new_k[25:]] = state[k]
        else:
            state[k[25:]] = state[k]
        del state[k]


class BertRanker(BaseRanker):

    def __init__(self, args, task):
        super(BertRanker, self).__init__(args, task)
        init_model = getattr(args, 'pretrained_model', '')
        self.joint_layers = nn.ModuleList()
        if os.path.isfile(init_model):
            None
            x = hub_utils.from_pretrained(os.path.dirname(init_model), checkpoint_file=os.path.basename(init_model))
            in_state_dict = x['models'][0].state_dict()
            init_args = x['args'].model
            num_positional_emb = init_args.max_positions + task.dictionary.pad() + 1
            self.model = TransformerSentenceEncoder(padding_idx=task.dictionary.pad(), vocab_size=len(task.dictionary), num_encoder_layers=getattr(args, 'encoder_layers', init_args.encoder_layers), embedding_dim=init_args.encoder_embed_dim, ffn_embedding_dim=init_args.encoder_ffn_embed_dim, num_attention_heads=init_args.encoder_attention_heads, dropout=init_args.dropout, attention_dropout=init_args.attention_dropout, activation_dropout=init_args.activation_dropout, num_segments=2, max_seq_len=num_positional_emb, offset_positions_by_padding=False, encoder_normalize_before=True, apply_bert_init=True, activation_fn=init_args.activation_fn, freeze_embeddings=args.freeze_embeddings, n_trans_layers_to_freeze=args.n_trans_layers_to_freeze)
            if args.freeze_embeddings:
                for p in self.model.segment_embeddings.parameters():
                    p.requires_grad = False
            update_init_roberta_model_state(in_state_dict)
            None
            self.model.load_state_dict(in_state_dict, strict=False)
            ffn_embedding_dim = init_args.encoder_ffn_embed_dim
            num_attention_heads = init_args.encoder_attention_heads
            dropout = init_args.dropout
            attention_dropout = init_args.attention_dropout
            activation_dropout = init_args.activation_dropout
            activation_fn = init_args.activation_fn
            classifier_embed_dim = getattr(args, 'embed_dim', init_args.encoder_embed_dim)
            if classifier_embed_dim != init_args.encoder_embed_dim:
                self.transform_layer = nn.Linear(init_args.encoder_embed_dim, classifier_embed_dim)
        else:
            self.model = TransformerSentenceEncoder(padding_idx=task.dictionary.pad(), vocab_size=len(task.dictionary), num_encoder_layers=args.encoder_layers, embedding_dim=args.embed_dim, ffn_embedding_dim=args.ffn_embed_dim, num_attention_heads=args.attention_heads, dropout=args.dropout, attention_dropout=args.attention_dropout, activation_dropout=args.activation_dropout, max_seq_len=task.max_positions() if task.max_positions() else args.tokens_per_sample, num_segments=2, offset_positions_by_padding=False, encoder_normalize_before=args.encoder_normalize_before, apply_bert_init=args.apply_bert_init, activation_fn=args.activation_fn)
            classifier_embed_dim = args.embed_dim
            ffn_embedding_dim = args.ffn_embed_dim
            num_attention_heads = args.attention_heads
            dropout = args.dropout
            attention_dropout = args.attention_dropout
            activation_dropout = args.activation_dropout
            activation_fn = args.activation_fn
        self.joint_classification = args.joint_classification
        if args.joint_classification == 'sent':
            if args.joint_normalize_before:
                self.joint_layer_norm = LayerNorm(classifier_embed_dim)
            else:
                self.joint_layer_norm = None
            self.joint_layers = nn.ModuleList([TransformerSentenceEncoderLayer(embedding_dim=classifier_embed_dim, ffn_embedding_dim=ffn_embedding_dim, num_attention_heads=num_attention_heads, dropout=dropout, attention_dropout=attention_dropout, activation_dropout=activation_dropout, activation_fn=activation_fn) for _ in range(args.num_joint_layers)])
        self.classifier = RobertaClassificationHead(classifier_embed_dim, classifier_embed_dim, 1, 'tanh', args.classifier_dropout)

    def forward(self, src_tokens, src_lengths):
        segment_labels = self.get_segment_labels(src_tokens)
        positions = self.get_positions(src_tokens, segment_labels)
        inner_states, _ = self.model(tokens=src_tokens, segment_labels=segment_labels, last_state_only=True, positions=positions)
        return inner_states[-1].transpose(0, 1)

    def sentence_forward(self, encoder_out, src_tokens=None, sentence_rep='head'):
        if sentence_rep == 'head':
            x = encoder_out[:, :1, :]
        else:
            assert src_tokens is not None, 'meanpool requires src_tokens input'
            segment_labels = self.get_segment_labels(src_tokens)
            padding_mask = src_tokens.ne(self.padding_idx)
            encoder_mask = segment_labels * padding_mask.type_as(segment_labels)
            if sentence_rep == 'meanpool':
                ntokens = torch.sum(encoder_mask, dim=1, keepdim=True)
                x = torch.sum(encoder_out * encoder_mask.unsqueeze(2), dim=1, keepdim=True) / ntokens.unsqueeze(2).type_as(encoder_out)
            else:
                encoder_out[(encoder_mask == 0).unsqueeze(2).repeat(1, 1, encoder_out.shape[-1])] = -float('inf')
                x, _ = torch.max(encoder_out, dim=1, keepdim=True)
        if hasattr(self, 'transform_layer'):
            x = self.transform_layer(x)
        return x

    def joint_forward(self, x):
        if self.joint_layer_norm:
            x = self.joint_layer_norm(x.transpose(0, 1))
            x = x.transpose(0, 1)
        for layer in self.joint_layers:
            x, _ = layer(x, self_attn_padding_mask=None)
        return x

    def classification_forward(self, x):
        return self.classifier(x)


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

    def __init__(self, num_layers, num_logits, soft_select=False, sampling_tau=5.0):
        super(LayerSelect, self).__init__()
        self.layer_logits = torch.nn.Parameter(torch.Tensor(num_logits, num_layers), requires_grad=True)
        self.hard_select = not soft_select
        self.tau = sampling_tau
        self.detach_grad = False
        self.layer_samples = [None] * num_logits

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

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

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


class BLSTM(nn.Module):

    def __init__(self, dim, layers=2, bi=True):
        super().__init__()
        klass = nn.LSTM
        self.lstm = klass(bidirectional=bi, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = None
        if bi:
            self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        if self.linear:
            x = self.linear(x)
        return x, hidden


def capture_init(init):
    """capture_init.

    Decorate `__init__` with this, and you can then
    recover the *args and **kwargs passed to it in `self._init_args_kwargs`
    """

    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = args, kwargs
        init(self, *args, **kwargs)
    return __init__


def sinc(t):
    """sinc.

    :param t: the input tensor
    """
    return th.where(t == 0, th.tensor(1.0, device=t.device, dtype=t.dtype), th.sin(t) / t)


def kernel_downsample2(zeros=56):
    """kernel_downsample2.

    """
    win = th.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = th.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t.mul_(math.pi)
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel


def downsample2(x, zeros=56):
    """
    Downsampling the input by 2 using sinc interpolation.
    Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
    ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
    Vol. 9. IEEE, 1984.
    """
    if x.shape[-1] % 2 != 0:
        x = F.pad(x, (0, 1))
    xeven = x[..., ::2]
    xodd = x[..., 1::2]
    *other, time = xodd.shape
    kernel = kernel_downsample2(zeros)
    out = xeven + F.conv1d(xodd.view(-1, 1, time), kernel, padding=zeros)[..., :-1].view(*other, time)
    return out.view(*other, -1).mul(0.5)


def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


def kernel_upsample2(zeros=56):
    """kernel_upsample2.

    """
    win = th.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = th.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t *= math.pi
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel


def upsample2(x, zeros=56):
    """
    Upsampling the input by 2 using sinc interpolation.
    Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
    ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
    Vol. 9. IEEE, 1984.
    """
    *other, time = x.shape
    kernel = kernel_upsample2(zeros)
    out = F.conv1d(x.view(-1, 1, time), kernel, padding=zeros)[..., 1:].view(*other, time)
    y = th.stack([x, out], dim=-1)
    return y.view(*other, -1)


class Demucs(nn.Module):
    """
    Demucs speech enhancement model.
    Args:
        - chin (int): number of input channels.
        - chout (int): number of output channels.
        - hidden (int): number of initial hidden channels.
        - depth (int): number of layers.
        - kernel_size (int): kernel size for each layer.
        - stride (int): stride for each layer.
        - causal (bool): if false, uses BiLSTM instead of LSTM.
        - resample (int): amount of resampling to apply to the input/output.
            Can be one of 1, 2 or 4.
        - growth (float): number of channels is multiplied by this for every layer.
        - max_hidden (int): maximum number of channels. Can be useful to
            control the size/speed of the model.
        - normalize (bool): if true, normalize the input.
        - glu (bool): if true uses GLU instead of ReLU in 1x1 convolutions.
        - rescale (float): controls custom weight initialization.
            See https://arxiv.org/abs/1911.13254.
        - floor (float): stability flooring when normalizing.

    """

    @capture_init
    def __init__(self, chin=1, chout=1, hidden=48, depth=5, kernel_size=8, stride=4, causal=True, resample=4, growth=2, max_hidden=10000, normalize=True, glu=True, rescale=0.1, floor=0.001):
        super().__init__()
        if resample not in [1, 2, 4]:
            raise ValueError('Resample should be 1, 2 or 4.')
        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.floor = floor
        self.resample = resample
        self.normalize = normalize
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1
        for index in range(depth):
            encode = []
            encode += [nn.Conv1d(chin, hidden, kernel_size, stride), nn.ReLU(), nn.Conv1d(hidden, hidden * ch_scale, 1), activation]
            self.encoder.append(nn.Sequential(*encode))
            decode = []
            decode += [nn.Conv1d(hidden, ch_scale * hidden, 1), activation, nn.ConvTranspose1d(hidden, chout, kernel_size, stride)]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden = min(int(growth * hidden), max_hidden)
        self.lstm = BLSTM(chin, bi=not causal)
        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.resample)
        for _ in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for _ in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)

    @property
    def total_stride(self):
        return self.stride ** self.depth // self.resample

    def forward(self, mix):
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)
        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (self.floor + std)
        else:
            std = 1
        length = mix.shape[-1]
        x = mix
        x = F.pad(x, (0, self.valid_length(length) - length))
        if self.resample == 2:
            x = upsample2(x)
        elif self.resample == 4:
            x = upsample2(x)
            x = upsample2(x)
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = decode(x)
        if self.resample == 2:
            x = downsample2(x)
        elif self.resample == 4:
            x = downsample2(x)
            x = downsample2(x)
        x = x[..., :length]
        return std * x


class LinearNorm(torch.nn.Module):

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class SpeechEmbedder(nn.Module):

    def __init__(self, hp):
        super(SpeechEmbedder, self).__init__()
        self.lstm = nn.LSTM(hp['num_mels'], hp['lstm_hidden'], num_layers=hp['lstm_layers'], batch_first=True)
        self.proj = LinearNorm(hp)
        self.hp = hp

    def forward(self, mel):
        mels = mel.unfold(1, self.hp['window'], self.hp['stride'])
        mels = mels.permute(1, 2, 0)
        x, _ = self.lstm(mels)
        x = x[:, -1, :]
        x = self.proj(x)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        x = x.mean(dim=0)
        if x.norm(p=2) != 0:
            x = x / x.norm(p=2)
        return x


EMBEDDER_PARAMS = {'num_mels': 40, 'n_fft': 512, 'emb_dim': 256, 'lstm_hidden': 768, 'lstm_layers': 3, 'window': 80, 'stride': 40}


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary
    computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class SpkrEmbedder(nn.Module):
    RATE = 16000

    def __init__(self, embedder_path, embedder_params=EMBEDDER_PARAMS, rate=16000, hop_length=160, win_length=400, pad=False):
        super(SpkrEmbedder, self).__init__()
        embedder_pt = torch.load(embedder_path, map_location='cpu')
        self.embedder = SpeechEmbedder(embedder_params)
        self.embedder.load_state_dict(embedder_pt)
        self.embedder.eval()
        set_requires_grad(self.embedder, requires_grad=False)
        self.embedder_params = embedder_params
        self.register_buffer('mel_basis', torch.from_numpy(librosa.filters.mel(sr=self.RATE, n_fft=self.embedder_params['n_fft'], n_mels=self.embedder_params['num_mels'])))
        self.resample = None
        if rate != self.RATE:
            self.resample = torchaudio.transforms.Resample(rate, self.RATE)
        self.hop_length = hop_length
        self.win_length = win_length
        self.pad = pad

    def get_mel(self, y):
        if self.pad and y.shape[-1] < 14000:
            y = F.pad(y, (0, 14000 - y.shape[-1]))
        window = torch.hann_window(self.win_length)
        y = torch.stft(y, n_fft=self.embedder_params['n_fft'], hop_length=self.hop_length, win_length=self.win_length, window=window)
        magnitudes = torch.norm(y, dim=-1, p=2) ** 2
        mel = torch.log10(self.mel_basis @ magnitudes + 1e-06)
        return mel

    def forward(self, inputs):
        dvecs = []
        for wav in inputs:
            mel = self.get_mel(wav)
            if mel.dim() == 3:
                mel = mel.squeeze(0)
            dvecs += [self.embedder(mel)]
        dvecs = torch.stack(dvecs)
        dvec = torch.mean(dvecs, dim=0)
        dvec = dvec / torch.norm(dvec)
        return dvec


class BenchmarkingBase(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.s2x_task = None

    def warm_up(self, sample, repeat):
        """Warm up the model"""
        for _i in range(repeat):
            self.forward(sample)
        logger.info(f'Model warmed up by running inference {repeat} times')

    def benchmark_run_time(self, dataset, repeat):
        """Benchmark average runtime for the model by calling benchmark_run_time_single_sample function"""
        logger.info('Starting run time benchmarking')
        time_elapsed = 0
        for i, sample in enumerate(dataset):
            time_elapsed += self.benchmark_run_time_single_sample(sample, repeat=repeat)
            if i % 100 == 0:
                logger.info(f'Benchmarked run time for {i}/{len(dataset)} samples')
        total_time_elapsed = time_elapsed / len(dataset)
        return total_time_elapsed

    def benchmark_run_time_single_sample(self, sample, repeat):
        """Benchmark average runtime for a single sample using timeit library. Units are seconds"""
        timer = timeit.Timer(lambda : self.forward(sample))
        time_elapsed = timer.timeit(repeat)
        return time_elapsed / repeat

    def count_flops(self, dataset, repeat):
        """Use PYPAPI library to count average flops for model inference.
        Note: It only works if the model is being run on cpu"""
        logger.info('Starting flop counter')
        high.start_counters([events.PAPI_DP_OPS])
        for i, sample in enumerate(dataset):
            for _r in range(repeat):
                self.forward(sample)
            if i % 100 == 0:
                logger.info(f'Counted flops for {i}/{len(dataset)} samples')
        flops = high.stop_counters()
        flops = round(flops[0] / (repeat * len(dataset)))
        return flops

    def max_memory(self, dataset, repeat):
        """Compute average max memory consumed by model inference. Units are MiB"""
        logger.info('Starting memory benchmarking')
        total_memory = 0
        for i, sample in enumerate(dataset):
            for _r in range(repeat):
                total_memory += max(memory_usage((self.forward, (sample,), {})))
            if i % 100 == 0:
                logger.info(f'Benchmarked memory for {i}/{len(dataset)} samples')
        total_memory = total_memory / (repeat * len(dataset))
        return total_memory

    def gather_all_metrics(self, dataset, repeat):
        run_time = self.benchmark_run_time(dataset, repeat)
        max_memory = self.max_memory(dataset, repeat)
        flops = self.count_flops(dataset, repeat)
        return run_time, max_memory, flops

    def dump_final_speech_output(self, dataset, output_dir, resample_fn, sample_rate, prefix=None):
        for i, sample in enumerate(dataset):
            hypo = self.forward(sample)[0]

            def to_np(x):
                return x.detach().cpu().numpy()
            try:
                wave_preds = to_np(resample_fn(hypo['waveform']))
                sf.write(f'{output_dir}/{prefix}_{i}_pred.wav', wave_preds, sample_rate)
            except Exception as e:
                raise Exception(f' Encountered {e} - Invalid waveform. Make sure the model outputs a waveform')


ARCH_MODEL_REGISTRY = {}


def eval_str_list(x, type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(type, x))
    except TypeError:
        return [type(x)]


def interpret_dc_type(field_type):
    if isinstance(field_type, str):
        raise RuntimeError('field should be a type')
    if field_type == Any:
        return str
    typestring = str(field_type)
    if re.match('(typing.|^)Union\\[(.*), NoneType\\]$', typestring) or typestring.startswith('typing.Optional'):
        return field_type.__args__[0]
    return field_type


def gen_parser_from_dataclass(parser: 'ArgumentParser', dataclass_instance: 'FairseqDataclass', delete_default: 'bool'=False, with_prefix: 'Optional[str]'=None) ->None:
    """
    convert a dataclass instance to tailing parser arguments.

    If `with_prefix` is provided, prefix all the keys in the resulting parser with it. It means that we are
    building a flat namespace from a structured dataclass (see transformer_config.py for example).
    """

    def argparse_name(name: 'str'):
        if name == 'data' and (with_prefix is None or with_prefix == ''):
            return name
        if name == '_name':
            return None
        full_name = '--' + name.replace('_', '-')
        if with_prefix is not None and with_prefix != '':
            full_name = with_prefix + '-' + full_name[2:]
        return full_name

    def get_kwargs_from_dc(dataclass_instance: 'FairseqDataclass', k: 'str') ->Dict[str, Any]:
        """k: dataclass attributes"""
        kwargs = {}
        field_type = dataclass_instance._get_type(k)
        inter_type = interpret_dc_type(field_type)
        field_default = dataclass_instance._get_default(k)
        if isinstance(inter_type, type) and issubclass(inter_type, Enum):
            field_choices = [t.value for t in list(inter_type)]
        else:
            field_choices = None
        field_help = dataclass_instance._get_help(k)
        field_const = dataclass_instance._get_argparse_const(k)
        if isinstance(field_default, str) and field_default.startswith('${'):
            kwargs['default'] = field_default
        else:
            if field_default is MISSING:
                kwargs['required'] = True
            if field_choices is not None:
                kwargs['choices'] = field_choices
            if isinstance(inter_type, type) and (issubclass(inter_type, List) or issubclass(inter_type, Tuple)) or ('List' in str(inter_type) or 'Tuple' in str(inter_type)):
                if 'int' in str(inter_type):
                    kwargs['type'] = lambda x: eval_str_list(x, int)
                elif 'float' in str(inter_type):
                    kwargs['type'] = lambda x: eval_str_list(x, float)
                elif 'str' in str(inter_type):
                    kwargs['type'] = lambda x: eval_str_list(x, str)
                else:
                    raise NotImplementedError('parsing of type ' + str(inter_type) + ' is not implemented')
                if field_default is not MISSING:
                    kwargs['default'] = ','.join(map(str, field_default)) if field_default is not None else None
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
        if with_prefix is not None and with_prefix != '' and field_help is not None:
            field_help = with_prefix[2:] + ': ' + field_help
        kwargs['help'] = field_help
        if field_const is not None:
            kwargs['const'] = field_const
            kwargs['nargs'] = '?'
        return kwargs
    for k in dataclass_instance._get_all_attributes():
        field_name = argparse_name(dataclass_instance._get_name(k))
        field_type = dataclass_instance._get_type(k)
        if field_name is None:
            continue
        elif inspect.isclass(field_type) and issubclass(field_type, FairseqDataclass):
            prefix = None
            if with_prefix is not None:
                prefix = field_name
            gen_parser_from_dataclass(parser, field_type(), delete_default, prefix)
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
            if delete_default and 'default' in kwargs:
                del kwargs['default']
        try:
            parser.add_argument(*field_args, **kwargs)
        except ArgumentError:
            pass


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
    def build_criterion(cls, cfg: 'FairseqDataclass', task):
        """Construct a criterion from command-line args."""
        init_args = {}
        for p in inspect.signature(cls).parameters.values():
            if p.kind == p.POSITIONAL_ONLY or p.kind == p.VAR_POSITIONAL or p.kind == p.VAR_KEYWORD:
                raise NotImplementedError('{} not supported'.format(p.kind))
            assert p.kind in {p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY}
            if p.name == 'task':
                init_args['task'] = task
            elif p.name == 'cfg':
                init_args['cfg'] = cfg
            elif hasattr(cfg, p.name):
                init_args[p.name] = getattr(cfg, p.name)
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


class FairseqOptimizer(object):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @classmethod
    def add_args(cls, parser):
        """Add optimizer-specific arguments to the parser."""
        dc = getattr(cls, '__dataclass', None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc())

    @property
    def optimizer(self):
        """Return a torch.optim.optimizer.Optimizer instance."""
        if not hasattr(self, '_optimizer'):
            raise NotImplementedError
        if not isinstance(self._optimizer, torch.optim.Optimizer):
            raise ValueError('_optimizer must be an instance of torch.optim.Optimizer')
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """Reset optimizer instance."""
        if not hasattr(self, '_optimizer'):
            raise NotImplementedError
        if not isinstance(self._optimizer, torch.optim.Optimizer):
            raise ValueError('_optimizer must be an instance of torch.optim.Optimizer')
        self._optimizer = optimizer

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        raise NotImplementedError

    @property
    def params(self):
        """Return an iterable of the parameters held by the optimizer."""
        for param_group in self.param_groups:
            for p in param_group['params']:
                yield p

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def __getstate__(self):
        return self._optimizer.__getstate__()

    def get_lr(self):
        """Return the current learning rate."""
        return self.param_groups[0]['lr']

    def set_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        """Return the optimizer's state dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        """Load an optimizer state dict.

        In general we should prefer the configuration of the existing optimizer
        instance (e.g., learning rate) over that found in the state_dict. This
        allows us to resume training from a checkpoint using a new set of
        optimizer args.
        """
        self.optimizer.load_state_dict(state_dict)
        if optimizer_overrides is not None and len(optimizer_overrides) > 0:
            for group in self.param_groups:
                group.update(optimizer_overrides)

    def backward(self, loss):
        """Computes the sum of gradients of the given tensor w.r.t. graph leaves."""
        loss.backward()

    def all_reduce_grads(self, module):
        """Manually all-reduce gradients (if required)."""
        if hasattr(module, 'all_reduce_grads'):
            module.all_reduce_grads()

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        for p in self.params:
            if p.grad is not None:
                if torch.is_tensor(c):
                    c = c
                p.grad.data.mul_(c)

    def clip_grad_norm(self, max_norm, aggregate_norm_fn=None):
        """Clips gradient norm."""
        return utils.clip_grad_norm_(self.params, max_norm, aggregate_norm_fn)

    def step(self, closure=None, scale=1.0, groups=None):
        """Performs a single optimization step."""
        if self.supports_step_with_scale:
            if self.supports_groups:
                self.optimizer.step(closure, scale=scale, groups=groups)
            else:
                self.optimizer.step(closure, scale=scale)
        else:
            if scale != 1.0:
                self.multiply_grads(1.0 / scale)
            if self.supports_groups:
                self.optimizer.step(closure, groups=groups)
            else:
                self.optimizer.step(closure)

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for p in self.params:
            p.grad = None
        self.optimizer.zero_grad()

    @property
    def supports_memory_efficient_fp16(self):
        if hasattr(self.optimizer, 'supports_memory_efficient_fp16'):
            return self.optimizer.supports_memory_efficient_fp16
        return False

    @property
    def supports_step_with_scale(self):
        if hasattr(self.optimizer, 'supports_step_with_scale'):
            return self.optimizer.supports_step_with_scale
        return False

    @property
    def supports_groups(self):
        if hasattr(self.optimizer, 'supports_groups'):
            return self.optimizer.supports_groups
        return False

    @property
    def supports_flat_params(self):
        """
        Whether the optimizer supports collapsing of the model
        parameters/gradients into a single contiguous Tensor.
        """
        if hasattr(self.optimizer, 'supports_flat_params'):
            return self.optimizer.supports_flat_params
        return False

    def average_params(self):
        pass

    def broadcast_global_state_dict(self, state_dict):
        """
        Broadcasts a global state dict to all ranks.
        Useful for optimizers that shard state between ranks.
        """
        if hasattr(self.optimizer, 'broadcast_global_state_dict'):
            return self.optimizer.broadcast_global_state_dict(state_dict)
        else:
            return state_dict


class FairseqLRScheduler(object):

    def __init__(self, cfg, optimizer):
        super().__init__()
        if optimizer is not None and not isinstance(optimizer, FairseqOptimizer):
            raise ValueError('optimizer must be an instance of FairseqOptimizer')
        self.cfg = cfg
        self.optimizer = optimizer
        self.best = None

    @classmethod
    def add_args(cls, parser):
        """Add arguments to the parser for this LR scheduler."""
        dc = getattr(cls, '__dataclass', None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc())

    def state_dict(self):
        """Return the LR scheduler state dict."""
        return {'best': self.best}

    def load_state_dict(self, state_dict):
        """Load an LR scheduler state dict."""
        self.best = state_dict['best']

    def step_begin_epoch(self, epoch):
        """Update the learning rate at the beginning of the given epoch."""
        pass

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        if val_loss is not None:
            if self.best is None:
                self.best = val_loss
            else:
                self.best = min(self.best, val_loss)

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        return self.optimizer.get_lr()


TASK_REGISTRY = {}


def _set_legacy_defaults(args, cls):
    """Helper to set default arguments based on *add_args*."""
    if not hasattr(cls, 'add_args'):
        return
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, allow_abbrev=False)
    cls.add_args(parser)
    defaults = argparse.Namespace()
    for action in parser._actions:
        if action.dest is not argparse.SUPPRESS:
            if not hasattr(defaults, action.dest):
                if action.default is not argparse.SUPPRESS:
                    setattr(defaults, action.dest, action.default)
    for key, default_value in vars(defaults).items():
        if not hasattr(args, key):
            setattr(args, key, default_value)


class omegaconf_no_object_check:

    def __init__(self):
        if hasattr(_utils, 'is_primitive_type'):
            self.old_is_primitive = _utils.is_primitive_type
        else:
            self.old_is_primitive = _utils.is_primitive_type_annotation

    def __enter__(self):
        if hasattr(_utils, 'is_primitive_type'):
            _utils.is_primitive_type = lambda _: True
        else:
            _utils.is_primitive_type_annotation = lambda _: True

    def __exit__(self, type, value, traceback):
        if hasattr(_utils, 'is_primitive_type'):
            _utils.is_primitive_type = self.old_is_primitive
        else:
            _utils.is_primitive_type_annotation = self.old_is_primitive


ARCH_MODEL_NAME_REGISTRY = {}


REGISTRIES = {}


TASK_DATACLASS_REGISTRY = {}

