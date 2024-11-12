
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


import collections


import logging


from typing import Optional


from typing import Tuple


import numpy as np


from functools import partial


import torch


from torchvision.transforms import CenterCrop


from torchvision.transforms import Compose


from torchvision.transforms import Normalize


from torchvision.transforms import Resize


from torchvision.transforms import ToTensor


import warnings


from torchvision.transforms import Lambda


from torchvision.transforms import RandomHorizontalFlip


from torchvision.transforms import RandomResizedCrop


import math


from itertools import chain


import random


import torch.nn.functional as F


import torch.utils.checkpoint


from torchvision import transforms


from typing import TYPE_CHECKING


import torch.nn as nn


import types


from copy import deepcopy


from typing import Dict


from typing import Union


from typing import Any


from typing import Callable


import copy


import enum


import inspect


import itertools


import re


from abc import ABC


from abc import abstractmethod


from collections import OrderedDict


from typing import Iterable


from typing import List


from inspect import signature


from typing import Literal


import functools


from typing import Set


from typing import Type


from torch.fx import GraphModule


from torch.fx import Node


import torch.distributed as dist


import torch.utils._pytree as pytree


from torch import SymBool


from torch import SymFloat


from torch import SymInt


from torch._decomp import core_aten_decompositions


from torch._functorch._aot_autograd.functional_utils import from_fun


from torch._functorch._aot_autograd.functional_utils import to_fun


from torch._subclasses.functional_tensor import FunctionalTensor


from torch._subclasses.functional_tensor import FunctionalTensorMode


from torch._subclasses.functional_tensor import disable_functional_mode


from torch.fx import Graph


from torch.fx import Interpreter


from torch.fx import Proxy


from torch.fx import traceback


from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode


from torch.fx.experimental.proxy_tensor import _ProxyTensor


from torch.fx.experimental.proxy_tensor import _SymNodeDict


from torch.fx.experimental.proxy_tensor import decompose


from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing


from torch.fx.experimental.proxy_tensor import fetch_object_proxy


from torch.fx.experimental.proxy_tensor import fetch_sym_proxy


from torch.fx.experimental.proxy_tensor import get_proxy_slot


from torch.fx.experimental.proxy_tensor import track_tensor_tree


from torch.fx.proxy import GraphAppendingTracer


from torch.utils.weak import WeakTensorKeyDictionary


from functools import wraps


from collections import defaultdict


from torch.fx.node import Argument


from torch.fx.node import Node


from torch.fx.node import Target


from torch.nn.intrinsic import _FusedModule


from torch.quantization.fx.graph_module import GraphModule


from torch.quantization.fx.graph_module import ObservedGraphModule


from torch.quantization.quantize_fx import Scope


from torch.quantization.quantize_fx import ScopeContextManager


from torch.quantization.quantize_fx import fuse_fx as orig_fuse_fx


from torch.quantization.quantize_fx import prepare_fx as orig_prepare_fx


from torch.quantization.quantize_fx import prepare_qat_fx as orig_prepare_qat_fx


from enum import Enum


from logging import getLogger


from torch import nn


from torch.nn import CrossEntropyLoss


from typing import DefaultDict


import time


from torch.utils.data import Dataset


from torch.utils.data import RandomSampler


from time import perf_counter_ns


from collections.abc import MutableMapping


import pandas as pd


from torch.utils.data import DataLoader


from torch.profiler import ProfilerActivity


from torch.profiler import profile


from torch.profiler import record_function


from torch.profiler import tensorboard_trace_handler


import torch.multiprocessing as mp


from torch.ao.quantization.quantize_fx import fuse_fx as orig_fuse_fx


from torch.ao.quantization.quantize_fx import prepare_fx as orig_prepare_fx


from torch.ao.quantization.quantize_fx import prepare_qat_fx as orig_prepare_qat_fx


KNOWN_ACTIVATION_ATTRIBUTES = ['hidden_act', 'activation', 'act_fn', 'activation_function']


KNOWN_NUM_LAYERS = ['num_hidden_layers', 'num_layers', 'encoder_layers', 'n_layers']


KNOWN_POS_EMB_ATTRIBUTES = ['position_embedding_type']


SUPPORTED_ACTIVATION_FUNCTIONS = ['gelu', 'relu', 'gelu_new']


USE_AT_OWN_RISK_ACTIVATION_FUNCTIONS = ['quick_gelu']


def recurse_getattr(obj, attr: 'str'):
    """
    Recursive `getattr`.

    Args:
        obj:
            A class instance holding the attribute.
        attr (`str`):
            The attribute that is to be retrieved, e.g. 'attribute1.attribute2'.
    """

    def _getattr(obj, attr):
        return getattr(obj, attr)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def recurse_setattr(module, name, value):
    """A function to recursively set attributes to a module."""
    if '.' not in name:
        setattr(module, name, value)
    else:
        name, rest = name.split('.', 1)
        recurse_setattr(getattr(module, name), rest, value)


class BetterTransformerBaseLayer:

    def __init__(self, config: "'PretrainedConfig'"):
        """
        Base layer for `BetterTransformer` integration. This class is used to wrap all the necessary
        components for the `BetterTransformer` integration.

        Args:
            config (`transformers.PretrainedConfig`):
                The config of the model.
        """
        self.norm_first = False
        self.use_gelu = False
        self.act_fn = None
        self.pos_emb_type = None
        self.num_heads = None
        self.embed_dim = None
        self.num_layers = None
        self.original_layers_mapping = {}
        self.module_mapping = None
        self.keys_to_ignore = []
        for attr in KNOWN_ACTIVATION_ATTRIBUTES:
            if hasattr(config, attr):
                self.act_fn = getattr(config, attr)
                break
        if self.act_fn is None and hasattr(self, '_get_activation_function'):
            self.act_fn = self._get_activation_function(config)
        for attr in KNOWN_POS_EMB_ATTRIBUTES:
            if hasattr(config, attr):
                self.pos_emb_type = getattr(config, attr)
                break
        for attr in KNOWN_NUM_LAYERS:
            if hasattr(config, attr):
                self.num_layers = getattr(config, attr)
                break

    def validate_bettertransformer(self):
        """
        A wrapper function to validate the `BetterTransformer` implementation. Implements most relevant checks
        that are present in: https://github.com/pytorch/pytorch/blob/0fc7de398636f4b53e6c3fde38b4e48a5ff5b37d/torch/nn/modules/transformer.py#L457-L475
        """
        if self.num_heads is None:
            raise ValueError('Number of heads not set for `BetterTransformer` integration.')
        if self.embed_dim is None:
            raise ValueError('Embedding dimension not set for `BetterTransformer` integration.')
        if self.norm2_eps is None or self.norm1_eps is None:
            raise ValueError('`norm2_eps` and `norm1_eps` not set for `BetterTransformer` integration.')
        if self.pos_emb_type is not None and self.pos_emb_type != 'absolute':
            raise ValueError(f'Positional embedding type {self.pos_emb_type} not supported for `BetterTransformer` integration')
        if self.norm1_eps != self.norm2_eps:
            raise ValueError('norm1_eps and norm2_eps must be equal for `BetterTransformer` integration.')
        if self.act_fn in USE_AT_OWN_RISK_ACTIVATION_FUNCTIONS:
            logger.warning(f'Overridding {self.act_fn} activation with gelu. Use the transformed model at your own risk, the output logits could be significantly different.')
            self.act_fn = 'gelu'
        elif self.act_fn not in SUPPORTED_ACTIVATION_FUNCTIONS:
            raise ValueError(f'Activation function {self.act_fn} not supported for `BetterTransformer` integration.')
        self.use_gelu = self.act_fn == 'gelu' or self.act_fn == 'gelu_new'
        if self.num_heads % 2 == 1:
            raise ValueError(f'Number of heads {self.num_heads} is not supported for `BetterTransformer` integration. Number of heads must be even.')

    def _revert(self, module: 'torch.nn.Module') ->torch.nn.Module:
        if self.module_mapping is not None:
            if '' in self.module_mapping.values():
                for bt_module_attr_name, value in self.module_mapping.items():
                    if value == '':
                        module = getattr(self, bt_module_attr_name)
                        return module
            else:
                raise NotImplementedError('replacing a submodule in revert is not supported')
        for modified_layer_key_names, original_layer_key_names in self.original_layers_mapping.items():
            if isinstance(original_layer_key_names, list):
                current_weight = getattr(self, modified_layer_key_names)
                split_index = current_weight.shape[0] // len(original_layer_key_names)
                for i, subparam_name in enumerate(original_layer_key_names):
                    if recurse_getattr(module, subparam_name) is None:
                        continue
                    if module not in self.keys_to_ignore:
                        parameter = current_weight[i * split_index:(i + 1) * split_index].clone()
                        if isinstance(recurse_getattr(module, subparam_name), torch.nn.Parameter):
                            parameter = torch.nn.Parameter(parameter)
                        recurse_setattr(module, subparam_name, parameter)
            elif isinstance(original_layer_key_names, str):
                if recurse_getattr(module, original_layer_key_names) is None:
                    continue
                parameter = getattr(self, modified_layer_key_names)
                if isinstance(recurse_getattr(module, original_layer_key_names), torch.nn.Parameter):
                    parameter = torch.nn.Parameter(parameter)
                recurse_setattr(module, original_layer_key_names, parameter)
            else:
                raise ValueError(f'Invalid type {type(modified_layer_key_names)} for `original_layers_mapping`', ' please use either `str` or `list`.')
        return module


def check_if_transformers_greater(target_version: 'Union[str, version.Version]') ->bool:
    """
    Checks whether the current install of transformers is greater than or equal to the target version.

    Args:
        target_version (`Union[str, packaging.version.Version]`): version used as the reference for comparison.

    Returns:
        bool: whether the check is True or not.
    """
    if isinstance(target_version, str):
        target_version = version.parse(target_version)
    return version.parse(transformers.__version__) >= target_version


def raise_on_head_mask(head_mask: 'Optional[torch.Tensor]'):
    if head_mask is not None:
        raise ValueError('layer_head_mask (or head_mask) different than None is unsupported for now with BetterTransformer, pleaseopen a PR or an issue at https://github.com/huggingface/optimum.')


def gptj_wrapped_scaled_dot_product(self, query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]'=None, head_mask: 'Optional[torch.Tensor]'=None):
    raise_on_head_mask(head_mask)
    batch_size = query.shape[0]
    mask_value = torch.finfo(value.dtype).min
    mask_value = torch.full([], mask_value, dtype=value.dtype)
    if self.downcast_qk:
        query = query
        key = key
    if batch_size == 1 and attention_mask is not None and attention_mask[0, 0, -1, -1] < -1:
        raise ValueError("BetterTransformer does not support padding='max_length' with a batch size of 1.")
    dropout_p = self.dropout_prob_attn if self.training else 0.0
    if batch_size == 1 or self.training:
        if query.shape[2] > 1:
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=True)
        else:
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=False)
    else:
        query_length, key_length = query.size(-2), key.size(-2)
        if query_length > 1:
            if not check_if_transformers_greater('4.44.99'):
                causal_mask = self.bias[:, :, key_length - query_length:key_length, :key_length]
                causal_mask = torch.where(causal_mask, 0, mask_value)
                causal_mask = causal_mask.expand(batch_size, -1, -1, -1)
                if attention_mask is not None:
                    attention_mask = causal_mask + attention_mask
            else:
                attention_mask = attention_mask[:, :, :, :key.shape[-2]]
        sdpa_result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=dropout_p, is_causal=False)
    if self.downcast_qk:
        sdpa_result = sdpa_result
    return sdpa_result, None


def gpt2_wrapped_scaled_dot_product(self, query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]'=None, head_mask: 'Optional[torch.Tensor]'=None):
    raise_on_head_mask(head_mask)
    batch_size = query.shape[0]
    mask_value = torch.finfo(value.dtype).min
    mask_value = torch.full([], mask_value, dtype=value.dtype)
    if self.downcast_qk:
        query = query
        key = key
    if batch_size == 1 and attention_mask is not None and attention_mask[0, 0, -1, -1] < -1:
        raise ValueError("BetterTransformer does not support padding='max_length' with a batch size of 1.")
    dropout_p = self.dropout_prob_attn if self.training else 0.0
    if batch_size == 1 or self.training:
        if query.shape[2] > 1:
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=True)
        else:
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=False)
    else:
        query_length, key_length = query.size(-2), key.size(-2)
        if query_length > 1:
            causal_mask = self.bias[:, :, key_length - query_length:key_length, :key_length]
            causal_mask = torch.where(causal_mask, 0, mask_value)
            causal_mask = causal_mask.expand(batch_size, -1, -1, -1)
            if attention_mask is not None:
                attention_mask = causal_mask + attention_mask
        sdpa_result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=dropout_p, is_causal=False)
    if self.downcast_qk:
        sdpa_result = sdpa_result
    return sdpa_result, None


def gpt_neo_wrapped_scaled_dot_product(self, query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]'=None, head_mask: 'Optional[torch.Tensor]'=None):
    raise_on_head_mask(head_mask)
    query = query * self.scale
    batch_size = query.shape[0]
    mask_value = torch.finfo(value.dtype).min
    mask_value = torch.full([], mask_value, dtype=value.dtype)
    if batch_size == 1 and attention_mask is not None and attention_mask[0, 0, -1, -1] < -1:
        raise ValueError("BetterTransformer does not support padding='max_length' with a batch size of 1.")
    dropout_p = self.dropout_prob_attn if self.training else 0.0
    if (batch_size == 1 or self.training) and self.attention_type == 'global':
        if query.shape[2] > 1:
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=True)
        else:
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=False)
    else:
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length:key_length, :key_length]
        causal_mask = torch.where(causal_mask, 0, mask_value)
        if batch_size > 1:
            causal_mask = causal_mask.expand(batch_size, -1, -1, -1)
        if attention_mask is not None:
            attention_mask = causal_mask + attention_mask
        sdpa_result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=dropout_p, is_causal=False)
    return sdpa_result, None


DIFFUSERS_IMPORT_ERROR = """
{0} requires the diffusers library but it was not found in your environment. You can install it with pip: `pip install
diffusers`. Please note that you may need to restart your runtime after installation.
"""


TRANSFORMERS_IMPORT_ERROR = """requires the transformers>={0} library but it was not found in your environment. You can install it with pip: `pip install
-U transformers`. Please note that you may need to restart your runtime after installation.
"""


def _is_package_available(pkg_name: 'str', return_version: 'bool'=False) ->Union[Tuple[bool, str], bool]:
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = 'N/A'
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
            package_exists = True
        except importlib.metadata.PackageNotFoundError:
            package_exists = False
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


def is_diffusers_available():
    return _diffusers_available


BACKENDS_MAPPING = OrderedDict([('diffusers', (is_diffusers_available, DIFFUSERS_IMPORT_ERROR)), ('transformers_431', (lambda : check_if_transformers_greater('4.31'), '{0} ' + TRANSFORMERS_IMPORT_ERROR.format('4.31'))), ('transformers_432', (lambda : check_if_transformers_greater('4.32'), '{0} ' + TRANSFORMERS_IMPORT_ERROR.format('4.32'))), ('transformers_434', (lambda : check_if_transformers_greater('4.34'), '{0} ' + TRANSFORMERS_IMPORT_ERROR.format('4.34')))])


def requires_backends(obj, backends):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]
    name = obj.__name__ if hasattr(obj, '__name__') else obj.__class__.__name__
    checks = (BACKENDS_MAPPING[backend] for backend in backends)
    failed = [msg.format(name) for available, msg in checks if not available()]
    if failed:
        raise ImportError(''.join(failed))


class DummyObject(type):
    """
    Metaclass for the dummy objects. Any class inheriting from it will return the ImportError generated by
    `requires_backend` each time a user tries to access any method of that class.
    """

    def __getattr__(cls, key):
        if key.startswith('_'):
            return super().__getattr__(cls, key)
        requires_backends(cls, cls._backends)


class BarkSelfAttention(metaclass=DummyObject):
    _backends = ['transformers_431']

    def __init__(self, *args, **kwargs):
        requires_backends(self, ['transformers_431'])


def bark_wrapped_scaled_dot_product(self, query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]'=None, head_mask: 'Optional[torch.Tensor]'=None):
    raise_on_head_mask(head_mask)
    is_causal = self.is_causal and query.shape[2] != 1
    sdpa_result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal)
    return sdpa_result, None


class BarkAttentionLayerBetterTransformer(BetterTransformerBaseLayer, BarkSelfAttention, nn.Module):
    _attn = bark_wrapped_scaled_dot_product

    def __init__(self, layer: "'nn.Module'", config: "'PretrainedConfig'", is_causal: 'bool'=False):
        super().__init__(config)
        is_causal = layer.is_causal
        config.dropout = layer.dropout
        config.hidden_size = layer.embed_dim
        config.num_heads = layer.num_heads
        config.bias = layer.out_proj.bias is not None
        if is_causal:
            config.block_size = layer.bias.shape[-1]
        with torch.device('meta'):
            super(BetterTransformerBaseLayer, self).__init__(config, is_causal)
        self.module_mapping = None
        submodules = ['dropout', 'attn_dropout', 'resid_dropout', 'att_proj', 'out_proj']
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))
        self.original_layers_mapping = {submodule: submodule for submodule in submodules}
        if is_causal:
            setattr(self, 'bias', getattr(layer, 'bias'))
            self.original_layers_mapping['bias'] = 'bias'
        self.supports_training = False
        self.dropout_prob_attn = float(config.dropout)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


def codegen_wrapped_scaled_dot_product(self, query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor', attention_mask: 'Optional[torch.Tensor]'=None, head_mask: 'Optional[torch.Tensor]'=None):
    raise_on_head_mask(head_mask)
    batch_size = query.shape[0]
    mask_value = torch.finfo(value.dtype).min
    mask_value = torch.full([], mask_value, dtype=value.dtype)
    if batch_size == 1 and attention_mask is not None and attention_mask[0, 0, -1, -1] < -1:
        raise ValueError("BetterTransformer does not support padding='max_length' with a batch size of 1.")
    query = query
    key = key
    dropout_p = self.dropout_prob_attn if self.training else 0.0
    if batch_size == 1 or self.training:
        if query.shape[2] > 1:
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=True)
        else:
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=False)
    else:
        query_length, key_length = query.size(-2), key.size(-2)
        if query_length > 1:
            if not check_if_transformers_greater('4.44.99'):
                causal_mask = self.causal_mask[:, :, key_length - query_length:key_length, :key_length]
                causal_mask = torch.where(causal_mask, 0, mask_value)
                causal_mask = causal_mask.expand(batch_size, -1, -1, -1)
                attention_mask = torch.min(causal_mask, attention_mask)
            else:
                attention_mask = attention_mask[:, :, :, :key.shape[-2]]
        sdpa_result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=dropout_p, is_causal=False)
    return sdpa_result, None


def opt_forward(self, hidden_states: 'torch.Tensor', key_value_states: 'Optional[torch.Tensor]'=None, past_key_value: 'Optional[Tuple[torch.Tensor]]'=None, attention_mask: 'Optional[torch.Tensor]'=None, layer_head_mask: 'Optional[torch.Tensor]'=None, output_attentions: 'bool'=False, **kwargs) ->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    raise_on_head_mask(layer_head_mask)
    if output_attentions is True:
        raise ValueError('output_attentions=True can not be supported with BetterTransformer.')
    is_cross_attention = key_value_states is not None
    batch_size, tgt_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states) * self.scaling
    if is_cross_attention and past_key_value is not None:
        key_states = past_key_value[0]
        value_states = past_key_value[1]
    elif is_cross_attention:
        key_states = self._shape(self.k_proj(key_value_states), -1, batch_size)
        value_states = self._shape(self.v_proj(key_value_states), -1, batch_size)
    elif past_key_value is not None:
        key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
        value_states = self._shape(self.v_proj(hidden_states), -1, batch_size)
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    else:
        key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
        value_states = self._shape(self.v_proj(hidden_states), -1, batch_size)
    if self.is_decoder:
        past_key_value = key_states, value_states
    query_states = self._shape(query_states, tgt_len, batch_size)
    query_states = query_states * self.scale
    dropout_p = self.dropout if self.training else 0.0
    if batch_size == 1 or self.training:
        if query_states.shape[2] > 1:
            attn_output = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=None, dropout_p=dropout_p, is_causal=True)
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=None, dropout_p=dropout_p, is_causal=False)
    else:
        attn_output = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=dropout_p, is_causal=False)
    if attn_output.size() != (batch_size, self.num_heads, tgt_len, self.head_dim):
        raise ValueError(f'`attn_output` should be of size {batch_size, self.num_heads, tgt_len, self.head_dim}, but is {attn_output.size()}')
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(batch_size, tgt_len, self.embed_dim)
    attn_output = self.out_proj(attn_output)
    return attn_output, None, past_key_value


def bart_bettertransformer_init(self, layer: "'nn.Module'", config: "'PretrainedConfig'"):
    with torch.device('meta'):
        super(BetterTransformerBaseLayer, self).__init__(layer.embed_dim, layer.num_heads, layer.dropout, layer.is_decoder, layer.k_proj.bias is not None)
    self.module_mapping = None
    submodules = ['k_proj', 'v_proj', 'q_proj', 'out_proj']
    for attr in submodules:
        setattr(self, attr, getattr(layer, attr))
    self.original_layers_mapping = {submodule: submodule for submodule in submodules}
    self.is_decoder = layer.is_decoder


def bart_forward(self, hidden_states: 'torch.Tensor', key_value_states: 'Optional[torch.Tensor]'=None, past_key_value: 'Optional[Tuple[torch.Tensor]]'=None, attention_mask: 'Optional[torch.Tensor]'=None, layer_head_mask: 'Optional[torch.Tensor]'=None, output_attentions: 'bool'=False, **kwargs) ->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""
    raise_on_head_mask(layer_head_mask)
    if output_attentions is True:
        raise ValueError('output_attentions=True can not be supported with BetterTransformer.')
    is_cross_attention = key_value_states is not None
    bsz, tgt_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    if is_cross_attention and past_key_value is not None and past_key_value[0].shape[2] == key_value_states.shape[1]:
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
    query_states = self._shape(query_states, tgt_len, bsz)
    key_states = key_states
    value_states = value_states
    attn_output = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=False)
    if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
        raise ValueError(f'`attn_output` should be of size {bsz, self.num_heads, tgt_len, self.head_dim}, but is {attn_output.size()}')
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output = self.out_proj(attn_output)
    return attn_output, None, past_key_value


class AlbertLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):

    def __init__(self, albert_layer, config):
        """
        A simple conversion of the ALBERT layer to its `BetterTransformer` implementation.

        Args:
            albert_layer (`torch.nn.Module`):
                The original ALBERT Layer where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        self.in_proj_weight = nn.Parameter(torch.cat([albert_layer.attention.query.weight, albert_layer.attention.key.weight, albert_layer.attention.value.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([albert_layer.attention.query.bias, albert_layer.attention.key.bias, albert_layer.attention.value.bias]))
        self.out_proj_weight = albert_layer.attention.dense.weight
        self.out_proj_bias = albert_layer.attention.dense.bias
        self.linear1_weight = albert_layer.ffn.weight
        self.linear1_bias = albert_layer.ffn.bias
        self.linear2_weight = albert_layer.ffn_output.weight
        self.linear2_bias = albert_layer.ffn_output.bias
        self.norm1_eps = albert_layer.attention.LayerNorm.eps
        self.norm1_weight = albert_layer.attention.LayerNorm.weight
        self.norm1_bias = albert_layer.attention.LayerNorm.bias
        self.norm2_eps = albert_layer.full_layer_layer_norm.eps
        self.norm2_weight = albert_layer.full_layer_layer_norm.weight
        self.norm2_bias = albert_layer.full_layer_layer_norm.bias
        self.num_heads = albert_layer.attention.num_attention_heads
        self.embed_dim = albert_layer.attention.all_head_size
        self.is_last_layer = False
        self.original_layers_mapping = {'in_proj_weight': ['attention.query.weight', 'attention.key.weight', 'attention.value.weight'], 'in_proj_bias': ['attention.query.bias', 'attention.key.bias', 'attention.value.bias'], 'out_proj_weight': 'attention.dense.weight', 'out_proj_bias': 'attention.dense.bias', 'linear1_weight': 'ffn.weight', 'linear1_bias': 'ffn.bias', 'linear2_weight': 'ffn_output.weight', 'linear2_bias': 'ffn_output.bias', 'norm1_eps': 'attention.LayerNorm.eps', 'norm1_weight': 'attention.LayerNorm.weight', 'norm1_bias': 'attention.LayerNorm.bias', 'norm2_eps': 'full_layer_layer_norm.eps', 'norm2_weight': 'full_layer_layer_norm.weight', 'norm2_bias': 'full_layer_layer_norm.bias'}
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.act_fn_callable = ACT2FN[self.act_fn]
        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, *_):
        if not self.training and not torch.is_autocast_enabled() and not torch.is_autocast_cpu_enabled():
            if hidden_states.is_nested:
                attention_mask = None
            if attention_mask is not None:
                attention_mask = attention_mask.bool()
                attention_mask = torch.reshape(attention_mask, (attention_mask.shape[0], attention_mask.shape[-1]))
                hidden_states = torch._nested_tensor_from_mask(hidden_states, ~attention_mask)
                attention_mask = None
            hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attention_mask)
            if hidden_states.is_nested and self.is_last_layer:
                hidden_states = hidden_states.to_padded_tensor(0.0)
        else:
            qkv = F.linear(hidden_states, weight=self.in_proj_weight, bias=self.in_proj_bias)
            qkv = qkv.view(qkv.size()[:-1] + (3, self.num_heads, self.attention_head_size)).permute(2, 0, 3, 1, 4)
            query, key, value = qkv[0], qkv[1], qkv[2]
            if self.training:
                attention_mask = None
            attention_out = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, is_causal=False, dropout_p=self.attention_probs_dropout_prob if self.training else 0.0)
            attention_out = attention_out.permute(0, 2, 1, 3).contiguous()
            new_attention_out_shape = attention_out.size()[:-2] + (self.num_heads * self.attention_head_size,)
            attention_out = attention_out.view(new_attention_out_shape)
            attention_out = F.layer_norm(F.dropout(F.linear(attention_out, self.out_proj_weight, self.out_proj_bias), p=self.hidden_dropout_prob, training=self.training) + hidden_states, normalized_shape=self.norm1_weight.shape, weight=self.norm1_weight, bias=self.norm1_bias)
            hidden_states = self.act_fn_callable(F.linear(attention_out, self.linear1_weight, self.linear1_bias))
            hidden_states = F.layer_norm(attention_out + F.dropout(F.linear(hidden_states, self.linear2_weight, self.linear2_bias), p=self.hidden_dropout_prob, training=self.training), normalized_shape=self.norm2_weight.shape, weight=self.norm2_weight, bias=self.norm2_bias)
        return hidden_states,


class BertLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):

    def __init__(self, bert_layer, config):
        """
        A simple conversion of the BERT layer to its `BetterTransformer` implementation.

        Args:
            bert_layer (`torch.nn.Module`):
                The original BERT Layer where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        self.in_proj_weight = nn.Parameter(torch.cat([bert_layer.attention.self.query.weight, bert_layer.attention.self.key.weight, bert_layer.attention.self.value.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([bert_layer.attention.self.query.bias, bert_layer.attention.self.key.bias, bert_layer.attention.self.value.bias]))
        self.out_proj_weight = bert_layer.attention.output.dense.weight
        self.out_proj_bias = bert_layer.attention.output.dense.bias
        self.linear1_weight = bert_layer.intermediate.dense.weight
        self.linear1_bias = bert_layer.intermediate.dense.bias
        self.linear2_weight = bert_layer.output.dense.weight
        self.linear2_bias = bert_layer.output.dense.bias
        self.norm1_eps = bert_layer.attention.output.LayerNorm.eps
        self.norm1_weight = bert_layer.attention.output.LayerNorm.weight
        self.norm1_bias = bert_layer.attention.output.LayerNorm.bias
        self.norm2_eps = bert_layer.output.LayerNorm.eps
        self.norm2_weight = bert_layer.output.LayerNorm.weight
        self.norm2_bias = bert_layer.output.LayerNorm.bias
        self.num_heads = bert_layer.attention.self.num_attention_heads
        self.embed_dim = bert_layer.attention.self.all_head_size
        self.is_last_layer = False
        self.original_layers_mapping = {'in_proj_weight': ['attention.self.query.weight', 'attention.self.key.weight', 'attention.self.value.weight'], 'in_proj_bias': ['attention.self.query.bias', 'attention.self.key.bias', 'attention.self.value.bias'], 'out_proj_weight': 'attention.output.dense.weight', 'out_proj_bias': 'attention.output.dense.bias', 'linear1_weight': 'intermediate.dense.weight', 'linear1_bias': 'intermediate.dense.bias', 'linear2_weight': 'output.dense.weight', 'linear2_bias': 'output.dense.bias', 'norm1_eps': 'attention.output.LayerNorm.eps', 'norm1_weight': 'attention.output.LayerNorm.weight', 'norm1_bias': 'attention.output.LayerNorm.bias', 'norm2_eps': 'output.LayerNorm.eps', 'norm2_weight': 'output.LayerNorm.weight', 'norm2_bias': 'output.LayerNorm.bias'}
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.act_fn_callable = ACT2FN[self.act_fn]
        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, *_):
        if not self.training and not torch._C._is_any_autocast_enabled():
            if hidden_states.is_nested:
                attention_mask = None
            if attention_mask is not None:
                attention_mask = attention_mask.bool()
                attention_mask = torch.reshape(attention_mask, (attention_mask.shape[0], attention_mask.shape[-1]))
                hidden_states = torch._nested_tensor_from_mask(hidden_states, ~attention_mask)
                attention_mask = None
            hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attention_mask)
            if hidden_states.is_nested and self.is_last_layer:
                hidden_states = hidden_states.to_padded_tensor(0.0)
        else:
            qkv = F.linear(hidden_states, weight=self.in_proj_weight, bias=self.in_proj_bias)
            qkv = qkv.view(qkv.size()[:-1] + (3, self.num_heads, self.attention_head_size)).permute(2, 0, 3, 1, 4)
            query, key, value = qkv[0], qkv[1], qkv[2]
            if self.training:
                attention_mask = None
            attention_out = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, is_causal=False, dropout_p=self.attention_probs_dropout_prob if self.training else 0.0)
            attention_out = attention_out.permute(0, 2, 1, 3).contiguous()
            new_attention_out_shape = attention_out.size()[:-2] + (self.num_heads * self.attention_head_size,)
            attention_out = attention_out.view(new_attention_out_shape)
            attention_out = F.layer_norm(F.dropout(F.linear(attention_out, self.out_proj_weight, self.out_proj_bias), p=self.hidden_dropout_prob, training=self.training) + hidden_states, normalized_shape=self.norm1_weight.shape, weight=self.norm1_weight, bias=self.norm1_bias)
            hidden_states = self.act_fn_callable(F.linear(attention_out, self.linear1_weight, self.linear1_bias))
            hidden_states = F.layer_norm(attention_out + F.dropout(F.linear(hidden_states, self.linear2_weight, self.linear2_bias), p=self.hidden_dropout_prob, training=self.training), normalized_shape=self.norm2_weight.shape, weight=self.norm2_weight, bias=self.norm2_bias)
        return hidden_states,


class BartEncoderLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):

    def __init__(self, bart_layer, config):
        """
        A simple conversion of the `BartEncoderLayer` to its `BetterTransformer` implementation.

        Args:
            bart_layer (`torch.nn.Module`):
                The original `BartEncoderLayer` where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        self.in_proj_weight = nn.Parameter(torch.cat([bart_layer.self_attn.q_proj.weight, bart_layer.self_attn.k_proj.weight, bart_layer.self_attn.v_proj.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([bart_layer.self_attn.q_proj.bias, bart_layer.self_attn.k_proj.bias, bart_layer.self_attn.v_proj.bias]))
        self.out_proj_weight = bart_layer.self_attn.out_proj.weight
        self.out_proj_bias = bart_layer.self_attn.out_proj.bias
        self.linear1_weight = bart_layer.fc1.weight
        self.linear1_bias = bart_layer.fc1.bias
        self.linear2_weight = bart_layer.fc2.weight
        self.linear2_bias = bart_layer.fc2.bias
        self.norm1_eps = bart_layer.self_attn_layer_norm.eps
        self.norm1_weight = bart_layer.self_attn_layer_norm.weight
        self.norm1_bias = bart_layer.self_attn_layer_norm.bias
        self.norm2_eps = bart_layer.final_layer_norm.eps
        self.norm2_weight = bart_layer.final_layer_norm.weight
        self.norm2_bias = bart_layer.final_layer_norm.bias
        self.num_heads = bart_layer.self_attn.num_heads
        self.embed_dim = bart_layer.self_attn.embed_dim
        self.is_last_layer = False
        self.original_layers_mapping = {'in_proj_weight': ['self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight'], 'in_proj_bias': ['self_attn.q_proj.bias', 'self_attn.k_proj.bias', 'self_attn.v_proj.bias'], 'out_proj_weight': 'self_attn.out_proj.weight', 'out_proj_bias': 'self_attn.out_proj.bias', 'linear1_weight': 'fc1.weight', 'linear1_bias': 'fc1.bias', 'linear2_weight': 'fc2.weight', 'linear2_bias': 'fc2.bias', 'norm1_eps': 'self_attn_layer_norm.eps', 'norm1_weight': 'self_attn_layer_norm.weight', 'norm1_bias': 'self_attn_layer_norm.bias', 'norm2_eps': 'final_layer_norm.eps', 'norm2_weight': 'final_layer_norm.weight', 'norm2_bias': 'final_layer_norm.bias'}
        self.dropout = config.attention_dropout
        self.activation_dropout = config.activation_dropout
        self.attention_head_size = config.d_model // config.encoder_attention_heads
        self.act_fn_callable = ACT2FN[self.act_fn]
        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, output_attentions: 'bool', position_bias=None, *_, **__):
        if output_attentions:
            raise ValueError('output_attentions=True can not be supported with BetterTransformer.')
        if not self.training and not torch.is_autocast_enabled() and not torch.is_autocast_cpu_enabled():
            if not hasattr(hidden_states, 'original_shape'):
                original_shape = hidden_states.shape
            else:
                original_shape = hidden_states.original_shape
            if hidden_states.is_nested:
                attention_mask = None
            if attention_mask is not None:
                if len(attention_mask.shape) == 4:
                    attention_mask = attention_mask.squeeze(1)[:, 0]
                attention_mask = attention_mask.bool()
                attention_mask = torch.reshape(attention_mask, (attention_mask.shape[0], attention_mask.shape[-1]))
                hidden_states = torch._nested_tensor_from_mask(hidden_states, ~attention_mask)
                attention_mask = None
            hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attention_mask)
            if not self.is_last_layer:
                hidden_states.original_shape = original_shape
            elif hidden_states.is_nested and self.is_last_layer:
                hidden_states = hidden_states.to_padded_tensor(0.0, original_shape)
        else:
            qkv = F.linear(hidden_states, weight=self.in_proj_weight, bias=self.in_proj_bias)
            qkv = qkv.view(qkv.size()[:-1] + (3, self.num_heads, self.attention_head_size)).permute(2, 0, 3, 1, 4)
            query, key, value = qkv[0], qkv[1], qkv[2]
            if self.training:
                attention_mask = None
            attention_out = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, is_causal=False, dropout_p=self.dropout if self.training else 0.0)
            attention_out = attention_out.permute(0, 2, 1, 3).contiguous()
            new_attention_out_shape = attention_out.size()[:-2] + (self.num_heads * self.attention_head_size,)
            attention_out = attention_out.view(new_attention_out_shape)
            attention_out = F.layer_norm(F.dropout(F.linear(attention_out, self.out_proj_weight, self.out_proj_bias), p=self.dropout, training=self.training) + hidden_states, normalized_shape=self.norm1_weight.shape, weight=self.norm1_weight, bias=self.norm1_bias)
            hidden_states = F.dropout(self.act_fn_callable(F.linear(attention_out, self.linear1_weight, self.linear1_bias)), p=self.activation_dropout, training=self.training)
            hidden_states = F.layer_norm(attention_out + F.dropout(F.linear(hidden_states, self.linear2_weight, self.linear2_bias), p=self.dropout, training=self.training), normalized_shape=self.norm2_weight.shape, weight=self.norm2_weight, bias=self.norm2_bias)
        return hidden_states,


class MBartEncoderLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):

    def __init__(self, mbart_layer, config):
        """
        A simple conversion of the `MBartEncoderLayer` to its `BetterTransformer` implementation.
        Args:
            mbart_layer (`torch.nn.Module`):
                The original `MBartEncoderLayer` where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        self.in_proj_weight = nn.Parameter(torch.cat([mbart_layer.self_attn.q_proj.weight, mbart_layer.self_attn.k_proj.weight, mbart_layer.self_attn.v_proj.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([mbart_layer.self_attn.q_proj.bias, mbart_layer.self_attn.k_proj.bias, mbart_layer.self_attn.v_proj.bias]))
        self.out_proj_weight = mbart_layer.self_attn.out_proj.weight
        self.out_proj_bias = mbart_layer.self_attn.out_proj.bias
        self.linear1_weight = mbart_layer.fc1.weight
        self.linear1_bias = mbart_layer.fc1.bias
        self.linear2_weight = mbart_layer.fc2.weight
        self.linear2_bias = mbart_layer.fc2.bias
        self.norm1_eps = mbart_layer.self_attn_layer_norm.eps
        self.norm1_weight = mbart_layer.self_attn_layer_norm.weight
        self.norm1_bias = mbart_layer.self_attn_layer_norm.bias
        self.norm2_eps = mbart_layer.final_layer_norm.eps
        self.norm2_weight = mbart_layer.final_layer_norm.weight
        self.norm2_bias = mbart_layer.final_layer_norm.bias
        self.num_heads = mbart_layer.self_attn.num_heads
        self.embed_dim = mbart_layer.self_attn.embed_dim
        self.is_last_layer = False
        self.norm_first = True
        self.original_layers_mapping = {'in_proj_weight': ['self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight'], 'in_proj_bias': ['self_attn.q_proj.bias', 'self_attn.k_proj.bias', 'self_attn.v_proj.bias'], 'out_proj_weight': 'self_attn.out_proj.weight', 'out_proj_bias': 'self_attn.out_proj.bias', 'linear1_weight': 'fc1.weight', 'linear1_bias': 'fc1.bias', 'linear2_weight': 'fc2.weight', 'linear2_bias': 'fc2.bias', 'norm1_weight': 'self_attn_layer_norm.weight', 'norm1_bias': 'self_attn_layer_norm.bias', 'norm1_eps': 'self_attn_layer_norm.eps', 'norm2_weight': 'final_layer_norm.weight', 'norm2_bias': 'final_layer_norm.bias', 'norm2_eps': 'final_layer_norm.eps'}
        self.dropout = config.attention_dropout
        self.activation_dropout = config.activation_dropout
        self.attention_head_size = config.d_model // config.encoder_attention_heads
        self.act_fn_callable = ACT2FN[self.act_fn]
        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, output_attentions: 'bool', position_bias=None, *_, **__):
        if output_attentions:
            raise ValueError('output_attentions=True can not be supported with BetterTransformer.')
        if not self.training and not torch.is_autocast_enabled() and not torch.is_autocast_cpu_enabled():
            if not hasattr(hidden_states, 'original_shape'):
                original_shape = hidden_states.shape
            else:
                original_shape = hidden_states.original_shape
            if hidden_states.is_nested:
                attention_mask = None
            if attention_mask is not None:
                if len(attention_mask.shape) == 4:
                    attention_mask = attention_mask.squeeze(1)[:, 0]
                attention_mask = attention_mask.bool()
                attention_mask = torch.reshape(attention_mask, (attention_mask.shape[0], attention_mask.shape[-1]))
                hidden_states = torch._nested_tensor_from_mask(hidden_states, ~attention_mask)
                attention_mask = None
            hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attention_mask)
            if not self.is_last_layer:
                hidden_states.original_shape = original_shape
            elif hidden_states.is_nested and self.is_last_layer:
                hidden_states = hidden_states.to_padded_tensor(0.0, original_shape)
        else:
            residual = hidden_states
            hidden_states = F.layer_norm(hidden_states, normalized_shape=self.norm1_weight.shape, weight=self.norm1_weight, bias=self.norm1_bias)
            qkv = F.linear(hidden_states, weight=self.in_proj_weight, bias=self.in_proj_bias)
            qkv = qkv.view(qkv.size()[:-1] + (3, self.num_heads, self.attention_head_size)).permute(2, 0, 3, 1, 4)
            query, key, value = qkv[0], qkv[1], qkv[2]
            if self.training:
                attention_mask = None
            attention_out = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, is_causal=False, dropout_p=self.dropout if self.training else 0.0)
            attention_out = attention_out.permute(0, 2, 1, 3).contiguous()
            new_attention_out_shape = attention_out.size()[:-2] + (self.num_heads * self.attention_head_size,)
            attention_out = attention_out.view(new_attention_out_shape)
            hidden_states = residual + F.dropout(F.linear(attention_out, self.out_proj_weight, self.out_proj_bias), p=self.dropout, training=self.training)
            residual = hidden_states
            hidden_states = F.layer_norm(hidden_states, normalized_shape=self.norm2_weight.shape, weight=self.norm2_weight, bias=self.norm2_bias)
            hidden_states = F.dropout(self.act_fn_callable(F.linear(hidden_states, self.linear1_weight, self.linear1_bias)), p=self.activation_dropout, training=self.training)
            hidden_states = residual + F.dropout(F.linear(hidden_states, self.linear2_weight, self.linear2_bias), p=self.dropout, training=self.training)
        return hidden_states,


class DistilBertLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):

    def __init__(self, bert_layer, config):
        """
        A simple conversion of the Distill-BERTLayer to its `BetterTransformer` implementation.

        Args:
            bert_layer (`torch.nn.Module`):
                The original Distill-BERT Layer where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        self.in_proj_weight = nn.Parameter(torch.cat([bert_layer.attention.q_lin.weight, bert_layer.attention.k_lin.weight, bert_layer.attention.v_lin.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([bert_layer.attention.q_lin.bias, bert_layer.attention.k_lin.bias, bert_layer.attention.v_lin.bias]))
        self.out_proj_weight = bert_layer.attention.out_lin.weight
        self.out_proj_bias = bert_layer.attention.out_lin.bias
        self.linear1_weight = bert_layer.ffn.lin1.weight
        self.linear1_bias = bert_layer.ffn.lin1.bias
        self.linear2_weight = bert_layer.ffn.lin2.weight
        self.linear2_bias = bert_layer.ffn.lin2.bias
        self.norm1_eps = bert_layer.sa_layer_norm.eps
        self.norm1_weight = bert_layer.sa_layer_norm.weight
        self.norm1_bias = bert_layer.sa_layer_norm.bias
        self.norm2_eps = bert_layer.output_layer_norm.eps
        self.norm2_weight = bert_layer.output_layer_norm.weight
        self.norm2_bias = bert_layer.output_layer_norm.bias
        self.num_heads = bert_layer.attention.n_heads
        self.embed_dim = bert_layer.attention.dim
        self.is_last_layer = False
        self.original_layers_mapping = {'in_proj_weight': ['attention.q_lin.weight', 'attention.k_lin.weight', 'attention.v_lin.weight'], 'in_proj_bias': ['attention.q_lin.bias', 'attention.k_lin.bias', 'attention.v_lin.bias'], 'out_proj_weight': 'attention.out_lin.weight', 'out_proj_bias': 'attention.out_lin.bias', 'linear1_weight': 'ffn.lin1.weight', 'linear1_bias': 'ffn.lin1.bias', 'linear2_weight': 'ffn.lin2.weight', 'linear2_bias': 'ffn.lin2.bias', 'norm1_weight': 'sa_layer_norm.weight', 'norm1_bias': 'sa_layer_norm.bias', 'norm2_weight': 'output_layer_norm.weight', 'norm2_bias': 'output_layer_norm.bias'}
        self.attention_dropout = config.attention_dropout
        self.dropout = config.dropout
        self.attention_head_size = config.dim // config.n_heads
        self.act_fn_callable = ACT2FN[self.act_fn]
        self.validate_bettertransformer()

    def forward(self, hidden_states, attn_mask, output_attentions: 'bool', head_mask=None, *_):
        if output_attentions:
            raise ValueError('output_attentions=True can not be supported with BetterTransformer.')
        if not self.training and not torch.is_autocast_enabled() and not torch.is_autocast_cpu_enabled():
            if hidden_states.is_nested:
                attn_mask = None
            if attn_mask is not None:
                attn_mask = attn_mask.bool()
                attn_mask = torch.reshape(attn_mask, (attn_mask.shape[0], attn_mask.shape[-1]))
                seqlen = attn_mask.shape[1]
                lengths = torch.sum(~attn_mask, 1)
                if not all(l == seqlen for l in lengths):
                    hidden_states = torch._nested_tensor_from_mask(hidden_states, attn_mask)
                attn_mask = None
            hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attn_mask)
            if hidden_states.is_nested and self.is_last_layer:
                hidden_states = hidden_states.to_padded_tensor(0.0)
        else:
            qkv = F.linear(hidden_states, weight=self.in_proj_weight, bias=self.in_proj_bias)
            qkv = qkv.view(qkv.size()[:-1] + (3, self.num_heads, self.attention_head_size)).permute(2, 0, 3, 1, 4)
            query, key, value = qkv[0], qkv[1], qkv[2]
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = (1.0 - attn_mask) * torch.finfo(query.dtype).min
            if self.training:
                attn_mask = None
            attention_out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, is_causal=False, dropout_p=self.attention_dropout if self.training else 0.0)
            attention_out = attention_out.permute(0, 2, 1, 3).contiguous()
            new_attention_out_shape = attention_out.size()[:-2] + (self.num_heads * self.attention_head_size,)
            attention_out = attention_out.view(new_attention_out_shape)
            attention_out = F.layer_norm(F.dropout(F.linear(attention_out, self.out_proj_weight, self.out_proj_bias), p=self.dropout, training=self.training) + hidden_states, normalized_shape=self.norm1_weight.shape, weight=self.norm1_weight, bias=self.norm1_bias)
            hidden_states = self.act_fn_callable(F.linear(attention_out, self.linear1_weight, self.linear1_bias))
            hidden_states = F.layer_norm(attention_out + F.dropout(F.linear(hidden_states, self.linear2_weight, self.linear2_bias), p=self.dropout, training=self.training), normalized_shape=self.norm2_weight.shape, weight=self.norm2_weight, bias=self.norm2_bias)
        return hidden_states,


class ViTLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):

    def __init__(self, vit_layer, config):
        """
        A simple conversion of the ViTLayer to its `BetterTransformer` implementation.

        Args:
            vit_layer (`torch.nn.Module`):
                The original `ViTLayer` where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        self.in_proj_weight = nn.Parameter(torch.cat([vit_layer.attention.attention.query.weight, vit_layer.attention.attention.key.weight, vit_layer.attention.attention.value.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([vit_layer.attention.attention.query.bias, vit_layer.attention.attention.key.bias, vit_layer.attention.attention.value.bias]))
        self.out_proj_weight = vit_layer.attention.output.dense.weight
        self.out_proj_bias = vit_layer.attention.output.dense.bias
        self.linear1_weight = vit_layer.intermediate.dense.weight
        self.linear1_bias = vit_layer.intermediate.dense.bias
        self.linear2_weight = vit_layer.output.dense.weight
        self.linear2_bias = vit_layer.output.dense.bias
        self.norm1_eps = vit_layer.layernorm_before.eps
        self.norm1_weight = vit_layer.layernorm_before.weight
        self.norm1_bias = vit_layer.layernorm_before.bias
        self.norm2_eps = vit_layer.layernorm_after.eps
        self.norm2_weight = vit_layer.layernorm_after.weight
        self.norm2_bias = vit_layer.layernorm_after.bias
        self.num_heads = vit_layer.attention.attention.num_attention_heads
        self.embed_dim = int(vit_layer.attention.attention.attention_head_size * self.num_heads)
        self.is_last_layer = False
        self.norm_first = True
        self.original_layers_mapping = {'in_proj_weight': ['attention.attention.query.weight', 'attention.attention.key.weight', 'attention.attention.value.weight'], 'in_proj_bias': ['attention.attention.query.bias', 'attention.attention.key.bias', 'attention.attention.value.bias'], 'out_proj_weight': 'attention.output.dense.weight', 'out_proj_bias': 'attention.output.dense.bias', 'linear1_weight': 'intermediate.dense.weight', 'linear1_bias': 'intermediate.dense.bias', 'linear2_weight': 'output.dense.weight', 'linear2_bias': 'output.dense.bias', 'norm1_weight': 'layernorm_before.weight', 'norm1_bias': 'layernorm_before.bias', 'norm2_weight': 'layernorm_after.weight', 'norm2_bias': 'layernorm_after.bias'}
        self.validate_bettertransformer()

    def forward(self, hidden_states, output_attentions: 'bool', *_, **__):
        if output_attentions:
            raise ValueError('output_attentions=True can not be supported with BetterTransformer.')
        if not self.training and not torch.is_autocast_enabled() and not torch.is_autocast_cpu_enabled():
            attention_mask = None
            hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attention_mask)
            if hidden_states.is_nested and self.is_last_layer:
                hidden_states = hidden_states.to_padded_tensor(0.0)
        else:
            raise NotImplementedError('Training and Autocast are not implemented for BetterTransformer + ViT. Please open an issue.')
        return hidden_states,


class ViltLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):

    def __init__(self, vilt_layer, config):
        """
        A simple conversion of the VilTLayer to its `BetterTransformer` implementation.

        Args:
            vilt_layer (`torch.nn.Module`):
                The original `VilTLayer` where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        self.in_proj_weight = nn.Parameter(torch.cat([vilt_layer.attention.attention.query.weight, vilt_layer.attention.attention.key.weight, vilt_layer.attention.attention.value.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([vilt_layer.attention.attention.query.bias, vilt_layer.attention.attention.key.bias, vilt_layer.attention.attention.value.bias]))
        self.out_proj_weight = vilt_layer.attention.output.dense.weight
        self.out_proj_bias = vilt_layer.attention.output.dense.bias
        self.linear1_weight = vilt_layer.intermediate.dense.weight
        self.linear1_bias = vilt_layer.intermediate.dense.bias
        self.linear2_weight = vilt_layer.output.dense.weight
        self.linear2_bias = vilt_layer.output.dense.bias
        self.norm1_eps = vilt_layer.layernorm_before.eps
        self.norm1_weight = vilt_layer.layernorm_before.weight
        self.norm1_bias = vilt_layer.layernorm_before.bias
        self.norm2_eps = vilt_layer.layernorm_after.eps
        self.norm2_weight = vilt_layer.layernorm_after.weight
        self.norm2_bias = vilt_layer.layernorm_after.bias
        self.num_heads = vilt_layer.attention.attention.num_attention_heads
        self.embed_dim = int(vilt_layer.attention.attention.attention_head_size * self.num_heads)
        self.is_last_layer = False
        self.norm_first = True
        self.original_layers_mapping = {'in_proj_weight': ['attention.attention.query.weight', 'attention.attention.key.weight', 'attention.attention.value.weight'], 'in_proj_bias': ['attention.attention.query.bias', 'attention.attention.key.bias', 'attention.attention.value.bias'], 'out_proj_weight': 'attention.output.dense.weight', 'out_proj_bias': 'attention.output.dense.bias', 'linear1_weight': 'intermediate.dense.weight', 'linear1_bias': 'intermediate.dense.bias', 'linear2_weight': 'output.dense.weight', 'linear2_bias': 'output.dense.bias', 'norm1_weight': 'layernorm_before.weight', 'norm1_bias': 'layernorm_before.bias', 'norm2_weight': 'layernorm_after.weight', 'norm2_bias': 'layernorm_after.bias'}
        self.validate_bettertransformer()

    def forward(self, hidden_states, layer_head_mask, output_attentions: 'bool', *_, **__):
        if output_attentions:
            raise ValueError('output_attentions=True can not be supported with BetterTransformer.')
        if not self.training and not torch.is_autocast_enabled() and not torch.is_autocast_cpu_enabled():
            attention_mask = None
            hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attention_mask)
            if hidden_states.is_nested and self.is_last_layer:
                hidden_states = hidden_states.to_padded_tensor(0.0)
        else:
            raise NotImplementedError('Training and Autocast are not implemented for BetterTransformer + Vilt. Please open an issue.')
        return hidden_states,


class Wav2Vec2EncoderLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):

    def __init__(self, wav2vec2_layer, config):
        """
        A simple conversion of the Wav2Vec2EncoderLayer to its `BetterTransformer` implementation.

        Args:
            wav2vec2_layer (`torch.nn.Module`):
                The original `Wav2Vec2EncoderLayer` where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        self.in_proj_weight = nn.Parameter(torch.cat([wav2vec2_layer.attention.q_proj.weight, wav2vec2_layer.attention.k_proj.weight, wav2vec2_layer.attention.v_proj.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([wav2vec2_layer.attention.q_proj.bias, wav2vec2_layer.attention.k_proj.bias, wav2vec2_layer.attention.v_proj.bias]))
        self.out_proj_weight = wav2vec2_layer.attention.out_proj.weight
        self.out_proj_bias = wav2vec2_layer.attention.out_proj.bias
        self.linear1_weight = wav2vec2_layer.feed_forward.intermediate_dense.weight
        self.linear1_bias = wav2vec2_layer.feed_forward.intermediate_dense.bias
        self.linear2_weight = wav2vec2_layer.feed_forward.output_dense.weight
        self.linear2_bias = wav2vec2_layer.feed_forward.output_dense.bias
        self.norm1_eps = wav2vec2_layer.layer_norm.eps
        self.norm1_weight = wav2vec2_layer.layer_norm.weight
        self.norm1_bias = wav2vec2_layer.layer_norm.bias
        self.norm2_eps = wav2vec2_layer.final_layer_norm.eps
        self.norm2_weight = wav2vec2_layer.final_layer_norm.weight
        self.norm2_bias = wav2vec2_layer.final_layer_norm.bias
        self.num_heads = wav2vec2_layer.attention.num_heads
        self.embed_dim = wav2vec2_layer.attention.embed_dim
        self.is_last_layer = False
        self.original_layers_mapping = {'in_proj_weight': ['attention.q_proj.weight', 'attention.k_proj.weight', 'attention.v_proj.weight'], 'in_proj_bias': ['attention.q_proj.bias', 'attention.k_proj.bias', 'attention.v_proj.bias'], 'out_proj_weight': 'attention.out_proj.weight', 'out_proj_bias': 'attention.out_proj.bias', 'linear1_weight': 'feed_forward.intermediate_dense.weight', 'linear1_bias': 'feed_forward.intermediate_dense.bias', 'linear2_weight': 'feed_forward.output_dense.weight', 'linear2_bias': 'feed_forward.output_dense.bias', 'norm1_weight': 'layer_norm.weight', 'norm1_bias': 'layer_norm.bias', 'norm1_eps': 'layer_norm.eps', 'norm2_weight': 'final_layer_norm.weight', 'norm2_bias': 'final_layer_norm.bias', 'norm2_eps': 'final_layer_norm.eps'}
        if config.do_stable_layer_norm:
            self.norm_first = True
        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, output_attentions: 'bool', **__):
        if output_attentions:
            raise ValueError('output_attentions=True can not be supported with BetterTransformer.')
        if not self.training and not torch.is_autocast_enabled() and not torch.is_autocast_cpu_enabled():
            if hidden_states.is_nested:
                attention_mask = None
            if attention_mask is not None:
                attention_mask = attention_mask.bool()
                if len(attention_mask.shape) == 4:
                    attention_mask = attention_mask.squeeze(1)[:, 0]
                attention_mask = torch.reshape(attention_mask, (attention_mask.shape[0], attention_mask.shape[-1]))
                hidden_states = torch._nested_tensor_from_mask(hidden_states, ~attention_mask)
                attention_mask = None
            hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attention_mask)
            if hidden_states.is_nested and self.is_last_layer:
                hidden_states = hidden_states.to_padded_tensor(0.0)
        else:
            raise NotImplementedError('Training and Autocast are not implemented for BetterTransformer + Wav2Vec2. Please open an issue.')
        return hidden_states,


class FSMTEncoderLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):

    def __init__(self, fsmt_layer, config):
        """
        A simple conversion of the FSMT Encoder layer to its `BetterTransformer` implementation.

        Args:
            fsmt_layer (`torch.nn.Module`):
                The original FSMT Layer where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        self.in_proj_weight = nn.Parameter(torch.cat([fsmt_layer.self_attn.q_proj.weight, fsmt_layer.self_attn.k_proj.weight, fsmt_layer.self_attn.v_proj.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([fsmt_layer.self_attn.q_proj.bias, fsmt_layer.self_attn.k_proj.bias, fsmt_layer.self_attn.v_proj.bias]))
        self.out_proj_weight = fsmt_layer.self_attn.out_proj.weight
        self.out_proj_bias = fsmt_layer.self_attn.out_proj.bias
        self.linear1_weight = fsmt_layer.fc1.weight
        self.linear1_bias = fsmt_layer.fc1.bias
        self.linear2_weight = fsmt_layer.fc2.weight
        self.linear2_bias = fsmt_layer.fc2.bias
        self.norm1_eps = fsmt_layer.self_attn_layer_norm.eps
        self.norm1_weight = fsmt_layer.self_attn_layer_norm.weight
        self.norm1_bias = fsmt_layer.self_attn_layer_norm.bias
        self.norm2_eps = fsmt_layer.final_layer_norm.eps
        self.norm2_weight = fsmt_layer.final_layer_norm.weight
        self.norm2_bias = fsmt_layer.final_layer_norm.bias
        self.num_heads = fsmt_layer.self_attn.num_heads
        self.embed_dim = fsmt_layer.self_attn.embed_dim
        self.is_last_layer = False
        self.original_layers_mapping = {'in_proj_weight': ['self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight'], 'in_proj_bias': ['self_attn.q_proj.bias', 'self_attn.k_proj.bias', 'self_attn.v_proj.bias'], 'out_proj_weight': 'self_attn.out_proj.weight', 'out_proj_bias': 'self_attn.out_proj.bias', 'linear1_weight': 'fc1.weight', 'linear1_bias': 'fc1.bias', 'linear2_weight': 'fc2.weight', 'linear2_bias': 'fc2.bias', 'norm1_weight': 'self_attn_layer_norm.weight', 'norm1_bias': 'self_attn_layer_norm.bias', 'norm2_weight': 'final_layer_norm.weight', 'norm2_bias': 'final_layer_norm.bias'}
        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, output_attentions: 'bool', position_bias=None, *_, **__):
        if output_attentions:
            raise ValueError('output_attentions=True can not be supported with BetterTransformer.')
        if not self.training and not torch.is_autocast_enabled() and not torch.is_autocast_cpu_enabled():
            if not hasattr(hidden_states, 'original_shape'):
                original_shape = hidden_states.shape
            else:
                original_shape = hidden_states.original_shape
            if hidden_states.is_nested:
                attention_mask = None
            if attention_mask is not None:
                attention_mask = attention_mask.bool()
                attention_mask = torch.reshape(attention_mask, (attention_mask.shape[0], attention_mask.shape[-1]))
                if hidden_states.shape[0] != attention_mask.shape[0]:
                    hidden_states = hidden_states.transpose(1, 0)
                    original_shape = hidden_states.shape
                hidden_states = torch._nested_tensor_from_mask(hidden_states, ~attention_mask)
                attention_mask = None
            hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attention_mask)
            if not self.is_last_layer:
                hidden_states.original_shape = original_shape
            elif hidden_states.is_nested and self.is_last_layer:
                hidden_states = hidden_states.to_padded_tensor(0.0, original_shape)
        else:
            raise NotImplementedError('Training and Autocast are not implemented for BetterTransformer + FSMT. Please open an issue.')
        return hidden_states, attention_mask


class ProphetNetEncoderLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):

    def __init__(self, prophetnet_layer, config):
        """
        A simple conversion of the ProphetNet Encoder layer to its `BetterTransformer` implementation.

        Args:
            prophet_net_layer (`torch.nn.Module`):
                The original ProphetNet Layer where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        self.config = config
        self.in_proj_weight = nn.Parameter(torch.cat([prophetnet_layer.self_attn.query_proj.weight, prophetnet_layer.self_attn.key_proj.weight, prophetnet_layer.self_attn.value_proj.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([prophetnet_layer.self_attn.query_proj.bias, prophetnet_layer.self_attn.key_proj.bias, prophetnet_layer.self_attn.value_proj.bias]))
        self.out_proj_weight = prophetnet_layer.self_attn.out_proj.weight
        self.out_proj_bias = prophetnet_layer.self_attn.out_proj.bias
        self.linear1_weight = prophetnet_layer.feed_forward.intermediate.weight
        self.linear1_bias = prophetnet_layer.feed_forward.intermediate.bias
        self.linear2_weight = prophetnet_layer.feed_forward.output.weight
        self.linear2_bias = prophetnet_layer.feed_forward.output.bias
        self.norm1_eps = prophetnet_layer.self_attn_layer_norm.eps
        self.norm1_weight = prophetnet_layer.self_attn_layer_norm.weight
        self.norm1_bias = prophetnet_layer.self_attn_layer_norm.bias
        self.norm2_eps = prophetnet_layer.feed_forward_layer_norm.eps
        self.norm2_weight = prophetnet_layer.feed_forward_layer_norm.weight
        self.norm2_bias = prophetnet_layer.feed_forward_layer_norm.bias
        self.num_heads = prophetnet_layer.self_attn.num_attn_heads
        self.embed_dim = prophetnet_layer.self_attn.head_dim * self.num_heads
        self.is_last_layer = False
        self.original_layers_mapping = {'in_proj_weight': ['self_attn.query_proj.weight', 'self_attn.key_proj.weight', 'self_attn.value_proj.weight'], 'in_proj_bias': ['self_attn.query_proj.bias', 'self_attn.key_proj.bias', 'self_attn.value_proj.bias'], 'out_proj_weight': 'self_attn.out_proj.weight', 'out_proj_bias': 'self_attn.out_proj.bias', 'linear1_weight': 'feed_forward.intermediate.weight', 'linear1_bias': 'feed_forward.intermediate.bias', 'linear2_weight': 'feed_forward.output.weight', 'linear2_bias': 'feed_forward.output.bias', 'norm1_weight': 'self_attn_layer_norm.weight', 'norm1_bias': 'self_attn_layer_norm.bias', 'norm2_weight': 'feed_forward_layer_norm.weight', 'norm2_bias': 'feed_forward_layer_norm.bias'}
        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, output_attentions: 'bool', *_, **__):
        if output_attentions:
            raise ValueError('output_attentions=True can not be supported with BetterTransformer.')
        if not self.training and not torch.is_autocast_enabled() and not torch.is_autocast_cpu_enabled():
            if not hasattr(hidden_states, 'original_shape'):
                original_shape = hidden_states.shape
            else:
                original_shape = hidden_states.original_shape
            if hidden_states.is_nested:
                attention_mask = None
            if attention_mask is not None:
                attention_mask = attention_mask.squeeze(1)[:, 0]
                attention_mask = attention_mask.bool()
                attention_mask = torch.reshape(attention_mask, (attention_mask.shape[0], attention_mask.shape[-1]))
                hidden_states = torch._nested_tensor_from_mask(hidden_states, ~attention_mask)
                attention_mask = None
            hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attention_mask)
            if not self.is_last_layer:
                hidden_states.original_shape = original_shape
            elif hidden_states.is_nested and self.is_last_layer:
                hidden_states = hidden_states.to_padded_tensor(0.0, original_shape)
        else:
            raise ValueError('Training and Autocast are not implemented for BetterTransformer + ProphetNet. Please open an issue.')
        return hidden_states,


class CLIPLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):

    def __init__(self, layer, config):
        """
        A simple conversion of the CLIPEncoderLayer to its `BetterTransformer` implementation.

        **The implementation is valid only for the vision model, that does not use `causal_attention_mask`.**

        Args:
            layer (`torch.nn.Module`):
                The original `CLIPEncoderLayer` where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        self.in_proj_weight = nn.Parameter(torch.cat([layer.self_attn.q_proj.weight, layer.self_attn.k_proj.weight, layer.self_attn.v_proj.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([layer.self_attn.q_proj.bias, layer.self_attn.k_proj.bias, layer.self_attn.v_proj.bias]))
        self.out_proj_weight = layer.self_attn.out_proj.weight
        self.out_proj_bias = layer.self_attn.out_proj.bias
        self.linear1_weight = layer.mlp.fc1.weight
        self.linear1_bias = layer.mlp.fc1.bias
        self.linear2_weight = layer.mlp.fc2.weight
        self.linear2_bias = layer.mlp.fc2.bias
        self.norm1_eps = layer.layer_norm1.eps
        self.norm1_weight = layer.layer_norm1.weight
        self.norm1_bias = layer.layer_norm1.bias
        self.norm2_eps = layer.layer_norm2.eps
        self.norm2_weight = layer.layer_norm2.weight
        self.norm2_bias = layer.layer_norm2.bias
        self.num_heads = layer.self_attn.num_heads
        self.embed_dim = layer.self_attn.embed_dim
        self.is_last_layer = False
        self.norm_first = True
        self.original_layers_mapping = {'in_proj_weight': ['self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight'], 'in_proj_bias': ['self_attn.q_proj.bias', 'self_attn.k_proj.bias', 'self_attn.v_proj.bias'], 'out_proj_weight': 'self_attn.out_proj.weight', 'out_proj_bias': 'self_attn.out_proj.bias', 'linear1_weight': 'mlp.fc1.weight', 'linear1_bias': 'mlp.fc1.bias', 'linear2_weight': 'mlp.fc2.weight', 'linear2_bias': 'mlp.fc2.bias', 'norm1_eps': 'layer_norm1.eps', 'norm1_weight': 'layer_norm1.weight', 'norm1_bias': 'layer_norm1.bias', 'norm2_eps': 'layer_norm2.eps', 'norm2_weight': 'layer_norm2.weight', 'norm2_bias': 'layer_norm2.bias'}
        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, causal_attention_mask, output_attentions: 'bool', *_, **__):
        if output_attentions:
            raise ValueError('output_attentions=True can not be supported with BetterTransformer.')
        if not self.training and not torch.is_autocast_enabled() and not torch.is_autocast_cpu_enabled():
            if attention_mask is not None or causal_attention_mask is not None:
                raise ValueError('Please do not use attention masks when using `BetterTransformer` converted vision models')
            hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attention_mask)
        else:
            raise NotImplementedError('Training and Autocast are not implemented for BetterTransformer + CLIP. Please open an issue.')
        return hidden_states,

    def _get_activation_function(self, config: "'PretrainedConfig'"):
        if hasattr(config, 'vision_config') and hasattr(config, 'text_config'):
            assert config.vision_config.hidden_act == config.text_config.hidden_act
            return config.vision_config.hidden_act
        else:
            return config.hidden_act


def all_reduce(group: 'dist.ProcessGroup', tensor: 'torch.Tensor') ->torch.Tensor:
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, group=group)
    return tensor


def differentiable_all_reduce_sum(tensor: 'torch.Tensor', group: 'dist.ProcessGroup') ->torch.Tensor:
    return DifferentiableAllReduceSum.apply(tensor, group)


def ensure_divisibility(numerator: 'int', denominator: 'int') ->None:
    if numerator % denominator != 0:
        raise RuntimeError(f'{numerator} is not divisible by {denominator}, check if the parallel dimension of weight parameters is divisible by parallelism level(world size of tensor parallel group)')


class VocabParallelEmbedding(nn.Module):
    """
    Embedding layer parallelized in vocabulary dimension.

    Arguments:
        ctx(`ParallelExecutionCtx`): parallel execution context which contains runtime information.
        embedding(`torch.nn.Embedding`): the original embedding module being replaced.
    """

    def __init__(self, ctx: 'ParallelExecutionCtx', embedding: 'nn.Embedding'):
        super(VocabParallelEmbedding, self).__init__()
        self.process_group = ctx.tp_group
        world_size = dist.get_world_size(self.process_group)
        tp_rank = dist.get_rank(self.process_group)
        ensure_divisibility(embedding.num_embeddings, world_size)
        num_embeddings = embedding.num_embeddings // world_size
        self.padding_idx = embedding.padding_idx
        self.max_norm = embedding.max_norm
        self.norm_type = embedding.norm_type
        self.scale_grad_by_freq = embedding.scale_grad_by_freq
        self.sparse = embedding.sparse
        self.vocab_start_idx = tp_rank * num_embeddings
        self.vocab_end_idx = (tp_rank + 1) * num_embeddings
        weight_meta = getattr(embedding.weight, 'meta', None)
        assert isinstance(weight_meta, ParameterMeta), 'should have run `initialize_parameter_meta` after moving model to current device'
        if weight_meta.is_modified_meta:
            assert weight_meta.is_tied, 'only tied parameters could already have modified meta'
        else:
            weight_meta.need_initialize = True
            weight_meta.is_parallel = True
            weight_meta.dim = 0
            for _, Slice in weight_meta.mapping.items():
                Slice.index = slice(self.vocab_start_idx, self.vocab_end_idx)
            weight_meta.is_modified_meta = True
        self.weight = embedding.weight

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        input_mask = (input < self.vocab_start_idx) | (input >= self.vocab_end_idx)
        masked_input = input.clone() - self.vocab_start_idx
        masked_input[input_mask] = 0
        output = F.embedding(masked_input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        output[input_mask, :] = 0.0
        output = differentiable_all_reduce_sum(output, self.process_group)
        return output

