
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


import copy


import logging


from typing import Any


from typing import Union


from torch.utils.data import DataLoader


from typing import Optional


import torch


import warnings


from copy import deepcopy


from typing import Sequence


import math


import re


import time


import numpy as np


import torch.nn as nn


from torch.distributed._tensor import DTensor


from torch.distributed.checkpoint.state_dict import StateDictOptions


from torch.distributed.checkpoint.state_dict import get_model_state_dict


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


from collections import deque


from typing import Mapping


from enum import Enum


from typing import Iterable


from numpy.typing import NDArray


from torch.utils.data import Dataset


from torch.utils.data import IterableDataset


from typing import Callable


import pandas as pd


import torch.distributed


from itertools import islice


from typing import Literal


from abc import ABC


from abc import abstractmethod


import inspect


from typing import cast


from torch.utils.data import DataLoader as TorchDataloader


import random


import functools


import string


from torch import Tensor


from torch.nn import functional as F


from time import sleep


from typing import TYPE_CHECKING


from torch import nn


import torch.nn.functional as F


from functools import partial


from torch.distributed import ProcessGroup


from torch.distributed._tensor import DeviceMesh


from torch.distributed._tensor import Placement


from torch.distributed._tensor import Shard


from typing import MutableMapping


from torch.distributed.nn.functional import all_gather


from functools import cached_property


from torch import distributed


from collections.abc import Sequence


from torch.optim.optimizer import Optimizer


from torch.distributed.checkpoint import LoadPlanner


from torch.distributed.checkpoint import SavePlanner


from torch.distributed.tensor.parallel.style import ParallelStyle


from torch.optim import Optimizer


from torch.distributed._tensor import Replicate


from torch.distributed.tensor.parallel import ColwiseParallel


from torch.distributed.tensor.parallel import PrepareModuleInput


from torch.distributed.tensor.parallel import RowwiseParallel


from collections import OrderedDict


from typing import ContextManager


import itertools


from torch.nn.utils.rnn import pad_sequence


from torch.distributed._tensor.api import DTensor


from collections import Counter


import torch.distributed as dist


import torch.optim as optim


from torch.distributed._tensor.device_mesh import init_device_mesh


from torch.distributed.tensor.parallel.ddp import _pre_dp_module_transform


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper


from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner


from torch.distributed.checkpoint.default_planner import DefaultSavePlanner


_attention_implementations_description = """The attention_implementations registry is used to register functions that implement the attention operation.

    One example is 'flash'. See attention.py for examples.

    Args:
        query (torch.Tensor): The query tensor.
        key (torch.Tensor): The key tensor.
        value (torch.Tensor): The value tensor.
        n_heads (int): The number of attention heads.
        kv_n_heads (int): The number of attention heads for the key and value tensors.
        past_key_value (Optional[tuple[torch.Tensor, torch.Tensor]]): The past key and value tensors.
        softmax_scale (Optional[float]) = None
        attn_bias (Optional[torch.Tensor]) = None
        is_causal (bool) = False
        dropout_p (float) = 0.0
        training (bool) = True
        needs_weights (bool) = False
        kwargs: Dict[str, Any]: Additional keyword arguments the implementation accepts.

    Returns:
        tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor, torch.Tensor]]]:
            The output tensor, the attention weights, and the past key and value tensors.
    """


def construct_from_registry(name: 'str', registry: 'TypedRegistry', partial_function: 'bool'=True, pre_validation_function: 'Optional[Union[Callable[[Any], None], type]]'=None, post_validation_function: 'Optional[Callable[[Any], None]]'=None, kwargs: 'Optional[dict[str, Any]]'=None) ->Any:
    """Helper function to build an item from the registry.

    Args:
        name (str): The name of the registered item
        registry (catalogue.Registry): The registry to fetch the item from
        partial_function (bool, optional): Whether to return a partial function for registered callables. Defaults to True.
        pre_validation_function (Optional[Union[Callable[[Any], None], type]], optional): An optional validation function called
            before constructing the item to return. This should throw an exception if validation fails. Defaults to None.
        post_validation_function (Optional[Callable[[Any], None]], optional): An optional validation function called after
            constructing the item to return. This should throw an exception if validation fails. Defaults to None.
        kwargs (Optional[Dict[str, Any]]): Other relevant keyword arguments.

    Raises:
        ValueError: If the validation functions failed or the registered item is invalid

    Returns:
        Any: The constructed item from the registry
    """
    if kwargs is None:
        kwargs = {}
    registered_constructor = registry.get(name)
    if pre_validation_function is not None:
        if isinstance(pre_validation_function, type):
            if not issubclass(registered_constructor, pre_validation_function):
                raise ValueError(f'Expected {name} to be of type {pre_validation_function}, but got {type(registered_constructor)}')
        elif isinstance(pre_validation_function, Callable):
            pre_validation_function(registered_constructor)
        else:
            raise ValueError(f'Expected pre_validation_function to be a callable or a type, but got {type(pre_validation_function)}')
    if isinstance(registered_constructor, type) or callable(registered_constructor) and not partial_function:
        constructed_item = registered_constructor(**kwargs)
    elif callable(registered_constructor):
        constructed_item = functools.partial(registered_constructor, **kwargs)
    else:
        raise ValueError(f'Expected {name} to be a class or function, but got {type(registered_constructor)}')
    if post_validation_function is not None:
        post_validation_function(constructed_item)
    return constructed_item


_fcs_description = """The fcs registry is used to register classes that implement fully connected layers (i.e. torch.nn.Linear).

    See fc.py for examples.

    Args:
        in_features: int: The number of input features.
        out_features: int: The number of output features.
        kwargs: Dict[str, Any]: Additional keyword arguments to pass to the layer.

    Returns:
        torch.nn.Module: The fully connected layer.
    """


def build_fc(name: 'str', in_features: 'int', out_features: 'int', fc_kwargs: 'dict[str, Any]'):
    kwargs = {'in_features': in_features, 'out_features': out_features, **{k: v for k, v in fc_kwargs.items() if k != 'name'}}
    return construct_from_registry(name=name, registry=fcs, pre_validation_function=torch.nn.Module, kwargs=kwargs)


_norms_description = """The norms registry is used to register classes that implement normalization layers.

    One example of this is torch.nn.LayerNorm. See norm.py for examples.

    Args:
        normalized_shape Union[int, List[int], torch.Size]: The shape of the input tensor.
        device: Optional[torch.device]: The device to use for the normalization layer.

    Returns:
        torch.nn.Module: The normalization layer.
    """


def build_norm(name: 'str', normalized_shape: 'Union[int, list[int], torch.Size]', eps: 'Optional[float]'=1e-05, device: 'Optional[str]'=None):
    kwargs = {'normalized_shape': normalized_shape, 'eps': eps, 'device': device}
    return construct_from_registry(name=name, registry=norms, pre_validation_function=torch.nn.Module, kwargs=kwargs)


def is_transformers_version_gte(hf_version: 'str') ->bool:
    return version.parse(transformers.__version__) >= version.parse(hf_version)


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) is a generalization of Multi-head (MHA).

    and Multi-query attention (MQA).

    This allows the user to set a variable of number of kv_n_heads, rather than
    just n_heads or 1, as in MHA and MQA. Using torch attention implementation
    enables user to also use additive bias. This class also supports
    cross-attention with different `in_features` for key and value fc projections.
    """

    def __init__(self, d_model: 'int', n_heads: 'int', kv_n_heads: 'int', attn_impl: 'str'='flash', clip_qkv: 'Optional[float]'=None, qk_ln: 'bool'=False, qk_gn: 'bool'=False, fused_qkv: 'bool'=True, softmax_scale: 'Optional[float]'=None, attn_pdrop: 'float'=0.0, norm_type: 'str'='low_precision_layernorm', norm_eps: 'float'=1e-05, fc_type: 'Optional[dict[str, Any]]'=None, device: 'Optional[str]'=None, bias: 'bool'=True, sliding_window_size: 'int'=-1, reuse_kv_layer_idx: 'Optional[int]'=None, attn_logit_softcapping: 'Optional[float]'=None, kv_dim: 'Optional[int]'=None):
        super().__init__()
        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.qk_ln = qk_ln
        self.qk_gn = qk_gn
        self.fused_qkv = fused_qkv
        self.d_model = d_model
        self.n_heads = n_heads
        self.kv_n_heads = kv_n_heads
        self.sliding_window_size = sliding_window_size
        self.reuse_kv_layer_idx = reuse_kv_layer_idx
        self.attn_logit_softcapping = attn_logit_softcapping
        self.kv_dim = kv_dim if kv_dim is not None else self.d_model
        self.head_dim = d_model // n_heads
        if fc_type is None:
            fc_type = copy.deepcopy(fc_type_defaults)
            fc_type['bias'] = bias
            fc_type['device'] = device
        fc_type_name = fc_type['name']
        if self.kv_n_heads <= 0:
            raise ValueError('kv_n_heads should be greater than zero.')
        if self.kv_n_heads > self.n_heads:
            raise ValueError('The number of KV heads should be less than or equal to Q heads.')
        if self.n_heads % self.kv_n_heads != 0:
            raise ValueError('Each Q head should get the same number of KV heads, so n_heads must be divisible by kv_n_heads.')
        if qk_ln and qk_gn:
            raise ValueError('Only one of qk_ln and qk_gn can be set to True.')
        self.softmax_scale = softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model / self.n_heads)
        self.attn_dropout_p = attn_pdrop
        if self.reuse_kv_layer_idx is not None:
            self.Wq = build_fc(name=fc_type_name, in_features=self.d_model, out_features=self.d_model, fc_kwargs=fc_type)
            fuse_splits = [(i * self.head_dim) for i in range(1, self.n_heads)]
            self.Wq._fused = 0, fuse_splits
        elif self.fused_qkv:
            self.Wqkv = build_fc(name=fc_type_name, in_features=self.d_model, out_features=self.d_model + 2 * self.kv_n_heads * self.head_dim, fc_kwargs=fc_type)
            fuse_splits = [(i * self.head_dim) for i in range(1, self.n_heads + 2 * self.kv_n_heads)]
            self.Wqkv._fused = 0, fuse_splits
        else:
            self.Wq = build_fc(name=fc_type_name, in_features=self.d_model, out_features=self.d_model, fc_kwargs=fc_type)
            self.Wk = build_fc(name=fc_type_name, in_features=self.kv_dim, out_features=self.kv_n_heads * self.head_dim, fc_kwargs=fc_type)
            self.Wv = build_fc(name=fc_type_name, in_features=self.kv_dim, out_features=self.kv_n_heads * self.head_dim, fc_kwargs=fc_type)
            q_fuse_splits = [(i * self.head_dim) for i in range(1, self.n_heads)]
            kv_fuse_splits = [(i * self.head_dim) for i in range(1, self.kv_n_heads)]
            self.Wq._fused = 0, q_fuse_splits
            self.Wk._fused = 0, kv_fuse_splits
            self.Wv._fused = 0, kv_fuse_splits
        if self.qk_ln or self.qk_gn:
            norm_size = self.head_dim if qk_gn else d_model
            self.q_ln = build_norm(name=norm_type.lower(), normalized_shape=norm_size, eps=norm_eps, device=device)
            if self.reuse_kv_layer_idx is None:
                if qk_ln:
                    norm_size = self.head_dim * kv_n_heads
                self.k_ln = build_norm(name=norm_type.lower(), normalized_shape=norm_size, eps=norm_eps, device=device)
        self.attn_fn = attention_implementations.get(self.attn_impl)
        self.out_proj = build_fc(name=fc_type_name, in_features=self.d_model, out_features=self.d_model, fc_kwargs=fc_type)
        self.out_proj._is_residual = True

    def forward(self, x: 'torch.Tensor', past_key_value: 'Optional[tuple[torch.Tensor, torch.Tensor]]'=None, attn_bias: 'Optional[torch.Tensor]'=None, attention_mask: 'Optional[torch.Tensor]'=None, rotary_emb_w_meta_info: 'Optional[dict]'=None, is_causal: 'bool'=True, needs_weights: 'bool'=False, alibi_slopes: 'Optional[torch.Tensor]'=None, flash_attn_padding_info: 'Optional[dict[str, torch.Tensor]]'=None, prev_layer_key_value: 'Optional[tuple[torch.Tensor, torch.Tensor]]'=None, key_value_states: 'Optional[torch.Tensor]'=None) ->tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor, torch.Tensor]]]:
        extra_kwargs = {}
        if prev_layer_key_value is not None:
            extra_kwargs['prev_layer_key_value'] = prev_layer_key_value
        query, key, value = self.get_qkv(x=x, key_value_states=key_value_states, **extra_kwargs)
        if rotary_emb_w_meta_info is not None:
            query, key, value = self._apply_rotary_embeddings(rotary_emb_w_meta_info, query, key, value)
        extra_attn_kwargs = self.get_implementation_specific_args(attention_mask, alibi_slopes, flash_attn_padding_info)
        context, attn_weights, past_key_value = self.attn_fn(query, key, value, n_heads=self.n_heads, kv_n_heads=self.kv_n_heads, past_key_value=past_key_value, softmax_scale=self.softmax_scale, attn_bias=attn_bias, is_causal=is_causal, dropout_p=self.attn_dropout_p, training=self.training, needs_weights=needs_weights, attn_logit_softcapping=self.attn_logit_softcapping, sliding_window_size=self.sliding_window_size, **extra_attn_kwargs)
        return self.out_proj(context), attn_weights, past_key_value

    def get_qkv(self, x: 'torch.Tensor', prev_layer_key_value: 'Optional[tuple[torch.Tensor, torch.Tensor]]'=None, key_value_states: 'Optional[torch.Tensor]'=None) ->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes and returns the query, key, and value tensors.

        Args:
            x (torch.Tensor): The input query tensor.
            prev_layer_key_value  (Optional[Tuple[torch.Tensor, torch.Tensor]]): The key value of the previous layer.
            key_value_states (Optional[torch.Tensor]): The input tensor for keys and values.

        Returns:
            query (torch.Tensor): The query tensor.
            key (torch.Tensor): The key tensor.
            value (torch.Tensor): The value tensor.
        """
        if self.reuse_kv_layer_idx is not None:
            if prev_layer_key_value is None:
                raise ValueError('prev_layer_key_value is None, cannot reuse_prev_layer_kv.')
            key, value = prev_layer_key_value
            if self.attn_impl == 'torch':
                key = rearrange(key, 'b h d s -> b s (h d)')
                value = rearrange(value, 'b h s d -> b s (h d)')
            query = self.Wq(x)
            if self.clip_qkv:
                query = query.clamp(min=-self.clip_qkv, max=self.clip_qkv)
            if self.qk_ln or self.qk_gn:
                q_shape = query.shape
                if self.qk_gn:
                    b, s = query.shape[:2]
                    query = query.view(b, s, self.n_heads, -1)
                dtype = query.dtype
                query = self.q_ln(query).view(q_shape)
            return query, key, value
        if self.fused_qkv:
            if key_value_states is not None:
                raise ValueError('Cannot use separate hidden and key_value states when fused_qkv = True.')
            qkv = self.Wqkv(x)
            if self.clip_qkv:
                qkv = qkv.clamp(min=-self.clip_qkv, max=self.clip_qkv)
            query, key, value = qkv.split([self.d_model, self.kv_n_heads * self.head_dim, self.kv_n_heads * self.head_dim], dim=2)
        else:
            query = self.Wq(x)
            if key_value_states is not None:
                key = self.Wk(key_value_states)
                value = self.Wv(key_value_states)
            else:
                key = self.Wk(x)
                value = self.Wv(x)
            if self.clip_qkv:
                query = query.clamp(min=-self.clip_qkv, max=self.clip_qkv)
                key = key.clamp(min=-self.clip_qkv, max=self.clip_qkv)
                value = value.clamp(min=-self.clip_qkv, max=self.clip_qkv)
        if self.qk_ln or self.qk_gn:
            q_shape, k_shape = query.shape, key.shape
            if self.qk_gn:
                b, s = query.shape[:2]
                query = query.view(b, s, self.n_heads, -1)
                key = key.view(b, s, self.kv_n_heads, -1)
            dtype = query.dtype
            query = self.q_ln(query).view(q_shape)
            key = self.k_ln(key).view(k_shape)
        return query, key, value

    def _apply_rotary_embeddings(self, rotary_emb_w_meta_info: 'dict[str, Any]', query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor') ->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.reuse_kv_layer_idx is not None:
            orig_key, orig_value = key, value
            key, value = torch.empty_like(key), torch.empty_like(value)
        rotary_emb = rotary_emb_w_meta_info['rotary_emb']
        seq_len = rotary_emb_w_meta_info['seq_len']
        offset_info = rotary_emb_w_meta_info['offset_info']
        bsz, seqlen = query.shape[:2]
        query = query.view(bsz, seqlen, -1, self.head_dim)
        key = key.view(bsz, seqlen, -1, self.head_dim)
        if rotary_emb_w_meta_info['impl'] == 'dail':
            value = value.view(bsz, seqlen, -1, self.head_dim)
            kv = torch.stack([key, value], dim=2)
            query, kv = rotary_emb(query, kv, seqlen_offset=offset_info, max_seqlen=seq_len)
            [key, value] = torch.unbind(kv, dim=2)
            value = value.view(bsz, seqlen, -1)
        elif rotary_emb_w_meta_info['impl'] == 'hf':
            if is_transformers_version_gte('4.38'):
                cos, sin = rotary_emb(x=value, position_ids=offset_info)
            else:
                cos, sin = rotary_emb(x=value, seq_len=seq_len)
            if is_transformers_version_gte('4.38'):
                cos = cos
                sin = sin
                query, key = apply_rotary_pos_emb(q=query, k=key, cos=cos, sin=sin, position_ids=None, unsqueeze_dim=2)
            elif is_transformers_version_gte('4.36'):
                query, key = apply_rotary_pos_emb(q=query, k=key, cos=cos, sin=sin, position_ids=offset_info, unsqueeze_dim=2)
            else:
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                query, key = apply_rotary_pos_emb(q=query, k=key, cos=cos, sin=sin, position_ids=offset_info)
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
        query = query.view(bsz, seqlen, -1)
        key = key.view(bsz, seqlen, -1)
        if self.reuse_kv_layer_idx is not None:
            return query, orig_key, orig_value
        return query, key, value

    def get_implementation_specific_args(self, attention_mask: 'Optional[torch.Tensor]'=None, alibi_slopes: 'Optional[torch.Tensor]'=None, flash_attn_padding_info: 'Optional[dict[str, torch.Tensor]]'=None) ->dict[str, Any]:
        """Returns attention implementation specific args.

        Args:
            attention_mask (Optional[torch.Tensor]): The attention mask.
            alibi_slopes (Optional[torch.Tensor]): The alibi slopes.
            flash_attn_padding_info (Optional[dict[str, torch.Tensor]]): The padding information, only required for flash attention.

        Returns:
            extra_attn_kwargs (dict[str, Any]): Implementation specific args.
        """
        if self.attn_impl == 'flash':
            extra_attn_kwargs = {'should_repeat_kv_for_gqa': not is_flash_v2_installed(), 'alibi_slopes': alibi_slopes, 'flash_attn_padding_info': flash_attn_padding_info, 'key_padding_mask': None}
        else:
            extra_attn_kwargs = {'key_padding_mask': attention_mask}
        return extra_attn_kwargs


class MultiheadAttention(GroupedQueryAttention):
    """Multi-head self attention.

    Using torch attention implementation enables user to also use additive bias.
    """

    def __init__(self, d_model: 'int', n_heads: 'int', attn_impl: 'str'='flash', clip_qkv: 'Optional[float]'=None, qk_ln: 'bool'=False, qk_gn: 'bool'=False, fused_qkv: 'bool'=True, softmax_scale: 'Optional[float]'=None, attn_pdrop: 'float'=0.0, norm_type: 'str'='low_precision_layernorm', norm_eps: 'float'=1e-05, fc_type: 'Optional[dict[str, Any]]'=None, device: 'Optional[str]'=None, bias: 'bool'=True, sliding_window_size: 'int'=-1, reuse_kv_layer_idx: 'Optional[int]'=None, attn_logit_softcapping: 'Optional[float]'=None, kv_dim: 'Optional[int]'=None):
        super().__init__(d_model=d_model, n_heads=n_heads, kv_n_heads=n_heads, attn_impl=attn_impl, clip_qkv=clip_qkv, qk_ln=qk_ln, qk_gn=qk_gn, fused_qkv=fused_qkv, softmax_scale=softmax_scale, attn_pdrop=attn_pdrop, norm_type=norm_type, norm_eps=norm_eps, fc_type=fc_type, device=device, bias=bias, sliding_window_size=sliding_window_size, reuse_kv_layer_idx=reuse_kv_layer_idx, attn_logit_softcapping=attn_logit_softcapping, kv_dim=kv_dim)


class MultiQueryAttention(GroupedQueryAttention):
    """Multi-Query self attention.

    Using torch attention implementation enables user to also use additive bias.
    """

    def __init__(self, d_model: 'int', n_heads: 'int', attn_impl: 'str'='flash', clip_qkv: 'Optional[float]'=None, qk_ln: 'bool'=False, qk_gn: 'bool'=False, fused_qkv: 'bool'=True, softmax_scale: 'Optional[float]'=None, attn_pdrop: 'float'=0.0, norm_type: 'str'='low_precision_layernorm', norm_eps: 'float'=1e-05, fc_type: 'Optional[dict[str, Any]]'=None, device: 'Optional[str]'=None, bias: 'bool'=True, sliding_window_size: 'int'=-1, reuse_kv_layer_idx: 'Optional[int]'=None, attn_logit_softcapping: 'Optional[float]'=None, kv_dim: 'Optional[int]'=None):
        super().__init__(d_model=d_model, n_heads=n_heads, kv_n_heads=1, attn_impl=attn_impl, clip_qkv=clip_qkv, qk_ln=qk_ln, qk_gn=qk_gn, fused_qkv=fused_qkv, softmax_scale=softmax_scale, attn_pdrop=attn_pdrop, norm_type=norm_type, norm_eps=norm_eps, fc_type=fc_type, device=device, bias=bias, sliding_window_size=sliding_window_size, reuse_kv_layer_idx=reuse_kv_layer_idx, attn_logit_softcapping=attn_logit_softcapping, kv_dim=kv_dim)


_attention_classes_description = """The attention_classes registry is used to register classes that implement attention layers.

    The kwargs are passed directly to the constructor of the class.
    One example is GroupedQueryAttention. See attention.py for examples.

    Args:
        kwargs: Dict[str, Any]: Additional keyword arguments to pass to the layer.

    Returns:
        torch.nn.Module: The attention layer.
    """


def build_attention_layer(name: 'str', attn_kwargs: 'dict[str, Any]'):
    return construct_from_registry(name=name, registry=attention_classes, pre_validation_function=torch.nn.Module, kwargs=attn_kwargs)


class FusedNormAttentionNorm(nn.Module):

    def __init__(self, d_model: 'int', n_heads: 'int', args_to_exclude_in_attn_class: 'set[str]', attn_config: 'Optional[dict]'=None, ffn_has_norm: 'bool'=False, fc_type: 'Optional[dict[str, Any]]'=None, resid_pdrop: 'float'=0.0, norm_type: 'str'='low_precision_layernorm', norm_eps: 'float'=1e-05, device: 'Optional[str]'=None, no_bias: 'bool'=False, **kwargs: Any):
        super().__init__()
        assert attn_config is not None
        assert isinstance(attn_config['attn_type'], str)
        if fc_type is None:
            fc_type = copy.deepcopy(fc_type_defaults)
            fc_type['bias'] = not no_bias
            fc_type['device'] = device
        attn_config_subset_for_attn_class = {k: v for k, v in attn_config.items() if k not in args_to_exclude_in_attn_class}
        self.norm_1 = build_norm(name=norm_type.lower(), normalized_shape=d_model, eps=norm_eps, device=device)
        self.attn = build_attention_layer(name=attn_config['attn_type'], attn_kwargs={'d_model': d_model, 'n_heads': n_heads, 'fc_type': fc_type, 'device': device, 'bias': not no_bias, **attn_config_subset_for_attn_class})
        self.norm_2 = None
        if not ffn_has_norm:
            self.norm_2 = build_norm(name=norm_type.lower(), normalized_shape=d_model, eps=norm_eps, device=device)
        self.resid_attn_dropout = nn.Dropout(resid_pdrop)

    def forward(self, x: 'torch.Tensor', past_key_value: 'Optional[tuple[torch.Tensor, torch.Tensor]]'=None, attn_bias: 'Optional[torch.Tensor]'=None, rotary_emb_w_meta_info: 'Optional[dict]'=None, attention_mask: 'Optional[torch.ByteTensor]'=None, is_causal: 'bool'=True, output_attentions: 'bool'=False, alibi_slopes: 'Optional[torch.Tensor]'=None, flash_attn_padding_info: 'Optional[dict[str, torch.Tensor]]'=None, prev_layer_key_value: 'Optional[tuple[torch.Tensor, torch.Tensor]]'=None, key_value_states: 'Optional[torch.Tensor]'=None) ->tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor, torch.Tensor]]]:
        a = self.norm_1(x)
        extra_kwargs = {}
        if prev_layer_key_value is not None:
            extra_kwargs['prev_layer_key_value'] = prev_layer_key_value
        if key_value_states is not None:
            extra_kwargs['key_value_states'] = key_value_states
        b, attn_weights, past_key_value = self.attn(a, past_key_value=past_key_value, attn_bias=attn_bias, rotary_emb_w_meta_info=rotary_emb_w_meta_info, attention_mask=attention_mask, is_causal=is_causal, needs_weights=output_attentions, alibi_slopes=alibi_slopes, flash_attn_padding_info=flash_attn_padding_info, **extra_kwargs)
        x = x + self.resid_attn_dropout(b)
        m = x
        if self.norm_2 is not None:
            m = self.norm_2(x)
        return x, m, attn_weights, past_key_value


_ffns_description = """The ffns registry is used to register functions that build FFN layers.

    These layers are generally composed of fc layers and activation functions.
    One example is MPTMLP. See ffn.py for examples.

    Args:
        d_model: int: The size of the input and output tensors.
        expansion_ratio: float: The expansion ratio for the hidden layer.
        device: Optional[str]: The device to use for the layer.
        bias: bool: Whether or not to include a bias term.
        kwargs: Dict[str, Any]: Additional keyword arguments to pass to the layer.

    Returns:
        torch.nn.Module: The FFN layer.
    """


_ffns_with_megablocks_description = """The ffns_with_megablocks registry is used to register functions that build FFN layers using MegaBlocks.

    The resulting layer will have ._uses_megablocks set on it.
    One example is megablocks.layers.dmoe.dMoE. See ffn.py for examples.

    Returns:
        torch.nn.Module: The FFN layer.
    """


_ffns_with_norm_description = """The ffns_with_norm registry is used to register functions that build FFN layers with normalization.

    The resulting layer will have ._has_norm set on it.
    One example is te.LayerNormMLP. See ffn.py for examples.

    Args:
        d_model: int: The size of the input and output tensors.
        expansion_ratio: float: The expansion ratio for the hidden layer.
        device: Optional[str]: The device to use for the layer.
        bias: bool: Whether or not to include a bias term.
        kwargs: Dict[str, Any]: Additional keyword arguments to pass to the layer.

    Returns:
        torch.nn.Module: The FFN layer.
    """


def build_ffn(name: 'str', d_model: 'int', expansion_ratio: 'float', device: 'Optional[str]', bias: 'bool', ffn_kwargs: 'dict[str, Any]'):
    registry_to_use = ffns
    if name in ffns_with_norm:
        registry_to_use = ffns_with_norm
    if name in ffns_with_megablocks:
        registry_to_use = ffns_with_megablocks
    kwargs = {'d_model': d_model, 'expansion_ratio': expansion_ratio, 'device': device, 'bias': bias, **{k: v for k, v in ffn_kwargs.items() if k != 'ffn_type'}}

    def _validation_function(maybe_module: 'Any'):
        if not isinstance(maybe_module, torch.nn.Module):
            raise ValueError(f'Function {name} must return a torch.nn.Module.')
    result = construct_from_registry(name=name, registry=registry_to_use, post_validation_function=_validation_function, partial_function=False, kwargs=kwargs)
    if name in ffns_with_norm:
        result._has_norm = True
    if name in ffns_with_megablocks:
        result._uses_megablocks = True
    return result


class MPTBlock(nn.Module):

    def __init__(self, d_model: 'int', n_heads: 'int', expansion_ratio: 'int', attn_config: 'Optional[dict]'=None, ffn_config: 'Optional[dict]'=None, resid_pdrop: 'float'=0.0, norm_type: 'str'='low_precision_layernorm', norm_eps: 'float'=1e-05, fc_type: 'Optional[dict[str, Any]]'=None, device: 'Optional[str]'=None, no_bias: 'bool'=False, use_pad_tok_in_ffn: 'bool'=True, **kwargs: Any):
        if attn_config is None:
            attn_config = attn_config_defaults
        if ffn_config is None:
            self.ffn_config: 'dict[str, Any]' = {'ffn_type': 'mptmlp'}
        else:
            self.ffn_config = ffn_config
        if fc_type is None:
            fc_type = copy.deepcopy(fc_type_defaults)
        fc_type['bias'] = not no_bias
        fc_type['device'] = device
        self.ffn_config['fc_type'] = fc_type
        self.fuse_norm_attn_norm = kwargs.get('fuse_norm_attn_norm', False)
        del kwargs
        super().__init__()
        ffn_type = self.ffn_config['ffn_type']
        ffn_has_norm = ffn_type in ffns_with_norm
        if self.fuse_norm_attn_norm:
            self.norm_attn_norm = FusedNormAttentionNorm(d_model=d_model, n_heads=n_heads, args_to_exclude_in_attn_class=self.args_to_exclude_in_attn_class, attn_config=attn_config, ffn_has_norm=ffn_has_norm, fc_type=fc_type, resid_pdrop=resid_pdrop, norm_type=norm_type, norm_eps=norm_eps, device=device, no_bias=no_bias)
        else:
            assert isinstance(attn_config['attn_type'], str)
            attn_config_subset_for_attn_class = {k: v for k, v in attn_config.items() if k not in self.args_to_exclude_in_attn_class}
            self.norm_1 = build_norm(name=norm_type.lower(), normalized_shape=d_model, eps=norm_eps, device=device)
            self.attn = build_attention_layer(name=attn_config['attn_type'], attn_kwargs={'d_model': d_model, 'n_heads': n_heads, 'fc_type': fc_type, 'device': device, 'bias': not no_bias, **attn_config_subset_for_attn_class})
            self.norm_2 = None
            if not ffn_has_norm:
                self.norm_2 = build_norm(name=norm_type.lower(), normalized_shape=d_model, eps=norm_eps, device=device)
        self.ffn = build_ffn(name=ffn_type, d_model=d_model, expansion_ratio=expansion_ratio, device=device, bias=not no_bias, ffn_kwargs=self.ffn_config)
        self.resid_attn_dropout = nn.Dropout(resid_pdrop)
        self.resid_ffn_dropout = nn.Dropout(resid_pdrop)
        self.use_pad_tok_in_ffn = use_pad_tok_in_ffn

    @property
    def args_to_exclude_in_attn_class(self):
        return {'attn_type', 'alibi', 'attn_uses_sequence_id', 'alibi_bias_max', 'rope', 'rope_theta', 'rope_impl', 'rope_dail_config', 'rope_hf_config'}

    def forward(self, x: 'torch.Tensor', past_key_value: 'Optional[tuple[torch.Tensor, torch.Tensor]]'=None, attn_bias: 'Optional[torch.Tensor]'=None, rotary_emb_w_meta_info: 'Optional[dict]'=None, attention_mask: 'Optional[torch.ByteTensor]'=None, is_causal: 'bool'=True, output_attentions: 'bool'=False, alibi_slopes: 'Optional[torch.Tensor]'=None, flash_attn_padding_info: 'Optional[dict[str, torch.Tensor]]'=None, prev_layer_key_value: 'Optional[tuple[torch.Tensor, torch.Tensor]]'=None, key_value_states: 'Optional[torch.Tensor]'=None) ->tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor, torch.Tensor]]]:
        extra_kwargs = {}
        if prev_layer_key_value is not None:
            extra_kwargs['prev_layer_key_value'] = prev_layer_key_value
        if key_value_states is not None:
            extra_kwargs['key_value_states'] = key_value_states
        if self.fuse_norm_attn_norm:
            x, m, attn_weights, past_key_value = self.norm_attn_norm(x, past_key_value=past_key_value, attn_bias=attn_bias, rotary_emb_w_meta_info=rotary_emb_w_meta_info, attention_mask=attention_mask, is_causal=is_causal, output_attentions=output_attentions, alibi_slopes=alibi_slopes, flash_attn_padding_info=flash_attn_padding_info, **extra_kwargs)
        else:
            a = self.norm_1(x)
            b, attn_weights, past_key_value = self.attn(a, past_key_value=past_key_value, attn_bias=attn_bias, rotary_emb_w_meta_info=rotary_emb_w_meta_info, attention_mask=attention_mask, is_causal=is_causal, needs_weights=output_attentions, alibi_slopes=alibi_slopes, flash_attn_padding_info=flash_attn_padding_info, **extra_kwargs)
            x = x + self.resid_attn_dropout(b)
            m = x
            if self.norm_2 is not None:
                m = self.norm_2(x)
        n = self.apply_ffn(attention_mask, m)
        x = x + self.resid_ffn_dropout(n)
        return x, attn_weights, past_key_value

    def apply_ffn(self, attention_mask: 'Optional[torch.ByteTensor]', m: 'torch.Tensor') ->torch.Tensor:
        """Apply feed forward layers to the input.

        Args:
            attention_mask (Optional[torch.ByteTensor]): The attention mask.
            m (torch.Tensor): The input.

        Returns:
            n (torch.Tensor): The output.
        """
        batch_size, seq_len = m.size()[:2]
        indices = None
        if not self.use_pad_tok_in_ffn and attention_mask is not None:
            assert unpad_input is not None
            attention_mask = self.slice_attention_mask(attention_mask, seq_len)
            m, indices, _, _ = unpad_input(m, attention_mask)
        n = self.ffn(m)
        if not self.use_pad_tok_in_ffn and attention_mask is not None:
            assert pad_input is not None
            n = pad_input(n, indices, batch_size, seq_len)
        return n

    def slice_attention_mask(self, attention_mask: 'torch.ByteTensor', seq_len: 'int') ->torch.ByteTensor:
        """Slice attention mask to the correct size.

        Can be overridden by subclasses to apply different slicing logic.

        Args:
            attention_mask (torch.ByteTensor): The attention mask.
            seq_len (int): The sequence length.

        Returns:
            torch.ByteTensor: The sliced attention mask.
        """
        return attention_mask


class SharedEmbedding(nn.Embedding):

    def forward(self, input: 'Tensor', unembed: 'bool'=False) ->Tensor:
        if unembed:
            return F.linear(input, self.weight)
        return super().forward(input)


class _UniformExpertAssignment(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: 'torch.Tensor', num_experts: 'int'):
        out = torch.arange(x.numel(), dtype=x.dtype, device=x.device)
        out = torch.remainder(out, num_experts)
        return out.view(x.shape)


class LearnedRouter(torch.nn.Module):

    def __init__(self, hidden_size: 'int', moe_num_experts: 'int', moe_top_k: 'int', moe_jitter_eps: 'Optional[float]', moe_normalize_expert_weights: 'Optional[Union[int, float]]', uniform_expert_assignment: 'bool', device: 'Optional[torch.device]') ->None:
        super().__init__()
        self.hidden_size: 'int' = hidden_size
        self.moe_num_experts: 'int' = moe_num_experts
        self.moe_top_k: 'int' = moe_top_k
        self.moe_jitter_eps: 'Optional[float]' = moe_jitter_eps
        self.moe_normalize_expert_weights: 'Optional[Union[int, float]]' = moe_normalize_expert_weights
        self.uniform_expert_assignment: 'bool' = uniform_expert_assignment
        self.layer: 'torch.nn.Module' = torch.nn.Linear(hidden_size, moe_num_experts, bias=False, device=device)

    def jitter(self, x: 'torch.Tensor') ->torch.Tensor:
        assert self.moe_jitter_eps is not None
        low: 'float' = 1.0 - self.moe_jitter_eps
        high: 'float' = 1.0 + self.moe_jitter_eps
        noise: 'torch.Tensor' = torch.rand(x.size(), dtype=x.dtype, device=x.device)
        return low + noise * (high - low)

    def _top_k(self, scores: 'torch.Tensor') ->tuple[torch.Tensor, torch.Tensor]:
        if self.moe_top_k == 1:
            values, indices = scores.max(dim=-1)
            return values.unsqueeze(-1), indices.unsqueeze(-1)
        return torch.topk(scores, self.moe_top_k, dim=-1)

    def forward(self, x: 'torch.Tensor'):
        if self.training and self.moe_jitter_eps is not None:
            x = x * self.jitter(x)
        scores = self.layer(x.view(-1, x.shape[-1])).softmax(dim=-1)
        expert_weights, top_experts = self._top_k(scores)
        if self.moe_normalize_expert_weights:
            expert_weights = expert_weights / torch.norm(expert_weights, p=self.moe_normalize_expert_weights, dim=-1, keepdim=True)
        top_experts = _UniformExpertAssignment.apply(top_experts, self.moe_num_experts) if self.uniform_expert_assignment else top_experts
        scores = scores
        expert_weights = expert_weights
        return scores, expert_weights, top_experts


class MLP(nn.Module):

    def __init__(self, cfg: 'Union[ListConfig, DictConfig]'):
        super().__init__()
        self.fc1 = nn.Linear(cfg.in_features, cfg.out_features, bias=True)
        self.ln_1 = nn.LayerNorm(cfg.out_features)
        self.fc2 = nn.Linear(cfg.out_features, cfg.out_features, bias=True)
        self.fc2._is_residual = True

    def forward(self, x: 'torch.Tensor'):
        y = self.ln_1(self.fc1(x))
        res = y
        y = self.fc2(y)
        y = y + res
        return y


class GLU(torch.nn.Module):

    def __init__(self, hidden_size: 'int', ffn_hidden_size: 'int', moe_num_experts: 'int', activation_fn: 'Callable', device: 'Optional[torch.device]'):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.moe_num_experts = moe_num_experts
        self.w1 = torch.nn.Parameter(torch.rand(moe_num_experts * ffn_hidden_size, hidden_size, device=device))
        self.v1 = torch.nn.Parameter(torch.rand(moe_num_experts * ffn_hidden_size, hidden_size, device=device))
        self.w2 = torch.nn.Parameter(torch.rand(moe_num_experts * ffn_hidden_size, hidden_size, device=device))
        self.activation_fn = activation_fn

    def forward(self, x: 'torch.Tensor', expert_idx: 'torch.Tensor'):
        expert_w1 = self.w1.view(self.moe_num_experts, self.ffn_hidden_size, self.hidden_size)[expert_idx]
        expert_v1 = self.v1.view(self.moe_num_experts, self.ffn_hidden_size, self.hidden_size)[expert_idx]
        expert_w2 = self.w2.view(self.moe_num_experts, self.ffn_hidden_size, self.hidden_size)[expert_idx]
        x1 = x.matmul(expert_w1.t())
        x2 = x.matmul(expert_v1.t())
        x1 = self.activation_fn(x1)
        x1 = x1 * x2
        x1 = x1.matmul(expert_w2)
        return x1


class DroplessMLP(torch.nn.Module):

    def __init__(self, hidden_size: 'int', ffn_hidden_size: 'int', mlp_type: 'str', moe_num_experts: 'int', activation_fn: 'Callable', bias: 'bool', device: 'Optional[torch.device]'):
        super().__init__()
        self.moe_num_experts = moe_num_experts
        if mlp_type == 'mlp':
            self.mlp = MLP(hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size, moe_num_experts=moe_num_experts, activation_fn=activation_fn, device=device)
        elif mlp_type == 'glu':
            self.mlp = GLU(hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size, moe_num_experts=moe_num_experts, activation_fn=activation_fn, device=device)
        else:
            raise ValueError(f'Received unknown mlp_type={mlp_type!r}')

    def forward(self, x: 'torch.Tensor', scores: 'torch.Tensor', expert_weights: 'torch.Tensor', top_experts: 'torch.Tensor'):
        in_shape = x.shape
        hidden_size = in_shape[-1]
        x = x.view(-1, hidden_size)
        out = torch.zeros_like(x)
        expert_mask = torch.nn.functional.one_hot(top_experts, num_classes=self.moe_num_experts).permute(2, 1, 0)
        for expert_idx in range(0, self.moe_num_experts):
            topk_idx, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.shape[0] == 0:
                continue
            token_list = token_idx.tolist()
            topk_list = topk_idx.tolist()
            expert_tokens = x[None, token_list].reshape(-1, hidden_size)
            mlp_output = self.mlp(expert_tokens, expert_idx)
            expert_weights = expert_weights
            expert_out = mlp_output * expert_weights[token_list, topk_list, None]
            out = out
            token_idx = token_idx
            out.index_add_(0, token_idx, expert_out)
        out = out.view(in_shape)
        return out


DEFAULT_ACTIVATION_FN = partial(F.gelu, approximate='tanh')


class dMoE(torch.nn.Module):

    def __init__(self, device: 'Optional[torch.device]', hidden_size: 'int'=1024, ffn_hidden_size: 'int'=4096, moe_num_experts: 'int'=1, moe_top_k: 'int'=1, mlp_type: 'str'='mlp', activation_fn: 'Callable'=DEFAULT_ACTIVATION_FN, moe_jitter_eps: 'Optional[float]'=None, moe_normalize_expert_weights: 'Optional[Union[int, float]]'=None, uniform_expert_assignment: 'bool'=False, bias: 'bool'=True):
        super().__init__()
        self.router = LearnedRouter(hidden_size, moe_num_experts=moe_num_experts, moe_top_k=moe_top_k, moe_jitter_eps=moe_jitter_eps, moe_normalize_expert_weights=moe_normalize_expert_weights, uniform_expert_assignment=uniform_expert_assignment, device=device)
        self.experts = DroplessMLP(hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size, mlp_type=mlp_type, moe_num_experts=moe_num_experts, activation_fn=activation_fn, bias=bias, device=device)

    def forward(self, x: 'torch.Tensor'):
        scores, expert_weights, top_experts = self.router(x)
        return self.experts(x, scores, expert_weights, top_experts)


_FFN_ACT_FN_DEFAULT = {'name': 'gelu', 'approximate': 'none'}


def quickgelu_activation(input: 'torch.Tensor') ->torch.Tensor:
    """Applies GELU approximation that is fast but somewhat inaccurate.

    Args:
        input (torch.Tensor): Input tensor of shape(*), where * means any
            number of dimensions

    Returns:
        torch.Tensor: Tensor with same shape as input tensor
    """
    return input * torch.sigmoid(1.702 * input)


def resolve_ffn_act_fn(config: 'Optional[dict]'=None) ->Callable[[torch.Tensor], torch.Tensor]:
    """Resolve the activation function for the feed-forward network.

    Args:
        config (Optional[dict]): The configuration dictionary for the activation function.
            The dict config must specify the 'name' of a torch.nn.functional activation
            function. All of other key values pairs are bound to the function as a partial.

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: The activation function.
    """
    if config is None:
        config = _FFN_ACT_FN_DEFAULT
    config = deepcopy(config)
    name = config.pop('name')
    if name == 'quick_gelu':
        return quickgelu_activation
    else:
        if not hasattr(torch.nn.functional, name):
            raise ValueError(f'Unrecognized activation function name ({name}).')
        act = getattr(torch.nn.functional, name)
        return partial(act, **config)


_DEFAULT_ACT_FN = resolve_ffn_act_fn(_FFN_ACT_FN_DEFAULT)


log = logging.getLogger(__name__)


def resolve_ffn_hidden_size(d_model: 'int', expansion_ratio: 'Union[int, float]', ffn_hidden_size: 'Optional[int]'=None) ->int:
    """Resolve the hidden size of the feed-forward network.

    Args:
        d_model (int): The dimension of the input and output of the feed-forward network.
        expansion_ratio (Union[int, float]): The expansion ratio of the feed-forward network.
        ffn_hidden_size (Optional[int]): The hidden size of the feed-forward network.

    Returns:
        int: The hidden size of the feed-forward network.
    """
    if ffn_hidden_size is not None:
        log.info(f'`expansion_ratio` (={expansion_ratio}) ignored when `ffn_hidden_size` (={ffn_hidden_size}) is specified.')
    else:
        ffn_hidden_size = int(d_model * expansion_ratio)
        if ffn_hidden_size != d_model * expansion_ratio:
            raise ValueError(f'`d_model * expansion_ratio` must be an integer (d_model={d_model!r}; expansion_ratio={expansion_ratio!r}; d_model * expansion_ratio={d_model * expansion_ratio!r}).')
    return ffn_hidden_size


class MPTMLP(nn.Module):

    def __init__(self, d_model: 'int', expansion_ratio: 'Union[int, float]', fc_type: 'Optional[dict[str, Any]]'=None, ffn_hidden_size: 'Optional[int]'=None, act_fn: 'Callable[[torch.Tensor], torch.Tensor]'=_DEFAULT_ACT_FN, device: 'Optional[str]'=None, bias: 'bool'=True):
        super().__init__()
        ffn_hidden_size = resolve_ffn_hidden_size(d_model, expansion_ratio, ffn_hidden_size)
        if fc_type is None:
            fc_type = fc_type_defaults
            fc_type['bias'] = bias
            fc_type['device'] = device
        self.fc_type = fc_type
        self.fc_type_name = self.fc_type['name']
        self.up_proj = build_fc(name=self.fc_type_name, in_features=d_model, out_features=ffn_hidden_size, fc_kwargs=self.fc_type)
        self.act = act_fn
        self.down_proj = build_fc(name=self.fc_type_name, in_features=ffn_hidden_size, out_features=d_model, fc_kwargs=self.fc_type)
        self.down_proj._is_residual = True

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self.down_proj(self.act(self.up_proj(x)))


class MPTGLU(MPTMLP):

    def __init__(self, d_model: 'int', expansion_ratio: 'Union[int, float]', fc_type: 'Optional[dict[str, Any]]'=None, ffn_hidden_size: 'Optional[int]'=None, act_fn: 'Callable[[torch.Tensor], torch.Tensor]'=_DEFAULT_ACT_FN, device: 'Optional[str]'=None, bias: 'bool'=True):
        super().__init__(d_model=d_model, expansion_ratio=expansion_ratio, fc_type=fc_type, ffn_hidden_size=ffn_hidden_size, act_fn=act_fn, device=device, bias=bias)
        self.gate_proj = build_fc(name=self.fc_type_name, in_features=d_model, out_features=self.up_proj.out_features, fc_kwargs=self.fc_type)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


def _cast_if_autocast_enabled(tensor: 'torch.Tensor') ->torch.Tensor:
    if torch.is_autocast_enabled():
        if tensor.device.type == 'cuda':
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == 'cpu':
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor
    return tensor


class LPLayerNorm(torch.nn.LayerNorm):

    def __init__(self, normalized_shape: 'Union[int, list[int], torch.Size]', eps: 'float'=1e-05, elementwise_affine: 'bool'=True, device: 'Optional[torch.device]'=None, dtype: 'Optional[torch.dtype]'=None):
        super().__init__(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device=device, dtype=dtype)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        module_device = x.device
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
        downcast_bias = _cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
        with torch.autocast(enabled=False, device_type=module_device.type):
            return torch.nn.functional.layer_norm(downcast_x, self.normalized_shape, downcast_weight, downcast_bias, self.eps)


def rms_norm(x: 'torch.Tensor', weight: 'Optional[torch.Tensor]'=None, eps: 'float'=1e-05) ->torch.Tensor:
    output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        return output * weight
    return output


class RMSNorm(torch.nn.Module):

    def __init__(self, normalized_shape: 'Union[int, list[int], torch.Size]', eps: 'float'=1e-05, weight: 'bool'=True, dtype: 'Optional[torch.dtype]'=None, device: 'Optional[torch.device]'=None):
        super().__init__()
        self.eps = eps
        if weight:
            self.weight = torch.nn.Parameter(torch.ones(normalized_shape, dtype=dtype, device=device))
        else:
            self.register_parameter('weight', None)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return rms_norm(x.float(), self.weight, self.eps)


class LPRMSNorm(RMSNorm):

    def __init__(self, normalized_shape: 'Union[int, list[int], torch.Size]', eps: 'float'=1e-05, weight: 'bool'=True, dtype: 'Optional[torch.dtype]'=None, device: 'Optional[torch.device]'=None):
        super().__init__(normalized_shape=normalized_shape, eps=eps, weight=weight, dtype=dtype, device=device)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
        with torch.autocast(enabled=False, device_type=x.device.type):
            return rms_norm(downcast_x, downcast_weight, self.eps)


class ModelWithIntParameter(nn.Module):

    def __init__(self):
        super().__init__()
        self.int_param = nn.Parameter(torch.tensor(0, dtype=torch.int64), requires_grad=False)
        self.float_param = nn.Parameter(torch.randn(10), requires_grad=True)

    def forward(self, x: 'torch.Tensor'):
        return x


class _DummyModule(nn.Module):

    def __init__(self, device: 'str'='cpu', dtype: 'torch.dtype'=torch.float32):
        super().__init__()
        self.linear0 = nn.Linear(4, 3, device=device, dtype=dtype)
        self.norm0 = nn.LayerNorm(3, device=device, dtype=dtype)
        self.linear1 = nn.Linear(3, 5, device=device, dtype=dtype)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self.linear1(self.norm0(self.linear0(x)))


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (LPLayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LPRMSNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'cfg': SimpleNamespace(in_features=4, out_features=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ModelWithIntParameter,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RMSNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (_DummyModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

