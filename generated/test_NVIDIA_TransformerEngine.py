
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


import time


import pandas as pd


import numpy as np


import torch


import functools


import re


from typing import List


from typing import Optional


from typing import Tuple


from typing import Union


import math


from typing import Callable


from torch import nn


from torch.optim import AdamW


from torch.utils.data import DataLoader


from functools import partial


import warnings


import torch.distributed as dist


from torch.nn.parallel import DistributedDataParallel


from torch.distributed.fsdp import FullyShardedDataParallel


from torch.distributed.fsdp import MixedPrecision


from torch.distributed.fsdp.wrap import always_wrap_policy


from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torchvision import datasets


from torchvision import transforms


from torch.optim.lr_scheduler import StepLR


from typing import Any


from typing import Dict


from typing import Sequence


from typing import Iterable


from functools import reduce


from torch.distributed.elastic.multiprocessing.errors import record


from functools import wraps


import itertools


import logging


from collections.abc import Iterable


from itertools import product


import copy


from torch.testing._internal.common_device_type import largeTensorTest


import random


from torch.nn import Parameter


from torch import nn as nn


from enum import Enum


from typing import Literal


from typing import NamedTuple


from math import sqrt


import inspect


import collections


import torch.distributed


from functools import lru_cache


from torch.cuda import _lazy_call


from torch.cuda import _lazy_init


from torch.utils.checkpoint import detach_variable


from torch.utils.checkpoint import noop_context_fn


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


from torch.distributed.fsdp._common_utils import _get_module_fsdp_state


from torch.distributed.fsdp._traversal_utils import _get_fsdp_states_with_modules


from collections import deque


from typing import TypeVar


from torch.utils._pytree import tree_flatten as _tree_flatten


from torch.utils._pytree import tree_unflatten as _tree_unflatten


from torch._C import _graph_pool_handle


from abc import ABC


from abc import abstractmethod


from typing import Generator


from typing import Set


from torch.nn import init


from torch.nn.parameter import Parameter


from collections.abc import Callable


import abc


from collections.abc import Iterator


from copy import deepcopy


from itertools import chain


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


from torch.distributed._tensor import DTensor


from torch.utils.cpp_extension import BuildExtension


import torch._C._onnx as _C_onnx


from torch.onnx import _type_utils


from torch.onnx import symbolic_helper


from torch.onnx import register_custom_op_symbolic


from torch.onnx._internal import jit_utils


from torch.utils._pytree import tree_map


AttnMaskTypes = 'no_mask', 'padding', 'causal', 'padding_causal', 'causal_bottom_right', 'padding_causal_bottom_right', 'arbitrary'


AttnTypes = 'self', 'cross'


CPUOffloadEnabled = False


def graph_safe_rng_available() ->bool:
    """Returns whether cuda graph safe RNG state manipulation is supported."""
    return hasattr(torch.cuda.CUDAGraph, 'register_generator_state') and hasattr(torch.Generator, 'graphsafe_set_state') and hasattr(torch.Generator, 'graphsafe_get_state') and hasattr(torch.Generator, 'clone_state')


def _get_cuda_rng_state(device: 'Union[int, str, torch.device]'='cuda', clone: 'bool'=False, graph_safe: 'bool'=True) ->torch.Tensor:
    """Return the random number generator state of the specified GPU."""
    _lazy_init()
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device('cuda', device)
    idx = device.index
    if idx is None:
        idx = torch.cuda.current_device()
    default_generator = torch.cuda.default_generators[idx]
    if graph_safe_rng_available() and graph_safe:
        if clone:
            return default_generator.clone_state()
        return default_generator.graphsafe_get_state()
    return default_generator.get_state()


def _set_cuda_rng_state(new_state: 'torch.Tensor', device: 'Union[int, str]'=-1, graph_safe=True) ->None:
    """Sets the random number generator state of the current GPU."""
    if device == -1:
        device = torch.device('cuda')
    elif isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device('cuda', device)

    def cb() ->None:
        idx = device.index
        if idx is None:
            idx = torch.cuda.current_device()
        default_generator = torch.cuda.default_generators[idx]
        if graph_safe_rng_available() and graph_safe:
            default_generator.graphsafe_set_state(new_state)
            return
        default_generator.set_state(new_state)
    _lazy_call(cb)


def set_all_rng_states(states: 'List') ->None:
    """Updates all generator states used by `CudaRNGStatesTracker`."""
    global _ALL_ACTIVE_RNG_STATES
    _ALL_ACTIVE_RNG_STATES = states


class _FormatHelper(NamedTuple):
    """
    Stores max FP8 values for fprop and bprop a `Format`.
    """
    max_fwd: 'float'
    max_bwd: 'float'


class Format(Enum):
    """
    Supported FP8 formats.

    Values
    ------
    E4M3 :
          All FP8 tensors are in e4m3 format
    E5M2 :
          All FP8 tensors are in e5m2 format
    HYBRID :
            FP8 tensors in the forward pass are in e4m3 format,
            FP8 tensors in the backward pass are in e5m2 format
    """
    E4M3 = _FormatHelper(max_fwd=448, max_bwd=448)
    E5M2 = _FormatHelper(max_fwd=57344, max_bwd=57344)
    HYBRID = _FormatHelper(max_fwd=E4M3.max_fwd, max_bwd=E5M2.max_bwd)


class _OverrideLinearPrecision(NamedTuple):
    """
    Whether or not the execute the `fprop`, `dgrad`, and `wgrad`
    GEMMs in higher precision when using FP8.
    """
    fprop: 'bool' = False
    dgrad: 'bool' = False
    wgrad: 'bool' = False


def _update_amax_history(amax_history: 'torch.Tensor') ->torch.Tensor:
    """Update amax history and set next amax to zero."""
    if amax_history.shape[0] > 1:
        new_amax_history = torch.roll(amax_history, -1, 0)
        amax_history.copy_(new_amax_history)
    amax_history[0].fill_(0.0)
    return amax_history


@torch.jit.script
def _default_get_amax_and_update_history(amax_history: 'torch.Tensor', amax_compute_algo: 'str') ->Tuple[torch.Tensor, torch.Tensor]:
    """Default function to obtain amax from history."""
    if amax_compute_algo == 'max':
        amax = torch.max(amax_history, dim=0).values
    else:
        amax = amax_history[0].clone()
    amax_history = _update_amax_history(amax_history)
    return amax_history, amax


def _compute_amax_and_update_history(amax_history: 'torch.Tensor', amax_compute_algo: 'Union[Callable, str]') ->Tuple[torch.Tensor, torch.Tensor]:
    """Obtain the amax from the history."""
    if callable(amax_compute_algo):
        amax = amax_compute_algo(amax_history)
        amax_history = _update_amax_history(amax_history)
        return amax_history, amax
    return _default_get_amax_and_update_history(amax_history, amax_compute_algo)


jit_fuser = torch.jit.script


@jit_fuser
def _default_sf_compute(amax: 'torch.Tensor', scale: 'torch.Tensor', fp8_max: 'float', margin: 'int', _fp32_max: 'float'=torch.finfo(torch.float32).max) ->torch.Tensor:
    """Default function to convert amax to scaling factor.
    Computing the scaling factor requires consideration of the following scenarios:
    1. amax == 0:
       No action is possible, set scale to the previous scale (or 1).
    2. 0 < amax < tiny_amax
       The amax is too tiny that the scale becomes infinite in FP32.
       Set scale = FP32_max
    3. tiny_amax <= amax < FP32_max:
       Set scale = FP8_max (or scaled_max) / amax
    4. When amax == inf or amax == nan:
       No action is possible, set scale to the previous scale (or 1).
    """
    sf = fp8_max / amax / 2 ** margin
    sf = torch.where(amax > 0.0, sf, scale)
    sf = torch.where(torch.isfinite(amax), sf, scale)
    sf = torch.where(torch.isinf(sf), torch.full_like(sf, _fp32_max), sf)
    scale.copy_(sf)
    return scale


def _compute_scaling_factor(amax: 'torch.Tensor', scale: 'torch.Tensor', fp8_max: 'float', recipe: 'DelayedScaling') ->torch.Tensor:
    """Convert amax to scaling factor."""
    if recipe.scaling_factor_compute_algo is None:
        return _default_sf_compute(amax, scale, fp8_max, recipe.margin)
    return recipe.scaling_factor_compute_algo(amax, scale, fp8_max, recipe)


def _amax_and_scale_update(amax_history: 'torch.Tensor', scale: 'torch.Tensor', scale_inv: 'torch.Tensor', fp8_max: 'float', recipe: 'DelayedScaling') ->None:
    """Updates FP8 meta tensors."""
    new_amax_history, amax = _compute_amax_and_update_history(amax_history, recipe.amax_compute_algo)
    new_scale = _compute_scaling_factor(amax, scale, fp8_max, recipe)
    scale.copy_(new_scale)
    scale_inv.copy_(1.0 / new_scale)
    amax_history.copy_(new_amax_history)


def get_device_compute_capability() ->Tuple[int, int]:
    """CUDA compute capability of current GPU"""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.major, props.minor


def check_fp8_support() ->Tuple[bool, str]:
    """Return if fp8 support is available"""
    if get_device_compute_capability() >= (9, 0):
        return True, ''
    if get_device_compute_capability() < (8, 9):
        return False, 'Device compute capability 8.9 or higher required for FP8 execution.'
    if tex.get_cublasLt_version() < 120103:
        return False, 'CublasLt version 12.1.3.x or higher required for FP8 execution on Ada.'
    if float(torch.version.cuda) < 12.1:
        return False, 'Cuda version 12.1 or higher required for FP8 execution on Ada.'
    return True, ''


dist_group_type = torch.distributed.ProcessGroup


def split_and_copy(buffer: 'torch.Tensor', outputs: 'List[torch.Tensor]', chunk_sizes: 'List[int]') ->None:
    """Split `buffer` by `chunk_sizes` and copy into `outputs`."""
    splits = buffer.split(chunk_sizes)
    torch._foreach_copy_(outputs, splits)


class _DequantizeFunc(torch.autograd.Function):
    """Autograd function to convert quantized tensor to standard tensor"""

    @staticmethod
    def forward(_ctx: 'torch.autograd.function.FunctionCtx', tensor: 'QuantizedTensor', dtype: 'Optional[torch.dtype]'=None) ->torch.Tensor:
        return tensor.dequantize(dtype=dtype)

    @staticmethod
    def backward(_ctx: 'torch.autograd.function.FunctionCtx', grad: 'torch.Tensor') ->Tuple[Optional[torch.Tensor], ...]:
        return grad, None

