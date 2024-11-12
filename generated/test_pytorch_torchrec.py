
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


from typing import List


from typing import Tuple


import torch


import time


from typing import Dict


from typing import Optional


import numpy as np


from torch.utils.data.dataset import IterableDataset


import queue


from typing import Union


from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL


import torch.distributed as dist


from typing import Callable


from torch.distributed._shard.sharded_tensor import ShardedTensor


import torch.nn as nn


from typing import Any


from typing import cast


import torch.optim as optim


import torch.utils.data as data_utils


from torch import distributed as dist


from torch.nn.parallel import DistributedDataParallel as DDP


import pandas as pd


from torch.utils.data.distributed import DistributedSampler


import copy


import math


import uuid


from torch.distributed.launcher.api import elastic_launch


from torch.distributed.launcher.api import LaunchConfig


from torch.distributed.elastic.multiprocessing.errors import record


from torch.distributed.optim import _apply_optimizer_in_backward as apply_optimizer_in_backward


from torch.utils.data import IterableDataset


from typing import Iterator


from torch import nn


import logging


from torch.utils.data import DataLoader


from torch.package import PackageExporter


from torch.utils import data as data_utils


from torch.utils.data import Dataset


import torch.nn.functional as F


from torch.distributed import all_reduce


from torch.distributed import get_rank


from torch.distributed import get_world_size


from torch.distributed import init_process_group


from collections import OrderedDict


from typing import Mapping


from typing import OrderedDict


import torch.distributed.launcher as pet


from torch.multiprocessing.reductions import reduce_storage


from torch.multiprocessing.reductions import reduce_typed_storage


from torch.multiprocessing.reductions import reduce_typed_storage_child


from typing import Iterable


import torch.utils.data.datapipes as dp


from torch.utils.data import IterDataPipe


import itertools


import random


from functools import partial


from typing import Sequence


from typing import TypeVar


from torch.utils.data import functional_datapipe


from torch.utils.data import get_worker_info


import abc


import inspect


from typing import Generic


from enum import Enum


from typing import ContextManager


from torch import multiprocessing as mp


from torch.autograd.profiler import record_function


from functools import wraps


import torch.distributed._functional_collectives


from torch import Tensor


from torch.autograd import Function


from torch.distributed._composable import replicate


from torch.distributed._shard.api import ShardedTensor


from torch.distributed.checkpoint import FileSystemReader


from torch.distributed.checkpoint import FileSystemWriter


from torch.distributed.checkpoint import load_state_dict


from torch.distributed.checkpoint import save_state_dict


from torch.distributed._tensor.api import DTensor


from torch.distributed._composable import fully_shard


from torch.distributed._tensor import DTensor


from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel


from torch.distributed.fsdp.wrap import ModuleWrapPolicy


import warnings


from collections import defaultdict


from collections import deque


from itertools import accumulate


from typing import MutableMapping


from typing import Type


from typing import Union as TypeUnion


from torch.nn.parallel import DistributedDataParallel


from abc import ABC


from torch.autograd.function import FunctionCtx


from torch.nn.modules.module import _IncompatibleKeys


from itertools import filterfalse


from typing import Set


from enum import unique


from torch import fx


from torch.distributed._tensor import DeviceMesh


from torch.distributed._tensor.placement_types import Placement


from torch.nn.modules.module import _addindent


from torch.distributed._tensor import Shard


from typing import DefaultDict


from torch.distributed._shard.sharded_tensor import Shard


from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as ddp_default_hooks


from torch.distributed.fsdp import FullyShardedDataParallel


from abc import abstractmethod


from torch._prims_common import is_integer_dtype


from functools import reduce


from time import perf_counter


from copy import deepcopy


from torch.distributed import _remote_device


from torch.distributed._shard.sharded_tensor import ShardedTensorBase


from torch.distributed._shard.sharded_tensor import ShardedTensorMetadata


from torch.distributed._shard.sharded_tensor import ShardMetadata


from torch.distributed._composable.contract import contract


from torch.distributed._shard.sharded_tensor.api import ShardedTensor


from torch.distributed._tensor.placement_types import Shard


from torch.distributed._tensor.placement_types import Replicate


from torch.distributed.checkpoint.metadata import ChunkStorageMetadata


from torch.distributed.checkpoint.metadata import MetadataIndex


from torch.distributed.checkpoint.metadata import TensorProperties


from torch.distributed.checkpoint.metadata import TensorStorageMetadata


from torch.distributed.checkpoint.planner import TensorWriteData


from torch.distributed.checkpoint.planner import WriteItem


from torch.distributed.checkpoint.planner import WriteItemType


from torch import quantization as quant


from torch.distributed._shard.sharding_spec import ShardingSpec


from torch.utils import _pytree as pytree


from typing import Protocol


from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR


import functools


from torch.distributed.distributed_c10d import GroupMember


from typing import Generator


from torch.distributed import ProcessGroup


import torch.fx


from enum import auto


from torch._dynamo.testing import reduce_to_scalar_loss


from torch.testing._internal.distributed.fake_pg import FakeStore


import torch.quantization as quant


from torch import optim


from torch.optim import Optimizer


from torch._dynamo.utils import counters


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


import enum


from typing import Deque


from itertools import chain


from torch.fx.immutable_collections import immutable_dict as fx_immutable_dict


from torch.fx.immutable_collections import immutable_list as fx_immutable_list


from torch.fx.node import Node


from torch.profiler import record_function


from torch.distributed.device_mesh import DeviceMesh


from torch.distributed.device_mesh import init_device_mesh


from torch.distributed.distributed_c10d import _get_pg_default_device


from torch.distributed._shard.sharded_tensor import TensorProperties


from torch.distributed._shard.sharding_spec import EnumerableShardingSpec


from torch.distributed._shard.sharding_spec import ShardMetadata


from torch.testing import FileCheck


import typing


from torch.fx._compatibility import compatibility


from torch.fx.graph import Graph


from torch.fx.node import Argument


from torch.fx._symbolic_trace import is_fx_tracing


from typing import BinaryIO


from torch.package import PackageImporter


from torch.fx.passes.split_utils import getattr_recursive


from torch.export import Dim


from torch.export import ShapesCollection


from torch.export.dynamic_shapes import _Dim as DIM


from torch.export.unflatten import InterpreterModule


from torch.fx import Node


from torch import no_grad


from functools import update_wrapper


from torch.fx import wrap


from math import sqrt


from typing import NamedTuple


import torch.utils.hooks as hooks


from torch.nn.modules.lazy import _LazyProtocol


from torch.nn.modules.lazy import LazyModuleMixin


from torch.nn.modules.module import _global_backward_hooks


from torch.nn.modules.module import _global_forward_hooks


from torch.nn.modules.module import _global_forward_pre_hooks


from logging import getLogger


from logging import Logger


from torch.fx import GraphModule


from torch.fx import Tracer


import re


from warnings import warn


from typing import Collection


from torch.optim.optimizer import Optimizer


from torch.autograd import Variable


from torch.distributed._shard import sharded_tensor


from torch.distributed._shard import sharding_spec


from torch.fx.experimental.symbolic_shapes import guard_size_oblivious


from torch.fx._pytree import register_pytree_flatten_spec


from torch.fx._pytree import TreeSpec


from torch.utils._pytree import GetAttrKey


from torch.utils._pytree import KeyEntry


from torch.utils._pytree import register_pytree_node


import torch.utils._pytree as pytree


from torch.fx._pytree import tree_flatten_spec


import torch._prims_common as utils


@unique
class DataType(Enum):
    """
    Our fusion implementation supports only certain types of data
    so it makes sense to retrict in a non-fused version as well.
    """
    FP32 = 'FP32'
    FP16 = 'FP16'
    BF16 = 'BF16'
    INT64 = 'INT64'
    INT32 = 'INT32'
    INT8 = 'INT8'
    UINT8 = 'UINT8'
    INT4 = 'INT4'
    INT2 = 'INT2'

    def __str__(self) ->str:
        return self.value


DEFAULT_ROW_ALIGNMENT = 16


class JaggedTensorMeta(abc.ABCMeta, torch.fx._symbolic_trace.ProxyableClassMeta):
    pass


class Multistreamable(abc.ABC):
    """
    Objects implementing this interface are allowed to be transferred
    from one CUDA stream to another.
    torch.Tensor and (Keyed)JaggedTensor implement this interface.
    """

    @abc.abstractmethod
    def record_stream(self, stream: 'torch.Stream') ->None:
        """
        See https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html
        """
        ...


class Pipelineable(Multistreamable):
    """
    This interface contains two methods, one for moving an input across devices,
    the other one for marking streams that operate the input.

    torch.Tensor implements this interface and we can used it in many applications.
    Another example is torchrec.(Keyed)JaggedTensor, which we use as the input to
    torchrec.EmbeddingBagCollection, which in turn is often the first layer of many models.
    Some models take compound inputs, which should implement this interface.
    """

    @abc.abstractmethod
    def to(self, device: 'torch.device', non_blocking: 'bool') ->'Pipelineable':
        """
        Please be aware that according to https://pytorch.org/docs/stable/generated/torch.Tensor.to.html,
        `to` might return self or a copy of self.  So please remember to use `to` with the assignment operator,
        for example, `in = in.to(new_device)`.
        """
        ...


@torch.fx.wrap
def _arange(*args, **kwargs) ->torch.Tensor:
    return torch.arange(*args, **kwargs)


def _assert_offsets_or_lengths_is_provided(offsets: 'Optional[torch.Tensor]', lengths: 'Optional[torch.Tensor]') ->None:
    assert offsets is not None or lengths is not None, 'Must provide lengths or offsets'


def is_pt2_compiling() ->bool:
    return is_torchdynamo_compiling() or is_compiler_compiling()


def pt2_guard_size_oblivious(x: 'bool') ->bool:
    if torch.jit.is_scripting() or not is_pt2_compiling():
        return x
    return guard_size_oblivious(x)


def _assert_tensor_has_no_elements_or_has_integers(tensor: 'Optional[torch.Tensor]', tensor_name: 'str') ->None:
    if is_torchdynamo_compiling() or tensor is None:
        return
    assert pt2_guard_size_oblivious(tensor.numel() == 0) or tensor.dtype in [torch.long, torch.int, torch.short, torch.int8, torch.uint8], '{} must be of integer type, but got {}'.format(tensor_name, tensor.dtype)


def _get_weights_or_throw(weights: 'Optional[torch.Tensor]') ->torch.Tensor:
    assert weights is not None, "This (Keyed)JaggedTensor doesn't have weights."
    return weights


def _values_string(values: 'torch.Tensor', start: 'int', end: 'int') ->str:
    size = values.size()
    if len(size) == 1:
        return '[' + ', '.join([str(value.item()) for value in values[start:end]]) + ']'
    elif len(size) == 2:
        values_list: 'List[str]' = []
        for value in values[start:end]:
            values_list.append('[' + ', '.join([str(s.item()) for s in value]) + ']')
        return '[' + ', '.join(values_list) + ']'
    else:
        raise ValueError("the values dimension is larger than 2, we don't support printing")


def _jagged_values_string(values: 'torch.Tensor', offsets: 'torch.Tensor', offset_start: 'int', offset_end: 'int') ->str:
    return '[' + ', '.join([_values_string(values, offsets[index], offsets[index + 1]) for index in range(offset_start, offset_end)]) + ']'


def _to_lengths(offsets: 'torch.Tensor') ->torch.Tensor:
    return offsets[1:] - offsets[:-1]


def _maybe_compute_lengths(lengths: 'Optional[torch.Tensor]', offsets: 'Optional[torch.Tensor]') ->torch.Tensor:
    if lengths is None:
        assert offsets is not None
        lengths = _to_lengths(offsets)
    return lengths


def _to_offsets(lengths: 'torch.Tensor') ->torch.Tensor:
    return torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)


def _maybe_compute_offsets(lengths: 'Optional[torch.Tensor]', offsets: 'Optional[torch.Tensor]') ->torch.Tensor:
    if offsets is None:
        assert lengths is not None
        offsets = _to_offsets(lengths)
    return offsets


@torch.fx.wrap
def _optional_mask(tensor: 'Optional[torch.Tensor]', mask: 'torch.Tensor') ->Optional[torch.Tensor]:
    return tensor[mask] if tensor is not None else None


class JaggedTensor(Pipelineable, metaclass=JaggedTensorMeta):
    """
    Represents an (optionally weighted) jagged tensor.

    A `JaggedTensor` is a tensor with a *jagged dimension* which is dimension whose
    slices may be of different lengths. See `KeyedJaggedTensor` for full example.

    Implementation is torch.jit.script-able.

    NOTE:
        We will NOT do input validation as it's expensive, you should always pass in the
        valid lengths, offsets, etc.

    Args:
        values (torch.Tensor): values tensor in dense representation.
        weights (Optional[torch.Tensor]): if values have weights. Tensor with same shape
            as values.
        lengths (Optional[torch.Tensor]): jagged slices, represented as lengths.
        offsets (Optional[torch.Tensor]): jagged slices, represented as cumulative
            offsets.
    """
    _fields = ['_values', '_weights', '_lengths', '_offsets']

    def __init__(self, values: 'torch.Tensor', weights: 'Optional[torch.Tensor]'=None, lengths: 'Optional[torch.Tensor]'=None, offsets: 'Optional[torch.Tensor]'=None) ->None:
        self._values: 'torch.Tensor' = values
        self._weights: 'Optional[torch.Tensor]' = weights
        _assert_offsets_or_lengths_is_provided(offsets, lengths)
        if offsets is not None:
            _assert_tensor_has_no_elements_or_has_integers(offsets, 'offsets')
        if lengths is not None:
            _assert_tensor_has_no_elements_or_has_integers(lengths, 'lengths')
        self._lengths: 'Optional[torch.Tensor]' = lengths
        self._offsets: 'Optional[torch.Tensor]' = offsets

    @staticmethod
    def empty(is_weighted: 'bool'=False, device: 'Optional[torch.device]'=None, values_dtype: 'Optional[torch.dtype]'=None, weights_dtype: 'Optional[torch.dtype]'=None, lengths_dtype: 'torch.dtype'=torch.int32) ->'JaggedTensor':
        """
        Constructs an empty JaggedTensor.

        Args:
            is_weighted (bool): whether the JaggedTensor has weights.
            device (Optional[torch.device]): device for JaggedTensor.
            values_dtype (Optional[torch.dtype]): dtype for values.
            weights_dtype (Optional[torch.dtype]): dtype for weights.
            lengths_dtype (torch.dtype): dtype for lengths.

        Returns:
            JaggedTensor: empty JaggedTensor.
        """
        weights = torch.empty(0, dtype=weights_dtype, device=device) if is_weighted else None
        return JaggedTensor(values=torch.empty(0, dtype=values_dtype, device=device), offsets=torch.empty(0, dtype=lengths_dtype, device=device), lengths=torch.empty(0, dtype=lengths_dtype, device=device), weights=weights)

    @staticmethod
    def from_dense_lengths(values: 'torch.Tensor', lengths: 'torch.Tensor', weights: 'Optional[torch.Tensor]'=None) ->'JaggedTensor':
        """
        Constructs `JaggedTensor` from values and lengths tensors, with optional weights.
        Note that `lengths` is still of shape (B,), where B is the batch size.

        Args:
            values (torch.Tensor): dense representation of values.
            lengths (torch.Tensor): jagged slices, represented as lengths.
            weights (Optional[torch.Tensor]): if values have weights, tensor with
                the same shape as values.

        Returns:
            JaggedTensor: JaggedTensor created from 2D dense tensor.
        """
        mask2d = _arange(end=values.size(1), device=values.device).expand(values.size(0), -1) < lengths.unsqueeze(-1)
        return JaggedTensor(values=values[mask2d], weights=_optional_mask(weights, mask2d), lengths=lengths)

    @staticmethod
    def from_dense(values: 'List[torch.Tensor]', weights: 'Optional[List[torch.Tensor]]'=None) ->'JaggedTensor':
        """
        Constructs `JaggedTensor` from list of tensors as values, with optional weights.
        `lengths` will be computed, of shape (B,), where B is `len(values)` which
        represents the batch size.

        Args:
            values (List[torch.Tensor]): a list of tensors for dense representation
            weights (Optional[List[torch.Tensor]]): if values have weights, tensor with
                the same shape as values.

        Returns:
            JaggedTensor: JaggedTensor created from 2D dense tensor.

        Example::

            values = [
                torch.Tensor([1.0]),
                torch.Tensor(),
                torch.Tensor([7.0, 8.0]),
                torch.Tensor([10.0, 11.0, 12.0]),
            ]
            weights = [
                torch.Tensor([1.0]),
                torch.Tensor(),
                torch.Tensor([7.0, 8.0]),
                torch.Tensor([10.0, 11.0, 12.0]),
            ]
            j1 = JaggedTensor.from_dense(
                values=values,
                weights=weights,
            )

            # j1 = [[1.0], [], [7.0, 8.0], [10.0, 11.0, 12.0]]
        """
        values_tensor = torch.cat(values, dim=0)
        lengths = torch.tensor([value.size(0) for value in values], dtype=torch.int32, device=values_tensor.device)
        weights_tensor = torch.cat(weights, dim=0) if weights is not None else None
        return JaggedTensor(values=values_tensor, weights=weights_tensor, lengths=lengths)

    def to_dense(self) ->List[torch.Tensor]:
        """
        Constructs a dense-representation of the JT's values.

        Returns:
            List[torch.Tensor]: list of tensors.

        Example::

            values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])
            jt = JaggedTensor(values=values, offsets=offsets)

            values_list = jt.to_dense()

            # values_list = [
            #     torch.tensor([1.0, 2.0]),
            #     torch.tensor([]),
            #     torch.tensor([3.0]),
            #     torch.tensor([4.0]),
            #     torch.tensor([5.0]),
            #     torch.tensor([6.0, 7.0, 8.0]),
            # ]
        """
        tensor_list = []
        for index in range(self.offsets().size(0) - 1):
            offset = self.offsets()[index].item()
            next_offset = self.offsets()[index + 1].item()
            tensor_list.append(self.values()[offset:next_offset])
        return tensor_list

    def to_dense_weights(self) ->Optional[List[torch.Tensor]]:
        """
        Constructs a dense-representation of the JT's weights.

        Returns:
            Optional[List[torch.Tensor]]: list of tensors, `None` if no weights.

        Example::

            values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            weights = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
            offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])
            jt = JaggedTensor(values=values, weights=weights, offsets=offsets)

            weights_list = jt.to_dense_weights()

            # weights_list = [
            #     torch.tensor([0.1, 0.2]),
            #     torch.tensor([]),
            #     torch.tensor([0.3]),
            #     torch.tensor([0.4]),
            #     torch.tensor([0.5]),
            #     torch.tensor([0.6, 0.7, 0.8]),
            # ]
        """
        if self.weights_or_none() is None:
            return None
        tensor_list = []
        for index in range(self.offsets().size(0) - 1):
            offset = self.offsets()[index].item()
            next_offset = self.offsets()[index + 1].item()
            tensor_list.append(self.weights()[offset:next_offset])
        return tensor_list

    def to_padded_dense(self, desired_length: 'Optional[int]'=None, padding_value: 'float'=0.0) ->torch.Tensor:
        """
        Constructs a 2D dense tensor from the JT's values of shape (B, N,).

        Note that `B` is the length of self.lengths() and
        `N` is the longest feature length or `desired_length`.

        If `desired_length` > `length` we will pad with `padding_value`, otherwise we
        will select the last value at `desired_length`.

        Args:
            desired_length (int): the length of the tensor.
            padding_value (float): padding value if we need to pad.

        Returns:
            torch.Tensor: 2d dense tensor.

        Example::

            values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])
            jt = JaggedTensor(values=values, offsets=offsets)

            dt = jt.to_padded_dense(
                desired_length=2,
                padding_value=10.0,
            )

            # dt = [
            #     [1.0, 2.0],
            #     [10.0, 10.0],
            #     [3.0, 10.0],
            #     [4.0, 10.0],
            #     [5.0, 10.0],
            #     [6.0, 7.0],
            # ]
        """
        if desired_length is None:
            N = int(torch.max(self.lengths()).item())
        else:
            N = desired_length
        return torch.ops.fbgemm.jagged_to_padded_dense(self.values(), [self.offsets()], [N], padding_value)

    def to_padded_dense_weights(self, desired_length: 'Optional[int]'=None, padding_value: 'float'=0.0) ->Optional[torch.Tensor]:
        """
        Constructs a 2D dense tensor from the JT's weights of shape (B, N,).

        Note that `B` (batch size) is the length of self.lengths() and
        `N` is the longest feature length or `desired_length`.

        If `desired_length` > `length` we will pad with `padding_value`, otherwise we
        will select the last value at `desired_length`.

        Like `to_padded_dense` but for the JT's weights instead of values.

        Args:
            desired_length (int): the length of the tensor.
            padding_value (float): padding value if we need to pad.

        Returns:
            Optional[torch.Tensor]: 2d dense tensor, `None` if no weights.

        Example::

            values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            weights = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
            offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])
            jt = JaggedTensor(values=values, weights=weights, offsets=offsets)

            d_wt = jt.to_padded_dense_weights(
                desired_length=2,
                padding_value=1.0,
            )

            # d_wt = [
            #     [0.1, 0.2],
            #     [1.0, 1.0],
            #     [0.3, 1.0],
            #     [0.4, 1.0],
            #     [0.5, 1.0],
            #     [0.6, 0.7],
            # ]
        """
        if self.weights_or_none() is None:
            return None
        if desired_length is None:
            N = int(torch.max(self.lengths()).item())
        else:
            N = desired_length
        return torch.ops.fbgemm.jagged_to_padded_dense(self.weights(), [self.offsets()], [N], padding_value)

    def device(self) ->torch.device:
        """
        Get JaggedTensor device.

        Returns:
            torch.device: the device of the values tensor.
        """
        return self._values.device

    def lengths(self) ->torch.Tensor:
        """
        Get JaggedTensor lengths. If not computed, compute it from offsets.

        Returns:
            torch.Tensor: the lengths tensor.
        """
        _lengths = _maybe_compute_lengths(self._lengths, self._offsets)
        self._lengths = _lengths
        return _lengths

    def lengths_or_none(self) ->Optional[torch.Tensor]:
        """
        Get JaggedTensor lengths. If not computed, return None.

        Returns:
            Optional[torch.Tensor]: the lengths tensor.
        """
        return self._lengths

    def offsets(self) ->torch.Tensor:
        """
        Get JaggedTensor offsets. If not computed, compute it from lengths.

        Returns:
            torch.Tensor: the offsets tensor.
        """
        _offsets = _maybe_compute_offsets(self._lengths, self._offsets)
        self._offsets = _offsets
        return _offsets

    def offsets_or_none(self) ->Optional[torch.Tensor]:
        """
        Get JaggedTensor offsets. If not computed, return None.

        Returns:
            Optional[torch.Tensor]: the offsets tensor.
        """
        return self._offsets

    def values(self) ->torch.Tensor:
        """
        Get JaggedTensor values.

        Returns:
            torch.Tensor: the values tensor.
        """
        return self._values

    def weights(self) ->torch.Tensor:
        """
        Get JaggedTensor weights. If None, throw an error.

        Returns:
            torch.Tensor: the weights tensor.
        """
        return _get_weights_or_throw(self._weights)

    def weights_or_none(self) ->Optional[torch.Tensor]:
        """
        Get JaggedTensor weights. If None, return None.

        Returns:
            Optional[torch.Tensor]: the weights tensor.
        """
        return self._weights

    def to(self, device: 'torch.device', non_blocking: 'bool'=False) ->'JaggedTensor':
        """
        Move the JaggedTensor to the specified device.

        Args:
            device (torch.device): the device to move to.
            non_blocking (bool): whether to perform the copy asynchronously.

        Returns:
            JaggedTensor: the moved JaggedTensor.
        """
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        return JaggedTensor(values=self._values, weights=weights if weights is not None else None, lengths=lengths if lengths is not None else None, offsets=offsets if offsets is not None else None)

    @torch.jit.unused
    def record_stream(self, stream: 'torch.cuda.streams.Stream') ->None:
        self._values.record_stream(stream)
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        if weights is not None:
            weights.record_stream(stream)
        if lengths is not None:
            lengths.record_stream(stream)
        if offsets is not None:
            offsets.record_stream(stream)

    def __str__(self) ->str:
        offsets = self.offsets()
        if self._weights is None:
            return 'JaggedTensor({\n    ' + _jagged_values_string(self._values, offsets, 0, len(offsets) - 1) + '\n})\n'
        return 'JaggedTensor({\n' + '    "values": ' + _jagged_values_string(self._values, offsets, 0, len(offsets) - 1) + ',\n    "weights": ' + _jagged_values_string(_get_weights_or_throw(self._weights), offsets, 0, len(offsets) - 1) + '\n})\n'


class CopyMixIn:

    @abstractmethod
    def copy(self, device: 'torch.device') ->nn.Module:
        ...


class ModuleNoCopyMixin(CopyMixIn):
    """
    A mixin to allow modules to override copy behaviors in DMP.
    """

    def copy(self, device: 'torch.device') ->nn.Module:
        return self


def down_size(N: 'int', size: 'torch.Size') ->Tuple[int, int]:
    assert size[-1] % N == 0, f'{size} last dim not divisible by {N}'
    return *size[:-1], size[-1] // N


def fill_defaults(args, n, defaults_tail):
    """
    __torch_dispatch__ doesn't guarantee the number of arguments you are
    passed (e.g., defaulted arguments are not passed); but usually it is
    convenient to pad out the arguments list with defaults.  This function
    helps you do that.
    Args:
        args: the list of positional arguments passed to __torch_dispatch__
        n: the number of arguments you are expecting to get
        defaults_tail: default values for the arguments, starting from the
            end of the list
    Example:
        >>> fill_defaults([1, 2, 3], 5, [3, 4, 5])
        [1, 2, 3, 4, 5]
        >>> fill_defaults([1, 2, 3], 5, [None, None, None])
        [1, 2, 3, None, None]]
    """
    if n - len(defaults_tail) > len(args):
        raise RuntimeError('not enough defaults to fill arguments')
    r = list(args)
    for i in range(len(args), n):
        r.append(defaults_tail[i - n + len(defaults_tail)])
    return r


def find_arg_of_type(it, t):
    for x in it:
        if isinstance(x, t):
            return x
    return None


def up_size(N: 'int', size: 'torch.Size') ->Tuple[int, int]:
    return *size[:-1], size[-1] * N


class UIntXTensor(torch.Tensor):
    """
    A Tensor subclass of uint8 dtype, that represents Tensor with X-bit elements.
    The last dimension must be divisible by (8 // X).

    __torch_dispatch__ special handling:
    .view(dtype=torch.uint8) - returns the underlying uint8 data.

    .slice,.view - works in UIntX units, dimension values must be divisible by (8 // X).

    .detach,.clone - work as an op on underlying uint8 data.
    """
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, N: 'int', elem):
        assert elem.dtype is torch.uint8
        return torch.Tensor._make_wrapper_subclass(cls, up_size(N, elem.shape), dtype=torch.uint8)

    def __init__(self, N: 'int', elem: 'torch.Tensor') ->None:
        self.N: 'int' = N
        self.elem: 'torch.Tensor' = elem

    def tolist(self) ->List:
        return self.elem.tolist()

    def __repr__(self) ->str:
        return f'UInt{8 // self.N}Tensor(shape={self.shape}, elem={self.elem})'

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func is torch.ops.aten.detach.default:
            with torch.inference_mode(False):
                self, = args
                return cls(func(self.elem))
        elif func is torch.ops.aten.clone.default:
            self, = args
            return cls(func(self.elem))
        elif func is torch.ops.aten.copy_.default:
            self, src = args
            self.elem.copy_(src.elem)
            return self
        elif func is torch.ops.aten.view.dtype:
            self, dtype = args
            if dtype == torch.uint8:
                return self.elem
        elif func is torch.ops.aten._to_copy.default:
            self, = args
            dtype = find_arg_of_type(itertools.chain(args, kwargs.values()), torch.dtype)
            device = find_arg_of_type(itertools.chain(args, kwargs.values()), torch.device)
            if device:
                assert dtype is None or dtype == torch.uint8
                return cls(self.elem)
        elif func is torch.ops.aten.slice.Tensor:
            self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
            if dim == self.dim() - 1:
                if step != 1:
                    raise NotImplementedError(f'slice step={step}')
                assert start % self.N == 0, start
                assert end >= self.shape[dim] or end % self.N == 0, end
                return cls(torch.ops.aten.slice.Tensor(self.elem, dim, start // self.N, end // self.N, 1))
            else:
                return cls(torch.ops.aten.slice.Tensor(self.elem, dim, start, end, step))
        if func is torch.ops.aten.view.default:
            self, size = args
            size = utils.infer_size(size, self.numel())
            assert not kwargs
            return cls(self.elem.reshape(down_size(self.N, size)))
        elif func is torch.ops.aten.select.int:
            self, dim, index = args
            if dim != self.dim() - 1:
                return cls(torch.ops.aten.select.int(self.elem, dim, index))
            else:
                raise NotImplementedError(f'select dim={dim}')
        raise NotImplementedError(f'{func} args:{args} kwargs:{kwargs}')


class UInt2Tensor(UIntXTensor):
    N: 'int' = 4

    @staticmethod
    def __new__(cls, elem: 'torch.Tensor'):
        return UIntXTensor.__new__(cls, cls.N, elem)

    def __init__(self, elem: 'torch.Tensor') ->None:
        super().__init__(UInt2Tensor.N, elem)


class UInt4Tensor(UIntXTensor):
    N: 'int' = 2

    @staticmethod
    def __new__(cls, elem: 'torch.Tensor'):
        return UIntXTensor.__new__(cls, cls.N, elem)

    def __init__(self, elem: 'torch.Tensor') ->None:
        super().__init__(UInt4Tensor.N, elem)


@torch.fx.wrap
def _fx_trec_unwrap_kjt(kjt: 'KeyedJaggedTensor') ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Forced conversions to support TBE
    CPU - int32 or int64, offsets dtype must match
    GPU - int32 only, offsets dtype must match
    """
    indices = kjt.values()
    offsets = kjt.offsets()
    if kjt.device().type == 'cpu':
        return indices, offsets.type(dtype=indices.dtype)
    else:
        return indices.int(), offsets.int()


@torch.fx.wrap
def _get_batching_hinted_output(lengths: 'Tensor', output: 'Tensor') ->Tensor:
    return output


@torch.fx.wrap
def _get_feature_length(feature: 'KeyedJaggedTensor') ->Tensor:
    return feature.lengths()


@torch.fx.wrap
def _get_kjt_keys(feature: 'KeyedJaggedTensor') ->List[str]:
    return feature.keys()


@torch.fx.wrap
def _get_unflattened_lengths(lengths: 'torch.Tensor', num_features: 'int') ->torch.Tensor:
    """
    Unflatten lengths tensor from [F * B] to [F, B].
    """
    return lengths.view(num_features, -1)


class QuantConfig(NamedTuple):
    activation: 'torch.quantization.PlaceholderObserver'
    weight: 'torch.quantization.PlaceholderObserver'
    per_table_weight_dtype: 'Optional[Dict[str, torch.dtype]]' = None


def dtype_to_data_type(dtype: 'torch.dtype') ->DataType:
    if dtype == torch.float:
        return DataType.FP32
    elif dtype == torch.float16 or dtype == torch.half:
        return DataType.FP16
    elif dtype == torch.bfloat16:
        return DataType.BF16
    elif dtype in {torch.int, torch.int32}:
        return DataType.INT32
    elif dtype in {torch.long, torch.int64}:
        return DataType.INT64
    elif dtype in {torch.quint8, torch.qint8, torch.int8}:
        return DataType.INT8
    elif dtype == torch.uint8:
        return DataType.UINT8
    elif dtype == torch.quint4x2:
        return DataType.INT4
    elif dtype == torch.quint2x4:
        return DataType.INT2
    else:
        raise Exception(f'Invalid data type {dtype}')


def _update_embedding_configs(embedding_configs: 'List[BaseEmbeddingConfig]', quant_config: 'Union[QuantConfig, torch.quantization.QConfig]', tables_to_rows_post_pruning: 'Optional[Dict[str, int]]'=None) ->None:
    per_table_weight_dtype = quant_config.per_table_weight_dtype if isinstance(quant_config, QuantConfig) and quant_config.per_table_weight_dtype else {}
    for config in embedding_configs:
        config.data_type = dtype_to_data_type(per_table_weight_dtype[config.name] if config.name in per_table_weight_dtype else quant_config.weight().dtype)
        if tables_to_rows_post_pruning and config.name in tables_to_rows_post_pruning:
            config.num_embeddings_post_pruning = tables_to_rows_post_pruning[config.name]


@torch.fx.wrap
def _fx_to_list(tensor: 'torch.Tensor') ->List[int]:
    return tensor.long().tolist()


def _permute_indices(indices: 'List[int]', permute: 'List[int]') ->List[int]:
    permuted_indices = [0] * len(indices)
    for i, permuted_index in enumerate(permute):
        permuted_indices[i] = indices[permuted_index]
    return permuted_indices


@torch.fx.wrap
def _slice_1d_tensor(tensor: 'torch.Tensor', start: 'int', end: 'int') ->torch.Tensor:
    """
    Slice tensor.
    """
    return tensor[start:end]


def construct_jagged_tensors_inference(embeddings: 'torch.Tensor', lengths: 'torch.Tensor', values: 'torch.Tensor', embedding_names: 'List[str]', need_indices: 'bool'=False, features_to_permute_indices: 'Optional[Dict[str, List[int]]]'=None, reverse_indices: 'Optional[torch.Tensor]'=None, remove_padding: 'bool'=False) ->Dict[str, JaggedTensor]:
    with record_function('## construct_jagged_tensors_inference ##'):
        if reverse_indices is not None:
            embeddings = torch.index_select(embeddings, 0, reverse_indices)
        elif remove_padding:
            embeddings = _slice_1d_tensor(embeddings, 0, lengths.sum().item())
        ret: 'Dict[str, JaggedTensor]' = {}
        length_per_key: 'List[int]' = _fx_to_list(torch.sum(lengths, dim=1))
        lengths_tuple = torch.unbind(lengths, dim=0)
        embeddings_list = torch.split(embeddings, length_per_key, dim=0)
        values_list = torch.split(values, length_per_key) if need_indices else None
        key_indices = defaultdict(list)
        for i, key in enumerate(embedding_names):
            key_indices[key].append(i)
        for key, indices in key_indices.items():
            indices = _permute_indices(indices, features_to_permute_indices[key]) if features_to_permute_indices and key in features_to_permute_indices else indices
            ret[key] = JaggedTensor(lengths=lengths_tuple[indices[0]], values=embeddings_list[indices[0]] if len(indices) == 1 else torch.cat([embeddings_list[i] for i in indices], dim=1), weights=values_list[indices[0]] if need_indices else None)
        return ret


def get_embedding_names_by_table(tables: 'Union[List[EmbeddingBagConfig], List[EmbeddingConfig]]') ->List[List[str]]:
    shared_feature: 'Dict[str, bool]' = {}
    for embedding_config in tables:
        for feature_name in embedding_config.feature_names:
            if feature_name not in shared_feature:
                shared_feature[feature_name] = False
            else:
                shared_feature[feature_name] = True
    embedding_names_by_table: 'List[List[str]]' = []
    for embedding_config in tables:
        embedding_names: 'List[str]' = []
        for feature_name in embedding_config.feature_names:
            if shared_feature[feature_name]:
                embedding_names.append(feature_name + '@' + embedding_config.name)
            else:
                embedding_names.append(feature_name)
        embedding_names_by_table.append(embedding_names)
    return embedding_names_by_table


_T = TypeVar('_T')


def none_throws(optional: 'Optional[_T]', message: 'str'='Unexpected `None`') ->_T:
    """Convert an optional to its value. Raises an `AssertionError` if the
    value is `None`"""
    if optional is None:
        raise AssertionError(message)
    return optional


def quantize_state_dict(module: 'nn.Module', table_name_to_quantized_weights: 'Dict[str, Tuple[Tensor, Tensor]]', table_name_to_data_type: 'Dict[str, DataType]', table_name_to_num_embeddings_post_pruning: 'Optional[Dict[str, int]]'=None) ->torch.device:
    device = torch.device('cpu')
    if not table_name_to_num_embeddings_post_pruning:
        table_name_to_num_embeddings_post_pruning = {}
    for key, tensor in module.state_dict().items():
        splits = key.split('.')
        assert splits[-1] == 'weight'
        table_name = splits[-2]
        data_type = table_name_to_data_type[table_name]
        num_rows = tensor.shape[0]
        if table_name in table_name_to_num_embeddings_post_pruning:
            num_rows = table_name_to_num_embeddings_post_pruning[table_name]
        device = tensor.device
        num_bits = DATA_TYPE_NUM_BITS[data_type]
        if tensor.is_meta:
            quant_weight = torch.empty((num_rows, tensor.shape[1] * num_bits // 8), device='meta', dtype=torch.uint8)
            if data_type == DataType.INT8 or data_type == DataType.INT4 or data_type == DataType.INT2:
                scale_shift = torch.empty((num_rows, 4), device='meta', dtype=torch.uint8)
            else:
                scale_shift = None
        else:
            if num_rows != tensor.shape[0]:
                tensor = tensor[:num_rows, :]
            if tensor.dtype == torch.float or tensor.dtype == torch.float16:
                if data_type == DataType.FP16:
                    if tensor.dtype == torch.float:
                        tensor = tensor.half()
                    quant_res = tensor.view(torch.uint8)
                else:
                    quant_res = torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf(tensor, num_bits)
            else:
                raise Exception('Unsupported dtype: {tensor.dtype}')
            if data_type == DataType.INT8 or data_type == DataType.INT4 or data_type == DataType.INT2:
                quant_weight, scale_shift = quant_res[:, :-4], quant_res[:, -4:]
            else:
                quant_weight, scale_shift = quant_res, None
        table_name_to_quantized_weights[table_name] = quant_weight, scale_shift
    return device


class Model(nn.Module):

    def __init__(self, num_embeddings, init_max, init_min, batch_size):
        super().__init__()
        self.embedding_dim = 16
        self.batch_size = batch_size
        self.config = EmbeddingConfig(name='id', embedding_dim=self.embedding_dim, num_embeddings=num_embeddings, weight_init_max=init_max, weight_init_min=init_min)
        self.emb = EmbeddingCollection(tables=[self.config], device=torch.device('meta'))
        self.dense = nn.Linear(16, 1)

    def forward(self, x):
        embeddings = self.emb(x)['id'].values().reshape((self.batch_size, -1, self.embedding_dim))
        fused = embeddings.sum(dim=1)
        output = self.dense(fused)
        pred = torch.sigmoid(output)
        return pred


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention

    Args:
        query (torch.Tensor): the query tensor
        key (torch.Tensor): the key tensor
        value (torch.Tensor): the value tensor
        mask (torch.Tensor): the mask tensor
        dropout (nn.Dropout): the dropout layer

    Example::

        self.attention = Attention()
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
    """

    def forward(self, query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None, dropout: 'Optional[nn.Dropout]'=None) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        forward function

        Args:
            query (torch.Tensor): the query tensor
            key (torch.Tensor): the key tensor
            value (torch.Tensor): the value tensor
            mask (torch.Tensor): the mask tensor
            dropout (nn.Dropout): the dropout layer

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1000000000.0)
        p_attn = nn.functional.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Multi-headed attention block.

    Args:
        num_heads (int): number of attention heads
        dim_model (int): input/output dimensionality
        dropout (float): the dropout probability
        mask (torch.Tensor): the mask tensor
        device: (Optional[torch.device]).

    Example::

        self.attention = MultiHeadedAttention(
            num_heads=attn_heads, dim_model=hidden, dropout=dropout, device=device
        )
        self.attention.forward(query, key, value, mask=mask)
    """

    def __init__(self, num_heads: 'int', dim_model: 'int', dropout: 'float'=0.1, device: 'Optional[torch.device]'=None) ->None:
        super().__init__()
        assert dim_model % num_heads == 0
        self.d_k: 'int' = dim_model // num_heads
        self.num_heads = num_heads
        self.linear_layers = nn.ModuleList([nn.Linear(dim_model, dim_model, device=device) for _ in range(3)])
        self.output_linear = nn.Linear(dim_model, dim_model, device=device)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """
        forward function

        Args:
            query (torch.Tensor): the query tensor
            key (torch.Tensor): the key tensor
            value (torch.Tensor): the value tensor
            mask (torch.Tensor): the mask tensor

        Returns:
            torch.Tensor.
        """
        batch_size = query.size(0)
        query, key, value = [linearLayer(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2) for linearLayer, x in zip(self.linear_layers, (query, key, value))]
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    """
    Feed-forward block.

    Args:
        dim_model (int): input/output dimensionality
        d_ff (int): hidden dimensionality
        dropout (float): the dropout probability
        device: (Optional[torch.device]).

    Example::

        self.feed_forward = PositionwiseFeedForward(
            dim_model=hidden, d_ff=feed_forward_hidden, dropout=dropout, device=device
        )
    """

    def __init__(self, dim_model: 'int', d_ff: 'int', dropout: 'float'=0.1, device: 'Optional[torch.device]'=None) ->None:
        super().__init__()
        self.w_1 = nn.Linear(dim_model, d_ff, device=device)
        self.w_2 = nn.Linear(d_ff, dim_model, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        forward function including the first Linear layer, ReLu, Dropout
        and the final Linear layer

        Args:
            x (torch.Tensor): the input tensor

        Returns:
            torch.Tensor.
        """
        return self.w_2(self.dropout(nn.functional.relu(self.w_1(x))))


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.

    Args:
        size (int): layerNorm size
        dropout (float): the dropout probability
        device: (Optional[torch.device]).

    Example::

        self.input_sublayer = SublayerConnection(
            size=hidden, dropout=dropout, device=device
        )
    """

    def __init__(self, size: 'int', dropout: 'float', device: 'Optional[torch.device]'=None) ->None:
        super().__init__()
        self.norm = nn.LayerNorm(size, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: 'torch.Tensor', sublayer: 'Callable[[torch.Tensor], torch.Tensor]') ->torch.Tensor:
        """
        forward function including the norm layer, sublayer and dropout, finally
        add up with the input tensor

        Args:
            x (torch.Tensor): the input tensor
            sublayer (Callable[[torch.Tensor): callable layer

        Returns:
            torch.Tensor.
        """
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection

    Args:
        hidden (int): hidden size of transformer
        attn_heads (int): head sizes of multi-head attention
        feed_forward_hidden (int): feed_forward_hidden, usually 4*hidden_size
        dropout (float): the dropout probability
        device: (Optional[torch.device]).

    Example::

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(emb_dim, nhead, emb_dim * 4, dropout, device=device)
                for _ in range(num_layers)
            ]
        )
    """

    def __init__(self, hidden: 'int', attn_heads: 'int', feed_forward_hidden: 'int', dropout: 'float', device: 'Optional[torch.device]'=None) ->None:
        super().__init__()
        self.attention = MultiHeadedAttention(num_heads=attn_heads, dim_model=hidden, dropout=dropout, device=device)
        self.feed_forward = PositionwiseFeedForward(dim_model=hidden, d_ff=feed_forward_hidden, dropout=dropout, device=device)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout, device=device)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout, device=device)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: 'torch.Tensor', mask: 'torch.BoolTensor') ->torch.Tensor:
        """
        forward function

        Args:
            x (torch.Tensor): the input tensor
            mask (torch.BoolTensor): determine which position has been masked

        Returns:
            torch.Tensor.
        """
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class HistoryArch(torch.nn.Module):
    """
    embedding.HistoryArch is the input embedding layer the BERT4Rec model
    as described in Section 3.4 of the paper.  It consits of an item
    embedding table and a positional embedding table.

    The item embedding table consists of vocab_size vectors.  The
    positional embedding table has history_len vectors.  Both
    kinds of embedding vectors have length emb_dim.  As mentioned
    in Section 3.7, BERT4Rec differs from BERT lacking the
    sentence embedding.

    Note that for the item embedding table, we have applied TorchRec
    EmbeddingCollection which supports sharding

    Args:
        vocab_size (int): the item count including mask and padding
        history_len (int): the max length
        emb_dim (int): embedding dimension
        dropout (float): the dropout probability
        device: (Optional[torch.device]).

    Example::

        self.history = HistoryArch(
            vocab_size, max_len, emb_dim, dropout=dropout, device=device
        )
        x = self.history(input)
    """

    def __init__(self, vocab_size: 'int', history_len: 'int', emb_dim: 'int', dropout: 'float'=0.1, device: 'Optional[torch.device]'=None) ->None:
        super().__init__()
        self.emb_dim = emb_dim
        self.history_len = history_len
        self.positional = nn.Parameter(torch.randn(history_len, emb_dim, device=device))
        self.layernorm = torch.nn.LayerNorm([history_len, emb_dim], device=device)
        self.dropout = torch.nn.Dropout(p=dropout)
        item_embedding_config = EmbeddingConfig(name='item_embedding', embedding_dim=emb_dim, num_embeddings=vocab_size, feature_names=['item'], weight_init_max=1.0, weight_init_min=-1.0)
        self.ec = EmbeddingCollection(tables=[item_embedding_config], device=device)

    def forward(self, id_list_features: 'KeyedJaggedTensor') ->torch.Tensor:
        """
        forward function: first query the item embedding and do the padding
        then add up with positional parameters and do the norm and dropout

        Args:
            id_list_features (KeyedJaggedTensor): the input KeyedJaggedTensor

        Returns:
            torch.Tensor.
        """
        jt_dict = self.ec(id_list_features)
        padded_embeddings = [torch.ops.fbgemm.jagged_2d_to_dense(values=jt_dict[e].values(), offsets=jt_dict[e].offsets(), max_sequence_length=self.history_len).view(-1, self.history_len, self.emb_dim) for e in id_list_features.keys()]
        item_output = torch.cat(padded_embeddings, dim=1)
        batch_size = id_list_features.stride()
        positional_output = self.positional.unsqueeze(0).repeat(batch_size, 1, 1)
        x = item_output + positional_output
        return self.dropout(self.layernorm(x))


class BERT4Rec(nn.Module):
    """
    The overall arch described in the BERT4Rec paper: (https://arxiv.org/abs/1904.06690)
    the encoder_layer was described in the section of 3.3, the output_layer was described in the
    section of 3.5

    Args:
        vocab_size (int): the item count including mask and padding
        max_len (int): the max length
        emb_dim (int): embedding dimension
        nhead (int): number of the transformation headers
        num_layers (int): number of the transformation layers
        dropout (float): the dropout probability
        device: (Optional[torch.device]).

    Example::

        input_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["item"],
            values=torch.tensor([2, 4, 3, 4, 5]),
            lengths=torch.tensor([2, 3]),
        )
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        input_kjt = input_kjt.to(device)
        bert4rec = BERT4Rec(
            vocab_size=6, max_len=3, emb_dim=4, nhead=4, num_layers=4, device=device
        )
        logits = bert4rec(input_kjt)
    """

    def __init__(self, vocab_size: 'int', max_len: 'int', emb_dim: 'int', nhead: 'int', num_layers: 'int', dropout: 'float'=0.1, device: 'Optional[torch.device]'=None) ->None:
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.history = HistoryArch(vocab_size, max_len, emb_dim, dropout=dropout, device=device)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(emb_dim, nhead, emb_dim * 4, dropout, device=device) for _ in range(num_layers)])
        self.out = nn.Linear(self.emb_dim, self.vocab_size, device=device)

    def forward(self, input: 'KeyedJaggedTensor') ->torch.Tensor:
        """
        forward function: first get the item embedding result and
        fit into transformer blocks and fit into the last linaer
        layer to get the final output

        Args:
            input (KeyedJaggedTensor): the input KeyedJaggedTensor

        Returns:
            torch.Tensor.
        """
        dense_tensor = input['item'].to_padded_dense(desired_length=self.max_len)
        mask = (dense_tensor > 0).unsqueeze(1).repeat(1, dense_tensor.size(1), 1).unsqueeze(1)
        x = self.history(input)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        return self.out(x)


class Perceptron(torch.nn.Module):
    """
    Applies a linear transformation and activation.

    Args:
        in_size (int): number of elements in each input sample.
        out_size (int): number of elements in each output sample.
        bias (bool): if set to ``False``, the layer will not learn an additive bias.
            Default: ``True``.
        activation (Union[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]):
            the activation function to apply to the output of linear transformation.
            Default: torch.relu.
        device (Optional[torch.device]): default compute device.

    Example::

        batch_size = 3
        in_size = 40
        input = torch.randn(batch_size, in_size)

        out_size = 16
        perceptron = Perceptron(in_size, out_size, bias=True)

        output = perceptron(input)
        assert list(output) == [batch_size, out_size]
    """

    def __init__(self, in_size: 'int', out_size: 'int', bias: 'bool'=True, activation: 'Union[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]'=torch.relu, device: 'Optional[torch.device]'=None, dtype: 'torch.dtype'=torch.float32) ->None:
        super().__init__()
        torch._C._log_api_usage_once(f'torchrec.modules.{self.__class__.__name__}')
        self._out_size = out_size
        self._in_size = in_size
        self._linear: 'nn.Linear' = nn.Linear(self._in_size, self._out_size, bias=bias, device=device, dtype=dtype)
        self._activation_fn: 'Callable[[torch.Tensor], torch.Tensor]' = activation

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            input (torch.Tensor): tensor of shape (B, I) where I is number of elements
                in each input sample.

        Returns:
            torch.Tensor: tensor of shape (B, O) where O is number of elements per
                channel in each output sample (i.e. `out_size`).
        """
        return self._activation_fn(self._linear(input))


class SwishLayerNorm(nn.Module):
    """
    Applies the Swish function with layer normalization: `Y = X * Sigmoid(LayerNorm(X))`.

    Args:
        input_dims (Union[int, List[int], torch.Size]): dimensions to normalize over.
            If an input tensor has shape [batch_size, d1, d2, d3], setting
            input_dim=[d2, d3] will do the layer normalization on last two dimensions.
        device (Optional[torch.device]): default compute device.

    Example::

        sln = SwishLayerNorm(100)
    """

    def __init__(self, input_dims: 'Union[int, List[int], torch.Size]', device: 'Optional[torch.device]'=None) ->None:
        super().__init__()
        self.norm: 'torch.nn.modules.Sequential' = nn.Sequential(nn.LayerNorm(input_dims, device=device), nn.Sigmoid())

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            input (torch.Tensor): an input tensor.

        Returns:
            torch.Tensor: an output tensor.
        """
        return input * self.norm(input)


def extract_module_or_tensor_callable(module_or_callable: 'Union[Callable[[], torch.nn.Module], torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]') ->Union[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    try:
        module = module_or_callable()
        if isinstance(module, torch.nn.Module):
            return module
        else:
            raise ValueError('Expected callable that takes no input to return a torch.nn.Module, but got: {}'.format(type(module)))
    except TypeError as e:
        if 'required positional argument' in str(e):
            return module_or_callable
        raise


class MLP(torch.nn.Module):
    """
    Applies a stack of Perceptron modules sequentially (i.e. Multi-Layer Perceptron).

    Args:
        in_size (int): `in_size` of the input.
        layer_sizes (List[int]): `out_size` of each Perceptron module.
        bias (bool): if set to False, the layer will not learn an additive bias.
            Default: True.
        activation (str, Union[Callable[[], torch.nn.Module], torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]):
            the activation function to apply to the output of linear transformation of
            each Perceptron module.
            If `activation` is a `str`, we currently only support the follow strings, as
            "relu", "sigmoid", and "swish_layernorm".
            If `activation` is a `Callable[[], torch.nn.Module]`, `activation()` will be
            called once per Perceptron module to generate the activation module for that
            Perceptron module, and the parameters won't be shared between those activation
            modules.
            One use case is when all the activation modules share the same constructor
            arguments, but don't share the actual module parameters.
            Default: torch.relu.
        device (Optional[torch.device]): default compute device.

    Example::

        batch_size = 3
        in_size = 40
        input = torch.randn(batch_size, in_size)

        layer_sizes = [16, 8, 4]
        mlp_module = MLP(in_size, layer_sizes, bias=True)
        output = mlp_module(input)
        assert list(output.shape) == [batch_size, layer_sizes[-1]]
    """

    def __init__(self, in_size: 'int', layer_sizes: 'List[int]', bias: 'bool'=True, activation: 'Union[str, Callable[[], torch.nn.Module], torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]'=torch.relu, device: 'Optional[torch.device]'=None, dtype: 'torch.dtype'=torch.float32) ->None:
        super().__init__()
        if activation == 'relu':
            activation = torch.relu
        elif activation == 'sigmoid':
            activation = torch.sigmoid
        if not isinstance(activation, str):
            self._mlp: 'torch.nn.Module' = torch.nn.Sequential(*[Perceptron(layer_sizes[i - 1] if i > 0 else in_size, layer_sizes[i], bias=bias, activation=extract_module_or_tensor_callable(activation), device=device, dtype=dtype) for i in range(len(layer_sizes))])
        elif activation == 'swish_layernorm':
            self._mlp: 'torch.nn.Module' = torch.nn.Sequential(*[Perceptron(layer_sizes[i - 1] if i > 0 else in_size, layer_sizes[i], bias=bias, activation=SwishLayerNorm(layer_sizes[i], device=device), device=device) for i in range(len(layer_sizes))])
        else:
            assert ValueError, 'This MLP only support str version activation function of relu, sigmoid, and swish_layernorm'

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            input (torch.Tensor): tensor of shape (B, I) where I is number of elements
                in each input sample.

        Returns:
            torch.Tensor: tensor of shape (B, O) where O is `out_size` of the last Perceptron module.
        """
        return self._mlp(input)


class TwoTower(nn.Module):
    """
    Simple TwoTower (UV) Model. Embeds two different entities into the same space.
    A simplified version of the `A Dual Augmented Two-tower Model for Online Large-scale Recommendation
    <https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_4.pdf>`_ model.
    Used to train the retrieval model

    Embeddings trained with this model will be indexed and queried in the retrieval example.

    Args:
        embedding_bag_collection (EmbeddingBagCollection): embedding_bag_collection with two EmbeddingBags
        layer_sizes (List[int]): list of the layer_sizes for the MLP
        device (Optional[torch.device])

    Example::

        m = TwoTower(ebc, [16, 8], device)
    """

    def __init__(self, embedding_bag_collection: 'EmbeddingBagCollection', layer_sizes: 'List[int]', device: 'Optional[torch.device]'=None) ->None:
        super().__init__()
        assert len(embedding_bag_collection.embedding_bag_configs()) == 2, 'Expected two EmbeddingBags in the two tower model'
        assert embedding_bag_collection.embedding_bag_configs()[0].embedding_dim == embedding_bag_collection.embedding_bag_configs()[1].embedding_dim, 'Both EmbeddingBagConfigs must have the same dimension'
        embedding_dim: 'int' = embedding_bag_collection.embedding_bag_configs()[0].embedding_dim
        self._feature_names_query: 'List[str]' = embedding_bag_collection.embedding_bag_configs()[0].feature_names
        self._candidate_feature_names: 'List[str]' = embedding_bag_collection.embedding_bag_configs()[1].feature_names
        self.ebc = embedding_bag_collection
        self.query_proj = MLP(in_size=embedding_dim, layer_sizes=layer_sizes, device=device)
        self.candidate_proj = MLP(in_size=embedding_dim, layer_sizes=layer_sizes, device=device)

    def forward(self, kjt: 'KeyedJaggedTensor') ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            kjt (KeyedJaggedTensor): KJT containing query_ids and candidate_ids to query

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing embeddings for each tower
        """
        pooled_embeddings = self.ebc(kjt)
        query_embedding: 'torch.Tensor' = self.query_proj(torch.cat([pooled_embeddings[feature] for feature in self._feature_names_query], dim=1))
        candidate_embedding: 'torch.Tensor' = self.candidate_proj(torch.cat([pooled_embeddings[feature] for feature in self._candidate_feature_names], dim=1))
        return query_embedding, candidate_embedding


class TwoTowerTrainTask(nn.Module):
    """
    Train Task for the TwoTower model. Adds BinaryCrossEntropy Loss.  to use with train_pipeline

    Args:
        two_tower (TwoTower): two tower model

    Example::

        m = TwoTowerTrainTask(two_tower_model)
    """

    def __init__(self, two_tower: 'TwoTower') ->None:
        super().__init__()
        self.two_tower = two_tower
        self.loss_fn: 'nn.Module' = nn.BCEWithLogitsLoss()

    def forward(self, batch: 'Batch') ->Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            batch (Batch): batch from torchrec.datasets

        Returns:
            Tuple[loss, Tuple[loss, logits, labels]]: each of shape B x 1
        """
        query_embedding, candidate_embedding = self.two_tower(batch.sparse_features)
        logits = (query_embedding * candidate_embedding).sum(dim=1).squeeze()
        loss = self.loss_fn(logits, batch.labels.float())
        return loss, (loss.detach(), logits.detach(), batch.labels.detach())


def _get_inverse_indices_or_throw(inverse_indices: 'Optional[Tuple[List[str], torch.Tensor]]') ->Tuple[List[str], torch.Tensor]:
    assert inverse_indices is not None, "This KJT doesn't have inverse indices."
    return inverse_indices


def _get_lengths_offset_per_key_or_throw(lengths_offset_per_key: 'Optional[List[int]]') ->List[int]:
    assert lengths_offset_per_key is not None, "This (Keyed)JaggedTensor doesn't have lengths_offset_per_key."
    return lengths_offset_per_key


def _get_stride_per_key_or_throw(stride_per_key: 'Optional[List[int]]') ->List[int]:
    assert stride_per_key is not None, "This (Keyed)JaggedTensor doesn't have stride_per_key."
    return stride_per_key


def _jagged_tensor_string(key: 'str', values: 'torch.Tensor', weights: 'Optional[torch.Tensor]', offsets: 'torch.Tensor', offset_start: 'int', offset_end: 'int') ->str:
    if weights is None:
        return '"{}": '.format(key) + _jagged_values_string(values, offsets, offset_start, offset_end)
    return '"{}"'.format(key) + """: {
        "values": """ + _jagged_values_string(values, offsets, offset_start, offset_end) + """,
        "weights": """ + _jagged_values_string(_get_weights_or_throw(weights), offsets, offset_start, offset_end) + '\n    }'

