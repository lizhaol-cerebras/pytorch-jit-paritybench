
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


import time


from typing import Any


import uuid


from torch import nn


from torch.utils._pytree import tree_map


import functools


import collections


import logging


from functools import wraps


from torch import multiprocessing as mp


from torch.distributed import rpc


from torch.utils.data import DataLoader


from torchvision import datasets


import torch.nn as nn


from copy import deepcopy


from torch import vmap


from typing import List


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


import inspect


import warnings


from typing import Callable


from typing import cast


from typing import TypeVar


import numpy as np


import numbers


import re


from collections import defaultdict


from copy import copy


from typing import Dict


from typing import Iterator


from typing import OrderedDict


from typing import Sequence


from typing import Tuple


from typing import Type


import torch.distributed as dist


from torch import Tensor


from numbers import Number


from typing import Iterable


from warnings import warn


from torch._dynamo import graph_break


from torch._functorch.vmap import _maybe_remove_batch_dim


from torch.nn.parameter import UninitializedTensorMixin


from torch.nn.utils._named_member_accessor import swap_tensor


from torch.nn.parameter import UninitializedBuffer


from torch.nn.parameter import UninitializedParameter


import abc


import enum


import queue


from collections.abc import MutableMapping


from functools import partial


from typing import Generator


from typing import Optional


from typing import overload


from typing import Union


from torch import distributed as dist


from torch.nn.parameter import Parameter


from typing import Mapping


from torch.utils._pytree import SUPPORTED_NODES


from torch import distributions as d


from torch import distributions as D


import math


from torch.distributions import constraints


from torch.distributions import Distribution


from torch.distributions.utils import broadcast_all


from inspect import signature


import torch.utils._pytree


import torch._functorch.vmap as vmap_src


from torch._functorch.vmap import _add_batch_dim


from torch._functorch.vmap import _broadcast_to_and_flatten


from torch._functorch.vmap import _get_name


from torch._functorch.vmap import _validate_and_get_batch_size


from torch._functorch.vmap import Tensor


from torch._functorch.vmap import tree_flatten


from torch._functorch.vmap import tree_unflatten


from torch.utils._contextlib import _DecoratorContextManager


from enum import Enum


from itertools import filterfalse


from itertools import tee


from torch import fx


from typing import get_type_hints


from torch.multiprocessing import Manager


from collections.abc import KeysView


from typing import TYPE_CHECKING


from torch._C import _disabled_torch_function_impl


from torch.utils.data._utils.worker import _generate_state


from torch.distributed._tensor import DeviceMesh


from torch.distributed._tensor import distribute_module


from torch.distributed._tensor import distribute_tensor


from torch.distributed._tensor import Shard


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


import copy


from torch import distributions


from torch.distributions import Normal


from torch._subclasses import FakeTensor


from torch._subclasses import FakeTensorMode


from torchvision.transforms import ToTensor


from torch import distributions as dists


from torch.export import export


from torchvision import transforms


from torch.utils.benchmark import Timer


from torch import distributions as dist


class InvAffine(nn.Module):
    """A custom normalization layer."""

    def __init__(self, loc, scale):
        super().__init__()
        self.loc = loc
        self.scale = scale

    def forward(self, x):
        return (x - self.loc) / self.scale


class RandomHFlip(nn.Module):

    def forward(self, x: 'torch.Tensor'):
        idx = torch.zeros([*x.shape[:-3], 1, 1, 1], device=x.device, dtype=torch.bool).bernoulli_().expand_as(x)
        return x.masked_fill(idx, 0.0) + x.masked_fill(~idx, 0.0).flip(-1)


class RandomCrop(nn.Module):

    def __init__(self, w, h):
        super(RandomCrop, self).__init__()
        self.w = w
        self.h = h

    def forward(self, x):
        batch = x.shape[:-3]
        index0 = torch.randint(x.shape[-2] - self.h, (*batch, 1), device=x.device)
        index0 = index0 + torch.arange(self.h, device=x.device)
        index0 = index0.unsqueeze(1).unsqueeze(-1).expand(*batch, 3, self.h, x.shape[-1])
        index1 = torch.randint(x.shape[-1] - self.w, (*batch, 1), device=x.device)
        index1 = index1 + torch.arange(self.w, device=x.device)
        index1 = index1.unsqueeze(1).unsqueeze(-2).expand(*batch, 3, self.h, self.w)
        return x.gather(-2, index0).gather(-1, index1)


class Collate(nn.Module):

    def __init__(self, transform=None, device=None):
        super().__init__()
        self.transform = transform
        self.device = device

    @torch.inference_mode()
    def __call__(self, x: 'ImageNetData'):
        out = x.apply(lambda _tensor: _tensor.as_tensor()).pin_memory()
        if self.transform:
            out.images = self.transform(out.images)
        return out


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(nn.Linear(28 * 28, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class Masker(nn.Module):

    def forward(self, x, mask):
        return torch.softmax(x * mask, dim=1)


class FCLayer(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.relu(self.fc(x))


class Output(nn.Module):

    def __init__(self, input_size, output_size=10):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=1)


class _BEST_ATTEMPT_INPLACE:

    def __bool__(self):
        raise NotImplementedError


BEST_ATTEMPT_INPLACE = _BEST_ATTEMPT_INPLACE()


CompatibleType = Union[Tensor,]


DeviceType = Union[torch.device, str, int]


LAST_OP_MAPS = {}


IndexType = Union[None, int, slice, str, Tensor, List[Any], Tuple[Any, ...]]


NESTED_TENSOR_ERR = "The PyTorch version isn't compatible with nested tensors. Please upgrade to a more recent version."


def _is_writable(file_path):
    file_path = str(file_path)
    if os.path.exists(file_path):
        return os.access(file_path, os.W_OK)
    return True


def _proc_args_const(*args, **kwargs):
    if len(args) > 0:
        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            shape = args[0]
        elif len(args) == 1 and not isinstance(args[0], int):
            shape = torch.Size(args[0])
        else:
            shape = torch.Size(args)
    else:
        shape = kwargs.pop('shape', None)
        if shape is None:
            raise TypeError('Could not find the shape argument in the arguments.')
        if not isinstance(shape, torch.Tensor):
            shape = torch.Size(shape)
    return shape, kwargs.pop('device', None), kwargs.pop('dtype', None), kwargs.pop('fill_value', None), kwargs.pop('filename', None)


def _shape(tensor: 'Tensor', nested_shape=False) ->torch.Size:
    if isinstance(tensor, UninitializedTensorMixin):
        return torch.Size([*getattr(tensor, 'batch_size', ()), -1])
    elif not isinstance(tensor, Tensor):
        if type(tensor) is KeyedJaggedTensor:
            return torch.Size([len(tensor.lengths()) // len(tensor.keys())])
        return tensor.shape
    if tensor.is_nested:
        if nested_shape:
            return tensor._nested_tensor_size()
        shape = []
        for i in range(tensor.ndim):
            try:
                shape.append(tensor.size(i))
            except RuntimeError:
                shape.append(-1)
        return torch.Size(shape)
    return tensor.shape


class implement_for:
    """A version decorator that checks the version in the environment and implements a function with the fitting one.

    If specified module is missing or there is no fitting implementation, call of the decorated function
    will lead to the explicit error.
    In case of intersected ranges, last fitting implementation is used.

    Args:
        module_name (str or callable): version is checked for the module with this
            name (e.g. "gym"). If a callable is provided, it should return the
            module.
        from_version: version from which implementation is compatible. Can be open (None).
        to_version: version from which implementation is no longer compatible. Can be open (None).

    Examples:
        >>> @implement_for("torch", None, "1.13")
        >>> def fun(self, x):
        ...     # Older torch versions will return x + 1
        ...     return x + 1
        ...
        >>> @implement_for("torch", "0.13", "2.0")
        >>> def fun(self, x):
        ...     # More recent torch versions will return x + 2
        ...     return x + 2
        ...
        >>> @implement_for(lambda: import_module("torch"), "0.", None)
        >>> def fun(self, x):
        ...     # More recent gym versions will return x + 2
        ...     return x + 2
        ...
        >>> @implement_for("gymnasium", "0.27", None)
        >>> def fun(self, x):
        ...     # If gymnasium is to be used instead of gym, x+3 will be returned
        ...     return x + 3
        ...

        This indicates that the function is compatible with gym 0.13+, but doesn't with gym 0.14+.
    """
    _implementations = {}
    _setters = []
    _cache_modules = {}

    def __init__(self, module_name: 'Union[str, Callable]', from_version: 'str'=None, to_version: 'str'=None):
        self.module_name = module_name
        self.from_version = from_version
        self.to_version = to_version
        implement_for._setters.append(self)

    @staticmethod
    def check_version(version: 'str', from_version: 'str | None', to_version: 'str | None'):
        version = parse('.'.join([str(v) for v in parse(version).release]))
        return (from_version is None or version >= parse(from_version)) and (to_version is None or version < parse(to_version))

    @staticmethod
    def get_class_that_defined_method(f):
        """Returns the class of a method, if it is defined, and None otherwise."""
        return f.__globals__.get(f.__qualname__.split('.')[0])

    @classmethod
    def get_func_name(cls, fn):
        first = str(fn).split('.')[0][len('<function '):]
        last = str(fn).split('.')[1:]
        if last:
            first = [first]
            last[-1] = last[-1].split(' ')[0]
        else:
            last = [first.split(' ')[0]]
            first = []
        return '.'.join([fn.__module__] + first + last)

    def _get_cls(self, fn):
        cls = self.get_class_that_defined_method(fn)
        if cls is None:
            return
        if type(cls).__name__ == 'function':
            cls = inspect.getmodule(fn)
        return cls

    def module_set(self):
        """Sets the function in its module, if it exists already."""
        prev_setter = type(self)._implementations.get(self.get_func_name(self.fn))
        if prev_setter is not None:
            prev_setter.do_set = False
        type(self)._implementations[self.get_func_name(self.fn)] = self
        cls = self.get_class_that_defined_method(self.fn)
        if cls is not None:
            if type(cls).__name__ == 'function':
                cls = inspect.getmodule(self.fn)
        else:
            return
        setattr(cls, self.fn.__name__, self.fn)

    @classmethod
    def import_module(cls, module_name: 'Union[Callable, str]') ->str:
        """Imports module and returns its version."""
        if not callable(module_name):
            module = cls._cache_modules.get(module_name)
            if module is None:
                if module_name in sys.modules:
                    sys.modules[module_name] = module = import_module(module_name)
                else:
                    cls._cache_modules[module_name] = module = import_module(module_name)
        else:
            module = module_name()
        return module.__version__
    _lazy_impl = collections.defaultdict(list)

    def _delazify(self, func_name):
        for local_call in implement_for._lazy_impl[func_name]:
            out = local_call()
        return out

    def __call__(self, fn):
        self.func_name = self.get_func_name(fn)
        self.fn = fn
        implement_for._lazy_impl[self.func_name].append(self._call)

        @wraps(fn)
        def _lazy_call_fn(*args, **kwargs):
            return self._delazify(self.func_name)(*args, **kwargs)
        return _lazy_call_fn

    def _call(self):
        fn = self.fn
        func_name = self.func_name
        implementations = implement_for._implementations

        @wraps(fn)
        def unsupported(*args, **kwargs):
            raise ModuleNotFoundError(f"Supported version of '{func_name}' has not been found.")
        self.do_set = False
        if func_name in implementations:
            try:
                version = self.import_module(self.module_name)
                if self.check_version(version, self.from_version, self.to_version):
                    self.do_set = True
                if not self.do_set:
                    return implementations[func_name].fn
            except ModuleNotFoundError:
                return implementations[func_name].fn
        else:
            try:
                version = self.import_module(self.module_name)
                if self.check_version(version, self.from_version, self.to_version):
                    self.do_set = True
            except ModuleNotFoundError:
                return unsupported
        if self.do_set:
            self.module_set()
            return fn
        return unsupported

    @classmethod
    def reset(cls, setters_dict: 'Dict[str, implement_for]'=None):
        """Resets the setters in setter_dict.

        ``setter_dict`` is a copy of implementations. We just need to iterate through its
        values and call :meth:`~.module_set` for each.

        """
        if setters_dict is None:
            setters_dict = copy(cls._implementations)
        for setter in setters_dict.values():
            setter.module_set()

    def __repr__(self):
        return f'{type(self).__name__}(module_name={self.module_name}({self.from_version, self.to_version}), fn_name={self.fn.__name__}, cls={self._get_cls(self.fn)}, is_set={self.do_set})'


class _NoDefault(enum.IntEnum):
    ZERO = 0


NO_DEFAULT = _NoDefault.ZERO

