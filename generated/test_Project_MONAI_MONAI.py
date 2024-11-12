
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


import warnings


from types import ModuleType


import torch


from copy import deepcopy


from time import sleep


from typing import Any


from typing import cast


import re


import time


import numpy as np


from torch.multiprocessing import get_context


from abc import ABC


from abc import abstractmethod


from collections.abc import Mapping


from collections.abc import Sequence


from warnings import warn


import torch.distributed as dist


from collections.abc import Hashable


from collections.abc import Callable


import random


from collections.abc import Sized


from collections.abc import Iterable


import logging as logger


from torch import Tensor


from torch import nn


import math


from typing import TypeVar


from typing import List


from typing import Sequence


import torch.nn.functional as F


import torch.nn as nn


from typing import Callable


from torch.nn import CrossEntropyLoss


from torch.nn import functional as F


from torch.nn.modules.loss import _Loss


import copy


from numpy import ndarray


from numbers import Number


from typing import TYPE_CHECKING


from functools import partial


from torch.cuda import is_available


from collections import OrderedDict


from typing import TextIO


from collections.abc import Collection


from typing import Union


import inspect


from torch.utils.data import DataLoader as _TorchDataLoader


from torch.utils.data import Dataset


import collections.abc


from copy import copy


from inspect import signature


from typing import IO


from torch.multiprocessing import Manager


from torch.serialization import DEFAULT_PROTOCOL


from torch.utils.data import Dataset as _TorchDataset


from torch.utils.data import Subset


from itertools import chain


from collections.abc import Generator


from collections.abc import Iterator


from torch.utils.data._utils.collate import np_str_obj_array_pattern


from torch.utils.data import IterableDataset as _TorchIterableDataset


from torch.utils.data import get_worker_info


import itertools


import functools


from torch.utils.data import DistributedSampler as _TorchDistributedSampler


from queue import Empty


from queue import Full


from queue import Queue


from collections import abc


from collections import defaultdict


from functools import reduce


from itertools import product


from itertools import starmap


from itertools import zip_longest


from torch.utils.data._utils.collate import default_collate


from torch.utils.data import IterableDataset


from torch.utils.data import DataLoader


from torch.optim.optimizer import Optimizer


from torch.utils.data.distributed import DistributedSampler


from collections.abc import MutableMapping


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.optim.lr_scheduler import _LRScheduler


from inspect import _empty


from inspect import isclass


from typing import Optional


from torch.fft import fftn


from functools import lru_cache


from typing import Tuple


from torch.nn.functional import softmax


from torch.nn import LayerNorm


from itertools import repeat


from typing import Iterable


from torch.utils import model_zoo


from torch.autograd import Function


import torch.nn


from torch.hub import load_state_dict_from_url


from torch.nn.functional import interpolate


from typing import NamedTuple


from typing import Dict


from typing import Final


import torch.utils.checkpoint as checkpoint


from types import MethodType


import types


from torch.optim import Optimizer


from torch.optim.lr_scheduler import LambdaLR


from math import ceil


from torch.nn.functional import pad as pad_pt


from torch.utils.data.dataloader import DataLoader as TorchDataLoader


from math import sqrt


from torch.nn.functional import grid_sample


from enum import Enum


from functools import wraps


from inspect import getmembers


from logging import Filter


from typing import Literal


from typing import overload


from math import log10


import enum


from re import match


from types import FunctionType


from torch.nn import Module


from collections import namedtuple


from inspect import getframeinfo


from inspect import stack


from time import perf_counter


from time import perf_counter_ns


from torch.nn.functional import pad


from torch.autograd import gradcheck


from torch.cuda.amp import autocast


from torch.optim import SGD


import torch.optim as optim


import torch.multiprocessing as mp


from numpy.fft import fftn


from numpy.fft import fftshift


import string


import torch.multiprocessing


from torch.autograd import Variable


from torch.nn.functional import avg_pool2d


import scipy.ndimage


from scipy.ndimage import zoom as zoom_scipy


from math import prod


import queue


INF = float('inf')


COMPUTE_DTYPE = torch.float32


NdarrayOrTensor = Union[np.ndarray, torch.Tensor]


TO_REMOVE = 0.0


NdarrayTensor = TypeVar('NdarrayTensor', bound=NdarrayOrTensor)


OPTIONAL_IMPORT_MSG_FMT = '{}'


class OptionalImportError(ImportError):
    """
    Could not import APIs from an optional dependency.
    """


def min_version(the_module: 'Any', min_version_str: 'str'='', *_args: Any) ->bool:
    """
    Convert version strings into tuples of int and compare them.

    Returns True if the module's version is greater or equal to the 'min_version'.
    When min_version_str is not provided, it always returns True.
    """
    if not min_version_str or not hasattr(the_module, '__version__'):
        return True
    mod_version = tuple(int(x) for x in the_module.__version__.split('.')[:2])
    required = tuple(int(x) for x in min_version_str.split('.')[:2])
    return mod_version >= required


def optional_import(module: 'str', version: 'str'='', version_checker: 'Callable[..., bool]'=min_version, name: 'str'='', descriptor: 'str'=OPTIONAL_IMPORT_MSG_FMT, version_args: 'Any'=None, allow_namespace_pkg: 'bool'=False, as_type: 'str'='default') ->tuple[Any, bool]:
    """
    Imports an optional module specified by `module` string.
    Any importing related exceptions will be stored, and exceptions raise lazily
    when attempting to use the failed-to-import module.

    Args:
        module: name of the module to be imported.
        version: version string used by the version_checker.
        version_checker: a callable to check the module version, Defaults to monai.utils.min_version.
        name: a non-module attribute (such as method/class) to import from the imported module.
        descriptor: a format string for the final error message when using a not imported module.
        version_args: additional parameters to the version checker.
        allow_namespace_pkg: whether importing a namespace package is allowed. Defaults to False.
        as_type: there are cases where the optionally imported object is used as
            a base class, or a decorator, the exceptions should raise accordingly. The current supported values
            are "default" (call once to raise), "decorator" (call the constructor and the second call to raise),
            and anything else will return a lazy class that can be used as a base class (call the constructor to raise).

    Returns:
        The imported module and a boolean flag indicating whether the import is successful.

    Examples::

        >>> torch, flag = optional_import('torch', '1.1')
        >>> print(torch, flag)
        <module 'torch' from 'python/lib/python3.6/site-packages/torch/__init__.py'> True

        >>> the_module, flag = optional_import('unknown_module')
        >>> print(flag)
        False
        >>> the_module.method  # trying to access a module which is not imported
        OptionalImportError: import unknown_module (No module named 'unknown_module').

        >>> torch, flag = optional_import('torch', '42', exact_version)
        >>> torch.nn  # trying to access a module for which there isn't a proper version imported
        OptionalImportError: import torch (requires version '42' by 'exact_version').

        >>> conv, flag = optional_import('torch.nn.functional', '1.0', name='conv1d')
        >>> print(conv)
        <built-in method conv1d of type object at 0x11a49eac0>

        >>> conv, flag = optional_import('torch.nn.functional', '42', name='conv1d')
        >>> conv()  # trying to use a function from the not successfully imported module (due to unmatched version)
        OptionalImportError: from torch.nn.functional import conv1d (requires version '42' by 'min_version').
    """
    tb = None
    exception_str = ''
    if name:
        actual_cmd = f'from {module} import {name}'
    else:
        actual_cmd = f'import {module}'
    try:
        pkg = __import__(module)
        the_module = import_module(module)
        if not allow_namespace_pkg:
            is_namespace = getattr(the_module, '__file__', None) is None and hasattr(the_module, '__path__')
            if is_namespace:
                raise AssertionError
        if name:
            the_module = getattr(the_module, name)
    except Exception as import_exception:
        tb = import_exception.__traceback__
        exception_str = f'{import_exception}'
    else:
        if version_args and version_checker(pkg, f'{version}', version_args):
            return the_module, True
        if not version_args and version_checker(pkg, f'{version}'):
            return the_module, True
    msg = descriptor.format(actual_cmd)
    if version and tb is None:
        msg += f" (requires '{module} {version}' by '{version_checker.__name__}')"
    if exception_str:
        msg += f' ({exception_str})'


    class _LazyRaise:

        def __init__(self, *_args, **_kwargs):
            _default_msg = f'{msg}.' + '\n\nFor details about installing the optional dependencies, please visit:' + '\n    https://docs.monai.io/en/latest/installation.html#installing-the-recommended-dependencies'
            if tb is None:
                self._exception = OptionalImportError(_default_msg)
            else:
                self._exception = OptionalImportError(_default_msg).with_traceback(tb)

        def __getattr__(self, name):
            """
            Raises:
                OptionalImportError: When you call this method.
            """
            raise self._exception

        def __call__(self, *_args, **_kwargs):
            """
            Raises:
                OptionalImportError: When you call this method.
            """
            raise self._exception

        def __getitem__(self, item):
            raise self._exception

        def __iter__(self):
            raise self._exception
    if as_type == 'default':
        return _LazyRaise(), False


    class _LazyCls(_LazyRaise):

        def __init__(self, *_args, **kwargs):
            super().__init__()
            if not as_type.startswith('decorator'):
                raise self._exception
    return _LazyCls, False


cp, has_cp = optional_import('cupy')


cp_ndarray, _ = optional_import('cupy', name='ndarray')


UNSUPPORTED_TYPES = {np.dtype('uint16'): torch.int32, np.dtype('uint32'): torch.int64, np.dtype('uint64'): torch.int64}


def dtype_numpy_to_torch(dtype: 'np.dtype') ->torch.dtype:
    """Convert a numpy dtype to its torch equivalent."""
    return torch.from_numpy(np.empty([], dtype=dtype)).dtype


def dtype_torch_to_numpy(dtype: 'torch.dtype') ->np.dtype:
    """Convert a torch dtype to its numpy equivalent."""
    return torch.empty([], dtype=dtype).numpy().dtype


def get_equivalent_dtype(dtype, data_type):
    """Convert to the `dtype` that corresponds to `data_type`.

    The input dtype can also be a string. e.g., `"float32"` becomes `torch.float32` or
    `np.float32` as necessary.

    Example::

        im = torch.tensor(1)
        dtype = get_equivalent_dtype(np.float32, type(im))

    """
    if dtype is None:
        return None
    if data_type is torch.Tensor or data_type.__name__ == 'MetaTensor':
        if isinstance(dtype, torch.dtype):
            return dtype
        return dtype_numpy_to_torch(dtype)
    if not isinstance(dtype, torch.dtype):
        return dtype
    return dtype_torch_to_numpy(dtype)


def get_dtype_bound_value(dtype: 'DtypeLike | torch.dtype') ->tuple[float, float]:
    """
    Get dtype bound value
    Args:
        dtype: dtype to get bound value
    Returns:
        (bound_min_value, bound_max_value)
    """
    if dtype in UNSUPPORTED_TYPES:
        is_floating_point = False
    else:
        is_floating_point = get_equivalent_dtype(dtype, torch.Tensor).is_floating_point
    dtype = get_equivalent_dtype(dtype, np.array)
    if is_floating_point:
        return np.finfo(dtype).min, np.finfo(dtype).max
    else:
        return np.iinfo(dtype).min, np.iinfo(dtype).max


def safe_dtype_range(data: 'Any', dtype: 'DtypeLike | torch.dtype'=None) ->Any:
    """
    Utility to safely convert the input data to target dtype.

    Args:
        data: input data can be PyTorch Tensor, numpy array, list, dictionary, int, float, bool, str, etc.
            will convert to target dtype and keep the original type.
            for dictionary, list or tuple, convert every item.
        dtype: target data type to convert.
    """

    def _safe_dtype_range(data, dtype):
        output_dtype = dtype if dtype is not None else data.dtype
        dtype_bound_value = get_dtype_bound_value(output_dtype)
        if data.ndim == 0:
            data_bound = data, data
        elif isinstance(data, torch.Tensor):
            data_bound = torch.min(data), torch.max(data)
        else:
            data_bound = np.min(data), np.max(data)
        if data_bound[1] > dtype_bound_value[1] or data_bound[0] < dtype_bound_value[0]:
            if isinstance(data, torch.Tensor):
                return torch.clamp(data, dtype_bound_value[0], dtype_bound_value[1])
            elif isinstance(data, np.ndarray):
                return np.clip(data, dtype_bound_value[0], dtype_bound_value[1])
            elif has_cp and isinstance(data, cp_ndarray):
                return cp.clip(data, dtype_bound_value[0], dtype_bound_value[1])
        else:
            return data
    if has_cp and isinstance(data, cp_ndarray):
        return cp.asarray(_safe_dtype_range(data, dtype))
    elif isinstance(data, np.ndarray):
        return np.asarray(_safe_dtype_range(data, dtype))
    elif isinstance(data, torch.Tensor):
        return _safe_dtype_range(data, dtype)
    elif isinstance(data, (float, int, bool)) and dtype is None:
        return data
    elif isinstance(data, (float, int, bool)) and dtype is not None:
        output_dtype = dtype
        dtype_bound_value = get_dtype_bound_value(output_dtype)
        data = dtype_bound_value[1] if data > dtype_bound_value[1] else data
        data = dtype_bound_value[0] if data < dtype_bound_value[0] else data
        return data
    elif isinstance(data, list):
        return [safe_dtype_range(i, dtype=dtype) for i in data]
    elif isinstance(data, tuple):
        return tuple(safe_dtype_range(i, dtype=dtype) for i in data)
    elif isinstance(data, dict):
        return {k: safe_dtype_range(v, dtype=dtype) for k, v in data.items()}
    return data


def convert_to_cupy(data: 'Any', dtype: 'np.dtype | None'=None, wrap_sequence: 'bool'=False, safe: 'bool'=False) ->Any:
    """
    Utility to convert the input data to a cupy array. If passing a dictionary, list or tuple,
    recursively check every item and convert it to cupy array.

    Args:
        data: input data can be PyTorch Tensor, numpy array, cupy array, list, dictionary, int, float, bool, str, etc.
            Tensor, numpy array, cupy array, float, int, bool are converted to cupy arrays,
            for dictionary, list or tuple, convert every item to a numpy array if applicable.
        dtype: target data type when converting to Cupy array, tt must be an argument of `numpy.dtype`,
            for more details: https://docs.cupy.dev/en/stable/reference/generated/cupy.array.html.
        wrap_sequence: if `False`, then lists will recursively call this function.
            E.g., `[1, 2]` -> `[array(1), array(2)]`. If `True`, then `[1, 2]` -> `array([1, 2])`.
        safe: if `True`, then do safe dtype convert when intensity overflow. default to `False`.
            E.g., `[256, -12]` -> `[array(0), array(244)]`. If `True`, then `[256, -12]` -> `[array(255), array(0)]`.
    """
    if safe:
        data = safe_dtype_range(data, dtype)
    if isinstance(data, torch.Tensor) and data.device.type == 'cuda':
        if data.dtype == torch.bool:
            data = data.detach()
            if dtype is None:
                dtype = bool
        data = cp.asarray(data, dtype)
    elif isinstance(data, (cp_ndarray, np.ndarray, torch.Tensor, float, int, bool)):
        data = cp.asarray(data, dtype)
    elif isinstance(data, list):
        list_ret = [convert_to_cupy(i, dtype) for i in data]
        return cp.asarray(list_ret) if wrap_sequence else list_ret
    elif isinstance(data, tuple):
        tuple_ret = tuple(convert_to_cupy(i, dtype) for i in data)
        return cp.asarray(tuple_ret) if wrap_sequence else tuple_ret
    elif isinstance(data, dict):
        return {k: convert_to_cupy(v, dtype) for k, v in data.items()}
    if not isinstance(data, cp.ndarray):
        raise ValueError(f'The input data type [{type(data)}] cannot be converted into cupy arrays!')
    if data.ndim > 0:
        data = cp.ascontiguousarray(data)
    return data


def convert_to_numpy(data: 'Any', dtype: 'DtypeLike'=None, wrap_sequence: 'bool'=False, safe: 'bool'=False) ->Any:
    """
    Utility to convert the input data to a numpy array. If passing a dictionary, list or tuple,
    recursively check every item and convert it to numpy array.

    Args:
        data: input data can be PyTorch Tensor, numpy array, list, dictionary, int, float, bool, str, etc.
            will convert Tensor, Numpy array, float, int, bool to numpy arrays, strings and objects keep the original.
            for dictionary, list or tuple, convert every item to a numpy array if applicable.
        dtype: target data type when converting to numpy array.
        wrap_sequence: if `False`, then lists will recursively call this function.
            E.g., `[1, 2]` -> `[array(1), array(2)]`. If `True`, then `[1, 2]` -> `array([1, 2])`.
        safe: if `True`, then do safe dtype convert when intensity overflow. default to `False`.
            E.g., `[256, -12]` -> `[array(0), array(244)]`. If `True`, then `[256, -12]` -> `[array(255), array(0)]`.
    """
    if safe:
        data = safe_dtype_range(data, dtype)
    if isinstance(data, torch.Tensor):
        data = np.asarray(data.detach().numpy(), dtype=get_equivalent_dtype(dtype, np.ndarray))
    elif has_cp and isinstance(data, cp_ndarray):
        data = cp.asnumpy(data).astype(dtype, copy=False)
    elif isinstance(data, (np.ndarray, float, int, bool)):
        if isinstance(data, np.ndarray) and data.ndim > 0 and data.dtype.itemsize < np.dtype(dtype).itemsize:
            data = np.ascontiguousarray(data)
        data = np.asarray(data, dtype=dtype)
    elif isinstance(data, list):
        list_ret = [convert_to_numpy(i, dtype=dtype) for i in data]
        return np.asarray(list_ret) if wrap_sequence else list_ret
    elif isinstance(data, tuple):
        tuple_ret = tuple(convert_to_numpy(i, dtype=dtype) for i in data)
        return np.asarray(tuple_ret) if wrap_sequence else tuple_ret
    elif isinstance(data, dict):
        return {k: convert_to_numpy(v, dtype=dtype) for k, v in data.items()}
    if isinstance(data, np.ndarray) and data.ndim > 0:
        data = np.ascontiguousarray(data)
    return data


def convert_to_tensor(data: 'Any', dtype: 'DtypeLike | torch.dtype'=None, device: 'None | str | torch.device'=None, wrap_sequence: 'bool'=False, track_meta: 'bool'=False, safe: 'bool'=False) ->Any:
    """
    Utility to convert the input data to a PyTorch Tensor, if `track_meta` is True, the output will be a `MetaTensor`,
    otherwise, the output will be a regular torch Tensor.
    If passing a dictionary, list or tuple, recursively check every item and convert it to PyTorch Tensor.

    Args:
        data: input data can be PyTorch Tensor, numpy array, list, dictionary, int, float, bool, str, etc.
            will convert Tensor, Numpy array, float, int, bool to Tensor, strings and objects keep the original.
            for dictionary, list or tuple, convert every item to a Tensor if applicable.
        dtype: target data type to when converting to Tensor.
        device: target device to put the converted Tensor data.
        wrap_sequence: if `False`, then lists will recursively call this function.
            E.g., `[1, 2]` -> `[tensor(1), tensor(2)]`. If `True`, then `[1, 2]` -> `tensor([1, 2])`.
        track_meta: whether to track the meta information, if `True`, will convert to `MetaTensor`.
            default to `False`.
        safe: if `True`, then do safe dtype convert when intensity overflow. default to `False`.
            E.g., `[256, -12]` -> `[tensor(0), tensor(244)]`.
            If `True`, then `[256, -12]` -> `[tensor(255), tensor(0)]`.

    """

    def _convert_tensor(tensor: 'Any', **kwargs: Any) ->Any:
        if not isinstance(tensor, torch.Tensor):
            if isinstance(tensor, np.ndarray) and tensor.dtype in UNSUPPORTED_TYPES:
                tensor = tensor.astype(UNSUPPORTED_TYPES[tensor.dtype])
            tensor = torch.as_tensor(tensor, **kwargs)
        if track_meta and not isinstance(tensor, monai.data.MetaTensor):
            return monai.data.MetaTensor(tensor)
        if not track_meta and isinstance(tensor, monai.data.MetaTensor):
            return tensor.as_tensor()
        return tensor
    if safe:
        data = safe_dtype_range(data, dtype)
    dtype = get_equivalent_dtype(dtype, torch.Tensor)
    if isinstance(data, torch.Tensor):
        return _convert_tensor(data)
    if isinstance(data, np.ndarray):
        if re.search('[SaUO]', data.dtype.str) is None:
            if data.ndim > 0:
                data = np.ascontiguousarray(data)
            return _convert_tensor(data, dtype=dtype, device=device)
    elif has_cp and isinstance(data, cp_ndarray) or isinstance(data, (float, int, bool)):
        return _convert_tensor(data, dtype=dtype, device=device)
    elif isinstance(data, list):
        list_ret = [convert_to_tensor(i, dtype=dtype, device=device, track_meta=track_meta) for i in data]
        return _convert_tensor(list_ret, dtype=dtype, device=device) if wrap_sequence else list_ret
    elif isinstance(data, tuple):
        tuple_ret = tuple(convert_to_tensor(i, dtype=dtype, device=device, track_meta=track_meta) for i in data)
        return _convert_tensor(tuple_ret, dtype=dtype, device=device) if wrap_sequence else tuple_ret
    elif isinstance(data, dict):
        return {k: convert_to_tensor(v, dtype=dtype, device=device, track_meta=track_meta) for k, v in data.items()}
    return data


SUPPORTED_SPATIAL_DIMS = [2, 3]


def min(x: 'NdarrayTensor', dim: 'int | tuple | None'=None, **kwargs) ->NdarrayTensor:
    """`torch.min` with equivalent implementation for numpy

    Args:
        x: array/tensor.

    Returns:
        the minimum of x.
    """
    ret: 'NdarrayTensor'
    if dim is None:
        ret = np.min(x, **kwargs) if isinstance(x, (np.ndarray, list)) else torch.min(x, **kwargs)
    elif isinstance(x, (np.ndarray, list)):
        ret = np.min(x, axis=dim, **kwargs)
    else:
        ret = torch.min(x, int(dim), **kwargs)
    return ret[0] if isinstance(ret, tuple) else ret


def damerau_levenshtein_distance(s1: 'str', s2: 'str') ->int:
    """
    Calculates the Damerau–Levenshtein distance between two strings for spelling correction.
    https://en.wikipedia.org/wiki/Damerau–Levenshtein_distance
    """
    if s1 == s2:
        return 0
    string_1_length = len(s1)
    string_2_length = len(s2)
    if not s1:
        return string_2_length
    if not s2:
        return string_1_length
    d = {(i, -1): (i + 1) for i in range(-1, string_1_length + 1)}
    for j in range(-1, string_2_length + 1):
        d[-1, j] = j + 1
    for i, s1i in enumerate(s1):
        for j, s2j in enumerate(s2):
            cost = 0 if s1i == s2j else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
            if i and j and s1i == s2[j - 1] and s1[i - 1] == s2j:
                d[i, j] = min(d[i, j], d[i - 2, j - 2] + cost)
    return d[string_1_length - 1, string_2_length - 1]


def look_up_option(opt_str: 'Hashable', supported: 'Collection | enum.EnumMeta', default: 'Any'='no_default', print_all_options: 'bool'=True) ->Any:
    """
    Look up the option in the supported collection and return the matched item.
    Raise a value error possibly with a guess of the closest match.

    Args:
        opt_str: The option string or Enum to look up.
        supported: The collection of supported options, it can be list, tuple, set, dict, or Enum.
        default: If it is given, this method will return `default` when `opt_str` is not found,
            instead of raising a `ValueError`. Otherwise, it defaults to `"no_default"`,
            so that the method may raise a `ValueError`.
        print_all_options: whether to print all available options when `opt_str` is not found. Defaults to True

    Examples:

    .. code-block:: python

        from enum import Enum
        from monai.utils import look_up_option
        class Color(Enum):
            RED = "red"
            BLUE = "blue"
        look_up_option("red", Color)  # <Color.RED: 'red'>
        look_up_option(Color.RED, Color)  # <Color.RED: 'red'>
        look_up_option("read", Color)
        # ValueError: By 'read', did you mean 'red'?
        # 'read' is not a valid option.
        # Available options are {'blue', 'red'}.
        look_up_option("red", {"red", "blue"})  # "red"

    Adapted from https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/utilities/util_common.py#L249
    """
    if not isinstance(opt_str, Hashable):
        raise ValueError(f'Unrecognized option type: {type(opt_str)}:{opt_str}.')
    if isinstance(opt_str, str):
        opt_str = opt_str.strip()
    if isinstance(supported, enum.EnumMeta):
        if isinstance(opt_str, str) and opt_str in {item.value for item in supported}:
            return supported(opt_str)
        if isinstance(opt_str, enum.Enum) and opt_str in supported:
            return opt_str
    elif isinstance(supported, Mapping) and opt_str in supported:
        return supported[opt_str]
    elif isinstance(supported, Collection) and opt_str in supported:
        return opt_str
    if default != 'no_default':
        return default
    set_to_check: 'set'
    if isinstance(supported, enum.EnumMeta):
        set_to_check = {item.value for item in supported}
    else:
        set_to_check = set(supported) if supported is not None else set()
    if not set_to_check:
        raise ValueError(f'No options available: {supported}.')
    edit_dists = {}
    opt_str = f'{opt_str}'
    for key in set_to_check:
        edit_dist = damerau_levenshtein_distance(f'{key}', opt_str)
        if edit_dist <= 3:
            edit_dists[key] = edit_dist
    supported_msg = f'Available options are {set_to_check}.\n' if print_all_options else ''
    if edit_dists:
        guess_at_spelling = min(edit_dists, key=edit_dists.get)
        raise ValueError(f"By '{opt_str}', did you mean '{guess_at_spelling}'?\n" + f"'{opt_str}' is not a valid value.\n" + supported_msg)
    raise ValueError(f"Unsupported option '{opt_str}', " + supported_msg)


def get_spatial_dims(boxes: 'torch.Tensor | np.ndarray | None'=None, points: 'torch.Tensor | np.ndarray | None'=None, corners: 'Sequence | None'=None, spatial_size: 'Sequence[int] | torch.Tensor | np.ndarray | None'=None) ->int:
    """
    Get spatial dimension for the giving setting and check the validity of them.
    Missing input is allowed. But at least one of the input value should be given.
    It raises ValueError if the dimensions of multiple inputs do not match with each other.

    Args:
        boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray
        points: point coordinates, [x, y] or [x, y, z], Nx2 or Nx3 torch tensor or ndarray
        corners: corners of boxes, 4-element or 6-element tuple, each element is a Nx1 torch tensor or ndarray
        spatial_size: The spatial size of the image where the boxes are attached.
                len(spatial_size) should be in [2, 3].

    Returns:
        ``int``: spatial_dims, number of spatial dimensions of the bounding boxes.

    Example:
        .. code-block:: python

            boxes = torch.ones(10,6)
            get_spatial_dims(boxes, spatial_size=[100,200,200]) # will return 3
            get_spatial_dims(boxes, spatial_size=[100,200]) # will raise ValueError
            get_spatial_dims(boxes) # will return 3
    """
    spatial_dims_set = set()
    if boxes is not None:
        if len(boxes.shape) != 2:
            if boxes.shape[0] == 0:
                raise ValueError(f'Currently we support only boxes with shape [N,4] or [N,6], got boxes with shape {boxes.shape}. Please reshape it with boxes = torch.reshape(boxes, [0, 4]) or torch.reshape(boxes, [0, 6]).')
            else:
                raise ValueError(f'Currently we support only boxes with shape [N,4] or [N,6], got boxes with shape {boxes.shape}.')
        if int(boxes.shape[1] / 2) not in SUPPORTED_SPATIAL_DIMS:
            raise ValueError(f'Currently we support only boxes with shape [N,4] or [N,6], got boxes with shape {boxes.shape}.')
        spatial_dims_set.add(int(boxes.shape[1] / 2))
    if points is not None:
        if len(points.shape) != 2:
            if points.shape[0] == 0:
                raise ValueError(f'Currently we support only points with shape [N,2] or [N,3], got points with shape {points.shape}. Please reshape it with points = torch.reshape(points, [0, 2]) or torch.reshape(points, [0, 3]).')
            else:
                raise ValueError(f'Currently we support only points with shape [N,2] or [N,3], got points with shape {points.shape}.')
        if int(points.shape[1]) not in SUPPORTED_SPATIAL_DIMS:
            raise ValueError(f'Currently we support only points with shape [N,2] or [N,3], got points with shape {points.shape}.')
        spatial_dims_set.add(int(points.shape[1]))
    if corners is not None:
        if len(corners) // 2 not in SUPPORTED_SPATIAL_DIMS:
            raise ValueError(f'Currently we support only boxes with shape [N,4] or [N,6], got box corner tuple with length {len(corners)}.')
        spatial_dims_set.add(len(corners) // 2)
    if spatial_size is not None:
        if len(spatial_size) not in SUPPORTED_SPATIAL_DIMS:
            raise ValueError(f'Currently we support only boxes on 2-D and 3-D images, got image spatial_size {spatial_size}.')
        spatial_dims_set.add(len(spatial_size))
    spatial_dims_list = list(spatial_dims_set)
    if len(spatial_dims_list) == 0:
        raise ValueError('At least one of the inputs needs to be non-empty.')
    if len(spatial_dims_list) == 1:
        spatial_dims = int(spatial_dims_list[0])
        spatial_dims = look_up_option(spatial_dims, supported=[2, 3])
        return int(spatial_dims)
    raise ValueError('The dimensions of multiple inputs should match with each other.')


def is_valid_box_values(boxes: 'NdarrayOrTensor') ->bool:
    """
    This function checks whether the box size is non-negative.

    Args:
        boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``

    Returns:
        whether ``boxes`` is valid
    """
    spatial_dims = get_spatial_dims(boxes=boxes)
    for axis in range(0, spatial_dims):
        if (boxes[:, spatial_dims + axis] < boxes[:, axis]).sum() > 0:
            return False
    return True


def box_area(boxes: 'NdarrayOrTensor') ->NdarrayOrTensor:
    """
    This function computes the area (2D) or volume (3D) of each box.
    Half precision is not recommended for this function as it may cause overflow, especially for 3D images.

    Args:
        boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``

    Returns:
        area (2D) or volume (3D) of boxes, with size of (N,).

    Example:
        .. code-block:: python

            boxes = torch.ones(10,6)
            # we do computation with torch.float32 to avoid overflow
            compute_dtype = torch.float32
            area = box_area(boxes=boxes.to(dtype=compute_dtype))  # torch.float32, size of (10,)
    """
    if not is_valid_box_values(boxes):
        raise ValueError('Given boxes has invalid values. The box size must be non-negative.')
    spatial_dims = get_spatial_dims(boxes=boxes)
    area = boxes[:, spatial_dims] - boxes[:, 0] + TO_REMOVE
    for axis in range(1, spatial_dims):
        area = area * (boxes[:, axis + spatial_dims] - boxes[:, axis] + TO_REMOVE)
    area_t, *_ = convert_data_type(area, torch.Tensor)
    if area_t.isnan().any() or area_t.isinf().any():
        if area_t.dtype is torch.float16:
            raise ValueError('Box area is NaN or Inf. boxes is float16. Please change to float32 and test it again.')
        else:
            raise ValueError('Box area is NaN or Inf.')
    return area


def _box_inter_union(boxes1_t: 'torch.Tensor', boxes2_t: 'torch.Tensor', compute_dtype: 'torch.dtype'=torch.float32) ->tuple[torch.Tensor, torch.Tensor]:
    """
    This internal function computes the intersection and union area of two set of boxes.

    Args:
        boxes1: bounding boxes, Nx4 or Nx6 torch tensor. The box mode is assumed to be ``StandardMode``
        boxes2: bounding boxes, Mx4 or Mx6 torch tensor. The box mode is assumed to be ``StandardMode``
        compute_dtype: default torch.float32, dtype with which the results will be computed

    Returns:
        inter, with size of (N,M) and dtype of ``compute_dtype``.
        union, with size of (N,M) and dtype of ``compute_dtype``.

    """
    spatial_dims = get_spatial_dims(boxes=boxes1_t)
    area1 = box_area(boxes=boxes1_t)
    area2 = box_area(boxes=boxes2_t)
    lt = torch.max(boxes1_t[:, None, :spatial_dims], boxes2_t[:, :spatial_dims])
    rb = torch.min(boxes1_t[:, None, spatial_dims:], boxes2_t[:, spatial_dims:])
    wh = (rb - lt + TO_REMOVE).clamp(min=0)
    inter = torch.prod(wh, dim=-1, keepdim=False)
    union = area1[:, None] + area2 - inter
    return inter, union


def box_iou(boxes1: 'NdarrayOrTensor', boxes2: 'NdarrayOrTensor') ->NdarrayOrTensor:
    """
    Compute the intersection over union (IoU) of two set of boxes.

    Args:
        boxes1: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
        boxes2: bounding boxes, Mx4 or Mx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``

    Returns:
        IoU, with size of (N,M) and same data type as ``boxes1``

    """
    if not isinstance(boxes1, type(boxes2)):
        warnings.warn(f'boxes1 is {type(boxes1)}, while boxes2 is {type(boxes2)}. The result will be {type(boxes1)}.')
    boxes1_t, *_ = convert_data_type(boxes1, torch.Tensor)
    boxes2_t, *_ = convert_data_type(boxes2, torch.Tensor)
    box_dtype = boxes1_t.dtype
    inter, union = _box_inter_union(boxes1_t, boxes2_t, compute_dtype=COMPUTE_DTYPE)
    iou_t = inter / (union + torch.finfo(COMPUTE_DTYPE).eps)
    iou_t = iou_t
    if torch.isnan(iou_t).any() or torch.isinf(iou_t).any():
        raise ValueError('Box IoU is NaN or Inf.')
    iou, *_ = convert_to_dst_type(src=iou_t, dst=boxes1)
    return iou


class Matcher(ABC):
    """
    Base class of Matcher, which matches boxes and anchors to each other

    Args:
        similarity_fn: function for similarity computation between
            boxes and anchors
    """
    BELOW_LOW_THRESHOLD: 'int' = -1
    BETWEEN_THRESHOLDS: 'int' = -2

    def __init__(self, similarity_fn: 'Callable[[Tensor, Tensor], Tensor]'=box_iou):
        self.similarity_fn = similarity_fn

    def __call__(self, boxes: 'torch.Tensor', anchors: 'torch.Tensor', num_anchors_per_level: 'Sequence[int]', num_anchors_per_loc: 'int') ->tuple[torch.Tensor, torch.Tensor]:
        """
        Compute matches for a single image

        Args:
            boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
            anchors: anchors to match Mx4 or Mx6, also assumed to be ``StandardMode``.
            num_anchors_per_level: number of anchors per feature pyramid level
            num_anchors_per_loc: number of anchors per position

        Returns:
            - matrix which contains the similarity from each boxes to each anchor [N, M]
            - vector which contains the matched box index for all
                anchors (if background `BELOW_LOW_THRESHOLD` is used
                and if it should be ignored `BETWEEN_THRESHOLDS` is used) [M]

        Note:
            ``StandardMode`` = :class:`~monai.data.box_utils.CornerCornerModeTypeA`,
            also represented as "xyxy" ([xmin, ymin, xmax, ymax]) for 2D
            and "xyzxyz" ([xmin, ymin, zmin, xmax, ymax, zmax]) for 3D.
        """
        if boxes.numel() == 0:
            num_anchors = anchors.shape[0]
            match_quality_matrix = torch.tensor([])
            matches = torch.empty(num_anchors, dtype=torch.int64).fill_(self.BELOW_LOW_THRESHOLD)
            return match_quality_matrix, matches
        return self.compute_matches(boxes=boxes, anchors=anchors, num_anchors_per_level=num_anchors_per_level, num_anchors_per_loc=num_anchors_per_loc)

    @abstractmethod
    def compute_matches(self, boxes: 'torch.Tensor', anchors: 'torch.Tensor', num_anchors_per_level: 'Sequence[int]', num_anchors_per_loc: 'int') ->tuple[torch.Tensor, torch.Tensor]:
        """
        Compute matches

        Args:
            boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
            anchors: anchors to match Mx4 or Mx6, also assumed to be ``StandardMode``.
            num_anchors_per_level: number of anchors per feature pyramid level
            num_anchors_per_loc: number of anchors per position

        Returns:
            - matrix which contains the similarity from each boxes to each anchor [N, M]
            - vector which contains the matched box index for all
              anchors (if background `BELOW_LOW_THRESHOLD` is used
              and if it should be ignored `BETWEEN_THRESHOLDS` is used) [M]
        """
        raise NotImplementedError


class BoxMode(ABC):
    """
    An abstract class of a ``BoxMode``.

    A ``BoxMode`` is callable that converts box mode of ``boxes``, which are Nx4 (2D) or Nx6 (3D) torch tensor or ndarray.
    ``BoxMode`` has several subclasses that represents different box modes, including

    - :class:`~monai.data.box_utils.CornerCornerModeTypeA`:
      represents [xmin, ymin, xmax, ymax] for 2D and [xmin, ymin, zmin, xmax, ymax, zmax] for 3D
    - :class:`~monai.data.box_utils.CornerCornerModeTypeB`:
      represents [xmin, xmax, ymin, ymax] for 2D and [xmin, xmax, ymin, ymax, zmin, zmax] for 3D
    - :class:`~monai.data.box_utils.CornerCornerModeTypeC`:
      represents [xmin, ymin, xmax, ymax] for 2D and [xmin, ymin, xmax, ymax, zmin, zmax] for 3D
    - :class:`~monai.data.box_utils.CornerSizeMode`:
      represents [xmin, ymin, xsize, ysize] for 2D and [xmin, ymin, zmin, xsize, ysize, zsize] for 3D
    - :class:`~monai.data.box_utils.CenterSizeMode`:
      represents [xcenter, ycenter, xsize, ysize] for 2D and [xcenter, ycenter, zcenter, xsize, ysize, zsize] for 3D

    We currently define ``StandardMode`` = :class:`~monai.data.box_utils.CornerCornerModeTypeA`,
    and monai detection pipelines mainly assume ``boxes`` are in ``StandardMode``.

    The implementation should be aware of:

    - remember to define class variable ``name``,
      a dictionary that maps ``spatial_dims`` to :class:`~monai.utils.enums.BoxModeName`.
    - :func:`~monai.data.box_utils.BoxMode.boxes_to_corners` and :func:`~monai.data.box_utils.BoxMode.corners_to_boxes`
      should not modify inputs in place.
    """
    name: 'dict[int, BoxModeName]' = {}

    @classmethod
    def get_name(cls, spatial_dims: 'int') ->str:
        """
        Get the mode name for the given spatial dimension using class variable ``name``.

        Args:
            spatial_dims: number of spatial dimensions of the bounding boxes.

        Returns:
            ``str``: mode string name
        """
        return cls.name[spatial_dims].value

    @abstractmethod
    def boxes_to_corners(self, boxes: 'torch.Tensor') ->tuple:
        """
        Convert the bounding boxes of the current mode to corners.

        Args:
            boxes: bounding boxes, Nx4 or Nx6 torch tensor

        Returns:
            ``tuple``: corners of boxes, 4-element or 6-element tuple, each element is a Nx1 torch tensor.
            It represents (xmin, ymin, xmax, ymax) or (xmin, ymin, zmin, xmax, ymax, zmax)

        Example:
            .. code-block:: python

                boxes = torch.ones(10,6)
                boxmode = BoxMode()
                boxmode.boxes_to_corners(boxes) # will return a 6-element tuple, each element is a 10x1 tensor
        """
        raise NotImplementedError(f'Subclass {self.__class__.__name__} must implement this method.')

    @abstractmethod
    def corners_to_boxes(self, corners: 'Sequence') ->torch.Tensor:
        """
        Convert the given box corners to the bounding boxes of the current mode.

        Args:
            corners: corners of boxes, 4-element or 6-element tuple, each element is a Nx1 torch tensor.
                It represents (xmin, ymin, xmax, ymax) or (xmin, ymin, zmin, xmax, ymax, zmax)

        Returns:
            ``Tensor``: bounding boxes, Nx4 or Nx6 torch tensor

        Example:
            .. code-block:: python

                corners = (torch.ones(10,1), torch.ones(10,1), torch.ones(10,1), torch.ones(10,1))
                boxmode = BoxMode()
                boxmode.corners_to_boxes(corners) # will return a 10x4 tensor
        """
        raise NotImplementedError(f'Subclass {self.__class__.__name__} must implement this method.')


class StrEnum(str, Enum):
    """
    Enum subclass that converts its value to a string.

    .. code-block:: python

        from monai.utils import StrEnum

        class Example(StrEnum):
            MODE_A = "A"
            MODE_B = "B"

        assert (list(Example) == ["A", "B"])
        assert Example.MODE_A == "A"
        assert str(Example.MODE_A) == "A"
        assert monai.utils.look_up_option("A", Example) == "A"
    """

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class BoxModeName(StrEnum):
    """
    Box mode names.
    """
    XYXY = 'xyxy'
    XYZXYZ = 'xyzxyz'
    XXYY = 'xxyy'
    XXYYZZ = 'xxyyzz'
    XYXYZZ = 'xyxyzz'
    XYWH = 'xywh'
    XYZWHD = 'xyzwhd'
    CCWH = 'ccwh'
    CCCWHD = 'cccwhd'


class CenterSizeMode(BoxMode):
    """
    A subclass of ``BoxMode``.

    Also represented as "ccwh" or "cccwhd", with format of
    [xmin, ymin, xsize, ysize] or [xmin, ymin, zmin, xsize, ysize, zsize].

    Example:
        .. code-block:: python

            CenterSizeMode.get_name(spatial_dims=2) # will return "ccwh"
            CenterSizeMode.get_name(spatial_dims=3) # will return "cccwhd"
    """
    name = {(2): BoxModeName.CCWH, (3): BoxModeName.CCCWHD}

    def boxes_to_corners(self, boxes: 'torch.Tensor') ->tuple:
        corners: 'tuple'
        box_dtype = boxes.dtype
        spatial_dims = get_spatial_dims(boxes=boxes)
        if spatial_dims == 3:
            xc, yc, zc, w, h, d = boxes.split(1, dim=-1)
            xmin = xc - ((w - TO_REMOVE) / 2.0).to(dtype=COMPUTE_DTYPE).clamp(min=0)
            xmax = xc + ((w - TO_REMOVE) / 2.0).to(dtype=COMPUTE_DTYPE).clamp(min=0)
            ymin = yc - ((h - TO_REMOVE) / 2.0).to(dtype=COMPUTE_DTYPE).clamp(min=0)
            ymax = yc + ((h - TO_REMOVE) / 2.0).to(dtype=COMPUTE_DTYPE).clamp(min=0)
            zmin = zc - ((d - TO_REMOVE) / 2.0).to(dtype=COMPUTE_DTYPE).clamp(min=0)
            zmax = zc + ((d - TO_REMOVE) / 2.0).to(dtype=COMPUTE_DTYPE).clamp(min=0)
            corners = xmin, ymin, zmin, xmax, ymax, zmax
        elif spatial_dims == 2:
            xc, yc, w, h = boxes.split(1, dim=-1)
            xmin = xc - ((w - TO_REMOVE) / 2.0).to(dtype=COMPUTE_DTYPE).clamp(min=0)
            xmax = xc + ((w - TO_REMOVE) / 2.0).to(dtype=COMPUTE_DTYPE).clamp(min=0)
            ymin = yc - ((h - TO_REMOVE) / 2.0).to(dtype=COMPUTE_DTYPE).clamp(min=0)
            ymax = yc + ((h - TO_REMOVE) / 2.0).to(dtype=COMPUTE_DTYPE).clamp(min=0)
            corners = xmin, ymin, xmax, ymax
        return corners

    def corners_to_boxes(self, corners: 'Sequence') ->torch.Tensor:
        boxes: 'torch.Tensor'
        spatial_dims = get_spatial_dims(corners=corners)
        if spatial_dims == 3:
            xmin, ymin, zmin, xmax, ymax, zmax = corners[0], corners[1], corners[2], corners[3], corners[4], corners[5]
            boxes = torch.cat(((xmin + xmax + TO_REMOVE) / 2.0, (ymin + ymax + TO_REMOVE) / 2.0, (zmin + zmax + TO_REMOVE) / 2.0, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE, zmax - zmin + TO_REMOVE), dim=-1)
        elif spatial_dims == 2:
            xmin, ymin, xmax, ymax = corners[0], corners[1], corners[2], corners[3]
            boxes = torch.cat(((xmin + xmax + TO_REMOVE) / 2.0, (ymin + ymax + TO_REMOVE) / 2.0, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1)
        return boxes


class CornerCornerModeTypeA(BoxMode):
    """
    A subclass of ``BoxMode``.

    Also represented as "xyxy" or "xyzxyz", with format of
    [xmin, ymin, xmax, ymax] or [xmin, ymin, zmin, xmax, ymax, zmax].

    Example:
        .. code-block:: python

            CornerCornerModeTypeA.get_name(spatial_dims=2) # will return "xyxy"
            CornerCornerModeTypeA.get_name(spatial_dims=3) # will return "xyzxyz"
    """
    name = {(2): BoxModeName.XYXY, (3): BoxModeName.XYZXYZ}

    def boxes_to_corners(self, boxes: 'torch.Tensor') ->tuple:
        corners: 'tuple'
        corners = boxes.split(1, dim=-1)
        return corners

    def corners_to_boxes(self, corners: 'Sequence') ->torch.Tensor:
        boxes: 'torch.Tensor'
        boxes = torch.cat(tuple(corners), dim=-1)
        return boxes


StandardMode = CornerCornerModeTypeA


class CornerCornerModeTypeB(BoxMode):
    """
    A subclass of ``BoxMode``.

    Also represented as "xxyy" or "xxyyzz", with format of
    [xmin, xmax, ymin, ymax] or [xmin, xmax, ymin, ymax, zmin, zmax].

    Example:
        .. code-block:: python

            CornerCornerModeTypeB.get_name(spatial_dims=2) # will return "xxyy"
            CornerCornerModeTypeB.get_name(spatial_dims=3) # will return "xxyyzz"
    """
    name = {(2): BoxModeName.XXYY, (3): BoxModeName.XXYYZZ}

    def boxes_to_corners(self, boxes: 'torch.Tensor') ->tuple:
        corners: 'tuple'
        spatial_dims = get_spatial_dims(boxes=boxes)
        if spatial_dims == 3:
            xmin, xmax, ymin, ymax, zmin, zmax = boxes.split(1, dim=-1)
            corners = xmin, ymin, zmin, xmax, ymax, zmax
        elif spatial_dims == 2:
            xmin, xmax, ymin, ymax = boxes.split(1, dim=-1)
            corners = xmin, ymin, xmax, ymax
        return corners

    def corners_to_boxes(self, corners: 'Sequence') ->torch.Tensor:
        boxes: 'torch.Tensor'
        spatial_dims = get_spatial_dims(corners=corners)
        if spatial_dims == 3:
            boxes = torch.cat((corners[0], corners[3], corners[1], corners[4], corners[2], corners[5]), dim=-1)
        elif spatial_dims == 2:
            boxes = torch.cat((corners[0], corners[2], corners[1], corners[3]), dim=-1)
        return boxes


class CornerCornerModeTypeC(BoxMode):
    """
    A subclass of ``BoxMode``.

    Also represented as "xyxy" or "xyxyzz", with format of
    [xmin, ymin, xmax, ymax] or [xmin, ymin, xmax, ymax, zmin, zmax].

    Example:
        .. code-block:: python

            CornerCornerModeTypeC.get_name(spatial_dims=2) # will return "xyxy"
            CornerCornerModeTypeC.get_name(spatial_dims=3) # will return "xyxyzz"
    """
    name = {(2): BoxModeName.XYXY, (3): BoxModeName.XYXYZZ}

    def boxes_to_corners(self, boxes: 'torch.Tensor') ->tuple:
        corners: 'tuple'
        spatial_dims = get_spatial_dims(boxes=boxes)
        if spatial_dims == 3:
            xmin, ymin, xmax, ymax, zmin, zmax = boxes.split(1, dim=-1)
            corners = xmin, ymin, zmin, xmax, ymax, zmax
        elif spatial_dims == 2:
            corners = boxes.split(1, dim=-1)
        return corners

    def corners_to_boxes(self, corners: 'Sequence') ->torch.Tensor:
        boxes: 'torch.Tensor'
        spatial_dims = get_spatial_dims(corners=corners)
        if spatial_dims == 3:
            boxes = torch.cat((corners[0], corners[1], corners[3], corners[4], corners[2], corners[5]), dim=-1)
        elif spatial_dims == 2:
            boxes = torch.cat(tuple(corners), dim=-1)
        return boxes


class CornerSizeMode(BoxMode):
    """
    A subclass of ``BoxMode``.

    Also represented as "xywh" or "xyzwhd", with format of
    [xmin, ymin, xsize, ysize] or [xmin, ymin, zmin, xsize, ysize, zsize].

    Example:
        .. code-block:: python

            CornerSizeMode.get_name(spatial_dims=2) # will return "xywh"
            CornerSizeMode.get_name(spatial_dims=3) # will return "xyzwhd"
    """
    name = {(2): BoxModeName.XYWH, (3): BoxModeName.XYZWHD}

    def boxes_to_corners(self, boxes: 'torch.Tensor') ->tuple:
        corners: 'tuple'
        box_dtype = boxes.dtype
        spatial_dims = get_spatial_dims(boxes=boxes)
        if spatial_dims == 3:
            xmin, ymin, zmin, w, h, d = boxes.split(1, dim=-1)
            xmax = xmin + (w - TO_REMOVE).to(dtype=COMPUTE_DTYPE).clamp(min=0)
            ymax = ymin + (h - TO_REMOVE).to(dtype=COMPUTE_DTYPE).clamp(min=0)
            zmax = zmin + (d - TO_REMOVE).to(dtype=COMPUTE_DTYPE).clamp(min=0)
            corners = xmin, ymin, zmin, xmax, ymax, zmax
        elif spatial_dims == 2:
            xmin, ymin, w, h = boxes.split(1, dim=-1)
            xmax = xmin + (w - TO_REMOVE).to(dtype=COMPUTE_DTYPE).clamp(min=0)
            ymax = ymin + (h - TO_REMOVE).to(dtype=COMPUTE_DTYPE).clamp(min=0)
            corners = xmin, ymin, xmax, ymax
        return corners

    def corners_to_boxes(self, corners: 'Sequence') ->torch.Tensor:
        boxes: 'torch.Tensor'
        spatial_dims = get_spatial_dims(corners=corners)
        if spatial_dims == 3:
            xmin, ymin, zmin, xmax, ymax, zmax = corners[0], corners[1], corners[2], corners[3], corners[4], corners[5]
            boxes = torch.cat((xmin, ymin, zmin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE, zmax - zmin + TO_REMOVE), dim=-1)
        elif spatial_dims == 2:
            xmin, ymin, xmax, ymax = corners[0], corners[1], corners[2], corners[3]
            boxes = torch.cat((xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1)
        return boxes


SUPPORTED_MODES = [CornerCornerModeTypeA, CornerCornerModeTypeB, CornerCornerModeTypeC, CornerSizeMode, CenterSizeMode]


def get_boxmode(mode: 'str | BoxMode | type[BoxMode] | None'=None, *args, **kwargs) ->BoxMode:
    """
    This function that return a :class:`~monai.data.box_utils.BoxMode` object giving a representation of box mode

    Args:
        mode: a representation of box mode. If it is not given, this func will assume it is ``StandardMode()``.

    Note:
        ``StandardMode`` = :class:`~monai.data.box_utils.CornerCornerModeTypeA`,
        also represented as "xyxy" for 2D and "xyzxyz" for 3D.

        mode can be:
            #. str: choose from :class:`~monai.utils.enums.BoxModeName`, for example,
                - "xyxy": boxes has format [xmin, ymin, xmax, ymax]
                - "xyzxyz": boxes has format [xmin, ymin, zmin, xmax, ymax, zmax]
                - "xxyy": boxes has format [xmin, xmax, ymin, ymax]
                - "xxyyzz": boxes has format [xmin, xmax, ymin, ymax, zmin, zmax]
                - "xyxyzz": boxes has format [xmin, ymin, xmax, ymax, zmin, zmax]
                - "xywh": boxes has format [xmin, ymin, xsize, ysize]
                - "xyzwhd": boxes has format [xmin, ymin, zmin, xsize, ysize, zsize]
                - "ccwh": boxes has format [xcenter, ycenter, xsize, ysize]
                - "cccwhd": boxes has format [xcenter, ycenter, zcenter, xsize, ysize, zsize]
            #. BoxMode class: choose from the subclasses of :class:`~monai.data.box_utils.BoxMode`, for example,
                - CornerCornerModeTypeA: equivalent to "xyxy" or "xyzxyz"
                - CornerCornerModeTypeB: equivalent to "xxyy" or "xxyyzz"
                - CornerCornerModeTypeC: equivalent to "xyxy" or "xyxyzz"
                - CornerSizeMode: equivalent to "xywh" or "xyzwhd"
                - CenterSizeMode: equivalent to "ccwh" or "cccwhd"
            #. BoxMode object: choose from the subclasses of :class:`~monai.data.box_utils.BoxMode`, for example,
                - CornerCornerModeTypeA(): equivalent to "xyxy" or "xyzxyz"
                - CornerCornerModeTypeB(): equivalent to "xxyy" or "xxyyzz"
                - CornerCornerModeTypeC(): equivalent to "xyxy" or "xyxyzz"
                - CornerSizeMode(): equivalent to "xywh" or "xyzwhd"
                - CenterSizeMode(): equivalent to "ccwh" or "cccwhd"
            #. None: will assume mode is ``StandardMode()``

    Returns:
        BoxMode object

    Example:
        .. code-block:: python

            mode = "xyzxyz"
            get_boxmode(mode) # will return CornerCornerModeTypeA()
    """
    if isinstance(mode, BoxMode):
        return mode
    if inspect.isclass(mode) and issubclass(mode, BoxMode):
        return mode(*args, **kwargs)
    if isinstance(mode, str):
        for m in SUPPORTED_MODES:
            for n in SUPPORTED_SPATIAL_DIMS:
                if inspect.isclass(m) and issubclass(m, BoxMode) and m.get_name(n) == mode:
                    return m(*args, **kwargs)
    if mode is not None:
        raise ValueError(f'Unsupported box mode: {mode}.')
    return StandardMode(*args, **kwargs)


def convert_box_mode(boxes: 'NdarrayOrTensor', src_mode: 'str | BoxMode | type[BoxMode] | None'=None, dst_mode: 'str | BoxMode | type[BoxMode] | None'=None) ->NdarrayOrTensor:
    """
    This function converts the boxes in src_mode to the dst_mode.

    Args:
        boxes: source bounding boxes, Nx4 or Nx6 torch tensor or ndarray.
        src_mode: source box mode. If it is not given, this func will assume it is ``StandardMode()``.
            It follows the same format with ``mode`` in :func:`~monai.data.box_utils.get_boxmode`.
        dst_mode: target box mode. If it is not given, this func will assume it is ``StandardMode()``.
            It follows the same format with ``mode`` in :func:`~monai.data.box_utils.get_boxmode`.

    Returns:
        bounding boxes with target mode, with same data type as ``boxes``, does not share memory with ``boxes``

    Example:
        .. code-block:: python

            boxes = torch.ones(10,4)
            # The following three lines are equivalent
            # They convert boxes with format [xmin, ymin, xmax, ymax] to [xcenter, ycenter, xsize, ysize].
            convert_box_mode(boxes=boxes, src_mode="xyxy", dst_mode="ccwh")
            convert_box_mode(boxes=boxes, src_mode="xyxy", dst_mode=monai.data.box_utils.CenterSizeMode)
            convert_box_mode(boxes=boxes, src_mode="xyxy", dst_mode=monai.data.box_utils.CenterSizeMode())
    """
    if boxes.shape[0] == 0:
        return boxes
    src_boxmode = get_boxmode(src_mode)
    dst_boxmode = get_boxmode(dst_mode)
    if isinstance(src_boxmode, type(dst_boxmode)):
        return deepcopy(boxes)
    boxes_t, *_ = convert_data_type(boxes, torch.Tensor)
    corners = src_boxmode.boxes_to_corners(boxes_t)
    spatial_dims = get_spatial_dims(boxes=boxes_t)
    for axis in range(0, spatial_dims):
        if (corners[spatial_dims + axis] < corners[axis]).sum() > 0:
            warnings.warn('Given boxes has invalid values. The box size must be non-negative.')
    boxes_t_dst = dst_boxmode.corners_to_boxes(corners)
    boxes_dst, *_ = convert_to_dst_type(src=boxes_t_dst, dst=boxes)
    return boxes_dst


def box_centers(boxes: 'NdarrayOrTensor') ->NdarrayOrTensor:
    """
    Compute center points of boxes

    Args:
        boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``

    Returns:
        center points with size of (N, spatial_dims)

    """
    spatial_dims = get_spatial_dims(boxes=boxes)
    return convert_box_mode(boxes=boxes, src_mode=StandardMode, dst_mode=CenterSizeMode)[:, :spatial_dims]


def boxes_center_distance(boxes1: 'NdarrayOrTensor', boxes2: 'NdarrayOrTensor', euclidean: 'bool'=True) ->tuple[NdarrayOrTensor, NdarrayOrTensor, NdarrayOrTensor]:
    """
    Distance of center points between two sets of boxes

    Args:
        boxes1: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
        boxes2: bounding boxes, Mx4 or Mx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
        euclidean: computed the euclidean distance otherwise it uses the l1 distance

    Returns:
        - The pairwise distances for every element in boxes1 and boxes2,
          with size of (N,M) and same data type as ``boxes1``.
        - Center points of boxes1, with size of (N,spatial_dims) and same data type as ``boxes1``.
        - Center points of boxes2, with size of (M,spatial_dims) and same data type as ``boxes1``.

    Reference:
        https://github.com/MIC-DKFZ/nnDetection/blob/main/nndet/core/boxes/ops.py

    """
    if not isinstance(boxes1, type(boxes2)):
        warnings.warn(f'boxes1 is {type(boxes1)}, while boxes2 is {type(boxes2)}. The result will be {type(boxes1)}.')
    boxes1_t, *_ = convert_data_type(boxes1, torch.Tensor)
    boxes2_t, *_ = convert_data_type(boxes2, torch.Tensor)
    center1 = box_centers(boxes1_t)
    center2 = box_centers(boxes2_t)
    if euclidean:
        dists = (center1[:, None] - center2[None]).pow(2).sum(-1).sqrt()
    else:
        dists = (center1[:, None] - center2[None]).sum(-1)
    (dists, center1, center2), *_ = convert_to_dst_type(src=(dists, center1, center2), dst=boxes1)
    return dists, center1, center2


def centers_in_boxes(centers: 'NdarrayOrTensor', boxes: 'NdarrayOrTensor', eps: 'float'=0.01) ->NdarrayOrTensor:
    """
    Checks which center points are within boxes

    Args:
        boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``.
        centers: center points, Nx2 or Nx3 torch tensor or ndarray.
        eps: minimum distance to border of boxes.

    Returns:
        boolean array indicating which center points are within the boxes, sized (N,).

    Reference:
        https://github.com/MIC-DKFZ/nnDetection/blob/main/nndet/core/boxes/ops.py

    """
    spatial_dims = get_spatial_dims(boxes=boxes)
    center_to_border = [(centers[:, axis] - boxes[:, axis]) for axis in range(spatial_dims)] + [(boxes[:, axis + spatial_dims] - centers[:, axis]) for axis in range(spatial_dims)]
    if isinstance(boxes, np.ndarray):
        min_center_to_border: 'np.ndarray' = np.stack(center_to_border, axis=1).min(axis=1)
        return min_center_to_border > eps
    return torch.stack(center_to_border, dim=1).min(dim=1)[0] > eps


class ATSSMatcher(Matcher):

    def __init__(self, num_candidates: 'int'=4, similarity_fn: 'Callable[[Tensor, Tensor], Tensor]'=box_iou, center_in_gt: 'bool'=True, debug: 'bool'=False):
        """
        Compute matching based on ATSS https://arxiv.org/abs/1912.02424
        `Bridging the Gap Between Anchor-based and Anchor-free Detection
        via Adaptive Training Sample Selection`

        Args:
            num_candidates: number of positions to select candidates from.
                Smaller value will result in a higher matcher threshold and less matched candidates.
            similarity_fn: function for similarity computation between boxes and anchors
            center_in_gt: If False (default), matched anchor center points do not need
                to lie withing the ground truth box. Recommend False for small objects.
                If True, will result in a strict matcher and less matched candidates.
            debug: if True, will print the matcher threshold in order to
                tune ``num_candidates`` and ``center_in_gt``.
        """
        super().__init__(similarity_fn=similarity_fn)
        self.num_candidates = num_candidates
        self.min_dist = 0.01
        self.center_in_gt = center_in_gt
        self.debug = debug
        logging.info(f'Running ATSS Matching with num_candidates={self.num_candidates} and center_in_gt {self.center_in_gt}.')

    def compute_matches(self, boxes: 'torch.Tensor', anchors: 'torch.Tensor', num_anchors_per_level: 'Sequence[int]', num_anchors_per_loc: 'int') ->tuple[torch.Tensor, torch.Tensor]:
        """
        Compute matches according to ATTS for a single image
        Adapted from
        (https://github.com/sfzhang15/ATSS/blob/79dfb28bd1/atss_core/modeling/rpn/atss/loss.py#L180-L184)

        Args:
            boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
            anchors: anchors to match Mx4 or Mx6, also assumed to be ``StandardMode``.
            num_anchors_per_level: number of anchors per feature pyramid level
            num_anchors_per_loc: number of anchors per position

        Returns:
            - matrix which contains the similarity from each boxes to each anchor [N, M]
            - vector which contains the matched box index for all
              anchors (if background `BELOW_LOW_THRESHOLD` is used
              and if it should be ignored `BETWEEN_THRESHOLDS` is used) [M]

        Note:
            ``StandardMode`` = :class:`~monai.data.box_utils.CornerCornerModeTypeA`,
            also represented as "xyxy" ([xmin, ymin, xmax, ymax]) for 2D
            and "xyzxyz" ([xmin, ymin, zmin, xmax, ymax, zmax]) for 3D.
        """
        num_gt = boxes.shape[0]
        num_anchors = anchors.shape[0]
        distances_, _, anchors_center = boxes_center_distance(boxes, anchors)
        distances = convert_to_tensor(distances_)
        candidate_idx_list = []
        start_idx = 0
        for _, apl in enumerate(num_anchors_per_level):
            end_idx = start_idx + apl * num_anchors_per_loc
            topk = min(self.num_candidates * num_anchors_per_loc, apl)
            _, idx = distances[:, start_idx:end_idx].topk(topk, dim=1, largest=False)
            candidate_idx_list.append(idx + start_idx)
            start_idx = end_idx
        candidate_idx = torch.cat(candidate_idx_list, dim=1)
        match_quality_matrix = self.similarity_fn(boxes, anchors)
        candidate_ious = match_quality_matrix.gather(1, candidate_idx)
        if candidate_idx.shape[1] <= 1:
            matches = -1 * torch.ones((num_anchors,), dtype=torch.long, device=boxes.device)
            matches[candidate_idx] = 0
            return match_quality_matrix, matches
        iou_mean_per_gt = candidate_ious.mean(dim=1)
        iou_std_per_gt = candidate_ious.std(dim=1)
        iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
        is_pos = candidate_ious >= iou_thresh_per_gt[:, None]
        if self.debug:
            None
        if self.center_in_gt:
            boxes_idx = torch.arange(num_gt, device=boxes.device, dtype=torch.long)[:, None].expand_as(candidate_idx).contiguous()
            is_in_gt_ = centers_in_boxes(anchors_center[candidate_idx.view(-1)], boxes[boxes_idx.view(-1)], eps=self.min_dist)
            is_in_gt = convert_to_tensor(is_in_gt_)
            is_pos = is_pos & is_in_gt.view_as(is_pos)
        for ng in range(num_gt):
            candidate_idx[ng, :] += ng * num_anchors
        ious_inf = torch.full_like(match_quality_matrix, -INF).view(-1)
        index = candidate_idx.view(-1)[is_pos.view(-1)]
        ious_inf[index] = match_quality_matrix.view(-1)[index]
        ious_inf = ious_inf.view_as(match_quality_matrix)
        matched_vals, matches = ious_inf.max(dim=0)
        matches[matched_vals == -INF] = self.BELOW_LOW_THRESHOLD
        return match_quality_matrix, matches


BalancedPositiveNegativeSampler, _ = optional_import('torchvision.models.detection._utils', name='BalancedPositiveNegativeSampler')


class BlendMode(StrEnum):
    """
    See also: :py:class:`monai.data.utils.compute_importance_map`
    """
    CONSTANT = 'constant'
    GAUSSIAN = 'gaussian'


def encode_boxes(gt_boxes: 'Tensor', proposals: 'Tensor', weights: 'Tensor') ->Tensor:
    """
    Encode a set of proposals with respect to some reference ground truth (gt) boxes.

    Args:
        gt_boxes: gt boxes, Nx4 or Nx6 torch tensor. The box mode is assumed to be ``StandardMode``
        proposals: boxes to be encoded, Nx4 or Nx6 torch tensor. The box mode is assumed to be ``StandardMode``
        weights: the weights for ``(cx, cy, w, h) or (cx,cy,cz, w,h,d)``

    Return:
        encoded gt, target of box regression that is used to convert proposals into gt_boxes, Nx4 or Nx6 torch tensor.
    """
    if gt_boxes.shape[0] != proposals.shape[0]:
        raise ValueError('gt_boxes.shape[0] should be equal to proposals.shape[0].')
    spatial_dims = look_up_option(len(weights), [4, 6]) // 2
    if not is_valid_box_values(gt_boxes):
        raise ValueError('gt_boxes is not valid. Please check if it contains empty boxes.')
    if not is_valid_box_values(proposals):
        raise ValueError('proposals is not valid. Please check if it contains empty boxes.')
    ex_cccwhd: 'Tensor' = convert_box_mode(proposals, src_mode=StandardMode, dst_mode=CenterSizeMode)
    gt_cccwhd: 'Tensor' = convert_box_mode(gt_boxes, src_mode=StandardMode, dst_mode=CenterSizeMode)
    targets_dxyz = weights[:spatial_dims].unsqueeze(0) * (gt_cccwhd[:, :spatial_dims] - ex_cccwhd[:, :spatial_dims]) / ex_cccwhd[:, spatial_dims:]
    targets_dwhd = weights[spatial_dims:].unsqueeze(0) * torch.log(gt_cccwhd[:, spatial_dims:] / ex_cccwhd[:, spatial_dims:])
    targets = torch.cat((targets_dxyz, targets_dwhd), dim=1)
    if torch.isnan(targets).any() or torch.isinf(targets).any():
        raise ValueError('targets is NaN or Inf.')
    return targets


class BoxCoder:
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.

    Args:
        weights: 4-element tuple or 6-element tuple
        boxes_xform_clip: high threshold to prevent sending too large values into torch.exp()

    Example:
        .. code-block:: python

            box_coder = BoxCoder(weights=[1., 1., 1., 1., 1., 1.])
            gt_boxes = torch.tensor([[1,2,1,4,5,6],[1,3,2,7,8,9]])
            proposals = gt_boxes + torch.rand(gt_boxes.shape)
            rel_gt_boxes = box_coder.encode_single(gt_boxes, proposals)
            gt_back = box_coder.decode_single(rel_gt_boxes, proposals)
            # We expect gt_back to be equal to gt_boxes
    """

    def __init__(self, weights: 'Sequence[float]', boxes_xform_clip: 'float | None'=None) ->None:
        if boxes_xform_clip is None:
            boxes_xform_clip = math.log(1000.0 / 16)
        self.spatial_dims = look_up_option(len(weights), [4, 6]) // 2
        self.weights = weights
        self.boxes_xform_clip = boxes_xform_clip

    def encode(self, gt_boxes: 'Sequence[Tensor]', proposals: 'Sequence[Tensor]') ->tuple[Tensor]:
        """
        Encode a set of proposals with respect to some ground truth (gt) boxes.

        Args:
            gt_boxes: list of gt boxes, Nx4 or Nx6 torch tensor. The box mode is assumed to be ``StandardMode``
            proposals: list of boxes to be encoded, each element is Mx4 or Mx6 torch tensor.
                The box mode is assumed to be ``StandardMode``

        Return:
            A tuple of encoded gt, target of box regression that is used to
                convert proposals into gt_boxes, Nx4 or Nx6 torch tensor.
        """
        boxes_per_image = [len(b) for b in gt_boxes]
        concat_gt_boxes = torch.cat(tuple(gt_boxes), dim=0)
        concat_proposals = torch.cat(tuple(proposals), dim=0)
        concat_targets = self.encode_single(concat_gt_boxes, concat_proposals)
        targets: 'tuple[Tensor]' = concat_targets.split(boxes_per_image, 0)
        return targets

    def encode_single(self, gt_boxes: 'Tensor', proposals: 'Tensor') ->Tensor:
        """
        Encode proposals with respect to ground truth (gt) boxes.

        Args:
            gt_boxes: gt boxes, Nx4 or Nx6 torch tensor. The box mode is assumed to be ``StandardMode``
            proposals: boxes to be encoded, Nx4 or Nx6 torch tensor. The box mode is assumed to be ``StandardMode``

        Return:
            encoded gt, target of box regression that is used to convert proposals into gt_boxes, Nx4 or Nx6 torch tensor.
        """
        dtype = gt_boxes.dtype
        device = gt_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(gt_boxes, proposals, weights)
        return targets

    def decode(self, rel_codes: 'Tensor', reference_boxes: 'Sequence[Tensor]') ->Tensor:
        """
        From a set of original reference_boxes and encoded relative box offsets,

        Args:
            rel_codes: encoded boxes, Nx4 or Nx6 torch tensor.
            reference_boxes: a list of reference boxes, each element is Mx4 or Mx6 torch tensor.
                The box mode is assumed to be ``StandardMode``

        Return:
            decoded boxes, Nx1x4 or Nx1x6 torch tensor. The box mode will be ``StandardMode``
        """
        if not isinstance(reference_boxes, Sequence) or not isinstance(rel_codes, torch.Tensor):
            raise ValueError('Input arguments wrong type.')
        boxes_per_image = [b.size(0) for b in reference_boxes]
        concat_boxes = torch.cat(tuple(reference_boxes), dim=0)
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        if box_sum > 0:
            rel_codes = rel_codes.reshape(box_sum, -1)
        pred_boxes = self.decode_single(rel_codes, concat_boxes)
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 2 * self.spatial_dims)
        return pred_boxes

    def decode_single(self, rel_codes: 'Tensor', reference_boxes: 'Tensor') ->Tensor:
        """
        From a set of original boxes and encoded relative box offsets,

        Args:
            rel_codes: encoded boxes, Nx(4*num_box_reg) or Nx(6*num_box_reg) torch tensor.
            reference_boxes: reference boxes, Nx4 or Nx6 torch tensor. The box mode is assumed to be ``StandardMode``

        Return:
            decoded boxes, Nx(4*num_box_reg) or Nx(6*num_box_reg) torch tensor. The box mode will to be ``StandardMode``
        """
        reference_boxes = reference_boxes
        offset = reference_boxes.shape[-1]
        pred_boxes = []
        boxes_cccwhd = convert_box_mode(reference_boxes, src_mode=StandardMode, dst_mode=CenterSizeMode)
        for axis in range(self.spatial_dims):
            whd_axis = boxes_cccwhd[:, axis + self.spatial_dims]
            ctr_xyz_axis = boxes_cccwhd[:, axis]
            dxyz_axis = rel_codes[:, axis::offset] / self.weights[axis]
            dwhd_axis = rel_codes[:, self.spatial_dims + axis::offset] / self.weights[axis + self.spatial_dims]
            dwhd_axis = torch.clamp(dwhd_axis, max=self.boxes_xform_clip)
            pred_ctr_xyx_axis = dxyz_axis * whd_axis[:, None] + ctr_xyz_axis[:, None]
            pred_whd_axis = torch.exp(dwhd_axis) * whd_axis[:, None]
            pred_whd_axis = pred_whd_axis
            if torch.isnan(pred_whd_axis).any() or torch.isinf(pred_whd_axis).any():
                raise ValueError('pred_whd_axis is NaN or Inf.')
            c_to_c_whd_axis = torch.tensor(0.5, dtype=pred_ctr_xyx_axis.dtype, device=pred_whd_axis.device) * pred_whd_axis
            pred_boxes.append(pred_ctr_xyx_axis - c_to_c_whd_axis)
            pred_boxes.append(pred_ctr_xyx_axis + c_to_c_whd_axis)
        pred_boxes = pred_boxes[::2] + pred_boxes[1::2]
        pred_boxes_final = torch.stack(pred_boxes, dim=2).flatten(1)
        return pred_boxes_final


def non_max_suppression(boxes: 'NdarrayOrTensor', scores: 'NdarrayOrTensor', nms_thresh: 'float', max_proposals: 'int'=-1, box_overlap_metric: 'Callable'=box_iou) ->NdarrayOrTensor:
    """
    Non-maximum suppression (NMS).

    Args:
        boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
        scores: prediction scores of the boxes, sized (N,). This function keeps boxes with higher scores.
        nms_thresh: threshold of NMS. Discards all overlapping boxes with box_overlap > nms_thresh.
        max_proposals: maximum number of boxes it keeps.
            If ``max_proposals`` = -1, there is no limit on the number of boxes that are kept.
        box_overlap_metric: the metric to compute overlap between boxes.

    Returns:
        Indexes of ``boxes`` that are kept after NMS.

    Example:
        .. code-block:: python

            boxes = torch.ones(10,6)
            scores = torch.ones(10)
            keep = non_max_suppression(boxes, scores, num_thresh=0.1)
            boxes_after_nms = boxes[keep]
    """
    if boxes.shape[0] == 0:
        return convert_to_dst_type(src=np.array([]), dst=boxes, dtype=torch.long)[0]
    if boxes.shape[0] != scores.shape[0]:
        raise ValueError(f'boxes and scores should have same length, got boxes shape {boxes.shape}, scores shape {scores.shape}')
    boxes_t, *_ = convert_data_type(boxes, torch.Tensor)
    scores_t, *_ = convert_to_dst_type(scores, boxes_t)
    sort_idxs = torch.argsort(scores_t, dim=0, descending=True)
    boxes_sort = deepcopy(boxes_t)[sort_idxs, :]
    pick = []
    idxs = torch.Tensor(list(range(0, boxes_sort.shape[0])))
    while len(idxs) > 0:
        i = int(idxs[0].item())
        pick.append(i)
        if len(pick) >= max_proposals >= 1:
            break
        box_overlap = box_overlap_metric(boxes_sort[idxs, :], boxes_sort[i:i + 1, :])
        to_keep_idx = (box_overlap <= nms_thresh).flatten()
        to_keep_idx[0] = False
        idxs = idxs[to_keep_idx]
    pick_idx = sort_idxs[pick]
    return convert_to_dst_type(src=pick_idx, dst=boxes, dtype=pick_idx.dtype)[0]


def batched_nms(boxes: 'NdarrayOrTensor', scores: 'NdarrayOrTensor', labels: 'NdarrayOrTensor', nms_thresh: 'float', max_proposals: 'int'=-1, box_overlap_metric: 'Callable'=box_iou) ->NdarrayOrTensor:
    """
    Performs non-maximum suppression in a batched fashion.
    Each labels value correspond to a category, and NMS will not be applied between elements of different categories.

    Adapted from https://github.com/MIC-DKFZ/nnDetection/blob/main/nndet/core/boxes/nms.py

    Args:
        boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
        scores: prediction scores of the boxes, sized (N,). This function keeps boxes with higher scores.
        labels: indices of the categories for each one of the boxes. sized(N,), value range is (0, num_classes)
        nms_thresh: threshold of NMS. Discards all overlapping boxes with box_overlap > nms_thresh.
        max_proposals: maximum number of boxes it keeps.
            If ``max_proposals`` = -1, there is no limit on the number of boxes that are kept.
        box_overlap_metric: the metric to compute overlap between boxes.

    Returns:
        Indexes of ``boxes`` that are kept after NMS.
    """
    if boxes.shape[0] == 0:
        return convert_to_dst_type(src=np.array([]), dst=boxes, dtype=torch.long)[0]
    boxes_t, *_ = convert_data_type(boxes, torch.Tensor, dtype=torch.float32)
    scores_t, *_ = convert_to_dst_type(scores, boxes_t)
    labels_t, *_ = convert_to_dst_type(labels, boxes_t, dtype=torch.long)
    max_coordinate = boxes_t.max()
    offsets = labels_t * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = non_max_suppression(boxes_for_nms, scores_t, nms_thresh, max_proposals, box_overlap_metric)
    return convert_to_dst_type(src=keep, dst=boxes, dtype=keep.dtype)[0]


def spatial_crop_boxes(boxes: 'NdarrayTensor', roi_start: 'Sequence[int] | NdarrayOrTensor', roi_end: 'Sequence[int] | NdarrayOrTensor', remove_empty: 'bool'=True) ->tuple[NdarrayTensor, NdarrayOrTensor]:
    """
    This function generate the new boxes when the corresponding image is cropped to the given ROI.
    When ``remove_empty=True``, it makes sure the bounding boxes are within the new cropped image.

    Args:
        boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
        roi_start: voxel coordinates for start of the crop ROI, negative values allowed.
        roi_end: voxel coordinates for end of the crop ROI, negative values allowed.
        remove_empty: whether to remove the boxes that are actually empty

    Returns:
        - cropped boxes, boxes[keep], does not share memory with original boxes
        - ``keep``, it indicates whether each box in ``boxes`` are kept when ``remove_empty=True``.
    """
    boxes_t = convert_data_type(boxes, torch.Tensor)[0].clone()
    boxes_t = boxes_t
    roi_start_t = convert_to_dst_type(src=roi_start, dst=boxes_t, wrap_sequence=True)[0]
    roi_end_t = convert_to_dst_type(src=roi_end, dst=boxes_t, wrap_sequence=True)[0]
    roi_end_t = torch.maximum(roi_end_t, roi_start_t)
    spatial_dims = get_spatial_dims(boxes=boxes, spatial_size=roi_end)
    for axis in range(0, spatial_dims):
        boxes_t[:, axis] = boxes_t[:, axis].clamp(min=roi_start_t[axis], max=roi_end_t[axis] - TO_REMOVE)
        boxes_t[:, axis + spatial_dims] = boxes_t[:, axis + spatial_dims].clamp(min=roi_start_t[axis], max=roi_end_t[axis] - TO_REMOVE)
        boxes_t[:, axis] -= roi_start_t[axis]
        boxes_t[:, axis + spatial_dims] -= roi_start_t[axis]
    if remove_empty:
        keep_t = boxes_t[:, spatial_dims] >= boxes_t[:, 0] + 1 - TO_REMOVE
        for axis in range(1, spatial_dims):
            keep_t = keep_t & (boxes_t[:, axis + spatial_dims] >= boxes_t[:, axis] + 1 - TO_REMOVE)
        boxes_t = boxes_t[keep_t]
    else:
        keep_t = torch.full_like(boxes_t[:, 0], fill_value=True, dtype=torch.bool)
    boxes_keep, *_ = convert_to_dst_type(src=boxes_t, dst=boxes)
    keep, *_ = convert_to_dst_type(src=keep_t, dst=boxes, dtype=keep_t.dtype)
    return boxes_keep, keep


def clip_boxes_to_image(boxes: 'NdarrayOrTensor', spatial_size: 'Sequence[int] | NdarrayOrTensor', remove_empty: 'bool'=True) ->tuple[NdarrayOrTensor, NdarrayOrTensor]:
    """
    This function clips the ``boxes`` to makes sure the bounding boxes are within the image.

    Args:
        boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
        spatial_size: The spatial size of the image where the boxes are attached. len(spatial_size) should be in [2, 3].
        remove_empty: whether to remove the boxes that are actually empty

    Returns:
        - clipped boxes, boxes[keep], does not share memory with original boxes
        - ``keep``, it indicates whether each box in ``boxes`` are kept when ``remove_empty=True``.
    """
    spatial_dims = get_spatial_dims(boxes=boxes, spatial_size=spatial_size)
    return spatial_crop_boxes(boxes, roi_start=[0] * spatial_dims, roi_end=spatial_size, remove_empty=remove_empty)


def version_leq(lhs: 'str', rhs: 'str') ->bool:
    """
    Returns True if version `lhs` is earlier or equal to `rhs`.

    Args:
        lhs: version name to compare with `rhs`, return True if earlier or equal to `rhs`.
        rhs: version name to compare with `lhs`, return True if later or equal to `lhs`.

    """
    lhs, rhs = str(lhs), str(rhs)
    pkging, has_ver = optional_import('packaging.Version')
    if has_ver:
        try:
            return cast(bool, pkging.version.Version(lhs) <= pkging.version.Version(rhs))
        except pkging.version.InvalidVersion:
            return True
    lhs_, rhs_ = parse_version_strs(lhs, rhs)
    for l, r in zip(lhs_, rhs_):
        if l != r:
            if isinstance(l, int) and isinstance(r, int):
                return l < r
            return f'{l}' < f'{r}'
    return True


def is_module_ver_at_least(module, version):
    """Determine if a module's version is at least equal to the given value.

    Args:
        module: imported module's name, e.g., `np` or `torch`.
        version: required version, given as a tuple, e.g., `(1, 8, 0)`.
    Returns:
        `True` if module is the given version or newer.
    """
    test_ver = '.'.join(map(str, version))
    return module.__version__ != test_ver and version_leq(test_ver, module.__version__)


def floor_divide(a: 'NdarrayOrTensor', b) ->NdarrayOrTensor:
    """`np.floor_divide` with equivalent implementation for torch.

    As of pt1.8, use `torch.div(..., rounding_mode="floor")`, and
    before that, use `torch.floor_divide`.

    Args:
        a: first array/tensor
        b: scalar to divide by

    Returns:
        Element-wise floor division between two arrays/tensors.
    """
    if isinstance(a, torch.Tensor):
        if is_module_ver_at_least(torch, (1, 8, 0)):
            return torch.div(a, b, rounding_mode='floor')
        return torch.floor_divide(a, b)
    return np.floor_divide(a, b)


class BoxSelector:
    """
    Box selector which selects the predicted boxes.
    The box selection is performed with the following steps:

    #. For each level, discard boxes with scores less than self.score_thresh.
    #. For each level, keep boxes with top self.topk_candidates_per_level scores.
    #. For the whole image, perform non-maximum suppression (NMS) on boxes, with overlapping threshold nms_thresh.
    #. For the whole image, keep boxes with top self.detections_per_img scores.

    Args:
        apply_sigmoid: whether to apply sigmoid to get scores from classification logits
        score_thresh: no box with scores less than score_thresh will be kept
        topk_candidates_per_level: max number of boxes to keep for each level
        nms_thresh: box overlapping threshold for NMS
        detections_per_img: max number of boxes to keep for each image

    Example:

        .. code-block:: python

            input_param = {
                "apply_sigmoid": True,
                "score_thresh": 0.1,
                "topk_candidates_per_level": 2,
                "nms_thresh": 0.1,
                "detections_per_img": 5,
            }
            box_selector = BoxSelector(**input_param)
            boxes = [torch.randn([3,6]), torch.randn([7,6])]
            logits = [torch.randn([3,3]), torch.randn([7,3])]
            spatial_size = (8,8,8)
            selected_boxes, selected_scores, selected_labels = box_selector.select_boxes_per_image(
                boxes, logits, spatial_size
            )
    """

    def __init__(self, box_overlap_metric: 'Callable'=box_iou, apply_sigmoid: 'bool'=True, score_thresh: 'float'=0.05, topk_candidates_per_level: 'int'=1000, nms_thresh: 'float'=0.5, detections_per_img: 'int'=300):
        self.box_overlap_metric = box_overlap_metric
        self.apply_sigmoid = apply_sigmoid
        self.score_thresh = score_thresh
        self.topk_candidates_per_level = topk_candidates_per_level
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def select_top_score_idx_per_level(self, logits: 'Tensor') ->tuple[Tensor, Tensor, Tensor]:
        """
        Select indices with highest scores.

        The indices selection is performed with the following steps:

        #. If self.apply_sigmoid, get scores by applying sigmoid to logits. Otherwise, use logits as scores.
        #. Discard indices with scores less than self.score_thresh
        #. Keep indices with top self.topk_candidates_per_level scores

        Args:
            logits: predicted classification logits, Tensor sized (N, num_classes)

        Return:
            - topk_idxs: selected M indices, Tensor sized (M, )
            - selected_scores: selected M scores, Tensor sized (M, )
            - selected_labels: selected M labels, Tensor sized (M, )
        """
        num_classes = logits.shape[-1]
        if self.apply_sigmoid:
            scores = torch.sigmoid(logits).flatten()
        else:
            scores = logits.flatten()
        keep_idxs = scores > self.score_thresh
        scores = scores[keep_idxs]
        flatten_topk_idxs = torch.where(keep_idxs)[0]
        num_topk = min(self.topk_candidates_per_level, flatten_topk_idxs.size(0))
        selected_scores, idxs = scores.topk(num_topk)
        flatten_topk_idxs = flatten_topk_idxs[idxs]
        selected_labels = flatten_topk_idxs % num_classes
        topk_idxs = floor_divide(flatten_topk_idxs, num_classes)
        return topk_idxs, selected_scores, selected_labels

    def select_boxes_per_image(self, boxes_list: 'list[Tensor]', logits_list: 'list[Tensor]', spatial_size: 'list[int] | tuple[int]') ->tuple[Tensor, Tensor, Tensor]:
        """
        Postprocessing to generate detection result from classification logits and boxes.

        The box selection is performed with the following steps:

        #. For each level, discard boxes with scores less than self.score_thresh.
        #. For each level, keep boxes with top self.topk_candidates_per_level scores.
        #. For the whole image, perform non-maximum suppression (NMS) on boxes, with overlapping threshold nms_thresh.
        #. For the whole image, keep boxes with top self.detections_per_img scores.

        Args:
            boxes_list: list of predicted boxes from a single image,
                each element i is a Tensor sized (N_i, 2*spatial_dims)
            logits_list: list of predicted classification logits from a single image,
                each element i is a Tensor sized (N_i, num_classes)
            spatial_size: spatial size of the image

        Return:
            - selected boxes, Tensor sized (P, 2*spatial_dims)
            - selected_scores, Tensor sized (P, )
            - selected_labels, Tensor sized (P, )
        """
        if len(boxes_list) != len(logits_list):
            raise ValueError(f'len(boxes_list) should equal to len(logits_list). Got len(boxes_list)={len(boxes_list)}, len(logits_list)={len(logits_list)}')
        image_boxes = []
        image_scores = []
        image_labels = []
        boxes_dtype = boxes_list[0].dtype
        logits_dtype = logits_list[0].dtype
        for boxes_per_level, logits_per_level in zip(boxes_list, logits_list):
            topk_idxs: 'Tensor'
            topk_idxs, scores_per_level, labels_per_level = self.select_top_score_idx_per_level(logits_per_level)
            boxes_per_level = boxes_per_level[topk_idxs]
            keep: 'Tensor'
            boxes_per_level, keep = clip_boxes_to_image(boxes_per_level, spatial_size, remove_empty=True)
            image_boxes.append(boxes_per_level)
            image_scores.append(scores_per_level[keep])
            image_labels.append(labels_per_level[keep])
        image_boxes_t: 'Tensor' = torch.cat(image_boxes, dim=0)
        image_scores_t: 'Tensor' = torch.cat(image_scores, dim=0)
        image_labels_t: 'Tensor' = torch.cat(image_labels, dim=0)
        keep_t: 'Tensor' = batched_nms(image_boxes_t, image_scores_t, image_labels_t, self.nms_thresh, box_overlap_metric=self.box_overlap_metric, max_proposals=self.detections_per_img)
        selected_boxes = image_boxes_t[keep_t]
        selected_scores = image_scores_t[keep_t]
        selected_labels = image_labels_t[keep_t]
        return selected_boxes, selected_scores, selected_labels


class HardNegativeSamplerBase:
    """
    Base class of hard negative sampler.

    Hard negative sampler is used to suppress false positive rate in classification tasks.
    During training, it select negative samples with high prediction scores.

    The training workflow is described as the follows:
    1) forward network and get prediction scores (classification prob/logits) for all the samples;
    2) use hard negative sampler to choose negative samples with high prediction scores and some positive samples;
    3) compute classification loss for the selected samples;
    4) do back propagation.

    Args:
        pool_size: when we need ``num_neg`` hard negative samples, they will be randomly selected from
            ``num_neg * pool_size`` negative samples with the highest prediction scores.
            Larger ``pool_size`` gives more randomness, yet selects negative samples that are less 'hard',
            i.e., negative samples with lower prediction scores.
    """

    def __init__(self, pool_size: 'float'=10) ->None:
        self.pool_size = pool_size

    def select_negatives(self, negative: 'Tensor', num_neg: 'int', fg_probs: 'Tensor') ->Tensor:
        """
        Select hard negative samples.

        Args:
            negative: indices of all the negative samples, sized (P,),
                where P is the number of negative samples
            num_neg: number of negative samples to sample
            fg_probs: maximum foreground prediction scores (probability) across all the classes
                for each sample, sized (A,), where A is the number of samples.

        Returns:
            binary mask of negative samples to choose, sized (A,),
                where A is the number of samples in one image
        """
        if negative.numel() > fg_probs.numel():
            raise ValueError('The number of negative samples should not be larger than the number of all samples.')
        pool = int(num_neg * self.pool_size)
        pool = min(negative.numel(), pool)
        _, negative_idx_pool = fg_probs[negative].topk(pool, dim=0, sorted=True)
        hard_negative = negative[negative_idx_pool]
        perm2 = torch.randperm(hard_negative.numel(), device=hard_negative.device)[:num_neg]
        selected_neg_idx = hard_negative[perm2]
        neg_mask = torch.zeros_like(fg_probs, dtype=torch.uint8)
        neg_mask[selected_neg_idx] = 1
        return neg_mask


def max(x: 'NdarrayTensor', dim: 'int | tuple | None'=None, **kwargs) ->NdarrayTensor:
    """`torch.max` with equivalent implementation for numpy

    Args:
        x: array/tensor.

    Returns:
        the maximum of x.

    """
    ret: 'NdarrayTensor'
    if dim is None:
        ret = np.max(x, **kwargs) if isinstance(x, (np.ndarray, list)) else torch.max(x, **kwargs)
    elif isinstance(x, (np.ndarray, list)):
        ret = np.max(x, axis=dim, **kwargs)
    else:
        ret = torch.max(x, int(dim), **kwargs)
    return ret[0] if isinstance(ret, tuple) else ret


class HardNegativeSampler(HardNegativeSamplerBase):
    """
    HardNegativeSampler is used to suppress false positive rate in classification tasks.
    During training, it selects negative samples with high prediction scores.

    The training workflow is described as the follows:
    1) forward network and get prediction scores (classification prob/logits) for all the samples;
    2) use hard negative sampler to choose negative samples with high prediction scores and some positive samples;
    3) compute classification loss for the selected samples;
    4) do back propagation.

    Args:
        batch_size_per_image: number of training samples to be randomly selected per image
        positive_fraction: percentage of positive elements in the selected samples
        min_neg: minimum number of negative samples to select if possible.
        pool_size: when we need ``num_neg`` hard negative samples, they will be randomly selected from
            ``num_neg * pool_size`` negative samples with the highest prediction scores.
            Larger ``pool_size`` gives more randomness, yet selects negative samples that are less 'hard',
            i.e., negative samples with lower prediction scores.
    """

    def __init__(self, batch_size_per_image: 'int', positive_fraction: 'float', min_neg: 'int'=1, pool_size: 'float'=10) ->None:
        super().__init__(pool_size=pool_size)
        self.min_neg = min_neg
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        logging.info('Sampling hard negatives on a per batch basis')

    def __call__(self, target_labels: 'list[Tensor]', concat_fg_probs: 'Tensor') ->tuple[list[Tensor], list[Tensor]]:
        """
        Select positives and hard negatives from list samples per image.
        Hard negative sampler will be applied to each image independently.

        Args:
            target_labels: list of labels per image.
                For image i in the batch, target_labels[i] is a Tensor sized (A_i,),
                where A_i is the number of samples in image i.
                Positive samples have positive labels, negative samples have label 0.
            concat_fg_probs: concatenated maximum foreground probability for all the images, sized (R,),
                where R is the sum of all samples inside one batch, i.e., R = A_0 + A_1 + ...

        Returns:
            - list of binary mask for positive samples
            - list of binary mask for negative samples

        Example:
            .. code-block:: python

                sampler = HardNegativeSampler(
                    batch_size_per_image=6, positive_fraction=0.5, min_neg=1, pool_size=2
                )
                # two images with different number of samples
                target_labels = [ torch.tensor([0,1]), torch.tensor([1,0,2,1])]
                concat_fg_probs = torch.rand(6)
                pos_idx_list, neg_idx_list = sampler(target_labels, concat_fg_probs)
        """
        samples_per_image = [samples_in_image.shape[0] for samples_in_image in target_labels]
        fg_probs = concat_fg_probs.split(samples_per_image, 0)
        return self.select_samples_img_list(target_labels, fg_probs)

    def select_samples_img_list(self, target_labels: 'list[Tensor]', fg_probs: 'list[Tensor]') ->tuple[list[Tensor], list[Tensor]]:
        """
        Select positives and hard negatives from list samples per image.
        Hard negative sampler will be applied to each image independently.

        Args:
            target_labels: list of labels per image.
                For image i in the batch, target_labels[i] is a Tensor sized (A_i,),
                where A_i is the number of samples in image i.
                Positive samples have positive labels, negative samples have label 0.
            fg_probs: list of maximum foreground probability per images,
                For image i in the batch, target_labels[i] is a Tensor sized (A_i,),
                where A_i is the number of samples in image i.

        Returns:
            - list of binary mask for positive samples
            - list binary mask for negative samples

        Example:
            .. code-block:: python

                sampler = HardNegativeSampler(
                    batch_size_per_image=6, positive_fraction=0.5, min_neg=1, pool_size=2
                )
                # two images with different number of samples
                target_labels = [ torch.tensor([0,1]), torch.tensor([1,0,2,1])]
                fg_probs = [ torch.rand(2), torch.rand(4)]
                pos_idx_list, neg_idx_list = sampler.select_samples_img_list(target_labels, fg_probs)
        """
        pos_idx = []
        neg_idx = []
        if len(target_labels) != len(fg_probs):
            raise ValueError(f'Require len(target_labels) == len(fg_probs). Got len(target_labels)={len(target_labels)}, len(fg_probs)={len(fg_probs)}.')
        for labels_per_img, fg_probs_per_img in zip(target_labels, fg_probs):
            pos_idx_per_image_mask, neg_idx_per_image_mask = self.select_samples_per_img(labels_per_img, fg_probs_per_img)
            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)
        return pos_idx, neg_idx

    def select_samples_per_img(self, labels_per_img: 'Tensor', fg_probs_per_img: 'Tensor') ->tuple[Tensor, Tensor]:
        """
        Select positives and hard negatives from samples.

        Args:
            labels_per_img: labels, sized (A,).
                Positive samples have positive labels, negative samples have label 0.
            fg_probs_per_img: maximum foreground probability, sized (A,)

        Returns:
            - binary mask for positive samples, sized (A,)
            - binary mask for negative samples, sized (A,)

        Example:
            .. code-block:: python

                sampler = HardNegativeSampler(
                    batch_size_per_image=6, positive_fraction=0.5, min_neg=1, pool_size=2
                )
                # two images with different number of samples
                target_labels = torch.tensor([1,0,2,1])
                fg_probs = torch.rand(4)
                pos_idx, neg_idx = sampler.select_samples_per_img(target_labels, fg_probs)
        """
        if labels_per_img.numel() != fg_probs_per_img.numel():
            raise ValueError('labels_per_img and fg_probs_per_img should have same number of elements.')
        positive = torch.where(labels_per_img >= 1)[0]
        negative = torch.where(labels_per_img == 0)[0]
        num_pos = self.get_num_pos(positive)
        pos_idx_per_image_mask = self.select_positives(positive, num_pos, labels_per_img)
        num_neg = self.get_num_neg(negative, num_pos)
        neg_idx_per_image_mask = self.select_negatives(negative, num_neg, fg_probs_per_img)
        return pos_idx_per_image_mask, neg_idx_per_image_mask

    def get_num_pos(self, positive: 'torch.Tensor') ->int:
        """
        Number of positive samples to draw

        Args:
            positive: indices of positive samples

        Returns:
            number of positive sample
        """
        num_pos = int(self.batch_size_per_image * self.positive_fraction)
        num_pos = min(positive.numel(), num_pos)
        return num_pos

    def get_num_neg(self, negative: 'torch.Tensor', num_pos: 'int') ->int:
        """
        Sample enough negatives to fill up ``self.batch_size_per_image``

        Args:
            negative: indices of positive samples
            num_pos: number of positive samples to draw

        Returns:
            number of negative samples
        """
        num_neg = int(max(1, num_pos) * abs(1 - 1.0 / float(self.positive_fraction)))
        num_neg = min(negative.numel(), max(num_neg, self.min_neg))
        return num_neg

    def select_positives(self, positive: 'Tensor', num_pos: 'int', labels: 'Tensor') ->Tensor:
        """
        Select positive samples

        Args:
            positive: indices of positive samples, sized (P,),
                where P is the number of positive samples
            num_pos: number of positive samples to sample
            labels: labels for all samples, sized (A,),
                where A is the number of samples.

        Returns:
            binary mask of positive samples to choose, sized (A,),
                where A is the number of samples in one image
        """
        if positive.numel() > labels.numel():
            raise ValueError('The number of positive samples should not be larger than the number of all samples.')
        perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        pos_idx_per_image = positive[perm1]
        pos_idx_per_image_mask = torch.zeros_like(labels, dtype=torch.uint8)
        pos_idx_per_image_mask[pos_idx_per_image] = 1
        return pos_idx_per_image_mask


class PytorchPadMode(StrEnum):
    """
    See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    """
    CONSTANT = 'constant'
    REFLECT = 'reflect'
    REPLICATE = 'replicate'
    CIRCULAR = 'circular'


class Inferer(ABC):
    """
    A base class for model inference.
    Extend this class to support operations during inference, e.g. a sliding window method.

    Example code::

        device = torch.device("cuda:0")
        transform = Compose([ToTensor(), LoadImage(image_only=True)])
        data = transform(img_path).to(device)
        model = UNet(...).to(device)
        inferer = SlidingWindowInferer(...)

        model.eval()
        with torch.no_grad():
            pred = inferer(inputs=data, network=model)
        ...

    """

    @abstractmethod
    def __call__(self, inputs: 'torch.Tensor', network: 'Callable', *args: Any, **kwargs: Any) ->Any:
        """
        Run inference on `inputs` with the `network` model.

        Args:
            inputs: input of the model inference.
            network: model for inference.
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f'Subclass {self.__class__.__name__} must implement this method.')


def issequenceiterable(obj: 'Any') ->bool:
    """
    Determine if the object is an iterable sequence and is not a string.
    """
    try:
        if hasattr(obj, 'ndim') and obj.ndim == 0:
            return False
    except Exception:
        return False
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))


def ensure_tuple_rep(tup: 'Any', dim: 'int') ->tuple[Any, ...]:
    """
    Returns a copy of `tup` with `dim` values by either shortened or duplicated input.

    Raises:
        ValueError: When ``tup`` is a sequence and ``tup`` length is not ``dim``.

    Examples::

        >>> ensure_tuple_rep(1, 3)
        (1, 1, 1)
        >>> ensure_tuple_rep(None, 3)
        (None, None, None)
        >>> ensure_tuple_rep('test', 3)
        ('test', 'test', 'test')
        >>> ensure_tuple_rep([1, 2, 3], 3)
        (1, 2, 3)
        >>> ensure_tuple_rep(range(3), 3)
        (0, 1, 2)
        >>> ensure_tuple_rep([1, 2], 3)
        ValueError: Sequence must have length 3, got length 2.

    """
    if isinstance(tup, torch.Tensor):
        tup = tup.detach().cpu().numpy()
    if isinstance(tup, np.ndarray):
        tup = tup.tolist()
    if not issequenceiterable(tup):
        return (tup,) * dim
    if len(tup) == dim:
        return tuple(tup)
    raise ValueError(f'Sequence must have length {dim}, got {len(tup)}.')


def compute_importance_map(patch_size: 'tuple[int, ...]', mode: 'BlendMode | str'=BlendMode.CONSTANT, sigma_scale: 'Sequence[float] | float'=0.125, device: 'torch.device | int | str'='cpu', dtype: 'torch.dtype | str | None'=torch.float32) ->torch.Tensor:
    """Get importance map for different weight modes.

    Args:
        patch_size: Size of the required importance map. This should be either H, W [,D].
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: Sigma_scale to calculate sigma for each dimension
            (sigma = sigma_scale * dim_size). Used for gaussian mode only.
        device: Device to put importance map on.
        dtype: Data type of the output importance map.

    Raises:
        ValueError: When ``mode`` is not one of ["constant", "gaussian"].

    Returns:
        Tensor of size patch_size.

    """
    mode = look_up_option(mode, BlendMode)
    device = torch.device(device)
    if mode == BlendMode.CONSTANT:
        importance_map = torch.ones(patch_size, device=device, dtype=torch.float)
    elif mode == BlendMode.GAUSSIAN:
        sigma_scale = ensure_tuple_rep(sigma_scale, len(patch_size))
        sigmas = [(i * sigma_s) for i, sigma_s in zip(patch_size, sigma_scale)]
        for i in range(len(patch_size)):
            x = torch.arange(start=-(patch_size[i] - 1) / 2.0, end=(patch_size[i] - 1) / 2.0 + 1, dtype=torch.float, device=device)
            x = torch.exp(x ** 2 / (-2 * sigmas[i] ** 2))
            importance_map = importance_map.unsqueeze(-1) * x[(None,) * i] if i > 0 else x
    else:
        raise ValueError(f'Unsupported mode: {mode}, available options are [{BlendMode.CONSTANT}, {BlendMode.CONSTANT}].')
    min_non_zero = max(torch.min(importance_map).item(), 0.001)
    importance_map = torch.clamp_(importance_map.to(torch.float), min=min_non_zero)
    return importance_map


class LazyAttr(StrEnum):
    """
    MetaTensor with pending operations requires some key attributes tracked especially when the primary array
    is not up-to-date due to lazy evaluation.
    This class specifies the set of key attributes to be tracked for each MetaTensor.
    See also: :py:func:`monai.transforms.lazy.utils.resample` for more details.
    """
    SHAPE = 'lazy_shape'
    AFFINE = 'lazy_affine'
    PADDING_MODE = 'lazy_padding_mode'
    INTERP_MODE = 'lazy_interpolation_mode'
    DTYPE = 'lazy_dtype'
    ALIGN_CORNERS = 'lazy_align_corners'
    RESAMPLE_MODE = 'lazy_resample_mode'


class MetaKeys(StrEnum):
    """
    Typical keys for MetaObj.meta
    """
    AFFINE = 'affine'
    ORIGINAL_AFFINE = 'original_affine'
    SPATIAL_SHAPE = 'spatial_shape'
    SPACE = 'space'
    ORIGINAL_CHANNEL_DIM = 'original_channel_dim'
    SAVED_TO = 'saved_to'


class TraceKeys(StrEnum):
    """Extra metadata keys used for traceable transforms."""
    CLASS_NAME: 'str' = 'class'
    ID: 'str' = 'id'
    ORIG_SIZE: 'str' = 'orig_size'
    EXTRA_INFO: 'str' = 'extra_info'
    DO_TRANSFORM: 'str' = 'do_transforms'
    KEY_SUFFIX: 'str' = '_transforms'
    NONE: 'str' = 'none'
    TRACING: 'str' = 'tracing'
    STATUSES: 'str' = 'statuses'
    LAZY: 'str' = 'lazy'


class Transform(ABC):
    """
    An abstract class of a ``Transform``.
    A transform is callable that processes ``data``.

    It could be stateful and may modify ``data`` in place,
    the implementation should be aware of:

        #. thread safety when mutating its own states.
           When used from a multi-process context, transform's instance variables are read-only.
           thread-unsafe transforms should inherit :py:class:`monai.transforms.ThreadUnsafe`.
        #. ``data`` content unused by this transform may still be used in the
           subsequent transforms in a composed transform.
        #. storing too much information in ``data`` may cause some memory issue or IPC sync issue,
           especially in the multi-processing environment of PyTorch DataLoader.

    See Also

        :py:class:`monai.transforms.Compose`
    """
    backend: 'list[TransformBackends]' = []

    @abstractmethod
    def __call__(self, data: 'Any'):
        """
        ``data`` is an element which often comes from an iteration over an
        iterable, such as :py:class:`torch.utils.data.Dataset`. This method should
        return an updated version of ``data``.
        To simplify the input validations, most of the transforms assume that

        - ``data`` is a Numpy ndarray, PyTorch Tensor or string,
        - the data shape can be:

          #. string data without shape, `LoadImage` transform expects file paths,
          #. most of the pre-/post-processing transforms expect: ``(num_channels, spatial_dim_1[, spatial_dim_2, ...])``,
             except for example: `AddChannel` expects (spatial_dim_1[, spatial_dim_2, ...])

        - the channel dimension is often not omitted even if number of channels is one.

        This method can optionally take additional arguments to help execute transformation operation.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f'Subclass {self.__class__.__name__} must implement this method.')


KEY = 'test'


def _apply(x, fn):
    if isinstance(x, dict):
        d = deepcopy(x)
        d[KEY] = fn(d[KEY])
        return d
    return fn(x)


class T(Transform):

    def __call__(self, x):
        return _apply(x, convert_to_tensor)


def is_immutable(obj: 'Any') ->bool:
    """
    Determine if the object is an immutable object.

    see also https://github.com/python/cpython/blob/3.11/Lib/copy.py#L109
    """
    return isinstance(obj, (type(None), int, float, bool, complex, str, tuple, bytes, type, range, slice))


class MetaObj:
    """
    Abstract base class that stores data as well as any extra metadata.

    This allows for subclassing `torch.Tensor` and `np.ndarray` through multiple inheritance.

    Metadata is stored in the form of a dictionary.

    Behavior should be the same as extended class (e.g., `torch.Tensor` or `np.ndarray`)
    aside from the extended meta functionality.

    Copying of information:

        * For `c = a + b`, then auxiliary data (e.g., metadata) will be copied from the
          first instance of `MetaObj` if `a.is_batch` is False
          (For batched data, the metadata will be shallow copied for efficiency purposes).

    """

    def __init__(self) ->None:
        self._meta: 'dict' = MetaObj.get_default_meta()
        self._applied_operations: 'list' = MetaObj.get_default_applied_operations()
        self._pending_operations: 'list' = MetaObj.get_default_applied_operations()
        self._is_batch: 'bool' = False

    @staticmethod
    def flatten_meta_objs(*args: Iterable):
        """
        Recursively flatten input and yield all instances of `MetaObj`.
        This means that for both `torch.add(a, b)`, `torch.stack([a, b])` (and
        their numpy equivalents), we return `[a, b]` if both `a` and `b` are of type
        `MetaObj`.

        Args:
            args: Iterables of inputs to be flattened.
        Returns:
            list of nested `MetaObj` from input.
        """
        for a in itertools.chain(*args):
            if isinstance(a, (list, tuple)):
                yield from MetaObj.flatten_meta_objs(a)
            elif isinstance(a, MetaObj):
                yield a

    @staticmethod
    def copy_items(data):
        """returns a copy of the data. list and dict are shallow copied for efficiency purposes."""
        if is_immutable(data):
            return data
        if isinstance(data, (list, dict, np.ndarray)):
            return data.copy()
        if isinstance(data, torch.Tensor):
            return data.detach().clone()
        return deepcopy(data)

    def copy_meta_from(self, input_objs, copy_attr=True, keys=None):
        """
        Copy metadata from a `MetaObj` or an iterable of `MetaObj` instances.

        Args:
            input_objs: list of `MetaObj` to copy data from.
            copy_attr: whether to copy each attribute with `MetaObj.copy_item`.
                note that if the attribute is a nested list or dict, only a shallow copy will be done.
            keys: the keys of attributes to copy from the ``input_objs``.
                If None, all keys from the input_objs will be copied.
        """
        first_meta = input_objs if isinstance(input_objs, MetaObj) else first(input_objs, default=self)
        if not hasattr(first_meta, '__dict__'):
            return self
        first_meta = first_meta.__dict__
        keys = first_meta.keys() if keys is None else keys
        if not copy_attr:
            self.__dict__ = {a: first_meta[a] for a in keys if a in first_meta}
        else:
            self.__dict__.update({a: MetaObj.copy_items(first_meta[a]) for a in keys if a in first_meta})
        return self

    @staticmethod
    def get_default_meta() ->dict:
        """Get the default meta.

        Returns:
            default metadata.
        """
        return {}

    @staticmethod
    def get_default_applied_operations() ->list:
        """Get the default applied operations.

        Returns:
            default applied operations.
        """
        return []

    def __repr__(self) ->str:
        """String representation of class."""
        out: 'str' = '\nMetadata\n'
        if self.meta is not None:
            out += ''.join(f'\t{k}: {v}\n' for k, v in self.meta.items())
        else:
            out += 'None'
        out += '\nApplied operations\n'
        if self.applied_operations is not None:
            out += pprint.pformat(self.applied_operations, indent=2, compact=True, width=120)
        else:
            out += 'None'
        out += f'\nIs batch?: {self.is_batch}'
        return out

    @property
    def meta(self) ->dict:
        """Get the meta. Defaults to ``{}``."""
        return self._meta if hasattr(self, '_meta') else MetaObj.get_default_meta()

    @meta.setter
    def meta(self, d) ->None:
        """Set the meta."""
        if d == TraceKeys.NONE:
            self._meta = MetaObj.get_default_meta()
        else:
            self._meta = d

    @property
    def applied_operations(self) ->list[dict]:
        """Get the applied operations. Defaults to ``[]``."""
        if hasattr(self, '_applied_operations'):
            return self._applied_operations
        return MetaObj.get_default_applied_operations()

    @applied_operations.setter
    def applied_operations(self, t) ->None:
        """Set the applied operations."""
        if t == TraceKeys.NONE:
            self._applied_operations = MetaObj.get_default_applied_operations()
            return
        self._applied_operations = t

    def push_applied_operation(self, t: 'Any') ->None:
        self._applied_operations.append(t)

    def pop_applied_operation(self) ->Any:
        return self._applied_operations.pop()

    @property
    def pending_operations(self) ->list[dict]:
        """Get the pending operations. Defaults to ``[]``."""
        if hasattr(self, '_pending_operations'):
            return self._pending_operations
        return MetaObj.get_default_applied_operations()

    @property
    def has_pending_operations(self) ->bool:
        """
        Determine whether there are pending operations.
        Returns:
            True if there are pending operations; False if not
        """
        return self.pending_operations is not None and len(self.pending_operations) > 0

    def push_pending_operation(self, t: 'Any') ->None:
        self._pending_operations.append(t)

    def pop_pending_operation(self) ->Any:
        return self._pending_operations.pop()

    def clear_pending_operations(self) ->Any:
        self._pending_operations = MetaObj.get_default_applied_operations()

    @property
    def is_batch(self) ->bool:
        """Return whether object is part of batch or not."""
        return self._is_batch if hasattr(self, '_is_batch') else False

    @is_batch.setter
    def is_batch(self, val: 'bool') ->None:
        """Set whether object is part of batch or not."""
        self._is_batch = val


class PostFix(StrEnum):
    """Post-fixes."""

    @staticmethod
    def _get_str(prefix: 'str | None', suffix: 'str') ->str:
        return suffix if prefix is None else f'{prefix}_{suffix}'

    @staticmethod
    def meta(key: 'str | None'=None) ->str:
        return PostFix._get_str(key, 'meta_dict')

    @staticmethod
    def orig_meta(key: 'str | None'=None) ->str:
        return PostFix._get_str(key, 'orig_meta_dict')

    @staticmethod
    def transforms(key: 'str | None'=None) ->str:
        return PostFix._get_str(key, TraceKeys.KEY_SUFFIX[1:])


class SpaceKeys(StrEnum):
    """
    The coordinate system keys, for example, Nifti1 uses Right-Anterior-Superior or "RAS",
    DICOM (0020,0032) uses Left-Posterior-Superior or "LPS". This type does not distinguish spatial 1/2/3D.
    """
    RAS = 'RAS'
    LPS = 'LPS'


@functools.lru_cache(None)
def _get_named_tuple_like_type(func):
    if hasattr(torch, 'return_types') and hasattr(func, '__name__') and hasattr(torch.return_types, func.__name__) and isinstance(getattr(torch.return_types, func.__name__), type):
        return getattr(torch.return_types, func.__name__)
    return None

