
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


from typing import Optional


import warnings


import torch


from torch import Tensor


from torch.utils import cpp_extension


import math


from abc import ABC


from abc import abstractmethod


import enum


import logging


from typing import Callable


from typing import Generic


from typing import NamedTuple


from typing import Set


from typing import TypeVar


import torch._C


import types


import torch.jit


import torch._utils_internal


from collections import namedtuple


from typing import Any


from typing import Dict


from typing import Tuple


from typing import Type


from torch.utils._pytree import LeafSpec


from torch.utils._pytree import PyTree


from torch.utils._pytree import TreeSpec


import collections


import copy


import functools


import inspect


from itertools import chain


from types import CodeType


from types import FunctionType


from types import ModuleType


from typing import Union


from torch._C import ScriptObject


import torch.utils._pytree as pytree


from torch._dispatch.python import enable_python_dispatcher


from torch._subclasses import FakeTensor


from torch._subclasses.fake_tensor import FakeTensorMode


from torch.utils._python_dispatch import _get_current_dispatch_mode


from torch.utils._python_dispatch import _pop_mode_temporarily


from torch.utils._python_dispatch import TorchDispatchMode


from functools import lru_cache


import itertools


from typing import cast


from collections import defaultdict


import re


from typing import FrozenSet


from typing import TYPE_CHECKING


import torch.nn as nn


from torch.nn.modules.module import _addindent


import torch.overrides


from torch.package import Importer


from torch.package import PackageExporter


from torch.package import PackageImporter


from torch.package import sys_importer


from torch.utils._pytree import _register_pytree_node


from torch.utils._pytree import Context


from typing import Iterator


from torch.hub import tqdm


import numbers


import typing


from torch._jit_internal import boolean_dispatched


from typing import Iterable


from typing import OrderedDict


from torch.nn import Module


from torch.nn import Parameter


from torch import nn


from functools import wraps


from warnings import warn


from copy import copy


from functools import partial


from torch.autograd import Function


import torch.onnx


from torch.onnx.symbolic_helper import _get_tensor_sizes


from torch.nn import Sequential


from inspect import getcallargs


from torch.overrides import get_testing_overrides


from copy import deepcopy


import torch.nn.functional as F


from torch.fx import GraphModule as TorchGraphModule


import numpy as np


from functools import reduce


from inspect import isclass


from inspect import signature


from abc import ABCMeta


from torch.nn.utils.rnn import PackedSequence


from torch.nn import AdaptiveAvgPool2d


from torch.nn import AvgPool2d


from torch.nn import Conv1d


from torch.nn import Conv2d


from torch.nn import Conv3d


from torch.nn import functional as F


from torch.nn import ConvTranspose1d


from torch.nn import ConvTranspose2d


from torch.nn import ConvTranspose3d


from torch.nn.functional import conv_transpose1d


from torch.nn.functional import conv_transpose2d


from torch.nn.functional import conv_transpose3d


from torch.nn import Embedding


from torch.nn.functional import embedding


from torch.nn import Linear


from torch.nn.functional import linear


from torch.nn.init import constant_


from torch.nn.init import xavier_normal_


from torch.nn.init import xavier_uniform_


from torch.nn import Upsample


from torch.nn import UpsamplingBilinear2d


from torch.nn import UpsamplingNearest2d


from torch.nn.functional import interpolate


from torch import tensor


from torch.nn import Identity


from enum import Enum


from torch.nn import BatchNorm1d


from torch.nn import BatchNorm2d


from torch.nn import MaxPool2d


from torch.nn import ModuleList


from torch.nn import Dropout


from torch import hub


import torch.nn.init as init


import random


import time


import torch.optim as optim


from torch.optim.lr_scheduler import MultiStepLR


from torch.utils.data import DataLoader


import torchvision


from torchvision import transforms


from torchvision.datasets import CIFAR10


from torchvision.datasets import MNIST


import torch.optim.lr_scheduler as lrs


from torch.utils.data import Dataset


from torch.utils.data import Subset


import torchvision.transforms as transforms


from itertools import product


from types import SimpleNamespace


import pandas as pd


import torch.backends.cudnn as cudnn


import torch.nn.parallel


import torch.optim


import torch.utils.data


import torch.utils.data.distributed


from time import sleep


import torchvision.datasets as datasets


import torch.utils.cpp_extension


from torch.utils._pytree import tree_map


from torch._decomp import get_decompositions


from scipy import linalg


from torch.nn.functional import adaptive_avg_pool2d


import torchvision.transforms as TF


from typing import Mapping


import torch.utils.data as data


from torchvision.transforms import CenterCrop


from torchvision.transforms import Compose


from torchvision.transforms import RandomCrop


from torchvision.transforms import RandomHorizontalFlip


from torchvision.transforms import RandomVerticalFlip


from torchvision.transforms import Resize


from torchvision.transforms import ToTensor


from scipy.io.wavfile import write


from scipy.io.wavfile import read


from scipy.signal import get_window


from torch.autograd import Variable


from torchvision.models import alexnet


from torchvision.models import densenet121


from torchvision.models import mnasnet0_5


from torchvision.models import mobilenet_v2


from torchvision.models import resnet18


from torchvision.models import shufflenet_v2_x0_5


from torchvision.models import squeezenet1_0


from torchvision import models


from torch.utils.data import TensorDataset


from torch.nn import MultiheadAttention


from torch.nn import BatchNorm3d


import torchvision.models as modelzoo


def compatibility(is_backward_compatible: 'bool'):
    if is_backward_compatible:

        def mark_back_compat(fn):
            docstring = textwrap.dedent(getattr(fn, '__doc__', None) or '')
            docstring += '\n.. note::\n    Backwards-compatibility for this API is guaranteed.\n'
            fn.__doc__ = docstring
            _BACK_COMPAT_OBJECTS.setdefault(fn)
            _MARKED_WITH_COMATIBLITY.setdefault(fn)
            return fn
        return mark_back_compat
    else:

        def mark_not_back_compat(fn):
            docstring = textwrap.dedent(getattr(fn, '__doc__', None) or '')
            docstring += '\n.. warning::\n    This API is experimental and is *NOT* backward-compatible.\n'
            fn.__doc__ = docstring
            _MARKED_WITH_COMATIBLITY.setdefault(fn)
            return fn
        return mark_not_back_compat


def _format_target(base: 'str', target: 'str') ->str:
    elems = target.split('.')
    r = base
    for e in elems:
        if not e.isidentifier():
            r = f'getattr({r}, "{e}")'
        else:
            r = f'{r}.{e}'
    return r


def _find_module_of_method(orig_method: 'Callable[..., Any]') ->str:
    name = orig_method.__name__
    module = orig_method.__module__
    if module is not None:
        return module
    for guess in [torch, torch.nn.functional]:
        if getattr(guess, name, None) is orig_method:
            return guess.__name__
    raise RuntimeError(f'cannot find module for {orig_method}')


def _get_qualified_name(func: 'Callable[..., Any]') ->str:
    if getattr(builtins, func.__name__, None) is func:
        return func.__name__
    if isinstance(func, types.MethodDescriptorType) and func is getattr(torch.Tensor, func.__name__, None):
        return f'torch.Tensor.{func.__name__}'
    name = func.__name__
    module = _find_module_of_method(func)
    module = module.replace('torch._ops', 'torch.ops')
    module = module.replace('brevitas.backport._ops', 'brevitas.backport.ops')
    if module == 'torch' and name == 'segment_reduce':
        name = '_' + name
    return f'{module}.{name}'


def _is_from_torch(obj: 'Any') ->bool:
    module_name = getattr(obj, '__module__', None)
    if module_name is not None:
        base_module = module_name.partition('.')[0]
        return base_module == 'torch' and not module_name.startswith('torch._dynamo.') and not module_name.startswith('torch._inductor.')
    name = getattr(obj, '__name__', None)
    if name is not None and name != 'torch':
        for guess in [torch, torch.nn.functional]:
            if getattr(guess, name, None) is obj:
                return True
    return False


_origin_type_map = {list: List, dict: Dict, set: Set, frozenset: FrozenSet, tuple: Tuple}


def _type_repr(obj):
    """Return the repr() of an object, special-casing types (internal helper).
    If obj is a type, we return a shorter version than the default
    type.__repr__, based on the module and qualified name, which is
    typically enough to uniquely identify a type.  For everything
    else, we fall back on repr(obj).
    """
    if isinstance(obj, type):
        if obj.__module__ == 'builtins':
            return obj.__qualname__
        return f'{obj.__module__}.{obj.__qualname__}'
    if obj is ...:
        return '...'
    if isinstance(obj, types.FunctionType):
        return obj.__name__
    return repr(obj)


dtype_abbrs = {torch.bfloat16: 'bf16', torch.float64: 'f64', torch.float32: 'f32', torch.float16: 'f16', torch.complex64: 'c64', torch.complex128: 'c128', torch.int8: 'i8', torch.int16: 'i16', torch.int32: 'i32', torch.int64: 'i64', torch.bool: 'b8', torch.uint8: 'u8'}


inplace_methods = {'iadd': '{} += {}', 'iand': '{} &= {}', 'ifloordiv': '{} //= {}', 'ilshift': '{} <<= {}', 'imod': '{} %= {}', 'imul': '{} *= {}', 'imatmul': '{} @= {}', 'ior': '{} |= {}', 'ipow': '{} **= {}', 'irshift': '{} >>= {}', 'isub': '{} -= {}', 'itruediv': '{} /= {}', 'ixor': '{} ^= {}', 'setitem': '{}[{}] = {}'}


reflectable_magic_methods = {'add': '{} + {}', 'sub': '{} - {}', 'mul': '{} * {}', 'floordiv': '{} // {}', 'truediv': '{} / {}', 'div': '{} / {}', 'mod': '{} % {}', 'pow': '{} ** {}', 'lshift': '{} << {}', 'rshift': '{} >> {}', 'and_': '{} & {}', 'or_': '{} | {}', 'xor': '{} ^ {}', 'getitem': '{}[{}]', 'matmul': '{} @ {}'}


magic_methods = dict({'eq': '{} == {}', 'ne': '{} != {}', 'lt': '{} < {}', 'gt': '{} > {}', 'le': '{} <= {}', 'ge': '{} >= {}', 'pos': '+{}', 'neg': '-{}', 'invert': '~{}'}, **reflectable_magic_methods)


BaseArgumentTypes = Union[str, int, float, bool, complex, torch.dtype, torch.Tensor, torch.device, torch.memory_format, torch.layout]


Argument = Optional[Union[Tuple[Any, ...], List[Any], Dict[str, Any], slice, range, 'Node', BaseArgumentTypes]]

