
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


import numpy as np


import torch


import torch.nn as nn


import torch.optim as optim


import torch.multiprocessing as mp


import torch.distributed as dist


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data.distributed import DistributedSampler


import string


import random


import tensorflow


import tensorflow.data as tf_data


import tensorflow.strings as tf_strings


import re


import matplotlib.pyplot as plt


import torch.nn.functional as F


import torchvision


from torchvision import datasets


from torchvision import models


from torchvision import transforms


import math


import tensorflow as tf


from tensorflow import data as tf_data


import scipy.io


import matplotlib.patches as patches


import pandas as pd


from tensorflow import keras


import warnings


import types


from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice


import functools


import itertools


import torch.nn.functional as tnn


import logging


import collections


import tensorflow.summary as summary


from tensorflow.compat.v1 import SummaryMetadata


from tensorflow.core.util import event_pb2


from tensorflow.python.lib.io import tf_record


import inspect


from functools import wraps


from numpy.lib.stride_tricks import as_strided


import scipy.ndimage


import scipy.signal


from tensorflow.python.ops.numpy_ops import np_config


import pandas


class TorchModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(8192, 64)
        self.activation1 = torch.nn.ReLU()
        self.dense2 = torch.nn.Linear(64, 8)
        self.activation2 = torch.nn.ReLU()
        self.dense3 = torch.nn.Linear(8, num_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dense1(x)
        x = self.activation1(x)
        x = self.dense2(x)
        x = self.activation2(x)
        x = self.dense3(x)
        x = self.softmax(x)
        return x


class MyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = keras_core.Sequential([layers.Input(shape=(28, 28, 1)), layers.Conv2D(32, kernel_size=(3, 3), activation='relu'), layers.MaxPooling2D(pool_size=(2, 2)), layers.Conv2D(64, kernel_size=(3, 3), activation='relu'), layers.MaxPooling2D(pool_size=(2, 2)), layers.Flatten(), layers.Dropout(0.5), layers.Dense(num_classes)])

    def forward(self, x):
        return self.model(x)


class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = layers.Dense(1)

    def forward(self, x):
        x = self.fc1(x)
        return x


class CallContext:

    def __init__(self, entry_layer):
        self.entry_layer = entry_layer
        self.training = None


def to_snake_case(name):
    name = re.sub('\\W+', '', name)
    name = re.sub('(.)([A-Z][a-z]+)', '\\1_\\2', name)
    name = re.sub('([a-z])([A-Z])', '\\1_\\2', name).lower()
    return name


def uniquify(name):
    object_name_uids = global_state.get_global_attribute('object_name_uids', default=collections.defaultdict(int), set_to_default=True)
    if name in object_name_uids:
        unique_name = f'{name}_{object_name_uids[name]}'
    else:
        unique_name = name
    object_name_uids[name] += 1
    return unique_name


def auto_name(prefix):
    prefix = to_snake_case(prefix)
    return uniquify(prefix)


def current_path():
    name_scope_stack = global_state.get_global_attribute('name_scope_stack')
    if name_scope_stack is None:
        return ''
    return '/'.join(x.name for x in name_scope_stack)


def get_autocast_scope():
    return global_state.get_global_attribute('autocast_scope')


def get_stateless_scope():
    return global_state.get_global_attribute('stateless_scope')


def in_stateless_scope():
    return global_state.get_global_attribute('stateless_scope') is not None


ALLOWED_DTYPES = {'float16', 'float32', 'float64', 'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64', 'bfloat16', 'bool', 'string'}


class Linear(keras.layers.Layer):

    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True)

    def call(self, inputs):
        return ops.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


layer = Linear(64)


config = layer.get_config()


def register_uninitialized_variable(variable):
    uninitialized_variables = global_state.get_global_attribute('uninitialized_variables', [], set_to_default=True)
    uninitialized_variables.append(variable)


def shape_equal(a_shape, b_shape):
    """Return whether a_shape == b_shape (allows None entries)."""
    if len(a_shape) != len(b_shape):
        return False
    for e1, e2 in zip(a_shape, b_shape):
        if e1 is not None and e2 is not None and e1 != e2:
            return False
    return True


def standardize_shape(shape):
    if not isinstance(shape, tuple):
        if shape is None:
            raise ValueError('Undefined shapes are not supported.')
        if not hasattr(shape, '__iter__'):
            raise ValueError(f"Cannot convert '{shape}' to a shape.")
        shape = tuple(shape)
    if config.backend() == 'torch':
        shape = tuple(map(lambda x: int(x) if x is not None else None, shape))
    for e in shape:
        if e is None:
            continue
        if config.backend() == 'jax' and str(e) == 'b':
            continue
        if not isinstance(e, int):
            raise ValueError(f"Cannot convert '{shape}' to a shape. Found invalid entry '{e}'. ")
        if e < 0:
            raise ValueError(f"Cannot convert '{shape}' to a shape. Negative dimensions are not allowed.")
    return shape


class KerasVariable:

    def __init__(self, initializer, shape=None, dtype=None, trainable=True, name=None):
        name = name or auto_name(self.__class__.__name__)
        if not isinstance(name, str) or '/' in name:
            raise ValueError(f'Argument `name` must be a string and cannot contain character `/`. Received: name={name}')
        self.name = name
        parent_path = current_path()
        if parent_path:
            self.path = current_path() + '/' + self.name
        else:
            self.path = self.name
        dtype = standardize_dtype(dtype)
        self._dtype = dtype
        self._shape = None
        self._initializer = None
        self.trainable = trainable
        if callable(initializer):
            if shape is None:
                raise ValueError(f'When creating a Variable from an initializer, the `shape` argument should be specified. Received: initializer={initializer} and shape={shape}')
        if in_stateless_scope():
            if callable(initializer):
                self._value = None
                self._initializer = initializer
                self._shape = standardize_shape(shape)
                register_uninitialized_variable(self)
            else:
                raise ValueError('You are attempting to create a variable while in a stateless scope. This is disallowed. Make sure that all variables are created before you start using your layer/model objects.\n\nIn some cases, you might be seeing this error because you need to implement a `def build(self, input_shape)` method on your layer/model, which will create its variables.\n\nIn some other cases, you might be seeing this error because you are instantiating a `Variable` and assigning it to a layer without going through self.add_variable()/self.add_weight(). Always prefer using these methods (with a `shape` and `initializer` argument).')
        else:
            if callable(initializer):
                value = initializer(shape, dtype=dtype)
            else:
                value = initializer
            self._initialize(value)
            self._shape = tuple(self._value.shape)
        self._ndim = len(self._shape)

    def _deferred_initialize(self):
        if self._value is not None:
            raise ValueError(f'Variable {self.path} is already initialized.')
        if in_stateless_scope():
            raise ValueError('You are attempting to initialize a variable while in a stateless scope. This is disallowed. Make sure that all variables are initialized before you start using your layer/model objects.')
        value = self._initializer(self._shape, dtype=self._dtype)
        self._initialize(value)

    def _maybe_autocast(self, value):
        autocast_scope = get_autocast_scope()
        if autocast_scope is not None:
            return autocast_scope.maybe_cast(value)
        return value

    def numpy(self):
        return np.array(self)

    @property
    def value(self):
        if in_stateless_scope():
            scope = get_stateless_scope()
            value = scope.get_current_value(self)
            if value is not None:
                return self._maybe_autocast(value)
        if self._value is None:
            return self._maybe_autocast(self._initializer(self._shape, dtype=self._dtype))
        return self._maybe_autocast(self._value)

    def assign(self, value):
        value = self._convert_to_tensor(value, dtype=self.dtype)
        if not shape_equal(value.shape, self.shape):
            raise ValueError(f'The shape of the target variable and the shape of the target value in `variable.assign(value)` must match. variable.shape={self.value.shape}, Received: value.shape={value.shape}. Target variable: {self}')
        if in_stateless_scope():
            scope = get_stateless_scope()
            scope.add_update((self, value))
        else:
            self._direct_assign(value)

    def assign_add(self, value):
        self.assign(self + value)

    def assign_sub(self, value):
        self.assign(self - value)

    @property
    def dtype(self):
        autocast_scope = get_autocast_scope()
        if autocast_scope is not None and is_float_dtype(self._dtype):
            return autocast_scope.dtype
        return self._dtype

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return self._ndim

    def __repr__(self):
        return f'<KerasVariable shape={self.shape}, dtype={self.dtype}, path={self.path}>'

    def _initialize(self, value):
        raise NotImplementedError

    def _convert_to_tensor(self, value, dtype=None):
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.value.__getitem__(idx)

    def __array__(self, dtype=None):
        return self.value.__array__(dtype)

    def __bool__(self):
        raise TypeError('A Keras Variable cannot be used as a boolean.')

    def __neg__(self):
        return self.value.__neg__()

    def __pos__(self):
        return self.value.__pos__()

    def __abs__(self):
        return self.value.__abs__()

    def __invert__(self):
        return self.value.__invert__()

    def __eq__(self, other):
        value = self.value
        return value.__eq__(self._convert_to_tensor(other, dtype=value.dtype))

    def __ne__(self, other):
        value = self.value
        return value.__ne__(self._convert_to_tensor(other, dtype=value.dtype))

    def __lt__(self, other):
        value = self.value
        return value.__lt__(self._convert_to_tensor(other, dtype=value.dtype))

    def __le__(self, other):
        value = self.value
        return value.__le__(self._convert_to_tensor(other, dtype=value.dtype))

    def __gt__(self, other):
        value = self.value
        return value.__gt__(self._convert_to_tensor(other, dtype=value.dtype))

    def __ge__(self, other):
        value = self.value
        return value.__ge__(self._convert_to_tensor(other, dtype=value.dtype))

    def __add__(self, other):
        value = self.value
        return value.__add__(self._convert_to_tensor(other, dtype=value.dtype))

    def __radd__(self, other):
        value = self.value
        return value.__radd__(self._convert_to_tensor(other, dtype=value.dtype))

    def __sub__(self, other):
        value = self.value
        return value.__sub__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rsub__(self, other):
        value = self.value
        return value.__rsub__(self._convert_to_tensor(other, dtype=value.dtype))

    def __mul__(self, other):
        value = self.value
        return value.__mul__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rmul__(self, other):
        value = self.value
        return value.__rmul__(self._convert_to_tensor(other, dtype=value.dtype))

    def __div__(self, other):
        value = self.value
        return value.__div__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rdiv__(self, other):
        value = self.value
        return value.__rdiv__(self._convert_to_tensor(other, dtype=value.dtype))

    def __truediv__(self, other):
        value = self.value
        return value.__truediv__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rtruediv__(self, other):
        value = self.value
        return value.__rtruediv__(self._convert_to_tensor(other, dtype=value.dtype))

    def __floordiv__(self, other):
        value = self.value
        return value.__floordiv__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rfloordiv__(self, other):
        value = self.value
        return value.__rfloordiv__(self._convert_to_tensor(other, dtype=value.dtype))

    def __divmod__(self, other):
        value = self.value
        return value.__divmod__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rdivmod__(self, other):
        value = self.value
        return value.__rdivmod__(self._convert_to_tensor(other, dtype=value.dtype))

    def __mod__(self, other):
        value = self.value
        return value.__mod__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rmod__(self, other):
        value = self.value
        return value.__rmod__(self._convert_to_tensor(other, dtype=value.dtype))

    def __pow__(self, other):
        value = self.value
        return value.__pow__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rpow__(self, other):
        value = self.value
        return value.__rpow__(self._convert_to_tensor(other, dtype=value.dtype))

    def __matmul__(self, other):
        value = self.value
        return value.__matmul__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rmatmul__(self, other):
        value = self.value
        return value.__rmatmul__(self._convert_to_tensor(other, dtype=value.dtype))

    def __and__(self, other):
        value = self.value
        return value.__and__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rand__(self, other):
        value = self.value
        return value.__rand__(self._convert_to_tensor(other, dtype=value.dtype))

    def __or__(self, other):
        value = self.value
        return value.__or__(self._convert_to_tensor(other, dtype=value.dtype))

    def __ror__(self, other):
        value = self.value
        return value.__ror__(self._convert_to_tensor(other, dtype=value.dtype))

    def __xor__(self, other):
        value = self.value
        return value.__xor__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rxor__(self, other):
        value = self.value
        return value.__rxor__(self._convert_to_tensor(other, dtype=value.dtype))

    def __lshift__(self, other):
        value = self.value
        return value.__lshift__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rlshift__(self, other):
        value = self.value
        return value.__rlshift__(self._convert_to_tensor(other, dtype=self.dtype))

    def __rshift__(self, other):
        value = self.value
        return value.__rshift__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rrshift__(self, other):
        value = self.value
        return value.__rrshift__(self._convert_to_tensor(other, dtype=self.dtype))

    def __round__(self, ndigits=None):
        value = self.value
        return value.__round__(ndigits)


def is_tensor(x):
    return torch.is_tensor(x)


def convert_to_numpy(x):

    def transform(x):
        if is_tensor(x):
            if x.requires_grad:
                x = x.detach()
            if x.is_cuda or x.is_mps:
                x = x.cpu()
        return np.array(x)
    if isinstance(x, (list, tuple)):
        return np.array([transform(e) for e in x])
    return transform(x)

