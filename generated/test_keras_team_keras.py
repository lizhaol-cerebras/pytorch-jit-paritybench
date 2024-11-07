
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


import tensorflow as tf


import re


import warnings


import functools


import itertools


from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice


from tensorflow.python.eager import context as tf_context


from typing import Iterator


from typing import Tuple


import math


import torch.nn.functional as tnn


import torch._dynamo as dynamo


import logging


import collections


import random


import tensorflow.summary as summary


from tensorflow.compat.v1 import SummaryMetadata


from tensorflow.core.util import event_pb2


from tensorflow.python.lib.io import tf_record


import scipy.signal


from numpy.lib.stride_tricks import as_strided


import string


import inspect


from functools import wraps


from tensorflow import data as tf_data


import typing


import scipy.ndimage


from itertools import combinations


import types


import pandas


import scipy


import copy


from torch.utils.data import Dataset as TorchDataset


num_classes = 10


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
        self.model = keras.Sequential([layers.Input(shape=(28, 28, 1)), layers.Conv2D(32, kernel_size=(3, 3), activation='relu'), layers.MaxPooling2D(pool_size=(2, 2)), layers.Conv2D(64, kernel_size=(3, 3), activation='relu'), layers.MaxPooling2D(pool_size=(2, 2)), layers.Flatten(), layers.Dropout(0.5), layers.Dense(num_classes)])

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


_BACKEND = 'tensorflow'


class KerasHistory(collections.namedtuple('KerasHistory', ['operation', 'node_index', 'tensor_index'])):
    """Tracks the Operation call that created a Tensor.

    During construction of Keras Functions, this metadata is added to
    each Tensor produced as the output of an Operation.
    This allows Keras to track how each Tensor was produced, and
    this information is later retraced by the `Function` class to
    reconstruct the Operations graph.

    Attributes:
      operation: The Operation instance that produced the Tensor.
      node_index: The specific call to the Operation that produced this Tensor.
        Operations can be called multiple times in order to share weights. A new
        node is created every time an Operation is called. The corresponding
        node that represents the call event that produced the Tensor can be
        found at `op._inbound_nodes[node_index]`.
      tensor_index: The output index for this Tensor.
        Always zero if the Operation that produced this Tensor
        only has one output. Nested structures of
        Tensors are deterministically assigned an index via `nest.flatten`.
    """
    __slots__ = ()


class SymbolicArguments:

    def __init__(self, *args, **kwargs):
        self.args = tree.map_structure(lambda x: x, args)
        self.kwargs = tree.map_structure(lambda x: x, kwargs)
        self._flat_arguments = tree.flatten((self.args, self.kwargs))
        if not self.kwargs and len(self.args) == 1 and isinstance(self.args[0], KerasTensor):
            self._single_positional_tensor = self.args[0]
        else:
            self._single_positional_tensor = None
        self.keras_tensors = []
        for arg in self._flat_arguments:
            if isinstance(arg, KerasTensor):
                self.keras_tensors.append(arg)

    def convert(self, conversion_fn):
        args = tree.map_structure(conversion_fn, self.args)
        kwargs = tree.map_structure(conversion_fn, self.kwargs)
        return args, kwargs

    def fill_in(self, tensor_dict):
        """Maps KerasTensors to computed values using `tensor_dict`.

        `tensor_dict` maps `KerasTensor` instances to their current values.
        """
        if self._single_positional_tensor is not None:
            return (tensor_dict[id(self._single_positional_tensor)],), {}

        def switch_fn(x):
            if isinstance(x, KerasTensor):
                return tensor_dict.get(id(x), None)
            return x
        return self.convert(switch_fn)

