
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


import torch.nn.functional as F


from torch.nn import Linear


import time


from torch import tensor


from torch.optim import Adam


import warnings


from collections import defaultdict


from math import ceil


from torch.nn import BatchNorm1d as BN


from torch.nn import ReLU


from torch.nn import Sequential


from itertools import product


from torch.nn import Conv1d


from sklearn.model_selection import StratifiedKFold


from time import perf_counter


from typing import Any


from typing import Callable


from typing import Tuple


from typing import Union


import torch.distributed as dist


from torch.nn.parallel import DistributedDataParallel as DDP


import torch.multiprocessing as mp


from torch.nn import Linear as Lin


from torch.nn import Sequential as Seq


from torch.nn import Parameter


from torch.nn import Parameter as Param


import matplotlib.pyplot as plt


from sklearn.cluster import KMeans


from sklearn.manifold import TSNE


from sklearn.metrics.cluster import completeness_score


from sklearn.metrics.cluster import homogeneity_score


from sklearn.metrics.cluster import v_measure_score


from math import sqrt


from sklearn.metrics import f1_score


from torch.nn import ModuleList


import copy


from typing import Optional


from torch import Tensor


from torch.nn import BatchNorm1d


import torch.distributed


from torch.nn.parallel import DistributedDataParallel


import pandas as pd


from torch.optim.lr_scheduler import ReduceLROnPlateau


import numpy as np


from sklearn.metrics import roc_auc_score


from sklearn.model_selection import train_test_split


from typing import Dict


from torch.nn import Embedding


from sklearn.linear_model import LogisticRegression


from sklearn.linear_model import SGDClassifier


from sklearn.multioutput import MultiOutputClassifier


from typing import List


from torch import nn


from torch.nn import BatchNorm1d as BatchNorm


import torch.optim as optim


import math


import re


from torch.nn.utils import clip_grad_norm_


from torch.nn import LeakyReLU


from torch.utils.data.distributed import DistributedSampler


from torch.optim.lr_scheduler import StepLR


from torch.utils.tensorboard import SummaryWriter


from torch.nn import LayerNorm


from torch.nn import GRU


from itertools import chain


from scipy.sparse.csgraph import shortest_path


from torch.nn import BCEWithLogitsLoss


from torch.nn import MaxPool1d


from torch.utils.data import DataLoader


from sklearn.metrics import average_precision_score


from sklearn.exceptions import ConvergenceWarning


from sklearn.metrics import accuracy_score


from sklearn.svm import LinearSVC


from functools import partial


import torch.nn as nn


from typing import Iterator


from torch.optim import Adagrad


from torch.optim import Optimizer


import logging


import functools


from collections import namedtuple


import random


from time import sleep


from torch.nn import Linear as PTLinear


from torch.nn.parameter import UninitializedParameter


from torch import Tensor as T


from torch.optim.lr_scheduler import ConstantLR


from torch.optim.lr_scheduler import LambdaLR


from collections import OrderedDict


import torch.fx


from torch.nn import Dropout


import inspect


from typing import Final


from typing import Set


import numbers


from typing import Generator


from math import pi as PI


import itertools


import typing


from typing import Iterable


from collections.abc import Sequence


from typing import Type


from collections.abc import Mapping


from typing import TypeVar


from typing import NamedTuple


from typing import overload


from abc import ABC


from abc import abstractmethod


from functools import cached_property


from typing import Sequence


import torch.utils.data


from enum import Enum


from typing import Mapping


from typing import MutableSequence


from collections.abc import MutableMapping


from collections import Counter


from torch.utils.data import ConcatDataset


from torch.utils.data import Subset


from typing import Literal


from typing import no_type_check


from torch.distributed import rpc


from typing import get_args


import torch.utils._pytree as pytree


from torch.nn.parameter import Parameter


from torch.optim import SGD


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.optim.lr_scheduler import MultiStepLR


from uuid import uuid4


from torch._tensor_str import PRINT_OPTS


from torch._tensor_str import _tensor_str


from itertools import repeat


from torch.utils.data.dataloader import _BaseDataLoaderIter


from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter


from torch.utils.data.dataloader import default_collate


from math import log2


from torch.nn import GRUCell


from torch.nn import LSTM


from torch.nn import MultiheadAttention


from math import log


from torch.nn import InstanceNorm1d


from inspect import Parameter


from typing import OrderedDict


from torch.utils.hooks import RemovableHandle


from torch.nn import Sigmoid


from torch.nn import ELU


from torch.nn import Linear as L


from torch.nn import Sequential as S


from torch.nn import Module


from torch.nn import ModuleDict


from torch.utils.checkpoint import checkpoint


from torch.nn.modules.loss import _Loss


from torch.nn import Identity


from torch.autograd import grad


from torch.nn.modules.instancenorm import _InstanceNorm


from torch import LongTensor


from torch.nn import Conv2d


from torch.nn import KLDivLoss


from torch.jit import ScriptModule


from collections import deque


from torch.autograd.profiler import EventList


from torch.profiler import ProfilerActivity


from torch.profiler import profile


import torch.profiler as torch_profiler


from torch.multiprocessing import Manager


from torch.multiprocessing import Queue


from typing import cast


from torch.utils.dlpack import from_dlpack


from torch.utils.dlpack import to_dlpack


from torch.nn import ParameterDict


from copy import copy


def get_options(options: 'Options') ->List[str]:
    if options is None:
        options = list(__experimental_flag__.keys())
    if isinstance(options, str):
        options = [options]
    return options


def is_experimental_mode_enabled(options: 'Options'=None) ->bool:
    """Returns :obj:`True` if the experimental mode is enabled. See
    :class:`torch_geometric.experimental_mode` for a list of (optional)
    options.
    """
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return False
    options = get_options(options)
    return all([__experimental_flag__[option] for option in options])


def disable_dynamic_shapes(required_args: 'List[str]') ->Callable:
    """A decorator that disables the usage of dynamic shapes for the given
    arguments, i.e., it will raise an error in case :obj:`required_args` are
    not passed and needs to be automatically inferred.
    """

    def decorator(func: 'Callable') ->Callable:
        spec = inspect.getfullargspec(func)
        required_args_pos: 'Dict[str, int]' = {}
        for arg_name in required_args:
            if arg_name not in spec.args:
                raise ValueError(f"The function '{func}' does not have a '{arg_name}' argument")
            required_args_pos[arg_name] = spec.args.index(arg_name)
        num_args = len(spec.args)
        num_default_args = 0 if spec.defaults is None else len(spec.defaults)
        num_positional_args = num_args - num_default_args

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) ->Any:
            if not is_experimental_mode_enabled('disable_dynamic_shapes'):
                return func(*args, **kwargs)
            for required_arg in required_args:
                index = required_args_pos[required_arg]
                value: 'Optional[Any]' = None
                if index < len(args):
                    value = args[index]
                elif required_arg in kwargs:
                    value = kwargs[required_arg]
                elif num_default_args > 0:
                    assert spec.defaults is not None
                    value = spec.defaults[index - num_positional_args]
                if value is None:
                    raise ValueError(f"Dynamic shapes disabled. Argument '{required_arg}' needs to be set")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def expand_left(ptr: 'Tensor', dim: 'int', dims: 'int') ->Tensor:
    for _ in range(dims + dim if dim < 0 else dim):
        ptr = ptr.unsqueeze(0)
    return ptr


def _torch_segment(src: 'Tensor', ptr: 'Tensor', reduce: 'str'='sum') ->Tensor:
    if not torch_geometric.typing.WITH_PT20:
        raise ImportError("'segment' requires the 'torch-scatter' package")
    if ptr.dim() > 1:
        raise ImportError("'segment' in an arbitrary dimension requires the 'torch-scatter' package")
    if reduce == 'min' or reduce == 'max':
        reduce = f'a{reduce}'
    initial = 0 if reduce == 'mean' else None
    out = torch._segment_reduce(src, reduce, offsets=ptr, initial=initial)
    if reduce == 'amin' or reduce == 'amax':
        out = torch.where(out.isinf(), 0, out)
    return out


def is_compiling() ->bool:
    """Returns :obj:`True` in case :pytorch:`PyTorch` is compiling via
    :meth:`torch.compile`.
    """
    if torch_geometric.typing.WITH_PT23:
        return torch.compiler.is_compiling()
    if torch_geometric.typing.WITH_PT21:
        return torch._dynamo.is_compiling()
    return False


def segment(src: 'Tensor', ptr: 'Tensor', reduce: 'str'='sum') ->Tensor:
    """Reduces all values in the first dimension of the :obj:`src` tensor
    within the ranges specified in the :obj:`ptr`. See the `documentation
    <https://pytorch-scatter.readthedocs.io/en/latest/functions/
    segment_csr.html>`__ of the :obj:`torch_scatter` package for more
    information.

    Args:
        src (torch.Tensor): The source tensor.
        ptr (torch.Tensor): A monotonically increasing pointer tensor that
            refers to the boundaries of segments such that :obj:`ptr[0] = 0`
            and :obj:`ptr[-1] = src.size(0)`.
        reduce (str, optional): The reduce operation (:obj:`"sum"`,
            :obj:`"mean"`, :obj:`"min"` or :obj:`"max"`).
            (default: :obj:`"sum"`)
    """
    if not torch_geometric.typing.WITH_TORCH_SCATTER or is_compiling():
        return _torch_segment(src, ptr, reduce)
    if ptr.dim() == 1 and torch_geometric.typing.WITH_PT20 and src.is_cuda and reduce == 'mean':
        return _torch_segment(src, ptr, reduce)
    return torch_scatter.segment_csr(src, ptr, reduce=reduce)


class Aggregation(torch.nn.Module):
    """An abstract base class for implementing custom aggregations.

    Aggregation can be either performed via an :obj:`index` vector, which
    defines the mapping from input elements to their location in the output:

    |

    .. image:: https://raw.githubusercontent.com/rusty1s/pytorch_scatter/
            master/docs/source/_figures/add.svg?sanitize=true
        :align: center
        :width: 400px

    |

    Notably, :obj:`index` does not have to be sorted (for most aggregation
    operators):

    .. code-block:: python

       # Feature matrix holding 10 elements with 64 features each:
       x = torch.randn(10, 64)

       # Assign each element to one of three sets:
       index = torch.tensor([0, 0, 1, 0, 2, 0, 2, 1, 0, 2])

       output = aggr(x, index)  #  Output shape: [3, 64]

    Alternatively, aggregation can be achieved via a "compressed" index vector
    called :obj:`ptr`. Here, elements within the same set need to be grouped
    together in the input, and :obj:`ptr` defines their boundaries:

    .. code-block:: python

       # Feature matrix holding 10 elements with 64 features each:
       x = torch.randn(10, 64)

       # Define the boundary indices for three sets:
       ptr = torch.tensor([0, 4, 7, 10])

       output = aggr(x, ptr=ptr)  #  Output shape: [3, 64]

    Note that at least one of :obj:`index` or :obj:`ptr` must be defined.

    Shapes:
        - **input:**
          node features :math:`(*, |\\mathcal{V}|, F_{in})` or edge features
          :math:`(*, |\\mathcal{E}|, F_{in})`,
          index vector :math:`(|\\mathcal{V}|)` or :math:`(|\\mathcal{E}|)`,
        - **output:** graph features :math:`(*, |\\mathcal{G}|, F_{out})` or
          node features :math:`(*, |\\mathcal{V}|, F_{out})`
    """

    def __init__(self) ->None:
        super().__init__()
        self._deterministic: 'Final[bool]' = torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled()

    def forward(self, x: 'Tensor', index: 'Optional[Tensor]'=None, ptr: 'Optional[Tensor]'=None, dim_size: 'Optional[int]'=None, dim: 'int'=-2, max_num_elements: 'Optional[int]'=None) ->Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): The source tensor.
            index (torch.Tensor, optional): The indices of elements for
                applying the aggregation.
                One of :obj:`index` or :obj:`ptr` must be defined.
                (default: :obj:`None`)
            ptr (torch.Tensor, optional): If given, computes the aggregation
                based on sorted inputs in CSR representation.
                One of :obj:`index` or :obj:`ptr` must be defined.
                (default: :obj:`None`)
            dim_size (int, optional): The size of the output tensor at
                dimension :obj:`dim` after aggregation. (default: :obj:`None`)
            dim (int, optional): The dimension in which to aggregate.
                (default: :obj:`-2`)
            max_num_elements: (int, optional): The maximum number of elements
                within a single aggregation group. (default: :obj:`None`)
        """

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""

    @disable_dynamic_shapes(required_args=['dim_size'])
    def __call__(self, x: 'Tensor', index: 'Optional[Tensor]'=None, ptr: 'Optional[Tensor]'=None, dim_size: 'Optional[int]'=None, dim: 'int'=-2, **kwargs) ->Tensor:
        if dim >= x.dim() or dim < -x.dim():
            raise ValueError(f"Encountered invalid dimension '{dim}' of source tensor with {x.dim()} dimensions")
        if index is None and ptr is None:
            index = x.new_zeros(x.size(dim), dtype=torch.long)
        if ptr is not None:
            if dim_size is None:
                dim_size = ptr.numel() - 1
            elif dim_size != ptr.numel() - 1:
                raise ValueError(f"Encountered invalid 'dim_size' (got '{dim_size}' but expected '{ptr.numel() - 1}')")
        if index is not None and dim_size is None:
            dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
        try:
            return super().__call__(x, index=index, ptr=ptr, dim_size=dim_size, dim=dim, **kwargs)
        except (IndexError, RuntimeError) as e:
            if index is not None:
                if index.numel() > 0 and dim_size <= int(index.max()):
                    raise ValueError(f"Encountered invalid 'dim_size' (got '{dim_size}' but expected >= '{int(index.max()) + 1}')")
            raise e

    def __repr__(self) ->str:
        return f'{self.__class__.__name__}()'

    def assert_index_present(self, index: 'Optional[Tensor]'):
        if index is None:
            raise NotImplementedError("Aggregation requires 'index' to be specified")

    def assert_sorted_index(self, index: 'Optional[Tensor]'):
        if index is not None and not torch.all(index[:-1] <= index[1:]):
            raise ValueError("Can not perform aggregation since the 'index' tensor is not sorted. Specifically, if you use this aggregation as part of 'MessagePassing`, ensure that 'edge_index' is sorted by destination nodes, e.g., by calling `data.sort(sort_by_row=False)`")

    def assert_two_dimensional_input(self, x: 'Tensor', dim: 'int'):
        if x.dim() != 2:
            raise ValueError(f"Aggregation requires two-dimensional inputs (got '{x.dim()}')")
        if dim not in [-2, 0]:
            raise ValueError(f"Aggregation needs to perform aggregation in first dimension (got '{dim}')")

    def reduce(self, x: 'Tensor', index: 'Optional[Tensor]'=None, ptr: 'Optional[Tensor]'=None, dim_size: 'Optional[int]'=None, dim: 'int'=-2, reduce: 'str'='sum') ->Tensor:
        if ptr is not None:
            if index is None or self._deterministic:
                ptr = expand_left(ptr, dim, dims=x.dim())
                return segment(x, ptr, reduce=reduce)
        if index is None:
            raise RuntimeError("Aggregation requires 'index' to be specified")
        return scatter(x, index, dim, dim_size, reduce)

    def to_dense_batch(self, x: 'Tensor', index: 'Optional[Tensor]'=None, ptr: 'Optional[Tensor]'=None, dim_size: 'Optional[int]'=None, dim: 'int'=-2, fill_value: 'float'=0.0, max_num_elements: 'Optional[int]'=None) ->Tuple[Tensor, Tensor]:
        self.assert_index_present(index)
        self.assert_sorted_index(index)
        self.assert_two_dimensional_input(x, dim)
        return to_dense_batch(x, index, batch_size=dim_size, fill_value=fill_value, max_num_nodes=max_num_elements)


class EdgeIndex(NamedTuple):
    edge_index: 'Tensor'
    e_id: 'Optional[Tensor]'
    size: 'Tuple[int, int]'

    def to(self, *args, **kwargs):
        edge_index = self.edge_index
        e_id = self.e_id if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)


FUSE_AGGRS = {'add', 'sum', 'mean', 'min', 'max'}

