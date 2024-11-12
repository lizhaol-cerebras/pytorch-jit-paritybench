
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


import re


import torch


import typing as _typing


import pandas as pd


import numpy as np


import scipy.io


import itertools


import random


import torch.utils.data


from sklearn.model_selection import StratifiedKFold


from sklearn.model_selection import KFold


import math


from sklearn import preprocessing


from sklearn.metrics.pairwise import cosine_similarity as cos_sim


import copy


import time


from torch.utils.data import Dataset


from functools import wraps


import torch.nn.functional as F


from sklearn.ensemble import GradientBoostingClassifier


from sklearn.linear_model import LogisticRegression


from sklearn.metrics.pairwise import cosine_similarity


from scipy.sparse import csr_matrix


import scipy.sparse as ssp


import scipy.sparse.linalg


import logging


import torch.nn.functional


from typing import Optional


from typing import Tuple


from sklearn.model_selection import train_test_split


from copy import deepcopy


from typing import Sequence


from typing import Union


from numbers import Real


import torch.nn as nn


from torch.nn import Linear


from torch.nn import ReLU


from torch.nn import Sequential


from torch.nn import LeakyReLU


from torch.nn import Tanh


from torch.nn import ELU


from torch.nn import BatchNorm1d


from functools import partial


import torch.optim as optim


from sklearn.metrics.pairwise import euclidean_distances


from scipy.sparse import lil_matrix


from sklearn.preprocessing import normalize


import scipy.sparse as sp


from torch.nn import Parameter


from abc import abstractmethod


import torch.optim


from torch.autograd import Variable


from logging import Logger


from numpy.core.fromnumeric import sort


import collections


import typing as _typ


from torch import nn


from torch import Tensor


from torch.nn import Module


from collections import namedtuple


from typing import Callable


import inspect


from uuid import uuid1


from itertools import chain


from inspect import Parameter


from typing import List


from typing import Set


from itertools import product


from collections import OrderedDict


from typing import Dict


from scipy import sparse


import scipy


from sklearn.metrics import jaccard_score


import typing


from torch.optim.lr_scheduler import StepLR


from torch.optim.lr_scheduler import MultiStepLR


from torch.optim.lr_scheduler import ExponentialLR


from torch.optim.lr_scheduler import ReduceLROnPlateau


from typing import Type


import torch.multiprocessing as mp


from typing import Iterable


from sklearn.metrics import f1_score


from itertools import repeat


from typing import Any


import torch.backends.cudnn


from queue import Queue


import numpy


from torch.utils.data import DataLoader


from scipy import io as sio


from sklearn.metrics import roc_auc_score


class _LogSoftmaxDecoder(torch.nn.Module):

    def forward(self, features: '_typing.Sequence[torch.Tensor]', *__args, **__kwargs) ->torch.Tensor:
        return torch.nn.functional.log_softmax(features[-1], dim=1)


class _JKSumPoolDecoder(torch.nn.Module):

    def __init__(self, input_dimensions: '_typing.Sequence[int]', output_dimension: 'int', dropout: 'float', graph_pooling_type: 'str'):
        super(_JKSumPoolDecoder, self).__init__()
        self._linear_transforms: 'torch.nn.ModuleList' = torch.nn.ModuleList()
        for input_dimension in input_dimensions:
            self._linear_transforms.append(torch.nn.Linear(input_dimension, output_dimension))
        self._dropout: 'torch.nn.Dropout' = torch.nn.Dropout(dropout)
        if not isinstance(graph_pooling_type, str):
            raise TypeError
        elif graph_pooling_type.lower() == 'sum':
            self.__pool = SumPooling()
        elif graph_pooling_type.lower() == 'mean':
            self.__pool = AvgPooling()
        elif graph_pooling_type.lower() == 'max':
            self.__pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, features: '_typing.Sequence[torch.Tensor]', graph: 'dgl.DGLGraph', *__args, **__kwargs):
        if len(features) != len(self._linear_transforms):
            raise ValueError
        score_over_layer = 0
        for i, feature in enumerate(features):
            score_over_layer += self._dropout(self._linear_transforms[i](self.__pool(graph, feature)))
        return score_over_layer


class _SumPoolMLPDecoder(torch.nn.Module):

    def __init__(self, _final_dimension: 'int', hidden_dimension: 'int', output_dimension: 'int', _act: '_typing.Optional[str]', _dropout: '_typing.Optional[float]', num_graph_features: '_typing.Optional[int]'):
        super(_SumPoolMLPDecoder, self).__init__()
        if isinstance(num_graph_features, int) and num_graph_features > 0:
            _final_dimension += num_graph_features
            self.__num_graph_features: '_typing.Optional[int]' = num_graph_features
        else:
            self.__num_graph_features: '_typing.Optional[int]' = None
        self._fc1: 'torch.nn.Linear' = torch.nn.Linear(_final_dimension, hidden_dimension)
        self._fc2: 'torch.nn.Linear' = torch.nn.Linear(hidden_dimension, output_dimension)
        self._act: '_typing.Optional[str]' = _act
        self._dropout: '_typing.Optional[float]' = _dropout

    def forward(self, features: '_typing.Sequence[torch.Tensor]', data: 'torch_geometric.data.Data', *__args, **__kwargs):
        feature = features[-1]
        feature = global_add_pool(feature, data.batch)
        if isinstance(self.__num_graph_features, int) and self.__num_graph_features > 0:
            if hasattr(data, 'gf') and isinstance(data.gf, torch.Tensor) and data.gf.dim() == 2 and data.gf.size() == (feature.size(0), self.__num_graph_features):
                graph_features: 'torch.Tensor' = data.gf
            else:
                raise ValueError(f"The provided data is expected to contain property 'gf' with {self.__num_graph_features} dimensions as graph feature")
            feature: 'torch.Tensor' = torch.cat([feature, graph_features], dim=-1)
        feature: 'torch.Tensor' = self._fc1(feature)
        feature: 'torch.Tensor' = _utils.activation.activation_func(feature, self._act)
        if isinstance(self._dropout, float) and 0 <= self._dropout <= 1:
            feature: 'torch.Tensor' = torch.nn.functional.dropout(feature, self._dropout, self.training)
        feature: 'torch.Tensor' = self._fc2(feature)
        return torch.nn.functional.log_softmax(feature, dim=-1)


class _TopKPoolDecoder(torch.nn.Module):

    def __init__(self, input_dimensions: '_typing.Iterable[int]', output_dimension: 'int', dropout: 'float'):
        super(_TopKPoolDecoder, self).__init__()
        k: 'int' = min(len(list(input_dimensions)), 3)
        self.__pool: 'SortPooling' = SortPooling(k)
        self.__linear_predictions: 'torch.nn.ModuleList' = torch.nn.ModuleList()
        for layer, dimension in enumerate(input_dimensions):
            self.__linear_predictions.append(torch.nn.Linear(dimension * k, output_dimension))
        self._dropout = torch.nn.Dropout(dropout)

    def forward(self, features: '_typing.Sequence[torch.Tensor]', graph: 'dgl.DGLGraph', *__args, **__kwargs):
        cumulative_result = 0
        for i, h in enumerate(features):
            cumulative_result += self._dropout(self.__linear_predictions[i](self.__pool(graph, h)))
        return cumulative_result


class _DotProductLinkPredictionDecoder(torch.nn.Module):

    def forward(self, features: '_typing.Sequence[torch.Tensor]', graph: 'dgl.DGLGraph', pos_edge: 'torch.Tensor', neg_edge: 'torch.Tensor', **__kwargs):
        z = features[-1]
        edge_index = torch.cat([pos_edge, neg_edge], dim=-1)
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)


class _DiffPoolDecoder(torch.nn.Module):

    def __init__(self, input_dimension: 'int', output_dimension: 'int', _ratio: '_typing.Union[float, int]', _dropout: '_typing.Optional[float]', _act: '_typing.Optional[str]', num_graph_features: '_typing.Optional[int]'):
        super(_DiffPoolDecoder, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.ratio: '_typing.Union[float, int]' = _ratio
        self._act: '_typing.Optional[str]' = _act
        self.dropout: '_typing.Optional[float]' = _dropout
        self.num_graph_features: '_typing.Optional[int]' = num_graph_features
        self.conv1 = GraphConv(self.input_dimension, 128)
        self.pool1 = TopKPooling(128, ratio=self.ratio)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=self.ratio)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=self.ratio)
        if isinstance(self.num_graph_features, int) and self.num_graph_features > 0:
            self.lin1 = torch.nn.Linear(256 + self.num_graph_features, 128)
        else:
            self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, self.output_dimension)

    def forward(self, features: '_typing.Sequence[torch.Tensor]', data: 'torch_geometric.data.Data', *__args, **__kwargs):
        x: 'torch.Tensor' = features[-1]
        edge_index: 'torch.LongTensor' = data.edge_index
        batch = data.batch
        if self.num_graph_features is not None and isinstance(self.num_graph_features, int) and self.num_graph_features > 0:
            if not (hasattr(data, 'gf') and isinstance(data.gf, torch.Tensor) and data.gf.size() == (x.size(0), self.num_graph_features)):
                raise ValueError(f"The provided data is expected to contain property 'gf' with {self.num_graph_features} dimensions as graph feature")
        x = torch.nn.functional.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        x = torch.nn.functional.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        x = torch.nn.functional.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        x = x1 + x2 + x3
        if isinstance(self.num_graph_features, int) and self.num_graph_features > 0:
            x = torch.cat([x, data.gf], dim=-1)
        x = self.lin1(x)
        x = _utils.activation.activation_func(x, self._act)
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = _utils.activation.activation_func(x, self._act)
        x = torch.nn.functional.log_softmax(self.lin3(x), dim=-1)
        return x


class _DotProductLinkPredictonDecoder(torch.nn.Module):

    def forward(self, features: '_typing.Sequence[torch.Tensor]', graph: 'torch_geometric.data.Data', pos_edge: 'torch.Tensor', neg_edge: 'torch.Tensor', **__kwargs):
        z = features[-1]
        edge_index = torch.cat([pos_edge, neg_edge], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits


class _ClassificationModel(torch.nn.Module):

    def __init__(self):
        super(_ClassificationModel, self).__init__()

    def cls_encode(self, data) ->torch.Tensor:
        raise NotImplementedError

    def cls_decode(self, x: 'torch.Tensor') ->torch.Tensor:
        raise NotImplementedError

    def cls_forward(self, data) ->torch.Tensor:
        return self.cls_decode(self.cls_encode(data))


class ClassificationSupportedSequentialModel(_ClassificationModel):

    def __init__(self):
        super(ClassificationSupportedSequentialModel, self).__init__()

    @property
    def sequential_encoding_layers(self) ->torch.nn.ModuleList:
        raise NotImplementedError

    def cls_encode(self, data) ->torch.Tensor:
        raise NotImplementedError

    def cls_decode(self, x: 'torch.Tensor') ->torch.Tensor:
        raise NotImplementedError


__size_error_msg__ = 'All tensors which should get mapped to the same source or target nodes must be of same size in dimension 0.'


def scatter_(name, src, index, dim_size=None):
    """Aggregates all values from the :attr:`src` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.
    If multiple indices reference the same location, their contributions
    are aggregated according to :attr:`name` (either :obj:`"add"`,
    :obj:`"mean"` or :obj:`"max"`).

    Args:
        name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"max"`).
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim_size (int, optional): Automatically create output tensor with size
            :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
            minimal sized output tensor is returned. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    assert name in ['add', 'mean', 'max']
    op = getattr(torch_scatter, 'scatter_{}'.format(name))
    fill_value = -1000000000.0 if name == 'max' else 0
    out = op(src, index, 0, None, dim_size)
    if isinstance(out, tuple):
        out = out[0]
    if name == 'max':
        out[out == fill_value] = 0
    return out


special_args = ['edge_index', 'edge_index_i', 'edge_index_j', 'size', 'size_i', 'size_j']


class MessagePassing(torch.nn.Module):

    def __init__(self, aggr='add', flow='source_to_target'):
        super(MessagePassing, self).__init__()
        self.aggr = aggr
        assert self.aggr in ['add', 'mean', 'max']
        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']
        self.__message_args__ = getargspec(self.message)[0][1:]
        self.__special_args__ = [(i, arg) for i, arg in enumerate(self.__message_args__) if arg in special_args]
        self.__message_args__ = [arg for arg in self.__message_args__ if arg not in special_args]
        self.__update_args__ = getargspec(self.update)[0][2:]

    def propagate(self, edge_index, size=None, **kwargs):
        """The initial call to start propagating messages.

        Args:
            edge_index (Tensor): The indices of a general (sparse) assignment
                matrix with shape :obj:`[N, M]` (can be directed or
                undirected).
            size (list or tuple, optional): The size :obj:`[N, M]` of the
                assignment matrix. If set to :obj:`None`, the size is tried to
                get automatically inferrred. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct messages
                and to update node embeddings.
        """
        size = [None, None] if size is None else list(size)
        assert len(size) == 2
        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        ij = {'_i': i, '_j': j}
        message_args = []
        for arg in self.__message_args__:
            if arg[-2:] in ij.keys():
                tmp = kwargs.get(arg[:-2], None)
                if tmp is None:
                    message_args.append(tmp)
                else:
                    idx = ij[arg[-2:]]
                    if isinstance(tmp, tuple) or isinstance(tmp, list):
                        assert len(tmp) == 2
                        if tmp[1 - idx] is not None:
                            if size[1 - idx] is None:
                                size[1 - idx] = tmp[1 - idx].size(0)
                            if size[1 - idx] != tmp[1 - idx].size(0):
                                raise ValueError(__size_error_msg__)
                        tmp = tmp[idx]
                    if size[idx] is None:
                        size[idx] = tmp.size(0)
                    if size[idx] != tmp.size(0):
                        raise ValueError(__size_error_msg__)
                    tmp = torch.index_select(tmp, 0, edge_index[idx])
                    message_args.append(tmp)
            else:
                message_args.append(kwargs.get(arg, None))
        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]
        kwargs['edge_index'] = edge_index
        kwargs['size'] = size
        for idx, arg in self.__special_args__:
            if arg[-2:] in ij.keys():
                message_args.insert(idx, kwargs[arg[:-2]][ij[arg[-2:]]])
            else:
                message_args.insert(idx, kwargs[arg])
        update_args = [kwargs[arg] for arg in self.__update_args__]
        out = self.message(*message_args)
        if self.aggr in ['add', 'mean', 'max']:
            out = scatter_(self.aggr, out, edge_index[i], dim_size=size[i])
        else:
            pass
        out = self.update(out, *update_args)
        return out

    def message(self, x_j):
        """Constructs messages in analogy to :math:`\\phi_{\\mathbf{\\Theta}}`
        for each edge in :math:`(i,j) \\in \\mathcal{E}`.
        Can take any argument which was initially passed to :meth:`propagate`.
        In addition, features can be lifted to the source node :math:`i` and
        target node :math:`j` by appending :obj:`_i` or :obj:`_j` to the
        variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`."""
        return x_j

    def update(self, aggr_out):
        """Updates node embeddings in analogy to
        :math:`\\gamma_{\\mathbf{\\Theta}}` for each node
        :math:`i \\in \\mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`."""
        return aggr_out


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def gather_csr(src: 'torch.Tensor', indptr: 'torch.Tensor', out: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
    return torch.ops.torch_scatter.gather_csr(src, indptr, out)


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))


def scatter_max(src: 'torch.Tensor', index: 'torch.Tensor', dim: 'int'=-1, out: 'Optional[torch.Tensor]'=None, dim_size: 'Optional[int]'=None) ->Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.scatter_max(src, index, dim, out, dim_size)


def broadcast(src: 'torch.Tensor', other: 'torch.Tensor', dim: 'int'):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


def scatter_sum(src: 'torch.Tensor', index: 'torch.Tensor', dim: 'int'=-1, out: 'Optional[torch.Tensor]'=None, dim_size: 'Optional[int]'=None) ->torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter_mean(src: 'torch.Tensor', index: 'torch.Tensor', dim: 'int'=-1, out: 'Optional[torch.Tensor]'=None, dim_size: 'Optional[int]'=None) ->torch.Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)
    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1
    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.floor_divide_(count)
    return out


def scatter_min(src: 'torch.Tensor', index: 'torch.Tensor', dim: 'int'=-1, out: 'Optional[torch.Tensor]'=None, dim_size: 'Optional[int]'=None) ->Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.scatter_min(src, index, dim, out, dim_size)


def scatter_mul(src: 'torch.Tensor', index: 'torch.Tensor', dim: 'int'=-1, out: 'Optional[torch.Tensor]'=None, dim_size: 'Optional[int]'=None) ->torch.Tensor:
    return torch.ops.torch_scatter.scatter_mul(src, index, dim, out, dim_size)


def scatter(src: 'torch.Tensor', index: 'torch.Tensor', dim: 'int'=-1, out: 'Optional[torch.Tensor]'=None, dim_size: 'Optional[int]'=None, reduce: 'str'='sum') ->torch.Tensor:
    """
    |

    .. image:: https://raw.githubusercontent.com/rusty1s/pytorch_scatter/
            master/docs/source/_figures/add.svg?sanitize=true
        :align: center
        :width: 400px

    |

    Reduces all values from the :attr:`src` tensor into :attr:`out` at the
    indices specified in the :attr:`index` tensor along a given axis
    :attr:`dim`.
    For each value in :attr:`src`, its output index is specified by its index
    in :attr:`src` for dimensions outside of :attr:`dim` and by the
    corresponding value in :attr:`index` for dimension :attr:`dim`.
    The applied reduction is defined via the :attr:`reduce` argument.

    Formally, if :attr:`src` and :attr:`index` are :math:`n`-dimensional
    tensors with size :math:`(x_0, ..., x_{i-1}, x_i, x_{i+1}, ..., x_{n-1})`
    and :attr:`dim` = `i`, then :attr:`out` must be an :math:`n`-dimensional
    tensor with size :math:`(x_0, ..., x_{i-1}, y, x_{i+1}, ..., x_{n-1})`.
    Moreover, the values of :attr:`index` must be between :math:`0` and
    :math:`y - 1`, although no specific ordering of indices is required.
    The :attr:`index` tensor supports broadcasting in case its dimensions do
    not match with :attr:`src`.

    For one-dimensional tensors with :obj:`reduce="sum"`, the operation
    computes

    .. math::
        \\mathrm{out}_i = \\mathrm{out}_i + \\sum_j~\\mathrm{src}_j

    where :math:`\\sum_j` is over :math:`j` such that
    :math:`\\mathrm{index}_j = i`.

    .. note::

        This operation is implemented via atomic operations on the GPU and is
        therefore **non-deterministic** since the order of parallel operations
        to the same value is undetermined.
        For floating-point variables, this results in a source of variance in
        the result.

    :param src: The source tensor.
    :param index: The indices of elements to scatter.
    :param dim: The axis along which to index. (default: :obj:`-1`)
    :param out: The destination tensor.
    :param dim_size: If :attr:`out` is not given, automatically create output
        with size :attr:`dim_size` at dimension :attr:`dim`.
        If :attr:`dim_size` is not given, a minimal sized output tensor
        according to :obj:`index.max() + 1` is returned.
    :param reduce: The reduce operation (:obj:`"sum"`, :obj:`"mul"`,
        :obj:`"mean"`, :obj:`"min"` or :obj:`"max"`). (default: :obj:`"sum"`)

    :rtype: :class:`Tensor`

    .. code-block:: python

        from torch_scatter import scatter

        src = torch.randn(10, 6, 64)
        index = torch.tensor([0, 1, 0, 1, 2, 1])

        # Broadcasting in the first and last dim.
        out = scatter(src, index, dim=1, reduce="sum")

        print(out.size())

    .. code-block::

        torch.Size([10, 3, 64])
    """
    if reduce == 'sum' or reduce == 'add':
        return scatter_sum(src, index, dim, out, dim_size)
    if reduce == 'mul':
        return scatter_mul(src, index, dim, out, dim_size)
    elif reduce == 'mean':
        return scatter_mean(src, index, dim, out, dim_size)
    elif reduce == 'min':
        return scatter_min(src, index, dim, out, dim_size)[0]
    elif reduce == 'max':
        return scatter_max(src, index, dim, out, dim_size)[0]
    else:
        raise ValueError


def segment_max_csr(src: 'torch.Tensor', indptr: 'torch.Tensor', out: 'Optional[torch.Tensor]'=None) ->Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.segment_max_csr(src, indptr, out)


def segment_mean_csr(src: 'torch.Tensor', indptr: 'torch.Tensor', out: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
    return torch.ops.torch_scatter.segment_mean_csr(src, indptr, out)


def segment_min_csr(src: 'torch.Tensor', indptr: 'torch.Tensor', out: 'Optional[torch.Tensor]'=None) ->Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.segment_min_csr(src, indptr, out)


def segment_sum_csr(src: 'torch.Tensor', indptr: 'torch.Tensor', out: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
    return torch.ops.torch_scatter.segment_sum_csr(src, indptr, out)


def segment_csr(src: 'torch.Tensor', indptr: 'torch.Tensor', out: 'Optional[torch.Tensor]'=None, reduce: 'str'='sum') ->torch.Tensor:
    """
    Reduces all values from the :attr:`src` tensor into :attr:`out` within the
    ranges specified in the :attr:`indptr` tensor along the last dimension of
    :attr:`indptr`.
    For each value in :attr:`src`, its output index is specified by its index
    in :attr:`src` for dimensions outside of :obj:`indptr.dim() - 1` and by the
    corresponding range index in :attr:`indptr` for dimension
    :obj:`indptr.dim() - 1`.
    The applied reduction is defined via the :attr:`reduce` argument.
    Formally, if :attr:`src` and :attr:`indptr` are :math:`n`-dimensional and
    :math:`m`-dimensional tensors with
    size :math:`(x_0, ..., x_{m-1}, x_m, x_{m+1}, ..., x_{n-1})` and
    :math:`(x_0, ..., x_{m-2}, y)`, respectively, then :attr:`out` must be an
    :math:`n`-dimensional tensor with size
    :math:`(x_0, ..., x_{m-2}, y - 1, x_{m}, ..., x_{n-1})`.
    Moreover, the values of :attr:`indptr` must be between :math:`0` and
    :math:`x_m` in ascending order.
    The :attr:`indptr` tensor supports broadcasting in case its dimensions do
    not match with :attr:`src`.
    For one-dimensional tensors with :obj:`reduce="sum"`, the operation
    computes
    .. math::
        \\mathrm{out}_i =
        \\sum_{j = \\mathrm{indptr}[i]}^{\\mathrm{indptr}[i+1]-1}~\\mathrm{src}_j.
    Due to the use of index pointers, :meth:`segment_csr` is the fastest
    method to apply for grouped reductions.
    .. note::
        In contrast to :meth:`scatter()` and :meth:`segment_coo`, this
        operation is **fully-deterministic**.
    :param src: The source tensor.
    :param indptr: The index pointers between elements to segment.
        The number of dimensions of :attr:`index` needs to be less than or
        equal to :attr:`src`.
    :param out: The destination tensor.
    :param reduce: The reduce operation (:obj:`"sum"`, :obj:`"mean"`,
        :obj:`"min"` or :obj:`"max"`). (default: :obj:`"sum"`)
    :rtype: :class:`Tensor`
    .. code-block:: python
        from torch_scatter import segment_csr
        src = torch.randn(10, 6, 64)
        indptr = torch.tensor([0, 2, 5, 6])
        indptr = indptr.view(1, -1)  # Broadcasting in the first and last dim.
        out = segment_csr(src, indptr, reduce="sum")
        print(out.size())
    .. code-block::
        torch.Size([10, 3, 64])
    """
    if reduce == 'sum' or reduce == 'add':
        return segment_sum_csr(src, indptr, out)
    elif reduce == 'mean':
        return segment_mean_csr(src, indptr, out)
    elif reduce == 'min':
        return segment_min_csr(src, indptr, out)[0]
    elif reduce == 'max':
        return segment_max_csr(src, indptr, out)[0]
    else:
        raise ValueError


def softmax(src: 'Tensor', index: 'Optional[Tensor]'=None, ptr: 'Optional[Tensor]'=None, num_nodes: 'Optional[int]'=None, dim: 'int'=0) ->Tensor:
    if ptr is not None:
        dim = dim + src.dim() if dim < 0 else dim
        size = [1] * dim + [-1]
        ptr = ptr.view(size)
        src_max = gather_csr(segment_csr(src, ptr, reduce='max'), ptr)
        out = (src - src_max).exp()
        out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        src_max = scatter(src, index, dim, dim_size=N, reduce='max')
        src_max = src_max.index_select(dim, index)
        out = (src - src_max).exp()
        out_sum = scatter(out, index, dim, dim_size=N, reduce='sum')
        out_sum = out_sum.index_select(dim, index)
    else:
        raise NotImplementedError
    return out / (out_sum + 1e-16)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class GATConv(MessagePassing):
    """The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \\mathbf{x}^{\\prime}_i = \\alpha_{i,i}\\mathbf{\\Theta}\\mathbf{x}_{i} +
        \\sum_{j \\in \\mathcal{N}(i)} \\alpha_{i,j}\\mathbf{\\Theta}\\mathbf{x}_{j},

    where the attention coefficients :math:`\\alpha_{i,j}` are computed as

    .. math::
        \\alpha_{i,j} =
        \\frac{
        \\exp\\left(\\mathrm{LeakyReLU}\\left(\\mathbf{a}^{\\top}
        [\\mathbf{\\Theta}\\mathbf{x}_i \\, \\Vert \\, \\mathbf{\\Theta}\\mathbf{x}_j]
        \\right)\\right)}
        {\\sum_{k \\in \\mathcal{N}(i) \\cup \\{ i \\}}
        \\exp\\left(\\mathrm{LeakyReLU}\\left(\\mathbf{a}^{\\top}
        [\\mathbf{\\Theta}\\mathbf{x}_i \\, \\Vert \\, \\mathbf{\\Theta}\\mathbf{x}_k]
        \\right)\\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: 'OptTensor'

    def __init__(self, in_channels: 'Union[int, Tuple[int, int]]', out_channels: 'int', heads: 'int'=1, concat: 'bool'=True, negative_slope: 'float'=0.2, dropout: 'float'=0.0, add_self_loops: 'bool'=True, bias: 'bool'=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)
        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def forward(self, x: 'Union[Tensor, OptPairTensor]', edge_index: 'Adj', edge_weight: 'OptTensor'=None, size: 'Size'=None, return_attention_weights=None):
        """

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels
        x_l: 'OptTensor' = None
        x_r: 'OptTensor' = None
        alpha_l: 'OptTensor' = None
        alpha_r: 'OptTensor' = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)
        assert x_l is not None
        assert alpha_l is not None
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, edge_weight = remove_self_loops(edge_index, edge_attr=edge_weight)
                if edge_weight != None:
                    edge_index, edge_weight = add_self_loops(edge_index, edge_attr=edge_weight, num_nodes=num_nodes)
                else:
                    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=(alpha_l, alpha_r), edge_weight=edge_weight, size=size)
        alpha = self._alpha
        self._alpha = None
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        if self.bias is not None:
            out += self.bias
        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: 'Tensor', alpha_j: 'Tensor', alpha_i: 'OptTensor', index: 'Tensor', ptr: 'OptTensor', size_i: 'Optional[int]', edge_weight: 'OptTensor'=None) ->Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        if edge_weight != None:
            alpha = alpha.mul(edge_weight.unsqueeze(1))
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.heads)


class GAT(torch.nn.Module):

    def __init__(self, num_features, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, num_classes, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)


OptTensor = Optional[Tensor]


def add_remaining_self_loops(edge_index: 'Tensor', edge_attr: 'OptTensor'=None, fill_value: 'Union[float, Tensor, str]'=None, num_nodes: 'Optional[int]'=None) ->Tuple[Tensor, OptTensor]:
    """Adds remaining self-loop :math:`(i,i) \\in \\mathcal{E}` to every node
    :math:`i \\in \\mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted or has multi-dimensional edge features
    (:obj:`edge_attr != None`), edge features of non-existing self-loops will
    be added according to :obj:`fill_value`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_attr != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`1.`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    N = maybe_num_nodes(edge_index, num_nodes)
    mask = edge_index[0] != edge_index[1]
    loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    if edge_attr is not None:
        if fill_value is None:
            loop_attr = edge_attr.new_full((N,) + edge_attr.size()[1:], 1.0)
        elif isinstance(fill_value, (int, float)):
            loop_attr = edge_attr.new_full((N,) + edge_attr.size()[1:], fill_value)
        elif isinstance(fill_value, Tensor):
            loop_attr = fill_value
            if edge_attr.dim() != loop_attr.dim():
                loop_attr = loop_attr.unsqueeze(0)
            sizes = [N] + [1] * (loop_attr.dim() - 1)
            loop_attr = loop_attr.repeat(*sizes)
        elif isinstance(fill_value, str):
            loop_attr = scatter(edge_attr, edge_index[1], dim=0, dim_size=N, reduce=fill_value)
        else:
            raise AttributeError("No valid 'fill_value' provided")
        inv_mask = ~mask
        loop_attr[edge_index[0][inv_mask]] = edge_attr[inv_mask]
        edge_attr = torch.cat([edge_attr[mask], loop_attr], dim=0)
    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)
    return edge_index, edge_attr


def scatter_add(src: 'torch.Tensor', index: 'torch.Tensor', dim: 'int'=-1, out: 'Optional[torch.Tensor]'=None, dim_size: 'Optional[int]'=None) ->torch.Tensor:
    return scatter_sum(src, index, dim, out, dim_size)


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False, add_self_loops=True, dtype=None):
    fill_value = 2.0 if improved else 1.0
    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1.0, dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.0)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t
    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class GCNConv(MessagePassing):
    """The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \\mathbf{X}^{\\prime} = \\mathbf{\\hat{D}}^{-1/2} \\mathbf{\\hat{A}}
        \\mathbf{\\hat{D}}^{-1/2} \\mathbf{X} \\mathbf{\\Theta},

    where :math:`\\mathbf{\\hat{A}} = \\mathbf{A} + \\mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\\hat{D}_{ii} = \\sum_{j=0} \\hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Its node-wise formulation is given by:

    .. math::
        \\mathbf{x}^{\\prime}_i = \\mathbf{\\Theta} \\sum_{j}
        \\frac{1}{\\sqrt{\\hat{d}_j \\hat{d}_i}} \\mathbf{x}_j

    with :math:`\\hat{d}_i = 1 + \\sum_{j \\in \\mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`i` to target
    node :obj:`j` (default: :obj:`1`)

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\\mathbf{\\hat{A}}` as :math:`\\mathbf{A} + 2\\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\\mathbf{\\hat{D}}^{-1/2} \\mathbf{\\hat{A}}
            \\mathbf{\\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _cached_edge_index: 'Optional[Tuple[Tensor, Tensor]]'
    _cached_adj_t: 'Optional[SparseTensor]'

    def __init__(self, in_channels: 'int', out_channels: 'int', improved: 'bool'=False, cached: 'bool'=False, add_self_loops: 'bool'=True, normalize: 'bool'=True, bias: 'bool'=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GCNConv, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self._cached_edge_index = None
        self._cached_adj_t = None
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: 'Tensor', edge_index: 'Adj', edge_weight: 'OptTensor'=None) ->Tensor:
        """"""
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(edge_index, edge_weight, x.size(self.node_dim), self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = edge_index, edge_weight
                else:
                    edge_index, edge_weight = cache[0], cache[1]
            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(edge_index, edge_weight, x.size(self.node_dim), self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        x = torch.matmul(x, self.weight)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        if self.bias is not None:
            out += self.bias
        return out

    def message(self, x_j: 'Tensor', edge_weight: 'OptTensor') ->Tensor:
        if edge_weight is None:
            return x_j
        else:
            return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: 'SparseTensor', x: 'Tensor') ->Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class GCN(torch.nn.Module):

    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""

    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction
        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(MLP, self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        self.output_dim = output_dim
        if num_layers < 1:
            raise ValueError('number of layers should be positive!')
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


def reset(nn):

    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()
    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


class GINConv(MessagePassing):
    """The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
        \\mathbf{x}^{\\prime}_i = h_{\\mathbf{\\Theta}} \\left( (1 + \\epsilon) \\cdot
        \\mathbf{x}_i + \\sum_{j \\in \\mathcal{N}(i)} \\mathbf{x}_j \\right)

    or

    .. math::
        \\mathbf{X}^{\\prime} = h_{\\mathbf{\\Theta}} \\left( \\left( \\mathbf{A} +
        (1 + \\epsilon) \\cdot \\mathbf{I} \\right) \\cdot \\mathbf{X} \\right),

    here :math:`h_{\\mathbf{\\Theta}}` denotes a neural network, *.i.e.* an MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\\mathbf{\\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, nn: 'Callable', eps: 'float'=0.0, train_eps: 'bool'=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GINConv, self).__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: 'Union[Tensor, OptPairTensor]', edge_index: 'Adj', edge_weight: 'OptTensor'=None, size: 'Size'=None) ->Tensor:
        """"""
        if isinstance(x, Tensor):
            x: 'OptPairTensor' = (x, x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r
        return self.nn(out)

    def message(self, x_j: 'Tensor', edge_weight: 'OptTensor') ->Tensor:
        if edge_weight is None:
            return x_j
        else:
            return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: 'SparseTensor', x: 'OptPairTensor') ->Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class GIN(torch.nn.Module):

    def __init__(self):
        super(GIN, self).__init__()
        num_features = dataset.num_features
        dim = 32
        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)
        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)
        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)
        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)
        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)
        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class SAGEConv(MessagePassing):
    """The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \\mathbf{x}^{\\prime}_i = \\mathbf{W}_1 \\mathbf{x}_i + \\mathbf{W_2} \\cdot
        \\mathrm{mean}_{j \\in \\mathcal{N(i)}} \\mathbf{x}_j

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\\ell_2`-normalized, *i.e.*,
            :math:`\\frac{\\mathbf{x}^{\\prime}_i}
            {\\| \\mathbf{x}^{\\prime}_i \\|_2}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels: 'Union[int, Tuple[int, int]]', out_channels: 'int', normalize: 'bool'=False, bias: 'bool'=True, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        if isinstance(in_channels, int):
            in_channels = in_channels, in_channels
        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_r = Linear(in_channels[1], out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: 'Union[Tensor, OptPairTensor]', edge_index: 'Adj', edge_weight: 'OptTensor'=None, size: 'Size'=None) ->Tensor:
        """"""
        if isinstance(x, Tensor):
            x: 'OptPairTensor' = (x, x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        out = self.lin_l(out)
        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)
        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)
        return out

    def message(self, x_j: 'Tensor', edge_weight: 'OptTensor') ->Tensor:
        if edge_weight is None:
            return x_j
        else:
            return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: 'SparseTensor', x: 'OptPairTensor') ->Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class GraphSAGE(torch.nn.Module):

    def __init__(self, num_features, hidden_features):
        super(GraphSAGE, self).__init__()
        self.num_layers = 2
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers):
            inc = outc = hidden_features
            if i == 0:
                inc = num_features
            if i == self.num_layers - 1:
                outc = hidden_features // 2
            self.convs.append(SAGEConv(inc, outc))

    def encode(self, data):
        x, edge_index = data.x, data.train_pos_edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


class HeteroRGCNLayer(nn.Module):

    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        self.weight = nn.ModuleDict({name: nn.Linear(in_size, out_size) for name in etypes})

    def forward(self, G, feat_dict):
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            Wh = self.weight[etype](feat_dict[srctype])
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            funcs[etype] = fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h')
        G.multi_update_all(funcs, 'sum')
        return {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes}


class HeteroRGCN(nn.Module):

    def __init__(self, G, in_size, hidden_size, out_size):
        super(HeteroRGCN, self).__init__()
        self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.etypes)
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size, G.etypes)

    def forward(self, G, out_key):
        input_dict = {ntype: G.nodes[ntype].data['inp'] for ntype in G.ntypes}
        h_dict = self.layer1(G, input_dict)
        h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        return h_dict[out_key]


class SemanticAttention(nn.Module):

    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(nn.Linear(in_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1, bias=False))

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)


class HANLayer(torch.nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    num_metapath : number of metapath based sub-graph
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, num_metapath, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()
        self.gat_layers = nn.ModuleList()
        for i in range(num_metapath):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads, dropout, dropout, activation=F.elu, allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.num_metapath = num_metapath

    def forward(self, block_list, h_list):
        semantic_embeddings = []
        for i, block in enumerate(block_list):
            semantic_embeddings.append(self.gat_layers[i](block, h_list[i]).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
        return self.semantic_attention(semantic_embeddings)


class HAN(nn.Module):

    def __init__(self, num_metapath, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_metapath, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(num_metapath, hidden_size * num_heads[l - 1], hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)
        return self.predict(h)


class HGTLayer(nn.Module):

    def __init__(self, in_dim, out_dim, node_dict, edge_dict, n_heads, dropout=0.2, use_norm=False):
        super(HGTLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None
        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm
        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]
                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]
                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)
                e_id = self.edge_dict[etype]
                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]
                k = torch.einsum('bij,ijk->bik', k, relation_att)
                v = torch.einsum('bij,ijk->bik', v, relation_msg)
                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata['v_%d' % e_id] = v
                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')
                sub_graph.edata['t'] = attn_score.unsqueeze(-1)
            G.multi_update_all({etype: (fn.u_mul_e('v_%d' % e_id, 't', 'm'), fn.sum('m', 't')) for etype, e_id in edge_dict.items()}, cross_reducer='mean')
            new_h = {}
            for ntype in G.ntypes:
                """
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                """
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                t = G.nodes[ntype].data['t'].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype] * (1 - alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h


class HGT(nn.Module):

    def __init__(self, G, node_dict, edge_dict, n_inp, n_hid, n_out, n_layers, n_heads, use_norm=True):
        super(HGT, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws = nn.ModuleList()
        for t in range(len(node_dict)):
            self.adapt_ws.append(nn.Linear(n_inp, n_hid))
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(n_hid, n_hid, node_dict, edge_dict, n_heads, use_norm=use_norm))
        self.out = nn.Linear(n_hid, n_out)

    def forward(self, G, out_key):
        h = {}
        for ntype in G.ntypes:
            n_id = self.node_dict[ntype]
            h[ntype] = F.gelu(self.adapt_ws[n_id](G.nodes[ntype].data['inp']))
        for i in range(self.n_layers):
            h = self.gcs[i](G, h)
        return self.out(h[out_key])


def activate_func(x, func):
    if func == 'tanh':
        return torch.tanh(x)
    elif hasattr(F, func):
        return getattr(F, func)(x)
    elif func == '':
        pass
    else:
        raise TypeError('PyTorch does not support activation function {}'.format(func))
    return x


class Topkpool(torch.nn.Module):

    def __init__(self, args):
        super(Topkpool, self).__init__()
        self.args = args
        missing_keys = list(set(['features_num', 'num_class', 'num_graph_features', 'ratio', 'dropout', 'act']) - set(self.args.keys()))
        if len(missing_keys) > 0:
            raise Exception('Missing keys: %s.' % ','.join(missing_keys))
        self.num_features = self.args['features_num']
        self.num_classes = self.args['num_class']
        self.ratio = self.args['ratio']
        self.dropout = self.args['dropout']
        self.num_graph_features = self.args['num_graph_features']
        self.conv1 = GraphConv(self.num_features, 128)
        self.pool1 = TopKPooling(128, ratio=self.ratio)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=self.ratio)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=self.ratio)
        self.lin1 = torch.nn.Linear(256 + self.num_graph_features, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.num_graph_features > 0:
            graph_feature = data.gf
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = x1 + x2 + x3
        if self.num_graph_features > 0:
            x = torch.cat([x, graph_feature], dim=-1)
        x = self.lin1(x)
        x = activate_func(x, self.args['act'])
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = activate_func(x, self.args['act'])
        x = F.log_softmax(self.lin3(x), dim=-1)
        return x


class _GCN(torch.nn.Module):

    def __init__(self, input_dimension: 'int', dimensions: '_typing.Sequence[int]', _act: '_typing.Optional[str]', _dropout: '_typing.Optional[float]'):
        super(_GCN, self).__init__()
        self._act: '_typing.Optional[str]' = _act
        self._dropout: '_typing.Optional[float]' = _dropout
        self.__convolution_layers: 'torch.nn.ModuleList' = torch.nn.ModuleList()
        for layer, output_dimension in enumerate(dimensions):
            self.__convolution_layers.append(GCNConv(input_dimension if layer == 0 else dimensions[layer - 1], output_dimension))

    def forward(self, data: 'torch_geometric.data.Data', *__args, **__kwargs) ->_typing.Sequence[torch.Tensor]:
        x: 'torch.Tensor' = data.x
        edge_index: 'torch.LongTensor' = data.edge_index
        if hasattr(data, 'edge_weight') and isinstance(getattr(data, 'edge_weight'), torch.Tensor) and torch.is_tensor(data.edge_weight):
            edge_weight: '_typing.Optional[torch.Tensor]' = data.edge_weight
        else:
            edge_weight: '_typing.Optional[torch.Tensor]' = None
        results: '_typing.MutableSequence[torch.Tensor]' = [x]
        for layer, convolution_layer in enumerate(self.__convolution_layers):
            x = convolution_layer(x, edge_index, edge_weight)
            if layer < len(self.__convolution_layers) - 1:
                x: 'torch.Tensor' = _utils.activation.activation_func(x, self._act)
                if isinstance(self._dropout, float) and 0 <= self._dropout <= 1:
                    x = torch.nn.functional.dropout(x, self._dropout, self.training)
            results.append(x)
        return results


class _GIN(torch.nn.Module):

    def __init__(self, input_dimension: 'int', dimensions: '_typing.Sequence[int]', _act: 'str', _dropout: 'float', mlp_layers: 'int', _eps: 'str'):
        super(_GIN, self).__init__()
        self._act: 'str' = _act

        def _get_act() ->torch.nn.Module:
            if _act == 'leaky_relu':
                return torch.nn.LeakyReLU()
            elif _act == 'relu':
                return torch.nn.ReLU()
            elif _act == 'elu':
                return torch.nn.ELU()
            elif _act == 'tanh':
                return torch.nn.Tanh()
            elif _act == 'PReLU'.lower():
                return torch.nn.PReLU()
            else:
                return torch.nn.ReLU()
        convolution_layers: 'torch.nn.ModuleList' = torch.nn.ModuleList()
        batch_normalizations: 'torch.nn.ModuleList' = torch.nn.ModuleList()
        __mlp_layers = [torch.nn.Linear(input_dimension, dimensions[0])]
        for _ in range(mlp_layers - 1):
            __mlp_layers.append(_get_act())
            __mlp_layers.append(torch.nn.Linear(dimensions[0], dimensions[0]))
        convolution_layers.append(GINConv(torch.nn.Sequential(*__mlp_layers), train_eps=_eps == 'True'))
        batch_normalizations.append(torch.nn.BatchNorm1d(dimensions[0]))
        num_layers: 'int' = len(dimensions)
        for layer in range(num_layers - 1):
            __mlp_layers = [torch.nn.Linear(dimensions[layer], dimensions[layer + 1])]
            for _ in range(mlp_layers - 1):
                __mlp_layers.append(_get_act())
                __mlp_layers.append(torch.nn.Linear(dimensions[layer + 1], dimensions[layer + 1]))
            convolution_layers.append(GINConv(torch.nn.Sequential(*__mlp_layers), train_eps=_eps == 'True'))
            batch_normalizations.append(torch.nn.BatchNorm1d(dimensions[layer + 1]))
        self.__convolution_layers: 'torch.nn.ModuleList' = convolution_layers
        self.__batch_normalizations: 'torch.nn.ModuleList' = batch_normalizations

    def forward(self, data: 'torch_geometric.data.Data', *__args, **__kwargs) ->_typing.Sequence[torch.Tensor]:
        x: 'torch.Tensor' = data.x
        edge_index: 'torch.Tensor' = data.edge_index
        results: '_typing.MutableSequence[torch.Tensor]' = [x]
        num_layers = len(self.__convolution_layers)
        for layer in range(num_layers):
            x: 'torch.Tensor' = self.__convolution_layers[layer](x, edge_index)
            x: 'torch.Tensor' = _utils.activation.activation_func(x, self._act)
            x: 'torch.Tensor' = self.__batch_normalizations[layer](x)
            results.append(x)
        return results


class _SAGE(torch.nn.Module):

    def __init__(self, input_dimension: 'int', dimensions: '_typing.Sequence[int]', _act: '_typing.Optional[str]', _dropout: '_typing.Optional[float]', aggr: '_typing.Optional[str]'):
        super(_SAGE, self).__init__()
        self._act: '_typing.Optional[str]' = _act
        self._dropout: '_typing.Optional[float]' = _dropout
        self.__convolution_layers: 'torch.nn.ModuleList' = torch.nn.ModuleList()
        for layer, output_dimension in enumerate(dimensions):
            self.__convolution_layers.append(SAGEConv(input_dimension if layer == 0 else dimensions[layer - 1], output_dimension, aggr=aggr))

    def forward(self, data: 'torch_geometric.data.Data', *__args, **__kwargs) ->_typing.Sequence[torch.Tensor]:
        x: 'torch.Tensor' = data.x
        edge_index: 'torch.LongTensor' = data.edge_index
        results: '_typing.MutableSequence[torch.Tensor]' = [x]
        for layer, convolution_layer in enumerate(self.__convolution_layers):
            x = convolution_layer(x, edge_index)
            if layer < len(self.__convolution_layers) - 1:
                x = _utils.activation.activation_func(x, self._act)
                if isinstance(self._dropout, float) and 0 <= self._dropout <= 1:
                    x = torch.nn.functional.dropout(x, self._dropout, self.training)
            results.append(x)
        return results


class _TopK(torch.nn.Module):

    def __init__(self, input_dimension: 'int', dimensions: '_typing.Sequence[int]'):
        super(_TopK, self).__init__()
        self.__gcn_layers: 'torch.nn.ModuleList' = torch.nn.ModuleList()
        self.__batch_normalizations: 'torch.nn.ModuleList' = torch.nn.ModuleList()
        self.__num_layers = len(dimensions)
        for layer in range(self.__num_layers):
            self.__gcn_layers.append(GraphConv(input_dimension if layer == 0 else dimensions[layer - 1], dimensions[layer]))
            self.__batch_normalizations.append(torch.nn.BatchNorm1d(dimensions[layer]))

    def forward(self, graph: 'dgl.DGLGraph', *__args, **__kwargs) ->_typing.Sequence[torch.Tensor]:
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        if 'feat' in graph.ndata:
            h: 'torch.Tensor' = graph.ndata['feat']
        else:
            h: 'torch.Tensor' = graph.ndata['attr']
        hidden_rep = [h]
        for i in range(self.__num_layers):
            h = self.__gcn_layers[i](graph, h)
            h = self.__batch_normalizations[i](h)
            h = torch.nn.functional.relu(h)
            hidden_rep.append(h)
        return hidden_rep


class GATUtils:

    @classmethod
    def to_total_hidden_dimensions(cls, per_head_output_dimensions: '_typing.Sequence[int]', num_hidden_heads: 'int', num_output_heads: 'int', concat_last: 'bool'=False) ->_typing.Sequence[int]:
        return [(d * (num_hidden_heads if layer < len(per_head_output_dimensions) - 1 else num_output_heads if concat_last else 1)) for layer, d in enumerate(per_head_output_dimensions)]


class _GAT(torch.nn.Module):

    def __init__(self, input_dimension: 'int', per_head_output_dimensions: '_typing.Sequence[int]', num_hidden_heads: 'int', num_output_heads: 'int', _dropout: 'float', _act: '_typing.Optional[str]', concat_last: 'bool'=True):
        super(_GAT, self).__init__()
        self._dropout: 'float' = _dropout
        self._act: '_typing.Optional[str]' = _act
        total_output_dimensions: '_typing.Sequence[int]' = GATUtils.to_total_hidden_dimensions(per_head_output_dimensions, num_hidden_heads, num_output_heads, concat_last=concat_last)
        num_layers = len(per_head_output_dimensions)
        self.__convolution_layers: 'torch.nn.ModuleList' = torch.nn.ModuleList()
        for layer in range(len(per_head_output_dimensions)):
            self.__convolution_layers.append(GATConv(input_dimension if layer == 0 else total_output_dimensions[layer - 1], per_head_output_dimensions[layer], num_hidden_heads if layer < num_layers - 1 else num_output_heads, dropout=_dropout, concat=True if layer < num_layers - 1 or concat_last else False))

    def forward(self, data: 'torch_geometric.data.Data', *__args, **__kwargs):
        x: 'torch.Tensor' = data.x
        edge_index: 'torch.LongTensor' = data.edge_index
        if hasattr(data, 'edge_weight') and isinstance(getattr(data, 'edge_weight'), torch.Tensor) and torch.is_tensor(data.edge_weight):
            edge_weight: '_typing.Optional[torch.Tensor]' = data.edge_weight
        else:
            edge_weight: '_typing.Optional[torch.Tensor]' = None
        results: '_typing.MutableSequence[torch.Tensor]' = [x]
        for layer, _gat in enumerate(self.__convolution_layers):
            x: 'torch.Tensor' = torch.nn.functional.dropout(x, self._dropout, self.training)
            x: 'torch.Tensor' = _gat(x, edge_index, edge_weight)
            if layer < len(self.__convolution_layers) - 1:
                x: 'torch.Tensor' = _utils.activation.activation_func(x, self._act)
            results.append(x)
        logging.debug('{:d} layer, each layer shape {:s}'.format(len(results), ' '.join([str(x.shape) for x in results])))
        return results


class _GraphSAINTAggregationLayers:


    class MultiOrderAggregationLayer(torch.nn.Module):


        class Order0Aggregator(torch.nn.Module):

            def __init__(self, input_dimension: 'int', output_dimension: 'int', bias: 'bool'=True, activation: '_typing.Optional[str]'='ReLU', batch_norm: 'bool'=True):
                super().__init__()
                if not type(input_dimension) == type(output_dimension) == int:
                    raise TypeError
                if not (input_dimension > 0 and output_dimension > 0):
                    raise ValueError
                if not type(bias) == bool:
                    raise TypeError
                self.__linear_transform = torch.nn.Linear(input_dimension, output_dimension, bias)
                self.__linear_transform.reset_parameters()
                if type(activation) == str:
                    if activation.lower() == 'ReLU'.lower():
                        self.__activation = torch.nn.functional.relu
                    elif activation.lower() == 'elu':
                        self.__activation = torch.nn.functional.elu
                    elif hasattr(torch.nn.functional, activation) and callable(getattr(torch.nn.functional, activation)):
                        self.__activation = getattr(torch.nn.functional, activation)
                    else:
                        self.__activation = lambda x: x
                else:
                    self.__activation = lambda x: x
                if type(batch_norm) != bool:
                    raise TypeError
                else:
                    self.__optional_batch_normalization: '_typing.Optional[torch.nn.BatchNorm1d]' = torch.nn.BatchNorm1d(output_dimension, 1e-08) if batch_norm else None

            def forward(self, x: '_typing.Union[torch.Tensor, _typing.Tuple[torch.Tensor, torch.Tensor]]', _edge_index: 'torch.Tensor', _edge_weight: '_typing.Optional[torch.Tensor]'=None, _size: '_typing.Optional[_typing.Tuple[int, int]]'=None) ->torch.Tensor:
                __output: 'torch.Tensor' = self.__linear_transform(x)
                if self.__activation is not None and callable(self.__activation):
                    __output: 'torch.Tensor' = self.__activation(__output)
                if self.__optional_batch_normalization is not None and isinstance(self.__optional_batch_normalization, torch.nn.BatchNorm1d):
                    __output: 'torch.Tensor' = self.__optional_batch_normalization(__output)
                return __output


        class Order1Aggregator(MessagePassing):

            def __init__(self, input_dimension: 'int', output_dimension: 'int', bias: 'bool'=True, activation: '_typing.Optional[str]'='ReLU', batch_norm: 'bool'=True):
                super().__init__(aggr='add')
                if not type(input_dimension) == type(output_dimension) == int:
                    raise TypeError
                if not (input_dimension > 0 and output_dimension > 0):
                    raise ValueError
                if not type(bias) == bool:
                    raise TypeError
                self.__linear_transform = torch.nn.Linear(input_dimension, output_dimension, bias)
                self.__linear_transform.reset_parameters()
                if type(activation) == str:
                    if activation.lower() == 'ReLU'.lower():
                        self.__activation = torch.nn.functional.relu
                    elif activation.lower() == 'elu':
                        self.__activation = torch.nn.functional.elu
                    elif hasattr(torch.nn.functional, activation) and callable(getattr(torch.nn.functional, activation)):
                        self.__activation = getattr(torch.nn.functional, activation)
                    else:
                        self.__activation = lambda x: x
                else:
                    self.__activation = lambda x: x
                if type(batch_norm) != bool:
                    raise TypeError
                else:
                    self.__optional_batch_normalization: '_typing.Optional[torch.nn.BatchNorm1d]' = torch.nn.BatchNorm1d(output_dimension, 1e-08) if batch_norm else None

            def forward(self, x: '_typing.Union[torch.Tensor, _typing.Tuple[torch.Tensor, torch.Tensor]]', _edge_index: 'torch.Tensor', _edge_weight: '_typing.Optional[torch.Tensor]'=None, _size: '_typing.Optional[_typing.Tuple[int, int]]'=None) ->torch.Tensor:
                if type(x) == torch.Tensor:
                    x: '_typing.Tuple[torch.Tensor, torch.Tensor]' = (x, x)
                __output = self.propagate(_edge_index, x=x, edge_weight=_edge_weight, size=_size)
                __output: 'torch.Tensor' = self.__linear_transform(__output)
                if self.__activation is not None and callable(self.__activation):
                    __output: 'torch.Tensor' = self.__activation(__output)
                if self.__optional_batch_normalization is not None and isinstance(self.__optional_batch_normalization, torch.nn.BatchNorm1d):
                    __output: 'torch.Tensor' = self.__optional_batch_normalization(__output)
                return __output

            def message(self, x_j: 'torch.Tensor', edge_weight: '_typing.Optional[torch.Tensor]') ->torch.Tensor:
                return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

            def message_and_aggregate(self, adj_t: 'SparseTensor', x: '_typing.Union[torch.Tensor, _typing.Tuple[torch.Tensor, torch.Tensor]]') ->torch.Tensor:
                return matmul(adj_t, x[0], reduce=self.aggr)

        @property
        def integral_output_dimension(self) ->int:
            return (self._order + 1) * self._each_order_output_dimension

        def __init__(self, _input_dimension: 'int', _each_order_output_dimension: 'int', _order: 'int', bias: 'bool'=True, activation: '_typing.Optional[str]'='ReLU', batch_norm: 'bool'=True, _dropout: '_typing.Optional[float]'=...):
            super().__init__()
            if not (type(_input_dimension) == type(_order) == int and type(_each_order_output_dimension) == int):
                raise TypeError
            if _input_dimension <= 0 or _each_order_output_dimension <= 0:
                raise ValueError
            if _order not in (0, 1):
                raise ValueError('Unsupported order number')
            self._input_dimension: 'int' = _input_dimension
            self._each_order_output_dimension: 'int' = _each_order_output_dimension
            self._order: 'int' = _order
            if type(bias) != bool:
                raise TypeError
            self.__order0_transform = self.Order0Aggregator(self._input_dimension, self._each_order_output_dimension, bias, activation, batch_norm)
            if _order == 1:
                self.__order1_transform = self.Order1Aggregator(self._input_dimension, self._each_order_output_dimension, bias, activation, batch_norm)
            else:
                self.__order1_transform = None
            if _dropout is not None and type(_dropout) == float:
                if _dropout < 0:
                    _dropout = 0
                if _dropout > 1:
                    _dropout = 1
                self.__optional_dropout: '_typing.Optional[torch.nn.Dropout]' = torch.nn.Dropout(_dropout)
            else:
                self.__optional_dropout: '_typing.Optional[torch.nn.Dropout]' = None

        def _forward(self, x: '_typing.Union[torch.Tensor, _typing.Tuple[torch.Tensor, torch.Tensor]]', edge_index: 'torch.Tensor', edge_weight: '_typing.Optional[torch.Tensor]'=None, size: '_typing.Optional[_typing.Tuple[int, int]]'=None) ->torch.Tensor:
            if self.__order1_transform is not None and isinstance(self.__order1_transform, self.Order1Aggregator):
                __output: 'torch.Tensor' = torch.cat([self.__order0_transform(x, edge_index, edge_weight, size), self.__order1_transform(x, edge_index, edge_weight, size)], dim=1)
            else:
                __output: 'torch.Tensor' = self.__order0_transform(x, edge_index, edge_weight, size)
            if self.__optional_dropout is not None and isinstance(self.__optional_dropout, torch.nn.Dropout):
                __output: 'torch.Tensor' = self.__optional_dropout(__output)
            return __output

        def forward(self, data) ->torch.Tensor:
            x: 'torch.Tensor' = getattr(data, 'x')
            if type(x) != torch.Tensor:
                raise TypeError
            edge_index: 'torch.LongTensor' = getattr(data, 'edge_index')
            if type(edge_index) != torch.Tensor:
                raise TypeError
            edge_weight: '_typing.Optional[torch.Tensor]' = getattr(data, 'edge_weight', None)
            if edge_weight is not None and type(edge_weight) != torch.Tensor:
                raise TypeError
            return self._forward(x, edge_index, edge_weight)


    class WrappedDropout(torch.nn.Module):

        def __init__(self, dropout_module: 'torch.nn.Dropout'):
            super().__init__()
            self.__dropout_module: 'torch.nn.Dropout' = dropout_module

        def forward(self, tenser_or_data) ->torch.Tensor:
            if type(tenser_or_data) == torch.Tensor:
                return self.__dropout_module(tenser_or_data)
            elif hasattr(tenser_or_data, 'x') and type(getattr(tenser_or_data, 'x')) == torch.Tensor:
                return self.__dropout_module(getattr(tenser_or_data, 'x'))
            else:
                raise TypeError


class GraphSAINTMultiOrderAggregationModel(ClassificationSupportedSequentialModel):

    def __init__(self, num_features: 'int', num_classes: 'int', _output_dimension_for_each_order: 'int', _layers_order_list: '_typing.Sequence[int]', _pre_dropout: 'float', _layers_dropout: '_typing.Union[float, _typing.Sequence[float]]', activation: '_typing.Optional[str]'='ReLU', bias: 'bool'=True, batch_norm: 'bool'=True, normalize: 'bool'=True):
        super(GraphSAINTMultiOrderAggregationModel, self).__init__()
        if type(_output_dimension_for_each_order) != int:
            raise TypeError
        if not _output_dimension_for_each_order > 0:
            raise ValueError
        self._layers_order_list: '_typing.Sequence[int]' = _layers_order_list
        if isinstance(_layers_dropout, _typing.Sequence):
            if len(_layers_dropout) != len(_layers_order_list):
                raise ValueError
            else:
                self._layers_dropout: '_typing.Sequence[float]' = _layers_dropout
        elif type(_layers_dropout) == float:
            if _layers_dropout < 0:
                _layers_dropout = 0
            if _layers_dropout > 1:
                _layers_dropout = 1
            self._layers_dropout: '_typing.Sequence[float]' = [_layers_dropout for _ in _layers_order_list]
        else:
            raise TypeError
        if type(_pre_dropout) != float:
            raise TypeError
        else:
            if _pre_dropout < 0:
                _pre_dropout = 0
            if _pre_dropout > 1:
                _pre_dropout = 1
        self.__sequential_encoding_layers: 'torch.nn.ModuleList' = torch.nn.ModuleList((_GraphSAINTAggregationLayers.WrappedDropout(torch.nn.Dropout(_pre_dropout)), _GraphSAINTAggregationLayers.MultiOrderAggregationLayer(num_features, _output_dimension_for_each_order, _layers_order_list[0], bias, activation, batch_norm, _layers_dropout[0])))
        for _layer_index in range(1, len(_layers_order_list)):
            self.__sequential_encoding_layers.append(_GraphSAINTAggregationLayers.MultiOrderAggregationLayer(self.__sequential_encoding_layers[-1].integral_output_dimension, _output_dimension_for_each_order, _layers_order_list[_layer_index], bias, activation, batch_norm, _layers_dropout[_layer_index]))
        self.__apply_normalize: 'bool' = normalize
        self.__linear_transform: 'torch.nn.Linear' = torch.nn.Linear(self.__sequential_encoding_layers[-1].integral_output_dimension, num_classes, bias)
        self.__linear_transform.reset_parameters()

    def cls_decode(self, x: 'torch.Tensor') ->torch.Tensor:
        if self.__apply_normalize:
            x: 'torch.Tensor' = torch.nn.functional.normalize(x, p=2, dim=1)
        return torch.nn.functional.log_softmax(self.__linear_transform(x), dim=1)

    def cls_encode(self, data) ->torch.Tensor:
        if type(getattr(data, 'x')) != torch.Tensor:
            raise TypeError
        if type(getattr(data, 'edge_index')) != torch.Tensor:
            raise TypeError
        if getattr(data, 'edge_weight', None) is not None and type(getattr(data, 'edge_weight')) != torch.Tensor:
            raise TypeError
        for encoding_layer in self.__sequential_encoding_layers:
            setattr(data, 'x', encoding_layer(data))
        return getattr(data, 'x')

    @property
    def sequential_encoding_layers(self) ->torch.nn.ModuleList:
        return self.__sequential_encoding_layers


class GCN4GNNGuard(GCN):

    def __init__(self, nfeat, nclass, nhid, activation, dropout=0.5, lr=0.01, drop=False, weight_decay=0.0005, with_relu=True, with_bias=True, add_self_loops=True, normalize=True):
        super(GCN4GNNGuard, self).__init__(nfeat, nclass, nhid, activation, dropout=dropout, add_self_loops=add_self_loops, normalize=normalize)
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.nclass = nclass
        self.nfeat = nfeat
        self.hidden_sizes = nhid
        self.drop = drop
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.gc1 = GCNConv(nfeat, nhid[0], bias=True)
        self.gc2 = GCNConv(nhid[0], nclass, bias=True)

    def forward(self, x, adj):
        """we don't change the edge_index, just update the edge_weight;
        some edge_weight are regarded as removed if it equals to zero"""
        x = x.to_dense()
        """GCN and GAT"""
        if self.attention:
            adj = self.att_coef(x, adj, i=0)
        edge_index = adj._indices()
        x = self.gc1(x, edge_index, edge_weight=adj._values())
        x = F.relu(x)
        if self.attention:
            adj_2 = self.att_coef(x, adj, i=1)
            adj_memory = adj_2.to_dense()
            row, col = adj_memory.nonzero()[:, 0], adj_memory.nonzero()[:, 1]
            edge_index = torch.stack((row, col), dim=0)
            adj_values = adj_memory[row, col]
        else:
            edge_index = adj._indices()
            adj_values = adj._values()
        edge_index = edge_index
        adj_values = adj_values
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index, edge_weight=adj_values)
        return F.log_softmax(x, dim=1)


    class myData:

        def __init__(self, x, edge_index, edge_weight=None):
            self.x = x
            self.edge_index = edge_index
            self.edge_weight = edge_weight

    def att_coef(self, fea, edge_index, is_lil=False, i=0):
        if is_lil == False:
            edge_index = edge_index._indices()
        else:
            edge_index = edge_index.tocoo()
        n_node = fea.shape[0]
        row, col = edge_index[0].cpu().data.numpy()[:], edge_index[1].cpu().data.numpy()[:]
        fea_copy = fea.cpu().data.numpy()
        sim_matrix = cosine_similarity(X=fea_copy, Y=fea_copy)
        sim = sim_matrix[row, col]
        sim[sim < 0.1] = 0
        """build a attention matrix"""
        att_dense = lil_matrix((n_node, n_node), dtype=np.float32)
        att_dense[row, col] = sim
        if att_dense[0, 0] == 1:
            att_dense = att_dense - sp.diags(att_dense.diagonal(), offsets=0, format='lil')
        att_dense_norm = normalize(att_dense, axis=1, norm='l1')
        """add learnable dropout, make character vector"""
        if self.drop:
            character = np.vstack((att_dense_norm[row, col].A1, att_dense_norm[col, row].A1))
            character = torch.from_numpy(character.T)
            drop_score = self.drop_learn_1(character)
            drop_score = torch.sigmoid(drop_score)
            mm = torch.nn.Threshold(0.5, 0)
            drop_score = mm(drop_score)
            mm_2 = torch.nn.Threshold(-0.49, 1)
            drop_score = mm_2(-drop_score)
            drop_decision = drop_score.clone().requires_grad_()
            drop_matrix = lil_matrix((n_node, n_node), dtype=np.float32)
            drop_matrix[row, col] = drop_decision.cpu().data.numpy().squeeze(-1)
            att_dense_norm = att_dense_norm.multiply(drop_matrix.tocsr())
        if att_dense_norm[0, 0] == 0:
            degree = (att_dense_norm != 0).sum(1).A1
            lam = 1 / (degree + 1)
            self_weight = sp.diags(np.array(lam), offsets=0, format='lil')
            att = att_dense_norm + self_weight
        else:
            att = att_dense_norm
        row, col = att.nonzero()
        att_adj = np.vstack((row, col))
        att_edge_weight = att[row, col]
        att_edge_weight = np.exp(att_edge_weight)
        att_edge_weight = torch.tensor(np.array(att_edge_weight)[0], dtype=torch.float32)
        att_adj = torch.tensor(att_adj, dtype=torch.int64)
        shape = n_node, n_node
        new_adj = torch.sparse.FloatTensor(att_adj, att_edge_weight, shape)
        return new_adj

    def add_loop_sparse(self, adj, fill_value=1):
        row = torch.range(0, int(adj.shape[0] - 1), dtype=torch.int64)
        i = torch.stack((row, row), dim=0)
        v = torch.ones(adj.shape[0], dtype=torch.float32)
        shape = adj.shape
        I_n = torch.sparse.FloatTensor(i, v, shape)
        return adj + I_n

    def fit(self, features, adj, labels, idx_train, idx_val=None, idx_test=None, train_iters=81, att_0=None, attention=False, model_name=None, verbose=False, normalize=False, patience=510):
        """
            train the gcn model, when idx_val is not None, pick the best model
            according to the validation loss
        """
        sd = self.state_dict()
        for v in sd.values():
            self.device = v.device
            break
        self.sim = None
        self.idx_test = idx_test
        self.attention = attention
        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features
            adj = adj
            labels = labels
        adj = self.add_loop_sparse(adj)
        """The normalization gonna be done in the GCNConv"""
        self.adj_norm = adj
        self.features = features
        self.labels = labels
        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        elif patience < train_iters:
            self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
        else:
            self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train], weight=None)
            loss_train.backward()
            optimizer.step()
            if verbose and i % 20 == 0:
                None
        self.eval()
        output = self.forward(self.features, self.adj_norm)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            None
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss_val = 100
        best_acc_val = 0
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            self.eval()
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())
        if verbose:
            None
        self.load_state_dict(weights)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        if verbose:
            None
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        early_stopping = patience
        best_loss_val = 100
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            self.eval()
            output = self.forward(self.features, self.adj_norm)
            if verbose and i % 10 == 0:
                None
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break
        if verbose:
            None
        self.load_state_dict(weights)

    def test(self, idx_test):
        self.eval()
        output = self.predict()
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        return acc_test, output

    def _set_parameters(self):
        pass

    def predict(self, features=None, adj=None):
        """By default, inputs are unnormalized data"""
        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)
            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)


def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class GCN4GNNGuard_attack(GCN):

    def __init__(self, nfeat, nclass, nhid, activation, dropout=0.5, lr=0.01, drop=False, weight_decay=0.0005, with_relu=True, with_bias=True, add_self_loops=True, normalize=True):
        super(GCN4GNNGuard_attack, self).__init__(nfeat, nclass, nhid, activation, dropout=dropout, add_self_loops=add_self_loops, normalize=normalize)
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.nclass = nclass
        self.nfeat = nfeat
        self.hidden_sizes = nhid
        self.drop = drop
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.gc1 = GCNConv(nfeat, nhid[0], bias=True)
        self.gc2 = GCNConv(nhid[0], nclass, bias=True)

    def forward(self, x, adj_lil):
        """we don't change the edge_index, just update the edge_weight;
        some edge_weight are regarded as removed if it equals to zero"""
        x = x.to_dense()
        adj = adj_lil.coalesce().indices()
        edge_weight = adj_lil.coalesce().values()
        x = F.relu(self.gc1(x, adj, edge_weight=edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)

    def add_loop_sparse(self, adj, fill_value=1):
        row = torch.range(0, int(adj.shape[0] - 1), dtype=torch.int64)
        i = torch.stack((row, row), dim=0)
        v = torch.ones(adj.shape[0], dtype=torch.float32)
        shape = adj.shape
        I_n = torch.sparse.FloatTensor(i, v, shape)
        return adj + I_n

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def fit(self, features, adj, labels, idx_train, idx_val=None, idx_test=None, train_iters=81, att_0=None, attention=False, model_name=None, initialize=True, verbose=False, normalize=False, patience=510):
        """
            train the gcn model, when idx_val is not None, pick the best model
            according to the validation loss
        """
        sd = self.state_dict()
        for v in sd.values():
            self.device = v.device
            break
        self.sim = None
        self.attention = attention
        if self.attention:
            att_0 = self.att_coef_1(features, adj)
            adj = att_0
            self.sim = att_0
        self.idx_test = idx_test
        if initialize:
            self.initialize()
        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features
            adj = adj
            labels = labels
        normalize = False
        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj
        """Make the coefficient D^{-1/2}(A+I)D^{-1/2}"""
        self.adj_norm = adj_norm
        self.features = features
        self.labels = labels
        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        elif patience < train_iters:
            self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
        else:
            self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train], weight=None)
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                None
        self.eval()
        output = self.forward(self.features, self.adj_norm)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            None
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss_val = 100
        best_acc_val = 0
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            acc_test = accuracy(output[self.idx_test], labels[self.idx_test])
            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            if verbose and i % 200 == 0:
                None
            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())
        if verbose:
            None
        self.load_state_dict(weights)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        if verbose:
            None
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        early_stopping = patience
        best_loss_val = 100
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            self.eval()
            output = self.forward(self.features, self.adj_norm)
            if verbose and i % 10 == 0:
                None
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break
        if verbose:
            None
        self.load_state_dict(weights)

    def test(self, idx_test):
        self.eval()
        output = self.predict()
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        None
        return acc_test, output

    def _set_parameters(self):
        pass

    def predict(self, features=None, adj=None):
        """By default, inputs are unnormalized data"""
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)
            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)


class StackedLSTMCell(nn.Module):

    def __init__(self, layers, size, bias):
        super().__init__()
        self.lstm_num_layers = layers
        self.lstm_modules = nn.ModuleList([nn.LSTMCell(size, size, bias=bias) for _ in range(self.lstm_num_layers)])

    def forward(self, inputs, hidden):
        prev_h, prev_c = hidden
        next_h, next_c = [], []
        for i, m in enumerate(self.lstm_modules):
            curr_h, curr_c = m(inputs, (prev_h[i], prev_c[i]))
            next_c.append(curr_c)
            next_h.append(curr_h)
            inputs = curr_h[-1].view(1, -1)
        return next_h, next_c


class ReinforceController(nn.Module):
    """
    A controller that mutates the graph with RL.

    Parameters
    ----------
    fields : list of ReinforceField
        List of fields to choose.
    lstm_size : int
        Controller LSTM hidden units.
    lstm_num_layers : int
        Number of layers for stacked LSTM.
    tanh_constant : float
        Logits will be equal to ``tanh_constant * tanh(logits)``. Don't use ``tanh`` if this value is ``None``.
    skip_target : float
        Target probability that skipconnect will appear.
    temperature : float
        Temperature constant that divides the logits.
    entropy_reduction : str
        Can be one of ``sum`` and ``mean``. How the entropy of multi-input-choice is reduced.
    """

    def __init__(self, fields, lstm_size=64, lstm_num_layers=1, tanh_constant=1.5, skip_target=0.4, temperature=None, entropy_reduction='sum'):
        super(ReinforceController, self).__init__()
        self.fields = fields
        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.tanh_constant = tanh_constant
        self.temperature = temperature
        self.skip_target = skip_target
        self.lstm = StackedLSTMCell(self.lstm_num_layers, self.lstm_size, False)
        self.attn_anchor = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.attn_query = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)
        self.g_emb = nn.Parameter(torch.randn(1, self.lstm_size) * 0.1)
        self.skip_targets = nn.Parameter(torch.tensor([1.0 - self.skip_target, self.skip_target]), requires_grad=False)
        assert entropy_reduction in ['sum', 'mean'], 'Entropy reduction must be one of sum and mean.'
        self.entropy_reduction = torch.sum if entropy_reduction == 'sum' else torch.mean
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.soft = nn.ModuleDict({field.name: nn.Linear(self.lstm_size, field.total, bias=False) for field in fields})
        self.embedding = nn.ModuleDict({field.name: nn.Embedding(field.total, self.lstm_size) for field in fields})

    def resample(self):
        self._initialize()
        result = dict()
        for field in self.fields:
            result[field.name] = self._sample_single(field)
        return result

    def _initialize(self):
        self._inputs = self.g_emb.data
        self._c = [torch.zeros((1, self.lstm_size), dtype=self._inputs.dtype, device=self._inputs.device) for _ in range(self.lstm_num_layers)]
        self._h = [torch.zeros((1, self.lstm_size), dtype=self._inputs.dtype, device=self._inputs.device) for _ in range(self.lstm_num_layers)]
        self.sample_log_prob = 0
        self.sample_entropy = 0
        self.sample_skip_penalty = 0

    def _lstm_next_step(self):
        self._h, self._c = self.lstm(self._inputs, (self._h, self._c))

    def _sample_single(self, field):
        self._lstm_next_step()
        logit = self.soft[field.name](self._h[-1])
        if self.temperature is not None:
            logit /= self.temperature
        if self.tanh_constant is not None:
            logit = self.tanh_constant * torch.tanh(logit)
        if field.choose_one:
            sampled = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
            log_prob = self.cross_entropy_loss(logit, sampled)
            self._inputs = self.embedding[field.name](sampled)
        else:
            logit = logit.view(-1, 1)
            logit = torch.cat([-logit, logit], 1)
            sampled = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
            skip_prob = torch.sigmoid(logit)
            kl = torch.sum(skip_prob * torch.log(skip_prob / self.skip_targets))
            self.sample_skip_penalty += kl
            log_prob = self.cross_entropy_loss(logit, sampled)
            sampled = sampled.nonzero().view(-1)
            if sampled.sum().item():
                self._inputs = (torch.sum(self.embedding[field.name](sampled.view(-1)), 0) / (1.0 + torch.sum(sampled))).unsqueeze(0)
            else:
                self._inputs = torch.zeros(1, self.lstm_size, device=self.embedding[field.name].weight.device)
        sampled = sampled.detach().numpy().tolist()
        self.sample_log_prob += self.entropy_reduction(log_prob)
        entropy = (log_prob * torch.exp(-log_prob)).detach()
        self.sample_entropy += self.entropy_reduction(entropy)
        if len(sampled) == 1:
            sampled = sampled[0]
        return sampled


class AGNNReinforceController(ReinforceController):

    def resample(self, search_fields, selection):
        self._initialize()
        result = selection.copy()
        for field in self.fields:
            if field not in search_fields:
                self._update_state(field, selection[field.name])
        for field in search_fields:
            result[field.name] = self._sample_single(field)
        return result

    def _update_state(self, field, sampled):
        self._lstm_next_step()
        self._inputs = self.embedding[field.name](torch.LongTensor([sampled]))


class AGNNActionGuider(nn.Module):

    def __init__(self, fields, groups, guide_type, **controllargs):
        super(AGNNActionGuider, self).__init__()
        controllers = [AGNNReinforceController(fields, **controllargs) for group in groups]
        self.controllers = nn.ModuleList(controllers)
        self.fields = fields
        self.groups = groups
        self.guide_type = guide_type

    def dummy_selection(self):
        result = dict()
        for field in self.fields:
            result[field.name] = 0
        return result

    def resample(self, selection):
        entropys = []
        new_selections = []
        sample_probs = []
        for idx, cont in enumerate(self.controllers):
            cont = self.controllers[idx]
            group = self.groups[idx]
            new_selection = cont.resample(group, selection)
            new_selections.append(new_selection)
            entropy = cont.sample_entropy
            entropys.append(entropy)
            sample_probs.append(cont.sample_log_prob)
        None
        if self.guide_type == 0:
            idx = np.argmax(entropys)
        elif self.guide_type == 1:
            idx = torch.multinomial(F.softmax(torch.tensor(entropys), dim=0), 1).item()
        else:
            assert False, f'Not implemented guide type {self.guide_type}'
        group = self.groups[idx]
        None
        new_selection = new_selections[idx]
        self.sample_log_prob = sample_probs[idx]
        self.sample_entropy = entropys[idx]
        None
        return new_selection


class DartsLayerChoice(nn.Module):

    def __init__(self, layer_choice):
        super(DartsLayerChoice, self).__init__()
        self.name = layer_choice.key
        self.op_choices = nn.ModuleDict(layer_choice.named_children())
        self.alpha = nn.Parameter(torch.randn(len(self.op_choices)) * 0.001)

    def forward(self, *args, **kwargs):
        op_results = torch.stack([op(*args, **kwargs) for op in self.op_choices.values()])
        alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)
        return torch.sum(op_results * F.softmax(self.alpha, -1).view(*alpha_shape), 0)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super(DartsLayerChoice, self).named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return torch.argmax(self.alpha).item()


class DartsInputChoice(nn.Module):

    def __init__(self, input_choice):
        super(DartsInputChoice, self).__init__()
        self.name = input_choice.key
        self.alpha = nn.Parameter(torch.randn(input_choice.n_candidates) * 0.001)
        self.n_chosen = input_choice.n_chosen or 1

    def forward(self, inputs):
        inputs = torch.stack(inputs)
        alpha_shape = [-1] + [1] * (len(inputs.size()) - 1)
        return torch.sum(inputs * F.softmax(self.alpha, -1).view(*alpha_shape), 0)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super(DartsInputChoice, self).named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return torch.argsort(-self.alpha).cpu().numpy().tolist()[:self.n_chosen]


class AggAdd(nn.Module):

    def __init__(self, dim, att_head, dropout=0, norm=False, skip_connect=False, *args, **kwargs):
        super(AggAdd, self).__init__()
        self.dropout = dropout
        self.ln_add = nn.BatchNorm1d(dim, track_running_stats=True, affine=True)
        self.norm = norm
        self.skip_connect = skip_connect

    def forward(self, x, edge_index, *args, **kwargs):
        norm = self.norm
        x1, x2, x3 = x[0], x[1][0], x[1][1]
        if norm:
            return self.ln_add(x1 + x2)
        else:
            return x1 + x2


class Zero(nn.Module):

    def __init__(self, indim, outdim) ->None:
        super().__init__()
        self.outdim = outdim
        self.zero = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, x, edge_index):
        return torch.zeros(x.size(0), self.outdim) * self.zero


class Identity(nn.Module):

    def forward(self, x, edge_index, edge_weight=None):
        return x

    def __repr__(self):
        return 'Identity()'


class Linear(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.core = nn.Linear(in_dim, out_dim)

    def forward(self, x, *args):
        return self.core(x)


class StrModule(nn.Module):

    def __init__(self, lambd):
        super().__init__()
        self.name = lambd

    def forward(self, *args, **kwargs):
        return self.name

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.name)


class AutoModule:

    def _initialize(self, *args, **kwargs) ->_typing.Optional[bool]:
        """ Abstract initialization method to override """
        raise NotImplementedError

    @property
    def initialized(self) ->bool:
        return self._initialized

    def initialize(self, *args, **kwargs) ->bool:
        if self._initialized:
            return self._initialized
        else:
            init_flag = self._initialize(*args, **kwargs)
            self._initialized = init_flag if isinstance(init_flag, bool) else True
            return self._initialized

    @property
    def device(self) ->torch.device:
        return self._device

    @device.setter
    def device(self, __device: '_typing.Union[torch.device, str, int, None]'):
        if type(__device) == torch.device or type(__device) == str and __device.lower() != 'auto' or type(__device) == int:
            self._device: 'torch.device' = torch.device(__device)
        else:
            self._device: 'torch.device' = torch.device('cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')

    def __init__(self, device: '_typing.Union[torch.device, str, int, None]'=..., *args, **kwargs):
        self.__hyper_parameters: '_typing.Mapping[str, _typing.Any]' = dict()
        self.__hyper_parameter_space: '_typing.Iterable[_typing.Mapping[str, _typing.Any]]' = []
        self.device = device
        self.__args: '_typing.Tuple[_typing.Any, ...]' = args
        self.__kwargs: '_typing.Mapping[str, _typing.Any]' = kwargs
        self._initialized: 'bool' = False

    @property
    def hyper_parameters(self) ->_typing.Mapping[str, _typing.Any]:
        return self.__hyper_parameters

    @hyper_parameters.setter
    def hyper_parameters(self, hp: '_typing.Mapping[str, _typing.Any]'):
        self.__hyper_parameters = hp

    @property
    def hyper_parameter_space(self) ->_typing.Iterable[_typing.Mapping[str, _typing.Any]]:
        return self.__hyper_parameter_space

    @hyper_parameter_space.setter
    def hyper_parameter_space(self, hp_space: '_typing.Iterable[_typing.Mapping[str, _typing.Any]]'):
        self.__hyper_parameter_space = hp_space


class BaseAutoModel(AutoModule):

    def __init__(self, input_dimension, output_dimension, device, **kwargs):
        self._input_dimension = input_dimension
        self._output_dimension = output_dimension
        self._model = None
        self._kwargs = kwargs
        super(BaseAutoModel, self).__init__(device)

    def to(self, device):
        self.to_device(device)

    def to_device(self, device):
        self.device = device
        if self.model is not None:
            self.model

    @property
    def model(self):
        return self._model

    def from_hyper_parameter(self, hp, **kwargs):
        kw = deepcopy(self._kwargs)
        kw.update(kwargs)
        ret_self = self.__class__(self.input_dimension, self.output_dimension, self.device, **kw)
        hp_now = dict(self.hyper_parameters)
        hp_now.update(hp)
        ret_self.hyper_parameters = hp_now
        ret_self.initialize()
        return ret_self

    @property
    def input_dimension(self):
        return self._input_dimension

    @input_dimension.setter
    def input_dimension(self, input_dimension):
        self._input_dimension = input_dimension

    @property
    def output_dimension(self):
        return self._output_dimension

    @output_dimension.setter
    def output_dimension(self, output_dimension):
        self._output_dimension = output_dimension


class FixedInputChoice(nn.Module):
    """
    Use to replace `InputChoice` Mutable in fix process

    Parameters
    ----------
    mask : list
        The mask indicating which input to choose
    """

    def __init__(self, mask):
        self.mask_len = len(mask)
        for i in range(self.mask_len):
            if mask[i]:
                self.selected = i
                break
        super().__init__()

    def forward(self, optional_inputs):
        if len(optional_inputs) == self.mask_len:
            return optional_inputs[self.selected]


class OrderedMutable:
    """
    An abstract class with order, enabling to sort mutables with a certain rank.

    Parameters
    ----------
    order : int
        The order of the mutable
    """

    def __init__(self, order):
        self.order = order


def apply_fixed_architecture(model, fixed_arc, verbose=True):
    """
    Load architecture from `fixed_arc` and apply to model.

    Parameters
    ----------
    model : torch.nn.Module
        Model with mutables.
    fixed_arc : str or dict
        Path to the JSON that stores the architecture, or dict that stores the exported architecture.
    verbose : bool
        Print log messages if set to True

    Returns
    -------
    FixedArchitecture
        Mutator that is responsible for fixes the graph.
    """
    if isinstance(fixed_arc, str):
        with open(fixed_arc) as f:
            fixed_arc = json.load(f)
    architecture = CleanFixedArchitecture(model, fixed_arc, verbose)
    architecture.reset()
    architecture.replace_all_choice()
    return architecture


def _get_mask(sampled, total):
    multihot = [(i == sampled or isinstance(sampled, list) and i in sampled) for i in range(total)]
    return torch.tensor(multihot, dtype=torch.bool)


class PathSamplingLayerChoice(nn.Module):
    """
    Mixed module, in which fprop is decided by exactly one or multiple (sampled) module.
    If multiple module is selected, the result will be sumed and returned.

    Attributes
    ----------
    sampled : int or list of int
        Sampled module indices.
    mask : tensor
        A multi-hot bool 1D-tensor representing the sampled mask.
    """

    def __init__(self, layer_choice):
        super(PathSamplingLayerChoice, self).__init__()
        self.op_names = []
        for name, module in layer_choice.named_children():
            self.add_module(name, module)
            self.op_names.append(name)
        assert self.op_names, 'There has to be at least one op to choose from.'
        self.sampled = None

    def forward(self, *args, **kwargs):
        assert self.sampled is not None, 'At least one path needs to be sampled before fprop.'
        if isinstance(self.sampled, list):
            return sum([getattr(self, self.op_names[i])(*args, **kwargs) for i in self.sampled])
        else:
            return getattr(self, self.op_names[self.sampled])(*args, **kwargs)

    def sampled_choices(self):
        if self.sampled is None:
            return []
        elif isinstance(self.sampled, list):
            return [getattr(self, self.op_names[i]) for i in self.sampled]
        else:
            return [getattr(self, self.op_names[self.sampled])]

    def __len__(self):
        return len(self.op_names)

    @property
    def mask(self):
        return _get_mask(self.sampled, len(self))

    def __repr__(self):
        return f'PathSamplingLayerChoice(op_names={self.op_names}, chosen={self.sampled})'


def count_parameters(module, only_trainable=False):
    s = sum(p.numel() for p in module.parameters(recurse=False) if not only_trainable or p.requires_grad)
    if isinstance(module, PathSamplingLayerChoice):
        s += sum(count_parameters(m) for m in module.sampled_choices())
    else:
        s += sum(count_parameters(m) for m in module.children())
    return s


def _build_random_data(device, num_feat):
    node_nums = 3000
    edge_nums = 10000


    class Data:
        pass
    data = Data()
    data.x = torch.randn((node_nums, num_feat))
    data.edge_index = torch.randint(0, node_nums, (2, edge_nums))
    data.num_features = num_feat
    return data


def measure_latency(model, num_iters=200, *, warmup_iters=50):
    device = next(model.parameters()).device
    num_feat = model.input_dim
    model.eval()
    latencys = []
    data = _build_random_data(device, num_feat)
    with torch.no_grad():
        try:
            for i in range(warmup_iters + num_iters):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start = time.time()
                model(data)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                dt = time.time() - start
                if i >= warmup_iters:
                    latencys.append(dt)
        except RuntimeError as e:
            if 'cuda' in str(e) or 'CUDA' in str(e):
                INF = 100
                return INF
            else:
                raise e
    return np.mean(latencys)


def get_hardware_aware_metric(model, hardware_metric):
    """
    Get architectures' hardware-aware metrics

    Attributes
    ----------
    model : BaseSpace
        The architecture to be evaluated
    hardware_metric : str
        The name of hardware-aware metric. Can be 'parameter' or 'latency'
    """
    if hardware_metric == 'parameter':
        return count_parameters(model)
    elif hardware_metric == 'latency':
        return measure_latency(model, 20, warmup_iters=5)
    else:
        raise ValueError('Unsupported hardware-aware metric')


def get_logger(name):
    """
    Get the logger of given name

    Parameters
    ----------
    name: str
        The name of logger

    Returns
    -------
    logger: Logger
        The logger generated
    """
    return logging.getLogger(name)


class BoxModel(BaseAutoModel):
    """
    The box wrapping a space, can be passed to later procedure or trainer

    Parameters
    ----------
    space_model : BaseSpace
        The space which should be wrapped
    device : str or torch.device
        The device to place the model
    """
    _logger = get_logger('space model')

    def __init__(self, space_model, device):
        super().__init__(None, None, device)
        self.init = True
        self.space = []
        self.hyperparams = {}
        self._model = space_model
        self.num_features = self._model.input_dim
        self.num_classes = self._model.output_dim
        self.params = {'num_class': self.num_classes, 'features_num': self.num_features}
        self.selection = None

    def _initialize(self):
        return True

    def fix(self, selection):
        """
        To fix self._model with a selection

        Parameters
        ----------
        selection : dict
            A seletion indicating the choices of mutables
        """
        self.selection = selection
        self._model.instantiate()
        apply_fixed_architecture(self._model, selection, verbose=False)
        return self

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def from_hyper_parameter(self, hp):
        """
        receive no hp, just copy self and reset the learnable parameters.
        """
        ret_self = deepcopy(self)
        ret_self._model.instantiate()
        if ret_self.selection:
            apply_fixed_architecture(ret_self._model, ret_self.selection, verbose=False)
        return ret_self

    def __repr__(self) ->str:
        return str({'parameter': get_hardware_aware_metric(self.model, 'parameter'), 'model': self.model, 'selection': self.selection})


class BaseSpace(nn.Module):
    """
    Base space class of NAS module. Defining space containing all models.
    Please use mutables to define your whole space. Refer to
    `https://nni.readthedocs.io/en/stable/NAS/WriteSearchSpace.html`
    for detailed information.

    Parameters
    ----------
    init: bool
        Whether to initialize the whole space. Default: `False`
    """

    def __init__(self):
        super().__init__()
        self._initialized = False

    @abstractmethod
    def _instantiate(self):
        """
        Instantiate modules in the space
        """
        raise NotImplementedError()

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Define the forward pass of space model
        """
        raise NotImplementedError()

    @abstractmethod
    def parse_model(self, selection: 'dict', device) ->BaseAutoModel:
        """
        Export the searched model from space.

        Parameters
        ----------
        selection: Dict
            The dictionary containing all the choices of nni.
        device: str or torch.device
            The device to put model on.

        Return
        ------
        model: autogl.module.model.BaseModel
            model to be exported.
        """
        raise NotImplementedError()

    def instantiate(self):
        """
        Instantiate the space, reset default key for the mutables here/
        """
        self._default_key = 0
        if not self._initialized:
            self._initialized = True

    def setLayerChoice(self, order, op_candidates, reduction='sum', return_mask=False, key=None):
        """
        Give a unique key if not given
        """
        orikey = key
        if orikey == None:
            key = f'default_key_{self._default_key}'
            self._default_key += 1
            orikey = key
        layer = OrderedLayerChoice(order, op_candidates, reduction, return_mask, orikey)
        return layer

    def setInputChoice(self, order, n_candidates=None, choose_from=None, n_chosen=None, reduction='sum', return_mask=False, key=None):
        """
        Give a unique key if not given
        """
        orikey = key
        if orikey == None:
            key = f'default_key_{self._default_key}'
            self._default_key += 1
            orikey = key
        layer = OrderedInputChoice(order, n_candidates, choose_from, n_chosen, reduction, return_mask, orikey)
        return layer

    def wrap(self):
        """
        Return a BoxModel which wrap self as a model
        Used to pass to trainer
        To use this function, must contain `input_dim` and `output_dim`
        """
        device = next(self.parameters()).device
        return BoxModel(self, device)


class LinearConv(nn.Module):

    def __init__(self, in_channels, out_channels, bias=True):
        super(LinearConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = torch.nn.Linear(in_channels, out_channels, bias)

    def forward(self, x, edge_index, edge_weight=None):
        return self.linear(x)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class SkipConv(Module):

    def __init__(self, in_channels, out_channels, bias=True):
        super(SkipConv, self).__init__()
        self.out_dim = out_channels

    def forward(self, x, edge_index, edge_weight=None):
        return x

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class ZeroConv(nn.Module):

    def forward(self, x, edge_index, edge_weight=None):
        out = torch.zeros_like(x)
        out.requires_grad = True
        return out

    def __repr__(self):
        return 'ZeroConv()'


gnn_list = ['gat', 'gcn', 'gin', 'cheb', 'sage', 'arma', 'graph', 'fc', 'skip']


class ARMAConv(nn.Module):

    def __init__(self, in_dim, out_dim, num_stacks=1, num_layers=1, activation=None, dropout=0.0, bias=True):
        super(ARMAConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K = num_stacks
        self.T = num_layers
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.w_0 = nn.ModuleDict({str(k): nn.Linear(in_dim, out_dim, bias=False) for k in range(self.K)})
        self.w = nn.ModuleDict({str(k): nn.Linear(out_dim, out_dim, bias=False) for k in range(self.K)})
        self.v = nn.ModuleDict({str(k): nn.Linear(in_dim, out_dim, bias=False) for k in range(self.K)})
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.K, self.T, 1, self.out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        for k in range(self.K):
            glorot(self.w_0[str(k)].weight)
            glorot(self.w[str(k)].weight)
            glorot(self.v[str(k)].weight)
        zeros(self.bias)

    def forward(self, g, feats):
        with g.local_scope():
            init_feats = feats
            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).unsqueeze(1)
            output = None
            for k in range(self.K):
                feats = init_feats
                for t in range(self.T):
                    feats = feats * norm
                    g.ndata['h'] = feats
                    g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                    feats = g.ndata.pop('h')
                    feats = feats * norm
                    if t == 0:
                        feats = self.w_0[str(k)](feats)
                    else:
                        feats = self.w[str(k)](feats)
                    feats += self.dropout(self.v[str(k)](init_feats))
                    feats += self.v[str(k)](self.dropout(init_feats))
                    if self.bias is not None:
                        feats += self.bias[k][t]
                    if self.activation is not None:
                        feats = self.activation(feats)
                if output is None:
                    output = feats
                else:
                    output += feats
            return output / self.K


class ChebConv(MessagePassing):
    """The chebyshev spectral graph convolutional operator from the
    `"Convolutional Neural Networks on Graphs with Fast Localized Spectral
    Filtering" <https://arxiv.org/abs/1606.09375>`_ paper

    .. math::
        \\mathbf{X}^{\\prime} = \\sum_{k=1}^{K} \\mathbf{Z}^{(k)} \\cdot
        \\mathbf{\\Theta}^{(k)}

    where :math:`\\mathbf{Z}^{(k)}` is computed recursively by

    .. math::
        \\mathbf{Z}^{(1)} &= \\mathbf{X}

        \\mathbf{Z}^{(2)} &= \\mathbf{\\hat{L}} \\cdot \\mathbf{X}

        \\mathbf{Z}^{(k)} &= 2 \\cdot \\mathbf{\\hat{L}} \\cdot
        \\mathbf{Z}^{(k-1)} - \\mathbf{Z}^{(k-2)}

    and :math:`\\mathbf{\\hat{L}}` denotes the scaled and normalized Laplacian
    :math:`\\frac{2\\mathbf{L}}{\\lambda_{\\max}} - \\mathbf{I}`.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\\mathbf{L} = \\mathbf{D} - \\mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\\mathbf{L} = \\mathbf{I} - \\mathbf{D}^{-1/2} \\mathbf{A}
            \\mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\\mathbf{L} = \\mathbf{I} - \\mathbf{D}^{-1} \\mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, K, normalization='sym', bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(ChebConv, self).__init__(**kwargs)
        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def __norm__(self, edge_index, num_nodes: 'Optional[int]', edge_weight: 'OptTensor', normalization: 'Optional[str]', lambda_max, dtype: 'Optional[int]'=None, batch: 'OptTensor'=None):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = get_laplacian(edge_index, edge_weight, normalization, dtype, num_nodes)
        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]
        edge_weight = 2.0 * edge_weight / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=-1.0, num_nodes=num_nodes)
        assert edge_weight is not None
        return edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight: 'OptTensor'=None, batch: 'OptTensor'=None, lambda_max: 'OptTensor'=None):
        """"""
        if self.normalization != 'sym' and lambda_max is None:
            raise ValueError('You need to pass `lambda_max` to `forward() in`case the normalization is non-symmetric.')
        if lambda_max is None:
            lambda_max = torch.tensor(2.0, dtype=x.dtype, device=x.device)
        if not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=x.dtype, device=x.device)
        assert lambda_max is not None
        edge_index, norm = self.__norm__(edge_index, x.size(self.node_dim), edge_weight, self.normalization, lambda_max, dtype=x.dtype, batch=batch)
        Tx_0 = x
        Tx_1 = x
        out = torch.matmul(Tx_0, self.weight[0])
        if self.weight.size(0) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm, size=None)
            out = out + torch.matmul(Tx_1, self.weight[1])
        for k in range(2, self.weight.size(0)):
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
            Tx_2 = 2.0 * Tx_2 - Tx_0
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2
        if self.bias is not None:
            out += self.bias
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.weight.size(0), self.normalization)


def gnn_map(gnn_name, in_dim, out_dim, concat=False, bias=True) ->nn.Module:
    """

    :param gnn_name:
    :param in_dim:
    :param out_dim:
    :param concat: for gat, concat multi-head output or not
    :return: GNN model
    """
    if gnn_name == 'gat_8':
        return GATConv(in_dim, out_dim, 8, concat=concat, bias=bias)
    elif gnn_name == 'gat_6':
        return GATConv(in_dim, out_dim, 6, concat=concat, bias=bias)
    elif gnn_name == 'gat_4':
        return GATConv(in_dim, out_dim, 4, concat=concat, bias=bias)
    elif gnn_name == 'gat_2':
        return GATConv(in_dim, out_dim, 2, concat=concat, bias=bias)
    elif gnn_name in ['gat_1', 'gat']:
        return GATConv(in_dim, out_dim, 1, concat=concat, bias=bias)
    elif gnn_name == 'gcn':
        return GCNConv(in_dim, out_dim)
    elif gnn_name == 'cheb':
        return ChebConv(in_dim, out_dim, K=2, bias=bias)
    elif gnn_name == 'sage':
        return SAGEConv(in_dim, out_dim, bias=bias)
    elif gnn_name == 'gated':
        return GatedGraphConv(in_dim, out_dim, bias=bias)
    elif gnn_name == 'arma':
        return ARMAConv(in_dim, out_dim, bias=bias)
    elif gnn_name == 'sg':
        return SGConv(in_dim, out_dim, bias=bias)
    elif gnn_name == 'linear':
        return LinearConv(in_dim, out_dim, bias=bias)
    elif gnn_name == 'zero':
        return ZeroConv()
    elif gnn_name == 'identity':
        return Identity()
    elif hasattr(torch_geometric.nn, gnn_name):
        cls = getattr(torch_geometric.nn, gnn_name)
        assert isinstance(cls, type), 'Only support modules, get %s' % gnn_name
        kwargs = {'in_channels': in_dim, 'out_channels': out_dim, 'concat': concat, 'bias': bias}
        kwargs = {key: kwargs[key] for key in cls.__init__.__code__.co_varnames if key in kwargs}
        return cls(**kwargs)
    raise KeyError('Cannot parse key %s' % gnn_name)


class MixedOp(nn.Module):

    def __init__(self, in_c, out_c):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for action in gnn_list:
            self._ops.append(gnn_map(action, in_c, out_c))

    def forward(self, x, edge_index, edge_weight, weights, selected_idx=None):
        if selected_idx is None:
            fin = []
            for w, op, op_name in zip(weights, self._ops, gnn_list):
                """if op_name == "gcn":
                    w = 1.0
                else:
                    continue"""
                if edge_weight == None:
                    fin.append(w * op(x, edge_index))
                else:
                    fin.append(w * op(x, edge_index, edge_weight=edge_weight))
            return sum(fin)
        else:
            return self._ops[selected_idx](x, edge_index)


def Get_edges(adjs):
    edges = []
    edges_weights = []
    for adj in adjs:
        edges.append(adj[0])
        edges_weights.append(torch.sigmoid(adj[1]))
    return edges, edges_weights


class CellWS(nn.Module):

    def __init__(self, steps, his_dim, hidden_dim, out_dim, dp, bias=True):
        super(CellWS, self).__init__()
        self.steps = steps
        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        self.use2 = False
        self.dp = 0.8
        for i in range(self.steps):
            if i == 0:
                inpdim = his_dim
            else:
                inpdim = hidden_dim
            if i == self.steps - 1:
                oupdim = out_dim
            else:
                oupdim = hidden_dim
            op = MixedOp(inpdim, oupdim)
            self._ops.append(op)
            self._bns.append(nn.BatchNorm1d(oupdim))

    def forward(self, x, adjs, weights):
        edges, ews = Get_edges(adjs)
        for i in range(self.steps):
            if i > 0:
                x = F.relu(x)
                x = F.dropout(x, p=self.dp, training=self.training)
            x = self._ops[i](x, edges[i], ews[i], weights[i])
        return x


class GassoSpace(BaseSpace):

    def __init__(self, hidden_dim: '_typ.Optional[int]'=64, layer_number: '_typ.Optional[int]'=2, dropout: '_typ.Optional[float]'=0.8, input_dim: '_typ.Optional[int]'=None, output_dim: '_typ.Optional[int]'=None, ops: '_typ.Tuple'=gnn_list):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.steps = layer_number
        self.dropout = dropout
        self.ops = ops
        self.use_forward = True
        self.dead_tensor = torch.nn.Parameter(torch.FloatTensor([1]), requires_grad=True)

    def instantiate(self, hidden_dim: '_typ.Optional[int]'=64, layer_number: '_typ.Optional[int]'=2, dropout: '_typ.Optional[float]'=0.8, input_dim: '_typ.Optional[int]'=None, output_dim: '_typ.Optional[int]'=None, ops: '_typ.Tuple'=gnn_list):
        super().instantiate()
        self.input_dim = input_dim or self.input_dim
        self.output_dim = output_dim or self.output_dim
        self.hidden_dim = hidden_dim or self.hidden_dim
        self.steps = layer_number or self.steps
        self.dropout = dropout or self.dropout
        self.ops = ops or self.ops
        his_dim, cur_dim, hidden_dim, out_dim = self.input_dim, self.input_dim, self.hidden_dim, self.hidden_dim
        self.cells = nn.ModuleList()
        self.cell = CellWS(self.steps, his_dim, hidden_dim, self.output_dim, self.dropout)
        his_dim = cur_dim
        cur_dim = self.steps * out_dim
        self.classifier = nn.Linear(cur_dim, self.output_dim)
        self.initialize_alphas()

    def forward(self, data):
        if self.use_forward:
            x, adjs = data.x, data.adj
            x = F.dropout(x, p=self.dropout, training=self.training)
            weights = []
            for j in range(self.steps):
                weights.append(F.softmax(self.alphas_normal[j], dim=-1))
            x = self.cell(x, adjs, weights)
            x = F.log_softmax(x, dim=1)
            self.current_pred = x.detach()
            return x
        else:
            x = self.prediction + self.dead_tensor * 0
            return x

    def keep_prediction(self):
        self.prediction = self.current_pred
    """def to(self, *args, **kwargs):
        fin = super().to(*args, **kwargs)
        device = next(fin.parameters()).device
        fin.alphas_normal = [i.to(device) for i in self.alphas_normal]
        return fin"""

    def initialize_alphas(self):
        num_ops = len(self.ops)
        self.alphas_normal = []
        for i in range(self.steps):
            self.alphas_normal.append(Variable(0.001 * torch.randn(num_ops), requires_grad=True))
        self._arch_parameters = [self.alphas_normal]

    def arch_parameters(self):
        return self.alphas_normal

    def parse_model(self, selection, device) ->BaseAutoModel:
        self.use_forward = False
        return self.wrap()


class LambdaModule(nn.Module):

    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.lambd)


GRAPHNAS_DEFAULT_ACT_OPS = ['sigmoid', 'tanh', 'relu', 'linear', 'elu']


GRAPHNAS_DEFAULT_CON_OPS = ['add', 'product', 'concat']


GRAPHNAS_DEFAULT_GNN_OPS = ['gat_8', 'gat_6', 'gat_4', 'gat_2', 'gat_1', 'gcn', 'cheb', 'sage', 'arma', 'sg', 'linear', 'zero']


def act_map(act):
    if act == 'linear':
        return lambda x: x
    elif act == 'elu':
        return F.elu
    elif act == 'sigmoid':
        return torch.sigmoid
    elif act == 'tanh':
        return torch.tanh
    elif act == 'relu':
        return torch.nn.functional.relu
    elif act == 'relu6':
        return torch.nn.functional.relu6
    elif act == 'softplus':
        return torch.nn.functional.softplus
    elif act == 'leaky_relu':
        return torch.nn.functional.leaky_relu
    else:
        raise Exception('wrong activate function')


def act_map_nn(act):
    return LambdaModule(act_map(act))


def map_nn(l):
    return [StrModule(x) for x in l]


class GraphNasNodeClassificationSpace(BaseSpace):

    def __init__(self, hidden_dim: '_typ.Optional[int]'=64, layer_number: '_typ.Optional[int]'=2, dropout: '_typ.Optional[float]'=0.9, input_dim: '_typ.Optional[int]'=None, output_dim: '_typ.Optional[int]'=None, gnn_ops: '_typ.Sequence[_typ.Union[str, _typ.Any]]'=GRAPHNAS_DEFAULT_GNN_OPS, act_ops: '_typ.Sequence[_typ.Union[str, _typ.Any]]'=GRAPHNAS_DEFAULT_ACT_OPS, con_ops: '_typ.Sequence[_typ.Union[str, _typ.Any]]'=GRAPHNAS_DEFAULT_CON_OPS):
        super().__init__()
        self.layer_number = layer_number
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gnn_ops = gnn_ops
        self.act_ops = act_ops
        self.con_ops = con_ops
        self.dropout = dropout

    def instantiate(self, hidden_dim: '_typ.Optional[int]'=None, layer_number: '_typ.Optional[int]'=None, dropout: '_typ.Optional[float]'=None, input_dim: '_typ.Optional[int]'=None, output_dim: '_typ.Optional[int]'=None, gnn_ops: '_typ.Sequence[_typ.Union[str, _typ.Any]]'=None, act_ops: '_typ.Sequence[_typ.Union[str, _typ.Any]]'=None, con_ops: '_typ.Sequence[_typ.Union[str, _typ.Any]]'=None):
        super().instantiate()
        self.dropout = dropout or self.dropout
        self.hidden_dim = hidden_dim or self.hidden_dim
        self.layer_number = layer_number or self.layer_number
        self.input_dim = input_dim or self.input_dim
        self.output_dim = output_dim or self.output_dim
        self.gnn_ops = gnn_ops or self.gnn_ops
        self.act_ops = act_ops or self.act_ops
        self.con_ops = con_ops or self.con_ops
        self.preproc0 = nn.Linear(self.input_dim, self.hidden_dim)
        self.preproc1 = nn.Linear(self.input_dim, self.hidden_dim)
        node_labels = [mutables.InputChoice.NO_KEY, mutables.InputChoice.NO_KEY]
        for layer in range(2, self.layer_number + 2):
            node_labels.append(f'op_{layer}')
            setattr(self, f'in_{layer}', self.setInputChoice(layer, choose_from=node_labels[:-1], n_chosen=1, return_mask=False, key=f'in_{layer}'))
            setattr(self, f'op_{layer}', self.setLayerChoice(layer, [gnn_map(op, self.hidden_dim, self.hidden_dim) for op in self.gnn_ops], key=f'op_{layer}'))
        setattr(self, 'act', self.setLayerChoice(2 * layer, [act_map_nn(a) for a in self.act_ops], key='act'))
        if len(self.con_ops) > 1:
            setattr(self, 'concat', self.setLayerChoice(2 * layer + 1, map_nn(self.con_ops), key='concat'))
        self._initialized = True
        self.classifier1 = nn.Linear(self.hidden_dim * self.layer_number, self.output_dim)
        self.classifier2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, data):
        x = bk_feat(data)
        x = F.dropout(x, p=self.dropout, training=self.training)
        pprev_, prev_ = self.preproc0(x), self.preproc1(x)
        prev_nodes_out = [pprev_, prev_]
        for layer in range(2, self.layer_number + 2):
            node_in = getattr(self, f'in_{layer}')(prev_nodes_out)
            op = getattr(self, f'op_{layer}')
            node_out = bk_gconv(op, data, node_in)
            prev_nodes_out.append(node_out)
        act = getattr(self, 'act')
        if len(self.con_ops) > 1:
            con = getattr(self, 'concat')()
        elif len(self.con_ops) == 1:
            con = self.con_ops[0]
        else:
            con = 'concat'
        states = prev_nodes_out
        if con == 'concat':
            x = torch.cat(states[2:], dim=1)
        else:
            tmp = states[2]
            for i in range(3, len(states)):
                if con == 'add':
                    tmp = torch.add(tmp, states[i])
                elif con == 'product':
                    tmp = torch.mul(tmp, states[i])
            x = tmp
        x = act(x)
        if con == 'concat':
            x = self.classifier1(x)
        else:
            x = self.classifier2(x)
        return F.log_softmax(x, dim=1)

    def parse_model(self, selection, device) ->BaseAutoModel:
        return self.wrap().fix(selection)


class GeoLayer(MessagePassing):

    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0, bias=True, att_type='gat', agg_type='sum', pool_dim=0):
        if agg_type in ['sum', 'mlp']:
            super(GeoLayer, self).__init__('add')
        elif agg_type in ['mean', 'max']:
            super(GeoLayer, self).__init__(agg_type)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.att_type = att_type
        self.agg_type = agg_type
        self.gcn_weight = None
        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        if self.att_type in ['generalized_linear']:
            self.general_att_layer = torch.nn.Linear(out_channels, 1, bias=False)
        if self.agg_type in ['mean', 'max', 'mlp']:
            if pool_dim <= 0:
                pool_dim = 128
        self.pool_dim = pool_dim
        if pool_dim != 0:
            self.pool_layer = torch.nn.ModuleList()
            self.pool_layer.append(torch.nn.Linear(self.out_channels, self.pool_dim))
            self.pool_layer.append(torch.nn.Linear(self.pool_dim, self.out_channels))
        else:
            pass
        self.reset_parameters()

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)
        if self.att_type in ['generalized_linear']:
            glorot(self.general_att_layer.weight)
        if self.pool_dim != 0:
            for layer in self.pool_layer:
                glorot(layer.weight)
                zeros(layer.bias)

    def forward(self, x, edge_index):
        """"""
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        return self.propagate(edge_index, x=x, num_nodes=x.size(0))

    def message(self, x_i, x_j, edge_index, num_nodes):
        if self.att_type == 'const':
            if self.training and self.dropout > 0:
                x_j = F.dropout(x_j, p=self.dropout, training=True)
            neighbor = x_j
        elif self.att_type == 'gcn':
            if self.gcn_weight is None or self.gcn_weight.size(0) != x_j.size(0):
                _, norm = self.norm(edge_index, num_nodes, None)
                self.gcn_weight = norm
            neighbor = self.gcn_weight.view(-1, 1, 1) * x_j
        else:
            alpha = self.apply_attention(edge_index, num_nodes, x_i, x_j)
            alpha = softmax(alpha, edge_index[0], num_nodes=num_nodes)
            if self.training and self.dropout > 0:
                alpha = F.dropout(alpha, p=self.dropout, training=True)
            neighbor = x_j * alpha.view(-1, self.heads, 1)
        if self.pool_dim > 0:
            for layer in self.pool_layer:
                neighbor = layer(neighbor)
        return neighbor

    def apply_attention(self, edge_index, num_nodes, x_i, x_j):
        if self.att_type == 'gat':
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
        elif self.att_type == 'gat_sym':
            wl = self.att[:, :, :self.out_channels]
            wr = self.att[:, :, self.out_channels:]
            alpha = (x_i * wl).sum(dim=-1) + (x_j * wr).sum(dim=-1)
            alpha_2 = (x_j * wl).sum(dim=-1) + (x_i * wr).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope) + F.leaky_relu(alpha_2, self.negative_slope)
        elif self.att_type == 'linear':
            wl = self.att[:, :, :self.out_channels]
            wr = self.att[:, :, self.out_channels:]
            al = x_j * wl
            ar = x_j * wr
            alpha = al.sum(dim=-1) + ar.sum(dim=-1)
            alpha = torch.tanh(alpha)
        elif self.att_type == 'cos':
            wl = self.att[:, :, :self.out_channels]
            wr = self.att[:, :, self.out_channels:]
            alpha = x_i * wl * x_j * wr
            alpha = alpha.sum(dim=-1)
        elif self.att_type == 'generalized_linear':
            wl = self.att[:, :, :self.out_channels]
            wr = self.att[:, :, self.out_channels:]
            al = x_i * wl
            ar = x_j * wr
            alpha = al + ar
            alpha = torch.tanh(alpha)
            alpha = self.general_att_layer(alpha)
        else:
            raise Exception('Wrong attention type:', self.att_type)
        return alpha

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.heads)

    def get_param_dict(self):
        params = {}
        key = f'{self.att_type}_{self.agg_type}_{self.in_channels}_{self.out_channels}_{self.heads}'
        weight_key = key + '_weight'
        att_key = key + '_att'
        agg_key = key + '_agg'
        bais_key = key + '_bais'
        params[weight_key] = self.weight
        params[att_key] = self.att
        params[bais_key] = self.bias
        if hasattr(self, 'pool_layer'):
            params[agg_key] = self.pool_layer.state_dict()
        return params

    def load_param(self, params):
        key = f'{self.att_type}_{self.agg_type}_{self.in_channels}_{self.out_channels}_{self.heads}'
        weight_key = key + '_weight'
        att_key = key + '_att'
        agg_key = key + '_agg'
        bais_key = key + '_bais'
        if weight_key in params:
            self.weight = params[weight_key]
        if att_key in params:
            self.att = params[att_key]
        if bais_key in params:
            self.bias = params[bais_key]
        if agg_key in params and hasattr(self, 'pool_layer'):
            self.pool_layer.load_state_dict(params[agg_key])


class GraphNet(BaseSpace):

    def __init__(self, actions, num_feat, num_label, drop_out=0.6, multi_label=False, batch_normal=True, state_num=5, residual=False, layers=2):
        self.residual = residual
        self.batch_normal = batch_normal
        self.layer_nums = layers
        self.multi_label = multi_label
        self.num_feat = num_feat
        self.num_label = num_label
        self.input_dim = num_feat
        self.output_dim = num_label
        self.dropout = drop_out
        super().__init__()
        self.build_model(actions, batch_normal, drop_out, num_feat, num_label, state_num)

    def build_model(self, actions, batch_normal, drop_out, num_feat, num_label, state_num):
        if self.residual:
            self.fcs = torch.nn.ModuleList()
        if self.batch_normal:
            self.bns = torch.nn.ModuleList()
        self.layers = torch.nn.ModuleList()
        self.acts = []
        self.gates = torch.nn.ModuleList()
        self.build_hidden_layers(actions, batch_normal, drop_out, self.layer_nums, num_feat, num_label, state_num)

    def build_hidden_layers(self, actions, batch_normal, drop_out, layer_nums, num_feat, num_label, state_num=6):
        for i in range(layer_nums):
            if i == 0:
                in_channels = num_feat
            else:
                in_channels = out_channels * head_num
            attention_type = actions[i * state_num + 0]
            aggregator_type = actions[i * state_num + 1]
            act = actions[i * state_num + 2]
            head_num = actions[i * state_num + 3]
            out_channels = actions[i * state_num + 4]
            concat = True
            if i == layer_nums - 1:
                concat = False
            if self.batch_normal:
                self.bns.append(torch.nn.BatchNorm1d(in_channels, momentum=0.5))
            self.layers.append(GeoLayer(in_channels, out_channels, head_num, concat, dropout=self.dropout, att_type=attention_type, agg_type=aggregator_type))
            self.acts.append(act_map(act))
            if self.residual:
                if concat:
                    self.fcs.append(torch.nn.Linear(in_channels, out_channels * head_num))
                else:
                    self.fcs.append(torch.nn.Linear(in_channels, out_channels))

    def forward(self, data):
        output = bk_feat(data)
        if self.residual:
            for i, (act, layer, fc) in enumerate(zip(self.acts, self.layers, self.fcs)):
                output = F.dropout(output, p=self.dropout, training=self.training)
                if self.batch_normal:
                    output = self.bns[i](output)
                output = act(bk_gconv(layer, data, output) + fc(output))
        else:
            for i, (act, layer) in enumerate(zip(self.acts, self.layers)):
                output = F.dropout(output, p=self.dropout, training=self.training)
                if self.batch_normal:
                    output = self.bns[i](output)
                output = act(bk_gconv(layer, data, output))
        if not self.multi_label:
            output = F.log_softmax(output, dim=1)
        return output

    def __repr__(self):
        result_lines = ''
        for each in self.layers:
            result_lines += str(each)
        return result_lines

    @staticmethod
    def merge_param(old_param, new_param, update_all):
        for key in new_param:
            if update_all or key not in old_param:
                old_param[key] = new_param[key]
        return old_param

    def get_param_dict(self, old_param=None, update_all=True):
        if old_param is None:
            result = {}
        else:
            result = old_param
        for i in range(self.layer_nums):
            key = 'layer_%d' % i
            new_param = self.layers[i].get_param_dict()
            if key in result:
                new_param = self.merge_param(result[key], new_param, update_all)
                result[key] = new_param
            else:
                result[key] = new_param
        if self.residual:
            for i, fc in enumerate(self.fcs):
                key = f'layer_{i}_fc_{fc.weight.size(0)}_{fc.weight.size(1)}'
                result[key] = self.fcs[i]
        if self.batch_normal:
            for i, bn in enumerate(self.bns):
                key = f'layer_{i}_fc_{bn.weight.size(0)}'
                result[key] = self.bns[i]
        return result

    def load_param(self, param):
        if param is None:
            return
        for i in range(self.layer_nums):
            self.layers[i].load_param(param['layer_%d' % i])
        if self.residual:
            for i, fc in enumerate(self.fcs):
                key = f'layer_{i}_fc_{fc.weight.size(0)}_{fc.weight.size(1)}'
                if key in param:
                    self.fcs[i] = param[key]
        if self.batch_normal:
            for i, bn in enumerate(self.bns):
                key = f'layer_{i}_fc_{bn.weight.size(0)}'
                if key in param:
                    self.bns[i] = param[key]


class GraphNasMacroNodeClassificationSpace(BaseSpace):

    def __init__(self, hidden_dim: '_typ.Optional[int]'=64, layer_number: '_typ.Optional[int]'=2, dropout: '_typ.Optional[float]'=0.6, input_dim: '_typ.Optional[int]'=None, output_dim: '_typ.Optional[int]'=None, ops: '_typ.Tuple'=None, search_act_con=False):
        super().__init__()
        self.layer_number = layer_number
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ops = ops
        self.dropout = dropout
        self.search_act_con = search_act_con

    def instantiate(self, hidden_dim: '_typ.Optional[int]'=None, layer_number: '_typ.Optional[int]'=None, input_dim: '_typ.Optional[int]'=None, output_dim: '_typ.Optional[int]'=None, ops: '_typ.Tuple'=None, dropout=None):
        super().instantiate()
        self.hidden_dim = hidden_dim or self.hidden_dim
        self.layer_number = layer_number or self.layer_number
        self.input_dim = input_dim or self.input_dim
        self.output_dim = output_dim or self.output_dim
        self.ops = ops or self.ops
        self.dropout = dropout or self.dropout
        num_feat = self.input_dim
        num_label = self.output_dim
        layer_nums = self.layer_number
        state_num = 5
        for i in range(layer_nums):
            setattr(self, f'attention_{i}', self.setLayerChoice(i * state_num + 0, map_nn(['gat', 'gcn', 'cos', 'const', 'gat_sym', 'linear', 'generalized_linear']), key=f'attention_{i}'))
            setattr(self, f'aggregator_{i}', self.setLayerChoice(i * state_num + 1, map_nn(['sum', 'mean', 'max', 'mlp']), key=f'aggregator_{i}'))
            setattr(self, f'act_{i}', self.setLayerChoice(i * state_num + 0, map_nn(['sigmoid', 'tanh', 'relu', 'linear', 'softplus', 'leaky_relu', 'relu6', 'elu']), key=f'act_{i}'))
            setattr(self, f'head_{i}', self.setLayerChoice(i * state_num + 0, map_nn([1, 2, 4, 6, 8, 16]), key=f'head_{i}'))
            if i < layer_nums - 1:
                setattr(self, f'out_channels_{i}', self.setLayerChoice(i * state_num + 0, map_nn([4, 8, 16, 32, 64, 128, 256]), key=f'out_channels_{i}'))

    def parse_model(self, selection, device) ->BaseAutoModel:
        sel_list = []
        for i in range(self.layer_number):
            sel_list.append(['gat', 'gcn', 'cos', 'const', 'gat_sym', 'linear', 'generalized_linear'][selection[f'attention_{i}']])
            sel_list.append(['sum', 'mean', 'max', 'mlp'][selection[f'aggregator_{i}']])
            sel_list.append(['sigmoid', 'tanh', 'relu', 'linear', 'softplus', 'leaky_relu', 'relu6', 'elu'][selection[f'act_{i}']])
            sel_list.append([1, 2, 4, 6, 8, 16][selection[f'head_{i}']])
            if i < self.layer_number - 1:
                sel_list.append([4, 8, 16, 32, 64, 128, 256][selection[f'out_channels_{i}']])
        sel_list.append(self.output_dim)
        model = GraphNet(sel_list, self.input_dim, self.output_dim, self.dropout, multi_label=False, batch_normal=False, layers=self.layer_number).wrap()
        return model


class RobustIdentity(nn.Module):

    def __init__(self):
        super(RobustIdentity, self).__init__()

    def forward(self, edge_index, edge_weight, features):
        return edge_weight

    def check_dense_matrix(self, symmetric=False):
        torch.cuda.empty_cache()
        self.modified_adj = torch.clamp(self.modified_adj, min=0, max=1)
        torch.cuda.empty_cache()


def cal_svd(sp_adj, k):
    adj = sp_adj.asfptype()
    U, S, V = sp.linalg.svds(adj, k=k)
    diag_S = np.diag(S)
    return U @ diag_S @ V


class SVD(RobustIdentity):

    def __init__(self):
        super(SVD, self).__init__()

    def forward(self, edge_index, edge_weight, features, k=20):
        torch.cuda.empty_cache()
        device = edge_index.device
        i = edge_index.cpu().numpy()
        sp_A = sp.csr_matrix((edge_weight.detach().cpu().numpy(), (i[0], i[1])), shape=(features.size(0), features.size(0)))
        row, col = sp_A.tocoo().row, sp_A.tocoo().col
        modified_adj = cal_svd(sp_A, k=k)
        adj_values = torch.tensor(modified_adj[row, col], dtype=torch.float32)
        adj_values = torch.clamp(adj_values, min=0)
        return adj_values


def torch_sparse_to_scipy_sparse(m, return_iv=False):
    """
    Parameter:
    ----------
    m: torch.sparse matrix
    """
    i = m.coalesce().indices().detach().cpu().numpy()
    v = m.coalesce().values().detach().cpu().numpy()
    shape = m.coalesce().size()
    sp_m = sp.csr_matrix((v, (i[0], i[1])), shape=shape)
    if return_iv:
        return sp_m, i, v
    else:
        return sp_m


class Jaccard(RobustIdentity):

    def __init__(self):
        super(Jaccard, self).__init__()

    def forward(self, edge_index, edge_weight, features, threshold=0.01):
        torch.cuda.empty_cache()
        """Drop dissimilar edges.(Faster version using numba)
        """
        self.threshold = threshold
        features = features.detach().cpu().numpy()
        self.binary_feature = (features[features > 0] == 1).sum() == len(features[features > 0])
        if sp.issparse(features):
            features = features.todense().A
        _A = torch.sparse.FloatTensor(edge_index, edge_weight, (features.shape[0], features.shape[0]))
        adj = torch_sparse_to_scipy_sparse(_A)
        adj_triu = sp.triu(adj, format='csr')
        if self.binary_feature:
            removed_cnt = dropedge_jaccard(adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=self.threshold)
        else:
            removed_cnt = dropedge_cosine(adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=self.threshold)
        modified_adj = adj_triu + adj_triu.transpose()
        row, col = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
        adj_values = torch.tensor(modified_adj.toarray()[row, col], dtype=torch.float32)
        return adj_values

    def drop_dissimilar_edges(self, features, adj):
        """Drop dissimilar edges. (Slower version)
        """
        edges = np.array(adj.nonzero().detach().cpu()).T
        removed_cnt = 0
        for edge in edges:
            n1 = edge[0]
            n2 = edge[1]
            if n1 > n2:
                continue
            if self.binary_feature:
                J = self._jaccard_similarity(features[n1], features[n2])
                if J < self.threshold:
                    adj[n1, n2] = 0
                    adj[n2, n1] = 0
                    removed_cnt += 1
            else:
                C = self._cosine_similarity(features[n1], features[n2])
                if C < self.threshold:
                    adj[n1, n2] = 0
                    adj[n2, n1] = 0
                    removed_cnt += 1
        None
        return adj

    def _jaccard_similarity(self, a, b):
        intersection = np.count_nonzero(np.multiply(a, b))
        if np.count_nonzero(a) + np.count_nonzero(b) - intersection == 0:
            None
            with open('jaccard.txt', 'a') as f:
                f.write(f'{np.count_nonzero(a)}, {np.count_nonzero(b)}, {intersection}  \n')
            return 0
        J = intersection * 1.0 / (np.count_nonzero(a) + np.count_nonzero(b) - intersection)
        return J

    def _cosine_similarity(self, a, b):
        inner_product = (a * b).sum()
        C = inner_product / (np.sqrt(np.square(a).sum()) * np.sqrt(np.square(b).sum()) + 1e-10)
        return C


class GNNGuard(RobustIdentity):

    def __init__(self, use_gate=True, drop=False):
        super(GNNGuard, self).__init__()
        self.drop = drop
        self.use_gate = use_gate
        if self.use_gate:
            self.gate = Parameter(torch.rand(1))

    def forward(self, edge_index, edge_weight, features):
        adj_values = self.att_coef(features, edge_index, i=0)
        if self.use_gate:
            adj_values = self.gate * edge_weight + (1 - self.gate) * adj_values
        adj_values = torch.clamp(adj_values, min=0)
        return adj_values

    def att_coef(self, fea, edge_index, is_lil=False, i=0):
        n_node = fea.shape[0]
        row, col = edge_index[0].cpu().numpy()[:], edge_index[1].cpu().numpy()[:]
        fea_copy = fea.cpu().data.numpy()
        sim_matrix = cosine_similarity(X=fea_copy, Y=fea_copy)
        sim = sim_matrix[row, col]
        sim[sim < 0.1] = 0
        """build a attention matrix"""
        att_dense = lil_matrix((n_node, n_node), dtype=np.float32)
        att_dense[row, col] = sim
        if att_dense[0, 0] == 1:
            att_dense = att_dense - sp.diags(att_dense.diagonal(), offsets=0, format='lil')
        att_dense_norm = normalize(att_dense, axis=1, norm='l1')
        """add learnable dropout, make character vector"""
        if self.drop:
            character = np.vstack((att_dense_norm[row, col].A1, att_dense_norm[col, row].A1))
            character = torch.from_numpy(character.T)
            drop_score = self.drop_learn_1(character)
            drop_score = torch.sigmoid(drop_score)
            mm = torch.nn.Threshold(0.5, 0)
            drop_score = mm(drop_score)
            mm_2 = torch.nn.Threshold(-0.49, 1)
            drop_score = mm_2(-drop_score)
            drop_decision = drop_score.clone().requires_grad_()
            drop_matrix = lil_matrix((n_node, n_node), dtype=np.float32)
            drop_matrix[row, col] = drop_decision.cpu().data.numpy().squeeze(-1)
            att_dense_norm = att_dense_norm.multiply(drop_matrix.tocsr())
        if att_dense_norm[0, 0] == 0:
            degree = (att_dense_norm != 0).sum(1).A1
            lam = 1 / (degree + 1)
            self_weight = sp.diags(np.array(lam), offsets=0, format='lil')
            att = att_dense_norm + self_weight
        else:
            att = att_dense_norm
        att_adj = edge_index
        att_edge_weight = att[row, col]
        att_edge_weight = np.exp(att_edge_weight)
        att_edge_weight = torch.tensor(np.array(att_edge_weight)[0], dtype=torch.float32)
        return att_edge_weight

    def add_loop_sparse(self, adj, fill_value=1):
        row = torch.range(0, int(adj.shape[0] - 1), dtype=torch.int64)
        i = torch.stack((row, row), dim=0)
        v = torch.ones(adj.shape[0], dtype=torch.float32)
        shape = adj.shape
        I_n = torch.sparse.FloatTensor(i, v, shape)
        return adj + I_n


def generate_sparse_ids(_A, _A_robust, X, sparse_rate, metric):
    """
    _A: origin adjacency matrix , sp.csr_matrix
    X: feature matrix. numpy.array
    return: list n
    """
    sparse_ids = []
    _A.setdiag(np.zeros(_A.shape[0]))
    _D = np.count_nonzero(_A.toarray(), axis=1)
    _A_robust.setdiag(np.zeros(_A_robust.shape[0]))
    _D_r = np.count_nonzero(_A_robust.toarray(), axis=1)
    d_mean = np.mean(_D)
    d_std = np.std(_D)
    if metric == 'correlation':
        d_thres = d_mean + 2.5 * d_std
    else:
        d_thres = d_mean + 2 * d_std
    highD, sparseN, nosparseN = 0, 0, 0
    for u in range(X.shape[0]):
        neighbors = _A_robust[u, :].nonzero()[1]
        x_neighbors = X[neighbors]
        x_u = X[u]
        if _D[u] > d_thres:
            sparse_t = np.setdiff1d(np.arange(len(_D)), _A[u, :].nonzero()[1])
            sparse_ids.append(sparse_t)
            highD += 1
        elif round(sparse_rate * _D[u]) > _D_r[u]:
            sparse_ids.append(np.array([]))
            nosparseN += 1
        else:
            if metric == 'correlation':
                dist_2 = np.squeeze(scipy.spatial.distance.cdist(x_u, x_neighbors, 'correlation'))
            else:
                dist_2 = np.sum((x_neighbors - x_u) ** 2, axis=1)
            nz_sel = int(_D_r[u] - round(sparse_rate * _D[u]))
            sparse_ids.append(neighbors[dist_2.argsort()[-nz_sel:]])
            sparseN += 1
    return sparse_ids


class VPN(RobustIdentity):

    def __init__(self, r=2):
        super(VPN, self).__init__()
        self.r = r
        self.theta_1 = Parameter(torch.FloatTensor(1), requires_grad=True)
        self.theta_2 = Parameter(torch.FloatTensor(1), requires_grad=True)
        self.theta_1.data.fill_(1)
        self.theta_2.data.fill_(0)
        if r >= 3:
            self.theta_3 = Parameter(torch.FloatTensor(1), requires_grad=True)
            self.theta_3.data.fill_(0)
        elif r == 4:
            self.theta_4 = Parameter(torch.FloatTensor(1), requires_grad=True)
            self.theta_4.data.fill_(0)
        self.edge_index = None

    def preprocess_adj_alls(self, edge_index, edge_weight, features):
        num_nodes = features.size(0)
        self.device = edge_index.device
        self.use_sparse_adj = True
        i = edge_index.cpu().numpy()
        sp_A = sp.csr_matrix((edge_weight.detach().cpu().numpy(), (i[0], i[1])), shape=(num_nodes, num_nodes))
        sp_A = sp_A + sp.eye(num_nodes)
        sp_A[sp_A > 1] = 1
        adj_alls = [sp_A]
        for k in range(2, self.r + 1):
            adj_k = sp_A ** k
            adj_alls.append(adj_k)
        return adj_alls

    def forward(self, edge_index, edge_weight, features):
        self.adj_alls = self.preprocess_adj_alls(edge_index, edge_weight, features)
        adj_values = self.sparsification(edge_index, edge_weight, self.adj_alls, features.detach().cpu().numpy())
        adj_values = torch.clamp(adj_values, min=0)
        return adj_values

    def sparsification(self, edge_index, edge_weight, adjs, X, sparse_rate=2.0, metric='euclidean'):
        """
        Parameters
        --------
        adjs: list of torch.Tensor dense matrix /scipy sparse
            [A^(k)]
        x: numpy.array

        Returns:
        --------
        modified_adj: torch 
            modified dense adjacency matrix
        """
        _A, _A_robust = adjs[0], adjs[-1]
        sparse_ids = generate_sparse_ids(_A, _A_robust, X, sparse_rate=sparse_rate, metric=metric)
        row, col = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
        n_nodes = adjs[0].shape[0]
        adj_values = self.theta_1 * edge_weight
        if self.r >= 2:
            adj_k = (adjs[1] - adjs[0]).tolil()
            adj_k.setdiag(np.zeros(n_nodes))
            for u in range(n_nodes):
                adj_k[u, sparse_ids[u]] = 0
            adj_values = adj_values + self.theta_2 * torch.tensor(adj_k.toarray()[row, col], dtype=torch.float32)
        if self.r >= 3:
            adj_k = (adjs[2] - adjs[1]).tolil()
            adj_k.setdiag(np.zeros(n_nodes))
            for u in range(n_nodes):
                adj_k[u, sparse_ids[u]] = 0
            adj_values = adj_values + self.theta_3 * torch.tensor(adj_k.toarray()[row, col], dtype=torch.float32)
        return adj_values


ROB_OPS = {'identity': lambda : RobustIdentity(), 'svd': lambda : SVD(), 'jaccard': lambda : Jaccard(), 'gnnguard': lambda : GNNGuard(), 'vpn': lambda : VPN()}


class GRNASpace(BaseSpace):

    def __init__(self, hidden_dim: '_typ.Optional[int]'=64, layer_number: '_typ.Optional[int]'=2, dropout: '_typ.Optional[float]'=0.6, input_dim: '_typ.Optional[int]'=None, output_dim: '_typ.Optional[int]'=None, ops: '_typ.Tuple'=['gcn', 'gat_2'], rob_ops: '_typ.Tuple'=['identity', 'svd', 'jaccard', 'gnnguard'], act_ops: '_typ.Sequence[_typ.Union[str, _typ.Any]]'=GRAPHNAS_DEFAULT_ACT_OPS):
        super().__init__()
        self.layer_number = layer_number
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ops = ops
        self.rob_ops = rob_ops
        self.dropout = dropout
        self.act_ops = act_ops

    def instantiate(self, hidden_dim: '_typ.Optional[int]'=None, layer_number: '_typ.Optional[int]'=None, input_dim: '_typ.Optional[int]'=None, output_dim: '_typ.Optional[int]'=None, ops: '_typ.Tuple'=None, rob_ops: '_typ.Tuple'=None, act_ops: '_typ.Tuple'=None, dropout=None):
        super().instantiate()
        self.hidden_dim = hidden_dim or self.hidden_dim
        self.layer_number = layer_number or self.layer_number
        self.input_dim = input_dim or self.input_dim
        self.output_dim = output_dim or self.output_dim
        self.ops = ops or self.ops
        self.rob_ops = rob_ops or self.rob_ops
        self.act_ops = act_ops or self.act_ops
        self.dropout = dropout or self.dropout
        for layer in range(self.layer_number):
            setattr(self, f'op_{layer}', self.setLayerChoice(layer, [(op(self.input_dim if layer == 0 else self.hidden_dim, self.output_dim if layer == self.layer_number - 1 else self.hidden_dim) if isinstance(op, type) else gnn_map(op, self.input_dim if layer == 0 else self.hidden_dim, self.output_dim if layer == self.layer_number - 1 else self.hidden_dim)) for op in self.ops]))
            setattr(self, f'rob_op_{layer}', self.setLayerChoice(layer, [(op() if isinstance(op, type) else ROB_OPS[op]()) for op in self.rob_ops]))
            setattr(self, 'act', self.setLayerChoice(2 * layer, [act_map_nn(a) for a in self.act_ops], key='act'))
        self._initialized = True

    def forward(self, data):
        x = bk_feat(data)
        edge_weight = data.edge_weight if data.edge_weight is not None else torch.ones(data.edge_index.size(1))
        for layer in range(self.layer_number):
            rob_op = getattr(self, f'rob_op_{layer}')
            edge_weight = rob_op(data.edge_index, edge_weight, x)
            op = getattr(self, f'op_{layer}')
            x = op(x, data.edge_index, edge_weight)
            if layer != self.layer_number - 1:
                act = getattr(self, 'act')
                x = act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

    def parse_model(self, selection, device) ->BaseAutoModel:
        return self.wrap().fix(selection)


class SinglePathNodeClassificationSpace(BaseSpace):

    def __init__(self, hidden_dim: '_typ.Optional[int]'=64, layer_number: '_typ.Optional[int]'=2, dropout: '_typ.Optional[float]'=0.2, input_dim: '_typ.Optional[int]'=None, output_dim: '_typ.Optional[int]'=None, ops: '_typ.Tuple'=['gcn', 'gat_8']):
        super().__init__()
        self.layer_number = layer_number
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ops = ops
        self.dropout = dropout

    def instantiate(self, hidden_dim: '_typ.Optional[int]'=None, layer_number: '_typ.Optional[int]'=None, input_dim: '_typ.Optional[int]'=None, output_dim: '_typ.Optional[int]'=None, ops: '_typ.Tuple'=None, dropout=None):
        super().instantiate()
        self.hidden_dim = hidden_dim or self.hidden_dim
        self.layer_number = layer_number or self.layer_number
        self.input_dim = input_dim or self.input_dim
        self.output_dim = output_dim or self.output_dim
        self.ops = ops or self.ops
        self.dropout = dropout or self.dropout
        for layer in range(self.layer_number):
            setattr(self, f'op_{layer}', self.setLayerChoice(layer, [(op(self.input_dim if layer == 0 else self.hidden_dim, self.output_dim if layer == self.layer_number - 1 else self.hidden_dim) if isinstance(op, type) else gnn_map(op, self.input_dim if layer == 0 else self.hidden_dim, self.output_dim if layer == self.layer_number - 1 else self.hidden_dim)) for op in self.ops]))
        self._initialized = True

    def forward(self, data):
        x = bk_feat(data)
        for layer in range(self.layer_number):
            op = getattr(self, f'op_{layer}')
            x = bk_gconv(op, data, x)
            if layer != self.layer_number - 1:
                x = F.leaky_relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

    def parse_model(self, selection, device) ->BaseAutoModel:
        return self.wrap().fix(selection)


class PathSamplingInputChoice(nn.Module):
    """
    Mixed input. Take a list of tensor as input, select some of them and return the sum.

    Attributes
    ----------
    sampled : int or list of int
        Sampled module indices.
    mask : tensor
        A multi-hot bool 1D-tensor representing the sampled mask.
    """

    def __init__(self, input_choice):
        super(PathSamplingInputChoice, self).__init__()
        self.n_candidates = input_choice.n_candidates
        self.n_chosen = input_choice.n_chosen
        self.sampled = None

    def forward(self, input_tensors):
        if isinstance(self.sampled, list):
            return sum([input_tensors[t] for t in self.sampled])
        else:
            return input_tensors[self.sampled]

    def __len__(self):
        return self.n_candidates

    @property
    def mask(self):
        return _get_mask(self.sampled, len(self))

    def __repr__(self):
        return f'PathSamplingInputChoice(n_candidates={self.n_candidates}, chosen={self.sampled})'


class _DummyModel(torch.nn.Module):

    def __init__(self, encoder: '_typing.Union[BaseEncoderMaintainer, BaseAutoModel]', decoder: '_typing.Optional[BaseDecoderMaintainer]'):
        super().__init__()
        if isinstance(encoder, BaseAutoModel):
            self.encoder = encoder.model
            self.decoder = None
        else:
            self.encoder = encoder.encoder
            self.decoder = None if decoder is None else decoder.decoder

    def __str__(self):
        return 'DummyModel(encoder={}, decoder={})'.format(self.encoder, self.decoder)

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        if self.decoder is None:
            return args[0]
        return self.decoder(*args, **kwargs)

    def forward(self, *args, **kwargs):
        res = self.encode(*args, **kwargs)
        return self.decode(res, *args, **kwargs)


class _DummyLinkModel(torch.nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.automodelflag = False
        if isinstance(encoder, BaseAutoModel):
            self.automodelflag = True
            self.encoder = encoder.model
            self.decoder = None
        else:
            self.encoder = encoder.encoder
            self.decoder = None if decoder is None else decoder.decoder

    def encode(self, data):
        if self.automodelflag:
            return self.encoder.lp_encode(data)
        return self.encoder(data)

    def decode(self, features, data, pos_edges, neg_edges):
        if self.automodelflag:
            return self.encoder.lp_decode(features, pos_edges, neg_edges)
        if self.decoder is None:
            return features
        return self.decoder(features, data, pos_edges, neg_edges)


gnn_list_proteins = ['gcn', 'cheb', 'arma', 'fc', 'skip']


class Arch:

    def __init__(self, lk=None, op=None):
        self.link = lk
        self.ops = op

    def hash_arch(self, use_proteins=False):
        lk = self.link
        op = self.ops
        if use_proteins:
            gnn_g = {name: i for i, name in enumerate(gnn_list_proteins)}
            b = len(gnn_list_proteins) + 1
        else:
            gnn_g = {name: i for i, name in enumerate(gnn_list)}
            b = len(gnn_list) + 1
        if lk == [0, 0, 0, 0]:
            lk_hash = 0
        elif lk == [0, 0, 0, 1]:
            lk_hash = 1
        elif lk == [0, 0, 1, 1]:
            lk_hash = 2
        elif lk == [0, 0, 1, 2]:
            lk_hash = 3
        elif lk == [0, 0, 1, 3]:
            lk_hash = 4
        elif lk == [0, 1, 1, 1]:
            lk_hash = 5
        elif lk == [0, 1, 1, 2]:
            lk_hash = 6
        elif lk == [0, 1, 2, 2]:
            lk_hash = 7
        elif lk == [0, 1, 2, 3]:
            lk_hash = 8
        for i in op:
            lk_hash = lk_hash * b + gnn_g[i]
        return lk_hash

    def regularize(self):
        lk = self.link[:]
        ops = self.ops[:]
        if lk == [0, 0, 0, 2]:
            lk = [0, 0, 0, 1]
            ops = [ops[1], ops[0], ops[2], ops[3]]
        elif lk == [0, 0, 0, 3]:
            lk = [0, 0, 0, 1]
            ops = [ops[2], ops[0], ops[1], ops[3]]
        elif lk == [0, 0, 1, 0]:
            lk = [0, 0, 0, 1]
            ops = [ops[0], ops[1], ops[3], ops[2]]
        elif lk == [0, 0, 2, 0]:
            lk = [0, 0, 0, 1]
            ops = [ops[1], ops[0], ops[3], ops[2]]
        elif lk == [0, 0, 2, 1]:
            lk = [0, 0, 1, 2]
            ops = [ops[1], ops[0], ops[2], ops[3]]
        elif lk == [0, 0, 2, 2]:
            lk = [0, 0, 1, 1]
            ops = [ops[1], ops[0], ops[2], ops[3]]
        elif lk == [0, 0, 2, 3]:
            lk = [0, 0, 1, 3]
            ops = [ops[1], ops[0], ops[2], ops[3]]
        elif lk == [0, 1, 0, 0]:
            lk = [0, 0, 0, 1]
            ops = [ops[0], ops[2], ops[3], ops[1]]
        elif lk == [0, 1, 0, 1]:
            lk = [0, 0, 1, 1]
            ops = [ops[0], ops[2], ops[1], ops[3]]
        elif lk == [0, 1, 0, 2]:
            lk = [0, 0, 1, 3]
            ops = [ops[0], ops[2], ops[1], ops[3]]
        elif lk == [0, 1, 0, 3]:
            lk = [0, 0, 1, 2]
            ops = [ops[0], ops[2], ops[1], ops[3]]
        elif lk == [0, 1, 1, 0]:
            lk = [0, 0, 1, 1]
            ops = [ops[0], ops[3], ops[1], ops[2]]
        elif lk == [0, 1, 1, 3]:
            lk = [0, 1, 1, 2]
            ops = [ops[0], ops[2], ops[1], ops[3]]
        elif lk == [0, 1, 2, 0]:
            lk = [0, 0, 1, 3]
            ops = [ops[0], ops[3], ops[1], ops[2]]
        elif lk == [0, 1, 2, 1]:
            lk = [0, 1, 1, 2]
            ops = [ops[0], ops[1], ops[3], ops[2]]
        return Arch(lk, ops)

    def equalpart_sort(self):
        lk = self.link
        op = self.ops
        ops = op[:]

        def part_sort(ids, ops):
            gnn_g = {name: i for i, name in enumerate(gnn_list)}
            opli = [gnn_g[ops[i]] for i in ids]
            opli.sort()
            for posid, opid in zip(ids, opli):
                ops[posid] = gnn_list[opid]
            return ops

        def sort0012(ops):
            gnn_g = {name: i for i, name in enumerate(gnn_list)}
            if gnn_g[op[0]] > gnn_g[op[1]] or op[0] == op[1] and gnn_g[op[2]] > gnn_g[op[3]]:
                ops = [ops[1], ops[0], ops[3], ops[2]]
            return ops
        if lk == [0, 0, 0, 0]:
            ids = [0, 1, 2, 3]
        elif lk == [0, 0, 0, 1]:
            ids = [1, 2]
        elif lk == [0, 0, 1, 1]:
            ids = [2, 3]
        elif lk == [0, 0, 1, 2]:
            ids = None
            ops = sort0012(ops)
        elif lk == [0, 1, 1, 1]:
            ids = [1, 2, 3]
        elif lk == [0, 1, 2, 2]:
            ids = [2, 3]
        else:
            ids = None
        if ids:
            ops = part_sort(ids, ops)
        self.ops = ops

    def move_skip_op(self):
        link = self.link[:]
        ops = self.ops[:]

        def move_one(k, link, ops):
            ops = [ops[k]] + ops[:k] + ops[k + 1:]
            for i, father in enumerate(link):
                if father == k + 1:
                    link[i] = link[k]
                if father <= k:
                    link[i] = link[i] + 1
            link = [0] + link[:k] + link[k + 1:]
            return link, ops

        def check_dim(k, link, ops):
            while k > -1:
                if ops[k] != 'skip':
                    return False
                k = link[k] - 1
            return True
        for i in range(len(link)):
            if ops[i] != 'skip':
                continue
            son = False
            brother = False
            for j, fa in enumerate(link):
                if fa == i + 1:
                    son = True
                elif j != i and fa == link[i]:
                    brother = True
            if son or not brother or check_dim(i, link, ops) and not son:
                link, ops = move_one(i, link, ops)
        if link == [0, 1, 2, 1]:
            link = [0, 1, 1, 2]
            ops = ops[:2] + [ops[3], ops[2]]
        elif link == [0, 1, 1, 3]:
            link = [0, 1, 1, 2]
            ops = [ops[0], ops[2], ops[1], ops[3]]
        self.link = link
        self.ops = ops

    def valid_hash(self):
        b = self.regularize()
        b.move_skip_op()
        b.equalpart_sort()
        return b.hash_arch()

    def check_isomorph(self):
        link, ops = self.link, self.ops
        linkm = link[:]
        opsm = ops[:]
        self.move_skip_op()
        self.equalpart_sort()
        return linkm == self.link and opsm == self.ops


class BenchSpace(BaseSpace):

    def __init__(self, hidden_dim: '_typ.Optional[int]'=64, layer_number: '_typ.Optional[int]'=2, dropout: '_typ.Optional[float]'=0.9, input_dim: '_typ.Optional[int]'=None, output_dim: '_typ.Optional[int]'=None, ops_type=0):
        super().__init__()
        self.layer_number = layer_number
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.ops_type = ops_type

    def instantiate(self, hidden_dim: '_typ.Optional[int]'=None, layer_number: '_typ.Optional[int]'=None, dropout: '_typ.Optional[float]'=None, input_dim: '_typ.Optional[int]'=None, output_dim: '_typ.Optional[int]'=None, ops_type=None):
        super().instantiate()
        self.dropout = dropout or self.dropout
        self.hidden_dim = hidden_dim or self.hidden_dim
        self.layer_number = layer_number or self.layer_number
        self.input_dim = input_dim or self.input_dim
        self.output_dim = output_dim or self.output_dim
        self.ops_type = ops_type or self.ops_type
        self.ops = [gnn_list, gnn_list_proteins][self.ops_type]
        for layer in range(4):
            setattr(self, f'in{layer}', self.setInputChoice(layer, n_candidates=layer + 1, n_chosen=1, return_mask=False, key=f'in{layer}'))
            setattr(self, f'op{layer}', self.setLayerChoice(layer, list(map(lambda x: StrModule(x), self.ops)), key=f'op{layer}'))
        self.dummy = nn.Linear(1, 1)

    def forward(self, bench):
        lks = [getattr(self, 'in' + str(i)).selected for i in range(4)]
        ops = [getattr(self, 'op' + str(i)).name for i in range(4)]
        arch = Arch(lks, ops)
        h = arch.valid_hash()
        if h == '88888' or h == 88888:
            return 0
        return bench[h]['perf']

    def parse_model(self, selection, device) ->BaseAutoModel:
        return self.wrap().fix(selection)


class TopKPool(torch.nn.Module):

    def __init__(self):
        super(TopKPool, self).__init__()
        self.conv1 = GraphConv(dataset.num_features, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, dataset.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = x1 + x2 + x3
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)
        return x


class DotPredictor(nn.Module):

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return g.edata['score'][:, 0]


class DummyModel(torch.nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data):
        out1 = self.encoder(data)
        return self.decoder(out1, data)


class SAGE(torch.nn.Module):

    def __init__(self, num_features, hidden_channels, num_layers, num_classes):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            inc = outc = hidden_channels
            if i == 0:
                inc = num_features
            if i == num_layers - 1:
                outc = num_classes
            self.convs.append(SAGEConv(inc, outc))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(x, dim=-1)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AggAdd,
     lambda: ([], {'dim': 4, 'att_head': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (FixedInputChoice,
     lambda: ([], {'mask': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (LambdaModule,
     lambda: ([], {'lambd': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Linear,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LinearConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'num_layers': 1, 'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RobustIdentity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SemanticAttention,
     lambda: ([], {'in_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SkipConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (StrModule,
     lambda: ([], {'lambd': 4}),
     lambda: ([], {})),
    (Zero,
     lambda: ([], {'indim': 4, 'outdim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ZeroConv,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (_LogSoftmaxDecoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

